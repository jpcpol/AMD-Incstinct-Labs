// ===========================================================================
// SUPERSEDED by fa_mfma2.hip — that file fixes the bug below (1-wave/block
// design) and PASSES correctness (max_err 0.00004, 87 µs). This file is kept
// as the record of the failed Br-warps approach and the mapping-probe story.
//
// STATUS: WORK IN PROGRESS — correctness FAIL (max_err ~0.42).
//
// ROOT CAUSE (diagnosed 2026-06-05): this kernel uses Br=16 INDEPENDENT warps
// (threadIdx.y), each a separate 64-lane wavefront. But v_mfma_f32_16x16x16f16
// requires ONE wave's 64 lanes to collectively hold all 16 rows of the tile.
// Here each warp only computed its own P row, yet the MFMA A-mapping makes every
// lane read P[L%16] from shared p_lds — so all 16 warps redundantly compute the
// same 16×D output and write it 16× to O_lds (a write race + wrong structure).
//
// FIX REQUIRED (future session): restructure to 1 wave per query-block. The
// softmax, S=Q·Kᵀ, and the thread layout all change. This is a substantial
// rewrite, deferred. Steps 1–3 (fa_naive/dpp/dme) remain the validated path.
//
// VALUABLE ARTIFACT PRESERVED: the v_mfma_f32_16x16x16f16 wave64 lane mapping,
// verified empirically on gfx942 (probe6.hip, full 16×16×16 GEMM vs host ref,
// max_err 0.001). The unique correct combination is:
//   A input : lane L, frag f → A[row = L%16      ][k   = (L/16)*4 + f]
//   B input : lane L, frag f → B[k   = (L/16)*4+f][col = L%16        ]
//   C/D out : lane L, frag f → C[row = (L/16)*4+f][col = L%16        ]
// (i.e. A is read [idx16][quad], B and the accumulator are [quad][idx16].)
// All four mapping-combination tests are in probe6.hip; only this one matched.
// ===========================================================================
//
// fa_mfma.hip — Flash Attention Step 4: MFMA matrix cores on MI300X (gfx942)
//
// Build order:
//   Step 1: fa_naive — portable dot product + __shfl softmax    [97.2 µs]
//   Step 2: fa_dpp   — DPP softmax                              [90.5 µs]
//   Step 3: fa_dme   — DME double-buffer K/V                    [82.4 µs]
//   Step 4: fa_mfma  — MFMA for Q*K^T and P*V                  [THIS FILE]
//
// MFMA instruction: __builtin_amdgcn_mfma_f32_16x16x16f16
//   C(16×16 fp32) += A(16×16 fp16) * B(16×16 fp16)
//   In wave64: each lane holds 4 fp32 accumulators (C fragment)
//              and contributes 4 fp16 of A and 4 fp16 of B per call.
//
// Geometry (Br=16, Bc=64, D=64):
//
//   Q·K^T: Q(16×64) × K^T(64×16) → S(16×64)
//     - One warp = one Q row (Br=16 warps, one per query row)
//     - K tile is (Bc×D) = (64×64), laid out row-major
//     - Tile the D dimension in blocks of 16: 4 MFMA calls per (warp, Bc-block)
//     - Bc=64 = 4 blocks of 16 → total 4×4 = 16 MFMA calls per warp per KV tile
//     - Each MFMA: A = Q[warp, d_block*16 : (d_block+1)*16]  (one row fragment)
//                  B = K[j*16 : (j+1)*16, d_block*16 : (d_block+1)*16] (col fragment)
//                  C accumulates S[warp, j*16 : (j+1)*16]
//
//   P·V: P(16×64) × V(64×64) → O(16×64)
//     - Similar tiling: P rows × V columns, both in 16×16 blocks
//     - 4 blocks of Bc × 4 blocks of D = 16 MFMA calls per warp per KV tile
//
// Register layout for v_mfma_f32_16x16x16f16 (wave64 mode):
//   A: vector of 4 __fp16 — 4 consecutive elements of a 16-wide row slice
//   B: vector of 4 __fp16 — 4 consecutive elements of a 16-wide col slice
//   C/D: vector of 4 float — 4 accumulators for a 16×16 output fragment
//   Each lane contributes/receives elements at positions determined by lane_id.
//
// IMPORTANT: In wave64, the MFMA 16×16×16 instruction is issued per-wave.
//   The 64 lanes of a wave collectively hold the full 16×16 tile:
//   - A rows: 16 groups of 4 lanes each → lane L holds A[L/4][L%4 * 4 .. +4]
//     Wait — the actual mapping is: lane L holds columns (L%4)*4..(L%4)*4+3
//     of row (L/4) of the 16×16 A fragment. See AMD CDNA3 ISA §6.
//   - For simplicity we load A/B as __half2 vectors and let the compiler handle
//     the lane mapping, which matches the MFMA intrinsic's expectations.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../wave-primitives/include/wave_primitives \
//         -I../dme-abstraction/include \
//         fa_mfma.hip -o fa_mfma
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

#include "wave_reduce_dpp.hpp"
#include "dme/dme_copy.hpp"

static constexpr int Br  = 16;
static constexpr int Bc  = 64;
static constexpr int D   = 64;
static constexpr int kN  = 512;

static constexpr int kBatch  = 1;
static constexpr int kHeads  = 8;
static constexpr int kSeqQ   = 512;
static constexpr int kBlocks = kBatch * kHeads * kSeqQ / Br;
static constexpr int kWarmup = 20;
static constexpr int kIter   = 200;

// MFMA tile size
static constexpr int MT = 16;   // 16×16×16 instruction
// Number of MT tiles along D and Bc
static constexpr int nD  = D  / MT;   // 4
static constexpr int nBc = Bc / MT;   // 4

// DME tile floats (fp16 reinterpreted as float, 2 fp16 per float)
static constexpr int kTileFloats = Bc * D / 2;

// ---------------------------------------------------------------------------
// MFMA intrinsic wrapper
// v_mfma_f32_16x16x16f16: D = C + A * B
//   A, B: __attribute__((ext_vector_type(4))) __fp16  (4 fp16 = two __half2)
//   C, D: __attribute__((ext_vector_type(4))) float
// ---------------------------------------------------------------------------
using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;

__device__ __forceinline__
f32x4 mfma16(fp16x4 a, fp16x4 b, f32x4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

// __half and __fp16 are distinct types on ROCm — cast via reinterpret
__device__ __forceinline__ __fp16 to_fp16(__half h) {
    return *reinterpret_cast<__fp16*>(&h);
}
__device__ __forceinline__ __fp16 float_to_fp16(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<__fp16*>(&h);
}

// Load 4 consecutive __half from ptr into fp16x4
__device__ __forceinline__
fp16x4 load4h(const __half* ptr) {
    fp16x4 v;
    v[0] = to_fp16(ptr[0]); v[1] = to_fp16(ptr[1]);
    v[2] = to_fp16(ptr[2]); v[3] = to_fp16(ptr[3]);
    return v;
}

// ---------------------------------------------------------------------------
// Flash Attention MFMA kernel
// ---------------------------------------------------------------------------
__global__ void fa_mfma_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int N)
{
    const int qblk  = blockIdx.x % (kSeqQ / Br);
    const int head  = (blockIdx.x / (kSeqQ / Br)) % kHeads;
    const int batch = blockIdx.x / (kSeqQ / Br) / kHeads;
    const int lane  = threadIdx.x;   // 0..63
    const int warp  = threadIdx.y;   // 0..Br-1 (one wave per query row)

    const __half* Qrow  = Q + ((size_t)batch*kHeads+head)*kSeqQ*D + (qblk*Br+warp)*D;
    const __half* Kbase = K + ((size_t)batch*kHeads+head)*N*D;
    const __half* Vbase = V + ((size_t)batch*kHeads+head)*N*D;
    __half*       Orow  = O + ((size_t)batch*kHeads+head)*kSeqQ*D + (qblk*Br+warp)*D;

    // Double-buffer LDS (same as fa_dme)
    __shared__ __half K_buf[2][Bc * D];
    __shared__ __half V_buf[2][Bc * D];
    __shared__ float  O_lds[Br * D];

    // Zero O_lds
    {
        const int tid = warp*64+lane, total = Br*D;
        for (int i = tid; i < total; i += Br*64) O_lds[i] = 0.f;
    }

    // -----------------------------------------------------------------------
    // S accumulator: S[nBc] = nBc fragments, each fragment is f32x4
    // Lane L holds 4 elements of the S row — the exact elements depend on
    // the MFMA lane mapping. After the MFMA, we extract the full row via shfl.
    // -----------------------------------------------------------------------
    // S_frag[j] accumulates S[warp, j*MT .. (j+1)*MT-1] as a 16×16 fragment.
    // Each lane holds 4 of the 16×16=256 elements; for a single-row result
    // (Br=1 within the fragment), the 4 accumulators per lane represent the
    // 4 columns that this lane owns in the output row.
    //
    // After MFMA, lane L holds S[row_in_frag][col_group] where:
    //   row_in_frag = L / (MT/4) ... but for our Br=16 = MT case each
    //   warp maps to exactly one row of the fragment. The 64 lanes cover
    //   4 output cols each (4 × 16 = 64 = Bc/nBc ✓).
    //   Lane L's 4 accumulators → S[warp_row_in_frag][L*4 mod MT .. +4]
    //   where warp_row_in_frag = warp % MT = warp (since Br=MT=16).
    // -----------------------------------------------------------------------

    float m = -1e38f, l = 0.f;
    const float scale = 1.f / sqrtf((float)D);
    const int kv_tiles = N / Bc;

    // Q row: load into registers (stays fixed for all KV tiles)
    // Q[warp][0..D-1] — each lane loads D/64 = 1 fp16 elements... but we need
    // 4 consecutive elements per MFMA call. Load as fp16x4 arrays.
    // Q is D=64 wide, split into nD=4 blocks of MT=16.
    fp16x4 Qfrag[nD];   // Q fragments for MFMA
    #pragma unroll
    for (int id = 0; id < nD; ++id) {
        // Each lane loads 4 fp16 from its assigned columns.
        // For A matrix (16 rows × 16 cols), lane L owns cols (L%4)*4..(L%4)*4+3
        // of row (L/4) within the 16×16 block.
        // Since warp = query row (0..15) and our Br=MT=16, warp = L/4 row.
        // So each lane loads from Qrow[id*MT + (lane%4)*4 .. +4].
        // BUT: all 64 lanes of the wave participate in one MFMA call.
        // The correct lane-to-element mapping for MFMA A in wave64:
        //   lane L holds A[row=L>>2][col=(L&3)*4 .. (L&3)*4+3]
        // For our warp structure (one warp per query row, warp=threadIdx.y):
        //   we want A[warp][id*MT + col_offset] for this warp.
        //   All warps run simultaneously, each as separate wavefronts.
        //   Within ONE wave, all 64 lanes cover the full 16×16 A tile:
        //     lane L → A row = L >> 2 (0..15), col group = L & 3 (0..3) → 4 cols
        // But we have ONE query row per warp, not 16 rows.
        // Correct approach: broadcast Q[warp][d] to all lanes that need it.
        // The MFMA A tile has 16 rows; since Br=16 and Br warps exist,
        // each warp contributes its Q row to the A tile.
        // In the wave (64 lanes): lanes 0-3 → row 0, lanes 4-7 → row 1, ...
        // warp w contributes Q[w][...] to lanes [4w .. 4w+3].
        // Since warp w IS the wave of lanes 0-63, all in the same CTA,
        // the MFMA is a BLOCK-level operation across all 16 warps.
        // But each warp is a SEPARATE wave of 64 lanes — they don't share VGPRs.
        //
        // Therefore: MFMA within a single wave computes one 16×16 tile of S
        // using ALL 16 query rows simultaneously, not just warp w's row.
        // This requires restructuring: one wave handles all Br rows at once.
        //
        // RESTRUCTURE: use one wave per (qblk) with Br lanes per row group.
        // With Bc=64=wave_size and Br=16, the natural layout is:
        //   - 1 wave, 64 lanes, block=(64) not (64, Br)
        //   - Lane L computes output for query row (L / (64/Br)) and key col (L % Bc)
        // This is a different thread layout. For now, use the simpler approach:
        // compute S via direct dot product (as in fa_dme) but using fp16 accumulation
        // and then call MFMA only for the P·V step where the geometry fits perfectly.
        //
        // SIMPLIFIED STRATEGY for Step 4:
        //   Q·K^T: keep direct fp16 dot product (same as fa_dme, but in fp16)
        //   P·V:   use MFMA 16×16×16 with proper tiling
        // This still demonstrates MFMA usage and should improve P·V performance.
        Qfrag[id] = load4h(Qrow + id*MT + (lane & 3)*4);
        // Note: only lanes where (lane >> 2) == warp use their Q data correctly.
        // Others load garbage but MFMA uses lane mapping, so we load Q[warp][...]:
        // override: all lanes load from Qrow (same warp's Q row).
    }

    // Prefetch tile 0
    {
        const float* Ks = reinterpret_cast<const float*>(Kbase);
        const float* Vs = reinterpret_cast<const float*>(Vbase);
        float* Kd = reinterpret_cast<float*>(K_buf[0]);
        float* Vd = reinterpret_cast<float*>(V_buf[0]);
        const int tid = warp*64+lane, stride = Br*64;
        for (int i = tid; i < kTileFloats; i += stride) {
            dme::copy_element_stream<float>(Ks+i, Kd+i);
            dme::copy_element_stream<float>(Vs+i, Vd+i);
        }
    }
    dme::wait(); __syncthreads();

    for (int kv = 0; kv < kv_tiles; ++kv) {
        const int cur = kv & 1, next = cur ^ 1;

        // Async prefetch next tile
        if (kv + 1 < kv_tiles) {
            const float* Ks = reinterpret_cast<const float*>(Kbase+(size_t)(kv+1)*Bc*D);
            const float* Vs = reinterpret_cast<const float*>(Vbase+(size_t)(kv+1)*Bc*D);
            float* Kd = reinterpret_cast<float*>(K_buf[next]);
            float* Vd = reinterpret_cast<float*>(V_buf[next]);
            const int tid = warp*64+lane, stride = Br*64;
            for (int i = tid; i < kTileFloats; i += stride) {
                dme::copy_element_stream<float>(Ks+i, Kd+i);
                dme::copy_element_stream<float>(Vs+i, Vd+i);
            }
        }

        // ---------------------------------------------------------------
        // Q·K^T via direct dot product (fp32 accum, same as fa_dme)
        // Lane L computes S[warp][L] = dot(Q[warp], K_buf[cur][L])
        // ---------------------------------------------------------------
        float S_lane = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d)
            S_lane += __half2float(Qrow[d]) * __half2float(K_buf[cur][lane*D+d]);
        S_lane *= scale;

        // DPP softmax
        float row_max = wave::dpp::reduce_max_bcast<float>(S_lane);
        float m_new   = fmaxf(m, row_max);
        float p       = expf(S_lane - m_new);
        float row_sum = wave::dpp::reduce_sum_bcast<float>(p);
        float l_new   = expf(m - m_new) * l + row_sum;

        // O rescale
        float rescale = expf(m - m_new);
        { int d = lane; if (d < D) O_lds[warp*D+d] *= rescale; }

        __syncthreads();

        // ---------------------------------------------------------------
        // P·V via MFMA 16×16×16 (Step 4 key change)
        //
        // Layout: P[Br×Bc] = softmax weights, V[Bc×D]
        // We compute O[Br×D] = P × V
        //
        // MFMA tile: 16×16×16
        //   - A = P tile  (16 rows × 16 cols of P, in fp16)
        //   - B = V tile  (16 rows × 16 cols of V, in fp16)
        //   - C = O tile  (16×16 fp32 accumulator)
        //
        // P[Br×Bc]: nBc=4 tiles of 16 cols each (along Bc)
        // V[Bc×D]:  nBc=4 row-tiles × nD=4 col-tiles of 16×16
        // O[Br×D]:  nD=4 tiles of 16 cols each (along D)
        //
        // For each output tile [id] (16 output cols):
        //   O_tile[id] += sum_{ib=0}^{nBc-1} P_tile[ib] * V_tile[ib][id]
        //
        // p (float, lane L = P[warp][L]) needs to be cast to fp16 for MFMA.
        // P is (Br×Bc) = (16×64) — each warp holds one row of 64 fp16 weights.
        // For the MFMA A tile (16×16), lanes cover: row=(lane>>2), col_grp=(lane&3)
        // Since warp w's P row = p[0..63] (all lanes), we need to scatter p into
        // the A tile layout. For tile ib (covering P cols ib*16..(ib+1)*16-1):
        //   A lane L = P[warp_in_frag=lane>>2][ib*16 + (lane&3)*4 .. +4]
        // Since this wave's warp owns row 'warp' of P, we need P[warp][ib*16+col]
        // where col depends on lane. But ALL warps run simultaneously and each
        // contributes its own row. The MFMA A tile covers rows 0..15 = all warps.
        // Lane L carries A[L>>2][...], meaning lane L serves warp (L>>2).
        // So lane L uses P[(L>>2)][ib*16 + (L&3)*4 .. +4] — NOT warp w's row,
        // but the row of the warp indexed by (L>>2).
        //
        // Key insight: we need to READ p from OTHER warps' lanes.
        // Since p is a register in each lane, and lane L is in warp w = threadIdx.y,
        // we need cross-warp communication — which goes through LDS.
        //
        // Alternative: store p to LDS first, then each lane reads the 4 values it
        // needs for the A fragment, then call MFMA.
        // ---------------------------------------------------------------

        // Store p to LDS (reuse O_lds space is unsafe; use K_buf[next] as scratch
        // — it's being filled by DME but we haven't waited yet, so it's being
        // written asynchronously. Use a separate LDS region.)
        // Simpler: store p in the first Br*64 floats of a dedicated shared area.
        // We have O_lds[Br*D] = 16*64*4 = 4KB. Use K_buf[cur] as scratch after
        // S computation (K_buf[cur] is fully consumed, DME is writing K_buf[next]).
        // Cast K_buf[cur] as float scratch: Bc*D/2 = 2048 floats available (8KB).
        // We need Br*64 = 1024 floats for p. Fine.

        float* p_lds = reinterpret_cast<float*>(K_buf[cur]);   // scratch: 2KB of 8KB
        p_lds[warp * 64 + lane] = p;   // store this lane's p value
        __syncthreads();

        // Now compute O via MFMA
        // Each warp computes O[warp, 0..D-1] += sum_ib P_tile[ib] * V_tile[ib][id]
        // We accumulate 4 MFMA output tiles (one per nD output block).
        f32x4 O_frag[nD];
        #pragma unroll
        for (int id = 0; id < nD; ++id) {
            // Load current O_lds into accumulators
            // O_lds[warp*D + id*MT + (lane>>2)*4 .. +4] ... complex mapping.
            // Simpler: initialize to 0 and add to O_lds at the end.
            O_frag[id] = f32x4{0.f, 0.f, 0.f, 0.f};
        }

        #pragma unroll
        for (int ib = 0; ib < nBc; ++ib) {
            // Load P fragment [ib]: P[0..15, ib*16..(ib+1)*16-1] from p_lds
            // Lane L → A[row=L>>2][col=(L&3)*4..(L&3)*4+3] = P[L>>2][ib*16+(L&3)*4..]
            fp16x4 A;
            // Verified A lane mapping (probe3, gfx942):
            //   lane L, frag f → A[row = L%16][k = (L/16)*4 + f]
            // For P·V: A = P, row = query row, k = Bc index.
            int a_row = lane % MT;                       // query row in [0, Br=16)
            int a_k0  = ib*MT + (lane/MT)*4;             // Bc index base for this lane
            A[0] = float_to_fp16(p_lds[a_row*64 + a_k0 + 0]);
            A[1] = float_to_fp16(p_lds[a_row*64 + a_k0 + 1]);
            A[2] = float_to_fp16(p_lds[a_row*64 + a_k0 + 2]);
            A[3] = float_to_fp16(p_lds[a_row*64 + a_k0 + 3]);

            #pragma unroll
            for (int id = 0; id < nD; ++id) {
                // Load V fragment [ib][id]: V[ib*16..(ib+1)*16-1, id*16..(id+1)*16-1]
                // B lane mapping: B[row=lane>>2][col=(lane&3)*4..(lane&3)*4+3]
                //              = V[ib*MT + (lane>>2)][id*MT + (lane&3)*4..]
                fp16x4 B;
                // Verified B mapping (probe6, A=[i][q] B=[q][i] out=[q][i]):
                //   lane L, frag f → B[row=(L/16)*4+f][col=L%16]
                // For P·V: B = V, row = Bc index (V row, the K dim), col = D output dim.
                int v_dcol = id*MT + (lane % MT);            // D output column = L%16
                #pragma unroll
                for (int f = 0; f < 4; ++f) {
                    int v_k = ib*MT + (lane/MT)*4 + f;       // Bc index (V row)
                    B[f] = to_fp16(V_buf[cur][v_k*D + v_dcol]);
                }

                O_frag[id] = mfma16(A, B, O_frag[id]);
            }
        }

        // Write O_frag back to O_lds
        // Verified output layout (probe6, gfx942):
        //   Lane L, frag f → row = (L/16)*4 + f,  col = L%16
        // For O = P·V: row = query row [0,Br), col = D output dimension.
        #pragma unroll
        for (int id = 0; id < nD; ++id) {
            int out_col = id*MT + (lane % MT);            // D output column = L%16
            #pragma unroll
            for (int f = 0; f < 4; ++f) {
                int out_row = (lane/MT)*4 + f;            // query row in [0, Br=16)
                O_lds[out_row*D + out_col] += O_frag[id][f];
            }
        }

        if (kv + 1 < kv_tiles) { dme::wait(); __syncthreads(); }
        else                    { __syncthreads(); }

        m = m_new;
        l = l_new;
    }

    // Normalize and write output — each lane writes one element (d = lane)
    { int d = lane; if (d < D) Orow[d] = __float2half(O_lds[warp*D+d] / l); }
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------
void attn_cpu_ref(const std::vector<float>& Q, const std::vector<float>& K,
                  const std::vector<float>& V, std::vector<float>& O,
                  int seq_q, int seq_k, int d)
{
    const float scale = 1.f / sqrtf((float)d);
    for (int q = 0; q < seq_q; ++q) {
        std::vector<float> S(seq_k);
        float mx = -1e38f;
        for (int k = 0; k < seq_k; ++k) {
            float dot = 0.f;
            for (int dd = 0; dd < d; ++dd) dot += Q[q*d+dd] * K[k*d+dd];
            S[k] = dot * scale; mx = std::max(mx, S[k]);
        }
        float se = 0.f;
        for (int k = 0; k < seq_k; ++k) { S[k] = expf(S[k]-mx); se += S[k]; }
        for (int k = 0; k < seq_k; ++k) S[k] /= se;
        for (int dd = 0; dd < d; ++dd) {
            float out = 0.f;
            for (int k = 0; k < seq_k; ++k) out += S[k] * V[k*d+dd];
            O[q*d+dd] = out;
        }
    }
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device : %s  (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("Config : Br=%d Bc=%d D=%d seqQ=%d seqK=%d heads=%d\n",
           Br, Bc, D, kSeqQ, kN, kHeads);
    printf("P*V    : MFMA 16×16×16 fp16 (4×4 tiles per KV iter per warp)\n\n");

    const size_t QKV_e = (size_t)kBatch*kHeads*kSeqQ*D;
    const size_t KV_e  = (size_t)kBatch*kHeads*kN  *D;

    std::vector<float> hQ_f(QKV_e), hK_f(KV_e), hV_f(KV_e);
    std::mt19937 rng(42); std::normal_distribution<float> dist(0.f,0.5f);
    for (auto& v : hQ_f) v = dist(rng);
    for (auto& v : hK_f) v = dist(rng);
    for (auto& v : hV_f) v = dist(rng);

    std::vector<__half> hQ(QKV_e), hK(KV_e), hV(KV_e);
    for (size_t i=0;i<QKV_e;++i) hQ[i]=__float2half(hQ_f[i]);
    for (size_t i=0;i<KV_e; ++i) hK[i]=__float2half(hK_f[i]);
    for (size_t i=0;i<KV_e; ++i) hV[i]=__float2half(hV_f[i]);

    __half *d_Q,*d_K,*d_V,*d_O;
    hipMalloc(&d_Q,QKV_e*sizeof(__half)); hipMalloc(&d_K,KV_e*sizeof(__half));
    hipMalloc(&d_V,KV_e*sizeof(__half));  hipMalloc(&d_O,QKV_e*sizeof(__half));
    hipMemcpy(d_Q,hQ.data(),QKV_e*sizeof(__half),hipMemcpyHostToDevice);
    hipMemcpy(d_K,hK.data(),KV_e *sizeof(__half),hipMemcpyHostToDevice);
    hipMemcpy(d_V,hV.data(),KV_e *sizeof(__half),hipMemcpyHostToDevice);

    dim3 grid(kBlocks), block(64, Br);

    hipLaunchKernelGGL(fa_mfma_kernel, grid, block, 0, 0, d_Q,d_K,d_V,d_O,kN);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) { printf("KERNEL ERROR: %s\n", hipGetErrorString(e)); return 1; }

    std::vector<__half> hO_gpu(QKV_e);
    hipMemcpy(hO_gpu.data(),d_O,QKV_e*sizeof(__half),hipMemcpyDeviceToHost);

    std::vector<float> hO_ref(kSeqQ*D);
    attn_cpu_ref(hQ_f,hK_f,hV_f,hO_ref,kSeqQ,kN,D);

    float max_err = 0.f;
    for (int i=0; i<kSeqQ*D; ++i)
        max_err = std::max(max_err, fabsf(__half2float(hO_gpu[i]) - hO_ref[i]));
    printf("--- Correctness (first head) ---\n");
    printf("  max |gpu - ref| = %.5f  [%s]\n\n",
           max_err, max_err < 0.1f ? "PASS" : "FAIL");
    if (max_err >= 0.1f) { printf("Abort benchmark.\n"); return 1; }

    // Benchmark
    hipEvent_t t0,t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i=0;i<kWarmup;++i)
        hipLaunchKernelGGL(fa_mfma_kernel,grid,block,0,0,d_Q,d_K,d_V,d_O,kN);
    hipDeviceSynchronize();
    std::vector<float> ts(kIter);
    for (int i=0;i<kIter;++i) {
        hipEventRecord(t0);
        hipLaunchKernelGGL(fa_mfma_kernel,grid,block,0,0,d_Q,d_K,d_V,d_O,kN);
        hipEventRecord(t1); hipEventSynchronize(t1);
        hipEventElapsedTime(&ts[i],t0,t1); ts[i]*=1000.f;
    }
    std::sort(ts.begin(),ts.end());
    float med=ts[kIter/2], p99=ts[(int)(kIter*0.99f)];

    double total_flops = 2.0*2.0*Br*Bc*D*(kN/Bc)*(double)kBlocks;
    double hbm = (double)kBlocks*(kN/Bc)*(Bc*D+Bc*D)*sizeof(__half)
               + (double)kBlocks*kSeqQ*D*sizeof(__half)*2;

    printf("=== Flash Attention MFMA — benchmark (Step 4) ===\n");
    printf("  Blocks: %d | Tiles/block: %d | FLOPs: %.2f GFLOP\n",
           kBlocks, kN/Bc, total_flops/1e9);
    printf("  Median: %.1f µs | P99: %.1f µs\n", med, p99);
    printf("  Achieved: %.2f TFLOPS | %.1f GB/s HBM\n",
           total_flops/med/1e6, hbm/med/1e3);
    printf("\n  fa_dme (Step 3): 82.4 µs\n");
    printf("  fa_mfma (Step 4): %.1f µs\n", med);
    printf("  Speedup vs dme: %.3fx\n", 82.4f/med);

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);
    return 0;
}

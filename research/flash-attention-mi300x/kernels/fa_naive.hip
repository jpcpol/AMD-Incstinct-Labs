// ---------------------------------------------------------------------------
// fa_naive.hip — Flash Attention (step 1): FA2 algorithm, portable reductions,
//                synchronous K/V loads. No DME, no MFMA intrinsics.
//
// PURPOSE: Establish a correct, measurable baseline on MI300X before adding
//   DPP reductions (step 2), DME prefetch (step 3), and MFMA tuning (step 4).
//
// ALGORITHM: FlashAttention-2 online softmax with double-pass output rescaling.
//
//   For each query block (Br rows, each of dim d):
//     for j in range(Tc):             // iterate over K/V blocks
//       S = Q · Kj^T                  // Br×Bc dot products (manual)
//       m_new = max(m, rowmax(S))     // online row-max (wave reduction)
//       P = exp(S - m_new)            // Br×Bc (element-wise)
//       l = exp(m-m_new)*l + rowsum(P)// update row-sum (wave reduction)
//       O = exp(m-m_new)*O + P·Vj    // accumulate output (manual)
//       m = m_new
//   O = O / l                         // final normalize
//
// THREAD LAYOUT:
//   grid  = (seq_len/Br, n_heads, batch)
//   block = (64, 1, 1)   // one wave per query row (Br=1 for this naive version)
//
//   With Br=1 each wave handles ONE query row. This is the simplest
//   correct layout — no cross-row coordination within a block.
//   Parallelism: all query rows, heads, batches are independent blocks.
//
// TILE SIZES:
//   Bc = 64   (key/value block — fits in one wave64 reduction exactly)
//   Br = 1    (query block — one wave per row, step 2/4 will explore Br>1)
//   d  = 64   (head dimension — compile-time, standard for many models)
//
// MEMORY:
//   Q: [batch, n_heads, seq_len, d]    row-major
//   K: [batch, n_heads, seq_len, d]    row-major
//   V: [batch, n_heads, seq_len, d]    row-major
//   O: [batch, n_heads, seq_len, d]    row-major
//   All fp32. Step 2 will add fp16/bf16 variants.
//
// SCALING: Q·K^T divided by sqrt(d) per attention paper.
//
// CORRECTNESS TEST: CPU reference (naive O(N²) attention) must match
//   within 1e-3 relative error.
//
// Build:
//   hipcc -O3 --offload-arch=gfx942 -I../../wave-primitives/include \
//         -DFA_WAVESIZE=64 fa_naive.hip -o fa_naive
// ---------------------------------------------------------------------------

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>
#include <limits>

// Include shared bench infrastructure
#include "../../wave-primitives/benchmarks/bench_common.hpp"

// ---------------------------------------------------------------------------
// Configuration (override at compile time with -D flags)
// ---------------------------------------------------------------------------

#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 64
#endif

#ifndef FA_BC
#define FA_BC 64   // K/V tile: must be <= 64 for single-wave reduction
#endif

static constexpr int kHeadDim = FA_HEAD_DIM;
static constexpr int kBc      = FA_BC;
static constexpr int kBr      = 1;    // one query row per wave (naive version)

static_assert(kBc <= 64,       "Bc > 64 requires cross-wave reduction — use step 2+");
static_assert(kHeadDim % 64 == 0, "head dim must be multiple of 64 for lane tiling");
static constexpr int kDChunks = kHeadDim / 64;  // how many 64-elem d-chunks per lane pass

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Portable wave-64 reduction (will be swapped for DPP in step 2).
// Result on lane 0; all lanes receive via final __shfl broadcast.
__device__ __forceinline__ float wave_max_bcast(float v) {
    for (int o = 32; o > 0; o >>= 1) {
        float x = __shfl_down(v, o, 64);
        v = x > v ? x : v;
    }
    return __shfl(v, 0, 64);
}

__device__ __forceinline__ float wave_sum_bcast(float v) {
    for (int o = 32; o > 0; o >>= 1) v += __shfl_down(v, o, 64);
    return __shfl(v, 0, 64);
}

// ---------------------------------------------------------------------------
// Kernel: fa_naive_kernel
// ---------------------------------------------------------------------------
// One wave (64 threads) handles one query row.
// grid  = (Tr, n_heads, batch)   where Tr = seq_len / Br
// block = (64, 1, 1)
//
// Each lane is responsible for elements at positions:
//   within the d-dimension: lane, lane+64, lane+128, ... (kDChunks elements)
//   within the Bc-dimension during S computation: lane index within the tile
//
// For Bc=64 and d=64: lane i computes S[0, i] = dot(Q[row, :], K[j*Bc+i, :])
//   This means each lane computes the full dot product for ONE key position.
//
// ---------------------------------------------------------------------------

__global__ void fa_naive_kernel(
    const float* __restrict__ Q,   // [batch, heads, seq_len, d]
    const float* __restrict__ K,   // [batch, heads, seq_len, d]
    const float* __restrict__ V,   // [batch, heads, seq_len, d]
    float*       __restrict__ O,   // [batch, heads, seq_len, d]
    int seq_len,
    int n_heads,
    float scale)                   // 1/sqrt(d)
{
    // Block indices
    const int head_batch_id = blockIdx.y;               // flattened head+batch
    const int batch_id = head_batch_id / n_heads;
    const int head_id  = head_batch_id % n_heads;
    const int query_row = blockIdx.x;                    // which query row
    const int lane = threadIdx.x;                        // 0..63

    // Stride helpers
    const long long head_stride = (long long)seq_len * kHeadDim;
    const long long batch_stride = (long long)n_heads * head_stride;
    const long long base = (long long)batch_id * batch_stride + (long long)head_id * head_stride;

    // Pointer to this query row in Q
    const float* q_row = Q + base + (long long)query_row * kHeadDim;

    // Load Q row: each lane loads its d-chunks into registers
    // With kHeadDim=64 and 64 lanes: lane i holds Q[lane i]
    float q_reg[kDChunks];
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c) {
        int d_idx = c * 64 + lane;
        q_reg[c] = q_row[d_idx];
    }

    // Running online softmax state
    float m = -3.402823466e+38f;  // -FLT_MAX
    float l = 0.0f;
    float o_acc[kDChunks];
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c) o_acc[c] = 0.0f;

    const int Tc = (seq_len + kBc - 1) / kBc;   // number of K/V tiles

    for (int j = 0; j < Tc; ++j) {
        int kv_start = j * kBc;

        // --- Compute S = Q_row · K_j^T  (Br=1 × Bc dot products) ------------
        // Each lane computes the dot product for key position (kv_start + lane).
        // For causal masking we'd need to mask here; skipped for naive baseline.
        int k_pos = kv_start + lane;

        float s_lane = 0.0f;   // S[0, lane] = dot(q_row, K[k_pos, :])
        if (k_pos < seq_len) {
            const float* k_row = K + base + (long long)k_pos * kHeadDim;
            #pragma unroll
            for (int c = 0; c < kDChunks; ++c) {
                int d_idx = c * 64 + lane;
                // Each lane multiplies its own d-chunk of q against k
                // but we need the full dot, so we must communicate across lanes.
                // With kDChunks=1 (d=64) each lane only holds ONE element of q and k,
                // so s_lane is a partial product; sum across lanes gives the full dot.
                s_lane += q_reg[c] * k_row[d_idx];
            }
        } else {
            s_lane = -3.402823466e+38f;  // pad out-of-bounds with -inf
        }

        // s_lane is each lane's contribution to the dot product of
        // Q_row with K[kv_start + lane, :].
        // BUT: for d=64, each lane holds exactly 1 element of the d-dim,
        // so s_lane = q[lane] * k[kv_start+lane, lane] — this is one term
        // of ONE dot product, not the complete dot product for key position `lane`.
        //
        // We need: S[lane] = sum_d( Q[lane_d] * K[kv_start+lane, lane_d] )
        // The issue: lane `i` holds q[i] and k[kv_start+i, i] in the trivial
        // mapping, but we need lane `i` to compute the full inner product
        // sum over all d dimensions for key index `i`.
        //
        // With d=64=wavesize, the natural mapping is:
        //   Thread (lane) i is responsible for KEY position i in the tile.
        //   It must fetch ALL d elements of K[kv_start+i] and all d elements of Q.
        //
        // We already have q_reg[0] = Q[lane] (element lane of Q, NOT all of Q).
        // So the above accumulation is WRONG for d=64 with the per-lane q_reg layout.
        //
        // Correct approach: use __shfl to share q_reg across lanes, or use
        // a different d-tiling. For d <= wavesize (64), the cleanest mapping is:
        //   Each thread i computes the FULL dot product for key i by
        //   iterating over d-chunks and using __shfl_sync to get Q[d_idx].
        //
        // We recompute below with shfl-based broadcast of Q.

        // Reset and recompute S correctly using shfl to broadcast Q elements
        s_lane = 0.0f;
        if (k_pos < seq_len) {
            const float* k_row = K + base + (long long)k_pos * kHeadDim;
            // For d=64: iterate d; for each d-index, broadcast Q[d] from the
            // lane that holds it (lane d % 64) and load K[k_pos, d] directly.
            for (int d = 0; d < kHeadDim; ++d) {
                // Lane that holds Q[d] in q_reg is: lane = d % 64, chunk = d / 64
                int src_lane  = d % 64;
                int src_chunk = d / 64;
                float q_d = __shfl(q_reg[src_chunk], src_lane, 64);
                float k_d = k_row[d];   // every lane loads its own k_row[d]
                s_lane += q_d * k_d;
            }
        } else {
            s_lane = -3.402823466e+38f;
        }
        s_lane *= scale;

        // --- Online softmax update -------------------------------------------
        float m_new = wave_max_bcast(s_lane);   // rowmax(S) across all 64 lanes
        // m_new: max of all S[0..Bc-1], broadcast to all lanes

        float p_lane = expf(s_lane - m_new);    // P[lane] = exp(S[lane] - m_new)
        if (k_pos >= seq_len) p_lane = 0.0f;    // zero out-of-bounds

        float l_new_partial = wave_sum_bcast(p_lane);  // rowsum(P)

        float alpha = expf(m - m_new);           // rescale factor for old accumulator
        m = m_new;
        l = alpha * l + l_new_partial;

        // --- Accumulate O += alpha*O_old + P · Vj ----------------------------
        // Each lane i has p_lane = P[i]. We need:
        //   O_acc[d] += alpha * O_acc[d] + sum_k(P[k] * V[kv_start+k, d])
        //
        // Lane i holds output element d=i. We need sum over k of P[k]*V[k,i].
        // P[k] is on lane k. V[kv_start+k, i] must be loaded.
        // Strategy: iterate over k (Bc=64 iterations), shfl P[k] from lane k.
        #pragma unroll
        for (int c = 0; c < kDChunks; ++c) {
            o_acc[c] *= alpha;
        }

        for (int k = 0; k < kBc; ++k) {
            int v_pos = kv_start + k;
            if (v_pos >= seq_len) break;

            // P[k] is on lane k — broadcast it to all lanes
            float p_k = __shfl(p_lane, k % 64, 64);
            // When Bc=64, k < 64, so k%64 == k exactly.

            const float* v_row = V + base + (long long)v_pos * kHeadDim;
            #pragma unroll
            for (int c = 0; c < kDChunks; ++c) {
                int d_idx = c * 64 + lane;
                o_acc[c] += p_k * v_row[d_idx];
            }
        }
    }

    // --- Final normalize: O = O / l ------------------------------------------
    float* o_row = O + base + (long long)query_row * kHeadDim;
    float l_inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c) {
        int d_idx = c * 64 + lane;
        o_row[d_idx] = o_acc[c] * l_inv;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: naive O(N^2) attention, single head, single batch
// ---------------------------------------------------------------------------

void cpu_attention_ref(
    const float* Q,   // [seq, d]
    const float* K,   // [seq, d]
    const float* V,   // [seq, d]
    float*       O,   // [seq, d]
    int seq_len, int d, float scale)
{
    std::vector<float> scores(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        float max_s = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
            float s = 0.0f;
            for (int dd = 0; dd < d; ++dd)
                s += Q[i*d + dd] * K[j*d + dd];
            scores[j] = s * scale;
            max_s = std::max(max_s, scores[j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = expf(scores[j] - max_s);
            sum_exp += scores[j];
        }
        for (int dd = 0; dd < d; ++dd) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; ++j)
                acc += scores[j] * V[j*d + dd];
            O[i*d + dd] = acc / sum_exp;
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    auto info = get_device_info(0);
    print_device_info(info);

    // Problem sizes: seq_len must be multiple of kBc=64
    static constexpr int kSeqLen  = 512;
    static constexpr int kNHeads  = 8;
    static constexpr int kBatch   = 2;
    static constexpr int kD       = kHeadDim;
    static constexpr float kScale = 1.0f / sqrtf((float)kD);

    printf("Config: batch=%d, heads=%d, seq=%d, d=%d, Bc=%d\n",
           kBatch, kNHeads, kSeqLen, kD, kBc);
    printf("Scale:  %.6f\n\n", kScale);

    const long long N = (long long)kBatch * kNHeads * kSeqLen * kD;

    // Host buffers
    std::vector<float> h_Q(N), h_K(N), h_V(N);
    std::vector<float> h_O_gpu(N, 0.0f);
    std::vector<float> h_O_cpu((long long)kSeqLen * kD, 0.0f);

    // Random init
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    for (auto& x : h_Q) x = dist(rng);
    for (auto& x : h_K) x = dist(rng);
    for (auto& x : h_V) x = dist(rng);

    // Device buffers
    float *d_Q, *d_K, *d_V, *d_O;
    HIP_CHECK(hipMalloc(&d_Q, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_K, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_V, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_O, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_O, 0, N * sizeof(float)));

    // Launch config
    dim3 grid(kSeqLen / kBr, kBatch * kNHeads, 1);
    dim3 block(64, 1, 1);

    printf("Grid:  (%d, %d, 1)\n", grid.x, grid.y);
    printf("Block: (64, 1, 1)\n\n");

    // Correctness check — compare GPU vs CPU on batch=0, head=0
    hipLaunchKernelGGL(fa_naive_kernel, grid, block, 0, 0,
                       d_Q, d_K, d_V, d_O,
                       kSeqLen, kNHeads, kScale);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_O_gpu.data(), d_O, N * sizeof(float), hipMemcpyDeviceToHost));

    printf("Running CPU reference (seq=%d, d=%d) ...\n", kSeqLen, kD);
    cpu_attention_ref(h_Q.data(), h_K.data(), h_V.data(),
                      h_O_cpu.data(), kSeqLen, kD, kScale);

    // Compare first head/batch
    double max_err = 0.0, sum_sq = 0.0;
    long long n_elem = (long long)kSeqLen * kD;
    for (long long i = 0; i < n_elem; ++i) {
        float ref = h_O_cpu[i];
        float got = h_O_gpu[i];
        double err = fabs((double)(got - ref)) / (fabs((double)ref) + 1e-6);
        if (err > max_err) max_err = err;
        sum_sq += (double)(got - ref) * (double)(got - ref);
    }
    double rmse = sqrt(sum_sq / n_elem);
    printf("Correctness (batch=0, head=0): max_rel_err=%.6f  RMSE=%.6f\n",
           max_err, rmse);
    if (max_err < 1e-3)
        printf("  PASS (max_rel_err < 1e-3)\n\n");
    else
        printf("  FAIL — check kernel\n\n");

    // Benchmark
    static constexpr int kWarmup = 20;
    static constexpr int kIter   = 200;

    // FLOP count: seq_len * (Tc * (2*d*Bc + 3*Bc + 2*d*Bc)) ≈ 4 * seq^2 * d
    double flops = 4.0 * kSeqLen * kSeqLen * kD * kBatch * kNHeads;

    printf("Benchmarking (warmup=%d, iter=%d) ...\n", kWarmup, kIter);
    auto res = time_kernel([&]() {
        hipLaunchKernelGGL(fa_naive_kernel, grid, block, 0, 0,
                           d_Q, d_K, d_V, d_O,
                           kSeqLen, kNHeads, kScale);
    }, 0.0, kWarmup, kIter);

    double tflops = flops / (res.median_us * 1e-6) / 1e12;
    double bw_tb_s = (3.0 * N * sizeof(float) / (res.median_us * 1e-6)) / 1e12;

    printf("\n=== fa_naive (seq=%d, d=%d, Bc=%d, %d heads × %d batch) ===\n",
           kSeqLen, kD, kBc, kNHeads, kBatch);
    printf("  Median  : %10.3f µs\n", res.median_us);
    printf("  P99     : %10.3f µs\n", res.p99_us);
    printf("  TFLOPS  : %10.3f  (vs MI300X peak ~%.0f TFLOPS fp32)\n", tflops, 163.4);
    printf("  BW (R)  : %10.3f TB/s (vs MI300X peak 5.3 TB/s)\n", bw_tb_s);
    printf("\n");
    printf("NOTE: naive kernel is bandwidth-limited (no MFMA, no tiling optimization).\n");
    printf("  The TFLOPS/BW numbers establish the baseline that step 2+ must beat.\n");

    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O));
    return 0;
}

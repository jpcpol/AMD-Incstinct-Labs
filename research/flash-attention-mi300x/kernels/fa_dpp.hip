// ---------------------------------------------------------------------------
// fa_dpp.hip — Flash Attention (step 2): DPP reductions for online softmax.
//
// DIFF FROM fa_naive: the only algorithmic change is replacing the two wave
// reductions (rowmax, rowsum) with DPP variants. Everything else — Q loading,
// dot product via shfl, P·V accumulation, online rescaling — is identical.
//
// WHY THIS MATTERS FOR FA (not just microbenchmarks):
//   In the naive kernel, wave_max_bcast / wave_sum_bcast lower to ds_bpermute
//   (LDS roundtrip, 6 steps × 2 per iteration × Tc iterations).
//   Those LDS accesses compete with K/V tile storage for the 64 KB LDS budget.
//
//   DPP reductions use ZERO LDS. This has two effects:
//     1. Direct: fewer LDS instructions → lower latency per softmax step.
//     2. Indirect (the key claim): freed LDS budget → larger Bc → fewer
//        K/V tiles to iterate over → better MFMA/BW utilization.
//
// TWO-PHASE REDUCTION FOR Bc > 64:
//   When Bc > 64, one wave cannot cover the full row. We use a block of
//   (64 * kWavesPerBlock) threads, kWavesPerBlock waves per query row.
//   Phase 1: DPP reduce within each wave       → partial result on lane 63
//   Phase 2: LDS for inter-wave reduce          → block-level result
//   Phase 3: readlane(0) broadcast to all lanes
//
//   With kWavesPerBlock=1 (Bc=64): phase 2 is a no-op. Zero LDS.
//   With kWavesPerBlock=2 (Bc=128): phase 2 uses 2 LDS slots per reduction.
//   With kWavesPerBlock=4 (Bc=256): phase 2 uses 4 LDS slots per reduction.
//   Compare to naive (Bc=64): 6 LDS round-trips per reduction, Tc tiles.
//   The DPP path's LDS use is O(kWavesPerBlock) not O(Tc * 12).
//
// TILE SIZES (compile-time, override with -D):
//   FA_BC           = 64 / 128 / 256   (K/V tile; Bc = 64 * kWavesPerBlock)
//   FA_HEAD_DIM     = 64               (head dimension)
//   FA_WAVES_PER_BLOCK = 1 / 2 / 4    (derived: Bc / 64)
//
// Build (single-wave baseline — apples-to-apples with fa_naive Bc=64):
//   hipcc -O3 --offload-arch=gfx942 -I../../wave-primitives/include \
//         -DFA_BC=64 fa_dpp.hip -o fa_dpp_bc64
//
// Build (wider tile — the LDS-relaxation hypothesis):
//   hipcc -O3 --offload-arch=gfx942 -I../../wave-primitives/include \
//         -DFA_BC=128 fa_dpp.hip -o fa_dpp_bc128
//   hipcc -O3 --offload-arch=gfx942 -I../../wave-primitives/include \
//         -DFA_BC=256 fa_dpp.hip -o fa_dpp_bc256
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

#include "../../wave-primitives/include/wave_primitives/wave_reduce_dpp.hpp"
#include "../../wave-primitives/benchmarks/bench_common.hpp"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 64
#endif

#ifndef FA_BC
#define FA_BC 64
#endif

static constexpr int kHeadDim       = FA_HEAD_DIM;
static constexpr int kBc            = FA_BC;
static constexpr int kBr            = 1;
static constexpr int kWavesPerBlock = kBc / 64;   // waves handling one query row
static constexpr int kBlockSize     = kWavesPerBlock * 64;

static_assert(kBc % 64 == 0,         "Bc must be a multiple of 64 (wave width)");
static_assert(kWavesPerBlock <= 8,    "kWavesPerBlock > 8 is unreasonably large");
static_assert(kHeadDim % 64 == 0,    "head dim must be a multiple of 64");
static constexpr int kDChunks = kHeadDim / 64;

// ---------------------------------------------------------------------------
// Device: two-phase DPP reduction for kWavesPerBlock waves
//
// Phase 1: DPP reduce within each 64-lane wave  → partial on lane 63 of wave
// Phase 2: gather partials into LDS, reduce across waves
// Phase 3: broadcast result to every lane in the block
//
// LDS layout: one float per wave for each of the two reductions (max, sum).
//   lds_scratch[0..kWavesPerBlock-1] : max partials
//   lds_scratch[kWavesPerBlock..2k-1]: sum partials
// Total: 2 * kWavesPerBlock * 4 bytes (8 bytes for kWavesPerBlock=1, trivial).
// ---------------------------------------------------------------------------

__device__ __forceinline__
void block_reduce_max_sum(
    float  s,                   // each lane's S value (after scaling)
    float& out_max,             // broadcast row-max
    float& out_sum_p,           // broadcast rowsum(exp(s - out_max))
    float* __restrict__ lds)    // shared scratch [2 * kWavesPerBlock floats]
{
    const int lane     = threadIdx.x & 63;             // lane within wave
    const int wave_id  = threadIdx.x >> 6;             // which wave in block

    // Phase 1: DPP reduce within wave (result on lane 63 of each wave)
    float wave_max = wave::dpp::reduce_max(s);         // lane 63 of this wave
    float wave_exp_s;   // placeholder — computed after we know the global max

    // We need the global max before computing exp, so we do max first.

    // Phase 2a: gather wave-max partials into LDS, reduce across waves
    if (kWavesPerBlock == 1) {
        // Fast path: single wave, no LDS needed. Broadcast lane 63 → all.
        out_max = wave::dpp::bcast_lane63(wave_max);
    } else {
        if (lane == 63) lds[wave_id] = wave_max;
        __syncthreads();

        // Lane 0 of wave 0 reduces across kWavesPerBlock partials
        float block_max = -3.402823466e+38f;
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int w = 0; w < kWavesPerBlock; ++w)
                block_max = block_max > lds[w] ? block_max : lds[w];
            lds[0] = block_max;   // overwrite with final result
        }
        __syncthreads();
        out_max = lds[0];         // all threads read the same value
    }

    // Now compute exp(s - out_max) and reduce for rowsum
    wave_exp_s = expf(s - out_max);

    // Phase 1b: DPP reduce-sum within wave
    float wave_sum = wave::dpp::reduce_sum(wave_exp_s);

    // Phase 2b: gather wave-sum partials
    if (kWavesPerBlock == 1) {
        out_sum_p = wave::dpp::bcast_lane63(wave_sum);
    } else {
        if (lane == 63) lds[kWavesPerBlock + wave_id] = wave_sum;
        __syncthreads();

        float block_sum = 0.0f;
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int w = 0; w < kWavesPerBlock; ++w)
                block_sum += lds[kWavesPerBlock + w];
            lds[kWavesPerBlock] = block_sum;
        }
        __syncthreads();
        out_sum_p = lds[kWavesPerBlock];
    }
}

// ---------------------------------------------------------------------------
// Kernel: fa_dpp_kernel
//
// grid  = (seq_len / kBr,  batch * n_heads,  1)
// block = (kBlockSize, 1, 1)   = (64 * kWavesPerBlock, 1, 1)
//
// Each block handles one query row. Multiple waves share the row
// when kWavesPerBlock > 1 (i.e., Bc > 64).
// ---------------------------------------------------------------------------

__global__ void fa_dpp_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int seq_len,
    int n_heads,
    float scale)
{
    // 2 * kWavesPerBlock floats for the two-phase reduction scratch
    __shared__ float lds_scratch[2 * kWavesPerBlock < 2 ? 2 : 2 * kWavesPerBlock];

    const int head_batch_id = blockIdx.y;
    const int batch_id = head_batch_id / n_heads;
    const int head_id  = head_batch_id % n_heads;
    const int query_row = blockIdx.x;
    const int tid  = threadIdx.x;           // 0..kBlockSize-1
    const int lane = tid & 63;              // lane within wave

    const long long head_stride  = (long long)seq_len * kHeadDim;
    const long long batch_stride = (long long)n_heads * head_stride;
    const long long base = (long long)batch_id * batch_stride
                         + (long long)head_id  * head_stride;

    // Load Q row into registers (each lane holds kDChunks elements)
    const float* q_row = Q + base + (long long)query_row * kHeadDim;
    float q_reg[kDChunks];
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c)
        q_reg[c] = q_row[c * 64 + lane];

    // Online softmax state (per query row, shared across waves in the block
    // via the block-level reduction in block_reduce_max_sum)
    float m = -3.402823466e+38f;
    float l = 0.0f;
    float o_acc[kDChunks];
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c) o_acc[c] = 0.0f;

    const int Tc = (seq_len + kBc - 1) / kBc;

    for (int j = 0; j < Tc; ++j) {
        const int kv_start = j * kBc;

        // --- Dot product S[tid] = Q_row · K[kv_start + tid, :] ---------------
        // tid ranges 0..kBc-1 across the full block.
        const int k_pos = kv_start + tid;
        float s_val;

        if (k_pos < seq_len) {
            const float* k_row = K + base + (long long)k_pos * kHeadDim;
            s_val = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                // Broadcast Q[d] from the lane that holds it within wave 0.
                // All waves have the same Q loaded, so shfl within-wave suffices.
                int src_lane  = d & 63;
                int src_chunk = d >> 6;
                float q_d = __shfl(q_reg[src_chunk], src_lane, 64);
                s_val += q_d * k_row[d];
            }
        } else {
            s_val = -3.402823466e+38f;
        }
        s_val *= scale;

        // --- DPP two-phase reduction: rowmax + rowsum(exp) -------------------
        float m_new, sum_p;
        block_reduce_max_sum(s_val, m_new, sum_p, lds_scratch);

        // m_new >= max(all S in this tile), broadcast to all threads
        // sum_p  = sum(exp(S - m_new)) for all valid positions

        float p_val = expf(s_val - m_new);
        if (k_pos >= seq_len) p_val = 0.0f;

        // Online update
        float alpha = expf(m - m_new);
        m = m_new;
        l = alpha * l + sum_p;

        #pragma unroll
        for (int c = 0; c < kDChunks; ++c)
            o_acc[c] *= alpha;

        // --- P · V accumulation ----------------------------------------------
        // Each tid holds p_val = P[tid]. lane i in the block holds output
        // element d=lane (mod 64) for chunk c.
        // Strategy: iterate k over kBc positions, shfl p_val from thread k.
        //
        // With kWavesPerBlock > 1: tid = wave_id*64 + lane.
        // Thread `k` is in wave (k/64), lane (k%64).
        // __shfl within a 64-wide wave can only reach threads in the same wave.
        // For cross-wave broadcast we need LDS or __shfl with wave_id check.
        //
        // Efficient approach for kWavesPerBlock > 1:
        //   Store p_val for each tid into LDS (kBc floats), then all threads
        //   read from LDS for the accumulation. This is one LDS write + kBc
        //   LDS reads — much cheaper than Tc * 12 round-trips in naive.
        //
        // For kWavesPerBlock == 1 (Bc=64): pure shfl, zero LDS.

        if constexpr (kWavesPerBlock == 1) {
            // Single wave: shfl within wave (tid == lane, k < 64)
            for (int k = 0; k < kBc; ++k) {
                int v_pos = kv_start + k;
                if (v_pos >= seq_len) break;
                float p_k = __shfl(p_val, k, 64);
                const float* v_row = V + base + (long long)v_pos * kHeadDim;
                #pragma unroll
                for (int c = 0; c < kDChunks; ++c)
                    o_acc[c] += p_k * v_row[c * 64 + lane];
            }
        } else {
            // Multi-wave: stage p_val into LDS, then read back
            // Reuse lds_scratch (it's done after block_reduce_max_sum's __syncthreads).
            // We need kBc floats — declare a separate shared array.
            // Since kBc can be 128/256 and lds_scratch is only 2*kWavesPerBlock,
            // we use a statically-sized shared array declared here via template trick.
            // Workaround: use the dynamic-shared approach with a static max size.
            // Max kBc we support = 256 → 256 floats = 1 KB LDS for P staging.
            __shared__ float p_shared[kBc <= 256 ? kBc : 256];

            p_shared[tid] = p_val;
            __syncthreads();

            for (int k = 0; k < kBc; ++k) {
                int v_pos = kv_start + k;
                if (v_pos >= seq_len) break;
                float p_k = p_shared[k];
                const float* v_row = V + base + (long long)v_pos * kHeadDim;
                #pragma unroll
                for (int c = 0; c < kDChunks; ++c)
                    o_acc[c] += p_k * v_row[c * 64 + lane];
            }
            __syncthreads();   // protect p_shared before next iteration overwrites
        }
    }

    // Final normalize
    float* o_row = O + base + (long long)query_row * kHeadDim;
    float l_inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c)
        o_row[c * 64 + lane] = o_acc[c] * l_inv;
}

// ---------------------------------------------------------------------------
// fa_naive_kernel (inline copy for head-to-head comparison in same binary)
// Identical to fa_naive.hip — avoids a second compilation unit dependency.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float naive_wave_max_bcast(float v) {
    for (int o = 32; o > 0; o >>= 1) { float x = __shfl_down(v, o, 64); v = x > v ? x : v; }
    return __shfl(v, 0, 64);
}
__device__ __forceinline__ float naive_wave_sum_bcast(float v) {
    for (int o = 32; o > 0; o >>= 1) v += __shfl_down(v, o, 64);
    return __shfl(v, 0, 64);
}

// Only Bc=64 variant for the naive baseline (single-wave, apples-to-apples)
__global__ void fa_naive_bc64_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int seq_len, int n_heads, float scale)
{
    static constexpr int kNaiveBc = 64;
    const int head_batch_id = blockIdx.y;
    const int batch_id = head_batch_id / n_heads;
    const int head_id  = head_batch_id % n_heads;
    const int query_row = blockIdx.x;
    const int lane = threadIdx.x;
    const long long base = (long long)batch_id * n_heads * seq_len * kHeadDim
                         + (long long)head_id  * seq_len * kHeadDim;
    const float* q_row = Q + base + (long long)query_row * kHeadDim;
    float q_reg[kDChunks];
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c) q_reg[c] = q_row[c * 64 + lane];
    float m = -3.402823466e+38f, l = 0.0f;
    float o_acc[kDChunks]; for (int c = 0; c < kDChunks; ++c) o_acc[c] = 0.0f;
    for (int j = 0; j < (seq_len + kNaiveBc - 1) / kNaiveBc; ++j) {
        int kv_start = j * kNaiveBc;
        int k_pos = kv_start + lane;
        float s = (k_pos < seq_len) ? 0.0f : -3.402823466e+38f;
        if (k_pos < seq_len) {
            const float* k_row = K + base + (long long)k_pos * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d) {
                float q_d = __shfl(q_reg[d >> 6], d & 63, 64);
                s += q_d * k_row[d];
            }
        }
        s *= scale;
        float m_new = naive_wave_max_bcast(s);
        float p = (k_pos < seq_len) ? expf(s - m_new) : 0.0f;
        float lp = naive_wave_sum_bcast(p);
        float alpha = expf(m - m_new);
        m = m_new; l = alpha * l + lp;
        #pragma unroll
        for (int c = 0; c < kDChunks; ++c) o_acc[c] *= alpha;
        for (int k = 0; k < kNaiveBc && (kv_start + k) < seq_len; ++k) {
            float p_k = __shfl(p, k, 64);
            const float* v_row = V + base + (long long)(kv_start + k) * kHeadDim;
            #pragma unroll
            for (int c = 0; c < kDChunks; ++c) o_acc[c] += p_k * v_row[c * 64 + lane];
        }
    }
    float* o_row = O + base + (long long)query_row * kHeadDim;
    float l_inv = l > 0.0f ? 1.0f / l : 0.0f;
    #pragma unroll
    for (int c = 0; c < kDChunks; ++c) o_row[c * 64 + lane] = o_acc[c] * l_inv;
}

// ---------------------------------------------------------------------------
// CPU reference (same as fa_naive.hip)
// ---------------------------------------------------------------------------

void cpu_attention_ref(const float* Q, const float* K, const float* V,
                       float* O, int seq_len, int d, float scale)
{
    std::vector<float> scores(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        float max_s = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
            float s = 0.0f;
            for (int dd = 0; dd < d; ++dd) s += Q[i*d+dd] * K[j*d+dd];
            scores[j] = s * scale;
            max_s = std::max(max_s, scores[j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) { scores[j] = expf(scores[j] - max_s); sum_exp += scores[j]; }
        for (int dd = 0; dd < d; ++dd) {
            float acc = 0.0f;
            for (int j = 0; j < seq_len; ++j) acc += scores[j] * V[j*d+dd];
            O[i*d+dd] = acc / sum_exp;
        }
    }
}

// ---------------------------------------------------------------------------
// main — correctness + benchmark at Bc=FA_BC, plus naive Bc=64 for comparison
// ---------------------------------------------------------------------------

int main() {
    auto info = get_device_info(0);
    print_device_info(info);

    static constexpr int kSeqLen  = 512;
    static constexpr int kNHeads  = 8;
    static constexpr int kBatch   = 2;
    static constexpr int kD       = kHeadDim;
    static constexpr float kScale = 1.0f / sqrtf((float)kD);

    printf("Config: batch=%d, heads=%d, seq=%d, d=%d\n", kBatch, kNHeads, kSeqLen, kD);
    printf("DPP Bc=%d (%d waves/block)\n", kBc, kWavesPerBlock);
    printf("Scale: %.6f\n\n", kScale);

    const long long N = (long long)kBatch * kNHeads * kSeqLen * kD;
    std::vector<float> h_Q(N), h_K(N), h_V(N), h_O_dpp(N), h_O_naive(N), h_O_cpu(kSeqLen * kD);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    for (auto& x : h_Q) x = dist(rng);
    for (auto& x : h_K) x = dist(rng);
    for (auto& x : h_V) x = dist(rng);

    float *d_Q, *d_K, *d_V, *d_O;
    HIP_CHECK(hipMalloc(&d_Q, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_K, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_V, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_O, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // --- Correctness: DPP kernel vs CPU ---
    dim3 grid_dpp(kSeqLen / kBr, kBatch * kNHeads, 1);
    dim3 block_dpp(kBlockSize, 1, 1);
    HIP_CHECK(hipMemset(d_O, 0, N * sizeof(float)));
    hipLaunchKernelGGL(fa_dpp_kernel, grid_dpp, block_dpp, 0, 0,
                       d_Q, d_K, d_V, d_O, kSeqLen, kNHeads, kScale);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_O_dpp.data(), d_O, N * sizeof(float), hipMemcpyDeviceToHost));

    printf("Running CPU reference ...\n");
    cpu_attention_ref(h_Q.data(), h_K.data(), h_V.data(),
                      h_O_cpu.data(), kSeqLen, kD, kScale);

    double max_err_dpp = 0.0, sum_sq_dpp = 0.0;
    for (long long i = 0; i < (long long)kSeqLen * kD; ++i) {
        float ref = h_O_cpu[i], got = h_O_dpp[i];
        double err = fabs((double)(got - ref)) / (fabs((double)ref) + 1e-6);
        if (err > max_err_dpp) max_err_dpp = err;
        sum_sq_dpp += (double)(got - ref) * (double)(got - ref);
    }
    printf("fa_dpp Bc=%d: max_rel_err=%.6f  RMSE=%.6f  %s\n\n",
           kBc, max_err_dpp, sqrt(sum_sq_dpp / (kSeqLen * kD)),
           max_err_dpp < 1e-3 ? "PASS" : "FAIL");

    // --- Benchmark ---
    static constexpr int kWarmup = 20, kIter = 200;
    double flops = 4.0 * kSeqLen * kSeqLen * kD * kBatch * kNHeads;

    // DPP kernel
    auto res_dpp = time_kernel([&]() {
        hipLaunchKernelGGL(fa_dpp_kernel, grid_dpp, block_dpp, 0, 0,
                           d_Q, d_K, d_V, d_O, kSeqLen, kNHeads, kScale);
    }, 0.0, kWarmup, kIter);

    // Naive Bc=64 baseline (always single-wave, for apples-to-apples with dpp Bc=64)
    dim3 grid_n(kSeqLen, kBatch * kNHeads, 1);
    dim3 block_n(64, 1, 1);
    auto res_naive = time_kernel([&]() {
        hipLaunchKernelGGL(fa_naive_bc64_kernel, grid_n, block_n, 0, 0,
                           d_Q, d_K, d_V, d_O, kSeqLen, kNHeads, kScale);
    }, 0.0, kWarmup, kIter);

    double tflops_dpp   = flops / (res_dpp.median_us   * 1e-6) / 1e12;
    double tflops_naive = flops / (res_naive.median_us * 1e-6) / 1e12;

    printf("=== Benchmark (seq=%d, d=%d, batch=%d×%d heads) ===\n",
           kSeqLen, kD, kBatch, kNHeads);
    printf("%-30s  %10s  %10s  %10s\n", "Kernel", "Median(µs)", "P99(µs)", "TFLOPS");
    printf("%-30s  %10.3f  %10.3f  %10.3f\n",
           "fa_naive (Bc=64, shfl_down)",
           res_naive.median_us, res_naive.p99_us, tflops_naive);
    printf("%-30s  %10.3f  %10.3f  %10.3f\n",
           (kBc == 64  ? "fa_dpp  (Bc=64,  DPP)" :
            kBc == 128 ? "fa_dpp  (Bc=128, DPP)" :
                         "fa_dpp  (Bc=256, DPP)"),
           res_dpp.median_us, res_dpp.p99_us, tflops_dpp);
    printf("\nSpeedup (dpp Bc=%d vs naive Bc=64): %.3fx\n",
           kBc, res_naive.median_us / res_dpp.median_us);
    printf("\n");
    printf("Interpretation:\n");
    printf("  Bc=64  speedup > 1: DPP is faster even at the same tile size\n");
    printf("  Bc=128 speedup vs Bc=64 naive: LDS-relaxation hypothesis confirmed\n");
    printf("  Bc=256 speedup vs Bc=64 naive: diminishing returns or cache pressure\n");
    printf("\n");
    printf("rocprofv3 SQ_INSTS_LDS comparison (run run_fa_dpp.sh for counters):\n");
    printf("  naive Bc=64: O(Tc * 12) LDS instructions from ds_bpermute\n");
    printf("  dpp   Bc=64: 0 LDS from reductions (only p_shared if Bc>64)\n");

    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O));
    return 0;
}

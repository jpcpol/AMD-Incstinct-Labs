// ---------------------------------------------------------------------------
// fa_dpp.hip — Flash Attention Step 2: DPP softmax reductions on MI300X
//
// Build order (design.md):
//   Step 1: fa_naive.hip  — portable __shfl_down softmax   [DONE, 97.2 µs]
//   Step 2: fa_dpp.hip    — wave::dpp::reduce_max/sum_bcast [THIS FILE]
//   Step 3: + DME async double-buffer for K/V tiles
//   Step 4: + MFMA for Q*K^T and P*V
//
// Key change vs fa_naive:
//   wave_max_f32(__shfl_down) → wave::dpp::reduce_max_bcast<float>
//   wave_sum_f32(__shfl_down) → wave::dpp::reduce_sum_bcast<float>
//
// Both *_bcast variants broadcast the result to all 64 lanes (result lands on
// lane 63 after the DPP cascade; bcast_lane63 uses __builtin_amdgcn_readlane).
//
// Hypothesis: freeing the LDS pipeline from ds_bpermute (used by __shfl_down)
// may reduce LDS bank conflicts during the simultaneous K_tile/V_tile loads
// by other warps, or marginally reduce per-warp latency for the softmax step.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../wave-primitives/include/wave_primitives \
//         fa_dpp.hip -o fa_dpp
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

// DPP reductions — the only change vs fa_naive
#include "wave_reduce_dpp.hpp"

static constexpr int Br = 16;    // query block rows
static constexpr int Bc = 64;    // key/value block cols (= wave width)
static constexpr int D  = 64;    // head dimension
static constexpr int kN = 512;   // sequence length

static constexpr int kBatch   = 1;
static constexpr int kHeads   = 8;
static constexpr int kSeqQ    = 512;
static constexpr int kBlocks  = (kBatch * kHeads * kSeqQ / Br);
static constexpr int kWarmup  = 20;
static constexpr int kIter    = 200;

// ---------------------------------------------------------------------------
// Flash Attention kernel — DPP softmax variant
// ---------------------------------------------------------------------------
__global__ void fa_dpp_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int N)
{
    const int qblk  = blockIdx.x % (kSeqQ / Br);
    const int head  = (blockIdx.x / (kSeqQ / Br)) % kHeads;
    const int batch = blockIdx.x / (kSeqQ / Br) / kHeads;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const __half* Qrow  = Q + ((size_t)batch * kHeads + head) * kSeqQ * D
                             + (qblk * Br + warp) * D;
    const __half* Kbase = K + ((size_t)batch * kHeads + head) * N * D;
    const __half* Vbase = V + ((size_t)batch * kHeads + head) * N * D;
    __half*       Orow  = O + ((size_t)batch * kHeads + head) * kSeqQ * D
                             + (qblk * Br + warp) * D;

    __shared__ __half K_tile[Bc * D];
    __shared__ __half V_tile[Bc * D];
    __shared__ float  O_lds[Br * D];

    // Zero O_lds
    {
        const int total  = Br * D;
        const int tid    = warp * 64 + lane;
        for (int i = tid; i < total; i += Br * 64)
            O_lds[i] = 0.f;
    }
    __syncthreads();

    float m = -1e38f;
    float l = 0.f;
    const float scale = 1.f / sqrtf((float)D);

    const int kv_tiles = N / Bc;
    for (int kv = 0; kv < kv_tiles; ++kv) {
        const __half* Kcol = Kbase + (size_t)kv * Bc * D;
        const __half* Vcol = Vbase + (size_t)kv * Bc * D;

        // Load K tile cooperatively
        {
            const int total  = Bc * D;
            const int tid    = warp * 64 + lane;
            const int stride = Br * 64;
            for (int i = tid; i < total; i += stride)
                K_tile[i] = Kcol[i];
        }
        __syncthreads();

        // S[warp][lane] = dot(Q[warp], K_tile[lane]) * scale
        float S_lane = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            float q = __half2float(Qrow[d]);
            float k = __half2float(K_tile[lane * D + d]);
            S_lane += q * k;
        }
        S_lane *= scale;

        // ---------------------------------------------------------------
        // Softmax: DPP reductions (Step 2 change — zero ds_bpermute)
        // *_bcast variants: result broadcast to all 64 lanes via readlane.
        // ---------------------------------------------------------------
        float row_max = wave::dpp::reduce_max_bcast<float>(S_lane);
        float m_new   = fmaxf(m, row_max);
        float p       = expf(S_lane - m_new);   // P[warp][lane] unnormalized
        float row_sum = wave::dpp::reduce_sum_bcast<float>(p);
        float l_new   = expf(m - m_new) * l + row_sum;

        // O rescale
        float rescale = expf(m - m_new);
        {
            int d = lane;
            if (d < D)
                O_lds[warp * D + d] *= rescale;
        }

        // Load V tile cooperatively
        __syncthreads();
        {
            const int total  = Bc * D;
            const int tid    = warp * 64 + lane;
            const int stride = Br * 64;
            for (int i = tid; i < total; i += stride)
                V_tile[i] = Vcol[i];
        }
        __syncthreads();

        // O[warp][d] += sum_k p_k * V_tile[k][d]  (d = lane)
        {
            int d = lane;
            float o_d = 0.f;
            for (int k = 0; k < Bc; ++k) {
                float p_k  = __shfl(p, k, 64);
                float v_kd = __half2float(V_tile[k * D + d]);
                o_d += p_k * v_kd;
            }
            O_lds[warp * D + d] += o_d;
        }
        __syncthreads();

        m = m_new;
        l = l_new;
    }

    // Normalize and write output
    {
        int d = lane;
        if (d < D)
            Orow[d] = __float2half(O_lds[warp * D + d] / l);
    }
}

// ---------------------------------------------------------------------------
// CPU reference (same as fa_naive)
// ---------------------------------------------------------------------------
void attn_cpu_ref(const std::vector<float>& Q,
                  const std::vector<float>& K,
                  const std::vector<float>& V,
                  std::vector<float>& O,
                  int seq_q, int seq_k, int d)
{
    const float scale = 1.f / sqrtf((float)d);
    for (int q = 0; q < seq_q; ++q) {
        std::vector<float> S(seq_k);
        float max_s = -1e38f;
        for (int k = 0; k < seq_k; ++k) {
            float dot = 0.f;
            for (int dd = 0; dd < d; ++dd)
                dot += Q[q*d+dd] * K[k*d+dd];
            S[k] = dot * scale;
            max_s = std::max(max_s, S[k]);
        }
        float sum_exp = 0.f;
        for (int k = 0; k < seq_k; ++k) { S[k] = expf(S[k] - max_s); sum_exp += S[k]; }
        for (int k = 0; k < seq_k; ++k) S[k] /= sum_exp;
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
    printf("Device : %s  (gfx%d)\n", prop.name, prop.major * 100 + prop.minor);
    printf("Config : Br=%d Bc=%d d=%d seqQ=%d seqK=%d heads=%d batch=%d\n\n",
           Br, Bc, D, kSeqQ, kN, kHeads, kBatch);
    printf("Softmax: wave::dpp::reduce_max_bcast + reduce_sum_bcast (Step 2)\n\n");

    const size_t QKV_elems = (size_t)kBatch * kHeads * kSeqQ * D;
    const size_t KV_elems  = (size_t)kBatch * kHeads * kN   * D;

    std::vector<float> hQ_f(QKV_elems), hK_f(KV_elems), hV_f(KV_elems);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 0.5f);
    for (auto& v : hQ_f) v = dist(rng);
    for (auto& v : hK_f) v = dist(rng);
    for (auto& v : hV_f) v = dist(rng);

    std::vector<__half> hQ(QKV_elems), hK(KV_elems), hV(KV_elems);
    for (size_t i = 0; i < QKV_elems; ++i) hQ[i] = __float2half(hQ_f[i]);
    for (size_t i = 0; i < KV_elems;  ++i) hK[i] = __float2half(hK_f[i]);
    for (size_t i = 0; i < KV_elems;  ++i) hV[i] = __float2half(hV_f[i]);

    __half *d_Q, *d_K, *d_V, *d_O;
    hipMalloc(&d_Q, QKV_elems * sizeof(__half));
    hipMalloc(&d_K, KV_elems  * sizeof(__half));
    hipMalloc(&d_V, KV_elems  * sizeof(__half));
    hipMalloc(&d_O, QKV_elems * sizeof(__half));
    hipMemcpy(d_Q, hQ.data(), QKV_elems * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_K, hK.data(), KV_elems  * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_V, hV.data(), KV_elems  * sizeof(__half), hipMemcpyHostToDevice);

    dim3 grid(kBlocks);
    dim3 block(64, Br);

    // Correctness
    hipLaunchKernelGGL(fa_dpp_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("KERNEL ERROR: %s\n", hipGetErrorString(e));
        return 1;
    }

    std::vector<__half> hO_gpu(QKV_elems);
    hipMemcpy(hO_gpu.data(), d_O, QKV_elems * sizeof(__half), hipMemcpyDeviceToHost);

    std::vector<float> hO_ref(kSeqQ * D);
    attn_cpu_ref(hQ_f, hK_f, hV_f, hO_ref, kSeqQ, kN, D);

    float max_err = 0.f;
    for (int i = 0; i < kSeqQ * D; ++i) {
        float gpu_val = __half2float(hO_gpu[i]);
        float err = fabsf(gpu_val - hO_ref[i]);
        max_err = std::max(max_err, err);
    }
    printf("--- Correctness (first head) ---\n");
    printf("  max |gpu - ref| = %.5f  [%s]\n\n",
           max_err, max_err < 0.1f ? "PASS" : "FAIL (check kernel)");

    if (max_err >= 0.1f) {
        printf("Aborting benchmark due to correctness failure.\n");
        return 1;
    }

    // Benchmark
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i = 0; i < kWarmup; ++i)
        hipLaunchKernelGGL(fa_dpp_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN);
    hipDeviceSynchronize();

    std::vector<float> times(kIter);
    for (int i = 0; i < kIter; ++i) {
        hipEventRecord(t0);
        hipLaunchKernelGGL(fa_dpp_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        hipEventElapsedTime(&times[i], t0, t1);
        times[i] *= 1000.f;  // ms → µs
    }
    std::sort(times.begin(), times.end());
    float med_us = times[kIter/2];
    float p99_us = times[(int)(kIter*0.99f)];

    double flops_per_block = 2.0 * 2.0 * Br * Bc * D * (kN / Bc);
    double total_flops = flops_per_block * kBlocks;
    double hbm_bytes = (double)kBlocks * (kN / Bc) * (Bc * D + Bc * D) * sizeof(__half)
                     + (double)kBlocks * kSeqQ * D * sizeof(__half) * 2;

    printf("=== Flash Attention DPP — benchmark (Step 2) ===\n");
    printf("  Blocks: %d  |  Tiles/block: %d  |  FLOPs: %.2f GFLOP\n",
           kBlocks, kN/Bc, total_flops/1e9);
    printf("  Median: %.1f µs  |  P99: %.1f µs\n", med_us, p99_us);
    printf("  Achieved: %.2f TFLOPS  |  %.1f GB/s HBM\n",
           total_flops / med_us / 1e6,
           hbm_bytes   / med_us / 1e3);
    printf("\n  Baseline (fa_naive): 97.2 µs\n");
    printf("  Speedup DPP vs naive: %.3fx\n", 97.2f / med_us);

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);
    hipEventDestroy(t0); hipEventDestroy(t1);
    return 0;
}

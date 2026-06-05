// ---------------------------------------------------------------------------
// fa_dme.hip — Flash Attention Step 3: DME double-buffer K/V prefetch on MI300X
//
// Build order (design.md):
//   Step 1: fa_naive.hip — portable __shfl_down softmax            [DONE, 97.2 µs]
//   Step 2: fa_dpp.hip   — DPP softmax reductions                  [DONE, 90.5 µs]
//   Step 3: fa_dme.hip   — DME async double-buffer K/V tiles       [THIS FILE]
//   Step 4: + MFMA tuning
//
// DME pipeline pattern (per KV iteration j):
//
//   Prefetch phase (async, no stall):
//     dme::copy_tile_1d_stream(K_{j+1}, K_buf[next])
//     dme::copy_tile_1d_stream(V_{j+1}, V_buf[next])
//
//   Compute phase (on current tile j):
//     dme::wait(); __syncthreads();         <- K_{j} now ready in K_buf[cur]
//     S[warp][lane] = dot(Q[warp], K_buf[cur][lane])
//     DPP softmax: row_max, row_sum
//     dme::wait(); __syncthreads();         <- V_{j} now ready in V_buf[cur]
//     O[warp][d] += sum_k p_k * V_buf[cur][k][d]
//
// NOTE: K and V of the SAME tile are prefetched together — this keeps the
// loads contiguous and avoids stalling between K and V of tile j+1.
// The wait for K fires before the S computation (so K is ready), and the
// wait for V fires before the O accumulation (so V is ready).
//
// LDS budget (Bc=64, D=64):
//   2 × K_buf: 2 × 8192 B = 16384 B
//   2 × V_buf: 2 × 8192 B = 16384 B
//   O_lds:          4096 B
//   Total:         36864 B  (~36 KB of 64 KB limit)
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../wave-primitives/include/wave_primitives \
//         -I../dme-abstraction/include \
//         fa_dme.hip -o fa_dme
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

static constexpr int Br = 16;
static constexpr int Bc = 64;
static constexpr int D  = 64;
static constexpr int kN = 512;

static constexpr int kBatch   = 1;
static constexpr int kHeads   = 8;
static constexpr int kSeqQ    = 512;
static constexpr int kBlocks  = (kBatch * kHeads * kSeqQ / Br);
static constexpr int kWarmup  = 20;
static constexpr int kIter    = 200;

// Tile size in elements (float32-equivalent for DME — but actual elements are fp16)
// DME copies 4 bytes per call; for __half (2 bytes) we reinterpret as float* and
// copy 2 fp16 per DME slot, OR we use size=2.
// Simpler: cast __half* to float* and copy Bc*D/2 floats (each float = 2 fp16).
// BUT: copy_tile_1d<float, N> uses size=4 per element, so N = Bc*D/2 for fp16 tiles.
// Bc*D = 64*64 = 4096 fp16 = 2048 floats.
static constexpr int kTileFloats = Bc * D / 2;   // 2048 — treat fp16 tile as float32

// ---------------------------------------------------------------------------
// Flash Attention kernel — DPP softmax + DME double-buffer K/V
// ---------------------------------------------------------------------------
__global__ void fa_dme_kernel(
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

    // Double-buffer LDS: two K buffers + two V buffers + O accumulator
    // Layout: ping/pong indexed by (cur & 1)
    __shared__ __half K_buf[2][Bc * D];   // 2 × 8192 B
    __shared__ __half V_buf[2][Bc * D];   // 2 × 8192 B
    __shared__ float  O_lds[Br * D];      // 4096 B   — total ~36 KB

    // Zero O_lds
    {
        const int total  = Br * D;
        const int tid    = warp * 64 + lane;
        for (int i = tid; i < total; i += Br * 64)
            O_lds[i] = 0.f;
    }

    float m = -1e38f;
    float l = 0.f;
    const float scale = 1.f / sqrtf((float)D);
    const int kv_tiles = N / Bc;

    // ---------------------------------------------------------------------------
    // Prefetch tile 0 (K and V) into buffer 0 before the loop starts.
    // All warps participate — same cooperative pattern as the synchronous load.
    // tid covers Bc*D = 4096 fp16 elements using Br*64 = 1024 threads.
    // Each thread does kTileFloats/512 = 4 DME copies (treating 2 fp16 = 1 float).
    // ---------------------------------------------------------------------------
    {
        const float* Ksrc0 = reinterpret_cast<const float*>(Kbase);
        const float* Vsrc0 = reinterpret_cast<const float*>(Vbase);
        float* Kdst0 = reinterpret_cast<float*>(K_buf[0]);
        float* Vdst0 = reinterpret_cast<float*>(V_buf[0]);

        const int tid    = warp * 64 + lane;
        const int stride = Br * 64;   // 1024 threads total
        for (int i = tid; i < kTileFloats; i += stride) {
            dme::copy_element_stream<float>(Ksrc0 + i, Kdst0 + i);
            dme::copy_element_stream<float>(Vsrc0 + i, Vdst0 + i);
        }
    }
    // Wait for tile 0 to land before the loop — no compute to overlap yet
    dme::wait();
    __syncthreads();

    // ---------------------------------------------------------------------------
    // Main loop
    // ---------------------------------------------------------------------------
    for (int kv = 0; kv < kv_tiles; ++kv) {
        const int cur  = kv & 1;
        const int next = cur ^ 1;

        // ---- Prefetch tile (kv+1) into buf[next] — async, no stall ----
        if (kv + 1 < kv_tiles) {
            const float* Ksrc = reinterpret_cast<const float*>(Kbase + (size_t)(kv + 1) * Bc * D);
            const float* Vsrc = reinterpret_cast<const float*>(Vbase + (size_t)(kv + 1) * Bc * D);
            float* Kdst = reinterpret_cast<float*>(K_buf[next]);
            float* Vdst = reinterpret_cast<float*>(V_buf[next]);

            const int tid    = warp * 64 + lane;
            const int stride = Br * 64;
            for (int i = tid; i < kTileFloats; i += stride) {
                dme::copy_element_stream<float>(Ksrc + i, Kdst + i);
                dme::copy_element_stream<float>(Vsrc + i, Vdst + i);
            }
            // No wait here — DME runs while we compute on cur tile
        }

        // ---- Compute S = Q · K_buf[cur]^T ----
        // K_buf[cur] was prefetched and waited on (either in prologue or end of prev iter)
        float S_lane = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            float q = __half2float(Qrow[d]);
            float k = __half2float(K_buf[cur][lane * D + d]);
            S_lane += q * k;
        }
        S_lane *= scale;

        // ---- DPP softmax (Step 2) ----
        float row_max = wave::dpp::reduce_max_bcast<float>(S_lane);
        float m_new   = fmaxf(m, row_max);
        float p       = expf(S_lane - m_new);
        float row_sum = wave::dpp::reduce_sum_bcast<float>(p);
        float l_new   = expf(m - m_new) * l + row_sum;

        // O rescale
        {
            int d = lane;
            if (d < D)
                O_lds[warp * D + d] *= expf(m - m_new);
        }

        // ---- Wait for V_buf[cur] to be ready, then accumulate O ----
        // At this point: DME is filling K_buf[next] + V_buf[next].
        // We now need V_buf[cur] — it was filled in the SAME prefetch batch
        // as K_buf[cur], so it's already ready (same dme::wait() as K).
        // The syncthreads here is only needed to ensure O_lds rescale is visible.
        __syncthreads();

        {
            int d = lane;
            float o_d = 0.f;
            for (int k = 0; k < Bc; ++k) {
                float p_k  = __shfl(p, k, 64);
                float v_kd = __half2float(V_buf[cur][k * D + d]);
                o_d += p_k * v_kd;
            }
            O_lds[warp * D + d] += o_d;
        }

        // ---- Wait for next tile DME to complete before next iteration ----
        if (kv + 1 < kv_tiles) {
            dme::wait();
            __syncthreads();
        } else {
            __syncthreads();
        }

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
// CPU reference
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
    printf("Config : Br=%d Bc=%d d=%d seqQ=%d seqK=%d heads=%d batch=%d\n",
           Br, Bc, D, kSeqQ, kN, kHeads, kBatch);
    printf("LDS    : 2xK_buf + 2xV_buf + O_lds = %zu B (%.1f KB of 64 KB)\n\n",
           2*Bc*D*2 + 2*Bc*D*2 + Br*D*4,
           (2.0*Bc*D*2 + 2.0*Bc*D*2 + Br*D*4) / 1024.0);
    printf("K/V    : DME double-buffer (async prefetch, sc0 streaming)\n\n");

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
    hipLaunchKernelGGL(fa_dme_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN);
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
        printf("Aborting benchmark.\n");
        return 1;
    }

    // Benchmark
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i = 0; i < kWarmup; ++i)
        hipLaunchKernelGGL(fa_dme_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN);
    hipDeviceSynchronize();

    std::vector<float> times(kIter);
    for (int i = 0; i < kIter; ++i) {
        hipEventRecord(t0);
        hipLaunchKernelGGL(fa_dme_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        hipEventElapsedTime(&times[i], t0, t1);
        times[i] *= 1000.f;
    }
    std::sort(times.begin(), times.end());
    float med_us = times[kIter/2];
    float p99_us = times[(int)(kIter*0.99f)];

    double flops_per_block = 2.0 * 2.0 * Br * Bc * D * (kN / Bc);
    double total_flops = flops_per_block * kBlocks;
    double hbm_bytes = (double)kBlocks * (kN / Bc) * (Bc * D + Bc * D) * sizeof(__half)
                     + (double)kBlocks * kSeqQ * D * sizeof(__half) * 2;

    printf("=== Flash Attention DME — benchmark (Step 3) ===\n");
    printf("  Blocks: %d  |  Tiles/block: %d  |  FLOPs: %.2f GFLOP\n",
           kBlocks, kN/Bc, total_flops/1e9);
    printf("  Median: %.1f µs  |  P99: %.1f µs\n", med_us, p99_us);
    printf("  Achieved: %.2f TFLOPS  |  %.1f GB/s HBM\n",
           total_flops / med_us / 1e6,
           hbm_bytes   / med_us / 1e3);

    printf("\n  fa_naive baseline: 97.2 µs\n");
    printf("  fa_dpp  (Step 2):  90.5 µs\n");
    printf("  fa_dme  (Step 3): %5.1f µs\n", med_us);
    printf("  Speedup vs naive:  %.3fx\n", 97.2f / med_us);
    printf("  Speedup vs dpp:    %.3fx\n", 90.5f / med_us);

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);
    hipEventDestroy(t0); hipEventDestroy(t1);
    return 0;
}

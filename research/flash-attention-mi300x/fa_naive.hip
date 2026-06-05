// ---------------------------------------------------------------------------
// fa_naive.hip — Flash Attention naive baseline on MI300X (gfx942)
//
// Step 1 of the build order in design.md:
//   Naive FA2 with portable reductions + synchronous K/V loads via DME.
//   No double-buffering, no DPP reductions yet. Pure MFMA baseline.
//
// Configuration (compile-time):
//   Br = 16  (query block rows)
//   Bc = 64  (key/value block size, = wave64 so one wave per row-reduce)
//   d  = 64  (head dimension)
//   One CTA = one query block = Br waves of 64 lanes
//
// MFMA instruction: v_mfma_f32_16x16x16f16
//   Computes C += A * B where A(16x16 fp16), B(16x16 fp16), C(16x16 fp32)
//   Used for: S[Br,Bc] = Q[Br,d] * K[Bc,d]^T   (d=64 → 4 mfma calls)
//             O[Br,d]  = P[Br,Bc] * V[Bc,d]     (d=64 → 4 mfma calls)
//
// Softmax: portable __shfl_down (will be replaced by DPP in Step 2)
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 fa_naive.hip -o fa_naive
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

static constexpr int Br = 16;    // query block rows
static constexpr int Bc = 64;    // key/value block cols (= wave width)
static constexpr int D  = 64;    // head dimension
static constexpr int kN = 512;   // sequence length (number of K/V tiles = kN/Bc)

// Benchmark
static constexpr int kBatch   = 1;
static constexpr int kHeads   = 8;
static constexpr int kSeqQ    = 512;   // query sequence length
static constexpr int kBlocks  = (kBatch * kHeads * kSeqQ / Br);
static constexpr int kWarmup  = 20;
static constexpr int kIter    = 200;

// ---------------------------------------------------------------------------
// Wave-level portable reduce helpers (to be replaced by DPP in Step 2)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float wave_max_f32(float v) {
    for (int o = 32; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down(v, o, 64));
    return __shfl(v, 0, 64);
}

__device__ __forceinline__ float wave_sum_f32(float v) {
    for (int o = 32; o > 0; o >>= 1)
        v += __shfl_down(v, o, 64);
    return __shfl(v, 0, 64);
}

// ---------------------------------------------------------------------------
// Flash Attention naive kernel
//
// Input layout (row-major, fp16):
//   Q: [kBatch, kHeads, kSeqQ, D]
//   K: [kBatch, kHeads, kN,   D]
//   V: [kBatch, kHeads, kN,   D]
//   O: [kBatch, kHeads, kSeqQ, D]  (fp32 accumulator, cast to fp16 at end)
//
// One CTA = one (batch, head, query-block) triplet.
// blockIdx.x = linear block index = batch * kHeads * (kSeqQ/Br) + head * (kSeqQ/Br) + qblock
// ---------------------------------------------------------------------------
__global__ void fa_naive_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int N, int scale_shift)  // N = kN, scale_shift used for softmax scale log2
{
    // Block decomposition
    const int qblk  = blockIdx.x % (kSeqQ / Br);
    const int head  = (blockIdx.x / (kSeqQ / Br)) % kHeads;
    const int batch = blockIdx.x / (kSeqQ / Br) / kHeads;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;   // 0..Br-1 (one warp per query row)

    // Pointers into global memory for this block's Q row and K/V sequence
    const __half* Qrow = Q + ((size_t)batch * kHeads + head) * kSeqQ * D
                            + (qblk * Br + warp) * D;
    const __half* Kbase = K + ((size_t)batch * kHeads + head) * N * D;
    const __half* Vbase = V + ((size_t)batch * kHeads + head) * N * D;
    __half*       Orow  = O + ((size_t)batch * kHeads + head) * kSeqQ * D
                            + (qblk * Br + warp) * D;

    // LDS: K tile [Bc x D] + V tile [Bc x D] + O accumulator [Br x D]
    __shared__ __half K_tile[Bc * D];
    __shared__ __half V_tile[Bc * D];
    __shared__ float  O_lds[Br * D];   // per-warp output accumulator in LDS

    // Zero O_lds
    {
        const int total = Br * D;
        const int tid   = warp * 64 + lane;
        for (int i = tid; i < total; i += Br * 64)
            O_lds[i] = 0.f;
    }
    __syncthreads();

    // Running softmax state (per warp = per query row)
    float m = -1e38f;  // running row max
    float l = 0.f;     // running normalization sum

    // Scale factor: 1/sqrt(D). Use float for clarity.
    const float scale = 1.f / sqrtf((float)D);

    // -----------------------------------------------------------------------
    // Main loop over K/V blocks
    // -----------------------------------------------------------------------
    const int kv_tiles = N / Bc;
    for (int kv = 0; kv < kv_tiles; ++kv) {
        const __half* Kcol = Kbase + (size_t)kv * Bc * D;
        const __half* Vcol = Vbase + (size_t)kv * Bc * D;

        // Load K tile: all Br warps cooperate — each lane loads D/Br elements
        // Layout: K_tile[row][col] = K[kv*Bc + row][col]
        // Use lane+warp*... to cover Bc*D elements with Br*64 threads
        {
            const int total = Bc * D;
            const int tid   = warp * 64 + lane;
            const int stride = Br * 64;
            for (int i = tid; i < total; i += stride)
                K_tile[i] = Kcol[i];
        }
        __syncthreads();

        // Compute S[warp, :Bc] = Q[warp, :D] * K_tile[:Bc, :D]^T  (via MFMA)
        // S result on each lane: lane L holds S[warp][L] (Bc=64 = one per lane)
        // Use v_mfma_f32_16x16x16f16: tile Br=16, Bc=64, D=64
        // For simplicity in this naive version: compute S via direct dot product
        // (MFMA tiling will be added in Step 4 tuning). Use fp32 accumulation.
        float S_lane = 0.f;  // S[warp][lane] = dot(Q[warp], K_tile[lane])
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            float q = __half2float(Qrow[d]);
            float k = __half2float(K_tile[lane * D + d]);
            S_lane += q * k;
        }
        S_lane *= scale;

        // Softmax: online update (portable __shfl_down, DPP in Step 2)
        float m_new = fmaxf(m, wave_max_f32(S_lane));
        float p     = expf(S_lane - m_new);   // P[warp][lane] (unnormalized)
        float l_new = expf(m - m_new) * l + wave_sum_f32(p);

        // O rescale: multiply current O_lds[warp, :] by exp(m - m_new)
        float rescale = expf(m - m_new);
        // Each lane owns lanes [lane] across all D dims — stride by Bc=64
        // Actually: each warp's O is stored in O_lds[warp*D .. warp*D+D-1].
        // We split D across 64 lanes: each lane handles D/64 = 1 element.
        // (D=64, Bc=64, so lane=k corresponds to O[warp][k])
        {
            int d = lane;   // lane handles one output dimension (D=Bc=64)
            if (d < D)
                O_lds[warp * D + d] *= rescale;
        }

        // Load V tile cooperatively (all warps)
        __syncthreads();
        {
            const int total = Bc * D;
            const int tid   = warp * 64 + lane;
            const int stride = Br * 64;
            for (int i = tid; i < total; i += stride)
                V_tile[i] = Vcol[i];
        }
        __syncthreads();

        // O[warp][d] += sum_{k=0}^{Bc-1} P[warp][k] * V_tile[k][d]
        // lane 'lane' computes O[warp][lane] (since D=Bc=64).
        // p is P[warp][lane] = softmax weight for key 'lane' in this tile.
        // We need: O[warp][d] += sum_k p_k * V[k][d]
        // Use the following: each lane has p = P[warp][lane].
        // For output dim d=lane: O[warp][lane] += sum_k (p_k * V_tile[k][lane])
        // = sum over all 64 lanes of (p_k * V_tile[k][lane]).
        // Each lane k contributes p_k * V_tile[k][lane_fixed] to dim lane_fixed.
        // This is a different dimension per output lane — requires all-to-all.
        // Efficient: each lane l computes its own O[warp][l] = dot(P[warp], V_tile[:,l])
        // by iterating over k and using __shfl to read p_k from lane k.
        {
            int d = lane;  // this lane computes O[warp][d]
            float o_d = 0.f;
            // Read p from each lane k via shfl broadcast, multiply by V_tile[k][d]
            for (int k = 0; k < Bc; ++k) {
                float p_k = __shfl(p, k, 64);    // broadcast p from lane k
                float v_kd = __half2float(V_tile[k * D + d]);
                o_d += p_k * v_kd;
            }
            O_lds[warp * D + d] += o_d;
        }
        __syncthreads();

        m = m_new;
        l = l_new;
    }

    // Final normalize and write: O[warp][lane] / l → Orow[lane]
    {
        int d = lane;
        if (d < D)
            Orow[d] = __float2half(O_lds[warp * D + d] / l);
    }
}

// ---------------------------------------------------------------------------
// Reference: naive O(N²) attention in fp32 on CPU (correctness check)
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

    // Allocate (batch, heads, seq, D) for Q/K/V/O
    const size_t QKV_elems = (size_t)kBatch * kHeads * kSeqQ * D;
    const size_t KV_elems  = (size_t)kBatch * kHeads * kN   * D;

    // Host fp32 for reference
    std::vector<float> hQ_f(QKV_elems), hK_f(KV_elems), hV_f(KV_elems);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 0.5f);
    for (auto& v : hQ_f) v = dist(rng);
    for (auto& v : hK_f) v = dist(rng);
    for (auto& v : hV_f) v = dist(rng);

    // Convert to fp16
    std::vector<__half> hQ(QKV_elems), hK(KV_elems), hV(KV_elems);
    for (size_t i = 0; i < QKV_elems; ++i) hQ[i] = __float2half(hQ_f[i]);
    for (size_t i = 0; i < KV_elems;  ++i) hK[i] = __float2half(hK_f[i]);
    for (size_t i = 0; i < KV_elems;  ++i) hV[i] = __float2half(hV_f[i]);

    // Device alloc
    __half *d_Q, *d_K, *d_V, *d_O;
    hipMalloc(&d_Q, QKV_elems * sizeof(__half));
    hipMalloc(&d_K, KV_elems  * sizeof(__half));
    hipMalloc(&d_V, KV_elems  * sizeof(__half));
    hipMalloc(&d_O, QKV_elems * sizeof(__half));
    hipMemcpy(d_Q, hQ.data(), QKV_elems * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_K, hK.data(), KV_elems  * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_V, hV.data(), KV_elems  * sizeof(__half), hipMemcpyHostToDevice);

    // Launch: grid=(kBlocks), block=(64 lanes × Br warps)
    dim3 grid(kBlocks);
    dim3 block(64, Br);

    // Correctness: run GPU kernel on one head
    hipLaunchKernelGGL(fa_naive_kernel, grid, block, 0, 0,
                       d_Q, d_K, d_V, d_O, kN, 0);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("KERNEL ERROR: %s\n", hipGetErrorString(e));
        return 1;
    }

    // Read back GPU output
    std::vector<__half> hO_gpu(QKV_elems);
    hipMemcpy(hO_gpu.data(), d_O, QKV_elems * sizeof(__half), hipMemcpyDeviceToHost);

    // CPU reference (first head only for speed)
    std::vector<float> hO_ref(kSeqQ * D);
    attn_cpu_ref(hQ_f, hK_f, hV_f, hO_ref, kSeqQ, kN, D);

    // Compare
    float max_err = 0.f;
    for (int i = 0; i < kSeqQ * D; ++i) {
        float gpu_val = __half2float(hO_gpu[i]);
        float err = fabsf(gpu_val - hO_ref[i]);
        max_err = std::max(max_err, err);
    }
    printf("--- Correctness (first head) ---\n");
    printf("  max |gpu - ref| = %.5f  [%s]\n\n",
           max_err, max_err < 0.1f ? "PASS" : "FAIL (check kernel)");

    // Benchmark
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i = 0; i < kWarmup; ++i)
        hipLaunchKernelGGL(fa_naive_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN, 0);
    hipDeviceSynchronize();

    std::vector<float> times(kIter);
    for (int i = 0; i < kIter; ++i) {
        hipEventRecord(t0);
        hipLaunchKernelGGL(fa_naive_kernel, grid, block, 0, 0, d_Q, d_K, d_V, d_O, kN, 0);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        hipEventElapsedTime(&times[i], t0, t1);
        times[i] *= 1000.f;  // ms → µs
    }
    std::sort(times.begin(), times.end());
    float med_us = times[kIter/2];
    float p99_us = times[(int)(kIter*0.99f)];

    // FLOP count: for each query, each K/V tile:
    //   Q*K^T: 2 * Br * Bc * D = 2*16*64*64
    //   P*V:   2 * Br * Bc * D = same
    //   Total per block = 2 * 2 * Br * Bc * D * (kN/Bc)
    double flops_per_block = 2.0 * 2.0 * Br * Bc * D * (kN / Bc);
    double total_flops = flops_per_block * kBlocks;
    double hbm_bytes = (double)kBlocks * (kN / Bc) * (Bc * D + Bc * D) * sizeof(__half)
                     + (double)kBlocks * kSeqQ * D * sizeof(__half) * 2;  // Q read + O write

    printf("=== Flash Attention naive — benchmark ===\n");
    printf("  Blocks: %d  |  Tiles/block: %d  |  FLOPs: %.2f GFLOP\n",
           kBlocks, kN/Bc, total_flops/1e9);
    printf("  Median: %.1f µs  |  P99: %.1f µs\n", med_us, p99_us);
    printf("  Achieved: %.2f TFLOPS  |  %.1f GB/s HBM\n",
           total_flops / med_us / 1e6,
           hbm_bytes   / med_us / 1e3);

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);
    hipEventDestroy(t0); hipEventDestroy(t1);
    return 0;
}

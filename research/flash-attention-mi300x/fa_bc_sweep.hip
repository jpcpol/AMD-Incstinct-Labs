// ---------------------------------------------------------------------------
// fa_bc_sweep.hip — Bc tile-size sweep on MI300X (gfx942)
//
// Tests the "relaxed LDS pressure hypothesis" from design.md:
//   "DPP softmax uses zero LDS, leaving more LDS budget for K/V tiles.
//    Does a larger Bc (wider key block) improve MFMA utilization and
//    reduce the number of KV iterations?"
//
// Bc sweep: 32, 64, 96  (Bc=128 exceeds 64 KB LDS with double-buffer)
//
// LDS budget per config (D=64, Br=16, double-buffer):
//   Bc=32:  2×(32×64×2)×2 + 16×64×4 = 16384 + 4096 = 20 KB
//   Bc=64:  2×(64×64×2)×2 + 16×64×4 = 32768 + 4096 = 36 KB  (fa_dme baseline)
//   Bc=96:  2×(96×64×2)×2 + 16×64×4 = 49152 + 4096 = 53 KB
//   Bc=128: 2×(128×64×2)×2 + 16×64×4= 65536 + 4096 = 69 KB  EXCEEDS 64KB
//
// For Bc != 64: one wave (64 lanes) can no longer hold one full row of S.
//   Bc=32: each lane holds S[warp][lane] for lane<32, lanes 32-63 idle in S step
//   Bc=96: requires 2 waves to cover all 96 key positions — handled via two passes
//          or by mapping lane → k with stride. Here we use a simple per-lane
//          loop over k in [0, Bc), broadcasting p_k via __shfl across the wave.
//          This is correct for any Bc; it just means some lanes are less busy
//          during the dot product when Bc < 64.
//
// Constraint: kN must be divisible by Bc for all tested values.
//   kN=512: 512/32=16, 512/64=8, 512/96 not integer → use kN=576 (=lcm(32,64,96)/…
//   Actually lcm(32,64,96)=192, 576=3×192. Use kN=576.
//   But CPU ref uses kN=512 for comparison. Keep kN=576 and adjust ref.
//   Simpler: use kN=512 for Bc=32 and Bc=64; kN=480 for Bc=96 (480/96=5).
//   Cleanest: test each Bc with its own kN that is divisible, same total FLOPs.
//   We fix kN=576 = divisible by 32(18), 64(9), 96(6).  seq_k = 576.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../wave-primitives/include/wave_primitives \
//         -I../dme-abstraction/include \
//         fa_bc_sweep.hip -o fa_bc_sweep
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

// Fixed geometry
static constexpr int Br    = 16;
static constexpr int D     = 64;
static constexpr int kN    = 576;   // divisible by 32, 64, 96
static constexpr int kBatch  = 1;
static constexpr int kHeads  = 8;
static constexpr int kSeqQ   = 576;
static constexpr int kWarmup = 20;
static constexpr int kIter   = 200;

// ---------------------------------------------------------------------------
// Templated FA kernel — Bc is a compile-time parameter
// ---------------------------------------------------------------------------
template <int Bc>
__global__ void fa_bc_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int N)
{
    static_assert(Bc % 32 == 0, "Bc must be multiple of 32");

    const int qblk  = blockIdx.x % (kSeqQ / Br);
    const int head  = (blockIdx.x / (kSeqQ / Br)) % kHeads;
    const int batch = blockIdx.x / (kSeqQ / Br) / kHeads;
    const int lane  = threadIdx.x;
    const int warp  = threadIdx.y;

    const __half* Qrow  = Q + ((size_t)batch * kHeads + head) * kSeqQ * D
                             + (qblk * Br + warp) * D;
    const __half* Kbase = K + ((size_t)batch * kHeads + head) * N * D;
    const __half* Vbase = V + ((size_t)batch * kHeads + head) * N * D;
    __half*       Orow  = O + ((size_t)batch * kHeads + head) * kSeqQ * D
                             + (qblk * Br + warp) * D;

    // Double-buffer LDS — Bc is template param so sizes are compile-time
    __shared__ __half K_buf[2][Bc * D];
    __shared__ __half V_buf[2][Bc * D];
    __shared__ float  O_lds[Br * D];

    // Zero O_lds
    {
        const int tid    = warp * 64 + lane;
        const int total  = Br * D;
        for (int i = tid; i < total; i += Br * 64)
            O_lds[i] = 0.f;
    }

    float m = -1e38f;
    float l = 0.f;
    const float scale    = 1.f / sqrtf((float)D);
    const int   kv_tiles = N / Bc;

    // Number of fp32-equivalent elements per tile (fp16 → reinterpret as float)
    constexpr int kTileFloats = Bc * D / 2;

    // Prefetch tile 0
    {
        const float* Ks = reinterpret_cast<const float*>(Kbase);
        const float* Vs = reinterpret_cast<const float*>(Vbase);
        float* Kd = reinterpret_cast<float*>(K_buf[0]);
        float* Vd = reinterpret_cast<float*>(V_buf[0]);
        const int tid    = warp * 64 + lane;
        const int stride = Br * 64;
        for (int i = tid; i < kTileFloats; i += stride) {
            dme::copy_element_stream<float>(Ks + i, Kd + i);
            dme::copy_element_stream<float>(Vs + i, Vd + i);
        }
    }
    dme::wait();
    __syncthreads();

    for (int kv = 0; kv < kv_tiles; ++kv) {
        const int cur  = kv & 1;
        const int next = cur ^ 1;

        // Async prefetch next tile
        if (kv + 1 < kv_tiles) {
            const float* Ks = reinterpret_cast<const float*>(Kbase + (size_t)(kv+1)*Bc*D);
            const float* Vs = reinterpret_cast<const float*>(Vbase + (size_t)(kv+1)*Bc*D);
            float* Kd = reinterpret_cast<float*>(K_buf[next]);
            float* Vd = reinterpret_cast<float*>(V_buf[next]);
            const int tid    = warp * 64 + lane;
            const int stride = Br * 64;
            for (int i = tid; i < kTileFloats; i += stride) {
                dme::copy_element_stream<float>(Ks + i, Kd + i);
                dme::copy_element_stream<float>(Vs + i, Vd + i);
            }
        }

        // Compute S[warp][k] for k in [0, Bc) — each lane handles k=lane mod Bc
        // For Bc<=64: lane iterates over its assigned key(s)
        // Result: p[k] = exp(S[k] - m_new), needed for O accumulation
        // We store p in a small array — Bc/64 elements per lane (1 for Bc=64, <1 for Bc<64)
        // For Bc=96: 2 elements per lane for lanes 0-31, 1 for lanes 32-63.
        // Simplification: compute p[k] inline during O accumulation via shfl.
        // Here: compute S for all k in [0,Bc), each lane contributes one S value.

        // lane L handles key index k = L (for Bc <= 64, only lanes 0..Bc-1 are active)
        // For Bc > 64 (Bc=96): lane L handles keys L and L+64 if L < Bc-64
        float S0 = 0.f, S1 = 0.f;
        constexpr bool has_second = (Bc > 64);

        // Key 0: lane L → K_buf[cur][L * D + d]
        if (lane < Bc) {
            #pragma unroll
            for (int d = 0; d < D; ++d)
                S0 += __half2float(Qrow[d]) * __half2float(K_buf[cur][lane * D + d]);
            S0 *= scale;
        }
        // Key 1 (Bc=96): lane L handles key L+64 for L in [0, 32)
        if constexpr (has_second) {
            if (lane < Bc - 64) {
                #pragma unroll
                for (int d = 0; d < D; ++d)
                    S1 += __half2float(Qrow[d]) * __half2float(K_buf[cur][(lane + 64) * D + d]);
                S1 *= scale;
            }
        }

        // Row max over all Bc scores — need max across [0, Bc)
        // For Bc=64: standard wave reduce
        // For Bc<64: only lanes [0,Bc) have valid scores; others contribute -inf identity
        // For Bc=96: S0 on lanes [0,96), S1 on lanes [0,32); need inter-lane max
        float S0_for_reduce = (lane < Bc) ? S0 : -1e38f;
        float row_max_val;
        if constexpr (has_second) {
            // max of S0 and S1 locally, then wave reduce
            float local_max = S0_for_reduce;
            if (lane < Bc - 64) local_max = fmaxf(local_max, S1);
            row_max_val = wave::dpp::reduce_max_bcast<float>(local_max);
        } else {
            row_max_val = wave::dpp::reduce_max_bcast<float>(S0_for_reduce);
        }

        float m_new = fmaxf(m, row_max_val);

        // Compute exp(S - m_new) for all keys
        float p0 = (lane < Bc)      ? expf(S0 - m_new) : 0.f;
        float p1 = (lane < Bc - 64 && has_second) ? expf(S1 - m_new) : 0.f;

        // Row sum
        float local_sum = p0;
        if constexpr (has_second) local_sum += p1;
        float row_sum = wave::dpp::reduce_sum_bcast<float>(local_sum);
        float l_new   = expf(m - m_new) * l + row_sum;

        // Rescale O
        float rescale = expf(m - m_new);
        { int d = lane; if (d < D) O_lds[warp * D + d] *= rescale; }

        // Wait for V tile
        __syncthreads();

        // O[warp][d] += sum_{k=0}^{Bc-1} p_k * V_buf[cur][k * D + d]
        // lane d computes O[warp][d]. For Bc<=64: shfl p from lane k.
        // For Bc=96: lanes 0-63 cover k=0..63 (p0), lanes 0-31 also cover k=64..95 (p1).
        {
            int d = lane;
            float o_d = 0.f;
            // First 64 keys: shfl p0 from each lane
            int k_max0 = (Bc < 64) ? Bc : 64;
            for (int k = 0; k < k_max0; ++k) {
                float p_k  = __shfl(p0, k, 64);
                float v_kd = __half2float(V_buf[cur][k * D + d]);
                o_d += p_k * v_kd;
            }
            // Extra keys [64, Bc) for Bc=96: shfl p1 from lanes 0..(Bc-65)
            if constexpr (has_second) {
                for (int k = 64; k < Bc; ++k) {
                    float p_k  = __shfl(p1, k - 64, 64);
                    float v_kd = __half2float(V_buf[cur][k * D + d]);
                    o_d += p_k * v_kd;
                }
            }
            O_lds[warp * D + d] += o_d;
        }

        // Wait for next tile DME
        if (kv + 1 < kv_tiles) { dme::wait(); __syncthreads(); }
        else                    { __syncthreads(); }

        m = m_new;
        l = l_new;
    }

    { int d = lane; if (d < D) Orow[d] = __float2half(O_lds[warp * D + d] / l); }
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

// ---------------------------------------------------------------------------
// Bench helper
// ---------------------------------------------------------------------------
template <int Bc>
float bench_bc(dim3 grid, dim3 block,
               __half* d_Q, __half* d_K, __half* d_V, __half* d_O)
{
    for (int i = 0; i < kWarmup; ++i)
        hipLaunchKernelGGL(fa_bc_kernel<Bc>, grid, block, 0, 0,
                           d_Q, d_K, d_V, d_O, kN);
    hipDeviceSynchronize();

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    std::vector<float> ts(kIter);
    for (int i = 0; i < kIter; ++i) {
        hipEventRecord(t0);
        hipLaunchKernelGGL(fa_bc_kernel<Bc>, grid, block, 0, 0,
                           d_Q, d_K, d_V, d_O, kN);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        hipEventElapsedTime(&ts[i], t0, t1);
        ts[i] *= 1000.f;
    }
    hipEventDestroy(t0); hipEventDestroy(t1);
    std::sort(ts.begin(), ts.end());
    return ts[kIter/2];
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device : %s  (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("Config : Br=%d D=%d seqQ=%d seqK=%d heads=%d batch=%d\n\n",
           Br, D, kSeqQ, kN, kHeads, kBatch);

    const int kBlocks = kBatch * kHeads * kSeqQ / Br;
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
    hipMalloc(&d_Q, QKV_elems*sizeof(__half));
    hipMalloc(&d_K, KV_elems *sizeof(__half));
    hipMalloc(&d_V, KV_elems *sizeof(__half));
    hipMalloc(&d_O, QKV_elems*sizeof(__half));
    hipMemcpy(d_Q, hQ.data(), QKV_elems*sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_K, hK.data(), KV_elems *sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(d_V, hV.data(), KV_elems *sizeof(__half), hipMemcpyHostToDevice);

    // CPU reference (first head)
    std::vector<float> hO_ref(kSeqQ * D);
    attn_cpu_ref(hQ_f, hK_f, hV_f, hO_ref, kSeqQ, kN, D);

    dim3 grid(kBlocks);
    dim3 block(64, Br);

    std::vector<__half> hO_gpu(QKV_elems);

    auto check = [&](const char* name, int Bc_val) {
        hipMemcpy(hO_gpu.data(), d_O, QKV_elems*sizeof(__half), hipMemcpyDeviceToHost);
        float mx = 0.f;
        for (int i = 0; i < kSeqQ*D; ++i)
            mx = std::max(mx, fabsf(__half2float(hO_gpu[i]) - hO_ref[i]));
        int lds_kb = (2*Bc_val*D*2*2 + Br*D*4) / 1024;
        printf("  Bc=%-3d  LDS=%-2d KB  max_err=%.5f  [%s]\n",
               Bc_val, lds_kb, mx, mx < 0.1f ? "PASS" : "FAIL");
        return mx < 0.1f;
    };

    printf("--- Correctness ---\n");

    hipLaunchKernelGGL(fa_bc_kernel<32>, grid, block, 0, 0, d_Q,d_K,d_V,d_O,kN);
    hipDeviceSynchronize(); check("Bc=32", 32);

    hipLaunchKernelGGL(fa_bc_kernel<64>, grid, block, 0, 0, d_Q,d_K,d_V,d_O,kN);
    hipDeviceSynchronize(); check("Bc=64", 64);

    hipLaunchKernelGGL(fa_bc_kernel<96>, grid, block, 0, 0, d_Q,d_K,d_V,d_O,kN);
    hipDeviceSynchronize(); check("Bc=96", 96);

    printf("\n--- Benchmark ---\n");
    printf("  %-10s  %8s  %10s  %8s  %8s\n",
           "Config", "Bc", "Median(us)", "TFLOPS", "Speedup");
    printf("  %-10s  %8s  %10s  %8s  %8s\n",
           "------", "--", "----------", "------", "-------");

    // FLOPs = 2 * 2 * Br * Bc * D * (kN/Bc) * kBlocks  (Q*K + P*V, per launch)
    auto flops = [&](int Bc_val) {
        return 2.0 * 2.0 * Br * Bc_val * D * (kN / Bc_val) * (double)kBlocks;
    };

    float t32 = bench_bc<32>(grid, block, d_Q, d_K, d_V, d_O);
    float t64 = bench_bc<64>(grid, block, d_Q, d_K, d_V, d_O);
    float t96 = bench_bc<96>(grid, block, d_Q, d_K, d_V, d_O);

    auto pr = [&](const char* label, int Bc_val, float med_us, float ref_us) {
        double tf = flops(Bc_val) / med_us / 1e6;
        printf("  %-10s  %8d  %10.1f  %8.3f  %8.3fx\n",
               label, Bc_val, med_us, tf, ref_us / med_us);
    };

    pr("Bc=32", 32, t32, t64);
    pr("Bc=64", 64, t64, t64);
    pr("Bc=96", 96, t96, t64);

    printf("\n  Hypothesis: larger Bc (more LDS freed by DPP) → fewer KV iters → faster?\n");
    printf("  kv_tiles: Bc=32 → %d iters,  Bc=64 → %d iters,  Bc=96 → %d iters\n",
           kN/32, kN/64, kN/96);

    hipFree(d_Q); hipFree(d_K); hipFree(d_V); hipFree(d_O);
    return 0;
}

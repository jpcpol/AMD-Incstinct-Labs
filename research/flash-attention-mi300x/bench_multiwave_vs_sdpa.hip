// ---------------------------------------------------------------------------
// bench_multiwave_vs_sdpa.hip — Phase-1 occupancy gap measurement.
//
// Measures fa_multiwave_kernel at all valid (W, D) configurations over the
// exact shapes used in 2-C (seqLen 512→4096, GQA 32q/8kv, causal) and
// compares against the 1-wave fa_robust baseline from 2-C.
//
// Companion: bench_sdpa_baseline.py — run separately to get F.sdpa numbers,
// then compute gap = our_us / sdpa_us offline (see Notes at bottom).
//
// Pre-registered hypotheses (DESIGN_phase1_multiwave_gap.md):
//   H-MW1: multi-wave W>1 is faster than W=1 at matched (seqLen, D).
//   H-MW2: SQ_INSTS_VMEM_RD drops ≥(1-1/W) vs W independent blocks.
//   H-MW3: gap vs SDPA narrows vs 2-C measurement (8.5-15×).
//
// LDS budget per configuration (64 KB CU limit, S_lds=half since 2026-06-19):
//   W=1 D=64 : 28800 B  = 28.1 KB  [OK]
//   W=2 D=64 : 33152 B  = 32.4 KB  [OK]
//   W=4 D=64 : 57856 B  = 56.5 KB  [OK]
//   W=1 D=128: 49152 B  = 48.0 KB  [OK]
//   W=2 D=128: ~85 KB               [OVERFLOW — skipped]
//   W=4 D=128: ~153 KB              [OVERFLOW — skipped]
//
// Note on LDS per-wave (D=128, S_lds now half):
//   S_lds[Br*Bc*f16] = 16*64*2 =  2048 B  ← was f32 (4096B), halved
//   P_lds[Br*Bc*f16] = 16*64*2 =  2048 B
//   O_lds[Br*D *f32] = 16*128*4=  8192 B
//   Q_lds[Br*D *f16] = 16*128*2=  4096 B
//   m_lds+l_lds[Br]  = 16*4*2  =   128 B
//   Total per wave   =          16512 B ≈ 16 KB
// W=2 D=128: shared K+V=32KB + 2×16KB = 65KB  [OVERFLOW — skip]
//
// Compile (from research/flash-attention-mi300x/):
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../../cdna3-prims/include \
//     bench_multiwave_vs_sdpa.hip -o bench_multiwave_vs_sdpa
//
// Run:
//   ./bench_multiwave_vs_sdpa
//   ./bench_multiwave_vs_sdpa --rocprof   # print note to run with rocprofv3
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>

#include "../../cdna3-prims/include/cdna3/detail/fa_prefill_kernel.hpp"

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------
static constexpr int kWarmup = 20;
static constexpr int kIter   = 100;

// 2-C shapes: D=128, GQA 32q/8kv, causal, seqLen sweep
static const int kSeqs[]  = {512, 1024, 2048, 4096};
static constexpr int kNSeq = 4;

// GQA config matching 2-C
static constexpr int kNQH  = 32;
static constexpr int kNKVH = 8;
static constexpr int kBatch = 1;

// ---------------------------------------------------------------------------
// CPU reference (GQA + causal) — reused from fa_multiwave.hip
// ---------------------------------------------------------------------------
static void attn_ref(const std::vector<float>& Q, const std::vector<float>& K,
                     const std::vector<float>& V, std::vector<float>& O,
                     const PrefillKCfg& c)
{
    const float scale = 1.f / sqrtf((float)c.D);
    const int group = c.nQHeads / c.nKVHeads;
    for (int b = 0; b < c.batch; ++b)
    for (int h = 0; h < c.nQHeads; ++h) {
        int kvh = h / group;
        for (int q = 0; q < c.seqQ; ++q) {
            int kmax = c.causal ? std::min(c.seqK, q+1) : c.seqK;
            std::vector<float> S(kmax); float mx = -1e38f;
            const float* qp = &Q[(((size_t)b*c.nQHeads+h)*c.seqQ+q)*c.D];
            for (int k = 0; k < kmax; ++k) {
                float dot = 0;
                const float* kp = &K[(((size_t)b*c.nKVHeads+kvh)*c.seqK+k)*c.D];
                for (int d = 0; d < c.D; ++d) dot += qp[d]*kp[d];
                S[k] = dot*scale; mx = std::max(mx, S[k]);
            }
            float se = 0;
            for (int k = 0; k < kmax; ++k) { S[k] = expf(S[k]-mx); se += S[k]; }
            for (int k = 0; k < kmax; ++k) S[k] /= se;
            float* op = &O[(((size_t)b*c.nQHeads+h)*c.seqQ+q)*c.D];
            for (int d = 0; d < c.D; ++d) {
                float o = 0;
                for (int k = 0; k < kmax; ++k) {
                    const float* vp = &V[(((size_t)b*c.nKVHeads+kvh)*c.seqK+k)*c.D];
                    o += S[k]*vp[d];
                }
                op[d] = o;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner for fa_multiwave_kernel<D, W>
// Returns median µs, or -1 on kernel error.
// verify=true: checks correctness vs CPU reference (small configs only).
// ---------------------------------------------------------------------------
template<int D, int W>
float run_multiwave(int seq, bool verify, float* err_out)
{
    // seqQ must be divisible by W*Br=W*16
    if (seq % (W * _Br) != 0) {
        if (err_out) *err_out = -1.f;
        return -1.f;
    }

    PrefillKCfg c;
    c.D = D; c.seqQ = seq; c.seqK = seq;
    c.nQHeads = kNQH; c.nKVHeads = kNKVH;
    c.batch = kBatch; c.causal = 1;

    const size_t Qel  = (size_t)c.batch * c.nQHeads  * seq * D;
    const size_t KVel = (size_t)c.batch * c.nKVHeads * seq * D;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 0.5f);
    std::vector<float> hQf(Qel), hKf(KVel), hVf(KVel);
    for (auto& v : hQf) v = dist(rng);
    for (auto& v : hKf) v = dist(rng);
    for (auto& v : hVf) v = dist(rng);

    std::vector<__half> hQ(Qel), hK(KVel), hV(KVel);
    for (size_t i = 0; i < Qel;  ++i) hQ[i] = __float2half(hQf[i]);
    for (size_t i = 0; i < KVel; ++i) { hK[i] = __float2half(hKf[i]); hV[i] = __float2half(hVf[i]); }

    __half *dQ, *dK, *dV, *dO;
    hipMalloc(&dQ, Qel*2); hipMalloc(&dK, KVel*2);
    hipMalloc(&dV, KVel*2); hipMalloc(&dO, Qel*2);
    hipMemcpy(dQ, hQ.data(), Qel*2, hipMemcpyHostToDevice);
    hipMemcpy(dK, hK.data(), KVel*2, hipMemcpyHostToDevice);
    hipMemcpy(dV, hV.data(), KVel*2, hipMemcpyHostToDevice);

    const int sblocks = seq / (W * _Br);
    const int blocks  = c.batch * c.nQHeads * sblocks;
    dim3 grid(blocks), block(W*64);

    // single warm-up launch to check for kernel errors
    hipLaunchKernelGGL((fa_multiwave_kernel<D,W>), grid, block, 0, 0, dQ, dK, dV, dO, c);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("  KERNEL ERROR D=%d W=%d seq=%d: %s\n", D, W, seq, hipGetErrorString(e));
        hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
        if (err_out) *err_out = -1.f;
        return -1.f;
    }

    if (verify) {
        std::vector<__half> hOg(Qel);
        hipMemcpy(hOg.data(), dO, Qel*2, hipMemcpyDeviceToHost);
        std::vector<float> hOr(Qel, 0.f);
        attn_ref(hQf, hKf, hVf, hOr, c);
        float err = 0.f;
        for (size_t i = 0; i < Qel; ++i)
            err = std::max(err, fabsf(__half2float(hOg[i]) - hOr[i]));
        if (err_out) *err_out = err;
    }

    // benchmark
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i = 0; i < kWarmup; ++i)
        hipLaunchKernelGGL((fa_multiwave_kernel<D,W>), grid, block, 0, 0, dQ, dK, dV, dO, c);
    hipDeviceSynchronize();

    std::vector<float> ts(kIter);
    for (int i = 0; i < kIter; ++i) {
        hipEventRecord(t0);
        hipLaunchKernelGGL((fa_multiwave_kernel<D,W>), grid, block, 0, 0, dQ, dK, dV, dO, c);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        hipEventElapsedTime(&ts[i], t0, t1);
        ts[i] *= 1000.f; // ms → µs
    }
    hipEventDestroy(t0); hipEventDestroy(t1);
    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);

    std::sort(ts.begin(), ts.end());
    return ts[kIter/2];
}

static double tflops(int seq, int nH, int D, float us) {
    // causal: 0.5× work
    double flops = 0.5 * 2.0 * 2.0 * nH * (double)seq * seq * D;
    return flops / (us * 1e-6) / 1e12;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    bool rocprof_note = false;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--rocprof") == 0) rocprof_note = true;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device : %s\n", prop.name);
    printf("Config : GQA %dq/%dkv, causal, batch=%d, warmup=%d, iters=%d\n\n",
           kNQH, kNKVH, kBatch, kWarmup, kIter);

    // -------------------------------------------------------------------------
    // Section 1: Correctness — small shapes, CPU-verified
    // (use a small seq to keep the CPU reference fast)
    // -------------------------------------------------------------------------
    printf("=== Correctness (seq=256, CPU-verified) ===\n");
    {
        // W=1 D=64 (always valid: 256 % 16 == 0)
        float err; float us;
        us = run_multiwave<64,1>(256, true, &err);
        printf("  W=1 D=64  seq=256  err=%.5f [%s]  %.1f µs\n",
               err, err < 0.15f ? "PASS" : "FAIL", us);
        us = run_multiwave<128,1>(256, true, &err);
        printf("  W=1 D=128 seq=256  err=%.5f [%s]  %.1f µs\n",
               err, err < 0.15f ? "PASS" : "FAIL", us);
        us = run_multiwave<64,2>(256, true, &err);
        printf("  W=2 D=64  seq=256  err=%.5f [%s]  %.1f µs\n",
               err, err < 0.15f ? "PASS" : "FAIL", us);
        us = run_multiwave<64,4>(256, true, &err);
        printf("  W=4 D=64  seq=256  err=%.5f [%s]  %.1f µs\n",
               err, err < 0.15f ? "PASS" : "FAIL", us);
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // Section 2: seqLen sweep — D=64, W={1,2,4} — GQA 32q/8kv, causal
    // -------------------------------------------------------------------------
    printf("=== seqLen sweep: D=64, W={1,2,4}, GQA %dq/%dkv, causal ===\n\n", kNQH, kNKVH);
    printf("%-12s %10s %10s %10s %10s %10s %10s\n",
           "seq", "W=1 µs", "W=1 TFLOPS", "W=2 µs", "W=2 TFLOPS", "W=4 µs", "W=4 TFLOPS");
    printf("%-12s %10s %10s %10s %10s %10s %10s\n",
           "---", "------", "----------", "------", "----------", "------", "----------");
    for (int i = 0; i < kNSeq; ++i) {
        int seq = kSeqs[i];
        float us1 = run_multiwave<64,1>(seq, false, nullptr);
        float us2 = run_multiwave<64,2>(seq, false, nullptr);
        float us4 = run_multiwave<64,4>(seq, false, nullptr);
        printf("%-12d %10.1f %10.2f %10.1f %10.2f %10.1f %10.2f\n",
               seq,
               us1, us1 > 0 ? tflops(seq, kNQH, 64, us1) : 0.0,
               us2, us2 > 0 ? tflops(seq, kNQH, 64, us2) : 0.0,
               us4, us4 > 0 ? tflops(seq, kNQH, 64, us4) : 0.0);
    }

    // -------------------------------------------------------------------------
    // Section 3: seqLen sweep — D=128, W=1 only (W=2+ overflows LDS at D=128)
    // This is the apples-to-apples comparison with 2-C (fa_robust was also W=1 D=128).
    // -------------------------------------------------------------------------
    printf("\n=== seqLen sweep: D=128, W=1, GQA %dq/%dkv, causal ===\n", kNQH, kNKVH);
    printf("(W=2+ at D=128 exceeds 64KB LDS budget — skipped; see DESIGN doc)\n\n");
    printf("%-12s %10s %10s\n", "seq", "W=1 µs", "W=1 TFLOPS");
    printf("%-12s %10s %10s\n", "---", "------", "----------");

    float us128[kNSeq];
    for (int i = 0; i < kNSeq; ++i) {
        int seq = kSeqs[i];
        us128[i] = run_multiwave<128,1>(seq, false, nullptr);
        printf("%-12d %10.1f %10.2f\n",
               seq, us128[i], us128[i] > 0 ? tflops(seq, kNQH, 128, us128[i]) : 0.0);
    }

    // -------------------------------------------------------------------------
    // Section 4: W speedup summary for D=64
    // Reports W=2/W=1 and W=4/W=1 at each seqLen.
    // H-MW1 acceptance: multi-wave faster on ≥3/4 seqLens.
    // -------------------------------------------------------------------------
    printf("\n=== W speedup summary (D=64, W>1 vs W=1) — H-MW1 ===\n\n");
    printf("%-12s %12s %12s\n", "seq", "W=2/W=1", "W=4/W=1");
    printf("%-12s %12s %12s\n", "---", "-------", "-------");

    int hw1_w2 = 0, hw1_w4 = 0;
    float us1_d64[kNSeq], us2_d64[kNSeq], us4_d64[kNSeq];
    for (int i = 0; i < kNSeq; ++i) {
        int seq = kSeqs[i];
        us1_d64[i] = run_multiwave<64,1>(seq, false, nullptr);
        us2_d64[i] = run_multiwave<64,2>(seq, false, nullptr);
        us4_d64[i] = run_multiwave<64,4>(seq, false, nullptr);
        float r2 = (us1_d64[i] > 0 && us2_d64[i] > 0) ? us1_d64[i]/us2_d64[i] : 0.f;
        float r4 = (us1_d64[i] > 0 && us4_d64[i] > 0) ? us1_d64[i]/us4_d64[i] : 0.f;
        if (r2 > 1.0f) ++hw1_w2;
        if (r4 > 1.0f) ++hw1_w4;
        printf("%-12d %11.3f× %11.3f×\n", seq, r2, r4);
    }
    printf("\nH-MW1 W=2: %s (%d/4 seqLens faster)\n",
           hw1_w2 >= 3 ? "CONFIRMED" : "NOT CONFIRMED", hw1_w2);
    printf("H-MW1 W=4: %s (%d/4 seqLens faster)\n",
           hw1_w4 >= 3 ? "CONFIRMED" : "NOT CONFIRMED", hw1_w4);

    // -------------------------------------------------------------------------
    // Section 5: 2-C gap update — compare D=128 W=1 multiwave vs 2-C numbers.
    // 2-C measured fa_robust (same kernel, W=1) at these latencies:
    //   seq=512: 389µs, seq=1024: 1052µs, seq=2048: 3408µs, seq=4096: 11829µs
    // fa_robust had identical architecture to fa_multiwave W=1 — any difference
    // reveals compilation/tuning changes since 2-C, not a design change.
    // -------------------------------------------------------------------------
    static const float kRobust2C[4] = {389.5f, 1051.9f, 3407.9f, 11829.4f};
    static const float kSDPA2C[4]   = {45.8f, 77.3f, 223.2f, 980.6f};

    printf("\n=== 2-C gap update: fa_multiwave W=1 D=128 vs fa_robust (2-C) vs SDPA (2-C) ===\n");
    printf("(SDPA numbers from 2-C session 2026-06-10 — run bench_sdpa_baseline.py for fresh numbers)\n\n");
    printf("%-6s %12s %12s %12s %10s %10s\n",
           "seq", "multiwave µs", "robust-2C µs", "SDPA-2C µs", "vs robust", "vs SDPA-2C");
    printf("%-6s %12s %12s %12s %10s %10s\n",
           "---", "------------", "------------", "----------", "---------", "----------");
    for (int i = 0; i < kNSeq; ++i) {
        int seq = kSeqs[i];
        float us = us128[i];
        float vr = (us > 0 && kRobust2C[i] > 0) ? kRobust2C[i]/us : 0.f;
        float vs = (us > 0 && kSDPA2C[i] > 0) ? us/kSDPA2C[i] : 0.f;
        printf("%-6d %12.1f %12.1f %12.1f %9.3f× %9.2f×\n",
               seq, us, kRobust2C[i], kSDPA2C[i], vr, vs);
    }
    printf("\n(vs robust >1× means multiwave W=1 is faster than fa_robust 2-C — expected if any tuning improved)\n");
    printf("(vs SDPA-2C: the gap number — compare with 8.5-15× from 2-C to assess H-MW3 direction)\n");

    // -------------------------------------------------------------------------
    // Notes for rocprofv3 (H-MW2)
    // -------------------------------------------------------------------------
    printf("\n=== Notes ===\n");
    printf("H-MW2 (VMEM_RD reduction) requires rocprofv3. Run:\n");
    printf("  rocprofv3 --pmc SQ_INSTS_VMEM_RD,SQ_INSTS_LDS,SQ_INSTS_VALU \\\n");
    printf("    ./bench_multiwave_vs_sdpa\n");
    printf("Expected: W=2 VMEM_RD ≈ 0.5× W=1; W=4 VMEM_RD ≈ 0.25× W=1 (K/V LDS reuse).\n");
    printf("\nFor fresh SDPA numbers (H-MW3 precise gap):\n");
    printf("  python bench_sdpa_baseline.py --seqs 512,1024,2048,4096 --D 128 --nH 32 --nKV 8\n");
    if (rocprof_note)
        printf("\n[--rocprof flag set: results above are suitable for rocprofv3 profiling]\n");

    return 0;
}

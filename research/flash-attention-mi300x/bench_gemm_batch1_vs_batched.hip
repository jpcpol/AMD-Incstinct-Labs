// ---------------------------------------------------------------------------
// bench_gemm_batch1_vs_batched.hip — Phase-2 GEMM batch=1 occupancy gap.
//
// Measures rocBLAS gemm_ex at M=1,8,16,32 for the exact GEMM shapes in
// the Qwen2.5-0.5B decode path (D=896, hidden=4864) to quantify the
// occupancy cliff at M=1 vs batched decode.
//
// Pre-registered hypotheses (DESIGN_gemm_batch1_occupancy.md):
//   H-G1: TFLOPS(M=1) / TFLOPS(M=32) < 0.10 (occupancy cliff exists)
//   H-G2: TFLOPS scales ~linearly up to M=16, saturates at M=32
//   H-G3: M=1 is bandwidth-bound (GB/s ratio > TFLOPS ratio vs peak)
//
// Compile (from research/flash-attention-mi300x/):
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -lrocblas \
//     bench_gemm_batch1_vs_batched.hip -o bench_gemm_batch1
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <random>

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------
static constexpr int kWarmup = 20;
static constexpr int kIter   = 200;   // more iters: M=1 is very fast

// MI300X roofline reference (MI300X specs)
static constexpr double kPeakTFLOPS = 383.0;  // fp16 tensor peak
static constexpr double kPeakGBs    = 5300.0; // HBM3 peak

// ---------------------------------------------------------------------------
// rocBLAS error check
// ---------------------------------------------------------------------------
#define RBCHECK(x) do { \
    rocblas_status _s = (x); \
    if (_s != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        fflush(stderr); \
        abort(); \
    } \
} while(0)

#define HIPCHECK(x) do { \
    hipError_t _e = (x); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
        fflush(stderr); \
        abort(); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// GEMM shapes from Qwen2.5-0.5B layer (real decode shapes)
// ---------------------------------------------------------------------------
struct GemmShape { const char* name; int M, N, K; };

static const GemmShape kShapes[] = {
    // QKV projection: hidden→3*heads*head_dim = 896→2688
    {"qkv_m1",  1,  2688, 896},
    {"qkv_m8",  8,  2688, 896},
    {"qkv_m16", 16, 2688, 896},
    {"qkv_m32", 32, 2688, 896},
    // MLP gate/up projection: hidden→intermediate = 896→4864
    {"mlp_m1",  1,  4864, 896},
    {"mlp_m8",  8,  4864, 896},
    {"mlp_m16", 16, 4864, 896},
    {"mlp_m32", 32, 4864, 896},
};
static constexpr int kNShapes = sizeof(kShapes) / sizeof(kShapes[0]);

// ---------------------------------------------------------------------------
// Benchmark one GEMM config
//   C[M,N] = A[M,K] × W[N,K]^T    (fp16 in, fp32 out — matches session.hpp)
// rocBLAS call (row-major → col-major identity, same as rocblas_gemm.hpp):
//   gemm(transA=T, transB=N, m=N, n=M, k=K, W, lda=K, A, ldb=K, C, ldc=N)
// ---------------------------------------------------------------------------
struct BenchResult { double us; double tflops; double gbs; };

static BenchResult bench_gemm(rocblas_handle h,
                               const __half* A, const __half* W, float* C,
                               int M, int N, int K)
{
    float alpha = 1.f, beta = 0.f;

    // Warmup
    for (int i = 0; i < kWarmup; ++i) {
        RBCHECK(rocblas_gemm_ex(
            h,
            rocblas_operation_transpose, rocblas_operation_none,
            N, M, K,
            &alpha,
            W, rocblas_datatype_f16_r, K,
            A, rocblas_datatype_f16_r, K,
            &beta,
            C, rocblas_datatype_f32_r, N,
            C, rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0
        ));
    }
    HIPCHECK(hipDeviceSynchronize());

    // Timed iterations
    std::vector<double> times(kIter);
    for (int i = 0; i < kIter; ++i) {
        hipEvent_t t0, t1;
        HIPCHECK(hipEventCreate(&t0));
        HIPCHECK(hipEventCreate(&t1));
        HIPCHECK(hipEventRecord(t0));
        RBCHECK(rocblas_gemm_ex(
            h,
            rocblas_operation_transpose, rocblas_operation_none,
            N, M, K,
            &alpha,
            W, rocblas_datatype_f16_r, K,
            A, rocblas_datatype_f16_r, K,
            &beta,
            C, rocblas_datatype_f32_r, N,
            C, rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0
        ));
        HIPCHECK(hipEventRecord(t1));
        HIPCHECK(hipEventSynchronize(t1));
        float ms = 0;
        HIPCHECK(hipEventElapsedTime(&ms, t0, t1));
        times[i] = ms * 1000.0;  // µs
        HIPCHECK(hipEventDestroy(t0));
        HIPCHECK(hipEventDestroy(t1));
    }

    std::sort(times.begin(), times.end());
    double us = times[kIter / 2];

    // TFLOPS: 2*M*N*K flops (one multiply-add = 2 flops)
    double flops = 2.0 * M * N * K;
    double tflops = flops / (us * 1e-6) / 1e12;

    // Effective bandwidth: read A[M,K] + W[N,K] + write C[M,N], all fp16/fp32
    double bytes = (double)M*K*2 + (double)N*K*2 + (double)M*N*4;
    double gbs = bytes / (us * 1e-6) / 1e9;

    return {us, tflops, gbs};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device : %s\n", prop.name);
    printf("Config : rocBLAS gemm_ex, fp16 in / fp32 out, %d warmup / %d iters\n",
           kWarmup, kIter);
    printf("Shapes : Qwen2.5-0.5B decode (D=896, intermediate=4864)\n\n");

    rocblas_handle h;
    RBCHECK(rocblas_create_handle(&h));

    // Allocate max-size buffers (M=32, N=4864, K=896)
    static constexpr int kMaxM = 32, kMaxN = 4864, kMaxK = 896;
    __half *dA, *dW; float *dC;
    HIPCHECK(hipMalloc(&dA, (size_t)kMaxM * kMaxK * sizeof(__half)));
    HIPCHECK(hipMalloc(&dW, (size_t)kMaxN * kMaxK * sizeof(__half)));
    HIPCHECK(hipMalloc(&dC, (size_t)kMaxM * kMaxN * sizeof(float)));

    // Fill with random fp16 values
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        std::vector<__half> hA(kMaxM * kMaxK), hW(kMaxN * kMaxK);
        for (auto& x : hA) x = __float2half(dist(rng));
        for (auto& x : hW) x = __float2half(dist(rng));
        HIPCHECK(hipMemcpy(dA, hA.data(), hA.size()*2, hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(dW, hW.data(), hW.size()*2, hipMemcpyHostToDevice));
    }

    // ---------------------------------------------------------------------------
    // Section 1: raw latency and TFLOPS per shape
    // ---------------------------------------------------------------------------
    printf("=== Section 1: Latency and TFLOPS per shape ===\n");
    printf("%-12s %6s %6s %6s   %8s  %10s  %10s\n",
           "shape", "M", "N", "K", "us", "TFLOPS", "GB/s");
    printf("%-12s %6s %6s %6s   %8s  %10s  %10s\n",
           "------------", "------", "------", "------",
           "--------", "----------", "----------");

    BenchResult results[kNShapes];
    for (int i = 0; i < kNShapes; ++i) {
        const GemmShape& s = kShapes[i];
        results[i] = bench_gemm(h, dA, dW, dC, s.M, s.N, s.K);
        printf("%-12s %6d %6d %6d   %8.1f  %10.3f  %10.1f\n",
               s.name, s.M, s.N, s.K,
               results[i].us, results[i].tflops, results[i].gbs);
    }

    // ---------------------------------------------------------------------------
    // Section 2: H-G3 — bandwidth vs compute utilization at M=1
    // ---------------------------------------------------------------------------
    printf("\n=== Section 2: H-G3 — roofline utilization at M=1 ===\n");
    printf("Peak TFLOPS: %.0f  Peak GB/s: %.0f\n", kPeakTFLOPS, kPeakGBs);
    printf("%-12s  %8s  %8s  %12s\n", "shape", "TF util%", "BW util%", "bottleneck");
    printf("%-12s  %8s  %8s  %12s\n", "------------", "--------", "--------", "------------");
    for (int i = 0; i < kNShapes; ++i) {
        if (kShapes[i].M != 1) continue;
        double tf_pct = results[i].tflops / kPeakTFLOPS * 100.0;
        double bw_pct = results[i].gbs    / kPeakGBs    * 100.0;
        const char* bot = (bw_pct > tf_pct) ? "compute" : "bandwidth";
        printf("%-12s  %7.2f%%  %7.2f%%  %12s\n",
               kShapes[i].name, tf_pct, bw_pct, bot);
    }

    // ---------------------------------------------------------------------------
    // Section 3: H-G1 — occupancy cliff: TFLOPS(M=1) / TFLOPS(M=32)
    // ---------------------------------------------------------------------------
    printf("\n=== Section 3: H-G1 — occupancy cliff (M=1 vs M=32) ===\n");
    // qkv: index 0 (M=1) vs 3 (M=32), mlp: index 4 (M=1) vs 7 (M=32)
    auto cliff = [&](int idx_m1, int idx_m32, const char* group) {
        double ratio = results[idx_m1].tflops / results[idx_m32].tflops;
        const char* verdict = (ratio < 0.10) ? "H-G1 ACCEPT" : "H-G1 REJECT";
        printf("  %-6s  TFLOPS(M=1)=%.3f  TFLOPS(M=32)=%.3f  ratio=%.3f  [%s]\n",
               group, results[idx_m1].tflops, results[idx_m32].tflops, ratio, verdict);
    };
    cliff(0, 3, "qkv");
    cliff(4, 7, "mlp");

    // ---------------------------------------------------------------------------
    // Section 4: H-G2 — linear scaling check
    // ---------------------------------------------------------------------------
    printf("\n=== Section 4: H-G2 — TFLOPS scaling with M ===\n");
    printf("%-12s  %10s  %10s  %10s  %10s  %12s\n",
           "group", "M=1", "M=8", "M=16", "M=32", "verdict");
    printf("%-12s  %10s  %10s  %10s  %10s  %12s\n",
           "------------", "----------", "----------", "----------", "----------", "------------");

    auto scaling = [&](int base, const char* group) {
        // base: index of M=1, base+1: M=8, base+2: M=16, base+3: M=32
        double tf1  = results[base].tflops;
        double tf8  = results[base+1].tflops;
        double tf16 = results[base+2].tflops;
        double tf32 = results[base+3].tflops;
        // H-G2: tf16 > 0.8*tf32 AND tf8 > 0.4*tf32
        bool h_g2 = (tf16 > 0.8*tf32) && (tf8 > 0.4*tf32);
        printf("%-12s  %10.3f  %10.3f  %10.3f  %10.3f  %12s\n",
               group, tf1, tf8, tf16, tf32,
               h_g2 ? "H-G2 ACCEPT" : "H-G2 REJECT");
    };
    scaling(0, "qkv");
    scaling(4, "mlp");

    // ---------------------------------------------------------------------------
    // Copy-paste block for results file
    // ---------------------------------------------------------------------------
    printf("\n--- paste into results file ---\n");
    printf("# GEMM batch=1 occupancy — Qwen2.5-0.5B shapes, MI300X\n");
    for (int i = 0; i < kNShapes; ++i) {
        printf("  %-12s M=%2d N=%4d K=%3d: %8.1f us  %8.3f TFLOPS  %8.1f GB/s\n",
               kShapes[i].name, kShapes[i].M, kShapes[i].N, kShapes[i].K,
               results[i].us, results[i].tflops, results[i].gbs);
    }

    HIPCHECK(hipFree(dA));
    HIPCHECK(hipFree(dW));
    HIPCHECK(hipFree(dC));
    RBCHECK(rocblas_destroy_handle(h));
    return 0;
}

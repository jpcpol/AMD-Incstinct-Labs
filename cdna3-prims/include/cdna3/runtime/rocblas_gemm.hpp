// ---------------------------------------------------------------------------
// cdna3/runtime/rocblas_gemm.hpp — rocBLAS GEMM wrappers for the layer loop.
//
// Replaces the naive tile-GEMM kernels in session.hpp with rocBLAS calls.
// All GEMMs in the layer loop are mixed-precision:
//   activations = float (fp32-acc, C4 hardening)
//   weights     = __half (fp16, loaded from model)
//   accumulation= float (rocblas_compute_type_f32)
//
// Row-major ↔ col-major identity:
//   C[M,N] = A[M,K] × B[K,N]   (row-major)
//   Cᵀ[N,M] = Bᵀ[N,K] × Aᵀ[K,M]  (col-major, what rocBLAS sees)
// So we call: gemm(B, A) with m=N, n=M, k=K, leading dims lda=N, ldb=K, ldc=N.
//
// Bias add after GEMM: rocBLAS gemm_ex does not support fused bias.
// We add a separate elementwise kernel for QKV bias (only those 3 projections
// have bias in Qwen2).
//
// Usage:
//   RocblasHandle rblas;
//   // C[M,N] = A[M,K] * W[N,K]^T  (W stored row-major as [N,K])
//   rblas.sgemm_f32_f16w(A, W, C, M, N, K);
// ---------------------------------------------------------------------------
#pragma once
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace cdna3 {
namespace runtime {

// ---------------------------------------------------------------------------
// Bias add kernel: Y[m, n] += B[n]  (broadcast bias over M rows)
// ---------------------------------------------------------------------------
__global__ void bias_add_f16_kernel(float* Y, const __half* B, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) Y[idx] += __half2float(B[idx % N]);
}

// Same but output is __half (used for QKV: float-acc GEMM then cast+bias)
__global__ void bias_add_store_f16_kernel(__half* Y, const __half* B, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N)
        Y[idx] = __float2half(__half2float(Y[idx]) + __half2float(B[idx % N]));
}

// ---------------------------------------------------------------------------
// RocblasHandle — RAII wrapper, one per Session.
// ---------------------------------------------------------------------------
class RocblasHandle {
    rocblas_handle h_ = nullptr;

    static void check(rocblas_status s, const char* where){
        if (s != rocblas_status_success){
            printf("rocBLAS error %d at %s\n", (int)s, where);
            abort();
        }
    }

public:
    RocblasHandle(){
        check(rocblas_create_handle(&h_), "create_handle");
    }
    ~RocblasHandle(){
        if (h_) rocblas_destroy_handle(h_);
    }
    RocblasHandle(const RocblasHandle&)            = delete;
    RocblasHandle& operator=(const RocblasHandle&) = delete;

    rocblas_handle get() const { return h_; }

    // -------------------------------------------------------------------------
    // gemm_f32_f16w_f32out
    //
    // Computes C[M, N] = A[M, K] × W[N, K]ᵀ  →  float output.
    //
    // A : float   [M, K] row-major
    // W : __half  [N, K] row-major  (weight matrix, transposed in GEMM)
    // C : float   [M, N] row-major
    //
    // rocBLAS sees col-major:  C'[N,M] = W'[N,K] × A'[K,M]
    //   transA=N (W already N×K), transB=N (A as K×M = Aᵀ col-major)
    //   Wait — re-derive carefully:
    //     Row-major C[M,N]=A[M,K]*W^T[K,N]  ↔  col-major C^T[N,M]=W[N,K]*A^T[K,M]
    //   So: op(W)=no-trans (m=N, k=K), op(A^T)=no-trans (k=K, n=M)
    //   rocBLAS: gemm(transA=N, transB=N, m=N, n=M, k=K, alpha, W, lda=N, A, ldb=K, beta, C, ldc=N)
    //
    // Compute type: f32 (accumulate in float even though B is f16).
    // -------------------------------------------------------------------------
    void gemm_f32_f16w_f32out(
        const float* A, const __half* W, float* C,
        int M, int N, int K,
        hipStream_t stream = nullptr)
    {
        if (stream) rocblas_set_stream(h_, stream);

        float alpha = 1.f, beta = 0.f;
        check(rocblas_gemm_ex(
            h_,
            rocblas_operation_none,      // transA: W is [N,K], use as-is
            rocblas_operation_none,      // transB: A^T is [K,M], use as-is
            N, M, K,                     // m, n, k
            &alpha,
            W,    rocblas_datatype_f16_r, N,   // a, a_type, lda
            A,    rocblas_datatype_f32_r, K,   // b, b_type, ldb
            &beta,
            C,    rocblas_datatype_f32_r, N,   // c, c_type, ldc
            C,    rocblas_datatype_f32_r, N,   // d, d_type, ldd
            rocblas_datatype_f32_r,  // compute type (f32 accumulation)
            rocblas_gemm_algo_standard, 0, 0
        ), "gemm_f32_f16w_f32out");
    }

    // -------------------------------------------------------------------------
    // gemm_f32_f16w_f16out
    //
    // Same GEMM but stores result as __half.
    // Used for QKV projections (output is fp16 for attention).
    // Bias (if any) must be added separately after this call.
    // -------------------------------------------------------------------------
    void gemm_f32_f16w_f16out(
        const float* A, const __half* W, __half* C,
        int M, int N, int K,
        hipStream_t stream = nullptr)
    {
        if (stream) rocblas_set_stream(h_, stream);

        float alpha = 1.f, beta = 0.f;
        check(rocblas_gemm_ex(
            h_,
            rocblas_operation_none,
            rocblas_operation_none,
            N, M, K,
            &alpha,
            W,    rocblas_datatype_f16_r, N,
            A,    rocblas_datatype_f32_r, K,
            &beta,
            C,    rocblas_datatype_f16_r, N,
            C,    rocblas_datatype_f16_r, N,
            rocblas_datatype_f32_r,  // compute type (f32 accumulation)
            rocblas_gemm_algo_standard, 0, 0
        ), "gemm_f32_f16w_f16out");
    }

    // -------------------------------------------------------------------------
    // gemm_f16_f16w_f32out
    //
    // C[M,N] = A[M,K] × W[N,K]ᵀ  with A in fp16, output in float.
    // Used for O-projection: attn output (fp16) × Wo (fp16) → float residual.
    // -------------------------------------------------------------------------
    void gemm_f16_f16w_f32out(
        const __half* A, const __half* W, float* C,
        int M, int N, int K,
        hipStream_t stream = nullptr)
    {
        if (stream) rocblas_set_stream(h_, stream);

        float alpha = 1.f, beta = 0.f;
        check(rocblas_gemm_ex(
            h_,
            rocblas_operation_none,
            rocblas_operation_none,
            N, M, K,
            &alpha,
            W,    rocblas_datatype_f16_r, N,
            A,    rocblas_datatype_f16_r, K,
            &beta,
            C,    rocblas_datatype_f32_r, N,
            C,    rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r,  // compute type (f32 accumulation)
            rocblas_gemm_algo_standard, 0, 0
        ), "gemm_f16_f16w_f32out");
    }
};

} // namespace runtime
} // namespace cdna3

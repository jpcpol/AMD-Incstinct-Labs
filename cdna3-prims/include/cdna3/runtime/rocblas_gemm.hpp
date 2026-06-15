// ---------------------------------------------------------------------------
// cdna3/runtime/rocblas_gemm.hpp — rocBLAS GEMM wrappers for the layer loop.
//
// All 7 GEMMs per layer use rocBLAS gemm_ex. The supported mixed-precision
// combination in rocBLAS 5.2 (ROCm 7.2) is:
//   a_type = b_type = f16_r,  c_type = d_type = f32_r,  compute_type = f32_r
//
// Our activations are float (fp32-acc hardening, C4). We convert them to fp16
// via a cast kernel before each GEMM call. The conversion is cheap (~1% of
// GEMM time) and preserves the fp32 accumulation path within the GEMM itself.
//
// Row-major → col-major identity:
//   Want:  C[M,N] = A[M,K] × W[N,K]ᵀ   (row-major)
//   rocBLAS sees row-major[R,C] as col-major[C,R] (the transpose).
//   Both A and W are row-major in memory; to get A×W^T we apply transA=T to
//   undo the implicit transpose on A, leaving transB=N for W (which rocBLAS reads
//   as W^T col-major = W row-major, exactly what we want to transpose).
//   Call:  gemm(transA=T, transB=N, m=N, n=M, k=K,
//               W, lda=K,  A, ldb=K,  C, ldc=N)
//   Verified empirically: test_gemm4 passes for M=2,N=3,K=4.
//
// Bias: rocBLAS gemm_ex doesn't support fused bias. Added as a separate kernel.
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
// cast_f32_to_f16_kernel: tmp_f16[i] = __float2half(src_f32[i])
// Used to down-cast float activations to fp16 before rocBLAS GEMM.
// ---------------------------------------------------------------------------
__global__ void cast_f32_to_f16_kernel(const float* src, __half* dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// ---------------------------------------------------------------------------
// bias_add_f32_kernel: Y[m,n] += B[n]  (broadcast fp16 bias into float Y)
// Used after MLP GEMMs (output is float, bias is fp16).
// ---------------------------------------------------------------------------
__global__ void bias_add_f32_kernel(float* Y, const __half* B, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) Y[idx] += __half2float(B[idx % N]);
}

// ---------------------------------------------------------------------------
// bias_add_f16_kernel: Y[m,n] += B[n]  (fp16 in/out, fp16 bias)
// Used after QKV GEMMs (output is fp16).
// ---------------------------------------------------------------------------
__global__ void bias_add_f16_kernel(__half* Y, const __half* B, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N)
        Y[idx] = __float2half(__half2float(Y[idx]) + __half2float(B[idx % N]));
}

// ---------------------------------------------------------------------------
// RocblasHandle — RAII wrapper. One per Session.
// Holds a temporary fp16 scratch buffer for activation down-casting.
// ---------------------------------------------------------------------------
class RocblasHandle {
    rocblas_handle h_   = nullptr;
    __half*        tmp_ = nullptr;   // device scratch for fp16 activations
    int            cap_ = 0;         // capacity in elements

    static void check(rocblas_status s, const char* where){
        if (s != rocblas_status_success){
            fprintf(stderr, "rocBLAS error %d at %s\n", (int)s, where);
            fflush(stderr);
            abort();
        }
    }

    // Ensure tmp_ can hold n fp16 elements.
    void ensure_tmp(int n){
        if (n <= cap_) return;
        if (tmp_) hipFree(tmp_);
        hipMalloc(&tmp_, (size_t)n * 2);
        cap_ = n;
    }

public:
    RocblasHandle(){
        check(rocblas_create_handle(&h_), "create_handle");
    }
    ~RocblasHandle(){
        if (h_)   rocblas_destroy_handle(h_);
        if (tmp_) hipFree(tmp_);
    }
    RocblasHandle(const RocblasHandle&)            = delete;
    RocblasHandle& operator=(const RocblasHandle&) = delete;

    rocblas_handle get() const { return h_; }

    // -------------------------------------------------------------------------
    // gemm_f32act_f16w_f32out
    //
    // C[M,N] = A[M,K] × W[N,K]ᵀ    A=float activations, W=fp16 weights, C=float
    //
    // Steps: (1) cast A→fp16 scratch, (2) rocBLAS gemm_ex f16+f16→f32.
    // rocBLAS call (see header comment for derivation):
    //   gemm(transA=T, transB=N, m=N, n=M, k=K, W, lda=K, A_f16, ldb=K, C, ldc=N)
    // -------------------------------------------------------------------------
    void gemm_f32act_f16w_f32out(
        const float* A, const __half* W, float* C,
        int M, int N, int K,
        hipStream_t stream = nullptr)
    {
        if (stream) rocblas_set_stream(h_, stream);

        // Cast float A[M,K] → fp16 tmp[M,K]
        ensure_tmp(M * K);
        int elems_a = M * K;
        hipLaunchKernelGGL(cast_f32_to_f16_kernel,
            dim3((elems_a + 255) / 256), dim3(256), 0, stream,
            A, tmp_, elems_a);

        float alpha = 1.f, beta = 0.f;
        check(rocblas_gemm_ex(
            h_,
            rocblas_operation_transpose,  // op(W[N,K] row-major) = W^T ... gives W
            rocblas_operation_none,       // op(A[M,K] row-major) = A^T ... leaves W×A^T
            N, M, K,                      // m, n, k
            &alpha,
            W,    rocblas_datatype_f16_r, K,   // W[N,K] row-major, lda=K
            tmp_, rocblas_datatype_f16_r, K,   // A[M,K] row-major, ldb=K
            &beta,
            C,    rocblas_datatype_f32_r, N,   // C^T[N,M] col-major → read as C[M,N] row-major
            C,    rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0
        ), "gemm_f32act_f16w_f32out");
    }

    // -------------------------------------------------------------------------
    // gemm_f32act_f16w_f16out
    //
    // C[M,N] = A[M,K] × W[N,K]ᵀ    A=float, W=fp16, C=fp16.
    // Used for QKV projections (attention expects fp16 input).
    // Same rocBLAS call as f32out variant, output type changed to f16.
    // -------------------------------------------------------------------------
    void gemm_f32act_f16w_f16out(
        const float* A, const __half* W, __half* C,
        int M, int N, int K,
        hipStream_t stream = nullptr)
    {
        if (stream) rocblas_set_stream(h_, stream);

        ensure_tmp(M * K);
        int elems_a = M * K;
        hipLaunchKernelGGL(cast_f32_to_f16_kernel,
            dim3((elems_a + 255) / 256), dim3(256), 0, stream,
            A, tmp_, elems_a);

        float alpha = 1.f, beta = 0.f;
        check(rocblas_gemm_ex(
            h_,
            rocblas_operation_transpose,
            rocblas_operation_none,
            N, M, K,
            &alpha,
            W,    rocblas_datatype_f16_r, K,
            tmp_, rocblas_datatype_f16_r, K,
            &beta,
            C,    rocblas_datatype_f16_r, N,
            C,    rocblas_datatype_f16_r, N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0
        ), "gemm_f32act_f16w_f16out");
    }

    // -------------------------------------------------------------------------
    // gemm_f16act_f16w_f32out
    //
    // C[M,N] = A[M,K] × W[N,K]ᵀ    A=fp16, W=fp16, C=float.
    // Used for O-projection: fp16 attn output → float residual.
    // No cast needed (A is already fp16).
    // -------------------------------------------------------------------------
    void gemm_f16act_f16w_f32out(
        const __half* A, const __half* W, float* C,
        int M, int N, int K,
        hipStream_t stream = nullptr)
    {
        if (stream) rocblas_set_stream(h_, stream);

        float alpha = 1.f, beta = 0.f;
        check(rocblas_gemm_ex(
            h_,
            rocblas_operation_transpose,
            rocblas_operation_none,
            N, M, K,
            &alpha,
            W,    rocblas_datatype_f16_r, K,
            A,    rocblas_datatype_f16_r, K,
            &beta,
            C,    rocblas_datatype_f32_r, N,
            C,    rocblas_datatype_f32_r, N,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0
        ), "gemm_f16act_f16w_f32out");
    }
};

} // namespace runtime
} // namespace cdna3

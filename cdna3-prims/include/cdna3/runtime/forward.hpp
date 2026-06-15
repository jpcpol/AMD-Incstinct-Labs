// ---------------------------------------------------------------------------
// cdna3/runtime/forward.hpp — supporting kernels for the Stage-C layer loop.
//
// These are correctness-first reference kernels (RMSNorm, GEMM, RoPE, SiLU-MLP,
// embedding gather, argmax). The RESEARCH artifact is the attention path
// (cdna3::attn::*); these surrounding ops use simple, numerically-faithful
// kernels — the GEMM and RoPE kernels are lifted verbatim from the 2-B layer
// validation (validate_attn_layer.hip), which matched HF to rel err < 0.05.
//
// All tensors fp16 on-device unless noted. Row-major.
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

namespace cdna3 {
namespace runtime {

// ---------------------------------------------------------------------------
// RMSNorm: y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]   (per row of [M, H])
// One block per row; block reduces sum of squares. H up to a few thousand.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                               __half* __restrict__ Y, int M, int H, float eps){
    int row = blockIdx.x;
    if (row >= M) return;
    const __half* x = X + (size_t)row*H;
    __half* y = Y + (size_t)row*H;

    __shared__ float red[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x){
        float v = __half2float(x[i]); local += v*v;
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x+s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0]/H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x){
        y[i] = __float2half(__half2float(x[i]) * inv * __half2float(W[i]));
    }
}

// ---------------------------------------------------------------------------
// GEMM: Y[M,N] = X[M,K] · Wᵀ where W is [N,K] (HF nn.Linear weight layout).
// Correctness-only (one thread per output). Lifted from 2-B validation.
// ---------------------------------------------------------------------------
__global__ void gemm_xwT_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                                __half* __restrict__ Y, int M, int N, int K){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(X[(size_t)row*K+k]) * __half2float(W[(size_t)col*K+k]);
    Y[(size_t)row*N+col] = __float2half(acc);
}

// Same as gemm_xwT_kernel but adds a per-output-column bias B[N] (Qwen2 QKV).
// If B==nullptr behaves like gemm_xwT_kernel.
__global__ void gemm_xwT_bias_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                                     const __half* __restrict__ B,
                                     __half* __restrict__ Y, int M, int N, int K){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = B ? __half2float(B[col]) : 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(X[(size_t)row*K+k]) * __half2float(W[(size_t)col*K+k]);
    Y[(size_t)row*N+col] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// RoPE in place on [seq, heads*D]. cos/sin are [seq, D] (precomputed per pos).
// rotate-half convention (Llama/Qwen2). Lifted from 2-B validation.
// ---------------------------------------------------------------------------
__global__ void rope_kernel(__half* __restrict__ T, const float* __restrict__ cos,
                            const float* __restrict__ sin, int seq, int heads, int D){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = seq*heads*D;
    if (idx >= total) return;
    int d = idx % D;
    int s = idx / (heads*D);
    int base = idx - d;
    float c = cos[s*D + d], sn = sin[s*D + d];
    float t = __half2float(T[base + d]);
    float tr = (d < D/2) ? -__half2float(T[base + d + D/2])
                         :  __half2float(T[base + d - D/2]);
    T[base + d] = __float2half(t*c + tr*sn);
}

// ---------------------------------------------------------------------------
// SiLU-gated MLP fusion: out[i] = silu(gate[i]) * up[i]
//   silu(x) = x * sigmoid(x)
// gate and up are [M, F]; writes [M, F] in place into `gate`.
// ---------------------------------------------------------------------------
__global__ void silu_mul_kernel(__half* __restrict__ gate, const __half* __restrict__ up,
                                size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float s = g / (1.f + expf(-g));
    gate[i] = __float2half(s * __half2float(up[i]));
}

// ---------------------------------------------------------------------------
// Residual add: x[i] += y[i]
// ---------------------------------------------------------------------------
__global__ void add_kernel(__half* __restrict__ x, const __half* __restrict__ y, size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half(__half2float(x[i]) + __half2float(y[i]));
}

// ---------------------------------------------------------------------------
// Embedding gather: out[t, :] = embed_tokens[ ids[t], : ]   (ids host-side)
// One block per token row.
// ---------------------------------------------------------------------------
__global__ void embed_gather_kernel(const __half* __restrict__ embed,
                                    const int* __restrict__ ids,
                                    __half* __restrict__ out, int T, int H){
    int t = blockIdx.x;
    if (t >= T) return;
    int id = ids[t];
    const __half* src = embed + (size_t)id*H;
    __half* dst = out + (size_t)t*H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Repack [seq, heads, D] (token-major) <-> [heads, seq, D] (head-major).
// The cdna3::attn kernels expect head-major (batch×head×seq×D contiguous).
// dir=0: token-major -> head-major; dir=1: head-major -> token-major.
// ---------------------------------------------------------------------------
__global__ void repack_kernel(const __half* __restrict__ src, __half* __restrict__ dst,
                              int seq, int heads, int D, int dir){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = seq*heads*D;
    if (idx >= total) return;
    int d = idx % D;
    int h = (idx / D) % heads;
    int s = idx / (heads*D);
    size_t tok_major  = ((size_t)s*heads + h)*D + d;
    size_t head_major = ((size_t)h*seq + s)*D + d;
    if (dir == 0) dst[head_major] = src[tok_major];
    else          dst[tok_major]  = src[head_major];
}

// ---------------------------------------------------------------------------
// Scatter token-major K/V [T, heads, D] into a head-major cache slot
// [heads, max_seq, D] starting at token offset pos0.
//   cache[h, pos0+s, d] = src[s, h, d]
// ---------------------------------------------------------------------------
__global__ void scatter_kv_kernel(const __half* __restrict__ src, __half* __restrict__ cache,
                                  int T, int heads, int D, int pos0, int max_seq){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = T*heads*D;
    if (idx >= total) return;
    int d = idx % D;
    int h = (idx / D) % heads;
    int s = idx / (heads*D);
    size_t tok_major  = ((size_t)s*heads + h)*D + d;
    size_t cache_pos  = ((size_t)h*max_seq + (pos0 + s))*D + d;
    cache[cache_pos] = src[tok_major];
}

// ---------------------------------------------------------------------------
// Gather the first `N` tokens of each head from a head-major cache slot
// [heads, max_seq, D] into a contiguous packed buffer [heads, N, D].
// This bridges the cache stride (max_seq) to what cdna3::attn expects (N).
//   packed[h, n, d] = cache[h, n, d]   (n in [0, N))
// ---------------------------------------------------------------------------
__global__ void pack_kv_kernel(const __half* __restrict__ cache, __half* __restrict__ packed,
                               int heads, int N, int D, int max_seq){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = heads*N*D;
    if (idx >= total) return;
    int d = idx % D;
    int n = (idx / D) % N;
    int h = idx / (N*D);
    size_t cache_pos  = ((size_t)h*max_seq + n)*D + d;
    size_t packed_pos = ((size_t)h*N + n)*D + d;
    packed[packed_pos] = cache[cache_pos];
}

// ---------------------------------------------------------------------------
// Argmax over a [V] logit row → single int (greedy next token). One block.
// ---------------------------------------------------------------------------
__global__ void argmax_kernel(const __half* __restrict__ logits, int V, int* __restrict__ out){
    __shared__ float bestv[256];
    __shared__ int   besti[256];
    float bv = -1e30f; int bi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x){
        float v = __half2float(logits[i]);
        if (v > bv){ bv = v; bi = i; }
    }
    bestv[threadIdx.x] = bv; besti[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (threadIdx.x < s && bestv[threadIdx.x+s] > bestv[threadIdx.x]){
            bestv[threadIdx.x] = bestv[threadIdx.x+s];
            besti[threadIdx.x] = besti[threadIdx.x+s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = besti[0];
}

} // namespace runtime
} // namespace cdna3

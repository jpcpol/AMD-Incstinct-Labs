// ---------------------------------------------------------------------------
// cdna3/runtime/forward.hpp — supporting kernels for the Stage-C layer loop.
//
// These are correctness-first reference kernels (RMSNorm, GEMM, RoPE, SiLU-MLP,
// embedding gather, argmax). The RESEARCH artifact is the attention path
// (cdna3::attn::*); these surrounding ops use simple, numerically-faithful
// kernels — the GEMM and RoPE kernels are lifted verbatim from the 2-B layer
// validation (validate_attn_layer.hip), which matched HF to rel err < 0.05.
//
// fp32-acc hardening (C4): d_hidden and d_norm are float* to avoid fp16 drift
// accumulating over 24 layers of residual adds. Weights remain fp16; only the
// hidden state accumulator is promoted. H3/T-030: fp32-acc is faster AND more
// precise on CDNA3 (v_add_f32 < __hadd on gfx942).
//
// Tensor layouts:
//   d_hidden, d_norm: float  [T, H]      ← promoted
//   d_q, d_k, d_v, d_attn_tok: __half   ← attention still fp16
//   d_gate, d_up, d_mlp: float           ← MLP activations promoted
//   weights (Wq, Wk, ...): __half        ← unchanged
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
// fp32-acc variant: X is float (hidden state), W is __half (weights), Y is float.
// ---------------------------------------------------------------------------
__global__ void rmsnorm_f32_kernel(const float* __restrict__ X, const __half* __restrict__ W,
                                   float* __restrict__ Y, int M, int H, float eps){
    int row = blockIdx.x;
    if (row >= M) return;
    const float* x = X + (size_t)row*H;
    float* y = Y + (size_t)row*H;

    __shared__ float red[256];
    float local = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x){
        local += x[i]*x[i];
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x+s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0]/H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x){
        y[i] = x[i] * inv * __half2float(W[i]);
    }
}

// ---------------------------------------------------------------------------
// GEMM variants. Row-major. W is always __half [N,K] (HF nn.Linear layout).
//
// gemm_f32in_f16out: X float [M,K] → Y __half [M,N]  (hidden→QKV/O projections)
// gemm_f32in_f32out: X float [M,K] → Y float  [M,N]  (norm→gate/up/down in MLP)
// gemm_f16in_f32out: X __half[M,K] → Y float  [M,N]  (attn_out → O proj result)
// gemm_f16in_f16out: X __half[M,K] → Y __half [M,N]  (lm_head, diag)
//
// _bias variants add per-column bias B[N] (__half). B==nullptr → no bias.
// ---------------------------------------------------------------------------

// X float → Y fp16 (hidden state → QKV projections)
__global__ void gemm_f32in_f16out_kernel(const float* __restrict__ X, const __half* __restrict__ W,
                                         __half* __restrict__ Y, int M, int N, int K){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += X[(size_t)row*K+k] * __half2float(W[(size_t)col*K+k]);
    Y[(size_t)row*N+col] = __float2half(acc);
}

// X float → Y fp16, +bias (Qwen2 QKV bias)
__global__ void gemm_f32in_f16out_bias_kernel(const float* __restrict__ X, const __half* __restrict__ W,
                                              const __half* __restrict__ B,
                                              __half* __restrict__ Y, int M, int N, int K){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = B ? __half2float(B[col]) : 0.f;
    for (int k = 0; k < K; ++k)
        acc += X[(size_t)row*K+k] * __half2float(W[(size_t)col*K+k]);
    Y[(size_t)row*N+col] = __float2half(acc);
}

// X float → Y float (hidden/norm → MLP gate/up/down)
__global__ void gemm_f32in_f32out_kernel(const float* __restrict__ X, const __half* __restrict__ W,
                                         float* __restrict__ Y, int M, int N, int K){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += X[(size_t)row*K+k] * __half2float(W[(size_t)col*K+k]);
    Y[(size_t)row*N+col] = acc;
}

// X fp16 → Y float (attn_out [fp16] → O-proj result, written back into d_hidden float)
__global__ void gemm_f16in_f32out_kernel(const __half* __restrict__ X, const __half* __restrict__ W,
                                         float* __restrict__ Y, int M, int N, int K){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += __half2float(X[(size_t)row*K+k]) * __half2float(W[(size_t)col*K+k]);
    Y[(size_t)row*N+col] = acc;
}

// X fp16 → Y fp16 (kept for diag_logits which passes fp16 hidden directly)
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
// gemm_f32in_f16out_kernel also serves as lm_head kernel (same body, reused by greedy_from_last)

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
// SiLU-gated MLP fusion (fp32): out[i] = silu(gate[i]) * up[i]
// gate and up are float [M, F]; writes in place into `gate`.
// ---------------------------------------------------------------------------
__global__ void silu_mul_f32_kernel(float* __restrict__ gate, const float* __restrict__ up,
                                    size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    gate[i] = (g / (1.f + expf(-g))) * up[i];
}

// ---------------------------------------------------------------------------
// Residual add (fp32): x[i] += y[i]  (both float)
// ---------------------------------------------------------------------------
__global__ void add_f32_kernel(float* __restrict__ x, const float* __restrict__ y, size_t n){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// ---------------------------------------------------------------------------
// Embedding gather → float: out[t, :] = float(embed_tokens[ ids[t], : ])
// One block per token row. Promotes fp16 weights to float hidden state.
// ---------------------------------------------------------------------------
__global__ void embed_gather_f32_kernel(const __half* __restrict__ embed,
                                        const int* __restrict__ ids,
                                        float* __restrict__ out, int T, int H){
    int t = blockIdx.x;
    if (t >= T) return;
    int id = ids[t];
    const __half* src = embed + (size_t)id*H;
    float* dst = out + (size_t)t*H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = __half2float(src[i]);
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

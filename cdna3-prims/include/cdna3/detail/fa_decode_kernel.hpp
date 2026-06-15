// ---------------------------------------------------------------------------
// cdna3/detail/fa_decode_kernel.hpp — split-KV decode kernel (header-only).
//
// Contains ONLY __global__ template fa_decode_kernel<D,W> and its supporting
// types. No main(), no launcher, no benchmark harness. Single source of truth
// for the decode kernel symbol; the standalone binary (fa_decode.hip) adds the
// launcher and includes this.
//
// Consumed by cdna3/attention.hpp (the Stage-B unified API).
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../cdna3.hpp"   // wave::dpp::reduce_sum_bcast

using wave::dpp::reduce_sum_bcast;

struct DecodeKCfg { int D, N, nQHeads, nKVHeads; };

template<int D, int W>
__global__ void fa_decode_kernel(const __half* __restrict__ q, const __half* __restrict__ K,
                                 const __half* __restrict__ V, __half* __restrict__ o,
                                 DecodeKCfg c)
{
    constexpr int EPL = D / 64;
    const int wave = threadIdx.x / 64;
    const int L    = threadIdx.x % 64;
    const int qhead = blockIdx.x;
    const int group = c.nQHeads / c.nKVHeads;
    const int kvhead = qhead / group;

    const __half* qh = q + (size_t)qhead*D;
    const __half* Kh = K + (size_t)kvhead*c.N*D;
    const __half* Vh = V + (size_t)kvhead*c.N*D;
    __half*       oh = o + (size_t)qhead*D;

    float qreg[EPL];
    #pragma unroll
    for (int e=0;e<EPL;++e) qreg[e] = __half2float(qh[L*EPL+e]);

    const float scale = 1.f/sqrtf((float)D);
    int per = (c.N + W - 1) / W;
    int k0 = wave*per, k1 = min(k0+per, c.N);

    float m_w = -1e38f, l_w = 0.f;
    float O_w[EPL];
    #pragma unroll
    for (int e=0;e<EPL;++e) O_w[e]=0.f;

    for (int k=k0; k<k1; ++k){
        const __half* Kk = Kh + (size_t)k*D;
        float partial = 0.f;
        #pragma unroll
        for (int e=0;e<EPL;++e) partial += qreg[e] * __half2float(Kk[L*EPL+e]);
        float s = reduce_sum_bcast(partial) * scale;   // zero-LDS DPP, broadcast
        float mnew = fmaxf(m_w, s);
        float p    = __expf(s - mnew);
        float resc = __expf(m_w - mnew);
        const __half* Vk = Vh + (size_t)k*D;
        #pragma unroll
        for (int e=0;e<EPL;++e) O_w[e] = O_w[e]*resc + p*__half2float(Vk[L*EPL+e]);
        l_w = l_w*resc + p;
        m_w = mnew;
    }

    // Split-K combine across W waves via LDS (wave 0 merges).
    __shared__ float m_sh[W], l_sh[W];
    __shared__ float O_sh[W][D];
    if (L==0){ m_sh[wave]=m_w; l_sh[wave]=l_w; }
    #pragma unroll
    for (int e=0;e<EPL;++e) O_sh[wave][L*EPL+e] = O_w[e];
    __syncthreads();

    if (wave==0){
        float m = -1e38f;
        #pragma unroll
        for (int w=0; w<W; ++w) m = fmaxf(m, m_sh[w]);
        float l = 0.f;
        float acc[EPL];
        #pragma unroll
        for (int e=0;e<EPL;++e) acc[e]=0.f;
        #pragma unroll
        for (int w=0; w<W; ++w){
            float f = __expf(m_sh[w] - m);
            l += f * l_sh[w];
            #pragma unroll
            for (int e=0;e<EPL;++e) acc[e] += f * O_sh[w][L*EPL+e];
        }
        float inv = (l>0.f) ? 1.f/l : 1.f;
        #pragma unroll
        for (int e=0;e<EPL;++e) oh[L*EPL+e] = __float2half(acc[e]*inv);
    }
}

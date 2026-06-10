// ---------------------------------------------------------------------------
// fa_decode_kernel.hpp — split-KV decode FA kernel, extracted from fa_decode.hip
// so the standalone benchmark and the torch op share ONE kernel (single source of
// truth, same pattern as fa_attn_kernel.hpp for prefill).
//
// Layout contract:
//   q : [nQH, D]        (the single new token's query, per head)
//   K : [nKVH, N, D]    (KV-cache, N = current length)
//   V : [nKVH, N, D]
//   o : [nQH, D]
//   grid.x = nQH, block = W*64. D in {64,128}. GQA: nQH % nKVH == 0.
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

namespace fa_decode {

struct DCfg { int D, N, nQHeads, nKVHeads; };

template<int CTRL> __device__ __forceinline__ float dpp_mov(float v, float old){
    return __builtin_amdgcn_update_dpp(old, v, CTRL, 0xf, 0xf, false);
}
__device__ __forceinline__ float wave_reduce_sum(float v){
    float t;
    t=dpp_mov<0x111>(v,0.f); v+=t;
    t=dpp_mov<0x112>(v,0.f); v+=t;
    t=dpp_mov<0x114>(v,0.f); v+=t;
    t=dpp_mov<0x118>(v,0.f); v+=t;
    t=dpp_mov<0x142>(v,0.f); v+=t;
    t=dpp_mov<0x143>(v,0.f); v+=t;
    return __builtin_amdgcn_readlane(v, 63);
}

template<int D, int W>
__global__ void fa_decode_kernel(const __half* __restrict__ q, const __half* __restrict__ K,
                                 const __half* __restrict__ V, __half* __restrict__ o,
                                 DCfg c)
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

    float m_w=-1e38f, l_w=0.f, O_w[EPL];
    #pragma unroll
    for (int e=0;e<EPL;++e) O_w[e]=0.f;

    for (int k=k0; k<k1; ++k){
        const __half* Kk = Kh + (size_t)k*D;
        float partial=0.f;
        #pragma unroll
        for (int e=0;e<EPL;++e) partial += qreg[e]*__half2float(Kk[L*EPL+e]);
        float s = wave_reduce_sum(partial) * scale;   // uniform across lanes
        float mnew=fmaxf(m_w,s), p=__expf(s-mnew), resc=__expf(m_w-mnew);
        const __half* Vk = Vh + (size_t)k*D;
        #pragma unroll
        for (int e=0;e<EPL;++e) O_w[e]=O_w[e]*resc + p*__half2float(Vk[L*EPL+e]);
        l_w=l_w*resc+p; m_w=mnew;
    }

    __shared__ float m_sh[W], l_sh[W];
    __shared__ float O_sh[W][D];
    if (L==0){ m_sh[wave]=m_w; l_sh[wave]=l_w; }
    #pragma unroll
    for (int e=0;e<EPL;++e) O_sh[wave][L*EPL+e]=O_w[e];
    __syncthreads();

    if (wave==0){
        float m=-1e38f;
        #pragma unroll
        for (int w=0;w<W;++w) m=fmaxf(m,m_sh[w]);
        float l=0.f, acc[EPL];
        #pragma unroll
        for (int e=0;e<EPL;++e) acc[e]=0.f;
        #pragma unroll
        for (int w=0;w<W;++w){ float f=__expf(m_sh[w]-m); l+=f*l_sh[w];
            #pragma unroll
            for (int e=0;e<EPL;++e) acc[e]+=f*O_sh[w][L*EPL+e]; }
        float inv=(l>0.f)?1.f/l:1.f;
        #pragma unroll
        for (int e=0;e<EPL;++e) oh[L*EPL+e]=__float2half(acc[e]*inv);
    }
}

// Host launcher. D in {64,128}, W the split-KV wave count.
inline hipError_t launch(const __half* q, const __half* K, const __half* V, __half* o,
                         DCfg c, int W, hipStream_t stream=0){
    dim3 grid(c.nQHeads), block(W*64);
    if      (c.D==64  && W==4) hipLaunchKernelGGL((fa_decode_kernel<64, 4>),grid,block,0,stream,q,K,V,o,c);
    else if (c.D==128 && W==4) hipLaunchKernelGGL((fa_decode_kernel<128,4>),grid,block,0,stream,q,K,V,o,c);
    else if (c.D==64  && W==2) hipLaunchKernelGGL((fa_decode_kernel<64, 2>),grid,block,0,stream,q,K,V,o,c);
    else if (c.D==128 && W==2) hipLaunchKernelGGL((fa_decode_kernel<128,2>),grid,block,0,stream,q,K,V,o,c);
    else return hipErrorInvalidValue;
    return hipGetLastError();
}

} // namespace fa_decode

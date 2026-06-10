// ---------------------------------------------------------------------------
// fa_attn_kernel.hpp — the verified 1-wave/block MFMA Flash-Attention kernel,
// extracted from validate_attn_layer.hip (Step 2-B, PASS) so BOTH the numerical
// validator (B) and the PyTorch custom op (A) include the SAME kernel.
//
// Single source of truth: this is the exact kernel that reproduced a real Qwen2.5
// layer within fp16 tolerance (4.34% of signal scale). Step 2-A must not fork it.
//
// Layout contract (head-major, GQA + causal):
//   Q : [nQH, seq, D]   (query heads major)
//   K : [nKVH, seq, D]  (kv heads major; GQA: nKVH <= nQH, divides it)
//   V : [nKVH, seq, D]
//   O : [nQH, seq, D]
//   grid.x = nQH * (seq / Br), block = 64 (one wave per (head, query-block))
//   D is a template parameter (64 or 128); seq must be a multiple of Br=16,
//   and the kv dimension a multiple of Bc=64. Causal masking is always on
//   (decoder LLMs); tiles fully above the diagonal are skipped.
//
// MFMA lane mapping (verified probe_mfma_mapping.hip):
//   A in : lane L,f -> A[row=L%16][k=(L/16)*4+f]
//   B in : lane L,f -> B[k=(L/16)*4+f][col=L%16]
//   C/out: lane L,f -> C[row=(L/16)*4+f][col=L%16]
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

namespace fa_attn {

static constexpr int Br = 16;
static constexpr int Bc = 64;
static constexpr int MT = 16;

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;
__device__ __forceinline__ __fp16 h2f16(__half h){ return *reinterpret_cast<__fp16*>(&h); }
__device__ __forceinline__ f32x4 mfma(fp16x4 a, fp16x4 b, f32x4 c){
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,c,0,0,0);
}

// Flash Attention, 1 wave/block, GQA + causal. Identical to validate_attn_layer.hip's
// fa_kernel — do not edit one without the other (or better: both include this header).
template<int D>
__global__ void fa_kernel(const __half* Q, const __half* K, const __half* V, __half* O,
                          int seq, int nQH, int nKVH)
{
    constexpr int nD=D/MT, nBc=Bc/MT;
    const int qblocks = seq/Br;
    const int qblk  = blockIdx.x % qblocks;
    const int qhead = blockIdx.x / qblocks;     // grid.x = nQH * qblocks
    const int L = threadIdx.x;
    const int group = nQH / nKVH;
    const int kvhead = qhead / group;

    const __half* Qh = Q + ((size_t)qhead*seq + qblk*Br)*D;
    const __half* Kh = K + (size_t)kvhead*seq*D;
    const __half* Vh = V + (size_t)kvhead*seq*D;
    __half*       Oh = O + ((size_t)qhead*seq + qblk*Br)*D;

    __shared__ __half Q_lds[Br*D];
    __shared__ __half K_lds[Bc*D];
    __shared__ __half V_lds[Bc*D];
    __shared__ float  S_lds[Br*Bc];
    __shared__ __half P_lds[Br*Bc];
    __shared__ float  O_lds[Br*D];

    for(int i=L;i<Br*D;i+=64) Q_lds[i]=Qh[i];
    for(int i=L;i<Br*D;i+=64) O_lds[i]=0.f;
    float m_run=-1e38f, l_run=0.f;
    const float scale=1.f/sqrtf((float)D);
    const int last_q = qblk*Br + Br - 1;
    int kv_tiles = (last_q/Bc)+1;            // causal: skip tiles fully above diagonal
    if (kv_tiles > seq/Bc) kv_tiles = seq/Bc;
    __syncthreads();

    for(int kv=0;kv<kv_tiles;++kv){
        for(int i=L;i<Bc*D;i+=64){ K_lds[i]=Kh[(size_t)kv*Bc*D+i]; V_lds[i]=Vh[(size_t)kv*Bc*D+i]; }
        __syncthreads();
        #pragma unroll
        for(int jc=0;jc<nBc;++jc){
            f32x4 acc={0,0,0,0};
            #pragma unroll
            for(int kd=0;kd<nD;++kd){
                fp16x4 a,b; int q_row=L%MT, d_idx=kd*MT+(L/MT)*4;
                #pragma unroll
                for(int f=0;f<4;++f) a[f]=h2f16(Q_lds[q_row*D+d_idx+f]);
                int key=jc*MT+(L%MT);
                #pragma unroll
                for(int f=0;f<4;++f){ int dd=kd*MT+(L/MT)*4+f; b[f]=h2f16(K_lds[key*D+dd]); }
                acc=mfma(a,b,acc);
            }
            #pragma unroll
            for(int f=0;f<4;++f){
                int sr=(L/MT)*4+f, sc=jc*MT+(L%MT);
                float s=acc[f]*scale;
                int gq=qblk*Br+sr, gk=kv*Bc+sc;
                if (gk>gq) s=-1e30f;
                S_lds[sr*Bc+sc]=s;
            }
        }
        __syncthreads();
        if(L<Br){
            int row=L; float rmax=-1e38f;
            #pragma unroll
            for(int cc=0;cc<Bc;++cc) rmax=fmaxf(rmax,S_lds[row*Bc+cc]);
            float mnew=fmaxf(m_run,rmax), rsum=0.f;
            #pragma unroll
            for(int cc=0;cc<Bc;++cc){ float p=__expf(S_lds[row*Bc+cc]-mnew); P_lds[row*Bc+cc]=__float2half(p); rsum+=p; }
            float resc=__expf(m_run-mnew);
            for(int d=0;d<D;++d) O_lds[row*D+d]*=resc;
            m_run=mnew; l_run=resc*l_run+rsum;
        }
        __syncthreads();
        #pragma unroll
        for(int jd=0;jd<nD;++jd){
            f32x4 acc={0,0,0,0};
            #pragma unroll
            for(int kb=0;kb<nBc;++kb){
                fp16x4 a,b; int p_row=L%MT, bc_idx=kb*MT+(L/MT)*4;
                #pragma unroll
                for(int f=0;f<4;++f) a[f]=h2f16(P_lds[p_row*Bc+bc_idx+f]);
                int d_col=jd*MT+(L%MT);
                #pragma unroll
                for(int f=0;f<4;++f){ int kk=kb*MT+(L/MT)*4+f; b[f]=h2f16(V_lds[kk*D+d_col]); }
                acc=mfma(a,b,acc);
            }
            #pragma unroll
            for(int f=0;f<4;++f){ int orr=(L/MT)*4+f, oc=jd*MT+(L%MT); O_lds[orr*D+oc]+=acc[f]; }
        }
        __syncthreads();
    }
    __shared__ float ls[Br];
    if(L<Br) ls[L]=(l_run>0.f)?l_run:1.f;
    __syncthreads();
    for(int i=L;i<Br*D;i+=64){ int row=i/D; Oh[i]=__float2half(O_lds[i]/ls[row]); }
}

// Host launcher: dispatches the compiled template on D (64/128). Caller owns the
// device buffers in the head-major layout described above.
inline hipError_t launch(const __half* Q, const __half* K, const __half* V, __half* O,
                         int seq, int nQH, int nKVH, int D, hipStream_t stream=0){
    int blocks = nQH * (seq / Br);
    if (D==64)  hipLaunchKernelGGL(fa_kernel<64>,  dim3(blocks), dim3(64), 0, stream, Q,K,V,O,seq,nQH,nKVH);
    else if (D==128) hipLaunchKernelGGL(fa_kernel<128>, dim3(blocks), dim3(64), 0, stream, Q,K,V,O,seq,nQH,nKVH);
    else return hipErrorInvalidValue;  // only D=64/128 instantiated
    return hipGetLastError();
}

} // namespace fa_attn

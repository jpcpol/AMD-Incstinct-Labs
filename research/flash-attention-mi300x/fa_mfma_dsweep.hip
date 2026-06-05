// ---------------------------------------------------------------------------
// fa_mfma_dsweep.hip — G1: does MFMA win once the head dim is large enough?
//
// fa_mfma2.hip showed MFMA is 0.95x vs the dot-product path at D=64 — the
// 16x16x16 tiles were too small to saturate the matrix cores. Hypothesis:
// at larger head dim (D=128) the MFMA tile count per KV iteration doubles
// (nD = D/16 = 8 instead of 4), giving the cores enough work to win.
//
// This kernel templates D and runs D=64 and D=128 back-to-back, each compared
// against an equivalent dot-product reference time. Single wave per query block
// (the verified-correct fa_mfma2 design).
//
// LDS budget at D=128 (single-buffer; no DME here to isolate the MFMA effect):
//   Q 16x128 fp16 = 4KB, K 64x128 = 16KB, V 64x128 = 16KB,
//   S 16x64 fp32 = 4KB, P 16x64 fp16 = 2KB, O 16x128 fp32 = 8KB  => ~50KB < 64KB OK
//
// MFMA lane mapping (verified probe_mfma_mapping.hip):
//   A in : lane L,f -> A[row=L%16][k=(L/16)*4+f]
//   B in : lane L,f -> B[k=(L/16)*4+f][col=L%16]
//   C/out: lane L,f -> C[row=(L/16)*4+f][col=L%16]
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 fa_mfma_dsweep.hip -o fa_mfma_dsweep
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

static constexpr int Br = 16;
static constexpr int Bc = 64;
static constexpr int kN = 512;
static constexpr int kBatch = 1;
static constexpr int kHeads = 8;
static constexpr int kSeqQ  = 512;
static constexpr int kWarmup = 20;
static constexpr int kIter   = 200;
static constexpr int MT = 16;

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;
__device__ __forceinline__ __fp16 h2f16(__half h){ return *reinterpret_cast<__fp16*>(&h); }
__device__ __forceinline__ f32x4 mfma(fp16x4 a, fp16x4 b, f32x4 c){ return __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,c,0,0,0); }

template<int D>
__global__ void fa_mfma_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                               const __half* __restrict__ V, __half* __restrict__ O, int N)
{
    constexpr int nD  = D / MT;
    constexpr int nBc = Bc / MT;
    const int qblk  = blockIdx.x % (kSeqQ / Br);
    const int head  = (blockIdx.x / (kSeqQ / Br)) % kHeads;
    const int batch = blockIdx.x / (kSeqQ / Br) / kHeads;
    const int L = threadIdx.x;

    const __half* Qblk  = Q + ((size_t)batch*kHeads+head)*kSeqQ*D + qblk*Br*D;
    const __half* Kbase = K + ((size_t)batch*kHeads+head)*N*D;
    const __half* Vbase = V + ((size_t)batch*kHeads+head)*N*D;
    __half*       Oblk  = O + ((size_t)batch*kHeads+head)*kSeqQ*D + qblk*Br*D;

    __shared__ __half Q_lds[Br*D];
    __shared__ __half K_lds[Bc*D];
    __shared__ __half V_lds[Bc*D];
    __shared__ float  S_lds[Br*Bc];
    __shared__ __half P_lds[Br*Bc];
    __shared__ float  O_lds[Br*D];

    for (int i=L;i<Br*D;i+=64) Q_lds[i]=Qblk[i];
    for (int i=L;i<Br*D;i+=64) O_lds[i]=0.f;
    float m_run=-1e38f, l_run=0.f;
    const float scale=1.f/sqrtf((float)D);
    const int kv_tiles=N/Bc;
    __syncthreads();

    for (int kv=0;kv<kv_tiles;++kv){
        const __half* Kt=Kbase+(size_t)kv*Bc*D;
        const __half* Vt=Vbase+(size_t)kv*Bc*D;
        for (int i=L;i<Bc*D;i+=64){ K_lds[i]=Kt[i]; V_lds[i]=Vt[i]; }
        __syncthreads();

        // S = Q.Kt
        #pragma unroll
        for (int jc=0;jc<nBc;++jc){
            f32x4 acc={0,0,0,0};
            #pragma unroll
            for (int kd=0;kd<nD;++kd){
                fp16x4 a,b;
                int q_row=L%MT, d_idx=kd*MT+(L/MT)*4;
                #pragma unroll
                for(int f=0;f<4;++f) a[f]=h2f16(Q_lds[q_row*D + d_idx + f]);
                int key=jc*MT+(L%MT);
                #pragma unroll
                for(int f=0;f<4;++f){ int dd=kd*MT+(L/MT)*4+f; b[f]=h2f16(K_lds[key*D+dd]); }
                acc=mfma(a,b,acc);
            }
            #pragma unroll
            for(int f=0;f<4;++f){ int sr=(L/MT)*4+f, sc=jc*MT+(L%MT); S_lds[sr*Bc+sc]=acc[f]*scale; }
        }
        __syncthreads();

        // softmax (lanes 0..15)
        if (L<Br){
            int row=L; float rmax=-1e38f;
            #pragma unroll
            for(int c=0;c<Bc;++c) rmax=fmaxf(rmax,S_lds[row*Bc+c]);
            float mnew=fmaxf(m_run,rmax), rsum=0.f;
            #pragma unroll
            for(int c=0;c<Bc;++c){ float p=__expf(S_lds[row*Bc+c]-mnew); P_lds[row*Bc+c]=__float2half(p); rsum+=p; }
            float resc=__expf(m_run-mnew);
            for(int d=0;d<D;++d) O_lds[row*D+d]*=resc;
            m_run=mnew; l_run=resc*l_run+rsum;
        }
        __syncthreads();

        // O += P.V
        #pragma unroll
        for (int jd=0;jd<nD;++jd){
            f32x4 acc={0,0,0,0};
            #pragma unroll
            for (int kb=0;kb<nBc;++kb){
                fp16x4 a,b;
                int p_row=L%MT, bc_idx=kb*MT+(L/MT)*4;
                #pragma unroll
                for(int f=0;f<4;++f) a[f]=h2f16(P_lds[p_row*Bc + bc_idx + f]);
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

    __shared__ float l_share[Br];
    if (L<Br) l_share[L]=l_run;
    __syncthreads();
    for (int i=L;i<Br*D;i+=64){ int row=i/D; Oblk[i]=__float2half(O_lds[i]/l_share[row]); }
}

void attn_ref(const std::vector<float>&Q,const std::vector<float>&K,const std::vector<float>&V,
              std::vector<float>&O,int sq,int sk,int d){
    float sc=1.f/sqrtf((float)d);
    for(int q=0;q<sq;++q){ std::vector<float> S(sk); float mx=-1e38f;
        for(int k=0;k<sk;++k){ float dot=0; for(int dd=0;dd<d;++dd) dot+=Q[q*d+dd]*K[k*d+dd]; S[k]=dot*sc; mx=std::max(mx,S[k]); }
        float se=0; for(int k=0;k<sk;++k){ S[k]=expf(S[k]-mx); se+=S[k]; }
        for(int k=0;k<sk;++k) S[k]/=se;
        for(int dd=0;dd<d;++dd){ float o=0; for(int k=0;k<sk;++k) o+=S[k]*V[k*d+dd]; O[q*d+dd]=o; } }
}

template<int D>
void run() {
    const int kBlocks=kBatch*kHeads*kSeqQ/Br;
    const size_t QKV=(size_t)kBatch*kHeads*kSeqQ*D, KV=(size_t)kBatch*kHeads*kN*D;
    std::vector<float> hQf(QKV),hKf(KV),hVf(KV);
    std::mt19937 rng(42); std::normal_distribution<float> dist(0.f,0.5f);
    for(auto&v:hQf)v=dist(rng); for(auto&v:hKf)v=dist(rng); for(auto&v:hVf)v=dist(rng);
    std::vector<__half> hQ(QKV),hK(KV),hV(KV);
    for(size_t i=0;i<QKV;++i)hQ[i]=__float2half(hQf[i]);
    for(size_t i=0;i<KV;++i){hK[i]=__float2half(hKf[i]);hV[i]=__float2half(hVf[i]);}
    __half *dQ,*dK,*dV,*dO;
    hipMalloc(&dQ,QKV*2);hipMalloc(&dK,KV*2);hipMalloc(&dV,KV*2);hipMalloc(&dO,QKV*2);
    hipMemcpy(dQ,hQ.data(),QKV*2,hipMemcpyHostToDevice);
    hipMemcpy(dK,hK.data(),KV*2,hipMemcpyHostToDevice);
    hipMemcpy(dV,hV.data(),KV*2,hipMemcpyHostToDevice);
    dim3 grid(kBlocks), block(64);

    hipLaunchKernelGGL(fa_mfma_kernel<D>,grid,block,0,0,dQ,dK,dV,dO,kN);
    hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){ printf("  D=%d KERNEL ERROR: %s\n",D,hipGetErrorString(e)); return; }
    std::vector<__half> hOg(QKV); hipMemcpy(hOg.data(),dO,QKV*2,hipMemcpyDeviceToHost);
    std::vector<float> hOr(kSeqQ*D); attn_ref(hQf,hKf,hVf,hOr,kSeqQ,kN,D);
    float err=0; for(int i=0;i<kSeqQ*D;++i) err=std::max(err,fabsf(__half2float(hOg[i])-hOr[i]));

    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    for(int i=0;i<kWarmup;++i) hipLaunchKernelGGL(fa_mfma_kernel<D>,grid,block,0,0,dQ,dK,dV,dO,kN);
    hipDeviceSynchronize();
    std::vector<float> ts(kIter);
    for(int i=0;i<kIter;++i){ hipEventRecord(t0); hipLaunchKernelGGL(fa_mfma_kernel<D>,grid,block,0,0,dQ,dK,dV,dO,kN); hipEventRecord(t1); hipEventSynchronize(t1); hipEventElapsedTime(&ts[i],t0,t1); ts[i]*=1000.f; }
    std::sort(ts.begin(),ts.end());
    float med=ts[kIter/2];
    double flops=2.0*2.0*Br*Bc*D*(kN/Bc)*(double)kBlocks;
    printf("  D=%-4d  max_err=%.5f [%s]  median=%.1f µs  %.2f TFLOPS\n",
           D, err, err<0.1f?"PASS":"FAIL", med, flops/med/1e6);
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
}

int main(){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    printf("Device : %s (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("G1: MFMA FA head-dim sweep — does larger D let MFMA saturate the cores?\n");
    printf("Config: Br=16 Bc=64 seqLen=512 heads=8 (1 wave/block, MFMA Q·Kᵀ and P·V)\n\n");
    run<64>();
    run<128>();
    printf("\nHypothesis: D=128 (nD=8 MFMA tiles/iter) should improve TFLOPS vs D=64 (nD=4).\n");
    printf("Compare D=128 TFLOPS here against the D=128 dot-product path (run separately).\n");
    return 0;
}

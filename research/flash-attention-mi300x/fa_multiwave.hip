// ---------------------------------------------------------------------------
// fa_multiwave.hip — Axis-A multi-wave Flash Attention (closing the occupancy gap)
//
// Cold design: DESIGN_multiwave_fa.md. Motivation: 2-C measured fa_robust 8.5-15×
// slower than production fused SDPA, localized to the 1-wave/block design idling
// 48/64 lanes during softmax. This kernel uses W waves per block, each owning its
// own 16 query rows, sharing the K/V LDS tiles, and computing softmax across the
// FULL wave via the verified DPP reduction (Bc=64 == wavesize → one DPP reduce per
// row, all 64 lanes active, zero ds_bpermute).
//
// What is UNCHANGED from fa_robust (de-risks the rewrite): the MFMA 16×16×16 Q·Kᵀ /
// P·V inner loops + verified lane mapping, causal tile-skip + element-mask, GQA
// kvhead mapping, online-softmax math, the fp32 CPU reference, and the seqLen sweep.
//
// KEY DESIGN POINT (resolved cold): with softmax reduced across all 64 lanes per
// row, the per-row running state (m_run, l_run) can no longer live in a per-lane
// register (it did in the 1-wave design, where lane `row` owned row `row`). It now
// lives in LDS: m_lds[W][Br], l_lds[W][Br], one slot per (wave,row). The O
// accumulator already lived in LDS (O_lds). All 64 lanes of a row read/write the
// same m/l slot; lane 0 commits the scalar update, a barrier publishes it.
//
// Single-GPU kernel — does NOT need multi-GPU. Validate on 1×MI300X.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 fa_multiwave.hip -o fa_multiwave
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

static constexpr int Br = 16;
static constexpr int Bc = 64;   // == wavesize: one DPP wave-reduce covers one row's scores
static constexpr int MT = 16;
static constexpr int kWarmup = 10;
static constexpr int kIter   = 100;

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;
__device__ __forceinline__ __fp16 h2f16(__half h){ return *reinterpret_cast<__fp16*>(&h); }
__device__ __forceinline__ f32x4 mfma(fp16x4 a, fp16x4 b, f32x4 c){ return __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,c,0,0,0); }

// --- DPP wave64 reductions (from the verified wave_reduce_dpp.hpp, inlined) -------
// row_shr 1,2,4,8 → per-row reduction; row_bcast15/31 carry across boundaries;
// result on lane 63; readlane(63) broadcasts. Identity: max→-inf, sum→0.
template<int CTRL> __device__ __forceinline__ float dpp_mov(float v, float old){
    return __builtin_amdgcn_update_dpp(old, v, CTRL, 0xf, 0xf, false);
}
__device__ __forceinline__ float wave_reduce_max(float v){
    float t;
    t=dpp_mov<0x111>(v,-1e38f); v=fmaxf(v,t);
    t=dpp_mov<0x112>(v,-1e38f); v=fmaxf(v,t);
    t=dpp_mov<0x114>(v,-1e38f); v=fmaxf(v,t);
    t=dpp_mov<0x118>(v,-1e38f); v=fmaxf(v,t);
    t=dpp_mov<0x142>(v,-1e38f); v=fmaxf(v,t);
    t=dpp_mov<0x143>(v,-1e38f); v=fmaxf(v,t);
    return __builtin_amdgcn_readlane(v, 63);
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

struct Cfg { int D, seqQ, seqK, nQHeads, nKVHeads, batch, causal; };

// ---------------------------------------------------------------------------
// Multi-wave kernel — W waves per block, each wave owns one Br=16 query sub-block.
//   block = W*64 threads;  grid.x = batch * nQHeads * (seqQ / (W*Br))
//   K/V tiles loaded ONCE per block (all W*64 lanes cooperate), reused by all waves.
// ---------------------------------------------------------------------------
template<int D, int W>
__global__ void fa_multiwave_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                    const __half* __restrict__ V, __half* __restrict__ O,
                                    Cfg c)
{
    constexpr int nD  = D / MT;
    constexpr int nBc = Bc / MT;
    const int wave = threadIdx.x / 64;          // which wave in the block [0,W)
    const int L    = threadIdx.x % 64;          // lane within the wave
    const int qblocks = c.seqQ / (W*Br);        // super-blocks of W*Br query rows

    const int sblk  = blockIdx.x % qblocks;
    const int qhead = (blockIdx.x / qblocks) % c.nQHeads;
    const int batch = blockIdx.x / qblocks / c.nQHeads;

    const int group  = c.nQHeads / c.nKVHeads;
    const int kvhead = qhead / group;

    const int qrow0 = sblk*(W*Br) + wave*Br;    // this wave's first query row in the head
    const __half* Qblk  = Q + (((size_t)batch*c.nQHeads + qhead)*c.seqQ + qrow0)*D;
    const __half* Kbase = K + ((size_t)batch*c.nKVHeads + kvhead)*c.seqK*D;
    const __half* Vbase = V + ((size_t)batch*c.nKVHeads + kvhead)*c.seqK*D;
    __half*       Oblk  = O + (((size_t)batch*c.nQHeads + qhead)*c.seqQ + qrow0)*D;

    // K/V shared across the whole block; Q/S/P/O + running (m,l) are per-wave.
    __shared__ __half K_lds[Bc*D];
    __shared__ __half V_lds[Bc*D];
    __shared__ __half Q_lds[W][Br*D];
    __shared__ float  S_lds[W][Br*Bc];
    __shared__ __half P_lds[W][Br*Bc];
    __shared__ float  O_lds[W][Br*D];
    __shared__ float  m_lds[W][Br];             // per-row running max  (lives in LDS now)
    __shared__ float  l_lds[W][Br];             // per-row running sum

    for (int i=L;i<Br*D;i+=64) Q_lds[wave][i]=Qblk[i];
    for (int i=L;i<Br*D;i+=64) O_lds[wave][i]=0.f;
    if (L<Br){ m_lds[wave][L]=-1e38f; l_lds[wave][L]=0.f; }
    const float scale=1.f/sqrtf((float)D);

    const int last_q = qrow0 + Br - 1;
    int kv_tiles = c.seqK / Bc;
    if (c.causal){ int mt=(last_q/Bc)+1; if(mt<kv_tiles) kv_tiles=mt; }
    __syncthreads();

    for (int kv=0; kv<kv_tiles; ++kv){
        // Cooperative K/V load: ALL W*64 lanes load the shared tile once.
        const __half* Kt=Kbase+(size_t)kv*Bc*D;
        const __half* Vt=Vbase+(size_t)kv*Bc*D;
        for (int i=threadIdx.x; i<Bc*D; i+=W*64){ K_lds[i]=Kt[i]; V_lds[i]=Vt[i]; }
        __syncthreads();

        // S = Q·Kᵀ  (per-wave MFMA, unchanged from fa_robust)
        #pragma unroll
        for (int jc=0;jc<nBc;++jc){
            f32x4 acc={0,0,0,0};
            #pragma unroll
            for (int kd=0;kd<nD;++kd){
                fp16x4 a,b; int q_row=L%MT, d_idx=kd*MT+(L/MT)*4;
                #pragma unroll
                for(int f=0;f<4;++f) a[f]=h2f16(Q_lds[wave][q_row*D + d_idx + f]);
                int key=jc*MT+(L%MT);
                #pragma unroll
                for(int f=0;f<4;++f){ int dd=kd*MT+(L/MT)*4+f; b[f]=h2f16(K_lds[key*D+dd]); }
                acc=mfma(a,b,acc);
            }
            #pragma unroll
            for(int f=0;f<4;++f){
                int sr=(L/MT)*4+f, sc=jc*MT+(L%MT);
                float s = acc[f]*scale;
                if (c.causal){ int gq=qrow0+sr, gk=kv*Bc+sc; if(gk>gq) s=-1e30f; }
                S_lds[wave][sr*Bc+sc]=s;
            }
        }
        __syncthreads();

        // --- softmax across the FULL wave (the occupancy fix) ---
        // Each row reduced over its Bc=64 scores using all 64 lanes (lane L = col L).
        // The running (m,l) for the row lives in LDS; lane 0 commits the scalar update
        // and the O rescale, then we write P for the P·V matmul. No idle lanes.
        #pragma unroll
        for (int row=0; row<Br; ++row){
            float s    = S_lds[wave][row*Bc + L];
            float rmax = wave_reduce_max(s);                 // all 64 lanes
            float mold = m_lds[wave][row];
            float mnew = fmaxf(mold, rmax);
            float p    = __expf(s - mnew);
            float rsum = wave_reduce_sum(p);                 // all 64 lanes
            P_lds[wave][row*Bc + L] = __float2half(p);
            if (L==0){
                float resc = __expf(mold - mnew);
                for (int d=0; d<D; ++d) O_lds[wave][row*D+d] *= resc;
                m_lds[wave][row] = mnew;
                l_lds[wave][row] = resc*l_lds[wave][row] + rsum;
            }
        }
        __syncthreads();

        // O += P·V  (per-wave MFMA, unchanged)
        #pragma unroll
        for (int jd=0;jd<nD;++jd){
            f32x4 acc={0,0,0,0};
            #pragma unroll
            for (int kb=0;kb<nBc;++kb){
                fp16x4 a,b; int p_row=L%MT, bc_idx=kb*MT+(L/MT)*4;
                #pragma unroll
                for(int f=0;f<4;++f) a[f]=h2f16(P_lds[wave][p_row*Bc + bc_idx + f]);
                int d_col=jd*MT+(L%MT);
                #pragma unroll
                for(int f=0;f<4;++f){ int kk=kb*MT+(L/MT)*4+f; b[f]=h2f16(V_lds[kk*D+d_col]); }
                acc=mfma(a,b,acc);
            }
            #pragma unroll
            for(int f=0;f<4;++f){ int orr=(L/MT)*4+f, oc=jd*MT+(L%MT); O_lds[wave][orr*D+oc]+=acc[f]; }
        }
        __syncthreads();
    }

    // Final normalize: O[row] /= l[row]  (guard fully-masked rows).
    if (L<Br) l_lds[wave][L] = (l_lds[wave][L]>0.f)?l_lds[wave][L]:1.f;
    __syncthreads();
    for (int i=L;i<Br*D;i+=64){ int row=i/D; Oblk[i]=__float2half(O_lds[wave][i]/l_lds[wave][row]); }
}

// ---------------------------------------------------------------------------
// CPU reference (GQA + causal) — identical to fa_robust's attn_ref.
// ---------------------------------------------------------------------------
void attn_ref(const std::vector<float>& Q, const std::vector<float>& K,
              const std::vector<float>& V, std::vector<float>& O, const Cfg& c){
    const float scale=1.f/sqrtf((float)c.D);
    const int group=c.nQHeads/c.nKVHeads;
    for (int b=0;b<c.batch;++b)
    for (int h=0;h<c.nQHeads;++h){
        int kvh=h/group;
        for (int q=0;q<c.seqQ;++q){
            int kmax=c.causal?std::min(c.seqK,q+1):c.seqK;
            std::vector<float> S(kmax); float mx=-1e38f;
            const float* qp=&Q[(((size_t)b*c.nQHeads+h)*c.seqQ+q)*c.D];
            for (int k=0;k<kmax;++k){ float dot=0;
                const float* kp=&K[(((size_t)b*c.nKVHeads+kvh)*c.seqK+k)*c.D];
                for(int d=0;d<c.D;++d)dot+=qp[d]*kp[d];
                S[k]=dot*scale; mx=std::max(mx,S[k]); }
            float se=0; for(int k=0;k<kmax;++k){S[k]=expf(S[k]-mx);se+=S[k];}
            for(int k=0;k<kmax;++k)S[k]/=se;
            float* op=&O[(((size_t)b*c.nQHeads+h)*c.seqQ+q)*c.D];
            for(int d=0;d<c.D;++d){ float o=0;
                for(int k=0;k<kmax;++k){ const float* vp=&V[(((size_t)b*c.nKVHeads+kvh)*c.seqK+k)*c.D]; o+=S[k]*vp[d]; }
                op[d]=o; }
        }
    }
}

// ---------------------------------------------------------------------------
// Launcher + verify/timing. Mirrors fa_robust's run_cfg so speedups are
// apples-to-apples. W is the wave-count template (super-block = W*Br query rows).
// ---------------------------------------------------------------------------
template<int D, int W>
float run_cfg(Cfg c, const char* label, bool verify){
    c.D=D;
    const size_t Qel=(size_t)c.batch*c.nQHeads*c.seqQ*D;
    const size_t KVel=(size_t)c.batch*c.nKVHeads*c.seqK*D;
    std::vector<float> hQf(Qel),hKf(KVel),hVf(KVel);
    std::mt19937 rng(123); std::normal_distribution<float> dist(0.f,0.5f);
    for(auto&v:hQf)v=dist(rng); for(auto&v:hKf)v=dist(rng); for(auto&v:hVf)v=dist(rng);
    std::vector<__half> hQ(Qel),hK(KVel),hV(KVel);
    for(size_t i=0;i<Qel;++i)hQ[i]=__float2half(hQf[i]);
    for(size_t i=0;i<KVel;++i){hK[i]=__float2half(hKf[i]);hV[i]=__float2half(hVf[i]);}
    __half *dQ,*dK,*dV,*dO;
    hipMalloc(&dQ,Qel*2);hipMalloc(&dK,KVel*2);hipMalloc(&dV,KVel*2);hipMalloc(&dO,Qel*2);
    hipMemcpy(dQ,hQ.data(),Qel*2,hipMemcpyHostToDevice);
    hipMemcpy(dK,hK.data(),KVel*2,hipMemcpyHostToDevice);
    hipMemcpy(dV,hV.data(),KVel*2,hipMemcpyHostToDevice);

    // grid covers super-blocks of W*Br query rows. Requires seqQ % (W*Br) == 0.
    int sblocks = c.seqQ/(W*Br);
    int blocks  = c.batch*c.nQHeads*sblocks;
    dim3 grid(blocks), block(W*64);

    hipLaunchKernelGGL((fa_multiwave_kernel<D,W>),grid,block,0,0,dQ,dK,dV,dO,c);
    hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){ printf("  %-30s KERNEL ERROR: %s\n",label,hipGetErrorString(e));
        hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO); return -1.f; }

    float err=-1.f;
    if(verify){
        std::vector<__half> hOg(Qel); hipMemcpy(hOg.data(),dO,Qel*2,hipMemcpyDeviceToHost);
        std::vector<float> hOr(Qel,0.f); attn_ref(hQf,hKf,hVf,hOr,c);
        err=0; for(size_t i=0;i<Qel;++i) err=std::max(err,fabsf(__half2float(hOg[i])-hOr[i]));
    }
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    for(int i=0;i<kWarmup;++i) hipLaunchKernelGGL((fa_multiwave_kernel<D,W>),grid,block,0,0,dQ,dK,dV,dO,c);
    hipDeviceSynchronize();
    std::vector<float> ts(kIter);
    for(int i=0;i<kIter;++i){ hipEventRecord(t0); hipLaunchKernelGGL((fa_multiwave_kernel<D,W>),grid,block,0,0,dQ,dK,dV,dO,c); hipEventRecord(t1); hipEventSynchronize(t1); hipEventElapsedTime(&ts[i],t0,t1); ts[i]*=1000.f; }
    std::sort(ts.begin(),ts.end());
    float med=ts[kIter/2];
    double work=c.causal?0.5:1.0;
    double flops=2.0*2.0*(double)c.batch*c.nQHeads*c.seqQ*c.seqK*D*work;
    if(verify) printf("  %-30s D=%d W=%d  err=%.5f [%s]  %.1f µs  %.2f TFLOPS\n",
                      label,D,W,err,err<0.15f?"PASS":"FAIL",med,flops/med/1e6);
    else       printf("  %-30s D=%d W=%d  (timing only)  %.1f µs  %.2f TFLOPS\n",
                      label,D,W,med,flops/med/1e6);
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
    return med;
}

int main(){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    printf("Device : %s (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("Multi-wave FA (Axis-A: W waves/block, DPP softmax, shared K/V) — vs fa_robust\n\n");

    // Correctness (CPU-verified). seqQ must be a multiple of W*Br.
    // W=2: super-block 32 rows; W=4: super-block 64 rows. seq256 divisible by both.
    printf("--- correctness (small configs, CPU-verified) ---\n");
    run_cfg<64, 2>({0,256,256,2,2,1,0}, "MHA  full   seq256 h2", true);
    run_cfg<128,2>({0,256,256,2,2,1,0}, "MHA  full   seq256 h2", true);
    run_cfg<64, 4>({0,256,256,2,2,1,0}, "MHA  full   seq256 h2", true);
    run_cfg<128,2>({0,256,256,4,2,1,0}, "GQA  full   seq256 4q2kv", true);
    run_cfg<128,2>({0,256,256,2,2,1,1}, "MHA  causal seq256 h2", true);
    run_cfg<128,2>({0,256,256,4,2,1,1}, "GQA  causal seq256 4q2kv", true);

    // Speedup sweep vs fa_robust: same seqLens, D=128, MHA full. Compare medians to
    // fa_robust's published curve (n^1.90). H-MW1: multi-wave faster at matched work.
    printf("\n--- multi-wave seqLen sweep (D=128, MHA, full) — compare to fa_robust ---\n");
    const int seqs[6]={512,1024,1536,2048,3072,4096};
    for(int i=0;i<6;++i){ char l[40]; snprintf(l,sizeof(l),"MHA full seq%d",seqs[i]);
        run_cfg<128,2>({0,seqs[i],seqs[i],8,8,1,0}, l, false); }

    printf("\n--- W=4 variant (D=64, fits LDS) ---\n");
    run_cfg<64,4>({0,2048,2048,8,8,1,0}, "MHA full seq2k D64", false);

    printf("\nNotes: causal TFLOPS uses 0.5x work. Compare medians to fa_robust same config.\n");
    printf("LDS budget: W=2 D=128 fits; W=4 needs D=64 (see DESIGN_multiwave_fa.md §3).\n");
    return 0;
}

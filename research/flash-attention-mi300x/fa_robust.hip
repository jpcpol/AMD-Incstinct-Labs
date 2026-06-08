// ---------------------------------------------------------------------------
// fa_robust.hip — Flash Attention robustness sweep for LLM-realistic shapes
//
// Step 1 of the research->application arc: take the verified fa_mfma2 design
// (1 wave/block, MFMA Q·Kᵀ and P·V, DPP-free softmax on lanes 0-15) and
// generalize it to the shapes real transformers use:
//   - head dim D = 64 or 128 (template)
//   - sequence length runtime (512 / 1k / 2k / 4k)
//   - n_q_heads / n_kv_heads  -> GQA (Grouped-Query Attention, e.g. Llama-3)
//   - causal masking (decoder LLMs are causal; skip fully-above-diagonal tiles)
//   - batch runtime
//
// Validates correctness against a CPU reference that also implements GQA + causal,
// then sweeps timing across configs to find where DPP+DME gains grow or shrink.
//
// MFMA lane mapping (verified probe_mfma_mapping.hip):
//   A in : lane L,f -> A[row=L%16][k=(L/16)*4+f]
//   B in : lane L,f -> B[k=(L/16)*4+f][col=L%16]
//   C/out: lane L,f -> C[row=(L/16)*4+f][col=L%16]
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 fa_robust.hip -o fa_robust
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
static constexpr int MT = 16;
static constexpr int kWarmup = 10;
static constexpr int kIter   = 100;

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;
__device__ __forceinline__ __fp16 h2f16(__half h){ return *reinterpret_cast<__fp16*>(&h); }
__device__ __forceinline__ f32x4 mfma(fp16x4 a, fp16x4 b, f32x4 c){ return __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,c,0,0,0); }

// Runtime config passed to the kernel
struct Cfg {
    int D;          // head dim (also templated for LDS sizing; passed for ptr math)
    int seqQ;       // query sequence length
    int seqK;       // key/value sequence length
    int nQHeads;    // number of query heads
    int nKVHeads;   // number of KV heads (GQA: <= nQHeads, divides it)
    int batch;
    int causal;     // 1 = causal mask, 0 = full attention
};

// ---------------------------------------------------------------------------
// Kernel — one wave per (batch, q_head, query-block)
// grid.x = batch * nQHeads * (seqQ / Br)
// ---------------------------------------------------------------------------
template<int D>
__global__ void fa_robust_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                 const __half* __restrict__ V, __half* __restrict__ O,
                                 Cfg c)
{
    constexpr int nD  = D / MT;
    constexpr int nBc = Bc / MT;
    const int qblocks = c.seqQ / Br;

    const int qblk  = blockIdx.x % qblocks;
    const int qhead = (blockIdx.x / qblocks) % c.nQHeads;
    const int batch = blockIdx.x / qblocks / c.nQHeads;
    const int L = threadIdx.x;

    // GQA: which KV head does this query head map to?
    const int group = c.nQHeads / c.nKVHeads;   // queries per kv head
    const int kvhead = qhead / group;

    // Pointers. Q laid out [batch, nQHeads, seqQ, D]; K/V [batch, nKVHeads, seqK, D]
    const __half* Qblk  = Q + (((size_t)batch*c.nQHeads + qhead)*c.seqQ + qblk*Br)*D;
    const __half* Kbase = K + ((size_t)batch*c.nKVHeads + kvhead)*c.seqK*D;
    const __half* Vbase = V + ((size_t)batch*c.nKVHeads + kvhead)*c.seqK*D;
    __half*       Oblk  = O + (((size_t)batch*c.nQHeads + qhead)*c.seqQ + qblk*Br)*D;

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

    // Causal: this query block covers global query rows [qblk*Br, qblk*Br+Br-1].
    // A KV tile kv covers keys [kv*Bc, kv*Bc+Bc-1]. With causal masking, the
    // block can skip any tile whose first key index exceeds the block's last
    // query position — that tile is entirely above the diagonal.
    const int last_q = qblk*Br + Br - 1;
    int kv_tiles = c.seqK / Bc;
    if (c.causal) {
        int max_tile = (last_q / Bc) + 1;   // tiles up to and including the diagonal
        if (max_tile < kv_tiles) kv_tiles = max_tile;
    }
    __syncthreads();

    for (int kv=0; kv<kv_tiles; ++kv){
        const __half* Kt=Kbase+(size_t)kv*Bc*D;
        const __half* Vt=Vbase+(size_t)kv*Bc*D;
        for (int i=L;i<Bc*D;i+=64){ K_lds[i]=Kt[i]; V_lds[i]=Vt[i]; }
        __syncthreads();

        // S = Q·Kᵀ
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
            for(int f=0;f<4;++f){
                int sr=(L/MT)*4+f, sc=jc*MT+(L%MT);
                float s = acc[f]*scale;
                // Causal element-mask: query global row = qblk*Br+sr, key global = kv*Bc+sc
                if (c.causal) {
                    int gq = qblk*Br + sr;
                    int gk = kv*Bc + sc;
                    if (gk > gq) s = -1e30f;   // mask out future keys
                }
                S_lds[sr*Bc+sc] = s;
            }
        }
        __syncthreads();

        // online softmax (lanes 0..15 = query rows)
        if (L<Br){
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

        // O += P·V
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
    if (L<Br) l_share[L] = (l_run > 0.f) ? l_run : 1.f;  // guard fully-masked rows
    __syncthreads();
    for (int i=L;i<Br*D;i+=64){ int row=i/D; Oblk[i]=__float2half(O_lds[i]/l_share[row]); }
}

// ---------------------------------------------------------------------------
// CPU reference with GQA + causal
// ---------------------------------------------------------------------------
void attn_ref(const std::vector<float>& Q, const std::vector<float>& K,
              const std::vector<float>& V, std::vector<float>& O, const Cfg& c)
{
    const float scale = 1.f/sqrtf((float)c.D);
    const int group = c.nQHeads / c.nKVHeads;
    for (int b=0;b<c.batch;++b)
    for (int h=0;h<c.nQHeads;++h){
        int kvh = h / group;
        for (int q=0;q<c.seqQ;++q){
            int kmax = c.causal ? std::min(c.seqK, q+1) : c.seqK;  // causal cutoff
            std::vector<float> S(kmax);
            float mx=-1e38f;
            for (int k=0;k<kmax;++k){
                float dot=0;
                const float* qp = &Q[(((size_t)b*c.nQHeads+h)*c.seqQ+q)*c.D];
                const float* kp = &K[(((size_t)b*c.nKVHeads+kvh)*c.seqK+k)*c.D];
                for (int d=0;d<c.D;++d) dot+=qp[d]*kp[d];
                S[k]=dot*scale; mx=std::max(mx,S[k]);
            }
            float se=0; for(int k=0;k<kmax;++k){ S[k]=expf(S[k]-mx); se+=S[k]; }
            for(int k=0;k<kmax;++k) S[k]/=se;
            float* op = &O[(((size_t)b*c.nQHeads+h)*c.seqQ+q)*c.D];
            for (int d=0;d<c.D;++d){
                float o=0;
                for (int k=0;k<kmax;++k){
                    const float* vp = &V[(((size_t)b*c.nKVHeads+kvh)*c.seqK+k)*c.D];
                    o+=S[k]*vp[d];
                }
                op[d]=o;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// verify=true runs the (expensive) CPU reference. For large seq/head configs the
// O(seq²·heads·D) CPU cost is minutes — so we verify only on small configs and
// time-only on large ones. GQA/causal correctness is established on the small
// configs; the large configs reuse the same kernel and only measure performance.
template<int D>
float run_cfg(Cfg c, const char* label, bool verify) {
    c.D = D;
    const size_t Qel = (size_t)c.batch*c.nQHeads*c.seqQ*D;
    const size_t KVel= (size_t)c.batch*c.nKVHeads*c.seqK*D;

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

    int blocks = c.batch*c.nQHeads*(c.seqQ/Br);
    dim3 grid(blocks), block(64);

    hipLaunchKernelGGL(fa_robust_kernel<D>,grid,block,0,0,dQ,dK,dV,dO,c);
    hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){ printf("  %-28s KERNEL ERROR: %s\n",label,hipGetErrorString(e));
        hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO); return -1.f; }

    float err=-1.f;
    if (verify) {
        std::vector<__half> hOg(Qel); hipMemcpy(hOg.data(),dO,Qel*2,hipMemcpyDeviceToHost);
        std::vector<float> hOr(Qel,0.f); attn_ref(hQf,hKf,hVf,hOr,c);
        err=0; for(size_t i=0;i<Qel;++i) err=std::max(err,fabsf(__half2float(hOg[i])-hOr[i]));
    }

    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    for(int i=0;i<kWarmup;++i) hipLaunchKernelGGL(fa_robust_kernel<D>,grid,block,0,0,dQ,dK,dV,dO,c);
    hipDeviceSynchronize();
    std::vector<float> ts(kIter);
    for(int i=0;i<kIter;++i){ hipEventRecord(t0); hipLaunchKernelGGL(fa_robust_kernel<D>,grid,block,0,0,dQ,dK,dV,dO,c); hipEventRecord(t1); hipEventSynchronize(t1); hipEventElapsedTime(&ts[i],t0,t1); ts[i]*=1000.f; }
    std::sort(ts.begin(),ts.end());
    float med=ts[kIter/2];
    // FLOPs: causal halves the work approximately
    double work = c.causal ? 0.5 : 1.0;
    double flops = 2.0*2.0*(double)c.batch*c.nQHeads*c.seqQ*c.seqK*D*work;
    if (verify)
        printf("  %-28s D=%d  err=%.5f [%s]  %.1f µs  %.2f TFLOPS\n",
               label, D, err, err<0.15f?"PASS":"FAIL", med, flops/med/1e6);
    else
        printf("  %-28s D=%d  (timing only)    %.1f µs  %.2f TFLOPS\n",
               label, D, med, flops/med/1e6);
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
    return med;
}

int main(){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    printf("Device : %s (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("FA robustness sweep — LLM-realistic shapes (GQA + causal + seqLen)\n\n");

    // === Correctness at small scale (CPU ref affordable): verify GQA + causal ===
    printf("--- correctness (small configs, CPU-verified) ---\n");
    run_cfg<64> ({0,256,256,2,2,1,0}, "MHA  full  seq256 h2",  true);
    run_cfg<128>({0,256,256,2,2,1,0}, "MHA  full  seq256 h2",  true);
    run_cfg<128>({0,256,256,4,2,1,0}, "GQA  full  seq256 4q2kv",true);   // GQA path
    run_cfg<128>({0,256,256,2,2,1,1}, "MHA  causal seq256 h2",  true);   // causal path
    run_cfg<128>({0,256,256,4,2,1,1}, "GQA  causal seq256 4q2kv",true);  // GQA+causal

    // === CAL-L4 flat-context O(n²) baseline (Rol 1) ===
    // 7-point seqLen sweep matching the L4 paper §5.2 protocol. Measures how the
    // flat-context attention cost scales with n — the curve the L4 Efficiency
    // Hypothesis claims to beat. Log-log fit → exponent (accept ∈ [1.8, 2.2]).
    // All seqLens are multiples of Br=16 and Bc=64. See cal-l4 paper §5.2.
    printf("\n--- CAL-L4 baseline: flat-context O(n²) sweep (D=128, MHA, full) ---\n");
    const int seqs[7] = {512, 768, 1024, 1536, 2048, 3072, 4096};
    float lat[7];
    for (int i = 0; i < 7; ++i) {
        char lbl[32]; snprintf(lbl, sizeof(lbl), "MHA full seq%d", seqs[i]);
        lat[i] = run_cfg<128>({0, seqs[i], seqs[i], 8, 8, 1, 0}, lbl, false);
    }

    // Log-log least-squares fit: cost ~ n^p. p≈2 confirms O(n²).
    {
        double sx=0,sy=0,sxx=0,sxy=0; int N=7;
        for(int i=0;i<N;++i){ double x=log((double)seqs[i]), y=log((double)lat[i]);
            sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; }
        double p=(N*sxy - sx*sy)/(N*sxx - sx*sx);
        // R² of the fit (how cleanly quadratic)
        double my=sy/N, sstot=0, ssres=0, b=(sy-p*sx)/N;
        for(int i=0;i<N;++i){ double x=log((double)seqs[i]), y=log((double)lat[i]);
            double yh=p*x+b; ssres+=(y-yh)*(y-yh); sstot+=(y-my)*(y-my); }
        double r2 = 1.0 - ssres/sstot;
        printf("\n  === CAL-L4 BASELINE RESULT ===\n");
        printf("  attention cost scales as n^%.2f  (R²=%.4f)\n", p, r2);
        printf("  accept window [1.8, 2.2]: %s\n", (p>=1.8&&p<=2.2)?"CONFIRMED O(n²)":"OUTSIDE — investigate");
        printf("  seq 512→4k: %.1f → %.1f µs (%.1f× over 8× seqLen)\n",
               lat[0], lat[6], lat[6]/lat[0]);
    }

    printf("\n--- GQA (32 q / 8 kv, Llama-3 style), D=128 ---\n");
    run_cfg<128>({0,2048,2048,32,8,1,0}, "GQA full seq2k",  false);

    printf("\n--- causal masking, D=128 ---\n");
    run_cfg<128>({0,2048,2048,8, 8,1,1}, "MHA causal seq2k",false);
    run_cfg<128>({0,2048,2048,32,8,1,1}, "GQA causal seq2k",false);

    printf("\nNotes: causal TFLOPS uses 0.5x work estimate. err threshold 0.15 (fp16).\n");
    printf("Correctness verified on small configs (MHA/GQA/causal); large configs timing-only.\n");
    return 0;
}

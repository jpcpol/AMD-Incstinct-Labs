// ---------------------------------------------------------------------------
// fa_decode.hip — split-KV decode Flash Attention (1 query × N cache keys).
//
// Cold design: DESIGN_decode_kvcache.md. Decode is the other half of LLM inference
// (one new token attending a growing KV-cache). It is structurally different from
// prefill: a 1×N matrix-VECTOR problem, memory-bound, where MFMA 16×16×16 tiles
// would be 15/16 empty. The right structure is split-KV (Axis-B): split the N keys
// across W waves, each computes a partial (m,l,O) over its key range, then a split-K
// combine across waves merges them.
//
// Uses the verified DPP wave-reduce for BOTH the q·K dot product and the softmax —
// its natural home (a true reduction, not a matmul). No MFMA in decode.
//
// Single-GPU — validate on 1×MI300X.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 fa_decode.hip -o fa_decode
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

static constexpr int kWarmup = 10;
static constexpr int kIter   = 200;   // decode is fast; more iters for a stable median

// --- DPP wave64 reduction (verified wave_reduce_dpp.hpp, inlined) ----------------
template<int CTRL> __device__ __forceinline__ float dpp_mov(float v, float old){
    return __builtin_amdgcn_update_dpp(old, v, CTRL, 0xf, 0xf, false);
}
__device__ __forceinline__ float wave_reduce_sum(float v){
    float t;
    t=dpp_mov<0x111>(v,0.f); v+=t;   // row_shr 1
    t=dpp_mov<0x112>(v,0.f); v+=t;   // row_shr 2
    t=dpp_mov<0x114>(v,0.f); v+=t;   // row_shr 4
    t=dpp_mov<0x118>(v,0.f); v+=t;   // row_shr 8
    t=dpp_mov<0x142>(v,0.f); v+=t;   // row_bcast15
    t=dpp_mov<0x143>(v,0.f); v+=t;   // row_bcast31
    return __builtin_amdgcn_readlane(v, 63);   // result on lane 63
}

struct DCfg { int D, N, nQHeads, nKVHeads; };   // N = current cache length

// ---------------------------------------------------------------------------
// Split-KV decode kernel.
//   grid = nQHeads (one block per query head); block = W*64 lanes.
//   q[nQH, D]  (the single new token's query, per head)
//   K[nKVH, N, D], V[nKVH, N, D]  (the cache)
//   o[nQH, D]  (attention output for the new token)
//
// Each wave handles a contiguous key range; lanes map across D for the q·K dot
// product (D/64 elements per lane) and the DPP reduce sums it. Online softmax per
// wave → partial (m_w, l_w, O_w[D]); split-K combine across waves via LDS.
// ---------------------------------------------------------------------------
template<int D, int W>
__global__ void fa_decode_kernel(const __half* __restrict__ q, const __half* __restrict__ K,
                                 const __half* __restrict__ V, __half* __restrict__ o,
                                 DCfg c)
{
    constexpr int EPL = D / 64;          // elements of D per lane (D=64→1, D=128→2)
    const int wave = threadIdx.x / 64;
    const int L    = threadIdx.x % 64;
    const int qhead = blockIdx.x;        // grid.x = nQHeads
    const int group = c.nQHeads / c.nKVHeads;
    const int kvhead = qhead / group;

    const __half* qh = q + (size_t)qhead*D;
    const __half* Kh = K + (size_t)kvhead*c.N*D;
    const __half* Vh = V + (size_t)kvhead*c.N*D;
    __half*       oh = o + (size_t)qhead*D;

    // Load q into registers: lane L owns elements [L*EPL, L*EPL+EPL) of the D vector
    // for the dot product. (D laid contiguous; this lane-maps D across 64 lanes.)
    float qreg[EPL];
    #pragma unroll
    for (int e=0;e<EPL;++e) qreg[e] = __half2float(qh[L*EPL+e]);

    const float scale = 1.f/sqrtf((float)D);

    // This wave's contiguous key range [k0, k1).
    int per = (c.N + W - 1) / W;
    int k0 = wave*per, k1 = min(k0+per, c.N);

    // Per-wave online-softmax partial over its key range.
    float m_w = -1e38f, l_w = 0.f;
    float O_w[EPL];                       // this lane's slice of the D-dim output accumulator
    #pragma unroll
    for (int e=0;e<EPL;++e) O_w[e]=0.f;

    for (int k=k0; k<k1; ++k){
        // s_k = (q · K[k]) * scale, reduced across the wave (all 64 lanes, DPP).
        const __half* Kk = Kh + (size_t)k*D;
        float partial = 0.f;
        #pragma unroll
        for (int e=0;e<EPL;++e) partial += qreg[e] * __half2float(Kk[L*EPL+e]);
        float s = wave_reduce_sum(partial) * scale;   // full dot product on lane 63...
        s = __builtin_amdgcn_readlane(s, 63);          // ...broadcast to all lanes
        // online softmax update
        float mnew = fmaxf(m_w, s);
        float p    = __expf(s - mnew);
        float resc = __expf(m_w - mnew);
        const __half* Vk = Vh + (size_t)k*D;
        #pragma unroll
        for (int e=0;e<EPL;++e) O_w[e] = O_w[e]*resc + p*__half2float(Vk[L*EPL+e]);
        l_w = l_w*resc + p;
        m_w = mnew;
    }

    // --- split-K combine across the W waves (via LDS) ---
    // Each wave publishes (m_w, l_w, O_w[D]); wave 0 merges to the global (m,l,O).
    __shared__ float m_sh[W], l_sh[W];
    __shared__ float O_sh[W][D];
    if (L==0){ m_sh[wave]=m_w; l_sh[wave]=l_w; }
    #pragma unroll
    for (int e=0;e<EPL;++e) O_sh[wave][L*EPL+e] = O_w[e];
    __syncthreads();

    if (wave==0){
        // global max over waves
        float m = -1e38f;
        #pragma unroll
        for (int w=0; w<W; ++w) m = fmaxf(m, m_sh[w]);
        // rescale-and-sum each wave's partial to the global max
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

// ---------------------------------------------------------------------------
// CPU reference: 1 query × N keys, GQA. (Decode causal is trivial — the new token
// attends all N cached keys.)
// ---------------------------------------------------------------------------
void decode_ref(const std::vector<float>& q, const std::vector<float>& K,
                const std::vector<float>& V, std::vector<float>& o, const DCfg& c){
    const float scale=1.f/sqrtf((float)c.D);
    const int group=c.nQHeads/c.nKVHeads;
    for (int h=0;h<c.nQHeads;++h){
        int kvh=h/group;
        std::vector<float> S(c.N); float mx=-1e38f;
        const float* qp=&q[(size_t)h*c.D];
        for (int k=0;k<c.N;++k){ float dot=0;
            const float* kp=&K[((size_t)kvh*c.N+k)*c.D];
            for(int d=0;d<c.D;++d)dot+=qp[d]*kp[d];
            S[k]=dot*scale; mx=std::max(mx,S[k]); }
        float se=0; for(int k=0;k<c.N;++k){S[k]=expf(S[k]-mx);se+=S[k];}
        float* op=&o[(size_t)h*c.D];
        for(int d=0;d<c.D;++d){ float acc=0;
            for(int k=0;k<c.N;++k){ const float* vp=&V[((size_t)kvh*c.N+k)*c.D]; acc+=(S[k]/se)*vp[d]; }
            op[d]=acc; }
    }
}

template<int D, int W>
float run_decode(DCfg c, const char* label, bool verify){
    c.D=D;
    const size_t qel=(size_t)c.nQHeads*D, kvel=(size_t)c.nKVHeads*c.N*D;
    std::vector<float> qf(qel),Kf(kvel),Vf(kvel);
    std::mt19937 rng(7); std::normal_distribution<float> dist(0.f,0.5f);
    for(auto&v:qf)v=dist(rng); for(auto&v:Kf)v=dist(rng); for(auto&v:Vf)v=dist(rng);
    std::vector<__half> qh(qel),Kh(kvel),Vh(kvel);
    for(size_t i=0;i<qel;++i)qh[i]=__float2half(qf[i]);
    for(size_t i=0;i<kvel;++i){Kh[i]=__float2half(Kf[i]);Vh[i]=__float2half(Vf[i]);}
    __half *dq,*dK,*dV,*doo;
    hipMalloc(&dq,qel*2);hipMalloc(&dK,kvel*2);hipMalloc(&dV,kvel*2);hipMalloc(&doo,qel*2);
    hipMemcpy(dq,qh.data(),qel*2,hipMemcpyHostToDevice);
    hipMemcpy(dK,Kh.data(),kvel*2,hipMemcpyHostToDevice);
    hipMemcpy(dV,Vh.data(),kvel*2,hipMemcpyHostToDevice);

    dim3 grid(c.nQHeads), block(W*64);
    hipLaunchKernelGGL((fa_decode_kernel<D,W>),grid,block,0,0,dq,dK,dV,doo,c);
    hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){ printf("  %-26s KERNEL ERROR: %s\n",label,hipGetErrorString(e));
        hipFree(dq);hipFree(dK);hipFree(dV);hipFree(doo); return -1.f; }

    float err=-1.f;
    if(verify){
        std::vector<__half> og(qel); hipMemcpy(og.data(),doo,qel*2,hipMemcpyDeviceToHost);
        std::vector<float> orf(qel,0.f); decode_ref(qf,Kf,Vf,orf,c);
        err=0; for(size_t i=0;i<qel;++i) err=std::max(err,fabsf(__half2float(og[i])-orf[i]));
    }
    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    for(int i=0;i<kWarmup;++i) hipLaunchKernelGGL((fa_decode_kernel<D,W>),grid,block,0,0,dq,dK,dV,doo,c);
    hipDeviceSynchronize();
    std::vector<float> ts(kIter);
    for(int i=0;i<kIter;++i){ hipEventRecord(t0); hipLaunchKernelGGL((fa_decode_kernel<D,W>),grid,block,0,0,dq,dK,dV,doo,c); hipEventRecord(t1); hipEventSynchronize(t1); hipEventElapsedTime(&ts[i],t0,t1); ts[i]*=1000.f; }
    std::sort(ts.begin(),ts.end());
    float med=ts[kIter/2];
    if(verify) printf("  %-26s D=%d W=%d N=%d  err=%.5f [%s]  %.2f µs\n",
                      label,D,W,c.N,err,err<0.15f?"PASS":"FAIL",med);
    else       printf("  %-26s D=%d W=%d N=%d  (timing)  %.2f µs\n",label,D,W,c.N,med);
    hipFree(dq);hipFree(dK);hipFree(dV);hipFree(doo);
    return med;
}

int main(){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    printf("Device : %s (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("Decode FA (split-KV, 1 query × N cache, DPP dot+softmax) — vs CPU ref\n\n");

    // Correctness (CPU ref): MHA + GQA, D=64/128, a few N.
    printf("--- correctness (CPU-verified) ---\n");
    run_decode<64, 4>({64, 128, 8, 8}, "MHA  D64  N128",  true);
    run_decode<128,4>({128,128, 8, 8}, "MHA  D128 N128",  true);
    run_decode<128,4>({128,512, 32,8}, "GQA  D128 N512 32q8kv", true);
    run_decode<64, 2>({64, 256, 8, 8}, "MHA  D64  N256 W2", true);

    // H-DEC3: N-sweep — decode should be ~linear in N (1×N, memory-bound), not N².
    printf("\n--- N-sweep (D=128, GQA 32q/8kv, W=4) — expect ~linear in N ---\n");
    const int Ns[6]={256,512,1024,2048,4096,8192};
    for(int i=0;i<6;++i){ char l[40]; snprintf(l,sizeof(l),"decode N%d",Ns[i]);
        run_decode<128,4>({128,Ns[i],32,8}, l, false); }

    // H-DEC2: split-KV scaling — W=4 vs W=1 at large N.
    printf("\n--- split-KV scaling (D=128, GQA, N=4096): W=1 vs W=4 ---\n");
    run_decode<128,1>({128,4096,32,8}, "decode W1", false);
    run_decode<128,4>({128,4096,32,8}, "decode W4", false);

    printf("\nNotes: decode is memory-bound (1×N). err threshold 0.15 (fp16).\n");
    return 0;
}

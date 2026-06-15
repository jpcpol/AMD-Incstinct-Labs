// ---------------------------------------------------------------------------
// fa_multiwave.hip — Axis-A multi-wave Flash Attention benchmark driver.
//
// Stage-B: the kernel itself lives in cdna3-prims/include/cdna3/detail/
// fa_prefill_kernel.hpp (single source of truth). This file adds the CPU
// reference, the launcher/timing harness, and main() for standalone testing.
//
// Design: DESIGN_multiwave_fa.md. Motivation: 2-C measured fa_robust 8.5-15×
// slower than production fused SDPA, localized to 1-wave/block idling 48/64
// lanes during softmax. Multi-wave: W waves/block, each owns Br=16 query rows,
// shares K/V LDS, softmax over full 64 lanes (zero-LDS DPP).
//
// Compile (from research/flash-attention-mi300x/):
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../../cdna3-prims/include fa_multiwave.hip -o fa_multiwave
// ---------------------------------------------------------------------------
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

// The kernel: one source of truth in cdna3-prims.
#include "../../cdna3-prims/include/cdna3/detail/fa_prefill_kernel.hpp"

static constexpr int kWarmup = 10;
static constexpr int kIter   = 100;

// Alias the kernel config struct to the name used by the harness below.
using Cfg = PrefillKCfg;

// ---------------------------------------------------------------------------
// CPU reference (GQA + causal)
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

    int sblocks = c.seqQ/(W*_Br);
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
    printf("Multi-wave FA (Stage-B cdna3-prims, W waves/block, DPP softmax, MFMA via cdna3::mfma)\n");
    printf("cdna3-prims version: %d.%d\n\n", cdna3::VERSION_MAJOR, cdna3::VERSION_MINOR);

    printf("--- correctness (small configs, CPU-verified) ---\n");
    run_cfg<64, 2>({0,256,256,2,2,1,0}, "MHA  full   seq256 h2", true);
    run_cfg<128,2>({0,256,256,2,2,1,0}, "MHA  full   seq256 h2", true);
    run_cfg<64, 4>({0,256,256,2,2,1,0}, "MHA  full   seq256 h2", true);
    run_cfg<128,2>({0,256,256,4,2,1,0}, "GQA  full   seq256 4q2kv", true);
    run_cfg<128,2>({0,256,256,2,2,1,1}, "MHA  causal seq256 h2", true);
    run_cfg<128,2>({0,256,256,4,2,1,1}, "GQA  causal seq256 4q2kv", true);

    printf("\n--- multi-wave seqLen sweep (D=128, MHA, full) --- compare to fa_robust ---\n");
    const int seqs[6]={512,1024,1536,2048,3072,4096};
    for(int i=0;i<6;++i){ char l[40]; snprintf(l,sizeof(l),"MHA full seq%d",seqs[i]);
        run_cfg<128,2>({0,seqs[i],seqs[i],8,8,1,0}, l, false); }

    printf("\n--- W=4 variant (D=64, fits LDS) ---\n");
    run_cfg<64,4>({0,2048,2048,8,8,1,0}, "MHA full seq2k D64", false);

    printf("\nNotes: causal TFLOPS uses 0.5x work. Compare medians to fa_robust same config.\n");
    printf("LDS budget: W=2 D=128 fits; W=4 needs D=64 (see DESIGN_multiwave_fa.md).\n");
    return 0;
}

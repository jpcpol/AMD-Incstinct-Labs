// ---------------------------------------------------------------------------
// fa_decode.hip — split-KV decode Flash Attention benchmark driver.
//
// Stage-B: the kernel lives in cdna3-prims/include/cdna3/detail/
// fa_decode_kernel.hpp (single source of truth). This file adds the CPU
// reference, launcher/timing harness, and main() for standalone testing.
//
// Design: DESIGN_decode_kvcache.md. 1 query × N cache keys, memory-bound.
// Split-KV (Axis-B): W waves split N keys, each partial (m,l,O), combine
// in LDS. DPP for q·K dot product (zero-LDS reduce_sum_bcast).
//
// Compile (from research/flash-attention-mi300x/):
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../../cdna3-prims/include fa_decode.hip -o fa_decode
// ---------------------------------------------------------------------------
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

// The kernel: one source of truth in cdna3-prims.
#include "../../cdna3-prims/include/cdna3/detail/fa_decode_kernel.hpp"

static constexpr int kWarmup = 10;
static constexpr int kIter   = 200;

// Alias to keep the harness readable.
using DCfg = DecodeKCfg;

// ---------------------------------------------------------------------------
// CPU reference: 1 query × N keys, GQA
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
    printf("Decode FA (Stage-B cdna3-prims, split-KV, DPP dot+softmax via wave::dpp::)\n");
    printf("cdna3-prims version: %d.%d\n\n", cdna3::VERSION_MAJOR, cdna3::VERSION_MINOR);

    printf("--- correctness (CPU-verified) ---\n");
    run_decode<64, 4>({64, 128, 8, 8}, "MHA  D64  N128",  true);
    run_decode<128,4>({128,128, 8, 8}, "MHA  D128 N128",  true);
    run_decode<128,4>({128,512, 32,8}, "GQA  D128 N512 32q8kv", true);
    run_decode<64, 2>({64, 256, 8, 8}, "MHA  D64  N256 W2", true);

    printf("\n--- N-sweep (D=128, GQA 32q/8kv, W=4) — expect ~linear in N ---\n");
    const int Ns[6]={256,512,1024,2048,4096,8192};
    for(int i=0;i<6;++i){ char l[40]; snprintf(l,sizeof(l),"decode N%d",Ns[i]);
        run_decode<128,4>({128,Ns[i],32,8}, l, false); }

    printf("\n--- split-KV scaling (D=128, GQA, N=4096): W=1 vs W=4 ---\n");
    run_decode<128,1>({128,4096,32,8}, "decode W1", false);
    run_decode<128,4>({128,4096,32,8}, "decode W4", false);

    printf("\nNotes: decode is memory-bound (1×N). err threshold 0.15 (fp16).\n");
    return 0;
}

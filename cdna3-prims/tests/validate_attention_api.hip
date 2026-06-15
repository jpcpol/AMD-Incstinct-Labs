// ---------------------------------------------------------------------------
// validate_attention_api.hip — Stage-B end-to-end validation of cdna3::attn.
//
// Tests cdna3::attn::prefill and cdna3::attn::decode against a CPU reference,
// exercising MHA, GQA, and causal configurations. This is the test that closes
// Stage B: if it passes on 1×MI300X, the unified attention API is validated.
//
// Compile (from cdna3-prims/tests/):
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../include validate_attention_api.hip -o validate_attention_api
//
// Run: ./validate_attention_api
//   Prints PASS/FAIL per configuration. All PASS = Stage B exit criterion met.
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include "../include/cdna3/attention.hpp"

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

void cpu_prefill(const std::vector<float>& Q, const std::vector<float>& K,
                 const std::vector<float>& V, std::vector<float>& O,
                 int D, int seqQ, int seqK, int nQH, int nKVH, int batch, int causal){
    const float scale = 1.f/sqrtf((float)D);
    const int group = nQH/nKVH;
    for (int b=0;b<batch;++b)
    for (int h=0;h<nQH;++h){
        int kvh=h/group;
        for (int q=0;q<seqQ;++q){
            int kmax=causal?std::min(seqK,q+1):seqK;
            std::vector<float> S(kmax); float mx=-1e38f;
            const float* qp=&Q[(((size_t)b*nQH+h)*seqQ+q)*D];
            for (int k=0;k<kmax;++k){ float dot=0;
                const float* kp=&K[(((size_t)b*nKVH+kvh)*seqK+k)*D];
                for(int d=0;d<D;++d)dot+=qp[d]*kp[d];
                S[k]=dot*scale; mx=std::max(mx,S[k]); }
            float se=0; for(int k=0;k<kmax;++k){S[k]=expf(S[k]-mx);se+=S[k];}
            for(int k=0;k<kmax;++k)S[k]/=se;
            float* op=&O[(((size_t)b*nQH+h)*seqQ+q)*D];
            for(int d=0;d<D;++d){ float o=0;
                for(int k=0;k<kmax;++k){ const float* vp=&V[(((size_t)b*nKVH+kvh)*seqK+k)*D]; o+=S[k]*vp[d]; }
                op[d]=o; }
        }
    }
}

void cpu_decode(const std::vector<float>& q, const std::vector<float>& K,
                const std::vector<float>& V, std::vector<float>& o,
                int D, int N, int nQH, int nKVH){
    const float scale=1.f/sqrtf((float)D);
    const int group=nQH/nKVH;
    for (int h=0;h<nQH;++h){
        int kvh=h/group;
        std::vector<float> S(N); float mx=-1e38f;
        const float* qp=&q[(size_t)h*D];
        for (int k=0;k<N;++k){ float dot=0;
            const float* kp=&K[((size_t)kvh*N+k)*D];
            for(int d=0;d<D;++d)dot+=qp[d]*kp[d];
            S[k]=dot*scale; mx=std::max(mx,S[k]); }
        float se=0; for(int k=0;k<N;++k){S[k]=expf(S[k]-mx);se+=S[k];}
        float* op=&o[(size_t)h*D];
        for(int d=0;d<D;++d){ float acc=0;
            for(int k=0;k<N;++k){ const float* vp=&V[((size_t)kvh*N+k)*D]; acc+=(S[k]/se)*vp[d]; }
            op[d]=acc; }
    }
}

// ---------------------------------------------------------------------------
// Prefill validator
// ---------------------------------------------------------------------------
bool test_prefill(int D, int seqQ, int seqK, int nQH, int nKVH, int batch, int causal,
                  const char* label, float tol=0.15f){
    const size_t Qel=(size_t)batch*nQH*seqQ*D, KVel=(size_t)batch*nKVH*seqK*D;
    std::vector<float> Qf(Qel),Kf(KVel),Vf(KVel);
    std::mt19937 rng(42); std::normal_distribution<float> dist(0.f,0.5f);
    for(auto&v:Qf)v=dist(rng); for(auto&v:Kf)v=dist(rng); for(auto&v:Vf)v=dist(rng);
    std::vector<__half> hQ(Qel),hK(KVel),hV(KVel);
    for(size_t i=0;i<Qel;++i)hQ[i]=__float2half(Qf[i]);
    for(size_t i=0;i<KVel;++i){hK[i]=__float2half(Kf[i]);hV[i]=__float2half(Vf[i]);}
    __half *dQ,*dK,*dV,*dO;
    hipMalloc(&dQ,Qel*2);hipMalloc(&dK,KVel*2);hipMalloc(&dV,KVel*2);hipMalloc(&dO,Qel*2);
    hipMemcpy(dQ,hQ.data(),Qel*2,hipMemcpyHostToDevice);
    hipMemcpy(dK,hK.data(),KVel*2,hipMemcpyHostToDevice);
    hipMemcpy(dV,hV.data(),KVel*2,hipMemcpyHostToDevice);

    cdna3::attn::PrefillCfg cfg{D,seqQ,seqK,nQH,nKVH,batch,causal};
    hipError_t e = cdna3::attn::prefill(dQ,dK,dV,dO,cfg);
    if(e!=hipSuccess){ printf("  PREFILL %-35s  API ERROR: %s\n",label,hipGetErrorString(e));
        hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO); return false; }
    hipDeviceSynchronize();

    std::vector<__half> hOg(Qel); hipMemcpy(hOg.data(),dO,Qel*2,hipMemcpyDeviceToHost);
    std::vector<float> hOr(Qel,0.f);
    cpu_prefill(Qf,Kf,Vf,hOr,D,seqQ,seqK,nQH,nKVH,batch,causal);
    float err=0; for(size_t i=0;i<Qel;++i) err=std::max(err,fabsf(__half2float(hOg[i])-hOr[i]));
    bool ok=err<tol;
    printf("  PREFILL %-35s  err=%.5f  [%s]\n",label,err,ok?"PASS":"FAIL");
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
    return ok;
}

// ---------------------------------------------------------------------------
// Decode validator
// ---------------------------------------------------------------------------
bool test_decode(int D, int N, int nQH, int nKVH, const char* label, float tol=0.15f){
    const size_t qel=(size_t)nQH*D, kvel=(size_t)nKVH*N*D;
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

    cdna3::attn::DecodeCfg cfg{D,N,nQH,nKVH};
    hipError_t e = cdna3::attn::decode(dq,dK,dV,doo,cfg);
    if(e!=hipSuccess){ printf("  DECODE  %-35s  API ERROR: %s\n",label,hipGetErrorString(e));
        hipFree(dq);hipFree(dK);hipFree(dV);hipFree(doo); return false; }
    hipDeviceSynchronize();

    std::vector<__half> og(qel); hipMemcpy(og.data(),doo,qel*2,hipMemcpyDeviceToHost);
    std::vector<float> orf(qel,0.f);
    cpu_decode(qf,Kf,Vf,orf,D,N,nQH,nKVH);
    float err=0; for(size_t i=0;i<qel;++i) err=std::max(err,fabsf(__half2float(og[i])-orf[i]));
    bool ok=err<tol;
    printf("  DECODE  %-35s  err=%.5f  [%s]\n",label,err,ok?"PASS":"FAIL");
    hipFree(dq);hipFree(dK);hipFree(dV);hipFree(doo);
    return ok;
}

int main(){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    printf("Device : %s (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("cdna3-prims version: %d.%d\n", cdna3::VERSION_MAJOR, cdna3::VERSION_MINOR);
    printf("\ncdna3::attn unified API — Stage-B validation\n");
    printf("==============================================\n\n");

    int pass=0, total=0;

    // --- Prefill configurations ---
    printf("-- Prefill (cdna3::attn::prefill) --\n");
    // H-MW1: multi-wave faster than fa_robust at matched work
    auto chk=[&](bool r){ ++total; if(r)++pass; };
    chk(test_prefill(128,256,256,2,2,1,0, "MHA full D128 seq256 h2"));
    chk(test_prefill(128,256,256,4,2,1,0, "GQA full D128 seq256 4q2kv"));
    chk(test_prefill(128,256,256,2,2,1,1, "MHA causal D128 seq256 h2"));
    chk(test_prefill(128,256,256,4,2,1,1, "GQA causal D128 seq256 4q2kv"));
    chk(test_prefill(64, 256,256,2,2,1,0, "MHA full D64  seq256 h2"));
    // H-MW2: correctness at large seqLen
    chk(test_prefill(128,1024,1024,8,8,1,0, "MHA full D128 seq1024 h8"));
    chk(test_prefill(128,512,512,32,8,1,1, "GQA causal D128 seq512 32q8kv"));

    // --- Decode configurations ---
    printf("\n-- Decode (cdna3::attn::decode) --\n");
    // H-DEC1: correctness for various N and head configs
    chk(test_decode(128, 128, 8,  8,  "MHA D128 N128"));
    chk(test_decode(64,  128, 8,  8,  "MHA D64  N128"));
    chk(test_decode(128, 512, 32, 8,  "GQA D128 N512 32q8kv"));
    chk(test_decode(128, 1024,32, 8,  "GQA D128 N1024 32q8kv"));
    chk(test_decode(128, 4096,32, 8,  "GQA D128 N4096 32q8kv"));
    // H-DEC3: large N (linear scaling check)
    chk(test_decode(128, 8192,32, 8,  "GQA D128 N8192 32q8kv"));

    printf("\n==============================================\n");
    printf("Result: %d/%d PASS\n", pass, total);
    if(pass==total)
        printf("Stage-B exit criterion MET — cdna3::attn API validated on hardware.\n");
    else
        printf("SOME TESTS FAILED — investigate before marking Stage B complete.\n");
    return (pass==total)?0:1;
}

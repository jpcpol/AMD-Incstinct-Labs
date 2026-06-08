// ---------------------------------------------------------------------------
// validate_attn_layer.hip — Step 2-B: validate the HIP FA kernel against ONE real
// attention layer of a HF model (weights exported by export_attn_layer.py).
//
// Pipeline reconstructed on GPU:
//   x [seq,H]
//     -> Q = x·Wqᵀ  [seq, nH*D]      (simple GEMM, correctness not speed)
//     -> K = x·Wkᵀ  [seq, nKV*D]
//     -> V = x·Wvᵀ  [seq, nKV*D]
//     -> RoPE(Q), RoPE(K)            (using exported cos/sin)
//     -> O_attn = FlashAttention(Q,K,V, causal, GQA)   <-- the kernel under test
//     -> O = O_attn·Woᵀ  [seq, H]
//   compare O against ref [seq,H] (fp32) from PyTorch.
//
// The attention core is the verified 1-wave/block MFMA design from fa_mfma2/fa_robust.
// Projections use a naive GEMM — this validates the ATTENTION numerics with real
// weights + RoPE + GQA, which is the point of Step 2-B.
//
// Reads: meta.json (parsed minimally here via fixed filenames + CLI dims) and the
// .bin tensors from export_attn_layer.py.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 validate_attn_layer.hip -o validate_attn_layer
// Run:
//   ./validate_attn_layer <dump_dir> <seq> <hidden> <nH> <nKV> <D>
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>

static constexpr int Br = 16;
static constexpr int Bc = 64;
static constexpr int MT = 16;

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;
__device__ __forceinline__ __fp16 h2f16(__half h){ return *reinterpret_cast<__fp16*>(&h); }
__device__ __forceinline__ f32x4 mfma(fp16x4 a, fp16x4 b, f32x4 c){ return __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,c,0,0,0); }

// --- naive GEMM: Y[M,N] = X[M,K] · Wᵀ where W is [N,K] (HF Linear layout) ---
// One thread per output element. Correctness-only.
__global__ void gemm_xwT(const __half* X, const __half* W, __half* Y, int M, int N, int K){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row>=M || col>=N) return;
    float acc=0.f;
    for (int k=0;k<K;++k) acc += __half2float(X[row*K+k]) * __half2float(W[col*K+k]);
    Y[row*N+col] = __float2half(acc);
}

// --- RoPE in place on a [seq, heads*D] tensor (cos/sin are [seq, D]) ---
__global__ void rope_kernel(__half* T, const float* cos, const float* sin,
                            int seq, int heads, int D){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;       // over seq*heads*D
    int total = seq*heads*D;
    if (idx>=total) return;
    int d = idx % D;
    int s = (idx / (heads*D));
    // rotate-half: out[d] = t[d]*cos[d] + rot[d]*sin[d], rot = (-t[d+D/2], t[d-D/2])
    int base = idx - d;            // start of this (s,head) row
    float c = cos[s*D + d], sn = sin[s*D + d];
    float t  = __half2float(T[base + d]);
    float tr;
    if (d < D/2) tr = -__half2float(T[base + d + D/2]);
    else         tr =  __half2float(T[base + d - D/2]);
    T[base + d] = __float2half(t*c + tr*sn);
}

// --- Flash Attention, 1 wave/block, GQA + causal (from fa_robust) ---
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

    // Q [seq, nQH, D] row-major over (seq, head); here laid [head? no] -> we store
    // Q as [nQH, seq, D] for coalesced head-major access. See host packing.
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

// --- host helpers ---
static std::vector<__half> load_h(const std::string& p, size_t n){
    std::vector<__half> v(n); FILE* f=fopen(p.c_str(),"rb");
    if(!f){ printf("cannot open %s\n",p.c_str()); exit(1);} fread(v.data(),2,n,f); fclose(f); return v;
}
static std::vector<float> load_f(const std::string& p, size_t n){
    std::vector<float> v(n); FILE* f=fopen(p.c_str(),"rb");
    if(!f){ printf("cannot open %s\n",p.c_str()); exit(1);} fread(v.data(),4,n,f); fclose(f); return v;
}

template<int D>
int run(const std::string& dir, int seq, int H, int nH, int nKV){
    auto P=[&](const char* n){ return dir+"/"+n+std::string(".bin"); };
    size_t QKV_o = (size_t)nH*D, KV_o=(size_t)nKV*D;
    auto x   = load_h(P("x"),   (size_t)seq*H);
    auto qw  = load_h(P("q_w"), QKV_o*H);   // [nH*D, H]
    auto kw  = load_h(P("k_w"), KV_o*H);
    auto vw  = load_h(P("v_w"), KV_o*H);
    auto ow  = load_h(P("o_w"), (size_t)H*(nH*D));
    auto cs  = load_f(P("cos"), (size_t)seq*D);
    auto sn  = load_f(P("sin"), (size_t)seq*D);
    auto ref = load_f(P("ref"), (size_t)seq*H);

    __half *dX,*dQW,*dKW,*dVW,*dOW,*dQ,*dK,*dV,*dAttn,*dO;
    float *dCos,*dSin;
    hipMalloc(&dX,x.size()*2); hipMemcpy(dX,x.data(),x.size()*2,hipMemcpyHostToDevice);
    hipMalloc(&dQW,qw.size()*2);hipMemcpy(dQW,qw.data(),qw.size()*2,hipMemcpyHostToDevice);
    hipMalloc(&dKW,kw.size()*2);hipMemcpy(dKW,kw.data(),kw.size()*2,hipMemcpyHostToDevice);
    hipMalloc(&dVW,vw.size()*2);hipMemcpy(dVW,vw.data(),vw.size()*2,hipMemcpyHostToDevice);
    hipMalloc(&dOW,ow.size()*2);hipMemcpy(dOW,ow.data(),ow.size()*2,hipMemcpyHostToDevice);
    hipMalloc(&dCos,cs.size()*4);hipMemcpy(dCos,cs.data(),cs.size()*4,hipMemcpyHostToDevice);
    hipMalloc(&dSin,sn.size()*4);hipMemcpy(dSin,sn.data(),sn.size()*4,hipMemcpyHostToDevice);
    hipMalloc(&dQ,(size_t)seq*nH*D*2);
    hipMalloc(&dK,(size_t)seq*nKV*D*2);
    hipMalloc(&dV,(size_t)seq*nKV*D*2);
    hipMalloc(&dAttn,(size_t)seq*nH*D*2);
    hipMalloc(&dO,(size_t)seq*H*2);

    // Q/K/V projections (output is [seq, heads*D], "seq-major")
    dim3 tb(16,16);
    auto gemm=[&](const __half* W, __half* Y, int N){
        dim3 gr((N+15)/16,(seq+15)/16); hipLaunchKernelGGL(gemm_xwT,gr,tb,0,0,dX,W,Y,seq,N,H); };
    gemm(dQW,dQ,nH*D); gemm(dKW,dK,nKV*D); gemm(dVW,dV,nKV*D);
    hipDeviceSynchronize();

    // RoPE on Q and K (heads-major view: [seq, heads, D] == seq-major contiguous)
    int thr=256;
    hipLaunchKernelGGL(rope_kernel,dim3((seq*nH*D+thr-1)/thr),dim3(thr),0,0,dQ,dCos,dSin,seq,nH,D);
    hipLaunchKernelGGL(rope_kernel,dim3((seq*nKV*D+thr-1)/thr),dim3(thr),0,0,dK,dCos,dSin,seq,nKV,D);
    hipDeviceSynchronize();

    // DIAG: compare Q post-proj+RoPE (seq-major [seq,nH*D]) vs q_diag.bin from export.
    // Localizes proj/RoPE bug vs attention/repack bug.
    {
        std::string qd = dir+"/q_diag.bin";
        FILE* f=fopen(qd.c_str(),"rb");
        if(f){
            std::vector<float> qref((size_t)seq*nH*D); fread(qref.data(),4,qref.size(),f); fclose(f);
            std::vector<__half> qgpu((size_t)seq*nH*D); hipMemcpy(qgpu.data(),dQ,qgpu.size()*2,hipMemcpyDeviceToHost);
            float me=0,mr=0;
            for(size_t i=0;i<qref.size();++i){ float g=__half2float(qgpu[i]),r=qref[i];
                float e=fabsf(g-r); me=std::max(me,e); mr=std::max(mr,e/(fabsf(r)+1e-3f)); }
            printf("  [DIAG] Q post-proj+RoPE: max|gpu-ref|=%.4f rel=%.4f %s\n",
                   me, mr, mr<0.05f?"proj/RoPE OK -> bug in attn/repack":"BUG in proj/RoPE");
        }
    }

    // Repack Q [seq,nH,D] -> [nH,seq,D] (head-major) for the FA kernel; same for K/V.
    // Do it on host for simplicity (B is correctness, not speed).
    auto repack=[&](__half* dsrc,int heads)->__half*{
        std::vector<__half> tmp((size_t)seq*heads*D);
        hipMemcpy(tmp.data(),dsrc,tmp.size()*2,hipMemcpyDeviceToHost);
        std::vector<__half> out((size_t)heads*seq*D);
        for(int s=0;s<seq;++s)for(int h=0;h<heads;++h)for(int d=0;d<D;++d)
            out[((size_t)h*seq+s)*D+d]=tmp[((size_t)s*heads+h)*D+d];
        __half* dd; hipMalloc(&dd,out.size()*2); hipMemcpy(dd,out.data(),out.size()*2,hipMemcpyHostToDevice);
        return dd;
    };
    __half* dQh=repack(dQ,nH); __half* dKh=repack(dK,nKV); __half* dVh=repack(dV,nKV);

    // Flash Attention (head-major), output dAttnH [nH, seq, D]
    __half* dAttnH; hipMalloc(&dAttnH,(size_t)nH*seq*D*2);
    int blocks=nH*(seq/Br);
    hipLaunchKernelGGL(fa_kernel<D>,dim3(blocks),dim3(64),0,0,dQh,dKh,dVh,dAttnH,seq,nH,nKV);
    hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){ printf("FA KERNEL ERROR: %s\n",hipGetErrorString(e)); return 1; }

    // Repack attn output [nH,seq,D] -> [seq,nH*D] for the O projection
    {
        std::vector<__half> tmp((size_t)nH*seq*D);
        hipMemcpy(tmp.data(),dAttnH,tmp.size()*2,hipMemcpyDeviceToHost);
        std::vector<__half> out((size_t)seq*nH*D);
        for(int h=0;h<nH;++h)for(int s=0;s<seq;++s)for(int d=0;d<D;++d)
            out[((size_t)s*nH+h)*D+d]=tmp[((size_t)h*seq+s)*D+d];
        hipMemcpy(dAttn,out.data(),out.size()*2,hipMemcpyHostToDevice);
    }

    // O projection: O[seq,H] = Attn[seq, nH*D] · Woᵀ  (Wo is [H, nH*D])
    { dim3 gr((H+15)/16,(seq+15)/16); hipLaunchKernelGGL(gemm_xwT,gr,tb,0,0,dAttn,dOW,dO,seq,H,nH*D); }
    hipDeviceSynchronize();

    std::vector<__half> hO((size_t)seq*H); hipMemcpy(hO.data(),dO,hO.size()*2,hipMemcpyDeviceToHost);
    float maxerr=0,maxrel=0;
    for(size_t i=0;i<(size_t)seq*H;++i){
        float g=__half2float(hO[i]), r=ref[i];
        float e=fabsf(g-r); maxerr=std::max(maxerr,e);
        float rel=e/(fabsf(r)+1e-3f); maxrel=std::max(maxrel,rel);
    }
    printf("--- Step 2-B: real attention layer validation (D=%d) ---\n",D);
    printf("  seq=%d H=%d nH=%d nKV=%d (GQA group=%d) causal=yes\n",seq,H,nH,nKV,nH/nKV);
    printf("  max|gpu-ref| = %.4f   max rel err = %.4f   [%s]\n",
           maxerr, maxrel, maxrel<0.05f ? "PASS" : "CHECK");
    printf("  (fp16 kernel vs fp32 ref through full layer incl. naive GEMM projections)\n");
    return 0;
}

int main(int argc,char**argv){
    if(argc<7){ printf("usage: %s <dir> <seq> <hidden> <nH> <nKV> <D>\n",argv[0]); return 1; }
    std::string dir=argv[1];
    int seq=atoi(argv[2]),H=atoi(argv[3]),nH=atoi(argv[4]),nKV=atoi(argv[5]),D=atoi(argv[6]);
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop,0);
    printf("Device: %s (gfx%d)\n",prop.name,prop.major*100+prop.minor);
    if(D==64)  return run<64>(dir,seq,H,nH,nKV);
    if(D==128) return run<128>(dir,seq,H,nH,nKV);
    printf("D=%d not compiled (only 64/128).\n",D); return 1;
}

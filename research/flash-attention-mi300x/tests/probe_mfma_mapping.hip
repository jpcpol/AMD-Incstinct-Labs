// ---------------------------------------------------------------------------
// probe_mfma_mapping.hip — empirically determine the v_mfma_f32_16x16x16f16
// wave64 register-to-matrix lane mapping on CDNA3 (gfx942).
//
// Runs a full 16×16×16 GEMM (C = A·B) with positional data and compares against
// a host reference, enumerating all 4 combinations of {A layout, B layout} ×
// {output layout}. Only one combination reproduces the reference.
//
// VERIFIED RESULT (MI300X VF, ROCm 7.2, 2026-06-05):
//   A input : lane L, frag f → A[row = L%16      ][k   = (L/16)*4 + f]   ([idx16][quad])
//   B input : lane L, frag f → B[k   = (L/16)*4+f][col = L%16        ]   ([quad][idx16])
//   C/D out : lane L, frag f → C[row = (L/16)*4+f][col = L%16        ]   ([quad][idx16])
//   → match max_err = 0.001; all other combinations err > 1.7.
//
// Compile:
//   hipcc -O2 --offload-arch=gfx942 -std=c++17 probe_mfma_mapping.hip -o probe_mfma_mapping
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;
__device__ __fp16 cvt(__half h){ return *reinterpret_cast<__fp16*>(&h); }

// AM/BM: 0 = read matrix as [idx16][quad], 1 = read as [quad][idx16]
template<int AM, int BM>
__global__ void gemm(const __half* gA, const __half* gB, float* gOut){
    int L = threadIdx.x; fp16x4 A, B;
    for(int f=0;f<4;++f){
        int idx16 = L%16, quad = (L/16)*4+f;
        int ar=(AM==0)?idx16:quad, ak=(AM==0)?quad:idx16;
        int br=(BM==0)?idx16:quad, bk=(BM==0)?quad:idx16;
        A[f]=cvt(gA[ar*16+ak]); B[f]=cvt(gB[br*16+bk]);
    }
    f32x4 C={0,0,0,0};
    f32x4 D=__builtin_amdgcn_mfma_f32_16x16x16f16(A,B,C,0,0,0);
    for(int f=0;f<4;++f) gOut[L*4+f]=D[f];
}

int main(){
    float A[256],B[256],Cref[256];
    for(int i=0;i<256;++i){A[i]=(i*7%13)*0.1f; B[i]=(i*5%11)*0.1f;}
    for(int r=0;r<16;++r)for(int c=0;c<16;++c){
        float s=0; for(int k=0;k<16;++k) s+=A[r*16+k]*B[k*16+c];
        Cref[r*16+c]=s;
    }
    __half hA[256],hB[256];
    for(int i=0;i<256;++i){hA[i]=__float2half(A[i]); hB[i]=__float2half(B[i]);}
    __half *dA,*dB; float* dO;
    hipMalloc(&dA,512); hipMalloc(&dB,512); hipMalloc(&dO,1024);
    hipMemcpy(dA,hA,512,hipMemcpyHostToDevice);
    hipMemcpy(dB,hB,512,hipMemcpyHostToDevice);
    float out[256];

    auto run=[&](auto kern,const char* an,const char* bn){
        hipLaunchKernelGGL(kern,dim3(1),dim3(64),0,0,dA,dB,dO);
        hipDeviceSynchronize();
        hipMemcpy(out,dO,1024,hipMemcpyDeviceToHost);
        for(int om=0;om<2;++om){
            float err=0;
            for(int L=0;L<64;++L)for(int f=0;f<4;++f){
                int idx16=L%16,quad=(L/16)*4+f,r,c;
                if(om==0){r=idx16;c=quad;}else{r=quad;c=idx16;}
                err=fmaxf(err,fabsf(out[L*4+f]-Cref[r*16+c]));
            }
            printf("  A=%s B=%s out=%s: err=%.4f %s\n",
                   an,bn,om==0?"[i][q]":"[q][i]",err,err<0.1f?"<== MATCH":"");
        }
    };
    run(gemm<0,0>,"[i][q]","[i][q]");
    run(gemm<0,1>,"[i][q]","[q][i]");
    run(gemm<1,0>,"[q][i]","[i][q]");
    run(gemm<1,1>,"[q][i]","[q][i]");
    hipFree(dA); hipFree(dB); hipFree(dO);
    return 0;
}

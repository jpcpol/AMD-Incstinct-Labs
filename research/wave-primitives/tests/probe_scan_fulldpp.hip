// Validate the full-DPP wave64 inclusive scan (0 LDS). Cross-row carry built
// from a cascade of row_bcast15 over the row totals — never re-reading a mutated
// v, so no double counting. Input lane=1 → inclusive lane i must = i+1.
#include "wave_primitives/detail/config.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ __forceinline__ unsigned lane(){ return __builtin_amdgcn_mbcnt_hi(~0u,__builtin_amdgcn_mbcnt_lo(~0u,0u)); }
template <int C> __device__ __forceinline__ float dpp(float old,float v){ return __builtin_amdgcn_update_dpp(old,v,C,0xf,0xf,false); }
template <int Sh> __device__ __forceinline__ float shr_add(float v){ return v + dpp<0x110+Sh>(0.0f,v); }

__global__ void probe(float* out) {
    float v = 1.0f;
    v = shr_add<1>(v); v = shr_add<2>(v); v = shr_add<4>(v); v = shr_add<8>(v);
    float c1  = dpp<0x142>(0.0f, v);
    float c1b = dpp<0x142>(0.0f, c1);
    float c1c = dpp<0x142>(0.0f, c1b);
    float carry = 0.0f;
    unsigned L = lane();
    if (L >= 16) carry = c1;
    if (L >= 32) carry += c1b;
    if (L >= 48) carry += c1c;
    out[L] = (L >= 16) ? v + carry : v;
}

int main(){
    float *d; hipMalloc(&d,64*4);
    hipLaunchKernelGGL(probe,dim3(1),dim3(64),0,0,d);
    hipDeviceSynchronize();
    float h[64]; hipMemcpy(h,d,64*4,hipMemcpyDeviceToHost);
    int fail=-1; for(int i=0;i<64;i++) if(h[i]!=(float)(i+1)){fail=i;break;}
    if(fail<0) printf("ALL PASS (full-DPP scan, 0 LDS)\n");
    else { printf("FAIL lane %d got %.1f exp %d\n",fail,h[fail],fail+1);
           for(int i=12;i<=52;i++) printf("   %2d: %5.1f (exp %d)\n",i,h[i],i+1); }
    return 0;
}

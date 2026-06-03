// Validate the generalized DPP scan lane-by-lane across type and op.
//   sum: input lane=1 → inclusive lane i = i+1
//   max: input lane=i → inclusive lane i = i
//   min: input lane=i → inclusive lane i = 0
// Covered: float, int32, double, int64. Lane-by-lane checking is essential —
// a "total only" check passes while intermediate lanes are wrong (this caught
// a cross-row carry double-count bug at lane 32).
#include "wave_primitives/wave_scan_dpp.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

template <typename T> __global__ void k_sum(T* o){ T v=(T)1;           o[threadIdx.x]=wave::dpp::scan_inclusive_sum(v); }
template <typename T> __global__ void k_max(T* o){ T v=(T)threadIdx.x; o[threadIdx.x]=wave::dpp::scan_inclusive_max(v); }
template <typename T> __global__ void k_min(T* o){ T v=(T)threadIdx.x; o[threadIdx.x]=wave::dpp::scan_inclusive_min(v); }

template <typename T>
int check(const char* tn) {
    T* d; hipMalloc(&d,64*sizeof(T));
    T h[64]; int fails=0;
    hipLaunchKernelGGL(k_sum<T>,dim3(1),dim3(64),0,0,d); hipDeviceSynchronize();
    hipMemcpy(h,d,64*sizeof(T),hipMemcpyDeviceToHost);
    for(int i=0;i<64;i++) if((double)h[i]!=(double)(i+1)){printf("  %s sum FAIL lane %d got %.1f exp %d\n",tn,i,(double)h[i],i+1);fails++;break;}
    hipLaunchKernelGGL(k_max<T>,dim3(1),dim3(64),0,0,d); hipDeviceSynchronize();
    hipMemcpy(h,d,64*sizeof(T),hipMemcpyDeviceToHost);
    for(int i=0;i<64;i++) if((double)h[i]!=(double)i){printf("  %s max FAIL lane %d got %.1f exp %d\n",tn,i,(double)h[i],i);fails++;break;}
    hipLaunchKernelGGL(k_min<T>,dim3(1),dim3(64),0,0,d); hipDeviceSynchronize();
    hipMemcpy(h,d,64*sizeof(T),hipMemcpyDeviceToHost);
    for(int i=0;i<64;i++) if((double)h[i]!=0.0){printf("  %s min FAIL lane %d got %.1f exp 0\n",tn,i,(double)h[i]);fails++;break;}
    if(fails==0) printf("  %s sum/max/min ALL PASS\n",tn);
    hipFree(d); return fails;
}

int main(){
    int f=0;
    f+=check<float>("f32");
    f+=check<int32_t>("i32");
    f+=check<double>("f64");
    f+=check<int64_t>("i64");
    printf(f==0?"\nALL TYPES/OPS PASS\n":"\n%d FAILURES\n",f);
    return f?1:0;
}

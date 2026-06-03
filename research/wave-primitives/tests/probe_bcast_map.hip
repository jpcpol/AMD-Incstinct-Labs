// Map the exact lane-movement of row_bcast15 (0x142) and row_bcast31 (0x143)
// on gfx942. Each lane starts = its id; we print what each lane RECEIVES.
// This map is the basis for the full-DPP scan cross-row carry.
//
// Verified result (gfx942):
//   bcast15: lanes 0-15 unchanged; 16-31 <- lane15; 32-47 <- lane31; 48-63 <- lane47
//   bcast31: lanes 0-31 unchanged; 32-63 <- lane31
#include <hip/hip_runtime.h>
#include <cstdio>
__device__ __forceinline__ unsigned lane(){ return __builtin_amdgcn_mbcnt_hi(~0u,__builtin_amdgcn_mbcnt_lo(~0u,0u)); }

__global__ void probe(float* b15, float* b31) {
    float v = (float)lane();
    b15[lane()] = __builtin_amdgcn_update_dpp(-1.0f, v, 0x142, 0xf, 0xf, false);
    b31[lane()] = __builtin_amdgcn_update_dpp(-1.0f, v, 0x143, 0xf, 0xf, false);
}
int main(){
    float *d0,*d1; hipMalloc(&d0,64*4); hipMalloc(&d1,64*4);
    hipLaunchKernelGGL(probe,dim3(1),dim3(64),0,0,d0,d1);
    hipDeviceSynchronize();
    float a[64],b[64];
    hipMemcpy(a,d0,64*4,hipMemcpyDeviceToHost);
    hipMemcpy(b,d1,64*4,hipMemcpyDeviceToHost);
    printf("lane : bcast15  bcast31   (value RECEIVED; -1 = read old)\n");
    for(int i=0;i<64;i++) printf("%3d  : %6.0f  %6.0f\n", i, a[i], b[i]);
    return 0;
}

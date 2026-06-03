// Probe DPP broadcast modes (row_bcast15 / row_bcast31) on gfx942.
// Each lane starts with value = lane_id; we print what each lane receives so
// the lane-movement mapping can be read directly. old = -1.0f marks lanes
// that read the 'old' operand (no in-range source).
#include <hip/hip_runtime.h>
#include <cstdio>
__device__ __forceinline__ unsigned lane() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
__global__ void probe(float* bcast15, float* bcast31, float* sl15) {
    float v = static_cast<float>(lane());
    bcast15[lane()] = __builtin_amdgcn_update_dpp(-1.0f, v, 0x142, 0xf, 0xf, false);
    bcast31[lane()] = __builtin_amdgcn_update_dpp(-1.0f, v, 0x143, 0xf, 0xf, false);
    sl15[lane()]    = __builtin_amdgcn_update_dpp(-1.0f, v, 0x100 + 15, 0xf, 0xf, false);
}
int main() {
    float *d0,*d1,*d2;
    hipMalloc(&d0,64*sizeof(float)); hipMalloc(&d1,64*sizeof(float)); hipMalloc(&d2,64*sizeof(float));
    hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, d0,d1,d2);
    hipDeviceSynchronize();
    float a[64],b[64],c[64];
    hipMemcpy(a,d0,64*sizeof(float),hipMemcpyDeviceToHost);
    hipMemcpy(b,d1,64*sizeof(float),hipMemcpyDeviceToHost);
    hipMemcpy(c,d2,64*sizeof(float),hipMemcpyDeviceToHost);
    printf("lane : bcast15   bcast31   shl15\n");
    for (int i=0;i<64;++i) printf("%3d  :  %5.0f    %5.0f    %5.0f\n", i, a[i], b[i], c[i]);
    return 0;
}

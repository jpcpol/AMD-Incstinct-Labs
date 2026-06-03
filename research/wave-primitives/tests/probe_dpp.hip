// Probe the exact lane-movement semantics of DPP modes on gfx942.
// Each lane starts with its lane_id as value. We apply one DPP op and print
// what each lane received, so we can read off the actual mapping instead of
// guessing the row_shr / row_shl direction.
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ __forceinline__ unsigned lane() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

__global__ void probe(float* out_shr, float* out_shl, float* out_shfldn) {
    float v = static_cast<float>(lane());
    // row_shr:1  (0x111)
    out_shr[lane()]    = __builtin_amdgcn_update_dpp(0.0f, v, 0x111, 0xf, 0xf, false);
    // row_shl:1  (0x101)
    out_shl[lane()]    = __builtin_amdgcn_update_dpp(0.0f, v, 0x101, 0xf, 0xf, false);
    // reference: __shfl_down by 1
    out_shfldn[lane()] = __shfl_down(v, 1, 64);
}

int main() {
    float *d_shr, *d_shl, *d_dn;
    hipMalloc(&d_shr, 64*sizeof(float));
    hipMalloc(&d_shl, 64*sizeof(float));
    hipMalloc(&d_dn,  64*sizeof(float));
    hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, d_shr, d_shl, d_dn);
    hipDeviceSynchronize();
    float shr[64], shl[64], dn[64];
    hipMemcpy(shr, d_shr, 64*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(shl, d_shl, 64*sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(dn,  d_dn,  64*sizeof(float), hipMemcpyDeviceToHost);
    printf("lane : row_shr:1  row_shl:1  shfl_down:1   (value each lane RECEIVED)\n");
    for (int i = 0; i < 64; ++i)
        printf("%3d  :   %5.0f      %5.0f      %5.0f\n", i, shr[i], shl[i], dn[i]);
    return 0;
}

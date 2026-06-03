// Validate a full-DPP wave64 reduce (no bpermute) and print the partial state
// after each stage so we can debug the bcast geometry empirically.
//
// Plan (accumulate toward lane 63, the high lane):
//   Stage A: row_shr 1,2,4,8 -> per-row sum lands on lanes 15,31,47,63.
//   Stage B: bcast15 then add -> propagate row sums upward across rows.
//   Stage C: bcast31 then add -> propagate across the 32-lane bank.
// Final total expected on lane 63.
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ __forceinline__ unsigned lane() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}
template <int C>
__device__ __forceinline__ float dpp(float old, float v) {
    return __builtin_amdgcn_update_dpp(old, v, C, 0xf, 0xf, false);
}

__global__ void probe(float* after_rows, float* after_b15, float* after_b31) {
    float v = static_cast<float>(lane());
    // Stage A: row_shr accumulate within rows -> sum on high lane of each row.
    v += dpp<0x111>(0.0f, v); // shr 1
    v += dpp<0x112>(0.0f, v); // shr 2
    v += dpp<0x114>(0.0f, v); // shr 4
    v += dpp<0x118>(0.0f, v); // shr 8
    after_rows[lane()] = v;   // expect lanes 15,31,47,63 hold row sums

    // Stage B: bcast15 brings lane15->[16..31], lane31->[32..47], lane47->[48..63].
    // Add it: lane31 += rowsum0, lane47 += rowsum1, lane63 += rowsum2.
    float b15 = dpp<0x142>(0.0f, v);
    v += b15;
    after_b15[lane()] = v;

    // Stage C: bcast31 brings lane31->[32..63]. Add: lane63 += (whatever lane31 holds).
    float b31 = dpp<0x143>(0.0f, v);
    v += b31;
    after_b31[lane()] = v;    // hope: lane 63 == 2016
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
    printf("Expected total = 2016. Showing key lanes:\n");
    printf("lane :  afterRows  afterB15  afterB31\n");
    for (int i : {0,15,16,31,32,47,48,63})
        printf("%3d  :  %8.0f  %8.0f  %8.0f\n", i, a[i], b[i], c[i]);
    return 0;
}

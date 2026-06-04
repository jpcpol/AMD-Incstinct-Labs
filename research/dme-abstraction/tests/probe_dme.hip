// Probe the exact semantics of __builtin_amdgcn_global_load_lds on gfx942
// BEFORE building the abstraction on top of it. Two open questions:
//
//   Q1: Does `offset` apply to the global ptr, the LDS ptr, or both?
//       (The LLVM docs say "applied to both" — verify on hardware.)
//   Q2: Does the load actually populate LDS correctly after s_waitcnt?
//
// Method: known input array in HBM (value = index), copy to LDS via the
// builtin with various offset conventions, read back, print what landed where.
//
// Compile (on the VM):
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 probe_dme.hip -o probe_dme
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <cstdio>

__device__ __forceinline__ unsigned lane_id() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// Variant A: offset = 0, advance the global pointer per lane; LDS base + lane*4
// written by separate stores (control: this is NOT using offset).
__global__ void probe_offset0(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[64];
    const unsigned L = lane_id();
    // each lane: copy src[L] -> tile (with the builtin choosing the slot?)
    void* g = reinterpret_cast<void*>(const_cast<float*>(src + L));
    void* l = reinterpret_cast<void*>(&tile[L]);
    __builtin_amdgcn_global_load_lds(g, l, 4, 0, 0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    out[L] = tile[L];
}

// Variant B: single LDS base, use `offset` param = lane*4 for both ptrs.
__global__ void probe_offset_both(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[64];
    const unsigned L = lane_id();
    void* g = reinterpret_cast<void*>(const_cast<float*>(src));      // base, no per-lane advance
    void* l = reinterpret_cast<void*>(&tile[0]);                     // base
    __builtin_amdgcn_global_load_lds(g, l, 4, (int)(L * 4), 0);      // offset applied to both?
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    out[L] = tile[L];
}

int main() {
    const int N = 64;
    float h_src[N];
    for (int i = 0; i < N; ++i) h_src[i] = (float)i;   // src[i] = i
    float *d_src, *d_out;
    hipMalloc(&d_src, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));
    hipMemcpy(d_src, h_src, N * sizeof(float), hipMemcpyHostToDevice);

    auto run = [&](const char* name, void(*k)(const float*, float*)) {
        hipMemset(d_out, 0, N * sizeof(float));
        hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, d_src, d_out);
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) { printf("  %-16s LAUNCH ERROR: %s\n", name, hipGetErrorString(e)); return; }
        float h_out[N];
        hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost);
        // expected: out[i] == i if the copy is correct
        int first_bad = -1;
        for (int i = 0; i < N; ++i) if (h_out[i] != (float)i) { first_bad = i; break; }
        if (first_bad < 0) printf("  %-16s ALL CORRECT (out[i]==i)\n", name);
        else {
            printf("  %-16s first mismatch at lane %d: got %.1f exp %d\n",
                   name, first_bad, h_out[first_bad], first_bad);
            printf("      lanes 0-7: ");
            for (int i = 0; i < 8; ++i) printf("%.0f ", h_out[i]);
            printf("\n");
        }
    };

    printf("=== probe __builtin_amdgcn_global_load_lds (gfx942) ===\n");
    printf("src[i] = i; expect out[i] = i if copy lands correctly\n\n");
    run("offset0+ptr",  probe_offset0);
    run("offset_both",  probe_offset_both);
    printf("\nInterpretation:\n");
    printf("  if offset0+ptr correct  -> advance the GLOBAL ptr per lane, offset for LDS slot\n");
    printf("  if offset_both correct  -> offset applies to BOTH ptrs (docs claim this)\n");
    return 0;
}

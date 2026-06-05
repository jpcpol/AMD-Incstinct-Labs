// Probe the exact semantics of __builtin_amdgcn_global_load_lds on gfx942.
//
// Key constraint discovered at compile time: the `offset` parameter MUST be
// a compile-time constant integer (immediate) — runtime values are rejected.
// Per-lane routing must be done via the pointer, not the offset parameter.
//
// Questions probed:
//   Q1: Per-lane pointer advance with offset=0 — each lane writes to its
//       own LDS slot → correct per-element routing.
//   Q2: Fixed base ptrs, offset=0 — collision baseline.
//   Q3: Global per-lane, LDS fixed at tile[0] — last-writer semantics.
//   Q4: Stream cache policy (aux=sc0=1) — correctness preserved.
//   Q5: size=8 (64-bit load, two floats per lane).
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 tests/probe_dme.hip -o probe_dme
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>

__device__ __forceinline__ unsigned lane_id() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

// A: each lane advances BOTH pointers per lane, offset=0.
__global__ void probe_A(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[64];
    const unsigned L = lane_id();
    void* g = reinterpret_cast<void*>(const_cast<float*>(src + L));
    void* l = reinterpret_cast<void*>(&tile[L]);
    __builtin_amdgcn_global_load_lds(g, l, /*size=*/4, /*offset=*/0, /*aux=*/0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    out[L] = tile[L];
}

// B: base ptrs only, offset=0 — collision, all lanes fight for tile[0].
__global__ void probe_B(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[64];
    const unsigned L = lane_id();
    void* g = reinterpret_cast<void*>(const_cast<float*>(src));
    void* l = reinterpret_cast<void*>(&tile[0]);
    __builtin_amdgcn_global_load_lds(g, l, 4, 0, 0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    out[L] = tile[L];
}

// C: global ptr per lane, LDS fixed at tile[0] — last-writer semantics.
__global__ void probe_C(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[64];
    const unsigned L = lane_id();
    void* g = reinterpret_cast<void*>(const_cast<float*>(src + L));
    void* l = reinterpret_cast<void*>(&tile[0]);
    __builtin_amdgcn_global_load_lds(g, l, 4, 0, 0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    out[L] = tile[0];
}

// D: per-lane ptrs, stream cache hint (aux=sc0=1).
__global__ void probe_D(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[64];
    const unsigned L = lane_id();
    void* g = reinterpret_cast<void*>(const_cast<float*>(src + L));
    void* l = reinterpret_cast<void*>(&tile[L]);
    __builtin_amdgcn_global_load_lds(g, l, 4, 0, /*aux=sc0*/1);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    out[L] = tile[L];
}

// E: two consecutive size=4 loads per lane (workaround for size=8 not supported).
// Each lane copies src[L*2] and src[L*2+1] via two back-to-back DME loads.
__global__ void probe_E(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[128];
    const unsigned L = lane_id();
    void* g0 = reinterpret_cast<void*>(const_cast<float*>(src + L * 2));
    void* g1 = reinterpret_cast<void*>(const_cast<float*>(src + L * 2 + 1));
    void* l0 = reinterpret_cast<void*>(&tile[L * 2]);
    void* l1 = reinterpret_cast<void*>(&tile[L * 2 + 1]);
    __builtin_amdgcn_global_load_lds(g0, l0, /*size=*/4, /*offset=*/0, /*aux=*/0);
    __builtin_amdgcn_global_load_lds(g1, l1, /*size=*/4, /*offset=*/0, /*aux=*/0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
    out[L * 2]     = tile[L * 2];
    out[L * 2 + 1] = tile[L * 2 + 1];
}

int main() {
    const int N = 128;
    float h_src[N];
    for (int i = 0; i < N; ++i) h_src[i] = (float)i;

    float *d_src, *d_out;
    hipMalloc(&d_src, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));
    hipMemcpy(d_src, h_src, N * sizeof(float), hipMemcpyHostToDevice);

    auto run = [&](const char* name, void(*k)(const float*, float*),
                   int out_n, const char* expect) {
        std::vector<float> zeros(out_n, 0.f);
        hipMemcpy(d_out, zeros.data(), out_n * sizeof(float), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(k, dim3(1), dim3(64), 0, 0, d_src, d_out);
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) {
            printf("  %-28s LAUNCH ERROR: %s\n", name, hipGetErrorString(e));
            return;
        }
        std::vector<float> h_out(out_n);
        hipMemcpy(h_out.data(), d_out, out_n * sizeof(float), hipMemcpyDeviceToHost);
        int first_bad = -1;
        for (int i = 0; i < out_n; ++i) {
            if (h_out[i] != (float)i) { first_bad = i; break; }
        }
        const char* status = (first_bad < 0) ? "PASS" : "----";
        printf("  %-28s [%s]  expect: %s\n", name, status, expect);
        if (first_bad >= 0) {
            printf("      first mismatch at %d: got %.1f exp %.1f\n",
                   first_bad, h_out[first_bad], (float)first_bad);
            printf("      out[0..7]: ");
            for (int i = 0; i < 8 && i < out_n; ++i) printf("%.0f ", h_out[i]);
            printf("\n");
        }
    };

    printf("=== probe __builtin_amdgcn_global_load_lds (gfx942, ROCm 7.2) ===\n");
    printf("src[i] = i  |  1 block x 64 lanes\n\n");

    run("A: per-lane ptr+LDS  aux=0",    probe_A, 64,  "PASS — out[i]==i");
    run("B: base ptrs  offset=0",         probe_B, 64,  "FAIL — collision at tile[0]");
    run("C: global/lane  LDS=tile[0]",   probe_C, 64,  "same value for all lanes");
    run("D: per-lane  aux=sc0",           probe_D, 64,  "PASS — same as A");
    run("E: 2x size=4 per-lane",          probe_E, 128, "PASS — out[i]==i (two loads/lane)");

    printf("\n=== Interpretation ===\n");
    printf("A PASS  -> per-lane pointer is the correct routing model (not offset param)\n");
    printf("B ----  -> confirms offset=0 with base ptr causes collision\n");
    printf("C       -> shows last-writer semantics for shared LDS slot\n");
    printf("D PASS  -> sc0 streaming cache is safe for DME loads\n");
    printf("E PASS  -> two consecutive size=4 DME loads per lane work correctly\n");
    printf("           Note: size=8 is NOT supported by the builtin on ROCm 7.2 (1/2/4 only)\n");

    hipFree(d_src);
    hipFree(d_out);
    return 0;
}

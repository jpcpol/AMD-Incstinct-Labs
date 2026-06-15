// ---------------------------------------------------------------------------
// compile_smoke.hip — Stage-1 facade smoke test (compile-only / minimal run).
//
// Verifies that <cdna3/cdna3.hpp> includes cleanly and the three primitive families
// are reachable through the facade, plus the two composition examples. This is the
// cold gate before the VM "compile-validate on MI300X" step: if this compiles with
// hipcc --offload-arch=gfx942, the facade wiring (relative includes, namespaces,
// API surface) is correct.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../include compile_smoke.hip -o compile_smoke
// (paths are relative to cdna3-prims/tests/)
//
// Run (optional, tiny): ./compile_smoke  → prints the reduced/softmaxed values.
// ---------------------------------------------------------------------------
#include <cstdio>
#include "../include/cdna3/cdna3.hpp"
#include "../examples/fused_softmax.hpp"
#include "../examples/async_tiled_gemm.hpp"

// Touch each primitive family so the symbols are instantiated and the mapping math
// is type-checked. One wave, trivial data.
__global__ void smoke(float* out){
    int L = threadIdx.x;
    float v = (float)(L + 1);

    // 1. DPP wave reduction (facade → wave::dpp::*)
    float rsum = wave::dpp::reduce_sum_bcast(v);
    float rmax = wave::dpp::reduce_max_bcast(v);

    // 2. DPP-composed row softmax (example building block)
    float p = cdna3::ex::row_softmax_lane(v);

    // 3. MFMA typed tile (facade → cdna3::mfma::*) — exercise the verified mapping.
    __shared__ __half A[cdna3::mfma::M * cdna3::mfma::K];
    __shared__ __half B[cdna3::mfma::K * cdna3::mfma::N];
    __shared__ float  C[cdna3::mfma::M * cdna3::mfma::N];
    for (int i = L; i < cdna3::mfma::M * cdna3::mfma::K; i += 64) A[i] = __float2half(0.01f);
    for (int i = L; i < cdna3::mfma::K * cdna3::mfma::N; i += 64) B[i] = __float2half(0.01f);
    __syncthreads();
    cdna3::mfma::fp16x4 a = cdna3::mfma::load_A(A, L);
    cdna3::mfma::fp16x4 b = cdna3::mfma::load_B(B, L);
    cdna3::mfma::f32x4  c = {0,0,0,0};
    c = cdna3::mfma::mma(a, b, c);
    cdna3::mfma::store_C(C, c, L);
    __syncthreads();

    if (L == 0) out[0] = rsum + rmax + p + C[0];
}

int main(){
    float* d; hipMalloc(&d, sizeof(float));
    hipLaunchKernelGGL(smoke, dim3(1), dim3(64), 0, 0, d);
    hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess){ printf("smoke FAILED: %s\n", hipGetErrorString(e)); return 1; }
    float h; hipMemcpy(&h, d, sizeof(float), hipMemcpyDeviceToHost);
    printf("cdna3-prims facade smoke OK (out=%.4f) — version %d.%d\n",
           h, cdna3::VERSION_MAJOR, cdna3::VERSION_MINOR);
    hipFree(d);
    return 0;
}

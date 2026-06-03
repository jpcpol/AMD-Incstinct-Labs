#include "wave_primitives/wave_primitives.hpp"
#include "wave_primitives/wave_reduce_dpp.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>
#include <cmath>

// ---------------------------------------------------------------------------
// bench_softmax.hip — DPP reductions inside a REAL kernel (row-wise softmax)
//
// Answers the consultant's "microbenchmark gap" critique: wave primitives are
// rarely the bottleneck in isolation; the relevant figure is the speedup when
// embedded in a real kernel. Softmax does a max-reduction and a sum-reduction
// per row, surrounded by exp/divide and HBM loads/stores — the reductions are
// real work here, not a synthetic loop.
//
// Layout: one wave (64 lanes) per row, each lane owns kCols/64 elements.
// We compare the SAME softmax with three reduction backends:
//   1. DPP        — wave::dpp::reduce_max / reduce_sum (0 LDS)
//   2. portable   — __shfl_down reductions (lowers to ds_bpermute, LDS)
//   3. hipCUB     — WarpReduce::Reduce(max) + WarpReduce::Sum (LDS TempStorage)
//
// Each lane holds ELEMS_PER_LANE columns; the row reduction combines them.
// ---------------------------------------------------------------------------

// kElemsPerLane sets the row width (= 64 * kElemsPerLane). Override at compile
// time to switch regimes:
//   -DELEMS_PER_LANE=8  → 512 cols, realistic, likely HBM-bound (end-to-end view)
//   -DELEMS_PER_LANE=1  →  64 cols, reduction-dominated (isolates the DPP effect)
#ifndef ELEMS_PER_LANE
#define ELEMS_PER_LANE 8
#endif
static constexpr int kRows         = 65536;   // number of softmax rows (e.g. attention scores)
static constexpr int kElemsPerLane = ELEMS_PER_LANE;
static constexpr int kCols         = 64 * kElemsPerLane;
static constexpr int kWarmup       = 50;
static constexpr int kIter         = 500;

// ---- reduction backends (max then sum), result broadcast to all lanes -------
enum Backend { DPP, PORTABLE, HIPCUB };

__device__ __forceinline__ float row_max_dpp(float local_max) {
    return wave::dpp::reduce_max_bcast(local_max);
}
__device__ __forceinline__ float row_sum_dpp(float local_sum) {
    return wave::dpp::reduce_sum_bcast(local_sum);
}

__device__ __forceinline__ float row_max_portable(float v) {
    for (int o = 32; o > 0; o >>= 1) { float x = __shfl_down(v, o, 64); v = x > v ? x : v; }
    return __shfl(v, 0, 64);
}
__device__ __forceinline__ float row_sum_portable(float v) {
    for (int o = 32; o > 0; o >>= 1) v += __shfl_down(v, o, 64);
    return __shfl(v, 0, 64);
}

// Softmax kernel templated on backend. One wave per row.
template <Backend B>
__global__ void softmax_kernel(const float* __restrict__ in, float* __restrict__ out) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;                 // 0..63
    const float* rin  = in  + (size_t)row * kCols;
    float*       rout = out + (size_t)row * kCols;

    // Load this lane's elements, track local max.
    float vals[kElemsPerLane];
    float lmax = -INFINITY;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        vals[e] = rin[lane * kElemsPerLane + e];
        lmax = vals[e] > lmax ? vals[e] : lmax;
    }

    // Row max via the selected backend.
    float m;
    if      (B == DPP)      m = row_max_dpp(lmax);
    else if (B == PORTABLE) m = row_max_portable(lmax);
    else {
        using WR = hipcub::WarpReduce<float>;
        __shared__ typename WR::TempStorage ts;
        float r = WR(ts).Reduce(lmax, hipcub::Max());
        m = __shfl(r, 0, 64);
    }

    // exp(x - m), local sum.
    float lsum = 0.0f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) { vals[e] = __expf(vals[e] - m); lsum += vals[e]; }

    // Row sum via the selected backend.
    float s;
    if      (B == DPP)      s = row_sum_dpp(lsum);
    else if (B == PORTABLE) s = row_sum_portable(lsum);
    else {
        using WR = hipcub::WarpReduce<float>;
        __shared__ typename WR::TempStorage ts;
        float r = WR(ts).Sum(lsum);
        s = __shfl(r, 0, 64);
    }

    // Normalize and store.
    const float inv = 1.0f / s;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) rout[lane * kElemsPerLane + e] = vals[e] * inv;
}

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);
    if (dev.wave_size != 64) { printf("This benchmark assumes wave64.\n"); return 1; }

    const size_t n = (size_t)kRows * kCols;
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, n * sizeof(float)));
    {
        std::vector<float> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = (float)((i * 1103515245u + 12345u) & 0xff) / 32.0f - 4.0f;
        HIP_CHECK(hipMemcpy(d_in, h.data(), n * sizeof(float), hipMemcpyHostToDevice));
    }

    // Correctness: row sums of output must each be ~1.0.
    auto check = [&](const char* name, void(*k)(const float*, float*)) {
        hipLaunchKernelGGL(k, dim3(kRows), dim3(64), 0, 0, d_in, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> h(kCols);
        HIP_CHECK(hipMemcpy(h.data(), d_out, kCols * sizeof(float), hipMemcpyDeviceToHost));
        float sum = 0.0f; for (int i = 0; i < kCols; ++i) sum += h[i];
        printf("  %-10s row0 sum = %.4f (expect 1.0) [%s]\n", name, sum,
               (fabsf(sum - 1.0f) < 1e-2f) ? "PASS" : "FAIL");
    };
    printf("\n--- Correctness ---\n");
    check("DPP",      softmax_kernel<DPP>);
    check("portable", softmax_kernel<PORTABLE>);
    check("hipCUB",   softmax_kernel<HIPCUB>);

    const double elems = (double)n;  // elements processed per launch
    print_bench_header("Row-wise softmax (one wave/row, 512 cols) — DPP vs portable vs hipCUB");
    auto rd = time_kernel([&](){ hipLaunchKernelGGL(softmax_kernel<DPP>,      dim3(kRows),dim3(64),0,0,d_in,d_out);}, elems,kWarmup,kIter);
    auto rp = time_kernel([&](){ hipLaunchKernelGGL(softmax_kernel<PORTABLE>, dim3(kRows),dim3(64),0,0,d_in,d_out);}, elems,kWarmup,kIter);
    auto rc = time_kernel([&](){ hipLaunchKernelGGL(softmax_kernel<HIPCUB>,   dim3(kRows),dim3(64),0,0,d_in,d_out);}, elems,kWarmup,kIter);
    print_bench_result("softmax DPP",      rd);
    print_bench_result("softmax portable", rp);
    print_bench_result("softmax hipCUB",   rc);
    print_speedup("DPP", "portable", rd, rp);
    print_speedup("DPP", "hipCUB",   rd, rc);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

#include "wave_primitives/wave_primitives.hpp"
#include "wave_primitives/wave_reduce_dpp.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>
#include <cmath>

// ---------------------------------------------------------------------------
// bench_layernorm.hip — DPP reductions inside LayerNorm (real kernel)
//
// LayerNorm per row: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Requires TWO reductions per row: reduce_sum (mean) + reduce_sum (variance).
// This is the simplest kernel that uses the DPP primitives in context, chosen
// as the first "real kernel" benchmark per the consultant's advice (simpler
// than softmax, less memory-bound, better isolation of the reduction effect).
//
// Layout: one wave (64 lanes) per row, each lane holds ELEMS_PER_LANE elements.
// Three backends compared on the SAME kernel:
//   DPP      — wave::dpp::reduce_sum_bcast (0 LDS, verified Run 2)
//   portable — __shfl_down loop (lowers to ds_bpermute)
//   hipCUB   — WarpReduce::Sum with TempStorage
//
// Override row width at compile time:
//   -DELEMS_PER_LANE=1  →  64 cols: reduction-dominated, isolates DPP effect
//   -DELEMS_PER_LANE=8  → 512 cols: realistic, likely more memory-bound
// ---------------------------------------------------------------------------

#ifndef ELEMS_PER_LANE
#define ELEMS_PER_LANE 1
#endif

static constexpr int kRows         = 65536;
static constexpr int kElemsPerLane = ELEMS_PER_LANE;
static constexpr int kCols         = 64 * kElemsPerLane;
static constexpr float kEps        = 1e-5f;
static constexpr int kWarmup       = 50;
static constexpr int kIter         = 500;

enum Backend { DPP, PORTABLE, HIPCUB };

// --- per-backend reduce_sum broadcast (result on all lanes) -----------------

__device__ __forceinline__ float wsum_dpp(float v) {
    return wave::dpp::reduce_sum_bcast(v);
}
__device__ __forceinline__ float wsum_portable(float v) {
    for (int o = 32; o > 0; o >>= 1) v += __shfl_down(v, o, 64);
    return __shfl(v, 0, 64);
}

// LayerNorm kernel templated on backend. One wave per row.
template <Backend B>
__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const float* xrow   = x   + (size_t)row * kCols;
    float*       outrow = out + (size_t)row * kCols;

    // Load elements and accumulate local sum.
    float vals[kElemsPerLane];
    float lsum = 0.0f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        vals[e] = xrow[lane * kElemsPerLane + e];
        lsum += vals[e];
    }

    // --- Reduction 1: mean ---
    float total_sum;
    if constexpr (B == DPP) {
        total_sum = wsum_dpp(lsum);
    } else if constexpr (B == PORTABLE) {
        total_sum = wsum_portable(lsum);
    } else {
        using WR = hipcub::WarpReduce<float>;
        __shared__ typename WR::TempStorage ts;
        float r = WR(ts).Sum(lsum);
        total_sum = __shfl(r, 0, 64);
    }
    const float mean = total_sum / kCols;

    // Compute local sum of squared deviations.
    float lvar = 0.0f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        float d = vals[e] - mean;
        lvar += d * d;
    }

    // --- Reduction 2: variance ---
    float total_var;
    if constexpr (B == DPP) {
        total_var = wsum_dpp(lvar);
    } else if constexpr (B == PORTABLE) {
        total_var = wsum_portable(lvar);
    } else {
        using WR = hipcub::WarpReduce<float>;
        __shared__ typename WR::TempStorage ts;
        float r = WR(ts).Sum(lvar);
        total_var = __shfl(r, 0, 64);
    }
    const float inv_std = rsqrtf(total_var / kCols + kEps);

    // Normalize, scale, and store.
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        int col = lane * kElemsPerLane + e;
        outrow[col] = (vals[e] - mean) * inv_std * gamma[col] + beta[col];
    }
}

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);
    if (dev.wave_size != 64) { printf("This benchmark assumes wave64.\n"); return 1; }

    const size_t n = (size_t)kRows * kCols;
    float *d_x, *d_gamma, *d_beta, *d_out;
    HIP_CHECK(hipMalloc(&d_x,     n     * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gamma, kCols * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_beta,  kCols * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out,   n     * sizeof(float)));

    {
        std::vector<float> hx(n), hg(kCols, 1.0f), hb(kCols, 0.0f);
        for (size_t i = 0; i < n; ++i)
            hx[i] = (float)((i * 1103515245u + 12345u) & 0xff) / 32.0f - 4.0f;
        HIP_CHECK(hipMemcpy(d_x,     hx.data(), n     * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_gamma, hg.data(), kCols * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_beta,  hb.data(), kCols * sizeof(float), hipMemcpyHostToDevice));
    }

    // Correctness: output mean ≈ 0, variance ≈ 1 per row.
    auto check = [&](const char* name, auto kern) {
        hipLaunchKernelGGL(kern, dim3(kRows), dim3(64), 0, 0, d_x, d_gamma, d_beta, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> h(kCols);
        HIP_CHECK(hipMemcpy(h.data(), d_out, kCols * sizeof(float), hipMemcpyDeviceToHost));
        float m = 0.f, v = 0.f;
        for (int i = 0; i < kCols; ++i) m += h[i];
        m /= kCols;
        for (int i = 0; i < kCols; ++i) v += (h[i]-m)*(h[i]-m);
        v /= kCols;
        bool ok = fabsf(m) < 1e-3f && fabsf(v - 1.0f) < 1e-2f;
        printf("  %-10s row0: mean=%.5f var=%.5f [%s]\n", name, m, v, ok?"PASS":"FAIL");
    };
    printf("\n--- Correctness (gamma=1, beta=0: mean≈0, var≈1) ---\n");
    check("DPP",      layernorm_kernel<DPP>);
    check("portable", layernorm_kernel<PORTABLE>);
    check("hipCUB",   layernorm_kernel<HIPCUB>);

    const double elems = (double)n;
    printf("\nConfig: %d rows x %d cols (ELEMS_PER_LANE=%d), 2 reductions per row\n\n",
           kRows, kCols, kElemsPerLane);
    print_bench_header("LayerNorm (2× reduce_sum per row) — DPP vs portable vs hipCUB");
    auto rd = time_kernel([&](){ hipLaunchKernelGGL(layernorm_kernel<DPP>,      dim3(kRows),dim3(64),0,0,d_x,d_gamma,d_beta,d_out);}, elems,kWarmup,kIter);
    auto rp = time_kernel([&](){ hipLaunchKernelGGL(layernorm_kernel<PORTABLE>, dim3(kRows),dim3(64),0,0,d_x,d_gamma,d_beta,d_out);}, elems,kWarmup,kIter);
    auto rc = time_kernel([&](){ hipLaunchKernelGGL(layernorm_kernel<HIPCUB>,   dim3(kRows),dim3(64),0,0,d_x,d_gamma,d_beta,d_out);}, elems,kWarmup,kIter);
    print_bench_result("layernorm DPP",      rd);
    print_bench_result("layernorm portable", rp);
    print_bench_result("layernorm hipCUB",   rc);
    print_speedup("DPP", "portable", rd, rp);
    print_speedup("DPP", "hipCUB",   rd, rc);

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_gamma));
    HIP_CHECK(hipFree(d_beta));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

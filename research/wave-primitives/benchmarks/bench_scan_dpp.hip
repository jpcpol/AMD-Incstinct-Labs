#include "wave_primitives/wave_primitives.hpp"
#include "wave_primitives/wave_scan_dpp.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>

// Scan: DPP hybrid (4 DPP in-row + 3 shfl row-carry) vs portable Hillis-Steele
// (shfl_up → bpermute) vs hipCUB WarpScan. reduce_sum<float>, wave64.

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;
static constexpr int kReps   = 4096;

// portable Hillis-Steele inclusive scan (shfl_up; lowers to bpermute)
__device__ __forceinline__ float scan_portable(float v) {
    const unsigned lid = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    for (int o = 1; o < 64; o <<= 1) {
        float u = __shfl_up(v, o, 64);
        if (lid >= (unsigned)o) v += u;
    }
    return v;
}

__global__ void verify_dpp(const float* in, float* out) {
    float v = in[threadIdx.x];
    float s = wave::dpp::scan_inclusive_sum(v);
    if (threadIdx.x == 63) out[0] = s;  // inclusive: lane 63 = full sum
}

__global__ void bench_portable(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid], acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float s = scan_portable(val);
        acc += s;
        val = s * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (threadIdx.x == 63) out[blockIdx.x] = acc;
}
__global__ void bench_dpp(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid], acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float s = wave::dpp::scan_inclusive_sum(val);
        acc += s;
        val = s * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (threadIdx.x == 63) out[blockIdx.x] = acc;
}
__global__ void bench_cub(const float* __restrict__ in, float* __restrict__ out) {
    using WS = hipcub::WarpScan<float>;
    __shared__ typename WS::TempStorage temp;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid], acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float s;
        WS(temp).InclusiveSum(val, s);
        acc += s;
        val = s * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (threadIdx.x == 63) out[blockIdx.x] = acc;
}

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);
    const int total = kBlocks * dev.wave_size;
    const double work = static_cast<double>(total) * kReps;
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  total   * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, kBlocks * sizeof(float)));
    {
        std::vector<float> h(total);
        for (int i = 0; i < total; ++i) h[i] = 1.0f;
        HIP_CHECK(hipMemcpy(d_in, h.data(), total * sizeof(float), hipMemcpyHostToDevice));
    }
    // inclusive scan of 64 ones → lane 63 = 64
    hipLaunchKernelGGL(verify_dpp, dim3(1), dim3(dev.wave_size), 0, 0, d_in, d_out);
    HIP_CHECK(hipDeviceSynchronize());
    float r = 0.0f; HIP_CHECK(hipMemcpy(&r, d_out, sizeof(float), hipMemcpyDeviceToHost));
    printf("\n--- Correctness ---\n  dpp::scan_inclusive_sum lane63 = %.1f (expected 64.0) [%s]\n\n",
           r, (r == 64.0f) ? "PASS" : "FAIL");
    if (r != 64.0f) { printf("ABORT\n"); return 1; }

    print_bench_header("scan_inclusive_sum float — portable vs DPP-hybrid vs hipCUB (kReps amortized)");
    auto rp = time_kernel([&](){ hipLaunchKernelGGL(bench_portable, dim3(kBlocks), dim3(dev.wave_size),0,0,d_in,d_out);}, work,kWarmup,kIter);
    auto rd = time_kernel([&](){ hipLaunchKernelGGL(bench_dpp,      dim3(kBlocks), dim3(dev.wave_size),0,0,d_in,d_out);}, work,kWarmup,kIter);
    auto rc = time_kernel([&](){ hipLaunchKernelGGL(bench_cub,      dim3(kBlocks), dim3(dev.wave_size),0,0,d_in,d_out);}, work,kWarmup,kIter);
    print_bench_result("scan portable (shfl_up/bpermute)", rp);
    print_bench_result("scan DPP-hybrid (4 DPP + 3 shfl)", rd);
    print_bench_result("hipCUB WarpScan::InclusiveSum",    rc);
    print_speedup("dpp", "portable", rd, rp);
    print_speedup("dpp", "hipCUB",   rd, rc);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

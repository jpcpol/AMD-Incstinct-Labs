#include "wave_primitives/wave_primitives.hpp"
#include "wave_primitives/wave_reduce_dpp.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;
static constexpr int kReps   = 4096;

__global__ void verify_wave(const float* in, float* out) {
    float v = in[blockIdx.x * blockDim.x + threadIdx.x];
    float s = wave::reduce_sum(v);
    if (wave::lane_id() == 0) out[blockIdx.x] = s;
}
__global__ void verify_dpp(const float* in, float* out) {
    float v = in[blockIdx.x * blockDim.x + threadIdx.x];
    float s = wave::dpp::reduce_sum_bcast(v);   // bcast → result on all lanes
    if (wave::lane_id() == 0) out[blockIdx.x] = s;
}

__global__ void bench_wave(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid], acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float s = wave::reduce_sum(val);
        acc += s;
        val = s * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = acc;
}
__global__ void bench_dpp(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid], acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float s = wave::dpp::reduce_sum_bcast(val);
        acc += s;
        val = s * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = acc;
}
__global__ void bench_dpp_nobcast(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid], acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float s = wave::dpp::reduce_sum(val);   // result on lane 63
        acc += s;
        val = s * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (wave::lane_id() == 63) out[blockIdx.x] = acc;
}
__global__ void bench_cub(const float* __restrict__ in, float* __restrict__ out) {
    using WR = hipcub::WarpReduce<float>;
    __shared__ typename WR::TempStorage temp;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid], acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float s = WR(temp).Sum(val);
        acc += s;
        val = s * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
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
        for (int i = 0; i < total; ++i) h[i] = static_cast<float>(i % dev.wave_size);
        HIP_CHECK(hipMemcpy(d_in, h.data(), total * sizeof(float), hipMemcpyHostToDevice));
    }
    const float expected = static_cast<float>(dev.wave_size * (dev.wave_size - 1) / 2);
    auto check = [&](const char* name, void(*k)(const float*, float*)) {
        hipLaunchKernelGGL(k, dim3(1), dim3(dev.wave_size), 0, 0, d_in, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        float r = 0.0f;
        HIP_CHECK(hipMemcpy(&r, d_out, sizeof(float), hipMemcpyDeviceToHost));
        bool ok = fabsf(r - expected) < 0.5f;
        printf("  %-26s got %.1f  expected %.1f  [%s]\n", name, r, expected, ok?"PASS":"FAIL");
        return ok;
    };
    printf("\n--- Correctness ---\n");
    bool ok = true;
    ok &= check("wave::reduce_sum", verify_wave);
    ok &= check("wave::dpp::reduce_sum_bcast", verify_dpp);
    if (!ok) { printf("\nCORRECTNESS FAILED — aborting.\n"); return 1; }

    print_bench_header("reduce_sum float — bpermute vs DPP(full) vs hipCUB (kReps amortized)");
    auto rw = time_kernel([&](){ hipLaunchKernelGGL(bench_wave, dim3(kBlocks), dim3(dev.wave_size),0,0,d_in,d_out);}, work,kWarmup,kIter);
    auto rd = time_kernel([&](){ hipLaunchKernelGGL(bench_dpp, dim3(kBlocks), dim3(dev.wave_size),0,0,d_in,d_out);}, work,kWarmup,kIter);
    auto rn = time_kernel([&](){ hipLaunchKernelGGL(bench_dpp_nobcast, dim3(kBlocks), dim3(dev.wave_size),0,0,d_in,d_out);}, work,kWarmup,kIter);
    auto rc = time_kernel([&](){ hipLaunchKernelGGL(bench_cub, dim3(kBlocks), dim3(dev.wave_size),0,0,d_in,d_out);}, work,kWarmup,kIter);
    print_bench_result("wave::reduce_sum (bpermute)",  rw);
    print_bench_result("wave::dpp (full, +bcast)",     rd);
    print_bench_result("wave::dpp (full, no bcast)",   rn);
    print_bench_result("hipCUB WarpReduce::Sum",       rc);
    print_speedup("dpp_nobcast", "bpermute", rn, rw);
    print_speedup("dpp_nobcast", "hipCUB",   rn, rc);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

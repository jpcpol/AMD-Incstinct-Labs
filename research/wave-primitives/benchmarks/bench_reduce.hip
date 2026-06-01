#include "wave_primitives/wave_primitives.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>

// ---------------------------------------------------------------------------
// bench_reduce.hip — wave::reduce_sum vs hipCUB WarpReduce
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 \
//         -I../include -I/opt/rocm/include \
//         benchmarks/bench_reduce.hip -o bench_reduce
// ---------------------------------------------------------------------------

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void kernel_wave_reduce(const float* __restrict__ in, float* __restrict__ out) {
    float val = in[blockIdx.x * blockDim.x + threadIdx.x];
    float sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = sum;
}

__global__ void kernel_hipcub_reduce(const float* __restrict__ in, float* __restrict__ out) {
    using WarpReduce = hipcub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    float val = in[blockIdx.x * blockDim.x + threadIdx.x];
    float sum = WarpReduce(temp).Sum(val);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);

    const int total_elems = kBlocks * dev.wave_size;

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  total_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, kBlocks     * sizeof(float)));

    {
        std::vector<float> h_in(total_elems);
        for (int i = 0; i < total_elems; ++i) h_in[i] = static_cast<float>(i % dev.wave_size);
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), total_elems * sizeof(float), hipMemcpyHostToDevice));
    }

    print_bench_header("reduce_sum  float  (1024 blocks x wave_size threads)");

    auto res_wave = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_wave_reduce, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out);
    }, total_elems, kWarmup, kIter);

    auto res_cub = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_hipcub_reduce, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out);
    }, total_elems, kWarmup, kIter);

    print_bench_result("wave::reduce_sum", res_wave);
    print_bench_result("hipCUB WarpReduce::Sum", res_cub);
    print_speedup("wave", "hipCUB", res_wave, res_cub);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

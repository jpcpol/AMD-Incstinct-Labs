#include "wave_primitives/wave_primitives.hpp"
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

// ---------------------------------------------------------------------------
// Benchmark: wave::reduce_sum vs hipCUB WarpReduce
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -I../include \
//         -I/opt/rocm/include \
//         benchmarks/bench_reduce.hip -o bench_reduce
//
// Run:
//   ./bench_reduce
//
// Methodology:
//   - 1024 blocks × warpSize threads = one reduce per wavefront
//   - Warmup: 100 iterations (not timed)
//   - Measure: 1000 iterations, report median latency and throughput
//   - Both kernels do identical work: sum N float values per wavefront
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd) do {                                       \
    hipError_t e = (cmd);                                         \
    if (e != hipSuccess) {                                        \
        fprintf(stderr, "HIP error %s at %s:%d\n",               \
                hipGetErrorString(e), __FILE__, __LINE__);        \
        exit(1);                                                  \
    }                                                             \
} while(0)

static constexpr int kBlocks    = 1024;
static constexpr int kWarmup    = 100;
static constexpr int kIter      = 1000;

// ---------------------------------------------------------------------------
// Kernel A: wave::reduce_sum (this library)
// ---------------------------------------------------------------------------

__global__ void kernel_wave_reduce(const float* __restrict__ in, float* __restrict__ out) {
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    float val      = in[tid];
    float sum      = wave::reduce_sum(val);
    if (wave::lane_id() == 0) {
        out[blockIdx.x] = sum;
    }
}

// ---------------------------------------------------------------------------
// Kernel B: hipCUB WarpReduce
// ---------------------------------------------------------------------------

__global__ void kernel_hipcub_reduce(const float* __restrict__ in, float* __restrict__ out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val     = in[tid];

    using WarpReduce = hipcub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp[1]; // one warp per block
    float sum = WarpReduce(temp[0]).Sum(val);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

struct BenchResult {
    double median_us;
    double throughput_gelems;
};

template <typename Kernel>
BenchResult time_kernel(Kernel kernel, int blocks, int threads,
                        const float* d_in, float* d_out, int total_elems) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warmup
    for (int i = 0; i < kWarmup; ++i) {
        hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), 0, 0, d_in, d_out);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed iterations
    float times[kIter];
    for (int i = 0; i < kIter; ++i) {
        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threads), 0, 0, d_in, d_out);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        HIP_CHECK(hipEventElapsedTime(&times[i], start, stop));
    }

    // Sort for median
    for (int i = 0; i < kIter - 1; ++i)
        for (int j = i + 1; j < kIter; ++j)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }

    double median_ms = times[kIter / 2];
    double median_us = median_ms * 1e3;
    double throughput = static_cast<double>(total_elems) / (median_ms * 1e-3) / 1e9;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return { median_us, throughput };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    int device = 0;
    int wave_size = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&wave_size, hipDeviceAttributeWarpSize, device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("warpSize: %d\n", wave_size);
    printf("Blocks: %d  Threads per block: %d\n\n", kBlocks, wave_size);

    const int total_elems = kBlocks * wave_size;

    // Allocate
    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  total_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, kBlocks     * sizeof(float)));

    // Initialize input on device
    {
        float* h_in = new float[total_elems];
        for (int i = 0; i < total_elems; ++i) h_in[i] = static_cast<float>(i % wave_size);
        HIP_CHECK(hipMemcpy(d_in, h_in, total_elems * sizeof(float), hipMemcpyHostToDevice));
        delete[] h_in;
    }

    // --- Benchmark wave::reduce_sum -----------------------------------------
    auto res_wave = time_kernel(kernel_wave_reduce, kBlocks, wave_size,
                                d_in, d_out, total_elems);

    // --- Benchmark hipCUB WarpReduce ----------------------------------------
    auto res_cub = time_kernel(kernel_hipcub_reduce, kBlocks, wave_size,
                               d_in, d_out, total_elems);

    // --- Report -------------------------------------------------------------
    printf("%-30s %10s %15s\n", "Kernel", "Latency", "Throughput");
    printf("%-30s %10s %15s\n", "------", "-------", "----------");
    printf("%-30s %9.2f us %12.3f Gelems/s\n",
           "wave::reduce_sum", res_wave.median_us, res_wave.throughput_gelems);
    printf("%-30s %9.2f us %12.3f Gelems/s\n",
           "hipCUB WarpReduce::Sum", res_cub.median_us, res_cub.throughput_gelems);
    printf("\nSpeedup wave vs hipCUB: %.3fx\n",
           res_cub.median_us / res_wave.median_us);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

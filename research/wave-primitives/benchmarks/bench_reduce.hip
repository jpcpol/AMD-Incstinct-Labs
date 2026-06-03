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

// Internal repetition count. A single wave reduction is ~6 VALU ops — far
// below the ~5.7us kernel launch floor on MI300X, so a one-shot kernel measures
// launch overhead, not compute. We repeat the reduction kReps times per thread,
// feeding each result back into the next iteration so the optimizer cannot fold
// or eliminate the loop (anti-DCE). This makes compute dominate the launch floor.
static constexpr int kReps = 4096;

// ---------------------------------------------------------------------------
// Kernels — each thread performs kReps dependent reductions.
//
// The dependency chain (acc feeds the next input) prevents dead-code
// elimination and constant folding, forcing the compiler to emit kReps real
// reduction sequences. The final write to global memory anchors the result.
// ---------------------------------------------------------------------------

__global__ void kernel_wave_reduce(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid];
    float acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float sum = wave::reduce_sum(val);
        acc += sum;
        // Perturb val per-lane so the next iteration is not constant-foldable.
        val = sum * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = acc;
}

__global__ void kernel_hipcub_reduce(const float* __restrict__ in, float* __restrict__ out) {
    using WarpReduce = hipcub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[gid];
    float acc = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float sum = WarpReduce(temp).Sum(val);
        acc += sum;
        // Identical perturbation to the wave kernel: depends on the per-lane
        // sum AND threadIdx.x. Both APIs leave the meaningful result on lane 0
        // and partial sums on other lanes; we do NOT broadcast, so neither
        // kernel's dependency chain collapses to a constant. This keeps the
        // comparison symmetric — the only difference is the reduction primitive.
        val = sum * 1.0000001f + static_cast<float>(threadIdx.x);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);

    const int total_elems = kBlocks * dev.wave_size;
    // Each thread performs kReps reductions, so the effective number of
    // reduction operations per launch is total_elems * kReps. Throughput is
    // reported against this amortized work, not the input size.
    const double work_elems = static_cast<double>(total_elems) * kReps;

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  total_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, kBlocks     * sizeof(float)));

    {
        std::vector<float> h_in(total_elems);
        for (int i = 0; i < total_elems; ++i) h_in[i] = static_cast<float>(i % dev.wave_size);
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), total_elems * sizeof(float), hipMemcpyHostToDevice));
    }

    print_bench_header("reduce_sum  float  (1024 blocks x wave_size, kReps amortized)");

    auto res_wave = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_wave_reduce, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out);
    }, work_elems, kWarmup, kIter);

    auto res_cub = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_hipcub_reduce, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out);
    }, work_elems, kWarmup, kIter);

    print_bench_result("wave::reduce_sum", res_wave);
    print_bench_result("hipCUB WarpReduce::Sum", res_cub);
    print_speedup("wave", "hipCUB", res_wave, res_cub);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

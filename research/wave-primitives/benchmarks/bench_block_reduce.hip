#include "wave_primitives/wave_primitives.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>

// ---------------------------------------------------------------------------
// bench_block_reduce.hip — two-phase block reduce vs hipCUB BlockReduce
//
// This is the real-world use case: reducing an entire CTA, not just one wave.
//
// wave-primitives approach (two phases, dynamic block size):
//   Phase 1: each wavefront reduces its slice → partial sum written to LDS
//   Phase 2: first wavefront reduces the partial sums → final result
//   Shared memory: blockDim.x / warpSize floats (minimal footprint)
//
// hipCUB approach: BlockReduce<float, BlockSize> — templated on block size,
//   uses optimized shared memory reduction internally.
//
// Block sizes tested: 128, 256, 512, 1024 threads
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 \
//         -I../include -I/opt/rocm/include \
//         benchmarks/bench_block_reduce.hip -o bench_block_reduce
// ---------------------------------------------------------------------------

static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;

// ---------------------------------------------------------------------------
// wave-primitives block reduce — works for any block size (dynamic)
// Shared memory: blockDim.x / warpSize floats
// ---------------------------------------------------------------------------

__global__ void kernel_wave_block_reduce(const float* __restrict__ in, float* __restrict__ out) {
    extern __shared__ float smem[];

    const unsigned int lane    = wave::lane_id();
    const unsigned int wave_id = threadIdx.x / warpSize;
    const unsigned int n_waves = blockDim.x  / warpSize;

    float val         = in[blockIdx.x * blockDim.x + threadIdx.x];
    float wave_result = wave::reduce_sum(val);

    if (lane == 0) smem[wave_id] = wave_result;
    __syncthreads();

    if (wave_id == 0) {
        float partial = (lane < n_waves) ? smem[lane] : 0.0f;
        float total   = wave::reduce_sum(partial);
        if (lane == 0) out[blockIdx.x] = total;
    }
}

// ---------------------------------------------------------------------------
// hipCUB block reduce — templated on compile-time block size
// ---------------------------------------------------------------------------

template <int BlockSize>
__global__ void kernel_hipcub_block_reduce(const float* __restrict__ in, float* __restrict__ out) {
    using BR = hipcub::BlockReduce<float, BlockSize>;
    __shared__ typename BR::TempStorage temp;
    float val = in[blockIdx.x * blockDim.x + threadIdx.x];
    float sum = BR(temp).Sum(val);
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------

static bool verify_block_reduce(const float* h_out, int n_blocks, int block_size, float input_val) {
    float expected = input_val * block_size;
    for (int b = 0; b < n_blocks; ++b) {
        if (fabsf(h_out[b] - expected) > 0.5f) {
            printf("  [VERIFY FAIL] block %d: got %.1f expected %.1f\n",
                   b, h_out[b], expected);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Run one block size configuration
// ---------------------------------------------------------------------------

static void run_config(int block_size, int n_blocks, const DeviceInfo& dev) {
    const int total = n_blocks * block_size;
    const int smem  = (block_size / dev.wave_size) * sizeof(float);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  total   * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, n_blocks * sizeof(float)));

    {
        std::vector<float> h_in(total, 1.0f);
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), total * sizeof(float), hipMemcpyHostToDevice));
    }

    // Correctness check
    hipLaunchKernelGGL(kernel_wave_block_reduce, n_blocks, block_size, smem, 0, d_in, d_out);
    HIP_CHECK(hipDeviceSynchronize());
    {
        std::vector<float> h_out(n_blocks);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, n_blocks * sizeof(float), hipMemcpyDeviceToHost));
        bool ok = verify_block_reduce(h_out.data(), n_blocks, block_size, 1.0f);
        if (!ok) { printf("  Skipping block_size=%d due to correctness failure.\n", block_size); goto cleanup; }
    }

    {
        char label_wave[64], label_cub[64];
        snprintf(label_wave, sizeof(label_wave), "wave_block_reduce  bs=%4d", block_size);
        snprintf(label_cub,  sizeof(label_cub),  "hipCUB BlockReduce bs=%4d", block_size);

        auto res_wave = time_kernel([&]() {
            hipLaunchKernelGGL(kernel_wave_block_reduce,
                               dim3(n_blocks), dim3(block_size), smem, 0, d_in, d_out);
        }, total, kWarmup, kIter);

        // Dispatch the correctly-typed hipCUB template based on block_size
        BenchResult res_cub{};
        if      (block_size == 128)  res_cub = time_kernel([&]() { hipLaunchKernelGGL((kernel_hipcub_block_reduce<128>),  dim3(n_blocks), dim3(block_size), 0, 0, d_in, d_out); }, total, kWarmup, kIter);
        else if (block_size == 256)  res_cub = time_kernel([&]() { hipLaunchKernelGGL((kernel_hipcub_block_reduce<256>),  dim3(n_blocks), dim3(block_size), 0, 0, d_in, d_out); }, total, kWarmup, kIter);
        else if (block_size == 512)  res_cub = time_kernel([&]() { hipLaunchKernelGGL((kernel_hipcub_block_reduce<512>),  dim3(n_blocks), dim3(block_size), 0, 0, d_in, d_out); }, total, kWarmup, kIter);
        else if (block_size == 1024) res_cub = time_kernel([&]() { hipLaunchKernelGGL((kernel_hipcub_block_reduce<1024>), dim3(n_blocks), dim3(block_size), 0, 0, d_in, d_out); }, total, kWarmup, kIter);

        print_bench_result(label_wave, res_wave);
        print_bench_result(label_cub,  res_cub);
        print_speedup("wave", "hipCUB", res_wave, res_cub);
        printf("\n");
    }

cleanup:
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);

    // Number of blocks chosen so each config processes the same total elements
    const int total_target = 262144; // 256K elements

    print_bench_header("block reduce — wave-primitives vs hipCUB BlockReduce");
    printf("  (two-phase: wave reduce → LDS → wave reduce)\n\n");

    for (int bs : {128, 256, 512, 1024}) {
        int n_blocks = total_target / bs;
        run_config(bs, n_blocks, dev);
        print_separator();
    }

    return 0;
}

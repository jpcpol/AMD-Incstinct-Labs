#include "wave_primitives/wave_primitives.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>

// ---------------------------------------------------------------------------
// bench_scan.hip — wave::scan_inclusive_sum vs hipCUB WarpScan
//
// This is the first published performance comparison for wave64 inclusive
// scan on MI300X. hipCUB WarpScan uses shared memory; wave::scan_inclusive_sum
// is pure register-based (zero shared memory).
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 \
//         -I../include -I/opt/rocm/include \
//         benchmarks/bench_scan.hip -o bench_scan
// ---------------------------------------------------------------------------

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// wave::scan_inclusive_sum — zero shared memory, pure register scan
__global__ void kernel_wave_scan(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val     = in[gid];
    float prefix  = wave::scan_inclusive_sum(val);
    out[gid]      = prefix;
}

// hipCUB WarpScan — uses shared memory TempStorage
__global__ void kernel_hipcub_scan(const float* __restrict__ in, float* __restrict__ out) {
    using WarpScan = hipcub::WarpScan<float>;
    __shared__ typename WarpScan::TempStorage temp;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val     = in[gid];
    float prefix;
    WarpScan(temp).InclusiveSum(val, prefix);
    out[gid] = prefix;
}

// wave::scan_exclusive_sum — included for completeness
__global__ void kernel_wave_scan_excl(const float* __restrict__ in, float* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float val     = in[gid];
    float prefix  = wave::scan_exclusive_sum(val);
    out[gid]      = prefix;
}

// ---------------------------------------------------------------------------
// Correctness check: verify scan output matches CPU reference
// ---------------------------------------------------------------------------

static bool verify_scan(const float* h_out, int wave_size, bool inclusive) {
    // Input was all 1.0f per lane, so:
    // inclusive: lane i → i+1
    // exclusive: lane i → i
    for (int lane = 0; lane < wave_size; ++lane) {
        float expected = inclusive ? static_cast<float>(lane + 1)
                                   : static_cast<float>(lane);
        if (fabsf(h_out[lane] - expected) > 0.5f) {
            printf("  [VERIFY FAIL] lane %d: got %.1f expected %.1f\n",
                   lane, h_out[lane], expected);
            return false;
        }
    }
    return true;
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
    HIP_CHECK(hipMalloc(&d_out, total_elems * sizeof(float)));

    {
        std::vector<float> h_in(total_elems, 1.0f);
        HIP_CHECK(hipMemcpy(d_in, h_in.data(), total_elems * sizeof(float), hipMemcpyHostToDevice));
    }

    // --- Correctness check before benchmarking ------------------------------
    {
        hipLaunchKernelGGL(kernel_wave_scan, dim3(1), dim3(dev.wave_size), 0, 0, d_in, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> h_out(dev.wave_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, dev.wave_size * sizeof(float), hipMemcpyDeviceToHost));
        bool ok = verify_scan(h_out.data(), dev.wave_size, /*inclusive=*/true);
        printf("wave::scan_inclusive_sum correctness: %s\n\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }
    {
        hipLaunchKernelGGL(kernel_wave_scan_excl, dim3(1), dim3(dev.wave_size), 0, 0, d_in, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> h_out(dev.wave_size);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, dev.wave_size * sizeof(float), hipMemcpyDeviceToHost));
        bool ok = verify_scan(h_out.data(), dev.wave_size, /*inclusive=*/false);
        printf("wave::scan_exclusive_sum correctness: %s\n\n", ok ? "PASS" : "FAIL");
        if (!ok) return 1;
    }

    // --- Benchmark: inclusive scan ------------------------------------------
    print_bench_header("scan_inclusive_sum  float  (1024 blocks x wave_size)");

    auto res_wave_incl = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_wave_scan, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out);
    }, total_elems, kWarmup, kIter);

    auto res_cub = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_hipcub_scan, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out);
    }, total_elems, kWarmup, kIter);

    auto res_wave_excl = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_wave_scan_excl, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out);
    }, total_elems, kWarmup, kIter);

    print_bench_result("wave::scan_inclusive_sum (reg-only)", res_wave_incl);
    print_bench_result("hipCUB WarpScan::InclusiveSum",       res_cub);
    print_bench_result("wave::scan_exclusive_sum (reg-only)", res_wave_excl);
    print_speedup("wave_incl", "hipCUB", res_wave_incl, res_cub);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}

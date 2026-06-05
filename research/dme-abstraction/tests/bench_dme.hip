// ---------------------------------------------------------------------------
// bench_dme.hip — Validate and benchmark dme::copy_tile_1d on gfx942.
//
// Compares three tile-load strategies for a 256-float tile per block:
//   A. DME load via __builtin_amdgcn_global_load_lds (aux=0, cached)
//   B. DME load with sc0 streaming cache hint (aux=1)
//   C. Manual global load: each lane reads directly into registers, stores to LDS
//
// Metric: throughput in GB/s (tile_bytes * blocks / elapsed_us)
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../include tests/bench_dme.hip -o bench_dme
// ---------------------------------------------------------------------------
#include "dme/dme_copy.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cmath>

static constexpr int kTile    = 256;    // floats per block (4 iters × 64 lanes)
static constexpr int kBlocks  = 65536;
static constexpr int kWarmup  = 50;
static constexpr int kIter    = 500;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// A: DME load, default cache (aux=0)
__global__ void kernel_dme_cached(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[kTile];
    const int base = blockIdx.x * kTile;
    dme::copy_tile_1d<float, kTile>(src + base, tile);
    dme::wait();
    __syncthreads();
    // Consume tile: each lane reads 4 elements and writes sum to out.
    const int L = threadIdx.x;
    float s = tile[L] + tile[L+64] + tile[L+128] + tile[L+192];
    if (L == 0) out[blockIdx.x] = s;
}

// B: DME load, streaming cache (aux=sc0=1)
__global__ void kernel_dme_stream(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[kTile];
    const int base = blockIdx.x * kTile;
    dme::copy_tile_1d_stream<float, kTile>(src + base, tile);
    dme::wait();
    __syncthreads();
    const int L = threadIdx.x;
    float s = tile[L] + tile[L+64] + tile[L+128] + tile[L+192];
    if (L == 0) out[blockIdx.x] = s;
}

// C: manual load — each lane loads 4 floats directly into registers, stores to LDS.
__global__ void kernel_manual(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[kTile];
    const int base = blockIdx.x * kTile;
    const int L = threadIdx.x;
    tile[L]     = src[base + L];
    tile[L+64]  = src[base + L + 64];
    tile[L+128] = src[base + L + 128];
    tile[L+192] = src[base + L + 192];
    __syncthreads();
    float s = tile[L] + tile[L+64] + tile[L+128] + tile[L+192];
    if (L == 0) out[blockIdx.x] = s;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

struct Result { float median_us, p99_us, gb_s; };

template <typename F>
Result time_kern(F&& launch, double bytes, int warmup, int iters) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0);
    hipEventCreate(&t1);
    for (int i = 0; i < warmup; ++i) launch();
    hipDeviceSynchronize();
    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        hipEventRecord(t0);
        launch();
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        hipEventElapsedTime(&times[i], t0, t1);
        times[i] *= 1000.f;  // ms → µs
    }
    std::sort(times.begin(), times.end());
    float med = times[iters / 2];
    float p99 = times[(int)(iters * 0.99f)];
    float gbs = (float)(bytes / med / 1e3);  // bytes/µs = GB/s
    hipEventDestroy(t0);
    hipEventDestroy(t1);
    return {med, p99, gbs};
}

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------
bool check(float* d_out, int n_blocks, const float* h_src) {
    std::vector<float> h(n_blocks);
    hipMemcpy(h.data(), d_out, n_blocks * sizeof(float), hipMemcpyDeviceToHost);
    for (int b = 0; b < n_blocks; ++b) {
        float exp = 0.f;
        for (int i = b * kTile; i < b * kTile + kTile; ++i) exp += h_src[i] / 4.f;
        // Each lane sums 4 elements; only lane 0 writes, so out[b] = sum of first lane's 4
        // Actually lane 0: tile[0]+tile[64]+tile[128]+tile[192] = src[base]+src[base+64]+...
        float exp0 = h_src[b*kTile+0] + h_src[b*kTile+64] + h_src[b*kTile+128] + h_src[b*kTile+192];
        if (fabsf(h[b] - exp0) > 0.1f) return false;
    }
    return true;
}

int main() {
    // Device info
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device : %s\n", prop.name);
    printf("CUs    : %d\n", prop.multiProcessorCount);
    printf("Clock  : %d MHz\n", prop.clockRate / 1000);
    printf("VRAM   : %zu GB\n\n", prop.totalGlobalMem / (1024*1024*1024));

    const int N = kBlocks * kTile;
    std::vector<float> h_src(N);
    for (int i = 0; i < N; ++i) h_src[i] = (float)(i % 256);

    float *d_src, *d_out;
    hipMalloc(&d_src, N * sizeof(float));
    hipMalloc(&d_out, kBlocks * sizeof(float));
    hipMemcpy(d_src, h_src.data(), N * sizeof(float), hipMemcpyHostToDevice);

    // Correctness
    printf("--- Correctness (%d blocks x %d floats/block) ---\n", kBlocks, kTile);
    auto run_check = [&](const char* name, auto kern) {
        hipMemset(d_out, 0, kBlocks * sizeof(float));
        hipLaunchKernelGGL(kern, dim3(kBlocks), dim3(64), 0, 0, d_src, d_out);
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) {
            printf("  %-25s LAUNCH ERROR: %s\n", name, hipGetErrorString(e));
            return;
        }
        bool ok = check(d_out, kBlocks, h_src.data());
        printf("  %-25s [%s]\n", name, ok ? "PASS" : "FAIL");
    };
    run_check("DME cached (aux=0)",   kernel_dme_cached);
    run_check("DME stream (aux=sc0)", kernel_dme_stream);
    run_check("manual load",          kernel_manual);

    // Benchmark
    double bytes = (double)kBlocks * kTile * sizeof(float);  // total HBM read per launch
    printf("\n=== Tile load throughput (%d blocks x %d floats = %.0f MB/launch) ===\n",
           kBlocks, kTile, bytes / 1e6);
    printf("%-28s  %10s  %10s  %8s\n", "Kernel", "Median(us)", "P99(us)", "GB/s");
    printf("%-28s  %10s  %10s  %8s\n", "------", "----------", "-------", "----");

    auto pr = [](const char* name, Result r) {
        printf("%-28s  %10.3f  %10.3f  %8.2f\n", name, r.median_us, r.p99_us, r.gb_s);
    };

    auto rd = time_kern([&](){ hipLaunchKernelGGL(kernel_dme_cached, dim3(kBlocks),dim3(64),0,0,d_src,d_out); }, bytes, kWarmup, kIter);
    auto rs = time_kern([&](){ hipLaunchKernelGGL(kernel_dme_stream, dim3(kBlocks),dim3(64),0,0,d_src,d_out); }, bytes, kWarmup, kIter);
    auto rm = time_kern([&](){ hipLaunchKernelGGL(kernel_manual,     dim3(kBlocks),dim3(64),0,0,d_src,d_out); }, bytes, kWarmup, kIter);

    pr("DME cached (aux=0)",   rd);
    pr("DME stream (aux=sc0)", rs);
    pr("manual load",          rm);

    printf("\nSpeedup DME-cached  vs manual: %.3fx\n", rm.median_us / rd.median_us);
    printf("Speedup DME-stream  vs manual: %.3fx\n",   rm.median_us / rs.median_us);

    hipFree(d_src);
    hipFree(d_out);
    return 0;
}

// ---------------------------------------------------------------------------
// bench_dme_async.hip — DME async double-buffer pipeline vs synchronous baseline
//
// Goal: measure whether launching copy_tile_1d BEFORE the barrier (non-blocking)
// and doing compute while the DME fills the next buffer reduces wall time vs
// a serial load-then-compute loop.
//
// Pattern:
//   sync (serial):    prefetch(tile[0]) → wait → compute(tile[0])
//                     prefetch(tile[1]) → wait → compute(tile[1]) ...
//
//   async (pipeline): prefetch(tile[0])
//                     loop i:
//                       prefetch(tile[i+1])   ← DME fills buf[(i+1)&1]
//                       wait + sync           ← wait for tile[i] to be ready
//                       compute(tile[i])      ← process buf[i&1]
//                     wait + compute(last)
//
// The compute is a dot product of the tile with itself (sum of squares) — purely
// register-based after the LDS load, modeled to simulate the compute phase of
// Flash Attention's MFMA accumulation. kFLOPS_PER_ELEM controls compute density.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../include tests/bench_dme_async.hip -o bench_dme_async
// ---------------------------------------------------------------------------
#include "dme/dme_copy.hpp"
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <cmath>

// Tile size per block (floats). Must be multiple of 64.
static constexpr int kTile    = 256;
// Number of tiles to process per block (pipeline depth).
static constexpr int kTiles   = 16;
// Synthetic compute density: each element is accumulated kFLOPS_PER_ELEM times.
// Increase to make compute longer relative to memory load — exposes more overlap.
static constexpr int kFLOPS   = 64;

static constexpr int kBlocks  = 4096;
static constexpr int kWarmup  = 30;
static constexpr int kIter    = 300;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Synthetic compute: dot product of tile[0..kTile-1] with itself, kFLOPS times.
// Result written to out[blockIdx.x] to prevent DCE.
__device__ __forceinline__
float compute_tile(const float* __restrict__ tile) {
    float acc = 0.f;
    #pragma unroll 1
    for (int rep = 0; rep < kFLOPS; ++rep) {
        #pragma unroll
        for (int i = threadIdx.x; i < kTile; i += 64)
            acc += tile[i] * tile[i];
    }
    return acc;
}

// --- Kernel A: serial (load → wait → compute per tile) ---
__global__ void kernel_serial(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[kTile];
    const float* base = src + (size_t)blockIdx.x * kTile * kTiles;
    float result = 0.f;

    for (int t = 0; t < kTiles; ++t) {
        dme::copy_tile_1d_stream<float, kTile>(base + t * kTile, tile);
        dme::wait();
        __syncthreads();
        result += compute_tile(tile);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = result;
}

// --- Kernel B: async double-buffer pipeline ---
// Two LDS buffers (ping/pong). While computing buffer[i&1], the DME fills buffer[(i+1)&1].
__global__ void kernel_async(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float buf[2][kTile];   // ping-pong buffers
    const float* base = src + (size_t)blockIdx.x * kTile * kTiles;
    float result = 0.f;

    // Prefetch tile 0 into buf[0] — no wait yet
    dme::copy_tile_1d_stream<float, kTile>(base, buf[0]);

    for (int t = 0; t < kTiles - 1; ++t) {
        // Launch prefetch for next tile into the OTHER buffer — still async
        int next_buf = (t + 1) & 1;
        dme::copy_tile_1d_stream<float, kTile>(base + (t + 1) * kTile, buf[next_buf]);

        // Now wait for CURRENT tile (t) to be ready in buf[t&1]
        dme::wait();
        __syncthreads();

        // Compute on current tile while DME is filling next_buf
        result += compute_tile(buf[t & 1]);
    }

    // Last tile: just wait and compute
    dme::wait();
    __syncthreads();
    result += compute_tile(buf[(kTiles - 1) & 1]);

    if (threadIdx.x == 0) out[blockIdx.x] = result;
}

// --- Kernel C: manual load serial (no DME, control) ---
__global__ void kernel_manual_serial(const float* __restrict__ src, float* __restrict__ out) {
    __shared__ float tile[kTile];
    const float* base = src + (size_t)blockIdx.x * kTile * kTiles;
    float result = 0.f;

    for (int t = 0; t < kTiles; ++t) {
        const int L = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < kTile / 64; ++i)
            tile[L + i * 64] = base[t * kTile + L + i * 64];
        __syncthreads();
        result += compute_tile(tile);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = result;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
struct Result { float med, p99, ms; };

template <typename F>
Result bench(F&& f, int warmup, int iters) {
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    for (int i = 0; i < warmup; ++i) f();
    hipDeviceSynchronize();
    std::vector<float> ts(iters);
    for (int i = 0; i < iters; ++i) {
        hipEventRecord(t0); f(); hipEventRecord(t1);
        hipEventSynchronize(t1);
        hipEventElapsedTime(&ts[i], t0, t1);
    }
    std::sort(ts.begin(), ts.end());
    float med = ts[iters/2], p99 = ts[(int)(iters*0.99f)];
    hipEventDestroy(t0); hipEventDestroy(t1);
    return {med, p99, med};
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device : %s\n", prop.name);
    printf("CUs    : %d\n", prop.multiProcessorCount);
    printf("Clock  : %d MHz\n\n", prop.clockRate / 1000);

    const size_t N = (size_t)kBlocks * kTile * kTiles;
    std::vector<float> h(N);
    for (size_t i = 0; i < N; ++i) h[i] = (float)(i & 255) * 0.01f;

    float *d_src, *d_out;
    hipMalloc(&d_src, N * sizeof(float));
    hipMalloc(&d_out, kBlocks * sizeof(float));
    hipMemcpy(d_src, h.data(), N * sizeof(float), hipMemcpyHostToDevice);

    // Correctness: serial and async should give identical results
    std::vector<float> h_serial(kBlocks), h_async(kBlocks), h_manual(kBlocks);
    auto get = [&](float* dst, float* d) {
        hipMemcpy(dst, d, kBlocks * sizeof(float), hipMemcpyDeviceToHost);
    };

    hipLaunchKernelGGL(kernel_serial,        dim3(kBlocks), dim3(64), 0, 0, d_src, d_out);
    hipDeviceSynchronize(); get(h_serial.data(), d_out);

    hipLaunchKernelGGL(kernel_async,         dim3(kBlocks), dim3(64), 0, 0, d_src, d_out);
    hipDeviceSynchronize(); get(h_async.data(), d_out);

    hipLaunchKernelGGL(kernel_manual_serial, dim3(kBlocks), dim3(64), 0, 0, d_src, d_out);
    hipDeviceSynchronize(); get(h_manual.data(), d_out);

    float max_diff_async = 0.f, max_diff_manual = 0.f;
    for (int i = 0; i < kBlocks; ++i) {
        max_diff_async  = std::max(max_diff_async,  fabsf(h_serial[i] - h_async[i]));
        max_diff_manual = std::max(max_diff_manual, fabsf(h_serial[i] - h_manual[i]));
    }
    printf("--- Correctness ---\n");
    printf("  serial vs async  max_diff = %.6f  [%s]\n",
           max_diff_async,  max_diff_async  < 0.1f ? "PASS" : "FAIL");
    printf("  serial vs manual max_diff = %.6f  [%s]\n\n",
           max_diff_manual, max_diff_manual < 0.1f ? "PASS" : "FAIL");

    // Config info
    double hbm_bytes = (double)kBlocks * kTile * kTiles * sizeof(float);
    double flops     = (double)kBlocks * kTile * kTiles * kFLOPS * 2;  // multiply-add
    printf("Config: %d blocks x %d tiles x %d floats/tile\n",
           kBlocks, kTiles, kTile);
    printf("HBM per launch : %.1f MB\n", hbm_bytes / 1e6);
    printf("FLOPs per launch: %.1f GFLOP (kFLOPS=%d)\n\n", flops / 1e9, kFLOPS);

    // Benchmark
    auto rs = bench([&](){ hipLaunchKernelGGL(kernel_serial,        dim3(kBlocks),dim3(64),0,0,d_src,d_out); }, kWarmup, kIter);
    auto ra = bench([&](){ hipLaunchKernelGGL(kernel_async,         dim3(kBlocks),dim3(64),0,0,d_src,d_out); }, kWarmup, kIter);
    auto rm = bench([&](){ hipLaunchKernelGGL(kernel_manual_serial, dim3(kBlocks),dim3(64),0,0,d_src,d_out); }, kWarmup, kIter);

    printf("=== DME async pipeline vs serial (kFLOPS=%d per elem) ===\n", kFLOPS);
    printf("%-28s  %10s  %10s\n", "Kernel", "Median(ms)", "P99(ms)");
    printf("%-28s  %10s  %10s\n", "------", "----------", "-------");
    printf("%-28s  %10.3f  %10.3f\n", "DME serial (sync)",    rs.med, rs.p99);
    printf("%-28s  %10.3f  %10.3f\n", "DME async (pipeline)", ra.med, ra.p99);
    printf("%-28s  %10.3f  %10.3f\n", "manual serial",        rm.med, rm.p99);
    printf("\nSpeedup async vs serial:       %.3fx\n", rs.med / ra.med);
    printf("Speedup async vs manual:       %.3fx\n",   rm.med / ra.med);

    // Roofline context
    printf("\n--- Roofline context ---\n");
    printf("Achieved BW  (serial):  %.1f GB/s\n", hbm_bytes / rs.med / 1e6);
    printf("Achieved BW  (async):   %.1f GB/s\n", hbm_bytes / ra.med / 1e6);
    printf("Achieved TFLOPS (serial): %.3f\n",    flops / rs.med / 1e12);
    printf("Achieved TFLOPS (async):  %.3f\n",    flops / ra.med / 1e12);

    hipFree(d_src); hipFree(d_out);
    return 0;
}

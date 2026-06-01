#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// bench_common.hpp
//
// Shared infrastructure for all wave-primitives benchmarks.
// Provides: HIP_CHECK, DeviceInfo, BenchResult, time_kernel, print helpers.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// HIP_CHECK
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd) do {                                            \
    hipError_t _e = (cmd);                                             \
    if (_e != hipSuccess) {                                            \
        fprintf(stderr, "[HIP ERROR] %s  at %s:%d\n",                 \
                hipGetErrorString(_e), __FILE__, __LINE__);            \
        exit(1);                                                       \
    }                                                                  \
} while(0)

// ---------------------------------------------------------------------------
// DeviceInfo
// ---------------------------------------------------------------------------

struct DeviceInfo {
    char name[256];
    int  wave_size;
    int  cu_count;
    int  clock_mhz;
    long long mem_bytes;
};

inline DeviceInfo get_device_info(int device = 0) {
    DeviceInfo info{};
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    strncpy(info.name, prop.name, 255);
    HIP_CHECK(hipDeviceGetAttribute(&info.wave_size,  hipDeviceAttributeWarpSize,              device));
    HIP_CHECK(hipDeviceGetAttribute(&info.cu_count,   hipDeviceAttributeMultiprocessorCount,   device));
    HIP_CHECK(hipDeviceGetAttribute(&info.clock_mhz,  hipDeviceAttributeClockRate,             device));
    info.mem_bytes = prop.totalGlobalMem;
    return info;
}

inline void print_device_info(const DeviceInfo& d) {
    printf("Device : %s\n", d.name);
    printf("CUs    : %d\n", d.cu_count);
    printf("WaveSz : %d\n", d.wave_size);
    printf("Clock  : %d MHz\n", d.clock_mhz / 1000);
    printf("VRAM   : %.0f GB\n", d.mem_bytes / 1e9);
    printf("\n");
}

// ---------------------------------------------------------------------------
// BenchResult
// ---------------------------------------------------------------------------

struct BenchResult {
    double median_us        = 0.0;
    double p99_us           = 0.0;
    double throughput_gelems = 0.0;
};

// ---------------------------------------------------------------------------
// time_kernel
//
// Takes any zero-argument lambda that launches exactly one kernel.
// Returns median and P99 latency in microseconds, plus throughput if
// total_elems > 0.
//
// Usage:
//   auto res = time_kernel([&]() {
//       hipLaunchKernelGGL(my_kernel, grid, block, 0, 0, d_in, d_out);
//   }, total_elems, warmup, iters);
// ---------------------------------------------------------------------------

inline BenchResult time_kernel(
    const std::function<void()>& launch_fn,
    double                       total_elems = 0.0,
    int                          warmup      = 100,
    int                          iters       = 1000)
{
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    for (int i = 0; i < warmup; ++i) launch_fn();
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> times(iters);
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(ev_start));
        launch_fn();
        HIP_CHECK(hipEventRecord(ev_stop));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        HIP_CHECK(hipEventElapsedTime(&times[i], ev_start, ev_stop));
    }

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    std::sort(times.begin(), times.end());
    double median_ms = times[iters / 2];
    double p99_ms    = times[static_cast<int>(iters * 0.99)];
    double tput      = (total_elems > 0.0)
                       ? total_elems / (median_ms * 1e-3) / 1e9
                       : 0.0;

    return { median_ms * 1e3, p99_ms * 1e3, tput };
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

inline void print_bench_header(const char* title) {
    printf("=== %s ===\n", title);
    printf("%-36s %12s %10s %16s\n",
           "Kernel", "Median(us)", "P99(us)", "Throughput");
    printf("%-36s %12s %10s %16s\n",
           "------", "----------", "-------", "----------");
}

inline void print_bench_result(const char* name, const BenchResult& r) {
    if (r.throughput_gelems > 0.0)
        printf("%-36s %12.3f %10.3f %13.3f Ge/s\n",
               name, r.median_us, r.p99_us, r.throughput_gelems);
    else
        printf("%-36s %12.3f %10.3f\n",
               name, r.median_us, r.p99_us);
}

inline void print_speedup(const char* a, const char* b, const BenchResult& ra, const BenchResult& rb) {
    printf("\nSpeedup %s vs %s: %.3fx\n", a, b, rb.median_us / ra.median_us);
}

inline void print_separator() {
    printf("------------------------------------------------------------\n");
}

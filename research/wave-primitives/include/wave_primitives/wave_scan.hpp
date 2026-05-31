#pragma once
#include "wave_shuffle.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// ---------------------------------------------------------------------------
// wave_scan.hpp
//
// Wave-level prefix scan (prefix sum) — correct for wave32 and wave64.
//
// Algorithm: Hillis-Steele parallel scan
//   Step 1:  each lane adds value from lane-1   → partial prefix
//   Step 2:  each lane adds value from lane-2   → partial prefix
//   ...
//   Step log2(warpSize): done
//
// Total steps: 6 for wave64 (log2(64)), 5 for wave32 (log2(32)).
// Compiler unrolls all steps because warpSize is early-folded.
//
// Why Hillis-Steele (not Kogge-Stone via shfl_up)?
//   Both are equivalent for a warp-level scan. shfl_up is the correct
//   intrinsic (shifts values toward higher lane IDs → lane N reads from N-delta).
//   This is the standard approach; hipCUB WarpScan uses the same pattern.
//
// What doesn't exist anywhere:
//   - No standalone header-only AMD scan primitive covers wave64.
//   - hipCUB::WarpScan exists but requires shared memory and block context.
//   - This is a pure register scan — zero shared memory.
// ---------------------------------------------------------------------------

namespace wave {

// ---------------------------------------------------------------------------
// scan_inclusive_sum — each lane i gets sum(input[0..i])
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T scan_inclusive_sum(T val) {
    const unsigned int lid = lane_id();
    // warpSize is early-folded: this loop unrolls to log2(warpSize) iterations.
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        T other = shfl_up(val, offset);
        if (lid >= static_cast<unsigned int>(offset)) {
            val += other;
        }
    }
    return val;
}

// __half: accumulate in float32
__device__ __forceinline__ __half scan_inclusive_sum(__half val) {
    float f = __half2float(val);
    const unsigned int lid = lane_id();
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        float other = shfl_up(f, offset);
        if (lid >= static_cast<unsigned int>(offset)) {
            f += other;
        }
    }
    return __float2half(f);
}

__device__ __forceinline__ hip_bfloat16 scan_inclusive_sum(hip_bfloat16 val) {
    float f = __bfloat162float(val);
    const unsigned int lid = lane_id();
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        float other = shfl_up(f, offset);
        if (lid >= static_cast<unsigned int>(offset)) {
            f += other;
        }
    }
    return __float2bfloat16(f);
}

// ---------------------------------------------------------------------------
// scan_exclusive_sum — each lane i gets sum(input[0..i-1]), lane 0 gets 0
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T scan_exclusive_sum(T val) {
    // Inclusive scan, then shift right by one lane.
    T inclusive = scan_inclusive_sum(val);
    T exclusive = shfl_up(inclusive, 1);
    // Lane 0 has no predecessor — identity element for sum is 0.
    return (lane_id() == 0) ? static_cast<T>(0) : exclusive;
}

__device__ __forceinline__ __half scan_exclusive_sum(__half val) {
    __half inclusive = scan_inclusive_sum(val);
    __half exclusive = shfl_up(inclusive, 1);
    return (lane_id() == 0) ? __float2half(0.0f) : exclusive;
}

__device__ __forceinline__ hip_bfloat16 scan_exclusive_sum(hip_bfloat16 val) {
    hip_bfloat16 inclusive = scan_inclusive_sum(val);
    hip_bfloat16 exclusive = shfl_up(inclusive, 1);
    return (lane_id() == 0) ? __float2bfloat16(0.0f) : exclusive;
}

// ---------------------------------------------------------------------------
// scan_inclusive_max / scan_inclusive_min
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T scan_inclusive_max(T val) {
    const unsigned int lid = lane_id();
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        T other = shfl_up(val, offset);
        if (lid >= static_cast<unsigned int>(offset)) {
            val = max(val, other);
        }
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T scan_inclusive_min(T val) {
    const unsigned int lid = lane_id();
    for (int offset = 1; offset < warpSize; offset <<= 1) {
        T other = shfl_up(val, offset);
        if (lid >= static_cast<unsigned int>(offset)) {
            val = min(val, other);
        }
    }
    return val;
}

} // namespace wave

#pragma once
#include "wave_shuffle.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <type_traits>

// ---------------------------------------------------------------------------
// wave_reduce.hpp
//
// Wave-level reduction operations — correct for wave32 and wave64.
//
// Key insight (ROCm 7+):
//   warpSize is "early-folded" by the compiler — it behaves as a compile-time
//   constant in loop bounds. The compiler generates a fully-unrolled loop for
//   wave64 (6 iterations: 32,16,8,4,2,1) and wave32 (5 iterations: 16,8,4,2,1)
//   with zero runtime overhead. No template<int WarpSize> dispatch needed.
//
// What exists:
//   - rocm-examples/warp_size_reduction: correct but uses a static_for helper
//     from a private template library; not a standalone usable primitive.
//   - hipCUB BlockReduce: excellent block-level, but operates over shared memory
//     and requires a shared memory buffer — not suitable for intra-wave use.
//
// Our improvement:
//   - Single header, zero dependencies
//   - All numeric types including __half and hip_bfloat16
//   - reduce_sum, reduce_max, reduce_min in one place
//   - Explicit correctness for wave64: loop starts at warpSize/2 = 32
// ---------------------------------------------------------------------------

namespace wave {

// ---------------------------------------------------------------------------
// reduce_sum
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T reduce_sum(T val) {
    // warpSize/2 = 32 on wave64, 16 on wave32 — unrolled at compile time.
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += shfl_down(val, offset);
    }
    return val;
}

// __half: accumulate in float32 to avoid half-precision rounding cascade.
__device__ __forceinline__ __half reduce_sum(__half val) {
    float f = __half2float(val);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        f += shfl_down(f, offset);
    }
    return __float2half(f);
}

// hip_bfloat16: same strategy.
__device__ __forceinline__ hip_bfloat16 reduce_sum(hip_bfloat16 val) {
    float f = __bfloat162float(val);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        f += shfl_down(f, offset);
    }
    return __float2bfloat16(f);
}

// ---------------------------------------------------------------------------
// reduce_max
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T reduce_max(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = max(val, shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ __half reduce_max(__half val) {
    float f = __half2float(val);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        f = max(f, shfl_down(f, offset));
    }
    return __float2half(f);
}

__device__ __forceinline__ hip_bfloat16 reduce_max(hip_bfloat16 val) {
    float f = __bfloat162float(val);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        f = max(f, shfl_down(f, offset));
    }
    return __float2bfloat16(f);
}

// ---------------------------------------------------------------------------
// reduce_min
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T reduce_min(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = min(val, shfl_down(val, offset));
    }
    return val;
}

__device__ __forceinline__ __half reduce_min(__half val) {
    float f = __half2float(val);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        f = min(f, shfl_down(f, offset));
    }
    return __float2half(f);
}

__device__ __forceinline__ hip_bfloat16 reduce_min(hip_bfloat16 val) {
    float f = __bfloat162float(val);
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        f = min(f, shfl_down(f, offset));
    }
    return __float2bfloat16(f);
}

// ---------------------------------------------------------------------------
// reduce_and / reduce_or — for integer types
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T reduce_and(T val) {
    static_assert(std::is_integral<T>::value, "reduce_and requires an integer type");
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val &= shfl_down(val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T reduce_or(T val) {
    static_assert(std::is_integral<T>::value, "reduce_or requires an integer type");
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val |= shfl_down(val, offset);
    }
    return val;
}

} // namespace wave

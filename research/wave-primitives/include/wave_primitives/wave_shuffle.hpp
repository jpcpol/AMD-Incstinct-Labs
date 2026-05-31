#pragma once
#include "detail/config.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// ---------------------------------------------------------------------------
// wave_shuffle.hpp
//
// Cross-lane data movement — portable across wave32 and wave64.
//
// Design notes:
//   - AMD HIP: __shfl_* work without a mask argument (whole-wavefront semantics).
//   - NVIDIA CUDA: __shfl_*_sync require an explicit mask. We always pass FULL_MASK
//     since we operate over the entire active wavefront.
//   - __half and hip_bfloat16 are handled by converting to/from float32 because
//     __shfl on sub-32-bit types is implementation-defined on some compilers.
// ---------------------------------------------------------------------------

namespace wave {

// --- lane_id ----------------------------------------------------------------

__device__ __forceinline__ unsigned int lane_id() {
#if WP_AMD
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
#else
    unsigned int id;
    asm("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
#endif
}

// --- shfl (read from arbitrary lane) ----------------------------------------

template <typename T>
__device__ __forceinline__ T shfl(T val, int src_lane, int width = warpSize) {
#if WP_AMD
    return __shfl(val, src_lane, width);
#else
    return __shfl_sync(WP_FULL_MASK, val, src_lane, width);
#endif
}

// Specializations for sub-32-bit types via float promotion
__device__ __forceinline__ __half shfl(__half val, int src_lane, int width = warpSize) {
    float f = shfl(__half2float(val), src_lane, width);
    return __float2half(f);
}

__device__ __forceinline__ hip_bfloat16 shfl(hip_bfloat16 val, int src_lane, int width = warpSize) {
    float f = shfl(__bfloat162float(val), src_lane, width);
    return __float2bfloat16(f);
}

// --- shfl_down (shift lanes toward higher lane IDs) -------------------------

template <typename T>
__device__ __forceinline__ T shfl_down(T val, unsigned int delta, int width = warpSize) {
#if WP_AMD
    return __shfl_down(val, delta, width);
#else
    return __shfl_down_sync(WP_FULL_MASK, val, delta, width);
#endif
}

__device__ __forceinline__ __half shfl_down(__half val, unsigned int delta, int width = warpSize) {
    float f = shfl_down(__half2float(val), delta, width);
    return __float2half(f);
}

__device__ __forceinline__ hip_bfloat16 shfl_down(hip_bfloat16 val, unsigned int delta, int width = warpSize) {
    float f = shfl_down(__bfloat162float(val), delta, width);
    return __float2bfloat16(f);
}

// --- shfl_up (shift lanes toward lower lane IDs) ----------------------------

template <typename T>
__device__ __forceinline__ T shfl_up(T val, unsigned int delta, int width = warpSize) {
#if WP_AMD
    return __shfl_up(val, delta, width);
#else
    return __shfl_up_sync(WP_FULL_MASK, val, delta, width);
#endif
}

__device__ __forceinline__ __half shfl_up(__half val, unsigned int delta, int width = warpSize) {
    float f = shfl_up(__half2float(val), delta, width);
    return __float2half(f);
}

__device__ __forceinline__ hip_bfloat16 shfl_up(hip_bfloat16 val, unsigned int delta, int width = warpSize) {
    float f = shfl_up(__bfloat162float(val), delta, width);
    return __float2bfloat16(f);
}

// --- shfl_xor (XOR lane mask — butterfly patterns) --------------------------

template <typename T>
__device__ __forceinline__ T shfl_xor(T val, int lane_mask, int width = warpSize) {
#if WP_AMD
    return __shfl_xor(val, lane_mask, width);
#else
    return __shfl_xor_sync(WP_FULL_MASK, val, lane_mask, width);
#endif
}

__device__ __forceinline__ __half shfl_xor(__half val, int lane_mask, int width = warpSize) {
    float f = shfl_xor(__half2float(val), lane_mask, width);
    return __float2half(f);
}

__device__ __forceinline__ hip_bfloat16 shfl_xor(hip_bfloat16 val, int lane_mask, int width = warpSize) {
    float f = shfl_xor(__bfloat162float(val), lane_mask, width);
    return __float2bfloat16(f);
}

} // namespace wave

#pragma once
#include "detail/config.hpp"
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// wave_ballot.hpp
//
// Vote and predicate operations — correctly 64-bit on wave64.
//
// Critical difference from CUDA:
//   CUDA __ballot_sync → uint32_t (32-lane warp)
//   HIP  __ballot       → uint64_t (covers 64-lane wavefront on CDNA)
//
// We always return uint64_t. On NVIDIA the upper 32 bits are zero.
// This means ballot results are not bit-comparable across platforms,
// but popcount() and lane-index operations work correctly on both.
// ---------------------------------------------------------------------------

namespace wave {

// --- activemask -------------------------------------------------------------
// Returns a bitmask of currently active lanes in the wavefront.
// On wave64: up to 64 bits set. On wave32/NVIDIA: up to 32 bits set.

__device__ __forceinline__ wp_mask_t activemask() {
#if WP_AMD
    return static_cast<wp_mask_t>(__activemask());
#else
    return static_cast<wp_mask_t>(__activemask());
#endif
}

// --- ballot -----------------------------------------------------------------
// Returns a bitmask with bit N set if lane N evaluated predicate as true.

__device__ __forceinline__ wp_mask_t ballot(int predicate) {
#if WP_AMD
    return static_cast<wp_mask_t>(__ballot(predicate));
#else
    return static_cast<wp_mask_t>(__ballot_sync(WP_FULL_MASK, predicate));
#endif
}

// --- any --------------------------------------------------------------------
// Returns non-zero if any active lane has predicate != 0.

__device__ __forceinline__ int any(int predicate) {
#if WP_AMD
    return __any(predicate);
#else
    return __any_sync(WP_FULL_MASK, predicate);
#endif
}

// --- all --------------------------------------------------------------------
// Returns non-zero if all active lanes have predicate != 0.

__device__ __forceinline__ int all(int predicate) {
#if WP_AMD
    return __all(predicate);
#else
    return __all_sync(WP_FULL_MASK, predicate);
#endif
}

// --- popcount ---------------------------------------------------------------
// Number of lanes with predicate true — wave-size independent.

__device__ __forceinline__ int popcount(int predicate) {
    return __builtin_popcountll(ballot(predicate));
}

// --- lane_mask_lt / lane_mask_le --------------------------------------------
// Masks useful for prefix operations: bits set for lanes < this lane.
// Works correctly for both wave32 (up to bit 31) and wave64 (up to bit 63).

__device__ __forceinline__ wp_mask_t lane_mask_lt() {
    // (1 << lane_id) - 1 but using 64-bit arithmetic
    const unsigned int lid = lane_id();
    return (lid == 0) ? 0ULL : ((1ULL << lid) - 1ULL);
}

__device__ __forceinline__ wp_mask_t lane_mask_le() {
    return (1ULL << lane_id());
}

} // namespace wave

// Pull in lane_id dependency
#include "wave_shuffle.hpp"

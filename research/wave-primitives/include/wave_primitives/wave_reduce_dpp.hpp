#pragma once
#include "detail/config.hpp"
#include <hip/hip_runtime.h>
#include <type_traits>
#include <limits>

// ---------------------------------------------------------------------------
// wave_reduce_dpp.hpp
//
// CDNA3-optimized wave64 reductions using the DPP (Data-Parallel Primitives)
// cross-lane datapath exclusively — NO ds_bpermute, NO LDS roundtrip.
//
// MOTIVATION (measured on MI300X, gfx942, ROCm 7.2 — see docs/research-outline.md §10):
//   The portable wave::reduce_sum (built on __shfl_down) lowers to 6×
//   ds_bpermute_b32 — every step round-trips through LDS → 922µs (kReps=4096).
//   The full-DPP version below lowers to 6× v_add_f32_dpp (zero bpermute) and
//   beats hipCUB by 1.35–1.79×, and the portable path by 4.39×.
//
// DPP control codes (gfx942), verified empirically (tests/probe_dpp*.hip):
//   row_shr:n  = 0x110 | n   lane i receives lane (i-n) within its 16-lane row;
//                            out-of-row lanes read 'old'. Per-row result lands
//                            on the HIGH lane of each row (15,31,47,63).
//   row_bcast15 = 0x142      lane15→[16..31], lane31→[32..47], lane47→[48..63]
//   row_bcast31 = 0x143      lane31→[32..63]
//
// Geometry (result lands on lane 63):
//   row_shr 1,2,4,8  → per-row reduction on lanes 15,31,47,63
//   row_bcast15      → propagate across 16-lane row boundaries
//   row_bcast31      → propagate across the 32-lane bank boundary
//
// IDENTITY ELEMENT (critical for correctness):
//   bound_ctrl=false makes out-of-range lanes read the 'old' operand. That old
//   value is combined into the result, so it MUST be the identity of the op:
//     sum → 0,  max → lowest(T),  min → highest(T).
//   Passing 0 as 'old' for a max over negative numbers would be a silent bug.
//
// 64-bit types (double, int64): we pass them straight to __builtin_amdgcn_update_dpp
//   and rely on the backend to split into two 32-bit DPP ops. This is the
//   "direct 64-bit" path chosen for Run 2 validation. If correctness fails on
//   hardware, fall back to the explicit hi/lo split (see split_dpp64 below,
//   currently unused, kept for the fallback).
//
// NOTE: result is on lane 63 (highest lane), not lane 0. Use the *_bcast
//       variants to get the result on every lane.
// ---------------------------------------------------------------------------

namespace wave {
namespace dpp {

#if WP_AMD

// --- identity helpers -------------------------------------------------------
// std::numeric_limits members are constexpr and callable in device code for the
// arithmetic types we use (float/double/int32/int64). We keep them but provide
// the sum identity as a plain cast so it works for any numeric T.
template <typename T> __device__ __forceinline__ T id_sum() { return T(0); }
template <typename T> __device__ __forceinline__ T id_max() { return std::numeric_limits<T>::lowest(); }
template <typename T> __device__ __forceinline__ T id_min() { return std::numeric_limits<T>::max(); }

// --- generic DPP move (direct 64-bit path; backend splits as needed) --------
template <int Ctrl, typename T>
__device__ __forceinline__ T dpp_move(T v, T old_val) {
    return __builtin_amdgcn_update_dpp(old_val, v, Ctrl, 0xf, 0xf, false);
}

// The 6-step DPP combine, parameterized on a binary op and its identity.
// Op is a functor: T operator()(T a, T b).
template <typename T, typename Op>
__device__ __forceinline__ T dpp_reduce(T v, Op op, T identity) {
    v = op(v, dpp_move<0x111>(v, identity));  // row_shr 1
    v = op(v, dpp_move<0x112>(v, identity));  // row_shr 2
    v = op(v, dpp_move<0x114>(v, identity));  // row_shr 4
    v = op(v, dpp_move<0x118>(v, identity));  // row_shr 8  → row results on 15,31,47,63
    v = op(v, dpp_move<0x142>(v, identity));  // row_bcast15
    v = op(v, dpp_move<0x143>(v, identity));  // row_bcast31
    return v;                                  // result on lane 63
}

// --- broadcast lane 63 to all lanes (works for 32- and 64-bit) --------------
// readlane operates on a 32-bit int; we reinterpret the bits of T (float/int/
// double/long) through int halves so no value is lost. Templated on size so the
// 32- and 64-bit paths are selected at compile time.
template <typename T>
__device__ __forceinline__ T bcast_lane63(T v) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "bcast_lane63: only 32/64-bit types");
    if constexpr (sizeof(T) == 4) {
        int bits = __builtin_amdgcn_readlane(reinterpret_cast<int&>(v), 63);
        return reinterpret_cast<T&>(bits);
    } else {
        int* p = reinterpret_cast<int*>(&v);
        int lo = __builtin_amdgcn_readlane(p[0], 63);
        int hi = __builtin_amdgcn_readlane(p[1], 63);
        T out;
        int* q = reinterpret_cast<int*>(&out);
        q[0] = lo; q[1] = hi;
        return out;
    }
}

// --- op functors ------------------------------------------------------------
struct SumOp { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a + b; } };
struct MaxOp { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a > b ? a : b; } };
struct MinOp { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? a : b; } };

// --- public reductions (result on lane 63) ---------------------------------
template <typename T> __device__ __forceinline__ T reduce_sum(T v) { return dpp_reduce(v, SumOp{}, id_sum<T>()); }
template <typename T> __device__ __forceinline__ T reduce_max(T v) { return dpp_reduce(v, MaxOp{}, id_max<T>()); }
template <typename T> __device__ __forceinline__ T reduce_min(T v) { return dpp_reduce(v, MinOp{}, id_min<T>()); }

// --- broadcast variants (result on all lanes) ------------------------------
template <typename T> __device__ __forceinline__ T reduce_sum_bcast(T v) { return bcast_lane63(reduce_sum(v)); }
template <typename T> __device__ __forceinline__ T reduce_max_bcast(T v) { return bcast_lane63(reduce_max(v)); }
template <typename T> __device__ __forceinline__ T reduce_min_bcast(T v) { return bcast_lane63(reduce_min(v)); }

#else  // wave32 / NVIDIA fallback — plain shuffle reductions (result on lane 0).

template <typename T> __device__ __forceinline__ T reduce_sum(T v) {
    for (int o = warpSize >> 1; o > 0; o >>= 1) v += __shfl_down(v, o, warpSize); return v;
}
template <typename T> __device__ __forceinline__ T reduce_max(T v) {
    for (int o = warpSize >> 1; o > 0; o >>= 1) { T x = __shfl_down(v, o, warpSize); v = x > v ? x : v; } return v;
}
template <typename T> __device__ __forceinline__ T reduce_min(T v) {
    for (int o = warpSize >> 1; o > 0; o >>= 1) { T x = __shfl_down(v, o, warpSize); v = x < v ? x : v; } return v;
}
template <typename T> __device__ __forceinline__ T reduce_sum_bcast(T v) { return __shfl(reduce_sum(v), 0, warpSize); }
template <typename T> __device__ __forceinline__ T reduce_max_bcast(T v) { return __shfl(reduce_max(v), 0, warpSize); }
template <typename T> __device__ __forceinline__ T reduce_min_bcast(T v) { return __shfl(reduce_min(v), 0, warpSize); }

#endif

} // namespace dpp
} // namespace wave

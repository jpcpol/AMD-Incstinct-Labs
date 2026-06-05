#pragma once
#include "detail/config.hpp"
#include <hip/hip_runtime.h>
#include <limits>

// ---------------------------------------------------------------------------
// wave_scan_dpp.hpp
//
// CDNA3 wave64 inclusive/exclusive prefix scan — FULL-DPP, zero LDS, generalized
// over type and op. Measured on MI300X VF (ROCm 7.2): 2.61× over the portable
// shfl_up scan and 1.028× over hipCUB WarpScan, with SQ_INSTS_LDS = 0.
//
// Geometry (verified empirically — probe_bcast_map.hip, probe_scan_fulldpp.hip):
//   in-row: row_shr 1,2,4,8 → each 16-lane row holds its local prefix; row
//           totals land on lanes 15,31,47,63.
//   cross-row carry WITHOUT shfl: row_bcast15 (0x142) shifts each row's total
//           down to the NEXT row (16-31<-l15, 32-47<-l31, 48-63<-l47). Applying
//           it repeatedly to the carry moves a total one more row down each time:
//             c1  = bcast15(v)    → row1:t0  row2:t1  row3:t2
//             c1b = bcast15(c1)   → row2:t0  row3:t1
//             c1c = bcast15(c1b)  → row3:t0
//           accumulated carry: row1=t0; row2=t0+t1; row3=t0+t1+t2. Combined into
//           v exactly once. No mutated-v re-read → no double counting.
// ---------------------------------------------------------------------------

namespace wave {
namespace dpp {

#if WP_AMD

template <typename T> __device__ __forceinline__ T scan_id_sum() { return T(0); }
template <typename T> __device__ __forceinline__ T scan_id_max() { return std::numeric_limits<T>::lowest(); }
template <typename T> __device__ __forceinline__ T scan_id_min() { return std::numeric_limits<T>::max(); }

struct ScanSum { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a + b; } };
struct ScanMax { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a > b ? a : b; } };
struct ScanMin { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? a : b; } };

__device__ __forceinline__ unsigned scan_lane_id() {
    return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}

template <int Shift, typename T, typename Op>
__device__ __forceinline__ T scan_dpp_step(T v, Op op, T identity) {
    // Assign DPP result to T before combining (64-bit operands otherwise break
    // Op's type deduction).
    T moved = __builtin_amdgcn_update_dpp(identity, v, 0x110 + Shift, 0xf, 0xf, false);
    return op(v, moved);
}

template <typename T, typename Op>
__device__ __forceinline__ T scan_inclusive(T v, Op op, T identity) {
    const unsigned L = scan_lane_id();
    // In-row prefix (offsets 1,2,4,8) via DPP. Row totals on lanes 15,31,47,63.
    v = scan_dpp_step<1>(v, op, identity);
    v = scan_dpp_step<2>(v, op, identity);
    v = scan_dpp_step<4>(v, op, identity);
    v = scan_dpp_step<8>(v, op, identity);
    // Cross-row carry via cascaded row_bcast15 (no LDS, no shfl).
    T c1  = __builtin_amdgcn_update_dpp(identity, v,   0x142, 0xf, 0xf, false);
    T c1b = __builtin_amdgcn_update_dpp(identity, c1,  0x142, 0xf, 0xf, false);
    T c1c = __builtin_amdgcn_update_dpp(identity, c1b, 0x142, 0xf, 0xf, false);
    T carry = identity;
    if (L >= 16) carry = c1;
    if (L >= 32) carry = op(carry, c1b);
    if (L >= 48) carry = op(carry, c1c);
    return (L >= 16) ? op(v, carry) : v;
}

template <typename T> __device__ __forceinline__ T scan_inclusive_sum(T v) { return scan_inclusive(v, ScanSum{}, scan_id_sum<T>()); }
template <typename T> __device__ __forceinline__ T scan_inclusive_max(T v) { return scan_inclusive(v, ScanMax{}, scan_id_max<T>()); }
template <typename T> __device__ __forceinline__ T scan_inclusive_min(T v) { return scan_inclusive(v, ScanMin{}, scan_id_min<T>()); }

template <typename T>
__device__ __forceinline__ T scan_exclusive_sum(T v) {
    // Full-DPP exclusive scan: no shfl_up, no bpermute.
    // lane L needs inc[L-1] (or 0 if L==0).
    // row_shr1 on inc gives inc[L-1] within each 16-lane row, but resets to identity=0
    // at the start of each new row (lanes 16, 32, 48). Fix those with row_bcast15 of inc,
    // which broadcasts inc[15]→lanes16..31, inc[31]→32..47, inc[47]→48..63.
    T inc = scan_inclusive_sum(v);
    T ex    = __builtin_amdgcn_update_dpp(T(0), inc, 0x111, 0xf, 0xf, false);  // row_shr1
    T carry = __builtin_amdgcn_update_dpp(T(0), inc, 0x142, 0xf, 0xf, false);  // row_bcast15
    const unsigned L = scan_lane_id();
    if ((L & 15u) == 0u && L != 0u) ex = carry;  // patch lanes 16, 32, 48
    return ex;
}

#else  // wave32 / NVIDIA fallback — portable Hillis-Steele.

template <typename T, typename Op>
__device__ __forceinline__ T scan_inclusive(T v, Op op, T) {
    const unsigned lid = (threadIdx.x & (warpSize - 1));
    for (int o = 1; o < warpSize; o <<= 1) { T u = __shfl_up(v, o, warpSize); if (lid >= (unsigned)o) v = op(v, u); }
    return v;
}
struct ScanSum { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a + b; } };
struct ScanMax { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a > b ? a : b; } };
struct ScanMin { template <typename T> __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? a : b; } };
template <typename T> __device__ __forceinline__ T scan_inclusive_sum(T v) { return scan_inclusive(v, ScanSum{}, T(0)); }
template <typename T> __device__ __forceinline__ T scan_inclusive_max(T v) { return scan_inclusive(v, ScanMax{}, std::numeric_limits<T>::lowest()); }
template <typename T> __device__ __forceinline__ T scan_inclusive_min(T v) { return scan_inclusive(v, ScanMin{}, std::numeric_limits<T>::max()); }
template <typename T> __device__ __forceinline__ T scan_exclusive_sum(T v) {
    T inc = scan_inclusive_sum(v); T ex = __shfl_up(inc, 1, warpSize);
    return ((threadIdx.x & (warpSize - 1)) == 0) ? T(0) : ex;
}

#endif

} // namespace dpp
} // namespace wave

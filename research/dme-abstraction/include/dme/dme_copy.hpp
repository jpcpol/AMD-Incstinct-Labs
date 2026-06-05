#pragma once
// ---------------------------------------------------------------------------
// dme_copy.hpp — C++ API over the CDNA3 Data Movement Engine (DME)
//
// The DME is a dedicated hardware engine on CDNA3 (gfx942) for async
// HBM → LDS copies. It bypasses the VMEM pipeline, freeing VALU/MFMA
// units to compute while data is prefetched into LDS.
//
// Without DME: load → stall → compute  (serial, VMEM blocks VALU)
// With DME:    async prefetch ──────────────┐
//              compute (MFMA) ──────────────┤ overlap
//              barrier + use data ──────────┘
//
// HARDWARE REQUIREMENT: CDNA3+ (gfx942). Does NOT work on RDNA or CDNA2.
//
// UNDERLYING INTRINSIC (ROCm 7.x / gfx942) — VERIFIED SEMANTICS:
//   void __builtin_amdgcn_global_load_lds(
//       void* global_ptr,  // per-lane HBM source (addrspace 1 or 7)
//       void* lds_ptr,     // per-lane LDS destination (addrspace 3)
//       int   size,        // bytes: 1, 2, or 4 ONLY (immediate, constant)
//       int   offset,      // immediate byte offset — must be 0 for runtime routing
//       int   aux          // cache policy: 0=default, 1=sc0 (streaming, recommended)
//   );
//
// VERIFIED CONSTRAINTS (probe_dme.hip, gfx942, ROCm 7.2, 2026-06-05):
//   1. ROUTING: per-lane pointer advance is the correct model. offset=0 always.
//      Each lane must advance BOTH src and dst pointers independently.
//   2. OFFSET: must be compile-time constant. Runtime values rejected at compile.
//      Do NOT use offset for per-lane routing — use the pointer.
//   3. SIZE: valid values are 1, 2, 4 bytes only. size=8 is NOT supported.
//   4. CACHE: aux=1 (sc0, streaming) is safe and recommended for HBM→LDS prefetch.
//   5. COLLISION: when multiple lanes target the same LDS slot, result is 0 (not
//      undefined, not last-writer). Avoid aliased LDS destinations.
//   6. WAIT: __builtin_amdgcn_s_waitcnt(0) is sufficient after DME loads.
//
// STATUS: probe_dme.hip validates semantics on real hardware (Run 4).
//         bench_dme.hip measures throughput vs memcpy-style baseline (TODO).
// ---------------------------------------------------------------------------

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__AMDGCN__)
#  error "dme_copy.hpp requires AMD HIP (gfx942). CDNA3 only."
#endif

#include <hip/hip_runtime.h>

namespace dme {

// ---------------------------------------------------------------------------
// copy_element — async copy of one 4-byte element from HBM → LDS.
//
// Each lane must provide its own per-lane src and dst pointers.
// offset=0 always; per-lane routing is done via pointer arithmetic.
//
// Usage (per lane in a kernel):
//   const int L = lane_id();
//   dme::copy_element(src + L, &tile[L]);   // each lane routes independently
//   dme::wait();
//   __syncthreads();
//   float v = tile[L];
// ---------------------------------------------------------------------------
// copy_element — the size parameter MUST be an integer literal (compiler constraint).
// Specializations handle float (4 bytes). Extend with if constexpr for other sizes.
template <typename T>
__device__ __forceinline__ void copy_element(
    const T* __restrict__ src_global,  // per-lane HBM source
    T* __restrict__        dst_lds)    // per-lane LDS destination
{
    static_assert(sizeof(T) == 4, "copy_element: T must be 4 bytes on gfx942");
    auto* g = reinterpret_cast<void*>(const_cast<T*>(src_global));
    auto* l = reinterpret_cast<void*>(dst_lds);
    // size=4 as integer literal — sizeof(T) is not accepted by the builtin
    __builtin_amdgcn_global_load_lds(g, l, 4, /*offset=*/0, /*aux=*/0);
}

// copy_element_stream — same as copy_element but with sc0 streaming cache hint.
// Bypasses L1/L2: recommended for large tiles that won't fit in cache.
template <typename T>
__device__ __forceinline__ void copy_element_stream(
    const T* __restrict__ src_global,
    T* __restrict__        dst_lds)
{
    static_assert(sizeof(T) == 4, "copy_element_stream: T must be 4 bytes on gfx942");
    auto* g = reinterpret_cast<void*>(const_cast<T*>(src_global));
    auto* l = reinterpret_cast<void*>(dst_lds);
    __builtin_amdgcn_global_load_lds(g, l, 4, /*offset=*/0, /*aux=*/1);
}

// ---------------------------------------------------------------------------
// wait — wait for all outstanding DME / async LDS loads to complete.
// Must be called (then __syncthreads()) before reading LDS written by copy_element.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wait() {
    __builtin_amdgcn_s_waitcnt(0);  // lgkmcnt(0) + vmcnt(0)
}

// wait_lds — wait only for LDS operations (lighter if no VMEM in flight).
__device__ __forceinline__ void wait_lds() {
    __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt=0, vmcnt=max
}

// ---------------------------------------------------------------------------
// copy_tile_1d — cooperative tile copy: all 64 lanes copy kElems floats
// from HBM to LDS using the DME. kElems must be a multiple of 64.
//
// Each lane handles elements at indices: lane, lane+64, lane+128, ...
// The loop is fully unrolled when kElems is a compile-time constant.
//
// Usage:
//   __shared__ float tile[kElems];
//   dme::copy_tile_1d<float, kElems>(src + block_offset, tile);
//   dme::wait();
//   __syncthreads();
//   // tile[] is now valid
// ---------------------------------------------------------------------------
template <typename T, int kElems>
__device__ __forceinline__ void copy_tile_1d(
    const T* __restrict__ src,
    T* __restrict__       dst_lds)
{
    static_assert(sizeof(T) == 4, "copy_tile_1d: T must be 4 bytes on gfx942");
    static_assert(kElems % 64 == 0, "copy_tile_1d: kElems must be a multiple of 64");
    const int lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    // Per-lane pointer advance: each lane copies its strided elements.
    // offset=0 always (offset must be compile-time constant per LLVM constraint).
    #pragma unroll
    for (int i = 0; i < kElems / 64; ++i) {
        const int idx = lane + i * 64;
        copy_element<T>(src + idx, dst_lds + idx);
    }
}

// copy_tile_1d_stream — same as copy_tile_1d with sc0 streaming cache hint.
template <typename T, int kElems>
__device__ __forceinline__ void copy_tile_1d_stream(
    const T* __restrict__ src,
    T* __restrict__       dst_lds)
{
    static_assert(sizeof(T) == 4, "copy_tile_1d_stream: T must be 4 bytes on gfx942");
    static_assert(kElems % 64 == 0, "copy_tile_1d_stream: kElems must be a multiple of 64");
    const int lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    #pragma unroll
    for (int i = 0; i < kElems / 64; ++i) {
        const int idx = lane + i * 64;
        copy_element_stream<T>(src + idx, dst_lds + idx);
    }
}

} // namespace dme

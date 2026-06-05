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
// UNDERLYING INTRINSIC (ROCm 7.x / gfx942):
//   void __builtin_amdgcn_global_load_lds(
//       void* global_ptr,  // per-lane HBM source (addrspace 1 or 7)
//       void* lds_ptr,     // uniform LDS destination (addrspace 3)
//       int   size,        // bytes: 1, 2, 4, or 8 (immediate)
//       int   offset,      // immediate byte offset applied to both ptrs
//       int   aux          // cache policy flags (sc0 recommended on gfx942)
//   );
//
// STATUS: probe_dme.hip validates correctness on real hardware.
//         bench_dme.hip measures throughput vs memcpy-style baseline.
// ---------------------------------------------------------------------------

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__AMDGCN__)
#  error "dme_copy.hpp requires AMD HIP (gfx942). CDNA3 only."
#endif

#include <hip/hip_runtime.h>

namespace dme {

// ---------------------------------------------------------------------------
// copy_tile — async copy of a 2D tile from HBM → LDS using the DME.
//
// Each lane copies one element. The copy is initiated on all lanes; the caller
// must issue a barrier (dme::wait() or __syncthreads()) before reading LDS.
//
// Template param T must be 4 bytes (float / int32) for gfx942.
// For 8-byte types (double) use copy_tile_64.
//
// Parameters:
//   src      — pointer to HBM source (per-lane: src + lane * stride_bytes)
//   dst_lds  — pointer to LDS destination (addrspace 3, uniform base)
//   lane_offset_bytes — byte offset from dst_lds for this lane's element
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void copy_element(
    const T* __restrict__ src_global,
    T* __restrict__        dst_lds,
    int                    offset_bytes = 0)
{
    static_assert(sizeof(T) == 4, "copy_element: T must be 4 bytes on gfx942");
    // Cast to addrspace 1 (global) and addrspace 3 (LDS) as required by the builtin.
    auto* g = reinterpret_cast<void*>(const_cast<T*>(src_global));
    auto* l = reinterpret_cast<void*>(dst_lds);
    // aux=0: default cache policy. Use aux=1 (sc0) for streaming (non-temporal).
    __builtin_amdgcn_global_load_lds(g, l, sizeof(T), offset_bytes, /*aux=*/0);
}

// copy_element streaming — sc0 cache hint: bypass L1/L2, good for one-shot tiles.
template <typename T>
__device__ __forceinline__ void copy_element_stream(
    const T* __restrict__ src_global,
    T* __restrict__        dst_lds,
    int                    offset_bytes = 0)
{
    static_assert(sizeof(T) == 4, "copy_element_stream: T must be 4 bytes on gfx942");
    auto* g = reinterpret_cast<void*>(const_cast<T*>(src_global));
    auto* l = reinterpret_cast<void*>(dst_lds);
    __builtin_amdgcn_global_load_lds(g, l, sizeof(T), offset_bytes, /*aux=*/1);
}

// ---------------------------------------------------------------------------
// wait — wait for all outstanding DME / async LDS loads to complete.
// Must be called before reading data written by copy_element.
// Equivalent to s_waitcnt lgkmcnt(0) + vmcnt(0).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wait() {
    // lgkmcnt(0) waits for all LDS and GDS operations.
    // vmcnt(0)   waits for all VMEM operations.
    __builtin_amdgcn_s_waitcnt(/*lgkmcnt(0) vmcnt(0)*/ 0);
}

// wait_lds — wait only for LDS operations (lighter than full wait).
__device__ __forceinline__ void wait_lds() {
    // lgkmcnt(0) only — faster if no VMEM ops are in flight.
    // encoding: lgkmcnt in bits [11:8], vmcnt in bits [3:0]
    // lgkmcnt=0, vmcnt=max(0xf) → 0x00f = only wait LDS
    __builtin_amdgcn_s_waitcnt(0xc07f);  // vmcnt=max, lgkmcnt=0
}

// ---------------------------------------------------------------------------
// copy_tile_1d — cooperative tile copy: all lanes in the wave copy
// consecutive elements from HBM to LDS. Covers kElems elements total.
//
// Usage:
//   __shared__ float lds_tile[kElems];
//   dme::copy_tile_1d<float, kElems>(src + block_offset, lds_tile);
//   dme::wait();
//   // now lds_tile[] is valid
// ---------------------------------------------------------------------------
template <typename T, int kElems>
__device__ __forceinline__ void copy_tile_1d(
    const T* __restrict__ src,
    T* __restrict__       dst_lds)
{
    static_assert(kElems % 64 == 0 || kElems <= 64,
        "copy_tile_1d: kElems should be a multiple of wave size or ≤ 64");
    const int lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    // Each lane copies one element per iteration.
    // offset param applies to BOTH ptrs inside the builtin (confirmed by probe_dme).
    // Pass the base pointers and let offset drive both src and dst advancement.
    #pragma unroll
    for (int i = lane; i < kElems; i += 64) {
        copy_element<T>(src, dst_lds, i * sizeof(T));
    }
}

} // namespace dme

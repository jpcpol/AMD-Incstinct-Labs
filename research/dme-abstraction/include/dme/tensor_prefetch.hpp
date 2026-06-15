// ---------------------------------------------------------------------------
// dme/tensor_prefetch.hpp — 2D tensor tile prefetch via the DME.
//
// Abstracts the double-buffer K/V pattern from fa_dme.hip into a reusable
// cooperative API. Each call issues async DME loads for one contiguous tile
// of a strided 2D tensor. A wave (64 lanes) or a cooperative thread block
// can call this from device code.
//
// HARDWARE REQUIREMENT: CDNA3+ (gfx942). Requires dme_copy.hpp.
//
// Validated pattern (fa_dme.hip, MI300X, Run 4):
//   - K/V tiles are [Bc, D] fp16 in HBM, stored as [N, D] row-major.
//   - Each tile reinterpreted as [Bc*D/2] floats for 4-byte DME transfers.
//   - Prefetch overlaps with compute on the prior tile (double-buffer ping/pong).
//   - sc0 streaming hint (aux=1) recommended: tiles don't fit in L2 on MI300X.
//
// USAGE — cooperative tile prefetch (all threads in block participate):
//
//   // Declaration of double buffers in shared memory:
//   __shared__ __half K_buf[2][Bc * D];
//   __shared__ __half V_buf[2][Bc * D];
//
//   // Prologue: load tile 0 before loop
//   dme::prefetch_tile2d_fp16<Bc, D>(Kbase, K_buf[0], tile_row=0, stride_row=D);
//   dme::prefetch_tile2d_fp16<Bc, D>(Vbase, V_buf[0], tile_row=0, stride_row=D);
//   dme::wait(); __syncthreads();
//
//   for (int j = 0; j < N/Bc; ++j) {
//       int cur = j & 1, next = cur ^ 1;
//       // Async prefetch tile j+1 into buf[next]
//       if (j+1 < N/Bc)
//           dme::prefetch_tile2d_fp16<Bc,D>(Kbase, K_buf[next], (j+1)*Bc, D);
//       // ... compute on K_buf[cur] ...
//       dme::wait(); __syncthreads();  // barrier before reading buf[next]
//   }
//
// For the KV-cache layout [n_heads, max_seq, D] used in the cdna3 runtime,
// use prefetch_tile2d_fp16 — within a single head, key positions are contiguous
// (stride between consecutive keys = D, equal to kCols). stride_kv = max_seq is
// only the head-to-head stride, not the key-to-key stride within a head.
// ---------------------------------------------------------------------------
#pragma once

#include "dme_copy.hpp"  // dme::copy_element_stream, dme::wait

namespace dme {

// ---------------------------------------------------------------------------
// prefetch_tile2d_fp16<kRows, kCols>
//
// Cooperatively copies a [kRows, kCols] fp16 tile from HBM to LDS using DME.
//
//   base      : pointer to HBM tile origin (row tile_row0, column 0).
//               Caller computes: Kbase + tile_row0 * kCols.
//   dst_lds   : destination in shared memory (must be [kRows * kCols] __half).
//   tid       : per-thread flat index (warp*warpsize + lane, or just threadIdx.x
//               for single-warp kernels). Used to distribute work cooperatively.
//   nthreads  : total participating threads (blockDim.x * blockDim.y or similar).
//
// The tile is treated as a flat array of kRows*kCols fp16 = kRows*kCols/2 float32.
// Each participating thread issues DME copies for its strided elements.
// kRows*kCols must be a multiple of 128 (= 64 * 2) for even float distribution.
// ---------------------------------------------------------------------------
template <int kRows, int kCols>
__device__ __forceinline__ void prefetch_tile2d_fp16(
    const __half* __restrict__ base_hbm,
    __half* __restrict__       dst_lds,
    int tid,
    int nthreads)
{
    static_assert(kRows * kCols % 128 == 0,
        "prefetch_tile2d_fp16: kRows*kCols must be a multiple of 128");
    // Reinterpret: treat 2 fp16 as 1 float32 for 4-byte DME transfers.
    const float* src = reinterpret_cast<const float*>(base_hbm);
    float*       dst = reinterpret_cast<float*>(dst_lds);
    constexpr int kFloats = kRows * kCols / 2;
    for (int i = tid; i < kFloats; i += nthreads)
        copy_element_stream<float>(src + i, dst + i);
}

// ---------------------------------------------------------------------------
// prefetch_tile2d_strided_fp16<kRows, kCols>
//
// Like prefetch_tile2d_fp16 but the source tensor has an arbitrary row stride
// in HBM (stride_row elements between consecutive rows).
//
// Use case: KV-cache [n_heads, max_seq, D] — rows are seq positions,
// stride_row = max_seq (or any value != kCols when tensor is not packed).
//
//   base_hbm  : pointer to element [0, 0] of the tile in HBM.
//               For a cache slice starting at key position k0:
//                 K_layer(li) + head * max_seq * D + k0 * D
//   dst_lds   : flat [kRows * kCols] __half destination in LDS.
//   stride_row: number of __half elements between the start of consecutive rows
//               in HBM (= kCols when packed, = max_seq when cache-strided).
//   tid, nthreads: cooperative distribution as above.
//
// NOTE: Because rows are not contiguous in HBM, we cannot use a flat
// reinterpret cast. Each (row, col) pair is addressed individually via
// base_hbm + row * stride_row + col. The DME still issues async loads.
// For performance: kCols should be a multiple of 64 so each row fills a
// full wave's worth of loads without partial-wave tails.
// ---------------------------------------------------------------------------
template <int kRows, int kCols>
__device__ __forceinline__ void prefetch_tile2d_strided_fp16(
    const __half* __restrict__ base_hbm,
    __half* __restrict__       dst_lds,
    int stride_row,
    int tid,
    int nthreads)
{
    static_assert(kRows * kCols % 128 == 0,
        "prefetch_tile2d_strided_fp16: kRows*kCols must be a multiple of 128");
    // Reinterpret rows in pairs to keep 4-byte transfers.
    // Works when kCols is even (guaranteed by static_assert kRows*kCols%128==0
    // AND by convention: kCols is head_dim, always a power of 2 >= 64).
    constexpr int kFloats = kRows * kCols / 2;
    const int stride_f = stride_row / 2;  // stride in float units
    for (int i = tid; i < kFloats; i += nthreads) {
        // Map flat index i (in float32 units) back to (row, col_pair).
        const int row  = i / (kCols / 2);
        const int colp = i % (kCols / 2);
        const float* src = reinterpret_cast<const float*>(base_hbm)
                         + row * stride_f + colp;
        float* dst = reinterpret_cast<float*>(dst_lds) + i;
        copy_element_stream<float>(src, dst);
    }
}

// ---------------------------------------------------------------------------
// AsyncTile2D — RAII double-buffer manager for one K or V tensor.
//
// Encapsulates the ping/pong state and exposes a clean prefetch+wait interface.
// Used inside attention kernels that own their own LDS buffers.
//
// Template parameters:
//   kRows, kCols  — tile dimensions in __half elements.
//
// Usage (cooperative across all threads in block):
//
//   __shared__ __half K_buf[2][Bc * D];
//   dme::AsyncTile2D<Bc, D> ktile;
//   ktile.prologue(Kbase, K_buf, /*tile_idx=*/0, tid, nthreads);
//   for (int j = 0; j < N/Bc; ++j) {
//       ktile.prefetch_next(Kbase, K_buf, j+1, tid, nthreads, N/Bc);
//       // ... compute on ktile.cur_buf(K_buf) ...
//       ktile.wait_next();  // barrier before next iter reads buf[next]
//       ktile.advance();
//   }
// ---------------------------------------------------------------------------
template <int kRows, int kCols>
struct AsyncTile2D {
    int cur_ = 0;

    __device__ __forceinline__ int cur()  const { return cur_; }
    __device__ __forceinline__ int next() const { return cur_ ^ 1; }
    __device__ __forceinline__ void advance() { cur_ ^= 1; }

    // Prologue: load tile 0 synchronously (no compute to overlap yet).
    __device__ __forceinline__ void prologue(
        const __half* __restrict__ base_hbm,
        __half dst_lds[][kRows * kCols],
        int tid, int nthreads)
    {
        prefetch_tile2d_fp16<kRows, kCols>(base_hbm, dst_lds[0], tid, nthreads);
        wait();
        __syncthreads();
        cur_ = 0;
    }

    // Prefetch tile (j+1) into buf[next] — async, does NOT stall.
    // Call this at the top of iteration j, before computing on cur.
    __device__ __forceinline__ void prefetch_next(
        const __half* __restrict__ base_hbm,
        __half dst_lds[][kRows * kCols],
        int next_tile_idx,
        int n_tiles,
        int tid, int nthreads)
    {
        if (next_tile_idx < n_tiles) {
            const __half* src = base_hbm + (size_t)next_tile_idx * kRows * kCols;
            prefetch_tile2d_fp16<kRows, kCols>(src, dst_lds[next()], tid, nthreads);
        }
    }

    // Barrier before the next iteration reads buf[next].
    // Only issues wait+sync if there was a prefetch in flight.
    __device__ __forceinline__ void wait_next(int next_tile_idx, int n_tiles) {
        if (next_tile_idx < n_tiles) {
            wait();
            __syncthreads();
        } else {
            __syncthreads();
        }
    }
};

} // namespace dme

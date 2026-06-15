// ---------------------------------------------------------------------------
// examples/async_tiled_gemm.hpp — DME-prefetched MFMA tile loop building block.
//
// Extracted from the validated Flash Attention inner loop (paper §5.8, fa_dme/fa_mfma):
// the pattern that prefetches the NEXT K/V tile via DME async copy while the CURRENT
// tile's MFMA accumulation runs, hiding HBM→LDS latency behind compute. In FA this
// delivered −44.7% VMEM reads (the overlap signature) and was a key part of the 18%
// end-to-end speedup.
//
// This is a COMPOSITION EXAMPLE — it shows wave/dme/mfma composing into the canonical
// CDNA3 "async-load + matrix-core" loop. It is intentionally a skeleton with the
// barriers and prefetch placement spelled out, because getting THOSE right (not the
// MFMA math) is what the DME advantage depends on.
//
//   #include <cdna3/cdna3.hpp>
//   #include <cdna3/../examples/async_tiled_gemm.hpp>
// ---------------------------------------------------------------------------
#pragma once
#include "../include/cdna3/cdna3.hpp"

namespace cdna3 {
namespace ex {

// Double-buffered tile loop pattern (pseudocode-as-code; D, Bc, tile counts are the
// caller's). The point is the ORDER: issue the next-tile DME prefetch BEFORE the
// current-tile compute, then wait_lds() before consuming the prefetched tile.
//
// Usage sketch (one wave, K/V tiles of Bc rows × D cols in LDS, ping-pong buffers):
//
//   __shared__ __half Kbuf[2][Bc*D], Vbuf[2][Bc*D];
//   int cur = 0;
//   // prime: load tile 0 synchronously (or via dme + wait)
//   load_tile(Kbuf[0], Vbuf[0], 0);
//   for (int t = 0; t < nTiles; ++t) {
//       int nxt = cur ^ 1;
//       if (t + 1 < nTiles) {                         // prefetch next tile async
//           dme::copy_tile_1d_stream<__half, Bc*D>(Kbuf[nxt], K_global + (t+1)*Bc*D);
//           dme::copy_tile_1d_stream<__half, Bc*D>(Vbuf[nxt], V_global + (t+1)*Bc*D);
//       }
//       // ---- compute on the CURRENT tile while the prefetch is in flight ----
//       mfma_accumulate(Kbuf[cur], Vbuf[cur]);        // your Q·Kᵀ / P·V via cdna3::mfma
//       if (t + 1 < nTiles) dme::wait_lds();           // ensure prefetch landed
//       cur = nxt;
//   }
//
// The two rules the validated kernel encodes:
//   1. Issue the prefetch for tile t+1 BEFORE computing tile t (overlap window).
//   2. wait_lds() only AFTER the compute, just before consuming — not earlier (that
//      would serialize and erase the overlap, the bug that loses the −44.7%).
//
// A concrete single-tile MFMA accumulate using the verified mapping:
__device__ __forceinline__ void mma_tile_accumulate(
        const __half* A_lds, const __half* B_lds, float* C_lds, int lane,
        int ldA, int ldB, int ldC){
    mfma::fp16x4 a = mfma::load_A(A_lds, lane, ldA);
    mfma::fp16x4 b = mfma::load_B(B_lds, lane, ldB);
    mfma::f32x4  c = {0,0,0,0};
    c = mfma::mma(a, b, c);
    mfma::store_C(C_lds, c, lane, ldC);
}

} // namespace ex
} // namespace cdna3

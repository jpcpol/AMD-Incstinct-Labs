// ---------------------------------------------------------------------------
// cdna3/detail/fa_decode_dme_kernel.hpp — decode attention with DME prefetch.
//
// Drop-in upgrade of fa_decode_kernel.hpp that:
//   1. Accepts KV-cache with stride_kv (= max_seq) instead of contiguous N×D.
//      Eliminates the pack_kv_kernel round-trip in session.hpp.
//   2. Uses dme::prefetch_tile2d_fp16 to async-load the next K/V tile
//      from HBM while computing the dot-product on the current tile.
//
// Layout contract (same as cdna3::runtime::KVCache):
//   K[kvhead, max_seq, D]   stride_kv = max_seq (head-to-head stride only)
//   V[kvhead, max_seq, D]   stride_kv = max_seq
//   Caller passes K_layer(li, 0) — pointer to element [kvhead=0, pos=0, d=0].
//
// Key insight: within a single KV head, positions k0..k0+kBc-1 are stored
// contiguously as Kh[k0*D .. (k0+kBc)*D-1]. The packed prefetch_tile2d_fp16
// is correct; stride_kv only appears in the per-head base pointer calculation.
//
// DME tiling:
//   Tile size: kBc keys per tile (= 64, one wave width in key dimension).
//   Each tile is [kBc, D] fp16 = kBc*D/2 float32 elements for 4-byte DME.
//   Two LDS double-buffers (K_buf[2][kBc*D] + V_buf[2][kBc*D]).
//   LDS: 2×kBc×D×2 + 2×kBc×D×2 bytes per wave group. For D=64, kBc=64:
//        2×64×64×2 + 2×64×64×2 = 32 KB.
//
// Compatibility: same grid/block as fa_decode_kernel (grid=nQHeads, block=W*64).
//   Requires CDNA3 (gfx942).
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../cdna3.hpp"
// DME path relative to cdna3-prims/include/cdna3/detail/
// → ../../../../research/dme-abstraction/include
#include "../../../../research/dme-abstraction/include/dme/tensor_prefetch.hpp"

using wave::dpp::reduce_sum_bcast;

// DecodeDmeCfg is defined in cdna3::attn namespace (attention.hpp).
// Declared here as a forward for use inside the __global__ template.
// The struct body is in attention.hpp to keep it co-located with DecodeCfg.
// We need a local alias to avoid a circular include; declare an identical struct
// in the global namespace and static_assert field layout matches.
struct DecodeDmeCfg {
    int D, N, nQHeads, nKVHeads;
    int stride_kv;
};

// ---------------------------------------------------------------------------
// fa_decode_dme_kernel<D, W, kBc>
//
// Template parameters:
//   D     — head dimension (64 or 128)
//   W     — number of waves per block (split-K factor)
//   kBc   — keys per DME tile (must be multiple of 8).
//            Each wave has private K+V double-buffers: W×2×kBc×D×2 bytes.
//            D=64,  kBc=16, W=4: 4×2×16×64×2  = 16 KB + merge 1 KB OK
//            D=128, kBc=8,  W=4: 4×2×8×128×2  = 16 KB + merge 2 KB OK
//
//   Key design: each wave has private LDS buffers (K_buf[W][2][kBc*D]) so
//   split-K waves don't race over the same tile. DME prefetch is per-wave.
// ---------------------------------------------------------------------------
template<int D, int W, int kBc=16>
__global__ void fa_decode_dme_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ o,
    DecodeDmeCfg c)
{
    static_assert(kBc % 8 == 0, "kBc must be a multiple of 8");
    static_assert(W*2*kBc*D*2 + W*D*4 + W*4*2 <= 65536,
        "fa_decode_dme_kernel: LDS budget exceeded — reduce kBc or W");
    constexpr int EPL  = D / 64;

    const int wave   = threadIdx.x / 64;
    const int L      = threadIdx.x % 64;
    const int qhead  = blockIdx.x;
    const int group  = c.nQHeads / c.nKVHeads;
    const int kvhead = qhead / group;

    // Per-head pointers into the strided KV-cache.
    // K[kvhead, seq, d] = K_base + kvhead * stride_kv * D + seq * D + d.
    const __half* Kh = K + (size_t)kvhead * c.stride_kv * D;
    const __half* Vh = V + (size_t)kvhead * c.stride_kv * D;
    const __half* qh = q + (size_t)qhead * D;
    __half*       oh = o + (size_t)qhead * D;

    // Load Q into registers.
    float qreg[EPL];
    #pragma unroll
    for (int e = 0; e < EPL; ++e) qreg[e] = __half2float(qh[L * EPL + e]);

    const float scale = 1.f / sqrtf((float)D);

    // Split-K: wave processes keys [k0, k1).
    int per = (c.N + W - 1) / W;
    int k0  = wave * per;
    int k1  = k0 + per < c.N ? k0 + per : c.N;
    int n_tiles = (k1 > k0) ? (k1 - k0 + kBc - 1) / kBc : 0;

    float m_w = -1e38f, l_w = 0.f;
    float O_w[EPL];
    #pragma unroll
    for (int e = 0; e < EPL; ++e) O_w[e] = 0.f;

    // Per-wave double-buffer LDS: each wave has its own K+V ping/pong slots.
    // Layout: K_buf[W][2][kBc*D], V_buf[W][2][kBc*D].
    // This prevents split-K waves from racing over the same tile.
    __shared__ __half K_buf[W][2][kBc * D];
    __shared__ __half V_buf[W][2][kBc * D];
    // Split-K merge LDS (wave 0 merges, same as fa_decode_kernel).
    __shared__ float m_sh[W], l_sh[W];
    __shared__ float O_sh[W][D];

    // Each wave independently prefetches its own tiles using its own 64 lanes.
    // tid_wave: lane index within the wave (0..63), used as tid for per-wave prefetch.
    const int tid_wave = L;         // lane index within this wave
    const int nthreads_wave = 64;   // one wave = 64 threads

    if (n_tiles > 0) {
        // ---------------------------------------------------------------------------
        // Prologue: each wave prefetches its own tile 0 into K_buf[wave][0].
        // ---------------------------------------------------------------------------
        {
            const __half* Ksrc0 = Kh + (size_t)k0 * D;
            const __half* Vsrc0 = Vh + (size_t)k0 * D;
            dme::prefetch_tile2d_fp16<kBc, D>(Ksrc0, K_buf[wave][0], tid_wave, nthreads_wave);
            dme::prefetch_tile2d_fp16<kBc, D>(Vsrc0, V_buf[wave][0], tid_wave, nthreads_wave);
            dme::wait();
            __syncthreads();
        }

        // ---------------------------------------------------------------------------
        // Main tile loop
        // ---------------------------------------------------------------------------
        for (int jt = 0; jt < n_tiles; ++jt) {
            const int cur     = jt & 1;
            const int nxt     = cur ^ 1;
            const int tile_k0 = k0 + jt * kBc;
            const int tile_k1 = (tile_k0 + kBc < k1) ? (tile_k0 + kBc) : k1;

            // Each wave async-prefetches its own next tile into K_buf[wave][nxt].
            if (jt + 1 < n_tiles) {
                const int next_k0   = k0 + (jt + 1) * kBc;
                const __half* Ksrc = Kh + (size_t)next_k0 * D;
                const __half* Vsrc = Vh + (size_t)next_k0 * D;
                dme::prefetch_tile2d_fp16<kBc, D>(Ksrc, K_buf[wave][nxt], tid_wave, nthreads_wave);
                dme::prefetch_tile2d_fp16<kBc, D>(Vsrc, V_buf[wave][nxt], tid_wave, nthreads_wave);
            }

            // Compute Q · K_buf[wave][cur] + online softmax
            for (int k = tile_k0; k < tile_k1; ++k) {
                const int ki = k - tile_k0;
                float partial = 0.f;
                #pragma unroll
                for (int e = 0; e < EPL; ++e)
                    partial += qreg[e] * __half2float(K_buf[wave][cur][ki * D + L * EPL + e]);
                float s    = reduce_sum_bcast(partial) * scale;
                float mnew = fmaxf(m_w, s);
                float p    = __expf(s - mnew);
                float resc = __expf(m_w - mnew);
                #pragma unroll
                for (int e = 0; e < EPL; ++e)
                    O_w[e] = O_w[e] * resc + p * __half2float(V_buf[wave][cur][ki * D + L * EPL + e]);
                l_w = l_w * resc + p;
                m_w = mnew;
            }

            // Wait for DME + sync before next iter reads buf[nxt]
            if (jt + 1 < n_tiles) {
                dme::wait();
                __syncthreads();
            } else {
                __syncthreads();
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Split-K merge across W waves via LDS (same as fa_decode_kernel).
    // ---------------------------------------------------------------------------
    if (L == 0) { m_sh[wave] = m_w; l_sh[wave] = l_w; }
    #pragma unroll
    for (int e = 0; e < EPL; ++e) O_sh[wave][L * EPL + e] = O_w[e];
    __syncthreads();

    if (wave == 0) {
        float m = -1e38f;
        #pragma unroll
        for (int w = 0; w < W; ++w) m = fmaxf(m, m_sh[w]);
        float l = 0.f;
        float acc[EPL];
        #pragma unroll
        for (int e = 0; e < EPL; ++e) acc[e] = 0.f;
        #pragma unroll
        for (int w = 0; w < W; ++w) {
            float f = __expf(m_sh[w] - m);
            l += f * l_sh[w];
            #pragma unroll
            for (int e = 0; e < EPL; ++e) acc[e] += f * O_sh[w][L * EPL + e];
        }
        float inv = (l > 0.f) ? 1.f / l : 1.f;
        #pragma unroll
        for (int e = 0; e < EPL; ++e) oh[L * EPL + e] = __float2half(acc[e] * inv);
    }
}

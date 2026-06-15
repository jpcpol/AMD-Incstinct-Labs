// ---------------------------------------------------------------------------
// cdna3/attention.hpp — unified CDNA3 Flash Attention module (Stage B).
//
// Provides cdna3::attn::prefill(...) and cdna3::attn::decode(...): the two
// attention modes of LLM inference as a single, composable entry point. Both
// are built FROM the Stage-A primitives (wave::dpp::*, cdna3::mfma::*) — this
// module IS a composition of cdna3-prims, not a separate codebase.
//
// ## Usage
//
//   #include <cdna3/attention.hpp>
//
//   // Prefill (B×H×T×D tensors in fp16 on-device):
//   cdna3::attn::PrefillCfg pcfg{D, seqQ, seqK, nQH, nKVH, batch, causal};
//   hipError_t e = cdna3::attn::prefill(dQ, dK, dV, dO, pcfg);
//
//   // Decode (1×H×D query, H_kv×N×D cache):
//   cdna3::attn::DecodeCfg dcfg{D, N, nQH, nKVH};
//   hipError_t e = cdna3::attn::decode(dq, dK, dV, do_, dcfg);
//
// ## Kernel dispatch (D → W selected for LDS budget on MI300X, limit 64 KB)
//
//   Prefill D=128 → W=1 (LDS ≈ 51 KB); D=64 → W=2 (LDS ≈ 41 KB)
//   Decode  D=128 → W=4; D=64 → W=4  (memory-bound; decode LDS is tiny)
//
// ## Status (Stage B gates)
//
//   PENDING VM validation: H-MW1/2/3 (prefill), H-DEC1/2/3 (decode).
//   The kernels are cold-correct (CPU-verified in the standalone .hip drivers)
//   but hardware validation on 1×MI300X is the exit criterion for Stage B.
//
// ## Kernel sources
//
//   detail/fa_prefill_kernel.hpp  — fa_multiwave_kernel<D,W>  (header-only)
//   detail/fa_decode_kernel.hpp   — fa_decode_kernel<D,W>     (header-only)
//
//   The standalone drivers (fa_multiwave.hip / fa_decode.hip in the research
//   area) include these same headers and add the benchmark harness + main().
//   One source of truth per kernel symbol.
//
// Include separately from cdna3.hpp (which gives the primitives only) — this
// header pulls in the kernel __global__ templates.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 -I<cdna3-prims>/include <your>.hip
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "cdna3.hpp"                        // Stage-A primitives
#include "detail/fa_prefill_kernel.hpp"     // fa_multiwave_kernel<D,W>
#include "detail/fa_decode_kernel.hpp"      // fa_decode_kernel<D,W>
#include "detail/fa_decode_dme_kernel.hpp"  // fa_decode_dme_kernel<D,W>

namespace cdna3 {
namespace attn {

// ---------------------------------------------------------------------------
// Configuration structs.
// ---------------------------------------------------------------------------

struct PrefillCfg {
    int D;          // head dimension (64 or 128)
    int seqQ;       // query sequence length (must be multiple of W*16, typically W*32+)
    int seqK;       // key/value sequence length
    int nQHeads;    // number of query heads
    int nKVHeads;   // number of KV heads (nQH for MHA; nQH/G for GQA)
    int batch;      // batch size
    int causal;     // 1 = causal mask, 0 = full attention
};

struct DecodeCfg {
    int D;          // head dimension (64 or 128)
    int N;          // KV-cache length (number of cached keys)
    int nQHeads;    // number of query heads
    int nKVHeads;   // number of KV heads
};

// ---------------------------------------------------------------------------
// cdna3::attn::prefill — multi-wave prefill attention.
//
// All pointers are fp16 on-device:
//   Q[batch × nQH × seqQ × D], K/V[batch × nKVH × seqK × D], O same as Q.
// ---------------------------------------------------------------------------
inline hipError_t prefill(const __half* Q, const __half* K, const __half* V,
                           __half* O, const PrefillCfg& cfg,
                           hipStream_t stream = nullptr)
{
    PrefillKCfg kc{ cfg.D, cfg.seqQ, cfg.seqK, cfg.nQHeads, cfg.nKVHeads,
                    cfg.batch, cfg.causal };

    if (cfg.D == 128) {
        constexpr int W = 1;
        int sblocks = cfg.seqQ / (W * _Br);
        dim3 grid(cfg.batch * cfg.nQHeads * sblocks), block(W * 64);
        hipLaunchKernelGGL((fa_multiwave_kernel<128, W>), grid, block, 0, stream,
                           Q, K, V, O, kc);
    } else if (cfg.D == 64) {
        constexpr int W = 2;
        int sblocks = cfg.seqQ / (W * _Br);
        dim3 grid(cfg.batch * cfg.nQHeads * sblocks), block(W * 64);
        hipLaunchKernelGGL((fa_multiwave_kernel<64, W>), grid, block, 0, stream,
                           Q, K, V, O, kc);
    } else {
        return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

// ---------------------------------------------------------------------------
// cdna3::attn::decode — split-KV decode attention (1 query × N cached keys).
//
// Pointers (all fp16, device):
//   q[nQH × D], K[nKVH × N × D], V[nKVH × N × D], o[nQH × D]
// ---------------------------------------------------------------------------
inline hipError_t decode(const __half* q, const __half* K, const __half* V,
                          __half* o, const DecodeCfg& cfg,
                          hipStream_t stream = nullptr)
{
    DecodeKCfg kc{ cfg.D, cfg.N, cfg.nQHeads, cfg.nKVHeads };

    if (cfg.D == 128) {
        constexpr int W = 4;
        dim3 grid(cfg.nQHeads), block(W * 64);
        hipLaunchKernelGGL((fa_decode_kernel<128, W>), grid, block, 0, stream,
                           q, K, V, o, kc);
    } else if (cfg.D == 64) {
        constexpr int W = 4;
        dim3 grid(cfg.nQHeads), block(W * 64);
        hipLaunchKernelGGL((fa_decode_kernel<64, W>), grid, block, 0, stream,
                           q, K, V, o, kc);
    } else {
        return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

// Expose DecodeDmeCfg (defined globally in fa_decode_dme_kernel.hpp) inside
// the cdna3::attn namespace so callers can write cdna3::attn::DecodeDmeCfg.
using ::DecodeDmeCfg;

// ---------------------------------------------------------------------------
// cdna3::attn::decode_dme — decode attention reading directly from the
// strided KV-cache (eliminates pack_kv_kernel). Uses DME async prefetch.
//
// K/V layout: [nKVHeads, max_seq, D] head-major, stride_kv = max_seq.
// Caller passes K_layer(li, 0) — pointer to [kvhead=0, seq=0, d=0].
//
// Requires CDNA3 (gfx942). For non-DME hardware use decode() with pack_kv.
// ---------------------------------------------------------------------------
inline hipError_t decode_dme(const __half* q, const __half* K, const __half* V,
                              __half* o, const DecodeDmeCfg& cfg,
                              hipStream_t stream = nullptr)
{
    // LDS budget (64 KB limit):
    //   D=64,  kBc=64, W=4: 2×2×64×64×2 + W×D×4 = 32768+1024 = 33 KB  OK
    //   D=128, kBc=32, W=4: 2×2×32×128×2 + W×D×4 = 32768+2048 = 34 KB  OK
    if (cfg.D == 128) {
        constexpr int W = 4, kBc = 32;
        dim3 grid(cfg.nQHeads), block(W * 64);
        hipLaunchKernelGGL((fa_decode_dme_kernel<128, W, kBc>), grid, block, 0, stream,
                           q, K, V, o, cfg);
    } else if (cfg.D == 64) {
        constexpr int W = 4, kBc = 64;
        dim3 grid(cfg.nQHeads), block(W * 64);
        hipLaunchKernelGGL((fa_decode_dme_kernel<64, W, kBc>), grid, block, 0, stream,
                           q, K, V, o, cfg);
    } else {
        return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

} // namespace attn
} // namespace cdna3

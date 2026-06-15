// ---------------------------------------------------------------------------
// examples/fused_softmax.hpp — reusable row-softmax building block on the DPP path.
//
// Extracted from the validated LayerNorm/Softmax kernels (paper §5.6) and the FA
// softmax (§5.8): a wave-wide row reduction for max and sum using the zero-LDS DPP
// primitives, composed into an online-softmax-friendly building block.
//
// This is a COMPOSITION EXAMPLE for cdna3-prims, not a new finding — it shows how
// wave::dpp::* composes into a real kernel pattern. The DPP advantage here is
// reduction-domain-specific (paper §5.6): strong for short rows / register-bound
// kernels; collapses to 1–3% when the kernel is memory-bound (wide rows). Use
// accordingly.
//
//   #include <cdna3/cdna3.hpp>
//   #include <cdna3/../examples/fused_softmax.hpp>   // or copy into your tree
// ---------------------------------------------------------------------------
#pragma once
#include "../include/cdna3/cdna3.hpp"

namespace cdna3 {
namespace ex {

// One-shot row softmax for a row whose Bc==wavesize scores are held one-per-lane.
// `score` is this lane's score for the row; returns this lane's softmax probability.
// Uses two zero-LDS DPP reductions (max then sum). Numerically stable (subtracts max).
__device__ __forceinline__ float row_softmax_lane(float score){
    float m = wave::dpp::reduce_max_bcast(score);   // row max, broadcast to all lanes
    float e = __expf(score - m);
    float s = wave::dpp::reduce_sum_bcast(e);        // row sum, broadcast
    return (s > 0.f) ? (e / s) : 0.f;
}

// Online-softmax update step (FlashAttention-style), for streaming Bc-wide tiles.
// Maintains running (m_run, l_run); returns the rescale factor to apply to the
// accumulator O and updates m_run/l_run in place. `score` is this lane's score.
//   resc = exp(m_old - m_new);  O *= resc;  O += p · V;   (caller does the O update)
__device__ __forceinline__ float online_softmax_step(float score, float& m_run,
                                                     float& l_run, float& p_out){
    float rmax = wave::dpp::reduce_max_bcast(score);
    float mnew = fmaxf(m_run, rmax);
    float p    = __expf(score - mnew);
    float rsum = wave::dpp::reduce_sum_bcast(p);
    float resc = __expf(m_run - mnew);
    m_run = mnew;
    l_run = resc * l_run + rsum;
    p_out = p;
    return resc;
}

} // namespace ex
} // namespace cdna3

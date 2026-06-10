# Design Note — Multi-Wave Flash Attention (closing the occupancy gap)

**Cold design.** 2-C measured our kernel 8.5–15× slower than production fused SDPA on
MI300X, and localized the cause precisely: the **single-wave-per-block** design idles
48 of 64 lanes during softmax (only lanes 0–15 = the Br=16 query rows are active). The
paper (§5.9) states the gap is an *occupancy* gap, not a *primitive* gap, and that a
multi-wave kernel is the indicated path. This note designs that kernel before writing
HIP, so the VM session is implement-and-validate, not design.

The DPP/DME primitive findings are orthogonal and **carry over unchanged** — this is a
scheduling/occupancy redesign, not a primitive change.

---

## 1. Where the current kernel wastes the machine

`fa_robust_kernel<D>` (the validated 1-wave/block design):

- **grid.x = batch · nQHeads · (seqQ / Br)**, block = 64 lanes (one wave).
- One wave owns one query-block of **Br=16 rows**, sweeps all KV tiles sequentially.
- Q·Kᵀ and P·V use all 64 lanes via MFMA 16×16×16 (good).
- **Softmax (the online rescale) runs on lanes 0–15 only** (`if (L<Br)`), one lane per
  query row — **48 lanes idle**, with a `__syncthreads()` before and after.
- One CU has 4 SIMDs; one wave/block uses **one SIMD**, leaving occupancy on the table
  at the block level too (though the grid is large enough to fill CUs across blocks).

Two distinct inefficiencies: (a) intra-wave — 48/64 lanes idle in softmax; (b) the
softmax is serialized against the matmuls by two barriers. A production kernel overlaps
and parallelizes both.

## 2. The three parallelization axes (and which to pick)

| Axis | Idea | Helps the softmax-idle gap? | Cost | Best regime |
|------|------|------------------------------|------|-------------|
| **A — wider query block** | W waves/block, each owns its own Br=16 rows → block covers Br_tot = W·16 query rows; all W waves share the SAME K/V LDS tiles (loaded once, reused W×) | **YES** — each wave runs its own softmax on its own 16 rows → all 64 lanes of every wave active; no idle | K/V LDS reused across W waves (amortized load); more S/P/O LDS (W×) | **Prefill** (many query rows) — our case |
| **B — split-KV** | W waves/block split the KV tiles; each computes partial (m,l,O), then an inter-wave LDS reduction combines them (FlashAttention split-K) | Indirectly — keeps lanes busy on KV work, but adds an inter-wave combine | Inter-wave reduction of (m,l,O) via LDS; sync overhead | **Decode** (1 query row, long KV) — not our case |
| **C — more heads/block** | One wave/head, several heads per block | No — each head still 1-wave with idle softmax | Amortizes launch only | Tiny seq, launch-bound |

**Decision: Axis A (wider query block).** It directly removes the idle-lane gap that
2-C measured, it matches the prefill regime our whole 2-A/2-B/2-C arc lives in, and it
reuses the K/V tile loads across the waves (the K/V LDS load is the shared cost — doing
it once for W query-blocks is a real bandwidth win on top of the occupancy win). Axis B
is the right tool for decode (future work, ties into the KV-cache item), and Axis C
doesn't address the actual bottleneck.

## 3. Axis-A design (concrete)

```
block = W waves × 64 lanes   (W ∈ {2,4}; W=4 → 256 threads, Br_tot=64)
grid.x = batch · nQHeads · (seqQ / Br_tot)

Each wave w ∈ [0,W):
  owns query rows [blockQ + w·16, blockQ + w·16 + 15]   (its own Q sub-block)
  has its own Q_lds[16·D], S_lds[16·Bc], P_lds[16·Bc], O_lds[16·D], m_run/l_run
Shared across the block:
  K_lds[Bc·D], V_lds[Bc·D]   — loaded ONCE per KV tile by all W·64 lanes cooperatively
```

Per KV tile (the inner loop), each wave independently:
1. Q·Kᵀ via MFMA on its own 16 rows (all 64 lanes busy) — unchanged from fa_robust.
2. **Softmax on its own 16 rows using all 64 lanes** — this is the change. Instead of
   `if (L<Br)` (16 lanes), distribute the 16 rows across the 64 lanes so each lane owns
   ≥1 row's reduction, or better: use the verified DPP wave reduction
   (`wave::dpp::reduce_max`/`reduce_sum`) so the row-max and row-sum are computed across
   the wave with **zero idle lanes and zero LDS** — exactly the primitive this whole
   project built. (The current kernel's scalar per-row softmax was the shortcut; the
   DPP path is the natural fit and reuses our header.)
3. P·V via MFMA — unchanged.

K/V tiles are loaded once into the shared LDS by all W·64 lanes (W× fewer HBM reads per
tile relative to W independent 1-wave blocks covering the same rows).

**LDS budget check (W=4, D=128):** K_lds+V_lds = 2·64·128·2 = 32 KB (shared). Per-wave
S+P+O = 16·64·4 + 16·64·2 + 16·128·4 = 4+2+8 = 14 KB; ×4 waves = 56 KB. Total 88 KB >
64 KB/CU → **W=4 at D=128 overflows LDS.** Options: W=2 at D=128 (44 KB, fits), or W=4
at D=64 (K/V 16 KB + per-wave 12 KB×4 = 64 KB, exactly at budget — tight). **Plan:
parameterize W and D; sweep {W=2,D=128}, {W=4,D=64}, {W=4,D=128-with-S/P-in-registers}.**
Keeping S/P in registers instead of LDS (they're transient per-tile) is the path to W=4
at D=128 and is itself a worthwhile variant to measure.

## 4. What stays identical (de-risks the rewrite)

- The MFMA Q·Kᵀ and P·V inner loops + the verified `v_mfma_f32_16x16x16f16` lane mapping.
- The causal tile-skip + element-mask logic.
- The GQA kvhead mapping.
- The online-softmax math (m_run/l_run rescale).
- Correctness reference (`attn_ref`) and the seqLen sweep harness — reused verbatim, so
  the multi-wave kernel is checked against the same fp32 reference that validated
  fa_robust, and timed on the same sweep for an apples-to-apples speedup number.

Only the **block shape + the softmax lane distribution** change. This is the minimal
edit that removes the idle-lane gap, which keeps the rewrite low-risk (the parts that
were hard to get right — MFMA mapping, causal, GQA — are untouched).

## 5. Pre-registered hypotheses (commit before running)

- **H-MW1 (occupancy):** distributing softmax across all 64 lanes (DPP) + W-wide query
  blocks reduces single-block latency vs fa_robust at the same total work, by keeping
  lanes busy through the softmax phase. Accept if multi-wave is faster at matched
  (seqLen, D) on ≥3 of the 4 swept seqLens.
- **H-MW2 (K/V reuse):** loading K/V once per tile for W query sub-blocks reduces
  VMEM read instructions vs W independent 1-wave blocks (rocprofv3 SQ_INSTS_VMEM_RD).
  Accept if VMEM_RD drops ≥ (1 − 1/W) of the K/V-load share.
- **H-MW3 (gap closure):** multi-wave narrows the 2-C gap vs SDPA. This is reported as a
  measurement, not a pass/fail — we do NOT expect to match SDPA (it has fp8, full
  pipelining); the claim is *movement in the right direction*, with the residual gap
  attributed to remaining scheduling differences. Honest framing, like 2-C.

Negative result is publishable (as the MFMA-at-D64 and Welford refutations were): if
multi-wave does NOT help, that localizes the gap further (e.g. to instruction
scheduling / wave specialization à la HipKittens), which is itself a finding.

## 6. Scope (what this is and is not)

- ✅ Axis-A multi-wave prefill kernel; DPP softmax across the full wave; K/V LDS reuse.
- ✅ Validated against the same fp32 reference + seqLen sweep as fa_robust.
- ❌ NOT wave specialization (producer/consumer, à la HipKittens) — that's a heavier
  redesign; if H-MW1 underperforms, it becomes the next hypothesis.
- ❌ NOT decode/split-KV (Axis B) — separate item, ties to the KV-cache work.
- ❌ NOT expected to beat production SDPA — the goal is to close the occupancy gap and
  measure the remainder.

## 7. Cold deliverables (this note + next)

1. This design note (committed = the pre-registration timestamp).
2. `fa_multiwave.hip` — the Axis-A kernel, parameterized by W and D, reusing fa_robust's
   MFMA/causal/GQA/reference/sweep scaffolding. Written cold; compiled+validated on the
   next 1×MI300X session (no multi-GPU needed — this is a single-GPU occupancy kernel).
3. The VM guion: build → correctness vs fp32 ref → seqLen sweep → rocprofv3 VMEM/LDS →
   compare vs fa_robust and vs SDPA (reuse compare_vs_production_fa.py) → capture.

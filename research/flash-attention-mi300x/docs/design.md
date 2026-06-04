# Flash Attention on MI300X — Design (cold draft, pre-implementation)

> **Status:** architecture design only. No kernel written yet — this is the plan
> that connects the validated pieces (DPP reductions) with the in-progress piece
> (DME prefetch) and the MFMA matrix cores. Written cold while GPU is unavailable.

## Why this is the integration point of the project

Flash Attention is where the "kernel as backbone" strategy converges. It needs
all three building blocks the project is producing:

| Building block | Status | Role in FA |
| --- | --- | --- |
| Wave reductions (DPP) | ✅ validated (Run 2) | the online-softmax row max + row sum |
| DME async copy | 🟡 probe written | prefetch next K/V tile while computing current |
| MFMA matrix cores | documented | the Q·Kᵀ and P·V matmuls |

If the DPP reductions speed up the softmax step *and* the DME prefetch hides the
HBM latency of K/V loads, the result is an FA kernel that is both compute- and
bandwidth-efficient on CDNA3 — the thing vLLM/PyTorch actually run.

## Recap: the FlashAttention-2 algorithm (the baseline we build on)

For a query block Q (Br × d) attending over all K,V (each N × d):

```
m = -inf  (running row max, per query row)
l = 0     (running row sum of exp)
O = 0     (running output accumulator, Br × d)

for each key/value block (Kj, Vj) of size Bc × d:
    S   = Q · Kjᵀ                     # Br × Bc      (MFMA)
    m_new = max(m, rowmax(S))         # ROW REDUCTION (max)   ← DPP
    P   = exp(S - m_new)              # Br × Bc
    l   = exp(m - m_new) * l + rowsum(P)   # ROW REDUCTION (sum) ← DPP
    O   = exp(m - m_new) * O + P · Vj  # Br × d       (MFMA)
    m   = m_new
O = O / l                             # final normalize
```

The two `rowmax` / `rowsum` are wave-level reductions over the Bc dimension —
**exactly our `wave::dpp::reduce_max` / `reduce_sum`** when a row maps to a wave.

## Where CDNA3 hardware maps

```
        HBM (192 GB, 5.3 TB/s)
          │  Q, K, V tiles
          │
   ┌──────┴───────┐
   │     DME      │  ← async prefetch K_{j+1}, V_{j+1}  (this project's DME API)
   └──────┬───────┘
          ▼
        LDS (64 KB/CU)  ← K_j, V_j tiles staged here
          │
   ┌──────┴───────┐
   │  MFMA cores  │  ← S = Q·Kjᵀ  and  O += P·Vj
   └──────┬───────┘
          │  S in VGPRs
          ▼
   ┌──────────────┐
   │ DPP reduce   │  ← rowmax(S), rowsum(P)  (0 LDS — frees LDS for tiles)
   └──────────────┘
```

The key synergy: **DPP reductions use zero LDS** (Run 2 finding). In Flash
Attention, LDS is the scarce resource — it holds the K/V tiles. A softmax
reduction that spills to LDS (`ds_bpermute`) competes with the tiles for LDS
budget, forcing smaller tiles. A DPP reduction leaves the entire LDS for tiles →
larger Bc → fewer iterations → better MFMA utilization. **The zero-LDS property
is not just faster in isolation; it relaxes the LDS pressure that limits FA tile
size.** This is the argument that makes the wave-primitives result matter for FA.

## The DME pipelining (FA3-style, the novel part on CDNA3)

FlashAttention-3 (Hopper) overlaps the next tile's load with the current tile's
compute using `cp.async` + warp specialization. The CDNA3 equivalent:

```
            iteration j            iteration j+1
DME:    [prefetch K_{j+1},V_{j+1}] ───────────────►
MFMA:   [S=Q·Kjᵀ] [O+=P·Vj]        [S=Q·K_{j+1}ᵀ] ...
DPP:        [rowmax][rowsum]            [rowmax]...
                                   ▲
                            dme::wait() here — tile j+1 ready
```

Double-buffer the LDS: while MFMA computes on buffer A (tile j), DME fills
buffer B (tile j+1). Swap each iteration. This is exactly what `dme_copy.hpp`'s
async path must enable — hence DME is the prerequisite.

## Build order (each step is independently testable)

1. **Naive FA** — FA2 with portable reductions + synchronous K/V loads. Baseline.
2. **+ DPP reductions** — swap rowmax/rowsum to `wave::dpp::*`. Measure: does
   freeing LDS let us grow Bc? (the relaxed-LDS-pressure hypothesis)
3. **+ DME prefetch** — double-buffer K/V via `dme::async`. Measure overlap.
4. **+ MFMA tuning** — tile-size selection (ties into mlir-mfma-tiling area).

Each step compared against the ROCm baseline (the existing CK / Triton FA).

## Open risks (honest)

- **LDS double-buffering may not fit.** Two K/V tiles + Q in 64 KB is tight for
  large d. May need to stage only K via DME and keep V in registers, or smaller Bc.
- **DPP reduction is over Bc, but Bc spans multiple waves** for realistic tiles.
  Our wave64 reduction handles one wave; a Bc > 64 needs a block-level reduction
  (two-phase: DPP wave-reduce → LDS for inter-wave → DPP again). The inter-wave
  step reintroduces *some* LDS — need to measure if it stays a net win.
- **DME may not beat a well-tuned `buffer_load` for the FA access pattern.** The
  DME's advantage is async overlap, not raw bandwidth. If the FA loop is already
  compute-bound, the prefetch buys little. Step 3 measures this honestly.

## Dependencies

- `../../wave-primitives/include/wave_primitives/wave_reduce_dpp.hpp` ✅
- `../dme-abstraction/include/dme/dme_copy.hpp` 🟡 (needs Run 4 validation)
- MFMA via `__builtin_amdgcn_mfma_*` or rocWMMA (to be decided)

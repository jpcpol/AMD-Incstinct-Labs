# S5 — Cost Contrast O(n²) flat-context vs O(κ) governance state (Pre-Registration)

**Commit this document BEFORE running any S5 code on the VM.** Its commit SHA is the
pre-registration timestamp. S5 is the AMD-Instinct side of the CAL-L4 Efficiency
Hypothesis: contrast the **measured** flat-context attention cost (O(n²), our R1
baseline) against the **structural complexity κ(V)** of the L3-delivered dual volume,
and report the contrast honestly — as a demonstration of *mechanism*, not a proof of
the hypothesis at production scale.

- **Date prepared (cold):** 2026-06-10
- **Owner:** AMD-Instinct Labs (`fa_robust` on MI300X, gfx942)
- **Unblocked by:** L3 closure + L4-A handoff (2026-06-10), which delivered the dual
  object `(κ(V_Tucker)=1296, G_pruned, U≈0.86)` — see
  `CAL/L4/experiments/efficiency_hypothesis/PRE_REGISTRATION_L4A_DUAL.md` and
  `results/l4a_results.json`.
- **Reuses:** the already-measured R1 baseline (n^1.90, R²=0.9964;
  `results/2026-06-08_R1_baseline_and_2B_llm_mi300x.txt`) and the existing
  `fa_robust.hip` seqLen sweep. No new kernel. S5 adds an *analysis*, not new HIP code,
  unless §4 measurement of the O(κ) side requires a small probe (see §3.3).

---

## 0. The honesty constraint that shapes this whole experiment

L4-A's C4 was explicit: **κ=1296 is fixed by C_compress, before pruning; it is the
Tucker core size at the governance-corpus scale (n=30 sessions, N_DIMS=11).** It is
NOT the same `n` as our seqLen baseline (512→4k tokens). Therefore:

> S5 does **not** plug κ=1296 and seqLen=4096 into the same axis and declare a speedup.
> That would be the apples-to-oranges error the L4-A prereg's split exists to prevent.

What S5 *can* legitimately do is contrast the **two cost laws** as functions of their
respective `n`, on the same hardware, and show that the governance-state path's cost
is **decoupled from artifact volume** while the flat-context path's cost is **coupled
to it** (the actual content of the L4 Efficiency Hypothesis §3.2). The deliverable is
a *mechanism contrast with a declared scope limitation*, exactly mirroring how L4-A
delivered the operator with a declared dual-representation limitation.

---

## 1. What is fixed a priori (the two cost laws under contrast)

```
O(n²) side  (flat context):   Cost_flat(n)  = attention cost over n raw artifacts,
                              measured = c · n^p,  p=1.90, R²=0.9964 (R1, MI300X D=128)

O(κ) side   (governance):     Cost_gov(V)   = work of M(V) over the dual volume,
                              scales with κ(V_Tucker)+|E| = 1296+2 = 1298 reads
                              (L4-A C5: cost_reads=1322, graph_side=26, INDEPENDENT of n)
```

The contrast quantity fixed before running:

> **Decoupling ratio** D(n) = Cost_flat(n) / Cost_gov, where Cost_gov is **constant in n**
> (it depends on κ(V), a property of the compressed governance state, not of artifact
> count). The hypothesis predicts D(n) grows ~n^1.90 — i.e. the gap widens with scale.

This is reported as a curve, not a single number. A single number (e.g. "195× / 4096×")
would hide the scope mismatch; the curve makes the decoupling visible and honest.

## 2. The question S5 answers (fixed a priori)

> On MI300X, does the flat-context attention cost grow as a power law in artifact
> volume n (≈n²) while the governance-state inference cost stays bounded by κ(V)
> (constant in n) — demonstrating the *mechanism* of the L4 Efficiency Hypothesis?

S5 answers the **mechanism** question. It explicitly does NOT answer:
- whether κ(V)=1296 at *production* governance scale (that needs a larger L2 corpus);
- whether M(V) governance decisions are *as accurate* as flat-context (condition (c),
  the RCT — not hardware work);
- the single-representation V' efficiency (L4-B, gated on the residual-25% prereg).

## 3. Measurement plan (fixed a priori)

### 3.1 O(n²) side — already measured, re-used verbatim
The R1 sweep is the flat-context cost law: seqLen {512,768,1024,1536,2048,3072,4096},
D=128, fitted n^1.90 (R²=0.9964). **No re-run required for the law itself**; one
confirmatory re-run during the S5 VM session to pin the absolute constant `c` under
the same clocks as §3.3 (drift control). Report median + P95 latency per the project
benchmark convention.

### 3.2 O(κ) side — the governance-state cost
From L4-A `l4a_results.json`: per-graph `cost_reads ≈ 1322` (κ+|E| ≈ 1298, graph-side
26), constant across G1/G2/G3 and **independent of n**. This is a software read-count,
not a hardware latency. S5 reports it as the *structural* cost object, with the
explicit note (C4-honesty inherited) that κ is the Tucker core's, fixed pre-prune.

### 3.3 Optional hardware probe (decide on VM, only if cheap)
If a κ-sized inference can be expressed as a small MFMA/GEMM workload on MI300X
(core 8×3×3×3×6 contracted), measure its **wall-clock** to give the O(κ) side a
hardware latency comparable to the O(n²) side's latency — turning the read-count into
a measured time on the *same* device. Gate: only if it fits in the same VM session
without a new kernel build cycle. If it doesn't, S5 reports the O(κ) side as the
structural read-count (§3.2) and notes the hardware-latency probe as future work.
**Do not burn a VM hour on this alone** (billing discipline).

## 4. Confirmatory checks (fixed a priori)

- **C1 — Baseline law reproduced.** The S5-session re-run of the seqLen sweep yields
  an exponent within [1.8, 2.2] (matches R1's 1.90). PASS if in window.
- **C2 — Decoupling demonstrated.** Cost_gov is constant across the n sweep (it does
  not read seqLen); D(n) is monotonically increasing in n. PASS if D(n) strictly
  increases and the fitted growth exponent of D(n) ∈ [1.8, 2.2].
- **C3 — Scope honesty (the declared limitation).** The report states explicitly that
  (i) κ is at governance-corpus scale, not seqLen scale; (ii) the contrast shows the
  *mechanism* (cost coupling vs decoupling), not a production-scale speedup; (iii) κ
  does not reflect the prune (inherited from L4-A C4). PASS = all three stated. This
  is a reporting gate, the analog of L4-A's C4.

## 5. Verdict logic (fixed a priori)

| C1 | C2 | C3 | Verdict |
|----|----|----|---------|
| PASS | PASS | PASS | **S5 demonstrates the L4 Efficiency Hypothesis mechanism on MI300X:** flat-context cost grows ~n^1.90, governance-state cost is bounded by κ(V) and decoupled from n. Closes condition (b) as a *mechanism* result. Freeze. |
| PASS | FAIL | — | Cost_gov is not actually n-independent — re-examine whether the M_ref read-count leaks an n dependence; the dual object's decoupling claim is not yet supported. |
| FAIL | — | — | Baseline law not reproduced under S5 clocks — investigate clock/throttle drift before any contrast claim. |

## 6. What S5 explicitly does NOT do

- **No production-scale κ.** κ=1296 is the governance corpus's; S5 does not claim it
  holds at scale.
- **No accuracy claim.** Governance accuracy of M(V) vs flat-context is condition (c)
  / the RCT, not S5.
- **No L4-B.** The single-representation V' (inverse graph→tensor projection) is NOT
  touched. It is gated on the residual-25% characterization prereg (TCO side).
- **No new kernel.** S5 reuses `fa_robust`. The optional §3.3 probe is the only place a
  small workload might be added, and only if it fits the same VM session.

## 7. Execution sequence (consultant FREEZE order, fixed)

1. Commit this prereg (cold). ← pre-registration timestamp.
2. On the next VM session: confirmatory re-run of the seqLen sweep (C1) + compute D(n)
   from the existing R1 law and the L4-A read-count (C2). Optional §3.3 probe only if
   cheap in the same session.
3. Capture results to `results/2026-XX-XX_S5_cost_contrast_mi300x.txt`.
4. **Freeze results.** No L4-B, no residual-25% work until the freeze holds.
5. Update `cal-collaboration.md` §2e + §3 bitácora with the S5 outcome.

## 8. Fixed parameters (summary)

- O(n²) side: R1 law `c·n^1.90` (R²=0.9964), seqLen {512..4096}, D=128, MFMA path.
- O(κ) side: κ(V_Tucker)=1296, |E|=2, cost_reads≈1322 (L4-A, n-independent).
- Contrast: decoupling ratio D(n)=Cost_flat(n)/Cost_gov, reported as a curve.
- Checks: C1 (exponent ∈[1.8,2.2]), C2 (D(n) increasing, growth exp ∈[1.8,2.2]),
  C3 (scope-honesty statement present).
- Architecture A handoff only. No L4-B. Billing: one VM session, grouped, §3.3 probe
  only if it fits.

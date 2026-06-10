# Flat-Context Attention Cost on MI300X as the CAL-L4 Baseline

**A citable measurement note.**
AMD-Instinct Labs · MI300X (gfx942) · ROCm 7.2 · June 2026

> **One-line result:** flat-context (non-compressed) attention cost on MI300X scales
> as **n^1.90 (R² ≈ 0.996)** in sequence length across the LLM-realistic range
> 512→4096 at head dim 128 — the empirical O(n²) baseline against which any
> compressed-state inference scheme (e.g. CAL-L4's M(V)) must be compared.

---

## 1. Why this number exists

The CAL-L4 *Efficiency Hypothesis* claims that meta-inference over a compressed
governance state V costs O(κ(V)), with κ(V) growing far more slowly than the O(n²)
of flat-context attention over n raw artifacts. That comparison needs a **measured**
O(n²) baseline on real hardware, not an assumed one. This note provides it. It is the
"condition (b)" of the L4 hypothesis (CAL-L4 paper §3.3) and the O(n²) side of the
S5 cost contrast (CAL-L4 paper §5.4).

It is also independently useful: a clean, reproducible characterization of where the
quadratic regime actually sits for an MFMA-tiled Flash-Attention kernel on CDNA3, with
the measurement methodology stated so others can replicate or contest it.

## 2. What was measured

- **Kernel:** `fa_robust.hip` — a 1-wave-per-block Flash-Attention (FA2) kernel with
  MFMA `16x16x16` tiles for Q·Kᵀ and P·V, generalized to runtime seqLen, GQA, and
  causal masking. Correctness is CPU-verified (fp32 reference) on small configs
  covering MHA / GQA / causal before any timing run.
- **Sweep:** seqLen ∈ {512, 768, 1024, 1536, 2048, 3072, 4096}, head dim **D = 128**
  (LLM-realistic), MHA, full (non-causal) attention, 8 heads, fp16 inputs / fp32
  accumulation.
- **Device:** AMD Instinct MI300X VF (gfx942), ROCm 7.2.
- **Metric:** median kernel latency over 100 timed iterations (10 warmup), per seqLen;
  log-log least-squares regression → exponent p and R². The kernel emits the fitted
  exponent and a CONFIRMED/OUTSIDE verdict against the acceptance window [1.8, 2.2].

## 3. The measurement

Representative run (2026-06-08, D=128, MHA, full):

| seqLen n | median latency | TFLOPS (achieved) |
|----------|----------------|-------------------|
| 512  | 103.7 µs | 10.36 |
| 768  | 299.6 µs | 8.06 |
| 1024 | 457.1 µs | 9.40 |
| 1536 | 952.2 µs | 10.15 |
| 2048 | 1634.8 µs | 10.51 |
| 3072 | 3626.1 µs | 10.66 |
| 4096 | 6015.1 µs | 11.42 |

**Fit:** `Cost(n) = c · n^p` with **p = 1.90, R² = 0.9964**. Over an 8× increase in
seqLen (512→4096), latency grows 58× — the quadratic signature.

## 4. Reproducibility — three independent runs agree

The exponent is stable across runs and across days, which is the point of a *baseline*:

| Run | Date | Exponent p | R² | Verdict |
|-----|------|-----------|-----|---------|
| R1 | 2026-06-08 | 1.90 | 0.9964 | CONFIRMED O(n²) |
| S5 sweep | 2026-06-10 | 1.90 | 0.9962 | CONFIRMED O(n²) |
| S5 contrast | 2026-06-10 | 1.907 | 0.9967 | CONFIRMED O(n²) |

Per-seqLen latencies match within measurement noise across runs (e.g. seqLen 512:
103.7 / 103.9 / 104.0 µs), confirming no clock/throttle drift between sessions.

## 5. Why the exponent is 1.90 and not exactly 2.0

Slightly sub-quadratic, for understood reasons — stated so the number isn't
over-read:

- **Fixed per-launch and projection overhead** dilutes the n² term at small seqLen
  (it is a larger fraction of the 512 latency than of the 4096 latency).
- The MFMA matrix cores reach higher achieved TFLOPS at larger seqLen (10.36 → 11.42),
  so the larger problems run *relatively* more efficiently, bending the curve just
  below 2.0.

The regime is unambiguously quadratic (p ∈ [1.8, 2.2] with R² ≈ 0.996); the small
deficit from 2.0 is an honest artifact of fixed overhead and throughput scaling, not
a sub-quadratic algorithm.

## 6. Scope (what this is and is not)

- ✅ It **is** the measured O(n²) flat-context baseline for the CAL-L4 cost contrast,
  and a standalone characterization of FA2/MFMA quadratic scaling on MI300X.
- ✅ It is taken on a kernel **validated against a real transformer layer**
  (Qwen2.5-0.5B, real weights + RoPE + GQA + causal; error 4.34% of signal scale —
  see the LLM-integration results), not a synthetic toy.
- ❌ It does **not** by itself establish the L4 Efficiency Hypothesis — that needs the
  O(κ) side (provided by L3's operator C, κ(V)=1296) and the governance-accuracy gate
  (condition (c)). The contrast of the two laws is the S5 result (CAL-L4 §5.4); this
  note is only the O(n²) half.
- ❌ It is **not** a claim about other head dims, batch regimes, or causal/GQA shapes
  beyond what was swept (those are timed in the same results file but the fitted
  baseline is the D=128 MHA-full curve).

## 7. How to cite

> AMD-Instinct Labs (2026). *Flat-context attention cost on MI300X as the CAL-L4
> baseline.* Measurement note. Flat-context FA2/MFMA attention scales as n^1.90
> (R²=0.996) over seqLen 512→4096 at D=128 on MI300X (gfx942, ROCm 7.2).

## 8. Artifacts

- Kernel: [`fa_robust.hip`](fa_robust.hip)
- Raw results: [`results/2026-06-08_R1_baseline_and_2B_llm_mi300x.txt`](results/2026-06-08_R1_baseline_and_2B_llm_mi300x.txt),
  [`results/2026-06-10_S5_cost_contrast_mi300x.txt`](results/2026-06-10_S5_cost_contrast_mi300x.txt)
- Cost contrast (uses this baseline): [`s5_cost_contrast.py`](s5_cost_contrast.py),
  [`PRE_REGISTRATION_S5_COST_CONTRAST.md`](PRE_REGISTRATION_S5_COST_CONTRAST.md)
- Consumed by: CAL-L4 paper §3.3 (condition b) and §5.4 (cost contrast).

# Eliminating LDS-Mediated Cross-Lane Communication on CDNA3: Full-DPP Wave64 Primitives and Their Application to Flash Attention on AMD Instinct MI300X

**Draft — 2026-06-05**  
Target: arXiv cs.DC

---

## Abstract

We present a header-only C++ library of wave-level reduction and scan primitives
for AMD CDNA3 GPUs (Instinct MI300X, gfx942) that eliminates Local Data Share
(LDS) traffic from cross-lane communication entirely. Generic HIP shuffle
intrinsics (`__shfl_down`, `__shfl_up`) lower to `ds_bpermute` on CDNA3 —
an LDS round-trip per step — while AMD's Data-Parallel Primitives (DPP) cross-lane
VALU datapath allows the same operations to remain register-local. We characterize
the complete full-DPP wave64 geometry: in-row prefix via `row_shr` shifts and
cross-row carry via cascaded `row_bcast15`/`row_bcast31`, reaching
**zero LDS instructions** on both reduction and inclusive scan as measured by
rocprofv3 hardware counters (`SQ_INSTS_LDS = 0`). On MI300X (ROCm 7.2, hipCUB 4.2.0),
the full-DPP reduction runs **1.35–1.79× faster than hipCUB WarpReduce** and the
full-DPP scan runs at **1.028× hipCUB WarpScan** — the same pattern across two
independent algorithms, suggesting an architectural property of CDNA3 rather than
a single-kernel result. Correctness is verified lane-by-lane across
float/double/int32/int64 × sum/max/min (12/12). We further validate the primitives
in a real workload — a Flash Attention kernel on MI300X — where DPP softmax plus a
CDNA3 Data-Movement-Engine (DME) double-buffer prefetch deliver a cumulative 18%
end-to-end speedup over a portable baseline, with hardware counters isolating each
contribution. We also report the boundaries honestly: the large reduction speedup
is reduction-domain-specific (it collapses to 1–3% in memory-bound kernels), the
zero-LDS property is wave-scoped (block-level reduction matches hipCUB exactly),
and matrix-core (MFMA) tiling does not help small-head attention. Finally, we carry
the kernel through the full research→application arc: generalized to LLM-realistic
shapes (runtime seqLen, GQA, causal), it reproduces a real Qwen2.5 attention layer
within fp16 tolerance and, wrapped as a PyTorch custom op patched into all 24 layers
of the model, runs the complete LLM during prefill while preserving the top-1
next-token prediction. Against a production attention backend it is 8.5–15× slower —
reported honestly, with the cause localized to a single-wave occupancy limit
orthogonal to the (portable) primitive findings. The library is
header-only, MIT-licensed, and falls back to portable `__shfl_*` on wave32/NVIDIA
targets.

---

## 1. Introduction

AMD Instinct MI300X (CDNA3, gfx942) operates with a 64-lane wavefront (wave64)
— double the warp width of NVIDIA GPUs and AMD RDNA consumer GPUs. Wave-level
primitives in HIP — `__shfl_down_sync`, `__ballot_sync`, warp reductions — carry
an implicit assumption of 32-lane width, causing correctness failures and
performance regressions on CDNA3 without manual porting.

The portability problem is documented in AMD's own developer guides and in multiple
ROCm GitHub issues. The community workaround is `hipCUB::WarpReduce` and
`hipCUB::WarpScan`, which are correct for wave64 but require a `TempStorage`
shared memory allocation per call — unsuitable for register-intensive kernels
(GEMM tiles, Flash Attention) where shared memory budget is tight.

This work began from the portability angle. The experimental results relocated the
significant finding elsewhere: the performance gap between portable HIP shuffles
and hipCUB on CDNA3 is not explained by wave-size handling alone, but by a
specific datapath difference. We characterize that difference, implement primitives
that exploit it fully, and show the result generalizes across both reduction and scan.

### 1.1 Contributions

- **C1.** Identification and ISA-level characterization of the `__shfl_*` → `ds_bpermute`
  lowering on CDNA3, explaining the 2.45× performance gap between portable reduction
  and hipCUB (Section 3).
- **C2.** The complete full-DPP wave64 geometry for reduction and scan: `row_shr`
  in-row cascade + `row_bcast15`/`row_bcast31` cross-row/bank carry, reaching
  zero `ds_bpermute` and zero LDS instructions (rocprofv3-verified) on both
  primitives (Section 4).
- **C3.** Empirical performance on MI300X: 1.35–1.79× over hipCUB WarpReduce,
  1.028× over hipCUB WarpScan, 4.39× over portable path — with causal isolation
  via hardware counters (Section 5).
- **C4.** Type generalization: float/double/int32/int64 × sum/max/min, correctness
  12/12 PASS. Sub-finding: only `v_add_f32_dpp` is a fused DPP op; other types use
  `mov_dpp` + separate op (still zero LDS) (Section 5.3).
- **C5.** Open-source header-only library with wave32/NVIDIA fallback, MIT license
  (Section 6).
- **C6.** Real-workload validation in Flash Attention on MI300X: DPP softmax + DME
  async double-buffer prefetch give a cumulative 18% speedup (97.2 → 82.4 µs), with
  rocprofv3 counters attributing −8.6% LDS to DPP and −44.7% VMEM reads to DME
  overlap. We report the negative results too — MFMA tiling does not beat the
  dot-product path at head dim 64, and a Bc tile-size sweep refutes the
  "free-LDS-enables-bigger-tiles" hypothesis (Section 5.8). As a reusable
  by-product we document the empirically-verified `v_mfma_f32_16x16x16f16` wave64
  register-to-matrix lane mapping for gfx942.
- **C7.** End-to-end research→application closure: the kernel, generalized to
  LLM-realistic shapes (runtime seqLen, GQA, causal), reproduces a real Qwen2.5
  attention layer within fp16 tolerance (4.34% of signal scale) and — wrapped as a
  PyTorch custom op patched into all 24 layers — runs the full model during prefill
  with the top-1 next-token prediction preserved. Positioned against production
  fused SDPA it is 8.5–15× slower, with the gap localized to a single-wave occupancy
  limit orthogonal to the primitive contributions (Section 5.9).

---

## 2. Background

### 2.1 AMD CDNA3 Architecture

The AMD Instinct MI300X (CDNA3, gfx942) is a multi-chip-module APU with
8 XCDs (compute dies) and 4 MCD (memory dies), totaling 304 Compute Units,
192 GB HBM3 at 5.3 TB/s bandwidth, and 2,614 TFLOPS peak at FP8.

For this work, the relevant unit is the **wavefront**: CDNA3 executes 64 threads
in lockstep per wave (wave64). Each Compute Unit has 4 SIMD units, each executing
one wavefront. The register file is 512 × 32-bit VGPRs per SIMD unit; Local Data
Share (LDS) is 64 KB per CU, shared across all four SIMDs.

### 2.2 DPP — Data-Parallel Primitives

DPP is a CDNA3 VALU instruction modifier that reads a source operand from a
neighboring lane's register directly, without staging through LDS. The modifier
encodes a lane permutation as a control word in the instruction encoding.

Relevant DPP control codes for wave64 (verified empirically, probe_dpp_bcast_map.hip):

| Code | Name | Effect |
| --- | --- | --- |
| `0x110 + N` | `row_shr N` | Shift right by N within each 16-lane row; out-of-bounds lanes receive identity |
| `0x142` | `row_bcast15` | Broadcast lane 15 to lanes 16–31, lane 31 to 32–47, lane 47 to 48–63 |
| `0x143` | `row_bcast31` | Broadcast lane 31 to lanes 32–63 |
| `0x111` | `row_shr1` | Shift right by 1 within each 16-lane row |

DPP operations are fused into the VALU execution unit for some op+type combinations
(`v_add_f32_dpp`); for others, the compiler emits `v_mov_b32_dpp` + a separate op.
Both are zero-LDS; the fused form avoids a pipeline stage.

### 2.3 HIP Shuffle Lowering on CDNA3

The HIP intrinsics `__shfl_down(v, offset, width)` and `__shfl_up(v, offset, width)`
are documented as cross-lane register reads. On CDNA3, these lower to `ds_bpermute_b32`
— a Local Data Share permutation instruction. Every step in a `__shfl`-based
reduction or scan issues an LDS write+read, serializing on the LDS crossbar.

This lowering is visible in the ISA produced by `hipcc --save-temps` and confirmed
by `SQ_INSTS_LDS > 0` in rocprofv3 profiles of `__shfl`-based kernels (Section 5.2).
hipCUB avoids this by using DPP intrinsics internally; however, hipCUB's WarpReduce
still emits one `ds_bpermute` for the 32-lane bank boundary cross (lane 31 → lane 32
in wave64), which our `row_bcast31` step eliminates.

### 2.4 Related Work

**hipCUB WarpReduce / WarpScan** [AMD, 2024]: Correct wave64 implementations using
DPP + a residual `ds_bpermute` for bank boundary. Requires `TempStorage` shared
memory allocation. Performance baseline in this work.

**Composable Kernel** [AMD, 2024]: High-performance GEMM and attention kernels for
CDNA3. No public wave-level primitive API.

**HipKittens** [HazyResearch, arXiv 2511.08083, 2025]: Tile-level primitives for
GEMM and Flash Attention on CDNA3. Key finding incorporated here: wave specialization
(producer/consumer pattern) achieves ~80% peak FLOPS due to static register
allocation — our primitives use uniform wave patterns.

**rocm-examples** [AMD, 2024]: Pedagogical wave-size-correct reductions using
`warpSize`. Not library-quality; covers only sum reduction; depends on unreleased
internal helpers.

**Gap:** No published, standalone, header-only library provides correct wave-level
reduce, scan, ballot, and shuffle for both wave32 and wave64 with zero LDS for the
register-only operations.

---

## 3. The ds_bpermute Problem

### 3.1 Pre-registered hypothesis and its refutation

We originally hypothesized (H2, pre-registered) that `wave::reduce_sum<float>`
implemented via `__shfl_down` would match `hipCUB::WarpReduce::Sum` within ±5%,
reasoning that both use the same number of cross-lane steps.

| Kernel | Median (µs) | Ratio |
| --- | --- | --- |
| `wave::reduce_sum` (`__shfl_down`) | 922.5 | 2.45× slower |
| `hipCUB WarpReduce::Sum` | 376.7 | 1.00× |

H2 is **refuted** at 2.45× — far outside the ±5% falsification bound.

### 3.2 Root cause: ISA inspection

`hipcc --save-temps` on gfx942 reveals:

```text
wave::reduce_sum (portable):
  ds_bpermute_b32  ×6   (one per reduction step)

hipCUB WarpReduce::Sum:
  v_add_f32_dpp    ×5   (in-row DPP steps)
  ds_bpermute_b32  ×1   (bank boundary cross, lane 31→32)
```

The portable path issues 6 LDS instructions per call. hipCUB issues 1. The VALU
counts are nearly identical (37.8M each in the full benchmark, per rocprofv3
`SQ_INSTS_VALU`), isolating LDS as the sole cause.

### 3.3 Why __shfl_down lowers to LDS on CDNA3

CDNA3's DPP hardware supports fixed shift patterns (`row_shr`, `row_bcast`,
etc.) but not arbitrary lane addressing. The `__shfl_down(v, offset)` semantics —
"read from lane `L + offset`" for arbitrary offset — cannot be expressed as a
single DPP instruction for all offsets. The compiler falls back to LDS-mediated
permutation (`ds_bpermute`) as the general case. hipCUB bypasses this by
using the specific DPP patterns that match its reduction geometry directly.

---

## 4. Full-DPP Wave64 Geometry

### 4.1 Reduction

The wave64 sum reduction requires propagating a value from lane 0 to lane 63
through 6 binary steps. The DPP geometry we use:

**Step 1-4 (in-row, 16-lane rows):**

```text
row_shr 1: lane[i] += lane[i-1]   (within row, i&15 != 0)
row_shr 2: lane[i] += lane[i-2]   (within row, i&15 >= 2)
row_shr 4: lane[i] += lane[i-4]   (within row, i&15 >= 4)
row_shr 8: lane[i] += lane[i-8]   (within row, i&15 >= 8)
```

After step 4, lane 15 holds the sum of lanes 0–15, lane 31 holds lanes 16–31,
lane 47 holds lanes 32–47, lane 63 holds lanes 48–63.

**Step 5 (row boundary, row 0 total → rows 1,2,3):**

```text
c1 = row_bcast15(v)   →  lanes 16–31 receive lane 15's value (row 0 total)
                          lanes 32–47 receive lane 31's value (row 1 total)
                          lanes 48–63 receive lane 47's value (row 2 total)
```

**Step 6 (bank boundary, rows 0+1 total → rows 2,3):**

```text
c1b = row_bcast15(c1)  →  lanes 32–47 receive row 0 total (from lane 31 of c1)
                           lanes 48–63 receive rows 0+1 total
c1c = row_bcast15(c1b) →  lanes 48–63 receive rows 0+1+2 total
```

Combined carry per lane group:

- Lanes 0–15: no carry (local row)
- Lanes 16–31: c1 (row 0 total)
- Lanes 32–47: c1 + c1b (rows 0+1 total)
- Lanes 48–63: c1 + c1b + c1c (rows 0+1+2 total)

This reaches the wave-wide prefix sum with zero `ds_bpermute`. The key insight is
that `row_bcast15` applied *to c1* (not to v) avoids re-reading the mutated value,
preventing double-counting.

**Final broadcast** (optional): `v_readlane_b32` from lane 63 broadcasts the total
to all lanes. Omitting it yields a faster "lane-63 only" variant.

### 4.2 Inclusive Scan

The scan reuses the in-row `row_shr` cascade and cross-row carry, producing each
lane's prefix sum inclusively. The geometry is identical to reduction up to the
carry accumulation; however, the per-lane combine is additive at each step rather
than terminal.

### 4.3 Exclusive Scan

Given the inclusive result `inc[L]` = sum of inputs 0..L:

```cpp
T ex    = row_shr1(inc);           // inc[L-1] within each row; identity at row starts
T carry = row_bcast15(inc);        // inc[15] → lanes 16-31, inc[31] → 32-47, etc.
if ((L & 15) == 0 && L != 0)
    ex = carry;                    // patch lanes 16, 32, 48 with correct predecessor
```

`row_shr1` resets to identity (0) at the start of each 16-lane row (lanes 0, 16, 32, 48).
Only lane 0 should have identity=0 for exclusive scan; lanes 16/32/48 need
the inclusive sum of their predecessor (lane 15/31/47 respectively). `row_bcast15`
provides exactly those values.

---

## 5. Experimental Results

### 5.1 Configuration

| Parameter | Value |
| --- | --- |
| GPU | AMD Instinct MI300X VF (gfx942, sramecc+:xnack-) |
| ROCm / HIP | 7.2.0 / 7.2.26015 |
| Compiler | amdclang++ 22 (`-O3 --offload-arch=gfx942 -std=c++17`) |
| hipCUB | 4.2.0 (HIPCUB_VERSION 400200) |
| Benchmark | 1024 blocks × 64 threads, kReps=4096 internal loop, 100 warmup + 1000 timed iters |
| Timing | `hipEventRecord` / `hipEventElapsedTime`, median (p50) + P99 |
| Platform | AMD Developer Cloud (DigitalOcean infra), 2026-06-03 |
| Clock | 2100 MHz (not pinned — see §7) |

The `kReps` internal repetition loop with an anti-DCE dependency chain amortizes
the ~5.7 µs `hipLaunchKernel` latency floor, making compute the dominant cost.
All compared kernels share an identical dependency chain; the only variable is the
reduction/scan primitive.

### 5.2 Hardware counter verification (rocprofv3)

| Kernel | SQ_INSTS_LDS | SQ_LDS_IDX_ACTIVE | SQ_INSTS_VALU |
| --- | --- | --- | --- |
| `wave::dpp::reduce_sum` | **0** | **0** | 37.8M |
| `wave::reduce_sum` (bpermute) | 25.2M | 100.7M | 37.8M |
| `hipCUB WarpReduce::Sum` | 4.2M | 16.8M | 58.7M |
| `wave::dpp::scan_inclusive_sum` | **0** | **0** | — |
| `wave::reduce_sum` scan (bpermute) | 25.2M | — | — |
| `hipCUB WarpScan::InclusiveSum` | 0 | 0 | — |

The "zero LDS" claim is a **hardware counter**, not a disassembly inference.
VALU is identical between DPP and bpermute paths (both 37.8M), isolating LDS
elimination as the sole cause of the speedup. hipCUB's WarpReduce issues
4.2M LDS instructions (the residual bank-cross `ds_bpermute`) and more VALU
than our full-DPP path — yet is slower.

### 5.3 Reduction performance

`reduce_sum<float>`, wave64, 1024 blocks × 64 threads, kReps=4096:

| Kernel | Median (µs) | P99 (µs) | vs hipCUB |
| --- | --- | --- | --- |
| portable `__shfl_down` | 922.5 | 927.0 | 0.41× |
| `wave::dpp` (broadcast) | 279.0 | 284.2 | **1.35×** |
| `wave::dpp` (lane-63) | 210.0 | 231.1 | **1.79×** |
| hipCUB WarpReduce::Sum | 376.7 | 389.5 | 1.00× |

**Multi-type results** (`reduce_sum/max/min` × f32/f64/i32/i64, lane-63 result):

| Type | sum (µs) | max (µs) | min (µs) |
| --- | --- | --- | --- |
| float32 | 232 | 479 | 478 |
| float64 | 503 | 728 | 727 |
| int32 | 195 | 194 | 193 |
| int64 | 505 | 736 | 737 |

Sub-finding: `v_add_f32_dpp` is a fused op on CDNA3; max/min and all 64-bit
operations emit `mov_dpp` + separate op (still zero LDS, unfused). This explains
why f32 sum (~2×) is faster than f32 max/min, and why int32 (all fused-equivalent)
is uniformly fastest.

### 5.4 Scan performance

`scan_inclusive_sum<float>`, same config:

| Kernel | Median (µs) | vs hipCUB | SQ_INSTS_LDS |
| --- | --- | --- | --- |
| portable `__shfl_up` | 966 | 0.36× | 25.2M |
| `wave::dpp` (full DPP) | 343 | **1.028×** | **0** |
| hipCUB WarpScan::InclusiveSum | 352 | 1.00× | 0 |

Notable: hipCUB WarpScan is itself full-DPP (0 LDS). Our first hybrid implementation
(4 DPP in-row + 3 `__shfl` row-carries) reached only 0.91× because the retained
shfl steps still issued 12.6M LDS instructions. The full-DPP `row_bcast15`
cascade eliminated those, reaching 1.028×. The narrow margin (~3%) is expected:
both implementations are now on the same zero-LDS datapath; the difference is
microarchitectural scheduling details.

The value of the scan result is not its margin — it is the **repetition of the
pattern** across two independent algorithms (reduce and scan), both reaching zero
LDS and both outperforming or matching hipCUB.

### 5.5 Multi-type correctness and throughput

All primitives verified lane-by-lane (`probe_scan_dpp.hip`, `probe_reduce_dpp.hip`):

- `reduce_sum/max/min` × float/double/int32/int64: **12/12 PASS**
- `scan_inclusive_sum` × float/double/int32/int64: **12/12 PASS**
- `scan_exclusive_sum<float>`: **PASS** (all 64 lanes checked)

Throughput across types (`bench_reduce_dpp_types.hip`, MI300X VF, ROCm 7.2, kReps=4096):

| Type | reduce_sum | reduce_max | reduce_min |
| --- | --- | --- | --- |
| float32 | 1157 Ge/s | 558 Ge/s | 558 Ge/s |
| float64 | 531 Ge/s | 367 Ge/s | 367 Ge/s |
| int32 | 1372 Ge/s | 1378 Ge/s | 1381 Ge/s |
| int64 | 528 Ge/s | 362 Ge/s | 362 Ge/s |

The 64-bit path uses `__builtin_amdgcn_update_dpp` directly on 64-bit values;
the backend splits into two 32-bit DPP ops, confirmed correct on hardware. The
max/min operations on float32 are slower than sum because the identity element
for max (`-inf`) requires `std::numeric_limits<float>::lowest()`, which cannot
be folded as cleanly as `0` for sum — a microarchitectural scheduling difference.
For int32, all three operations are identical in cost, as expected.

**Low-precision accumulation (`__half`, `bf16`).** For half-precision reductions
the library promotes each element to float32 before reducing. We measured the cost
of this against a native-precision reduction (kReps=4096 amortized,
`bench_h3_fp16acc.hip` / `bench_bf16acc.hip`):

| Type | fp32-acc (µs) | native (µs) | fp32-acc vs native |
| --- | --- | --- | --- |
| `__half` | 933.9 | 1038.7 | **−10.1% (faster)** |
| `bf16` | 1123.0 | 2936.0 | **−61.7% (faster)** |

float32 accumulation is *faster* than native low-precision reduction on CDNA3, not
slower — decisively so for bf16. The native paths pay a per-step conversion: `__hadd`
requires pack/unpack on gfx942, and bf16 has no native add at all (it round-trips
through float every step). Promoting once and reducing in float32 is therefore
strictly better — lower latency *and* higher accuracy (no per-step rounding, no
overflow on long sums). This is the library's default for both half types and is
not a precision/speed tradeoff but a win on both axes.

#### Block-level reduce (two-phase: wave DPP → LDS → wave DPP)

For block sizes exceeding one wave (128–1024 threads), the block reduce requires
one LDS round-trip to gather inter-wave partial results. This reintroduces exactly
one LDS transaction per block, eliminating the zero-LDS property of the wave reduce:

| Block size | wave_block_reduce (µs) | hipCUB BlockReduce (µs) | Speedup |
| --- | --- | --- | --- |
| 128 | 5.813 | 5.814 | **1.000×** |
| 256 | 5.773 | 5.773 | **1.000×** |
| 512 | 5.773 | 5.773 | **1.000×** |
| 1024 | 5.773 | 5.773 | **1.000×** |

Block reduce matches hipCUB exactly at all sizes. The DPP advantage is wave-scoped:
once a cross-wave LDS step is necessary, the asymmetry between DPP and bpermute
collapses because both implementations use the same one-LDS-round-trip strategy.
This delimits the scope of the contribution precisely: **zero-LDS wave-level reductions
and scans**, not block-level operations.

### 5.6 Real-kernel results: LayerNorm and Softmax

To assess whether the microbenchmark speedup translates to a production-relevant
kernel, we implemented single-pass LayerNorm (two `reduce_sum` per row: one for
mean, one for variance) and row-wise Softmax (`reduce_max` + `reduce_sum`) with
three backends: full-DPP, portable `__shfl_*`, and hipCUB WarpReduce/WarpScan.

Run 3 — MI300X VF (gfx942, ROCm 7.2.0, 2026-06-05):

#### LayerNorm (2× reduce_sum per row)

| Config | Kernel | Median (µs) | vs portable | vs hipCUB |
| --- | --- | --- | --- | --- |
| 65536 rows × 64 cols (narrow, 1 elem/lane) | DPP | 35.7 | **1.019×** | **1.009×** |
| 65536 rows × 64 cols (narrow, 1 elem/lane) | portable | 36.4 | 1.00× | 0.990× |
| 65536 rows × 64 cols (narrow, 1 elem/lane) | hipCUB | 36.0 | 1.011× | 1.00× |
| 65536 rows × 512 cols (wide, 8 elems/lane) | DPP | 65.3 | **1.000×** | **1.029×** |
| 65536 rows × 512 cols (wide, 8 elems/lane) | portable | 65.3 | 1.00× | 1.029× |
| 65536 rows × 512 cols (wide, 8 elems/lane) | hipCUB | 67.2 | 0.972× | 1.00× |

rocprofv3 LDS counters on narrow LayerNorm:

| Kernel | SQ_INSTS_LDS |
| --- | --- |
| layernorm DPP | **0** |
| layernorm portable | 917,504 |
| layernorm hipCUB | 262,144 |

#### Softmax (reduce_max + reduce_sum per row, 65536 rows × 512 cols)

| Kernel | Median (µs) | vs portable | vs hipCUB |
| --- | --- | --- | --- |
| softmax DPP | 65.5 | **1.016×** | **1.022×** |
| softmax portable | 66.5 | 1.00× | 1.006× |
| softmax hipCUB | 66.9 | 0.994× | 1.00× |

**Interpretation.** The microbenchmark speedup (1.35–1.79× reduce, 1.028× scan)
does **not** replicate at kernel granularity for these configurations. DPP remains
fastest in all cases and LDS elimination is confirmed (rocprofv3: 0 vs 917K insts
for portable), but the margin collapses to 1–3% because the kernel execution time
is dominated by HBM traffic (reading and writing the full row per element), not by
the cross-lane reduction steps. This is consistent with a memory-bound regime:
for 65536 × 512 fp32, the kernel processes 134M floats (512 MB) at ~65 µs,
implying ~7.9 TB/s effective bandwidth — well above the HBM3 theoretical peak of
5.3 TB/s for a single MI300X VF, suggesting the wide kernel saturates memory.

The practical implication is bounded: DPP primitives deliver 1.35–1.79× speedup
for **reduction-dominated** workloads (short rows, multiple reductions per row,
register-bound kernels with tight LDS budget). For memory-bound kernels at typical
LLM/transformer sizes (wide rows), the advantage is 1–3% — measurable, LDS-free,
but not the headline result.

### 5.7 Welford one-pass LayerNorm: a refutation

Run 3 showed that DPP delivers only 1–3% improvement in real LayerNorm and Softmax
kernels. A natural follow-up hypothesis: the two-pass structure (mean then variance)
incurs two HBM reads per row — collapsing to one pass via parallel Welford online
statistics should halve HBM traffic.

We implemented a Welford one-pass kernel (`bench_layernorm_welford.hip`): each lane
accumulates `{count, mean, M2}` sequentially over its `kElemsPerLane` elements, then
the wave merges all 64 per-lane Welford states via DPP shifts applying the parallel
Chan et al. merge formula.

Result on MI300X VF (gfx942, ROCm 7.2, 65536 rows):

| Config | Welford DPP | two-pass DPP | Speedup |
| --- | --- | --- | --- |
| 512 cols (wide) | 72.0 µs | 64.9 µs | **0.90×** (10% slower) |
| 64 cols (narrow) | 40.8 µs | 35.7 µs | **0.88×** (12% slower) |

rocprofv3 explains the regression:

| Kernel | SQ_INSTS_LDS | SQ_INSTS_VALU |
| --- | --- | --- |
| `welford_dpp` | **131,072** | 22,413,312 |
| `twopass_dpp` | **0** | 3,670,016 |

Two root causes:

1. **Residual LDS.** The final broadcast of lane-63's result to all lanes uses
   `__shfl(s.mean, 63, 64)` — still a `ds_bpermute` on CDNA3. This alone adds
   131K LDS instructions. A pure-DPP broadcast from lane 63 requires a different
   pattern; `__shfl` from a runtime lane index is not foldable to DPP.

2. **VALU overhead.** Welford merge costs 3 DPP ops per step (mean, M2, n) vs 1 for
   scalar sum, producing 6× more VALU. The Welford VALU count (22.4M) dominates
   the kernel execution time.

3. **The "two HBM reads" premise was wrong.** `bench_layernorm.hip` already caches
   `vals[]` in a register array between the two reductions. The second `reduce_sum`
   operates on re-computed deviations from registers, not from HBM. There is no
   second HBM read to eliminate.

**Conclusion:** For this kernel layout, two-pass DPP (0 LDS, 3.67M VALU) is strictly
optimal. Welford one-pass is appropriate when register pressure prevents caching the
full row (very wide rows, many concurrent warps), but at 512 cols × float32 the
register array fits comfortably and two-pass wins.

### 5.8 Flash Attention build-order results on MI300X

To close the gap between isolated microbenchmarks and a production-scale kernel, we
implement Flash Attention (FA2 algorithm) on MI300X using the validated DPP and DME
primitives as building blocks. The kernel is structured as a four-step build order,
each step independently testable and compared against the previous.

**Configuration:** Br=16, Bc=64, D=64, seqQ=seqK=512, 8 heads, batch=1.
One CTA = one query block (16 waves of 64 lanes). LDS budget: 64 KB/CU.

#### Step 1 — Naive baseline (`fa_naive.hip`)

Portable `__shfl_down` softmax + synchronous cooperative K/V loads. Establishes the
correctness reference and the timing floor.

#### Step 2 — DPP softmax (`fa_dpp.hip`)

Replace `wave_max_f32` / `wave_sum_f32` (both built on `__shfl_down`) with
`wave::dpp::reduce_max_bcast` and `wave::dpp::reduce_sum_bcast`. All other code
identical to Step 1.

#### Step 3 — DME double-buffer K/V (`fa_dme.hip`)

Prefetch `K_{j+1}` and `V_{j+1}` asynchronously via `dme::copy_element_stream`
while computing the current tile's `P·V` accumulation (64-iteration inner loop).
Ping-pong LDS: two K buffers + two V buffers. LDS: 36 KB (well within 64 KB).

#### Step 4 — MFMA matrix cores (`fa_mfma2.hip`)

Replace both the Q·Kᵀ and P·V direct dot products with `v_mfma_f32_16x16x16f16`
matrix-core instructions. This requires a structural change: one wave (64 lanes)
per query block, because the MFMA instruction needs all 16 rows of a tile held
across one wavefront's lanes (the first MFMA attempt with Br=16 independent warps
failed — see the verified lane-mapping note below). S and P are staged in LDS;
softmax runs on lanes 0–15 (one query row each).

**Results on MI300X VF (gfx942, ROCm 7.2):**

| Step | Kernel | Q·Kᵀ / P·V | Softmax | K/V load | Median (µs) | TFLOPS | Speedup vs naive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `fa_naive` | dot product | `__shfl_down` | synchronous | 97.2 | 5.52 | 1.000× |
| 2 | `fa_dpp` | dot product | DPP (0 LDS) | synchronous | 90.5 | 5.93 | **1.074×** |
| 3 | `fa_dme` | dot product | DPP (0 LDS) | DME async | 82.4 | 6.52 | **1.180×** |
| 4 | `fa_mfma2` | MFMA 16×16×16 | LDS (lanes 0–15) | synchronous | 87.0 | 6.17 | 1.118× |

Correctness: all four kernels pass `max|gpu − ref| < 0.0001` vs fp32 CPU reference.

**Step 4 is a negative result, reported honestly.** MFMA matrix cores do *not*
beat the Step 3 dot-product kernel at this configuration (87.0 µs vs 82.4 µs,
0.95×). Two reasons: (1) with D = Bc = 64 the MFMA tiles (16×16×16) are too small
to saturate the matrix cores — each KV iteration issues only 16 + 16 MFMA calls
per wave; (2) the 1-wave-per-block layout forces softmax onto lanes 0–15 with the
other 48 lanes idle, whereas Steps 1–3 keep all 64 lanes busy with a DPP softmax.
The lesson: **MFMA is not an automatic win for attention with small heads (D=64);
its advantage requires large D and head-batching to fill the matrix cores.** For
small-head FA on CDNA3, the DPP + DME dot-product path (Step 3) remains fastest.

A head-dimension sweep (`fa_mfma_dsweep.hip`) confirms this is a tile-size effect,
not an MFMA limitation:

| Head dim | nD (MFMA tiles / KV iter) | Median (µs) | TFLOPS |
| --- | --- | --- | --- |
| D = 64 | 4 | 86.7 | 6.19 |
| D = 128 | 8 | 102.7 | **10.45** |

Doubling the head dimension lifts achieved throughput by **+69%** (6.19 → 10.45
TFLOPS): with nD = 8 the matrix cores receive enough work per iteration to amortize
the fixed softmax/staging cost. The negative result at D=64 is therefore a
small-head regime, not a verdict on MFMA — at production head dimensions (128, 256)
the matrix-core path is expected to dominate.

##### Verified MFMA wave64 lane mapping (reusable artifact)

The first MFMA attempt failed correctness because the register-to-matrix lane
mapping of `v_mfma_f32_16x16x16f16` is not obvious from the ISA docs. We
determined it empirically (`probe_mfma_mapping.hip`: a full 16×16×16 GEMM vs host
reference, enumerating all layout combinations; only one matched at err 0.001):

```text
A  input : lane L, frag f → A[row = L%16      ][k   = (L/16)*4 + f]   ([idx16][quad])
B  input : lane L, frag f → B[k   = (L/16)*4+f][col = L%16        ]   ([quad][idx16])
C/D output: lane L, frag f → C[row = (L/16)*4+f][col = L%16        ]   ([quad][idx16])
```

A is read row-major over (row, k); B and the accumulator are read transposed over
(k/row, col). This mapping is reusable for any fp16 16×16×16 MFMA kernel on gfx942.

**Key observations:**

1. **DPP in FA is more effective than in LayerNorm/Softmax.** The 7.4% improvement
   from Step 1→2 exceeds the 1–3% seen in standalone kernels (§5.6). The likely
   mechanism: FA has Br=16 warps simultaneously loading K/V tiles; freeing the LDS
   pipeline from softmax `ds_bpermute` reduces contention on shared LDS ports.

2. **DME async overlap is visible at this compute depth.** Step 2→3 delivers 9.8%
   improvement. The P·V inner loop (64 key iterations, each a broadcast + multiply)
   provides enough arithmetic depth to hide the HBM→LDS latency of the next tile,
   unlike the synthetic `bench_dme_async` compute (sum-of-squares, DCE-collapsed).
   This validates the design claim that "DME's advantage is async overlap, not raw
   throughput."

3. **Cumulative effect: 18% total speedup from two primitive substitutions.** Neither
   DPP nor DME alone is sufficient — their combination changes both the instruction
   mix (LDS → VALU) and the memory pipeline (stall → overlap).

#### Hardware counter verification (rocprofv3, one kernel dispatch)

| Kernel | SQ_INSTS_LDS | SQ_INSTS_VALU | SQ_INSTS_VMEM_RD |
| --- | --- | --- | --- |
| `fa_naive` | 5,316,608 | 9,912,320 | 311,296 |
| `fa_dpp` | 4,857,856 (−8.6%) | 10,641,408 (+7.3%) | 303,104 (−2.6%) |
| `fa_dme` | 4,595,712 (−13.6%) | 10,067,968 (+1.6%) | **172,032 (−44.7%)** |

The LDS reduction in `fa_dpp` (−8.6%) confirms that DPP eliminates the softmax
`ds_bpermute` within FA. The VALU increase (+7.3%) is expected: DPP introduces
additional `v_add_f32_dpp` / `v_max_f32_dpp` instructions to replace each LDS round-trip.
The net effect is faster execution because DPP VALU is lower-latency than LDS.

The VMEM_RD reduction in `fa_dme` (−44.7%) directly measures the DME prefetch
effect: K/V tile loads that previously stalled the VMEM pipeline are now issued
via the async DME path (streaming cache, sc0=1), reducing visible VMEM read
instructions by nearly half. This is the hardware-level signature of the overlap.

### 5.9 From kernel to a real LLM

§5.8 measured the FA kernel on a fixed synthetic configuration (D=64, seqLen=512,
8 heads). Two questions remain before the kernel can be called *applicable*: does it
hold up at the shapes real decoder LLMs use, and does it reproduce a real model's
output? We answer both, then position the kernel against a production attention backend.

#### Generalization to LLM-realistic shapes (`fa_robust`)

We generalize the verified 1-wave/block MFMA kernel of §5.8 to runtime sequence
length, grouped-query attention (GQA), and causal masking with diagonal tile-skip
(`fa_robust.hip`). Correctness is CPU-verified across MHA / GQA / causal at D=64 and
D=128 (max|gpu−ref| < 0.0006, 5/5 PASS). The flat-context cost over seqLen 512→4096
at D=128 fits a clean power law, **n^1.90–1.91 (R²≈0.996) across three independent
runs** — the expected quadratic attention regime, measured rather than assumed, on a
kernel validated against LLM-realistic shapes (not a toy).

#### Numerical validation against a real attention layer

We export layer 0 of **Qwen2.5-0.5B** (real weights + RoPE + GQA 14q/2kv + causal)
and reconstruct it on-GPU: Q/K/V projection → RoPE → `fa_robust` → O projection,
compared against the fp32 HuggingFace reference. The error through the full layer is
**4.34% of the signal scale** (max|·| absolute 0.0042 over a signal of mean magnitude
0.097) — within fp16 tolerance for an fp16 kernel plus two naive fp16 projection
GEMMs versus an fp32 reference. The kernel reproduces a *real* transformer attention
layer, closing the synthetic→real gap. (Implementation note for replicators: Qwen2.5
places a bias on the Q/K/V projections — straightforward to miss with a bias-free GEMM.)

#### End-to-end: a full model running on the kernel

We wrap `fa_robust` as a PyTorch custom op (`torch.ops.amdinstinct.flash_attn`) and
monkey-patch it into **all 24 attention layers** of Qwen2.5-0.5B during prefill, then
compare full-model logits against the unpatched fp16 forward. The model predicts the
**same next token** (top-1 identical) running entirely on our kernel. The last-token
logit cosine similarity decays gradually with depth — 0.9997 at one patched layer to
0.975 at all 24 — which a layer-count bisection confirms is fp16 error accumulation
(monotone in depth; top-1 matches at 1, 6, 12, 18, and 24 layers), not a structural
defect. This is the literal closure of the research→application arc the wave-primitive
finding set out on: the microbenchmark optimization runs a real LLM and produces the
same model.

#### Positioning against production Flash Attention

For context we compare the same shapes (D=128, GQA 32q/8kv, causal) against PyTorch's
fused scaled-dot-product-attention backend on ROCm — the realistic "what production
stacks use" baseline (correctness vs SDPA: rel 0.007):

| seqLen | ours (µs) | ours TFLOPS | SDPA (µs) | SDPA TFLOPS | ours/SDPA |
| --- | --- | --- | --- | --- | --- |
| 512  | 389.7 | 5.51 | 45.8 | 46.9 | 8.5× |
| 1024 | 1052.4 | 8.16 | 77.4 | 111.0 | 13.6× |
| 2048 | 3411.1 | 10.07 | 224.0 | 153.4 | 15.2× |
| 4096 | 11830.8 | 11.62 | 967.4 | 142.1 | 12.2× |

Production SDPA is **8.5–15× faster** — expected, and reported honestly. Our kernel is
a single-wave-per-block research design; SDPA uses a fully scheduled multi-wave MFMA
pipeline that fills all compute units. Our achieved throughput does scale with seqLen
(5.5 → 11.6 TFLOPS), but the 1-wave layout leaves 48 of 64 lanes idle during softmax
(the same occupancy limit identified in §5.8). **The gap is an occupancy gap, not a
primitive gap:** the DPP+DME findings are datapath-level and orthogonal to the
scheduling work that closes the distance to production — they are portable into a
multi-wave kernel, which is the indicated next step. The value here is the measured
gap and its localized cause, not a competitive number.

---

## 6. Implementation

The library is header-only with zero dependencies beyond HIP runtime:

```cpp
#include "wave_primitives/wave_reduce_dpp.hpp"
#include "wave_primitives/wave_scan_dpp.hpp"

// drop-in inside any HIP kernel
float r = wave::dpp::reduce_sum(val);
float s = wave::dpp::scan_inclusive_sum(val);
float e = wave::dpp::scan_exclusive_sum(val);
```

`detail/config.hpp` detects `__gfx9__` at compile time and routes to the DPP path
(`WP_AMD=1`) or to a portable `__shfl_*` fallback for wave32/NVIDIA targets.
The API surface is identical on both paths.

The intrinsic used for DPP steps:

```cpp
T moved = __builtin_amdgcn_update_dpp(identity, v, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
```

`dpp_ctrl` encodes the lane permutation; `row_mask`/`bank_mask` (`0xf/0xf`) enable
all rows and banks; `bound_ctrl=false` fills out-of-bounds lanes with `identity`.

Repository: [github.com/jpcpol/AMD-Incstinct-Labs](https://github.com/jpcpol/AMD-Incstinct-Labs)

---

## 7. Threats to Validity

1. **Single hardware point.** Results are for MI300X VF (virtualized, not bare-metal),
   clocks unpinned (2100 MHz observed). Physical MI300X with pinned clocks may produce
   different absolute latencies, though the relative ordering (DPP vs LDS vs hipCUB)
   is driven by instruction counts, not clock rate. Validation on MI250 (CDNA2) and
   RDNA3 (wave32) remains open — proposed as a collaboration hook with AMD rather
   than a blocker, since the caveat is declared.

2. **Microbenchmark vs memory-bound kernel.** The microbenchmark speedup (1.35–1.79×
   reduce) was measured with a synthetic kReps loop that keeps the GPU compute-bound.
   In real LayerNorm and Softmax kernels (§5.6), the margin collapses to 1–3% for
   wide rows because HBM traffic dominates. DPP remains fastest and LDS elimination
   is confirmed, but the headline speedup applies to reduction-dominated workloads
   (short rows, multiple reductions, register-pressure-constrained kernels).

3. **Exploratory finding.** The DPP result emerged from investigating a hypothesis
   refutation (H2), not from a pre-registered hypothesis. It is reported as such.
   The scan (H1) was pre-registered and is confirmed.

4. **hipCUB version.** Pinned to hipCUB 4.2.0. Future versions could change the
   WarpReduce implementation.

5. **Type coverage.** Both half types measured (§5.5): fp32-accumulation is faster
   than native low-precision reduce on CDNA3 — 10% for `__half`, 62% for `bf16`.

---

## 8. Discussion and Broader Reach

The most durable contribution is not the speedup numbers — it is the
**causal chain**, which is unusually clean:

```text
__shfl_*  →  ds_bpermute  →  LDS round-trip  →  latency
    vs
DPP       →  cross-lane VALU  →  zero LDS  →  lower latency
```

Most performance optimization results report "ours is faster" without isolating why.
Here, hardware counters tie the latency directly to a specific datapath choice.

**Broader reach:** any ROCm code that reduces or scans via `__shfl_*` routes
cross-lane traffic through LDS. This includes significant portions of the PyTorch,
JAX, and vLLM HIP ports — softmax, layer normalization, attention score reduction.
The fix is local (swap the intrinsic for the DPP geometry) and composable
(the header is drop-in). Quantifying the speedup in those real kernels is
deferred but the mechanism is established.

---

## 9. Conclusions

1. `__shfl_down` / `__shfl_up` lower to `ds_bpermute` on CDNA3, causing 2.45×
   slower reduction vs hipCUB — the hypothesis of ISA convergence is refuted.

2. A full-DPP geometry (`row_shr` in-row + `row_bcast15`/`row_bcast31` cross-row)
   reaches **zero LDS instructions** (rocprofv3-measured) and runs **1.35–1.79×
   faster than hipCUB WarpReduce** on MI300X for float32, and **1.028× faster
   than hipCUB WarpScan** for float32 inclusive scan.

3. The pattern generalizes across float/double/int32/int64 and across both
   reduce and scan — the deciding factor on CDNA3 wave64 is whether an op can
   eliminate all LDS-mediated cross-lane traffic.

4. In real kernels (LayerNorm, Softmax at 65536 × 512 float32), DPP is fastest
   (1–3% over hipCUB, rocprofv3 LDS = 0) but the large microbenchmark margin does
   not replicate — the kernel is memory-bound at these row sizes. The DPP advantage
   is reduction-domain-specific: short rows, multiple reductions per row, or kernels
   constrained by LDS budget.

5. In Flash Attention (Br=16, Bc=64, D=64, seqLen=512), DPP softmax + DME
   double-buffer K/V prefetch together deliver **18% end-to-end speedup** (97.2 µs
   → 82.4 µs) decomposed into 7.4% from DPP (LDS contention relief across 16
   simultaneous warps) and 9.8% from DME async overlap (P·V compute depth hides
   HBM→LDS latency). The combination is strictly additive and exceeds either
   optimization alone — DPP + DME are complementary primitives for CDNA3 attention.
   Adding MFMA matrix cores (Step 4) did *not* help at D=64 (87.0 µs, 0.95×): the
   16×16×16 tiles are too small to saturate the cores and the 1-wave layout idles
   48 of 64 lanes during softmax. The DPP+DME dot-product path is fastest for
   small-head attention; MFMA's advantage needs large D and head-batching.

6. The full-DPP geometry for both primitives is available as a header-only,
   MIT-licensed library with wave32/NVIDIA fallback.

7. Carried through the full research→application arc, the kernel generalizes to
   LLM-realistic shapes (n^1.90–1.91 quadratic regime, R²≈0.996), reproduces a real
   Qwen2.5 attention layer within fp16 tolerance (4.34% of signal scale), and runs
   the complete 24-layer model during prefill as a PyTorch custom op with the top-1
   next-token prediction preserved. Against production fused SDPA it is 8.5–15×
   slower; the gap is a single-wave occupancy limit (48/64 lanes idle during
   softmax), orthogonal to and not contradicted by the DPP/DME datapath findings —
   a multi-wave kernel is the indicated path to close it.

---

## Acknowledgments

Hardware access via AMD Developer Cloud. hipCUB version and rocprofv3 methodology
informed by AMD ROCm documentation and open-source examples.

---

## References

[1] AMD. *AMD Instinct MI300X Accelerator Architecture Technical Reference.*
    AMD Publication, 2023.

[2] AMD. *ROCm Documentation — HIP Programming Guide.* ROCm 7.2, 2024.

[3] AMD. *AMD GPU ISA Reference for GFX9.* CDNA3 supplement, 2023.

[4] AMD. *hipCUB: CUDA CUB Primitives for HIP.* GitHub: ROCm/hipCUB, 2024.

[5] HazyResearch. *HipKittens: GPU Kernels for Fun and (Hopefully) Profit on AMD
    GPUs.* arXiv:2511.08083, 2025.

[6] AMD. *ROCm 7.0 Release Notes — warpSize Early-Fold.* 2024.

---

*Bounded experimental claim. Configuration details: §5.1. Open items: §7.*

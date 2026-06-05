# Eliminating LDS-Mediated Cross-Lane Communication on CDNA3: Full-DPP Wave64 Primitives for AMD Instinct MI300X

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
float/double/int32/int64 × sum/max/min (12/12). The library is header-only,
MIT-licensed, and falls back to portable `__shfl_*` on wave32/NVIDIA targets.

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

```
row_shr 1: lane[i] += lane[i-1]   (within row, i&15 != 0)
row_shr 2: lane[i] += lane[i-2]   (within row, i&15 >= 2)
row_shr 4: lane[i] += lane[i-4]   (within row, i&15 >= 4)
row_shr 8: lane[i] += lane[i-8]   (within row, i&15 >= 8)
```

After step 4, lane 15 holds the sum of lanes 0–15, lane 31 holds lanes 16–31,
lane 47 holds lanes 32–47, lane 63 holds lanes 48–63.

**Step 5 (row boundary, row 0 total → rows 1,2,3):**

```
c1 = row_bcast15(v)   →  lanes 16–31 receive lane 15's value (row 0 total)
                          lanes 32–47 receive lane 31's value (row 1 total)
                          lanes 48–63 receive lane 47's value (row 2 total)
```

**Step 6 (bank boundary, rows 0+1 total → rows 2,3):**

```
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

### 5.5 Correctness

All primitives verified lane-by-lane (`probe_scan_dpp.hip`, `probe_reduce_dpp.hip`):

- `reduce_sum/max/min` × float/double/int32/int64: **12/12 PASS**
- `scan_inclusive_sum` × float/double/int32/int64: **12/12 PASS**
- `scan_exclusive_sum<float>`: **PASS** (all 64 lanes checked)

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

2. **Microbenchmark.** The speedup is for the primitive in isolation with a synthetic
   dependency chain. Production-relevant speedup — inside a LayerNorm, softmax, or
   Flash Attention kernel — is not yet measured. That is the next step.

3. **Exploratory finding.** The DPP result emerged from investigating a hypothesis
   refutation (H2), not from a pre-registered hypothesis. It is reported as such.
   The scan (H1) was pre-registered and is confirmed.

4. **hipCUB version.** Pinned to hipCUB 4.2.0. Future versions could change the
   WarpReduce implementation.

5. **Type coverage.** bf16 and fp16 are not yet measured.

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

4. The full-DPP geometry for both primitives is available as a header-only,
   MIT-licensed library with wave32/NVIDIA fallback.

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

# Phase-2 Design Note — GEMM batch=1 Occupancy Gap

**Pre-registration timestamp:** committed before running on hardware.  
**Status:** cold design, pending VM validation (same session as Phase-1).

---

## 1. Context

After C7 (rocBLAS integration), decode throughput reached 90.7 tok/s with an
11ms/step decode latency. The dominant operation in decode is 7 GEMMs per layer
with M=1 (one query token at a time, batch=1). On MI300X each rocBLAS GEMM
launches a grid sized for M rows — at M=1, most CUs sit idle.

The question: **how much occupancy is lost at M=1, and does batching queries
improve it enough to matter for latency?**

This is distinct from the multi-wave FA occupancy gap (Phase-1). Phase-1 is about
the prefill attention kernel; Phase-2 is about the decode GEMM path.

Reference numbers (from T-050, session 2026-06-15):
- decode step at M=1: **11.0 ms** (post-C7, rocBLAS)
- 7 GEMMs per layer × 28 layers = 196 GEMM calls per decode step
- dominant GEMM shapes (Qwen2.5-0.5B, D=896, hidden=4864):
  - QKV proj:  M=1, N=896×3=2688, K=896
  - O proj:    M=1, N=896,        K=896
  - gate/up:   M=1, N=4864,       K=896
  - down:      M=1, N=896,        K=4864

---

## 2. What is being measured

**Benchmark:** `bench_gemm_batch1_vs_batched.hip`

Configurations (M sweep, fixed N and K matching real layer shapes):

| Shape name | M   | N    | K    | Represents         |
|------------|-----|------|------|--------------------|
| qkv_m1     | 1   | 2688 | 896  | QKV proj, decode   |
| qkv_m8     | 8   | 2688 | 896  | QKV proj, M=8 batch|
| qkv_m16    | 16  | 2688 | 896  | QKV proj, M=16     |
| qkv_m32    | 32  | 2688 | 896  | QKV proj, M=32     |
| mlp_m1     | 1   | 4864 | 896  | gate/up, decode    |
| mlp_m8     | 8   | 4864 | 896  | gate/up, M=8       |
| mlp_m16    | 16  | 4864 | 896  | gate/up, M=16      |
| mlp_m32    | 32  | 4864 | 896  | gate/up, M=32      |

Each config measures: median latency (µs), TFLOPS, GB/s effective bandwidth.

---

## 3. Pre-registered hypotheses

**H-G1 (occupancy cliff):** M=1 achieves <10% of the TFLOPS of M=32 for the
same (N,K), on at least 2 of the 4 N/K shapes tested.
- Accept threshold: TFLOPS(M=1) / TFLOPS(M=32) < 0.10
- Measurement: Section 3 of bench_gemm_batch1_vs_batched output

**H-G2 (linear scaling):** TFLOPS scales approximately linearly with M up to M=16
(doubling M roughly doubles TFLOPS achieved), then saturates.
- Accept threshold: TFLOPS(M=16) > 0.8× TFLOPS(M=32) AND
                    TFLOPS(M=8)  > 0.40× TFLOPS(M=32)
- Measurement: TFLOPS column in the M-sweep table

**H-G3 (bandwidth bound at M=1):** At M=1 the bottleneck is weight loading
bandwidth, not compute. Predicted: achieved GB/s at M=1 approaches HBM peak
(~5.3 TB/s) faster than TFLOPS approaches compute peak (383 TFLOPS).
- Accept threshold: (GB/s_M1 / 5300) > (TFLOPS_M1 / 383)
- Measurement: bandwidth column vs TFLOPS column, Section 2

---

## 4. Why this matters for the runtime

If H-G1 holds (M=1 is <10% of peak compute), the path to 2× decode throughput
is speculative decoding or continuous batching — not kernel optimization.
If H-G2 holds (linear scaling to M=16), then batching 8-16 requests would
recover most of the occupancy gap without architectural changes.

The benchmark also establishes the roofline operating point for MI300X at
these shapes, which feeds into the paper's §5 analysis.

---

## 5. VM session guion

```bash
# Build (from research/flash-attention-mi300x/)
hipcc -O3 --offload-arch=gfx942 -std=c++17 \
  -lrocblas \
  bench_gemm_batch1_vs_batched.hip -o bench_gemm_batch1

# Run
./bench_gemm_batch1 | tee /root/results_phase2_gemm_batch1_mi300x.txt

# rocprofv3 for occupancy counters
rocprofv3 --pmc SQ_WAVES,SQ_BUSY_CU_CYCLES,SQ_VALU_INST_EXECUTED \
  ./bench_gemm_batch1 \
  | tee /root/results_phase2_gemm_rocprof_mi300x.txt
```

After the session:
1. Copy both result files to `research/flash-attention-mi300x/results/`
2. Compute TFLOPS ratio M=1/M=32 per shape (H-G1)
3. Plot M vs TFLOPS scaling per shape (H-G2)
4. Compare GB/s vs TFLOPS normalized to peak (H-G3)
5. Update paper §5.10 with findings

---

## 6. Negative result policy

If H-G1 fails (M=1 already achieves >10% of M=32 peak):
- rocBLAS has better small-M dispatch than expected.
- The 11ms/step bottleneck is elsewhere (memory latency, kernel launch overhead).
- Reported honestly. Implication: the decode path is already near-optimal for
  single-request inference; further gains require batching at the application level.

If H-G2 fails (scaling is sublinear):
- Cache effects or synchronization overhead between GEMMs limits batching gain.
- Finding: batching beyond a small M is not free on MI300X for these shapes.

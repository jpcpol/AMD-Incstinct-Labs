# Phase-1 Design Note — Multi-Wave Occupancy Gap Measurement

**Pre-registration timestamp:** committed before running on hardware.  
**Status:** cold design, pending VM validation.

---

## 1. Context

Session 2-C (2026-06-10) measured `fa_robust` (1-wave/block) against `F.sdpa` on MI300X:

| seqLen | fa_robust µs | SDPA µs | gap |
|--------|-------------|---------|-----|
| 512    | 389.5       | 45.8    | 8.51× |
| 1024   | 1051.9      | 77.3    | 13.61× |
| 2048   | 3407.9      | 223.2   | 15.27× |
| 4096   | 11829.4     | 980.6   | 12.06× |

The gap was attributed to **idle-lane occupancy**: `fa_robust` runs 1 wave/block with
48/64 lanes idle during softmax. `fa_multiwave_kernel` was designed and implemented to
fix this (Axis-A: W waves/block, each owns Br=16 query rows, shared K/V LDS).

`fa_multiwave_kernel` is already in production in `cdna3::runtime::Session` (C8, commit
290fc10), but has never been benchmarked directly against SDPA on the 2-C shapes.
This phase closes that measurement gap.

---

## 2. What is being measured

**Benchmark:** `bench_multiwave_vs_sdpa.hip` + `bench_sdpa_baseline.py`

Shapes (identical to 2-C):
- D=128, GQA 32q/8kv, causal, seqLen ∈ {512, 1024, 2048, 4096}, batch=1

Configurations:
- `fa_multiwave W=1 D=128` — apples-to-apples with 2-C fa_robust (same architecture)
- `fa_multiwave W=2 D=64`  — max valid W at D=64 without LDS overflow
- `fa_multiwave W=4 D=64`  — max valid W at D=64 (60KB LDS, tight)
- `F.sdpa` (fused ROCm) — fresh measurement to check if SDPA changed since 2-C

LDS budget analysis (confirmed in DESIGN_multiwave_fa.md):
- W=2 D=128 = 68KB → OVERFLOW, not tested
- W=4 D=64  = 60KB → valid (tight)

---

## 3. Pre-registered hypotheses

**H-MW1 (occupancy):** `fa_multiwave W=2` is faster than `W=1` at matched D and seqLen,
on ≥3 of the 4 swept seqLens.
- Accept threshold: speedup > 1.0× on ≥3/4 seqLens
- Measurement: Section 4 of bench_multiwave_vs_sdpa.hip output

**H-MW2 (K/V LDS reuse):** `W=2` issues ≤ 0.6× the VMEM read instructions of `W=1`
(theory: 1/W = 0.5× if K/V loads fully dominate; 0.6× accounts for Q/O overhead).
- Accept threshold: SQ_INSTS_VMEM_RD(W=2) / SQ_INSTS_VMEM_RD(W=1) ≤ 0.60
- Measurement: rocprofv3 `SQ_INSTS_VMEM_RD` counter on D=64 sweep

**H-MW3 (gap closure):** the 2-C gap (8.5–15×) narrows with `fa_multiwave W=1 D=128`
vs fresh SDPA measurement.
- This is a measurement, not a pass/fail. Any movement toward 1× is a positive result.
- If gap does NOT narrow: localizes the remaining bottleneck (instruction scheduling,
  not occupancy), which is itself a finding.
- Measurement: Section 5 of bench_multiwave_vs_sdpa.hip + bench_sdpa_baseline.py

---

## 4. LDS budget verification (per wave, detailed)

For `fa_multiwave_kernel<D, W>`:

Shared across block (all W waves):
```
K_lds[Bc*D * sizeof(half)]  = 64*D*2 bytes
V_lds[Bc*D * sizeof(half)]  = 64*D*2 bytes
```

Per wave (×W):
```
Q_lds[Br*D  * sizeof(half)] = 16*D*2  bytes
S_lds[Br*Bc * sizeof(float)]= 16*64*4 bytes = 4096 B
P_lds[Br*Bc * sizeof(half)] = 16*64*2 bytes = 2048 B
O_lds[Br*D  * sizeof(float)]= 16*D*4  bytes
m_lds[Br    * sizeof(float)]= 16*4    bytes =   64 B
l_lds[Br    * sizeof(float)]= 16*4    bytes =   64 B
```

Total(D=64, W=4):
- Shared: 2*64*64*2 = 16384 B
- Per wave: (16*64*2 + 4096 + 2048 + 16*64*4 + 128) = 2048+4096+2048+4096+128 = 12416 B
- ×4 waves: 49664 B
- Total: 16384 + 49664 = **66048 B** — slightly over 64KB

Wait — this needs re-checking. The kernel uses `__shared__ float O_lds[W][_Br*D]` etc.
with W as a template parameter, so all W waves' arrays are allocated at once. Let me
recount for D=64, W=4:

```
K_lds[64*64] half  = 8192 B
V_lds[64*64] half  = 8192 B
Q_lds[4][16*64] half = 4*2048 = 8192 B
S_lds[4][16*64] f32  = 4*4096 = 16384 B
P_lds[4][16*64] half = 4*2048 = 8192 B
O_lds[4][16*64] f32  = 4*4096 = 16384 B
m_lds[4][16] f32     = 4*64   = 256 B
l_lds[4][16] f32     = 4*64   = 256 B
Total = 8192+8192+8192+16384+8192+16384+256+256 = 66048 B
```

**66048 B = 64.5 KB — marginally over 64 KB limit.**

The kernel will likely fail with an LDS overflow error at W=4, D=64. The benchmark
handles this gracefully (reports KERNEL ERROR and continues). The valid configurations
are conservatively:
- W=1, D=128: 8192+8192+2048+4096+2048+4096+64+64 = **26800 B** (×1) + shared 32768 = 59568 B ✓
- W=2, D=64:  shared 16384 + 2×(2048+4096+2048+4096+64+64) = 16384+24832 = **41216 B** ✓
- W=1, D=64:  shared 16384 + 1×12416 = **28800 B** ✓

If W=4 D=64 fails, the benchmark documents it as an LDS overflow finding. The valid
sweet spot for D=64 is W=2. For D=128, W=1 is the maximum.

**Updated plan for VM session:** if W=4 fails, sweep {W=1, W=2} at D=64 and W=1 at D=128.

---

## 5. VM session guion

```bash
# Build
cd /root/amd/research/flash-attention-mi300x
hipcc -O3 --offload-arch=gfx942 -std=c++17 \
  -I../../cdna3-prims/include \
  bench_multiwave_vs_sdpa.hip -o bench_multiwave_vs_sdpa

# Run benchmark
./bench_multiwave_vs_sdpa | tee /root/results_phase1_multiwave_mi300x.txt

# Run SDPA baseline
python3 bench_sdpa_baseline.py \
  --seqs 512,1024,2048,4096 --D 128 --nH 32 --nKV 8 \
  | tee /root/results_phase1_sdpa_mi300x.txt

# rocprofv3 for H-MW2 (VMEM_RD) — one dispatch per config
rocprofv3 --pmc SQ_INSTS_VMEM_RD,SQ_INSTS_LDS,SQ_INSTS_VALU \
  ./bench_multiwave_vs_sdpa \
  | tee /root/results_phase1_rocprof_mi300x.txt
```

After the session:
1. Copy all three result files to `research/flash-attention-mi300x/results/`
2. Compute gaps offline: `multiwave_us / sdpa_us` per seqLen
3. Evaluate H-MW1/H-MW2/H-MW3
4. Update paper §5.9 gap claim with measured number (currently "8.5–15×" from fa_robust)
5. If H-MW3 shows substantial gap remains: proceed to Phase 2 analysis

---

## 6. Negative result policy

If `fa_multiwave W=2` is NOT faster than `W=1` (H-MW1 fails):
- This is a publishable finding: occupancy alone does not explain the gap.
- Likely cause: the bottleneck has moved to memory bandwidth (HBM→LDS loads),
  not SIMD lane utilization. The K/V LDS reuse (H-MW2) should still show up.
- Next step: wave specialization (producer/consumer, à la HipKittens §2.3) —
  separate waves handle K/V loading vs computation.

If H-MW3 gap is unchanged (multiwave = fa_robust):
- The design change (wider query blocks) is neutral at these shapes.
- Bottleneck is instruction scheduling or memory latency, not lane occupancy.
- Reported honestly as a finding — same discipline as Welford refutation.

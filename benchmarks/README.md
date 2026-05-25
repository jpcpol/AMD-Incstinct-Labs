# Benchmarks

Reproducible performance comparisons for each research area. All benchmarks run on AMD Instinct MI300X unless noted.

## Hardware Reference

| Spec | MI300X |
| --- | --- |
| Architecture | CDNA3 (gfx942) |
| CUs | 304 |
| HBM3 | 192 GB, 5.3 TB/s |
| MFMA FP8 | 2,614.9 TFLOPS |
| MFMA BF16 | 1,307.4 TFLOPS |
| ROCm | 6.2+ |

## Results

| Research Area | Metric | Baseline | This Work | Delta |
| --- | --- | --- | --- | --- |
| wave-primitives | Reduce latency (μs) | hipCUB | TBD | TBD |
| flash-attention-mi300x | HBM utilization (%) | FA2 Triton ~55% | TBD | TBD |
| infinity-fabric-allreduce | AllReduce BW (GB/s) | RCCL ~316 | TBD | TBD |
| dme-abstraction | HBM→LDS BW (TB/s) | raw buffer load | TBD | TBD |
| mlir-mfma-tiling | GEMM TFLOPS | rocBLAS | TBD | TBD |

## Running Benchmarks

Each research area has a `benchmarks/` subdirectory with a self-contained script. From the repo root:

```bash
# wave-primitives
python research/wave-primitives/benchmarks/bench_reduce.py

# flash-attention-mi300x
python research/flash-attention-mi300x/benchmarks/bench_fa.py

# infinity-fabric-allreduce (requires 8× MI300X)
python research/infinity-fabric-allreduce/benchmarks/bench_allreduce.py

# dme-abstraction
./build/research/dme-abstraction/benchmarks/bench_bandwidth

# mlir-mfma-tiling
python research/mlir-mfma-tiling/benchmarks/bench_gemm.py
```

## Methodology

- **Warmup**: minimum 10 iterations before timing
- **Measurement**: median of 100 runs (or 30 for large tensors)
- **Variance**: report p50 and p99
- **Profiling**: `rocprofv3 --stats` for hardware counter validation
- **Reproducibility**: pin clock frequencies with `rocm-smi --setperflevel high`

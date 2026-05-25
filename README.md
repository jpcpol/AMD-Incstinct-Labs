# AMD Instinct Labs

Research and development targeting AMD Instinct MI300X/MI300A — filling gaps in the ROCm ecosystem and pushing CDNA3 hardware to its full potential.

## What This Is

AMD's MI300X has hardware advantages that software hasn't caught up to:
- **5.3 TB/s HBM3** vs 3.35 TB/s on H100, yet Flash Attention doesn't use DME for async prefetch
- **192 GB unified HBM3** (MI300A: shared CPU+GPU), yet no programming frameworks exploit it
- **7-link full-mesh Infinity Fabric**, yet RCCL uses Ring/Tree algorithms designed for PCIe ring topologies
- **Wave64 SIMT**, yet every CUDA kernel using warp-level primitives requires manual porting

This repo is structured research: each subdirectory under `research/` targets one confirmed gap, from identifying it against primary sources to benchmarking an implementation.

## Research Areas

| Area | Status | Summary |
| --- | --- | --- |
| [wave-primitives](research/wave-primitives/) | In progress | Header-only C++ library for wave32/wave64-agnostic warp operations |
| [flash-attention-mi300x](research/flash-attention-mi300x/) | In progress | Flash Attention with DME async prefetch + FA3-style pipelining |
| [infinity-fabric-allreduce](research/infinity-fabric-allreduce/) | Planned | Topology-aware AllReduce for MI300X full-mesh Infinity Fabric |
| [dme-abstraction](research/dme-abstraction/) | Planned | High-level C++ API over CDNA3 Data Movement Engine |
| [mlir-mfma-tiling](research/mlir-mfma-tiling/) | Planned | MLIR pass for automatic MFMA tile-size selection on MI300X |

## Hardware Target

- **Primary**: AMD Instinct MI300X (gfx942, 192 GB HBM3, 304 CUs, 5.3 TB/s)
- **Secondary**: AMD Instinct MI300A (gfx942, unified 128 GB HBM3+CPU DRAM)
- **ROCm version**: 6.2+
- **Compiler**: `hipcc` / `amdclang++` targeting `--offload-arch=gfx942`

## Build Requirements

```
ROCm 6.2+
CMake 3.21+
hipcc or amdclang++ with gfx942 support
Python 3.10+ (for Triton benchmarks)
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

To target a specific architecture:

```bash
cmake -B build -DGPU_TARGETS=gfx942
cmake --build build --parallel
```

## Benchmarks

Benchmark results are in [`benchmarks/`](benchmarks/). Each research area includes before/after comparisons against the relevant ROCm baseline (rocBLAS, MIOpen, RCCL).

## Gap Analysis

The full analysis of ROCm vs CUDA gaps with per-gap status verification against AMD primary sources is in [`docs/gap-analysis.md`](docs/gap-analysis.md).

## Contributing to AMD

This project follows the AMD contribution path for research work:

1. Develop + benchmark publicly (this repo)
2. Publish preprint on arXiv with results
3. Open GitHub Discussion in the relevant AMD repo
4. Submit PR to `develop` branch with metrics

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full process.

## License

MIT — see [LICENSE](LICENSE).

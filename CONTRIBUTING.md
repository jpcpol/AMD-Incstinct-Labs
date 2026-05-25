# Contributing

## Scope

Contributions should target confirmed gaps in the ROCm ecosystem with measurable impact on AMD Instinct MI300X/MI300A. See [`docs/gap-analysis.md`](docs/gap-analysis.md) for the current gap list and status.

## Before You Start

For new research areas or significant changes, open an issue first. Include:

- The gap being targeted (link to primary source confirming it's open)
- Proposed approach
- Expected measurable outcome (what metric improves, by how much)

For minor fixes or benchmarks: open a PR directly.

## Development Process

### 1. Fork and Branch

```bash
git checkout -b research/your-area-name
```

Branch names follow the pattern `research/<area>`, `fix/<description>`, or `bench/<area>`.

### 2. Implementation Standards

**C++/HIP code**:
- Target `gfx942` (MI300X/MI300A) as primary; note if code is architecture-generic
- Use `clang-format` with the project `.clang-format` (LLVM style)
- No `warpSize=32` assumptions — use `warpSize` or the `wave_primitives` library
- Test on ROCm 6.2+

**Python/Triton code**:
- Tested with `torch` ROCm build
- Include `requirements.txt` in the subdirectory

### 3. Benchmarks Are Required

Every implementation PR must include:

- Baseline: the current ROCm library equivalent (rocBLAS, MIOpen, RCCL, etc.)
- Proposed: your implementation
- Hardware: which GPU, ROCm version, driver version
- Metrics: TFLOPS, GB/s, latency — whichever is relevant
- Reproducible: a script in `benchmarks/` that generates the numbers

Format: add a row to `benchmarks/README.md`.

### 4. Pull Request

PRs should be < 1,000 lines of implementation code per iteration (AMD CK convention). For larger work, split into logical stages.

PR description must include:
- What gap this addresses
- Benchmark results (before/after table)
- Known limitations

### 5. Upstream Path

If the work is mature enough for AMD upstream:

1. Add arXiv citation once preprint is posted
2. Reference the AMD repo where the contribution belongs (MIOpen, composable_kernel, triton-lang, etc.)
3. Tag the PR with `upstream-candidate`

The upstream process is documented in the AMD CONTRIBUTING.md and the Composable Kernel Contributors Guide. AMD may pull and modify contributions internally before merge — this is expected behavior.

## Code of Conduct

Standard technical conduct: be direct, support claims with data, and keep discussions focused on the technical merits.

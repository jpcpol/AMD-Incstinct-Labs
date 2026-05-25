# infinity-fabric-allreduce

Topology-aware AllReduce for AMD MI300X 8-GPU clusters with full-mesh Infinity Fabric.

## Problem

RCCL (AMD's NCCL fork) uses Ring and Tree algorithms inherited from PCIe-based NVIDIA topologies. The MI300X cluster has a fundamentally different topology:

- Each GPU has **7 bidirectional xGMI links** — full-mesh connectivity
- Theoretical aggregate bandwidth: **336 GB/s** (8 × 42 GB/s)
- Practical with RCCL Ring/Tree: **310–330 GB/s**

Two bottlenecks confirmed by arXiv 2410.00801:

1. **SDMA engines**: tuned for PCIe-4.0, underutilize Infinity Fabric bandwidth. Fix: `HSA_ENABLE_PEER_SDMA=0`
2. **Slowest-link bottleneck**: Ring forces all communication through every link sequentially; the weakest pair (45.21 GB/s in practice) caps throughput across all pairs

## Approach

### Full-Mesh AllReduce Algorithm

Instead of a ring that serializes through the slowest link, use all 7 links simultaneously:

```
Phase 1 — Reduce:
    Each GPU pair reduces its local contribution in parallel
    Fast pairs finish first; slow pair (GPU0↔GPU7) is not on the critical path

Phase 2 — Broadcast:
    Root GPU broadcasts result on all 7 links simultaneously
    Each receiving GPU retransmits to its neighbors
```

This eliminates the serialization bottleneck and matches bandwidth to the actual link topology.

### SDMA Bypass

For GPU-GPU transfers, disable SDMA routing through PCIe-tuned engines:

```cpp
setenv("HSA_ENABLE_PEER_SDMA", "0", 1);
// Now uses direct GPU-initiated DMA over xGMI
```

## Files

```
include/
    ifa_allreduce.hpp          # AllReduce API
src/
    topology.hip               # xGMI link detection and ranking
    allreduce_fullmesh.hip     # Full-mesh AllReduce kernel
benchmarks/
    bench_allreduce.py         # vs RCCL baseline on 8× MI300X
```

## Status

- [ ] xGMI topology detection (link BW measurement per pair)
- [ ] Full-mesh AllReduce prototype
- [ ] Benchmark vs RCCL on 8× MI300X (AllReduce across various tensor sizes)
- [ ] Upstream: proposal to RCCL maintainers

## Gap Reference

[Gap 5 — RCCL / Infinity Fabric](../../docs/gap-analysis.md#5-rccl--infinity-fabric--sdma-bottleneck)

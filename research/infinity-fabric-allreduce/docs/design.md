# Full-Mesh AllReduce for MI300X — Design (cold draft)

> **Status:** algorithm design + bandwidth analysis only. Validation requires an
> 8× MI300X node (multi-GPU, ~8× hourly cost) — deferred to the end of the credit
> budget. Everything here is analytical and written cold.

## The gap

RCCL (AMD's NCCL port) selects between **Ring** and **Tree** AllReduce. Both were
designed for topologies where each GPU has *few* links (PCIe, NVLink chains). An
8× MI300X node is a **full mesh**: every GPU has a direct xGMI/Infinity Fabric
link to every other GPU (7 links per GPU). Ring uses only 2 of those 7 links at
a time; the other 5 sit idle. This is bandwidth left on the table.

```
Ring AllReduce on a full mesh:        Full mesh actually available:
  GPU0 → GPU1 → GPU2 → ... → GPU7      every GPU ↔ every GPU
  (each GPU uses 2 of 7 links)         (7 links per GPU, all usable)
```

## Bandwidth analysis (the core argument)

Let:
- `N` = 8 GPUs
- `B` = per-link xGMI bandwidth (bidirectional)
- `D` = data size (bytes) to all-reduce

**Ring AllReduce** moves `2D(N-1)/N` bytes per GPU over **one link's worth** of
bandwidth at a time (the ring is serialized across links):

```
T_ring ≈ 2·D·(N-1)/N / B
```

**Full-mesh direct AllReduce** (each GPU sends its shard to all others
simultaneously over its 7 links): a reduce-scatter + all-gather where each
phase uses all `N-1` links in parallel:

```
T_mesh ≈ 2·D·(N-1)/N / ((N-1)·B)  =  2·D/N / B
```

**Theoretical speedup: (N-1)× = 7× on an 8-GPU node** for the bandwidth-bound
regime. Real-world will be less (sync overhead, link contention, non-ideal
scheduling) — the honest target is "uses substantially more than 2 of 7 links",
not a literal 7×.

## The algorithm — mesh reduce-scatter + mesh all-gather

Classic bandwidth-optimal AllReduce is reduce-scatter then all-gather. On a full
mesh both phases parallelize across all links:

**Phase 1 — reduce-scatter (mesh):**
Partition the buffer into N shards. GPU `i` is responsible for reducing shard `i`.
Every GPU sends its copy of shard `i` to GPU `i` — all sends happen in parallel
over distinct links. GPU `i` reduces the N contributions to shard `i`.

```
shard 0 → GPU0    (GPUs 1..7 each send their shard-0 to GPU0, in parallel)
shard 1 → GPU1
...
shard 7 → GPU7
```

**Phase 2 — all-gather (mesh):**
Each GPU `i` now holds the fully-reduced shard `i`. It broadcasts that shard to
all other GPUs — again all sends in parallel over distinct links.

Both phases are `O(D/N / B)` instead of Ring's `O(D·(N-1)/N / B)`.

## Implementation path

1. **Topology probe** — query xGMI link map (`rocm-smi --shownodesbw` /
   `hipDeviceGetP2PAttribute`). Confirm full-mesh and per-link bandwidth.
   **→ WRITTEN (cold): [`tests/probe_xgmi_topology.hip`](../tests/probe_xgmi_topology.hip).**
   It enumerates the P2P matrix, perf ranks, and — critically — measures whether
   GPU0 broadcasting to k peers on k streams scales with k (concurrent links) or
   stays flat (runtime serialization). This is the first thing to run on a
   multi-GPU node; it gates the entire design (see Open risk #1).
2. **Point-to-point primitive** — `hipMemcpyPeerAsync` or direct xGMI store on
   `N-1` streams, one per peer link, to drive all links concurrently.
3. **Mesh reduce-scatter** — the N-way parallel shard exchange + local reduce.
4. **Mesh all-gather** — the N-way parallel shard broadcast.
5. **Benchmark vs RCCL** — `all_reduce_perf` from rccl-tests as the baseline.

### Execution plan for the (expensive) multi-GPU run

The 8× node is ~8× hourly cost and billed per started hour — so the run must be
one tight session with everything pre-staged:

1. Boot, `git pull`, compile `probe_xgmi_topology` + the AllReduce kernel.
2. Run the probe FIRST. **Decision gate:**
   - per-link BW constant as k grows → links concurrent → proceed to mesh AllReduce.
   - aggregate BW flat → serialized → abort the run, pivot the design in cold,
     do NOT burn node-hours on a kernel that can't win.
3. If gate passes: run mesh reduce-scatter + all-gather, benchmark vs `rccl-tests`
   `all_reduce_perf` across message sizes (1 MB … 1 GB).
4. Capture results to `results/`, power off immediately.

## Why this is deferred to last

- Needs an **8× MI300X node** — the credit covers single-GPU comfortably but a
  full node is ~8× the hourly rate. Run it once, near the end, when the design
  is fully baked and the run is short.
- It is **independent** of the DPP / DME work — no shared code — so it loses
  nothing by waiting. The other three areas compound (DPP → DME → FA); this one
  stands alone.

## Open risks

- **xGMI may not expose all 7 links to user-driven P2P.** RCCL uses internal
  paths; whether `hipMemcpyPeerAsync` on 7 streams actually saturates 7 distinct
  links (vs being serialized by the runtime) is the first thing the topology
  probe must answer. If the runtime serializes, the mesh advantage evaporates.
- **The local reduce competes with communication.** Phase 1's reduce step uses
  VALU; if it doesn't overlap with the sends, it adds to the critical path.
- **Sync overhead at 8 GPUs** may dominate for small `D`. The mesh win is in the
  bandwidth-bound (large `D`) regime — exactly where training gradients live, so
  the relevant regime is favorable, but small-message latency may favor Tree.

## Relationship to the rest of the project

Standalone. Shares the methodology (probe hardware empirically before building)
but no code. It is the "HPC-flavored" result — relevant to distributed training
(DeepSpeed/Megatron on MI300X clusters) rather than single-GPU inference.

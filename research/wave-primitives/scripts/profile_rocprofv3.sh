#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# profile_rocprofv3.sh — Hardware counter profiling for wave-primitives
#
# Runs bench_reduce under rocprofv3 to capture:
#   - Kernel timing statistics (--stats)
#   - Hardware counters: VALU utilization, memory bandwidth, L2 hit rate
#
# Run from research/wave-primitives/:
#   ./scripts/profile_rocprofv3.sh [binary]
#
# Default binary: ./local_build/bench_reduce
# ---------------------------------------------------------------------------

set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BINARY="${1:-./local_build/bench_reduce}"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
OUT_DIR="./results/profile_${TIMESTAMP}"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found: $BINARY"
    echo "Run ./scripts/run_all.sh first to compile."
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " rocprofv3 profiling: $(basename $BINARY)"
echo " Output : $OUT_DIR"
echo " $(date)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Pass 1: kernel timing statistics (always works, no counter overhead)
# ---------------------------------------------------------------------------
echo ">>> Pass 1: Kernel timing statistics (--stats)"
echo ""

rocprofv3 \
    --stats \
    --output-directory "$OUT_DIR" \
    --output-file "stats" \
    -- "$BINARY"

echo ""
echo "Timing stats written to: $OUT_DIR/stats*.csv"
echo ""

# ---------------------------------------------------------------------------
# Pass 2: hardware counters
#
# Counter names for gfx942 (MI300X). These are the most relevant for
# wave-level reduction analysis:
#
#   SQ_BUSY_CYCLES         — cycles the SQ was processing waves
#   SQ_WAVES               — total waves dispatched
#   VALU_INSTS_VFMA32      — FP32 VFMA (fused mul-add) instructions
#   VALU_INSTS_VALU32      — general FP32 VALU instructions
#   LDS_BANK_CONFLICT      — LDS bank conflicts (0 expected for our kernels)
#   TCP_TCC_READ_REQ_sum   — L1 to L2 read requests
#   TCC_HIT_sum            — L2 cache hits
#   TCC_MISS_sum           — L2 cache misses
# ---------------------------------------------------------------------------

echo ">>> Pass 2: Hardware counters (PMC)"
echo ""

COUNTERS=(
    "SQ_BUSY_CYCLES"
    "SQ_WAVES"
    "VALU_INSTS_VFMA32"
    "VALU_INSTS_VALU32"
    "LDS_BANK_CONFLICT"
    "TCP_TCC_READ_REQ_sum"
    "TCC_HIT_sum"
    "TCC_MISS_sum"
)

# Build --pmc list
PMC_ARGS=""
for c in "${COUNTERS[@]}"; do
    PMC_ARGS="$PMC_ARGS --pmc $c"
done

rocprofv3 \
    $PMC_ARGS \
    --output-directory "$OUT_DIR" \
    --output-file "counters" \
    -- "$BINARY" 2>&1 || {
        echo ""
        echo "NOTE: PMC counter collection failed. This can happen if:"
        echo "  - The counter names differ on this ROCm version"
        echo "  - Another process is using the GPU profiler"
        echo "  - Running inside a container without perf_event access"
        echo ""
        echo "Timing stats (Pass 1) are still valid."
        echo "To list available counters: rocprofv3 --list-counters"
    }

echo ""

# ---------------------------------------------------------------------------
# Parse and display key metrics
# ---------------------------------------------------------------------------
echo ">>> Summary"
echo ""

# Display timing stats CSV if it exists
STATS_CSV=$(find "$OUT_DIR" -name "stats*.csv" | head -1)
if [ -f "$STATS_CSV" ]; then
    echo "Kernel timing (from $STATS_CSV):"
    echo ""
    # Print header + data rows (skip empty lines)
    awk -F',' 'NR<=5 && NF>0 { printf "  %s\n", $0 }' "$STATS_CSV"
    echo ""
fi

# Display counters CSV if it exists
COUNTERS_CSV=$(find "$OUT_DIR" -name "counters*.csv" | head -1)
if [ -f "$COUNTERS_CSV" ]; then
    echo "Hardware counters (from $COUNTERS_CSV):"
    echo ""
    awk -F',' 'NR<=10 && NF>0 { printf "  %s\n", $0 }' "$COUNTERS_CSV"
    echo ""

    # Compute L2 hit rate if columns present
    echo "L2 hit rate = TCC_HIT / (TCC_HIT + TCC_MISS)"
    python3 - "$COUNTERS_CSV" <<'PYEOF' 2>/dev/null || echo "  (python3 not available for L2 calculation)"
import csv, sys
with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
for row in rows:
    hit  = float(row.get('TCC_HIT_sum', 0) or 0)
    miss = float(row.get('TCC_MISS_sum', 0) or 0)
    total = hit + miss
    if total > 0:
        print(f"  Kernel: {row.get('Kernel_Name','?')}")
        print(f"    L2 hits:   {hit:.0f}")
        print(f"    L2 misses: {miss:.0f}")
        print(f"    Hit rate:  {100*hit/total:.1f}%")
PYEOF
fi

echo ""
echo "============================================================"
echo " All profiling output saved to: $OUT_DIR"
echo " To list available counters:    rocprofv3 --list-counters"
echo "============================================================"

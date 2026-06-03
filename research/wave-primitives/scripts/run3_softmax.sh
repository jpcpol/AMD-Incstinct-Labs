#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run3_softmax.sh — DPP reductions inside a real softmax kernel.
# Runs two regimes to separate the end-to-end view from the isolated effect:
#   wide   (512 cols) — realistic, likely HBM-bound
#   narrow ( 64 cols) — reduction-dominated, isolates DPP
# Plus rocprofv3 LDS counters on the wide DPP kernel.
#
# From research/wave-primitives/:  ./scripts/run3_softmax.sh
# ---------------------------------------------------------------------------
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/local_build"; RESULTS="$ROOT/results"; ROCM="${ROCM_PATH:-/opt/rocm}"
ARCH=$(rocminfo 2>/dev/null | grep -m1 -oP "gfx\d+" || echo gfx942)
TS=$(date +%Y-%m-%d_%H-%M); OUT="$RESULTS/${TS}_softmax_${ARCH}.txt"
mkdir -p "$BUILD" "$RESULTS"
FLAGS="-O3 --offload-arch=$ARCH -I$ROOT/include -I$ROOT/benchmarks -I$ROCM/include -std=c++17"

exec > >(tee "$OUT") 2>&1
echo "=== Run 3: softmax with DPP reductions — $ARCH — $(date) ==="

for EPL in 8 1; do
    W=$((64*EPL))
    echo ""
    echo ">>> Regime: ${W} cols (ELEMS_PER_LANE=$EPL)"
    "$ROCM/bin/hipcc" $FLAGS -DELEMS_PER_LANE=$EPL "$ROOT/benchmarks/bench_softmax.hip" \
        -o "$BUILD/bench_softmax_$EPL" || { echo "COMPILE FAILED (EPL=$EPL)"; continue; }
    "$BUILD/bench_softmax_$EPL"
done

echo ""
echo ">>> rocprofv3: LDS in the wide (512-col) softmax kernels"
printf "pmc: SQ_INSTS_LDS SQ_INSTS_VALU\n" > /tmp/sm.txt
rocprofv3 -i /tmp/sm.txt --output-format csv -d "$RESULTS/prof_softmax" -o sm -- "$BUILD/bench_softmax_8" 2>&1 | tail -2
CSV=$(find "$RESULTS/prof_softmax" -name "*counter_collection*.csv" 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    # Kernel_Name is column 9 (may contain commas → reconstruct by backend tag).
    # Counter_Name and Counter_Value are the 3rd/2nd fields from the end.
    awk -F',' 'NR>1 {
        v=$(NF-2); c=$(NF-3); b="other";
        if ($0 ~ /DPP/)      b="softmax_DPP";
        else if ($0 ~ /PORTABLE/) b="softmax_portable";
        else if ($0 ~ /HIPCUB/)   b="softmax_hipCUB";
        if (b!="other") { key=b" "c; s[key]+=v; n[key]++ }
    } END { for (x in s) printf "  %-26s avg %.0f\n", x, s[x]/n[x] }' "$CSV" | sort
fi
echo ""
echo "=== Done. Results: $OUT ==="

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run3_layernorm.sh — LayerNorm kernel with DPP reductions (first "real kernel")
#
# Runs two regimes:
#   narrow (64 cols, 1 elem/lane)  — reduction-dominated, isolates DPP effect
#   wide  (512 cols, 8 elems/lane) — realistic, may be memory-bound
#
# Plus rocprofv3 LDS counters on the narrow DPP kernel (expect 0 LDS).
# From research/wave-primitives/:  ./scripts/run3_layernorm.sh
# ---------------------------------------------------------------------------
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/local_build"; RESULTS="$ROOT/results"; ROCM="${ROCM_PATH:-/opt/rocm}"
ARCH=$(rocminfo 2>/dev/null | grep -m1 -oP "gfx\d+" || echo gfx942)
TS=$(date +%Y-%m-%d_%H-%M); OUT="$RESULTS/${TS}_layernorm_${ARCH}.txt"
mkdir -p "$BUILD" "$RESULTS"
FLAGS="-O3 --offload-arch=$ARCH -I$ROOT/include -I$ROOT/benchmarks -I$ROCM/include -std=c++17"

exec > >(tee "$OUT") 2>&1
echo "=== Run 3: LayerNorm with DPP reductions — $ARCH — $(date) ==="

for EPL in 1 8; do
    W=$((64*EPL))
    echo ""
    echo ">>> Regime: ${W} cols (ELEMS_PER_LANE=$EPL)"
    "$ROCM/bin/hipcc" $FLAGS -DELEMS_PER_LANE=$EPL \
        "$ROOT/benchmarks/bench_layernorm.hip" \
        -o "$BUILD/bench_layernorm_$EPL" \
        2>&1 | grep -v "^$" | tail -5 \
        || { echo "COMPILE FAILED (EPL=$EPL)"; continue; }
    "$BUILD/bench_layernorm_$EPL"
done

echo ""
echo ">>> rocprofv3: LDS counters on narrow (64-col) LayerNorm"
printf "pmc: SQ_INSTS_LDS SQ_INSTS_VALU\n" > /tmp/ln.txt
rocprofv3 -i /tmp/ln.txt --output-format csv \
    -d "$RESULTS/prof_layernorm" -o ln \
    -- "$BUILD/bench_layernorm_1" 2>&1 | tail -3
CSV=$(find "$RESULTS/prof_layernorm" -name "*counter_collection*.csv" 2>/dev/null | head -1)
if [ -n "$CSV" ]; then
    awk -F',' 'NR>1 {
        v=$(NF-2); c=$(NF-3); b="other";
        if ($0 ~ /DPP/)      b="layernorm_DPP";
        else if ($0 ~ /PORTABLE/) b="layernorm_portable";
        else if ($0 ~ /HIPCUB/)   b="layernorm_hipCUB";
        if (b != "other") { key=b" "c; s[key]+=v; n[key]++ }
    } END { for (x in s) printf "  %-30s avg %.0f\n", x, s[x]/n[x] }' "$CSV" | sort
fi

echo ""
echo "=== Done. Results: $OUT ==="

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run2_dpp_types.sh — Run 2 entry point: validate + benchmark generalized DPP
# reductions (sum/max/min × float/double/int32/int64) on MI300X.
#
# Assumes the repo is cloned and the updated wave_reduce_dpp.hpp +
# bench_reduce_dpp_types.hip are present (pull latest or paste via heredoc).
#
# Run from research/wave-primitives/:
#   chmod +x scripts/run2_dpp_types.sh && ./scripts/run2_dpp_types.sh
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/local_build"
RESULTS="$ROOT/results"
ROCM="${ROCM_PATH:-/opt/rocm}"
ARCH=$(rocminfo 2>/dev/null | grep -m1 -oP "gfx\d+" || echo gfx942)
TS=$(date +%Y-%m-%d_%H-%M)
OUT="$RESULTS/${TS}_dpp_types_${ARCH}.txt"
mkdir -p "$BUILD" "$RESULTS"

FLAGS="-O3 --offload-arch=$ARCH -I$ROOT/include -I$ROOT/benchmarks -I$ROCM/include -std=c++17"

exec > >(tee "$OUT") 2>&1
echo "=== Run 2: generalized DPP reductions — $ARCH — $(date) ==="

echo ">>> Compiling bench_reduce_dpp_types..."
if ! "$ROCM/bin/hipcc" $FLAGS "$ROOT/benchmarks/bench_reduce_dpp_types.hip" -o "$BUILD/bench_reduce_dpp_types"; then
    echo ""
    echo "!!! COMPILE FAILED."
    echo "    Most likely cause: __builtin_amdgcn_update_dpp does not accept 64-bit"
    echo "    (double/int64) directly on this toolchain. Fallback: edit dpp_move to"
    echo "    split into two 32-bit update_dpp calls on the hi/lo halves, OR limit"
    echo "    run_type<> to float/int32 in the benchmark. See header notes."
    exit 1
fi

echo ">>> Running..."
"$BUILD/bench_reduce_dpp_types"

echo ""
echo ">>> ISA check: confirm DPP usage per type (expect v_*_dpp, zero ds_bpermute)"
"$ROCM/bin/hipcc" $FLAGS --save-temps -c "$ROOT/benchmarks/bench_reduce_dpp_types.hip" -o /tmp/t.o 2>/dev/null
S=$(ls "$ROOT"/bench_reduce_dpp_types-hip-amdgcn-amd-amdhsa-${ARCH}.s 2>/dev/null | head -1)
if [ -n "$S" ] && [ -f "$S" ]; then
    echo "  DPP / bpermute instruction counts across all kernels:"
    grep -oiE "v_add_f32_dpp|v_max_f32_dpp|v_min_f32_dpp|v_add_f64|_dpp|ds_bpermute_b32" "$S" \
        | sort | uniq -c | sort -rn
else
    echo "  (.s not found — skip ISA check)"
fi

echo ""
echo "=== Done. Results: $OUT ==="

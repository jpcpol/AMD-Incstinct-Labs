#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_fa_dpp.sh — build and run fa_dpp at Bc=64/128/256
#
# What it measures:
#   1. Correctness vs CPU reference at each Bc
#   2. Speedup vs fa_naive Bc=64 (in-binary baseline)
#   3. rocprofv3 SQ_INSTS_LDS: should drop vs naive at Bc=64;
#      should grow modestly for Bc=128/256 (inter-wave LDS only)
#
# Expected outcome if hypothesis holds:
#   Bc=64 : dpp slightly faster than naive (DPP reduction latency wins)
#   Bc=128: bigger speedup (LDS relaxation → fewer iterations at better MFMA use)
#   Bc=256: plateau or slight regression (LDS/occupancy pressure)
#
# Usage: run from amd-instinct-labs/ root
#   bash research/flash-attention-mi300x/scripts/run_fa_dpp.sh
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FA_DIR="$ROOT/research/flash-attention-mi300x"
WAVE_INC="$ROOT/research/wave-primitives/include"
BENCH_INC="$ROOT/research/wave-primitives/benchmarks"

echo "=== Flash Attention DPP (step 2) — Bc=64/128/256 ==="
echo "Root : $ROOT"
echo ""

mkdir -p "$FA_DIR/build"

for BC in 64 128 256; do
    BIN="$FA_DIR/build/fa_dpp_bc${BC}"
    echo "--- Building fa_dpp Bc=${BC} ---"
    hipcc -O3 --offload-arch=gfx942 \
        -I"$WAVE_INC" \
        -I"$BENCH_INC" \
        -DFA_HEAD_DIM=64 \
        -DFA_BC=${BC} \
        -std=c++17 \
        "$FA_DIR/kernels/fa_dpp.hip" \
        -o "$BIN"
    echo "  Build OK: $BIN"

    echo "--- Running Bc=${BC} ---"
    "$BIN" 2>&1 | tee "$FA_DIR/build/result_dpp_bc${BC}.txt"
    echo ""
done

echo "--- rocprofv3: LDS counters (Bc=64 DPP vs naive comparison) ---"
if command -v rocprofv3 &>/dev/null; then
    # Profile dpp Bc=64 (expect near-zero LDS for reductions)
    echo "  Profiling fa_dpp Bc=64 ..."
    rocprofv3 \
        --pmc SQ_INSTS_LDS SQ_INSTS_VALU SQ_LDS_IDX_ACTIVE \
        "$FA_DIR/build/fa_dpp_bc64" \
        2>&1 | tee "$FA_DIR/build/rocprof_dpp_bc64.txt"

    echo ""
    echo "  Compare with naive (run run_fa_naive.sh first to get rocprof_naive_lds.txt)"
    if [ -f "$FA_DIR/build/rocprof_naive_lds.txt" ]; then
        echo "  Naive SQ_INSTS_LDS:"
        grep -i "SQ_INSTS_LDS" "$FA_DIR/build/rocprof_naive_lds.txt" || true
        echo "  DPP Bc=64 SQ_INSTS_LDS:"
        grep -i "SQ_INSTS_LDS" "$FA_DIR/build/rocprof_dpp_bc64.txt" || true
    fi
else
    echo "  rocprofv3 not found — skip profiling"
fi

echo ""
echo "=== run_fa_dpp.sh done ==="
echo ""
echo "Key numbers to record:"
echo "  Speedup dpp Bc=64  vs naive Bc=64: see result_dpp_bc64.txt"
echo "  Speedup dpp Bc=128 vs naive Bc=64: see result_dpp_bc128.txt"
echo "  Speedup dpp Bc=256 vs naive Bc=64: see result_dpp_bc256.txt"
echo "  LDS drop: rocprof_dpp_bc64.txt vs rocprof_naive_lds.txt"

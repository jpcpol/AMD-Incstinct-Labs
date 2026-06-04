#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_fa_naive.sh — build and run Flash Attention naive kernel (step 1)
#
# What it tests:
#   1. Correctness vs CPU reference (max_rel_err < 1e-3 required)
#   2. Throughput baseline (TFLOPS, bandwidth) — sets the bar for step 2+
#   3. rocprofv3 counters: LDS instructions should be ZERO for the
#      wave reductions once we swap to DPP (step 2 diff point)
#
# Usage: run from amd-instinct-labs/ root
#   bash research/flash-attention-mi300x/scripts/run_fa_naive.sh
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FA_DIR="$ROOT/research/flash-attention-mi300x"
WAVE_INC="$ROOT/research/wave-primitives/include"
BENCH_INC="$ROOT/research/wave-primitives/benchmarks"

echo "=== Flash Attention naive (step 1) ==="
echo "Root : $ROOT"
echo "Arch : gfx942"
echo ""

BIN="$FA_DIR/build/fa_naive"
mkdir -p "$FA_DIR/build"

echo "--- Building fa_naive ---"
hipcc -O3 --offload-arch=gfx942 \
    -I"$WAVE_INC" \
    -I"$BENCH_INC" \
    -DFA_HEAD_DIM=64 \
    -DFA_BC=64 \
    "$FA_DIR/kernels/fa_naive.hip" \
    -o "$BIN"
echo "Build OK: $BIN"
echo ""

echo "--- Correctness + benchmark (seq=512, d=64, 8 heads × 2 batch) ---"
"$BIN"

echo ""
echo "--- rocprofv3: LDS counter baseline (for step 2 comparison) ---"
echo "    SQ_INSTS_LDS will be non-zero here (portable shfl_down → ds_bpermute)"
echo "    After step 2 (DPP), we expect this to drop toward 0 for the reductions."
echo ""

if command -v rocprofv3 &>/dev/null; then
    rocprofv3 \
        --pmc SQ_INSTS_LDS SQ_INSTS_VALU SQ_LDS_IDX_ACTIVE \
        "$BIN" 2>&1 | tee "$FA_DIR/build/rocprof_naive_lds.txt"
    echo "Profile saved: $FA_DIR/build/rocprof_naive_lds.txt"
else
    echo "rocprofv3 not found — skip profiling (run on MI300X to get counters)"
fi

echo ""
echo "=== run_fa_naive.sh done ==="

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh — Compile and run the full wave-primitives test + benchmark suite
#
# Run from the research/wave-primitives/ directory:
#   cd research/wave-primitives
#   chmod +x scripts/run_all.sh
#   ./scripts/run_all.sh
#
# Output: results/YYYY-MM-DD_HH-MM_<arch>.txt
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT/local_build"
RESULTS_DIR="$ROOT/results"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="$ROCM_PATH/bin/hipcc"
INCLUDE_DIR="$ROOT/include"
BENCH_DIR="$ROOT/benchmarks"
TEST_DIR="$ROOT/tests"

# ---------------------------------------------------------------------------
# Detect GPU arch
# ---------------------------------------------------------------------------
GPU_ARCH=$(rocminfo 2>/dev/null | grep -m1 "gfx" | grep -oP "gfx\d+" || echo "gfx942")
TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
RESULT_FILE="$RESULTS_DIR/${TIMESTAMP}_${GPU_ARCH}.txt"

mkdir -p "$BUILD_DIR"
mkdir -p "$RESULTS_DIR"

HIPCC_FLAGS="-O3 --offload-arch=$GPU_ARCH -I$INCLUDE_DIR -I$ROCM_PATH/include -std=c++17"

# ---------------------------------------------------------------------------
# Logging: write to stdout AND result file
# ---------------------------------------------------------------------------
exec > >(tee "$RESULT_FILE") 2>&1

echo "============================================================"
echo " AMD Instinct Labs — wave-primitives test suite"
echo " $(date)"
echo " GPU arch : $GPU_ARCH"
echo " ROCm     : $(cat $ROCM_PATH/.info/version 2>/dev/null || hipcc --version | head -1)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
echo ">>> Compiling..."

echo "  test_correctness..."
$HIPCC $HIPCC_FLAGS \
    "$TEST_DIR/test_correctness.hip" \
    -o "$BUILD_DIR/test_correctness"

echo "  bench_reduce..."
$HIPCC $HIPCC_FLAGS \
    "$BENCH_DIR/bench_reduce.hip" \
    -o "$BUILD_DIR/bench_reduce"

echo "  bench_scan..."
$HIPCC $HIPCC_FLAGS \
    "$BENCH_DIR/bench_scan.hip" \
    -o "$BUILD_DIR/bench_scan"

echo "  bench_all_types..."
$HIPCC $HIPCC_FLAGS \
    "$BENCH_DIR/bench_all_types.hip" \
    -o "$BUILD_DIR/bench_all_types"

echo "  bench_block_reduce..."
$HIPCC $HIPCC_FLAGS \
    "$BENCH_DIR/bench_block_reduce.hip" \
    -o "$BUILD_DIR/bench_block_reduce"

echo ""
echo ">>> All compiled successfully."
echo ""

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run_step() {
    local label="$1"
    local binary="$2"
    echo "------------------------------------------------------------"
    echo " $label"
    echo "------------------------------------------------------------"
    "$binary"
    echo ""
}

run_step "CORRECTNESS TESTS" "$BUILD_DIR/test_correctness"
run_step "BENCHMARK: reduce_sum vs hipCUB WarpReduce" "$BUILD_DIR/bench_reduce"
run_step "BENCHMARK: scan_inclusive_sum vs hipCUB WarpScan" "$BUILD_DIR/bench_scan"
run_step "BENCHMARK: reduce_sum all types + accuracy" "$BUILD_DIR/bench_all_types"
run_step "BENCHMARK: block reduce vs hipCUB BlockReduce" "$BUILD_DIR/bench_block_reduce"

echo "============================================================"
echo " Done. Results saved to:"
echo "   $RESULT_FILE"
echo "============================================================"

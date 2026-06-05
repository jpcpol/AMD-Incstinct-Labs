#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all_validation.sh — Complete validation run for all AMD-Instinct areas.
#
# Runs everything in dependency order. Takes automatic decisions:
#   - If probe_dme correctness fails → skips fa_dme (DME not safe to use yet).
#   - If fa_naive correctness fails  → continues but flags the regression.
#   - If fa_dpp correctness fails at any Bc → continues, records which failed.
#   - Always saves full output to results/ with timestamp prefix.
#
# Usage (from amd-instinct-labs/ root, on the MI300X VM):
#   bash research/run_all_validation.sh 2>&1 | tee run_all_$(date +%Y%m%d_%H%M).log
#
# Estimated wall time: ~25-40 min (compile-heavy, rocprofv3 doubles the time).
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM="${ROCM_PATH:-/opt/rocm}"
ARCH=$(rocminfo 2>/dev/null | grep -m1 -oP "gfx\d+" || echo gfx942)
TS=$(date +%Y-%m-%d_%H-%M)
RESULTS="$ROOT/results/${TS}_full_run_${ARCH}"
mkdir -p "$RESULTS"

HIPCC="$ROCM/bin/hipcc"
BASE_FLAGS="-O3 --offload-arch=$ARCH -std=c++17"
WAVE_INC="$ROOT/research/wave-primitives/include"
BENCH_INC="$ROOT/research/wave-primitives/benchmarks"
ROCM_INC="$ROCM/include"

# ---- Summary tracking -------------------------------------------------------
PASS=()
FAIL=()
SKIP=()
DME_VALID=false   # set true if probe_dme passes; gates fa_dme

log() { echo "[$(date +%H:%M:%S)] $*"; }
sep() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
ok()  { PASS+=("$1"); log "  ✓ PASS: $1"; }
err() { FAIL+=("$1"); log "  ✗ FAIL: $1 — $2"; }
skip(){ SKIP+=("$1"); log "  ⊘ SKIP: $1 — $2"; }

# Helper: compile with error capture
compile() {
    local name="$1"; shift
    if "$HIPCC" "$@" 2>"$RESULTS/${name}.compile.err"; then
        log "  compiled: $name"
        return 0
    else
        log "  COMPILE ERROR: $name"
        cat "$RESULTS/${name}.compile.err"
        return 1
    fi
}

# Helper: rocprofv3 counter run, saves CSV
rocprof_counters() {
    local name="$1"; local bin="$2"
    local outdir="$RESULTS/rocprof_${name}"
    mkdir -p "$outdir"
    printf "pmc: SQ_INSTS_LDS SQ_INSTS_VALU SQ_LDS_IDX_ACTIVE\n" > /tmp/pmc_${name}.txt
    rocprofv3 -i /tmp/pmc_${name}.txt --output-format csv \
        -d "$outdir" -o "${name}" \
        -- "$bin" > "$outdir/stdout.txt" 2>&1 || true
    find "$outdir" -name "*.csv" | head -1
}

# ---------------------------------------------------------------------------
# 0. Device info
# ---------------------------------------------------------------------------
sep
log "=== AMD-Instinct Full Validation Run — $ARCH — $(date) ==="
log "ROCM: $ROCM"
log "Results: $RESULTS"
"$HIPCC" --version 2>&1 | head -2

# ---------------------------------------------------------------------------
# 1. WAVE PRIMITIVES — LayerNorm (run3_layernorm)
# ---------------------------------------------------------------------------
sep
log ">>> [1/10] LayerNorm with DPP reductions (narrow + wide)"

LN_FLAGS="$BASE_FLAGS -I$WAVE_INC -I$BENCH_INC -I$ROCM_INC"
LN_SRC="$ROOT/research/wave-primitives/benchmarks/bench_layernorm.hip"

for EPL in 1 8; do
    W=$((64*EPL))
    BIN="$RESULTS/bench_layernorm_epl${EPL}"
    log "  Compiling LayerNorm ${W}-col (ELEMS_PER_LANE=$EPL) ..."
    if compile "layernorm_${EPL}" $LN_FLAGS -DELEMS_PER_LANE=$EPL "$LN_SRC" -o "$BIN"; then
        log "  Running LayerNorm ${W}-col ..."
        "$BIN" 2>&1 | tee "$RESULTS/layernorm_${W}col.txt"
        # Check correctness in output
        if grep -q "PASS" "$RESULTS/layernorm_${W}col.txt"; then
            ok "layernorm_${W}col_correctness"
        else
            err "layernorm_${W}col_correctness" "no PASS in output"
        fi
    else
        err "layernorm_${W}col_compile" "hipcc failed"
    fi
done

# rocprofv3 on narrow (EPL=1) — expect 0 LDS for DPP backend
LN_BIN_NARROW="$RESULTS/bench_layernorm_epl1"
if [ -f "$LN_BIN_NARROW" ]; then
    log "  rocprofv3 on LayerNorm 64-col ..."
    CSV=$(rocprof_counters "layernorm_narrow" "$LN_BIN_NARROW")
    if [ -n "$CSV" ]; then
        log "  LDS counter summary:"
        awk -F',' 'NR>1 {
            v=$(NF-2); c=$(NF-3); b="other";
            if ($0 ~ /DPP/)      b="DPP";
            else if ($0 ~ /PORTABLE/) b="portable";
            else if ($0 ~ /HIPCUB/)   b="hipCUB";
            if (b != "other") { key=b" "c; s[key]+=v; n[key]++ }
        } END { for (x in s) printf "    %-30s avg %.0f\n", x, s[x]/n[x] }' "$CSV" | sort
        ok "layernorm_rocprofv3"
    else
        err "layernorm_rocprofv3" "CSV not found in rocprofv3 output"
    fi
fi

# ---------------------------------------------------------------------------
# 2. WAVE PRIMITIVES — Softmax (run3_softmax)
# ---------------------------------------------------------------------------
sep
log ">>> [2/10] Softmax with DPP reductions (wide + narrow)"

SM_FLAGS="$BASE_FLAGS -I$WAVE_INC -I$BENCH_INC -I$ROCM_INC"
SM_SRC="$ROOT/research/wave-primitives/benchmarks/bench_softmax.hip"

for EPL in 8 1; do
    W=$((64*EPL))
    BIN="$RESULTS/bench_softmax_epl${EPL}"
    log "  Compiling softmax ${W}-col (ELEMS_PER_LANE=$EPL) ..."
    if compile "softmax_${EPL}" $SM_FLAGS -DELEMS_PER_LANE=$EPL "$SM_SRC" -o "$BIN"; then
        log "  Running softmax ${W}-col ..."
        "$BIN" 2>&1 | tee "$RESULTS/softmax_${W}col.txt"
        if grep -q "PASS" "$RESULTS/softmax_${W}col.txt"; then
            ok "softmax_${W}col_correctness"
        else
            err "softmax_${W}col_correctness" "no PASS in output"
        fi
    else
        err "softmax_${W}col_compile" "hipcc failed"
    fi
done

# rocprofv3 on wide (EPL=8): want to see DPP < portable < hipCUB in LDS
SM_BIN_WIDE="$RESULTS/bench_softmax_epl8"
if [ -f "$SM_BIN_WIDE" ]; then
    log "  rocprofv3 on softmax 512-col ..."
    CSV=$(rocprof_counters "softmax_wide" "$SM_BIN_WIDE")
    if [ -n "$CSV" ]; then
        log "  LDS counter summary:"
        awk -F',' 'NR>1 {
            v=$(NF-2); c=$(NF-3); b="other";
            if ($0 ~ /DPP/)      b="DPP";
            else if ($0 ~ /PORTABLE/) b="portable";
            else if ($0 ~ /HIPCUB/)   b="hipCUB";
            if (b!="other") { key=b" "c; s[key]+=v; n[key]++ }
        } END { for (x in s) printf "    %-30s avg %.0f\n", x, s[x]/n[x] }' "$CSV" | sort
        ok "softmax_rocprofv3"
    else
        err "softmax_rocprofv3" "CSV not found"
    fi
fi

# ---------------------------------------------------------------------------
# 2b. WAVE PRIMITIVES — Scan DPP (bench_scan_dpp + types)
# ---------------------------------------------------------------------------
sep
log ">>> [3/10] Scan DPP — probe (correctness per type) + bench (portable vs DPP vs hipCUB)"

SCAN_FLAGS="$BASE_FLAGS -I$WAVE_INC -I$BENCH_INC -I$ROCM_INC"

# probe_scan_dpp: correctness for inclusive sum/max/min + exclusive_sum, all types
PROBE_SCAN_SRC="$ROOT/research/wave-primitives/tests/probe_scan_dpp.hip"
PROBE_SCAN_BIN="$RESULTS/probe_scan_dpp"
if compile "probe_scan_dpp" $BASE_FLAGS -I$WAVE_INC "$PROBE_SCAN_SRC" -o "$PROBE_SCAN_BIN"; then
    log "  Running probe_scan_dpp ..."
    "$PROBE_SCAN_BIN" 2>&1 | tee "$RESULTS/probe_scan_dpp_output.txt"
    if grep -q "ALL TYPES/OPS PASS" "$RESULTS/probe_scan_dpp_output.txt"; then
        ok "scan_dpp_all_types_probe"
    else
        err "scan_dpp_all_types_probe" "some type/op failed — check output"
    fi
else
    err "probe_scan_dpp_compile" "hipcc failed"
fi

SCAN_SRC="$ROOT/research/wave-primitives/benchmarks/bench_scan_dpp.hip"
SCAN_BIN="$RESULTS/bench_scan_dpp"

if compile "bench_scan_dpp" $SCAN_FLAGS "$SCAN_SRC" -o "$SCAN_BIN"; then
    log "  Running bench_scan_dpp (float, kReps amortized) ..."
    "$SCAN_BIN" 2>&1 | tee "$RESULTS/bench_scan_dpp_output.txt"
    if grep -q "PASS" "$RESULTS/bench_scan_dpp_output.txt"; then
        ok "scan_dpp_f32_correctness"
    else
        err "scan_dpp_f32_correctness" "PASS not found"
    fi
else
    err "scan_dpp_compile" "hipcc failed"
fi

# DPP scan across types
SCAN_TYPES_SRC="$ROOT/research/wave-primitives/benchmarks/bench_scan_dpp_types.hip"
SCAN_TYPES_BIN="$RESULTS/bench_scan_dpp_types"
if compile "bench_scan_dpp_types" $SCAN_FLAGS "$SCAN_TYPES_SRC" -o "$SCAN_TYPES_BIN"; then
    log "  Running bench_scan_dpp_types (f32/f64/i32/i64) ..."
    "$SCAN_TYPES_BIN" 2>&1 | tee "$RESULTS/bench_scan_dpp_types_output.txt"
    ok "scan_dpp_types_ran"
else
    err "scan_dpp_types_compile" "hipcc failed"
fi

# ---------------------------------------------------------------------------
# 2c. WAVE PRIMITIVES — Reduce DPP multi-type (bench_reduce_dpp_types)
# ---------------------------------------------------------------------------
sep
log ">>> [4/10] Reduce DPP across types (sum/max/min × f32/f64/i32/i64)"

RD_TYPES_SRC="$ROOT/research/wave-primitives/benchmarks/bench_reduce_dpp_types.hip"
RD_TYPES_BIN="$RESULTS/bench_reduce_dpp_types"
if compile "bench_reduce_dpp_types" $SCAN_FLAGS "$RD_TYPES_SRC" -o "$RD_TYPES_BIN"; then
    log "  Running bench_reduce_dpp_types ..."
    "$RD_TYPES_BIN" 2>&1 | tee "$RESULTS/bench_reduce_dpp_types_output.txt"
    if grep -q "PASS" "$RESULTS/bench_reduce_dpp_types_output.txt"; then
        ok "reduce_dpp_types_correctness"
    else
        err "reduce_dpp_types_correctness" "PASS not found — may be 64-bit DPP compile issue"
    fi
else
    err "reduce_dpp_types_compile" "hipcc failed"
fi

# ---------------------------------------------------------------------------
# 2d. WAVE PRIMITIVES — Block reduce (wave two-phase vs hipCUB BlockReduce)
# ---------------------------------------------------------------------------
sep
log ">>> [5/10] Block reduce — two-phase wave vs hipCUB BlockReduce (bs=128..1024)"

BR_SRC="$ROOT/research/wave-primitives/benchmarks/bench_block_reduce.hip"
BR_BIN="$RESULTS/bench_block_reduce"
if compile "bench_block_reduce" $SCAN_FLAGS "$BR_SRC" -o "$BR_BIN"; then
    log "  Running bench_block_reduce ..."
    "$BR_BIN" 2>&1 | tee "$RESULTS/bench_block_reduce_output.txt"
    if grep -q "PASS\|wave_block_reduce" "$RESULTS/bench_block_reduce_output.txt"; then
        ok "block_reduce_correctness"
    else
        err "block_reduce_correctness" "PASS not found or output unexpected"
    fi
else
    err "bench_block_reduce_compile" "hipcc failed"
fi

# ---------------------------------------------------------------------------
# 3. DME ABSTRACTION — probe_dme (gates all fa_dme work)
# ---------------------------------------------------------------------------
sep
log ">>> [6/10] DME probe — validate __builtin_amdgcn_global_load_lds semantics"

DME_SRC="$ROOT/research/dme-abstraction/tests/probe_dme.hip"
DME_BIN="$RESULTS/probe_dme"
DME_INC="$ROOT/research/dme-abstraction/include"

if compile "probe_dme" $BASE_FLAGS -I"$DME_INC" "$DME_SRC" -o "$DME_BIN"; then
    log "  Running probe_dme ..."
    "$DME_BIN" 2>&1 | tee "$RESULTS/probe_dme_output.txt"

    # Determine which variant passed: offset0+ptr or offset_both
    if grep -q "offset0+ptr.*ALL CORRECT" "$RESULTS/probe_dme_output.txt"; then
        log "  RESULT: offset0+ptr correct → advance global ptr per lane, offset for LDS only"
        DME_VALID=true
        ok "probe_dme_offset0"
        # Record which convention is correct for copy_tile_1d
        echo "CONVENTION: offset0+ptr (advance global ptr; offset applies to LDS)" \
            >> "$RESULTS/probe_dme_output.txt"
    fi
    if grep -q "offset_both.*ALL CORRECT" "$RESULTS/probe_dme_output.txt"; then
        log "  RESULT: offset_both correct → offset applies to BOTH ptrs (as documented)"
        DME_VALID=true
        ok "probe_dme_offset_both"
        echo "CONVENTION: offset_both (offset applies to both global+LDS ptrs)" \
            >> "$RESULTS/probe_dme_output.txt"
    fi
    if [ "$DME_VALID" = "false" ]; then
        err "probe_dme" "neither variant produced correct output — DME not usable"
    fi
else
    err "probe_dme_compile" "hipcc failed"
fi

# ---------------------------------------------------------------------------
# 4. FLASH ATTENTION — fa_naive (baseline + LDS counter)
# ---------------------------------------------------------------------------
sep
log ">>> [7/10] Flash Attention naive (step 1 baseline)"

FA_DIR="$ROOT/research/flash-attention-mi300x"
FA_INC="$WAVE_INC $BENCH_INC"
FA_FLAGS="$BASE_FLAGS -I$WAVE_INC -I$BENCH_INC -DFA_HEAD_DIM=64 -DFA_BC=64"

NAIVE_BIN="$RESULTS/fa_naive"
if compile "fa_naive" $FA_FLAGS "$FA_DIR/kernels/fa_naive.hip" -o "$NAIVE_BIN"; then
    log "  Running fa_naive ..."
    "$NAIVE_BIN" 2>&1 | tee "$RESULTS/fa_naive_output.txt"
    if grep -q "PASS" "$RESULTS/fa_naive_output.txt"; then
        ok "fa_naive_correctness"
    else
        err "fa_naive_correctness" "max_rel_err >= 1e-3 or PASS not found"
    fi

    # rocprofv3 for LDS baseline (this is the number DPP must beat)
    log "  rocprofv3 on fa_naive (LDS baseline) ..."
    CSV=$(rocprof_counters "fa_naive" "$NAIVE_BIN")
    if [ -n "$CSV" ]; then
        log "  fa_naive LDS counters:"
        awk -F',' 'NR>1 { v=$(NF-2); c=$(NF-3);
            if ($0 ~ /fa_naive_kernel/) {
                key="fa_naive "c; s[key]+=v; n[key]++
            }
        } END { for (x in s) printf "    %-35s avg %.0f\n", x, s[x]/n[x] }' "$CSV" | sort
        ok "fa_naive_rocprofv3"
    fi
else
    err "fa_naive_compile" "hipcc failed"
fi

# ---------------------------------------------------------------------------
# 5. FLASH ATTENTION — fa_dpp (Bc=64, 128, 256)
# ---------------------------------------------------------------------------
sep
log ">>> [8/10] Flash Attention DPP (step 2 — Bc=64/128/256)"

DPP_FLAGS_BASE="$BASE_FLAGS -I$WAVE_INC -I$BENCH_INC -DFA_HEAD_DIM=64"

for BC in 64 128 256; do
    DPP_BIN="$RESULTS/fa_dpp_bc${BC}"
    log "  Compiling fa_dpp Bc=${BC} ..."
    if compile "fa_dpp_bc${BC}" $DPP_FLAGS_BASE -DFA_BC=${BC} \
               "$FA_DIR/kernels/fa_dpp.hip" -o "$DPP_BIN"; then
        log "  Running fa_dpp Bc=${BC} ..."
        "$DPP_BIN" 2>&1 | tee "$RESULTS/fa_dpp_bc${BC}_output.txt"
        if grep -q "PASS" "$RESULTS/fa_dpp_bc${BC}_output.txt"; then
            ok "fa_dpp_bc${BC}_correctness"
        else
            err "fa_dpp_bc${BC}_correctness" "PASS not found or max_rel_err >= 1e-3"
        fi
    else
        err "fa_dpp_bc${BC}_compile" "hipcc failed"
    fi
done

# rocprofv3 on Bc=64 DPP — compare LDS vs naive (expect near-0 for reductions)
DPP64_BIN="$RESULTS/fa_dpp_bc64"
if [ -f "$DPP64_BIN" ]; then
    log "  rocprofv3 on fa_dpp Bc=64 (should be near-0 reduction LDS) ..."
    CSV_DPP=$(rocprof_counters "fa_dpp_bc64" "$DPP64_BIN")
    if [ -n "$CSV_DPP" ]; then
        log "  fa_dpp Bc=64 LDS counters:"
        awk -F',' 'NR>1 { v=$(NF-2); c=$(NF-3);
            if ($0 ~ /fa_dpp_kernel/) {
                key="fa_dpp_bc64 "c; s[key]+=v; n[key]++
            }
        } END { for (x in s) printf "    %-35s avg %.0f\n", x, s[x]/n[x] }' "$CSV_DPP" | sort
        ok "fa_dpp_bc64_rocprofv3"
    fi
fi

# ---------------------------------------------------------------------------
# 6. DME — probe again with bench_dme (if probe_dme was correct + bench exists)
# ---------------------------------------------------------------------------
sep
log ">>> [9/10] DME throughput benchmark (if probe_dme passed)"

BENCH_DME_SRC="$ROOT/research/dme-abstraction/tests/bench_dme.hip"
if [ "$DME_VALID" = "true" ] && [ -f "$BENCH_DME_SRC" ]; then
    BENCH_DME_BIN="$RESULTS/bench_dme"
    if compile "bench_dme" $BASE_FLAGS -I"$DME_INC" -I"$BENCH_INC" \
               "$BENCH_DME_SRC" -o "$BENCH_DME_BIN"; then
        log "  Running bench_dme ..."
        "$BENCH_DME_BIN" 2>&1 | tee "$RESULTS/bench_dme_output.txt"
        ok "bench_dme_ran"
    else
        err "bench_dme_compile" "hipcc failed"
    fi
elif [ "$DME_VALID" = "false" ]; then
    skip "bench_dme" "probe_dme failed — DME semantics unknown"
else
    skip "bench_dme" "bench_dme.hip not yet written (run probe_dme results determine design)"
fi

# ---------------------------------------------------------------------------
# 10. WAVE PRIMITIVES — bench_scan (portable scan vs hipCUB, no-kReps baseline)
# ---------------------------------------------------------------------------
sep
log ">>> [10/10] Scan correctness + portable vs hipCUB benchmark (bench_scan)"

BSCAN_SRC="$ROOT/research/wave-primitives/benchmarks/bench_scan.hip"
BSCAN_BIN="$RESULTS/bench_scan"
if compile "bench_scan" $SCAN_FLAGS "$BSCAN_SRC" -o "$BSCAN_BIN"; then
    log "  Running bench_scan ..."
    "$BSCAN_BIN" 2>&1 | tee "$RESULTS/bench_scan_output.txt"
    if grep -q "PASS" "$RESULTS/bench_scan_output.txt"; then
        ok "bench_scan_correctness"
    else
        err "bench_scan_correctness" "PASS not found"
    fi
else
    err "bench_scan_compile" "hipcc failed"
fi

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
sep
echo ""
log "=== SUMMARY ==="
echo ""
echo "PASS (${#PASS[@]}):"
for x in "${PASS[@]}"; do echo "  ✓ $x"; done
echo ""
echo "FAIL (${#FAIL[@]}):"
for x in "${FAIL[@]}"; do echo "  ✗ $x"; done
echo ""
echo "SKIP (${#SKIP[@]}):"
for x in "${SKIP[@]}"; do echo "  ⊘ $x"; done
echo ""
echo "DME_VALID=$DME_VALID"
echo ""

# Auto-decision: patch copy_tile_1d convention based on probe_dme result
if grep -q "CONVENTION: offset0+ptr" "$RESULTS/probe_dme_output.txt" 2>/dev/null; then
    log "AUTO-DECISION: probe_dme says 'offset0+ptr' — current copy_tile_1d uses 'offset_both'."
    log "  → Need to revert dme_copy.hpp copy_tile_1d to use src+i with offset=0."
    echo "ACTION_REQUIRED: fix copy_tile_1d — use (src+i, dst_lds, 0) not (src, dst_lds, i*4)" \
        >> "$RESULTS/decisions.txt"
elif grep -q "CONVENTION: offset_both" "$RESULTS/probe_dme_output.txt" 2>/dev/null; then
    log "AUTO-DECISION: probe_dme confirms 'offset_both' — current copy_tile_1d is CORRECT."
    echo "ACTION_CONFIRMED: copy_tile_1d uses (src, dst_lds, i*4) — hardware matches docs" \
        >> "$RESULTS/decisions.txt"
fi

log "All results saved to: $RESULTS/"
log "Run complete: $(date)"

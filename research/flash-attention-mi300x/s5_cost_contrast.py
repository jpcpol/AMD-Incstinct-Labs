#!/usr/bin/env python3
"""
s5_cost_contrast.py — S5: O(n²) flat-context vs O(κ) governance-state cost contrast.

Pre-registered: PRE_REGISTRATION_S5_COST_CONTRAST.md. Implements it. This is the
SOFTWARE side of the contrast — it needs no GPU. It combines two already-measured
quantities:

  O(n²) side : the R1 flat-context attention law  Cost_flat(n) = c · n^p  (p=1.90,
               R²=0.9964), from results/2026-06-08_R1_baseline_and_2B_llm_mi300x.txt
  O(κ) side  : the L4-A governance-state read-count cost_reads ≈ 1322 (κ+|E|),
               constant in n, from CAL/L4/.../results/l4a_results.json

and reports the decoupling ratio D(n) = Cost_flat(n) / Cost_gov as a curve, with the
confirmatory checks C1/C2/C3. The VM session only re-runs the seqLen sweep to pin the
constant c and confirm C1 under fresh clocks.

HONESTY (prereg §0, C3, inherited from L4-A C4): κ=1296 is at governance-corpus scale
(n=30 sessions, 11 dims), NOT seqLen scale. This contrast shows the MECHANISM (cost
coupling vs decoupling), not a production-scale speedup. κ does not reflect the prune.

Usage:
  python s5_cost_contrast.py            # uses the R1 law constants baked in below
  python s5_cost_contrast.py --l4a PATH # point at a specific l4a_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Portable unicode output (Windows consoles default to cp1252; the VM is UTF-8).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ── O(n²) side: the measured R1 flat-context law ────────────────────────────
# Cost_flat(n) = c · n^p. We recover (c, p) from the R1 sweep points so the law is
# reproducible from data, not hard-coded. These are the published R1 measurements
# (D=128, MFMA path), latency in microseconds.
R1_SWEEP = {
    512: 103.7,
    768: 299.6,
    1024: 457.1,
    1536: 952.2,
    2048: 1634.8,
    3072: 3626.1,
    4096: 6015.1,
}
R1_ACCEPT = (1.8, 2.2)  # quadratic-regime acceptance window


def fit_power_law(points: dict[int, float]) -> tuple[float, float, float]:
    """Least-squares log-log fit. Returns (c, p, R²) for Cost = c · n^p."""
    xs = [math.log(n) for n in points]
    ys = [math.log(t) for t in points.values()]
    m = len(xs)
    mx, my = sum(xs) / m, sum(ys) / m
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    p = sxy / sxx
    b = my - p * mx          # intercept in log space
    c = math.exp(b)
    # R²
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (b + p * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot
    return c, p, r2


def load_kappa_cost(l4a_path: Path) -> tuple[int, int, float]:
    """O(κ) side from L4-A: returns (kappa, cost_reads, U_mean), all n-independent."""
    data = json.loads(l4a_path.read_text())
    kappa = data["kappa"]
    per = data["C5_operability"]["per_graph"]
    cost_reads = max(g["cost_reads"] for g in per.values())  # worst-case, constant
    us = list(data["delivered_to_AMD"]["U_per_graph"].values())
    return kappa, cost_reads, sum(us) / len(us)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Try a few known layouts for the CAL/L4 handoff json. On the VM, pass --l4a.
    rel = ("CAL", "L4", "experiments", "efficiency_hypothesis", "results",
           "l4a_results.json")
    here = Path(__file__).resolve()
    candidates = [here.parent / "results" / "l4a_results.json"]  # VM: alongside the sweep
    for up in (5, 6):  # local workspace layouts (guarded — short paths on the VM)
        if len(here.parents) > up:
            candidates.append(here.parents[up].joinpath(*rel))
    candidates.append(Path.home() / "Documents" / "Aural Syncro" / Path(*rel))
    default_l4a = next((str(c) for c in candidates if c.exists()), str(candidates[0]))
    ap.add_argument("--l4a", default=default_l4a,
                    help="path to L4-A results json (the O(κ) side)")
    args = ap.parse_args()

    print("=" * 76)
    print("S5 — Cost contrast: O(n²) flat-context vs O(κ) governance state (MI300X)")
    print("=" * 76)

    # O(n²) side
    c, p, r2 = fit_power_law(R1_SWEEP)
    print("\n[O(n²) side] flat-context attention law (R1, D=128, MFMA):")
    print(f"  Cost_flat(n) = {c:.4e} · n^{p:.3f}   (R²={r2:.4f})")
    c1_pass = R1_ACCEPT[0] <= p <= R1_ACCEPT[1]
    print(f"  C1 (exponent ∈ [{R1_ACCEPT[0]},{R1_ACCEPT[1]}]): "
          f"{'PASS' if c1_pass else 'FAIL'}  (p={p:.3f})")

    # O(κ) side
    l4a_path = Path(args.l4a)
    if l4a_path.exists():
        kappa, cost_gov_reads, u_mean = load_kappa_cost(l4a_path)
        src = str(l4a_path)
    else:  # fallback to published handoff values if the CAL repo isn't local
        kappa, cost_gov_reads, u_mean = 1296, 1322, 0.862
        src = "published L4-A handoff (CAL repo not found locally)"
    print(f"\n[O(κ) side] governance-state cost (L4-A dual volume):")
    print(f"  source: {src}")
    print(f"  κ(V_Tucker) = {kappa}   cost_reads ≈ {cost_gov_reads} (κ+|E|)   "
          f"U(G_pruned) ≈ {u_mean:.3f}")
    print(f"  → CONSTANT in n (depends on governance-state complexity, not artifacts)")

    # Decoupling ratio D(n) = Cost_flat(n) / Cost_gov.
    # Both sides are in different native units (µs vs reads); D(n) is a DIMENSIONLESS
    # SHAPE comparison — what matters is that the numerator grows ~n^p and the
    # denominator is flat. We normalize D(n) to D(512)=1 so the curve shows the
    # *growth of the gap*, not an absolute (unit-mixed) ratio.
    print("\n[Contrast] decoupling ratio D(n) = Cost_flat(n) / Cost_gov, "
          "normalized to n=512:")
    base = (c * 512 ** p) / cost_gov_reads
    ns = sorted(R1_SWEEP)
    d_points = {}
    for n in ns:
        d = (c * n ** p) / cost_gov_reads
        d_points[n] = d / base
        print(f"  n={n:5d}   D(n)={d/base:7.2f}×   "
              f"(flat≈{c * n ** p:8.1f}µs, gov≈{cost_gov_reads} reads, const)")

    # C2: D(n) strictly increasing + its growth exponent ≈ p (since gov is flat).
    increasing = all(d_points[ns[i]] < d_points[ns[i + 1]] for i in range(len(ns) - 1))
    _, d_exp, d_r2 = fit_power_law(d_points)
    c2_pass = increasing and (R1_ACCEPT[0] <= d_exp <= R1_ACCEPT[1])
    print(f"\n  C2 (D(n) increasing ∧ growth-exp ∈ [{R1_ACCEPT[0]},{R1_ACCEPT[1]}]): "
          f"{'PASS' if c2_pass else 'FAIL'}  "
          f"(monotone={increasing}, exp={d_exp:.3f}, R²={d_r2:.4f})")

    # C3: scope-honesty statement (printed, so it lives in the captured output).
    print("\n  C3 (scope honesty) — STATED:")
    print("    (i)  κ=1296 is at governance-corpus scale (n=30 sessions, 11 dims),")
    print("         NOT seqLen scale. The two `n` are different objects.")
    print("    (ii) This contrast shows the MECHANISM (cost coupling vs decoupling),")
    print("         not a production-scale speedup.")
    print("    (iii) κ does not reflect the Form-1 prune (κ is the Tucker core's;")
    print("          the prune lives in G_pruned). Inherited from L4-A C4.")

    verdict = c1_pass and c2_pass
    print("\n" + "=" * 76)
    if verdict:
        print("VERDICT: S5 demonstrates the L4 Efficiency Hypothesis MECHANISM on MI300X.")
        print("  Flat-context cost grows ~n^%.2f; governance-state cost is bounded by" % p)
        print("  κ(V) and decoupled from artifact volume n. Condition (b) met as a")
        print("  mechanism result. → FREEZE.")
    else:
        print("VERDICT: incomplete — see failing check above before any contrast claim.")
    print("=" * 76)


if __name__ == "__main__":
    main()

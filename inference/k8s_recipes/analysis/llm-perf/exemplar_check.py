#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa

"""exemplar_check.py — is this run an EXEMPLAR of its recipe's committed reference bar?

Comparison scope: hold the recipe CONSTANT (same model + same GPU
type + same config) and compare the SAME cell run on a DIFFERENT cluster to the cell's committed
`envelope.exemplar.reference`. It is a cross-CLUSTER reproducibility check — NOT a cross-hardware or
cross-config comparison. Each cell owns its own bar, so comparing a run against its own cell's
reference is automatically same-recipe/same-GPU.

CRITICAL: only compare runs from the SAME GPU type. B200 vs GB300 is a hardware comparison, not an
exemplar check. Pass --actual-gpu <GPU_PRODUCT> (e.g. "NVIDIA-B200") from your cluster profile to
enforce this — the check will fail fast if the GPU type doesn't match envelope.gpu_type.

Deterministic + CSV-only: it reads the cell's `recipe.yaml` (the exemplar contract + the SLA) and the
run's `metrics_summary.csv` (produced by aggregate_metrics.py). It re-uses the aggregator's own numbers
and judges the SLA exactly like sla_compare_dashboard.py: a rung PASSES iff BOTH the TTFT and the TPOT
(=ITL) at `bench.sla.stop_stat` are within their limits. No re-aggregation, no hand-tuning.

Usage:
  exemplar_check.py <cell-dir> <metrics_summary.csv> [--reference N] [--tolerance-pct P]
                    [--incomplete-max PCT] [--actual-gpu GPU_PRODUCT] [--json]

Exit: 0 = EXEMPLAR (or no reference bar set yet — advisory) · 1 = NOT EXEMPLAR · 2 = usage/data error.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("exemplar_check: requires pyyaml (pip install pyyaml)")

# Metric registry. Each entry: how to compute the headline number from the passing rungs, and its
# direction (all current metrics are higher-is-better). `higher_better` drives the tolerance test:
# a run is exemplar if it is at most tolerance_pct BELOW the bar (being better than the bar is fine).
METRICS = {
    # goal=max-concurrency-sla: highest measured SLA-passing rung unless the recipe comparison asks
    # for interpolation explicitly.
    "max_concurrency_at_sla": {"higher_better": True, "unit": "concurrency"},
    # decode Output-TPS/GPU at that crossing (the operating point at the SLA ceiling).
    "tps_per_gpu_at_sla": {"higher_better": True, "unit": "tok/s/gpu"},
    # goal=pareto: geomean over rungs of the per-rung geomean g(p)=√(Output-TPS/GPU · Output-TPS/user).
    # Equal per-point weight (no frontier-spacing bias), reference-free, always defined, valid for one point.
    "pareto_geomean": {"higher_better": True, "unit": "geomean(tps/gpu·tps/user)"},
}


def pareto_points(rows):
    """Per-rung {concurrency, g} where g = geomean(TPS/GPU, TPS/user), sorted by concurrency."""
    pts = []
    for r in rows:
        c = as_float(r.get("concurrency"))
        x = as_float(r.get("tokens_per_s_per_user_from_itl"))
        if x is None:
            x = as_float(r.get("output_token_throughput_per_user_p50"))
        y = as_float(r.get("throughput_per_gpu_tok_per_s"))
        if c is not None and x and y and x > 0 and y > 0:
            pts.append({"concurrency": c, "g": round(math.sqrt(x * y), 4)})
    pts.sort(key=lambda p: p["concurrency"])
    return pts


def pareto_geomean(rows):
    """(value, per_point). value = geomean over rungs of g(p); per_point = the per-rung geomeans. A SINGLE
    point is valid (geomean of one value). Every concurrency rung carries equal weight.
    """
    pts = pareto_points(rows)
    if not pts:
        return None, []
    gs = [p["g"] for p in pts]
    return round(math.exp(sum(math.log(g) for g in gs) / len(gs)), 1), pts


def pareto_point_compare(a_pts, b_pts, tol_pct=5.0):
    """Point-by-point compare of two cells' per-rung geomeans. verdict='exemplar' iff rung sets MATCH and every
    shared rung |diff|<=tol; else 'outside' (shared rungs) or 'not-comparable' (no overlap).
    """
    a = {p["concurrency"]: p["g"] for p in a_pts}
    b = {p["concurrency"]: p["g"] for p in b_pts}
    shared = sorted(set(a) & set(b))
    matched = set(a) == set(b)
    per = []
    worst = 0.0
    for c in shared:
        d = abs(a[c] - b[c]) / b[c] * 100.0 if b[c] else None
        per.append(
            {
                "concurrency": c,
                "a": a[c],
                "b": b[c],
                "diff_pct": round(d, 2) if d is not None else None,
            }
        )
        if d is not None:
            worst = max(worst, d)
    verdict = "exemplar" if (matched and shared and worst <= tol_pct) else ("outside" if shared else "not-comparable")
    return {
        "per_point": per,
        "worst_pct": round(worst, 2),
        "matched": matched,
        "verdict": verdict,
    }


def as_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN (f != f) is bad/missing data, not a value: NaN <= limit is False, which
    # would silently mark the rung as FAILING. Treat NaN like missing → skipped.


def read_rows(csv_path: Path):
    """Rows from metrics_summary.csv, skipping the leading `# schema_version=` comment line."""
    with csv_path.open() as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def passing_rungs(rows, ttft_limit, tpot_limit, stat, incomplete_max):
    """Rungs meeting the SLA: BOTH ttft_p{stat} <= ttft_limit AND itl_p{stat} <= tpot_limit, and whose
    error/incomplete rate is under the guard (a rung with many cancellations is not a real point).
    """
    out = []
    for r in rows:
        conc = as_float(r.get("concurrency"))
        ttft = as_float(r.get(f"ttft_{stat}_ms"))
        tpot = as_float(r.get(f"itl_{stat}_ms"))
        err = as_float(r.get("error_rate_pct")) or 0.0
        if conc is None or ttft is None or tpot is None:
            continue
        if ttft <= ttft_limit and tpot <= tpot_limit and err <= incomplete_max:
            out.append(
                {
                    "concurrency": int(conc),
                    "ttft": ttft,
                    "tpot": tpot,
                    "err": err,
                    "tps_per_gpu": as_float(r.get("throughput_per_gpu_tok_per_s")),
                }
            )
    return out


def all_rungs(rows, stat):
    """Every rung (pass AND fail), sorted by concurrency — needed to bracket the SLA crossing."""
    out = []
    for r in rows:
        conc, ttft, tpot = (
            as_float(r.get("concurrency")),
            as_float(r.get(f"ttft_{stat}_ms")),
            as_float(r.get(f"itl_{stat}_ms")),
        )
        if conc is None or ttft is None or tpot is None:
            continue
        out.append(
            {
                "concurrency": conc,
                "ttft": ttft,
                "tpot": tpot,
                "err": as_float(r.get("error_rate_pct")) or 0.0,
                "tps_per_gpu": as_float(r.get("throughput_per_gpu_tok_per_s")),
            }
        )
    return sorted(out, key=lambda x: x["concurrency"])


def _interp(c_lo, v_lo, c_hi, v_hi, limit):
    """Concurrency where the metric line crosses `limit` (linear between the bracketing rungs)."""
    return c_hi if v_hi == v_lo else c_lo + (c_hi - c_lo) * (limit - v_lo) / (v_hi - v_lo)


def sla_crossing(rungs, ttft_limit, tpot_limit, incomplete_max, warn_ratio=1.5):
    """INTERPOLATED max_concurrency_at_sla: the concurrency where the first SLA metric crosses its limit,
    = min(TTFT-crossing, TPOT-crossing) over the pass→fail bracket. Continuous (not quantized to a rung),
    so A-vs-B is comparable to any precision. Returns a dict with value, bracket, binding metric, the
    interpolated tps/gpu at the crossing, and a guardrail note (flags a too-wide bracket / out-of-range).
    """

    def P(r):
        return r["ttft"] <= ttft_limit and r["tpot"] <= tpot_limit and r["err"] <= incomplete_max

    if not rungs:
        return {"value": None, "status": "no_data", "note": "no rungs"}
    lo = hi = None
    for i in range(len(rungs) - 1):
        if P(rungs[i]) and not P(rungs[i + 1]):
            lo, hi = rungs[i], rungs[i + 1]
            break
    if lo is None:
        if all(P(r) for r in rungs):
            return {
                "value": None,
                "status": "above_range",
                "note": f"all rungs pass up to c={rungs[-1]['concurrency']:g} — ceiling is ABOVE the swept range; extend the sweep",
            }
        if not P(rungs[0]):
            return {
                "value": None,
                "status": "below_range",
                "note": f"lowest rung c={rungs[0]['concurrency']:g} already fails — ceiling is BELOW the swept range; add lower rungs",
            }
        return {
            "value": None,
            "status": "no_bracket",
            "note": "no clean pass→fail bracket (non-monotonic curve?)",
        }
    c_lo, c_hi = lo["concurrency"], hi["concurrency"]
    cross_ttft = _interp(c_lo, lo["ttft"], c_hi, hi["ttft"], ttft_limit) if hi["ttft"] > ttft_limit else float("inf")
    cross_tpot = _interp(c_lo, lo["tpot"], c_hi, hi["tpot"], tpot_limit) if hi["tpot"] > tpot_limit else float("inf")
    binding = "TTFT" if cross_ttft < cross_tpot else "TPOT"
    c_star = min(cross_ttft, cross_tpot)
    tps = None
    if lo.get("tps_per_gpu") is not None and hi.get("tps_per_gpu") is not None and c_hi != c_lo:
        tps = round(
            lo["tps_per_gpu"] + (hi["tps_per_gpu"] - lo["tps_per_gpu"]) * (c_star - c_lo) / (c_hi - c_lo),
            1,
        )
    ratio = c_hi / c_lo
    note = f"bracket [{c_lo:g},{c_hi:g}] ({ratio:.2f}×), binding={binding}"
    if ratio > warn_ratio:
        note += f" — ⚠ bracket >{warn_ratio:g}×; add a rung near {c_star:.0f} to tighten (interpolation is approximate here)"
    return {
        "value": round(c_star, 1),
        "bracket": [c_lo, c_hi],
        "ratio": round(ratio, 2),
        "binding": binding,
        "tps_per_gpu_at_crossing": tps,
        "status": "ok",
        "note": note,
    }


def measured_rung_policy(exemplar):
    """True when the recipe contract names a measured rung metric, not a continuous crossing."""
    comparison = str((exemplar or {}).get("comparison") or "").lower()
    return "highest" in comparison and "rung" in comparison and "interpol" not in comparison


def measured_ceiling(passing):
    """Highest measured SLA-passing concurrency rung, or None when no rung passes."""
    values = [p["concurrency"] for p in passing if p.get("concurrency") is not None]
    return max(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Check a run against its recipe's exemplar reference bar.")
    ap.add_argument("cell", help="cell dir (contains recipe.yaml)")
    ap.add_argument("csv", help="path to the run's metrics_summary.csv")
    ap.add_argument(
        "--reference",
        type=float,
        default=None,
        help="override envelope.exemplar.reference",
    )
    ap.add_argument(
        "--tolerance-pct",
        type=float,
        default=None,
        help="override envelope.exemplar.tolerance_pct",
    )
    ap.add_argument(
        "--incomplete-max",
        type=float,
        default=5.0,
        help="max error/incomplete %% for a rung to count (default 5)",
    )
    ap.add_argument(
        "--actual-gpu",
        default=None,
        metavar="GPU_PRODUCT",
        help="GPU type of the cluster that produced this run (e.g. NVIDIA-B200, from GPU_PRODUCT "
        "in your cluster profile). If provided, must match envelope.gpu_type — mismatches "
        "are a hard error because cross-GPU comparisons are hardware comparisons, not "
        "exemplar checks. Tip: set GPU_PRODUCT in your cluster profile and pass it here.",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--set",
        action="store_true",
        dest="set_ref",
        help="write the computed value into the cell's envelope.exemplar.reference (the baseline)",
    )
    args = ap.parse_args()

    cell = Path(args.cell)
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    env = recipe.get("envelope") or {}
    ex = env.get("exemplar") or {}
    metric = ex.get("metric")

    # Same-GPU guard: exemplar checks are cross-cluster, not cross-hardware.
    # B200 vs GB300 is a hardware comparison and will always show a difference — that's expected,
    # not a reproducibility signal. Enforce same GPU type when the caller provides --actual-gpu.
    if args.actual_gpu is not None:
        expected_gpu = env.get("gpu_type", "")
        if args.actual_gpu.strip() != expected_gpu.strip():
            print(
                f"exemplar_check: GPU MISMATCH — run was on '{args.actual_gpu}' but "
                f"envelope.gpu_type is '{expected_gpu}'.\n"
                f"  Exemplar checks compare the SAME cell across different CLUSTERS, not different HARDWARE.\n"
                f"  To benchmark on a different GPU: create a new cell with gpu_type: {args.actual_gpu}",
                file=sys.stderr,
            )
            return 2
    if metric not in METRICS:
        print(
            f"exemplar_check: envelope.exemplar.metric={metric!r} is not one of {list(METRICS)}",
            file=sys.stderr,
        )
        return 2

    # pareto (metric==pareto_geomean) is the per-point geomean of the frontier and carries NO SLA semantics:
    # it neither requires bench.sla nor computes/emits any SLA-derived field. Short-circuit accordingly.
    is_pareto = metric == "pareto_geomean"
    sla = (recipe.get("bench") or {}).get("sla") or {}
    ttft_limit = as_float(sla.get("ttft_ms"))
    tpot_limit = as_float(sla.get("tpot_ms"))
    stat = sla.get("stop_stat", "p50")
    if not is_pareto:
        if ttft_limit is None or tpot_limit is None:
            print(
                "exemplar_check: bench.sla.ttft_ms and bench.sla.tpot_ms are required",
                file=sys.stderr,
            )
            return 2
        if stat not in ("p50", "p90", "p99"):
            # p95 lives only in per-step json, not the CSV; keep this tool CSV-only.
            print(
                f"exemplar_check: stop_stat={stat!r} not available in metrics_summary.csv (use p50/p90/p99)",
                file=sys.stderr,
            )
            return 2

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"exemplar_check: no metrics_summary.csv at {csv_path}", file=sys.stderr)
        return 2
    rows = read_rows(csv_path)
    # value by goal/metric: pareto = geomean over rungs of the per-point g; max-concurrency can be either the
    # measured highest passing rung or the interpolated crossing depending on the recipe comparison contract.
    if is_pareto:
        # No SLA semantics on pareto: don't compute passing_rungs / sla_crossing (they need SLA limits).
        passing, crossing, use_measured_rung = [], {}, False
        value, pareto_pts = pareto_geomean(rows)
        # pareto_geomean is valid for a SINGLE operating point (geomean of one value); value is None ONLY
        # when NO usable frontier point resolves. A single-point pareto still passes CI — it is just not a
        # real frontier and should be evaluated with the other points in its recipe family.
        if len(pareto_pts) == 1:
            print(
                f"exemplar_check: WARN pareto '{env.get('name')}' is a single operating point, not a real "
                f"frontier — pareto_geomean is valid (geomean of one rung); evaluate it with the "
                f"other operating points in the same recipe family.",
                file=sys.stderr,
            )
        elif value is None:
            print(
                f"exemplar_check: WARN pareto '{env.get('name')}' resolves NO usable frontier point → "
                f"pareto_geomean undefined (need at least one rung with TPS/GPU and TPS/user).",
                file=sys.stderr,
            )
    else:
        passing = passing_rungs(rows, ttft_limit, tpot_limit, stat, args.incomplete_max)  # informational
        crossing = sla_crossing(all_rungs(rows, stat), ttft_limit, tpot_limit, args.incomplete_max)
        use_measured_rung = metric == "max_concurrency_at_sla" and measured_rung_policy(ex)
        if metric == "max_concurrency_at_sla":
            value = measured_ceiling(passing) if use_measured_rung else crossing["value"]
        else:  # tps_per_gpu_at_sla
            value = crossing.get("tps_per_gpu_at_crossing")

    if args.set_ref:
        # Format-preserving write of the computed value as the committed baseline (targeted line edit,
        # not a YAML round-trip, so comments/formatting survive). Only exemplar.reference matches.
        import re

        if value is None:
            print(
                "exemplar_check: no SLA-passing rung — nothing to set as reference",
                file=sys.stderr,
            )
            return 1
        rp = cell / "recipe.yaml"
        txt = rp.read_text()
        new = re.sub(
            r"^(\s*reference:)[^\n]*",
            lambda m: f"{m.group(1)} {value:g}",
            txt,
            count=1,
            flags=re.M,
        )
        if new == txt:
            print(
                "exemplar_check: could not find an envelope.exemplar.reference line to set",
                file=sys.stderr,
            )
            return 2
        rp.write_text(new)
        print(
            f"exemplar_check: set envelope.exemplar.reference = {value:g} in {rp}\n"
            f"  -> commit this as the {env.get('gpu_type')} {metric} baseline; other clusters now check against it."
        )
        return 0

    reference = args.reference if args.reference is not None else as_float(ex.get("reference"))
    tol = args.tolerance_pct if args.tolerance_pct is not None else as_float(ex.get("tolerance_pct")) or 5.0
    higher_better = METRICS[metric]["higher_better"]
    unit = ex.get("unit") or METRICS[metric]["unit"]

    result = {
        "cell": env.get("name"),
        "gpu_type": env.get("gpu_type"),
        "metric": metric,
        "unit": unit,
        "value": value,
        "reference": reference,
        "tolerance_pct": tol,
    }
    if is_pareto:
        # pareto carries NO SLA semantics — emit the geomean value + its per-point breakdown + axes, never
        # sla / passing_concurrencies / crossing.
        result["value_source"] = "geomean_over_rungs"
        result["per_point"] = pareto_pts
        result["axes"] = ["tps_per_gpu", "tps_per_user"]
    else:
        result["sla"] = {"ttft_ms": ttft_limit, "tpot_ms": tpot_limit, "stat": stat}
        result["passing_concurrencies"] = [p["concurrency"] for p in sorted(passing, key=lambda p: p["concurrency"])]
        result["value_source"] = "highest_sla_passing_rung" if use_measured_rung else "interpolated_sla_crossing"
        result["crossing"] = {k: crossing.get(k) for k in ("bracket", "ratio", "binding", "status", "note")}

    # Verdict.
    if value is None:
        if is_pareto:
            # No usable frontier point → geomean undefined. Advisory, not a fail.
            result["verdict"] = "NO_PARETO_GEOMEAN"
        else:
            # above_range = all rungs pass (the run is GOOD, just needs more rungs to pin the number) → advisory,
            # not a failure. Everything else (below_range / no_bracket) = doesn't meet SLA / can't determine.
            result["verdict"] = "ABOVE_RANGE" if crossing.get("status") == "above_range" else "NO_SLA_RUNG"
        result["delta_pct"] = None
    elif reference is None:
        # No committed bar yet — this run is a baseline CANDIDATE, not a pass/fail.
        result["verdict"], result["delta_pct"] = "NO_REFERENCE", None
    else:
        delta = (value - reference) / reference * 100.0 if reference else 0.0
        result["delta_pct"] = round(delta, 2)
        within = delta >= -tol if higher_better else delta <= tol
        result["verdict"] = "EXEMPLAR" if within else "NOT_EXEMPLAR"

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        v = "—" if value is None else f"{value:g} {unit}"
        ref = "—" if reference is None else f"{reference:g} {unit}"
        icon = {
            "EXEMPLAR": "✅",
            "NOT_EXEMPLAR": "❌",
            "NO_REFERENCE": "🟡",
            "NO_SLA_RUNG": "❌",
            "ABOVE_RANGE": "🟡",
            "NO_PARETO_GEOMEAN": "🟡",
        }[result["verdict"]]
        print(f"exemplar: {env.get('name')}  ·  metric={metric} ({unit})")
        if is_pareto:
            print(
                f"  pareto: geomean over rungs of g(p)=√(Output-TPS/GPU · Output-TPS/user); "
                f"exemplar gate is point-by-point (no SLA)"
            )
        else:
            print(
                f"  SLA: TTFT {stat} ≤ {ttft_limit:g}ms AND TPOT {stat} ≤ {tpot_limit:g}ms  ·  incomplete ≤ {args.incomplete_max:g}%"
            )
            print(f"  SLA-passing rungs: {result['passing_concurrencies'] or '(none)'}")
            if use_measured_rung:
                print(f"  measured ceiling: highest SLA-passing swept rung")
                print(f"  advisory crossing: {crossing.get('note', '—')}")
            else:
                print(f"  interpolated crossing: {crossing.get('note', '—')}")
        print(f"  value = {v} ({result['value_source']})   reference = {ref}   tolerance = ±{tol:g}%")
        if result["verdict"] == "NO_PARETO_GEOMEAN":
            print(
                f"  {icon} pareto_geomean undefined — no usable frontier point (need a rung with "
                f"TPS/GPU and TPS/user)."
            )
        elif result["verdict"] == "ABOVE_RANGE":
            print(f"  {icon} {crossing['note']} — not a fail; add higher rungs to pin the number.")
        elif result["verdict"] == "NO_SLA_RUNG":
            print(f"  {icon} NO SLA-passing rung — {crossing.get('note', 'does not meet the SLA at any concurrency')}.")
        elif result["verdict"] == "NO_REFERENCE":
            print(
                f"  {icon} no reference bar set. To adopt THIS run as the bar, set "
                f"envelope.exemplar.reference: {value:g} in {cell}/recipe.yaml."
            )
        else:
            print(
                f"  {icon} {result['verdict']}  (Δ {result['delta_pct']:+.2f}% vs bar; "
                f"{'≥' if higher_better else '≤'} -{tol:g}% required)"
            )

    # exit: 0 for EXEMPLAR / advisory states; 1 only for a real miss.
    return 1 if result["verdict"] in ("NOT_EXEMPLAR", "NO_SLA_RUNG") else 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""adaptive_sweep.py <cell> <profile> — run a max-concurrency-sla sweep the adaptive way (push-button).

Loops the deterministic planner over the existing primitives:
    plan (sweep_planner) → run ONE rung (sweep.sh --rungs) → fetch + aggregate → read TTFT/TPOT →
    feed back → … until the planner says done/stop → interpolate the crossing (exemplar_check).
~2-4 rungs instead of the full grid. The planner makes every decision (deterministic, reproducible); this is
just the glue. (The efficient single-Job version is the Phase-2 runner extraction — deliberately deferred.)

  scripts/adaptive_sweep.py <cell> <profile>                 # live: drives the cluster
  scripts/adaptive_sweep.py <cell> <profile> --dry-run --curve-crossing 150   # no cluster: simulate + preview

Reads adaptive_sweep {start,min,max,ratio,bracket_tolerance,max_runs} + bench.sla from the cell's recipe.yaml.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sweep_planner as planner  # the deterministic decision engine

try:
    import yaml
except ImportError:
    sys.exit("adaptive_sweep: requires pyyaml")


def sh(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd], **kw)
    # A failed sweep.sh / fetch / aggregate must ABORT — otherwise the planner reads a missing/stale CSV and
    # dies with a misleading "could not read latency" instead of the real error.
    if r.returncode != 0:
        sys.exit(f"adaptive_sweep: step failed (exit {r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


def read_rung_latency(csv_path: Path, stat: str):
    """(ttft, tpot) at bench.sla.stop_stat from a one-rung metrics_summary.csv."""
    import csv

    rows = [r for r in csv.DictReader([l for l in csv_path.read_text().splitlines() if not l.startswith("#")])]
    if not rows:
        return None
    r = rows[-1]

    def f(k):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return None

    return f(f"ttft_{stat}_ms"), f(f"itl_{stat}_ms")


def main() -> int:
    ap = argparse.ArgumentParser(description="Adaptive max-concurrency-sla sweep (planner-driven).")
    ap.add_argument("cell")
    ap.add_argument("profile")
    ap.add_argument("--run-prefix", default="adapt")
    ap.add_argument("--dry-run", action="store_true", help="simulate against a synthetic curve; no cluster")
    ap.add_argument(
        "--curve-crossing", type=float, default=150.0, help="(dry-run) where the synthetic TPOT curve crosses the limit"
    )
    # overrides — let an agent tune the search WITHOUT editing (and re-hashing) the recipe
    ap.add_argument("--start", type=int, help="override adaptive_sweep.start (the expected-crossing prior)")
    ap.add_argument("--grid-min", type=int, help="override grid min")
    ap.add_argument("--grid-max", type=int, help="override grid max")
    ap.add_argument("--ratio", type=float, help="override grid_ratio")
    ap.add_argument("--max-runs", type=int, help="override max_runs")
    args = ap.parse_args()

    cell = Path(args.cell)
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    bench = recipe.get("bench") or {}
    ad = bench.get("adaptive_sweep") or {}
    sla = bench.get("sla") or {}
    stat = sla.get("stop_stat", "p50")
    cfg = {
        "start": ad.get("start")
        or (max(bench.get("sweep_concurrency") or [128]) if bench.get("sweep_concurrency") else 128),
        "min": ad.get("min", 16),
        "max": ad.get("max") or 1024,
        "ratio": ad.get("grid_ratio", 1.3),
        "bracket_tolerance": ad.get("bracket_tolerance", 1.3),
        "max_runs": ad.get("max_runs", 6),
        "ttft_limit": float(sla.get("ttft_ms", 10000)),
        "tpot_limit": float(sla.get("tpot_ms", 100)),
    }
    for k, v in (
        ("start", args.start),
        ("min", args.grid_min),
        ("max", args.grid_max),
        ("ratio", args.ratio),
        ("max_runs", args.max_runs),
    ):
        if v is not None:
            cfg[k] = v
    print(
        f"adaptive_sweep: {cell.name}  cfg={{start:{cfg['start']}, grid:{cfg['min']}..{cfg['max']}@{cfg['ratio']}×, "
        f"tol:{cfg['bracket_tolerance']}×, max_runs:{cfg['max_runs']}}}  SLA TTFT≤{cfg['ttft_limit']:g}/TPOT≤{cfg['tpot_limit']:g}@{stat}"
    )

    meas, run_dir = [], None
    while True:
        out = planner.plan(cfg, meas)
        if out["action"] != "run":
            print(f"\n{out['action'].upper()}: {out.get('reason')}  {out.get('bracket') or out.get('note','')}")
            bracket = out.get("bracket")
            break
        c = out["concurrency"]
        print(f"\n[rung {len(meas)+1}] plan → c={c} ({out['phase']})")
        if args.dry_run:
            # synthetic monotonic TPOT curve crossing the limit at --curve-crossing; TTFT well under limit
            tpot = cfg["tpot_limit"] * (c / args.curve_crossing) ** 1.5
            ttft = cfg["ttft_limit"] * 0.15
            print(
                f"  (dry-run) c={c} → TTFT {ttft:.0f} / TPOT {tpot:.0f}  {'PASS' if tpot<=cfg['tpot_limit'] else 'FAIL'}"
            )
        else:
            run_id = f"{args.run_prefix}-c{c}"
            run_dir = ROOT / "results" / run_id
            sh(
                [ROOT / "scripts/sweep.sh", cell, args.profile, run_id, "--rungs", str(c)]
            )  # blocks until the Job finishes
            sh([ROOT / "scripts/fetch_results.sh", cell, args.profile, run_id])
            sh(
                [
                    sys.executable,
                    ROOT / "analysis/llm-perf/aggregate_metrics.py",
                    run_dir,
                    "--out",
                    run_dir / "metrics_summary.csv",
                ]
            )
            lat = read_rung_latency(run_dir / "metrics_summary.csv", stat)
            if lat is None or lat[0] is None:
                sys.exit(f"adaptive_sweep: could not read latency for c={c} from {run_dir}")
            ttft, tpot = lat
            print(
                f"  c={c} → TTFT {ttft:.0f} / TPOT {tpot:.0f}  {'PASS' if (ttft<=cfg['ttft_limit'] and tpot<=cfg['tpot_limit']) else 'FAIL'}"
            )
        meas.append({"concurrency": c, "ttft": round(ttft, 1), "tpot": round(tpot, 1)})

    print(f"\nrungs run: {[m['concurrency'] for m in meas]}  ({len(meas)}, not the full grid)")
    if bracket:
        lo, hi = bracket
        # interpolate the crossing from the bracket (same math as exemplar_check)
        p_lo = next(m for m in meas if m["concurrency"] == lo)
        p_hi = next(m for m in meas if m["concurrency"] == hi)

        def interp(vlo, vhi, lim):
            return hi if vhi == vlo else lo + (hi - lo) * (lim - vlo) / (vhi - vlo)

        ct = interp(p_lo["ttft"], p_hi["ttft"], cfg["ttft_limit"]) if p_hi["ttft"] > cfg["ttft_limit"] else float("inf")
        cp = interp(p_lo["tpot"], p_hi["tpot"], cfg["tpot_limit"]) if p_hi["tpot"] > cfg["tpot_limit"] else float("inf")
        crossing = min(ct, cp)
        print(
            f"interpolated max_concurrency_at_sla ≈ {crossing:.1f}  (bracket [{lo},{hi}] {hi/lo:.2f}×, "
            f"binding {'TTFT' if ct < cp else 'TPOT'})"
        )
        if not args.dry_run:
            print(
                f"\nNext: aggregate the {len(meas)} rungs into one CSV, then "
                f"`exemplar_check.py {cell} <combined.csv> --set` to commit the baseline."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

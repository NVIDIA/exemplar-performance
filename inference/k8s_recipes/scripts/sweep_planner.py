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

"""sweep_planner.py — deterministic next-rung planner for the max-concurrency-sla adaptive geometric-grid search.

STATELESS + PURE: given the adaptive_sweep config + the rungs measured so far, it returns the SINGLE next
concurrency to run (or done/stop). Same inputs → same output, so the whole search is reproducible and
comparable across clusters. An orchestrator loops: plan → run that one rung (sweep.sh --rungs) → aggregate →
feed the result back → plan … until `done`, then exemplar_check interpolates the crossing.

It's binary search over a fixed GEOMETRIC grid, seeded at the `start` prior: bisect the working bracket at
the GEOMETRIC midpoint. A wide bracket (we're far off) → a big leap; a narrow one (we're close) → a small
step — margin-proportional, but deterministic and snapped to grid values only. This is the algorithm pinned
in schema `adaptive_sweep`; keeping it standalone makes it unit-testable (no cluster) and is the seed for
folding into the runner later.

  echo '[{"concurrency":128,"ttft":900,"tpot":92}]' | sweep_planner.py \
       --start 128 --min 16 --max 512 --ttft-limit 10000 --tpot-limit 100 --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def build_grid(start, ratio, lo, hi):
    """Geometric ladder ANCHORED on `start` (so start is always a grid point), clamped to [lo, hi].
    The BOUNDARIES `lo`/`hi` are also anchors: the ceiling can sit right at the edge, so both bounds must be
    testable candidates. Without this the top ladder rung can fall short of `hi` (e.g. start=96, ratio=1.3 →
    …,96,125, and 125*1.3=162>128), so a pass at 125 with hi=128 wrongly reports `above_range` instead of
    probing 128. With hi as an anchor, `above_range` correctly means 'even at max it passes.'
    Guards against non-progressing steps: at small values round(v/ratio) can equal v (e.g. round(2/1.3)==2),
    which would loop forever — break the moment a step doesn't strictly move."""
    g = {int(start)}
    if lo <= hi:
        g.add(int(lo))  # min is always a candidate (so below_range means 'even at min it fails')
        g.add(int(hi))  # max is always a candidate (so above_range means 'even at max it passes')
    v = float(start)
    while True:
        nv = round(v * ratio)
        if nv <= v or nv > hi:
            break
        v = nv
        g.add(int(v))
    v = float(start)
    while True:
        nv = round(v / ratio)
        if nv >= v or nv < lo:
            break
        v = nv
        g.add(int(v))
    return sorted(x for x in g if lo <= x <= hi)


def _snap(cands, target):
    return min(cands, key=lambda g: abs(g - target)) if cands else None


def _geomean(a, b):
    return math.sqrt(a * b)


def plan(cfg, meas):
    """Return the next action. cfg: dict(start,ratio,min,max,bracket_tolerance,max_runs,ttft_limit,tpot_limit).
    meas: list of {concurrency, ttft, tpot} already run. Returns {action, ...}."""
    grid = build_grid(cfg["start"], cfg["ratio"], cfg["min"], cfg["max"])
    if not grid:
        return {"action": "stop", "reason": "empty_grid", "note": "min/max/start produce no grid rungs"}

    def passed(m):
        return m["ttft"] <= cfg["ttft_limit"] and m["tpot"] <= cfg["tpot_limit"]

    run_c = {m["concurrency"] for m in meas}
    passes = sorted(m["concurrency"] for m in meas if passed(m))
    fails = sorted(m["concurrency"] for m in meas if not passed(m))

    # start fresh: run the prior
    if not meas:
        return {"action": "run", "concurrency": _snap(grid, cfg["start"]), "phase": "seed"}

    lo = max(passes) if passes else grid[0]  # highest pass (working lower bound)
    hi = min(fails) if fails else grid[-1]  # lowest fail (working upper bound)

    # a real pass<fail bracket exists → bisect it (or finish)
    if passes and fails and lo < hi:
        if hi / lo <= cfg["bracket_tolerance"]:
            return {"action": "done", "bracket": [lo, hi], "reason": "bracket_tight"}
        cand = [g for g in grid if lo < g < hi and g not in run_c]
        if not cand:
            return {"action": "done", "bracket": [lo, hi], "reason": "grid_exhausted_in_bracket"}
        if len(meas) >= cfg["max_runs"]:
            return {"action": "done", "bracket": [lo, hi], "reason": "max_runs"}
        return {"action": "run", "concurrency": _snap(cand, _geomean(lo, hi)), "phase": "bisect"}

    if len(meas) >= cfg["max_runs"]:
        return {"action": "stop", "reason": "max_runs", "note": "hit max_runs before bracketing"}

    # all pass so far → probe UP toward a fail (binary search up)
    if not fails:
        cand = [g for g in grid if g > lo and g not in run_c]
        if not cand:
            return {
                "action": "stop",
                "reason": "above_range",
                "note": f"all rungs pass up to {lo} — ceiling is above max={cfg['max']}; raise max",
            }
        return {"action": "run", "concurrency": _snap(cand, _geomean(lo, grid[-1])), "phase": "probe_up"}

    # all fail so far → probe DOWN toward a pass
    if not passes:
        cand = [g for g in grid if g < hi and g not in run_c]
        if not cand:
            return {
                "action": "stop",
                "reason": "below_range",
                "note": f"all rungs fail down to {hi} — ceiling is below min={cfg['min']}; lower min",
            }
        return {"action": "run", "concurrency": _snap(cand, _geomean(grid[0], hi)), "phase": "probe_down"}

    # non-monotonic (a pass above a fail): report the best we have
    return (
        {"action": "done", "bracket": [lo, hi], "reason": "non_monotonic"}
        if lo < hi
        else {"action": "stop", "reason": "non_monotonic", "note": "no coherent pass<fail bracket"}
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic next-rung planner (max-concurrency-sla geometric grid).")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--min", type=int, default=1)
    ap.add_argument("--max", type=int, required=True)
    ap.add_argument("--ratio", type=float, default=1.3)
    ap.add_argument("--bracket-tolerance", type=float, default=1.3)
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--ttft-limit", type=float, required=True)
    ap.add_argument("--tpot-limit", type=float, required=True)
    ap.add_argument("--measurements", help="JSON list of {concurrency,ttft,tpot}; default reads stdin")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    raw = args.measurements if args.measurements is not None else (sys.stdin.read() or "[]")
    meas = json.loads(raw)
    cfg = {
        "start": args.start,
        "min": args.min,
        "max": args.max,
        "ratio": args.ratio,
        "bracket_tolerance": args.bracket_tolerance,
        "max_runs": args.max_runs,
        "ttft_limit": args.ttft_limit,
        "tpot_limit": args.tpot_limit,
    }
    out = plan(cfg, meas)
    if args.json:
        print(json.dumps(out))
    elif out["action"] == "run":
        print(f"NEXT rung: {out['concurrency']}  ({out['phase']})")
    else:
        print(f"{out['action'].upper()}: {out.get('reason')}  {out.get('bracket', '') or out.get('note', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

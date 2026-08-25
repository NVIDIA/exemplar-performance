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

"""fleet_status.py <cell...> [--json] — read-only fleet progress dashboard over runs.jsonl.

The coordinator's one-look "how is the fleet doing" view while N agents run in parallel. For each cell it
reports, purely from committed data (recipe.yaml exemplar + runs.jsonl ledger):
  - the exemplar metric,
  - how many VALUED runs exist (value-captured publishes; old stub entries without a value don't count),
  - per-cluster within-cluster reproducibility spread % (reusing compare.runs_within_cluster — same math the
    coordinator analysis needs: same cluster, N repeats → variance %),
  - the latest value,
  - a marker: ✅ reproducible (≥2 valued runs within tolerance) · 🟡 has data (<2 runs or single) · ⬜ no value yet.
Ends with a roll-up: cells with data, cells reproducible, and a hint toward `compare --all` for cross-hardware.

Never touches the cluster or git; reads only the given cells' files. Pass the fleet's cell dirs, e.g.
  fleet_status.py recipes/llm-perf/256k/nemotron-ultra-3-gb300-vllm-agg{,-pareto} recipes/llm-perf/1m/nemotron-ultra-3-gb300-vllm-agg
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import compare as _cmp  # reuse runs_within_cluster + HIGHER_BETTER (single source of the spread math)

try:
    import yaml
except ImportError:
    sys.exit("fleet_status: pyyaml required")


def load_runs(cell: Path) -> list:
    jl = cell / "runs.jsonl"
    if not jl.is_file():
        return []
    return [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]


def cell_label(cell: Path) -> str:
    """Human-readable, path-aware label; leaf recipe names are not unique across contexts."""
    try:
        return str(cell.relative_to(ROOT / "recipes"))
    except ValueError:
        try:
            return str(cell.relative_to(ROOT))
        except ValueError:
            return str(cell)


def cell_row(cell: Path) -> dict:
    """PURE-ish (reads two files) — one cell's fleet-status summary dict."""
    rec = (yaml.safe_load((cell / "recipe.yaml").read_text()) or {}) if (cell / "recipe.yaml").is_file() else {}
    env = rec.get("envelope") or {}
    ex = env.get("exemplar") or {}
    metric = ex.get("metric", "?")
    tol = float(ex.get("tolerance_pct", 5))
    hib = _cmp.HIGHER_BETTER.get(metric, True)
    runs = load_runs(cell)
    valued = [r for r in runs if r.get("value") is not None and (r.get("cluster") or "").strip()]
    # within-cluster reproducibility (per cluster with ≥2 valued runs)
    _, wobj = _cmp.runs_within_cluster(valued, metric, tol, hib)
    clusters = (wobj or {}).get("clusters", [])
    reproducible = bool(clusters) and all(c["reproducible"] for c in clusters)
    best_spread = min((c["spread_pct"] for c in clusters), default=None)
    latest = valued[-1]["value"] if valued else None
    if reproducible:
        mark = "✅ reproducible"
    elif valued:
        mark = "🟡 has data"
    else:
        mark = "⬜ no value yet"
    return {
        "cell": cell.name,
        "label": cell_label(cell),
        "metric": metric,
        "n_valued": len(valued),
        "clusters": [c["cluster"] for c in clusters] or sorted({r["cluster"] for r in valued}),
        "latest": latest,
        "spread_pct": best_spread,
        "reproducible": reproducible,
        "mark": mark,
        "status": (env.get("status") or "?"),
    }


def roll_up(rows: list) -> dict:
    with_data = [r for r in rows if r["n_valued"] > 0]
    return {
        "cells": len(rows),
        "with_data": len(with_data),
        "reproducible": sum(1 for r in rows if r["reproducible"]),
        "no_value": sum(1 for r in rows if r["n_valued"] == 0),
    }


def main() -> int:
    argv = sys.argv[1:]
    as_json = "--json" in argv
    cells = [Path(a).resolve() for a in argv if not a.startswith("--")]
    cells = [c for c in cells if (c / "recipe.yaml").is_file()]
    if not cells:
        sys.exit(__doc__)
    rows = [cell_row(c) for c in cells]
    ru = roll_up(rows)
    if as_json:
        print(json.dumps({"rows": rows, "rollup": ru}, indent=2))
        return 0
    print("── fleet status ───────────────────────────────────────────────")
    label_w = max(44, *(len(r.get("label") or r["cell"]) for r in rows))
    print(f"  {'cell':<{label_w}} {'metric':<22} runs  spread   latest      state")
    for r in rows:
        sp = f"{r['spread_pct']:.1f}%" if r["spread_pct"] is not None else "  —"
        lv = f"{r['latest']:.4g}" if isinstance(r["latest"], (int, float)) else "—"
        label = r.get("label") or r["cell"]
        print(f"  {label:<{label_w}} {r['metric']:<22} {r['n_valued']:>3}  {sp:>6}   {lv:<10}  {r['mark']}")
    print(
        f"  ── {ru['with_data']}/{ru['cells']} cells have a value · {ru['reproducible']} reproducible · "
        f"{ru['no_value']} awaiting first value"
    )
    print(
        "  cross-hardware: run `llmb-k8s compare --all` once ≥2 clusters share a facet; "
        "per-cell within-cluster detail: `llmb-k8s compare --repro <cell>`"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

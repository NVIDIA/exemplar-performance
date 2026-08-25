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

"""aggregate_cell.py <cell> [--out FILE] — the deterministic CROSS-RUN aggregate (docs/OUTPUT-DATA-PIPELINE §4).

Multiple runs of a cell are first-class: rather than let a new run silently overwrite history, this computes a
persisted, reproducible band over ALL runs in `runs/index.jsonl` and writes `aggregate/aggregate.json`:

  headline_band : {metric, median, min, max, spread_pct, n_runs, per_cluster:{cluster:{median,min,max,n}}}
  per_rung_band : one row per concurrency; per numeric rung column a {mean,min,max,spread_pct,n} band across
                  runs (reusing export_record._agg_stats / merge_rung_repeats — never a reinvented stat).
  chosen_run    : the run whose scalar metric == the published record.json value (a traceability pointer).

Same committed inputs → byte-identical aggregate.json + aggregate/charts/<goal>.png (charts.render_png is a
pure function of the record + runs). A single-run cell yields a well-formed n=1 band (honest: no fabricated
spread). REUSES the existing stat kernels; adds no new statistics.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
import export_record as _er  # noqa: E402 — reuse _agg_stats / merge_rung_repeats (do not reinvent)
import charts as _charts  # noqa: E402 — reuse render_png / _scalar_band
import goal_handlers as _gh  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("aggregate_cell: requires pyyaml")


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def read_index(cell: Path) -> list:
    """runs/index.jsonl → the list of run entries (append-only ledger). Empty when the cell has no runs."""
    idx = cell / "runs" / "index.jsonl"
    out = []
    if idx.is_file():
        for line in idx.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def read_rungs(cell: Path, entry: dict) -> list:
    """A run's normalized per-rung rows from its committed curated/rungs.csv (goal-agnostic schema)."""
    curated = cell / (entry.get("curated") or f"runs/{entry.get('run_id')}/curated")
    return _gh._read_csv(curated / "rungs.csv")


def _scalar_band(values: list) -> dict | None:
    """{median,min,max,spread_pct,n} over the non-None run values. Reuses export_record._agg_stats for
    min/max/spread/n (≥2); adds the median the headline reports; n=1 is honest (spread 0)."""
    vals = [v for v in (_f(x) for x in values) if v is not None]
    if not vals:
        return None
    med = round(statistics.median(vals), 6)
    st = _er._agg_stats(vals)
    if st is None:  # n<2 — honest single-value band, no fabricated spread
        return {"median": med, "min": round(vals[0], 6), "max": round(vals[0], 6), "spread_pct": 0.0, "n": len(vals)}
    return {"median": med, "min": st["min"], "max": st["max"], "spread_pct": st["spread_pct"], "n": st["n"]}


def headline_band(cell: Path, runs: list, metric: str) -> dict:
    vals = [r.get("value") for r in runs if r.get("value") is not None]
    band = _scalar_band(vals) or {"median": None, "min": None, "max": None, "spread_pct": 0.0, "n": 0}
    per_cluster = {}
    for cluster in sorted({(r.get("cluster") or "unknown") for r in runs}):
        cvals = [
            r.get("value") for r in runs if (r.get("cluster") or "unknown") == cluster and r.get("value") is not None
        ]
        cb = _scalar_band(cvals)
        if cb:
            per_cluster[cluster] = {k: cb[k] for k in ("median", "min", "max", "n")}
    return {
        "metric": metric,
        "median": band["median"],
        "min": band["min"],
        "max": band["max"],
        "spread_pct": band["spread_pct"],
        "n_runs": len(runs),
        "per_cluster": per_cluster,
    }


def _numeric_cols(handler) -> list:
    """The rung columns that carry a number (bandable). Excludes concurrency (the key) + label columns."""
    skip = {"concurrency", "publishable", "invalid_reason", "n_trials"}
    return [k for k in handler.rung_keys if k not in skip]


def per_rung_band(cell: Path, runs: list, handler) -> list:
    """For each concurrency, a per-column band across runs. Reuses export_record.merge_rung_repeats (the exact
    per-rung-across-legs aggregation the record/charts already use) to derive {mean,min,max,spread_pct,n}."""
    legs = [read_rungs(cell, r) for r in runs]
    cols = _numeric_cols(handler)
    concs = sorted({int(_f(r.get("concurrency"))) for leg in legs for r in leg if _f(r.get("concurrency")) is not None})
    base = [{"concurrency": c} for c in concs]
    merged = _er.merge_rung_repeats(base, legs, cols)  # attaches rung['repeats'][col] = {mean,min,max,spread,n} (n≥2)
    out = []
    for rung in merged:
        c = int(_f(rung.get("concurrency")))
        row = {"concurrency": c}
        reps = rung.get("repeats") or {}
        for col in cols:
            if col in reps:  # n≥2 band from merge_rung_repeats
                row[col] = reps[col]
            else:  # n=1 (or 0): honest single-value band, no fabricated spread
                vals = [
                    _f(r.get(col))
                    for leg in legs
                    for r in leg
                    if _f(r.get("concurrency")) == c and _f(r.get(col)) is not None
                ]
                if vals:
                    row[col] = {
                        "mean": round(vals[0], 6),
                        "min": round(vals[0], 6),
                        "max": round(vals[0], 6),
                        "spread_pct": 0.0,
                        "n": len(vals),
                    }
        out.append(row)
    return out


def build(cell: Path) -> dict:
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    env = recipe.get("envelope") or {}
    handler = _gh.resolve(env.get("scenario"), env.get("goal"))
    runs = read_index(cell)
    record = {}
    rp = cell / "record.json"
    if rp.is_file():
        record = json.loads(rp.read_text())
    metric = (record.get("result") or {}).get("metric") or handler.metric
    pub_value = (record.get("result") or {}).get("value")
    chosen = next((r.get("run_id") for r in runs if r.get("value") == pub_value), None)
    return {
        "aggregate_schema_version": 1,
        "cell": env.get("name"),
        "scenario": env.get("scenario"),
        "goal": env.get("goal"),
        "n_runs": len(runs),
        "headline_band": headline_band(cell, runs, metric),
        "per_rung_band": per_rung_band(cell, runs, handler),
        "chosen_run": chosen,
    }


def write_charts(cell: Path) -> str | None:
    """Best-effort deterministic aggregate PNG under aggregate/charts/<goal>.png, rendered from the committed
    record.json + the runs ledger (so it carries the cross-run scalar band). Byte-stable across re-renders."""
    rp = cell / "record.json"
    if not rp.is_file():
        return None
    record = json.loads(rp.read_text())
    runs = read_index(cell)
    ident = record.get("identity") or {}
    goal = ident.get("goal") or ident.get("scenario") or "chart"
    safe = str(goal).replace("/", "-").replace(" ", "_")
    png = cell / "aggregate" / "charts" / f"{safe}.png"
    if _charts.render_png(record, png, runs):
        return f"aggregate/charts/{safe}.png"
    return None


def main() -> int:
    argv = sys.argv[1:]
    pos = [a for a in argv if not a.startswith("--")]
    if not pos:
        sys.exit(__doc__)
    cell = Path(pos[0]).resolve()
    agg = build(cell)
    out = Path(next((argv[i + 1] for i, a in enumerate(argv) if a == "--out"), cell / "aggregate" / "aggregate.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2) + "\n")
    png = write_charts(cell)
    print(
        f"[aggregate_cell] {cell.name}: {agg['n_runs']} run(s) → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}"
        + (f" · {png}" if png else " · (chart skipped)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

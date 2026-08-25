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

"""selftest_aggregate_cell.py — offline guards for the deterministic cross-run aggregate (aggregate_cell.py).

No cluster, no network. Covers:
  A. HEADLINE BAND — median/min/max/spread_pct/n_runs over the runs/index.jsonl values, split per_cluster.
  B. PER-RUNG BAND — across runs, each concurrency's columns carry {mean,min,max,spread_pct,n} (reusing
     export_record._agg_stats / merge_rung_repeats); a rung seen once is honest n=1 (no fabricated spread).
  C. CHOSEN RUN — the pointer to the run whose scalar metric == the published record.json value.
  D. DETERMINISM — same committed inputs → byte-identical aggregate.json across rebuilds.
  E. REUSE — aggregate_cell calls the existing stat kernels, not a reinvented statistic.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
import aggregate_cell as ac  # noqa: E402
import export_record as er  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


RECIPE = """\
envelope:
  name: st-agg-pareto
  model: m
  gpu_type: B200
  arch: amd64
  engine: vllm
  serving_mode: aggregated
  framework: none
  scenario: llm-perf
  distribution: d
  mode: mooncake-trace
  launcher: aiperf
  goal: pareto
  status: runs
  exemplar: { metric: pareto_geomean, unit: u, reference: 100.0, tolerance_pct: 5 }
serving: { tp: 8 }
bench: { sla: { ttft_ms: 10000, tpot_ms: 100, stop_stat: p50 }, sweep_concurrency: [32, 64] }
"""


def _rungs_csv(cell: Path, run_id: str, rungs: list):
    cur = cell / "runs" / run_id / "curated"
    cur.mkdir(parents=True, exist_ok=True)
    hdr = "concurrency,ttft_p50_ms,tpot_p50_ms,tps_per_gpu,tps_per_user,error_rate_pct"
    lines = [hdr]
    for r in rungs:
        lines.append(
            ",".join(
                "" if v is None else (repr(v) if isinstance(v, float) else str(v))
                for v in [
                    r["concurrency"],
                    r["ttft_p50_ms"],
                    r["tpot_p50_ms"],
                    r["tps_per_gpu"],
                    r["tps_per_user"],
                    r.get("error_rate_pct"),
                ]
            )
        )
    (cur / "rungs.csv").write_text("\n".join(lines) + "\n")


with tempfile.TemporaryDirectory() as _td:
    cell = Path(_td) / "cell"
    cell.mkdir()
    (cell / "recipe.yaml").write_text(RECIPE)
    # two runs on the same cluster: values 100.0 and 110.0 → median 105, spread 10/105*100
    _rungs_csv(
        cell,
        "rA",
        [
            {
                "concurrency": 32,
                "ttft_p50_ms": 380.0,
                "tpot_p50_ms": 14.0,
                "tps_per_gpu": 96.0,
                "tps_per_user": 68.0,
                "error_rate_pct": None,
            },
            {
                "concurrency": 64,
                "ttft_p50_ms": 460.0,
                "tpot_p50_ms": 24.0,
                "tps_per_gpu": 124.0,
                "tps_per_user": 40.0,
                "error_rate_pct": None,
            },
        ],
    )
    _rungs_csv(
        cell,
        "rB",
        [
            {
                "concurrency": 32,
                "ttft_p50_ms": 400.0,
                "tpot_p50_ms": 15.0,
                "tps_per_gpu": 98.0,
                "tps_per_user": 70.0,
                "error_rate_pct": None,
            },
            {
                "concurrency": 64,
                "ttft_p50_ms": 470.0,
                "tpot_p50_ms": 25.0,
                "tps_per_gpu": 126.0,
                "tps_per_user": 42.0,
                "error_rate_pct": None,
            },
        ],
    )
    idx = cell / "runs" / "index.jsonl"
    idx.write_text(
        json.dumps(
            {
                "run_id": "rA",
                "cluster": "example-gpu-cluster",
                "metric": "pareto_geomean",
                "value": 100.0,
                "data_provenance": "archived",
                "curated": "runs/rA/curated",
            }
        )
        + "\n"
        + json.dumps(
            {
                "run_id": "rB",
                "cluster": "example-gpu-cluster",
                "metric": "pareto_geomean",
                "value": 110.0,
                "data_provenance": "archived",
                "curated": "runs/rB/curated",
            }
        )
        + "\n"
    )
    (cell / "record.json").write_text(
        json.dumps(
            {
                "identity": {
                    "cell": "st-agg-pareto",
                    "scenario": "llm-perf",
                    "goal": "pareto",
                },
                "result": {"metric": "pareto_geomean", "value": 110.0},
                "detail": {"rungs": []},
            }
        )
        + "\n"
    )

    agg = ac.build(cell)
    hb = agg["headline_band"]
    # A. headline band
    check(
        "A headline band median = 105.0 over the two runs",
        hb["median"] == 105.0,
        str(hb["median"]),
    )
    check("A headline band min/max = 100/110", hb["min"] == 100.0 and hb["max"] == 110.0)
    check("A headline band n_runs = 2", hb["n_runs"] == 2)
    check(
        "A headline spread_pct ≈ 9.52% ((110-100)/105*100)",
        abs(hb["spread_pct"] - 9.5238) < 1e-3,
        str(hb["spread_pct"]),
    )
    check(
        "A per_cluster groups the single cluster (n=2)",
        hb["per_cluster"].get("example-gpu-cluster", {}).get("n") == 2,
    )
    # B. per-rung band across runs
    r32 = next(r for r in agg["per_rung_band"] if r["concurrency"] == 32)
    tpg = r32["tps_per_gpu"]
    check(
        "B per-rung band tps_per_gpu@c32 = {mean 97, min 96, max 98, n 2}",
        tpg["mean"] == 97.0 and tpg["min"] == 96.0 and tpg["max"] == 98.0 and tpg["n"] == 2,
        str(tpg),
    )
    check(
        "B per-rung band matches export_record._agg_stats (reuse, not reinvented)",
        {k: tpg[k] for k in ("mean", "min", "max", "n")}
        == {k: er._agg_stats([96.0, 98.0])[k] for k in ("mean", "min", "max", "n")},
    )
    # C. chosen run = the run whose value == published record value (110.0 → rB)
    check(
        "C chosen_run points at the run matching the published record value",
        agg["chosen_run"] == "rB",
    )
    # D. determinism
    check(
        "D aggregate.json deterministic across rebuilds",
        json.dumps(ac.build(cell)) == json.dumps(agg),
    )

# single-run honesty: n=1 band has no fabricated spread
with tempfile.TemporaryDirectory() as _td:
    cell = Path(_td) / "cell"
    cell.mkdir()
    (cell / "recipe.yaml").write_text(RECIPE)
    _rungs_csv(
        cell,
        "r1",
        [
            {
                "concurrency": 32,
                "ttft_p50_ms": 380.0,
                "tpot_p50_ms": 14.0,
                "tps_per_gpu": 96.0,
                "tps_per_user": 68.0,
                "error_rate_pct": None,
            }
        ],
    )
    (cell / "runs" / "index.jsonl").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "cluster": "c",
                "metric": "pareto_geomean",
                "value": 42.0,
                "data_provenance": "reconstructed_from_record",
                "curated": "runs/r1/curated",
            }
        )
        + "\n"
    )
    (cell / "record.json").write_text(
        json.dumps(
            {
                "identity": {"scenario": "llm-perf", "goal": "pareto"},
                "result": {"metric": "pareto_geomean", "value": 42.0},
                "detail": {"rungs": []},
            }
        )
        + "\n"
    )
    agg1 = ac.build(cell)
    b = agg1["headline_band"]
    check(
        "n=1 headline band: median=min=max, spread 0 (no fabricated spread)",
        b["median"] == b["min"] == b["max"] == 42.0 and b["spread_pct"] == 0.0,
    )
    r = agg1["per_rung_band"][0]["tps_per_gpu"]
    check(
        "n=1 per-rung band: n=1, spread 0 (honest)",
        r["n"] == 1 and r["spread_pct"] == 0.0,
    )

print(("\nFAIL: " + ", ".join(fails)) if fails else "\nselftest_aggregate_cell: all checks passed")
sys.exit(1 if fails else 0)

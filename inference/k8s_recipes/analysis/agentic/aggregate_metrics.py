#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aggregate_metrics.py — the agentic (TaskTrace) aggregator: a fetched run directory → tasktrace_summary.csv.

One row per (rung, container) replay, from its summary.json and cpu_regime.json, for analysis/goal_handlers.py.
Each row's verdict is the shared Slurm gate's (tasktrace_recipe.summary_ok); nothing is re-judged here.

    aggregate_metrics.py <run-dir> [--out PATH]      # default: <run-dir>/tasktrace_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # inference/k8s_recipes
REPO = ROOT.parents[1]
RECIPE_MODULE = REPO / "inference/slurm_recipes/agentic/common/tasktrace_recipe.py"
SUMMARY_CSV = "tasktrace_summary.csv"

COLUMNS = [
    "rung",
    "container",
    "verdict",
    "fidelity",
    "replay_tool_time_ratio",
    "replay_to_live_tool_ratio",
    "replay_tool_busy_sec",
    "live_tool_busy_sec",
    "actions_compared",
    "actions_recorded",
    "actions_executed",
    "source_outcome_mismatch_count",
    "replay_wall_sec",
    "tasktrace_exit_status",
    "capture_exit_status",
    "capture_reward",
    "image_identity_match",
    "cpu_usage_sec",
    "cpu_wall_sec",
    "cpu_average_logical_cores_used",
    "cpu_utilization_percent_of_allocation",
    "cpu_allocated_logical_cpus",
    "cpu_throttled_sec",
    "cpu_valid_for_average_cpu_reporting",
    "cpu_regime",
    "pinned_cpuset",
    "diagnostics",
]


def _recipe_module():
    spec = importlib.util.spec_from_file_location("tasktrace_recipe", RECIPE_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def find_containers(run_dir: Path):
    """(rung, container, dir) for every replay rung K ran, c<K>/0 to c<K>/<K-1>, with or without a summary.json."""
    found = []
    for rung_dir in sorted(run_dir.glob("c*")):
        m = re.fullmatch(r"c(\d+)", rung_dir.name)
        if not m or not rung_dir.is_dir():
            continue
        rung = int(m.group(1))
        indices = set(range(rung)) | {int(p.name) for p in rung_dir.iterdir() if p.is_dir() and p.name.isdigit()}
        found.extend((rung, index, rung_dir / str(index)) for index in sorted(indices))
    return found


def row_for(rung: int, index: int, cdir: Path, recipe_module) -> dict:
    summary = _read_json(cdir / "summary.json")
    regime = _read_json(cdir / "cpu_regime.json")
    comparison = summary.get("comparison") or {}
    replay = summary.get("replay") or {}
    if isinstance(replay, dict) and "summary" in replay:  # capture+replay nests the replay report under replay.summary
        replay_summary = replay.get("summary") or {}
        replay_exit = replay.get("tasktrace_exit_status")
    else:  # replay-only: the replay report IS summary.replay
        replay_summary = replay if isinstance(replay, dict) else {}
        replay_exit = summary.get("tasktrace_exit_status")
    capture = summary.get("capture") or {}
    cpu = summary.get("cpu") or {}
    identity = summary.get("image_identity_check")
    ratio, _ = recipe_module.tool_time_ratio(comparison)  # capture+replay only: replay over its own capture
    return {
        "rung": rung,
        "container": index,
        "verdict": "PASS" if recipe_module.summary_ok(summary) else "FAIL",
        "fidelity": summary.get("fidelity"),
        "replay_tool_time_ratio": _num(ratio),
        "replay_to_live_tool_ratio": _num(comparison.get("replay_to_live_tool_ratio")),
        "replay_tool_busy_sec": _num(comparison.get("tool_busy_replay_sec")),
        "live_tool_busy_sec": _num(comparison.get("tool_busy_live_sec")),
        "actions_compared": _num(comparison.get("actions_compared")),
        "actions_recorded": _num(replay_summary.get("actions_recorded")),
        "actions_executed": _num(replay_summary.get("actions_executed")),
        "source_outcome_mismatch_count": _num(replay_summary.get("source_outcome_mismatch_count")),
        "replay_wall_sec": _num(replay_summary.get("wall_time_sec")),
        "tasktrace_exit_status": _num(replay_exit),
        "capture_exit_status": _num(capture.get("tasktrace_capture_exit_status")),
        "capture_reward": _num(capture.get("reward")),
        "image_identity_match": (identity or {}).get("match") if isinstance(identity, dict) else None,
        "cpu_usage_sec": _num(cpu.get("usage_sec")),
        "cpu_wall_sec": _num(cpu.get("wall_sec")),
        "cpu_average_logical_cores_used": _num(cpu.get("average_logical_cores_used")),
        "cpu_utilization_percent_of_allocation": _num(cpu.get("utilization_percent_of_allocation")),
        "cpu_allocated_logical_cpus": _num(cpu.get("allocated_logical_cpus")),
        "cpu_throttled_sec": _num((cpu.get("throttling") or {}).get("throttled_sec")),
        "cpu_valid_for_average_cpu_reporting": cpu.get("valid_for_average_cpu_reporting"),
        "cpu_regime": regime.get("regime"),
        "pinned_cpuset": regime.get("pinned_cpuset")
        or summary.get("cpuset")
        or (replay.get("cpuset") if isinstance(replay, dict) else None),
        "diagnostics": (
            " | ".join(str(d) for d in (summary.get("diagnostics") or []))
            if summary
            else "no summary.json: the replay did not finish"
        ),
    }


def aggregate(run_dir: Path, recipe_module=None) -> list:
    recipe_module = recipe_module or _recipe_module()
    return [row_for(rung, index, cdir, recipe_module) for rung, index, cdir in find_containers(run_dir)]


def write_csv(rows: list, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "run_dir", help="a fetched run dir (results/agentic/<goal>/<cell>/<run-id>/) holding c<K>/<i>/summary.json"
    )
    ap.add_argument("--out", help=f"CSV path (default: <run-dir>/{SUMMARY_CSV})")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / SUMMARY_CSV
    rows = aggregate(run_dir)
    write_csv(rows, out)
    if not rows:
        print(
            f"aggregate_metrics: no c<K>/<i>/summary.json under {run_dir} (empty CSV written to {out})", file=sys.stderr
        )
        return 1
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    print(
        f"aggregate_metrics: {len(rows)} container replay(s) over {len({r['rung'] for r in rows})} rung(s): {n_pass} PASS, {len(rows) - n_pass} FAIL -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

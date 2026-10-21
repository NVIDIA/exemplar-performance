#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

"""Resolve a cell scenario to its staging and benchmark scripts.

For ``llm-perf`` cells, the lane uses ``stage-dataset.sh`` and ``sweep.sh``. Shell callers use
``lane.py <cell-dir> <stage|bench|needs|kind>``; an unknown lane exits with status 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scenario → mode → lane. The None mode key means "this scenario ignores envelope.mode"
# (llm-perf's mode is a distribution detail, not a lane selector).
# `kind` = the Job-name infix `<cell>-<kind>-<run_id>` this lane's bench script uses (sweep.sh → bench).
# run.sh feeds it to run_id.py --fit so the default run-id is sized to keep the Job name ≤63-char DNS-label.
# `server` = `deploy` (an in-cluster model server) or `none`, for which run.sh skips every server phase. An
# agentic cell that declares a `serving` block (the -gpu cells) deploys one anyway; see cell_lane.
_LANES: dict[str, dict[str | None, dict[str, str]]] = {
    "llm-perf": {
        None: {
            "stage": "stage-dataset.sh",
            "bench": "sweep.sh",
            "needs": "",
            "kind": "bench",
            "server": "deploy",
        },
    },
    # agentic (TaskTrace): the Kubernetes twin of inference/slurm_recipes/agentic/, with no model server unless
    # the cell declares one (cell_lane).
    "agentic": {
        "tasktrace-replay": {
            "stage": "stage-tasktrace.sh",
            "bench": "tasktrace.sh",
            "needs": "",
            "kind": "tt",
            "server": "none",
        },
        "tasktrace-capture-replay": {
            "stage": "stage-tasktrace.sh",
            "bench": "tasktrace.sh",
            "needs": "",
            "kind": "tt",
            "server": "none",
        },
    },
}


def resolve_lane(scenario: str, mode: str | None) -> dict[str, str]:
    """(scenario, mode) → {stage, bench, needs, kind, server}. Raises SystemExit on an unknown scenario/mode."""
    modes = _LANES.get(scenario)
    if modes is None:
        raise SystemExit(f"lane.py: unknown scenario '{scenario}' (known: {', '.join(_LANES)})")
    if None in modes:  # scenario ignores mode
        return modes[None]
    lane = modes.get(mode)
    if lane is None:
        known = ", ".join(sorted(k for k in modes if k))
        raise SystemExit(f"lane.py: unknown mode '{mode}' for scenario '{scenario}' (known: {known})")
    return lane


def cell_lane(cell) -> dict[str, str]:
    """Load a cell's recipe and resolve its lane. A cell that declares a serving stack deploys its server."""
    import yaml

    recipe = yaml.safe_load((Path(cell) / "recipe.yaml").read_text()) or {}
    e = recipe.get("envelope") or {}
    lane = resolve_lane(e.get("scenario"), e.get("mode"))
    if lane["server"] == "none" and (recipe.get("serving") or {}).get("stack"):
        lane = {**lane, "server": "deploy"}
    return lane


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.exit("usage: lane.py <cell-dir> <stage|bench|needs|kind|server>")
    cell, key = argv
    lane = cell_lane(cell)
    if key not in lane:
        sys.exit(f"lane.py: unknown key '{key}' (stage|bench|needs|kind|server)")
    print(lane[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

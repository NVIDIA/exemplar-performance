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
_LANES: dict[str, dict[str | None, dict[str, str]]] = {
    "llm-perf": {
        None: {
            "stage": "stage-dataset.sh",
            "bench": "sweep.sh",
            "needs": "",
            "kind": "bench",
        },
    },
}


def resolve_lane(scenario: str, mode: str | None) -> dict[str, str]:
    """(scenario, mode) → {stage, bench, needs}. Raises SystemExit on an unknown scenario/mode."""
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
    """Load a cell's recipe and resolve its lane."""
    import yaml

    e = (yaml.safe_load((Path(cell) / "recipe.yaml").read_text()) or {}).get("envelope") or {}
    return resolve_lane(e.get("scenario"), e.get("mode"))


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.exit("usage: lane.py <cell-dir> <stage|bench|needs|kind>")
    cell, key = argv
    lane = cell_lane(cell)
    if key not in lane:
        sys.exit(f"lane.py: unknown key '{key}' (stage|bench|needs|kind)")
    print(lane[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

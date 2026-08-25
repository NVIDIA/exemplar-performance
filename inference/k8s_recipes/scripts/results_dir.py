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

"""Resolve the canonical artifact directory for a run.

Artifacts are stored under ``results/<scenario>/<goal>/<cell-name>/<run_id>/``. The shell and Python
callers use this module so they construct identical paths. A missing goal maps to ``default``.

CLI: ``python3 scripts/results_dir.py <cell-dir> <run_id>`` prints the relative path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def default_goal(scenario: str | None, goal: str | None) -> str:
    """The goal LABEL used in the path — the SAME defaulting as export_dataset.lane_key: a null goal maps to
    'default' for every scenario."""
    return goal or "default"


def results_subdir(scenario: str, goal: str | None, name: str) -> Path:
    """Pure: (scenario, goal, cell-name) → the nested run-dir PARENT `results/<scenario>/<goal>/<name>` (no
    recipe read, no run-id) — unit-testable without touching disk."""
    return Path("results") / scenario / default_goal(scenario, goal) / name


def results_dir(cell_dir, run_id: str) -> Path:
    """The RELATIVE nested path `results/<scenario>/<goal>/<name>/<run_id>` for one run of `cell_dir`.
    Reads the cell's recipe.yaml envelope (scenario, goal, name) and applies the goal default.
    """
    import yaml

    env = (yaml.safe_load((Path(cell_dir) / "recipe.yaml").read_text()) or {}).get("envelope") or {}
    scenario, goal, name = env.get("scenario"), env.get("goal"), env.get("name")
    if not scenario or not name:
        raise SystemExit(
            f"results_dir.py: {cell_dir}/recipe.yaml envelope is missing scenario/name "
            f"(scenario={scenario!r}, name={name!r}) — cannot build the nested results path"
        )
    return results_subdir(scenario, goal, name) / run_id


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.exit("usage: results_dir.py <cell-dir> <run_id>")
    cell, run_id = argv
    print(results_dir(cell, run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

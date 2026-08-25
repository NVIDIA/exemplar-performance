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

"""Offline regression tests for the immutable pre-launch recipe-hash receipt."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import launch_attestation as la  # noqa: E402
import recipe_hash as rh  # noqa: E402

fails: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(label)


with tempfile.TemporaryDirectory() as td:
    cell = Path(td) / "cell"
    cell.mkdir()
    recipe = cell / "recipe.yaml"
    recipe.write_text("envelope: {name: launch-fixture}\nbench: {synthetic: {isl: 1, osl: 1}}\n")
    receipt_path = Path(td) / "results" / "r1" / "launch_attestation.json"
    receipt = la.capture(cell, "r1", receipt_path, captured_at_utc="2026-08-06T00:00:00Z")
    saved = json.loads(receipt_path.read_text())
    check(
        "captures the current recipe hash before launch",
        saved == receipt and saved["recipe_hash"] == rh.recipe_hash(cell),
    )
    check(
        "receipt is explicitly marked as an at-launch fact",
        saved.get("kind") == "recipe_hash_at_launch" and saved.get("run_id") == "r1",
    )

    recipe.write_text("envelope: {name: launch-fixture}\nbench: {synthetic: {isl: 2, osl: 1}}\n")
    try:
        la.capture(cell, "r1", receipt_path)
    except ValueError as exc:
        check("refuses to overwrite an earlier launch hash after a recipe edit", "refusing to replace" in str(exc))
    else:
        check("refuses to overwrite an earlier launch hash after a recipe edit", False)


# Two different long recipes may still propose the same short candidate. Reserve the conflict before any
# deployment, keep the original owner's receipt immutable, and preserve the id's Kubernetes length budget.
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    cell_a, cell_b = root / "a", root / "b"
    cell_a.mkdir()
    cell_b.mkdir()
    (cell_a / "recipe.yaml").write_text("envelope: {name: recipe-a}\nbench: {synthetic: {isl: 1}}\n")
    (cell_b / "recipe.yaml").write_text("envelope: {name: recipe-b}\nbench: {synthetic: {isl: 2}}\n")
    results = root / "results"
    first = la.reserve(cell_a, "ttjn", results, max_attempts=10)
    resumed = la.reserve(cell_a, "ttjn", results, max_attempts=10)
    second = la.reserve(cell_b, "ttjn", results, max_attempts=10)
    check("first launch owns its proposed run id", first["run_id"] == "ttjn")
    check("same cell/hash resumes the existing reservation", resumed["run_id"] == "ttjn")
    check("different cell auto-mints a collision-free id", second["run_id"] != "ttjn", second["run_id"])
    check("collision suffix never exceeds the original Kubernetes id budget", len(second["run_id"]) <= 4)
    owner = json.loads((results / "ttjn" / "launch_attestation.json").read_text())
    check(
        "collision handling never overwrites the first owner",
        owner["cell"] == first["cell"] and owner["recipe_hash"] == first["recipe_hash"],
    )

    legacy = results / "old1"
    legacy.mkdir()
    (legacy / "artifact.txt").write_text("legacy run with no attestation")
    guarded = la.reserve(cell_a, "old1", results, max_attempts=10)
    check(
        "a non-empty legacy result directory is never claimed or mixed with a new run",
        guarded["run_id"] != "old1" and not (legacy / "launch_attestation.json").exists(),
        guarded["run_id"],
    )

_run_sh = (ROOT / "scripts/run.sh").read_text()
check(
    "run.sh reserves the attestation before phase 1/preflight",
    _run_sh.index("--reserve-root") < _run_sh.index('phase "1/8  preflight"'),
)

check(
    "run.sh marks the reserved identity final before any benchmark lane starts",
    _run_sh.index("export LLMB_RUN_ID_FINAL=1") < _run_sh.index('phase "1/8  preflight"'),
)
for _lane in ("sweep.sh",):
    _lane_text = (ROOT / f"scripts/{_lane}").read_text()
    check(
        f"{_lane} preserves the already-reserved run id verbatim",
        "LLMB_RUN_ID_FINAL" in _lane_text and "export RUN_ID=" in _lane_text,
    )

print("\nselftest_launch_attestation: all checks passed" if not fails else "\nFAIL: " + ", ".join(fails))
sys.exit(1 if fails else 0)

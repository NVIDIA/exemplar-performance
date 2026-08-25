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

"""Offline regression tests for the provenance ledger-evidence gate."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import provenance as prov  # noqa: E402
import recipe_hash as rh  # noqa: E402

fails: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        fails.append(label)


def recipe_text() -> str:
    return """envelope:
  name: provenance-fixture
  model: fixture
  gpu_type: B200
  arch: amd64
  engine: vllm
  serving_mode: aggregated
  framework: none
  scenario: llm-perf
  distribution: fixture
  mode: custom-trace
  launcher: aiperf
  goal: pareto
  status: runs
serving:
  tp: 1
bench:
  dataset: {sha256: fixture-dataset-sha256}
  sweep_concurrency: [1]
"""


def make_cell(root: Path, name: str, ledger_rows: list[dict]) -> tuple[Path, str]:
    cell = root / "recipes" / name
    (cell / "runs").mkdir(parents=True)
    (cell / "recipe.yaml").write_text(recipe_text())
    current = rh.recipe_hash(cell)
    (cell / "RESULTS.md").write_text(f"| recipe_hash | `{current}` |\n")
    (cell / "runs" / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger_rows))
    return cell, current


def run_check(root: Path) -> tuple[int, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        rc = prov.check_all(root)
    return rc, output.getvalue()


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    valid, current = make_cell(root, "valid", [{"run_id": "r1", "recipe_hash": "pending"}])
    (valid / "runs" / "index.jsonl").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "recipe_hash": current,
                "recipe_hash_at_launch": current,
                "recipe_hash_source": "launch_attestation",
            }
        )
        + "\n"
    )
    rc, output = run_check(root)
    check(
        "matching current hash in RESULTS.md and launch-attested ledger passes",
        rc == 0 and "launch-attested ledger" in output,
        output.strip(),
    )

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    _cell, current = make_cell(
        root,
        "post-edit",
        [
            {
                "run_id": "r1",
                "recipe_hash": "b" * 64,
                "recipe_hash_at_launch": "b" * 64,
                "recipe_hash_source": "launch_attestation",
            }
        ],
    )
    rc, output = run_check(root)
    check(
        "a launch-attested hash from before an edit fails against the current recipe",
        rc == 1 and current in output and "b" * 64 in output and "zero ledger runs match" in output,
        output.strip(),
    )

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    _cell, current = make_cell(root, "stale-ledger", [{"run_id": "old", "recipe_hash": "a" * 64}])
    rc, output = run_check(root)
    check(
        "stale ledger hash fails and names cell/current/found hashes",
        rc == 1
        and "recipes/stale-ledger" in output
        and current in output
        and "a" * 64 in output
        and "ledger hashes found" in output,
        output.strip(),
    )

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    _cell, current = make_cell(root, "missing-hash", [{"run_id": "legacy"}])
    rc, output = run_check(root)
    check(
        "ledger rows without recipe_hash are UNKNOWN and fail closed",
        rc == 1
        and "UNKNOWN recipes/missing-hash" in output
        and current in output
        and "lack recipe_hash" in output
        and "ledger hashes found: <missing> ×1" in output,
        output.strip(),
    )

print("\nselftest_provenance: all checks passed" if not fails else "\nFAIL: " + ", ".join(fails))
sys.exit(1 if fails else 0)

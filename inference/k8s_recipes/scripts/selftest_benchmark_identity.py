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

"""Offline regression tests for UNKNOWN workload identities and catalog collision rejection."""

from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_id as bid  # noqa: E402

fails: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        fails.append(label)


BASE = {
    "envelope": {
        "name": "identity-fixture",
        "model": "fixture",
        "gpu_type": "B200",
        "arch": "amd64",
        "engine": "vllm",
        "serving_mode": "aggregated",
        "framework": "none",
        "scenario": "llm-perf",
        "distribution": "synthetic",
        "mode": "synthetic",
        "launcher": "aiperf",
        "goal": "pareto",
    },
    "serving": {"tp": 1, "max_model_len": 1024, "extra_args": ["--flag-a"]},
    "bench": {"synthetic": {"isl": 1024, "osl": 256, "seed": 42}, "sweep_concurrency": [1]},
}

unknown = copy.deepcopy(BASE)
unknown.pop("bench")
try:
    bid.benchmark_identity(unknown)
    unknown_refused = False
    unknown_detail = "identity was produced"
except bid.UnknownBenchmarkIdentity as exc:
    unknown_refused = "dataset.sha256" in str(exc) and "synthetic" in str(exc) and "trace_sha256" in str(exc)
    unknown_detail = str(exc)
check("no dataset/synthetic/replay workload refuses an UNKNOWN benchmark_id", unknown_refused, unknown_detail)

changed_synthetic = copy.deepcopy(BASE)
changed_synthetic["bench"]["synthetic"]["osl"] = 512
check(
    "complete synthetic workload block participates in benchmark identity",
    bid.benchmark_identity(changed_synthetic) != bid.benchmark_identity(BASE),
)

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / "README.md").write_text("<!-- MATRIX:START -->\n<!-- MATRIX:END -->\n")
    for name, flag in (("first", "--flag-a"), ("second", "--flag-b")):
        cell = root / "recipes" / name
        cell.mkdir(parents=True)
        recipe = copy.deepcopy(BASE)
        recipe["envelope"]["name"] = name
        recipe["serving"]["extra_args"] = [flag]
        (cell / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_catalog.py"), "--check", str(root)],
        capture_output=True,
        text=True,
    )
    check(
        "matrix-check rejects same benchmark_id with different recipe content",
        result.returncode == 1
        and "BENCHMARK ID COLLISION" in result.stdout
        and "recipes/first" in result.stdout
        and "recipes/second" in result.stdout
        and "different recipe content" in result.stdout,
        (result.stdout + result.stderr).strip(),
    )

print("\nselftest_benchmark_identity: all checks passed" if not fails else "\nFAIL: " + ", ".join(fails))
sys.exit(1 if fails else 0)

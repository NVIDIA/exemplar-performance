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

"""selftest_gpu_basis.py — pin the TPS/GPU denominator to the GPUs the model USES.

Regression guard for the whole-node bug: with `requires.gpu.whole_node: true` the pod is
handed the ENTIRE node, so DCGM enumerates every GPU on it (4 on a GB300) even when the
model is tp=1 and uses ONE. The aggregator used to prefer `gpu_count_detected`, which
silently deflated TPS/GPU by node_gpus/model_gpus — a tp=1 GB300 cell read 4x low, halving
pareto_geomean (sqrt(4)=2.0) and turning a healthy run into a fake -52% regression.

`pareto_geomean` is a PERFORMANCE metric (throughput per accelerator DOING WORK). Node
exclusivity is an ISOLATION policy, not a redefinition of the metric — changing the
denominator would break comparability with every existing cell, the GB300<->B200
cross-hardware pair, and external 1-GPU references. Reserved-but-idle GPUs are an
OCCUPANCY COST and belong in gpu_hours accounting, never folded into the headline.

No cluster, no network — pure function checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis" / "llm-perf"))
import aggregate_metrics as am  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── A. THE WHOLE-NODE CASE (the bug this file exists for) ──────────────────────────────
# GB300 KVBM qwen3-0.6B: declared 1 (tp=1), detected 4 (whole node reserved).
basis, msg = am.resolve_gpu_basis(detected=4, gpu_count_override=None, run_meta_gpu_count=1)
check("whole_node: declared 1 + detected 4 → basis is 1 (per USED GPU, not reserved)", basis == 1, f"got {basis}")
check(
    "whole_node: disagreement does NOT fail — it returns a note, not an error",
    msg is not None and msg.startswith("note:"),
    f"got {msg!r}",
)
check(
    "whole_node note names both counts + says which is the denominator",
    bool(msg) and "1 GPU(s)" in msg and "4 reserved" in msg and "per USED GPU" in msg,
    f"got {msg!r}",
)

# The exact arithmetic the bug corrupted: 1546.43 tok/s at c=64 on a 4-GPU node.
tput = 1546.432433
check(
    "whole_node: TPS/GPU divides by the USED count (no 4x deflation)",
    abs(tput / basis - tput) < 1e-9,
    f"{tput / basis}",
)
check("the OLD (buggy) basis would have deflated TPS/GPU exactly 4x", abs((tput / 4) - 386.608108) < 1e-5)

# ── B. NORMAL CASES — declared == detected, and no whole-node reservation ───────────────
basis, msg = am.resolve_gpu_basis(detected=4, gpu_count_override=None, run_meta_gpu_count=4)
check(
    "declared == detected → basis 4, and NO note (nothing surprising to report)",
    basis == 4 and msg is None,
    f"basis={basis} msg={msg!r}",
)

basis, msg = am.resolve_gpu_basis(detected=None, gpu_count_override=None, run_meta_gpu_count=8)
check(
    "no detection (DCGM-less cluster) → declared 8 still wins, no warning",
    basis == 8 and msg is None,
    f"basis={basis} msg={msg!r}",
)

# ── C. OVERRIDE PRECEDENCE — an explicit --gpu-count beats everything ───────────────────
basis, _ = am.resolve_gpu_basis(detected=4, gpu_count_override=2, run_meta_gpu_count=1)
check("--gpu-count override outranks both run_meta and detected", basis == 2, f"got {basis}")

# ── D. MISSING DECLARED COUNT — the ONLY loud case ──────────────────────────────────────
basis, msg = am.resolve_gpu_basis(detected=4, gpu_count_override=None, run_meta_gpu_count=None)
check(
    "no declared count → falls back to detected AND warns loudly",
    basis == 4 and bool(msg) and msg.startswith("WARN:"),
    f"basis={basis} msg={msg!r}",
)

basis, msg = am.resolve_gpu_basis(detected=None, gpu_count_override=None, run_meta_gpu_count=None)
check(
    "no declared count and no detection → 8-GPU last resort, warned (WRONG on non-8-GPU nodes)",
    basis == 8 and bool(msg) and msg.startswith("WARN:"),
    f"basis={basis} msg={msg!r}",
)

# ── E. ZERO/FALSY GUARD — a 0 must not be mistaken for a declared count ─────────────────
basis, _ = am.resolve_gpu_basis(detected=4, gpu_count_override=0, run_meta_gpu_count=1)
check("override of 0 is ignored (falsy), declared run_meta count still wins", basis == 1, f"got {basis}")

print(f"\nselftest_gpu_basis: {'all checks passed' if not fails else str(len(fails)) + ' FAILED'}")
raise SystemExit(1 if fails else 0)

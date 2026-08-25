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

"""selftest_recovery_meta.py — unit tests for recovery.run_meta() (deferred bit).

run_meta() is the pure, machine-readable twin of recovery.report(): the EXIT trap writes its JSON to
results/<run-id>/run_meta.json so a failed run's results dir carries its own post-mortem. Pure (no cluster,
no side effects) — asserted here. Runs with `python3 scripts/selftest_recovery_meta.py`; exit 0 = all pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rec = _load("recovery")
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── the headline case: an OOM (exit 137) mid-sweep ────────────────────────────
# Sweep rungs 256/384/512/768/1024; 256 and 384 finished, then a pod was OOMKilled (137) at 512.
# classify() infers "oom" from the exit code (137) — the OOM signal is the exit code, and it beats even the
# phase name run.sh passes at trap time (a specific code is more informative than WHERE it died; see below).
CELL = "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"
PROFILE = "example-gpu-cluster"
DONE = ["256", "384"]
ALL = ["256", "384", "512", "768", "1024"]

m = rec.run_meta(
    CELL,
    PROFILE,
    reason=None,
    exit_code=137,
    run_id="20260716-142301",
    rungs_done=DONE,
    rungs_all=ALL,
)

check("run_meta: returns a dict", isinstance(m, dict))
check("run_meta: status is failed", m["status"] == "failed", repr(m.get("status")))
check(
    "run_meta: exit 137 classified as oom",
    m["failure_reason"] == "oom",
    repr(m.get("failure_reason")),
)
check("run_meta: step carries the raw trap reason", m["step"] is None, repr(m.get("step")))
check(
    "run_meta: run_id carried through",
    m["run_id"] == "20260716-142301",
    repr(m.get("run_id")),
)
check(
    "run_meta: rungs_completed = the done rungs",
    m["rungs_completed"] == DONE,
    repr(m.get("rungs_completed")),
)
check(
    "run_meta: rungs_remaining = unfinished, order preserved",
    m["rungs_remaining"] == ["512", "768", "1024"],
    repr(m.get("rungs_remaining")),
)
check(
    "run_meta: resume names the cluster + remaining rungs + --skip-server",
    m["resume"]
    == 'scripts/llmb-k8s run --recipe {c} --cluster {p} --rungs "512 768 1024" --skip-server'.format(c=CELL, p=PROFILE),
    repr(m.get("resume")),
)

# ── serializable (the trap pipes it straight to run_meta.json) ────────────────
try:
    round_trip = json.loads(json.dumps(m))
    check("run_meta: round-trips through JSON", round_trip == m)
except (TypeError, ValueError) as e:
    check("run_meta: round-trips through JSON", False, str(e))

# ── purity: identical inputs → identical output, and inputs are not mutated ───
done_copy, all_copy = list(DONE), list(ALL)
m2 = rec.run_meta(
    CELL,
    PROFILE,
    reason=None,
    exit_code=137,
    run_id="20260716-142301",
    rungs_done=done_copy,
    rungs_all=all_copy,
)
check("run_meta: deterministic (pure)", m2 == m)
check("run_meta: does not mutate rungs_done arg", done_copy == DONE)
check("run_meta: does not mutate rungs_all arg", all_copy == ALL)

# ── degenerate: no rung info (the trap doesn't track rungs) → empty lists, bare resume ─
m3 = rec.run_meta(
    CELL,
    PROFILE,
    reason="wait-ready",
    exit_code=1,
    run_id="r1",
    rungs_done=None,
    rungs_all=None,
)
check(
    "run_meta: no rungs → empty completed/remaining",
    m3["rungs_completed"] == [] and m3["rungs_remaining"] == [],
)
check(
    "run_meta: no rungs → resume has no --rungs",
    "--rungs" not in m3["resume"],
    m3["resume"],
)
check(
    "run_meta: wait-ready failure reason preserved",
    m3["failure_reason"] == "wait-ready",
)

# ── trap reality: STEP="benchmark" is what run.sh passes as --reason, and an OOM (137) must still read "oom" ──
# This is the actual trap call for a sweep OOM. A specific exit code (137=OOMKilled) is more informative than
# the PHASE it died in, so classify() lets the code win over a generic phase name — while `step` still preserves
# the phase verbatim for context.
m4 = rec.run_meta(
    CELL,
    PROFILE,
    reason="benchmark",
    exit_code=137,
    run_id="r2",
    rungs_done=DONE,
    rungs_all=ALL,
)
check(
    "run_meta: exit 137 wins over the phase name → oom (accurate post-mortem)",
    m4["failure_reason"] == "oom",
)
check("run_meta: step preserves the phase verbatim for context", m4["step"] == "benchmark")
check("run_meta: status still failed regardless of reason", m4["status"] == "failed")
# an explicit ROOT-CAUSE reason (not a phase) still wins over the code — e.g. the idle-guard's hang-kill.
m5 = rec.run_meta(
    CELL,
    PROFILE,
    reason="hang",
    exit_code=137,
    run_id="r3",
    rungs_done=DONE,
    rungs_all=ALL,
)
check(
    "run_meta: explicit root-cause reason (hang) wins over the exit code",
    m5["failure_reason"] == "hang",
)


if fails:
    print(f"\n{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("\nall recovery.run_meta() checks passed")
sys.exit(0)

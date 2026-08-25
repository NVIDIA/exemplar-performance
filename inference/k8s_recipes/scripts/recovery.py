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

"""recovery.py — turn a failed/interrupted run into an actionable next step.

When run.sh dies (crash, hang-kill, server failure, Ctrl-C) its EXIT trap calls this to print the likely
cause + the exact resume command, so a run ends with "here's what happened and how to continue" instead of
a bare non-zero exit. Pure (no cluster, no side effects) — unit-tested in selftest.py.

CLI:
  recovery.py <cell> <profile> [--reason R] [--exit N] [--run-id ID]
              [--rungs-done "a b"] [--rungs-all "a b c"]
  recovery.py --meta <cell> <profile> [--reason R] [--exit N] [--run-id ID]
              [--rungs-done "a b"] [--rungs-all "a b c"]   # emit run_meta.json (a failure record) as JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Failure signal → operator-facing cause. Keys are what run.sh knows at trap time (a phase name or code).
HINTS = {
    "oom": "a pod was OOMKilled (exit 137) — max-num-seqs is likely too low for this concurrency, "
    "or the server needs more GPU-memory headroom.",
    "hang": "the server produced no new tokens for the idle window — likely deadlocked under load "
    "(the idle-guard cancelled the run).",
    "crashloop": "the server pod is CrashLoopBackOff — check its log for an engine crash or GPU OOM.",
    "interrupt": "interrupted (Ctrl-C) — nothing wrong, just stopped.",
    "wait-ready": "the server never became Ready — check the server pod (image pull? weights? scheduling?).",
    "benchmark": "the benchmark step failed before completing — see the sweep/live log above.",
    "unknown": "the run exited non-zero before completing.",
}


# Phase names run.sh passes as --reason (WHERE it died) — informative, but less specific than a root-cause
# exit code. A specific code (137=OOM, 130=SIGINT) should win over these; an explicit root-cause reason
# (oom/hang/crashloop/interrupt) still wins over everything.
PHASE_REASONS = {"benchmark", "wait-ready", "startup", "done"}


def classify(reason: str | None = None, exit_code: int | None = None) -> str:
    """Map a raw failure signal to a hint key. An explicit root-cause reason wins; else a specific exit code
    (137→oom, 130→interrupt) beats a generic phase name; else fall back to the phase's own hint."""
    if reason in HINTS and reason not in PHASE_REASONS:  # explicit root cause (oom/hang/crashloop/interrupt)
        return reason
    if exit_code == 137:  # OOMKilled — more specific than the phase it died in
        return "oom"
    if exit_code == 130:  # SIGINT (Ctrl-C)
        return "interrupt"
    if reason in HINTS:  # fall back to the phase's own hint (benchmark/wait-ready)
        return reason
    return "unknown"


def remaining_rungs(rungs_done: list[str], rungs_all: list[str]) -> str:
    """The rungs still to run, as a space list for --rungs (order preserved from rungs_all)."""
    done = set(rungs_done or [])
    return " ".join(r for r in (rungs_all or []) if r not in done)


def resume_cmd(cell: str, profile: str, rungs: str = "") -> str:
    # --skip-server: a crash usually leaves the server Deployment defined (scaled to 0) — resume reuses it.
    r = f' --rungs "{rungs}"' if rungs else ""
    return f"scripts/llmb-k8s run --recipe {cell} --cluster {profile}{r} --skip-server"


def report(
    cell: str, profile: str, *, reason=None, exit_code=None, run_id=None, rungs_done=None, rungs_all=None
) -> str:
    key = classify(reason, exit_code)
    out = [f"✗ run did not complete — {HINTS[key]}"]
    rungs = ""
    if rungs_all:
        rungs = remaining_rungs(rungs_done or [], rungs_all)
        out.append(
            f"  rungs: {len(rungs_done or [])}/{len(rungs_all)} done"
            + (f"; remaining {rungs}" if rungs else "; all rungs ran")
        )
    if run_id:
        out.append(f"  recover partial artifacts:  scripts/fetch_results.sh --partial {run_id}")
    out.append("  resume:")
    out.append(f"    {resume_cmd(cell, profile, rungs)}")
    return "\n".join(out)


def run_meta(
    cell: str, profile: str, *, reason=None, exit_code=None, run_id=None, rungs_done=None, rungs_all=None
) -> dict:
    """Pure failure record for results/<run-id>/run_meta.json — machine-readable twin of report().

    The trap writes this so a failed run's results dir carries its own post-mortem (status, classified
    reason, which rungs finished, and the exact resume command) without re-deriving anything at publish time.
    """
    rungs_done = list(rungs_done or [])
    rungs_all = list(rungs_all or [])
    done = set(rungs_done)
    remaining = [r for r in rungs_all if r not in done]
    return {
        "status": "failed",
        "failure_reason": classify(reason, exit_code),
        "step": reason,
        "run_id": run_id,
        "rungs_completed": rungs_done,
        "rungs_remaining": remaining,
        "resume": resume_cmd(cell, profile, " ".join(remaining)),
    }


def _print_failure_box(
    cell: str, profile: str, *, reason=None, exit_code=None, run_id=None, rungs_done=None, rungs_all=None
) -> None:
    """Print a prominent failure box to stderr — the first thing an operator sees after a crash."""
    key = classify(reason, exit_code)
    hint = HINTS[key]

    rungs_str = ""
    if rungs_all:
        remaining = remaining_rungs(rungs_done or [], rungs_all)
        n_done = len(rungs_done or [])
        n_all = len(rungs_all)
        rungs_str = f"{n_done}/{n_all} rungs done" + (f"; remaining: {remaining}" if remaining else "; all rungs ran")

    resume = resume_cmd(cell, profile, remaining_rungs(rungs_done or [], rungs_all) if rungs_all else "")

    # Diagnosis rows (inside the box — no long command lines here)
    rows = [("phase", reason or "unknown")]
    if exit_code is not None:
        rows.append(("exit", str(exit_code)))
    if run_id:
        rows.append(("run-id", run_id))
    rows.append(("cause", hint))
    if rungs_str:
        rows.append(("rungs", rungs_str))

    k_w = max(len(k) for k, _ in rows) + 2
    MAX_W = 100
    content = []
    for k, v in rows:
        raw = f"  {k:<{k_w}} {v}"
        content.append(raw[:MAX_W] + "…" if len(raw) > MAX_W else raw)
    header = f"  ❌  run failed  ·  {Path(cell).name}"
    inner_w = max(len(header), max(len(c) for c in content))
    border = "═" * (inner_w + 1)

    out = sys.stderr
    print(f"\n╔{border}╗", file=out)
    print(f"║{header:<{inner_w + 1}}║", file=out)
    print(f"╠{border}╣", file=out)
    for c in content:
        print(f"║{c:<{inner_w + 1}}║", file=out)
    print(f"╚{border}╝", file=out)
    # Full cause when truncated; resume command always outside the box (copy-pasteable)
    if len(f"  {'cause':<{k_w}} {hint}") > MAX_W:
        print(f"\n  cause:   {hint}", file=out)
    if run_id:
        print(f"  partial: scripts/fetch_results.sh --partial {run_id}", file=out)
    print(f"  resume:  {resume}", file=out)


def main(argv: list[str]) -> int:
    # --meta: emit the machine-readable run_meta.json instead of the human report (same options).
    meta = False
    if argv and argv[0] == "--meta":
        meta = True
        argv = argv[1:]
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cell, profile = argv[0], argv[1]
    opt = {"reason": None, "exit": None, "run_id": None, "rungs_done": None, "rungs_all": None}
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--reason" and i + 1 < len(argv):
            opt["reason"] = argv[i + 1]
            i += 2
            continue
        if a == "--exit" and i + 1 < len(argv):
            try:
                opt["exit"] = int(argv[i + 1])
            except ValueError:
                opt["exit"] = None
            i += 2
            continue
        if a == "--run-id" and i + 1 < len(argv):
            opt["run_id"] = argv[i + 1]
            i += 2
            continue
        if a == "--rungs-done" and i + 1 < len(argv):
            opt["rungs_done"] = argv[i + 1].split()
            i += 2
            continue
        if a == "--rungs-all" and i + 1 < len(argv):
            opt["rungs_all"] = argv[i + 1].split()
            i += 2
            continue
        i += 1
    if meta:
        print(
            json.dumps(
                run_meta(
                    cell,
                    profile,
                    reason=opt["reason"],
                    exit_code=opt["exit"],
                    run_id=opt["run_id"],
                    rungs_done=opt["rungs_done"],
                    rungs_all=opt["rungs_all"],
                )
            )
        )
        return 0
    _print_failure_box(
        cell,
        profile,
        reason=opt["reason"],
        exit_code=opt["exit"],
        run_id=opt["run_id"],
        rungs_done=opt["rungs_done"],
        rungs_all=opt["rungs_all"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

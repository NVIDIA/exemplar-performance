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

"""watch_runs.py [--interval N] [run-id ...] — live timeline dashboard for parallel runs.

Finds all active run phase logs under results/*/phases.log, renders a Gantt timeline
for each one, and refreshes every N seconds (default 15).  Clears the terminal between
frames so the view stays in-place like `watch`.

Useful when running multiple cells in parallel across terminals or clusters:
  Terminal A:  llmb-k8s run <cell1> <profile1>
  Terminal B:  llmb-k8s run <cell2> <profile2>
  Terminal C:  llmb-k8s watch

A run is shown if its phases.log was modified within the last 2 hours. Completed runs
(✅ _END_ sentinel) and failed runs (❌ _FAILED_ sentinel) remain visible until the
2-hour staleness window expires — they are NOT deleted on run completion.

Pass specific run-ids to watch only those:
  llmb-k8s watch abc123 def456

Exit with Ctrl-C.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STALE_SECONDS = 7200  # 2 hours — drop logs not updated within this window


def find_phase_logs(run_ids=None):
    """Return list of phases.log paths for active (or requested) runs, newest-modified first."""
    if not RESULTS.exists():
        return []
    candidates = sorted(RESULTS.glob("*/phases.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    now = time.time()
    out = []
    for log in candidates:
        run_id = log.parent.name
        if run_ids and run_id not in run_ids:
            continue
        age = now - log.stat().st_mtime
        if age > STALE_SECONDS and not run_ids:
            continue
        out.append(log)
    return out


def is_complete(log: Path) -> tuple:
    """Return (done, status): done=True when a terminal sentinel is present.

    Sentinels:
      _END_     — written on successful completion → status "ok"
      _FAILED_  — written by EXIT trap on failure  → status "failed"
    """
    try:
        text = log.read_text()
    except OSError:
        return False, "unknown"
    if "_FAILED_" in text:
        return True, "failed"
    if "_END_" in text:
        return True, "ok"
    return False, "running"


def render_one(log: Path) -> str:
    """Render one run's timeline as a string block."""
    run_id = log.parent.name
    # Read _START_ epoch for t0
    t0 = int(time.time())
    try:
        for line in log.read_text().splitlines():
            parts = line.strip().split("\t", 1)
            if len(parts) == 2 and parts[1] == "_START_":
                t0 = int(parts[0])
                break
    except (OSError, ValueError):
        pass
    done, status = is_complete(log)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_summary.py"),
            "--phases",
            str(log),
            "--t0",
            str(t0),
            "--run-id",
            run_id,
            "--status",
            "ok" if (done and status == "ok") else ("failed" if done else "running"),
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout


def clear():
    print("\033[H\033[J", end="", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_ids", nargs="*", help="Specific run-ids to watch (default: all active)")
    ap.add_argument("--interval", "-n", type=int, default=15, help="Refresh interval in seconds (default 15)")
    ap.add_argument("--once", action="store_true", help="Render once and exit (no loop)")
    args = ap.parse_args()

    run_ids = set(args.run_ids)

    def frame():
        logs = find_phase_logs(run_ids or None)
        if not logs:
            print(f"  no active runs found under {RESULTS}/")
            print(f"  (looking for results/*/phases.log modified within {STALE_SECONDS//3600}h)")
            return
        for log in logs:
            out = render_one(log)
            if out.strip():
                print(out)

    if args.once:
        frame()
        return

    try:
        while True:
            clear()
            ts = time.strftime("%H:%M:%S")
            print(f"llmb-k8s watch  —  {ts}  (refreshing every {args.interval}s, Ctrl-C to exit)\n")
            frame()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nwatch stopped")


if __name__ == "__main__":
    main()

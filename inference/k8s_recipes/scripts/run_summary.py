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

"""run_summary.py --phases <log> --t0 <epoch> [--run-id ID] [--cell NAME] [--status ok|failed]

Reads a TSV phase-timing log written by run.sh and prints a compact Gantt timeline
showing when each phase ran and how long it took.

Phase log format (one line per phase, written by run.sh's phase() function):
  <epoch_seconds>\t<phase_label>

The last entry has label "_END_" and epoch = end of run (T1).

Output (example):
  ╔══════════════════════════════════════════════════════════════════╗
  ║  timeline  run-id=abc123  cell=glm5-fp8-b200  status=✅ ok
  ╠══════════════════════════════════════════════════════════════════╣
    1/8 preflight    ▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    12s
    2/8 namespace    ░▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     1s
    3/8 stage        ░░▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░    45s
    4/8 server       ░░░░▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     8s
    5/8 wait-ready   ░░░░░▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2m04s
    6/8 benchmark    ░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░  18m32s
    7/8 fetch        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓░░░░░░░░░   1m15s
    8/8 teardown     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓░░░░░░░░     3s
  ╠══════════════════════════════════════════════════════════════════╣
  ║  total: 22m40s
  ╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
from pathlib import Path

BAR_WIDTH = 38
FILL = "▓"
EMPTY = "░"


def fmt_duration(secs):
    secs = max(0, int(secs))
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def parse_phases(path, t0):
    """Parse phase log → (phases, total_start, total_end).

    Phase log entries:
      <epoch>\t_START_    — written at T0 when run.sh begins (baseline anchor)
      <epoch>\t<label>    — each phase() call
      <epoch>\t_END_      — written at T1 on successful completion
      <epoch>\t_FAILED_   — written in EXIT trap on failure

    Returns ([(label, start, end), ...], total_start, total_end).
    Sentinel labels (_START_, _END_, _FAILED_) are stripped; they anchor the timeline.
    """
    entries = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            ts = int(parts[0])
        except ValueError:
            continue
        entries.append((ts, parts[1]))

    if not entries:
        return []

    # _START_ entry anchors the timeline; fall back to t0 argument
    start_entries = [ts for ts, lbl in entries if lbl == "_START_"]
    total_start = start_entries[0] if start_entries else t0

    end_entries = [ts for ts, lbl in entries if lbl in ("_END_", "_FAILED_")]
    now = max(ts for ts, _ in entries)
    total_end = end_entries[-1] if end_entries else now

    _SENTINELS = {"_START_", "_END_", "_FAILED_"}
    # Build phase list (skip sentinels)
    real = [(ts, lbl) for ts, lbl in entries if lbl not in _SENTINELS]
    phases = []
    for i, (ts, label) in enumerate(real):
        end_ts = real[i + 1][0] if i + 1 < len(real) else total_end
        phases.append((label, ts, end_ts))

    return phases, total_start, total_end


def render_bar(start, end, total_start, total_end):
    """Render a BAR_WIDTH-char Gantt bar for one phase."""
    span = max(total_end - total_start, 1)
    bar_start = int((start - total_start) / span * BAR_WIDTH)
    bar_end = max(bar_start + 1, int((end - total_start) / span * BAR_WIDTH))
    bar_end = min(bar_end, BAR_WIDTH)
    bar = EMPTY * bar_start + FILL * (bar_end - bar_start) + EMPTY * (BAR_WIDTH - bar_end)
    return bar


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--phases", required=True)
    ap.add_argument("--t0", type=int, required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--cell", default="")
    ap.add_argument("--status", default="ok")
    args = ap.parse_args()

    try:
        result = parse_phases(args.phases, args.t0)
    except Exception as e:
        print(f"run_summary: could not parse phase log ({e})", file=sys.stderr)
        return

    if not result:
        return

    phases, total_start, total_end = result
    if not phases:
        return

    status_icon = {"ok": "✅", "failed": "❌"}.get(args.status, "⏳")
    label_w = max(len(p[0]) for p in phases) + 2

    # Header
    inner = BAR_WIDTH + label_w + 12
    border = "═" * (inner + 2)
    header_parts = ["timeline"]
    if args.run_id:
        header_parts.append(f"run-id={args.run_id}")
    if args.cell:
        header_parts.append(f"cell={args.cell}")
    header_parts.append(f"status={status_icon} {args.status}")
    header_line = "  ".join(header_parts)

    print()
    print(f"╔{border}╗")
    print(f"║  {header_line:<{inner}}║")
    print(f"╠{border}╣")

    for label, start, end in phases:
        bar = render_bar(start, end, total_start, total_end)
        dur = fmt_duration(end - start)
        print(f"  {label:<{label_w}} {bar}  {dur:>7}")

    total_dur = fmt_duration(total_end - total_start)
    print(f"╠{border}╣")
    print(f"║  total: {total_dur:<{inner - 8}}║")
    print(f"╚{border}╝")


if __name__ == "__main__":
    main()

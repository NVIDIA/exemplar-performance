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

"""rung_progress.py [--quiet] [--annotate-job <name> --namespace <ns> [--total N]]
                     — tee bench-job log stream, injecting rung summaries.

Reads `kubectl logs -f` output from stdin, passes every line through to stdout,
and injects compact ✓/✗ summary lines at rung boundaries so the operator can
track sweep progress without wading through aiperf's verbose output.

--quiet  suppress raw log passthrough (show only the progress lines). Useful
         when piped to a file or when brevity matters.

--annotate-job / --namespace  LIVE fleet progress: after EACH rung completes,
         best-effort `kubectl annotate --overwrite job/<name>` the bench Job with
         `llmb.nvidia.com/completed-rungs=<n>` (+ `.../total-rungs=<N>`), so
         `llmb-k8s fleet` advances the SWEEP dot-bar (●●●◐○ 3/5) rung-by-rung
         while the sweep runs. The total is auto-learned from the fixed-list log
         line (or --total N). Read-only for fleet (a plain annotation on the Job
         fleet already lists — NOT the control-PVC heartbeat / concurrency_* dirs).
         Uses KUBE_CONTEXT / KUBECTL from the environment; failures are ignored
         (a detached follower / missing RBAC simply leaves the plain rung list).

Recognised log patterns (from the bench-job embedded runner script):
  [HH:MM:SS] Running aiperf at concurrency=N (PHASE)...       ← rung START
  [HH:MM:SS] Concurrency=N decision: passed=X reason=Y        ← rung END
  [HH:MM:SS] Adaptive Phase 2: binary-searching bracket lo=L hi=H
  [HH:MM:SS] Skipping Phase 2: REASON
  [HH:MM:SS] Sweep complete: PATH

Example progress output (injected between raw log lines):

  ═══════════════════════════════════════════════════
  ⏳  c=128  (fixed)  running...            [12:31:45]
  ✓   c=128  PASS                           [12:33:27]  1m42s
  ⏳  c=256  (fixed)  running...            [12:33:28]
  ✗   c=256  FAIL  threshold_exceeded       [12:37:46]  4m18s
  → adaptive: binary-searching [128, 256]
  ⏳  c=192  (refine)  running...           [12:37:47]
  ✓   c=192  PASS                           [12:40:11]  2m24s
  ═══════════════════════════════════════════════════
  sweep complete  ·  2 passed  1 failed
"""

import os
import re
import subprocess
import sys
import time

_START_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] Running aiperf at concurrency=(\d+) \(([^)]+)\)")
_DONE_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] Concurrency=(\d+) decision: passed=(\w+) reason=([^\s]*)")
_BINARY_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] Adaptive Phase 2: binary-searching bracket lo=(\d+) hi=(\d+)")
_SKIP_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] Skipping Phase 2: (.+)")
_COMPLETE_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\] Sweep complete:")
# The fixed-list announcement (bench runner) → the TOTAL rung count for the live progress annotation.
_FIXEDLIST_RE = re.compile(r"running fixed list:\s*([\d\s]+)")
_BAR = "  " + "═" * 51

ANN_COMPLETED = "llmb.nvidia.com/completed-rungs"
ANN_TOTAL = "llmb.nvidia.com/total-rungs"


def _opt(argv: list, name: str):
    """The value following `name` in argv, or None."""
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _annotate_job(job: str, ns: str, done: int, total) -> None:
    """Best-effort `kubectl annotate --overwrite job/<job>` with the live rung progress. Never raises — a
    detached follower, missing RBAC, or a transient auth blip simply leaves fleet's plain rung list.
    """
    if not job:
        return
    cmd = [os.environ.get("KUBECTL", "kubectl")]
    ctx = os.environ.get("KUBE_CONTEXT")
    if ctx:
        cmd += ["--context", ctx]
    if ns:
        cmd += ["-n", ns]
    cmd += [
        "annotate",
        "--overwrite",
        "--request-timeout=10s",
        f"job/{job}",
        f"{ANN_COMPLETED}={max(0, int(done))}",
    ]
    if total:
        cmd += [f"{ANN_TOTAL}={int(total)}"]
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except Exception:
        pass


def _dur(seconds: float) -> str:
    m, s = divmod(max(0, int(seconds)), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main() -> None:
    argv = sys.argv[1:]
    quiet = "--quiet" in argv
    ann_job = _opt(argv, "--annotate-job")  # publish live rung progress to this bench Job (best-effort)
    ann_ns = _opt(argv, "--namespace")
    _t = _opt(argv, "--total")
    total = int(_t) if (_t and _t.isdigit()) else None  # else auto-learned from the fixed-list log line
    rung_starts: dict[str, float] = {}  # concurrency_str → wall clock when start seen
    n_pass = 0
    n_fail = 0
    header_printed = False

    def _ensure_header() -> None:
        nonlocal header_printed
        if not header_printed:
            print(_BAR, flush=True)
            header_printed = True

    try:
        for raw in sys.stdin:
            if not quiet:
                sys.stdout.write(raw)
                sys.stdout.flush()
            line = raw.rstrip()

            if total is None:  # learn the TOTAL rung count from the fixed-list announcement
                mf = _FIXEDLIST_RE.search(line)
                if mf:
                    total = len(mf.group(1).split())

            m = _START_RE.search(line)
            if m:
                ts, c, phase = m.group(1), m.group(2), m.group(3)
                rung_starts[c] = time.time()
                _ensure_header()
                print(f"  ⏳  c={c}  ({phase})  running...{'':<12}  [{ts}]", flush=True)
                continue

            m = _DONE_RE.search(line)
            if m:
                ts, c = m.group(1), m.group(2)
                ok = m.group(3) == "true"
                reason = m.group(4)
                elapsed = time.time() - rung_starts.pop(c, time.time())
                icon = "✓" if ok else "✗"
                status = "PASS" if ok else f"FAIL  {reason}"
                _ensure_header()
                print(
                    f"  {icon}   c={c}  {status:<34}  [{ts}]  {_dur(elapsed)}",
                    flush=True,
                )
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                _annotate_job(ann_job, ann_ns, n_pass + n_fail, total)  # LIVE: bar advances after each rung
                continue

            m = _BINARY_RE.search(line)
            if m:
                _, lo, hi = m.group(1), m.group(2), m.group(3)
                _ensure_header()
                print(f"  → adaptive: binary-searching [{lo}, {hi}]", flush=True)
                continue

            m = _SKIP_RE.search(line)
            if m:
                reason = m.group(2).strip()
                _ensure_header()
                print(f"  → skipping refinement: {reason}", flush=True)
                continue

            m = _COMPLETE_RE.search(line)
            if m:
                _ensure_header()
                decided = n_pass + n_fail
                verdict = "✅" if n_fail == 0 else ("⚠" if n_pass > 0 else "❌")
                print(_BAR, flush=True)
                print(
                    f"  sweep complete {verdict}  ·  {n_pass}/{decided} rungs passed",
                    flush=True,
                )
                # settled → every rung filled: publish done==total (the fixed-list total if known, else decided).
                _annotate_job(ann_job, ann_ns, total or decided, total or decided)

    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()

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

"""selftest_dispatcher_help.py — discoverability tests for the `llmb-k8s` dispatcher (no cluster needed).

Exercises only the help/usage surface — nothing here touches a cluster:
  - `llmb-k8s help`         → grouped verb list mentions every known verb (rc 0)
  - `llmb-k8s help run`     → that verb's one-line usage (rc 0)
  - `llmb-k8s <bogus>`      → known-verbs list + `help <verb>` hint (rc 1)
  - `llmb-k8s --help`       → still prints the full module __doc__ (backward-compatible, rc 0)
  - `bash -n llmb-k8s-completion.bash`  → the completion script parses

Mirrors the check()/summary pattern used in scripts/selftest_onboarding.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISPATCH = ROOT / "scripts" / "llmb-k8s"
COMPLETION = ROOT / "scripts" / "llmb-k8s-completion.bash"

# Verbs the dispatcher advertises (KNOWN_VERBS in scripts/llmb-k8s). Kept here as an independent
# expectation so this test catches a verb silently dropped from the help surface.
EXPECTED_VERBS = [
    "analyze",
    "cancel",
    "capacity",
    "collect",
    "compare",
    "deploy",
    "dry-run",
    "fleet",
    "install",
    "jobs",
    "logs",
    "port-recipe",
    "preflight",
    "profile",
    "publish",
    "reclaim",
    "run",
    "stage",
    "status",
    "submit",
]

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    marker = "PASS" if cond else "FAIL"
    print(f"  {marker}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def run(*args: str) -> tuple[int, str]:
    """Run `llmb-k8s <args>` as a subprocess; return (returncode, stdout+stderr)."""
    p = subprocess.run(
        [sys.executable, str(DISPATCH), *args],
        capture_output=True,
        text=True,
    )
    return p.returncode, (p.stdout + p.stderr)


# ---------------------------------------------------------------------------
# 1. `llmb-k8s help` — grouped verb list, rc 0, every verb present
# ---------------------------------------------------------------------------
rc, out = run("help")
check("help: exits 0", rc == 0, f"rc={rc}")
missing = [v for v in EXPECTED_VERBS if v not in out]
check("help: lists every known verb", not missing, f"missing={missing}")
check("help: mentions the profile forms", "--recipe" in out and "--cluster" in out, out[:120])

# ---------------------------------------------------------------------------
# 2. `llmb-k8s help run` — single verb usage
# ---------------------------------------------------------------------------
rc, out = run("help", "run")
check("help run: exits 0", rc == 0, f"rc={rc}")
check("help run: shows the run usage", "run" in out and "run.sh" in out, out.strip())
check("help run: is a single line", out.strip().count("\n") == 0, repr(out))

# `help <bogus>` should be a clean rc-1 with the known-verbs list, not a crash.
rc, out = run("help", "no-such-verb")
check("help <bogus>: exits 1", rc == 1, f"rc={rc}")
check("help <bogus>: shows known verbs", "known verbs" in out, out.strip())

# ---------------------------------------------------------------------------
# 3. `llmb-k8s <bogus>` — unknown-verb path: list + hint, rc 1
# ---------------------------------------------------------------------------
rc, out = run("definitely-not-a-verb")
check("bogus verb: exits 1", rc == 1, f"rc={rc}")
check("bogus verb: names the offending verb", "definitely-not-a-verb" in out, out.strip())
check("bogus verb: lists known verbs", "known verbs" in out, out.strip())
check("bogus verb: hints at `help <verb>`", "help <verb>" in out, out.strip())

# ---------------------------------------------------------------------------
# 4. Backward-compatible: `--help` / no-args still print the module __doc__
# ---------------------------------------------------------------------------
rc, out = run("--help")
check("--help: exits 0", rc == 0, f"rc={rc}")
check("--help: prints the module doc (Scenario registry)", "Scenario registry" in out, out[:120])

rc, out = run()
check("no-args: exits 0", rc == 0, f"rc={rc}")
check("no-args: prints the module doc", "Scenario registry" in out, out[:120])

# ---------------------------------------------------------------------------
# 5. Completion script parses under `bash -n`
# ---------------------------------------------------------------------------
p = subprocess.run(["bash", "-n", str(COMPLETION)], capture_output=True, text=True)
check("completion: bash -n parses clean", p.returncode == 0, p.stderr.strip())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if fails:
    print(f"selftest_dispatcher_help: {len(fails)} FAILED: {fails}")
    sys.exit(1)
else:
    total = sum(1 for line in open(__file__).read().splitlines() if line.strip().startswith("check("))
    print(f"selftest_dispatcher_help: all {total} checks PASSED ✓")
    sys.exit(0)

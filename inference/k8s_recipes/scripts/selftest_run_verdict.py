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

"""selftest_run_verdict.py — offline guards for scripts/_kubectl_resilient.sh, the cert-resilient
AUTHORITATIVE benchmark verdict used by sweep.sh (and the re-auth used by fetch_results.sh).

Motivating live-QA bug: a `tsh` kube cert expired mid-run, the client's `kubectl logs -f` dropped, and an old
inline verdict loop defaulted a FAILED `kubectl get job` query to "0|1|0" — so a fully-successful run (Job
Complete server-side, 5/5 rungs) was falsely printed as "❌ run failed". The purpose of the helper is that
a dropped log stream / unreachable apiserver is NEVER by itself a failure — only an AUTHORITATIVE
.status.failed>=1 is.

This branch's consolidation kept the EVOLVED helper (llmb::job_state + llmb::follow_job_to_terminal, which
already re-auth and treat a transient/unreadable query as keep-waiting) as the single implementation — it
subsumes the earlier resilient_job_state/resilient_job_verdict draft. So this selftest drives:
  1. the PURE kernel `_verdict_from_state` (the side-effect-free spec of the rule), and
  2. the SHIPPING `llmb::job_state` classifier + `llmb::follow_job_to_terminal` loop, with `llmb::kc_resilient`
     mocked so we assert the verdict with no real kubectl,
plus that sweep.sh / fetch_results.sh actually wire the helper in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "scripts" / "_kubectl_resilient.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def _bash(snippet: str) -> subprocess.CompletedProcess:
    """Run a bash snippet with the helper sourced; ROOT + a non-Teleport KUBE_CONTEXT + NAMESPACE so nothing
    shells out to tsh. Returns the CompletedProcess (stdout carries the verdict word / echo).
    """
    prelude = f'set -uo pipefail\nROOT={str(ROOT)!r}\nKUBE_CONTEXT=""\nNAMESPACE="ns"\n' f". {str(HELPER)!r}\n"
    return subprocess.run(["bash", "-c", prelude + snippet], capture_output=True, text=True)


# ── 0. static lint ────────────────────────────────────────────────────────────
shn = subprocess.run(["bash", "-n", str(HELPER)], capture_output=True, text=True)
check(
    "_kubectl_resilient.sh is valid bash (bash -n)",
    shn.returncode == 0,
    shn.stderr.strip()[:200],
)


# ── 1. PURE kernel: _verdict_from_state <succ> <fail> <active> <query_ok> ──────
# The exact truth table the fix hinges on.
def verdict(s: str, f: str, a: str, ok: str) -> str:
    return _bash(f"_verdict_from_state {s} {f} {a} {ok}").stdout.strip()


# succeeded>=1 → success (the disconnect-proof win; logs-follow may have died AFTER the Job Succeeded)
check("succeeded=1 (query ok)      → success", verdict("1", "0", "0", "1") == "success")
check("succeeded=5 many rungs      → success", verdict("5", "0", "0", "1") == "success")
# failed>=1 AND authoritatively read → failure (the ONLY path to a ❌)
check("failed=1 (query ok)         → failure", verdict("0", "1", "0", "1") == "failure")
# THE REGRESSION GUARD: unreachable apiserver must NEVER read as failure.
check(
    "UNREACHABLE '?|?|?' ok=0    → wait (NOT failure)",
    verdict("?", "?", "?", "0") == "wait",
)
check(
    "failed=1 but query_ok=0     → wait (NOT failure — unread/stale)",
    verdict("0", "1", "0", "0") == "wait",
)
# still-active / no terminal yet → wait
check("active=1 no terminal        → wait", verdict("0", "0", "1", "1") == "wait")
check("all-zero (mid-run)          → wait", verdict("0", "0", "0", "1") == "wait")
# succeeded wins even if a failed count is also present (Complete is terminal-good)
check(
    "succeeded=1 & failed=1      → success (Complete wins)",
    verdict("1", "1", "0", "1") == "success",
)


# ── 2. SHIPPING classifier llmb::job_state, with llmb::kc_resilient mocked ─────
# job_state redirects kc_resilient stdout → a temp file and reads its rc + $LLMB_KC_STDERR. We override
# kc_resilient AFTER sourcing to drive each branch. Echoes: succeeded|failed|running|unknown|notfound.
def job_state(kc_body: str) -> str:
    return _bash(f"llmb::kc_resilient() {{ {kc_body}; }}\nllmb::job_state job ns").stdout.strip()


check(
    "job_state: succeeded triple '1|0|0' → succeeded",
    job_state('printf "1|0|0"; return 0') == "succeeded",
)
check(
    "job_state: failed triple '0|1|0' (query ok) → failed",
    job_state('printf "0|1|0"; return 0') == "failed",
)
check(
    "job_state: active only '0|0|1' → running",
    job_state('printf "0|0|1"; return 0') == "running",
)
# THE REGRESSION GUARD on the real function: a TRANSIENT query error (cert expiry / apiserver unreachable)
# must classify as 'unknown' (→ follow keeps waiting + re-auths), NEVER 'failed'.
check(
    "job_state: transient rc (cert expiry) → unknown (NOT failed)",
    job_state(
        'LLMB_KC_STDERR="Unable to connect to the server: x509 certificate has expired"; ' 'return "$LLMB_RC_TRANSIENT"'
    )
    == "unknown",
)
check(
    "job_state: generic non-NotFound error → unknown (safe keep-waiting)",
    job_state('LLMB_KC_STDERR="some unexpected error"; return 1') == "unknown",
)
check(
    "job_state: NotFound after query error → notfound",
    job_state('LLMB_KC_STDERR="Error from server (NotFound): jobs.batch \\"job\\" not found"; return 1') == "notfound",
)


# ── 3. SHIPPING loop llmb::follow_job_to_terminal maps state → exit code ───────
# Override llmb::job_state so the first poll is terminal (no sleep). succeeded→0, failed→1.
def follow_rc(state: str) -> str:
    return (
        _bash(f"llmb::job_state() {{ echo {state}; }}\n" f"llmb::follow_job_to_terminal job ns 0; echo RC=$?")
        .stdout.strip()
        .splitlines()[-1]
    )


check("follow_job_to_terminal: state succeeded → RC=0", follow_rc("succeeded") == "RC=0")
check("follow_job_to_terminal: state failed    → RC=1", follow_rc("failed") == "RC=1")


# ── 4. sweep.sh / fetch_results.sh actually wire the helper in ─────────────────
sweep = (ROOT / "scripts" / "sweep.sh").read_text()
fetch = (ROOT / "scripts" / "fetch_results.sh").read_text()
check("sweep.sh sources _kubectl_resilient.sh", "_kubectl_resilient.sh" in sweep)
check(
    "sweep.sh uses llmb::follow_job_to_terminal (authoritative verdict)",
    "llmb::follow_job_to_terminal" in sweep,
)
# the false-failure default must be gone from LIVE code (a comment documenting the old bug is fine).
_sweep_code = "\n".join(ln for ln in sweep.splitlines() if not ln.lstrip().startswith("#"))
check(
    "sweep.sh no longer defaults a failed query to '0|1|0' (live code)",
    '"0|1|0"' not in _sweep_code,
)
check(
    "fetch_results.sh re-auths between retries (llmb::reauth / llmb::heal_auth)",
    "llmb::reauth" in fetch and "llmb::heal_auth" in fetch,
)
check(
    "fetch_results.sh guards a stale mounter pod (delete pod --ignore-not-found)",
    "delete pod" in fetch and "--ignore-not-found" in fetch,
)

# ── done ──
print()
if fails:
    print(f"selftest_run_verdict: {len(fails)} FAILED: {fails}")
    sys.exit(1)
print("selftest_run_verdict: all checks passed")

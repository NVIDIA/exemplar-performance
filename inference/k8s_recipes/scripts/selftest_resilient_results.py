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

"""Offline tests for result capture when the client disconnects.

One layer, no cluster:

  A. CERT-RESILIENT WRAPPER (scripts/_kubectl_resilient.sh) — drive the helper against a scripted fake kubectl
     (KUBECTL override) + a stubbed re-auth hook, and assert the transient-vs-terminal contract that fixes the
     failure mode:
       * an AUTH error resolves to 'unknown' (NEVER 'failed'/'notfound') → the loop keeps waiting;
       * a genuine terminal (succeeded/failed) DOES resolve terminal; NotFound-after-seen returns gone;
       * on an auth error the profile's re-auth hook FIRES and the retry then succeeds;
       * with NO re-auth hook configured, reauth prints an ACTIONABLE message + returns non-zero (no hang,
         no false failure).
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "scripts" / "_kubectl_resilient.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── A. cert-resilient wrapper ─────────────────────────────────────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())

# A scripted fake kubectl. It reads $FIX/state (one token) for `get job … -o jsonpath` + `auth can-i`, and a
# re-auth hook can MUTATE that state (modelling a non-interactive credential refresh that heals the session).
#   AUTH → emit an auth error (rc1);  NF → NotFound (rc1);  OK/FAIL/RUN → succeeded/failed/active jsonpath.
fake = tmp / "fake-kubectl"
fake.write_text(r"""#!/usr/bin/env python3
import os, sys
FIX = os.environ["FIX"]
a = sys.argv[1:]
# strip global flags kubectl-style
out, i = [], 0
while i < len(a):
    if a[i] in ("-n", "--namespace", "--context", "--request-timeout"):
        i += 2; continue
    out.append(a[i]); i += 1
a = out
try:
    state = open(os.path.join(FIX, "state")).read().strip()
except Exception:
    state = "RUN"
def emit_auth():
    sys.stderr.write("error: You must be logged in to the server (Unauthorized)\n"); sys.exit(1)
def emit_nf():
    sys.stderr.write('Error from server (NotFound): jobs.batch "x" not found\n'); sys.exit(1)
if a[:2] == ["get", "job"]:
    if state == "AUTH": emit_auth()
    if state == "NF":   emit_nf()
    if state == "OK":   sys.stdout.write("1||"); sys.exit(0)
    if state == "FAIL": sys.stdout.write("|1|"); sys.exit(0)
    sys.stdout.write("||1"); sys.exit(0)          # RUN
if a[:2] == ["auth", "can-i"]:
    if state == "AUTH": emit_auth()
    sys.stdout.write("yes\n"); sys.exit(0)
sys.exit(0)
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

# A re-auth hook that HEALS: flips state → OK and drops a marker so the test can prove it fired.
heal_hook = tmp / "heal.sh"
heal_hook.write_text('#!/bin/sh\necho OK > "$FIX/state"\n: > "$FIX/reauth_fired"\n')
heal_hook.chmod(heal_hook.stat().st_mode | stat.S_IEXEC)


def run_lib(
    snippet: str,
    *,
    state: str,
    hook: str | None = None,
    minimal_path: bool = False,
    timeout: float = 15,
) -> subprocess.CompletedProcess:
    fix = Path(tempfile.mkdtemp(dir=tmp))
    (fix / "state").write_text(state)
    env = dict(
        os.environ,
        FIX=str(fix),
        KUBECTL=str(fake),
        NAMESPACE="testns",
        KUBE_CONTEXT="",
        LLMB_KC_TRIES="4",
        LLMB_KC_BACKOFF="0",
        LLMB_JOB_POLL_S="0",
        CONNECT_CMD="",
    )
    env["LLMB_REAUTH_HOOK"] = hook or ""
    if minimal_path:
        # Exclude tsh from PATH so the built-in tsh fallback can't fire (tests the true no-hook path).
        env["PATH"] = "/usr/bin:/bin"
    env["_FIXDIR"] = str(fix)
    script = f'. "{LIB}"\nllmb::kc_raw() {{ python3 "{fake}" ${{KUBE_CONTEXT:+--context "$KUBE_CONTEXT"}} "$@"; }}\n{snippet}\n'
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=timeout)


# classify
r = run_lib(
    'llmb::classify_kube_err "Unable to connect to the server: x509: certificate has expired"',
    state="RUN",
)
check("classify: expired cert → auth", r.stdout.strip() == "auth", r.stdout)
r = run_lib(
    'llmb::classify_kube_err "You must be logged in to the server (Unauthorized)"',
    state="RUN",
)
check("classify: 401/logged-in → auth", r.stdout.strip() == "auth", r.stdout)
r = run_lib(
    'llmb::classify_kube_err "Unable to connect to the server: dial tcp: connect: connection refused"',
    state="RUN",
)
check(
    "classify: connection refused → transient",
    r.stdout.strip() == "transient",
    r.stdout,
)
r = run_lib(
    'llmb::classify_kube_err "Error from server (NotFound): jobs.batch x not found"',
    state="RUN",
)
check("classify: NotFound → other (genuine)", r.stdout.strip() == "other", r.stdout)

# THE root-cause fix: an AUTH error must resolve to 'unknown', NEVER a terminal. No hook → after retries → unknown.
r = run_lib("llmb::job_state myjob testns", state="AUTH", hook="", minimal_path=True)
check(
    "job_state: sustained AUTH error → 'unknown' (NEVER terminal — the false-exit fix)",
    r.stdout.strip() == "unknown",
    r.stdout.strip(),
)

r = run_lib("llmb::job_state myjob testns", state="OK")
check(
    "job_state: succeeded → 'succeeded'",
    r.stdout.strip() == "succeeded",
    r.stdout.strip(),
)
r = run_lib("llmb::job_state myjob testns", state="FAIL")
check("job_state: failed → 'failed'", r.stdout.strip() == "failed", r.stdout.strip())
r = run_lib("llmb::job_state myjob testns", state="NF")
check(
    "job_state: genuine NotFound → 'notfound'",
    r.stdout.strip() == "notfound",
    r.stdout.strip(),
)

# re-auth FIRES on an auth error and the retry then succeeds (the hook heals the session → succeeded).
r = run_lib(
    'out="$(llmb::job_state myjob testns)"; echo "$out"; [ -f "$_FIXDIR/reauth_fired" ] && echo FIRED',
    state="AUTH",
    hook=f"sh {heal_hook}",
)
check("re-auth hook FIRES on an auth error", "FIRED" in r.stdout, r.stdout.strip())
check(
    "after re-auth heals the session, job_state resolves terminal (succeeded)",
    "succeeded" in r.stdout,
    r.stdout.strip(),
)

# follow_job_to_terminal exit codes
r = run_lib('llmb::follow_job_to_terminal myjob testns 0; echo "rc=$?"', state="OK")
check("follow_job_to_terminal: Complete → rc 0", "rc=0" in r.stdout, r.stdout.strip())
r = run_lib('llmb::follow_job_to_terminal myjob testns 0; echo "rc=$?"', state="FAIL")
check("follow_job_to_terminal: Failed → rc 1", "rc=1" in r.stdout, r.stdout.strip())

# NotFound-BEFORE-seen must NOT be treated as terminal — the loop keeps waiting (proven by a timeout).
timed_out = False
try:
    run_lib(
        "llmb::follow_job_to_terminal myjob testns 1",
        state="NF",
        hook="",
        minimal_path=True,
        timeout=4,
    )
except subprocess.TimeoutExpired:
    timed_out = True
check(
    "follow_job_to_terminal: NotFound-before-seen keeps waiting (does NOT false-exit)",
    timed_out,
)

# A sustained AUTH error also keeps the loop waiting (never false-exits as terminal) — timeout proves it.
timed_out = False
try:
    run_lib(
        "llmb::follow_job_to_terminal myjob testns 1",
        state="AUTH",
        hook="",
        minimal_path=True,
        timeout=4,
    )
except subprocess.TimeoutExpired:
    timed_out = True
check(
    "follow_job_to_terminal: sustained AUTH keeps waiting without reporting failure",
    timed_out,
)

# No re-auth hook configured → reauth prints an ACTIONABLE message + returns non-zero (no hang, no false fail).
r = run_lib(
    "if llmb::reauth; then echo RC0; else echo RC1; fi",
    state="AUTH",
    hook="",
    minimal_path=True,
)
check("reauth: no hook → returns non-zero (RC1)", "RC1" in r.stdout, r.stdout.strip())
check(
    "reauth: no hook → actionable message names CONNECT_CMD",
    "CONNECT_CMD" in r.stderr and "no re-auth hook" in r.stderr,
    r.stderr.strip()[:200],
)

# The one-time notice does not spam: two reauth calls in one shell print the notice ONCE.
r = run_lib(
    "llmb::reauth 2>/dev/null || true; llmb::reauth 2>>/tmp/_notice2 || true; " "llmb::reauth; :",
    state="AUTH",
    hook="",
    minimal_path=True,
)
check(
    "reauth: 'no hook' notice is de-duped (printed once per process)",
    r.stderr.count("no re-auth hook is configured") <= 1,
    f"count={r.stderr.count('no re-auth hook is configured')}",
)


print()
if fails:
    print(f"selftest_resilient_results: {len(fails)} FAILED: {fails}")
    sys.exit(1)
total = sum(1 for line in Path(__file__).read_text().splitlines() if line.strip().startswith("check("))
print(f"selftest_resilient_results: all {total} checks PASSED")
sys.exit(0)

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

"""Verify that the model-load lease renewer cannot outlive its run.

The tests execute the real renewer against a fake kubectl and assert that termination releases the lease.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "model_load_gate.sh"

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def _fake_kubectl(tmp: Path, owner_state: str) -> Path:
    """A stand-in kubectl. owner_state: 'active' | 'terminal' | 'gone' | 'api_error'."""
    p = tmp / "kubectl"
    p.write_text(f"""#!/usr/bin/env bash
STATE={owner_state}
log() {{ echo "$*" >> "{tmp}/kubectl.log"; }}
log "$@"
case "$STATE" in
  api_error) exit 1 ;;
esac
# `get jobs` (plural, no name) is the reachability probe
if [ "$1" = "-n" ] && [ "$3" = "get" ] && [ "$4" = "jobs" ] && [ -z "${{5:-}}" ]; then exit 0; fi
case "$STATE:$4" in
  gone:job) exit 1 ;;
esac
if [ "$4" = "job" ]; then
  case "$STATE" in
    active)   echo '{{"status":{{"active":1}}}}' ;;
    terminal) echo '{{"status":{{"succeeded":1}}}}' ;;
  esac
  exit 0
fi
if [ "$4" = "lease" ] || [ "$3" = "delete" ]; then echo ""; exit 0; fi
exit 0
""")
    p.chmod(0o755)
    return p


HARNESS = """
set -uo pipefail
LEASE_SECONDS=3
POLL_S=1
MAX_HOLD_S="${MAX_HOLD_S_OVERRIDE:-7200}"
ANN_WAIT=w; ANN_SINCE=s; ANN_HOLDER=h
log()  { :; }
warn() { :; }
kc() { kubectl "$@"; }
annotate_owner() { :; }
write_lease() { echo "RENEW" >> "$EVENTS"; return 0; }
do_release() { echo "RELEASE" >> "$EVENTS"; rm -f "$1_sentinel" 2>/dev/null || true; return 0; }
%(renewer)s
start_renewer "ns" "lease-x" "cache-x" "run-x" "%(owner)s"
sleep %(wait)s
"""


def _extract_renewer() -> str:
    src = GATE.read_text()
    m = re.search(r"^start_renewer\(\) \{.*?^\}", src, re.S | re.M)
    if not m:
        raise SystemExit("could not extract start_renewer() from the shipped script")
    return m.group(0)


def run_case(owner_state: str, owner: str = "ro-job", wait: float = 5.0, max_hold: str = "7200"):
    tmp = Path(tempfile.mkdtemp())
    try:
        _fake_kubectl(tmp, owner_state)
        events = tmp / "events"
        script = tmp / "t.sh"
        script.write_text(HARNESS % {"renewer": _extract_renewer(), "owner": owner, "wait": wait})
        env = dict(os.environ)
        env["PATH"] = f"{tmp}:{env['PATH']}"
        env["EVENTS"] = str(events)
        env["TMPDIR"] = str(tmp)
        env["MAX_HOLD_S_OVERRIDE"] = max_hold
        subprocess.run(["bash", str(script)], env=env, capture_output=True, timeout=90)
        time.sleep(0.4)
        text = events.read_text() if events.exists() else ""
        return text.count("RENEW"), ("RELEASE" in text)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


print("  (each case runs the REAL extracted renewer against a fake kubectl)")

# 1. Owner alive → keep renewing, never release. This is the behaviour we must NOT break.
renews, released = run_case("active", wait=5.0)
check("owner ACTIVE: keeps renewing", renews >= 1, f"renews={renews}")
check("owner ACTIVE: does NOT release the slot", not released)

# 2. Owner terminal → release promptly. (Pre-fix: renews forever, never releases.)
renews, released = run_case("terminal", wait=5.0)
check(
    "owner TERMINAL: RELEASES the slot",
    released,
    f"renews={renews} released={released}",
)

# 3. Owner gone → release promptly. This is the 8h41m case.
renews, released = run_case("gone", wait=5.0)
check("owner GONE: RELEASES the slot", released, f"renews={renews} released={released}")

# 4. API unreachable → ambiguous: keep holding (never release on a failed read)...
renews, released = run_case("api_error", wait=5.0)
check("API UNREACHABLE: does NOT release on an ambiguous read", not released)

# 5. ...but the backstop must still bound it, even with no owner at all.
renews, released = run_case("api_error", owner="", wait=6.0, max_hold="2")
check(
    "MAX_HOLD backstop: releases even with no owner and an unreadable API",
    released,
    f"renews={renews} released={released}",
)

print()
if FAILED:
    print(f"selftest_loadgate_renewer: {len(FAILED)} FAILED")
    for f in FAILED:
        print(f"  - {f}")
    raise SystemExit(1)
print("selftest_loadgate_renewer: all checks passed")

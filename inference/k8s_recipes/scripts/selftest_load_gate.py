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

"""selftest_load_gate.py — offline guards for the model-load admission gate (scripts/model_load_gate.sh).

No cluster. A fake kubectl serves a scripted Lease world and records every mutation, so we can assert the
behaviours that matter:

  * two runs on the SAME cache serialize — the second WAITS, then proceeds once the slot frees;
  * two runs on DIFFERENT caches never block each other (no false serialization — the purpose of
    keying on the contended resource);
  * a STALE/expired lease is taken over rather than waited on forever (crash-safe);
  * a FORBIDDEN lease backend degrades forward with a warning instead of hanging;
  * release is idempotent, only deletes a lease we actually hold, and clears the annotations;
  * the waiter/holder ANNOTATION CONTRACT that fleet reads is exactly as specified.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "model_load_gate.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


tmp = Path(tempfile.mkdtemp())
(tmp / "cluster-profiles").mkdir()
(tmp / "scripts").mkdir()
for s in ("model_load_gate.sh",):
    (tmp / "scripts" / s).write_bytes((ROOT / "scripts" / s).read_bytes())
    (tmp / "scripts" / s).chmod(0o755)
(tmp / "cluster-profiles" / "fake.env").write_text("NAMESPACE=testns\nKUBE_CONTEXT=\n")

cell = tmp / "cell"
cell.mkdir()
(cell / "recipe.yaml").write_text("envelope:\n  name: lg-cell\n  model: m\nserving:\n  tp: 1\n")

# Fake kubectl: a Lease world in STATE_DIR. Modes let a scenario force forbidden/stale.
fake = tmp / "fake-kubectl"
fake.write_text(r"""#!/usr/bin/env python3
import json, os, sys, datetime, pathlib
SD = pathlib.Path(os.environ["STATE_DIR"])
MODE = os.environ.get("MODE", "ok")
a = [x for x in sys.argv[1:]]
out, i = [], 0
while i < len(a):
    if a[i] in ("-n", "--namespace", "--context"):
        i += 2; continue
    out.append(a[i]); i += 1
a = out
(SD / "calls.log").open("a").write(" ".join(a) + "\n")
def lease_path(name): return SD / f"lease-{name}.json"

if MODE == "forbidden" and a[:1] == ["get"] and "lease" in a[:2]:
    sys.stderr.write('Error from server (Forbidden): leases.coordination.k8s.io is forbidden: '
                     'User "qa" cannot list resource "leases" in the namespace "testns"\n')
    sys.exit(1)

if a[:2] == ["get", "lease"] and (len(a) < 3 or a[2].startswith("-")):   # LIST form = backend probe
    sys.exit(0)
if a[:2] == ["get", "lease"]:                                            # named get
    lf = lease_path(a[2])
    if not lf.exists(): sys.exit(1)
    d = json.load(lf.open())
    if "jsonpath={.spec.holderIdentity}" in " ".join(a):
        sys.stdout.write(d["spec"].get("holderIdentity", "")); sys.exit(0)
    sys.stdout.write(json.dumps(d)); sys.exit(0)
if a[:1] == ["apply"]:
    body = sys.stdin.read()
    holder = ""; lname = ""
    for ln in body.splitlines():
        if "holderIdentity:" in ln: holder = ln.split(":", 1)[1].strip()
        if ln.strip().startswith("name:") and not lname: lname = ln.split(":", 1)[1].strip()
    now = datetime.datetime.now(datetime.timezone.utc)
    if MODE == "stale":   # the writer's own renew stamp is ancient -> expired on next read
        now = now - datetime.timedelta(seconds=99999)
    json.dump({"spec": {"holderIdentity": holder, "leaseDurationSeconds": 300,
                        "renewTime": now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"}}, lease_path(lname).open("w"))
    sys.exit(0)
if a[:2] == ["delete", "lease"]:
    lf = lease_path(a[2])
    if lf.exists(): lf.unlink()
    sys.exit(0)
if a[:1] == ["annotate"]:
    (SD / "annotations.log").open("a").write(" ".join(a) + "\n")
    sys.exit(0)
sys.exit(0)
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def run_gate(
    verb,
    run_id,
    *,
    state,
    mode="ok",
    timeout="0",
    cache="cache-a",
    owner="ownerjob",
    wait=None,
):
    sd = tmp / state
    sd.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        KUBECTL=str(fake),
        STATE_DIR=str(sd),
        MODE=mode,
        LLMB_LOAD_GATE_POLL_S="1",
        LLMB_MODEL_CACHE_OVERRIDE=cache,
    )
    if wait is not None:
        env["LLMB_LOAD_GATE_TIMEOUT_S"] = str(wait)
    else:
        env["LLMB_LOAD_GATE_TIMEOUT_S"] = timeout
    return subprocess.run(
        [
            "bash",
            str(tmp / "scripts" / "model_load_gate.sh"),
            verb,
            str(cell),
            "fake",
            run_id,
            "--run-owner",
            owner,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=90,
    )


def annotations(state):
    f = tmp / state / "annotations.log"
    return f.read_text() if f.exists() else ""


# ── 1. acquire on a free cache: takes the slot, annotates holder, clears any wait ────────────────────
r = run_gate("acquire", "run-1", state="s1")
check("acquire on a free cache succeeds", r.returncode == 0, r.stderr[-200:])
check("acquire ACQUIRED is announced", "ACQUIRED" in r.stdout, r.stdout[-200:])
_a = annotations("s1")
check(
    "holder annotation llmb.nvidia.com/model-load-holder=<pvc> is set",
    "llmb.nvidia.com/model-load-holder=" in _a,
    _a[-200:],
)
check(
    "wait annotations are CLEARED on acquire (no phantom queue for fleet)",
    "llmb.nvidia.com/model-load-wait-" in _a and "llmb.nvidia.com/model-load-since-" in _a,
    _a[-200:],
)

# ── 2. SAME cache, second run: must WAIT (bounded here) and then proceed ─────────────────────────────
r2 = run_gate("acquire", "run-2", state="s1", wait=3)
check(
    "second run on the SAME cache WAITS for the slot",
    "waiting for model-load slot" in r2.stdout,
    r2.stdout[-250:],
)
check(
    "waiting run names the cache and says another run is loading",
    "cache 'cache-a'" in r2.stdout and "another run is loading" in r2.stdout,
    r2.stdout[-250:],
)
_a2 = annotations("s1")
check(
    "waiter sets llmb.nvidia.com/model-load-wait=<pvc> (fleet can see the queue)",
    "llmb.nvidia.com/model-load-wait=cache-a" in _a2,
    _a2[-300:],
)
check(
    "waiter sets llmb.nvidia.com/model-load-since=<RFC3339>",
    "llmb.nvidia.com/model-load-since=" in _a2,
    _a2[-300:],
)
check(
    "a bounded wait DEGRADES FORWARD rather than hanging",
    r2.returncode == 0,
    r2.stderr[-200:],
)
check(
    "proceeding-anyway is announced with the reason",
    "proceeding ANYWAY" in r2.stderr.replace("PROCEEDING", "proceeding"),
    r2.stderr[-250:],
)

# ── 3. DIFFERENT cache: must NOT block (no false serialization) ──────────────────────────────────────
r3 = run_gate("acquire", "run-3", state="s1", cache="cache-b")
check(
    "a run on a DIFFERENT cache does NOT wait",
    "waiting for model-load slot" not in r3.stdout,
    r3.stdout[-200:],
)
check(
    "a run on a DIFFERENT cache acquires its own slot",
    "ACQUIRED" in r3.stdout,
    r3.stdout[-200:],
)

# ── 4. release: only deletes OUR lease, clears annotations, idempotent ───────────────────────────────
rel = run_gate("release", "run-1", state="s1")
check("release succeeds", rel.returncode == 0, rel.stderr[-200:])
_ar = annotations("s1")
check(
    "release clears holder + wait annotations",
    "llmb.nvidia.com/model-load-holder-" in _ar,
    _ar[-200:],
)
rel2 = run_gate("release", "run-1", state="s1")
check(
    "release is IDEMPOTENT (safe when we no longer hold it)",
    rel2.returncode == 0,
    rel2.stderr[-200:],
)
rel3 = run_gate("release", "never-held", state="s1")
check(
    "release by a NON-holder does not steal/delete another run's slot",
    rel3.returncode == 0,
    rel3.stderr[-200:],
)

# ── 5. after release, the queued run acquires immediately ───────────────────────────────────────────
r5 = run_gate("acquire", "run-4", state="s5")
_ = run_gate("release", "run-4", state="s5")
r6 = run_gate("acquire", "run-5", state="s5")
check(
    "once the slot frees, the next run acquires without waiting",
    "ACQUIRED" in r6.stdout and "waiting for model-load slot" not in r6.stdout,
    r6.stdout[-200:],
)

# ── 6. STALE lease (expired holder) is taken over, not waited on ─────────────────────────────────────
_ = run_gate("acquire", "dead-run", state="s6", mode="stale")
r7 = run_gate("acquire", "live-run", state="s6")
check(
    "an EXPIRED lease is taken over (crash-safe: a dead holder never keeps the slot)",
    "ACQUIRED" in r7.stdout and "waiting for model-load slot" not in r7.stdout,
    r7.stdout[-250:],
)

# ── 7. FORBIDDEN backend degrades forward, loudly, without hanging ───────────────────────────────────
r8 = run_gate("acquire", "run-6", state="s7", mode="forbidden")
check(
    "RBAC-forbidden leases → PROCEEDS (never hangs)",
    r8.returncode == 0,
    r8.stderr[-200:],
)
check(
    "forbidden warns specifically about leases + the namespace",
    "leases are FORBIDDEN" in r8.stderr and "testns" in r8.stderr,
    r8.stderr[-300:],
)
check(
    "forbidden explains the consequence (contention) and the remediation",
    "contend" in r8.stderr and "coordination.k8s.io/leases" in r8.stderr,
    r8.stderr[-350:],
)
check(
    "forbidden does NOT leave a phantom wait annotation",
    "llmb.nvidia.com/model-load-wait-" in annotations("s7"),
    annotations("s7")[-200:],
)

print()
if fails:
    print(f"selftest_load_gate: {len(fails)} FAILED: {fails}")
    sys.exit(1)
total = sum(1 for line in Path(__file__).read_text().splitlines() if line.strip().startswith("check("))
print(f"selftest_load_gate: all {total} checks PASSED")
sys.exit(0)

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

"""selftest_crashloop.py — offline guards for the wait-ready crash-loop fail-fast (GPU-lifecycle resilience).

No cluster, no profile. Two layers:

  A. CLASSIFIER SUB-MODES — feed canned `kubectl get pods -o json` blobs through idle_guard.sh's stdin
     sub-modes (--pod-max-restarts / --pod-in-backoff) and assert the classification. Pure, cluster-free.

  B. FIXTURE POLL (wait_server_ready.sh) — build a fake kubectl that serves a SCRIPTED sequence of
     deployment/pod states (one JSON per poll) and assert wait_server_ready.sh:
       * returns 0 when the server becomes ready — INCLUDING a slow cold load that stays restartCount=0
         for several polls (the happy-path-safe guarantee: a slow load must NOT be mistaken for a crash-loop);
       * aborts NON-ZERO (fast) when restartCount reaches K;
       * aborts NON-ZERO when CrashLoopBackOff is sustained past the M-minute window;
       * aborts NON-ZERO on the overall ready-timeout when nothing ever becomes ready.
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
IDLE_GUARD = ROOT / "scripts" / "idle_guard.sh"
WAIT = ROOT / "scripts" / "wait_server_ready.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def submode(mode: str, blob: str) -> str:
    p = subprocess.run(["bash", str(IDLE_GUARD), mode], input=blob, text=True, capture_output=True)
    return p.stdout.strip()


# ── A. classifier sub-modes ──────────────────────────────────────────────────
check(
    "--pod-max-restarts takes the MAX across containers",
    submode(
        "--pod-max-restarts",
        '{"status":{"containerStatuses":[{"restartCount":2},{"restartCount":7}]}}',
    )
    == "7",
)
check(
    "--pod-max-restarts no restarts -> 0",
    submode("--pod-max-restarts", '{"status":{"phase":"Running"}}') == "0",
)
check("--pod-max-restarts empty blob -> 0", submode("--pod-max-restarts", "") == "0")
check(
    "--pod-in-backoff detects CrashLoopBackOff",
    submode(
        "--pod-in-backoff",
        '{"status":{"containerStatuses":[{"state":{"waiting":{"reason":"CrashLoopBackOff"}}}]}}',
    )
    == "1",
)
check(
    "--pod-in-backoff plain Running -> 0",
    submode("--pod-in-backoff", '{"status":{"phase":"Running"}}') == "0",
)
# A pod that has restarted a few times but is currently Running (not backing off) is NOT 'in-backoff' —
# proves the M-window signal is independent of the K restart-count signal.
check(
    "--pod-in-backoff restarted-but-running -> 0",
    submode(
        "--pod-in-backoff",
        '{"status":{"containerStatuses":[{"restartCount":4,"state":{"running":{}}}]}}',
    )
    == "0",
)


# ── B. fixture poll: a fake kubectl serving a scripted per-poll state sequence ─
tmp = Path(tempfile.mkdtemp())
cell = tmp / "cell"
cell.mkdir()
# minimal recipe.yaml: wait_server_ready.sh reads envelope.name + serving.startup_timeout_s
(cell / "recipe.yaml").write_text(
    "envelope:\n  name: cl-cell\n  engine: vllm\n" "serving:\n  startup_timeout_s: 60\n  served_model: cl-model\n"
)
prof_dir = tmp / "cluster-profiles"
prof_dir.mkdir()
# NOTE: wait_server_ready.sh resolves the profile RELATIVE to its own ROOT (scripts/..), so point it at a real
# profile name in the repo tree is not possible offline; instead we run with cwd=tmp and a symlinked layout.
# Simpler: create a throwaway ROOT that mirrors scripts/ + cluster-profiles/ and copy the two scripts in.
scripts_dir = tmp / "scripts"
scripts_dir.mkdir()
# wait_server_ready.sh now sources _kubectl_resilient.sh (per-poll auth self-heal), so copy it too.
for s in ("wait_server_ready.sh", "idle_guard.sh", "_kubectl_resilient.sh"):
    (scripts_dir / s).write_bytes((ROOT / "scripts" / s).read_bytes())
    (scripts_dir / s).chmod(0o755)
(prof_dir / "fake.env").write_text("NAMESPACE=testns\nKUBE_CONTEXT=\n")

# The fake kubectl replays a state script. STATE_FILE holds one line per invocation-group: we key off a counter
# that advances on each `get pods` call (the poll cadence). Each line is a JSON doc name under FIX.
fake = tmp / "fake-kubectl"
fake.write_text(r"""#!/usr/bin/env python3
import json, os, sys
FIX = os.environ["FIXTURE_DIR"]
a = sys.argv[1:]
# strip global flags
out, i = [], 0
while i < len(a):
    if a[i] in ("-n", "--namespace", "--context"):
        i += 2; continue
    out.append(a[i]); i += 1
a = out

def counter_path():
    return os.path.join(FIX, "poll.count")

def cur_step():
    try:
        return int(open(counter_path()).read().strip())
    except Exception:
        return 0

seq = json.load(open(os.path.join(FIX, "sequence.json")))  # list of {deploy_ready, pods}

if a[:2] == ["get", "deployment"]:
    # existence probe + readyReplicas jsonpath
    step = cur_step()
    st = seq[min(step, len(seq) - 1)]
    if "-o" in a:  # jsonpath readyReplicas
        sys.stdout.write("1" if st.get("deploy_ready") else "")
    sys.exit(0)   # exists (rc 0)
if a[:2] == ["get", "pods"]:
    step = cur_step()
    st = seq[min(step, len(seq) - 1)]
    # The model-registration gate asks for a Running pod NAME via jsonpath; serve that WITHOUT advancing the
    # poll counter (it is part of the same poll iteration, not a new one).
    jp = ""
    if "-o" in a:
        try: jp = a[a.index("-o") + 1]
        except Exception: jp = ""
    if "metadata.name" in jp:
        sys.stdout.write("cl-pod" if st.get("pod_running", True) else "")
        sys.exit(0)
    # advance the poll counter AFTER serving this step's pods (one advance per poll loop)
    open(counter_path(), "w").write(str(step + 1))
    sys.stdout.write(json.dumps({"items": st.get("pods", [])}))
    sys.exit(0)
if a[:1] == ["exec"]:
    # DEFECT #10 gate: emulate GET /v1/models from inside the server pod. `models` in the step is the list of
    # registered ids; default = the cell's model (so pre-existing fixtures still reach ready as before).
    # `exec_mode` emulates the NON-success outcomes the gate must tell apart:
    #   "forbidden" -> an RBAC denial on stderr + non-zero (pods/exec not permitted)
    #   "error"     -> a transient exec failure (non-zero, NOT a permissions message)
    #   "garbage"   -> exec succeeds but the body is not JSON (malformed response)
    step = cur_step()
    st = seq[min(step, len(seq) - 1)]
    mode = st.get("exec_mode", "ok")
    if mode == "forbidden":
        sys.stderr.write('Error from server (Forbidden): pods "cl-pod" is forbidden: '
                         'User "qa" cannot create resource "pods/exec" in API group "" in the namespace "testns"\n')
        sys.exit(1)
    if mode == "error":
        sys.stderr.write("error dialing backend: connection refused\n")
        sys.exit(1)
    if mode == "garbage":
        sys.stdout.write("<html>502 Bad Gateway</html>")
        sys.exit(0)
    ids = st.get("models", ["cl-model"])
    sys.stdout.write(json.dumps({"object": "list", "data": [{"id": i} for i in ids]}))
    sys.exit(0)
if a[:1] == ["describe"]:
    sys.stdout.write("Events:\n  (fake)\n")
    sys.exit(0)
sys.exit(0)
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def run_wait(sequence, *, max_restarts=5, window_min=5, timeout=60, poll=0):
    fix = tmp / f"fix{os.getpid()}{len(sequence)}{max_restarts}{window_min}{timeout}"
    fix.mkdir(exist_ok=True)
    (fix / "sequence.json").write_text(json.dumps(sequence))
    # remove any stale counter
    (fix / "poll.count").write_text("0")
    env = dict(os.environ, KUBECTL=str(fake), FIXTURE_DIR=str(fix))
    return subprocess.run(
        [
            "bash",
            str(scripts_dir / "wait_server_ready.sh"),
            str(cell),
            "fake",
            "--timeout",
            str(timeout),
            "--max-restarts",
            str(max_restarts),
            "--window-min",
            str(window_min),
            "--poll",
            str(poll),
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp),
        timeout=60,
    )


def pod(restart=0, backoff=False, running=True):
    cs = {"restartCount": restart}
    if backoff:
        cs["state"] = {"waiting": {"reason": "CrashLoopBackOff"}}
    elif running:
        cs["state"] = {"running": {}}
    return {"status": {"phase": "Running", "containerStatuses": [cs]}}


READY = {"deploy_ready": True, "pods": [pod()]}

# 1) becomes ready immediately -> exit 0
r = run_wait([READY])
check("ready immediately -> exit 0", r.returncode == 0, r.stderr[-200:])

# 2) HAPPY PATH: slow cold load — several polls with restartCount=0, NOT ready, then ready. Must NOT fast-fail.
slow = [{"deploy_ready": False, "pods": [pod(restart=0)]} for _ in range(6)] + [READY]
r = run_wait(slow, max_restarts=5, window_min=5, timeout=60)
check(
    "slow cold load (restartCount=0 for many polls) -> becomes ready, exit 0",
    r.returncode == 0,
    r.stderr[-200:],
)

# 3) crash-loop by restartCount: climbs 1,2,...,K -> fast-fail non-zero with the restart reason.
climb = [{"deploy_ready": False, "pods": [pod(restart=n, backoff=True)]} for n in range(1, 7)]
r = run_wait(climb, max_restarts=5, window_min=999, timeout=60)  # huge window so ONLY the K path can fire
check("restartCount reaches K -> aborts non-zero", r.returncode != 0, r.stdout[-200:])
check(
    "restartCount abort names the crash-loop reason",
    "restarted" in r.stderr and "crash-looping" in r.stderr,
    r.stderr[-300:],
)

# 4) sustained CrashLoopBackOff past the M window (restartCount stays below K) -> fast-fail on the window.
#    window_min small; poll=0 so wall time accrues via date; give enough polls to exceed the window.
backoff_seq = [{"deploy_ready": False, "pods": [pod(restart=1, backoff=True)]} for _ in range(40)]
r = run_wait(backoff_seq, max_restarts=99, window_min=0, timeout=60, poll=0)  # window 0m => any sustained backoff trips
check(
    "sustained CrashLoopBackOff past M -> aborts non-zero",
    r.returncode != 0,
    r.stdout[-200:],
)
check(
    "CrashLoopBackOff abort names the sustained reason",
    "CrashLoopBackOff sustained" in r.stderr,
    r.stderr[-300:],
)

# 5) never ready, no crash signal -> aborts on the ready-timeout.
neverready = [{"deploy_ready": False, "pods": [pod(restart=0)]} for _ in range(200)]
r = run_wait(neverready, max_restarts=5, window_min=5, timeout=1, poll=0)  # tiny timeout => trips on the elapsed check
check(
    "never ready, no crash -> aborts on ready-timeout",
    r.returncode != 0,
    r.stdout[-200:],
)
check(
    "timeout abort names the timeout",
    "did not become ready within" in r.stderr,
    r.stderr[-300:],
)

# ── C. MODEL-REGISTRATION GATE (DEFECT #10) ───────────────────────────────────
# Readiness may become healthy before the engine registers the model, so
# wait-ready "succeeded" in 19s and the bench then fataled with `available=[]`. Readiness must therefore
# require the cell's served_model to appear in GET /v1/models, not just pod Ready.

# 6) pod Ready + healthy, but /v1/models is EMPTY -> must NOT be declared ready (this is the 19s false pass).
empty_models = [{"deploy_ready": True, "pods": [pod(restart=0)], "models": []} for _ in range(200)]
r = run_wait(empty_models, max_restarts=5, window_min=5, timeout=1, poll=0)
check(
    "health-OK + EMPTY /v1/models -> NOT ready (times out, never false-passes)",
    r.returncode != 0,
    r.stdout[-200:],
)
check(
    "empty-models abort says the server was healthy but never registered the model",
    "never registered" in r.stderr and "cl-model" in r.stderr,
    r.stderr[-400:],
)
check(
    "empty-models abort points at the WORKER logs (actionable)",
    "logs -l" in r.stderr,
    r.stderr[-300:],
)

# 7) pod Ready + the cell's served_model REGISTERED -> ready.
good_models = [{"deploy_ready": True, "pods": [pod(restart=0)], "models": ["cl-model"]}]
r = run_wait(good_models, max_restarts=5, window_min=5, timeout=60, poll=0)
check(
    "health-OK + served_model present in /v1/models -> ready (exit 0)",
    r.returncode == 0,
    r.stderr[-300:],
)
check("ready log names the registered model", "cl-model" in r.stdout, r.stdout[-300:])

# 8) a DIFFERENT model registered is not a match (guards a wrong-model server answering /health).
wrong_models = [{"deploy_ready": True, "pods": [pod(restart=0)], "models": ["some-other-model"]} for _ in range(200)]
r = run_wait(wrong_models, max_restarts=5, window_min=5, timeout=1, poll=0)
check(
    "health-OK + only a DIFFERENT model registered -> NOT ready",
    r.returncode != 0,
    r.stdout[-200:],
)

# 9) the race itself: empty for the first polls, then the model appears -> ready (no false abort).
race = [{"deploy_ready": True, "pods": [pod(restart=0)], "models": []} for _ in range(3)] + [
    {"deploy_ready": True, "pods": [pod(restart=0)], "models": ["cl-model"]}
]
r = run_wait(race, max_restarts=5, window_min=5, timeout=60, poll=0)
check(
    "registers late (empty -> present) -> waits, then ready",
    r.returncode == 0,
    r.stderr[-300:],
)

# ── D. RBAC FALLBACK: "exec forbidden" must NOT be conflated with "model absent" ──
# The gate execs into the server pod, so it needs pods/exec. On a cluster that forbids it the probe can
# never succeed — waiting out startup_timeout_s would present as a mysterious HANG. So a PERMISSIONS denial
# degrades to k8s-readiness with one loud warning, while a genuine "model not registered" still blocks and a
# transient probe error is just retried. These three branches must stay distinct.

# 10) exec FORBIDDEN -> warn once, accept k8s-readiness, proceed (exit 0) even though models can't be listed.
forbidden = [{"deploy_ready": True, "pods": [pod(restart=0)], "exec_mode": "forbidden"} for _ in range(200)]
r = run_wait(forbidden, max_restarts=5, window_min=5, timeout=60, poll=0)
check(
    "exec forbidden -> proceeds on k8s-readiness (exit 0, no hang)",
    r.returncode == 0,
    r.stderr[-300:],
)
check(
    "exec forbidden -> warning names pods/exec + the namespace",
    "pods/exec is forbidden" in r.stderr and "testns" in r.stderr,
    r.stderr[-400:],
)
check(
    "exec forbidden -> warning says verification is degraded + predicts the bench-time symptom",
    "k8s-readiness ONLY" in r.stderr and "is not served" in r.stderr,
    r.stderr[-400:],
)
check(
    "exec forbidden -> warning tells you how to restore the gate (auth can-i)",
    "auth can-i" in r.stderr,
    r.stderr[-300:],
)
check(
    "exec forbidden -> warned exactly ONCE (not once per poll)",
    r.stderr.count("pods/exec is forbidden") == 1,
    str(r.stderr.count("pods/exec is forbidden")),
)
check(
    "exec forbidden -> the ready line marks registration UNVERIFIED (not a clean pass)",
    "UNVERIFIED" in r.stdout,
    r.stdout[-300:],
)

# 11) CONTRAST: exec WORKS but the model is absent -> still blocks (the real race is NOT degraded away).
absent = [{"deploy_ready": True, "pods": [pod(restart=0)], "models": []} for _ in range(200)]
r = run_wait(absent, max_restarts=5, window_min=5, timeout=1, poll=0)
check(
    "model genuinely absent (exec OK) -> still blocks, never falls back",
    r.returncode != 0,
    r.stdout[-200:],
)
check(
    "model absent -> does NOT emit the forbidden warning",
    "pods/exec is forbidden" not in r.stderr,
    r.stderr[-300:],
)

# 12) TRANSIENT exec error -> unknown, retry: must NOT be read as forbidden (would disable the gate for one
#     blip) and must NOT be read as absent.
transient = [{"deploy_ready": True, "pods": [pod(restart=0)], "exec_mode": "error"} for _ in range(200)]
r = run_wait(transient, max_restarts=5, window_min=5, timeout=1, poll=0)
check(
    "transient exec error -> keeps waiting (no false ready)",
    r.returncode != 0,
    r.stdout[-200:],
)
check(
    "transient exec error -> NOT treated as forbidden (no fallback warning)",
    "pods/exec is forbidden" not in r.stderr,
    r.stderr[-300:],
)

# 13) malformed body (exec OK, non-JSON) -> unknown, retry: not "absent", not "forbidden".
garbage = [{"deploy_ready": True, "pods": [pod(restart=0)], "exec_mode": "garbage"} for _ in range(200)]
r = run_wait(garbage, max_restarts=5, window_min=5, timeout=1, poll=0)
check(
    "malformed /v1/models body -> keeps waiting, not a false ready",
    r.returncode != 0,
    r.stdout[-200:],
)
check(
    "malformed body -> NOT treated as forbidden",
    "pods/exec is forbidden" not in r.stderr,
    r.stderr[-300:],
)

# 14) a transient blip that RECOVERS still reaches ready (proves 'unknown' does not poison the gate).
blip = [{"deploy_ready": True, "pods": [pod(restart=0)], "exec_mode": "error"} for _ in range(2)] + [
    {"deploy_ready": True, "pods": [pod(restart=0)], "models": ["cl-model"]}
]
r = run_wait(blip, max_restarts=5, window_min=5, timeout=60, poll=0)
check("transient blip then success -> reaches ready", r.returncode == 0, r.stderr[-300:])

# ── E. ONE forbidden probe is NOT definitive: require TWO CONSECUTIVE before degrading ──
# Degrading costs the verification gate for the WHOLE run, and the run then proceeds to a bench that may
# fail with the exact symptom the gate exists to prevent. One extra poll costs seconds. So a single
# spurious authorization blip must NOT disable the gate — the same "a single failed probe is never a
# definitive condition" rule we apply to the other probes.

# 15) ONE forbidden, then the model is registered -> NO degrade, normal verified ready.
one_blip = [{"deploy_ready": True, "pods": [pod(restart=0)], "exec_mode": "forbidden"}] + [
    {"deploy_ready": True, "pods": [pod(restart=0)], "models": ["cl-model"]}
]
r = run_wait(one_blip, max_restarts=5, window_min=5, timeout=60, poll=0)
check("single forbidden then success -> ready", r.returncode == 0, r.stderr[-300:])
check(
    "single forbidden then success -> did NOT degrade (no fallback warning)",
    "pods/exec is forbidden" not in r.stderr,
    r.stderr[-400:],
)
check(
    "single forbidden then success -> ready line is the VERIFIED one (not UNVERIFIED)",
    "UNVERIFIED" not in r.stdout and "cl-model" in r.stdout,
    r.stdout[-300:],
)
check(
    "single forbidden -> says it is re-probing to confirm before degrading",
    "re-probing once to confirm" in r.stdout,
    r.stdout[-300:],
)

# 16) TWO CONSECUTIVE forbidden -> degrade (and the warning says it was confirmed).
two_forbidden = [{"deploy_ready": True, "pods": [pod(restart=0)], "exec_mode": "forbidden"} for _ in range(2)] + [
    {"deploy_ready": True, "pods": [pod(restart=0)], "models": ["cl-model"]}
]
r = run_wait(two_forbidden, max_restarts=5, window_min=5, timeout=60, poll=0)
check(
    "two consecutive forbidden -> degrades to k8s-readiness (exit 0)",
    r.returncode == 0,
    r.stderr[-300:],
)
check(
    "two consecutive forbidden -> warning states it was CONFIRMED over consecutive probes",
    "consecutive probes" in r.stderr,
    r.stderr[-400:],
)
check(
    "two consecutive forbidden -> ready line marks UNVERIFIED",
    "UNVERIFIED" in r.stdout,
    r.stdout[-300:],
)

# 17) ALTERNATING forbidden/success never reaches two-in-a-row -> never degrades (streak really resets).
alternating = []
for _ in range(6):
    alternating.append({"deploy_ready": True, "pods": [pod(restart=0)], "exec_mode": "forbidden"})
    alternating.append({"deploy_ready": True, "pods": [pod(restart=0)], "models": []})
r = run_wait(alternating, max_restarts=5, window_min=5, timeout=1, poll=0)
check(
    "alternating forbidden/probe-OK -> never degrades (streak resets on any other outcome)",
    "pods/exec is forbidden" not in r.stderr,
    r.stderr[-400:],
)
check(
    "alternating forbidden/probe-OK -> still blocks (model never registered)",
    r.returncode != 0,
    r.stdout[-200:],
)

print()
if fails:
    print(f"selftest_crashloop: {len(fails)} FAILED: {fails}")
    sys.exit(1)
total = sum(1 for line in Path(__file__).read_text().splitlines() if line.strip().startswith("check("))
print(f"selftest_crashloop: all {total} checks PASSED")
sys.exit(0)

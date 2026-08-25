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

"""selftest_governor.py — offline guards for the Phase-2 in-cluster resource governor.

No cluster, no profile. Two layers, both from the in-cluster governor contract:

  A. CLASSIFIER SUB-MODES (§8) — feed canned blobs through the reconcile's stdin sub-modes
     (--sum-tokens / --pod-crashlooping / --pod-pending / --job-verdict, ported verbatim from idle_guard.sh)
     and assert the classification. Pure, cluster-free.

  B. DRY_RUN FIXTURE RECONCILE (§2/§5/§12) — build a FIXTURE control-PVC directory with one <run-id>/ subdir
     per state (healthy-progressing, stalled, heartbeat-lost, timed-out, server-crashed, bootstrap-not-
     generating, completed) plus a FAKE kubectl that serves canned Job/Deployment/Pod JSON, run the reconcile
     with DRY_RUN=1, and assert the correct action (or no-op) is chosen for each — orphan-sweep included — and
     that ZERO mutating kubectl calls were issued.

Also lints the reconcile with `sh -n` and asserts the manifest bundle carries the exact RBAC verbs (§4).
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECONCILE = ROOT / "serving" / "governor" / "reconcile" / "governor_reconcile.sh"
TEMPLATE = ROOT / "serving" / "governor" / "templates" / "governor.yaml"
OBSERVE_TEMPLATE = ROOT / "serving" / "governor" / "templates" / "governor-observe.yaml"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def submode(mode: str, blob: str) -> str:
    p = subprocess.run(["sh", str(RECONCILE), mode], input=blob, text=True, capture_output=True)
    return p.stdout.strip()


# ── 0. static lints ──────────────────────────────────────────────────────────
shn = subprocess.run(["sh", "-n", str(RECONCILE)], capture_output=True, text=True)
check("reconcile is valid POSIX sh (sh -n)", shn.returncode == 0, shn.stderr.strip()[:200])

manifest = TEMPLATE.read_text()
# Exact RBAC verbs from the design §4 (order-insensitive within a rule).
rbac_expect = {
    ('["batch"]', '["jobs"]'): {"get", "list", "delete"},
    ('[""]', '["pods"]'): {"get", "list", "delete"},
    ('["apps"]', '["deployments"]'): {"get", "list", "delete"},
    ('["apps"]', '["deployments/scale"]'): {"get", "update", "patch"},
    ('[""]', '["services"]'): {"get", "list", "delete"},
    ('["networking.k8s.io"]', '["networkpolicies"]'): {"get", "list", "delete"},
}
for (grp, res), verbs in rbac_expect.items():
    m = re.search(
        r"apiGroups:\s*" + re.escape(grp) + r",\s*resources:\s*" + re.escape(res) + r",\s*verbs:\s*(\[[^\]]*\])",
        manifest,
    )
    got = set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
    check(f"RBAC {grp} {res} verbs == {sorted(verbs)}", got == verbs, f"got {sorted(got)}")
check(
    "RBAC has NO configmaps rule (§4: PVC replaced the CM channel)",
    '["configmaps"]' not in manifest,
)
# ignore explanatory comments — assert no actual rule GRANTS jobs/patch or a configmaps resource.
_rules = "\n".join(ln for ln in manifest.splitlines() if not ln.lstrip().startswith("#"))
check(
    "RBAC has NO jobs/patch grant (§4: progress is on the PVC)",
    '"jobs/patch"' not in _rules and "jobs/patch" not in _rules,
)
check(
    "CronJob has Forbid concurrency + DRY_RUN env toggle",
    "concurrencyPolicy: Forbid" in manifest and "name: DRY_RUN" in manifest,
)
# envsubst does NOT honor ${VAR:-default}; the template must use plain ${VAR} refs only (defaults live in
# apply-governor.sh). Assert no default-syntax leaked back in — it would render invalid YAML.
check(
    "template uses plain ${VAR} refs (no ${VAR:-default} that envsubst leaves literal)",
    ":-" not in "".join(ln for ln in manifest.splitlines() if "${" in ln and not ln.lstrip().startswith("#")),
)

# Render the bundle exactly as apply-governor.sh does (envsubst + defaults) and assert the pinned image /
# schedule / DRY_RUN survive into valid rendered YAML.
_renv = dict(
    os.environ,
    NAMESPACE="testns",
    CONTROL_STORAGE_CLASS="efs-sc",
    CONTROL_SIZE="5Gi",
    GOVERNOR_IMAGE="alpine/k8s:1.34.1",
    GOVERNOR_SCHEDULE="*/3 * * * *",
    STALL_THRESHOLD="1800",
    HEARTBEAT_DEAD="300",
    STUCK_THRESHOLD="900",
    TIMEOUT_MULT="2",
    MIN_KILL_SECONDS="3600",
    DRY_RUN="0",
)
_rendered = subprocess.run(["envsubst"], stdin=open(TEMPLATE), env=_renv, capture_output=True, text=True).stdout
try:
    import yaml as _yaml

    _docs = [d for d in _yaml.safe_load_all(_rendered) if isinstance(d, dict)]
    _cj = next(d for d in _docs if d.get("kind") == "CronJob")
    _c = _cj["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
    check(
        "rendered CronJob pins image alpine/k8s:1.34.1",
        _c["image"] == "alpine/k8s:1.34.1",
        _c.get("image"),
    )
    check(
        "rendered CronJob schedule == */3 * * * *",
        _cj["spec"]["schedule"] == "*/3 * * * *",
    )
    check(
        "rendered CronJob has DRY_RUN=0 env",
        {"name": "DRY_RUN", "value": "0"} in _c["env"],
    )
    check("rendered manifest has no unresolved ${...}", "${" not in _rendered)
except ImportError:
    check("pyyaml available to parse rendered bundle", False, "pyyaml missing")


# ── 0b. OBSERVE-mode manifest lints (the report-only safety-net RBAC must be REDUCED, never reap-capable) ──
obs = OBSERVE_TEMPLATE.read_text()
obs_rbac_expect = {
    ('["apps"]', '["deployments"]'): {"get", "list", "watch"},
    ('["batch"]', '["jobs"]'): {"get", "list", "watch"},
    ('[""]', '["pods"]'): {"get", "list", "watch"},
    ('[""]', '["configmaps"]'): {"get", "create", "update"},
}
for (grp, res), verbs in obs_rbac_expect.items():
    m = re.search(
        r"apiGroups:\s*" + re.escape(grp) + r",\s*resources:\s*" + re.escape(res) + r",\s*verbs:\s*(\[[^\]]*\])",
        obs,
    )
    got = set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
    check(
        f"OBSERVE RBAC {grp} {res} verbs == {sorted(verbs)}",
        got == verbs,
        f"got {sorted(got)}",
    )
_obs_rules = "\n".join(ln for ln in obs.splitlines() if not ln.lstrip().startswith("#"))
check(
    "OBSERVE RBAC grants NO delete (report-only cannot reap)",
    '"delete"' not in _obs_rules,
)
check("OBSERVE RBAC grants NO patch", '"patch"' not in _obs_rules)
check("OBSERVE RBAC grants NO deployments/scale", "deployments/scale" not in _obs_rules)
check(
    "OBSERVE RBAC is a namespaced Role (no ClusterRole)",
    "kind: Role" in obs and "kind: ClusterRole" not in obs,
)
check(
    "OBSERVE CronJob sets GOVERNOR_MODE=observe",
    re.search(r'name:\s*GOVERNOR_MODE,\s*value:\s*"observe"', obs) is not None,
)
check(
    "OBSERVE CronJob mounts emptyDir /control (no control-PVC dependency)",
    "emptyDir: {}" in obs and "persistentVolumeClaim" not in obs,
)
check(
    "OBSERVE CronJob has imagePullSecret + Forbid concurrency",
    "imagePullSecrets" in obs and "concurrencyPolicy: Forbid" in obs,
)
check(
    "enforce template opts into GOVERNOR_MODE=enforce (script fail-safe-defaults to observe)",
    re.search(r'name:\s*GOVERNOR_MODE,\s*value:\s*"enforce"', manifest) is not None,
)
check(
    "enforce template stamps ORPHAN_GRACE env (S1 20m never-adopted hold, cold-load-safe)",
    re.search(r'name:\s*ORPHAN_GRACE,\s*value:\s*"\$\{ORPHAN_GRACE\}"', manifest) is not None,
)


# ── A. classifier sub-modes ──────────────────────────────────────────────────
check(
    "--sum-tokens sums vllm:generation_tokens_total",
    submode(
        "--sum-tokens",
        'vllm:generation_tokens_total{m="a"} 100.0\nvllm:generation_tokens_total 40\n',
    )
    == "140",
)
check(
    "--sum-tokens on no-token blob = 0",
    submode("--sum-tokens", "irrelevant 1\n") == "0",
)
check(
    "--pod-crashlooping detects CrashLoopBackOff",
    submode("--pod-crashlooping", '{"x":{"reason": "CrashLoopBackOff"}}') == "1",
)
check(
    "--pod-crashlooping healthy pod = 0",
    submode("--pod-crashlooping", '{"phase":"Running"}') == "0",
)
check(
    "--pod-pending detects Pending",
    submode("--pod-pending", '{"status":{"phase": "Pending"}}') == "1",
)
check(
    "--pod-pending detects ImagePullBackOff",
    submode("--pod-pending", '{"reason": "ImagePullBackOff"}') == "1",
)
check(
    "--pod-pending running pod = 0",
    submode("--pod-pending", '{"phase":"Running"}') == "0",
)
check(
    "--job-verdict active Job ('||1') = active",
    submode("--job-verdict", "||1") == "active",
)
check("--job-verdict complete", submode("--job-verdict", "1|0|0") == "complete")
check("--job-verdict failed", submode("--job-verdict", "0|1|0") == "failed")


# ── B. DRY_RUN fixture reconcile ─────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp())
control = tmp / "control"
fixt = tmp / "fixtures"  # canned kubectl responses + mutation log
control.mkdir()
fixt.mkdir()
NOW = int(time.time())


def iso(delta: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(NOW + delta))


def status(rid: str, *, root: Path | None = None, **kw) -> None:
    """Write a wrapper-shaped status.json fixture.

    Defaults describe a HEALTHY run: fresh progress, nothing in flight, and an idle_utc that (like the live
    wrapper) is refreshed on every poll where work is absent or unmeasurable. `inflight_requests=-1` means
    UNKNOWN — the reading a pre-hang-short-circuit wrapper (no such field at all) also degrades to.
    """
    d = (root or control) / rid
    d.mkdir()
    base = dict(
        run_id=rid,
        cell=f"/repo/recipes/{rid}",
        profile="p",
        namespace="testns",
        recipe=f"cell-{rid}",
        state="running",
        phase="generating",
        reason="",
        heartbeat_utc=iso(0),
        progress_counter=10,
        progress_utc=iso(0),
        progress_note="ok",
        inflight_requests=0,
        queued_requests=0,
        idle_utc=iso(0),
        expected_runtime_seconds=0,
        updated_utc=iso(0),
    )
    base.update(kw)
    (d / "status.json").write_text(json.dumps(base))


def run_job(rid: str, *, root: Path | None = None, **kw) -> None:
    """Canned active bench Job + healthy pod for a control-PVC run fixture."""
    write(
        f"job_cell-{rid}-bench-{rid}.json",
        job_json(f"cell-{rid}-bench-{rid}", active=1, **kw),
    )
    write(f"pod_cell-{rid}-bench-{rid}.json", {"status": {"phase": "Running"}})


def job_json(name: str, active=0, succeeded=0, failed=0, start_delta=0, uid="u-x") -> dict:
    st = {}
    if active:
        st["active"] = active
    if succeeded:
        st["succeeded"] = succeeded
    if failed:
        st["failed"] = failed
    st["startTime"] = iso(start_delta)
    return {
        "metadata": {
            "name": name,
            "uid": uid,
            "labels": {"llmb.nvidia.com/cell": name.split("-bench-")[0]},
        },
        "status": st,
    }


def write(name: str, obj) -> None:
    (fixt / name).write_text(json.dumps(obj))


# --- control-PVC scan scenarios (resp 2 & 3) ---
# 1) healthy — generating, fresh progress+hb, active Job -> NO action.
status("healthy", progress_utc=iso(-10), heartbeat_utc=iso(-10))
write(
    "job_cell-healthy-bench-healthy.json",
    job_json("cell-healthy-bench-healthy", active=1),
)
write("pod_cell-healthy-bench-healthy.json", {"status": {"phase": "Running"}})

# 2) No token progress with active requests beyond the threshold -> HALT stalled.
#    progress_note carries a quote + backslash on purpose: ANY string field of status.json can end up in
#    the halt detail, and an unescaped one would emit invalid governor.json — i.e. an UNREADABLE kill.
status(
    "stalled",
    progress_utc=iso(-3600),
    heartbeat_utc=iso(-10),
    inflight_requests=15,
    queued_requests=0,
    idle_utc=iso(-3600),
    progress_note='ok "quoted" \\ backslash',
)
run_job("stalled")

# 3) heartbeat-lost — hb stale (>HEARTBEAT_DEAD) -> HALT stalled (heartbeat lost).
status("hblost", progress_utc=iso(-10), heartbeat_utc=iso(-3600))
write("job_cell-hblost-bench-hblost.json", job_json("cell-hblost-bench-hblost", active=1))
write("pod_cell-hblost-bench-hblost.json", {"status": {"phase": "Running"}})

# 4) timed-out — fresh progress+hb but Job age >> 2xexpected -> HALT timed-out.
status(
    "timedout",
    progress_utc=iso(-10),
    heartbeat_utc=iso(-10),
    expected_runtime_seconds=60,
)
write(
    "job_cell-timedout-bench-timedout.json",
    job_json("cell-timedout-bench-timedout", active=1, start_delta=-3600),
)
write("pod_cell-timedout-bench-timedout.json", {"status": {"phase": "Running"}})

# 5) server-crashed — pod CrashLoopBackOff while active -> HALT server-crashed.
status("crashed", progress_utc=iso(-10), heartbeat_utc=iso(-10))
write(
    "job_cell-crashed-bench-crashed.json",
    job_json("cell-crashed-bench-crashed", active=1),
)
write(
    "pod_cell-crashed-bench-crashed.json",
    {"status": {"containerStatuses": [{"state": {"waiting": {"reason": "CrashLoopBackOff"}}}]}},
)

# 6) bootstrap — phase=waiting-server, progress STALE, but phase gate -> NO stall kill (healthy pod).
status("bootstrap", phase="waiting-server", progress_utc=iso(-3600), heartbeat_utc=iso(-10))
write(
    "job_cell-bootstrap-bench-bootstrap.json",
    job_json("cell-bootstrap-bench-bootstrap", active=1),
)
write("pod_cell-bootstrap-bench-bootstrap.json", {"status": {"phase": "Running"}})

# 7) completed — state=complete -> scan skips entirely (never queries kubectl).
status("done", state="complete", phase="terminal", reason="completed")

# ── False-positive termination cases. Every one of these has token progress stale WELL past the halt threshold. None
#    may be halted, because the second half of the signature — work outstanding at the server — is absent
#    or unknown. These runs cost hours; killing one is worse than missing a hang. ────────────────────────
# 8) HEALTHY GAP BETWEEN RUNGS: the sweep finished a rung and is writing artifacts / ramping the next one.
#    Tokens paused for an hour, but NOTHING is in flight, so the outstanding-work window keeps resetting.
status(
    "gapnowork",
    progress_utc=iso(-7200),
    heartbeat_utc=iso(-10),
    inflight_requests=0,
    queued_requests=0,
    idle_utc=iso(-5),
)
run_job("gapnowork")

# 9) UNMEASURABLE: /metrics is unreachable, so in-flight is UNKNOWN (-1) and the wrapper keeps resetting
#    idle_utc. Never halt on evidence we do not have.
status(
    "unknownwork",
    progress_utc=iso(-7200),
    heartbeat_utc=iso(-10),
    inflight_requests=-1,
    queued_requests=-1,
    idle_utc=iso(-5),
    progress_note="metrics-unreachable",
)
run_job("unknownwork")

# 10) LEGACY WRAPPER: a run submitted before the hang short-circuit existed has NO in-flight fields at all.
#     Absent must degrade to UNKNOWN -> never halted (it is also never falsely killed).
_legacy = dict(
    run_id="legacy",
    cell="/repo/recipes/legacy",
    profile="p",
    namespace="testns",
    recipe="cell-legacy",
    state="running",
    phase="generating",
    reason="",
    heartbeat_utc=iso(-10),
    progress_counter=10,
    progress_utc=iso(-7200),
    expected_runtime_seconds=0,
    updated_utc=iso(-10),
)
(control / "legacy").mkdir()
(control / "legacy" / "status.json").write_text(json.dumps(_legacy))
run_job("legacy")

# 11) WORK ONLY JUST STARTED: tokens stale for 2h, but the outstanding-work window is 100s old (the server
#     just took requests again). The stall must be SUSTAINED on BOTH halves, not merely instantaneous.
status(
    "freshwork",
    progress_utc=iso(-7200),
    heartbeat_utc=iso(-10),
    inflight_requests=15,
    queued_requests=0,
    idle_utc=iso(-100),
)
run_job("freshwork")

# 12) WARN BAND: both halves past STALL_THRESHOLD but below the 2x halt threshold -> warn, delete NOTHING.
status(
    "warnband",
    progress_utc=iso(-2000),
    heartbeat_utc=iso(-10),
    inflight_requests=15,
    queued_requests=0,
    idle_utc=iso(-2000),
)
run_job("warnband")

# 12b/12c) BELT-AND-BRACES on the in-flight conjunct itself: a status.json whose outstanding-work window is
#     old but whose in-flight reading is 0 / UNKNOWN is self-inconsistent (the live wrapper resets the window
#     on both). The halt must still require a POSITIVE in-flight reading, so neither may be killed.
status(
    "staleidle",
    progress_utc=iso(-7200),
    heartbeat_utc=iso(-10),
    inflight_requests=0,
    queued_requests=0,
    idle_utc=iso(-7200),
)
run_job("staleidle")
status(
    "unknownidle",
    progress_utc=iso(-7200),
    heartbeat_utc=iso(-10),
    inflight_requests=-1,
    queued_requests=-1,
    idle_utc=iso(-7200),
    progress_note="metrics-unreachable",
)
run_job("unknownidle")

# 13) INVALID JSON: every `jq -r` read returns empty, which used to drop the run out of ALL supervision
#     SILENTLY. Absence of supervision must be announced.
(control / "badjson").mkdir()
(control / "badjson" / "status.json").write_text('{"state":"running","reason":"he said "boom""}')
run_job("badjson")

# --- orphan-sweep scenarios (resp 1), served via `get deploy -l sel` + `get jobs -l sel` ---
# The fail-safe predicate reaps ONLY on POSITIVE orphan evidence (owner UID gone AND a run demonstrably
# existed AND no live bench Job for the cell) and never before ORPHAN_GRACE (=MIN_KILL_SECONDS=100s here).
deploy_list = {
    "items": [
        # POSITIVE orphan: ownerRef UID is GONE from the Job list, the server carries its /cell label, there is no
        # live bench Job for that cell, and it is OLD (age >> ORPHAN_GRACE) -> scale->0 + delete.
        {
            "metadata": {
                "name": "cell-orphan-server",
                "creationTimestamp": iso(-99999),
                "labels": {"llmb.nvidia.com/cell": "cell-orphan"},
                "ownerReferences": [{"kind": "Job", "uid": "u-orphan"}],
            }
        },
        {
            "metadata": {
                "name": "cell-term-server",
                "labels": {},
                "ownerReferences": [{"kind": "Job", "uid": "u-term"}],
            }
        },
        {
            "metadata": {
                "name": "cell-live-server",
                "labels": {},
                "ownerReferences": [{"kind": "Job", "uid": "u-live"}],
            }
        },
        # FAIL-SAFE 1: owner UID unresolved (not among Jobs) BUT the server is RECENT (age < ORPHAN_GRACE) — the
        # owner may still be materializing (deploy.sh brings the server up BEFORE the bench Job / adopt_server).
        # Must be HELD, never reaped. This is the destructive-bug regression guard.
        {
            "metadata": {
                "name": "cell-recent-server",
                "creationTimestamp": iso(-5),
                "labels": {"llmb.nvidia.com/cell": "cell-recent"},
                "ownerReferences": [{"kind": "Job", "uid": "u-missing"}],
            }
        },
        # FAIL-SAFE 2: OLD but ownerless AND no /cell label -> ambiguous, NO positive orphan evidence -> HELD.
        # (The legacy code classified this "gone" and scale->0 + DELETED it — a healthy live server.)
        {
            "metadata": {
                "name": "cell-legacy-server",
                "creationTimestamp": iso(-99999),
                "labels": {},
            }
        },
        # NEVER-ADOPTED orphan, AVAILABLE (finished loading), OLD, has /cell, no live Job -> REAP (branch 2).
        {
            "metadata": {
                "name": "cell-nadopt-server",
                "creationTimestamp": iso(-99999),
                "labels": {"llmb.nvidia.com/cell": "cell-nadopt"},
            },
            "status": {"availableReplicas": 1},
        },
        # COLD-LOAD PROTECTION (the live GB300 1M-study case): never-adopted, OLD (past grace), has /cell, NO live
        # Job, but availableReplicas=0 (a giant server still loading weights) -> HELD, never reaped mid-load.
        {
            "metadata": {
                "name": "cell-cold-server",
                "creationTimestamp": iso(-99999),
                "labels": {"llmb.nvidia.com/cell": "cell-cold"},
            },
            "status": {"availableReplicas": 0},
        },
        # MAX-LIFETIME HARD CEILING (S2 belt): created-at label is older than max-lifetime-s, server NOT Available
        # (stuck / perma-cold-load — invisible to the availability-gated branch), no live Job -> REAP via ceiling.
        {
            "metadata": {
                "name": "cell-ceiling-server",
                "creationTimestamp": iso(-10),
                "labels": {
                    "llmb.nvidia.com/cell": "cell-ceiling",
                    "llmb.nvidia.com/created-at": str(NOW - 99999),
                    "llmb.nvidia.com/max-lifetime-s": "100",
                },
            },
            "status": {"availableReplicas": 0},
        },
        # CEILING SAFETY: past the max-lifetime ceiling BUT a live bench Job for the cell exists -> KEEP. The
        # ceiling must NEVER kill a running benchmark (CRITICAL SAFETY); the active-Job keep gate wins.
        {
            "metadata": {
                "name": "cell-ceilactive-server",
                "creationTimestamp": iso(-10),
                "labels": {
                    "llmb.nvidia.com/cell": "cell-ceilactive",
                    "llmb.nvidia.com/created-at": str(NOW - 99999),
                    "llmb.nvidia.com/max-lifetime-s": "100",
                },
            },
            "status": {"availableReplicas": 1},
        },
    ]
}
jobs_list = {
    "items": [
        job_json("cell-term-bench-x", succeeded=1, uid="u-term"),  # terminal owner -> scale->0 only
        job_json("cell-live-bench-x", active=1, uid="u-live"),  # active owner  -> LEAVE ALONE
        job_json("cell-ceilactive-bench-x", active=1, uid="u-ceilactive"),  # live bench Job for the ceiling cell
        # NOTE: u-orphan / u-missing have NO matching Job; no bench Job carries /cell=cell-orphan|cell-recent.
    ]
}
write("deploy_list.json", deploy_list)
write("jobs_list.json", jobs_list)

# --- the fake kubectl ---
fake = tmp / "fake-kubectl"
fake.write_text(r"""#!/usr/bin/env python3
import json, sys, os
FIX = os.environ["FIXTURE_DIR"]
a = sys.argv[1:]
# strip global flags: -n <ns>, --context <c>
out, i = [], 0
while i < len(a):
    if a[i] in ("-n", "--namespace", "--context"):
        i += 2; continue
    out.append(a[i]); i += 1
a = out
def emit(path):
    p = os.path.join(FIX, path)
    sys.stdout.write(open(p).read() if os.path.exists(p) else json.dumps({"items": []}))
    sys.exit(0)
if a[:1] == ["get"]:
    if a[1] == "deploy":
        emit("deploy_list.json")
    if a[1] == "jobs":
        emit("jobs_list.json")
    if a[1] == "job":
        emit(f"job_{a[2]}.json")
    if a[1] == "pods":
        sel = next((x.split("job-name=")[1] for x in a if "job-name=" in x), "")
        emit(f"pod_{sel}.json")
    sys.exit(0)
if a[:1] in (["scale"], ["delete"]):
    with open(os.path.join(FIX, "mutations.log"), "a") as f:
        f.write(" ".join(a) + "\n")
    sys.exit(0)
sys.exit(0)
""")
fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

env = dict(
    os.environ,
    NAMESPACE="testns",
    CONTROL_ROOT=str(control),
    KUBECTL=str(fake),
    FIXTURE_DIR=str(fixt),
    GOVERNOR_MODE="enforce",
    DRY_RUN="1",
    STALL_THRESHOLD="1800",
    HEARTBEAT_DEAD="300",
    STUCK_THRESHOLD="900",
    TIMEOUT_MULT="2",
    MIN_KILL_SECONDS="100",
)
p = subprocess.run(["sh", str(RECONCILE)], env=env, capture_output=True, text=True)
log = p.stdout + p.stderr
check("reconcile DRY_RUN exits 0", p.returncode == 0, log[-300:])

# expected DRY_RUN intents
check("healthy run -> NO action", "job/cell-healthy-bench-healthy" not in log)
check(
    "stalled run -> would delete reason=stalled",
    bool(re.search(r"would delete job/cell-stalled-bench-stalled reason=stalled", log)),
    log,
)

# ── THE HANG SHORT-CIRCUIT ────────────────────────────────────────────────────
# (i) The kill is EXPLAINABLE: the detail must name both halves of the signature and the threshold it
#     crossed, so an operator never has to read code to learn why a 4-hour run was killed.
_halt_line = next(
    (ln for ln in log.splitlines() if "cell-stalled-bench-stalled" in ln and "would delete" in ln),
    "",
)
check(
    "HALT detail names BOTH halves of the signature + the threshold crossed",
    all(t in _halt_line for t in ("no output tokens", "outstanding", "halt threshold")),
    _halt_line,
)
check(
    "HALT detail carries the measured evidence (in-flight count, queue depth, token high-water)",
    all(t in _halt_line for t in ("15 request", "queued=0", "tokens=10")),
    _halt_line,
)

# (ii) FALSE-KILL GUARDS — each of these has token progress stale far past the halt threshold and must
#      still NOT be halted, because work is not outstanding (or is unmeasurable).
for _rid, _why in [
    ("gapnowork", "healthy gap between rungs: nothing in flight"),
    ("unknownwork", "/metrics unreachable: in-flight UNKNOWN, never assumed busy"),
    ("legacy", "older wrapper: no in-flight fields at all -> UNKNOWN"),
    (
        "freshwork",
        "work outstanding only 100s: the stall is not SUSTAINED on both halves",
    ),
    ("staleidle", "in-flight measured 0: a halt needs a POSITIVE in-flight reading"),
    ("unknownidle", "in-flight UNKNOWN (-1): never inferred to be work"),
]:
    check(
        f"NO FALSE KILL ({_why}) -> stale tokens but never halted",
        f"would delete job/cell-{_rid}-bench-{_rid}" not in log,
        log,
    )
    check(
        f"NO FALSE KILL ({_rid}) -> no governor.json written",
        not (control / _rid / "governor.json").exists(),
    )
    check(
        f"no-halt for {_rid} is EXPLAINED in the log, not silent",
        f"no-halt run={_rid}" in log,
        log,
    )

# (iii) WARN band: past STALL_THRESHOLD on both halves but under the 2x halt threshold -> warn only.
check(
    "WARN band -> warns, deletes NOTHING",
    "WARN job/cell-warnband-bench-warnband reason=stall-warning" in log
    and "would delete job/cell-warnband-bench-warnband" not in log,
    log,
)
_wj = control / "warnband" / "governor.json"
check(
    "WARN band -> governor.json action=warn (visible to `llmb-k8s status`, not a halt)",
    _wj.is_file() and json.loads(_wj.read_text()).get("action") == "warn",
    _wj.read_text() if _wj.is_file() else "absent",
)

# (iv) An UNPARSEABLE status.json silently dropped the run from every supervision path. Now it is loud.
check(
    "invalid status.json -> announced as UNSUPERVISED (never a silent skip)",
    "UNSUPERVISED run=badjson" in log,
    log,
)
check(
    "invalid status.json -> still never halted on a guess",
    "job/cell-badjson-bench-badjson" not in log,
    log,
)

# (v) The governor's OWN record must stay valid JSON however hostile the strings it quotes — an unreadable
#     governor.json is an unexplained kill, which is the thing this whole feature exists to prevent.
_sj = control / "stalled" / "governor.json"
try:
    _sg = json.loads(_sj.read_text())
    check(
        "HALT governor.json is valid JSON even when the detail quotes a hostile status.json field",
        _sg.get("action") == "halt" and "quoted" in _sg.get("detail", ""),
        _sj.read_text(),
    )
except Exception as _e:  # noqa: BLE001
    check(
        "HALT governor.json is valid JSON even when the detail quotes a hostile status.json field",
        False,
        f"{_e}: {_sj.read_text() if _sj.is_file() else 'absent'}",
    )
check(
    "heartbeat-lost -> would delete reason=stalled (heartbeat lost)",
    "job/cell-hblost-bench-hblost reason=stalled" in log and "heartbeat lost" in log,
    log,
)
check(
    "timed-out -> would delete reason=timed-out",
    bool(re.search(r"would delete job/cell-timedout-bench-timedout reason=timed-out", log)),
    log,
)
check(
    "server-crashed -> would delete reason=server-crashed",
    bool(re.search(r"would delete job/cell-crashed-bench-crashed reason=server-crashed", log)),
    log,
)
check(
    "bootstrap (phase!=generating, stale progress) -> NO action",
    "job/cell-bootstrap-bench-bootstrap" not in log,
)
check("completed run -> NO action", "cell-done-bench-done" not in log)
# orphan-sweep
check(
    "orphan server (owner gone, has /cell, old) -> would scale->0 + delete",
    "scale->0 + delete deploy/cell-orphan-server" in log,
    log,
)
check(
    "terminal-owner server -> would scale->0 (no delete)",
    "would scale->0 deploy/cell-term-server (owner Job terminal)" in log,
    log,
)
check("active-owner server -> LEFT ALONE", "cell-live-server" not in log)
# fail-safe: an unresolved-owner but RECENT server is NEVER reaped (the destructive-bug regression guard).
check(
    "FAIL-SAFE: unresolved-owner RECENT server -> HELD (never scaled/deleted)",
    "cell-recent-server" not in log,
    log,
)
# fail-safe: an old ownerless server with no /cell label is ambiguous -> HELD (no positive orphan evidence).
check(
    "FAIL-SAFE: old ownerless server, no /cell label -> HELD (not reaped on absence)",
    "cell-legacy-server" not in log,
    log,
)
# never-adopted branch: AVAILABLE + old + /cell + no live Job -> REAP (the 11h-idle-orphan gap).
check(
    "never-adopted AVAILABLE server (old, /cell, no live Job) -> would scale->0 + delete",
    "scale->0 + delete deploy/cell-nadopt-server" in log,
    log,
)
# COLD-LOAD PROTECTION regression guard (the live GB300 1M-study case): availableReplicas=0 -> HELD.
check(
    "COLD-LOAD: never-adopted past grace but availableReplicas=0 -> HELD (never reaped mid cold-load)",
    "cell-cold-server" not in log,
    log,
)
# max-lifetime hard ceiling: past ceiling + not Available + no live Job -> REAP via ceiling (belt).
check(
    "max-lifetime ceiling exceeded (stuck, not Available) -> would scale->0 + delete via ceiling",
    "scale->0 + delete deploy/cell-ceiling-server" in log and "max-lifetime ceiling exceeded" in log,
    log,
)
# ceiling safety: past ceiling BUT a live bench Job for the cell -> KEEP (never kill a running benchmark).
check(
    "CEILING SAFETY: past ceiling but live bench Job for cell -> HELD (never kills a running run)",
    "cell-ceilactive-server" not in log,
    log,
)

# governor.json written for each HALT (governor-owned file), NOT for healthy/bootstrap.
for rid, reason in [
    ("stalled", "stalled"),
    ("timedout", "timed-out"),
    ("crashed", "server-crashed"),
    ("hblost", "stalled"),
]:
    gj = control / rid / "governor.json"
    ok = gj.is_file() and json.loads(gj.read_text()).get("reason") == reason
    check(
        f"governor.json written for {rid} with reason={reason}",
        ok,
        gj.read_text() if gj.is_file() else "absent",
    )
check(
    "no governor.json for healthy run",
    not (control / "healthy" / "governor.json").exists(),
)
check(
    "no governor.json for bootstrap run",
    not (control / "bootstrap" / "governor.json").exists(),
)

# CRITICAL: DRY_RUN issued ZERO mutating kubectl calls.
mut = fixt / "mutations.log"
check(
    "DRY_RUN issued ZERO mutating kubectl (scale/delete) calls",
    not mut.exists(),
    mut.read_text() if mut.exists() else "",
)


# ── C. BusyBox-echo embedded-wrapper JSON-read regression (Phase-2 CRITICAL bug) ─────────────
# The deployed image alpine/k8s:1.34.1 runs BusyBox /bin/sh, whose `echo` INTERPRETS backslash
# escapes. The real bench Job JSON now EMBEDS the injected wrapper script, whose source carries
# literal \n and \"/control/${RUN_ID}\" escapes. Read via the old `echo "$var" | jq` idiom on
# BusyBox those escapes are expanded (\n -> a real newline), corrupting the JSON so jq throws
# "Invalid numeric literal" / control-character errors. Consequences on the in-cluster shell:
#   * sweep_orphans could not match the owner Job -> misclassified an ACTIVE-owner server as
#     `gone` -> scale->0 + DELETE of a LIVE serving Deployment;
#   * scan_runs' active-guard `jq -e` errored -> `|| continue` skipped the run -> the stall /
#     timeout watchdog silently NEVER fired.
# FIX: EVERY Job/pod-JSON read uses `printf '%s'` (no escape interpretation). The offline
# fixtures above passed only because they used clean JSON with no embedded wrapper.

# A REAL bench Job JSON that embeds the wrapper. json.dumps serializes the real newlines / quotes
# in the wrapper source to \n / \" escapes in the JSON *text* — exactly what kubectl emits.
wrapper = (
    "set -eu\n"
    'RUN_ID="$1"\n'
    'CONTROL="/control/${RUN_ID}"\n'
    'printf \'{"state":"running"}\' > "/control/${RUN_ID}/status.json"\n'
)


def embed_wrapper(job: dict) -> dict:
    job = dict(job)
    job["spec"] = {
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "bench",
                        "command": ["/bin/sh", "-c", wrapper, "run-embed"],
                    }
                ]
            }
        }
    }
    return job


_bench_text = json.dumps(embed_wrapper(job_json("cell-embed-bench-embed", active=1, uid="u-embed")))
check(
    'fixture Job JSON carries the embedded-wrapper \\n and \\"/control escapes',
    "\\n" in _bench_text and '\\"/control/' in _bench_text,
    _bench_text[:120],
)


# C.0 — host-independent proof of the exact failure mode. `printf '%b'` reproduces BusyBox-echo
# escape interpretation (the old `echo "$var"`); `printf '%s'` is the applied fix. Feed the SAME
# embedded-wrapper Job JSON through each and read `.status.active` the way the reconcile does.
def _read_active(feeder: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "sh",
            "-c",
            f"""{feeder} "$1" | jq -e '(.status.active//0)>=1' >/dev/null""",
            "_",
            _bench_text,
        ],
        capture_output=True,
        text=True,
    )


_busybox = _read_active("printf '%b'")  # emulates BusyBox `echo "$var"` (the BUG)
_fixed = _read_active("printf '%s'")  # the applied fix
check(
    "BusyBox-echo idiom CORRUPTS embedded-wrapper Job JSON (jq errors)",
    _busybox.returncode != 0,
    _busybox.stderr.strip()[:160],
)
check(
    "printf '%s' PRESERVES embedded-wrapper Job JSON (jq reads .status.active)",
    _fixed.returncode == 0,
    _fixed.stderr.strip()[:160],
)


# C.1 — end-to-end: run the REAL reconcile against a fixture whose Job JSONs embed the wrapper.
# NOTE: on hosts whose /bin/sh interprets echo escapes (dash / BusyBox), this whole section FAILS
# against the legacy `echo "$var" | jq` code and PASSES after the `printf '%s'` fix.
control2 = tmp / "control2"
fixt2 = tmp / "fixtures2"
control2.mkdir()
fixt2.mkdir()

# active-owner server whose owner Job embeds the wrapper -> sweep must LEAVE IT ALONE.
(fixt2 / "deploy_list.json").write_text(
    json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "cell-embed-server",
                        "labels": {},
                        "ownerReferences": [{"kind": "Job", "uid": "u-embed"}],
                    }
                },
            ]
        }
    )
)
(fixt2 / "jobs_list.json").write_text(
    json.dumps(
        {
            "items": [
                embed_wrapper(job_json("cell-embed-bench-embed", active=1, uid="u-embed")),
            ]
        }
    )
)
# active + STALLED run whose Job embeds the wrapper -> scan must NOT skip -> HALT stalled.
_d2 = control2 / "embed"
_d2.mkdir()
(_d2 / "status.json").write_text(
    json.dumps(
        dict(
            run_id="embed",
            cell="/repo/recipes/embed",
            profile="p",
            namespace="testns",
            recipe="cell-embed",
            state="running",
            phase="generating",
            reason="",
            heartbeat_utc=iso(-10),
            progress_counter=10,
            progress_utc=iso(-3600),
            progress_note="ok",
            inflight_requests=15,
            queued_requests=0,
            idle_utc=iso(-3600),
            expected_runtime_seconds=0,
            updated_utc=iso(-10),
        )
    )
)
(fixt2 / "job_cell-embed-bench-embed.json").write_text(
    json.dumps(embed_wrapper(job_json("cell-embed-bench-embed", active=1, uid="u-embed")))
)
(fixt2 / "pod_cell-embed-bench-embed.json").write_text(json.dumps({"status": {"phase": "Running"}}))

env2 = dict(env, CONTROL_ROOT=str(control2), FIXTURE_DIR=str(fixt2))
p2 = subprocess.run(["sh", str(RECONCILE)], env=env2, capture_output=True, text=True)
log2 = p2.stdout + p2.stderr
check("embedded-wrapper reconcile DRY_RUN exits 0", p2.returncode == 0, log2[-300:])
# (a) ACTIVE-owner server NOT misclassified `gone`: never in the would-scale/would-delete list.
check(
    "(a) active-owner server w/ embedded-wrapper Job -> LEFT ALONE (not scale->0 + delete)",
    "cell-embed-server" not in log2,
    log2,
)
# (b) scan_runs did NOT skip the active run: the stall-guard evaluated and would fire.
check(
    "(b) active run w/ embedded-wrapper Job -> stall-guard evaluates (would delete reason=stalled)",
    bool(re.search(r"would delete job/cell-embed-bench-embed reason=stalled", log2)),
    log2,
)
_mut2 = fixt2 / "mutations.log"
check(
    "embedded-wrapper reconcile issued ZERO mutating kubectl calls",
    not _mut2.exists(),
    _mut2.read_text() if _mut2.exists() else "",
)


# ── D. OBSERVE / REPORT-ONLY reconcile (the deployed safety-net mode) ─────────────────────────────
# GOVERNOR_MODE=observe must: (a) identify OUR resources, (b) flag a healthy active run as NOT a reap
# candidate, (c) report FOREIGN (unmanaged) workloads and NEVER act on them, (d) compute WOULD-REAP /
# WOULD-HALT candidates, (e) publish the report ConfigMap using only get/create, and (f) mutate NOTHING
# (zero scale/delete) and touch NO per-run governor.json.
control3 = tmp / "control3"
fixt3 = tmp / "fixtures3"
control3.mkdir()
fixt3.mkdir()

# One stalled run (control PVC) -> WOULD-HALT stalled.
_d3 = control3 / "obsstall"
_d3.mkdir()
(_d3 / "status.json").write_text(
    json.dumps(
        dict(
            run_id="obsstall",
            cell="/repo/recipes/obsstall",
            profile="p",
            namespace="testns",
            recipe="cell-obsstall",
            state="running",
            phase="generating",
            reason="",
            heartbeat_utc=iso(-10),
            progress_counter=10,
            progress_utc=iso(-3600),
            progress_note="ok",
            inflight_requests=15,
            queued_requests=0,
            idle_utc=iso(-3600),
            expected_runtime_seconds=0,
            updated_utc=iso(-10),
        )
    )
)
(fixt3 / "job_cell-obsstall-bench-obsstall.json").write_text(
    json.dumps(job_json("cell-obsstall-bench-obsstall", active=1, uid="u-obsstall"))
)
(fixt3 / "pod_cell-obsstall-bench-obsstall.json").write_text(json.dumps({"status": {"phase": "Running"}}))

# OUR (labeled) deployments (-l sel): one terminal-owner (WOULD-REAP), one active-owner (HEALTHY-ACTIVE).
_ours_deploy = {
    "items": [
        {
            "metadata": {
                "name": "cell-obsterm-server",
                "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"},
                "ownerReferences": [{"kind": "Job", "uid": "u-obsterm"}],
            }
        },
        {
            "metadata": {
                "name": "cell-obslive-server",
                "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"},
                "ownerReferences": [{"kind": "Job", "uid": "u-obslive"}],
            }
        },
    ]
}
# OUR (labeled) jobs (-l sel): terminal + active.
_ours_jobs = {
    "items": [
        dict(
            job_json("cell-obsterm-bench-x", succeeded=1, uid="u-obsterm"),
            **{
                "metadata": {
                    "name": "cell-obsterm-bench-x",
                    "uid": "u-obsterm",
                    "labels": {
                        "app.kubernetes.io/managed-by": "llmb-recipe",
                        "llmb.nvidia.com/cell": "cell-obsterm",
                    },
                }
            },
        ),
        dict(
            job_json("cell-obslive-bench-x", active=1, uid="u-obslive"),
            **{
                "metadata": {
                    "name": "cell-obslive-bench-x",
                    "uid": "u-obslive",
                    "labels": {
                        "app.kubernetes.io/managed-by": "llmb-recipe",
                        "llmb.nvidia.com/cell": "cell-obslive",
                    },
                }
            },
        ),
    ]
}
# ALL deployments/jobs (no -l): OURS + a FOREIGN dynamo-platform workload the governor must NEVER act on.
_all_deploy = {
    "items": _ours_deploy["items"]
    + [
        {
            "metadata": {
                "name": "dynamo-platform-frontend",
                "labels": {"app": "dynamo-platform"},
            }
        },
    ]
}
_all_jobs = {
    "items": _ours_jobs["items"]
    + [
        {
            "metadata": {
                "name": "lmsysdyn-eval-runner",
                "labels": {"app.kubernetes.io/name": "lmsysdyn"},
            },
            "status": {"active": 1},
        },
    ]
}
(fixt3 / "ours_deploy.json").write_text(json.dumps(_ours_deploy))
(fixt3 / "ours_jobs.json").write_text(json.dumps(_ours_jobs))
(fixt3 / "all_deploy.json").write_text(json.dumps(_all_deploy))
(fixt3 / "all_jobs.json").write_text(json.dumps(_all_jobs))

# fake kubectl for observe: distinguishes label-filtered (-l) from unfiltered reads; logs mutations + all calls.
fake3 = tmp / "fake-kubectl-observe"
fake3.write_text(r"""#!/usr/bin/env python3
import json, sys, os
FIX = os.environ["FIXTURE_DIR"]
raw = sys.argv[1:]
has_l = any(x == "-l" or x.startswith("-l=") or x.startswith("--selector") for x in raw)
a, i = [], 0
while i < len(raw):
    if raw[i] in ("-n", "--namespace", "--context", "-l", "--selector", "-o"):
        i += 2; continue
    if raw[i].startswith(("-l", "--selector=", "-o")) or raw[i] in ("json", "yaml"):
        i += 1; continue
    a.append(raw[i]); i += 1
with open(os.path.join(FIX, "calls.log"), "a") as f:
    f.write(" ".join(raw) + "\n")
def emit(path):
    p = os.path.join(FIX, path)
    sys.stdout.write(open(p).read() if os.path.exists(p) else json.dumps({"items": []}))
    sys.exit(0)
if a[:1] == ["get"]:
    if a[1] == "deploy":
        emit("ours_deploy.json" if has_l else "all_deploy.json")
    if a[1] == "jobs":
        emit("ours_jobs.json" if has_l else "all_jobs.json")
    if a[1] == "job":
        emit(f"job_{a[2]}.json")
    if a[1] == "pods":
        sel = next((x.split("job-name=")[1] for x in raw if "job-name=" in x), "")
        emit(f"pod_{sel}.json")
    if a[1] in ("configmap", "cm"):
        sys.exit(1)   # not found -> reconcile takes the create path
    sys.exit(0)
if a[:1] in (["scale"], ["delete"]):
    with open(os.path.join(FIX, "mutations.log"), "a") as f:
        f.write(" ".join(a) + "\n")
    sys.exit(0)
if a[:1] in (["create"], ["replace"]):
    sys.stdin.read()
    sys.exit(0)
sys.exit(0)
""")
fake3.chmod(fake3.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

env3 = dict(
    os.environ,
    NAMESPACE="testns",
    CONTROL_ROOT=str(control3),
    KUBECTL=str(fake3),
    FIXTURE_DIR=str(fixt3),
    GOVERNOR_MODE="observe",
    STALL_THRESHOLD="1800",
    HEARTBEAT_DEAD="300",
    STUCK_THRESHOLD="900",
    TIMEOUT_MULT="2",
    MIN_KILL_SECONDS="100",
)
p3 = subprocess.run(["sh", str(RECONCILE)], env=env3, capture_output=True, text=True)
log3 = p3.stdout + p3.stderr
check("observe reconcile exits 0", p3.returncode == 0, log3[-300:])
check(
    "observe: (a) identifies OUR active job as HEALTHY-ACTIVE (not a candidate)",
    "HEALTHY-ACTIVE job/cell-obslive-bench-x" in log3,
    log3,
)
check(
    "observe: (a) our terminal job reported as OURS",
    "OURS job/cell-obsterm-bench-x (complete)" in log3,
    log3,
)
check(
    "observe: (b) active-owner server -> HEALTHY-ACTIVE (not a reap candidate)",
    "HEALTHY-ACTIVE deploy/cell-obslive-server" in log3,
    log3,
)
check(
    "observe: (c) FOREIGN deployment reported + never acted on",
    "FOREIGN (unmanaged) deploy/dynamo-platform-frontend" in log3,
    log3,
)
check(
    "observe: (c) FOREIGN job reported + never acted on",
    "FOREIGN (unmanaged) jobs/lmsysdyn-eval-runner" in log3,
    log3,
)
check(
    "observe: (d) terminal-owner server -> WOULD-REAP (report-only)",
    "WOULD-REAP completion-cleanup deploy/cell-obsterm-server" in log3,
    log3,
)
check(
    "observe: (d) stalled run -> WOULD-HALT stalled (report-only)",
    "WOULD-HALT job/cell-obsstall-bench-obsstall reason=stalled" in log3,
    log3,
)
check(
    "observe: (e) publishes report ConfigMap via get->create (no patch/apply)",
    "create configmap governor-observe-report" in (fixt3 / "calls.log").read_text(),
    log3,
)
_mut3 = fixt3 / "mutations.log"
check(
    "observe: (f) ZERO mutating kubectl (scale/delete) calls",
    not _mut3.exists(),
    _mut3.read_text() if _mut3.exists() else "",
)
check(
    "observe: (f) NO per-run governor.json written (report-only, touches no run dir)",
    not (control3 / "obsstall" / "governor.json").exists(),
)
# The fake logs no `patch` because reconcile never issues one in observe mode.
check("observe: issued NO patch verb", "patch" not in (fixt3 / "calls.log").read_text())

print()
if fails:
    print(f"selftest_governor: {len(fails)} FAILED: {fails}")
    sys.exit(1)
total = sum(1 for line in Path(__file__).read_text().splitlines() if line.strip().startswith("check("))
print(f"selftest_governor: all {total} checks PASSED")
sys.exit(0)

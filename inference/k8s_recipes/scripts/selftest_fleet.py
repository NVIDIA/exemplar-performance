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

"""selftest_fleet.py — OFFLINE regression tests for the multi-cluster fleet view (fleet.sh + fleet_render.py).

No cluster needed: a fake `kubectl` shim on PATH replays canned `get pods/deploy/jobs/nodes/pods -A -o json`
fixtures per context, and one context deliberately fails auth. Runs the REAL fleet.sh end-to-end (so the
bash discovery/dedupe/auth-robust plumbing is exercised, not just the pure renderer) and asserts:

  • GPU + CPU sums — per-workload, per-cluster OURS, capacity triple (TOTAL/OCCUPIED/free); CPU shown only
    when >0 (GPU server shows GPU, CPU-only bench shows cores, neither shows nothing).
  • JOB-primary layout — each run (Job) is a top-level row with its server(s) nested (└svc) beneath; a
    cluster with no jobs says "no jobs running".
  • Status classification — RUNNING / LOADING / STARTING / UNSCHEDULED / STUCK / ORPHAN / PARKED /
    COMPLETE, incl. the
    ORPHAN-vs-PARKED accuracy fix: a parked run (control-plane up, workers 0/0) and a fresh grace-window
    server are NOT ORPHAN, while a genuinely-gone old server still is.
  • Elapsed vs expected — `/ ~4.8h exp (41%)`, ⚠ over-median, ⚠ near-deadline (deadline ceiling), and the
    no-expectation fallback (elapsed only).
  • Idle collapse (default) + --idle/--all expand; --watch double-buffer holds a full frame before repaint.
  • Auth failure on ONE cluster does NOT abort others; every CONFIGURED cluster always shown (dedupe);
    CONNECT_CMD hint (profile value, else tsh fallback); bash-3.2 empty-array safety.
  • Deterministic ordering; color stripped when piped / emitted with --color; graceful RBAC degradation.

`make test` runs this. Exit 0 = all pass.
"""

from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
FLEET_SH = SCRIPTS / "fleet.sh"
NOW = "2026-07-19T12:00:00Z"  # fixed clock → deterministic ages

fails: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── fixture builders ─────────────────────────────────────────────────────────────────────────────


def pod(
    name,
    *,
    app=None,
    job=None,
    gpus=0,
    cpu=0,
    phase="Running",
    start="2026-07-19T10:02:00Z",
    managed=True,
    ns="ours",
    waiting=None,
    unsched=False,
    restarts=0,
    node="gpu-node-1",
    run_id=None,
    sweep=None,
    terminated=None,
):
    labels = {}
    if managed:
        labels["app.kubernetes.io/managed-by"] = "llmb-recipe"
    if app:
        labels["app"] = app
    if job:
        labels["job-name"] = job
    if run_id:
        labels["llmb.nvidia.com/run-id"] = run_id
    if sweep:
        labels["llmb.nvidia.com/sweep-id"] = sweep
    cs = {"restartCount": restarts, "state": {}, "lastState": {}}
    if waiting:
        cs["state"] = {"waiting": {"reason": waiting}}
    if terminated:  # lastState.terminated → the ✗ FAILED cause (OOMKilled/exit)
        cs["lastState"] = {"terminated": terminated}
    status = {
        "phase": phase,
        "startTime": start,
        "containerStatuses": [cs],
        "conditions": [],
    }
    if unsched:
        status["conditions"] = [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable"}]
    reqs = {}
    if gpus:
        reqs["nvidia.com/gpu"] = str(gpus)
    if cpu:
        reqs["cpu"] = str(cpu)
    return {
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": labels,
            "creationTimestamp": start,
        },
        "spec": {
            "nodeName": node,
            "containers": [{"name": "c", "image": "img:tag1", "resources": {"requests": reqs}}],
        },
        "status": status,
    }


def deploy(
    name,
    *,
    app,
    desired=1,
    ready=1,
    run_id=None,
    component=None,
    cell=None,
    created="2026-07-19T10:00:00Z",
):
    labels = {"app.kubernetes.io/managed-by": "llmb-recipe"}
    if run_id:
        labels["llmb.nvidia.com/run-id"] = run_id
    if component:
        labels["app.kubernetes.io/component"] = component
    if cell:
        labels["llmb.nvidia.com/cell"] = cell
    return {
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": "ours",
            "labels": labels,
            "creationTimestamp": created,
        },
        "spec": {"replicas": desired, "selector": {"matchLabels": {"app": app}}},
        "status": {"replicas": desired, "readyReplicas": ready},
    }


def job(
    name,
    *,
    recipe,
    active=0,
    succeeded=0,
    failed=0,
    run_id=None,
    sweep=None,
    cell=None,
    expected_env=None,
    deadline=None,
    start="2026-07-19T10:01:00Z",
    concurrencies=None,
    live_rungs=None,
    component=None,
    finished=None,
    fail_reason=None,
    fail_message=None,
):
    labels = {
        "app.kubernetes.io/managed-by": "llmb-recipe",
        "llmb.nvidia.com/recipe": recipe,
    }
    if cell:
        labels["llmb.nvidia.com/cell"] = cell
    if run_id:
        labels["llmb.nvidia.com/run-id"] = run_id
    if sweep:
        labels["llmb.nvidia.com/sweep-id"] = sweep
    if component:  # governor cron etc. → is_infra_job → not a benchmark run
        labels["app.kubernetes.io/component"] = component
    # EXPECTED_RUNTIME_SECONDS is stamped by submit.sh as a bench-container env (source the renderer reads).
    # CONCURRENCIES (aiperf fixed) / LIVE_RUNGS (sweep subset) are the SWEEP-column rung sources that
    # exist for BOTH submit and run.sh launch paths.
    env = []
    if expected_env is not None:
        env.append({"name": "EXPECTED_RUNTIME_SECONDS", "value": str(expected_env)})
    if concurrencies is not None:
        env.append({"name": "CONCURRENCIES", "value": str(concurrencies)})
    if live_rungs is not None:
        env.append({"name": "LIVE_RUNGS", "value": str(live_rungs)})
    spec = {"template": {"spec": {"containers": [{"name": "bench", "env": env}]}}}
    if deadline is not None:
        spec["activeDeadlineSeconds"] = deadline
    status = {
        "active": active,
        "succeeded": succeeded,
        "failed": failed,
        "startTime": start,
    }
    if finished is not None:  # terminal-run end time (drives 'ended Nm ago' history)
        if succeeded:
            status["completionTime"] = finished
        else:
            cond = {"type": "Failed", "status": "True", "lastTransitionTime": finished}
            if fail_reason:  # Job-level Failed cause (DeadlineExceeded/BackoffLimit…)
                cond["reason"] = fail_reason
            if fail_message:
                cond["message"] = fail_message
            status["conditions"] = [cond]
    return {
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": "ours",
            "labels": labels,
            "creationTimestamp": start,
        },
        "spec": spec,
        "status": status,
    }


def node(name, gpus):
    return {
        "metadata": {"name": name},
        "status": {"allocatable": {"nvidia.com/gpu": str(gpus)}},
    }


def items(objs):
    return {"apiVersion": "v1", "kind": "List", "items": objs}


def _glm5_pin() -> str:
    """The revision glm5-fp8's recipes actually PIN, read the same way fleet does. Used by the fixtures so a
    ✓ in the e2e means "the stamp matches the pin", not "the stamp matches a string in this test file".
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    return fr.load_model_pins().get("glm5-fp8", "")


def write_fixtures(root: Path):
    """Connected contexts exercising the JOB-primary layout (jobs + nested servers), CPU-only jobs,
    elapsed-vs-expected (median/deadline/none) + idle collapse; one RBAC-degraded and two auth-failing.
    Fixed clock 12:00:00Z; job start 10:01 → elapsed 1h59m (7140s); pod start 10:02 → 1h58m.
    """
    fx = root / "fixtures"
    fx.mkdir()

    # ── ctxA 'alpha': a RUNNING run (bench Job, CPU-only) with its server nested; +3 idle scale-to-0 ────
    a_pods = [
        pod("modelx-server-1", app="modelx-server", gpus=8, run_id="r1c1"),  # server: GPU
        pod("modelx-bench-r1c1-1", job="modelx-bench-r1c1", gpus=0, cpu=4, run_id="r1c1"),  # bench: CPU-only
    ]
    (fx / "ctxA.pods.json").write_text(json.dumps(items(a_pods)))
    (fx / "ctxA.deploys.json").write_text(
        json.dumps(
            items(
                [
                    deploy("modelx-server", app="modelx-server", ready=1, run_id="r1c1"),
                    deploy("idle-1-server", app="idle-1-server", desired=0, ready=0),  # 0/0 → collapse
                    deploy("idle-2-server", app="idle-2-server", desired=0, ready=0),
                    deploy("idle-3-server", app="idle-3-server", desired=0, ready=0),
                ]
            )
        )
    )
    # expected_env 17280s = 4.8h → elapsed 7140s = 41% (well under median → no flag)
    # Two ✓ done history rows exercising the RESULT-triage suffix:
    #  • metriccell md1 — its (cell, run_id) IS in the fake runs.jsonl (see FLEET_RECIPES_ROOT) → shows metric.
    #  • nocache nc1    — no runs.jsonl for the cell → degrades to the `collect <run-id>` pointer (no fabrication).
    (fx / "ctxA.jobs.json").write_text(
        json.dumps(
            items(
                [
                    job(
                        "modelx-bench-r1c1",
                        recipe="modelx",
                        cell="modelx",
                        active=1,
                        run_id="r1c1",
                        expected_env=17280,
                    ),
                    job(
                        "metriccell-bench-md1",
                        recipe="metriccell",
                        cell="metriccell",
                        run_id="md1",
                        succeeded=1,
                        start="2026-07-19T11:00:00Z",
                        finished="2026-07-19T11:50:00Z",
                    ),  # ✓ done 10m ago, metric cached
                    job(
                        "nocache-bench-nc1",
                        recipe="nocache",
                        cell="nocache",
                        run_id="nc1",
                        succeeded=1,
                        start="2026-07-19T11:10:00Z",
                        finished="2026-07-19T11:52:00Z",
                    ),  # ✓ done 8m ago, metric NOT cached
                ]
            )
        )
    )
    (fx / "ctxA.nodes.json").write_text(json.dumps(items([node("gpu-node-1", 8), node("gpu-node-2", 8)])))  # TOTAL 16
    # MODEL-CACHE PVCs for the LEDGER e2e: one claim carrying a genuine download-Job stamp AT THE PINNED
    # revision (the only shape that may render ✓), one claim nothing routes to, and the control volume. The
    # revision is read from the REAL catalog pin so the fixture cannot drift into a false ✓ if a recipe
    # re-pins — the ledger's whole contract is "matches the pin", not "matches a literal we typed here".
    _pin12 = (_glm5_pin() or "0" * 40)[:12]

    def _pvcfx(name, size, labels=None, ns="ours"):
        return {
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": name, "namespace": ns, "labels": labels or {}},
            "spec": {"resources": {"requests": {"storage": size}}},
            "status": {"phase": "Bound", "capacity": {"storage": size}},
        }

    (fx / "ctxA.pvcs.json").write_text(
        json.dumps(
            items(
                [
                    _pvcfx(
                        "glm5-fp8-model-cache",
                        "1200Gi",
                        {
                            "llmb.nvidia.com/download-complete": "true",
                            "llmb.nvidia.com/model-name": "glm5-fp8",
                            "llmb.nvidia.com/model-revision": _pin12,
                        },
                    ),
                    _pvcfx("stray-cache", "100Gi"),
                    _pvcfx("llmb-control", "5Gi"),
                ]
            )
        )
    )
    a_all = a_pods + [pod("other-tenant-x", ns="teamB", gpus=4, managed=False, node="gpu-node-2")]  # +4 foreign
    (fx / "ctxA.all.json").write_text(json.dumps(items(a_all)))  # OCCUPIED = 8 + 4 = 12 → free 4

    # ── ctxB 'bravo': many runs — over-median, near-deadline, no-expectation, sweep, COMPLETE, UNSCHEDULED,
    #    a LOADING server nested under its active job, plus standalone ORPHAN + STUCK servers. ───────────
    # GPU pods pinned to distinct nodes so the per-NODE capacity math is coherent: b-node-1/2 full (8/8),
    # b-node-3 half (4/8) → 0 free WHOLE nodes though 4 GPUs are 'free' (the fragmentation the header must
    # expose: 20/24 gpu used, 4 free, but biggest free node only 4g).
    b_pods = [
        pod("loady-server-1", app="loady-server", gpus=4, node="b-node-3"),  # 0/1 → LOADING
        pod("orphan-server-1", app="orphan-server", gpus=8, node="b-node-2"),  # 1/1, no job → ORPHAN
        pod(
            "crash-server-1",
            app="crash-server",
            gpus=8,
            node="b-node-1",
            waiting="CrashLoopBackOff",
            restarts=7,
        ),  # STUCK
        pod("loady-bench-r5c1-1", job="loady-bench-r5c1", cpu=2),
        pod("over-bench-r6c1-1", job="over-bench-r6c1", cpu=2),  # CPU-only, over-median
        pod("dead-bench-r7c1-1", job="dead-bench-r7c1", cpu=1),  # CPU-only, near-deadline
        pod("plain-bench-r8c1-1", job="plain-bench-r8c1", cpu=1),  # CPU-only, no expectation
        pod(
            "sweep-bench-r3c1-1",
            job="sweep-bench-r3c1",
            cpu=1,
            sweep="s-2026",
            run_id="r3c1",
        ),
        pod("coord-bench-r2c1-1", job="coord-bench-r2c1", phase="Pending", unsched=True),  # → UNSCHEDULED
    ]
    (fx / "ctxB.pods.json").write_text(json.dumps(items(b_pods)))
    (fx / "ctxB.deploys.json").write_text(
        json.dumps(
            items(
                [
                    deploy("loady-server", app="loady-server", ready=0),
                    deploy("orphan-server", app="orphan-server", ready=1),
                    deploy("crash-server", app="crash-server", ready=0),
                ]
            )
        )
    )
    (fx / "ctxB.jobs.json").write_text(
        json.dumps(
            items(
                [
                    job("loady-bench-r5c1", recipe="loady", active=1),  # active → loady-server owned (LOADING)
                    job(
                        "over-bench-r6c1", recipe="over", active=1, expected_env=3600
                    ),  # 7140/3600 = 198% → ⚠ over-median
                    job("dead-bench-r7c1", recipe="dead", active=1, deadline=7500),  # 7140/7500 = 95% → ⚠ near-deadline
                    job("plain-bench-r8c1", recipe="plain", active=1),  # no env/deadline → elapsed only
                    # SWEEP column = concurrency rungs. CONCURRENCIES env exists on BOTH submit- and run.sh-launched
                    # bench jobs, so the column is populated regardless of launch path (not the submit-only sweep-id label).
                    job(
                        "sweep-bench-r3c1",
                        recipe="sweep",
                        active=1,
                        sweep="s-2026",
                        run_id="r3c1",
                        concurrencies="16 32 64",
                    ),
                    job(
                        "done-bench-r0c0",
                        recipe="done",
                        succeeded=1,
                        finished="2026-07-18T20:00:00Z",
                    ),  # COMPLETE 16h ago → beyond 12h window → hidden (--history hint)
                    job("coord-bench-r2c1", recipe="coord", active=1),  # active, pod never scheduled
                ]
            )
        )
    )
    (fx / "ctxB.nodes.json").write_text(
        json.dumps(items([node("b-node-1", 8), node("b-node-2", 8), node("b-node-3", 8)]))
    )  # 24
    (fx / "ctxB.all.json").write_text(json.dumps(items(b_pods)))  # OCCUPIED = 4+8+8 = 20 → free 4

    # ── ctxRBAC 'charlie': cluster-scoped FORBIDDEN → n/a; NO jobs (→ "no jobs"). Also the ORPHAN-vs-PARKED
    #    accuracy cases: a genuine old orphan, a PARKED run (control-plane up + workers 0/0), and a fresh
    #    grace-window server (recent, no job → RUNNING not ORPHAN). ──────────────────────────────────────
    recent = "2026-07-19T11:56:00Z"  # 4 min before the fixed 12:00 clock → inside the 10-min grace
    c_pods = [
        pod("small-server-1", app="small-server", gpus=2),  # old, alone → ORPHAN
        pod("park-frontend-1", app="park-frontend", gpus=0, cpu=2, start=recent),  # control-plane up
        pod("fresh-server-1", app="fresh-server", gpus=4, start=recent),  # recent → RUNNING (grace)
        # park-decode is scaled to 0/0 → no pod
        # a FAILED run's leftover pod, OOMKilled — the ✗ history row must name the cause from lastState.
        # phase=Failed + gpus=0 → contributes nothing to the OURS/CPU sums (only Running pods count).
        pod(
            "oomcell-bench-ro1-x",
            job="oomcell-bench-ro1",
            gpus=0,
            phase="Failed",
            terminated={"reason": "OOMKilled", "exitCode": 137},
            run_id="ro1",
        ),
    ]
    (fx / "ctxRBAC.pods.json").write_text(json.dumps(items(c_pods)))
    (fx / "ctxRBAC.deploys.json").write_text(
        json.dumps(
            items(
                [
                    deploy("small-server", app="small-server", ready=1),  # replicas>0, old, no run → ORPHAN
                    deploy(
                        "park-frontend",
                        app="park-frontend",
                        desired=1,
                        ready=1,
                        component="frontend",
                        cell="parkrun",
                        created=recent,
                    ),  # control-plane 1/1
                    deploy(
                        "park-decode",
                        app="park-decode",
                        desired=0,
                        ready=0,
                        cell="parkrun",
                        created=recent,
                    ),  # workers 0/0 → PARKED
                    deploy(
                        "fresh-server",
                        app="fresh-server",
                        desired=1,
                        ready=1,
                        created=recent,
                    ),  # fresh → RUNNING (grace)
                ]
            )
        )
    )
    # Lifecycle/history on charlie: a RECENTLY-FAILED run (must surface as ✗ FAILED, never hide as 'idle')
    # and a governor housekeeping cron (COMPLETE but is_infra_job → must NOT count as a benchmark 'done').
    (fx / "ctxRBAC.jobs.json").write_text(
        json.dumps(
            items(
                [
                    # ✗ FAILED via a Job-level Failed condition (no surviving pod): cause = the condition reason, and
                    # --detail reveals the fuller message.
                    job(
                        "failcell-bench-rf1",
                        recipe="failcell",
                        cell="failcell",
                        run_id="rf1",
                        failed=1,
                        start="2026-07-19T11:20:00Z",
                        finished="2026-07-19T11:45:00Z",
                        fail_reason="DeadlineExceeded",
                        fail_message="Job was active longer than specified deadline",
                    ),  # ✗ FAILED, ended 15m ago
                    # ✗ FAILED via a pod's OOMKilled lastState (see oomcell pod above) — the pod signal wins over any
                    # generic Job condition, so the row reads the operator's own vocabulary ('OOMKilled').
                    job(
                        "oomcell-bench-ro1",
                        recipe="oomcell",
                        cell="oomcell",
                        run_id="ro1",
                        failed=1,
                        start="2026-07-19T11:30:00Z",
                        finished="2026-07-19T11:50:00Z",
                    ),  # ✗ FAILED, ended 10m ago
                    job(
                        "llmb-governor-observe-1",
                        recipe="llmb-governor-observe",
                        component="governor-observe",
                        succeeded=1,
                        finished="2026-07-19T11:59:00Z",
                    ),  # infra cron → excluded
                ]
            )
        )
    )
    # NOTE: no ctxRBAC.nodes.json / ctxRBAC.all.json → shim returns Forbidden for cluster-scoped gets.

    # ── 'hotel' (physical cluster example-gpu-cluster, reached via teleport context): a disagg run whose GPU
    #    worker POD carries ONLY {app, role} labels — the llmb labels are on the DEPLOYMENT, not the pod.
    #    The owner-aware matcher must count its 16 GPUs as OURS, not FOREIGN. ──────────────────────────────
    hctx = "proxy.example.teleport.sh-example-gpu-cluster"
    recent_h = "2026-07-19T11:58:00Z"
    # pod has NO managed-by / llmb.* labels — only {app, role}; owned by the c240-decode Deployment.
    decode_pod = pod(
        "c240-decode-7d8f-abc",
        app="c240-decode",
        gpus=16,
        managed=False,
        start=recent_h,
        node="h-1",
    )
    decode_pod["metadata"]["labels"]["role"] = "decode"
    decode_pod["metadata"]["ownerReferences"] = [{"kind": "ReplicaSet", "name": "c240-decode-7d8f"}]
    (fx / f"{hctx}.pods.json").write_text(json.dumps(items([decode_pod])))
    (fx / f"{hctx}.deploys.json").write_text(
        json.dumps(
            items(
                [
                    deploy(
                        "c240-decode", app="c240-decode", ready=1, created=recent_h
                    ),  # DEPLOYMENT carries llmb labels
                ]
            )
        )
    )
    (fx / f"{hctx}.jobs.json").write_text(json.dumps(items([])))
    (fx / f"{hctx}.nodes.json").write_text(json.dumps(items([node("h-1", 16), node("h-2", 16)])))  # TOTAL 32
    # -A occupancy: our 16-GPU decode pod (unlabeled) + a genuinely foreign 16-GPU pod in another namespace.
    foreign_pod = pod("someoneelse-0", app="other", gpus=16, managed=False, ns="teamZ", node="h-2")
    (fx / f"{hctx}.all.json").write_text(json.dumps(items([decode_pod, foreign_pod])))  # OCC 32 → free 0

    # ── ctxG 'golf': IDLE but INSTALLED + a CROSS-NAMESPACE LOADING run. Its configured ns is empty, but a
    #    parked server Deployment (cell-labelled, 0/0) is staged here → live-discovered INSTALLED inventory
    #    (fix #2: an idle-but-installed cluster must NOT collapse). AND `pods -A` holds one of OUR pods in a
    #    DIFFERENT namespace (llmb-glm5) that is Running-but-not-Ready — a server loading weights with no
    #    bench Job → the cross-ns LOADING run a single-ns view is blind to (ROOT CAUSE 1 + 2). It must surface
    #    as a RUN row tagged with its namespace, and its 4 GPUs must count toward OURS.
    (fx / "ctxG.deploys.json").write_text(
        json.dumps(
            items(
                [
                    deploy(
                        "golfperf-server",
                        app="golfperf-server",
                        desired=0,
                        ready=0,
                        cell="golf-llmperf-1m",
                    ),
                ]
            )
        )
    )
    (fx / "ctxG.all.json").write_text(
        json.dumps(
            items(
                [
                    pod(
                        "glm5-server-7c9d-abc",
                        app="glm5-server",
                        gpus=4,
                        ns="llmb-glm5",
                        node="g-node-1",
                        start="2026-07-19T11:52:00Z",
                    ),  # Running, not-Ready → LOADING (704GB weight load)
                ]
            )
        )
    )
    return fx


def write_shim(root: Path, fx: Path) -> Path:
    """A fake `kubectl` that replays fixtures by (context, resource). ctxBAD* → auth failure;
    ctxRBAC → Forbidden on cluster-scoped (nodes / pods -A). A missing fixture for an OTHERWISE-reachable
    context returns an empty list (so an ambient-context / no-namespace cluster renders connected-empty
    instead of erroring) — this is what exercises the empty-ctxargs/nsargs path."""
    shim_dir = root / "bin"
    shim_dir.mkdir()
    shim = shim_dir / "kubectl"
    shim.write_text(f"""#!/usr/bin/env python3
import sys, pathlib, os, time
FX = pathlib.Path({str(fx)!r})
a = sys.argv[1:]
ctx = ""
if "--context" in a:
    ctx = a[a.index("--context") + 1]
allns = "-A" in a or "--all-namespaces" in a
combined = any(("," in x and "pods" in x) for x in a)   # `get pods,deployments,jobs` (one namespaced call)
combined_llmb = allns and any(("," in x and "deploy" in x) for x in a)  # `get deployments,jobs -A -l managed-by`
res = ""
for r in ("pods","deploy","deployments","jobs","nodes","leases","pvc"):
    if r in a:
        res = "deploys" if r in ("deploy","deployments") else r
        break
if combined:
    key = "nsread"
elif combined_llmb:
    key = "allllmb"          # cluster-wide OUR workloads (deploy,jobs -A -l managed-by=llmb-recipe)
elif res == "pvc":
    key = "pvcs"            # model-cache PVCs (-A); absent fixture -> empty list
elif res == "leases":
    key = "leases"          # model-load Leases (-A -l llmb.nvidia.com/managed=true); absent fixture -> empty
elif res == "pods" and allns:
    key = "all"
else:
    key = res
# optional call log (one line per kubectl invocation) for cache-hit / call-count assertions
log = os.environ.get("SHIM_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(f"{{ctx}} {{key}}\\n")
# optional artificial latency for a context (tests the --watch per-frame deadline / laggard path)
slow = os.environ.get("SHIM_SLOW_CTX")
if slow and (slow == "ALL" or ctx == slow):
    # SHIM_SLOW_AFTER=N → the first N calls for this context are FAST, later ones sleep. That is how a
    # --watch session gets a genuine last-good frame and THEN a laggard (the stale-frame path).
    after = int(os.environ.get("SHIM_SLOW_AFTER", "0"))
    n = 0
    if after:
        cnt = pathlib.Path(os.environ.get("SHIM_COUNT", "/tmp/.shimcount")).with_suffix(f".{{ctx}}")
        n = int(cnt.read_text()) if cnt.is_file() else 0
        cnt.write_text(str(n + 1))
    if n >= after:
        time.sleep(float(os.environ.get("SHIM_SLOW_SECS", "3")))
if ctx.startswith("ctxBAD"):
    sys.stderr.write("error: You must be logged in to the server (Unauthorized)\\n"); sys.exit(1)
if ctx == "ctxRBAC" and res == "pvc":
    sys.stderr.write('Error from server (Forbidden): persistentvolumeclaims is forbidden: User cannot list resource "persistentvolumeclaims" at the cluster scope\\n'); sys.exit(1)
if ctx == "ctxRBAC" and (res == "nodes" or (res == "pods" and allns)):
    sys.stderr.write('Error from server (Forbidden): nodes is forbidden: User cannot list resource "nodes" at the cluster scope\\n'); sys.exit(1)
import json as _j
def _items(kk):
    p = FX / f"{{ctx}}.{{kk}}.json"
    return _j.loads(p.read_text()).get("items", []) if p.is_file() else []
if key == "nsread":     # one combined `get pods,deployments,jobs` → merge the three fixtures (items carry kind)
    merged = _items("pods") + _items("deploys") + _items("jobs")
    sys.stdout.write(_j.dumps({{"apiVersion": "v1", "kind": "List", "items": merged}}))
else:
    f = FX / f"{{ctx}}.{{key}}.json"
    sys.stdout.write(f.read_text() if f.is_file() else '{{"apiVersion":"v1","kind":"List","items":[]}}')
""")
    shim.chmod(0o755)
    return shim_dir


_UNSET = object()  # `cache` not passed at all — see prof() below


def write_profiles(root: Path) -> Path:
    d = root / "cluster-profiles"
    d.mkdir()

    def prof(name, ctx, ns, connect=None, gpu=None, cluster=None, cache=_UNSET):
        # ctx/ns == None → OMIT the line entirely (mirrors a real profile that never sets it — e.g. our
        # dynamo-gcp-dev-02.env has no KUBE_CONTEXT → ambient context → empty ctxargs at the kubectl call).
        txt = f'CLUSTER="{name}"\n'
        if ctx is not None:
            txt += f'KUBE_CONTEXT="{ctx}"\n'
        if cluster is not None:
            txt += f'KUBE_CLUSTER="{cluster}"\n'
        if ns is not None:
            txt += f'NAMESPACE="{ns}"\n'
        if gpu is not None:
            txt += f'GPU_PRODUCT="{gpu}"\n'
        if connect is not None:
            txt += f'CONNECT_CMD="{connect}"\n'
        # MODEL_CACHE_PVC is CLUSTER truth (model_cache.py) and the ONLY input that says where a model's
        # weights belong. It reaches the renderer through meta field 10 → --profiles-dir, and without it the
        # ledger cannot tell an empty claim from an unconfigured one.
        # EVERY profile gets one by default, because a real one does: a profile WITHOUT it cannot install a
        # single cell (install fails closed), which fleet now treats as a hard blocker that EXPANDS the
        # cluster. Leaving it off here would have made half the fixture clusters permanently un-collapsed
        # and silently destroyed what the idle-collapse assertions are testing. `cache=None` asks for that
        # unconfigured profile explicitly; the pure unit covers it.
        if cache is _UNSET:
            cache = f"{name}-model-cache"  # named, absent from the cluster → `not installed`
        if cache is not None:
            txt += f'MODEL_CACHE_PVC="{cache}"\n'
        (d / f"{name}.env").write_text(txt)

    # THE FULL GLYPH LADDER, END TO END: alpha's claim exists and is Job-stamped at the pin (✓) while the
    # models sharing it are unvouched (~); bravo names a claim that does not exist (○ MISSING); charlie's PVC
    # read is RBAC-Forbidden (? UNKNOWN, never MISSING).
    prof("alpha", "ctxA", "ours", gpu="NVIDIA-B200", cache="glm5-fp8-model-cache")
    prof("bravo", "ctxB", "ours", gpu="NVIDIA-GB200", cache="bravo-model-cache")
    prof("charlie", "ctxRBAC", "ours", gpu="NVIDIA-GB300", cache="charlie-model-cache")
    prof("delta", "ctxBAD", "ours")  # auth-fail, NO CONNECT_CMD → tsh fallback hint
    # auth-fail WITH a profile-specified CONNECT_CMD → fleet shows that exact command
    prof(
        "echo",
        "ctxBAD2",
        "ours",
        connect="tsh login --proxy=tp.example:443 && tsh kube login ctxBAD2",
    )
    # REGRESSION (bash-3.2 empty-array / set -u): a profile with NO KUBE_CONTEXT (ambient → empty ctxargs)
    # and a profile with NO NAMESPACE (empty nsargs) must run to completion, not crash with
    # `ctxargs[@]: unbound variable`. Both must still render a section.
    prof("foxtrot", None, "amb-ns")  # no KUBE_CONTEXT → ambient context → EMPTY ctxargs
    prof("golf", "ctxG", None)  # no NAMESPACE → EMPTY nsargs
    # same context string as alpha → dedup, one row [2 profiles]. It names a DIFFERENT claim on purpose: the
    # renderer must receive BOTH profiles and match on the UNION, or a second profile's MODEL_CACHE_PVC is
    # invisible (the hazard discover_model_caches' own docstring warns about).
    prof("alpha-dup", "ctxA", "ours", cache="stray-cache")
    # SAME physical cluster (KUBE_CONTEXT=ctxG) as golf, but a DIFFERENT namespace → must collapse to ONE
    # section (not a duplicate) with golf, and NOT re-list golf's cross-ns runs a second time. The dedup key is
    # the physical cluster (context), never (context, namespace).
    prof("golf-kvbm", "ctxG", "ns-kvbm")
    # PHYSICAL-CLUSTER dedup: two profiles → the SAME cluster via DIFFERENT representations. `hotel` pins a
    # teleport KUBE_CONTEXT (proxy prefix + cluster token); `hotel-glm5` sets KUBE_CLUSTER to the bare token.
    # cluster_key must resolve both to "example-gpu-cluster" → one row, [2 profiles]. `hotel` (sorted first) wins.
    prof(
        "hotel",
        "proxy.example.teleport.sh-example-gpu-cluster",
        "ours",
        gpu="NVIDIA-B200",
    )
    prof("hotel-glm5", None, "ours", cluster="example-gpu-cluster", gpu="NVIDIA-B200")
    return d


def run_fleet_full(profiles: Path, shim_dir: Path, *extra: str, bash_bin: str = "bash"):
    """Run fleet.sh under `bash_bin`; return the CompletedProcess (stdout/stderr/returncode)."""
    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
    env["FLEET_PROFILES_DIR"] = str(profiles)
    env["FLEET_NOW"] = NOW
    env["NO_COLOR"] = "1" if "--color" not in extra else ""
    return subprocess.run([bash_bin, str(FLEET_SH), *extra], capture_output=True, text=True, env=env)


def run_fleet(profiles: Path, shim_dir: Path, *extra: str) -> str:
    p = run_fleet_full(profiles, shim_dir, *extra)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
    return p.stdout


ANSI = re.compile(r"\033\[[0-9;]*m")


def _unit_node_capacity():
    """Direct unit checks on node_capacity (the pure free-NODE aggregator) — with full control over per-node
    placement, independent of the loaded integration fixtures. Covers the POSITIVE free-node path, the
    FRAGMENTATION case (GPUs free but 0 free whole-nodes → the exact GB200 pathology the header must expose),
    ours-node attribution, and the two RBAC-degraded shapes (free n/a · capacity n/a).
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    def N(name, g):
        return {
            "metadata": {"name": name},
            "status": {"allocatable": {"nvidia.com/gpu": str(g)}},
        }

    def P(node, g, *, ours=True):
        labels = {"app.kubernetes.io/managed-by": "llmb-recipe"} if ours else {}
        return {
            "metadata": {"labels": labels},
            "spec": {
                "nodeName": node,
                "containers": [{"resources": {"requests": {"nvidia.com/gpu": str(g)}}}],
            },
            "status": {"phase": "Running"},
        }

    def L(x):
        return {"items": x}

    nodes = L([N("n1", 8), N("n2", 8), N("n3", 8), N("n4", 8)])  # 4 GPU nodes ×8 = 32

    # FRAGMENTATION: two nodes full (8/8), two half (4/8) → 8 GPUs 'free' but 0 free WHOLE nodes, and the
    # biggest single free node is only 4g. A tp=8 run cannot schedule despite '8 free GPUs'.
    frag = fr.node_capacity(nodes, L([P("n1", 8), P("n2", 8), P("n3", 4), P("n4", 4)]))
    check(
        "node_capacity fragmentation: 0 free whole-nodes despite 8 free GPUs",
        frag["free_nodes"] == 0 and frag["total_nodes"] == 4,
        str(frag),
    )
    check(
        "node_capacity fragmentation: biggest_free = largest single-node free (4g), not the 8 free GPUs",
        frag["biggest_free"] == 4,
        str(frag),
    )

    # POSITIVE: one node fully empty → 1 launchable whole node, biggest_free = a full node (8g).
    pos = fr.node_capacity(nodes, L([P("n1", 8), P("n2", 8), P("n3", 4)]))  # n4 untouched → free
    check(
        "node_capacity positive: an empty node counts as 1 free whole-node (biggest_free 8g)",
        pos["free_nodes"] == 1 and pos["biggest_free"] == 8,
        str(pos),
    )

    # OURS: only nodes hosting OUR GPU pods count toward ours_nodes (foreign occupant excluded).
    own = fr.node_capacity(nodes, L([P("n1", 8, ours=True), P("n2", 8, ours=False)]), is_ours=fr.is_llmb)
    check(
        "node_capacity ours_nodes counts only nodes WE hold (1, not the foreign one)",
        own["ours_nodes"] == 1,
        str(own),
    )

    # DEGRADED: nodes readable but pods -A forbidden → free/biggest/ours = None (honest 'free n/a').
    deg = fr.node_capacity(nodes, None)
    check(
        "node_capacity degraded (no pods -A): total known, free/biggest = None",
        deg is not None and deg["total_nodes"] == 4 and deg["free_nodes"] is None and deg["biggest_free"] is None,
        str(deg),
    )
    # DEGRADED: no node list at all → None (capacity n/a).
    check(
        "node_capacity degraded (no node list): returns None",
        fr.node_capacity(None, L([P("n1", 8)])) is None,
    )


def _write_stamps(profiles: Path):
    """Drop STAGE-1 (readiness.json) + STAGE-2 (install.jsonl) stamps into <profiles>/.state so the
    3-stage `--stages` journey has real local data — including for the AUTH-FAIL `delta` cluster, to prove
    stages ① & ② render even when the cluster context is unreachable (they read local files, not kubectl).
    """
    st = profiles / ".state"
    st.mkdir(parents=True, exist_ok=True)

    def readiness(name, run_ready, passc=8, warn=0, fail=0, skip=0):
        (st / f"{name}.readiness.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "profile": name,
                    "run_ready": run_ready,
                    "level_counts": {
                        "PASS": passc,
                        "WARN": warn,
                        "FAIL": fail,
                        "SKIP": skip,
                    },
                    "checks": [],
                    "profile_hash": "x",
                    "ts": NOW,
                }
            )
        )

    def installs(name, rows):  # rows: list of (cell, preflight)
        (st / f"{name}.install.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "cell": c,
                        "recipe_hash": "h",
                        "model_repo": "m",
                        "staged": {"dataset": {"ok": True}},
                        "preflight": pf,
                        "job_mode": "bench",
                        "stamped_at": NOW,
                    }
                )
                + "\n"
                for c, pf in rows
            )
        )

    readiness("alpha", True, warn=1)  # ① RUN-READY (1 warn) → silent (a ready init is quiet)
    installs(
        "alpha",
        [
            ("recipes/llm-perf/a", "pass"),
            ("recipes/llm-perf/b", "warn"),
            ("recipes/llm-perf/c", "fail"),
        ],
    )  # ② 2/3 ready (1 blocked) → shows an INSTALL section
    readiness("bravo", False, passc=6, warn=1, fail=2, skip=1)  # ① NOT ready (2 fail) → shows an INIT section
    installs("bravo", [("recipes/llm-perf/z", "pass")])  # ② 1/1 cell fully staged → silent
    # delta is AUTH-FAIL: its INIT (blocked) + INSTALL (blocked cell) STILL render from LOCAL stamps → proves
    # the stages view is auth-robust (reads .state, not kubectl) for an unreachable cluster.
    readiness("delta", False, passc=7, fail=1)  # ① NOT ready (1 fail)
    installs("delta", [("recipes/x/y", "pass"), ("recipes/x/z", "fail")])  # ② 1/2 ready (1 blocked)
    # charlie/foxtrot/golf/hotel/echo: NO stamps → init/install silent (init ? not-run · install none staged)


def _unit_stages():
    """END-TO-END for the REDESIGNED `--stages` pane (signal-over-noise): runs the REAL fleet.sh with local
    .state stamps and asserts the new hierarchy — CLUSTER (only the active ones in full) → NAMESPACE →
    labeled INIT/INSTALL/RUN sub-sections; idle-connected clusters collapsed to one compact line each;
    idle+unreachable folded into a `+N` tail; the auth-robust render of INIT/INSTALL for an unreachable
    cluster from local stamps; and the dropped noise (no `journey ①②③`, no idle-server/helper-job counts).
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = write_fixtures(root)
        shim_dir = write_shim(root, fx)
        profiles = write_profiles(root)
        _write_stamps(profiles)
        out = run_fleet(profiles, shim_dir, "--stages")  # NOT --fast → idle lines carry free-capacity
        plain = ANSI.sub("", out)

        # ── headline reused from the default pane; the confusing `journey ①②③` / `JOURNEY` rollup is GONE ──
        check(
            "stages: ACTIVE headline present, no `journey ①②③` / `JOURNEY` rollup noise",
            plain.startswith("ACTIVE ") and "journey  ①" not in plain and "JOURNEY  init:" not in plain,
            plain.splitlines()[0] if plain else "",
        )

        # ── FULL cluster alpha: header · connected → ns → INSTALL (partial) + RUN (active); init ready = silent ──
        alpha = _cblock(plain, "alpha")
        check(
            "stages: alpha header carries the CLUSTER label + connection state (━━ CLUSTER alpha … · connected)",
            alpha and alpha.splitlines()[0].startswith("━━ CLUSTER alpha") and "· connected" in alpha.splitlines()[0],
            alpha.splitlines()[0] if alpha else "",
        )
        # CAPACITY is now a CLUSTER-level ├─ child branch (same idiom as INIT), and the GPU product moved OFF the bar.
        # alpha: 2 GPU nodes (8+8); gpu-node-1 full, gpu-node-2 half → 0 fully-FREE nodes, biggest free 4 GPUs.
        check(
            "stages: the GPU product is no longer crammed on the cluster bar ([B200] gone from the header)",
            "[B200]" not in alpha.splitlines()[0],
            alpha.splitlines()[0] if alpha else "",
        )
        check(
            "stages: alpha shows a CLUSTER-level CAPACITY branch — B200 · 0/2 GPU nodes free · biggest free 4 GPUs",
            any(l.lstrip("│├└─ ").startswith("CAPACITY") for l in alpha.splitlines())
            and "B200 · 0/2 GPU nodes free · biggest free 4 GPUs" in alpha,
            alpha,
        )
        check(
            "stages: alpha's zero-free-node state carries the whole-node blocker hint",
            "whole-node cells cannot schedule" in alpha,
            alpha,
        )
        # CAPACITY sits at the CLUSTER level: above the first NAMESPACE bar, on a ├─ branch like INIT.
        _cap_i = next(n for n, l in enumerate(alpha.splitlines()) if l.lstrip("│├└─ ").startswith("CAPACITY"))
        _ns_i = next(n for n, l in enumerate(alpha.splitlines()) if "━━ NAMESPACE" in l)
        check(
            "stages: CAPACITY sits at the CLUSTER level — above the first NAMESPACE bar, on a ├─ child branch",
            _cap_i < _ns_i and alpha.splitlines()[_cap_i].startswith("├─ CAPACITY"),
            alpha.splitlines()[_cap_i],
        )
        # NAMESPACE header is a magenta `━━ NAMESPACE <ns> … ━━` bar hanging off the cluster bar on a ├─/└─ tree
        # branch (grey guides). So the nesting reads unmistakably as a tree.
        check(
            "stages: alpha's namespace bar hangs off the cluster on a tree branch (─ ━━ NAMESPACE ours)",
            any("─ ━━ NAMESPACE ours" in l for l in alpha.splitlines()),
            alpha,
        )
        # INSTALLED = the "what can be run here" inventory: EVERY staged cell (not only blocked ones), each
        # ✓ ready · ⚠ warn · ✗ FAILED, sitting ABOVE the RUN section.
        check(
            "stages: alpha shows an INSTALLED inventory (3 cells · 1 ready · 1 warn · 1 FAILED · stamp) — every staged cell",
            "INSTALLED" in alpha
            and "3 cells · 1 ready · 1 warn · 1 FAILED · stamp" in alpha
            and "✓ llm-perf/a" in alpha
            and "⚠ llm-perf/b" in alpha,
            alpha,
        )
        check(
            "stages: alpha INSTALLED ✗ FAILED cell carries a why + concrete fix hint",
            "✗ llm-perf/c" in alpha
            and "FAILED — preflight blocked" in alpha
            and "fix: llmb-k8s install --cluster alpha" in alpha,
            alpha,
        )
        check(
            "stages: alpha INSTALLED section sits ABOVE the RUN / SERVER section",
            "INSTALLED" in alpha and "RUN / SERVER" in alpha and alpha.index("INSTALLED") < alpha.index("RUN / SERVER"),
            alpha,
        )
        check(
            "stages: alpha init is READY → silent (no INIT section)",
            "INIT" not in alpha,
            alpha,
        )
        check(
            "stages: alpha RUN section shows the active run (modelx-r1c1, 8g, 41%) + its nested └ svc",
            "RUN" in alpha
            and "● RUNNING" in alpha
            and "modelx-r1c1" in alpha
            and "8 GPUs" in alpha
            and "└ svc" in alpha,
            alpha,
        )
        check(
            "stages: ✓-done runs omitted from the RUN section (settled successes are not 'happening')",
            "metriccell" not in alpha and "nocache" not in alpha,
            alpha,
        )

        # ── FULL cluster bravo: INIT (blocked) shown; install ok = silent; RUN surfaces an ORPHAN + STUCK ──
        bravo = _cblock(plain, "bravo")
        check(
            "stages: bravo shows an INIT section (blocked: NOT ready, 2 fail) with a `llmb-k8s init` fix hint",
            "INIT" in bravo and "NOT ready (2 fail)" in bravo and "llmb-k8s init bravo" in bravo,
            bravo,
        )
        check(
            "stages: bravo INSTALLED shows its staged cell as ✓ ready (all-ready inventory is still shown)",
            "INSTALLED" in bravo and "1 cell · 1 ready" in bravo and "✓ llm-perf/z" in bravo,
            bravo,
        )
        check(
            "stages: bravo ordering — INIT (blocked) ABOVE INSTALLED ABOVE RUN / SERVER",
            bravo.index("INIT") < bravo.index("INSTALLED") < bravo.index("RUN / SERVER"),
            bravo,
        )
        check(
            "stages: bravo RUN surfaces the standalone ORPHAN holding GPU with a reclaim hint",
            "● ORPHAN" in bravo and "orphan-server" in bravo and "reclaim: llmb-k8s reclaim-gpu bravo" in bravo,
            bravo,
        )
        check(
            "stages: bravo RUN shows a STUCK wedged run row",
            "STUCK" in bravo and "crash-server" in bravo,
            bravo,
        )

        # ── FULL cluster charlie: no stamps, but a parked server Deployment (cell=parkrun) is LIVE-DISCOVERED →
        #    INSTALLED shows it as ✓ from ground truth (source: live), even with zero stamps. Plus a FAILED run. ──
        charlie = _cblock(plain, "charlie")
        check(
            "stages: charlie INSTALLED is LIVE-DISCOVERED (no stamps) — ✓ parkrun · source live",
            "INSTALLED" in charlie
            and "· live" in charlie
            and "✓ parkrun" in charlie
            and "server deployed" in charlie
            and "INIT" not in charlie,
            charlie,
        )
        check(
            "stages: charlie shown full and still surfaces its ✗ FAILED run",
            "✗ FAILED" in charlie and "failcell-rf1" in charlie,
            charlie,
        )
        check(
            "stages: charlie ✗ FAILED row carries the why + logs one-liner",
            "why: DeadlineExceeded" in charlie and "logs: llmb-k8s logs rf1" in charlie,
            charlie,
        )
        # charlie is RBAC-degraded (cluster-scoped gets FORBIDDEN) → CAPACITY still renders, honestly saying the
        # node read was unavailable rather than implying an empty cluster or printing a bogus `0 free`.
        check(
            "stages: RBAC-degraded charlie still shows CAPACITY, honestly (`node capacity unknown (no node read)`)",
            "CAPACITY" in charlie and "node capacity unknown (no node read)" in charlie and "0 free" not in charlie,
            charlie,
        )
        # …and the SAME honesty one level down: `get pvc -A` is Forbidden here, so the model-cache half of the
        # inventory is MISSING. The cells it did see are still listed, flagged ⚠ PARTIAL — never passed off as
        # the whole inventory, and never (the B200 bug) as "— nothing installed —".
        check(
            "stages: RBAC-degraded charlie flags its INSTALLED list ⚠ PARTIAL (model caches unreadable)",
            "⚠ PARTIAL" in charlie and "RBAC" in charlie and "nothing installed" not in charlie,
            charlie,
        )

        # ── THE MODEL LEDGER, END TO END THROUGH fleet.sh. The profile FILES ride meta field 10 and are
        #    resolved against --profiles-dir (now passed for EVERY pane, not just --stages), so the ledger
        #    knows which claim each model's weights belong in. The four glyphs are exercised on four real
        #    clusters: alpha ✓/~ · bravo ○ · charlie ? · and no cluster may fake the ones it cannot prove. ──
        check(
            "ledger (e2e): alpha's Job-stamped claim AT THE PINNED revision is the only ✓ on the fleet",
            "MODELS" in alpha and "✓ glm5-fp8" in alpha and "download-Job stamp · matches pin" in alpha,
            alpha,
        )
        check(
            "ledger (e2e): the models SHARING that claim read ~ UNVERIFIED, and the claim's size prints once",
            "~ nemotron-ultra-nvfp4" in alpha
            and "~ qwen3-0-6b" in alpha
            and alpha.count("1200Gi") == 1
            and alpha.count("shared") == 2
            and "1 attested · 0 verified (no in-volume check) · 2 UNVERIFIED · 0 missing" in alpha,
            alpha,
        )
        check(
            "ledger (e2e): the UNION of BOTH collapsed profiles' claims is matched — alpha-dup's is not hidden",
            "stray-cache" in alpha,
            alpha,
        )
        check(
            "ledger (e2e): a revision is 12 chars wide on EVERY row, so no label length can skew the table",
            all(
                len(l.split("@", 1)[1].split()[0]) == 12
                for l in alpha.splitlines()
                if "@" in l and l.lstrip().startswith(("├─", "└─"))
            ),
            alpha,
        )
        check(
            "ledger (e2e): bravo's configured claim does not exist → ○ MISSING rows, an ABSENCE you can see",
            "○ glm5-fp8" in bravo and "3 missing" in bravo and "claim bravo-model-cache does not exist" in bravo,
            bravo,
        )
        check(
            "ledger (e2e): charlie's PVC read is FORBIDDEN → ? UNKNOWN naming the cause, never ○ MISSING",
            "? glm5-fp8" in charlie
            and "3 UNKNOWN" in charlie
            and "0 missing" in charlie
            and "can-i list pvc" in charlie
            and "○ " not in charlie,
            charlie,
        )
        check(
            "ledger (e2e): NO cluster in the whole pane offers a `kubectl label pvc` backfill any more",
            "label pvc" not in plain and "download-complete=true" not in plain,
            plain,
        )

        # ── AUTH-ROBUST: delta is AUTH-FAIL yet shown FULL — INIT + INSTALL render from LOCAL stamps ──
        delta = _cblock(plain, "delta")
        check(
            "stages: AUTH-FAIL delta shown full; INIT + INSTALLED render from local stamps (auth-robust)",
            delta and "· auth✗" in delta.splitlines()[0] and "INIT" in delta and "INSTALLED" in delta,
            delta,
        )
        check(
            "stages: delta INSTALLED renders ✓ ready + ✗ FAILED from LOCAL stamps even while unreachable (source stamp)",
            "2 cells · 1 ready · 1 FAILED · stamp" in delta
            and "✓ x/y" in delta
            and "✗ x/z" in delta
            and "fix: llmb-k8s install --cluster delta" in delta,
            delta,
        )
        # (delta is the LAST full cluster, so its _cblock bleeds into the trailing idle/legend text — assert on
        #  the RUN *column header* which appears ONLY inside a real RUN section, not the word 'RUN' in the legend.)
        check(
            "stages: delta has NO RUN section (run state unknown while unreachable) + a connect hint",
            "─ RUN / SERVER" not in delta and "run state unknown" in delta and "tsh kube login" in delta,
            delta,
        )

        # ── IDLE-but-CONNECTED clusters collapse to ONE compact line each, under an `idle · connected:` header ──
        check(
            "stages: idle-connected clusters collapse under an `idle · connected:` header",
            "idle · connected:" in plain,
            plain,
        )
        # foxtrot is TRULY empty (no installs, no runs, no held GPUs) → the ONLY kind that still collapses.
        check(
            "stages: TRULY-empty foxtrot is a one-line compact entry, NOT a full block",
            not _cblock(plain, "foxtrot") and any("· foxtrot" in l for l in plain.splitlines()),
            plain,
        )

        # ── FIX #2 + ROOT CAUSE 1/2 + DEDUP + NS-GROUPING: golf is IDLE-in-its-configured-ns but has an INSTALL
        #    + a CROSS-NS LOADING run → shown in FULL. golf-kvbm (SAME context ctxG, different ns) must COLLAPSE
        #    into golf's ONE section (dedup by context), and each namespace gets its OWN sub-block.
        golf = _cblock(plain, "golf")
        check(
            "stages: golf does NOT collapse — idle-but-installed cluster is shown in FULL (fix #2)",
            bool(golf) and not any(l.strip().startswith("· golf") for l in plain.splitlines()),
            golf or plain,
        )
        # DEDUP: two profiles (golf + golf-kvbm) on the SAME context → exactly ONE section (no `golf-kvbm` header).
        check(
            "stages: DEDUP — golf-kvbm (same KUBE_CONTEXT, diff ns) collapses into golf → ONE section, noted",
            not _cblock(plain, "golf-kvbm")
            and plain.count("━━ CLUSTER golf ") == 1
            and "2 profiles map to this cluster" in golf,
            plain,
        )
        # NS-GROUPING: golf's INSTALLED (parked golf-llmperf-1m in its ns) and the cross-ns glm5 LOADING run each
        # sit under their OWN `ns <name>` header; the cross-ns run carries NO redundant inline `ns …·` tag.
        check(
            "stages: golf INSTALLED is live-discovered (parked server) — ✓ golf-llmperf-1m · source live",
            "INSTALLED" in golf and "· live" in golf and "✓ golf-llmperf-1m" in golf,
            golf,
        )
        check(
            "stages: golf's cross-ns LOADING run is grouped UNDER its own `━━ NAMESPACE llmb-glm5 ━━` bar (not inline-tagged)",
            any("─ ━━ NAMESPACE llmb-glm5" in l for l in golf.splitlines())
            and "● LOADING" in golf
            and "server warming (image/weights)" in golf
            and "4 GPUs" in golf
            and "ns llmb-glm5 ·" not in golf,
            golf,
        )  # note kept as the row suffix; NO inline `ns …·` tag
        gl = golf.splitlines()
        _load_i = next((n for n, l in enumerate(gl) if "● LOADING" in l), -1)
        _nsg_i = next((n for n, l in enumerate(gl) if "─ ━━ NAMESPACE llmb-glm5" in l), 99)
        check(
            "stages: the LOADING run row appears AFTER its `━━ NAMESPACE llmb-glm5 ━━` bar (correct grouping order)",
            0 <= _nsg_i < _load_i,
            "\n".join(gl),
        )

        # hotel holds 16 GPU (a real footprint) → held GPUs are a run we must SHOW: hotel no longer collapses.
        check(
            "stages: hotel (16 GPU held) is shown in FULL, never collapsed to idle (held GPUs = a run to show)",
            bool(_cblock(plain, "hotel")) and not any(l.strip().startswith("· hotel") for l in plain.splitlines()),
            plain,
        )

        # ── IDLE + UNREACHABLE folds into the terse tail (echo: auth-fail, no stamps), never a full block ──
        check(
            "stages: idle+unreachable cluster folds into a `+N unreachable/idle` tail (echo), not a full block",
            "unreachable/idle" in plain and "echo" in plain and not _cblock(plain, "echo"),
            plain,
        )

        # ── DROPPED NOISE: no idle-server / helper-job / parked counts, no `no active runs of ours` filler ──
        check(
            "stages: dropped noise (no idle-server/helper-job/parked, no `no active runs of ours`)",
            "idle-server" not in plain
            and "helper-job" not in plain
            and "no active runs of ours" not in plain
            and "parked-run" not in plain,
            plain,
        )

        # ── DEFAULT (no --stages) unchanged: no stage sections, still the active-runs-first pane ──
        plain_default = ANSI.sub("", run_fleet(profiles, shim_dir, "--fast"))
        check(
            "stages: default view unchanged (no `idle · connected:`, still `no active runs of ours`)",
            "idle · connected:" not in plain_default
            and "no active runs of ours" in plain_default
            and "journey  ①" not in plain_default,
            plain_default[:200],
        )


def _unit_installed_inventory():
    """PURE unit for the INSTALLED-inventory classifier + the stamp-on-stage round-trip (PART A ↔ PART B).
    Asserts (1) fleet_render classifies each install-stamp record into ready/warn/FAILED with the right why +
    fix, and (2) a stamp written by the `run` inline-stage path (install.record_stage_stamp) is read back by
    fleet_render.read_install_stamps and lands in that same inventory — so a cell staged by `run`, not only by
    `install`, shows up in fleet's INSTALLED section."""
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr
    import install as inst

    # (1) classifier — one record per state, incl. a run-path 'skipped' re-stage (staged ok=None from --skip-stage)
    installs = {
        "recipes/a": {"staged": {"stage-dataset": {"ok": True}}, "preflight": "pass"},
        "recipes/b": {"staged": {"stage-dataset": {"ok": True}}, "preflight": "warn"},
        "recipes/c": {
            "staged": {"stage-dataset": {"ok": False}},
            "preflight": "skipped",
        },  # staging FAILED
        "recipes/d": {
            "staged": {"stage-dataset": {"ok": None}},
            "preflight": "pass",
        },  # skipped re-stage
        "recipes/e": {
            "staged": {"stage-dataset": {"ok": None, "reason": "required input missing"}},
            "preflight": "skipped",
        },  # needs-input
        "recipes/f": {
            "staged": {"stage-dataset": {"ok": True}},
            "preflight": "fail",
        },  # preflight blocked
    }
    rows, counts = fr._install_inventory(installs)
    by = {r["cell"]: r for r in rows}
    check(
        "installed: ready cell (staged ok + preflight pass) → ✓ ready",
        by["recipes/a"]["state"] == "ready" and by["recipes/a"]["glyph"] == "✓",
        str(by["recipes/a"]),
    )
    check(
        "installed: preflight WARN → ⚠ warn (advisory, still runnable)",
        by["recipes/b"]["state"] == "warn" and "WARN" in by["recipes/b"]["why"],
        str(by["recipes/b"]),
    )
    check(
        "installed: staging FAILED (stage ok=False) → ✗ FAILED with why 'staging failed'",
        by["recipes/c"]["state"] == "failed" and by["recipes/c"]["why"] == "staging failed",
        str(by["recipes/c"]),
    )
    check(
        "installed: skipped re-stage (stage ok=None + preflight pass) → ⚠ (not a false ready/failed)",
        by["recipes/d"]["state"] == "warn",
        str(by["recipes/d"]),
    )
    check(
        "installed: needs-input surfaces its reason + an install fix hint",
        by["recipes/e"]["state"] == "warn"
        and by["recipes/e"]["fix_kind"] == "needs-input"
        and "llmb-k8s install" in fr._install_fix("needs-input", "gb300", "recipes/e"),
        str(by["recipes/e"]),
    )
    check(
        "installed: preflight FAIL (staged ok) → ✗ FAILED 'preflight blocked'",
        by["recipes/f"]["state"] == "failed" and by["recipes/f"]["why"] == "preflight blocked",
        str(by["recipes/f"]),
    )
    check(
        "installed: counts roll up — total==6 · ready==1 · warn==3 · failed==2",
        counts["total"] == 6 and counts["failed"] == 2 and counts["ready"] == 1 and counts["warn"] == 3,
        str(counts),
    )
    check(
        "installed: attention-first order — FAILED rows precede warn rows precede ready rows",
        [r["state"] for r in rows] == sorted([r["state"] for r in rows], key=lambda s: fr._INSTALL_RANK[s]),
        [r["cell"] for r in rows],
    )
    # `cells`, not `installed`: the header already says INSTALLED, and it sits beside MODELS and STORAGE
    # headers counting models and PVCs — three sections, three units, each naming its own.
    check(
        "installed: summary line reads 'N cells · X ready · Y warn · Z FAILED · <source>'",
        fr._install_summary(counts, "stamp") == "6 cells · 1 ready · 3 warn · 2 FAILED · stamp",
        fr._install_summary(counts, "stamp"),
    )

    # (2) stamp-on-stage round-trip: run.sh's inline stage → install.record_stage_stamp → fleet reads it.
    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / ".state"
        # an OK stage (the `run` happy path) and a FAILED stage — both must be recorded (failures not silent).
        inst.record_stage_stamp(
            "recipes/llm-perf/ok",
            "gb300",
            step_key="stage-dataset",
            staged="ok",
            preflight="pass",
            state_dir=state,
            catalog=[],
        )
        inst.record_stage_stamp(
            "recipes/llm-perf/bad",
            "gb300",
            step_key="stage-dataset",
            staged="failed",
            preflight="pass",
            state_dir=state,
            catalog=[],
        )
        got = fr.read_install_stamps(str(Path(td)), "gb300")
        check(
            "stamp-on-stage: run-path stamp round-trips through fleet's install-stamp reader (2 cells)",
            set(got) == {"recipes/llm-perf/ok", "recipes/llm-perf/bad"},
            str(list(got)),
        )
        rrows, rc2 = fr._install_inventory(got)
        rby = {r["cell"]: r for r in rrows}
        check(
            "stamp-on-stage: an OK `run` stage lands as ✓ ready in fleet INSTALLED",
            rby["recipes/llm-perf/ok"]["state"] == "ready",
            str(rby.get("recipes/llm-perf/ok")),
        )
        check(
            "stamp-on-stage: a FAILED `run` stage lands as ✗ FAILED in fleet INSTALLED (not silent)",
            rby["recipes/llm-perf/bad"]["state"] == "failed",
            str(rby.get("recipes/llm-perf/bad")),
        )
        check(
            "stamp-on-stage: stamp file is 0600 (owner-only, per the install-stamp contract)",
            (inst.install_stamp_path("gb300", state).stat().st_mode & 0o777) == 0o600,
            oct(inst.install_stamp_path("gb300", state).stat().st_mode & 0o777),
        )


def _mkpod(
    name,
    ns,
    *,
    gpus=0,
    ready=None,
    phase="Running",
    managed=True,
    waiting=None,
    labels=None,
    start="2026-07-19T11:50:00Z",
):
    """A minimal pod dict for the PURE cross-ns discovery unit (explicit Ready condition, since the discovery
    classifier keys on readiness — a Running-but-not-Ready server is our LOADING signal).
    """
    lab = dict(labels or {})
    if managed:
        lab["app.kubernetes.io/managed-by"] = "llmb-recipe"
    conds = [] if ready is None else [{"type": "Ready", "status": "True" if ready else "False"}]
    cs = {
        "ready": bool(ready),
        "state": {"waiting": {"reason": waiting}} if waiting else {},
        "lastState": {},
        "restartCount": 0,
    }
    reqs = {"nvidia.com/gpu": str(gpus)} if gpus else {}
    return {
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": lab,
            "creationTimestamp": start,
        },
        "spec": {
            "nodeName": "n1",
            "containers": [{"name": "c", "image": "i:t", "resources": {"requests": reqs}}],
        },
        "status": {
            "phase": phase,
            "startTime": start,
            "conditions": conds,
            "containerStatuses": [cs],
        },
    }


def _unit_live_discovery():
    """PURE units for the LIVE-DISCOVERY paths (ROOT CAUSE 1 + 2 + INSTALLED ground truth):
    • discover_installed  — server Deployments (cell label) + model-download Jobs → INSTALLED ground truth.
    • build_install_inventory — LIVE wins over stamps; stamps are the offline fallback; correct source tag.
    • discover_cross_ns_runs — OUR pods in OTHER namespaces classified RUNNING/LOADING/ORPHAN, configured ns
      skipped, held GPUs surfaced with the namespace on the row."""
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    # ── discover_installed: a deployed cell, a parked cell, a cached model, a failed model, a downloading model ──
    deploys = {
        "items": [
            {
                "metadata": {"labels": {"llmb.nvidia.com/cell": "cellA"}},
                "spec": {"replicas": 1},
                "status": {"readyReplicas": 1},
            },  # deployed
            {
                "metadata": {"labels": {"llmb.nvidia.com/cell": "cellB"}},
                "spec": {"replicas": 0},
                "status": {"readyReplicas": 0},
            },  # parked
        ]
    }
    jobs = {
        "items": [
            {
                "metadata": {
                    "labels": {
                        "llmb.nvidia.com/component": "model-download",
                        "llmb.nvidia.com/model-name": "nemotron",
                    }
                },
                "status": {"succeeded": 1},
            },
            {
                "metadata": {
                    "labels": {
                        "llmb.nvidia.com/component": "model-download",
                        "llmb.nvidia.com/model-name": "glm5",
                    }
                },
                "status": {"failed": 1},
            },
            {
                "metadata": {
                    "labels": {
                        "llmb.nvidia.com/component": "model-download",
                        "llmb.nvidia.com/model-name": "qwen",
                    }
                },
                "status": {"active": 1},
            },
        ]
    }
    disc = fr.discover_installed(deploys, jobs)
    dby = {(d["kind"], d["name"]): d for d in disc}
    check(
        "discover_installed: a deployed server (cell label) → cell ready 'server deployed'",
        dby[("cell", "cellA")]["state"] == "ready" and "deployed" in dby[("cell", "cellA")]["why"],
        str(disc),
    )
    check(
        "discover_installed: a parked 0/0 server (cell label) → still a ready installed cell (parked)",
        dby[("cell", "cellB")]["state"] == "ready" and "parked" in dby[("cell", "cellB")]["why"],
        str(disc),
    )
    check(
        "discover_installed: a Complete model-download Job → model cached (ready)",
        dby[("model", "nemotron")]["state"] == "ready",
        str(disc),
    )
    check(
        "discover_installed: a Failed model-download Job → model download FAILED",
        dby[("model", "glm5")]["state"] == "failed",
        str(disc),
    )
    check(
        "discover_installed: an active model-download Job → model downloading (warn)",
        dby[("model", "qwen")]["state"] == "warn",
        str(disc),
    )

    # ── build_install_inventory: LIVE wins over a stamp for the same cell; models never collide; source tag ──
    name_map = {"cellA-name": "recipes/x/cellA"}
    discovered = [
        {
            "kind": "cell",
            "name": "cellA-name",
            "state": "ready",
            "why": "server deployed",
        },
        {"kind": "model", "name": "nemotron", "state": "ready", "why": "model cached"},
    ]
    installs = {
        "recipes/x/cellA": {
            "staged": {"stage-dataset": {"ok": False}},
            "preflight": "skipped",
        },  # stamp says FAILED
        "recipes/y/only-stamped": {
            "staged": {"stage-dataset": {"ok": True}},
            "preflight": "pass",
        },
    }
    rows, counts, source = fr.build_install_inventory(installs, discovered, name_map)
    rby = {r["cell"]: r for r in rows}
    check(
        "inventory: LIVE discovery WINS over a stale stamp — the live cell reads ✓ ready, not the stamp's ✗",
        rby["recipes/x/cellA"]["state"] == "ready" and rby["recipes/x/cellA"]["source"] == "live",
        str(rby),
    )
    check(
        "inventory: a stamp-only cell (no live evidence) still shows as the offline fallback (source stamp)",
        rby["recipes/y/only-stamped"]["state"] == "ready" and rby["recipes/y/only-stamped"]["source"] == "stamp",
        str(rby),
    )
    # A MODEL IS NOT A CELL, so it is not in this section at all — INSTALLED counts what can be RUN here.
    # The Job-derived model evidence feeds the MODEL LEDGER instead (fleet_render.download_job_facts), which
    # is keyed by catalog model and can therefore represent two models in one claim, and a model in none.
    check(
        "inventory: a discovered MODEL is not an INSTALLED row — INSTALLED counts cells, MODELS counts models",
        not any(k.startswith("model:") for k in rby) and set(rby) == {"recipes/x/cellA", "recipes/y/only-stamped"},
        str(list(rby)),
    )
    check(
        "inventory: the model evidence is not lost — it reaches the ledger as a download-Job fact",
        [(f["model"], f["done"], f["claim"]) for f in fr.download_job_facts(jobs["items"]) if f["model"] == "nemotron"]
        == [("nemotron", True, "")],
        str(fr.download_job_facts(jobs["items"])),
    )
    check(
        "inventory: source is 'live+stamp' when both a live item and a stamp-only item are present",
        source == "live+stamp",
        source,
    )
    check(
        "inventory: an offline-only cluster (no discovery) tags source 'stamp'",
        fr.build_install_inventory(installs, [], name_map)[2] == "stamp",
        fr.build_install_inventory(installs, [], name_map)[2],
    )

    # ── discover_cross_ns_runs: OUR pods across namespaces, classified; configured ns skipped ──
    now = fr._parse_ts(NOW)
    is_ours = fr.make_ours_matcher([], [], "cfg-ns")
    old = "2026-07-19T09:00:00Z"  # 3h before NOW → past GRACE (not "recently created")
    allpods = {
        "items": [
            # configured ns → MUST be skipped (the single-ns path renders it richly)
            _mkpod("home-server-1", "cfg-ns", gpus=4, ready=True),
            # RUNNING: a ready bench pod in another ns
            _mkpod(
                "glm-bench-r1-0",
                "ns-glm",
                gpus=0,
                ready=True,
                labels={
                    "llmb.nvidia.com/component": "bench",
                    "llmb.nvidia.com/cell": "glmrun",
                },
            ),
            # LOADING: a server up but NOT ready (cold weight load), holding GPUs, no bench
            _mkpod(
                "serving-server-xyz",
                "ns-serving",
                gpus=8,
                ready=False,
                labels={"llmb.nvidia.com/cell": "servingrun"},
            ),
            # PARKED (SAFE default): a ready GPU server, no bench, no owning Job → could be a --skip-server park →
            # NEVER a reclaim footgun.
            _mkpod(
                "park-server-abc",
                "ns-park",
                gpus=2,
                ready=True,
                start=old,
                labels={"llmb.nvidia.com/cell": "parkrun"},
            ),
            # ORPHAN (only-safe reclaim): a stale ready server whose run-owner Job in the ns already FINISHED.
            _mkpod(
                "orph-server-xyz",
                "ns-orph",
                gpus=4,
                ready=True,
                start=old,
                labels={"llmb.nvidia.com/cell": "orphrun"},
            ),
            # a DISAGG server pod carrying ONLY {app,role} (llmb labels live on its Deployment, not the pod) in a
            # non-configured ns — attributed to us via a cluster-wide Deployment selector (the real GLM-5 gap).
            _mkpod(
                "glm5-1p1d-abc123",
                "ns-disagg",
                gpus=8,
                ready=False,
                managed=False,
                labels={"app": "glm5-1p1d-prefill", "role": "prefill"},
            ),
            # FOREIGN: another team's GPU pod (not llmb, no matching deploy) → never ours
            _mkpod("someone-else", "ns-teamz", gpus=8, ready=True, managed=False),
        ]
    }
    # cluster-wide OUR Deployments (deploy -A -l managed-by) + Jobs — the read that unlocks disagg attribution,
    # cell names, and the owner-state (parked-vs-orphan) judgement.
    all_deploys = [
        {
            "metadata": {
                "namespace": "ns-disagg",
                "name": "glm5-1p1d-prefill",
                "labels": {
                    "app.kubernetes.io/managed-by": "llmb-recipe",
                    "llmb.nvidia.com/cell": "glm5-1p1d",
                },
            },
            "spec": {"selector": {"matchLabels": {"app": "glm5-1p1d-prefill"}}},
        }
    ]
    all_jobs = [
        # ns-orph: a run-owner Job that has already COMPLETED → its server is a genuine orphan (GC lagging).
        {
            "metadata": {
                "namespace": "ns-orph",
                "name": "orphrun-runowner-o1",
                "labels": {"app.kubernetes.io/component": "run-owner"},
            },
            "status": {"succeeded": 1},
        },
    ]
    xrows = fr.discover_cross_ns_runs(
        allpods,
        is_ours,
        "cfg-ns",
        "gb300",
        now,
        all_deploys=all_deploys,
        all_jobs=all_jobs,
    )
    xby = {r["ns"]: r for r in xrows}
    check(
        "cross-ns: the CONFIGURED ns is skipped (covered by the single-ns path — no double render)",
        "cfg-ns" not in xby,
        str(list(xby)),
    )
    check(
        "cross-ns: a foreign team's GPU pod is NOT discovered as ours",
        "ns-teamz" not in xby,
        str(list(xby)),
    )
    check(
        "cross-ns: a ready bench pod in another ns → ● RUNNING, carrying its ns (the ns drives the ns-grouping)",
        xby["ns-glm"]["status"] == fr.RUNNING
        and xby["ns-glm"]["ns"] == "ns-glm"
        and "ns " not in xby["ns-glm"]["suffix"],
        str(xby.get("ns-glm")),
    )  # no redundant inline `ns` tag
    check(
        "cross-ns: a Running-but-not-Ready server (weights loading) → ● LOADING with a warming note + held GPUs",
        xby["ns-serving"]["status"] == fr.LOADING
        and xby["ns-serving"]["gpus"] == 8
        and "warming" in xby["ns-serving"]["suffix"]
        and xby["ns-serving"]["ns"] == "ns-serving",
        str(xby.get("ns-serving")),
    )
    # ── the SAFETY refinement (parked-vs-orphan): the reclaim footgun ──
    check(
        "cross-ns SAFE: a ready server with NO owning Job → ● PARKED, held, and NO reclaim hint (footgun avoided)",
        xby["ns-park"]["status"] == fr.PARKED
        and "reclaim" not in xby["ns-park"]["suffix"]
        and "held" in xby["ns-park"]["suffix"],
        str(xby.get("ns-park")),
    )
    check(
        "cross-ns SAFE: reclaim hint ONLY when a run-owner FINISHED but its server lingers → ● ORPHAN + reclaim",
        xby["ns-orph"]["status"] == fr.ORPHAN
        and "reclaim: llmb-k8s reclaim-gpu gb300 -n ns-orph" in xby["ns-orph"]["suffix"],
        str(xby.get("ns-orph")),
    )
    # ── the cluster-wide Deployment attribution that fixes the real GLM-5 gap ──
    check(
        "cross-ns: a disagg SERVER pod (labels only on its Deployment) is attributed via a cluster-wide "
        "selector + named by the Deployment's cell → ● LOADING glm5-1p1d in ns-disagg (the real GLM-5 case)",
        "ns-disagg" in xby
        and xby["ns-disagg"]["name"] == "glm5-1p1d"
        and xby["ns-disagg"]["status"] == fr.LOADING
        and xby["ns-disagg"]["gpus"] == 8,
        str(xby.get("ns-disagg")),
    )
    active_owner = [
        {
            "metadata": {
                "namespace": "ns-orph",
                "name": "orphrun-runowner-a1",
                "labels": {"app.kubernetes.io/component": "run-owner"},
            },
            "status": {"active": 1},
        }
    ]  # same server, but its run-owner is ALIVE this time
    xby2 = {
        r["ns"]: r
        for r in fr.discover_cross_ns_runs(
            allpods,
            is_ours,
            "cfg-ns",
            "gb300",
            now,
            all_deploys=all_deploys,
            all_jobs=active_owner,
        )
    }
    check(
        "cross-ns SAFE: an ACTIVE run-owner keeps the same server ● PARKED (owned, in-use), never reclaim",
        xby2["ns-orph"]["status"] == fr.PARKED and "reclaim" not in xby2["ns-orph"]["suffix"],
        str(xby2.get("ns-orph")),
    )

    # ── loading-cluster-doesnt-collapse: a cluster whose ONLY signal is a cross-ns LOADING run is shown FULL ──
    e_loading = {
        "state": "connected",
        "name": "c",
        "namespace": "cfg-ns",
        "jobs": [],
        "standalone": [],
        "idle": [],
        "ours_gpu": 8,
        "readiness": None,
        "installs": {},
        "discovered": [],
        "xns_runs": [dict(xby["ns-serving"])],
    }
    info = fr._stage_classify(e_loading, now, gpu_only=False)
    check(
        "collapse-fix: a cluster with ONLY a cross-ns LOADING run (held GPUs) is classified FULL, not idle",
        info["full"] and info["held"],
        str({k: info[k] for k in ("full", "held")}),
    )
    check(
        "collapse-fix: that LOADING run appears in the cluster's RUN rows",
        any(r["status"] == fr.LOADING for r in info["run_rows"]),
        str(info["run_rows"]),
    )
    e_empty = {
        "state": "connected",
        "name": "d",
        "namespace": "cfg-ns",
        "jobs": [],
        "standalone": [],
        "idle": [],
        "ours_gpu": 0,
        "readiness": None,
        "installs": {},
        "discovered": [],
        "xns_runs": [],
    }
    check(
        "collapse-fix: a TRULY-empty cluster (no installs/runs/held GPUs) still collapses (full == False)",
        not fr._stage_classify(e_empty, now, gpu_only=False)["full"],
        "expected full=False",
    )

    # ── NS-GROUPING (Cluster → Namespace → INSTALLED/RUN): a cluster with runs in 3 namespaces → 3 ns groups,
    #    ordered by activity (RUNNING > LOADING > held), each holding ONLY its own runs/installs (no dup). ──
    inv = [
        {
            "cell": "a",
            "disp": "a",
            "ns": "ns-run",
            "state": "ready",
            "why": "",
            "fix_kind": "",
            "source": "live",
        },
        {
            "cell": "b",
            "disp": "b",
            "ns": "ns-load",
            "state": "ready",
            "why": "",
            "fix_kind": "",
            "source": "live",
        },
        {
            "cell": "c",
            "disp": "c",
            "ns": "ns-park",
            "state": "ready",
            "why": "",
            "fix_kind": "",
            "source": "stamp",
        },
    ]
    runrows = [
        {"level": "job", "status": fr.PARKED, "name": "pk", "ns": "ns-park", "gpus": 2},
        {"level": "job", "status": fr.RUNNING, "name": "rn", "ns": "ns-run", "gpus": 4},
        {
            "level": "job",
            "status": fr.LOADING,
            "name": "ld",
            "ns": "ns-load",
            "gpus": 8,
        },
    ]
    grps = fr._ns_groups(inv, runrows)
    check(
        "ns-grouping: 3 namespaces → exactly 3 ns sub-groups (one per namespace)",
        len(grps) == 3 and [g["ns"] for g in grps] == ["ns-run", "ns-load", "ns-park"],  # activity order
        str([g["ns"] for g in grps]),
    )
    check(
        "ns-grouping: each ns group holds ONLY its own run (no cross-namespace leakage / duplication)",
        all(len(g["runs"]) == 1 and g["runs"][0]["ns"] == g["ns"] for g in grps)
        and all(len(g["inv"]) == 1 and g["inv"][0]["ns"] == g["ns"] for g in grps),
        str(grps),
    )
    check(
        "ns-grouping: per-ns INSTALLED counts + source are computed per namespace",
        grps[0]["counts"]["total"] == 1 and grps[2]["source"] == "stamp",
        str(grps),
    )
    check(
        "ns-grouping: a namespace with NOTHING (no install, no run) is omitted (no empty ns block)",
        all(g["ns"] in ("ns-run", "ns-load", "ns-park") for g in grps),
        str([g["ns"] for g in grps]),
    )


def _unit_ns_installed_hierarchy():
    """ACCEPTANCE for the Cluster → Namespace → {INSTALLED, RUN / SERVER} hierarchy with every namespace
    carrying its OWN Installed section. Builds ONE connected cluster with TWO managed namespaces:
      • `ours`         — a cached model (model-download Job Complete) + a GPU-holding PARKED server (up, no
                         active bench) → INSTALLED shows the model-cache + the staged cell, RUN / SERVER shows
                         the ● PARKED server row (with the `server up, no active bench` hint).
      • `llmb-cache` — ONLY a cached model, no runs → still shows its OWN INSTALLED section AND a RUN /
                         SERVER section reading `no active runs`.
    Asserts both nest under their own INDENTED `━━ NAMESPACE … ━━` bar beneath the full-width cluster bar.
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    now = fr._parse_ts(NOW)
    recent = "2026-07-19T11:56:00Z"  # inside the grace window → the ready server reads PARKED, not ORPHAN

    def _dl(name, ns, model):  # a Complete model-download Job → a cached model in `ns`
        return {
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": ns,
                "labels": {
                    "app.kubernetes.io/managed-by": "llmb-recipe",
                    "llmb.nvidia.com/component": "model-download",
                    "llmb.nvidia.com/model-name": model,
                },
            },
            "spec": {"template": {"spec": {"containers": [{"name": "c", "env": []}]}}},
            "status": {"active": 0, "succeeded": 1, "failed": 0},
        }

    srv = deploy(
        "qa-agg-server",
        app="qa-agg-server",
        desired=1,
        ready=1,
        cell="llm-perf/qa-agg",
        created=recent,
    )  # server UP holding GPUs, no bench
    srvpod = pod("qa-agg-server-abc", app="qa-agg-server", gpus=8, start=recent)
    e = fr.build_cluster(
        "qa",
        "ctxQ",
        "ours",
        items([srvpod]),
        items([srv]),
        items([_dl("nemo-download", "ours", "nemotron-ultra")]),
        None,
        None,
        now=now,
        all_deploys=[],
        all_jobs=[_dl("glm5-download", "llmb-cache", "glm5")],
    )
    e["gpu_type"], e["profiles"], e["stale"], e["readiness"], e["installs"] = (
        "B200",
        1,
        "",
        None,
        {},
    )
    plain = ANSI.sub("", fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True))

    # two-tier bars: a full-width CLUSTER bar, then an INDENTED NAMESPACE bar per ns.
    check(
        "ns-hierarchy: cluster renders a full-width `━━ CLUSTER qa … · connected ━━` bar",
        any(l.startswith("━━ CLUSTER qa ") and "· connected" in l for l in plain.splitlines()),
        plain,
    )
    check(
        "ns-hierarchy: each namespace hangs off the cluster on its OWN `━━ NAMESPACE … ━━` branch (ours + llmb-cache)",
        any("─ ━━ NAMESPACE ours" in l for l in plain.splitlines())
        and any("─ ━━ NAMESPACE llmb-cache" in l for l in plain.splitlines()),
        plain,
    )

    ours = "\n".join(_ns_sub(plain, "ours"))
    cache = "\n".join(_ns_sub(plain, "llmb-cache"))
    # (1) INSTALLED renders the staged CELL; the cached MODEL is a MODELS row (a model is not a cell, and
    #     the two were counted together — `3 installed · 3 ready` over one model and two volumes).
    check(
        "ns-hierarchy: `ours` INSTALLED shows the staged cell, MODELS shows the cached model",
        "INSTALLED" in ours and "qa-agg" in ours and "MODELS" in ours and "nemotron-ultra" in ours,
        ours,
    )
    check(
        "ns-hierarchy: a MODEL never appears in the INSTALLED (cells) section",
        "model-cache:" not in ours and "model:" not in ours,
        ours,
    )
    # (2) RUN / SERVER renders the PARKED server holding GPUs, with the `no active bench` hint.
    check(
        "ns-hierarchy: `ours` RUN / SERVER shows the ● PARKED server (8g held · server up, no active bench)",
        "RUN / SERVER" in ours
        and "● PARKED" in ours
        and "qa-agg-server" in ours
        and "server up, no active bench" in ours
        and "8 GPUs" in ours,
        ours,
    )
    # (3) both sections nest under the right ns, INSTALLED above RUN / SERVER.
    check(
        "ns-hierarchy: under `ours`, INSTALLED sits ABOVE RUN / SERVER (both nested beneath the ns bar)",
        "INSTALLED" in ours and "RUN / SERVER" in ours and ours.index("INSTALLED") < ours.index("RUN / SERVER"),
        ours,
    )
    # (4) a managed namespace with ONLY a cache (no runs) STILL shows its own sections + `no active runs`.
    check(
        "ns-hierarchy: cache-only `llmb-cache` still shows MODELS (glm5) + INSTALLED + `no active runs`",
        "INSTALLED" in cache
        and "MODELS" in cache
        and "glm5" in cache
        and "RUN / SERVER" in cache
        and "no active runs" in cache,
        cache,
    )

    # (5) TREE GUIDES — the box-drawing structure threads cluster → namespace → section → row so the nesting
    #     reads at a glance. Namespaces hang off the cluster (└─ for the LAST one), sections + rows hang off
    #     with ├─/└─, and grey │ rails DOUBLE as the breathing-room separators between blocks.
    pl = plain.splitlines()
    ns_bars = [l for l in pl if "━━ NAMESPACE " in l and l.lstrip().startswith(("├─", "└─"))]  # exclude the legend line
    check(
        "ns-hierarchy (tree): the LAST namespace hangs off `└─` and a non-last off `├─`",
        len(ns_bars) == 2
        and any(l.lstrip().startswith("├─") for l in ns_bars)
        and any(l.lstrip().startswith("└─") for l in ns_bars),
        str(ns_bars),
    )
    check(
        "ns-hierarchy (tree): section headers hang off the namespace on ├─/└─ branches (INSTALLED · RUN / SERVER)",
        any(("├─ INSTALLED" in l or "└─ INSTALLED" in l) for l in pl)
        and any(("├─ RUN / SERVER" in l or "└─ RUN / SERVER" in l) for l in pl),
        ours,
    )
    check(
        "ns-hierarchy (tree): rows hang off their section on ├─/└─ leaves (installed cell + run/server row)",
        any(("├─ ✓" in l or "└─ ✓" in l) for l in pl)  # an installed cell leaf
        and any(("├─ ● " in l or "└─ ● " in l or "└─ " in l and "svc" in l) for l in pl),
        ours,
    )

    def _is_rail(l):
        return l != "" and "│" in l and set(l) <= {"│", " "}  # a pure guide rail (│ + spaces), not blank, not content

    _ob = _ns_sub(plain, "ours")
    _bar_i = next(n for n, l in enumerate(pl) if "━━ NAMESPACE ours" in l)
    _inst_i = next(n for n, l in enumerate(_ob) if "INSTALLED" in l)
    _run_i = next(n for n, l in enumerate(_ob) if "RUN / SERVER" in l)
    check(
        "ns-hierarchy (tree): a │ rail precedes each namespace bar + sits above INSTALLED (breathing room via guides)",
        _is_rail(pl[_bar_i - 1]) and _is_rail(_ob[_inst_i - 1]),
        f"{pl[_bar_i-1]!r} / {_ob[_inst_i-1]!r}",
    )
    check(
        "ns-hierarchy (tree): a │ rail separates INSTALLED from RUN / SERVER (distinct sub-blocks)",
        any(_is_rail(_ob[n]) for n in range(_inst_i + 1, _run_i)),
        str(_ob[_inst_i : _run_i + 1]),
    )

    # (6) STRUCTURAL COLOR — a palette DISJOINT from status hues (green 32 / red 31 / yellow 33 / cyan 36).
    #     READABILITY: the structural color wraps ONLY the `━━` rule segments; the header TEXT (CLUSTER/NAMESPACE
    #     + name + summary) stays default-foreground so the words read on any background.
    #       cluster rule = bold BLUE (1;34) · namespace rule = bold MAGENTA (1;35) · section labels = bold (1) ·
    #       tree guides = GREY (90).
    col = fr.render([e], now, wide=False, gpu_only=False, color=True, stages=True)
    ESC = "\033"
    cbar = next(l for l in col.splitlines() if "CLUSTER qa" in l)
    nbar = next(l for l in col.splitlines() if "NAMESPACE ours" in l)
    check(
        "ns-hierarchy (color): CLUSTER rule bold BLUE (1;34) wraps ONLY the ━━ segments (text stays plain) — no status hue",
        cbar.startswith(ESC + "[1;34m━")
        and all(seg[:1] == "━" for seg in cbar.split(ESC + "[1;34m")[1:])
        and not any(c in cbar for c in (ESC + "[31m", ESC + "[32m", ESC + "[33m")),
        repr(cbar),
    )
    check(
        "ns-hierarchy (color): NAMESPACE rule bold MAGENTA (1;35) wraps ONLY the ━━ segments (text plain) + GREY (90) guides",
        all(seg[:1] == "━" for seg in nbar.split(ESC + "[1;35m")[1:])
        and (ESC + "[1;35m") in nbar
        and (ESC + "[90m") in nbar,
        repr(nbar),
    )
    check(
        "ns-hierarchy (color): SECTION labels are bold default-fg (readable, not tinted) + guides GREY (90) present",
        (ESC + "[1mINSTALLED") in col and (ESC + "[1mRUN / SERVER") in col and (ESC + "[90m") in col,
        "",
    )


def _unit_sweep_progress():
    """The SWEEP progress dot-bar (● done · ◐ current · ○ pending · done/total) + its TRUTHFUL completed-rung
    derivation. The total comes from the rung list fleet already resolves; the done count from signals the pure
    kubectl renderer already sees — a ✓ done Job ran every rung, a Job carrying COMPLETED_RUNGS names the rungs
    finished — and a LIVE sweep with no such signal degrades to the plain rung list (never a fabricated bar,
    since live per-rung state lives in the control-PVC heartbeat / concurrency_* dirs fleet never reads).
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    # ── dot-bar shapes ──
    check(
        "sweep-bar: a partial live sweep → ● done · ◐ current · ○ pending + done/total",
        fr.sweep_dotbar(2, 5, running=True) == "●●◐○○ 2/5",
        fr.sweep_dotbar(2, 5, running=True),
    )
    check(
        "sweep-bar: a settled full sweep → every rung filled, no current marker",
        fr.sweep_dotbar(5, 5, running=False) == "●●●●● 5/5",
        fr.sweep_dotbar(5, 5),
    )
    check(
        "sweep-bar: a just-started live sweep → first rung current (◐), the rest pending",
        fr.sweep_dotbar(0, 3, running=True) == "◐○○ 0/3",
        fr.sweep_dotbar(0, 3, running=True),
    )
    check(
        "sweep-bar: unknown total → empty (no bar)",
        fr.sweep_dotbar(0, 0) == "",
        repr(fr.sweep_dotbar(0, 0)),
    )
    check(
        "sweep-bar: a wide sweep (>8 rungs) collapses to a fixed 8-segment proportional bar (never blows up width)",
        len(fr.sweep_dotbar(6, 12).split(" ")[0]) == 8 and fr.sweep_dotbar(6, 12).endswith(" 6/12"),
        fr.sweep_dotbar(6, 12),
    )
    # ── _sweep_field truthful derivation (done, total, status, live) ──
    check(
        "sweep-field: a ✓ done run → the full dot-bar (every rung ran)",
        fr._sweep_field("16·32·64", 0, 0, fr.COMPLETE) == "●●● 3/3",
        fr._sweep_field("16·32·64", 0, 0, fr.COMPLETE),
    )
    check(
        "sweep-field: a RUNNING run WITH a progress signal → a partial live bar (◐ marks the current rung)",
        fr._sweep_field("16·32·64·128", 2, 4, fr.RUNNING, live=True) == "●●◐○ 2/4",
        fr._sweep_field("16·32·64·128", 2, 4, fr.RUNNING, live=True),
    )
    check(
        "sweep-field: a live run with NO progress signal degrades to the plain rung list (no fabrication)",
        fr._sweep_field("16·32·64", 0, 3, fr.RUNNING, live=False) == "16·32·64",
        fr._sweep_field("16·32·64", 0, 3, fr.RUNNING, live=False),
    )
    check(
        "sweep-field: adaptive / unknown rungs pass through untouched (no bar)",
        fr._sweep_field("adaptive", 0, 0, fr.RUNNING, live=True) == "adaptive"
        and fr._sweep_field("?", 0, 0, fr.RUNNING, live=True) == "?",
        "",
    )

    # ── resolve_sweep_progress: LIVE annotation wins; else COMPLETED_RUNGS env; else no signal ──
    def _jobwith(*, ann=None, env=None):
        j = job("s-bench", recipe="s", active=1, concurrencies="16 32 64 128 256")
        if env is not None:
            j["spec"]["template"]["spec"]["containers"][0]["env"].append({"name": "COMPLETED_RUNGS", "value": env})
        if ann is not None:
            j["metadata"]["annotations"] = ann
        return j

    live_j = _jobwith(ann={"llmb.nvidia.com/completed-rungs": "3", "llmb.nvidia.com/total-rungs": "5"})
    check(
        "sweep-progress: the LIVE completed-rungs annotation → (done=3, total=5, live) — rung-by-rung real time",
        fr.resolve_sweep_progress(live_j, "16·32·64·128·256") == (3, 5, True),
        str(fr.resolve_sweep_progress(live_j, "16·32·64·128·256")),
    )
    check(
        "sweep-progress: completed-rungs=0 annotation still counts as a LIVE signal (bar shows ◐ on rung 1)",
        fr.resolve_sweep_progress(_jobwith(ann={"llmb.nvidia.com/completed-rungs": "0"}), "16·32·64·128·256")
        == (0, 5, True),
        "",
    )
    check(
        "sweep-progress: no annotation → falls back to the COMPLETED_RUNGS env (resume signal)",
        fr.resolve_sweep_progress(_jobwith(env="16 32"), "16·32·64·128·256") == (2, 5, True),
        "",
    )
    check(
        "sweep-progress: neither annotation nor env → no signal (done=0, total from rung count, live=False)",
        fr.resolve_sweep_progress(_jobwith(), "16·32·64·128·256") == (0, 5, False),
        "",
    )

    # ── end-to-end via build_cluster + the stages renderer: a LIVE sweep annotated 3/5 shows the real-time bar ──
    now = fr._parse_ts(NOW)
    jd = job(
        "sweepy-bench-r9",
        recipe="sweepy",
        cell="sweepy",
        run_id="r9",
        active=1,
        concurrencies="16 32 64 128 256",
    )
    jd["metadata"]["annotations"] = {
        "llmb.nvidia.com/completed-rungs": "3",
        "llmb.nvidia.com/total-rungs": "5",
    }
    jp = pod("sweepy-bench-r9-x", job="sweepy-bench-r9", gpus=8, run_id="r9")
    e = fr.build_cluster("sw", "ctxS", "ours", items([jp]), items([]), items([jd]), None, None, now=now)
    e["gpu_type"], e["profiles"], e["stale"], e["readiness"], e["installs"] = (
        "B200",
        1,
        "",
        None,
        {},
    )
    plain = ANSI.sub("", fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True))
    check(
        "sweep-bar (end-to-end): a LIVE sweep annotated 3/5 shows the real-time progress dot-bar (●●●◐○ 3/5)",
        "●●●◐○ 3/5" in plain,
        plain,
    )


def _ns_sub(text: str, ns: str) -> list:
    """The lines of a single `━━ NAMESPACE <ns> ━━` sub-block: from its namespace bar (which hangs off a ├─/└─
    tree branch) up to the next `━━ …` bar (the next namespace or cluster) — so per-namespace INSTALLED/RUN
    assertions don't bleed across namespaces."""
    lines = text.splitlines()
    start = next(
        (n for n, l in enumerate(lines) if f"━━ NAMESPACE {ns} " in l or f"━━ NAMESPACE {ns}  " in l),
        None,
    )
    if start is None:
        return []
    end = next(
        (n for n, l in enumerate(lines[start + 1 :], start + 1) if "━━" in l),
        len(lines),
    )
    return lines[start:end]


def _unit_capacity_branch():
    """The CLUSTER-level CAPACITY sub-branch: the GPU product (moved OFF the cluster bar) + the truthful
    launchable unit — total GPU nodes, fully-FREE whole nodes, and the biggest single run schedulable right
    now. Pure over `node_capacity()` data fleet already fetched (no new cluster call). Asserts the happy line,
    GPUs stay SPELLED OUT, singular/plural, every graceful-degradation path (never a bogus 0), and the
    zero-free-node whole-node blocker hint."""
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    def _e(gtype="GB300", *, tn=18, fn=18, bf=4, on=0, nodes=True):
        nc = {"total_nodes": tn, "free_nodes": fn, "biggest_free": bf, "ours_nodes": on} if nodes else None
        return {"gpu_type": gtype, "nodes": nc}

    # THE FREE COUNT CARRIES ITS UNIT. `18 nodes · 18 free · biggest free 4 GPUs` put a bare number next to a
    # clause whose number IS GPUs, so `18 free` read as free GPUs. `_free_note` already rendered the same
    # fact as `18/18 nodes free`; two renderers of one fact are now one phrasing.
    check(
        "capacity: happy line = GPU product · free/total NODES (unit stated) · biggest launchable run",
        fr._capacity_line(_e()) == "GB300 · 18/18 GPU nodes free · biggest free 4 GPUs",
        fr._capacity_line(_e()),
    )
    check(
        "capacity: the free count is never a bare unitless number beside a GPU count",
        " free ·" not in fr._capacity_line(_e()).replace("nodes free ·", "")
        and "nodes free" in fr._capacity_line(_e()),
        fr._capacity_line(_e()),
    )
    # …and it names WHICH nodes. node_capacity filters to `allocatable nvidia.com/gpu > 0`, so on the
    # reported 11-node cluster this read `2/2 nodes free` on the same frame as the scheduler's own
    # `0/11 nodes are available` — two denominators wearing one word, on the screen used to decide
    # whether a pod can be placed.
    check(
        "capacity: the node count says GPU nodes, the set it actually counts",
        "GPU node" in fr._capacity_line(_e())
        and "GPU node"
        in fr._free_note(
            {
                "nodes": {
                    "total_nodes": 2,
                    "free_nodes": 2,
                    "biggest_free": 8,
                    "ours_nodes": 0,
                }
            }
        ),
        "",
    )
    check(
        "capacity: GPUs stay SPELLED OUT (no bare `4g`) and singularize correctly (1 node · 1 GPU)",
        fr._capacity_line(_e("B200", tn=1, fn=1, bf=1)) == "B200 · 1/1 GPU node free · biggest free 1 GPU",
        fr._capacity_line(_e("B200", tn=1, fn=1, bf=1)),
    )
    # ── graceful degradation — show what we KNOW, never a fabricated 0 ──
    check(
        "capacity: nodes known but pods -A unreadable → `capacity unknown (no pod read)`, never a bogus 0 free",
        fr._capacity_line(_e(fn=None, bf=None)) == "GB300 · 18 GPU nodes · capacity unknown (no pod read)",
        fr._capacity_line(_e(fn=None, bf=None)),
    )
    check(
        "capacity: no node list at all (no cluster-scope RBAC) → `node capacity unknown (no node read)`",
        fr._capacity_line(_e(nodes=False)) == "GB300 · node capacity unknown (no node read)",
        fr._capacity_line(_e(nodes=False)),
    )
    check(
        "capacity: --fast (capacity reads skipped) says so instead of implying an empty cluster",
        fr._capacity_line(_e(nodes=False), fast=True) == "GB300 · capacity skipped (--fast)",
        fr._capacity_line(_e(nodes=False), fast=True),
    )
    check(
        "capacity: an unknown GPU product just drops the leading token (no empty separator)",
        fr._capacity_line(_e("", tn=4, fn=2, bf=8)) == "2/4 GPU nodes free · biggest free 8 GPUs",
        fr._capacity_line(_e("", tn=4, fn=2, bf=8)),
    )
    # ── the whole-node blocker hint (cells with requires.gpu.whole_node need a FULLY-free node) ──
    check(
        "capacity: ZERO fully-free nodes → a whole-node blocker hint (pre-empts a confusing preflight failure)",
        "whole-node cells cannot schedule" in fr._whole_node_note(_e(fn=0, bf=2)),
        fr._whole_node_note(_e(fn=0, bf=2)),
    )
    check(
        "capacity: the blocker hint stays OFF the happy path (free nodes exist → no hint)",
        fr._whole_node_note(_e(fn=1)) == "",
        fr._whole_node_note(_e(fn=1)),
    )
    check(
        "capacity: no hint when occupancy is UNKNOWN (never claim a blocker we cannot see)",
        fr._whole_node_note(_e(fn=None, bf=None)) == ""
        and fr._whole_node_note(_e(nodes=False)) == ""
        and fr._whole_node_note(_e(fn=0), fast=True) == "",
        "",
    )


def _unit_model_load_queue():
    """The MODEL-LOAD queue: runs sharing one model-cache PVC serialize their checkpoint load behind a Lease
    (three concurrent ~500GB loads off one FSx measured 7.2x slower per shard). Fleet must make the queue
    legible WITHOUT inventing one. Covers: holder-only, holder+queue, a STALE lease (no phantom waiters),
    unreadable Leases (RBAC → say nothing, never imply "no queue"), and nothing-loading (line absent).
    Also asserts a queued run re-badges from the misleading `● LOADING` to `◔ QUEUED` — the actual bug.
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr
    from datetime import timedelta

    now = fr._parse_ts(NOW)

    def lease(pvc, holder, ns="ours", renew_age=10, dur=60):
        t = (now - timedelta(seconds=renew_age)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "metadata": {
                "name": f"llmb-modelload-{pvc}",
                "namespace": ns,
                "labels": {
                    "llmb.nvidia.com/managed": "true",
                    "llmb.nvidia.com/model-cache": pvc,
                },
            },
            "spec": {
                "holderIdentity": holder,
                "renewTime": t,
                "leaseDurationSeconds": dur,
                "acquireTime": (now - timedelta(seconds=900)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        }

    def waiter(run, pvc, ns="ours"):
        return {
            "kind": "Job",
            "metadata": {
                "name": f"{run}-runowner",
                "namespace": ns,
                "labels": {"llmb.nvidia.com/run-id": run},
                "annotations": {
                    "llmb.nvidia.com/model-load-wait": pvc,
                    "llmb.nvidia.com/model-load-since": "2026-07-19T11:40:00Z",
                },
            },
            "status": {},
        }

    # (1) holder only — a line, no queue segment
    q = fr.discover_model_load({"items": [lease("model-cache", "c16")]}, [], now)
    check(
        "model-load: a live holder alone renders one line, no phantom queue",
        len(q) == 1
        and q[0]["holder"] == "c16"
        and q[0]["waiters"] == []
        and "queued" not in fr._model_load_line(q[0], now),
        str(q),
    )
    # (2) holder + queue — waiters listed, in queue order, and exposed for row re-badging
    q = fr.discover_model_load(
        {"items": [lease("model-cache", "c16")]},
        [waiter("c32", "model-cache"), waiter("c64", "model-cache")],
        now,
    )
    line = fr._model_load_line(q[0], now)
    check(
        "model-load: holder + queue lists both waiters (`⏳ queued: c32, c64`)",
        "● c16 loading" in line and "⏳ queued: c32, c64" in line,
        line,
    )
    check(
        "model-load: queued runs are exposed so their RUN/SERVER row can be re-badged",
        fr.model_load_waiting_runs(q) == {"c32", "c64"},
        str(fr.model_load_waiting_runs(q)),
    )
    # (3) STALE lease — flagged as stale, and its annotated waiters are NOT reported as a queue
    q = fr.discover_model_load(
        {"items": [lease("model-cache", "c16", renew_age=9999)]},
        [waiter("c32", "model-cache")],
        now,
    )
    check(
        "model-load: an EXPIRED lease reads `(stale lock)` and reports NO holder",
        len(q) == 1 and q[0]["stale"] and not q[0]["holder"] and "stale lock" in fr._model_load_line(q[0], now),
        str(q),
    )
    check(
        "model-load: a stale lock never invents phantom waiters (annotations without a live holder)",
        q[0]["waiters"] == [] and fr.model_load_waiting_runs(q) == set(),
        str(q),
    )
    # (4) Leases unreadable (RBAC) — say NOTHING rather than implying "no queue"
    check(
        "model-load: unreadable Leases (RBAC) → no rows at all, never a false 'nothing loading'",
        fr.discover_model_load(None, [waiter("c32", "model-cache")], now) == [],
        "",
    )
    # (5) nothing loading — the line stays entirely off the happy path
    check(
        "model-load: no lock → no rows (the MODEL LOAD branch is absent on the happy path)",
        fr.discover_model_load({"items": []}, [], now) == [],
        "",
    )

    # (6) END-TO-END: a queued run re-badges ◔ QUEUED, and the MODEL LOAD branch renders beside CAPACITY.
    ns = "llmb-serving"
    lp = pod(
        "c32-server-abc",
        app="c32-server",
        gpus=4,
        ns=ns,
        node="n1",
        start="2026-07-19T11:52:00Z",
    )  # Running-but-not-Ready → would read LOADING
    lp["status"]["conditions"] = [{"type": "Ready", "status": "False"}]
    e = fr.build_cluster(
        "qa",
        "ctxA",
        "ours",
        items([]),
        items([]),
        items([]),
        None,
        items([lp]),
        now=now,
        all_deploys=[],
        all_jobs=[waiter("c32", "model-cache", ns=ns)],
        leases_j={"items": [lease("model-cache", "c16", ns=ns)]},
    )
    e["gpu_type"], e["profiles"], e["stale"], e["readiness"], e["installs"] = (
        "GB300",
        1,
        "",
        None,
        {},
    )
    plain = ANSI.sub("", fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True))
    check(
        "model-load (e2e): the waiting run reads ◔ QUEUED, NOT `● LOADING server warming` (the bug)",
        "◔ QUEUED" in plain and "waiting for model-load slot" in plain and "server warming" not in plain,
        plain,
    )
    check(
        "model-load (e2e): a MODEL LOAD branch renders at CLUSTER level with the holder + queue",
        any(l.lstrip("│├└─ ").startswith("MODEL LOAD") for l in plain.splitlines())
        and "● c16 loading" in plain
        and "⏳ queued: c32" in plain,
        plain,
    )
    # and it is ABSENT when nothing is loading
    e2 = fr.build_cluster(
        "qa",
        "ctxA",
        "ours",
        items([]),
        items([]),
        items([]),
        None,
        items([lp]),
        now=now,
        all_deploys=[],
        all_jobs=[],
        leases_j={"items": []},
    )
    e2["gpu_type"], e2["profiles"], e2["stale"], e2["readiness"], e2["installs"] = (
        "GB300",
        1,
        "",
        None,
        {},
    )
    plain2 = ANSI.sub("", fr.render([e2], now, wide=False, gpu_only=False, color=False, stages=True))
    check(
        "model-load (e2e): no lock → no MODEL LOAD line anywhere (stays off the happy path)",
        "MODEL LOAD" not in plain2,
        plain2,
    )


def _unit_model_caches():
    """THE STORAGE half: what each PVC IS and whether it is USABLE. (Which MODELS are on it is the ledger's
    job — see _unit_model_ledger.) Sourced from the CLUSTER (PVCs + download Jobs), never from local .state
    stamps — a model downloaded from another worktree, by a colleague, or before a fresh clone must still be
    visible. Also covers the honest-state contract: a Bound but unvouched-for PVC must NOT read as
    downloaded, and an unreadable PVC list must say NOTHING rather than "not downloaded".
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    def pvc(name, ns="ours", phase="Bound", size="721Gi", labels=None):
        return {
            "metadata": {"name": name, "namespace": ns, "labels": labels or {}},
            "spec": {"resources": {"requests": {"storage": size}}},
            "status": {"phase": phase, "capacity": {"storage": size}},
        }

    def dl(claim, model, rev, ns="ours", ok=True, running=False, failed=False):
        return {
            "kind": "Job",
            "metadata": {
                "name": f"llmb-download-{model}",
                "namespace": ns,
                "labels": {
                    "app.kubernetes.io/managed-by": "llmb-recipe",
                    "llmb.nvidia.com/component": "model-download",
                    "llmb.nvidia.com/model-name": model,
                    "llmb.nvidia.com/model-revision": rev,
                },
            },
            "spec": {"template": {"spec": {"volumes": [{"name": "c", "persistentVolumeClaim": {"claimName": claim}}]}}},
            "status": {
                "succeeded": 1 if ok else 0,
                "failed": 1 if failed else 0,
                "active": 1 if running else 0,
            },
        }

    # ── A CLAIM ROW IS ABOUT THE VOLUME. It is named `claim: <pvc>` and NEVER carries a model name — the
    #    branch that let a PVC name occupy the model slot (`model = st_model or info and info["model"] or nm`)
    #    is the reported defect and is gone. ──
    r = fr.discover_model_caches(
        {"items": [pvc("glm5-fp8-model-cache")]},
        [dl("glm5-fp8-model-cache", "glm5-fp8", "abc123def456")],
    )
    check(
        "claim: a cache row names the CLAIM, carries its size, and never claims to be a model",
        len(r) == 1
        and r[0]["kind"] == "cache"
        and r[0]["name"] == "glm5-fp8-model-cache"
        and r[0]["size"] == "721Gi"
        and "@" not in r[0]["name"],
        str(r),
    )
    r = fr.discover_model_caches({"items": [pvc("nemotron-ultra-hf-cache", size="900Gi")]}, [])
    check(
        "claim: a Bound PVC with NO download record reads ⚠ 'not attributed', never ✓ downloaded",
        len(r) == 1 and r[0]["state"] == "warn" and "not attributed" in r[0]["why"],
        str(r),
    )
    r = fr.discover_model_caches({"items": [pvc("x")]}, [dl("x", "X", "r1", ok=False, running=True)])
    # `downloading` is load-bearing vocabulary: install.py's own panel reads this verdict verbatim
    # (cell_cluster_state) so the two tools can never disagree about the same PVC.
    check(
        "claim: an in-flight download reads ⚠ downloading (never ✓, never a bare 'unverified')",
        r[0]["state"] == "warn" and "downloading" in r[0]["why"] and "download Job in flight" in r[0]["why"],
        str(r),
    )
    r = fr.discover_model_caches({"items": [pvc("y")]}, [dl("y", "Y", "r2", ok=False, failed=True)])
    check(
        "claim: a FAILED download reads ✗ (never silently absent)",
        r[0]["state"] == "failed",
        str(r),
    )
    r = fr.discover_model_caches({"items": [pvc("z", phase="Pending")]}, [])
    check(
        "claim: an unbound PVC reads ✗ 'cannot serve weights'",
        r[0]["state"] == "failed" and "Pending" in r[0]["why"],
        str(r),
    )
    check(
        "claim: an unreadable PVC list (RBAC) yields NO rows — never a false 'not downloaded'",
        fr.discover_model_caches(None, [dl("q", "Q", "r")]) == [],
        "",
    )

    # ── STAMP PROVENANCE — the ONLY gate on "stamped by a download Job". `download-complete=true` alone was
    #    the entire former basis for the words "verified by PVC stamp"; a label is not an inspection. ──
    PIN = "abc123def4560000000000000000000000000000"
    S = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-name": "glm5-fp8",
        "llmb.nvidia.com/model-revision": "abc123def456",
    }
    check(
        "provenance: model-name + a 12-char revision equal to pinned[:12] → 'job' (the ONLY ✓ source)",
        fr._stamp_provenance(S, PIN) == "job",
        fr._stamp_provenance(S, PIN),
    )
    check(
        "provenance: NO model-name → 'hand' — the template writes it in the SAME patch, so a Job did not",
        fr._stamp_provenance({k: v for k, v in S.items() if k != "llmb.nvidia.com/model-name"}, PIN) == "hand",
        "",
    )
    check(
        "provenance: a 40-char revision → 'hand' — the template writes model_revision[:12], never 40",
        fr._stamp_provenance({**S, "llmb.nvidia.com/model-revision": PIN}, PIN) == "hand",
        "",
    )
    check(
        "provenance: Job-SHAPED but no pin to check against → 'unknown', never 'job'",
        fr._stamp_provenance(S, "") == "unknown",
        fr._stamp_provenance(S, ""),
    )
    check(
        "provenance: an unstamped PVC is 'unknown' (nothing to attribute), never 'hand'",
        fr._stamp_provenance({}, PIN) == "unknown",
        "",
    )

    # ── _short_rev — 12 chars, ALWAYS; a label that names a different revision is FLAGGED, not reprinted ──
    check(
        "short-rev: a 40-char label renders at 12 and reports no anomaly when it agrees with the pin",
        fr._short_rev(PIN, PIN) == ("abc123def456", "") and fr._short_rev("abc123def456", PIN) == ("abc123def456", ""),
        str(fr._short_rev(PIN, PIN)),
    )
    check(
        "short-rev: a label that is neither pinned[:12] nor a prefix of the pin → WRONG REVISION",
        fr._short_rev("deadbeefcafe", PIN) == ("deadbeefcafe", "WRONG REVISION"),
        str(fr._short_rev("deadbeefcafe", PIN)),
    )
    check(
        "short-rev: no pin → the label truncated to 12, and NEVER a WRONG-REVISION accusation",
        fr._short_rev(PIN, "") == ("abc123def456", "") and fr._short_rev("ab", "") == ("ab", ""),
        "",
    )
    check(
        "short-rev: an ABSENT label is not a wrong one — the pin is shown, unflagged",
        fr._short_rev("", PIN) == ("abc123def456", ""),
        str(fr._short_rev("", PIN)),
    )
    check(
        "short-rev: EVERY output is at most 12 chars — this is the table's alignment invariant",
        all(len(fr._short_rev(v, p)[0]) <= 12 for v in ("", "ab", "abc123def456", PIN, "z" * 80) for p in ("", PIN)),
        "",
    )

    # a Job-provenance stamp makes the CLAIM usable-and-attributed; a hand stamp does not
    r = fr.discover_model_caches({"items": [pvc("c", labels=S)]}, [], pins={"glm5-fp8": PIN})
    check(
        "claim: a Job-stamped PVC reads ✓ even with the download Job GC'd (the actual failure mode)",
        r[0]["state"] == "ready" and "stamped by a download Job" in r[0]["why"],
        str(r),
    )
    hand = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-revision": PIN,
    }
    r = fr.discover_model_caches({"items": [pvc("nemotron-ultra-nvfp4-cache", labels=hand)]}, [])
    check(
        "claim: THE REPORTED ROW — stamped, no model-name → ⚠, named as a CLAIM, says the stamp names none",
        r[0]["state"] == "warn"
        and r[0]["name"] == "nemotron-ultra-nvfp4-cache"
        and "does NOT name a model" in r[0]["why"]
        and "verified" not in r[0]["why"],
        str(r),
    )
    # THE GREEN-SIGNAL-THAT-ISN'T GUARD: a stamp must never promote past a failure.
    r = fr.discover_model_caches(
        {"items": [pvc("c", labels=S)]},
        [dl("c", "glm5-fp8", "zzz", ok=False, failed=True)],
        pins={"glm5-fp8": PIN},
    )
    check(
        "claim: a stamp NEVER hides a FAILED download — state stays ✗, prior revision noted",
        r[0]["state"] == "failed"
        and "FAILED" in r[0]["why"]
        and "previously stamped glm5-fp8 @abc123def456" in r[0]["why"],
        str(r),
    )
    # BACKWARD COMPATIBILITY: pre-existing populated PVCs carry no labels and must NOT regress to ✗.
    r = fr.discover_model_caches({"items": [pvc("legacy-cache", size="900Gi")]}, [])
    check(
        "claim: an UNLABELLED pre-existing PVC still reads ⚠ (never ✗) and never ✓",
        r[0]["state"] == "warn" and "not attributed" in r[0]["why"],
        str(r),
    )
    check(
        "claim: no row ever inlines a kubectl mutation (80 copies was the unreadable half of the 82 rows)",
        "kubectl" not in r[0]["why"],
        str(r),
    )

    # END-TO-END: the reported failure — a ready cluster, no runs, two profiles collapsed onto one ns.
    # BOTH caches present in the namespace must be accounted for (the `(N profiles)` collapse hides
    # MODEL_CACHE_PVC), and the sections must render even with nothing blocked and nothing running.
    ns = "ours"
    j = dl("glm5-fp8-model-cache", "glm5-fp8", "abc123def456", ns=ns)
    e = fr.build_cluster(
        "example-gpu-cluster",
        "ctxA",
        ns,
        items([]),
        items([]),
        items([j]),
        items([node("n1", 8)]),
        items([]),
        now=fr._parse_ts(NOW),
        all_deploys=[],
        all_jobs=[j],
        pvcs_j={
            "items": [
                pvc("glm5-fp8-model-cache", ns=ns),
                pvc("nemotron-ultra-hf-cache", ns=ns, size="900Gi"),
            ]
        },
    )
    e["gpu_type"], e["profiles"], e["stale"], e["readiness"], e["installs"] = (
        "B200",
        2,
        "",
        None,
        {},
    )
    e["profile_envs"] = [
        {"MODEL_CACHE_PVC": "glm5-fp8-model-cache", "NAMESPACE": ns},
        {"MODEL_CACHE_PVC": "nemotron-ultra-hf-cache", "NAMESPACE": ns},
    ]
    plain = ANSI.sub(
        "",
        fr.render([e], fr._parse_ts(NOW), wide=False, gpu_only=False, color=False, stages=True),
    )
    check(
        "model-cache (e2e): a downloaded model is VISIBLE on an idle, unblocked cluster with no runs",
        "glm5-fp8" in plain and "download Job Complete" in plain and "MODELS" in plain,
        plain,
    )
    check(
        "model-cache (e2e): BOTH profiles' claims are matched despite the `(2 profiles)` collapse — the "
        "ledger takes the UNION, so the second profile's MODEL_CACHE_PVC is not hidden",
        "glm5-fp8-model-cache" in plain and "nemotron-ultra-hf-cache" in plain,
        plain,
    )
    check(
        "model-cache (e2e): one model, ONE row (no Job-derived duplicate beside the claim-derived one)",
        len([l for l in plain.splitlines() if " glm5-fp8 " in l and ("✓" in l or "~" in l)]) == 1,
        plain,
    )

    # ── THE B200 BUG: "INSTALLED — nothing installed —" over a namespace holding 20 PVCs (a 721GB GLM-5-FP8
    #    cache among them). Nothing was filtering the unstamped PVCs out — the PVC READ never landed
    #    (--fast skips it; an RBAC-Forbidden or timed-out `get pvc -A` writes no file), and a MISSING read was
    #    rendered as the positive claim that the namespace is empty. Absence of evidence, sold as evidence of
    #    absence, on the one screen an operator uses to decide whether to re-download 700GB. ──
    check(
        "inventory-honesty: an unstamped, unvouched-for PVC still RENDERS (⚠ unverified) — never omitted",
        [r["state"] for r in fr.discover_model_caches({"items": [pvc("bare-cache")]}, [])] == ["warn"],
        "",
    )
    e_ok = fr.build_cluster(
        "c",
        "ctxA",
        ns,
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=fr._parse_ts(NOW),
        pvcs_j={"items": []},
    )
    e_no = fr.build_cluster(
        "c",
        "ctxA",
        ns,
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=fr._parse_ts(NOW),
        pvcs_j=None,
    )
    check(
        "inventory-honesty: build_cluster records WHETHER the PVC read landed (empty list ≠ no read)",
        e_ok["inventory_unavailable"] is False and e_no["inventory_unavailable"] is True,
        f"{e_ok['inventory_unavailable']}/{e_no['inventory_unavailable']}",
    )
    check(
        "inventory-honesty: inventory_note is '' on a good read, and names the CAUSE otherwise",
        fr.inventory_note(False, False) == ""
        and "--fast" in fr.inventory_note(True, True)
        and "RBAC" in fr.inventory_note(True, False),
        "",
    )

    def _inst(inv, unavail):
        out = []
        fr._tree_installed(
            out,
            {
                "inv": inv,
                "counts": {
                    "total": len(inv),
                    "ready": len(inv),
                    "warn": 0,
                    "failed": 0,
                },
                "source": "live",
                "cluster": "c",
            },
            fr.Paint(False),
            lambda s: s,
            "",
            "└─ ",
            "   ",
            unavail=unavail,
        )
        return "\n".join(out)

    row = [
        {
            "cell": "cache:ours:m",
            "disp": "model: m",
            "ns": ns,
            "state": "ready",
            "glyph": "✓",
            "why": "downloaded",
            "fix_kind": "",
            "source": "live",
        }
    ]
    unk = _inst([], fr.inventory_note(True, False))
    check(
        "inventory-honesty: an UNREAD inventory renders 'inventory UNKNOWN' + why — NEVER 'nothing installed'",
        "inventory UNKNOWN" in unk and "nothing installed" not in unk and "RBAC" in unk,
        unk,
    )
    check(
        "inventory-honesty: a genuinely empty ns (read LANDED) still reads '— nothing installed —'",
        "— nothing installed —" in _inst([], ""),
        _inst([], ""),
    )
    part = _inst(row, fr.inventory_note(True, True))
    check(
        "inventory-honesty: a PARTIAL list (cells listed, caches unread) is flagged ⚠ PARTIAL, not complete",
        "⚠ PARTIAL" in part and "model: m" in part,
        part,
    )
    check(
        "inventory-honesty: a complete list carries NO partial/unknown caveat",
        "PARTIAL" not in _inst(row, "") and "UNKNOWN" not in _inst(row, ""),
        _inst(row, ""),
    )

    # E2E: --fast skips the PVC read, so the pane must NOT claim the namespace is empty (this is the exact
    # shape of the reported B200 block: a ns present because of its RUNS, with an empty INSTALLED section).
    e_fast = fr.build_cluster(
        "example-gpu-cluster",
        "ctxA",
        ns,
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=fr._parse_ts(NOW),
        pvcs_j=None,
    )
    e_fast["gpu_type"], e_fast["profiles"], e_fast["stale"] = "B200", 2, ""
    e_fast["readiness"], e_fast["installs"] = None, {"llm-perf/x": {"steps": {}}}
    pf = ANSI.sub(
        "",
        fr.render(
            [e_fast],
            fr._parse_ts(NOW),
            wide=False,
            gpu_only=False,
            color=False,
            stages=True,
            fast=True,
        ),
    )
    check(
        "inventory-honesty (e2e --fast): the pane never states 'nothing installed' on an UNREAD inventory",
        "nothing installed" not in pf and ("PARTIAL" in pf or "inventory UNKNOWN" in pf),
        pf,
    )


# ── the MODEL LEDGER ─────────────────────────────────────────────────────────────────────────────────────
# The reported defect, in one screenshot:
#     ✓ model: nemotron-ultra-nvfp4-cache @183968f87ae4cedce3039313cac1fd43d112c578  … verified by PVC stamp
# A PVC name in the model column · a 40-char label in a 12-char column · the word "verified" backed by a
# label an operator had typed — and the panel's own backfill hint told them to type it. Each of those is
# encoded below so it cannot come back, plus the two things ONE-ROW-PER-PVC could never express: a second
# model inside one claim, and a catalog model with no claim at all.
_LEDGER_PIN = {
    "glm5-fp8": "4f96cc5eec29dcee5d6ded54f7ffe889438f9516",
    "qwen3-0-6b": "c1899de289a04d12100db370d81485cdf75e47ca",
    "nemotron-ultra-nvfp4": "183968f87ae4cedce3039313cac1fd43d112c578",
}
_LEDGER_CAT = [{"model": "glm5-fp8"}] * 27 + [{"model": "nemotron-ultra-nvfp4"}] * 3 + [{"model": "qwen3-0-6b"}]
_LEDGER_PROF = {
    "NAMESPACE": "example-benchmark",
    "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
    "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4": "nemotron-ultra-nvfp4-cache",
}


def _cfact(name, *, phase="Bound", size="1200Gi", labels=None, job=None, unreadable=""):
    return {
        "ns": "example-benchmark",
        "name": name,
        "phase": phase,
        "size": size,
        "labels": labels or {},
        "job": job,
        "unreadable": unreadable,
    }


def _unit_model_ledger():
    """The catalog-keyed MODEL LEDGER: `cache_inventory.expected_models` + `reconcile` + the glyph ladder."""
    sys.path.insert(0, str(SCRIPTS))
    import cache_inventory as ci
    import fleet_render as fr

    exp = ci.expected_models(_LEDGER_CAT, _LEDGER_PROF, _LEDGER_PIN)
    by = {e["model"]: e for e in exp}
    check(
        "ledger: the expected set is the CATALOG's models, with the cell count that needs each",
        [(e["model"], e["cells"]) for e in exp] == [("glm5-fp8", 27), ("nemotron-ultra-nvfp4", 3), ("qwen3-0-6b", 1)],
        str(exp),
    )
    # THE CLAIM RULE IS CONSUMED, NOT RE-DERIVED. Two resolvers is the bug model_cache.py was extracted to
    # end (~300 GiB downloaded into a claim the server never mounted) — so the ledger must agree with it
    # exactly, including the per-model override key.
    import model_cache as mc

    check(
        "ledger: every claim comes from model_cache.resolve_cache_claim — never a second rule",
        all(e["claim"] == mc.resolve_cache_claim({"model": e["model"]}, _LEDGER_PROF)[0] for e in exp)
        and by["nemotron-ultra-nvfp4"]["claim_source"] == "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4",
        str([(e["model"], e["claim"], e["claim_source"]) for e in exp]),
    )
    # SEVERAL PROFILES, ONE CLUSTER ROW: match on the UNION or the second profile's claim is invisible.
    two = ci.expected_models(
        [{"model": "glm5-fp8"}],
        [_LEDGER_PROF, {"MODEL_CACHE_PVC": "other-cache"}],
        _LEDGER_PIN,
    )
    check(
        "ledger: several profiles collapsed onto one cluster contribute the UNION of their claims",
        two[0]["claims"] == ["glm5-fp8-model-cache", "other-cache"],
        str(two),
    )

    # ── (1) THE REPORTED ROW. Stamped `download-complete=true`, NO model-name, 40-char revision. ──
    hand = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-revision": _LEDGER_PIN["nemotron-ultra-nvfp4"],
    }
    facts = [_cfact("nemotron-ultra-nvfp4-cache", size="800Gi", labels=hand)]
    rows = ci.reconcile([by["nemotron-ultra-nvfp4"]], facts, ns="example-benchmark")
    r = rows[0]
    check(
        "ledger(1): a stamped PVC with NO model-name never renders a model row bearing the PVC's name",
        len(rows) == 1
        and r["model"] == "nemotron-ultra-nvfp4"
        and not any(x["model"].endswith("-cache") for x in rows),
        str(rows),
    )
    check(
        "ledger(1): …and it is NEVER ✓ — a stamp no Job wrote proves nothing about the weights",
        r["state"] == "present"
        and r["glyph"] == "~"
        and r["grade"] == ci.GRADE_HAND_STAMP
        and "NOT written by a download Job" in r["evidence"],
        str(r),
    )

    # ── (2) FOLLOWING THE OLD BACKFILL HINT VERBATIM. `kubectl label pvc <x> download-complete=true` and
    #        nothing else — exactly the label state the removed remedy produced. ──
    for pvc_labels in (
        {"llmb.nvidia.com/download-complete": "true"},
        {
            "llmb.nvidia.com/download-complete": "true",
            "llmb.nvidia.com/model-revision": _LEDGER_PIN["glm5-fp8"],
        },
    ):
        got = ci.reconcile(
            [by["glm5-fp8"]],
            [_cfact("glm5-fp8-model-cache", labels=pvc_labels)],
            ns="example-benchmark",
        )
        check(
            "ledger(2): following the old `label pvc … download-complete=true` hint cannot produce a ✓",
            got[0]["state"] != "verified" and got[0]["glyph"] != "✓",
            str(got),
        )
    hint = fr.backfill_hint(rows, "example-benchmark")
    check(
        "ledger(2): the remediation line offers NO label write at all — the loop is closed at the source",
        "kubectl" not in hint and "label pvc" not in hint and "download-complete" not in hint and "unvouched" in hint,
        hint,
    )

    # ── (3) REVISION WIDTH + WRONG REVISION ──
    job40 = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-name": "glm5-fp8",
        "llmb.nvidia.com/model-revision": _LEDGER_PIN["glm5-fp8"][:12],
    }
    got = ci.reconcile(
        [by["glm5-fp8"]],
        [_cfact("glm5-fp8-model-cache", labels=job40)],
        ns="example-benchmark",
    )
    check(
        "ledger(3): a 40-char pinned revision renders at exactly 12 characters",
        got[0]["rev"] == "4f96cc5eec29" and len(got[0]["rev"]) == 12,
        str(got),
    )
    wrong = dict(job40, **{"llmb.nvidia.com/model-revision": "deadbeefcafe"})
    got = ci.reconcile(
        [by["glm5-fp8"]],
        [_cfact("glm5-fp8-model-cache", labels=wrong)],
        ns="example-benchmark",
    )
    check(
        "ledger(3): a revision that is neither pinned[:12] nor a prefix of the pin reads WRONG REVISION",
        "WRONG REVISION" in got[0]["evidence"]
        and got[0]["state"] == "present"
        and got[0]["grade"] == ci.GRADE_WRONG_REV,
        str(got),
    )

    # ── (4) ALIGNMENT INVARIANT: the row's display width cannot depend on label length. This is what a
    #        40-char label broke — `cw = max(len(disp))` sized the column off ONE row's raw label. ──
    def _widths(rev_label):
        lab = dict(job40, **{"llmb.nvidia.com/model-revision": rev_label})
        rr = ci.reconcile(
            [by["glm5-fp8"]],
            [_cfact("glm5-fp8-model-cache", labels=lab)],
            ns="example-benchmark",
        )
        return len(rr[0]["model"]) + len(rr[0]["rev"])

    check(
        "ledger(4): row width is IDENTICAL for a 12-char and a 40-char revision label",
        _widths("4f96cc5eec29") == _widths(_LEDGER_PIN["glm5-fp8"]) == _widths("z" * 200),
        f'{_widths("4f96cc5eec29")}/{_widths(_LEDGER_PIN["glm5-fp8"])}/{_widths("z" * 200)}',
    )

    # ── (5) llmb-control and *-artifacts are STORAGE, never models ──
    disc = fr.discover_model_caches(
        {
            "items": [
                {
                    "metadata": {
                        "name": n,
                        "namespace": "example-benchmark",
                        "labels": {},
                    },
                    "spec": {"resources": {"requests": {"storage": "5Gi"}}},
                    "status": {"phase": "Bound", "capacity": {"storage": "5Gi"}},
                }
                for n in ("llmb-control", "cell-r1-artifacts", "glm5-fp8-model-cache")
            ]
        },
        [],
        our_namespaces={"example-benchmark"},
        live_claims=set(),
    )
    inv, icounts, _ = fr.build_install_inventory({}, disc, name_map={}, fallback_ns="example-benchmark")
    ledger = ci.reconcile(exp, fr.claim_facts(disc, "example-benchmark"), ns="example-benchmark")
    check(
        "ledger(5): llmb-control and *-artifacts are excluded from the INSTALLED (cell) count",
        icounts["total"] == 0 and inv == [],
        str(inv),
    )
    check(
        "ledger(5): they are excluded from the MODEL count too — model rows == catalog models",
        ci.ledger_counts(ledger)["catalog"] == 3
        and ci.ledger_counts(ledger)["extra"] == 0
        and not any(r["model"] in ("llmb-control", "cell-r1-artifacts") for r in ledger),
        str(ledger),
    )
    sto, scounts = fr.build_storage_inventory(disc, fallback_ns="example-benchmark")
    check(
        "ledger(5): every PVC lands in STORAGE instead, bucketed by role (1 cache · 1 control · 1 output)",
        (scounts["caches"], scounts["control"], scounts["artifacts_pvcs"]) == (1, 1, 1),
        str(scounts),
    )

    # ── (6) a catalog model with NO PVC anywhere is a ROW (○ MISSING), not an omission ──
    rows = ci.reconcile(exp, [], ns="example-benchmark")
    miss = {r["model"]: r for r in rows}
    check(
        "ledger(6): a catalog model with no PVC anywhere renders ○ MISSING — absence is a row",
        len(rows) == 3
        and all(r["state"] == "missing" and r["glyph"] == "○" for r in rows)
        and "does not exist" in miss["glm5-fp8"]["evidence"],
        str(rows),
    )

    # ── (7) ONE CLAIM, TWO MODELS. The accumulating per-model key is what makes this expressible; the
    #        single-valued pair is merge-patched and records only the LAST model written. ──
    both = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-name": "qwen3-0-6b",
        "llmb.nvidia.com/model-revision": _LEDGER_PIN["qwen3-0-6b"][:12],
        ci.model_label_key("glm5-fp8"): _LEDGER_PIN["glm5-fp8"][:12],
        ci.model_label_key("qwen3-0-6b"): _LEDGER_PIN["qwen3-0-6b"][:12],
    }
    rows = ci.reconcile(
        [by["glm5-fp8"], by["qwen3-0-6b"]],
        [_cfact("glm5-fp8-model-cache", labels=both)],
        ns="example-benchmark",
    )
    check(
        "ledger(7): a claim holding TWO models renders TWO rows, both from the one claim",
        [(r["model"], r["state"]) for r in rows] == [("glm5-fp8", "verified"), ("qwen3-0-6b", "verified")]
        and {r["claim"] for r in rows} == {"glm5-fp8-model-cache"},
        str(rows),
    )
    check(
        "ledger(7): only the FIRST row of a shared claim quotes its size (the rest read `shared`)",
        [r["size"] for r in rows] == ["1200Gi", "shared"],
        str([r["size"] for r in rows]),
    )
    # …and with only the merge-patched pair present, the second model is honestly UNVOUCHED, not invented.
    solo = {k: v for k, v in both.items() if not k.startswith(ci.MODEL_LABEL_PREFIX)}
    rows = ci.reconcile(
        [by["glm5-fp8"], by["qwen3-0-6b"]],
        [_cfact("glm5-fp8-model-cache", labels=solo)],
        ns="example-benchmark",
    )
    check(
        "ledger(7): the merge-patched pair speaks for ONE model — the other reads ~, never ✓ and never ○",
        [(r["model"], r["glyph"]) for r in rows] == [("glm5-fp8", "~"), ("qwen3-0-6b", "✓")],
        str([(r["model"], r["glyph"], r["evidence"]) for r in rows]),
    )

    # ── (8) ONE unreadable claim → exactly one `?`, a non-zero UNKNOWN count, nothing else disturbed ──
    ok = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-name": "glm5-fp8",
        "llmb.nvidia.com/model-revision": _LEDGER_PIN["glm5-fp8"][:12],
    }
    base_facts = [
        _cfact("glm5-fp8-model-cache", labels=ok),
        _cfact("nemotron-ultra-nvfp4-cache", size="800Gi", labels=hand),
    ]
    before = ci.reconcile(exp, base_facts, ns="example-benchmark")
    blind = [dict(base_facts[0]), dict(base_facts[1], unreadable="cannot-mount")]
    after = ci.reconcile(exp, blind, ns="example-benchmark")
    cb, ca = ci.ledger_counts(before), ci.ledger_counts(after)
    unk = [r for r in after if r["state"] == "unknown"]
    check(
        "ledger(8): one unreadable claim yields EXACTLY one ? row and a non-zero UNKNOWN count",
        len(unk) == 1 and unk[0]["glyph"] == "?" and ca["unknown"] == 1,
        str(after),
    )
    check(
        "ledger(8): the other rows are untouched (an unread claim does not reclassify a read one)",
        [(r["model"], r["state"]) for r in before if r["state"] != "present"]
        == [(r["model"], r["state"]) for r in after if r["state"] not in ("present", "unknown")],
        f"{before} || {after}",
    )
    check(
        "ledger(8): UNKNOWN is its OWN bucket — never folded into verified/present/missing",
        ca["verified"] == cb["verified"] and ca["missing"] == cb["missing"] and ca["present"] == cb["present"] - 1,
        f"{cb} || {ca}",
    )

    # ── (9) a ? row NAMES its cause, and the four causes are four distinct sentences ──
    causes = ["rbac", "fast", "unschedulable", "cannot-mount"]
    texts = [ci.unknown_cause(c) for c in causes]
    check(
        "ledger(9): the four UNKNOWN causes produce four DISTINCT strings (an unattributed ? is a dead end)",
        len(set(texts)) == 4 and all(t for t in texts),
        str(texts),
    )
    check(
        "ledger(9): each cause names the read that failed (RBAC · --fast · unschedulable · mount)",
        "can-i list pvc" in texts[0] and "--fast" in texts[1] and "unschedulable" in texts[2] and "mount" in texts[3],
        str(texts),
    )
    for c, t in zip(causes, texts):
        got = ci.reconcile(
            [by["glm5-fp8"]],
            [_cfact("glm5-fp8-model-cache", unreadable=c)],
            ns="example-benchmark",
        )
        check(
            f"ledger(9): a claim unreadable because `{c}` renders ? carrying that exact cause",
            got[0]["state"] == "unknown" and got[0]["evidence"] == t,
            str(got),
        )
    check(
        "ledger(9): a read that did not land AT ALL is UNKNOWN for every model, never MISSING",
        [r["state"] for r in ci.reconcile(exp, None, unreadable="rbac", ns="example-benchmark")] == ["unknown"] * 3,
        "",
    )

    # ── (10) PROPERTY: ✓ appears IFF the evidence grade is sentinel or job-stamp. Swept over every label
    #         shape the cluster can produce — no other input may reach a green tick. ──
    P = _LEDGER_PIN["glm5-fp8"]
    shapes = [
        {},
        {"llmb.nvidia.com/download-complete": "true"},
        {
            "llmb.nvidia.com/download-complete": "true",
            "llmb.nvidia.com/model-revision": P,
        },
        {
            "llmb.nvidia.com/download-complete": "true",
            "llmb.nvidia.com/model-revision": P[:12],
        },
        {
            "llmb.nvidia.com/download-complete": "true",
            "llmb.nvidia.com/model-name": "glm5-fp8",
        },
        {
            "llmb.nvidia.com/download-complete": "true",
            "llmb.nvidia.com/model-name": "glm5-fp8",
            "llmb.nvidia.com/model-revision": P[:12],
        },
        {
            "llmb.nvidia.com/download-complete": "true",
            "llmb.nvidia.com/model-name": "glm5-fp8",
            "llmb.nvidia.com/model-revision": "deadbeefcafe",
        },
        {
            "llmb.nvidia.com/model-name": "glm5-fp8",
            "llmb.nvidia.com/model-revision": P[:12],
        },
        {ci.model_label_key("glm5-fp8"): P[:12]},
        {ci.model_label_key("glm5-fp8"): "deadbeefcafe"},
    ]
    bad = []
    for lab in shapes:
        for jb in (
            None,
            {
                "model": "glm5-fp8",
                "rev": P[:12],
                "done": True,
                "failed": False,
                "running": False,
            },
            {
                "model": "glm5-fp8",
                "rev": "deadbeefcafe",
                "done": True,
                "failed": False,
                "running": False,
            },
        ):
            for ph in ("Bound", "Pending"):
                rr = ci.reconcile(
                    [by["glm5-fp8"]],
                    [_cfact("glm5-fp8-model-cache", phase=ph, labels=lab, job=jb)],
                    ns="example-benchmark",
                )[0]
                if (rr["glyph"] == "✓") != (rr["grade"] in ci.VERIFIED_GRADES):
                    bad.append((lab, jb, ph, rr["glyph"], rr["grade"]))
    check(
        "ledger(10): ✓ appears IFF the grade is sentinel|job-stamp — no other input reaches a green tick",
        not bad,
        str(bad[:2]),
    )
    check(
        "ledger(10): VERIFIED_GRADES is exactly {sentinel, job-stamp} — the whitelist is one fact",
        ci.VERIFIED_GRADES == frozenset({"sentinel", "job-stamp"}),
        str(ci.VERIFIED_GRADES),
    )

    # ── the accumulating label key: the template truncates the WHOLE key at 63, so the name half is capped
    #    tighter than the API allows, and the value must be matched on the same normalisation both writers use
    check(
        "ledger: the per-model key matches the template's ('llmb.nvidia.com/model.' + name)[:63]",
        ci.model_label_key("glm5-fp8") == "llmb.nvidia.com/model.glm5-fp8"
        and len(ci.model_label_key("x" * 90)) == 63
        and ci.MODEL_LABEL_NAME_CAP == 63 - len("llmb.nvidia.com/model."),
        ci.model_label_key("x" * 90),
    )
    check(
        "ledger: a name TRUNCATED into the key still matches its full catalog model",
        ci._slug_matches(ci.model_label_key("m" * 60)[len(ci.MODEL_LABEL_PREFIX) :], "m" * 60),
        "",
    )
    check(
        "ledger: '_' and '.' in a model name fold to the same slug on BOTH sides of the compare",
        ci.label_slug("Qwen3_0.6B") == ci.label_slug("qwen3-0-6b") == "qwen3-0-6b",
        ci.label_slug("Qwen3_0.6B"),
    )

    # ── the pinned revisions actually come from the recipes, so the ledger's pin is the recipe's pin ──
    pins = fr.load_model_pins()
    check(
        "ledger: pins are read from the cells' serving.model_revision (the recipe IS the pin)",
        pins.get("glm5-fp8", "").startswith("4f96cc5eec29") and len(pins.get("nemotron-ultra-nvfp4", "")) == 40,
        str(pins),
    )


# ── COLLAPSE SAFETY + the INSTALLED fold ─────────────────────────────────────────────────────────────────
# THE REPORTED PANE. A healthy namespace on a short terminal rendered:
#       … 31 lines hidden (installed inventory)
#       … 6 lines hidden (installed inventory)
#       … 2 lines hidden (installed inventory)
# The operator's words: "not actionable since I don't know what to do with it or verify if it's okay or
# bad. I would rather either not see it or be able to know if it matters." That is the same defect class as
# `3 installed · 3 ready` — a render he cannot classify as fine-or-broken. The old one asserted too much;
# this one asserted nothing. The fix has two halves, both pinned here: a long tail of HEALTHY rows is
# folded at the source (so the viewport rarely has to act at all), and whatever the viewport does hide is
# GUARANTEED healthy — which is what entitles the marker to say so.
_ATTENTION_GLYPHS = ("✗", "○", "?", "~", "⚠")


def _live_entry(fr, *, n_cells=30, failed=0, ns="example-benchmark"):
    """The operator's cluster: 31-cell catalog, N installed cells, 3 models (1 verified / 2 unverified),
    3 PVCs, a run holding GPUs. The state that produced the reported frame."""
    now = fr._parse_ts(NOW)
    pins, cat = fr.load_model_pins(), fr.load_catalog()

    def _pvc(name, size, labels=None):
        return {
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": name, "namespace": ns, "labels": labels or {}},
            "spec": {"resources": {"requests": {"storage": size}}},
            "status": {"phase": "Bound", "capacity": {"storage": size}},
        }

    pvcs = {
        "items": [
            _pvc(
                "glm5-fp8-model-cache",
                "1200Gi",
                {
                    "llmb.nvidia.com/download-complete": "true",
                    "llmb.nvidia.com/model-name": "glm5-fp8",
                    "llmb.nvidia.com/model-revision": (pins.get("glm5-fp8") or "0" * 40)[:12],
                },
            ),
            _pvc(
                "nemotron-ultra-nvfp4-cache",
                "800Gi",
                {
                    "llmb.nvidia.com/download-complete": "true",
                    "llmb.nvidia.com/model-revision": pins.get("nemotron-ultra-nvfp4", "0" * 40),
                },
            ),
            _pvc("llmb-control", "5Gi"),
        ]
    }
    srv = deploy("glm5-a-server", app="glm5-a-server", ready=1, cell=cat[0]["name"])
    jb = job(
        "glm5-a-bench-r1",
        recipe=cat[0]["name"],
        cell=cat[0]["name"],
        active=1,
        run_id="r1",
    )
    for o in (srv, jb):  # one namespace, or the inventory splits across two blocks
        o["metadata"]["namespace"] = ns
    sp = pod("glm5-a-server-1", app="glm5-a-server", gpus=8, ns=ns)
    jp = pod("glm5-a-bench-r1-1", job="glm5-a-bench-r1", gpus=8, ns=ns)
    e = fr.build_cluster(
        "example-gpu-cluster",
        "ctxA",
        ns,
        items([sp, jp]),
        items([srv]),
        items([jb]),
        items([node("b200-1", 8), node("b200-2", 8)]),
        items([sp, jp]),
        now=now,
        all_deploys=[srv],
        all_jobs=[jb],
        pvcs_j=pvcs,
    )
    installs = {}
    for i, c in enumerate(cat[:n_cells]):
        # mark the LAST cells bad: cat[0] is also the LIVE-discovered one, and live discovery wins over a
        # stamp, so a failure planted there would be silently upgraded to ✓ and the fixture would lie.
        bad = i >= n_cells - failed
        installs[c["_path"]] = {
            "staged": {"stage-dataset": {"ok": not bad}},
            "preflight": "skipped" if bad else "pass",
        }
    e.update(
        gpu_type="B200",
        profiles=1,
        stale="",
        readiness=None,
        installs=installs,
        profile_envs=[
            {
                "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
                "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4": "nemotron-ultra-nvfp4-cache",
                "NAMESPACE": ns,
            }
        ],
    )
    return e, now


def _unit_collapse_safety():
    """THE INVARIANT: a collapsed range contains no row that needs action — so an inventory marker's
    `all ✓` is a FACT, not a hope — plus the INSTALLED fold that keeps the viewport from having to act.
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    e, now = _live_entry(fr)

    def _render(rows=0, cols=200, detail=False, ent=None, **kw):
        ent = ent if ent is not None else (_live_entry(fr, **kw)[0] if kw else e)
        return ANSI.sub(
            "",
            fr.render(
                [ent],
                now,
                wide=False,
                gpu_only=False,
                color=False,
                stages=True,
                detail=detail,
                viewport_rows=rows,
                viewport_cols=cols,
            ),
        )

    # ── (0) THE STRUCTURAL GUARANTEE, in one comparison: no rung of the ladder can reach an attention row.
    check(
        "collapse: P_ATTENTION outranks every ladder level — an actionable row is UNCOLLAPSIBLE by design",
        fr.P_ATTENTION < min(fr._VP_LADDER) and fr.P_STRUCT < min(fr._VP_LADDER),
        f"P_ATTENTION={fr.P_ATTENTION} ladder_min={min(fr._VP_LADDER)}",
    )

    # ── (1) THE FOLD. 30 near-identical `✓ <110-char path>  ready` rows were 30 of the namespace's 45 lines
    #        and carried one bit of information between them. They fold into a line that STATES the verdict.
    full = _render()
    check(
        "fold: a 30-cell healthy inventory renders ONE folded line, not 30 rows",
        "✓ all 30 ready — nothing needs attention" in full and full.count("✓ llm-perf/") == 0,
        full,
    )
    check(
        "fold: the fold line names the flag that expands it — `--detail`, which KEEPS this pane "
        "(`--all` sets FLAT_ONLY_FLAG in fleet.sh, switching --watch to the FLAT pane)",
        "(--detail to list them)" in full and "--all to list" not in full,
        full,
    )
    check(
        "fold: the header still carries the count + verdict, so the fold hides no ANSWER",
        "INSTALLED  30 cells · 30 ready" in full,
        full,
    )
    det = _render(detail=True)
    check(
        "fold: --detail lists every cell (nothing is unreachable, only un-defaulted)",
        det.count("✓ llm-perf/") == 30 and "all 30 ready" not in det,
        str(det.count("✓ llm-perf/")),
    )
    # a SHORT list is its own summary — folding `✓ parkrun` into `all 1 ready` would be strictly worse
    small = _render(n_cells=3)
    check(
        "fold: a list shorter than the fold threshold stays listed (there the names ARE the summary)",
        small.count("✓ llm-perf/") == 3 and "nothing needs attention" not in small,
        small,
    )
    mixed = _render(failed=2)
    check(
        "fold: ✗ FAILED cells are listed in full while the healthy tail folds behind them",
        mixed.count("✗ llm-perf/") == 2 and "28 more ✓ ready — nothing needs attention" in mixed,
        mixed,
    )

    # ── (2) THE PROPERTY, end to end over every viewport height. Nothing the COLLAPSE hides may need
    #        action, and no SECTION HEADER may be hidden — under the old per-section priority the header
    #        was the FIRST casualty.
    #        Scoped to frames that did not OVERFLOW: overflow is fit_viewport's last resort, where even
    #        runs + structure exceed the terminal and the TAIL is cut BY POSITION. That is a different
    #        mechanism with its own contract (it names itself in the footer, and selftest_fleet_viewport
    #        pins it); folding it in here would be asserting that a 12-row terminal can show a 25-row frame.
    def _hidden_offenders(ent, base, detail=False):
        off_rows, off_hdrs, seen = [], [], 0
        for rows in range(12, 60):
            cut = _render(rows=rows, ent=ent, detail=detail)
            if "overflow" in cut:
                continue
            seen += 1
            kept = collections.Counter(cut.splitlines())
            for ln, n in collections.Counter(base.splitlines()).items():
                if kept.get(ln, 0) >= n:
                    continue
                body = ln.strip()
                if any(g in body.split("  ")[0] for g in _ATTENTION_GLYPHS):
                    off_rows.append((rows, body[:70]))
                if any(body.startswith(f"{b} {h}") for b in ("├─", "└─") for h in ("INSTALLED", "MODELS", "STORAGE")):
                    off_hdrs.append((rows, body[:50]))
        return off_rows, off_hdrs, seen

    bad_rows, bad_hdrs, n_seen = _hidden_offenders(e, full)
    check(
        "collapse (property): across EVERY collapsing viewport height, nothing hidden needs action",
        not bad_rows and n_seen >= 10,
        f"{bad_rows[:3]} over {n_seen} heights",
    )
    check(
        "collapse (property): a SECTION HEADER is never hidden — the verdict outlives its rows",
        not bad_hdrs,
        str(bad_hdrs[:3]),
    )
    # a healthy frame proves less than one that actually HAS rows needing action
    e2 = _live_entry(fr, failed=3)[0]
    bad2, _h2, n2 = _hidden_offenders(e2, _render(ent=e2))
    check(
        "collapse (property): with ✗ and ~ rows PRESENT, not one of them is ever collapsed away",
        not bad2 and n2 >= 10,
        f"{bad2[:3]} over {n2} heights",
    )
    # …and under --detail the 30 healthy rows are back, so the collapse has real work to do again
    bad3, hdr3, n3 = _hidden_offenders(e2, _render(ent=e2, detail=True), detail=True)
    check(
        "collapse (property): --detail restores 30 collapsible rows — still nothing actionable is hidden",
        not bad3 and not hdr3 and n3 >= 5,
        f"{bad3[:3]} / {hdr3[:2]} over {n3} heights",
    )

    # ── (3) THE MARKER ANSWERS "DOES THIS MATTER?" — not merely how many lines are gone. Exercised under
    #        --detail, because the DEFAULT pane now folds the long tail at the SOURCE and leaves the
    #        viewport nothing to collapse: that absence is itself the headline fix, asserted just below.
    marked = [
        ln
        for r in range(20, 60)
        for ln in _render(rows=r, ent=e2, detail=True).splitlines()
        if fr.VP_MARK in ln and "installed cells" in ln
    ]
    check(
        "collapse: every installed-cells marker states the VERDICT and names the flag",
        marked and all("all ✓, nothing needs attention" in m and "--detail" in m for m in marked),
        str(marked[:2]),
    )
    # …and it names the SECTION it replaced. Three sections all labelled `installed inventory` is what made
    # one namespace read as a rendering fault rather than as three lists.
    check(
        "collapse: a marker names its OWN section, never a label borrowed from another one",
        all("installed cells" in m and "model rows" not in m and "storage rows" not in m for m in marked),
        str(marked[:2]),
    )
    plain_marks = [ln for r in range(20, 60) for ln in _render(rows=r).splitlines() if fr.VP_MARK in ln]
    check(
        "collapse: the DEFAULT pane needs NO inventory marker at all — the fold removed the pressure",
        not any(x in m for m in plain_marks for x in ("installed cells", "model rows", "storage rows")),
        str(plain_marks[:3]),
    )

    # ── (4) A ONE-LINE REGION IS NEVER TRADED FOR A ONE-LINE APOLOGY ──
    lines = ["keep", "solo", "keep2", "a", "b", "c"]
    prios = [
        fr.P_RUN,
        fr.P_INSTALLED,
        fr.P_RUN,
        fr.P_INSTALLED,
        fr.P_INSTALLED,
        fr.P_INSTALLED,
    ]
    out, _op, omark, _lbl = fr._vp_collapse(lines, prios, fr.P_INSTALLED)
    check(
        "collapse: a 1-line region is KEPT (a marker costs that same line and loses the content)",
        "solo" in out and sum(1 for m in omark if m) == 1,
        str(out),
    )
    check(
        "collapse: a 3-line region IS collapsed (there the marker actually buys rows back)",
        any(fr.VP_MARK in o and "3 lines hidden" in o for o in out),
        str(out),
    )
    check(
        "collapse: no `… 1 line hidden` marker is producible at all",
        not any("1 line hidden" in o for o in out),
        str(out),
    )


# ── STARTING vs UNSCHEDULED ──────────────────────────────────────────────────────────────────────────────
# THE REPORTED PANE. Two rows, in COMPLETELY different conditions, rendered as one word:
#       ├─ ● PENDING  qwen3-…-kvbm-pareto-…-m3m0   8 GPUs   10m25s / ~31m exp (33%)  8·16
#       ├─   └ svc    qwen3-…-kvbm-pareto-server   8 GPUs   1/1  11m44s
#       └─ ● LOADING  glm5-agg-8k1k-…-server       8 GPUs   0/1  10m21s
# The operator: "what does pending mean? Are the models loading? pending is vague so it's not clear if
# something is possibly wrong or this is indeed just loading the model." He was right to ask: the glm5
# server was genuinely warming up, while the qwen3 BENCH pod had been FailedScheduling for ELEVEN MINUTES
# and would have sat there forever — and its own server row read `1/1`, which is what made it look fine.
# Kubernetes' `Pending` phase covers both conditions, so passing the phase through is the lossy step.
def _unit_schedule_states():
    """UNSCHEDULED is not STARTING; a run that never started has no percentage and no ETA; and the reason
    no node took the pod rides the row instead of a `kubectl describe` the reader has to go and run.
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    now = fr._parse_ts(NOW)
    MSG = (
        "0/11 nodes are available: 1 node(s) didn't match pod affinity rules, "
        "10 Insufficient cpu, 3 Insufficient memory."
    )

    def _p(name, *, created, sched=True, msg=MSG):
        p = pod(name, job="b", gpus=8, phase="Pending", start=created)
        p["metadata"]["creationTimestamp"] = created
        if not sched:
            p["spec"].pop("nodeName", None)
            p["status"]["conditions"] = [
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": msg,
                }
            ]
        return p

    T_OLD = "2026-07-19T11:49:35Z"  # 10m25s before NOW — the reported age
    T_NEW = "2026-07-19T11:59:30Z"  # 30s before NOW — a normal transient

    # ── (1) the two conditions are two STATES, and k8s' own phase cannot tell them apart ──
    uns, starting = _p("bench-u", created=T_OLD, sched=False), _p("bench-s", created=T_OLD)
    check(
        "sched: both pods are k8s-phase Pending — the phase is exactly what cannot answer the question",
        fr._phase(uns) == fr._phase(starting) == "Pending",
        "",
    )
    jb = {"status": {"active": 1}}
    check(
        "sched: a pod NO node accepted → UNSCHEDULED; one merely coming up → STARTING",
        (fr.classify_job(jb, [uns]), fr.classify_job(jb, [starting])) == (fr.UNSCHED, fr.STARTING),
        f"{fr.classify_job(jb, [uns])} / {fr.classify_job(jb, [starting])}",
    )
    check(
        "sched: the two words are distinct and neither is the old vague `PENDING`",
        fr.UNSCHED == "UNSCHEDULED" and fr.STARTING == "STARTING",
        f"{fr.UNSCHED}/{fr.STARTING}",
    )
    check(
        "sched: both remain ACTIVE run states (splitting the word must not drop a run from the pane)",
        {fr.UNSCHED, fr.STARTING} <= fr.ACTIVE_JOB_STATES,
        str(fr.ACTIVE_JOB_STATES),
    )
    check(
        "sched: the STATUS column still fits the longest state (`● UNSCHEDULED`) without truncating",
        fr.STATUS_CAP >= len("● " + fr.UNSCHED),
        f"cap={fr.STATUS_CAP}",
    )

    # ── (2) THE REASON IS ONE FIELD AWAY AND BELONGS INLINE ──
    why = fr.unschedulable_reason(uns)
    check(
        "sched: the row carries the scheduler's own verdict — nodes tried + why each was rejected",
        why.startswith("FailedScheduling: 0/11 nodes") and "Insufficient cpu" in why,
        why,
    )
    check(
        "sched: causes are ordered by how many nodes each eliminated (the biggest number is the fix)",
        why.index("10 Insufficient cpu")
        < why.index("3 Insufficient memory")
        < why.index("1 didn't match pod affinity"),
        why,
    )
    check(
        "sched: `node(s)` scheduler noise is stripped so the causes read as a list",
        "node(s)" not in why,
        why,
    )
    check(
        "sched: a SCHEDULABLE pod gets no reason at all (this never decorates a healthy row)",
        fr.unschedulable_reason(starting) == "",
        fr.unschedulable_reason(starting),
    )
    # absence-as-signal: a missing message names the absence rather than going quiet
    quiet = _p("bench-q", created=T_OLD, sched=False, msg="")
    check(
        "sched: an unschedulable pod with NO message says the reason is missing, never nothing",
        "no reason" in fr.unschedulable_reason(quiet),
        fr.unschedulable_reason(quiet),
    )
    weird = _p(
        "bench-w",
        created=T_OLD,
        sched=False,
        msg="something the scheduler has never said before",
    )
    check(
        "sched: an UNPARSEABLE message is passed through, not swallowed",
        "something the scheduler" in fr.unschedulable_reason(weird),
        fr.unschedulable_reason(weird),
    )
    many = _p(
        "bench-m",
        created=T_OLD,
        sched=False,
        msg="0/40 nodes are available: 9 a, 8 b, 7 c, 6 d, 5 e.",
    )
    check(
        "sched: a long cause list is capped and SAYS it was capped (never silently truncated)",
        "+2 more" in fr.unschedulable_reason(many),
        fr.unschedulable_reason(many),
    )

    # ── (3) NO PERCENTAGE FOR WORK THAT HAS NOT STARTED. The ratio is clock-driven, so the longer the run
    #        stayed broken the more FINISHED it looked; at ~31m it would have read 100% having done nothing.
    row_u = {
        "status": fr.UNSCHED,
        "start": T_OLD,
        "expected": 1860,
        "expected_src": "median",
    }
    row_s = {
        "status": fr.STARTING,
        "start": T_OLD,
        "expected": 1860,
        "expected_src": "median",
    }
    row_r = {
        "status": fr.RUNNING,
        "start": T_OLD,
        "expected": 1860,
        "expected_src": "median",
    }
    te_u, te_s = fr._elapsed_expected(row_u, now), fr._elapsed_expected(row_s, now)
    te_r = fr._elapsed_expected(row_r, now)
    check(
        "sched: an UNSCHEDULED run shows the WAIT, never an elapsed-vs-expected percentage",
        "waiting 10m25s" in te_u and "%" not in te_u and "exp" not in te_u,
        te_u,
    )
    # STARTING IS THE SAME LIE IN A DIFFERENT WORD. classify_job returns it for ContainerCreating, an image
    # pull, and a volume that will not mount — so a run wedged on a bad mount read `40m0s / ~31m exp (129%)`,
    # growing toward "finished" while nothing ran. A percentage means something only once a container Runs.
    check(
        "sched: a STARTING run shows the WAIT too — no percentage before a container reaches Running",
        "starting 10m25s" in te_s and "%" not in te_s and "exp" not in te_s,
        te_s,
    )
    check(
        "sched: a RUNNING run keeps its honest progress reading (the fix is scoped, not blanket)",
        "34%" in te_r and "exp" in te_r,
        te_r,
    )
    long_s = fr._elapsed_expected({"status": fr.STARTING, "start": "2026-07-19T11:10:00Z", "expected": 1860}, now)
    check(
        "sched: a 50-minute STARTING run is badged — a wedged image pull or mount is not 'coming up'",
        "⚠" in long_s and "no container has reached Running" in long_s,
        long_s,
    )
    check(
        "sched: …and a long wait earns a BADGE, the same discipline as ⚠ STALE on a laggard cluster",
        "⚠" in te_u and "no node has accepted it" in te_u,
        te_u,
    )
    fresh = fr._elapsed_expected({"status": fr.UNSCHED, "start": T_NEW, "expected": 1860}, now)
    check(
        "sched: a FRESH unscheduled pod is not badged — seconds of scheduling is normal, minutes is not",
        "waiting 30s" in fresh and "⚠" not in fresh,
        fresh,
    )
    check(
        "sched: severity tracks the clock — yellow while transient, red once established",
        (
            fr._status_code({"status": fr.UNSCHED, "level": "job", "waited": 10}),
            fr._status_code({"status": fr.UNSCHED, "level": "job", "waited": 900}),
        )
        == ("33", "31"),
        "",
    )

    # ── (4) NO ETA EITHER. The fleet headline advertised `⏱ soonest … ~ETA 20m` for a run no node had
    #        accepted — and it got SOONER the longer it stayed broken.
    check(
        "sched: neither NOT-STARTED state contributes an ETA — only a run that is actually running does",
        fr._remaining(row_u, now) is None
        and fr._remaining(row_s, now) is None
        and fr._remaining(row_r, now) is not None,
        f"{fr._remaining(row_u, now)} / {fr._remaining(row_s, now)} / {fr._remaining(row_r, now)}",
    )

    # ── (5) END TO END on the reported shape: a dead bench, its healthy 1/1 server, and a real loader ──
    srv = deploy(
        "qwen3-kvbm-pareto-server",
        app="qwen3-kvbm-pareto-server",
        ready=1,
        cell="qwen3-kvbm",
    )
    srvp = pod(
        "qwen3-kvbm-pareto-server-a",
        app="qwen3-kvbm-pareto-server",
        gpus=8,
        start=T_OLD,
    )
    load = deploy(
        "glm5-agg-8k1k-server",
        app="glm5-agg-8k1k-server",
        ready=0,
        cell="glm5-agg-8k1k",
    )
    loadp = pod("glm5-agg-8k1k-server-a", app="glm5-agg-8k1k-server", gpus=8, start=T_OLD)
    loadp["status"]["conditions"] = [{"type": "Ready", "status": "False"}]  # 0/1 → genuinely loading
    bench = job(
        "qwen3-kvbm-bench-m3m0",
        recipe="qwen3-kvbm",
        cell="qwen3-kvbm",
        active=1,
        run_id="m3m0",
        expected_env=1860,
        start=T_OLD,
    )
    bp = _p("qwen3-kvbm-bench-m3m0-1", created=T_OLD, sched=False)
    bp["metadata"]["labels"]["job-name"] = "qwen3-kvbm-bench-m3m0"
    e = fr.build_cluster(
        "example-gpu-cluster",
        "ctxA",
        "ours",
        items([srvp, loadp, bp]),
        items([srv, load]),
        items([bench]),
        items([node("b200-1", 8), node("b200-2", 8)]),
        items([srvp, loadp, bp]),
        now=now,
        all_deploys=[srv, load],
        all_jobs=[bench],
        pvcs_j={"items": []},
    )
    e.update(
        gpu_type="B200",
        profiles=1,
        stale="",
        readiness=None,
        installs={},
        profile_envs=[],
    )
    plain = ANSI.sub("", fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True))
    check(
        "sched (e2e): the dead bench reads ● UNSCHEDULED, not ● PENDING",
        "● UNSCHEDULED" in plain and "● PENDING" not in plain,
        plain,
    )
    check(
        "sched (e2e): the genuinely-warming server still reads ● LOADING — the two are now separable",
        "● LOADING" in plain,
        plain,
    )
    check(
        "sched (e2e): the row states nodes-tried and the top cause, so no `kubectl describe` is needed",
        "0/11 nodes" in plain and "10 Insufficient cpu" in plain,
        plain,
    )
    check(
        "sched (e2e): no fabricated progress anywhere on the frame for the unscheduled run",
        "waiting 10m25s" in plain and "%" not in plain and "~ETA" not in plain,
        plain,
    )


# ── ROUND-3 AUDIT: three ways a reader still could not tell fine from broken ──────────────────────────────
def _unit_audit_join_and_absence():
    """(1) the two-profile UNION join, (2) an empty label certifying, (3) a hard blocker that never renders."""
    sys.path.insert(0, str(SCRIPTS))
    import cache_inventory as ci
    import fleet_render as fr

    PIN = fr.load_model_pins()
    NP, GP = PIN["nemotron-ultra-nvfp4"], PIN["glm5-fp8"]

    def fct(name, labels=None, size="800Gi", phase="Bound", unreadable=""):
        return {
            "ns": "n",
            "name": name,
            "phase": phase,
            "size": size,
            "labels": labels or {},
            "job": None,
            "unreadable": unreadable,
        }

    NEXP = [
        {
            "model": "nemotron-ultra-nvfp4",
            "claim": "claim-a",
            "claims": ["claim-a", "claim-b"],
            "claim_source": "MODEL_CACHE_PVC",
            "pinned_rev": NP,
            "cells": 3,
        }
    ]
    STAMP = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-name": "nemotron-ultra-nvfp4",
        "llmb.nvidia.com/model-revision": NP[:12],
    }

    # ── (1) FIRST-CLAIM-WINS HID THE SECOND PROFILE'S CLAIM. `expected_models` collects the UNION precisely
    #        so a collapsed second profile is not hidden; the join then picked the first claim that EXISTED,
    #        and the label-scan fallback was guarded by `if fact is None` so it could never run. Two false
    #        lines resulted, and since the documented remedy for "unvouched" is to RE-DOWNLOAD, that is the
    #        ~300 GiB wrong-claim failure re-entering by a new route.
    got = ci.reconcile(NEXP, [fct("claim-a", {}), fct("claim-b", STAMP)], ns="n")
    r = got[0]
    check(
        "join: the claim carrying the EVIDENCE wins, not the first one that happens to exist",
        r["claim"] == "claim-b" and r["state"] == "verified",
        str([(x["claim"], x["state"]) for x in got]),
    )
    # THE AUDITOR'S OWN CONTROL: deleting the decoy from the read must not change the verdict.
    solo = ci.reconcile(NEXP, [fct("claim-b", STAMP)], ns="n")[0]
    check(
        "join: deleting the decoy claim from the read changes NOTHING (it never carried the answer)",
        (solo["claim"], solo["state"], solo["evidence"]) == (r["claim"], r["state"], r["evidence"]),
        f"{solo} || {r}",
    )
    check(
        "join: …and no second row appears calling the real claim unattributed storage",
        len(got) == 1,
        str(got),
    )
    # ties keep PROFILE order, so with nothing to choose between them the primary profile still wins
    tie = ci.reconcile(NEXP, [fct("claim-a", {}), fct("claim-b", {})], ns="n")[0]
    check(
        "join: with NO evidence anywhere, the profile's own order still decides (a stable default)",
        tie["claim"] == "claim-a",
        tie["claim"],
    )
    # an UNREAD claim outranks "nothing here names it" — never assert an absence over a claim we could not read
    blind = ci.reconcile(NEXP, [fct("claim-a", unreadable="unread"), fct("claim-b", {})], ns="n")[0]
    check(
        "join: an UNREAD candidate beats a read-but-silent one — ? over a false 'no stamp names it'",
        blind["state"] == "unknown",
        f"{blind['state']} {blind['evidence']}",
    )

    # ── (2) AN EMPTY LABEL VALUE CERTIFIED. `short_rev` returns the PIN when the label is absent — a display
    #        rule — and the caller compared that against the pin, which is the pin compared with itself.
    #        `kubectl label pvc <c> llmb.nvidia.com/model.glm5-fp8=` reached `✓ … matches pin`.
    GEXP = [
        {
            "model": "glm5-fp8",
            "claim": "c",
            "claims": ["c"],
            "claim_source": "k",
            "pinned_rev": GP,
            "cells": 27,
        }
    ]
    for labels, what in (
        ({ci.model_label_key("glm5-fp8"): ""}, "per-model key with an EMPTY value"),
        (
            {
                "llmb.nvidia.com/download-complete": "true",
                "llmb.nvidia.com/model-name": "glm5-fp8",
                "llmb.nvidia.com/model-revision": "",
            },
            "single-valued pair with an empty revision",
        ),
    ):
        row = ci.reconcile(GEXP, [fct("c", labels)], ns="n")[0]
        check(
            f"certify: {what} yields NO ✓ — an absent value states nothing to compare",
            row["state"] != "verified" and row["glyph"] != "✓" and row["grade"] == ci.GRADE_NO_REV,
            f"{row['glyph']} {row['grade']} {row['evidence']}",
        )
        check(
            f"certify: …and the row SAYS the revision is missing ({what})",
            "records NO revision" in row["evidence"],
            row["evidence"],
        )
    ok = ci.reconcile(GEXP, [fct("c", {ci.model_label_key("glm5-fp8"): GP[:12]})], ns="n")[0]
    check(
        "certify: a REAL 12-char value still attests (the guard is on emptiness, not on the path)",
        ok["state"] == "verified" and ok["grade"] == ci.GRADE_JOB_STAMP,
        str(ok),
    )
    # the non-typed route the auditor named: an UNKNOWN pin must not be certified against either
    nopin = [{**GEXP[0], "pinned_rev": ""}]
    row = ci.reconcile(nopin, [fct("c", {ci.model_label_key("glm5-fp8"): GP[:12]})], ns="n")[0]
    check(
        "certify: with NO pin known, a Job-shaped label reads ~, never ✓ (nothing to check it against)",
        row["state"] == "present" and row["grade"] == ci.GRADE_JOB_UNPINNED,
        str(row),
    )
    # cells that DISAGREE on a model's pin make the pin unknown rather than accusing the cluster
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cat = []
        for i, rev in enumerate(("a" * 40, "b" * 40)):
            d = root / f"cells/c{i}"
            d.mkdir(parents=True)
            (d / "recipe.yaml").write_text(f"serving:\n  model_revision: {rev}\n")
            cat.append({"model": "m", "_path": f"cells/c{i}"})
        (root / "catalog.json").write_text(json.dumps(cat))
        pins = fr.load_model_pins(catalog_path=str(root / "catalog.json"), root=str(root))
        check(
            "pins: cells DISAGREEING on a model's revision make the pin UNKNOWN, never first-cell-wins",
            pins.get("m") == "",
            str(pins),
        )

    # ── (3) A HARD BLOCKER THAT NEVER RENDERED. A profile with no MODEL_CACHE_PVC cannot install one cell —
    #        install fails closed — but the row is `state="missing"` like any absent model, so it inherited
    #        the idle-collapse and the whole cluster rendered as one `· b200 · 2/2 GPU nodes free` line.
    now = fr._parse_ts(NOW)
    nodes = items([node("b-1", 8)])

    def cluster(profile):
        e = fr.build_cluster(
            "b200",
            "ctxA",
            "ns",
            items([]),
            items([]),
            items([]),
            nodes,
            items([]),
            now=now,
            all_deploys=[],
            all_jobs=[],
            pvcs_j={"items": []},
        )
        e.update(
            gpu_type="B200",
            profiles=1,
            stale="",
            readiness=None,
            installs={},
            profile_envs=[profile] if profile else [],
        )
        return ANSI.sub(
            "",
            fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True),
        )

    unconf = cluster({"NAMESPACE": "ns"})
    check(
        "blocker: a profile with NO MODEL_CACHE_PVC EXPANDS the cluster — it cannot install one cell",
        "━━ CLUSTER b200" in unconf and "MODELS" in unconf and "idle · connected" not in unconf,
        unconf,
    )
    check(
        "blocker: …and the remediation line backfill_hint ranks as HARD is finally reachable",
        "have nowhere to live" in unconf and 'MODEL_CACHE_PVC="<pvc>"' in unconf,
        unconf,
    )
    check(
        "blocker: is_config_blocker is ONE predicate — the renderer and the hint cannot drift apart",
        ci.is_config_blocker({"evidence": ci.NO_CLAIM_EVIDENCE})
        and not ci.is_config_blocker({"evidence": "claim x does not exist in this namespace"}),
        "",
    )
    # "configured but not created yet" is the ORDINARY state of an uninstalled cluster: it stays collapsed,
    # but the one line it gets must no longer be indistinguishable from a fully-stocked one.
    absent = cluster({"NAMESPACE": "ns", "MODEL_CACHE_PVC": "nope-cache"})
    check(
        "blocker: a claim that merely does not exist YET stays collapsed (that is every fresh cluster)",
        "idle · connected" in absent and "━━ CLUSTER b200" not in absent,
        absent,
    )
    check(
        "blocker: …but its idle line NAMES the gap — `not installed` no longer looks like `ready`",
        "models not installed" in absent,
        absent,
    )


def _unit_audit_render_honesty():
    """Unreadable inputs, the attested/verified distinction, and the tail-cut's mislabelled loss."""
    sys.path.insert(0, str(SCRIPTS))
    import cache_inventory as ci
    import fleet_render as fr

    now = fr._parse_ts(NOW)
    nodes = items([node("b-1", 8)])

    def pvcitem(n, s="1200Gi"):
        return {
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": n, "namespace": "ns", "labels": {}},
            "spec": {"resources": {"requests": {"storage": s}}},
            "status": {"phase": "Bound", "capacity": {"storage": s}},
        }

    def cluster(profiles, catalog=None):
        e = fr.build_cluster(
            "b200",
            "ctxA",
            "ns",
            items([]),
            items([]),
            items([]),
            nodes,
            items([]),
            now=now,
            all_deploys=[],
            all_jobs=[],
            pvcs_j={"items": [pvcitem("glm5-fp8-model-cache")]},
        )
        e.update(
            gpu_type="B200",
            profiles=1,
            stale="",
            readiness=None,
            installs={"recipes/x": {"staged": {"s": {"ok": True}}, "preflight": "pass"}},
            profile_envs=profiles,
        )
        orig = fr.load_catalog
        if catalog is not None:
            fr.load_catalog = lambda *a, **k: catalog
        try:
            return ANSI.sub(
                "",
                fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True),
            )
        finally:
            fr.load_catalog = orig

    # ── UNREADABLE INPUTS. `expected = []` was used for BOTH "no models needed" and "cannot tell", so the
    #    MODELS section vanished entirely and STORAGE asserted `no catalog model routes to this claim`
    #    about a volume three models route to — a confident negative from a file we could not open.
    for profiles, catalog, why in (
        ([], None, "cluster profile unreadable"),
        (
            [{"NAMESPACE": "ns", "MODEL_CACHE_PVC": "glm5-fp8-model-cache"}],
            [],
            "catalog unreadable",
        ),
    ):
        out = cluster(profiles, catalog)
        check(
            f"unknown-inputs: MODELS still renders, as UNKNOWN, when {why}",
            "MODELS" in out and "ledger UNKNOWN" in out and why in out,
            out,
        )
        check(
            f"unknown-inputs: the header never states a confident count when {why}",
            "0 catalog models" not in out and "0 missing" not in out,
            out,
        )
        check(
            f"unknown-inputs: STORAGE stops asserting `no catalog model routes` when {why}",
            "no catalog model routes" not in out and "cannot tell what routes here" in out,
            out,
        )

    # ── Distinguish label-based attestation from in-volume verification.
    c = ci.ledger_counts(
        [
            {"state": "verified", "grade": ci.GRADE_JOB_STAMP, "catalog": True},
            {"state": "present", "grade": ci.GRADE_HAND_STAMP, "catalog": True},
        ]
    )
    check(
        "attested: a Job-stamp ✓ counts as ATTESTED, and `verified` stays 0 with no in-volume check",
        (c["attested"], c["sentinel"]) == (1, 0),
        str(c),
    )
    summ = fr._models_summary(c, "live")
    check(
        "attested: the header says `1 attested · 0 verified (no in-volume check)`, not `1 verified`",
        "1 attested" in summ and "0 verified (no in-volume check)" in summ,
        summ,
    )
    c2 = ci.ledger_counts([{"state": "verified", "grade": ci.GRADE_SENTINEL, "catalog": True}])
    check(
        "attested: a SENTINEL — the grade that has seen bytes — is what earns the word `verified`",
        "1 verified" in fr._models_summary(c2, "") and "no in-volume check" not in fr._models_summary(c2, ""),
        fr._models_summary(c2, ""),
    )
    doc = __import__("cache_inventory").__doc__ or ""
    check(
        "attested: module documentation distinguishes label evidence from in-volume evidence",
        "label-based evidence" in doc and "in-volume sentinel" in doc,
        "",
    )
    check(
        "attested: module documentation states that sentinel evidence is not currently produced",
        "does not currently produce sentinel evidence" in doc,
        "",
    )

    # ── AN UNREAD PVC LIST IS NOT PROOF OF A DENIAL. INSTALLED hedged ("PVC list failed / RBAC-forbidden")
    #    while MODELS asserted FORBIDDEN in the same frame.
    e = fr.build_cluster(
        "b200",
        "ctxA",
        "ns",
        items([]),
        items([]),
        items([]),
        nodes,
        items([]),
        now=now,
        all_deploys=[],
        all_jobs=[],
        pvcs_j=None,
    )
    e.update(
        gpu_type="B200",
        profiles=1,
        stale="",
        readiness=None,
        installs={"recipes/x": {"staged": {"s": {"ok": True}}, "preflight": "pass"}},
        profile_envs=[{"NAMESPACE": "ns", "MODEL_CACHE_PVC": "c"}],
    )
    unread = ANSI.sub("", fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True))
    check(
        "unread: a PVC list that did not land is reported as NOT LANDED, never as FORBIDDEN",
        "did not land" in unread and "read FORBIDDEN" not in unread,
        unread,
    )
    check(
        "unread: …and it still names the check that would settle it",
        "can-i list pvc" in unread,
        unread,
    )

    # ── THE TAIL CUT. `labels` came from the LADDER's dropped priorities alone, so a cluster removed by the
    #    position-cut contributed only the word `overflow` — the footer named healthy categories while a
    #    BROKEN cluster had vanished whole. This section exists to stop exactly that.
    lines, prios, lbls = [], [], []
    for name in ("a-b200", "z-b200"):
        lines.append(f"━━ CLUSTER {name}  · connected  ━━━")
        prios.append(fr.P_STRUCT)
        lbls.append("")
        for k in range(6):
            lines.append(f"      ● RUNNING  {name}-run-{k}   8 GPUs   1h")
            prios.append(fr.P_RUN)
            lbls.append("")
    got = fr.fit_viewport(lines, prios, 8, cols=200, lbls=lbls)
    foot = [g for g in got if fr.VP_FOOTER_MARK in g]
    check(
        "tail-cut: a cluster removed ENTIRELY is NAMED in the footer, not left as a line count",
        foot and "CLUSTERS CUT: z-b200" in foot[0],
        foot[0] if foot else "no footer",
    )
    check(
        "tail-cut: the label list describes what was ACTUALLY cut (runs), not only what the ladder took",
        foot and "runs" in foot[0],
        foot[0] if foot else "",
    )
    check(
        "tail-cut: the surviving cluster is still named, so the reader can tell which half they have",
        any("a-b200" in g for g in got),
        str(got[:3]),
    )


def _unit_staleness_and_build():
    """STALE-VS-LIVE + BUILD IDENTITY — the other half of the B200 report: PARKED servers that had been
    deleted 20 minutes earlier, and `0 free` on a cluster whose nodes were empty.

    Under --watch a cluster that misses the frame deadline is re-rendered from its LAST GOOD frame. That
    frame's meta still says OK, and the hierarchy pane (the DEFAULT under --watch) only ever printed
    `· connected` — `_fresh_note` was wired into the FLAT pane alone. So old GPU/run state was indistinguish-
    able from live. fleet's whole job is answering "is this cluster busy"; a stale frame that cannot be told
    from a live one answers it backwards."""
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    now = fr._parse_ts(NOW)

    check(
        "stale: a fresh frame gets NO badge",
        fr._stale_badge("") == "",
        fr._stale_badge(""),
    )
    check(
        "stale: an aged last-good frame is badged ⚠ STALE with its age",
        "⚠ STALE" in fr._stale_badge("180") and "3m" in fr._stale_badge("180"),
        fr._stale_badge("180"),
    )
    check(
        "stale: a never-yet-seen cluster reads …refreshing (no age to claim)",
        fr._stale_badge("pending") == " · …refreshing",
        fr._stale_badge("pending"),
    )

    def _pane(stale):
        e = fr.build_cluster(
            "example-gpu-cluster",
            "ctxA",
            "ours",
            items([]),
            items([]),
            items([]),
            items([node("n1", 8)]),
            items([]),
            now=now,
            pvcs_j={"items": []},
        )
        e["gpu_type"], e["profiles"], e["readiness"], e["installs"] = (
            "B200",
            2,
            None,
            {"llm-perf/x": {}},
        )
        e["stale"] = stale
        return ANSI.sub(
            "",
            fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True),
        )

    stale_pane, live_pane = _pane("180"), _pane("")
    check(
        "stale (stages): the CLUSTER bar itself carries ⚠ STALE — the pane can't pass for live",
        "⚠ STALE" in stale_pane and "last live 3m ago" in stale_pane,
        stale_pane,
    )
    check(
        "stale (stages): a line says IN WORDS that CAPACITY/RUN below are not live",
        "NOT live" in stale_pane,
        stale_pane,
    )
    check(
        "stale (stages): a LIVE cluster gains no staleness noise (signal stays cheap)",
        "STALE" not in live_pane and "NOT live" not in live_pane,
        live_pane,
    )

    # BUILD STAMP — this repo is worked from ~40 worktrees; two sessions were lost to "that build predates
    # the feature". The pane now names the build that produced it.
    check(
        "build-stamp: build_line renders the stamp",
        fr.build_line("abc12345 · br · /w") == "fleet build abc12345 · br · /w",
        fr.build_line("abc12345 · br · /w"),
    )
    check(
        "build-stamp: an unknown build adds NO footer line (never a fake identity)",
        fr.build_line("") == "" and fr.build_line("   ") == "",
        "",
    )
    e = fr.build_cluster(
        "c",
        "ctxA",
        "ours",
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=now,
        pvcs_j={"items": []},
    )
    e["gpu_type"], e["profiles"], e["stale"], e["readiness"], e["installs"] = (
        "B200",
        1,
        "",
        None,
        {},
    )
    for st in (True, False):
        p = ANSI.sub(
            "",
            fr.render(
                [e],
                now,
                wide=False,
                gpu_only=False,
                color=False,
                stages=st,
                build="abc12345 · nvidia/llmb-k8s · /w",
            ),
        )
        check(
            f"build-stamp: the {'stages' if st else 'flat'} pane footers the build that produced it",
            "fleet build abc12345 · nvidia/llmb-k8s · /w" in p,
            p,
        )


def _unit_unattributed_gpu():
    """THIRD GPU BUCKET: a GPU pod in OUR OWN namespace that fleet cannot attribute is neither ours nor a
    foreign tenant's. Folding it into FOREIGN both under-reports our footprint and inflates apparent foreign
    usage, so the operator reads "the cluster is busy with someone else's work" when it is their own.
    Modelled on the real example-gpu-cluster case: a hand-rolled `nemotron-ultra-50k2k-sglang-dynamo-disagg-1p1d`
    Deployment in example-benchmark holding 8 B200s with only app.kubernetes.io/* labels — no llmb labels, no
    run-owner ownerRef, no matching name prefix."""
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    now = fr._parse_ts(NOW)

    third = pod(
        "nemotron-ultra-50k2k-sglang-dynamo-disagg-1p1d-0",
        gpus=8,
        ns="example-benchmark",
        managed=False,
        node="b-1",
    )
    third["metadata"]["labels"] = {
        "app.kubernetes.io/name": "nemotron-ultra-50k2k",
        "app.kubernetes.io/component": "worker",
        "app.kubernetes.io/part-of": "sglang-dynamo",
        "app.kubernetes.io/role": "decode",
    }
    foreign = pod("someoneelse-0", gpus=8, ns="teamZ", managed=False, node="b-2")
    ourjob = {
        "kind": "Job",
        "metadata": {
            "name": "j",
            "namespace": "example-benchmark",
            "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"},
        },
        "status": {},
    }

    occ, fgn, unattr = fr.gpu_occupied_and_foreign(items([third, foreign]), fr.is_llmb, {"example-benchmark"})
    check(
        "unattributed-gpu: an unattributable GPU pod in OUR ns lands in the THIRD bucket, not foreign",
        (occ, fgn, unattr) == (16, 8, 8),
        f"occ={occ} foreign={fgn} unattributed={unattr}",
    )
    check(
        "unattributed-gpu: with no namespaces of ours, behaviour is unchanged (all not-ours is foreign)",
        fr.gpu_occupied_and_foreign(items([third, foreign]), fr.is_llmb, frozenset()) == (16, 16, 0),
        "",
    )
    check(
        "unattributed-gpu: an unreadable pods -A still degrades to (None, None, None)",
        fr.gpu_occupied_and_foreign(None, fr.is_llmb, {"x"}) == (None, None, None),
        "",
    )

    e = fr.build_cluster(
        "example-gpu-cluster",
        "ctxA",
        "example-benchmark",
        items([]),
        items([]),
        items([]),
        items([node("b-1", 8), node("b-2", 8)]),
        items([third, foreign]),
        now=now,
        all_deploys=[],
        all_jobs=[ourjob],
    )
    e["gpu_type"], e["profiles"], e["stale"] = "B200", 1, ""
    check(
        "unattributed-gpu: build_cluster derives our namespaces (configured ns + ns of our llmb workloads)",
        e["unattributed_gpu"] == 8 and e["foreign"] == 8,
        str((e["unattributed_gpu"], e["foreign"])),
    )
    cap = fr._capacity_line(e)
    check(
        "unattributed-gpu: CAPACITY surfaces it explicitly (`8 GPUs unattributed in ours`)",
        "8 GPUs unattributed in ours" in cap,
        cap,
    )
    check(
        "unattributed-gpu: the ours-matcher was NOT broadened — the pod is still not counted as ours",
        e["ours_gpu"] == 0,
        str(e["ours_gpu"]),
    )
    # anomaly-only: a cluster with nothing unattributed must not carry the token
    e2 = fr.build_cluster(
        "clean",
        "ctxA",
        "ours",
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=now,
        all_deploys=[],
        all_jobs=[],
    )
    e2["gpu_type"], e2["profiles"], e2["stale"] = "B200", 1, ""
    check(
        "unattributed-gpu: stays OFF the happy path (no token when nothing is unattributed)",
        "unattributed" not in fr._capacity_line(e2),
        fr._capacity_line(e2),
    )


def _unit_installed_ns_scoping():
    """INSTALLED IS SCOPED TO OUR NAMESPACES. The model-cache inventory is sourced from a cluster-wide
    `get pvc -A` (our per-worktree namespaces are DISCOVERED, not configured, so a per-ns read would miss
    them), but the cluster-wide READ was rendered as a cluster-wide VIEW: on the shared cluster-b cluster
    every one of 229 tenant namespaces holds a `shared-model-cache` PVC, so the pane grew 229 NAMESPACE blocks
    — 2081 lines — each with an empty RUN section and, worse, a `backfill: kubectl label pvc …` hint pointing
    at a COLLEAGUE'S 50Ti volume. Nobody should be nudged to relabel another team's storage.

    The scope is the SAME ownership set `build_cluster` derives for the unattributed-GPU split (configured ns
    + every ns holding our llmb-labelled workloads) — deliberately one notion, not two.
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    now = fr._parse_ts(NOW)

    def pvc(name, ns, size="50Ti", phase="Bound", labels=None):
        return {
            "metadata": {"name": name, "namespace": ns, "labels": labels or {}},
            "spec": {"resources": {"requests": {"storage": size}}},
            "status": {"phase": phase, "capacity": {"storage": size}},
        }

    ours, theirs = "example-benchmark", "team-b"
    allpvcs = {
        "items": [
            pvc("shared-model-cache", ours, size="50Ti"),
            pvc("shared-model-cache", theirs, size="50Ti"),
            pvc("shared-model-cache", "team-c", size="50Ti"),
        ]
    }

    # ── the pure function: unfiltered (None) is unchanged; an explicit set drops everything outside it ──
    check(
        "ns-scope: discover_model_caches with no scope is UNCHANGED (pure-unit default lists every ns)",
        {r["ns"] for r in fr.discover_model_caches(allpvcs, [])} == {ours, theirs, "team-c"},
        "",
    )
    scoped = fr.discover_model_caches(allpvcs, [], our_namespaces={ours})
    check(
        "ns-scope: a PVC in a FOREIGN namespace produces no inventory row at all",
        [r["ns"] for r in scoped] == [ours],
        str(scoped),
    )
    # THE MUTATION HINT IS GONE ENTIRELY (see fleet_render.backfill_hint): it told operators to write
    # `download-complete=true`, which the panel then rendered as a green ✓ naming the PVC — the remedy
    # manufactured the defect. What remains is the scoping guarantee it existed to satisfy: no row and no
    # footer may ever name a colleague's namespace or storage.
    check(
        "ns-scope: no row and no footer names a FOREIGN namespace's PVC — and none suggests mutating one",
        all(theirs not in r["why"] and "team-c" not in r["why"] for r in scoped)
        and theirs not in fr.backfill_hint(scoped, ours)
        and "kubectl" not in fr.backfill_hint(scoped, ours),
        str(scoped),
    )
    check(
        "ns-scope: an empty ownership set scopes to NOTHING (never silently falls back to cluster-wide)",
        fr.discover_model_caches(allpvcs, [], our_namespaces=set()) == [],
        "",
    )
    # The artifacts ROLLUP is scoped by the same set — a colleague's run output is not our leak to report.
    foreign_art = {"items": [pvc("their-cell-artifacts", theirs), pvc("our-cell-artifacts", ours)]}
    ra = fr.discover_model_caches(foreign_art, [], our_namespaces={ours}, live_claims=set())
    check(
        "ns-scope: a FOREIGN namespace's artifacts PVCs produce no rollup — not our storage to account for",
        [(r["kind"], r["ns"]) for r in ra] == [("artifacts", ours)] and "1 PVC" in ra[0]["why"],
        str(ra),
    )

    # ── build_cluster: ownership = configured ns + ns of our llmb workloads (ONE set, shared with the GPU
    #    accounting). A colleague's ns must not reach the renderer at all. ──
    ourjob = {
        "kind": "Job",
        "metadata": {
            "name": "j",
            "namespace": "llmb-worktree-2",
            "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"},
        },
        "status": {},
    }
    e = fr.build_cluster(
        "cluster-b",
        "ctxA",
        ours,
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=now,
        all_deploys=[],
        all_jobs=[ourjob],
        pvcs_j={"items": allpvcs["items"] + [pvc("wt2-cache", "llmb-worktree-2")]},
    )
    e["gpu_type"], e["profiles"], e["stale"], e["readiness"], e["installs"] = (
        "B200",
        1,
        "",
        None,
        {},
    )
    check(
        "ns-scope: build_cluster exposes the ONE ownership set (configured ns + our llmb workloads' ns)",
        e["our_namespaces"] == [ours, "llmb-worktree-2"],
        str(e["our_namespaces"]),
    )
    check(
        "ns-scope: a DISCOVERED per-worktree ns of ours is still inventoried (scoping ≠ configured-ns-only)",
        {d["ns"] for d in e["discovered"]} == {ours, "llmb-worktree-2"},
        str(e["discovered"]),
    )
    plain = ANSI.sub("", fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True))
    check(
        "ns-scope (e2e): a foreign namespace renders NO `━━ NAMESPACE` block",
        "NAMESPACE team-b" not in plain and "NAMESPACE team-c" not in plain,
        plain,
    )
    check(
        "ns-scope (e2e): OUR namespaces still render their blocks",
        f"NAMESPACE {ours}" in plain and "NAMESPACE llmb-worktree-2" in plain,
        plain,
    )
    check(
        "ns-scope (e2e): exactly as many NAMESPACE blocks as namespaces we own",
        plain.count("━━ NAMESPACE") == 2,
        plain,
    )

    # ── REGRESSION GUARD: scoping must not touch the honest inventory states inside OUR namespace ──
    PIN = "abc123def4560000000000000000000000000000"
    S = {
        "llmb.nvidia.com/download-complete": "true",
        "llmb.nvidia.com/model-name": "glm5-fp8",
        "llmb.nvidia.com/model-revision": "abc123def456",
    }
    mixed = fr.discover_model_caches(
        {
            "items": [
                pvc("stamped", ours, size="721Gi", labels=S),
                pvc("bare", ours, size="900Gi"),
                pvc("pending", ours, phase="Pending"),
                pvc("stamped", theirs, labels=S),
            ]
        },
        [],
        our_namespaces={ours},
        pins={"glm5-fp8": PIN},
    )
    check(
        "ns-scope: EVERY honest state survives inside our ns (✓ Job-stamped · ⚠ unattributed · ✗ unbound)",
        [(r["state"], r["ns"]) for r in mixed] == [("ready", ours), ("warn", ours), ("failed", ours)],
        str(mixed),
    )

    # …and the UNREAD-vs-EMPTY distinction is untouched by scoping (an unread list is still UNKNOWN, and a
    # scoped-to-empty list is still an honest empty — the flag comes from the READ, not from the filter).
    e_no = fr.build_cluster(
        "c",
        "ctxA",
        ours,
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=now,
        pvcs_j=None,
    )
    e_fg = fr.build_cluster(
        "c",
        "ctxA",
        ours,
        items([]),
        items([]),
        items([]),
        items([node("n1", 8)]),
        items([]),
        now=now,
        pvcs_j={"items": [pvc("shared-model-cache", theirs)]},
    )
    check(
        "ns-scope: filtering out every FOREIGN PVC is not an unread inventory (read landed = not UNKNOWN)",
        e_no["inventory_unavailable"] is True and e_fg["inventory_unavailable"] is False,
        f"{e_no['inventory_unavailable']}/{e_fg['inventory_unavailable']}",
    )

    # ── the GPU bucket must be UNAFFECTED: a foreign GPU pod is still foreign, ours still ours ──
    theirpod = pod("their-worker-0", gpus=8, ns=theirs, managed=False, node="n2")
    e_gpu = fr.build_cluster(
        "cluster-b",
        "ctxA",
        ours,
        items([]),
        items([]),
        items([]),
        items([node("n1", 8), node("n2", 8)]),
        items([theirpod]),
        now=now,
        all_deploys=[],
        all_jobs=[ourjob],
        pvcs_j=allpvcs,
    )
    check(
        "ns-scope: the unattributed/foreign GPU accounting is untouched by the inventory scoping",
        (e_gpu["occupied"], e_gpu["foreign"], e_gpu["unattributed_gpu"]) == (8, 8, 0),
        str((e_gpu["occupied"], e_gpu["foreign"], e_gpu["unattributed_gpu"])),
    )


def _unit_pvc_roles_and_backfill():
    """A PVC's ROLE, and the ONE place the model-only backfill remedy may appear.

    THE 82-ROW BUG. A namespace holds three kinds of PVC: the model cache (weights), one `<cell>-artifacts`
    output volume PER RUN (50Gi, created by submit/sweep, deleted by nothing), and the RWX
    `llmb-control` wrapper-state volume. Fleet called all three `⚠ model: … contents unverified` and offered on
    each a `kubectl label pvc <62-char-name> llmb.nvidia.com/download-complete=true`. On the real GB300 campaign
    namespace that was 81 rows, 80 of them wrong — and the hint on an artifacts volume is not noise but an
    invitation to stamp a FALSE claim ("these are the weights, the download finished") onto run OUTPUT.

    Artifacts are ROLLED UP, not silently filtered: one line per namespace carrying the count, the bytes and
    how many belong to runs that no longer exist. Filtering would have fixed the rows and hidden the finding —
    that namespace leaks one 50Gi PVC per run, forever (77 of 81 orphaned when this test was written).
    """
    sys.path.insert(0, str(SCRIPTS))
    import fleet_render as fr

    now = fr._parse_ts(NOW)
    OURS = "llmb-serving-gb300"

    def pvc(name, ns=OURS, phase="Bound", size="50Gi", labels=None):
        return {
            "metadata": {"name": name, "namespace": ns, "labels": labels or {}},
            "spec": {"resources": {"requests": {"storage": size}}},
            "status": {"phase": phase, "capacity": {"storage": size}},
        }

    # ── role classification: by NAME, because no creator labels these volumes ──
    check(
        "pvc-role: `<cell>-artifacts` is run OUTPUT, `llmb-control` is control state, the rest is a cache",
        (
            fr._pvc_role("nemotron-ultra-3-gb300-workload-stable16-pareto-r11cc-artifacts"),
            fr._pvc_role("llmb-control"),
            fr._pvc_role("serving-gb300-model-cache"),
        )
        == ("artifacts", "control", "cache"),
        "",
    )

    # ── ONE definition of "this is run output", shared with the tool that DELETES it ──
    # reclaim_storage.py SWEEPS artifacts PVCs; fleet REPORTS them. A disagreement means the pane either
    # flags a volume the sweeper will not touch, or stays silent about one it will. The two differ on purpose
    # only in the RESIDUE: the sweeper buckets unrecognised volumes as 'other' and refuses to delete them
    # (right for a destructive tool), fleet reports them as a cache with an honest ⚠ (right for a reporting
    # one). The ARTIFACTS and CONTROL verdicts must be identical, and are pinned here.
    import reclaim_storage as rs

    check(
        "pvc-role: fleet's artifacts suffixes are EXACTLY the sweeper's (what we report = what it deletes)",
        tuple(fr.ARTIFACTS_PVC_SUFFIXES) == tuple(rs.ARTIFACT_SUFFIXES),
        f"{fr.ARTIFACTS_PVC_SUFFIXES} vs {rs.ARTIFACT_SUFFIXES}",
    )
    for nm in (
        "cell-artifacts",
        "llmb-benchmark-artifacts-rwx",
        "llmb-control",
        "serving-gb300-model-cache",
        "nemotron-ultra-hf-cache",
        "serving-cluster-b-results",
    ):
        theirs, mine = rs.classify({"metadata": {"name": nm}}), fr._pvc_role(nm)
        check(
            f"pvc-role: fleet and the sweeper agree on `{nm}` (artifacts + control verdicts identical)",
            (mine == "artifacts") == (theirs == "artifacts") and (mine == "control") == (theirs == "control"),
            f"fleet={mine} sweeper={theirs}",
        )

    # ── 3 caches + 1 control + 20 artifacts → 3 cache rows + 1 control row + ONE rollup, not 24 model rows ──
    arts = [pvc(f"cell-r{i}-artifacts") for i in range(20)]
    caches = [
        pvc("serving-gb300-model-cache", size="1200Gi"),
        pvc("cache-r2", size="1200Gi"),
        pvc("cache-r3", size="1200Gi"),
    ]
    ctrl = [pvc("llmb-control", size="1200Gi")]
    live = {
        (OURS, "cell-r0-artifacts"),
        (OURS, "cell-r1-artifacts"),
    }  # only 2 runs still exist
    rows = fr.discover_model_caches({"items": arts + caches + ctrl}, [], our_namespaces={OURS}, live_claims=live)
    kinds = collections.Counter(r["kind"] for r in rows)
    check(
        "pvc-role: 20 per-run artifacts PVCs collapse to ONE rollup row (was 20 bogus `model:` rows)",
        kinds == {"cache": 3, "control": 1, "artifacts": 1},
        str(kinds),
    )
    check(
        "pvc-role: NO artifacts PVC is ever emitted as a model-cache row",
        not any(r["kind"] == "cache" and r["name"].endswith("-artifacts") for r in rows),
        str(rows),
    )
    roll = [r for r in rows if r["kind"] == "artifacts"][0]
    check(
        "pvc-role: the rollup NAMES what these volumes are, so they cannot be misread as weights",
        "run OUTPUT, not model weights" in roll["why"],
        roll["why"],
    )
    check(
        "pvc-role: the rollup SURFACES the leak (18 of 20 belong to runs that no longer exist) + the bytes",
        "20 PVCs" in roll["why"]
        and "18 from runs no longer present" in roll["why"]
        and "1000Gi" in roll["why"]
        and roll["state"] == "warn",
        roll["why"],
    )
    check(
        "pvc-role: leaked run output is ⚠, NOT ✓ — a silent filter would have hidden a growing leak",
        roll["state"] == "warn",
        str(roll),
    )
    clean = fr.discover_model_caches(
        {"items": arts},
        [],
        our_namespaces={OURS},
        live_claims={(OURS, p["metadata"]["name"]) for p in arts},
    )
    check(
        "pvc-role: artifacts all accounted for by live runs read ✓ (no false leak alarm)",
        [r["state"] for r in clean] == ["ready"] and "no longer present" not in clean[0]["why"],
        str(clean),
    )

    # ── A CORPSE IS NOT A REFERENCE. A Succeeded pod / completed Job is the RESIDUE of a finished run; the
    #    namespace keeps it until GC. Counting it live was the difference between "71 leaked" and "0 leaked"
    #    on the real GB300 namespace, where 128 lingering pods named 82 artifacts claims between them. ──
    def _pod(nm, phase, claim):
        return {
            "kind": "Pod",
            "metadata": {"name": nm, "namespace": OURS},
            "status": {"phase": phase},
            "spec": {"volumes": [{"name": "a", "persistentVolumeClaim": {"claimName": claim}}]},
        }

    def _job(nm, claim, **st):
        return {
            "kind": "Job",
            "metadata": {"name": nm, "namespace": OURS},
            "status": st,
            "spec": {"template": {"spec": {"volumes": [{"name": "a", "persistentVolumeClaim": {"claimName": claim}}]}}},
        }

    refs = fr.referenced_claims(
        [
            _pod("live", "Running", "a-artifacts"),
            _pod("done", "Succeeded", "b-artifacts"),
            _pod("dead", "Failed", "c-artifacts"),
            _job("jactive", "d-artifacts", active=1),
            _job("jdone", "e-artifacts", succeeded=1),
            {
                "kind": "Deployment",
                "metadata": {"namespace": OURS},
                "status": {},
                "spec": {
                    "replicas": 0,
                    "template": {
                        "spec": {
                            "volumes": [
                                {
                                    "name": "a",
                                    "persistentVolumeClaim": {"claimName": "f-artifacts"},
                                }
                            ]
                        }
                    },
                },
            },
        ]
    )
    check(
        "pvc-role: a finished run's CORPSE (Succeeded/Failed pod, completed Job) is not a live reference",
        {c for _, c in refs} == {"a-artifacts", "d-artifacts", "f-artifacts"},
        str(sorted(refs)),
    )
    check(
        "pvc-role: a PARKED Deployment (replicas=0) still holds its artifacts volume — an installed cell",
        (OURS, "f-artifacts") in refs,
        str(sorted(refs)),
    )
    kindless = dict(_pod("done", "Succeeded", "h-artifacts"))
    kindless.pop("kind")  # some kubectl versions omit `kind` on a single-kind list's items
    check(
        "pvc-role: a kind-less Succeeded pod is still a corpse (the leak signal cannot silently switch off)",
        fr.referenced_claims([kindless]) == set(),
        str(fr.referenced_claims([kindless])),
    )
    check(
        "pvc-role: an unrecognised object kind counts as LIVE (fail safe — never accuse)",
        fr.referenced_claims(
            [
                {
                    "kind": "StatefulSet",
                    "metadata": {"namespace": OURS},
                    "spec": {
                        "template": {
                            "spec": {
                                "volumes": [
                                    {
                                        "name": "a",
                                        "persistentVolumeClaim": {"claimName": "g-artifacts"},
                                    }
                                ]
                            }
                        }
                    },
                }
            ]
        )
        == {(OURS, "g-artifacts")},
        "",
    )

    # ── UNKNOWN ≠ ZERO: no workload read → assert no leak count at all ──
    unk = fr.discover_model_caches({"items": arts}, [], our_namespaces={OURS}, live_claims=None)
    check(
        "pvc-role: with the workload read missing, AGE is unknown — no leak count is invented",
        "no longer present" not in unk[0]["why"] and "age unknown" in unk[0]["why"],
        str(unk),
    )
    # …and the gate that FEEDS it. `_split_llmb` turns a FAILED cluster-wide llmb read into ([], []) — the
    # same shape as "we own no workloads". Without an explicit read-landed flag, one timed-out call would
    # reclassify every artifacts PVC as leaked and accuse a LIVE campaign of leaking what it is writing to.
    live_dep = {
        "kind": "Deployment",
        "metadata": {
            "name": "d",
            "namespace": OURS,
            "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"},
        },
        "spec": {
            "template": {
                "spec": {
                    "volumes": [
                        {
                            "name": "a",
                            "persistentVolumeClaim": {"claimName": "cell-r0-artifacts"},
                        }
                    ]
                }
            }
        },
        "status": {},
    }
    kw = dict(now=now, pvcs_j={"items": arts}, all_deploys=[live_dep], all_jobs=[])
    e_ok = fr.build_cluster(
        "c",
        "ctxA",
        OURS,
        items([]),
        items([]),
        items([]),
        items([node("n1", 4)]),
        items([]),
        workloads_read=True,
        **kw,
    )
    e_no = fr.build_cluster(
        "c",
        "ctxA",
        OURS,
        items([]),
        items([]),
        items([]),
        items([node("n1", 4)]),
        items([]),
        workloads_read=False,
        **kw,
    )
    w_ok = [d["why"] for d in e_ok["discovered"] if d["kind"] == "artifacts"][0]
    w_no = [d["why"] for d in e_no["discovered"] if d["kind"] == "artifacts"][0]
    check(
        "pvc-role: a FAILED cluster-wide workload read yields NO leak count (never a false accusation)",
        "no longer present" in w_ok
        and "19 from runs" in w_ok
        and "no longer present" not in w_no
        and "age unknown" in w_no,
        f"{w_ok} || {w_no}",
    )

    # ── a Pending artifacts PVC UNDER A LIVE RUN is a real failure and must not be averaged away ──
    blocked = fr.discover_model_caches(
        {"items": [pvc("live-cell-artifacts", phase="Pending")] + arts[:3]},
        [],
        our_namespaces={OURS},
        live_claims={(OURS, "live-cell-artifacts")},
    )
    check(
        "pvc-role: an unbound artifacts PVC a LIVE run needs reads ✗ (results cannot be written)",
        blocked[0]["state"] == "failed" and "results cannot be written" in blocked[0]["why"],
        str(blocked),
    )

    # ── the control PVC: its own row, no download verdict, and the Pending signal preserved verbatim ──
    ok = [r for r in rows if r["kind"] == "control"][0]
    bad = fr.discover_model_caches(
        {"items": [pvc("llmb-control", phase="Pending", size="1200Gi")]},
        [],
        our_namespaces={OURS},
    )[0]
    check(
        "pvc-role: llmb-control is control state, never a model — ✓ Bound, ✗ Pending, no download claim",
        (ok["state"], bad["state"]) == ("ready", "failed")
        and "run control plane" in ok["why"]
        and "Pending" in bad["why"]
        and "unverified" not in ok["why"],
        str((ok, bad)),
    )

    # ── THE BACKFILL HINT IS DELETED, AND THAT IS THE FIX ──────────────────────────────────────────────
    # It printed `kubectl -n <ns> label pvc <PVC> llmb.nvidia.com/download-complete=true`. Follow it and the
    # PVC ends up stamped complete with NO model-name — which the panel rendered as
    # `✓ model: <the PVC's own name> @<40-char label> … verified by PVC stamp`. The panel's own remedy
    # manufactured the panel's own bug, and it is the likeliest provenance of the reported nemotron row.
    # There is no kubectl one-liner that turns a label into evidence about 700 GiB, so none is offered.
    check(
        "backfill: NO row anywhere carries a download-complete mutation hint (the hint slot is gone)",
        all(not r.get("hint_pvc") for r in rows),
        str(rows),
    )
    check(
        "backfill: the kubectl line is NOT repeated on any row (80 copies was the unreadable half)",
        all("kubectl" not in r["why"] for r in rows),
        str(rows),
    )
    for h in (
        fr.backfill_hint(rows, OURS),
        fr.backfill_hint([{"state": "present", "model": "m"}], OURS),
        fr.backfill_hint([{"state": "present"}] * 3, OURS),
    ):
        check(
            "backfill: whatever the input, the remediation line contains NO label write at all",
            "kubectl" not in h and "label pvc" not in h and "download-complete" not in h,
            h,
        )
    check(
        "backfill: unvouched models get an honest advisory naming what is NOT proven",
        fr.backfill_hint([{"state": "present"}] * 2, OURS)
        == "↳ 2 models present but unvouched — weights may be complete; nothing proves it.",
        fr.backfill_hint([{"state": "present"}] * 2, OURS),
    )
    check(
        "backfill: nothing unvouched (or nothing but storage rows) → NO line at all",
        fr.backfill_hint(clean, OURS) == "" and fr.backfill_hint([], OURS) == "",
        "",
    )

    # ── E2E on the reported shape: 81 artifacts + 5 caches + control in ONE namespace, plus a colleague's ──
    def art(i):
        return pvc(f"nemotron-ultra-3-gb300-workload-stable16-pareto-r{i:04x}-artifacts")

    e = fr.build_cluster(
        "example-gb300-cluster",
        "ctxA",
        OURS,
        items([]),
        items([]),
        items([]),
        items([node("n1", 4)]),
        items([]),
        now=now,
        all_deploys=[],
        all_jobs=[],
        pvcs_j={
            "items": [art(i) for i in range(81)]
            + [pvc(f"serving-gb300-model-cache{s}", size="1200Gi") for s in ("", "-r2", "-r3", "-r4", "-r5")]
            + [
                pvc("llmb-control", size="1200Gi"),
                pvc("shared-model-cache", ns="team-b"),
                pvc("their-run-artifacts", ns="team-c"),
            ]
        },
    )
    e["gpu_type"], e["profiles"], e["stale"], e["readiness"], e["installs"] = (
        "GB300",
        1,
        "",
        None,
        {},
    )
    plain = ANSI.sub("", fr.render([e], now, wide=False, gpu_only=False, color=False, stages=True))
    check(
        "82-row bug (e2e): 87 PVCs render as 7 STORAGE rows — 5 caches + control + ONE artifacts rollup",
        plain.count("⚠ claim:") == 5
        and plain.count("ARTIFACTS") == 1
        and plain.count("✓ llmb-control") == 1
        and "81 PVCs" in plain
        and "run OUTPUT, not model weights" in plain,
        plain,
    )
    check(
        "82-row bug (e2e): every PVC is accounted for in the STORAGE header, bucketed by role",
        "87 PVCs" in plain and "5 model caches · 1 control · 81 run-output" in plain,
        plain,
    )
    check(
        "82-row bug (e2e): a claim is never called a model — the word `model:` names nothing here",
        "model: " not in plain,
        plain,
    )
    check(
        "82-row bug (e2e): not one artifacts PVC name survives as its own row",
        "-artifacts " not in plain and plain.count("-artifacts") == 0,
        plain,
    )
    check(
        "82-row bug (e2e): NO kubectl mutation appears anywhere in the pane (the hint is gone, not moved)",
        plain.count("kubectl label") == 0 and plain.count("label pvc") == 0,
        plain,
    )
    check(
        "82-row bug (e2e): a COLLEAGUE'S namespace gets no block, no rollup and no kubectl hint",
        "team-b" not in plain and "team-c" not in plain and "their-run-artifacts" not in plain,
        plain,
    )


def main() -> int:
    _unit_node_capacity()
    _unit_capacity_branch()
    _unit_model_load_queue()
    _unit_model_caches()
    _unit_model_ledger()
    _unit_installed_ns_scoping()
    _unit_pvc_roles_and_backfill()
    _unit_collapse_safety()
    _unit_schedule_states()
    _unit_audit_join_and_absence()
    _unit_audit_render_honesty()
    _unit_staleness_and_build()
    _unit_unattributed_gpu()
    _unit_stages()
    _unit_ns_installed_hierarchy()
    _unit_sweep_progress()
    _unit_installed_inventory()
    _unit_live_discovery()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = write_fixtures(root)
        shim_dir = write_shim(root, fx)
        profiles = write_profiles(root)
        # A self-contained fake recipes/ tree so make_result_lookup resolves a KNOWN ✓-done metric OFFLINE
        # (decoupled from mutable real repo data). FLEET_RECIPES_ROOT points fleet_render's result lookup here;
        # a metriccell/runs.jsonl row carries {run_id, metric, value} exactly like a real collected run.
        rec = root / "fake_recipes" / "recipes" / "llm-perf" / "metriccell"
        rec.mkdir(parents=True)
        (rec / "runs.jsonl").write_text(
            json.dumps(
                {
                    "run_id": "md1",
                    "date": "2026-07-19",
                    "metric": "net_behavior_score",
                    "value": 0.72,
                    "wall_seconds": 3000,
                    "gpu_count": 8,
                }
            )
            + "\n"
        )
        os.environ["FLEET_RECIPES_ROOT"] = str(root / "fake_recipes")

        out = run_fleet(profiles, shim_dir)
        plain = ANSI.sub("", out)
        lines = plain.splitlines()
        line1 = lines[0] if lines else ""
        alpha, bravo, charlie = (
            _cblock(plain, "alpha"),
            _cblock(plain, "bravo"),
            _cblock(plain, "charlie"),
        )
        hotel, foxtrot, delta_sec = (
            _cblock(plain, "hotel"),
            _cblock(plain, "foxtrot"),
            _cblock(plain, "delta"),
        )

        # ── LINE 1 answers the operator's questions at a glance ──────────────────────────────────────
        check("summary line leads with ACTIVE", line1.startswith("ACTIVE "), line1)
        # OURS is CLUSTER-WIDE: alpha 8 + bravo 20 + charlie 6 + hotel 16 + golf's cross-ns loader 4 = 54
        # (golf's 4 GPUs are held by OUR pod in another namespace, discovered from pods -A → they ARE ours).
        check(
            "summary: OURS grand total = 54 (single-ns 50 + golf's cross-ns 4)",
            "54 GPUs (ours)" in line1,
            line1,
        )
        # soonest run is `dead@bravo` (deadline-sourced) → the headline marks it a deadline CAP (≤), never a
        # median ETA, so a wall-clock ceiling is not misread as an expected finish.
        check(
            "summary: soonest run shown as a deadline cap (≤ … to deadline cap), not a fake ~ETA",
            "⏱ soonest" in line1 and "≤" in line1 and "deadline cap" in line1 and "~ETA" not in line1,
            line1,
        )
        check(
            "summary: fleet up/auth breakdown (6 up · 2 auth✗)",
            "6 up" in line1 and "2 auth✗" in line1,
            line1,
        )
        # OCCUPIED counts every Running GPU pod cluster-wide (pods -A): the prior 64 + golf's glm5 loader (4) = 68.
        check("summary: fleet GPU used (68/72)", "68/72 GPUs used" in line1, line1)
        check(
            "GPU-type breakdown line (2×B200 · 1×GB200 · 1×GB300)",
            "2×B200 · 1×GB200 · 1×GB300" in plain,
            lines[1] if len(lines) > 1 else "",
        )

        # ── NESTED TREE: cluster header (rule) → namespace line → jobs → servers ──────────────────────
        check(
            "cluster header is a rule with name + [TYPE]",
            alpha.splitlines()[0].startswith("━━ alpha") and "[B200]" in alpha.splitlines()[0],
            alpha.splitlines()[0] if alpha else "",
        )
        # Capacity HEADLINE is free WHOLE NODES (the schedulable unit), not free GPUs. alpha is a fragmentation
        # case: 12/16 gpu used → 4 GPUs 'free', but both nodes carry something (gpu-node-1 full, gpu-node-2 half)
        # → 0 free whole-nodes, biggest single free node only 4g. The raw gpu total is kept terse as a tail.
        check(
            "namespace line headlines free NODES not free GPUs (nodes 0/2 free · biggest free: 4 GPUs)",
            alpha.splitlines()[1].strip().startswith("ns ours") and "nodes 0/2 free (biggest free: 4 GPUs)" in alpha,
            alpha.splitlines()[1] if alpha else "",
        )
        check(
            "namespace line keeps the raw gpu total terse (12/16 gpu) + ours in nodes (ours 1 node (8 GPUs))",
            "12/16 GPUs" in alpha and "ours 1 node (8 GPUs)" in alpha,
            alpha.splitlines()[1] if alpha else "",
        )
        check(
            "a column header row is present",
            "STATUS" in alpha and "RUN / SERVER" in alpha and "GPU" in alpha and "AGE / EXPECTED" in alpha,
            alpha,
        )

        # ── ALIGNED COLUMNS — the main ask: every data row shares the NAME + GPU column offsets ───────
        check(
            "bravo table columns are ALIGNED (name + gpu offsets identical across rows)",
            _aligned(bravo),
            bravo,
        )
        check(
            "alpha table columns are ALIGNED (job + its nested server line up)",
            _aligned(alpha),
            alpha,
        )

        # ── the active RUN and its SERVERS nested beneath it ──────────────────────────────────────────
        # run rows show the FULL cell-clone name + run-id, un-truncated (no `…`): `modelx-r1c1`
        _alpha_run = next((l for l in alpha.splitlines() if "modelx-r1c1" in l), "")
        check(
            "active run row shows the FULL run name un-truncated (modelx-r1c1), 8g, 41%",
            bool(_alpha_run) and "…" not in _alpha_run and "8 GPUs" in _alpha_run and "41%" in _alpha_run,
            _alpha_run,
        )
        _alpha_svc = next((l for l in alpha.splitlines() if "└ svc" in l), "")
        check(
            "server nested under its run with its FULL Deployment name (└ svc modelx-server)",
            bool(_alpha_svc) and "modelx-server" in _alpha_svc,
            _alpha_svc,
        )
        check(
            "run named by cell/recipe + run-id, not the raw Job name",
            "modelx-bench-r1c1" not in alpha,
            alpha,
        )
        check(
            "loady's server nested (└ svc loady-server)",
            "└ svc" in bravo and "loady-server" in bravo,
            bravo,
        )
        check(
            "over-median run flagged",
            "over" in bravo and "⚠ over-median" in bravo,
            bravo,
        )
        check(
            "near-deadline run flagged",
            "dead" in bravo and "⚠ near-deadline" in bravo,
            bravo,
        )
        check(
            "STUCK (wedged) server surfaces as a run row",
            "STUCK" in bravo and "crash-server" in bravo,
            bravo,
        )
        # ── SWEEP column = concurrency rungs (any launch path), NOT the submit-only variance sweep-id ────
        check(
            "SWEEP column shows concurrency rungs from CONCURRENCIES env (16·32·64)",
            "16·32·64" in bravo,
            bravo,
        )
        check("SWEEP column header present", "SWEEP" in bravo, bravo)

        # ── control-plane / bare servers are NOT rows — they collapse to the footer counts ────────────
        for bare in (
            "park-frontend",
            "fresh-server",
            "small-server",
            "c240-decode",
            "park-decode",
            "orphan-server",
        ):
            check(f"collapsed server '{bare}' is NOT a tree row", bare not in plain)
        check(
            "charlie footer: no active runs of ours · 2 parked-runs · 1 infra-pod · orphan(2g held)",
            "no active runs of ours" in charlie
            and "2 parked-run" in charlie
            and "1 infra-pod" in charlie
            and "orphan(2 GPUs held)" in charlie,
            charlie,
        )
        check(
            "alpha footer collapses idle-servers (3 idle-servers)",
            "3 idle-server" in alpha,
            alpha,
        )
        check(
            "bravo footer splits ✓done / orphan(8g held)",
            "✓1 done" in bravo and "orphan(8 GPUs held)" in bravo,
            bravo,
        )

        # ── LIFECYCLE / HISTORY: a recently-FAILED run must be VISIBLE (never hidden as 'idle') ────────
        check(
            "recently-FAILED run surfaces as a ✗ FAILED history row (full name failcell-rf1)",
            "✗ FAILED" in charlie and "failcell-rf1" in charlie,
            charlie,
        )
        check("failed run shows when it ended (ended … ago)", "ended" in charlie, charlie)
        check(
            "line1 surfaces fleet-wide recent failures (⚠ N recently FAILED)",
            "recently FAILED" in line1,
            line1,
        )
        # ── governor housekeeping cron is NOT counted as a benchmark run/done ──────────────────────────
        check(
            "governor cron excluded from run rows/counts (no llmb-governor row)",
            "llmb-governor" not in plain,
            plain[:0],
        )
        check(
            "governor cron NOT counted as a benchmark 'done' in charlie",
            "✓1 done" not in charlie,
            charlie,
        )
        # ── older terminal run beyond the default 1h window is NOTED, not silently dropped ─────────────
        check(
            "older terminal run noted, not silently truncated (--history hint)",
            "older run" in bravo and "--history" in bravo,
            bravo,
        )

        # ── TRIAGE: a ✗ FAILED row must say WHY it died + the exact one-liner to dig in ────────────────
        # (a) cause from the Job's own Failed condition reason:
        check(
            "✗ FAILED row shows the cause from the Job condition (why: DeadlineExceeded)",
            "why: DeadlineExceeded" in charlie,
            charlie,
        )
        check(
            "✗ FAILED row shows the exact investigate one-liner (logs: llmb-k8s logs rf1)",
            "logs: llmb-k8s logs rf1" in charlie,
            charlie,
        )
        # (b) cause from a pod's OOMKilled lastState (pod signal wins over any generic condition):
        check(
            "✗ FAILED row shows the pod-level cause (why: OOMKilled) + its logs one-liner",
            "why: OOMKilled" in charlie and "logs: llmb-k8s logs ro1" in charlie,
            charlie,
        )
        # default view stays terse — the fuller condition message is held back for --detail:
        check(
            "default ✗ row is terse (full condition message withheld until --detail)",
            "Job was active longer than specified deadline" not in charlie,
            charlie,
        )

        # ── TRIAGE: a ✓ done row must say the RESULT + where it is ─────────────────────────────────────
        # (a) metric resolvable from the run's collected runs.jsonl → the headline number is shown:
        check(
            "✓ done row shows the resolved headline metric (net_behavior_score=0.72)",
            "net_behavior_score=0.72" in alpha,
            alpha,
        )
        # (b) metric NOT yet collected → honest pointer, never a fabricated number:
        check(
            "✓ done row (uncollected) degrades to a collect pointer (result: collect nc1)",
            "result: collect nc1" in alpha and "metriccell" in alpha,
            alpha,
        )

        # ── --detail deepens the triage: full ✗ message + ✓ artifacts/analyze pointer ──────────────────
        tdet = ANSI.sub("", run_fleet(profiles, shim_dir, "--detail"))
        td_charlie, td_alpha = _cblock(tdet, "charlie"), _cblock(tdet, "alpha")
        check(
            "--detail reveals the full ✗ failure message",
            "Job was active longer than specified deadline" in td_charlie,
            td_charlie,
        )
        check(
            "--detail adds the ✓ artifacts pointer next to the metric (collect md1)",
            "net_behavior_score=0.72" in td_alpha and "collect md1" in td_alpha,
            td_alpha,
        )
        check(
            "--detail points an uncollected ✓ run at analyze (analyze nc1)",
            "analyze nc1" in td_alpha,
            td_alpha,
        )

        # ── GPU TYPE per cluster ──────────────────────────────────────────────────────────────────
        check("alpha [B200]", "[B200]" in alpha, alpha)
        check("bravo [GB200]", "[GB200]" in bravo, bravo)
        check("charlie [GB300]", "[GB300]" in charlie, charlie)

        # ── CPU shown only when >0 ────────────────────────────────────────────────────────────────
        check("alpha ns line shows cpu 4c", "cpu 4c" in alpha, alpha)
        check("bravo ns line shows cpu 7c", "cpu 7c" in bravo, bravo)
        check("0-CPU cluster shows no cpu token", "cpu" not in foxtrot, foxtrot)

        # ── PHYSICAL-cluster dedup + [N profiles] ─────────────────────────────────────────────────
        check("shared context-string deduped (alpha-dup absent)", "alpha-dup" not in plain)
        check(
            "physical-cluster dedup: hotel-glm5 not a separate row",
            "hotel-glm5" not in plain,
        )
        check(
            "hotel [2 profiles] (teleport-ctx + KUBE_CLUSTER → example-gpu-cluster)",
            "2 profiles" in hotel,
            hotel,
        )
        check("alpha [2 profiles] (alpha + alpha-dup)", "2 profiles" in alpha, alpha)

        # ── OWNERSHIP: disagg pod with only {app,role}, llmb labels on the Deployment → OURS not FOREIGN ─
        # owner-aware attribution now surfaces in NODE terms: our unlabeled disagg pod holds one whole node.
        # hotel is fully full (h-1 ours 16/16, h-2 foreign 16/16) → 0 free nodes AND biggest free 0g (the
        # extreme 'nothing launchable' signal), 32/32 gpu.
        check(
            "owner-aware: unlabeled disagg pod counted OURS (ours 1 node (16 GPUs), not 0)",
            "ours 1 node (16 GPUs)" in hotel,
            hotel,
        )
        check(
            "owner-aware: cluster headlines free-nodes + terse gpu total (foreign implied, not repeated)",
            "nodes 0/2 free (biggest free: 0 GPUs)" in hotel
            and "32/32 GPUs" in hotel
            and "ours 1 node (16 GPUs)" in hotel
            and "foreign" not in hotel,
            hotel,
        )
        detail = ANSI.sub("", run_fleet(profiles, shim_dir, "--detail"))
        check(
            "owner-aware (--detail): disagg server row shows its 16 GPUs",
            "c240-decode" in _cblock(detail, "hotel") and "16 GPUs" in _cblock(detail, "hotel"),
            _cblock(detail, "hotel"),
        )

        # ── ORPHAN vs PARKED vs INFRA accuracy ────────────────────────────────────────────────────
        check(
            "parked (workers 0/0, control-plane up) collapses as parked, not orphan",
            "2 parked-run" in charlie and "park-decode" not in plain,
            charlie,
        )
        check(
            "control-plane collapses as INFRA (never a run row)",
            "1 infra-pod" in charlie,
            charlie,
        )
        check(
            "GENUINE orphan still counted (charlie small-server, 2g)",
            "orphan(2 GPUs held)" in charlie,
            charlie,
        )
        check(
            "bravo standalone orphan counted (8g)",
            "orphan(8 GPUs held)" in bravo,
            bravo,
        )

        # ── --detail expands the collapsed servers into the tree ──────────────────────────────────
        check(
            "--detail lists otherwise-collapsed servers (park-frontend infra)",
            "park-frontend" in detail or "small-server" in detail,
            "",
        )

        # ── cluster with nothing active is compact (header + ns + one footer line) ────────────────
        check(
            "idle cluster says 'no active runs of ours' (unambiguous)",
            "no active runs of ours" in foxtrot,
            foxtrot,
        )

        # ── auth failure isolated + CONNECT_CMD hint ──────────────────────────────────────────────
        check("delta shows auth✗", "auth✗" in delta_sec, delta_sec)
        check(
            "delta (no CONNECT_CMD) falls back to tsh login hint",
            "run `tsh kube login ctxBAD`" in delta_sec,
            delta_sec,
        )
        check("auth-fail did NOT abort others (alpha still rendered)", bool(alpha))
        echo_sec = _cblock(plain, "echo")
        check(
            "echo shows profile CONNECT_CMD verbatim",
            "run `tsh login --proxy=tp.example:443 && tsh kube login ctxBAD2`" in echo_sec,
            echo_sec,
        )
        check(
            "echo does NOT show the tsh fallback",
            "run `tsh kube login ctxBAD2`" not in echo_sec,
            echo_sec,
        )

        # ── REGRESSION: bash-3.2 `set -u` empty-array crash (ctxargs/nsargs unbound when a profile has ──
        # NO KUBE_CONTEXT / NO NAMESPACE). Before the ${arr[@]+"${arr[@]}"} guard this aborted the whole
        # command with `ctxargs[@]: unbound variable` before any cluster rendered.
        full = run_fleet_full(profiles, shim_dir)
        check(
            "no-context/no-namespace profile: fleet exits 0 (no unbound-variable crash)",
            full.returncode == 0,
            f"rc={full.returncode} stderr={full.stderr.strip()[:120]}",
        )
        check(
            "no unbound-variable error on stderr",
            "unbound variable" not in full.stderr,
            full.stderr.strip()[:120],
        )
        check(
            "ambient-context cluster (foxtrot, empty ctxargs) still renders",
            bool(_cline(plain, "foxtrot")),
        )
        check(
            "no-namespace cluster (golf, empty nsargs) still renders",
            bool(_cline(plain, "golf")),
        )
        # Exercise the entry once under the platform's real bash (macOS ships 3.2.57) to catch a 3.2-only
        # expansion regression that a newer bash would tolerate.
        sys_bash = "/bin/bash" if Path("/bin/bash").exists() else "bash"
        rb = run_fleet_full(profiles, shim_dir, bash_bin=sys_bash)
        ver = subprocess.run([sys_bash, "-c", "echo $BASH_VERSION"], capture_output=True, text=True).stdout.strip()
        check(
            f"runs clean under system bash ({ver or '?'})",
            rb.returncode == 0 and "unbound variable" not in rb.stderr,
            f"rc={rb.returncode} stderr={rb.stderr.strip()[:120]}",
        )

        # ── deterministic ordering ────────────────────────────────────────────────────────────────
        out2 = run_fleet(profiles, shim_dir)
        check("deterministic: two runs byte-identical", out == out2)

        # ── color handling ────────────────────────────────────────────────────────────────────────
        check("piped output has NO ANSI (color stripped)", "\033[" not in out)
        colored = run_fleet(profiles, shim_dir, "--color")
        check("--color emits ANSI", "\033[" in colored)

        # ── --gpu-only hides 0-GPU active runs (CPU-only benches) ─────────────────────────────────
        gbravo = _cblock(ANSI.sub("", run_fleet(profiles, shim_dir, "--gpu-only")), "bravo")
        check(
            "--gpu-only hides a 0-GPU run row (over → 0g, dropped)",
            "⚠ over-median" not in gbravo,
            gbravo,
        )
        check(
            "--gpu-only keeps GPU run rows (loady 4g nested server)",
            "loady-server" in gbravo,
            gbravo,
        )

        # ── --wide adds columns in the detail tree ────────────────────────────────────────────────
        wide = ANSI.sub("", run_fleet(profiles, shim_dir, "--detail", "--wide"))
        check(
            "--detail --wide adds node/img/rst columns",
            "node=gpu-node-1" in wide and "img=tag1" in wide,
        )

        # ── --cluster filter ──────────────────────────────────────────────────────────────────────
        only = ANSI.sub("", run_fleet(profiles, shim_dir, "--cluster", "bravo"))
        check(
            "--cluster bravo shows only bravo",
            bool(_cline(only, "bravo")) and not _cline(only, "alpha"),
        )

        # ── --watch REDRAWS IN PLACE (like top/watch): runs on the ALTERNATE SCREEN BUFFER + cursor-home +
        # clear-to-end each frame, so frames refresh instead of scrolling/appending, and the terminal is
        # restored on exit. Bounded to 1 frame; the buffer must still hold a FULL render. ─
        env = dict(os.environ)
        env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
        env["FLEET_PROFILES_DIR"] = str(profiles)
        env["FLEET_NOW"] = NOW
        env["NO_COLOR"] = "1"
        env["FLEET_WATCH_ITERATIONS"] = "1"
        env["FLEET_FRAME_DEADLINE"] = "30"  # cold first frame must complete (the 1s interval is just cadence)
        w = subprocess.run(
            ["bash", str(FLEET_SH), "--watch", "1"],
            capture_output=True,
            text=True,
            env=env,
        )
        check(
            "--watch exits cleanly (bounded frames)",
            w.returncode == 0,
            w.stderr.strip()[:120],
        )
        home = "\033[H"
        check("--watch repaints with cursor-home control seq", home in w.stdout)
        # REDRAW-IN-PLACE: enter the alternate screen buffer on start, leave it on exit (no scrollback pollution).
        check(
            "--watch enters the alternate screen buffer (\\033[?1049h) so frames redraw in place, not append",
            "\033[?1049h" in w.stdout,
            repr(w.stdout[:12]),
        )
        check(
            "--watch leaves the alternate screen buffer on exit (\\033[?1049l) — terminal restored",
            "\033[?1049l" in w.stdout,
            repr(w.stdout[-12:]),
        )
        check(
            "--watch hides the cursor while live and restores it on exit",
            "\033[?25l" in w.stdout and "\033[?25h" in w.stdout,
            "",
        )
        # the repaint control must PRECEDE a complete frame (header + a cluster section) in the byte stream
        after = w.stdout.split(home, 1)[-1]
        wplain = ANSI.sub("", after)
        check(
            "--watch buffer contains a FULL render before repaint",
            "ACTIVE  " in wplain and bool(_cblock(wplain, "alpha")) and bool(_cblock(wplain, "golf")),
            wplain[:80],
        )
        check(
            "--watch shows a live footer",
            "refresh 1s" in ANSI.sub("", w.stdout) and "gathered in" in ANSI.sub("", w.stdout),
        )
        check("--watch clears to end of screen (no stale frame)", "\033[J" in w.stdout)

        # ── REGRESSION: Ctrl-C during the background prefetch must NOT collapse temp paths to "/" ───────
        # (bug: the EXIT trap reaped $PERSIST while a background gather was mid-flight → mktemp failed →
        #  empty WORK → writes to /meta.tsv, /<cluster>.reg on a read-only fs. Fix: guard WORK + kill the
        #  prefetch before removing $PERSIST.) Run unbounded --watch, SIGINT it mid-interval, inspect stderr.
        import signal as _sig, time as _time

        ienv = dict(os.environ)
        ienv["PATH"] = f"{shim_dir}{os.pathsep}{ienv['PATH']}"
        ienv["FLEET_PROFILES_DIR"] = str(profiles)
        ienv["FLEET_NOW"] = NOW
        ienv["NO_COLOR"] = "1"
        ienv.pop("FLEET_WATCH_ITERATIONS", None)  # unbounded, so a prefetch is in flight when we interrupt
        p = subprocess.Popen(
            ["bash", str(FLEET_SH), "--watch", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=ienv,
        )
        _time.sleep(2.0)  # let the first frame render + a background prefetch start
        p.send_signal(_sig.SIGINT)
        try:
            _o, _e = p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            _o, _e = p.communicate()
            _e = (_e or "") + "\n[TIMEOUT]"
        bad = [
            m
            for m in (
                "Read-only file system",
                "/meta.tsv",
                ".reg: ",
                "mkdtemp failed",
                "No such file or directory",
            )
            if m in (_e or "")
        ]
        check(
            "Ctrl-C during prefetch: no /-write / temp-collapse errors on stderr",
            not bad,
            "; ".join(bad) or (_e or "")[:120],
        )

        # ── PARALLEL gather correctness: parallel output == sequential output (byte-identical) ─────────
        seq_env = dict(os.environ)
        seq_env["FLEET_SEQUENTIAL"] = "1"
        seq = run_fleet_full(profiles, shim_dir, bash_bin="bash")  # parallel (default)
        seq2 = subprocess.run(
            ["bash", str(FLEET_SH)],
            capture_output=True,
            text=True,
            env={
                **dict(os.environ),
                "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
                "FLEET_PROFILES_DIR": str(profiles),
                "FLEET_NOW": NOW,
                "NO_COLOR": "1",
                "FLEET_SEQUENTIAL": "1",
            },
        )
        check(
            "parallel gather == sequential gather (byte-identical)",
            seq.stdout == seq2.stdout,
            "outputs differ",
        )

        # ── PARALLELISM is a real speedup: with a per-cluster latency, wall-time ≈ max not sum ───────
        # SHIM_SLOW_CTX=ALL makes every kubectl call sleep; parallel gathers all clusters at once.
        slow_env = {
            **dict(os.environ),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "FLEET_PROFILES_DIR": str(profiles),
            "FLEET_NOW": NOW,
            "NO_COLOR": "1",
            "SHIM_SLOW_CTX": "ALL",
            "SHIM_SLOW_SECS": "0.15",
        }
        t0 = time.time()
        subprocess.run(["bash", str(FLEET_SH)], capture_output=True, text=True, env=slow_env)
        par_t = time.time() - t0
        t0 = time.time()
        subprocess.run(
            ["bash", str(FLEET_SH)],
            capture_output=True,
            text=True,
            env={**slow_env, "FLEET_SEQUENTIAL": "1"},
        )
        seq_t = time.time() - t0
        check(
            f"parallel faster than sequential ({par_t:.2f}s vs {seq_t:.2f}s)",
            par_t < seq_t * 0.7,
            f"par={par_t:.2f} seq={seq_t:.2f}",
        )

        # ── kubectl-call reduction: 3→1 namespaced collapse + TTL caches across --watch frames ────────
        def _watch_calls(iters, ns_ttl=None, extra_env=None):
            lg = Path(td) / f"calls_{iters}_{ns_ttl}.log"
            lg.write_text("")
            env = {
                **dict(os.environ),
                "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
                "FLEET_PROFILES_DIR": str(profiles),
                "FLEET_NOW": NOW,
                "NO_COLOR": "1",
                "SHIM_LOG": str(lg),
                "FLEET_WATCH_ITERATIONS": str(iters),
                "FLEET_NODES_TTL": "300",
                "FLEET_ALL_TTL": "300",
                "FLEET_FRAME_DEADLINE": "30",
            }
            if ns_ttl is not None:
                env["FLEET_NS_TTL"] = str(ns_ttl)
            env.update(extra_env or {})
            subprocess.run(
                ["bash", str(FLEET_SH), "--watch", "1"],
                capture_output=True,
                text=True,
                env=env,
            )
            return lg.read_text().splitlines()

        c1 = _watch_calls(1)
        ctxA_ns = sum(1 for c in c1 if c == "ctxA nsread")
        ctxA_sep = sum(1 for c in c1 if c.startswith("ctxA ") and c.split()[1] in ("pods", "deploys", "jobs"))
        check(
            "namespaced reads COLLAPSED 3→1 (one 'nsread' call, no separate pods/deploys/jobs)",
            ctxA_ns == 1 and ctxA_sep == 0,
            f"ctxA nsread={ctxA_ns}, separate={ctxA_sep}",
        )
        # The namespaced TTL is asserted with an EXPLICIT window, not the 4s default. The invariant under
        # test is "a read inside the TTL is REUSED"; with the default the assertion also silently required
        # two whole frames (gather + render + repaint, ~3s of canned-shim work on this fixture) to finish
        # inside 4 seconds — a wall-clock race against the harness, on a machine running CI. That race is
        # what made this check the suite's known flake; the other direction (NS_TTL=0 → always refetch) is
        # pinned separately below, so widening the window here weakens nothing.
        c2 = _watch_calls(2, ns_ttl=60)
        nodes1 = sum(1 for c in c1 if c.endswith(" nodes"))
        nodes2 = sum(1 for c in c2 if c.endswith(" nodes"))
        nsA2 = sum(1 for c in c2 if c == "ctxA nsread")
        check(
            "TTL-cached nodes NOT re-fetched on frame 2 (2-frame nodes == 1-frame nodes)",
            nodes2 == nodes1 and nodes1 > 0,
            f"1-frame nodes={nodes1}, 2-frame nodes={nodes2}",
        )
        check(
            "namespaced cache WITHIN its TTL: 2 frames reuse one OURS read (ctxA nsread fetched ONCE)",
            nsA2 == 1,
            f"ctxA nsread over 2 frames inside a 60s TTL={nsA2}",
        )
        c2_fresh = _watch_calls(2, ns_ttl=0)
        nsA2_fresh = sum(1 for c in c2_fresh if c == "ctxA nsread")
        check(
            "FLEET_NS_TTL=0 → OURS refetched every frame (ctxA nsread twice)",
            nsA2_fresh == 2,
            f"ctxA nsread with NS_TTL=0={nsA2_fresh}",
        )

        # ── --fast/--mine skips cluster-scoped reads (no nodes / pods -A at all) ─────────────────────
        flog = Path(td) / "fast.log"
        fenv = {
            **dict(os.environ),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "FLEET_PROFILES_DIR": str(profiles),
            "FLEET_NOW": NOW,
            "NO_COLOR": "1",
            "SHIM_LOG": str(flog),
        }
        flog.write_text("")
        fast = subprocess.run(["bash", str(FLEET_SH), "--fast"], capture_output=True, text=True, env=fenv)
        fplain = ANSI.sub("", fast.stdout)
        fcalls = flog.read_text().splitlines()
        check(
            "--fast issues NO cluster-scoped reads (no get nodes / pods -A)",
            not any(c.endswith(" nodes") or c.endswith(" all") for c in fcalls),
            f"cluster-scoped calls: {[c for c in fcalls if c.endswith(' nodes') or c.endswith(' all')]}",
        )
        check(
            "--fast still shows OUR active runs + ours GPU",
            "ours 8 GPUs" in _cblock(fplain, "alpha") and "modelx-r1c1" in _cblock(fplain, "alpha"),
            _cblock(fplain, "alpha"),
        )
        check(
            "--fast flagged in the breakdown line",
            "fast (capacity skipped)" in fplain,
            fplain.splitlines()[1] if len(fplain.splitlines()) > 1 else "",
        )

        # ── VIEW RESOLUTION: --watch defaults to the HIERARCHY pane; one-shot stays FLAT ────────────────
        # The hierarchy pane is identified by its `━━ CLUSTER <name>` bar (the flat pane's bar has no label).
        def _view(*args, iters="1"):
            venv = {
                **dict(os.environ),
                "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
                "FLEET_PROFILES_DIR": str(profiles),
                "FLEET_NOW": NOW,
                "NO_COLOR": "1",
                "FLEET_WATCH_ITERATIONS": iters,
                "FLEET_FRAME_DEADLINE": "30",
            }
            o = ANSI.sub(
                "",
                subprocess.run(
                    ["bash", str(FLEET_SH), *args],
                    capture_output=True,
                    text=True,
                    env=venv,
                ).stdout,
            )
            return "stages" if "━━ CLUSTER " in o else "flat"

        check(
            "view: bare --watch DEFAULTS to the hierarchy pane (the good format is no longer opt-in twice)",
            _view("--watch", "1") == "stages",
        )
        check(
            "view: --watch --flat still reaches the old flat pane (escape hatch)",
            _view("--watch", "1", "--flat") == "flat",
        )
        check(
            "view: --no-stages is an accepted alias for --flat",
            _view("--watch", "1", "--no-stages") == "flat",
        )
        check(
            "view: --watch --stages is unchanged (explicit choice honoured)",
            _view("--watch", "1", "--stages") == "stages",
        )
        # A flag only the flat pane implements IS an implicit view choice — never silently swallow it.
        for f in ("--wide", "--idle", "--all", "--failed"):
            check(
                f"view: --watch {f} keeps the FLAT pane (that pane implements {f}; don't drop the flag)",
                _view("--watch", "1", f) == "flat",
            )
        check(
            "view: ONE-SHOT default is unchanged (flat = the forensic/scripting surface)",
            _view() == "flat",
        )
        check(
            "view: one-shot --stages still opts into the hierarchy",
            _view("--stages") == "stages",
        )

        # ── --watch per-frame deadline: one slow cluster does NOT block the frame ─────────────────────
        # ctxB sleeps 5s; deadline 1s → frame returns fast, bravo shown as …refreshing (no data yet frame 1).
        # Pinned to --flat: this asserts the gather/deadline mechanics plus the FLAT pane's "every configured
        # cluster is ALWAYS shown" guarantee. The hierarchy pane (now the --watch default) deliberately folds
        # idle/refreshing clusters into a `+N refreshing` tail, which is a different, separately-tested contract.
        denv = {
            **dict(os.environ),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "FLEET_PROFILES_DIR": str(profiles),
            "FLEET_NOW": NOW,
            "NO_COLOR": "1",
            "SHIM_SLOW_CTX": "ctxB",
            "SHIM_SLOW_SECS": "5",
            "FLEET_WATCH_ITERATIONS": "1",
            "FLEET_FRAME_DEADLINE": "1",
        }
        t0 = time.time()
        dwatch = subprocess.run(
            ["bash", str(FLEET_SH), "--watch", "1", "--flat"],
            capture_output=True,
            text=True,
            env=denv,
        )
        frame_t = time.time() - t0
        dplain = ANSI.sub("", dwatch.stdout)
        check(
            f"laggard frame returns before the slow cluster ({frame_t:.2f}s < 5s)",
            frame_t < 4.0,
            f"frame took {frame_t:.2f}s",
        )
        check("fast clusters rendered despite one laggard", bool(_cblock(dplain, "alpha")))
        check(
            "laggard cluster shown as …refreshing (not blocking, not dropped)",
            "refreshing" in dplain and "bravo" in dplain,
            dplain,
        )

        # ── STALE LAST-GOOD FRAME: frame 1 is fast (bravo good), frame 2 laggards → bravo is re-rendered from
        #    its last good frame. It MUST be badged ⚠ STALE, and past FLEET_LAST_GOOD_MAX it must not be
        #    rendered at all. (The B200 report: PARKED servers deleted 20 minutes earlier, and `0 free` on an
        #    empty cluster, both shown under a plain `· connected` bar.) ──
        def _laggard_frame2(last_good_max, stages=True):
            cnt = root / f"shimcount-{last_good_max}"
            env2 = {
                **dict(os.environ),
                "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
                "FLEET_PROFILES_DIR": str(profiles),
                "FLEET_NOW": NOW,
                "NO_COLOR": "1",
                # frame 1 = 6 reads (nsread + nodes/all/allllmb/leases/pvcs) → fast; frame 2 laggards.
                # Margins are deliberately WIDE (deadline 6s vs a 20s sleep) so a loaded CI box can't
                # make frame 1 miss the deadline and turn this into a flake.
                "SHIM_SLOW_CTX": "ctxB",
                "SHIM_SLOW_SECS": "20",
                "SHIM_SLOW_AFTER": "6",
                "SHIM_COUNT": str(cnt),
                "FLEET_WATCH_ITERATIONS": "2",
                "FLEET_FRAME_DEADLINE": "6",
                "FLEET_NS_TTL": "0",
                "FLEET_LAST_GOOD_MAX": str(last_good_max),
            }
            r = subprocess.run(
                ["bash", str(FLEET_SH), "--watch", "1"] + ([] if stages else ["--flat"]),
                capture_output=True,
                text=True,
                env=env2,
            )
            return ANSI.sub("", r.stdout).split("\033[H")[-1] if "\033[H" in r.stdout else ANSI.sub("", r.stdout)

        kept = _laggard_frame2(120)
        check(
            "stale (e2e): a laggard re-shown from its last good frame is BADGED ⚠ STALE, never plain `connected`",
            "⚠ STALE" in kept,
            kept[-2500:],
        )
        dropped = _laggard_frame2(0)
        check(
            "stale (e2e): FLEET_LAST_GOOD_MAX=0 drops the stale frame entirely (…refreshing, no old data)",
            "⚠ STALE" not in dropped and "refreshing" in dropped,
            dropped[-2500:],
        )

        # ── --watch PIPELINING: the 2nd frame is pre-gathered during the interval → footer 'prefetched' ──
        penv = {
            **dict(os.environ),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
            "FLEET_PROFILES_DIR": str(profiles),
            "FLEET_NOW": NOW,
            "NO_COLOR": "1",
            "FLEET_WATCH_ITERATIONS": "2",
            "FLEET_FRAME_DEADLINE": "30",
        }
        pw = subprocess.run(
            ["bash", str(FLEET_SH), "--watch", "1"],
            capture_output=True,
            text=True,
            env=penv,
        )
        pplain = ANSI.sub("", pw.stdout)
        check(
            "--watch first frame gathered synchronously (footer 'gathered in')",
            "gathered in" in pplain,
            "",
        )
        check(
            "--watch later frame pre-gathered during the interval (footer 'prefetched')",
            "prefetched (instant)" in pplain,
            "",
        )

    if fails:
        print(f"\nselftest_fleet: {len(fails)} FAILURE(S): {', '.join(fails)}")
        return 1
    print("\nselftest_fleet: ALL CHECKS PASSED")
    return 0


def _grab(text: str, header: str) -> str:
    """The section starting at a `── <cluster>` header up to the next blank line."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header):
            block = [ln]
            for nxt in lines[i + 1 :]:
                if nxt.strip() == "" and len(block) > 1:
                    break
                block.append(nxt)
            return "\n".join(block)
    return ""


def _idx(text: str, needle: str) -> int:
    """Character offset of the first occurrence — for asserting render ORDER (nesting)."""
    return text.find(needle)


STATUSES = (
    "RUNNING",
    "LOADING",
    "STARTING",
    "UNSCHEDULED",
    "STUCK",
    "ORPHAN",
    "PARKED",
    "COMPLETE",
    "FAILED",
    "IDLE",
)


def _row(text: str, workload: str) -> str:
    """A workload's row (status token first, leading indent/nesting stripped)."""
    for ln in text.splitlines():
        s = ln.strip()
        if workload in ln and s.split(" ", 1)[0] in STATUSES:
            return s
    return ""


def _active_block(text: str) -> str:
    """The ACTIVE-RUN rows region: between the first blank line (after the summary) and the `clusters:` marker."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == ""), 0) + 1
    end = next((i for i, l in enumerate(lines) if l.strip() == "clusters:"), len(lines))
    return "\n".join(lines[start:end])


def _is_cbar(l: str, name: str) -> bool:
    """A flush-left CLUSTER bar for <name>, in the default view (`━━ <name> …`) OR the stages view (which adds
    an ALL-CAPS `CLUSTER` label → `━━ CLUSTER <name> …`)."""
    return (
        l.startswith(f"━━ {name} ")
        or l.startswith(f"━━ {name}  ")
        or l.startswith(f"━━ CLUSTER {name} ")
        or l.startswith(f"━━ CLUSTER {name}  ")
    )


def _cline(text: str, name: str) -> str:
    """The `━━ <name>` cluster rule line (stripped)."""
    for ln in text.splitlines():
        if _is_cbar(ln, name):
            return ln.strip()
    return ""


def _cblock(text: str, name: str) -> str:
    """The whole section for cluster <name>: its cluster rule line down to the next flush `━━ ` bar (or end)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if _is_cbar(l, name)), None)
    if start is None:
        return ""
    end = next(
        (i for i, l in enumerate(lines[start + 1 :], start + 1) if l.startswith("━━ ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _aligned(block: str) -> bool:
    """True iff the data rows in a cluster block share column offsets — the NAME column starts at the same
    char position in the header and every data row, and the GPU column's right edge lines up. This is the
    'columns line up like a real table' guarantee."""
    lines = block.splitlines()
    hdr = next((l for l in lines if "RUN / SERVER" in l and "GPU" in l), None)
    if hdr is None:
        return False
    name_off = hdr.index("RUN / SERVER")
    gpu_off = hdr.index("GPU", name_off)
    gpu_end = gpu_off + len("GPU")  # header 'GPU' is right-justified to the gpu column width
    data = [l for l in lines if l.strip().startswith(("● ", "  └ ")) or l.lstrip().startswith(("● ", "└ "))]
    if not data:
        return False
    for r in data:
        if len(r) <= name_off or r[name_off] == " " or r[name_off - 1] != " ":
            return False  # NAME column must begin exactly at name_off for every row
        if len(r) < gpu_end or r[gpu_end - 1] == " ":
            return False  # GPU value right-edge must reach the column's right edge
    return True


if __name__ == "__main__":
    raise SystemExit(main())

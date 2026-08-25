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

"""selftest_run_owner.py — offline guards for the PRIMARY, intrinsic GPU-lifecycle guarantee:
scripts/run_owner.sh (the per-run run-owner Job + child-adoption helpers) and scripts/merge_run_owner.py
(the apply-time ownerReference injector). No cluster: a fake `kubectl` shim (KUBECTL override) captures every
applied manifest + records every patch/delete, so we assert the exact object graph and teardown semantics.

§8 goes further than static text: it EXTRACTS the watcher script out of the rendered run-owner Job and RUNS
it (POSIX sh) against a fake cluster whose `get jobs` listing we control, so the release/hold decision itself
is under test — the in-cluster half of the ownership fix 558503a0 made on the client half.

The acceptance contract (task §Prove-it #1 — the object graph is correct):
  - run-owner Job carries activeDeadlineSeconds (run ceiling) + a SHORT ttlSecondsAfterFinished (prompt
    cascade), backoffLimit 0, and a watcher that releases on ANY terminal state of a Job IT OWNS — matched by
    the run-owner's own uid in .metadata.ownerReferences, never by the caller's PREDICTED Job name (the lane
    script may re-mint its own run-id, so the predicted name can be wrong and a name-matching watcher would
    hold the GPU until its 24h activeDeadlineSeconds instead of releasing promptly at completion);
  - the watcher is FAIL-SAFE: it releases on exactly ONE unambiguous reading (a successful list showing
    exactly one owned Job, terminal). Read error, zero owned Jobs, >1 owned Job, or an unreadable own-uid
    (missing RBAC) all HOLD — holding a GPU too long beats releasing live work;
  - `ensure` also applies the namespaced RBAC (SA/Role/RoleBinding, jobs: get/list/watch) and prints
    eval-able RUN_OWNER_NAME/RUN_OWNER_UID (uid non-empty on success);
  - the injector stamps a controller ownerReference (blockOwnerDeletion:true) with the REAL owner uid onto
    Deployment/Job kinds, is a NO-OP passthrough when RUN_OWNER_NAME/UID are unset or the uid is empty
    (a bad uid would make GC delete the child on sight), and never lets a run-owner Job own itself;
  - adopt-job / adopt-deploy patch a non-... controller ownerRef with the real uid, refuse an empty uid,
    and adopt-deploy matches EXACT server names only (never a sibling …-1m-offload — the destructive case);
  - teardown DELETES the run-owner (cascade), never scales to 0 (no idle-server 0/0 shell).
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
RUN_OWNER = ROOT / "scripts" / "run_owner.sh"
MERGE = ROOT / "scripts" / "merge_run_owner.py"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── 0. static lint ────────────────────────────────────────────────────────────
shn = subprocess.run(["bash", "-n", str(RUN_OWNER)], capture_output=True, text=True)
check(
    "run_owner.sh is valid bash (bash -n)",
    shn.returncode == 0,
    shn.stderr.strip()[:200],
)


def _shim(
    tmp: Path,
    *,
    uid: str = "uid-owner-123",
    deploys: list[str] | None = None,
    svcs: list[str] | None = None,
) -> Path:
    """Fake kubectl: captures `apply -f -` stdin to applied/*.yaml, logs patch/delete/scale, serves uid +
    canned deploy/svc listings for jsonpath queries."""
    applied = tmp / "applied"
    applied.mkdir(exist_ok=True)
    log = tmp / "cmd.log"
    listing = "".join(n + "\n" for n in (deploys or []))
    svc_listing = "".join(n + "\n" for n in (svcs or []))
    shim = tmp / "kubectl"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f'LOG="{log}"; APPLIED="{applied}"\n'
        'echo "$*" >> "$LOG"\n'
        'args="$*"\n'
        'case "$args" in\n'
        '  *"apply -f -"*) cat > "$APPLIED/m.$$.$RANDOM.yaml" ; exit 0 ;;\n'
        '  *"get deploy"*) printf %s ' + _q(listing) + " ; exit 0 ;;\n"
        '  *"get svc"*) printf %s ' + _q(svc_listing) + " ; exit 0 ;;\n"
        '  *"get job"*"metadata.uid"*) printf %s ' + _q(uid) + " ; exit 0 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _run(shim: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    bash_env = shim.parent / "bash-env.sh"
    bash_env.write_text(f'kubectl() {{ bash "{shim}" "$@"; }}\nexport -f kubectl\n')
    env["BASH_ENV"] = str(bash_env)
    env["KUBECTL"] = "kubectl"
    env.pop("KUBE_CONTEXT", None)
    return subprocess.run(["bash", str(RUN_OWNER), *args], capture_output=True, text=True, env=env)


def _applied_text(tmp: Path) -> str:
    return "\n".join(p.read_text() for p in sorted((tmp / "applied").glob("*.yaml")))


def _log(tmp: Path) -> str:
    p = tmp / "cmd.log"
    return p.read_text() if p.exists() else ""


# ── 1. ensure: RBAC + run-owner Job object graph + eval-able vars ─────────────
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(tmp, uid="uid-owner-123")
    p = _run(shim, "ensure", "ns1", "mycell", "run1", "mycell-bench-run1", "3600", "90")
    applied = _applied_text(tmp)
    out = p.stdout
    check("ensure: exits 0", p.returncode == 0, p.stderr)
    check(
        "ensure: prints eval-able RUN_OWNER_NAME=mycell-runowner-run1",
        "RUN_OWNER_NAME=mycell-runowner-run1" in out,
        out,
    )
    check(
        "ensure: prints RUN_OWNER_UID with the real uid",
        "RUN_OWNER_UID=uid-owner-123" in out,
        out,
    )
    check(
        "ensure: logging goes to stderr, not stdout (stdout stays eval-safe)",
        "run_owner:" not in out,
        out,
    )
    # RBAC
    check(
        "ensure: applies run-owner ServiceAccount llmb-runowner",
        "kind: ServiceAccount" in applied and "name: llmb-runowner" in applied,
        applied[:400],
    )
    check(
        "ensure: RBAC Role grants jobs get/list/watch (read-only, minimal)",
        'resources: ["jobs"]' in applied and '"get"' in applied and '"watch"' in applied,
        applied[:400],
    )
    # the run-owner Job object graph — the acceptance contract
    check(
        "ensure: creates the run-owner Job",
        "kind: Job" in applied and "name: mycell-runowner-run1" in applied,
    )
    check(
        "ensure: Job carries activeDeadlineSeconds (run/hang ceiling)",
        "activeDeadlineSeconds: 3600" in applied,
        applied,
    )
    check(
        "ensure: Job carries a SHORT ttlSecondsAfterFinished (prompt cascade)",
        "ttlSecondsAfterFinished: 90" in applied,
        applied,
    )
    check(
        "ensure: Job backoffLimit 0 (a fired watcher terminates the owner; no retries)",
        "backoffLimit: 0" in applied,
        applied,
    )
    check(
        "ensure: control-only run-owner does not inherit benchmark/GPU node placement",
        "nodeSelector:" not in applied,
        applied,
    )
    check(
        "ensure: watcher is told the caller's predicted Job name as a hint (WATCHED=mycell-bench-run1)",
        "mycell-bench-run1" in applied,
        applied,
    )
    check(
        "ensure: watcher exits on Complete AND on Failed (both terminal states) → prompt fail-freeing",
        ".status.succeeded" in applied and ".status.failed" in applied,
        applied,
    )
    # THE DEFECT: `ensure` runs BEFORE the Job exists, so WATCHED is only a PREDICTION and the lane script
    # may re-mint its own run-id. The watcher must key on OWNERSHIP (its own uid in .ownerReferences) — which
    # cannot drift — and must know its own name so it can self-look-up that uid.
    check(
        "ensure: watcher matches by ownerReferences, not by the predicted name",
        ".metadata.ownerReferences" in applied,
        applied,
    )
    check(
        "ensure: watcher is given its OWN Job name (OWNER) so it can self-look-up its uid",
        "name: OWNER" in applied and "mycell-runowner-run1" in applied,
        applied,
    )
    check(
        "ensure: run-owner labelled component=run-owner + cell + run-id",
        "component: run-owner" in applied and "llmb.nvidia.com/cell: mycell" in applied,
        applied,
    )

# ── 2. ensure: 63-char DNS label overflow is COMPRESSED (not skipped) so the run-owner still exists ─
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(tmp)
    longcell = "c" * 60
    p = _run(shim, "ensure", "ns1", longcell, "run1", f"{longcell}-bench-run1")
    import re as _re

    _m = _re.search(r"RUN_OWNER_NAME=(\S+)", p.stdout)
    _name = _m.group(1) if _m else ""
    check("ensure(too-long name): exits 0", p.returncode == 0, p.stdout)
    check(
        "ensure(too-long name): compressed owner name is non-empty and <=63 chars",
        bool(_name) and len(_name) <= 63,
        f"name={_name!r} len={len(_name)}",
    )
    check(
        "ensure(too-long name): compressed name uses the -ro-<hash8> form",
        "-ro-" in _name and not _name.endswith("-"),
        _name,
    )
    check(
        "ensure(too-long name): DOES create a run-owner Job (no skip to backstop)",
        "kind: Job" in _applied_text(tmp),
    )

# ── 3. adopt-job: patch a bench Job's ownerRef → run-owner (controller, real uid) ─
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(tmp)
    p = _run(
        shim,
        "adopt-job",
        "ns1",
        "mycell-bench-run1",
        "mycell-runowner-run1",
        "uid-owner-123",
    )
    log = _log(tmp)
    check("adopt-job: patches the bench Job", "patch job mycell-bench-run1" in log, log)
    check(
        "adopt-job: ownerRef carries run-owner name + real uid + controller:true + blockOwnerDeletion:true",
        '"name":"mycell-runowner-run1"' in log
        and '"uid":"uid-owner-123"' in log
        and '"controller":true' in log
        and '"blockOwnerDeletion":true' in log,
        log,
    )
    check("adopt-job: exits 0", p.returncode == 0, p.stderr)

# ── 4. adopt-job: empty uid → refuse (a bad uid makes GC delete the child on sight) ─
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(tmp)
    p = _run(shim, "adopt-job", "ns1", "mycell-bench-run1", "mycell-runowner-run1", "")
    check("adopt-job(empty uid): issues no patch", "patch job" not in _log(tmp), _log(tmp))
    check(
        "adopt-job(empty uid): exits 0 and warns",
        p.returncode == 0 and "empty owner uid" in p.stderr,
        p.stderr,
    )

# ── 5. adopt-deploy: EXACT server-name match only (never a sibling …-offload) ─
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(
        tmp,
        deploys=[
            "foo-1m-server",
            "foo-1m-decode",
            "foo-1m-offload-server",
            "foo-1m-server-extra",
        ],
        svcs=[
            "foo-1m-server",
            "foo-1m-etcd",
            "foo-1m-offload-server",
            "foo-1m-server-extra",
        ],
    )
    p = _run(shim, "adopt-deploy", "ns1", "foo-1m", "foo-1m-runowner-run1", "uid-9")
    log = _log(tmp)
    check(
        "adopt-deploy: adopts own foo-1m-server",
        "patch deploy foo-1m-server --type" in log,
        log,
    )
    check(
        "adopt-deploy: adopts own foo-1m-decode",
        "patch deploy foo-1m-decode --type" in log,
        log,
    )
    check(
        "adopt-deploy: NEVER adopts sibling foo-1m-offload-server",
        "patch deploy foo-1m-offload-server" not in log,
        log,
    )
    check(
        "adopt-deploy: NEVER adopts non-exact foo-1m-server-extra",
        "patch deploy foo-1m-server-extra" not in log,
        log,
    )
    check("adopt-deploy: exits 0", p.returncode == 0, p.stderr)
    # THE SERVICE MOVES WITH THE DEPLOYMENT. Since merge_run_owner.py now owns the Service from birth, the
    # --skip-server path must re-point it too — otherwise the PREVIOUS owner's terminal state GCs the Service
    # out from under the live server this run just adopted.
    check(
        "adopt-deploy: re-points the cell's Service too (svc/foo-1m-server)",
        "patch svc foo-1m-server --type" in log,
        log,
    )
    check(
        "adopt-deploy: re-points the disagg infra Service (svc/foo-1m-etcd)",
        "patch svc foo-1m-etcd --type" in log,
        log,
    )
    check(
        "adopt-deploy: NEVER adopts sibling svc/foo-1m-offload-server",
        "patch svc foo-1m-offload-server" not in log,
        log,
    )
    check(
        "adopt-deploy: NEVER adopts non-exact svc/foo-1m-server-extra",
        "patch svc foo-1m-server-extra" not in log,
        log,
    )

# ── 5b. adopt-deploy with a stale Service but NO Deployment still re-points the Service ──
# The Deployment is gone (crash/manual delete) but its Service survived. That leftover selects on
# `app: <cell>-server` and would silently adopt the NEXT run's pods, so it must not be skipped.
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(tmp, deploys=[], svcs=["foo-1m-server"])
    p = _run(shim, "adopt-deploy", "ns1", "foo-1m", "foo-1m-runowner-run1", "uid-9")
    log = _log(tmp)
    check(
        "adopt-deploy: no Deployment but a stale Service → the Service is still adopted (not skipped)",
        "patch svc foo-1m-server --type" in log,
        log,
    )
    check("adopt-deploy(stale svc only): exits 0", p.returncode == 0, p.stderr)

# ── 6. teardown: DELETE the run-owner (cascade), never scale-to-0 ─────────────
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(tmp)
    p = _run(shim, "teardown", "ns1", "mycell-runowner-run1")
    log = _log(tmp)
    check(
        "teardown: deletes the run-owner Job (cascade GC of server + bench)",
        "delete job mycell-runowner-run1" in log,
        log,
    )
    check(
        "teardown: background cascade (--wait=false) + idempotent (--ignore-not-found)",
        "--wait=false" in log and "--ignore-not-found" in log,
        log,
    )
    check(
        "teardown: NEVER scales anything to 0 (no idle-server 0/0 shell)",
        "scale" not in log,
        log,
    )
    check("teardown: exits 0", p.returncode == 0, p.stderr)

# ── 6b. teardown does not TRUST --ignore-not-found ────────────────────────────
# Filed defect: kc() only INHERITS KUBE_CONTEXT, so a standalone invocation silently targets the operator's
# current context — where the run-owner does not exist — and --ignore-not-found turns that into a cheerful
# exit 0 while the GPU stays held on the real cluster. Teardown must (a) check the object is actually THERE,
# (b) name the context it is talking to, (c) accept an explicit context, and (d) verify it is actually gone.
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    # a shim where NOTHING exists: `get job` fails (the wrong-context case)
    shim = tmp / "kubectl"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$*" >> "{tmp}/cmd.log"\n'
        'case "$*" in\n'
        '  *"get job"*) exit 1 ;;\n'
        '  *"current-context"*) printf some-other-cluster ; exit 0 ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    p = _run(shim, "teardown", "ns1", "mycell-runowner-run1")
    log, err = _log(tmp), p.stderr
    check(
        "teardown(not found): issues NO delete and says NOTHING was deleted (never 'deleted successfully')",
        "delete job" not in log and "NOT FOUND" in err and "NOTHING was deleted" in err,
        err + "|" + log,
    )
    check(
        "teardown(not found): names the context it actually talked to (the wrong-cluster tell)",
        "some-other-cluster" in err,
        err,
    )
    check(
        "teardown(not found): still exits 0 (best-effort; never aborts a caller's trap)",
        p.returncode == 0,
    )

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim = _shim(tmp)  # `get job` succeeds forever → the delete "worked" but the object never goes away
    p = _run(shim, "teardown", "ns1", "mycell-runowner-run1", "prod-ctx")
    log, err = _log(tmp), p.stderr
    check(
        "teardown: accepts an explicit <kube-context> arg (no silent reliance on ambient context)",
        "--context prod-ctx" in log,
        log,
    )
    check(
        "teardown: VERIFIES the object is gone; warns when delete returned 0 but it is still there",
        "still there" in err,
        err,
    )


# ── 7. merge_run_owner.py — the apply-time injector ───────────────────────────
def _merge(stdin: str, **env: str) -> str:
    e = dict(os.environ)
    e.update(env)
    return subprocess.run(["python3", str(MERGE)], input=stdin, capture_output=True, text=True, env=e).stdout


DEP = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: foo-server\nspec:\n  replicas: 1\n"

out = _merge(DEP, RUN_OWNER_NAME="foo-runowner-r1", RUN_OWNER_UID="uid-abc")
check(
    "merge: stamps a controller ownerRef with the real uid onto the server Deployment",
    "ownerReferences" in out
    and "uid: uid-abc" in out
    and "controller: true" in out
    and "blockOwnerDeletion: true" in out
    and "name: foo-runowner-r1" in out,
    out,
)
check("merge: preserves the original spec", "replicas: 1" in out, out)

out = _merge(DEP)  # no env
check(
    "merge: env unset → passthrough, no ownerReferences",
    "ownerReferences" not in out,
    out,
)

out = _merge(DEP, RUN_OWNER_NAME="foo-runowner-r1", RUN_OWNER_UID="")  # empty uid
check(
    "merge: empty uid → passthrough (never stamp a GC-suicidal ownerRef)",
    "ownerReferences" not in out,
    out,
)

out = _merge(
    "kind: Job\nmetadata:\n  name: foo-runowner-r1\n",
    RUN_OWNER_NAME="foo-runowner-r1",
    RUN_OWNER_UID="uid-abc",
)
check(
    "merge: a run-owner Job is never made to own itself",
    "ownerReferences" not in out,
    out,
)

# THE SERVICE IS RUN-SCOPED. It is born in the same apply as the Deployment, is named after the cell, and
# selects on `app: <cell>-server` — so a stale one from a dead run silently adopts the NEXT run's pods. It was
# previously left unstamped and carried as "LEAK, ACCEPTED"; it is now owned from birth and GC'd with the
# Deployment. (resource_inventory.json's Service entry names this file as the reclaim path.)
out = _merge(
    "kind: Service\nmetadata:\n  name: foo-server\n",
    RUN_OWNER_NAME="foo-runowner-r1",
    RUN_OWNER_UID="uid-abc",
)
check(
    "merge: the cell's Service is owned from birth (no stale Service pointing at a future run's pods)",
    "ownerReferences" in out and "uid: uid-abc" in out and "controller: true" in out,
    out,
)

out = _merge(
    "kind: Service\nmetadata:\n  name: foo-server\n",
    RUN_OWNER_NAME="foo-runowner-r1",
    RUN_OWNER_UID="",
)
check(
    "merge: Service with an empty uid → passthrough (a bad uid makes GC delete it on sight)",
    "ownerReferences" not in out,
    out,
)

# Namespace singletons / deliberately-surviving objects must NOT be dragged into the cascade.
for _kind in (
    "ConfigMap",
    "PersistentVolumeClaim",
    "ServiceAccount",
    "Role",
    "RoleBinding",
):
    out = _merge(
        f"kind: {_kind}\nmetadata:\n  name: foo-thing\n",
        RUN_OWNER_NAME="foo-runowner-r1",
        RUN_OWNER_UID="uid-abc",
    )
    check(
        f"merge: {_kind} is left untouched (singleton / deliberately outlives the run)",
        "ownerReferences" not in out,
        out,
    )

# The purpose: one apply stream, Deployment AND Service, both bound to the same owner.
out = _merge(
    DEP + "---\nkind: Service\nmetadata:\n  name: foo-server\n",
    RUN_OWNER_NAME="foo-runowner-r1",
    RUN_OWNER_UID="uid-abc",
)
check(
    "merge: a server stream stamps BOTH the Deployment and its Service (one GC cascade frees both)",
    out.count("uid: uid-abc") == 2,
    out,
)

# idempotent: re-stamping an already-owned object adds no duplicate ownerReference
owned = _merge(DEP, RUN_OWNER_NAME="foo-runowner-r1", RUN_OWNER_UID="uid-abc")
twice = _merge(owned, RUN_OWNER_NAME="foo-runowner-r1", RUN_OWNER_UID="uid-abc")
check(
    "merge: idempotent (re-stamp adds no duplicate ownerReference)",
    twice.count("uid: uid-abc") == 1,
    twice,
)


# ── 8. THE WATCHER, ACTUALLY RUN ──────────────────────────────────────────────────────────────────────────
# Everything above reads the manifest as text. This section pulls the watcher script OUT of the rendered
# run-owner Job and executes it (POSIX sh, as alpine/k8s would) against a fake cluster whose `get jobs`
# listing we author — so the release/hold DECISION is under test, not just its source text.
#
# THE DEFECT this pins: `run_owner.sh ensure` runs BEFORE the bench/driver Job exists, so <watched-job> is a
# PREDICTION. The lane script may re-mint its own run-id (run_id.py's t<b36> token is not idempotent once
# truncated for a long cell name), so the applied Job is named differently — observed on GB300, run-owner
# …-r36dc-ro-c8482acc had WATCHED=…-driver-ttj4it while the real Job was …-driver-ttj4jo. A name-matching
# watcher never sees a terminal state and holds its GPUs for the full 24h activeDeadlineSeconds. Ownership (the
# run-owner's uid on the Job's .ownerReferences, stamped by every lane's `adopt-job`) cannot drift.
def _watcher_script(tmp: Path) -> tuple[str, dict]:
    """Render the run-owner Job via `ensure` and return (watcher script, container env)."""
    import yaml as _yaml

    _run(
        _shim(tmp),
        "ensure",
        "ns1",
        "mycell",
        "run1",
        "mycell-driver-predicted",
        "3600",
        "90",
    )
    for f in sorted((tmp / "applied").glob("*.yaml")):
        for doc in _yaml.safe_load_all(f.read_text()):
            if doc and doc.get("kind") == "Job":
                c = doc["spec"]["template"]["spec"]["containers"][0]
                return c["args"][0], {e["name"]: e["value"] for e in c["env"]}
    return "", {}


OWNER_NAME = "mycell-runowner-run1"
OWNER_UID = "uid-owner-123"


def _drive(listing: str, *, list_rc: int = 0, uid_rc: int = 0, budget: float = 4.0) -> tuple[object, str]:
    """Run the watcher against a fake cluster. Returns (rc | 'HOLDING', output).

    'HOLDING' means the watcher was still running when the budget expired — i.e. it did NOT release, which is
    exactly the fail-safe behaviour for every ambiguous reading.
    """
    with tempfile.TemporaryDirectory() as td2:
        d = Path(td2)
        (d / "listing").write_text(listing)
        k = d / "kubectl"
        k.write_text(
            "#!/usr/bin/env bash\n"
            'case "$*" in\n'
            f'  *"get job "*"metadata.uid"*) [ {uid_rc} = 0 ] || exit 1; printf %s "{OWNER_UID}"; exit 0;;\n'
            f'  *"get jobs"*) [ {list_rc} = 0 ] || exit 1; cat "{d}/listing"; exit 0;;\n'
            "  *) exit 0;;\n"
            "esac\n"
        )
        k.chmod(k.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        e = dict(os.environ)
        e["PATH"] = f"{d}:{e['PATH']}"  # the watcher calls a bare `kubectl`, as it does in-cluster
        e.update(NS="ns1", OWNER=OWNER_NAME, WATCHED="mycell-driver-predicted", POLL="1")
        local_watcher = d / "watcher.sh"
        local_watcher.write_text(WATCHER.read_text().replace("kubectl ", f"""bash "{k}" """))
        try:
            pr = subprocess.run(
                ["sh", str(local_watcher)],
                capture_output=True,
                text=True,
                env=e,
                timeout=budget,
            )
            return pr.returncode, pr.stdout + pr.stderr
        except subprocess.TimeoutExpired as t:
            return "HOLDING", (t.stdout or b"").decode() + (t.stderr or b"").decode()


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    _script, _env = _watcher_script(tmp)
    WATCHER = tmp / "watcher.sh"
    WATCHER.write_text(_script)

    check(
        "watcher: the rendered Job's script is valid POSIX sh (alpine/k8s runs it under /bin/sh)",
        bool(_script) and subprocess.run(["sh", "-n", str(WATCHER)]).returncode == 0,
    )
    check(
        "watcher: env carries OWNER (its own Job name — for the self-lookup of its own uid)",
        _env.get("OWNER") == OWNER_NAME,
        str(_env),
    )

    # ── Regression case: driver lane, Job named with a DIFFERENT run-id than the one baked into WATCHED ──
    rc, out = _drive(f"mycell-driver-ttj4jo|{OWNER_UID}|1|\n")
    check(
        "watcher(driver lane, differing run-id): RELEASES on the owned terminal Job — this is the defect",
        rc == 0 and "mycell-driver-ttj4jo" in out,
        f"rc={rc} {out[-300:]}",
    )
    check(
        "watcher: releases via ownership, not the predicted name (the predicted Job never existed)",
        rc == 0 and "mycell-driver-predicted|" not in out,
        f"rc={rc} {out[-300:]}",
    )

    # ── llm-perf lane: Job DOES carry run.sh's run-id — behaviour must be unchanged ──
    rc, out = _drive(f"mycell-bench-run1|{OWNER_UID}|1|\n")
    check(
        "watcher(llm-perf lane, Complete): releases as before (no regression)",
        rc == 0,
        f"rc={rc} {out[-300:]}",
    )
    rc, out = _drive(f"mycell-bench-run1|{OWNER_UID}|0|1\n")
    check(
        "watcher(llm-perf lane, Failed): releases too — prompt fail-freeing, not the 24h deadline",
        rc == 0,
        f"rc={rc} {out[-300:]}",
    )

    # ── FAIL-SAFE MATRIX: every ambiguous reading HOLDS. Releasing live work ≫ worse than holding a GPU. ──
    rc, out = _drive(f"mycell-driver-x|{OWNER_UID}|1|\n", list_rc=1)
    check(
        "watcher(read failure): HOLDS — a failed list is never 'the run finished'",
        rc == "HOLDING" and "HOLDING" in out,
        f"rc={rc} {out[-300:]}",
    )
    rc, out = _drive("someone-elses-bench|other-uid|1|\n")
    check(
        "watcher(zero owned Jobs): HOLDS — pre-adoption, not completion",
        rc == "HOLDING",
        f"rc={rc} {out[-300:]}",
    )
    rc, out = _drive("")
    check("watcher(empty listing): HOLDS", rc == "HOLDING", f"rc={rc} {out[-300:]}")
    rc, out = _drive(f"mycell-driver-a|{OWNER_UID}|1|\nmycell-driver-b|{OWNER_UID}||\n")
    check(
        "watcher(>1 owned Job): HOLDS and logs the ambiguity — refuses to guess",
        rc == "HOLDING" and "ambiguous" in out,
        f"rc={rc} {out[-300:]}",
    )
    rc, out = _drive(f"mycell-driver-x|{OWNER_UID}|1|\n", uid_rc=1, budget=10.0)
    check(
        "watcher(missing RBAC / own uid unreadable): HOLDS to the deadline, does not error or release blind",
        rc == "HOLDING" and "HOLDING until activeDeadlineSeconds" in out,
        f"rc={rc} {out[-300:]}",
    )

    rc, out = _drive(f"{OWNER_NAME}|{OWNER_UID}|1|\n")
    check(
        "watcher: never treats its own object as the watched Job",
        rc == "HOLDING",
        f"rc={rc} {out[-300:]}",
    )

    # ── one owned Job, still running → hold (the ordinary case) ──
    rc, out = _drive(f"mycell-driver-x|{OWNER_UID}||\n")
    check(
        "watcher(owned Job running): HOLDS the GPU, as intended",
        rc == "HOLDING",
        f"rc={rc} {out[-300:]}",
    )

# ── 8b. the two implementations of ONE predicate must not drift apart ─────────
# run.sh's `_detach_scan_job` (558503a0) answers the same question on the client side. They cannot share a
# FILE — this half is POSIX sh inside an alpine container with no repo mount, that half is bash on the
# operator's laptop — so the shared thing is the CONTRACT, cross-checked here.
_ro = RUN_OWNER.read_text()
_run_sh = (ROOT / "scripts" / "run.sh").read_text()
for _needle, _what in ((".metadata.ownerReferences[*].uid", "the same ownerReferences jsonpath"),):
    check(
        f"predicate parity: run_owner.sh and run.sh use {_what}",
        _needle in _ro and _needle in _run_sh,
        _needle,
    )
check(
    "run.sh no longer claims the in-cluster run-owner waits out its deadline on a re-minted run-id",
    "still watches the predicted name" not in _run_sh,
)


# ── 9. wiring: run.sh creates the owner FIRST (before server) and tears it down on abort/teardown ──
run_sh = (ROOT / "scripts" / "run.sh").read_text()
# The run-owner phase (2.5) must appear before the server-deploy phase (4).
i_owner = run_sh.find('run_owner.sh" ensure')
i_deploy = run_sh.find('deploy.sh" "$CELL" "$PROFILE"')
check(
    "run.sh: run-owner is ensured BEFORE the server is deployed (owned from birth)",
    0 < i_owner < i_deploy,
    f"owner@{i_owner} deploy@{i_deploy}",
)
check(
    "run.sh: exports RUN_OWNER_NAME/UID so deploy.sh + sweep.sh inherit them",
    "export RUN_OWNER_NAME RUN_OWNER_UID" in run_sh,
)
check(
    "run.sh: abort trap tears down the run-owner (cascade), not scale-to-0",
    'run_owner.sh" teardown "$NAMESPACE" "$RUN_OWNER_NAME"' in run_sh,
)
check(
    "run.sh: --teardown deletes the run-owner (cascade)",
    re.search(r"TEARDOWN.*=.*1", run_sh) is not None and 'teardown "$NAMESPACE" "$RUN_OWNER_NAME"' in run_sh,
)

deploy_sh = (ROOT / "scripts" / "deploy.sh").read_text()
check(
    "deploy.sh: merge_run_owner.py is in the apply pipe (server owned from birth)",
    "merge_run_owner.py" in deploy_sh,
)

for lane in ("sweep.sh",):
    t = (ROOT / "scripts" / lane).read_text()
    check(
        f"{lane}: adopts the bench/driver Job to the run-owner when present",
        'run_owner.sh" adopt-job' in t,
        lane,
    )
    check(
        f"{lane}: falls back to adopt_server.sh only when no run-owner (RUN_OWNER_UID empty)",
        "RUN_OWNER_UID" in t and "adopt_server.sh" in t,
        lane,
    )


print()
if fails:
    print(f"selftest_run_owner: {len(fails)} FAILED: {fails}")
    sys.exit(1)
print("selftest_run_owner: all checks passed")

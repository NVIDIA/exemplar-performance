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

"""reclaim.py <cluster-profile> [--apply] [--only-job NAME] — release resources that FAILED or FINISHED but still squat.

k8s never frees GPUs for you: a CrashLooping server Deployment or a finished/failed
Job keeps holding its slot until someone acts. This audits the profile's namespace and, with --apply,
reclaims the clear cases:
  - CrashLoopBackOff / Error / ImagePullBackOff GPU pods  -> scale their owning Deployment to 0.
  - Complete / Failed Jobs                                -> delete (frees the Job + its pods).
Running+Ready GPU pods are left alone (they may be an active benchmark). Dry-run by default — always shows
what it WOULD do; pass --apply to act. Use --only-job NAME for a narrow cleanup in a shared namespace. Run it
periodically, or before claiming capacity on a shared cluster.

--all (teardown-all): the whole-cluster hammer for end-of-session cleanup. Idempotent + label-based, it tears
down EVERY GPU-holding server Deployment for our cells (scale->0 + delete), then RE-VERIFIES-UNTIL-ZERO:
settle, re-count GPU-holding pods across ALL phases (Running/Pending/Terminating) via an owner/selector match
(name-agnostic — server pods carry no managed-by label, only their Deployment does), re-delete any respawn
(the staggered-launch race), and repeat until 0 GPU pods or a cycle cap. Unlike the default mode it does NOT
spare a still-loading / actively-serving server — it is a deliberate operator action, so the dry-run prints
each server's readiness first. Tune with --settle-seconds / --max-cycles. Exit 0 when verified zero, 2 if not.

Exit 0 (advisory) for the default/only-job modes; teardown-all exits 2 if GPUs remain after the cycle cap.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BROKEN = {
    "CrashLoopBackOff",
    "Error",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerError",
    "RunContainerError",
}


SEL = "app.kubernetes.io/managed-by=llmb-recipe"  # our master label gate — every llmb manifest carries it.
GPU_RES = "nvidia.com/gpu"
SETTLE_SECONDS_DEFAULT = 12
MAX_CYCLES_DEFAULT = 8


class Args:
    def __init__(self):
        self.apply = False
        self.only_job = None
        self.all = False  # teardown-all: reap EVERY GPU-holding server for our cells (S3)
        self.settle_seconds = SETTLE_SECONDS_DEFAULT
        self.max_cycles = MAX_CYCLES_DEFAULT
        self.pos = []


def parse_args(args):
    a = Args()
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "--apply":
            a.apply = True
            i += 1
            continue
        if tok == "--all":
            a.all = True
            i += 1
            continue
        if tok == "--only-job":
            if i + 1 >= len(args):
                sys.exit("reclaim: --only-job needs a Job name")
            a.only_job = args[i + 1]
            i += 2
            continue
        if tok == "--settle-seconds":
            if i + 1 >= len(args):
                sys.exit("reclaim: --settle-seconds needs a value")
            a.settle_seconds = int(args[i + 1])
            i += 2
            continue
        if tok == "--max-cycles":
            if i + 1 >= len(args):
                sys.exit("reclaim: --max-cycles needs a value")
            a.max_cycles = int(args[i + 1])
            i += 2
            continue
        if tok.startswith("--"):
            sys.exit(f"reclaim: unknown flag {tok}")
        a.pos.append(tok)
        i += 1
    return a


def kc(*args):
    """Build kubectl command list with optional --context (from KUBE_CONTEXT env var, set from profile)."""
    ctx = os.environ.get("KUBE_CONTEXT", "").strip()
    cmd = ["kubectl"]
    if ctx:
        cmd += ["--context", ctx]
    return cmd + list(args)


def kubectl(ns, *args, timeout=40):
    r = subprocess.run(
        [*kc(), "-n", ns, "--request-timeout=30s", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def gpus(pod):
    return sum(
        int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0) or 0)
        for c in pod["spec"].get("containers", []) + pod["spec"].get("initContainers", [])
    )


def owning_deployment(ns, pod):
    """pod -> ReplicaSet owner -> Deployment owner (best effort)."""
    for o in pod["metadata"].get("ownerReferences", []):
        if o.get("kind") == "ReplicaSet":
            rc, out, _ = kubectl(ns, "get", "rs", o["name"], "-o", "json")
            if rc == 0:
                for oo in json.loads(out)["metadata"].get("ownerReferences", []):
                    if oo.get("kind") == "Deployment":
                        return oo["name"]
        if o.get("kind") == "Deployment":
            return o["name"]
    return None


# ── S4: trustworthy, name-AGNOSTIC verification ──────────────────────────────────────────────────────────
# Our server pods do NOT carry the managed-by label (only `app: <deploy>-server` + pod-template-hash) — only
# the Deployment carries it. So a pod is "ours" iff its labels satisfy the spec.selector.matchLabels of one of
# our managed-by Deployments. That is exactly how k8s itself associates a pod to a Deployment: OWNER/selector,
# never a name substring. This makes the count immune to renames and to any single cell's naming scheme.
def our_selectors(ns):
    """Return [(deploy_name, selector_dict)] for every managed-by=llmb-recipe Deployment (any GPU or not)."""
    rc, out, _ = kubectl(ns, "get", "deploy", "-l", SEL, "-o", "json")
    if rc != 0:
        return []
    sels = []
    for d in json.loads(out or '{"items":[]}').get("items", []):
        ml = (d.get("spec", {}).get("selector", {}) or {}).get("matchLabels") or {}
        sels.append((d["metadata"]["name"], ml))
    return sels


def _pod_matches(pod, selector):
    if not selector:
        return False
    labels = pod.get("metadata", {}).get("labels") or {}
    return all(labels.get(k) == v for k, v in selector.items())


def count_gpu_pods(ns, selectors):
    """Count GPU-holding pods that belong to OUR deployments, across ALL phases (Running/Pending/Terminating/
    Unknown). A Terminating pod (metadata.deletionTimestamp set) still pins its GPU until it is fully gone, so
    it is counted as non-zero — this is what makes 'confirm zero' honest rather than a racy single snapshot.
    Returns (n_pods, n_gpu, [pod descriptors])."""
    rc, out, err = kubectl(ns, "get", "pods", "-o", "json")
    if rc != 0:
        return None, None, [f"list-error: {err.strip()[:80]}"]
    n_pods = n_gpu = 0
    held = []
    for p in json.loads(out or '{"items":[]}').get("items", []):
        g = gpus(p)
        if not g:
            continue
        if not any(_pod_matches(p, sel) for _, sel in selectors):
            continue
        phase = p.get("status", {}).get("phase", "?")
        terminating = "Terminating" if p.get("metadata", {}).get("deletionTimestamp") else phase
        n_pods += 1
        n_gpu += g
        held.append(f"{p['metadata']['name']}({g}gpu,{terminating})")
    return n_pods, n_gpu, held


def gpu_holding_deploys(ns):
    """Every managed-by=llmb-recipe Deployment whose pod template requests GPU. Returns
    [(name, gpu_per_replica, spec_replicas, ready_replicas)]. Name-agnostic (label-scoped, no name matching).
    """
    rc, out, _ = kubectl(ns, "get", "deploy", "-l", SEL, "-o", "json")
    if rc != 0:
        return []
    res = []
    for d in json.loads(out or '{"items":[]}').get("items", []):
        tmpl = d.get("spec", {}).get("template", {})
        g = gpus({"spec": tmpl.get("spec", {})})
        if not g:
            continue
        res.append(
            (
                d["metadata"]["name"],
                g,
                d.get("spec", {}).get("replicas", 0),
                (d.get("status", {}) or {}).get("readyReplicas", 0),
            )
        )
    return res


def teardown_all(ns, args) -> int:
    """S3: idempotent, label-based, all-phase teardown of EVERY GPU-holding server for our cells, then
    re-verify-until-zero. Dry-run by default; --apply acts. Deliberately a big, operator-initiated hammer
    (end-of-session cleanup) — it does NOT spare a still-loading or actively-serving server, so the dry-run
    lists readiness so a human can see what would die before passing --apply."""
    selectors = our_selectors(ns)
    deploys = gpu_holding_deploys(ns)
    n0, gpu0, held0 = count_gpu_pods(ns, selectors)

    print(
        f"reclaim teardown-all: namespace '{ns}'  ·  mode: {'APPLY (tearing down)' if args.apply else 'dry-run (report only)'}"
    )
    print(f"  GPU-holding server Deployments (managed-by=llmb-recipe): {len(deploys)}")
    for name, g, rep, ready in deploys:
        live = "  <-- LIVE/serving" if ready else ("  <-- coming-up/cold-load" if rep else "  (scaled to 0)")
        print(f"    deploy/{name}  {g}gpu x rep={rep} ready={ready}{live}")
    print(f"  GPU-holding pods (ours, all phases) BEFORE: {n0 if n0 is not None else '?'} pods / {gpu0 or 0} GPU")
    for h in held0 or []:
        print(f"      {h}")

    if not args.apply:
        if deploys or n0:
            print("  → re-run with --apply to tear down ALL of the above and re-verify-until-zero.")
        else:
            print("  nothing to tear down — 0 GPU-holding servers.")
        return 0

    # APPLY: scale->0 then delete every GPU-holding deploy (idempotent — repeat runs are no-ops once gone).
    for name, g, _rep, _ready in deploys:
        r = kubectl(ns, "scale", "deploy", name, "--replicas=0")
        print(f"  scaled deploy/{name} -> 0 ({g} GPU)  {'ok' if r[0] == 0 else r[2].strip()[:80]}")
    for name, g, _rep, _ready in deploys:
        r = kubectl(ns, "delete", "deploy", name, "--wait=false", "--ignore-not-found")
        print(f"  deleted deploy/{name}  {'ok' if r[0] == 0 else r[2].strip()[:80]}")

    # re-verify-until-zero: settle, re-query across all phases, repeat until 0 GPU pods or the cycle cap.
    for cycle in range(1, args.max_cycles + 1):
        time.sleep(args.settle_seconds)
        n, g, held = count_gpu_pods(ns, selectors)
        if n is None:
            print(f"  [verify {cycle}/{args.max_cycles}] list error — retrying")
            continue
        print(f"  [verify {cycle}/{args.max_cycles}] GPU-holding pods (ours, all phases): {n} / {g} GPU")
        for h in held:
            print(f"      still-held: {h}")
        if n == 0:
            print("  CONFIRMED: 0 GPU-holding servers for our cells. Teardown complete.")
            return 0
        # Repeat deletion to cover deployments that created a pod during cancellation.
        for name, gg, _rep, _ready in gpu_holding_deploys(ns):
            kubectl(ns, "delete", "deploy", name, "--wait=false", "--ignore-not-found")
    n, g, _ = count_gpu_pods(ns, selectors)
    print(
        f"  WARNING: after {args.max_cycles} cycles, {n} GPU-holding pod(s) / {g} GPU still present — "
        f"investigate (stuck Terminating? node NotReady?). Exit 2."
    )
    return 2


def main() -> int:
    args = parse_args(sys.argv[1:])
    pos, apply, only_job = args.pos, args.apply, args.only_job
    if not pos:
        sys.exit(
            "usage: reclaim.py <cluster-profile> [--apply] [--only-job NAME | --all] [--settle-seconds N] [--max-cycles N]"
        )
    root = Path(__file__).resolve().parent.parent
    envf = root / "cluster-profiles" / f"{pos[0]}.env"
    if not envf.is_file():
        sys.exit(f"reclaim: no profile at {envf}")
    lines = envf.read_text().splitlines()
    ns = next(
        (ln.split("=", 1)[1].strip().strip('"').strip("'") for ln in lines if ln.strip().startswith("NAMESPACE=")),
        None,
    )
    if not ns:
        sys.exit("reclaim: NAMESPACE not found in profile")
    ctx = next(
        (ln.split("=", 1)[1].strip().strip('"').strip("'") for ln in lines if ln.strip().startswith("KUBE_CONTEXT=")),
        "",
    )
    if ctx:
        os.environ["KUBE_CONTEXT"] = ctx

    # teardown-all (S3): the whole-cluster, re-verify-until-zero path — a distinct, deliberate hammer.
    if args.all:
        return teardown_all(ns, args)

    # Scope to llmb-managed resources ONLY. In a SHARED namespace, an unscoped enumeration + --apply could scale
    # down a concurrent NON-llmb recipe that transiently crashed. All llmb manifests carry this label; the trade
    # is safety-over-completeness (better to miss an unlabeled llmb pod than to kill someone else's workload).
    rc, out, err = kubectl(ns, "get", "pods", "-l", SEL, "-o", "json")
    if rc != 0:
        sys.exit(f"reclaim: cannot list pods in {ns}: {err.strip()[:120]}")
    pods = json.loads(out).get("items", [])
    jobs = json.loads(kubectl(ns, "get", "jobs", "-l", SEL, "-o", "json")[1] or '{"items":[]}').get("items", [])

    active, broken_deploys, freed = [], {}, 0
    for p in pods:
        g = gpus(p)
        if not g:
            continue
        name, phase = p["metadata"]["name"], p.get("status", {}).get("phase")
        waiting = {
            cs.get("state", {}).get("waiting", {}).get("reason")
            for cs in p.get("status", {}).get("containerStatuses", [])
        }
        if waiting & BROKEN:
            dep = owning_deployment(ns, p)
            broken_deploys.setdefault(dep or f"(pod){name}", 0)
            broken_deploys[dep or f"(pod){name}"] += g
            freed += g
        elif phase in ("Succeeded", "Failed"):
            freed += g
        else:
            active.append((name, g, phase))

    finished_jobs = [
        (
            j["metadata"]["name"],
            "Complete" if (j.get("status") or {}).get("succeeded") else "Failed",
        )
        for j in jobs
        if (j.get("status") or {}).get("succeeded") or (j.get("status") or {}).get("failed")
    ]
    if only_job:
        finished_jobs = [(n, why) for n, why in finished_jobs if n == only_job]
        broken_deploys = {}
        freed = 0

    print(f"reclaim: namespace '{ns}'  ·  mode: {'APPLY (reclaiming)' if apply else 'dry-run (report only)'}")
    if only_job:
        print(f"  target Job filter: {only_job}")
    print(f"  active GPU pods (LEFT ALONE): {[(n, g) for n, g, _ in active] or '—'}")
    print(f"  broken GPU deployments to scale->0: {dict(broken_deploys) or '—'}")
    print(f"  finished Jobs to delete: {finished_jobs or '—'}")
    print(f"  ~{freed} GPU(s) reclaimable")

    if apply:
        for dep, g in broken_deploys.items():
            if dep.startswith("(pod)"):
                print(f"  ! {dep} has no Deployment owner — delete it manually if it's yours")
                continue
            r = kubectl(ns, "scale", "deploy", dep, "--replicas=0")
            print(f"  scaled deploy/{dep} -> 0 ({g} GPU)  {'ok' if r[0] == 0 else r[2].strip()[:80]}")
        for jn, why in finished_jobs:
            r = kubectl(ns, "delete", "job", jn)
            print(f"  deleted job/{jn} ({why})  {'ok' if r[0] == 0 else r[2].strip()[:80]}")
    elif broken_deploys or finished_jobs:
        print("  → re-run with --apply to reclaim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

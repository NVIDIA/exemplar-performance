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

"""selftest_reclaim.py — offline guards for the hardened reclaim.py teardown-all + verification loop (S3/S4).

No cluster. Imports reclaim.py as a module and monkeypatches its `kubectl()` with an in-memory fake k8s that
serves canned Deployment/Pod JSON and records mutations, then asserts:

  S4 verification (count_gpu_pods):
    * name-AGNOSTIC: counts GPU pods by OWNER/selector match (our Deployments' spec.selector.matchLabels),
      NOT by a name substring — a foreign GPU pod that does not match any selector is NOT counted.
    * all-phase: a Terminating pod (metadata.deletionTimestamp set) is still counted (it pins its GPU).
  S3 teardown-all:
    * dry-run mutates NOTHING and lists the GPU-holding deploys.
    * --apply scales->0 + deletes every GPU-holding deploy, then RE-VERIFIES-UNTIL-ZERO (settle, repoll all
      phases, re-delete respawns) and returns 0 once the GPU-pod count reaches 0.
    * if pods never clear, it returns 2 (honest failure, not a false 'zero').
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("reclaim", ROOT / "scripts" / "reclaim.py")
reclaim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reclaim)

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def deploy(name, selapp, gpu, replicas=1, ready=1):
    return {
        "metadata": {"name": name, "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"}},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": {"app": selapp}},
            "template": {
                "spec": {"containers": [{"name": "s", "resources": {"requests": {"nvidia.com/gpu": str(gpu)}}}]}
            },
        },
        "status": {"replicas": replicas, "readyReplicas": ready},
    }


def pod(name, app, gpu, phase="Running", terminating=False):
    m = {"name": name, "labels": {"app": app, "pod-template-hash": "abc"}}
    if terminating:
        m["deletionTimestamp"] = "2026-07-27T00:00:00Z"
    return {
        "metadata": m,
        "status": {"phase": phase},
        "spec": {"containers": [{"name": "s", "resources": {"requests": {"nvidia.com/gpu": str(gpu)}}}]},
    }


class World:
    """Mutable fake cluster. `pods_by_cycle` is a list: each successive `get pods` returns the next entry
    (so we can model Terminating -> gone across settle cycles). Mutations append to `.log`."""

    def __init__(self, deploys, pods_by_cycle):
        self.deploys = list(deploys)
        self.pods_by_cycle = list(pods_by_cycle)
        self._pod_calls = 0
        self.log = []

    def kubectl(self, ns, *args, timeout=40):
        a = list(args)
        if a[:2] == ["get", "deploy"]:
            return 0, json.dumps({"items": self.deploys}), ""
        if a[:2] == ["get", "pods"]:
            idx = min(self._pod_calls, len(self.pods_by_cycle) - 1)
            self._pod_calls += 1
            return 0, json.dumps({"items": self.pods_by_cycle[idx]}), ""
        if a[:1] == ["scale"]:
            self.log.append(" ".join(a))
            return 0, "", ""
        if a[:1] == ["delete"]:
            self.log.append(" ".join(a))
            return 0, "", ""
        return 0, json.dumps({"items": []}), ""


def run_teardown(world, apply):
    reclaim.kubectl = world.kubectl
    args = reclaim.Args()
    args.apply = apply
    args.settle_seconds = 0  # no real sleep in the test
    args.max_cycles = 4
    return reclaim.teardown_all("testns", args)


# ── S4: count_gpu_pods is name-agnostic (owner/selector) + all-phase ──────────────────────────────────────
selectors = [("srv-a", {"app": "srv-a"}), ("srv-b", {"app": "srv-b"})]
mixed_pods = [
    pod("srv-a-xxx", "srv-a", 4),  # ours, Running
    pod("srv-b-yyy", "srv-b", 2, terminating=True),  # ours, Terminating (still holds GPU)
    pod("foreign-gpu", "someone-else", 8),  # NOT ours (no selector match) -> excluded
    pod("srv-a-nogpu", "srv-a", 0),  # ours but 0 GPU -> excluded
]
w = World([], [mixed_pods])
reclaim.kubectl = w.kubectl
n, g, held = reclaim.count_gpu_pods("testns", selectors)
check("count_gpu_pods counts only OUR GPU pods (name-agnostic selector match)", n == 2, f"n={n}")
check("count_gpu_pods sums GPU across matched pods", g == 6, f"g={g}")
check("count_gpu_pods counts a Terminating pod (all-phase)", any("Terminating" in h for h in held), str(held))
check("count_gpu_pods EXCLUDES a foreign (unmatched) GPU pod", not any("foreign" in h for h in held), str(held))


# ── S3: teardown-all dry-run mutates nothing ──────────────────────────────────────────────────────────────
deploys = [
    deploy("srv-a-server", "srv-a", 4, replicas=1, ready=1),
    deploy("srv-b-server", "srv-b", 2, replicas=1, ready=0),
    deploy("non-gpu-server", "srv-c", 0, replicas=1, ready=1),
]
live_pods = [pod("srv-a-server-1", "srv-a", 4), pod("srv-b-server-1", "srv-b", 2)]
w = World(deploys, [live_pods])
rc = run_teardown(w, apply=False)
check("teardown-all dry-run exits 0", rc == 0)
check("teardown-all dry-run issues ZERO mutations", w.log == [], str(w.log))

# gpu_holding_deploys excludes the non-GPU deploy (label-scoped, template-GPU-gated).
reclaim.kubectl = w.kubectl
ghd = reclaim.gpu_holding_deploys("testns")
check(
    "gpu_holding_deploys returns only GPU-typed deploys",
    sorted(n for n, *_ in ghd) == ["srv-a-server", "srv-b-server"],
    str(ghd),
)


# ── S3/S4: teardown-all --apply scales+deletes then re-verifies-until-ZERO ─────────────────────────────────
# cycle 0 = BEFORE snapshot (2 held); after scale+delete, verify cycle 1 shows one Terminating; cycle 2 = 0.
w = World(deploys, [live_pods, [pod("srv-a-server-1", "srv-a", 4, terminating=True)], []])  # settling  # cleared
rc = run_teardown(w, apply=True)
check("teardown-all --apply exits 0 once verified zero", rc == 0)
check(
    "teardown-all --apply scaled both GPU deploys to 0",
    sum(1 for x in w.log if x.startswith("scale") and "--replicas=0" in x) >= 2,
    str(w.log),
)
check(
    "teardown-all --apply deleted both GPU deploys",
    sum(1 for x in w.log if x.startswith("delete deploy srv-")) >= 2,
    str(w.log),
)
check(
    "teardown-all --apply did NOT touch the non-GPU deploy", not any("non-gpu-server" in x for x in w.log), str(w.log)
)


# ── S3: non-convergence is reported honestly (exit 2), never a false 'zero' ────────────────────────────────
w = World(
    [deploy("stuck-server", "stuck", 8, replicas=1, ready=1)], [[pod("stuck-1", "stuck", 8)]] * 12
)  # pods NEVER clear (stuck Terminating / NotReady node)
rc = run_teardown(w, apply=True)
check("teardown-all --apply returns 2 when GPUs never clear (honest, not a false zero)", rc == 2, f"rc={rc}")


print()
if fails:
    print(f"selftest_reclaim: {len(fails)} FAILED: {fails}")
    raise SystemExit(1)
total = sum(1 for ln in Path(__file__).read_text().splitlines() if ln.strip().startswith("check("))
print(f"selftest_reclaim: all {total} checks PASSED")

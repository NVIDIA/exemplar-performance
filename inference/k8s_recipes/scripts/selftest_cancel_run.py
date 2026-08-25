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

"""Pure/offline safety tests for `llmb-k8s cancel`."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cancel_run", ROOT / "scripts" / "cancel_run.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" [{detail}]" if detail and not ok else ""))
    if not ok:
        fails.append(name)


job = {
    "kind": "Job",
    "metadata": {
        "name": "cell-bench-r1",
        "uid": "u1",
        "labels": {"llmb.nvidia.com/run-id": "r1", "app.kubernetes.io/managed-by": "llmb-recipe"},
    },
}
check("identity: exact run-id + managed-by passes", not mod.validate_job_identity(job, "r1"))
check("identity: wrong run-id fails closed", bool(mod.validate_job_identity(job, "r2")))
unmanaged = {"kind": "Job", "metadata": {"labels": {"llmb.nvidia.com/run-id": "r1"}}}
check("identity: unmanaged Job fails closed", bool(mod.validate_job_identity(unmanaged, "r1")))

deps = [
    {
        "metadata": {"name": "ours", "ownerReferences": [{"kind": "Job", "name": "cell-bench-r1", "uid": "u1"}]},
        "spec": {"selector": {"matchLabels": {"app": "ours"}}},
    },
    {
        "metadata": {"name": "theirs", "ownerReferences": [{"kind": "Job", "name": "other", "uid": "u2"}]},
        "spec": {"selector": {"matchLabels": {"app": "theirs"}}},
    },
]
pods = [
    {"metadata": {"name": "ours-pod", "labels": {"app": "ours"}}},
    {"metadata": {"name": "their-pod", "labels": {"app": "theirs"}}},
]
dn, pn = mod.owned_targets(job, deps, pods)
check("ownership: selects only Job-owned deployment", dn == ["ours"], str(dn))
check("ownership: selects only that deployment's pod", pn == ["ours-pod"], str(pn))

owner = {
    "kind": "Job",
    "metadata": {
        "name": "cell-runowner-r1",
        "uid": "owner-u1",
        "labels": {
            "llmb.nvidia.com/run-id": "r1",
            "app.kubernetes.io/managed-by": "llmb-recipe",
            "app.kubernetes.io/component": "run-owner",
        },
    },
}
owned_driver = {
    "kind": "Job",
    "metadata": {
        "name": "cell-driver-other-id",
        "uid": "driver-u1",
        "labels": {"llmb.nvidia.com/run-id": "r1", "app.kubernetes.io/managed-by": "llmb-recipe"},
        "ownerReferences": [{"kind": "Job", "name": "cell-runowner-r1", "uid": "owner-u1", "controller": True}],
    },
}
check(
    "run-owner: driver resolves its lifecycle root", mod.run_owner_ref(owned_driver) == ("cell-runowner-r1", "owner-u1")
)
driver_owned_deps = [
    {
        "metadata": {
            "name": "driver-server",
            "ownerReferences": [{"kind": "Job", "name": "cell-driver-other-id", "uid": "driver-u1"}],
        },
        "spec": {"selector": {"matchLabels": {"app": "driver-server"}}},
    }
]
driver_owned_pods = [{"metadata": {"name": "driver-server-pod", "labels": {"app": "driver-server"}}}]
chain_deps, chain_pods = mod.owned_targets([owner, owned_driver], driver_owned_deps, driver_owned_pods)
check("ownership chain: finds deployment owned by the driver Job", chain_deps == ["driver-server"], str(chain_deps))
check(
    "ownership chain: waits for driver deployment pod",
    chain_pods == ["driver-server-pod"],
    str(chain_pods),
)

check("run-owner discovery: selects the unique labelled root", mod.select_run_owner([owner], "r1") is owner)
check("run-owner discovery: no matching root is already stopped", mod.select_run_owner([], "r1") is None)
wrong_root = {
    **owner,
    "metadata": {**owner["metadata"], "labels": {**owner["metadata"]["labels"], "llmb.nvidia.com/run-id": "other"}},
}
check("run-owner discovery: ignores a different run", mod.select_run_owner([wrong_root], "r1") is None)
try:
    mod.select_run_owner([owner, {**owner, "metadata": {**owner["metadata"], "name": "second-owner"}}], "r1")
    duplicate_refused = False
except ValueError:
    duplicate_refused = True
check("run-owner discovery: duplicate roots fail closed", duplicate_refused)
try:
    mod.select_run_owner([{**owner, "metadata": {**owner["metadata"], "uid": ""}}], "r1")
    incomplete_refused = False
except ValueError:
    incomplete_refused = True
check("run-owner discovery: missing UID fails closed", incomplete_refused)

check(
    "run-owner: validated lifecycle root passes",
    not mod.validate_run_owner(owner, "cell-runowner-r1", "owner-u1", "r1"),
)
bad_owner = {
    "kind": "Job",
    "metadata": {
        "name": "cell-runowner-r1",
        "uid": "wrong",
        "labels": {
            "llmb.nvidia.com/run-id": "r1",
            "app.kubernetes.io/managed-by": "llmb-recipe",
            "app.kubernetes.io/component": "run-owner",
        },
    },
}
check(
    "run-owner: uid mismatch fails closed",
    bool(mod.validate_run_owner(bad_owner, "cell-runowner-r1", "owner-u1", "r1")),
)
ambiguous_driver = {
    "metadata": {
        "ownerReferences": [
            {"kind": "Job", "name": "a", "uid": "u-a", "controller": True},
            {"kind": "Job", "name": "b", "uid": "u-b", "controller": True},
        ]
    }
}
try:
    mod.run_owner_ref(ambiguous_driver)
    ambiguous_refused = False
except ValueError:
    ambiguous_refused = True
check("run-owner: ambiguous controller ownership fails closed", ambiguous_refused)

good_list = subprocess.CompletedProcess([], 0, stdout='{"items":[{"metadata":{"name":"one"}}]}', stderr="")
check("discovery: valid Kubernetes list decodes", len(mod.decoded_items(good_list, "Pods")) == 1)
failed_list = subprocess.CompletedProcess([], 1, stdout="", stderr="forbidden")
try:
    mod.decoded_items(failed_list, "Pods")
    failed_discovery_refused = False
except ValueError:
    failed_discovery_refused = True
check("discovery: API failure is not treated as an empty list", failed_discovery_refused)
malformed_list = subprocess.CompletedProcess([], 0, stdout='{"kind":"PodList"}', stderr="")
try:
    mod.decoded_items(malformed_list, "Pods")
    malformed_discovery_refused = False
except ValueError:
    malformed_discovery_refused = True
check("discovery: malformed response fails closed", malformed_discovery_refused)

if fails:
    print(f"selftest_cancel_run: {len(fails)} failure(s)")
    raise SystemExit(1)
print("selftest_cancel_run: all checks passed")

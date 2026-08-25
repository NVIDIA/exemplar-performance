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

"""merge_run_owner.py — apply-time ownerReference injection binding a GPU-holding object to the per-run
run-owner Job, so native Kubernetes garbage-collection frees the GPU the instant the run halts (no client,
no polling governor). Reads manifests from stdin, writes the patched manifests to stdout.

THE INVARIANT. A GPU-holding object (the vLLM server Deployment) must have an in-cluster owner whose lifetime
== the run's lifetime. scripts/run_owner.sh creates that owner (a tiny watcher Job) FIRST — BEFORE the server
is applied — and exports its name + uid as RUN_OWNER_NAME / RUN_OWNER_UID. This stage runs inside the
`envsubst | ... | kubectl apply` pipe in deploy.sh, so every server Deployment is created WITH the
ownerReference already on it (owned from birth — ZERO unowned window, unlike a post-apply `kubectl patch`
which leaves a gap between apply and patch in which a hard-killed orchestrator orphans the server).

  ownerReferences:
  - {apiVersion: batch/v1, kind: Job, name: $RUN_OWNER_NAME, uid: $RUN_OWNER_UID,
     controller: true, blockOwnerDeletion: true}

When the run-owner Job reaches a terminal state (its watcher exits when the bench Job Completes OR
Fails; or its activeDeadlineSeconds fires on a hang) its short ttlSecondsAfterFinished deletes it, and GC
then cascade-deletes this owned server — promptly, deterministically, disconnect-proof.

GATED ON THE ENV (never the recipe). If RUN_OWNER_NAME / RUN_OWNER_UID are unset/empty (e.g. a standalone
`deploy.sh` with no run-owner), stdin passes through UNCHANGED — the legacy adopt_server.sh + governor remain
the backstop. A non-empty uid is MANDATORY: an ownerReference whose uid doesn't match a live owner makes GC
treat the owner as already-gone and delete the child immediately, so we refuse to stamp without one.

HASH-NEUTRAL. This patches the LIVE apply stream, never the committed rendered/*.yaml, so no cell's
recipe_hash moves — ownerReferences are runtime metadata, correctly excluded from the fingerprint (identical
discipline to merge_imex_claim.py / merge_rdma_selector.py). Exit 0 always; any failure → passthrough so a
deploy is never blocked.

Deployment, Job and Service kinds are stamped (the run-scoped objects); ServiceAccounts, Roles, PVCs and
ConfigMaps in the same stream are left untouched — they are namespace singletons or deliberately outlive the
run. A Job whose name marks it as a run-owner (…-runowner-…) is skipped so the owner never accidentally owns
itself.

WHY SERVICE. A cell's Service is applied in this same stream, beside its server Deployment, and is named
after the cell — so a stale one from a dead run keeps a ClusterIP + DNS name that selects on
`app: <cell>-server` and therefore SILENTLY ADOPTS THE NEXT RUN'S PODS. It costs no GPU, which is why it was
carried as "LEAK, ACCEPTED" in scripts/testdata/resource_inventory.json, but a Service that points at a
future run's pods is a correctness hazard, not just untidiness. It is born in the same apply as the
Deployment, has the same lifetime, and has no other legitimate controller — so the same owned-from-birth
ownerReference is the right answer, and the GC cascade now collects Deployment and Service together.
"""

import os
import sys

try:
    import yaml
except ImportError:  # no PyYAML → never block a deploy
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)

_STAMP_KINDS = {"Deployment", "Job", "Service"}


def _owner_ref(name: str, uid: str) -> dict:
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "name": name,
        "uid": uid,
        "controller": True,
        "blockOwnerDeletion": True,
    }


def _stamp(doc: dict, name: str, uid: str) -> bool:
    """Set doc.metadata.ownerReferences to the run-owner. Returns True if it stamped."""
    if not isinstance(doc, dict):
        return False
    if doc.get("kind") not in _STAMP_KINDS:
        return False
    meta = doc.get("metadata")
    if not isinstance(meta, dict):
        return False
    # Never let a run-owner Job own itself.
    if "runowner" in str(meta.get("name", "")):
        return False
    existing = meta.get("ownerReferences")
    ref = _owner_ref(name, uid)
    # Idempotent: if we've already stamped this exact owner, do nothing. Otherwise install ours as the sole
    # controller ownerReference (these objects are BORN in this apply stream — they have no other legitimate
    # owner to preserve; a second controller:true ownerRef is illegal, so we replace rather than append).
    if isinstance(existing, list) and any(
        isinstance(o, dict) and o.get("uid") == uid and o.get("kind") == "Job" for o in existing
    ):
        return False
    meta["ownerReferences"] = [ref]
    return True


def main() -> None:
    content = sys.stdin.read()
    name = (os.environ.get("RUN_OWNER_NAME") or "").strip()
    uid = (os.environ.get("RUN_OWNER_UID") or "").strip()

    # No run-owner in play (standalone deploy) OR a missing uid (would make GC delete the child on sight):
    # pass the stream through untouched. Legacy adopt_server.sh + the governor remain the backstop.
    if not name or not uid:
        sys.stdout.write(content)
        return

    try:
        docs = [d for d in yaml.safe_load_all(content) if d is not None]
        n = sum(1 for d in docs if _stamp(d, name, uid))
        if n:
            print(
                f"merge_run_owner: bound {n} object(s) to run-owner job/{name} "
                f"(uid={uid}) → GC cascade-frees the GPU when the run-owner terminates",
                file=sys.stderr,
            )
        sys.stdout.write(yaml.dump_all(docs, default_flow_style=False, allow_unicode=True))
    except Exception as e:  # any parse/dump failure → passthrough, never block a deploy
        print(f"merge_run_owner: warning — injection failed ({e}), applying unmodified", file=sys.stderr)
        sys.stdout.write(content)


if __name__ == "__main__":
    main()

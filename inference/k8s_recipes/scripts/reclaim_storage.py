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

"""Delete artifact PVCs whose results have been verified on local storage.

``fetch_results.sh`` marks a PVC reclaimable only after verifying the fetched files. This command keeps
verification and deletion separate, supports an operator keep label, and defaults to dry-run.

SAFETY GATES (all must pass; each is independent so no single bug can delete the wrong thing):
  1. label `llmb.nvidia.com/reclaimable=true`     — the fetch vouched for it
  2. name ends `-artifacts` / `-artifacts-rwx`    — shape gate
  3. name does NOT match the model-cache denylist — model caches are never eligible
  4. label `llmb.nvidia.com/keep=true` absent     — sticky operator opt-out, honoured forever
  5. not mounted by any live pod                  — re-checked against the cluster, never inferred from names
  6. reclaimable-at older than --older-than       — optional cooling-off window
Dry-run is the DEFAULT. `--apply` additionally requires `--yes` when stdin is not a TTY.

On a ``Retain`` storage class, deleting the PVC does not delete the backing volume. Those claims are
reported as ``RETAIN-SKIP`` for deliberate operator cleanup.

FLEET CONTRACT: `--json` emits the exact per-namespace rollup fleet renders. See FLEET_CONTRACT below.

Exit 0 always (advisory), 2 if --apply hit a delete error.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── the label/annotation contract, shared with mark_reclaimable.py ───────────────────────────────────────
NS_PREFIX = "llmb.nvidia.com"
L_RECLAIMABLE = f"{NS_PREFIX}/reclaimable"  # "true" — set by a VERIFIED fetch only
L_RECLAIM_AT = f"{NS_PREFIX}/reclaimable-at"  # "20260802T034512Z" (label-safe: no colons)
L_KEEP = f"{NS_PREFIX}/keep"  # "true" — sticky operator opt-out; sweeper never touches
A_LOCAL_PATH = f"{NS_PREFIX}/verified-local-path"  # absolute local dir that vouches for the PVC
A_LOCAL_BYTES = f"{NS_PREFIX}/verified-local-bytes"
A_ENTRIES = f"{NS_PREFIX}/verified-entries"  # "2/2" from the fetch receipt
A_RUN_ID = f"{NS_PREFIX}/verified-run-id"
# Only marks from the current receipt contract are trusted.
A_RECEIPT_VERSION = f"{NS_PREFIX}/verified-receipt-version"
MIN_RECEIPT_VERSION = 2

ARTIFACT_SUFFIXES = ("-artifacts", "-artifacts-rwx")

# Gate 3. Independent of any label: if a name looks like a model cache we refuse, full stop. Deliberately
# broad — a false negative here costs a re-download of hundreds of GiB.
CACHE_DENY = re.compile(
    r"(model-cache|hf-cache|nvfp4-cache|fp8-cache|shared-cache|-cache$|-cache-r\d+$|shared-cache)",
    re.IGNORECASE,
)

# FLEET CONTRACT (agreed with the fleet/INSTALLED rework — fleet renders, this produces):
#   `reclaim_storage.py <profile> --json` → {"cluster","namespace","total_pvcs","artifacts_pvcs",
#     "artifacts_bytes","reclaimable_pvcs","reclaimable_bytes","kept_pvcs","cache_pvcs","cache_bytes",
#     "items":[{name,bytes,class,state,evidence}]}
#   `class` ∈ {artifacts, model-cache, control, other}   ← fleet's INSTALLED section must key off THIS,
#                                                           not a name heuristic (artifacts were being
#                                                           mislabelled as models)
#   `state` ∈ {live, reclaimable, keep, unverified, retain-skip}
# Suggested fleet line (fleet owns the rendering; we own the numbers):
#   ARTIFACTS 47 PVCs · 2.3Ti · 43 reclaimable → llmb-k8s reclaim --storage
FLEET_CONTRACT_VERSION = 1


class Args:
    def __init__(self):
        self.apply = False
        self.yes = False
        self.json = False
        self.older_than = 0  # seconds
        self.pos = []


def parse_duration(s):
    """PURE — '7d' / '12h' / '30m' / '3600' → seconds. Raises SystemExit on garbage."""
    m = re.fullmatch(r"(\d+)([smhdw]?)", s.strip())
    if not m:
        sys.exit(f"reclaim --storage: bad --older-than {s!r} (want e.g. 7d, 12h, 30m)")
    n, unit = int(m.group(1)), m.group(2) or "s"
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


def parse_args(args):
    a = Args()
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("--storage", "--apply", "--yes", "--json", "--dry-run"):
            if tok == "--apply":
                a.apply = True
            elif tok == "--yes":
                a.yes = True
            elif tok == "--json":
                a.json = True
            # --storage is the routing flag (consumed by llmb-k8s); --dry-run is the default, accept as a no-op
            i += 1
            continue
        if tok == "--older-than":
            if i + 1 >= len(args):
                sys.exit("reclaim --storage: --older-than needs a value (e.g. 7d)")
            a.older_than = parse_duration(args[i + 1])
            i += 2
            continue
        if tok.startswith("--"):
            sys.exit(f"reclaim --storage: unknown flag {tok}")
        a.pos.append(tok)
        i += 1
    return a


def kc(*args):
    ctx = os.environ.get("KUBE_CONTEXT", "").strip()
    cmd = ["kubectl"]
    if ctx:
        cmd += ["--context", ctx]
    return cmd + list(args)


def kubectl(ns, *args, timeout=60):
    r = subprocess.run(
        [*kc(), "-n", ns, "--request-timeout=45s", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


_UNITS = {
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
    "Pi": 2**50,
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
}


def qty_bytes(q):
    """PURE — k8s quantity ('50Gi', '1.2Ti', '100G', '1024') → int bytes. 0 on garbage."""
    q = (q or "").strip()
    if not q:
        return 0
    for suf in ("Ki", "Mi", "Gi", "Ti", "Pi"):
        if q.endswith(suf):
            try:
                return int(float(q[:-2]) * _UNITS[suf])
            except ValueError:
                return 0
    for suf in ("K", "M", "G", "T"):
        if q.endswith(suf):
            try:
                return int(float(q[:-1]) * _UNITS[suf])
            except ValueError:
                return 0
    try:
        return int(q)
    except ValueError:
        return 0


def human(n):
    """PURE — bytes → '2.3Ti'."""
    for suf, div in (("Ti", 2**40), ("Gi", 2**30), ("Mi", 2**20)):
        if n >= div:
            return f"{n / div:.1f}{suf}"
    return f"{n}B"


def classify(pvc):
    """PURE — a PVC object → its fleet `class`. Name-based, deliberately conservative: anything that could
    be a cache is a cache."""
    name = pvc["metadata"]["name"]
    if name == "llmb-control":
        return "control"
    if CACHE_DENY.search(name):
        return "model-cache"
    if name.endswith(ARTIFACT_SUFFIXES):
        return "artifacts"
    return "other"


def mounted_pvcs(ns):
    """PVC names mounted by any non-terminal pod. Cluster truth, never a name heuristic."""
    rc, out, _ = kubectl(ns, "get", "pods", "-o", "json")
    if rc != 0:
        return None  # unknown — caller must treat as "assume everything is mounted"
    live = set()
    for p in json.loads(out).get("items", []):
        if p.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        for v in p["spec"].get("volumes", []):
            c = v.get("persistentVolumeClaim", {}).get("claimName")
            if c:
                live.add(c)
    return live


def retain_classes(ns):
    """{storageClassName} whose reclaimPolicy is Retain — deleting those PVCs leaks a billing PV."""
    r = subprocess.run(
        [*kc(), "get", "sc", "-o", "json", "--request-timeout=30s"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return set()  # can't list SCs (RBAC) — no Retain warnings, gates 1-6 still apply
    return {
        sc["metadata"]["name"] for sc in json.loads(r.stdout).get("items", []) if sc.get("reclaimPolicy") == "Retain"
    }


def parse_stamp(v):
    """PURE — '20260802T034512Z' → epoch seconds, or None."""
    try:
        return int(time.mktime(time.strptime(v, "%Y%m%dT%H%M%SZ")) - time.timezone)
    except Exception:
        return None


def evaluate(pvc, live, retain, now, older_than):
    """PURE (given its inputs) — a PVC → (state, evidence). `live` may be None = unknown.

    state: live | keep | reclaimable | unverified | retain-skip | too-fresh
    Only `reclaimable` is ever deleted.
    """
    m = pvc["metadata"]
    name = m["name"]
    labels = m.get("labels") or {}
    ann = m.get("annotations") or {}
    cls = classify(pvc)

    if cls != "artifacts":  # gate 2 + gate 3
        return "protected", f"class={cls} — never eligible"
    if labels.get(L_KEEP) == "true":  # gate 4
        return "keep", f"{L_KEEP}=true (operator opt-out)"
    if live is None:
        return "unverified", "could not list pods — refusing to assume unmounted"
    if name in live:  # gate 5
        return "live", "mounted by a non-terminal pod"
    if labels.get(L_RECLAIMABLE) != "true":  # gate 1
        return "unverified", "no verified-fetch mark — run `llmb-k8s collect` first"
    try:  # gate 1b — receipt contract version
        _rv = int(ann.get(A_RECEIPT_VERSION, "0"))
    except ValueError:
        _rv = 0
    if _rv < MIN_RECEIPT_VERSION:
        return "unverified", (
            f"mark predates the evidence-bearing fetch receipt (v{_rv} < "
            f"v{MIN_RECEIPT_VERSION}): it vouched for entries enumerated, not files "
            "landed — re-fetch to re-verify before reclaiming"
        )
    at = parse_stamp(labels.get(L_RECLAIM_AT, ""))
    if older_than and at is not None and (now - at) < older_than:  # gate 6
        return "too-fresh", f"marked {(now - at) // 3600}h ago (< --older-than)"
    if pvc["spec"].get("storageClassName") in retain:
        return "retain-skip", (
            f"storageClass {pvc['spec'].get('storageClassName')} is reclaimPolicy=Retain — "
            f"deleting the PVC leaks PV {pvc['spec'].get('volumeName')}; clean up deliberately"
        )
    return "reclaimable", (
        f"verified {ann.get(A_ENTRIES, '?')} entries → "
        f"{ann.get(A_LOCAL_PATH, '?')} ({human(qty_bytes(ann.get(A_LOCAL_BYTES, '0')))}) "
        f"at {labels.get(L_RECLAIM_AT, '?')}"
    )


def main(argv):
    a = parse_args(argv)
    if not a.pos:
        sys.exit("usage: llmb-k8s reclaim --storage <cluster-profile> [--apply] [--older-than 7d] [--json]")
    ns = os.environ.get("NAMESPACE", "").strip()
    if not ns:
        sys.exit("reclaim --storage: NAMESPACE not set (the profile should export it)")
    cluster = os.environ.get("CLUSTER", a.pos[0])

    rc, out, err = kubectl(ns, "get", "pvc", "-o", "json")
    if rc != 0:
        sys.exit(f"reclaim --storage: cannot list PVCs in {ns}: {err.strip()}")
    items = json.loads(out).get("items", [])
    live = mounted_pvcs(ns)
    retain = retain_classes(ns)
    now = int(time.time())

    rows, roll = [], {
        "artifacts": [0, 0],
        "model-cache": [0, 0],
        "control": [0, 0],
        "other": [0, 0],
    }
    for pvc in sorted(items, key=lambda x: x["metadata"]["name"]):
        b = qty_bytes(
            (pvc.get("status", {}).get("capacity") or {}).get("storage")
            or pvc["spec"]["resources"]["requests"]["storage"]
        )
        cls = classify(pvc)
        state, ev = evaluate(pvc, live, retain, now, a.older_than)
        roll[cls][0] += 1
        roll[cls][1] += b
        rows.append(
            {
                "name": pvc["metadata"]["name"],
                "bytes": b,
                "class": cls,
                "state": state,
                "evidence": ev,
            }
        )

    doomed = [r for r in rows if r["state"] == "reclaimable"]
    payload = {
        "contract_version": FLEET_CONTRACT_VERSION,
        "cluster": cluster,
        "namespace": ns,
        "total_pvcs": len(rows),
        "artifacts_pvcs": roll["artifacts"][0],
        "artifacts_bytes": roll["artifacts"][1],
        "cache_pvcs": roll["model-cache"][0],
        "cache_bytes": roll["model-cache"][1],
        "reclaimable_pvcs": len(doomed),
        "reclaimable_bytes": sum(r["bytes"] for r in doomed),
        "kept_pvcs": sum(1 for r in rows if r["state"] == "keep"),
        "items": rows,
    }
    if a.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"reclaim --storage: {cluster} / {ns}   ({len(rows)} PVCs)")
    print(
        f"  ARTIFACTS   {roll['artifacts'][0]:>3} PVCs · {human(roll['artifacts'][1]):>7}"
        f"  ({len(doomed)} reclaimable · {human(payload['reclaimable_bytes'])})"
    )
    print(f"  MODEL CACHE {roll['model-cache'][0]:>3} PVCs · {human(roll['model-cache'][1]):>7}" f"  (never eligible)")
    for r in rows:
        if r["state"] in ("protected",):
            continue
        mark = {
            "reclaimable": "DELETE ",
            "live": "live   ",
            "keep": "keep   ",
            "unverified": "hold   ",
            "too-fresh": "fresh  ",
            "retain-skip": "RETAIN ",
        }[r["state"]]
        print(f"  {mark} {r['name']:<64} {human(r['bytes']):>7}  {r['evidence']}")

    if not doomed:
        print(
            "\nNothing to reclaim. (An artifacts PVC becomes reclaimable only after a VERIFIED "
            "`llmb-k8s collect` — that is by design.)"
        )
        return 0

    print(f"\n  → {len(doomed)} PVCs, {human(payload['reclaimable_bytes'])} reclaimable")
    if not a.apply:
        print("  DRY RUN (default). Re-run with --apply to delete.")
        return 0
    if not a.yes and not sys.stdin.isatty():
        print("  refusing --apply without --yes on a non-interactive stdin.")
        return 0
    if not a.yes:
        if input(f"  delete {len(doomed)} PVCs in {ns}? type 'yes': ").strip() != "yes":
            print("  aborted.")
            return 0

    bad = 0
    for r in doomed:
        rc, _, err = kubectl(ns, "delete", "pvc", r["name"], "--wait=false")
        if rc == 0:
            print(f"  deleted {r['name']}")
        else:
            bad += 1
            print(f"  FAILED  {r['name']}: {err.strip()}")
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

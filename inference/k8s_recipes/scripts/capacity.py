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

"""capacity.py <profile> [--product PRODUCT] [--require-nodes N [--node-gpus G]] [--require-free N] — read-only GPU capacity + who is holding it.

Answers "why won't my 8-GPU server schedule?" at a glance for the profile's GPU_PRODUCT:
  - per-node allocatable / used / free (DRA-aware accounting, reused from preflight),
  - the biggest single free node (a TP=8 server needs 8 free on ONE node, not 8 spread across the cluster),
  - a holders table (namespace/pod -> GPUs) that SEPARATES llmb-owned pods (reclaimable via
    `llmb-k8s reclaim <profile> --apply`) from EXTERNAL occupants in other namespaces you can't touch.

Never mutates. Honors the profile's KUBE_CONTEXT (via preflight's pinned kubectl). Built for the recurring
capacity churn on small B200 pools — so an agent can see instantly whether to wait, reclaim, or move clusters.

  capacity.py <profile>                print the capacity view for the profile's GPU_PRODUCT
  capacity.py <profile> --product X    override the product label (e.g. NVIDIA-B200)

Machine-checkable launch/pre-deploy gate (exit 0 = met, 3 = SHORT) — scripts the "capacity check before
deploy" so it isn't eyeballed. Requirements combine (all must hold):
  --require-nodes N [--node-gpus G]    ≥ N nodes each with ≥ G free GPUs (G default 1) — e.g. a tp=4 fleet
                                       needs `--require-nodes 6 --node-gpus 4` (6 schedulable one-node cells)
  --require-free N                     ≥ N free GPUs total across the pool
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight as pf  # reuse the pinned kubectl (_KUBE_CONTEXT), parse_env, and DRA-aware gpu_availability


def is_llmb_pod(pod: dict) -> bool:
    """A pod this recipe collection owns (so `llmb-k8s reclaim` could free it) vs a foreign occupant."""
    labels = (pod.get("metadata") or {}).get("labels") or {}
    return labels.get("app.kubernetes.io/managed-by") == "llmb-recipe" or "llmb.nvidia.com/recipe" in labels


def gpu_holders(pods: list, node_names) -> list:
    """PURE — GPU-requesting pods on the given nodes → [{node, namespace, pod, gpus, llmb}], busiest first.
    Counts device-plugin requests (containers + initContainers); skips terminal pods (they hold nothing)."""
    want = set(node_names)
    out = []
    for p in pods:
        if (p.get("status") or {}).get("phase") in ("Succeeded", "Failed"):
            continue
        node = (p.get("spec") or {}).get("nodeName")
        if node not in want:
            continue
        spec = p.get("spec") or {}
        gpus = sum(
            int((c.get("resources") or {}).get("requests", {}).get("nvidia.com/gpu", 0) or 0)
            for c in ((spec.get("containers") or []) + (spec.get("initContainers") or []))
        )
        if gpus <= 0:
            continue
        out.append(
            {
                "node": node,
                "namespace": p["metadata"]["namespace"],
                "pod": p["metadata"]["name"],
                "gpus": gpus,
                "llmb": is_llmb_pod(p),
            }
        )
    return sorted(out, key=lambda h: h["gpus"], reverse=True)


def _alloc(node: dict) -> int:
    return int((node.get("status", {}).get("allocatable", {}) or {}).get("nvidia.com/gpu", 0) or 0)


def evaluate_requirement(free_per_node, *, need_nodes=None, node_gpus=1, need_free_total=None):
    """PURE — given [free_gpus, ...] per node, decide whether a capacity requirement is met.
    Requirements combine (all must hold); returns (ok: bool, reasons: [str]) where each reason is a
    human line, prefixed ✓ (met) or ✗ (short). `free_per_node` is post-cap free counts (never negative)."""
    reasons = []
    ok = True
    if need_nodes is not None:
        have = sum(1 for f in free_per_node if f >= node_gpus)
        met = have >= need_nodes
        ok = ok and met
        reasons.append(
            f"{'✓' if met else '✗'} nodes with ≥{node_gpus} free: {have} "
            f"({'≥' if met else '<'} {need_nodes} required)"
        )
    if need_free_total is not None:
        have = sum(max(0, f) for f in free_per_node)
        met = have >= need_free_total
        ok = ok and met
        reasons.append(
            f"{'✓' if met else '✗'} total free GPUs: {have} " f"({'≥' if met else '<'} {need_free_total} required)"
        )
    return ok, reasons


def _int_flag(argv, flag):
    """Return int value following `flag` in argv, or None if absent. Exits on a missing/non-int value."""
    if flag not in argv:
        return None
    i = argv.index(flag)
    if i + 1 >= len(argv):
        sys.exit(f"capacity: {flag} needs an integer value")
    try:
        return int(argv[i + 1])
    except ValueError:
        sys.exit(f"capacity: {flag} value must be an integer, got {argv[i + 1]!r}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    profile = args[0]
    prof_path = pf.ROOT / "cluster-profiles" / f"{profile}.env"
    if not prof_path.is_file():
        sys.exit(f"capacity: no profile at {prof_path} — llmb-k8s profile init --cluster {profile}")
    prof = pf.parse_env(prof_path)
    pf._KUBE_CONTEXT = (prof.get("KUBE_CONTEXT") or "").strip()

    product = None
    if "--product" in sys.argv:
        i = sys.argv.index("--product")
        product = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    product = (product or prof.get("GPU_PRODUCT", "")).strip()
    if not product:
        sys.exit(f"capacity: no GPU_PRODUCT in {profile}.env and no --product given")

    ctx = pf._KUBE_CONTEXT or "(ambient)"
    title = f"capacity: {profile}"
    print(f"── {title} {'─' * max(4, 60 - len(title))}")
    print(f"  product: {product}   context: {ctx}")

    # a --require gate must FAIL closed if no capacity is even visible (don't green-light on an empty read).
    gated = any(f in sys.argv for f in ("--require-nodes", "--require-free"))

    rc, nout, _ = pf.krun(["get", "nodes", "-l", f"nvidia.com/gpu.product={product}", "-o", "json"])
    if rc != 0 or not nout:
        print(f"  ❌ no nodes labelled nvidia.com/gpu.product={product} (or cluster unreachable)")
        return 3 if gated else 0
    _, pout, _ = pf.krun(["get", "pods", "-A", "-o", "json"])
    rc_dra, dcout, _ = pf.krun(["get", "resourceclaims", "-A", "-o", "json"])
    nodes = json.loads(nout).get("items", [])
    pods = json.loads(pout or '{"items":[]}').get("items", [])
    claims = json.loads(dcout or '{"items":[]}').get("items", []) if rc_dra == 0 else []

    avail = pf.gpu_availability(nodes, pods, claims, set(), "")  # no taint/arch filter — show ALL product nodes
    used = avail["used"]
    holders = gpu_holders(pods, [n["metadata"]["name"] for n in nodes])

    total = sum(_alloc(n) for n in nodes)
    in_use = sum(min(used.get(n["metadata"]["name"], 0), _alloc(n)) for n in nodes)
    free = total - in_use
    # cap per-node `used` at allocatable — a node can never use more GPUs than it has; guards the
    # display against any residual DRA over-count so an impossible `used > alloc` row can't print.
    per_node = sorted(
        ([n["metadata"]["name"], _alloc(n), min(used.get(n["metadata"]["name"], 0), _alloc(n))] for n in nodes),
        key=lambda r: (r[1] - r[2]),
        reverse=True,
    )
    biggest_free = max((a - u for _, a, u in per_node), default=0)

    print(
        f"  {len(nodes)} node(s) · {total} GPUs total · {in_use} in use · {free} free   "
        f"(biggest free node: {biggest_free})"
    )
    if avail.get("dra_gpus"):
        print(f"  (includes {avail['dra_gpus']} GPU(s) reserved via DRA ResourceClaims)")
    print("  per node:")
    for name, a, u in per_node:
        print(f"    {name:<40} {a} alloc · {u} used · {a - u} free")

    if holders:
        print("  holders (GPU-requesting pods on these nodes, busiest first):")
        for h in holders:
            tag = "llmb    " if h["llmb"] else "external"
            print(f"    {h['gpus']:>3}  {tag}  {h['namespace']}/{h['pod']:<44} {h['node']}")
    reclaimable = sum(h["gpus"] for h in holders if h["llmb"])
    external = sum(h["gpus"] for h in holders if not h["llmb"])
    if reclaimable:
        print(f"  reclaimable (llmb-owned): {reclaimable} GPU(s) — free with: llmb-k8s reclaim {profile} --apply")
    if external:
        print(f"  external (other namespaces, not yours): {external} GPU(s)")
    if biggest_free < 8:
        print(
            "  ⚠ no single node has 8 free GPUs — a TP=8 server can't schedule until one frees "
            "(wait: llmb-k8s preflight <cell> {p} --wait-on-resources)".format(p=profile)
        )

    # machine-checkable gate: if any --require-* given, evaluate and set exit code (0 met / 3 SHORT).
    need_nodes = _int_flag(sys.argv, "--require-nodes")
    node_gpus = _int_flag(sys.argv, "--node-gpus") or 1
    need_free_total = _int_flag(sys.argv, "--require-free")
    if need_nodes is not None or need_free_total is not None:
        free_per_node = [a - u for _, a, u in per_node]  # already capped: u ≤ a, so free ≥ 0
        met, reasons = evaluate_requirement(
            free_per_node, need_nodes=need_nodes, node_gpus=node_gpus, need_free_total=need_free_total
        )
        print(f"  {'✅ capacity OK' if met else '❌ capacity SHORT'} — requirement check:")
        for r in reasons:
            print(f"    {r}")
        return 0 if met else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

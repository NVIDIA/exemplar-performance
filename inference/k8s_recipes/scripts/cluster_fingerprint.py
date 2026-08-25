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

"""cluster_fingerprint.py — capture a run's CLUSTER fingerprint into run_meta.cluster_info.

The export's `cluster_details` column is rich only when a run recorded WHERE it ran. The bench pod has limited
node-label RBAC, so the orchestration (which has kubectl access) captures it: read `kubectl get nodes -o json`,
distil the GPU nodes' labels/status into a small `cluster_info` dict, and merge it into the curated
`run_meta.json`. Pure parse core (`fingerprint_from_nodes`) → deterministic + unit-testable without a cluster.

Usage:
  cluster_fingerprint.py --context <kube-context>                 # print cluster_info JSON
  cluster_fingerprint.py --context <ctx> --into <run_meta.json>   # merge cluster_info into an existing run_meta
  cluster_fingerprint.py --nodes-json <file>                      # parse a saved `kubectl get nodes -o json`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_GPU_KEY = "nvidia.com/gpu"


def _provider(provider_id: str) -> str | None:
    if not provider_id:
        return None
    scheme = provider_id.split("://", 1)[0].lower()
    return {"aws": "aws", "gce": "gcp", "azure": "azure"}.get(scheme, scheme or None)


def fingerprint_from_nodes(nodes: list) -> dict:
    """Pure: a list of node objects (kubectl get nodes -o json `.items`) → the cluster_info dict. Considers only
    GPU nodes (allocatable nvidia.com/gpu > 0); derives the shape from the FIRST GPU node + counts the rest.
    Every field degrades to None/absent rather than guessing, so the blob is honest on partial data."""
    gpu_nodes = []
    for n in nodes:
        alloc = (n.get("status") or {}).get("allocatable") or {}
        try:
            g = int(alloc.get(_GPU_KEY, 0))
        except (TypeError, ValueError):
            g = 0
        if g > 0:
            gpu_nodes.append((n, g))
    if not gpu_nodes:
        return {"node_count": 0}
    first, gpus = gpu_nodes[0]
    lbl = (first.get("metadata") or {}).get("labels") or {}
    node_info = (first.get("status") or {}).get("nodeInfo") or {}
    info = {
        "provider": _provider((first.get("spec") or {}).get("providerID") or ""),
        "region": lbl.get("topology.kubernetes.io/region") or lbl.get("failure-domain.beta.kubernetes.io/region"),
        "zone": lbl.get("topology.kubernetes.io/zone") or lbl.get("failure-domain.beta.kubernetes.io/zone"),
        "instance_type": lbl.get("node.kubernetes.io/instance-type") or lbl.get("beta.kubernetes.io/instance-type"),
        "gpu_product": lbl.get("nvidia.com/gpu.product"),
        "gpus_per_node": gpus,
        "k8s_version": node_info.get("kubeletVersion"),
        "node_count": len(gpu_nodes),
    }
    return {k: v for k, v in info.items() if v is not None}


def capture(context: str | None) -> dict:
    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    if context:
        cmd = ["kubectl", "--context", context, "get", "nodes", "-o", "json"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cluster_fingerprint: kubectl failed: {r.stderr.strip()[:200]}")
    return fingerprint_from_nodes(json.loads(r.stdout).get("items") or [])


def merge_into(run_meta_path: Path, cluster_info: dict) -> None:
    """Merge cluster_info into an existing run_meta.json (adds/refreshes the `cluster_info` key)."""
    meta = json.loads(run_meta_path.read_text()) if run_meta_path.is_file() else {}
    meta["cluster_info"] = cluster_info
    run_meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", help="kube-context to query")
    ap.add_argument("--into", help="run_meta.json to merge cluster_info into")
    ap.add_argument("--nodes-json", help="parse a saved `kubectl get nodes -o json` instead of querying")
    args = ap.parse_args(argv[1:])
    if args.nodes_json:
        info = fingerprint_from_nodes(json.loads(Path(args.nodes_json).read_text()).get("items") or [])
    else:
        info = capture(args.context)
    if args.into:
        merge_into(Path(args.into), info)
        print(f"[cluster_fingerprint] merged cluster_info into {args.into}: {json.dumps(info)}")
    else:
        print(json.dumps(info, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

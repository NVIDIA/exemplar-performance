#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

"""selftest_cluster_fingerprint.py — the pure parse core of cluster_fingerprint (no cluster needed)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_fingerprint as cf  # noqa: E402

_fail = 0


def check(label, cond):
    global _fail
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        _fail += 1


def _gpu_node(name):
    return {
        "metadata": {
            "name": name,
            "labels": {
                "topology.kubernetes.io/region": "us-east-2",
                "topology.kubernetes.io/zone": "us-east-2a",
                "node.kubernetes.io/instance-type": "p6e-gb300.48xlarge",
                "nvidia.com/gpu.product": "NVIDIA-GB300",
            },
        },
        "spec": {"providerID": "aws:///us-east-2a/i-0abc"},
        "status": {
            "allocatable": {"nvidia.com/gpu": "4"},
            "nodeInfo": {"kubeletVersion": "v1.29.4"},
        },
    }


NODES = [
    _gpu_node("gpu-0"),
    _gpu_node("gpu-1"),
    {"metadata": {"name": "cpu-0", "labels": {}}, "status": {"allocatable": {}}},
]  # non-GPU → ignored

fp = cf.fingerprint_from_nodes(NODES)
check("counts only GPU nodes (2 of 3)", fp.get("node_count") == 2)
check("gpu_product from the nvidia label", fp.get("gpu_product") == "NVIDIA-GB300")
check("gpus_per_node from allocatable", fp.get("gpus_per_node") == 4)
check(
    "region + zone + instance_type + k8s_version",
    fp.get("region") == "us-east-2"
    and fp.get("zone") == "us-east-2a"
    and fp.get("instance_type") == "p6e-gb300.48xlarge"
    and fp.get("k8s_version") == "v1.29.4",
)
check("provider derived from providerID scheme (aws)", fp.get("provider") == "aws")

check(
    "no GPU nodes → {node_count: 0} (honest, not a guess)",
    cf.fingerprint_from_nodes([{"status": {"allocatable": {}}}]) == {"node_count": 0},
)
check(
    "partial labels degrade to absent keys (never fabricated)",
    "region" not in cf.fingerprint_from_nodes([{"status": {"allocatable": {"nvidia.com/gpu": "8"}}}]),
)

# merge_into round-trips into a run_meta.json
with tempfile.TemporaryDirectory() as td:
    rm = Path(td) / "run_meta.json"
    rm.write_text(json.dumps({"run_id": "t1", "cluster": "cluster-b"}))
    cf.merge_into(rm, fp)
    got = json.loads(rm.read_text())
    check(
        "merge_into adds cluster_info, preserves existing run_meta keys",
        got["cluster_info"]["gpu_product"] == "NVIDIA-GB300" and got["run_id"] == "t1",
    )

print(f"\nselftest_cluster_fingerprint: {'all checks passed' if _fail == 0 else f'{_fail} FAILED'}")
raise SystemExit(1 if _fail else 0)

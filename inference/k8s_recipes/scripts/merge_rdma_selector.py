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

"""merge_rdma_selector.py — deploy-time RDMA nodeSelector override.

Reads Kubernetes manifests from stdin, patches the nodeSelector of disagg worker
Deployments (role: prefill or decode) by replacing baked RDMA-indicator labels with
the cluster-specific ones in RDMA_NODE_SELECTOR, then writes the result to stdout.

When RDMA_NODE_SELECTOR is unset or empty, passes stdin through unchanged (no-op).

Called from deploy.sh as a pipeline stage between envsubst and kubectl apply:

  envsubst "$wl" < workers.yaml | merge_rdma_selector.py | kubectl apply -f -

RDMA_NODE_SELECTOR format (set by probe-fabric or manually in the cluster profile):
  key1=value1,key2=value2
  e.g. feature.node.kubernetes.io/rdma.available=true,feature.node.kubernetes.io/rdma.capable=true

Exit 0 always — failures fall back to passthrough so deploy is never blocked.
"""

import os
import sys

try:
    import yaml
except ImportError:
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)

_RDMA_PATTERNS = ("rdma", "infiniband")


def _is_rdma_label(key: str) -> bool:
    k = key.lower()
    return any(p in k for p in _RDMA_PATTERNS)


def parse_selector(raw: str) -> dict:
    """'key1=val1,key2=val2' → {key1: val1, key2: val2}."""
    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip().strip('"')
        if k:
            out[k] = v
    return out


def is_worker_deployment(doc: dict) -> bool:
    """True for disagg worker Deployments (pod template label role=prefill or decode)."""
    if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
        return False
    pod_labels = (doc.get("spec") or {}).get("template", {}).get("metadata", {}).get("labels") or {}
    return pod_labels.get("role") in ("prefill", "decode")


def patch_node_selector(doc: dict, rdma_labels: dict) -> dict:
    """Replace RDMA-indicator labels in the pod nodeSelector with rdma_labels."""
    try:
        node_sel = doc["spec"]["template"]["spec"].setdefault("nodeSelector", {})
    except (KeyError, TypeError):
        return doc
    for key in [k for k in node_sel if _is_rdma_label(k)]:
        del node_sel[key]
    node_sel.update(rdma_labels)
    return doc


def main() -> None:
    content = sys.stdin.read()
    raw = (os.environ.get("RDMA_NODE_SELECTOR") or "").strip()

    if not raw:
        sys.stdout.write(content)
        return

    rdma_labels = parse_selector(raw)
    if not rdma_labels:
        sys.stdout.write(content)
        return

    try:
        docs = [d for d in yaml.safe_load_all(content) if d is not None]
        n_patched = 0
        for doc in docs:
            if is_worker_deployment(doc):
                patch_node_selector(doc, rdma_labels)
                n_patched += 1
        if n_patched:
            print(
                f"merge_rdma_selector: patched nodeSelector on {n_patched} worker deployment(s) "
                f"→ {list(rdma_labels)}",
                file=sys.stderr,
            )
        sys.stdout.write(yaml.dump_all(docs, default_flow_style=False, allow_unicode=True))
    except Exception as e:
        print(f"merge_rdma_selector: warning — patch failed ({e}), applying unmodified", file=sys.stderr)
        sys.stdout.write(content)


if __name__ == "__main__":
    main()

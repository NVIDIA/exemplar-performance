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

"""repro_discover_nodes.py <profile> --need-gpus G --count N [--product PRODUCT]

Discover the N best FREE serving nodes on a cluster for the reproducibility-floor campaign — one distinct
serving node per parallel repeat (parallel_repro.sh's SERVING_NODES). Read-only; never mutates the cluster.

Free is REAL free: allocatable-minus-used counted across pods in ANY namespace (other teams hold GPUs on the
same pool, so a node with idle allocatable in OUR namespace may still be full). Reuses capacity.py's DRA-aware
accounting (the same numbers `llmb-k8s capacity` prints) so discovery can never disagree with the human view.

Prints, to stdout, the chosen node names — one per line, most-free first — for `SERVING_NODES="$(...)"`. Nothing
is printed if fewer than N nodes each have >=G free GPUs; the exit code says whether the requirement was met:
  exit 0  — N nodes with >=G free GPUs found (names on stdout)
  exit 3  — SHORT: fewer than N qualifying nodes (diagnostic to stderr; partial list still on stdout)
  exit 2  — cluster unreachable / no nodes labelled for the product (diagnostic to stderr)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight as pf  # pinned kubectl (_KUBE_CONTEXT), parse_env, DRA-aware gpu_availability
import capacity as cap  # _alloc (allocatable GPUs per node)


def free_per_node(nodes, pods, claims):
    """PURE — [(node_name, free_gpus), ...], most-free first. free = allocatable - used, capped at >=0.
    `used` is the DRA-aware per-node usage across ALL namespaces (from preflight.gpu_availability)."""
    avail = pf.gpu_availability(nodes, pods, claims, set(), "")  # no taint/arch filter — all product nodes
    used = avail["used"]
    rows = []
    for n in nodes:
        name = n["metadata"]["name"]
        alloc = cap._alloc(n)
        u = min(used.get(name, 0), alloc)
        rows.append((name, max(0, alloc - u)))
    return sorted(rows, key=lambda r: r[1], reverse=True)


def choose(free_rows, need_gpus: int, count: int):
    """PURE — pick up to `count` node names each with >=need_gpus free, most-free first.
    Returns (chosen_names, met) where met is True iff len(chosen)==count."""
    chosen = [name for name, free in free_rows if free >= need_gpus][:count]
    return chosen, len(chosen) == count


def _int_flag(argv, flag, default=None):
    if flag not in argv:
        return default
    i = argv.index(flag)
    if i + 1 >= len(argv):
        sys.exit(f"repro_discover_nodes: {flag} needs an integer value")
    try:
        return int(argv[i + 1])
    except ValueError:
        sys.exit(f"repro_discover_nodes: {flag} value must be an integer, got {argv[i + 1]!r}")


def main() -> int:
    argv = sys.argv[1:]
    positional = [a for a in argv if not a.startswith("--")]
    # first positional is the profile; ints after flags are consumed by _int_flag
    if not positional:
        sys.exit(__doc__)
    profile = positional[0]
    need_gpus = _int_flag(argv, "--need-gpus")
    count = _int_flag(argv, "--count")
    if need_gpus is None or count is None:
        sys.exit("repro_discover_nodes: --need-gpus G and --count N are required")

    prof_path = pf.ROOT / "cluster-profiles" / f"{profile}.env"
    if not prof_path.is_file():
        print(f"repro_discover_nodes: no profile at {prof_path}", file=sys.stderr)
        return 2
    prof = pf.parse_env(prof_path)
    pf._KUBE_CONTEXT = (prof.get("KUBE_CONTEXT") or "").strip()

    product = None
    if "--product" in argv:
        i = argv.index("--product")
        product = argv[i + 1] if i + 1 < len(argv) else None
    product = (product or prof.get("GPU_PRODUCT", "")).strip()
    if not product:
        print(f"repro_discover_nodes: no GPU_PRODUCT in {profile}.env and no --product given", file=sys.stderr)
        return 2

    rc, nout, _ = pf.krun(["get", "nodes", "-l", f"nvidia.com/gpu.product={product}", "-o", "json"])
    if rc != 0 or not nout:
        print(
            f"repro_discover_nodes: no nodes for nvidia.com/gpu.product={product} " f"(or cluster unreachable)",
            file=sys.stderr,
        )
        return 2
    _, pout, _ = pf.krun(["get", "pods", "-A", "-o", "json"])
    rc_dra, dcout, _ = pf.krun(["get", "resourceclaims", "-A", "-o", "json"])
    nodes = json.loads(nout).get("items", [])
    pods = json.loads(pout or '{"items":[]}').get("items", [])
    claims = json.loads(dcout or '{"items":[]}').get("items", []) if rc_dra == 0 else []

    rows = free_per_node(nodes, pods, claims)
    chosen, met = choose(rows, need_gpus, count)
    for name in chosen:
        print(name)
    if not met:
        print(
            f"repro_discover_nodes: SHORT — need {count} node(s) with >={need_gpus} free GPUs, "
            f"found {len(chosen)} (product={product}). Free per node: " + ", ".join(f"{n}={f}" for n, f in rows),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Reconcile variables referenced by rendered manifests with cluster-profile and runtime values.

Each cell is checked dynamically: required profile variables must be present and non-empty, runtime variables
are exempt, and explicitly optional profile variables may remain empty. Unrecognized variables produce a
warning and CI verifies that profile-backed variables are documented in the profile template. The same pure
logic is used by preflight, profile validation, and CI.
"""

from __future__ import annotations

import re
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "cluster-profiles" / "_template.env.example"

_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# VarGap: one reconciliation finding. level ∈ {FAIL, WARN}. `var` names the offending ${VAR}.
VarGap = namedtuple("VarGap", ["var", "level", "message", "fix"])

# Variables filled at run time rather than sourced from the cluster profile.
RUNTIME_VARS = frozenset(
    {
        "RUN_ID",
        "LIVE_RUNGS",
        "SERVING_NODE",
        "KARCH",
        "KUBECTL_VERSION",
        "REQUEST_COUNT_MULTIPLIER",
        "RUN_WALL_SECONDS",
        "STOP_STAT",
        "TTFT_LIMIT_MS",
        "TPOT_LIMIT_MS",
        "CACHE_BUST",
        "DCGM_EXPORTER_URL",  # both carry a shell default (: "${X:=}") and are per-run/optional
        "ARTIFACTS_ROOT",
        "SERVER_URL",
        "TPS",
        "CONCURRENCY",
        "host",
    }
)

# Profile variables for which an empty value is a documented choice.
EMPTY_VALID = frozenset(
    {
        "BENCH_NODE_SELECTOR",  # empty = schedule anywhere
        "BENCH_CPU_REQUEST",  # has a default (16)
        "OWNER",  # defaults to whoami
        "CONNECT_CMD",  # empty = fleet falls back to tsh
        "GPU_TYPE",  # empty = derive from GPU_PRODUCT
        "ARCH",  # optional hint
        "MODEL_CACHE_SUBPATH",  # empty = PVC root IS the HF cache root
        "ARTIFACTS_SIZE",
        "CONTROL_SIZE",
        "ARTIFACTS_ACCESS_MODE",
        # UCX fabric values have documented runtime defaults; preflight reports fabric-specific advisories.
        "RDMA_UCX_IB_ADDR_TYPE",  # empty = native InfiniBand
        "RDMA_UCX_NET_DEVICES",  # empty → "all" default
        "RDMA_UCX_TLS",  # empty → run.sh default (the InfiniBand list the recipes used to bake)
        "RDMA_UCX_MAX_RNDV_RAILS",  # empty → run.sh default (4)
        "RDMA_NODE_SELECTOR",
        "RDMA_RESOURCE",
        "NVLINK_MULTICAST_IMEX",
        "IMEX_CLAIM_TEMPLATE",  # empty = fusion absent (deploy strips the forced flag)
        "HF_SECRET",  # some cells' cache is pre-staged; empty is tolerated (download-time only)
    }
)


def profile_vars(template: Path = TEMPLATE) -> set:
    """The canonical profile-key universe: every `KEY=` at column 0 in _template.env.example (uncommented).
    This is what makes reconciliation dynamic — the template is the single source of 'what a profile provides'.
    """
    out: set = set()
    if not Path(template).exists():
        return out
    for ln in Path(template).read_text().splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=", ln)
        if m:
            out.add(m.group(1))
    return out


def referenced_vars(cell: Path) -> set:
    """Every ${VAR} the cell's rendered/*.yaml reference. Empty when the cell has no rendered/ (declaration-only)."""
    refs: set = set()
    rdir = Path(cell) / "rendered"
    if not rdir.is_dir():
        return refs
    for f in sorted(rdir.glob("*.yaml")):
        refs |= set(_VAR_RE.findall(f.read_text()))
    return refs


def _empty(prof: dict, var: str) -> bool:
    return not (prof.get(var) or "").strip()


def reconcile(
    cell: Path,
    prof: dict,
    *,
    template: Path = TEMPLATE,
    known_profile_vars: Optional[set] = None,
) -> list:
    """PURE. Reconcile ONE cell's referenced ${VAR}s against a resolved profile dict. Returns [VarGap].
    FAIL — a referenced var that is a profile var, not EMPTY_VALID, and empty/absent in the profile (the
           NO_INTERNET_DNS_IP case: envsubst would resolve it to '' and the workload dies at runtime).
    WARN — a referenced var that is neither a known profile var nor a runtime placeholder (an undocumented
           var; probably a new profile key missing from the template — CI hard-fails that separately).
    """
    pvars = known_profile_vars if known_profile_vars is not None else profile_vars(template)
    gaps: list = []
    for var in sorted(referenced_vars(cell)):
        if var in RUNTIME_VARS:
            continue
        if var in pvars:
            if var in EMPTY_VALID:
                continue
            if _empty(prof, var):
                gaps.append(
                    VarGap(
                        var,
                        "FAIL",
                        f"manifest references ${{{var}}} but the profile leaves it EMPTY — envsubst resolves it to "
                        f"an empty string and the workload fails at RUNTIME (not caught by a leftover-${{VAR}} scan)",
                        fix=f"set {var} in the cluster profile (a non-empty value this cell requires)",
                    )
                )
        else:
            gaps.append(
                VarGap(
                    var,
                    "WARN",
                    f"manifest references ${{{var}}} which is neither a documented profile var nor a known runtime "
                    "placeholder — if it is profile-sourced, add it to cluster-profiles/_template.env.example",
                    fix=None,
                )
            )
    return gaps


def reconcile_broad(prof: dict, cells: Iterable[Path], *, compatible=None, template: Path = TEMPLATE) -> list:
    """PURE-ish. Cluster-level reconciliation for `profile validate`/`init` WITHOUT a chosen cell: reconcile
    every committed cell that is target-compatible with this cluster and aggregate UNIQUE gaps. `compatible`
    is an injected predicate cell_dir->bool (default: all cells); the live caller passes a GPU/arch match so a
    B200 profile isn't faulted for a GB300-only cell's vars. De-dups by (var, level)."""
    pvars = profile_vars(template)
    seen: dict = {}
    for cell in cells:
        if compatible is not None and not compatible(cell):
            continue
        for g in reconcile(cell, prof, template=template, known_profile_vars=pvars):
            seen.setdefault((g.var, g.level), g)
    # FAILs first, then WARNs; stable by var name.
    return sorted(seen.values(), key=lambda g: (0 if g.level == "FAIL" else 1, g.var))


def template_coverage_gaps(cells: Iterable[Path], *, template: Path = TEMPLATE) -> list:
    """PURE. CI guard: every ${VAR} any committed cell references must be either a documented profile var
    (in _template.env.example) or a known RUNTIME_VAR. A var that is neither means someone added a cell that
    needs a new ${VAR} without documenting it — so init/preflight can't know to verify it. Returns a sorted
    list of (var, [cell_names]) offenders (empty = fully covered)."""
    pvars = profile_vars(template)
    offenders: dict = {}
    for cell in cells:
        for var in referenced_vars(cell):
            if var in pvars or var in RUNTIME_VARS:
                continue
            offenders.setdefault(var, []).append(Path(cell).name)
    return sorted((var, sorted(set(names))) for var, names in offenders.items())


if __name__ == "__main__":
    # `python3 scripts/manifest_vars.py` — run the CI template-coverage guard across all committed cells.
    import sys

    cells = sorted({p.parent for p in (ROOT / "recipes").glob("**/recipe.yaml")})
    gaps = template_coverage_gaps(cells)
    if gaps:
        print(
            "manifest-vars: template coverage GAPS — these ${VAR}s are referenced but neither documented in "
            "cluster-profiles/_template.env.example nor a known runtime placeholder:"
        )
        for var, names in gaps:
            print(f"  ✗ {var}  (referenced by: {', '.join(names)})")
        print(
            "  → add each to _template.env.example (if profile-sourced) or manifest_vars.RUNTIME_VARS "
            "(if runtime-filled)"
        )
        sys.exit(1)
    print(
        f"manifest-vars: template coverage OK — every referenced ${{VAR}} across {len(cells)} cells is "
        "documented (profile var) or a known runtime placeholder"
    )
    sys.exit(0)

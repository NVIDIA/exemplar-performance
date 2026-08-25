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

"""Offline checks for rendered-manifest and cluster-profile variable reconciliation.

Required variables referenced by a manifest must not resolve to empty values. Variables that permit empty
values and runtime-provided placeholders are excluded from that check. The suite also verifies warnings for
undocumented variables and template coverage.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mv = _load("manifest_vars")

fails: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# A tiny template documenting the profile vars used in the fixtures.
_TEMPLATE = """\
NAMESPACE=""
GPU_PRODUCT=""
IMAGE_PULL_SECRET=""
NO_INTERNET_DNS_IP=""
BENCH_NODE_SELECTOR=""
SOME_NEW_VAR=""
"""


def _cell(tmp: Path, manifest_body: str) -> Path:
    cell = tmp / "cell"
    (cell / "rendered").mkdir(parents=True)
    (cell / "rendered" / "job.yaml").write_text(manifest_body)
    (cell / "recipe.yaml").write_text("envelope: {gpu_type: B200}\n")
    return cell


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    tmpl = tmp / "_template.env.example"
    tmpl.write_text(_TEMPLATE)

    # profile_vars reads the KEYs from the template
    pv = mv.profile_vars(tmpl)
    check(
        "profile_vars: reads documented KEYs from the template",
        {"NAMESPACE", "NO_INTERNET_DNS_IP", "BENCH_NODE_SELECTOR"} <= pv and "not_a_key" not in pv,
        str(pv),
    )

    # referenced_vars enumerates every ${VAR} in rendered/*.yaml
    manifest = (
        "env:\n"
        "  - SERVING_NO_INTERNET_DNS_IP=${NO_INTERNET_DNS_IP}\n"
        "  - NS=${NAMESPACE}\n"
        "  - SEL=${BENCH_NODE_SELECTOR}\n"
        "  - RID=${RUN_ID}\n"
    )
    cell = _cell(tmp, manifest)
    refs = mv.referenced_vars(cell)
    check(
        "referenced_vars: enumerates all ${VAR}s",
        refs == {"NO_INTERNET_DNS_IP", "NAMESPACE", "BENCH_NODE_SELECTOR", "RUN_ID"},
        str(refs),
    )

    # A required referenced profile variable cannot be empty.
    prof_empty = {
        "NAMESPACE": "ns",
        "GPU_PRODUCT": "NVIDIA-B200",
        "IMAGE_PULL_SECRET": "s",
        "NO_INTERNET_DNS_IP": "",
        "BENCH_NODE_SELECTOR": "",
    }
    gaps = mv.reconcile(cell, prof_empty, template=tmpl)
    dns = [g for g in gaps if g.var == "NO_INTERNET_DNS_IP"]
    check(
        "reconcile: referenced profile var EMPTY → FAIL",
        len(dns) == 1 and dns[0].level == "FAIL",
        str(gaps),
    )
    check(
        "reconcile: empty-valid var (BENCH_NODE_SELECTOR) empty → NOT flagged",
        not any(g.var == "BENCH_NODE_SELECTOR" for g in gaps),
    )
    check(
        "reconcile: runtime placeholder (RUN_ID) → never a profile gap",
        not any(g.var == "RUN_ID" for g in gaps),
    )

    # fixed: populate the IP → no FAIL
    prof_ok = dict(prof_empty, NO_INTERNET_DNS_IP="172.20.0.10")
    gaps_ok = mv.reconcile(cell, prof_ok, template=tmpl)
    check(
        "reconcile: populated var → no FAIL",
        not any(g.level == "FAIL" for g in gaps_ok),
        str(gaps_ok),
    )

    # undocumented var referenced → WARN (neither profile var nor runtime placeholder)
    cell2 = _cell(tmp / "u", "env:\n  - X=${TOTALLY_UNDOCUMENTED_VAR}\n")
    gaps2 = mv.reconcile(cell2, prof_ok, template=tmpl)
    check(
        "reconcile: undocumented ${VAR} → WARN",
        any(g.var == "TOTALLY_UNDOCUMENTED_VAR" and g.level == "WARN" for g in gaps2),
        str(gaps2),
    )

    # CI template-coverage guard: undocumented profile-ish var referenced by a cell → offender
    cov = mv.template_coverage_gaps([cell2], template=tmpl)
    check(
        "template_coverage_gaps: undocumented var flagged for CI",
        any(v == "TOTALLY_UNDOCUMENTED_VAR" for v, _ in cov),
        str(cov),
    )
    # a cell using only documented + runtime vars → no coverage gap
    cov_ok = mv.template_coverage_gaps([cell], template=tmpl)
    check(
        "template_coverage_gaps: fully-documented cell → no offenders",
        cov_ok == [],
        str(cov_ok),
    )

    # reconcile_broad: aggregate across cells, honoring the compat predicate
    broad = mv.reconcile_broad(prof_empty, [cell, cell2], compatible=lambda c: True, template=tmpl)
    check(
        "reconcile_broad: aggregates FAIL + WARN across compatible cells",
        any(g.level == "FAIL" for g in broad) and any(g.level == "WARN" for g in broad),
        str(broad),
    )
    broad_excl = mv.reconcile_broad(prof_empty, [cell, cell2], compatible=lambda c: c != cell, template=tmpl)
    check(
        "reconcile_broad: compat predicate excludes a cell (no FAIL from the excluded one)",
        not any(g.var == "NO_INTERNET_DNS_IP" for g in broad_excl),
        str(broad_excl),
    )


# The REAL repo must stay covered (this also runs as a CI guard via check_invariants + manifest_vars __main__).
_real_cells = sorted({p.parent for p in (SCRIPTS.parent / "recipes").glob("**/recipe.yaml")})
check(
    "template_coverage_gaps: the committed recipe tree is fully covered (no undocumented ${VAR})",
    mv.template_coverage_gaps(_real_cells) == [],
    str(mv.template_coverage_gaps(_real_cells)),
)


if fails:
    print(f"\n{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("\nall manifest-vars selftests passed")

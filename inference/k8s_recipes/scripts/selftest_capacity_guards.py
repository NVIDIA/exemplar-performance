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

"""Offline checks for install and preflight capacity safeguards.

The tests cover free node capacity, failed and immutable download Jobs, and
per-claim model-cache capacity. Source checks use regular expressions so code
formatting does not change their meaning.
"""

import pathlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


import preflight as pf  # noqa: E402

PREFLIGHT = (ROOT / "scripts" / "preflight.py").read_text()
INSTALL = (ROOT / "scripts" / "install.py").read_text()

# ── 1. whole_node must gate on FREE, not ALLOCATABLE ─────────────────────────────────────────────────────
# Arithmetic first, against preflight's own quantity parsers, so this breaks if they change meaning.
NODE_CPU = "192"  # a representative GPU node's allocatable cpu
NODE_MEM = "2113352952Ki"  # the same node's allocatable memory (= 2015 GiB)
RESIDENT = ["16", "200m", "15m"]  # a co-tenant holding 16 cores, plus DaemonSets

alloc_m = pf._cpu_millicores(NODE_CPU)
free_m = alloc_m - sum((pf._cpu_millicores(x) or 0) for x in RESIDENT)
check("parsers: cpu cores -> millicores", alloc_m == 192000, str(alloc_m))
check(
    "parsers: Ki memory -> whole Gi",
    pf._mem_gib(NODE_MEM) == 2015,
    str(pf._mem_gib(NODE_MEM)),
)

too_big = pf._cpu_millicores("180")
check("an over-large request fits allocatable capacity", too_big <= alloc_m)
check("the same request exceeds free capacity", too_big > free_m, f"free={free_m}m")
derived = pf._cpu_millicores("163")
check(
    "a headroom-derived request fits FREE even with a co-tenant resident",
    derived <= free_m,
    f"derived={derived}m free={free_m}m",
)

# Verify that the gate uses free capacity rather than allocatable capacity.
check(
    "1: the gate computes free capacity from allocatable and requested CPU",
    bool(re.search(r"_free_cpu\s*=\s*\(\s*_alloc_cpu\s*-\s*_used_cpu", PREFLIGHT)),
)
check(
    "1: the gate FAILS on exceeding FREE, naming the node and the arithmetic",
    "exceeds FREE cpu on" in PREFLIGHT,
)
check(
    "1: it sums only Running/Pending pods ON THE CANDIDATE NODE",
    'get("nodeName") != _cand' in PREFLIGHT,
)
check(
    "1: an unreadable pod list renders UNKNOWN, never a clean bill of health",
    "FREE capacity is UNKNOWN" in PREFLIGHT and "not a clean bill of health" in PREFLIGHT,
)
check(
    "1: preflight exposes pods so the gate can see them",
    'avail["pods"] = pods' in PREFLIGHT,
)
check(
    "1: absent container requests coerce to 0 rather than poisoning the sum",
    "poisoning the sum" in PREFLIGHT,
)

# ── 2. a failed apply must never render as success ───────────────────────────────────────────────────────
check(
    "2: a failed download apply is recorded, not merely printed",
    "failures.append((job_name" in INSTALL,
)
check(
    "2: 'All done' is gated on there being no failures",
    "if failures:" in INSTALL and INSTALL.index("if failures:") < INSTALL.index('"  ✓ All done. To benchmark:"'),
)
check(
    "2: the failure summary names the consequence",
    "CrashLoop on a missing snapshot" in INSTALL,
)

# ── 3. an immutable Job spec must be reconciled, not skipped ─────────────────────────────────────────────
check(
    "3: an immutable-spec apply failure triggers delete+recreate",
    bool(re.search(r'if\s+p\.returncode\s*!=\s*0\s*and\s*"immutable"\s+in', INSTALL)),
)
check(
    "3: the replace path cannot run for an ACTIVE Job (never kills a live download)",
    "can never kill a download in\n        # flight" in INSTALL or "never kill a download in" in INSTALL,
)

# ── 4. auto-download must be bounded — now PER RESOLVED CLAIM, on BOTH doors ─────────────────────────────
# The old assertions checked for a single whole-union total measured against ONE claim's free space. That
# arithmetic answered the wrong question once models could land in different claims, so the check now asserts
# the stronger property: the sizing/refusal happens inside capacity_gate, keyed on the claim each model
# actually lands in, and BOTH the headless and interactive doors call it.
check(
    "4: the download size is computed per RESOLVED claim",
    "def capacity_gate(" in INSTALL and bool(re.search(r"by_claim\.setdefault\(\s*cache_by_repo\.get\(", INSTALL)),
)
check(
    "4: a download that provably exceeds free space is REFUSED",
    'pvc_space_verdict(free - total, total) == "block"' in INSTALL
    and "'WOULD NOT FIT' if plan_only else 'REFUSING'" in INSTALL,
)
# --live-plan is the PROBING mode; it must MEASURE and report, not skip the check. A live plan proposed
# ~1104 GiB into a claim with ~0.5 TiB free and never said the aggregate would not fit.
check(
    "4: live-plan still measures capacity (only fully-offline --dry-run skips it)",
    "if plan_only and not do_probe:" in INSTALL,
)
check(
    "4: ...but a plan never removes a model from the plan",
    "if blocked and not plan_only:" in INSTALL,
)
# The headless door must consult the cache BEFORE queueing the union, or a complete model is re-fetched.
check(
    "4: headless checks what is already on the PVC before downloading",
    "probe_model_on_pvc(" in INSTALL
    and "sentinel_worthy(_facts)" in INSTALL
    and "complete — skipping download" in INSTALL,
)
check(
    "4: unknown free space REFUSES a large write (it used to warn and proceed)",
    "is UNKNOWN and this is a" in INSTALL and "Unknown capacity does not authorise it" in INSTALL,
)
check(
    "4: ...with an explicit escape hatch, so it is careful and not merely impossible",
    "--allow-unmeasured-download" in INSTALL and "allow_unmeasured" in INSTALL,
)
# THE ESCAPE HATCH MUST NOT OPEN ITSELF. The env default was
# `os.environ.get(...).strip() not in ("", "0")`, so `false`, `no`, `off` and `disabled` all ENABLED the
# override — an operator exporting `=false` to disable the risk authorised a multi-hundred-GB write onto a
# claim whose free space is unknown. And with `store_true` alone there was no way to say NO on the command
# line, so an exported value could not be countermanded for a single run.
import contextlib as _ctx  # noqa: E402
import io as _io  # noqa: E402
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import install as _install  # noqa: E402

_flag_bad = []
for _v, _want in (
    ("1", True),
    ("true", True),
    ("TRUE", True),
    ("yes", True),
    ("on", True),
    ("", False),
    ("0", False),
    ("false", False),
    ("no", False),
    ("off", False),
    ("disabled", False),
    ("maybe", False),
):
    _os.environ["LLMB_ALLOW_UNMEASURED_DOWNLOAD"] = _v
    with _ctx.redirect_stderr(_io.StringIO()):
        _got = _install._env_flag("LLMB_ALLOW_UNMEASURED_DOWNLOAD")
    if _got is not _want:
        _flag_bad.append(f"{_v!r}->{_got} (want {_want})")
_os.environ.pop("LLMB_ALLOW_UNMEASURED_DOWNLOAD", None)
check(
    "4: LLMB_ALLOW_UNMEASURED_DOWNLOAD only enables on an explicit true-word "
    "('false'/'no'/'off' must DISABLE, not enable)",
    not _flag_bad,
    "; ".join(_flag_bad),
)
check(
    "4: ...and an unrecognised value does not silently authorise",
    (
        not _install._env_flag("LLMB_ALLOW_UNMEASURED_DOWNLOAD")
        if _os.environ.setdefault("LLMB_ALLOW_UNMEASURED_DOWNLOAD", "sure")
        else True
    ),
)
_os.environ.pop("LLMB_ALLOW_UNMEASURED_DOWNLOAD", None)
check(
    "4: ...and there is a way to say NO on the command line",
    "--no-allow-unmeasured-download" in INSTALL and 'action="store_false"' in INSTALL,
)
_capacity_gate_calls = re.findall(r"\bcapacity_gate\s*\(\s*dl_models\s*,\s*cache_by_repo\b", INSTALL)
check(
    "4: BOTH doors run the gate (headless AND interactive)",
    len(_capacity_gate_calls) == 2,
    str(len(_capacity_gate_calls)),
)

# ── the profile-pinned cache (what sent weights to a PVC the server never mounts) ─────────────────────────
# STRONGER THAN THE ORIGINAL INTENT: discovery cannot outrank the profile because discovery no longer
# decides anything. There is no in-memory `prof["MODEL_CACHE_PVC"] = <discovered>` assignment left — the
# claim is resolved from the profile FILE by resolve_cache_claim, which every consumer shares.
check(
    "no code path assigns a DISCOVERED cache into the in-memory profile",
    'prof["MODEL_CACHE_PVC"] = _shared' not in INSTALL and '{**prof, "MODEL_CACHE_PVC": shared}' not in INSTALL,
)
MODEL_CACHE = (pathlib.Path(__file__).resolve().parent / "model_cache.py").read_text()
check(
    "the claim comes from the profile only (resolve_cache_claim is the single definition)",
    "def resolve_cache_claim(" in MODEL_CACHE and "name, name_source = resolve_cache_claim(cell, prof)" in INSTALL,
)
check(
    "discovery survives only as ADVICE the operator can act on",
    "render_cache_candidates_advice" in INSTALL and "--adopt-cache" in INSTALL,
)

print()
if FAILED:
    print(f"selftest_capacity_guards: {len(FAILED)} FAILED")
    for f in FAILED:
        print(f"  - {f}")
    raise SystemExit(1)
print("selftest_capacity_guards: all checks passed")

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

"""selftest_multiarch_manifest.py — the pure offline shape core of verify_multiarch_manifest (no registry)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_multiarch_manifest as vm  # noqa: E402

_fail = 0
REG = "nvcr.io/nvidian/gsw"


def check(label, cond):
    global _fail
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        _fail += 1


def _row(repo, digest="a" * 64, platform=None, extra_refs=True):
    r = {
        "task_ref": "t",
        "role": "agent_environment",
        "platform": platform,
        "image_digest_ref": f"{REG}/{repo}@sha256:{digest}",
    }
    if extra_refs:
        r["image_ref"] = f"{REG}/{repo}:tag"
        r["build_result_ref"] = f"{REG}/{repo}:tag"
    return r


# --- arch_suffix detection ---
check(
    "arch_suffix strips -arm64 from a digest ref",
    vm.arch_suffix(f"{REG}/task-workload-arm64@sha256:{'a'*64}") == "arm64",
)
check("arch_suffix on neutral repo → None", vm.arch_suffix(f"{REG}/task-workload@sha256:{'a'*64}") is None)
check("arch_suffix on -amd64 tag ref", vm.arch_suffix(f"{REG}/foo-amd64:v1") == "amd64")
check(
    "arch_suffix does NOT false-match a registry :port",
    vm.arch_suffix("registry.local:5000/task-workload") is None,
)
check(
    "arch_suffix does NOT match 'workload' substrings (only real arch tokens)",
    vm.arch_suffix(f"{REG}/task-workload@sha256:{'a'*64}") is None,
)

# --- mode: multiarch ---
ma = {
    "repository": "task-workload",
    "platform": "multi",
    "tasks": [_row("task-workload"), _row("task-workload", digest="b" * 64)],
}
rep = vm.shape_report(ma)
check("neutral repo + neutral platform + one index digest/row → mode=multiarch", rep["mode"] == "multiarch")
check("multiarch has no violations", rep["violations"] == [])

# --- mode: multiarch with platform ABSENT (not just 'multi') ---
ma2 = {"repository": "task-workload", "tasks": [_row("task-workload")]}
check("absent top-level platform is also neutral → multiarch", vm.shape_report(ma2)["mode"] == "multiarch")

# --- mode: legacy-single (today's committed manifest shape) ---
legacy = {
    "repository": "task-workload-arm64",
    "platform": "linux/arm64",
    "tasks": [
        _row("task-workload-arm64", platform="linux/arm64"),
        _row("task-workload-arm64", digest="b" * 64, platform="linux/arm64"),
    ],
}
rep = vm.shape_report(legacy)
check("all-arm64 refs + single-arch platform → mode=legacy-single", rep["mode"] == "legacy-single")
check(
    "legacy-single is a soft advisory, NOT a hard violation (CI stays green)",
    rep["violations"] == [] and len(rep["advisories"]) == 1,
)

# --- mode: partial (half-migrated → hard failure) ---
partial = {
    "repository": "task-workload",
    "platform": "multi",
    "tasks": [_row("task-workload"), _row("task-workload-arm64", digest="b" * 64)],
}  # one row still arch-pinned
rep = vm.shape_report(partial)
check("mixed neutral + arch-suffixed refs → mode=partial", rep["mode"] == "partial")
check("partial produces hard violations", len(rep["violations"]) >= 1)

# --- neutral refs but a leftover single-arch top-level platform is still partial ---
mixed_plat = {"repository": "task-workload", "platform": "linux/arm64", "tasks": [_row("task-workload")]}
check(
    "neutral refs but platform=linux/arm64 → partial (not silently 'multiarch')",
    vm.shape_report(mixed_plat)["mode"] == "partial",
)

# --- multiarch row missing its index digest → violation ---
bad_digest = {
    "repository": "task-workload",
    "platform": "multi",
    "tasks": [
        {"task_ref": "t", "role": "agent_environment", "platform": None, "image_ref": f"{REG}/task-workload:tag"}
    ],
}  # no image_digest_ref
rep = vm.shape_report(bad_digest)
check(
    "multiarch-shaped row lacking image_digest_ref → violation",
    rep["mode"] == "multiarch" and any("image_digest_ref" in v for v in rep["violations"]),
)

# --- online platform parse (no registry; feed raw index JSON directly) ---
index_raw = {
    "mediaType": "application/vnd.oci.image.index.v1+json",
    "manifests": [
        {"platform": {"os": "linux", "architecture": "amd64"}},
        {"platform": {"os": "linux", "architecture": "arm64"}},
        {
            "platform": {"os": "unknown", "architecture": "unknown"},
            "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
        },
    ],
}
check(
    "platforms_of_raw returns both real arches, skips attestation/unknown",
    vm.platforms_of_raw(index_raw) == {"linux/amd64", "linux/arm64"},
)
check(
    "a bare image manifest (not an index) → empty platform set (single-arch)",
    vm.platforms_of_raw({"mediaType": "application/vnd.oci.image.manifest.v1+json"}) == set(),
)

print(f"\nselftest_multiarch_manifest: {'all checks passed' if _fail == 0 else f'{_fail} FAILED'}")
raise SystemExit(1 if _fail else 0)

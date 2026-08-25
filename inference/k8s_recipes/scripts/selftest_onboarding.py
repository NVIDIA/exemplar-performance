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

"""selftest_onboarding.py — unit tests for Phase 0 onboarding scripts (no cluster needed).

Tests the pure logic in install.py and profile_init.py:
  - PVC free-space math (pvc_space_math)
  - Model-on-PVC detection (probe_model_on_pvc, via injected kubectl)
  - Catalog → model list (catalog_models)
  - Profile-write formatting (format_profile)
  - Namespace-case selection logic (select_namespace, via injected kubectl)
  - Download Job template rendering (render_download_job)
  - Profile env parsing roundtrip

All kubectl calls are injected as mock functions — runs fully offline.
Mirrors the pattern used in scripts/selftest.py and scripts/profile_resolver.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    marker = "PASS" if cond else "FAIL"
    print(f"  {marker}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------------------
# 1. PVC free-space math (pure — in install.py)
# ---------------------------------------------------------------------------

from install import pvc_space_math  # type: ignore[import]

# 3.8 TiB free, selecting a 400 GiB model.
projected, will_fit = pvc_space_math(3800, [400])
check(
    "pvc_space_math: 3800 GiB free - 400 GiB = 3400 GiB, fits",
    abs(projected - 3400) < 1 and will_fit,
    f"projected={projected}, will_fit={will_fit}",
)

# Not enough space.
projected2, will_fit2 = pvc_space_math(200, [400])
check(
    "pvc_space_math: 200 GiB free - 400 GiB = -200 GiB, does NOT fit",
    abs(projected2 - (-200)) < 1 and not will_fit2,
    f"projected={projected2}, will_fit={will_fit2}",
)

# Multiple models: 3800 - 400 - 80 = 3320.
projected3, will_fit3 = pvc_space_math(3800, [400, 80])
check(
    "pvc_space_math: multiple models (400+80 GiB), 3800 GiB free → 3320 GiB",
    abs(projected3 - 3320) < 1 and will_fit3,
    f"projected={projected3}",
)

# Unknown size (None): should not block.
projected4, will_fit4 = pvc_space_math(3800, [None, 400])
check(
    "pvc_space_math: unknown sizes are excluded from sum (only known sizes count)",
    abs(projected4 - 3400) < 1 and will_fit4,
    f"projected={projected4}",
)

# No free-space data: should not block.
projected5, will_fit5 = pvc_space_math(None, [400])
check(
    "pvc_space_math: free_gib=None → returns (None, True) — can't block",
    projected5 is None and will_fit5,
    f"projected={projected5}, will_fit={will_fit5}",
)


# ---------------------------------------------------------------------------
# 2. Catalog → model list
# ---------------------------------------------------------------------------

from install import catalog_models, load_catalog  # type: ignore[import]

# Build a minimal fake catalog + recipe tree in a temp dir.
with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)
    # Recipe 1: nemotron-ultra-3 with pinned revision.
    r1 = td_path / "recipes/llm-perf/256k/nemotron-b200"
    r1.mkdir(parents=True)
    (r1 / "recipe.yaml").write_text(
        "envelope:\n  name: nemotron-b200-test\nserving:\n"
        "  model_repo: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4\n"
        "  model_revision: abc123def456abc123def456abc123def456abc123def456abc123def4561234\n"
    )
    # Recipe 2: glm5 with pinned revision.
    r2 = td_path / "recipes/llm-perf/glm5/glm5-b200"
    r2.mkdir(parents=True)
    (r2 / "recipe.yaml").write_text(
        "envelope:\n  name: glm5-b200-test\nserving:\n"
        "  model_repo: zai-org/GLM-5-FP8\n"
        "  model_revision: 4f96cc5eec29dcee5d6ded54f7ffe889438f9516\n"
    )
    # Recipe 3: duplicate of glm5 (different cell, same model).
    r3 = td_path / "recipes/llm-perf/glm5/glm5-b200-pareto"
    r3.mkdir(parents=True)
    (r3 / "recipe.yaml").write_text(
        "envelope:\n  name: glm5-b200-pareto-test\nserving:\n"
        "  model_repo: zai-org/GLM-5-FP8\n"
        "  model_revision: 4f96cc5eec29dcee5d6ded54f7ffe889438f9516\n"
    )
    # Recipe 4: missing model_repo (should be skipped).
    r4 = td_path / "recipes/llm-perf/workload/no-serving"
    r4.mkdir(parents=True)
    (r4 / "recipe.yaml").write_text("envelope:\n  name: no-serving\nserving: {}\n")

    fake_catalog = [
        {"model": "nemotron-ultra-3", "_path": "recipes/llm-perf/256k/nemotron-b200"},
        {"model": "glm5-fp8", "_path": "recipes/llm-perf/glm5/glm5-b200"},
        {"model": "glm5-fp8", "_path": "recipes/llm-perf/glm5/glm5-b200-pareto"},
        {"model": "no-serving", "_path": "recipes/llm-perf/workload/no-serving"},
    ]

    models = catalog_models(fake_catalog, td_path)

    check(
        "catalog_models: deduplicates by model_repo (2 glm5 cells → 1 model entry)",
        len(models) == 2,
        f"got {len(models)} models: {[m['model_name'] for m in models]}",
    )

    repo_set = {m["model_repo"] for m in models}
    check(
        "catalog_models: contains nemotron and glm5",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4" in repo_set and "zai-org/GLM-5-FP8" in repo_set,
        f"repos={repo_set}",
    )

    glm5 = next((m for m in models if "GLM" in m["model_repo"]), None)
    check(
        "catalog_models: glm5 recipe_count=2 (two cells)",
        glm5 is not None and glm5["recipe_count"] == 2,
        f"recipe_count={glm5['recipe_count'] if glm5 else 'N/A'}",
    )

    check(
        "catalog_models: no-serving entry excluded (missing model_repo)",
        not any(m["model_name"] == "no-serving" for m in models),
        str([m["model_name"] for m in models]),
    )

    check(
        "catalog_models: pinned revision preserved",
        any(m.get("model_revision") == "4f96cc5eec29dcee5d6ded54f7ffe889438f9516" for m in models),
        str([(m["model_name"], m.get("model_revision", "")[:8]) for m in models]),
    )


# ---------------------------------------------------------------------------
# 3. Profile-write formatting (pure)
# ---------------------------------------------------------------------------

from profile_init import format_profile  # type: ignore[import]

profile_text = format_profile(
    cluster="test-cluster",
    context="test-context",
    namespace="test-ns",
    owner="testuser",
    gpu_product="NVIDIA-B200",
    arch="amd64",
    pull_secret="ngc-registry",
    hf_secret="hf-token",
    model_cache_pvc="model-cache",
    artifacts_sc="ebs",
    scheduler="default-scheduler",
    cache_rwx_class="fsx-lustre",
    cache_rwo_class="ebs",
)

check("format_profile: CLUSTER set", 'CLUSTER="test-cluster"' in profile_text)
check("format_profile: NAMESPACE set", 'NAMESPACE="test-ns"' in profile_text)
check("format_profile: KUBE_CONTEXT set", 'KUBE_CONTEXT="test-context"' in profile_text)
check("format_profile: GPU_PRODUCT set", 'GPU_PRODUCT="NVIDIA-B200"' in profile_text)
check("format_profile: ARCH set", 'ARCH="amd64"' in profile_text)
check(
    "format_profile: IMAGE_PULL_SECRET",
    'IMAGE_PULL_SECRET="ngc-registry"' in profile_text,
)
check("format_profile: HF_SECRET", 'HF_SECRET="hf-token"' in profile_text)
check("format_profile: MODEL_CACHE_PVC", 'MODEL_CACHE_PVC="model-cache"' in profile_text)
check(
    "format_profile: MODEL_CACHE_RWX_CLASS",
    'MODEL_CACHE_RWX_CLASS="fsx-lustre"' in profile_text,
)
check(
    "format_profile: MODEL_CACHE_RWO_CLASS",
    'MODEL_CACHE_RWO_CLASS="ebs"' in profile_text,
)
check(
    "format_profile: ARTIFACTS_STORAGE_CLASS",
    'ARTIFACTS_STORAGE_CLASS="ebs"' in profile_text,
)
check(
    "format_profile: SCHEDULER_NAME",
    'SCHEDULER_NAME="default-scheduler"' in profile_text,
)

# Roundtrip: parse the generated profile and check all keys come back correctly.
from profile_resolver import _read_env  # type: ignore[import]

with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as tf:
    tf.write(profile_text)
    tf_path = Path(tf.name)

parsed = _read_env(tf_path)
tf_path.unlink()

check("format_profile roundtrip: CLUSTER", parsed.get("CLUSTER") == "test-cluster")
check("format_profile roundtrip: NAMESPACE", parsed.get("NAMESPACE") == "test-ns")
check(
    "format_profile roundtrip: KUBE_CONTEXT",
    parsed.get("KUBE_CONTEXT") == "test-context",
)
check("format_profile roundtrip: GPU_PRODUCT", parsed.get("GPU_PRODUCT") == "NVIDIA-B200")
check("format_profile roundtrip: ARCH", parsed.get("ARCH") == "amd64")
check("format_profile roundtrip: HF_SECRET", parsed.get("HF_SECRET") == "hf-token")


# ---------------------------------------------------------------------------
# 4. Namespace-case selection (injected kubectl mock)
# ---------------------------------------------------------------------------

from profile_init import select_namespace  # type: ignore[import]
import io
import unittest.mock as mock


# Case A: namespaces exist, user picks one.
def _krun_namespaces_exist(args, timeout=30):
    """Mock: returns a list of namespaces for get-namespaces, yes for auth can-i, etc."""
    cmd = " ".join(args)
    if "get-contexts" in cmd and "--no-headers" in cmd:
        return 0, "example-eks\nlocal\n", ""
    if "get namespaces" in cmd or ("get" in cmd and "namespaces" in cmd and "--no-headers" in cmd):
        return 0, "llmb\nllmb-dev\nteam-perf\n", ""
    if "get pods" in cmd:
        # Simulate llmb namespace has GPU workloads.
        if "-n llmb" in cmd or "-n\nllmb" in cmd:
            payload = {"items": [{"spec": {"containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}]}}]}
            return 0, json.dumps(payload), ""
        return 0, '{"items": []}', ""
    return 0, "", ""


with mock.patch("builtins.input", side_effect=["1"]):
    result = select_namespace("test-ctx", krun=_krun_namespaces_exist)

check(
    "namespace Case A: selecting '1' returns first namespace (llmb)",
    result == "llmb",
    f"got {result!r}",
)

# Case A: user picks by index=2.
with mock.patch("builtins.input", side_effect=["2"]):
    result2 = select_namespace("test-ctx", krun=_krun_namespaces_exist)

check(
    "namespace Case A: selecting '2' returns second namespace (llmb-dev)",
    result2 == "llmb-dev",
    f"got {result2!r}",
)

# ── The Enter-key default must never be a SYSTEM namespace ────────────────────
# Rank user namespaces ahead of platform namespaces so accepting the default cannot target system services.
from profile_init import is_system_namespace, rank_namespaces  # noqa: E402

check(
    "is_system_namespace: infra namespaces are recognized",
    all(
        is_system_namespace(n)
        for n in (
            "argocd",
            "cert-manager",
            "default",
            "kube-system",
            "kube-public",
            "nvidia-gpu-operator",
        )
    ),
)
check(
    "is_system_namespace: a user/benchmark namespace is NOT system",
    not any(is_system_namespace(n) for n in ("llmb", "llmb-dev", "llmb-example-bench", "team-bench")),
)

_real = [
    "argocd",
    "cert-manager",
    "default",
    "kube-public",
    "llmb-example-bench",
    "team-b",
    "kube-system",
]
_ranked = rank_namespaces(_real, owner="llmb", gpu_namespaces={"team-b"})
check(
    "rank_namespaces: the user's OWN namespace ranks first (Enter-key default)",
    _ranked[0] == "llmb-example-bench",
    str(_ranked),
)
check(
    "rank_namespaces: a GPU-bearing namespace outranks system ones",
    _ranked.index("team-b") < _ranked.index("argocd"),
    str(_ranked),
)
check(
    "rank_namespaces: EVERY system namespace sorts after every non-system one",
    max(_ranked.index(n) for n in ("llmb-example-bench", "team-b"))
    < min(_ranked.index(n) for n in ("argocd", "cert-manager", "default", "kube-public", "kube-system")),
    str(_ranked),
)
check(
    "rank_namespaces: no namespace is lost or duplicated",
    sorted(_ranked) == sorted(_real),
)

# When only infrastructure is visible, "1" would still be a system namespace → steer to create-new.
_all_sys = ["argocd", "cert-manager", "kube-system"]
check(
    "rank_namespaces: an all-system cluster stays all-system (caller must default to create-new)",
    all(is_system_namespace(n) for n in rank_namespaces(_all_sys, owner="llmb")),
)

# ── The recommended default is CREATE `<simplified-username>-llmb-k8s` ────────
from profile_init import suggested_namespace  # noqa: E402

check(
    "suggested_namespace: <user>-llmb-k8s",
    suggested_namespace("llmb") == "llmb-llmb-k8s",
    suggested_namespace("llmb"),
)
check(
    "suggested_namespace: an email login is simplified to the local part",
    suggested_namespace("first.last@example.com") == "first-last-llmb-k8s",
    suggested_namespace("first.last@example.com"),
)
check(
    "suggested_namespace: no owner → bare llmb-k8s",
    suggested_namespace("") == "llmb-k8s",
)
check(
    "suggested_namespace: always a valid RFC-1123 label",
    all(
        __import__("re").fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", suggested_namespace(o))
        and len(suggested_namespace(o)) <= 63
        for o in ("llmb", "first.last@example.com", "", "a_b.c-d", "X" * 90)
    ),
)

# Pressing ENTER must CREATE the suggested namespace — never adopt namespaces[0] (the `argocd` bug).
_created: list = []


def _krun_enter_creates(args, timeout=30):
    if "namespaces" in args and "get" in args:
        return 0, "argocd\ncert-manager\nllmb-old\n", ""
    if args[:2] == ["--context", "test-ctx"] and "create" in args and "namespace" in args:
        _created.append(args[args.index("namespace") + 1])
        return 0, "", ""
    if "label" in args:
        return 0, "", ""
    return 0, '{"items": []}', ""


with mock.patch("builtins.input", side_effect=[""]):
    _r = select_namespace("test-ctx", krun=_krun_enter_creates, owner="llmb")
check(
    "namespace default: ENTER creates '<user>-llmb-k8s' (NOT argocd)",
    _r == "llmb-llmb-k8s" and _created == ["llmb-llmb-k8s"],
    f"got {_r!r} created={_created}",
)
check(
    "namespace default: ENTER never returns a system namespace",
    not is_system_namespace(_r or ""),
    f"got {_r!r}",
)

# An existing namespace is still selectable by number, in the ranked order.
with mock.patch("builtins.input", side_effect=["1"]):
    _r2 = select_namespace("test-ctx", krun=_krun_enter_creates, owner="llmb")
check(
    "namespace: picking '1' selects the user's OWN existing namespace (ranked first)",
    _r2 == "llmb-old",
    f"got {_r2!r}",
)

# Choosing a system namespace by number requires an explicit confirmation.
with mock.patch("builtins.input", side_effect=["2", "n", "1"]):
    _r3 = select_namespace("test-ctx", krun=_krun_enter_creates, owner="llmb")
check(
    "namespace: selecting a SYSTEM namespace warns and requires confirmation (declined → re-prompt)",
    _r3 == "llmb-old",
    f"got {_r3!r}",
)


# Case B: no namespaces found — user creates one.
def _krun_no_namespaces(args, timeout=30):
    cmd = " ".join(args)
    if "namespaces" in cmd and "--no-headers" in cmd:
        return 0, "", ""  # empty — no namespaces
    if "create" in cmd and "namespace" in cmd:
        return 0, "namespace/my-ns created", ""
    if "label" in cmd:
        return 0, "", ""
    return 0, "", ""


with mock.patch("builtins.input", side_effect=["my-ns", "y"]):
    result3 = select_namespace("test-ctx", krun=_krun_no_namespaces)

check(
    "namespace Case B: no namespaces → user creates 'my-ns'",
    result3 == "my-ns",
    f"got {result3!r}",
)


# Case B: user chooses 'n' to create from existing list.
def _krun_ns_with_create(args, timeout=30):
    cmd = " ".join(args)
    if "namespaces" in cmd and "--no-headers" in cmd:
        return 0, "existing-ns\n", ""
    if "get pods" in cmd:
        return 0, '{"items": []}', ""
    if "create" in cmd and "namespace" in cmd:
        return 0, "namespace/brand-new created", ""
    if "label" in cmd:
        return 0, "", ""
    return 0, "", ""


with mock.patch("builtins.input", side_effect=["n", "brand-new", "y"]):
    result4 = select_namespace("test-ctx", krun=_krun_ns_with_create)

check(
    "namespace Case B (from list): picking 'n' then creating 'brand-new'",
    result4 == "brand-new",
    f"got {result4!r}",
)


# ---------------------------------------------------------------------------
# 5. Model-on-PVC detection (injected kubectl mock)
# ---------------------------------------------------------------------------

from install import probe_model_on_pvc  # type: ignore[import]

_probe_calls: list[list[str]] = []

# The probe no longer runs a private sequence of `test -f` / `test -d`. It runs the SHARED
# model_cache.cache_probe_script and classifies the report with model_cache.cache_completeness -- the same
# script and predicate preflight uses -- so these mocks return the REPORT the real busybox pod would print.
# Three answers about the same bytes (install's sentinel check, preflight's shard index, fleet's PVC label)
# is what let fleet call a claim "verified" while install offered to re-download it.
import model_cache as _mc  # type: ignore[import]


def _report(**kw):
    # The per-file SIZE block is part of every real report (cache_probe_script always emits SHARD_N /
    # SHARD_MIN / SHARD_MAX beside SHARD_BYTES). A fixture without it describes a report the probe cannot
    # produce — and specifically the one where SHARDS>0 with nothing sized, which used to disable all three
    # byte guards and fall through to COMPLETE and is now UNKNOWN.
    f = {
        "EXISTS": 1,
        "SENTINEL": 0,
        "CONFIG": 1,
        "INDEX": 1,
        "SHARDS": 113,
        "REQ_SHARDS": 113,
        "MISSING_SHARDS": 0,
        "INCOMPLETE_FILES": 0,
        "INDEX_TOTAL_BYTES": 1000,
        "SHARD_BYTES": 1100,
        "SHARD_MIN": 9,
        "SHARD_MAX": 11,
        "SHARD_N": 113,
        "REF": "",
        "REFSNAP": 0,
    }
    f.update(kw)
    return "\n".join(f"{k}={v}" for k, v in f.items())


def _krun_reporting(report):
    def _k(args, timeout=30):
        _probe_calls.append(list(args))
        cmd = " ".join(args)
        if "wait" in cmd:
            return 0, "condition met", ""
        if "exec" in cmd:
            return 0, report, ""
        return 0, "", ""

    return _k


_ARGS = dict(
    ns="test-ns",
    pvc="model-cache",
    model_repo="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    model_revision="abc123def456abc123",
)

# THE LIVE NEMOTRON CASE: all 113 shards resolve, and there is NO .llmb_download_done at all.
status = probe_model_on_pvc(**_ARGS, krun=_krun_reporting(_report()))
check(
    "probe_model_on_pvc: complete sharded model with NO sentinel → 'complete' (not a re-download)",
    status == _mc.STATE_COMPLETE,
    f"got {status!r}",
)

status_s = probe_model_on_pvc(**_ARGS, krun=_krun_reporting(_report(SENTINEL=1)))
check(
    "probe_model_on_pvc: sentinel present → 'complete'",
    status_s == _mc.STATE_COMPLETE,
    f"got {status_s!r}",
)

status2 = probe_model_on_pvc(**_ARGS, krun=_krun_reporting(_report(EXISTS=0)))
check(
    "probe_model_on_pvc: no snapshot dir → 'absent'",
    status2 == _mc.STATE_ABSENT,
    f"got {status2!r}",
)

status3 = probe_model_on_pvc(**_ARGS, krun=_krun_reporting(_report(MISSING_SHARDS=7)))
check(
    "probe_model_on_pvc: shards the index names are missing → 'incomplete'",
    status3 == _mc.STATE_INCOMPLETE,
    f"got {status3!r}",
)

# Unverified state: weights are present, but completeness is not proven.
status5 = probe_model_on_pvc(
    **_ARGS,
    krun=_krun_reporting(_report(CONFIG=0, INDEX=0, REQ_SHARDS=0, INDEX_TOTAL_BYTES="", SHARD_BYTES="")),
)
check(
    "probe_model_on_pvc: weights present but unprovable → 'present-unverified' (never 'absent')",
    status5 == _mc.STATE_PRESENT_UNVERIFIED,
    f"got {status5!r}",
)

# AN INDEX THAT EXISTS BUT ENUMERATES NOTHING is not a single-file model. This reading used to verdict
# COMPLETE with the proof string "no shard index (single-file model)" about a 113-shard model — and install
# then wrote the permanent completion sentinel on it.
status5b = probe_model_on_pvc(
    **_ARGS,
    krun=_krun_reporting(_report(REQ_SHARDS=0, INDEX_TOTAL_BYTES="", SHARD_BYTES="")),
)
check(
    "probe_model_on_pvc: index present but unenumerated → 'present-unverified', never 'complete'",
    status5b == _mc.STATE_PRESENT_UNVERIFIED,
    f"got {status5b!r}",
)

# THE SIZING PASS RETURNING NOTHING is 'we could not measure', not 'the shards are empty'. Both the
# empty and zero-valued forms must land on UNKNOWN.
for _tag, _sz in (
    ("empty", {"SHARD_BYTES": "", "SHARD_MIN": "", "SHARD_MAX": "", "SHARD_N": ""}),
    ("zeros", {"SHARD_BYTES": 0, "SHARD_MIN": 0, "SHARD_MAX": 0, "SHARD_N": 0}),
):
    _st_sz = probe_model_on_pvc(**_ARGS, krun=_krun_reporting(_report(**_sz)))
    check(
        f"probe_model_on_pvc: 113 shards but nothing sized ({_tag}) → 'unknown', never 'complete'",
        _st_sz == _mc.STATE_UNKNOWN,
        f"got {_st_sz!r}",
    )

# A pod that never became ready is UNKNOWN, and the caller is told WHICH failure it was.
_why: list = []


def _krun_unschedulable(args, timeout=30):
    cmd = " ".join(args)
    if "wait" in cmd:
        return 1, "", "timed out"
    if "events" in cmd:
        return 0, "0/11 nodes are available: FailedScheduling", ""
    if "jsonpath={.status.phase}" in cmd:
        return 0, "Pending", ""
    return 0, "", ""


status6 = probe_model_on_pvc(**_ARGS, krun=_krun_unschedulable, prof={}, why=_why)
check(
    "probe_model_on_pvc: pod never scheduled → 'unknown'",
    status6 == _mc.STATE_UNKNOWN,
    f"got {status6!r}",
)
check(
    "probe_model_on_pvc: the caller is told it was UNSCHEDULABLE, not just 'unknown'",
    any(c == "unschedulable" for c, _, _ in _why),
    str(_why),
)
check(
    "probe_model_on_pvc: ...and the remediation names MODEL_CACHE_NODE_SELECTOR",
    any("MODEL_CACHE_NODE_SELECTOR" in h for _, _, h in _why),
    str(_why),
)

# Placement: the probe pod must carry the profile's cache selector AND tolerate taints.
_probe_calls.clear()
probe_model_on_pvc(
    **_ARGS,
    krun=_krun_reporting(_report()),
    prof={"MODEL_CACHE_NODE_SELECTOR": 'nvidia.com/gpu.present: "true"'},
)
_ov = [a for call in _probe_calls for a in call if a.startswith("--overrides=")]
check(
    "probe_model_on_pvc: pod carries the profile's cache nodeSelector",
    bool(_ov) and '"nodeSelector": {"nvidia.com/gpu.present": "true"}' in _ov[0],
    str(_ov[:1])[:200],
)
check(
    "probe_model_on_pvc: pod tolerates SCHEDULING taints (the mountable nodes may be tainted)",
    bool(_ov) and '"tolerations": [{"operator": "Exists", "effect": "NoSchedule"}' in _ov[0],
    str(_ov[:1])[:200],
)
check(
    "probe_model_on_pvc: ...but NOT NoExecute — a NotReady node must still evict the probe "
    "(a blanket Exists also suppresses the default not-ready/unreachable evictions, so the pod hangs "
    "to its wait budget and reports UNKNOWN)",
    bool(_ov) and '{"operator": "Exists"}' not in _ov[0] and '"NoExecute"' not in _ov[0],
    str(_ov[:1])[:200],
)

# Unknown revision → should not probe.
status4 = probe_model_on_pvc(
    ns="test-ns",
    pvc="model-cache",
    model_repo="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    model_revision="",
    krun=_krun_reporting(_report()),
)
check(
    "probe_model_on_pvc: empty model_revision → 'unknown' (no probe)",
    status4 == _mc.STATE_UNKNOWN,
    f"got {status4!r}",
)


# ---------------------------------------------------------------------------
# 6. Download Job template rendering
# ---------------------------------------------------------------------------

from install import render_download_job  # type: ignore[import]

fake_prof = {
    "NAMESPACE": "test-ns",
    "MODEL_CACHE_PVC": "model-cache",
    "MODEL_CACHE_SUBPATH": ".",
    "HF_SECRET": "hf-token",
    "IMAGE_PULL_SECRET": "ngc-registry",
}

fake_model = {
    "model_repo": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "model_revision": "abc123def456abc123def456abc123def456abc123def456abc123def4561234",
    "model_name": "nemotron-ultra-3",
}

tmpl_path = ROOT / "serving" / "download" / "templates" / "model-download.yaml.j2"
if tmpl_path.exists():
    rendered = render_download_job(fake_model, fake_prof, tmpl_path)
    check(
        "render_download_job: produces non-empty output",
        len(rendered) > 100,
        f"len={len(rendered)}",
    )
    check(
        "render_download_job: contains Job kind",
        "kind: Job" in rendered,
        rendered[:200],
    )
    check(
        "render_download_job: bakes model_repo into manifest",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4" in rendered,
        rendered[:300],
    )
    check(
        "render_download_job: bakes model_revision into manifest",
        "abc123def456abc123" in rendered,
        rendered[:300],
    )
    check(
        "render_download_job: references HF_SECRET via ${HF_SECRET}",
        "${HF_SECRET}" in rendered,
        "(${HF_SECRET} not found in rendered manifest)",
    )
    check(
        "render_download_job: references MODEL_CACHE_PVC via ${MODEL_CACHE_PVC}",
        "${MODEL_CACHE_PVC}" in rendered,
        "(${MODEL_CACHE_PVC} not found in rendered manifest)",
    )
    check(
        "render_download_job: references ${NAMESPACE}",
        "${NAMESPACE}" in rendered,
        "(${NAMESPACE} not found)",
    )
    check(
        "render_download_job: activeDeadlineSeconds=7200 (2h)",
        "7200" in rendered,
        rendered[:500],
    )
    # OOM FIX: the download Job memory LIMIT must be ≥ the 32Gi floor (req 4Gi/limit 8Gi OOMKilled a ~400GB
    # model download; backoffLimit=0 → terminal). Parse the rendered manifest and assert the real limit.
    import install  # type: ignore[import]
    import yaml as _yaml

    # The manifest is a MULTI-doc stream: ServiceAccount + Role + RoleBinding (the least-privilege RBAC that
    # lets the Job stamp its PVC with model-name/revision/download-complete on success) + the Job itself.
    _docs = [d for d in _yaml.safe_load_all(rendered) if d]
    _doc = next(d for d in _docs if d.get("kind") == "Job")
    check(
        "render_download_job: ships PVC-stamp RBAC (ServiceAccount + Role + RoleBinding) alongside the Job",
        {d.get("kind") for d in _docs} == {"ServiceAccount", "Role", "RoleBinding", "Job"},
        str([d.get("kind") for d in _docs]),
    )
    _role = next(d for d in _docs if d.get("kind") == "Role")
    check(
        "render_download_job: the stamp Role is LEAST-PRIVILEGE (patch/get on PVCs only)",
        _role["rules"]
        == [
            {
                "apiGroups": [""],
                "resources": ["persistentvolumeclaims"],
                "verbs": ["get", "patch"],
            }
        ],
        str(_role["rules"]),
    )
    check(
        "render_download_job: the Job runs under that ServiceAccount",
        _doc["spec"]["template"]["spec"].get("serviceAccountName", "").startswith("llmb-download-"),
        str(_doc["spec"]["template"]["spec"].get("serviceAccountName")),
    )
    # ORDERING GUARANTEE: the stamp must come AFTER the download + sentinel, so a label can never claim a
    # completion that did not happen (the green-signal-that-isn't).
    _script = "\n".join(_doc["spec"]["template"]["spec"]["containers"][0]["command"])
    check(
        "render_download_job: the PVC stamp runs strictly AFTER the download + sentinel write",
        _script.index("sentinel written") < _script.index("STAMP THE PVC") < _script.index("=== download complete ==="),
        "stamp is not last",
    )
    _res = _doc["spec"]["template"]["spec"]["containers"][0]["resources"]
    _lim_gib = install._parse_quantity_gib(_res["limits"]["memory"])
    _req_gib = install._parse_quantity_gib(_res["requests"]["memory"])
    check(
        "render_download_job: memory limit ≥ 32Gi floor (no OOM on large models)",
        _lim_gib is not None and _lim_gib >= install._DOWNLOAD_MEM_LIMIT_GIB,
        f"limit={_res['limits']['memory']} ({_lim_gib} GiB)",
    )
    check(
        "render_download_job: memory request ≥ 8Gi",
        _req_gib is not None and _req_gib >= install._DOWNLOAD_MEM_REQUEST_GIB,
        f"request={_res['requests']['memory']} ({_req_gib} GiB)",
    )
    check(
        "download_mem_limit_gib(): flat 32Gi floor regardless of model size",
        install.download_mem_limit_gib(400) == 32 and install.download_mem_limit_gib(None) == 32,
        str(install.download_mem_limit_gib(400)),
    )
else:
    print(f"  SKIP  render_download_job tests: template not found at {tmpl_path}")


# ---------------------------------------------------------------------------
# 7. Profile env path resolution
# ---------------------------------------------------------------------------

from profile_resolver import profile_env_path, list_profiles  # type: ignore[import]

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)
    # Create some fake profiles.
    (td_path / "my-cluster.env").write_text('CLUSTER="my-cluster"\n')
    (td_path / "another.env").write_text('CLUSTER="another"\n')
    (td_path / "_template.env.example").write_text('CLUSTER="template"\n')
    (td_path / "something.env.example").write_text('CLUSTER="example"\n')

    profiles = list_profiles(td_path)
    check(
        "list_profiles: finds real .env files (excludes .example and _template)",
        set(profiles) == {"my-cluster", "another"},
        f"got {profiles}",
    )

    path = profile_env_path("my-cluster", td_path)
    check(
        "profile_env_path: returns correct path",
        path == td_path / "my-cluster.env",
        f"got {path}",
    )


# ---------------------------------------------------------------------------
# 8. _envsubst_profile (pure)
# ---------------------------------------------------------------------------

from install import _envsubst_profile  # type: ignore[import]

manifest = "namespace: ${NAMESPACE}\npvc: ${MODEL_CACHE_PVC}\nother: ${UNKNOWN_VAR}\n"
prof = {"NAMESPACE": "prod-ns", "MODEL_CACHE_PVC": "model-cache"}
substed = _envsubst_profile(manifest, prof)

check(
    "_envsubst_profile: substitutes known vars",
    "namespace: prod-ns" in substed and "pvc: model-cache" in substed,
    substed,
)
check("_envsubst_profile: leaves unknown vars as-is", "${UNKNOWN_VAR}" in substed, substed)


# ---------------------------------------------------------------------------
# 9. make_krun threads --context (audit #1 — install must target the profile's cluster)
# ---------------------------------------------------------------------------

from install import make_krun  # type: ignore[import]

_krun_calls: list[list[str]] = []


def _fake_run(argv, **kw):
    _krun_calls.append(argv)

    class _R:
        returncode, stdout, stderr = 0, "", ""

    return _R()


with mock.patch("install.subprocess.run", side_effect=_fake_run):
    make_krun("ctx-prod")(["get", "ns"])
    make_krun("")(["get", "ns"])
check(
    "make_krun: pins --context when the profile sets KUBE_CONTEXT",
    _krun_calls[0][:3] == ["kubectl", "--context", "ctx-prod"],
    str(_krun_calls[0][:4]),
)
check(
    "make_krun: no --context when unset (current-context, backward-compatible)",
    "--context" not in _krun_calls[1],
    str(_krun_calls[1][:3]),
)


# ---------------------------------------------------------------------------
# 10. create_secret_interactive — NGC secret uses the literal $oauthtoken (audit #2/#3)
# ---------------------------------------------------------------------------

from install import create_secret_interactive  # type: ignore[import]

_secret_calls: list[list[str]] = []


def _cap_krun(args, timeout=30):
    _secret_calls.append(args)
    return (0, "", "")


with mock.patch("getpass.getpass", return_value="fake-ngc-key"):
    create_secret_interactive("ns", "ngc-registry", "ngc-key", "NGC API key", krun=_cap_krun)
_ngc = _secret_calls[-1]
check(
    "create_secret: NGC docker-username is the literal $oauthtoken (NGC requires it)",
    "--docker-username=$oauthtoken" in _ngc,
    str(_ngc),
)
check(
    "create_secret: no backslash-escaped \\$oauthtoken leaks to kubectl",
    not any("\\$oauthtoken" in a for a in _ngc),
    str(_ngc),
)
with mock.patch("getpass.getpass", return_value="hf_xxx"):
    create_secret_interactive("ns", "hf-token", "hf-token", "HF token", krun=_cap_krun)
check(
    "create_secret: hf-token path uses --from-literal=token=<value>",
    "--from-literal=token=hf_xxx" in _secret_calls[-1],
    str(_secret_calls[-1]),
)


# ---------------------------------------------------------------------------
# 11. download manifest survives render → envsubst with no profile ${VARS} left (audit #7)
# ---------------------------------------------------------------------------

if tmpl_path.exists():
    _post = _envsubst_profile(render_download_job(fake_model, fake_prof, tmpl_path), fake_prof)
    check(
        "download manifest: post-envsubst leaves no unresolved profile ${VARS}",
        not any(
            v in _post
            for v in (
                "${NAMESPACE}",
                "${MODEL_CACHE_PVC}",
                "${HF_SECRET}",
                "${IMAGE_PULL_SECRET}",
            )
        ),
        _post[:300],
    )
    check(
        "download manifest: post-envsubst still a Job with the model baked in",
        "kind: Job" in _post and "test-ns" in _post,
        _post[:200],
    )


# ---------------------------------------------------------------------------
# 12. profile_init namespace length cap (audit #9)
# ---------------------------------------------------------------------------

# Case B: no namespaces. First a >63-char name (rejected → reprompt), then valid.
_too_long = "a" * 64
with mock.patch("builtins.input", side_effect=[_too_long, "good-ns", "y"]):
    ns_result = select_namespace("test-ctx", krun=_krun_no_namespaces)
check(
    "namespace #9: name >63 chars is rejected, reprompt accepts a valid name",
    ns_result == "good-ns",
    f"got {ns_result!r}",
)

# Reject an invalid-char / bad-shape name (uppercase, trailing hyphen), then accept.
with mock.patch("builtins.input", side_effect=["Bad_NS-", "ok-ns", "y"]):
    ns_result2 = select_namespace("test-ctx", krun=_krun_no_namespaces)
check(
    "namespace #9: invalid-shape name is rejected, reprompt accepts a valid name",
    ns_result2 == "ok-ns",
    f"got {ns_result2!r}",
)


# ---------------------------------------------------------------------------
# 13. profile write mode 0o600 + dry-run writes nothing (audits #8, #10)
# ---------------------------------------------------------------------------

from profile_init import maybe_write_profile, write_profile  # type: ignore[import]

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)
    # #8 — a real write is chmod'd to owner-only 0o600.
    p_write = td_path / "real.env"
    wrote = maybe_write_profile(p_write, profile_text, dry_run=False)
    mode = p_write.stat().st_mode & 0o777
    check(
        "profile #8: maybe_write_profile(dry_run=False) writes the file",
        wrote and p_write.exists(),
    )
    check(
        "profile #8: written .env is mode 0o600 (owner-only, secrets not world-readable)",
        mode == 0o600,
        f"mode={oct(mode)}",
    )

    # #10 — dry-run writes nothing and reports it did not write.
    p_dry = td_path / "dryrun.env"
    wrote_dry = maybe_write_profile(p_dry, profile_text, dry_run=True)
    check(
        "profile #10: maybe_write_profile(dry_run=True) writes nothing",
        (not wrote_dry) and (not p_dry.exists()),
        f"wrote={wrote_dry}, exists={p_dry.exists()}",
    )


# ---------------------------------------------------------------------------
# 14. download-Job label sanitization (audit #11)
# ---------------------------------------------------------------------------

if tmpl_path.exists():
    import re as _re

    # A pathological HF repo name: uppercase, slashes/dots, invalid chars, way over 63.
    _patho = "Some.Very/Long_Repo@Name!!!With$$Weird##Chars////" + "X" * 80
    _patho_model = {
        "model_repo": "org/" + _patho,
        "model_revision": "abc123def456abc123def456abc123def456abc123def456abc123def4561234",
        "model_name": _patho,
    }
    _rendered = render_download_job(_patho_model, fake_prof, tmpl_path)

    # k8s label-value shape: ≤63 chars, [a-z0-9] at the ends, [-a-z0-9_.] inside.
    _label_re = _re.compile(r"^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$")
    _found = _re.findall(r'llmb\.nvidia\.com/model-name:\s*"([^"]*)"', _rendered)

    check(
        "download label #11: both model-name label occurrences are present",
        len(_found) == 2,
        f"found {_found}",
    )
    check(
        "download label #11: sanitized label is <= 63 chars",
        all(len(v) <= 63 for v in _found),
        f"lengths={[len(v) for v in _found]}: {_found}",
    )
    check(
        "download label #11: sanitized label is a valid k8s label value",
        all(bool(_label_re.match(v)) for v in _found),
        f"values={_found}",
    )
    check(
        "download label #11: label is lowercased with invalid chars replaced",
        all(v == v.lower() and "/" not in v and "@" not in v and "!" not in v for v in _found),
        f"values={_found}",
    )


# ---------------------------------------------------------------------------
# 12. catalog_models WARNS on a revision conflict (audit #6 — test was vacuous: shared revision)
# ---------------------------------------------------------------------------

import io
import contextlib

with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    for sub, rev in [
        ("recipes/a/cell1", "aaaa1111aaaa"),
        ("recipes/a/cell2", "bbbb2222bbbb"),
    ]:
        d = tdp / sub
        d.mkdir(parents=True)
        (d / "recipe.yaml").write_text(
            f"envelope:\n  name: n\nserving:\n  model_repo: org/samerepo\n  model_revision: {rev}\n"
        )
    _cat = [
        {"model": "m", "_path": "recipes/a/cell1"},
        {"model": "m", "_path": "recipes/a/cell2"},
    ]
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        catalog_models(_cat, tdp)
    check(
        "catalog_models: WARNS when two cells pin different revisions for one repo",
        "revision conflict" in _buf.getvalue().lower(),
        _buf.getvalue()[:160] or "(no output)",
    )


# ---------------------------------------------------------------------------
# 13. download Job metadata.name is a valid RFC-1123 name for a pathological model (audit #11 name)
# ---------------------------------------------------------------------------

if tmpl_path.exists():
    import re as _re
    from install import _model_job_slug  # type: ignore[import]

    _weird = {
        "model_repo": "Org/Weird_Model.Name@v2!!",
        "model_revision": "f" * 40,
        "model_name": "Org/Weird_Model.Name@v2!!",
    }
    _r = render_download_job(_weird, fake_prof, tmpl_path)
    # Scope to the JOB doc: the stream also carries the stamp RBAC (ServiceAccount/Role/RoleBinding), whose
    # names deliberately omit the revision suffix, so a bare `^  name:` regex would match those first.
    import yaml as _y

    _jobdoc = next(d for d in _y.safe_load_all(_r) if d and d.get("kind") == "Job")
    _nm = _jobdoc["metadata"]["name"]
    check(
        "download Job metadata.name is valid RFC-1123 (≤63, lowercase, no invalid chars)",
        0 < len(_nm) <= 63 and _re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", _nm) is not None,
        f"name={_nm!r} len={len(_nm)}",
    )
    _expected = f"llmb-download-{_model_job_slug(_weird['model_name'])}-{_weird['model_revision'][:12]}"
    check(
        "download Job name: template metadata.name == install.py job_name (single source, no drift)",
        _nm == _expected,
        f"{_nm} vs {_expected}",
    )


# ---------------------------------------------------------------------------
# 16. probe_gpu_nodes carries the whole-node size facts (gpus + raw allocatable cpu/memory)
#     ONE node probe feeds GPU_PER_NODE *and* CPU_PER_NODE / MEM_PER_NODE / WHOLE_NODE_* — no extra kubectl.
# ---------------------------------------------------------------------------

from profile_init import probe_gpu_nodes  # type: ignore[import]

_nodes_payload = json.dumps(
    {
        "items": [
            {
                "metadata": {
                    "name": "gpu-1",
                    "labels": {
                        "nvidia.com/gpu.product": "NVIDIA-GB300",
                        "kubernetes.io/arch": "arm64",
                    },
                },
                "status": {
                    "allocatable": {
                        "nvidia.com/gpu": "4",
                        "cpu": "139580m",
                        "memory": "948007936Ki",
                    }
                },
            },
            {
                "metadata": {
                    "name": "cpu-only",
                    "labels": {"kubernetes.io/arch": "arm64"},
                },
                "status": {"allocatable": {"cpu": "8", "memory": "32Gi"}},
            },
        ]
    }
)
_probed = probe_gpu_nodes("ctx", krun=lambda *a, **k: (0, _nodes_payload, ""))
check(
    "probe_gpu_nodes: only GPU-labelled nodes are returned",
    [n["name"] for n in _probed] == ["gpu-1"],
    _probed,
)
check(
    "probe_gpu_nodes: allocatable GPU count (GPU_PER_NODE source)",
    _probed and _probed[0]["gpus"] == 4,
    _probed,
)
check(
    "probe_gpu_nodes: raw allocatable cpu rides along (CPU_PER_NODE source)",
    _probed and _probed[0]["cpu_alloc"] == "139580m",
    _probed,
)
check(
    "probe_gpu_nodes: raw allocatable memory rides along (MEM_PER_NODE source)",
    _probed and _probed[0]["mem_alloc"] == "948007936Ki",
    _probed,
)

# Degradation: no RBAC to list nodes (non-zero rc) → empty list, never a crash; and a node with no
# allocatable block yields empty facts so the wizard simply omits CPU_PER_NODE / MEM_PER_NODE.
check(
    "probe_gpu_nodes: no RBAC to list nodes → [] (graceful, not a crash)",
    probe_gpu_nodes("ctx", krun=lambda *a, **k: (1, "", "forbidden")) == [],
)
_no_alloc = json.dumps(
    {
        "items": [
            {
                "metadata": {
                    "name": "n",
                    "labels": {
                        "nvidia.com/gpu.product": "NVIDIA-B200",
                        "kubernetes.io/arch": "amd64",
                    },
                }
            }
        ]
    }
)
_p2 = probe_gpu_nodes("ctx", krun=lambda *a, **k: (0, _no_alloc, ""))
check(
    "probe_gpu_nodes: node without .status.allocatable → empty size facts (gpus=0, cpu/mem '')",
    _p2 and _p2[0]["gpus"] == 0 and _p2[0]["cpu_alloc"] == "" and _p2[0]["mem_alloc"] == "",
    _p2,
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
if fails:
    print(f"selftest_onboarding: {len(fails)} FAILED: {fails}")
    sys.exit(1)
else:
    total = sum(1 for line in open(__file__).read().splitlines() if line.strip().startswith("check("))
    print(f"selftest_onboarding: all {total} checks PASSED ✓")
    sys.exit(0)

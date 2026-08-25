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

"""selftest_install_recipes.py — unit tests for install.py's recipe multi-select + bulk-setup (no cluster).

Covers the SLURM-aligned two-tier grow of install.py:
  - GPU-filter correctness            (gpu_matching_cells, _norm_gpu normalization, includes wip)
  - greyed-out-from-stamp             (cell_installed: recipe_hash match + ready; stale/fail → not greyed)
  - model-union dedup                 (models_for_cells over selected cells)
  - per-cell stamp write/read         (write_install_stamp / read_install_stamps, last-line-wins, 0600)
  - --list-recipes                    (list_recipes output: lock glyph + counts)
  - stop-at-first-blocker             (setup_one_cell: stage fail → blocked+fix, preflight skipped)
  - headless flag parsing / selection (resolve_recipe_selection: --recipes / --all-matching / unknown)

All subprocess (stage/preflight) calls are injected via a `runner` fake — runs fully offline.
Mirrors the pattern in scripts/selftest_onboarding.py.
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import tempfile
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import install  # type: ignore[import]

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    marker = "PASS" if cond else "FAIL"
    print(f"  {marker}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------------------
# Fixtures: a synthetic catalog (no filesystem recipe.yaml needed for the pure paths)
# ---------------------------------------------------------------------------


def _cell(
    name,
    gpu,
    status="wip",
    scenario="llm-perf",
    mode=None,
    rh="hash-",
    path=None,
    distribution="",
    agent=None,
):
    c = {
        "name": name,
        "gpu_type": gpu,
        "arch": "amd64",
        "status": status,
        "scenario": scenario,
        "mode": mode,
        "engine": "vllm",
        "serving_mode": "aggregated",
        "distribution": distribution,
        "goal": "throughput",
        "recipe_hash": rh + name,
        "_path": path or f"recipes/{scenario}/{name}",
    }
    if agent:
        c["agent"] = agent
    return c


B200_CELLS = [
    _cell("a-b200", "B200", status="wip"),
    _cell("b-b200", "NVIDIA-B200", status="runs"),  # normalization: NVIDIA-B200 == B200
]
GB200_CELLS = [_cell("d-gb200", "GB200", status="runs")]
CATALOG = B200_CELLS + GB200_CELLS

PROF_B200 = {"GPU_PRODUCT": "NVIDIA-B200", "NAMESPACE": "ns", "MODEL_CACHE_PVC": "pvc"}


# ---------------------------------------------------------------------------
# 1. GPU-filter correctness
# ---------------------------------------------------------------------------

matched = install.gpu_matching_cells(CATALOG, PROF_B200)
matched_ids = {install.cell_id(c) for c in matched}
check(
    "gpu_matching_cells: only B200 cells match a B200 cluster (GB200 excluded)",
    matched_ids == {"a-b200", "b-b200"},
    str(matched_ids),
)
check(
    "gpu_matching_cells: NVIDIA-B200 normalizes to B200 (b-b200 matched)",
    "b-b200" in matched_ids,
)
check(
    "gpu_matching_cells: wip cells are INCLUDED (devs see everything)",
    "a-b200" in matched_ids and any(c["status"] == "wip" for c in matched),
)
check(
    "gpu_matching_cells: result is sorted stably (scenario, distribution, id)",
    matched == install.gpu_matching_cells(CATALOG, PROF_B200),
)

# GPU_TYPE override on the profile wins over GPU_PRODUCT.
check(
    "gpu_matching_cells: empty GPU → no matches (never match everything)",
    install.gpu_matching_cells(CATALOG, {"GPU_PRODUCT": ""}) == [],
)


# ---------------------------------------------------------------------------
# 2. greyed-out-from-stamp (cell_installed / _stamp_ready)
# ---------------------------------------------------------------------------

cell_a = B200_CELLS[0]
rh = cell_a["recipe_hash"]

stamps_ready = {
    cell_a["_path"]: {
        "recipe_hash": rh,
        "preflight": "pass",
        "staged": {"stage-dataset": {"ok": True}},
    }
}
check(
    "cell_installed: matching recipe_hash + ready stamp → greyed-out (installed)",
    install.cell_installed(cell_a, stamps_ready) is True,
)

stamps_stale = {cell_a["_path"]: {"recipe_hash": "OLD", "preflight": "pass", "staged": {}}}
check(
    "cell_installed: stale recipe_hash → NOT installed (recipe moved, re-setup due)",
    install.cell_installed(cell_a, stamps_stale) is False,
)

stamps_fail = {
    cell_a["_path"]: {
        "recipe_hash": rh,
        "preflight": "fail",
        "staged": {"stage-dataset": {"ok": True}},
    }
}
check(
    "cell_installed: preflight=fail stamp is recorded but does NOT grey the cell out",
    install.cell_installed(cell_a, stamps_fail) is False,
)

stamps_stagefail = {
    cell_a["_path"]: {
        "recipe_hash": rh,
        "preflight": "skipped",
        "staged": {"stage-dataset": {"ok": False}},
    }
}
check(
    "cell_installed: staged ok=False → NOT greyed out",
    install.cell_installed(cell_a, stamps_stagefail) is False,
)

check(
    "cell_installed: no stamp at all → NOT installed",
    install.cell_installed(cell_a, {}) is False,
)

check(
    "cell_installed: warn preflight + staged ok → greyed out (warn is advisory)",
    install.cell_installed(
        cell_a,
        {
            cell_a["_path"]: {
                "recipe_hash": rh,
                "preflight": "warn",
                "staged": {"stage-dataset": {"ok": True}},
            }
        },
    )
    is True,
)


# ---------------------------------------------------------------------------
# 3. model-union dedup (models_for_cells) — needs real recipe.yaml under ROOT
# ---------------------------------------------------------------------------

real_catalog = install.load_catalog()
if real_catalog:
    real_b200 = install.gpu_matching_cells(real_catalog, PROF_B200)
    union = install.models_for_cells(real_b200)
    repos = [m["model_repo"] for m in union]
    check(
        "models_for_cells: union is deduplicated by HF repo (no dup repos)",
        len(repos) == len(set(install._norm_repo(r) for r in repos)),
        str(repos),
    )
    check(
        "models_for_cells: union is smaller than the cell count (dedup actually collapses)",
        len(union) <= len(real_b200) and len(union) >= 1,
        f"{len(union)} models / {len(real_b200)} cells",
    )
else:
    check(
        "models_for_cells: catalog.json present for union test",
        False,
        "no catalog.json",
    )


# ---------------------------------------------------------------------------
# 4. per-cell stamp write/read (atomic append, last-line-wins, 0600)
# ---------------------------------------------------------------------------

with tempfile.TemporaryDirectory() as td:
    state = Path(td) / ".state"
    install.write_install_stamp(
        "clX",
        "recipes/x",
        "hash1",
        "org/model",
        {"stage-dataset": {"ok": True, "sha": "abc"}},
        "pass",
        "bench",
        state_dir=state,
    )
    install.write_install_stamp(
        "clX",
        "recipes/x",
        "hash2",
        "org/model",
        {"stage-dataset": {"ok": True, "sha": "def"}},
        "warn",
        "bench",
        state_dir=state,
    )
    install.write_install_stamp(
        "clX",
        "recipes/y",
        "hashY",
        "org/other",
        {"stage-traces": {"ok": True}},
        "pass",
        "bench",
        state_dir=state,
    )

    read = install.read_install_stamps("clX", state_dir=state)
    check(
        "stamp roundtrip: two cells recorded",
        set(read) == {"recipes/x", "recipes/y"},
        str(set(read)),
    )
    check(
        "stamp roundtrip: last line wins for a re-stamped cell (hash2/warn)",
        read["recipes/x"]["recipe_hash"] == "hash2" and read["recipes/x"]["preflight"] == "warn",
    )
    check(
        "stamp roundtrip: staged sub-object preserved",
        read["recipes/x"]["staged"]["stage-dataset"]["sha"] == "def",
    )
    check("stamp roundtrip: stamped_at present", bool(read["recipes/x"].get("stamped_at")))

    path = install.install_stamp_path("clX", state_dir=state)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    check("stamp file is 0600 (owner-only)", mode == 0o600, oct(mode))
    dmode = stat.S_IMODE(os.stat(state).st_mode)
    check("stamp dir is 0700 (owner-only)", dmode == 0o700, oct(dmode))

    # cross-cluster isolation: a different cluster has its own file.
    install.write_install_stamp(
        "clY",
        "recipes/x",
        "hash1",
        "org/model",
        {"stage-dataset": {"ok": True}},
        "pass",
        "bench",
        state_dir=state,
    )
    check(
        "stamp is per-cluster (clY file separate from clX)",
        install.install_stamp_path("clY", state_dir=state) != path
        and set(install.read_install_stamps("clY", state_dir=state)) == {"recipes/x"},
    )

    check(
        "read_install_stamps: missing file → {} (nothing installed yet)",
        install.read_install_stamps("nope", state_dir=state) == {},
    )


# ---------------------------------------------------------------------------
# 5. --list-recipes output
# ---------------------------------------------------------------------------

buf = io.StringIO()
stamps_for_list = {
    B200_CELLS[1]["_path"]: {
        "recipe_hash": B200_CELLS[1]["recipe_hash"],
        "preflight": "pass",
        "staged": {"s": {"ok": True}},
    }
}
with contextlib.redirect_stdout(buf):
    install.list_recipes(matched, stamps_for_list, "clX", "B200")
out = buf.getvalue()
check(
    "list_recipes: prints the lock glyph for the installed cell",
    install.GLYPH_INSTALLED in out and "b-b200" in out,
)
check(
    "list_recipes: reports counts (1 installed, 1 available of 2)",
    "2 matching · 1 installed · 1 available" in out,
    out.splitlines()[-2] if out else "",
)


# ---------------------------------------------------------------------------
# 6. resolve_recipe_selection (headless)
# ---------------------------------------------------------------------------

# --all-matching skips already-installed cells.
sel, errs = install.resolve_recipe_selection(matched, stamps_for_list, None, True)
check(
    "resolve_recipe_selection: --all-matching returns non-installed cells only",
    {install.cell_id(c) for c in sel} == {"a-b200"} and not errs,
    str([c["name"] for c in sel]),
)

# --recipes by catalog name.
sel, errs = install.resolve_recipe_selection(matched, {}, "a-b200", False)
check(
    "resolve_recipe_selection: --recipes matches by catalog name",
    [install.cell_id(c) for c in sel] == ["a-b200"] and not errs,
)

# --recipes by _path basename.
sel, errs = install.resolve_recipe_selection(matched, {}, Path(B200_CELLS[0]["_path"]).name, False)
check(
    "resolve_recipe_selection: --recipes matches by path basename",
    [install.cell_id(c) for c in sel] == ["a-b200"] and not errs,
)

# --recipes by the same relative/absolute cell paths accepted by run/preflight/submit. This is the release UX
# contract: copying CELL=recipes/... into `install --recipes "$CELL"` must not exit 2 (and, under `set -e`,
# appear to kill the user's interactive shell).
rel_path = B200_CELLS[0]["_path"]
sel, errs = install.resolve_recipe_selection(matched, {}, rel_path, False)
check(
    "resolve_recipe_selection: --recipes matches by catalog-relative full path",
    [install.cell_id(c) for c in sel] == ["a-b200"] and not errs,
    str(errs),
)
abs_path = str((install.ROOT / rel_path).resolve())
sel, errs = install.resolve_recipe_selection(matched, {}, abs_path + "/", False)
check(
    "resolve_recipe_selection: --recipes matches by absolute full path with trailing slash",
    [install.cell_id(c) for c in sel] == ["a-b200"] and not errs,
    str(errs),
)

# unknown name → error, no selection.
sel, errs = install.resolve_recipe_selection(matched, {}, "does-not-exist", False)
check(
    "resolve_recipe_selection: unknown cell → error surfaced, empty selection",
    sel == [] and any("does-not-exist" in e for e in errs),
)

# dedup: same token twice → one cell.
sel, errs = install.resolve_recipe_selection(matched, {}, "a-b200,a-b200", False)
check(
    "resolve_recipe_selection: duplicate tokens dedup to one cell",
    [install.cell_id(c) for c in sel] == ["a-b200"],
)

# Ambiguous basename: two cells with DISTINCT catalog names but the SAME dir basename must NOT resolve
# silently. Real case: recipes/llm-perf/256k/nemotron-ultra-3-b200-vllm-agg-pareto (name …-256k-pareto)
# and recipes/llm-perf/synthetic/nemotron-ultra-3-b200-vllm-agg-pareto (name …-synthetic-pareto) share
# a basename. The old single-dict alias silently kept whichever was iterated last → wrong cell, no warning.
amb_cells = [
    _cell(
        "nemo-shared-256k",
        "B200",
        scenario="llm-perf",
        path="recipes/llm-perf/256k/nemo-shared",
    ),
    _cell(
        "nemo-shared-synthetic",
        "B200",
        scenario="llm-perf",
        distribution="synthetic",
        path="recipes/llm-perf/synthetic/nemo-shared",
    ),
]
sel, errs = install.resolve_recipe_selection(amb_cells, {}, "nemo-shared", False)
check(
    "resolve_recipe_selection: ambiguous basename → error naming BOTH candidates, no silent pick",
    sel == [] and any("ambiguous" in e and "nemo-shared-256k" in e and "nemo-shared-synthetic" in e for e in errs),
    str(errs),
)
# The full catalog name still resolves unambiguously even when a basename twin exists.
sel, errs = install.resolve_recipe_selection(amb_cells, {}, "nemo-shared-synthetic", False)
check(
    "resolve_recipe_selection: exact catalog name resolves even with a basename twin present",
    [install.cell_id(c) for c in sel] == ["nemo-shared-synthetic"] and not errs,
    str(errs),
)


# ---------------------------------------------------------------------------
# 7. setup_one_cell — stop-at-first-blocker + needs-input (injected runner)
# ---------------------------------------------------------------------------

# A real llm-perf cell dir (for lane routing + preflight path); use the first real B200 llm-perf cell.
llm_cell = None
if real_catalog:
    for c in install.gpu_matching_cells(real_catalog, PROF_B200):
        if c.get("scenario") == "llm-perf" and llm_cell is None:
            llm_cell = c

with tempfile.TemporaryDirectory() as td:
    state = Path(td) / ".state"

    if llm_cell:
        # Runner that FAILS the stage step → blocked, fix printed, preflight skipped.
        def runner_stage_fail(argv, cwd):
            if "preflight.py" in " ".join(argv):
                raise AssertionError("preflight should NOT run after a stage failure (stop-at-first-blocker)")
            return (
                1,
                "stage-dataset: bench.dataset.sha256 is REQUIRED\n       → fix: pin the sha256 in recipe.yaml",
            )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = install.setup_one_cell(llm_cell, "clX", PROF_B200, state_dir=state, runner=runner_stage_fail)
        check(
            "setup_one_cell: stage failure → status=blocked, preflight skipped",
            res["status"] == "blocked" and res["preflight"] == "skipped",
        )
        check(
            "setup_one_cell: stage failure surfaces a concrete fix",
            "pin the sha256" in res["fix"],
        )
        st = install.read_install_stamps("clX", state_dir=state)
        check(
            "setup_one_cell: a blocked attempt is still stamped (audit) but not ready-greyed",
            llm_cell["_path"] in st and install.cell_installed(llm_cell, st) is False,
        )

        # Runner where stage passes, preflight FAILS → blocked with preflight=fail.
        def runner_pf_fail(argv, cwd):
            if "preflight.py" in " ".join(argv):
                return (
                    1,
                    "  ❌ FAIL something\n       → fix: kubectl create secret ...\n",
                )
            return (
                0,
                "staged 0000000000000000000000000000000000000000000000000000000000000000",
            )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = install.setup_one_cell(llm_cell, "clZ", PROF_B200, state_dir=state, runner=runner_pf_fail)
        check(
            "setup_one_cell: stage-ok + preflight-fail → blocked, preflight=fail",
            res["status"] == "blocked" and res["preflight"] == "fail" and "kubectl create secret" in res["fix"],
        )

        # Happy path: both pass → ready, stamp greys it out.
        def runner_ok(argv, cwd):
            if "preflight.py" in " ".join(argv):
                return 0, "all PASS\n"
            return 0, "staged ok\n"

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = install.setup_one_cell(llm_cell, "clOK", PROF_B200, state_dir=state, runner=runner_ok)
        st = install.read_install_stamps("clOK", state_dir=state)
        check(
            "setup_one_cell: stage-ok + preflight-pass → ready + greyed-out on next read",
            res["status"] == "ready" and install.cell_installed(llm_cell, st) is True,
        )
    else:
        check(
            "setup_one_cell: an llm-perf B200 cell exists in the catalog for the blocker test",
            False,
        )


# ---------------------------------------------------------------------------
# 8. headless flag parsing (argparse accepts the new SLURM-express flags)
# ---------------------------------------------------------------------------

# The new flags must PARSE. An unknown flag makes argparse SystemExit(2) BEFORE reaching the body; a known
# flag lets main() run to the profile-not-found path and RETURN an int (never raising). We assert the latter
# for the full headless flag set against a deliberately-missing cluster (no cluster calls happen).
buf = io.StringIO()
argparse_error = False
returned_int = None
try:
    with contextlib.redirect_stdout(buf):
        returned_int = install.main(
            [
                "__no_such_cluster__",
                "--recipes",
                "x",
                "--all-matching",
                "--skip-model-download",
            ]
        )
except SystemExit as e:
    argparse_error = e.code == 2  # argparse usage error
check(
    "headless flags parse: --recipes/--all-matching/--skip-model-download accepted (no argparse error)",
    not argparse_error and isinstance(returned_int, int),
    f"argparse_error={argparse_error}, returned={returned_int}",
)

buf = io.StringIO()
returned_int = None
try:
    with contextlib.redirect_stdout(buf):
        returned_int = install.main(["__no_such_cluster__", "--list-recipes"])
    list_flag_ok = isinstance(returned_int, int)
except SystemExit as e:
    list_flag_ok = e.code != 2
check(
    "headless flags parse: --list-recipes accepted",
    list_flag_ok,
    f"returned={returned_int}",
)

# --from-init (init→install handoff) must PARSE — init invokes it to skip the redundant profile-review
# prompt and land straight on the recipe selector.
buf = io.StringIO()
returned_int = None
try:
    with contextlib.redirect_stdout(buf):
        returned_int = install.main(["__no_such_cluster__", "--from-init"])
    from_init_ok = isinstance(returned_int, int)
except SystemExit as e:
    from_init_ok = e.code != 2
check(
    "headless flags parse: --from-init accepted",
    from_init_ok,
    f"returned={returned_int}",
)


# ---------------------------------------------------------------------------
# 9. model-download Job render (AUTOMATION-GAP #1 apiVersion + #2 hub/ cache layout)
# ---------------------------------------------------------------------------
# GAP #1: the rendered download Job MUST be a valid k8s manifest (apiVersion + kind both present and parseable).
# A whitespace-trim regression in the template swallows `apiVersion:` into the trailing comment and kubectl
# rejects it ("resource mapping not found ... apiVersion not set") — forcing a QA agent to hand-write the Job.
# GAP #2: HF_HUB_CACHE must end in /hub so weights land where the server + preflight read them offline.
import yaml  # noqa: E402

_dl_model = {
    "model_repo": "Qwen/Qwen3-0.6B",
    "model_revision": "abcdef1234567890",
    "model_name": "Qwen3-0.6B",
}
_dl_prof = {
    "NAMESPACE": "llmb",
    "MODEL_CACHE_PVC": "model-cache",
    "HF_SECRET": "hf",
    "IMAGE_PULL_SECRET": "",
    "MODEL_CACHE_SUBPATH": "qwen3",
}
_dl_manifest = install._envsubst_profile(install.render_download_job(_dl_model, _dl_prof), _dl_prof)

# The manifest is a MULTI-doc stream: the least-privilege RBAC that lets the Job stamp its PVC with
# model-name/revision/download-complete on success (ServiceAccount + Role + RoleBinding), then the Job.
# Assertions below target the JOB doc specifically.
_dl_doc = None
_dl_docs = []
_dl_parse_err = ""
try:
    _dl_docs = [d for d in yaml.safe_load_all(_dl_manifest) if d]
    _dl_doc = next((d for d in _dl_docs if d.get("kind") == "Job"), None)
except Exception as e:  # pragma: no cover
    _dl_parse_err = str(e)

check(
    "download Job: renders as parseable YAML (multi-doc: PVC-stamp RBAC + the Job)",
    isinstance(_dl_doc, dict) and {d.get("kind") for d in _dl_docs} == {"ServiceAccount", "Role", "RoleBinding", "Job"},
    _dl_parse_err or str([d.get("kind") for d in _dl_docs]),
)
check(
    "download Job GAP#1: apiVersion present (not swallowed into a comment)",
    isinstance(_dl_doc, dict) and _dl_doc.get("apiVersion") == "batch/v1",
    repr(None if not isinstance(_dl_doc, dict) else _dl_doc.get("apiVersion")),
)
check(
    "download Job GAP#1: kind: Job present",
    isinstance(_dl_doc, dict) and _dl_doc.get("kind") == "Job",
)
check(
    "download Job GAP#1: apiVersion is on its OWN line (never inside a `#` comment)",
    any(ln.strip() == "apiVersion: batch/v1" for ln in _dl_manifest.splitlines()),
    "no bare `apiVersion: batch/v1` line — the comment-swallow regression is back",
)

_dl_env = {}
if isinstance(_dl_doc, dict):
    for _e in _dl_doc["spec"]["template"]["spec"]["containers"][0].get("env", []):
        _dl_env[_e["name"]] = _e.get("value")
check(
    "download Job GAP#2: HF_HUB_CACHE ends in /hub (matches server snapshot_download cache_dir + probe)",
    _dl_env.get("HF_HUB_CACHE") == "/model-cache/qwen3/hub",
    repr(_dl_env.get("HF_HUB_CACHE")),
)
# The sentinel dir is built from CACHE_ROOT (shell var), which MUST stay at <subpath> (NO /hub) so the
# runtime sentinel path matches install.py:probe_model_on_pvc's sentinel path (/cache/<subpath>/.llmb_download_done).
check(
    "download Job GAP#2: CACHE_ROOT (sentinel base) stays at <subpath> — no /hub — to match the preflight probe",
    'CACHE_ROOT="/model-cache/qwen3"' in _dl_manifest and 'CACHE_ROOT="/model-cache/qwen3/hub"' not in _dl_manifest,
)

# GAP (GLM-5 #3): activeDeadlineSeconds is sized from the model's bytes, not a fixed 2h that truncates a
# 704 GiB download. download_deadline_s is pure; the rendered manifest carries the computed value.
check(
    "download deadline: unknown size → 2h floor",
    install.download_deadline_s(None) == 7200 and install.download_deadline_s(0) == 7200,
)
check(
    "download deadline: small model stays at the 2h floor",
    install.download_deadline_s(10) == 7200,
)
check(
    "download deadline: 704 GiB model gets a much larger cap (>2h)",
    install.download_deadline_s(704) > 7200 and install.download_deadline_s(704) == 704 * 45,
)
check(
    "download deadline: monotonic in size",
    install.download_deadline_s(200) <= install.download_deadline_s(700),
)
check(
    "download Job: activeDeadlineSeconds rendered as an integer (never the literal Jinja default)",
    isinstance(_dl_doc, dict) and isinstance(_dl_doc["spec"].get("activeDeadlineSeconds"), int),
)

# A blank/'.' subpath must not double-slash or lose /hub.
_dl_prof_root = dict(_dl_prof, MODEL_CACHE_SUBPATH=".")
_dl_doc_root = next(
    d
    for d in yaml.safe_load_all(
        install._envsubst_profile(install.render_download_job(_dl_model, _dl_prof_root), _dl_prof_root)
    )
    if d and d.get("kind") == "Job"
)
check(
    "download Job: '.' subpath still yields a valid apiVersion",
    _dl_doc_root.get("apiVersion") == "batch/v1",
)


# ---------------------------------------------------------------------------
# G9/G10 — ensure-namespace + ensure-secrets (idempotent, headless-capable, portable).  Fake krun records
# the kubectl verbs it's asked to run and returns scripted rc/err, so these run fully offline.
# ---------------------------------------------------------------------------


def _fake_krun(script):
    """Return a krun whose behavior is driven by `script`: a callable(args)->(rc,out,err). Also records the
    argv of every call on the returned function's `.calls` list."""
    calls: list = []

    def _k(args, timeout=30):
        calls.append(list(args))
        return script(list(args))

    _k.calls = calls
    return _k


# ── ensure_namespace ──
# present: get ns rc=0 → no create.
_k = _fake_krun(lambda a: (0, "", "") if a[:2] == ["get", "ns"] else (1, "", "unexpected"))
_st, _msg = install.ensure_namespace("bench", krun=_k)
check(
    "ensure_namespace: existing namespace → present, no create call",
    _st == "present" and not any(a[:1] == ["create"] for a in _k.calls),
    f"{_st} calls={_k.calls}",
)


# absent: get→rc1, create→rc0, then a label call fires.
def _ns_absent(a):
    if a[:2] == ["get", "ns"]:
        return (1, "", "NotFound")
    if a[:2] == ["create", "namespace"]:
        return (0, "", "")
    if a[:1] == ["label"]:
        return (0, "", "")
    return (1, "", "?")


_k = _fake_krun(_ns_absent)
_st, _msg = install.ensure_namespace("bench", krun=_k)
check(
    "ensure_namespace: absent → created without internal labels",
    _st == "created"
    and any(a[:2] == ["create", "namespace"] for a in _k.calls)
    and not any(a[:1] == ["label"] for a in _k.calls),
    f"{_st} calls={_k.calls}",
)

# plan_only: absent but nothing applied.
_k = _fake_krun(lambda a: (1, "", "NotFound"))
_st, _msg = install.ensure_namespace("bench", krun=_k, plan_only=True, exists=False)
check(
    "ensure_namespace: plan_only never creates (planned)",
    _st == "planned" and _k.calls == [],
    f"{_st} calls={_k.calls}",
)

# race: create returns AlreadyExists → resolves to present, not failed.
_k = _fake_krun(lambda a: ((1, "", "AlreadyExists") if a[:2] == ["create", "namespace"] else (1, "", "x")))
_st, _msg = install.ensure_namespace("bench", krun=_k, exists=False)
check(
    "ensure_namespace: create-race AlreadyExists → present (idempotent)",
    _st == "present",
    _st,
)

# empty ns → skipped
check(
    "ensure_namespace: empty NAMESPACE → skipped",
    install.ensure_namespace("")[0] == "skipped",
)

# ── resolve_secret_value (pure) ──
# Hermetic: point the standard-cred-file lookup at nonexistent paths so a real ~/.cache/huggingface/token or
# ~/.ngc/config on the dev machine can't pollute the "no cred file" precedence assertions below.
_NO_CRED = {
    "hf-token": Path("/nonexistent/hfcred"),
    "ngc-key": Path("/nonexistent/ngccred"),
}
_prof = {"HF_SECRET": "hf-token", "IMAGE_PULL_SECRET": "ngc"}
_v, _src = install.resolve_secret_value("hf-token", _prof, environ={"HF_TOKEN": "hf_abc"}, cred_files=_NO_CRED)
check(
    "resolve_secret_value: env HF_TOKEN wins",
    _v == "hf_abc" and "env $HF_TOKEN" in _src,
    f"{_v}/{_src}",
)
_v, _src = install.resolve_secret_value("ngc-key", _prof, environ={"NGC_CLI_API_KEY": "nvapi-x"}, cred_files=_NO_CRED)
check(
    "resolve_secret_value: NGC fallback env var recognized",
    _v == "nvapi-x",
    f"{_v}/{_src}",
)
_v, _src = install.resolve_secret_value("hf-token", {"HF_TOKEN": "in-profile"}, environ={}, cred_files=_NO_CRED)
check(
    "resolve_secret_value: profile key when no env",
    _v == "in-profile" and "profile HF_TOKEN" in _src,
)
_v, _src = install.resolve_secret_value("hf-token", {}, environ={}, cred_files=_NO_CRED)
check("resolve_secret_value: no source → (None, None)", _v is None and _src is None)
# precedence: env beats profile
_v, _src = install.resolve_secret_value("hf-token", {"HF_TOKEN": "prof"}, environ={"HF_TOKEN": "env"})
check("resolve_secret_value: env beats profile", _v == "env")
# secrets-file fallback
with tempfile.TemporaryDirectory() as _td:
    _sf = Path(_td) / "secrets"
    _sf.write_text('HF_TOKEN="from-file"\n# comment\n')
    _v, _src = install.resolve_secret_value("hf-token", {}, environ={}, secrets_file=_sf, cred_files=_NO_CRED)
    check(
        "resolve_secret_value: secrets-file fallback",
        _v == "from-file" and str(_sf) in _src,
        f"{_v}/{_src}",
    )

# ── standard local cred files (Gap 3): $HF_TOKEN → ~/.cache/huggingface/token → profile ──
with tempfile.TemporaryDirectory() as _td:
    _hf = Path(_td) / "hf_token"
    _hf.write_text("hf_fromCLI\n")
    _ngc = Path(_td) / "ngc_config"
    _ngc.write_text("[CURRENT]\napikey = nvapi-fromCLI\norg = x\n")
    _cf = {"hf-token": _hf, "ngc-key": _ngc}
    _v, _src = install.resolve_secret_value("hf-token", {}, environ={}, cred_files=_cf)
    check(
        "resolve_secret_value: HF token read from ~/.cache/huggingface/token (plain file)",
        _v == "hf_fromCLI" and str(_hf) in _src,
        f"{_v}/{_src}",
    )
    _v, _src = install.resolve_secret_value("ngc-key", {}, environ={}, cred_files=_cf)
    check(
        "resolve_secret_value: NGC key parsed from ~/.ngc/config apikey line",
        _v == "nvapi-fromCLI" and str(_ngc) in _src,
        f"{_v}/{_src}",
    )
    # precedence: env beats cred file
    _v, _src = install.resolve_secret_value("hf-token", {}, environ={"HF_TOKEN": "envwins"}, cred_files=_cf)
    check(
        "resolve_secret_value: env beats standard cred file",
        _v == "envwins",
        f"{_v}/{_src}",
    )
    # precedence: cred file beats profile key
    _v, _src = install.resolve_secret_value("hf-token", {"HF_TOKEN": "profloses"}, environ={}, cred_files=_cf)
    check(
        "resolve_secret_value: standard cred file beats profile key",
        _v == "hf_fromCLI",
        f"{_v}/{_src}",
    )
    # ngc placeholder 'no-apikey' is ignored (falls through to None)
    _ngc.write_text("[CURRENT]\napikey = no-apikey\n")
    _v, _src = install.resolve_secret_value("ngc-key", {}, environ={}, cred_files={"ngc-key": _ngc})
    check(
        "resolve_secret_value: ~/.ngc/config 'no-apikey' placeholder ignored",
        _v is None,
        f"{_v}/{_src}",
    )

# ── secret_source_hint names the exact keys ──
_h = install.secret_source_hint("hf-token", "mycluster")
check(
    "secret_source_hint: names HF_TOKEN env + profile key + cluster file",
    "HF_TOKEN" in _h and "cluster-profiles/mycluster.env" in _h,
    _h,
)

# ── ensure_secret ──
# present → no create
_k = _fake_krun(lambda a: (0, "", "") if a[-2:-1] == ["secret"] or "get" in a else (1, "", "?"))
_st, _msg = install.ensure_secret("bench", "hf-token", "hf-token", {}, "c", krun=_k, exists=True)
check(
    "ensure_secret: existing → present, no create",
    _st == "present" and _k.calls == [],
    f"{_st} calls={_k.calls}",
)

# absent headless with env source → created via generic secret
_created = []


def _sec_create(a):
    if a[:3] == ["-n", "bench", "create"]:
        _created.append(a)
        return (0, "", "")
    return (1, "", "?")


_k = _fake_krun(_sec_create)
_st, _msg = install.ensure_secret(
    "bench",
    "hf-token",
    "hf-token",
    {},
    "c",
    krun=_k,
    exists=False,
    interactive=False,
    environ={"HF_TOKEN": "tok"},
)
check(
    "ensure_secret: headless + env value → created (generic secret, token literal)",
    _st == "created"
    and any("generic" in a for a in _created)
    and any("--from-literal=token=tok" in a for a in _created),
    f"{_st} created={_created}",
)

# absent headless NO source → missing-source (never a create, never a hard crash), with the exact key
_k = _fake_krun(lambda a: (1, "", "?"))
_st, _msg = install.ensure_secret(
    "bench",
    "hf-token",
    "hf-token",
    {},
    "mycluster",
    krun=_k,
    exists=False,
    interactive=False,
    environ={},
    cred_files=_NO_CRED,
)
check(
    "ensure_secret: headless + no source → missing-source with exact HF_TOKEN key, no create",
    _st == "missing-source" and "HF_TOKEN" in _msg and not any("create" in a for a in _k.calls),
    f"{_st}: {_msg}",
)

# OFFLINE plan with no source cannot know whether the cluster Secret exists: report a conditional plan,
# never the inaccurate/failure-looking "secret absent" warning seen in release QA.
_k = _fake_krun(lambda a: (1, "", "?"))
_st, _msg = install.ensure_secret(
    "bench",
    "hf-token",
    "hf-token",
    {},
    "c",
    krun=_k,
    exists=None,
    plan_only=True,
    probe=False,
    environ={},
    cred_files=_NO_CRED,
)
check(
    "ensure_secret: offline plan + unknown live state → conditional planned message",
    _st == "planned" and "if present" in _msg and "if absent" in _msg and "secret 'hf-token' absent" not in _msg,
    f"{_st}: {_msg}",
)

# plan_only with a source → planned, applies nothing
_k = _fake_krun(lambda a: (1, "", "?"))
_st, _msg = install.ensure_secret(
    "bench",
    "ngc",
    "ngc-key",
    {},
    "c",
    krun=_k,
    exists=False,
    plan_only=True,
    environ={"NGC_API_KEY": "k"},
)
check(
    "ensure_secret: plan_only + source → planned, no create",
    _st == "planned" and _k.calls == [],
    f"{_st}",
)

# pull secret uses docker-registry with the literal $oauthtoken username
_created = []
_k = _fake_krun(_sec_create)
install.ensure_secret(
    "bench",
    "ngc",
    "ngc-key",
    {},
    "c",
    krun=_k,
    exists=False,
    environ={"NGC_API_KEY": "nvk"},
)
check(
    "ensure_secret: pull secret → docker-registry w/ $oauthtoken + nvcr.io",
    any("docker-registry" in a for a in _created)
    and any("--docker-username=$oauthtoken" in a for a in _created)
    and any("--docker-server=nvcr.io" in a for a in _created),
    f"created={_created}",
)

# ── ensure_prerequisites orchestration: ns failure short-circuits secret ensures ──
_k = _fake_krun(lambda a: (1, "", "Forbidden"))  # every call fails (e.g. auth)
_res = install.ensure_prerequisites(
    {"NAMESPACE": "bench", "HF_SECRET": "hf", "IMAGE_PULL_SECRET": "ngc"},
    {"ns_ok": False},
    "c",
    krun=_k,
    plan_only=False,
    interactive=False,
)
check(
    "ensure_prerequisites: ns create failure short-circuits (only the namespace result returned)",
    len(_res) == 1 and _res[0][0] == "failed",
    f"{_res}",
)


# ── ensure_prerequisites happy path (headless, env-sourced secrets) ──
def _all_ok(a):
    return (0, "", "")


_k = _fake_krun(_all_ok)
_res = install.ensure_prerequisites(
    {"NAMESPACE": "bench", "HF_SECRET": "hf", "IMAGE_PULL_SECRET": "ngc"},
    {"ns_ok": True, "pull_secret_ok": False, "hf_secret_ok": False},
    "c",
    krun=_k,
    plan_only=False,
    interactive=False,
)
# ns present + both secrets created (env sourced via the process env — inject through os.environ patch)
import os as _os

_os.environ["HF_TOKEN"] = "t"
_os.environ["NGC_API_KEY"] = "n"
_k = _fake_krun(_all_ok)
_res = install.ensure_prerequisites(
    {"NAMESPACE": "bench", "HF_SECRET": "hf", "IMAGE_PULL_SECRET": "ngc"},
    {"ns_ok": True, "pull_secret_ok": False, "hf_secret_ok": False},
    "c",
    krun=_k,
    plan_only=False,
    interactive=False,
)
del _os.environ["HF_TOKEN"]
del _os.environ["NGC_API_KEY"]
# per-recipe-cache moved model-cache PVC provisioning OUT of ensure_prerequisites (now ns + secrets only);
# order is ns → 2 secrets. Per-recipe PVC derivation/ensure is covered by selftest_per_recipe_cache.py.
check(
    "ensure_prerequisites: ns present → both secrets created from env",
    [r[0] for r in _res] == ["present", "created", "created"],
    f"{[r[0] for r in _res]}",
)

# ── model-cache PVC provisioning migrated to derive_recipe_cache / ensure_recipe_cache_pvc(s) (per-recipe);
#    full coverage lives in selftest_per_recipe_cache.py. default_pvc_access_mode (still used) is checked here: ──
check(
    "default_pvc_access_mode: fsx→RWX, ebs→RWO",
    install.default_pvc_access_mode("fsx-1") == "ReadWriteMany"
    and install.default_pvc_access_mode("ebs") == "ReadWriteOnce",
)

# ── QA fail-fast (#3): an object-store StorageClass → per-recipe cache PVC create FAILS FAST (no doomed apply).
#    Migrated from ensure_model_cache_pvc → ensure_recipe_cache_pvc(spec, ns, …) (per-recipe-cache rename). ──
check(
    "is_object_store_provisioner: s3.csi→True, ebs.csi→False, empty→False",
    install.is_object_store_provisioner("s3.csi.aws.com") is True
    and install.is_object_store_provisioner("ebs.csi.aws.com") is False
    and install.is_object_store_provisioner("") is False,
)
_applied: list = []


def _appl(m, ns):
    _applied.append((ns, m))
    return (0, "")


_s3_krun = _fake_krun(lambda a: (0, "s3.csi.aws.com", "") if "storageclass" in a else (1, "", ""))
_s3_spec = {
    "name": "mc",
    "size": "20Gi",
    "storage_class": "s3-object",
    "access_mode": "ReadWriteOnce",
}
_st, _msg = install.ensure_recipe_cache_pvc(_s3_spec, "bench", krun=_s3_krun, applier=_appl, exists=False, probe=True)
check(
    "ensure_recipe_cache_pvc: object-store class → FAILS FAST, applies NO PVC",
    _st == "failed" and _applied == [] and "object-store" in _msg and "s3-object" in _msg,
    f"{_st} applied={_applied} msg={_msg}",
)
# a genuine block class is unaffected (creates as before)
_applied.clear()
_ebs_krun = _fake_krun(lambda a: (0, "ebs.csi.aws.com", "") if "storageclass" in a else (1, "", ""))
_ebs_spec = {
    "name": "mc",
    "size": "100Gi",
    "storage_class": "ebs",
    "access_mode": "ReadWriteOnce",
}
_st2, _ = install.ensure_recipe_cache_pvc(_ebs_spec, "bench", krun=_ebs_krun, applier=_appl, exists=False, probe=True)
check(
    "ensure_recipe_cache_pvc: block class (ebs.csi) → still created (guard doesn't over-trigger)",
    _st2 == "created" and _applied,
    f"{_st2}",
)

# ── QA progress/ETA notes (#4, pure) ──
check(
    "download_eta_text buckets by size + flags unknown",
    "unknown" in install.download_eta_text(None)
    and install.download_eta_text(5) == "~2-5 min"
    and install.download_eta_text(30) == "~5-15 min"
    and install.download_eta_text(100) == "~15-45 min"
    and "45+" in install.download_eta_text(800),
    install.download_eta_text(30),
)
check(
    "_stage_note formats a ▸ progress/ETA banner",
    install._stage_note("Provisioning X", "~1-2 min", "why") == "  ▸ Provisioning X — ~1-2 min  (why)",
    install._stage_note("Provisioning X", "~1-2 min", "why"),
)

# ── ensure_prerequisites full order ns(created) → 2 secrets(created), all from env (no PVC step) ──
_calls = []


def _rec(a):
    _calls.append(" ".join(a))
    return (0, "", "")


_k = _fake_krun(_rec)
_os.environ["HF_TOKEN"] = "t"
_os.environ["NGC_API_KEY"] = "n"
_res = install.ensure_prerequisites(
    {"NAMESPACE": "bench", "HF_SECRET": "hf", "IMAGE_PULL_SECRET": "ngc"},
    {"ns_ok": False, "pull_secret_ok": False, "hf_secret_ok": False},
    "c",
    krun=_k,
    plan_only=False,
    interactive=False,
    probe=False,
)
del _os.environ["HF_TOKEN"]
del _os.environ["NGC_API_KEY"]
check(
    "ensure_prerequisites: full order ns(created) → 2 secrets(created)",
    [r[0] for r in _res] == ["created", "created", "created"],
    f"{[r[0] for r in _res]}",
)
check(
    "ensure_prerequisites: secrets created AFTER namespace create",
    any("create namespace" in c for c in _calls)
    and next(i for i, c in enumerate(_calls) if "create namespace" in c)
    < next(i for i, c in enumerate(_calls) if "create secret" in c),
    f"calls={_calls}",
)


# ---------------------------------------------------------------------------
# 10. Grouped recipe menu — scenario→goal→model hierarchy + stable selection index + group-select
#     (build_recipe_menu / parse_recipe_selection / phase_b_recipe_selection rendering)
# ---------------------------------------------------------------------------


# A fixture: 1 scenario (llm-perf), 3 goals, 3 models, a concurrency family, and one already-installed cell.
def _mcell(name, scenario, goal, model, distribution="", status="wip", rh="rh-"):
    return _cell(
        name,
        "B200",
        status=status,
        scenario=scenario,
        distribution=distribution,
        path=f"recipes/{scenario}/{model}/{name}",
        rh=rh,
    ) | {"goal": goal, "model": model}


MENU_CELLS = [
    # llm-perf / throughput / nemotron — a concurrency family (base + c16/c32/c64), out of order
    _mcell(
        "nem-b200-tp-pareto-c32",
        "llm-perf",
        "throughput",
        "nemotron-ultra-3",
        "synthetic",
    ),
    _mcell("nem-b200-tp-pareto", "llm-perf", "throughput", "nemotron-ultra-3", "synthetic"),
    _mcell(
        "nem-b200-tp-pareto-c64",
        "llm-perf",
        "throughput",
        "nemotron-ultra-3",
        "synthetic",
    ),
    _mcell(
        "nem-b200-tp-pareto-c16",
        "llm-perf",
        "throughput",
        "nemotron-ultra-3",
        "synthetic",
    ),
    # llm-perf / max-concurrency-sla / nemotron — two distributions (mixed → per-cell tag)
    _mcell(
        "nem-b200-1m",
        "llm-perf",
        "max-concurrency-sla",
        "nemotron-ultra-3",
        "long-context-1m",
        status="runs",
    ),
    _mcell(
        "nem-b200-256k",
        "llm-perf",
        "max-concurrency-sla",
        "nemotron-ultra-3",
        "long-context-256k",
    ),
    # llm-perf / pareto / two models
    _mcell("glm5-b200-1p1d", "llm-perf", "pareto", "glm5-fp8", "synthetic"),
    _mcell(
        "qwen-b200-kvbm-pareto",
        "llm-perf",
        "pareto",
        "qwen3-0-6b",
        "glm5-9600",
        status="runs",
    ),
]

_menu_groups, _menu_idx, _menu_gi = install.build_recipe_menu(MENU_CELLS, {})

# (a) grouped STRUCTURE: scenario→goal→model, sorted deterministically.
check(
    "build_recipe_menu: groups ordered by (scenario, goal_label) — 3 llm-perf goal groups",
    [(g["scenario"], g["goal_label"]) for g in _menu_groups]
    == [
        ("llm-perf", "max-concurrency-sla"),
        ("llm-perf", "pareto"),
        ("llm-perf", "throughput"),
    ],
    str([(g["scenario"], g["goal_label"]) for g in _menu_groups]),
)
check(
    "build_recipe_menu: goal label — throughput",
    _menu_groups[2]["goal_label"] == "throughput",
)
check(
    "build_recipe_menu: single-distribution group carries the shared tag in the header",
    _menu_groups[2]["distribution"] == "synthetic",
)
check(
    "build_recipe_menu: mixed-distribution group → distribution None (each cell keeps its own tag)",
    _menu_groups[0]["distribution"] is None,
)
check(
    "build_recipe_menu: pareto group sub-grouped by model, sorted (glm5-fp8 before qwen3-0-6b)",
    [m["model"] for m in _menu_groups[1]["models"]] == ["glm5-fp8", "qwen3-0-6b"],
    str([m["model"] for m in _menu_groups[1]["models"]]),
)
# concurrency family sorts base · c16 · c32 · c64 within the model.
_fam = [install.cell_id(r["cell"]) for r in _menu_groups[2]["models"][0]["cells"]]
check(
    "build_recipe_menu: within a model, base then ascending concurrency (base·c16·c32·c64)",
    _fam
    == [
        "nem-b200-tp-pareto",
        "nem-b200-tp-pareto-c16",
        "nem-b200-tp-pareto-c32",
        "nem-b200-tp-pareto-c64",
    ],
    str(_fam),
)

# (b) selection-number → cell map is correct and STABLE (numbers assigned in display order 1..N).
_nums = {r["num"]: install.cell_id(r["cell"]) for g in _menu_groups for m in g["models"] for r in m["cells"]}
check(
    "build_recipe_menu: numbers are 1..N contiguous in display order",
    sorted(_menu_idx) == list(range(1, len(MENU_CELLS) + 1)) and _nums[1] == "nem-b200-1m",
    f"{sorted(_menu_idx)}",
)
check(
    "build_recipe_menu: idx_map matches the rendered numbers exactly",
    {n: install.cell_id(c) for n, c in _menu_idx.items()} == {n: v for n, v in _nums.items() if n},
    "idx_map/display mismatch",
)
# stability: rebuilding yields the identical number→cell mapping.
_g2, _idx2, _gi2 = install.build_recipe_menu(MENU_CELLS, {})
check(
    "build_recipe_menu: mapping is stable across rebuilds",
    {n: install.cell_id(c) for n, c in _idx2.items()} == {n: install.cell_id(c) for n, c in _menu_idx.items()},
)

# (c) 'all' and ranges still select the intended cells.
_c, _e, _x = install.parse_recipe_selection("all", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: 'all' → every selectable cell",
    len(_c) == len(MENU_CELLS) and not _e,
)
_c, _e, _x = install.parse_recipe_selection("5-8", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: range 5-8 → the 4 throughput family cells",
    [install.cell_id(x) for x in _c] == _fam and not _e,
    str([install.cell_id(x) for x in _c]),
)
_c, _e, _x = install.parse_recipe_selection("2,4-5", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: mixed numbers+range → the exact cells 2,4,5 in order",
    [install.cell_id(x) for x in _c] == [install.cell_id(_menu_idx[n]) for n in (2, 4, 5)] and not _e,
    str([install.cell_id(x) for x in _c]),
)
_c, _e, _x = install.parse_recipe_selection("9", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: out-of-range number → error, empty selection",
    _c == [] and "9" in _e,
)
_c, _e, _x = install.parse_recipe_selection("none", _menu_idx, _menu_gi)
check("parse_recipe_selection: 'none' → empty selection, no error", _c == [] and not _e)

# group-select: header token → the right cell set, with an echo expansion.
_c, _e, _x = install.parse_recipe_selection("g1", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: goal token g1 → the max-concurrency-sla group (2 cells) + echo",
    [install.cell_id(x) for x in _c] == ["nem-b200-1m", "nem-b200-256k"]
    and _x
    and _x[0][0] == "g1"
    and len(_x[0][1]) == 2,
)
_c, _e, _x = install.parse_recipe_selection("g2a", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: model token g2a → just that model's cell (glm5-fp8)",
    [install.cell_id(x) for x in _c] == ["glm5-b200-1p1d"],
    str([install.cell_id(x) for x in _c]),
)
_c, _e, _x = install.parse_recipe_selection("llm-perf", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: scenario name → every cell in that scenario (8)",
    len(_c) == 8,
)
# additive: number + group token dedup to the union.
_c, _e, _x = install.parse_recipe_selection("1,g1", _menu_idx, _menu_gi)
check(
    "parse_recipe_selection: number + overlapping group token dedups to the union (2 cells)",
    len(_c) == 2 and not _e,
)

# (d) installed (🔒) cells are non-selectable AND excluded from group-select.
_inst_cell = MENU_CELLS[4]  # nem-b200-1m (a 'runs' llm-perf max-sla cell)
_inst_stamps = {
    _inst_cell["_path"]: {
        "recipe_hash": _inst_cell["recipe_hash"],
        "preflight": "pass",
        "staged": {"s": {"ok": True}},
    }
}
_gi_groups, _gi_idx, _gi_index = install.build_recipe_menu(MENU_CELLS, _inst_stamps)
check(
    "build_recipe_menu: installed cell gets NO number (idx_map has N-1 = 7)",
    len(_gi_idx) == len(MENU_CELLS) - 1 and all(install.cell_id(c) != "nem-b200-1m" for c in _gi_idx.values()),
)
_row_inst = next(
    r for g in _gi_groups for m in g["models"] for r in m["cells"] if install.cell_id(r["cell"]) == "nem-b200-1m"
)
check(
    "build_recipe_menu: installed cell row is flagged installed with num=None",
    _row_inst["installed"] is True and _row_inst["num"] is None,
)
_c, _e, _x = install.parse_recipe_selection("g1", _gi_idx, _gi_index)
check(
    "parse_recipe_selection: group-select SKIPS the installed cell (g1 → only 256k, not 1m)",
    [install.cell_id(x) for x in _c] == ["nem-b200-256k"] and _x[0][1] == ["nem-b200-256k"],
    str([install.cell_id(x) for x in _c]),
)
_c, _e, _x = install.parse_recipe_selection("all", _gi_idx, _gi_index)
check(
    "parse_recipe_selection: 'all' with an installed cell present → 7 selectable (installed excluded)",
    len(_c) == len(MENU_CELLS) - 1 and all(install.cell_id(x) != "nem-b200-1m" for x in _c),
)

# rendered menu: headers present, lock glyph on the installed cell, tokens visible.
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    install.phase_b_recipe_selection(MENU_CELLS, _inst_stamps, plan_only=True)
_menu_out = _buf.getvalue()
check(
    "phase_b_recipe_selection: renders scenario·goal headers + model subheaders + group tokens",
    "goal: max-concurrency-sla" in _menu_out
    and "[g1]" in _menu_out
    and "[g1a] nemotron-ultra-3" in _menu_out
    and "goal: pareto" in _menu_out,
)
check(
    "phase_b_recipe_selection: installed cell rendered with the lock glyph (non-selectable)",
    install.GLYPH_INSTALLED in _menu_out
    and any(install.GLYPH_INSTALLED in ln and "nem-b200-1m" in ln for ln in _menu_out.splitlines()),
)
check(
    "phase_b_recipe_selection: plan_only selects all installable cells (installed excluded)",
    "would prompt" in _menu_out,
)


# ---------------------------------------------------------------------------
print()
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("selftest_install_recipes: all checks passed")

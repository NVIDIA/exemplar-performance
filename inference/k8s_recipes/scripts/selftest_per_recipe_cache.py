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

"""selftest_per_recipe_cache.py — unit tests for install.py's PER-RECIPE model-cache system (no cluster).

Covers per-recipe cache allocation and validation:
  - derive_recipe_cache: RWX vs RWO storage-class selection from the profile candidates
  - derive name: DELEGATED to resolve_cache_claim (profile-only; see selftest_cache_claim_agreement.py).
    A recipe can no longer name a claim -- `requires.cache.name` is ignored here and REJECTED by validate.
  - derive size: recipe → profile MODEL_CACHE_SIZE → 100Gi floor
  - ensure_recipe_cache_pvc: create-when-absent (captures manifest), present→untouched, plan_only, never-clobber
  - ensure_recipe_cache_pvcs: multi-recipe install → N correct caches; dedup when two recipes share a model
  - cache_pvc_by_repo: model_repo → per-recipe claim map (drives Phase C download target)
  - HASH DISCIPLINE: the recipe cache fields are NOT recipe_hash determinants (byte-identical present vs absent)

Fully offline: the applier + krun are injected fakes; recipe_hash uses a real on-disk migrated cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import install  # noqa: E402  # type: ignore[import]
import recipe_hash as rh  # noqa: E402  # type: ignore[import]

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    marker = "PASS" if cond else "FAIL"
    print(f"  {marker}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# A profile that names BOTH storage-class candidates (the new cluster-specific part).
# A claim NAME is cluster truth, so every profile here must name one -- an unnamed claim is UNRESOLVED and
# install refuses it (that refusal is covered in selftest_cache_claim_agreement.py). These fixtures exercise
# the parts derive_recipe_cache still owns: SIZE and STORAGE CLASS.
PROF = {
    "NAMESPACE": "bench",
    "KUBE_CONTEXT": "ctx",
    "MODEL_CACHE_PVC": "model-cache",
    "MODEL_CACHE_RWX_CLASS": "fsx-lustre",
    "MODEL_CACHE_RWO_CLASS": "ebs",
}


def _cell(model, cache=None, path="recipes/x/y"):
    c = {"model": model, "_path": path, "requires": {"gpu": {"count": 4}}}
    if cache is not None:
        c["requires"]["cache"] = cache
    return c


def _disagg_cell(model, cache=None, path="recipes/x/y"):
    c = _cell(model, cache, path)
    c["serving_mode"] = "disaggregated"
    return c


# ---------------------------------------------------------------------------
# derive_recipe_cache — access-mode → storage-class selection
# ---------------------------------------------------------------------------
rwx = install.derive_recipe_cache(_cell("glm5-fp8", {"size": "1200Gi", "access_mode": "rwx"}), PROF)
check("RWX recipe → MODEL_CACHE_RWX_CLASS", rwx["storage_class"] == "fsx-lustre", str(rwx))
check("RWX recipe → ReadWriteMany", rwx["access_mode"] == "ReadWriteMany", str(rwx))
check(
    "RWX recipe → the PROFILE's claim (never a recipe-invented name)",
    rwx["name"] == "model-cache",
    str(rwx),
)
check("RWX recipe → recipe size honored", rwx["size"] == "1200Gi", str(rwx))
check(
    "RWX recipe → source=recipe (the SHAPE came from the recipe)",
    rwx["source"] == "recipe",
    str(rwx),
)
check(
    "RWX recipe → name_source names the profile key that decided it",
    rwx["name_source"] == "MODEL_CACHE_PVC",
    str(rwx),
)

rwo = install.derive_recipe_cache(_cell("qwen3-kvbm", {"size": "20Gi", "access_mode": "rwo"}), PROF)
check("RWO recipe → MODEL_CACHE_RWO_CLASS", rwo["storage_class"] == "ebs", str(rwo))
check("RWO recipe → ReadWriteOnce", rwo["access_mode"] == "ReadWriteOnce", str(rwo))
check("RWO recipe → 20Gi", rwo["size"] == "20Gi", str(rwo))

# Serving topology is authoritative even when an old profile or recipe default points at RWO.
# Disaggregated prefill/decode workers can span nodes and therefore always provision shared access.
disagg = install.derive_recipe_cache(_disagg_cell("glm5-fp8"), PROF, model_sizes_gib=[704])
check(
    "disagg topology → MODEL_CACHE_RWX_CLASS",
    disagg["storage_class"] == "fsx-lustre",
    str(disagg),
)
check(
    "disagg topology → ReadWriteMany",
    disagg["access_mode"] == "ReadWriteMany",
    str(disagg),
)
forced_rwo = install.derive_recipe_cache(
    _disagg_cell("glm5-fp8"),
    {**PROF, "MODEL_CACHE_ACCESS_MODE": "rwo"},
    model_sizes_gib=[704],
)
check(
    "disagg topology overrides unsafe profile RWO default",
    forced_rwo["access_mode"] == "ReadWriteMany",
    str(forced_rwo),
)

# ---------------------------------------------------------------------------
# name overrides + back-compat
# ---------------------------------------------------------------------------
# A recipe naming a claim is a CATEGORY ERROR: install honoured it, the ${MODEL_CACHE_PVC} mount ignored it,
# so the weights landed in one PVC and the server mounted another. It is now inert here and REJECTED by
# `make validate` (schema/envelope.yaml drops the property).
ovr = install.derive_recipe_cache(_cell("glm5-fp8", {"access_mode": "rwx", "name": "my-cache"}), PROF)
check(
    "requires.cache.name is IGNORED (the profile claim wins)",
    ovr["name"] == "model-cache",
    str(ovr),
)

# Unmigrated cell (no requires.cache) on a legacy cluster → uses the single MODEL_CACHE_PVC + its class/size.
legacy_prof = {
    **PROF,
    "MODEL_CACHE_PVC": "shared-model-cache",
    "MODEL_CACHE_STORAGE_CLASS": "old-sc",
    "MODEL_CACHE_SIZE": "800Gi",
}
legacy = install.derive_recipe_cache(_cell("nemotron-ultra-3"), legacy_prof)
check(
    "unmigrated + back-compat → legacy PVC name",
    legacy["name"] == "shared-model-cache",
    str(legacy),
)
check(
    "unmigrated + back-compat → legacy class",
    legacy["storage_class"] == "old-sc",
    str(legacy),
)
check("unmigrated + back-compat → legacy size", legacy["size"] == "800Gi", str(legacy))
check(
    "unmigrated → source names the profile key",
    legacy["source"] == "MODEL_CACHE_PVC",
    str(legacy),
)

# Unmigrated cell on a NEW cluster (no MODEL_CACHE_PVC), synthetic _path (no recipe.yaml on disk → model
# size UNRESOLVABLE) → smart derived default name + 100Gi floor + a LOUD size_warning (never a silent undersize).
derived = install.derive_recipe_cache(_cell("nemotron-ultra-3"), PROF)
check(
    "unmigrated → still the profile's claim",
    derived["name"] == "model-cache",
    str(derived),
)
check("unresolvable model size → 100Gi floor", derived["size"] == "100Gi", str(derived))
check(
    "unresolvable model size → WARNS (not silent undersize)",
    bool(derived["size_warning"]),
    str(derived),
)
check(
    "unmigrated → source names the profile key",
    derived["source"] == "MODEL_CACHE_PVC",
    str(derived),
)

# access_mode absent but class implies FSx → RWX (default_pvc_access_mode).
implied = install.derive_recipe_cache(
    _cell("m", {"size": "10Gi"}),
    {
        "NAMESPACE": "b",
        "MODEL_CACHE_PVC": "c",
        "MODEL_CACHE_STORAGE_CLASS": "fsx-lustre",
    },
)
check(
    "absent access_mode + fsx class → ReadWriteMany",
    implied["access_mode"] == "ReadWriteMany",
    str(implied),
)

# ---------------------------------------------------------------------------
# ensure_recipe_cache_pvc — create / present / plan / never-clobber
# ---------------------------------------------------------------------------
captured: list[tuple[str, str]] = []


def _applier_ok(manifest, ns):
    captured.append((manifest, ns))
    return (0, "")


spec = install.derive_recipe_cache(_cell("glm5-fp8", {"size": "1200Gi", "access_mode": "rwx"}), PROF)
captured.clear()
st, msg = install.ensure_recipe_cache_pvc(spec, "bench", applier=_applier_ok, exists=False)
check("ensure create → status=created", st == "created", msg)
man = captured[0][0] if captured else ""
check("manifest carries the resolved claim name", "name: model-cache" in man, man)
check("manifest carries RWX access mode", "ReadWriteMany" in man, man)
check("manifest carries RWX class", "storageClassName: fsx-lustre" in man, man)
check("manifest carries size", "storage: 1200Gi" in man, man)
check("manifest labeled managed", 'llmb.nvidia.com/managed: "true"' in man, man)

captured.clear()


# exists=True + a krun reporting AMPLE capacity → present, never clobbered. (Inject krun so the capacity
# probe stays hermetic — no real kubectl.)
def _krun_cap(gib, access_mode="ReadWriteMany"):
    def _k(args, timeout=30):
        joined = " ".join(args)
        if "accessModes" in joined:
            return (0, f"[{access_mode}]", "")
        return (0, f"{gib}Gi", "") if "jsonpath" in joined else (0, "", "")

    return _k


st, _ = install.ensure_recipe_cache_pvc(spec, "bench", applier=_applier_ok, exists=True, krun=_krun_cap(2000))
check("ensure present → never clobbers (no apply)", st == "present" and not captured)
st_bad, msg_bad = install.ensure_recipe_cache_pvc(
    disagg,
    "bench",
    applier=_applier_ok,
    exists=True,
    krun=_krun_cap(2000, "ReadWriteOnce"),
)
check(
    "existing RWO claim is rejected for disaggregated serving",
    st_bad == "access-mismatch" and "ReadWriteOnce" in msg_bad and not captured,
    f"{st_bad}: {msg_bad}",
)

st, msg = install.ensure_recipe_cache_pvc(spec, "bench", applier=_applier_ok, plan_only=True, exists=False)
check("ensure plan_only → planned (no apply)", st == "planned", msg)

# Empty storage class → omit the storageClassName line (cluster default).
nocls = install.render_model_cache_pvc_manifest("ns", "n", "5Gi", "", "ReadWriteOnce")
check("empty class → no storageClassName line", "storageClassName" not in nocls, nocls)

# ---------------------------------------------------------------------------
# ensure_recipe_cache_pvcs — multi-recipe install → N caches, dedup shared model
# ---------------------------------------------------------------------------
cells = [
    _cell("glm5-fp8", {"size": "1200Gi", "access_mode": "rwx"}, path="recipes/a"),
    _cell("qwen3-kvbm", {"size": "20Gi", "access_mode": "rwo"}, path="recipes/b"),
    _cell("nemotron-ultra-3", {"size": "500Gi", "access_mode": "rwx"}, path="recipes/c"),
    # a 4th recipe reusing the SAME model as the 3rd → shares one cache (dedup)
    _cell("nemotron-ultra-3", {"size": "500Gi", "access_mode": "rwx"}, path="recipes/d"),
]
applies: list[str] = []


def _ap(manifest, ns):
    applies.append(manifest)
    return (0, "")


# Distinct claims now come from PER-MODEL PROFILE KEYS (cluster truth), not from recipe-invented names.
PROF_MULTI = {
    **PROF,
    "MODEL_CACHE_PVC_GLM5_FP8": "glm5-fp8-model-cache",
    "MODEL_CACHE_PVC_QWEN3_KVBM": "qwen3-kvbm-model-cache",
    "MODEL_CACHE_PVC_NEMOTRON_ULTRA_3": "nemotron-ultra-3-model-cache",
}
res = install.ensure_recipe_cache_pvcs(cells, PROF_MULTI, applier=_ap, probe=False, plan_only=False)
names = sorted({s["name"] for _, _, s in res})
check(
    "multi-recipe → one spec per DISTINCT claim (dedup shared model)",
    names
    == [
        "glm5-fp8-model-cache",
        "nemotron-ultra-3-model-cache",
        "qwen3-kvbm-model-cache",
    ],
    str(names),
)
check(
    "multi-recipe → 3 PVCs applied (4 recipes, 1 shared)",
    len(applies) == 3,
    str(len(applies)),
)


def _sc_of(manifest: str) -> str:
    for line in manifest.splitlines():
        if "storageClassName" in line:
            return line.split(":", 1)[1].strip()
    return ""


classes = {_sc_of(m) for m in applies}
check(
    "multi-recipe → BOTH RWX and RWO classes selected",
    classes == {"fsx-lustre", "ebs"},
    str(sorted(classes)),
)

# ---------------------------------------------------------------------------
# HASH DISCIPLINE — recipe cache fields are NOT a recipe_hash determinant
# ---------------------------------------------------------------------------
# fingerprint_input must never surface the cache block (it lives under requires, which is excluded save gpu).
fi = rh.fingerprint_input(
    {
        "envelope": {
            "requires": {
                "gpu": {"count": 4},
                "cache": {"size": "1200Gi", "access_mode": "rwx"},
            }
        },
        "serving": {},
        "bench": {},
    },
    ROOT / "recipes",
)
import json as _json  # noqa: E402

check(
    "recipe_hash: cache never enters the fingerprint",
    "cache" not in _json.dumps(fi),
    _json.dumps(fi)[:200],
)
check(
    "recipe_hash: only requires.gpu is carried from requires",
    fi.get("requires_gpu") == {"count": 4},
    str(fi.get("requires_gpu")),
)

# A real on-disk cell fingerprints identically with vs without a requires.cache block. The migrated
# glm5-9600 cell that carried a real cache block was pruned in the consolidation, so we inject a synthetic
# cache block onto a surviving cell's recipe dict (never written to disk) — the fingerprint MUST ignore it.
CELL = ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"
if (CELL / "recipe.yaml").exists():
    import yaml as _yaml

    with_cache = _yaml.safe_load((CELL / "recipe.yaml").read_text())
    with_cache["envelope"].setdefault("requires", {})["cache"] = {
        "name": "synthetic-model-cache",
        "size": "100Gi",
        "access_mode": "rwo",
    }
    without = _yaml.safe_load((CELL / "recipe.yaml").read_text())
    without["envelope"].get("requires", {}).pop("cache", None)
    h_with = rh.fingerprint_input(with_cache, CELL)
    h_without = rh.fingerprint_input(without, CELL)
    check(
        "cell fingerprint byte-identical with vs without a requires.cache block",
        _json.dumps(h_with, sort_keys=True) == _json.dumps(h_without, sort_keys=True),
    )

# ---------------------------------------------------------------------------
# CACHE AUTO-SIZING (bug fix) — a cache PVC must FIT the model(s), never a blind 100Gi floor for a 400GB
# model. shared implementation = model_size_gib / _KNOWN_SIZES_GIB (the SAME table the space-check uses).
# ---------------------------------------------------------------------------
# quantity parse/format round-trips
check("parse 500Gi → 500", install._parse_quantity_gib("500Gi") == 500)
check(
    "parse 1.2Ti → 1229 (ceil)",
    install._parse_quantity_gib("1.2Ti") == 1229,
    str(install._parse_quantity_gib("1.2Ti")),
)
check("parse junk → None", install._parse_quantity_gib("banana") is None)
check("fmt 500 → 500Gi", install._fmt_gib(500) == "500Gi")

# auto-fit: model + headroom (max(+25%, +50Gi)); sharing models SUMMED
check(
    "auto-fit [400] → 500 (400 + 100 headroom)",
    install._auto_cache_size_gib([400]) == 500,
    str(install._auto_cache_size_gib([400])),
)
check(
    "auto-fit SUMS shared models [400,704] → 1380",
    install._auto_cache_size_gib([400, 704]) == 1380,
    str(install._auto_cache_size_gib([400, 704])),
)
check("auto-fit [None] → None (unknown)", install._auto_cache_size_gib([None]) is None)

# Regression case: nemotron-ultra-3 (~400GB) cache must be ≥ model+headroom, NOT the 100Gi floor.
nem = install.derive_recipe_cache(_cell("nemotron-ultra-3"), PROF, model_sizes_gib=[400])
check(
    "nemotron cache auto-sizes ≥ model+headroom (NOT 100Gi floor)",
    nem["size_gib"] >= 500 and nem["size"] != "100Gi",
    str(nem),
)

# Same, resolved from a REAL on-disk cell that declares NO requires.cache (the exact retained large-model cell).
QA_CELL = _cell(
    "nemotron-ultra-3",
    path="recipes/llm-perf/nemotron_ultra_disagg/sglang_dynamo/50k2k/nemotron-ultra-nvfp4-b200-sglang-dynamo14-50k2k-1p1d",
)
if (ROOT / QA_CELL["_path"] / "recipe.yaml").exists():
    qa = install.derive_recipe_cache(QA_CELL, PROF)
    check(
        "retained large-model cell (no requires.cache) auto-sizes to 500Gi (not 100Gi)",
        qa["size"] == "500Gi",
        str(qa),
    )

# multi-model SHARED cache sums to fit both (nemotron 400 + glm5 704 → 1380Gi).
# Two models SHARING one claim -- expressed the only way it can be now: one profile claim covers both.
PROF_SHARED = {**PROF, "MODEL_CACHE_PVC": "shared-cache"}
share_cells = [
    _cell(
        "nemotron-ultra-3",
        path="recipes/llm-perf/nemotron_ultra_disagg/sglang_dynamo/50k2k/nemotron-ultra-nvfp4-b200-sglang-dynamo14-50k2k-1p1d",
    ),
    _cell("glm5-fp8", path="recipes/llm-perf/glm5-9600/glm5-fp8-b200-sglang-1p1d"),
]
if all((ROOT / c["_path"] / "recipe.yaml").exists() for c in share_cells):
    shared_applies: list[str] = []

    def _ap_share(manifest, ns):
        shared_applies.append(manifest)
        return (0, "")

    sres = install.ensure_recipe_cache_pvcs(share_cells, PROF_SHARED, applier=_ap_share, probe=False)
    check(
        "shared cache → ONE PVC applied (dedup)",
        len(shared_applies) == 1,
        str(len(shared_applies)),
    )
    check(
        "shared cache → sized to fit BOTH models (1380Gi)",
        "storage: 1380Gi" in (shared_applies[0] if shared_applies else ""),
        shared_applies[0] if shared_applies else "",
    )

# A too-small DECLARATION can never undersize the weights (floored up to model-fit).
tiny_decl = install.derive_recipe_cache(_cell("nemotron-ultra-3", {"size": "100Gi"}), PROF, model_sizes_gib=[400])
check(
    "declared 100Gi but 400GB model → floored up to model-fit (≥500Gi)",
    tiny_decl["size_gib"] >= 500,
    str(tiny_decl),
)

# A legitimately small model with an explicit small declaration is honored (not bumped to the 100Gi floor).
small = install.derive_recipe_cache(
    _cell("qwen3-kvbm", {"size": "20Gi", "access_mode": "rwo"}),
    PROF,
    model_sizes_gib=[None],
)
check(
    "explicit 20Gi for a tiny model is honored (floor doesn't override a declaration)",
    small["size"] == "20Gi",
    str(small),
)

# UNKNOWN model size + nothing declared → floor + WARNING (never silent undersize).
unk = install.derive_recipe_cache(_cell("mystery-model"), PROF, model_sizes_gib=[None])
check("unknown model + no declaration → 100Gi floor", unk["size"] == "100Gi", str(unk))
check(
    "unknown model → emits size_warning (not silent)",
    bool(unk["size_warning"]),
    str(unk),
)

# ensure_recipe_cache_pvc: a PRE-EXISTING PVC smaller than needed → refuse + guide (resize), no download.
under_spec = install.derive_recipe_cache(_cell("nemotron-ultra-3"), PROF, model_sizes_gib=[400])
st_u, msg_u = install.ensure_recipe_cache_pvc(
    under_spec, "bench", applier=_applier_ok, exists=True, krun=_krun_cap(100)
)  # existing PVC only 100Gi
check(
    "pre-existing undersized PVC → status=undersized (refuse+guide, no clobber)",
    st_u == "undersized",
    f"{st_u}: {msg_u}",
)
check("undersized message names the required size", "≥500Gi" in msg_u, msg_u)
# a marginally-smaller PVC (within ~5%) is tolerated (headroom is slack).
st_m, _ = install.ensure_recipe_cache_pvc(under_spec, "bench", applier=_applier_ok, exists=True, krun=_krun_cap(490))
check("marginally-smaller PVC (≤5%) → present (tolerated)", st_m == "present", st_m)

# space-check verdict — the download-path guard. A guaranteed-negative deficit HARD-BLOCKS (no casual proceed).
check(
    "space verdict: -302 GiB deficit → block (no proceed prompt)",
    install.pvc_space_verdict(-302, 400) == "block",
)
check("space verdict: positive → ok", install.pvc_space_verdict(50, 400) == "ok")
check(
    "space verdict: tiny overage → marginal (warn+proceed)",
    install.pvc_space_verdict(-5, 400) == "marginal",
)

# ---------------------------------------------------------------------------
# CACHE NAME ↔ SERVER MOUNT RECONCILIATION (Option B) — install MUST provision + download into the SAME
# claim the rendered server/bench manifests mount (they hardcode `claimName: ${MODEL_CACHE_PVC}`).
# A migrated cell must NOT jump to `<slug>-model-cache` when the profile sets MODEL_CACHE_PVC — that created
# an orphan PVC + a latent model-not-found (download → derived name, server → ${MODEL_CACHE_PVC}).
# ---------------------------------------------------------------------------
# Profiles mirroring the real ones: MODEL_CACHE_PVC set, RWX/RWO classes EMPTY.
PROF_LEGACY = {"NAMESPACE": "bench", "MODEL_CACHE_PVC": "model-cache"}
# Surviving cells (post strict-prune) used only for model-SIZE resolution (serving.model_repo → GiB).
# The migrated-cell tests pass an explicit requires.cache dict via _cell(), so only the model size is read here.
# nemotron-ultra-3 → 400 GiB (LARGE). Read from a retained release recipe. This test needs the model-SIZE
# resolution (serving.model_repo → GiB), not the catalog. Pointing at a path that no longer exists
# silently fell back to the 100Gi floor -- which the assertions then read as "large-model cache defaults
# to RWO/ebs", i.e. a wrong PASS-shaped failure.
NEM_PATH = (
    "recipes/llm-perf/nemotron_ultra_disagg/sglang_dynamo/50k2k/nemotron-ultra-nvfp4-b200-sglang-dynamo14-50k2k-1p1d"
)
GLM_PATH = "recipes/llm-perf/Glm5/B200_k8s/Disagg/sglang_dynamo/1k_1k/glm5-fp8-b200-sglang-dynamo14-1k1k-hightpt-c2576-1p1d"  # glm5-fp8

# THE QA-HIT CASE: migrated nemotron cell + MODEL_CACHE_PVC=model-cache → name == 'model-cache' (what the
# server mounts), NOT the orphan 'nemotron-ultra-3-model-cache'.
mig_nem = _cell("nemotron-ultra-3", {"size": "500Gi", "access_mode": "rwo"}, path=NEM_PATH)
d_nem = install.derive_recipe_cache(mig_nem, PROF_LEGACY)
check(
    "migrated cell + MODEL_CACHE_PVC → name == server mount (no orphan)",
    d_nem["name"] == "model-cache",
    str(d_nem),
)
check(
    "migrated cell → still auto-sizes from the model (≥500Gi, not the mount's caseal size)",
    d_nem["size_gib"] >= 500,
    str(d_nem),
)

mig_glm = _disagg_cell("glm5-fp8", {"size": "1200Gi", "access_mode": "rwx"}, path=GLM_PATH)
check(
    "migrated glm5 cell + MODEL_CACHE_PVC → name == 'model-cache'",
    install.derive_recipe_cache(mig_glm, PROF_LEGACY)["name"] == "model-cache",
)

# ensure_recipe_cache_pvcs on the QA recipe set → provisions 'model-cache' ONLY; NO orphan applied.
recon_applies: list[str] = []


def _ap_recon(manifest, ns):
    recon_applies.append(manifest)
    return (0, "")


recon = install.ensure_recipe_cache_pvcs([mig_nem, mig_glm], PROF_LEGACY, applier=_ap_recon, probe=False)
recon_names = sorted({s["name"] for _, _, s in recon})
check(
    "ensure → provisions the mount PVC 'model-cache' (dedup across both migrated cells)",
    recon_names == ["model-cache"],
    str(recon_names),
)
check(
    "aggregate-first shared claim adopts disaggregated RWX requirement",
    len(recon) == 1
    and recon[0][2]["access_mode"] == "ReadWriteMany"
    and any("ReadWriteMany" in m for m in recon_applies),
    str(recon),
)
check(
    "ensure → NO orphan '<slug>-model-cache' PVC created",
    not any("model-cache" != s["name"] for _, _, s in recon)
    and not any("nemotron-ultra-3-model-cache" in m or "glm5-fp8-model-cache" in m for m in recon_applies),
    str(recon_names),
)

# requires.cache.name NO LONGER wins -- it is inert. THIS is the fix: honouring it meant install downloaded
# into 'my-cache' while every rendered manifest mounted ${MODEL_CACHE_PVC}='model-cache'.
d_expl = install.derive_recipe_cache(
    _cell("nemotron-ultra-3", {"name": "my-cache", "size": "500Gi"}, path=NEM_PATH),
    PROF_LEGACY,
)
check(
    "requires.cache.name is INERT (mount and download can no longer diverge)",
    d_expl["name"] == "model-cache",
    str(d_expl),
)
# The supported way to give ONE model its own claim: a per-model key in the profile (cluster truth).
d_perm = install.derive_recipe_cache(
    _cell("nemotron-ultra-3", {"size": "500Gi"}, path=NEM_PATH),
    {**PROF_LEGACY, "MODEL_CACHE_PVC_NEMOTRON_ULTRA_3": "nemotron-dedicated"},
)
check(
    "per-model profile key routes ONE model to its own claim",
    d_perm["name"] == "nemotron-dedicated",
    str(d_perm),
)

# Unmigrated cell is UNCHANGED → MODEL_CACHE_PVC (already agreed; regression guard).
d_unmig = install.derive_recipe_cache(_cell("nemotron-ultra-3", path=NEM_PATH), PROF_LEGACY)
check(
    "unmigrated cell + MODEL_CACHE_PVC → 'model-cache' (unchanged)",
    d_unmig["name"] == "model-cache",
    str(d_unmig),
)

# NO cache key at all → UNRESOLVED. install must REFUSE, never invent a name: an invented claim is one the
# server never mounts (weights downloaded, model-not-found at serve time, 31 cells blocked in Phase D).
d_fresh = install.derive_recipe_cache(mig_nem, {"NAMESPACE": "bench"})
check(
    "no cache key → UNRESOLVED (install refuses; never invents a claim)",
    d_fresh["name"] == "",
    str(d_fresh),
)

# ---------------------------------------------------------------------------
# Size-aware cache defaults: large models prefer high-throughput RWX storage; smaller models may use RWO block storage.
# (loads in seconds — FSx-Lustre's large min-filesystem + slow provision is pure waste). Threshold =
# install._FAST_CACHE_MODEL_GIB. install-side only → recipe_hash unchanged.
# ---------------------------------------------------------------------------
# A fresh per-recipe profile: fast FSx RWX class + EBS RWO class, no legacy STORAGE_CLASS, no MODEL_CACHE_PVC.
PROF_FAST = {
    "NAMESPACE": "bench",
    "MODEL_CACHE_PVC": "model-cache",
    "MODEL_CACHE_RWX_CLASS": "fsx-lustre",
    "MODEL_CACHE_RWO_CLASS": "ebs",
}
KVBM_PATH = "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"  # ~0.6B model → SMALL

# LARGE model (nemotron ~400GB), unmigrated → RWX on the FSx class, NOT ebs.
d_def = install.derive_recipe_cache(_cell("nemotron-ultra-3", path=NEM_PATH), PROF_FAST)
check(
    "large-model cache DEFAULTS to RWX (fast shared FS), not RWO",
    d_def["access_mode"] == "ReadWriteMany",
    str(d_def),
)
check(
    "large-model cache DEFAULTS to the FSx class 'fsx-lustre', not ebs",
    d_def["storage_class"] == "fsx-lustre",
    str(d_def),
)

# SMALL model (KVBM qwen3-0.6B ≈1.2GB, unmigrated, colocated) → EBS RWO AUTOMATICALLY (no requires.cache edit,
# no hash move). This is the size-aware resolution of the tiny-model FSx-over-provision flag.
if (ROOT / KVBM_PATH / "recipe.yaml").exists():
    d_small = install.derive_recipe_cache(_cell("qwen3-0-6b", path=KVBM_PATH), PROF_FAST)
    check(
        "SMALL-model cache DEFAULTS to EBS RWO (not FSx — no min-filesystem waste)",
        d_small["access_mode"] == "ReadWriteOnce" and d_small["storage_class"] == "ebs",
        str(d_small),
    )

# UNKNOWN model size is treated as SMALL → EBS RWO (a genuinely-large model should declare its size/mode).
d_unk = install.derive_recipe_cache(_cell("mystery", path="recipes/x/y"), PROF_FAST)
check(
    "unknown-size model → EBS RWO (treated as small)",
    d_unk["access_mode"] == "ReadWriteOnce" and d_unk["storage_class"] == "ebs",
    str(d_unk),
)

# Threshold boundary: exactly at _FAST_CACHE_MODEL_GIB → FSx; just below → EBS.
thr = install._FAST_CACHE_MODEL_GIB
at = install.derive_recipe_cache(_cell("m", path="recipes/x/y"), PROF_FAST, model_sizes_gib=[thr])
below = install.derive_recipe_cache(_cell("m", path="recipes/x/y"), PROF_FAST, model_sizes_gib=[thr - 1])
check(
    f"model == {thr}GiB threshold → FSx RWX",
    at["storage_class"] == "fsx-lustre",
    str(at),
)
check(
    f"model <  {thr}GiB threshold → EBS RWO",
    below["storage_class"] == "ebs",
    str(below),
)

# A recipe that DECLARES rwx → FSx class (honored regardless of size).
d_rwx = install.derive_recipe_cache(
    _cell("glm5-fp8", {"size": "1200Gi", "access_mode": "rwx"}, path=GLM_PATH),
    PROF_FAST,
)
check("declared rwx cell → FSx class", d_rwx["storage_class"] == "fsx-lustre", str(d_rwx))

# A recipe that DECLARES rwo → EBS class, HONORED (not swept onto FSx) — the single-node opt-in.
d_rwo = install.derive_recipe_cache(
    _cell("nemotron-ultra-3", {"size": "500Gi", "access_mode": "rwo"}, path=NEM_PATH),
    PROF_FAST,
)
check(
    "declared rwo cell → EBS (honored, not swept to FSx)",
    d_rwo["access_mode"] == "ReadWriteOnce" and d_rwo["storage_class"] == "ebs",
    str(d_rwo),
)

# BACK-COMPAT: a legacy profile that pinned MODEL_CACHE_STORAGE_CLASS is honored verbatim (not moved to FSx).
d_legacy = install.derive_recipe_cache(
    _cell("nemotron-ultra-3", path=NEM_PATH),
    {
        "NAMESPACE": "b",
        "MODEL_CACHE_PVC": "c",
        "MODEL_CACHE_STORAGE_CLASS": "old-sc",
        "MODEL_CACHE_RWX_CLASS": "fsx-lustre",
    },
)  # legacy pin present alongside a new RWX class
check(
    "legacy MODEL_CACHE_STORAGE_CLASS pin honored (not silently moved to FSx)",
    d_legacy["storage_class"] == "old-sc",
    str(d_legacy),
)

# No RWX class on the cluster → RWO on the block class (safe fallback, unchanged behavior).
d_norwx = install.derive_recipe_cache(
    _cell("nemotron-ultra-3", path=NEM_PATH),
    {"NAMESPACE": "b", "MODEL_CACHE_PVC": "c", "MODEL_CACHE_RWO_CLASS": "ebs"},
)
check(
    "no RWX class → RWO on the block class (fallback)",
    d_norwx["access_mode"] == "ReadWriteOnce" and d_norwx["storage_class"] == "ebs",
    str(d_norwx),
)

# default_pvc_access_mode recognizes the FSx CSI class NAME (fsx-lustre has no 'fsx' substring).
check(
    "default_pvc_access_mode('fsx-lustre') → RWX (shared-FS name match)",
    install.default_pvc_access_mode("fsx-lustre") == "ReadWriteMany",
)
check(
    "default_pvc_access_mode('ebs') → RWO",
    install.default_pvc_access_mode("ebs") == "ReadWriteOnce",
)

# ---------------------------------------------------------------------------
# STORAGE-TYPE ROBUSTNESS — (1) discover + prefer a pre-existing shared cache; (2) fast-fail + fall back when
# a derived block class won't attach (the shared-cluster block-storage FailedAttachVolume / 8-min-hang case).
# ---------------------------------------------------------------------------
import json as _json2  # noqa: E402

# (1) pick_shared_cache / discover_shared_model_cache were DELETED (T10) — auto-adopting a discovered
# cache wrote prof["MODEL_CACHE_PVC"] in memory, which only install could see. The replacement is
# list_cache_candidates (advisory, no size floor, explicit probe_error); it is covered by
# selftest_cache_claim_agreement.py. Assert the trap cannot come back by accident.
check(
    "pick_shared_cache is gone (discovery must never decide a claim again)",
    not hasattr(install, "pick_shared_cache"),
)
check(
    "discover_shared_model_cache is gone",
    not hasattr(install, "discover_shared_model_cache"),
)

# A namespace listing containing one big shared RWX cache — used by the advisory-path checks below.
_J = _json2.dumps(
    {
        "items": [
            {
                "metadata": {"name": "shared-model-cache", "labels": {}},
                "spec": {
                    "accessModes": ["ReadWriteMany"],
                    "resources": {"requests": {"storage": "50Ti"}},
                },
                "status": {"phase": "Bound", "capacity": {"storage": "50Ti"}},
            }
        ]
    }
)

# (2a) cache_bind_verdict — bound / cinder-attach-fail / benign WaitForFirstConsumer pending.
check("bind verdict: Bound → bound", install.cache_bind_verdict("Bound", "") == "bound")
check(
    "bind verdict: FailedAttachVolume → failed (cinder case)",
    install.cache_bind_verdict("Pending", "Warning FailedAttachVolume: could not be attached") == "failed",
)
check(
    "bind verdict: WaitForFirstConsumer Pending → pending (NOT a failure)",
    install.cache_bind_verdict("Pending", "waiting for first consumer to be created before binding") == "pending",
)


# (2b) wait_for_cache_bind FAST-FAILS on an attach error (no 8-min hang); returns bound immediately when Bound.
def _krun_attachfail(args, timeout=30):
    a = " ".join(args)
    if "jsonpath={.status.phase}" in a:
        return (0, "Pending", "")
    if "events" in a:
        return (0, "Warning FailedAttachVolume volume cannot be attached", "")
    return (0, "", "")


v, _d = install.wait_for_cache_bind("ns", "c", _krun_attachfail, budget_s=90, poll_s=0)
check("wait_for_cache_bind: attach error → 'failed' FAST (no long hang)", v == "failed", v)


def _krun_bound(args, timeout=30):
    return (0, "Bound", "") if "jsonpath={.status.phase}" in " ".join(args) else (0, "", "")


check(
    "wait_for_cache_bind: Bound → 'bound'",
    install.wait_for_cache_bind("ns", "c", _krun_bound, poll_s=0)[0] == "bound",
)

# (2c) ensure_recipe_cache_pvc_validated: block class fails to attach → FALL BACK to the RWX class + message.
_fb_applies: list = []


def _ap_fb(m, ns):
    _fb_applies.append(m)
    return (0, "")


def _krun_block_attachfail(args, timeout=30):
    a = " ".join(args)
    if "jsonpath={.status.phase}" in a:
        return (0, "Pending", "")
    if "events" in a:
        return (0, "Warning FailedAttachVolume could not be attached", "")
    if "get" in a and "pvc" in a:
        return (1, "", "NotFound")  # exists-check → absent → will create
    return (0, "", "")


_spec_block = {
    "name": "nemotron-ultra-3-model-cache",
    "size": "500Gi",
    "size_gib": 500,
    "storage_class": "cinder",
    "access_mode": "ReadWriteOnce",
    "source": "derived",
    "size_source": "size-aware",
}
st_fb, msg_fb, eff_fb = install.ensure_recipe_cache_pvc_validated(
    _spec_block,
    "ns",
    {"NAMESPACE": "ns", "MODEL_CACHE_RWX_CLASS": "vast-rwx"},
    krun=_krun_block_attachfail,
    applier=_ap_fb,
    probe=True,
)
check("attach-fail → status 'fell-back'", st_fb == "fell-back", f"{st_fb}: {msg_fb}")
check(
    "attach-fail → fell back onto the RWX class",
    eff_fb["storage_class"] == "vast-rwx" and eff_fb["access_mode"] == "ReadWriteMany",
    str(eff_fb),
)
check(
    "attach-fail message names the failed class AND the fallback",
    "cinder" in msg_fb and "vast-rwx" in msg_fb,
    msg_fb,
)
check(
    "attach-fail → recreated on RWX (2 applies: block then RWX)",
    len(_fb_applies) == 2,
    str(len(_fb_applies)),
)
# no distinct RWX fallback configured → clean 'attach-failed' with remediation (not a silent hang).
st_af, msg_af, _ = install.ensure_recipe_cache_pvc_validated(
    _spec_block,
    "ns",
    {"NAMESPACE": "ns"},
    krun=_krun_block_attachfail,
    applier=_ap_fb,
    probe=True,
)
check(
    "attach-fail + no RWX fallback → 'attach-failed' with remediation",
    st_af == "attach-failed" and "MODEL_CACHE_RWX_CLASS" in msg_af,
    f"{st_af}: {msg_af}",
)

# (3) A pre-existing shared cache is NO LONGER auto-adopted. Discovery used to write
# prof["MODEL_CACHE_PVC"] = <found> IN MEMORY -- a value only install could see, while deploy.sh, sweep.sh,
# stage-*.sh and preflight all re-read the profile FILE. So install downloaded into the discovered claim and
# the server mounted whatever the file said (including ""). Discovery is now ADVISORY: install prints the
# candidates + the exact profile lines (render_cache_candidates_advice) or writes them under --adopt-cache,
# and REFUSES to proceed on a claim that exists only in its own memory.
_disc_applies: list = []


def _ap_disc(m, ns):
    _disc_applies.append(m)
    return (0, "")


def _krun_shared_present(args, timeout=30):
    a = " ".join(args)
    if "json" in a:
        return (0, _J, "")
    if "jsonpath={.status.capacity" in a or "jsonpath={.spec.resources" in a:
        return (0, "50Ti", "")
    if "get" in a and "pvc" in a:
        return (0, "shared-model-cache", "")  # exists
    return (0, "", "")


_cell_disc = _cell("nemotron-ultra-3", path=NEM_PATH)
_PROF_NOCACHE = {
    "NAMESPACE": "ns",
    "MODEL_CACHE_RWX_CLASS": "fsx-lustre",
    "MODEL_CACHE_RWO_CLASS": "ebs",
}
res_disc = install.ensure_recipe_cache_pvcs(
    [_cell_disc], _PROF_NOCACHE, krun=_krun_shared_present, applier=_ap_disc, probe=True
)
check(
    "a discovered shared cache is NOT silently adopted as the claim",
    res_disc and res_disc[0][2]["name"] == "",
    str(res_disc),
)
check(
    "...and nothing is provisioned on a claim install only imagined",
    len(_disc_applies) == 0,
    str(len(_disc_applies)),
)
# It is still LISTABLE -- so the advice can name it and --adopt-cache can persist it to the FILE.
_cands_adv, _err_adv = install.list_cache_candidates("ns", _krun_shared_present)
check(
    "list_cache_candidates surfaces it for the operator",
    [c["name"] for c in _cands_adv] == ["shared-model-cache"] and _err_adv == "",
    f"{_cands_adv} {_err_adv}",
)
_adv = install.render_cache_candidates_advice(_cands_adv, _err_adv, [_cell_disc], "somecluster")
check(
    "the advice prints the exact profile line to paste",
    'MODEL_CACHE_PVC="' in _adv and "shared-model-cache" in _adv,
    _adv[:200],
)
check(
    "the advice offers the one-command --adopt-cache path",
    "--adopt-cache" in _adv,
    _adv[-200:],
)

# ---------------------------------------------------------------------------
print()
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("selftest_per_recipe_cache: all checks passed")

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

"""Resolve the model-cache PVC consistently for every workflow stage.

The cluster profile owns PVC identity; recipes declare only cache shape such as size and access mode.
Install, preflight, deploy, staging, and model-load checks all call this module so the download target and
mounted claim cannot diverge. Rendered manifests remain cluster-portable and receive the claim through
profile substitution.

    MODEL_CACHE_PVC="glm5-fp8-model-cache"                             # default for every model
    MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4="nemotron-ultra-nvfp4-cache"  # this model only

`requires.cache.name` is rejected by the schema. Recipes may declare cache `size` and `access_mode` for
provisioning.

CLI: `model_cache.py resolve <cell-dir> <profile.env>` → prints the claim, or exits 2 with the exact fix.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def model_cache_env_key(model: str) -> str:
    """PURE — the per-model cluster-profile key naming THIS model's claim, or '' when the model is unknown.
    `nemotron-ultra-nvfp4` → `MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4`.

    Non-alphanumerics all fold to '_', so `a-b`, `a.b` and `a_b` would collide on one key. Unreachable on
    the real catalog (GLM5_FP8 / NEMOTRON_ULTRA_NVFP4 / QWEN3_0_6B are distinct), and a collision could only
    make two models SHARE a claim — the safe direction — but it is a real constraint on model naming.
    """
    slug = re.sub(r"[^A-Z0-9]+", "_", (model or "").upper()).strip("_")
    return f"MODEL_CACHE_PVC_{slug}" if slug else ""


def resolve_cache_claim(cell: dict, prof: dict) -> tuple[str, str]:
    """PURE + TOTAL — the SINGLE definition of which PVC a cell's model weights live in.

    Returns (claim_name, source_key). `source_key` names the profile variable that decided it, so every
    error message can print the exact key to edit. ('', '') means UNRESOLVED — a hard error everywhere,
    never a default, because an empty claimName renders an invalid manifest (or mounts the wrong thing).

    Inputs are PROFILE ONLY (plus the cell's `model`, which selects the override key). No recipe field, no
    cluster probe, no discovery — so the download target and the server mount cannot diverge.

    ACCEPTED LIMITATION: granularity is per-MODEL, not per-CELL. Two cells sharing an `envelope.model` but
    wanting DIFFERENT claims cannot be expressed (the removed `requires.cache.name` could). Deliberate:
    cells sharing a model SHOULD share one copy of the weights, which is the dedup a cache exists for. The
    one real per-invocation case — spreading parallel loads across replica filesystems — is covered by
    MODEL_CACHE_PVC_OVERRIDE (see scripts/_model_cache.sh, used by parallel_repro.sh).
    """
    key = model_cache_env_key(cell.get("model") or "")
    if key:
        v = (prof.get(key) or "").strip()
        if v:
            return (v, key)
    v = (prof.get("MODEL_CACHE_PVC") or "").strip()
    if v:
        return (v, "MODEL_CACHE_PVC")
    return ("", "")


def cache_claim_fix_hint(cell: dict, cluster: str) -> str:
    """PURE — the exact profile edit that resolves an UNRESOLVED claim for this cell. Shared by the early
    install gate, the download guard, preflight and the shell resolver so they all say the same thing.
    """
    key = model_cache_env_key(cell.get("model") or "")
    lines = [f'set MODEL_CACHE_PVC="<pvc>" in cluster-profiles/{cluster}.env']
    if key:
        lines.append(f'or, to give this model its own claim, {key}="<pvc>"')
    return "; ".join(lines)


# ---------------------------------------------------------------------------
# Cache-mounting pod placement
# ---------------------------------------------------------------------------
# Some storage classes mount only on selected nodes. MODEL_CACHE_NODE_SELECTOR records that cluster-specific
# constraint, while tolerations allow cache helpers to use tainted worker nodes.

# Tolerate scheduling taints without suppressing Kubernetes default eviction for unreachable nodes.
_CACHE_TOLERATIONS = [
    {"operator": "Exists", "effect": "NoSchedule"},
    {"operator": "Exists", "effect": "PreferNoSchedule"},
]


def parse_node_selector(spec: str) -> dict:
    """PURE — parse a profile node-selector fragment into a dict.

    Accepts the inline-YAML shape BENCH_NODE_SELECTOR already uses, so an operator learns one syntax:
        'nvidia.com/gpu.present: "true"'        -> {'nvidia.com/gpu.present': 'true'}
        'kubernetes.io/arch: amd64, pool: gpu'  -> {'kubernetes.io/arch': 'amd64', 'pool': 'gpu'}
    Empty or invalid input returns `{}`. Callers reject a non-empty specification that parses to no selector.
    """
    out: dict = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        k, v = k.strip().strip("'\""), v.strip().strip("'\"")
        if k and v:
            out[k] = v
    return out


def cache_pod_placement(prof: dict) -> tuple[dict, list, str]:
    """The placement EVERY cache-mounting pod must use: (node_selector, tolerations, source).

    Download, probe, preflight, and staging pods all use this function so placement remains consistent.
    """
    sel = parse_node_selector(prof.get("MODEL_CACHE_NODE_SELECTOR") or "")
    return (sel, list(_CACHE_TOLERATIONS), "MODEL_CACHE_NODE_SELECTOR" if sel else "")


def cache_placement_hint(prof: dict) -> str:
    """The remediation line to append whenever a cache-mounting pod cannot schedule or cannot mount.

    The message identifies the profile setting that constrains cache mounts to compatible nodes.
    """
    cur = (prof.get("MODEL_CACHE_NODE_SELECTOR") or "").strip()
    if cur:
        return (
            f"MODEL_CACHE_NODE_SELECTOR='{cur}' — if the mount still fails, that selector may be naming "
            f"nodes where this storage class does not actually mount."
        )
    return (
        "MODEL_CACHE_NODE_SELECTOR is unset, so the pod may land on any node. If the storage class is "
        "available only on selected nodes, set an appropriate selector in the cluster profile, for example "
        "MODEL_CACHE_NODE_SELECTOR='nvidia.com/gpu.present: \"true\"'."
    )


# ---------------------------------------------------------------------------
# Model-cache completeness states
# ---------------------------------------------------------------------------
# COMPLETE requires positive evidence for the pinned revision. PRESENT_UNVERIFIED represents existing weights
# whose completeness cannot be proven. INCOMPLETE and ABSENT remain distinct from an unreadable UNKNOWN state.

STATE_COMPLETE = "complete"  # provably all shards present -> safe to skip the download
STATE_PRESENT_UNVERIFIED = "present-unverified"  # weights on disk, completeness NOT provable
STATE_INCOMPLETE = "incomplete"  # provably missing shards -> must re-download
STATE_ABSENT = "absent"  # nothing there
STATE_UNKNOWN = "unknown"  # we could not look (NEVER "absent")

# How far below the index-implied mean shard size a measurement may fall before it is treated as broken
# rather than as evidence. Two orders of magnitude: real truncation shrinks a shard by a factor of ~2-100,
# while a units/dereference bug is off by 1e6 or more (the symlink regression was 7e7).
SIZE_PLAUSIBILITY_FACTOR = 100


def snapshot_dir(subpath: str, repo: str, revision: str) -> str:
    """PURE — the exact huggingface_hub path the SERVER loads from, relative to the cache mount root."""
    org, _, name = (repo or "").partition("/")
    sp = (subpath or ".").strip() or "."
    return f"{sp}/hub/models--{org}--{name}/snapshots/{revision}".replace("//", "/").lstrip("./") or "."


def sentinel_path(subpath: str, revision: str) -> str:
    """PURE — the completion sentinel the download Job writes on success, relative to the mount root."""
    sp = (subpath or ".").strip() or "."
    return f"{sp}/.llmb_download_done/{revision}".replace("//", "/")


def cache_completeness(facts: dict) -> tuple[str, str]:
    """PURE — the ONE completeness predicate. Returns (state, why).

    `facts` is the parsed probe report (see parse_cache_integrity_report):
      exists, config_json, index_json, shard_count, required_shards, missing_shards,
      index_total_bytes, shard_bytes, sentinel  (+ probe_error)

    EVIDENCE, in order:
      * probe_error / no facts                    -> UNKNOWN. We could not look; absence is not zero.
      * snapshot dir missing                      -> ABSENT.
      * the index names shards and ANY is missing -> INCOMPLETE. A partial sharded download crash-loops.
      * the index names shards, all resolve, and the on-disk bytes cover metadata.total_size
                                                  -> COMPLETE. This is the strong proof: it is a BYTE
                                                     claim, so a truncated shard cannot pass by being
                                                     merely present. (Each .safetensors file is header +
                                                     tensor bytes, so sum(file sizes) >= total_size holds
                                                     for an intact snapshot and fails for a truncated one.)
      * a completion SENTINEL written by our own download Job -> COMPLETE (it is written only after
        snapshot_download returned, under `set -e`).
      * single-file model (NO INDEX FILE) + config.json + EXACTLY 1 resolved shard -> COMPLETE.
      * weights present but none of the above provable -> PRESENT_UNVERIFIED, never COMPLETE.

    Note what is deliberately NOT evidence: a bare shard COUNT, and the PVC's download-complete LABEL. The
    label is merge-patched per download, so on a multi-model cache it describes only the last model written
    — which is why fleet called a sentinel-less claim "verified"."""
    if not isinstance(facts, dict) or facts.get("probe_error"):
        why = (facts or {}).get("probe_error") if isinstance(facts, dict) else "no probe result"
        return (STATE_UNKNOWN, f"could not read the cache: {why}")
    if not facts.get("exists"):
        return (STATE_ABSENT, "no snapshot directory at the server's resolution path")

    req = int(facts.get("required_shards", 0) or 0)
    missing = int(facts.get("missing_shards", 0) or 0)
    shards = int(facts.get("shard_count", 0) or 0)
    has_cfg = bool(facts.get("config_json"))
    has_index = bool(facts.get("index_json"))
    total = facts.get("index_total_bytes")
    on_disk = facts.get("shard_bytes")
    files = facts.get("shard_files")
    meta_total = int(facts.get("metadata_total_shards", 0) or 0)
    meta_files = int(facts.get("metadata_shard_files", 0) or 0)
    meta_totals = int(facts.get("metadata_distinct_totals", 0) or 0)

    # (a0) huggingface_hub's own partial marker. Conclusive, and it fires on the failure that actually
    # happens: a shard interrupted mid-write leaves blobs/<sha>.incomplete and no snapshots/ symlink.
    if int(facts.get("incomplete_files", 0) or 0) > 0:
        return (
            STATE_INCOMPLETE,
            f"{facts['incomplete_files']} .incomplete blob(s) — a download was interrupted mid-shard",
        )

    # (a) The index names a shard set and one of them does not resolve -> a partial download.
    if req > 0 and missing > 0:
        return (
            STATE_INCOMPLETE,
            f"{missing} of {req} index-referenced shards are missing or dangling — a partial download",
        )

    # Per-shard metadata may declare a sharded model even when the standard shard index is absent. Verify the
    # declared shard count before accepting a completion sentinel.
    if meta_totals > 1:
        return (
            STATE_INCOMPLETE,
            f"per-shard metadata declares {meta_totals} different shard totals — the snapshot is inconsistent",
        )
    if meta_total > 0 and (shards != meta_total or meta_files != meta_total):
        return (
            STATE_INCOMPLETE,
            f"per-shard metadata requires {meta_total} shards, but found {shards} safetensors and "
            f"{meta_files} metadata files — a partial download",
        )

    # A cache cannot be certified when shards exist but the size probe measured none of them. Report UNKNOWN
    # rather than INCOMPLETE because the files may be valid even though the measurement failed.
    if shards > 0 and not (isinstance(files, int) and files > 0):
        return (
            STATE_UNKNOWN,
            f"{shards} shard file(s) are present but NONE of them could be sized "
            f"(SHARD_N={'0' if files == 0 else 'unset'}) — the probe's file-size pass returned nothing, "
            f"so every byte check below is inoperative. This is 'we could not measure it', not "
            f"'the shards are empty'.",
        )

    # Reject implausible measurements before applying byte-based completeness checks. A probe that measures
    # symlink path lengths instead of target sizes can otherwise misclassify a valid cache.
    if (
        isinstance(total, int)
        and total > 0
        and facts.get("shard_files")
        and isinstance(facts.get("shard_max_bytes"), int)
        and facts["shard_max_bytes"] > 0
    ):
        _expected_mean = total / facts["shard_files"]
        if facts["shard_max_bytes"] * SIZE_PLAUSIBILITY_FACTOR < _expected_mean:
            return (
                STATE_UNKNOWN,
                f"the shard sizes measured are not credible — largest of {facts['shard_files']} shards "
                f"is {facts['shard_max_bytes']} B against an index-implied mean of "
                f"{int(_expected_mean)} B, so the MEASUREMENT is wrong rather than the data "
                f"(a probe that fails to dereference the hub's blob symlinks reports exactly this)",
            )

    # (a1) A ZERO-BYTE SHARD is conclusive: the file EXISTS, so the missing-set check in (a) passes, but it
    # holds nothing. Only visible now that sizes are read per file rather than summed.
    if isinstance(facts.get("shard_min_bytes"), int) and facts.get("shard_files") and facts["shard_min_bytes"] == 0:
        return (
            STATE_INCOMPLETE,
            f"at least one of {facts['shard_files']} shard files is 0 bytes — present but empty",
        )

    # Aggregate byte veto: metadata.total_size is tensor bytes, so an intact snapshot must have at least
    # that many bytes across its shard files. This can demote a verdict but cannot prove per-file integrity;
    # the standard index does not provide exact sizes for individual shards.
    if isinstance(total, int) and total > 0 and isinstance(on_disk, int) and on_disk > 0 and on_disk < total:
        _span = ""
        if isinstance(facts.get("shard_min_bytes"), int) and isinstance(facts.get("shard_max_bytes"), int):
            _span = (
                f"; smallest shard {facts['shard_min_bytes']} B vs largest "
                f"{facts['shard_max_bytes']} B across {facts.get('shard_files')} files"
            )
        return (
            STATE_INCOMPLETE,
            f"shards present but only {on_disk} of the index's declared {total} bytes are on disk "
            f"({total - on_disk} B short){_span}. The likely cause is a truncated shard; the other "
            f"explanation consistent with this evidence is an index whose metadata.total_size "
            f"OVER-declares (tied/shared tensors are counted once per name in weight_map but stored "
            f"once). Either way completeness is NOT proven, so this cache is not safe to serve from.",
        )

    # (c) COMPLETE — TWO DIFFERENT PROOFS, and they must not be confused for one another.
    #
    #   (c1) SHARDED, index read: every shard the index NAMES resolves. Set membership, not a count:
    #        "113 files present" never certifies; "the 113 files the index names all resolve" does.
    #   (c2) SINGLE-FILE: config.json + >=1 resolved shard AND **NO INDEX FILE ON DISK**.
    #
    # The `no index` half of (c2) used to be missing, and that made the branch a lie detector's blind spot:
    # `req == 0` was read as "single-file model" even when INDEX=1 — i.e. when the index file EXISTS and the
    # probe simply failed to enumerate it (a grep that matched nothing, an unreadable/oddly-formatted index,
    # a shell hiccup). The verdict was ('complete', 'config.json + 113 resolved shard(s), no shard index
    # (single-file model)') for a 113-shard model, i.e. a FALSE proof string about a sharded model whose
    # shard set was never checked. `index_json` was parsed by parse_cache_integrity_report and read by
    # nothing.
    #
    # The cost of that particular false COMPLETE is why it is a blocker rather than a wording nit: install
    # then calls stamp_download_sentinel on it, the sentinel is CONCLUSIVE evidence at (d), so ONE probe hiccup used to make a wrong answer persist across
    # install, preflight and fleet until another integrity probe corrected it, while the server crash-looped
    # on a missing shard. The downloader now always resumes/verifies before rewriting the sentinel.
    #
    # Index present but nothing enumerated from it -> fall through to PRESENT_UNVERIFIED (e): the weights are
    # there, but the evidence that would prove them complete could not be read.
    if has_cfg and shards > 0 and req > 0 and missing == 0:
        proof = f"all {req} index-referenced shards resolve"
        if isinstance(total, int) and total > 0 and isinstance(on_disk, int) and on_disk >= total:
            proof += f"; {on_disk} bytes on disk clears the index's declared {total} (aggregate check)"
        return (STATE_COMPLETE, proof)
    if has_cfg and meta_total > 0 and shards == meta_total and meta_files == meta_total:
        return (
            STATE_COMPLETE,
            f"all {meta_total} filename-declared shards and per-shard metadata files resolve",
        )
    if has_cfg and shards == 1 and req == 0 and not has_index and meta_total == 0:
        return (
            STATE_COMPLETE,
            "config.json + exactly one resolved shard, no shard index (single-file model)",
        )

    # (d) Our own download Job's completion sentinel — written only after snapshot_download returned, under
    # `set -e`, so its presence means THIS revision finished.
    if facts.get("sentinel"):
        return (STATE_COMPLETE, "completion sentinel written by the llmb download Job")

    # Without either an index, filename-declared set, or successful sentinel, multiple shards are present
    # but unproven. This check belongs after the sentinel: only concrete 41/113 metadata above invalidates a
    # stale stamp; a sentinel from a successful download remains valid for other repository layouts.
    if not has_index and shards > 1 and meta_total == 0:
        return (
            STATE_PRESENT_UNVERIFIED,
            f"{shards} shard files exist without a central index or complete per-shard metadata; "
            "this is a multi-file snapshot whose completeness is not otherwise provable",
        )

    # Weights exist, but no available evidence proves the snapshot complete. The caller decides whether to
    # verify and stamp it; preflight blocks unverified content.
    if shards > 0 or has_cfg:
        bits = []
        if not has_cfg:
            bits.append("no config.json")
        if shards == 0:
            bits.append("no resolved shards")
        if req == 0:
            # Say WHICH of the two it is. "there is no index" is a property of the model; "there is an index
            # and we could not read it" is a property of this probe run, and only the second is a reason to
            # look again. Collapsing them is what let an unread index pose as a single-file model.
            bits.append(
                "model.safetensors.index.json IS PRESENT but no shard could be enumerated from it "
                "— the completeness evidence could not be read"
                if has_index
                else "no readable shard index"
            )
        if not facts.get("sentinel"):
            bits.append("no completion sentinel")
        return (
            STATE_PRESENT_UNVERIFIED,
            f"weights are present ({shards} resolved shard(s)) but completeness is not provable: " + ", ".join(bits),
        )
    return (STATE_ABSENT, "snapshot directory exists but holds no weights")


def sentinel_worthy(facts: dict) -> tuple[bool, str]:
    """PURE — may THIS probe's evidence be written down as a PERMANENT completion sentinel? (ok, why-not).

    A VERDICT and a RECORD are not the same decision, and this is the asymmetry that makes the difference
    matter: `cache_completeness` is re-derived from the disk on every call, so a wrong COMPLETE costs one
    run and is corrected by the next probe. A sentinel is treated as strong evidence by
    `cache_completeness` at (d). The download Job no longer trusts it as a reason to skip fetching—it always
    resumes/verifies the immutable snapshot—but preflight must still fail closed on a bad probe stamp. So a
    sentinel written by a PROBE (as
    opposed to one written by a download that actually completed, under `set -e`) must require MORE
    evidence than the verdict does, not less.

    STRONG ENOUGH TO RECORD — exactly two shapes:
      * sharded: the shard index was READ, it names >=1 shard, all of them resolve, no .incomplete blob,
        and every shard was successfully SIZED (so the byte vetoes above were live, not silently inoperative);
      * single-file: config.json + EXACTLY 1 resolved sized shard AND no index file on disk at all.
      * filename-indexed sharded: every model-NNNNN-of-MMMMM.json and all M shard files resolve.

    NOT strong enough (each of these could previously have been stamped permanently):
      * an index file that exists but from which nothing could be enumerated (req == 0, index_json true);
      * a shard set whose per-file sizes never came back (the awk-zeros case);
      * COMPLETE reached only via an existing sentinel — there is nothing to add."""
    if not isinstance(facts, dict) or facts.get("probe_error"):
        return (False, "the probe did not return a usable reading")
    state, why = cache_completeness(facts)
    if state != STATE_COMPLETE:
        return (False, f"not proven complete ({state}: {why})")
    if int(facts.get("incomplete_files", 0) or 0) > 0:
        return (False, "an .incomplete blob is present")
    files = facts.get("shard_files")
    if not (isinstance(files, int) and files > 0):
        return (
            False,
            "no shard file was successfully sized, so the byte checks did not actually run",
        )
    req = int(facts.get("required_shards", 0) or 0)
    missing = int(facts.get("missing_shards", 0) or 0)
    if req > 0:
        if missing:
            return (False, f"{missing} of {req} index-referenced shards do not resolve")
        return (
            True,
            f"all {req} index-referenced shards resolve and all {files} were sized",
        )
    meta_total = int(facts.get("metadata_total_shards", 0) or 0)
    meta_files = int(facts.get("metadata_shard_files", 0) or 0)
    if meta_total > 0:
        if int(facts.get("metadata_distinct_totals", 0) or 0) != 1:
            return (False, "per-shard metadata declares an inconsistent shard total")
        if int(facts.get("shard_count", 0) or 0) != meta_total or meta_files != meta_total:
            return (
                False,
                f"only part of the filename-declared {meta_total}-shard set is present",
            )
        return (
            True,
            f"all {meta_total} filename-declared shards and metadata files resolve and were sized",
        )
    if facts.get("index_json"):
        return (
            False,
            "a shard index EXISTS but no shard could be enumerated from it — this is exactly the "
            "reading that must NOT become permanent",
        )
    if facts.get("config_json") and int(facts.get("shard_count", 0) or 0) == 1:
        return (
            True,
            "single-file model: config.json + exactly one sized shard file, no shard index",
        )
    return (False, "COMPLETE was not reached by a proof this run can vouch for")


# The accumulating per-model stamp the download Job writes: llmb.nvidia.com/model.<name>=<rev12>.
MODEL_STAMP_PREFIX = "llmb.nvidia.com/model."


def model_cache_slug(model: str) -> str:
    """PURE — the label-safe form of a model name, matching the download Job's PVC-stamp label filter."""
    return re.sub(r"[^a-z0-9]+", "-", (model or "").lower()).strip("-")


def model_from_stamp_key(label_key: str) -> str:
    """PURE — the model slug encoded in a per-model stamp key, or '' if the key is not one.

    Exists because the obvious `key.split(".", 1)[1]` is WRONG: the prefix itself contains dots, so
    `llmb.nvidia.com/model.nemotron-ultra-nvfp4` split on the first dot yields
    `nvidia.com/model.nemotron-ultra-nvfp4`. Every per-model stamp comparison silently failed to match,
    which meant the accumulating label added to fix the single-valued-stamp problem was never actually
    read. One helper, so the three call sites cannot each get it wrong differently."""
    if not label_key.startswith(MODEL_STAMP_PREFIX):
        return ""
    return model_cache_slug(label_key[len(MODEL_STAMP_PREFIX) :])


def find_misplaced_weights(model: str, resolved_claim: str, pvcs: list, revision: str = "") -> tuple[str, str]:
    """PURE — does some OTHER claim's stamp say it already holds this model's weights? Returns
    (other_claim, why), or ('', '') when nothing contradicts the resolution.

    `pvcs`: [{name, labels}] — the shape list_cache_candidates returns.

    This advisory compares existing model stamps with the resolved destination before a new download. It does
    not change routing; install and the fleet view share this function so they report the same contradiction.
    """
    if not model or not resolved_claim:
        return ("", "")
    slug = model_cache_slug(model)
    for pvc in pvcs or []:
        name = pvc.get("name") or ""
        if not name or name == resolved_claim:
            continue
        lbl = pvc.get("labels") or {}
        for lk, lv in lbl.items():
            if model_from_stamp_key(lk) == slug:
                return (name, f"PVC '{name}' is stamped {lk}={lv}")
        if model_cache_slug(lbl.get("llmb.nvidia.com/model-name") or "") == slug:
            return (
                name,
                f"PVC '{name}' is stamped model-name={lbl['llmb.nvidia.com/model-name']}",
            )
        lrev = lbl.get("llmb.nvidia.com/model-revision") or ""
        if revision and lrev and revision.startswith(lrev):
            return (name, f"PVC '{name}' is stamped model-revision={lrev[:12]}")
    return ("", "")


def cache_probe_script(subpath: str, repo: str, revision: str) -> str:
    """PURE — the busybox script that gathers the completeness evidence from the SERVER's exact resolution
    path. ONE script, so install's "should I download this?" and preflight's "may this run start?" read the
    same bytes off the same paths. Emits KEY=value lines for parse_cache_integrity_report.

    `find -L … -type f` and `test -e` FOLLOW symlinks, so a dangling blob symlink — the partial-download
    signature — counts as MISSING rather than present."""
    snap = snapshot_dir(subpath, repo, revision)
    org, _, name = (repo or "").partition("/")
    sp = (subpath or ".").strip() or "."
    repo_dir = f"{sp}/hub/models--{org}--{name}".replace("//", "/").lstrip("./")
    sent = sentinel_path(subpath, revision)
    return (
        f'd="/cache/{snap}"; c="/cache/{repo_dir}"; s="/cache/{sent}"\n'
        'printf "EXISTS=%s\\n" "$([ -d "$d" ] && echo 1 || echo 0)"\n'
        'printf "SENTINEL=%s\\n" "$([ -f "$s" ] && echo 1 || echo 0)"\n'
        'printf "CONFIG=%s\\n" "$([ -f "$d/config.json" ] && echo 1 || echo 0)"\n'
        'printf "INDEX=%s\\n" "$([ -f "$d/model.safetensors.index.json" ] && echo 1 || echo 0)"\n'
        'printf "SHARDS=%s\\n" "$(find -L "$d" -maxdepth 1 -name \'*.safetensors\' -type f 2>/dev/null | wc -l | tr -d \' \')"\n'
        # Large snapshots may use per-shard metadata rather than model.safetensors.index.json. Read the
        # declared total from model-NNNNN-of-MMMMM.json filenames without opening hundreds of large shards.
        'mt="$(find -L "$d" -maxdepth 1 -name \'model-*-of-*.json\' -type f 2>/dev/null'
        " | sed -n 's|.*/model-[0-9][0-9]*-of-\\([0-9][0-9]*\\)\\.json$|\\1|p' | sort -u)\"\n"
        'printf "META_FILES=%s\\n" "$(find -L "$d" -maxdepth 1 -name \'model-*-of-*.json\' -type f 2>/dev/null | wc -l | tr -d \' \')"\n'
        "printf \"META_TOTALS=%s\\n\" \"$(printf '%s\\n' \"$mt\" | sed '/^$/d' | wc -l | tr -d ' ')\"\n"
        'printf "META_TOTAL=%s\\n" "$(printf \'%s\\n\' "$mt" | sed -n \'1{s/^0*//;s/^$/0/;p;}\')"\n'
        'idx="$d/model.safetensors.index.json"; rq=0; ms=0\n'
        'if [ -f "$idx" ]; then for f in $(grep -oE \'"[^"]*\\.safetensors"\' "$idx" 2>/dev/null | tr -d \'"\' | sort -u); do rq=$((rq+1)); [ -e "$d/$f" ] || ms=$((ms+1)); done; fi\n'
        'printf "REQ_SHARDS=%s\\n" "$rq"\n'
        'printf "MISSING_SHARDS=%s\\n" "$ms"\n'
        # BYTES, not just counts. metadata.total_size is the summed tensor bytes; each .safetensors file is
        # header+tensors, so an INTACT snapshot satisfies sum(sizes) >= total_size while a TRUNCATED shard
        # fails it. This is what stops "113 files present" being mistaken for "the 113 expected files at
        # full size". It can only ever DEMOTE a verdict (see cache_completeness).
        'printf "INDEX_TOTAL_BYTES=%s\\n" "$(grep -oE \'"total_size"[[:space:]]*:[[:space:]]*[0-9]+\' "$idx" 2>/dev/null | grep -oE \'[0-9]+$\' | head -1)"\n'
        # Use stat rather than reading shard contents, and follow Hugging Face snapshot symlinks. Emit no
        # numeric size fields when no shard was measured so callers can distinguish missing data from zero.
        "find -L \"$d\" -maxdepth 1 -name '*.safetensors' -type f -exec stat -Lc %s {} + 2>/dev/null"
        ' | awk \'{s+=$1; if(m==""||$1<m)m=$1; if($1>M)M=$1; n++}'
        ' END{if(n>0) printf "SHARD_BYTES=%s\\nSHARD_MIN=%s\\nSHARD_MAX=%s\\nSHARD_N=%s\\n", s, m, M, n;'
        ' else printf "SHARD_BYTES=\\nSHARD_MIN=\\nSHARD_MAX=\\nSHARD_N=\\n"}\'\n'
        # A remaining .incomplete blob identifies an interrupted Hugging Face download.
        'printf "INCOMPLETE_FILES=%s\\n" "$(find "$c" -name \'*.incomplete\' -type f 2>/dev/null | wc -l | tr -d \' \')"\n'
        'REF="$(cat "$c/refs/main" 2>/dev/null | head -1 | tr -d \' \\n\')"\n'
        'printf "REF=%s\\n" "$REF"\n'
        'printf "REFSNAP=%s\\n" "$([ -n "$REF" ] && [ -d "$c/snapshots/$REF" ] && echo 1 || echo 0)"\n'
    )


def parse_cache_integrity_report(text: str, server_path: str = "") -> dict:
    """Parse cache-probe key-value output into completeness facts.

    Unparseable numbers remain unknown rather than being treated as zero.
    """
    kv: dict = {}
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if "=" in ln:
            k, v = ln.split("=", 1)
            kv[k.strip()] = v.strip()

    def _int(key):
        try:
            return int(kv.get(key, "") or "")
        except ValueError:
            return None

    ref = kv.get("REF", "")
    return {
        "exists": kv.get("EXISTS") == "1",
        "sentinel": kv.get("SENTINEL") == "1",
        "config_json": kv.get("CONFIG") == "1",
        "index_json": kv.get("INDEX") == "1",
        "shard_count": _int("SHARDS") or 0,
        "metadata_shard_files": _int("META_FILES") or 0,
        "metadata_distinct_totals": _int("META_TOTALS") or 0,
        "metadata_total_shards": _int("META_TOTAL") or 0,
        "required_shards": _int("REQ_SHARDS") or 0,
        "missing_shards": _int("MISSING_SHARDS") or 0,
        "incomplete_files": _int("INCOMPLETE_FILES") or 0,
        "shard_min_bytes": _int("SHARD_MIN"),
        "shard_max_bytes": _int("SHARD_MAX"),
        "shard_files": _int("SHARD_N"),
        "index_total_bytes": _int("INDEX_TOTAL_BYTES"),
        "shard_bytes": _int("SHARD_BYTES"),
        "refs_main": ref,
        "refs_consistent": (ref == "" or kv.get("REFSNAP") == "1"),
        "server_path": server_path,
    }


def classify_mounter_failure(phase: str, events_text: str, create_err: str = "") -> tuple[str, str]:
    """PURE — why did a cache-mounting pod fail? Returns (code, human_reason).

    codes: rbac-denied | unschedulable | mount-failed | image-pull | pending | timeout

    These are THREE DIFFERENT OPERATOR ACTIONS and they used to collapse into one silent `None`:
      rbac-denied    you cannot create pods here          -> ask for RBAC
      unschedulable  no node matches selector/taints      -> fix MODEL_CACHE_NODE_SELECTOR
      mount-failed   scheduled, but the volume won't mount-> wrong nodes selected, or the class is broken
    Reporting all three as "probe unavailable" is what made a 9-in-11 placement failure look like a broken
    cache."""
    low_err = (create_err or "").lower()
    if "forbidden" in low_err or "cannot create resource" in low_err or "unauthorized" in low_err:
        return (
            "rbac-denied",
            f"not allowed to create the probe pod: {(create_err or '').strip()[:160]}",
        )
    low = (events_text or "").lower()
    for sig, code, why in (
        (
            "failedscheduling",
            "unschedulable",
            "no node satisfies the pod's nodeSelector/taints — it never started",
        ),
        (
            "didn't match pod's node affinity",
            "unschedulable",
            "no node satisfies the pod's nodeSelector — it never started",
        ),
        (
            "failedmount",
            "mount-failed",
            "the pod was scheduled but the model-cache volume did not mount",
        ),
        (
            "rpc.statd",
            "mount-failed",
            "NFS mount refused on this node (rpc.statd not running) — this node cannot mount the claim",
        ),
        (
            "operation not permitted",
            "mount-failed",
            "the model-cache volume was refused on this node",
        ),
        (
            "failedattachvolume",
            "mount-failed",
            "the model-cache volume could not be attached to this node",
        ),
        ("errimagepull", "image-pull", "the probe image could not be pulled"),
        ("imagepullbackoff", "image-pull", "the probe image could not be pulled"),
    ):
        if sig in low:
            return (code, why)
    if (phase or "").strip() == "Pending":
        return ("pending", "the probe pod was still Pending when the wait expired")
    return ("timeout", "the probe pod did not become ready within the wait budget")


def parse_profile_env(path: Path) -> dict:
    """Read a cluster-profile .env into a dict, matching what `sh` actually sources.

    Three things this MUST get right, because deploy.sh sources the same file with `sh` and any disagreement
    is the download-vs-mount divergence all over again by a new route:
      * INLINE COMMENTS. `MODEL_CACHE_PVC="c"   # holds X` is the claim `c` to sh. Naive .strip('"') left
        the whole trailing comment glued to the value, producing a claim name no PVC will ever match.
      * LAST-WINS on a duplicated key, as sourcing does.
      * `export KEY=value`. Sourcing honours it — `. profile.env` with `export MODEL_CACHE_PVC="c"` sets
        MODEL_CACHE_PVC — while splitting on the first `=` produced a key literally named
        `export MODEL_CACHE_PVC`, so the claim read as UNRESOLVED and every consumer refused. deploy.sh used
        to `. "$ENVF"` and now also asks this parser, so the two must agree. `export` with no `=` (a bare
        re-export) assigns nothing and is skipped, exactly as sh treats it.
    Mirrors profile_resolver._read_env / preflight.parse_env deliberately — a third .env dialect would be a
    third opinion."""
    prof: dict = {}
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("export "):
            ln = ln[len("export ") :].lstrip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'"):
            q = v[0]
            j = v.find(q, 1)
            v = v[1:j] if j != -1 else v[1:]
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        prof[k.strip()] = v
    return prof


def _cli_resolve(argv: list[str]) -> int:
    """`model_cache.py resolve <cell-dir> <profile.env>` → the claim on stdout, or exit 2 + fix on stderr.

    An unresolved claim exits non-zero. Standard output contains only the resolved name; diagnostics use
    standard error."""
    if len(argv) != 2:
        print("usage: model_cache.py resolve <cell-dir> <profile.env>", file=sys.stderr)
        return 2
    cell_dir, envf = Path(argv[0]), Path(argv[1])
    cluster = envf.name[:-4] if envf.name.endswith(".env") else envf.name
    try:
        prof = parse_profile_env(envf)
    except Exception as e:
        print(f"model_cache: cannot read profile {envf}: {e}", file=sys.stderr)
        return 2
    try:
        import yaml

        recipe = yaml.safe_load((cell_dir / "recipe.yaml").read_text()) or {}
    except Exception as e:
        print(f"model_cache: cannot read {cell_dir}/recipe.yaml: {e}", file=sys.stderr)
        return 2
    env = recipe.get("envelope") or {}
    cell = {
        "model": env.get("model"),
        "requires": env.get("requires") or {},
        "name": env.get("name"),
        "_path": str(cell_dir),
    }
    name, _src = resolve_cache_claim(cell, prof)
    if not name:
        print(
            f"model_cache: no model-cache PVC for model '{env.get('model') or '?'}' — "
            f"{cache_claim_fix_hint(cell, cluster)}",
            file=sys.stderr,
        )
        return 2
    print(name)
    return 0


def _cli_node_selector(argv: list[str]) -> int:
    """Render a canonical inline-YAML selector.

    Invalid non-empty selectors fail closed; an empty selector means no placement constraint.
    """
    spec = argv[0] if argv else ""
    sel = parse_node_selector(spec)
    if not sel:
        if (spec or "").strip():
            print(
                f"model_cache: MODEL_CACHE_NODE_SELECTOR={spec!r} parses to NO selector — refusing to "
                f"place a cache-mounting pod without the constraint it asks for. Expected "
                f"'key: value' pairs, e.g. 'nvidia.com/gpu.present: \"true\"'.",
                file=sys.stderr,
            )
            return 2
        return 0
    print(", ".join(f'{k}: "{v}"' for k, v in sel.items()))
    return 0


def main(argv: list[str] | None = None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if a and a[0] == "resolve":
        return _cli_resolve(a[1:])
    if a and a[0] == "node-selector":
        return _cli_node_selector(a[1:])
    print(__doc__.strip().splitlines()[0], file=sys.stderr)
    print(
        "usage: model_cache.py resolve <cell-dir> <profile.env>\n"
        "       model_cache.py node-selector '<MODEL_CACHE_NODE_SELECTOR>'",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

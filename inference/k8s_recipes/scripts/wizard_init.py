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

"""wizard_init.py — the `llmb-k8s init` fresh-install wizard (the wizard contract, Phase-1).

An ORCHESTRATOR, not a reimplementation: it composes the shipped k8s building blocks
(profile_init, profile_resolver, cluster_readiness, capability_registry, manifest_vars, k8s_config)
behind one S-tier flow — preamble → collect → confirm → provision → done — that ends only after a
live no-GPU readiness proof says the written profile will survive a real run.

Modes (the wizard contract §7):
  interactive (default)  full §4 walk-through
  --play <f>             headless: load a playfile, write the profile non-interactively, run the Done
                         battery, exit non-zero on any ❌. NEVER provisions a live PVC (provision-now
                         is interactive-only, §6.2/S6).
  --record <f>           STUBBED (Phase-1): NotImplemented + TODO.
  --express              STUBBED (Phase-1): NotImplemented + TODO.

Flags mirror SLURM where semantics match (§2): -v/--verbose; -d/--dev-mode (near-no-op for k8s — there
is no repo copy to skip; accepted for surface-parity, NOT a dry-run); --dry-run/-n (the touch-nothing
mode → maps onto profile_init's dry-run: detect + render, write nothing live). `-i` is dropped.

Exit codes (§2, mirrors cli/llmb-install/constants): 0 success, 1 error/blocker, 130 cancelled.

PHASE-1 no-internet policy (§6.1, NEW-1): NO_INTERNET_DNS_IP and NO_INTERNET_KUBE_API_IP are written as a
SINGLE proven IP each (kube-dns ClusterIP; apiserver endpoint IP), surfaced + confirmed SYMMETRICALLY, both
carrying the same Phase-2 limitation. A comma-list is NEVER emitted (the runner parses each via
ipaddress.ip_address() and fail-closes on a comma). Phase-2 allowlist-both is gated on a runner change.

The pure functions (render_done_panel, rank_rwx_classes, model_cache_check, valid_single_ip, pvc_manifest,
build_profile_text, load_playfile) are unit-tested offline in selftest_wizard_init.py.
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import profile_resolver as pr  # noqa: E402
import profile_init as pi  # noqa: E402
import cluster_readiness as cr  # noqa: E402
import capability_registry as cap  # noqa: E402
import runner_identity as ri  # noqa: E402 — WHO ran it (run_by), persisted user-level once here

PROFILES_DIR = pr.PROFILES_DIR

# Exit codes (mirror cli/llmb-install/src/llmb_install/constants.py).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CANCELLED = 130

PLAYFILE_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# The NEW init-only Done renderer (the wizard contract §4/§8, NEW-2)
# Consumes the STRUCTURED Check list from cluster_readiness.run_battery — NOT a string-remap of
# format_battery output. id→label column, level→wizard-glyph, first-❌ truncation keyed on Check.level.
# ─────────────────────────────────────────────────────────────────────────────

# level → wizard glyph (the house "No 🟡" vocab). PASS→✓ (native ✅), WARN→⚠ (native 🟡), FAIL→❌, SKIP→·.
DONE_GLYPH = {cr.PASS: "✓", cr.WARN: "⚠", cr.FAIL: "❌", cr.SKIP: "·"}

# Check.id → short check-name column label for the Done panel.
DONE_LABELS = {
    "reachability": "reachability",
    "gpu-nodes": "nodes",
    "pull-secret": "pull-secret",
    "staging-roundtrip": "artifacts SC",
    "control-rwx": "control RWX",
    "model-cache": "model-cache",
    "var-reconcile": "var-reconcile",
    "oidc-issuer": "oidc-issuer",
    "fast": "fast-skip",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-name validation (the wizard contract §5 — the profile name is the on-disk filename)
# ─────────────────────────────────────────────────────────────────────────────

import re as _re  # noqa: E402

# RFC-1123 label: lowercase alphanumeric + '-', start/end alphanumeric, ≤63 chars. The profile name becomes
# cluster-profiles/<name>.env, so an unvalidated name is a path-traversal write (`../evil` escapes the dir).
_CLUSTER_NAME_RE = _re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def validate_cluster_name(name: str) -> Optional[str]:
    """PURE. Return an error message if `name` is not a safe cluster/profile name, else None. Enforces an
    RFC-1123 label ≤63 chars — rejecting '/', '..', spaces, uppercase, and empty up front (closes the
    path-traversal write and a silent bad-name). None = OK."""
    n = (name or "").strip()
    if not n:
        return "profile name is empty — run `llmb-k8s init` (pick from connected clusters) or pass --cluster <label>"
    if len(n) > 63:
        return f"profile name too long ({len(n)} chars) — must be ≤63 (RFC-1123 label)"
    if not _CLUSTER_NAME_RE.match(n):
        return (
            f"invalid profile name {n!r} — use a lowercase RFC-1123 label (a-z, 0-9, '-'; start/end "
            "alphanumeric; no '/', '..', spaces, or uppercase)"
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cluster DISCOVERY front door (no-flag `llmb-k8s init`) — the wizard contract §4 Preamble/Collect
# A fresh operator cannot be expected to KNOW the profile label to type into `--cluster`. So a no-flag init
# DISCOVERS what kubectl is connected to (contexts), presents them (full names, current-context marked), and
# lets the operator pick — then derives + confirms a sensible PROFILE NAME from the pick. The primitives
# (pi.list_contexts) are the same mockable runner the rest of the wizard uses → fully offline-testable.
# ─────────────────────────────────────────────────────────────────────────────


def slugify_context(name: str) -> str:
    """PURE. Derive a short, RFC-1123-safe DEFAULT PROFILE NAME (label) from a kube context name. Context
    names are long / structured (arn:aws:eks:…:cluster/qwen3-qa, proxy.example.teleport.sh-…-qwen3-qa), so:
    take the segment after the last '/' or ':' (EKS-arn style → the bare cluster name), lowercase, replace
    every run of non [a-z0-9] with a single '-', strip leading/trailing '-', clamp to 63. Falls back to
    'cluster' if nothing survives. This is only a SUGGESTION — the operator accepts or overrides it.
    """
    raw = (name or "").strip()
    # Teleport proxy contexts prefix the cluster id with the FULL proxy hostname:
    #   proxy.example.teleport.sh-<clusterid>
    # Slugifying the whole thing yields an absurd 60+ char label (the QA bug). Strip everything through
    # '.teleport.sh-' so the default label is the bare cluster-id token (example-cluster-dgxc-…), still
    # user-overridable. Case-insensitive (Teleport hostnames are lowercase, but be defensive).
    _tele = _re.split(r"\.teleport\.sh-", raw, maxsplit=1, flags=_re.IGNORECASE)
    if len(_tele) == 2 and _tele[1].strip():
        raw = _tele[1]
    # EKS-arn / path-style contexts carry the clean cluster name after the last '/' or ':'.
    for sep in ("/", ":"):
        if sep in raw:
            raw = raw.rsplit(sep, 1)[-1]
    low = raw.lower()
    slug = _re.sub(r"[^a-z0-9]+", "-", low).strip("-")
    slug = slug[:63].strip("-")
    return slug or "cluster"


CONTEXT_MENU_LIMIT = 8  # how many contexts to show before collapsing (a kubeconfig can hold hundreds)


def render_context_menu(
    contexts: list,
    *,
    limit: int = CONTEXT_MENU_LIMIT,
    show_all: bool = False,
    query: str = "",
) -> str:
    """PURE. Render the connected-clusters picker: a numbered menu of kubectl contexts by FULL name, the
    current-context marked with '(current)' so the operator recognizes the one they just logged into. Empty
    list → a single 'no connected clusters' line (the caller turns that into a login hint).

    Keep long kubeconfig context lists compact and searchable; unbounded output is difficult to use.
    a flat wall that buries the one you want — and the CURRENT one could sit at #192. So:
      * the CURRENT context is listed FIRST and marked as the Enter-default, and
      * the rest are collapsed to `limit` entries with a 'show all / type to filter' hint.
    The NUMBERS are the original 1-based positions in `contexts` and never change with filtering or
    collapsing, so the caller's default index and any number the operator types keep working — the default
    is presentation-only and cannot regress.
    """
    if not contexts:
        return "  (no connected clusters — kubectl has no contexts)"
    q = (query or "").strip().lower()
    idxed = list(enumerate(contexts, 1))  # (original_index, ctx) — index is STABLE
    if q:
        idxed = [(i, c) for i, c in idxed if q in str(c.get("name", "")).lower()]
    # current first, then the rest in their original order
    cur = [(i, c) for i, c in idxed if c.get("current")]
    rest = [(i, c) for i, c in idxed if not c.get("current")]
    ordered = cur + rest

    header = "  Connected clusters (kubectl contexts):"
    if q:
        header = f"  Connected clusters matching {query!r}  ({len(ordered)} of {len(contexts)}):"
    if not ordered:
        return header + f"\n    (no context matches {query!r} — type another filter, or 'a' to show all)"

    shown = ordered if (show_all or q or len(ordered) <= limit) else ordered[:limit]
    width = len(str(len(contexts)))
    lines = [header]
    for i, c in shown:
        mark = "  (current — press Enter)" if c.get("current") else ""
        lines.append(f"    {i:>{width}}  {c.get('name', '')}{mark}")
    hidden = len(ordered) - len(shown)
    if hidden > 0:
        lines.append(
            f"    … {hidden} more not shown." "  Type 'a' to list them all, or type text to filter (e.g. 'gb300')."
        )
    return "\n".join(lines)


def _reconnect_hint(context: str = "") -> str:
    """The RECONNECT guidance shown when the operator doesn't see the cluster they expect (or the list is
    empty). Reuses the shared connect_hint copy so the wording can't drift from fleet/preflight — a derived
    `tsh kube login <ctx>` when a context is known, else the generic SSO/Teleport/VPN login instruction.
    """
    return pr.connect_hint({}, context)


# Picker sentinels — distinguish an empty context list (error) from an operator quit (cancelled).
_PICK_EMPTY = "__pick_empty__"
_PICK_QUIT = "__pick_quit__"


def _pick_context_interactive(krun=pi.default_krun):
    """INTERACTIVE cluster-discovery for a no-flag `llmb-k8s init`. Lists the CONNECTED kubectl contexts
    (full names, current marked) as a numbered menu, lets the operator pick one, then derives + confirms a
    PROFILE NAME (label) for the pick. Returns (profile_name, context) on success, _PICK_EMPTY when there
    are no contexts (→ login hint), or _PICK_QUIT on an explicit quit. Pure I/O (input()/print) over the
    injected krun → offline-testable with the same _FakeKrun + monkeypatched input as run_interactive.
    """
    contexts = pi.list_contexts(krun)
    print("\n── Connect ────────────────────────────────────────────────────────")
    if not contexts:
        print("  ❌ no connected clusters — kubectl has no contexts to choose from.")
        print(f"     Log in first, then re-run `llmb-k8s init`:\n       {_reconnect_hint()}")
        return _PICK_EMPTY
    show_all, query = False, ""
    print(render_context_menu(contexts, show_all=show_all, query=query))
    print("\n  Don't see the cluster you expect?  It isn't connected yet — reconnect and re-run init:")
    print(f"     {_reconnect_hint()}")
    default_idx = next((i for i, c in enumerate(contexts, 1) if c.get("current")), 1)
    while True:
        raw = input(
            f"\n? Pick a cluster [1-{len(contexts)}]  (a = show all, text = filter, "
            f"r = reconnect help, q = quit) [{default_idx}]: "
        ).strip()
        low = raw.lower()
        if low in ("q", "quit"):
            return _PICK_QUIT
        if low in ("r", "reconnect"):
            print(f"  Reconnect with your cluster's login, then re-run `llmb-k8s init`:\n     {_reconnect_hint()}")
            continue
        # 'a' expands the collapsed list; any other non-numeric text filters it. Both only change what is
        # DISPLAYED — the numbering and the Enter-default are untouched.
        if low in ("a", "all"):
            show_all, query = True, ""
            print(render_context_menu(contexts, show_all=True, query=""))
            continue
        if raw and not raw.isdigit():
            query = raw
            print(render_context_menu(contexts, show_all=show_all, query=query))
            continue
        if not raw:
            raw = str(default_idx)
        if raw.isdigit() and 1 <= int(raw) <= len(contexts):
            chosen = contexts[int(raw) - 1]["name"]
            break
        print(f"    Enter 1–{len(contexts)}, a, text to filter, r, or q.")
    # Derive + confirm the PROFILE NAME (label) for the chosen cluster. Clarify it is a name for THIS
    # profile/lane, NOT a cluster the operator must already know.
    print(f"\n  Selected cluster (context): {chosen}")
    suggested = slugify_context(chosen)
    while True:
        name = pi._prompt("Profile name (a label for THIS cluster profile — you choose it)", suggested)
        err = validate_cluster_name(name)
        if not err:
            return name, chosen
        print(f"    ⚠ {err}")


# Profile env-key → Answers-field mapping (identity keys are locked on resume; the rest are editable defaults).
_ENV_TO_FIELD = {
    "KUBE_CONTEXT": "context",
    "NAMESPACE": "namespace",
    "OWNER": "owner",
    "CONNECT_CMD": "connect_cmd",
    "GPU_PRODUCT": "gpu_product",
    "GPU_PER_NODE": "gpu_per_node",
    "CPU_PER_NODE": "cpu_per_node",
    "MEM_PER_NODE": "mem_per_node",
    "WHOLE_NODE_CPU": "whole_node_cpu",
    "WHOLE_NODE_MEM": "whole_node_mem",
    "ARCH": "arch",
    "IMAGE_PULL_SECRET": "pull_secret",
    "HF_SECRET": "hf_secret",
    "MODEL_CACHE_PVC": "model_cache_pvc",
    "MODEL_CACHE_RWX_CLASS": "cache_rwx_class",
    "MODEL_CACHE_RWO_CLASS": "cache_rwo_class",
    "ARTIFACTS_STORAGE_CLASS": "artifacts_sc",
    "CONTROL_STORAGE_CLASS": "control_sc",
    "SCHEDULER_NAME": "scheduler",
    "NO_INTERNET_DNS_IP": "no_internet_dns_ip",
    "NO_INTERNET_KUBE_API_IP": "no_internet_kube_api_ip",
}
# Identity (🔒) — changing any of these means a DIFFERENT cluster, so it's a new profile, not an edit.
# (name is the filename itself; context + GPU_PRODUCT are hard-locked from the existing profile on resume.)
_IDENTITY_ENV_KEYS = ("KUBE_CONTEXT", "GPU_PRODUCT")


def _fill_answers_from_env(a: "Answers", env: dict, *, only_missing: bool = True) -> None:
    """Populate Answers fields from an existing profile dict (profile_resolver._read_env output). With
    only_missing=True (the resume default) an already-set answer wins over the stored value, so a playfile /
    live detection can still refine a non-identity field; identity fields are handled separately by the lock.
    """
    for key, field in _ENV_TO_FIELD.items():
        val = (env.get(key) or "").strip()
        if not val:
            continue
        if only_missing and str(getattr(a, field, "") or "").strip():
            continue
        setattr(a, field, val)


def _enforce_identity_lock(a: "Answers", env: dict) -> Optional[str]:
    """HARD identity lock on resume/edit (the wizard contract §5). If the existing profile pins KUBE_CONTEXT /
    GPU_PRODUCT, they cannot change — a mismatch is a DIFFERENT profile. Returns an error message on a
    refused change, else None (and pins the locked value onto `a`)."""
    field_of = {"KUBE_CONTEXT": "context", "GPU_PRODUCT": "gpu_product"}
    for key in _IDENTITY_ENV_KEYS:
        old = (env.get(key) or "").strip()
        if not old:
            continue
        field = field_of[key]
        new = str(getattr(a, field, "") or "").strip()
        if new and new != old:
            return (
                f"{key} is locked 🔒 on the existing profile '{a.cluster}' ({old!r} → {new!r} refused) — "
                "that's a DIFFERENT cluster. Pick a new profile name."
            )
        setattr(a, field, old)  # pin the locked identity value
    return None


def render_done_panel(cluster: str, checks: list) -> str:
    """PURE. Render the readiness battery in the wizard vocab with first-❌ truncation.

    Works from Check.level (the structured field), not a glyph substring — so a FAIL truncates the
    presentation robustly. On a FAIL: print the failing check, its one-line fix, a stop notice, and STOP
    (later checks are computed but not shown, §6.3). With no FAIL: print every line + the verdict summary.
    """
    ok, summary = cr.verdict(checks)
    bar = "─" * max(4, 40 - len(cluster))
    out = [f"── Done: run-ready proof (no GPU) — {cluster} {bar}"]
    truncated = False
    for c in checks:
        label = DONE_LABELS.get(c.id, c.id)
        glyph = DONE_GLYPH.get(c.level, "?")
        out.append(f"  {glyph} {label:<13} {c.message}")
        if c.level == cr.FAIL:
            if c.fix:
                out.append(f"       fix: {c.fix}")
            out.append("  (Stopping here — later checks are not shown until this is fixed.)")
            truncated = True
            break
    if not truncated:
        head = f"Cluster {cluster} is RUN-READY ✓" if ok else f"Cluster {cluster} is NOT run-ready"
        out.append(f"  → {head}   ({summary})")
        if ok:
            out.append(f"  Next:  llmb-k8s preflight <cell> {cluster}")
    else:
        out.append(f"  Profile was written; it is NOT yet run-ready.  ({summary})")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# NEW RWX storage-class detector (the wizard contract §6.4 RWX-rank)
# discover_storage_classes returns NAMES only (no access-mode), so this cannot PROVE RWX offline. Phase-1
# ranks candidate names by substring and returns the top as a ⚠ hint requiring confirmation; safe-degrades
# to None when nothing matches (→ the field is left for the operator, never a fabricated value).
# ─────────────────────────────────────────────────────────────────────────────

_RWX_HINTS = (
    "rwx",
    "efs",
    "nfs",
    "filestore",
    "shared",
    "cephfs",
    # high-throughput PARALLEL filesystems — the RIGHT default for large model weights (a shared
    # Prefer a high-throughput shared class for large, read-heavy model caches. The shared-filesystem class
    # on these clusters is named `fsx-lustre` (no 'fsx' substring), so match the name too.
    "fsx",
    "lustre",
    "enterprise-file",
)

# ─────────────────────────────────────────────────────────────────────────────
# Provisioner-aware storage-class defaulting (QA storage-default fix)
# discover_storage_classes_detailed returns {name, provisioner, is_default}. Name substrings alone are
# NOT enough: `s3-object` matches the "standard" RWO hint yet is an s3.csi.aws.com OBJECT store
# that must NEVER back a block/file PVC. So an object/S3 provisioner is EXCLUDED from both the RWO and RWX
# candidate sets; RWO prefers the cluster is-default-class then a block provisioner; RWX prefers a FILE
# provisioner (fsx/efs/nfs). All pure → offline-testable against a mocked storageclass list.
# ─────────────────────────────────────────────────────────────────────────────

# Provisioners that are OBJECT stores (buckets) — invalid as a block/file PVC backend, excluded from both sets.
_OBJECT_PROVISIONERS = ("s3.csi", "blob.csi", "gcs", "objectstorage")
# FILE (ReadWriteMany-capable shared filesystem) provisioners — the RWX control-plane class comes from here.
_FILE_PROVISIONERS = ("fsx.csi", "efs.csi", "filestore.csi", "azurefile", "nfs")
# BLOCK (RWO single-writer) provisioners — the artifacts / model-cache class comes from here.
_BLOCK_PROVISIONERS = (
    "ebs.csi",
    "aws-ebs",
    "pd.csi",
    "gce-pd",
    "disk.csi",
    "azuredisk",
    "cinder",
)
# Name-substring fallbacks for a class whose provisioner is unknown (e.g. a text-only / mocked list).
_RWO_NAME_HINTS = ("ebs", "gp", "ssd", "block", "standard", "managed")


def _sc_provisioner(sc: dict) -> str:
    return (sc.get("provisioner") or "").lower()


def _is_object_sc(sc: dict) -> bool:
    """True iff the class is an OBJECT/S3 store (by provisioner, then an -object name fallback)."""
    prov = _sc_provisioner(sc)
    if prov:
        return any(o in prov for o in _OBJECT_PROVISIONERS)
    return "object" in (sc.get("name") or "").lower()  # unknown provisioner → last-resort name fallback


def _is_file_sc(sc: dict) -> bool:
    return any(f in _sc_provisioner(sc) for f in _FILE_PROVISIONERS)


def _is_block_sc(sc: dict) -> bool:
    return any(b in _sc_provisioner(sc) for b in _BLOCK_PROVISIONERS)


def select_rwo_class(classes: list) -> Optional[str]:
    """PURE. Pick the DEFAULT RWO (block) class for artifacts + the model-cache PVC from the detailed list
    ({name,provisioner,is_default}). NEVER an object/S3 provisioner. Preference order:
      1. the cluster is-default-class annotation (when it is not an object store);
      2. a BLOCK provisioner (ebs/pd/disk/…);
      3. a name-hint (ebs/gp/ssd/standard/…) among the remaining non-object classes;
      4. else the first non-object class.
    None only when EVERY class is an object store (the caller leaves it for the operator).
    """
    cands = [c for c in (classes or []) if c.get("name") and not _is_object_sc(c)]
    if not cands:
        return None
    for c in cands:
        if c.get("is_default"):
            return c["name"]
    for c in cands:
        if _is_block_sc(c):
            return c["name"]
    for c in cands:
        if any(h in c["name"].lower() for h in _RWO_NAME_HINTS):
            return c["name"]
    return cands[0]["name"]


def select_rwx_class(classes: list) -> Optional[str]:
    """PURE. Pick the DEFAULT RWX (shared file) class for the control PVC from the detailed list. NEVER an
    object/S3 provisioner. Preference: a FILE provisioner (fsx/efs/filestore/nfs/azurefile), else a name-hint
    (efs/nfs/file/shared/rwx/cephfs/filestore) among non-object classes. None when no file/shared class
    exists — the caller then falls back to the RWO class for single-pod caching (never an object store).
    """
    cands = [c for c in (classes or []) if c.get("name") and not _is_object_sc(c)]
    for c in cands:
        if _is_file_sc(c):
            return c["name"]
    for c in cands:
        low = c["name"].lower()
        if any(h in low for h in _RWX_HINTS) or "file" in low:
            return c["name"]
    return None


def rank_rwx_classes(names: list) -> tuple[Optional[str], list]:
    """PURE. Rank storage-class names by RWX-likelihood via the _RWX_HINTS substrings (case-insensitive).
    Returns (best_or_None, ranked_candidates). best is a ⚠ HINT requiring confirmation, never a proven ✓;
    None when nothing matches (safe-degrade — the caller leaves CONTROL_STORAGE_CLASS for the operator).
    """
    ranked = []
    for n in names or []:
        low = (n or "").lower()
        score = sum(1 for h in _RWX_HINTS if h in low)
        if score:
            ranked.append((score, n))
    ranked.sort(key=lambda t: (-t[0], t[1]))
    ordered = [n for _, n in ranked]
    return (ordered[0] if ordered else None), ordered


# ─────────────────────────────────────────────────────────────────────────────
# Model-cache hard-fail gate (the wizard contract §6.2, S1 — BOTH silent cases)
# ─────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER_PVC = "shared-model-cache"
_CACHE_NAME_HINTS = ("model", "cache", "weights")

# ─────────────────────────────────────────────────────────────────────────────
# Secrets — NAME auto-default + credential-SOURCE detection (Collect). The k8s Secret NAME is namespace-scoped
# boilerplate, so the wizard NEVER asks for it — it auto-defaults <profile>-nvcr-cred / <profile>-hf-token and
# `install` creates the Secret from the operator's local creds. What actually matters (and what a fresh user
# gets stuck on) is whether a CREDENTIAL SOURCE exists. So init CHECKS the source: present → a quiet ✓ and move
# on (zero friction); MISSING → the exact where-to-get instruction. Every recipe pulls nvcr.io images, so the
# pull secret is universal. Detection is PURE (environ/home injected) → offline-testable.
# ─────────────────────────────────────────────────────────────────────────────


def default_pull_secret_name(cluster: str = "") -> str:
    """PURE. The auto-assigned image-pull Secret NAME for a profile (never prompted). nvcr.io-scoped.

    BARE, namespace-scoped suffix (QA fix): these Secrets/PVCs are NAMESPACE-scoped, so a `<profile>-`
    prefix adds nothing but length — and when the profile label is an auto-derived teleport hostname it
    produced absurd 60+ char names. The default is the bare suffix; a user (or an existing profile) can
    still override it. `cluster` is accepted for call-site compatibility and intentionally unused.
    """
    return "nvcr-cred"


def default_hf_secret_name(cluster: str = "") -> str:
    """PURE. The auto-assigned HuggingFace-token Secret NAME for a profile (never prompted). BARE,
    namespace-scoped suffix (QA fix — see default_pull_secret_name). `cluster` accepted but unused.
    """
    return "hf-token"


def default_model_cache_name(cluster: str = "") -> str:
    """PURE. The DEFAULT model-cache PVC NAME to create at stage time. init runs BEFORE recipe selection, so
    it can't know the cell's cache size/class — install.py's ensure_model_cache_pvc (G3) creates this PVC
    correctly sized + storage-classed from the profile/recipe at stage time. This is the deferred default,
    NOT an eager RWX/100Gi provision. BARE, namespace-scoped suffix (QA fix — the PVC is namespace-scoped,
    so no `<profile>-` prefix). `cluster` accepted for call-site compatibility but unused.
    """
    return "model-cache"


_NGC_PLACEHOLDERS = {"", "no-apikey", "none", "<your key>", "your-key"}


def detect_ngc_cred(environ: Optional[dict] = None, home: Optional[Path] = None) -> tuple[bool, str]:
    """PURE (env/home injected). Is an NGC credential available for `install` to build the pull Secret?
    Checks $NGC_API_KEY / $NGC_CLI_API_KEY, then a REAL apikey in ~/.ngc/config. Returns (found, source).
    """
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    for var in ("NGC_API_KEY", "NGC_CLI_API_KEY"):
        if (environ.get(var) or "").strip():
            return True, f"${var}"
    cfg = home / ".ngc" / "config"
    try:
        text = cfg.read_text()
    except (FileNotFoundError, OSError):
        text = ""
    m = _re.search(r"(?im)^\s*apikey\s*=\s*(\S+)", text)
    if m and m.group(1).strip().lower() not in _NGC_PLACEHOLDERS:
        return True, "~/.ngc/config"
    return False, ""


def detect_hf_cred(environ: Optional[dict] = None, home: Optional[Path] = None) -> tuple[bool, str]:
    """PURE (env/home injected). Is a HuggingFace token available for `install` to build the HF Secret?
    Checks $HF_TOKEN / $HUGGING_FACE_HUB_TOKEN, then a non-empty ~/.cache/huggingface/token. (found, source).
    """
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if (environ.get(var) or "").strip():
            return True, f"${var}"
    tok = home / ".cache" / "huggingface" / "token"
    try:
        if tok.read_text().strip():
            return True, "~/.cache/huggingface/token"
    except (FileNotFoundError, OSError):
        pass
    return False, ""


_NGC_MISSING_MSG = (
    "⚠ No NGC credential found (checked $NGC_API_KEY and ~/.ngc/config). Every recipe pulls images from "
    "nvcr.io, so `install` needs one:\n"
    "       Get a key at ngc.nvidia.com → Setup → Generate API Key, then run `ngc config set` (writes "
    "~/.ngc/config) or `export NGC_API_KEY=<key>`, and re-run."
)
_HF_MISSING_MSG = (
    "⚠ No HuggingFace token found (checked $HF_TOKEN and ~/.cache/huggingface/token). `install` needs one to "
    "download gated model weights:\n"
    "       Get a token at huggingface.co → Settings → Access Tokens, then save it to "
    "~/.cache/huggingface/token or `export HF_TOKEN=<token>`, and re-run."
)


def _report_cred_sources(environ: Optional[dict] = None, home: Optional[Path] = None) -> tuple[bool, bool]:
    """Print the Secrets section: quiet ✓ when a credential SOURCE is present, the exact where-to-get
    instruction when it is MISSING. Non-blocking (init still writes the profile; install re-checks). Returns
    (ngc_found, hf_found). environ/home injected for offline tests."""
    print("Secrets  (Secret names auto-assigned; `install` creates them from your local creds)")
    ngc_ok, ngc_src = detect_ngc_cred(environ, home)
    if ngc_ok:
        print(f"  ✓ NGC creds found ({ngc_src}) — nvcr.io image pulls are covered")
    else:
        print("  " + _NGC_MISSING_MSG)
    hf_ok, hf_src = detect_hf_cred(environ, home)
    if hf_ok:
        print(f"  ✓ HuggingFace token found ({hf_src})")
    else:
        print("  " + _HF_MISSING_MSG)
    return ngc_ok, hf_ok


def model_cache_check(chosen: str, pvcs: list) -> cr.Check:
    """PURE. Classify the chosen MODEL_CACHE_PVC against the live PVC list — catching BOTH silent cases in
    profile_init.py:628's `cache_candidates[0] if cache_candidates else (pvcs[0] if pvcs else placeholder)`:

      - not in the namespace (incl. the literal placeholder) → hard ❌ (a run starts against a cache that
        does not exist and dies after GPU allocation). The §4 three-way fix card is offered interactively.
      - present but matched by NAME heuristic (*model/*cache/*weights*) → ✓.
      - present but matched by POSITION (pvcs[0], no name hit) → ⚠ "matched by position, not by name —
        confirm this is really the model cache" (never a silent ✓).
    """
    chosen = (chosen or "").strip()
    names = [p for p in (pvcs or [])]
    if not chosen or chosen not in names:
        detail = (
            f'MODEL_CACHE_PVC "{chosen or _PLACEHOLDER_PVC}" does not exist in the namespace — a run '
            "WILL start against a non-existent cache and fail after GPU allocation"
        )
        fix = (
            "pick an existing RWX PVC, provision one now (interactive), or name a PVC to create at stage "
            "time; candidates: " + (", ".join(names) if names else "(none in namespace)")
        )
        return cr.Check("model-cache", cr.FAIL, detail, fix=fix)
    if any(h in chosen.lower() for h in _CACHE_NAME_HINTS):
        return cr.Check(
            "model-cache",
            cr.PASS,
            f'model cache PVC "{chosen}" exists (matched by name)',
        )
    return cr.Check(
        "model-cache",
        cr.WARN,
        f'model cache PVC "{chosen}" exists but matched by POSITION, not by name — confirm this '
        "is really the model cache",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-IP writing (the wizard contract §6.1, Phase-1) — never a comma-list
# ─────────────────────────────────────────────────────────────────────────────


def valid_single_ip(value: str) -> bool:
    """PURE. True iff `value` is exactly ONE valid IP address. A comma (a Phase-2 allowlist-both list) or
    any unparseable form → False. Mirrors the runner's ipaddress.ip_address() single-IP, fail-closed
    parse — Phase-1 must never write a comma-list."""
    v = (value or "").strip()
    if not v or "," in v:
        return False
    try:
        ipaddress.ip_address(v)
        return True
    except ValueError:
        return False


def render_no_internet_block(dns_ip: str, kube_api_ip: str) -> str:
    """The no-internet lane IP block appended to the profile (Phase-1 SINGLE IP each, DNS/API symmetric).
    Empty values are written as-is (only no-internet lanes reference them; manifest_vars.reconcile hard-❌s
    an empty referenced var later only if a chosen recipe uses it)."""
    return (
        "\n# --- no-internet lane IPs (auto-discovered; Phase-1 SINGLE IP each — DNS/API symmetric) ---\n"
        "# Phase-2: allowlist-both — emit a comma-list of the .1 kube-API ClusterIP + ALL apiserver\n"
        "#          endpoint IPs (and multi-resolver DNS) here, correct under every CNI datapath. Deferred:\n"
        "#          the runner parses each var via ipaddress.ip_address() and FAIL-CLOSES on a comma,\n"
        "#          so a list needs a runner change first (the wizard contract §11 preconditions #1 / #1b).\n"
        f'NO_INTERNET_DNS_IP="{dns_ip}"\n'
        f'NO_INTERNET_KUBE_API_IP="{kube_api_ip}"\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Provision-now PVC manifest (the wizard contract §6.2 opt-2) — offline-testable; apply is interactive-only
# ─────────────────────────────────────────────────────────────────────────────


def pvc_manifest(name: str, ns: str, sc: str, size: str = "100Gi") -> str:
    """PURE. The ReadWriteMany PVC manifest provision-now WOULD apply. Returned as text so the offline
    build/selftest can validate exactly what would be applied WITHOUT a cluster; the live `kubectl apply`
    of this manifest happens ONLY in the interactive flow on explicit y/N (never under --play, §6.2/S6).
    """
    doc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": {"app.kubernetes.io/managed-by": "llmb-k8s-init"},
        },
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "resources": {"requests": {"storage": size}},
        },
    }
    if sc:
        doc["spec"]["storageClassName"] = sc
    return json.dumps(doc, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Node-size facts: CPU_PER_NODE / MEM_PER_NODE / WHOLE_NODE_CPU / WHOLE_NODE_MEM
# Same shape as GPU_PER_NODE: cluster truth, auto-detected by `init`, written to the PROFILE — never a recipe.
# ─────────────────────────────────────────────────────────────────────────────

# WHY headroom at all: `allocatable` already excludes kube-reserved/system-reserved, but it does NOT exclude
# the DaemonSets ALREADY RUNNING on the node (CNI, DCGM/GPU-operator, node-exporter, fluent-bit, …). Those
# pods hold real requests against the same allocatable pool, so a whole-node pod asking for 100% of
# allocatable can never be scheduled — it sits Pending forever. 85% leaves room for that resident set.
WHOLE_NODE_HEADROOM_PCT = 85


def _cpu_to_millicores(s: str) -> int:
    """PURE. Normalize a k8s cpu quantity to MILLICORES. Accepts the two forms `.status.allocatable.cpu`
    actually takes: millicores ("139580m") or a bare/fractional core count ("192", "191.5"). Junk/empty → 0
    (callers then simply omit the fact, exactly like GPU_PER_NODE degrades)."""
    s = str(s or "").strip()
    if not s:
        return 0
    try:
        if s.endswith("m"):
            return int(float(s[:-1]))
        return int(float(s) * 1000)
    except (TypeError, ValueError):
        return 0


_MEM_UNITS = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
}


def _mem_to_gib(s: str) -> int:
    """PURE. Normalize a k8s memory quantity to WHOLE Gi, rounding DOWN (never over-promise memory a node
    doesn't have). Accepts "948007936Ki" (the usual allocatable form), "903Gi", plain bytes, and the
    decimal SI suffixes. Junk/empty → 0."""
    s = str(s or "").strip()
    if not s:
        return 0
    mult = 1
    for suf in ("Ki", "Mi", "Gi", "Ti", "K", "M", "G", "T"):
        if s.endswith(suf):
            mult, s = _MEM_UNITS[suf], s[: -len(suf)]
            break
    try:
        return int(float(s) * mult) // (1024**3)
    except (TypeError, ValueError):
        return 0


# A whole-node cell may pin its BENCH pod to the server's own node (bench.colocate_with_server → a hard
# podAffinity). When it does, the reservation below and the bench pod compete for one node, so the value
# written here has to leave the bench pod room. Defaults match the shipped BENCH_CPU_REQUEST / bench memory.
BENCH_RESERVE_CPU_CORES = 16
BENCH_RESERVE_MEM_GIB = 16


def whole_node_cpu(
    cpu_millicores: int,
    pct: int = WHOLE_NODE_HEADROOM_PCT,
    reserve_cores: int = BENCH_RESERVE_CPU_CORES,
) -> int:
    """PURE. What a whole-node cell requests AND limits for cpu, as an INTEGER CORE COUNT (no unit suffix).
    Integer cores are deliberate: requests==limits gives the pod Guaranteed QoS, and Guaranteed + an INTEGER
    cpu request is the precondition for exclusive pinned CPUs under a static CPUManager policy. A fractional
    millicore value like "118643m" silently forfeits that, so we floor to whole cores. 0 when undetected.

    `reserve_cores` is a CAP, not a second subtraction. The percentage headroom already covers the resident
    DaemonSets; on a large node it also happens to cover a co-scheduled bench pod (192 cpu → 163, +16 = 179,
    fits), so the cap changes nothing there. On a SMALL node it does not: 85% of 32 cpu is 27, and 27+16=43
    does not fit 32. Taking the min keeps big-node behaviour identical while stopping the wizard from
    writing a value that cannot co-schedule."""
    pct_based = int(cpu_millicores) * int(pct) // 100 // 1000
    capped = int(cpu_millicores) // 1000 - max(0, int(reserve_cores))
    return max(0, min(pct_based, capped))


def whole_node_mem_gib(
    mem_gib: int,
    pct: int = WHOLE_NODE_HEADROOM_PCT,
    reserve_gib: int = BENCH_RESERVE_MEM_GIB,
) -> int:
    """PURE. What a whole-node cell requests AND limits for memory, in whole Gi, rounded DOWN. 0 when
    undetected. `reserve_gib` is the same co-scheduling CAP as in whole_node_cpu."""
    pct_based = int(mem_gib) * int(pct) // 100
    capped = int(mem_gib) - max(0, int(reserve_gib))
    return max(0, min(pct_based, capped))


def node_size_facts(nodes: list) -> tuple:
    """PURE. Derive (cpu_millicores, mem_gib, warning) from the ALREADY-PROBED GPU node list. Reports the
    SMALLEST node of each dimension — a whole-node pod must fit the smallest node it might land on — and
    returns a one-line warning when the GPU nodes are MIXED sizes (warn, never fail). No detectable value
    (no RBAC to list nodes, no GPU nodes, junk quantities) → (0, 0, "") and the facts are simply omitted.
    """
    cpus = sorted({_cpu_to_millicores(n.get("cpu_alloc")) for n in nodes} - {0})
    mems = sorted({_mem_to_gib(n.get("mem_alloc")) for n in nodes} - {0})
    warn = ""
    if len(cpus) > 1 or len(mems) > 1:
        warn = (
            f"mixed GPU node sizes (cpu {[f'{c}m' for c in cpus]}, mem {[f'{m}Gi' for m in mems]}) — "
            f"using the SMALLEST, since a whole-node pod must fit the smallest node it can land on."
        )
    return (cpus[0] if cpus else 0), (mems[0] if mems else 0), warn


# ─────────────────────────────────────────────────────────────────────────────
# Answers → profile text
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Answers:
    cluster: str = ""
    context: str = ""
    namespace: str = ""
    owner: str = ""
    # CONNECT_CMD — the exact SSO/Teleport/VPN login command for this cluster, echoed verbatim by fleet /
    # resolve / preflight whenever the apiserver won't answer. Empty = tooling derives `tsh kube login <ctx>`.
    connect_cmd: str = ""
    gpu_product: str = ""
    arch: str = ""
    gpu_per_node: str = ""
    # Node-size cluster facts (auto-detected from the same GPU-node probe as gpu_per_node). cpu_per_node /
    # mem_per_node are the raw allocatable truth; whole_node_* are that truth minus DaemonSet headroom and
    # are what a whole_node cell requests==limits.
    cpu_per_node: str = ""
    mem_per_node: str = ""
    whole_node_cpu: str = ""
    whole_node_mem: str = ""
    pull_secret: str = ""
    hf_secret: str = ""
    model_cache_pvc: str = ""
    artifacts_sc: str = ""
    control_sc: str = ""
    # Per-recipe model-cache STORAGE-CLASS candidates (PER-RECIPE-CACHE-DESIGN). The model cache DEFAULTS to
    # the fast high-throughput RWX class (FSx/Lustre) — see the Storage step; ebs kept as the RWO opt-in.
    cache_rwx_class: str = ""
    cache_rwo_class: str = ""
    # Empty by default so a resume can load the EXISTING profile's SCHEDULER_NAME (a non-empty dataclass
    # default would mask it under _fill_answers_from_env's only_missing rule, #45). The "default-scheduler"
    # fallback is applied at the point of use (run_interactive / run_play) for a FRESH profile.
    scheduler: str = ""
    no_internet_dns_ip: str = ""
    no_internet_kube_api_ip: str = ""
    cni: str = ""  # audit annotation only


def build_profile_text(a: Answers) -> str:
    """PURE. Render the full profile .env text from collected answers: the profile_init body (incl. the NEW
    CONTROL_STORAGE_CLASS field) + the schema-version header + the no-internet single-IP block + a CNI audit
    comment. This is what write_profile persists atomically."""
    body = pi.format_profile(
        cluster=a.cluster,
        context=a.context,
        namespace=a.namespace,
        owner=a.owner,
        gpu_product=a.gpu_product,
        arch=a.arch,
        pull_secret=a.pull_secret,
        hf_secret=a.hf_secret,
        model_cache_pvc=a.model_cache_pvc,
        artifacts_sc=a.artifacts_sc,
        scheduler=a.scheduler,
        control_sc=a.control_sc,
        connect_cmd=a.connect_cmd,
        cache_rwx_class=a.cache_rwx_class,
        cache_rwo_class=a.cache_rwo_class,
    )
    header = f"# {pr.SCHEMA_HEADER}={pr.PROFILE_SCHEMA_VERSION}\n"
    text = header + body
    if a.cni:
        text += f"\n# CNI (audit annotation only — does not select the no-internet IP value): {a.cni}\n"
    if a.gpu_per_node:
        text += (
            "\n# GPU_PER_NODE — allocatable GPUs on one GPU node (auto-detected). Whole-node cells\n"
            "# (requires.gpu.whole_node) request exactly this many so a benchmark never shares a node.\n"
            f'GPU_PER_NODE="{a.gpu_per_node}"\n'
        )
    if a.cpu_per_node or a.mem_per_node:
        text += (
            "\n# CPU_PER_NODE / MEM_PER_NODE — allocatable cpu + memory on one GPU node (auto-detected;\n"
            "# smallest node when node sizes are mixed). Cluster facts, like GPU_PER_NODE.\n"
        )
        if a.cpu_per_node:
            text += f'CPU_PER_NODE="{a.cpu_per_node}"\n'
        if a.mem_per_node:
            text += f'MEM_PER_NODE="{a.mem_per_node}"\n'
    if a.whole_node_cpu or a.whole_node_mem:
        text += (
            f"# WHOLE_NODE_CPU / WHOLE_NODE_MEM — what a whole-node cell (requires.gpu.whole_node)\n"
            f"# requests AND limits (requests==limits ⇒ Guaranteed QoS). {WHOLE_NODE_HEADROOM_PCT}% of\n"
            "# allocatable: allocatable does not exclude the node's resident DaemonSets (CNI, DCGM,\n"
            "# node-exporter, ...), so asking for 100% leaves the pod Pending forever. CPU is whole\n"
            "# cores (Guaranteed + integer cpu = exclusive CPUs under a static CPUManager policy).\n"
        )
        if a.whole_node_cpu:
            text += f'WHOLE_NODE_CPU="{a.whole_node_cpu}"\n'
        if a.whole_node_mem:
            text += f'WHOLE_NODE_MEM="{a.whole_node_mem}"\n'
    text += render_no_internet_block(a.no_internet_dns_ip, a.no_internet_kube_api_ip)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Resume/edit write — PRESERVE-then-overlay (fix #45)
# The fresh-install path (build_profile_text) RECONSTRUCTS the whole .env from pi.format_profile()'s fixed
# field set. On resume over an EXISTING profile that silently drops every key/block it doesn't enumerate —
# MODEL_CACHE_SUBPATH (forced back to "."), custom / RDMA / CONNECT_CMD blocks, any future block — and
# clobbers CLUSTER with the profile filename. The resume path below instead PRESERVES the existing file and
# overlays ONLY the fields the wizard actually re-derived/confirmed, so nothing outside that set is lost.
# ─────────────────────────────────────────────────────────────────────────────


def overlay_env_text(existing_text: str, updates: dict) -> str:
    """PURE. Return `existing_text` with each KEY in `updates` reset to its (double-quoted) value, rewriting
    the existing ``KEY=…`` line IN PLACE — order, comments, blank lines, and (crucially) every key NOT in
    `updates` preserved verbatim. A KEY not already present is appended at the end. This is the resume-write
    primitive behind fix #45: the caller passes ONLY the wizard-confirmed keys, so MODEL_CACHE_SUBPATH, the
    custom/RDMA/CONNECT_CMD blocks, the CLUSTER identity, and any future block survive untouched.
    """
    seen: set = set()
    out: list = []
    for ln in existing_text.splitlines():
        stripped = ln.lstrip()
        m = _re.match(r"([A-Za-z_][A-Za-z0-9_]*)=", stripped)
        if m and not stripped.startswith("#") and m.group(1) in updates:
            key = m.group(1)
            out.append(f'{key}="{updates[key]}"')
            seen.add(key)
        else:
            out.append(ln)
    tail = [k for k in updates if k not in seen]
    if tail:
        if out and out[-1].strip():
            out.append("")
        out.extend(f'{k}="{updates[k]}"' for k in tail)
    return "\n".join(out) + "\n"


def _answers_env_updates(a: Answers) -> dict:
    """PURE. The wizard-confirmed Answers fields mapped back to their profile env KEYs, empties dropped. These
    are the ONLY keys a resume-write overlays (see overlay_env_text / build_resume_profile_text) — everything
    else in an existing profile is preserved. CLUSTER is deliberately absent from _ENV_TO_FIELD, so a resume
    never rewrites it to the profile filename (fix #45)."""
    field_to_env = {field: key for key, field in _ENV_TO_FIELD.items()}
    updates: dict = {}
    for field, key in field_to_env.items():
        val = str(getattr(a, field, "") or "").strip()
        if val:
            updates[key] = val
    return updates


def build_resume_profile_text(a: Answers, existing_text: str) -> str:
    """PURE. The resume/edit write path (fix #45): PRESERVE the existing profile's full text and overlay ONLY
    the wizard-re-derived/confirmed fields. Overlaying instead of reconstructing (build_profile_text) makes
    every existing key/block — MODEL_CACHE_SUBPATH, custom/RDMA/CONNECT_CMD, CLUSTER, any future block — survive
    a re-init/resume that the old fixed-field-set reconstruction dropped."""
    return overlay_env_text(existing_text, _answers_env_updates(a))


# ─────────────────────────────────────────────────────────────────────────────
# Playfile (the wizard contract §7 / M2 / M3)
# ─────────────────────────────────────────────────────────────────────────────

_PLAYFILE_REQUIRED = (
    "cluster",
    "context",
    "namespace",
    "gpu_product",
    "pull_secret",
    "model_cache_pvc",
)


def load_playfile(path: Path) -> Answers:
    """Load + schema-validate a playfile into Answers. Raises ValueError (a single-line message) on malformed
    YAML, a missing schema_version, a missing required field, the literal placeholder model-cache PVC, or a
    comma-list in either no-internet IP (Phase-1 fail-closed)."""
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except yaml.YAMLError as e:
        first = str(e).splitlines()[0] if str(e) else "unparseable"
        raise ValueError(f"malformed YAML ({first})")
    if not isinstance(data, dict):
        raise ValueError("playfile is not a mapping")
    if "schema_version" not in data:
        raise ValueError("playfile missing schema_version (intentional k8s extension, M3)")
    missing = [k for k in _PLAYFILE_REQUIRED if not str(data.get(k) or "").strip()]
    if missing:
        raise ValueError(f"playfile missing required field(s): {', '.join(missing)}")
    # Fail-closed on the literal placeholder PVC — a playfile that carries `shared-model-cache` would write a
    # profile pointing at a cache that does not exist and die after GPU allocation (§6.2). Headless never
    # provisions, so a placeholder here can only be a mistake.
    if str(data.get("model_cache_pvc") or "").strip() == _PLACEHOLDER_PVC:
        raise ValueError(
            f"model_cache_pvc is the placeholder {_PLACEHOLDER_PVC!r} — name a real, existing "
            "RWX PVC (headless never provisions one)"
        )
    for ipkey in ("no_internet_dns_ip", "no_internet_kube_api_ip"):
        val = str(data.get(ipkey) or "").strip()
        if val and not valid_single_ip(val):
            raise ValueError(f"playfile {ipkey}={val!r} is not a single IP — Phase-1 forbids a comma-list")
    fields = {f.name for f in Answers.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return Answers(**{k: v for k, v in data.items() if k in fields})


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────


def _emit_readiness(profile: str, checks: list, profile_path: Path, profiles_dir: Path = PROFILES_DIR) -> bool:
    """Write the per-profile readiness stamp cluster-profiles/.state/<profile>.readiness.json (gitignored,
    local-only, atomic 0600) and echo a one-line machine-readable signal. Reuses the STRUCTURED Check list
    from run_battery + verdict() — never recomputes. Per-profile filename → parallel inits never collide.
    Emit-only in Phase-1 (a follow-on `fleet --grid` reads it). Best-effort: a write hiccup never changes
    the exit code. Returns run_ready."""
    import hashlib
    import json as _json
    from datetime import datetime, timezone
    import k8s_config

    ok, summary = cr.verdict(checks)
    counts = {cr.PASS: 0, cr.WARN: 0, cr.FAIL: 0, cr.SKIP: 0}
    for c in checks:
        counts[c.level] = counts.get(c.level, 0) + 1
    try:
        phash = hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        phash = ""
    payload = {
        "schema": k8s_config.READINESS_STATE_SCHEMA,
        "profile": profile,
        "run_ready": bool(ok),
        "level_counts": {k: counts.get(k, 0) for k in (cr.PASS, cr.WARN, cr.FAIL, cr.SKIP)},
        "checks": [{"id": c.id, "level": c.level} for c in checks],
        "profile_hash": phash,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        k8s_config.atomic_write_json(k8s_config.readiness_state_path(profile, profiles_dir), payload)
    except Exception:  # noqa: BLE001
        pass
    print("LLMB_CLUSTER_READY=" + _json.dumps({"profile": profile, "run_ready": ok, "summary": summary}))
    return ok


def _probe_pvcs(kctx: Callable, ns: str) -> tuple[bool, list]:
    """Return (listed_ok, names) via a context-pinned kubectl runner. listed_ok=False means the list call
    FAILED (RBAC/unreachable) — the caller safe-degrades to ⚠ 'unverified' rather than a false ❌.
    """
    if not ns:
        return False, []
    try:
        rc, out, _ = kctx(
            [
                "-n",
                ns,
                "get",
                "pvc",
                "--no-headers",
                "-o",
                "custom-columns=NAME:.metadata.name",
            ]
        )
    except Exception:  # noqa: BLE001
        return False, []
    if rc != 0:
        return False, []
    return True, [ln.strip() for ln in (out or "").splitlines() if ln.strip()]


def _headless_model_cache_check(chosen: str, pvcs: list, listed_ok: bool) -> cr.Check:
    """Fold the §6.2 model-cache existence check into the headless battery. Headless only VERIFIES (never
    provisions): a positively-listed namespace missing the chosen PVC is a ❌; an unlistable namespace
    safe-degrades to ⚠ (preflight re-checks at run time)."""
    if not listed_ok:
        return cr.Check(
            "model-cache",
            cr.WARN,
            f'MODEL_CACHE_PVC "{chosen}" existence unverified (PVC listing unavailable — '
            "RBAC/unreachable); preflight re-checks the cache at run time",
        )
    return model_cache_check(chosen, pvcs)


def run_play(
    cluster: str,
    playfile: str,
    *,
    profiles_dir: Path = PROFILES_DIR,
    battery_fn: Optional[Callable] = None,
    dry_run: bool = False,
    verbose: bool = False,
    pvcs: Optional[list] = None,
) -> int:
    """--play: headless. Load a playfile, write the profile non-interactively, run the Done battery + a live
    model-cache existence check, gate on it (non-zero on any ❌). NEVER provisions a live PVC (§6.2/S6 —
    structurally: no provision path here). `battery_fn(prof)->list[Check]` is injected in tests; live default
    is cluster_readiness.run_battery. `pvcs` injects the namespace PVC list in tests (live path probes it).
    """
    try:
        a = load_playfile(Path(playfile))
    except Exception as e:
        print(f"llmb-k8s init --play: bad playfile {playfile}: {e}")
        return EXIT_ERROR
    if cluster and a.cluster and cluster != a.cluster:
        print(
            f"llmb-k8s init --play: --cluster {cluster} != playfile cluster {a.cluster} "
            "(cluster identity is locked 🔒)"
        )
        return EXIT_ERROR
    a.cluster = cluster or a.cluster
    name_err = validate_cluster_name(a.cluster)
    if name_err:
        print(f"llmb-k8s init --play: {name_err}")
        return EXIT_ERROR
    path = pr.profile_env_path(a.cluster, profiles_dir)

    # Identity lock on resume/edit (§5): if the profile exists, its KUBE_CONTEXT / GPU_PRODUCT are 🔒 — a
    # playfile that changes either is a DIFFERENT cluster and is refused; existing non-identity values are
    # preserved as defaults for anything the playfile left blank.
    existing_text: Optional[str] = None
    if path.exists():
        try:
            existing_text = path.read_text()
        except Exception:  # noqa: BLE001
            existing_text = None
        try:
            existing = pr._read_env(path)
        except Exception:  # noqa: BLE001
            existing = {}
        lock_err = _enforce_identity_lock(a, existing)
        if lock_err:
            print(f"llmb-k8s init --play: {lock_err}")
            return EXIT_ERROR
        _fill_answers_from_env(a, existing, only_missing=True)

    a.scheduler = a.scheduler or "default-scheduler"  # fresh-profile fallback (resume keeps the existing one)
    # Resume-write PRESERVES the existing profile and overlays only the confirmed fields (fix #45); a fresh
    # profile is reconstructed from scratch.
    text = build_resume_profile_text(a, existing_text) if existing_text is not None else build_profile_text(a)
    if dry_run:
        print(f"── init --play --dry-run — would write {path} (mode 0600) ──")
        print(text)
        return EXIT_OK
    pi.write_profile(path, text)
    print(f"✓ Wrote {path}  (mode 0600, schema_version={pr.PROFILE_SCHEMA_VERSION})")
    prof = pr._read_env(path)
    battery = battery_fn or cr.run_battery
    checks = list(battery(prof))

    # Fold the live model-cache existence check into the battery BEFORE gating (§6.2). `pvcs` injected in
    # tests; in a real headless run (battery_fn is None) probe the namespace via a context-pinned runner.
    if pvcs is not None:
        checks.append(model_cache_check(a.model_cache_pvc, list(pvcs)))
    elif battery_fn is None:
        listed_ok, names = _probe_pvcs(_ctx_krun(a.context), a.namespace)
        checks.append(_headless_model_cache_check(a.model_cache_pvc, names, listed_ok))

    print(render_done_panel(a.cluster, checks))
    ok = _emit_readiness(a.cluster, checks, path, profiles_dir)
    return EXIT_OK if ok else EXIT_ERROR


def run_record(cluster: str, playfile: str, **_kw) -> int:
    """--record: STUBBED in Phase-1 (the wizard contract §7). TODO Phase-2: run Collect+Confirm interactively, WRITE
    the profile, and capture the answered values to a replayable playfile via the atomic k8s_config writer.
    """
    raise NotImplementedError(
        "llmb-k8s init --record is not implemented in Phase-1. "
        "TODO Phase-2: capture Collect/Confirm answers → atomic playfile + write the profile."
    )


def run_express(cluster: str, **_kw) -> int:
    """--express: STUBBED in Phase-1 (the wizard contract §7). TODO Phase-2: load ~/.config/llmb/k8s-system.yaml
    stable prefs, re-detect only cluster-specific facts, one confirm, write, battery."""
    raise NotImplementedError(
        "llmb-k8s init --express is not implemented in Phase-1. "
        "TODO Phase-2: k8s-system.yaml prefs + re-detect cluster facts + one-confirm fast path."
    )


# ── interactive helpers ──────────────────────────────────────────────────────


def _retry_probe(fn: Callable, connect_cmd: str, *, tries: int = 3):
    """S4/#9 — Teleport/auth-blip resilience for a single Collect probe. Retry `fn` a few times; on
    exhaustion print the CONNECT_CMD hint but DO NOT abort the wizard (return whatever fn last produced).
    Distinct from resume-state (a killed wizard) — this is a transient auth blip mid-Collect.
    """
    import time

    last = None
    for t in range(1, tries + 1):
        try:
            last = fn()
            return last
        except Exception as e:  # noqa: BLE001 — a probe blip must never crash the wizard
            last = e
            if t < tries:
                time.sleep(t * 2)
    hint = connect_cmd or "tsh kube login <context>"
    print(f"  ⚠ probe kept failing (auth blip / Teleport?). Reconnect and re-run — hint: {hint}")
    if isinstance(last, Exception):
        return None
    return last


def _confirm(a: Answers, *, assume_yes: bool = False) -> bool:
    """The NEW pre-write 'Installation Summary' confirm screen + Yes/Edit loop (§3 Confirm, NEW-4). This is
    NOT profile_init.print_summary (that is a POST-write panel). 🔒 marks identity (name/context/GPU).
    """
    print("\n── Confirm ────────────────────────────────────────────────────────")
    print("Installation Summary")
    print("=" * 50)
    print(f"🔒 Cluster (profile name): {a.cluster}")
    print(f"🔒 Kube context:           {a.context}")
    print(f"🔒 GPU product / arch:     {a.gpu_product} / {a.arch}")
    print(f"   Namespace:              {a.namespace}")
    print(f"   Owner:                  {a.owner}")
    print(f"   Connect cmd (SSO):      {a.connect_cmd or '(none — tooling derives `tsh kube login`)'}")
    print(f"   Image-pull secret:      {a.pull_secret}")
    print(f"   HF secret:              {a.hf_secret}")
    print(f"   Model-cache PVC:        {a.model_cache_pvc}")
    print(f"   Artifacts SC (RWO):     {a.artifacts_sc}")
    print(f"   Control SC (RWX):       {a.control_sc or '(unset — RWX detector had no confirmed match)'}")
    print(f"   CNI (audit note):       {a.cni or '(unknown)'}  (annotation only)")
    print(f"   NO_INTERNET_DNS_IP:     {a.no_internet_dns_ip or '(empty ⚠)'}")
    print(f"   NO_INTERNET_KUBE_API_IP:{a.no_internet_kube_api_ip or '(empty ⚠)'}")
    print("\n   Fields marked 🔒 are identity — changing them means a different profile.")
    if assume_yes:
        return True
    while True:
        raw = input("Continue?  [Y]es, write profile / [e]dit / [q]uit: ").strip().lower()
        if raw in ("", "y", "yes"):
            return True
        if raw in ("q", "quit"):
            return False
        if raw in ("e", "edit"):
            _edit_loop(a)
            return _confirm(a, assume_yes=assume_yes)
        print("    Enter y, e, or q.")


def _edit_loop(a: Answers) -> None:
    """Minimal Edit: re-prompt the non-identity fields with the current answers as defaults (identity 🔒)."""
    a.namespace = pi._prompt("Namespace", a.namespace)
    a.owner = pi._prompt("Owner", a.owner)
    a.connect_cmd = pi._prompt("Connect cmd (SSO/Teleport login)", a.connect_cmd)
    # Secret NAMES are auto-defaulted (<profile>-nvcr-cred / <profile>-hf-token) and never prompted — the
    # credential is auto-created by `install` from your local NGC/HF creds. Not offered for edit here.
    # Model-cache PVC is NOT re-prompted either — per-recipe-cache defers cache provisioning entirely to
    # `install` (derive_recipe_cache + ensure_recipe_cache_pvcs, one claim per recipe).
    a.artifacts_sc = pi._prompt("Artifacts SC (RWO)", a.artifacts_sc)
    a.control_sc = pi._prompt("Control SC (RWX)", a.control_sc)
    a.no_internet_dns_ip = pi._prompt("NO_INTERNET_DNS_IP", a.no_internet_dns_ip)
    a.no_internet_kube_api_ip = pi._prompt("NO_INTERNET_KUBE_API_IP", a.no_internet_kube_api_ip)


def _model_cache_fix_card(a: Answers, pvcs: list, ns: str, krun, rbac: dict) -> None:
    """§6.2 three-way fix card, INTERACTIVE-ONLY. [1] pick existing / [2] provision-now (RBAC-gated, manifest
    shown, explicit y/N, then apply) / [3] name to create at stage time. Option [2] is the ONE place init
    applies a live non-throwaway resource, and only on explicit approval; never under --play (S6).
    """
    cands = [p for p in pvcs]
    # The default must be an option the user can actually TAKE. On a fresh namespace there are no PVCs, so
    # [1] is unselectable — offering "[1]" as the default meant Enter fell through to "Enter 1, 2, or 3."
    # and re-prompted forever (observed: 38 rejections, then the wizard ran out of input and cancelled).
    # With no candidates the sane default is [3]: let `install` create the cache at stage time.
    default_pick = "1" if cands else "3"
    print("     Fix — choose one:")
    print(f"       [1] Select an existing PVC:  {', '.join(cands) if cands else '(none available)'}")
    print("       [2] Provision one now  (applies a ReadWriteMany PVC; manifest shown first)")
    print("       [3] Enter a name I'll create at stage time" + ("   ← recommended" if not cands else ""))
    while True:
        raw = input(f"     Selection [{default_pick}]: ").strip() or default_pick
        if raw == "1" and not cands:
            print("     No existing PVC to select — choose [2] to provision now, or [3] to create at stage time.")
            continue
        if raw == "1" and cands:
            for i, c in enumerate(cands, 1):
                print(f"       {i}  {c}")
            sel = input(f"     Pick [1-{len(cands)}]: ").strip() or "1"
            try:
                a.model_cache_pvc = cands[int(sel) - 1]
                return
            except (ValueError, IndexError):
                print("     invalid pick.")
                continue
        if raw == "2":
            # RBAC-first (M5): provision-now needs `create persistentvolumeclaims`; assert before offering.
            if not rbac.get("create_pvcs", False):
                print("     ⚠ RBAC forbids `create persistentvolumeclaims` — cannot provision. Pick [1] or [3].")
                continue
            name = input("     New PVC name [shared-model-cache]: ").strip() or _PLACEHOLDER_PVC
            sc = a.control_sc or input("     RWX storage class: ").strip()
            manifest = pvc_manifest(name, ns, sc)
            print("     Would apply:\n" + "\n".join("       " + ln for ln in manifest.splitlines()))
            ok = input("     Apply this PVC now? [y/N]: ").strip().lower()
            if ok in ("y", "yes"):
                rc, _, err = (
                    krun(["-n", ns, "apply", "-f", "-"], stdin=manifest)
                    if _krun_accepts_stdin(krun)
                    else (1, "", "stdin unsupported")
                )
                if rc == 0:
                    print(f"     ✓ applied PVC {name}")
                    a.model_cache_pvc = name
                    return
                print(f"     ✗ apply failed: {str(err)[:120]} — pick [1] or [3].")
                continue
            continue
        if raw == "3":
            a.model_cache_pvc = input("     PVC name to create at stage time: ").strip() or _PLACEHOLDER_PVC
            return
        print("     Enter 1, 2, or 3.")


def _model_cache_step(a: Answers, pvcs: list, ns: str, krun, rbac: dict, *, cluster: str) -> None:
    """Model-cache selection, DEFAULTING to defer-to-install (§6.2 / install G3). init runs before recipe
    selection, so it cannot size/class the cache for the chosen cell — install.py's ensure_model_cache_pvc
    creates the correct PVC (right size + storage class per the profile/recipe) at STAGE time. So:

      - an already-carried value (resume/existing profile) is classified & kept (fix card only on a hard ❌);
      - an existing NAME-matched cache PVC is used directly (zero friction);
      - otherwise the DEFAULT is 'create <cluster>-model-cache at stage time' (deferred). Picking an existing
        PVC or eagerly provisioning an RWX PVC now are the NON-default options — eager provisioning is the
        only case that needs an RWX class, called out explicitly so a no-RWX cluster isn't a dead-end.
    """
    print("Model cache")
    # A value carried from an existing profile / resume: classify and keep; only a hard ❌ opens the fix card.
    if a.model_cache_pvc:
        mc = model_cache_check(a.model_cache_pvc, pvcs)
        print(f"  {DONE_GLYPH.get(mc.level, '?')} {mc.message}")
        if mc.level == cr.FAIL:
            _model_cache_fix_card(a, pvcs, a.namespace, krun, rbac)
        return
    name_matched = [p for p in pvcs if any(h in p.lower() for h in _CACHE_NAME_HINTS)]
    if name_matched:
        a.model_cache_pvc = name_matched[0]
        print(f"  ✓ using existing model-cache PVC '{a.model_cache_pvc}' (matched by name)")
        return
    default_name = default_model_cache_name(cluster)
    cands = [p for p in pvcs]
    print("  init runs before recipe selection, so `install` creates the cache PVC at stage time —")
    print(f"  sized + storage-classed for the recipe you pick. Default: create '{default_name}' then.")
    print("  Options:")
    print("    [Enter]  Defer to install — create at stage time  (recommended)")
    if cands:
        print(f"    [1]      Use an existing PVC:  {', '.join(cands)}")
    print("    [2]      Provision an RWX PVC now  (only if you need multi-node shared caching)")
    while True:
        raw = input("     Selection [Enter = defer to install]: ").strip()
        if not raw:
            a.model_cache_pvc = default_name
            print(f"  · deferred — `install` will create '{default_name}' at stage time (G3).")
            return
        if raw == "1" and cands:
            for i, c in enumerate(cands, 1):
                print(f"       {i}  {c}")
            sel = input(f"     Pick [1-{len(cands)}]: ").strip() or "1"
            try:
                a.model_cache_pvc = cands[int(sel) - 1]
                return
            except (ValueError, IndexError):
                print("     invalid pick.")
                continue
        if raw == "2":
            if not rbac.get("create_pvcs", False):
                print(
                    "     ⚠ RBAC forbids `create persistentvolumeclaims` — cannot provision now. "
                    "Press Enter to defer to install instead."
                )
                continue
            name = input(f"     New PVC name [{default_name}]: ").strip() or default_name
            # RWX is required for multi-node sharing; on a no-RWX cluster, an RWO class works for single-pod.
            sc = a.control_sc or input("     Storage class (RWX for multi-node; RWO like ebs for single-pod): ").strip()
            manifest = pvc_manifest(name, ns, sc)
            print("     Would apply:\n" + "\n".join("       " + ln for ln in manifest.splitlines()))
            ok = input("     Apply this PVC now? [y/N]: ").strip().lower()
            if ok in ("y", "yes"):
                rc, _, err = (
                    krun(["-n", ns, "apply", "-f", "-"], stdin=manifest)
                    if _krun_accepts_stdin(krun)
                    else (1, "", "stdin unsupported")
                )
                if rc == 0:
                    print(f"     ✓ applied PVC {name}")
                    a.model_cache_pvc = name
                    return
                print(f"     ✗ apply failed: {str(err)[:120]} — press Enter to defer to install instead.")
                continue
            continue
        print("     Enter (defer), 1, or 2.")


def _krun_accepts_stdin(krun) -> bool:
    try:
        import inspect

        return "stdin" in inspect.signature(krun).parameters
    except (TypeError, ValueError):
        return False


def _ctx_krun(context: str):
    """A profile-context-pinned kubectl runner with an optional stdin= (for apply -f -)."""
    import subprocess

    def _run(args, timeout: int = 30, stdin: Optional[str] = None):
        ctx = ["--context", context] if context else []
        argv = ["kubectl", *ctx, "--request-timeout=25s", *args]
        try:
            p = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin if stdin is not None else None,
            )
            return p.returncode, p.stdout, p.stderr
        except Exception as e:  # noqa: BLE001
            return 1, "", str(e)

    return _run


def run_interactive(
    cluster: Optional[str] = None,
    *,
    profiles_dir: Path = PROFILES_DIR,
    dry_run: bool = False,
    verbose: bool = False,
    krun=pi.default_krun,
    battery_fn: Optional[Callable] = None,
    preselect_context: Optional[str] = None,
) -> int:
    """The full §4 walk-through: preamble → collect (auto-detect-then-confirm, read-only CNI/IP probes
    HOISTED ahead of Confirm, S2) → confirm → provision (atomic write) → done (battery + gated render).

    No-flag discovery: when `cluster` is falsy the wizard first DISCOVERS the connected clusters (kubectl
    contexts) and has the operator PICK one (a fresh user need not know the profile label up front, §4). The
    pick derives a suggested profile name and pins the chosen context (preselect_context) so the Collect
    step doesn't re-ask it."""
    import k8s_config

    # ── Discover: no --cluster → pick from the connected clusters (kubectl contexts) ──
    if not cluster:
        picked = _pick_context_interactive(krun)
        if picked == _PICK_EMPTY:
            return EXIT_ERROR
        if picked == _PICK_QUIT:
            print("  Aborted (nothing written).")
            return EXIT_CANCELLED
        cluster, preselect_context = picked
    name_err = validate_cluster_name(cluster)
    if name_err:
        print(f"llmb-k8s init: {name_err}")
        return EXIT_ERROR
    profile_path = pr.profile_env_path(cluster, profiles_dir)

    # ── Preamble ──────────────────────────────────────────────────────────
    print("\nllmb-k8s init — fresh cluster profile wizard")
    exists = profile_path.exists()
    print(
        f"Target profile: {profile_path}  "
        f"({'exists — resume/edit (identity 🔒)' if exists else 'does not exist — creating'})"
    )
    print("\n── Preamble ───────────────────────────────────────────────────────")
    contexts = pi.list_contexts(krun)
    if not contexts:
        print("  ❌ no kubectl contexts found — set up a kubeconfig first (kubectl config get-contexts)")
        return EXIT_ERROR
    print(f"  ✓ kubectl reachable · {len(contexts)} context(s) available")

    # Two runners: pi.* discovery helpers add `--context` THEMSELVES (they take the context arg), so they get
    # the RAW runner; direct calls (apply, capability probe, CNI detect) get a context-PINNED runner. Passing
    # a pinned runner to a pi.* helper double-stamps `--context` — the bug this split fixes.
    raw_krun = krun  # ctx_krun is bound after the context is known (below), so pi.* helpers never double-stamp

    # Load the EXISTING profile (if any) as defaults + HARD identity lock (§5). Resume state (7-day, from a
    # killed wizard) layers on top as further non-identity defaults.
    existing = {}
    existing_text: Optional[str] = None
    if exists:
        try:
            existing_text = profile_path.read_text()  # verbatim, for the PRESERVE-then-overlay resume write
        except Exception:  # noqa: BLE001
            existing_text = None
        try:
            existing = pr._read_env(profile_path)
        except Exception:  # noqa: BLE001
            existing = {}
    resumed = k8s_config.load_init_state(cluster) or {}
    a = Answers(cluster=cluster)
    _fill_answers_from_env(a, existing, only_missing=True)  # existing profile → defaults
    if resumed:
        # Saved state is a convenience, NOT a source of truth. Two rules, both learned the hard way:
        #  1. NEVER resume a system namespace. A user who lands on one (the old `[1]`→`argocd` default),
        #     deletes the profile and re-runs init got the SAME bad answer replayed for 7 days — the
        #     profile was gone but the state wasn't, and only a one-line notice hinted why.
        #  2. Say WHAT was resumed and HOW to clear it, so a stale answer is never silently authoritative.
        dropped = []
        if pi.is_system_namespace(str(resumed.get("namespace") or "")):
            dropped.append(f"namespace={resumed.pop('namespace')!r} (system namespace — re-asking)")
        for k, v in resumed.items():
            if hasattr(a, k) and not str(getattr(a, k, "") or "").strip():
                setattr(a, k, v)
        carried = ", ".join(sorted(k for k in resumed if hasattr(a, k))) or "(nothing)"
        print(f"  ⓘ resuming from saved wizard state (<7 days old): {carried}")
        for d in dropped:
            print(f"     ↷ discarded {d}")
        print(f"     start clean with:  llmb-k8s init --reset   (or rm {k8s_config.init_state_path()})")
    # Enforce the identity lock now so a locked context/GPU can never be edited away below.
    lock_err = _enforce_identity_lock(a, existing)
    if lock_err:
        print(f"  ❌ {lock_err}")
        return EXIT_ERROR
    locked_context = bool((existing.get("KUBE_CONTEXT") or "").strip())
    locked_gpu = bool((existing.get("GPU_PRODUCT") or "").strip())

    # ── Collect ───────────────────────────────────────────────────────────
    print("\n── Collect ────────────────────────────────────────────────────────")
    current = next((c["name"] for c in contexts if c["current"]), contexts[0]["name"])
    a.context = a.context or preselect_context or current
    if locked_context:
        print(f"Context\n  🔒 {a.context}  (locked from existing profile — a different context is a new profile)")
    elif preselect_context:
        # Already chosen from the discovery menu — don't re-ask; just confirm the pick.
        print(f"Context\n  ✓ {a.context}  (from your cluster pick)")
    else:
        print(f"Context\n  Detected: {a.context}   [Enter to accept, or type another]")
        raw = input("? context: ").strip()
        if raw:
            a.context = raw
    ctx_krun = raw_krun if raw_krun is not pi.default_krun else _ctx_krun(a.context)
    # CONNECT_CMD (SSO/Teleport login) — carried from an existing profile if any; the retry hint + the saved
    # profile echo it back on any future auth✗. A fresh profile has none, so the hint falls back to tsh.
    ok, detail = _retry_probe(lambda: pi.check_reachability(a.context, raw_krun), a.connect_cmd) or (False, "")
    print(f"  {'✓' if ok else '⚠'} {a.context}  → {'reachable' if ok else detail}")
    # If the cluster won't answer AND we have no reconnect command on file, capture the EXACT login line now so
    # every later auth✗ (fleet/run/install) prints a one-line fix instead of a raw kubectl auth error. Purely
    # optional — Enter skips (tooling then derives `tsh kube login <ctx>`). We never store credentials, only the
    # command the operator runs. Skipped in --dry-run (touch-nothing) and when stdin isn't a tty.
    if not ok and not a.connect_cmd and not dry_run and sys.stdin.isatty():
        print("  ⓘ cluster unreachable — this is usually an SSO/Teleport login you run out-of-band.")
        raw = input(f"? exact login command for '{a.context}' (Enter to skip → derive `tsh kube login`): ").strip()
        if raw:
            a.connect_cmd = raw

    # namespace + gpu
    # OWNER is prompted later, so fall back to the login here — otherwise the suggested namespace is a
    # bare "llmb-k8s" and the user's own namespaces do not rank first.
    _owner_hint = (a.owner or os.environ.get("USER") or os.environ.get("LOGNAME") or "").strip()
    a.namespace = a.namespace or (pi.select_namespace(a.context, raw_krun, owner=_owner_hint) or "")
    nodes = _retry_probe(lambda: pi.probe_gpu_nodes(a.context, raw_krun), a.connect_cmd) or []
    # GPU_PER_NODE — cluster truth a WHOLE-NODE cell needs (it requests the node's full GPU count so no
    # other tenant lands beside the benchmark). Derived here because the wizard already has the nodes;
    # without it `install` hard-fails with "set GPU_PER_NODE in the cluster profile" and the user has to
    # hand-edit a file the wizard just wrote.
    if not a.gpu_per_node:
        _counts = sorted({int(n.get("gpus") or 0) for n in nodes if int(n.get("gpus") or 0) > 0})
        if len(_counts) == 1:
            a.gpu_per_node = str(_counts[0])
        elif len(_counts) > 1:
            a.gpu_per_node = str(_counts[-1])
            print(
                f"GPU per node\n  ⚠ mixed GPU node sizes {_counts} — using the largest ({_counts[-1]}); "
                f"set GPU_PER_NODE by hand if your benchmark targets the smaller nodes."
            )
    # CPU_PER_NODE / MEM_PER_NODE / WHOLE_NODE_* — the rest of the whole-node envelope, from the SAME probe
    # (no extra kubectl call). Same contract as GPU_PER_NODE: cluster truth lives in the profile, a mixed
    # fleet warns instead of failing, and an undetectable value is simply omitted from the written profile.
    if not a.cpu_per_node and not a.mem_per_node:
        _cpu_m, _mem_gi, _warn = node_size_facts(nodes)
        if _cpu_m:
            a.cpu_per_node = f"{_cpu_m}m"
            a.whole_node_cpu = str(whole_node_cpu(_cpu_m))
        if _mem_gi:
            a.mem_per_node = f"{_mem_gi}Gi"
            a.whole_node_mem = f"{whole_node_mem_gib(_mem_gi)}Gi"
        if _warn:
            print(f"Node size\n  ⚠ {_warn}")
        elif _cpu_m or _mem_gi:
            print(
                f"Node size\n  ✓ {a.cpu_per_node} cpu · {a.mem_per_node} mem  → whole-node cell requests "
                f"{a.whole_node_cpu} cpu · {a.whole_node_mem} ({WHOLE_NODE_HEADROOM_PCT}% headroom)"
            )
    if locked_gpu:
        print(f"GPU nodes\n  🔒 {a.gpu_product}  arch={a.arch}  (GPU product locked from existing profile)")
        if nodes and not a.arch:
            a.arch = nodes[0]["arch"]
    elif nodes:
        a.gpu_product = a.gpu_product or nodes[0]["gpu_product"]
        a.arch = a.arch or nodes[0]["arch"]
        print(f"GPU nodes\n  ✓ {len(nodes)} node(s)  {a.gpu_product}  arch={a.arch}")
    else:
        a.gpu_product = a.gpu_product or pi._prompt("GPU product", a.gpu_product)
        a.arch = a.arch or pi._prompt("CPU arch (amd64/arm64)", a.arch)

    # discovery (secret NAMES are auto-defaulted below, so discover_secrets is no longer probed here)
    pvcs = pi.discover_pvcs(a.context, a.namespace, raw_krun)
    # DETAILED storage classes (name + provisioner + is-default) — provisioner is what lets us EXCLUDE an
    # object/S3 CSI from the block/file PVC defaults (QA fix).
    sclasses_detailed = pi.discover_storage_classes_detailed(a.context, raw_krun)
    scheds = pi.discover_schedulers(a.context, raw_krun)
    rbac = pi.check_rbac_permissions(a.context, a.namespace, raw_krun)
    # provision-now (§6.2 opt-2) needs `create persistentvolumeclaims` — add it so approval can't hit a 403 (M5).
    rc_pvc, out_pvc, _ = ctx_krun(["auth", "can-i", "create", "persistentvolumeclaims", "-n", a.namespace])
    rbac["create_pvcs"] = rc_pvc == 0 and (out_pvc or "").strip().lower() == "yes"

    a.owner = a.owner or os.environ.get("USER", "")
    a.scheduler = a.scheduler or (scheds[0] if scheds else "default-scheduler")

    # ── Runner identity (run_by) — WHO is running these benchmarks. A PERSON, not a cluster property, so it
    # is persisted USER-level (~/.config/llmb/user, ri.save_runner) and applies across every cluster — NOT
    # written into this cluster's profile .env. Asked once: an already-stored identity is the default, so a
    # re-run just confirms it. Skipped when non-interactive (--play/--dry-run/no tty) — the export's fallback
    # chain (stored → git email → $USER → "unknown") still stamps a runner, so this NEVER blocks a run.
    if not dry_run and sys.stdin.isatty():
        _run_default = ri.prompt_default()
        _prompt = (
            f"? Who is running these benchmarks? [{_run_default}]: "
            if _run_default
            else "? Who is running these benchmarks?: "
        )
        _runner = input(_prompt).strip() or _run_default
        if _runner:
            ri.save_runner(_runner)
            print(f"  ✓ run_by = {_runner}  (stored user-level → {ri.config_file()}; applies to all clusters)")

    # Secrets: the k8s Secret NAME is namespace-scoped boilerplate — auto-default it, NEVER prompt
    # ("don't ask me things the tool can figure out"). What matters is whether a CREDENTIAL SOURCE exists;
    # _report_cred_sources stays quiet (✓) when it does and gives the exact where-to-get fix when it doesn't.
    # An existing profile's names win (only_missing); otherwise the <profile>-scoped defaults are used.
    a.pull_secret = a.pull_secret or default_pull_secret_name(cluster)
    a.hf_secret = a.hf_secret or default_hf_secret_name(cluster)
    _report_cred_sources()

    # Storage classes — two roles, EXPLAINED so bare class names aren't intimidating ("don't
    # intimidate new users; explain or auto-decide").
    # Provisioner-aware defaults (QA fix): an object/S3 CSI is excluded from BOTH pick lists — RWO prefers
    # the cluster default / a block class, RWX prefers a file class. A user can still force an excluded
    # class with !<name> at the prompt.
    non_object_names = [s["name"] for s in sclasses_detailed if not _is_object_sc(s)]
    print("Storage")
    print("  Artifacts SC — where run artifacts + logs go (RWO / ReadWriteOnce, per-run single-writer).")
    print("    A general block class (ebs / gp* / standard) is a safe default.")
    rwo_best = select_rwo_class(sclasses_detailed)
    a.artifacts_sc = a.artifacts_sc or pi._prompt_from_list(
        "Artifacts SC (RWO)",
        non_object_names,
        rwo_best or (non_object_names or [""])[0],
    )
    rwx_best = select_rwx_class(sclasses_detailed)
    if rwx_best:
        print("  Control SC — shared across pods (RWX / ReadWriteMany); needed for MULTI-NODE model caches.")
        print(f"    ⚠ ranked hint: {rwx_best}  (RWX not yet proven on this cluster — confirm)")
        rwx_choices = [rwx_best] + [n for n in non_object_names if n != rwx_best]
        a.control_sc = a.control_sc or pi._prompt_from_list("Control SC (RWX)", rwx_choices, rwx_best)
    else:
        # No RWX class on the cluster — DON'T dead-end with a blank prompt. Single-pod caches run on RWO.
        print("  Control SC — RWX / ReadWriteMany (shared across pods) NOT detected on this cluster.")
        print(f"    Single-pod caches work on RWO (e.g. '{a.artifacts_sc}'); enter that here, or provision an")
        print("    RWX class first if you need multi-node shared caching.")
        a.control_sc = a.control_sc or pi._prompt(
            "Control SC (RWO fallback for single-pod, or an RWX class if you have one)",
            a.artifacts_sc,
        )

    # Model-cache STORAGE CLASSES (PER-RECIPE-CACHE-DESIGN) — DEFAULT the model cache to the FAST path: a
    # high-throughput shared filesystem (FSx/Lustre, RWX). Model weights are large + read-heavy + shareable,
    # while smaller caches may use the selected RWO block-storage class
    # AND (RWX) lets server + bench co-mount without the RWO Multi-Attach colocation constraint. So
    # MODEL_CACHE_RWX_CLASS defaults to the detected RWX/FSx class, and install defaults any recipe with NO
    # declared access_mode to RWX on it. MODEL_CACHE_RWO_CLASS (EBS) is kept for cells that explicitly opt into
    # single-node RWO. (This is the MODEL-CACHE default only — artifacts stay RWO, control stays on its class.)
    a.cache_rwx_class = a.cache_rwx_class or (rwx_best or "")
    a.cache_rwo_class = a.cache_rwo_class or (rwo_best or a.artifacts_sc or "")
    if a.cache_rwx_class:
        print(
            f"  Model cache  → {a.cache_rwx_class}  (RWX / high-throughput) — DEFAULT for large model weights;"
            f" fast cold-load"
        )
        print(
            f"                 RWO class {a.cache_rwo_class or '(none)'} kept for single-node opt-in caches "
            "(requires.cache.access_mode: rwo)."
        )
    else:
        print(
            "  Model cache  → no RWX/FSx class detected; model cache falls back to the RWO/block class "
            f"({a.cache_rwo_class or 'cluster-default'})."
        )
        print("                 For large models, provision a high-throughput shared " "class for the fast path.")

    # Model cache PVC NAME — DEFAULT to defer-to-install (§6.2 / G3). init runs BEFORE recipe selection, so it
    # can't size/class the cache for the chosen cell; install.py's ensure_model_cache_pvc creates the right PVC
    # at stage time from the classes above. An existing name-matched cache is used directly; otherwise the
    # default is "create at stage time" (eager provision is a non-default option).
    _model_cache_step(a, pvcs, a.namespace, ctx_krun, rbac, cluster=cluster)

    # ── HOISTED read-only capability probes (S2 — no-internet IPs + CNI) BEFORE Confirm ──
    try:
        facts = cap.gather_facts(
            {"GPU_PRODUCT": a.gpu_product},
            lambda args, timeout=30: ctx_krun(args, timeout),
        )
        ni = facts.get("no_internet") or {}
        a.no_internet_dns_ip = a.no_internet_dns_ip or (ni.get("dns_ip") or "")
        a.no_internet_kube_api_ip = a.no_internet_kube_api_ip or (ni.get("kube_api_ip") or "")
    except Exception as e:  # noqa: BLE001 — safe-degrade, never block onboarding
        if verbose:
            print(f"  ⚠ capability probe skipped: {e}")
    a.cni = a.cni or _detect_cni(ctx_krun)
    print("Networking (no-internet lanes)  [Phase-1: single proven IP each, DNS/API symmetric]")
    for label, val in (
        ("NO_INTERNET_DNS_IP", a.no_internet_dns_ip),
        ("NO_INTERNET_KUBE_API_IP", a.no_internet_kube_api_ip),
    ):
        glyph = "✓" if valid_single_ip(val) else "⚠"
        print(f"  {glyph} {label} = {val or '(empty — confirm; only no-internet lanes need it)'}")
        if val and not valid_single_ip(val):
            print("     ⚠ not a single IP — Phase-1 writes ONE proven IP; a comma-list is Phase-2")

    # persist resume state before the (interruptible) confirm/provision — but NEVER under --dry-run, which is
    # the touch-nothing mode (persisting resume state is a side effect a dry-run must not have).
    if not dry_run:
        try:
            k8s_config.save_init_state(cluster, asdict(a))
        except Exception:  # noqa: BLE001
            pass

    # ── Confirm ───────────────────────────────────────────────────────────
    if not _confirm(a):
        print("  Aborted (nothing written).")
        return EXIT_CANCELLED

    # ── Provision ─────────────────────────────────────────────────────────
    # Resume-write PRESERVES the existing profile and overlays only the confirmed fields (fix #45); a fresh
    # profile is reconstructed from scratch.
    text = build_resume_profile_text(a, existing_text) if existing_text is not None else build_profile_text(a)
    if dry_run:
        print("\n── Provision (--dry-run — writing nothing) ────────────────────────")
        print(text)
        print("  (dry-run) re-run without --dry-run to write + prove run-ready.")
        return EXIT_OK
    print("\n── Provision ──────────────────────────────────────────────────────")
    pi.write_profile(profile_path, text)
    print(f"✓ Wrote {profile_path}  (mode 0600, schema_version={pr.PROFILE_SCHEMA_VERSION})")

    # ── Done: run-ready proof ─────────────────────────────────────────────
    prof = pr._read_env(profile_path)
    try:
        checks = (battery_fn or cr.run_battery)(prof)
    except Exception as e:  # noqa: BLE001
        print(
            f"  ⚠ readiness battery could not run ({e}) — validate later: "
            f"llmb-k8s profile validate --cluster {cluster}"
        )
        return EXIT_OK
    print("\n" + render_done_panel(cluster, checks))
    ok = _emit_readiness(cluster, checks, profile_path, profiles_dir)
    if ok:
        k8s_config.clear_init_state(cluster)  # success clears THIS cluster's resume entry (not the map)
        # persist stable cluster-agnostic prefs for --express (Q5)
        try:
            k8s_config.save_system_config(
                {
                    "owner": a.owner,
                    "artifacts_sc_hint": a.artifacts_sc,
                    "control_sc_hint": a.control_sc,
                }
            )
        except Exception:  # noqa: BLE001
            pass
        # init→install handoff: the profile is proven run-ready; recipes still need prereqs + model weights.
        # Only prompt/invoke on a REAL run (battery_fn injected == test/offline → print-only, no input/subprocess).
        _offer_install_handoff(cluster, interactive=(battery_fn is None))
    return EXIT_OK if ok else EXIT_ERROR


def _invoke_install(cluster: str) -> None:
    """Hand off to `llmb-k8s install --cluster <name> --from-init` via the dispatcher. install is owned by
    another component — we only invoke the verb, never reimplement it. --from-init tells install that this
    cluster was JUST proven run-ready, so it skips its own profile-review prompt and lands directly on the
    recipe selector. Best-effort: a launch hiccup prints the command to run by hand rather than failing the
    (already-successful) init."""
    import subprocess

    disp = ROOT / "scripts" / "llmb-k8s"
    try:
        subprocess.run([sys.executable, str(disp), "install", "--cluster", cluster, "--from-init"])
    except Exception as e:  # noqa: BLE001
        print(f"  (couldn't launch install automatically: {e}) — run: llmb-k8s install --cluster {cluster}")


def _offer_install_handoff(cluster: str, *, interactive: bool = True, invoke: Optional[Callable] = None) -> None:
    """init→install handoff. The profile is proven run-ready, so a real interactive init flows STRAIGHT into
    the install stage's recipe selector (the selector's 'empty to skip' is the natural exit) — no redundant
    "set up recipes now? [Y/n]" gate. Does NOT build install (another component owns it) — it invokes the
    verb with --from-init. In non-interactive/offline mode (test/--play) it only PRINTS the next command.
    """
    cmd = f"llmb-k8s install --cluster {cluster}"
    print("\n── Next: set up recipes on this cluster ───────────────────────────")
    print(f"  Downloads model weights + prereqs onto the cluster:  {cmd}")
    if not interactive:
        return
    print("  Continuing to recipe selection…")
    (invoke or _invoke_install)(cluster)


def _detect_cni(krun) -> str:
    """AUDIT ANNOTATION ONLY (§6.1) — record the CNI/datapath; it does NOT select the written IP. Cheapest
    first: cilium-config kube-proxy-replacement; safe-degrade to '' (unknown) on any error/RBAC.
    """
    try:
        rc, out, _ = krun(
            [
                "-n",
                "kube-system",
                "get",
                "cm",
                "cilium-config",
                "-o",
                "jsonpath={.data.kube-proxy-replacement}",
            ]
        )
        if rc == 0 and (out or "").strip():
            return f"Cilium (kube-proxy-replacement={out.strip()})"
    except Exception:  # noqa: BLE001
        pass
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _usage() -> str:
    return (
        "usage: llmb-k8s init [--cluster <label>] [--play <f> | --record <f> | --express] "
        "[--reset] [--dry-run|-n] [-v] [-d]\n"
        "\n"
        "  Run `llmb-k8s init` with NO flags to DISCOVER the connected clusters (kubectl contexts) and\n"
        "  pick one — you do not need to know a cluster/profile name up front.\n"
        "\n"
        "  --cluster <label>  Optional. The PROFILE label (the on-disk cluster-profiles/<label>.env name) —\n"
        "                     NOT a cluster you must already know. Use it for an explicit / non-interactive\n"
        "                     init, a resume/edit of an existing profile, or with --play/--record."
    )


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cluster = None
    mode = "interactive"
    playfile = None
    dry_run = False
    verbose = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-h", "--help"):
            print(_usage())
            return EXIT_OK
        if a == "--cluster" and i + 1 < len(argv):
            cluster = argv[i + 1]
            i += 2
            continue
        if a in ("--play", "--record") and i + 1 < len(argv):
            mode = a.lstrip("-")
            playfile = argv[i + 1]
            i += 2
            continue
        if a == "--express":
            mode = "express"
            i += 1
            continue
        if a in ("--dry-run", "-n"):
            dry_run = True
            i += 1
            continue
        if a == "--reset":
            # Discard the saved wizard state so init re-asks EVERYTHING. Deleting the profile alone does
            # not do this — the state lives outside the repo and otherwise replays for 7 days.
            import k8s_config as _kc

            _sp = _kc.init_state_path()
            try:
                _sp.unlink()
                print(f"llmb-k8s init: cleared saved wizard state ({_sp}) — init will re-ask every question.")
            except FileNotFoundError:
                print("llmb-k8s init: no saved wizard state to clear — init already starts fresh.")
            except OSError as e:
                print(f"llmb-k8s init: could not clear {_sp}: {e}")
                return EXIT_ERROR
            # --reset is a discrete maintenance action: clear and EXIT. Falling through into the
            # interactive wizard would block on stdin for anyone scripting it.
            print("  Now run:  llmb-k8s init")
            return EXIT_OK
        if a in ("--verbose", "-v"):
            verbose = True
            i += 1
            continue
        if a in ("--dev-mode", "-d"):
            # SLURM parity surface only — near-no-op for k8s (no repo copy to skip); NOT a dry-run (§2/M1).
            i += 1
            continue
        if not a.startswith("-") and cluster is None:
            cluster = a
            i += 1
            continue
        print(f"llmb-k8s init: unrecognized argument '{a}'\n{_usage()}")
        return EXIT_ERROR

    # interactive with NO --cluster is the fresh front door: run_interactive(None) discovers + picks. express
    # (Phase-1 stub) still needs a name; play/record take the name from the playfile / --cluster.
    if not cluster and mode == "express":
        print(_usage())
        return EXIT_ERROR

    try:
        if mode == "play":
            return run_play(cluster or "", playfile, dry_run=dry_run, verbose=verbose)
        if mode == "record":
            return run_record(cluster or "", playfile, dry_run=dry_run, verbose=verbose)
        if mode == "express":
            return run_express(cluster or "", dry_run=dry_run, verbose=verbose)
        return run_interactive(cluster, dry_run=dry_run, verbose=verbose)
    except NotImplementedError as e:
        print(str(e))
        return EXIT_ERROR
    except (KeyboardInterrupt, EOFError):
        # EOFError: no-flag discovery reached the picker with no interactive stdin (e.g. piped/non-tty) —
        # exit cleanly cancelled rather than dumping a traceback.
        print("\n  Cancelled.")
        return EXIT_CANCELLED


if __name__ == "__main__":
    raise SystemExit(main())

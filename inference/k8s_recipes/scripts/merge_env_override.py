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

"""Apply container environment overrides to manifests read from standard input.

Overrides apply only to workload containers and leave committed rendered files unchanged. Patched resources receive variant labels and an annotation containing the normalized override set. Runs with that marker cannot be published as pinned results. Invalid input exits non-zero before apply.

Modes:
  (default)      patch stdin → stdout
  --variant-id   print the 8-hex variant id (nothing, exit 1, when no override is set)
  --json         print the canonical overrides JSON  {"set": {...}, "unset": [...], "variant_id": "..."}
  --describe     print a one-line human summary  (e.g. `-FOO -BAR +BAZ=1`)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "Job", "ReplicaSet", "Pod"}
LABEL_VARIANT = "llmb.nvidia.com/variant"
LABEL_VARIANT_ID = "llmb.nvidia.com/variant-id"
ANNOT_OVERRIDES = "llmb.nvidia.com/env-overrides"


def _lines(name: str) -> list[str]:
    return [ln.strip() for ln in (os.environ.get(name) or "").splitlines() if ln.strip()]


def parse_spec(set_lines: list[str], unset_lines: list[str]) -> dict:
    """PURE: the raw --env-set/--env-unset lines → the canonical override spec. Raises ValueError on a
    malformed entry (fail-closed: a typo'd override must not quietly become a pinned-config run).
    """
    env_set: dict[str, str] = {}
    for ln in set_lines:
        if "=" not in ln:
            raise ValueError(f"--env-set expects KEY=VALUE, got {ln!r}")
        k, v = ln.split("=", 1)
        k = k.strip()
        if not _KEY_RE.match(k):
            raise ValueError(f"--env-set: {k!r} is not a valid environment variable name")
        env_set[k] = v
    env_unset: list[str] = []
    for k in unset_lines:
        if not _KEY_RE.match(k):
            raise ValueError(f"--env-unset: {k!r} is not a valid environment variable name")
        if k not in env_unset:
            env_unset.append(k)
    both = sorted(set(env_set) & set(env_unset))
    if both:
        raise ValueError(f"{', '.join(both)} passed to BOTH --env-set and --env-unset (ambiguous)")
    spec = {"set": dict(sorted(env_set.items())), "unset": sorted(env_unset)}
    spec["variant_id"] = variant_id(spec)
    return spec


def variant_id(spec: dict) -> str:
    """A stable 8-hex digest of the override SET (order-independent) — a k8s-label-safe variant handle."""
    canon = json.dumps(
        {"set": spec.get("set") or {}, "unset": sorted(spec.get("unset") or [])},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def describe(spec: dict) -> str:
    """A one-line human summary for banners/logs: `-FOO -BAR +BAZ=1`."""
    parts = [f"-{k}" for k in spec.get("unset") or []]
    parts += [f"+{k}={v}" for k, v in (spec.get("set") or {}).items()]
    return " ".join(parts)


def spec_from_env() -> dict | None:
    """The override spec from LLMB_ENV_SET / LLMB_ENV_UNSET, or None when no override is in play."""
    s, u = _lines("LLMB_ENV_SET"), _lines("LLMB_ENV_UNSET")
    if not s and not u:
        return None
    return parse_spec(s, u)


# ── manifest patching ────────────────────────────────────────────────────────────────────────────────
def _pod_spec(doc: dict):
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        return None
    if doc.get("kind") == "Pod":
        return spec
    tmpl = spec.get("template")
    if isinstance(tmpl, dict) and isinstance(tmpl.get("spec"), dict):
        return tmpl["spec"]
    return None


def _patch_container(c: dict, spec: dict, hits: dict) -> bool:
    """Upsert spec['set'] / drop spec['unset'] on ONE container's env list. Returns True if changed."""
    env = c.get("env")
    if env is None:
        env = []
    if not isinstance(env, list):
        return False
    changed = False
    for k in spec.get("unset") or []:
        keep = [e for e in env if not (isinstance(e, dict) and e.get("name") == k)]
        if len(keep) != len(env):
            env, changed = keep, True
            hits[k] = hits.get(k, 0) + 1
    for k, v in (spec.get("set") or {}).items():
        found = False
        for e in env:
            if isinstance(e, dict) and e.get("name") == k:
                e.pop("valueFrom", None)  # a literal override replaces any secret/field binding
                if e.get("value") != v:
                    changed = True
                e["value"] = v
                found = True
        if not found:
            env.append({"name": k, "value": v})
            changed = True
        hits[k] = hits.get(k, 0) + 1
    if changed or c.get("env") is not None:
        c["env"] = env
    return changed


def _mark(meta_owner: dict, spec: dict, canon: str) -> None:
    """Stamp the variant labels + the exact-overrides annotation onto one metadata block."""
    meta = meta_owner.setdefault("metadata", {})
    if not isinstance(meta, dict):
        return
    labels = meta.setdefault("labels", {})
    if isinstance(labels, dict):
        labels[LABEL_VARIANT] = "true"
        labels[LABEL_VARIANT_ID] = spec["variant_id"]
    annots = meta.setdefault("annotations", {})
    if isinstance(annots, dict):
        annots[ANNOT_OVERRIDES] = canon


def patch_docs(docs: list, spec: dict) -> tuple[int, dict]:
    """Apply the override to every workload doc. Returns (objects_patched, per-key container hit counts)."""
    canon = json.dumps(
        {"set": spec.get("set") or {}, "unset": spec.get("unset") or []},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    hits: dict = {}
    n = 0
    for doc in docs:
        if not isinstance(doc, dict) or doc.get("kind") not in _WORKLOAD_KINDS:
            continue
        ps = _pod_spec(doc)
        if ps is None:
            continue
        containers = ps.get("containers")
        if not isinstance(containers, list):
            continue
        touched = False
        for c in containers:
            if isinstance(c, dict) and _patch_container(c, spec, hits):
                touched = True
        # MARK every workload object in a variant deploy — even one whose containers happened not to change —
        # so nothing in the stack reads as a clean pinned-config object.
        _mark(doc, spec, canon)
        tmpl = (doc.get("spec") or {}).get("template")
        if isinstance(tmpl, dict):
            _mark(tmpl, spec, canon)
        n += 1
        _ = touched
    return n, hits


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        spec = spec_from_env()
    except ValueError as e:
        print(f"merge_env_override: {e}", file=sys.stderr)
        return 2

    if mode == "--variant-id":
        if not spec:
            return 1
        print(spec["variant_id"])
        return 0
    if mode == "--json":
        if not spec:
            return 1
        print(json.dumps(spec, sort_keys=True))
        return 0
    if mode == "--describe":
        if not spec:
            return 1
        print(describe(spec))
        return 0

    content = sys.stdin.read()
    if not spec:  # no override → byte-identical passthrough
        sys.stdout.write(content)
        return 0
    try:
        import yaml
    except ImportError:
        print(
            "merge_env_override: PyYAML is required to apply an env override — refusing to apply the "
            "UNMODIFIED manifests (a silent pinned-config run must never masquerade as a variant)",
            file=sys.stderr,
        )
        return 2
    try:
        docs = [d for d in yaml.safe_load_all(content) if d is not None]
        n, hits = patch_docs(docs, spec)
        out = yaml.dump_all(docs, default_flow_style=False, allow_unicode=True)
    except Exception as e:  # FAIL-CLOSED — see the module docstring
        print(
            f"merge_env_override: REFUSING to apply — could not patch the manifests ({e}). "
            "Nothing was applied; the pinned config was NOT silently used.",
            file=sys.stderr,
        )
        return 2
    missed = [k for k in (spec.get("unset") or []) if not hits.get(k)]
    print(
        f"merge_env_override: VARIANT {spec['variant_id']} — {describe(spec)} "
        f"→ {n} object(s) patched + marked (llmb.nvidia.com/variant=true); NOT publishable",
        file=sys.stderr,
    )
    if missed:
        print(
            f"merge_env_override: WARN — --env-unset {' '.join(missed)} matched no container env entry "
            "(already absent, or set via args/ConfigMap rather than env:)",
            file=sys.stderr,
        )
    sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

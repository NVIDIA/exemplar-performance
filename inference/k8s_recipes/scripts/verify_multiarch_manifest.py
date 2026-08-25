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

"""verify_multiarch_manifest.py [<manifest>] — validate a task image-digest-manifest is MULTI-ARCH.

Standard going forward (Option A): ONE fused OCI index per task image; kubelet auto-selects the per-node
arch; the manifest keeps its v1 shape but points every row at the INDEX digest and drops the `-arm64` repo
suffix. B200 (x86) + GB200/GB300 (arm64) then share ONE arch-neutral manifest. This tool is how we validate
the operator's future dual-arch build and lint future manifests in CI.

Two modes:
  ONLINE  (default): resolve each row's `image_digest_ref` with `docker buildx imagetools inspect --raw`
                     (read-only; nvcr Tier-2 reads work unauthenticated) and report which images are a real
                     multi-arch OCI index (linux/amd64 + linux/arm64) vs still single-arch.
  --offline (--lint): NO registry access. Check manifest SHAPE only — arch-neutral repo (no -arm64/-amd64
                     suffix), neutral top-level `platform`, exactly one index digest per row. Safe for CI:
                     a PURE legacy single-arch manifest passes as an advisory (multi-arch swap pending),
                     a fully-migrated manifest passes clean, and a HALF-migrated (partial) manifest FAILS.

Usage:
  verify_multiarch_manifest.py                                 online-inspect the default task manifest
  verify_multiarch_manifest.py <manifest> --offline            shape-lint (CI); advisory-pass on legacy
  verify_multiarch_manifest.py <manifest> --offline --require-multiarch   fail unless fully arch-neutral
  verify_multiarch_manifest.py <manifest> --allow-single       online, but don't fail on single-arch rows
  verify_multiarch_manifest.py <manifest> --json               machine-readable summary on stdout

Exit: 0 = OK (or advisory legacy pass); 1 = violation / single-arch found; 2 = usage/parse error;
      3 = online mode requested but docker buildx unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "serving/tasks/image-digest-manifest.json"

# An arch baked into a repo NAME (e.g. …/task-image-arm64@sha256:…) — the thing a fused index drops.
# Matches the arch token as a trailing segment of the repo path (before the @digest or :tag).
_ARCH_TOKENS = (
    "arm64",
    "amd64",
    "x86_64",
    "x86-64",
    "aarch64",
    "arm",
    "amd",
    "ppc64le",
    "s390x",
)
_ARCH_SUFFIX_RE = re.compile(r"-(" + "|".join(_ARCH_TOKENS) + r")(?=$|[@:/])")
# A neutral top-level `platform` value (or absence) — anything else (e.g. "linux/arm64") is single-arch.
_NEUTRAL_PLATFORM = {None, "", "multi", "multiarch", "multi-arch", "index", "oci-index"}
# Ref fields on each row that must go arch-neutral under Option A.
_REF_FIELDS = ("image_digest_ref", "image_ref", "build_result_ref")


def _repo_of(ref: str) -> str:
    """The repository portion of an image ref (strip @digest and :tag, keep registry/path)."""
    if not ref:
        return ""
    ref = ref.split("@", 1)[0]
    # a tag is the last ':' AFTER the last '/'; a registry ':port' has no '/' after it
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    if colon > slash:
        ref = ref[:colon]
    return ref


def arch_suffix(ref_or_repo: str) -> str | None:
    """Return the arch token baked into a repo name, or None if arch-neutral."""
    m = _ARCH_SUFFIX_RE.search(_repo_of(ref_or_repo))
    return m.group(1) if m else None


def _digest_count(ref: str) -> int:
    """How many @sha256: digests the ref carries (a well-formed row pins exactly one index digest)."""
    return len(re.findall(r"@sha256:[0-9a-f]{64}", ref or ""))


def shape_report(manifest: dict) -> dict:
    """PURE offline shape analysis (no registry). Returns {mode, violations, advisories, rows, counts}.

    mode:
      'multiarch'      every ref arch-neutral, neutral top-level platform, one index digest per row.
      'legacy-single'  every ref arch-suffixed AND a single-arch top-level platform (a consistent v1 arm64
                       manifest) — the pre-migration state; advisory-passes so today's CI stays green.
      'partial'        a MIX (some refs neutral, some arch-suffixed, or neutral refs with a single-arch
                       platform) — a half-done migration; always a hard failure.
      'empty'          no rows.
    """
    violations: list[str] = []
    advisories: list[str] = []
    tasks = manifest.get("tasks") or []

    top_repo = manifest.get("repository") or ""
    top_platform = manifest.get("platform")
    top_repo_arch = arch_suffix(top_repo)
    top_platform_neutral = top_platform in _NEUTRAL_PLATFORM

    arch_hits = 0  # refs/repos that still carry an arch suffix
    neutral_hits = 0
    single_platform_rows = 0

    rows_info = []
    for i, t in enumerate(tasks):
        row = {
            "index": i,
            "task_ref": t.get("task_ref"),
            "role": t.get("role"),
            "arch_refs": [],
            "issues": [],
        }
        for f in _REF_FIELDS:
            ref = t.get(f)
            if not ref:
                continue
            sfx = arch_suffix(ref)
            if sfx:
                arch_hits += 1
                row["arch_refs"].append(f"{f}=…-{sfx}")
            else:
                neutral_hits += 1
        idr = t.get("image_digest_ref") or ""
        if not idr:
            row["issues"].append("missing image_digest_ref")
        elif _digest_count(idr) != 1:
            row["issues"].append(
                f"image_digest_ref must pin exactly one @sha256 index digest (found {_digest_count(idr)})"
            )
        rp = t.get("platform")
        if rp not in _NEUTRAL_PLATFORM:
            single_platform_rows += 1
            row["single_platform"] = rp
        rows_info.append(row)

    # Whole-manifest arch signals (top-level repo + platform folded in).
    top_arch_signal = bool(top_repo_arch) or (top_platform is not None and not top_platform_neutral)
    top_neutral_signal = (top_repo and not top_repo_arch) or top_platform_neutral

    if not tasks:
        mode = "empty"
    elif arch_hits == 0 and single_platform_rows == 0 and not top_arch_signal:
        mode = "multiarch"
    elif neutral_hits == 0 and top_repo_arch and not top_platform_neutral:
        # every ref arch-suffixed AND a single-arch top-level platform → a clean, consistent legacy manifest.
        mode = "legacy-single"
    else:
        mode = "partial"

    # Violations (hard) vs advisories (soft) depend on mode.
    if mode == "multiarch":
        for r in rows_info:
            for iss in r["issues"]:
                violations.append(f"row {r['index']} ({r['task_ref']}): {iss}")
    elif mode == "legacy-single":
        advisories.append(
            f"Legacy single-architecture manifest (repository={top_repo!r}, platform={top_platform!r}): "
            f"{arch_hits} architecture-specific references across {len(tasks)} rows."
        )
    elif mode == "partial":
        violations.append(
            f"PARTIAL/inconsistent migration: {neutral_hits} arch-neutral + {arch_hits} arch-suffixed refs"
            + (f", top-level platform='{top_platform}'" if not top_platform_neutral else "")
            + (f", top-level repository='{top_repo}' still arch-suffixed" if top_repo_arch else "")
            + f", {single_platform_rows} rows with a single-arch platform."
        )
        for r in rows_info:
            if r["arch_refs"]:
                violations.append(f"row {r['index']} ({r['task_ref']}): still arch-pinned {', '.join(r['arch_refs'])}")
            for iss in r["issues"]:
                violations.append(f"row {r['index']} ({r['task_ref']}): {iss}")

    return {
        "mode": mode,
        "violations": violations,
        "advisories": advisories,
        "rows": rows_info,
        "counts": {
            "tasks": len(tasks),
            "arch_refs": arch_hits,
            "neutral_refs": neutral_hits,
            "single_platform_rows": single_platform_rows,
        },
        "top": {"repository": top_repo, "platform": top_platform},
    }


# ---- online (registry) inspection --------------------------------------------------------------------

_INDEX_MEDIA = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}


def _buildx_available() -> bool:
    try:
        subprocess.run(["docker", "buildx", "version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def platforms_of_raw(raw: dict) -> set[str]:
    """The real (os/arch) platforms an inspected index advertises, skipping attestation/unknown manifests."""
    plats: set[str] = set()
    if raw.get("mediaType") not in _INDEX_MEDIA:
        return plats  # a bare image manifest is single-arch by construction (its own platform only)
    for m in raw.get("manifests") or []:
        p = m.get("platform") or {}
        arch = p.get("architecture")
        os_ = p.get("os")
        ann = m.get("annotations") or {}
        if arch in (None, "unknown") or os_ in (None, "unknown"):
            continue
        if ann.get("vnd.docker.reference.type") == "attestation-manifest":
            continue
        plats.add(f"{os_}/{arch}")
    return plats


def inspect_ref(ref: str) -> dict:
    """Resolve one image ref → {ref, ok, is_index, platforms, error}. Read-only registry call."""
    try:
        cp = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", ref, "--raw"],
            capture_output=True,
            text=True,
        )
    except OSError as e:
        return {"ref": ref, "ok": False, "error": str(e)}
    if cp.returncode != 0:
        lines = (cp.stderr or cp.stdout).strip().splitlines()
        return {
            "ref": ref,
            "ok": False,
            "error": lines[-1] if lines else "inspect failed",
        }
    try:
        raw = json.loads(cp.stdout)
    except json.JSONDecodeError as e:
        return {"ref": ref, "ok": False, "error": f"non-JSON inspect output: {e}"}
    plats = sorted(platforms_of_raw(raw))
    return {
        "ref": ref,
        "ok": True,
        "is_index": raw.get("mediaType") in _INDEX_MEDIA,
        "platforms": plats,
    }


def online_report(manifest: dict) -> dict:
    """Inspect each UNIQUE image_digest_ref in the registry; classify multi-arch vs single-arch vs missing."""
    refs, seen = [], set()
    for t in manifest.get("tasks") or []:
        r = t.get("image_digest_ref")
        if r and r not in seen:
            seen.add(r)
            refs.append(r)
    results = [inspect_ref(r) for r in refs]
    multiarch, single, missing = [], [], []
    for res in results:
        if not res.get("ok"):
            missing.append(res)
        elif len(res.get("platforms") or []) >= 2 and res.get("is_index"):
            multiarch.append(res)
        else:
            single.append(res)
    return {
        "unique": len(refs),
        "multiarch": multiarch,
        "single": single,
        "missing": missing,
    }


# ---- CLI ---------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a task image-digest-manifest is multi-arch.")
    ap.add_argument(
        "manifest",
        nargs="?",
        default=str(DEFAULT_MANIFEST),
        help="path to image-digest-manifest.json",
    )
    ap.add_argument(
        "--offline",
        "--lint",
        dest="offline",
        action="store_true",
        help="shape-only check (no registry access); advisory-pass on a legacy single-arch manifest",
    )
    ap.add_argument(
        "--require-multiarch",
        action="store_true",
        help="offline: FAIL (not advise) if the manifest is still legacy single-arch",
    )
    ap.add_argument(
        "--allow-single",
        action="store_true",
        help="online: report but don't fail on single-arch rows",
    )
    ap.add_argument("--json", action="store_true", help="emit a machine-readable JSON summary")
    args = ap.parse_args()

    path = Path(args.manifest)
    if not path.is_file():
        print(f"verify-multiarch: no manifest at {path}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"verify-multiarch: {path} is not valid JSON — {e}", file=sys.stderr)
        return 2

    if args.offline:
        rep = shape_report(manifest)
        if args.json:
            print(json.dumps(rep, indent=2))
        else:
            print(f"[shape] {path}")
            print(f"[shape] mode={rep['mode']}  {rep['counts']}")
            for a in rep["advisories"]:
                print(f"  ADVISORY  {a}")
            for v in rep["violations"]:
                print(f"  VIOLATION {v}")
        if rep["violations"]:
            print("verify-multiarch: SHAPE FAIL", file=sys.stderr)
            return 1
        if rep["mode"] == "legacy-single" and args.require_multiarch:
            print(
                "verify-multiarch: --require-multiarch set but manifest is still legacy single-arch",
                file=sys.stderr,
            )
            return 1
        print(f"verify-multiarch: shape OK ({rep['mode']})")
        return 0

    # online
    if not _buildx_available():
        print(
            "verify-multiarch: online mode needs `docker buildx` (use --offline for shape-only CI lint)",
            file=sys.stderr,
        )
        return 3
    rep = online_report(manifest)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(f"[inspect] {path}: {rep['unique']} unique image_digest_ref(s)")
        print(
            f"[inspect] multi-arch: {len(rep['multiarch'])}  single-arch: {len(rep['single'])}  "
            f"unresolved: {len(rep['missing'])}"
        )
        for res in rep["single"]:
            print(f"  SINGLE-ARCH {res['ref']}  platforms={res.get('platforms') or ['<bare-manifest>']}")
        for res in rep["missing"]:
            print(f"  UNRESOLVED  {res['ref']}  ({res.get('error')})")
        for res in rep["multiarch"]:
            print(f"  multi-arch  {res['ref']}  platforms={res['platforms']}")
    bad = len(rep["missing"]) + (0 if args.allow_single else len(rep["single"]))
    if bad:
        print(
            f"verify-multiarch: {bad} row(s) not multi-arch / unresolved",
            file=sys.stderr,
        )
        return 1
    print(f"verify-multiarch: all {rep['unique']} image(s) are multi-arch OCI indices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

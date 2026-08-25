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

"""Build a model-centric inventory from the recipe catalog and Kubernetes cache claims.

Rows distinguish verified, present, downloading, missing, failed, and unread states. A download-job stamp
is label-based evidence and is reported as attested. An in-volume sentinel is direct evidence and is reported
as verified. This inventory does not currently produce sentinel evidence. Claim existence alone does not
establish that model weights are complete.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_cache import resolve_cache_claim  # noqa: E402

# Labels written by successful model-download Jobs.
L_MODEL_NAME = "llmb.nvidia.com/model-name"  # single-valued: MERGE-patched, so LAST model wins
L_MODEL_REVISION = "llmb.nvidia.com/model-revision"  # single-valued revision prefix
L_DOWNLOAD_COMPLETE = "llmb.nvidia.com/download-complete"
MODEL_LABEL_PREFIX = "llmb.nvidia.com/model."  # ACCUMULATING per-model key → <rev12>
# Per-model label keys follow the writer's 63-character key limit.
MODEL_LABEL_KEY_CAP = 63
MODEL_LABEL_NAME_CAP = MODEL_LABEL_KEY_CAP - len(MODEL_LABEL_PREFIX)  # 41 chars of model name survive

# Evidence grades used by reconciliation and rendering.
GRADE_SENTINEL = "sentinel"
GRADE_JOB_STAMP = "job-stamp"
GRADE_JOB_UNPINNED = "job-unpinned"
GRADE_HAND_STAMP = "hand-stamp"
GRADE_WRONG_REV = "wrong-revision"
GRADE_NO_REV = "no-revision"
GRADE_CLAIM_ONLY = "claim-only"
GRADE_NONE = "none"

# Only sentinel and download-Job evidence can verify a model.
VERIFIED_GRADES = frozenset({GRADE_SENTINEL, GRADE_JOB_STAMP})

# Rank evidence used to select among candidate claims.
GRADE_RANK = {
    GRADE_SENTINEL: 6,
    GRADE_JOB_STAMP: 5,
    GRADE_JOB_UNPINNED: 4,
    GRADE_WRONG_REV: 3,
    GRADE_NO_REV: 3,
    GRADE_HAND_STAMP: 2,
    GRADE_CLAIM_ONLY: 0,
    GRADE_NONE: 0,
}
RANK_UNREADABLE = 1  # unread claims rank below revision-bearing evidence

# User-facing causes for an unread claim.
UNKNOWN_CAUSES = {
    "unread": "PVC read did not land (RBAC-forbidden, timed out, or the call failed) — "
    "check: kubectl auth can-i list pvc",
    "rbac": "PVC read FORBIDDEN — check: kubectl auth can-i list pvc",
    "fast": "--fast skipped the PVC read — drop --fast to inventory model caches",
    "unschedulable": "no node could run the read (unschedulable) — claim contents unread",
    "cannot-mount": "the claim would not mount where the read ran (RWX/NFS) — claim contents unread",
}
UNKNOWN_FALLBACK = "claim contents unread (cause not recorded)"

# A missing cache claim is a profile configuration error.
NO_CLAIM_EVIDENCE = "no claim configured in the cluster profile"

# Sort urgent states first while keeping shared-claim rows together.
STATE_RANK = {
    "failed": 0,
    "unknown": 1,
    "missing": 2,
    "downloading": 3,
    "present": 4,
    "verified": 5,
}
STATE_GLYPH = {
    "verified": "✓",
    "present": "~",
    "downloading": "↓",
    "missing": "○",
    "failed": "✗",
    "unknown": "?",
}


def label_slug(name) -> str:
    """Normalize a model name for cache labels."""
    return re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")


def model_label_key(model) -> str:
    """Return the bounded per-model cache label key, or an empty string."""
    slug = label_slug(model)
    if not slug:
        return ""
    return (MODEL_LABEL_PREFIX + slug)[:MODEL_LABEL_KEY_CAP]


def models_in_labels(labels) -> dict:
    """Return the per-model revision labels stored on a claim."""
    out = {}
    for k, v in (labels or {}).items():
        if str(k).startswith(MODEL_LABEL_PREFIX):
            slug = str(k)[len(MODEL_LABEL_PREFIX) :]
            if slug:
                out[slug] = str(v or "")
    return out


def _slug_matches(key_slug: str, model_slug: str) -> bool:
    """Match an exact model slug or a writer-truncated label slug."""
    if not key_slug or not model_slug:
        return False
    if key_slug == model_slug:
        return True
    return len(key_slug) >= MODEL_LABEL_NAME_CAP and model_slug.startswith(key_slug)


def unknown_cause(kind: str) -> str:
    """Return an actionable explanation for an unread claim."""
    return UNKNOWN_CAUSES.get(str(kind or ""), UNKNOWN_FALLBACK)


def expected_models(catalog, prof, pins=None) -> list:
    """Resolve the catalog models, cache claims, and pinned revisions for a cluster."""
    profs = [prof] if isinstance(prof, dict) else [p for p in (prof or []) if isinstance(p, dict)]
    by_model: dict = {}
    for cell in catalog or []:
        m = str((cell or {}).get("model") or "").strip()
        if not m:
            continue
        by_model.setdefault(m, 0)
        by_model[m] += 1
    out = []
    for m in sorted(by_model):
        claims, source = [], ""
        for p in profs:
            claim, src = resolve_cache_claim({"model": m}, p)
            if claim and claim not in claims:
                claims.append(claim)
                source = source or src
        out.append(
            {
                "model": m,
                "claim": claims[0] if claims else "",
                "claims": claims,
                "claim_source": source,
                "pinned_rev": str((pins or {}).get(m) or ""),
                "cells": by_model[m],
            }
        )
    return out


REV_DISPLAY = 12  # display width for revision prefixes


def short_rev(rev, pinned="") -> tuple:
    """Return a fixed-width revision and flag a mismatch with the pinned revision."""
    r, p = str(rev or "").strip(), str(pinned or "").strip()
    if not p:
        return (r[:REV_DISPLAY], "")
    if not r:
        return (p[:REV_DISPLAY], "")
    if r == p[:REV_DISPLAY] or p.startswith(r) or r.startswith(p):
        return (p[:REV_DISPLAY], "")
    return (r[:REV_DISPLAY], "WRONG REVISION")


_rev12 = short_rev  # internal alias (this module's call sites read better short)


def stamp_provenance(labels, pinned="") -> str:
    """Classify a completion stamp as download-Job, manual, or unknown evidence."""
    lab = labels or {}
    if lab.get(L_DOWNLOAD_COMPLETE) != "true":
        return "unknown"  # not stamped at all — nothing to attribute
    name = str(lab.get(L_MODEL_NAME) or "").strip()
    rev = str(lab.get(L_MODEL_REVISION) or "").strip()
    if not name:
        return "hand"
    if rev and len(rev) != REV_DISPLAY:
        return "hand"
    p = str(pinned or "").strip()
    if p and rev == p[:REV_DISPLAY]:
        return "job"
    return "unknown"


def _certify(raw_rev, pinned: str, *, witness: str, model: str, strong: str) -> tuple:
    """Convert revision-bearing evidence into an inventory grade."""
    raw = str(raw_rev or "").strip()
    p = str(pinned or "").strip()
    if not raw:
        return (
            GRADE_NO_REV,
            p[:REV_DISPLAY],
            f"{witness} names {model} but records NO revision — nothing states which weights are there",
        )
    rev, wrong = short_rev(raw, p)
    if wrong:
        return (
            GRADE_WRONG_REV,
            rev,
            f"{witness} names {model} · {wrong} (pinned @{p[:REV_DISPLAY]})",
        )
    if p:
        return (strong, rev, f"{witness} · matches pin")
    return (
        GRADE_JOB_UNPINNED,
        rev,
        f"{witness} · no pinned revision to check it against",
    )


def _names_model(fact: dict, slug: str) -> bool:
    """Return whether the claim labels identify this model (per-model or single-valued
    pair)? Attribution only — says nothing about whether the contents can be certified.
    """
    lab = fact.get("labels") or {}
    if any(_slug_matches(k, slug) for k in models_in_labels(lab)):
        return True
    return label_slug(lab.get(L_MODEL_NAME) or "") == slug


def _claim_evidence(fact: dict, model: str, pinned: str) -> tuple:
    """Return the evidence grade, revision, and explanation for a model and claim.

    Order matters: the strongest witness that actually names THIS model wins, and nothing that merely names
    completion evidence)."""
    lab = fact.get("labels") or {}
    slug = label_slug(model)

    # Prefer an in-volume sentinel when available.
    sent = fact.get("sentinels") or {}
    if slug in {label_slug(s) for s in sent}:
        srev = next((v for k, v in sent.items() if label_slug(k) == slug), "")
        return _certify(
            srev,
            pinned,
            witness="sentinel in the claim",
            model=model,
            strong=GRADE_SENTINEL,
        )

    # Otherwise use the accumulating per-model download stamp.
    for key_slug, val in models_in_labels(lab).items():
        if not _slug_matches(key_slug, slug):
            continue
        return _certify(
            val,
            pinned,
            witness="download-Job stamp",
            model=model,
            strong=GRADE_JOB_STAMP,
        )

    # Fall back to the single-valued completion stamp.
    if lab.get(L_DOWNLOAD_COMPLETE) == "true":
        st_model, st_rev = str(lab.get(L_MODEL_NAME) or ""), str(lab.get(L_MODEL_REVISION) or "")
        if stamp_provenance(lab, pinned) == "hand":
            tell = (
                "names no model"
                if not st_model
                else f"revision is {len(st_rev)} chars, a download Job writes {REV_DISPLAY}"
            )
            return (
                GRADE_HAND_STAMP,
                _rev12(st_rev, pinned)[0],
                f"stamp NOT written by a download Job — it {tell}",
            )
        if label_slug(st_model) == slug:
            return _certify(
                st_rev,
                pinned,
                witness="download-Job stamp",
                model=model,
                strong=GRADE_JOB_STAMP,
            )
        # A stamp for another model does not verify this one.
        return (
            GRADE_CLAIM_ONLY,
            _rev12("", pinned)[0],
            f"no stamp names this model in the claim (its only stamp names {st_model})",
        )

    # A retained completed download Job is the final evidence source.
    job = fact.get("job") or {}
    if job.get("done") and label_slug(job.get("model") or "") == slug:
        return _certify(
            job.get("rev"),
            pinned,
            witness="download Job Complete",
            model=model,
            strong=GRADE_JOB_STAMP,
        )

    return (
        GRADE_CLAIM_ONLY,
        _rev12("", pinned)[0],
        "no stamp names this model in the claim",
    )


def _pick_claim(model: str, claims, facts, by_name, pinned):
    """Select the candidate claim carrying the strongest evidence for a model."""
    slug = label_slug(model)
    seen, cands = set(), []
    for i, c in enumerate(claims):
        f = by_name.get(c)
        if f is not None and c not in seen:
            seen.add(c)
            cands.append((i, f))
    for k, f in enumerate(facts):
        nm = f.get("name", "")
        if nm not in seen and _names_model(f, slug):
            seen.add(nm)
            cands.append((len(claims) + k, f))
    if not cands:
        return None

    def _score(item):
        i, f = item
        if f.get("unreadable"):
            return (RANK_UNREADABLE, -i)
        return (GRADE_RANK.get(_claim_evidence(f, model, pinned)[0], 0), -i)

    return max(cands, key=_score)[1]


def is_config_blocker(row) -> bool:
    """Return whether a row represents a missing profile cache configuration."""
    return (row or {}).get("evidence") == NO_CLAIM_EVIDENCE


def _row(
    model,
    *,
    state,
    grade,
    rev,
    claim,
    size,
    cells,
    ns,
    evidence,
    source="live",
    catalog=True,
):
    return {
        "model": model,
        "state": state,
        "glyph": STATE_GLYPH[state],
        "grade": grade,
        "rev": rev,
        "claim": claim,
        "size": size,
        "cells": cells,
        "ns": ns,
        "evidence": evidence,
        "source": source,
        "catalog": catalog,
    }


def reconcile(expected, pvc_rows, *, jobs=None, unreadable="", ns="") -> list:
    """Join expected models with claim and download-Job facts to produce inventory rows."""
    read_landed = pvc_rows is not None
    facts = [dict(f) for f in (pvc_rows or [])]
    jobfacts = [dict(j) for j in (jobs or [])]
    for f in facts:  # attach each claim's most informative download Job
        if "job" not in f:
            best = None
            for j in jobfacts:
                if j.get("claim") != f.get("name"):
                    continue
                if best is None or (j.get("done") and not best.get("done")):
                    best = j
            f["job"] = best
    by_name = {f.get("name", ""): f for f in facts}
    rows, seen = [], set()

    for exp in expected or []:
        model = exp["model"]
        pin = str(exp.get("pinned_rev") or "")
        claims = list(exp.get("claims") or ([exp["claim"]] if exp.get("claim") else []))
        seen.add(label_slug(model))
        cells, primary = exp.get("cells", 0), (claims[0] if claims else "")

        if not read_landed:
            rows.append(
                _row(
                    model,
                    state="unknown",
                    grade=GRADE_NONE,
                    rev=pin[:REV_DISPLAY],
                    claim=primary,
                    size="",
                    cells=cells,
                    ns=ns,
                    evidence=unknown_cause(unreadable),
                )
            )
            continue

        fact = _pick_claim(model, claims, facts, by_name, pin)
        if fact is None:
            jb = next(
                (j for j in jobfacts if label_slug(j.get("model") or "") == label_slug(model)),
                None,
            )
            if jb and jb.get("running"):
                rows.append(
                    _row(
                        model,
                        state="downloading",
                        grade=GRADE_NONE,
                        rev=_rev12(jb.get("rev"), pin)[0],
                        claim=jb.get("claim") or primary,
                        size="",
                        cells=cells,
                        ns=ns,
                        evidence="download Job in flight · claim not in the PVC read",
                    )
                )
                continue
            if jb and jb.get("failed"):
                rows.append(
                    _row(
                        model,
                        state="failed",
                        grade=GRADE_NONE,
                        rev=_rev12(jb.get("rev"), pin)[0],
                        claim=jb.get("claim") or primary,
                        size="",
                        cells=cells,
                        ns=ns,
                        evidence="model download Job FAILED",
                    )
                )
                continue
            if jb and jb.get("done"):
                rev, wrong = _rev12(jb.get("rev"), pin)
                rows.append(
                    _row(
                        model,
                        state="present",
                        grade=GRADE_JOB_UNPINNED,
                        rev=rev,
                        claim=jb.get("claim") or primary,
                        size="",
                        cells=cells,
                        ns=ns,
                        evidence=(
                            f"download Job Complete · {wrong}"
                            if wrong
                            else "download Job Complete · the claim itself is not in the PVC read"
                        ),
                    )
                )
                continue
            why = f"claim {primary} does not exist in this namespace" if primary else NO_CLAIM_EVIDENCE
            rows.append(
                _row(
                    model,
                    state="missing",
                    grade=GRADE_NONE,
                    rev=pin[:REV_DISPLAY],
                    claim=primary,
                    size="",
                    cells=cells,
                    ns=ns,
                    evidence=why,
                )
            )
            continue

        rows.append(_fact_row(model, pin, fact, cells, ns, catalog=True))

    # Include visible models that are not present in the catalog.
    for f in facts:
        for key_slug in models_in_labels(f.get("labels")):
            if key_slug in seen:
                continue
            seen.add(key_slug)
            rows.append(_fact_row(key_slug, "", f, 0, ns, catalog=False))
    for j in jobfacts:
        s = label_slug(j.get("model") or "")
        if not s or s in seen:
            continue
        seen.add(s)
        state = (
            "downloading"
            if j.get("running")
            else ("failed" if j.get("failed") else "present" if j.get("done") else "downloading")
        )
        rows.append(
            _row(
                j.get("model"),
                state=state,
                grade=GRADE_JOB_UNPINNED if state == "present" else GRADE_NONE,
                rev=_rev12(j.get("rev"), "")[0],
                claim=j.get("claim") or "",
                size="",
                cells=0,
                ns=ns,
                catalog=False,
                evidence=(
                    "download Job in flight"
                    if state == "downloading"
                    else (
                        "model download Job FAILED"
                        if state == "failed"
                        else "download Job Complete · not a catalog model"
                    )
                ),
            )
        )

    group_rank: dict = {}
    for r in rows:
        k = r["claim"] or "~"
        group_rank[k] = min(group_rank.get(k, 99), STATE_RANK[r["state"]])
    rows.sort(key=lambda r: (group_rank[r["claim"] or "~"], r["claim"] or "~", r["model"]))
    printed = set()
    for r in rows:  # only the first row of a claim quotes its size
        if not r["claim"]:
            continue
        if r["claim"] in printed:
            r["size"] = "shared"
        elif r["size"]:
            printed.add(r["claim"])
    return rows


def _fact_row(model, pin, fact, cells, ns, *, catalog=True):
    """Build one inventory row; claim phase takes precedence over label evidence."""
    phase = str(fact.get("phase") or "")
    job = fact.get("job") or {}
    claim, size = fact.get("name", ""), str(fact.get("size") or "")
    if fact.get("unreadable"):
        return _row(
            model,
            state="unknown",
            grade=GRADE_NONE,
            rev=pin[:REV_DISPLAY],
            claim=claim,
            size=size,
            cells=cells,
            ns=ns,
            evidence=unknown_cause(fact["unreadable"]),
            catalog=catalog,
        )
    if phase != "Bound":
        return _row(
            model,
            state="failed",
            grade=GRADE_NONE,
            rev=pin[:REV_DISPLAY],
            claim=claim,
            size=size,
            cells=cells,
            ns=ns,
            catalog=catalog,
            evidence=f"claim {phase or 'not Bound'} — cannot serve weights",
        )
    if job.get("failed") and label_slug(job.get("model") or "") == label_slug(model):
        return _row(
            model,
            state="failed",
            grade=GRADE_NONE,
            rev=_rev12(job.get("rev"), pin)[0],
            claim=claim,
            size=size,
            cells=cells,
            ns=ns,
            evidence="model download Job FAILED",
            catalog=catalog,
        )
    grade, rev, why = _claim_evidence(fact, model, pin)
    if job.get("running") and label_slug(job.get("model") or "") == label_slug(model) and grade not in VERIFIED_GRADES:
        return _row(
            model,
            state="downloading",
            grade=GRADE_NONE,
            rev=rev,
            claim=claim,
            size=size,
            cells=cells,
            ns=ns,
            evidence="download Job in flight",
            catalog=catalog,
        )
    state = "verified" if grade in VERIFIED_GRADES else "present"
    if job.get("running") and state == "verified":
        why += " · re-downloading"
    return _row(
        model,
        state=state,
        grade=grade,
        rev=rev,
        claim=claim,
        size=size,
        cells=cells,
        ns=ns,
        evidence=why,
        catalog=catalog,
    )


def ledger_counts(rows) -> dict:
    """Return summary counts while keeping unknown, attested, and verified evidence distinct."""
    c = {
        "catalog": 0,
        "extra": 0,
        "verified": 0,
        "present": 0,
        "downloading": 0,
        "missing": 0,
        "failed": 0,
        "unknown": 0,
        "total": 0,
        "attested": 0,
        "sentinel": 0,
    }
    for r in rows or []:
        c["total"] += 1
        c["catalog" if r.get("catalog", True) else "extra"] += 1
        c[r["state"]] += 1
        if r.get("grade") == GRADE_SENTINEL:
            c["sentinel"] += 1
        elif r.get("grade") == GRADE_JOB_STAMP:
            c["attested"] += 1
    return c

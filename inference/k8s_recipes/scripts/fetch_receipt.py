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

"""Read and validate `_fetch_status.json` receipts.

A verified receipt records the files and bytes copied and, when available, reconciles them with the source count. `safe_to_delete()` requires verified, reconciled evidence before the remote copy may be removed. Older or incomplete receipts remain non-destructive.
"""

from __future__ import annotations

import json
from pathlib import Path

RECEIPT_NAME = "_fetch_status.json"

# Bump when the evidence contract changes. A reader that wants evidence requires >= 2.
RECEIPT_VERSION = 2

#: Required evidence fields for a verified receipt.
EVIDENCE_FIELDS = ("receipt_version", "files_written", "bytes_written")

#: Minimum payload required for a non-empty fetch.
MIN_FILES = 1
MIN_BYTES = 4096

ABSENT = "ABSENT"
UNVERIFIED = "UNVERIFIED"
INCOMPLETE = "INCOMPLETE"
VERIFIED = "VERIFIED"


def read(run_dir) -> dict | None:
    """Return the parsed receipt in run_dir, or None if it is absent OR unparseable.

    Unparseable deliberately collapses to None → ABSENT → "no evidence", never to "fine".
    """
    p = Path(run_dir) / RECEIPT_NAME
    if not p.is_file():
        return None
    try:
        r = json.loads(p.read_text())
    except Exception:
        return None
    return r if isinstance(r, dict) else None


def missing_evidence(receipt: dict) -> list[str]:
    """PURE — which EVIDENCE_FIELDS are absent or the wrong type. Empty list ⇒ the receipt can be judged."""
    missing = []
    for f in EVIDENCE_FIELDS:
        v = receipt.get(f)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            missing.append(f)
    if not missing and receipt.get("receipt_version", 0) < RECEIPT_VERSION:
        missing.append(f"receipt_version>={RECEIPT_VERSION}")
    return missing


def verdict(receipt: dict | None) -> tuple[str, str]:
    """PURE — (status, human reason) for a parsed receipt (or None).

    Fails toward *less* authority at every branch: anything ambiguous is UNVERIFIED, anything short is
    INCOMPLETE. VERIFIED is only reachable by positive evidence that files and bytes landed.
    """
    if receipt is None:
        return (
            ABSENT,
            "no fetch receipt — nothing asserts that results were transferred",
        )

    miss = missing_evidence(receipt)
    if miss:
        return UNVERIFIED, (
            "receipt predates the evidence fields (missing: " + ", ".join(miss) + ") — it records entries "
            "enumerated, not files landed, so its 'complete' claim cannot be trusted; re-run "
            "scripts/fetch_results.sh (it resumes) to re-verify"
        )

    files = receipt.get("files_written", 0)
    nbytes = receipt.get("bytes_written", 0)
    failed = receipt.get("failed") or []

    if failed:
        shown = ", ".join(map(str, failed[:5])) + ("…" if len(failed) > 5 else "")
        return INCOMPLETE, (
            f"{len(failed)} entr{'y' if len(failed) == 1 else 'ies'} failed to stream "
            f"({shown}); re-run scripts/fetch_results.sh (it resumes) to finish"
        )

    if files < MIN_FILES or nbytes < MIN_BYTES:
        return INCOMPLETE, (
            f"fetch wrote {files} file(s) / {nbytes} B — nothing meaningful landed "
            "(an enumerated-but-empty fetch is a FAILURE, not a success)"
        )

    # Source-vs-destination reconciliation, when the source could be counted.
    rf = receipt.get("remote_files")
    if isinstance(rf, int) and not isinstance(rf, bool) and rf >= 0:
        if files < rf:
            rb, nb = receipt.get("remote_bytes"), nbytes
            byts = f", {nb} of {rb} bytes" if isinstance(rb, int) else ""
            return INCOMPLETE, (
                f"SHORT FETCH — {files} of {rf} source files landed{byts} "
                f"({rf - files} missing); the remote copy is NOT fully captured"
            )

    # The writer's own summary must agree with the evidence; disagreement means a buggy/edited receipt.
    if receipt.get("complete") is not True:
        return (
            INCOMPLETE,
            f"receipt says complete=false: {receipt.get('incomplete_reason') or 'unspecified'}",
        )

    total, done = receipt.get("entries_total"), receipt.get("entries_done")
    if isinstance(total, int) and isinstance(done, int) and (total <= 0 or done != total):
        return INCOMPLETE, f"entry count not satisfied ({done}/{total})"

    rec = " (reconciled against source)" if receipt.get("reconciled") is True else " (source not counted)"
    return VERIFIED, f"{files} files / {nbytes} B landed{rec}"


def is_verified(receipt: dict | None) -> bool:
    """PURE — did the fetch demonstrably land its payload? (Publish-grade: non-destructive.)"""
    return verdict(receipt)[0] == VERIFIED


def safe_to_delete(receipt: dict | None) -> tuple[bool, str]:
    """PURE — may the REMOTE copy be destroyed on the strength of this receipt?

    Deliberately stricter than is_verified: deletion additionally requires that the source was counted and
    matched (`reconciled`). An unreconciled fetch may be complete — but nothing proved it, and the cost of
    being wrong is the only copy of a multi-hour run."""
    status, why = verdict(receipt)
    if status != VERIFIED:
        return False, f"{status}: {why}"
    if receipt.get("reconciled") is not True:
        return False, (
            "UNRECONCILED: the source file count was never compared against what landed, so a "
            "shortfall would be invisible — refusing to authorise deletion"
        )
    # SCOPE: the receipt vouches for ONE run dir; reclaim destroys the WHOLE PVC. The audit found
    # additional sibling directories holding raw per-attempt records (30+ GB fleet-wide) that no fetch
    # has ever pulled. A verified fetch of one run says nothing about them.
    unfetched = receipt.get("pvc_unfetched")
    if unfetched:
        top = ", ".join(f"{e.get('name')} ({e.get('files')} files)" for e in unfetched[:4] if isinstance(e, dict))
        more = "…" if len(unfetched) > 4 else ""
        return False, (
            f"UNFETCHED PVC CONTENT: this receipt vouches for run {receipt.get('run_id')!r} only, "
            f"but the PVC also holds {len(unfetched)} un-fetched top-level entr"
            f"{'y' if len(unfetched) == 1 else 'ies'}: {top}{more} — deleting the PVC would "
            "destroy them"
        )
    return True, why


def blocking_reason(receipt: dict | None, *, allow_absent: bool = True) -> str | None:
    """Return why a run directory is unusable, or ``None`` when it is safe to consume.

    ``allow_absent`` preserves compatibility with runs created before fetch receipts were introduced. A
    present receipt must include complete verification evidence.
    """
    status, why = verdict(receipt)
    if status == ABSENT:
        return None if allow_absent else why
    if status == VERIFIED:
        return None
    return why

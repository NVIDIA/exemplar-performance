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

"""mark_reclaimable.py --pvc NAME --run-id ID --local-dir DIR [--namespace NS] — mark an artifacts PVC
reclaimable, but ONLY if the local copy is genuinely verified.

Called by fetch_results.sh at the one moment that matters: immediately after `write_receipt true`, when the
results are byte-verified on local disk. That is when the PVC stops being data and starts being garbage.
`reclaim_storage.py` will later delete only what this script vouched for.

The PVC is marked only when `_fetch_status.json` contains reconciled file and byte
counts and the local result directory still contains the verified payload. An
unverified or partial fetch leaves the PVC unchanged.

Opt-out: `--no-reclaim` (or LLMB_NO_RECLAIM=1) skips marking entirely; an operator can also make the
decision sticky at any time with `kubectl label pvc <name> llmb.nvidia.com/keep=true`, which the sweeper
honours forever.

The command is advisory unless `--strict` is used.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_receipt  # noqa: E402  — shared implementation for reading `_fetch_status.json`
from reclaim_storage import (  # noqa: E402  — shared implementation for the contract
    A_ENTRIES,
    A_LOCAL_BYTES,
    A_LOCAL_PATH,
    A_RECEIPT_VERSION,
    A_RUN_ID,
    ARTIFACT_SUFFIXES,
    CACHE_DENY,
    L_RECLAIM_AT,
    L_RECLAIMABLE,
)

# Reject receipts that point to an effectively empty result directory.
MIN_LOCAL_BYTES = 4096


def tree_bytes(d: Path) -> int:
    """PURE-ish — total payload bytes under d, EXCLUDING the bookkeeping the fetch itself wrote
    (`_fetch_status.json`, `.fetch_done/`, `.fetch_staging/`). Those must not be able to vouch for
    themselves; an interrupted private staging tree is not fetched payload."""
    total = 0
    for p in d.rglob("*"):
        if p.is_file() and not p.is_symlink():
            if p.name == "_fetch_status.json" or any(x in p.parts for x in (".fetch_done", ".fetch_staging")):
                continue
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def tree_files(d: Path) -> int:
    """PURE-ish — count of payload files under d, same exclusions as tree_bytes."""
    n = 0
    for p in d.rglob("*"):
        if p.is_file() and not p.is_symlink():
            if p.name == "_fetch_status.json" or any(x in p.parts for x in (".fetch_done", ".fetch_staging")):
                continue
            n += 1
    return n


def verify(local_dir, run_id):
    """PURE (filesystem-reading) — (ok: bool, reason: str, entries: str, nbytes: int).

    Validation requires reconciled file and byte counts from the fetch receipt.
    The current filesystem contents are counted again before the PVC is marked.
    """
    d = Path(local_dir)
    r = fetch_receipt.read(d)
    if r is None:
        return False, "no readable _fetch_status.json receipt", "", 0
    ok, why = fetch_receipt.safe_to_delete(r)
    if not ok:
        return False, why, "", 0
    if run_id and r.get("run_id") not in (None, "", run_id):
        return False, f"receipt is for run {r.get('run_id')!r}, not {run_id!r}", "", 0

    n = tree_bytes(d)
    if n < MIN_LOCAL_BYTES:
        return (
            False,
            f"local tree holds only {n}B of payload — refusing to vouch",
            "",
            n,
        )
    # Independent re-count: the receipt is a claim about the past, this is the present.
    nfiles, claimed = tree_files(d), r.get("files_written", 0)
    if nfiles < claimed:
        return (
            False,
            (
                f"local tree now holds {nfiles} files but the receipt vouched for {claimed} — "
                "results moved or deleted since the fetch; refusing to vouch"
            ),
            "",
            n,
        )
    total, done = r.get("entries_total"), r.get("entries_done")
    return True, "verified", f"{done}/{total}", n


def kc(*args):
    ctx = os.environ.get("KUBE_CONTEXT", "").strip()
    return ["kubectl"] + (["--context", ctx] if ctx else []) + list(args)


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--pvc", required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--namespace", default=os.environ.get("NAMESPACE", ""))
    ap.add_argument(
        "--no-reclaim",
        action="store_true",
        help="skip marking (also honoured via LLMB_NO_RECLAIM=1)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the patch, change nothing")
    a = ap.parse_args(argv)

    if a.no_reclaim or os.environ.get("LLMB_NO_RECLAIM", "").strip() not in ("", "0"):
        print("  reclaim: skipped (--no-reclaim) — artifacts PVC kept")
        return 0

    # Shape + cache gates, mirrored from the sweeper so a bad call site can't create a bad mark.
    if not a.pvc.endswith(ARTIFACT_SUFFIXES):
        print(f"  reclaim: {a.pvc} is not an artifacts PVC — not marking")
        return 0
    if CACHE_DENY.search(a.pvc):
        print(f"  reclaim: {a.pvc} looks like a model cache — never marking")
        return 0
    if not a.namespace:
        print("  reclaim: no namespace — not marking")
        return 0

    ok, why, entries, nbytes = verify(a.local_dir, a.run_id)
    if not ok:
        print(f"  reclaim: NOT marking {a.pvc} — {why}")
        return 0

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    patch = {
        "metadata": {
            "labels": {L_RECLAIMABLE: "true", L_RECLAIM_AT: stamp},
            "annotations": {
                A_LOCAL_PATH: str(Path(a.local_dir).resolve()),
                A_LOCAL_BYTES: str(nbytes),
                A_ENTRIES: entries,
                A_RUN_ID: a.run_id,
                # Stamp WHICH receipt contract vouched for this mark, so the sweeper can refuse a mark made
                # under the old evidence-free receipt (including ones already on cluster PVCs today).
                A_RECEIPT_VERSION: str(fetch_receipt.RECEIPT_VERSION),
            },
        }
    }
    if a.dry_run:
        print(json.dumps(patch, indent=2))
        return 0
    r = subprocess.run(
        kc(
            "-n",
            a.namespace,
            "patch",
            "pvc",
            a.pvc,
            "--type=merge",
            "-p",
            json.dumps(patch),
            "--request-timeout=30s",
        ),
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print(f"  reclaim: marked {a.pvc} reclaimable ({entries} entries verified at {a.local_dir})")
        print(
            "           free it with: llmb-k8s reclaim --storage <profile> --apply   "
            "(keep it: kubectl label pvc "
            f"{a.pvc} llmb.nvidia.com/keep=true)"
        )
    else:
        print(f"  reclaim: could not label {a.pvc} ({r.stderr.strip()}) — harmless, fetch succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

"""Capture the recipe hash at the instant a benchmark launch is submitted.

The archive/publish path must not derive this value later: a recipe may change after
the Job starts, and re-archiving must preserve the hash that existed when that Job
was launched. The sidecar is written locally before ``kubectl apply``; detached
``run.sh`` also copies its scalar evidence into the durable submit ConfigMap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import recipe_hash as _rh  # noqa: E402

SCHEMA_VERSION = 1


class AttestationCollisionError(ValueError):
    """A run-id is already owned by a different immutable launch identity."""

    def __init__(self, run_id: str, existing: dict, recipe_hash: str):
        self.run_id = run_id
        self.existing = existing
        super().__init__(
            f"refusing to replace launch attestation for run_id {run_id}: "
            f"existing owner cell={existing.get('cell')!r}, hash={existing.get('recipe_hash')!r}; "
            f"new hash={recipe_hash!r}"
        )


def _relative(cell: Path) -> str:
    try:
        return str(cell.relative_to(ROOT))
    except ValueError:
        return str(cell)


def _existing(out: Path, record: dict) -> dict:
    try:
        existing = json.loads(out.read_text())
    except (OSError, ValueError) as exc:
        raise AttestationCollisionError(
            record["run_id"], {"cell": "<unreadable>", "recipe_hash": str(exc)}, record["recipe_hash"]
        ) from exc
    same = all(existing.get(k) == record[k] for k in ("kind", "run_id", "cell", "recipe_hash"))
    if not same:
        raise AttestationCollisionError(record["run_id"], existing, record["recipe_hash"])
    return existing


def capture(cell: Path, run_id: str, out: Path, captured_at_utc: str | None = None) -> dict:
    """Atomically persist the immutable capture-time recipe fingerprint without replacing another owner.

    The previous exists-check + os.replace sequence had a TOCTOU window: two launchers could both observe
    an absent receipt and the later replace silently won. Write a same-filesystem temporary file, then use
    hard-link creation as the atomic O_EXCL operation. A loser reads the winner and either resumes the same
    identity or reports a typed collision.
    """
    cell = cell.resolve()
    if not (cell / "recipe.yaml").is_file():
        raise ValueError(f"{cell} is not a recipe cell")
    if not run_id:
        raise ValueError("run_id must not be empty")
    recipe_hash = _rh.recipe_hash(cell)
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "recipe_hash_at_launch",
        "run_id": run_id,
        "cell": _relative(cell),
        "recipe_hash": recipe_hash,
        "captured_at_utc": captured_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if out.exists():
        return _existing(out, record)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(temp, out)  # atomic create-if-absent; unlike replace, never overwrites another owner
        except FileExistsError:
            return _existing(out, record)
    finally:
        Path(temp).unlink(missing_ok=True)
    return record


def _base36(n: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n:
        out = chars[n % 36] + out
        n //= 36
    return out or "0"


def _counter_candidate(base: str, counter: int) -> str:
    """Append a counter while never lengthening the already-fit Kubernetes run id."""
    width = max(len(base), 1)
    suffix = f"-{_base36(counter)}"
    if len(suffix) < width:
        stem = base[: width - len(suffix)].rstrip("-")
        if stem:
            return f"{stem}{suffix}"
    return _base36(counter)[-width:]


def reserve(cell: Path, run_id: str, results_root: Path, max_attempts: int = 1296) -> dict:
    """Reserve a collision-free run-id and its attestation before any GPU deployment begins.

    The original id remains idempotent for the same cell/hash. If another recipe owns it, try bounded
    counter variants of the same length, so every literal `<cell>-<kind>-<run-id>` remains DNS-1123-safe.
    """
    for counter in range(max_attempts):
        candidate = run_id if counter == 0 else _counter_candidate(run_id, counter)
        run_dir = results_root / candidate
        out = run_dir / "launch_attestation.json"
        # A pre-attestation/legacy run directory can already carry artifacts. Without an immutable owner
        # receipt we cannot prove it belongs to this recipe, so fail closed and choose another id rather
        # than mixing two runs in one directory. An empty directory is safe to claim.
        if not out.exists() and run_dir.is_dir():
            try:
                next(run_dir.iterdir())
            except StopIteration:
                pass
            else:
                continue
        try:
            receipt = capture(cell, candidate, out)
        except AttestationCollisionError:
            continue
        if candidate != run_id:
            print(f"launch-attestation: collision on {run_id}; reserved {candidate} instead", file=sys.stderr)
        return receipt
    raise ValueError(f"could not reserve a collision-free run_id after {max_attempts} attempts from {run_id!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--reserve-root", type=Path, help="atomically reserve a collision-free id below this results directory"
    )
    parser.add_argument(
        "--print-run-id", action="store_true", help="print only the reserved id (for shell orchestration)"
    )
    args = parser.parse_args()
    if not args.reserve_root and not args.out:
        parser.error("one of --out or --reserve-root is required")
    try:
        receipt = (
            reserve(args.cell, args.run_id, args.reserve_root)
            if args.reserve_root
            else capture(args.cell, args.run_id, args.out)
        )
    except ValueError as exc:
        print(f"launch-attestation: {exc}", file=sys.stderr)
        return 2
    if args.print_run_id:
        print(receipt["run_id"])
    else:
        print(
            f"launch-attestation: captured {receipt['recipe_hash']} for {receipt['cell']} " f"run {receipt['run_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

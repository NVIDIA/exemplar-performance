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

"""repro_consolidate.py <original-cell> <copy-dir>... — merge parallel-reproducibility scratch-copy values
into the ORIGINAL cell's runs.jsonl for a within-cluster spread read.

parallel_repro.sh clones a cell into N name-suffixed scratch copies (distinct k8s objects → parallel-safe) and
runs them at once. Each copy is the SAME benchmark_id as the original (benchmark_id excludes envelope.name), so
its measured value is a legitimate reproducibility sample. We read each copy's latest valued runs.jsonl entry
and append it to the original's runs.jsonl — stamped with the ORIGINAL's recipe_hash (the name-only difference
doesn't change the benchmark), so `compare --repro` sees a clean same-setup spread rather than a mixed-hash flag.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import export_record as _er  # shared per-rung repeat aggregation (merge_rung_repeats / _REPEAT_KEYS)
import benchmark_id as _bid  # stable benchmark identity — the setup-equality gate (D7)


def latest_valued(cell: Path) -> dict | None:
    """The most recent runs.jsonl entry that carries a value (a published run), or None."""
    jl = cell / "runs.jsonl"
    if not jl.is_file():
        return None
    vals = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
    vals = [v for v in vals if v.get("value") is not None]
    return vals[-1] if vals else None


def existing_run_ids(cell: Path) -> set:
    """Every run_id already in the cell's runs.jsonl — the dedup key for idempotent consolidation (D4)."""
    jl = cell / "runs.jsonl"
    if not jl.is_file():
        return set()
    ids = set()
    for x in jl.read_text().splitlines():
        if not x.strip():
            continue
        try:
            rid = json.loads(x).get("run_id")
        except json.JSONDecodeError:
            continue
        if rid:
            ids.add(rid)
    return ids


def consolidate(copies, orig_recipe_hash, orig_benchmark_id=None, existing_ids=frozenset()) -> list:
    """PURE — given copy dirs (Paths/strs), return the runs.jsonl entries to append to the original: each
    copy's latest valued entry, re-stamped with the original's recipe_hash and tagged with its source.

    A copy is SKIPPED when:
      - it has no valued run (never published);
      - D4 (idempotency): its run_id is already in `existing_ids` (already consolidated) — so re-running
        `collect --sweep` is a true no-op and never double-counts a leg into the committed ledger;
      - D7 (setup-equality gate): `orig_benchmark_id` is given AND the copy's own benchmark_id does not match
        it. repro re-stamps the ORIGINAL's recipe_hash onto each leg; without this gate a copy that is NOT the
        same benchmark (a mis-cloned scratch dir, a mutated recipe) would be laundered into a clean same-hash
        repeat, silently defeating compare's same_setup guard. benchmark_id is the stable identity that
        EXCLUDES envelope.name (so the name-suffixed scratch copies still match) but MOVES if the
        dataset/sweep/SLA/serving actually differ. A copy whose benchmark_id is missing or mismatched is
        refused (positive same-setup evidence is required before a leg can be merged as a repeat).

    Does not touch the filesystem beyond reading each copy's runs.jsonl."""
    out = []
    for c in copies:
        c = Path(c)
        e = latest_valued(c)
        if not e:
            continue
        rid = e.get("run_id")
        if rid and rid in existing_ids:
            print(f"repro_consolidate: skip {c.name} — run_id {rid} already in the ledger (idempotent no-op).")
            continue
        if orig_benchmark_id is not None and e.get("benchmark_id") != orig_benchmark_id:
            print(
                f"repro_consolidate: REFUSING to merge {c.name} — benchmark_id "
                f"{str(e.get('benchmark_id'))[:12] if e.get('benchmark_id') else '<missing>'} "
                f"!= original {str(orig_benchmark_id)[:12]} (NOT the same setup; cannot be a repeat)."
            )
            continue
        out.append({**e, "recipe_hash": orig_recipe_hash, "repro_source": c.name})
    return out


def consolidate_rungs(original_record: dict, copy_records: list) -> dict | None:
    """PURE: given the ORIGINAL cell's record.json (dict) and the parallel copies' record.json dicts (each a
    legitimate same-benchmark_id repeat), return the original record with a per-rung `repeats` block overlaid
    (mean/min/max/spread/n across original+copies, matched by concurrency) so the chart can draw per-point
    error bars. The original curve stays the headline; `repeats` is purely additive. Returns None when there's
    nothing to merge (no copies / no rungs) so the caller can leave record.json untouched. Never mutates inputs."""
    base = ((original_record or {}).get("detail") or {}).get("rungs") or []
    if not base or not copy_records:
        return None
    scenario = (original_record.get("identity") or {}).get("scenario")
    keys = _er._REPEAT_KEYS.get(scenario, [])
    if not keys:
        return None
    legs = [base] + [((rec or {}).get("detail") or {}).get("rungs") or [] for rec in copy_records]
    merged = _er.merge_rung_repeats(base, legs, keys)
    rec = json.loads(json.dumps(original_record))  # deep copy; don't mutate the caller's object
    rec.setdefault("detail", {})["rungs"] = merged
    rec["detail"]["repeat_legs"] = len(copy_records) + 1
    return rec


def _recipe_hash(cell: Path) -> str:
    r = subprocess.run(["python3", str(SCRIPTS / "recipe_hash.py"), str(cell)], capture_output=True, text=True)
    return r.stdout.split()[-1] if r.returncode == 0 and r.stdout.strip() else ""


def main(argv) -> int:
    if len(argv) < 2:
        sys.exit("usage: repro_consolidate.py <original-cell> <copy-dir>...")
    original = Path(argv[0])
    copies = argv[1:]
    rh = _recipe_hash(original)
    orig_bid = _bid.benchmark_id(original)  # D7: only same-benchmark legs may be merged as repeats
    already = existing_run_ids(original)  # D4: never re-append a leg already in the ledger
    entries = consolidate(copies, rh, orig_benchmark_id=orig_bid, existing_ids=already)
    if not entries:
        print(
            "repro_consolidate: no NEW valued copy runs to merge "
            "(already consolidated, a mismatched setup, or the parallel runs never published)."
        )
        return 0
    jl = original / "runs.jsonl"
    with jl.open("a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"repro_consolidate: appended {len(entries)} reproducibility sample(s) to {jl}")
    for e in entries:
        print(f"  {e.get('repro_source')}: {e.get('metric')}={e.get('value')} (run {e.get('run_id','?')})")

    # per-rung variance: overlay {mean,min,max,spread,n} per rung onto the original's record.json from the
    # copies' committed record.json (each a same-benchmark_id repeat). Additive + best-effort — never blocks
    # the scalar consolidation above, and leaves record.json untouched if there's nothing to merge.
    orig_rec_p = original / "record.json"
    if orig_rec_p.is_file():
        try:
            orig_rec = json.loads(orig_rec_p.read_text())
            copy_recs = []
            for c in copies:
                rp = Path(c) / "record.json"
                if rp.is_file():
                    copy_recs.append(json.loads(rp.read_text()))
            merged = consolidate_rungs(orig_rec, copy_recs)
            if merged is not None:
                orig_rec_p.write_text(json.dumps(merged, indent=2) + "\n")
                print(
                    f"repro_consolidate: overlaid per-rung variance from {len(copy_recs)} copy record(s) "
                    f"onto {orig_rec_p.name}"
                )
        except Exception as exc:  # noqa: BLE001 — additive; never block the scalar consolidation
            print(f"repro_consolidate: per-rung variance overlay skipped ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

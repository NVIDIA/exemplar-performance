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

"""selftest_reclaim_storage.py — offline guards for the storage reclaim path (mark-on-verified-fetch → sweep).

No cluster. Imports reclaim_storage.py + mark_reclaimable.py and drives their PURE decision functions
(`classify`, `evaluate`, `verify`) plus the argv/CLI surface, asserting the five properties that make this
safe to hand a `--apply` flag:

  1. MARKED-ONLY        — an unmarked artifacts PVC is never selected, however old or idle it looks.
  2. CACHES NEVER       — every model-cache name shape is `protected`, even if something labels it
                          reclaimable=true. Two independent gates (class + denylist), so one bug is not enough.
  3. DRY-RUN DEFAULT    — parse_args() defaults apply=False; only an explicit --apply flips it.
  4. UNVERIFIED ≠ MARK  — a partial/failed/empty/mismatched fetch receipt does NOT produce a mark (fails CLOSED).
  5. `keep` HONOURED    — llmb.nvidia.com/keep=true beats a reclaimable mark, forever.

Plus: live-mount protection, unknown-pod-list protection, --older-than cooling-off, Retain-class refusal,
and the fleet JSON contract keys (the fleet renderer consumes these — see FLEET_CONTRACT in reclaim_storage).
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rs = _load("reclaim_storage")
mr = _load("mark_reclaimable")

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


NOW = int(time.time())
STAMP = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(NOW - 3600))  # marked 1h ago


def pvc(name, labels=None, ann=None, sc="ebs", size="50Gi"):
    return {
        "metadata": {"name": name, "labels": labels or {}, "annotations": ann or {}},
        "spec": {
            "storageClassName": sc,
            "volumeName": "pv-x",
            "resources": {"requests": {"storage": size}},
        },
        "status": {"capacity": {"storage": size}},
    }


MARKED = {rs.L_RECLAIMABLE: "true", rs.L_RECLAIM_AT: STAMP}
ANN = {
    rs.A_LOCAL_PATH: "/local/r1a78",
    rs.A_LOCAL_BYTES: str(777 << 20),
    rs.A_ENTRIES: "2/2",
    rs.A_RECEIPT_VERSION: str(rs.MIN_RECEIPT_VERSION),
}


def ev(p, live=frozenset(), retain=frozenset(), older=0):
    return rs.evaluate(p, set(live), set(retain), NOW, older)[0]


print("selftest_reclaim_storage")

# ── 1. marked-only deletion ──────────────────────────────────────────────────────────────────────────────
check(
    "1 marked+verified artifacts PVC → reclaimable",
    ev(pvc("cell-artifacts", MARKED, ANN)) == "reclaimable",
)
check(
    "1 UNMARKED artifacts PVC → unverified (never deleted)",
    ev(pvc("cell-artifacts")) == "unverified",
)
check(
    "1 reclaimable=false is not truthy",
    ev(pvc("cell-artifacts", {rs.L_RECLAIMABLE: "false"})) == "unverified",
)

# ── 2. model caches are never selected ───────────────────────────────────────────────────────────────────
CACHES = [
    "glm5-fp8-model-cache",
    "nemotron-ultra-hf-cache",
    "nemotron-ultra-nvfp4-cache",
    "serving-gb300-model-cache",
    "serving-gb300-model-cache-r5",
    "shared-model-cache",
    "qwen3-32b-shared-cache",
    "nemotron-hf-cache",
]
check(
    "2 every known model cache classifies as model-cache",
    all(rs.classify(pvc(c)) == "model-cache" for c in CACHES),
    str([c for c in CACHES if rs.classify(pvc(c)) != "model-cache"]),
)
check(
    "2 a model cache MISLABELLED reclaimable=true is still protected",
    all(ev(pvc(c, dict(MARKED), dict(ANN))) == "protected" for c in CACHES),
)
check(
    "2 llmb-control is protected (control, not artifacts)",
    ev(pvc("llmb-control", dict(MARKED), dict(ANN), sc="fsx-lustre", size="1200Gi")) == "protected",
)
check(
    "2 a non-artifacts PVC is protected",
    ev(pvc("serving-cluster-b-results", dict(MARKED))) == "protected",
)
check(
    "2 mark_reclaimable REFUSES to mark a cache name",
    mr.main(
        [
            "--pvc",
            "serving-gb300-model-cache",
            "--local-dir",
            "/nope",
            "--namespace",
            "ns",
            "--dry-run",
        ]
    )
    == 0
    and mr.CACHE_DENY.search("serving-gb300-model-cache") is not None,
)

# ── 3. dry-run is the default ────────────────────────────────────────────────────────────────────────────
check(
    "3 parse_args() defaults to dry-run (apply=False)",
    rs.parse_args(["prof"]).apply is False,
)
check(
    "3 --apply is the only way to set apply",
    rs.parse_args(["prof", "--apply"]).apply is True,
)
check(
    "3 --dry-run accepted as an explicit no-op",
    rs.parse_args(["prof", "--dry-run"]).apply is False,
)
check(
    "3 --older-than parses duration units",
    rs.parse_duration("7d") == 604800 and rs.parse_duration("12h") == 43200 and rs.parse_duration("30m") == 1800,
)


# ── 4. an UNVERIFIED fetch does NOT mark ─────────────────────────────────────────────────────────────────
def receipt_dir(receipt, payload=b"x" * 8192, name="r1a78"):
    d = Path(tempfile.mkdtemp()) / name
    (d / ".fetch_done").mkdir(parents=True)
    if payload:
        (d / "result.json").write_bytes(payload)
    if receipt is not None:
        (d / "_fetch_status.json").write_text(json.dumps(receipt))
    return d


# A v2 receipt carries EVIDENCE of what landed (files/bytes) plus a source reconciliation — not just "the
# loop ran". See scripts/fetch_receipt.py and selftest_fetch_receipt.py.
GOOD = {
    "receipt_version": 2,
    "run_id": "r1a78",
    "entries_total": 2,
    "entries_done": 2,
    "failed": [],
    "complete": True,
    "files_written": 1,
    "bytes_written": 8192,
    "remote_files": 1,
    "remote_bytes": 8192,
    "reconciled": True,
    "pvc_unfetched": [],
}
check(
    "4 a complete, non-empty, zero-failure, reconciled fetch DOES verify",
    mr.verify(str(receipt_dir(GOOD)), "r1a78")[0] is True,
)
check(
    "4 a v1 receipt (no evidence fields) → no mark, however 'complete' it claims to be",
    mr.verify(
        str(
            receipt_dir(
                {
                    "run_id": "r1a78",
                    "entries_total": 2,
                    "entries_done": 2,
                    "failed": [],
                    "complete": True,
                }
            )
        ),
        "r1a78",
    )[0]
    is False,
)
check(
    "4 a SHORT fetch (fewer files landed than the source held) → no mark",
    mr.verify(str(receipt_dir({**GOOD, "remote_files": 6427})), "r1a78")[0] is False,
)
check(
    "4 complete=false → no mark",
    mr.verify(str(receipt_dir({**GOOD, "complete": False})), "r1a78")[0] is False,
)
check(
    "4 failed entries → no mark",
    mr.verify(str(receipt_dir({**GOOD, "failed": ["c16"]})), "r1a78")[0] is False,
)
check(
    "4 entries_done < entries_total → no mark",
    mr.verify(str(receipt_dir({**GOOD, "entries_done": 1})), "r1a78")[0] is False,
)
check(
    "4 missing receipt → no mark",
    mr.verify(str(receipt_dir(None)), "r1a78")[0] is False,
)
check(
    "4 receipt over an EMPTY tree → no mark (a receipt cannot vouch for itself)",
    mr.verify(str(receipt_dir(GOOD, payload=b"")), "r1a78")[0] is False,
)
check(
    "4 receipt for a DIFFERENT run → no mark",
    mr.verify(str(receipt_dir(GOOD)), "r99zz")[0] is False,
)
check(
    "4 bookkeeping files are excluded from the payload total",
    mr.tree_bytes(receipt_dir(GOOD, payload=b"y" * 100)) == 100,
)

# ── 5. `keep` label is honoured ──────────────────────────────────────────────────────────────────────────
check(
    "5 keep=true beats a reclaimable mark",
    ev(pvc("cell-artifacts", {**MARKED, rs.L_KEEP: "true"}, ANN)) == "keep",
)
check(
    "5 keep=true on an unmarked PVC is still keep (sticky, pre-emptive)",
    ev(pvc("cell-artifacts", {rs.L_KEEP: "true"})) == "keep",
)

# ── live-mount / unknown-state / cooling-off / Retain ────────────────────────────────────────────────────
check(
    "live: a mounted PVC is never reclaimable even if marked",
    ev(pvc("cell-artifacts", MARKED, ANN), live={"cell-artifacts"}) == "live",
)
check(
    "live: an unlistable pod set refuses to assume unmounted",
    rs.evaluate(pvc("cell-artifacts", MARKED, ANN), None, set(), NOW, 0)[0] == "unverified",
)
check(
    "--older-than 7d holds a PVC marked 1h ago",
    ev(pvc("cell-artifacts", MARKED, ANN), older=604800) == "too-fresh",
)
check(
    "--older-than 30m releases a PVC marked 1h ago",
    ev(pvc("cell-artifacts", MARKED, ANN), older=1800) == "reclaimable",
)
check(
    "Retain storage class → retain-skip, not a silent PV leak",
    ev(pvc("cell-artifacts", MARKED, ANN, sc="fsx-lustre"), retain={"fsx-lustre"}) == "retain-skip",
)

# ── quantity math + fleet contract ───────────────────────────────────────────────────────────────────────
check(
    "qty_bytes handles Gi/Ti/G/bare",
    rs.qty_bytes("50Gi") == 50 << 30
    and rs.qty_bytes("50Ti") == 50 << 40
    and rs.qty_bytes("100G") == 100 * 10**9
    and rs.qty_bytes("1024") == 1024
    and rs.qty_bytes("") == 0
    and rs.qty_bytes("junk") == 0,
)
check(
    "human() renders Ti/Gi",
    rs.human(50 << 30) == "50.0Gi" and rs.human(3 << 40) == "3.0Ti",
)
CONTRACT = {
    "contract_version",
    "cluster",
    "namespace",
    "total_pvcs",
    "artifacts_pvcs",
    "artifacts_bytes",
    "cache_pvcs",
    "cache_bytes",
    "reclaimable_pvcs",
    "reclaimable_bytes",
    "kept_pvcs",
    "items",
}
check(
    "fleet contract keys are documented in the module docstring",
    all(k in rs.__doc__ or k in open(ROOT / "scripts" / "reclaim_storage.py").read() for k in CONTRACT),
)
check(
    "fleet `class` vocabulary is exactly {artifacts,model-cache,control,other}",
    {rs.classify(pvc(n)) for n in ("a-artifacts", "x-model-cache", "llmb-control", "some-pv")}
    == {"artifacts", "model-cache", "control", "other"},
)

print(f"\n{'OK' if not fails else 'FAILED: ' + ', '.join(fails)}")
raise SystemExit(1 if fails else 0)

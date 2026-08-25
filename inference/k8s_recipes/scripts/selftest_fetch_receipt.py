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

"""selftest_fetch_receipt.py — the postcondition tests that were missing when a fetch receipt could say
`complete: true` over a tree holding two files (data-integrity case, 2026-07-31).

This survived because NOTHING asserted the postcondition. So both halves are tested here:

  WRITER (`fetch_results.sh` write_receipt, driven for real in bash with a stubbed `llmb::kc`)
    W1  a fetch that writes ZERO files              → complete: false          [required]
    W2  a fetch shorter than the source reported    → complete: false + shortfall recorded  [required]
    W3  a genuine full fetch                        → complete: true           [no regression]
    W4  the evidence fields are actually populated (files_written/bytes_written/remote_files/reconciled)

  READER (`fetch_receipt.py`, consumed by publish.py and mark_reclaimable.py)
    R1  a receipt lacking the evidence fields (v1)  → UNVERIFIED, never safe-to-delete   [required]
    R2  zero-file / short / failed receipts         → INCOMPLETE, never safe-to-delete
    R3  a full receipt                              → VERIFIED and safe-to-delete
    R4  an unreconciled but verified receipt        → publishable, NOT deletable
    R5  un-fetched sibling content on the PVC       → NOT deletable (the `driver-jobs` hazard)
    R6  mark_reclaimable.verify agrees with the reader on every one of the above

Fixtures are the REAL rescued trees where present (`~/gb300-rescue/{c16,c32}`,
`~/llmb-archive`), read-only — the writer is pointed at them and the receipt is written to a
tempdir. They are skipped (not failed) on a machine that lacks them, so CI stays green off-workstation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FETCH_SH = ROOT / "scripts" / "fetch_results.sh"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fr = _load("fetch_receipt")
mr = _load("mark_reclaimable")

fails: list[str] = []
skips: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def skip(name: str, why: str) -> None:
    print(f"  SKIP  {name}  [{why}]")
    skips.append(name)


# ── real rescued trees (fixtures) ───────────────────────────────────────────────────────────────────────
RESCUE = Path.home() / "gb300-rescue"
ARCHIVE = Path.home() / "llmb-archive"


def real_tree() -> Path | None:
    """A genuine fetched-shaped run tree from the manual rescue, or None on a machine without them."""
    for cand in (
        RESCUE / "c16" / "ttj2o4g",
        RESCUE / "c32" / "ttj2o4g",
        RESCUE / "c16",
        ARCHIVE,
    ):
        if cand.is_dir() and any(cand.rglob("*")):
            return cand
    return None


# ── WRITER: drive the real bash write_receipt with a stubbed cluster ─────────────────────────────────────
def _extract_writer() -> str:
    """Slice the receipt-writer functions out of fetch_results.sh so they can be exercised without a
    cluster. Deliberately reads the SHIPPING script — a test against a copy would not have caught this bug.
    """
    src = FETCH_SH.read_text()
    start = src.index("local_stats() {")
    end = src.index('POD="${RECIPE_SHORTNAME}-fetch-${RUN_ID}"')
    return src[start:end]


HARNESS = r"""
set -eu
log() { printf '%s\n' "$*" >&2; }
err() { printf '%s\n' "$*" >&2; }
# Stub the cluster: remote_stats asks for `ls -Lln`, sibling_stats loops `for e in`.
llmb::kc() {
  case "$*" in
    *"ls -Lln"*) [ -n "${STUB_REMOTE:-}" ] && printf '%s\n' "$STUB_REMOTE"; return 0 ;;
    *"for e in"*) [ -n "${STUB_SIBLINGS:-}" ] && printf '%s\n' "$STUB_SIBLINGS"; return 0 ;;
  esac
  return 0
}
. "$WRITER_SNIPPET"
write_receipt "$INTENT"
"""


def run_writer(
    local_dir: Path,
    receipt: Path,
    *,
    intent="true",
    remote="",
    siblings="",
    with_inputs=0,
    failed="",
    total=2,
) -> dict | None:
    with tempfile.TemporaryDirectory() as td:
        snippet = Path(td) / "writer.sh"
        snippet.write_text(_extract_writer())
        harness = Path(td) / "harness.sh"
        harness.write_text(HARNESS)
        env = dict(
            os.environ,
            WRITER_SNIPPET=str(snippet),
            INTENT=intent,
            STUB_REMOTE=remote,
            STUB_SIBLINGS=siblings,
            local_dir=str(local_dir),
            RECEIPT=str(receipt),
            RUN_ID="rtest",
            REMOTE_RUN_DIR="/artifacts/rtest",
            POD="stub-fetch-pod",
            WITH_INPUTS=str(with_inputs),
            FAILED=failed,
            TOTAL=str(total),
            RESUMED="0",
        )
        p = subprocess.run(["bash", str(harness)], env=env, capture_output=True, text=True)
        if not receipt.is_file():
            print("    writer stderr:", p.stderr.strip()[-500:])
            return None
        return json.loads(receipt.read_text())


print("fetch receipt — WRITER (real bash, stubbed cluster)")

tree = real_tree()
with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # Zero files landed even though entries were enumerated.
    empty = td / "empty_run"
    empty.mkdir()
    r = run_writer(empty, td / "r1.json", remote="6427 5200000000")
    check(
        "W1 zero files written → complete:false",
        bool(r) and r.get("complete") is False,
        json.dumps(r),
    )
    check(
        "W1 zero-file receipt records files_written=0",
        bool(r) and r.get("files_written") == 0,
    )
    check(
        "W1 zero-file receipt is INCOMPLETE to the reader",
        bool(r) and fr.verdict(r)[0] == fr.INCOMPLETE,
        str(r and fr.verdict(r)),
    )

    # W1b — the receipt must not be able to vouch for itself: a tree holding ONLY the receipt +
    # done-markers is still zero payload.
    selfonly = td / "selfonly"
    (selfonly / ".fetch_done").mkdir(parents=True)
    (selfonly / ".fetch_done" / "concurrency_16.noinputs").write_text("")
    r = run_writer(selfonly, selfonly / "_fetch_status.json", remote="100 100000")
    check(
        "W1b tree holding only fetch bookkeeping → complete:false",
        bool(r) and r.get("complete") is False and r.get("files_written") == 0,
        json.dumps(r),
    )

    if tree is None:
        skip(
            "W2/W3 full + short fetch against a real rescued tree",
            f"{RESCUE} not present",
        )
    else:
        # W3 — a genuine full fetch of a REAL rescued tree still succeeds (no regression).
        n_files = sum(1 for p in tree.rglob("*") if p.is_file())
        n_bytes = sum(p.lstat().st_size for p in tree.rglob("*") if p.is_file())
        full = run_writer(tree, td / "r3.json", remote=f"{n_files} {n_bytes}")
        check(
            "W3 genuine full fetch (real rescued tree) → complete:true",
            bool(full) and full.get("complete") is True,
            full and full.get("incomplete_reason"),
        )
        check(
            "W3 evidence fields populated from the real tree",
            bool(full)
            and full.get("files_written") == n_files
            and full.get("bytes_written") == n_bytes
            and full.get("reconciled") is True,
            json.dumps({k: full.get(k) for k in ("files_written", "bytes_written", "reconciled")} if full else {}),
        )
        check(
            "W3 full receipt reads VERIFIED and safe-to-delete",
            bool(full) and fr.verdict(full)[0] == fr.VERIFIED and fr.safe_to_delete(full)[0],
            str(full and fr.safe_to_delete(full)),
        )

        # W2 — the source holds MORE than landed (the r1bcd shape: 2 local vs 6,427 remote).
        short = run_writer(tree, td / "r2.json", remote=f"{n_files + 500} {n_bytes * 2}")
        check(
            "W2 short fetch (source > landed) → complete:false",
            bool(short) and short.get("complete") is False,
            json.dumps(short),
        )
        check(
            "W2 shortfall is RECORDED, not just implied",
            bool(short)
            and "500 missing" in (short.get("incomplete_reason") or "")
            and short.get("remote_files") == n_files + 500,
            short and short.get("incomplete_reason"),
        )
        check(
            "W2 short receipt is INCOMPLETE + never safe-to-delete",
            bool(short) and fr.verdict(short)[0] == fr.INCOMPLETE and not fr.safe_to_delete(short)[0],
        )

        # W4 — un-fetched sibling content on the PVC (the `driver-jobs` hazard) is recorded.
        sib = run_writer(
            tree,
            td / "r4.json",
            remote=f"{n_files} {n_bytes}",
            siblings="driver-jobs 6427",
        )
        check(
            "W4 un-fetched PVC siblings recorded in the receipt",
            bool(sib) and sib.get("pvc_unfetched") == [{"name": "driver-jobs", "files": 6427}],
            json.dumps(sib and sib.get("pvc_unfetched")),
        )
        check(
            "W4 a run whose PVC holds un-fetched siblings is NOT safe-to-delete",
            bool(sib) and not fr.safe_to_delete(sib)[0],
            str(sib and fr.safe_to_delete(sib)),
        )

    # W5 — a failed entry still yields complete:false (pre-existing behaviour preserved).
    r = run_writer(
        tree or empty,
        td / "r5.json",
        intent="false",
        failed="concurrency_256",
        remote="1 1",
    )
    check(
        "W5 failed entry → complete:false with the entry named",
        bool(r) and r.get("complete") is False and r.get("failed") == ["concurrency_256"],
        json.dumps(r),
    )


# ── ATOMIC TRANSFER / RESUME ────────────────────────────────────────────────────────────────────────────
print("fetch receipt — ATOMIC TRANSFER / RESUME")


def _extract_transfer() -> str:
    """Exercise the shipping staging, promotion, and marker functions without a cluster."""
    src = FETCH_SH.read_text()
    start = src.index("entry_stats() {")
    end = src.index('\nFAILED=""', start)
    return src[start:end]


ATOMIC_HARNESS = r"""
set -eu
set -o pipefail
err() { printf '%s\n' "$*" >&2; }
local_dir="$TEST_LOCAL"
STAGE_DIR="$local_dir/.fetch_staging"
DONE_DIR="$local_dir/.fetch_done"
MODE_TAG=noinputs
TAR_EXCLUDES=""
POD=stub
REMOTE_RUN_DIR=/artifacts/test
mkdir -p "$STAGE_DIR" "$DONE_DIR"
. "$TRANSFER_SNIPPET"

entry=concurrency_64
mkdir -p "$local_dir/$entry"
printf old > "$local_dir/$entry/payload.bin"

# Simulate Teleport dropping the tar stream after creating output. The old
# final entry must remain byte-for-byte intact and no partial stage may remain.
llmb::kc() { printf 'not-a-tar-stream'; return 1; }
if stream_entry "$entry"; then exit 21; fi
[ "$(cat "$local_dir/$entry/payload.bin")" = old ] || exit 22
[ -z "$(find "$STAGE_DIR" -mindepth 1 -print -quit)" ] || exit 23
printf 'failed-preserved\n'

# A clean retry is extracted privately, then replaces the old entry.
llmb::kc() { tar -ch -C "$REMOTE_FIXTURE" -- "$ENTRY"; }
ENTRY="$entry"
stream_entry "$entry"
[ "$(cat "$local_dir/$entry/payload.bin")" = complete-payload ] || exit 24
printf 'success-promoted\n'

marker="$DONE_DIR/${entry}.${MODE_TAG}"
write_done_marker "$marker" "$local_dir/$entry" "$MODE_TAG"
marker_matches "$marker" "$local_dir/$entry" "$MODE_TAG" || exit 25
printf 'marker-valid\n'

# Any later truncation invalidates the marker; a zero-byte legacy marker is
# never accepted merely because the top-level directory exists.
printf x >> "$local_dir/$entry/payload.bin"
if marker_matches "$marker" "$local_dir/$entry" "$MODE_TAG"; then exit 26; fi
: > "$marker"
if marker_matches "$marker" "$local_dir/$entry" "$MODE_TAG"; then exit 27; fi
printf 'stale-rejected\n'
"""


with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    local = td / "local"
    remote = td / "remote"
    (remote / "concurrency_64").mkdir(parents=True)
    (remote / "concurrency_64" / "payload.bin").write_text("complete-payload")
    snippet = td / "transfer.sh"
    snippet.write_text(_extract_transfer())
    harness = td / "atomic.sh"
    harness.write_text(ATOMIC_HARNESS)
    env = dict(
        os.environ,
        TEST_LOCAL=str(local),
        REMOTE_FIXTURE=str(remote),
        TRANSFER_SNIPPET=str(snippet),
    )
    p = subprocess.run(["bash", str(harness)], env=env, capture_output=True, text=True)
    check(
        "A1 failed stream preserves previous entry",
        p.returncode == 0 and "failed-preserved" in p.stdout,
        p.stderr[-500:],
    )
    check(
        "A2 clean retry atomically promotes complete entry",
        p.returncode == 0 and "success-promoted" in p.stdout,
        p.stderr[-500:],
    )
    check(
        "A3 evidence marker permits verified resume",
        p.returncode == 0 and "marker-valid" in p.stdout,
        p.stderr[-500:],
    )
    check(
        "A4 truncated entry and legacy marker force re-fetch",
        p.returncode == 0 and "stale-rejected" in p.stdout,
        p.stderr[-500:],
    )

    # A hard-killed fetch cannot run its trap. The next invocation must remove
    # an abandoned partial attempt. If the process died in the narrower window
    # after parking the old target as .previous, cleanup must restore it first.
    abandoned = local / ".fetch_staging" / "concurrency_16.abandoned"
    abandoned.mkdir(parents=True)
    (abandoned / "partial.bin").write_bytes(b"partial")
    interrupted = local / ".fetch_staging" / "concurrency_32.killed"
    (interrupted / ".previous").mkdir(parents=True)
    (interrupted / ".previous" / "payload.bin").write_text("previous-valid")
    source = FETCH_SH.read_text()
    start = source.index("clear_abandoned_staging() {")
    end = source.index("\nMODE_TAG=", start)
    cleanup = td / "cleanup.sh"
    cleanup.write_text(
        "set -eu\n"
        "die() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
        "log() { printf '%s\\n' \"$*\" >&2; }\n"
        'local_dir="$TEST_LOCAL"\n'
        'STAGE_DIR="$local_dir/.fetch_staging"\n' + source[start:end] + "\n"
    )
    cleaned = subprocess.run(
        ["bash", str(cleanup)],
        env={**os.environ, "TEST_LOCAL": str(local)},
        capture_output=True,
        text=True,
    )
    restored = local / "concurrency_32" / "payload.bin"
    check(
        "A5 retry restores a valid target parked during interrupted promotion",
        cleaned.returncode == 0 and restored.read_text() == "previous-valid",
        cleaned.stderr[-500:],
    )
    check(
        "A6 retry removes abandoned private staging attempts",
        cleaned.returncode == 0 and not any((local / ".fetch_staging").iterdir()),
        cleaned.stderr[-500:],
    )


# ── READER ──────────────────────────────────────────────────────────────────────────────────────────────
print("fetch receipt — READER (fetch_receipt.verdict / safe_to_delete)")

V1_FALSE_POSITIVE = {  # verbatim shape of the 24 bad receipts found on disk
    "run_id": "r1bcd",
    "remote_dir": "/artifacts/r1bcd",
    "with_inputs": False,
    "entries_total": 2,
    "entries_done": 2,
    "failed": [],
    "complete": True,
    "updated_utc": "2026-07-31T04:00:00Z",
}
GOOD = {
    "receipt_version": 2,
    "run_id": "r1bcd",
    "entries_total": 2,
    "entries_done": 2,
    "failed": [],
    "files_written": 6427,
    "bytes_written": 5_200_000_000,
    "remote_files": 6427,
    "remote_bytes": 5_200_000_000,
    "reconciled": True,
    "pvc_unfetched": [],
    "complete": True,
}

st, why = fr.verdict(V1_FALSE_POSITIVE)
check(
    "R1 v1 receipt (no evidence fields) reads UNVERIFIED, NOT complete",
    st == fr.UNVERIFIED,
    f"{st}: {why}",
)
check("R1 v1 receipt is never safe-to-delete", not fr.safe_to_delete(V1_FALSE_POSITIVE)[0])
check(
    "R1 v1 receipt BLOCKS publish (not grandfathered)",
    fr.blocking_reason(V1_FALSE_POSITIVE) is not None,
)
check(
    "R1 absent receipt → ABSENT (old runs still publishable, still not deletable)",
    fr.verdict(None)[0] == fr.ABSENT and fr.blocking_reason(None) is None and not fr.safe_to_delete(None)[0],
)

check(
    "R2 zero files → INCOMPLETE",
    fr.verdict({**GOOD, "files_written": 0, "bytes_written": 0})[0] == fr.INCOMPLETE,
)
check(
    "R2 trivial byte count → INCOMPLETE",
    fr.verdict({**GOOD, "files_written": 2, "bytes_written": 12})[0] == fr.INCOMPLETE,
)
_short = fr.verdict({**GOOD, "files_written": 2, "bytes_written": 5000})
check(
    "R2 short fetch → INCOMPLETE naming the shortfall",
    _short[0] == fr.INCOMPLETE and "6425 missing" in _short[1],
    str(_short),
)
check(
    "R2 failed entries → INCOMPLETE even with evidence present",
    fr.verdict({**GOOD, "failed": ["concurrency_256"]})[0] == fr.INCOMPLETE,
)
check(
    "R2 complete:false is honoured even when evidence looks fine",
    fr.verdict({**GOOD, "complete": False})[0] == fr.INCOMPLETE,
)

check(
    "R3 full receipt → VERIFIED + safe-to-delete",
    fr.verdict(GOOD)[0] == fr.VERIFIED and fr.safe_to_delete(GOOD)[0] and fr.blocking_reason(GOOD) is None,
)

_unrec = {**GOOD, "reconciled": False, "remote_files": None}
check(
    "R4 unreconciled fetch is publishable but NOT deletable",
    fr.verdict(_unrec)[0] == fr.VERIFIED and fr.blocking_reason(_unrec) is None and not fr.safe_to_delete(_unrec)[0],
    str(fr.safe_to_delete(_unrec)),
)

_sib = {**GOOD, "pvc_unfetched": [{"name": "driver-jobs", "files": 6427}]}
ok, why = fr.safe_to_delete(_sib)
check(
    "R5 un-fetched PVC content blocks deletion and names it",
    not ok and "driver-jobs" in why,
    why,
)
check(
    "R5 un-fetched PVC content does NOT block publish (the run itself is complete)",
    fr.blocking_reason(_sib) is None,
)

# Type-confusion / hand-edited receipts must not sneak past the evidence gate.
for bad, label in (
    (True, "bool"),
    ("6427", "string"),
    (-1, "negative"),
    (None, "null"),
):
    check(
        f"R1 files_written={label} is not evidence → UNVERIFIED",
        fr.verdict({**GOOD, "files_written": bad})[0] == fr.UNVERIFIED,
    )


# ── R6: mark_reclaimable agrees with the reader on a REAL tree ──────────────────────────────────────────
print("fetch receipt — mark_reclaimable.verify (deletion authority)")

with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "run"
    d.mkdir()
    (d / "payload.bin").write_bytes(b"x" * 8192)
    (d / "concurrency_16").mkdir()
    (d / "concurrency_16" / "report.json").write_text('{"ok": true}')
    nfiles = 3
    (d / "extra.json").write_text('{"a": 1}')

    def verdict_for(receipt):
        (d / "_fetch_status.json").write_text(json.dumps(receipt))
        return mr.verify(str(d), receipt.get("run_id", ""))

    ok, why, _e, _n = verdict_for(V1_FALSE_POSITIVE)
    check(
        "R6 mark_reclaimable REFUSES a v1 receipt (auto-reclaim can't be fooled by the old shape)",
        not ok and "UNVERIFIED" in why,
        why,
    )

    good_local = {
        **GOOD,
        "run_id": "",
        "files_written": nfiles,
        "bytes_written": 8300,
        "remote_files": nfiles,
        "remote_bytes": 8300,
    }
    ok, why, entries, nb = verdict_for(good_local)
    check(
        "R6 mark_reclaimable ACCEPTS a genuine verified+reconciled receipt (no regression)",
        ok,
        why,
    )

    # A killed fetch may leave a private staging attempt behind. It is neither
    # landed payload nor evidence and must not inflate the reclaim annotation.
    payload_bytes = nb
    (d / ".fetch_staging" / "concurrency_64.partial").mkdir(parents=True)
    (d / ".fetch_staging" / "concurrency_64.partial" / "partial.bin").write_bytes(b"p" * 65536)
    ok, why, _entries, nb = verdict_for(good_local)
    check(
        "R6 mark_reclaimable excludes interrupted private staging data",
        ok and nb == payload_bytes,
        f"{why}; bytes={nb}, expected={payload_bytes}",
    )

    ok, why, _e, _n = verdict_for({**good_local, "files_written": nfiles + 50})
    check(
        "R6 mark_reclaimable re-counts the tree: receipt claiming more files than exist → refuse",
        not ok and "moved or deleted" in why,
        why,
    )

    ok, why, _e, _n = verdict_for({**good_local, "reconciled": False, "remote_files": None})
    check(
        "R6 mark_reclaimable refuses an unreconciled receipt",
        not ok and "UNRECONCILED" in why,
        why,
    )

    ok, why, _e, _n = verdict_for({**good_local, "pvc_unfetched": [{"name": "driver-jobs", "files": 6427}]})
    check(
        "R6 mark_reclaimable refuses while un-fetched PVC content remains",
        not ok and "driver-jobs" in why,
        why,
    )

    # An emptied tree with a good receipt: the receipt is a claim about the past, verify re-checks the present.
    shutil.rmtree(d / "concurrency_16")
    (d / "payload.bin").unlink()
    (d / "extra.json").unlink()
    ok, why, _e, _n = verdict_for(good_local)
    check("R6 mark_reclaimable refuses once the local copy is gone", not ok, why)


# ── the SWEEPER must not trust a mark made under the old receipt ────────────────────────────────────────
# mark_reclaimable only ever LABELS; reclaim_storage does the deleting. Labels written by the PRE-FIX code
# may already sit on cluster PVCs right now — they were made on the strength of a receipt that counted
# entries, not files. Those marks must read as unverified, exactly like the receipts that produced them.
print("fetch receipt — reclaim_storage sweeper gate (stale marks)")
rs = _load("reclaim_storage")


def _pvc(ann):
    return {
        "metadata": {
            "name": "cell-artifacts",
            "labels": {rs.L_RECLAIMABLE: "true", rs.L_RECLAIM_AT: "20260731T040000Z"},
            "annotations": ann,
        },
        "spec": {
            "storageClassName": "ebs",
            "volumeName": "pv-x",
            "resources": {"requests": {"storage": "50Gi"}},
        },
        "status": {"capacity": {"storage": "50Gi"}},
    }


_base = {
    rs.A_LOCAL_PATH: "/local/r1a78",
    rs.A_LOCAL_BYTES: "1000000",
    rs.A_ENTRIES: "2/2",
}
_st, _ev = rs.evaluate(_pvc(_base), set(), set(), 2 << 30, 0)
check(
    "S1 a mark with NO receipt-version annotation (legacy) → unverified, never deleted",
    _st == "unverified" and "predates" in _ev,
    f"{_st}: {_ev}",
)
_st, _ev = rs.evaluate(_pvc({**_base, rs.A_RECEIPT_VERSION: "1"}), set(), set(), 2 << 30, 0)
check("S1 a v1 mark → unverified", _st == "unverified", f"{_st}: {_ev}")
_st, _ev = rs.evaluate(
    _pvc({**_base, rs.A_RECEIPT_VERSION: str(fr.RECEIPT_VERSION)}),
    set(),
    set(),
    2 << 30,
    0,
)
check(
    "S1 a mark made under the evidence-bearing receipt IS reclaimable (no regression)",
    _st == "reclaimable",
    f"{_st}: {_ev}",
)
# …and the mark_reclaimable → reclaim_storage handshake actually carries that annotation.
with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "run"
    d.mkdir()
    (d / "payload.bin").write_bytes(b"z" * 8192)
    (d / "_fetch_status.json").write_text(
        json.dumps(
            {
                "receipt_version": 2,
                "run_id": "r1a78",
                "entries_total": 1,
                "entries_done": 1,
                "failed": [],
                "complete": True,
                "files_written": 1,
                "bytes_written": 8192,
                "remote_files": 1,
                "remote_bytes": 8192,
                "reconciled": True,
                "pvc_unfetched": [],
            }
        )
    )
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mr.main(
            [
                "--pvc",
                "cell-artifacts",
                "--run-id",
                "r1a78",
                "--local-dir",
                str(d),
                "--namespace",
                "ns",
                "--dry-run",
            ]
        )
    _patch = buf.getvalue()
    check(
        "S1 mark_reclaimable stamps the receipt version it verified under",
        rs.A_RECEIPT_VERSION in _patch and f'"{rs.MIN_RECEIPT_VERSION}"' in _patch,
        _patch[:200],
    )


# ── publish.py agrees with the reader ────────────────────────────────────────────────────────────────────
pub = _load("publish")
check(
    "publish.fetch_incomplete blocks a v1 receipt",
    pub.fetch_incomplete(V1_FALSE_POSITIVE) is not None,
)
check(
    "publish.fetch_incomplete allows a verified receipt",
    pub.fetch_incomplete(GOOD) is None,
)
check(
    "publish.fetch_incomplete allows an absent receipt (old runs)",
    pub.fetch_incomplete(None) is None,
)


# The transient mounter must obey the same storage placement contract as every
# other PVC-mounting helper. Otherwise an RWO claim can attach to a CPU node
# whose kubelet cannot mount that storage class, stranding fetch after a valid
# benchmark has already completed.
_fetch_src = FETCH_SH.read_text()
check(
    "fetch mounter canonicalizes MODEL_CACHE_NODE_SELECTOR",
    "llmb::model_cache_node_selector_yaml" in _fetch_src
    and "nodeSelector: { ${MODEL_CACHE_NODE_SELECTOR_YAML} }" in _fetch_src,
)
check(
    "fetch mounter tolerates the selected tainted storage pool",
    "operator: Exists, effect: NoSchedule" in _fetch_src and "operator: Exists, effect: PreferNoSchedule" in _fetch_src,
)


print()
if skips:
    print(f"({len(skips)} fixture-dependent check(s) skipped: {', '.join(skips)})")
if fails:
    print(f"FAILED ({len(fails)}): " + "; ".join(fails))
    raise SystemExit(1)
print("selftest_fetch_receipt: OK")

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

"""selftest_results_check.py — offline guards for the KEYSTONE: RESULTS.md + record.json regenerate from
COMMITTED curated data byte-for-byte, and the NOTES.md split. No cluster, no network.

Covers:
  A. RECORD FROM COMMITTED CURATED — export_record pointed at a cell's committed curated/ dir reproduces the
     committed record.json byte-for-byte (the run-dir is gone; git is pinned in run_meta) → deterministic.
  B. RESULTS FROM COMMITTED RECORD — publish.render_results_md is a pure function of committed record.json +
     aggregate.json; re-rendering yields the committed RESULTS.md byte-for-byte, and is stable across calls.
  C. results_check.check_cell reports NO drift for an on-pipeline cell (the CI gate is green on committed data).
  D. NOTES SPLIT — narrative above the old PUBLISH marker migrates into a NOTES.md stamped non-authoritative;
     idempotent (never overwrites); RESULTS.md rendering does NOT depend on NOTES.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
import publish as pub  # noqa: E402
import results_check as rc  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── A–C. the real on-pipeline cell(s): regenerate-from-committed byte-for-byte ────────────────────────────
cells = rc.on_pipeline_cells(ROOT)

# SYNTHESIZED on-pipeline cell (fallback). This suite proves that a published cell's record.json and
# RESULTS.md regenerate from its COMMITTED curated data — a real guarantee worth keeping armed. But the
# corpus can legitimately have ZERO on-pipeline cells (right now every cell is `wip` while the
# KVBM numbers are re-baselined). Rather than delete or weaken the proof, synthesize a cell: the LIVE cell
# dir (real recipe.yaml + rendered/) plus the committed curated fixture.
# WHAT THE FALLBACK ASSERTS, precisely: with a real cell we compare against COMMITTED bytes. Synthesized,
# there are no committed bytes to compare to, so we assert DETERMINISM (regenerate twice -> identical) and
# that RESULTS.md renders from the generated record. The committed-byte form re-arms automatically once any
# cell publishes again.
_SYNTH = None
if not cells:
    _fx = ROOT / "scripts" / "fixtures" / "published_cell"
    _src = ROOT / "scripts/fixtures/sample_cells/nemotron-ultra-3-gb300-vllm-agg-pareto-c16"
    if _fx.is_dir() and _src.is_dir():
        import shutil as _sh

        # export_record.py resolves the cell RELATIVE TO the repo root, so the synthesized corpus must
        # live under ROOT (a system tempdir raises). Use a gitignored, always-cleaned scratch dir; it is
        # outside recipes/ so no real scan (catalog, on_pipeline_cells, CI gates) can ever pick it up.
        import atexit as _atexit

        _SYNTH = ROOT / ".selftest-tmp" / "results-check"
        if _SYNTH.exists():
            _sh_pre = __import__("shutil")
            _sh_pre.rmtree(_SYNTH, ignore_errors=True)
        _SYNTH.mkdir(parents=True, exist_ok=True)
        _atexit.register(lambda: __import__("shutil").rmtree(ROOT / ".selftest-tmp", ignore_errors=True))
        _cell = _SYNTH / "recipes" / "synth-cell"
        _sh.copytree(_src, _cell)
        for _stale in ("record.json", "runs"):
            _p = _cell / _stale
            if _p.is_dir():
                _sh.rmtree(_p)
            elif _p.exists():
                _p.unlink()
        _cur = _cell / "runs" / "ttj01ui" / "curated"
        _cur.mkdir(parents=True)
        for _f in ("goodput_summary.csv", "rungs.csv", "run_meta.json"):
            _sh.copy(_fx / "curated" / _f, _cur / _f)
        _sh.copy(_fx / "index.jsonl", _cell / "runs" / "index.jsonl")
        # generate the record ONCE from the curated data, then prove regeneration is deterministic
        _r1 = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_record.py"),
                str(_cell),
                str(_cur),
                "--out",
                str(_cell / "record.json"),
            ],
            capture_output=True,
            text=True,
        )
        check(
            "synthesized cell: record.json generates from curated data", _r1.returncode == 0, _r1.stderr.strip()[:200]
        )
        if _r1.returncode == 0:
            _first = (_cell / "record.json").read_text()
            _r2 = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/export_record.py"),
                    str(_cell),
                    str(_cur),
                    "--out",
                    str(_cell / "record.json"),
                ],
                capture_output=True,
                text=True,
            )
            check(
                "synthesized cell: record.json regeneration is DETERMINISTIC (byte-identical)",
                _r2.returncode == 0 and (_cell / "record.json").read_text() == _first,
                _r2.stderr.strip()[:200],
            )
            check(
                "synthesized cell: RESULTS.md renders from the generated record",
                bool(pub.render_results_md(_cell).strip()),
            )
        print(
            "  (no on-pipeline cell committed — synthesized one from the fixture; "
            "committed-byte proof re-arms on the next publish)"
        )

check(
    "at least one on-pipeline cell exists, or a synthesized proof cell stands in",
    len(cells) >= 1 or _SYNTH is not None,
    "no on-pipeline cell and no fixture to synthesize from",
)
for cell in cells:
    rel = cell.relative_to(ROOT)
    idx = [json.loads(l) for l in (cell / "runs" / "index.jsonl").read_text().splitlines() if l.strip()]
    curated = cell / (idx[-1].get("curated") or f"runs/{idx[-1]['run_id']}/curated")
    # A. record.json from committed curated (via the real export_record entry point)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "record.json"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/export_record.py"), str(cell), str(curated), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        ok = r.returncode == 0 and out.read_text() == (cell / "record.json").read_text()
        check(f"A {rel}: record.json regenerates from committed curated byte-for-byte", ok, r.stderr.strip()[:160])
    # B. RESULTS.md from committed record.json + aggregate.json (pure render)
    rendered = pub.render_results_md(cell)
    check(
        f"B {rel}: RESULTS.md regenerates from committed record.json byte-for-byte",
        rendered == (cell / "RESULTS.md").read_text(),
    )
    check(f"B {rel}: render_results_md is stable across calls", rendered == pub.render_results_md(cell))
    # C. the CI gate is green
    check(f"C {rel}: results_check.check_cell reports no drift", rc.check_cell(cell, verbose=False) == [])

# ── D. NOTES split ───────────────────────────────────────────────────────────────────────────────────────
START = pub.START
with tempfile.TemporaryDirectory() as _td:
    cell = Path(_td) / "cell"
    cell.mkdir()
    (cell / "recipe.yaml").write_text("envelope: { name: st-notes, scenario: llm-perf, goal: pareto }\n")
    narrative = (
        "This cell measures the pareto frontier. The methodology note here is a human interpretation "
        "that should survive the migration."
    )
    (cell / "RESULTS.md").write_text(f"# st-notes — results\n\n{narrative}\n\n{START}\n_auto_\n<!-- PUBLISH:END -->\n")
    wrote = pub.split_notes(cell)
    notes = cell / "NOTES.md"
    check("D split_notes writes NOTES.md", wrote and notes.is_file())
    ntext = notes.read_text()
    check("D NOTES.md is stamped non-authoritative", "Non-authoritative" in ntext and "no numbers" in ntext.lower())
    check("D NOTES.md carries the migrated narrative", "human interpretation" in ntext)
    check("D NOTES.md drops the H1 + the auto block", "— results" not in ntext and "_auto_" not in ntext)
    # idempotent: a second split never overwrites an existing NOTES.md
    notes_before = ntext
    check(
        "D split_notes is idempotent (returns False, does not overwrite)",
        pub.split_notes(cell) is False and notes.read_text() == notes_before,
    )
    # a cell with no narrative → no NOTES.md fabricated
    cell2 = Path(_td) / "cell2"
    cell2.mkdir()
    (cell2 / "recipe.yaml").write_text("envelope: { name: st2, scenario: llm-perf, goal: pareto }\n")
    (cell2 / "RESULTS.md").write_text(f"# st2 — results\n\n{START}\n_auto_\n<!-- PUBLISH:END -->\n")
    check(
        "D no narrative → no NOTES.md fabricated", pub.split_notes(cell2) is False and not (cell2 / "NOTES.md").exists()
    )

print(("\nFAIL: " + ", ".join(fails)) if fails else "\nselftest_results_check: all checks passed")
sys.exit(1 if fails else 0)

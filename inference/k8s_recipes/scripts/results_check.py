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

"""results_check.py [<root>] — CI drift guard proving RESULTS.md + record.json are regenerable from committed data.

The determinism proof at the heart of the output-data pipeline (docs/OUTPUT-DATA-PIPELINE §3): for every
ON-PIPELINE cell (one carrying a `runs/index.jsonl`), regenerate `record.json` (from the committed curated
dir, via export_record) and `RESULTS.md` (from the committed record.json + aggregate.json, via
publish.render_results_md) and BYTE-DIFF against what's committed. Any drift fails — which is exactly what
proves nothing was hand-edited and no LLM sits in the number-path. Regeneration touches NO run-dir and NO
live git (git is pinned in curated/run_meta.json), so it is a pure function of committed bytes.

Read-only: it regenerates into a temp dir and diffs; it NEVER mutates a tracked file. `--verbose` prints the
first differing lines. Exit 0 = clean, 1 = drift (or a regeneration error).
"""

from __future__ import annotations

import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
import publish as _pub  # noqa: E402 — the pure render_results_md lives here


def on_pipeline_cells(root: Path) -> list:
    """Cells that have opted into the pipeline: they carry a runs/index.jsonl (archive_run wrote it)."""
    return sorted({p.parent.parent for p in (root / "recipes").glob("**/runs/index.jsonl")})


def _diff(label: str, want: str, got: str, verbose: bool) -> str | None:
    if want == got:
        return None
    if not verbose:
        return f"{label}: byte drift ({len(got)} regenerated vs {len(want)} committed)"
    d = list(
        difflib.unified_diff(
            want.splitlines(),
            got.splitlines(),
            fromfile=f"committed/{label}",
            tofile=f"regenerated/{label}",
            lineterm="",
        )
    )
    return f"{label}: drift\n    " + "\n    ".join(d[:40])


def check_cell(cell: Path, verbose: bool) -> list:
    problems = []
    rel = cell.relative_to(ROOT)
    # 1) record.json — regenerated from the committed curated dir (latest run in the index)
    idx = cell / "runs" / "index.jsonl"
    import json

    entries = [json.loads(l) for l in idx.read_text().splitlines() if l.strip()]
    latest = entries[-1]
    curated = cell / (latest.get("curated") or f"runs/{latest['run_id']}/curated")
    committed_record = (cell / "record.json").read_text() if (cell / "record.json").exists() else ""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "record.json"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts/export_record.py"), str(cell), str(curated), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            problems.append(f"{rel}: export_record failed: {r.stderr.strip()[:200]}")
        else:
            p = _diff(f"{rel}/record.json", committed_record, out.read_text(), verbose)
            if p:
                problems.append(p)
    # 2) RESULTS.md — regenerated from the committed record.json + aggregate.json (pure render)
    committed_results = (cell / "RESULTS.md").read_text() if (cell / "RESULTS.md").exists() else ""
    try:
        regenerated = _pub.render_results_md(cell)
    except Exception as e:
        problems.append(f"{rel}: render_results_md raised {e!r}")
    else:
        p = _diff(f"{rel}/RESULTS.md", committed_results, regenerated, verbose)
        if p:
            problems.append(p)
    return problems


def main() -> int:
    argv = sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    pos = [a for a in argv if not a.startswith("-")]
    root = Path(pos[0]).resolve() if pos else ROOT
    cells = on_pipeline_cells(root)
    if not cells:
        print("results-check: no on-pipeline cells (none carry runs/index.jsonl yet) — nothing to verify")
        return 0
    fails = 0
    for c in cells:
        probs = check_cell(c, verbose)
        if probs:
            fails += 1
            for p in probs:
                print(f"DRIFT  {p}")
        else:
            print(f"OK     {c.relative_to(root)} (record.json + RESULTS.md regenerate byte-identically)")
    print(
        f"results-check: {len(cells) - fails}/{len(cells)} on-pipeline cells regenerate from committed data "
        "byte-for-byte" + ("" if not fails else " — DRIFT above (a tracked file was hand-edited or is stale)")
    )
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

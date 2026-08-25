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

"""Build and display the result summary for a completed run.

The command aggregates local artifacts, evaluates configured references, and writes the summary files. It reports a measurement without a pass/fail verdict when no reference is configured, and reports missing or invalid inputs explicitly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# exemplar_check verdicts → (glyph, one-line meaning). Anything not listed renders as UNKNOWN rather than
# being silently treated as a pass.
_VERDICT_GLYPH = {
    "EXEMPLAR": ("✅", "EXEMPLAR"),
    "NOT_EXEMPLAR": ("❌", "NOT EXEMPLAR"),
    "NO_REFERENCE": ("📊", "MEASURED"),
    "NO_SLA_RUNG": ("❌", "NO SLA RUNG"),
    "ABOVE_RANGE": ("🟡", "ABOVE RANGE"),
    "NO_PARETO_GEOMEAN": ("🟡", "NO PARETO GEOMEAN"),
}


def _fmt(v, nd=1):
    """Format a number for display, or '?' when it is genuinely absent."""
    if v is None:
        return "?"
    try:
        s = f"{float(v):.{nd}f}"
        # Strip trailing decimal zeros (e.g. "1.50" → "1.5"), but only
        # when a decimal point is present — without this guard, nd=0
        # produces e.g. "240" and rstrip("0") eats the significant
        # trailing zero → "24" (the concurrency-240-shows-as-c=24 bug).
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s or "0"
    except (TypeError, ValueError):
        return str(v)


def render_result_block(payload: dict, run_dir: str = "", csv_present: bool = True) -> str:
    """PURE — the operator-facing result block. `payload` is exemplar_check's --json dict, or a dict with
    an `error` key when we could not get one.

    Returns the text to print. Never returns '' — a run that finished always says something, because
    silence after a 35-minute benchmark reads as success."""
    lines = [
        "",
        "──── Result ─────────────────────────────────────────────────────────────",
    ]

    # An EMPTY payload is not a result. `{}` used to slip past this guard and render `⚠ ? = ?` with a
    # fabricated shape — a Result block built entirely out of absent fields reads as a measurement that
    # came out blank, rather than as an evaluation that never happened.
    err = (
        (payload.get("error") or ("no result payload" if not payload else ""))
        if isinstance(payload, dict)
        else "result payload was not an object"
    )
    if err:
        lines.append(f"  ⚠  UNKNOWN — could not evaluate this run: {err}")
        if run_dir:
            lines.append(f"     artifacts are on disk: {run_dir}")
            lines.append(f"     retry by hand: llmb-k8s analyze <cell> {run_dir}")
        lines.append("     This is 'we could not measure it', NOT 'the run was fine'.")
        return "\n".join(lines)

    metric = payload.get("metric") or "?"
    unit = payload.get("unit") or ""
    value = payload.get("value")
    ref = payload.get("reference")
    tol = payload.get("tolerance_pct")
    verdict = payload.get("verdict") or ""
    delta = payload.get("delta_pct")
    glyph, label = _VERDICT_GLYPH.get(verdict, ("⚠", f"UNKNOWN ({verdict or 'no verdict'})"))

    if value is None:
        # NO NUMBER AT ALL — and this is the DOMINANT path: 30 of the 31 catalog cells have no reference,
        # so `value=None, reference=None` was hitting the no-bar branch below and rendering
        #     ❌ tps_per_gpu = ?   no bar set — this run cannot pass or fail, only report
        # which is wrong three ways. The stated cause is wrong (no rung produced a number; the bar is
        # irrelevant), a VERDICT GLYPH sits beside a bare value — the one thing this module's own honesty
        # rules forbid — and `❌` beside "cannot pass or fail" contradicts itself. `run.sh` names "a SILENT
        # 0-scored run" as a known hazard; this is what that looks like on screen.
        lines.append(f"  📊 {metric} = ?" + (f"  [{unit}]" if unit else ""))
        lines.append(
            "     NO NUMBER — aggregation produced no value for this metric "
            "(no rung completed, or the metric is absent from metrics_summary.csv)."
        )
        lines.append("     This is 'we could not measure it', NOT 'the run was fine'.")
    elif ref is None and verdict == "NO_REFERENCE":
        # NO BAR, but a real measurement. Print it and say plainly that it cannot pass or fail. The glyph is
        # HARD-CODED, not taken from the verdict map: a stray verdict must never colour a bare number.
        lines.append(f"  📊 {metric} = {_fmt(value, 4)}" + (f"  [{unit}]" if unit else ""))
        lines.append(
            "     no bar set — this run cannot pass or fail, only report "
            f"(envelope.exemplar.reference is null for this cell)"
        )
    elif ref is None:
        # A verdict that claims to have judged something against a bar that is not there. Do not guess which
        # half is wrong — say the payload is inconsistent and refuse to render either as fact.
        lines.append(f"  ⚠  UNKNOWN — verdict '{verdict or 'none'}' but no reference to compare against")
        lines.append(f"     measured {metric} = {_fmt(value, 4)}" + (f"  [{unit}]" if unit else ""))
    else:
        d = f"{delta:+.2f}%" if isinstance(delta, (int, float)) else "?"
        lines.append(
            f"  {glyph} {metric} {_fmt(value, 4)} vs bar {_fmt(ref, 4)} "
            f"({d}, ±{_fmt(tol)}%) {label}" + (f"  [{unit}]" if unit else "")
        )

    pts = payload.get("per_point") or []
    if pts:
        rungs = " · ".join(f"c={_fmt(p.get('concurrency'), 0)} {_fmt(p.get('g'), 4)}" for p in pts)
        lines.append(f"     rungs: {rungs}")
    if payload.get("value_source"):
        lines.append(f"     source: {payload['value_source']}")
    if not csv_present:
        lines.append(
            "     ⚠ metrics_summary.csv is NOT in the run directory — the numbers above came from "
            "an in-memory aggregation only"
        )
    return "\n".join(lines)


def _run_json(args: list) -> tuple[int, dict]:
    """Run a script that supports --json and parse it. A non-zero exit is NOT itself an error here:
    exemplar_check exits 1 for NOT_EXEMPLAR, which is a real verdict we must render, not a failure.
    """
    try:
        p = subprocess.run(
            [sys.executable, *[str(a) for a in args]],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as e:
        return (2, {"error": f"{type(e).__name__}: {e}"})
    out = (p.stdout or "").strip()
    try:
        return (p.returncode, json.loads(out))
    except Exception:
        detail = (p.stderr or out or "no output").strip().splitlines()
        return (2, {"error": (detail[-1] if detail else "no output")[:200]})


def cell_scenario(cell: str) -> str:
    """envelope.scenario for a cell, or '' when unreadable. Decides which aggregator owns this lane."""
    try:
        import yaml

        r = yaml.safe_load((Path(cell) / "recipe.yaml").read_text()) or {}
        return str(((r.get("envelope") or {}).get("scenario") or "")).strip()
    except Exception:
        return ""


def evaluate(cell: str, run_dir: str, root: Path = ROOT) -> tuple[dict, bool]:
    """Aggregate (if needed) then evaluate. Returns (payload, csv_present).

    Aggregation runs HERE, not only inside `llmb-k8s analyze`, so metrics_summary.csv exists in the run
    directory for whoever opens it later — it was previously absent until someone ran the aggregator by
    hand, leaving the directory missing the one file a human looks for first."""
    # LANE. Only llm-perf is scored by aggregate_metrics + exemplar_check. For any other lane, say which
    # command owns the verdict rather than rendering a confident blank — an unrecognised lane must not look
    # like a run with nothing to report.
    scen = cell_scenario(cell)
    if scen and scen != "llm-perf":
        return (
            {
                "error": f"this cell is scenario '{scen}', which is scored by its own aggregator — "
                f"run `llmb-k8s analyze {cell} <profile> {run_dir}`"
            },
            False,
        )
    rd = Path(run_dir)
    csv = rd / "metrics_summary.csv"
    if not csv.exists():
        rc, _ = _run_json([root / "analysis/llm-perf/aggregate_metrics.py", rd, "--out", csv])
        if rc not in (0, 1) and not csv.exists():
            return ({"error": f"aggregation failed; {csv} was not written"}, False)
    if not csv.exists():
        return (
            {"error": f"{csv} does not exist (aggregation produced no summary)"},
            False,
        )
    rc, payload = _run_json([root / "analysis/llm-perf/exemplar_check.py", cell, csv, "--json"])
    # `json.loads` happily returns a list, a string or a number. `"error" in payload` then raises TypeError
    # on a scalar — a traceback where a Result block belongs, and `run.sh` calls this under `|| true`, so
    # the run would end having printed NOTHING about what it measured. Non-object JSON is itself the error.
    if not isinstance(payload, dict):
        return (
            {"error": f"exemplar_check: expected a JSON object, got {type(payload).__name__}"},
            True,
        )
    if payload.get("error"):
        payload["error"] = f"exemplar_check: {payload['error']}"
    return (payload, True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render the result of a finished run (aggregate + evaluate).")
    ap.add_argument("cell")
    ap.add_argument("run_dir")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    payload, csv_present = evaluate(a.cell, a.run_dir, Path(a.root))
    print(render_result_block(payload, a.run_dir, csv_present))
    return 0  # NEVER fail the run over its own reporting; the verdict is in the text.


if __name__ == "__main__":
    raise SystemExit(main())

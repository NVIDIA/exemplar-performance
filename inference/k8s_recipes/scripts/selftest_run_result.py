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

"""Verify that a completed run renders a clear result from locally verified metrics.

A bare metric must not imply a verdict, unavailable evidence must render UNKNOWN, and the verdict must
appear before follow-up instructions.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load("run_result")

# Synthetic payload used to exercise verdict rendering without publishing run data.
LIVE = {
    "cell": "example-cell",
    "gpu_type": "example-gpu",
    "metric": "pareto_geomean",
    "unit": "geomean(tps/gpu·tps/user)",
    "value": 103.0,
    "reference": 100.0,
    "tolerance_pct": 5.0,
    "value_source": "geomean_over_rungs",
    "per_point": [{"concurrency": 8.0, "g": 96.0}, {"concurrency": 16.0, "g": 110.0}],
    "verdict": "EXEMPLAR",
    "delta_pct": 3.0,
}


class TestVerdictRendering(unittest.TestCase):
    def test_pass_states_metric_value_bar_delta_tolerance_and_verdict(self):
        out = rr.render_result_block(LIVE)
        for token in ("pareto_geomean", "103", "100", "+3.00%", "±5%", "EXEMPLAR"):
            self.assertIn(token, out, out)

    def test_per_rung_values_are_shown(self):
        out = rr.render_result_block(LIVE)
        self.assertIn("c=8", out)
        self.assertIn("c=8 96", out)
        self.assertIn("c=16", out)
        self.assertIn("c=16 110", out)

    def test_fail_is_unmistakable(self):
        out = rr.render_result_block({**LIVE, "verdict": "NOT_EXEMPLAR", "delta_pct": -45.7, "reference": 400.0})
        self.assertIn("❌", out)
        self.assertIn("NOT EXEMPLAR", out)

    def test_no_reference_says_it_cannot_pass_or_fail(self):
        """A number with a glyph beside it reads as a verdict. When there is no bar, say so in words."""
        out = rr.render_result_block({**LIVE, "reference": None, "verdict": "NO_REFERENCE", "delta_pct": None})
        self.assertIn("no bar set", out)
        self.assertIn("cannot pass or fail", out)
        self.assertIn("103", out)  # the measurement is still reported
        self.assertNotIn("EXEMPLAR", out.replace("NO_REFERENCE", ""))
        self.assertNotIn("✅", out)

    def test_no_value_is_NOT_attributed_to_the_missing_bar(self):
        """Common path: 30 of the 31 catalog cells have no reference, so `value=None, reference=None`
        is the ordinary shape of an unmeasurable run. It used to branch on the REFERENCE first and render
            ❌ tps_per_gpu = ?   no bar set — this run cannot pass or fail, only report
        which states the wrong cause (no rung produced a number; the bar is irrelevant), puts a VERDICT
        GLYPH beside a bare value, and contradicts itself by pairing ❌ with "cannot pass or fail".
        """
        out = rr.render_result_block(
            {
                **LIVE,
                "value": None,
                "reference": None,
                "verdict": "NO_REFERENCE",
                "delta_pct": None,
            }
        )
        self.assertIn("NO NUMBER", out)
        self.assertIn("could not measure it", out)
        self.assertNotIn("no bar set", out)  # the bar is not why there is no number
        for glyph in ("❌", "✅", "🟡"):  # no verdict glyph may sit beside a bare value
            self.assertNotIn(glyph, out)
        self.assertIn("📊", out)

    def test_no_value_never_wears_a_verdict_glyph_whatever_the_verdict_says(self):
        for verdict in ("NOT_EXEMPLAR", "EXEMPLAR", "NO_SLA_RUNG", "", "something-new"):
            out = rr.render_result_block({**LIVE, "value": None, "reference": None, "verdict": verdict})
            self.assertIn("NO NUMBER", out, verdict)
            self.assertNotIn("❌", out, verdict)
            self.assertNotIn("✅", out, verdict)

    def test_a_verdict_with_no_bar_to_judge_against_is_UNKNOWN(self):
        """`reference=None` with a pass/fail verdict is an inconsistent payload. Render neither half as
        fact rather than guessing which one is wrong."""
        out = rr.render_result_block({**LIVE, "reference": None, "verdict": "NOT_EXEMPLAR"})
        self.assertIn("UNKNOWN", out)
        self.assertNotIn("❌", out)

    def test_an_empty_payload_is_an_absent_evaluation_not_a_blank_measurement(self):
        out = rr.render_result_block({})
        self.assertIn("UNKNOWN", out)
        self.assertIn("no result payload", out)

    def test_non_object_json_does_not_traceback(self):
        """`run.sh` calls this under `|| true`, so a TypeError here ends the run having printed NOTHING
        about what it measured — the silence this module exists to prevent."""
        for payload in ([], "boom", 3):
            out = rr.render_result_block(payload)
            self.assertIn("UNKNOWN", out)

    def test_unevaluable_is_UNKNOWN_and_names_the_cause(self):
        out = rr.render_result_block(
            {"error": "aggregation failed; /r/metrics_summary.csv was not written"},
            run_dir="/r",
        )
        self.assertIn("UNKNOWN", out)
        self.assertIn("metrics_summary.csv", out)
        self.assertIn("NOT 'the run was fine'", out)
        self.assertNotIn("✅", out)

    def test_never_silent(self):
        """A run that finished always says something — silence after 35 minutes reads as success."""
        for payload in (
            {},
            {"error": "x"},
            LIVE,
            {**LIVE, "verdict": "WEIRD_NEW_STATE"},
        ):
            self.assertTrue(rr.render_result_block(payload).strip(), payload)

    def test_unrecognised_verdict_is_not_treated_as_a_pass(self):
        out = rr.render_result_block({**LIVE, "verdict": "WEIRD_NEW_STATE"})
        self.assertIn("UNKNOWN", out)
        self.assertNotIn("✅", out)

    def test_missing_csv_is_flagged_even_when_the_numbers_render(self):
        out = rr.render_result_block(LIVE, run_dir="/r", csv_present=False)
        self.assertIn("metrics_summary.csv is NOT in the run directory", out)


class TestLaneRouting(unittest.TestCase):
    def test_non_llm_perf_lane_names_the_owning_command(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cell = Path(td) / "c"
            cell.mkdir()
            (cell / "recipe.yaml").write_text("envelope:\n  name: c\n  scenario: custom-eval\n")
            payload, _ = rr.evaluate(str(cell), td)
        self.assertIn("custom-eval", payload.get("error", ""))
        self.assertIn("llmb-k8s analyze", payload.get("error", ""))


class TestWiring(unittest.TestCase):
    RUN_SH = (SCRIPTS / "run.sh").read_text()

    def test_run_sh_prints_the_result_before_Next(self):
        self.assertIn("run_result.py", self.RUN_SH)
        self.assertLess(
            self.RUN_SH.index("run_result.py"),
            self.RUN_SH.index('echo "Next:"'),
            "the verdict must come BEFORE Next: — 'Next: publish' is not an answer to " "'was my run any good?'",
        )

    def test_reporting_never_fails_the_run(self):
        """A reporting problem is reported, not escalated: main() exits 0 whatever the verdict, so a
        rendering failure can never turn a good run into a failed one."""
        self.assertIn(
            'scripts/run_result.py" "$CELL" "$ROOT/results/$RUN_ID" || true',
            self.RUN_SH,
        )
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rr.main([str(SCRIPTS.parent / "recipes"), "/nonexistent-run-dir"])
        self.assertEqual(rc, 0)
        self.assertIn("UNKNOWN", buf.getvalue())

    def test_no_fetch_says_NOT_EVALUATED_rather_than_nothing(self):
        self.assertIn("NOT EVALUATED", self.RUN_SH)
        self.assertIn("This is not a statement about the run.", self.RUN_SH)

    def test_a_missing_results_dir_is_not_blamed_on_a_flag_nobody_passed(self):
        """One `else` covered BOTH `--no-fetch` and "the directory is not there", so a fetch that RAN and
        landed nothing was reported as a flag the operator never passed — sending them to re-run without a
        flag they had already not used. Two causes, two branches, two messages."""
        self.assertIn('if [ "$NO_FETCH" = 1 ]; then', self.RUN_SH)
        self.assertIn("the fetch ran, but", self.RUN_SH)
        self.assertIn("the transfer failed", self.RUN_SH)

    def test_analyze_accepts_a_run_directory(self):
        cli = (SCRIPTS / "llmb-k8s").read_text()
        self.assertIn('verb == "analyze"', cli)
        self.assertIn("is neither a cluster profile nor an existing directory", cli)


if __name__ == "__main__":
    unittest.main(verbosity=2)

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

"""selftest_rung_coverage.py — BEHAVIOURAL tests for publish.py's rung-coverage guard.

Drives the real `rung_coverage` and the real publish card. A sweep that dies partway still writes a
well-formed run_meta.json, so enumerating the steps that ran reports a truncated run as a clean one —
and then scores it against a full-sweep reference. These assert the distinctions that prevents:

  truncated (rungs missing, nothing recorded why)   -> INCOMPLETE, and never ✅/❌ against a reference
  stopped by design (SLA breach / adaptive ceiling) -> complete, verdict explains why
  unreadable plan                                   -> UNKNOWN, never complete
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish  # noqa: E402


def meta(**kw):
    base = {
        "concurrencies": "1 8 16 32 64",
        "executed_sequence": [1, 8, 16, 32, 64],
        "sweep_mode": "fixed",
        "sweep_stop_reason": "",
        "sweep_steps": [
            {"concurrency": c, "passed": True, "breaches": [], "stop_reason": ""} for c in (1, 8, 16, 32, 64)
        ],
    }
    base.update(kw)
    return base


def truncated():
    """The real shape of a sweep killed mid-flight: 3 of 5 rungs, no reason recorded."""
    return meta(
        executed_sequence=[1, 8, 16],
        sweep_steps=[{"concurrency": c, "passed": True, "breaches": [], "stop_reason": ""} for c in (1, 8, 16)],
    )


class TestRungCoverage(unittest.TestCase):
    def test_full_sweep_is_complete_with_no_verdict(self):
        cov = publish.rung_coverage(meta())
        self.assertIs(cov["complete"], True)
        self.assertEqual(cov["missing"], [])
        self.assertEqual(cov["verdict"], "")

    def test_missing_rungs_with_no_recorded_reason_is_incomplete(self):
        cov = publish.rung_coverage(truncated())
        self.assertIs(cov["complete"], False)
        self.assertEqual(cov["missing"], [32, 64])
        self.assertIn("INCOMPLETE", cov["verdict"])
        self.assertIn("c=32", cov["verdict"])
        self.assertIn("c=64", cov["verdict"])

    def test_breach_in_FIXED_mode_does_NOT_excuse_a_truncated_sweep(self):
        """The fixed-mode loop has no `break`, so a breach can never be why it stopped. And pareto
        cells breach a phantom 100ms TPOT default on real rungs (104-125ms measured) — treating that
        as by-design would disarm this guard on exactly the cells it exists to protect.
        """
        m = truncated()  # sweep_mode == "fixed"
        m["sweep_steps"][-1]["breaches"] = ["tpot"]
        cov = publish.rung_coverage(m)
        self.assertIs(cov["complete"], False, "a fixed sweep cannot stop early on a breach")
        self.assertIn("INCOMPLETE", cov["verdict"])

    def test_breach_in_ADAPTIVE_mode_DOES_excuse_it(self):
        """Adaptive genuinely halts at a breach/ceiling, so there the per-step reason IS the answer."""
        m = truncated()
        m["sweep_mode"] = "adaptive"
        m["sweep_steps"][-1]["breaches"] = ["tpot"]
        cov = publish.rung_coverage(m)
        self.assertIs(cov["complete"], True)
        self.assertIn("breached", cov["verdict"])

    def test_unparseable_step_reason_never_excuses_a_fixed_sweep(self):
        """`stop_reason=unparseable_metrics` means a rung's aiperf CRASHED — the opposite of by-design."""
        m = truncated()
        m["sweep_steps"][-1]["stop_reason"] = "unparseable_metrics"
        self.assertIs(publish.rung_coverage(m)["complete"], False)

    def test_adaptive_empty_concurrency_plan_is_UNKNOWN_not_complete(self):
        """Adaptive mode writes CONCURRENCIES="" — an ABSENT plan. It must not certify coverage."""
        cov = publish.rung_coverage(meta(concurrencies="", sweep_mode="adaptive"))
        self.assertIsNone(cov["complete"], "an empty plan cannot prove nothing was missed")
        self.assertIn("UNKNOWN", cov["verdict"])

    def test_non_sweeping_lane_is_silent_not_UNKNOWN(self):
        """A non-sweeping lane writes no run_meta.json. It has no rungs to be incomplete about, so it must
        NOT be branded UNKNOWN — that would strip the exemplar verdict from every non-sweeping run.
        """
        cov = publish.rung_coverage({})
        self.assertIs(cov["complete"], True)
        self.assertEqual(cov["verdict"], "")
        self.assertEqual(cov["missing"], [])

    def test_recorded_sweep_stop_reason_is_complete_not_truncated(self):
        m = truncated()
        m["sweep_stop_reason"] = "adaptive ceiling reached"
        cov = publish.rung_coverage(m)
        self.assertIs(cov["complete"], True)
        self.assertIn("adaptive ceiling reached", cov["verdict"])

    def test_DECLARED_but_unreadable_plan_is_UNKNOWN_never_complete(self):
        """A run that says it swept but whose plan won't parse cannot certify coverage.

        Note `{}` is deliberately NOT in this list: no sweep keys at all means a non-sweeping lane,
        covered by test_non_sweeping_lane_is_silent_not_UNKNOWN. Conflating the two
        is what branded every non-sweeping run UNKNOWN and stripped its exemplar verdict.
        """
        for bad in (
            {"concurrencies": "1 eight 16"},
            {"concurrencies": "1 8", "executed_sequence": "not-a-list"},
            {"sweep_mode": "fixed", "concurrencies": ""},
        ):
            cov = publish.rung_coverage(bad)
            self.assertIsNone(cov["complete"], f"{bad!r} must be UNKNOWN, not complete")
            self.assertIn("UNKNOWN", cov["verdict"])

    def test_executed_sequence_falls_back_to_sweep_steps(self):
        m = truncated()
        del m["executed_sequence"]
        cov = publish.rung_coverage(m)
        self.assertEqual(cov["executed"], [1, 8, 16])
        self.assertEqual(cov["missing"], [32, 64])

    def test_list_form_of_concurrencies_parses(self):
        cov = publish.rung_coverage(meta(concurrencies=[1, 8, 16, 32, 64]))
        self.assertIs(cov["complete"], True)


class TestPublishCard(unittest.TestCase):
    """The card is what an operator actually reads — assert on its rendered text."""

    def _card(self, run_meta, *, value, reference):
        # The card reads its reference from <cell>/record.json and its rungs from <run_dir>/run_meta.json.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cell, run_dir = root / "cell", root / "run"
            cell.mkdir()
            run_dir.mkdir()
            (run_dir / "run_meta.json").write_text(json.dumps(run_meta))
            (cell / "record.json").write_text(
                json.dumps(
                    {
                        "result": {
                            "metric": "pareto_geomean",
                            "value": value,
                            "unit": "geomean",
                            "reference": reference,
                            "tolerance_pct": 10,
                        },
                        "identity": {},
                        "provenance": {},
                        "detail": {},
                    }
                )
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                publish._print_publish_card(cell, run_dir, "pareto_geomean", value, None, False, "wip")
            out = buf.getvalue()
            # Guard the harness itself: if the reference never reached the card, these tests would
            # pass vacuously (no ✅ to find because there was nothing to compare, not because we
            # suppressed it). Assert the fixture is actually wired.
            if reference is not None:
                self.assertNotIn(
                    "no reference set",
                    out,
                    "HARNESS BUG: reference never reached the card; test is vacuous",
                )
            return out

    def test_truncated_run_names_the_missing_rungs(self):
        out = self._card(truncated(), value=147.0, reference=None)
        self.assertIn("c=32", out)
        self.assertIn("c=64", out)
        self.assertIn("INCOMPLETE", out)

    def test_truncated_run_never_prints_a_pass_against_a_reference(self):
        out = self._card(truncated(), value=147.0, reference=133.8)
        self.assertNotIn("✅", out, "an INCOMPLETE run must never certify against a full-sweep ref")
        self.assertIn("not comparable", out)

    @staticmethod
    def _row(out, label):
        """The single card row starting with `label` — so a row-specific assertion cannot be
        satisfied by the same text appearing in a DIFFERENT row."""
        for line in out.splitlines():
            body = line.strip("║ ").strip()
            if body.startswith(label):
                return body
        return ""

    def test_missing_rungs_named_in_the_RUNGS_row_specifically(self):
        # Guards against the coverage row alone satisfying a whole-card substring check.
        out = self._card(truncated(), value=147.0, reference=None)
        rungs = self._row(out, "rungs")
        self.assertIn("c=32", rungs, f"rungs row must name the missing rung; got {rungs!r}")
        self.assertIn("c=64", rungs)
        self.assertIn("NOT RUN", rungs)

    def test_incomplete_verdict_is_in_the_COVERAGE_row_specifically(self):
        out = self._card(truncated(), value=147.0, reference=None)
        self.assertIn("INCOMPLETE", self._row(out, "coverage"))

    def test_UNKNOWN_coverage_also_refuses_to_certify_against_a_reference(self):
        # complete is None (plan unreadable) — distinct from INCOMPLETE, and equally uncertifiable.
        out = self._card(
            {"sweep_steps": [{"concurrency": 1, "passed": True, "breaches": []}]},
            value=147.0,
            reference=133.8,
        )
        self.assertNotIn("✅", out, "UNKNOWN coverage must never certify against a reference")
        self.assertIn("not comparable", out)
        self.assertIn("UNKNOWN", self._row(out, "exemplar"))

    def test_complete_run_still_gets_its_normal_verdict(self):
        out = self._card(meta(), value=147.0, reference=133.8)
        self.assertIn("✅", out)
        self.assertNotIn("INCOMPLETE", out)
        self.assertNotIn("not comparable", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)

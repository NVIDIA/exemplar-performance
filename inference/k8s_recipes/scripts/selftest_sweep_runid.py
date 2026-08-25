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

"""Tests for resolving sweep leg labels to generated run IDs.

Submission records store both values. Serial execution and collection must use the generated run ID rather than assuming it matches the user-facing leg label.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resilient_status as rs  # noqa: E402


class SweepRunIdTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)
        self.subs = self.root / ".submits"
        self.subs.mkdir()
        self._orig = rs.SUBMITS_DIR
        rs.SUBMITS_DIR = self.subs
        self.addCleanup(lambda: setattr(rs, "SUBMITS_DIR", self._orig))

    def _sweep(self, tags, *, submitted=None):
        root = str(self.root / ".sweeps" / "sw1")
        return {
            "sweep_id": "sw1",
            "namespace": "ns",
            "run_ids": list(tags),
            "legs": [
                {
                    "run_id": t,
                    "tag": t,
                    "cell_name": f"cell-{t}",
                    "submitted_run_id": (submitted or {}).get(t),
                    "scratch_cell": f"{root}/legs/{t}",
                }
                for t in tags
            ],
        }

    def _submit_record(self, run_id, legdir):
        (self.subs / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "cell": legdir}))

    # ---- the bug ----------------------------------------------------------
    def test_a_tag_NEVER_resolves_to_itself(self):
        """The defect in one line. Every consumer keys off the minted id; handing back the tag makes
        the lookup fail downstream, far from the cause."""
        sw = self._sweep(["r174cf19"])
        self._submit_record("ttje16h", f"{self.root}/.sweeps/sw1/legs/r174cf19")
        got = rs._leg_run_id("r174cf19", sw)
        self.assertEqual(got, "ttje16h")
        self.assertNotEqual(got, "r174cf19")

    def test_resolves_from_the_submit_record_path(self):
        """Recovery route for sweeps taken BEFORE submit recorded the id — the mapping survives only
        as the leg's cell path, so a real sweep already on disk stays collectable."""
        sw = self._sweep(["r1aa", "r2aa", "r3aa"])
        for tag, rid in (("r1aa", "ttjx1"), ("r2aa", "ttjx2"), ("r3aa", "ttjx3")):
            self._submit_record(rid, f"{self.root}/.sweeps/sw1/legs/{tag}")
        self.assertEqual([rs._leg_run_id(t, sw) for t in sw["run_ids"]], ["ttjx1", "ttjx2", "ttjx3"])

    def test_the_recorded_id_WINS_over_path_inference(self):
        """Once submit records the id it is authoritative: inference is the fallback, not the source."""
        sw = self._sweep(["r1aa"], submitted={"r1aa": "recorded-id"})
        self._submit_record("inferred-id", f"{self.root}/.sweeps/sw1/legs/r1aa")
        self.assertEqual(rs._leg_run_id("r1aa", sw), "recorded-id")

    # ---- must not guess ---------------------------------------------------
    def test_unresolvable_returns_EMPTY_not_a_guess(self):
        """No record anywhere -> "" so the caller can NAME the unresolved leg. Falling back to the tag
        would send the operator to the cluster to debug a run that was fine."""
        self.assertEqual(rs._leg_run_id("r9zz", self._sweep(["r9zz"])), "")

    def test_a_DIFFERENT_leg_of_the_same_sweep_is_not_matched(self):
        """Suffix matching must be anchored on /legs/<tag>: r1aa must not match a record for r1aab."""
        sw = self._sweep(["r1aa"])
        self._submit_record("wrong", f"{self.root}/.sweeps/sw1/legs/r1aab")
        self.assertEqual(rs._leg_run_id("r1aa", sw), "")

    def test_a_corrupt_submit_record_does_not_abort_resolution(self):
        """One unreadable JSON file must not cost the other legs their mapping."""
        sw = self._sweep(["r1aa"])
        (self.subs / "broken.json").write_text("{not json")
        self._submit_record("ttjok", f"{self.root}/.sweeps/sw1/legs/r1aa")
        self.assertEqual(rs._leg_run_id("r1aa", sw), "ttjok")

    def test_trailing_slash_on_the_cell_path_still_matches(self):
        sw = self._sweep(["r1aa"])
        self._submit_record("ttjok", f"{self.root}/.sweeps/sw1/legs/r1aa/")
        self.assertEqual(rs._leg_run_id("r1aa", sw), "ttjok")


if __name__ == "__main__":
    unittest.main(verbosity=2)

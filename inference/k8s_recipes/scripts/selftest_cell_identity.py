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

"""selftest_cell_identity.py — BEHAVIOURAL tests for the cell-identity gate + a repo-wide scan.

Two jobs, because a guard that only runs on one code path is not a guarantee:

  1. Unit — drive the real `cell_identity.check()` over hand-built cells and assert what it DID.
  2. Scan — assert EVERY committed cell under recipes/ is self-consistent. This is what makes drift
     impossible repo-wide instead of merely detectable by whichever launcher happens to check.

A cell name appears in both `envelope.name` in recipe.yaml and a hardcoded
`metadata.name` in each rendered/*.yaml. Launchers BUILD resource names from the first and APPLY the
second. A partial rename therefore applied a Job named after the PRODUCTION cell, then failed looking
for a Job that never existed, reported as "apiserver lag?". The wrong-identity apply is the dangerous
half: it can collide with a live benchmark.

The real cure is to stop duplicating (render metadata.name from ${LLMB_CELL_NAME} the way $NAMESPACE
and $RUN_ID already are). That rewrites every rendered manifest and rolls every recipe_hash, so until
that trade is worth making, this test is what holds the invariant.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_identity as ci  # noqa: E402

K8S = Path(__file__).resolve().parent.parent
RECIPES = K8S / "recipes"


def _cell(root: Path, name: str, *, manifest_name: str | None = None) -> Path:
    d = root / name
    (d / "rendered").mkdir(parents=True)
    (d / "recipe.yaml").write_text(f"envelope:\n  name: {name}\n  scenario: llm-perf\n")
    obj = manifest_name if manifest_name is not None else f"{name}-bench"
    (d / "rendered" / "bench-job.yaml").write_text(
        f"apiVersion: batch/v1\nkind: Job\nmetadata:\n  name: {obj}\n  labels:\n"
        f"    app: thing\nspec:\n  template:\n    spec:\n      containers:\n"
        f"        - name: bench\n          image: x\n"
    )
    return d


class CellIdentityUnit(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def test_consistent_cell_has_no_problems(self):
        self.assertEqual(ci.check(_cell(self.root, "alpha-cell")), [])

    def test_suffixed_object_names_are_fine(self):
        self.assertEqual(
            ci.check(_cell(self.root, "alpha-cell", manifest_name="alpha-cell-bench")),
            [],
        )

    def test_partial_rename_is_caught(self):
        """A renamed recipe must not retain rendered files from the previous identity."""
        c = _cell(self.root, "probe-cell", manifest_name="production-cell-bench")
        problems = ci.check(c)
        self.assertTrue(problems)
        self.assertIn("production-cell-bench", problems[0])

    def test_foreign_name_is_called_out_specifically(self):
        """Colliding with ANOTHER cell is worse than a generic mismatch — it must say so."""
        c = _cell(self.root, "probe-cell", manifest_name="production-cell-bench")
        problems = ci.check(c, other_cell_names={"production-cell", "probe-cell"})
        self.assertTrue(any("FOREIGN" in p for p in problems))

    def test_unrelated_mismatch_is_flagged_but_not_FOREIGN(self):
        c = _cell(self.root, "probe-cell", manifest_name="typo-cell-bench")
        problems = ci.check(c, other_cell_names={"production-cell", "probe-cell"})
        self.assertTrue(problems)
        self.assertFalse(any("FOREIGN" in p for p in problems))

    def test_templated_name_is_not_drift(self):
        """`${LLMB_CELL_NAME}` is resolved at apply time — the eventual cure, not a defect."""
        c = _cell(self.root, "alpha-cell", manifest_name="${LLMB_CELL_NAME}-bench")
        self.assertEqual(ci.check(c), [])

    def test_unreadable_recipe_is_UNKNOWN_not_pass(self):
        d = self.root / "broken"
        (d / "rendered").mkdir(parents=True)
        (d / "recipe.yaml").write_text("envelope:\n  scenario: llm-perf\n")  # no name
        problems = ci.check(d)
        self.assertTrue(problems)
        self.assertIn("UNKNOWN", problems[0])

    def test_repeated_object_names_report_ONCE_not_per_object(self):
        """A multi-doc manifest names Job + ServiceAccount + Role identically. Three copies of the
        same sentence buries the signal — the operator needs distinct problems, not object counts.
        """
        c = _cell(self.root, "probe-cell", manifest_name="production-cell-bench")
        f = c / "rendered" / "bench-job.yaml"
        f.write_text(
            f.read_text() + "---\napiVersion: v1\nkind: ServiceAccount\nmetadata:\n"
            "  name: production-cell-bench\n"
            "---\napiVersion: rbac.authorization.k8s.io/v1\nkind: Role\nmetadata:\n"
            "  name: production-cell-bench\n"
        )
        self.assertEqual(len(ci.check(c)), 1, "one distinct (file, name) problem, not one per object")

    def test_a_bare_name_outside_metadata_is_not_an_identity(self):
        """Container/port/volume `name:` keys must not be mistaken for object identity."""
        d = _cell(self.root, "alpha-cell")
        f = d / "rendered" / "bench-job.yaml"
        f.write_text(f.read_text() + "        - name: sidecar\n          image: y\n")
        self.assertEqual(ci.check(d), [])


class CommittedCellsScan(unittest.TestCase):
    def test_every_committed_cell_is_self_consistent(self):
        if not RECIPES.is_dir():
            self.skipTest("recipes/ not present")
        others = ci.all_cell_names(RECIPES)
        problems = []
        cells = sorted(RECIPES.rglob("recipe.yaml"))
        for rc in cells:
            problems += ci.check(rc.parent, other_cell_names=others)
        self.assertEqual(
            problems,
            [],
            f"{len(cells)} cells scanned; identity drift:\n" + "\n".join(f"  {p}" for p in problems),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

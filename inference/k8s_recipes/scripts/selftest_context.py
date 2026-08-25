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

"""selftest_context.py — verify the multi-cluster kubectl safety guarantee.

Two checks:
  1. Grep: no bare `kubectl` invocation remains in the owned primitives, except
     the single `kc()` definition line in each bash script.
  2. Unit: reclaim.py's kc() adds --context iff KUBE_CONTEXT is set in the
     environment (mirrors the bash ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"}).

Runs with `python3 scripts/selftest_context.py` or via `make test`.
Exit 0 = all checks pass.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

# ── owned files ───────────────────────────────────────────────────────────────
OWNED_SH = [
    "run.sh",  # the lifecycle orchestrator — its ns/server/teardown must pin KUBE_CONTEXT too
    "deploy.sh",
    "sweep.sh",
    "stage-dataset.sh",
    "idle_guard.sh",
]
OWNED_PY = ["reclaim.py"]

# ── grep patterns for bash files ─────────────────────────────────────────────
# A line is a "bare kubectl invocation" if kubectl appears as a shell command:
#   - starts the line (possibly indented), or
#   - follows a pipe |, subshell $(, process-sub (, brace {, semicolon ;, or & operator.
_BARE_RE = re.compile(r'(?:^|\s*[|$({;&]\s*)kubectl\b')

# Lines to exclude from the check:
#   - comment lines (# ...)
#   - the kc() definition line itself (kc() { kubectl … })
#   - echo / printf statements (human-readable hints that spell out kubectl commands)
_SKIP_RE = re.compile(
    r'^\s*#'  # shell comment
    r'|kc\(\)\s*\{'  # the kc() definition
    r'|^\s*echo\b'  # echo hint text
    r'|^\s*printf\b'  # printf hint text
)


# ── check 1: grep ─────────────────────────────────────────────────────────────
def check_no_bare_kubectl() -> list[str]:
    """Return a list of violation strings (file:line: text) or [] if clean."""
    violations: list[str] = []
    for name in OWNED_SH:
        path = SCRIPTS / name
        if not path.exists():
            violations.append(f"{name}: FILE MISSING")
            continue
        for lineno, raw in enumerate(path.read_text().splitlines(), 1):
            line = raw.strip()
            if _SKIP_RE.search(line):
                continue
            if _BARE_RE.search(line):
                violations.append(f"{name}:{lineno}: {raw.rstrip()}")
    return violations


# ── check 2: unit test for reclaim.py kc() ────────────────────────────────────
def _load_reclaim():
    """Import reclaim.py without executing main()."""
    spec = importlib.util.spec_from_file_location("reclaim", SCRIPTS / "reclaim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestKcArgConstruction(unittest.TestCase):
    """kc() in reclaim.py must mirror bash ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"}."""

    def setUp(self):
        self.reclaim = _load_reclaim()
        # Save and clear env so tests are isolated.
        self._saved = os.environ.pop("KUBE_CONTEXT", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["KUBE_CONTEXT"] = self._saved
        else:
            os.environ.pop("KUBE_CONTEXT", None)

    def test_no_context_gives_plain_kubectl(self):
        """When KUBE_CONTEXT is absent, kc() returns ['kubectl', ...]."""
        os.environ.pop("KUBE_CONTEXT", None)
        result = self.reclaim.kc("get", "pods")
        self.assertEqual(result[0], "kubectl")
        self.assertNotIn("--context", result)
        self.assertIn("get", result)
        self.assertIn("pods", result)

    def test_empty_context_gives_plain_kubectl(self):
        """When KUBE_CONTEXT is empty string, kc() does NOT add --context."""
        os.environ["KUBE_CONTEXT"] = ""
        result = self.reclaim.kc("get", "pods")
        self.assertNotIn("--context", result)

    def test_whitespace_only_context_gives_plain_kubectl(self):
        """When KUBE_CONTEXT is whitespace, kc() strips and does NOT add --context."""
        os.environ["KUBE_CONTEXT"] = "   "
        result = self.reclaim.kc("get", "pods")
        self.assertNotIn("--context", result)

    def test_set_context_prepends_flag(self):
        """When KUBE_CONTEXT is set, kc() adds --context <value> right after kubectl."""
        os.environ["KUBE_CONTEXT"] = "prod-b200"
        result = self.reclaim.kc("get", "pods", "-n", "bench")
        self.assertEqual(result[0], "kubectl")
        self.assertEqual(result[1], "--context")
        self.assertEqual(result[2], "prod-b200")
        # Original args still present
        self.assertIn("get", result)
        self.assertIn("pods", result)
        self.assertIn("-n", result)
        self.assertIn("bench", result)

    def test_context_does_not_duplicate_on_repeated_calls(self):
        """Each kc() call is independent — no global state accumulation."""
        os.environ["KUBE_CONTEXT"] = "ctx-a"
        r1 = self.reclaim.kc("get", "pods")
        os.environ["KUBE_CONTEXT"] = "ctx-b"
        r2 = self.reclaim.kc("get", "pods")
        self.assertEqual(r1[2], "ctx-a")
        self.assertEqual(r2[2], "ctx-b")
        self.assertNotIn("--context", r1[3:])  # no duplicate flag
        self.assertNotIn("--context", r2[3:])

    def test_reclaim_only_job_arg_parse(self):
        """--only-job narrows an apply run without becoming a positional profile arg."""
        a = self.reclaim.parse_args(["gb300", "--only-job", "agent7-dummy", "--apply"])
        self.assertEqual(a.pos, ["gb300"])
        self.assertTrue(a.apply)
        self.assertEqual(a.only_job, "agent7-dummy")

    def test_reclaim_unknown_flag_fails_fast(self):
        """Unknown reclaim flags must not be silently treated as profile names."""
        with self.assertRaises(SystemExit):
            self.reclaim.parse_args(["gb300", "--typo"])


# ── kc() definition present in each bash file ─────────────────────────────────
def check_kc_definition_present() -> list[str]:
    """Each owned bash script must have exactly one kc() definition line."""
    missing: list[str] = []
    _KC_DEF = re.compile(r"kc\(\)\s*\{.*kubectl.*\$\{KUBE_CONTEXT")
    for name in OWNED_SH:
        path = SCRIPTS / name
        if not path.exists():
            missing.append(f"{name}: FILE MISSING")
            continue
        text = path.read_text()
        if not _KC_DEF.search(text):
            missing.append(f"{name}: missing kc() definition")
    return missing


# ── runner ────────────────────────────────────────────────────────────────────
def main() -> int:
    ok = True

    print("selftest_context: checking kc() definitions …")
    def_violations = check_kc_definition_present()
    if def_violations:
        ok = False
        for v in def_violations:
            print(f"  FAIL  {v}")
    else:
        print(f"  ok  all {len(OWNED_SH)} bash scripts have kc() definition")

    print("selftest_context: grep check — no bare kubectl invocations …")
    grep_violations = check_no_bare_kubectl()
    if grep_violations:
        ok = False
        for v in grep_violations:
            print(f"  FAIL  {v}")
    else:
        print(f"  ok  {len(OWNED_SH)} files clean")

    print("selftest_context: unit-testing reclaim.py kc() arg construction …")
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestKcArgConstruction)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    if not result.wasSuccessful():
        ok = False

    if ok:
        print("selftest_context: ALL CHECKS PASSED")
        return 0
    else:
        print("selftest_context: FAILURES — see above")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

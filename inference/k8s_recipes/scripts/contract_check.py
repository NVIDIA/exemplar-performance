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

"""contract_check.py — enforce the recipe contract.

Every cell must ship: recipe.yaml + README.md (with a 'How to reproduce' section) + RESULTS.md +
an envelope.exemplar block. Fails the build if any is missing, so the pattern stays consistent as
agents add recipes.
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("requires: pip install pyyaml")

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cells = sorted({p.parent for p in (ROOT / "recipes").glob("**/recipe.yaml")})
    if not cells:
        print("contract-check: no cells yet")
        return 0
    fails = 0
    for c in cells:
        rel = c.relative_to(ROOT)
        problems = []
        readme = c / "README.md"
        if not readme.exists():
            problems.append("missing README.md")
        elif "How to reproduce" not in readme.read_text():
            problems.append("README.md has no 'How to reproduce' section")
        results = c / "RESULTS.md"
        rtext = results.read_text() if results.exists() else ""
        if not results.exists():
            problems.append("missing RESULTS.md")
        env = (yaml.safe_load((c / "recipe.yaml").read_text()) or {}).get("envelope") or {}
        ex = env.get("exemplar") or {}
        if "exemplar" not in env:
            problems.append("recipe.yaml missing envelope.exemplar (the comparison bar)")
        # performant/exemplar are HUMAN-approved: they must have populated RESULTS + the per-scenario graphs
        # (and exemplar also a committed reference bar). Lower statuses have no committed run yet -> lenient.
        status = env.get("status")
        if status in ("performant", "exemplar"):
            if any(
                m in rtext.lower()
                for m in (
                    "pending canonical",
                    "no canonical",
                    "no perf-data run",
                    "no perf run",
                    "sweep in progress",
                    "top-out — sweep",
                    "_pending",
                )
            ):
                problems.append(f"status={status} but RESULTS.md is still a placeholder (no real numbers)")
            if "## graphs" not in rtext.lower() and "dashboard" not in rtext.lower():
                problems.append(
                    f"status={status} but RESULTS.md has no Graphs/dashboard section (the per-scenario views)"
                )
        if status == "exemplar" and ex.get("reference") in (None, "", "null"):
            problems.append(
                "status=exemplar but envelope.exemplar.reference is null (no committed baseline — run exemplar_check --set)"
            )
        if problems:
            fails += 1
            print(f"FAIL {rel}: " + "; ".join(problems))
        else:
            print(f"OK   {rel}")
    print(f"contract-check: {len(cells) - fails}/{len(cells)} cells satisfy the recipe contract")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

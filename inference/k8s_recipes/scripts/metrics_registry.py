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

"""Resolve common serving metrics across runtime-specific Prometheus names.

A missing metric returns an unknown reading rather than zero. Runtime mappings live in
``serving/metrics_registry.json`` and should be added only after checking a real metrics scrape.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "serving" / "metrics_registry.json"


class Reading(NamedTuple):
    """A metric reading that carries WHY it is absent, so absence never renders as a value."""

    value: float | None
    family: str  # which series actually supplied it ("" when unknown)
    why: str  # "" when known; otherwise the reason no value exists

    @property
    def known(self) -> bool:
        return self.value is not None


def load_registry(path: Path | None = None) -> dict:
    return json.loads((path or REGISTRY_PATH).read_text())


def known_runtimes(registry: dict | None = None) -> list[str]:
    return sorted((registry or load_registry())["runtimes"])


def candidates(runtime: str, concept: str, registry: dict | None = None) -> list[str]:
    """Series names to try, in priority order. Empty list = this runtime has no evidenced source."""
    reg = registry or load_registry()
    rt = reg["runtimes"].get(runtime)
    if rt is None:
        return []
    return [c for c in rt.get(concept, []) if isinstance(c, str)]


def sum_family(surface: str, family: str) -> float | None:
    """Sum every series of ONE metric family across its labels. None if the family is absent.

    Deliberately excludes `_bucket`/`_sum`/`_count`/`_created` suffixes: a histogram bucket is a
    request count and `_created` is a unix timestamp (~1.8e9) — folding either into a token counter
    swamps the real value.
    """
    pattern = re.compile(rf"^{re.escape(family)}(\{{[^}}]*\}})?[ \t]+([0-9eE+.\-]+)\s*$", re.M)
    total, seen = 0.0, False
    for m in pattern.finditer(surface):
        try:
            total += float(m.group(2))
        except ValueError:
            continue
        seen = True
    return total if seen else None


def resolve(surface: str, runtime: str, concept: str, registry: dict | None = None) -> Reading:
    """First PRESENT family wins; series within it are summed, families are never summed together.

    A frontend re-exports the same tokens its workers also export — summing across families
    double-counts, and a worker restart then makes the total FALL, which reads as a stall.
    """
    reg = registry or load_registry()
    if runtime not in reg["runtimes"]:
        return Reading(
            None,
            "",
            f"unknown runtime {runtime!r}; known: {', '.join(known_runtimes(reg))}",
        )
    names = candidates(runtime, concept, reg)
    if not names:
        return Reading(
            None,
            "",
            f"no metric name is registered for {runtime}/{concept} "
            f"(capture a real /metrics surface, then fill it in)",
        )
    for family in names:
        value = sum_family(surface, family)
        if value is not None:
            return Reading(value, family, "")
    return Reading(None, "", f"none of {names} present on this /metrics surface")


def coverage(surface: str, runtime: str, registry: dict | None = None) -> dict[str, Reading]:
    """Every concept resolved against one surface — the honest per-runtime capability report."""
    reg = registry or load_registry()
    return {c: resolve(surface, runtime, c, reg) for c in reg["concepts"]}


def _main(argv: list[str]) -> int:
    """CLI: metrics_registry.py <runtime> [surface.prom] — show what IS and IS NOT measurable."""
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print(f"  runtimes: {', '.join(known_runtimes())}")
        return 0
    runtime = argv[0]
    surface = Path(argv[1]).read_text() if len(argv) > 1 else ""
    print(f"runtime: {runtime}")
    for concept, r in coverage(surface, runtime).items():
        if r.known:
            print(f"  ✓ {concept:24} {r.value:>18,.2f}   [{r.family}]")
        else:
            print(f"  ? {concept:24} {'UNKNOWN':>18}   {r.why}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))

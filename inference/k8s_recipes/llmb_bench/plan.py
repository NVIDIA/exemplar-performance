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

"""llmb_bench.plan — the sweep rung plan (fixed list or adaptive grid).

The bench-job.yaml.j2 shell currently re-implements the adaptive exponential ramp + binary search that
scripts/sweep_planner.py already owns (and unit-tests). This module is the single entry point the runner
uses, DELEGATING to sweep_planner rather than duplicating it — collapsing that divergence risk.
(M2 will vendor sweep_planner into this package for the baked image; for now we import the committed one.)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reuse the committed, unit-tested planner as the shared implementation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import sweep_planner  # noqa: E402


def fixed_rungs(concurrencies: str) -> list[int]:
    """sweep_mode=fixed: the exact rung list from bench.sweep_concurrency (space-separated ints)."""
    return [int(x) for x in (concurrencies or "").split() if x.strip()]


def adaptive_grid(start: int, ratio: float, lo: int, hi: int) -> list[int]:
    """sweep_mode=adaptive: the geometric candidate ladder (anchored on start + both bounds), clamped to
    [lo, hi]. Straight delegation to sweep_planner.build_grid — no re-implementation."""
    return sweep_planner.build_grid(start, ratio, lo, hi)

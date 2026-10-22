# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

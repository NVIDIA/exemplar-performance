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

"""Statistical helpers for pooled published results.

`pooled()` returns a mean and confidence interval when enough runs are available. `pooling_key()` prevents results with different workload or scoring settings from being combined. The module performs no I/O.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = [
    "pooled",
    "required_n",
    "single_run_interval_pct",
    "ci_within_tolerance",
    "pooling_key",
    "POOLING_KEY_FIELDS",
    "t_quantile_975",
    "format_pooled",
]

# Exact two-sided 97.5% Student-t quantiles by degrees of freedom (n-1).
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
}
_Z975 = 1.95996398454

# The fields that must ALL match before two values may enter the same pool.
# Adding a field here is always safe; removing one is a methodology change.
POOLING_KEY_FIELDS = (
    "cell",  # same recipe cell
    "recipe_hash",  # same fingerprinted configuration
    "metric",  # same metric name
    "rung",  # concurrency rung: c16 / c32 / c64 — different noise floors
    "context_len",  # 256k vs 1M — report separately even when indistinguishable
    "attempts_per_task",  # sets the metric's own sampling noise floor
    "scoring_policy",  # persist (ctx-exhausted counted at 0.0) vs rich (excluded as infra)
)


def t_quantile_975(df: int) -> float:
    """Two-sided 97.5% Student-t quantile for `df` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be >= 1")
    if df in _T975:
        return _T975[df]
    if df > 100:
        return _Z975
    # linear interpolation between tabulated neighbours (monotone, conservative enough)
    keys = sorted(_T975)
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    f = (df - lo) / (hi - lo)
    return _T975[lo] + f * (_T975[hi] - _T975[lo])


def pooled(values: Sequence[float]) -> dict | None:
    """Pool replicate measurements into {n, mean, sd, cv_pct, se, t, ci_lo, ci_hi, ci_half_pct}.

    n == 0 -> None.  n == 1 -> honest single-run record: mean set, sd/ci all None, so no caller
    can render a fabricated zero-width band.
    """
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return None
    mean = math.fsum(vals) / n
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": None,
            "cv_pct": None,
            "se": None,
            "t": None,
            "ci_lo": None,
            "ci_hi": None,
            "ci_half_pct": None,
            "single_run": True,
        }
    var = math.fsum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    t = t_quantile_975(n - 1)
    half = t * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "cv_pct": (100.0 * sd / mean) if mean else None,
        "se": se,
        "t": t,
        "ci_lo": mean - half,
        "ci_hi": mean + half,
        "ci_half_pct": (100.0 * half / mean) if mean else None,
        "single_run": False,
    }


def single_run_interval_pct(sigma_pct: float) -> float:
    """95% interval half-width, in %, carried by a SINGLE run drawn from a sigma_pct population.

    This is the number that makes the case: sigma 12.6% -> a single run is +/-24.7%.
    """
    return _Z975 * float(sigma_pct)


def required_n(sigma_pct: float, tol_pct: float, max_n: int = 400) -> int:
    """Smallest n whose 95% CI half-width (Student-t) is <= tol_pct, at population sigma_pct.

    Solved by iteration rather than the closed form because t depends on n.
    """
    sigma_pct = float(sigma_pct)
    tol_pct = float(tol_pct)
    if tol_pct <= 0:
        raise ValueError("tol_pct must be > 0")
    if sigma_pct <= 0:
        return 1
    for n in range(2, max_n + 1):
        if t_quantile_975(n - 1) * sigma_pct / math.sqrt(n) <= tol_pct:
            return n
    return max_n


def ci_within_tolerance(pool: dict, reference: float, tol_pct: float, higher_is_better: bool = True) -> dict:
    """CI-based replacement for the point-value tolerance gate.

    Point comparison asks "is this one draw within +/-tol of the reference?" — which a noisy
    metric answers at random. This asks the decidable question instead:

      PASS       the pooled CI lies entirely inside the tolerance band -> proven within tolerance
      FAIL       the pooled CI lies entirely OUTSIDE it -> proven a real regression
      INCONCLUSIVE  the CI straddles the band edge -> the data cannot decide; collect more runs

    An INCONCLUSIVE verdict is a feature: it is the honest answer that the old point gate hid.
    """
    if pool is None:
        return {"verdict": "NO_DATA", "n": 0}
    if reference in (None, 0):
        return {"verdict": "NO_REFERENCE", "n": pool["n"]}
    floor = reference * (1.0 - tol_pct / 100.0)
    ceil_ = reference * (1.0 + tol_pct / 100.0)
    lo, hi = pool.get("ci_lo"), pool.get("ci_hi")
    out = {
        "n": pool["n"],
        "mean": pool["mean"],
        "reference": reference,
        "tolerance_pct": tol_pct,
        "band_lo": floor,
        "band_hi": ceil_,
        "ci_lo": lo,
        "ci_hi": hi,
        "delta_pct": 100.0 * (pool["mean"] - reference) / reference,
    }
    if lo is None:  # n == 1: structurally undecidable, do not pretend otherwise
        out["verdict"] = "INCONCLUSIVE_SINGLE_RUN"
        return out
    if higher_is_better:
        if lo >= floor:
            out["verdict"] = "PASS"
        elif hi < floor:
            out["verdict"] = "FAIL"
        else:
            out["verdict"] = "INCONCLUSIVE"
    else:
        if hi <= ceil_:
            out["verdict"] = "PASS"
        elif lo > ceil_:
            out["verdict"] = "FAIL"
        else:
            out["verdict"] = "INCONCLUSIVE"
    return out


def pooling_key(entry: dict) -> tuple:
    """The comparability tuple. Two values may share a pool only if their keys are equal."""
    return tuple(entry.get(f) for f in POOLING_KEY_FIELDS)


def assert_poolable(entries: Iterable[dict]) -> list[str]:
    """Return a list of human-readable reasons why `entries` must NOT be pooled ([] == poolable)."""
    keys = {}
    for e in entries:
        keys.setdefault(pooling_key(e), []).append(e.get("run_id"))
    if len(keys) <= 1:
        return []
    reasons = []
    for i, f in enumerate(POOLING_KEY_FIELDS):
        vals = {k[i] for k in keys}
        if len(vals) > 1:
            reasons.append(f"{f} differs across runs: {sorted(map(str, vals))}")
    return reasons or ["runs differ on an unnamed pooling-key field"]


def format_pooled(pool: dict, unit: str = "", places: int = 5) -> str:
    """One-line human rendering that ALWAYS states n and never hides a single-run headline."""
    if pool is None:
        return "no data"
    u = f" {unit}" if unit else ""
    if pool["n"] == 1:
        return f"{pool['mean']:.{places}g}{u} (n=1 — SINGLE RUN, no confidence interval)"
    return (
        f"{pool['mean']:.{places}g}{u}  n={pool['n']}  "
        f"95% CI [{pool['ci_lo']:.{places}g}, {pool['ci_hi']:.{places}g}] "
        f"(+/-{pool['ci_half_pct']:.1f}%)  sd={pool['cv_pct']:.1f}%"
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import json

    demo = [
        0.31353,
        0.33975,
        0.36452,
        0.37179,
        0.39629,
        0.39713,
        0.40258,
        0.41663,
        0.42665,
        0.43267,
        0.43623,
        0.46296,
        0.46455,
        0.48631,
        0.49219,
    ]
    p = pooled(demo)
    print(format_pooled(p))
    print("single-run 95pct interval at that sigma: +/-{:.1f}%".format(single_run_interval_pct(p["cv_pct"])))
    print("n required for +/-5%: {}".format(required_n(p["cv_pct"], 5.0)))
    print(json.dumps(ci_within_tolerance(p, 0.46296, 5.0), indent=1))

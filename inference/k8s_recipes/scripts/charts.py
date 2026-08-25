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

"""charts.py — deterministic, goal-keyed result charts computed ONLY from committed data.

A pure renderer dispatched on ``(record.identity.scenario, record.identity.goal)`` → the RIGHT chart for
that lane, built exclusively from a cell's ``record.json`` (``detail.rungs[]`` + ``result``) and its
``runs.jsonl`` ledger. The SAME committed data ALWAYS yields byte-identical output — no timestamps, no
wall-clock, no ambient state — so a re-render never churns the tree and CI can diff it.

Two renderings per chart:
  (a) an inline ASCII chart — the diffable source of truth embedded in RESULTS.md;
  (b) a PNG (BEST-EFFORT) — matplotlib Agg backend, fixed DPI, stripped metadata, sorted inputs → byte-stable
      across re-runs. If matplotlib is unavailable the PNG is skipped (logged) and the ASCII still emits, so
      the repo stays fully usable without the optional dependency.

Lanes (keyed on scenario · goal):
  - llm-perf · max-concurrency-sla : 2D SLA plane — x=TTFT p50, y=TPOT p50, each rung a labeled POINT; the
                                     two limits cut it into GREEN (both met) / RED (one fails) / DARK-RED
                                     (both fail) quadrants. ASCII renders the same zones as a per-rung view.
  - llm-perf · pareto              : the tps/gpu (y) vs tps/user (x) frontier; the pareto_geomean annotated.

Per-point variance: when a rung carries repeat statistics (``rung["repeats"][<field>] = {mean,min,max,n}``,
persisted by the --repeat consolidate/record path), the point is drawn with an ERROR BAR (min…max, n). A rung
with n=1 (no repeats block) is drawn as a PLAIN point — never a fabricated band (honesty rule). The scalar
goal-metric band (mean ± spread across same-cluster repeats, via ``analysis/compare.runs_within_cluster``)
is shown on the headline when ≥2 valued runs exist.

CLI:
  charts.py <cell-dir> [--no-png]     print the ASCII chart; write charts/<goal>.png best-effort
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# small deterministic helpers
# ─────────────────────────────────────────────────────────────────────────────


def _f(v):
    """float(v) or None — NaN is treated as missing (NaN comparisons silently mislead)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def _bar_cells(value, vmax, width):
    """A width-char list of block/space cells filled proportionally to value/vmax. Deterministic."""
    if not vmax or value is None or value <= 0:
        n = 0
    else:
        n = int(round((value / vmax) * width))
    n = max(0, min(width, n))
    return ["█" if i < n else " " for i in range(width)]


def _col(value, vmax, width):
    """The integer column (0..width-1) a value maps to on the bar axis, or None."""
    if not vmax or value is None:
        return None
    c = int(round((value / vmax) * width))
    return max(0, min(width - 1, c))


def _rung_stat(rung, field):
    """(value, lo, hi, n) for a rung field: prefer the repeats block (per-point variance) when present,
    else the plain scalar with n=1 and no band. NEVER fabricates a band for n=1 (honesty rule).
    """
    reps = (rung.get("repeats") or {}).get(field) if isinstance(rung.get("repeats"), dict) else None
    if isinstance(reps, dict) and (reps.get("n") or 0) >= 2:
        return (
            _f(reps.get("mean")),
            _f(reps.get("min")),
            _f(reps.get("max")),
            int(reps.get("n")),
        )
    return _f(rung.get(field)), None, None, 1


def _sla_zone(ttft, tpot, ttft_limit, tpot_limit):
    """Classify a rung into an SLA quadrant of the (TTFT × TPOT) plane.

    Returns ``(code, failed)`` where ``code`` ∈ {"pass","one","both","unknown"} and ``failed`` lists the
    axis names that exceeded their limit:
      - "pass" (GREEN, bottom-left)  : TTFT ≤ ttft_limit AND TPOT ≤ tpot_limit — both SLAs met.
      - "one"  (RED, adjacent quads) : exactly ONE axis exceeds its limit (the other is met).
      - "both" (DARK-RED, top-right) : BOTH axes exceed — both SLAs failed.
      - "unknown"                    : a value or a limit is missing (never guessed).
    Pure + deterministic; the shared implementation shared by the ASCII rows and the PNG scatter.
    """
    if ttft is None or tpot is None or ttft_limit is None or tpot_limit is None:
        return "unknown", []
    failed = []
    if ttft > ttft_limit:
        failed.append("TTFT")
    if tpot > tpot_limit:
        failed.append("TPOT")
    return {0: "pass", 1: "one", 2: "both"}[len(failed)], failed


def _interp_lin(c_lo, v_lo, c_hi, v_hi, limit):
    """Concurrency where a metric line crosses ``limit``, linear between two bracketing rungs — the SAME
    formula as analysis/llm-perf/exemplar_check._interp. We never re-derive the crossing with a different
    method; we only re-project the committed bracket onto the chart."""
    return c_hi if v_hi == v_lo else c_lo + (c_hi - c_lo) * (limit - v_lo) / (v_hi - v_lo)


def _max_passing_conc(rungs, ttft_limit, tpot_limit):
    """Highest concurrency whose rung is both-pass (the VERIFIED last-pass floor — a conservative ceiling).
    None when no rung passes. Deterministic."""
    best = None
    for r in rungs:
        conc = _f(r.get("concurrency"))
        tf, _a, _b, _c = _rung_stat(r, "ttft_p50_ms")
        tp, _d, _e, _g = _rung_stat(r, "tpot_p50_ms")
        if conc is not None and _sla_zone(tf, tp, ttft_limit, tpot_limit)[0] == "pass":
            best = conc if best is None else max(best, conc)
    return best


def _crossing_point(record):
    """Re-project ``result.crossing`` (computed by exemplar_check.sla_crossing) onto the TTFT×TPOT plane from
    committed data ONLY, using the identical linear interpolation.

    Always echoes the committed crossing meta: ``{status, note, bracket, binding, ratio}``. When ``status ==
    "ok"`` and the two bracketing rungs + the binding limit resolve, ALSO returns the geometry:
      - ``conc`` : the interpolated crossing concurrency (matches sla_crossing's c_star for the binding axis),
      - ``x``,``y`` : the point where the last-pass→first-fail segment meets the BINDING limit line
                      (y = tpot_limit if binding=TPOT; x = ttft_limit if binding=TTFT).
    Degenerate / non-"ok" / missing data → the geometry keys are omitted so NO bogus marker is drawn.
    """
    res = record.get("result") or {}
    crossing = res.get("crossing") or {}
    out = {k: crossing.get(k) for k in ("status", "note", "bracket", "binding", "ratio")}
    status, binding = crossing.get("status"), crossing.get("binding")
    bracket = crossing.get("bracket") or []
    if status != "ok" or len(bracket) != 2 or binding not in ("TTFT", "TPOT"):
        return out
    sla = res.get("sla") or {}
    ttft_limit, tpot_limit = _f(sla.get("ttft_ms")), _f(sla.get("tpot_ms"))
    by_conc = {_f(r.get("concurrency")): r for r in (record.get("detail") or {}).get("rungs") or []}
    lo, hi = by_conc.get(_f(bracket[0])), by_conc.get(_f(bracket[1]))
    if lo is None or hi is None:
        return out
    c_lo, c_hi = _f(bracket[0]), _f(bracket[1])
    tf_lo, tf_hi = _rung_stat(lo, "ttft_p50_ms")[0], _rung_stat(hi, "ttft_p50_ms")[0]
    tp_lo, tp_hi = _rung_stat(lo, "tpot_p50_ms")[0], _rung_stat(hi, "tpot_p50_ms")[0]
    if None in (c_lo, c_hi, tf_lo, tf_hi, tp_lo, tp_hi):
        return out
    if binding == "TPOT" and tpot_limit is not None and tp_hi != tp_lo:
        t = (tpot_limit - tp_lo) / (tp_hi - tp_lo)
        out.update(
            conc=_interp_lin(c_lo, tp_lo, c_hi, tp_hi, tpot_limit),
            x=tf_lo + (tf_hi - tf_lo) * t,
            y=tpot_limit,
        )
    elif binding == "TTFT" and ttft_limit is not None and tf_hi != tf_lo:
        t = (ttft_limit - tf_lo) / (tf_hi - tf_lo)
        out.update(
            conc=_interp_lin(c_lo, tf_lo, c_hi, tf_hi, ttft_limit),
            x=ttft_limit,
            y=tp_lo + (tp_hi - tp_lo) * t,
        )
    return out


def _value_is_crossing(res, cp, max_pass):
    """Is the headline metric value the INTERPOLATED crossing (True) or the VERIFIED last-pass rung floor
    (False)? Prefer the explicit ``value_source``; when it's absent (older records), infer from the data —
    the value is the crossing if it sits nearer the interpolated crossing than the last-pass floor.
    """
    vsrc = (res.get("value_source") or "").lower()
    if "interpol" in vsrc:
        return True
    if "rung" in vsrc or "passing" in vsrc:
        return False
    value, conc = _f(res.get("value")), _f(cp.get("conc"))
    if value is None or conc is None:
        return False
    if max_pass is None:
        return abs(value - conc) < 0.5
    return abs(value - conc) <= abs(value - max_pass)


def _scalar_band(runs, metric):
    """Headline scalar band from the runs.jsonl ledger: mean ± spread% across same-cluster repeats.
    Reuses analysis/compare.runs_within_cluster (the N-repeats-on-one-cluster reproducibility read) so the
    band matches /compare-results exactly. Returns a short annotation string, or '' when <2 valued runs.
    """
    if not runs:
        return ""
    try:
        sys.path.insert(0, str(ROOT / "analysis"))
        import compare as _cmp  # noqa: E402
    except Exception:
        return ""
    higher = _cmp.HIGHER_BETTER.get(metric, True) if hasattr(_cmp, "HIGHER_BETTER") else True
    _, obj = _cmp.runs_within_cluster(runs, metric, 5.0, higher)
    if not obj or not obj.get("clusters"):
        return ""
    g = max(obj["clusters"], key=lambda c: c.get("n", 0))
    return f"band: mean {g['mean']:.4g} · spread {g['spread_pct']:.1f}% over {g['n']} repeats on {g['cluster']}"


# ─────────────────────────────────────────────────────────────────────────────
# ASCII renderers (one per lane) — the diffable source of truth
# ─────────────────────────────────────────────────────────────────────────────

_WIDTH = 40


def _ascii_header(title, headline):
    out = [f"```text", title]
    if headline:
        out.append(headline)
    out.append("")
    return out


_ZONE_VERDICT = {
    "pass": "✅ both-pass",
    "both": "❌ both-fail",
    "unknown": "· n/a (missing value)",
}


def _sla_crossing_lines(cp):
    """The interpolated-crossing lines for the ASCII header: the bracket + binding + the honest committed
    caveat (``crossing.note``). When there's no clean bracket (status != 'ok') say so instead of implying a
    point. Empty list when the record carries no crossing block."""
    status = cp.get("status")
    if not status:
        return []
    if status == "ok" and cp.get("conc") is not None:
        b = cp.get("bracket") or []
        lo = f"{_f(b[0]):g}" if len(b) == 2 and _f(b[0]) is not None else "?"
        hi = f"{_f(b[1]):g}" if len(b) == 2 and _f(b[1]) is not None else "?"
        lines = [
            f"  SLA crossing ≈ c{_fmt(cp['conc'],0)} · bracket [{lo}–{hi}], binding {cp.get('binding')}"
            "  (interpolated on the last-pass → first-fail segment)"
        ]
        if cp.get("note"):
            lines.append(f"  note: {cp['note']}")
        return lines
    return [f"  SLA crossing: {cp.get('note') or status}  (no clean bracket — no crossing marker drawn)"]


def _ascii_sla(record, runs):
    """The SLA has TWO dimensions (TTFT and TPOT), so the source-of-truth ASCII is a per-rung 2D-zone view
    (a legible text stand-in for the PNG's quadrant scatter): each concurrency rung shows its TTFT p50 and
    TPOT p50 with a per-axis ✓/✗, then a ZONE verdict — ✅ both-pass / ⚠ one-fails: <axis> / ❌ both-fail.
    The two SLA limits sit in the header; the max both-pass concurrency is highlighted. Deterministic.
    """
    res = record.get("result") or {}
    sla = res.get("sla") or {}
    stat = sla.get("stat", "p50")
    ttft_limit = _f(sla.get("ttft_ms"))
    tpot_limit = _f(sla.get("tpot_ms"))
    rungs = sorted(
        (record.get("detail") or {}).get("rungs") or [],
        key=lambda r: _f(r.get("concurrency")) or 0,
    )
    metric = res.get("metric", "max_concurrency_at_sla")
    band = _scalar_band(runs, metric)
    cp = _crossing_point(record)
    max_pass = _max_passing_conc(rungs, ttft_limit, tpot_limit)
    # Headline makes the safe-floor vs interpolated-crossing distinction explicit — a reader sees BOTH the
    # verified last-pass rung AND the interpolated crossing (and which one the metric value actually is).
    interp = _value_is_crossing(res, cp, max_pass)  # is the metric value the crossing, or the last-pass floor?
    head = f"metric {metric} = {res.get('value')} {res.get('unit') or 'concurrency'}"
    parts = []
    if interp:
        head += f" (interpolated crossing" + (f", binding {cp['binding']}" if cp.get("binding") else "") + ")"
        if max_pass is not None:
            parts.append(f"verified last-pass rung c{_fmt(max_pass,0)}")
    else:
        head += " (verified last-pass rung)"
        if cp.get("status") == "ok" and cp.get("conc") is not None:
            parts.append(
                f"interp crossing ≈ c{_fmt(cp['conc'],0)}"
                + (f" (binding {cp['binding']})" if cp.get("binding") else "")
            )
        elif cp.get("status") and cp.get("status") != "ok":
            parts.append(f"interp crossing: {cp['status']}")
    headline = "  ·  ".join([head] + parts) + (f"  ·  {band}" if band else "")
    title = f"llm-perf · max-concurrency-sla — 2D SLA zones: TTFT {stat} (x) × TPOT {stat} (y)"
    out = _ascii_header(title, headline)
    out.append(
        f"  SLA limits: TTFT {stat} ≤ {_fmt(ttft_limit,0)} ms  ·  TPOT {stat} ≤ {_fmt(tpot_limit,0)} ms"
        "   (zone = quadrant of the TTFT×TPOT plane)"
    )
    out.extend(_sla_crossing_lines(cp))
    out.append("")
    # First pass: classify every rung (max both-pass concurrency = max_pass, the highlighted floor).
    rows = []
    for r in rungs:
        conc = _f(r.get("concurrency"))
        tf, tf_lo, tf_hi, tf_n = _rung_stat(r, "ttft_p50_ms")
        tp, tp_lo, tp_hi, tp_n = _rung_stat(r, "tpot_p50_ms")
        code, failed = _sla_zone(tf, tp, ttft_limit, tpot_limit)
        rows.append((conc, tf, tf_lo, tf_hi, tf_n, tp, tp_lo, tp_hi, tp_n, code, failed))
    for conc, tf, tf_lo, tf_hi, tf_n, tp, tp_lo, tp_hi, tp_n, code, failed in rows:
        ttft_ok = "✓" if (tf is not None and ttft_limit is not None and tf <= ttft_limit) else "✗"
        tpot_ok = "✓" if (tp is not None and tpot_limit is not None and tp <= tpot_limit) else "✗"
        verdict = f"⚠ one-fails: {failed[0]}" if code == "one" else _ZONE_VERDICT.get(code, "· n/a")
        band_txt = ""
        if tp_n >= 2 and tp_lo is not None and tp_hi is not None:
            band_txt += f"  TPOT⟨{_fmt(tp_lo)}–{_fmt(tp_hi)} n={tp_n}⟩"
        if tf_n >= 2 and tf_lo is not None and tf_hi is not None:
            band_txt += f"  TTFT⟨{_fmt(tf_lo)}–{_fmt(tf_hi)} n={tf_n}⟩"
        mark = "  ⟵ max passing" if (conc is not None and conc == max_pass) else ""
        cs = f"c={int(conc):>5}" if conc is not None else "c=    ?"
        out.append(
            f"  {cs} | TTFT {_fmt(tf):>8} ms {ttft_ok}  ·  TPOT {_fmt(tp):>7} ms {tpot_ok}"
            f"  | {verdict}{band_txt}{mark}"
        )
    out.append("```")
    return "\n".join(out)


def _ascii_pareto(record, runs):
    res = record.get("result") or {}
    rungs = sorted(
        (record.get("detail") or {}).get("rungs") or [],
        key=lambda r: _f(r.get("tps_per_user")) or 0,
    )
    field = "tps_per_gpu"
    vals = [(_f(r.get("tps_per_user")), *_rung_stat(r, field), _f(r.get("concurrency"))) for r in rungs]
    ys = [v for _, v, lo, h, n, c in vals if v is not None]
    vmax = (max(ys) * 1.08) if ys else 1.0
    metric = res.get("metric", "pareto_geomean")
    band = _scalar_band(runs, metric)
    headline = f"metric {metric} = {res.get('value')} {res.get('unit') or 'geomean(tps/gpu·tps/user)'}" + (
        f"  ·  {band}" if band else ""
    )
    title = "llm-perf · pareto — throughput frontier: tps/gpu (bar) vs tps/user (label), by concurrency"
    out = _ascii_header(title, headline)
    for x, y, lo, hi, n, conc in vals:
        cells = _bar_cells(y, vmax, _WIDTH)
        if n >= 2 and lo is not None and hi is not None:
            for edge, ch in ((lo, "├"), (hi, "┤")):
                cc = _col(edge, vmax, _WIDTH)
                if cc is not None:
                    cells[cc] = ch
        band_txt = f" ⟨{_fmt(lo)}–{_fmt(hi)} n={n}⟩" if n >= 2 else ""
        cs = f"c={int(conc)}" if conc is not None else "c=?"
        out.append(f"  {cs:>7} |{''.join(cells)}| {_fmt(y):>7} tps/gpu @ {_fmt(x):>6} tps/user{band_txt}")
    out.append("```")
    return "\n".join(out)


# ONE renderer table, keyed on the handler's chart_key().
_ASCII_BY_KEY = {
    "sla": _ascii_sla,
    "pareto": _ascii_pareto,
}


def _dispatch(scenario, goal):
    """(scenario, goal) → ASCII renderer via the goal handler's chart_key(). An unregistered
    lane → None (rendered as a graceful note)."""
    sys.path.insert(0, str(ROOT / "analysis"))
    import goal_handlers as _gh  # noqa: E402 — no cycle: goal_handlers never imports charts

    try:
        key = _gh.resolve(scenario, goal).chart_key()
    except KeyError:
        return None
    return _ASCII_BY_KEY.get(key)


def render_ascii(record, runs=None):
    """PURE: (record dict, runs list) → the fenced ASCII chart for this lane, or a short note if the
    (scenario, goal) has no renderer / the cell has no rungs. Deterministic — same data, same bytes.
    """
    runs = runs or []
    ident = record.get("identity") or {}
    scenario, goal = ident.get("scenario"), ident.get("goal")
    rungs = (record.get("detail") or {}).get("rungs") or []
    fn = _dispatch(scenario, goal)
    if fn is None:
        return f"_(no chart renderer for scenario={scenario!r} goal={goal!r})_"
    if not rungs:
        return f"_(no rungs in record.json — nothing to chart for {scenario} · {goal})_"
    return fn(record, runs)


# ─────────────────────────────────────────────────────────────────────────────
# PNG renderers (BEST-EFFORT) — deterministic matplotlib Agg, byte-stable across re-runs
# ─────────────────────────────────────────────────────────────────────────────

# Strip every writer-injected, non-deterministic PNG chunk (Software version string / creation time) so two
# renders of identical data are byte-for-byte equal. Fixed DPI + sorted inputs complete the determinism.
_PNG_META = {"Software": None, "Creation Time": None}
_DPI = 100


def _mpl():
    """Import matplotlib with the Agg backend, or return None (PNG is optional — never a hard dep)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _sla_png_caveat(cp):
    """A one-line PNG subtitle: bracket + ratio + binding, plus the honest ⚠ caveat when the bracket is coarse
    (>1.5×, so linear interpolation overestimates). When there's no clean bracket, surface the committed note
    instead (explains why no marker). Derived only from committed structured fields — deterministic.
    """
    status = cp.get("status")
    if status != "ok" or cp.get("conc") is None:
        return (cp.get("note") or (f"SLA crossing: {status}" if status else "")) or ""
    b = cp.get("bracket") or []
    lo = f"{_f(b[0]):g}" if len(b) == 2 and _f(b[0]) is not None else "?"
    hi = f"{_f(b[1]):g}" if len(b) == 2 and _f(b[1]) is not None else "?"
    ratio = _f(cp.get("ratio"))
    s = (
        f"bracket [{lo}–{hi}]"
        + (f" ({ratio:g}×)" if ratio else "")
        + (f", binding {cp['binding']}" if cp.get("binding") else "")
    )
    if ratio and ratio > 1.5:
        s += " — ⚠ coarse bracket; interpolation approximate"
    return s


def _run_descriptor(record):
    """A compact two-tier caption of WHAT was run — line 1 = model·hardware·parallelism·engine;
    line 2 = context·workload·metric. Pure function of committed identity/config → deterministic.
    """
    import re

    ident = record.get("identity") or {}
    serving = (record.get("config") or {}).get("serving") or {}
    res = record.get("result") or {}
    model = ident.get("model") or "?"
    gpu = ident.get("gpu_type") or "?"
    n = ident.get("gpu_count")
    tp = serving.get("tp")
    hw = f"{gpu}×{int(n)}" if n else gpu
    tptxt = f" · TP{int(tp)}" if tp else ""
    eng = " ".join(x for x in (ident.get("engine"), ident.get("serving_mode")) if x)
    line1 = f"{model} · {hw}{tptxt}" + (f" · {eng}" if eng else "")
    m = re.search(r"\d+\s*[kKmM]", ident.get("distribution") or "") or re.search(
        r"\d+\s*[kKmM]", ident.get("cell") or ""
    )
    ctx = m.group(0).replace(" ", "").lower() if m else ""
    metric, val = res.get("metric", "pareto_geomean"), res.get("value")
    line2 = " · ".join([b for b in (ctx, ident.get("mode") or "") if b] + [f"{metric} = {val}"])
    return line1, line2


def _better_arrows(ax, x_text, y_text):
    """Two subtle 'higher is better' arrows radiating from the origin corner (empty on an anti-diagonal
    frontier), one per axis. Axes-fraction coords → never clipped, data-independent placement.
    """
    ap = dict(arrowstyle="-|>", color="#8a8a8a", lw=1.4, shrinkA=0, shrinkB=0)
    # An "L" open at the origin corner: vertical arrow (efficiency ↑) + horizontal arrow (user speed →).
    # Labels sit at each shaft's MIDPOINT (well away from the shared corner) so they never collide.
    ax.annotate(
        "",
        xy=(0.045, 0.30),
        xytext=(0.045, 0.10),
        xycoords="axes fraction",
        arrowprops=ap,
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(0.30, 0.045),
        xytext=(0.10, 0.045),
        xycoords="axes fraction",
        arrowprops=ap,
        zorder=5,
    )
    ax.text(
        0.075,
        0.205,
        y_text,
        transform=ax.transAxes,
        fontsize=7,
        color="#6a6a6a",
        ha="left",
        va="center",
        rotation=90,
        zorder=5,
    )
    ax.text(
        0.205,
        0.072,
        x_text,
        transform=ax.transAxes,
        fontsize=7,
        color="#6a6a6a",
        ha="center",
        va="bottom",
        zorder=5,
    )


def render_png(record, out_path: Path, runs=None) -> bool:
    """BEST-EFFORT deterministic PNG for this lane → out_path. Returns True on write, False if matplotlib is
    unavailable or the lane/rungs are unchartable (caller keeps the ASCII regardless).
    """
    plt = _mpl()
    if plt is None:
        return False
    ident = record.get("identity") or {}
    scenario, goal = ident.get("scenario"), ident.get("goal")
    rungs = sorted(
        (record.get("detail") or {}).get("rungs") or [],
        key=lambda r: _f(r.get("concurrency")) or 0,
    )
    res = record.get("result") or {}
    if not rungs or _dispatch(scenario, goal) is None:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    try:
        if scenario == "llm-perf" and goal == "max-concurrency-sla":
            # 2D SLA view: x = TTFT p50 (ms), y = TPOT p50 (ms). The two SLA limits cut the plane into
            # quadrants — GREEN (both met, bottom-left), RED (one fails, the two adjacent quads), DARK-RED
            # (both fail, top-right). Each concurrency rung is a POINT, colored by its zone + labeled.
            sla = res.get("sla") or {}
            ttft_limit = _f(sla.get("ttft_ms"))
            tpot_limit = _f(sla.get("tpot_ms"))
            pts = []  # (concurrency, ttft, tpot, zone) — rungs already sorted by concurrency
            for r in rungs:
                tf, _l1, _h1, _n1 = _rung_stat(r, "ttft_p50_ms")
                tp, _l2, _h2, _n2 = _rung_stat(r, "tpot_p50_ms")
                if tf is None or tp is None:
                    continue
                code, _fl = _sla_zone(tf, tp, ttft_limit, tpot_limit)
                pts.append((_f(r.get("concurrency")), tf, tp, code))
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            xhi = max(xs + ([ttft_limit] if ttft_limit is not None else []) + [1.0]) * 1.12
            yhi = max(ys + ([tpot_limit] if tpot_limit is not None else []) + [1.0]) * 1.12
            ax.set_xlim(0, xhi)
            ax.set_ylim(0, yhi)
            # Shade the three zones with Rectangle patches, then draw the two limit lines.
            if ttft_limit is not None and tpot_limit is not None:
                ax.add_patch(
                    plt.Rectangle(
                        (0, 0),
                        ttft_limit,
                        tpot_limit,
                        color="#2e7d32",
                        alpha=0.12,
                        lw=0,
                        zorder=0,
                    )
                )  # green
                ax.add_patch(
                    plt.Rectangle(
                        (ttft_limit, 0),
                        xhi - ttft_limit,
                        tpot_limit,
                        color="#c62828",
                        alpha=0.12,
                        lw=0,
                        zorder=0,
                    )
                )  # red
                ax.add_patch(
                    plt.Rectangle(
                        (0, tpot_limit),
                        ttft_limit,
                        yhi - tpot_limit,
                        color="#c62828",
                        alpha=0.12,
                        lw=0,
                        zorder=0,
                    )
                )  # red
                ax.add_patch(
                    plt.Rectangle(
                        (ttft_limit, tpot_limit),
                        xhi - ttft_limit,
                        yhi - tpot_limit,
                        color="#7f0000",
                        alpha=0.20,
                        lw=0,
                        zorder=0,
                    )
                )  # dark-red
                ax.axvline(ttft_limit, color="#1565c0", ls="--", lw=1.2, zorder=1)
                ax.axhline(tpot_limit, color="#1565c0", ls="--", lw=1.2, zorder=1)
            # Thin connector in concurrency order, then the colored, annotated points.
            ax.plot(xs, ys, color="#888888", lw=0.8, zorder=2)
            zcolor = {
                "pass": "#2e7d32",
                "one": "#c62828",
                "both": "#7f0000",
                "unknown": "#9e9e9e",
            }
            for conc, tf, tp, code in pts:
                ax.scatter(
                    [tf],
                    [tp],
                    c=[zcolor.get(code, "#9e9e9e")],
                    s=55,
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.6,
                )
                lbl = f"c={int(conc)}" if conc is not None else "c=?"
                ax.annotate(
                    lbl,
                    (tf, tp),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    zorder=4,
                )
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    ls="",
                    mfc="#2e7d32",
                    mec="white",
                    label="both SLAs met",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    ls="",
                    mfc="#c62828",
                    mec="white",
                    label="one SLA failed",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    ls="",
                    mfc="#7f0000",
                    mec="white",
                    label="both SLAs failed",
                ),
                plt.Line2D([0], [0], color="#1565c0", ls="--", label="SLA limits"),
            ]
            # Interpolated crossing: a hollow diamond where the last-pass→first-fail segment meets the BINDING
            # limit line. The headline value is the conservative last-pass FLOOR; the true crossing is between
            # rungs and a coarse bracket overestimates it — so show both + the honest caveat.
            cp = _crossing_point(record)
            max_pass = _max_passing_conc(rungs, ttft_limit, tpot_limit)
            interp = _value_is_crossing(res, cp, max_pass)
            if cp.get("status") == "ok" and cp.get("x") is not None and cp.get("y") is not None:
                ax.scatter(
                    [cp["x"]],
                    [cp["y"]],
                    marker="D",
                    facecolors="none",
                    edgecolors="#6a1b9a",
                    s=120,
                    lw=1.8,
                    zorder=6,
                )
                ax.annotate(
                    f"crossing ≈ c{_fmt(cp['conc'],0)} (interp)",
                    (cp["x"], cp["y"]),
                    textcoords="offset points",
                    xytext=(7, -11),
                    fontsize=7,
                    color="#6a1b9a",
                    zorder=6,
                )
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="D",
                        ls="",
                        mfc="none",
                        mec="#6a1b9a",
                        label="interp SLA crossing",
                    )
                )
            ax.legend(handles=handles, loc="upper right", fontsize=7)
            ax.set_xlabel("TTFT p50 (ms)")
            ax.set_ylabel("TPOT p50 (ms)")
            # Title carries BOTH the verified floor and the interpolated crossing; the subtitle the caveat.
            value = res.get("value")
            if interp:
                title = f"max_concurrency_at_sla = {value} (interp crossing)"
                if max_pass is not None:
                    title += f" · last-pass rung c{_fmt(max_pass,0)}"
            else:
                title = f"max_concurrency_at_sla = {value} (verified last-pass rung)"
                if cp.get("status") == "ok" and cp.get("conc") is not None:
                    title += f" · interp crossing ≈ c{_fmt(cp['conc'],0)}"
                elif cp.get("status") and cp.get("status") != "ok":
                    title += f" · crossing {cp['status']}"
            sub = _sla_png_caveat(cp)
            ax.set_title(title + (("\n" + sub) if sub else ""), fontsize=9)
        elif scenario == "llm-perf" and goal == "pareto":
            # Throughput frontier: x = tps/user (single-user speed), y = tps/gpu (serving efficiency).
            # Both axes higher=better; the curve is the efficiency↔speed tradeoff across concurrency rungs.
            pts = []
            for r in rungs:
                x, y, c = (
                    _f(r.get("tps_per_user")),
                    _f(r.get("tps_per_gpu")),
                    _f(r.get("concurrency")),
                )
                if x is not None and y is not None:
                    pts.append((x, y, c))
            pts.sort(key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.set_xlim(0, (max(xs) if xs else 1.0) * 1.15)  # both axes anchored at 0
            ax.set_ylim(0, (max(ys) if ys else 1.0) * 1.15)
            ax.plot(xs, ys, "-", color="#2e7d32", lw=1.6, zorder=2)
            ax.scatter(xs, ys, c="#2e7d32", s=60, zorder=3, edgecolors="white", linewidths=0.8)
            for x, y, c in pts:  # concurrency label per point
                ax.annotate(
                    f"c{int(c)}" if c is not None else "c?",
                    (x, y),
                    textcoords="offset points",
                    xytext=(7, 5),
                    fontsize=8,
                    fontweight="bold",
                    color="#1b5e20",
                    zorder=4,
                )
            ax.set_xlabel("tokens/s per user")
            ax.set_ylabel("tokens/s per GPU")
            _better_arrows(ax, "faster user experience", "more efficient")
            line1, line2 = _run_descriptor(record)
            ax.set_title(line1, fontsize=10, fontweight="bold", pad=20)
            ax.text(
                0.5,
                1.03,
                line2,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
        else:
            plt.close(fig)
            return False
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_path, dpi=_DPI, metadata=_PNG_META)
        return True
    finally:
        plt.close(fig)


def render_cell(cell_dir: Path, write_png=True):
    """Read a cell's committed record.json + runs.jsonl → (ascii_str, png_relpath_or_None). The one entry
    point publish.py wires in: pure ASCII always; PNG best-effort under <cell>/charts/<goal>.png.
    """
    cell_dir = Path(cell_dir)
    rec_p = cell_dir / "record.json"
    if not rec_p.is_file():
        return "_(no record.json — publish the cell first)_", None
    record = json.loads(rec_p.read_text())
    runs = []
    jl = cell_dir / "runs.jsonl"
    if jl.is_file():
        for line in jl.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except ValueError:
                    pass
    ascii_chart = render_ascii(record, runs)
    png_rel = None
    if write_png:
        ident = record.get("identity") or {}
        goal = ident.get("goal") or ident.get("scenario") or "chart"
        safe = str(goal).replace("/", "-").replace(" ", "_")
        png = cell_dir / "charts" / f"{safe}.png"
        if render_png(record, png, runs):
            png_rel = f"charts/{safe}.png"
    return ascii_chart, png_rel


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    write_png = "--no-png" not in argv
    if not args:
        sys.exit("usage: charts.py <cell-dir> [--no-png]")
    ascii_chart, png_rel = render_cell(Path(args[0]), write_png=write_png)
    print(ascii_chart)
    if write_png:
        print(
            f"\n[charts] PNG: {png_rel}"
            if png_rel
            else "\n[charts] PNG skipped (matplotlib unavailable or lane not chartable) — ASCII is the source of truth"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

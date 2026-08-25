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

"""goal_handlers.py — the GOAL-AWARE spine: one typed handler per (scenario, goal).

The value-producing pipeline (aggregate -> export_record -> publish -> charts) used to dispatch on SCENARIO
and then HARDCODE that scenario's DEFAULT goal metric. This module makes the goal a first-class dispatch key.

A handler owns everything the pipeline needs to know about ONE (scenario, goal) lane:
  - `required_fields`             : fields the recipe/bench/aggregated rows must provide for this goal
                                    (e.g. SLA only for A).
  - `compute_metric(rows,...)`    : (metric_name, value, value_source) -- the canonical metric + how it was
                                    derived. This is the ONE place a goal's number is computed; export_record
                                    and aggregate both call it, so a goal can never silently report the wrong
                                    scenario-default metric again.
  - `result_fields(rows,...)`     : the FULL result.* block a goal emits into record.json (value + value_source
                                    + goal-specific fields: sla/crossing ONLY for A; per-point geomean for B).
                                    Keeps SLA semantics off the pareto lane.
  - `headline(hres)`              : the RESULTS.md one-liner (publish presentation).
  - `table(rows, recipe)`         : the per-rung markdown table (publish presentation).
  - `chart_key()`                 : which charts.py renderer this lane uses.

A registry keyed on (scenario, goal) resolves the handler; `assert_every_goal_has_a_handler()` is the
STARTUP ASSERTION -- every goal declared in schema/envelope.yaml (and check_invariants.KNOWN_METRICS) must
have a registered handler, so a newly-declared goal with no handler fails LOUDLY instead of silently leaking
into the scenario default.

DESIGN NOTE ON BYTE-IDENTICAL BEHAVIOR: combos A (llm-perf - max-concurrency-sla) and B (llm-perf - pareto)
reproduce the pre-refactor headline/table/record/chart EXACTLY.
"""

from __future__ import annotations

import csv as _csv
import importlib.util
import subprocess
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_csv(p) -> list:
    """A run-dir/curated CSV → list of row dicts (comment lines dropped). Missing file → []."""
    p = Path(p)
    if not p.is_file():
        return []
    return list(_csv.DictReader([l for l in p.read_text().splitlines() if not l.startswith("#")]))


def _csv_cell(v) -> str:
    """One curated-CSV cell that round-trips byte-for-byte: None→"" (re-reads as None via _f), float→repr
    (full precision), everything else→str. So rungs.csv → native → detail.rungs reproduces the record exactly."""
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _load(mod_name: str, rel_path: str):
    """Import a module by file path (the analysis dirs have dashes, so they're not importable by name)."""
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ec():
    return _load("exemplar_check", "analysis/llm-perf/exemplar_check.py")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── shared table renderers (moved verbatim from publish.py so a handler owns its own table) ────────────
def rung_table(rows, sla, is_sla=True):
    """A per-rung table. For max-concurrency-sla it carries the SLA pass/fail column (numbers-at-a-glance
    against the gate); for pareto — which has no SLA attached — that column is omitted."""
    tl, pl, stat = sla.get("ttft_ms"), sla.get("tpot_ms"), sla.get("stop_stat", "p50")
    hdr = "| concurrency | TTFT " + stat + " (ms) | ITL/TPOT " + stat + " (ms) | TPS/GPU | error% |"
    sep = "|---|---|---|---|---|"
    if is_sla:
        hdr += " SLA |"
        sep += "---|"
    out = [hdr, sep]
    for r in rows:
        c = r.get("concurrency", "?")
        ttft = r.get(f"ttft_{stat}_ms") or ""
        itl = r.get(f"itl_{stat}_ms") or ""
        tpg = r.get("throughput_per_gpu_tok_per_s") or ""
        err = r.get("error_rate_pct") or "0"
        fmt = lambda v: f"{float(v):.0f}" if v not in ("", None) else "—"
        fmtf = lambda v: f"{float(v):.1f}" if v not in ("", None) else "—"
        row = f"| {c} | {fmt(ttft)} | {fmtf(itl)} | {fmt(tpg)} | {err} |"
        if is_sla:
            try:
                ok = float(ttft) <= tl and float(itl) <= pl
            except (ValueError, TypeError):
                ok = None
            verdict = "✅ pass" if ok else ("❌ fail" if ok is False else "—")
            row += f" {verdict} |"
        out.append(row)
    return "\n".join(out)


def pareto_rung_table(rows, stat="p50"):
    """The pareto lane's per-rung table. Leads with the TWO metrics the pareto frontier is built from —
    Output-TPS/GPU (y) and Output-TPS/user (x) — then the per-rung `geomean` g=√(TPS/GPU · TPS/user) (the
    quantity pareto_geomean averages over rungs), then TTFT (informational) and error%. Deliberately does NOT
    carry a TPOT column or an SLA pass/fail column (pareto has no SLA gate); TPS/user is the user-facing
    decode-rate (= 1000/ITL), which is what the frontier trades against TPS/GPU."""
    import math

    hdr = "| concurrency | Out TPS/GPU | Out TPS/user | geomean | TTFT " + stat + " (ms) | error% |"
    sep = "|---|---|---|---|---|---|"
    out = [hdr, sep]
    for r in rows:
        c = r.get("concurrency", "?")
        tpg = r.get("throughput_per_gpu_tok_per_s")
        tpu = r.get("tokens_per_s_per_user_from_itl")
        if tpu in ("", None):
            tpu = r.get(f"output_token_throughput_per_user_{stat}")
        ttft = r.get(f"ttft_{stat}_ms")
        err = r.get("error_rate_pct") or "0"
        f0 = lambda v: f"{float(v):.0f}" if v not in ("", None) else "—"
        f1 = lambda v: f"{float(v):.1f}" if v not in ("", None) else "—"
        gx, gy = _f(tpg), _f(tpu)
        geo = f"{math.sqrt(gx * gy):.1f}" if (gx and gy and gx > 0 and gy > 0) else "—"
        out.append(f"| {c} | {f1(tpg)} | {f1(tpu)} | {geo} | {f0(ttft)} | {err} |")
    return "\n".join(out)


# ── the handler abstraction ────────────────────────────────────────────────────────────────────────────
class GoalHandler:
    """One (scenario, goal) lane. Subclasses fill in the metric + presentation."""

    scenario: str = ""
    goal = None
    metric: str = ""  # the metric stored in record.json (the canonical one)
    alt_metrics: tuple = ()  # other exemplar metrics this SAME goal-lane can report (e.g. tps_per_gpu_at_sla
    # is an alternate exemplar of the max-concurrency-sla goal — same sweep, same
    # SLA crossing, a different reported number). Counted as covered by the handler.
    unit: str = ""
    higher_better: bool = True
    _chart_key: str = ""
    required_fields: tuple = ()

    # ── the storage/aggregation contract a lane owns (Phase-4 fork removal) ──────────────────────────────
    # `summary_csv` = the native per-rung CSV a run/curated dir carries (the aggregator's output; the SAME
    # column schema compute_metric/table already consume). `rung_keys` = the normalized, goal-agnostic
    # detail.rungs column order persisted to curated/rungs.csv. Both are declared by the SCENARIO base class
    # so aggregation, record building, publishing and charts share one seam instead of `if scenario ==` forks.
    summary_csv: str = ""
    rung_keys: tuple = ()

    def chart_key(self) -> str:
        return self._chart_key

    # ── native ⇄ normalized rung mapping (the single detail.rungs schema, owned by the scenario) ─────────
    def rung_from_native(self, native_row: dict) -> dict:
        """A native summary-CSV row → the normalized detail.rungs dict this lane persists (incl. detail_extra)."""
        raise NotImplementedError

    def native_from_rung(self, rung: dict) -> dict:
        """The inverse: a normalized rungs.csv row → the native-column dict compute_metric/table consume, so a
        record + tables REGENERATE from the committed curated rungs.csv when the raw summary CSV is gone."""
        raise NotImplementedError

    def aggregate_summary(self, run_dir, goal=None):
        """Run the scenario aggregator over a raw run-dir → its `summary_csv`. Override per scenario."""
        raise NotImplementedError

    def read_native_rows(self, source_dir, goal=None) -> list:
        """Native rows for compute/table from a run-dir OR a committed curated dir. Prefers the native
        `summary_csv`; falls back to the normalized `rungs.csv` (curated, raw GC'd); else aggregates a raw
        run-dir. This is the KEYSTONE seam: pointed at curated/ it reads COMMITTED data, never a scratch dir."""
        source_dir = Path(source_dir)
        rows = _read_csv(source_dir / self.summary_csv)
        if rows:
            return rows
        rc = source_dir / "rungs.csv"
        if rc.is_file():
            return [self.native_from_rung(r) for r in _read_csv(rc)]
        self.aggregate_summary(source_dir, goal)
        return _read_csv(source_dir / self.summary_csv)

    def rungs_csv_row(self, rung: dict) -> list:
        """A normalized detail.rungs dict → the ordered cell list written to curated/rungs.csv (round-trips
        through native_from_rung/rung_from_native to byte-identical numbers)."""
        return [_csv_cell(rung.get(k)) for k in self.rung_keys]

    def set_baseline(self, cell, csv_path=None, value=None) -> None:
        """Commit this lane's exemplar reference into recipe.yaml (the ONLY --set-baseline path). Override."""
        raise NotImplementedError

    # ── presentation sourced from a COMMITTED record.json (Phase-3 publish inversion) ───────────────────
    def headline_from_record(self, record: dict) -> str:
        """The RESULTS.md headline computed ONLY from a committed record.json (result + detail.rungs)."""
        res = dict(record.get("result") or {})
        res.setdefault("unit", self.unit)
        return self.headline(res)

    def table_from_record(self, record: dict, recipe=None) -> str:
        """The per-rung table re-derived from the committed detail.rungs (mapped back to native columns)."""
        rows = [self.native_from_rung(r) for r in (record.get("detail") or {}).get("rungs") or []]
        return self.table(rows, recipe)

    # publish_label = the metric NAME shown in the headline. Usually == self.metric.
    def publish_label(self, hres: dict) -> str:
        return self.metric

    # --- computation (export_record + aggregate) ---
    def compute_metric(self, rows, recipe=None, prices=None):
        """(metric_name, value, value_source). rows = the aggregated per-rung dicts."""
        raise NotImplementedError

    def result_fields(self, rows, recipe=None, prices=None) -> dict:
        """The result.* block this goal emits into record.json (value + value_source + goal-specific fields).
        Default: metric/value/value_source only. Overridden by A (sla/crossing)."""
        metric, value, source = self.compute_metric(rows, recipe, prices)
        return {"value": value, "value_source": source}

    def detail_extra(self, row) -> dict:
        """Extra per-rung fields this goal appends to record.detail.rungs (beyond the shared base). Default:
        none (so scenario-default lanes stay byte-identical)."""
        return {}

    # --- presentation (publish) ---
    def headline(self, hres: dict) -> str:
        raise NotImplementedError

    def table(self, rows, recipe=None) -> str:
        raise NotImplementedError


# ════════════════════════════════ scenario base classes (own the storage/aggregation seam) ═════════════
class LlmPerfHandler(GoalHandler):
    """llm-perf lanes: aiperf sweep → metrics_summary.csv (native aiperf columns). The normalized rungs.csv
    schema + the native⇄normalized mapping live here so BOTH goal lanes (A sla / B pareto) share one loop."""

    scenario = "llm-perf"
    summary_csv = "metrics_summary.csv"
    # kv_cache_usage_perc = the PEAK vLLM KV-cache utilization (%) during the rung (server memory pressure at
    # that concurrency). Sourced from the aggregator's peak_vllm_kv_cache_usage_pct; empty on runs whose raw
    # server timeseries is gone (reconstructed cells) — a nullable diagnostic, not a metric-path input.
    # rung_wall_seconds = how long THIS rung ran (aiperf benchmark_duration). Nullable: reconstructed runs
    # archived before this column existed carry None. Distinct from the run-level wall_seconds (whole sweep).
    rung_keys = (
        "concurrency",
        "ttft_p50_ms",
        "tpot_p50_ms",
        "tps_per_gpu",
        "tps_per_user",
        "error_rate_pct",
        "kv_cache_usage_perc",
        "rung_wall_seconds",
    )

    def rung_from_native(self, x: dict) -> dict:
        conc = _f(x.get("concurrency"))
        rung = {
            "concurrency": int(conc) if conc is not None else None,
            "ttft_p50_ms": _f(x.get("ttft_p50_ms")),
            "tpot_p50_ms": _f(x.get("itl_p50_ms")),
            "tps_per_gpu": _f(x.get("throughput_per_gpu_tok_per_s")),
            "tps_per_user": _f(x.get("tokens_per_s_per_user_from_itl")),
            "error_rate_pct": _f(x.get("error_rate_pct")),
            "kv_cache_usage_perc": _f(x.get("peak_vllm_kv_cache_usage_pct")),
        }
        # HOW LONG THIS RUNG TOOK. aggregate_metrics already sources it per-rung from aiperf's
        # benchmark_duration, but it stopped here: the normalized rung dropped it, so the export's
        # `wall_seconds` was the whole-run total repeated on every row and no consumer could tell how
        # long an individual rung ran.
        # ADDITIVE, NOT BREAKING: the key is emitted only when a duration exists. Records archived before
        # this column existed have none, and must keep regenerating from their committed curated/ BYTE-FOR-
        # BYTE (CI enforces this) — emitting an explicit null would silently rewrite every historical
        # record.json to gain a field its run never measured.
        rws = _f(x.get("wall_seconds"))
        if rws is not None:
            rung["rung_wall_seconds"] = rws
        return rung

    def native_from_rung(self, r: dict) -> dict:
        return {
            "concurrency": r.get("concurrency"),
            "ttft_p50_ms": r.get("ttft_p50_ms"),
            "itl_p50_ms": r.get("tpot_p50_ms"),
            "throughput_per_gpu_tok_per_s": r.get("tps_per_gpu"),
            "tokens_per_s_per_user_from_itl": r.get("tps_per_user"),
            "error_rate_pct": r.get("error_rate_pct"),
            "peak_vllm_kv_cache_usage_pct": r.get("kv_cache_usage_perc"),
            # inverse of rung_from_native's rung_wall_seconds — keeps the rungs.csv → native → rungs
            # round-trip exact (selftest_archive_run asserts byte-identical re-archive). A blank CSV
            # cell round-trips to key-absent, matching the additive contract above.
            "wall_seconds": r.get("rung_wall_seconds") or None,
        }

    def aggregate_summary(self, run_dir, goal=None):
        out = Path(run_dir) / self.summary_csv
        subprocess.run(
            [_sys.executable, str(ROOT / "analysis/llm-perf/aggregate_metrics.py"), str(run_dir), "--out", str(out)],
            capture_output=True,
            text=True,
        )
        return out

    def set_baseline(self, cell, csv_path=None, value=None) -> None:
        # the single exemplar_check --set path (publish no longer calls exemplar_check directly)
        if csv_path is not None:
            subprocess.run(
                [_sys.executable, str(ROOT / "analysis/llm-perf/exemplar_check.py"), str(cell), str(csv_path), "--set"],
                capture_output=True,
                text=True,
            )


# ════════════════════════════════ llm-perf ══════════════════════════════════════════════════════════════
class LlmSlaHandler(LlmPerfHandler):
    """A — llm-perf · max-concurrency-sla → max_concurrency_at_sla (SLA crossing / highest passing rung)."""

    scenario = "llm-perf"
    goal = "max-concurrency-sla"
    metric = "max_concurrency_at_sla"
    alt_metrics = ("tps_per_gpu_at_sla",)  # decode TPS/GPU AT the SLA crossing — same lane, alternate report
    unit = "concurrency"
    higher_better = True
    _chart_key = "sla"
    required_fields = ("bench.sla.ttft_ms", "bench.sla.tpot_ms", "bench.sla.stop_stat")

    def _effective_metric(self, recipe):
        m = (((recipe or {}).get("envelope") or {}).get("exemplar") or {}).get("metric")
        return m if m in (self.metric, *self.alt_metrics) else self.metric

    def compute_metric(self, rows, recipe=None, prices=None):
        ec = _ec()
        bench = (recipe or {}).get("bench") or {}
        ex = ((recipe or {}).get("envelope") or {}).get("exemplar") or {}
        sla = bench.get("sla") or {}
        stat = sla.get("stop_stat", "p50")
        tl, pl = _f(sla.get("ttft_ms")), _f(sla.get("tpot_ms"))
        cx = ec.sla_crossing(ec.all_rungs(rows, stat), tl, pl, 5.0)
        metric = self._effective_metric(recipe)
        if metric == "tps_per_gpu_at_sla":
            return metric, cx.get("tps_per_gpu_at_crossing"), "interpolated_sla_crossing"
        if ec.measured_rung_policy(ex):
            passing = ec.passing_rungs(rows, tl, pl, stat, 5.0)
            return metric, ec.measured_ceiling(passing), "highest_sla_passing_rung"
        return metric, cx.get("value"), "interpolated_sla_crossing"

    def result_fields(self, rows, recipe=None, prices=None) -> dict:
        ec = _ec()
        bench = (recipe or {}).get("bench") or {}
        sla = bench.get("sla") or {}
        stat = sla.get("stop_stat", "p50")
        tl, pl = _f(sla.get("ttft_ms")), _f(sla.get("tpot_ms"))
        out = {"sla": {"ttft_ms": sla.get("ttft_ms"), "tpot_ms": sla.get("tpot_ms"), "stat": stat}}
        _metric, value, source = self.compute_metric(rows, recipe, prices)
        out["value"], out["value_source"] = value, source
        cx = ec.sla_crossing(ec.all_rungs(rows, stat), tl, pl, 5.0)
        out["crossing"] = {k: cx.get(k) for k in ("bracket", "ratio", "binding", "status", "note")}
        return out

    def headline(self, hres: dict) -> str:
        sla = hres.get("sla") or {}
        return (
            f"`{self.metric}` = **{hres.get('value')}** (unit: {hres.get('unit', '?')}) · passing rungs: "
            f"{hres.get('passing_concurrencies', [])}"
            f" · SLA TTFT≤{sla.get('ttft_ms')}ms / TPOT≤{sla.get('tpot_ms')}ms @ {sla.get('stop_stat', 'p50')}"
        )

    def table(self, rows, recipe=None) -> str:
        sla = ((recipe or {}).get("bench") or {}).get("sla") or {}
        return rung_table(rows, sla, is_sla=True)

    def headline_from_record(self, record: dict) -> str:
        """The SLA headline from a committed record: value/unit/sla come from result.*; passing_concurrencies
        are recomputed from detail.rungs so the one-liner is a pure function of committed data."""
        ec = _ec()
        res = record.get("result") or {}
        sla = res.get("sla") or {}
        stat = sla.get("stat", "p50")
        rows = [self.native_from_rung(r) for r in (record.get("detail") or {}).get("rungs") or []]
        tl, pl = _f(sla.get("ttft_ms")), _f(sla.get("tpot_ms"))
        passing = [p.get("concurrency") for p in ec.passing_rungs(rows, tl, pl, stat, 5.0)] if (tl and pl) else []
        return self.headline(
            {
                "value": res.get("value"),
                "unit": res.get("unit") or self.unit,
                "passing_concurrencies": passing,
                "sla": {"ttft_ms": sla.get("ttft_ms"), "tpot_ms": sla.get("tpot_ms"), "stop_stat": stat},
            }
        )


class LlmParetoHandler(LlmPerfHandler):
    """B — llm-perf · pareto → pareto_geomean (per-point geomean of the frontier; NO SLA semantics).

    value = geomean over rungs of g(p)=√(Output-TPS/GPU · Output-TPS/user); result.per_point carries the
    per-rung geomeans (equal per-point weight). The exemplar gate is point-by-point (compare.pareto_point_compare)."""

    scenario = "llm-perf"
    goal = "pareto"
    metric = "pareto_geomean"
    unit = "geomean(tps/gpu·tps/user)"
    higher_better = True
    _chart_key = "pareto"
    required_fields = ("bench.sweep_concurrency",)

    def compute_metric(self, rows, recipe=None, prices=None):
        ec = _ec()
        v, _pts = ec.pareto_geomean(rows)
        return self.metric, v, "geomean_over_rungs"

    def result_fields(self, rows, recipe=None, prices=None) -> dict:
        # pareto carries NO SLA semantics — emit the geomean value + its per-point breakdown + axes (the
        # per_point block is what the point-by-point exemplar comparison reads).
        ec = _ec()
        v, pts = ec.pareto_geomean(rows)
        return {
            "value": v,
            "value_source": "geomean_over_rungs",
            "per_point": pts,
            "axes": ["tps_per_gpu", "tps_per_user"],
        }

    def headline(self, hres: dict) -> str:
        n = len(hres.get("per_point") or [])
        return (
            f"`{self.metric}` = **{hres.get('value')}** (geomean of {n} points; "
            f"axes Output-TPS/GPU × Output-TPS/user) · swept: {hres.get('swept', [])}"
        )

    def table(self, rows, recipe=None) -> str:
        stat = (((recipe or {}).get("bench") or {}).get("sla") or {}).get("stop_stat", "p50")
        return pareto_rung_table(rows, stat)

    def headline_from_record(self, record: dict) -> str:
        res = record.get("result") or {}
        swept = [r.get("concurrency") for r in (record.get("detail") or {}).get("rungs") or []]
        return self.headline(
            {
                "value": res.get("value"),
                "unit": res.get("unit") or self.unit,
                "per_point": res.get("per_point") or [],
                "swept": swept,
            }
        )


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────────
_HANDLERS = [
    LlmSlaHandler(),
    LlmParetoHandler(),
]
REGISTRY = {(h.scenario, h.goal): h for h in _HANDLERS}

# The scenario DEFAULT goal (used when envelope.goal is absent). llm-perf always declares a goal.
SCENARIO_DEFAULT_GOAL: dict = {}

# metric → its default handler goal, for legacy/back-compat lookups by metric alone.
METRIC_TO_KEY = {m: (h.scenario, h.goal) for h in _HANDLERS for m in (h.metric, *h.alt_metrics)}


def resolve(scenario, goal):
    """The handler for (scenario, goal). Falls back to the scenario default lane when `goal` is None/absent.
    Raises KeyError with a loud message if nothing matches -- a declared goal with no handler is a
    bug, not a silent leak."""
    if (scenario, goal) in REGISTRY:
        return REGISTRY[(scenario, goal)]
    if goal in (None, ""):
        default = SCENARIO_DEFAULT_GOAL.get(scenario, "__missing__")
        if (scenario, default) in REGISTRY:
            return REGISTRY[(scenario, default)]
    raise KeyError(
        f"no goal handler registered for (scenario={scenario!r}, goal={goal!r}) — "
        f"every (scenario, goal) in the pipeline must have a handler in goal_handlers.REGISTRY"
    )


def assert_every_goal_has_a_handler(schema_goals=None, known_metrics=None):
    """STARTUP ASSERTION: every goal declared in the schema (envelope.yaml goal enum) — under EACH scenario
    that admits it — and every metric in check_invariants.KNOWN_METRICS must resolve to a handler. Fails
    loudly on the first gap. This turns the old leak-surface (a declared goal with no handler silently
    dispatching to the scenario default) into a hard error.

    `schema_goals`: iterable of goal strings (defaults to reading schema/envelope.yaml).
    `known_metrics`: {scenario: {metric,...}} (defaults to check_invariants.KNOWN_METRICS)."""
    problems = []
    # 1) every (scenario, goal) a handler claims must be unique + resolvable
    for (scenario, goal), h in REGISTRY.items():
        try:
            got = resolve(scenario, goal)
        except KeyError as e:
            problems.append(str(e))
            continue
        if got is not h:
            problems.append(f"registry inconsistency for ({scenario!r},{goal!r})")

    # 2) every metric declared per scenario must be produced by exactly one handler for that scenario
    if known_metrics is None:
        known_metrics = _known_metrics_from_check_invariants()
    handler_metrics_by_scenario: dict = {}
    for h in _HANDLERS:
        s = handler_metrics_by_scenario.setdefault(h.scenario, set())
        s.add(h.metric)
        s.update(h.alt_metrics)  # alternate exemplar metrics of the same goal-lane (e.g. tps_per_gpu_at_sla)
    for scenario, metrics in (known_metrics or {}).items():
        have = handler_metrics_by_scenario.get(scenario, set())
        missing = set(metrics) - have
        if missing:
            problems.append(
                f"scenario {scenario!r}: metrics {sorted(missing)} declared in KNOWN_METRICS "
                f"have NO handler (handlers produce {sorted(have)})"
            )

    # 3) every schema goal must have at least one handler (under some scenario)
    if schema_goals is None:
        schema_goals = _goals_from_schema()
    handler_goals = {h.goal for h in _HANDLERS if h.goal is not None}
    for g in schema_goals or []:
        if g not in handler_goals:
            problems.append(f"schema goal {g!r} has NO registered handler")

    if problems:
        raise AssertionError("goal_handlers startup assertion FAILED:\n  - " + "\n  - ".join(problems))
    return True


def _goals_from_schema():
    """Read the goal enum from schema/envelope.yaml (best-effort; empty on any failure)."""
    try:
        import yaml

        env = yaml.safe_load((ROOT / "schema/envelope.yaml").read_text())
        props = ((env.get("$defs") or {}).get("envelope") or {}).get("properties") or {}
        return list((props.get("goal") or {}).get("enum") or [])
    except Exception:
        return []


def _known_metrics_from_check_invariants():
    """Read KNOWN_METRICS from scripts/check_invariants.py without importing (it has heavy side effects at
    import time? no — but keep it decoupled). Falls back to parsing the module dict."""
    try:
        ci = _load("check_invariants", "scripts/check_invariants.py")
        return getattr(ci, "KNOWN_METRICS", None)
    except Exception:
        return None

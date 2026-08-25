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

"""Overlay two (or more) benchmark runs on one interactive comparison dashboard.

  SCALING CALC (top):  target-concurrency -> GPUs-needed per run, live from SLA.
  TOP SCATTER:         Output TPS/GPU vs Output TPS/User, one line per run.
                       Marker fill is recolored in the browser GREEN/RED from the
                       selected TTFT/TPOT percentile + limit controls (the run's
                       line + marker-outline color preserves run identity).
  BOTTOM PANELS:       cISL / nISL / OSL request-size distributions (linear,
                       stacked) for BOTH runs at a selectable, matched concurrency
                       -- explains *why* the two curves differ.

Scatter reads each run's metrics_summary.csv (schema-v6):
  throughput_per_gpu_tok_per_s   -> y (Output TPS/GPU, decode)
  tokens_per_s_per_user_from_itl -> x (Output TPS/User, 1000/ITL_p50)
  ttft_p50/p90/p99_ms, itl_p50/p90/p99_ms -> per-point SLA percentiles.
  p95 is NOT in the CSV, so ttft/itl p95 are read from each rung's
  logs/aiperf/profile_export_aiperf.json (nodes time_to_first_token /
  inter_token_latency, field .p95).
Distributions read each rung's logs/aiperf/profile_export.jsonl:
  cISL = usage_prompt_cache_read_tokens, nISL = ISL - cached, OSL = output_sequence_length

Usage:
  # single run (what report.sh calls):
  sla_compare_dashboard.py --run "<run-id>=results/<run-id>" --out results/<run-id>/dashboard.html
  # overlay two or more runs for comparison:
  sla_compare_dashboard.py --run "256K=results/<id256k>" --run "1M=results/<id1m>" \
      --out results/compare.html [--ttft-limit 10000] [--tpot-limit 100]
"""

from __future__ import annotations
import argparse, csv, json, math, re, sys
from pathlib import Path

try:
    import plotly.offline as poff
except ImportError:
    sys.stderr.write("ERROR: plotly required (pip install 'plotly>=5.20')\n")
    sys.exit(2)

# 2x2 encoding: COLOR = CONTEXT (all runs of one context share a hue), and
# LINE-STYLE + MARKER = OFFLOAD (offload = solid+filled, no-offload = dashed+open).
# Context hues are drawn (in first-seen order) from this palette -- deliberately
# NOT green or red so the pass/fail marker fill stays unambiguous. Blue/orange is
# the classic colorblind-safe pair used for the 2-context (256K / 1M) case.
CONTEXT_PALETTE = ["#2f7ed8", "#e0883b", "#8e6fb5", "#2aa198"]
PCTS = ("p50", "p90", "p95", "p99")
DIST = [
    ("cISL (cached input)", "cisl"),
    ("nISL (new input)", "nisl"),
    ("OSL (output)", "osl"),
]


def as_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def percentile(values, pct):
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * (pct / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_aiperf_p95(step_dir: Path):
    """Return {'ttft': p95, 'tpot': p95} from profile_export_aiperf.json (CSV lacks p95)."""
    f = step_dir / "logs" / "aiperf" / "profile_export_aiperf.json"
    out = {"ttft": None, "tpot": None}
    if not f.exists():
        return out
    try:
        d = json.load(open(f))
    except (OSError, json.JSONDecodeError):
        return out
    for key, node in (("ttft", "time_to_first_token"), ("tpot", "inter_token_latency")):
        n = d.get(node)
        if isinstance(n, dict):
            out[key] = as_float(n.get("p95"))
    return out


def load_incomplete_pct(step_dir: Path):
    """Fraction of issued requests that did NOT complete at this rung, as a percent.

    failed_pct = 100 * (1 - completed/issued), where
      completed = request_latency.count (requests that returned a full response)
      issued    = input_config.loadgen.request_count (requests the loadgen sent)
    both from profile_export_aiperf.json. Requests that do not complete within the configured client timeout are flagged because those points performed unequal work. Returns None when it can't be computed.
    """
    f = step_dir / "logs" / "aiperf" / "profile_export_aiperf.json"
    if not f.exists():
        return None
    try:
        d = json.load(open(f))
    except (OSError, json.JSONDecodeError):
        return None
    completed = as_float((d.get("request_latency") or {}).get("count"))
    issued = as_float(((d.get("input_config") or {}).get("loadgen") or {}).get("request_count"))
    if completed is None or issued is None or issued <= 0:
        return None
    return max(0.0, 100.0 * (1.0 - completed / issued))


def load_summary(path: Path):
    """One scatter point per concurrency with ttft/tpot {p50,p90,p95,p99}.

    p50/p90/p99 come from metrics_summary.csv; p95 (absent from the CSV) is
    supplemented from each rung's profile_export_aiperf.json.
    """
    csv_path = path / "metrics_summary.csv"
    if not csv_path.exists():
        raise SystemExit(f"metrics_summary.csv not found in {path} (run report.sh first)")
    pts = []
    for r in csv.DictReader(line for line in open(csv_path) if not line.startswith("#")):
        c = as_float(r.get("concurrency"))
        y = as_float(r.get("throughput_per_gpu_tok_per_s"))
        x = as_float(r.get("tokens_per_s_per_user_from_itl"))
        if None in (c, y, x):
            continue
        c = int(c)
        ttft = {p: as_float(r.get(f"ttft_{p}_ms")) for p in ("p50", "p90", "p99")}
        tpot = {p: as_float(r.get(f"itl_{p}_ms")) for p in ("p50", "p90", "p99")}
        step_dir = path / f"concurrency_{c}"
        p95 = load_aiperf_p95(step_dir)
        ttft["p95"] = p95["ttft"]
        tpot["p95"] = p95["tpot"]
        failed_pct = load_incomplete_pct(step_dir)
        # KV prefix-cache hit rate (sparsely populated); prefer vllm column, fall
        # back to overall hit rate, else None -> rendered "KV n/a" in the browser.
        kv_hit = as_float(r.get("vllm_prefix_cache_hit_rate_pct"))
        if kv_hit is None:
            kv_hit = as_float(r.get("hit_rate_overall_pct"))
        # Backfill any percentile the CSV was missing from the aiperf JSON as well.
        pts.append(
            {
                "concurrency": c,
                "tps_gpu": y,
                "tps_user": x,
                "ttft": {p: ttft.get(p) for p in PCTS},
                "tpot": {p: tpot.get(p) for p in PCTS},
                "kv_hit": kv_hit,
                "failed_pct": failed_pct,
            }
        )
    pts.sort(key=lambda p: p["concurrency"])
    return pts


def dist_stats(samples):
    if not samples:
        return {
            "samples": [],
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "samples": samples,
        "p50": percentile(samples, 50),
        "p90": percentile(samples, 90),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples),
    }


def load_distributions(path: Path):
    """{concurrency: {'cisl':stats, 'nisl':stats, 'osl':stats}} from profile_export.jsonl."""
    out = {}
    for d in sorted(path.glob("concurrency_*")):
        try:
            c = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        f = d / "logs" / "aiperf" / "profile_export.jsonl"
        if not f.exists():
            continue
        islt, cisl, nisl, osl = [], [], [], []
        for line in open(f):
            try:
                m = json.loads(line)["metrics"]
            except Exception:
                continue
            if "input_sequence_length" not in m:
                continue
            isl = m["input_sequence_length"]["value"]  # total input = trace-fixed request size
            cr = (m.get("usage_prompt_cache_read_tokens") or {}).get("value", 0) or 0
            cr = max(0, min(cr, isl))
            islt.append(isl)  # ISL total (fixed by the trace)
            cisl.append(cr)  # cached split (RUNTIME, varies w/ concurrency)
            nisl.append(max(isl - cr, 0))  # recomputed split (RUNTIME)
            osl.append((m.get("output_sequence_length") or {}).get("value", 0))
        if cisl:
            out[c] = {
                "isl": dist_stats(islt),
                "cisl": dist_stats(cisl),
                "nisl": dist_stats(nisl),
                "osl": dist_stats(osl),
            }
    return out


def context_token(label: str):
    """Extract a normalized max-context token (e.g. '256K', '1M') from a run label."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([kKmM])\b", label)
    if not m:
        return None
    return f"{float(m.group(1)):g}{m.group(2).upper()}"


def offload_token(label: str):
    """Classify a run label as 'offload' or 'no-offload' (CPU-KV-offload dimension).

    Order matters: 'no-offload'/'noofl' must be matched before the bare 'offload'
    substring (which they contain). Returns None if the label carries no signal.
    """
    l = label.lower()
    if re.search(r"no[\s._-]*offload", l) or "noofl" in l or "no-ofl" in l:
        return "no-offload"
    if "offload" in l or re.search(r"\bofl\b", l) or "ofl" in l:
        return "offload"
    return None


def _signature_vector(run):
    """Pooled-over-all-rungs cISL/nISL/OSL P50/P90/P95 vector (fallback detector)."""
    pooled = {"cisl": [], "nisl": [], "osl": []}
    for dd in run["dist"].values():
        for k in pooled:
            pooled[k] += dd[k]["samples"]
    vec = []
    for k in ("cisl", "nisl", "osl"):
        for p in (50, 90, 95):
            vec.append(percentile(pooled[k], p) or 0.0)
    return vec


def same_workload(runs) -> bool:
    """Do all runs replay the SAME request-size distribution?

    Primary signal is the run label's max-context token ("1M" vs "1M" == same;
    "256K" vs "1M" == different) -- robust even when one run only completed a
    partial (early, low-context) slice of the trace. If labels carry no context
    token, fall back to comparing pooled cISL/nISL/OSL P50/P90/P95 within ~12%.
    """
    tokens = [context_token(r["label"]) for r in runs]
    if all(t is not None for t in tokens):
        return len(set(tokens)) == 1
    base = _signature_vector(runs[0])
    for r in runs[1:]:
        for x, y in zip(base, _signature_vector(r)):
            m = max(abs(x), abs(y), 1.0)
            if abs(x - y) / m > 0.12:
                return False
    return True


def fairness_info(runs, same) -> dict:
    """Fairness guard: when runs are INTENDED to be the same workload (same==True)
    but their OBSERVED workloads diverge, the comparison is not apples-to-apples.

    At a shared (mid common) concurrency rung, compute per run the completed-request
    count (len of samples) and the mean TOTAL INPUT tokens (cISL + nISL, the same
    samples that feed the distribution panels). Flag `unfair` when same==True and
    either the mean input differs by >25% or the completed-request counts differ by
    >2x across runs. Never flag when same==False (intentionally different workloads).
    """
    common = sorted(
        (set.intersection(*[set(r["dist"]) for r in runs]) if runs else set()),
        key=lambda x: int(x),
    )
    if not common:
        return {"applicable": False, "sameWorkload": same, "unfair": False, "runs": []}
    rung = common[len(common) // 2]  # matches the browser's default distConc pick
    stats = []
    for r in runs:
        d = r["dist"][rung]
        cisl, nisl = d["cisl"]["samples"], d["nisl"]["samples"]
        n = len(cisl)
        mean_in = (sum(cisl) + sum(nisl)) / n if n else 0.0
        stats.append(
            {
                "label": r["label"],
                "reqCount": n,
                "meanInput": mean_in,
                "meanInputK": mean_in / 1000.0,
            }
        )

    unfair = False
    if same and len(stats) >= 2:
        means = [s["meanInput"] for s in stats if s["reqCount"] > 0]
        counts = [s["reqCount"] for s in stats if s["reqCount"] > 0]
        if means and min(means) > 0 and max(means) / min(means) > 1.25:
            unfair = True
        if counts and min(counts) > 0 and max(counts) / min(counts) > 2.0:
            unfair = True
    return {
        "applicable": True,
        "sameWorkload": same,
        "unfair": unfair,
        "rung": int(rung),
        "runs": stats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ttft-limit", type=float, default=10000.0)
    ap.add_argument("--tpot-limit", type=float, default=100.0)
    ap.add_argument(
        "--gpus-per-instance",
        type=int,
        default=8,
        help="GPU count for one serving instance, used by the scaling calculator (default: 8)",
    )
    args = ap.parse_args()
    if len(args.run) < 1:
        raise SystemExit(
            "Need at least one --run LABEL=PATH entry "
            "(one run renders a single-run dashboard; "
            "two or more overlay them for comparison)."
        )

    # Assign one hue per CONTEXT (in first-seen order) so both runs of a context
    # share a color; offload/no-offload are then distinguished by line+marker style.
    ctx_color = {}
    runs = []
    for spec in args.run:
        label, _, p = spec.partition("=")
        rp = Path(p)
        ctx = context_token(label)
        offload = offload_token(label)
        key = ctx if ctx is not None else label  # fall back to per-label hue
        if key not in ctx_color:
            ctx_color[key] = CONTEXT_PALETTE[len(ctx_color) % len(CONTEXT_PALETTE)]
        dist = load_distributions(rp)
        runs.append(
            {
                "label": label,
                "context": ctx if ctx is not None else "?",
                "offload": offload if offload is not None else "offload",
                "color": ctx_color[key],
                "points": load_summary(rp),
                "dist": {str(c): v for c, v in dist.items()},
            }
        )

    same = same_workload(runs)
    fairness = fairness_info(runs, same)
    payload = {
        "runs": runs,
        "ttftLimit": args.ttft_limit,
        "tpotLimit": args.tpot_limit,
        "gpuPerServer": args.gpus_per_instance,
        "sameWorkload": same,
        "fairness": fairness,
    }
    data_json = json.dumps(payload)
    plotly_js = poff.get_plotlyjs()

    n_common = len(set.intersection(*[set(r["dist"]) for r in runs])) if runs else 0
    html_text = (
        PAGE.replace("__PLOTLY_JS__", plotly_js).replace("__METHODOLOGY__", METHODOLOGY).replace("__DATA__", data_json)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text)
    n_ctx = len({r["context"] for r in runs})
    layout = (
        "single 4-panel overlay (one pooled context)"
        if n_ctx <= 1
        else f"per-metric bordered groups, one pooled row per context ({n_ctx} contexts)"
    )
    if fairness.get("unfair"):
        detail = ", ".join(f"{s['label']}={s['reqCount']} reqs/~{s['meanInputK']:.0f}k" for s in fairness["runs"])
        fair = f"FAIRNESS WARNING fired (workloads diverge: {detail})"
    elif same and fairness.get("applicable"):
        fair = "fair comparison (workloads matched)"
    else:
        fair = "fairness guard N/A (intentionally different workloads)"
    print(f"Wrote {args.out} ({len(runs)} runs, {n_common} common concurrencies; " f"distributions: {layout}; {fair})")
    return 0


METHODOLOGY = r"""
<details>
  <summary>Methodology &#9662;</summary>
  <p>The dashboard overlays one or more AIPerf runs and uses each run metadata and metrics.
  It does not assume a particular model, cluster, or serving command.</p>
  <p><b>Output TPS/GPU</b> is decode throughput per GPU. <b>Output TPS/User</b>
  is the per-user decode rate (1000 / median inter-token latency). A point passes
  only when both the selected TTFT and TPOT percentiles are within their limits.</p>
  <p>The distribution panels show cached input, new input, and output token lengths
  at matched concurrency so workload differences remain visible. The scaling
  calculator uses the GPU count supplied with <code>--gpus-per-instance</code>.</p>
</details>
"""

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark run comparison</title>
<script>__PLOTLY_JS__</script>
<style>
body{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;max-width:1180px;
  margin:22px auto;padding:0 16px;color:#111}
details{border:1px solid #e2e2df;border-radius:8px;padding:10px 18px;margin:0 0 18px;background:#fafaf9}
summary{font-weight:700;font-size:16px;cursor:pointer;user-select:none}
details.sub{margin:10px 0;padding:8px 14px;background:#fff;border-color:#e7e7e4}
details.sub>summary{font-size:14px;font-weight:650}
details h3{margin:15px 0 3px;font-size:14.5px}
details p,details li{color:#2c2c2a;line-height:1.55;font-size:13.5px;margin:4px 0}
.lead{font-size:14px;color:#52514e}
.note{margin-top:12px;padding-top:10px;border-top:1px solid #ececea;color:#52514e}
code{background:#f0f0ee;padding:1px 4px;border-radius:3px;font-size:12.5px}
.modelmeta{font-size:13.5px;line-height:1.6;color:#2c2c2a}
details pre{background:#1e232b;color:#e6edf3;border-radius:7px;padding:12px 14px;overflow-x:auto;
  font-size:12.5px;line-height:1.5;margin:10px 0 0}
details pre code{background:none;color:inherit;padding:0;font-size:12.5px}
.panel{border:1px solid #e2e2df;border-radius:8px;padding:14px 16px;margin:0 0 18px;background:#fff}
.panel h2{margin:0 0 10px;font-size:15px}
.dist-group{border:1px solid #d3d8de;border-radius:10px;padding:10px 12px 4px;margin:0 0 16px;
  background:#fcfcfb}
.dist-group-hd{font-weight:750;font-size:14px;color:#263241;margin:2px 2px 6px}
.chart-cap{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;margin:10px 2px 0;
  font-size:12.5px;color:#3b4654}
.chart-cap .lg{display:inline-flex;align-items:center;gap:6px}
.chart-cap .dot{width:12px;height:12px;border-radius:50%;display:inline-block;
  border:1px solid rgba(0,0,0,0.18)}
.chart-cap .star{font-size:14px;line-height:1}
.chart-cap .cap-txt{flex-basis:100%;color:#52514e;margin-top:1px}
.fair-warn{display:flex;gap:13px;align-items:flex-start;
  background:linear-gradient(90deg,#ffe1c2,#ffd0cb);border:2px solid #c0392b;
  border-left:8px solid #c0392b;border-radius:9px;padding:14px 18px;margin:0 0 18px;
  color:#4a140c;font-size:14px;line-height:1.5;box-shadow:0 2px 8px rgba(192,57,43,0.18)}
.fair-warn b{color:#96160b}
.fair-warn .fw-icon{font-size:24px;line-height:1.2}
.fair-warn .fw-nums{font-weight:700;color:#7a1a10}
.fair-ok{background:#eef7ee;border:1px solid #bcdcbc;border-radius:8px;padding:8px 14px;
  margin:0 0 16px;color:#256029;font-size:12.5px;font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:end}
.controls .field{display:flex;flex-direction:column;gap:4px}
label{font-size:12px;font-weight:650;color:#3b4654}
select,input{border:1px solid #bdc7d5;border-radius:6px;padding:7px 9px;background:#fff;
  color:#17202a;font-size:14px}
input[type=number]{width:110px}
.scaling{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.scard{border:1px solid #e2e2df;border-radius:8px;padding:10px 13px;min-width:230px;flex:1}
.scard .lbl{font-weight:700;font-size:13px;display:flex;align-items:center;gap:7px}
.swatch{width:13px;height:13px;border-radius:3px;display:inline-block}
.scard .big{font-size:24px;font-weight:750;margin:5px 0 2px}
.scard .sub{color:#52514e;font-size:12.5px}
.scard.unmet .big{color:#c0392b;font-size:15px}
.runsel{border:1px solid #d3d8de;border-radius:9px;padding:10px 13px;margin:0 0 12px;background:#f7f9fb}
.runsel-hd{font-weight:750;font-size:13.5px;color:#263241;margin:0 0 8px}
.runchecks{display:flex;flex-wrap:wrap;gap:8px 10px;margin-bottom:9px}
.runchk{display:inline-flex;align-items:center;gap:7px;border:1px solid #cdd6e0;border-radius:999px;
  padding:5px 11px 5px 9px;background:#fff;font-size:12.5px;font-weight:600;cursor:pointer;
  color:#26303b;user-select:none}
.runchk input{margin:0;cursor:pointer}
.runchk .rc-dot{width:11px;height:11px;border-radius:50%;display:inline-block;
  border:1.5px solid rgba(0,0,0,0.15)}
.runpresets{display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.runpresets .pre-lbl{font-size:11.5px;font-weight:650;color:#52606d;margin-right:2px}
.preset-btn{border:1px solid #b9c4d0;border-radius:6px;background:#fff;color:#1f2c38;
  font-size:12px;font-weight:600;padding:5px 10px;cursor:pointer}
.preset-btn:hover{background:#eef3f8;border-color:#8fa1b3}
.preset-btn:active{background:#e2eaf2}
</style></head><body>
<div id="fairnessBanner"></div>
__METHODOLOGY__

<div class="panel">
  <h2>Scaling calculator — how many GPUs for the SLA below?</h2>
  <div class="controls">
    <div class="field">
      <label for="targetConc">Target concurrency (concurrent users)</label>
      <input id="targetConc" type="number" min="1" step="50" value="1000">
    </div>
    <div class="field" style="justify-content:flex-end">
      <div class="note" style="margin:0;padding:0;border:0">
        GPUs&nbsp;=&nbsp;ceil(target / best-passing-concurrency) &times; <span id="gpsLbl">8</span> GPUs / serving instance
      </div>
    </div>
  </div>
  <div id="scaling" class="scaling"></div>
</div>

<div class="panel">
  <div class="runsel">
    <div class="runsel-hd">Runs to compare (toggle any subset — everything below updates live)</div>
    <div id="runChecks" class="runchecks"></div>
    <div class="runpresets"><span class="pre-lbl">Presets:</span><span id="runPresets"></span></div>
  </div>
  <h2>SLA filter</h2>
  <div class="controls">
    <div class="field">
      <label for="ttftPct">TTFT percentile</label>
      <select id="ttftPct">
        <option value="p50">P50</option><option value="p90">P90</option>
        <option value="p95">P95</option><option value="p99">P99</option>
      </select>
    </div>
    <div class="field">
      <label for="ttftLimit">TTFT limit (ms)</label>
      <input id="ttftLimit" type="number" min="0" step="100" value="10000">
    </div>
    <div class="field">
      <label for="tpotPct">TPOT percentile</label>
      <select id="tpotPct">
        <option value="p50">P50</option><option value="p90">P90</option>
        <option value="p95">P95</option><option value="p99">P99</option>
      </select>
    </div>
    <div class="field">
      <label for="tpotLimit">TPOT limit (ms)</label>
      <input id="tpotLimit" type="number" min="0" step="1" value="100">
    </div>
    <div class="field" style="justify-content:flex-end">
      <label style="display:flex;align-items:center;gap:7px;font-weight:600;cursor:pointer">
        <input id="showDetail" type="checkbox">
        Show TTFT + KV hit rate on points
      </label>
    </div>
  </div>
  <div id="scatter" style="width:100%;height:560px;margin-top:8px"></div>
  <div id="chartCap" class="chart-cap"></div>
</div>

<div class="panel">
  <h2>Request-size distributions (ISL / cISL / nISL / OSL)</h2>
  <div class="controls">
    <div class="field">
      <label for="distConc">Concurrency rung</label>
      <select id="distConc"></select>
    </div>
  </div>
  <div id="dist" style="width:100%;height:1080px;margin-top:8px"></div>
  <div id="distGroups" style="margin-top:8px;display:none"></div>
  <p class="note" style="margin-top:10px"><b>ISL is the trace-fixed request size.</b>
  cISL + nISL = ISL, but the split is a <b>RUNTIME</b> property — under higher concurrency
  the prefix cache evicts more, so cISL falls and nISL rises even for the same trace.
  This makes workload changes under load visible.</p>
</div>

<script>
const PAYLOAD = __DATA__;
const RUNS = PAYLOAD.runs;
const GPU_PER_SERVER = PAYLOAD.gpuPerServer;
const FAIL = "#d73027";  // failing points turn red; passing points use each run's CONTEXT color
const INCOMPLETE_THRESHOLD = 1.0;   // % of dropped requests above which a point is flagged
const WARN_RED = "#c0140b";         // deep red for the incomplete-requests overlay
function fmtPct(p){ return (p >= 3 ? Math.round(p) : Number(p.toFixed(1))) + "%"; }
document.getElementById("gpsLbl").textContent = GPU_PER_SERVER;
document.getElementById("ttftLimit").value = PAYLOAD.ttftLimit;
document.getElementById("tpotLimit").value = PAYLOAD.tpotLimit;

// ---------- run selection (2x2: context x offload) ----------
// SEL[i] = is run i currently shown. Default: the "Offload: 256K vs 1M" view (both
// offload runs); falls back to all runs when that selects <2 (2-run HEADLINE pages).
let SEL = RUNS.map(r => r.offload === "offload");   // default view: Offload 256K vs 1M
if (SEL.filter(Boolean).length < 2) SEL = RUNS.map(() => true);   // fallback (2-run pairwise dashboards show both)
const IDX = RUNS.map((_, i) => i);
function isSel(i){ return SEL[i]; }
function isOpen(run){ return run.offload === "no-offload"; }   // open marker + dashed line
function selectedRuns(){ return RUNS.filter((_, i) => SEL[i]); }
function anySelected(){ return SEL.some(Boolean); }

function num(id){ const v = Number(document.getElementById(id).value); return Number.isFinite(v) ? v : 0; }
function fmtK(v){
  if(v==null||!Number.isFinite(v)) return "-";
  if(v>=1000) return (v/1000).toLocaleString(undefined,{maximumFractionDigits:1})+"K";
  return v.toLocaleString(undefined,{maximumFractionDigits:0});
}

// --- pass/fail evaluation shared by scatter recolor + scaling calc ---
function passes(pt, ttftPct, ttftLimit, tpotPct, tpotLimit){
  const a = pt.ttft[ttftPct], b = pt.tpot[tpotPct];
  return (a!=null && Number.isFinite(a) && a<=ttftLimit) &&
         (b!=null && Number.isFinite(b) && b<=tpotLimit);
}
function slaSel(){
  return {
    ttftPct: document.getElementById("ttftPct").value,
    tpotPct: document.getElementById("tpotPct").value,
    ttftLimit: num("ttftLimit"),
    tpotLimit: num("tpotLimit"),
  };
}
// highest-concurrency point that PASSES the selected SLA (the capacity number), or null.
function bestPassing(run, s){
  let best = null;
  for(const p of run.points){                       // points are sorted ascending
    if(passes(p, s.ttftPct, s.ttftLimit, s.tpotPct, s.tpotLimit)) best = p.concurrency;
  }
  return best;
}

// ---------- top scatter ----------
// Point text labels: default "c=<N>"; the best-passing point is starred + bolded.
// When the detail toggle is ON, add the TTFT (seconds, at selected ttftPct) and KV hit rate.
function pointLabels(run){
  const detail = document.getElementById("showDetail").checked;
  const ttftPct = document.getElementById("ttftPct").value;
  const best = bestPassing(run, slaSel());
  return run.points.map(p => {
    const isBest = p.concurrency === best;
    const head = isBest ? "★ <b>c="+p.concurrency+" (best passing)</b>" : "c="+p.concurrency;
    if(!detail) return head;
    const t = p.ttft[ttftPct];
    const ttftStr = (t!=null && Number.isFinite(t))
      ? "TTFT("+ttftPct+") "+(t/1000).toPrecision(2)+"s"
      : "TTFT("+ttftPct+") n/a";
    const kv = p.kv_hit;
    const kvStr = (kv!=null && Number.isFinite(kv)) ? "KV "+Math.round(kv)+"%" : "KV n/a";
    const fp = p.failed_pct;
    const incStr = (fp!=null && Number.isFinite(fp) && fp > INCOMPLETE_THRESHOLD)
      ? "<br><span style='color:"+WARN_RED+"'>incomplete "+fmtPct(fp)+"</span>" : "";
    return head+"<br>"+ttftStr+"<br>"+kvStr+incStr;
  });
}
function updateScatterText(){
  Plotly.restyle("scatter", {text: RUNS.map(run => pointLabels(run))}, RUNS.map((_,i)=>i));
}

function buildScatter(){
  // One trace per run (ALL runs built once). COLOR = context; LINE = solid (offload)
  // vs dashed (no-offload). Marker fill/open + pass/fail red are applied in
  // recolorScatter(); visibility follows the run-selection checkboxes.
  const traces = RUNS.map((run, i) => {
    const pts = run.points;
    const open = isOpen(run);
    return {
      type:"scatter", mode:"lines+markers+text",
      name: run.label,
      visible: SEL[i] ? true : false,
      x: pts.map(p=>p.tps_user), y: pts.map(p=>p.tps_gpu),
      text: pointLabels(run), textposition:"top center",
      textfont:{size:10, color:"#52514e"},
      line:{color:run.color, width:2, dash: open ? "dash" : "solid"},
      marker:{
        // open (no-offload) = white fill + colored ring; filled (offload) = colored fill.
        size:13, symbol:"circle",
        color: pts.map(_=> open ? "#ffffff" : run.color),
        line:{color: pts.map(_=> open ? run.color : "#fff"), width: open ? 1.8 : 1.2},
      },
      customdata: pts.map(p=>[p.ttft.p50,p.ttft.p90,p.ttft.p95,p.ttft.p99,
                              p.tpot.p50,p.tpot.p90,p.tpot.p95,p.tpot.p99,p.concurrency,
                              (p.failed_pct!=null && Number.isFinite(p.failed_pct)) ? p.failed_pct : -1]),
      hovertemplate:"<b>"+run.label+"</b> c=%{customdata[8]}<br>"+
        "TPS/User %{x:.2f}<br>TPS/GPU %{y:,.0f}<br>"+
        "TTFT p50/p90/p95/p99: %{customdata[0]:,.0f}/%{customdata[1]:,.0f}/%{customdata[2]:,.0f}/%{customdata[3]:,.0f} ms<br>"+
        "TPOT p50/p90/p95/p99: %{customdata[4]:.1f}/%{customdata[5]:.1f}/%{customdata[6]:.1f}/%{customdata[7]:.1f} ms<br>"+
        "incomplete: %{customdata[9]:.1f}%<extra></extra>",
    };
  });
  Plotly.newPlot("scatter", traces, {
    template:"plotly_white",
    margin:{l:72,r:24,t:12,b:56},
    legend:{title:{text:"Run"}, x:0.99, y:0.99, xanchor:"right", yanchor:"top",
            bgcolor:"rgba(255,255,255,0.7)"},
    xaxis:{title:{text:"Output TPS/User (higher = faster UX)"}, rangemode:"tozero"},
    yaxis:{title:{text:"Output TPS/GPU (higher = GPU efficiency)"}, rangemode:"tozero"},
  }, {displayModeBar:false, responsive:true});
}

// recolor EVERY point across all series via Plotly.restyle (applied to all traces;
// hidden runs simply don't show). 2x2 encoding is preserved:
//   pass + offload      -> context-color fill, white halo ring
//   pass + no-offload   -> WHITE fill, context-color ring (open marker)
//   fail (either)       -> RED fill, white outline (red overrides marker fill)
// The best-passing point per run is enlarged into a ★ (keeping its fill rule).
function recolorScatter(){
  const s = slaSel();
  const color = [], size = [], symbol = [], lineColor = [], lineWidth = [];
  RUNS.forEach(run => {
    const best = bestPassing(run, s);
    const open = isOpen(run);
    color.push(run.points.map(p => {
      const pass = passes(p, s.ttftPct, s.ttftLimit, s.tpotPct, s.tpotLimit);
      if(!pass) return FAIL;
      return open ? "#ffffff" : run.color;
    }));
    lineColor.push(run.points.map(p => {
      const pass = passes(p, s.ttftPct, s.ttftLimit, s.tpotPct, s.tpotLimit);
      if(!pass) return "#fff";              // failing points keep the white outline
      return open ? run.color : "#fff";     // open marker keeps a context-color ring
    }));
    size.push(run.points.map(p => p.concurrency === best ? 22 : 13));
    symbol.push(run.points.map(p => p.concurrency === best ? "star" : "circle"));
    lineWidth.push(run.points.map(p => p.concurrency === best ? 3 : (open ? 1.8 : 1.2)));
  });
  Plotly.restyle("scatter", {
    "marker.color": color, "marker.size": size, "marker.symbol": symbol,
    "marker.line.color": lineColor, "marker.line.width": lineWidth,
  }, IDX);
}

// show/hide traces per the current selection (no page reload).
function applyScatterVisibility(){
  Plotly.restyle("scatter", {visible: RUNS.map((_, i) => (SEL[i] ? true : false))}, IDX);
}

// Flag selected scatter points whose incomplete-request percentage exceeds the threshold.
// These points performed unequal work and should not be compared directly.
function updateIncompleteAnnotations(){
  const anns = [];
  selectedRuns().forEach(run => {
    run.points.forEach(p => {
      const fp = p.failed_pct;
      if(fp == null || !Number.isFinite(fp) || fp <= INCOMPLETE_THRESHOLD) return;
      anns.push({
        x: p.tps_user, y: p.tps_gpu, xref: "x", yref: "y",
        text: "<b>⚠ " + fmtPct(fp) + " incomplete</b>",
        showarrow: true, arrowhead: 0, arrowwidth: 1, arrowcolor: WARN_RED,
        ax: 6, ay: 30, xanchor: "left", yanchor: "top",
        font: {color: WARN_RED, size: 10.5},
        bgcolor: "rgba(255,255,255,0.88)", bordercolor: WARN_RED, borderwidth: 1, borderpad: 2,
      });
    });
  });
  Plotly.relayout("scatter", {annotations: anns});
}

// ---------- scaling calculator ----------
function updateScaling(){
  const s = slaSel();
  const target = Math.max(1, Math.ceil(num("targetConc")));
  const host = document.getElementById("scaling");
  host.innerHTML = "";
  selectedRuns().forEach(run => {
    let best = null;
    for(const p of run.points){                    // points are sorted ascending
      if(passes(p, s.ttftPct, s.ttftLimit, s.tpotPct, s.tpotLimit)) best = p.concurrency;
    }
    const card = document.createElement("div");
    card.className = "scard" + (best==null ? " unmet" : "");
    if(best==null){
      card.innerHTML =
        '<div class="lbl"><span class="swatch" style="background:'+run.color+'"></span>'+run.label+'</div>'+
        '<div class="big">SLA unmet at lowest concurrency</div>'+
        '<div class="sub">no swept rung meets the selected TTFT/TPOT SLA</div>';
    } else {
      const gpus = Math.ceil(target / best) * GPU_PER_SERVER;
      const instances = Math.ceil(target / best);
      card.innerHTML =
        '<div class="lbl"><span class="swatch" style="background:'+run.color+'"></span>'+run.label+'</div>'+
        '<div class="big">'+gpus.toLocaleString()+' GPUs</div>'+
        '<div class="sub">sustains <b>c='+best+'</b> @ SLA → '+instances.toLocaleString()+
        ' serving instance(s) &times; '+GPU_PER_SERVER+' GPUs for '+target.toLocaleString()+' concurrent</div>';
    }
    host.appendChild(card);
  });
}

// ---------- distribution panels (linear, stacked) ----------
// title uses a real em dash (Plotly renders text literally, so no HTML entities).
const PANELS = [
  {key:"isl",  title:"ISL — total input (trace request size)",          xmax:1050000, binsize:32000, zoom:false},
  {key:"cisl", title:"cISL — cached input · prefix-cache read (runtime)", xmax:1050000, binsize:32000, zoom:false},
  {key:"nisl", title:"nISL — new input · recomputed (runtime)",           xmaxCap:120000, binsize:2000, zoom:true},
  {key:"osl",  title:"OSL — output",                                      xmaxCap:15000,  binsize:300,  zoom:true},
];
// P50 black dashed, P90 purple dash-dot, P95 red dotted
const PLINES = [
  {p:"p50", color:"#111111", dash:"dash",    name:"P50"},
  {p:"p90", color:"#7d3cbe", dash:"dashdot", name:"P90"},
  {p:"p95", color:"#d1352b", dash:"dot",     name:"P95"},
];

function poolPct(distsAt, key, p){
  // pooled percentile across runs present at this rung (the workload is the same trace).
  // distsAt is an array of per-concurrency dist objects, each with .cisl/.nisl/.osl.
  let all = [];
  distsAt.forEach(d => { all = all.concat(d[key].samples); });
  if(!all.length) return null;
  all.sort((a,b)=>a-b);
  const pos = (all.length-1)*(p/100);
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo===hi ? all[lo] : all[lo]*(hi-pos)+all[hi]*(pos-lo);
}

// Distributions are keyed by CONTEXT, not per run: within a context the offload
// and no-offload runs replay the identical trace, so there is exactly ONE pooled
// distribution per context. Group the SELECTED runs (that have data at this rung)
// by context, in first-seen order.
function ctxGroupsAt(c){
  const runsAt = selectedRuns().filter(r => r.dist[c]);
  const order = [], map = {};
  runsAt.forEach(r => {
    if(!(r.context in map)){ map[r.context] = {ctx:r.context, color:r.color, runs:[]}; order.push(r.context); }
    map[r.context].runs.push(r);
  });
  return order.map(k => map[k]);
}
// pooled samples / percentile for one context group at rung c
function grpSamples(group, c, key){
  let a = []; group.runs.forEach(r => { a = a.concat(r.dist[c][key].samples); }); return a;
}
function grpDistObjs(group, c){ return group.runs.map(r => r.dist[c]); }
function grpStat(group, c, key){
  const objs = grpDistObjs(group, c);
  const all = grpSamples(group, c, key);
  return {p50:poolPct(objs,key,50), p90:poolPct(objs,key,90), p95:poolPct(objs,key,95),
          p99:poolPct(objs,key,99), max: all.length ? Math.max(...all) : null, samples:all};
}

// Dispatcher: ONE unique context selected -> a single overlaid 3-panel block;
// MULTIPLE contexts -> the full-width bordered per-metric-group layout, one pooled
// row per context. Reflows on every selection change.
function drawDist(){
  const c = document.getElementById("distConc").value;
  const groups = ctxGroupsAt(c);
  const distDiv = document.getElementById("dist");
  const groupsDiv = document.getElementById("distGroups");

  if(groups.length <= 1){
    groupsDiv.style.display = "none";
    groupsDiv.innerHTML = "";
    distDiv.style.display = "";
    drawOverlay(c, groups[0] || null);
  } else {
    Plotly.purge(distDiv);
    distDiv.style.display = "none";
    groupsDiv.style.display = "";
    drawGroups(c, groups, groupsDiv);
  }
}

function panelXmax(distObjs, panel){
  const p99 = poolPct(distObjs, panel.key, 99) || 0;
  return panel.zoom ? Math.max(panel.binsize*8, Math.min(panel.xmaxCap, p99*1.12)) : panel.xmax;
}

// ----- single unique context: one overlaid 3-panel figure (pooled histogram per panel) -----
function drawOverlay(c, group){
  const distDiv = document.getElementById("dist");
  if(!group){                                  // nothing selected / no data at rung
    Plotly.purge(distDiv);
    distDiv.innerHTML = '<div class="note" style="padding:24px 8px">No selected run has '+
      'distribution data at this concurrency rung.</div>';
    return;
  }
  const wlName = group.ctx + " workload" +
    (group.runs.length > 1 ? " (pooled offload + no-offload)" : "");
  const traces = [], shapes = [], annotations = [], axes = {};
  const nP = PANELS.length;
  const gap = 0.08, ph = (1 - gap*(nP-1))/nP;
  PANELS.forEach((panel, i) => {
    const s = i === 0 ? "" : String(i+1);
    const xax = "x"+s, yax = "y"+s;
    const top = 1 - i*(ph+gap);
    const ydom = [top-ph, top];
    const distObjs = grpDistObjs(group, c);
    const xmax = panelXmax(distObjs, panel);
    const bins = {start:0, end:panel.zoom ? panel.xmaxCap*2 : panel.xmax, size:panel.binsize};

    const all = grpSamples(group, c, panel.key);
    traces.push({
      type:"histogram", x: all, histnorm:"percent", xbins:bins,
      marker:{color:group.color, line:{width:0}}, opacity:0.72,
      name:wlName, showlegend:i===0, xaxis:xax, yaxis:yax,
      hovertemplate:wlName+"<br>%{x} tok: %{y:.1f}%<extra></extra>",
    });

    PLINES.forEach((pl, pi) => {
      const v = poolPct(distObjs, panel.key, Number(pl.p.slice(1)));
      if(v==null || v>xmax) return;
      shapes.push({type:"line", x0:v, x1:v, xref:xax, y0:0, y1:1, yref:yax+" domain",
                   line:{color:pl.color, width:2, dash:pl.dash}});
      annotations.push({x:v, xref:xax, y:0.96 - pi*0.14, yref:yax+" domain",
                        xanchor:(v > 0.62*xmax ? "right" : "left"), yanchor:"top",
                        text:"<b>"+pl.name+" "+fmtK(v)+"</b>", showarrow:false,
                        font:{color:pl.color, size:10.5}});
    });

    const st = {p50:poolPct(distObjs,panel.key,50), p90:poolPct(distObjs,panel.key,90),
                p95:poolPct(distObjs,panel.key,95), p99:poolPct(distObjs,panel.key,99),
                max:all.length?Math.max(...all):null};
    annotations.push({
      x:0.985, xref:xax+" domain", y:0.98, yref:yax+" domain", xanchor:"right", yanchor:"top",
      align:"right", showarrow:false, font:{size:9.5, color:"#2c2c2a"},
      bgcolor:"rgba(255,255,255,0.82)", bordercolor:"#d9d9d6", borderwidth:1, borderpad:3,
      text:wlName+"  P50 "+fmtK(st.p50)+" · P90 "+fmtK(st.p90)+" · P95 "+fmtK(st.p95)+
           " · P99 "+fmtK(st.p99)+" · max "+fmtK(st.max),
    });

    if(panel.zoom){
      const beyond = all.length ? 100*all.filter(v=>v>xmax).length/all.length : 0;
      annotations.push({x:0.985, xref:xax+" domain", y:0.04, yref:yax+" domain",
        xanchor:"right", yanchor:"bottom", showarrow:false, font:{size:9, color:"#7a5a1e"},
        text:"tail → max "+fmtK(st.max)+" ("+beyond.toFixed(1)+"% beyond view)"});
    }

    axes["xaxis"+s] = {domain:[0,1], anchor:yax, range:[0,xmax],
      title:{text:panel.title, font:{size:11}}, zeroline:false};
    axes["yaxis"+s] = {domain:ydom, anchor:xax,
      title:{text:"% of requests", font:{size:10}}, rangemode:"tozero"};
  });

  const distH = nP * 270;                            // ~270px per stacked panel
  distDiv.style.height = distH + "px";
  if(!distDiv._fullLayout) distDiv.innerHTML = "";   // drop any prior "no data" placeholder before init
  Plotly.react("dist", traces, Object.assign({
    template:"plotly_white", barmode:"overlay", height:distH,
    margin:{l:58, r:20, t:44, b:40}, bargap:0.02, showlegend:true,
    legend:{orientation:"h", x:0, y:1.03, xanchor:"left", yanchor:"bottom"},
    shapes, annotations,
  }, axes), {displayModeBar:false, responsive:true});
}

// ----- multiple contexts: three bordered metric groups, each a full-width figure
// with one POOLED row per context (offload + no-offload of a context share a row). -----
function drawGroups(c, groups, host){
  host.innerHTML = "";
  const nRows = groups.length;
  PANELS.forEach(panel => {
    const box = document.createElement("div");
    box.className = "dist-group";
    const hd = document.createElement("div");
    hd.className = "dist-group-hd";
    hd.textContent = panel.title;           // e.g. "cISL — cached input tokens"
    const plot = document.createElement("div");
    plot.id = "distgrp_" + panel.key;
    plot.style.width = "100%";
    box.appendChild(hd); box.appendChild(plot); host.appendChild(box);

    // shared x-range across ALL contexts so the workloads are directly comparable
    let allObjs = [];
    groups.forEach(g => { allObjs = allObjs.concat(grpDistObjs(g, c)); });
    const xmax = panelXmax(allObjs, panel);
    const bins = {start:0, end:panel.zoom ? panel.xmaxCap*2 : panel.xmax, size:panel.binsize};

    const traces = [], shapes = [], annotations = [], axes = {};
    const rowGap = 0.20, rowH = (1 - rowGap*(nRows-1))/nRows;
    groups.forEach((group, ri) => {
      const s = ri === 0 ? "" : String(ri+1);
      const xax = "x"+s, yax = "y"+s;
      const yTop = 1 - ri*(rowH+rowGap);
      const ydom = [yTop-rowH, yTop];
      const st = grpStat(group, c, panel.key);   // pooled over the context's selected runs
      const rowName = group.ctx + " workload" +
        (group.runs.length > 1 ? " (pooled offload + no-offload)" : "");

      traces.push({
        type:"histogram", x: st.samples, histnorm:"percent", xbins:bins,
        marker:{color:group.color, line:{width:0}}, opacity:0.82,
        name: rowName, showlegend:false, xaxis:xax, yaxis:yax,
        hovertemplate:rowName+"<br>%{x} tok: %{y:.1f}%<extra></extra>",
      });

      // this context's pooled P50/P90/P95 lines, staggered vertically so they don't collide
      PLINES.forEach((pl, pi) => {
        const v = st[pl.p];                 // pl.p is "p50" / "p90" / "p95"
        if(v==null || v>xmax) return;
        shapes.push({type:"line", x0:v, x1:v, xref:xax, y0:0, y1:1, yref:yax+" domain",
                     line:{color:pl.color, width:2, dash:pl.dash}});
        annotations.push({x:v, xref:xax, y:0.82 - pi*0.18, yref:yax+" domain",
                          xanchor:(v > 0.6*xmax ? "right" : "left"), yanchor:"top",
                          text:"<b>"+pl.name+" "+fmtK(v)+"</b>", showarrow:false,
                          font:{color:pl.color, size:11}});
      });

      // row subtitle: context label + its pooled stats (top-left, via domain coords)
      annotations.push({
        x:0.002, xref:xax+" domain", y:1.10, yref:yax+" domain", xanchor:"left", yanchor:"bottom",
        showarrow:false, align:"left", font:{size:12.5, color:group.color},
        text:"<b>"+rowName+"</b>   P50 "+fmtK(st.p50)+" · P90 "+fmtK(st.p90)+
             " · P95 "+fmtK(st.p95)+" · P99 "+fmtK(st.p99)+" · max "+fmtK(st.max),
      });

      // tail note for zoomed metrics (bottom-right, inside the row)
      if(panel.zoom){
        const beyond = st.samples.length
          ? 100*st.samples.filter(v=>v>xmax).length/st.samples.length : 0;
        annotations.push({x:0.99, xref:xax+" domain", y:0.06, yref:yax+" domain",
          xanchor:"right", yanchor:"bottom", showarrow:false, font:{size:9.5, color:"#7a5a1e"},
          text:"tail → max "+fmtK(st.max)+" ("+beyond.toFixed(1)+"% beyond view)"});
      }

      axes["xaxis"+s] = {domain:[0,1], anchor:yax, range:[0,xmax],
        title:{text:(ri===nRows-1 ? "tokens" : ""), font:{size:11}},
        showticklabels:true, zeroline:false};
      axes["yaxis"+s] = {domain:ydom, anchor:xax,
        title:{text:"% of requests", font:{size:10}}, rangemode:"tozero"};
    });

    const height = Math.round(nRows*200 + 52);
    Plotly.newPlot(plot.id, traces, Object.assign({
      template:"plotly_white", barmode:"overlay", height,
      margin:{l:58, r:18, t:44, b:40}, bargap:0.02, showlegend:false,
      shapes, annotations,
    }, axes), {displayModeBar:false, responsive:true});
  });
}

// ---------- fairness guard (computed live over the SELECTED subset) ----------
// sameWorkload := all selected runs share the same context token (same trace).
// A same-context pair (e.g. 1M offload vs 1M no-offload) is the intended
// apples-to-apples comparison; cross-context selections are intentionally different.
function computeFairness(sel){
  if(!sel.length) return {applicable:false, sameWorkload:false};
  const same = new Set(sel.map(r => r.context)).size === 1;
  // concurrency rungs common to every selected run
  let common = null;
  sel.forEach(r => {
    const ks = new Set(Object.keys(r.dist));
    common = common ? new Set([...common].filter(x => ks.has(x))) : ks;
  });
  common = common ? [...common].sort((a,b) => Number(a)-Number(b)) : [];
  if(!common.length) return {applicable:false, sameWorkload:same};
  const rung = common[Math.floor(common.length/2)];   // matches the default distConc pick
  const stats = sel.map(r => {
    const d = r.dist[rung];
    const cisl = d.cisl.samples, nisl = d.nisl.samples, n = cisl.length;
    let sc = 0, sn = 0;
    for(const v of cisl) sc += v;
    for(const v of nisl) sn += v;
    const mean = n ? (sc+sn)/n : 0;
    return {label:r.label, reqCount:n, meanInput:mean, meanInputK:mean/1000};
  });
  let unfair = false;
  if(same && stats.length >= 2){
    const means = stats.filter(s => s.reqCount>0).map(s => s.meanInput);
    const counts = stats.filter(s => s.reqCount>0).map(s => s.reqCount);
    if(means.length && Math.min(...means)>0 && Math.max(...means)/Math.min(...means) > 1.25) unfair = true;
    if(counts.length && Math.min(...counts)>0 && Math.max(...counts)/Math.min(...counts) > 2.0) unfair = true;
  }
  return {applicable:true, sameWorkload:same, unfair, rung:Number(rung), runs:stats};
}

function populateFairness(){
  const f = computeFairness(selectedRuns());
  const el = document.getElementById("fairnessBanner");
  el.className = ""; el.innerHTML = "";
  if(!f || !f.applicable) return;
  if(f.unfair){
    const parts = f.runs.map(r =>
      '<span class="fw-nums">'+r.label+" completed "+r.reqCount.toLocaleString()+
      " requests (mean input ~"+r.meanInputK.toFixed(0)+"k tokens)</span>");
    el.className = "fair-warn";
    el.innerHTML =
      '<div class="fw-icon">⚠️</div>'+
      '<div><b>PROVISIONAL — NOT an apples-to-apples comparison.</b> '+
      parts.join(" vs ")+'. The selected same-context runs replayed diverging slices of the trace, '+
      'so throughput and SLA differences may reflect the '+
      '<b>WORKLOAD difference, not the setting under test</b>. A full-trace-replay re-run is '+
      'required for a fair comparison.</div>';
  } else if(f.sameWorkload){
    el.className = "fair-ok";
    el.innerHTML = f.runs.length >= 2
      ? "✓ Fair comparison: all selected runs replayed the full trace at the same context "+
        "(matched workload — the offload difference is the only variable)."
      : "✓ Single run selected (full-trace replay).";
  }
}

// ---------- inline chart caption / legend (2x2 encoding + red = SLA fail) ----------
// dotFor renders the marker style: filled disc (offload) vs hollow ring (no-offload).
function dotStyle(color, open){
  return open
    ? 'background:#fff;border:2px solid '+color
    : 'background:'+color+';border:1px solid rgba(0,0,0,0.18)';
}
function populateCaption(){
  // distinct context swatches (from the selected runs), then the offload style key.
  const seen = new Set(), ctxParts = [];
  selectedRuns().forEach(r => {
    if(seen.has(r.context)) return;
    seen.add(r.context);
    ctxParts.push('<span class="lg"><span class="dot" style="background:'+r.color+'"></span>'+
                  r.context+'</span>');
  });
  const parts = ['<span class="lg" style="font-weight:700">color = context:</span>'].concat(ctxParts);
  parts.push('<span class="lg" style="font-weight:700">style = offload:</span>');
  parts.push('<span class="lg"><span class="dot" style="'+dotStyle("#555",false)+'"></span>'+
             'solid + filled = offload</span>');
  parts.push('<span class="lg"><span class="dot" style="'+dotStyle("#555",true)+'"></span>'+
             'dashed + open = no-offload</span>');
  parts.push('<span class="lg"><span class="dot" style="background:'+FAIL+'"></span>'+
             '<b style="color:'+FAIL+'">red = fails selected SLA</b></span>');
  parts.push('<span class="lg"><span class="star">★</span> highest passing concurrency</span>');
  parts.push('<div class="cap-txt"><b>Encoding:</b> color = context (256K / 1M); '+
             'solid line + filled marker = offload, dashed line + open marker = no-offload; '+
             'a point turns <b style="color:'+FAIL+'">red</b> when it fails <i>either</i> the selected TTFT '+
             '<i>or</i> TPOT limit (it must pass <i>both</i> to count as passing) '+
             '(marker fill overridden, white outline kept); ★ = highest-concurrency point that still passes.</div>');
  parts.push('<div class="cap-txt" style="color:'+WARN_RED+'"><b>⚠ N% incomplete</b> = requests '+
             'not completed within the configured client timeout; these points are '+
             '<b>not equal-work</b>.</div>');
  document.getElementById("chartCap").innerHTML = parts.join("");
}

// ---------- concurrency dropdown (selection-aware; default = mid rung common to all selected) ----------
function updateDistConcOptions(){
  const sel = selectedRuns();
  const set = new Set();
  sel.forEach(r => Object.keys(r.dist).forEach(c => set.add(Number(c))));
  const all = [...set].sort((a,b)=>a-b);
  const selEl = document.getElementById("distConc");
  if(!all.length){ selEl.innerHTML = ""; return; }
  const common = all.filter(c => sel.every(r => r.dist[String(c)]));
  const cur = Number(selEl.value);
  const curOk = Number.isFinite(cur) && sel.every(r => r.dist[String(cur)]);
  const pick = curOk ? cur
    : (common.length ? common[Math.floor(common.length/2)]
                     : all[Math.floor(all.length/2)]);
  selEl.innerHTML = all.map(c => {
    const nrun = sel.filter(r=>r.dist[String(c)]).length;
    const tag = nrun < sel.length ? " ("+nrun+"/"+sel.length+" selected)" : "";
    return '<option value="'+c+'"'+(c===pick?" selected":"")+'>c='+c+tag+'</option>';
  }).join("");
}

// ---------- run-selection UI (checkboxes + presets) ----------
const PRESETS = [
  {name:"1M: offload vs no-offload",   pred:r => r.context === "1M"},
  {name:"256K: offload vs no-offload", pred:r => r.context === "256K"},
  {name:"No-offload: 256K vs 1M",      pred:r => r.offload === "no-offload"},
  {name:"Offload: 256K vs 1M",         pred:r => r.offload === "offload"},
  {name:"All four",                    pred:r => true},
];

// everything below the selection reacts here — no page reload.
function onSelectionChange(){
  applyScatterVisibility();
  recolorScatter();
  updateScatterText();
  updateIncompleteAnnotations();
  updateScaling();
  populateFairness();
  populateCaption();
  updateDistConcOptions();
  drawDist();
}

function buildRunSelector(){
  const checks = document.getElementById("runChecks");
  checks.innerHTML = "";
  RUNS.forEach((run, i) => {
    const lab = document.createElement("label");
    lab.className = "runchk";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = SEL[i]; cb.dataset.idx = i;
    cb.addEventListener("change", () => { SEL[i] = cb.checked; onSelectionChange(); });
    const dot = document.createElement("span");
    dot.className = "rc-dot";
    dot.style.cssText = dotStyle(run.color, isOpen(run));
    lab.appendChild(cb); lab.appendChild(dot);
    lab.appendChild(document.createTextNode(run.label));
    checks.appendChild(lab);
  });
  const host = document.getElementById("runPresets");
  host.innerHTML = "";
  PRESETS.forEach(preset => {
    const match = RUNS.map(r => !!preset.pred(r));
    if(!match.some(Boolean)) return;               // skip presets that match no run
    const btn = document.createElement("button");
    btn.type = "button"; btn.className = "preset-btn"; btn.textContent = preset.name;
    btn.addEventListener("click", () => {
      RUNS.forEach((_, i) => { SEL[i] = match[i]; });
      // reflect into checkboxes
      checks.querySelectorAll("input[type=checkbox]").forEach(cb => {
        cb.checked = SEL[Number(cb.dataset.idx)];
      });
      onSelectionChange();
    });
    host.appendChild(btn);
  });
}

// ---------- wire up ----------
buildScatter();
buildRunSelector();
populateFairness();
populateCaption();
updateDistConcOptions();
for(const id of ["ttftPct","tpotPct","ttftLimit","tpotLimit","targetConc"]){
  const h = () => { recolorScatter(); updateScaling(); updateScatterText(); };
  document.getElementById(id).addEventListener("input", h);
  document.getElementById(id).addEventListener("change", h);
}
document.getElementById("showDetail").addEventListener("change", updateScatterText);
document.getElementById("distConc").addEventListener("change", drawDist);
applyScatterVisibility();
recolorScatter();
updateIncompleteAnnotations();
updateScaling();
drawDist();
</script>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

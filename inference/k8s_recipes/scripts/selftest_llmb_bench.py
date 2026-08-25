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

"""selftest_llmb_bench.py — unit tests for the Phase-2 M1 harness carve-out (no cluster)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # inference/k8s_recipes
sys.path.insert(0, str(ROOT))

from llmb_bench import aiperf, gate, metrics, plan, smoke  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# --- plan: fixed + adaptive (delegates to sweep_planner) ---
check(
    "plan.fixed_rungs: parses the space-separated list",
    plan.fixed_rungs("32 64 128 256") == [32, 64, 128, 256],
)
check("plan.fixed_rungs: empty → []", plan.fixed_rungs("   ") == [])
_g = plan.adaptive_grid(16, 2.0, 8, 256)
check(
    "plan.adaptive_grid: sorted, non-empty, within [lo,hi], anchors start+bounds",
    _g == sorted(_g) and _g and _g[0] == 8 and _g[-1] == 256 and 16 in _g and all(8 <= x <= 256 for x in _g),
    str(_g),
)

# --- metrics: sum counters (across labels), read gauges, exclude prompt tokens ---
_blob = (
    'vllm:generation_tokens_total{model_name="m"} 1000.0\n'
    'vllm:generation_tokens_total{model_name="m2"} 234\n'
    "vllm:prompt_tokens_total 99999\n"
    "# HELP vllm:num_requests_running gauge\n"
    "vllm:num_requests_running 3\n"
    'vllm:num_requests_waiting{x="y"} 5\n'
)
_m = metrics.parse_metrics(_blob)
check(
    "metrics.parse_metrics: sums generation_tokens across labels (1234)",
    _m["generation_tokens"] == 1234,
)
check(
    "metrics.parse_metrics: prompt_tokens read separately (not folded into generation)",
    _m["prompt_tokens"] == 99999,
)
check("metrics.parse_metrics: running gauge = 3", _m["num_requests_running"] == 3.0)
check(
    "metrics.parse_metrics: waiting gauge (labelled) = 5",
    _m["num_requests_waiting"] == 5.0,
)
check(
    "metrics.parse_metrics: missing metric → 0",
    metrics.parse_metrics("")["generation_tokens"] == 0,
)

# --- smoke: the pre-sweep chat-completion gate (mirrors bench-job.yaml.j2 exactly) ---
check(
    "smoke.validate_smoke: HTTP 200 + choices → ok",
    smoke.validate_smoke(200, '{"choices":[{"message":{"content":"ok"}}]}')[0] is True,
)
check(
    "smoke.validate_smoke: non-2xx → fail",
    smoke.validate_smoke(503, '{"choices":[{}]}')[0] is False,
)
check(
    "smoke.validate_smoke: API error field → fail",
    smoke.validate_smoke(200, '{"error":"bad model"}')[0] is False,
)
check(
    "smoke.validate_smoke: empty choices → fail",
    smoke.validate_smoke(200, '{"choices":[]}')[0] is False,
)
check(
    "smoke.validate_smoke: non-JSON body → fail",
    smoke.validate_smoke(200, "not json")[0] is False,
)
check(
    "smoke.validate_smoke: non-numeric status → fail",
    smoke.validate_smoke("", '{"choices":[{}]}')[0] is False,
)

# --- aiperf: argv builder + the workload-cap PRIORITY (request-count > duration > full-trace) ---
_base = {
    "url": "http://s:8000",
    "model": "m",
    "tokenizer": "t",
    "endpoint_type": "chat",
    "dataset_type": "mooncake_trace",
    "dataset_path": "/d.jsonl",
    "num_profile_runs": 1,
    "out": "/o",
}
_a = aiperf.build_aiperf_args(_base, 128)
check(
    "aiperf.build: base args present + concurrency threaded",
    _a[0] == "profile"
    and "--concurrency" in _a
    and _a[_a.index("--concurrency") + 1] == "128"
    and "--streaming" in _a
    and _a[_a.index("--artifact-dir") + 1] == "/o/logs/aiperf",
    " ".join(_a),
)
check(
    "aiperf.build: full-trace replay when no cap set → NO --request-count / --benchmark-duration",
    "--request-count" not in _a and "--benchmark-duration" not in _a,
)
_rc = aiperf.build_aiperf_args({**_base, "request_count_multiplier": "4", "bench_duration": "300"}, 64)
check(
    "aiperf.build: request-count multiplier BEATS duration (4×64=256)",
    "--request-count" in _rc and _rc[_rc.index("--request-count") + 1] == "256" and "--benchmark-duration" not in _rc,
)
_bd = aiperf.build_aiperf_args({**_base, "bench_duration": "300", "concurrency_ramp_duration": "30"}, 64)
check(
    "aiperf.build: duration path (no request-count) adds --benchmark-duration + ramp",
    _bd[_bd.index("--benchmark-duration") + 1] == "300" and "--concurrency-ramp-duration" in _bd,
)
_wm = aiperf.build_aiperf_args({**_base, "warmup_request_multiplier": "2"}, 100)
check(
    "aiperf.build: warmup-request-count = multiplier × concurrency (2×100=200)",
    _wm[_wm.index("--warmup-request-count") + 1] == "200",
)
check(
    "aiperf.cfg_from_env: maps env vars, drops empties",
    aiperf.cfg_from_env({"SERVER_URL": "u", "AIPERF_MODEL_ID": "m", "CACHE_BUST": ""}) == {"url": "u", "model": "m"},
)
# synthetic (mode: synthetic): --isl/--osl replace --custom-dataset-type/--input-file; stddev + seed threaded.
_syn = aiperf.build_aiperf_args(
    {
        "url": "http://s:8000",
        "model": "m",
        "tokenizer": "t",
        "endpoint_type": "chat",
        "out": "/o",
        "syn_isl": "14746",
        "syn_isl_stddev": "946",
        "syn_osl": "461",
        "syn_osl_stddev": "30",
        "syn_seed": "42",
        "request_count_multiplier": "10",
    },
    240,
)
check(
    "aiperf.build: synthetic uses --isl/--osl (+stddev,+seed), NO --input-file / --custom-dataset-type",
    "--input-file" not in _syn
    and "--custom-dataset-type" not in _syn
    and _syn[_syn.index("--isl") + 1] == "14746"
    and _syn[_syn.index("--osl") + 1] == "461"
    and _syn[_syn.index("--isl-stddev") + 1] == "946"
    and _syn[_syn.index("--random-seed") + 1] == "42",
    " ".join(_syn),
)
check(
    "aiperf.build: synthetic + request-count multiplier → --request-count 2400 (10×240)",
    "--request-count" in _syn and _syn[_syn.index("--request-count") + 1] == "2400",
)
# adaptive depth on a TRACE cell (bench.depth): no_fixed_schedule ⇒ --no-fixed-schedule so aiperf honors the
# concurrency-scaled --request-count instead of auto-replaying the timestamped mooncake trace once.
_nfs = aiperf.build_aiperf_args({**_base, "request_count_multiplier": "4", "no_fixed_schedule": "true"}, 256)
check(
    "aiperf.build: trace + no_fixed_schedule → --no-fixed-schedule AND --request-count depth×N (4×256=1024)",
    "--no-fixed-schedule" in _nfs and "--input-file" in _nfs and _nfs[_nfs.index("--request-count") + 1] == "1024",
    " ".join(_nfs),
)
check(
    "aiperf.build: trace WITHOUT no_fixed_schedule → NO --no-fixed-schedule (legacy fixed-trace pass)",
    "--no-fixed-schedule" not in _a,
)
check(
    "aiperf.build: synthetic ignores no_fixed_schedule (no trace to auto-schedule)",
    "--no-fixed-schedule"
    not in aiperf.build_aiperf_args(
        {
            **_base,
            "syn_isl": "1",
            "syn_osl": "1",
            "dataset_path": "",
            "no_fixed_schedule": "true",
        },
        8,
    ),
)
# dataset_path present ⇒ trace path even if syn_* leaked in (mutual exclusion; trace wins).
_mix = aiperf.build_aiperf_args(
    {
        "url": "u",
        "model": "m",
        "tokenizer": "t",
        "endpoint_type": "chat",
        "out": "/o",
        "dataset_type": "mooncake_trace",
        "dataset_path": "/d.jsonl",
        "syn_isl": "100",
        "syn_osl": "50",
    },
    8,
)
check(
    "aiperf.build: dataset_path present ⇒ trace path wins over syn_* (--input-file, no --isl)",
    "--input-file" in _mix and "--isl" not in _mix,
)

# --- bench-job.yaml.j2 TEMPLATE render: the synthetic Jinja-gate (mode: synthetic → aiperf --isl/--osl,
#     no trace file), and its byte-neutral twin for trace cells. Renders the SHARED aiperf bench template
#     the way scripts/_render.py does (same StrictUndefined + trim/lstrip block env). ---
import yaml as _yaml  # noqa: E402
from jinja2 import Environment as _Env  # noqa: E402
from jinja2 import FileSystemLoader as _FSL  # noqa: E402,N814
from jinja2 import StrictUndefined as _SU  # noqa: E402,N814

_TPL_DIR = ROOT / "serving" / "aiperf" / "templates"


def _render_bench(recipe_path: Path, strip_depth: bool = False, extra_inputs: str | None = None) -> str:
    recipe = _yaml.safe_load(recipe_path.read_text()) or {}
    if extra_inputs is not None:
        recipe.setdefault("bench", {})["extra_inputs"] = extra_inputs
    if strip_depth:
        # Force the depth-LESS legacy fixed-trace path even on a cell that pins bench.depth (the surviving
        # KVBM trace cell pins depth=8) — so the byte-neutral "no depth" assertion still exercises that path.
        (recipe.get("bench") or {}).pop("depth", None)
    ctx = dict(recipe)
    prov = (recipe.get("envelope") or {}).get("provenance") or {}
    ctx.setdefault("image_ref", prov.get("image_ref") or prov.get("image_digest") or "")
    ctx.setdefault("model", (recipe.get("envelope") or {}).get("model", ""))
    jenv = _Env(
        loader=_FSL(str(_TPL_DIR)),
        undefined=_SU,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return jenv.get_template("bench-job.yaml.j2").render(**ctx)


# Fixtures migrated off the pruned glm5-16k512 / glm5-9600 cells to the surviving north-star cells:
#   _GLM_SYN → the retained GLM-5 B200 disagg cell (mode: synthetic, isl/osl 922, seed 42, num_entries 25760)
#   _TRACE   → the kept Qwen3-KVBM gb300 cell (mode: mooncake-trace; pins bench.depth=8)
_GLM_SYN = ROOT / (
    "recipes/llm-perf/Glm5/B200_k8s/Disagg/sglang_dynamo/1k_1k/"
    "glm5-fp8-b200-sglang-dynamo14-1k1k-hightpt-c2576-1p1d/recipe.yaml"
)
_TRACE = ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto/recipe.yaml"
_syn_bench = _render_bench(_GLM_SYN)
check(
    "bench-tpl(synthetic): aiperf argv uses --isl/--osl, drops --input-file/--custom-dataset-type",
    '--isl "$SYN_ISL"' in _syn_bench
    and '--osl "$SYN_OSL"' in _syn_bench
    and "--input-file" not in _syn_bench
    and "--custom-dataset-type" not in _syn_bench,
)
check(
    "bench-tpl(synthetic): stddev + random-seed argv threaded from bench.synthetic",
    '--isl-stddev "$SYN_ISL_STDDEV"' in _syn_bench
    and '--osl-stddev "$SYN_OSL_STDDEV"' in _syn_bench
    and '--random-seed "$SYN_SEED"' in _syn_bench,
)
check(
    "bench-tpl(synthetic): SYN_* env baked from recipe (isl=922, osl=922, seed=42)",
    'name: SYN_ISL, value: "922"' in _syn_bench
    and 'name: SYN_OSL, value: "922"' in _syn_bench
    and 'name: SYN_SEED, value: "42"' in _syn_bench,
)
check(
    "bench-tpl(synthetic): NO trace dataset — DATASET_PATH empty + no /model-cache dataset mount, "
    "no dataset-existence FATAL check",
    'name: DATASET_PATH, value: "" }' in _syn_bench
    and "/model-cache/datasets" not in _syn_bench
    and "FATAL: dataset path does not exist" not in _syn_bench
    and "Synthetic load (mode: synthetic)" in _syn_bench,
)
check(
    "bench-tpl(synthetic): request-count multiplier + num-dataset-entries baked (10 / 25760)",
    'name: REQUEST_COUNT_MULTIPLIER, value: "10" }' in _syn_bench
    and 'name: NUM_DATASET_ENTRIES, value: "25760" }' in _syn_bench,
)
_multi_bench = _render_bench(_GLM_SYN, extra_inputs="min_tokens:922 ignore_eos:true")
check(
    "bench-tpl(extra_inputs): multiple values use ONE --extra-inputs option (pinned AIPerf consume_multiple contract)",
    'set -- "$@" --extra-inputs; for _ei in $AIPERF_EXTRA_INPUTS' in _multi_bench
    and 'set -- "$@" "$_ei"' in _multi_bench
    and 'set -- "$@" --extra-inputs "$_ei"' not in _multi_bench,
)
_trace_bench = _render_bench(_TRACE, strip_depth=True)
check(
    "bench-tpl(trace): the gate is byte-neutral - trace cell STILL renders --input-file/--custom-dataset-type, "
    "no synthetic --isl leaks in",
    "--input-file" in _trace_bench
    and '--custom-dataset-type "$DATASET_TYPE"' in _trace_bench
    and '--isl "$SYN_ISL"' not in _trace_bench
    and "Synthetic load (mode: synthetic)" not in _trace_bench,
)
# bench.depth adaptive-depth fix: opt-in trace cell renders REQUEST_COUNT_MULTIPLIER + --no-fixed-schedule;
# a depth-less trace cell is byte-identical (no NO_FIXED_SCHEDULE line / no --no-fixed-schedule argv).
check(
    "bench-tpl(trace, no depth): legacy fixed-trace pass - REQUEST_COUNT_MULTIPLIER empty, "
    "no NO_FIXED_SCHEDULE env, no --no-fixed-schedule argv",
    'name: REQUEST_COUNT_MULTIPLIER, value: "" }' in _trace_bench
    and "NO_FIXED_SCHEDULE" not in _trace_bench
    and "--no-fixed-schedule" not in _trace_bench,
)


def _render_bench_with_depth(recipe_path: Path, depth: int) -> str:
    recipe = _yaml.safe_load(recipe_path.read_text()) or {}
    recipe.setdefault("bench", {})["depth"] = depth
    ctx = dict(recipe)
    prov = (recipe.get("envelope") or {}).get("provenance") or {}
    ctx.setdefault("image_ref", prov.get("image_ref") or prov.get("image_digest") or "")
    ctx.setdefault("model", (recipe.get("envelope") or {}).get("model", ""))
    jenv = _Env(
        loader=_FSL(str(_TPL_DIR)),
        undefined=_SU,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return jenv.get_template("bench-job.yaml.j2").render(**ctx)


_trace_depth = _render_bench_with_depth(_TRACE, 4)
check(
    "bench-tpl(trace, depth=4): REQUEST_COUNT_MULTIPLIER baked to depth (4) → request-count = 4×concurrency",
    'name: REQUEST_COUNT_MULTIPLIER, value: "4" }' in _trace_depth,
)
check(
    "bench-tpl(trace, depth=4): NO_FIXED_SCHEDULE env true + --no-fixed-schedule argv (steady-state, "
    "cycles the trace pool instead of one fixed t=0 pass)",
    'name: NO_FIXED_SCHEDULE, value: "true" }' in _trace_depth and "--no-fixed-schedule \\" in _trace_depth,
)


# --- Source-port tuning: the sysctl settings are
#     OPT-IN and byte-identical when absent; sticky-user-sessions is PRESERVED (not switched to pooled). ---
def _render_bench_with_fields(recipe_path: Path, **bench_overrides) -> str:
    recipe = _yaml.safe_load(recipe_path.read_text()) or {}
    recipe.setdefault("bench", {}).update(bench_overrides)
    ctx = dict(recipe)
    prov = (recipe.get("envelope") or {}).get("provenance") or {}
    ctx.setdefault("image_ref", prov.get("image_ref") or prov.get("image_digest") or "")
    ctx.setdefault("model", (recipe.get("envelope") or {}).get("model", ""))
    jenv = _Env(
        loader=_FSL(str(_TPL_DIR)),
        undefined=_SU,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return jenv.get_template("bench-job.yaml.j2").render(**ctx)


# Recipe-scoped client upgrades are immutable and can drop flags removed by the new CLI without
# changing unrelated published cells. AIPerf 0.11 includes the upstream NaN-metric fix.
_AIPERF_011_REF = "git+https://github.com/ai-dynamo/aiperf.git@38687855e98044fcf12ee48c6794128f10b6780b"
_aiperf_011_bench = _render_bench_with_fields(_GLM_SYN, aiperf_ref=_AIPERF_011_REF, aiperf_unsafe_override=False)
_legacy_aiperf_bench = _render_bench_with_fields(_TRACE)
_no_server_metrics_bench = _render_bench_with_fields(
    _GLM_SYN,
    aiperf_ref=_AIPERF_011_REF,
    aiperf_unsafe_override=False,
    aiperf_server_metrics=False,
)
check(
    "bench-tpl(aiperf 0.11 opt-in): exact immutable pin rendered and legacy option omitted",
    _AIPERF_011_REF in _aiperf_011_bench and "--unsafe-override" not in _aiperf_011_bench,
)
check(
    "bench-tpl(aiperf legacy default): unrelated published cell keeps old pin and legacy option",
    "fef78a96ee30df4c135a9e5691b48e50ba91133d" in _legacy_aiperf_bench and "--unsafe-override" in _legacy_aiperf_bench,
)
check(
    "bench-tpl(server metrics opt-out): emits the supported AIPerf flag only when recipe requests it",
    "--no-server-metrics" in _no_server_metrics_bench and "--no-server-metrics" not in _legacy_aiperf_bench,
)

# absent → BYTE-IDENTICAL to the committed render (no securityContext.sysctls, no port-tuning initContainer).
# Compare against the AS-IS render of _TRACE (depth preserved) so the byte-identity isolates the port-tuning
# block — _trace_bench above is depth-stripped for the legacy-trace assertions, an orthogonal concern.
_trace_asis = _render_bench(_TRACE)
check(
    "bench-tpl(port-tuning absent): renders byte-identically — no sysctls / no port-tuning initContainer, "
    "zero re-fingerprint for published cells",
    "securityContext:\n        sysctls:" not in _trace_asis
    and "name: port-tuning" not in _trace_asis
    and _render_bench_with_fields(_TRACE) == _trace_asis,
)

# initcontainer mode (preferred) → privileged NET_ADMIN initContainer writing the sysctls directly via
# /proc/sys in the shared netns. The /proc-write approach (from glm5-gb300-live) is image-agnostic —
# python:3.12-slim ships no `sysctl` binary (would exit 127) — and supersedes the earlier `sysctl -w`.
_ppt_init = _render_bench_with_fields(_TRACE, bench_pod_port_tuning="initcontainer")
check(
    "bench-tpl(port-tuning=initcontainer): privileged NET_ADMIN initContainer sets the sysctls via "
    "/proc/sys writes in the shared netns (portable, no kubelet allowlist, image-agnostic)",
    "initContainers:" in _ppt_init
    and "name: port-tuning" in _ppt_init
    and "privileged: true" in _ppt_init
    and 'add: ["NET_ADMIN"]' in _ppt_init
    and '"1024 65535" > /proc/sys/net/ipv4/ip_local_port_range' in _ppt_init
    and "1           > /proc/sys/net/ipv4/tcp_tw_reuse" in _ppt_init
    and "securityContext:\n        sysctls:" not in _ppt_init,
)

# sysctls mode (alternative) → pod securityContext.sysctls widening ip_local_port_range + tcp_tw_reuse.
_ppt_sysctls = _render_bench_with_fields(_TRACE, bench_pod_port_tuning="sysctls")
check(
    "bench-tpl(port-tuning=sysctls): pod securityContext.sysctls widens ip_local_port_range + tcp_tw_reuse "
    "(unsafe sysctls, kubelet allowlist required)",
    "securityContext:\n        sysctls:" in _ppt_sysctls
    and 'name: net.ipv4.ip_local_port_range, value: "1024 65535"' in _ppt_sysctls
    and 'name: net.ipv4.tcp_tw_reuse,        value: "1"' in _ppt_sysctls
    and "name: port-tuning" not in _ppt_sysctls,
)

# sticky-user-sessions is PRESERVED — the fix does NOT touch connection_reuse (unset still defaults to sticky).
check(
    "bench-tpl(connection_reuse preserved): port-tuning does not alter CONNECTION_REUSE_STRATEGY — "
    "still sticky-user-sessions",
    'name: CONNECTION_REUSE_STRATEGY, value: "sticky-user-sessions" }' in _trace_bench
    and 'name: CONNECTION_REUSE_STRATEGY, value: "sticky-user-sessions" }' in _ppt_init,
)

# --- gate: the per-rung SLA decision (THE pass/fail that max_concurrency_at_sla hangs on) ---
_ex_pass = {
    "time_to_first_token": {"p50": 8000.0},
    "inter_token_latency": {"p50": 80.0},
}
_gp = gate.evaluate_rung(_ex_pass, ttft_limit_ms=10000, tpot_limit_ms=100, concurrency=128)
check(
    "gate.evaluate_rung: within both limits → passed, no breaches, gating_ratio=max(0.8,0.8)",
    _gp["passed"] and _gp["breaches"] == [] and abs(_gp["gating_ratio"] - 0.8) < 1e-9,
)
_ex_tpot = {
    "time_to_first_token": {"p50": 8000.0},
    "inter_token_latency": {"p50": 120.0},
}
_gt = gate.evaluate_rung(_ex_tpot, ttft_limit_ms=10000, tpot_limit_ms=100, concurrency=192)
check(
    "gate.evaluate_rung: TPOT over limit → not passed, breaches=[tpot], threshold_exceeded",
    (not _gt["passed"]) and _gt["breaches"] == ["tpot"] and _gt["stop_reason"] == "threshold_exceeded",
)
_gr = gate.evaluate_rung(_ex_pass, ttft_limit_ms=10000, tpot_limit_ms=100, aiperf_rc=1)
check(
    "gate.evaluate_rung: aiperf non-zero exit → NOT a pass even if metrics ok (unparseable_metrics)",
    (not _gr["passed"]) and _gr["stop_reason"] == "unparseable_metrics",
)
_gm = gate.evaluate_rung({"time_to_first_token": {"p50": 8000.0}}, ttft_limit_ms=10000, tpot_limit_ms=100)
check(
    "gate.evaluate_rung: missing a metric → parse_ok False, not passed",
    (not _gm["parse_ok"]) and (not _gm["passed"]) and "parse_error" in _gm,
)
_gs = gate.evaluate_rung(
    {"time_to_first_token": {"p90": 5.0}, "inter_token_latency": {"p90": 5.0}},
    stat="p90",
    ttft_limit_ms=10000,
    tpot_limit_ms=100,
)
check(
    "gate.evaluate_rung: honors a non-p50 stop-stat (p90)",
    _gs["passed"] and _gs["stop_stat"] == "p90",
)

# --- CLI wiring (python -m llmb_bench) ---
_r = subprocess.run(
    [sys.executable, "-m", "llmb_bench", "plan", "fixed", "32 64"],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
)
check(
    "cli: `plan fixed` prints the rung list",
    _r.stdout.strip() == "32 64",
    _r.stdout + _r.stderr[:80],
)
_r2 = subprocess.run(
    [sys.executable, "-m", "llmb_bench", "metrics"],
    cwd=str(ROOT),
    input=_blob,
    capture_output=True,
    text=True,
)
check(
    "cli: `metrics` on stdin → JSON with generation_tokens=1234",
    '"generation_tokens": 1234' in _r2.stdout,
    _r2.stdout[:120] + _r2.stderr[:80],
)
_r3 = subprocess.run(
    [sys.executable, "-m", "llmb_bench", "smoke", "200"],
    cwd=str(ROOT),
    input='{"choices":[{}]}',
    capture_output=True,
    text=True,
)
check(
    "cli: `smoke 200` with valid body → exit 0",
    _r3.returncode == 0,
    _r3.stdout + _r3.stderr[:80],
)
_r4 = subprocess.run(
    [sys.executable, "-m", "llmb_bench", "smoke", "500"],
    cwd=str(ROOT),
    input='{"choices":[{}]}',
    capture_output=True,
    text=True,
)
check(
    "cli: `smoke 500` → exit 67 (SMOKE_FAIL_EXIT, unchanged from the shell)",
    _r4.returncode == 67,
)
_r5 = subprocess.run(
    [sys.executable, "-m", "llmb_bench", "aiperf", "128"],
    cwd=str(ROOT),
    env={
        **__import__("os").environ,
        "SERVER_URL": "u",
        "AIPERF_MODEL_ID": "m",
        "REQUEST_COUNT_MULTIPLIER": "4",
    },
    capture_output=True,
    text=True,
)
check(
    "cli: `aiperf 128` reads env → argv with --request-count 512 (4×128)",
    "--request-count 512" in _r5.stdout,
    _r5.stdout[:160] + _r5.stderr[:80],
)
_r6 = subprocess.run(
    [sys.executable, "-m", "llmb_bench", "gate"],
    cwd=str(ROOT),
    input='{"time_to_first_token":{"p50":8000},"inter_token_latency":{"p50":80}}',
    env={
        **__import__("os").environ,
        "TTFT_LIMIT_MS": "10000",
        "TPOT_LIMIT_MS": "100",
        "STEP_CONCURRENCY": "128",
    },
    capture_output=True,
    text=True,
)
check(
    "cli: `gate` on an export → JSON verdict with passed=true",
    '"passed": true' in _r6.stdout,
    _r6.stdout[:160] + _r6.stderr[:80],
)

print()
if fails:
    print(f"selftest_llmb_bench: {len(fails)} FAILED: {', '.join(fails)}")
    sys.exit(1)
print("selftest_llmb_bench: ALL PASS ✓")
sys.exit(0)

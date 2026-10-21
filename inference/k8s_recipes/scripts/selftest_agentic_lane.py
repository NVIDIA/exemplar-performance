#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""selftest_agentic_lane.py — offline guards for the agentic (TaskTrace) lane: the aggregator over the Job's
c<K>/<i>/summary.json layout, the replay-fidelity goal handler (metric, round-trip, tables, baseline), the
benchmark identity / recipe fingerprint of the two shipped cells, the run-result verdict, and the export lane.
No cluster, no tasktrace binary: fixtures are Slurm-shaped summary.json files the shared recipe module judges."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/agentic"))
import aggregate_metrics as agg  # noqa: E402
import benchmark_id as bid  # noqa: E402
import export_dataset as exp  # noqa: E402
import goal_handlers as gh  # noqa: E402
import lane  # noqa: E402
import recipe_hash as rh  # noqa: E402
import run_result  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    print(("PASS " if ok else "FAIL ") + label + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS += 1


REPLAY_CELL = ROOT / "recipes/agentic/tasktrace-replay/swebench-verified-gb300-cpu"
CAPTURE_CELL = ROOT / "recipes/agentic/tasktrace-capture-replay/swebench-verified-gb300-cpu"


def replay_summary(ratio=1.02, fidelity="PASS", exit_status=0, cpu=True, regime="exclusive-cores"):
    s = {
        "schema_version": 1,
        "workload": "tasktrace_replay",
        "trace_id": "pytest-dev__pytest-7236",
        "pacing": "recorded",
        "allocated_cpus": 4,
        "cpuset": "0,1,2,3",
        "tasktrace_exit_status": exit_status,
        "fidelity": fidelity,
        "replay": {
            "actions_recorded": 120,
            "actions_executed": 118,
            "actions_skipped_not_executed": 2,
            "source_outcome_mismatch_count": 0 if fidelity == "PASS" else 3,
            "wall_time_sec": 812.5,
        },
        "comparison": {
            "tool_busy_live_sec": 400.0,
            "tool_busy_replay_sec": 400.0 * ratio,
            "replay_to_live_tool_ratio": ratio,
            "actions_compared": 118,
        },
        "pipeline_status": "valid",
        "diagnostics": [] if fidelity == "PASS" else ["3 source-outcome mismatches"],
    }
    if cpu:
        s["cpu"] = {
            "runtime": "current",
            "method": "current-container-cgroup-v2-cpu.stat",
            "wall_sec": 812.5,
            "usage_sec": 1400.25,
            "average_logical_cores_used": 1.723,
            "allocated_logical_cpus": 4.0,
            "utilization_percent_of_allocation": 43.08,
            "cpu_max": "800000 100000",
            "cgroup_cpuset": "4-11",
            "throttling": {"throttled_sec": 0.0},
            "valid_for_average_cpu_reporting": True,
        }
    return s, {"regime": regime, "pinned_cpuset": "0,1,2,3", "cpuset_cpus_effective": "4-11"}


def capture_summary(match=True):
    s, regime = replay_summary(ratio=1.01)
    s["workload"] = "tasktrace_capture_replay"
    s["replay"] = {"ran": True, "tasktrace_exit_status": 0, "summary": s.pop("replay")}
    s.pop("tasktrace_exit_status")
    s["capture"] = {"tasktrace_capture_exit_status": 0, "stage_exit_status": 0, "reward": 1.0, "llm_calls": 42}
    s["image_identity_check"] = {
        "capture_image_sha256": "945da72e",
        "replay_image_sha256": "945da72e" if match else None,
        "match": match,
        "source": "current-container: image digest supplied by the launcher",
    }
    return s, regime


def write_run(run_dir: Path, containers):
    """containers: {(rung, index): (summary, regime)} → the Job's artifact layout."""
    for (rung, index), (summary, regime) in containers.items():
        d = run_dir / f"c{rung}" / str(index)
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(summary))
        (d / "cpu_regime.json").write_text(json.dumps(regime))


tmp = Path(tempfile.mkdtemp(prefix="agentic-lane-"))
try:
    # ── lane routing ────────────────────────────────────────────────────────────────────────────────────
    for mode in ("tasktrace-replay", "tasktrace-capture-replay"):
        ln = lane.resolve_lane("agentic", mode)
        check(
            f"lane agentic/{mode}: stage-tasktrace.sh + tasktrace.sh, kind tt, server none",
            ln == {"stage": "stage-tasktrace.sh", "bench": "tasktrace.sh", "needs": "", "kind": "tt", "server": "none"},
            str(ln),
        )

    # ── aggregator: one row per (rung, container), verdict = the shared Slurm check ───────────────────
    run = tmp / "run-a"
    write_run(
        run,
        {(1, 0): replay_summary(1.02), (2, 0): replay_summary(1.03), (2, 1): replay_summary(1.02, fidelity="FAIL")},
    )
    rows = agg.aggregate(run)
    check(
        "aggregator finds every c<K>/<i>/summary.json in rung/container order",
        [(r["rung"], r["container"]) for r in rows] == [(1, 0), (2, 0), (2, 1)],
    )
    check(
        "aggregator verdict = shared check (fidelity FAIL → FAIL, others PASS)",
        [r["verdict"] for r in rows] == ["PASS", "PASS", "FAIL"],
    )
    check(
        "aggregator lifts the CPU block + regime",
        rows[0]["cpu_usage_sec"] == 1400.25
        and rows[0]["cpu_regime"] == "exclusive-cores"
        and rows[0]["cpu_valid_for_average_cpu_reporting"] is True,
    )
    check(
        "aggregator reports the replay's tool time, its ratio and the live tool times",
        rows[0]["replay_tool_time_ratio"] == 1.02
        and rows[0]["live_tool_busy_sec"] == 400.0
        and abs(rows[0]["replay_tool_busy_sec"] - 408.0) < 1e-6,
    )
    run_missing = tmp / "run-missing-replay"
    write_run(run_missing, {(2, 0): replay_summary(1.0)})
    rows_m = agg.aggregate(run_missing)
    check(
        "aggregator: a replay that wrote no summary.json is a FAIL row, so the run cannot pass",
        [(r["rung"], r["container"], r["verdict"]) for r in rows_m] == [(2, 0, "PASS"), (2, 1, "FAIL")]
        and "no summary.json" in rows_m[1]["diagnostics"]
        and gh.agentic_replay_tool_time(rows_m)[0] is None,
    )
    run_untimed = tmp / "run-replay-only-untimed"
    untimed, untimed_regime = replay_summary()
    untimed["comparison"] = {"tool_busy_replay_sec": 410.0}  # a replay-only summary that compares nothing
    write_run(run_untimed, {(1, 0): (untimed, untimed_regime)})
    rows_u = agg.aggregate(run_untimed)
    check(
        "aggregator reads the replay's tool time from a summary without a ratio",
        rows_u[0]["replay_tool_busy_sec"] == 410.0
        and rows_u[0]["replay_tool_time_ratio"] is None
        and rows_u[0]["live_tool_busy_sec"] is None,
    )
    out = run / agg.SUMMARY_CSV
    agg.write_csv(rows, out)
    back = list(csv.DictReader(out.open()))
    check(
        "aggregator CSV round-trips its columns",
        [b["verdict"] for b in back] == ["PASS", "PASS", "FAIL"] and set(back[0]) == set(agg.COLUMNS),
    )

    # ── handler ─────────────────────────────────────────────────────────────────────────────────────────
    h = gh.resolve("agentic", "replay-fidelity")
    check(
        "handler declares the exemplar metric as LOWER-is-better replay tool seconds",
        h.metric == "replay_tool_busy_sec" and h.higher_better is False and h.unit == "s",
    )
    native = h.read_native_rows(run)  # prefers the CSV the aggregator wrote
    metric, value, source = h.compute_metric(native)
    check(
        "compute_metric: any FAILED replay → no value (a failed replay is not a slower one)",
        value is None and source == "max_over_passing_container_replays",
    )
    run_ok = tmp / "run-b"
    write_run(run_ok, {(1, 0): replay_summary(1.02), (2, 0): replay_summary(1.03), (2, 1): replay_summary(1.045)})
    native_ok = h.read_native_rows(run_ok)  # no CSV yet → aggregate_summary runs the aggregator
    check(
        "read_native_rows aggregates a raw run-dir when the CSV is absent",
        (run_ok / h.summary_csv).is_file() and len(native_ok) == 3,
    )
    metric, value, _ = h.compute_metric(native_ok)
    check("compute_metric: longest PASSING replay tool time over rungs and containers", value == 418.0)
    fields = h.result_fields(native_ok)
    check(
        "result_fields: per_point covers every container replay + pass counts + regimes",
        len(fields["per_point"]) == 3
        and fields["replays_passed"] == 3
        and fields["replays_total"] == 3
        and fields["cpu_regimes"] == ["exclusive-cores"],
    )
    rungs = [h.rung_from_native(r) for r in native_ok]
    round_trip = [h.rung_from_native(h.native_from_rung(r)) for r in rungs]
    check(
        "rung_from_native ⇄ native_from_rung round-trips exactly",
        round_trip == rungs and rungs[0]["concurrency"] == 1 and rungs[2]["container"] == 1,
    )
    check("rungs_csv_row follows rung_keys", len(h.rungs_csv_row(rungs[0])) == len(h.rung_keys))
    table = h.table(native_ok)
    check(
        "table renders one row per container replay with the tool time and CPU columns",
        table.count("\n") == 4 and "418.0" in table and "exclusive-cores" in table,
    )
    head = h.headline(fields)
    check("headline names the longest tool time and the pass count", "418.0" in head and "3 passing" in head)
    record = {"result": fields, "detail": {"rungs": rungs}}
    check(
        "headline_from_record / table_from_record regenerate from a committed record",
        h.headline_from_record(record) == head and h.table_from_record(record) == table,
    )
    cell_copy = tmp / "cell"
    shutil.copytree(REPLAY_CELL, cell_copy, ignore=shutil.ignore_patterns("rendered"))
    h.set_baseline(cell_copy, csv_path=run_ok / h.summary_csv)
    check(
        "set_baseline writes the measured value into exemplar.reference (comments preserved)",
        "reference: 418.0" in (cell_copy / "recipe.yaml").read_text()
        and "# first Grace" in (cell_copy / "recipe.yaml").read_text(),
    )

    # ── capture-shaped summaries go through the same aggregator ───────────────────────────────────────
    run_c = tmp / "run-c"
    write_run(run_c, {(1, 0): capture_summary(match=True), (1, 1): capture_summary(match=False)})
    rows_c = agg.aggregate(run_c)
    check(
        "capture summaries: identity mismatch fails the check; nested replay summary is read",
        [r["verdict"] for r in rows_c] == ["PASS", "FAIL"]
        and rows_c[0]["actions_executed"] == 118
        and rows_c[0]["capture_reward"] == 1.0
        and rows_c[1]["image_identity_match"] is False,
    )

    # ── run_result: the agentic verdict payload ───────────────────────────────────────────────────────
    payload, csv_present = run_result.evaluate(str(cell_copy), str(run_ok), ROOT)
    check(
        "run_result agentic: with a reference set, EXEMPLAR when the value is within tolerance of it",
        payload.get("verdict") == "EXEMPLAR"
        and payload.get("value") == 418.0
        and payload.get("metric") == "replay_tool_busy_sec"
        and csv_present is True,
        json.dumps(payload)[:200],
    )
    payload_bad, _ = run_result.evaluate(str(REPLAY_CELL), str(run), ROOT)
    check(
        "run_result agentic: a failed replay is NOT_VALID with the pass count in the error",
        payload_bad.get("verdict") == "NOT_VALID" and "2/3" in (payload_bad.get("error") or ""),
        json.dumps(payload_bad)[:200],
    )
    run_ref = tmp / "run-d"
    write_run(run_ref, {(1, 0): replay_summary(1.03)})
    payload_ref, _ = run_result.evaluate(str(REPLAY_CELL), str(run_ref), ROOT)  # shipped cell: no reference
    check(
        "run_result agentic: the shipped cell is measured, not graded (NO_REFERENCE, value reported)",
        payload_ref.get("verdict") == "NO_REFERENCE"
        and payload_ref.get("value") == 412.0
        and payload_ref.get("reference") is None,
        json.dumps(payload_ref)[:160],
    )
    check(
        "run_result agentic: the result block lists each replay's tool time and verdict",
        "replays: c1/0 412s PASS" in run_result.render_result_block(payload_ref, str(run_ref)),
    )
    run_far = tmp / "run-e"
    write_run(run_far, {(1, 0): replay_summary(1.03)})
    (cell_far := tmp / "cell-far").mkdir()
    (cell_far / "recipe.yaml").write_text(
        (REPLAY_CELL / "recipe.yaml").read_text().replace("reference: null\n", "reference: 380.0\n")
    )
    payload_far, _ = run_result.evaluate(str(cell_far), str(run_far), ROOT)
    check(
        "run_result agentic: NOT_EXEMPLAR when worse than the reference by more than the tolerance",
        payload_far.get("verdict") == "NOT_EXEMPLAR" and payload_far.get("delta_pct") > 5,
    )

    # ── identity + fingerprint of the shipped cells ───────────────────────────────────────────────────
    import yaml

    r_id, c_id = bid.benchmark_id(REPLAY_CELL), bid.benchmark_id(CAPTURE_CELL)
    check("benchmark_id: the two shipped agentic cells have distinct identities", r_id != c_id and len(r_id) == 64)
    replay_recipe = yaml.safe_load((REPLAY_CELL / "recipe.yaml").read_text())
    ident = bid.benchmark_identity(replay_recipe)
    check(
        "benchmark_identity carries the trajectory pin, pacing, cpus, rungs — not the image digest",
        ident["tasktrace"]["trace_sha256"] == replay_recipe["tasktrace"]["trace"]["sha256"]
        and ident["tasktrace"]["rungs"] == [1]
        and "digest" not in json.dumps(ident),
    )
    recipe = yaml.safe_load((REPLAY_CELL / "recipe.yaml").read_text())
    fp = rh.fingerprint_input(recipe, REPLAY_CELL)
    check(
        "recipe_hash fingerprints the tasktrace block and the rendered Job",
        fp.get("tasktrace") == recipe["tasktrace"] and "tasktrace-job.yaml" in fp["rendered"],
    )
    llm_cells = sorted(p.parent for p in (ROOT / "recipes/llm-perf").glob("**/recipe.yaml"))
    fp_llm = rh.fingerprint_input(yaml.safe_load((llm_cells[0] / "recipe.yaml").read_text()), llm_cells[0])
    check(
        "recipe_hash: llm-perf fingerprints gain NO tasktrace key (byte-for-byte unchanged)", "tasktrace" not in fp_llm
    )

    # ── export lane ────────────────────────────────────────────────────────────────────────────────────
    cols = exp.lane_columns("agentic", "replay-fidelity")
    check(
        "export: agentic lane columns = CORE + per-container replay metrics + blobs",
        "replay_tool_time_ratio" in cols
        and "tool_time_baseline" not in cols
        and "cpu_regime" in cols
        and "container" in cols
        and cols[: len(exp.CORE)] == exp.CORE,
    )
    check(
        "export: every handler rung key except concurrency is an export column",
        set(h.rung_keys) - {"concurrency"} <= set(cols),
    )
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nselftest_agentic_lane: {'OK' if FAILS == 0 else f'{FAILS} FAILED'}")
raise SystemExit(1 if FAILS else 0)

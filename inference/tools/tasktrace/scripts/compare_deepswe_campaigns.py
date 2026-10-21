#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two completed single-allocation DeepSWE campaigns task by task."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_campaign(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if len(manifest["allocations"]) != 1 or manifest["repetitions"] != 1:
        raise SystemExit("Comparison expects one allocation and one repetition.")
    allocation = manifest["allocations"][0]
    output_root = Path(manifest["output_root"])
    tasks: dict[str, dict[str, Any]] = {}
    for task in manifest["tasks"]:
        run_dir = output_root / allocation["name"] / task["name"] / "rep-01"
        report = read_json(run_dir / "pipeline_report.json")
        status = read_json(run_dir / "sweep_run_status.json")
        replay = report["replay_summary"]
        scheduler = report["scheduler_concurrency"]
        cpu = report["container_cpu"]
        scheduler_valid = bool(scheduler["cgroup_validation"]["valid_for_concurrency_reporting"])
        timeout = int(replay.get("actions_timed_out", 0)) > 0
        mismatches = int(replay.get("source_outcome_mismatch_count", 0))
        image_valid = report["image_identity"]["id"] == task["expected_image_id"]
        tasks[task["name"]] = {
            "task_name": task["name"],
            "elapsed_sec": float(status["elapsed_sec"]),
            "wall_sec": float(cpu["wall_sec"]),
            "cpu_sec": float(cpu["usage_sec"]),
            "average_cores": float(cpu["average_logical_cores_used"]),
            "utilization_percent": float(cpu["utilization_percent_of_allocation"]),
            "peak": int(scheduler["peak_simultaneous_logical_cores"]),
            "scheduler_valid": scheduler_valid,
            "lost_events": int(scheduler.get("lost_perf_events", 0)),
            "anomaly_count": len(scheduler.get("anomalies", [])),
            "timeout": timeout,
            "mismatches": mismatches,
            "image_valid": image_valid,
            "fidelity_valid": (scheduler_valid and image_valid and not timeout and mismatches == 0),
            "complete_non_timeout": scheduler_valid and image_valid and not timeout,
        }
    return {
        "manifest": str(manifest_path),
        "name": manifest["name"],
        "logical_cores": int(allocation["logical_cores"]),
        "cpuset_cpus": allocation["cpuset_cpus"],
        "tasks": tasks,
    }


def aggregate(
    names: list[str],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    pairs = [(baseline["tasks"][name], candidate["tasks"][name]) for name in names]
    base_wall = sum(left["wall_sec"] for left, _right in pairs)
    cand_wall = sum(right["wall_sec"] for _left, right in pairs)
    base_cpu = sum(left["cpu_sec"] for left, _right in pairs)
    cand_cpu = sum(right["cpu_sec"] for _left, right in pairs)
    speedups = [left["wall_sec"] / right["wall_sec"] for left, right in pairs]
    return {
        "task_count": len(pairs),
        "baseline_total_wall_sec": round(base_wall, 9),
        "candidate_total_wall_sec": round(cand_wall, 9),
        "aggregate_wall_speedup": round(base_wall / cand_wall, 6),
        "aggregate_wall_reduction_percent": round((1.0 - cand_wall / base_wall) * 100.0, 6),
        "baseline_total_cpu_sec": round(base_cpu, 9),
        "candidate_total_cpu_sec": round(cand_cpu, 9),
        "candidate_cpu_time_change_percent": round((cand_cpu / base_cpu - 1.0) * 100.0, 6),
        "baseline_weighted_average_cores": round(base_cpu / base_wall, 9),
        "candidate_weighted_average_cores": round(cand_cpu / cand_wall, 9),
        "baseline_weighted_utilization_percent": round(base_cpu / base_wall / baseline["logical_cores"] * 100.0, 6),
        "candidate_weighted_utilization_percent": round(cand_cpu / cand_wall / candidate["logical_cores"] * 100.0, 6),
        "median_task_speedup": round(statistics.median(speedups), 6),
        "geometric_mean_task_speedup": round(math.exp(statistics.mean(math.log(value) for value in speedups)), 6),
        "tasks_faster": sum(value > 1.0 for value in speedups),
        "tasks_slower": sum(value < 1.0 for value in speedups),
        "tasks_at_least_10_percent_faster": sum(value >= 1.1 for value in speedups),
        "tasks_at_least_10_percent_slower": sum(value <= 1 / 1.1 for value in speedups),
    }


def fmt_duration(seconds: float) -> str:
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_manifest", type=Path)
    parser.add_argument("candidate_manifest", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_campaign(args.baseline_manifest)
    candidate = load_campaign(args.candidate_manifest)
    names = sorted(baseline["tasks"])
    if names != sorted(candidate["tasks"]):
        raise SystemExit("Campaign task populations differ.")

    both_fidelity_valid = [
        name
        for name in names
        if baseline["tasks"][name]["fidelity_valid"] and candidate["tasks"][name]["fidelity_valid"]
    ]
    both_complete = [
        name
        for name in names
        if baseline["tasks"][name]["complete_non_timeout"] and candidate["tasks"][name]["complete_non_timeout"]
    ]
    task_rows = []
    for name in names:
        left = baseline["tasks"][name]
        right = candidate["tasks"][name]
        task_rows.append(
            {
                "task_name": name,
                "baseline_wall_sec": left["wall_sec"],
                "candidate_wall_sec": right["wall_sec"],
                "wall_speedup": round(left["wall_sec"] / right["wall_sec"], 6),
                "baseline_average_cores": left["average_cores"],
                "candidate_average_cores": right["average_cores"],
                "baseline_peak": left["peak"],
                "candidate_peak": right["peak"],
                "baseline_fidelity_valid": left["fidelity_valid"],
                "candidate_fidelity_valid": right["fidelity_valid"],
            }
        )

    elapsed_base = sum(item["elapsed_sec"] for item in baseline["tasks"].values())
    elapsed_cand = sum(item["elapsed_sec"] for item in candidate["tasks"].values())
    summary = {
        "schema_version": "deepswe-campaign-comparison.v1",
        "baseline": {key: baseline[key] for key in ("manifest", "name", "logical_cores", "cpuset_cpus")},
        "candidate": {key: candidate[key] for key in ("manifest", "name", "logical_cores", "cpuset_cpus")},
        "task_count": len(names),
        "saved_task_attempt_time": {
            "baseline_sec": round(elapsed_base, 6),
            "candidate_sec": round(elapsed_cand, 6),
            "speedup": round(elapsed_base / elapsed_cand, 6),
            "reduction_percent": round((1.0 - elapsed_cand / elapsed_base) * 100.0, 6),
        },
        "fidelity_transitions": {
            "valid_in_both": len(both_fidelity_valid),
            "valid_only_at_baseline": sum(
                baseline["tasks"][name]["fidelity_valid"] and not candidate["tasks"][name]["fidelity_valid"]
                for name in names
            ),
            "valid_only_at_candidate": sum(
                candidate["tasks"][name]["fidelity_valid"] and not baseline["tasks"][name]["fidelity_valid"]
                for name in names
            ),
            "invalid_in_both": sum(
                not baseline["tasks"][name]["fidelity_valid"] and not candidate["tasks"][name]["fidelity_valid"]
                for name in names
            ),
        },
        "bcc_validation": {
            "baseline_valid": sum(item["scheduler_valid"] for item in baseline["tasks"].values()),
            "candidate_valid": sum(item["scheduler_valid"] for item in candidate["tasks"].values()),
            "baseline_lost_events": sum(item["lost_events"] for item in baseline["tasks"].values()),
            "candidate_lost_events": sum(item["lost_events"] for item in candidate["tasks"].values()),
            "baseline_anomaly_reports": sum(item["anomaly_count"] > 0 for item in baseline["tasks"].values()),
            "candidate_anomaly_reports": sum(item["anomaly_count"] > 0 for item in candidate["tasks"].values()),
        },
        "paired_fidelity_valid": aggregate(both_fidelity_valid, baseline, candidate),
        "paired_complete_non_timeout": aggregate(both_complete, baseline, candidate),
        "all_reports_diagnostic": aggregate(names, baseline, candidate),
        "fastest_candidate_speedups": sorted(task_rows, key=lambda item: item["wall_speedup"], reverse=True)[:15],
        "slowest_candidate_speedups": sorted(task_rows, key=lambda item: item["wall_speedup"])[:15],
        "tasks": task_rows,
    }
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    paired = summary["paired_fidelity_valid"]
    complete = summary["paired_complete_non_timeout"]
    attempt = summary["saved_task_attempt_time"]
    bcc = summary["bcc_validation"]
    base_cores = baseline["logical_cores"]
    cand_cores = candidate["logical_cores"]
    pop = len(names)
    lines = [
        f"# DeepSWE {base_cores}-Core vs {cand_cores}-Core Replay",
        "",
        f"- Population: **{pop} identical pinned trajectories/images**.",
        f"- BCC-valid: **{bcc['baseline_valid']}/{pop}** at {base_cores} "
        f"cores and **{bcc['candidate_valid']}/{pop}** at {cand_cores} cores.",
        f"- Lost BCC events: **{bcc['baseline_lost_events']}** baseline, "
        f"**{bcc['candidate_lost_events']}** candidate.",
        f"- Scheduler anomaly reports: **{bcc['baseline_anomaly_reports']}** baseline, "
        f"**{bcc['candidate_anomaly_reports']}** candidate.",
        f"- Fidelity-valid in both runs: **{len(both_fidelity_valid)}**.",
        "",
        "## Conservative Paired Result",
        "",
        f"Across the {len(both_fidelity_valid)} tasks that were fidelity-valid at both widths:",
        "",
        f"- Aggregate measured wall time: **{fmt_duration(paired['baseline_total_wall_sec'])}** "
        f"at {base_cores} cores vs **{fmt_duration(paired['candidate_total_wall_sec'])}** at {cand_cores}.",
        f"- Aggregate speedup: **{paired['aggregate_wall_speedup']:.3f}x** "
        f"({paired['aggregate_wall_reduction_percent']:.2f}% less wall time).",
        f"- Weighted average demand: **{paired['baseline_weighted_average_cores']:.3f}** "
        f"vs **{paired['candidate_weighted_average_cores']:.3f}** logical cores.",
        f"- Allocation utilization: **{paired['baseline_weighted_utilization_percent']:.2f}%** "
        f"of {base_cores} vs **{paired['candidate_weighted_utilization_percent']:.2f}%** of {cand_cores}.",
        f"- Candidate CPU-time change: **{paired['candidate_cpu_time_change_percent']:+.2f}%**.",
        f"- Tasks faster/slower: **{paired['tasks_faster']}/{paired['tasks_slower']}**.",
        "",
        "## Sensitivity",
        "",
        f"Across all {complete['task_count']} complete non-timeout task pairs, aggregate "
        f"speedup was **{complete['aggregate_wall_speedup']:.3f}x** "
        f"({complete['aggregate_wall_reduction_percent']:.2f}% less wall time).",
        "",
        f"Saved task-attempt time fell from **{fmt_duration(attempt['baseline_sec'])}** "
        f"to **{fmt_duration(attempt['candidate_sec'])}**, a "
        f"**{attempt['speedup']:.3f}x** speedup.",
        "",
    ]
    args.markdown_output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

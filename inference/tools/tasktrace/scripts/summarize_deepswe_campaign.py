#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a completed single-allocation DeepSWE core-sweep campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _aggregate(rows: list[dict[str, Any]], logical_cores: int) -> dict[str, Any]:
    if not rows:
        return {
            "task_count": 0,
            "total_wall_sec": 0.0,
            "total_cpu_sec": 0.0,
            "weighted_average_logical_cores_used": None,
            "weighted_utilization_percent_of_allocation": None,
            "median_task_utilization_percent": None,
            "median_task_wall_sec": None,
            "peak_simultaneous_logical_cores_distribution": {},
        }
    total_wall = sum(row["wall_sec"] for row in rows)
    total_cpu = sum(row["cpu_sec"] for row in rows)
    average_cores = total_cpu / total_wall if total_wall else 0.0
    return {
        "task_count": len(rows),
        "total_wall_sec": round(total_wall, 9),
        "total_cpu_sec": round(total_cpu, 9),
        "weighted_average_logical_cores_used": round(average_cores, 9),
        "weighted_utilization_percent_of_allocation": round(average_cores / logical_cores * 100.0, 6),
        "median_task_utilization_percent": round(statistics.median(row["utilization_percent"] for row in rows), 6),
        "median_task_wall_sec": round(statistics.median(row["wall_sec"] for row in rows), 6),
        "peak_simultaneous_logical_cores_distribution": dict(sorted(Counter(row["peak"] for row in rows).items())),
    }


def _strata(rows: list[dict[str, Any]], field: str, logical_cores: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result = []
    for value, group in sorted(grouped.items()):
        fidelity_rows = [row for row in group if row["fidelity_valid"]]
        result.append(
            {
                "value": value,
                "planned_task_count": len(group),
                "fidelity_valid_task_count": len(fidelity_rows),
                **_aggregate(fidelity_rows, logical_cores),
            }
        )
    return result


def _campaign_timing(output_root: Path) -> dict[str, Any]:
    controls = []
    for started_path in sorted(output_root.glob("_campaign_control/**/started_at")):
        finished_path = started_path.with_name("finished_at")
        if not finished_path.exists():
            continue
        started = datetime.fromisoformat(started_path.read_text().strip())
        finished = datetime.fromisoformat(finished_path.read_text().strip())
        controls.append(
            {
                "name": str(started_path.parent.relative_to(output_root)),
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "elapsed_sec": round((finished - started).total_seconds(), 6),
            }
        )
    if not controls:
        return {"invocations": [], "active_elapsed_sec": None, "span_elapsed_sec": None}
    controls.sort(key=lambda item: item["started_at"])
    return {
        "invocations": controls,
        "active_elapsed_sec": round(sum(item["elapsed_sec"] for item in controls), 6),
        "span_elapsed_sec": round(
            (
                datetime.fromisoformat(controls[-1]["finished_at"]) - datetime.fromisoformat(controls[0]["started_at"])
            ).total_seconds(),
            6,
        ),
    }


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h {minutes}m {secs}s"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    counts = summary["counts"]
    headline = summary["aggregates"]["fidelity_valid"]
    sensitivity = summary["aggregates"]["complete_non_timeout"]
    timing = summary["campaign_timing"]
    measurement = summary["measurement"]
    cpuset = measurement["cpuset_cpus"]
    cores = measurement["logical_cores"]
    lines = [
        f"# DeepSWE {counts['planned_tasks']}-Task CPU Replay Summary",
        "",
        "## Result",
        "",
        f"- Reports present: **{counts['reports_present']}/{counts['planned_tasks']}**.",
        f"- Fidelity-valid replays: **{counts['fidelity_valid']}**.",
        f"- Source-outcome mismatches: **{counts['source_outcome_mismatch']}**.",
        f"- Timed-out replays: **{counts['timeout']}**.",
        f"- BCC-valid reports: **{counts['scheduler_valid']}**; lost events: "
        f"**{counts['lost_perf_events']}**; reports with anomalies: "
        f"**{counts['reports_with_scheduler_anomalies']}**.",
        "",
        "The conservative headline includes only complete replays whose recorded "
        "source outcomes, pinned image identity, and BCC/cgroup CPU accounting all validate.",
        "",
        f"- Weighted average CPU use: **{headline['weighted_average_logical_cores_used']:.3f} " f"cores** of {cores}.",
        f"- Weighted utilization: **{headline['weighted_utilization_percent_of_allocation']:.2f}%**.",
        f"- Aggregate measured wall time: **{_fmt_duration(headline['total_wall_sec'])}**.",
        f"- Peak distribution: `{headline['peak_simultaneous_logical_cores_distribution']}`.",
        "",
        "As a sensitivity check, all complete non-timeout measurements, including " "source-outcome mismatches, used:",
        "",
        f"- **{sensitivity['weighted_average_logical_cores_used']:.3f} cores** of {cores}, or "
        f"**{sensitivity['weighted_utilization_percent_of_allocation']:.2f}%**.",
        "",
        "## Runtime",
        "",
        f"- Sum of task-attempt elapsed time: " f"**{_fmt_duration(summary['task_attempt_elapsed']['total_sec'])}**.",
        f"- Campaign process active time: **{_fmt_duration(timing['active_elapsed_sec'])}**.",
        f"- First start through final finish: **{_fmt_duration(timing['span_elapsed_sec'])}**.",
        f"- Tasks ran {measurement['execution_order']} on CPUs `{cpuset}`.",
        "- Verification and reporting occurred after the measured replay windows.",
        "",
        "## Strata",
        "",
    ]
    for field, title in (
        ("source_action_quartile", "Recorded Action Quartile"),
        ("source_duration_quartile", "Source Duration Quartile"),
        ("source_language", "Language"),
        ("source_reward", "Source Reward"),
    ):
        lines.extend(
            [
                f"### {title}",
                "",
                "| Stratum | Planned | Valid | Avg cores | Utilization | Wall time |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["strata"][field]:
            average = row["weighted_average_logical_cores_used"]
            utilization = row["weighted_utilization_percent_of_allocation"]
            lines.append(
                f"| {row['value']} | {row['planned_task_count']} | "
                f"{row['fidelity_valid_task_count']} | "
                f"{average:.3f} | {utilization:.2f}% | "
                f"{_fmt_duration(row['total_wall_sec'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Validation Contract",
            "",
            "- Runtime and utilization come from container cgroup v2 `cpu.stat`.",
            "- Peak concurrency comes from BCC/eBPF `finish_task_switch` events over " "the measured cgroup subtree.",
            "- Every BCC report was reconciled against authoritative cgroup CPU time.",
            "- Replay fidelity compares each action's normalized outcome against the "
            "recorded trajectory; it does not claim byte-for-byte filesystem identity.",
            f"- Network policy: {measurement['network_policy']}.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument(
        "--scheduler-backend",
        default="bcc",
        help="Scheduler tracing backend used for peak concurrency (default: bcc)",
    )
    parser.add_argument(
        "--execution-order",
        default="sequential",
        help="How tasks were executed: sequential, bounded-concurrent, etc. (default: sequential)",
    )
    parser.add_argument(
        "--network-policy",
        default="docker-default",
        help="Network policy applied during profiling (default: docker-default)",
    )
    args = parser.parse_args()

    manifest = _read_json(args.manifest)
    if len(manifest["allocations"]) != 1 or manifest.get("repetitions") != 1:
        raise SystemExit("This summary expects one allocation and one repetition.")
    allocation = manifest["allocations"][0]
    logical_cores = int(allocation["logical_cores"])
    output_root = Path(manifest["output_root"])
    rows = []
    for task in manifest["tasks"]:
        run_dir = output_root / allocation["name"] / task["name"] / "rep-01"
        report = _read_json(run_dir / "pipeline_report.json")
        status = _read_json(run_dir / "sweep_run_status.json")
        replay = report["replay_summary"]
        scheduler = report["scheduler_concurrency"]
        cpu = report["container_cpu"]
        scheduler_valid = bool(scheduler["cgroup_validation"]["valid_for_concurrency_reporting"])
        image_valid = report["image_identity"]["id"] == task["expected_image_id"]
        timeout = int(replay.get("actions_timed_out", 0)) > 0
        mismatch_count = int(replay.get("source_outcome_mismatch_count", 0))
        row = {
            "task_name": task["name"],
            "status": status["status"],
            "elapsed_sec": float(status["elapsed_sec"]),
            "wall_sec": float(cpu["wall_sec"]),
            "cpu_sec": float(cpu["usage_sec"]),
            "utilization_percent": float(cpu["utilization_percent_of_allocation"]),
            "average_cores": float(cpu["average_logical_cores_used"]),
            "peak": int(scheduler["peak_simultaneous_logical_cores"]),
            "scheduler_valid": scheduler_valid,
            "lost_perf_events": int(scheduler.get("lost_perf_events", 0)),
            "scheduler_anomaly_count": len(scheduler.get("anomalies", [])),
            "image_valid": image_valid,
            "timeout": timeout,
            "source_outcome_mismatch_count": mismatch_count,
            "fidelity_valid": (scheduler_valid and image_valid and not timeout and mismatch_count == 0),
            "complete_non_timeout": scheduler_valid and image_valid and not timeout,
            "source_action_quartile": task["source_action_quartile"],
            "source_duration_quartile": task["source_duration_quartile"],
            "source_language": task["source_language"],
            "source_reward": task["source_reward"],
            "recorded_action_count": task["recorded_action_count"],
            "source_agent_duration_sec": task["source_agent_duration_sec"],
        }
        rows.append(row)

    fidelity_rows = [row for row in rows if row["fidelity_valid"]]
    non_timeout_rows = [row for row in rows if row["complete_non_timeout"]]
    counts = {
        "planned_tasks": len(manifest["tasks"]),
        "reports_present": len(rows),
        "passed_status": sum(row["status"] == "passed" for row in rows),
        "failed_validation_status": sum(row["status"] == "failed_validation" for row in rows),
        "fidelity_valid": len(fidelity_rows),
        "source_outcome_mismatch": sum(row["source_outcome_mismatch_count"] > 0 for row in rows),
        "timeout": sum(row["timeout"] for row in rows),
        "scheduler_valid": sum(row["scheduler_valid"] for row in rows),
        "image_valid": sum(row["image_valid"] for row in rows),
        "lost_perf_events": sum(row["lost_perf_events"] for row in rows),
        "reports_with_scheduler_anomalies": sum(row["scheduler_anomaly_count"] > 0 for row in rows),
    }
    task_elapsed = [row["elapsed_sec"] for row in rows]
    summary = {
        "schema_version": "deepswe-cpu-population-summary.v1",
        "manifest": str(args.manifest),
        "experiment_name": manifest["name"],
        "measurement": {
            "logical_cores": logical_cores,
            "cpuset_cpus": allocation["cpuset_cpus"],
            "scheduler_backend": args.scheduler_backend,
            "execution_order": args.execution_order,
            "network_policy": args.network_policy,
        },
        "counts": counts,
        "campaign_timing": _campaign_timing(output_root),
        "task_attempt_elapsed": {
            "total_sec": round(sum(task_elapsed), 6),
            "average_sec": round(statistics.mean(task_elapsed), 6),
            "median_sec": round(statistics.median(task_elapsed), 6),
            "tasks_at_least_60_sec": sum(value >= 60 for value in task_elapsed),
            "tasks_at_least_120_sec": sum(value >= 120 for value in task_elapsed),
            "tasks_at_least_300_sec": sum(value >= 300 for value in task_elapsed),
        },
        "aggregates": {
            "fidelity_valid": _aggregate(fidelity_rows, logical_cores),
            "complete_non_timeout": _aggregate(non_timeout_rows, logical_cores),
            "all_reports_diagnostic": _aggregate(rows, logical_cores),
        },
        "strata": {
            field: _strata(rows, field, logical_cores)
            for field in (
                "source_action_quartile",
                "source_duration_quartile",
                "source_language",
                "source_reward",
            )
        },
        "invalid_tasks": [row for row in rows if not row["fidelity_valid"]],
        "longest_tasks": sorted(rows, key=lambda row: row["elapsed_sec"], reverse=True)[:15],
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, args.markdown_output)


if __name__ == "__main__":
    main()

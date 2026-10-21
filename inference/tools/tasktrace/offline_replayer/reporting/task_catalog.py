#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a browser-friendly Terminal-Bench task and CPU evidence catalog.

The catalog keeps task descriptions for the entire benchmark inventory while
preserving measurement quality flags.  The dashboard may display diagnostic
tasks, but only validated tasks receive exact concurrency/command attribution.
Reviewed explanations are deliberately static narrative attached to validated
evidence rather than generated claims from task names alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "tb2-task-catalog.v1"


class TaskCatalogError(RuntimeError):
    """Raised when required catalog source data cannot be read."""


REVIEWED_CPU_EXPLANATIONS = {
    "bn-fit-modify": (
        "statistics and Bayesian-network fitting",
        "NumPy/SciPy statistical kernels and Bayesian-network operations create a real "
        "short CPU burst, although the full run is approximately single-core.",
    ),
    "break-filter-js-from-html": (
        "browser validation and HTML security testing",
        "The measured ceiling touch occurs during Selenium/Chromium validation and is "
        "too short to treat as meaningful 16-core demand.",
    ),
    "build-cython-ext": (
        "native extension build and validation",
        "Cython extension compilation, installation, and import/test verification "
        "produce short native build bursts; the overall build remains narrow.",
    ),
    "distribution-search": (
        "numerical optimization",
        "SciPy root solving and NumPy operations over a 150,000-element distribution "
        "produce a repeatable, meaningful wide numerical burst.",
    ),
    "largest-eigenval": (
        "linear algebra optimization",
        "Repeated NumPy eigenvalue/eigenvector evaluation during performance validation "
        "produces a real native linear-algebra burst.",
    ),
    "model-extraction-relu-logits": (
        "neural-network probing and numerical reconstruction",
        "NumPy-based black-box model queries and verification produce a real numerical "
        "burst, but the total task is very short.",
    ),
    "modernize-scientific-stack": (
        "dataframe climate analysis",
        "Pandas/NumPy startup and dataframe computation cause a repeatable scientific "
        "library burst, but it is brief.",
    ),
    "multi-source-data-merger": (
        "dataframe and parquet processing",
        "Pandas/PyArrow parsing, merging, and Parquet serialization execute native "
        "code and produce meaningful short multi-core activity.",
    ),
    "portfolio-optimization": (
        "native numerical extension and benchmark",
        "Extension compilation and numerical benchmark commands produce a repeatable "
        "native burst, not sustained sixteen-core optimizer demand.",
    ),
    "pytorch-model-recovery": (
        "machine-learning tensor computation",
        "PyTorch model loading, output-layer fitting, and verification use CPU tensor "
        "kernels and create a material multi-core burst.",
    ),
    "rstan-to-pystan": (
        "probabilistic sampling",
        "PyStan model compilation and posterior sampling create real parallel work; "
        "the run is primarily a sustained four-core workload with a brief 16-core touch.",
    ),
    "sanitize-git-repo": (
        "repository secret scanning",
        "The observed burst is concentrated in repository/secret scans such as git "
        "grep and lasts only milliseconds, so it is negligible for capacity planning.",
    ),
    "tune-mjcf": (
        "physics simulation tuning",
        "Repeated MuJoCo/NumPy simulation evaluation produces genuine numeric bursts, "
        "but peak usage is tiny relative to the long task duration.",
    ),
    "video-processing": (
        "computer vision video analysis",
        "OpenCV video decoding and frame analysis create a repeatable, meaningful "
        "multi-core burst while identifying takeoff and landing frames.",
    ),
}

MECHANISM_ONLY_REVIEWED_CPU_EXPLANATIONS = {
    "bn-fit-modify": (
        "statistics and Bayesian-network fitting",
        "CPU work comes from NumPy/SciPy statistics plus Bayesian-network fitting, " "intervention, and sampling.",
    ),
    "break-filter-js-from-html": (
        "browser validation and HTML security testing",
        "CPU work comes from Selenium/Chromium loading and validating the crafted HTML payload.",
    ),
    "build-cython-ext": (
        "native extension build and validation",
        "CPU work comes from Cython compilation, native extension linking, installation, " "and test execution.",
    ),
    "distribution-search": (
        "numerical optimization",
        "CPU work comes from SciPy root solving and NumPy calculations over a "
        "150,000-element probability distribution.",
    ),
    "largest-eigenval": (
        "linear algebra optimization",
        "CPU work comes from repeated eigenvalue/eigenvector calculations and benchmark validation.",
    ),
    "model-extraction-relu-logits": (
        "neural-network probing and numerical reconstruction",
        "CPU work comes from repeated neural-network queries and NumPy reconstruction checks.",
    ),
    "modernize-scientific-stack": (
        "dataframe climate analysis",
        "CPU work comes from pandas CSV processing and NumPy-backed climate aggregation.",
    ),
    "multi-source-data-merger": (
        "dataframe and parquet processing",
        "CPU work comes from Pandas/PyArrow parsing, record merging, and Parquet serialization.",
    ),
    "portfolio-optimization": (
        "native numerical extension and benchmark",
        "CPU work comes from native extension compilation and numerical risk/return benchmarking.",
    ),
    "pytorch-model-recovery": (
        "machine-learning tensor computation",
        "CPU work comes from PyTorch model reconstruction, output-layer fitting, and tensor validation.",
    ),
    "rstan-to-pystan": (
        "probabilistic sampling",
        "CPU work comes from PyStan model compilation and posterior sampling.",
    ),
    "sanitize-git-repo": (
        "repository secret scanning",
        "CPU work comes from scanning repository contents for secret values and verifying replacements.",
    ),
    "tune-mjcf": (
        "physics simulation tuning",
        "CPU work comes from repeated MuJoCo physics simulations and numerical correctness evaluation.",
    ),
    "video-processing": (
        "computer vision video analysis",
        "CPU work comes from OpenCV video decoding and per-frame motion analysis.",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskCatalogError(f"unable to read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TaskCatalogError(f"JSON source is not an object: {path}")
    return value


def _load_commands(path: Path | None) -> dict[str, list[dict[str, str]]]:
    if path is None:
        return {}
    try:
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
    except OSError as exc:
        raise TaskCatalogError(f"unable to read command CSV {path}: {exc}") from exc
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("task_name"):
            by_task.setdefault(row["task_name"], []).append(row)
    return by_task


def _instruction_from_pipeline_report(report_path: Path) -> tuple[str, str]:
    report = _read_json(report_path)
    source = report.get("source_trajectory")
    if not isinstance(source, str) or not source:
        return "", ""
    trajectory_path = Path(source)
    trajectory = _read_json(trajectory_path)
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return str(trajectory_path), ""
    messages = [
        step["message"]
        for step in steps
        if isinstance(step, dict)
        and step.get("source") == "user"
        and isinstance(step.get("message"), str)
        and not step["message"].startswith("<environment_context>")
    ]
    return str(trajectory_path), messages[0] if messages else ""


def _instruction_excerpt(instruction: str, *, limit: int = 360) -> str:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph.strip().lstrip("#").strip())
        for paragraph in re.split(r"\n\s*\n", instruction)
        if paragraph.strip()
    ]
    if not paragraphs:
        return ""
    selected = paragraphs[0]
    if (len(selected) < 80 or selected.endswith(":")) and len(paragraphs) > 1:
        separator = " " if selected.endswith(":") else ": "
        selected = f"{selected}{separator}{paragraphs[1]}"
    if len(selected) > limit:
        selected = selected[: limit - 3].rstrip() + "..."
    return selected


def _display_name(task_name: str) -> str:
    return task_name.replace("-", " ").title()


def _float(row: dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, 0.0) or 0.0)
    except ValueError:
        return 0.0


def _int(row: dict[str, str], field: str) -> int:
    try:
        return int(float(row.get(field, 0) or 0))
    except ValueError:
        return 0


def _ranked_commands(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Choose UI evidence by high-concurrency duration before generic wall time."""
    ranked = sorted(
        rows,
        key=lambda row: (
            _float(row, "time_at_or_above_8_cores_sec"),
            _float(row, "time_at_or_above_4_cores_sec"),
            _float(row, "average_simultaneous_logical_cores"),
            _float(row, "lifecycle_wall_sec"),
        ),
        reverse=True,
    )
    return [
        {
            "command_family": row.get("command_family", ""),
            "command_preview": row.get("command_preview", ""),
            "lifecycle_wall_sec": _float(row, "lifecycle_wall_sec"),
            "average_simultaneous_logical_cores": _float(row, "average_simultaneous_logical_cores"),
            "peak_simultaneous_logical_cores": _int(row, "peak_simultaneous_logical_cores"),
            "time_at_or_above_4_cores_sec": _float(row, "time_at_or_above_4_cores_sec"),
            "time_at_or_above_8_cores_sec": _float(row, "time_at_or_above_8_cores_sec"),
        }
        for row in ranked[:3]
    ]


def _workload_category(task_name: str, instruction: str, command_text: str) -> str:
    text = f"{task_name} {instruction} {command_text}".lower()
    patterns = (
        (("stan", "mcmc", "posterior"), "probabilistic sampling"),
        (("torch", "pytorch", "caffe", "cifar", "inference", "model"), "machine-learning computation"),
        (("video", "cv2", "image", "cell-seg"), "media or image processing"),
        (("cython", "compile", "build", "gcc", "compcert"), "native compilation or build"),
        (("numpy", "scipy", "eigen", "portfolio", "bayesian", "raman"), "numerical or statistical computation"),
        (("qemu", "windows 3.11"), "virtualization or emulation"),
        (("git", "repository", "secret"), "repository processing"),
        (("sqlite", "database", "wal"), "database processing"),
        (("nginx", "server", "grpc"), "server or network workload"),
        (("regex", "html", "text", "log", "cobol", "gcode"), "text or document processing"),
    )
    for keywords, category in patterns:
        if any(keyword in text for keyword in keywords):
            return category
    return "general command-line workload"


def _generic_reason(category: str, dominant: dict[str, Any] | None, *, valid: bool) -> str:
    if not valid:
        return (
            f"The task concerns {category}. CPU observations are diagnostic only because "
            "the replay failed validation, so its recorded peak should not be presented "
            "as an exact concurrency measurement."
        )
    if dominant is None:
        return (
            f"The task concerns {category}; no command-level driver was available in " "the validated command report."
        )
    family = dominant["command_family"] or "an executed command"
    peak = dominant["peak_simultaneous_logical_cores"]
    at_eight = dominant["time_at_or_above_8_cores_sec"]
    if at_eight >= 0.01:
        return (
            f"The task concerns {category}. Measured high-core activity is concentrated "
            f"in `{family}`, which reached peak {peak} and spent {at_eight:.3f} s at "
            "eight or more simultaneous logical cores."
        )
    return (
        f"The task concerns {category}. The highest-ranked measured command is "
        f"`{family}` (peak {peak}), but it did not spend material time at eight or "
        "more simultaneous logical cores."
    )


def _mechanism_only_reason(category: str, dominant: dict[str, Any] | None) -> str:
    category_reasons = {
        "probabilistic sampling": "CPU work comes from probabilistic-model compilation and sampling.",
        "media or image processing": "CPU work comes from image/video decoding and pixel-level processing.",
        "machine-learning computation": "CPU work comes from neural-network execution, training, or tensor kernels.",
        "native compilation or build": "CPU work comes from compiling, linking, installing, and validating native software.",
        "numerical or statistical computation": "CPU work comes from numerical array, linear-algebra, or statistical kernels.",
        "virtualization or emulation": "CPU work comes from virtual-machine or emulator execution.",
        "repository processing": "CPU work comes from scanning, comparing, and rewriting repository contents.",
        "database processing": "CPU work comes from database parsing, recovery, query, and verification operations.",
        "server or network workload": "CPU work comes from server execution, request handling, and validation.",
        "text or document processing": "CPU work comes from parsing, transforming, searching, and validating text files.",
        "general command-line workload": "CPU work comes from the programs executed to solve and validate the task.",
    }
    reason = category_reasons[category]
    if dominant and dominant["command_family"]:
        return f"{reason} Primary measured driver: `{dominant['command_family']}`."
    return reason


def _duration_class(task: dict[str, Any], validated_summary: dict[str, dict[str, Any]]) -> str:
    if task.get("status") != "passed":
        return "diagnostic-only observation"
    task_name = str(task.get("task_name"))
    summary = validated_summary.get(task_name, {})
    peak = int(task.get("scheduler_peak_simultaneous_logical_cores") or 0)
    average = float(task.get("average_logical_cores_used") or 0.0)
    wall = float(task.get("wall_sec") or 0.0)
    distribution = summary.get("time_sec_by_simultaneous_logical_cores", {})
    at_ceiling = float(distribution.get("16", 0.0)) if isinstance(distribution, dict) else 0.0
    if peak == 16 and at_ceiling < 0.01:
        return "transient 16-core touch"
    if average >= 2.0:
        return "elevated average multi-core use"
    if peak >= 8 and wall and at_ceiling / wall < 0.01:
        return "brief high-core burst in mostly narrow run"
    if peak >= 8:
        return "measured high-core burst"
    return "mostly narrow measured run"


def build_catalog(
    analysis_path: Path,
    validated_summary_path: Path | None,
    commands_path: Path | None,
) -> dict[str, Any]:
    """Join prompt text, validation status, metrics, and display explanations."""
    analysis = _read_json(analysis_path)
    tasks = analysis.get("tasks")
    if not isinstance(tasks, list):
        raise TaskCatalogError(f"{analysis_path} has no task list")
    validated_summary: dict[str, dict[str, Any]] = {}
    if validated_summary_path is not None:
        payload = _read_json(validated_summary_path)
        for row in payload.get("tasks", []):
            if isinstance(row, dict) and isinstance(row.get("task_name"), str):
                validated_summary[row["task_name"]] = row
    commands = _load_commands(commands_path)
    output_tasks: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda row: str(row.get("task_name", ""))):
        task_name = str(task.get("task_name") or "")
        pipeline_report = task.get("pipeline_report")
        trajectory_path, instruction = (
            _instruction_from_pipeline_report(Path(pipeline_report))
            if isinstance(pipeline_report, str) and pipeline_report
            else ("", "")
        )
        top_commands = _ranked_commands(commands.get(task_name, []))
        dominant = top_commands[0] if top_commands else None
        command_text = dominant["command_preview"] if dominant else ""
        valid = task.get("status") == "passed"
        reviewed = REVIEWED_CPU_EXPLANATIONS.get(task_name)
        mechanism_only_reviewed = MECHANISM_ONLY_REVIEWED_CPU_EXPLANATIONS.get(task_name)
        excerpt = _instruction_excerpt(instruction)
        category = reviewed[0] if reviewed else _workload_category(task_name, instruction, command_text)
        reason = reviewed[1] if reviewed and valid else _generic_reason(category, dominant, valid=valid)
        mechanism_only_reason = (
            mechanism_only_reviewed[1] if mechanism_only_reviewed else _mechanism_only_reason(category, dominant)
        )
        output_tasks.append(
            {
                "task_name": task_name,
                "display_name": _display_name(task_name),
                "status": task.get("status"),
                "measurement_quality": task.get("distribution_quality"),
                "task": {
                    "what_it_does": excerpt,
                    "instruction_excerpt": excerpt,
                    "instruction": instruction,
                    "source_trajectory": trajectory_path,
                },
                "cpu": {
                    "allocated_logical_cpus": task.get("allocated_logical_cpus"),
                    "wall_sec": task.get("wall_sec"),
                    "cpu_usage_sec": task.get("cpu_usage_sec"),
                    "average_logical_cores_used": task.get("average_logical_cores_used"),
                    "peak_simultaneous_logical_cores": task.get("scheduler_peak_simultaneous_logical_cores"),
                    "exact_concurrency_valid": bool(task.get("included_in_exact_concurrency_distribution")),
                    "failure_reasons": task.get("failure_reasons", []),
                    "burst_class": _duration_class(task, validated_summary),
                    "top_command_drivers": top_commands,
                },
                "explanation": {
                    "workload_category": category,
                    "why_it_uses_cores": reason,
                    "why_it_uses_cores_mechanism_only": mechanism_only_reason,
                    "method": (
                        "reviewed_from_prompt_and_command_evidence"
                        if reviewed and valid
                        else "rule_based_from_prompt_and_command_evidence"
                    ),
                },
            }
        )
    passed = sum(task["status"] == "passed" for task in output_tasks)
    diagnostic = len(output_tasks) - passed
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "task_analysis": str(analysis_path.resolve()),
            "validated_summary": (str(validated_summary_path.resolve()) if validated_summary_path else None),
            "validated_commands": str(commands_path.resolve()) if commands_path else None,
        },
        "summary": {
            "task_count": len(output_tasks),
            "validated_task_count": passed,
            "diagnostic_only_task_count": diagnostic,
            "tasks_with_command_attribution": sum(bool(task["cpu"]["top_command_drivers"]) for task in output_tasks),
            "notes": [
                "Exact concurrency and command attribution are intended for passed validated replays.",
                "Failed-validation tasks retain their task instruction and observation-window CPU fields, but their peaks are diagnostic only.",
            ],
        },
        "tasks": output_tasks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Terminal-Bench task and CPU explanation data.")
    parser.add_argument("--task-analysis", type=Path, required=True)
    parser.add_argument("--validated-summary", type=Path)
    parser.add_argument("--commands", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        catalog = build_catalog(args.task_analysis, args.validated_summary, args.commands)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    except TaskCatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Tasks: {catalog['summary']['task_count']}")
    print(f"Validated: {catalog['summary']['validated_task_count']}")
    print(f"Diagnostic only: {catalog['summary']['diagnostic_only_task_count']}")
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

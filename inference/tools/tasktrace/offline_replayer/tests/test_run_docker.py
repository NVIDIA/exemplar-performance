# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-Docker E2E for the small public YAML-run interface.

The test uses an already-local image and never pulls::

    TASKTRACE_DOCKER_RUN_E2E=1 \
      TASKTRACE_DOCKER_TEST_IMAGE=python:3.12-slim \
      uv run --extra test python -m pytest -q \
      offline_replayer/tests/test_run_docker.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from offline_replayer import cli

ENABLED = os.environ.get("TASKTRACE_DOCKER_RUN_E2E") == "1"
IMAGE = os.environ.get("TASKTRACE_DOCKER_TEST_IMAGE")

pytestmark = pytest.mark.skipif(
    not ENABLED or not IMAGE,
    reason=("set TASKTRACE_DOCKER_RUN_E2E=1 and TASKTRACE_DOCKER_TEST_IMAGE " "to run the real-Docker YAML-run E2E"),
)


@pytest.fixture(scope="module")
def local_image_id() -> str:
    assert IMAGE is not None
    assert shutil.which("docker") is not None, "Docker CLI is required"
    inspected = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        text=True,
        capture_output=True,
        check=False,
    )
    assert inspected.returncode == 0, (
        f"pre-pull {IMAGE!r}; this E2E never pulls images: " f"{(inspected.stderr or inspected.stdout).strip()}"
    )
    payload = json.loads(inspected.stdout)
    image_id = payload[0].get("Id") if isinstance(payload, list) and payload else None
    assert isinstance(image_id, str) and image_id.startswith("sha256:")
    return image_id


def _write_atif(path: Path, task_id: str, *, workload_sec: float = 0.12) -> str:
    workload = "\n".join(
        (
            "import time",
            f"deadline = time.monotonic() + {workload_sec!r}",
            "value = 0",
            "while time.monotonic() < deadline:",
            "    value += 1",
            f"print({task_id!r}, value > 0)",
        )
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.5",
                "session_id": task_id,
                "agent": {"name": "sequential-run-e2e"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-07-02T00:00:00+00:00",
                        "tool_calls": [
                            {
                                "function_name": "exec_command",
                                "arguments": {"cmd": "python3 -c " + shlex.quote(workload)},
                            }
                        ],
                        "observation": {"content": "Process exited with code 0"},
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict)
    return payload


_PROFILE_CONTAINER_NAME = re.compile(r"tasktrace-profile-[0-9a-f]{12}")


def _profile_container_ids_for_output(
    output: Path,
    *,
    include_stopped: bool = False,
) -> set[str]:
    """Find only profiler containers whose output bind belongs to this run."""
    command = ["docker", "ps"]
    if include_stopped:
        command.append("--all")
    command.extend(["--format", "{{.ID}}\t{{.Names}}"])
    listed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stderr.strip()

    output = output.resolve()
    matched: set[str] = set()
    for line in listed.stdout.splitlines():
        candidate_id, separator, listed_name = line.partition("\t")
        if not separator or _PROFILE_CONTAINER_NAME.fullmatch(listed_name) is None:
            continue
        inspected = subprocess.run(
            ["docker", "inspect", candidate_id],
            text=True,
            capture_output=True,
            check=False,
        )
        if inspected.returncode != 0:
            # A container may disappear normally between `ps` and `inspect`.
            if "No such object" in inspected.stderr or "No such container" in inspected.stderr:
                continue
            raise AssertionError(inspected.stderr.strip())
        payload = json.loads(inspected.stdout)
        assert isinstance(payload, list) and len(payload) == 1
        container = payload[0]
        assert isinstance(container, dict)
        inspected_name = str(container.get("Name", "")).removeprefix("/")
        if _PROFILE_CONTAINER_NAME.fullmatch(inspected_name) is None:
            continue
        mounts = container.get("Mounts")
        if not isinstance(mounts, list):
            continue
        for mount in mounts:
            if not isinstance(mount, dict):
                continue
            if mount.get("Destination") != "/tasktrace-profile-output":
                continue
            source_value = mount.get("Source")
            if not isinstance(source_value, str):
                continue
            source = Path(source_value).resolve()
            if source == output or output in source.parents:
                container_id = container.get("Id")
                assert isinstance(container_id, str) and container_id
                matched.add(container_id)
                break
    return matched


def test_public_run_executes_two_tasks_twice_sequentially(
    tmp_path: Path,
    local_image_id: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert IMAGE is not None
    tasks = []
    source_digests: dict[str, str] = {}
    for index in range(2):
        task_id = f"task-{index}"
        atif = tmp_path / f"{task_id}.json"
        source_digests[task_id] = _write_atif(atif, task_id)
        tasks.append(
            {
                "id": task_id,
                "atif": atif.name,
                "image": IMAGE,
                "expected_source_sha256": source_digests[task_id],
                "expected_image_id": local_image_id,
                "action_timeout_sec": 20,
            }
        )
    output = tmp_path / "results"
    manifest = tmp_path / "run.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "tasktrace-run.v1",
                "output": output.name,
                "cpus": str(min(os.sched_getaffinity(0))),
                "repetitions": 2,
                "tasks": tasks,
            },
            sort_keys=False,
        )
    )

    assert cli.main(["run", str(manifest)]) == 0
    assert "Completed: 4; failed: 0; valid durations: 4" in capsys.readouterr().out

    with (output / "results.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [(row["task_id"], int(row["repetition"])) for row in rows] == [
        ("task-0", 1),
        ("task-1", 1),
        ("task-0", 2),
        ("task-1", 2),
    ]
    assert all(row["status"] == "completed" for row in rows)
    assert all(row["duration_valid"] == "True" for row in rows)
    assert all(float(row["duration_sec"]) > 0 for row in rows)
    assert all(row["image_id"] == local_image_id for row in rows)

    reports: list[dict[str, Any]] = []
    for row in rows:
        profile = output / row["profile_dir"]
        report = _object(profile / "pipeline_report.json")
        reports.append(report)
        assert report["source_trajectory_sha256"] == source_digests[row["task_id"]]
        assert report["image_identity"]["id"] == local_image_id
        cpu = report["container_cpu"]
        assert cpu["schema_version"] == "cgroup-cpu-coordinated.v1"
        assert cpu["valid_for_cpu_performance"] is True
        assert cpu["window_end_ns"] > cpu["window_start_ns"]
        assert cpu["wall_sec"] == pytest.approx(float(row["duration_sec"]))
        assert report["canonical_clock_reference"]["clock_id"] == "CLOCK_MONOTONIC"
        assert report["replay_summary"]["actions_timed_out"] == 0
        assert report["replay_summary"]["source_outcome_mismatch_count"] == 0
        assert report["profiling_valid"] is True
        assert row["execution_plan_digest"] == report["execution_plan_digest"]
    windows = [
        (report["container_cpu"]["window_start_ns"], report["container_cpu"]["window_end_ns"]) for report in reports
    ]
    assert all(left[1] <= right[0] for left, right in zip(windows, windows[1:], strict=False))

    summary = _object(output / "summary.json")
    assert summary["duration_clock"] == "host-monotonic"
    assert summary["host_cpu"]["cpu_model"]
    assert [cpu["logical_cpu"] for cpu in summary["host_cpu"]["selected_cpus"]] == [min(os.sched_getaffinity(0))]
    assert summary["controller_runtime_content_bound"] is False
    assert summary["overall"]["attempt_count"] == 4
    assert summary["overall"]["completed"] == 4
    assert summary["overall"]["failed"] == 0
    assert summary["overall"]["valid_duration_count"] == 4
    assert summary["overall"]["valid_duration_total_sec"] > 0
    assert summary["overall"]["replay_tasks_per_sec"] > 0
    assert [task["valid_duration_count"] for task in summary["tasks"]] == [2, 2]
    assert all(task["median_duration_sec"] > 0 for task in summary["tasks"])
    assert all(len(task["execution_plan_digests"]) == 1 for task in summary["tasks"])


def test_public_v2_run_overlaps_fresh_containers_and_aggregates_cgroups(
    tmp_path: Path,
    local_image_id: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert IMAGE is not None
    available_cpus = sorted(os.sched_getaffinity(0))
    selected_cpus = available_cpus[: min(2, len(available_cpus))]
    tasks = []
    for index in range(4):
        task_id = f"parallel-task-{index}"
        atif = tmp_path / f"{task_id}.json"
        source_sha256 = _write_atif(atif, task_id, workload_sec=0.8)
        tasks.append(
            {
                "id": task_id,
                "atif": atif.name,
                "image": IMAGE,
                "expected_source_sha256": source_sha256,
                "expected_image_id": local_image_id,
                "action_timeout_sec": 20,
            }
        )
    output = tmp_path / "parallel-results"
    manifest = tmp_path / "parallel-run.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "tasktrace-run.v2",
                "output": output.name,
                "cpus": ",".join(str(cpu) for cpu in selected_cpus),
                "concurrency": 2,
                "repetitions": 1,
                "tasks": tasks,
            },
            sort_keys=False,
        )
    )

    assert cli.main(["run", str(manifest)]) == 0
    stdout = capsys.readouterr().out
    assert "Completed: 4; failed: 0; valid durations: 4" in stdout
    assert "Throughput:" in stdout

    summary = _object(output / "summary.json")
    overall = summary["overall"]
    assert summary["schema_version"] == "tasktrace-run-summary.v2"
    assert summary["concurrency"] == 2
    assert overall["valid_for_throughput"] is True
    assert overall["achieved_peak_measured_attempts"] == 2
    assert overall["requested_concurrency_sec"] > 0
    assert overall["requested_concurrency_fraction"] > 0
    assert overall["throughput_tasks_per_sec"] > 0
    assert overall["total_cgroup_cpu_sec"] > 0
    assert overall["aggregate_average_logical_cores"] > 0
    assert overall["pool_utilization_percent"] > 0
    assert overall["batch_wall_sec"] < overall["summed_service_time_sec"]

    with (output / "results.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [int(row["attempt_index"]) for row in rows] == [1, 2, 3, 4]
    assert sorted(int(row["completion_index"]) for row in rows) == [1, 2, 3, 4]
    assert all(float(row["cpu_usage_sec"]) > 0 for row in rows)


def test_public_v3_run_enforces_explicit_placements_and_wave_barrier(
    tmp_path: Path,
    local_image_id: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert IMAGE is not None
    caller_affinity = set(os.sched_getaffinity(0))
    available_cpus = sorted(caller_affinity)
    if len(available_cpus) < 3:
        pytest.skip("v3 Docker E2E needs one controller and two worker CPUs")
    controller, worker_a, worker_b = available_cpus[:3]

    tasks = []
    for index in range(2):
        task_id = f"wave-task-{index}"
        atif = tmp_path / f"{task_id}.json"
        source_sha256 = _write_atif(atif, task_id, workload_sec=0.5)
        tasks.append(
            {
                "id": task_id,
                "atif": atif.name,
                "image": IMAGE,
                "expected_source_sha256": source_sha256,
                "expected_image_id": local_image_id,
            }
        )

    output = tmp_path / "wave-results"
    manifest = tmp_path / "wave-run.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "tasktrace-run.v3",
                "output": output.name,
                "defaults": {
                    "pacing": "llm-gaps",
                    "network_policy": "public",
                    "profile_sets": ["cpu-average"],
                    "controller_runtime": "host-mount",
                    "controller_cpus": str(controller),
                    "worker_cpus": f"{worker_a},{worker_b}",
                    "placement": "round-robin-single-cpu",
                    "action_timeout_sec": 20,
                },
                "tasks": tasks,
                "waves": [
                    {
                        "id": "n1-v1",
                        "allocation": 1,
                        "repeat": 1,
                    },
                    {
                        "id": "n2-v1",
                        "allocation": 2,
                        "repeat": 1,
                    },
                ],
            },
            sort_keys=False,
        )
    )

    assert cli.main(["run", str(manifest)]) == 0
    assert os.sched_getaffinity(0) == caller_affinity
    assert "Completed: 4; failed: 0; valid durations: 4" in capsys.readouterr().out

    summary = _object(output / "summary.json")
    assert summary["schema_version"] == "tasktrace-run-summary.v3"
    assert summary["overall"]["valid_for_analysis"] is True
    assert summary["controller_affinity"]["caller_affinity_restored"] is True
    assert summary["scheduling"]["authoring_mode"] == "compact-worker-pool"
    assert [wave["logical_cpu_count"] for wave in summary["waves"]] == [1, 2]
    assert all(wave["overall"]["valid_for_throughput"] is True for wave in summary["waves"])

    with (output / "results.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["wave_id"] for row in rows] == [
        "n1-v1",
        "n1-v1",
        "n2-v1",
        "n2-v1",
    ]
    assert [row["cpus"] for row in rows] == [
        str(worker_a),
        str(worker_a),
        str(worker_a),
        str(worker_b),
    ]
    wave_one_end = max(int(row["window_end_ns"]) for row in rows[:2])
    wave_two_start = min(int(row["window_start_ns"]) for row in rows[2:])
    assert wave_one_end <= wave_two_start

    resolved = yaml.safe_load((output / "resolved-run.yaml").read_text())
    assert "worker_cpus" not in resolved["defaults"]
    assert "placement" not in resolved["defaults"]
    assert resolved["waves"][1]["launches"] == [
        {"task": "wave-task-0", "cpus": str(worker_a)},
        {"task": "wave-task-1", "cpus": str(worker_b)},
    ]


def test_public_v2_sigint_cancels_profiles_and_removes_its_containers(
    tmp_path: Path,
    local_image_id: str,
) -> None:
    assert IMAGE is not None
    public_cli = shutil.which("tasktrace")
    assert public_cli is not None, "the installed tasktrace CLI is required"

    available_cpus = sorted(os.sched_getaffinity(0))
    selected_cpus = available_cpus[: min(2, len(available_cpus))]
    tasks = []
    for index in range(2):
        task_id = f"interrupt-task-{index}"
        atif = tmp_path / f"{task_id}.json"
        source_sha256 = _write_atif(atif, task_id, workload_sec=120.0)
        tasks.append(
            {
                "id": task_id,
                "atif": atif.name,
                "image": IMAGE,
                "expected_source_sha256": source_sha256,
                "expected_image_id": local_image_id,
                "action_timeout_sec": 300,
            }
        )

    output = tmp_path / "interrupt-results"
    manifest = tmp_path / "interrupt-run.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "tasktrace-run.v2",
                "output": output.name,
                "cpus": ",".join(str(cpu) for cpu in selected_cpus),
                "concurrency": 2,
                "repetitions": 1,
                "tasks": tasks,
            },
            sort_keys=False,
        )
    )

    process = subprocess.Popen(
        [public_cli, "run", str(manifest)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout = ""
    stderr = ""
    try:
        deadline = time.monotonic() + 45
        running: set[str] = set()
        while time.monotonic() < deadline:
            running = _profile_container_ids_for_output(output)
            if len(running) == 2:
                break
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(
                    "run exited before two scoped profiler containers appeared:\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            time.sleep(0.1)
        assert len(running) == 2, (
            "timed out waiting for two running profiler containers scoped to " f"{output}; observed {sorted(running)}"
        )

        process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            pytest.fail("tasktrace did not finish cleanup within 30s of SIGINT")
        assert process.returncode == 130, f"stdout:\n{stdout}\nstderr:\n{stderr}"

        summary = _object(output / "summary.json")
        assert summary["schema_version"] == "tasktrace-run-summary.v2"
        overall = summary["overall"]
        assert overall["interrupted"] is True
        assert overall["valid_for_throughput"] is False
        assert overall["throughput_tasks_per_sec"] is None
        assert overall["pool_utilization_percent"] is None

        cleanup_deadline = time.monotonic() + 10
        remaining = _profile_container_ids_for_output(
            output,
            include_stopped=True,
        )
        while remaining and time.monotonic() < cleanup_deadline:
            time.sleep(0.1)
            remaining = _profile_container_ids_for_output(
                output,
                include_stopped=True,
            )
        assert not remaining, "SIGINT cleanup left scoped profiler containers behind: " f"{sorted(remaining)}"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        leaked = _profile_container_ids_for_output(output, include_stopped=True)
        if leaked:
            subprocess.run(
                ["docker", "rm", "-f", *sorted(leaked)],
                text=True,
                capture_output=True,
                check=False,
            )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest
import yaml
from offline_replayer.runs import execute
from offline_replayer.runs.execute import RunExecutionError, execute_run
from offline_replayer.runs.host import HostIdentityError
from offline_replayer.runs.spec import load_run_spec


@pytest.fixture(autouse=True)
def _host_cpu_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        execute,
        "collect_host_cpu_identity",
        lambda cpu_ids: {
            "schema_version": "tasktrace-run-host.v1",
            "machine": "x86_64",
            "kernel_release": "test",
            "cpu_vendor": "Test",
            "cpu_model": "Test CPU",
            "microcode": None,
            "host_logical_cpu_count": 128,
            "selected_cpus": [
                {
                    "logical_cpu": cpu_id,
                    "package_id": 0,
                    "core_id": cpu_id,
                    "thread_siblings": [cpu_id],
                    "numa_node": 0,
                }
                for cpu_id in cpu_ids
            ],
        },
    )


def _manifest(
    tmp_path: Path,
    *,
    task_count: int = 2,
    repetitions: int = 2,
    expected_images: bool = False,
    schema_version: str = "tasktrace-run.v1",
    concurrency: int | None = None,
) -> Path:
    tasks = []
    for index in range(task_count):
        source = tmp_path / f"task-{index}.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": "ATIF-v1.5",
                    "session_id": f"task-{index}",
                    "steps": [],
                }
            )
            + "\n"
        )
        task = {
            "id": f"task-{index}",
            "atif": source.name,
            "image": f"image-{index}:latest",
        }
        if expected_images:
            task["expected_image_id"] = f"sha256:image-{index}"
        tasks.append(task)
    manifest = tmp_path / "run.yaml"
    payload = {
        "schema_version": schema_version,
        "output": "results",
        "cpus": "20-23",
        "repetitions": repetitions,
        "tasks": tasks,
    }
    if concurrency is not None:
        payload["concurrency"] = concurrency
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
    return manifest


def _image_index(config) -> str:
    assert config.image is not None
    return config.image.removeprefix("image-").partition(":")[0]


def _preflight(config) -> dict:
    index = _image_index(config)
    return {
        "ready": True,
        "source_sha256": hashlib.sha256(config.trajectory.read_bytes()).hexdigest(),
        "image_identity": {"id": f"sha256:image-{index}"},
        "profile_sets": list(config.profile_sets),
        "cpuset_cpus": config.cpuset_cpus,
    }


def _report(
    config,
    duration: float,
    *,
    window_start_ns: int | None = None,
    cpu_usage_sec: float | None = None,
) -> dict:
    index = _image_index(config)
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "pipeline_report.json").write_text("{}\n")
    cpu = {
        "wall_sec": duration,
        "valid_for_cpu_performance": True,
    }
    if window_start_ns is not None:
        usage = duration if cpu_usage_sec is None else cpu_usage_sec
        cpu.update(
            {
                "window_start_ns": window_start_ns,
                "window_end_ns": window_start_ns + round(duration * 1_000_000_000),
                "usage_sec": usage,
                "average_logical_cores": usage / duration,
            }
        )
    return {
        "source_trajectory_sha256": hashlib.sha256(config.trajectory.read_bytes()).hexdigest(),
        "image_identity": {"id": f"sha256:image-{index}"},
        "container_cpu": cpu,
        "replay_summary": {
            "actions_timed_out": 0,
            "source_outcome_mismatch_count": 0,
        },
        "profiling_valid": True,
        "status": "valid",
        "execution_plan_digest": hashlib.sha256(f"plan-{index}".encode()).hexdigest(),
    }


def _ordered_manifest(tmp_path: Path) -> tuple[Path, tuple[int, int, int]]:
    available = sorted(os.sched_getaffinity(0))
    assert len(available) >= 3
    controller, worker_a, worker_b = available[:3]
    tasks = []
    for index in range(2):
        source = tmp_path / f"ordered-{index}.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": "ATIF-v1.5",
                    "session_id": f"task-{index}",
                    "steps": [],
                }
            )
            + "\n"
        )
        tasks.append(
            {
                "id": f"task-{index}",
                "atif": source.name,
                "image": f"image-{index}:latest",
            }
        )
    payload = {
        "schema_version": "tasktrace-run.v3",
        "output": "ordered-results",
        "defaults": {
            "pacing": "llm-gaps",
            "network_policy": "public",
            "profile_sets": ["cpu-average"],
            "controller_runtime": "host-mount",
            "controller_cpus": str(controller),
            "action_timeout_sec": 90,
        },
        "tasks": tasks,
        "waves": [
            {
                "id": "r1-n1",
                "scored": True,
                "metadata": {"repeat": 1, "allocation": 1},
                "launches": [
                    {"task": "task-0", "cpus": str(worker_a)},
                    {"task": "task-1", "cpus": str(worker_a)},
                ],
            },
            {
                "id": "r1-n2",
                "scored": True,
                "metadata": {"repeat": 1, "allocation": 2},
                "launches": [
                    {"task": "task-0", "cpus": str(worker_b)},
                    {"task": "task-1", "cpus": str(worker_b)},
                ],
            },
        ],
    }
    manifest = tmp_path / "ordered.yaml"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
    return manifest, (controller, worker_a, worker_b)


def _enroot_ordered_spec(tmp_path: Path):
    manifest, _cpus = _ordered_manifest(tmp_path)
    payload = yaml.safe_load(manifest.read_text())
    payload["defaults"].update(
        {
            "runtime": "enroot",
            "network_policy": "no-network",
            "profile_sets": [],
        }
    )
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False))
    return load_run_spec(manifest)


def test_enroot_ordered_preflight_hashes_each_task_image_once_across_placements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _enroot_ordered_spec(tmp_path)
    calls: list[tuple[str, str]] = []

    def preflight(config):
        calls.append((config.task_name, config.cpuset_cpus))
        return _preflight(config)

    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", preflight)

    results = execute._preflight_ordered(spec, execute._wave_attempts(spec))

    assert len(results) == 4
    assert [task for task, _cpus in calls] == ["task-0", "task-1"]


def _ordered_report(config, *, start_ns: int, duration: float = 1.0) -> dict:
    report = _report(
        config,
        duration,
        window_start_ns=start_ns,
        cpu_usage_sec=0.5,
    )
    report["container_cpu"].update(
        {
            "allocated_cpu_source": "cpuset-cpus",
            "allocated_logical_cpus": 1.0,
        }
    )
    report.update(
        {
            "pacing": config.pacing,
            "controller_runtime": config.controller_runtime,
            "profile_provenance": {"profile_sets": list(config.profile_sets)},
            "network_policy": {"comparability_sha256": config.network_policy.comparability_digest},
            "execution_plan": {"extensions": {"cpuset_cpus": config.cpuset_cpus}},
            "execution_plan_digest": hashlib.sha256(
                f"plan-{_image_index(config)}-{config.cpuset_cpus}".encode()
            ).hexdigest(),
        }
    )
    return report


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ENROOT_CONFIG_PATH", "/unsealed"),
        ("ENROOT_SYSCONF_PATH", ""),
        ("ENROOT_MOUNT_HOME", "yes"),
        ("ENROOT_FUTURE_SETTING", "yes"),
        ("BASH_ENV", "/unsealed-shell"),
        ("PYTHONHOME", "/unsealed-python"),
    ],
)
def test_enroot_ordered_run_rejects_unsealed_ambient_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    spec = _enroot_ordered_spec(tmp_path)
    monkeypatch.setenv(name, value)

    with pytest.raises(RunExecutionError, match=name) as raised:
        execute_run(spec)

    if value:
        assert value not in str(raised.value)
    assert not spec.output.exists()


def test_ordered_runtime_environment_allows_state_paths_and_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enroot_spec = _enroot_ordered_spec(tmp_path)
    for inherited in tuple(os.environ):
        if inherited in execute.ENROOT_FORBIDDEN_ENVIRONMENT or (
            inherited.startswith("ENROOT_") and inherited not in execute.ENROOT_STATE_ENVIRONMENT
        ):
            monkeypatch.delenv(inherited)
    for name in ("ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_RUNTIME_PATH"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    execute._validate_ordered_runtime_environment(enroot_spec)

    docker_root = tmp_path / "docker"
    docker_root.mkdir()
    docker_manifest, _cpus = _ordered_manifest(docker_root)
    docker_spec = load_run_spec(docker_manifest)
    monkeypatch.setenv("ENROOT_CONFIG_PATH", "/ignored-for-docker")
    execute._validate_ordered_runtime_environment(docker_spec)


def test_enroot_summary_invalidates_duration_outside_outer_wave_first(
    tmp_path: Path,
) -> None:
    spec = _enroot_ordered_spec(tmp_path)
    rows = [
        {
            "attempt_index": index,
            "wave_index": 1 if index <= 2 else 2,
            "task_id": f"task-{(index - 1) % 2}",
            "status": "completed",
            "duration_valid": True,
            "duration_sec": 30.0 if index == 1 else 1.0,
            "reason": "",
            "image_id": f"sha256:image-{(index - 1) % 2}",
            "execution_plan_digest": f"plan-{index}",
        }
        for index in range(1, 5)
    ]

    summary = execute._ordered_summary(
        spec,
        rows,
        host_cpu={},
        interrupted=False,
        outer_run_wall_sec=8.0,
        wave_outer_wall_sec={1: 4.0, 2: 4.0},
        controller_affinity={},
    )

    assert rows[0]["duration_valid"] is False
    assert "lifecycle-inclusive outer wave bound" in rows[0]["reason"]
    assert summary["duration_clock"] == "linux-proc-uptime"
    assert summary["tasks"][0]["valid_duration_count"] == 1
    assert summary["waves"][0]["overall"]["valid_duration_count"] == 1
    assert summary["waves"][0]["overall"]["valid_for_throughput"] is False
    assert summary["waves"][0]["overall"]["wall_only_throughput_tasks_per_sec"] is None
    assert summary["overall"]["valid_duration_count"] == 3
    assert summary["overall"]["valid_for_analysis"] is False


def test_enroot_summary_reports_requested_taskset_and_requires_full_overlap(
    tmp_path: Path,
) -> None:
    spec = _enroot_ordered_spec(tmp_path)
    clock_reference = {
        "clock_name": "CLOCK_BOOTTIME",
        "boot_id": "test-boot",
        "time_namespace": "time:[1]",
    }
    windows = {
        1: (0, 1_000_000_000),
        2: (0, 1_000_000_000),
        3: (2_000_000_000, 3_000_000_000),
        4: (3_000_000_000, 4_000_000_000),
    }
    rows = [
        {
            "attempt_index": index,
            "wave_index": 1 if index <= 2 else 2,
            "task_id": f"task-{(index - 1) % 2}",
            "status": "completed",
            "duration_valid": True,
            "duration_sec": 1.0,
            "reason": "",
            "image_id": f"sha256:image-{(index - 1) % 2}",
            "execution_plan_digest": f"plan-{index}",
            "window_start_ns": windows[index][0],
            "window_end_ns": windows[index][1],
            "clock_reference": clock_reference,
        }
        for index in range(1, 5)
    ]

    summary = execute._ordered_summary(
        spec,
        rows,
        host_cpu={},
        interrupted=False,
        outer_run_wall_sec=5.0,
        wave_outer_wall_sec={1: 2.0, 2: 3.0},
        controller_affinity={},
    )

    scheduling = summary["scheduling"]
    assert scheduling["container_cpu_placement"] == ("taskset-requested-per-launch-affinity")
    assert scheduling["measured_start_alignment"] == "observed-not-synchronized"
    assert scheduling["cpu_exclusivity_verified"] is False

    overlapping, nonoverlapping = summary["waves"]
    assert overlapping["overall"]["achieved_peak_measured_attempts"] == 2
    assert overlapping["overall"]["valid_for_throughput"] is True
    assert nonoverlapping["overall"]["achieved_peak_measured_attempts"] == 1
    assert nonoverlapping["overall"]["valid_for_throughput"] is False
    assert nonoverlapping["overall"]["throughput_invalid_reasons"] == [
        "replay commands did not achieve all-launch overlap: peak=1, expected=2"
    ]
    assert summary["overall"]["valid_for_wall_only_analysis"] is False
    assert summary["overall"]["valid_for_cpu_analysis"] is False
    assert "valid_for_fixed_core_analysis" not in summary["overall"]
    assert all("valid_for_fixed_core_analysis" not in wave["overall"] for wave in summary["waves"])


def test_preflights_all_then_runs_declared_order_and_writes_small_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_run_spec(_manifest(tmp_path, expected_images=True))
    events: list[tuple[str, str]] = []
    durations = iter((1.0, 2.0, 3.0, 4.0))

    def preflight(config):
        events.append(("preflight", config.task_name))
        assert config.cpuset_cpus == "20-23"
        assert config.pacing == "unpaced"
        assert config.controller_runtime == "host-mount"
        assert config.strace_mode == "none"
        assert config.runtime == "docker"
        assert config.profile_sets == ("cpu-average",)
        return _preflight(config)

    def profile(config, **_kwargs):
        events.append(("execute", config.task_name))
        return _report(config, next(durations))

    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", preflight)
    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 0
    assert events == [
        ("preflight", "task-0"),
        ("preflight", "task-1"),
        ("execute", "task-0"),
        ("execute", "task-1"),
        ("execute", "task-0"),
        ("execute", "task-1"),
    ]
    assert [row["duration_sec"] for row in result.rows] == [1.0, 2.0, 3.0, 4.0]
    assert all(row["duration_valid"] is True for row in result.rows)
    assert all("cleanup_valid" not in row for row in result.rows)
    assert all("clock_reference" not in row for row in result.rows)
    assert result.summary["tasks"][0]["median_duration_sec"] == 2.0
    assert result.summary["tasks"][1]["median_duration_sec"] == 3.0
    assert result.summary["tasks"][0]["source_sha256"] == spec.tasks[0].source_sha256
    assert result.summary["tasks"][0]["image_ids"] == ["sha256:image-0"]
    assert result.summary["tasks"][0]["execution_plan_digests"] == [hashlib.sha256(b"plan-0").hexdigest()]
    assert result.summary["overall"] == {
        "attempt_count": 4,
        "completed": 4,
        "failed": 0,
        "valid_duration_count": 4,
        "valid_duration_total_sec": 10.0,
        "replay_tasks_per_sec": 0.4,
    }
    assert result.summary["duration_clock"] == "host-monotonic"
    assert result.summary["host_cpu"]["cpu_model"] == "Test CPU"
    assert result.summary["controller_runtime_content_bound"] is False

    with (spec.output / "results.csv").open(newline="") as source:
        csv_rows = list(csv.DictReader(source))
    assert tuple(csv_rows[0]) == execute.RESULT_COLUMNS
    assert [row["profile_dir"] for row in csv_rows] == [
        "tasks/task-0/rep-0001/profile",
        "tasks/task-1/rep-0001/profile",
        "tasks/task-0/rep-0002/profile",
        "tasks/task-1/rep-0002/profile",
    ]
    summary = json.loads((spec.output / "summary.json").read_text())
    assert summary["overall"]["valid_duration_count"] == 4
    assert (spec.output / "tasks/task-1/rep-0002/profile/pipeline_report.json").is_file()


def test_v2_runs_bounded_shared_cpuset_and_reports_common_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_run_spec(
        _manifest(
            tmp_path,
            task_count=4,
            repetitions=1,
            schema_version="tasktrace-run.v2",
            concurrency=2,
        )
    )
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    barriers = (threading.Barrier(2), threading.Barrier(2))
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def profile(config, **_kwargs):
        nonlocal active, peak_active
        index = int(_image_index(config))
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        barriers[index // 2].wait(timeout=2)
        time.sleep(0.01)
        with lock:
            active -= 1
        return _report(
            config,
            2.0,
            window_start_ns=(index // 2) * 2_000_000_000,
            cpu_usage_sec=1.5,
        )

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 0
    assert peak_active == 2
    assert [row["attempt_index"] for row in result.rows] == [1, 2, 3, 4]
    assert sorted(row["completion_index"] for row in result.rows) == [1, 2, 3, 4]
    overall = result.summary["overall"]
    assert result.summary["schema_version"] == "tasktrace-run-summary.v2"
    assert result.summary["concurrency"] == 2
    assert result.summary["scheduling"]["completion_index_semantics"] == ("worker-return-order")
    assert overall["valid_for_throughput"] is True
    assert overall["batch_wall_sec"] == pytest.approx(4.0)
    assert overall["throughput_tasks_per_sec"] == pytest.approx(1.0)
    assert overall["total_cgroup_cpu_sec"] == pytest.approx(6.0)
    assert overall["aggregate_average_logical_cores"] == pytest.approx(1.5)
    assert overall["pool_utilization_percent"] == pytest.approx(37.5)
    assert overall["achieved_peak_measured_attempts"] == 2
    assert overall["time_sec_by_measured_attempts"] == {"2": 4.0}
    assert overall["requested_concurrency_fraction"] == pytest.approx(1.0)

    with (spec.output / "results.csv").open(newline="") as source:
        csv_rows = list(csv.DictReader(source))
    assert tuple(csv_rows[0]) == execute.RESULT_COLUMNS_V2


def test_v2_failure_suppresses_common_window_throughput(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(
        _manifest(
            tmp_path,
            task_count=2,
            repetitions=1,
            schema_version="tasktrace-run.v2",
            concurrency=2,
        )
    )
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)

    def profile(config, **_kwargs):
        report = _report(
            config,
            1.0,
            window_start_ns=0,
            cpu_usage_sec=0.9,
        )
        if config.task_name == "task-1":
            report["replay_summary"]["source_outcome_mismatch_count"] = 1
        return report

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 1
    assert result.summary["overall"]["valid_for_throughput"] is False
    assert result.summary["overall"]["throughput_tasks_per_sec"] is None
    assert result.summary["overall"]["pool_utilization_percent"] is None
    assert result.summary["overall"]["measured_window_count"] == 2
    assert result.summary["overall"]["achieved_peak_measured_attempts"] == 2
    assert result.summary["overall"]["time_sec_by_measured_attempts"] == {"2": 1.0}


def test_v2_impossible_aggregate_cpu_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(
        _manifest(
            tmp_path,
            task_count=2,
            repetitions=1,
            schema_version="tasktrace-run.v2",
            concurrency=2,
        )
    )
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    monkeypatch.setattr(
        execute,
        "run_atif_trajectory_profile",
        lambda config, **_kwargs: _report(
            config,
            1.0,
            window_start_ns=0,
            cpu_usage_sec=3.0,
        ),
    )

    result = execute_run(spec)

    overall = result.summary["overall"]
    assert result.exit_code == 1
    assert all(row["duration_valid"] is True for row in result.rows)
    assert overall["valid_for_throughput"] is False
    assert overall["throughput_tasks_per_sec"] is None
    assert overall["total_cgroup_cpu_sec"] is None
    assert overall["pool_utilization_percent"] is None
    assert any("exceeds declared pool capacity" in reason for reason in overall["throughput_invalid_reasons"])


def test_v3_dispatches_exact_ordered_waves_with_per_launch_cpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, (controller, worker_a, worker_b) = _ordered_manifest(tmp_path)
    spec = load_run_spec(manifest)
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    lock = threading.Lock()
    active_by_wave = {1: 0, 2: 0}
    completed_by_wave = {1: 0, 2: 0}
    barriers = {1: threading.Barrier(2), 2: threading.Barrier(2)}

    def profile(config, **_kwargs):
        assert (spec.output / "resolved-run.yaml").is_file()
        wave_index = 1 if "0001-r1-n1" in str(config.output_dir) else 2
        expected_cpu = str(worker_a if wave_index == 1 else worker_b)
        assert config.cpuset_cpus == expected_cpu
        assert config.pacing == "llm-gaps"
        assert config.controller_runtime == "host-mount"
        assert config.timeout_sec == 90.0
        assert config.network_policy.mode.value == "public"
        assert os.sched_getaffinity(0) == {controller}
        with lock:
            if wave_index == 2:
                assert completed_by_wave[1] == 2
            active_by_wave[wave_index] += 1
        barriers[wave_index].wait(timeout=2)
        time.sleep(0.005)
        with lock:
            active_by_wave[wave_index] -= 1
            completed_by_wave[wave_index] += 1
        return _ordered_report(config, start_ns=(wave_index - 1) * 2_000_000_000)

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 0
    assert completed_by_wave == {1: 2, 2: 2}
    assert [row["attempt_index"] for row in result.rows] == [1, 2, 3, 4]
    assert [row["wave_id"] for row in result.rows] == [
        "r1-n1",
        "r1-n1",
        "r1-n2",
        "r1-n2",
    ]
    assert [row["cpus"] for row in result.rows] == [
        str(worker_a),
        str(worker_a),
        str(worker_b),
        str(worker_b),
    ]
    assert all(row["duration_valid"] is True for row in result.rows)
    assert result.rows[0]["profile_dir"] == ("waves/0001-r1-n1/launch-0001-task-0/profile")
    assert result.summary["schema_version"] == "tasktrace-run-summary.v3"
    assert result.summary["input_root"] == str(tmp_path.resolve())
    assert result.summary["overall"]["valid_for_analysis"] is True
    assert result.summary["controller_affinity"]["caller_affinity_restored"] is True
    assert result.summary["scheduling"]["mode"] == "ordered-waves"
    assert "independent_performance_unit" not in result.summary["scheduling"]
    assert result.summary["scheduling"]["within_wave_dispatch"] == "concurrent-submit"
    assert result.summary["scheduling"]["measured_start_alignment"] == "not-enforced"
    assert result.summary["scheduling"]["container_cpu_placement"] == ("per-launch cpuset")
    assert [wave["metadata"]["allocation"] for wave in result.summary["waves"]] == [
        1,
        2,
    ]
    resolved_path = spec.output / "resolved-run.yaml"
    resolved_text = resolved_path.read_text()
    assert resolved_text.startswith(
        "# AUTO-GENERATED: fully resolved tasktrace-run.v3 execution plan.\n"
        "# DO NOT EDIT: edit the source manifest and run it again.\n"
    )
    assert f"# source_manifest: {json.dumps(str(manifest.resolve()))}\n" in resolved_text
    assert f"# source_manifest_sha256: {spec.manifest_sha256}\n" in resolved_text
    resolved = yaml.safe_load(resolved_text)
    assert resolved["schema_version"] == "tasktrace-run.v3"
    assert resolved["input_root"] == str(tmp_path.resolve())
    assert resolved["defaults"] == {
        "pacing": "llm-gaps",
        "network_policy": "public",
        "allowed_hosts": [],
        "profile_sets": ["cpu-average"],
        "controller_runtime": "host-mount",
        "controller_cpus": str(controller),
        "action_timeout_sec": 90.0,
    }
    assert "worker_cpus" not in resolved["defaults"]
    assert "placement" not in resolved["defaults"]
    assert resolved["tasks"][0]["atif"] == "ordered-0.json"
    assert (
        resolved["tasks"][0]["expected_source_sha256"]
        == hashlib.sha256((tmp_path / "ordered-0.json").read_bytes()).hexdigest()
    )
    assert resolved["tasks"][0]["expected_image_id"] == "sha256:image-0"
    assert resolved["waves"][0]["launches"] == [
        {"task": "task-0", "cpus": str(worker_a)},
        {"task": "task-1", "cpus": str(worker_a)},
    ]
    resolved_spec = load_run_spec(resolved_path)
    assert [launch.cpus for launch in resolved_spec.waves[1].launches] == [
        str(worker_b),
        str(worker_b),
    ]
    assert result.summary["resolved_manifest"] == "resolved-run.yaml"
    assert result.summary["resolved_manifest_sha256"] == hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    for wave in result.summary["waves"]:
        assert wave["logical_cpu_count"] == 1
        assert wave["overall"]["valid_for_throughput"] is True
        assert wave["overall"]["throughput_tasks_per_sec"] == pytest.approx(2.0)
        assert wave["overall"]["pool_utilization_percent"] == pytest.approx(100.0)
        assert wave["overall"]["median_task_duration_sec"] == pytest.approx(1.0)
    with (spec.output / "results.csv").open(newline="") as source:
        csv_rows = list(csv.DictReader(source))
    assert tuple(csv_rows[0]) == execute.RESULT_COLUMNS_V3


def test_v3_wave_failure_suppresses_that_wave_and_overall_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _cpus = _ordered_manifest(tmp_path)
    spec = load_run_spec(manifest)
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)

    def profile(config, **_kwargs):
        wave_index = 1 if "0001-r1-n1" in str(config.output_dir) else 2
        report = _ordered_report(
            config,
            start_ns=(wave_index - 1) * 2_000_000_000,
        )
        if wave_index == 2 and config.task_name == "task-1":
            report["replay_summary"]["source_outcome_mismatch_count"] = 1
        return report

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 1
    assert result.summary["overall"]["valid_for_analysis"] is False
    assert result.summary["waves"][0]["overall"]["valid_for_throughput"] is True
    assert result.summary["waves"][1]["overall"]["valid_for_throughput"] is False
    assert result.summary["waves"][1]["overall"]["throughput_tasks_per_sec"] is None


def test_v3_resolved_manifest_expands_compact_authoring_form(
    tmp_path: Path,
) -> None:
    manifest, (_controller, worker_a, worker_b) = _ordered_manifest(tmp_path)
    compact = yaml.safe_load(manifest.read_text())
    compact["defaults"].update(
        {
            "worker_cpus": f"{worker_a},{worker_b}",
            "placement": "round-robin-single-cpu",
        }
    )
    compact["waves"] = [
        {"id": "r1-n1", "allocation": 1, "repeat": 1},
        {"id": "r1-n2", "allocation": 2, "repeat": 1},
    ]
    manifest.write_text(yaml.safe_dump(compact, sort_keys=False))
    spec = load_run_spec(manifest)
    spec.output.mkdir()
    preflights = {
        (launch.task_id, launch.cpus): execute._Preflight(image_id=f"sha256:image-{launch.task_id.rsplit('-', 1)[1]}")
        for wave in spec.waves
        for launch in wave.launches
    }

    filename, digest = execute._write_resolved_ordered_manifest(
        spec,
        preflights,
    )

    resolved_path = spec.output / filename
    resolved = yaml.safe_load(resolved_path.read_text())
    assert digest == hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    assert "worker_cpus" not in resolved["defaults"]
    assert "placement" not in resolved["defaults"]
    assert resolved["waves"][0]["launches"] == [
        {"task": "task-0", "cpus": str(worker_a)},
        {"task": "task-1", "cpus": str(worker_a)},
    ]
    assert resolved["waves"][1]["launches"] == [
        {"task": "task-0", "cpus": str(worker_a)},
        {"task": "task-1", "cpus": str(worker_b)},
    ]
    resolved_spec = load_run_spec(resolved_path)
    assert resolved_spec.defaults.worker_cpus is None
    assert resolved_spec.defaults.placement is None


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda report: report.__setitem__("pacing", "recorded"),
            "profile pacing differs",
        ),
        (
            lambda report: report["execution_plan"]["extensions"].__setitem__("cpuset_cpus", "999"),
            "execution plan CPU set differs",
        ),
        (
            lambda report: report["network_policy"].__setitem__("comparability_sha256", "0" * 64),
            "network policy differs",
        ),
        (
            lambda report: report["profile_provenance"].__setitem__("profile_sets", ["hardware-deep"]),
            "profile sets differ",
        ),
        (
            lambda report: report["container_cpu"].__setitem__("allocated_logical_cpus", 2.0),
            "allocation width differs",
        ),
    ],
)
def test_v3_fails_closed_when_profile_evidence_drifts_from_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    expected_reason: str,
) -> None:
    manifest, _cpus = _ordered_manifest(tmp_path)
    spec = load_run_spec(manifest)
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)

    def profile(config, **_kwargs):
        wave_index = 1 if "0001-r1-n1" in str(config.output_dir) else 2
        report = _ordered_report(
            config,
            start_ns=(wave_index - 1) * 2_000_000_000,
        )
        if wave_index == 1 and config.task_name == "task-0":
            mutate(report)
        return report

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 1
    assert result.rows[0]["duration_valid"] is False
    assert expected_reason in result.rows[0]["reason"]
    assert result.summary["waves"][0]["overall"]["valid_for_throughput"] is False
    assert result.summary["overall"]["valid_for_analysis"] is False


def test_v2_impossible_reported_overlap_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(
        _manifest(
            tmp_path,
            task_count=3,
            repetitions=1,
            schema_version="tasktrace-run.v2",
            concurrency=2,
        )
    )
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    monkeypatch.setattr(
        execute,
        "run_atif_trajectory_profile",
        lambda config, **_kwargs: _report(
            config,
            1.0,
            window_start_ns=0,
            cpu_usage_sec=0.25,
        ),
    )

    result = execute_run(spec)

    overall = result.summary["overall"]
    assert result.exit_code == 1
    assert overall["achieved_peak_measured_attempts"] == 3
    assert overall["valid_for_throughput"] is False
    assert any("overlap exceeds requested concurrency" in reason for reason in overall["throughput_invalid_reasons"])


def test_v2_interrupt_during_initial_dispatch_retains_every_attempt_and_cleans_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_run_spec(
        _manifest(
            tmp_path,
            task_count=4,
            repetitions=1,
            schema_version="tasktrace-run.v2",
            concurrency=2,
        )
    )
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    real_executor = execute.ThreadPoolExecutor

    class InterruptingExecutor(real_executor):
        submit_count = 0

        def submit(self, *args, **kwargs):
            type(self).submit_count += 1
            if type(self).submit_count == 2:
                super().submit(*args, **kwargs)
                raise KeyboardInterrupt
            return super().submit(*args, **kwargs)

    def profile(_config, *, cancel_requested):
        deadline = time.monotonic() + 2
        while not cancel_requested() and time.monotonic() < deadline:
            time.sleep(0.001)
        raise RuntimeError("cancelled by test")

    monkeypatch.setattr(execute, "ThreadPoolExecutor", InterruptingExecutor)
    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 130
    assert result.summary["overall"]["interrupted"] is True
    assert result.summary["overall"]["valid_for_throughput"] is False
    assert [row["attempt_index"] for row in result.rows] == [1, 2, 3, 4]
    assert len({row["attempt_index"] for row in result.rows}) == 4
    assert result.rows[1]["completion_index"] is not None
    assert "cancelled by test" in result.rows[1]["reason"]
    assert not any(thread.name.startswith("tasktrace-run") and thread.is_alive() for thread in threading.enumerate())


def test_v2_interrupt_during_refill_retains_every_attempt_and_cleans_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_run_spec(
        _manifest(
            tmp_path,
            task_count=4,
            repetitions=1,
            schema_version="tasktrace-run.v2",
            concurrency=2,
        )
    )
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    real_executor = execute.ThreadPoolExecutor
    second_started = threading.Event()

    class InterruptingExecutor(real_executor):
        submit_count = 0

        def submit(self, *args, **kwargs):
            type(self).submit_count += 1
            if type(self).submit_count == 3:
                raise KeyboardInterrupt
            return super().submit(*args, **kwargs)

    def profile(config, *, cancel_requested):
        if config.task_name == "task-0":
            assert second_started.wait(timeout=2)
            return _report(
                config,
                1.0,
                window_start_ns=0,
                cpu_usage_sec=0.5,
            )
        second_started.set()
        deadline = time.monotonic() + 2
        while not cancel_requested() and time.monotonic() < deadline:
            time.sleep(0.001)
        raise RuntimeError("cancelled by test")

    monkeypatch.setattr(execute, "ThreadPoolExecutor", InterruptingExecutor)
    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 130
    assert result.summary["overall"]["interrupted"] is True
    assert [row["attempt_index"] for row in result.rows] == [1, 2, 3, 4]
    assert len({row["attempt_index"] for row in result.rows}) == 4
    assert result.rows[0]["status"] == "completed"
    assert all(row["status"] == "failed" for row in result.rows[1:])
    completed_indices = [row["completion_index"] for row in result.rows if row["completion_index"] is not None]
    assert sorted(completed_indices) == list(range(1, len(completed_indices) + 1))
    assert not any(thread.name.startswith("tasktrace-run") and thread.is_alive() for thread in threading.enumerate())


def test_ten_task_cpu_comparison_is_one_fixed_sequential_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(_manifest(tmp_path, task_count=10, repetitions=1))
    executed: list[str] = []
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)

    def profile(config):
        assert config.cpuset_cpus == "20-23"
        executed.append(config.task_name)
        return _report(config, float(len(executed)))

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    expected = [f"task-{index}" for index in range(10)]
    assert executed == expected
    assert [row["task_id"] for row in result.rows] == expected
    assert result.summary["overall"]["valid_duration_count"] == 10
    assert result.exit_code == 0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda report: report["replay_summary"].__setitem__("actions_timed_out", 1), "timed out"),
        (
            lambda report: report["replay_summary"].__setitem__("source_outcome_mismatch_count", 1),
            "source-outcome mismatch",
        ),
        (
            lambda report: report["container_cpu"].__setitem__("valid_for_cpu_performance", False),
            "not valid for performance",
        ),
        (lambda report: report.__setitem__("profiling_valid", False), "not explicitly valid"),
        (lambda report: report.__setitem__("status", "invalid"), "pipeline status"),
        (
            lambda report: report.__setitem__("execution_plan_digest", "bad"),
            "execution plan digest",
        ),
    ],
)
def test_duration_evidence_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, reason: str) -> None:
    spec = load_run_spec(_manifest(tmp_path, task_count=1, repetitions=1))
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)

    def profile(config):
        report = _report(config, 1.25)
        mutate(report)
        return report

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 1
    assert result.rows[0]["status"] == "completed"
    assert result.rows[0]["duration_sec"] == 1.25
    assert result.rows[0]["duration_valid"] is False
    assert reason in result.rows[0]["reason"]
    assert result.summary["overall"]["valid_duration_count"] == 0
    assert result.summary["overall"]["replay_tasks_per_sec"] is None
    assert result.summary["tasks"][0]["median_duration_sec"] is None


def test_preflight_failure_blocks_every_execution_and_checks_expected_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_run_spec(_manifest(tmp_path, expected_images=True))
    calls: list[str] = []

    def preflight(config):
        calls.append(config.task_name)
        report = _preflight(config)
        if config.task_name == "task-1":
            report["image_identity"] = {"id": "sha256:wrong"}
        return report

    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", preflight)
    monkeypatch.setattr(
        execute,
        "run_atif_trajectory_profile",
        lambda _config: pytest.fail("execution must not begin after failed preflight"),
    )

    with pytest.raises(RunExecutionError, match="task-1.*differs from expected"):
        execute_run(spec)

    assert calls == ["task-0", "task-1"]
    assert not spec.output.exists()


def test_execution_plan_drift_invalidates_every_repetition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(_manifest(tmp_path, task_count=1, repetitions=2))
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    calls = 0

    def profile(config):
        nonlocal calls
        calls += 1
        report = _report(config, 1.0)
        report["execution_plan_digest"] = hashlib.sha256(f"drift-{calls}".encode()).hexdigest()
        return report

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 1
    assert all(row["duration_valid"] is False for row in result.rows)
    assert all("plan changed" in row["reason"] for row in result.rows)
    assert result.summary["tasks"][0]["median_duration_sec"] is None
    assert result.summary["overall"]["replay_tasks_per_sec"] is None


def test_source_drift_and_profile_failure_are_retained_and_later_tasks_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_run_spec(_manifest(tmp_path, task_count=3, repetitions=1))
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    calls: list[str] = []

    def profile(config):
        calls.append(config.task_name)
        if config.task_name == "task-0":
            spec.tasks[1].atif.write_text("changed\n")
            raise RuntimeError("sandbox crashed")
        return _report(config, 2.0)

    monkeypatch.setattr(execute, "run_atif_trajectory_profile", profile)

    result = execute_run(spec)

    assert result.exit_code == 1
    assert calls == ["task-0", "task-2"]
    assert [row["status"] for row in result.rows] == [
        "failed",
        "failed",
        "completed",
    ]
    assert "sandbox crashed" in result.rows[0]["reason"]
    assert "source changed" in result.rows[1]["reason"]
    assert result.rows[2]["duration_valid"] is True
    assert all("cleanup_valid" not in row for row in result.rows)
    assert all("clock_reference" not in row for row in result.rows)
    assert result.summary["overall"]["completed"] == 1
    assert result.summary["overall"]["failed"] == 2
    assert result.summary["overall"]["replay_tasks_per_sec"] is None


def test_rejects_existing_output_before_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(_manifest(tmp_path, task_count=1, repetitions=1))
    spec.output.mkdir()
    (spec.output / "foreign.txt").write_text("do not overwrite\n")
    monkeypatch.setattr(
        execute,
        "preflight_atif_trajectory_profile",
        lambda _config: pytest.fail("preflight must not run for a dirty output"),
    )

    with pytest.raises(RunExecutionError, match="must not already exist"):
        execute_run(spec)

    assert (spec.output / "foreign.txt").read_text() == "do not overwrite\n"


def test_host_cpu_identity_failure_blocks_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(_manifest(tmp_path, task_count=1, repetitions=1))
    monkeypatch.setattr(
        execute,
        "collect_host_cpu_identity",
        lambda _cpu_ids: (_ for _ in ()).throw(HostIdentityError("topology missing")),
    )
    monkeypatch.setattr(
        execute,
        "preflight_atif_trajectory_profile",
        lambda _config: pytest.fail("profile preflight must not run"),
    )

    with pytest.raises(RunExecutionError, match="host CPU identity.*topology missing"):
        execute_run(spec)


def test_exclusive_output_claim_rejects_a_concurrent_creator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = load_run_spec(_manifest(tmp_path, task_count=1, repetitions=1))

    def preflight(config):
        spec.output.mkdir()
        return _preflight(config)

    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", preflight)
    monkeypatch.setattr(
        execute,
        "run_atif_trajectory_profile",
        lambda _config: pytest.fail("execution must not enter a concurrently claimed root"),
    )

    with pytest.raises(RunExecutionError, match="claimed while preflight ran"):
        execute_run(spec)


def test_leaf_cli_help_has_one_yaml_positional(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        execute.main(["--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "FILE.yaml" in output
    assert "fresh sandboxes" in output
    assert "CPU-pinned" not in output
    assert "sequentially" in output
    assert "bounded concurrent" in output
    assert "experiment" not in output


def test_leaf_cli_reports_spec_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit) as raised:
        execute.main([str(missing)])

    assert raised.value.code == 2
    assert "unable to read run file" in capsys.readouterr().err


def test_leaf_cli_runs_and_prints_only_counts_and_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _manifest(tmp_path, task_count=1, repetitions=1)
    monkeypatch.setattr(execute, "preflight_atif_trajectory_profile", _preflight)
    monkeypatch.setattr(
        execute,
        "run_atif_trajectory_profile",
        lambda config: _report(config, 2.5),
    )

    assert execute.main([str(manifest)]) == 0

    output = capsys.readouterr().out
    assert "Completed: 1; failed: 0; valid durations: 1" in output
    assert f"Results: {tmp_path / 'results/results.csv'}" in output
    assert f"Summary: {tmp_path / 'results/summary.json'}" in output

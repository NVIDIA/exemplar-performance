# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-Docker Milestone-1 phase/alignment validation.

Example (the image must contain Python, Bash, and strace)::

    TASKTRACE_M1_DOCKER_TEST_IMAGE=<immutable-image-id> \
    TASKTRACE_M1_CPUSET=16-19 \
      uv run pytest -q offline_replayer/tests/test_m1_labeled_phase_docker.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from offline_replayer.measurement.container import ProfileConfig, run_profile
from offline_replayer.measurement.evidence import read_bundle, validate_raw_artifacts
from offline_replayer.measurement.pipeline import run_profiled_replay
from offline_replayer.timebase import (
    monotonic_clock_reference,
    monotonic_mapping_from_references,
)

IMAGE = os.environ.get("TASKTRACE_M1_DOCKER_TEST_IMAGE")
CPUSET = os.environ.get("TASKTRACE_M1_CPUSET")
pytestmark = pytest.mark.skipif(
    not IMAGE or not CPUSET,
    reason="set TASKTRACE_M1_DOCKER_TEST_IMAGE and TASKTRACE_M1_CPUSET for real M1 E2E",
)
MIB = 1024 * 1024
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def _command(*, replay_report: bool = False) -> str:
    command = (
        "PYTHONPATH=/tasktrace-source python -m "
        "offline_replayer.examples.m1_labeled_phase "
        "--phase-log /tasktrace-profile-output/phases.jsonl "
        "--sentinel /tasktrace-profile-output/executed.once "
        "--serial-cpu-sec 0.35 --memory-mib 64 --memory-hold-sec 0.45 "
        "--parallel-workers 4 --parallel-cpu-sec 0.45 "
        "--teardown-hold-sec 0.25"
    )
    if replay_report:
        command += " --replay-report /tasktrace/replay_report.json"
    return command


def _profile(output_dir: Path, *, strace_mode: str) -> dict[str, Any]:
    return run_profile(
        ProfileConfig(
            command=_command(),
            prepare_command=None,
            output_dir=output_dir,
            image=IMAGE,
            container=None,
            cpuset_cpus=CPUSET,
            cpu_limit=None,
            allocated_cpus=4,
            workdir="/tmp",
            user=None,
            env=(),
            mounts=(f"type=bind,src={SOURCE_ROOT},dst=/tasktrace-source,readonly",),
            strace_mode=strace_mode,
            string_limit=4096,
            shell="/bin/bash",
            memory_profile=True,
            memory_sample_interval_sec=0.025,
            network="none",
        )
    )


def _coordinated_profile(output_dir: Path, *, profile_sets: tuple[str, ...]) -> dict[str, Any]:
    return run_profile(
        ProfileConfig(
            command=_command(),
            prepare_command=None,
            output_dir=output_dir,
            image=IMAGE,
            container=None,
            cpuset_cpus=CPUSET,
            cpu_limit=None,
            allocated_cpus=4,
            workdir="/tmp",
            user=None,
            env=(),
            mounts=(f"type=bind,src={SOURCE_ROOT},dst=/tasktrace-source,readonly",),
            strace_mode="execve",  # profile sets must override this safely.
            string_limit=4096,
            shell="/bin/bash",
            memory_sample_interval_sec=0.025,
            network="none",
            terminate_descendants_at_finish=True,
            profile_sets=profile_sets,
        )
    )


def _coordinated_pipeline_profile(output_dir: Path) -> dict[str, Any]:
    fixture_source = SOURCE_ROOT / "offline_replayer/examples/m1_labeled_phase.py"
    profile = ProfileConfig(
        command=_command(replay_report=True),
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        image=IMAGE,
        container=None,
        cpuset_cpus=CPUSET,
        cpu_limit=None,
        allocated_cpus=4,
        workdir="/tmp",
        user=None,
        env=(),
        mounts=(
            f"type=bind,src={SOURCE_ROOT},dst=/tasktrace-source,readonly",
            f"type=bind,src={output_dir / 'replay'},dst=/tasktrace",
        ),
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        memory_sample_interval_sec=0.025,
        network="none",
        terminate_descendants_at_finish=True,
        profile_sets=("cpu-compare", "software-deep", "hardware-deep"),
        workload_identity={
            "fixture": "m1-labeled-phase",
            "serial_cpu_sec": 0.35,
            "memory_mib": 64,
            "parallel_workers": 4,
        },
        source_identity={
            "format": "m1-labeled-fixture",
            "sha256": hashlib.sha256(fixture_source.read_bytes()).hexdigest(),
        },
        expected_outcome={"kind": "fixture-replay", "action_count": 1},
    )
    return run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={
            "schema_version": "container-m1-labeled-profile.v1",
            "task_name": "m1-labeled-phase",
            "phase_log_path": str((output_dir / "container_profile" / "phases.jsonl").resolve()),
        },
    )


def _phases(output_dir: Path) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    records = [json.loads(line) for line in (output_dir / "phases.jsonl").read_text().splitlines()]
    fixture = next(record for record in records if record["record_type"] == "fixture")
    mapping = monotonic_mapping_from_references(fixture["clock_reference"], monotonic_clock_reference())
    assert mapping["status"] == "valid"
    assert mapping["max_error_ns"] == 0
    offset_ns = int(mapping["offset_ns"])
    starts = {
        record["phase"]: int(record["timestamp_ns"]) + offset_ns
        for record in records
        if record["record_type"] == "phase_start"
    }
    ends = {
        record["phase"]: int(record["timestamp_ns"]) + offset_ns
        for record in records
        if record["record_type"] == "phase_end"
    }
    assert set(starts) == set(ends)
    return ({name: (starts[name], ends[name]) for name in starts}, mapping)


def _timeseries(output_dir: Path) -> list[dict[str, Any]]:
    return json.loads((output_dir / "resource_timeseries.json").read_text())["samples"]


def _inside(samples: list[dict[str, Any]], interval: tuple[int, int]) -> list[dict[str, Any]]:
    start_ns, end_ns = interval
    return [sample for sample in samples if start_ns <= int(sample["timestamp_ns"]) <= end_ns]


def _assert_expected_fixture_process_tree(report: dict[str, Any], *, replay_report: bool = False) -> None:
    process_exec = report["process_exec"]
    scope = process_exec["scope"]
    summary = process_exec["summary"]
    processes = process_exec["processes"]
    root_pid = scope["root_pid"]
    expected_python_argv = [
        "python",
        "-m",
        "offline_replayer.examples.m1_labeled_phase",
        "--phase-log",
        "/tasktrace-profile-output/phases.jsonl",
        "--sentinel",
        "/tasktrace-profile-output/executed.once",
        "--serial-cpu-sec",
        "0.35",
        "--memory-mib",
        "64",
        "--memory-hold-sec",
        "0.45",
        "--parallel-workers",
        "4",
        "--parallel-cpu-sec",
        "0.45",
        "--teardown-hold-sec",
        "0.25",
    ]
    if replay_report:
        expected_python_argv.extend(["--replay-report", "/tasktrace/replay_report.json"])

    assert process_exec["valid"] is True
    assert process_exec["trace_complete"] is True
    assert scope["process_escape_detected"] is False
    assert scope["process_escape_pids"] == []
    assert scope["unresolved_parent_pids"] == []
    assert summary["process_count"] == 6
    assert summary["successful_exec_count"] == 3
    assert summary["failed_exec_count"] == 0
    assert summary["anomaly_count"] == 0
    assert len(processes) == 6
    assert sorted(process["binary"] for process in processes) == [
        "/bin/echo",
        "/usr/local/bin/python",
        "/usr/local/bin/python",
        "/usr/local/bin/python",
        "/usr/local/bin/python",
        "/usr/local/bin/python",
    ]
    assert all(process["cwd"] == "/tmp" for process in processes)
    assert all(process["exit_code"] == 0 for process in processes)
    assert all(process["end_ns"] >= process["start_ns"] for process in processes)

    python_processes = [process for process in processes if process["binary"] == "/usr/local/bin/python"]
    assert len(python_processes) == 5
    root = next(process for process in python_processes if process["pid"] == root_pid)
    assert root["parent_pid"] is None
    assert all(process["argv"] == expected_python_argv for process in python_processes)
    assert {process["parent_pid"] for process in python_processes if process["pid"] != root_pid} == {root_pid}

    echo = [process for process in processes if process["binary"] == "/bin/echo"]
    assert len(echo) == 1
    assert echo[0]["parent_pid"] == root_pid
    assert echo[0]["argv"] == ["/bin/echo", "tasktrace-m1-child"]


def test_real_labeled_fixture_aligns_cpu_memory_and_phase_truth(
    tmp_path: Path,
) -> None:
    output = tmp_path / "untraced"
    report = _profile(output, strace_mode="none")
    phases, clock_mapping = _phases(output)
    samples = _timeseries(output)

    assert report["command_exit_code"] == 0
    assert report["memory"]["natural_peak_trustworthy"] is True
    assert report["memory"]["ancestor_memory"]["natural_demand_trustworthy"] is True
    assert report["memory"]["lifetime"]["peak_bytes"] >= 56 * MIB
    assert (output / "executed.once").read_text() == "executed-once\n"
    assert clock_mapping["method"] == "linux-time-namespace-offset"

    serial = _inside(samples, phases["serial_cpu"])
    parallel = _inside(samples, phases["parallel_cpu"])
    plateau = _inside(samples, phases["memory_plateau"])
    teardown = _inside(samples, phases["teardown"])
    assert len(serial) >= 5 and len(parallel) >= 5 and len(plateau) >= 5
    assert max(sample["cpu_average_logical_cores"] or 0 for sample in serial) >= 0.75
    assert max(sample["cpu_average_logical_cores"] or 0 for sample in serial) < 2.0
    assert max(sample["cpu_average_logical_cores"] or 0 for sample in parallel) >= 3.0
    assert max(sample["memory_current_bytes"] for sample in plateau) >= 56 * MIB
    assert teardown[-1]["memory_current_bytes"] < 32 * MIB
    assert abs(report["cpu"]["usage_sec"] - report["memory"]["command_window"]["sampled_cpu_usage_sec"]) < 0.05


def test_real_software_deep_reconstructs_expected_process_tree_once(
    tmp_path: Path,
) -> None:
    output = tmp_path / "software-deep"
    report = _profile(output, strace_mode="process")

    assert report["command_exit_code"] == 0
    _assert_expected_fixture_process_tree(report)
    assert report["memory"]["peak_valid"] is True
    assert report["memory"]["natural_peak_trustworthy"] is False
    assert report["memory"]["natural_peak_invalid_reasons"] == [
        "strace ran inside the measured cgroup and changed the sandbox memory "
        "footprint; rerun with --strace-mode none to measure an unperturbed "
        "natural peak"
    ]
    assert (output / "executed.once").read_text() == "executed-once\n"


def test_real_coordinated_cpu_compare_writes_aligned_bundle_and_views_once(
    tmp_path: Path,
) -> None:
    output = tmp_path / "coordinated-cpu"
    report = _coordinated_profile(output, profile_sets=("cpu-compare",))
    bundle_path = Path(report["evidence_bundle"]["path"])
    bundle = read_bundle(bundle_path)
    validate_raw_artifacts(bundle, output)

    assert report["command_exit_code"] == 0
    assert report["strace"]["mode"] == "none"
    assert report["memory"]["natural_peak_trustworthy"] is True
    assert report["memory"]["ancestor_memory"]["natural_demand_trustworthy"] is True
    statuses = {item.profiler_id: item for item in bundle.profilers}
    assert statuses["cgroup-cpu"].status == "valid"
    assert statuses["cgroup-memory"].status == "valid"
    assert statuses["scheduler-concurrency"].status in {"valid", "invalid"}
    if statuses["scheduler-concurrency"].status == "invalid":
        assert statuses["scheduler-concurrency"].reasons
    metrics = {stream.metric: stream for stream in bundle.streams}
    assert metrics["cpu.interval_average_logical_cores"].records
    assert metrics["memory.current"].records
    assert metrics["memory.pressure.some.total_stall"].records
    assert metrics["memory.events.oom"].records[0].value == 0
    artifact_ids = [artifact.artifact_id for artifact in bundle.artifacts]
    assert len(artifact_ids) == len(set(artifact_ids))
    assert artifact_ids.count("cgroup-resource-timeseries") == 1
    assert artifact_ids.count("cgroup-memory-source-snapshots") == 1
    memory_streams = [stream for stream in bundle.streams if stream.profiler_id == "cgroup-memory"]
    assert memory_streams and all("cgroup-resource-timeseries" in stream.raw_artifact_ids for stream in memory_streams)
    assert "cgroup-memory-source-snapshots" in metrics["memory.peak"].raw_artifact_ids
    assert "cgroup-memory-source-snapshots" in metrics["memory.events.oom"].raw_artifact_ids
    source_snapshot = json.loads((output / "cgroup_memory_source_snapshots.json").read_text())
    finish_peak = [
        row
        for row in source_snapshot["reads"]
        if row["phase"] == "controlled_finish_gate"
        and row["source_scope"] == "leaf"
        and row["filename"] == "memory.peak"
    ]
    assert len(finish_peak) == 1
    assert int(finish_peak[0]["raw_text"].strip()) == report["memory"]["lifetime"]["peak_bytes"]
    assert (output / "executed.once").read_text() == "executed-once\n"

    rendered = json.loads((output / "unified_profile_artifacts.json").read_text())
    assert rendered["workload_executions"] == 1
    for key in ("html", "svg", "animated_svg"):
        artifact = Path(rendered[key])
        assert artifact.is_file()
        artifact_text = artifact.read_text()
        assert rendered["view_model_sha256"] in artifact_text
        assert "scheduler-concurrency" in artifact_text
        assert "cgroup-resource-timeseries" in artifact_text
        assert "cgroup-memory-source-snapshots" in artifact_text
        if statuses["scheduler-concurrency"].status == "invalid":
            assert statuses["scheduler-concurrency"].reasons[0] in artifact_text


def test_real_all_profile_sets_share_one_execution_and_fail_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "coordinated-all"
    report = _coordinated_profile(
        output,
        profile_sets=("cpu-compare", "software-deep", "hardware-deep"),
    )
    bundle = read_bundle(Path(report["evidence_bundle"]["path"]))

    assert report["command_exit_code"] == 0
    assert report["strace"]["mode"] == "process"
    assert report["process_exec"]["valid"] is True
    assert report["process_exec"]["summary"]["process_count"] == 6
    assert report["memory"]["peak_valid"] is True
    assert report["memory"]["natural_peak_trustworthy"] is False
    assert "strace" in " ".join(report["memory"]["natural_peak_invalid_reasons"])
    statuses = {item.profiler_id: item for item in bundle.profilers}
    assert statuses["process-exec"].status == "unalignable"
    assert statuses["cgroup-cpu"].status == "invalid"
    assert statuses["cgroup-memory"].status == "invalid"
    assert statuses["scheduler-concurrency"].status == "invalid"
    assert statuses["hardware-perf"].status == "invalid"
    counters = report["hardware_perf"]["counters"]
    for counter in counters:
        if counter["valid"]:
            assert counter["time_running_fraction"] >= 0.95
        else:
            assert counter["value"] is None
            assert counter["invalid_reason"]
    assert (output / "executed.once").read_text() == "executed-once\n"
    assert json.loads((output / "unified_profile_artifacts.json").read_text())["workload_executions"] == 1


def test_real_pipeline_unifies_all_profile_families_and_phases_once(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pipeline-all"
    report = _coordinated_pipeline_profile(output)
    bundle = read_bundle(output / "evidence_bundle.json")
    validate_raw_artifacts(bundle, output)

    assert report["replay_summary"]["actions_executed"] == 1
    assert report["replay_summary"]["actions_timed_out"] == 0
    assert report["replay_summary"]["source_outcome_mismatch_count"] == 0
    assert report["evidence_bundle"]["authoritative"] is True
    assert report["evidence_bundle"]["path"] == str((output / "evidence_bundle.json").resolve())
    assert (output / "container_profile" / "executed.once").read_text() == ("executed-once\n")

    profilers = {item.profiler_id: item for item in bundle.profilers}
    assert {
        "cgroup-cpu",
        "cgroup-memory",
        "scheduler-concurrency",
        "process-exec",
        "hardware-perf",
        "replay-actions",
        "labeled-phases",
    } <= set(profilers)
    assert profilers["replay-actions"].status == "valid"
    assert profilers["labeled-phases"].status == "valid"
    assert profilers["process-exec"].status == "unalignable"
    assert profilers["cgroup-cpu"].status == "invalid"
    assert profilers["cgroup-memory"].status == "invalid"
    assert profilers["scheduler-concurrency"].status == "invalid"
    assert profilers["hardware-perf"].status == "invalid"

    streams = {stream.metric: stream for stream in bundle.streams}
    phases = streams["workload.phase"]
    assert phases.status == "valid"
    assert [record.value["phase"] for record in phases.records] == [
        "serial_cpu",
        "memory_plateau",
        "parallel_cpu",
        "child_exec",
        "teardown",
    ]
    assert all(record.alignment_error_ns == 0 for record in phases.records)
    action = streams["replay.action"].records[0]
    assert action.canonical_start_ns <= phases.records[0].canonical_start_ns
    assert action.canonical_end_ns >= phases.records[-1].canonical_end_ns

    plateau = next(record for record in phases.records if record.value["phase"] == "memory_plateau")
    plateau_samples = [
        record
        for record in streams["memory.current"].records
        if record.acquisition_bounds is not None
        and plateau.canonical_start_ns
        <= (record.acquisition_bounds.canonical_start_ns + record.acquisition_bounds.canonical_end_ns) // 2
        <= plateau.canonical_end_ns
    ]
    assert plateau_samples
    assert max(int(record.value) for record in plateau_samples) >= 56 * MIB

    parallel = next(record for record in phases.records if record.value["phase"] == "parallel_cpu")
    parallel_cpu = [
        record
        for record in streams["cpu.interval_average_logical_cores"].records
        if record.canonical_start_ns < parallel.canonical_end_ns
        and record.canonical_end_ns > parallel.canonical_start_ns
    ]
    assert parallel_cpu
    assert max(float(record.value) for record in parallel_cpu) >= 2.0

    _assert_expected_fixture_process_tree(report, replay_report=True)
    assert all(record.scope_id == bundle.scope.scope_id for stream in bundle.streams for record in stream.records)

    rendered = json.loads((output / "unified_profile_artifacts.json").read_text())
    assert rendered["workload_executions"] == 1
    assert rendered["generation_method"] == ("verified-direct-coordinator-bundle-augmentation")
    for key in ("html", "svg", "animated_svg"):
        artifact_text = Path(rendered[key]).read_text()
        assert rendered["view_model_sha256"] in artifact_text
        for expected in (
            "replay-actions",
            "labeled-phases",
            "process-exec",
            "hardware-perf",
            "scheduler-concurrency",
            "memory.current",
            "parallel_cpu",
        ):
            assert expected in artifact_text


def test_real_post_measurement_verifier_observes_state_without_charging_counters(
    tmp_path: Path,
) -> None:
    output = tmp_path / "verified"
    command = (
        "python -c 'from pathlib import Path; "
        'Path("/tasktrace-profile-output/workload.once").open("x").write("once"); '
        'Path("/tmp/workload-state").write_text("ready")\''
    )
    verification_command = (
        "python -c 'from pathlib import Path; "
        'state=Path("/tmp/workload-state").read_text(); '
        'assert state == "ready"; data=bytearray(192*1024*1024); '
        'data[::4096]=b"x"*(len(data)//4096); '
        'sum(i*i for i in range(8000000)); print("verified:"+state)\''
    )
    report = run_profile(
        ProfileConfig(
            command=command,
            prepare_command=None,
            verification_command=verification_command,
            output_dir=output,
            image=IMAGE,
            container=None,
            cpuset_cpus=CPUSET,
            cpu_limit=None,
            allocated_cpus=4,
            workdir="/tmp",
            user=None,
            env=(),
            mounts=(),
            strace_mode="none",
            string_limit=4096,
            shell="/bin/bash",
            memory_sample_interval_sec=0.025,
            network="none",
            profile_sets=("cpu-compare",),
        )
    )

    assert (output / "workload.once").read_text() == "once"
    verification = report["verification"]
    assert verification["status"] == "passed"
    assert verification["exit_code"] == 0
    assert verification["stdout"] == "verified:ready\n"
    assert verification["stderr"] == ""
    assert verification["measured_counters_include_verification"] is False
    assert report["cpu"]["usage_sec"] < 0.5
    assert report["memory"]["lifetime"]["peak_bytes"] < 128 * MIB
    markers = [marker["name"] for marker in report["canonical_lifecycle"]["lifecycle_markers"]]
    assert markers[-4:] == [
        "measured_work_end",
        "verification_start",
        "verification_end",
        "run_end",
    ]
    bundle = read_bundle(Path(report["evidence_bundle"]["path"]))
    validate_raw_artifacts(bundle, output)
    assert {
        "workload-verification-stdout",
        "workload-verification-stderr",
        "workload-verification-exit_code",
        "workload-verification-status",
    } <= {artifact.artifact_id for artifact in bundle.artifacts}


def test_real_nonzero_post_measurement_verifier_invalidates_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "verification-failed"
    report = run_profile(
        ProfileConfig(
            command="true",
            prepare_command=None,
            verification_command="printf 'verification failed\\n' >&2; exit 7",
            output_dir=output,
            image=IMAGE,
            container=None,
            cpuset_cpus=CPUSET,
            cpu_limit=None,
            allocated_cpus=4,
            workdir="/tmp",
            user=None,
            env=(),
            mounts=(),
            strace_mode="none",
            string_limit=4096,
            shell="/bin/bash",
            network="none",
            coordinated_baseline=True,
        )
    )

    assert report["command_exit_code"] == 0
    assert report["verification"]["status"] == "failed"
    assert report["verification"]["exit_code"] == 7
    assert report["profiling_valid"] is False
    assert report["evidence_bundle"]["coordination_status"] == "invalid"
    reason = next(
        item for item in report["profiling_invalid_reasons"] if item["profiler_id"] == "workload-verification"
    )
    assert reason["reasons"] == ["verification command exited with status 7"]
    bundle = read_bundle(Path(report["evidence_bundle"]["path"]))
    assert bundle.extensions["evidence_status"] == "invalid"
    assert bundle.extensions["workload_verification_valid"] is False

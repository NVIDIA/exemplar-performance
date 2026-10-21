# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.measurement.evidence import (
    bundle_bytes,
    read_bundle,
    validate_raw_artifacts,
)
from offline_replayer.measurement.evidence_projection import (
    EvidenceProjectionError,
    build_evidence_bundle,
)
from offline_replayer.reporting.unified_view import write_unified_view_artifacts

RUN_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
SCOPE_ID = "docker-cgroup:synthetic-full-run"
BOOT_ID = "11111111-2222-4333-8444-555555555555"
TIME_NS = "time:[4026531834]"


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    return path


def _clock_reference() -> dict[str, object]:
    return {
        "clock_id": "CLOCK_MONOTONIC",
        "timestamp_unit": "nanoseconds",
        "time_namespace": TIME_NS,
        "boot_id": BOOT_ID,
        "storage_precision_ns": 1,
        "accuracy_claim": None,
    }


def _full_retained_report(tmp_path: Path) -> tuple[Path, Path]:
    sentinel = tmp_path / "invocation-count.txt"
    sentinel.write_text("1\n")
    timeseries = _write_json(
        tmp_path / "container_profile" / "resource_timeseries.json",
        {
            "schema_version": "container-cgroup-resource-timeseries.v1",
            "scope": "/sys/fs/cgroup/docker/synthetic",
            "clock": "host-monotonic",
            "timestamp_semantics": "acquisition-start",
            "sample_interval_sec": 0.5,
            "markers": [
                {"name": "sandbox_observation_start", "timestamp_ns": 1_000_000_000},
                {"name": "command_start", "timestamp_ns": 1_100_000_000},
                {"name": "command_end", "timestamp_ns": 2_500_000_000},
                {"name": "sandbox_observation_end", "timestamp_ns": 2_900_000_000},
            ],
            "samples": [
                {
                    "timestamp_ns": 1_000_000_000,
                    "acquisition_start_ns": 1_000_000_000,
                    "acquisition_end_ns": 1_000_001_000,
                    "memory_current_bytes": 10_000_000,
                    "memory_stat": {"anon": 6_000_000, "file": 2_000_000, "kernel": 2_000_000},
                    "cpu_average_logical_cores": None,
                },
                {
                    "timestamp_ns": 1_500_000_000,
                    "acquisition_start_ns": 1_500_000_000,
                    "acquisition_end_ns": 1_500_002_000,
                    "memory_current_bytes": 80_000_000,
                    "memory_stat": {"anon": 70_000_000, "file": 5_000_000, "kernel": 5_000_000},
                    "cpu_average_logical_cores": 1.0,
                },
                {
                    "timestamp_ns": 2_000_000_000,
                    "acquisition_start_ns": 2_000_000_000,
                    "acquisition_end_ns": 2_000_003_000,
                    "memory_current_bytes": 82_000_000,
                    "memory_stat": {"anon": 71_000_000, "file": 6_000_000, "kernel": 5_000_000},
                    "cpu_average_logical_cores": 2.0,
                },
                {
                    "timestamp_ns": 2_900_000_000,
                    "acquisition_start_ns": 2_900_000_000,
                    "acquisition_end_ns": 2_900_004_000,
                    "memory_current_bytes": 20_000_000,
                    "memory_stat": {"anon": 10_000_000, "file": 5_000_000, "kernel": 5_000_000},
                    "cpu_average_logical_cores": 0.2,
                },
            ],
        },
    )
    scheduler_events = _write_jsonl(
        tmp_path / "container_profile" / "scheduler_events.jsonl",
        [
            {"timestamp_ns": 1_200_000_000, "cpu": 0, "tid": 10, "entering": True},
            {"timestamp_ns": 1_400_000_000, "cpu": 1, "tid": 11, "entering": True},
            {"timestamp_ns": 1_800_000_000, "cpu": 1, "tid": 11, "entering": False},
            {"timestamp_ns": 2_200_000_000, "cpu": 0, "tid": 10, "entering": False},
        ],
    )
    replay = _write_json(
        tmp_path / "replay" / "replay_report.json",
        {
            "schema_version": "container-atif-replay.v2",
            "clock_reference": _clock_reference(),
            "actions": [
                {
                    "action_index": 1,
                    "tool": "exec_command",
                    "command": "make -j2",
                    "status": "ok",
                    "returncode": 0,
                    "outcome_matches_source": True,
                    "replay_start_monotonic_ns": 1_250_000_000,
                    "replay_end_monotonic_ns": 1_650_000_000,
                }
            ],
            "summary": {"actions_ok": 1},
        },
    )
    phase_log = _write_jsonl(
        tmp_path / "container_profile" / "phases.jsonl",
        [
            {
                "record_type": "fixture",
                "schema_version": "tasktrace-m1-labeled-phase.v1",
                "clock_reference": _clock_reference(),
                "timestamp_ns": 1_050_000_000,
            },
            {
                "record_type": "phase_start",
                "phase": "serial_cpu",
                "timestamp_ns": 1_150_000_000,
                "attributes": {"requested_duration_sec": 0.2},
            },
            {"record_type": "phase_end", "phase": "serial_cpu", "timestamp_ns": 1_350_000_000},
            {
                "record_type": "phase_start",
                "phase": "parallel_cpu",
                "timestamp_ns": 1_400_000_000,
                "attributes": {"workers": 2},
            },
            {"record_type": "phase_end", "phase": "parallel_cpu", "timestamp_ns": 1_900_000_000},
        ],
    )
    raw_trace = tmp_path / "container_profile" / "process.strace.log"
    raw_trace.write_text('42 1700000000.000000000 execve("/bin/sh", ["sh"], 0) = 0\n')
    perf_stats = tmp_path / "container_profile" / "hardware_perf.csv"
    perf_stats.write_text("1000;;cycles;1000000000;100.00\n")
    perf_stderr = tmp_path / "container_profile" / "hardware_perf.stderr.txt"
    perf_stderr.write_text("")

    process_report = {
        "schema_version": "software-deep-process-exec.v1",
        "method": "strace-f-ttt-process-chdir",
        "source_clock": "realtime",
        "valid": True,
        "trace_complete": True,
        "anomalies": [],
        "processes": [
            {
                "pid": 42,
                "parent_pid": None,
                "start_ns": 1_700_000_000_000_000_000,
                "end_ns": 1_700_000_001_000_000_000,
                "binary": "/bin/sh",
                "argv": ["sh"],
                "cwd": "/work",
                "exit_code": 0,
                "exit_signal": None,
                "in_scope": True,
                "process_escape": False,
                "execs": [
                    {
                        "syscall": "execve",
                        "started_ns": 1_700_000_000_100_000_000,
                        "completed_ns": 1_700_000_000_100_010_000,
                        "binary": "/bin/sh",
                        "argv": ["sh"],
                        "cwd": "/work",
                    }
                ],
            }
        ],
        "artifacts": {"raw_trace_path": str(raw_trace)},
    }
    hardware_report = {
        "schema_version": "cgroup-perf-hardware.v1",
        "method": "perf-stat-cgroup-cpu-wide",
        "window_start_ns": 1_100_000_000,
        "window_end_ns": 2_500_000_000,
        "window_start_alignment_error_ns": 1_234,
        "window_end_alignment_error_ns": 5_678,
        "instrumentation_valid": True,
        "all_requested_metrics_valid": False,
        "anomalies": [],
        "provenance": {
            "perf_binary": "perf",
            "perf_version": "perf version fixture",
            "kernel_release": "6.fixture",
            "machine": "x86_64",
            "event_source_names": ["cpu", "software"],
            "provenance_valid": True,
            "provenance_errors": [],
        },
        "stats_path": str(perf_stats),
        "stderr_path": str(perf_stderr),
        "counters": [
            {
                "event": "cycles",
                "namespace": "portable",
                "status": "counted",
                "value": 1_000_000,
                "unit": "event",
                "time_enabled_ns": 1_400_000_000,
                "time_running_ns": 1_390_000_000,
                "time_running_fraction": 0.992857,
                "valid": True,
                "invalid_reason": None,
            },
            {
                "event": "cache-misses",
                "namespace": "portable",
                "status": "not-supported",
                "value": None,
                "unit": None,
                "time_enabled_ns": None,
                "time_running_ns": None,
                "time_running_fraction": None,
                "valid": False,
                "invalid_reason": "event is unsupported by this PMU",
            },
        ],
    }
    profile = _write_json(
        tmp_path / "container_profile" / "profile_report.json",
        {
            "schema_version": "container-command-profile.v1",
            "run_id": RUN_ID,
            "runtime": "docker",
            "scope": "disposable-image-container",
            "image_identity": {"id": "sha256:synthetic"},
            "cpu": {
                "method": "container-cgroup-v2-cpu.stat",
                "wall_sec": 1.4,
                "usage_sec": 2.1,
                "user_sec": 1.8,
                "system_sec": 0.3,
                "average_logical_cores_used": 1.5,
                "utilization_fraction_of_allocation": 0.75,
            },
            "memory": {
                "samples_path": str(timeseries),
                "timeline_valid_for_shape": True,
                "timeline_invalid_reasons": [],
                "lifetime": {
                    "peak_bytes": 90_000_000,
                    "peak_source": "memory.peak-since-cgroup-creation-read-at-finish-gate",
                    "exact_through_controlled_finish_gate": True,
                },
                "events_delta": {"high": 0, "max": 0, "oom": 0, "oom_kill": 0},
                "pressure_delta": {"some": {"total_stall_usec": 123, "stall_fraction_of_wall": 0.0001}},
                "peak_valid": True,
                "natural_peak_trustworthy": True,
                "censoring": "none",
                "limits": {
                    "memory_max_state": "finite",
                    "memory_max_bytes": 100_000_000,
                },
                "observed_limit_margin": {
                    "remaining_to_limit_bytes": 10_000_000,
                    "peak_fraction_of_limit": 0.9,
                },
            },
            "scheduler_concurrency": {
                "method": "perf-record-cgroup-native-switch-events",
                "window_start_ns": 1_100_000_000,
                "window_end_ns": 2_500_000_000,
                "trace_complete": True,
                "lost_perf_events": 0,
                "anomalies": [],
                "events_path": str(scheduler_events),
                "cgroup_validation": {"valid_for_concurrency_reporting": True},
            },
            "process_exec": process_report,
            "strace": {"mode": "process", "log_path": str(raw_trace)},
            "hardware_perf": hardware_report,
        },
    )
    report = _write_json(
        tmp_path / "pipeline_report.json",
        {
            "schema_version": "container-atif-trajectory-profile.v1",
            "run_id": RUN_ID,
            "container_profile_report": str(profile),
            "replay_report": str(replay),
            "phase_log_path": str(phase_log),
            "container_cpu": json.loads(profile.read_text())["cpu"],
            "container_memory": json.loads(profile.read_text())["memory"],
            "scheduler_concurrency": json.loads(profile.read_text())["scheduler_concurrency"],
            "measured_scope": {
                "scope_id": SCOPE_ID,
                "kind": "disposable-container-cgroup-subtree",
                "identity": {"runtime": "docker", "container_id": "synthetic"},
                "includes_descendants": True,
            },
            "canonical_clock_reference": _clock_reference(),
            "canonical_lifecycle": {
                "clock_id": "host-monotonic",
                "clock_name": "CLOCK_MONOTONIC",
                "clock_reference": _clock_reference(),
                "origin_ns": 1_100_000_000,
                "start_ns": 1_000_000_000,
                "end_ns": 3_000_000_000,
                "lifecycle_markers": [
                    {"name": "run_start", "canonical_timestamp_ns": 1_000_000_000},
                    {"name": "command_start", "canonical_timestamp_ns": 1_100_000_000},
                    {"name": "command_end", "canonical_timestamp_ns": 2_500_000_000},
                    {"name": "run_end", "canonical_timestamp_ns": 3_000_000_000},
                ],
            },
        },
    )
    return report, sentinel


def test_projects_full_retained_run_into_typed_evidence_without_execution(
    tmp_path: Path,
) -> None:
    report, sentinel = _full_retained_report(tmp_path)
    output = tmp_path / "evidence_bundle.json"

    bundle = build_evidence_bundle(report, output)

    assert sentinel.read_text() == "1\n"
    assert bundle.run_id == RUN_ID
    assert bundle.scope.scope_id == SCOPE_ID
    assert bundle.timeline.origin_ns == 1_100_000_000
    assert read_bundle(output).to_dict() == bundle.to_dict()
    assert output.read_bytes() == bundle_bytes(bundle)
    validate_raw_artifacts(bundle, tmp_path)

    streams = {stream.stream_id: stream for stream in bundle.streams}
    assert streams["cpu-usage"].records[0].value == 2.1
    assert len(streams["cpu-interval-average"].records) == 3
    assert streams["memory-current"].records[1].acquisition_bounds is not None
    assert streams["memory-current"].records[1].alignment_error_ns == 2_000
    assert {stream_id for stream_id in streams if stream_id.startswith("memory-event-")} >= {
        "memory-event-oom",
        "memory-event-oom_kill",
    }
    assert streams["memory-pressure-some"].shape == "window_aggregate"

    peak = streams["memory-peak"].records[0]
    assert peak.value == 90_000_000
    assert peak.attributes["exact"] is True
    assert peak.attributes["natural_peak_trustworthy"] is True
    assert peak.source_clock_id is None
    assert peak.native_timestamp_ns is None
    assert peak.canonical_timestamp_ns is None
    assert peak.acquisition_bounds is None
    assert streams["memory-max"].records[0].value == 100_000_000
    margin = streams["memory-observed-limit-margin"].records[0]
    assert margin.value == 10_000_000
    assert margin.attributes["not_a_recommendation"] is True
    assert margin.attributes["peak_fraction_of_limit"] == 0.9

    assert streams["scheduler-transitions"].status == "valid"
    assert max(record.value for record in streams["scheduler-concurrency"].records) == 2
    assert {record.attributes["lane"] for record in streams["scheduler-per-core"].records} == {
        "logical-cpu-0",
        "logical-cpu-1",
    }
    assert streams["replay-actions"].status == "valid"
    assert streams["labeled-phases"].status == "valid"

    process = streams["process-lifecycles"]
    assert process.status == "unalignable"
    assert process.records[0].native_start_ns == 1_700_000_000_000_000_000
    assert process.records[0].canonical_start_ns is None
    assert "CLOCK_REALTIME" in bundle.clocks[-1].clock_name or any(
        clock.clock_name == "CLOCK_REALTIME" for clock in bundle.clocks
    )

    assert streams["hardware-cycles"].status == "valid"
    assert streams["hardware-cycles"].records[0].alignment_error_ns == 5_678
    unsupported = streams["hardware-cache-misses"]
    assert unsupported.status == "invalid"
    assert unsupported.records[0].value is None
    assert "unsupported" in unsupported.reasons[0]
    hardware_manifest = next(
        manifest for manifest in bundle.profilers if manifest.profiler_id == "compat-hardware-perf"
    )
    assert hardware_manifest.provenance["perf_version"] == "perf version fixture"
    assert hardware_manifest.status == "invalid"
    assert bundle.extensions["evidence_status"] == "invalid"
    assert bundle.extensions["all_selected_profilers_valid"] is False

    relative_paths = {artifact.relative_path for artifact in bundle.artifacts}
    assert {
        "pipeline_report.json",
        "container_profile/profile_report.json",
        "container_profile/resource_timeseries.json",
        "container_profile/scheduler_events.jsonl",
        "container_profile/process.strace.log",
        "container_profile/hardware_perf.csv",
        "container_profile/phases.jsonl",
        "replay/replay_report.json",
    } <= relative_paths

    rendered = write_unified_view_artifacts(bundle, tmp_path)
    digest = rendered["view_model_sha256"]
    assert sentinel.read_text() == "1\n"
    for name in ("html", "svg", "animated_svg"):
        artifact_text = Path(rendered[name]).read_text()
        assert digest in artifact_text
        assert "memory-peak" in artifact_text
    assert "<animate" in Path(rendered["animated_svg"]).read_text()
    assert "<animate" not in Path(rendered["svg"]).read_text()


def test_hardware_projection_preserves_global_invalidity_and_uncertainty(
    tmp_path: Path,
) -> None:
    report, _sentinel = _full_retained_report(tmp_path)
    source = json.loads(report.read_text())
    profile_path = Path(source["container_profile_report"])
    profile = json.loads(profile_path.read_text())
    hardware = profile["hardware_perf"]
    hardware["counters"] = hardware["counters"][:1]
    hardware["instrumentation_valid"] = False
    hardware["all_requested_metrics_valid"] = False
    hardware["anomalies"] = ["truncated perf output"]
    hardware["window_start_alignment_error_ns"] = 2_000
    hardware["window_end_alignment_error_ns"] = 7_000
    hardware["provenance"]["perf_version"] = "perf version retained"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")

    bundle = build_evidence_bundle(report)

    stream = next(stream for stream in bundle.streams if stream.metric == "hardware.cycles")
    assert stream.status == "invalid"
    assert stream.records[0].alignment_error_ns == 7_000
    assert "truncated perf output" in " ".join(stream.reasons)
    manifest = next(item for item in bundle.profilers if item.profiler_id == "compat-hardware-perf")
    assert manifest.status == "invalid"
    assert manifest.provenance["perf_version"] == "perf version retained"
    assert bundle.extensions["evidence_status"] == "invalid"
    assert bundle.extensions["all_selected_profilers_valid"] is False


def test_hardware_projection_fails_closed_when_invalid_report_has_no_rows(
    tmp_path: Path,
) -> None:
    report, _sentinel = _full_retained_report(tmp_path)
    source = json.loads(report.read_text())
    profile_path = Path(source["container_profile_report"])
    profile = json.loads(profile_path.read_text())
    hardware = profile["hardware_perf"]
    hardware["counters"] = []
    hardware["instrumentation_valid"] = False
    hardware["all_requested_metrics_valid"] = False
    hardware["anomalies"] = ["perf output omitted every counter"]
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")

    bundle = build_evidence_bundle(report)

    stream = next(stream for stream in bundle.streams if stream.stream_id == "hardware-counters")
    assert stream.status == "invalid"
    assert stream.records[0].value is None
    assert "omitted every counter" in " ".join(stream.reasons)
    assert bundle.extensions["evidence_status"] == "invalid"


def test_legacy_hardware_window_without_uncertainty_is_not_claimed_exact(
    tmp_path: Path,
) -> None:
    report, _sentinel = _full_retained_report(tmp_path)
    source = json.loads(report.read_text())
    profile_path = Path(source["container_profile_report"])
    profile = json.loads(profile_path.read_text())
    hardware = profile["hardware_perf"]
    hardware["counters"] = hardware["counters"][:1]
    hardware.pop("window_start_alignment_error_ns")
    hardware.pop("window_end_alignment_error_ns")
    hardware["instrumentation_valid"] = True
    hardware["all_requested_metrics_valid"] = True
    hardware["anomalies"] = []
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")

    bundle = build_evidence_bundle(report)

    stream = next(stream for stream in bundle.streams if stream.metric == "hardware.cycles")
    assert stream.status == "invalid"
    assert stream.records[0].alignment_error_ns == 1_400_000_000
    assert "lacks counter-window alignment uncertainty" in " ".join(stream.reasons)


def test_fallback_scope_and_timeline_are_explicitly_limited(tmp_path: Path) -> None:
    report, _sentinel = _full_retained_report(tmp_path)
    payload = json.loads(report.read_text())
    payload.pop("measured_scope")
    payload.pop("canonical_lifecycle")
    payload.pop("canonical_clock_reference")
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    bundle = build_evidence_bundle(report)

    assert bundle.scope.kind == "compatibility-projected-disposable-container-cgroup-subtree"
    assert bundle.scope.extensions["scope_identity_limited"] is True
    assert bundle.timeline.start_ns == 1_000_000_000
    assert bundle.timeline.end_ns == 2_900_004_000
    assert bundle.provenance["scope_identity_limited"] is True
    assert (
        bundle.streams[
            next(index for index, stream in enumerate(bundle.streams) if stream.stream_id == "replay-actions")
        ].status
        == "unalignable"
    )


def test_accepts_direct_retained_profile_report_path(tmp_path: Path) -> None:
    report, sentinel = _full_retained_report(tmp_path)
    pipeline = json.loads(report.read_text())
    profile_path = Path(pipeline["container_profile_report"])

    bundle = build_evidence_bundle(profile_path)

    streams = {stream.stream_id: stream for stream in bundle.streams}
    assert bundle.run_id == RUN_ID
    assert streams["memory-peak"].records[0].value == 90_000_000
    assert streams["process-lifecycles"].status == "unalignable"
    assert streams["hardware-cache-misses"].status == "invalid"
    assert streams["replay-actions"].status == "missing"
    assert streams["labeled-phases"].status == "missing"
    assert sentinel.read_text() == "1\n"


@pytest.mark.parametrize(
    ("canonical_offset", "source_offset", "expected_offset"),
    [(0, 0, 0), (-250, 1_000, -1_250)],
)
def test_replay_and_phase_use_authoritative_time_namespace_offset_mapping(
    tmp_path: Path,
    canonical_offset: int,
    source_offset: int,
    expected_offset: int,
) -> None:
    report, _sentinel = _full_retained_report(tmp_path)
    pipeline = json.loads(report.read_text())
    canonical_reference = {
        **_clock_reference(),
        "time_namespace": "time:[host]",
        "time_namespace_monotonic_offset_ns": canonical_offset,
    }
    pipeline["canonical_clock_reference"] = canonical_reference
    pipeline["canonical_lifecycle"]["clock_reference"] = canonical_reference
    report.write_text(json.dumps(pipeline, indent=2, sort_keys=True) + "\n")

    replay_path = Path(pipeline["replay_report"])
    replay = json.loads(replay_path.read_text())
    replay["clock_reference"] = {
        **_clock_reference(),
        "time_namespace": "time:[sandbox]",
        "time_namespace_monotonic_offset_ns": source_offset,
    }
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")

    phase_path = Path(pipeline["phase_log_path"])
    phase_rows = [json.loads(line) for line in phase_path.read_text().splitlines()]
    phase_rows[0]["clock_reference"] = replay["clock_reference"]
    _write_jsonl(phase_path, phase_rows)

    bundle = build_evidence_bundle(report)
    streams = {stream.stream_id: stream for stream in bundle.streams}
    mappings = {mapping.source_clock_id: mapping for mapping in bundle.clock_mappings}

    replay_mapping = mappings["replay-monotonic"]
    phase_mapping = mappings["phase-monotonic"]
    assert replay_mapping.status == phase_mapping.status == "valid"
    assert replay_mapping.method == phase_mapping.method == "linux-time-namespace-offset"
    assert replay_mapping.offset_ns == phase_mapping.offset_ns == expected_offset
    action = streams["replay-actions"].records[0]
    assert action.canonical_start_ns == action.native_start_ns + expected_offset
    phase = streams["labeled-phases"].records[0]
    assert phase.canonical_start_ns == phase.native_start_ns + expected_offset


def test_replay_and_phase_remain_unalignable_across_boots(tmp_path: Path) -> None:
    report, _sentinel = _full_retained_report(tmp_path)
    pipeline = json.loads(report.read_text())
    canonical_reference = {
        **_clock_reference(),
        "boot_id": "canonical-boot",
        "time_namespace": "time:[host]",
        "time_namespace_monotonic_offset_ns": 0,
    }
    pipeline["canonical_clock_reference"] = canonical_reference
    pipeline["canonical_lifecycle"]["clock_reference"] = canonical_reference
    report.write_text(json.dumps(pipeline, indent=2, sort_keys=True) + "\n")

    source_reference = {
        **_clock_reference(),
        "boot_id": "source-boot",
        "time_namespace": "time:[sandbox]",
        "time_namespace_monotonic_offset_ns": 0,
    }
    replay_path = Path(pipeline["replay_report"])
    replay = json.loads(replay_path.read_text())
    replay["clock_reference"] = source_reference
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    phase_path = Path(pipeline["phase_log_path"])
    phase_rows = [json.loads(line) for line in phase_path.read_text().splitlines()]
    phase_rows[0]["clock_reference"] = source_reference
    _write_jsonl(phase_path, phase_rows)

    bundle = build_evidence_bundle(report)
    streams = {stream.stream_id: stream for stream in bundle.streams}
    mappings = {mapping.source_clock_id: mapping for mapping in bundle.clock_mappings}

    assert mappings["replay-monotonic"].status == "unalignable"
    assert mappings["phase-monotonic"].status == "unalignable"
    action = streams["replay-actions"].records[0]
    phase = streams["labeled-phases"].records[0]
    assert action.status == phase.status == "unalignable"
    assert action.canonical_start_ns is None
    assert phase.canonical_start_ns is None
    assert "proven common boot" in action.reasons[0]


def test_projection_requires_top_level_run_id_and_safe_output_location(
    tmp_path: Path,
) -> None:
    report = _write_json(tmp_path / "profile_report.json", {"schema_version": "x"})

    with pytest.raises(EvidenceProjectionError, match="top-level run_id"):
        build_evidence_bundle(report)

    report.write_text(json.dumps({"run_id": RUN_ID}) + "\n")
    with pytest.raises(EvidenceProjectionError, match="written beside"):
        build_evidence_bundle(report, tmp_path / "nested" / "bundle.json")
    with pytest.raises(EvidenceProjectionError, match="must not overwrite"):
        build_evidence_bundle(report, report)

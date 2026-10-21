# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
from offline_replayer.measurement import container as container_module
from offline_replayer.measurement import container_coordinator as bridge
from offline_replayer.measurement.cgroup_resources import MEMORY_REPORT_SCHEMA_VERSION
from offline_replayer.measurement.container import ProfileConfig, run_profile
from offline_replayer.measurement.coordinator import ExecutionPlan, MountBehavior
from offline_replayer.measurement.evidence import (
    LifecycleMarker,
    read_bundle,
    validate_raw_artifacts,
)
from offline_replayer.measurement.network_policy import NetworkPolicy


def _config(tmp_path: Path, **overrides: object) -> ProfileConfig:
    values: dict[str, object] = {
        "command": "printf work",
        "prepare_command": None,
        "output_dir": tmp_path,
        "image": "example/task:latest",
        "container": None,
        "cpuset_cpus": "0",
        "cpu_limit": None,
        "allocated_cpus": None,
        "workdir": "/workdir",
        "user": None,
        "env": (),
        "mounts": (),
        "strace_mode": "none",
        "string_limit": 4096,
        "shell": "/bin/sh",
    }
    values.update(overrides)
    return ProfileConfig(**values)  # type: ignore[arg-type]


def _write_cgroup(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "cpu.max").write_text("max 100000\n")
    (path / "cpu.stat").write_text(
        "usage_usec 100000\nuser_usec 70000\nsystem_usec 30000\n" "nr_periods 10\nnr_throttled 0\nthrottled_usec 0\n"
    )
    (path / "memory.current").write_text("33554432\n")
    (path / "memory.peak").write_text("33554432\n")
    (path / "memory.swap.current").write_text("0\n")
    (path / "memory.swap.peak").write_text("0\n")
    (path / "memory.high").write_text("max\n")
    (path / "memory.max").write_text("max\n")
    (path / "memory.swap.high").write_text("max\n")
    (path / "memory.swap.max").write_text("0\n")
    events = "high 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n"
    (path / "memory.events").write_text(events)
    (path / "memory.events.local").write_text(events)
    (path / "memory.pressure").write_text(
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n" "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
    )
    (path / "memory.stat").write_text("anon 16777216\nfile 8388608\nkernel 4194304\n")


class _FakeScheduler:
    def __init__(self) -> None:
        self.window: tuple[int, int] | None = None

    def start(self, _root_pid: int, *, cgroup_path: Path) -> None:
        assert cgroup_path.is_dir()

    def poll(self, *, timeout_ms: int = 0) -> None:
        assert timeout_ms >= 0

    def finish(self, *, window_start_ns: int, window_end_ns: int) -> dict[str, object]:
        self.window = (window_start_ns, window_end_ns)
        return {
            "schema_version": "test-scheduler.v1",
            "window_start_ns": window_start_ns,
            "window_end_ns": window_end_ns,
            "scheduled_runtime_sec": 0.2,
            "peak_simultaneous_logical_cores": 1,
            "events_observed": 2,
            "trace_complete": True,
            "anomalies": [],
        }

    def write_events(self, path: Path) -> None:
        assert self.window is not None
        start, end = self.window
        first = start + max(1, (end - start) // 4)
        second = start + max(2, (end - start) // 2)
        path.write_text(
            json.dumps({"timestamp_ns": first, "cpu": 0, "tid": 101, "entering": True})
            + "\n"
            + json.dumps({"timestamp_ns": second, "cpu": 0, "tid": 101, "entering": False})
            + "\n"
        )

    def close(self) -> None:
        pass


class _FakeHardware:
    def __init__(self, *, output_dir: Path, events: tuple[str, ...], **_kwargs: object) -> None:
        self.output_dir = output_dir
        self.events = events

    def attach(self, *, cgroup_path: Path) -> None:
        assert cgroup_path.is_dir()

    def enable(self) -> None:
        pass

    def poll(self, *, timeout_ms: int = 0) -> None:
        assert timeout_ms >= 0

    def finish(self, *, window_start_ns: int, window_end_ns: int) -> dict[str, object]:
        (self.output_dir / "hardware_perf_stat.csv").write_text("1;;cycles;1;100.0\n")
        return {
            "schema_version": "cgroup-perf-hardware.v1",
            "window_start_ns": window_start_ns,
            "window_end_ns": window_end_ns,
            "window_start_alignment_error_ns": 100,
            "window_end_alignment_error_ns": 100,
            "instrumentation_valid": True,
            "all_requested_metrics_valid": True,
            "counters": [
                {
                    "event": event,
                    "status": "valid",
                    "value": index + 1,
                    "unit": "count",
                    "time_running_fraction": 1.0,
                    "valid": True,
                    "invalid_reason": None,
                }
                for index, event in enumerate(self.events)
            ],
        }

    def write_report(self, report: dict[str, object], path: Path) -> None:
        path.write_text(json.dumps(report, sort_keys=True) + "\n")

    def close(self) -> None:
        pass


class _FakeSession:
    capabilities = bridge.CoordinatedDockerSession.capabilities
    created = 0
    launches = 0
    returncode = 0

    def __init__(self, config: ProfileConfig) -> None:
        type(self).created += 1
        self.config = config
        self.output_dir = config.output_dir
        self.cgroup_root = config.output_dir / "fake-cgroup-root"
        self.cgroup_parent = self.cgroup_root / "job"
        self.cgroup_path = self.cgroup_parent / "fake-cgroup"
        _write_cgroup(self.cgroup_root)
        _write_cgroup(self.cgroup_parent)
        _write_cgroup(self.cgroup_path)
        self.root_pid = 100
        self.command = ["docker", "run", "one-session"]
        self.shared: dict[str, object] = {}
        self._poll_callbacks: list[object] = []
        self._remaining_descendants: list[dict[str, object]] = []
        self.legacy = container_module

    def register_poll_callback(self, callback: object) -> None:
        self._poll_callbacks.append(callback)

    def launch(self, plan: object, *, run_id: str, marker: object):
        type(self).launches += 1
        return bridge.MeasuredScope(
            scope_id="fake-descendant-cgroup",
            kind="disposable-container-cgroup-subtree",
            identity={
                "cgroup_path": str(self.cgroup_path),
                "cgroup_inode": self.cgroup_path.stat().st_ino,
                "time_namespace": "time:[123]",
                "run_id": run_id,
            },
            includes_descendants=True,
        )

    def release_sandbox(self, marker: object) -> None:
        pass

    def wait_command_ready(self) -> None:
        for callback in self._poll_callbacks:
            callback()  # type: ignore[operator]

    def release_command(self, marker: object) -> None:
        # Match the real entrypoint: scheduler mode overwrites the host's
        # early start value with a placeholder after the gate is released.
        self.output_dir.joinpath("start_ns").write_text("0\n")

    def poll_command(self, timeout_sec: float) -> bool:
        time.sleep(max(timeout_sec, 0.006))
        (self.cgroup_path / "cpu.stat").write_text(
            "usage_usec 300000\nuser_usec 220000\nsystem_usec 80000\n"
            "nr_periods 11\nnr_throttled 0\nthrottled_usec 0\n"
        )
        (self.cgroup_path / "memory.current").write_text("50331648\n")
        (self.cgroup_path / "memory.peak").write_text("67108864\n")
        self.output_dir.joinpath("exit_code").write_text(f"{self.returncode}\n")
        return True

    def begin_finish(self, marker: object) -> None:
        pass

    def poll_finish(self, timeout_sec: float) -> bool:
        time.sleep(max(timeout_sec, 0.006))
        return True

    def finish(self, marker: object) -> dict[str, object]:
        self.output_dir.joinpath("exit_code").write_text(f"{self.returncode}\n")
        self.output_dir.joinpath("command.stdout.txt").write_text("work")
        self.output_dir.joinpath("command.stderr.txt").write_text("")
        self.output_dir.joinpath("process.strace.log").write_text(
            '100 1700000000.000000001 execve("/bin/sh", '
            '["/bin/sh", "-c", "printf work"], 0x1) = 0\n'
            "100 1700000000.100000001 +++ exited with 0 +++\n"
        )
        return {
            "returncode": self.returncode,
            "stdout": "",
            "stderr": "",
            "remaining_descendant_pids_at_finish_gate": [],
        }

    def abort(self, reason: str) -> dict[str, object]:
        return {"returncode": 1, "stderr": reason}

    def close(self) -> None:
        pass


class _CpuConstraintSession(_FakeSession):
    mutation = ""

    def __init__(self, config: ProfileConfig) -> None:
        super().__init__(config)
        if self.mutation == "leaf-finite":
            (self.cgroup_path / "cpu.max").write_text("100000 100000\n")

    def poll_command(self, timeout_sec: float) -> bool:
        complete = super().poll_command(timeout_sec)
        if self.mutation == "leaf-throttle":
            (self.cgroup_path / "cpu.stat").write_text(
                "usage_usec 300000\nuser_usec 220000\nsystem_usec 80000\n"
                "nr_periods 11\nnr_throttled 1\nthrottled_usec 2000\n"
            )
        elif self.mutation == "leaf-regression":
            (self.cgroup_path / "cpu.stat").write_text(
                "usage_usec 90000\nuser_usec 60000\nsystem_usec 30000\n"
                "nr_periods 9\nnr_throttled 0\nthrottled_usec 0\n"
            )
        elif self.mutation == "ancestor-throttle":
            (self.cgroup_parent / "cpu.stat").write_text(
                "usage_usec 110000\nuser_usec 75000\nsystem_usec 35000\n"
                "nr_periods 11\nnr_throttled 1\nthrottled_usec 1000\n"
            )
        return complete


class _FailedCommandSession(_FakeSession):
    returncode = 7


def test_run_profile_forwards_cancellation_to_coordinated_docker_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cancel_requested():
        return False

    captured: list[object] = []
    monkeypatch.setattr(
        container_module,
        "_inspect_image_identity",
        lambda _config: {"id": "sha256:" + "a" * 64},
    )

    def fake_coordinated(
        _config,
        *,
        image_identity,
        run_id,
        cancel_requested=None,
    ):
        captured.extend((image_identity, run_id, cancel_requested))
        raise container_module.ProfileError("coordinated sentinel")

    monkeypatch.setattr(bridge, "run_coordinated_docker_profile", fake_coordinated)

    with pytest.raises(container_module.ProfileError, match="coordinated sentinel"):
        run_profile(
            _config(tmp_path, profile_sets=("cpu-average",)),
            cancel_requested=cancel_requested,
        )

    assert captured[0] == {"id": "sha256:" + "a" * 64}
    assert isinstance(captured[1], str)
    assert captured[2] is cancel_requested


def test_coordinated_command_failure_updates_memory_before_bundle_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bridge, "CoordinatedDockerSession", _FailedCommandSession)
    monkeypatch.setattr(
        container_module,
        "controller_source_sha256",
        lambda _source_root=None: "f" * 64,
    )
    monkeypatch.setattr(
        container_module,
        "_build_scheduler_tracer",
        lambda _config: _FakeScheduler(),
    )
    monkeypatch.setattr(
        container_module,
        "_inspect_image_identity",
        lambda _config: {
            "id": "sha256:" + "a" * 64,
            "repo_digests": [],
            "created": "2026-01-01T00:00:00Z",
        },
    )

    report = run_profile(_config(tmp_path, profile_sets=("cpu-compare",), strace_mode="none"))
    bundle = read_bundle(Path(report["evidence_bundle"]["path"]))
    raw_memory = bundle.extensions["raw_profiler_reports"]["cgroup-memory"]
    peak = next(stream for stream in bundle.streams if stream.metric == "memory.peak")

    assert report["command_exit_code"] == 7
    assert report["memory"]["censoring"] == "functional_failure"
    assert report["memory"]["natural_peak_trustworthy"] is False
    assert raw_memory["censoring"] == "functional_failure"
    assert raw_memory["natural_peak_trustworthy"] is False
    assert peak.records[0].attributes["natural_peak_trustworthy"] is False


def test_all_profile_sets_share_one_scope_and_write_one_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeSession.created = 0
    _FakeSession.launches = 0
    monkeypatch.setattr(bridge, "CoordinatedDockerSession", _FakeSession)
    monkeypatch.setattr(bridge, "PerfCgroupHardwareProfiler", _FakeHardware)
    monkeypatch.setattr(
        container_module,
        "controller_source_sha256",
        lambda _source_root=None: "f" * 64,
    )
    monkeypatch.setattr(
        container_module,
        "_build_scheduler_tracer",
        lambda _config: _FakeScheduler(),
    )
    monkeypatch.setattr(
        container_module,
        "_inspect_image_identity",
        lambda _config: {
            "id": "sha256:" + "a" * 64,
            "repo_digests": ["example/task@sha256:" + "b" * 64],
            "created": "2026-01-01T00:00:00Z",
        },
    )
    config = _config(
        tmp_path,
        profile_sets=("cpu-compare", "software-deep", "hardware-deep"),
        memory_sample_interval_sec=0.005,
        mounts=("type=bind,src=/fixture/checkout,dst=/tasktrace-source,readonly",),
        workload_identity={"task_id": "fixture-1", "api_token": "do-not-archive"},
        source_identity={"trajectory_sha256": "c" * 64},
        expected_outcome={"kind": "exit", "code": 0},
    )

    report = run_profile(config)

    assert _FakeSession.created == 1
    assert _FakeSession.launches == 1
    attachment = report["evidence_bundle"]
    assert attachment["scope_id"] == "fake-descendant-cgroup"
    assert attachment["resolved_profiler_ids"] == [
        "scheduler-concurrency",
        "hardware-perf",
        "cgroup-cpu",
        "cgroup-memory",
        "process-exec",
    ]
    bundle = read_bundle(Path(attachment["path"]))
    validate_raw_artifacts(bundle, tmp_path)
    artifact_ids = [artifact.artifact_id for artifact in bundle.artifacts]
    assert len(artifact_ids) == len(set(artifact_ids))
    artifact_paths = {artifact.artifact_id: artifact.relative_path for artifact in bundle.artifacts}
    assert artifact_paths["cgroup-cpu-before"] == ("cgroup_cpu.stat.observer_start.txt")
    assert artifact_paths["cgroup-cpu-after"] == ("cgroup_cpu.stat.controlled_finish_gate.txt")
    assert artifact_paths["cgroup-cpu-source-snapshots"] == ("cgroup_cpu_source_snapshots.json")
    assert artifact_paths["cgroup-cpu-before"] != "cpu.stat.before"
    assert artifact_paths["cgroup-cpu-after"] != "cpu.stat.after"
    assert artifact_ids.count("cgroup-resource-timeseries") == 1
    assert artifact_ids.count("cgroup-memory-source-snapshots") == 1
    assert artifact_ids.count("cgroup-cpu-source-snapshots") == 1
    cpu_streams = [stream for stream in bundle.streams if stream.profiler_id == "cgroup-cpu"]
    assert cpu_streams
    assert all("cgroup-cpu-source-snapshots" in stream.raw_artifact_ids for stream in cpu_streams)
    memory_streams = [stream for stream in bundle.streams if stream.profiler_id == "cgroup-memory"]
    assert memory_streams
    assert all("cgroup-resource-timeseries" in stream.raw_artifact_ids for stream in memory_streams)
    exact_memory_streams = [
        stream
        for stream in memory_streams
        if stream.metric == "memory.peak"
        or stream.metric.startswith("memory.events.")
        or stream.metric.startswith("memory.pressure.")
    ]
    assert exact_memory_streams
    assert all("cgroup-memory-source-snapshots" in stream.raw_artifact_ids for stream in exact_memory_streams)
    source_snapshot = json.loads((tmp_path / "cgroup_memory_source_snapshots.json").read_text())
    assert source_snapshot["controllers"] == ["memory"]
    assert "ancestor_cpu" not in source_snapshot
    finish_peak = [
        row
        for row in source_snapshot["reads"]
        if row["phase"] == "controlled_finish_gate"
        and row["source_scope"] == "leaf"
        and row["filename"] == "memory.peak"
    ]
    assert [row["raw_text"] for row in finish_peak] == ["67108864\n"]
    cpu_source_snapshot = json.loads((tmp_path / "cgroup_cpu_source_snapshots.json").read_text())
    assert (tmp_path / "cgroup_cpu.max.observer_start.txt").read_text() == ("max 100000\n")
    assert (tmp_path / "cgroup_cpu.max.controlled_finish_gate.txt").read_text() == "max 100000\n"
    assert [(row["phase"], row["filename"]) for row in cpu_source_snapshot["leaf"]["reads"]] == [
        ("observer_start", "cpu.max"),
        ("observer_start", "cpu.stat"),
        ("controlled_finish_gate", "cpu.max"),
        ("controlled_finish_gate", "cpu.stat"),
    ]
    ancestor_reads = cpu_source_snapshot["ancestors"]["reads"]
    assert {row["source_scope"] for row in ancestor_reads} == {"ancestor:job", "ancestor:."}
    assert bundle.run_id == report["run_id"]
    assert bundle.scope.includes_descendants is True
    assert bundle.scope.kind == "disposable-container-cgroup-subtree"
    assert report["cpu"]["wall_sec"] < 1
    assert report["cpu"]["average_logical_cores_used"] == pytest.approx(report["cpu"]["average_logical_cores"])
    assert report["cpu"]["valid_for_cpu_performance"] is True
    assert report["cpu"]["leaf_cpu"]["valid_for_cpu_performance"] is True
    assert report["cpu"]["ancestor_cpu"]["valid_for_cpu_performance"] is True
    assert report["canonical_clock_reference"]["clock_id"] == "CLOCK_MONOTONIC"
    assert bundle.provenance["canonical_lifecycle"]["clock_reference"] == report["canonical_clock_reference"]
    assert {stream.metric for stream in bundle.streams} >= {
        "cpu.average_logical_cores",
        "cpu.interval_average_logical_cores",
        "memory.current",
        "memory.peak",
        "memory.events.oom",
        "memory.pressure.some.total_stall",
        "scheduler.concurrent_logical_cpus",
        "scheduler.logical_cpu_activity",
        "software.process_count",
        "hardware.cycles",
    }
    process_manifest = next(item for item in bundle.profilers if item.profiler_id == "process-exec")
    assert process_manifest.intrusive is True
    manifests = {item.profiler_id: item for item in bundle.profilers}
    for profiler_id, domain in (
        ("cgroup-cpu", "cpu.capacity"),
        ("scheduler-concurrency", "cpu.capacity"),
        ("hardware-perf", "cpu.capacity"),
        ("cgroup-memory", "memory.capacity"),
    ):
        manifest = manifests[profiler_id]
        assert manifest.status == "invalid"
        perturbation = manifest.extensions["measurement_perturbations"][0]
        assert perturbation["invalidating_profiler_id"] == "process-exec"
        assert perturbation["measurement_domains"] == [domain]
        affected_streams = [stream for stream in bundle.streams if stream.profiler_id == profiler_id]
        assert affected_streams
        assert all(stream.status != "valid" for stream in affected_streams)
        assert all(record.status != "valid" for stream in affected_streams for record in stream.records)
    memory_peak = next(stream for stream in bundle.streams if stream.metric == "memory.peak")
    assert memory_peak.records[0].attributes["natural_peak_trustworthy"] is False
    assert process_manifest.status == "unalignable"
    assert process_manifest.extensions["measurement_domains"] == ["software.process"]
    assert process_manifest.extensions["measurement_perturbations"] == []
    process_clock = next(item for item in bundle.clock_mappings if item.source_clock_id == "strace-realtime")
    assert process_clock.status == "unalignable"
    assert report["memory"]["peak_valid"] is True
    assert report["memory"]["natural_peak_trustworthy"] is False
    assert report["memory"]["schema_version"] == MEMORY_REPORT_SCHEMA_VERSION
    assert report["memory"]["lifetime"]["exact_through_controlled_finish_gate"] is True
    assert report["memory"]["observation_boundary"] == "controlled-finish-gate"
    assert report["memory"]["observation_boundary_complete"] is True
    assert report["profiling_valid"] is False
    assert "strace" in " ".join(report["memory"]["natural_peak_invalid_reasons"])
    plan = bundle.provenance["execution_plan"]
    assert plan["workload_identity"]["task_id"] == "fixture-1"
    assert plan["workload_identity"]["api_token"]["redacted"] is True
    assert plan["source_identity"] == {
        "trajectory_sha256": "c" * 64,
        "controller_source_sha256": "f" * 64,
    }
    controller_mount = next(mount for mount in plan["mounts"] if mount["target"] == "/tasktrace-source")
    assert controller_mount["source_identity"] == {
        "kind": "controller-source-tree",
        "controller_source_sha256": "f" * 64,
        "host_path_in_comparability": False,
    }
    assert "do-not-archive" not in Path(attachment["path"]).read_text()
    hardware = next(stream for stream in bundle.streams if stream.metric == "hardware.cycles")
    assert hardware.shape == "window_aggregate"
    assert hardware.records[0].canonical_end_ns > hardware.records[0].canonical_start_ns
    request = bundle.provenance["profile_request"]
    assert request["configuration"]["cgroup-memory"]["sample_interval_sec"] == 0.005
    assert request["configuration"]["hardware-perf"]["events"][0] == "cycles"
    rendered = json.loads((tmp_path / "unified_profile_artifacts.json").read_text())
    assert rendered["workload_executions"] == 1
    assert Path(rendered["html"]).is_file()
    assert Path(rendered["svg"]).is_file()
    assert Path(rendered["animated_svg"]).is_file()
    for key in ("html", "svg", "animated_svg"):
        rendered_text = Path(rendered[key]).read_text()
        assert rendered["view_model_sha256"] in rendered_text
        assert "cgroup-resource-timeseries" in rendered_text
        assert "cgroup-memory-source-snapshots" in rendered_text


def test_explicit_baseline_uses_coordinator_with_empty_profiler_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeSession.created = 0
    _FakeSession.launches = 0
    monkeypatch.setattr(bridge, "CoordinatedDockerSession", _FakeSession)
    config = _config(
        tmp_path,
        coordinated_baseline=True,
        strace_mode="execve",
        workload_identity={"task_id": "baseline-fixture"},
        source_identity={"trajectory_sha256": "c" * 64},
        expected_outcome={"kind": "exit", "code": 0},
    )

    result = bridge.run_coordinated_docker_profile(
        config,
        image_identity={"id": "sha256:" + "a" * 64},
        run_id="00000000-0000-4000-8000-000000000009",
    )

    bundle = result.coordination.evidence_bundle
    assert _FakeSession.created == 1
    assert _FakeSession.launches == 1
    assert result.effective_config.strace_mode == "none"
    assert result.effective_config.scheduler_concurrency is False
    assert result.effective_config.memory_profile is False
    assert result.coordination.resolved_profiler_ids == ("cgroup-cpu",)
    assert [item.profiler_id for item in bundle.profilers] == ["cgroup-cpu"]
    assert bundle.streams
    assert {stream.profiler_id for stream in bundle.streams} == {"cgroup-cpu"}
    assert all("cgroup-cpu-source-snapshots" in stream.raw_artifact_ids for stream in bundle.streams)
    assert result.coordination.raw_reports["cgroup-cpu"]["valid_for_cpu_performance"] is True
    assert result.cpu_report["schema_version"] == "cgroup-cpu-coordinated.v1"
    assert result.cpu_report["scope_id"] == "fake-descendant-cgroup"
    assert result.cpu_report["valid_for_cpu_performance"] is True
    assert result.cpu_report["leaf_cpu"]["valid_for_cpu_performance"] is True
    assert result.cpu_report["ancestor_cpu"]["valid_for_cpu_performance"] is True
    assert bundle.provenance["profile_request"] == {
        "schema_version": "tasktrace-profile-request.v1",
        "mode": "baseline",
        "profile_sets": [],
        "profilers": [],
        "configuration": {},
    }
    assert bundle.provenance["execution_plan_sha256"] == (result.coordination.execution_plan_digest)
    assert bundle.scope.includes_descendants is True
    assert result.completed.returncode == 0


def test_coordinator_plan_uses_policy_comparability_not_ephemeral_network_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeSession.created = 0
    _FakeSession.launches = 0
    monkeypatch.setattr(bridge, "CoordinatedDockerSession", _FakeSession)
    policy = NetworkPolicy("allowlist", ("pypi.org",))
    comparable_enforcement = {
        "schema_version": "tasktrace-docker-network-enforcement-comparability.v1",
        "policy": policy.to_comparability_payload(),
        "backend": "docker-forward-proxy-v1",
        "allowed_ports": [80, 443],
        "external_network": "bridge",
        "external_network_identity": {"configuration_sha256": "d" * 64},
        "proxy_source_sha256": "a" * 64,
        "proxy_image_identity": {"id": "sha256:" + "b" * 64},
    }

    def configured(path: Path, ephemeral: str) -> ProfileConfig:
        return _config(
            path,
            coordinated_baseline=True,
            network_policy=policy,
            network_attachment=ephemeral,
            network_policy_enforcement={
                "state": "active",
                "ephemeral_internal_network": ephemeral,
                "proxy": {"container_name": ephemeral + "-proxy"},
                "comparability": comparable_enforcement,
            },
            env=(
                "HTTP_PROXY=http://tasktrace-egress-proxy:3128",
                "HTTPS_PROXY=http://tasktrace-egress-proxy:3128",
            ),
        )

    first = bridge.run_coordinated_docker_profile(
        configured(tmp_path / "first", "tasktrace-policy-first"),
        image_identity={"id": "sha256:" + "c" * 64},
        run_id="00000000-0000-4000-8000-000000000021",
    )
    second = bridge.run_coordinated_docker_profile(
        configured(tmp_path / "second", "tasktrace-policy-second"),
        image_identity={"id": "sha256:" + "c" * 64},
        run_id="00000000-0000-4000-8000-000000000022",
    )

    first_plan = first.coordination.evidence_bundle.provenance["execution_plan"]
    network_identity = first_plan["extensions"]["network_policy"]
    assert first_plan["network"] == "tasktrace-allowlist"
    assert network_identity["effective_mode"] == "allowlist"
    assert network_identity["allowed_hosts"] == ["pypi.org"]
    assert network_identity["enforcement"]["allowed_ports"] == [80, 443]
    assert "tasktrace-policy-first" not in json.dumps(network_identity)
    assert first.coordination.execution_plan_digest == (second.coordination.execution_plan_digest)


@pytest.mark.parametrize(
    ("mutation", "reason_fragment", "invalid_scope"),
    [
        ("leaf-finite", "leaf cgroup has a finite cpu.max", "leaf"),
        ("leaf-throttle", "leaf cgroup recorded CPU throttling", "leaf"),
        ("leaf-regression", "moved backwards", "leaf"),
        (
            "ancestor-throttle",
            "ancestor cgroup 'job' recorded CPU throttling",
            "ancestor",
        ),
    ],
)
def test_common_cpu_gate_fails_closed_in_coordinated_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason_fragment: str,
    invalid_scope: str,
) -> None:
    constrained_session = type(
        f"ConstrainedSession_{mutation.replace('-', '_')}",
        (_CpuConstraintSession,),
        {"mutation": mutation},
    )
    monkeypatch.setattr(bridge, "CoordinatedDockerSession", constrained_session)
    config = _config(tmp_path, coordinated_baseline=True)

    result = bridge.run_coordinated_docker_profile(
        config,
        image_identity={"id": "sha256:" + "a" * 64},
        run_id="00000000-0000-4000-8000-000000000010",
    )

    raw = result.coordination.raw_reports["cgroup-cpu"]
    assert raw["valid_for_cpu_performance"] is False
    assert reason_fragment in " ".join(raw["invalid_reasons"])
    assert raw[f"{invalid_scope}_cpu"]["valid_for_cpu_performance"] is False
    manifest = result.coordination.manifests[0]
    assert manifest.profiler_id == "cgroup-cpu"
    assert manifest.status == "invalid"
    cpu_streams = result.coordination.evidence_bundle.streams
    assert cpu_streams
    assert all(stream.profiler_id == "cgroup-cpu" for stream in cpu_streams)
    assert all(stream.status != "valid" for stream in cpu_streams)
    assert all("cgroup-cpu-source-snapshots" in stream.raw_artifact_ids for stream in cpu_streams)
    source = json.loads((tmp_path / "cgroup_cpu_source_snapshots.json").read_text())
    assert all(row["acquisition_end_ns"] >= row["acquisition_start_ns"] for row in source["leaf"]["reads"])


def test_run_local_replay_mount_does_not_break_plan_comparability(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline"
    profile_root = tmp_path / "profile"
    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    for checkout in (checkout_a, checkout_b):
        package = checkout / "offline_replayer"
        package.mkdir(parents=True)
        package.joinpath("controller.py").write_text("CONTROLLER_VERSION = 1\n")
    baseline = _config(
        baseline_root / "container_profile",
        mounts=(
            f"type=bind,src={baseline_root / 'replay'},dst=/tasktrace",
            f"type=bind,src={checkout_a},dst=/tasktrace-source,readonly",
        ),
    )
    profiled = _config(
        profile_root / "container_profile",
        mounts=(
            f"type=bind,src={profile_root / 'replay'},dst=/tasktrace",
            f"type=bind,src={checkout_b},dst=/tasktrace-source,readonly",
        ),
    )

    digest = container_module.controller_source_sha256(checkout_a)
    baseline_mounts = bridge._mounts(baseline)
    profile_mounts = bridge._mounts(profiled)
    assert baseline_mounts == profile_mounts
    replay_mount = next(mount for mount in baseline_mounts if mount.target == "/tasktrace")
    assert replay_mount.source_identity == {
        "kind": "run-local-replay-output",
        "host_path_in_comparability": False,
    }
    source_mount = next(mount for mount in baseline_mounts if mount.target == "/tasktrace-source")
    assert source_mount.source_identity == {
        "kind": "controller-source-tree",
        "controller_source_sha256": digest,
        "host_path_in_comparability": False,
    }


def test_controller_source_digest_is_content_based_and_not_process_cached(
    tmp_path: Path,
) -> None:
    first_checkout = tmp_path / "checkout-a" / "offline_replayer"
    second_checkout = tmp_path / "checkout-b" / "offline_replayer"
    first_checkout.mkdir(parents=True)
    second_checkout.mkdir(parents=True)
    first_file = first_checkout / "controller.py"
    second_file = second_checkout / "controller.py"
    first_file.write_text("VERSION = 1\n")
    second_file.write_text("VERSION = 1\n")

    initial = container_module.controller_source_sha256(first_checkout.parent)
    assert container_module.controller_source_sha256(second_checkout.parent) == initial

    first_file.write_text("VERSION = 2\n")

    assert container_module.controller_source_sha256(first_checkout.parent) != initial


def test_rust_controller_mount_is_content_bound_and_change_fails_closed(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "tasktrace"
    executable.write_bytes(b"native-controller-v1")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    config = _config(
        tmp_path / "container-profile",
        mounts=(f"type=bind,src={executable}," "dst=/tasktrace-controller/tasktrace,readonly",),
        source_identity={"rust_controller_binary_sha256": digest},
    )

    mounts = bridge._mounts(config)

    assert mounts[0].source_identity == {
        "kind": "rust-controller-binary",
        "sha256": digest,
        "host_path_in_comparability": False,
    }
    executable.write_bytes(b"native-controller-v2")
    with pytest.raises(
        container_module.ProfileError,
        match="changed after adapter preflight",
    ):
        bridge._mounts(config)


def test_controller_source_change_changes_authoritative_execution_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeSession.created = 0
    _FakeSession.launches = 0
    monkeypatch.setattr(bridge, "CoordinatedDockerSession", _FakeSession)
    config_a = _config(
        tmp_path / "a",
        coordinated_baseline=True,
        source_identity={
            "trajectory_sha256": "c" * 64,
            "controller_source_sha256": "0" * 64,
        },
    )
    config_b = _config(
        tmp_path / "b",
        coordinated_baseline=True,
        source_identity={
            "trajectory_sha256": "c" * 64,
            "controller_source_sha256": "0" * 64,
        },
    )
    digests = iter(("1" * 64, "2" * 64))
    monkeypatch.setattr(
        container_module,
        "controller_source_sha256",
        lambda: next(digests),
    )

    first = bridge.run_coordinated_docker_profile(
        config_a,
        image_identity={"id": "sha256:" + "a" * 64},
        run_id="00000000-0000-4000-8000-00000000000a",
    )
    second = bridge.run_coordinated_docker_profile(
        config_b,
        image_identity={"id": "sha256:" + "a" * 64},
        run_id="00000000-0000-4000-8000-00000000000b",
    )

    first_plan = first.coordination.evidence_bundle.provenance["execution_plan"]
    second_plan = second.coordination.evidence_bundle.provenance["execution_plan"]
    assert first_plan["source_identity"]["controller_source_sha256"] == "1" * 64
    assert second_plan["source_identity"]["controller_source_sha256"] == "2" * 64
    assert first.coordination.execution_plan_digest != (second.coordination.execution_plan_digest)


def test_profile_sets_own_strace_perturbation_contract(tmp_path: Path) -> None:
    cpu = bridge.coordinated_effective_config(_config(tmp_path, strace_mode="execve", profile_sets=("cpu-compare",)))
    software = bridge.coordinated_effective_config(
        _config(tmp_path, strace_mode="none", profile_sets=("software-deep",))
    )

    assert cpu.strace_mode == "none"
    assert cpu.scheduler_concurrency is True
    assert cpu.memory_profile is True
    assert software.strace_mode == "process"


def test_unavailable_hardware_is_invalid_evidence_not_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class UnavailableHardware:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def attach(self, *, cgroup_path: Path) -> None:
            raise RuntimeError("PMU access denied")

        def close(self) -> None:
            pass

    monkeypatch.setattr(bridge, "CoordinatedDockerSession", _FakeSession)
    monkeypatch.setattr(bridge, "PerfCgroupHardwareProfiler", UnavailableHardware)
    config = _config(tmp_path, profile_sets=("hardware-deep",))

    result = bridge.run_coordinated_docker_profile(
        config,
        image_identity={"id": "sha256:" + "a" * 64},
        run_id="00000000-0000-4000-8000-000000000002",
    )

    manifest = result.coordination.manifests[0]
    assert manifest.profiler_id == "hardware-perf"
    assert manifest.status == "invalid"
    assert "PMU access denied" in " ".join(manifest.reasons)
    assert result.coordination.evidence_bundle.streams
    hardware_streams = [
        stream for stream in result.coordination.evidence_bundle.streams if stream.profiler_id == "hardware-perf"
    ]
    assert hardware_streams
    assert all(
        stream.status == "invalid" and stream.records and stream.records[0].value is None for stream in hardware_streams
    )


def test_docker_session_launches_one_popen_and_freezes_descendant_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    (tmp_path / "sandbox_attach_ready").write_text("ready\n")
    cgroup = tmp_path / "scope"
    cgroup.mkdir()
    popen_calls: list[list[str]] = []

    class FakePopen:
        returncode: int | None = None

        def __init__(self, command: list[str], **_kwargs: object) -> None:
            popen_calls.append(command)

        def poll(self) -> int | None:
            return self.returncode

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.returncode = 0
            return "", ""

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(bridge.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(
        container_module,
        "build_image_command",
        lambda _config, *, container_name: ["docker", "run", container_name],
    )
    monkeypatch.setattr(container_module, "_container_init_pid", lambda *_args: 321)
    monkeypatch.setattr(container_module, "_wait_for_blocked_init", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(container_module, "_host_cgroup_path_for_pid", lambda _pid: cgroup)
    monkeypatch.setattr(
        bridge.CoordinatedDockerSession,
        "_validated_cpu_placement",
        lambda _self: {
            "requested_cpuset_cpus": "0",
            "docker_host_config_cpuset_cpus": "0",
            "container_init_cpus_allowed_list": "0",
            "cgroup_cpuset_cpus_effective": "0",
            "validation_status": "passed",
        },
    )
    monkeypatch.setattr(
        container_module,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    plan = ExecutionPlan(
        workload_identity={"task": "one"},
        source_identity={"trace": "one"},
        command=("/bin/sh", "-c", "true"),
        cwd="/",
        mounts=tuple[MountBehavior, ...](),
        image_reference="sha256:" + "a" * 64,
        image_identity={"id": "sha256:" + "a" * 64},
        runtime="docker",
        network=None,
        expected_outcome={"kind": "exit"},
    )
    session = bridge.CoordinatedDockerSession(config)
    marker = LifecycleMarker("run_start", 1)

    scope = session.launch(
        plan,
        run_id="00000000-0000-4000-8000-000000000001",
        marker=marker,
    )

    assert len(popen_calls) == 1
    assert scope.includes_descendants is True
    assert scope.identity["root_pid"] == 321
    assert scope.identity["cgroup_inode"] == cgroup.stat().st_ino
    assert scope.identity["cpu_placement"]["validation_status"] == "passed"
    with pytest.raises(container_module.ProfileError, match="only once"):
        session.launch(
            plan,
            run_id="00000000-0000-4000-8000-000000000001",
            marker=marker,
        )
    session.close()


def test_docker_session_cpu_placement_requires_matching_runtime_and_kernel_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = Path(f"/proc/{os.getpid()}/status").read_text()
    allowed = next(
        line.partition(":")[2].strip() for line in status.splitlines() if line.startswith("Cpus_allowed_list:")
    )
    cgroup = tmp_path / "placement-cgroup"
    cgroup.mkdir()
    (cgroup / "cpuset.cpus.effective").write_text(allowed + "\n")
    config = _config(tmp_path / "output", cpuset_cpus=allowed)
    session = bridge.CoordinatedDockerSession(config)
    session.root_pid = os.getpid()
    session.cgroup_path = cgroup
    monkeypatch.setattr(
        container_module,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, json.dumps(allowed) + "\n", ""),
    )

    placement = session._validated_cpu_placement()

    assert placement == {
        "requested_cpuset_cpus": allowed,
        "docker_host_config_cpuset_cpus": allowed,
        "container_init_cpus_allowed_list": allowed,
        "cgroup_cpuset_cpus_effective": allowed,
        "validation_status": "passed",
    }

    monkeypatch.setattr(
        container_module,
        "_run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, json.dumps("999") + "\n", ""),
    )
    with pytest.raises(container_module.ProfileError, match="HostConfig.*differs"):
        session._validated_cpu_placement()


def test_docker_session_abort_never_reopens_consumed_fifo_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    released: list[str] = []

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return "", ""

    monkeypatch.setattr(
        bridge.CoordinatedDockerSession,
        "_write_gate_nonblocking",
        lambda _self, path: released.append(path.name),
    )
    for ready in (
        "sandbox_attach_ready",
        "scheduler_attach_ready",
        "scheduler_measurement_done",
    ):
        (tmp_path / ready).write_text("ready\n")

    session = bridge.CoordinatedDockerSession(config)
    session.process = FakeProcess()  # type: ignore[assignment]
    marker = LifecycleMarker("test", 1)
    session.release_sandbox(marker)
    session.release_command(marker)

    report = session.abort("later profiler failed")

    assert released == [
        "sandbox.start.gate",
        "scheduler.start.gate",
        "scheduler.finish.gate",
    ]
    assert report["aborted"] is True


def test_docker_session_abort_removes_container_after_client_already_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    remove_calls: list[list[str]] = []

    class ExitedProcess:
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return "", ""

    def fake_run(command, **_kwargs):
        remove_calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(container_module, "_run", fake_run)
    session = bridge.CoordinatedDockerSession(config)
    session.process = ExitedProcess()  # type: ignore[assignment]
    session._docker_run_started = True

    report = session.abort("cancelled")
    session.close()
    session.close()

    assert report["aborted"] is True
    assert remove_calls == [["docker", "rm", "-f", session.container_name]]
    assert session._container_cleanup_complete is True


def test_docker_session_gate_release_uses_fifo_and_is_idempotent(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, scheduler_attach_timeout_sec=0.2)
    gate_name = "scheduler.start.gate"
    gate = tmp_path / gate_name
    os.mkfifo(gate)
    payloads: list[bytes] = []
    reader_started = threading.Event()

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    def read_gate() -> None:
        reader_started.set()
        with gate.open("rb", buffering=0) as stream:
            payloads.append(stream.readline())

    session = bridge.CoordinatedDockerSession(config)
    session.process = FakeProcess()  # type: ignore[assignment]
    reader = threading.Thread(target=read_gate, daemon=True)
    reader.start()
    assert reader_started.wait(timeout=1)

    session._release_gate_once(gate_name)
    reader.join(timeout=1)

    assert not reader.is_alive()
    assert payloads == [b"continue\n"]
    assert session._released_gates == {gate_name}

    # The sole FIFO reader has exited. A second call must return from the
    # session's one-shot state rather than trying to open the FIFO again.
    started = time.monotonic()
    session._release_gate_once(gate_name)
    assert time.monotonic() - started < 0.1
    assert payloads == [b"continue\n"]


def test_docker_session_abort_is_bounded_after_fifo_reader_dies(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, scheduler_attach_timeout_sec=0.03)
    gate = tmp_path / "sandbox.start.gate"
    os.mkfifo(gate)
    # Model a reader that existed but exited after publishing readiness.
    reader_fd = os.open(gate, os.O_RDONLY | os.O_NONBLOCK)
    os.close(reader_fd)
    (tmp_path / "sandbox_attach_ready").write_text("ready\n")

    class FakeProcess:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return "", ""

    session = bridge.CoordinatedDockerSession(config)
    session.process = FakeProcess()  # type: ignore[assignment]

    started = time.monotonic()
    report = session.abort("reader died")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert report["aborted"] is True
    assert session.process.returncode == -15
    assert "sandbox.start.gate" not in session._released_gates


def test_docker_session_gate_release_fails_fast_after_container_exit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, scheduler_attach_timeout_sec=1.0)

    class ExitedProcess:
        returncode = 17

        def poll(self) -> int:
            return self.returncode

    session = bridge.CoordinatedDockerSession(config)
    session.process = ExitedProcess()  # type: ignore[assignment]

    started = time.monotonic()
    with pytest.raises(container_module.ProfileError, match="container exited"):
        session._release_gate_once("scheduler.start.gate")

    assert time.monotonic() - started < 0.1
    assert "scheduler.start.gate" not in session._released_gates

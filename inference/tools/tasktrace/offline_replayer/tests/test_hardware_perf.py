# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest
from offline_replayer.measurement.hardware_perf import (
    HardwarePerfError,
    PerfCgroupHardwareProfiler,
    PerfControlBounds,
    collect_hardware_perf_provenance,
    parse_perf_stat_csv,
)


def test_parse_perf_stat_csv_preserves_counts_and_runtime_validity() -> None:
    counters, anomalies = parse_perf_stat_csv(
        "1000;;cycles;2000000;100.00;;\n" "500;;instructions;2000000;96.00;;\n" "25;;branches;2000000;90.00;;\n",
        requested_events=("cycles", "instructions", "branches"),
    )

    assert anomalies == []
    assert counters[0].to_dict() == {
        "event": "cycles",
        "namespace": "portable",
        "status": "valid",
        "value": 1000,
        "unit": None,
        "time_enabled_ns": 2_000_000,
        "time_running_ns": 2_000_000,
        "time_running_fraction": 1.0,
        "valid": True,
        "invalid_reason": None,
        "observed_cgroup": None,
    }
    assert counters[1].valid is True
    assert counters[1].time_running_ns == 1_920_000
    assert counters[2].status == "multiplexed"
    assert counters[2].valid is False
    assert "below" in (counters[2].invalid_reason or "")


def test_parse_perf_stat_csv_marks_unsupported_not_counted_and_missing() -> None:
    counters, anomalies = parse_perf_stat_csv(
        "<not supported>;;cycles;0;100.00;;\n" "<not counted>;;instructions;0;100.00;;\n",
        requested_events=("cycles", "instructions", "branches"),
    )

    assert anomalies == []
    assert [counter.status for counter in counters] == [
        "unsupported",
        "not_counted",
        "missing",
    ]
    assert all(counter.value is None and not counter.valid for counter in counters)


def test_parse_perf_stat_csv_fails_closed_on_malformed_and_duplicate_rows() -> None:
    counters, anomalies = parse_perf_stat_csv(
        "bad;;cycles;oops;100.00;;\n"
        "12;;instructions;1000;100.00;;\n"
        "13;;instructions;1000;100.00;;\n"
        "too;short\n",
        requested_events=("cycles", "instructions"),
    )

    assert counters[0].status == "malformed"
    assert counters[1].status == "duplicate"
    assert counters[1].valid is False
    assert counters[1].value == 12
    assert any("malformed values" in anomaly for anomaly in anomalies)
    assert any("duplicate" in anomaly for anomaly in anomalies)
    assert any("fewer than 5" in anomaly for anomaly in anomalies)


def test_parse_perf_stat_csv_understands_and_verifies_cgroup_layout() -> None:
    cgroup = "/system.slice/docker-example.scope"
    counters, anomalies = parse_perf_stat_csv(
        "1200;;cycles;/system.slice/docker-example.scope;2000000;99.50;;\n"
        "<not supported>;;instructions;/system.slice/docker-example.scope;0;100.00;;\n",
        requested_events=("cycles", "instructions"),
        expected_cgroup=cgroup,
    )

    assert anomalies == []
    assert counters[0].valid is True
    assert counters[0].value == 1200
    assert counters[0].time_enabled_ns == 2_000_000
    assert counters[0].time_running_ns == 1_990_000
    assert counters[0].observed_cgroup == cgroup
    assert counters[1].status == "unsupported"
    assert counters[1].observed_cgroup == cgroup


def test_parse_perf_stat_csv_fails_closed_on_wrong_or_missing_cgroup() -> None:
    wrong, wrong_anomalies = parse_perf_stat_csv(
        "1200;;cycles;/system.slice/wrong.scope;2000000;100.00;;\n",
        requested_events=("cycles",),
        expected_cgroup="/system.slice/expected.scope",
    )
    missing, missing_anomalies = parse_perf_stat_csv(
        "1200;;cycles;2000000;100.00;;\n",
        requested_events=("cycles",),
        expected_cgroup="/system.slice/expected.scope",
    )

    assert wrong[0].status == "scope_mismatch"
    assert wrong[0].valid is False
    assert wrong[0].observed_cgroup == "/system.slice/wrong.scope"
    assert any("expected" in anomaly for anomaly in wrong_anomalies)
    assert missing[0].status == "scope_mismatch"
    assert missing[0].valid is False
    assert any("expected" in anomaly for anomaly in missing_anomalies)


def test_perf_counter_results_are_deterministically_json_serializable() -> None:
    counters, _anomalies = parse_perf_stat_csv(
        "1.5;msec;task-clock;1000;100.00;;\n",
        requested_events=("task-clock",),
    )

    first = json.dumps([counter.to_dict() for counter in counters], sort_keys=True)
    second = json.dumps([counter.to_dict() for counter in counters], sort_keys=True)

    assert first == second
    assert json.loads(first)[0]["value"] == 1.5
    assert json.loads(first)[0]["namespace"] == "custom-unclassified"


class _ExitedPerf:
    returncode = 0

    def poll(self) -> int:
        return 0

    def send_signal(self, _signal: signal.Signals) -> None:
        raise AssertionError("an exited perf process must not be signaled")

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_finish_rejects_anomalous_output_even_when_each_counter_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup = "/system.slice/docker-example.scope"
    collector = PerfCgroupHardwareProfiler(
        output_dir=tmp_path,
        cpuset_cpus="2-3",
        events=("cycles", "instructions"),
    )
    collector._process = _ExitedPerf()  # type: ignore[assignment]
    collector._perf_cgroup = cgroup
    collector._attach_ready_bounds = PerfControlBounds("disable", 10, 20)
    collector._enable_bounds = PerfControlBounds("enable", 100, 110)
    collector._enabled = True
    collector._provenance = {
        "perf_binary": "test-perf",
        "perf_version": "perf version test",
    }
    collector._stats_path.write_text(
        f"100;;cycles;{cgroup};1000;100.00;;\n"
        f"101;;cycles;{cgroup};1000;100.00;;\n"
        f"200;;instructions;{cgroup};1000;100.00;;\n"
    )
    monkeypatch.setattr(
        collector,
        "_control",
        lambda action: PerfControlBounds(action, 190, 200),
    )

    report = collector.finish(window_start_ns=100, window_end_ns=200)

    assert report["instrumentation_valid"] is False
    assert report["all_requested_metrics_valid"] is False
    assert report["hybrid_pmu_counts_merged"] is True
    assert report["anomalies"] == ["perf stat emitted duplicate event 'cycles'"]
    assert report["window_start_ns"] == 100
    assert report["window_end_ns"] == 200
    assert report["requested_lifecycle_window_start_ns"] == 100
    assert report["requested_lifecycle_window_end_ns"] == 200
    assert report["control_bounds"]["enable"]["uncertainty_ns"] == 10
    assert report["control_bounds"]["disable"]["uncertainty_ns"] == 10
    assert report["provenance"]["perf_version"] == "perf version test"


def test_start_requests_hybrid_pmu_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> _ExitedPerf:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _ExitedPerf()

    collector = PerfCgroupHardwareProfiler(
        output_dir=tmp_path,
        cpuset_cpus="2-3",
        events=("cycles",),
        provenance_provider=lambda: {
            "perf_binary": "perf",
            "perf_version": "perf version injected",
        },
    )
    actions: list[str] = []

    def fake_control(action: str) -> PerfControlBounds:
        actions.append(action)
        start = 100 * len(actions)
        return PerfControlBounds(action, start, start + 5)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(collector, "_control", fake_control)

    collector.start(
        cgroup_path=Path("/sys/fs/cgroup/system.slice/example.scope"),
        enable=False,
    )
    assert actions == ["disable"]
    assert collector._enabled is False
    enable_bounds = collector.enable()
    collector.close()

    assert "--hybrid-merge" in captured["command"]
    assert actions == ["disable", "enable"]
    assert enable_bounds.action == "enable"
    assert enable_bounds.uncertainty_ns == 5
    assert collector.stats_path == tmp_path / "hardware_perf_stat.csv"
    assert collector.stderr_path == tmp_path / "hardware_perf.stderr.txt"
    assert collector.provenance["perf_version"] == "perf version injected"


def test_collect_hardware_perf_provenance_is_compact_and_injectable(
    tmp_path: Path,
) -> None:
    event_sources = tmp_path / "event-sources"
    event_sources.mkdir()
    for name in ("tracepoint", "cpu_core", "software"):
        (event_sources / name).mkdir()
    (event_sources / "not-a-pmu-file").write_text("ignored")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="perf version 9.9\n", stderr="")

    provenance = collect_hardware_perf_provenance(
        perf="/opt/perf",
        event_source_root=event_sources,
        command_runner=runner,
        kernel_release="6.test",
        machine="aarch64",
    )

    assert provenance == {
        "perf_binary": "/opt/perf",
        "perf_version": "perf version 9.9",
        "kernel_release": "6.test",
        "machine": "aarch64",
        "event_source_root": str(event_sources),
        "event_source_names": ["cpu_core", "software", "tracepoint"],
        "provenance_valid": True,
        "provenance_errors": [],
    }
    assert calls == [
        (
            ["/opt/perf", "--version"],
            {
                "text": True,
                "capture_output": True,
                "timeout": 5,
                "check": False,
            },
        )
    ]


def test_collect_hardware_perf_provenance_retains_discovery_failures(
    tmp_path: Path,
) -> None:
    def unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("perf unavailable")

    provenance = collect_hardware_perf_provenance(
        perf="missing-perf",
        event_source_root=tmp_path / "missing",
        command_runner=unavailable,
        kernel_release="6.test",
        machine="x86_64",
    )

    assert provenance["perf_version"] is None
    assert provenance["event_source_names"] == []
    assert provenance["provenance_valid"] is False
    assert len(provenance["provenance_errors"]) == 2


class _RunningPerf:
    def poll(self) -> None:
        return None


def test_control_acknowledgement_returns_monotonic_acquisition_bounds(
    tmp_path: Path,
) -> None:
    times = iter((1_000, 1_075))
    collector = PerfCgroupHardwareProfiler(
        output_dir=tmp_path,
        cpuset_cpus="2",
        events=("cycles",),
        clock_ns=lambda: next(times),
    )
    control_read_fd, control_write_fd = os.pipe()
    ack_read_fd, ack_write_fd = os.pipe()
    collector._process = _RunningPerf()  # type: ignore[assignment]
    collector._control_write_fd = control_write_fd
    collector._ack_read_fd = ack_read_fd
    os.write(ack_write_fd, b"ack\n")
    try:
        bounds = collector._control("enable")
        assert os.read(control_read_fd, 64) == b"enable\n"
    finally:
        collector._close_fds()
        os.close(control_read_fd)
        os.close(ack_write_fd)

    assert bounds.to_dict() == {
        "action": "enable",
        "request_monotonic_ns": 1_000,
        "acknowledgement_monotonic_ns": 1_075,
        "uncertainty_ns": 75,
        "clock": "CLOCK_MONOTONIC",
    }


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1])
def test_parse_perf_stat_csv_rejects_invalid_minimum_ratio(ratio: float) -> None:
    with pytest.raises(HardwarePerfError, match="running ratio"):
        parse_perf_stat_csv("", minimum_running_ratio=ratio)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess

import pytest
from offline_replayer.measurement.scheduler_bcc import (
    SchedulerTraceError,
    summarize_scheduler_events,
)
from offline_replayer.measurement.scheduler_perf import (
    PerfSchedulerConcurrencyTracer,
    parse_perf_script_switch_events,
)


def test_native_perf_cpu_wide_switch_records_measure_concurrency() -> None:
    events, lost, anomalies = parse_perf_script_switch_events("""
 root 42/42 [000] 1.100000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0
child 43/43 [001] 1.150000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0
   :-1 -1/-1 [001] 1.250000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0
 root 42/42 [000] 1.300000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0
""")

    report = summarize_scheduler_events(
        events,
        window_start_ns=1_000_000_000,
        window_end_ns=1_400_000_000,
        method="perf-record-cgroup-native-switch-events",
    )

    assert lost == 0
    assert anomalies == []
    assert report["method"] == "perf-record-cgroup-native-switch-events"
    assert report["scheduled_runtime_sec"] == 0.3
    assert report["peak_simultaneous_logical_cores"] == 2
    assert report["time_sec_by_simultaneous_logical_cores"] == {
        "0": 0.2,
        "1": 0.1,
        "2": 0.1,
    }


def test_native_perf_parser_counts_reported_lost_events() -> None:
    _events, lost, _anomalies = parse_perf_script_switch_events(
        " perf 1/1 [000] 1.000000000: PERF_RECORD_LOST lost: 17\n"
    )

    assert lost == 17


def test_native_perf_parser_surfaces_duplicate_running_cpu() -> None:
    _events, _lost, anomalies = parse_perf_script_switch_events("""
 root 42/42 [000] 1.100000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0
child 43/43 [000] 1.200000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 42/42
""")

    assert "while recorded TID 42 was active" in anomalies[0]


def test_perf_control_accepts_nul_prefixed_ack(tmp_path) -> None:
    class RunningRecorder:
        def poll(self) -> None:
            return None

    tracer = PerfSchedulerConcurrencyTracer(output_dir=tmp_path)
    control_read_fd, tracer._control_write_fd = os.pipe()
    tracer._ack_read_fd, ack_write_fd = os.pipe()
    tracer._recorder = RunningRecorder()  # type: ignore[assignment]
    os.write(ack_write_fd, b"\x00ack\n")
    try:
        tracer._control("enable")
        assert os.read(control_read_fd, 64) == b"enable\n"
    finally:
        os.close(control_read_fd)
        os.close(ack_write_fd)
        tracer._close_control_fds()


def test_empty_perf_capture_is_not_reported_as_zero_concurrency(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = PerfSchedulerConcurrencyTracer(output_dir=tmp_path)
    tracer._data_path.write_bytes(b"perf data")

    def decode_without_switches(_command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output = kwargs["stdout"]
        output.write("bash 0/0 [000] 0.000000000: PERF_RECORD_COMM: bash:42/42\n")
        return subprocess.CompletedProcess(_command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", decode_without_switches)

    with pytest.raises(SchedulerTraceError, match="no context-switch transitions"):
        tracer._decode_records()

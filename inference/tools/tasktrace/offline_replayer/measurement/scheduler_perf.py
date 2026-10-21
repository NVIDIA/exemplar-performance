#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-concurrency collection from native ``perf record`` switch events.

This backend records native context-switch events only while tasks in the
gated container cgroup run. Unlike task inheritance, cgroup-scoped CPU-wide
recording observes rapidly created descendants immediately. It also requests
``CLOCK_MONOTONIC`` so event timestamps share the profiler controller's
measurement clock.
"""

from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .scheduler_bcc import (
    SchedulerEvent,
    SchedulerTraceError,
    summarize_scheduler_events,
)

PERF_RECORD_LINE = re.compile(
    r"^\s*.*?\s+(?P<pid>-?\d+)/(?P<tid>-?\d+)\s+"
    r"\[(?P<cpu>\d+)\]\s+(?P<timestamp>[0-9]+\.[0-9]+):\s+"
    r"PERF_RECORD_(?P<event>SWITCH(?:_CPU_WIDE)?|EXIT)\b(?P<details>.*)$"
)
PERF_SWITCH_DIRECTION = re.compile(r"^\s+(?P<direction>IN|OUT)\b")
LOST_COUNT = re.compile(r"\blost(?:\s+events?)?\s*[:=]?\s*(?P<count>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class NativePerfRecord:
    timestamp_ns: int
    cpu: int
    tid: int
    event: str
    entering: bool | None = None


def _timestamp_ns(timestamp_sec: str) -> int:
    try:
        return int(Decimal(timestamp_sec) * Decimal(1_000_000_000))
    except (InvalidOperation, ValueError) as exc:
        raise SchedulerTraceError(f"invalid perf script timestamp: {timestamp_sec!r}") from exc


def _parse_native_records(text: str) -> tuple[list[NativePerfRecord], int]:
    records: list[NativePerfRecord] = []
    lost_events = 0
    for line in text.splitlines():
        if "PERF_RECORD_LOST" in line:
            count = LOST_COUNT.search(line)
            lost_events += int(count.group("count")) if count is not None else 1
        match = PERF_RECORD_LINE.match(line)
        if match is None:
            continue
        event = match.group("event")
        entering: bool | None = None
        if event.startswith("SWITCH"):
            direction = PERF_SWITCH_DIRECTION.match(match.group("details"))
            if direction is None:
                raise SchedulerTraceError(f"unable to decode perf switch direction: {line!r}")
            entering = direction.group("direction") == "IN"
        records.append(
            NativePerfRecord(
                timestamp_ns=_timestamp_ns(match.group("timestamp")),
                cpu=int(match.group("cpu")),
                tid=int(match.group("tid")),
                event=event,
                entering=entering,
            )
        )
    return records, lost_events


def parse_perf_script_switch_events(
    text: str,
) -> tuple[list[SchedulerEvent], int, list[str]]:
    """Convert cgroup-scoped native switch records into CPU occupancy transitions."""
    records, lost_events = _parse_native_records(text)
    active_tid_by_cpu: dict[int, int] = {}
    active_cpu_by_tid: dict[int, int] = {}
    events: list[SchedulerEvent] = []
    anomalies: list[str] = []
    for record in sorted(records, key=lambda item: (item.timestamp_ns, item.cpu)):
        if record.event.startswith("SWITCH") and record.entering:
            active_tid = active_tid_by_cpu.get(record.cpu)
            if active_tid is not None:
                anomalies.append(
                    f"logical CPU {record.cpu} switched into TID {record.tid} "
                    f"while recorded TID {active_tid} was active"
                )
            previous_cpu = active_cpu_by_tid.get(record.tid)
            if previous_cpu is not None:
                anomalies.append(
                    f"TID {record.tid} switched into logical CPU {record.cpu} "
                    f"while active on logical CPU {previous_cpu}"
                )
            if active_tid is None and previous_cpu is None:
                active_tid_by_cpu[record.cpu] = record.tid
                active_cpu_by_tid[record.tid] = record.cpu
            events.append(SchedulerEvent(record.timestamp_ns, record.cpu, record.tid, True))
            continue
        if record.event.startswith("SWITCH"):
            active_tid = active_tid_by_cpu.get(record.cpu)
            leaving_tid = record.tid
            if record.tid == -1 and active_tid is not None:
                leaving_tid = active_tid
            elif active_tid != record.tid:
                anomalies.append(
                    f"logical CPU {record.cpu} switched out TID {record.tid} "
                    f"while recorded active TID was {active_tid}"
                )
            if active_tid == leaving_tid:
                active_tid_by_cpu.pop(record.cpu, None)
                active_cpu_by_tid.pop(leaving_tid, None)
            events.append(SchedulerEvent(record.timestamp_ns, record.cpu, leaving_tid, False))
            continue
        # CPU-wide collection emits the definitive SWITCH OUT after task exit.
        # EXIT is retained in decoded artifacts for diagnosis, not occupancy.
        continue
    return events, lost_events, anomalies


class PerfSchedulerConcurrencyTracer:
    """Collect native perf context switches for one gated container cgroup."""

    def __init__(
        self,
        *,
        output_dir: Path,
        cpuset_cpus: str | None = None,
        perf: str = "perf",
        attach_timeout_sec: float = 20.0,
        mmap_pages: int | None = None,
    ) -> None:
        if attach_timeout_sec <= 0:
            raise SchedulerTraceError("perf scheduler attach timeout must be positive")
        if mmap_pages is not None and mmap_pages <= 0:
            raise SchedulerTraceError("perf scheduler mmap page count must be positive")
        self.output_dir = output_dir
        self.cpuset_cpus = cpuset_cpus
        self.perf = perf
        self.attach_timeout_sec = attach_timeout_sec
        self.mmap_pages = mmap_pages
        self.root_tid: int | None = None
        self.cgroup_path: Path | None = None
        self.perf_cgroup: str | None = None
        self._recorder: subprocess.Popen[bytes] | None = None
        self._control_write_fd: int | None = None
        self._ack_read_fd: int | None = None
        self._ack_buffer = b""
        self._data_path = self.output_dir / "scheduler_perf.data"
        self._decoded_path = self.output_dir / "scheduler_perf_switch_events.txt"
        self._stderr_path = self.output_dir / "scheduler_perf.stderr.txt"
        self._events: list[SchedulerEvent] = []
        self._lost_events = 0
        self._decoder_anomalies: list[str] = []

    def _stderr(self) -> str:
        if not self._stderr_path.is_file():
            return ""
        return self._stderr_path.read_text(errors="replace").strip()

    def _close_control_fds(self) -> None:
        for fd in (self._control_write_fd, self._ack_read_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._control_write_fd = None
        self._ack_read_fd = None

    def _control(self, action: str) -> None:
        if self._recorder is None or self._control_write_fd is None or self._ack_read_fd is None:
            raise SchedulerTraceError("perf scheduler tracer has not been started")
        if self._recorder.poll() is not None:
            raise SchedulerTraceError(f"perf scheduler recorder exited before {action}: {self._stderr()}")
        try:
            os.write(self._control_write_fd, f"{action}\n".encode())
        except OSError as exc:
            raise SchedulerTraceError(f"unable to send perf scheduler {action} command: {exc}") from exc
        deadline = time.monotonic() + self.attach_timeout_sec
        while True:
            if b"\n" in self._ack_buffer:
                response, self._ack_buffer = self._ack_buffer.split(b"\n", 1)
                if response.strip(b"\x00 \t\r") != b"ack":
                    raise SchedulerTraceError(f"unexpected perf scheduler control response: {response!r}")
                return
            if self._recorder.poll() is not None:
                raise SchedulerTraceError(
                    f"perf scheduler recorder exited before {action} acknowledgement: " f"{self._stderr()}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SchedulerTraceError(f"timed out waiting for perf scheduler {action} acknowledgement")
            readable, _writable, _exceptional = select.select([self._ack_read_fd], [], [], min(remaining, 0.1))
            if readable:
                chunk = os.read(self._ack_read_fd, 4096)
                if not chunk:
                    raise SchedulerTraceError(f"perf scheduler control pipe closed before {action} acknowledgement")
                self._ack_buffer += chunk

    def start(self, root_tid: int, *, cgroup_path: Path) -> None:
        self.root_tid = root_tid
        self.cgroup_path = cgroup_path
        try:
            relative_cgroup = cgroup_path.relative_to("/sys/fs/cgroup")
        except ValueError as exc:
            raise SchedulerTraceError(f"perf scheduler cgroup is outside /sys/fs/cgroup: {cgroup_path}") from exc
        self.perf_cgroup = "/" + relative_cgroup.as_posix()
        control_read_fd, control_write_fd = os.pipe()
        ack_read_fd, ack_write_fd = os.pipe()
        command = [
            self.perf,
            "record",
            "--switch-events",
            "--sample-cpu",
            "--clockid",
            "mono",
            "-T",
            "-B",
            "-N",
            "-D",
            "-1",
            "--control",
            f"fd:{control_read_fd},{ack_write_fd}",
            "--synth",
            "no",
        ]
        if self.mmap_pages is not None:
            command.extend(["-m", str(self.mmap_pages)])
        command.extend(
            [
                "-o",
                str(self._data_path),
                "-e",
                "dummy",
                "-G",
                self.perf_cgroup,
                "-a",
            ]
        )
        if self.cpuset_cpus is not None:
            command.extend(["-C", self.cpuset_cpus])
        try:
            stderr = self._stderr_path.open("wb")
            self._recorder = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                pass_fds=(control_read_fd, ack_write_fd),
            )
            stderr.close()
        except OSError as exc:
            os.close(control_read_fd)
            os.close(control_write_fd)
            os.close(ack_read_fd)
            os.close(ack_write_fd)
            raise SchedulerTraceError(f"unable to start perf scheduler recorder: {exc}") from exc
        os.close(control_read_fd)
        os.close(ack_write_fd)
        self._control_write_fd = control_write_fd
        self._ack_read_fd = ack_read_fd
        try:
            self._control("enable")
        except SchedulerTraceError:
            self.close()
            raise

    def poll(self, timeout_ms: int = 100) -> None:
        if self._recorder is None:
            raise SchedulerTraceError("perf scheduler tracer has not been started")
        if self._recorder.poll() is not None:
            raise SchedulerTraceError(
                "perf scheduler recorder exited before measurement completed: " f"{self._stderr()}"
            )
        if timeout_ms > 0:
            time.sleep(timeout_ms / 1000)

    def timestamp_ns(self) -> int:
        """Return the CLOCK_MONOTONIC measurement clock requested from perf."""
        return time.monotonic_ns()

    def _stop_recorder(self) -> None:
        assert self._recorder is not None
        if self._recorder.poll() is None:
            self._recorder.send_signal(signal.SIGINT)
        try:
            self._recorder.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            self._recorder.kill()
            self._recorder.wait()
            raise SchedulerTraceError("timed out stopping perf scheduler recorder") from exc
        if self._recorder.returncode not in (0, -signal.SIGINT):
            raise SchedulerTraceError(f"perf scheduler recorder failed while stopping: {self._stderr()}")

    def _decode_records(self) -> None:
        if not self._data_path.is_file():
            raise SchedulerTraceError(f"perf scheduler recorder produced no data file: {self._data_path}")
        command = [
            self.perf,
            "script",
            "--ns",
            "-i",
            str(self._data_path),
            "--show-switch-events",
            "--show-task-events",
            "--show-lost-events",
            "-F",
            "comm,pid,tid,time,cpu,event",
        ]
        with self._decoded_path.open("w") as output:
            decoded = subprocess.run(command, text=True, stdout=output, stderr=subprocess.PIPE)
        if decoded.returncode != 0:
            raise SchedulerTraceError(f"unable to decode perf scheduler records: {decoded.stderr.strip()}")
        self._events, self._lost_events, self._decoder_anomalies = parse_perf_script_switch_events(
            self._decoded_path.read_text(errors="replace")
        )
        if not self._events:
            raise SchedulerTraceError(
                "perf scheduler record produced no context-switch transitions; "
                "do not interpret this capture as zero concurrency"
            )

    def finish(self, *, window_start_ns: int, window_end_ns: int) -> dict[str, Any]:
        if self.root_tid is None or self._recorder is None:
            raise SchedulerTraceError("perf scheduler tracer has not been started")
        self._control("disable")
        self._stop_recorder()
        self._close_control_fds()
        self._decode_records()
        report = summarize_scheduler_events(
            self._events,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
            lost_events=self._lost_events,
            method="perf-record-cgroup-native-switch-events",
            scope=("native perf CPU-wide context-switch records restricted to the " "measured container cgroup"),
        )
        if self._decoder_anomalies:
            report["anomalies"] = self._decoder_anomalies + report["anomalies"]
            report["trace_complete"] = False
        report["root_tid"] = self.root_tid
        report["backend"] = "perf"
        report["perf_clockid"] = "mono"
        report["perf_cgroup"] = self.perf_cgroup
        report["perf_cpuset_cpus"] = self.cpuset_cpus
        report["perf_mmap_pages"] = self.mmap_pages
        report["perf_data_path"] = str(self._data_path)
        report["perf_decoded_switch_events_path"] = str(self._decoded_path)
        report["perf_stderr_path"] = str(self._stderr_path)
        return report

    def write_events(self, path: Path) -> None:
        with path.open("w") as output:
            for event in sorted(self._events, key=lambda item: item.timestamp_ns):
                output.write(
                    json.dumps(
                        {
                            "timestamp_ns": event.timestamp_ns,
                            "cpu": event.cpu,
                            "tid": event.tid,
                            "entering": event.entering,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    def close(self) -> None:
        if self._recorder is not None and self._recorder.poll() is None:
            self._recorder.terminate()
            try:
                self._recorder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._recorder.kill()
                self._recorder.wait()
        self._close_control_fds()
        self._recorder = None

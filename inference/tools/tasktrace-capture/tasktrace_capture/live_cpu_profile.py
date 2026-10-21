# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker cgroup CPU profiling for live TaskTrace Capture trials."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .scheduler_concurrency import (
    SchedulerTraceError,
    add_cgroup_validation,
    parse_perf_script_switch_events,
    summarize_scheduler_events,
)

LIVE_CPU_ENABLE_ENV = "_TASKTRACE_CAPTURE_PROFILE_LIVE_CPU"
LIVE_CPU_WINDOWS_ENV = "_TASKTRACE_CAPTURE_PROFILE_WINDOWS"
LIVE_CPU_SCHEDULER_ENV = "_TASKTRACE_CAPTURE_PROFILE_SCHEDULER"
LIVE_CPU_COLLECTOR_IMAGE_ENV = "_TASKTRACE_CAPTURE_PROFILE_COLLECTOR_IMAGE"
LIVE_CPU_ANCHOR_IMAGE_ENV = "_TASKTRACE_CAPTURE_PROFILE_ANCHOR_IMAGE"
LIVE_CPU_CPUSET_ENV = "_TASKTRACE_CAPTURE_PROFILE_CPUSET_CPUS"

WINDOW_AGENT_EXECUTION = "agent_execution"
WINDOW_LIFECYCLE = "container_lifecycle_to_verified"
SUPPORTED_WINDOWS = frozenset({WINDOW_AGENT_EXECUTION, WINDOW_LIFECYCLE})
PROFILE_FILENAME = "tasktrace_capture_live_cpu_profile.json"
# Browser-heavy verifiers can emit more than one million switch records. Keep
# enough perf ring-buffer headroom to preserve validated lifecycle captures.
SCHEDULER_PERF_MMAP_PAGES = 1024


class LiveCpuProfileError(RuntimeError):
    """Raised when requested live cgroup profiling cannot be completed."""


def live_cpu_enabled() -> bool:
    return os.environ.get(LIVE_CPU_ENABLE_ENV) == "1"


def requested_windows() -> set[str]:
    raw = os.environ.get(LIVE_CPU_WINDOWS_ENV, WINDOW_AGENT_EXECUTION)
    windows = {item.strip() for item in raw.split(",") if item.strip()}
    invalid = windows - SUPPORTED_WINDOWS
    if invalid:
        raise LiveCpuProfileError("unsupported live CPU profile window(s): " + ", ".join(sorted(invalid)))
    return windows or {WINDOW_AGENT_EXECUTION}


def scheduler_required() -> bool:
    mode = os.environ.get(LIVE_CPU_SCHEDULER_ENV, "off")
    if mode not in {"off", "required"}:
        raise LiveCpuProfileError(f"unsupported live scheduler mode: {mode!r}")
    return mode == "required"


def parse_cpuset_cpu_ids(cpuset: str) -> tuple[int, ...]:
    """Return the distinct logical CPU IDs from Docker cpuset syntax."""
    ids: set[int] = set()
    for component in cpuset.split(","):
        component = component.strip()
        if not component:
            raise LiveCpuProfileError(f"invalid empty CPU set component in {cpuset!r}")
        if "-" in component:
            bounds = component.split("-", 1)
            if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
                raise LiveCpuProfileError(f"invalid CPU set component: {component!r}")
            first, last = (int(bound) for bound in bounds)
            if first > last:
                raise LiveCpuProfileError(f"invalid CPU set range: {component!r}")
            ids.update(range(first, last + 1))
        elif component.isdigit():
            ids.add(int(component))
        else:
            raise LiveCpuProfileError(f"invalid CPU set component: {component!r}")
    return tuple(sorted(ids))


def validate_profile_cpuset_cpus(
    *,
    profile_scheduler: str,
    requested_cpuset: str | None,
    override_cpus: int | None,
) -> str | None:
    """Validate user-selected placement for exact scheduler profiling."""
    if requested_cpuset is not None:
        cpu_ids = parse_cpuset_cpu_ids(requested_cpuset)
        if profile_scheduler == "required" and override_cpus is not None:
            if len(cpu_ids) != override_cpus:
                raise LiveCpuProfileError(
                    "--profile-cpuset-cpus width must match --override-cpus " "for controlled scheduler profiling"
                )
        return requested_cpuset
    if profile_scheduler == "required":
        raise LiveCpuProfileError(
            "--profile-scheduler required needs --profile-cpuset-cpus CPUSET "
            "so you explicitly choose the fixed logical CPUs to measure; "
            "for example, --override-cpus 8 --profile-cpuset-cpus 8-15. "
            "For average CPU without fixed placement, omit --profile-scheduler required"
        )
    return None


def requested_cpuset_cpus() -> str | None:
    cpuset = os.environ.get(LIVE_CPU_CPUSET_ENV)
    if cpuset is None:
        return None
    cpuset = cpuset.strip()
    if not cpuset:
        raise LiveCpuProfileError("live CPU profile cpuset must not be empty")
    parse_cpuset_cpu_ids(cpuset)
    return cpuset


def _read_cpu_stat(path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    for line in (path / "cpu.stat").read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            stats[fields[0]] = int(fields[1])
    return stats


def _cpu_window(
    name: str,
    *,
    before: dict[str, int],
    after: dict[str, int],
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    if end_ns <= start_ns:
        raise LiveCpuProfileError(f"invalid live profile window: {name}")
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(before) | set(after))}
    wall_sec = (end_ns - start_ns) / 1_000_000_000
    usage_sec = delta.get("usage_usec", 0) / 1_000_000
    return {
        "name": name,
        "start_monotonic_ns": start_ns,
        "end_monotonic_ns": end_ns,
        "wall_sec": round(wall_sec, 9),
        "cpu_stat_before": before,
        "cpu_stat_after": after,
        "cpu_stat_delta": delta,
        "usage_sec": round(usage_sec, 9),
        "user_sec": round(delta.get("user_usec", 0) / 1_000_000, 9),
        "system_sec": round(delta.get("system_usec", 0) / 1_000_000, 9),
        "average_logical_cores_used": round(usage_sec / wall_sec, 9),
    }


async def _run(*command: str, check: bool = True) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    out = stdout.decode(errors="replace")
    err = stderr.decode(errors="replace")
    if check and process.returncode != 0:
        raise LiveCpuProfileError(f"command failed ({process.returncode}): {' '.join(command)}: {err.strip()}")
    return process.returncode, out, err


class DockerLiveCpuProfiler:
    """Own one pre-created Docker slice and its live measurement artifacts."""

    def __init__(self, *, agent_dir: Path, session_id: str) -> None:
        token = hashlib.sha256(session_id.encode()).hexdigest()[:12]
        self.agent_dir = agent_dir
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.windows = requested_windows()
        self.require_scheduler = scheduler_required()
        self.cpuset_cpus = requested_cpuset_cpus()
        self.parent_name = f"tasktrace-capture-live-{token}.slice"
        self.anchor_name = f"tasktrace-capture-live-anchor-{token}"
        self.collector_name = f"tasktrace-capture-live-perf-{token}"
        self.anchor_image = os.environ.get(LIVE_CPU_ANCHOR_IMAGE_ENV, "ubuntu:22.04")
        self.collector_image = os.environ.get(LIVE_CPU_COLLECTOR_IMAGE_ENV, "")
        self.cgroup_path: Path | None = None
        self.compose_override_path = self.agent_dir / "tasktrace_capture_live_cpu_compose_override.json"
        self.profile_path = self.agent_dir / PROFILE_FILENAME
        self.perf_data_path = self.agent_dir / "tasktrace_capture_scheduler_perf.data"
        self._perf_capture_path = Path("/tmp") / f"tasktrace-capture-live-{token}-scheduler_perf.data"
        self.perf_decoded_path = self.agent_dir / "tasktrace_capture_scheduler_perf_switch_events.txt"
        self.perf_stderr_path = self.agent_dir / "tasktrace_capture_scheduler_perf.stderr.txt"
        self.events_path = self.agent_dir / "tasktrace_capture_scheduler_events.jsonl"
        self._pending: dict[str, tuple[dict[str, int], int]] = {}
        self._scheduler_started = False
        self._data: dict[str, Any] = {
            "schema_version": "tasktrace-capture-live-cpu-profile.v1",
            "runtime": "docker",
            "scope": {
                "kind": "main_container_parent_slice",
                "includes_compose_sidecars": False,
                "paused_anchor_excluded_by_measurement_boundaries": True,
                "cgroup_parent": self.parent_name,
                "cpuset_cpus": self.cpuset_cpus,
                "allowed_logical_cpu_count": (
                    len(parse_cpuset_cpu_ids(self.cpuset_cpus)) if self.cpuset_cpus is not None else None
                ),
                "lifecycle_start_boundary": (
                    "immediately before docker compose up, after image preparation " "and stale-container cleanup"
                ),
                "lifecycle_end_boundary": (
                    "entry to agent-environment stop, before log chown or container "
                    "teardown; includes verification only for shared-environment trials"
                ),
            },
            "windows": {},
            "scheduler_concurrency": {
                "requested": "required" if self.require_scheduler else "off",
                "backend": "perf" if self.require_scheduler else None,
                "windows": {},
            },
        }

    def _write(self) -> None:
        temporary = self.profile_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.profile_path)

    async def prepare_scope(self) -> Path:
        await _run("docker", "rm", "-f", self.anchor_name, check=False)
        await _run(
            "docker",
            "run",
            "-d",
            "--name",
            self.anchor_name,
            "--cgroup-parent",
            self.parent_name,
            self.anchor_image,
            "sleep",
            "infinity",
        )
        await _run("docker", "pause", self.anchor_name)
        _rc, stdout, _stderr = await _run("docker", "inspect", "--format", "{{.State.Pid}}", self.anchor_name)
        try:
            pid = int(stdout.strip())
        except ValueError as exc:
            raise LiveCpuProfileError("unable to resolve live profile anchor PID") from exc
        cgroup_rel = None
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                cgroup_rel = line.split("::", 1)[1].lstrip("/")
                break
        if cgroup_rel is None:
            raise LiveCpuProfileError("unable to resolve cgroup v2 path for live profile")
        self.cgroup_path = Path("/sys/fs/cgroup") / Path(cgroup_rel).parent
        if not (self.cgroup_path / "cpu.stat").is_file():
            raise LiveCpuProfileError(f"live profile cgroup has no readable cpu.stat: {self.cgroup_path}")
        main_override = {"cgroup_parent": self.parent_name}
        if self.cpuset_cpus is not None:
            main_override["cpuset"] = self.cpuset_cpus
        self.compose_override_path.write_text(json.dumps({"services": {"main": main_override}}, indent=2) + "\n")
        self._data["scope"]["cgroup_path"] = str(self.cgroup_path)
        self._data["scope"]["anchor_container"] = self.anchor_name
        self._write()
        return self.compose_override_path

    def begin_window(self, name: str) -> None:
        if name not in self.windows or name in self._pending:
            return
        assert self.cgroup_path is not None
        self._pending[name] = (_read_cpu_stat(self.cgroup_path), time.monotonic_ns())

    def finish_window(self, name: str) -> None:
        pending = self._pending.pop(name, None)
        if pending is None:
            return
        assert self.cgroup_path is not None
        before, start_ns = pending
        after = _read_cpu_stat(self.cgroup_path)
        end_ns = time.monotonic_ns()
        self._data["windows"][name] = _cpu_window(name, before=before, after=after, start_ns=start_ns, end_ns=end_ns)
        self._write()

    async def start_scheduler(self) -> None:
        if not self.require_scheduler or self._scheduler_started:
            return
        if not self.collector_image:
            raise LiveCpuProfileError(
                f"--profile-scheduler required but no collector image provided. "
                f"Set {LIVE_CPU_COLLECTOR_IMAGE_ENV} to a Docker image with perf/BCC tools."
            )
        assert self.cgroup_path is not None
        relative = "/" + self.cgroup_path.relative_to("/sys/fs/cgroup").as_posix()
        await _run("docker", "rm", "-f", self.collector_name, check=False)
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.collector_name,
            "--privileged",
            "--pid=host",
            "--cgroupns=host",
            "-v",
            "/:/host",
            "--entrypoint",
            "/usr/sbin/chroot",
            self.collector_image,
            "/host",
            "/usr/bin/perf",
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
            "--synth",
            "no",
            "-m",
            str(SCHEDULER_PERF_MMAP_PAGES),
            "-o",
            str(self._perf_capture_path),
            "-e",
            "dummy",
            "-G",
            relative,
            "-a",
        ]
        if self.cpuset_cpus is not None:
            command.extend(["-C", self.cpuset_cpus])
        _rc, _out, stderr = await _run(*command)
        if stderr.strip():
            self.perf_stderr_path.write_text(stderr)
        await asyncio.sleep(0.25)
        rc, running, logs = await _run(
            "docker",
            "inspect",
            "--format",
            "{{.State.Running}}",
            self.collector_name,
            check=False,
        )
        if rc != 0 or running.strip() != "true":
            _log_rc, collector_logs, _log_stderr = await _run("docker", "logs", self.collector_name, check=False)
            self.perf_stderr_path.write_text(collector_logs + logs)
            raise LiveCpuProfileError(
                "native perf scheduler collector failed to remain running: " + (collector_logs + logs).strip()
            )
        self._scheduler_started = True
        self._data["scheduler_concurrency"]["perf_cgroup"] = relative
        self._data["scheduler_concurrency"]["perf_cpuset_cpus"] = self.cpuset_cpus
        self._data["scheduler_concurrency"]["perf_mmap_pages"] = SCHEDULER_PERF_MMAP_PAGES
        self._data["scheduler_concurrency"]["perf_data_path"] = str(self.perf_data_path)
        self._write()

    async def finish_scheduler(self) -> None:
        if not self._scheduler_started:
            return
        await _run("docker", "kill", "--signal", "INT", self.collector_name, check=False)
        await _run("docker", "wait", self.collector_name, check=False)
        _rc, logs, _stderr = await _run("docker", "logs", self.collector_name, check=False)
        self.perf_stderr_path.write_text(logs)
        await _run(
            "docker",
            "run",
            "--rm",
            "--privileged",
            "-v",
            "/:/host",
            "--entrypoint",
            "/usr/sbin/chroot",
            self.collector_image,
            "/host",
            "/bin/chown",
            f"{os.getuid()}:{os.getgid()}",
            str(self._perf_capture_path),
        )
        shutil.copyfile(self._perf_capture_path, self.perf_data_path)
        rc, decoded, stderr = await _run(
            "/usr/bin/perf",
            "script",
            "--ns",
            "-i",
            str(self.perf_data_path),
            "--show-switch-events",
            "--show-task-events",
            "--show-lost-events",
            "-F",
            "comm,pid,tid,time,cpu,event",
            check=False,
        )
        if rc != 0:
            raise LiveCpuProfileError("unable to decode native perf scheduler output: " + stderr.strip())
        self.perf_decoded_path.write_text(decoded)
        try:
            events, lost_events, anomalies = parse_perf_script_switch_events(decoded)
        except SchedulerTraceError as exc:
            raise LiveCpuProfileError(str(exc)) from exc
        if not events:
            raise LiveCpuProfileError("native perf scheduler collection produced no switch transitions")
        with self.events_path.open("w") as output:
            for event in events:
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
        scheduler = self._data["scheduler_concurrency"]
        scheduler["decoded_switch_events_path"] = str(self.perf_decoded_path)
        scheduler["events_path"] = str(self.events_path)
        for name, cpu_report in self._data["windows"].items():
            report = summarize_scheduler_events(
                events,
                window_start_ns=cpu_report["start_monotonic_ns"],
                window_end_ns=cpu_report["end_monotonic_ns"],
                lost_events=lost_events,
                decoder_anomalies=anomalies,
            )
            scheduler["windows"][name] = add_cgroup_validation(report, cgroup_usage_sec=float(cpu_report["usage_sec"]))
        self._scheduler_started = False
        self._write()

    async def cleanup(self) -> None:
        await _run("docker", "rm", "-f", self.collector_name, check=False)
        await _run("docker", "unpause", self.anchor_name, check=False)
        await _run("docker", "rm", "-f", self.anchor_name, check=False)
        try:
            self._perf_capture_path.unlink()
        except FileNotFoundError:
            pass

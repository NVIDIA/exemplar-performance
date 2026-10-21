# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker runtime bridge for the reusable profiling coordinator.

This module is deliberately a bridge rather than another runner.  One
``CoordinatedDockerSession`` owns the disposable container and its gates;
small profiler adapters observe the same descendant-aware cgroup scope and
emit typed evidence.  The legacy container entry point remains the execution
harness, so selecting several profile sets never replays the workload twice.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from ..timebase import monotonic_clock_reference
from .cgroup_resources import (
    ANCESTOR_CPU_THROTTLE_KEYS,
    REQUIRED_ANCESTOR_CPU_STAT_KEYS,
    AncestorCpuObserver,
    CgroupResourceSampler,
    CgroupResourceSamplingError,
    parse_cpu_max,
    parse_flat_keyed,
)
from .coordinator import (
    CANONICAL_CLOCK_ID,
    DEFAULT_PROFILE_SETS,
    CoordinatorValidationError,
    ExecutionPlan,
    MountBehavior,
    ProfileCoordinator,
    ProfilerContext,
    ProfilerDescriptor,
    ProfileRequest,
    ProfilerRegistry,
    ProfilerReport,
    ProfileRunError,
    ResolvedProfiler,
)
from .evidence import (
    AcquisitionBounds,
    Cadence,
    ClockMapping,
    ClockReference,
    EvidenceRecord,
    EvidenceStream,
    LifecycleMarker,
    MeasuredScope,
    RawArtifact,
    bundle_sha256,
    write_bundle,
)
from .hardware_perf import PerfCgroupHardwareProfiler
from .process_exec import load_process_exec_trace, write_process_exec_report
from .scheduler_bcc import add_cgroup_validation

_VERSION = "container-coordinated-profiler.v1"
_CGROUP_CPU_SOURCE_SNAPSHOT_SCHEMA_VERSION = "container-cgroup-cpu-source-snapshots.v1"
_SECRET_IDENTITY_KEY = re.compile(
    r"(?:^|[_-])(secret|token|password|passwd|api[_-]?key|credential|authorization)(?:$|[_-])",
    re.IGNORECASE,
)


def _plain_json(value: Any) -> Any:
    """Thaw coordinator MappingProxyType/tuples for evidence JSON fields."""
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _secret_safe_identity(value: Any, *, key: str | None = None) -> Any:
    """Retain identity equality without archiving secret-looking values."""
    if key is not None and _SECRET_IDENTITY_KEY.search(key):
        encoded = json.dumps(
            _plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return {"redacted": True, "value_sha256": hashlib.sha256(encoded).hexdigest()}
    if isinstance(value, Mapping):
        return {str(item_key): _secret_safe_identity(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_secret_safe_identity(item) for item in value]
    return value


def _monotonic_offset_for_pid(pid: int) -> int | None:
    try:
        lines = Path(f"/proc/{pid}/timens_offsets").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) != 3 or fields[0] != "monotonic":
            continue
        try:
            seconds, nanoseconds = int(fields[1]), int(fields[2])
        except ValueError:
            return None
        if not 0 <= nanoseconds < 1_000_000_000:
            return None
        return seconds * 1_000_000_000 + nanoseconds
    return None


def _json_file(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _artifact(
    output_dir: Path,
    path: Path,
    *,
    artifact_id: str,
    media_type: str,
    provenance: Mapping[str, Any] | None = None,
) -> RawArtifact | None:
    if not path.is_file():
        return None
    return RawArtifact.from_path(
        artifact_id=artifact_id,
        path=path,
        base_dir=output_dir,
        media_type=media_type,
        provenance=dict(provenance or {}),
    )


def _artifacts(*values: RawArtifact | None) -> tuple[RawArtifact, ...]:
    return tuple(value for value in values if value is not None)


def _scalar_stream(
    context: ProfilerContext,
    *,
    profiler_id: str,
    stream_id: str,
    metric: str,
    unit: str,
    value: Any,
    artifact_ids: tuple[str, ...] = (),
    invalid_reason: str | None = None,
    missing_reason: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> EvidenceStream:
    if value is None or missing_reason is not None:
        return EvidenceStream(
            stream_id=stream_id,
            profiler_id=profiler_id,
            metric=metric,
            shape="scalar",
            unit=unit,
            status="missing",
            reasons=(missing_reason or f"{metric} was not observed",),
            raw_artifact_ids=artifact_ids,
        )
    status = "invalid" if invalid_reason else "valid"
    reasons = (invalid_reason,) if invalid_reason else ()
    record = EvidenceRecord(
        record_id=f"{stream_id}:value",
        run_id=context.run_id,
        scope_id=context.scope.scope_id,
        profiler_id=profiler_id,
        stream_id=stream_id,
        shape="scalar",
        status=status,
        reasons=reasons,
        value=value,
        raw_artifact_ids=artifact_ids,
        attributes=dict(attributes or {}),
    )
    return EvidenceStream(
        stream_id=stream_id,
        profiler_id=profiler_id,
        metric=metric,
        shape="scalar",
        unit=unit,
        status=status,
        reasons=reasons,
        records=(record,),
        raw_artifact_ids=artifact_ids,
    )


def _invalid_scalar_stream(
    context: ProfilerContext,
    *,
    profiler_id: str,
    stream_id: str,
    metric: str,
    unit: str,
    reason: str,
    artifact_ids: tuple[str, ...] = (),
) -> EvidenceStream:
    record = EvidenceRecord(
        record_id=f"{stream_id}:invalid",
        run_id=context.run_id,
        scope_id=context.scope.scope_id,
        profiler_id=profiler_id,
        stream_id=stream_id,
        shape="scalar",
        status="invalid",
        reasons=(reason,),
        value=None,
        raw_artifact_ids=artifact_ids,
    )
    return EvidenceStream(
        stream_id=stream_id,
        profiler_id=profiler_id,
        metric=metric,
        shape="scalar",
        unit=unit,
        status="invalid",
        reasons=(reason,),
        records=(record,),
        raw_artifact_ids=artifact_ids,
    )


def _scheduler_interval_streams(
    context: ProfilerContext,
    *,
    report: Mapping[str, Any],
    events_path: Path,
    artifact_id: str | None,
) -> tuple[EvidenceStream, EvidenceStream]:
    """Integrate retained transitions into aligned concurrency/core lanes."""
    rows: list[dict[str, Any]] = []
    try:
        lines = events_path.read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        timestamp = row.get("timestamp_ns")
        cpu = row.get("cpu")
        tid = row.get("tid")
        entering = row.get("entering")
        if isinstance(timestamp, int) and isinstance(cpu, int) and isinstance(tid, int) and isinstance(entering, bool):
            rows.append(
                {
                    "timestamp_ns": timestamp,
                    "cpu": cpu,
                    "tid": tid,
                    "entering": entering,
                }
            )
    rows.sort(
        key=lambda row: (
            row["timestamp_ns"],
            not row["entering"],
            row["cpu"],
            row["tid"],
        )
    )
    refs = (artifact_id,) if artifact_id is not None else ()
    if not rows:
        reason = "retained scheduler trace has no parseable transition events"
        return (
            EvidenceStream(
                stream_id="scheduler-concurrency",
                profiler_id="scheduler-concurrency",
                metric="scheduler.concurrent_logical_cpus",
                shape="interval",
                unit="logical_cpu",
                status="missing",
                reasons=(reason,),
                raw_artifact_ids=refs,
            ),
            EvidenceStream(
                stream_id="scheduler-per-core",
                profiler_id="scheduler-concurrency",
                metric="scheduler.logical_cpu_activity",
                shape="interval",
                unit="scheduled_interval",
                status="missing",
                reasons=(reason,),
                raw_artifact_ids=refs,
            ),
        )
    validation = report.get("cgroup_validation")
    valid = (
        bool(report.get("trace_complete"))
        and isinstance(validation, Mapping)
        and bool(validation.get("valid_for_concurrency_reporting"))
    )
    reasons = [str(item) for item in report.get("anomalies", [])]
    if not valid:
        reasons.append("scheduler trace failed completeness or cgroup-integral validation")
    reason_tuple = tuple(sorted(set(reasons)))
    status = "valid" if valid and not reason_tuple else "invalid"
    start_ns = report.get("window_start_ns")
    end_ns = report.get("window_end_ns")
    if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns <= start_ns:
        start_marker = context.lifecycle.get("measured_work_start")
        end_marker = context.lifecycle.get("measured_work_end")
        assert start_marker is not None and end_marker is not None
        start_ns = start_marker.canonical_timestamp_ns
        end_ns = end_marker.canonical_timestamp_ns

    active: dict[int, tuple[int, int]] = {}
    for row in rows:
        if row["timestamp_ns"] >= start_ns:
            break
        if row["entering"]:
            active[row["cpu"]] = (start_ns, row["tid"])
        else:
            active.pop(row["cpu"], None)
    within = [row for row in rows if start_ns <= row["timestamp_ns"] <= end_ns]
    concurrency: list[EvidenceRecord] = []
    per_core: list[EvidenceRecord] = []
    cursor = start_ns
    index = 0
    while index < len(within):
        timestamp = within[index]["timestamp_ns"]
        if timestamp > cursor:
            concurrency.append(
                EvidenceRecord(
                    record_id=f"scheduler-concurrency:{len(concurrency):08d}",
                    run_id=context.run_id,
                    scope_id=context.scope.scope_id,
                    profiler_id="scheduler-concurrency",
                    stream_id="scheduler-concurrency",
                    shape="interval",
                    status=status,
                    reasons=reason_tuple if status == "invalid" else (),
                    value=len(active),
                    source_clock_id=CANONICAL_CLOCK_ID,
                    native_start_ns=cursor,
                    native_end_ns=timestamp,
                    canonical_start_ns=cursor,
                    canonical_end_ns=timestamp,
                    alignment_error_ns=0,
                    raw_artifact_ids=refs,
                    provenance={"method": "integrated-scheduler-transitions"},
                )
            )
        group: list[dict[str, Any]] = []
        while index < len(within) and within[index]["timestamp_ns"] == timestamp:
            group.append(within[index])
            index += 1
        for row in (item for item in group if not item["entering"]):
            started = active.pop(row["cpu"], None)
            if started is not None and timestamp > started[0]:
                per_core.append(
                    EvidenceRecord(
                        record_id=f"scheduler-per-core:{len(per_core):08d}",
                        run_id=context.run_id,
                        scope_id=context.scope.scope_id,
                        profiler_id="scheduler-concurrency",
                        stream_id="scheduler-per-core",
                        shape="interval",
                        status=status,
                        reasons=reason_tuple if status == "invalid" else (),
                        value={"cpu": row["cpu"], "tid": started[1]},
                        source_clock_id=CANONICAL_CLOCK_ID,
                        native_start_ns=started[0],
                        native_end_ns=timestamp,
                        canonical_start_ns=started[0],
                        canonical_end_ns=timestamp,
                        alignment_error_ns=0,
                        raw_artifact_ids=refs,
                        attributes={"lane": f"logical-cpu-{row['cpu']}"},
                    )
                )
        for row in (item for item in group if item["entering"]):
            active[row["cpu"]] = (timestamp, row["tid"])
        cursor = timestamp
    if cursor < end_ns:
        concurrency.append(
            EvidenceRecord(
                record_id=f"scheduler-concurrency:{len(concurrency):08d}",
                run_id=context.run_id,
                scope_id=context.scope.scope_id,
                profiler_id="scheduler-concurrency",
                stream_id="scheduler-concurrency",
                shape="interval",
                status=status,
                reasons=reason_tuple if status == "invalid" else (),
                value=len(active),
                source_clock_id=CANONICAL_CLOCK_ID,
                native_start_ns=cursor,
                native_end_ns=end_ns,
                canonical_start_ns=cursor,
                canonical_end_ns=end_ns,
                alignment_error_ns=0,
                raw_artifact_ids=refs,
                provenance={"method": "integrated-scheduler-transitions"},
            )
        )
    for cpu, (started_ns, tid) in sorted(active.items()):
        if end_ns > started_ns:
            per_core.append(
                EvidenceRecord(
                    record_id=f"scheduler-per-core:{len(per_core):08d}",
                    run_id=context.run_id,
                    scope_id=context.scope.scope_id,
                    profiler_id="scheduler-concurrency",
                    stream_id="scheduler-per-core",
                    shape="interval",
                    status=status,
                    reasons=reason_tuple if status == "invalid" else (),
                    value={"cpu": cpu, "tid": tid},
                    source_clock_id=CANONICAL_CLOCK_ID,
                    native_start_ns=started_ns,
                    native_end_ns=end_ns,
                    canonical_start_ns=started_ns,
                    canonical_end_ns=end_ns,
                    alignment_error_ns=0,
                    raw_artifact_ids=refs,
                    attributes={"lane": f"logical-cpu-{cpu}"},
                )
            )

    def stream(
        stream_id: str,
        metric: str,
        unit: str,
        records: list[EvidenceRecord],
    ) -> EvidenceStream:
        if not records:
            return EvidenceStream(
                stream_id=stream_id,
                profiler_id="scheduler-concurrency",
                metric=metric,
                shape="interval",
                unit=unit,
                status="missing",
                reasons=("scheduler transitions contain no positive-duration intervals",),
                raw_artifact_ids=refs,
            )
        return EvidenceStream(
            stream_id=stream_id,
            profiler_id="scheduler-concurrency",
            metric=metric,
            shape="interval",
            unit=unit,
            status=status,
            reasons=reason_tuple if status == "invalid" else (),
            records=tuple(records),
            raw_artifact_ids=refs,
        )

    return (
        stream(
            "scheduler-concurrency",
            "scheduler.concurrent_logical_cpus",
            "logical_cpu",
            concurrency,
        ),
        stream(
            "scheduler-per-core",
            "scheduler.logical_cpu_activity",
            "scheduled_interval",
            per_core,
        ),
    )


class CoordinatedDockerSession:
    """One gated Docker execution shared by every selected profiler."""

    capabilities = frozenset(
        {
            "docker-disposable-container",
            "cgroup-v2-descendant-scope",
            "host-monotonic-clock",
            "process-trace-artifact",
        }
    )

    def __init__(self, config: Any) -> None:
        # Import lazily: container.py calls this bridge after it is initialized.
        from . import container as legacy

        self.legacy = legacy
        self.config = config
        self.output_dir = config.output_dir
        self.container_name = f"tasktrace-profile-{os.urandom(6).hex()}"
        self.process: subprocess.Popen[str] | None = None
        self.command: list[str] = []
        self.root_pid: int | None = None
        self.cgroup_path: Path | None = None
        self.cgroup_root = Path("/sys/fs/cgroup")
        self.scope: MeasuredScope | None = None
        self._poll_callbacks: list[Callable[[], None]] = []
        self._finish_started: float | None = None
        self._finish_phase = "natural"
        self._signaled_term: list[int] = []
        self._signaled_kill: list[int] = []
        self._initial_descendants: list[dict[str, Any]] = []
        self._remaining_descendants: list[dict[str, Any]] = []
        self._verification_report_value: dict[str, Any] | None = None
        # Readiness markers intentionally persist as provenance, while each
        # FIFO has exactly one reader.  Remember successful releases so an
        # abort after a later profiler failure cannot reopen an already
        # consumed FIFO and block forever waiting for a reader that exited.
        self._released_gates: set[str] = set()
        self._docker_run_started = False
        self._container_cleanup_complete = False
        self._closed = False
        self.shared: dict[str, Any] = {}

    def register_poll_callback(self, callback: Callable[[], None]) -> None:
        self._poll_callbacks.append(callback)

    def _poll_observers(self) -> None:
        for callback in tuple(self._poll_callbacks):
            callback()

    def _require_live(self) -> subprocess.Popen[str]:
        if self.process is None:
            raise self.legacy.ProfileError("coordinated Docker session has not launched")
        return self.process

    def _validated_cpu_placement(self) -> dict[str, Any]:
        """Retain kernel/runtime observations proving the requested cpuset."""
        requested_text = self.config.cpuset_cpus
        if requested_text is None:
            return {
                "requested_cpuset_cpus": None,
                "validation_status": "not-requested",
            }
        assert self.root_pid is not None and self.cgroup_path is not None
        requested = set(self.legacy.parse_cpuset_cpu_ids(requested_text))
        inspected = self.legacy._run(
            [
                self.config.docker,
                "inspect",
                "--format",
                "{{json .HostConfig.CpusetCpus}}",
                self.container_name,
            ]
        )
        if inspected.returncode != 0:
            raise self.legacy.ProfileError(
                "could not inspect the running container's Docker cpuset: " + inspected.stderr.strip()
            )
        try:
            docker_cpuset = json.loads(inspected.stdout.strip())
        except json.JSONDecodeError as exc:
            raise self.legacy.ProfileError("Docker returned malformed HostConfig.CpusetCpus evidence") from exc
        if not isinstance(docker_cpuset, str) or not docker_cpuset:
            raise self.legacy.ProfileError("Docker did not retain a non-empty HostConfig.CpusetCpus value")

        try:
            status = Path(f"/proc/{self.root_pid}/status").read_text()
            process_cpuset = next(
                line.partition(":")[2].strip() for line in status.splitlines() if line.startswith("Cpus_allowed_list:")
            )
        except (OSError, StopIteration) as exc:
            raise self.legacy.ProfileError("could not read the container init process's Cpus_allowed_list") from exc
        try:
            cgroup_cpuset = (self.cgroup_path / "cpuset.cpus.effective").read_text().strip()
        except OSError as exc:
            raise self.legacy.ProfileError("could not read the measured cgroup's cpuset.cpus.effective") from exc

        observations = {
            "Docker HostConfig.CpusetCpus": docker_cpuset,
            "container init Cpus_allowed_list": process_cpuset,
            "measured cgroup cpuset.cpus.effective": cgroup_cpuset,
        }
        for source, observed_text in observations.items():
            try:
                observed = set(self.legacy.parse_cpuset_cpu_ids(observed_text))
            except self.legacy.ProfileError as exc:
                raise self.legacy.ProfileError(f"{source} is invalid: {observed_text!r}") from exc
            if observed != requested:
                raise self.legacy.ProfileError(
                    f"{source} differs from the requested cpuset: "
                    f"requested={requested_text!r}, observed={observed_text!r}"
                )
        return {
            "requested_cpuset_cpus": requested_text,
            "docker_host_config_cpuset_cpus": docker_cpuset,
            "container_init_cpus_allowed_list": process_cpuset,
            "cgroup_cpuset_cpus_effective": cgroup_cpuset,
            "validation_status": "passed",
        }

    def _write_gate_nonblocking(self, path: Path) -> None:
        """Release one FIFO without ever waiting indefinitely for its reader."""
        process = self._require_live()
        # The entrypoint emits its readiness marker immediately before opening
        # the FIFO, so tolerate that small race.  A one-second cap keeps a
        # stale marker or dead reader from turning cleanup into another long
        # attachment wait.
        timeout_sec = min(float(self.config.scheduler_attach_timeout_sec), 1.0)
        deadline = time.monotonic() + timeout_sec
        payload = b"continue\n"
        retryable_errnos = {
            errno.EAGAIN,
            errno.EINTR,
            errno.ENOENT,
            errno.ENXIO,
            errno.EPIPE,
            errno.EWOULDBLOCK,
        }
        open_flags = os.O_WRONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)

        while True:
            returncode = process.poll()
            if returncode is not None:
                raise self.legacy.ProfileError(
                    "container exited before the coordinator FIFO gate could "
                    f"be released ({path}, status {returncode})"
                )

            fd: int | None = None
            try:
                fd = os.open(path, open_flags)
                # This payload is far smaller than PIPE_BUF, so a successful
                # FIFO write is atomic.  Never retry a partial write because
                # the reader may already have consumed it.
                written = os.write(fd, payload)
                if written != len(payload):
                    raise self.legacy.ProfileError(
                        "coordinator FIFO gate accepted only " f"{written} of {len(payload)} bytes: {path}"
                    )
                return
            except OSError as exc:
                if exc.errno not in retryable_errnos:
                    raise self.legacy.ProfileError(f"could not release coordinator FIFO gate {path}: {exc}") from exc
                remaining_sec = deadline - time.monotonic()
                if remaining_sec <= 0:
                    raise self.legacy.ProfileError(
                        "timed out waiting for the coordinator FIFO reader " f"after {timeout_sec:g} seconds: {path}"
                    ) from exc
                time.sleep(min(0.01, remaining_sec))
            finally:
                if fd is not None:
                    os.close(fd)

    def _release_gate_once(self, gate: str) -> None:
        if gate in self._released_gates:
            return
        self._write_gate_nonblocking(self.output_dir / gate)
        self._released_gates.add(gate)

    def launch(
        self,
        plan: ExecutionPlan,
        *,
        run_id: str,
        marker: LifecycleMarker,
    ) -> MeasuredScope:
        if self.process is not None:
            raise self.legacy.ProfileError("coordinated Docker session may launch only once")
        self.command = self.legacy.build_image_command(self.config, container_name=self.container_name)
        self.process = subprocess.Popen(
            self.command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._docker_run_started = True
        self.legacy._wait_for_artifact(
            self.output_dir / "sandbox_attach_ready",
            process=self.process,
            timeout_sec=self.config.scheduler_attach_timeout_sec,
            description="coordinator sandbox attachment gate",
        )
        self.root_pid = self.legacy._container_init_pid(self.config, self.container_name)
        self.legacy._wait_for_blocked_init(
            self.root_pid,
            process=self.process,
            timeout_sec=self.config.scheduler_attach_timeout_sec,
            description="coordinator sandbox attachment gate",
        )
        self.cgroup_path = self.legacy._host_cgroup_path_for_pid(self.root_pid)
        cgroup_stat = self.cgroup_path.stat()
        cpu_placement = self._validated_cpu_placement()
        time_namespace = None
        time_children_namespace = None
        try:
            time_namespace = os.readlink(f"/proc/{self.root_pid}/ns/time")
        except OSError:
            pass
        try:
            time_children_namespace = os.readlink(f"/proc/{self.root_pid}/ns/time_for_children")
        except OSError:
            pass
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            boot_id = None
        identity = {
            "runtime": "docker",
            "container_name": self.container_name,
            "root_pid": self.root_pid,
            "cgroup_path": str(self.cgroup_path),
            "cgroup_inode": cgroup_stat.st_ino,
            "cgroup_device": cgroup_stat.st_dev,
            "image_identity": _plain_json(plan.image_identity),
            "time_namespace": time_namespace,
            "time_for_children_namespace": time_children_namespace,
            "time_namespace_monotonic_offset_ns": _monotonic_offset_for_pid(self.root_pid),
            "boot_id": boot_id,
            "cpu_placement": cpu_placement,
        }
        scope_digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[
            :24
        ]
        self.scope = MeasuredScope(
            scope_id=f"docker-cgroup-{scope_digest}",
            kind="disposable-container-cgroup-subtree",
            identity=identity,
            includes_descendants=True,
            description=(
                "Disposable Docker container leaf cgroup and every descendant "
                "cgroup/process through the controlled finish gate."
            ),
            extensions={
                "canonical_clock": "CLOCK_MONOTONIC",
                "measurement_method": "linux-cgroup-v2-descendant-subtree",
                "launch_marker_ns": marker.canonical_timestamp_ns,
                "run_id": run_id,
            },
        )
        return self.scope

    def release_sandbox(self, marker: LifecycleMarker) -> None:
        self._release_gate_once("sandbox.start.gate")

    def wait_command_ready(self) -> None:
        process = self._require_live()
        self.legacy._wait_for_artifact(
            self.output_dir / "scheduler_attach_ready",
            process=process,
            timeout_sec=None,
            description="coordinator command attachment gate",
            poll_callback=self._poll_observers,
        )
        assert self.root_pid is not None
        self.legacy._wait_for_blocked_init(
            self.root_pid,
            process=process,
            timeout_sec=self.config.scheduler_attach_timeout_sec,
            description="coordinator command attachment gate",
        )

    def release_command(self, marker: LifecycleMarker) -> None:
        self._release_gate_once("scheduler.start.gate")

    def poll_command(self, timeout_sec: float) -> bool:
        if (self.output_dir / "scheduler_measurement_done").exists():
            return True
        process = self._require_live()
        if process.poll() is not None:
            _stdout, stderr = process.communicate()
            raise self.legacy.ProfileError(
                "container exited before the coordinated command finish gate: " + stderr.strip()
            )
        if timeout_sec > 0:
            time.sleep(timeout_sec)
        return (self.output_dir / "scheduler_measurement_done").exists()

    def _descendant_pids(self) -> list[int]:
        assert self.root_pid is not None and self.cgroup_path is not None
        return [pid for pid in self.legacy._cgroup_subtree_pids(self.cgroup_path) if pid != self.root_pid]

    def begin_finish(self, marker: LifecycleMarker) -> None:
        assert self.root_pid is not None and self.cgroup_path is not None
        pids = self._descendant_pids()
        self._initial_descendants = [self.legacy._process_snapshot(pid) for pid in pids]
        self._finish_started = time.monotonic()
        if self.config.terminate_descendants_at_finish and pids:
            self._finish_phase = "term"
            self._signaled_term = self.legacy._signal_cgroup_descendants(
                self.cgroup_path, root_tid=self.root_pid, sig=signal.SIGTERM
            )

    def poll_finish(self, timeout_sec: float) -> bool:
        assert self._finish_started is not None
        pids = self._descendant_pids()
        elapsed = time.monotonic() - self._finish_started
        if not pids:
            self._remaining_descendants = []
            return True
        if not self.config.terminate_descendants_at_finish:
            if elapsed >= self.legacy.SCHEDULER_DESCENDANT_DRAIN_GRACE_SEC:
                self._remaining_descendants = [self.legacy._process_snapshot(pid) for pid in pids]
                return True
        elif self._finish_phase == "term" and elapsed >= self.legacy.SCHEDULER_DESCENDANT_DRAIN_GRACE_SEC:
            assert self.cgroup_path is not None and self.root_pid is not None
            self._signaled_kill = self.legacy._signal_cgroup_descendants(
                self.cgroup_path, root_tid=self.root_pid, sig=signal.SIGKILL
            )
            self._finish_phase = "kill"
            self._finish_started = time.monotonic()
        elif self._finish_phase == "kill" and elapsed >= self.legacy.SCHEDULER_DESCENDANT_DRAIN_GRACE_SEC:
            self._remaining_descendants = [self.legacy._process_snapshot(pid) for pid in pids]
            return True
        if timeout_sec > 0:
            time.sleep(timeout_sec)
        return False

    def _runtime_report(self, *, stdout: str, stderr: str) -> dict[str, Any]:
        process = self._require_live()
        report = {
            "runtime": "docker",
            "container_name": self.container_name,
            # Do not archive argv: Docker --env arguments contain verbatim
            # values. ExecutionPlan already retains a secret-safe digest for
            # every environment value and the immutable execution settings.
            "docker_argument_count": len(self.command),
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "root_pid": self.root_pid,
            "cgroup_path": str(self.cgroup_path) if self.cgroup_path else None,
            "descendant_finish_policy": (
                "terminate-at-trial-end" if self.config.terminate_descendants_at_finish else "require-natural-exit"
            ),
            "descendant_processes_at_finish_gate": self._initial_descendants,
            "remaining_descendant_processes_at_finish_gate": self._remaining_descendants,
            "remaining_descendant_pids_at_finish_gate": [row["pid"] for row in self._remaining_descendants],
            "descendants_signaled_at_finish": {
                "sigterm": self._signaled_term,
                "sigkill": self._signaled_kill,
            },
        }
        if self.config.verification_command is not None:
            report["verification"] = self._verification_report_value or {
                "requested": True,
                "status": "not_run",
                "valid": False,
                "reasons": ["sandbox execution ended before workload verification completed"],
                "command_sha256": hashlib.sha256(self.config.verification_command.encode()).hexdigest(),
            }
        else:
            report["verification"] = {
                "requested": False,
                "status": "not_requested",
                "valid": None,
                "reasons": [],
            }
        return report

    def _load_verification_report(self) -> dict[str, Any] | None:
        command = self.config.verification_command
        if command is None:
            return None
        paths = {
            "stdout": self.output_dir / "verification.stdout.txt",
            "stderr": self.output_dir / "verification.stderr.txt",
            "exit_code": self.output_dir / "verification_exit_code",
            "status": self.output_dir / "verification_status",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            raise self.legacy.ProfileError(
                "workload verification did not retain required artifacts: " + ", ".join(sorted(missing))
            )
        try:
            exit_code = int(paths["exit_code"].read_text().strip())
        except ValueError as exc:
            raise self.legacy.ProfileError("workload verification exit code is malformed") from exc
        status = paths["status"].read_text().strip()
        expected_status = "passed" if exit_code == 0 else "failed"
        if status != expected_status:
            raise self.legacy.ProfileError("workload verification status disagrees with its exit code")
        artifacts = {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        }
        return {
            "requested": True,
            "status": status,
            "valid": status == "passed",
            "exit_code": exit_code,
            "stdout": paths["stdout"].read_text(errors="replace"),
            "stderr": paths["stderr"].read_text(errors="replace"),
            "reasons": ([] if status == "passed" else [f"verification command exited with status {exit_code}"]),
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "artifacts": artifacts,
            "measurement_boundary": "after-profiler-stop",
            "measured_counters_include_verification": False,
        }

    def finish(self, marker: LifecycleMarker) -> Mapping[str, Any]:
        process = self._require_live()
        self._release_gate_once("scheduler.finish.gate")
        stdout, stderr = process.communicate()
        self._verification_report_value = self._load_verification_report()
        return self._runtime_report(stdout=stdout, stderr=stderr)

    def abort(self, reason: str) -> Mapping[str, Any]:
        process = self._require_live()
        # Readiness identifies gates where the entrypoint may be waiting. The
        # nonblocking writer still bounds a stale marker or reader-death race.
        for ready, gate in (
            ("sandbox_attach_ready", "sandbox.start.gate"),
            ("scheduler_attach_ready", "scheduler.start.gate"),
            ("scheduler_measurement_done", "scheduler.finish.gate"),
        ):
            if gate not in self._released_gates and (self.output_dir / ready).exists() and process.poll() is None:
                try:
                    self._release_gate_once(gate)
                except (OSError, self.legacy.ProfileError):
                    pass
        if process.poll() is None:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
        else:
            stdout, stderr = process.communicate()
        report = self._runtime_report(stdout=stdout, stderr=stderr)
        report["aborted"] = True
        report["abort_reason"] = reason
        self._force_remove_container()
        return report

    def _force_remove_container(self) -> None:
        """Idempotently remove a launched container, including after client exit."""
        if not self._docker_run_started or self._container_cleanup_complete:
            return
        completed = self.legacy._run([self.config.docker, "rm", "-f", self.container_name])
        diagnostic = (completed.stderr or completed.stdout or "").strip()
        if completed.returncode == 0 or "No such container" in diagnostic:
            self._container_cleanup_complete = True
            return
        raise self.legacy.ProfileError(
            f"could not remove profiler container {self.container_name!r}: "
            f"{diagnostic or f'status {completed.returncode}'}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                try:
                    self.process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.communicate()
            except Exception as exc:
                errors.append(f"could not stop Docker client: {exc}")
        try:
            self._force_remove_container()
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            raise self.legacy.ProfileError("; ".join(errors))


class _ProfilerBase:
    def attach(self, context: ProfilerContext) -> None:
        pass

    def start(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        pass

    def poll(self, context: ProfilerContext) -> None:
        pass

    def command_end(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        pass

    def stop(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        pass

    def close(self) -> None:
        pass


class CgroupCpuProfiler(_ProfilerBase):
    _CPU_STAT_START = "cgroup_cpu.stat.observer_start.txt"
    _CPU_STAT_FINISH = "cgroup_cpu.stat.controlled_finish_gate.txt"
    _CPU_MAX_START = "cgroup_cpu.max.observer_start.txt"
    _CPU_MAX_FINISH = "cgroup_cpu.max.controlled_finish_gate.txt"

    def __init__(self, session: CoordinatedDockerSession) -> None:
        self.session = session
        self.before: dict[str, int] | None = None
        self.report: dict[str, Any] | None = None
        self.ancestor_cpu_observer: AncestorCpuObserver | None = None
        self._leaf_start: dict[str, Any] | None = None
        self._leaf_source_reads: list[dict[str, Any]] = []

    def _read_leaf_source(self, filename: str, *, phase: str) -> str | None:
        assert self.session.cgroup_path is not None
        path = self.session.cgroup_path / filename
        acquisition_start_ns = time.monotonic_ns()
        try:
            text = path.read_text()
        except OSError as exc:
            acquisition_end_ns = time.monotonic_ns()
            self._leaf_source_reads.append(
                {
                    "read_index": len(self._leaf_source_reads),
                    "phase": phase,
                    "source_scope": "leaf",
                    "filename": filename,
                    "acquisition_start_ns": acquisition_start_ns,
                    "acquisition_end_ns": acquisition_end_ns,
                    "available": False,
                    "raw_text": None,
                    "error_type": type(exc).__name__,
                }
            )
            return None
        acquisition_end_ns = time.monotonic_ns()
        self._leaf_source_reads.append(
            {
                "read_index": len(self._leaf_source_reads),
                "phase": phase,
                "source_scope": "leaf",
                "filename": filename,
                "acquisition_start_ns": acquisition_start_ns,
                "acquisition_end_ns": acquisition_end_ns,
                "available": True,
                "raw_text": text,
                "error_type": None,
            }
        )
        return text

    def _capture_leaf(self, *, phase: str) -> tuple[dict[str, Any], str | None, str | None]:
        acquisition_start_ns = time.monotonic_ns()
        errors: list[str] = []

        cpu_max_text = self._read_leaf_source("cpu.max", phase=phase)
        cpu_max_quota_usec: int | None = None
        cpu_max_period_usec: int | None = None
        cpu_max_state = "unavailable"
        if cpu_max_text is None:
            errors.append("required leaf CPU limit is unavailable: cpu.max")
        else:
            try:
                (
                    cpu_max_quota_usec,
                    cpu_max_period_usec,
                    cpu_max_state,
                ) = parse_cpu_max(cpu_max_text, source="leaf cpu.max")
            except CgroupResourceSamplingError as exc:
                cpu_max_state = "invalid"
                errors.append(str(exc))

        cpu_stat_text = self._read_leaf_source("cpu.stat", phase=phase)
        cpu_stat: dict[str, int] = {}
        cpu_stat_available = cpu_stat_text is not None
        if cpu_stat_text is None:
            errors.append("required leaf CPU accounting is unavailable: cpu.stat")
        else:
            try:
                cpu_stat = parse_flat_keyed(cpu_stat_text, source="leaf cpu.stat")
            except CgroupResourceSamplingError as exc:
                cpu_stat_available = False
                errors.append(str(exc))
            missing_keys = [key for key in REQUIRED_ANCESTOR_CPU_STAT_KEYS if key not in cpu_stat]
            if missing_keys:
                errors.append("leaf cpu.stat lacks required counters: " + ", ".join(missing_keys))

        acquisition_end_ns = time.monotonic_ns()
        if acquisition_end_ns < acquisition_start_ns:
            errors.append("leaf CPU snapshot clock moved backwards: " f"{acquisition_start_ns} -> {acquisition_end_ns}")
        return (
            {
                "acquisition_start_ns": acquisition_start_ns,
                "acquisition_end_ns": acquisition_end_ns,
                "cpu_max_quota_usec": cpu_max_quota_usec,
                "cpu_max_period_usec": cpu_max_period_usec,
                "cpu_max_state": cpu_max_state,
                "cpu_stat": cpu_stat,
                "cpu_stat_available": cpu_stat_available,
                "errors": list(dict.fromkeys(errors)),
            },
            cpu_stat_text,
            cpu_max_text,
        )

    @staticmethod
    def _leaf_report(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
        evidence_errors = [
            *[str(error) for error in before.get("errors", [])],
            *[str(error) for error in after.get("errors", [])],
        ]
        before_stat = before.get("cpu_stat", {})
        after_stat = after.get("cpu_stat", {})
        if not isinstance(before_stat, Mapping):
            before_stat = {}
            evidence_errors.append("leaf start cpu.stat parse result is not an object")
        if not isinstance(after_stat, Mapping):
            after_stat = {}
            evidence_errors.append("leaf finish cpu.stat parse result is not an object")
        cpu_stat_delta: dict[str, int] = {}
        counter_regressions: list[str] = []
        for key in sorted(set(before_stat) | set(after_stat)):
            start_value = before_stat.get(key, 0)
            finish_value = after_stat.get(key, 0)
            if not isinstance(start_value, int) or not isinstance(finish_value, int):
                counter_regressions.append(f"leaf cpu.stat counter {key!r} was not an integer")
            elif finish_value < start_value:
                counter_regressions.append(
                    f"leaf cpu.stat counter {key!r} moved backwards: " f"{start_value} -> {finish_value}"
                )
            else:
                cpu_stat_delta[str(key)] = finish_value - start_value
        evidence_errors.extend(counter_regressions)

        before_limit = (
            before.get("cpu_max_state"),
            before.get("cpu_max_quota_usec"),
            before.get("cpu_max_period_usec"),
        )
        after_limit = (
            after.get("cpu_max_state"),
            after.get("cpu_max_quota_usec"),
            after.get("cpu_max_period_usec"),
        )
        cpu_max_changed = before_limit != after_limit
        if cpu_max_changed:
            evidence_errors.append("leaf cpu.max changed during observation: " f"{before_limit} -> {after_limit}")
        finite_limit_present = before.get("cpu_max_state") == "finite" or after.get("cpu_max_state") == "finite"
        throttle_deltas = {
            key: cpu_stat_delta.get(key, 0) for key in ANCESTOR_CPU_THROTTLE_KEYS if cpu_stat_delta.get(key, 0) > 0
        }
        evidence_errors = list(dict.fromkeys(evidence_errors))
        invalid_reasons = list(evidence_errors)
        if throttle_deltas:
            rendered = ", ".join(f"{key}={value}" for key, value in sorted(throttle_deltas.items()))
            invalid_reasons.append(
                "leaf cgroup recorded CPU throttling " f"({rendered}); CPU performance evidence fails closed"
            )
        if finite_limit_present:
            invalid_reasons.append("leaf cgroup has a finite cpu.max; unconstrained CPU performance " "is not proven")
        invalid_reasons = list(dict.fromkeys(invalid_reasons))
        return {
            "start": dict(before),
            "finish": dict(after),
            "cpu_stat_delta": cpu_stat_delta,
            "cpu_max_changed": cpu_max_changed,
            "counter_regression_observed": bool(counter_regressions),
            "finite_limit_present": finite_limit_present,
            "throttle_deltas": throttle_deltas,
            "throttling_observed": bool(throttle_deltas),
            "evidence_complete": not evidence_errors,
            "limits_stable": not cpu_max_changed,
            "valid_for_cpu_performance": not invalid_reasons,
            "evidence_errors": evidence_errors,
            "invalid_reasons": invalid_reasons,
        }

    def _write_cpu_source_snapshots(self) -> None:
        assert self.ancestor_cpu_observer is not None
        payload = {
            "schema_version": _CGROUP_CPU_SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "scope": {
                "kind": "cgroup-v2-leaf-and-ancestor-chain",
                "path_representation": "relative-to-configured-cgroup-root",
            },
            "phases": ["observer_start", "controlled_finish_gate"],
            "leaf": {
                "reads": list(self._leaf_source_reads),
                "meaning": ("Exact leaf cpu.max and cpu.stat text consumed by the CPU " "performance validity gate."),
            },
            "ancestors": self.ancestor_cpu_observer.raw_source_snapshot_payload(),
            "meaning": (
                "Exact leaf and ancestor CPU controller source text retained for "
                "the shared baseline/profile CPU performance validity gate."
            ),
        }
        _json_file(self.session.output_dir / "cgroup_cpu_source_snapshots.json", payload)

    def start(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        assert self.session.cgroup_path is not None
        assert self.session.cgroup_root is not None
        self._leaf_source_reads = []
        self.ancestor_cpu_observer = AncestorCpuObserver(
            cgroup_path=self.session.cgroup_path,
            cgroup_root=self.session.cgroup_root,
        )
        self.ancestor_cpu_observer.start()
        self._leaf_start, cpu_stat_text, cpu_max_text = self._capture_leaf(phase="observer_start")
        self.before = dict(self._leaf_start["cpu_stat"])
        if cpu_stat_text is not None:
            (self.session.output_dir / self._CPU_STAT_START).write_text(cpu_stat_text)
            if self.session.config.scheduler_concurrency:
                (self.session.output_dir / "cpu.stat.before").write_text(cpu_stat_text)
        if cpu_max_text is not None:
            (self.session.output_dir / self._CPU_MAX_START).write_text(cpu_max_text)

    def stop(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        assert self.before is not None and self.session.cgroup_path is not None
        assert self._leaf_start is not None and self.ancestor_cpu_observer is not None
        leaf_finish, cpu_stat_text, cpu_max_text = self._capture_leaf(phase="controlled_finish_gate")
        if cpu_stat_text is not None:
            (self.session.output_dir / self._CPU_STAT_FINISH).write_text(cpu_stat_text)
            if self.session.config.scheduler_concurrency:
                (self.session.output_dir / "cpu.stat.after").write_text(cpu_stat_text)
        if cpu_max_text is not None:
            (self.session.output_dir / self._CPU_MAX_FINISH).write_text(cpu_max_text)
        leaf_cpu = self._leaf_report(self._leaf_start, leaf_finish)
        ancestor_cpu = self.ancestor_cpu_observer.finish()
        self._write_cpu_source_snapshots()
        # Scheduler tracing asks the gated entrypoint to create writable zero
        # placeholders because its timestamps must use the host monotonic
        # clock. Baseline mode instead owns real in-container compatibility
        # timestamps; never overwrite those root-owned files from the host.
        start = context.lifecycle.get("measured_work_start")
        assert start is not None
        if self.session.config.scheduler_concurrency:
            (self.session.output_dir / "start_ns").write_text(f"{start.canonical_timestamp_ns}\n")
            (self.session.output_dir / "end_ns").write_text(f"{marker.canonical_timestamp_ns}\n")
        delta = leaf_cpu["cpu_stat_delta"]
        wall_sec = (marker.canonical_timestamp_ns - start.canonical_timestamp_ns) / 1_000_000_000
        usage_usec = delta.get("usage_usec")
        usage_sec = usage_usec / 1_000_000 if isinstance(usage_usec, int) else None
        combined_invalid_reasons = list(
            dict.fromkeys(
                [
                    *leaf_cpu["invalid_reasons"],
                    *ancestor_cpu["invalid_reasons"],
                ]
            )
        )
        self.report = {
            "schema_version": "cgroup-cpu-coordinated.v1",
            "method": "container-cgroup-v2-cpu.stat",
            "scope_id": context.scope.scope_id,
            "window_start_ns": start.canonical_timestamp_ns,
            "window_end_ns": marker.canonical_timestamp_ns,
            "wall_sec": wall_sec,
            "usage_sec": usage_sec,
            "user_sec": (delta["user_usec"] / 1_000_000 if isinstance(delta.get("user_usec"), int) else None),
            "system_sec": (delta["system_usec"] / 1_000_000 if isinstance(delta.get("system_usec"), int) else None),
            "average_logical_cores": (usage_sec / wall_sec if usage_sec is not None and wall_sec > 0 else None),
            "cpu_stat_delta": delta,
            "leaf_cpu": leaf_cpu,
            "ancestor_cpu": ancestor_cpu,
            "raw_source_snapshot_schema_version": (_CGROUP_CPU_SOURCE_SNAPSHOT_SCHEMA_VERSION),
            "raw_source_snapshot_read_count": (
                len(self._leaf_source_reads) + int(ancestor_cpu["raw_source_read_count"])
            ),
            "valid_for_cpu_performance": not combined_invalid_reasons,
            "invalid_reasons": combined_invalid_reasons,
        }
        self.session.shared["cpu_report"] = self.report

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        if self.report is None:
            return ProfilerReport(status="invalid", reasons=("CPU profiler did not stop",))
        before_artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / self._CPU_STAT_START,
            artifact_id="cgroup-cpu-before",
            media_type="text/plain",
        )
        after_artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / self._CPU_STAT_FINISH,
            artifact_id="cgroup-cpu-after",
            media_type="text/plain",
        )
        cpu_source_artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / "cgroup_cpu_source_snapshots.json",
            artifact_id="cgroup-cpu-source-snapshots",
            media_type="application/json",
            provenance={
                "role": "exact-cgroup-cpu-controller-gate-snapshots",
                "schema_version": self.report.get("raw_source_snapshot_schema_version"),
                "includes": ["leaf", "ancestor-chain"],
                "derived_values_are_in": "cgroup-cpu raw profiler report",
            },
        )
        if cpu_source_artifact is None:
            raise CoordinatorValidationError(
                "cgroup CPU profiler did not retain exact leaf and ancestor " "gate-time source snapshots"
            )
        timeseries = self.session.shared.get("resource_timeseries")
        timeseries_artifact = (
            _artifact(
                self.session.output_dir,
                self.session.output_dir / "resource_timeseries.json",
                artifact_id="cgroup-resource-timeseries",
                media_type="application/json",
            )
            if isinstance(timeseries, dict)
            else None
        )
        artifacts = _artifacts(
            before_artifact,
            after_artifact,
            timeseries_artifact,
            cpu_source_artifact,
        )
        refs = tuple(item.artifact_id for item in artifacts)
        invalid_reasons = tuple(str(reason) for reason in self.report.get("invalid_reasons", []))
        cpu_gate_reason = "; ".join(invalid_reasons) if invalid_reasons else None
        streams = [
            _scalar_stream(
                context,
                profiler_id="cgroup-cpu",
                stream_id=stream_id,
                metric=metric,
                unit=unit,
                value=self.report[field],
                artifact_ids=refs,
                invalid_reason=cpu_gate_reason,
            )
            for stream_id, metric, unit, field in (
                ("cpu-wall", "cpu.wall_seconds", "second", "wall_sec"),
                ("cpu-usage", "cpu.usage_seconds", "second", "usage_sec"),
                ("cpu-user", "cpu.user_seconds", "second", "user_sec"),
                ("cpu-system", "cpu.system_seconds", "second", "system_sec"),
                (
                    "cpu-average-logical-cores",
                    "cpu.average_logical_cores",
                    "logical_cpu",
                    "average_logical_cores",
                ),
            )
        ]
        interval_records: list[EvidenceRecord] = []
        timeline_reason = None
        memory_report = self.session.shared.get("memory_report")
        if isinstance(memory_report, dict) and not memory_report.get("timeline_valid_for_shape", True):
            timeline_reason = (
                "; ".join(str(item) for item in memory_report.get("timeline_invalid_reasons", []))
                or "resource sampling timeline is invalid"
            )
        interval_reason = "; ".join(reason for reason in (timeline_reason, cpu_gate_reason) if reason) or None
        interval_artifact_ids = tuple(
            item
            for item in (
                (timeseries_artifact.artifact_id if timeseries_artifact is not None else None),
                cpu_source_artifact.artifact_id,
            )
            if item is not None
        )
        samples = timeseries.get("samples", []) if isinstance(timeseries, dict) else []
        previous: Mapping[str, Any] | None = None
        for row in samples:
            if not isinstance(row, Mapping):
                continue
            timestamp = row.get("timestamp_ns")
            value = row.get("cpu_average_logical_cores")
            if previous is not None:
                start_ns = previous.get("timestamp_ns")
                if (
                    isinstance(start_ns, int)
                    and isinstance(timestamp, int)
                    and timestamp > start_ns
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    previous_end = previous.get("acquisition_end_ns", start_ns)
                    current_end = row.get("acquisition_end_ns", timestamp)
                    alignment_error = max(
                        (previous_end - start_ns) if isinstance(previous_end, int) else 0,
                        (current_end - timestamp) if isinstance(current_end, int) else 0,
                        0,
                    )
                    status = "invalid" if interval_reason else "valid"
                    interval_records.append(
                        EvidenceRecord(
                            record_id=f"cpu-interval-average:{len(interval_records):06d}",
                            run_id=context.run_id,
                            scope_id=context.scope.scope_id,
                            profiler_id="cgroup-cpu",
                            stream_id="cpu-interval-average",
                            shape="window_aggregate",
                            status=status,
                            reasons=(interval_reason,) if interval_reason else (),
                            value=value,
                            source_clock_id=CANONICAL_CLOCK_ID,
                            native_start_ns=start_ns,
                            native_end_ns=timestamp,
                            canonical_start_ns=start_ns,
                            canonical_end_ns=timestamp,
                            alignment_error_ns=alignment_error,
                            raw_artifact_ids=interval_artifact_ids,
                            provenance={"method": "delta-cgroup-cpu.stat-over-sample-window"},
                        )
                    )
            previous = row
        if interval_records:
            streams.append(
                EvidenceStream(
                    stream_id="cpu-interval-average",
                    profiler_id="cgroup-cpu",
                    metric="cpu.interval_average_logical_cores",
                    shape="window_aggregate",
                    unit="logical_cpu",
                    status="invalid" if interval_reason else "valid",
                    reasons=(interval_reason,) if interval_reason else (),
                    records=tuple(interval_records),
                    raw_artifact_ids=interval_artifact_ids,
                )
            )
        else:
            streams.append(
                EvidenceStream(
                    stream_id="cpu-interval-average",
                    profiler_id="cgroup-cpu",
                    metric="cpu.interval_average_logical_cores",
                    shape="window_aggregate",
                    unit="logical_cpu",
                    status="missing",
                    reasons=tuple(
                        reason
                        for reason in (
                            "resource sampler produced no CPU interval deltas",
                            cpu_gate_reason,
                        )
                        if reason
                    ),
                    raw_artifact_ids=(cpu_source_artifact.artifact_id,),
                )
            )
        return ProfilerReport(
            status="invalid" if invalid_reasons else "valid",
            reasons=invalid_reasons,
            streams=tuple(streams),
            artifacts=artifacts,
            raw_report=self.report,
            provenance={
                "scope_includes_descendants": True,
                "cpu_policy_gate_scope": "leaf-and-every-ancestor-to-cgroup-root",
                "cpu_policy_gate_shared_by_baseline_and_profile": True,
            },
        )


class CgroupMemoryProfiler(_ProfilerBase):
    def __init__(self, session: CoordinatedDockerSession) -> None:
        self.session = session
        self.sampler: CgroupResourceSampler | None = None
        self.report: dict[str, Any] | None = None
        self.timeseries: dict[str, Any] | None = None

    def attach(self, context: ProfilerContext) -> None:
        assert self.session.cgroup_path is not None
        self.sampler = CgroupResourceSampler(
            cgroup_path=self.session.cgroup_path,
            cgroup_root=self.session.cgroup_root,
            interval_sec=self.session.config.memory_sample_interval_sec,
            observe_ancestor_cpu=False,
        )
        self.sampler.start()
        self.session.register_poll_callback(self._poll_if_due)

    def _poll_if_due(self) -> None:
        if self.sampler is not None and self.report is None:
            self.sampler.poll()

    def start(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        assert self.sampler is not None
        self.sampler.mark("command_start")

    def poll(self, context: ProfilerContext) -> None:
        self._poll_if_due()

    def command_end(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        assert self.sampler is not None
        self.sampler.mark("command_end")

    def stop(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        assert self.sampler is not None
        self.report = self.sampler.finish()
        exit_code_path = self.session.output_dir / "exit_code"
        try:
            command_exit_code = int(exit_code_path.read_text().strip())
        except (OSError, ValueError) as exc:
            raise CoordinatorValidationError(
                "cgroup memory profiler could not read the command exit code at " "the controlled finish gate"
            ) from exc
        self.session.legacy._apply_memory_workload_outcome(self.report, command_exit_code=command_exit_code)
        path = self.session.output_dir / "resource_timeseries.json"
        self.sampler.write_timeseries(path)
        self.sampler.write_raw_memory_source_snapshots(self.session.output_dir / "cgroup_memory_source_snapshots.json")
        self.timeseries = self.sampler.timeseries_payload()
        self.session.shared["resource_timeseries"] = self.timeseries
        self.report["samples_path"] = str(path.resolve())
        self.report["scope_id"] = context.scope.scope_id
        self.report["timeline_scope"] = (
            "From the pre-prepare lifecycle gate through command completion and "
            "bounded descendant drain. Container startup before observer "
            "attachment is retained by the fresh-cgroup memory.peak."
        )
        self.report["scope_coverage"] = {
            "kind": "disposable-container-cgroup-subtree",
            "includes": [
                "container init and profiling entrypoint",
                "prepare command",
                "measured command and descendants",
                "cgroup-charged anonymous, file-cache, and kernel memory",
            ],
            "excludes": [
                "post-measurement workload verification, final finish-gate FIFO "
                "unblock, and shell/container-init exit",
                "host profiler process",
                "Docker daemon and container runtime processes",
                "sibling or sidecar containers outside this cgroup",
                "host page cache not charged to this cgroup",
            ],
            "complete_for_profiled_container": False,
            "complete_through_controlled_finish_gate": True,
            "container_init_exit_observed": False,
            "full_production_sandbox_requires_all_components_in_scope": True,
        }
        self.report["descendant_finish_policy"] = (
            "terminate-at-trial-end" if self.session.config.terminate_descendants_at_finish else "require-natural-exit"
        )
        self.report["remaining_descendant_pids_at_finish_gate"] = [
            row["pid"] for row in self.session._remaining_descendants
        ]
        if self.session._remaining_descendants:
            reason = "descendants remained after the bounded finish drain"
            self.report["observation_boundary_complete"] = False
            self.report["natural_peak_trustworthy"] = False
            reasons = self.report["natural_peak_invalid_reasons"]
            if reason not in reasons:
                reasons.append(reason)
        self.session.shared["memory_report"] = self.report

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        if self.report is None or self.timeseries is None:
            return ProfilerReport(status="invalid", reasons=("memory profiler did not stop",))
        path = self.session.output_dir / "resource_timeseries.json"
        artifact = _artifact(
            self.session.output_dir,
            path,
            artifact_id="cgroup-resource-timeseries",
            media_type="application/json",
        )
        source_artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / "cgroup_memory_source_snapshots.json",
            artifact_id="cgroup-memory-source-snapshots",
            media_type="application/json",
            provenance={
                "role": "exact-cgroup-memory-controller-gate-snapshots",
                "schema_version": self.report.get("raw_source_snapshot_schema_version"),
                "derived_values_are_in": "cgroup-memory raw profiler report",
            },
        )
        if artifact is None or source_artifact is None:
            raise CoordinatorValidationError(
                "cgroup memory profiler did not retain both its shared time series "
                "and exact gate-time source snapshots"
            )
        artifacts = _artifacts(artifact, source_artifact)
        timeline_refs = (artifact.artifact_id,)
        exact_refs = tuple(
            dict.fromkeys(
                (
                    *timeline_refs,
                    source_artifact.artifact_id,
                )
            )
        )
        timeline_reasons = tuple(str(value) for value in self.report.get("timeline_invalid_reasons", []))
        timeline_reason = "; ".join(timeline_reasons) if timeline_reasons else None
        cadence_ns = round(float(self.report["sample_interval_sec"]) * 1_000_000_000)
        cadence = Cadence(
            requested_interval_ns=cadence_ns,
            maximum_acceptable_gap_ns=2 * cadence_ns,
            method="cgroup-v2-periodic-read",
        )
        streams: list[EvidenceStream] = []
        samples = self.timeseries.get("samples", [])
        for stream_id, metric, field, parent in (
            ("memory-current", "memory.current", "memory_current_bytes", None),
            ("memory-anon", "memory.anon", "anon", "memory_stat"),
            ("memory-file", "memory.file", "file", "memory_stat"),
            ("memory-kernel", "memory.kernel", "kernel", "memory_stat"),
        ):
            records: list[EvidenceRecord] = []
            invalid_reason = timeline_reason
            for index, row in enumerate(samples):
                source = row.get(parent, {}) if parent else row
                value = source.get(field) if isinstance(source, dict) else None
                timestamp = row.get("timestamp_ns")
                acquisition_start = row.get("acquisition_start_ns", timestamp)
                acquisition_end = row.get("acquisition_end_ns", timestamp)
                if not all(
                    isinstance(item, int) for item in (timestamp, acquisition_start, acquisition_end)
                ) or not isinstance(value, (int, float)):
                    continue
                status = "invalid" if invalid_reason else "valid"
                records.append(
                    EvidenceRecord(
                        record_id=f"{stream_id}:{index:06d}",
                        run_id=context.run_id,
                        scope_id=context.scope.scope_id,
                        profiler_id="cgroup-memory",
                        stream_id=stream_id,
                        shape="sampled_gauge",
                        status=status,
                        reasons=(invalid_reason,) if invalid_reason else (),
                        value=value,
                        source_clock_id=CANONICAL_CLOCK_ID,
                        native_timestamp_ns=timestamp,
                        acquisition_bounds=AcquisitionBounds(acquisition_start, acquisition_end),
                        alignment_error_ns=acquisition_end - acquisition_start,
                        raw_artifact_ids=timeline_refs,
                    )
                )
            if records:
                streams.append(
                    EvidenceStream(
                        stream_id=stream_id,
                        profiler_id="cgroup-memory",
                        metric=metric,
                        shape="sampled_gauge",
                        unit="byte",
                        status="invalid" if invalid_reason else "valid",
                        reasons=(invalid_reason,) if invalid_reason else (),
                        records=tuple(records),
                        cadence=cadence,
                        raw_artifact_ids=timeline_refs,
                    )
                )
            else:
                streams.append(
                    EvidenceStream(
                        stream_id=stream_id,
                        profiler_id="cgroup-memory",
                        metric=metric,
                        shape="sampled_gauge",
                        unit="byte",
                        status="missing",
                        reasons=(f"no {metric} samples were observed",),
                        raw_artifact_ids=timeline_refs,
                    )
                )
        peak = self.report.get("lifetime", {}).get("peak_bytes")
        peak_reasons = [str(item) for item in self.report.get("peak_invalid_reasons", [])]
        peak_reason = "; ".join(dict.fromkeys(peak_reasons)) or None
        streams.append(
            _scalar_stream(
                context,
                profiler_id="cgroup-memory",
                stream_id="memory-peak",
                metric="memory.peak",
                unit="byte",
                value=peak,
                artifact_ids=exact_refs,
                invalid_reason=peak_reason,
                attributes={
                    "exact": self.report.get("peak_valid") is True,
                    "natural_peak_trustworthy": self.report.get("natural_peak_trustworthy") is True,
                    "censoring": self.report.get("censoring"),
                },
            )
        )
        limits = self.report.get("limits", {})
        memory_max = limits.get("memory_max_bytes")
        if limits.get("memory_max_state") == "finite" and isinstance(memory_max, int):
            streams.append(
                _scalar_stream(
                    context,
                    profiler_id="cgroup-memory",
                    stream_id="memory-max",
                    metric="memory.max",
                    unit="byte",
                    value=memory_max,
                    artifact_ids=exact_refs,
                    attributes={"exact": True, "timing": "observer-start"},
                )
            )
        margin = self.report.get("observed_limit_margin", {})
        remaining = margin.get("remaining_to_limit_bytes")
        if isinstance(remaining, int):
            streams.append(
                _scalar_stream(
                    context,
                    profiler_id="cgroup-memory",
                    stream_id="memory-observed-limit-margin",
                    metric="memory.observed_limit_margin",
                    unit="byte",
                    value=remaining,
                    artifact_ids=exact_refs,
                    invalid_reason=peak_reason,
                    attributes={
                        "exact": peak_reason is None,
                        "not_a_recommendation": True,
                        "peak_fraction_of_limit": margin.get("peak_fraction_of_limit"),
                    },
                )
            )
        instrumentation_reason = None
        if not self.report.get("instrumentation_valid", False):
            instrumentation_reason = (
                "; ".join(str(item) for item in self.report.get("instrumentation_invalid_reasons", []))
                or "memory instrumentation validity was not established"
            )
        for event_name, event_value in sorted(self.report.get("events_delta", {}).items()):
            if isinstance(event_value, int) and not isinstance(event_value, bool):
                streams.append(
                    _scalar_stream(
                        context,
                        profiler_id="cgroup-memory",
                        stream_id=f"memory-event-{event_name}",
                        metric=f"memory.events.{event_name}",
                        unit="event",
                        value=event_value,
                        artifact_ids=exact_refs,
                        invalid_reason=instrumentation_reason,
                    )
                )
        marker_times = {
            marker.get("name"): marker.get("timestamp_ns")
            for marker in self.report.get("markers", [])
            if isinstance(marker, Mapping)
        }
        pressure_start = marker_times.get("sandbox_observation_start")
        pressure_end = marker_times.get("sandbox_observation_end")
        for pressure_kind, pressure in sorted(self.report.get("pressure_delta", {}).items()):
            total_stall = pressure.get("total_stall_usec") if isinstance(pressure, Mapping) else None
            if (
                isinstance(total_stall, int)
                and isinstance(pressure_start, int)
                and isinstance(pressure_end, int)
                and pressure_end > pressure_start
            ):
                status_value = "invalid" if instrumentation_reason else "valid"
                pressure_record = EvidenceRecord(
                    record_id=f"memory-pressure-{pressure_kind}:window",
                    run_id=context.run_id,
                    scope_id=context.scope.scope_id,
                    profiler_id="cgroup-memory",
                    stream_id=f"memory-pressure-{pressure_kind}",
                    shape="window_aggregate",
                    status=status_value,
                    reasons=(instrumentation_reason,) if instrumentation_reason else (),
                    value=total_stall,
                    source_clock_id=CANONICAL_CLOCK_ID,
                    native_start_ns=pressure_start,
                    native_end_ns=pressure_end,
                    canonical_start_ns=pressure_start,
                    canonical_end_ns=pressure_end,
                    alignment_error_ns=0,
                    raw_artifact_ids=exact_refs,
                    provenance={"timing": "aggregate-only-stall-occurrence-time-unknown"},
                )
                streams.append(
                    EvidenceStream(
                        stream_id=f"memory-pressure-{pressure_kind}",
                        profiler_id="cgroup-memory",
                        metric=f"memory.pressure.{pressure_kind}.total_stall",
                        shape="window_aggregate",
                        unit="microsecond",
                        status=status_value,
                        reasons=(instrumentation_reason,) if instrumentation_reason else (),
                        records=(pressure_record,),
                        raw_artifact_ids=exact_refs,
                    )
                )
        status = "invalid" if instrumentation_reason or peak_reason or timeline_reason else "valid"
        reasons = tuple(
            dict.fromkeys(item for item in (instrumentation_reason, peak_reason, timeline_reason) if item is not None)
        )
        return ProfilerReport(
            status=status,
            reasons=reasons,
            streams=tuple(streams),
            artifacts=artifacts,
            raw_report=self.report,
            provenance={
                "scope_includes_descendants": True,
                "ancestor_chain_observed": True,
                "peak_source": "memory.peak",
            },
        )


class SchedulerProfiler(_ProfilerBase):
    def __init__(self, session: CoordinatedDockerSession) -> None:
        self.session = session
        self.tracer: Any | None = None
        self.error: str | None = None
        self.report: dict[str, Any] | None = None
        self.start_ns: int | None = None

    def attach(self, context: ProfilerContext) -> None:
        """Compile and attach while the disposable sandbox is still gated.

        BCC/perf initialization can take seconds. Doing it from ``start``
        would place that host-side setup delay after ``measured_work_start``,
        inflate workload wall time, and starve other coordinator-polled
        samplers. The tracer may retain pre-window transitions; its summary is
        explicitly clipped to the later canonical start/end markers.
        """
        try:
            self.tracer = self.session.legacy._build_scheduler_tracer(self.session.config)
            assert self.session.root_pid is not None and self.session.cgroup_path is not None
            self.tracer.start(self.session.root_pid, cgroup_path=self.session.cgroup_path)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if self.tracer is not None:
                self.tracer.close()
                self.tracer = None

    def start(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        self.start_ns = marker.canonical_timestamp_ns

    def poll(self, context: ProfilerContext) -> None:
        if self.tracer is None or self.error is not None:
            return
        try:
            self.tracer.poll(timeout_ms=0)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        if self.tracer is None or self.error is not None:
            return
        try:
            assert self.start_ns is not None
            report = self.tracer.finish(
                window_start_ns=self.start_ns,
                window_end_ns=marker.canonical_timestamp_ns,
            )
            report["window_start_ns"] = self.start_ns
            report["window_end_ns"] = marker.canonical_timestamp_ns
            cpu = self.session.shared.get("cpu_report")
            if isinstance(cpu, dict):
                report = add_cgroup_validation(
                    report,
                    cgroup_usage_sec=float(cpu["usage_sec"]),
                    tolerance_fraction=self.session.config.scheduler_cpu_tolerance_fraction,
                    tolerance_sec=self.session.config.scheduler_cpu_tolerance_sec,
                    per_observed_event_tolerance_sec=(
                        self.session.legacy.BCC_SCHEDULER_EVENT_BOUNDARY_TOLERANCE_SEC
                        if self.session.config.scheduler_backend == "bcc"
                        else 0.0
                    ),
                )
            report["scope_id"] = context.scope.scope_id
            report["remaining_descendant_pids_at_finish_gate"] = [
                row["pid"] for row in self.session._remaining_descendants
            ]
            events_path = self.session.output_dir / "scheduler_events.jsonl"
            self.tracer.write_events(events_path)
            report["events_path"] = str(events_path)
            self.report = report
            self.session.shared["scheduler_report"] = report
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        if self.error is not None or self.report is None:
            reason = self.error or "scheduler tracer produced no report"
            raw = {
                "schema_version": "scheduler-concurrency-unavailable.v1",
                "instrumentation_valid": False,
                "trace_complete": False,
                "peak_simultaneous_logical_cores": None,
                "scheduled_runtime_sec": None,
                "cgroup_validation": {
                    "valid_for_concurrency_reporting": False,
                    "within_tolerance": False,
                    "invalid_reason": reason,
                },
                "invalid_reasons": [reason],
                "scope_id": context.scope.scope_id,
            }
            self.session.shared["scheduler_report"] = raw
            return ProfilerReport(status="invalid", reasons=(reason,), raw_report=raw)
        artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / "scheduler_events.jsonl",
            artifact_id="scheduler-events",
            media_type="application/x-ndjson",
        )
        artifacts = _artifacts(artifact)
        refs = tuple(item.artifact_id for item in artifacts)
        valid = bool(self.report.get("cgroup_validation", {}).get("valid_for_concurrency_reporting", False))
        reason = None if valid else "scheduler trace failed cgroup-integral validation"
        streams = [
            _scalar_stream(
                context,
                profiler_id="scheduler-concurrency",
                stream_id="cpu-peak-simultaneous",
                metric="cpu.peak_simultaneous_logical_cores",
                unit="logical_cpu",
                value=self.report.get("peak_simultaneous_logical_cores"),
                artifact_ids=refs,
                invalid_reason=reason,
            ),
            _scalar_stream(
                context,
                profiler_id="scheduler-concurrency",
                stream_id="cpu-scheduled-runtime",
                metric="cpu.scheduled_runtime_seconds",
                unit="second",
                value=self.report.get("scheduled_runtime_sec"),
                artifact_ids=refs,
                invalid_reason=reason,
            ),
        ]
        streams.extend(
            _scheduler_interval_streams(
                context,
                report=self.report,
                events_path=self.session.output_dir / "scheduler_events.jsonl",
                artifact_id=artifact.artifact_id if artifact is not None else None,
            )
        )
        return ProfilerReport(
            status="valid" if valid else "invalid",
            reasons=() if valid else (reason,),
            streams=tuple(streams),
            artifacts=artifacts,
            raw_report=self.report,
        )

    def close(self) -> None:
        if self.tracer is not None:
            self.tracer.close()


class ProcessExecProfiler(_ProfilerBase):
    def __init__(self, session: CoordinatedDockerSession) -> None:
        self.session = session

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        trace_path = self.session.output_dir / "process.strace.log"
        report_path = self.session.output_dir / "process_exec_report.json"
        if not trace_path.is_file():
            reason = "process strace artifact is missing"
            return ProfilerReport(status="missing", reasons=(reason,))
        report = load_process_exec_trace(trace_path, initial_cwd=self.session.config.workdir)
        # Match build_report's retained compatibility payload byte-for-byte;
        # it parses the same trace after coordination and must not invalidate
        # the bundle's artifact digest by rewriting different metadata.
        report["artifacts"] = {
            "raw_trace_path": str(trace_path),
            "report_path": str(report_path),
        }
        write_process_exec_report(report, report_path)
        raw_artifact = _artifact(
            self.session.output_dir,
            trace_path,
            artifact_id="process-strace-raw",
            media_type="text/plain",
        )
        report_artifact = _artifact(
            self.session.output_dir,
            report_path,
            artifact_id="process-exec-report",
            media_type="application/json",
        )
        artifacts = _artifacts(raw_artifact, report_artifact)
        refs = tuple(item.artifact_id for item in artifacts)
        reasons = tuple(
            sorted(
                {
                    str(item.get("message"))
                    for item in report.get("anomalies", [])
                    if isinstance(item, dict) and item.get("message")
                }
            )
        )
        valid = bool(report.get("valid"))
        invalid_reason = None if valid else "; ".join(reasons) or "process trace is invalid"
        streams: list[EvidenceStream] = [
            _scalar_stream(
                context,
                profiler_id="process-exec",
                stream_id="process-count",
                metric="software.process_count",
                unit="process",
                value=report.get("summary", {}).get("process_count"),
                artifact_ids=refs,
                invalid_reason=invalid_reason,
            ),
            _scalar_stream(
                context,
                profiler_id="process-exec",
                stream_id="binary-count",
                metric="software.distinct_binary_count",
                unit="binary",
                value=len(report.get("binary_inventory", [])),
                artifact_ids=refs,
                invalid_reason=invalid_reason,
            ),
        ]
        events: list[EvidenceRecord] = []
        for process in report.get("processes", []):
            if not isinstance(process, dict):
                continue
            for event in process.get("execs", []):
                if not isinstance(event, dict) or not isinstance(event.get("started_ns"), int):
                    continue
                value = {
                    "pid": process.get("pid"),
                    "binary": event.get("binary"),
                    "argv": event.get("argv"),
                    "cwd": event.get("cwd"),
                    "syscall": event.get("syscall"),
                }
                events.append(
                    EvidenceRecord(
                        record_id=f"process-exec-event:{len(events):06d}",
                        run_id=context.run_id,
                        scope_id=context.scope.scope_id,
                        profiler_id="process-exec",
                        stream_id="process-exec-events",
                        shape="point_event",
                        status="unalignable",
                        reasons=(
                            "strace CLOCK_REALTIME was not correlated to host " "CLOCK_MONOTONIC during this run",
                        ),
                        value=value,
                        source_clock_id="strace-realtime",
                        native_timestamp_ns=event["started_ns"],
                        raw_artifact_ids=refs,
                    )
                )
        if events:
            streams.append(
                EvidenceStream(
                    stream_id="process-exec-events",
                    profiler_id="process-exec",
                    metric="software.exec",
                    shape="point_event",
                    unit="event",
                    status="unalignable",
                    reasons=("strace CLOCK_REALTIME was not correlated to host " "CLOCK_MONOTONIC during this run",),
                    records=tuple(events),
                    raw_artifact_ids=refs,
                )
            )
        return ProfilerReport(
            status="invalid" if not valid else "unalignable",
            reasons=(
                (invalid_reason,)
                if not valid
                else ("strace event timestamps cannot be placed on the canonical monotonic axis",)
            ),
            streams=tuple(streams),
            artifacts=artifacts,
            clocks=(
                ClockReference(
                    clock_id="strace-realtime",
                    clock_name="CLOCK_REALTIME",
                    # strace -ttt defaults to microsecond precision even
                    # though the parser stores the value as integer ns.
                    resolution_ns=1_000,
                    namespace_identity=str(context.scope.identity.get("time_namespace") or "unknown"),
                ),
            ),
            clock_mappings=(
                ClockMapping(
                    source_clock_id="strace-realtime",
                    canonical_clock_id=CANONICAL_CLOCK_ID,
                    status="unalignable",
                    method="no-runtime-clock-correlation",
                    reasons=("no bounded CLOCK_REALTIME to CLOCK_MONOTONIC correlation was captured",),
                ),
            ),
            raw_report=report,
        )


class HardwarePerfProfiler(_ProfilerBase):
    def __init__(self, session: CoordinatedDockerSession) -> None:
        self.session = session
        self.collector: PerfCgroupHardwareProfiler | None = None
        self.error: str | None = None
        self.start_ns: int | None = None
        self.report: dict[str, Any] | None = None

    def attach(self, context: ProfilerContext) -> None:
        """Start perf disabled so process setup is outside the counted window."""
        try:
            self.collector = PerfCgroupHardwareProfiler(
                output_dir=self.session.output_dir,
                cpuset_cpus=self.session.config.cpuset_cpus,
                perf=self.session.config.hardware_perf_binary,
                events=self.session.config.hardware_perf_events,
                attach_timeout_sec=self.session.config.hardware_perf_attach_timeout_sec,
                minimum_running_ratio=self.session.config.hardware_perf_minimum_running_ratio,
            )
            assert self.session.cgroup_path is not None
            self.collector.attach(cgroup_path=self.session.cgroup_path)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            if self.collector is not None:
                self.collector.close()

    def start(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        self.start_ns = marker.canonical_timestamp_ns
        if self.collector is None or self.error is not None:
            return
        try:
            self.collector.enable()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def poll(self, context: ProfilerContext) -> None:
        if self.collector is None or self.error is not None:
            return
        try:
            self.collector.poll(timeout_ms=0)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def stop(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        if self.collector is None or self.error is not None:
            return
        try:
            assert self.start_ns is not None
            self.report = self.collector.finish(
                window_start_ns=self.start_ns,
                window_end_ns=marker.canonical_timestamp_ns,
            )
            path = self.session.output_dir / "hardware_perf_report.json"
            self.collector.write_report(self.report, path)
            self.session.shared["hardware_report"] = self.report
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        if self.error is not None or self.report is None:
            reason = self.error or "hardware perf profiler produced no report"
            provenance = getattr(self.collector, "provenance", {}) if self.collector is not None else {}
            raw = {
                "schema_version": "cgroup-perf-hardware-unavailable.v1",
                "instrumentation_valid": False,
                "all_requested_metrics_valid": False,
                "invalid_reasons": [reason],
                "provenance": provenance,
                "counters": [
                    {
                        "event": event,
                        "status": "unavailable",
                        "value": None,
                        "valid": False,
                        "invalid_reason": reason,
                    }
                    for event in self.session.config.hardware_perf_events
                ],
            }
            self.session.shared["hardware_report"] = raw
            report_path = self.session.output_dir / "hardware_perf_report.json"
            _json_file(report_path, raw)
            artifacts = _artifacts(
                _artifact(
                    self.session.output_dir,
                    report_path,
                    artifact_id="hardware-perf-report",
                    media_type="application/json",
                ),
                _artifact(
                    self.session.output_dir,
                    self.session.output_dir / "hardware_perf_stat.csv",
                    artifact_id="hardware-perf-stat",
                    media_type="text/csv",
                ),
                _artifact(
                    self.session.output_dir,
                    self.session.output_dir / "hardware_perf.stderr.txt",
                    artifact_id="hardware-perf-stderr",
                    media_type="text/plain",
                ),
            )
            refs = tuple(item.artifact_id for item in artifacts)
            streams = tuple(
                _invalid_scalar_stream(
                    context,
                    profiler_id="hardware-perf",
                    stream_id=f"hardware-{event.replace('/', '-')}",
                    metric=f"hardware.{event}",
                    unit="count",
                    reason=reason,
                    artifact_ids=refs,
                )
                for event in self.session.config.hardware_perf_events
            )
            return ProfilerReport(
                status="invalid",
                reasons=(reason,),
                streams=streams,
                artifacts=artifacts,
                raw_report=raw,
                provenance=provenance,
            )
        report_artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / "hardware_perf_report.json",
            artifact_id="hardware-perf-report",
            media_type="application/json",
        )
        stat_artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / "hardware_perf_stat.csv",
            artifact_id="hardware-perf-stat",
            media_type="text/csv",
        )
        stderr_artifact = _artifact(
            self.session.output_dir,
            self.session.output_dir / "hardware_perf.stderr.txt",
            artifact_id="hardware-perf-stderr",
            media_type="text/plain",
        )
        artifacts = _artifacts(report_artifact, stat_artifact, stderr_artifact)
        refs = tuple(item.artifact_id for item in artifacts)
        streams: list[EvidenceStream] = []
        reasons: list[str] = []
        window_start = self.report.get("window_start_ns")
        window_end = self.report.get("window_end_ns")
        if not isinstance(window_start, int) or not isinstance(window_end, int) or window_end <= window_start:
            reason = "hardware report has no valid acknowledged counter window"
            return ProfilerReport(
                status="invalid",
                reasons=(reason,),
                artifacts=artifacts,
                raw_report=self.report,
            )
        alignment_error = max(
            int(self.report.get("window_start_alignment_error_ns") or 0),
            int(self.report.get("window_end_alignment_error_ns") or 0),
        )
        for counter in self.report.get("counters", []):
            event = str(counter.get("event"))
            valid = bool(counter.get("valid"))
            reason = None if valid else str(counter.get("invalid_reason") or "counter invalid")
            if reason:
                reasons.append(reason)
            stream_id = f"hardware-{event.replace('/', '-')}"
            record_status = "valid" if valid else "invalid"
            record_reason = reason or "hardware counter did not produce a value"
            record = EvidenceRecord(
                record_id=f"{stream_id}:window",
                run_id=context.run_id,
                scope_id=context.scope.scope_id,
                profiler_id="hardware-perf",
                stream_id=stream_id,
                shape="window_aggregate",
                status=record_status,
                reasons=() if valid else (record_reason,),
                value=counter.get("value"),
                source_clock_id=CANONICAL_CLOCK_ID,
                native_start_ns=window_start,
                native_end_ns=window_end,
                canonical_start_ns=window_start,
                canonical_end_ns=window_end,
                alignment_error_ns=alignment_error,
                raw_artifact_ids=refs,
                attributes={
                    "time_enabled_ns": counter.get("time_enabled_ns"),
                    "time_running_ns": counter.get("time_running_ns"),
                    "time_running_fraction": counter.get("time_running_fraction"),
                    "definite_active_window_start_ns": self.report.get("definite_active_window_start_ns"),
                    "definite_active_window_end_ns": self.report.get("definite_active_window_end_ns"),
                },
            )
            streams.append(
                EvidenceStream(
                    stream_id=stream_id,
                    profiler_id="hardware-perf",
                    metric=f"hardware.{event}",
                    shape="window_aggregate",
                    unit=str(counter.get("unit") or "count"),
                    status=record_status,
                    reasons=() if valid else (record_reason,),
                    records=(record,),
                    raw_artifact_ids=refs,
                )
            )
        valid = bool(self.report.get("all_requested_metrics_valid"))
        return ProfilerReport(
            status="valid" if valid else "invalid",
            reasons=() if valid else tuple(sorted(set(reasons))) or ("hardware counters invalid",),
            streams=tuple(streams),
            artifacts=artifacts,
            raw_report=self.report,
            provenance=(
                self.report.get("provenance", {}) if isinstance(self.report.get("provenance"), Mapping) else {}
            ),
        )

    def close(self) -> None:
        if self.collector is not None:
            self.collector.close()


class _ContainerProfilerRegistry(ProfilerRegistry):
    """Resolve the common CPU validity gate without changing user selection.

    The original request remains in bundle provenance.  This container bridge
    adds cgroup CPU accounting internally because both a coordinated baseline
    and every profiled run need the same leaf/ancestor constraint check.
    """

    def resolve(self, request: ProfileRequest) -> tuple[ResolvedProfiler, ...]:
        if request.mode == "baseline":
            effective_request = ProfileRequest(
                mode="profile",
                profilers=("cgroup-cpu",),
            )
        else:
            effective_request = ProfileRequest(
                mode="profile",
                profile_sets=request.profile_sets,
                profilers=(*request.profilers, "cgroup-cpu"),
                configuration=request.configuration,
            )
        return super().resolve(effective_request)


def build_profiler_registry(
    session: CoordinatedDockerSession,
) -> ProfilerRegistry:
    registry = _ContainerProfilerRegistry()
    registry.register(
        ProfilerDescriptor(
            profiler_id="scheduler-concurrency",
            version=_VERSION,
            requirements=("cgroup-v2-descendant-scope",),
            measurement_domains=("cpu.capacity",),
            order=100,
        ),
        lambda _configuration: SchedulerProfiler(session),
    )
    registry.register(
        ProfilerDescriptor(
            profiler_id="hardware-perf",
            version=_VERSION,
            requirements=("cgroup-v2-descendant-scope",),
            measurement_domains=("cpu.capacity",),
            order=150,
        ),
        lambda _configuration: HardwarePerfProfiler(session),
    )
    registry.register(
        ProfilerDescriptor(
            profiler_id="cgroup-cpu",
            version=_VERSION,
            requirements=("cgroup-v2-descendant-scope",),
            measurement_domains=("cpu.capacity",),
            order=200,
        ),
        lambda _configuration: CgroupCpuProfiler(session),
    )
    registry.register(
        ProfilerDescriptor(
            profiler_id="cgroup-memory",
            version=_VERSION,
            requirements=("cgroup-v2-descendant-scope",),
            measurement_domains=("memory.capacity",),
            order=300,
        ),
        lambda _configuration: CgroupMemoryProfiler(session),
    )
    registry.register(
        ProfilerDescriptor(
            profiler_id="process-exec",
            version=_VERSION,
            requirements=("process-trace-artifact",),
            measurement_domains=("software.process",),
            intrusive=True,
            intrusive_reason="strace runs inside the measured container cgroup",
            invalidates_metrics=("cpu.capacity", "memory.capacity"),
            order=400,
        ),
        lambda _configuration: ProcessExecProfiler(session),
    )
    return registry


def _environment(assignments: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if name:
            values[name] = value if separator else ""
    return values


def _mounts(
    config: Any,
) -> tuple[MountBehavior, ...]:
    from . import container as legacy

    mounts: list[MountBehavior] = []
    for raw in config.mounts:
        if raw.startswith("type=") or "src=" in raw or "source=" in raw:
            source, target, read_only = legacy._parse_docker_bind_mount(raw)
        else:
            fields = raw.split(":")
            if len(fields) < 2:
                raise legacy.ProfileError(f"invalid bind mount specification: {raw!r}")
            source, target = fields[0], fields[1]
            read_only = len(fields) > 2 and "ro" in fields[2].split(",")
        source_path = Path(source).resolve()
        pipeline_replay_output = (config.output_dir.parent / "replay").resolve()
        if target == "/tasktrace-source":
            # The raw checkout path is deployment detail.  This mount carries
            # the controller/replay implementation, so compare its content
            # identity using the same digest embedded in the execution plan.
            # Hash the mounted checkout itself: a caller cannot point the
            # mount at different content while retaining the current
            # controller's identity.
            source_identity = {
                "kind": "controller-source-tree",
                "controller_source_sha256": legacy.controller_source_sha256(source_path),
                "host_path_in_comparability": False,
            }
        elif not read_only and target == "/tasktrace" and source_path == pipeline_replay_output:
            # Every attempt gets a different host output directory. Its path is
            # operational state, not workload identity; source/workspace
            # content identity is already carried by ExecutionPlan.source_identity.
            # Keeping the path hash here would make an otherwise identical
            # baseline/profile pair incomparable by construction.
            source_identity = {
                "kind": "run-local-replay-output",
                "host_path_in_comparability": False,
            }
        elif target == "/tasktrace-controller/tasktrace":
            expected_digest = (config.source_identity or {}).get("rust_controller_binary_sha256")
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise legacy.ProfileError("Rust controller mount is missing its authoritative SHA-256")
            if not source_path.is_file():
                raise legacy.ProfileError(f"Rust controller mount is not a file: {source_path}")
            actual_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual_digest != expected_digest:
                raise legacy.ProfileError(
                    "Rust controller binary changed after adapter preflight: "
                    f"expected {expected_digest}, observed {actual_digest}"
                )
            source_identity = {
                "kind": "rust-controller-binary",
                "sha256": actual_digest,
                "host_path_in_comparability": False,
            }
        else:
            source_identity = {"source_sha256": hashlib.sha256(source.encode()).hexdigest()}
        mounts.append(
            MountBehavior(
                target=target,
                kind="bind",
                read_only=read_only,
                source_identity=source_identity,
            )
        )
    return tuple(mounts)


@dataclass(frozen=True)
class CoordinatedProfileResult:
    completed: subprocess.CompletedProcess[str]
    effective_config: Any
    coordination: Any
    bundle_path: Path
    bundle_sha256: str
    cpu_report: dict[str, Any]
    scheduler_report: dict[str, Any] | None
    memory_report: dict[str, Any] | None
    hardware_report: dict[str, Any] | None
    canonical_clock_reference: dict[str, Any]
    canonical_lifecycle: dict[str, Any]


def coordinated_effective_config(config: Any) -> Any:
    """Apply named profile-set perturbation and observer requirements."""
    sets = frozenset(config.profile_sets)
    if config.coordinated_baseline:
        return replace(
            config,
            strace_mode="none",
            scheduler_concurrency=False,
            proc_stat_comparison=False,
            memory_profile=False,
        )
    return replace(
        config,
        # Profile sets own their perturbation contract. CPU/hardware evidence
        # must never inherit the legacy CLI's execve-strace default silently;
        # only software-deep opts into the intrusive process trace.
        strace_mode=("process" if "software-deep" in sets else "none"),
        scheduler_concurrency=(config.scheduler_concurrency or bool({"cpu-compare", "cpu-peak"} & sets)),
        memory_profile=(config.memory_profile or bool({"cpu-compare", "memory-detail"} & sets)),
    )


def _profile_configuration(config: Any) -> dict[str, dict[str, Any]]:
    """Archive every result-affecting profiler knob under the selected ID."""
    selected: set[str] = set()
    for profile_set in config.profile_sets:
        selected.update(DEFAULT_PROFILE_SETS[profile_set])
    values: dict[str, dict[str, Any]] = {}
    if "scheduler-concurrency" in selected:
        values["scheduler-concurrency"] = {
            "backend": config.scheduler_backend,
            "bcc_buffer_pages": config.scheduler_bcc_buffer_pages,
            "perf_binary": config.scheduler_perf_binary,
            "perf_mmap_pages": config.scheduler_perf_mmap_pages,
            "cpu_tolerance_fraction": config.scheduler_cpu_tolerance_fraction,
            "cpu_tolerance_sec": config.scheduler_cpu_tolerance_sec,
        }
    if "hardware-perf" in selected:
        values["hardware-perf"] = {
            "perf_binary": config.hardware_perf_binary,
            "events": list(config.hardware_perf_events),
            "minimum_running_ratio": config.hardware_perf_minimum_running_ratio,
            "attach_timeout_sec": config.hardware_perf_attach_timeout_sec,
        }
    if "cgroup-cpu" in selected:
        values["cgroup-cpu"] = {
            "method": "cgroup-v2-cpu.stat",
            "cpuset_cpus": config.cpuset_cpus,
        }
    if "cgroup-memory" in selected:
        values["cgroup-memory"] = {
            "sample_interval_sec": config.memory_sample_interval_sec,
        }
    if "process-exec" in selected:
        values["process-exec"] = {
            "method": "strace-f-process-chdir",
            "string_limit": config.string_limit,
            "timestamp_clock": "CLOCK_REALTIME",
            "timestamp_resolution_ns": 1_000,
        }
    return values


def _attach_verification_evidence(coordination: Any, output_dir: Path) -> Any:
    """Hash post-measurement verifier outputs into the direct evidence bundle."""
    runtime = _plain_json(coordination.runtime_report)
    verification = runtime.get("verification")
    if not isinstance(verification, dict) or not verification.get("requested"):
        return coordination
    descriptors = verification.get("artifacts")
    if not isinstance(descriptors, dict):
        raise CoordinatorValidationError("workload verification report has no artifact descriptors")
    media_types = {
        "stdout": "text/plain",
        "stderr": "text/plain",
        "exit_code": "text/plain",
        "status": "text/plain",
    }
    artifacts: list[RawArtifact] = []
    artifact_ids: list[str] = []
    root = output_dir.resolve()
    for name in ("stdout", "stderr", "exit_code", "status"):
        descriptor = descriptors.get(name)
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
            raise CoordinatorValidationError(f"workload verification {name} artifact is malformed")
        path = Path(descriptor["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CoordinatorValidationError(
                f"workload verification {name} artifact is outside the output tree"
            ) from exc
        artifact_id = f"workload-verification-{name}"
        artifact = RawArtifact.from_path(
            artifact_id=artifact_id,
            path=path,
            base_dir=root,
            media_type=media_types[name],
            provenance={
                "phase": "post-measurement-verification",
                "measured_counters_include_artifact_work": False,
            },
        )
        if descriptor.get("sha256") != artifact.sha256 or descriptor.get("size_bytes") != artifact.size_bytes:
            raise CoordinatorValidationError(f"workload verification {name} artifact changed before bundling")
        artifacts.append(artifact)
        artifact_ids.append(artifact_id)
    existing_ids = {artifact.artifact_id for artifact in coordination.evidence_bundle.artifacts}
    if existing_ids & set(artifact_ids):
        raise CoordinatorValidationError("workload verification artifact IDs collide with profiler artifacts")
    verification["raw_artifact_ids"] = artifact_ids
    runtime["verification"] = verification
    bundle = coordination.evidence_bundle
    enriched_bundle = replace(
        bundle,
        artifacts=(*bundle.artifacts, *artifacts),
        provenance={
            **_plain_json(bundle.provenance),
            "workload_verification": {
                "requested": True,
                "status": verification.get("status"),
                "valid": verification.get("valid"),
                "exit_code": verification.get("exit_code"),
                "command_sha256": verification.get("command_sha256"),
                "raw_artifact_ids": artifact_ids,
                "measurement_boundary": "after-profiler-stop",
                "measured_counters_include_verification": False,
                "lifecycle_markers": [
                    "verification_start",
                    "verification_end",
                ],
            },
        },
        extensions={
            **_plain_json(bundle.extensions),
            "runtime_report": runtime,
            "workload_verification": verification,
        },
    )
    return replace(
        coordination,
        runtime_report=runtime,
        evidence_bundle=enriched_bundle,
    )


def _add_canonical_provenance(coordination: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Bind coordinator timestamps to the host time namespace explicitly."""
    reference = monotonic_clock_reference()
    timeline = coordination.timeline
    lifecycle = {
        "clock_id": timeline.clock_id,
        "clock_name": timeline.clock_name,
        "origin_ns": timeline.origin_ns,
        "start_ns": timeline.start_ns,
        "end_ns": timeline.end_ns,
        "lifecycle_markers": [
            {
                "name": marker.name,
                "canonical_timestamp_ns": marker.canonical_timestamp_ns,
                "alignment_error_ns": marker.alignment_error_ns,
                "provenance": dict(marker.provenance),
            }
            for marker in timeline.lifecycle_markers
        ],
        "clock_reference": reference,
    }
    namespace = reference.get("time_namespace")
    boot_id = reference.get("boot_id")
    namespace_identity = (
        f"{namespace}|boot:{boot_id}" if isinstance(namespace, str) and isinstance(boot_id, str) else None
    )
    canonical_clock = ClockReference(
        clock_id=CANONICAL_CLOCK_ID,
        clock_name="CLOCK_MONOTONIC",
        resolution_ns=int(reference.get("storage_precision_ns") or 1),
        namespace_identity=namespace_identity,
        provenance=reference,
    )
    bundle = coordination.evidence_bundle
    profiler_statuses = {manifest.profiler_id: manifest.status for manifest in coordination.manifests}
    if any(status in {"invalid", "missing", "dropped"} for status in profiler_statuses.values()):
        profiler_evidence_status = "invalid"
    elif any(status == "unalignable" for status in profiler_statuses.values()):
        profiler_evidence_status = "partial"
    else:
        profiler_evidence_status = "valid"
    verification = bundle.extensions.get("workload_verification")
    verification_failed = (
        isinstance(verification, Mapping)
        and verification.get("requested") is True
        and verification.get("status") != "passed"
    )
    evidence_status = "invalid" if verification_failed else profiler_evidence_status
    evidence_invalid_reasons = (
        list(verification.get("reasons") or []) if verification_failed and isinstance(verification, Mapping) else []
    )
    clocks = tuple(canonical_clock if clock.clock_id == CANONICAL_CLOCK_ID else clock for clock in bundle.clocks)
    enriched_bundle = replace(
        bundle,
        clocks=clocks,
        provenance={
            **_plain_json(bundle.provenance),
            "canonical_clock_reference": reference,
            "canonical_lifecycle": lifecycle,
        },
        extensions={
            **_plain_json(bundle.extensions),
            "evidence_status": evidence_status,
            "evidence_invalid_reasons": evidence_invalid_reasons,
            "all_selected_profilers_valid": profiler_evidence_status == "valid",
            "all_selected_profilers_usable": profiler_evidence_status != "invalid",
            "workload_verification_valid": (
                verification.get("valid")
                if isinstance(verification, Mapping) and verification.get("requested") is True
                else None
            ),
            "profiler_statuses": profiler_statuses,
        },
    )
    return (
        replace(coordination, evidence_bundle=enriched_bundle),
        reference,
        lifecycle,
    )


def run_coordinated_docker_profile(
    config: Any,
    *,
    image_identity: Mapping[str, Any],
    run_id: str,
    cancel_requested: Callable[[], bool] | None = None,
) -> CoordinatedProfileResult:
    """Run one profiled or explicitly unprofiled coordinated container."""
    from . import container as legacy

    effective = coordinated_effective_config(config)
    session = CoordinatedDockerSession(effective)
    registry = build_profiler_registry(session)
    controller_source_digest = legacy.controller_source_sha256()
    default_workload_identity = {
        "kind": "container-command",
        "command_sha256": hashlib.sha256(config.command.encode()).hexdigest(),
    }
    supplied_source_identity = (
        dict(config.source_identity) if config.source_identity is not None else {"kind": "tasktrace-container-profile"}
    )
    # Adapter metadata describes the trajectory/trace.  It must augment, not
    # replace, the implementation content identity.  Override a caller value
    # so the authoritative digest can never be supplied by untrusted config.
    supplied_source_identity["controller_source_sha256"] = controller_source_digest
    plan = ExecutionPlan(
        workload_identity=_secret_safe_identity(config.workload_identity or default_workload_identity),
        source_identity=_secret_safe_identity(supplied_source_identity),
        command=(config.shell, "-c", config.command),
        cwd=config.workdir or "/",
        mounts=_mounts(config),
        image_reference=str(config.image),
        image_identity=dict(image_identity),
        runtime="docker",
        # Keep the legacy projection stable while the typed policy in
        # ``extensions`` carries the complete, portable behavior.  In
        # particular, typed ``no-network`` must not look like public mode to
        # execution-plan identity checks.
        network=legacy._legacy_network_projection(config),
        expected_outcome=_secret_safe_identity(config.expected_outcome or {"kind": "observe-command-exit"}),
        environment=_environment(config.env),
        extensions={
            "network_policy": legacy._network_policy_comparability(config),
            "cpuset_cpus": config.cpuset_cpus,
            "cpu_limit": config.cpu_limit,
            "memory_limit_bytes": config.memory_limit_bytes,
            "memory_swap_total_limit_bytes": config.memory_swap_total_limit_bytes,
            "prepare_command_sha256": (
                hashlib.sha256(config.prepare_command.encode()).hexdigest()
                if config.prepare_command is not None
                else None
            ),
            "verification_command_sha256": (
                hashlib.sha256(config.verification_command.encode()).hexdigest()
                if config.verification_command is not None
                else None
            ),
            "user": config.user,
            "terminate_descendants_at_finish": config.terminate_descendants_at_finish,
        },
    )
    request = ProfileRequest(
        mode=("baseline" if config.coordinated_baseline else "profile"),
        profile_sets=tuple(config.profile_sets),
        configuration=_profile_configuration(effective),
    )
    coordinator = ProfileCoordinator(
        registry=registry,
        session_factory=lambda: session,
        run_id_factory=lambda: run_id,
        cancel_requested=cancel_requested,
    )
    try:
        coordination = coordinator.run(plan, request)
    except ProfileRunError as exc:
        if exc.result is not None:
            partial, _reference, _lifecycle = _add_canonical_provenance(exc.result)
            partial_path = config.output_dir / "evidence_bundle.json"
            write_bundle(partial.evidence_bundle, partial_path)
            raise legacy.ProfileError(
                f"coordinated profile failed closed; partial evidence: {partial_path}: {exc}"
            ) from exc
        raise legacy.ProfileError(f"coordinated profile failed before evidence: {exc}") from exc
    except CoordinatorValidationError as exc:
        raise legacy.ProfileError(f"invalid coordinated profile request: {exc}") from exc
    coordination = _attach_verification_evidence(coordination, config.output_dir)
    coordination, canonical_reference, canonical_lifecycle = _add_canonical_provenance(coordination)
    bundle_path = config.output_dir / "evidence_bundle.json"
    write_bundle(coordination.evidence_bundle, bundle_path)
    runtime = dict(coordination.runtime_report)
    completed = subprocess.CompletedProcess(
        session.command,
        int(runtime.get("returncode") or 0),
        str(runtime.get("stdout") or ""),
        str(runtime.get("stderr") or ""),
    )

    def retained_raw_report(profiler_id: str) -> dict[str, Any] | None:
        value = coordination.raw_reports.get(profiler_id)
        return _plain_json(value) if isinstance(value, Mapping) else None

    return CoordinatedProfileResult(
        completed=completed,
        effective_config=effective,
        coordination=coordination,
        bundle_path=bundle_path,
        bundle_sha256=bundle_sha256(coordination.evidence_bundle),
        cpu_report=retained_raw_report("cgroup-cpu") or {},
        scheduler_report=retained_raw_report("scheduler-concurrency"),
        memory_report=retained_raw_report("cgroup-memory"),
        hardware_report=retained_raw_report("hardware-perf"),
        canonical_clock_reference=canonical_reference,
        canonical_lifecycle=canonical_lifecycle,
    )

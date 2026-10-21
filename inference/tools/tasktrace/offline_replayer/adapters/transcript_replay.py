#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract recorded tool calls from transcript JSONL and replay CPU-side work.

This is intentionally a standalone prototype. It supports the common file and
shell tools visible in recorded agent transcripts: ``exec``, ``read``, and
``write``. Unsupported tools are retained in the output report and skipped
unless ``--strict`` is requested. Replay executes recorded actions, not model
inference, so its metrics describe task-side tool work only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..timebase import monotonic_clock_reference

TOOL_MARKER = re.compile(r"\[Tool Call:\s*([^\]]+)\]")
SUPPORTED_TOOLS = frozenset({"exec", "read", "write"})
PREVIEW_LENGTH = 1000


@dataclass(frozen=True)
class Extraction:
    """Extracted tool actions and any lossy-import warnings."""

    task_id: str
    trial_name: str | None
    actions: list[dict[str, Any]]
    warnings: list[str]


class ReplayError(RuntimeError):
    """Raised for an action that cannot safely be replayed."""


def _read_cpu_stat(path: Path) -> dict[str, int]:
    """Read cgroup v2 cpu.stat counters."""
    stats: dict[str, int] = {}
    for line in (path / "cpu.stat").read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            stats[fields[0]] = int(fields[1])
    return stats


class CgroupCpuMeter:
    """Place this process and replay descendants into a cgroup v2 accounting scope.

    Child-process resource accounting alone can miss descendants that outlive a
    shell wrapper. Cgroup counters instead attribute all descendants in the
    bounded replay scope, which is the number needed for task CPU comparison.
    """

    def __init__(self) -> None:
        self.parent_path = self._current_cgroup_path()
        self.path = self.parent_path / f"offline-replayer-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.active = False
        self.baseline_stats: dict[str, int] | None = None
        self.allowed_cpu_ids: list[int] | None = None

    @staticmethod
    def _current_cgroup_path() -> Path:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative_path = line.split("::", 1)[1].lstrip("/")
                return Path("/sys/fs/cgroup") / relative_path
        raise ReplayError("cgroup v2 membership not found in /proc/self/cgroup")

    def start(self) -> None:
        try:
            self.path.mkdir()
            (self.path / "cgroup.procs").write_text(str(os.getpid()))
        except OSError as exc:
            try:
                self.path.rmdir()
            except OSError:
                pass
            raise ReplayError(f"unable to create replay CPU cgroup: {exc}") from exc
        self.active = True

    def begin_workload(self) -> None:
        if not self.active:
            raise ReplayError("cgroup CPU meter was not started")
        self.allowed_cpu_ids = sorted(os.sched_getaffinity(0))
        self.baseline_stats = _read_cpu_stat(self.path)

    def finish(self, wall_time_sec: float) -> dict[str, Any]:
        if not self.active or self.baseline_stats is None:
            raise ReplayError("cgroup CPU meter workload window was not started")
        final_stats = _read_cpu_stat(self.path)
        stats = {key: final_stats.get(key, 0) - self.baseline_stats.get(key, 0) for key in final_stats}
        try:
            (self.parent_path / "cgroup.procs").write_text(str(os.getpid()))
        except OSError as exc:
            raise ReplayError(f"unable to leave replay CPU cgroup: {exc}") from exc
        remaining_pids = [int(pid) for pid in (self.path / "cgroup.procs").read_text().split() if pid.isdigit()]
        cpuset_path = self.path / "cpuset.cpus.effective"
        allowed_cpu_list = cpuset_path.read_text().strip() if cpuset_path.exists() else None
        retained = bool(remaining_pids)
        if not retained:
            try:
                self.path.rmdir()
            except OSError:
                retained = True
        usage_sec = stats.get("usage_usec", 0) / 1_000_000
        result: dict[str, Any] = {
            "method": "cgroup-v2-cpu.stat",
            "scope_path": str(self.path),
            "usage_sec": round(usage_sec, 9),
            "user_sec": round(stats.get("user_usec", 0) / 1_000_000, 9),
            "system_sec": round(stats.get("system_usec", 0) / 1_000_000, 9),
            "average_logical_cores_used": (round(usage_sec / wall_time_sec, 9) if wall_time_sec > 0 else None),
            "logical_core_percent": (round(100 * usage_sec / wall_time_sec, 6) if wall_time_sec > 0 else None),
            "allowed_logical_cpu_list": allowed_cpu_list,
            "allowed_logical_cpu_count": (len(self.allowed_cpu_ids) if self.allowed_cpu_ids is not None else None),
            "allowed_logical_cpu_ids": self.allowed_cpu_ids,
            "allowed_logical_cpu_source": "sched_getaffinity",
            "scope_retained": retained,
            "remaining_pids": remaining_pids,
            "meaning": (
                "Exact accumulated CPU time for work in this cgroup during the "
                "bounded replay window; it does not identify per-CPU residency."
            ),
        }
        if retained:
            result["warning"] = (
                "The replay left processes in the CPU cgroup; reported CPU time "
                "does not include work that occurs after replay completion."
            )
        self.active = False
        return result


def _json_object_after(text: str, position: int) -> dict[str, Any]:
    """Decode the first JSON object following a tool-call marker."""
    while position < len(text) and text[position].isspace():
        position += 1
    payload, _ = json.JSONDecoder().raw_decode(text, position)
    if not isinstance(payload, dict):
        raise ValueError("tool-call payload is not a JSON object")
    return payload


def _available_task_ids(records: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(record["task_id"])
            for record in records
            if record.get("task_id") is not None
            and isinstance(record.get("answer"), str)
            and TOOL_MARKER.search(str(record["answer"]))
        }
    )


def extract_actions(
    transcript_path: Path,
    task_id: str | None = None,
    trial_name: str | None = None,
) -> Extraction:
    """Extract tool calls from transcript answer fields in source order."""
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line_number, line in enumerate(transcript_path.read_text(errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"line {line_number}: invalid JSON skipped ({exc.msg})")
            continue
        if isinstance(record, dict):
            record["_line_number"] = line_number
            records.append(record)

    available = _available_task_ids(records)
    if task_id is None:
        if len(available) != 1:
            raise ReplayError(
                "provide --task-id because the transcript contains tool calls for "
                f"{len(available)} tasks: {', '.join(available) or '(none)'}"
            )
        task_id = available[0]
    elif task_id not in available:
        raise ReplayError(
            f"task {task_id!r} has no extractable tool calls; available: " f"{', '.join(available) or '(none)'}"
        )

    available_trials = sorted(
        {
            str(record["trial_name"])
            for record in records
            if record.get("task_id") == task_id
            and record.get("trial_name") is not None
            and isinstance(record.get("answer"), str)
            and TOOL_MARKER.search(str(record["answer"]))
        }
    )
    if trial_name is not None and trial_name not in available_trials:
        raise ReplayError(
            f"trial {trial_name!r} has no extractable tool calls for task {task_id!r}; "
            f"available: {', '.join(available_trials) or '(none)'}"
        )

    actions: list[dict[str, Any]] = []
    for record in records:
        if record.get("task_id") != task_id or not isinstance(record.get("answer"), str):
            continue
        # Datasets commonly contain multiple attempts of the same task. Mixing
        # them would replay several trajectories as if they were one workload.
        if trial_name is not None and record.get("trial_name") != trial_name:
            continue
        answer = str(record["answer"])
        for marker in TOOL_MARKER.finditer(answer):
            raw_tool = marker.group(1).strip().lower()
            tool = {"shell": "exec", "bash": "exec"}.get(raw_tool, raw_tool)
            try:
                payload = _json_object_after(answer, marker.end())
            except (ValueError, json.JSONDecodeError) as exc:
                warnings.append(f"line {record['_line_number']}: failed to parse {raw_tool!r} " f"tool payload ({exc})")
                continue
            actions.append(
                {
                    "action_index": len(actions) + 1,
                    "tool": tool,
                    "source_tool": raw_tool,
                    "payload": payload,
                    "source_line": record["_line_number"],
                    "turn": record.get("turn"),
                    "trial_name": record.get("trial_name"),
                    "supported": tool in SUPPORTED_TOOLS,
                }
            )

    if not actions:
        raise ReplayError(f"no tool calls extracted for task {task_id!r}")
    return Extraction(
        task_id=task_id,
        trial_name=trial_name,
        actions=actions,
        warnings=warnings,
    )


def _map_workspace_path(
    captured_path: str,
    *,
    workspace: Path,
    source_workspace: Path | None,
) -> Path:
    """Map captured file-tool paths into the fresh replay workspace.

    Captured tasks frequently embed an absolute temporary workspace path.
    Replaying that literal path would leak writes outside the isolated run, so
    only paths under the captured workspace root may be remapped.
    """
    raw_path = Path(captured_path)
    if raw_path.is_absolute():
        relative: Path | None = None
        if source_workspace is not None:
            try:
                relative = raw_path.relative_to(source_workspace)
            except ValueError:
                relative = None
        if relative is None and "agent_workspace" in raw_path.parts:
            marker_index = len(raw_path.parts) - 1 - raw_path.parts[::-1].index("agent_workspace")
            relative = Path(*raw_path.parts[marker_index + 1 :])
        if relative is None:
            raise ReplayError("absolute tool path is outside the captured agent workspace: " f"{captured_path}")
    else:
        relative = raw_path

    target = (workspace / relative).resolve()
    try:
        target.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ReplayError(f"tool path escapes replay workspace: {captured_path}") from exc
    return target


def _remap_command(command: str, *, workspace: Path, source_workspace: Path | None) -> str:
    """Best-effort remap of the captured workspace root in shell text.

    Shell command strings cannot be safely parsed into filesystem operations in
    general. Restricting replacement to the explicit captured root keeps this
    predictable while covering commands emitted by the benchmark trace.
    """
    if source_workspace is None:
        return command
    return command.replace(str(source_workspace), str(workspace))


def _cpu_usage() -> tuple[float, float]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + children.ru_utime, own.ru_stime + children.ru_stime


def _preview(value: str | bytes) -> str:
    # TimeoutExpired may return bytes even for text subprocesses; binary log
    # payloads can also occur in security tasks. Reports must remain writable.
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[:PREVIEW_LENGTH]


def _run_action(
    action: dict[str, Any],
    *,
    workspace: Path,
    source_workspace: Path | None,
    timeout_sec: float,
) -> dict[str, Any]:
    tool = str(action["tool"])
    payload = action["payload"]
    if not isinstance(payload, dict):
        raise ReplayError("tool payload is not a mapping")

    if tool == "exec":
        command = payload.get("command")
        if not isinstance(command, str):
            raise ReplayError("exec tool payload is missing string field 'command'")
        command = _remap_command(command, workspace=workspace, source_workspace=source_workspace)
        try:
            # Replacement decoding is required for logs containing exploit or
            # arbitrary binary payload bytes; replay records output previews
            # rather than treating non-UTF-8 task data as a fatal error.
            completed = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=workspace,
                text=True,
                errors="replace",
                capture_output=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "timed_out",
                "command": command,
                "stdout_preview": _preview(exc.stdout or ""),
                "stderr_preview": _preview(exc.stderr or ""),
            }
        return {
            "status": "ok" if completed.returncode == 0 else "failed",
            "command": command,
            "returncode": completed.returncode,
            "stdout_preview": _preview(completed.stdout),
            "stderr_preview": _preview(completed.stderr),
        }

    if tool == "write":
        captured_path = payload.get("path")
        content = payload.get("content")
        if not isinstance(captured_path, str) or not isinstance(content, str):
            raise ReplayError("write tool payload requires string 'path' and 'content'")
        target = _map_workspace_path(captured_path, workspace=workspace, source_workspace=source_workspace)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {
            "status": "ok",
            "path": str(target.relative_to(workspace)),
            "bytes_written": len(content.encode()),
        }

    if tool == "read":
        captured_path = payload.get("path")
        if not isinstance(captured_path, str):
            raise ReplayError("read tool payload requires string field 'path'")
        target = _map_workspace_path(captured_path, workspace=workspace, source_workspace=source_workspace)
        offset = payload.get("offset")
        limit = payload.get("limit")
        if offset is not None and (not isinstance(offset, int) or isinstance(offset, bool) or offset < 1):
            raise ReplayError("read tool payload 'offset' must be a positive integer")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ReplayError("read tool payload 'limit' must be a positive integer")
        if offset is None and limit is None:
            data = target.read_bytes()
        else:
            # The captured read tool expresses windows in lines. Honoring the
            # window avoids charging full-file I/O to tasks that only inspected
            # a preview of a large CSV or log.
            first_line = offset or 1
            stop_line = None if limit is None else first_line - 1 + limit
            with target.open("rb") as source:
                data = b"".join(itertools.islice(source, first_line - 1, stop_line))
        result = {
            "status": "ok",
            "path": str(target.relative_to(workspace)),
            "bytes_read": len(data),
        }
        if offset is not None:
            result["offset"] = offset
        if limit is not None:
            result["limit"] = limit
        return result

    return {"status": "unsupported", "reason": f"unsupported tool: {tool}"}


def replay_actions(
    extraction: Extraction,
    *,
    workspace: Path,
    source_workspace: Path | None = None,
    initial_workspace: Path | None = None,
    timeout_sec: float = 60.0,
    strict: bool = False,
    cgroup_cpu: bool = False,
    cgroup_ready_file: Path | None = None,
    cgroup_start_delay_sec: float = 0.0,
) -> dict[str, Any]:
    """Replay extracted typed actions and return timing/output metadata."""
    if workspace.exists() and any(workspace.iterdir()):
        raise ReplayError(f"replay workspace must be empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    if initial_workspace is not None:
        if not initial_workspace.is_dir():
            raise ReplayError(f"initial workspace does not exist: {initial_workspace}")
        # Seed files represent task inputs and must exist before action timing
        # begins; copying them would otherwise distort the replay cost.
        shutil.copytree(initial_workspace, workspace, dirs_exist_ok=True)

    cgroup_meter = CgroupCpuMeter() if cgroup_cpu else None
    if cgroup_meter is not None:
        cgroup_meter.start()
        if cgroup_ready_file is not None:
            cgroup_ready_file.write_text(str(cgroup_meter.path) + "\n")
        if cgroup_start_delay_sec > 0:
            time.sleep(cgroup_start_delay_sec)
        cgroup_meter.begin_workload()
    total_wall_before = time.perf_counter()
    total_user_before, total_system_before = _cpu_usage()
    results: list[dict[str, Any]] = []
    try:
        for action in extraction.actions:
            user_before, system_before = _cpu_usage()
            wall_before = time.perf_counter()
            action_start_monotonic_ns = time.monotonic_ns()
            try:
                outcome = _run_action(
                    action,
                    workspace=workspace,
                    source_workspace=source_workspace,
                    timeout_sec=timeout_sec,
                )
            except (OSError, ReplayError) as exc:
                outcome = {"status": "error", "error": str(exc)}
            wall_sec = time.perf_counter() - wall_before
            action_end_monotonic_ns = time.monotonic_ns()
            user_after, system_after = _cpu_usage()
            result = {
                **action,
                **outcome,
                "wall_time_sec": round(wall_sec, 9),
                "replay_start_monotonic_ns": action_start_monotonic_ns,
                "replay_end_monotonic_ns": action_end_monotonic_ns,
                "user_cpu_sec": round(user_after - user_before, 9),
                "system_cpu_sec": round(system_after - system_before, 9),
                "cpu_time_sec": round((user_after - user_before) + (system_after - system_before), 9),
            }
            results.append(result)
            if strict and result["status"] not in {"ok"}:
                break
    finally:
        total_wall_sec = time.perf_counter() - total_wall_before
        total_user_after, total_system_after = _cpu_usage()
        cgroup_measurement = cgroup_meter.finish(total_wall_sec) if cgroup_meter is not None else None
    return {
        "schema_version": "offline-tool-replay.v1",
        "task_id": extraction.task_id,
        "trial_name": extraction.trial_name,
        "workspace": str(workspace),
        "initial_workspace": str(initial_workspace) if initial_workspace else None,
        "source_workspace": str(source_workspace) if source_workspace else None,
        "extraction_warnings": extraction.warnings,
        "clock_reference": monotonic_clock_reference(),
        "actions": results,
        "summary": {
            "actions_extracted": len(extraction.actions),
            "actions_replayed": len(results),
            "actions_ok": sum(result["status"] == "ok" for result in results),
            "actions_unsupported": sum(result["status"] == "unsupported" for result in results),
            "actions_failed": sum(result["status"] in {"failed", "error", "timed_out"} for result in results),
            "wall_time_sec": round(total_wall_sec, 9),
            "user_cpu_sec": round(total_user_after - total_user_before, 9),
            "system_cpu_sec": round(total_system_after - total_system_before, 9),
            "cpu_time_sec": round(
                (total_user_after - total_user_before) + (total_system_after - total_system_before),
                9,
            ),
            "cgroup_cpu": cgroup_measurement,
        },
    }


def _write_extracted_actions(path: Path, extraction: Extraction) -> None:
    with path.open("w") as output:
        output.write(
            json.dumps(
                {
                    "record_type": "trace",
                    "schema_version": "offline-tool-replay.v1",
                    "task_id": extraction.task_id,
                    "trial_name": extraction.trial_name,
                    "warnings": extraction.warnings,
                },
                sort_keys=True,
            )
            + "\n"
        )
        for action in extraction.actions:
            output.write(json.dumps({"record_type": "action", **action}, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract transcript tool calls and replay supported CPU-side actions.")
    parser.add_argument("transcript", type=Path, help="Transcript JSONL to import.")
    parser.add_argument("--task-id", help="Task ID to replay when input includes multiple tasks.")
    parser.add_argument(
        "--trial-name",
        help="Optional trial name to replay a single recorded run for the selected task.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="New or empty directory used as the replayed agent workspace.",
    )
    parser.add_argument(
        "--initial-workspace",
        type=Path,
        help="Optional initial workspace contents copied before executing actions.",
    )
    parser.add_argument(
        "--source-workspace",
        type=Path,
        help="Captured workspace prefix to remap into --workspace.",
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-exec timeout in seconds.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on failed, timed-out, errored, or unsupported actions.",
    )
    parser.add_argument(
        "--cgroup-cpu",
        action="store_true",
        help="Measure task-scoped aggregate CPU usage with a temporary cgroup v2 scope.",
    )
    parser.add_argument(
        "--cgroup-ready-file",
        type=Path,
        help="When using --cgroup-cpu, write the temporary scope path before actions run.",
    )
    parser.add_argument(
        "--cgroup-start-delay-sec",
        type=float,
        default=0.0,
        help="When using --cgroup-cpu, wait after creating the scope so an external probe can attach.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for extracted_actions.jsonl and replay_report.json.",
    )
    args = parser.parse_args(argv)
    if (args.cgroup_ready_file is not None or args.cgroup_start_delay_sec) and not args.cgroup_cpu:
        parser.error("--cgroup-ready-file and --cgroup-start-delay-sec require --cgroup-cpu")
    if args.cgroup_start_delay_sec < 0:
        parser.error("--cgroup-start-delay-sec must not be negative")
    output_dir = args.output_dir or args.workspace.parent / f"{args.workspace.name}-output"
    try:
        extraction = extract_actions(
            args.transcript,
            task_id=args.task_id,
            trial_name=args.trial_name,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_extracted_actions(output_dir / "extracted_actions.jsonl", extraction)
        report = replay_actions(
            extraction,
            workspace=args.workspace,
            source_workspace=args.source_workspace,
            initial_workspace=args.initial_workspace,
            timeout_sec=args.timeout_sec,
            strict=args.strict,
            cgroup_cpu=args.cgroup_cpu,
            cgroup_ready_file=args.cgroup_ready_file,
            cgroup_start_delay_sec=args.cgroup_start_delay_sec,
        )
    except (OSError, ReplayError) as exc:
        parser.error(str(exc))
    report_path = output_dir / "replay_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = report["summary"]
    print(f"Task: {report['task_id']}")
    if report["trial_name"] is not None:
        print(f"Trial: {report['trial_name']}")
    print(
        "Actions: "
        f"{summary['actions_replayed']}/{summary['actions_extracted']} replayed, "
        f"{summary['actions_ok']} ok, "
        f"{summary['actions_unsupported']} unsupported, "
        f"{summary['actions_failed']} failed"
    )
    print(f"CPU time: {summary['cpu_time_sec']:.9f} sec")
    print(f"Wall time: {summary['wall_time_sec']:.9f} sec")
    if summary["cgroup_cpu"] is not None:
        cgroup = summary["cgroup_cpu"]
        print(
            "Cgroup CPU: "
            f"{cgroup['usage_sec']:.9f} sec, "
            f"average logical cores used: {cgroup['average_logical_cores_used']:.6f}"
        )
        if cgroup.get("warning"):
            print(f"Warning: {cgroup['warning']}")
    print(f"Report: {report_path}")
    if args.strict and (summary["actions_unsupported"] or summary["actions_failed"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

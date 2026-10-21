# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed process-lifecycle evidence from a scoped ``strace`` log.

The supported acquisition format is::

    strace -f -ttt -e trace=%process,chdir,fchdir ...

``-ttt`` timestamps are decimal Unix-epoch seconds.  They are converted with
``Decimal`` so the report never introduces binary floating-point rounding.
The parser intentionally understands only the lifecycle facts needed by the
software-deep profile.  Missing, truncated, ambiguous, or escaped lifecycle
evidence remains visible as an anomaly and makes the report invalid.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

PROCESS_EXEC_SCHEMA_VERSION = "software-deep-process-exec.v1"
PROCESS_EXEC_METHOD = "strace-f-ttt-process-chdir"

_LINE_RE = re.compile(r"^\s*(?P<pid>[0-9]+)\s+(?P<timestamp>[0-9]+(?:\.[0-9]+)?)\s+(?P<body>.*)$")
_SYSCALL_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<arguments>.*)\)\s+=\s+(?P<result>.+)$")
_UNFINISHED_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<prefix>.*)<unfinished \.\.\.>$")
_RESUMED_RE = re.compile(r"^<\.\.\. (?P<name>[A-Za-z_][A-Za-z0-9_]*) resumed>(?P<suffix>.*)$")
_EXIT_RE = re.compile(r"^\+\+\+ exited with (?P<status>-?[0-9]+) \+\+\+$")
_KILLED_RE = re.compile(r"^\+\+\+ killed by (?P<signal>SIG[A-Z0-9]+)(?: \([^)]*\))? \+\+\+$")
_INTEGER_RESULT_RE = re.compile(r"^(-?[0-9]+)(?:\s|$)")
_ERRNO_RE = re.compile(r"^-1\s+(?P<errno>[A-Z][A-Z0-9_]*)\b")

_RELEVANT_SYSCALLS = frozenset({"execve", "execveat", "clone", "clone3", "fork", "vfork", "chdir", "fchdir"})
_CREATION_SYSCALLS = frozenset({"clone", "clone3", "fork", "vfork"})


@dataclass
class _CwdState:
    value: str | None


@dataclass
class _TaskState:
    tid: int
    process_pid: int
    cwd: _CwdState
    first_seen_ns: int
    start_ns: int | None = None
    end_ns: int | None = None
    exit_code: int | None = None
    exit_signal: str | None = None


@dataclass
class _ProcessState:
    pid: int
    parent_pid: int | None
    parent_resolution: str
    start_ns: int
    start_source: str
    first_seen_ns: int
    cwd: _CwdState
    creation_syscall: str | None = None
    creation_completed_ns: int | None = None
    binary: str | None = None
    argv: list[str] | None = None
    exec_cwd: str | None = None
    binary_source: str | None = None
    end_ns: int | None = None
    exit_code: int | None = None
    exit_signal: str | None = None
    execs: list[dict[str, Any]] = field(default_factory=list)
    failed_execs: list[dict[str, Any]] = field(default_factory=list)
    cwd_changes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _PendingSyscall:
    pid: int
    name: str
    prefix: str
    started_ns: int
    line_number: int


class _Parser:
    def __init__(self, *, initial_cwd: str | None, expected_root_pid: int | None):
        self.initial_cwd = _normalize_absolute_cwd(initial_cwd)
        self.expected_root_pid = expected_root_pid
        self.processes: dict[int, _ProcessState] = {}
        self.tasks: dict[int, _TaskState] = {}
        self.pending: dict[tuple[int, str], _PendingSyscall] = {}
        self.anomalies: list[dict[str, Any]] = []
        self.split_syscalls = 0
        self.parsed_line_count = 0
        self.total_line_count = 0

    def anomaly(
        self,
        code: str,
        message: str,
        *,
        line_number: int | None = None,
        pid: int | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        item: dict[str, Any] = {"code": code, "message": message}
        if line_number is not None:
            item["line_number"] = line_number
        if pid is not None:
            item["pid"] = pid
        if timestamp_ns is not None:
            item["timestamp_ns"] = timestamp_ns
        self.anomalies.append(item)

    def _ensure_task(self, tid: int, timestamp_ns: int) -> _TaskState:
        existing = self.tasks.get(tid)
        if existing is not None:
            existing.first_seen_ns = min(existing.first_seen_ns, timestamp_ns)
            process = self.processes[existing.process_pid]
            process.first_seen_ns = min(process.first_seen_ns, timestamp_ns)
            return existing

        root_like = tid == self.expected_root_pid if self.expected_root_pid is not None else not self.processes
        cwd = _CwdState(self.initial_cwd if root_like else None)
        process = _ProcessState(
            pid=tid,
            parent_pid=None,
            parent_resolution="unresolved",
            start_ns=timestamp_ns,
            start_source="first-observed-event",
            first_seen_ns=timestamp_ns,
            cwd=cwd,
        )
        task = _TaskState(
            tid=tid,
            process_pid=tid,
            cwd=cwd,
            first_seen_ns=timestamp_ns,
            start_ns=timestamp_ns,
        )
        self.processes[tid] = process
        self.tasks[tid] = task
        return task

    def parse(self, text: str) -> dict[str, Any]:
        lines = text.splitlines()
        self.total_line_count = len(lines)
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            match = _LINE_RE.match(line)
            if match is None:
                self.anomaly(
                    "malformed_trace_line",
                    "trace line lacks a numeric task ID, decimal timestamp, or body",
                    line_number=line_number,
                )
                continue
            pid = int(match.group("pid"))
            timestamp_ns = _timestamp_ns(match.group("timestamp"))
            if timestamp_ns is None:
                self.anomaly(
                    "invalid_timestamp",
                    f"timestamp is not exactly representable in nanoseconds: {match.group('timestamp')!r}",
                    line_number=line_number,
                    pid=pid,
                )
                continue
            self.parsed_line_count += 1
            self._parse_body(
                pid=pid,
                timestamp_ns=timestamp_ns,
                body=match.group("body"),
                line_number=line_number,
            )

        for pending in sorted(self.pending.values(), key=lambda item: (item.line_number, item.pid, item.name)):
            self.anomaly(
                "unfinished_syscall_without_resume",
                f"{pending.name} syscall was unfinished at end of trace",
                line_number=pending.line_number,
                pid=pending.pid,
                timestamp_ns=pending.started_ns,
            )
        self.pending.clear()
        return self._finish()

    def _parse_body(self, *, pid: int, timestamp_ns: int, body: str, line_number: int) -> None:
        exit_match = _EXIT_RE.match(body)
        if exit_match is not None:
            self._handle_exit(
                pid,
                timestamp_ns,
                exit_code=int(exit_match.group("status")),
                exit_signal=None,
                line_number=line_number,
            )
            return
        killed_match = _KILLED_RE.match(body)
        if killed_match is not None:
            self._handle_exit(
                pid,
                timestamp_ns,
                exit_code=None,
                exit_signal=killed_match.group("signal"),
                line_number=line_number,
            )
            return

        unfinished = _UNFINISHED_RE.match(body)
        if unfinished is not None:
            name = unfinished.group("name")
            if name not in _RELEVANT_SYSCALLS:
                return
            key = (pid, name)
            if key in self.pending:
                self.anomaly(
                    "overlapping_unfinished_syscall",
                    f"a second unfinished {name} appeared before its resume",
                    line_number=line_number,
                    pid=pid,
                    timestamp_ns=timestamp_ns,
                )
                return
            self.pending[key] = _PendingSyscall(
                pid=pid,
                name=name,
                prefix=f"{name}({unfinished.group('prefix')}",
                started_ns=timestamp_ns,
                line_number=line_number,
            )
            return

        resumed = _RESUMED_RE.match(body)
        if resumed is not None:
            name = resumed.group("name")
            if name not in _RELEVANT_SYSCALLS:
                return
            pending = self.pending.pop((pid, name), None)
            if pending is None:
                self.anomaly(
                    "resumed_syscall_without_start",
                    f"{name} resumed without a matching unfinished syscall",
                    line_number=line_number,
                    pid=pid,
                    timestamp_ns=timestamp_ns,
                )
                return
            self.split_syscalls += 1
            # Split records are normal under concurrent strace collection. A
            # uniquely matched pair is losslessly reconstructable and retains
            # both acquisition endpoints. Unmatched, overlapping, or malformed
            # pairs still fail closed in the surrounding branches.
            self._handle_syscall(
                pid=pid,
                started_ns=pending.started_ns,
                completed_ns=timestamp_ns,
                body=pending.prefix + resumed.group("suffix"),
                line_number=pending.line_number,
            )
            return

        syscall = _SYSCALL_RE.match(body)
        if syscall is not None:
            if syscall.group("name") in _RELEVANT_SYSCALLS:
                self._handle_parsed_syscall(
                    pid=pid,
                    started_ns=timestamp_ns,
                    completed_ns=timestamp_ns,
                    name=syscall.group("name"),
                    arguments=syscall.group("arguments"),
                    result=syscall.group("result"),
                    line_number=line_number,
                )
            return

        possible_name = body.split("(", 1)[0]
        if possible_name in _RELEVANT_SYSCALLS:
            self.anomaly(
                "malformed_relevant_syscall",
                f"unable to parse {possible_name} syscall record",
                line_number=line_number,
                pid=pid,
                timestamp_ns=timestamp_ns,
            )
        # Signal-delivery records and unrelated %process syscalls are expected
        # and deliberately ignored.

    def _handle_syscall(
        self,
        *,
        pid: int,
        started_ns: int,
        completed_ns: int,
        body: str,
        line_number: int,
    ) -> None:
        syscall = _SYSCALL_RE.match(body)
        if syscall is None or syscall.group("name") not in _RELEVANT_SYSCALLS:
            self.anomaly(
                "malformed_resumed_syscall",
                "unfinished/resumed records could not be reconstructed",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )
            return
        self._handle_parsed_syscall(
            pid=pid,
            started_ns=started_ns,
            completed_ns=completed_ns,
            name=syscall.group("name"),
            arguments=syscall.group("arguments"),
            result=syscall.group("result"),
            line_number=line_number,
        )

    def _handle_parsed_syscall(
        self,
        *,
        pid: int,
        started_ns: int,
        completed_ns: int,
        name: str,
        arguments: str,
        result: str,
        line_number: int,
    ) -> None:
        if name in ("execve", "execveat"):
            self._handle_exec(
                pid=pid,
                started_ns=started_ns,
                completed_ns=completed_ns,
                name=name,
                arguments=arguments,
                result=result,
                line_number=line_number,
            )
        elif name in _CREATION_SYSCALLS:
            self._handle_creation(
                pid=pid,
                started_ns=started_ns,
                completed_ns=completed_ns,
                name=name,
                arguments=arguments,
                result=result,
                line_number=line_number,
            )
        elif name == "chdir":
            self._handle_chdir(
                pid=pid,
                timestamp_ns=completed_ns,
                arguments=arguments,
                result=result,
                line_number=line_number,
            )
        elif name == "fchdir":
            self._handle_fchdir(
                pid=pid,
                timestamp_ns=completed_ns,
                arguments=arguments,
                result=result,
                line_number=line_number,
            )

    def _handle_exec(
        self,
        *,
        pid: int,
        started_ns: int,
        completed_ns: int,
        name: str,
        arguments: str,
        result: str,
        line_number: int,
    ) -> None:
        task = self._ensure_task(pid, started_ns)
        process = self.processes[task.process_pid]
        if task.tid != task.process_pid:
            self.anomaly(
                "exec_from_nonleader_thread",
                "exec from a non-leader thread cannot be mapped safely without de-thread evidence",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )

        parts = _split_top_level(arguments)
        minimum = 2 if name == "execve" else 3
        if parts is None or len(parts) < minimum:
            self.anomaly(
                "malformed_exec_arguments",
                f"unable to split {name} arguments",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )
            return
        path_index = 0 if name == "execve" else 1
        argv_index = 1 if name == "execve" else 2
        invoked_path = _parse_c_string(parts[path_index])
        argv = _parse_argv(parts[argv_index])
        if invoked_path is None:
            self.anomaly(
                "truncated_or_invalid_exec_path",
                f"{name} path is truncated or is not a complete quoted string",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )
        if argv is None:
            self.anomaly(
                "truncated_or_invalid_argv",
                f"{name} argv is truncated or malformed",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )

        dirfd = parts[0].strip() if name == "execveat" else None
        flags = parts[4].strip() if name == "execveat" and len(parts) >= 5 else None
        binary, binary_resolution = _resolve_exec_path(invoked_path, cwd=task.cwd.value, syscall=name, dirfd=dirfd)
        if invoked_path is not None and binary is None:
            self.anomaly(
                "unresolved_exec_binary",
                f"cannot resolve {name} path {invoked_path!r} against the observed cwd/dirfd",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )
        if task.cwd.value is None:
            self.anomaly(
                "unresolved_exec_cwd",
                "working directory at exec time is unknown",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )

        event: dict[str, Any] = {
            "syscall": name,
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "invoked_path": invoked_path,
            "binary": binary,
            "binary_resolution": binary_resolution,
            "argv": argv,
            "cwd": task.cwd.value,
        }
        if name == "execveat":
            event["dirfd"] = dirfd
            event["flags"] = flags

        result_value = _integer_result(result)
        if result_value == 0:
            process.execs.append(event)
            process.binary = binary
            process.argv = argv
            process.exec_cwd = task.cwd.value
            process.binary_source = "successful-exec"
        elif result.startswith("-1"):
            event["errno"] = _errno(result)
            event["result"] = result
            process.failed_execs.append(event)
        else:
            self.anomaly(
                "ambiguous_exec_result",
                f"cannot determine whether {name} succeeded from result {result!r}",
                line_number=line_number,
                pid=pid,
                timestamp_ns=completed_ns,
            )

    def _handle_creation(
        self,
        *,
        pid: int,
        started_ns: int,
        completed_ns: int,
        name: str,
        arguments: str,
        result: str,
        line_number: int,
    ) -> None:
        caller = self._ensure_task(pid, started_ns)
        child_tid = _integer_result(result)
        if child_tid is None:
            self.anomaly(
                "ambiguous_process_creation_result",
                f"cannot parse {name} result {result!r}",
                line_number=line_number,
                pid=pid,
                timestamp_ns=completed_ns,
            )
            return
        if child_tid < 0:
            return
        if child_tid == 0:
            self.anomaly(
                "child_side_creation_result",
                f"{name} returned zero in a task-prefixed trace; parent relation is unresolved",
                line_number=line_number,
                pid=pid,
                timestamp_ns=completed_ns,
            )
            return

        flags = _clone_flags(name, arguments)
        if name in ("clone", "clone3") and flags is None:
            self.anomaly(
                "unresolved_clone_flags",
                f"cannot determine whether {name} result {child_tid} is a process or thread",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )
        is_thread = flags is not None and "CLONE_THREAD" in flags
        share_cwd = flags is not None and "CLONE_FS" in flags
        caller_process = self.processes[caller.process_pid]

        if is_thread:
            existing = self.tasks.get(child_tid)
            if existing is not None:
                provisional = self.processes.get(existing.process_pid)
                referenced_as_parent = any(process.parent_pid == child_tid for process in self.processes.values())
                if (
                    existing.process_pid == child_tid
                    and provisional is not None
                    and provisional.parent_pid is None
                    and not provisional.execs
                    and not provisional.failed_execs
                    and not provisional.cwd_changes
                    and not referenced_as_parent
                ):
                    # A new task can run (and even exit) before the parent-side
                    # clone syscall resumes.  The completed CLONE_THREAD result
                    # resolves that provisional task as a thread; it is not an
                    # escaped process.  Preserve its observed terminal record.
                    del self.processes[child_tid]
                    if provisional.cwd.value is None:
                        provisional.cwd.value = caller.cwd.value
                    for related_task in self.tasks.values():
                        if related_task.process_pid == child_tid:
                            related_task.process_pid = caller.process_pid
                    existing.cwd = caller.cwd if share_cwd else _CwdState(caller.cwd.value)
                    existing.start_ns = min(existing.start_ns or started_ns, started_ns)
                    existing.first_seen_ns = min(existing.first_seen_ns, started_ns)
                    return
                self.anomaly(
                    "late_or_conflicting_thread_relation",
                    f"thread {child_tid} was observed before or conflicts with its clone relation",
                    line_number=line_number,
                    pid=pid,
                    timestamp_ns=started_ns,
                )
                return
            child_cwd = caller.cwd if share_cwd else _CwdState(caller.cwd.value)
            self.tasks[child_tid] = _TaskState(
                tid=child_tid,
                process_pid=caller.process_pid,
                cwd=child_cwd,
                first_seen_ns=started_ns,
                start_ns=started_ns,
            )
            return

        parent_pid = caller.process_pid
        parent_resolution = "creation-edge"
        if flags is not None and "CLONE_PARENT" in flags:
            if caller_process.parent_pid is None:
                parent_pid = None
                parent_resolution = "unresolved-clone-parent"
                self.anomaly(
                    "unresolved_clone_parent",
                    f"CLONE_PARENT child {child_tid} has no resolvable traced parent",
                    line_number=line_number,
                    pid=pid,
                    timestamp_ns=started_ns,
                )
            else:
                parent_pid = caller_process.parent_pid
                parent_resolution = "clone-parent-edge"

        child_cwd = caller.cwd if share_cwd else _CwdState(caller.cwd.value)
        existing_task = self.tasks.get(child_tid)
        if existing_task is not None and existing_task.process_pid != child_tid:
            self.anomaly(
                "pid_task_collision",
                f"process result {child_tid} collides with a known thread task",
                line_number=line_number,
                pid=pid,
                timestamp_ns=started_ns,
            )
            return
        child = self.processes.get(child_tid)
        if child is None:
            child = _ProcessState(
                pid=child_tid,
                parent_pid=parent_pid,
                parent_resolution=parent_resolution,
                start_ns=started_ns,
                start_source=f"{name}-result",
                first_seen_ns=started_ns,
                cwd=child_cwd,
                creation_syscall=name,
                creation_completed_ns=completed_ns,
                binary=caller_process.binary,
                argv=(list(caller_process.argv) if caller_process.argv is not None else None),
                exec_cwd=caller.cwd.value,
                binary_source=("inherited-from-parent" if caller_process.binary is not None else None),
            )
            self.processes[child_tid] = child
            self.tasks[child_tid] = _TaskState(
                tid=child_tid,
                process_pid=child_tid,
                cwd=child_cwd,
                first_seen_ns=started_ns,
                start_ns=started_ns,
            )
        else:
            if child.parent_pid is not None and child.parent_pid != parent_pid:
                self.anomaly(
                    "conflicting_parent_relation",
                    f"process {child_tid} has conflicting parent relations",
                    line_number=line_number,
                    pid=pid,
                    timestamp_ns=started_ns,
                )
                return
            # A vfork child can run before the parent's syscall resumes.  Keep
            # already observed child facts, but resolve only the relationship
            # and inherited state that the creation record proves.
            child.parent_pid = parent_pid
            child.parent_resolution = parent_resolution
            child.start_ns = min(child.start_ns, started_ns)
            child.start_source = f"{name}-result"
            child.creation_syscall = name
            child.creation_completed_ns = completed_ns
            successful_cwd_change = any(event.get("success") is True for event in child.cwd_changes)
            if (
                name == "vfork"
                and existing_task is not None
                and existing_task.cwd.value is None
                and caller.cwd.value is not None
                and not successful_cwd_change
            ):
                # Under vfork the child normally runs through PATH candidates
                # and a successful exec before the parent-side syscall resumes.
                # Those records were provisionally observed without a parent or
                # cwd.  The completed creation edge now proves the inherited cwd;
                # reconcile only events inside that split syscall's interval.
                repaired_timestamps: set[int] = set()
                for event in (*child.failed_execs, *child.execs):
                    event_started_ns = event.get("started_ns")
                    if (
                        event.get("cwd") is None
                        and isinstance(event_started_ns, int)
                        and started_ns <= event_started_ns <= completed_ns
                    ):
                        event["cwd"] = caller.cwd.value
                        repaired_timestamps.add(event_started_ns)
                if repaired_timestamps:
                    self.anomalies = [
                        anomaly
                        for anomaly in self.anomalies
                        if not (
                            anomaly.get("code") == "unresolved_exec_cwd"
                            and anomaly.get("pid") == child_tid
                            and anomaly.get("timestamp_ns") in repaired_timestamps
                        )
                    ]
                    if child.execs and child.execs[-1].get("cwd") is not None:
                        child.exec_cwd = child.execs[-1]["cwd"]
            if child.binary is None and not child.execs:
                child.binary = caller_process.binary
                child.argv = list(caller_process.argv) if caller_process.argv is not None else None
                child.exec_cwd = caller.cwd.value
                child.binary_source = "inherited-from-parent" if caller_process.binary is not None else None
            if existing_task is not None and existing_task.cwd.value is None and not successful_cwd_change:
                existing_task.cwd = child_cwd
                child.cwd = child_cwd

    def _handle_chdir(
        self,
        *,
        pid: int,
        timestamp_ns: int,
        arguments: str,
        result: str,
        line_number: int,
    ) -> None:
        task = self._ensure_task(pid, timestamp_ns)
        process = self.processes[task.process_pid]
        parts = _split_top_level(arguments)
        path = _parse_c_string(parts[0]) if parts and len(parts) == 1 else None
        if path is None:
            self.anomaly(
                "truncated_or_invalid_chdir_path",
                "chdir path is truncated or malformed",
                line_number=line_number,
                pid=pid,
                timestamp_ns=timestamp_ns,
            )
            return
        result_value = _integer_result(result)
        old_cwd = task.cwd.value
        new_cwd = _resolve_cwd(path, old_cwd)
        event = {
            "syscall": "chdir",
            "timestamp_ns": timestamp_ns,
            "path": path,
            "from_cwd": old_cwd,
            "to_cwd": new_cwd,
            "success": result_value == 0,
        }
        if result_value == 0:
            if new_cwd is None:
                self.anomaly(
                    "unresolved_chdir_cwd",
                    f"cannot resolve relative chdir path {path!r} without a known cwd",
                    line_number=line_number,
                    pid=pid,
                    timestamp_ns=timestamp_ns,
                )
            task.cwd.value = new_cwd
        elif result_value is None:
            self.anomaly(
                "ambiguous_chdir_result",
                f"cannot determine chdir result from {result!r}",
                line_number=line_number,
                pid=pid,
                timestamp_ns=timestamp_ns,
            )
        process.cwd_changes.append(event)

    def _handle_fchdir(
        self,
        *,
        pid: int,
        timestamp_ns: int,
        arguments: str,
        result: str,
        line_number: int,
    ) -> None:
        task = self._ensure_task(pid, timestamp_ns)
        process = self.processes[task.process_pid]
        result_value = _integer_result(result)
        old_cwd = task.cwd.value
        event = {
            "syscall": "fchdir",
            "timestamp_ns": timestamp_ns,
            "fd": arguments.strip(),
            "from_cwd": old_cwd,
            "to_cwd": None,
            "success": result_value == 0,
        }
        if result_value == 0:
            task.cwd.value = None
            self.anomaly(
                "unresolved_fchdir_cwd",
                "successful fchdir cannot be resolved without file-descriptor evidence",
                line_number=line_number,
                pid=pid,
                timestamp_ns=timestamp_ns,
            )
        elif result_value is None:
            self.anomaly(
                "ambiguous_fchdir_result",
                f"cannot determine fchdir result from {result!r}",
                line_number=line_number,
                pid=pid,
                timestamp_ns=timestamp_ns,
            )
        process.cwd_changes.append(event)

    def _handle_exit(
        self,
        pid: int,
        timestamp_ns: int,
        *,
        exit_code: int | None,
        exit_signal: str | None,
        line_number: int,
    ) -> None:
        task = self._ensure_task(pid, timestamp_ns)
        if task.end_ns is not None:
            self.anomaly(
                "duplicate_task_exit",
                f"task {pid} has more than one terminal record",
                line_number=line_number,
                pid=pid,
                timestamp_ns=timestamp_ns,
            )
            return
        task.end_ns = timestamp_ns
        task.exit_code = exit_code
        task.exit_signal = exit_signal
        if task.tid == task.process_pid:
            process = self.processes[task.process_pid]
            process.end_ns = timestamp_ns
            process.exit_code = exit_code
            process.exit_signal = exit_signal

    def _finish(self) -> dict[str, Any]:
        root_pid: int | None = None
        root_inference = "unresolved"
        if self.expected_root_pid is not None:
            if self.expected_root_pid in self.processes:
                root_pid = self.expected_root_pid
                root_inference = "caller-supplied"
            else:
                self.anomaly(
                    "expected_root_not_observed",
                    f"expected root PID {self.expected_root_pid} was not observed",
                    pid=self.expected_root_pid,
                )
        else:
            unparented = sorted(
                (process for process in self.processes.values() if process.parent_pid is None),
                key=lambda process: (process.first_seen_ns, process.pid),
            )
            if unparented:
                root_pid = unparented[0].pid
                root_inference = "earliest-unparented-task"
            else:
                self.anomaly("trace_root_unresolved", "no trace root could be identified")

        if root_pid is not None:
            root = self.processes[root_pid]
            root.parent_pid = None
            root.parent_resolution = "trace-root"

        unresolved_parent_pids: list[int] = []
        for process in sorted(self.processes.values(), key=lambda item: item.pid):
            if process.pid == root_pid:
                continue
            if process.parent_pid is None or process.parent_pid not in self.processes:
                unresolved_parent_pids.append(process.pid)
                self.anomaly(
                    "unresolved_process_parent",
                    f"process {process.pid} has no resolved parent in the trace",
                    pid=process.pid,
                )

        reachable = _reachable_processes(self.processes, root_pid, self.anomaly)
        process_escape_pids = sorted(set(self.processes) - reachable)
        for pid in process_escape_pids:
            self.anomaly(
                "process_escape_or_unscoped_task",
                f"process {pid} is not connected to the trace root",
                pid=pid,
            )

        for process in sorted(self.processes.values(), key=lambda item: item.pid):
            if process.end_ns is None:
                self.anomaly(
                    "unresolved_process_lifecycle",
                    f"process {process.pid} has no terminal exit or signal record",
                    pid=process.pid,
                )
            elif process.end_ns < process.start_ns:
                self.anomaly(
                    "invalid_process_interval",
                    f"process {process.pid} ends before it starts",
                    pid=process.pid,
                )
            if process.binary is None:
                self.anomaly(
                    "unresolved_process_binary",
                    f"process {process.pid} has neither a successful nor inherited executable",
                    pid=process.pid,
                )
            if process.argv is None:
                self.anomaly(
                    "unresolved_process_argv",
                    f"process {process.pid} has no complete argv evidence",
                    pid=process.pid,
                )
            if process.exec_cwd is None:
                self.anomaly(
                    "unresolved_process_cwd",
                    f"process {process.pid} has no resolved exec/inherited cwd",
                    pid=process.pid,
                )

        for task in sorted(self.tasks.values(), key=lambda item: item.tid):
            if task.tid != task.process_pid and task.end_ns is None:
                self.anomaly(
                    "unresolved_thread_lifecycle",
                    f"thread task {task.tid} has no terminal record",
                    pid=task.tid,
                )

        children: dict[int, list[int]] = {pid: [] for pid in self.processes}
        for process in self.processes.values():
            if process.parent_pid in children:
                children[process.parent_pid].append(process.pid)

        process_rows: list[dict[str, Any]] = []
        for process in sorted(self.processes.values(), key=lambda item: item.pid):
            process_rows.append(
                {
                    "pid": process.pid,
                    "parent_pid": process.parent_pid,
                    "parent_resolution": process.parent_resolution,
                    "in_scope": process.pid in reachable,
                    "process_escape": process.pid in process_escape_pids,
                    "children": sorted(children[process.pid]),
                    "creation_syscall": process.creation_syscall,
                    "start_ns": process.start_ns,
                    "start_source": process.start_source,
                    "creation_completed_ns": process.creation_completed_ns,
                    "end_ns": process.end_ns,
                    "exit_code": process.exit_code,
                    "exit_signal": process.exit_signal,
                    "end_reason": (
                        "exit"
                        if process.exit_code is not None
                        else "signal" if process.exit_signal is not None else None
                    ),
                    "binary": process.binary,
                    "argv": process.argv,
                    "cwd": process.exec_cwd,
                    "final_observed_cwd": process.cwd.value,
                    "binary_source": process.binary_source,
                    "execs": process.execs,
                    "failed_execs": process.failed_execs,
                    "cwd_changes": process.cwd_changes,
                }
            )

        thread_rows = [
            {
                "tid": task.tid,
                "process_pid": task.process_pid,
                "start_ns": task.start_ns,
                "end_ns": task.end_ns,
                "exit_code": task.exit_code,
                "exit_signal": task.exit_signal,
            }
            for task in sorted(self.tasks.values(), key=lambda item: item.tid)
            if task.tid != task.process_pid
        ]

        anomalies = sorted(
            self.anomalies,
            key=lambda item: (
                item.get("line_number", 2**63 - 1),
                item.get("timestamp_ns", 2**63 - 1),
                item.get("pid", 2**63 - 1),
                item["code"],
                item["message"],
            ),
        )
        codes = [item["code"] for item in anomalies]
        inventory: dict[str, dict[str, Any]] = {}
        for process in self.processes.values():
            for event in process.execs:
                binary = event.get("binary")
                if not isinstance(binary, str):
                    continue
                item = inventory.setdefault(binary, {"binary": binary, "exec_count": 0, "process_pids": set()})
                item["exec_count"] += 1
                item["process_pids"].add(process.pid)
        binary_inventory = [
            {
                "binary": item["binary"],
                "exec_count": item["exec_count"],
                "process_pids": sorted(item["process_pids"]),
            }
            for item in sorted(inventory.values(), key=lambda item: item["binary"])
        ]
        loss_codes = {
            "invalid_trace_encoding",
            "trace_read_error",
            "malformed_trace_line",
            "invalid_timestamp",
            "overlapping_unfinished_syscall",
            "unfinished_syscall_without_resume",
            "resumed_syscall_without_start",
            "malformed_resumed_syscall",
            "malformed_relevant_syscall",
            "truncated_or_invalid_exec_path",
            "truncated_or_invalid_argv",
            "truncated_or_invalid_chdir_path",
        }
        lifecycle_codes = {
            "trace_root_unresolved",
            "expected_root_not_observed",
            "unresolved_process_parent",
            "process_escape_or_unscoped_task",
            "unresolved_process_lifecycle",
            "invalid_process_interval",
            "unresolved_process_binary",
            "unresolved_process_argv",
            "unresolved_process_cwd",
            "unresolved_thread_lifecycle",
        }
        valid = not anomalies
        return {
            "schema_version": PROCESS_EXEC_SCHEMA_VERSION,
            "method": PROCESS_EXEC_METHOD,
            "source_clock": "realtime",
            "timestamp_encoding": "decimal-epoch-seconds-converted-exactly-to-nanoseconds",
            "scope": {
                "kind": "trace-root-and-descendants",
                "root_pid": root_pid,
                "root_inference": root_inference,
                "process_escape_detected": bool(process_escape_pids),
                "process_escape_pids": process_escape_pids,
                "unresolved_parent_pids": sorted(unresolved_parent_pids),
            },
            "valid": valid,
            "trace_complete": not any(code in loss_codes | lifecycle_codes for code in codes),
            "loss": {
                "detected": any(code in loss_codes for code in codes),
                "split_syscall_count": self.split_syscalls,
                "unfinished_syscall_count": sum(code == "unfinished_syscall_without_resume" for code in codes),
                "resumed_without_start_count": sum(code == "resumed_syscall_without_start" for code in codes),
                "malformed_or_truncated_count": sum(code in loss_codes for code in codes),
                "unresolved_process_count": sum(code == "unresolved_process_lifecycle" for code in codes),
            },
            "summary": {
                "line_count": self.total_line_count,
                "pid_prefixed_line_count": self.parsed_line_count,
                "process_count": len(process_rows),
                "thread_count": len(thread_rows),
                "successful_exec_count": sum(len(process.execs) for process in self.processes.values()),
                "failed_exec_count": sum(len(process.failed_execs) for process in self.processes.values()),
                "anomaly_count": len(anomalies),
            },
            "anomalies": anomalies,
            "binary_inventory": binary_inventory,
            "processes": process_rows,
            "threads": thread_rows,
        }


def _normalize_absolute_cwd(value: str | None) -> str | None:
    if value is None or not value.startswith("/"):
        return None
    return posixpath.normpath(value)


def _timestamp_ns(text: str) -> int | None:
    try:
        nanoseconds = Decimal(text) * Decimal(1_000_000_000)
    except InvalidOperation:
        return None
    if not nanoseconds.is_finite() or nanoseconds != nanoseconds.to_integral_value():
        return None
    value = int(nanoseconds)
    return value if value >= 0 else None


def _integer_result(result: str) -> int | None:
    match = _INTEGER_RESULT_RE.match(result.strip())
    return int(match.group(1)) if match is not None else None


def _errno(result: str) -> str | None:
    match = _ERRNO_RE.match(result.strip())
    return match.group("errno") if match is not None else None


def _split_top_level(text: str) -> list[str] | None:
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    in_comment = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if in_comment:
            if char == "/" and index > 0 and text[index - 1] == "*":
                in_comment = False
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char == "*" and index > 0 and text[index - 1] == "/":
            in_comment = True
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack.pop() != pairs[char]:
                return None
        elif char == "," and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if quote is not None or stack or in_comment:
        return None
    parts.append(text[start:].strip())
    return parts


def _parse_c_string(token: str) -> str | None:
    token = token.strip()
    if not token or token.endswith("...") or not token.startswith(('"', "'")):
        return None
    try:
        value = ast.literal_eval(token)
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, str) else None


def _parse_argv(token: str) -> list[str] | None:
    token = token.strip()
    if not token.startswith("[") or not token.endswith("]"):
        return None
    items = _split_top_level(token[1:-1])
    if items is None:
        return None
    if len(items) == 1 and not items[0]:
        return []
    result: list[str] = []
    for item in items:
        if item.strip() == "...":
            return None
        value = _parse_c_string(item)
        if value is None:
            return None
        result.append(value)
    return result


def _resolve_cwd(path: str, cwd: str | None) -> str | None:
    if path.startswith("/"):
        return posixpath.normpath(path)
    if cwd is None:
        return None
    return posixpath.normpath(posixpath.join(cwd, path))


def _resolve_exec_path(
    path: str | None,
    *,
    cwd: str | None,
    syscall: str,
    dirfd: str | None,
) -> tuple[str | None, str]:
    if path is None:
        return None, "unresolved"
    if path.startswith("/"):
        return posixpath.normpath(path), "absolute"
    if syscall == "execve" or dirfd == "AT_FDCWD":
        resolved = _resolve_cwd(path, cwd)
        return resolved, "cwd-relative" if resolved is not None else "unresolved"
    return None, "unresolved-dirfd"


def _clone_flags(name: str, arguments: str) -> set[str] | None:
    if name not in ("clone", "clone3"):
        return set()
    match = re.search(r"(?:^|[, {])flags=(?P<flags>[^,})]+)", arguments)
    if match is None:
        return None
    return {part.strip() for part in match.group("flags").split("|") if part.strip()}


def _reachable_processes(
    processes: dict[int, _ProcessState],
    root_pid: int | None,
    anomaly: Any,
) -> set[int]:
    if root_pid is None or root_pid not in processes:
        return set()
    children: dict[int, list[int]] = {pid: [] for pid in processes}
    for process in processes.values():
        if process.parent_pid in children:
            children[process.parent_pid].append(process.pid)
    reachable: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in reachable:
            anomaly(
                "process_parent_cycle",
                f"process parent graph contains a cycle at PID {pid}",
                pid=pid,
            )
            continue
        reachable.add(pid)
        stack.extend(sorted(children[pid], reverse=True))
    return reachable


def parse_process_exec_trace(
    text: str,
    *,
    initial_cwd: str | None,
    expected_root_pid: int | None = None,
) -> dict[str, Any]:
    """Parse one PID-prefixed strace log into deterministic process evidence."""
    return _Parser(initial_cwd=initial_cwd, expected_root_pid=expected_root_pid).parse(text)


def load_process_exec_trace(
    path: Path,
    *,
    initial_cwd: str | None,
    expected_root_pid: int | None = None,
) -> dict[str, Any]:
    """Read and parse one strace log without replacing undecodable bytes."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        parser = _Parser(initial_cwd=initial_cwd, expected_root_pid=expected_root_pid)
        code = "invalid_trace_encoding" if isinstance(exc, UnicodeDecodeError) else "trace_read_error"
        parser.anomaly(code, f"unable to read process trace safely: {exc}")
        return parser._finish()
    return parse_process_exec_trace(text, initial_cwd=initial_cwd, expected_root_pid=expected_root_pid)


def serialize_process_exec_report(report: dict[str, Any]) -> str:
    """Return canonical, stable JSON for one process evidence report."""
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_process_exec_report(report: dict[str, Any], path: Path) -> None:
    """Write a process evidence report deterministically."""
    path.write_text(serialize_process_exec_report(report), encoding="utf-8")

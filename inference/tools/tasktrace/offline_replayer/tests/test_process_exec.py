# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.measurement.process_exec import (
    PROCESS_EXEC_SCHEMA_VERSION,
    parse_process_exec_trace,
    serialize_process_exec_report,
    write_process_exec_report,
)


def _process(report: dict[str, object], pid: int) -> dict[str, object]:
    return next(item for item in report["processes"] if item["pid"] == pid)  # type: ignore[index,union-attr]


def _codes(report: dict[str, object]) -> set[str]:
    return {item["code"] for item in report["anomalies"]}  # type: ignore[index,union-attr]


def test_realistic_shell_child_tree_tracks_execveat_clone_and_cwd() -> None:
    trace = """\
100 1700000000.000000001 execve("/bin/bash", ["/bin/bash", "-c", "build"], 0x1 /* 1 var */) = 0
100 1700000000.010000000 chdir("/app/work") = 0
100 1700000000.020000000 clone(child_stack=NULL, flags=CLONE_VM|CLONE_VFORK|SIGCHLD) = 101
101 1700000000.021000000 execve("/usr/bin/python3", ["python3", "build.py"], 0x1) = 0
101 1700000000.030000000 chdir("sub") = 0
101 1700000000.040000000 clone3({flags=CLONE_VM|CLONE_VFORK, exit_signal=SIGCHLD}, 88) = 102
102 1700000000.041000000 execveat(AT_FDCWD, "../bin/helper", ["helper", "--x"], 0x1, 0) = 0
102 1700000000.050000000 +++ exited with 0 +++
101 1700000000.060000000 +++ exited with 0 +++
100 1700000000.070000000 +++ exited with 0 +++
"""

    report = parse_process_exec_trace(trace, initial_cwd="/app")

    assert report["schema_version"] == PROCESS_EXEC_SCHEMA_VERSION
    assert report["valid"] is True
    assert report["trace_complete"] is True
    assert report["anomalies"] == []
    assert report["scope"] == {
        "kind": "trace-root-and-descendants",
        "root_pid": 100,
        "root_inference": "earliest-unparented-task",
        "process_escape_detected": False,
        "process_escape_pids": [],
        "unresolved_parent_pids": [],
    }

    root = _process(report, 100)
    python = _process(report, 101)
    helper = _process(report, 102)
    assert root["start_ns"] == 1_700_000_000_000_000_001
    assert root["binary"] == "/bin/bash"
    assert root["cwd"] == "/app"
    assert root["final_observed_cwd"] == "/app/work"
    assert root["children"] == [101]
    assert python["parent_pid"] == 100
    assert python["creation_syscall"] == "clone"
    assert python["binary"] == "/usr/bin/python3"
    assert python["argv"] == ["python3", "build.py"]
    assert python["cwd"] == "/app/work"
    assert python["final_observed_cwd"] == "/app/work/sub"
    assert helper["parent_pid"] == 101
    assert helper["creation_syscall"] == "clone3"
    assert helper["binary"] == "/app/work/bin/helper"
    assert helper["cwd"] == "/app/work/sub"
    assert helper["exit_code"] == 0
    assert helper["end_reason"] == "exit"
    assert helper["execs"][0]["syscall"] == "execveat"  # type: ignore[index]
    assert helper["execs"][0]["dirfd"] == "AT_FDCWD"  # type: ignore[index]
    assert [item["binary"] for item in report["binary_inventory"]] == [  # type: ignore[index,union-attr]
        "/app/work/bin/helper",
        "/bin/bash",
        "/usr/bin/python3",
    ]


@pytest.mark.parametrize("syscall", ["fork", "vfork"])
def test_fork_variants_inherit_executable_argv_and_cwd(syscall: str) -> None:
    report = parse_process_exec_trace(
        f"""\
200 1700000100.000000000 execve("/bin/sh", ["sh", "-c", "work"], 0x1) = 0
200 1700000100.100000000 chdir("/workspace") = 0
200 1700000100.200000000 {syscall}() = 201
201 1700000100.300000000 +++ exited with 7 +++
200 1700000100.400000000 +++ exited with 0 +++
""",
        initial_cwd="/",
    )

    assert report["valid"] is True
    child = _process(report, 201)
    assert child["parent_pid"] == 200
    assert child["creation_syscall"] == syscall
    assert child["binary"] == "/bin/sh"
    assert child["binary_source"] == "inherited-from-parent"
    assert child["argv"] == ["sh", "-c", "work"]
    assert child["cwd"] == "/workspace"
    assert child["exit_code"] == 7


def test_clone_thread_is_not_mislabeled_as_a_process() -> None:
    report = parse_process_exec_trace(
        """\
300 1700000200.000000000 execve("/bin/worker", ["worker"], 0x1) = 0
300 1700000200.100000000 clone(child_stack=0x1, flags=CLONE_VM|CLONE_FS|CLONE_SIGHAND|CLONE_THREAD) = 301
301 1700000200.200000000 +++ exited with 0 +++
300 1700000200.300000000 +++ exited with 0 +++
""",
        initial_cwd="/srv",
    )

    assert report["valid"] is True
    assert report["summary"]["process_count"] == 1  # type: ignore[index]
    assert report["summary"]["thread_count"] == 1  # type: ignore[index]
    assert [item["pid"] for item in report["processes"]] == [300]  # type: ignore[index,union-attr]
    assert report["threads"] == [
        {
            "tid": 301,
            "process_pid": 300,
            "start_ns": 1_700_000_200_100_000_000,
            "end_ns": 1_700_000_200_200_000_000,
            "exit_code": 0,
            "exit_signal": None,
        }
    ]


def test_thread_that_exits_before_parent_clone_resume_is_not_an_escape() -> None:
    report = parse_process_exec_trace(
        """\
310 1700000250.000000000 execve("/bin/worker", ["worker"], 0x1) = 0
310 1700000250.100000000 clone(child_stack=0x1, flags=CLONE_VM|CLONE_FS|CLONE_THREAD <unfinished ...>
311 1700000250.120000000 clone(child_stack=0x2, flags=CLONE_VM|CLONE_FS|CLONE_THREAD) = 312
312 1700000250.140000000 +++ exited with 0 +++
311 1700000250.150000000 +++ exited with 0 +++
310 1700000250.200000000 <... clone resumed>) = 311
310 1700000250.300000000 +++ exited with 0 +++
""",
        initial_cwd="/srv",
    )

    # A uniquely paired split syscall is losslessly reconstructed while the
    # completed relation proves that 311 was a thread, not an escaped process.
    assert report["valid"] is True
    assert _codes(report) == set()
    assert [item["pid"] for item in report["processes"]] == [310]  # type: ignore[index,union-attr]
    assert report["scope"]["process_escape_pids"] == []  # type: ignore[index]
    assert [item["tid"] for item in report["threads"]] == [311, 312]  # type: ignore[index,union-attr]
    assert report["threads"][0]["end_ns"] == 1_700_000_250_150_000_000  # type: ignore[index]


def test_vfork_child_execs_before_parent_resume_inherit_parent_cwd() -> None:
    report = parse_process_exec_trace(
        """\
320 1700000260.000000000 execve("/bin/sh", ["sh"], 0x1) = 0
320 1700000260.100000000 vfork( <unfinished ...>
321 1700000260.110000000 execve("/missing/path/tool", ["tool"], 0x1) = -1 ENOENT (No such file or directory)
321 1700000260.120000000 execve("/also/missing/tool", ["tool"], 0x1) = -1 ENOENT (No such file or directory)
321 1700000260.130000000 execve("/usr/bin/tool", ["tool", "--build"], 0x1) = 0
320 1700000260.140000000 <... vfork resumed>) = 321
321 1700000260.200000000 +++ exited with 0 +++
320 1700000260.300000000 +++ exited with 0 +++
""",
        initial_cwd="/workspace",
    )

    assert report["valid"] is True
    assert _codes(report) == set()
    child = _process(report, 321)
    assert child["parent_pid"] == 320
    assert child["creation_syscall"] == "vfork"
    assert child["cwd"] == "/workspace"
    assert child["binary"] == "/usr/bin/tool"
    assert [item["cwd"] for item in child["failed_execs"]] == [  # type: ignore[index]
        "/workspace",
        "/workspace",
    ]
    assert child["execs"][-1]["cwd"] == "/workspace"  # type: ignore[index]


def test_signaled_exit_is_a_complete_observed_lifecycle() -> None:
    report = parse_process_exec_trace(
        """\
400 1700000300.000000000 execve("/bin/bash", ["bash"], 0x1) = 0
400 1700000300.100000000 clone(child_stack=NULL, flags=SIGCHLD) = 401
401 1700000300.200000000 execve("/bin/sleep", ["sleep", "10"], 0x1) = 0
401 1700000300.300000000 +++ killed by SIGKILL +++
400 1700000300.400000000 +++ exited with 0 +++
""",
        initial_cwd="/tmp",
    )

    assert report["valid"] is True
    child = _process(report, 401)
    assert child["end_reason"] == "signal"
    assert child["exit_code"] is None
    assert child["exit_signal"] == "SIGKILL"
    assert child["end_ns"] == 1_700_000_300_300_000_000


def test_split_malformed_and_truncated_records_fail_closed_without_crashing() -> None:
    report = parse_process_exec_trace(
        """\
500 1700000400.000000000 execve("/bin/bash", ["bash"], 0x1) = 0
500 1700000400.100000000 clone(child_stack=NULL, flags=SIGCHLD <unfinished ...>
500 1700000400.200000000 <... clone resumed>) = 501
501 1700000400.300000000 execve("/bin/tool", ["tool", ...], 0x1) = 0
this is not a pid-prefixed strace line
501 1700000400.400000000 +++ exited with 1 +++
500 1700000400.500000000 chdir("/left-open" <unfinished ...>
500 1700000400.600000000 +++ exited with 0 +++
""",
        initial_cwd="/",
    )

    assert report["valid"] is False
    assert report["trace_complete"] is False
    assert report["loss"]["detected"] is True  # type: ignore[index]
    assert report["loss"]["split_syscall_count"] == 1  # type: ignore[index]
    assert {
        "truncated_or_invalid_argv",
        "malformed_trace_line",
        "unfinished_syscall_without_resume",
        "unresolved_process_argv",
    } <= _codes(report)
    assert _process(report, 501)["binary"] == "/bin/tool"
    assert _process(report, 501)["argv"] is None
    assert _process(report, 501)["binary_source"] == "successful-exec"


def test_resumed_record_without_start_fails_closed() -> None:
    report = parse_process_exec_trace(
        """\
600 1700000500.000000000 execve("/bin/sh", ["sh"], 0x1) = 0
600 1700000500.100000000 <... clone resumed>) = 601
600 1700000500.200000000 +++ exited with 0 +++
""",
        initial_cwd="/",
    )

    assert report["valid"] is False
    assert "resumed_syscall_without_start" in _codes(report)
    assert report["loss"]["resumed_without_start_count"] == 1  # type: ignore[index]


def test_unresolved_child_lifecycle_is_retained_and_invalid() -> None:
    report = parse_process_exec_trace(
        """\
700 1700000600.000000000 execve("/bin/bash", ["bash"], 0x1) = 0
700 1700000600.100000000 clone(child_stack=NULL, flags=SIGCHLD) = 701
700 1700000600.200000000 +++ exited with 0 +++
""",
        initial_cwd="/work",
    )

    assert report["valid"] is False
    child = _process(report, 701)
    assert child["parent_pid"] == 700
    assert child["binary"] == "/bin/bash"
    assert child["cwd"] == "/work"
    assert child["end_ns"] is None
    assert "unresolved_process_lifecycle" in _codes(report)
    assert report["loss"]["unresolved_process_count"] == 1  # type: ignore[index]


def test_unrelated_pid_is_reported_as_escape_with_unresolved_parent() -> None:
    report = parse_process_exec_trace(
        """\
800 1700000700.000000000 execve("/bin/bash", ["bash"], 0x1) = 0
999 1700000700.100000000 execve("/bin/escape", ["escape"], 0x1) = 0
999 1700000700.200000000 +++ exited with 0 +++
800 1700000700.300000000 +++ exited with 0 +++
""",
        initial_cwd="/",
    )

    assert report["valid"] is False
    assert report["scope"]["process_escape_detected"] is True  # type: ignore[index]
    assert report["scope"]["process_escape_pids"] == [999]  # type: ignore[index]
    assert report["scope"]["unresolved_parent_pids"] == [999]  # type: ignore[index]
    assert "unresolved_exec_cwd" in _codes(report)
    escaped = _process(report, 999)
    assert escaped["parent_pid"] is None
    assert escaped["parent_resolution"] == "unresolved"
    assert escaped["in_scope"] is False
    assert escaped["process_escape"] is True


def test_successful_fchdir_invalidates_unknown_cwd_instead_of_inventing_it() -> None:
    report = parse_process_exec_trace(
        """\
900 1700000800.000000000 execve("/bin/bash", ["bash"], 0x1) = 0
900 1700000800.100000000 fchdir(7) = 0
900 1700000800.200000000 execve("relative-tool", ["relative-tool"], 0x1) = 0
900 1700000800.300000000 +++ exited with 0 +++
""",
        initial_cwd="/known",
    )

    assert report["valid"] is False
    assert {
        "unresolved_fchdir_cwd",
        "unresolved_exec_binary",
        "unresolved_exec_cwd",
        "unresolved_process_binary",
        "unresolved_process_cwd",
    } <= _codes(report)
    root = _process(report, 900)
    assert root["execs"][-1]["binary"] is None  # type: ignore[index]
    assert root["execs"][-1]["cwd"] is None  # type: ignore[index]
    assert root["final_observed_cwd"] is None


def test_json_serialization_and_file_output_are_deterministic(tmp_path: Path) -> None:
    trace = """\
1000 1700000900.000000009 execve("/bin/echo", ["echo", "héllo"], 0x1) = 0
1000 1700000900.100000009 +++ exited with 0 +++
"""
    first = parse_process_exec_trace(trace, initial_cwd="/", expected_root_pid=1000)
    second = parse_process_exec_trace(trace, initial_cwd="/", expected_root_pid=1000)

    first_json = serialize_process_exec_report(first)
    second_json = serialize_process_exec_report(second)
    assert first_json == second_json
    assert "héllo" in first_json
    assert first_json.endswith("\n")
    assert json.loads(first_json) == first

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_process_exec_report(first, first_path)
    write_process_exec_report(second, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()

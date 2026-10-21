# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.reporting.scheduler_batch import (
    BatchReportError,
    _command_preview,
    _load_timeline,
    build_command_summary,
    build_summary,
    build_tool_family_summary,
    classify_command_family,
    write_outputs,
)


def _report(path: Path, task: str, *, valid: bool = True, mismatches: int = 0) -> Path:
    path.write_text(
        json.dumps(
            {
                "task_name": task,
                "source_trial": f"{task}__trial",
                "replay_summary": {"source_outcome_mismatch_count": mismatches},
                "container_cpu": {
                    "wall_sec": 2.0,
                    "usage_sec": 2.4,
                    "average_logical_cores_used": 1.2,
                    "allocated_logical_cpus": 8,
                },
                "scheduler_concurrency": {
                    "average_logical_cores_used": 1.19,
                    "peak_simultaneous_logical_cores": 3,
                    "time_sec_by_simultaneous_logical_cores": {
                        "0": 0.1,
                        "1": 1.4,
                        "2": 0.4,
                        "3": 0.1,
                    },
                    "lost_perf_events": 0,
                    "cgroup_validation": {
                        "valid_for_concurrency_reporting": valid,
                    },
                },
            }
        )
    )
    return path


def _attach_action_inputs(report_path: Path, task: str) -> None:
    replay_path = report_path.parent / f"{task}-replay.json"
    events_path = report_path.parent / f"{task}-events.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "action_index": 1,
                        "tool": "exec_command",
                        "command": "python3 work.py",
                        "status": "ok",
                        "source_status": "ok",
                        "replay_start_monotonic_ns": 100,
                        "replay_end_monotonic_ns": 400,
                    }
                ]
            }
        )
    )
    events_path.write_text(
        json.dumps({"timestamp_ns": 150, "cpu": 1, "tid": 1, "entering": True})
        + "\n"
        + json.dumps({"timestamp_ns": 350, "cpu": 1, "tid": 1, "entering": False})
        + "\n"
    )
    payload = json.loads(report_path.read_text())
    payload["replay_report"] = str(replay_path)
    payload["scheduler_concurrency"]["events_path"] = str(events_path)
    report_path.write_text(json.dumps(payload))


def test_write_outputs_renders_validated_batch_charts(tmp_path: Path) -> None:
    a = _report(tmp_path / "a.json", "task-a")
    b = _report(tmp_path / "b.json", "task-b")
    _attach_action_inputs(a, "task-a")
    _attach_action_inputs(b, "task-b")
    tasks = build_summary(
        [
            a,
            b,
        ]
    )

    outputs = write_outputs(tasks, tmp_path / "output")

    summary = json.loads(Path(outputs["json"]).read_text())
    assert summary["schema_version"] == "scheduler-batch-summary.v1"
    assert summary["tasks"][0]["time_above_one_core_sec"] == pytest.approx(0.5)
    assert "task-a" in Path(outputs["average_vs_peak_svg"]).read_text()
    assert "Per-Task Simultaneous Logical-CPU Distribution" in Path(outputs["core_distribution_svg"]).read_text()
    commands = json.loads(Path(outputs["bash_commands_json"]).read_text())
    assert commands["bash_commands"][0]["command"] == "python3 work.py"
    frequency = json.loads(Path(outputs["bash_command_frequency_json"]).read_text())
    assert frequency["bash_command_frequency"][0]["command"] == "python3 work.py"
    assert frequency["bash_command_frequency"][0]["invocation_count"] == 2
    assert frequency["bash_command_frequency"][0]["task_count"] == 2
    families = json.loads(Path(outputs["tool_family_distribution_json"]).read_text())
    assert families["tool_families"][0]["command_family"] == "python work.py"
    assert families["tool_families"][0]["invocation_count"] == 2
    for output_name in (
        "csv",
        "bash_commands_csv",
        "bash_command_frequency_csv",
        "tool_family_distribution_csv",
    ):
        assert b"\r" not in Path(outputs[output_name]).read_bytes()


def test_command_preview_does_not_leave_trailing_space_when_truncated() -> None:
    command, preview = _command_preview({"command": ("x" * 109) + " tail"})

    assert command.endswith(" tail")
    assert preview == "x" * 109


def test_build_summary_rejects_unvalidated_or_drifted_reports(tmp_path: Path) -> None:
    with pytest.raises(BatchReportError, match="not cgroup-validated"):
        build_summary([_report(tmp_path / "invalid.json", "invalid", valid=False)])
    with pytest.raises(BatchReportError, match="replay outcome mismatches"):
        build_summary([_report(tmp_path / "drifted.json", "drifted", mismatches=1)])


@pytest.mark.parametrize(
    ("command", "family"),
    [
        ("go test ./...", "go test"),
        ("go build ./cmd/service", "go build"),
        ("python -m pytest tests/test_api.py", "python -m pytest"),
        ("python -m pip install -e .", "python -m pip install"),
        ("set +e; valgrind /app/debug; printf '%s\\n' \"$?\"", "valgrind"),
        ("rm -f /tmp/frame.bmp; node /app/vm.js", "node vm.js"),
        ("python -m py_compile app.py && python app.py", "multi-tool: python -m py_compile + python app.py"),
        ("grep -E 'np\\.(int|float)' src.py | sed -n '1,20p'", "grep pipeline"),
        ("python - <<'PY'\nif left | right:\n    print('ok')\nPY", "python inline script"),
    ],
)
def test_classify_command_family_preserves_semantic_subcommands(command: str, family: str) -> None:
    assert classify_command_family(command) == family


def test_load_timeline_uses_recorded_action_window_and_scheduler_events(tmp_path: Path) -> None:
    report_path = _report(tmp_path / "pipeline.json", "task-a")
    replay_path = tmp_path / "replay.json"
    events_path = tmp_path / "events.jsonl"
    replay_path.write_text(
        json.dumps({"actions": [{"replay_start_monotonic_ns": 100, "replay_end_monotonic_ns": 400}]})
    )
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp_ns": 150, "cpu": 1, "tid": 1, "entering": True}),
                json.dumps({"timestamp_ns": 250, "cpu": 2, "tid": 2, "entering": True}),
                json.dumps({"timestamp_ns": 350, "cpu": 1, "tid": 1, "entering": False}),
            ]
        )
        + "\n"
    )
    payload = json.loads(report_path.read_text())
    payload["replay_report"] = str(replay_path)
    payload["scheduler_concurrency"]["events_path"] = str(events_path)
    report_path.write_text(json.dumps(payload))

    task = build_summary([report_path])[0]
    seconds, running, duration_sec = _load_timeline(task)

    assert duration_sec == pytest.approx(0.0000003)
    assert max(running) == 2
    assert seconds[0] == 0.0


def test_build_command_summary_attributes_concurrency_to_command(tmp_path: Path) -> None:
    report_path = _report(tmp_path / "pipeline.json", "task-a")
    replay_path = tmp_path / "replay.json"
    events_path = tmp_path / "events.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "action_index": 1,
                        "tool": "exec_command",
                        "command": "python3 work.py",
                        "status": "ok",
                        "source_status": "ok",
                        "replay_start_monotonic_ns": 100,
                        "replay_end_monotonic_ns": 400,
                    }
                ]
            }
        )
    )
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp_ns": 150, "cpu": 1, "tid": 1, "entering": True}),
                json.dumps({"timestamp_ns": 200, "cpu": 2, "tid": 2, "entering": True}),
                json.dumps({"timestamp_ns": 300, "cpu": 2, "tid": 2, "entering": False}),
                json.dumps({"timestamp_ns": 350, "cpu": 1, "tid": 1, "entering": False}),
            ]
        )
        + "\n"
    )
    payload = json.loads(report_path.read_text())
    payload["replay_report"] = str(replay_path)
    payload["scheduler_concurrency"]["events_path"] = str(events_path)
    report_path.write_text(json.dumps(payload))

    call = build_command_summary(build_summary([report_path]))[0]

    assert call.command == "python3 work.py"
    assert call.peak_simultaneous_logical_cores == 2
    assert call.time_at_or_above_2_cores_sec == pytest.approx(0.0000001)


def test_build_command_summary_collapses_write_stdin_polls_into_async_command(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path / "pipeline.json", "task-a")
    replay_path = tmp_path / "replay.json"
    events_path = tmp_path / "events.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "action_index": 1,
                        "tool": "exec_command",
                        "command": "make -j8",
                        "session_id": "9",
                        "status": "running",
                        "source_status": "running",
                        "replay_start_monotonic_ns": 100,
                        "replay_end_monotonic_ns": 200,
                    },
                    {
                        "action_index": 2,
                        "tool": "write_stdin",
                        "session_id": "9",
                        "status": "ok",
                        "source_status": "ok",
                        "replay_start_monotonic_ns": 300,
                        "replay_end_monotonic_ns": 600,
                    },
                ]
            }
        )
    )
    events_path.write_text("")
    payload = json.loads(report_path.read_text())
    payload["replay_report"] = str(replay_path)
    payload["scheduler_concurrency"]["events_path"] = str(events_path)
    report_path.write_text(json.dumps(payload))

    command = build_command_summary(build_summary([report_path]))[0]

    assert command.command == "make -j8"
    assert command.poll_count == 1
    assert command.lifecycle_wall_sec == pytest.approx(0.0000005)


def test_tool_family_active_time_unions_overlapping_same_family_commands(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path / "pipeline.json", "task-a")
    replay_path = tmp_path / "replay.json"
    events_path = tmp_path / "events.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "actions": [
                    {
                        "action_index": 1,
                        "tool": "exec_command",
                        "command": "make -j8 all",
                        "status": "ok",
                        "source_status": "ok",
                        "replay_start_monotonic_ns": 0,
                        "replay_end_monotonic_ns": 1_500_000_000,
                    },
                    {
                        "action_index": 2,
                        "tool": "exec_command",
                        "command": "make -j8 tests",
                        "status": "ok",
                        "source_status": "ok",
                        "replay_start_monotonic_ns": 500_000_000,
                        "replay_end_monotonic_ns": 2_000_000_000,
                    },
                ]
            }
        )
    )
    events_path.write_text("")
    payload = json.loads(report_path.read_text())
    payload["replay_report"] = str(replay_path)
    payload["scheduler_concurrency"]["events_path"] = str(events_path)
    report_path.write_text(json.dumps(payload))

    tasks = build_summary([report_path])
    family = build_tool_family_summary(tasks, build_command_summary(tasks))[0]

    assert family.command_family == "make"
    assert family.invocation_count == 2
    assert family.summed_invocation_lifecycle_wall_sec == pytest.approx(3.0)
    assert family.active_lifecycle_wall_sec == pytest.approx(2.0)
    assert family.overlap_collapsed_wall_sec == pytest.approx(1.0)
    assert family.active_lifecycle_percent_of_replay_wall == pytest.approx(100.0)


@pytest.mark.parametrize("status, unavailable", [("invalid", 0), ("valid", 1)])
def test_batch_rejects_invalid_or_unavailable_replay(tmp_path: Path, status: str, unavailable: int) -> None:
    path = _report(tmp_path / "report.json", "task")
    payload = json.loads(path.read_text())
    payload["status"] = status
    payload["replay_summary"]["source_outcome_comparison_unavailable_count"] = unavailable
    path.write_text(json.dumps(payload))
    with pytest.raises(BatchReportError):
        build_summary([path])


@pytest.mark.parametrize(
    "field, value",
    [
        ("actions_launch_failed", 1),
        ("actions_source_argument_parse_failed", 1),
        ("actions_skipped_after_timeout", 1),
        ("source_outcome_comparison_unavailable_count", False),
        ("tool_actions_recorded", 2),
    ],
)
def test_batch_rejects_incomplete_tools(tmp_path: Path, field: str, value) -> None:
    path = _report(tmp_path / "report.json", "task")
    payload = json.loads(path.read_text())
    payload["replay_summary"][field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(BatchReportError):
        build_summary([path])

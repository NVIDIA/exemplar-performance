# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import json
import os
import signal
import tempfile
import time
from pathlib import Path

import pytest
from offline_replayer.adapters.atif_replay import (
    AtifAction,
    AtifTrajectory,
    _claude_glob,
    _claude_grep,
    _claude_read,
    _recorded_tool_wall_sec,
    _workload_environment,
    read_trajectory,
    replay_trajectory,
    source_observed_python3_absent,
    validate_supported_actions,
)
from offline_replayer.adapters.transcript_replay import ReplayError
from offline_replayer.measurement.pipeline import _atif_summary_is_valid


def _action(observation_text: str) -> AtifAction:
    return AtifAction(
        index=1,
        step_id=1,
        timestamp_sec=0.0,
        tool="exec_command",
        source_tool="exec_command",
        code_mode=False,
        source_agent_original_format=None,
        arguments={"cmd": "true"},
        observation={"content": observation_text},
        observation_text=observation_text,
    )


def _write_trajectory(
    path: Path,
    actions: list[tuple[str, dict[str, object], object]],
    *,
    agent: dict[str, object] | None = None,
) -> Path:
    steps = []
    for index, (tool, arguments, observation) in enumerate(actions, start=1):
        observation_payload = observation if isinstance(observation, dict) else {"content": observation}
        steps.append(
            {
                "step_id": index,
                "timestamp": f"2026-05-24T01:00:0{index}+00:00",
                "tool_calls": [{"function_name": tool, "arguments": arguments}],
                "observation": observation_payload,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.5",
                "session_id": "session",
                "agent": agent or {"name": "codex"},
                "steps": steps,
            }
        )
    )
    return path


def _load_actions(
    path: Path,
    actions: list[tuple[str, dict[str, object], object]],
    *,
    agent: dict[str, object] | None = None,
) -> AtifTrajectory:
    return read_trajectory(_write_trajectory(path, actions, agent=agent))


def _write_code_mode_cell(path: Path, input_source: str, *, tool_call_id: str | None = None) -> Path:
    tool_call: dict[str, object] = {"function_name": "exec", "arguments": {"input": input_source}}
    if tool_call_id is not None:
        tool_call["tool_call_id"] = tool_call_id
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.5",
                "session_id": "session",
                "agent": {"name": "codex", "extra": {"originator": "codex_exec"}},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-05-24T01:00:01+00:00",
                        "tool_calls": [tool_call],
                    }
                ],
            }
        )
    )
    return path


def _mini_swe_observation(returncode: int, *, output: str = "", exception_info: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"returncode": returncode, "output": output}
    if exception_info is not None:
        result["exception_info"] = exception_info
    return {"results": [{"content": json.dumps(result)}]}


def test_read_trajectory_loads_gzipped_document(tmp_path: Path) -> None:
    # Large trajectories are stored gzipped to stay under hosting size limits.
    plain = _write_trajectory(
        tmp_path / "trajectory.json",
        [("exec_command", {"cmd": "true"}, "Process exited with code 0")],
    )
    raw = plain.read_bytes()
    plain.unlink()
    gz = tmp_path / "trajectory.json.gz"
    gz.write_bytes(gzip.compress(raw))

    # The .gz path loads directly...
    assert read_trajectory(gz).actions[0].tool == "exec_command"
    # ...and the plain-.json path transparently falls back to its .gz sibling,
    # so callers (the runner) need not know which form is on disk.
    assert read_trajectory(tmp_path / "trajectory.json").actions[0].tool == "exec_command"


def test_read_trajectory_detects_gzip_magic_without_gz_suffix(tmp_path: Path) -> None:
    plain = _write_trajectory(
        tmp_path / "trajectory.json",
        [("exec_command", {"cmd": "true"}, "Process exited with code 0")],
    )
    plain.write_bytes(gzip.compress(plain.read_bytes()))

    trajectory = read_trajectory(plain)

    assert trajectory.actions[0].tool == "exec_command"


def test_read_trajectory_unwraps_static_codex_exec_calls(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec",
                {"input": """const patch = `*** Begin Patch
*** Add File: result.txt
+done
*** End Patch`;
const first = await tools.exec_command({cmd: \"printf before\", workdir: \"/app\"});
await tools.apply_patch(patch);
await tools.write_stdin({session_id: 7, chars: \"\\n\"});
text(first.output);"""},
                "Process exited with code 0",
            )
        ],
        agent={"name": "codex", "extra": {"originator": "codex_exec"}},
    )

    assert [(action.tool, action.arguments) for action in trajectory.actions] == [
        ("exec_command", {"cmd": "printf before", "workdir": "/app"}),
        ("apply_patch", {"input": "*** Begin Patch\n*** Add File: result.txt\n+done\n*** End Patch"}),
        ("write_stdin", {"session_id": 7, "chars": "\n"}),
    ]
    assert all(action.observation is None for action in trajectory.actions)


def test_read_trajectory_extracts_literal_promise_all_without_replaying_it(tmp_path: Path) -> None:
    trajectory = read_trajectory(
        _write_code_mode_cell(
            tmp_path / "parallel.json",
            "const results = await Promise.all([tools.exec_command({cmd: 'one'}), "
            "tools.apply_patch('*** Begin Patch\\n*** End Patch')]); for (let i = 0; "
            "i < results.length; i++) { text(results[i].output); }",
            tool_call_id="outer-call",
        )
    )

    assert [(action.tool, action.arguments) for action in trajectory.actions] == [
        ("exec_command", {"cmd": "one"}),
        ("apply_patch", {"input": "*** Begin Patch\n*** End Patch"}),
    ]
    assert [action.parallel_group_id for action in trajectory.actions] == [
        "step-1-call-1:outer-call:promise_all:1",
        "step-1-call-1:outer-call:promise_all:1",
    ]
    assert all(action.observation is None for action in trajectory.actions)
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        validate_supported_actions(trajectory)


def test_promise_all_group_ids_are_unique_and_never_inherit_outer_outcomes(
    tmp_path: Path,
) -> None:
    path = _write_trajectory(
        tmp_path / "parallel.json",
        [
            (
                "exec",
                {"input": "await Promise.all ([tools.exec_command({cmd: 'one'})]);"},
                "Process exited with code 0",
            ),
            (
                "exec",
                {"input": "await Promise.all([tools.exec_command({cmd: 'two'})]);"},
                "Process exited with code 0",
            ),
        ],
        agent={"name": "codex", "extra": {"originator": "codex_exec"}},
    )
    payload = json.loads(path.read_text())
    for step in payload["steps"]:
        step["tool_calls"][0]["tool_call_id"] = "reused"
    path.write_text(json.dumps(payload))

    trajectory = read_trajectory(path)

    assert [action.parallel_group_id for action in trajectory.actions] == [
        "step-1-call-1:reused:promise_all:1",
        "step-2-call-1:reused:promise_all:1",
    ]
    assert all(action.observation is None for action in trajectory.actions)


@pytest.mark.parametrize(
    "input_source, message",
    [
        ("await Promise.all([]);", "must contain a stateful tool call"),
        ("await Promise.all([tools.update_plan({plan: []})]);", "only supports stateful"),
        ("await Promise.all([await tools.exec_command({cmd: 'one'})]);", "a tools call"),
        (
            "const results = await Promise.all([tools.exec_command({cmd: 'one'})]); "
            "for (let i = 0; i < results.length; i++) { text(\"text(results[i]\"); }",
            "invalid Promise.all presentation loop",
        ),
    ],
)
def test_read_trajectory_rejects_nonliteral_promise_all(tmp_path: Path, input_source: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        read_trajectory(_write_code_mode_cell(tmp_path / "parallel.json", input_source))


@pytest.mark.parametrize(
    "input_source, message",
    [
        ("if (ok) await tools.exec_command({cmd: 'true'});", "control flow or parallel"),
        ("if (false) x = await tools.exec_command({cmd: 'true'});", "control flow or parallel"),
        ("// x = await tools.exec_command({cmd: 'true'});", "control flow or parallel"),
        ("/* x = await tools.exec_command({cmd: 'true'}); */", "control flow or parallel"),
        ("await tools.exec_command({cmd: command});", "dynamic tool argument"),
        ("const value = ['a', 'b']; await tools.exec_command({cmd: `echo ${value}`});", "template interpolation"),
        ("const value = [true]; await tools.exec_command({cmd: `echo ${value}`});", "template interpolation"),
        ("const value = [null]; await tools.exec_command({cmd: `echo ${value}`});", "template interpolation"),
        ("await tools.apply_patch({input: 1, patch: 'valid'});", "string input"),
        ("await tools.unknown({});", "unsupported tool call"),
    ],
)
def test_read_trajectory_rejects_nonstatic_codex_exec(tmp_path: Path, input_source: str, message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        _load_actions(
            tmp_path / "trajectory.json",
            [("exec", {"input": input_source}, "")],
            agent={"name": "codex", "extra": {"originator": "codex_exec"}},
        )


def test_read_trajectory_does_not_unwrap_generic_exec(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("exec", {"input": "await tools.exec_command({cmd: 'true'});"}, "")],
    )

    assert trajectory.actions[0].tool == "exec"


def test_codex_exec_accepts_only_display_only_prefixed_tool_listing(tmp_path: Path) -> None:
    safe = (
        'const names = ALL_TOOLS.filter(x => '
        'x.name.startsWith("mcp__codex_apps__github_") && '
        '/(get|fetch|list|search).*(issue|pull|pr)/i.test(x.name))'
        '.map(x=>x.name); text(names); '
        'await tools.mcp__codex_apps__github_search_issues({repo:"gin-gonic/gin",query:"multipart"}); '
        'await tools.exec_command({cmd:"true"});'
    )
    assert len(read_trajectory(_write_code_mode_cell(tmp_path / "safe.json", safe)).actions) == 2
    for index, listing in enumerate(
        [
            'const hits=ALL_TOOLS.filter(x=>/github|pull request|issue/.test((x.name+" "+x.description).toLowerCase())); text(hits);',
            'const hits=ALL_TOOLS.filter(x=>/github_(get|fetch).*pull|github_search_prs|github_get_issue|github_fetch_issue/.test(x.name)).map(x=>x.name+" :: "+x.description.slice(0,240)); text(hits);',
            'const candidates=ALL_TOOLS.filter(x=>/search/i.test(x.name+" "+x.description)&&/web|github|internet/i.test(x.description)); text(candidates);',
        ]
    ):
        assert (
            len(
                read_trajectory(
                    _write_code_mode_cell(
                        tmp_path / f"safe-regex-{index}.json",
                        listing + ' await tools.exec_command({cmd:"true"});',
                    )
                ).actions
            )
            == 1
        )
    loop = (
        'for (const n of ["mcp__github_fetch_issue","mcp__github_fetch_pr"]) {'
        ' const x=ALL_TOOLS.find(y=>y.name===n); text(x); } '
        'await tools.exec_command({cmd:"true"});'
    )
    assert len(read_trajectory(_write_code_mode_cell(tmp_path / "loop.json", loop)).actions) == 1
    query_loop = (
        'const queries=["one","two"]; for (const q of queries) {'
        ' const r=await tools.mcp__codex_apps__github_search_issues('
        '{query:q,repository_full_name:"gin-gonic/gin",topn:30}); text(q); text(r); } '
        'await tools.exec_command({cmd:"true"});'
    )
    assert len(read_trajectory(_write_code_mode_cell(tmp_path / "query-loop.json", query_loop)).actions) == 1
    for index, unsafe in enumerate(
        [
            'const x=ALL_TOOLS.filter(x=>x.name.startsWith("mcp__")&&/github/g.test(x.name)).map(x=>x.name); text(x);',
            'const x=ALL_TOOLS.filter(x=>x.name.startsWith("mcp__")&&/github/i.test(tools.exec_command({cmd:"true"}))).map(x=>x.name); text(x);',
            'const x=ALL_TOOLS.filter(x=>/github/.test((tools.exec_command({cmd:"true"})+x.description).toLowerCase())); text(x);',
            'const x=ALL_TOOLS.filter(x=>/github/i.test(x.name)&&/web/i.test(tools.exec_command({cmd:"true"}))); text(x);',
        ]
    ):
        with pytest.raises(ReplayError, match="dynamic const binding"):
            read_trajectory(_write_code_mode_cell(tmp_path / f"unsafe-{index}.json", unsafe))
    with pytest.raises(ReplayError, match="unsupported tool call"):
        read_trajectory(
            _write_code_mode_cell(
                tmp_path / "effectful-github.json",
                'await tools.mcp__codex_apps__github_create_issue({repo:"gin-gonic/gin"});',
            )
        )
    with pytest.raises(ReplayError, match="unsupported tool call"):
        read_trajectory(
            _write_code_mode_cell(
                tmp_path / "unknown-read-prefixed-github.json",
                'await tools.mcp__codex_apps__github_fetch_repository_secret({repo:"gin-gonic/gin"});',
            )
        )


def test_codex_exec_direct_coordination_is_metadata_only(tmp_path: Path) -> None:
    code_mode = _load_actions(
        tmp_path / "code-mode.json",
        [
            ("spawn_agent", {"task_name": "audit", "message": "inspect"}, ""),
            ("exec_command", {"cmd": "true"}, "Process exited with code 0"),
        ],
        agent={"name": "codex", "extra": {"originator": "codex_exec"}},
    )
    generic = _load_actions(
        tmp_path / "generic.json",
        [
            ("spawn_agent", {"task_name": "audit", "message": "inspect"}, ""),
            ("exec_command", {"cmd": "true"}, "Process exited with code 0"),
        ],
    )

    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        validate_supported_actions(code_mode)
    with pytest.raises(ReplayError, match="spawn_agent"):
        validate_supported_actions(generic)


def test_read_trajectory_missing_with_no_gz_raises(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="does not exist"):
        read_trajectory(tmp_path / "nope.json")


def test_source_observed_python3_absent_detects_command_not_found(tmp_path: Path) -> None:
    absent = _load_actions(
        tmp_path / "absent.json",
        [("exec_command", {"cmd": "python3 -V"}, "/bin/bash: line 1: python3: command not found")],
    )
    present = _load_actions(
        tmp_path / "present.json",
        [("exec_command", {"cmd": "python3 -V"}, "Python 3.12.3\nProcess exited with code 0")],
    )
    assert source_observed_python3_absent(absent) is True
    assert source_observed_python3_absent(present) is False


def test_recorded_tool_wall_sec_parses_observation_wall_time() -> None:
    assert _recorded_tool_wall_sec(_action("Wall time: 1.50 seconds\nExit code: 0")) == 1.5
    assert _recorded_tool_wall_sec(_action("Wall time: 0 seconds")) == 0.0
    # No wall time recorded (e.g. apply_patch) collapses the gap to issue-to-issue.
    assert _recorded_tool_wall_sec(_action("patch applied")) == 0.0


def test_llm_gaps_pacing_reinjects_model_gap_minus_tool_time(tmp_path: Path) -> None:
    # Two instant tool calls issued 0.30s apart; the first recorded a 0.10s tool
    # wall, so the model round-trip before the second action is 0.30 - 0.10 = 0.20s.
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.5",
                "session_id": "s",
                "agent": {"name": "codex"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": 0.0,
                        "tool_calls": [{"function_name": "exec_command", "arguments": {"cmd": "true"}}],
                        "observation": {"content": "Wall time: 0.10 seconds\nProcess exited with code 0"},
                    },
                    {
                        "step_id": 2,
                        "timestamp": 0.30,
                        "tool_calls": [{"function_name": "exec_command", "arguments": {"cmd": "true"}}],
                        "observation": {"content": "Process exited with code 0"},
                    },
                ],
            }
        )
    )
    trajectory = read_trajectory(trajectory_path)

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="llm-gaps")

    second = report["actions"][1]
    assert second["source_llm_gap_sec"] == 0.2
    # The first action injects no gap; the second waits ~0.2s for the model gap.
    assert report["actions"][0].get("source_llm_gap_sec") is None
    assert report["summary"]["wall_time_sec"] >= 0.18


def test_replay_trajectory_runs_commands_and_ignores_plan_updates(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "printf done > result.txt", "workdir": str(nested)},
                "Process exited with code 0",
            ),
            ("update_plan", {"plan": []}, "Plan updated"),
            ("web_search_call", {"query": "numpy docs"}, "Search results"),
            ("view_image", {"path": "/tmp/diagram.png"}, "Image rendered"),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["tool_actions_recorded"] == 1
    assert report["summary"]["tool_actions_replayed"] == 1
    assert _atif_summary_is_valid(report["summary"]) is True

    assert (nested / "result.txt").read_text() == "done"
    assert report["summary"]["actions_executed"] == 1
    assert report["summary"]["metadata_actions_ignored"] == 3
    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert isinstance(report["actions"][0]["process_root_pid"], int)
    assert report["actions"][0]["process_root_pid"] > 0
    assert report["actions"][0]["replay_start_realtime_ns"] > 0
    assert report["actions"][0]["replay_start_realtime_uncertainty_ns"] >= 0
    assert report["actions"][0]["replay_end_monotonic_ns"] >= report["actions"][0]["replay_start_monotonic_ns"]


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize(
    "settings,command,expected",
    [
        ({"shell": "/bin/sh", "login": False, "tty": False}, "printf '%s' \"${BASH_VERSION-unset}\" > actual", "unset"),
        (
            {"shell": "/bin/bash", "login": False},
            "if shopt -q login_shell; then printf login; else printf nonlogin; fi > actual",
            "nonlogin",
        ),
        (
            {"shell": "/bin/bash", "login": True},
            "if shopt -q login_shell; then printf login; else printf nonlogin; fi > actual",
            "login",
        ),
    ],
)
def test_replay_honors_recorded_shell_and_login(tmp_path, background, settings, command, expected):
    actions = [
        (
            "exec_command",
            {"cmd": command, **settings},
            "Process running with session ID 41" if background else "Process exited with code 0",
        )
    ]
    if background:
        actions.append(
            ("write_stdin", {"session_id": 41, "chars": "", "yield_time_ms": 1000}, "Process exited with code 0")
        )
    trajectory = read_trajectory(_write_trajectory(tmp_path / "trajectory.json", actions))
    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")
    assert (tmp_path / "actual").read_text() == expected
    assert report["summary"]["actions_failed"] == 0


@pytest.mark.parametrize(
    "option,value",
    [
        ("tty", "true"),
        ("login", "false"),
        ("login", None),
        ("shell", "/bin/fish"),
        ("shell", "bash"),
        ("shell", "/bin/bash\0"),
        ("shell", None),
    ],
)
def test_replay_rejects_unsupported_execution_options_before_any_action(tmp_path, option, value):
    trajectory = read_trajectory(
        _write_trajectory(
            tmp_path / "trajectory.json",
            [
                ("exec_command", {"cmd": "touch executed-before-preflight"}, "Process exited with code 0"),
                ("exec_command", {"cmd": "true", option: value}, "Process exited with code 0"),
            ],
        )
    )
    with pytest.raises(ReplayError, match=option):
        replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")
    assert not (tmp_path / "executed-before-preflight").exists()


@pytest.mark.parametrize("background", [False, True])
def test_tty_allocates_controlling_terminal_and_drains_large_output(tmp_path, background):
    command = (
        "test -t 0 && test -t 1 && test -t 2 && test -r /dev/tty && "
        "printf 'terminal\\n' > /dev/tty && stty size && "
        "python3 -c \"import os; os.write(1, b'x' * 1048576); os.write(2, b'END')\""
    )
    actions = [
        (
            "exec_command",
            {"cmd": command, "tty": True, "login": False},
            "Process running with session ID 41" if background else "Process exited with code 0",
        )
    ]
    if background:
        actions.append(
            ("write_stdin", {"session_id": 41, "chars": "", "yield_time_ms": 1000}, "Process exited with code 0")
        )
    trajectory = read_trajectory(_write_trajectory(tmp_path / "trajectory.json", actions))
    report = replay_trajectory(
        trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=3, action_output_dir=tmp_path / "outputs"
    )
    assert report["summary"]["actions_failed"] == 0
    assert report["actions"][-1]["returncode"] == 0
    output = Path(report["actions"][0]["session_log" if background else "stdout_path"]).read_bytes()
    assert output == b"terminal\r\n24 80\r\n" + b"x" * 1048576 + b"END"
    if not background:
        assert Path(report["actions"][0]["stderr_path"]).read_bytes() == b""


def test_tty_failed_output_sink_still_drains_child(tmp_path):
    from offline_replayer.adapters.atif_replay import _spawn_pty

    with open("/dev/full", "wb") as output:
        process, terminal = _spawn_pty(
            ["/bin/bash", "-c", "python3 -c \"import os; os.write(1, b'x' * 1048576)\""],
            tmp_path,
            output,
        )
        try:
            assert process.wait(timeout=3) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            with pytest.raises(ReplayError, match="PTY output capture failed.*No space left"):
                terminal.close()


def test_tty_eof_and_poll_preserve_terminal_input_semantics(tmp_path):
    trajectory = read_trajectory(
        _write_trajectory(
            tmp_path / "trajectory.json",
            [
                (
                    "exec_command",
                    {"cmd": "cat > terminal-input", "tty": True, "yield_time_ms": 100},
                    "Process running with session ID 41",
                ),
                (
                    "write_stdin",
                    {"session_id": 41, "chars": "hello\n", "yield_time_ms": 50},
                    "Process running with session ID 41",
                ),
                (
                    "write_stdin",
                    {"session_id": 41, "chars": "", "yield_time_ms": 50},
                    "Process running with session ID 41",
                ),
                (
                    "write_stdin",
                    {"session_id": 41, "chars": "\x04", "yield_time_ms": 1000},
                    "Process exited with code 0",
                ),
            ],
        )
    )
    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=3)
    assert [action["status"] for action in report["actions"]] == ["running", "running", "running", "ok"]
    assert (tmp_path / "terminal-input").read_text() == "hello\n"


@pytest.mark.parametrize("background", [False, True])
def test_tty_timeout_terminates_and_reaps_process(tmp_path, background):
    actions = [
        (
            "exec_command",
            {"cmd": "exec sleep 30", "tty": True},
            "Process running with session ID 41" if background else "Process exited with code 0",
        )
    ]
    if background:
        actions.append(
            ("write_stdin", {"session_id": 41, "chars": "", "yield_time_ms": 1000}, "Process exited with code 0")
        )
    trajectory = read_trajectory(_write_trajectory(tmp_path / "trajectory.json", actions))
    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=0.1)
    assert report["actions"][-1]["status"] == "timed_out"
    with pytest.raises(ProcessLookupError):
        os.kill(report["actions"][0]["process_root_pid"], 0)


def test_tty_large_input_to_nonreading_child_obeys_timeout(tmp_path):
    trajectory = read_trajectory(
        _write_trajectory(
            tmp_path / "trajectory.json",
            [
                (
                    "exec_command",
                    {"cmd": "stty raw -echo; exec sleep 30", "tty": True, "yield_time_ms": 100},
                    "Process running with session ID 41",
                ),
                (
                    "write_stdin",
                    {"session_id": 41, "chars": "x" * 131072, "yield_time_ms": 1000},
                    "Process exited with code 0",
                ),
            ],
        )
    )
    started = time.monotonic()
    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=0.1)
    assert report["actions"][1]["status"] == "timed_out"
    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(report["actions"][0]["process_root_pid"], 0)


def test_tty_input_elapsed_time_counts_toward_hard_timeout(tmp_path):
    command = (
        "exec python3 -c 'import os,time,tty; tty.setraw(0); time.sleep(.35); remaining=131072\n"
        "while remaining:\n remaining-=len(os.read(0,remaining))\n"
        "time.sleep(30)'"
    )
    trajectory = read_trajectory(
        _write_trajectory(
            tmp_path / "trajectory.json",
            [
                (
                    "exec_command",
                    {"cmd": command, "tty": True, "yield_time_ms": 100},
                    "Process running with session ID 41",
                ),
                (
                    "write_stdin",
                    {"session_id": 41, "chars": "x" * 131072, "yield_time_ms": 500},
                    "Process exited with code 0",
                ),
            ],
        )
    )
    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=0.6)
    assert report["actions"][1]["status"] == "timed_out"
    with pytest.raises(ProcessLookupError):
        os.kill(report["actions"][0]["process_root_pid"], 0)


def test_missing_workdir_does_not_count_unexecuted_command(tmp_path):
    trajectory = read_trajectory(
        _write_trajectory(
            tmp_path / "trajectory.json",
            [
                (
                    "exec_command",
                    {
                        "cmd": "printf source-ran > executed-marker; nonexistent-command",
                        "workdir": str(tmp_path / "absent"),
                    },
                    "/bin/bash: line 1: nonexistent-command: command not found",
                ),
            ],
        )
    )
    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")
    assert report["actions"][0]["launch_failed"] is True
    assert report["actions"][0]["command_executed"] is False
    assert report["summary"]["actions_executed"] == 0
    assert not (tmp_path / "executed-marker").exists()
    assert report["summary"]["actions_launch_failed"] == 1
    assert report["actions"][0]["outcome_matches_source"] is None
    assert report["summary"]["source_outcome_comparison_unavailable_count"] == 1

    from offline_replayer.adapters.atif_replay import main as replay_main

    output = tmp_path / "cli-report.json"
    assert (
        replay_main(
            [
                str(tmp_path / "trajectory.json"),
                "--cwd",
                str(tmp_path),
                "--output",
                str(output),
                "--pacing",
                "unpaced",
            ]
        )
        == 1
    )
    cli_report = json.loads(output.read_text())
    assert cli_report["summary"]["actions_launch_failed"] == 1
    assert cli_report["actions"][0]["outcome_matches_source"] is None


def test_executed_source_failure_remains_comparable_and_cli_successful(tmp_path):
    from offline_replayer.adapters.atif_replay import main as replay_main

    trajectory = _write_trajectory(
        tmp_path / "trajectory.json",
        [
            ("exec_command", {"cmd": "exit 1"}, "Process exited with code 1"),
        ],
    )
    output = tmp_path / "cli-report.json"
    assert replay_main([str(trajectory), "--cwd", str(tmp_path), "--output", str(output), "--pacing", "unpaced"]) == 0
    report = json.loads(output.read_text())
    assert report["summary"]["actions_launch_failed"] == 0
    assert report["summary"]["actions_executed"] == 1
    assert report["actions"][0]["outcome_matches_source"] is True


def test_replay_trajectory_uses_login_shell_for_codex_exec_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv_seen: list[list[str]] = []

    class FakeProcess:
        pid = 123
        returncode = 0

        def communicate(self, timeout=None):
            return "", ""

    def fake_popen(argv, **_kwargs):
        argv_seen.append(argv)
        return FakeProcess()

    monkeypatch.setattr("offline_replayer.adapters.atif_replay.subprocess.Popen", fake_popen)
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("exec_command", {"cmd": "true"}, "Process exited with code 0")],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert argv_seen == [["/bin/bash", "-lc", "true"]]
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_normalizes_mini_swe_bash_actions_with_provenance(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "bash",
                {"command": "printf done > result.txt"},
                _mini_swe_observation(0),
            ),
            (
                "bash",
                {"command": "printf skipped > skipped.txt"},
                _mini_swe_observation(-1, exception_info="action was not executed"),
            ),
        ],
        agent={
            "name": "mini-swe-agent",
            "extra": {"original_format": "mini-swe-agent-1.1"},
        },
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["tool_actions_recorded"] == 1
    assert report["summary"]["tool_actions_replayed"] == 1
    assert _atif_summary_is_valid(report["summary"]) is True

    assert trajectory.source_agent_original_format == "mini-swe-agent-1.1"
    assert trajectory.actions[0].tool == "exec_command"
    assert trajectory.actions[0].source_tool == "bash"
    assert trajectory.actions[0].arguments["cmd"] == "printf done > result.txt"
    assert (tmp_path / "result.txt").read_text() == "done"
    assert not (tmp_path / "skipped.txt").exists()
    assert report["source_agent_original_format"] == "mini-swe-agent-1.1"
    assert report["actions"][0]["tool"] == "exec_command"
    assert report["actions"][0]["source_tool"] == "bash"
    assert report["actions"][1]["status"] == "skipped_not_executed"
    assert report["summary"]["actions_executed"] == 1
    assert report["summary"]["actions_skipped_not_executed"] == 1
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_captures_full_combined_mini_swe_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("bash", {"command": "printf out; printf err >&2"}, _mini_swe_observation(0, output="outerr"))],
        agent={
            "name": "mini-swe-agent",
            "extra": {"original_format": "mini-swe-agent-1.1"},
        },
    )

    report = replay_trajectory(
        trajectory,
        cwd=tmp_path,
        pacing="unpaced",
        action_output_dir=output_dir,
    )

    output_path = output_dir / "action-0001.output.txt"
    assert output_path.read_text() == "outerr"
    assert report["actions"][0]["output_path"] == str(output_path)
    assert report["actions"][0]["stdout_preview"] == "outerr"
    assert report["actions"][0]["stderr_preview"] == ""


def test_replay_trajectory_uses_posix_shell_for_mini_swe_bash_actions(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "bash",
                {"command": "echo <(true)"},
                _mini_swe_observation(2),
            )
        ],
        agent={
            "name": "mini-swe-agent",
            "extra": {"original_format": "mini-swe-agent-1.1"},
        },
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["actions"][0]["status"] == "failed"
    assert report["actions"][0]["returncode"] == 2
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_does_not_accept_bare_bash_as_generic_atif(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("bash", {"command": "true"}, "Process exited with code 0")],
    )

    with pytest.raises(ReplayError, match="unsupported executable tool calls"):
        replay_trajectory(trajectory, cwd=tmp_path)


def test_claude_read_honors_window_and_rejects_symlink_escape(tmp_path: Path) -> None:
    (tmp_path / "state.txt").write_text("first\nsecond\nthird\n")
    action = AtifAction(
        1, 1, 0.0, "claude_read", "Read", "claude-code", {"file_path": "state.txt", "offset": 2, "limit": 1}, None, ""
    )
    result = _claude_read(action, cwd=tmp_path, patch_roots=())
    assert {key: result[key] for key in ("offset", "limit", "lines_read", "bytes_read")} == {
        "offset": 2,
        "limit": 1,
        "lines_read": 1,
        "bytes_read": 7,
    }
    outside = tmp_path.parent / "claude-read-outside.txt"
    outside.write_text("outside\n")
    (tmp_path / "outside-link").symlink_to(outside)
    action.arguments["file_path"] = "outside-link"
    with pytest.raises(ReplayError, match="permitted replay patch roots"):
        _claude_read(action, cwd=tmp_path, patch_roots=())


def test_claude_glob_sorts_mtime_caps_and_rejects_symlink_escape(tmp_path: Path) -> None:
    for index in range(101):
        path = tmp_path / f"match-{index:03}.txt"
        path.write_text(str(index))
        path.touch(exist_ok=True)
    action = AtifAction(1, 1, 0.0, "claude_glob", "Glob", "claude-code", {"pattern": "*.txt"}, None, "")
    result = _claude_glob(action, cwd=tmp_path, patch_roots=())
    assert len(result["matches"]) == 100
    assert result["truncated"] is True
    outside = tmp_path.parent / "claude-glob-outside.txt"
    outside.write_text("outside")
    (tmp_path / "outside-link.txt").symlink_to(outside)
    with pytest.raises(ReplayError, match="permitted replay patch roots"):
        _claude_glob(action, cwd=tmp_path, patch_roots=())


def test_claude_grep_modes_offsets_and_invalid_regex(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("match one\nmiss\nmatch two\n")
    (tmp_path / "two.py").write_text("match three\n")
    action = AtifAction(
        1,
        1,
        0.0,
        "claude_grep",
        "Grep",
        "claude-code",
        {"pattern": "match", "glob": "*.py", "output_mode": "content"},
        None,
        "",
    )
    result = _claude_grep(action, cwd=tmp_path, patch_roots=())
    assert result["matches"] == 3
    action.arguments.update(offset=1, head_limit=1)
    assert _claude_grep(action, cwd=tmp_path, patch_roots=())["matches"] == 1
    action.arguments.pop("offset")
    action.arguments.pop("head_limit")
    action.arguments["output_mode"] = "count"
    count = _claude_grep(action, cwd=tmp_path, patch_roots=())
    assert (count["matches"], count["total_matches"]) == (2, 3)
    action.arguments["pattern"] = "["
    assert _claude_grep(action, cwd=tmp_path, patch_roots=())["status"] == "failed"


def test_claude_grep_case_insensitive_option_is_applied_and_typed(tmp_path: Path) -> None:
    (tmp_path / "code.txt").write_text("HELLO\n")
    action = AtifAction(
        1,
        1,
        0.0,
        "claude_grep",
        "Grep",
        "claude-code",
        {"pattern": "hello", "path": "code.txt", "output_mode": "content", "-i": True},
        None,
        "",
    )
    assert _claude_grep(action, cwd=tmp_path, patch_roots=())["matches"] == 1
    action.arguments["-i"] = False
    assert _claude_grep(action, cwd=tmp_path, patch_roots=())["matches"] == 0
    action.arguments["-i"] = "true"
    with pytest.raises(ReplayError, match="-i must be a boolean"):
        _claude_grep(action, cwd=tmp_path, patch_roots=())


def test_claude_code_background_bash_requires_structured_task_id(tmp_path: Path) -> None:
    path = tmp_path / "background.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "agent": {"name": "claude-code"},
                "steps": [
                    {
                        "timestamp": "2026-07-08T03:15:01Z",
                        "tool_calls": [
                            {
                                "tool_call_id": "bash",
                                "function_name": "Bash",
                                "arguments": {"command": "sleep 1", "run_in_background": True},
                            }
                        ],
                        "observation": {"results": [{"source_call_id": "bash", "content": ""}]},
                    }
                ],
            }
        )
    )
    with pytest.raises(ReplayError, match="backgroundTaskId"):
        read_trajectory(path)


@pytest.fixture
def claude_tmp_path():
    with tempfile.TemporaryDirectory(prefix="claude-", dir="/tmp") as directory:
        yield Path(directory)


def test_claude_code_rejects_dangerously_unsandboxed_bash(tmp_path: Path) -> None:
    path = tmp_path / "unsandboxed.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "agent": {"name": "claude-code"},
                "steps": [
                    {
                        "timestamp": "2026-07-08T03:15:01Z",
                        "tool_calls": [
                            {
                                "tool_call_id": "bash",
                                "function_name": "Bash",
                                "arguments": {"command": "true", "dangerouslyDisableSandbox": True},
                            }
                        ],
                        "observation": {"results": [{"source_call_id": "bash", "content": ""}]},
                    }
                ],
            }
        )
    )
    with pytest.raises(ReplayError, match="dangerouslyDisableSandbox"):
        read_trajectory(path)


@pytest.mark.parametrize("argument_name", ["input", "patch"])
def test_replay_trajectory_applies_add_and_update_patch(tmp_path: Path, argument_name: str) -> None:
    patch = """*** Begin Patch
*** Add File: added.txt
+before
*** Update File: original.txt
@@
-old
+new
*** End Patch"""
    (tmp_path / "original.txt").write_text("old\n")
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("apply_patch", {argument_name: patch}, "Success. Updated files.")],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert (tmp_path / "added.txt").read_text() == "before\n"
    assert (tmp_path / "original.txt").read_text() == "new\n"
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_allows_explicit_additional_patch_root(tmp_path: Path) -> None:
    cwd = tmp_path / "app"
    scratch = tmp_path / "scratch"
    cwd.mkdir()
    scratch.mkdir()
    patch = f"""*** Begin Patch
*** Add File: {scratch / 'generated.txt'}
+scratch
*** End Patch"""
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("apply_patch", {"input": patch}, "Success. Updated files.")],
    )

    restricted = replay_trajectory(trajectory, cwd=cwd, pacing="unpaced")
    assert restricted["summary"]["source_outcome_mismatch_count"] == 1

    report = replay_trajectory(trajectory, cwd=cwd, pacing="unpaced", patch_roots=(scratch,))

    assert (scratch / "generated.txt").read_text() == "scratch\n"
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_timeout_terminates_foreground_process_group(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "sleep 30 & echo $! > child.pid; wait"},
                "Process exited with code 0",
            )
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=0.05)

    child_pid = int((tmp_path / "child.pid").read_text())
    for _ in range(100):
        if not Path(f"/proc/{child_pid}").exists():
            break
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()
    assert report["summary"]["actions_timed_out"] == 1
    assert report["summary"]["actions_skipped_after_timeout"] == 0
    assert report["summary"]["aborted_after_timeout_action_index"] == 1


def test_replay_does_not_wait_for_background_descendant_output_handles(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "sleep 3 & printf '%s\n' \"$!\" > background.pid"},
                "Process exited with code 0",
            ),
            (
                "exec_command",
                {"cmd": "kill $(cat background.pid)"},
                "Process exited with code 0",
            ),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=0.5)

    assert report["actions"][0]["status"] == "ok"
    assert report["actions"][0]["wall_time_sec"] < 1.0
    assert report["actions"][1]["status"] == "ok"
    assert report["summary"]["actions_timed_out"] == 0
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_timeout_kills_term_ignoring_descendant_with_closed_pipes(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {
                    "cmd": (
                        "(trap '' TERM; while :; do sleep 30; done) "
                        "</dev/null >/dev/null 2>&1 & "
                        "echo $! > child.pid; wait"
                    )
                },
                "Process exited with code 0",
            )
        ],
    )
    report = None
    child_pid: int | None = None
    try:
        report = replay_trajectory(
            trajectory,
            cwd=tmp_path,
            pacing="unpaced",
            timeout_sec=0.05,
        )
        child_pid = int((tmp_path / "child.pid").read_text())
        for _ in range(100):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.01)

        assert not Path(f"/proc/{child_pid}").exists()
        assert report["summary"]["actions_timed_out"] == 1
    finally:
        root_pid: int | None = None
        if report is not None:
            actions = report.get("actions")
            if isinstance(actions, list) and actions and isinstance(actions[0], dict):
                candidate = actions[0].get("process_root_pid")
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    root_pid = candidate
        child_pid_path = tmp_path / "child.pid"
        if child_pid is None and child_pid_path.exists():
            try:
                child_pid = int(child_pid_path.read_text())
            except ValueError:
                pass
        process_group_id = root_pid
        if process_group_id is None and child_pid is not None:
            try:
                candidate_group = os.getpgid(child_pid)
            except ProcessLookupError:
                candidate_group = None
            if candidate_group is not None and candidate_group != os.getpgrp():
                process_group_id = candidate_group
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_replay_timeout_kills_setsid_term_ignoring_descendant(
    tmp_path: Path,
) -> None:
    command = (
        "python3 -c 'import os, signal, time; "
        "os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        'open("setsid-child.pid", "w").write(str(os.getpid())); '
        "time.sleep(30)' </dev/null >/dev/null 2>&1 & wait"
    )
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("exec_command", {"cmd": command}, "Process exited with code 0")],
    )
    child_pid: int | None = None
    try:
        report = replay_trajectory(
            trajectory,
            cwd=tmp_path,
            pacing="unpaced",
            timeout_sec=0.5,
        )
        child_pid = int((tmp_path / "setsid-child.pid").read_text())
        for _ in range(100):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.01)

        assert not Path(f"/proc/{child_pid}").exists()
        assert report["actions"][0]["status"] == "timed_out"
        assert report["summary"]["actions_timed_out"] == 1
    finally:
        child_pid_path = tmp_path / "setsid-child.pid"
        if child_pid is None and child_pid_path.exists():
            try:
                child_pid = int(child_pid_path.read_text())
            except ValueError:
                pass
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_replay_stops_after_first_timeout(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            ("exec_command", {"cmd": "sleep 30"}, "Process exited with code 0"),
            (
                "exec_command",
                {"cmd": "printf should-not-run > later.txt"},
                "Process exited with code 0",
            ),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=0.05)

    assert report["summary"]["tool_actions_recorded"] == 2
    assert report["summary"]["tool_actions_replayed"] == 1
    assert _atif_summary_is_valid(report["summary"]) is False

    assert not (tmp_path / "later.txt").exists()
    assert report["summary"]["actions_executed"] == 1
    assert report["summary"]["actions_timed_out"] == 1
    assert report["summary"]["actions_skipped_after_timeout"] == 1
    assert report["summary"]["aborted_after_timeout_action_index"] == 1


@pytest.mark.parametrize("tty", [False, True])
def test_replay_trajectory_supports_interactive_command_sessions(tmp_path: Path, tty: bool) -> None:
    trajectory = read_trajectory(
        _write_trajectory(
            tmp_path / "trajectory.json",
            [
                (
                    "exec_command",
                    {
                        "cmd": "read value; printf '%s' \"$value\" > interactive.txt",
                        "tty": tty,
                        "yield_time_ms": 1,
                    },
                    "Process running with session ID 44",
                ),
                (
                    "write_stdin",
                    {"session_id": 44, "chars": "done\n"},
                    "Process exited with code 0",
                ),
            ],
        )
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=2)

    assert (tmp_path / "interactive.txt").read_text() == "done"
    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert report["actions"][0]["process_root_pid"] == report["actions"][1]["process_root_pid"]


def test_write_stdin_observation_expiry_keeps_session_and_replay_continues(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {
                    "cmd": (
                        "(trap '' TERM; while :; do sleep 30; done) "
                        "</dev/null >/dev/null 2>&1 & "
                        "echo $! > child.pid; read value; wait"
                    ),
                    "yield_time_ms": 200,
                },
                "Process running with session ID 44",
            ),
            (
                "write_stdin",
                {"session_id": 44, "chars": "done\n", "yield_time_ms": 50},
                "Process exited with code 0",
            ),
            (
                "exec_command",
                {"cmd": "touch continued-after-session-observation"},
                "Process exited with code 0",
            ),
        ],
    )
    report = None
    child_pid: int | None = None
    try:
        report = replay_trajectory(
            trajectory,
            cwd=tmp_path,
            pacing="unpaced",
            timeout_sec=0.5,
        )
        child_pid = int((tmp_path / "child.pid").read_text())
        for _ in range(100):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.01)

        assert not Path(f"/proc/{child_pid}").exists()
        assert report["actions"][1]["status"] == "running"
        assert report["summary"]["actions_timed_out"] == 0
        assert report["summary"]["actions_executed"] == 3
        assert (tmp_path / "continued-after-session-observation").is_file()
        assert report["actions"][0]["process_root_pid"] == report["actions"][1]["process_root_pid"]
    finally:
        root_pid: int | None = None
        if report is not None:
            actions = report.get("actions")
            if isinstance(actions, list) and actions and isinstance(actions[0], dict):
                candidate = actions[0].get("process_root_pid")
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    root_pid = candidate
        child_pid_path = tmp_path / "child.pid"
        if child_pid is None and child_pid_path.exists():
            try:
                child_pid = int(child_pid_path.read_text())
            except ValueError:
                pass
        process_group_id = root_pid
        if process_group_id is None and child_pid is not None:
            try:
                candidate_group = os.getpgid(child_pid)
            except ProcessLookupError:
                candidate_group = None
            if candidate_group is not None and candidate_group != os.getpgrp():
                process_group_id = candidate_group
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_replay_finalizer_kills_term_ignoring_session_descendant(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {
                    "cmd": (
                        "(trap '' TERM; while :; do sleep 30; done) "
                        "</dev/null >/dev/null 2>&1 & "
                        "echo $! > child.pid; read value"
                    ),
                    "yield_time_ms": 200,
                },
                "Process running with session ID 44",
            )
        ],
    )
    report = None
    child_pid: int | None = None
    try:
        report = replay_trajectory(
            trajectory,
            cwd=tmp_path,
            pacing="unpaced",
            timeout_sec=0.05,
        )
        child_pid = int((tmp_path / "child.pid").read_text())
        for _ in range(100):
            if not Path(f"/proc/{child_pid}").exists():
                break
            time.sleep(0.01)

        assert not Path(f"/proc/{child_pid}").exists()
        assert report["actions"][0]["status"] == "running"
        assert report["summary"]["sessions_terminated_at_end"] == ["44"]
    finally:
        root_pid: int | None = None
        if report is not None:
            actions = report.get("actions")
            if isinstance(actions, list) and actions and isinstance(actions[0], dict):
                candidate = actions[0].get("process_root_pid")
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    root_pid = candidate
        child_pid_path = tmp_path / "child.pid"
        if child_pid is None and child_pid_path.exists():
            try:
                child_pid = int(child_pid_path.read_text())
            except ValueError:
                pass
        process_group_id = root_pid
        if process_group_id is None and child_pid is not None:
            try:
                candidate_group = os.getpgid(child_pid)
            except ProcessLookupError:
                candidate_group = None
            if candidate_group is not None and candidate_group != os.getpgrp():
                process_group_id = candidate_group
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_codex_authoritative_parallel_polls_require_distinct_sessions(
    tmp_path: Path,
) -> None:
    path = _write_trajectory(
        tmp_path / "trajectory.json",
        [("exec", {"input": ""}, None)],
        agent={"name": "codex", "extra": {"originator": "codex_exec"}},
    )
    payload = json.loads(path.read_text())
    payload["extra"] = {
        "riptide_normalization": {
            "kind": "codex-native-sessions-flattened",
            "native_session_count": 1,
            "normalized_session_count": 1,
            "native_executable_event_count": 2,
            "extracted_executable_event_count": 2,
            "outcome_bound_executable_event_count": 2,
            "outcome_unavailable_executable_event_count": 0,
            "unaccounted_executable_events": 0,
        }
    }
    payload["steps"][0]["extra"] = {"riptide_source_trajectory_id": "session"}

    def poll(index: int, session: int) -> dict:
        return {
            "index": index,
            "tool": "write_stdin",
            "arguments": {"session_id": session, "chars": "", "yield_time_ms": 1000},
            "source_item_id": f"poll-{index}",
            "outcome": {"status": "running", "returncode": None, "session_id": str(session)},
            "parallel_group_id": "parallel-polls",
        }

    payload["steps"][0]["tool_calls"][0]["extra"] = {"riptide_nested_invocations": [poll(0, 41), poll(1, 42)]}
    path.write_text(json.dumps(payload))
    trajectory = read_trajectory(path)
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        validate_supported_actions(trajectory)

    payload["steps"][0]["tool_calls"][0]["extra"] = {"riptide_nested_invocations": [poll(0, 41), poll(1, 41)]}
    path.write_text(json.dumps(payload))
    with pytest.raises(ReplayError, match="parallel poll group repeats a session"):
        read_trajectory(path)

    payload["steps"][0]["tool_calls"][0]["extra"] = {
        "riptide_nested_invocations": [
            {
                "index": 0,
                "tool": "exec_command",
                "arguments": {"cmd": "kill 1"},
                "source_item_id": "mixed-exec",
                "outcome": None,
                "parallel_group_id": "mixed-mutation",
            },
            {
                "index": 1,
                "tool": "apply_patch",
                "arguments": {"input": "*** Begin Patch\n*** End Patch"},
                "source_item_id": "mixed-patch",
                "outcome": None,
                "parallel_group_id": "mixed-mutation",
            },
        ]
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(ReplayError, match="unsafe parallel group"):
        read_trajectory(path)


def test_plain_command_without_exit_evidence_is_comparison_unavailable(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("exec_command", {"cmd": "true"}, "command output without an exit status")],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["actions"][0]["outcome_matches_source"] is None
    assert report["summary"]["source_outcome_comparison_unavailable_action_indexes"] == [1]
    assert report["summary"]["source_outcome_comparison_unavailable_count"] == 1


def test_replay_trajectory_accepts_early_success_for_source_running_session(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "true", "yield_time_ms": 100},
                "Process running with session ID 44",
            ),
            ("write_stdin", {"session_id": 44, "chars": ""}, "Process exited with code 0"),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert report["summary"]["actions_completed_before_recorded_poll"] == 1


def test_replay_trajectory_defers_early_failure_to_source_terminal_poll(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "exit 7", "yield_time_ms": 100},
                "Process running with session ID 44",
            ),
            (
                "write_stdin",
                {"session_id": 44, "chars": "", "yield_time_ms": 100},
                "Process running with session ID 44",
            ),
            (
                "write_stdin",
                {"session_id": 44, "chars": ""},
                "Process exited with code 7",
            ),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert report["summary"]["actions_completed_before_recorded_poll"] >= 1


def test_terminal_write_stdin_poll_expiry_is_running_and_replay_continues(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "sleep 0.05", "yield_time_ms": 1},
                "Process running with session ID 44",
            ),
            (
                "write_stdin",
                {"session_id": 44, "chars": "", "yield_time_ms": 1},
                "Process exited with code 0",
            ),
            (
                "exec_command",
                {"cmd": "touch continued-after-poll"},
                "Process exited with code 0",
            ),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=2)

    assert report["actions"][1]["status"] == "running"
    assert report["actions"][1]["outcome_matches_source"] is False
    assert report["summary"]["actions_timed_out"] == 0
    assert report["summary"]["source_outcome_mismatch_count"] == 1
    assert report["summary"]["actions_executed"] == 3
    assert (tmp_path / "continued-after-poll").is_file()


def test_write_stdin_hard_timeout_caps_long_observation_and_aborts(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "sleep 30", "yield_time_ms": 1},
                "Process running with session ID 44",
            ),
            (
                "write_stdin",
                {"session_id": 44, "chars": "", "yield_time_ms": 1000},
                "Process exited with code 0",
            ),
            (
                "exec_command",
                {"cmd": "touch should-not-run-after-write-timeout"},
                "Process exited with code 0",
            ),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=0.05)

    assert report["actions"][1]["status"] == "timed_out"
    assert report["actions"][1]["returncode"] == 124
    assert report["summary"]["actions_timed_out"] == 1
    assert report["summary"]["actions_skipped_after_timeout"] == 1
    assert not (tmp_path / "should-not-run-after-write-timeout").exists()


@pytest.mark.parametrize("tty", [False, True])
def test_write_stdin_ctrl_c_signals_the_contained_session(tmp_path: Path, tty: bool) -> None:
    trajectory = read_trajectory(
        _write_trajectory(
            tmp_path / "trajectory.json",
            [
                (
                    "exec_command",
                    {
                        "cmd": "sleep 30",
                        "tty": tty,
                        "yield_time_ms": 100,
                    },
                    "Process running with session ID 44",
                ),
                (
                    "write_stdin",
                    {"session_id": 44, "chars": "\x03", "yield_time_ms": 100},
                    "Process exited with code 130",
                ),
            ],
        )
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=2)

    assert report["actions"][1]["status"] == "failed"
    assert report["actions"][1]["returncode"] == 130
    assert report["actions"][1]["outcome_matches_source"] is True
    assert report["summary"]["actions_timed_out"] == 0
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_running_write_stdin_retains_single_poll_behavior(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "sleep 0.05", "yield_time_ms": 1},
                "Process running with session ID 44",
            ),
            (
                "write_stdin",
                {"session_id": 44, "chars": "", "yield_time_ms": 1},
                "Process running with session ID 44",
            ),
            (
                "write_stdin",
                {"session_id": 44, "chars": ""},
                "Process exited with code 0",
            ),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", timeout_sec=2)

    assert report["actions"][1]["status"] == "running"
    assert report["actions"][2]["status"] == "ok"
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_accepts_generic_source_failure_code(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("exec_command", {"cmd": "exit 23"}, "Process exited with code -1")],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_scrubs_controller_only_environment_from_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONHOME", "/tasktrace-controller-runtime")
    monkeypatch.setenv("PYTHONPATH", "/tasktrace-source")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tasktrace-controller-runtime/lib")
    monkeypatch.setenv("TASKTRACE_WORKLOAD_ENV_PYTHONPATH", "/app/lib:/app")
    monkeypatch.setenv("KEEP_VISIBLE", "yes")
    monkeypatch.setenv("TASKTRACE_CONTROLLER_ENV_VARS", "PYTHONHOME:PYTHONPATH:LD_LIBRARY_PATH")
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {
                    "cmd": (
                        "test -z \"$PYTHONHOME\" && test -z \"$LD_LIBRARY_PATH\" "
                        "&& test \"$PYTHONPATH\" = /app/lib:/app "
                        "&& test \"$KEEP_VISIBLE\" = yes "
                        "&& test -z \"$TASKTRACE_WORKLOAD_ENV_PYTHONPATH\""
                    )
                },
                "Process exited with code 0",
            )
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_workload_environment_preserves_empty_and_unset_values_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tasktrace-source")
    monkeypatch.setenv("PYTHONHOME", "/tasktrace-controller-runtime")
    monkeypatch.setenv("TASKTRACE_CONTROLLER_ENV_VARS", "PYTHONPATH:PYTHONHOME")
    monkeypatch.setenv("TASKTRACE_WORKLOAD_ENV_PYTHONPATH", "")
    monkeypatch.setenv("TASKTRACE_WORKLOAD_ENV_PRESENT_PYTHONPATH", "1")
    monkeypatch.setenv("TASKTRACE_WORKLOAD_ENV_PYTHONHOME", "")
    monkeypatch.setenv("TASKTRACE_WORKLOAD_ENV_PRESENT_PYTHONHOME", "")

    environment = _workload_environment()

    assert environment is not None
    assert environment["PYTHONPATH"] == ""
    assert "PYTHONHOME" not in environment
    assert not any(name.startswith("TASKTRACE_WORKLOAD_ENV_") for name in environment)


def test_replay_trajectory_records_expected_missing_workdir_launch_failure(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "created-by-command"
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": f"mkdir -p {missing}", "workdir": str(missing)},
                'exec_command failed for `/bin/bash -lc "mkdir"`: CreateProcess failed',
            ),
            ("exec_command", {"cmd": "true"}, "Process exited with code 0"),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["tool_actions_recorded"] == 2
    assert report["summary"]["tool_actions_replayed"] == 1
    assert _atif_summary_is_valid(report["summary"]) is False

    assert report["actions"][0]["status"] == "failed"
    assert report["actions"][0]["launch_failed"] is True
    assert report["actions"][1]["status"] == "ok"
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_preserves_source_argument_parse_failure(
    tmp_path: Path,
) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"not_cmd": "printf should-not-run > malformed.txt"},
                "failed to parse function arguments: missing field `cmd`",
            ),
            (
                "exec_command",
                {"cmd": "printf rejected > rejected.txt", "yield_time_ms": "1000"},
                "failed to parse function arguments: invalid type",
            ),
            (
                "exec_command",
                {"cmd": "printf continued > valid.txt"},
                "Process exited with code 0",
            ),
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["summary"]["tool_actions_recorded"] == 3
    assert report["summary"]["tool_actions_replayed"] == 1
    assert _atif_summary_is_valid(report["summary"]) is False

    assert not (tmp_path / "malformed.txt").exists()
    assert not (tmp_path / "rejected.txt").exists()
    assert (tmp_path / "valid.txt").read_text() == "continued"
    assert report["actions"][0]["status"] == "failed"
    assert report["actions"][0]["command_executed"] is False
    assert report["actions"][0]["source_argument_parse_failed"] is True
    assert report["actions"][1]["source_argument_parse_failed"] is True
    assert report["actions"][2]["status"] == "ok"
    assert report["summary"]["actions_executed"] == 1
    assert report["summary"]["actions_source_argument_parse_failed"] == 2
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_matches_expected_patch_verification_failure(tmp_path: Path) -> None:
    patch = """*** Begin Patch
*** Update File: missing.txt
@@
-old
+new
*** End Patch"""
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "apply_patch",
                {"input": patch},
                "apply_patch verification failed: Failed to find expected lines",
            )
        ],
    )

    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")

    assert report["actions"][0]["status"] == "failed"
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_replay_trajectory_rejects_unsupported_executable_tools(tmp_path: Path) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("functions.view_image", {"path": "x"}, "ok")],
    )

    with pytest.raises(ReplayError, match="unsupported executable tool calls"):
        replay_trajectory(trajectory, cwd=tmp_path)


def _printed_results(*values: object) -> dict[str, object]:
    return {
        "content": [
            {"type": "input_text", "text": "Script completed\nOutput:\n"},
            *({"type": "input_text", "text": json.dumps(value)} for value in values),
        ]
    }


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [0], [-1, 1], [False, 1]])
def test_trajectory_results_reject_unproven_indices(tmp_path: Path, indices: list[int]) -> None:
    source = "const r = await Promise.all([tools.exec_command({cmd:'true'}),tools.exec_command({cmd:'false'})]); r.forEach((r,i)=>{text(JSON.stringify({i,...r}));});"
    with pytest.raises(ReplayError, match="Codex printed"):
        _load_actions(
            tmp_path / "trajectory.json",
            [("exec", {"input": source}, _printed_results(*({"i": i, "exit_code": 0, "output": ""} for i in indices)))],
            agent={"name": "codex", "extra": {"originator": "codex_exec"}},
        )


@pytest.mark.parametrize(
    "source",
    [
        "const r = await tools.exec_command({cmd:'true'}); text(r.output);",
        "text(await tools.exec_command({cmd:'true'})); await tools.exec_command({cmd:'false'});",
    ],
)
def test_trajectory_results_do_not_invent_outcomes_for_partial_display(tmp_path: Path, source: str) -> None:
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [("exec", {"input": source}, _printed_results({"exit_code": 0, "output": "fake result in stdout"}))],
        agent={"name": "codex", "extra": {"originator": "codex_exec"}},
    )
    assert all(action.observation is None for action in trajectory.actions)


@pytest.mark.parametrize(
    "content",
    [
        "[__import__('os').system('false')]",
        "[]; dangerous()",
        "[{'text':`hello ${danger()}`}]",
        "[{'text':'a','text':'b'}]",
        "[" * 130 + "]" * 130,
    ],
)
def test_trajectory_result_data_is_not_executable(tmp_path: Path, content: str) -> None:
    with pytest.raises(ReplayError):
        _load_actions(
            tmp_path / "trajectory.json",
            [("exec", {"input": "text(await tools.exec_command({cmd:'true'}));"}, {"content": content})],
            agent={"name": "codex", "extra": {"originator": "codex_exec"}},
        )


@pytest.mark.parametrize("kind", ["outside", "traversal", "symlink", "wrong-session", "relative"])
def test_background_log_rejects_unowned_recorded_paths(tmp_path, claude_tmp_path, kind):
    outside = tmp_path / "unowned" / "tasks" / "44.output"
    (claude_tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    recorded = {
        "outside": outside,
        "traversal": claude_tmp_path / ".." / "unowned" / "tasks" / "44.output",
        "symlink": claude_tmp_path / "alias" / "unowned" / "tasks" / "44.output",
        "wrong-session": claude_tmp_path / "tasks" / "other.output",
        "relative": Path("relative/tasks/44.output"),
    }[kind]
    trajectory = _load_actions(
        tmp_path / "trajectory.json",
        [
            (
                "exec_command",
                {"cmd": "true"},
                f"Process running with session ID 44\nOutput is being written to: {recorded}",
            )
        ],
    )
    replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced")
    assert not recorded.is_symlink()
    assert not outside.parent.exists()


@pytest.mark.parametrize(
    "block,timeout,limit,expected",
    [
        (True, None, None, 5.0),
        (True, 1000, None, 1.0),
        (True, None, 0.2, 0.2),
        (True, 1000, 0.2, 0.2),
        (False, None, None, None),
    ],
)
def test_task_output_wait_is_bounded(tmp_path, block, timeout, limit, expected):
    import subprocess
    from unittest.mock import Mock

    from offline_replayer.adapters.atif_replay import (
        _claude_task_output,
        _RunningSession,
    )

    action = _action("")
    action.arguments.update(task_id="task.1", block=block)
    if timeout is not None:
        action.arguments["timeout"] = timeout
    log = tmp_path / "task.log"
    log.write_bytes(b"output")
    process = Mock()
    process.wait.side_effect = subprocess.TimeoutExpired("test", expected)
    process.poll.return_value = None
    session = _RunningSession("task.1", process, log, None)
    result = _claude_task_output(action, {"task.1": session}, limit)
    assert result["status"] == "running"
    if block:
        process.wait.assert_called_once_with(timeout=expected)
        process.poll.assert_not_called()
    else:
        process.wait.assert_not_called()
        process.poll.assert_called_once_with()


def test_claude_replay_is_rejected_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.5",
                "agent": {"name": "claude-code"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": 0,
                        "tool_calls": [
                            {"tool_call_id": "a", "function_name": "Bash", "arguments": {"command": "touch actual"}}
                        ],
                        "observation": {"results": [{"source_call_id": "a", "content": "ok"}]},
                    }
                ],
            }
        )
    )
    trajectory = read_trajectory(path)
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        replay_trajectory(trajectory, cwd=tmp_path, artifacts_dir=artifacts)
    assert not (tmp_path / "actual").exists()
    assert not artifacts.exists()

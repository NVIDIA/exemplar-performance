# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import shutil
import subprocess

import pytest
from tasktrace_capture.shell_capture import (
    CAPTURE_CONTAINER_SCRIPT,
    COMMAND_ENV_NAME,
    MINI_SWE_TRACE_CLASS,
    MINI_SWE_TRACE_MODULE_FILE,
    TRACE_EVENTS_NAME,
    TRACE_PERFETTO_NAME,
    TRACE_PRIOR_BASH_ENV,
    TRACE_PROFILE_LINE,
    TRACE_SCHEMA_VERSION,
    TRACE_TIMINGS_NAME,
    TRAJECTORY_TIMING_FIELDS,
    command_inherits_shell_capture,
    command_starts_login_shell,
    finalize_trajectories,
    install_login_shell_capture_command,
    instrument_agent_launch,
    omit_trajectory_timings,
    parse_trajectory_events,
    prepare_mini_swe_trace_adapter,
    prepare_shell_capture,
    supports_shell_capture_agent,
    with_shell_capture_env,
    write_perfetto_trace,
)


def test_supported_agents_and_command_routes():
    assert supports_shell_capture_agent("terminus-2")
    assert supports_shell_capture_agent("claude-code")
    assert supports_shell_capture_agent("mini-swe-agent")
    assert not supports_shell_capture_agent("codex")
    assert command_inherits_shell_capture("claude --verbose --permission-mode=bypassPermissions", "claude-code")
    assert command_inherits_shell_capture("mini-swe-agent --yolo --task=x", "mini-swe-agent")
    assert command_starts_login_shell("tmux new-session -d -s trial 'bash --login'", "terminus-2")


def test_with_shell_capture_env_points_at_mounted_bootstrap():
    assert with_shell_capture_env({"A": "b"}) == {
        "A": "b",
        "BASH_ENV": CAPTURE_CONTAINER_SCRIPT,
    }


def test_with_shell_capture_env_preserves_existing_bash_bootstrap():
    assert with_shell_capture_env({"BASH_ENV": "/opt/user/bash_env"}) == {
        "BASH_ENV": CAPTURE_CONTAINER_SCRIPT,
        TRACE_PRIOR_BASH_ENV: "/opt/user/bash_env",
    }


def test_trace_script_sources_existing_bash_bootstrap(tmp_path):
    prior = tmp_path / "prior.bash"
    prior.write_text("export TASKTRACE_CAPTURE_TEST_PRIOR_LOADED=1\n")
    script = prepare_shell_capture(tmp_path / "agent")
    events = tmp_path / TRACE_EVENTS_NAME
    env = {
        **os.environ,
        **with_shell_capture_env({"BASH_ENV": str(prior)}),
        "BASH_ENV": str(script),
        "_TASKTRACE_CAPTURE_RAW_TRAJECTORY_FILE": str(events),
    }
    subprocess.run(
        ["bash", "-c", 'test "$TASKTRACE_CAPTURE_TEST_PRIOR_LOADED" = 1'],
        env=env,
        check=True,
    )


def test_mini_swe_launch_uses_bash_backed_environment(tmp_path):
    command, env = instrument_agent_launch(
        "mini-swe-agent --yolo --model=openai/model --task=x --exit-immediately",
        "mini-swe-agent",
        {},
        tmp_path,
    )
    assert 'PYTHONPATH="/logs/agent' in command
    assert f"--environment-class {MINI_SWE_TRACE_CLASS}" in command
    assert env["BASH_ENV"] == CAPTURE_CONTAINER_SCRIPT
    assert (tmp_path / MINI_SWE_TRACE_MODULE_FILE).exists()


def test_mini_swe_trace_environment_executes_tool_action_in_bash(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    script = prepare_shell_capture(agent_dir)
    module_path = prepare_mini_swe_trace_adapter(agent_dir)
    events = tmp_path / TRACE_EVENTS_NAME
    spec = importlib.util.spec_from_file_location("tasktrace_capture_test_miniswe_trace", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("BASH_ENV", str(script))
    monkeypatch.setenv("_TASKTRACE_CAPTURE_RAW_TRAJECTORY_FILE", str(events))

    environment = module.BashTracingLocalEnvironment(timeout=2)
    result = environment.execute({"command": "cd /tmp && sleep 0.02"})

    assert result["returncode"] == 0
    commands = [str(record["command_shell_quoted"]).replace("\\ ", " ") for record in parse_trajectory_events(events)]
    assert "cd /tmp" in commands
    assert "sleep 0.02" in commands


def test_mini_swe_trace_environment_timeout_ends_the_whole_command(tmp_path, monkeypatch):
    module_path = prepare_mini_swe_trace_adapter(tmp_path / "agent")
    spec = importlib.util.spec_from_file_location("tasktrace_capture_test_miniswe_timeout", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "COMMAND_ENV_PATH", str(tmp_path / COMMAND_ENV_NAME))
    pid_file = tmp_path / "child.pid"

    environment = module.BashTracingLocalEnvironment(timeout=1)
    result = environment.execute({"command": f"echo started; sleep 30 & echo $! > {pid_file}; wait"})

    assert result["returncode"] == -1
    assert "timed out after 1 seconds" in result["exception_info"]
    assert "started" in result["output"]
    child = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def test_mini_swe_trace_environment_records_replayable_command_env(tmp_path, monkeypatch):
    module_path = prepare_mini_swe_trace_adapter(tmp_path / "agent")
    spec = importlib.util.spec_from_file_location("tasktrace_capture_test_miniswe_env", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env_path = tmp_path / COMMAND_ENV_NAME
    monkeypatch.setattr(module, "COMMAND_ENV_PATH", str(env_path))
    monkeypatch.setenv("LC_CTYPE", "C.UTF-8")
    monkeypatch.setenv("PYTHONPATH", "/logs/agent:/opt/lib")
    monkeypatch.setenv("BASH_ENV", "/logs/agent/tasktrace_capture_shell_instrument.bash")
    monkeypatch.setenv("_TASKTRACE_CAPTURE_PREVIOUS_BASH_ENV", "/root/.bashrc")
    for name in ("OPENAI_API_KEY", "OPENAI_API_BASE", "MSWEA_CONFIGURED", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "secret-value")

    environment = module.BashTracingLocalEnvironment(timeout=2, env={"PAGER": "cat"})
    environment.execute({"command": "true"})
    environment.execute({"command": "true"})

    recorded = json.loads(env_path.read_text())
    assert recorded["LC_CTYPE"] == "C.UTF-8"
    assert recorded["PAGER"] == "cat"
    assert recorded["PYTHONPATH"] == "/opt/lib"
    assert recorded["PATH"] == os.environ["PATH"]
    assert "secret-value" not in env_path.read_text()
    assert not {"BASH_ENV", "_TASKTRACE_CAPTURE_PREVIOUS_BASH_ENV"} & recorded.keys()


def test_trace_script_splits_compound_bash_commands(tmp_path):
    script = prepare_shell_capture(tmp_path / "agent")
    events = tmp_path / TRACE_EVENTS_NAME
    env = os.environ.copy()
    env["BASH_ENV"] = str(script)
    env["_TASKTRACE_CAPTURE_RAW_TRAJECTORY_FILE"] = str(events)

    subprocess.run(
        ["bash", "-c", "cd /tmp && sleep 0.02 && printf done >/dev/null"],
        env=env,
        check=True,
    )

    records = parse_trajectory_events(events)
    readable_commands = [str(record["command_shell_quoted"]).replace("\\ ", " ") for record in records]
    assert any(command == "cd /tmp" for command in readable_commands)
    sleep_record = next(
        record for record, command in zip(records, readable_commands, strict=False) if command == "sleep 0.02"
    )
    assert sleep_record["wall_time_sec"] >= 0.01


def test_finalize_trajectories_writes_jsonl(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t1.000000\t12\t1\t1\t-\tcd\\ /app\n"
        "end\t1.010000\t12\t1\t1\t0\tcd\\ /app\n"
        "begin\t1.010000\t12\t1\t1\t-\tsed\\ -n\\ 1p\\ x.py\n"
        "end\t1.210000\t12\t1\t1\t0\tsed\\ -n\\ 1p\\ x.py\n"
    )

    outputs = finalize_trajectories(tmp_path / "job")

    assert outputs == [agent_dir / TRACE_TIMINGS_NAME]
    rows = [json.loads(line) for line in outputs[0].read_text().splitlines()]
    header = next(row for row in rows if row["record_type"] == "trace")
    commands = [row for row in rows if row["record_type"] == "command"]
    assert header["schema_version"] == TRACE_SCHEMA_VERSION
    assert [row["wall_time_sec"] for row in commands] == [0.01, 0.2]
    assert commands[1]["command_shell_quoted"].startswith("sed")
    trace = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())
    command_events = [event for event in trace["traceEvents"] if event.get("cat") == "bash_command"]
    assert [event["name"] for event in command_events] == [
        "cd\\ /app",
        "sed\\ -n\\ 1p\\ x.py",
    ]


def test_finalize_trajectories_can_omit_the_timings(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t1.000000\t12\t1\t1\t-\tcd\\ /app\n"
        "end\t1.010000\t12\t1\t1\t0\tcd\\ /app\n"
        "begin\t1.010000\t12\t1\t1\t-\tsed\\ -n\\ 1p\\ x.py\n"
        "end\t1.210000\t12\t1\t1\t0\tsed\\ -n\\ 1p\\ x.py\n"
    )

    outputs = finalize_trajectories(tmp_path / "job", trajectory_timings="omit")

    rows = [json.loads(line) for line in outputs[0].read_text().splitlines()]
    assert not any(field in row for row in rows for field in TRAJECTORY_TIMING_FIELDS)
    assert [row["command_shell_quoted"] for row in rows if row["record_type"] == "command"][1].startswith("sed")
    assert not (agent_dir / TRACE_PERFETTO_NAME).exists()  # the timeline is the timings
    with pytest.raises(ValueError, match="trajectory_timings"):
        finalize_trajectories(tmp_path / "job", trajectory_timings="sometimes")


def test_omit_trajectory_timings_keeps_the_agent_loop_order():
    rows = [
        {"record_type": "trace", "schema_version": TRACE_SCHEMA_VERSION},
        {
            "record_type": "phase",
            "name": "setup",
            "started_at_epoch": 0.5,
            "finished_at_epoch": 0.9,
            "wall_time_sec": 0.4,
        },
        {
            "record_type": "action",
            "action_index": 1,
            "started_at_epoch": 1.0,
            "finished_at_epoch": 1.1,
            "wall_time_sec": 0.1,
        },
        {"record_type": "llm", "llm_index": 1, "prompt_tokens": 10, "started_at_epoch": 1.2, "wall_time_sec": 0.8},
        {
            "record_type": "action",
            "action_index": 2,
            "started_at_epoch": 2.1,
            "finished_at_epoch": 2.2,
            "wall_time_sec": 0.1,
        },
        {"record_type": "llm", "llm_index": 2, "prompt_tokens": 20, "started_at_epoch": 2.3, "wall_time_sec": 0.5},
        {
            "record_type": "action",
            "action_index": 3,
            "started_at_epoch": 2.9,
            "finished_at_epoch": 3.0,
            "wall_time_sec": 0.1,
        },
    ]
    stripped = omit_trajectory_timings(rows)
    assert not any(field in row for row in stripped for field in TRAJECTORY_TIMING_FIELDS)
    assert [row["after_llm_index"] for row in stripped if row["record_type"] == "action"] == [None, 1, 2]
    assert stripped[3] == {"record_type": "llm", "llm_index": 1, "prompt_tokens": 10}
    assert "started_at_epoch" in rows[2]  # the input is not modified


def test_perfetto_command_lane_excludes_bootstrap_and_nested_shells(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t1.000000\t10\t0\t1\t-\ttrap\\ handler\n"
        "end\t1.010000\t10\t0\t1\t0\ttrap\\ handler\n"
        "begin\t2.000000\t12\t7\t1\t-\tmake\n"
        "end\t4.000000\t12\t7\t1\t0\tmake\n"
        "begin\t2.100000\t13\t12\t2\t-\tcc\\ source.c\n"
        "end\t3.900000\t13\t12\t2\t0\tcc\\ source.c\n"
    )

    output = finalize_trajectories(tmp_path / "job")[0]
    trace = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())
    command_events = [event for event in trace["traceEvents"] if event.get("cat") == "bash_command"]

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len([row for row in rows if row["record_type"] == "command"]) == 3
    assert [event["name"] for event in command_events] == ["make"]


def test_perfetto_max_depth_renders_nested_shell_processes(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t1.000000\t10\t0\t1\t-\tagent-launch\n"
        "end\t1.010000\t10\t0\t1\t0\tagent-launch\n"
        "begin\t2.000000\t12\t7\t1\t-\tmake\n"
        "end\t4.000000\t12\t7\t1\t0\tmake\n"
        "begin\t2.100000\t13\t12\t2\t-\tcc\\ source.c\n"
        "end\t3.900000\t13\t12\t2\t0\tcc\\ source.c\n"
    )

    finalize_trajectories(tmp_path / "job", perfetto_depth="max")
    trace = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())
    command_events = [event for event in trace["traceEvents"] if event.get("cat") == "bash_command"]
    thread_names = {event["args"]["name"] for event in trace["traceEvents"] if event.get("name") == "thread_name"}

    assert {event["name"] for event in command_events} == {
        "agent-launch",
        "make",
        "cc\\ source.c",
    }
    assert "Direct Bash commands" in thread_names
    assert "Harness/startup Bash" in thread_names
    assert "Bash depth 1: pid 12" not in thread_names
    assert "Bash depth 1: pid 10" not in thread_names
    assert "Bash depth 2: pid 13" in thread_names
    assert trace["metadata"]["perfetto_bash_depth"] == "max"


def test_trace_script_and_perfetto_capture_real_bash_backed_make_fanout(tmp_path):
    if shutil.which("make") is None:
        pytest.skip("make is required for recursive shell trace coverage")

    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    build_dir = tmp_path / "build"
    sub_dir = build_dir / "sub"
    sub_dir.mkdir(parents=True)
    (build_dir / "Makefile").write_text(
        "SHELL := /bin/bash\n"
        "export BASH_ENV\n"
        ".PHONY: all child\n"
        "all: child\n"
        "child:\n"
        "\t$(MAKE) --no-print-directory -C sub all\n"
    )
    (sub_dir / "Makefile").write_text(
        "SHELL := /bin/bash\n" "export BASH_ENV\n" ".PHONY: all\n" "all:\n" "\tprintf 'nested action\\n' > output.txt\n"
    )
    script = prepare_shell_capture(agent_dir)
    events = agent_dir / TRACE_EVENTS_NAME
    env = {
        **os.environ,
        "BASH_ENV": str(script),
        "_TASKTRACE_CAPTURE_RAW_TRAJECTORY_FILE": str(events),
    }

    subprocess.run(
        ["bash", "-c", "make --no-print-directory all"],
        cwd=build_dir,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    timings = finalize_trajectories(tmp_path / "job")[0]
    shallow = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())
    maximum_path = agent_dir / "depth-max.perfetto.json"
    write_perfetto_trace(timings, perfetto_depth="max", output_path=maximum_path)
    maximum = json.loads(maximum_path.read_text())
    shallow_names = [event["name"] for event in shallow["traceEvents"] if event.get("cat") == "bash_command"]
    maximum_names = [event["name"] for event in maximum["traceEvents"] if event.get("cat") == "bash_command"]

    assert shallow_names == ["make\\ --no-print-directory\\ all"]
    assert "make\\ --no-print-directory\\ -C\\ sub\\ all" in maximum_names
    assert any(name.startswith("printf\\ ") for name in maximum_names)


def test_parse_trajectory_does_not_guess_end_for_unmatched_begin(tmp_path):
    events = tmp_path / TRACE_EVENTS_NAME
    events.write_text("begin\t1.000000\t12\t1\t1\t-\tmake\\ -j24\n")

    assert parse_trajectory_events(events) == []


def test_finalize_trajectories_bounds_interrupted_mini_swe_command(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text("begin\t10.500000\t12\t1\t1\t-\tmake\\ -j24\n")
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "assistant", "extra": {"timestamp": 10.0}},
                    {"role": "tool", "extra": {"timestamp": 40.0}},
                ]
            }
        )
    )

    output = finalize_trajectories(tmp_path / "job")[0]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    row = next(row for row in rows if row["record_type"] == "command")

    assert row["wall_time_sec"] == 29.5
    assert row["finished_at_epoch"] == 40.0
    assert row["exit_status"] is None
    assert row["incomplete"] is True
    assert row["termination_reason"] == "tool_call_ended_without_shell_exit"
    trace = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())
    command_event = next(event for event in trace["traceEvents"] if event.get("cat") == "bash_command")
    assert command_event["dur"] == 29_500_000
    assert command_event["args"]["incomplete"] is True
    assert any(event.get("name") == "Interrupted command" for event in trace["traceEvents"])


def test_finalize_trajectories_perfetto_enriches_mini_swe_timeline(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t3.100000\t12\t1\t1\t-\techo\\ ok\n" "end\t3.200000\t12\t1\t1\t0\techo\\ ok\n"
    )
    (agent_dir.parent / "result.json").write_text(
        json.dumps(
            {
                "task_name": "compilebench/jq",
                "environment_setup": {
                    "started_at": "1970-01-01T00:00:00Z",
                    "finished_at": "1970-01-01T00:00:01Z",
                },
                "agent_setup": {
                    "started_at": "1970-01-01T00:00:01Z",
                    "finished_at": "1970-01-01T00:00:02Z",
                },
                "agent_execution": {
                    "started_at": "1970-01-01T00:00:02Z",
                    "finished_at": "1970-01-01T00:00:08Z",
                },
                "verifier": {
                    "started_at": "1970-01-01T00:00:08Z",
                    "finished_at": "1970-01-01T00:00:09Z",
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "timestamp": 3.0,
                            "actions": [{"command": "echo ok"}],
                        },
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": 0}),
                        "extra": {"timestamp": 3.5},
                    },
                    {
                        "role": "assistant",
                        "content": "Running the next shell action.",
                        "extra": {
                            "timestamp": 5.0,
                            "actions": [{"command": "true"}],
                            "cost": 0.012,
                            "response": {
                                "model": "test-model",
                                "choices": [{"finish_reason": "tool_calls"}],
                                "usage": {
                                    "prompt_tokens": 100,
                                    "completion_tokens": 12,
                                    "total_tokens": 112,
                                    "prompt_tokens_details": {"cached_tokens": 64},
                                    "completion_tokens_details": {"reasoning_tokens": 5},
                                },
                            },
                        },
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": 0}),
                        "extra": {"timestamp": 5.1},
                    },
                ]
            }
        )
    )

    finalize_trajectories(tmp_path / "job")
    rows = [json.loads(line) for line in (agent_dir / TRACE_TIMINGS_NAME).read_text().splitlines()]
    trace = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())

    assert any(row["record_type"] == "phase" for row in rows)
    assert any(row["record_type"] == "action" for row in rows)
    assert any(row["record_type"] == "llm" for row in rows)
    assert any(row["record_type"] == "unclassified" for row in rows)
    thread_names = {event["args"]["name"] for event in trace["traceEvents"] if event.get("name") == "thread_name"}
    assert "TaskTrace Capture phases" in thread_names
    assert "Harbor phases" not in thread_names
    categories = {event.get("cat") for event in trace["traceEvents"]}
    assert {"phase", "tool_envelope", "llm", "unclassified", "bash_command"} <= categories
    llm_event = next(event for event in trace["traceEvents"] if event.get("cat") == "llm")
    finalization_event = next(event for event in trace["traceEvents"] if event.get("cat") == "unclassified")
    assert llm_event["dur"] == 1_500_000
    assert llm_event["args"]["response_kind"] == "text_and_bash_tool_call"
    assert llm_event["args"]["assistant_text"] == "Running the next shell action."
    assert llm_event["args"]["command_sent_by_llm"] == "true"
    assert llm_event["args"]["model"] == "test-model"
    assert llm_event["args"]["finish_reason"] == "tool_calls"
    assert llm_event["args"]["prompt_tokens"] == 100
    assert llm_event["args"]["cached_prompt_tokens"] == 64
    assert llm_event["args"]["uncached_prompt_tokens"] == 36
    assert llm_event["args"]["completion_tokens"] == 12
    assert llm_event["args"]["reasoning_tokens"] == 5
    assert llm_event["args"]["total_tokens"] == 112
    assert llm_event["args"]["cost"] == 0.012
    assert finalization_event["name"] == "Agent finalization (unclassified)"
    assert finalization_event["dur"] == 2_900_000
    assert trace["metadata"]["reward"] == 1.0


def test_trace_header_warns_about_failed_timed_out_and_unexecuted_actions(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text("")
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {"timestamp": 1.0, "actions": [{"command": "false"}]},
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": 2}),
                        "extra": {"timestamp": 1.1},
                    },
                    {
                        "role": "assistant",
                        "extra": {
                            "timestamp": 2.0,
                            "actions": [{"command": "sleep 60"}],
                        },
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": -1, "exception_info": "command timed out"}),
                        "extra": {"timestamp": 32.0},
                    },
                    {
                        "role": "assistant",
                        "extra": {
                            "timestamp": 33.0,
                            "actions": [{"command": "echo never"}],
                        },
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": -1, "exception_info": "action was not executed"}),
                        "extra": {"timestamp": 33.1},
                    },
                ]
            }
        )
    )

    output = finalize_trajectories(tmp_path / "job")[0]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    header = next(row for row in rows if row["record_type"] == "trace")

    assert ("Source run timed out while executing actions 2; " "replay includes timeout/retry control flow") in header[
        "limitations"
    ]
    assert (
        "Source run has non-zero completed actions 1 (status 2); " "replay includes failed command paths"
    ) in header["limitations"]
    assert ("Source trajectory contains actions not executed in the run: 3; " "replay must exclude them") in header[
        "limitations"
    ]
    assert header["environment"]["command_env"] is None


def test_trace_header_records_command_env(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text("")
    (agent_dir / COMMAND_ENV_NAME).write_text(json.dumps({"LC_CTYPE": "C.UTF-8", "PATH": "/usr/bin:/bin"}))

    output = finalize_trajectories(tmp_path / "job")[0]
    header = next(row for row in map(json.loads, output.read_text().splitlines()) if row["record_type"] == "trace")

    assert header["environment"]["command_env"] == {"LC_CTYPE": "C.UTF-8", "PATH": "/usr/bin:/bin"}


def test_perfetto_llm_slice_exposes_tool_call_only_response_details(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t3.100000\t12\t1\t1\t-\ttrue\n" "end\t3.200000\t12\t1\t1\t0\ttrue\n"
    )
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {"timestamp": 1.0, "actions": [{"command": "true"}]},
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": 0}),
                        "extra": {"timestamp": 2.0},
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "extra": {
                            "timestamp": 3.0,
                            "actions": [{"command": ("nohup bash -c 'make > build.log 2>&1' &")}],
                            "response": {
                                "model": "test-model",
                                "choices": [{"finish_reason": "tool_calls"}],
                                "usage": {
                                    "prompt_tokens": 90,
                                    "completion_tokens": 10,
                                    "total_tokens": 100,
                                    "prompt_tokens_details": {"cached_tokens": 64},
                                    "completion_tokens_details": {"reasoning_tokens": 4},
                                },
                            },
                        },
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": 0}),
                        "extra": {"timestamp": 3.5},
                    },
                ]
            }
        )
    )

    finalize_trajectories(tmp_path / "job")
    trace = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())
    llm_event = next(event for event in trace["traceEvents"] if event.get("cat") == "llm")

    assert llm_event["args"]["response_kind"] == "bash_tool_call_only"
    assert "assistant_text" not in llm_event["args"]
    assert llm_event["args"]["command_sent_by_llm"].startswith("nohup bash")
    assert llm_event["args"]["cached_prompt_tokens"] == 64
    assert llm_event["args"]["uncached_prompt_tokens"] == 26
    assert llm_event["args"]["reasoning_tokens"] == 4


def test_perfetto_rerender_supports_legacy_llm_field_and_background_overlap(tmp_path):
    timings = tmp_path / TRACE_TIMINGS_NAME
    timings.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "record_type": "trace",
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "source_trial": "legacy",
                },
                {
                    "record_type": "llm",
                    "name": "LLM response 1",
                    "started_at_epoch": 10.0,
                    "finished_at_epoch": 20.0,
                    "next_command": "tail build.log",
                },
                {
                    "record_type": "command",
                    "command_shell_quoted": "make",
                    "started_at_epoch": 5.0,
                    "finished_at_epoch": 25.0,
                    "shell_level": 2,
                    "pid": 31,
                    "ppid": 30,
                    "exit_status": 0,
                },
            )
        )
        + "\n"
    )
    output = tmp_path / "legacy-max.perfetto.json"

    write_perfetto_trace(timings, perfetto_depth="max", output_path=output)
    trace = json.loads(output.read_text())
    llm_event = next(event for event in trace["traceEvents"] if event.get("cat") == "llm")
    command_event = next(event for event in trace["traceEvents"] if event.get("cat") == "bash_command")

    assert llm_event["args"]["command_sent_by_llm"] == "tail build.log"
    assert llm_event["ts"] > command_event["ts"]
    assert llm_event["ts"] + llm_event["dur"] < command_event["ts"] + command_event["dur"]
    assert trace["metadata"]["perfetto_bash_depth"] == "max"


def test_perfetto_can_be_rerendered_from_finalized_jsonl_only(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t3.100000\t12\t1\t1\t-\techo\\ ok\n" "end\t3.200000\t12\t1\t1\t0\techo\\ ok\n"
    )
    (agent_dir.parent / "result.json").write_text(
        json.dumps(
            {
                "task_name": "compilebench/jq",
                "agent_execution": {
                    "started_at": "1970-01-01T00:00:02Z",
                    "finished_at": "1970-01-01T00:00:04Z",
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "timestamp": 3.0,
                            "actions": [{"command": "echo ok"}],
                        },
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"returncode": 0}),
                        "extra": {"timestamp": 3.3},
                    },
                ]
            }
        )
    )

    timings = finalize_trajectories(tmp_path / "job")[0]
    (agent_dir.parent / "result.json").unlink()
    (agent_dir / "mini-swe-agent.trajectory.json").unlink()
    rerendered = agent_dir / "rerendered.perfetto.json"

    write_perfetto_trace(timings, output_path=rerendered)
    trace = json.loads(rerendered.read_text())

    categories = {event.get("cat") for event in trace["traceEvents"]}
    assert {"phase", "tool_envelope", "bash_command"} <= categories
    assert trace["metadata"]["reward"] == 1.0


def test_perfetto_trajectory_intervals_do_not_have_rounding_seams(tmp_path):
    agent_dir = tmp_path / "job" / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / TRACE_EVENTS_NAME).write_text(
        "begin\t1780004346.700000\t12\t1\t1\t-\ttrue\n" "end\t1780004346.710000\t12\t1\t1\t0\ttrue\n"
    )
    (agent_dir / "mini-swe-agent.trajectory.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "extra": {
                            "timestamp": 1780004346.653454,
                            "actions": [{"command": "true"}],
                        },
                    },
                    {"role": "tool", "extra": {"timestamp": 1780004346.7526433}},
                    {
                        "role": "assistant",
                        "extra": {
                            "timestamp": 1780004348.8235297,
                            "actions": [{"command": "true"}],
                        },
                    },
                    {"role": "tool", "extra": {"timestamp": 1780004348.8420508}},
                ]
            }
        )
    )

    finalize_trajectories(tmp_path / "job")
    trace = json.loads((agent_dir / TRACE_PERFETTO_NAME).read_text())
    intervals = [event for event in trace["traceEvents"] if event.get("tid") == 2 and event.get("ph") == "X"]

    assert len(intervals) == 3
    assert all(
        following["ts"] == preceding["ts"] + preceding["dur"]
        for preceding, following in zip(intervals, intervals[1:], strict=False)
    )


def test_login_shell_install_command_is_idempotent(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home)}
    command = install_login_shell_capture_command()
    subprocess.run(["bash", "-c", command], env=env, check=True)
    subprocess.run(["bash", "-c", command], env=env, check=True)
    profile = (home / ".bash_profile").read_text()
    assert profile.splitlines().count(TRACE_PROFILE_LINE) == 1

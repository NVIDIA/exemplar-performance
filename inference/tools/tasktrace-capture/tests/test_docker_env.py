# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TasktraceCaptureDockerEnv.

Validates the thin subclass of Harbor's DockerEnvironment. We don't run
real Docker here — the override logic is all the new code, so we mock
super().exec() and verify the env dict, barrier, and rotation are applied
correctly before delegation.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest import mock

import pytest
from harbor.environments.docker.docker import DockerEnvironment
from tasktrace_capture.docker_env import TasktraceCaptureDockerEnv
from tasktrace_capture.runtimes import _shared as _shared_mod
from tasktrace_capture.shell_capture import (
    CAPTURE_CONTAINER_SCRIPT,
    CAPTURE_SCRIPT_NAME,
)


@pytest.fixture(autouse=True)
def reset_barrier():
    _shared_mod._reset_barrier_for_tests()
    yield
    _shared_mod._reset_barrier_for_tests()


def _make_env(session_id="trial-abc"):
    """Construct a TasktraceCaptureDockerEnv bypassing Harbor's validation.

    Harbor's __init__ needs real files (Dockerfile, compose YAML). Since
    we only test our override, skip construction and inject attributes.
    """
    obj = TasktraceCaptureDockerEnv.__new__(TasktraceCaptureDockerEnv)
    obj.session_id = session_id
    # logger attr exists on BaseEnvironment; supply a stub for our override's use.
    import logging

    obj.logger = logging.getLogger(f"test.docker.{session_id}")
    return obj


# -- subclass relationship ----------------------------------------------------


def test_is_subclass_of_harbor_docker_environment():
    assert issubclass(TasktraceCaptureDockerEnv, DockerEnvironment)


def test_import_path_resolves():
    """Verify the string tasktrace_capture.runtimes.runtime_import_path('docker') resolves."""
    import importlib

    mod = importlib.import_module("tasktrace_capture.docker_env")
    cls = mod.TasktraceCaptureDockerEnv
    assert issubclass(cls, DockerEnvironment)


def test_only_profile_lifecycle_and_exec_are_overridden():
    """Lifecycle overrides exist only to bracket Docker cgroup measurement."""
    # ABCmeta auto-adds `_abc_impl`; that's not a real override.
    overridden = {
        name
        for name in TasktraceCaptureDockerEnv.__dict__.keys()
        if not name.startswith("__") and not name.startswith("_abc_")
    }
    allowed = {"exec", "start", "stop"} | {n for n in overridden if n.startswith("_")}
    unexpected = overridden - allowed
    assert not unexpected, (
        f"Unexpected overrides in TasktraceCaptureDockerEnv: {unexpected}. "
        "Public overrides increase Harbor coupling — add to design spec + "
        "upgrade audit protocol before landing."
    )


def test_invocation_owner_scopes_compose_project(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_DOCKER_OWNER", "rt123")
    with mock.patch.object(DockerEnvironment, "__init__") as init:
        TasktraceCaptureDockerEnv(session_id="task__trial")
    assert init.call_args.kwargs["session_id"] == "rt123--task__trial"


def test_session_id_is_unchanged_without_owner(monkeypatch):
    monkeypatch.delenv("_TASKTRACE_CAPTURE_DOCKER_OWNER", raising=False)
    with mock.patch.object(DockerEnvironment, "__init__") as init:
        TasktraceCaptureDockerEnv(session_id="task__trial")
    assert init.call_args.kwargs["session_id"] == "task__trial"


# -- exec() override behavior -------------------------------------------------


def test_exec_calls_super_with_modified_env(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_FORWARD_ENVS", "FOO")
    monkeypatch.setenv("FOO", "bar")

    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        super_exec.return_value = "result"
        asyncio.run(env.exec("ls /tmp"))
    super_exec.assert_awaited_once()
    kwargs = super_exec.await_args.kwargs
    assert kwargs["env"]["FOO"] == "bar"


def test_exec_applies_api_key_rotation(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "k1,k2,k3")

    env = _make_env("my-trial")
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env.exec("ls"))
    kwargs = super_exec.await_args.kwargs
    assert kwargs["env"]["ANTHROPIC_API_KEY"] in {"k1", "k2", "k3"}


def test_exec_fires_barrier_only_for_agent_commands(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_BARRIER_COUNT", "2")

    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ), mock.patch(
        "tasktrace_capture.docker_env.wait_at_barrier",
        new_callable=mock.AsyncMock,
    ) as barrier:
        # Setup commands must NOT fire the barrier.
        asyncio.run(env.exec("mkdir -p /config"))
        asyncio.run(env.exec("curl install.sh | bash"))
        barrier.assert_not_awaited()

        # Real agent command DOES fire.
        asyncio.run(env.exec("claude --permission-mode acceptEdits"))
        barrier.assert_awaited_once()


def test_exec_forwards_all_kwargs_to_super():
    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(
            env.exec(
                "ls",
                cwd="/tmp",
                env={"X": "1"},
                timeout_sec=60,
                user="root",
            )
        )
    kwargs = super_exec.await_args.kwargs
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["timeout_sec"] == 60
    assert kwargs["user"] == "root"
    # env is merged with forwarding/rotation output, so at minimum contains X=1
    assert kwargs["env"]["X"] == "1"


def test_exec_positional_command_works():
    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env.exec("ls"))
    assert super_exec.await_args.kwargs.get("command") == "ls" or super_exec.await_args.args[0] == "ls"


def test_exec_session_id_drives_rotation(monkeypatch):
    """Deterministic key pick based on session_id, matching enroot behavior."""
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "k1,k2,k3")
    env_a = _make_env("alpha")
    env_b = _make_env("alpha")  # same id → same key
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env_a.exec("ls"))
        asyncio.run(env_b.exec("ls"))
    keys = [c.kwargs["env"]["ANTHROPIC_API_KEY"] for c in super_exec.await_args_list]
    assert keys[0] == keys[1], "same session_id should pick the same key"


def test_exec_session_id_normalized_for_rotation(monkeypatch):
    """Dots in session_id are replaced with dashes to match compose project name.

    This keeps rotation stable if Harbor ever changes how it generates
    trial ids (e.g. uuid with dots).
    """
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "k1,k2,k3,k4,k5")
    env = _make_env("Task.Name-WithDot")
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env.exec("ls"))
    # No assertion on which key — just that rotation ran without crashing.
    assert super_exec.await_args.kwargs["env"]["ANTHROPIC_API_KEY"] in {"k1", "k2", "k3", "k4", "k5"}


def test_exec_no_barrier_env_is_noop(monkeypatch):
    """Without _TASKTRACE_CAPTURE_BARRIER_COUNT, barrier never blocks."""
    monkeypatch.delenv("_TASKTRACE_CAPTURE_BARRIER_COUNT", raising=False)
    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ):
        # Agent command but no barrier configured — should complete immediately.
        asyncio.run(env.exec("claude --permission-mode acceptEdits"))


def test_exec_returns_super_result():
    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
        return_value="sentinel",
    ):
        result = asyncio.run(env.exec("ls"))
    assert result == "sentinel"


# -- chaos-finding-inspired edge cases ---------------------------------------


def test_exec_with_none_env_does_not_crash():
    """Calling exec(env=None) must not KeyError on downstream helpers."""
    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env.exec("ls", env=None))
    assert isinstance(super_exec.await_args.kwargs["env"], dict)


def test_exec_session_id_with_uppercase_normalized_to_lowercase():
    """session_id with uppercase still produces stable rotation key."""
    monkey_env = {"_TASKTRACE_CAPTURE_API_KEYS": "k1,k2,k3"}
    env_upper = _make_env("Trial.ABC-Upper")
    env_lower = _make_env("trial-abc-upper")
    with mock.patch.dict(os.environ, monkey_env), mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env_upper.exec("ls"))
        asyncio.run(env_lower.exec("ls"))
    keys = [c.kwargs["env"]["ANTHROPIC_API_KEY"] for c in super_exec.await_args_list]
    # Both normalize to `trial-abc-upper`, so rotation picks the same key.
    assert keys[0] == keys[1]


# -- claude-code cache short-circuit ------------------------------------------


def _claude_install_cmd():
    return (
        "if command -v apk; then npm install -g @anthropic-ai/claude-code@2.1.121; "
        "else curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.121; "
        "fi && claude --version"
    )


def test_cache_hook_fires_on_claude_install(tmp_path):
    """Detects the install command and short-circuits via cached binary."""
    bin_path = tmp_path / "claude"
    bin_path.write_text("#!/bin/sh\necho ok")
    os.chmod(bin_path, 0o755)
    env = _make_env()
    with mock.patch(
        "tasktrace_capture.docker_env.ensure_claude_code_cached_on_host",
        return_value=bin_path,
    ), mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(return_code=0, stdout="claude 2.1.121"),
    ) as super_exec, mock.patch.object(
        DockerEnvironment,
        "_run_docker_compose_command",
        new_callable=mock.AsyncMock,
    ) as compose_cmd:
        result = asyncio.run(env.exec(_claude_install_cmd()))
    # Expect: mkdir + cp + version-check (3 calls at minimum across both mocks)
    assert super_exec.await_count >= 2  # mkdir + version check
    assert compose_cmd.await_count >= 1  # cp
    assert result.return_code == 0


def test_cache_hook_falls_through_on_cache_miss():
    """If cache returns None, fall through to Harbor's normal install."""
    env = _make_env()
    with mock.patch(
        "tasktrace_capture.docker_env.ensure_claude_code_cached_on_host",
        return_value=None,
    ), mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env.exec(_claude_install_cmd()))
    assert super_exec.await_count == 1
    kwargs = super_exec.await_args.kwargs
    assert "install" in kwargs.get("command", "")


def test_cache_hook_skipped_for_non_install_commands():
    """Ordinary commands bypass the cache logic entirely."""
    env = _make_env()
    with mock.patch(
        "tasktrace_capture.docker_env.ensure_claude_code_cached_on_host",
    ) as cache_fn, mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ):
        asyncio.run(env.exec("ls /tmp"))
    cache_fn.assert_not_called()


def test_cache_hook_reads_version_env_var(monkeypatch, tmp_path):
    """`_TASKTRACE_CAPTURE_CLAUDE_CODE_VERSION` is forwarded to cache."""
    monkeypatch.setenv("_TASKTRACE_CAPTURE_CLAUDE_CODE_VERSION", "9.9.9")
    bin_path = tmp_path / "claude"
    bin_path.write_text("x")
    os.chmod(bin_path, 0o755)
    env = _make_env()
    with mock.patch(
        "tasktrace_capture.docker_env.ensure_claude_code_cached_on_host",
        return_value=bin_path,
    ) as cache_fn, mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(return_code=0, stdout="ok"),
    ), mock.patch.object(
        DockerEnvironment,
        "_run_docker_compose_command",
        new_callable=mock.AsyncMock,
    ):
        asyncio.run(env.exec(_claude_install_cmd()))
    kwargs = cache_fn.call_args.kwargs
    assert kwargs.get("version") == "9.9.9"


def test_cache_hook_falls_through_on_verify_nonzero(tmp_path):
    """Binary copied but version-check failed → fall through to Harbor install."""
    fake_bin = tmp_path / "claude"
    fake_bin.write_text("x")
    os.chmod(fake_bin, 0o755)
    env = _make_env()
    calls = []

    async def fake_super_exec(self, *a, **kw):
        cmd = kw.get("command") or (a[0] if a else "")
        calls.append(cmd)
        # mkdir (first) succeeds → return 0.
        # version-check (second) fails → return 1.
        # full install (third) succeeds → return 0.
        if "claude --version" in (cmd or "") and "install" not in (cmd or ""):
            return mock.Mock(return_code=1, stdout="not found")
        return mock.Mock(return_code=0, stdout="ok")

    with mock.patch(
        "tasktrace_capture.docker_env.ensure_claude_code_cached_on_host",
        return_value=fake_bin,
    ), mock.patch.object(
        DockerEnvironment,
        "exec",
        new=fake_super_exec,
    ), mock.patch.object(
        DockerEnvironment,
        "_run_docker_compose_command",
        new_callable=mock.AsyncMock,
    ):
        asyncio.run(env.exec(_claude_install_cmd()))
    assert any("claude --version" in c for c in calls if isinstance(c, str))
    assert any("install" in c and "curl" in c for c in calls if isinstance(c, str))


def test_exec_barrier_does_not_fire_for_setup_then_fires_for_agent(monkeypatch):
    """Pair of calls: mkdir (setup) then real agent cmd — barrier fires once."""
    monkeypatch.setenv("_TASKTRACE_CAPTURE_BARRIER_COUNT", "1")  # count=1 → barrier is no-op
    env = _make_env()
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ), mock.patch(
        "tasktrace_capture.docker_env.wait_at_barrier",
        new_callable=mock.AsyncMock,
    ) as barrier:
        asyncio.run(env.exec("mkdir /config"))
        asyncio.run(env.exec("claude --permission-mode ACC"))
        # Count=1 barrier returns immediately; the wrapper still gets called
        # exactly once — for the agent command only.
        assert barrier.await_count == 1


def test_exec_injects_portable_trace_env_for_mini_swe(monkeypatch, tmp_path):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS", "1")
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "mini-swe-agent")
    env = _make_env()
    env.trial_paths = SimpleNamespace(agent_dir=tmp_path)
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ) as super_exec:
        asyncio.run(env.exec("mini-swe-agent --yolo --model=openai/model --task=x --exit-immediately"))
    assert super_exec.await_args.kwargs["env"]["BASH_ENV"] == CAPTURE_CONTAINER_SCRIPT
    assert "--environment-class tasktrace_capture_miniswe_trace_env." in (super_exec.await_args.kwargs["command"])
    assert (tmp_path / CAPTURE_SCRIPT_NAME).exists()


def test_exec_profiles_only_real_mini_swe_launch(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "mini-swe-agent")
    env = _make_env()
    profiler = SimpleNamespace(
        start_scheduler=mock.AsyncMock(),
        begin_window=mock.Mock(),
        finish_window=mock.Mock(),
    )
    env._tasktrace_capture_live_profiler = profiler
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
    ):
        asyncio.run(env.exec("mini-swe-agent --version"))
        asyncio.run(env.exec("mini-swe-agent --yolo --model=openai/model --task=x"))
    profiler.start_scheduler.assert_awaited_once()
    profiler.begin_window.assert_called_once_with("agent_execution")
    profiler.finish_window.assert_called_once_with("agent_execution")


def test_exec_installs_trace_hook_before_terminus_login_shell(monkeypatch, tmp_path):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS", "1")
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "terminus-2")
    env = _make_env()
    env.trial_paths = SimpleNamespace(agent_dir=tmp_path)
    with mock.patch.object(
        DockerEnvironment,
        "exec",
        new_callable=mock.AsyncMock,
        return_value=mock.Mock(return_code=0),
    ) as super_exec:
        asyncio.run(env.exec("tmux new-session -d -s t 'bash --login'"))
    assert super_exec.await_count == 2
    assert ".bash_profile" in super_exec.await_args_list[0].kwargs["command"]
    assert "tmux new-session" in super_exec.await_args_list[1].kwargs["command"]
    assert (tmp_path / CAPTURE_SCRIPT_NAME).exists()

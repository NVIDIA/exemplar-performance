# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from harbor.environments.base import ExecResult
from tasktrace_capture import enroot_env
from tasktrace_capture.enroot_env import (
    EnrootEnvironment,
    _enroot_process_env,
    _mask_cmd_for_logging,
    image_env_exec_argv,
    pid_namespace_prefix,
)

_PROBE = 'printf "%s|%s|%s|%s" "$PATH" "${MODE-}" "${LEAK-unset}" "$HOME"'


def _run(tmp_path: Path, environment_file: str | None, env: dict[str, str], ambient: dict[str, str]) -> list[str]:
    argv = image_env_exec_argv(env, ["/bin/sh", "-c", _PROBE])
    etc_environment = tmp_path / "environment"
    if environment_file is not None:
        etc_environment.write_text(environment_file)
    argv[2] = argv[2].replace("/etc/environment", str(etc_environment))
    result = subprocess.run(argv, env={"HOME": "/root", **ambient}, capture_output=True, text=True, check=True)
    return result.stdout.split("|")


def test_image_env_applies_and_caller_env_wins(tmp_path):
    path, mode, leak, home = _run(
        tmp_path,
        "PATH=/opt/miniconda3/bin:/usr/bin:/bin\nMODE=image\n# comment\nnot an assignment\n",
        {"MODE": "caller"},
        {"PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin:/bin", "LEAK": "from-enroot-init"},
    )
    assert path == "/opt/miniconda3/bin:/usr/bin:/bin"
    assert mode == "caller"
    assert leak == "unset"
    assert home == "/root"


def test_docker_default_path_when_image_sets_none(tmp_path):
    path, _, _, _ = _run(tmp_path, None, {}, {"PATH": "/usr/bin:/bin"})
    assert path == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def test_last_line_without_newline_is_read(tmp_path):
    _, mode, _, _ = _run(tmp_path, "PATH=/usr/bin:/bin\nMODE=image", {}, {"PATH": "/usr/bin:/bin"})
    assert mode == "image"


def test_value_with_spaces_and_equals_survives(tmp_path):
    argv = image_env_exec_argv({"MODE": "a b=c"}, ["/bin/sh", "-c", 'printf "%s" "$MODE"'])
    argv[2] = argv[2].replace("/etc/environment", str(tmp_path / "absent"))
    result = subprocess.run(argv, env={"HOME": "/root", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert result.stdout == "a b=c"


def test_masks_bare_env_operands_of_the_exec_wrapper():
    cmd = ["enroot", "start", "c", *image_env_exec_argv({"OPENAI_API_KEY": "sk-x", "MODE": "a"}, ["bash"])]
    masked = _mask_cmd_for_logging(cmd)
    assert "OPENAI_API_KEY=****" in masked
    assert "OPENAI_API_KEY=sk-x" not in masked
    assert "MODE=a" in masked


def test_enroot_processes_never_mount_the_host_home():
    assert _enroot_process_env()["ENROOT_MOUNT_HOME"] == "no"


@pytest.fixture
def fresh_prefix():
    pid_namespace_prefix.cache_clear()
    yield
    pid_namespace_prefix.cache_clear()


def _probe_results(monkeypatch, allowed):
    calls = []

    def fake_run(argv, **_):
        calls.append(argv[0])
        return SimpleNamespace(returncode=0 if argv[0] in allowed else 1)

    monkeypatch.setattr(enroot_env.subprocess, "run", fake_run)
    return calls


def test_pid_namespace_prefix_prefers_util_linux_unshare(monkeypatch, fresh_prefix):
    calls = _probe_results(monkeypatch, {"unshare", "bwrap"})
    prefix = pid_namespace_prefix()
    assert prefix[0] == "unshare" and "--pid" in prefix and "--mount-proc" in prefix and prefix[-1] == "--"
    assert calls == ["unshare"]
    pid_namespace_prefix()
    assert calls == ["unshare"]


def test_pid_namespace_prefix_falls_back_to_bubblewrap(monkeypatch, fresh_prefix):
    _probe_results(monkeypatch, {"bwrap"})
    prefix = pid_namespace_prefix()
    assert prefix[0] == "bwrap" and "--unshare-pid" in prefix and prefix[-1] == "--"
    assert prefix[prefix.index("--proc") + 1] == "/proc"


def test_pid_namespace_prefix_keeps_unshare_when_nothing_works(monkeypatch, fresh_prefix):
    _probe_results(monkeypatch, set())
    assert pid_namespace_prefix()[0] == "unshare"


def test_exec_runs_enroot_start_in_its_own_pid_namespace(monkeypatch, tmp_path, fresh_prefix):
    monkeypatch.delenv("_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS", raising=False)
    _probe_results(monkeypatch, {"unshare"})
    env = object.__new__(EnrootEnvironment)
    env._container_name = "hb__pidns"
    env.trial_paths = SimpleNamespace(agent_dir=tmp_path)
    env.logger = logging.getLogger("test.enroot.pidns")
    env._barrier_passed = True
    env._mounts = []
    env._workdir = None
    env._resolve_user = lambda user: user
    env._merge_env = lambda values: values or {}
    env._run_cmd = mock.AsyncMock(return_value=ExecResult(return_code=0))

    asyncio.run(env.exec("ls /proc", user="root"))

    command = env._run_cmd.await_args.args[0]
    start = command.index("enroot")
    assert tuple(command[:start]) == pid_namespace_prefix()
    assert command[start : start + 3] == ["enroot", "start", "--rw"]
    assert command[-3:-1] == ["bash", "-c"] and command[-1].endswith("ls /proc")


def test_image_bash_env_only_when_requested(monkeypatch):
    from tasktrace_capture.shell_capture import IMAGE_BASH_ENV_ENV, with_image_bash_env

    monkeypatch.delenv(IMAGE_BASH_ENV_ENV, raising=False)
    assert with_image_bash_env({"A": "1"}) == {"A": "1"}
    monkeypatch.setenv(IMAGE_BASH_ENV_ENV, "/root/.bashrc")
    assert with_image_bash_env({"A": "1"}) == {"A": "1", "BASH_ENV": "/root/.bashrc"}
    assert with_image_bash_env({"BASH_ENV": "/caller"})["BASH_ENV"] == "/caller"


def test_shell_tracing_chains_the_image_bashrc_before_its_traps(monkeypatch, tmp_path):
    from tasktrace_capture.shell_capture import (
        IMAGE_BASH_ENV_ENV,
        TRACE_PRIOR_BASH_ENV,
        prepare_shell_capture,
        with_image_bash_env,
        with_shell_capture_env,
    )

    monkeypatch.setenv(IMAGE_BASH_ENV_ENV, "/root/.bashrc")
    env = with_shell_capture_env(with_image_bash_env({}))
    assert env[TRACE_PRIOR_BASH_ENV] == "/root/.bashrc"

    script = prepare_shell_capture(tmp_path)
    rc = tmp_path / "bashrc"
    rc.write_text("export ACTIVATED_ENV=testbed\nACTIVATION_STEP=1\n")
    raw = tmp_path / "raw.tsv"
    result = subprocess.run(
        ["/bin/bash", "-c", 'printf "%s" "$ACTIVATED_ENV"'],
        env={
            "PATH": "/usr/bin:/bin",
            "BASH_ENV": str(script),
            TRACE_PRIOR_BASH_ENV: str(rc),
            "_TASKTRACE_CAPTURE_RAW_TRAJECTORY_FILE": str(raw),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "testbed"
    assert "ACTIVATION_STEP" not in raw.read_text()


def test_trace_header_records_bash_env(tmp_path):
    from tasktrace_capture.shell_capture import _trace_header

    assert _trace_header(tmp_path, {}, [], "/root/.bashrc")["environment"]["bash_env"] == "/root/.bashrc"
    assert _trace_header(tmp_path, {}, [])["environment"]["bash_env"] is None


def test_exec_passes_the_image_bash_env_to_the_container_command(monkeypatch, tmp_path, fresh_prefix):
    from tasktrace_capture.shell_capture import IMAGE_BASH_ENV_ENV

    monkeypatch.delenv("_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS", raising=False)
    monkeypatch.setenv(IMAGE_BASH_ENV_ENV, "/root/.bashrc")
    _probe_results(monkeypatch, {"unshare"})
    env = object.__new__(EnrootEnvironment)
    env._container_name = "hb__bashenv"
    env.trial_paths = SimpleNamespace(agent_dir=tmp_path)
    env.logger = logging.getLogger("test.enroot.bashenv")
    env._barrier_passed = True
    env._mounts = []
    env._workdir = None
    env._resolve_user = lambda user: user
    env._merge_env = lambda values: values or {}
    env._run_cmd = mock.AsyncMock(return_value=ExecResult(return_code=0))

    asyncio.run(env.exec("python --version", user="root"))

    command = env._run_cmd.await_args.args[0]
    assert "BASH_ENV=/root/.bashrc" in command[command.index("/bin/sh") :]

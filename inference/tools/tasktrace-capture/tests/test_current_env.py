# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CurrentEnvironment (--runtime current).

No container is involved: the environment runs commands as local subprocesses, so the tests
exercise the real exec path, the trial-directory symlinks, the local file transfers, and the
tasktrace-capture semantics (env forwarding, key rotation, barrier, shell instrumentation)
that the other adapters also inject. Harbor's constructor validation is bypassed the same way
test_docker_env.py does it.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.base import BaseEnvironment
from tasktrace_capture import current_env
from tasktrace_capture.current_env import (
    CurrentEnvironment,
    image_refs_match,
    parse_dockerfile_image,
)
from tasktrace_capture.runtimes import _current_container_checks
from tasktrace_capture.runtimes import _shared as _shared_mod
from tasktrace_capture.shell_capture import CAPTURE_CONTAINER_SCRIPT, IMAGE_BASH_ENV_ENV


@pytest.fixture(autouse=True)
def reset_barrier():
    _shared_mod._reset_barrier_for_tests()
    yield
    _shared_mod._reset_barrier_for_tests()


@pytest.fixture(autouse=True)
def clean_current_env(monkeypatch):
    for name in (
        current_env.WORKDIR_ENV,
        current_env.IMAGE_REF_ENV,
        current_env.ALLOW_IMAGE_MISMATCH_ENV,
        current_env.LOGS_ROOT_ENV,
        "_TASKTRACE_CAPTURE_AGENT",
        "_TASKTRACE_CAPTURE_FORWARD_ENVS",
        "_TASKTRACE_CAPTURE_API_KEYS",
        "_TASKTRACE_CAPTURE_BARRIER_COUNT",
        "_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS",
        IMAGE_BASH_ENV_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def container_environ(tmp_path, monkeypatch):
    """Stand-in for /proc/1/environ: the environment the task container was started with."""
    path = tmp_path / "container-environ"
    entries = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path)), "IMAGE_VAR": "from-image"}
    path.write_bytes(b"".join(f"{name}={value}".encode() + b"\0" for name, value in entries.items()))
    monkeypatch.setattr(current_env, "CONTAINER_ENVIRON", path)
    return path


def _make_env(tmp_path: Path, *, workdir: Path | None = None, session_id: str = "task__trial-1") -> CurrentEnvironment:
    """Construct a CurrentEnvironment bypassing Harbor's __init__ (which wants a full task dir)."""
    env = CurrentEnvironment.__new__(CurrentEnvironment)
    env.session_id = session_id
    env.logger = logging.getLogger("test-current-env")
    env.default_user = None
    env._persistent_env = {}
    env._mounts = []
    env._barrier_passed = False
    env._links = []
    env._login_shell_capture_ready = False
    env._container_env = None
    env._image_from_dockerfile = None
    workdir = workdir or tmp_path / "testbed"
    workdir.mkdir(parents=True, exist_ok=True)
    env._workdir_from_dockerfile = str(workdir)
    env.task_env_config = SimpleNamespace(workdir=None, docker_image=None, gpus=0)
    trial_dir = tmp_path / "trial"
    env.trial_paths = SimpleNamespace(
        trial_dir=trial_dir,
        agent_dir=trial_dir / "agent",
        verifier_dir=trial_dir / "verifier",
        artifacts_dir=trial_dir / "artifacts",
    )
    env.environment_dir = tmp_path / "environment"
    env.environment_name = "task"
    return env


def _run(coro):
    return asyncio.run(coro)


# -- registration -------------------------------------------------------------


def test_is_a_harbor_environment_with_docker_type():
    assert issubclass(CurrentEnvironment, BaseEnvironment)
    assert CurrentEnvironment.type().value == "docker"  # import-path envs must name a real Harbor type


def test_import_path_resolves():
    from tasktrace_capture.runtimes import runtime_import_path

    module, _, cls = runtime_import_path("current").partition(":")
    import importlib

    assert getattr(importlib.import_module(module), cls) is CurrentEnvironment


def test_capabilities_are_mounted_and_fail_closed_on_network_isolation():
    env = _make_env(Path("/tmp"))
    assert env.capabilities.mounted is True
    assert env.capabilities.disable_internet is False
    assert env.capabilities.gpus is False
    assert CurrentEnvironment.resource_capabilities().cpu_limit is False


# -- image identity -----------------------------------------------------------


@pytest.mark.parametrize(
    "expected, actual, match",
    [
        # Tag-form digest pin vs @digest form: same content.
        (
            "ghcr.io/epoch-research/swe-bench.eval.arm64.pytest-dev__pytest-7236:sha256:945da72e",
            "ghcr.io/epoch-research/swe-bench.eval.arm64.pytest-dev__pytest-7236@sha256:945da72e",
            True,
        ),
        ("ghcr.io/org/img:sha256:aaaa", "ghcr.io/org/img@sha256:bbbb", False),  # different content
        ("ghcr.io/org/img:v1", "GHCR.io/org/IMG:v1", True),  # same tag, registry/repo case-insensitive
        ("ghcr.io/org/img", "ghcr.io/org/img:latest", True),  # an absent tag means latest
        ("ghcr.io/org/img:trusted", "ghcr.io/org/img:other", False),  # tags differ → different content
        ("ghcr.io/org/img:latest", "ghcr.io/org/img@sha256:bbbb", False),  # one-sided digest: unverifiable
        ("registry:5000/org/img:tag", "registry:5000/org/img:tag", True),  # a port is not a tag
        ("registry:5000/org/img:tag", "registry:5000/org/img@sha256:cc", False),
        ("ghcr.io/org/img", "ghcr.io/org/other", False),
        # A pin that names its tag matches that tag, and only that tag.
        ("ghcr.io/org/img:latest", "ghcr.io/org/img:latest@sha256:bbbb", True),
        ("ghcr.io/org/img", "ghcr.io/org/img:latest@sha256:bbbb", True),
        ("ghcr.io/org/img:latest@sha256:aaaa", "ghcr.io/org/img:latest", True),
        ("ghcr.io/org/img:v2", "ghcr.io/org/img:latest@sha256:bbbb", False),
        ("ghcr.io/org/img:latest@sha256:aaaa", "ghcr.io/org/img:latest@sha256:bbbb", False),
    ],
)
def test_image_refs_match(expected, actual, match):
    assert image_refs_match(expected, actual) is match


def test_parse_dockerfile_image(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "# comment\nFROM --platform=linux/arm64 ghcr.io/org/img:sha256:abc AS base\nRUN echo hi\nWORKDIR /testbed\n"
    )
    assert parse_dockerfile_image(dockerfile) == ("ghcr.io/org/img:sha256:abc", "/testbed")


def test_validate_definition_rejects_wrong_image(tmp_path, monkeypatch):
    env = _make_env(tmp_path)
    env.environment_dir.mkdir()
    (env.environment_dir / "Dockerfile").write_text(
        "FROM ghcr.io/org/img:sha256:aaaa\nWORKDIR " + str(tmp_path / "testbed") + "\n"
    )
    monkeypatch.setenv(current_env.IMAGE_REF_ENV, "ghcr.io/org/img@sha256:bbbb")
    with pytest.raises(ValueError, match="wrong image"):
        env._validate_definition()
    monkeypatch.setenv(current_env.ALLOW_IMAGE_MISMATCH_ENV, "1")
    env._validate_definition()  # downgraded to a warning
    monkeypatch.setenv(current_env.IMAGE_REF_ENV, "ghcr.io/org/img@sha256:aaaa")
    monkeypatch.delenv(current_env.ALLOW_IMAGE_MISMATCH_ENV)
    env._validate_definition()
    assert env._image_from_dockerfile == "ghcr.io/org/img:sha256:aaaa"


def _swebench_env(tmp_path, instance="pytest-dev__pytest-7236"):
    env = _make_env(tmp_path)
    env.environment_dir = tmp_path / instance / "environment"
    env.environment_dir.mkdir(parents=True)
    (env.environment_dir / "Dockerfile").write_text(
        f"FROM swebench/sweb.eval.x86_64.{instance.replace('__', '_1776_')}:latest\n"
        f"WORKDIR {tmp_path / 'testbed'}\n"
    )
    return env


def test_validate_definition_applies_the_epoch_research_swebench_remap(tmp_path, monkeypatch):
    from tasktrace_capture.enroot_env import _HOST_ARCH

    env = _swebench_env(tmp_path)
    epoch = f"ghcr.io/epoch-research/swe-bench.eval.{_HOST_ARCH}.pytest-dev__pytest-7236"
    monkeypatch.setenv(current_env.IMAGE_REF_ENV, f"{epoch}:latest@sha256:945da72e")
    monkeypatch.delenv(current_env.SWEBENCH_IMAGES_ENV, raising=False)
    with pytest.raises(ValueError, match="wrong image"):  # remap off: the Docker Hub image is expected
        env._validate_definition()
    monkeypatch.setenv(current_env.SWEBENCH_IMAGES_ENV, "epoch-research")
    env._validate_definition()  # remap on: the pinned epoch-research image is the task's image
    monkeypatch.setenv(current_env.IMAGE_REF_ENV, f"{epoch}@sha256:945da72e")
    with pytest.raises(ValueError, match="wrong image"):  # a bare digest cannot vouch for :latest
        env._validate_definition()


def test_epoch_swebench_image_ignores_non_swebench_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv(current_env.SWEBENCH_IMAGES_ENV, "epoch-research")
    assert current_env.epoch_swebench_image(tmp_path / "hello-world" / "environment") is None
    assert current_env.epoch_swebench_image(tmp_path / "astropy__astropy-13236" / "environment").endswith(
        ".astropy__astropy-13236:latest"
    )


def test_validate_definition_requires_workdir_to_exist(tmp_path, monkeypatch):
    env = _make_env(tmp_path)
    monkeypatch.setenv(current_env.WORKDIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="does not exist"):
        env._validate_definition()


def test_validate_definition_rejects_gpus(tmp_path):
    env = _make_env(tmp_path)
    env.task_env_config = SimpleNamespace(workdir=None, docker_image=None, gpus=1)
    with pytest.raises(ValueError, match="GPU"):
        env._validate_definition()


# -- lifecycle: /logs symlinks --------------------------------------------------


def test_start_links_trial_dirs_and_stop_removes_them(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"
    monkeypatch.setenv(current_env.LOGS_ROOT_ENV, str(logs_root))
    env = _make_env(tmp_path)
    _run(env.start(force_build=False))
    for name, target in [
        ("agent", env.trial_paths.agent_dir),
        ("verifier", env.trial_paths.verifier_dir),
        ("artifacts", env.trial_paths.artifacts_dir),
    ]:
        link = logs_root / name
        assert link.is_symlink() and os.path.samefile(link, target)
        assert target.is_dir()
    (logs_root / "agent" / "probe.txt").write_text("x")
    assert (env.trial_paths.agent_dir / "probe.txt").read_text() == "x"
    _run(env.stop(delete=True))
    assert not (logs_root / "agent").exists()
    assert (env.trial_paths.agent_dir / "probe.txt").exists()  # trial data is never deleted


def test_start_is_idempotent_for_the_verifier_session_and_replaces_stale_links(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"
    monkeypatch.setenv(current_env.LOGS_ROOT_ENV, str(logs_root))
    first = _make_env(tmp_path)
    _run(first.start(force_build=False))
    verifier = _make_env(tmp_path, session_id="task__trial-1__verifier__")
    _run(verifier.start(force_build=False))  # same trial dirs → links kept
    assert os.path.samefile(logs_root / "agent", first.trial_paths.agent_dir)
    other_root = tmp_path / "other"
    second = _make_env(other_root)
    _run(second.start(force_build=False))  # a later sequential trial re-points the links
    assert os.path.samefile(logs_root / "agent", second.trial_paths.agent_dir)


def test_start_refuses_to_hide_a_populated_directory(tmp_path, monkeypatch):
    logs_root = tmp_path / "logs"
    monkeypatch.setenv(current_env.LOGS_ROOT_ENV, str(logs_root))
    (logs_root / "agent").mkdir(parents=True)
    (logs_root / "agent" / "keep.txt").write_text("data")
    env = _make_env(tmp_path)
    with pytest.raises(RuntimeError, match="non-empty"):
        _run(env.start(force_build=False))


def test_start_rejects_force_build(tmp_path):
    env = _make_env(tmp_path)
    with pytest.raises(RuntimeError, match="cannot build"):
        _run(env.start(force_build=True))


# -- exec ---------------------------------------------------------------------


def test_exec_runs_locally_in_the_workdir_and_returns_output(tmp_path):
    env = _make_env(tmp_path)
    result = _run(env.exec("pwd; echo out; echo err >&2; exit 3"))
    assert result.return_code == 3
    assert result.stdout.strip().splitlines() == [str((tmp_path / "testbed").resolve()), "out"]
    assert result.stderr.strip() == "err"


def test_exec_honors_cwd_and_env(tmp_path):
    env = _make_env(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    env._persistent_env = {"PERSISTENT": "p"}
    result = _run(env.exec('echo "$PWD $PERSISTENT $PER_EXEC"', cwd=str(other), env={"PER_EXEC": "e"}))
    assert result.stdout.strip() == f"{other.resolve()} p e"


def test_exec_forwards_agent_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_FORWARD_ENVS", "OPENAI_API_BASE")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.invalid/v1")
    env = _make_env(tmp_path)
    result = _run(env.exec('echo "$OPENAI_API_BASE"'))
    assert result.stdout.strip() == "https://example.invalid/v1"


def test_exec_starts_from_the_container_environment_not_tasktrace_captures(tmp_path, monkeypatch):
    # The tool's venv first on PATH, as a launcher sets it up.
    tool_bin = tmp_path / "tool-venv" / "bin"
    tool_bin.mkdir(parents=True)
    (tool_bin / "python").write_text("#!/bin/sh\necho tool-python\n")
    (tool_bin / "python").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tool_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    env = _make_env(tmp_path)
    result = _run(env.exec('echo "$IMAGE_VAR ${PYTHONDONTWRITEBYTECODE:-unset} $PATH"; python 2>/dev/null; true'))
    first, *rest = result.stdout.strip().splitlines()
    image_var, bytecode, path = first.split(" ", 2)
    assert (image_var, bytecode) == ("from-image", "unset")
    assert str(tool_bin) not in path
    assert "tool-python" not in rest


def test_exec_sources_the_image_bashrc_when_requested(tmp_path, monkeypatch):
    bashrc = tmp_path / "bashrc"
    bashrc.write_text("export FROM_BASHRC=sourced\n")
    env = _make_env(tmp_path)
    assert _run(env.exec('echo "${FROM_BASHRC:-unset}"')).stdout.strip() == "unset"
    monkeypatch.setenv(IMAGE_BASH_ENV_ENV, str(bashrc))
    assert _run(env.exec('echo "${FROM_BASHRC:-unset}"')).stdout.strip() == "sourced"


def test_exec_env_overrides_the_container_environment(tmp_path):
    env = _make_env(tmp_path)
    result = _run(env.exec('echo "$IMAGE_VAR"', env={"IMAGE_VAR": "from-harbor"}))
    assert result.stdout.strip() == "from-harbor"


def test_start_fails_when_the_container_environment_is_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(current_env, "CONTAINER_ENVIRON", tmp_path / "missing")
    env = _make_env(tmp_path)
    with pytest.raises(RuntimeError, match="container's own environment"):
        _run(env.start(force_build=False))


def test_preflight_requires_the_container_environment(tmp_path, monkeypatch):
    def environ_check():
        return next(ok for name, ok, _ in _current_container_checks() if name.endswith("readable"))

    assert environ_check() is True
    monkeypatch.setattr(current_env, "CONTAINER_ENVIRON", tmp_path / "missing")
    assert environ_check() is False


def _process_gone(pid):
    """True once `pid` no longer runs. A killed process whose parent already exited stays a zombie until
    init reaps it, and a zombie still answers kill(pid, 0), so a zombie counts as gone."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state == "Z"


def test_exec_times_out_and_kills_the_process_group(tmp_path):
    env = _make_env(tmp_path)
    pid_file = tmp_path / "child.pid"
    with pytest.raises(asyncio.TimeoutError):
        _run(env.exec(f"sleep 30 & echo $! > {pid_file}; wait", timeout_sec=1))
    pid = int(pid_file.read_text().strip())
    # The killed sleep can linger briefly as a zombie; a live process after the deadline is a failure.
    deadline = time.monotonic() + 5
    while not _process_gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _process_gone(pid), f"background child {pid} survived the timeout kill"


def test_exec_barrier_fires_once_for_the_agent_command(tmp_path, monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "mini-swe-agent")
    calls = []

    async def fake_barrier(container_name, logger=None):
        calls.append(container_name)

    monkeypatch.setattr(current_env, "wait_at_barrier", fake_barrier)
    env = _make_env(tmp_path)
    _run(env.exec("echo setup"))
    assert calls == []
    _run(env.exec("mini-swe-agent --yolo -t task >/dev/null 2>&1; true"))
    _run(env.exec("mini-swe-agent --yolo -t task >/dev/null 2>&1; true"))
    assert calls == ["task__trial-1"]


def test_exec_instruments_mini_swe_launch_with_shell_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "mini-swe-agent")
    monkeypatch.setenv("_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS", "1")
    env = _make_env(tmp_path)
    env.trial_paths.agent_dir.mkdir(parents=True)
    seen = {}

    async def fake_run_local(command, *, cwd, env, timeout_sec, user):
        seen["command"], seen["env"] = command, env
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    monkeypatch.setattr(env, "_run_local", fake_run_local)
    _run(env.exec("mini-swe-agent --yolo -t task", env={}))
    assert "--environment-class" in seen["command"]
    assert seen["env"]["BASH_ENV"] == CAPTURE_CONTAINER_SCRIPT
    assert (env.trial_paths.agent_dir / Path(CAPTURE_CONTAINER_SCRIPT).name).exists()


def test_wrap_user_is_a_no_op_for_the_current_user(tmp_path):
    env = _make_env(tmp_path)
    argv = ["/bin/bash", "-c", "true"]
    assert env._wrap_user(argv, None) == argv
    assert env._wrap_user(argv, os.getuid()) == argv
    if os.getuid() == 0:
        assert env._wrap_user(argv, "root") == argv
    wrapped = env._wrap_user(argv, "someone-else")
    assert wrapped[:3] == ["su", "-s", "/bin/bash"] and wrapped[-1] == "true"


# -- file transfer --------------------------------------------------------------


def test_upload_and_download_are_local_copies(tmp_path):
    env = _make_env(tmp_path)
    src = tmp_path / "src.txt"
    src.write_text("payload")
    _run(env.upload_file(src, str(tmp_path / "dest" / "copy.txt")))
    assert (tmp_path / "dest" / "copy.txt").read_text() == "payload"
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "f").write_text("1")
    _run(env.upload_dir(tree, str(tmp_path / "tests")))
    assert (tmp_path / "tests" / "sub" / "f").read_text() == "1"
    _run(env.download_dir(str(tmp_path / "tests"), tmp_path / "back"))
    assert (tmp_path / "back" / "sub" / "f").read_text() == "1"
    _run(env.download_file(str(tmp_path / "tests" / "sub" / "f"), tmp_path / "single"))
    assert (tmp_path / "single").read_text() == "1"


def test_transfers_between_the_same_path_are_no_ops(tmp_path, monkeypatch):
    """/logs/agent IS the trial agent dir here; Harbor's download of it must not copy onto itself."""
    logs_root = tmp_path / "logs"
    monkeypatch.setenv(current_env.LOGS_ROOT_ENV, str(logs_root))
    env = _make_env(tmp_path)
    _run(env.start(force_build=False))
    (env.trial_paths.agent_dir / "trace.tsv").write_text("t")
    _run(env.download_dir(str(logs_root / "agent"), env.trial_paths.agent_dir))
    assert (env.trial_paths.agent_dir / "trace.tsv").read_text() == "t"
    _run(env.upload_dir(env.trial_paths.agent_dir, str(logs_root / "agent")))
    assert sorted(p.name for p in env.trial_paths.agent_dir.iterdir()) == ["trace.tsv"]


def test_is_dir_and_is_file_are_local(tmp_path):
    env = _make_env(tmp_path)
    assert _run(env.is_dir(str(tmp_path / "testbed"))) is True
    assert _run(env.is_file(str(tmp_path / "testbed"))) is False

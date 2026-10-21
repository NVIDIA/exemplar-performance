# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tasktrace_capture.runtimes._shared.

Covers host-side Python extracted from enroot_env.py that now applies to
both enroot and docker runtimes: env-var forwarding, API-key rotation,
startup barrier, agent-command detection.
"""

import asyncio
import logging
import os
from unittest import mock

import pytest
from tasktrace_capture.runtimes._shared import (
    _reset_barrier_for_tests,
    apply_api_key_rotation,
    forward_env_vars,
    is_agent_command,
    wait_at_barrier,
)

# -- forward_env_vars ---------------------------------------------------------


def test_forward_env_vars_no_manifest_returns_env_unchanged(monkeypatch):
    monkeypatch.delenv("_TASKTRACE_CAPTURE_FORWARD_ENVS", raising=False)
    env = {"X": "1"}
    out = forward_env_vars(env)
    assert out == {"X": "1"}


def test_forward_env_vars_copies_listed_keys_from_os_environ(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_FORWARD_ENVS", "FOO,BAR")
    monkeypatch.setenv("FOO", "x")
    monkeypatch.setenv("BAR", "y")
    out = forward_env_vars({})
    assert out["FOO"] == "x"
    assert out["BAR"] == "y"


def test_forward_env_vars_skips_missing_keys(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_FORWARD_ENVS", "FOO,MISSING")
    monkeypatch.setenv("FOO", "x")
    monkeypatch.delenv("MISSING", raising=False)
    out = forward_env_vars({})
    assert out == {"FOO": "x"}


def test_forward_env_vars_preserves_caller_overrides(monkeypatch):
    # Behaviour parity with enroot_env.py: caller-supplied env wins over forwarded.
    # Reference: enroot_env.py exec() line 879 — env = {**forwarded, **(env or {})}
    monkeypatch.setenv("_TASKTRACE_CAPTURE_FORWARD_ENVS", "FOO")
    monkeypatch.setenv("FOO", "from_os")
    out = forward_env_vars({"FOO": "from_caller"})
    assert out["FOO"] == "from_caller"


# -- apply_api_key_rotation ---------------------------------------------------


def test_api_key_rotation_no_keys_returns_env_unchanged(monkeypatch):
    monkeypatch.delenv("_TASKTRACE_CAPTURE_API_KEYS", raising=False)
    env = {"X": "1"}
    out = apply_api_key_rotation(env, container_name="hb__test")
    assert out == {"X": "1"}


def test_api_key_rotation_single_key_does_not_rotate(monkeypatch):
    # Parity: enroot_env.py only rotates when len(keys) > 1 (line 887).
    # Single key is handled upstream in runner.py by setting ANTHROPIC_API_KEY
    # directly, so apply_api_key_rotation treats single-key as no-op.
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "only-one-key")
    out = apply_api_key_rotation({}, container_name="hb__test")
    assert out == {}


def test_api_key_rotation_picks_key_by_container_name_hash(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "k1,k2,k3")
    out = apply_api_key_rotation({}, container_name="hb__test")
    assert out["ANTHROPIC_API_KEY"] in {"k1", "k2", "k3"}


def test_api_key_rotation_deterministic_same_container(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "k1,k2,k3")
    out1 = apply_api_key_rotation({}, container_name="hb__test")
    out2 = apply_api_key_rotation({}, container_name="hb__test")
    assert out1["ANTHROPIC_API_KEY"] == out2["ANTHROPIC_API_KEY"]


def test_api_key_rotation_different_containers_may_get_different_keys(monkeypatch):
    # Statistical — with enough containers across 3 keys we should see >1 distinct.
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "k1,k2,k3")
    picked = {apply_api_key_rotation({}, container_name=f"hb__test{i}")["ANTHROPIC_API_KEY"] for i in range(20)}
    assert len(picked) >= 2, "rotation should spread across keys over 20 containers"


def test_api_key_rotation_strips_whitespace_and_empty(monkeypatch):
    # Parity: enroot_env.py line 886 — [k.strip() for k in ... if k.strip()]
    monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "k1 , , k2")
    out = apply_api_key_rotation({}, container_name="hb__test")
    assert out["ANTHROPIC_API_KEY"] in {"k1", "k2"}


# -- is_agent_command ---------------------------------------------------------


def test_is_agent_command_claude_code_detected_by_permission_mode():
    assert is_agent_command("claude --permission-mode acceptEdits ...", agent="claude-code")


def test_is_agent_command_terminus_detected_by_literal():
    assert is_agent_command("python -m terminus run", agent="terminus-2")


def test_is_agent_command_mini_swe_detected_by_yolo_launch_only():
    assert is_agent_command(
        "mini-swe-agent --yolo --model=openai/model --task=x",
        agent="mini-swe-agent",
    )
    assert not is_agent_command("mini-swe-agent --version", agent="mini-swe-agent")


def test_is_agent_command_setup_commands_not_detected():
    # mkdir, curl install, etc. are the setup commands that should NOT trigger barrier
    assert not is_agent_command("mkdir -p /config", agent="claude-code")
    assert not is_agent_command("curl -fsSL https://... | bash", agent="claude-code")
    assert not is_agent_command("claude --version", agent="claude-code")


def test_is_agent_command_empty_agent_does_not_match_setup_commands():
    # Even with no agent hint, arbitrary shell commands must not fire the barrier.
    assert not is_agent_command("ls /tmp", agent="")


# -- wait_at_barrier ----------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_barrier():
    _reset_barrier_for_tests()
    yield
    _reset_barrier_for_tests()


def test_barrier_no_env_var_returns_immediately(monkeypatch):
    monkeypatch.delenv("_TASKTRACE_CAPTURE_BARRIER_COUNT", raising=False)

    async def run():
        await wait_at_barrier(container_name="hb__a")

    asyncio.run(run())  # should not hang


def test_barrier_count_one_returns_immediately(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_BARRIER_COUNT", "1")

    async def run():
        await wait_at_barrier(container_name="hb__a")

    asyncio.run(run())


def test_barrier_three_agents_release_together(monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_BARRIER_COUNT", "3")

    async def run():
        start_times = {}
        release_times = {}

        async def arrive(name, delay):
            await asyncio.sleep(delay)
            start_times[name] = asyncio.get_event_loop().time()
            await wait_at_barrier(container_name=name)
            release_times[name] = asyncio.get_event_loop().time()

        await asyncio.gather(
            arrive("a", 0.0),
            arrive("b", 0.01),
            arrive("c", 0.05),
        )
        # All three should release within a small window of each other,
        # bounded by the latest arrival.
        spread = max(release_times.values()) - min(release_times.values())
        assert spread < 0.05, f"barrier spread {spread*1000:.1f}ms too wide"

    asyncio.run(run())


# -- Claude Code binary cache -------------------------------------------------


def test_claude_install_detect_positive():
    from tasktrace_capture.runtimes._shared import is_claude_code_install_command

    assert is_claude_code_install_command(
        "if apk; then npm install -g @anthropic-ai/claude-code@X; "
        "else curl -fsSL https://claude.ai/install.sh | bash -s -- X; "
        "fi && claude --version"
    )


def test_claude_install_detect_false_for_unrelated_commands():
    from tasktrace_capture.runtimes._shared import is_claude_code_install_command

    assert not is_claude_code_install_command("ls /tmp")
    assert not is_claude_code_install_command("claude --permission-mode x")
    # Partial matches must not fire
    assert not is_claude_code_install_command("claude --version")
    assert not is_claude_code_install_command("curl install.sh")


def test_ensure_claude_cached_returns_existing(tmp_path, monkeypatch):
    from tasktrace_capture.runtimes import _shared

    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\necho ok")
    os.chmod(binary, 0o755)
    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_LOCK", tmp_path.with_suffix(".lock"))
    result = _shared.ensure_claude_code_cached_on_host()
    assert result == binary


def test_ensure_claude_cached_returns_none_on_install_failure(tmp_path, monkeypatch):
    from tasktrace_capture.runtimes import _shared

    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_LOCK", tmp_path / "cache.lock")
    fake = mock.Mock(returncode=1, stderr="install failed")
    monkeypatch.setattr(_shared.subprocess, "run", lambda *a, **kw: fake)
    assert _shared.ensure_claude_code_cached_on_host() is None


def test_ensure_claude_cached_returns_none_on_timeout(tmp_path, monkeypatch):
    import subprocess as sp

    from tasktrace_capture.runtimes import _shared

    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_LOCK", tmp_path / "cache.lock")

    def _raise(*a, **kw):
        raise sp.TimeoutExpired(cmd=["bash"], timeout=120)

    monkeypatch.setattr(_shared.subprocess, "run", _raise)
    assert _shared.ensure_claude_code_cached_on_host() is None


def test_ensure_claude_cached_returns_none_if_binary_missing_after_install(tmp_path, monkeypatch):
    from tasktrace_capture.runtimes import _shared

    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(_shared, "CLAUDE_CODE_CACHE_LOCK", tmp_path / "cache.lock")
    # install.sh returns 0 but doesn't actually create the binary
    monkeypatch.setattr(_shared.subprocess, "run", lambda *a, **kw: mock.Mock(returncode=0, stderr=""))
    assert _shared.ensure_claude_code_cached_on_host() is None


def test_barrier_timeout_logs_and_proceeds(monkeypatch, caplog):
    # Set barrier to 3 but only 2 callers arrive; expect timeout + proceed.
    monkeypatch.setenv("_TASKTRACE_CAPTURE_BARRIER_COUNT", "3")

    import tasktrace_capture.runtimes._shared as shared

    monkeypatch.setattr(shared, "_BARRIER_TIMEOUT_SEC", 0.2)

    async def run():
        async def arrive(name):
            await wait_at_barrier(container_name=name)

        with caplog.at_level(logging.WARNING, logger="tasktrace_capture.runtimes._shared"):
            # Two callers, barrier expects three — both should time out, not hang forever.
            await asyncio.gather(arrive("a"), arrive("b"))

        assert any("timed out" in rec.message.lower() for rec in caplog.records), "expected timeout warning"

    asyncio.run(run())

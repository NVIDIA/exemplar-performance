# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from pathlib import Path
from unittest import mock

from harbor.environments.base import ExecResult
from tasktrace_capture.apptainer.command_policy import TasktraceCaptureCommandPolicy
from tasktrace_capture.apptainer.config import RuntimeLayout
from tasktrace_capture.apptainer.workspace import prepare_workspace

MINI_SWE_DEPENDENCY_INSTALL = (
    "if command -v apt-get &>/dev/null; then"
    "  apt-get update && apt-get install -y curl build-essential git;"
    " elif command -v apk &>/dev/null; then"
    "  apk add --no-cache curl bash build-base git python3 py3-pip;"
    " elif command -v yum &>/dev/null; then"
    "  yum install -y curl git gcc make;"
    " elif command -v dnf &>/dev/null; then"
    "  dnf install -y curl git gcc make;"
    " else"
    '  echo "Warning: No known package manager found, assuming build tools are available" >&2;'
    " fi"
)
MINI_SWE_UV_BOOTSTRAP = (
    "set -euo pipefail; "
    "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh && "
    "if ! grep -q 'export PATH=\"$HOME/.local/bin:$PATH\"' "
    '"$HOME/.bashrc" 2>/dev/null; then  '
    "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> \"$HOME/.bashrc\"; "
    'fi && source "$HOME/.local/bin/env" && '
    "uv tool install mini-swe-agent && mini-swe-agent --help"
)


def test_root_mapped_mini_swe_bootstrap_disables_tar_ownership(tmp_path: Path):
    instance = mock.Mock(
        name="tasktrace-capture-hb-test",
        root_mapped_only=True,
    )
    instance.exec = mock.AsyncMock(return_value=ExecResult(return_code=0))
    policy = TasktraceCaptureCommandPolicy(
        instance=instance,
        logger=logging.getLogger("test.apptainer.policy"),
        agent_artifact_dir=tmp_path / "agent",
        image_environment={},
        default_workdir="/tmp",
    )

    asyncio.run(
        policy.execute(
            "set -o pipefail; " + MINI_SWE_UV_BOOTSTRAP,
            cwd=None,
            environment={},
            timeout_sec=30,
            user="root",
        )
    )

    assert instance.exec.await_args.kwargs["env"]["TAR_OPTIONS"] == ("--no-same-owner")


def test_full_fakeroot_does_not_change_tar_behavior(tmp_path: Path):
    instance = mock.Mock(
        name="tasktrace-capture-hb-test",
        root_mapped_only=False,
    )
    instance.exec = mock.AsyncMock(return_value=ExecResult(return_code=0))
    policy = TasktraceCaptureCommandPolicy(
        instance=instance,
        logger=logging.getLogger("test.apptainer.policy"),
        agent_artifact_dir=tmp_path / "agent",
        image_environment={},
        default_workdir="/tmp",
    )

    asyncio.run(
        policy.execute(
            "set -o pipefail; " + MINI_SWE_UV_BOOTSTRAP,
            cwd=None,
            environment={},
            timeout_sec=30,
            user="root",
        )
    )

    assert "TAR_OPTIONS" not in instance.exec.await_args.kwargs["env"]


def test_similar_mini_swe_bootstrap_does_not_change_tar_behavior(
    tmp_path: Path,
):
    instance = mock.Mock(
        name="tasktrace-capture-hb-test",
        root_mapped_only=True,
    )
    instance.exec = mock.AsyncMock(return_value=ExecResult(return_code=0))
    policy = TasktraceCaptureCommandPolicy(
        instance=instance,
        logger=logging.getLogger("test.apptainer.policy"),
        agent_artifact_dir=tmp_path / "agent",
        image_environment={},
        default_workdir="/tmp",
    )

    asyncio.run(
        policy.execute(
            "set -o pipefail; " + MINI_SWE_UV_BOOTSTRAP + "; tar -xf task-input.tar",
            cwd=None,
            environment={},
            timeout_sec=30,
            user="root",
        )
    )

    assert "TAR_OPTIONS" not in instance.exec.await_args.kwargs["env"]


def test_root_mapped_mini_swe_skips_redundant_dependency_install(
    tmp_path: Path,
):
    instance = mock.Mock(
        name="tasktrace-capture-hb-test",
        root_mapped_only=True,
    )
    instance.exec = mock.AsyncMock(return_value=ExecResult(return_code=0))
    policy = TasktraceCaptureCommandPolicy(
        instance=instance,
        logger=logging.getLogger("test.apptainer.policy"),
        agent_artifact_dir=tmp_path / "agent",
        image_environment={},
        default_workdir="/tmp",
    )

    result = asyncio.run(
        policy.execute(
            "set -o pipefail; " + MINI_SWE_DEPENDENCY_INSTALL,
            cwd=None,
            environment={},
            timeout_sec=30,
            user="root",
        )
    )

    assert result.return_code == 0
    instance.exec.assert_awaited_once()
    prerequisite_command = instance.exec.await_args.args[0][2]
    assert "command -v curl" in prerequisite_command
    assert "apt-get" not in prerequisite_command


def test_dependency_match_tolerates_harmless_whitespace(tmp_path: Path):
    instance = mock.Mock(
        name="tasktrace-capture-hb-test",
        root_mapped_only=True,
    )
    instance.exec = mock.AsyncMock(return_value=ExecResult(return_code=0))
    policy = TasktraceCaptureCommandPolicy(
        instance=instance,
        logger=logging.getLogger("test.apptainer.policy"),
        agent_artifact_dir=tmp_path / "agent",
        image_environment={},
        default_workdir="/tmp",
    )
    command = "set -o pipefail; " + MINI_SWE_DEPENDENCY_INSTALL.replace("; elif", ";\n  elif")

    result = asyncio.run(
        policy.execute(
            command,
            cwd=None,
            environment={},
            timeout_sec=30,
            user="root",
        )
    )

    assert result.return_code == 0
    instance.exec.assert_awaited_once()
    assert "apt-get" not in instance.exec.await_args.args[0][2]


def test_root_mapped_mini_swe_runs_install_when_dependencies_are_missing(
    tmp_path: Path,
):
    instance = mock.Mock(
        name="tasktrace-capture-hb-test",
        root_mapped_only=True,
    )
    instance.exec = mock.AsyncMock(
        side_effect=[
            ExecResult(return_code=1),
            ExecResult(return_code=100),
        ]
    )
    policy = TasktraceCaptureCommandPolicy(
        instance=instance,
        logger=logging.getLogger("test.apptainer.policy"),
        agent_artifact_dir=tmp_path / "agent",
        image_environment={},
        default_workdir="/tmp",
    )

    result = asyncio.run(
        policy.execute(
            "set -o pipefail; " + MINI_SWE_DEPENDENCY_INSTALL,
            cwd=None,
            environment={},
            timeout_sec=30,
            user="root",
        )
    )

    assert result.return_code == 100
    assert instance.exec.await_count == 2
    assert instance.exec.await_args.args[0] == [
        "bash",
        "-lc",
        "set -o pipefail; " + MINI_SWE_DEPENDENCY_INSTALL,
    ]


def test_similar_dependency_command_is_not_skipped(tmp_path: Path):
    instance = mock.Mock(
        name="tasktrace-capture-hb-test",
        root_mapped_only=True,
    )
    instance.exec = mock.AsyncMock(return_value=ExecResult(return_code=0))
    policy = TasktraceCaptureCommandPolicy(
        instance=instance,
        logger=logging.getLogger("test.apptainer.policy"),
        agent_artifact_dir=tmp_path / "agent",
        image_environment={},
        default_workdir="/tmp",
    )
    command = "set -o pipefail; " + MINI_SWE_DEPENDENCY_INSTALL + "; touch /task/extra-work"

    result = asyncio.run(
        policy.execute(
            command,
            cwd=None,
            environment={},
            timeout_sec=30,
            user="root",
        )
    )

    assert result.return_code == 0
    instance.exec.assert_awaited_once()
    assert instance.exec.await_args.args[0] == ["bash", "-lc", command]


def test_workspace_provides_uv_installer_path_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("_TASKTRACE_CAPTURE_AGENT", raising=False)
    layout = RuntimeLayout(tmp_path / "runtime", "tasktrace-capture-hb-test")

    prepare_workspace(layout, logging.getLogger("test.apptainer.workspace"))

    assert (layout.home / ".local" / "bin" / "env").read_text() == ('export PATH="$HOME/.local/bin:$PATH"\n')

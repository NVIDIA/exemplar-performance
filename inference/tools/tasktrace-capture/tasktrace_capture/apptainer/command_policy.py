# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TaskTrace Capture-specific command policy layered over an Apptainer instance."""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

from harbor.environments.base import ExecResult

from tasktrace_capture.apptainer.instance import ApptainerInstance
from tasktrace_capture.runtimes._shared import (
    apply_api_key_rotation,
    forward_env_vars,
    is_agent_command,
    is_claude_code_install_command,
    wait_at_barrier,
)
from tasktrace_capture.shell_capture import (
    command_starts_login_shell,
    install_login_shell_capture_command,
    instrument_agent_launch,
    prepare_shell_capture,
    shell_capture_enabled,
)

_MINI_SWE_UV_BOOTSTRAP = (
    "set -euo pipefail; "
    "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh && "
    "if ! grep -q 'export PATH=\"$HOME/.local/bin:$PATH\"' "
    '"$HOME/.bashrc" 2>/dev/null; then  '
    "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> \"$HOME/.bashrc\"; "
    'fi && source "$HOME/.local/bin/env" && '
    "uv tool install mini-swe-agent && mini-swe-agent --help"
)
_MINI_SWE_DEPENDENCY_INSTALL = (
    "if command -v apt-get &>/dev/null; then"
    "  apt-get update && apt-get install -y curl build-essential git;"
    " elif command -v apk &>/dev/null; then"
    "  apk add --no-cache curl bash build-base git python3 py3-pip;"
    " elif command -v yum &>/dev/null; then"
    "  yum install -y curl git gcc make;"
    " elif command -v dnf &>/dev/null; then"
    "  dnf install -y curl git gcc make;"
    " else"
    '  echo "Warning: No known package manager found, assuming build tools are '
    'available" >&2;'
    " fi"
)


def _shell_tokens(command: str) -> tuple[str, ...]:
    lexer = shlex.shlex(
        command,
        posix=True,
        punctuation_chars=";&|<>",
    )
    lexer.commenters = ""
    lexer.whitespace_split = True
    return tuple(lexer)


_MINI_SWE_UV_BOOTSTRAP_TOKENS = _shell_tokens(_MINI_SWE_UV_BOOTSTRAP)
_MINI_SWE_DEPENDENCY_INSTALL_TOKENS = _shell_tokens(_MINI_SWE_DEPENDENCY_INSTALL)
_HARBOR_PIPEFAIL_PREFIX = _shell_tokens("set -o pipefail;")


def _harbor_command_tokens(command: str) -> tuple[str, ...]:
    tokens = _shell_tokens(command)
    prefix_length = len(_HARBOR_PIPEFAIL_PREFIX)
    if tokens[:prefix_length] == _HARBOR_PIPEFAIL_PREFIX:
        return tokens[prefix_length:]
    return tokens


def _is_mini_swe_uv_bootstrap(command: str) -> bool:
    try:
        actual = list(_harbor_command_tokens(command))
    except ValueError:
        return False
    expected = list(_MINI_SWE_UV_BOOTSTRAP_TOKENS)
    if len(actual) != len(expected):
        return False

    install_index = expected.index("mini-swe-agent")
    versioned_package = actual[install_index]
    if not (versioned_package == "mini-swe-agent" or versioned_package.startswith("mini-swe-agent==")):
        return False
    actual[install_index] = "mini-swe-agent"
    return actual == expected


def _is_mini_swe_dependency_install(command: str) -> bool:
    try:
        return _harbor_command_tokens(command) == _MINI_SWE_DEPENDENCY_INSTALL_TOKENS
    except ValueError:
        return False


class TasktraceCaptureCommandPolicy:
    """Apply TaskTrace Capture agent hooks without owning container lifecycle."""

    def __init__(
        self,
        *,
        instance: ApptainerInstance,
        logger: logging.Logger,
        agent_artifact_dir: Path,
        image_environment: dict[str, str],
        default_workdir: str | None,
    ):
        self._instance = instance
        self._logger = logger
        self._agent_artifact_dir = agent_artifact_dir
        self._image_environment = dict(image_environment)
        self._default_workdir = default_workdir
        self._barrier_passed = False
        self._login_trace_ready = False

    async def execute(
        self,
        command: str,
        *,
        cwd: str | None,
        environment: dict[str, str] | None,
        timeout_sec: int | None,
        user: str | int | None,
    ) -> ExecResult:
        merged = {
            **self._image_environment,
            **(environment or {}),
        }
        merged = forward_env_vars(merged)
        merged = apply_api_key_rotation(
            merged,
            container_name=self._instance.name,
            logger=self._logger,
        )
        if self._instance.root_mapped_only and _is_mini_swe_uv_bootstrap(command):
            # A root-mapped-only namespace cannot represent archive owners.
            # Limit the workaround to Harbor's exact mini-swe bootstrap rather
            # than silently weakening ownership semantics for arbitrary tasks.
            merged.setdefault("TAR_OPTIONS", "--no-same-owner")

        if self._instance.root_mapped_only and _is_mini_swe_dependency_install(command):
            prerequisites = await self._instance.exec(
                [
                    "bash",
                    "-lc",
                    "command -v curl >/dev/null && "
                    "command -v git >/dev/null && "
                    "command -v gcc >/dev/null && "
                    "command -v g++ >/dev/null && "
                    "command -v make >/dev/null",
                ],
                env=merged,
                user=user,
            )
            if prerequisites.return_code == 0:
                self._logger.info(
                    "mini-swe build prerequisites already present; "
                    "skipping unsupported package installation in "
                    "root-mapped-only Apptainer"
                )
                return prerequisites

        agent = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
        if is_agent_command(command, agent) and not self._barrier_passed:
            self._barrier_passed = True
            await wait_at_barrier(
                container_name=self._instance.name,
                logger=self._logger,
            )
        if shell_capture_enabled():
            command, merged = instrument_agent_launch(
                command,
                agent,
                merged,
                self._agent_artifact_dir,
            )
            if command_starts_login_shell(command, agent):
                await self._prepare_login_shell_capture(merged, user)

        if is_claude_code_install_command(command):
            check = await self._instance.exec(
                [
                    "bash",
                    "-lc",
                    'export PATH="$HOME/.local/bin:$PATH"; claude --version',
                ],
                env=merged,
                user=user,
            )
            if check.return_code == 0:
                self._logger.info(
                    "claude-code already installed: %s",
                    (check.stdout or "").strip(),
                )
                return check

        return await self._instance.exec(
            ["bash", "-lc", command],
            cwd=cwd or self._default_workdir,
            env=merged,
            timeout_sec=timeout_sec,
            user=user,
        )

    async def _prepare_login_shell_capture(
        self,
        environment: dict[str, str],
        user: str | int | None,
    ) -> None:
        if self._login_trace_ready:
            return
        prepare_shell_capture(self._agent_artifact_dir)
        result = await self._instance.exec(
            ["bash", "-lc", install_login_shell_capture_command()],
            env=environment,
            user=user,
        )
        if result.return_code == 0:
            self._login_trace_ready = True
        else:
            self._logger.warning("Unable to install Bash command timing hook")

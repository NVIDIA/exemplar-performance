# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async subprocess execution with explicit timeout semantics."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
from pathlib import Path

from harbor.environments.base import ExecResult


class CommandTimeoutError(RuntimeError):
    """Raised after the timed-out process group has been terminated."""


class CommandRunner:
    """Run host commands without owning any container lifecycle."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    async def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path = "/",
        timeout_sec: float | None = None,
        check: bool = True,
    ) -> ExecResult:
        self._logger.debug("Running host command: %s", shlex.join(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(cwd),
            process_group=0,
        )

        try:
            if timeout_sec is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            await self._terminate_process_group(process)
            raise CommandTimeoutError(f"Command timed out after {timeout_sec}s: {shlex.join(command)}") from None
        except asyncio.CancelledError:
            await self._terminate_process_group(process)
            raise

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode if process.returncode is not None else 1,
        )
        if check and result.return_code != 0:
            raise RuntimeError(
                f"Command failed (rc={result.return_code}): {shlex.join(command)}"
                f"\nstdout: {stdout}\nstderr: {stderr}"
            )
        return result

    @staticmethod
    async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(process.communicate(), timeout=5)
            return
        except asyncio.TimeoutError:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        await process.communicate()

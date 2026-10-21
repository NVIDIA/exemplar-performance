# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apptainer named-instance lifecycle and command execution."""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

from harbor.environments.base import ExecResult

from tasktrace_capture.apptainer.capabilities import subordinate_id_mapping_available
from tasktrace_capture.apptainer.config import ApptainerSettings
from tasktrace_capture.apptainer.mounts import BindMount
from tasktrace_capture.apptainer.process import CommandRunner, CommandTimeoutError


class ApptainerInstance:
    """Own one named Apptainer instance.

    Image selection and validation deliberately live elsewhere. This class
    only starts, executes in, stops, and attaches to an instance.
    """

    def __init__(
        self,
        *,
        settings: ApptainerSettings,
        runner: CommandRunner,
        logger: logging.Logger,
        gpus: int,
    ):
        self._settings = settings
        self._runner = runner
        self._logger = logger
        self._gpus = gpus
        self._started = False
        # Determine this from host capabilities, not optional CLI wording.
        # Apptainer has changed its fakeroot diagnostics between releases.
        self._root_mapped_only = not subordinate_id_mapping_available(
            environment=self._settings.subprocess_environment()
        )

    @property
    def name(self) -> str:
        return self._settings.instance_name

    @property
    def started(self) -> bool:
        return self._started

    @property
    def root_mapped_only(self) -> bool:
        """Whether a full subordinate UID/GID mapping is unavailable."""
        return self._root_mapped_only

    async def start(
        self,
        image: Path,
        mounts: list[BindMount],
        *,
        timeout_sec: float,
    ) -> None:
        await self._runner.run(
            [
                self._settings.binary,
                "instance",
                "stop",
                "--force",
                self.name,
            ],
            env=self._settings.subprocess_environment(),
            timeout_sec=15,
            check=False,
        )

        layout = self._settings.layout
        command = [
            self._settings.binary,
            "instance",
            "start",
            "--cleanenv",
            "--containall",
            "--no-eval",
            "--no-mount",
            "bind-paths",
            "--workdir",
            str(layout.contain),
            "--pid-file",
            str(layout.pid_file),
        ]
        command.extend(
            [
                "--fakeroot",
                "--overlay",
                str(layout.overlay),
                "--home",
                f"{layout.home}:/root",
            ]
        )
        if self._gpus > 0:
            command.append("--nv")
        for mount in mounts:
            command.extend(["--mount", mount.apptainer_spec()])
        command.extend([str(image), self.name])

        result = await self._runner.run(
            command,
            env=self._settings.subprocess_environment(),
            timeout_sec=timeout_sec,
        )
        diagnostics = "\n".join(part for part in (result.stdout, result.stderr) if part)
        diagnostic = diagnostics.lower()
        self._root_mapped_only = self._root_mapped_only or any(
            marker in diagnostic
            for marker in (
                "root-mapped namespace",
                "not listed in /etc/subuid",
            )
        )
        if self._root_mapped_only:
            self._logger.warning(
                "Apptainer started without subordinate UID/GID mappings; "
                "package installation and archive ownership changes may fail"
            )
        self._started = True

    async def stop(self) -> None:
        try:
            if self._started:
                result = await self._runner.run(
                    [
                        self._settings.binary,
                        "instance",
                        "stop",
                        "--timeout",
                        "5",
                        self.name,
                    ],
                    env=self._settings.subprocess_environment(),
                    timeout_sec=20,
                    check=False,
                )
                if result.return_code != 0:
                    await self.force_stop()
        except BaseException:
            await self.force_stop()
            raise
        finally:
            self._started = False

    async def force_stop(self) -> None:
        try:
            await self._runner.run(
                [
                    self._settings.binary,
                    "instance",
                    "stop",
                    "--force",
                    self.name,
                ],
                env=self._settings.subprocess_environment(),
                timeout_sec=15,
                check=False,
            )
        except CommandTimeoutError:
            pass
        finally:
            self._started = False

    async def exec(
        self,
        payload: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        user: str | int | None = None,
        check: bool = False,
    ) -> ExecResult:
        if not self._started:
            raise RuntimeError("Apptainer environment is not started")

        if user is not None and str(user) not in {"root", "0"}:
            command = shlex.join(payload)
            user_value = shlex.quote(str(user))
            if str(user).isdigit():
                switch = (
                    f'_tasktrace_capture_user="$(getent passwd {user_value} | cut -d: -f1)"; '
                    'test -n "$_tasktrace_capture_user"; '
                    f'exec su -s /bin/bash "$_tasktrace_capture_user" -c '
                    f"{shlex.quote(command)}"
                )
            else:
                switch = f"exec su -s /bin/bash {user_value} -c " f"{shlex.quote(command)}"
            payload = ["bash", "-lc", switch]

        runtime_env = dict(env or {})
        command = [self._settings.binary, "exec", "--cleanenv", "--no-eval"]
        if cwd:
            command.extend(["--cwd", cwd])
        command.extend([f"instance://{self.name}", *payload])

        try:
            return await self._runner.run(
                command,
                env=self._settings.subprocess_environment(runtime_env),
                timeout_sec=timeout_sec,
                check=check,
            )
        except CommandTimeoutError as exc:
            await self.force_stop()
            raise RuntimeError(f"Apptainer {exc}") from exc

    async def attach(self) -> None:
        os.execvpe(
            self._settings.binary,
            [
                self._settings.binary,
                "shell",
                "--cleanenv",
                f"instance://{self.name}",
            ],
            self._settings.subprocess_environment(),
        )

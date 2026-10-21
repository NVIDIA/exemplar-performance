# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor adapter for prebuilt, architecture-native Apptainer SIF files."""

from __future__ import annotations

import logging
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from tasktrace_capture.apptainer.command_policy import TasktraceCaptureCommandPolicy
from tasktrace_capture.apptainer.config import (
    ApptainerSettings,
    require_apptainer_binary,
)
from tasktrace_capture.apptainer.images import (
    SifImageStore,
    swebench_sif_path,
)
from tasktrace_capture.apptainer.instance import ApptainerInstance
from tasktrace_capture.apptainer.mounts import BindMount, resolve_mounts
from tasktrace_capture.apptainer.process import CommandRunner
from tasktrace_capture.apptainer.transfers import StagingTransfer
from tasktrace_capture.apptainer.workspace import prepare_workspace


class ApptainerEnvironment(BaseEnvironment):
    """Harbor environment implemented with unprivileged Apptainer instances."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        apptainer_runtime_dir: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **kwargs,
        )
        self._settings = ApptainerSettings.load(
            session_id=session_id,
            runtime_root=apptainer_runtime_dir,
        )
        layout = self._settings.layout
        self._apptainer = self._settings.binary
        self._instance_name = self._settings.instance_name
        self._runtime_dir = layout.trial
        self._started = False
        self._sif_path: Path | None = None
        self._instance: ApptainerInstance | None = None
        self._command_policy: TasktraceCaptureCommandPolicy | None = None
        self._transfers: StagingTransfer | None = None
        self._runner = CommandRunner(self.logger)
        self._image_store = SifImageStore(
            settings=self._settings,
            runner=self._runner,
            environment_dir=self.environment_dir,
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.SINGULARITY

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # Apptainer's cgroup flags require a usable systemd/DBus cgroup manager.
        # TaskTrace Capture targets rootless HPC nodes where that is generally unavailable.
        return EnvironmentResourceCapabilities()

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(gpus=True, mounted=True)

    def _validate_definition(self) -> None:
        image = self.task_env_config.docker_image or swebench_sif_path(self.environment_dir)
        if not image:
            raise ValueError(
                "ApptainerEnvironment requires [environment].docker_image to "
                "name an existing .sif file, or --swebench-sif-template. "
                "Dockerfile-only tasks must use --runtime enroot."
            )
        if "://" in image or not image.endswith(".sif"):
            raise ValueError(
                "ApptainerEnvironment accepts existing .sif files only; it does "
                f"not convert or build {image!r}. Use --runtime enroot."
            )
        for mount in self._mounts:
            if mount.get("type") != "bind":
                raise ValueError(
                    "ApptainerEnvironment currently supports bind mounts only; "
                    f"got mount type {mount.get('type')!r} for {mount.get('target')!r}."
                )

    @classmethod
    def preflight(cls) -> None:
        try:
            require_apptainer_binary()
        except RuntimeError as exc:
            raise SystemExit("TaskTrace Capture could not prepare its userspace Apptainer runtime: " f"{exc}") from exc

    def _subprocess_env(
        self,
        injected: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return self._settings.subprocess_environment(injected)

    async def _force_stop_instance(self) -> None:
        try:
            if self._instance is not None:
                await self._instance.force_stop()
                return
            await self._runner.run(
                [
                    self._apptainer,
                    "instance",
                    "stop",
                    "--force",
                    self._instance_name,
                ],
                env=self._subprocess_env(),
                timeout_sec=15,
                check=False,
            )
        except BaseException as exc:
            self.logger.warning(
                "Failed to force-stop partial Apptainer instance %s: %s",
                self._instance_name,
                exc,
            )
        finally:
            self._started = False

    def _remove_runtime_layout(self) -> None:
        # Apptainer can finish tearing down its overlay helper just after the
        # stop command returns. Keep the final removal synchronous so task
        # cancellation cannot interrupt the cleanup halfway through.
        self._settings.layout.remove()

    def _create_instance(self) -> ApptainerInstance:
        return ApptainerInstance(
            settings=self._settings,
            runner=self._runner,
            logger=self.logger,
            gpus=self._effective_gpus,
        )

    async def _pull_image(self, image: str, force: bool = False) -> Path:
        return await self._image_store.resolve(image, force=force)

    async def _sif_architecture(self, sif_path: Path) -> str:
        metadata = await self._image_store.inspect(sif_path)
        return metadata.architecture

    async def _validate_image_arch(self, sif_path: Path) -> None:
        image_arch = await self._sif_architecture(sif_path)
        host_arch = self._settings.host_architecture
        if image_arch == host_arch:
            return
        raise RuntimeError(
            f"Apptainer image architecture {image_arch!r} does not match host "
            f"architecture {host_arch!r}. Provide a native SIF or use "
            "--runtime enroot; TaskTrace Capture does not emulate SIF architectures."
        )

    def _prepare_runtime_dirs(self) -> None:
        prepare_workspace(self._settings.layout, self.logger)

    def _resolved_mounts(self) -> list[BindMount]:
        return resolve_mounts(
            self._mounts,
            layout=self._settings.layout,
            environment_dir=self.environment_dir,
            include_build_context=False,
        )

    async def start(self, force_build: bool) -> None:
        image = self.task_env_config.docker_image or swebench_sif_path(self.environment_dir)
        if not image:
            raise RuntimeError("No SIF available for Apptainer environment")

        self._sif_path = await self._pull_image(image, force=force_build)
        await self._validate_image_arch(self._sif_path)

        try:
            self._prepare_runtime_dirs()
            mounts = self._resolved_mounts()
            self._instance = self._create_instance()
            await self._instance.start(
                self._sif_path,
                mounts,
                timeout_sec=self.task_env_config.build_timeout_sec,
            )
            self._started = self._instance.started

            default_workdir = self.task_env_config.workdir
            self._command_policy = TasktraceCaptureCommandPolicy(
                instance=self._instance,
                logger=self.logger,
                agent_artifact_dir=self.trial_paths.agent_dir,
                image_environment={},
                default_workdir=default_workdir,
            )
            self._transfers = StagingTransfer(
                instance=self._instance,
                layout=self._settings.layout,
            )
            target_dirs = self._mount_targets(writable_only=True)
            if target_dirs:
                await self.ensure_dirs(target_dirs)
            if default_workdir:
                await self._instance.exec(
                    ["mkdir", "-p", default_workdir],
                    cwd="/",
                    user="root",
                    check=True,
                )
            await self._upload_environment_dir_after_start()
        except BaseException:
            try:
                await self._force_stop_instance()
            finally:
                self._started = False
                self._remove_runtime_layout()
                self._instance = None
                self._command_policy = None
                self._transfers = None
            raise

    async def stop(self, delete: bool) -> None:
        try:
            if self._instance is not None:
                await self._instance.stop()
        finally:
            self._started = False
            self._remove_runtime_layout()
            self._instance = None
            self._command_policy = None
            self._transfers = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self._command_policy is None:
            raise RuntimeError("Apptainer environment is not started")
        try:
            return await self._command_policy.execute(
                command,
                cwd=cwd,
                environment=self._merge_env(env),
                timeout_sec=timeout_sec,
                user=self._resolve_user(user),
            )
        finally:
            if self._instance is not None:
                self._started = self._instance.started

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self._transfers is None:
            raise RuntimeError("Apptainer environment is not started")
        await self._transfers.upload_file(source_path, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if self._transfers is None:
            raise RuntimeError("Apptainer environment is not started")
        await self._transfers.upload_dir(source_dir, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if self._transfers is None:
            raise RuntimeError("Apptainer environment is not started")
        await self._transfers.download_file(source_path, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if self._transfers is None:
            raise RuntimeError("Apptainer environment is not started")
        await self._transfers.download_dir(source_dir, target_dir)

    async def attach(self) -> None:
        if self._instance is None:
            raise RuntimeError("Apptainer environment is not started")
        await self._instance.attach()

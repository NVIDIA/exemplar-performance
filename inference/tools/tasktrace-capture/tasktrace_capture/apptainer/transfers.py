# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File transfer through the runtime's private staging mount."""

from __future__ import annotations

import shlex
import shutil
import uuid
from pathlib import Path, PurePosixPath

from tasktrace_capture.apptainer.config import TRANSFER_TARGET, RuntimeLayout
from tasktrace_capture.apptainer.instance import ApptainerInstance


class StagingTransfer:
    def __init__(self, *, instance: ApptainerInstance, layout: RuntimeLayout):
        self._instance = instance
        self._host_staging = layout.transfer

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        transfer_name = self._name(source.suffix)
        host_transfer = self._host_staging / transfer_name
        shutil.copy2(source, host_transfer)
        try:
            parent = str(PurePosixPath(target_path).parent)
            container_transfer = str(PurePosixPath(TRANSFER_TARGET) / transfer_name)
            await self._instance.exec(
                [
                    "bash",
                    "-lc",
                    f"mkdir -p {shlex.quote(parent)} && "
                    f"cp -a {shlex.quote(container_transfer)} "
                    f"{shlex.quote(target_path)}",
                ],
                cwd="/",
                user="root",
                check=True,
            )
        finally:
            host_transfer.unlink(missing_ok=True)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")
        transfer_name = self._name()
        host_transfer = self._host_staging / transfer_name
        shutil.copytree(source, host_transfer, symlinks=True)
        try:
            container_transfer = str(PurePosixPath(TRANSFER_TARGET) / transfer_name)
            await self._instance.exec(
                [
                    "bash",
                    "-lc",
                    f"mkdir -p {shlex.quote(target_dir)} && "
                    f"cp -a {shlex.quote(container_transfer)}/. "
                    f"{shlex.quote(target_dir)}/",
                ],
                cwd="/",
                user="root",
                check=True,
            )
        finally:
            shutil.rmtree(host_transfer, ignore_errors=True)

    async def download_file(
        self,
        source_path: str,
        target_path: Path | str,
    ) -> None:
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        transfer_name = self._name(PurePosixPath(source_path).suffix)
        host_transfer = self._host_staging / transfer_name
        container_transfer = str(PurePosixPath(TRANSFER_TARGET) / transfer_name)
        try:
            await self._instance.exec(
                ["cp", "-a", source_path, container_transfer],
                cwd="/",
                user="root",
                check=True,
            )
            shutil.copy2(host_transfer, target)
        finally:
            host_transfer.unlink(missing_ok=True)

    async def download_dir(
        self,
        source_dir: str,
        target_dir: Path | str,
    ) -> None:
        target = Path(target_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        transfer_name = self._name()
        host_transfer = self._host_staging / transfer_name
        container_transfer = str(PurePosixPath(TRANSFER_TARGET) / transfer_name)
        try:
            await self._instance.exec(
                [
                    "bash",
                    "-lc",
                    f"mkdir -p {shlex.quote(container_transfer)} && "
                    f"cp -a {shlex.quote(source_dir)}/. "
                    f"{shlex.quote(container_transfer)}/",
                ],
                cwd="/",
                user="root",
                check=True,
            )
            for item in host_transfer.iterdir():
                destination = target / item.name
                if item.is_dir() and not item.is_symlink():
                    shutil.copytree(
                        item,
                        destination,
                        dirs_exist_ok=True,
                        symlinks=True,
                    )
                else:
                    shutil.copy2(item, destination, follow_symlinks=False)
        finally:
            shutil.rmtree(host_transfer, ignore_errors=True)

    @staticmethod
    def _name(suffix: str = "") -> str:
        return f"{uuid.uuid4().hex}{suffix}"

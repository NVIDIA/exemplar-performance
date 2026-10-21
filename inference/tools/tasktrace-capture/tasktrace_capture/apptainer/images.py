# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolution and inspection of existing native SIF images."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from tasktrace_capture.apptainer.config import ApptainerSettings, normalize_architecture
from tasktrace_capture.apptainer.process import CommandRunner

SWEBENCH_INSTANCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]+__[A-Za-z0-9][A-Za-z0-9_-]+-\d+$")


@dataclass(frozen=True)
class SifMetadata:
    path: Path
    architecture: str


def swebench_sif_path(environment_dir: Path) -> str | None:
    """Resolve a task-specific SIF using a NeMo-style path template."""
    template = os.environ.get("_TASKTRACE_CAPTURE_SWEBENCH_SIF_TEMPLATE") or os.environ.get(
        "TASKTRACE_CAPTURE_SWEBENCH_SIF_TEMPLATE"
    )
    instance_id = environment_dir.parent.name
    if not template or not SWEBENCH_INSTANCE_RE.fullmatch(instance_id):
        return None
    resolved = template.replace("{instance_id}", instance_id)
    if "{" in resolved or "}" in resolved:
        raise ValueError("SWE-bench SIF template supports only the {instance_id} placeholder")
    return str(Path(resolved).expanduser())


class SifImageStore:
    """Resolve existing SIF files; never build or convert container images."""

    def __init__(
        self,
        *,
        settings: ApptainerSettings,
        runner: CommandRunner,
        environment_dir: Path,
    ):
        self._settings = settings
        self._runner = runner
        self._environment_dir = environment_dir

    async def resolve(self, image: str, *, force: bool = False) -> Path:
        if force:
            raise ValueError(
                "Apptainer uses prebuilt SIF files and does not support "
                "--force-build; use --runtime enroot for Dockerfile builds"
            )
        if "://" in image or not image.endswith(".sif"):
            raise ValueError(
                "Apptainer requires an existing native .sif file and does not "
                f"convert OCI/Docker images: {image!r}. Use --runtime enroot, "
                "configure [environment].docker_image with a SIF path, or pass "
                "--swebench-sif-template."
            )

        expanded = Path(image).expanduser()
        candidates = [expanded]
        if not expanded.is_absolute():
            candidates.append(self._environment_dir / expanded)
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix == ".sif":
                return candidate.resolve()
        raise FileNotFoundError(
            f"Apptainer SIF image not found: {image}. TaskTrace Capture will not convert "
            "or build it; provide the native SIF or use --runtime enroot."
        )

    async def inspect(self, sif_path: Path) -> SifMetadata:
        listed = await self._runner.run(
            [self._settings.binary, "sif", "list", str(sif_path)],
            env=self._settings.subprocess_environment(),
            timeout_sec=30,
            check=False,
        )
        if listed.return_code != 0 or not listed.stdout:
            raise RuntimeError(
                f"Unable to read SIF partition table for {sif_path}: " f"{listed.stderr or 'unknown error'}"
            )
        partition_id: str | None = None
        for line in listed.stdout.splitlines():
            match = re.match(r"^\s*(\d+)\s+\|.*\|\s*FS\s+\(", line)
            if match:
                partition_id = match.group(1)
                break
        if partition_id is None:
            raise RuntimeError(f"No root filesystem partition found in {sif_path}")

        info = await self._runner.run(
            [
                self._settings.binary,
                "sif",
                "info",
                partition_id,
                str(sif_path),
            ],
            env=self._settings.subprocess_environment(),
            timeout_sec=30,
            check=False,
        )
        if info.return_code != 0 or not info.stdout:
            raise RuntimeError(
                f"Unable to inspect SIF root filesystem for {sif_path}: " f"{info.stderr or 'unknown error'}"
            )
        arch_match = re.search(r"^\s*Architecture:\s*(\S+)", info.stdout, re.M)
        if not arch_match:
            raise RuntimeError(f"Unable to parse SIF architecture for {sif_path}")
        return SifMetadata(
            path=sif_path,
            architecture=normalize_architecture(arch_match.group(1)),
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind-mount modeling and resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tasktrace_capture.apptainer.config import (
    CONTEXT_TARGET,
    TRANSFER_TARGET,
    RuntimeLayout,
)


@dataclass(frozen=True)
class BindMount:
    source: Path
    target: str
    read_only: bool = False

    def apptainer_spec(self) -> str:
        source = str(self.source)
        if any(char in source or char in self.target for char in (",", "\n", "\x00")):
            raise ValueError(
                "Apptainer mount paths cannot contain commas, newlines, or NUL " f"bytes: {source!r} -> {self.target!r}"
            )
        mode = ",ro" if self.read_only else ""
        return f"type=bind,src={source},dst={self.target}{mode}"


def mount_spec(source: Path, target: str, read_only: bool = False) -> str:
    """Compatibility helper for callers that do not need a mount object."""
    return BindMount(source=source, target=target, read_only=read_only).apptainer_spec()


def resolve_mounts(
    configured: Sequence[Mapping[str, Any]],
    *,
    layout: RuntimeLayout,
    environment_dir: Path,
    include_build_context: bool,
) -> list[BindMount]:
    """Resolve Harbor mount configuration into concrete host paths."""
    mounts: list[BindMount] = []
    configured_targets = {str(mount["target"]) for mount in configured if mount.get("type") == "bind"}

    for source_name, target in (
        ("/etc/resolv.conf", "/etc/resolv.conf"),
        ("/etc/hosts", "/etc/hosts"),
    ):
        source = Path(source_name)
        if target not in configured_targets and source.exists():
            mounts.append(BindMount(source.resolve(), target, read_only=True))

    for source, target in (
        (layout.tmp, "/tmp"),
        (layout.var_tmp, "/var/tmp"),
    ):
        if target not in configured_targets:
            mounts.append(BindMount(source.resolve(), target))

    for mount in configured:
        source = Path(str(mount["source"])).expanduser().resolve()
        if not source.exists():
            bind_options = mount.get("bind", {})
            create = bind_options.get("create_host_path", True)
            if create:
                source.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Apptainer bind source does not exist: {source}")
        mounts.append(
            BindMount(
                source=source,
                target=str(mount["target"]),
                read_only=bool(mount.get("read_only", False)),
            )
        )

    mounts.append(BindMount(layout.transfer.resolve(), TRANSFER_TARGET))
    if include_build_context:
        mounts.append(BindMount(environment_dir.resolve(), CONTEXT_TARGET, read_only=True))
    return mounts

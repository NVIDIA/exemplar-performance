# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host capability detection shared by Apptainer preflight and image setup."""

from __future__ import annotations

import os
import pwd
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_FUSE_HELPERS = ("squashfuse_ll", "squashfuse")
_TRUE_VALUES = {"1", "yes", "true"}


@dataclass(frozen=True)
class DirectSifSupport:
    """How this Apptainer installation can mount a SIF without extraction."""

    available: bool
    mode: str
    detail: str = ""
    helper: Path | None = None


def subordinate_id_mapping_available(
    *,
    subuid_path: Path = Path("/etc/subuid"),
    subgid_path: Path = Path("/etc/subgid"),
    environment: Mapping[str, str] | None = None,
    uid: int | None = None,
    username: str | None = None,
) -> bool:
    """Return whether Apptainer can create a full fakeroot ID mapping."""
    current_uid = os.getuid() if uid is None else uid
    if username is None:
        try:
            username = pwd.getpwuid(current_uid).pw_name
        except KeyError:
            username = os.environ.get("USER", str(current_uid))
    owners = {username, str(current_uid)}

    def has_owner(path: Path) -> bool:
        try:
            for line in path.read_text().splitlines():
                if line.partition(":")[0] in owners:
                    return True
        except OSError:
            return False
        return False

    path = environment.get("PATH") if environment is not None else None
    return (
        has_owner(subuid_path)
        and has_owner(subgid_path)
        and shutil.which("newuidmap", path=path) is not None
        and shutil.which("newgidmap", path=path) is not None
    )


def buildcfg_value(output: str, name: str) -> str | None:
    """Read one value from ``apptainer buildcfg`` output."""
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*(.*?)\s*$",
        output,
        re.MULTILINE,
    )
    return match.group(1) if match else None


def is_setuid_install(buildcfg_output: str) -> bool:
    value = buildcfg_value(buildcfg_output, "APPTAINER_SUID_INSTALL")
    return value is not None and value.strip().lower() in _TRUE_VALUES


def find_squashfuse(
    buildcfg_output: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Find the SIF mount helper using Apptainer's own search order."""
    libexec = buildcfg_value(buildcfg_output, "LIBEXECDIR")
    if libexec:
        helper_dir = Path(libexec) / "apptainer" / "bin"
        for name in _FUSE_HELPERS:
            candidate = helper_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

    path = environment.get("PATH") if environment is not None else None
    for name in _FUSE_HELPERS:
        resolved = shutil.which(name, path=path)
        if resolved is not None:
            return Path(resolved)
    return None


def fuse_device_access(
    device: Path = Path("/dev/fuse"),
) -> tuple[bool, str]:
    """Verify that the current user can open the FUSE device."""
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(device, flags)
    except OSError as exc:
        return False, f"{device}: {exc.strerror or exc}"
    os.close(descriptor)
    return True, ""


def direct_sif_support(
    buildcfg_output: str,
    *,
    environment: Mapping[str, str] | None = None,
    fuse_device: Path = Path("/dev/fuse"),
) -> DirectSifSupport:
    """Return the conservative FUSE path TaskTrace Capture requires for direct SIFs."""
    helper = find_squashfuse(buildcfg_output, environment=environment)
    if helper is None:
        return DirectSifSupport(
            available=False,
            mode="unavailable",
            detail=("this non-setuid installation has neither squashfuse_ll nor " "squashfuse"),
        )

    fuse_ok, fuse_detail = fuse_device_access(fuse_device)
    if not fuse_ok:
        return DirectSifSupport(
            available=False,
            mode="unavailable",
            detail=f"{helper.name} is installed but {fuse_detail}",
            helper=helper,
        )

    return DirectSifSupport(
        available=True,
        mode=f"{helper.name} + {fuse_device}",
        helper=helper,
    )


def probe_squashfuse(helper: Path) -> tuple[bool, str]:
    """Catch helpers that exist but cannot start, such as missing-library builds."""
    try:
        result = subprocess.run(
            [str(helper), "-h"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)

    output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    # squashfuse variants return 0, 1, or 2 for help depending on libfuse.
    if result.returncode in {0, 1, 2} and "squashfuse" in output:
        return True, ""
    detail = (result.stderr or result.stdout or "").strip()
    return False, detail or f"helper exited with status {result.returncode}"

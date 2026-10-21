# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and filesystem layout for the Apptainer runtime."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tasktrace_capture.apptainer.installation import (
    ensure_apptainer_installation,
    inspect_apptainer_installation,
    managed_compatibility_environment,
)

APPTAINER_INSTANCE_PREFIX = "tasktrace-capture-hb-"
CONTEXT_TARGET = "/run/tasktrace-capture-context"
TRANSFER_TARGET = "/run/tasktrace-capture-transfer"
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def apptainer_binary() -> str | None:
    """Return the already-selected Apptainer executable, without installing."""
    selected = inspect_apptainer_installation()
    return str(selected.binary) if selected.binary is not None else None


def require_apptainer_binary() -> str:
    """Return an explicit override or stage TaskTrace Capture's managed runtime."""
    selected = ensure_apptainer_installation()
    if selected.binary is None:
        raise RuntimeError(selected.detail)
    return str(selected.binary)


def apptainer_runtime_root() -> Path:
    """Return the node-local root for writable runtime state."""
    configured = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_TMPDIR")
    if configured:
        return Path(configured).expanduser()
    return Path(tempfile.gettempdir()) / f"tasktrace-capture-apptainer-{os.getuid()}"


def clean_apptainer_environment() -> dict[str, str]:
    """Return the isolated host environment used to launch Apptainer."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("APPTAINER_", "SINGULARITY_", "APPTAINERENV_", "SINGULARITYENV_")) or key in {
            "DOCKER_USERNAME",
            "DOCKER_PASSWORD",
        }:
            env.pop(key, None)
    return managed_compatibility_environment(env)


def sanitize_instance_name(session_id: str) -> str:
    """Create a stable, runtime-safe instance name for a Harbor session."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "-", session_id).strip(".-").lower()
    if not normalized:
        normalized = "trial"
    digest = hashlib.sha256(session_id.encode()).hexdigest()[:10]
    max_payload = 96 - len(APPTAINER_INSTANCE_PREFIX) - len(digest) - 1
    normalized = normalized[:max_payload].rstrip(".-")
    return f"{APPTAINER_INSTANCE_PREFIX}{normalized}-{digest}"


def normalize_architecture(value: str) -> str:
    """Normalize common OCI, SIF, and host architecture aliases."""
    normalized = value.strip().lower()
    return {
        "aarch64": "arm64",
        "arm64": "arm64",
        "x86_64": "amd64",
        "x86-64": "amd64",
        "amd64": "amd64",
    }.get(normalized, normalized)


@dataclass(frozen=True)
class RuntimeLayout:
    """All host paths owned by one Apptainer trial."""

    root: Path
    instance_name: str

    @property
    def trial(self) -> Path:
        return self.root / self.instance_name

    @property
    def overlay(self) -> Path:
        return self.trial / "overlay"

    @property
    def home(self) -> Path:
        return self.trial / "home"

    @property
    def transfer(self) -> Path:
        return self.trial / "transfer"

    @property
    def contain(self) -> Path:
        return self.trial / "contain"

    @property
    def tmp(self) -> Path:
        return self.trial / "tmp"

    @property
    def var_tmp(self) -> Path:
        return self.trial / "var-tmp"

    @property
    def apptainer_tmp(self) -> Path:
        return self.trial / "apptainer-tmp"

    @property
    def pid_file(self) -> Path:
        return self.trial / "instance.pid"

    def prepare(self) -> None:
        """Create a fresh, private workspace for this trial."""
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.trial.is_symlink():
            raise RuntimeError(f"Refusing symlinked Apptainer runtime directory: {self.trial}")
        if self.trial.exists():
            shutil.rmtree(self.trial)
        for path in (
            self.trial,
            self.overlay / "upper",
            self.overlay / "work",
            self.home,
            self.transfer,
            self.contain,
            self.tmp,
            self.var_tmp,
            self.apptainer_tmp,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.tmp.chmod(0o1777)
        self.var_tmp.chmod(0o1777)

    def remove(self) -> None:
        # OverlayFS leaves its private work directory mode 000 and Apptainer may
        # finish releasing it just after `instance stop` returns. Restore owner
        # traversal, then verify the tree stays gone rather than silently leaking
        # a directory that the runtime recreated after rmtree returned.
        if self.trial.exists():
            time.sleep(0.2)
        settle_delays = (0.05, 0.1, 0.2, 0.4)
        last_error: OSError | None = None
        for settle_delay in settle_delays:
            if not self.trial.exists():
                return
            if self.trial.is_dir():
                for root, directories, _ in os.walk(
                    self.trial,
                    topdown=True,
                    followlinks=False,
                ):
                    for name in directories:
                        path = Path(root) / name
                        if path.is_symlink():
                            continue
                        try:
                            mode = path.stat().st_mode
                            path.chmod(mode | stat.S_IRWXU)
                        except FileNotFoundError:
                            pass
            try:
                shutil.rmtree(self.trial)
            except FileNotFoundError:
                return
            except OSError as error:
                last_error = error
            time.sleep(settle_delay)

        if self.trial.exists():
            raise RuntimeError(f"Failed to remove Apptainer runtime directory: {self.trial}") from last_error


@dataclass(frozen=True)
class ApptainerSettings:
    """Runtime configuration captured once when an environment is created."""

    binary: str
    layout: RuntimeLayout

    @classmethod
    def load(
        cls,
        *,
        session_id: str,
        runtime_root: str | Path | None = None,
    ) -> "ApptainerSettings":
        instance_name = sanitize_instance_name(session_id)
        root = Path(runtime_root or apptainer_runtime_root()).expanduser()
        return cls(
            binary=require_apptainer_binary(),
            layout=RuntimeLayout(root=root, instance_name=instance_name),
        )

    @property
    def instance_name(self) -> str:
        return self.layout.instance_name

    @property
    def host_architecture(self) -> str:
        return normalize_architecture(platform.machine())

    def subprocess_environment(
        self,
        injected: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build a clean Apptainer process environment.

        Apptainer and Singularity variables are CLI controls, not ordinary
        container environment. Remove all inherited controls and add back only
        TaskTrace Capture-owned temporary state plus explicit container variables.
        """
        env = clean_apptainer_environment()

        env["APPTAINER_TMPDIR"] = str(self.layout.apptainer_tmp)

        for key, value in (injected or {}).items():
            if not ENV_KEY_RE.match(key):
                raise ValueError(f"Invalid environment variable name: {key!r}")
            env[f"APPTAINERENV_{key}"] = str(value)

        return env

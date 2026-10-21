# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime selection helpers.

Each supported runtime is a Harbor environment plugin. This module centralizes:
- The import-path string tasktrace-capture passes to ``harbor run --environment-import-path``
- Runtime-specific preflight checks
- Runtime-specific best-effort cleanup (orphaned-container sweep)

Supported runtimes today: ``enroot`` (default), ``docker``, ``apptainer``,
and ``opensandbox``. Future runtimes plug in by adding a branch here and a
``*_env.py`` module. The rest of tasktrace-capture is runtime-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

_logger = logging.getLogger(__name__)

SUPPORTED_RUNTIMES = ("enroot", "docker", "apptainer", "opensandbox", "current")
_MIN_APPTAINER_VERSION = (1, 4)


def _parse_apptainer_version(output: str) -> tuple[int, int, int] | None:
    """Extract a comparable Apptainer version from ``apptainer version``."""
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", output)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def runtime_import_path(name: str) -> str:
    """Return the Harbor --environment-import-path value for this runtime."""
    if name == "enroot":
        return "tasktrace_capture.enroot_env:EnrootEnvironment"
    if name == "docker":
        return "tasktrace_capture.docker_env:TasktraceCaptureDockerEnv"
    if name == "apptainer":
        return "tasktrace_capture.apptainer_env:ApptainerEnvironment"
    if name == "opensandbox":
        return "tasktrace_capture.opensandbox_env:OpenSandboxEnvironment"
    if name == "current":
        return "tasktrace_capture.current_env:CurrentEnvironment"
    raise ValueError(f"Unknown runtime: {name!r}. Supported: {', '.join(SUPPORTED_RUNTIMES)}.")


def _current_container_checks() -> list[tuple[str, bool, str]]:
    """Preflight for --runtime current: this process must already be inside the task container."""
    from tasktrace_capture.current_env import (
        CONTAINER_ENVIRON,
        IMAGE_REF_ENV,
        WORKDIR_ENV,
        container_environment,
        current_workdir,
        logs_root,
    )

    workdir = current_workdir()
    checks = [
        (
            f"task workdir exists ({workdir})",
            Path(workdir).is_dir(),
            f"Run tasktrace-capture INSIDE the task image, or point {WORKDIR_ENV} at the task workdir",
        )
    ]
    root = logs_root()
    probe = root if root.exists() else root.parent
    checks.append(
        (
            f"{root} creatable/writable",
            os.access(probe, os.W_OK),
            f"The shell instrumentation lives under {root}; make it writable in this container",
        )
    )
    image_ref = os.environ.get(IMAGE_REF_ENV)
    checks.append(
        (
            f"{IMAGE_REF_ENV} set" + (f" ({image_ref})" if image_ref else ""),
            bool(image_ref),
            f"Export {IMAGE_REF_ENV}=<image the container was created from> so the task image is verified",
        )
    )
    try:
        container_environment()
        readable = True
    except OSError:
        readable = False
    checks.append(
        (
            f"{CONTAINER_ENVIRON} readable",
            readable,
            "Agent commands start from the container's own environment; run tasktrace-capture as the "
            "container's user or root",
        )
    )
    return checks


def _claude_code_cache_check() -> tuple[str, bool, str]:
    """Detect a stale claude-code binary cached at /tmp/tasktrace_capture_claude_code_cache.

    The cache is shared across runtimes and is NOT auto-invalidated when
    ``DEFAULT_CLAUDE_CODE_VERSION`` is bumped. A stale binary silently
    propagates into every trial container and can break sweeps in subtle
    ways (e.g. the 2.1.97 → 2.1.121 bump fixed a hang where claude-code
    didn't exit after ``--print`` finished). Surface mismatches in
    preflight so users explicitly clear the cache instead of debugging
    a phantom AgentTimeoutError.
    """
    from tasktrace_capture.runtimes._shared import (
        CLAUDE_CODE_CACHE_DIR,
        DEFAULT_CLAUDE_CODE_VERSION,
        cached_claude_code_version,
    )

    cached = cached_claude_code_version()
    label = f"claude-code cache version (expected {DEFAULT_CLAUDE_CODE_VERSION})"
    # No cache yet is fine — the first run will populate it correctly.
    if cached is None or cached == DEFAULT_CLAUDE_CODE_VERSION:
        return (label, True, "")
    return (
        label,
        False,
        f"Cached binary at {CLAUDE_CODE_CACHE_DIR}/claude is {cached} but "
        f"DEFAULT_CLAUDE_CODE_VERSION is {DEFAULT_CLAUDE_CODE_VERSION}. "
        f"Run `rm -rf {CLAUDE_CODE_CACHE_DIR}*` to invalidate the stale "
        f"cache, then retry. Or pass --claude-code-version explicitly + "
        f"--skip-preflight if you know what you're doing.",
    )


def _docker_bridge_network_check() -> tuple[str, bool, str]:
    """Verify Docker can create the user-defined bridge network Harbor needs."""
    label = "docker bridge network creation"
    name = f"tasktrace-capture-preflight-{uuid.uuid4().hex[:12]}"
    try:
        created = subprocess.run(
            ["docker", "network", "create", "--driver", "bridge", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return (
            label,
            False,
            "Docker could not create a disposable bridge network. " "Run `sudo systemctl restart docker`, then retry.",
        )
    if created.returncode != 0:
        detail = (created.stderr or created.stdout or "").strip()
        return (
            label,
            False,
            "Docker cannot create the bridge network Harbor uses. "
            "Run `sudo systemctl restart docker`, then retry." + (f" Docker said: {detail}" if detail else ""),
        )
    try:
        removed = subprocess.run(
            ["docker", "network", "rm", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return (
            label,
            False,
            f"Created disposable network {name!r} but could not remove it. "
            "Remove it with `docker network rm` and retry.",
        )
    if removed.returncode != 0:
        return (
            label,
            False,
            f"Created disposable network {name!r} but could not remove it. "
            "Remove it with `docker network rm` and retry.",
        )
    return (label, True, "")


def runtime_preflight_checks(name: str) -> list[tuple[str, bool, str]]:
    """Return list of (label, passed, remediation_hint) for the runtime.

    cli._check consumes this shape. Does not raise; failed checks become
    entries in the returned list with ``passed=False``.
    """
    if name == "enroot":
        return [
            (
                "enroot on PATH",
                shutil.which("enroot") is not None,
                "On NVIDIA HPC clusters: module load enroot",
            ),
            _claude_code_cache_check(),
        ]
    if name == "docker":
        checks: list[tuple[str, bool, str]] = []
        docker_path = shutil.which("docker")
        checks.append(
            (
                "docker on PATH",
                docker_path is not None,
                "Install Docker or use --runtime enroot",
            )
        )
        if docker_path is not None:
            try:
                rc = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    timeout=10,
                ).returncode
            except (subprocess.TimeoutExpired, OSError):
                rc = 1
            checks.append(
                (
                    "docker daemon reachable",
                    rc == 0,
                    "Start the Docker daemon (systemctl start docker) or fix socket perms",
                )
            )
            if rc == 0:
                checks.append(_docker_bridge_network_check())
        checks.append(_claude_code_cache_check())
        return checks
    if name == "apptainer":
        from tasktrace_capture.apptainer.capabilities import (
            direct_sif_support,
            probe_squashfuse,
            subordinate_id_mapping_available,
        )
        from tasktrace_capture.apptainer_env import (
            apptainer_runtime_root,
            clean_apptainer_environment,
            ensure_apptainer_installation,
        )

        checks: list[tuple[str, bool, str]] = []
        installation = ensure_apptainer_installation()
        binary = str(installation.binary) if installation.binary is not None else None
        source_label = "TaskTrace Capture userspace Apptainer" if installation.managed else "user-provided Apptainer"
        checks.append(
            (
                source_label + (f" ({binary})" if binary is not None else ""),
                binary is not None,
                installation.detail
                + (
                    ". To use an existing Conda, module, or system installation, "
                    "set TASKTRACE_CAPTURE_APPTAINER_BIN to its executable."
                    if installation.source == "managed"
                    else ""
                ),
            )
        )
        if (
            installation.managed
            and installation.external_binary is not None
            and installation.external_binary != installation.binary
        ):
            checks.append(
                (
                    "external Apptainer left untouched " f"({installation.external_binary})",
                    True,
                    "",
                )
            )
        if binary is not None:
            runtime_env = clean_apptainer_environment()
            parsed_version: tuple[int, int, int] | None = None
            try:
                version_result = subprocess.run(
                    [binary, "version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=runtime_env,
                )
                version_text = (version_result.stdout or version_result.stderr or "").strip()
                version_ok = version_result.returncode == 0
                if version_ok:
                    parsed_version = _parse_apptainer_version(version_text)
            except (subprocess.TimeoutExpired, OSError):
                version_text = ""
                version_ok = False
            checks.append(
                (
                    "apptainer executable works" + (f" ({version_text})" if version_text else ""),
                    version_ok,
                    "Run `apptainer version` and fix the installation",
                )
            )
            minimum_ok = parsed_version is not None and parsed_version[:2] >= _MIN_APPTAINER_VERSION
            checks.append(
                (
                    "apptainer version is 1.4 or newer" + (f" ({version_text})" if version_text else ""),
                    minimum_ok,
                    "Install Apptainer 1.4 or newer; older releases do not provide "
                    "the instance and rootless behavior TaskTrace Capture depends on",
                )
            )

        runtime_root = apptainer_runtime_root()
        try:
            runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            fs_result = subprocess.run(
                ["stat", "-f", "-c", "%T", str(runtime_root)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            fs_type = fs_result.stdout.strip().lower()
            local_ok = fs_result.returncode == 0 and fs_type not in {
                "cifs",
                "fuseblk",
                "gpfs",
                "lustre",
                "nfs",
                "nfs4",
                "smb2",
            }
        except (OSError, subprocess.TimeoutExpired):
            fs_type = ""
            local_ok = False
        checks.append(
            (
                "apptainer writable state uses a local filesystem" + (f" ({fs_type})" if fs_type else ""),
                local_ok,
                "Set TASKTRACE_CAPTURE_APPTAINER_TMPDIR to node-local storage such as /tmp",
            )
        )

        if binary is not None:
            unshare = shutil.which("unshare")
            userns_ok = False
            if unshare is not None:
                try:
                    userns_ok = (
                        subprocess.run(
                            [unshare, "-Ur", "true"],
                            capture_output=True,
                            timeout=10,
                        ).returncode
                        == 0
                    )
                except (subprocess.TimeoutExpired, OSError):
                    userns_ok = False

            try:
                buildcfg = subprocess.run(
                    [binary, "buildcfg"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=runtime_env,
                )
                buildcfg_ok = buildcfg.returncode == 0
                buildcfg_text = buildcfg.stdout or ""
            except (subprocess.TimeoutExpired, OSError):
                buildcfg_ok = False
                buildcfg_text = ""

            checks.append(
                (
                    "apptainer unprivileged user namespaces",
                    userns_ok,
                    "Enable unprivileged user namespaces. TaskTrace Capture's managed "
                    "Apptainer is intentionally non-setuid.",
                )
            )

            if userns_ok:
                full_mapping = subordinate_id_mapping_available(environment=runtime_env)
                ownership_mode = (
                    "subordinate UID/GID mappings" if full_mapping else "root-mapped only; prepared SIFs required"
                )
                checks.append(
                    (
                        f"apptainer ownership mode ({ownership_mode})",
                        True,
                        "",
                    )
                )

            if buildcfg_ok:
                sif_support = direct_sif_support(buildcfg_text)
                helper_ok = True
                helper_detail = ""
                if sif_support.available and sif_support.helper is not None:
                    helper_ok, helper_detail = probe_squashfuse(sif_support.helper)
                direct_sif_ok = sif_support.available and helper_ok
                direct_sif_mode = sif_support.mode if direct_sif_ok else "unavailable"
                direct_sif_detail = sif_support.detail or helper_detail
            else:
                direct_sif_ok = False
                direct_sif_mode = "unavailable"
                direct_sif_detail = "`apptainer buildcfg` failed"

            checks.append(
                (
                    f"apptainer direct SIF mounting ({direct_sif_mode})",
                    direct_sif_ok,
                    "Fix the Apptainer installation: install squashfuse_ll "
                    "(preferred) or squashfuse under its LIBEXECDIR, and ensure "
                    "/dev/fuse exists and is openable by the current user. "
                    "Without direct mounting, Apptainer expands each SIF into "
                    "temporary storage." + (f" Detected: {direct_sif_detail}." if direct_sif_detail else ""),
                )
            )

        checks.append(_claude_code_cache_check())
        return checks
    if name == "opensandbox":
        endpoint = bool(os.environ.get("OPENSANDBOX_URL", "").strip())
        api_key = bool(os.environ.get("OPENSANDBOX_API_KEY", "").strip())
        return [
            (
                "OpenSandbox endpoint configured",
                endpoint,
                "Set OPENSANDBOX_URL to the OpenSandbox control-plane URL.",
            ),
            (
                "OpenSandbox API key configured",
                api_key,
                "Set OPENSANDBOX_API_KEY to the service API key.",
            ),
        ]
    if name == "current":
        return [*_current_container_checks(), _claude_code_cache_check()]
    raise ValueError(f"Unknown runtime: {name!r}. Supported: {', '.join(SUPPORTED_RUNTIMES)}.")


def runtime_cleanup(name: str, docker_owner: str | None = None) -> None:
    """Best-effort sweep of leftover tasktrace-capture containers.

    Never raises. Called from the SIGINT handler and between concurrency
    levels.

    enroot path delegates to tasktrace_capture.cli._cleanup_containers (the existing
    enroot-specific teardown).

    docker path removes only Compose projects owned by this invocation.

    apptainer path stops only named instances with TaskTrace Capture's private
    ``tasktrace-capture-hb-`` prefix, then removes inactive per-trial local state.

    Harbor itself handles normal cleanup via Trial.run()'s finally block.
    This sweep is a safety net for SIGKILL / crash / double-Ctrl+C cases
    where Harbor's finally never runs.
    """
    if name == "current":
        # Nothing was created: the surrounding container outlives tasktrace-capture by design.
        return
    if name == "enroot":
        # Lazy import to avoid cycle: cli imports runtimes.
        from tasktrace_capture.cli import _cleanup_containers  # type: ignore

        try:
            _cleanup_containers()
        except Exception as exc:  # pragma: no cover — defensive
            _logger.warning("enroot cleanup raised: %s", exc)
        return
    if name == "docker":
        if not docker_owner:
            return
        try:
            r = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    "label=com.docker.compose.project",
                    "--format",
                    '{{.ID}} {{.Label "com.docker.compose.project"}}',
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            prefix = f"{docker_owner}--"
            ids = [
                line.split()[0]
                for line in r.stdout.splitlines()
                if len(line.split()) == 2 and line.split()[1].startswith(prefix)
            ]
            if ids:
                subprocess.run(
                    ["docker", "rm", "-f", *ids],
                    capture_output=True,
                    timeout=30,
                )
        except Exception as exc:  # best-effort
            _logger.warning("docker cleanup raised: %s", exc)
        return
    if name == "apptainer":
        from tasktrace_capture.apptainer.config import RuntimeLayout
        from tasktrace_capture.apptainer_env import (
            APPTAINER_INSTANCE_PREFIX,
            apptainer_binary,
            apptainer_runtime_root,
            clean_apptainer_environment,
        )

        binary = apptainer_binary()
        if binary is None:
            return
        active_names: set[str] = set()
        runtime_env = clean_apptainer_environment()
        try:
            listed = subprocess.run(
                [binary, "instance", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd="/",
                env=runtime_env,
            )
            if listed.returncode != 0:
                return
            payload = json.loads(listed.stdout or "{}")
            for instance in payload.get("instances", []):
                name_value = instance.get("instance") or instance.get("name")
                if isinstance(name_value, str):
                    active_names.add(name_value)
            for instance_name in sorted(active_names):
                if not instance_name.startswith(APPTAINER_INSTANCE_PREFIX):
                    continue
                stopped = subprocess.run(
                    [
                        binary,
                        "instance",
                        "stop",
                        "--timeout",
                        "5",
                        instance_name,
                    ],
                    capture_output=True,
                    timeout=20,
                    cwd="/",
                    env=runtime_env,
                )
                if stopped.returncode != 0:
                    stopped = subprocess.run(
                        [
                            binary,
                            "instance",
                            "stop",
                            "--force",
                            instance_name,
                        ],
                        capture_output=True,
                        timeout=20,
                        cwd="/",
                        env=runtime_env,
                    )
                if stopped.returncode == 0:
                    active_names.discard(instance_name)

            runtime_root = apptainer_runtime_root()
            if runtime_root.is_dir() and not runtime_root.is_symlink():
                for child in runtime_root.iterdir():
                    if (
                        child.is_dir()
                        and child.name.startswith(APPTAINER_INSTANCE_PREFIX)
                        and child.name not in active_names
                    ):
                        try:
                            RuntimeLayout(runtime_root, child.name).remove()
                        except Exception as exc:
                            _logger.warning(
                                "failed to remove Apptainer runtime directory %s: %s",
                                child,
                                exc,
                            )
        except Exception as exc:  # best-effort
            _logger.warning("apptainer cleanup raised: %s", exc)
        return
    if name == "opensandbox":
        # Per-trial cleanup is performed by OpenSandboxEnvironment.stop().
        # The remote TTL remains the crash/signal safety net.
        return
    raise ValueError(f"Unknown runtime: {name!r}. Supported: {', '.join(SUPPORTED_RUNTIMES)}.")

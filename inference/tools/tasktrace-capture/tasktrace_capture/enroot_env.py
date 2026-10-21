# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
EnrootEnvironment -- Harbor environment backend using NVIDIA Enroot.

Runs agent tasks inside Enroot containers on HPC/Slurm clusters where Docker
is unavailable. Enroot is a rootless, unprivileged container runtime that
natively imports OCI/Docker images.

Usage:
    harbor jobs start --environment-import-path enroot_env:EnrootEnvironment ...
    harbor trials start --environment-import-path enroot_env:EnrootEnvironment ...

Or in YAML config:
    environment:
      import_path: enroot_env:EnrootEnvironment
      docker_image: ubuntu:22.04
"""

import asyncio
import base64
import binascii
import fcntl
import functools
import hashlib
import json
import os
import platform
import random
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

from tasktrace_capture.runtimes._shared import (
    apply_api_key_rotation,
    ensure_claude_code_cached_on_host,
    forward_env_vars,
    is_agent_command,
    wait_at_barrier,
)
from tasktrace_capture.shell_capture import (
    command_starts_login_shell,
    install_login_shell_capture_command,
    instrument_agent_launch,
    prepare_shell_capture,
    shell_capture_enabled,
    with_image_bash_env,
)


def _escape_mount_path(path: str) -> str:
    """Escape special characters in a path for enroot --mount (fstab format).

    Enroot parses mount specs using fstab-style rules where spaces are field
    separators. Paths containing spaces must have them escaped as ``\\040``
    (the octal representation), matching the fstab convention.
    """
    return path.replace(" ", "\\040")


def _format_mount(src: str, dst: str) -> str:
    """Build an enroot ``--mount`` value with properly escaped paths."""
    return f"{_escape_mount_path(src)}:{_escape_mount_path(dst)}"


_SENSITIVE_KEYWORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _is_sensitive_key(key: str) -> bool:
    """Return True if the env var key looks like it holds a secret."""
    upper = key.upper()
    return any(kw in upper for kw in _SENSITIVE_KEYWORDS)


_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def _mask_cmd_for_logging(cmd: list[str]) -> list[str]:
    """Return a copy of *cmd* with sensitive ``KEY=VALUE`` args masked."""
    masked = []
    for arg in cmd:
        if _ENV_ASSIGNMENT_RE.match(arg):
            key, _, _ = arg.partition("=")
            if _is_sensitive_key(key):
                masked.append(f"{key}=****")
                continue
        masked.append(arg)
    return masked


_DOCKER_DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_IMAGE_ENV_EXEC_SCRIPT = (
    "if [ -r /etc/environment ]; then "
    'while IFS= read -r line || [ -n "$line" ]; do '
    'case $line in "#"*) ;; *=*) set -- "$line" "$@" ;; esac; '
    "done < /etc/environment; "
    "fi; "
    f'exec /usr/bin/env -i HOME="$HOME" PATH={_DOCKER_DEFAULT_PATH} "$@"'
)


def image_env_exec_argv(env: dict[str, str] | None, argv: list[str]) -> list[str]:
    assignments = [f"{key}={value}" for key, value in (env or {}).items()]
    return ["/bin/sh", "-c", _IMAGE_ENV_EXEC_SCRIPT, "sh", *assignments, *argv]


def _enroot_process_env() -> dict[str, str]:
    return {**os.environ, "ENROOT_MOUNT_HOME": "no"}


def _unshare_prefix() -> tuple[str, ...]:
    return ("unshare", "--user", "--map-current-user", "--pid", "--fork", "--kill-child=SIGKILL", "--mount-proc", "--")


def _bubblewrap_prefix() -> tuple[str, ...]:
    uid, gid = str(os.geteuid()), str(os.getegid())
    return (
        "bwrap",
        "--dev-bind",
        "/",
        "/",
        "--unshare-user",
        "--uid",
        uid,
        "--gid",
        gid,
        "--unshare-pid",
        "--proc",
        "/proc",
        "--die-with-parent",
        "--",
    )


@functools.lru_cache(maxsize=1)
def pid_namespace_prefix() -> tuple[str, ...]:
    def works(argv: list[str]) -> bool:
        try:
            probe = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            return probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    if works(["unshare", "--user", "--map-current-user", "/bin/true"]):
        return _unshare_prefix()
    if works([*_bubblewrap_prefix()[:-1], "/bin/true"]):
        return _bubblewrap_prefix()
    return _unshare_prefix()


# Lock file path for serializing the one-time home symlink setup across
# all concurrent containers.  Every container needs /home/<user> -> /root
# but only the first one should perform the rm + ln; the rest just wait.
_HOME_SYMLINK_LOCK = Path(tempfile.gettempdir()) / "tasktrace_capture_home_symlink.lock"

# Claude-code binary cache (Issue #33) lives in tasktrace_capture.runtimes._shared
# and is used by both enroot and docker runtimes.
# Barrier, API-key rotation, env-var forwarding, agent-command detection
# also live there. This module re-exports them for backwards compatibility
# with existing test imports (imported at top of file).

# Shared cache directory for tmux (Issue #29). Terminus-2 builds tmux from
# source when the image does not provide it; at c>=8, N concurrent builds time
# out. Build/export once here, then inject the cached binary into containers so
# Harbor skips its fallback build path.
_TMUX_CACHE_DIR = Path(tempfile.gettempdir()) / "tasktrace_capture_tmux_cache"
_TMUX_CACHE_LOCK = _TMUX_CACHE_DIR.with_suffix(".lock")
_TMUX_VERSION = "3.4"
_TMUX_TARBALL = f"tmux-{_TMUX_VERSION}.tar.gz"
_TMUX_RELEASE_URL = f"https://github.com/tmux/tmux/releases/download/{_TMUX_VERSION}/{_TMUX_TARBALL}"

# Maximum retries for enroot create/start operations that can fail under
# high concurrency due to namespace exhaustion or filesystem contention.
_ENROOT_MAX_RETRIES = 5
_ENROOT_RETRY_BASE_SEC = 0.5  # exponential backoff: base * 2^attempt + jitter
_ENROOT_NETWORK_POLICY_ENV = "_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY"
_ENROOT_NETWORK_POLICY_MODES = {"strict", "best-effort"}

# SWE-bench image remapping — epoch-research/SWE-bench hosts pre-built images on
# GHCR with arm64 support (1819/2294 tasks), unlike the original Princeton Docker
# Hub images which are x86_64-only.  When _TASKTRACE_CAPTURE_SWEBENCH_IMAGES=epoch-research,
# tasks whose directory name matches the SWE-bench instance ID pattern are routed
# to the epoch-research GHCR registry for the host architecture.
#
# Instance ID format:  {owner}__{repo}-{issue}   e.g. astropy__astropy-13236
# Epoch GHCR format:   ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}
#
# NOTE: We derive the instance ID from the task directory name, NOT from the
# Dockerfile FROM line.  The Princeton Docker Hub images embed an internal build
# number (e.g. "astropy_1776_astropy-13236") that differs from the canonical
# instance ID used by epoch-research.  The Harbor task directory is always named
# after the canonical instance ID, making it the reliable source.
#
# GHCR authentication: even public GHCR packages require a bearer token obtained
# via the token endpoint.  Enroot's anonymous auth path sends a bare
# GET /token which GHCR rejects with 403.  Setting a GitHub PAT in
# ~/.config/enroot/.credentials fixes this.  TaskTrace Capture writes a temporary
# credentials entry when GITHUB_TOKEN is set in the environment.
_EPOCH_SWEBENCH_REGISTRY = "ghcr.io/epoch-research/swe-bench.eval"
_SWEBENCH_INSTANCE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]+__[a-zA-Z0-9][a-zA-Z0-9_-]+-\d+$")
_HOST_ARCH = "arm64" if platform.machine() == "aarch64" else "x86_64"
# enroot credentials file location (XDG convention)
_ENROOT_CREDENTIALS_PATH = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "enroot" / ".credentials"


def _swebench_instance_from_envdir(environment_dir: Path) -> str | None:
    """Extract the SWE-bench instance ID from the task environment directory.

    Harbor names the task directory after the canonical instance ID
    (e.g. ``astropy__astropy-13236``).  The environment directory sits one
    level below: ``<task_dir>/environment/``.

    Returns the instance ID string if the parent directory name matches the
    SWE-bench ``owner__repo-issue`` pattern, otherwise ``None``.
    """
    name = environment_dir.parent.name
    if _SWEBENCH_INSTANCE_RE.match(name):
        return name
    return None


def _remap_swebench_image(image: str) -> str | None:
    """Return the epoch-research GHCR image when *image* is already in either
    SWE-bench registry format, correcting the architecture for this host.

    This is a secondary detection path for when the environment directory is not
    available.  The primary path (``_swebench_instance_from_envdir``) is preferred
    because the Princeton Docker Hub images embed an internal build number that
    differs from the canonical instance ID used by epoch-research.

    Returns ``None`` if *image* does not match any recognised SWE-bench pattern.
    """
    m = re.search(r"(?:sweb|swe-bench)\.eval\.(x86_64|arm64)\.([^:]+?)(?::.+)?$", image)
    if not m:
        return None
    instance_id = m.group(2)
    return f"{_EPOCH_SWEBENCH_REGISTRY}.{_HOST_ARCH}.{instance_id}:latest"


def _ensure_ghcr_credentials(logger=None) -> None:
    """Write a GHCR credentials entry for enroot when GITHUB_TOKEN is set.

    Enroot reads ``~/.config/enroot/.credentials`` (netrc format) for registry
    authentication.  GHCR rejects anonymous token requests with 403; a PAT
    with at least ``read:packages`` scope resolves this.

    This function is idempotent: if a ``ghcr.io`` entry already exists it does
    nothing.  The entry is written only for the duration of the tasktrace-capture process
    (no persistent change if the token wasn't previously configured).
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return

    cred_path = _ENROOT_CREDENTIALS_PATH
    try:
        existing = cred_path.read_text() if cred_path.exists() else ""
    except OSError:
        existing = ""

    if "ghcr.io" in existing:
        return  # already configured — don't overwrite

    entry = f"machine ghcr.io login x-access-token password {token}\n"
    try:
        cred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cred_path, "a") as f:
            f.write(entry)
        if logger:
            logger.info("Wrote GHCR credentials to %s (from GITHUB_TOKEN)", cred_path)
    except OSError as exc:
        if logger:
            logger.warning("Could not write GHCR credentials to %s: %s", cred_path, exc)


async def _retry_cmd(coro_factory, max_retries=_ENROOT_MAX_RETRIES, base_sec=_ENROOT_RETRY_BASE_SEC, logger=None):
    """Retry an async command with exponential backoff and jitter.

    *coro_factory* is a zero-arg callable that returns a fresh coroutine on
    each invocation (because a coroutine object can only be awaited once).
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except RuntimeError as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            delay = base_sec * (2**attempt) + random.uniform(0, base_sec)
            if logger:
                logger.warning(
                    "Enroot command failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    exc,
                )
            await asyncio.sleep(delay)
    raise last_exc  # unreachable, but keeps type checkers happy


def _ensure_claude_code_binary(logger=None):
    """Backwards-compat shim: delegate to tasktrace_capture.runtimes._shared."""
    return ensure_claude_code_cached_on_host(logger=logger)


def _sanitize_container_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9_-]", "-", name)
    if not re.match(r"^[a-z0-9]", name):
        name = "hb" + name
    return name


def _sqsh_cache_key(image: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", image)
    short_hash = hashlib.sha256(image.encode()).hexdigest()[:8]
    return f"{sanitized}_{short_hash}.sqsh"


# The per-container home mounted over /root (Issue #50) hides the image's own /root
# dotfiles. SWE-bench images activate their `testbed` conda env from /root/.bashrc, and
# hiding it left the live agent on the base interpreter (no pytest) while a plain replay
# of the same image, with no /root mount, ran `testbed`. So /root/.bashrc is read from the
# image before the mount exists and carried into the isolated home.
#
# Only .bashrc is carried over, and only bash ever sources it: enroot's login init runs
# root's login shell from /etc/passwd, which is dash on some images, and under dash a
# failed `.` in .profile aborts the shell (and every exec) instead of continuing as bash
# does. The generated .profile therefore sources .bashrc only when running under bash, and
# the image's .profile/.bash_profile are not copied.
_HOME_BASHRC_ADDITION = (
    "# tasktrace-capture: per-container tool installs land in ~/.local/bin (Issue #50)\n"
    'export PATH="$HOME/.local/bin:$PATH"\n'
)
_HOME_PROFILE_STUB = (
    "# container profile: bash-only, so image shell code never runs under a dash login shell\n"
    'if [ -n "$BASH" ] && [ -f "$HOME/.bashrc" ]; then . "$HOME/.bashrc"; fi\n'
)
_HOME_BASH_PROFILE_STUB = (
    "# container bash_profile: defer to .profile as bash does when this file is absent\n"
    '[ -f "$HOME/.profile" ] && . "$HOME/.profile"\n'
)
# Prints the base64 of /root/.bashrc, or nothing. Stays silent (-> stub) when base64 is
# missing, when /root is a mount point (a host home mounted by enroot, not the image), or
# when the file is absent or unreadable, so a failure can never be mistaken for content.
_READ_IMAGE_BASHRC_SCRIPT = (
    "command -v base64 >/dev/null 2>&1 || exit 0; "
    "awk '$5 == \"/root\" { found = 1 } END { exit !found }' /proc/self/mountinfo 2>/dev/null && exit 0; "
    "[ -f /root/.bashrc ] || exit 0; "
    "enc=$(base64 -w0 /root/.bashrc) || exit 0; "
    'printf "%s\\n" "$enc"'
)

# Harbor's mini-swe-agent installer runs uv's install.sh and then hardcodes
# `source "$HOME/.local/bin/env"`. uv writes that file only when it also has to add
# ~/.local/bin to PATH; with the image's .bashrc preserved above, the login shell already
# exports ~/.local/bin, so uv skips the file and harbor's source fails. Ship the same
# script uv would write. uv leaves an existing file alone, so ours is what gets sourced.
_LOCAL_BIN_ENV_SCRIPT = (
    "#!/bin/sh\n"
    "# add binaries to PATH if they aren't added yet\n"
    "# affix colons on either side of $PATH to simplify matching\n"
    'case ":${PATH}:" in\n'
    '    *:"$HOME/.local/bin":*)\n'
    "        ;;\n"
    "    *)\n"
    "        # Prepending path in case a system-installed binary needs to be overridden\n"
    '        export PATH="$HOME/.local/bin:$PATH"\n'
    "        ;;\n"
    "esac\n"
)


def decode_image_bashrc(stdout: str) -> str | None:
    """Decode ``_READ_IMAGE_BASHRC_SCRIPT`` output; ``None`` when there was nothing usable."""
    payload = (stdout or "").strip()
    if not payload:
        return None
    try:
        return base64.b64decode(payload, validate=True).decode("utf-8", "replace")
    except (ValueError, binascii.Error):
        return None


def compose_home_dotfiles(image_bashrc: str | None) -> dict[str, str]:
    """The isolated home's login-shell files.

    ``.bashrc`` is the image's own (conda activation and the like) with our ``~/.local/bin``
    PATH export appended, or a stub plus the export when the image has none. ``.profile``
    sources ``.bashrc`` only under bash. ``.bash_profile`` defers to ``.profile`` the way
    bash does when the file is absent; a bare stub would shadow ``.profile`` entirely."""
    if image_bashrc is None:
        bashrc = "# container bashrc\n"
    else:
        bashrc = image_bashrc if image_bashrc.endswith("\n") else image_bashrc + "\n"
    return {
        ".bashrc": bashrc + _HOME_BASHRC_ADDITION,
        ".profile": _HOME_PROFILE_STUB,
        ".bash_profile": _HOME_BASH_PROFILE_STUB,
    }


class EnrootEnvironment(BaseEnvironment):
    """Harbor environment backend using NVIDIA Enroot containers."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        sqsh_cache_dir: str | None = None,
        *args,
        **kwargs,
    ):
        self._container_name = _sanitize_container_name(f"hb__{session_id}")
        self._sqsh_cache_dir = Path(
            sqsh_cache_dir
            or os.environ.get(
                "HARBOR_ENROOT_CACHE",
                str(Path.home() / ".cache" / "harbor" / "enroot"),
            )
        )
        self._mounts: list[tuple[str, str]] = []
        self._docker_image_from_dockerfile: str | None = None
        self._workdir: str | None = None
        self._barrier_passed = False

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

    # -- Required abstract properties / methods --------------------------------

    @staticmethod
    def type() -> EnvironmentType:
        # Unused when loaded via import_path; must return a valid enum member.
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(gpus=True, mounted=True)

    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        """Validate network policy without silently weakening task isolation.

        Enroot shares the host network namespace on the HPC systems supported
        by TaskTrace Capture. Strict mode therefore retains Harbor's fail-closed
        behavior. An explicit best-effort mode permits the task to run with
        public host networking, while writing machine-readable provenance into
        the trial directory and emitting a prominent warning.
        """
        policy = network_policy or self._network_policy
        mode = os.environ.get(_ENROOT_NETWORK_POLICY_ENV, "strict").strip().lower()
        if mode not in _ENROOT_NETWORK_POLICY_MODES:
            raise ValueError(
                f"{_ENROOT_NETWORK_POLICY_ENV} must be one of {sorted(_ENROOT_NETWORK_POLICY_MODES)}, got {mode!r}"
            )
        if policy.network_mode == NetworkMode.PUBLIC or mode == "strict":
            return super().validate_network_policy_support(policy)

        record = {
            "schema_version": 1,
            "runtime": "enroot",
            "requested": policy.network_mode.value,
            "requested_allowed_hosts": list(policy.allowed_hosts),
            "effective": NetworkMode.PUBLIC.value,
            "effective_allowed_hosts": [],
            "enforced": False,
            "compatibility_mode": mode,
            "reason": (
                "Enroot shares the host network namespace and cannot enforce "
                "the requested egress policy on this target."
            ),
        }
        record_path = self.trial_paths.trial_dir / "network_policy.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(record_path)
        self.logger.warning(
            "NETWORK ISOLATION NOT ENFORCED: task requested network_mode=%s%s, "
            "but Enroot is running with public host networking because "
            "%s=%s. Recorded provenance at %s.",
            policy.network_mode.value,
            (f" allowed_hosts={list(policy.allowed_hosts)!r}" if policy.allowed_hosts else ""),
            _ENROOT_NETWORK_POLICY_ENV,
            mode,
            record_path,
        )

    # -- Validation ------------------------------------------------------------

    def _validate_definition(self):
        if self.task_env_config.docker_image:
            return

        # No docker_image set -- try to extract the base image from the Dockerfile.
        dockerfile = self.environment_dir / "Dockerfile"
        if dockerfile.exists():
            image, workdir = self._parse_dockerfile(dockerfile)
            self._docker_image_from_dockerfile = image
            if workdir:
                self._workdir = workdir
                self.logger.info("Parsed WORKDIR from Dockerfile: %s", workdir)
            if image:
                self.logger.info(
                    "No docker_image set; using base image from Dockerfile: %s",
                    image,
                )
                return

        raise ValueError(
            "EnrootEnvironment requires either 'docker_image' in the task config "
            "or a Dockerfile with a FROM line in the environment directory. "
            "Provide a registry image (e.g., 'ubuntu:22.04'), a .sqsh path, "
            "or ensure the task has a Dockerfile."
        )

    @staticmethod
    def _parse_dockerfile(dockerfile: Path) -> tuple[str | None, str | None]:
        """Extract base image (FROM) and WORKDIR from a Dockerfile.

        Returns (image, workdir). Enroot only imports the base image and
        ignores Dockerfile directives like WORKDIR, so we need to parse
        and apply WORKDIR ourselves.
        """
        image = None
        workdir = None
        for line in dockerfile.read_text().splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM ") and image is None:
                parts = stripped.split()
                if len(parts) >= 2:
                    candidate = parts[1]
                    if candidate.lower() != "scratch":
                        image = candidate
            elif stripped.upper().startswith("WORKDIR "):
                parts = stripped.split(None, 1)
                if len(parts) >= 2:
                    workdir = parts[1]
        return image, workdir

    @classmethod
    def preflight(cls) -> None:
        if not shutil.which("enroot"):
            raise SystemExit("enroot is not installed or not on PATH. On NVIDIA HPC clusters, try: module load enroot")

    # -- Internal helpers ------------------------------------------------------

    async def _run_cmd(
        self,
        cmd: list[str],
        timeout_sec: int | None = None,
        check: bool = True,
    ) -> ExecResult:
        self.logger.debug("Running: %s", " ".join(_mask_cmd_for_logging(cmd)))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            env=_enroot_process_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            process_group=0,  # own PGID so timeout kills the whole tree
        )

        try:
            if timeout_sec:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            # Kill the entire process group, not just the enroot wrapper.
            # Without this, children (compilers, brute-force attacks, etc.)
            # survive and run indefinitely on the host.
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                stdout_bytes, stderr_bytes = await process.communicate()
            raise RuntimeError(
                f"Enroot command timed out after {timeout_sec}s: {' '.join(_mask_cmd_for_logging(cmd))}"
            ) from None

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None

        result = ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )

        if check and result.return_code != 0:
            raise RuntimeError(
                f"Enroot command failed (rc={result.return_code}): "
                f"{' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        return result

    async def _import_image(self, image: str, force: bool = False) -> Path:
        """Import a Docker/OCI image to a cached .sqsh file.

        Uses a file lock to prevent concurrent agents from importing
        the same image simultaneously (which corrupts the .sqsh file).
        """
        if image.endswith(".sqsh") and Path(image).is_file():
            self.logger.info("Using existing .sqsh file: %s", image)
            return Path(image)

        self._sqsh_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._sqsh_cache_dir / _sqsh_cache_key(image)

        lock_path = cache_path.with_suffix(".lock")
        lock_fd = open(lock_path, "w")
        try:
            self.logger.info("Acquiring import lock for %s ...", cache_path.name)
            await asyncio.get_event_loop().run_in_executor(None, lambda: fcntl.flock(lock_fd, fcntl.LOCK_EX))

            if cache_path.exists() and not force:
                self.logger.info("Using cached image: %s", cache_path)
                return cache_path

            docker_uri = image if image.startswith("docker://") else f"docker://{image}"
            self.logger.info("Importing image %s (this may take a while)...", docker_uri)

            tmp_path = cache_path.with_suffix(".sqsh.tmp")
            if tmp_path.exists():
                tmp_path.unlink()

            await self._run_cmd(
                ["enroot", "import", "--output", str(tmp_path), docker_uri],
                timeout_sec=self.task_env_config.build_timeout_sec,
            )

            tmp_path.rename(cache_path)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

        return cache_path

    # -- Lifecycle -------------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        image = self.task_env_config.docker_image or self._docker_image_from_dockerfile
        if not image:
            raise RuntimeError(
                "No container image available. Set docker_image or provide a Dockerfile with a FROM line."
            )

        # Remap SWE-bench images to epoch-research GHCR when requested.
        # epoch-research hosts arm64-compatible images for 1819/2294 SWE-bench tasks.
        if os.environ.get("_TASKTRACE_CAPTURE_SWEBENCH_IMAGES") == "epoch-research":
            # Primary: derive instance ID from the task directory name.
            # The Harbor task dir is always named after the canonical instance ID
            # (e.g. astropy__astropy-13236), which matches epoch-research's naming.
            # The Princeton Docker Hub image names embed an internal build number
            # (e.g. astropy_1776_astropy-13236) that differs — do NOT use the FROM line.
            instance_id = _swebench_instance_from_envdir(self.environment_dir)
            if instance_id:
                remapped = f"{_EPOCH_SWEBENCH_REGISTRY}.{_HOST_ARCH}.{instance_id}:latest"
                self.logger.info(
                    "Remapping SWE-bench task %s to epoch-research GHCR (%s): %s",
                    instance_id,
                    _HOST_ARCH,
                    remapped,
                )
                image = remapped
                _ensure_ghcr_credentials(self.logger)

        sqsh_path = await self._import_image(image, force=force_build)

        # Remove stale container with the same name.
        # Wrap in retry — at c>=128, concurrent `enroot list` calls can
        # transiently fail due to namespace/filesystem contention.
        try:
            existing = await _retry_cmd(
                lambda: self._run_cmd(["enroot", "list"], check=False),
                max_retries=3,
                logger=self.logger,
            )
            if existing.stdout and self._container_name in existing.stdout.split():
                self.logger.info("Removing stale container: %s", self._container_name)
                await self._run_cmd(["enroot", "remove", "-f", self._container_name], check=False)
        except RuntimeError:
            # If listing fails after retries, proceed anyway — the container
            # probably doesn't exist yet.  enroot create will fail if it does.
            pass

        # Create the container with retry + backoff.  At high concurrency
        # (c>=128), enroot create can fail due to namespace exhaustion,
        # filesystem mount-point limits, or /tmp contention.
        self.logger.info("Creating container: %s", self._container_name)
        await _retry_cmd(
            lambda: self._run_cmd(["enroot", "create", "--name", self._container_name, str(sqsh_path)]),
            logger=self.logger,
        )

        if force_build:
            # A forced refresh of a mutable tag may have changed the image; do not reuse the cached read.
            EnrootEnvironment._image_bashrc_cache.pop(image, None)
        image_bashrc = await self._read_image_bashrc(image)

        # --- Per-container isolation setup ---
        # Create an isolated home on local /tmp for performance (Lustre is
        # too slow for small-file I/O at c=128).  Use the container name for
        # uniqueness — each container already has a unique session-based name.
        # The home is cleaned up in stop() via shutil.rmtree.
        self._claude_cache = self.trial_paths.trial_dir / "home"
        # Security: reject pre-existing symlinks to prevent symlink attacks
        # (Issue #23). An attacker could plant a symlink at the predictable
        # path pointing to a sensitive directory.
        if self._claude_cache.is_symlink():
            raise RuntimeError(
                f"Refusing to use cache path that is a symlink: {self._claude_cache}. "
                f"This may indicate a symlink attack."
            )
        self._claude_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Build an isolated home directory structure inside the per-container
        # tmpdir.  This entire dir is mounted as /root, so the layout here
        # mirrors what the container sees as its home.  (Issue #50: prevents
        # task installs — opam, nvm, pip, cargo — from polluting host /home.)
        local_bin_path = self._claude_cache / ".local" / "bin"
        # Create dirs that tools commonly expect under $HOME.
        (self._claude_cache / ".claude").mkdir(parents=True, exist_ok=True)
        (self._claude_cache / ".local" / "share").mkdir(parents=True, exist_ok=True)
        local_bin_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Atomic-ish creation: write to a temp file then rename to
        # avoid partial reads if another process somehow accesses it.
        # The image's own .bashrc comes first so its login setup (e.g. conda activation)
        # still runs under bash; our PATH addition is appended.
        for name, content in compose_home_dotfiles(image_bashrc).items():
            path = self._claude_cache / name
            tmp = path.with_suffix(".tmp")
            tmp.write_text(content)
            tmp.rename(path)
        env_script = local_bin_path / "env"
        if not env_script.exists():
            env_script.write_text(_LOCAL_BIN_ENV_SCRIPT)
            env_script.chmod(0o755)
        # Pre-install claude-code binary (Issue #33).  At c>=64, concurrent
        # downloads fail for most containers.  Install once to a shared cache,
        # then copy the binary into each container's isolated .local/bin so
        # Harbor's agent setup finds it already on PATH and skips downloading.
        agent_name = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
        if agent_name == "claude-code":
            cached_binary = _ensure_claude_code_binary(logger=self.logger)
            if cached_binary and cached_binary.exists():
                dest = local_bin_path / "claude"
                if not dest.exists():
                    shutil.copy2(str(cached_binary), str(dest))
                    os.chmod(str(dest), 0o755)
                    self.logger.info(
                        "Pre-installed claude-code binary into container %s",
                        self._container_name,
                    )
        elif agent_name.startswith("terminus"):
            cached_binary = await self._ensure_tmux_binary()
            if cached_binary and cached_binary.exists():
                existing_tmux = await self.exec(
                    command="tmux -V || /usr/local/bin/tmux -V",
                    user="root",
                )
                if existing_tmux.return_code != 0:
                    await self.upload_file(cached_binary, "/usr/local/bin/tmux")
                    await self.exec(
                        command="chmod 755 /usr/local/bin/tmux",
                        user="root",
                    )
                    self.logger.info(
                        "Pre-installed tmux binary into container %s",
                        self._container_name,
                    )

        self._mounts = [
            (
                str(self.trial_paths.agent_dir.resolve()),
                str(EnvironmentPaths.agent_dir),
            ),
            (
                str(self.trial_paths.verifier_dir.resolve()),
                str(EnvironmentPaths.verifier_dir),
            ),
            (
                str(self.trial_paths.artifacts_dir.resolve()),
                str(EnvironmentPaths.artifacts_dir),
            ),
            # Isolate entire /root home per container (Issue #50).
            # Enroot bind-mounts the host $HOME at /root — without this,
            # task installs (opam, nvm, pip, cargo) pollute the real /home.
            # The tmpdir mirrors the home layout: .claude/, .local/bin/,
            # .bashrc, etc. are pre-created above.
            (str(self._claude_cache), "/root"),
        ]

        # Ensure log dirs and WORKDIR exist inside the container.
        # Use cwd="/" to avoid the default workdir cd (dir may not exist yet).
        # Retry — the first exec after create can fail if enroot hasn't
        # finished initializing the rootfs.
        mkdir_targets = f"{EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} {EnvironmentPaths.artifacts_dir}"
        if self._workdir:
            mkdir_targets += f" {shlex.quote(self._workdir)}"
        await _retry_cmd(
            lambda: self._exec_checked(
                f"mkdir -p {mkdir_targets} && "
                f"chmod 777 {EnvironmentPaths.agent_dir} "
                f"{EnvironmentPaths.verifier_dir} "
                f"{EnvironmentPaths.artifacts_dir}",
                cwd="/",
            ),
            max_retries=3,
            logger=self.logger,
        )

        # Enroot mounts the host home at /root but /etc/passwd still lists
        # the host user's home (e.g. /home/<user>). Tools that resolve HOME
        # from passwd (like Claude Code's install script) create symlinks
        # pointing to /home/<user>/... which is dangling inside the container.
        #
        # Fix: replace the passwd home path with a symlink to /root.
        # Use a file lock so that only one container performs this operation
        # at a time — without the lock, 128 concurrent containers race on
        # the same path (rm + ln is not atomic, causing TOCTOU failures
        # where one container removes the symlink another just created).
        host_home = os.path.expanduser("~")
        if host_home != "/root":
            await self._setup_home_symlink(host_home)

    # One read per image per process: the result is an image property, and a full
    # `enroot start` per trial would add a third contended container start to the
    # startup window at high concurrency. Failures are not cached so a later trial
    # can still succeed.
    _image_bashrc_cache: dict[str, str | None] = {}
    _image_bashrc_locks: dict[str, asyncio.Lock] = {}

    async def _read_image_bashrc(self, image: str) -> str | None:
        """/root/.bashrc as shipped in the image, read before the isolated home hides it.

        Never fails the run: any error after retries, or an image without the file,
        yields ``None`` and the isolated home gets the stub .bashrc it always had."""
        cache = EnrootEnvironment._image_bashrc_cache
        lock = EnrootEnvironment._image_bashrc_locks.setdefault(image, asyncio.Lock())
        async with lock:
            if image in cache:
                return cache[image]
            # ENROOT_MOUNT_HOME=no keeps enroot's home hook from mounting the host home
            # at /root for this start, so we read the image, not the user's dotfiles.
            # Same retry as the first exec after `enroot create`: the rootfs may not be
            # ready yet. The script always exits 0, so a non-zero status is enroot failing.
            try:
                result = await _retry_cmd(
                    lambda: self._run_cmd(
                        [
                            "env",
                            "ENROOT_MOUNT_HOME=no",
                            "enroot",
                            "start",
                            "--root",
                            self._container_name,
                            "bash",
                            "-c",
                            _READ_IMAGE_BASHRC_SCRIPT,
                        ],
                        timeout_sec=60,
                        check=True,
                    ),
                    max_retries=3,
                    logger=self.logger,
                )
            except Exception as exc:  # noqa: BLE001 - best effort, never fail the run
                self.logger.warning("Could not read the image's /root/.bashrc, using the stub: %s", exc)
                return None
            content = decode_image_bashrc(result.stdout or "")
            if content is None:
                self.logger.info(
                    "Image %s has no readable /root/.bashrc (or /root is a host mount); isolated home uses stubs",
                    image,
                )
            else:
                self.logger.info(
                    "Preserving the image's /root/.bashrc (%d bytes) in the isolated home for %s", len(content), image
                )
            cache[image] = content
            return content

    async def _wait_at_barrier(self) -> None:
        """Delegate to tasktrace_capture.runtimes._shared.wait_at_barrier."""
        await wait_at_barrier(
            container_name=self._container_name,
            logger=self.logger,
        )

    async def _prepare_login_shell_capture(self, env: dict[str, str], user: str | int | None) -> None:
        """Enable the shared tracer before Terminus creates its login shell."""
        if getattr(self, "_tasktrace_capture_trace_login_ready", False):
            return
        prepare_shell_capture(self.trial_paths.agent_dir)
        result = await self.exec(
            command=install_login_shell_capture_command(),
            env=env,
            user=user,
        )
        if result.return_code == 0:
            self._tasktrace_capture_trace_login_ready = True
        else:
            self.logger.warning("Unable to install Bash command timing hook")

    async def _ensure_tmux_binary(self) -> Path | None:
        """Build tmux once in a container and cache the resulting binary."""
        cache_dir = _TMUX_CACHE_DIR
        binary = cache_dir / "tmux"

        lock_fd = open(_TMUX_CACHE_LOCK, "w")
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: fcntl.flock(lock_fd, fcntl.LOCK_EX))

            if binary.exists() and os.access(str(binary), os.X_OK):
                self.logger.info("tmux binary already cached at %s", binary)
                return binary

            cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

            verify_result = await self.exec(
                command="tmux -V || /usr/local/bin/tmux -V",
                user="root",
            )
            if verify_result.return_code != 0:
                self.logger.info(
                    "Pre-installing tmux binary to shared cache %s ...",
                    cache_dir,
                )
                dep_commands = [
                    "DEBIAN_FRONTEND=noninteractive apt-get update && "
                    "DEBIAN_FRONTEND=noninteractive apt-get install -y "
                    "build-essential libevent-dev libncurses5-dev curl",
                    "yum groupinstall -y 'Development Tools' && yum install -y libevent-devel ncurses-devel curl",
                    "dnf groupinstall -y 'Development Tools' && dnf install -y libevent-devel ncurses-devel curl",
                    "apk add --no-cache build-base libevent-dev ncurses-dev curl",
                ]
                for cmd in dep_commands:
                    result = await self.exec(command=cmd, user="root")
                    if result.return_code == 0:
                        break

                build_result = await self.exec(
                    command=(
                        "cd /tmp && "
                        f"curl -L {_TMUX_RELEASE_URL} -o {_TMUX_TARBALL} && "
                        f"tar -xzf {_TMUX_TARBALL} && "
                        f"cd tmux-{_TMUX_VERSION} && "
                        "./configure --prefix=/usr/local && "
                        "make && "
                        "make install"
                    ),
                    user="root",
                )
                if build_result.return_code != 0:
                    self.logger.warning(
                        "Failed to pre-install tmux (rc=%d): %s",
                        build_result.return_code,
                        build_result.stderr[-500:] if build_result.stderr else "",
                    )
                    return None

                verify_result = await self.exec(
                    command="tmux -V || /usr/local/bin/tmux -V",
                    user="root",
                )
                if verify_result.return_code != 0:
                    self.logger.warning("tmux build succeeded but verification failed")
                    return None

            tmux_path = await self.exec(
                command="command -v tmux || { [ -x /usr/local/bin/tmux ] && echo /usr/local/bin/tmux; }",
                user="root",
            )
            if tmux_path.return_code != 0 or not tmux_path.stdout:
                self.logger.warning("Unable to locate tmux binary in container")
                return None

            source_path = tmux_path.stdout.strip().splitlines()[-1]
            tmp_binary = cache_dir / f"tmux.{self._container_name}.tmp"
            if tmp_binary.exists():
                tmp_binary.unlink()
            await self.download_file(source_path, tmp_binary)
            os.chmod(str(tmp_binary), 0o755)
            tmp_binary.replace(binary)
            self.logger.info("tmux binary cached at %s", binary)
            return binary
        except Exception as exc:
            self.logger.warning("Failed to pre-install tmux: %s", exc)
            return None
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    async def _exec_checked(self, command: str, cwd: str | None = None) -> ExecResult:
        """Run a command inside the container, raising on non-zero exit.

        Unlike ``exec()`` (which always returns an ExecResult with check=False),
        this wrapper raises ``RuntimeError`` on failure so that ``_retry_cmd``
        can catch it and retry.
        """
        result = await self.exec(command, cwd=cwd)
        if result.return_code != 0:
            raise RuntimeError(
                f"Container command failed (rc={result.return_code}): {command}"
                f"\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    async def _setup_home_symlink(self, host_home: str) -> None:
        """Create /home/<user> -> /root symlink inside the container.

        Serialized via a file lock to prevent the TOCTOU race where 128+
        concurrent containers all try to rm + ln the same path simultaneously.
        """
        hq = shlex.quote(host_home)
        lock_fd = open(_HOME_SYMLINK_LOCK, "w")
        try:
            await asyncio.get_event_loop().run_in_executor(None, lambda: fcntl.flock(lock_fd, fcntl.LOCK_EX))
            await _retry_cmd(
                lambda: self._exec_checked(
                    f"if [ -L {hq} ]; then true; "
                    f"elif [ -d {hq} ]; then rm -rf {hq} && ln -s /root {hq}; "
                    f"else mkdir -p $(dirname {hq}) && ln -s /root {hq}; fi",
                    cwd="/",
                ),
                max_retries=3,
                logger=self.logger,
            )
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

        # Ensure /bin/sh exists. Some base images (especially minimal ones)
        # have /bin/bash but not /bin/sh. Harbor's upload_dir and terminus-2's
        # tmux pipe-pane internally invoke /bin/sh, which causes
        # "enroot-switchroot: failed to execute: /bin/sh: No such file or
        # directory" when it's missing. Symlink /bin/sh -> /bin/bash as a fix.
        await self.exec(
            "if [ ! -e /bin/sh ] && [ -x /bin/bash ]; then ln -s /bin/bash /bin/sh; fi",
            cwd="/",
        )

    async def stop(self, delete: bool):
        # Don't try to "kill -9 -1" inside the container — enroot has no PID
        # namespaces, so that would kill ALL our processes, not just this
        # container's.  Just remove the container; the orphan sweep in
        # _cleanup_harbor_containers() handles escaped processes safely
        # via the protected-namespace approach.

        try:
            await self._run_cmd(["enroot", "remove", "-f", self._container_name], check=False)
        except Exception as e:
            self.logger.warning("Failed to remove container %s: %s", self._container_name, e)

        # Clean up per-container .claude cache
        if hasattr(self, "_claude_cache") and self._claude_cache.exists():
            shutil.rmtree(self._claude_cache, ignore_errors=True)

        if delete:
            image = self.task_env_config.docker_image
            if image and not (image.endswith(".sqsh") and Path(image).is_file()):
                cache_path = self._sqsh_cache_dir / _sqsh_cache_key(image)
                if cache_path.exists():
                    cache_path.unlink()
                    self.logger.info("Removed cached image: %s", cache_path)

    # -- Command execution -----------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        env = self._merge_env(env)

        # Forward --agent-env vars listed in _TASKTRACE_CAPTURE_FORWARD_ENVS, and
        # rotate API keys listed in _TASKTRACE_CAPTURE_API_KEYS. Both behaviours live
        # in tasktrace_capture.runtimes._shared so docker_env uses the same logic.
        env = forward_env_vars(env or {})
        env = apply_api_key_rotation(
            env,
            container_name=self._container_name,
            logger=self.logger,
        )
        env = with_image_bash_env(env)
        agent_hint = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
        if shell_capture_enabled():
            command, env = instrument_agent_launch(
                command,
                agent_hint,
                env,
                self.trial_paths.agent_dir,
            )
            if command_starts_login_shell(command, agent_hint):
                await self._prepare_login_shell_capture(env=env, user=user)

        # Skip harbor's Claude Code install if binary is already pre-installed
        # from the local cache (issue #40).  Harbor's install() unconditionally
        # curls install.sh, which can 429.  Detect the install command by the
        # presence of "install" + "curl" + "claude --version" (harbor always
        # appends a version check).  If claude is already on PATH, short-circuit.
        if "install" in command and "claude --version" in command and "curl" in command:
            check = await self._run_cmd(
                [
                    "enroot",
                    "start",
                    "--rw",
                    self._container_name,
                    "bash",
                    "-c",
                    'export PATH="$HOME/.local/bin:$PATH"; claude --version',
                ],
                check=False,
            )
            if check.return_code == 0:
                self.logger.info(
                    "claude-code already installed (%s), skipping download",
                    (check.stdout or "").strip(),
                )
                return check

        # Barrier: wait for all agents before the actual agent execution.
        # Harbor's run() makes two exec_as_agent() calls: a setup command
        # (mkdir for config dirs) then the real agent command.  We detect
        # the real command by looking for agent-specific signatures.
        if not self._barrier_passed and is_agent_command(command, agent_hint):
            self._barrier_passed = True
            await self._wait_at_barrier()

        cmd = [*pid_namespace_prefix(), "enroot", "start", "--rw"]

        for host_path, container_path in self._mounts:
            cmd.extend(["--mount", _format_mount(host_path, container_path)])

        # Pass env vars via --env. Sensitive values are masked in _run_cmd
        # debug logging via _mask_cmd_for_logging.
        if env:
            for key, value in env.items():
                cmd.extend(["--env", f"{key}={value}"])

        if user is not None and (str(user) == "root" or user == 0):
            cmd.append("--root")

        cmd.append(self._container_name)

        # Enroot has no --workdir flag and ignores Dockerfile WORKDIR.
        # Use explicit cwd if given, otherwise fall back to parsed WORKDIR.
        effective_cwd = cwd or self._workdir
        if effective_cwd:
            full_command = f"cd {shlex.quote(effective_cwd)} && {command}"
        else:
            full_command = command

        # Non-root user switching via su (similar to Modal's approach).
        if user is not None and str(user) != "root" and user != 0:
            full_command = f"su -s /bin/bash {shlex.quote(str(user))} -c {shlex.quote(full_command)}"

        cmd.extend(image_env_exec_argv(env, ["bash", "-c", full_command]))

        return await self._run_cmd(cmd, timeout_sec=timeout_sec, check=False)

    # -- File transfers --------------------------------------------------------
    # Use enroot start with temporary bind mounts rather than direct rootfs
    # access, since the rootfs location varies across enroot configurations.

    def _upload_mount_path(self) -> str:
        """Return a per-container upload mount path to avoid predictable /tmp paths.

        Uses the container name for uniqueness (Issue #23).
        """
        return f"/tmp/_hb_upload_{self._container_name}"

    async def upload_file(self, source_path: Path | str, target_path: str):
        source = Path(source_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        target_parent = str(Path(target_path).parent)
        source_name = source.name
        upload_mount = self._upload_mount_path()
        await self._run_cmd(
            [
                "enroot",
                "start",
                "--rw",
                "--mount",
                _format_mount(str(source.parent), upload_mount),
                self._container_name,
                "bash",
                "-c",
                f"mkdir -p {shlex.quote(target_parent)} && "
                f"cp {upload_mount}/{shlex.quote(source_name)} {shlex.quote(target_path)}",
            ]
        )
        self.logger.debug("Uploaded %s -> %s", source, target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        source = Path(source_dir).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source directory not found: {source}")

        upload_mount = self._upload_mount_path()
        await self._run_cmd(
            [
                "enroot",
                "start",
                "--rw",
                "--mount",
                _format_mount(str(source), upload_mount),
                self._container_name,
                "bash",
                "-c",
                f"mkdir -p {shlex.quote(target_dir)} && cp -a {upload_mount}/. {shlex.quote(target_dir)}/",
            ]
        )
        self.logger.debug("Uploaded dir %s -> %s", source, target_dir)

    def _download_mount_path(self) -> str:
        """Return a per-container download mount path to avoid predictable /tmp paths."""
        return f"/tmp/_hb_download_{self._container_name}"

    async def download_file(self, source_path: str, target_path: Path | str):
        dest = Path(target_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        download_mount = self._download_mount_path()
        await self._run_cmd(
            [
                "enroot",
                "start",
                "--rw",
                "--mount",
                _format_mount(str(dest.parent), download_mount),
                self._container_name,
                "bash",
                "-c",
                f"cp {shlex.quote(source_path)} {download_mount}/{shlex.quote(dest.name)}",
            ]
        )

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        dest = Path(target_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        download_mount = self._download_mount_path()
        await self._run_cmd(
            [
                "enroot",
                "start",
                "--rw",
                "--mount",
                _format_mount(str(dest), download_mount),
                self._container_name,
                "bash",
                "-c",
                f"cp -a {shlex.quote(source_dir)}/. {download_mount}/",
            ]
        )

    # -- Interactive access ----------------------------------------------------

    async def attach(self) -> None:
        os.execvpe(
            "enroot",
            ["enroot", "start", "--rw", "--root", self._container_name, "bash"],
            _enroot_process_env(),
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CurrentEnvironment: run the Harbor trial inside the container that already runs tasktrace-capture.

The other runtimes create the task container and exec into it; here someone else created it (for example
a Kubernetes pod running the task image) and this process runs inside it. ``start`` and ``stop`` create
nothing: they link Harbor's ``/logs/agent``, ``/logs/verifier`` and ``/logs/artifacts`` to the trial
directories. ``exec`` runs ``bash -c`` in the task workdir with the docker/enroot adapters' semantics,
starting from the container's own environment (``/proc/1/environ``), not tasktrace-capture's. One trial
runs at a time: ``/logs/agent`` is a single path.

Environment: ``TASKTRACE_CAPTURE_CURRENT_IMAGE_REF`` names the image the container was created from; a
trial whose task names another image is refused unless ``TASKTRACE_CAPTURE_CURRENT_ALLOW_IMAGE_MISMATCH=1``.
``TASKTRACE_CAPTURE_CURRENT_WORKDIR`` overrides the workdir, ``TASKTRACE_CAPTURE_CURRENT_LOGS_ROOT`` the
``/logs`` root (tests only).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

from tasktrace_capture.runtimes._shared import (
    apply_api_key_rotation,
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

WORKDIR_ENV = "TASKTRACE_CAPTURE_CURRENT_WORKDIR"
IMAGE_REF_ENV = "TASKTRACE_CAPTURE_CURRENT_IMAGE_REF"
ALLOW_IMAGE_MISMATCH_ENV = "TASKTRACE_CAPTURE_CURRENT_ALLOW_IMAGE_MISMATCH"
# Set by the runner from --swebench-images; the enroot runtime reads it too.
SWEBENCH_IMAGES_ENV = "_TASKTRACE_CAPTURE_SWEBENCH_IMAGES"
LOGS_ROOT_ENV = "TASKTRACE_CAPTURE_CURRENT_LOGS_ROOT"

_TRANSFER_TIMEOUT_SEC = 120
# The environment of the container's first process: what `docker exec` / `kubectl exec` give a command.
CONTAINER_ENVIRON = Path("/proc/1/environ")


def logs_root() -> Path:
    """The in-container root Harbor mounts trial logs under (``/logs`` unless overridden for tests)."""
    return Path(os.environ.get(LOGS_ROOT_ENV) or str(EnvironmentPaths.logs_dir))


def current_workdir(task_workdir: str | None = None, dockerfile_workdir: str | None = None) -> str:
    """Resolve the task workdir: explicit env override, task definition, Dockerfile WORKDIR, then ``/``."""
    return os.environ.get(WORKDIR_ENV) or task_workdir or dockerfile_workdir or "/"


def parse_dockerfile_image(dockerfile: Path) -> tuple[str | None, str | None]:
    """Return (FROM image, WORKDIR) of a task Dockerfile; nothing is built from it."""
    image: str | None = None
    workdir: str | None = None
    for raw_line in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        instruction, _, value = line.partition(" ")
        upper = instruction.upper()
        if upper == "FROM" and image is None:
            candidates = [part for part in value.split() if not part.startswith("--")]
            if candidates and candidates[0].lower() != "scratch":
                image = candidates[0]
        elif upper == "WORKDIR":
            workdir = value.strip()
    return image, workdir


def _split_image_ref(ref: str) -> tuple[str, str | None, str | None]:
    """(repository, tag-or-None, digest-or-None) for ``repo[:tag][@sha256:hex]`` and the ``repo:sha256:hex`` tag
    form Harbor task definitions use for content-pinned images."""
    ref = ref.strip()
    digest: str | None = None
    if "@sha256:" in ref:
        ref, digest = ref.split("@sha256:", 1)
    elif ":sha256:" in ref:
        ref, digest = ref.split(":sha256:", 1)
    tag: str | None = None
    slash = ref.rfind("/")
    colon = ref.rfind(":")
    if colon > slash:  # a ':' after the last '/' is a tag, not a registry port
        ref, tag = ref[:colon], ref[colon + 1 :]
    return ref.lower(), (tag or None), (digest.lower() if digest else None)


def image_refs_match(expected: str, actual: str) -> bool:
    """Is this container's image the one the task names? Fail closed: digests must be equal when both
    references carry one, tags when neither does; a pin that names its tag (``repo:tag@sha256:...``) matches it.
    TASKTRACE_CAPTURE_CURRENT_ALLOW_IMAGE_MISMATCH=1 is the operator's override."""
    expected_repo, expected_tag, expected_digest = _split_image_ref(expected)
    actual_repo, actual_tag, actual_digest = _split_image_ref(actual)
    if expected_repo != actual_repo:
        return False
    if expected_digest and actual_digest:
        return expected_digest == actual_digest
    if expected_digest or actual_digest:
        pinned_tag, other_tag = (expected_tag, actual_tag) if expected_digest else (actual_tag, expected_tag)
        return pinned_tag is not None and pinned_tag == (other_tag or "latest")
    return (expected_tag or "latest") == (actual_tag or "latest")


def epoch_swebench_image(environment_dir: Path) -> str | None:
    """The epoch-research image a SWE-bench task runs in when the remap is on (as in EnrootEnvironment.start),
    else None."""
    if os.environ.get(SWEBENCH_IMAGES_ENV) != "epoch-research":
        return None
    from tasktrace_capture.enroot_env import (
        _EPOCH_SWEBENCH_REGISTRY,
        _HOST_ARCH,
        _swebench_instance_from_envdir,
    )

    instance_id = _swebench_instance_from_envdir(environment_dir)
    if instance_id is None:
        return None
    return f"{_EPOCH_SWEBENCH_REGISTRY}.{_HOST_ARCH}.{instance_id}:latest"


def container_environment() -> dict[str, str]:
    """The environment the container was started with (image ENV plus the pod's), from CONTAINER_ENVIRON.
    Raises OSError when it cannot be read."""
    env = {}
    for entry in CONTAINER_ENVIRON.read_bytes().split(b"\0"):
        name, sep, value = entry.partition(b"=")
        if sep and name:
            env[os.fsdecode(name)] = os.fsdecode(value)
    return env


def _same_file(source: Path, target: Path) -> bool:
    try:
        return source.exists() and target.exists() and os.path.samefile(source, target)
    except OSError:
        return False


class CurrentEnvironment(BaseEnvironment):
    """Run a Harbor trial in the container this process already lives in."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        self._image_from_dockerfile: str | None = None
        self._workdir_from_dockerfile: str | None = None
        self._barrier_passed = False
        self._links: list[tuple[Path, Path]] = []  # (in-container path, trial directory)
        self._login_shell_capture_ready = False
        self._container_env: dict[str, str] | None = None
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            **kwargs,
        )

    @staticmethod
    def type() -> EnvironmentType:
        # Import-path environments must return an existing Harbor enum value.
        return EnvironmentType.DOCKER

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # The container's creator fixes its allocation.
        return EnvironmentResourceCapabilities()

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # The log directories are on this filesystem. Network isolation is the container creator's, so
        # tasks that require it fail closed.
        return EnvironmentCapabilities(mounted=True)

    @classmethod
    def preflight(cls) -> None:
        workdir = os.environ.get(WORKDIR_ENV)
        if workdir and not Path(workdir).is_dir():
            raise SystemExit(f"{WORKDIR_ENV}={workdir!r} is not a directory inside this container")

    def _validate_definition(self) -> None:
        if self.task_env_config.gpus:
            raise ValueError("CurrentEnvironment cannot allocate GPUs; the surrounding container fixes them")
        expected_image = self.task_env_config.docker_image
        dockerfile = self.environment_dir / "Dockerfile"
        if dockerfile.is_file():
            self._image_from_dockerfile, self._workdir_from_dockerfile = parse_dockerfile_image(dockerfile)
            expected_image = expected_image or self._image_from_dockerfile
        remapped = epoch_swebench_image(self.environment_dir)
        if remapped:
            self.logger.info("SWE-bench image remap: the task runs in %s, not %s", remapped, expected_image)
            expected_image = remapped
        workdir = self.workdir
        if not Path(workdir).is_dir():
            raise ValueError(
                f"task workdir {workdir!r} does not exist in this container; is tasktrace-capture "
                f"running inside the task image? (override with {WORKDIR_ENV})"
            )
        actual_image = os.environ.get(IMAGE_REF_ENV)
        if expected_image and actual_image and not image_refs_match(expected_image, actual_image):
            message = (
                f"this container was created from {actual_image!r} but the task expects "
                f"{expected_image!r}; the trial would run in the wrong image"
            )
            if os.environ.get(ALLOW_IMAGE_MISMATCH_ENV) == "1":
                self.logger.warning("%s (%s=1: continuing)", message, ALLOW_IMAGE_MISMATCH_ENV)
            else:
                raise ValueError(f"{message} (set {ALLOW_IMAGE_MISMATCH_ENV}=1 to override)")
        if self._mounts:
            self.logger.debug(
                "current-container runtime ignores %d requested host mounts (same filesystem)",
                len(self._mounts),
            )

    @property
    def workdir(self) -> str:
        return current_workdir(self.task_env_config.workdir, self._workdir_from_dockerfile)

    def _link_targets(self) -> list[tuple[Path, Path]]:
        root = logs_root()
        return [
            (root / EnvironmentPaths.agent_dir.name, self.trial_paths.agent_dir),
            (root / EnvironmentPaths.verifier_dir.name, self.trial_paths.verifier_dir),
            (root / EnvironmentPaths.artifacts_dir.name, self.trial_paths.artifacts_dir),
        ]

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise RuntimeError("CurrentEnvironment cannot build images; the surrounding container is the image")
        self._command_base_env()  # fail before the trial, not at the agent's first command
        root = logs_root()
        root.mkdir(parents=True, exist_ok=True)
        for link, target in self._link_targets():
            target.mkdir(parents=True, exist_ok=True)
            os.chmod(target, 0o777)
            if link.is_symlink():
                if _same_file(link, target):
                    continue  # the verifier session of the same trial, or a restart
                link.unlink()  # a previous (sequential) trial in this container
            elif link.is_dir():
                if any(link.iterdir()):
                    raise RuntimeError(
                        f"{link} is a non-empty directory; refusing to hide it behind a symlink to {target}"
                    )
                link.rmdir()
            elif link.exists():
                raise RuntimeError(f"{link} exists and is not a directory or symlink")
            link.symlink_to(target.resolve(), target_is_directory=True)
            self._links.append((link, target))
        self.logger.info("current-container runtime: trial logs linked under %s (workdir %s)", root, self.workdir)

    async def stop(self, delete: bool) -> None:
        for link, target in self._links:
            if link.is_symlink() and _same_file(link, target):
                link.unlink()
        self._links = []

    async def prepare_logs_for_host(self) -> None:
        return None  # same filesystem, same user

    def _command_base_env(self) -> dict[str, str]:
        if self._container_env is None:
            try:
                self._container_env = container_environment()
            except OSError as e:
                raise RuntimeError(
                    f"cannot read {CONTAINER_ENVIRON} ({e}); agent commands must start from the container's "
                    "own environment, not tasktrace-capture's"
                ) from e
        return self._container_env

    def _wrap_user(self, argv: list[str], user: str | int | None) -> list[str]:
        """Run as another user only when one is requested and it is not who we already are."""
        if user is None:
            return argv
        try:
            current_uid = os.getuid()
        except AttributeError:  # pragma: no cover — non-POSIX
            return argv
        if user in ("root", 0) and current_uid == 0:
            return argv
        if isinstance(user, int) or str(user).isdigit():
            if int(user) == current_uid:
                return argv
        elif os.environ.get("USER") == user:
            return argv
        # util-linux su takes names and #-prefixed numeric ids.
        target = f"#{user}" if isinstance(user, int) or str(user).isdigit() else str(user)
        return ["su", "-s", "/bin/bash", target, "-c", argv[-1]]

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        merged_env = forward_env_vars(dict(self._merge_env(env) or {}))
        merged_env = apply_api_key_rotation(merged_env, container_name=self.session_id, logger=self.logger)
        merged_env = with_image_bash_env(merged_env)
        agent = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
        if not self._barrier_passed and is_agent_command(command, agent):
            self._barrier_passed = True
            await wait_at_barrier(container_name=self.session_id, logger=self.logger)
        if shell_capture_enabled():
            command, merged_env = instrument_agent_launch(command, agent, merged_env, self.trial_paths.agent_dir)
            if command_starts_login_shell(command, agent):
                await self._prepare_login_shell_capture(env=merged_env, user=user)
        return await self._run_local(
            command,
            cwd=cwd or self.workdir,
            env=merged_env,
            timeout_sec=timeout_sec,
            user=self._resolve_user(user),
        )

    async def _run_local(
        self,
        command: str,
        *,
        cwd: str,
        env: dict[str, str],
        timeout_sec: int | None,
        user: str | int | None,
    ) -> ExecResult:
        argv = self._wrap_user(["/bin/bash", "-c", command], user)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env={**self._command_base_env(), **env},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # own process group → a timeout can kill the whole tree
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            with_children = process.pid
            try:
                os.killpg(with_children, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return ExecResult(
            stdout=stdout.decode(errors="replace") or None,
            stderr=stderr.decode(errors="replace") or None,
            return_code=process.returncode if process.returncode is not None else 1,
        )

    async def _prepare_login_shell_capture(self, env: dict[str, str], user: str | int | None) -> None:
        """Enable the shared tracer before Terminus creates its login shell."""
        if self._login_shell_capture_ready:
            return
        prepare_shell_capture(self.trial_paths.agent_dir)
        result = await self._run_local(
            install_login_shell_capture_command(),
            cwd=self.workdir,
            env=env,
            timeout_sec=_TRANSFER_TIMEOUT_SEC,
            user=self._resolve_user(user),
        )
        if result.return_code == 0:
            self._login_shell_capture_ready = True
        else:
            self.logger.warning("Unable to install Bash command timing hook")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source, target = Path(source_path), Path(target_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        if _same_file(source, target):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source, target = Path(source_dir), Path(target_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")
        if _same_file(source, target):
            return
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source, target = Path(source_path), Path(target_path)
        if _same_file(source, target):
            return
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source, target = Path(source_dir), Path(target_dir)
        if _same_file(source, target):
            return
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return Path(path).is_dir()

    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return Path(path).is_file()

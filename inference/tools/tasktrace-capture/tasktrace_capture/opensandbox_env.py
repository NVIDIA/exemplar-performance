# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor environment backend for remote OpenSandbox sandboxes.

The first supported contract is intentionally narrow: one prebuilt Linux
container image per trial, no Docker Compose, no image builds, no host bind
mounts, and no accelerators.  Harbor still owns the agent/verifier lifecycle;
this class translates its environment operations to the OpenSandbox SDK.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import tarfile
import tempfile
import uuid
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
)
from harbor.models.task.config import (
    NetworkPolicy as HarborNetworkPolicy,
)
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from opensandbox import Sandbox
from opensandbox.config import ConnectionConfig
from opensandbox.models import NetworkPolicy, NetworkRule
from opensandbox.models.execd import RunCommandOpts

from tasktrace_capture.runtimes._shared import (
    apply_api_key_rotation,
    forward_env_vars,
    is_agent_command,
    wait_at_barrier,
)

_DEFAULT_TTL_SECONDS = 4500
_DEFAULT_READY_TIMEOUT_SECONDS = 120
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 180
_DEFAULT_CPU_REQUEST = "0.25"
_DEFAULT_MEMORY_REQUEST = "512Mi"
_DEFAULT_CPU_LIMIT = "4"
_DEFAULT_MEMORY_LIMIT = "4Gi"
_INFERENCE_BASE_URL_ENV_NAMES = (
    "AGENT_API_BASE",
    "INFERENCE_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_BASE",
)
_SETUP_NETWORK_AGENTS = ("claude-code", "codex", "mini-swe-agent", "terminus")
_DEFAULT_SETUP_ALLOWED_HOSTS = (
    # OS packages needed by Harbor's agent installers.
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "ports.ubuntu.com",
    "deb.debian.org",
    "security.debian.org",
    # Some task images retain NodeSource as an apt source. Codex also installs
    # Node through nvm when the image is not Alpine-based.
    "deb.nodesource.com",
    "nodejs.org",
    "dl-cdn.alpinelinux.org",
    # uv and Python packages used by mini-swe-agent and Terminus.
    "astral.sh",
    "releases.astral.sh",
    "pypi.org",
    "files.pythonhosted.org",
    # Release artifacts used by uv and other supported agent installers.
    "github.com",
    "objects.githubusercontent.com",
    "raw.githubusercontent.com",
    "release-assets.githubusercontent.com",
    # Claude Code's native installer and npm fallback.
    "downloads.claude.ai",
    "registry.npmjs.org",
)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _connection_config() -> ConnectionConfig:
    raw_url = os.environ.get("OPENSANDBOX_URL", "").strip()
    api_key = os.environ.get("OPENSANDBOX_API_KEY", "").strip()
    if not raw_url:
        raise ValueError("OPENSANDBOX_URL is required for --runtime opensandbox")
    if not api_key:
        raise ValueError("OPENSANDBOX_API_KEY is required for --runtime opensandbox")
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OPENSANDBOX_URL must be an absolute http:// or https:// URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("OPENSANDBOX_URL must not include a path, query, or fragment")
    return ConnectionConfig(
        domain=parsed.netloc,
        api_key=api_key,
        # The lifecycle API consumes ``api_key`` directly, while sandbox
        # health/exec/file requests use the endpoint client headers.  Server
        # proxy deployments may require the same key on both paths.
        headers={"OPEN-SANDBOX-API-KEY": api_key},
        protocol=parsed.scheme,
        request_timeout=timedelta(
            seconds=_positive_int_env(
                "TASKTRACE_CAPTURE_OPENSANDBOX_REQUEST_TIMEOUT_SECONDS",
                _DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        ),
        use_server_proxy=True,
    )


def _attribution_metadata(*, environment_name: str, session_id: str) -> dict[str, str]:
    metadata = {
        "user": os.environ.get("TASKTRACE_CAPTURE_OPENSANDBOX_USER", os.environ.get("USER", "unknown")),
        "team": os.environ.get("TASKTRACE_CAPTURE_OPENSANDBOX_TEAM", "exemplar-performance"),
        "project": os.environ.get("TASKTRACE_CAPTURE_OPENSANDBOX_PROJECT", "tasktrace-capture"),
        "purpose": os.environ.get("TASKTRACE_CAPTURE_OPENSANDBOX_PURPOSE", "benchmark"),
        "task": environment_name,
        "trial": session_id,
    }
    raw_extra = os.environ.get("TASKTRACE_CAPTURE_OPENSANDBOX_METADATA", "").strip()
    if raw_extra:
        try:
            extra = json.loads(raw_extra)
        except json.JSONDecodeError as exc:
            raise ValueError("TASKTRACE_CAPTURE_OPENSANDBOX_METADATA must be a JSON object") from exc
        if not isinstance(extra, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in extra.items()
        ):
            raise ValueError("TASKTRACE_CAPTURE_OPENSANDBOX_METADATA must map strings to strings")
        metadata.update(extra)
    return metadata


def _inference_endpoint_host() -> str | None:
    """Return the host of the one configured inference endpoint, if any.

    A CI pipeline may set ``INFERENCE_BASE_URL``.  The provider-specific aliases keep
    raw TaskTrace Capture invocations compatible without broadening access to every URL
    that might happen to be present in the process environment.
    """

    source_name: str | None = None
    raw_url = ""
    for name in _INFERENCE_BASE_URL_ENV_NAMES:
        candidate = os.environ.get(name, "").strip()
        if candidate:
            source_name, raw_url = name, candidate
            break
    if source_name is None:
        return None

    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{source_name} must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{source_name} must not contain credentials")
    if parsed.fragment:
        raise ValueError(f"{source_name} must not contain a fragment")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{source_name} contains an invalid port") from exc
    if parsed.hostname is None:
        raise ValueError(f"{source_name} must contain a host")
    return parsed.hostname.lower()


def _network_policy(policy: HarborNetworkPolicy) -> NetworkPolicy:
    if policy.network_mode == NetworkMode.PUBLIC:
        return NetworkPolicy(defaultAction="allow", egress=[])

    allowed_hosts = list(policy.allowed_hosts)
    inference_host = _inference_endpoint_host()
    if inference_host is not None and inference_host not in allowed_hosts:
        allowed_hosts.append(inference_host)
    return NetworkPolicy(
        defaultAction="deny",
        egress=[NetworkRule(action="allow", target=host) for host in allowed_hosts],
    )


def _setup_allowed_hosts(agent: str) -> list[str]:
    if not any(agent == name or agent.startswith(f"{name}-") for name in _SETUP_NETWORK_AGENTS):
        return []
    raw_override = os.environ.get("TASKTRACE_CAPTURE_OPENSANDBOX_SETUP_ALLOWED_HOSTS")
    if raw_override is None:
        return list(_DEFAULT_SETUP_ALLOWED_HOSTS)
    return list(dict.fromkeys(host.strip().lower() for host in raw_override.split(",") if host.strip()))


def _add_allowed_hosts(policy: NetworkPolicy, hosts: list[str]) -> NetworkPolicy:
    if policy.default_action == "allow":
        return policy
    existing = [rule.target for rule in policy.egress]
    return NetworkPolicy(
        defaultAction="deny",
        egress=[
            NetworkRule(action="allow", target=host)
            for host in [*existing, *(host for host in hosts if host not in existing)]
        ],
    )


def _allowed_targets_digest(targets: list[str]) -> str:
    canonical = "\n".join(sorted(set(targets)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _mount_target(mount) -> str | None:
    if isinstance(mount, dict):
        value = mount.get("target")
    else:
        value = getattr(mount, "target", None)
    return str(value) if value else None


class OpenSandboxEnvironment(BaseEnvironment):
    """Run a Harbor trial in one OpenSandbox sandbox."""

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
        self._sandbox: Sandbox | None = None
        self._renew_task: asyncio.Task[None] | None = None
        self._image_from_dockerfile: str | None = None
        self._workdir_from_dockerfile: str | None = None
        self._barrier_passed = False
        self._ttl = timedelta(
            seconds=_positive_int_env("TASKTRACE_CAPTURE_OPENSANDBOX_TTL_SECONDS", _DEFAULT_TTL_SECONDS)
        )
        self._ready_timeout = timedelta(
            seconds=_positive_int_env(
                "TASKTRACE_CAPTURE_OPENSANDBOX_READY_TIMEOUT_SECONDS",
                _DEFAULT_READY_TIMEOUT_SECONDS,
            )
        )
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
        return EnvironmentResourceCapabilities(
            cpu_limit=True,
            cpu_request=True,
            memory_limit=True,
            memory_request=True,
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            disable_internet=True,
            network_allowlist=True,
        )

    @classmethod
    def preflight(cls) -> None:
        _connection_config()

    def _validate_definition(self) -> None:
        if self.task_env_config.gpus:
            raise ValueError("OpenSandboxEnvironment does not yet support GPU task requests")
        if self._mounts:
            self.logger.debug(
                "OpenSandbox uses transfer APIs instead of %d requested host mounts",
                len(self._mounts),
            )
        if self.task_env_config.docker_image:
            return

        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.is_file():
            raise ValueError(
                "OpenSandboxEnvironment requires [environment].docker_image or "
                "a Dockerfile containing only FROM and optional WORKDIR"
            )
        image: str | None = None
        workdir: str | None = None
        unsupported: list[str] = []
        for raw_line in dockerfile.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            instruction, _, value = line.partition(" ")
            upper = instruction.upper()
            if upper == "FROM" and image is None:
                parts = value.split()
                candidates = [part for part in parts if not part.startswith("--")]
                if candidates and candidates[0].lower() != "scratch":
                    image = candidates[0]
            elif upper == "WORKDIR":
                workdir = value.strip()
            else:
                unsupported.append(upper)
        if unsupported:
            raise ValueError(
                "OpenSandboxEnvironment cannot build Dockerfiles; unsupported "
                f"instructions: {', '.join(sorted(set(unsupported)))}. "
                "Publish a prebuilt image and set [environment].docker_image."
            )
        if not image:
            raise ValueError("OpenSandbox Dockerfile must contain a non-scratch FROM")
        self._image_from_dockerfile = image
        self._workdir_from_dockerfile = workdir

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError(
                "OpenSandboxEnvironment cannot force-build task Dockerfiles; " "publish a prebuilt image instead"
            )
        image = self.task_env_config.docker_image or self._image_from_dockerfile
        if not image:
            raise RuntimeError("OpenSandbox task image was not resolved")

        try:
            task_network_policy = _network_policy(self._network_policy)
            setup_hosts = _setup_allowed_hosts(os.environ.get("_TASKTRACE_CAPTURE_AGENT", ""))
            network_policy = _add_allowed_hosts(task_network_policy, setup_hosts)
            effective_mode = (
                "public"
                if network_policy.default_action == "allow"
                else "allowlist" if network_policy.egress else "no-network"
            )
            allowed_targets = [rule.target for rule in network_policy.egress]
            self.logger.info(
                "OpenSandbox network policy requested=%s effective=%s " "allowed_targets=%s enforced=true",
                self._network_policy.network_mode.value,
                effective_mode,
                allowed_targets,
            )
            metadata = _attribution_metadata(
                environment_name=self.environment_name,
                session_id=self.session_id,
            )
            metadata.update(
                {
                    "network_policy_requested": self._network_policy.network_mode.value,
                    "network_policy_effective": effective_mode,
                    "network_policy_enforced": "true",
                    "network_policy_allowed_target_count": str(len(allowed_targets)),
                    "network_policy_allowed_targets_hash": (_allowed_targets_digest(allowed_targets)),
                    "network_policy_agent_setup_target_count": str(
                        sum(host in allowed_targets for host in setup_hosts)
                    ),
                }
            )
            resource, resource_requests = self._resource_configuration()
            self._sandbox = await Sandbox.create(
                image,
                connection_config=_connection_config(),
                timeout=self._ttl,
                ready_timeout=self._ready_timeout,
                resource=resource,
                resource_requests=resource_requests,
                metadata=metadata,
                network_policy=network_policy,
            )
            self.logger.info("Created OpenSandbox sandbox %s for %s", self._sandbox.id, image)
            self._renew_task = asyncio.create_task(
                self._renew_until_stopped(),
                name=f"opensandbox-renew-{self.session_id}",
            )

            directories = {
                str(EnvironmentPaths.agent_dir),
                str(EnvironmentPaths.verifier_dir),
                str(EnvironmentPaths.artifacts_dir),
            }
            directories.update(target for target in map(_mount_target, self._mounts) if target)
            if self.task_env_config.workdir:
                directories.add(self.task_env_config.workdir)
            if self._workdir_from_dockerfile:
                directories.add(self._workdir_from_dockerfile)
            result = await self.exec(self._ensure_dirs_command(sorted(directories)), cwd="/", user="root")
            if result.return_code != 0:
                raise RuntimeError(
                    "Failed to prepare OpenSandbox trial directories: "
                    f"{result.stderr or result.stdout or 'no output'}"
                )
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(delete=True)
            raise

    def _resource_configuration(self) -> tuple[dict[str, str], dict[str, str]]:
        """Translate Harbor's effective task resource policy for OpenSandbox.

        ``BaseEnvironment`` has already applied numeric run overrides to the
        task configuration.  In AUTO mode, an explicit task value is treated
        as a Kubernetes guarantee (matching both request and limit); when the
        task omits a value, the OpenSandbox operator defaults remain the
        fallback.  Explicit REQUEST, LIMIT, GUARANTEE, and IGNORE modes are
        preserved through Harbor's policy helpers.
        """

        limits: dict[str, str] = {}
        requests: dict[str, str] = {}
        resources = (
            (
                "cpu",
                "",
                "TASKTRACE_CAPTURE_OPENSANDBOX_CPU_LIMIT",
                _DEFAULT_CPU_LIMIT,
                "TASKTRACE_CAPTURE_OPENSANDBOX_CPU_REQUEST",
                _DEFAULT_CPU_REQUEST,
            ),
            (
                "memory",
                "Mi",
                "TASKTRACE_CAPTURE_OPENSANDBOX_MEMORY_LIMIT",
                _DEFAULT_MEMORY_LIMIT,
                "TASKTRACE_CAPTURE_OPENSANDBOX_MEMORY_REQUEST",
                _DEFAULT_MEMORY_REQUEST,
            ),
        )
        for (
            resource,
            suffix,
            limit_env,
            default_limit,
            request_env,
            default_request,
        ) in resources:
            task_value = self._resource_value(resource)
            limit = self._resource_limit_value(resource, auto_mode=ResourceMode.GUARANTEE)
            request = self._resource_request_value(resource, auto_mode=ResourceMode.GUARANTEE)
            if limit is not None:
                limits[resource] = f"{limit}{suffix}"
            elif task_value is None and self._resource_mode(resource) == ResourceMode.AUTO:
                limits[resource] = os.environ.get(limit_env, default_limit)
            if request is not None:
                requests[resource] = f"{request}{suffix}"
            elif task_value is None and self._resource_mode(resource) == ResourceMode.AUTO:
                requests[resource] = os.environ.get(request_env, default_request)
        return limits, requests

    async def _renew_until_stopped(self) -> None:
        interval = max(30.0, self._ttl.total_seconds() / 2)
        try:
            while True:
                await asyncio.sleep(interval)
                if self._sandbox is None:
                    return
                await self._sandbox.renew(self._ttl)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception(
                "Failed to renew OpenSandbox sandbox %s",
                getattr(self._sandbox, "id", "unknown"),
            )

    async def stop(self, delete: bool) -> None:
        if self._renew_task is not None:
            self._renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._renew_task
            self._renew_task = None
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is None:
            return
        try:
            await sandbox.destroy()
            self.logger.info("Destroyed OpenSandbox sandbox %s", sandbox.id)
        except Exception:
            self.logger.exception(
                "Failed to destroy OpenSandbox sandbox %s; TTL cleanup remains active",
                sandbox.id,
            )

    def _require_sandbox(self) -> Sandbox:
        if self._sandbox is None:
            raise RuntimeError("OpenSandbox environment has not been started")
        return self._sandbox

    @staticmethod
    def _uid(user: str | int | None) -> int | None:
        if user is None:
            return None
        if user == "root":
            return 0
        if isinstance(user, int):
            if user < 0:
                raise ValueError("OpenSandbox command UID must be non-negative")
            return user
        if user.isdigit():
            return int(user)
        raise ValueError("OpenSandboxEnvironment supports root or numeric UIDs, " f"not named user {user!r}")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        sandbox = self._require_sandbox()
        effective_user = self._resolve_user(user)
        merged_env = forward_env_vars(self._merge_env(env) or {})
        merged_env = apply_api_key_rotation(merged_env, container_name=self.session_id, logger=self.logger)
        agent_hint = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
        if not self._barrier_passed and is_agent_command(command, agent_hint):
            self._barrier_passed = True
            await wait_at_barrier(container_name=self.session_id, logger=self.logger)
        opts = RunCommandOpts(
            working_directory=(cwd or self.task_env_config.workdir or self._workdir_from_dockerfile),
            timeout=timedelta(seconds=timeout_sec) if timeout_sec else None,
            uid=self._uid(effective_user),
            envs=merged_env or None,
        )
        execution = await sandbox.commands.run(command, opts=opts)
        stdout = "".join(message.text for message in execution.logs.stdout)
        stderr = "".join(message.text for message in execution.logs.stderr)
        return_code = execution.exit_code
        if return_code is None:
            return_code = 1 if execution.error is not None else 0
        return ExecResult(
            stdout=stdout or None,
            stderr=stderr or None,
            return_code=return_code,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        await self._require_sandbox().files.write_file(
            target_path,
            source.read_bytes(),
            mode=format(source.stat().st_mode & 0o777, "o"),
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")
        token = uuid.uuid4().hex
        remote_archive = f"/tmp/.tasktrace-capture-upload-{token}.tar.gz"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "upload.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for child in source.iterdir():
                    bundle.add(child, arcname=child.name)
            await self._require_sandbox().files.write_file(remote_archive, archive.read_bytes(), mode="600")
        command = (
            f"mkdir -p {shlex.quote(target_dir)} && "
            f"tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target_dir)}; "
            f"rc=$?; rm -f {shlex.quote(remote_archive)}; exit $rc"
        )
        result = await self.exec(command, cwd="/", user="root", timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to upload directory to {target_dir!r}: " f"{result.stderr or result.stdout or 'no output'}"
            )

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        destination = Path(target_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = await self._require_sandbox().files.read_bytes(source_path)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        destination = Path(target_dir)
        destination.mkdir(parents=True, exist_ok=True)
        remote_archive = f"/tmp/.tasktrace-capture-download-{uuid.uuid4().hex}.tar.gz"
        command = f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source_dir)} ."
        result = await self.exec(command, cwd="/", user="root", timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to archive OpenSandbox directory {source_dir!r}: "
                f"{result.stderr or result.stdout or 'no output'}"
            )
        try:
            content = await self._require_sandbox().files.read_bytes(remote_archive)
            with tempfile.TemporaryDirectory() as temporary:
                archive = Path(temporary) / "download.tar.gz"
                archive.write_bytes(content)
                with tarfile.open(archive, "r:gz") as bundle:
                    bundle.extractall(destination, filter="data")
        finally:
            with suppress(Exception):
                await self.exec(
                    f"rm -f {shlex.quote(remote_archive)}",
                    cwd="/",
                    user="root",
                    timeout_sec=30,
                )

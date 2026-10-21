# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TasktraceCaptureDockerEnv: thin subclass of Harbor's DockerEnvironment.

We inherit everything Harbor's DockerEnvironment does — start, stop,
upload/download, exec command assembly, mounts, compose lifecycle — and
override only the single public ``exec()`` to inject tasktrace-capture-owned
semantics:

1. ``_TASKTRACE_CAPTURE_FORWARD_ENVS`` — forward arbitrary --agent-env vars from
   the tasktrace-capture process into the container.
2. ``_TASKTRACE_CAPTURE_API_KEYS`` — rotate API keys across concurrent trials to
   spread load across rate-limit quotas.
3. ``_TASKTRACE_CAPTURE_BARRIER_COUNT`` — synchronize the first real agent command
   across N concurrent trials so "c=32" is actually 32-concurrent.

Harbor is pinned to a version that already removed the hardcoded ``-it``
flag from DockerEnvironment.exec(), so no command-assembly override is
needed.

Registered at import path ``tasktrace_capture.docker_env:TasktraceCaptureDockerEnv``
by tasktrace_capture.runtimes.runtime_import_path("docker").
"""

from __future__ import annotations

import asyncio
import os

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment

from tasktrace_capture.live_cpu_profile import (
    WINDOW_AGENT_EXECUTION,
    WINDOW_LIFECYCLE,
    DockerLiveCpuProfiler,
    live_cpu_enabled,
)
from tasktrace_capture.runtimes._shared import (
    DEFAULT_CLAUDE_CODE_VERSION,
    apply_api_key_rotation,
    ensure_claude_code_cached_on_host,
    forward_env_vars,
    is_agent_command,
    is_claude_code_install_command,
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


class TasktraceCaptureDockerEnv(DockerEnvironment):
    def __init__(self, *args, **kwargs):
        if owner := os.environ.get("_TASKTRACE_CAPTURE_DOCKER_OWNER"):
            kwargs["session_id"] = f"{owner}--{kwargs['session_id']}"
        super().__init__(*args, **kwargs)

    async def start(self, force_build: bool):
        """Start Docker with an optional pre-attached per-trial cgroup scope.

        This mirrors Harbor's current Docker ``start`` order so the lifecycle
        profile can begin after image preparation but before ``compose up``.
        """
        if not live_cpu_enabled() or "__verifier__" in self.session_id:
            return await super().start(force_build)
        if self._is_windows_container:
            raise RuntimeError("live CPU profiling supports Linux Docker trials only")

        self._mounts_compose_path = self._write_mounts_compose_file()
        self._resources_compose_path = self._write_resources_compose_file()
        self._use_prebuilt = not force_build and self.task_env_config.docker_image
        self._validate_daemon_mode()

        if not self._use_prebuilt:
            lock = self._image_build_locks.setdefault(self.environment_name, asyncio.Lock())
            async with lock:
                await self._run_docker_compose_command(["build"])

        image_to_check = self.task_env_config.docker_image if self._use_prebuilt else self._main_image_name
        if image_to_check:
            await self._validate_image_os(image_to_check)
        try:
            await self._run_docker_compose_command(["down", "--remove-orphans"])
        except RuntimeError:
            pass

        profiler = DockerLiveCpuProfiler(agent_dir=self.trial_paths.agent_dir, session_id=self.session_id)
        self._tasktrace_capture_live_profiler = profiler
        try:
            compose_override = await profiler.prepare_scope()
            self.extra_docker_compose_paths.append(compose_override)
            if WINDOW_LIFECYCLE in profiler.windows:
                await profiler.start_scheduler()
                profiler.begin_window(WINDOW_LIFECYCLE)
            await self._run_docker_compose_command(["up", "--detach", "--wait"])
            await self.ensure_dirs(self._mount_targets(writable_only=True))
        except Exception:
            await profiler.cleanup()
            raise

    async def stop(self, delete: bool):
        """Finalize any live profile before Harbor tears down the measured main container."""
        profiler = getattr(self, "_tasktrace_capture_live_profiler", None)
        profile_error: Exception | None = None
        if profiler is not None:
            try:
                profiler.finish_window(WINDOW_LIFECYCLE)
                await profiler.finish_scheduler()
            except Exception as exc:
                profile_error = exc
        try:
            await super().stop(delete)
        finally:
            if profiler is not None:
                await profiler.cleanup()
        if profile_error is not None:
            raise profile_error

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        container_name = self.session_id.lower().replace(".", "-")

        env = forward_env_vars(dict(env or {}))
        env = apply_api_key_rotation(env, container_name=container_name, logger=self.logger)

        env = with_image_bash_env(env)
        agent = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
        agent_command = is_agent_command(command, agent)
        if agent_command:
            await wait_at_barrier(container_name=container_name, logger=self.logger)
        if shell_capture_enabled():
            command, env = instrument_agent_launch(
                command,
                agent,
                env,
                self.trial_paths.agent_dir,
            )
            if command_starts_login_shell(command, agent):
                await self._prepare_login_shell_capture(env=env, user=user)

        # Short-circuit Harbor's claude-code install with our host-side cache.
        # At c>=8 on memory-constrained hosts, N parallel native-build installs
        # OOM. Cache builds the binary once, we `docker cp` it into each
        # container. On cache failure we fall through to Harbor's normal install.
        if is_claude_code_install_command(command):
            hit = await self._try_inject_cached_claude(user=user)
            if hit is not None:
                return hit

        profiler = getattr(self, "_tasktrace_capture_live_profiler", None)
        if profiler is not None and agent_command:
            await profiler.start_scheduler()
            profiler.begin_window(WINDOW_AGENT_EXECUTION)
        try:
            return await super().exec(
                command=command,
                cwd=cwd,
                env=env,
                timeout_sec=timeout_sec,
                user=user,
            )
        finally:
            if profiler is not None and agent_command:
                profiler.finish_window(WINDOW_AGENT_EXECUTION)

    async def _prepare_login_shell_capture(self, env: dict[str, str], user: str | int | None) -> None:
        """Enable the shared tracer before Terminus creates its login shell."""
        if getattr(self, "_tasktrace_capture_trace_login_ready", False):
            return
        prepare_shell_capture(self.trial_paths.agent_dir)
        result = await super().exec(
            command=install_login_shell_capture_command(),
            env=env,
            user=user,
        )
        if result.return_code == 0:
            self._tasktrace_capture_trace_login_ready = True
        elif self.logger:
            self.logger.warning("Unable to install Bash command timing hook")

    async def _try_inject_cached_claude(self, user):
        """If cache has a binary, copy it into container and verify.
        Returns ExecResult on success (signaling "skip Harbor's install"),
        or None on failure (caller falls through to Harbor's install).
        """
        version = os.environ.get(
            "_TASKTRACE_CAPTURE_CLAUDE_CODE_VERSION",
            DEFAULT_CLAUDE_CODE_VERSION,
        )
        cached = ensure_claude_code_cached_on_host(version=version, logger=self.logger)
        if not cached or not cached.exists():
            return None
        try:
            await super().exec("mkdir -p /root/.local/bin", user="root")
            await self._run_docker_compose_command(
                ["cp", str(cached), "main:/root/.local/bin/claude"],
            )
            check = await super().exec(
                'export PATH="$HOME/.local/bin:$PATH"; claude --version',
                user=user,
            )
            if check.return_code == 0:
                if self.logger:
                    self.logger.info(
                        "claude-code installed from host cache (%s): %s",
                        cached,
                        (check.stdout or "").strip(),
                    )
                return check
        except Exception as exc:
            if self.logger:
                self.logger.warning("claude-code cache injection failed: %s", exc)
        return None

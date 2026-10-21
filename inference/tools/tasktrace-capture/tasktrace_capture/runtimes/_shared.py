# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-agnostic host-side Python used by both EnrootEnvironment and
TasktraceCaptureDockerEnv.

This module holds the barrier/rotation/forwarding logic that was originally
inside enroot_env.py but applies to any Harbor environment plugin tasktrace-capture
manages. The functions here run in the harbor subprocess (same address
space as the plugin), NOT in the tasktrace-capture parent process.

Module-level barrier state is intentional: Harbor runs all trials of one
concurrency level in a single event loop, so an asyncio.Event here
coordinates every concurrent agent.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# -- Barrier state ------------------------------------------------------------

_barrier_event: asyncio.Event | None = None
_barrier_remaining: int = 0
_BARRIER_TIMEOUT_SEC = 1200  # 20 min max wait for all agents

_logger = logging.getLogger(__name__)


def _reset_barrier_for_tests() -> None:
    """Reset module-level barrier state. Tests only."""
    global _barrier_event, _barrier_remaining
    _barrier_event = None
    _barrier_remaining = 0


async def wait_at_barrier(container_name: str, logger: logging.Logger | None = None) -> None:
    """Block until all concurrent agents have finished setup.

    Reads ``_TASKTRACE_CAPTURE_BARRIER_COUNT`` from os.environ. If unset or <= 1,
    returns immediately. Otherwise, the first caller initializes a shared
    asyncio.Event; each caller decrements the counter; the last caller
    sets the event and releases everyone.

    ``container_name`` is used for logging only. ``logger`` defaults to
    this module's logger.
    """
    global _barrier_event, _barrier_remaining

    log = logger or _logger

    n_str = os.environ.get("_TASKTRACE_CAPTURE_BARRIER_COUNT", "")
    if not n_str:
        return
    n_expected = int(n_str)
    if n_expected <= 1:
        return

    # First caller initializes shared state.
    if _barrier_event is None or _barrier_remaining <= 0:
        _barrier_event = asyncio.Event()
        _barrier_remaining = n_expected

    _barrier_remaining -= 1
    ready = n_expected - _barrier_remaining
    log.info("Barrier: %s ready (%d/%d)", container_name, ready, n_expected)

    if _barrier_remaining <= 0:
        log.info("Barrier: all %d agents ready — releasing", n_expected)
        _barrier_event.set()
    else:
        try:
            await asyncio.wait_for(_barrier_event.wait(), timeout=_BARRIER_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            log.warning(
                "Barrier: timed out after %ds with %d/%d agents — proceeding",
                _BARRIER_TIMEOUT_SEC,
                ready,
                n_expected,
            )


# -- Env-var forwarding -------------------------------------------------------


def forward_env_vars(env: dict[str, str]) -> dict[str, str]:
    """Merge os.environ values for keys listed in ``_TASKTRACE_CAPTURE_FORWARD_ENVS`` into env.

    Caller-supplied env values win over the forwarded ones (matches the
    original behavior in enroot_env.py's exec).
    """
    forward_list = os.environ.get("_TASKTRACE_CAPTURE_FORWARD_ENVS", "")
    if not forward_list:
        return dict(env)
    forwarded = {k: os.environ[k] for k in forward_list.split(",") if k in os.environ}
    if not forwarded:
        return dict(env)
    return {**forwarded, **env}


# -- API-key rotation ---------------------------------------------------------


def apply_api_key_rotation(
    env: dict[str, str],
    container_name: str,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """When ``_TASKTRACE_CAPTURE_API_KEYS`` has 2+ keys, set env["ANTHROPIC_API_KEY"] to
    a deterministic pick based on ``hash(container_name) % len(keys)``.

    Single-key case is a no-op; runner.py sets ANTHROPIC_API_KEY directly
    when only one key is provided.
    """
    api_keys_str = os.environ.get("_TASKTRACE_CAPTURE_API_KEYS", "")
    if not api_keys_str:
        return dict(env)
    keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    if len(keys) <= 1:
        return dict(env)
    idx = hash(container_name) % len(keys)
    out = dict(env)
    out["ANTHROPIC_API_KEY"] = keys[idx]
    if logger:
        logger.info(
            "API key rotation: %s → key %d/%d (...%s)",
            container_name,
            idx + 1,
            len(keys),
            keys[idx][-6:],
        )
    return out


# -- Claude Code binary cache -------------------------------------------------
#
# Runtime-agnostic host-side download + cache of the claude-code binary so
# N concurrent trials don't each run `curl | bash` in their own container.
# At c>=8 on memory-constrained hosts the native-build install OOMs; at c>=64
# it also saturates bandwidth. By installing ONCE on the host and copying the
# binary into each container, we flatten that curve.
#
# Enroot uses this via enroot exec cp; docker uses `docker compose cp` —
# the shared piece is just "make sure the binary is on disk at a known path".

CLAUDE_CODE_CACHE_DIR = Path(tempfile.gettempdir()) / "tasktrace_capture_claude_code_cache"
CLAUDE_CODE_CACHE_LOCK = CLAUDE_CODE_CACHE_DIR.with_suffix(".lock")
DEFAULT_CLAUDE_CODE_VERSION = "2.1.121"


def cached_claude_code_version() -> str | None:
    """Return the version string of the cached claude-code binary, or None.

    None either means no cache exists (fresh install — fine) or the cache is
    unreadable. Used by ``runtime_preflight_checks`` to detect stale caches
    that pin to an older ``DEFAULT_CLAUDE_CODE_VERSION`` and would otherwise
    silently propagate the old binary into every trial container.
    """
    binary = CLAUDE_CODE_CACHE_DIR / "claude"
    if not binary.is_file():
        return None
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            timeout=5,
            text=True,
        )
        out = (result.stdout or "").strip()
        # `claude --version` prints e.g. "2.1.121 (Claude Code)"; take token 0.
        if out:
            return out.split()[0]
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None


def ensure_claude_code_cached_on_host(
    version: str = DEFAULT_CLAUDE_CODE_VERSION,
    logger: logging.Logger | None = None,
) -> Path | None:
    """Download the claude-code binary once to a shared host cache.

    Uses an fcntl exclusive lock so only the first caller performs the
    install; the rest block briefly, find the cached binary, and return.
    Returns the cached binary path, or ``None`` on failure — caller MUST
    fall back to the runtime's normal install path on None.

    ``version`` is forwarded to Claude's install.sh as the version argument.
    """
    binary = CLAUDE_CODE_CACHE_DIR / "claude"

    # Ensure lock file parent exists (lock file itself is created by open()).
    try:
        CLAUDE_CODE_CACHE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    lock_fd = open(CLAUDE_CODE_CACHE_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        # Another caller may have installed while we waited for the lock.
        if binary.exists() and os.access(str(binary), os.X_OK):
            if logger:
                logger.info("claude-code binary already cached at %s", binary)
            return binary

        CLAUDE_CODE_CACHE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        if logger:
            logger.info("Pre-installing claude-code@%s to %s ...", version, CLAUDE_CODE_CACHE_DIR)

        # Set HOME so the installer writes everything into our cache dir
        # instead of polluting the real host $HOME.
        install_env = os.environ.copy()
        install_env["HOME"] = str(CLAUDE_CODE_CACHE_DIR)

        result = subprocess.run(
            ["bash", "-c", f"curl -fsSL https://claude.ai/install.sh | bash -s -- {version}"],
            env=install_env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            if logger:
                tail = (result.stderr or "")[-500:]
                logger.warning("claude-code install failed (rc=%d): %s", result.returncode, tail)
            return None

        # install.sh creates $HOME/.local/bin/claude (possibly as symlink
        # to $HOME/.local/share/claude/versions/...). Resolve + copy the
        # real file to CLAUDE_CODE_CACHE_DIR/claude so we have a single
        # self-contained file to distribute.
        installed_bin = CLAUDE_CODE_CACHE_DIR / ".local" / "bin" / "claude"
        if not installed_bin.exists():
            if logger:
                logger.warning(
                    "claude-code install succeeded but binary missing at %s",
                    installed_bin,
                )
            return None

        shutil.copy2(str(installed_bin.resolve()), str(binary))
        os.chmod(str(binary), 0o755)
        if logger:
            logger.info("claude-code binary cached at %s", binary)
        return binary
    except Exception as exc:
        if logger:
            logger.warning("Failed to pre-install claude-code: %s", exc)
        return None
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            lock_fd.close()


def is_claude_code_install_command(command: str) -> bool:
    """True if the given exec command looks like Harbor's claude-code install.

    Matches Harbor's install shape (`if apk then npm else curl|bash` + version
    check). If Harbor ever changes the install script shape this stops matching,
    which means the cache no longer fires and we fall back to Harbor's install
    — graceful degradation.
    """
    return (
        "install" in command
        and "claude --version" in command
        and ("claude.ai/install.sh" in command or "@anthropic-ai/claude-code" in command)
    )


# -- Agent-command detection --------------------------------------------------

# claude-code's real invocation includes --permission-mode.
# terminus's real invocation contains the literal "terminus".
# mini-swe-agent's real invocation contains ``mini-swe-agent --yolo``.
# Setup calls (mkdir, curl install, claude --version) do NOT match either.
_CLAUDE_CODE_SIGNATURE = "--permission-mode"
_TERMINUS_SIGNATURE = "terminus"
_MINI_SWE_SIGNATURE = "mini-swe-agent --yolo"


def is_agent_command(command: str, agent: str) -> bool:
    """True if ``command`` is the agent's real invocation (not a setup step).

    Uses signatures specific to each agent: claude-code -> ``--permission-mode``,
    terminus -> literal ``terminus``, mini-swe-agent -> ``mini-swe-agent --yolo``.
    The ``agent`` parameter is an optional hint; detection falls back to
    scanning the signatures in the command.
    """
    if _CLAUDE_CODE_SIGNATURE in command:
        return True
    if _MINI_SWE_SIGNATURE in command:
        return True
    # terminus signature matches `python -m terminus` etc. Require agent hint
    # OR a command that actually starts terminus to avoid false positives on
    # commands that merely mention "terminus" in a path.
    if _TERMINUS_SIGNATURE in command and (
        agent.startswith("terminus")
        or command.startswith(_TERMINUS_SIGNATURE)
        or " terminus" in command
        or "-m terminus" in command
    ):
        return True
    if agent == "mini-swe-agent" and _MINI_SWE_SIGNATURE in command:
        return True
    return False

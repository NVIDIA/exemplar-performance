# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML experiment config loader.

Reads a tasktrace-capture experiment YAML, expands ``${VAR}`` / ``$VAR`` placeholders
against the process environment, validates keys against the tasktrace-capture CLI
dest-name whitelist, and normalises ``agent_env`` and ``concurrency`` to the
shapes the existing argparse-driven CLI expects.

Override semantics (CLI > yaml) live in ``tasktrace_capture/cli.py``: this module is a
pure transform from a YAML file to a ``dict[dest, value]``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

# argparse dest names that may appear as yaml keys. Kept in sync with
# ``tasktrace_capture/cli.py``'s argparse setup.
ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "dataset",
        "n_tasks",
        "include_tasks",
        "task",
        # Ad-hoc-prompt flags. Without these, you can't drive
        # SWE-bench-c-style workflows through yaml mode.
        "prompt",
        "prompt_file",
        "prompt_image",
        "prompt_name",
        "model",
        "api_base",
        "n_concurrent",
        "trials",
        "sweep",
        "concurrency",
        "agent",
        "agent_kwarg",
        "agent_env",
        "api_key",
        "agent_timeout",
        "output_dir",
        "repeat",
        "dry_run",
        "check",
        "skip_preflight",
        "allow_parallel_docker",
        "allow_private_api",
        "quiet",
        "runtime",
        "enroot_network_policy",
        "swebench_image_prefix",
        "swebench_sif_template",
        "override_cpus",
        "instrument_shell_commands",
        "instrument_shell_perfetto_depth",
        "source_image_bashrc",
        "trajectory_timings",
        "profile_live_cpu",
        "profile_windows",
        "profile_scheduler",
        "profile_scheduler_collector_image",
        "profile_cpuset_cpus",
        "no_start_barrier",
        "no_node_telemetry",
        "node_telemetry_interval",
        "claude_code_version",
    }
)

_VAR_RE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand(value: str, source: str) -> str:
    def _repl(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if name not in os.environ:
            raise KeyError(f"{source}: environment variable ${{{name}}} is referenced " f"but not set")
        return os.environ[name]

    return _VAR_RE.sub(_repl, value)


def _walk(node: Any, source: str) -> Any:
    if isinstance(node, str):
        return _expand(node, source)
    if isinstance(node, list):
        return [_walk(x, source) for x in node]
    if isinstance(node, dict):
        return {k: _walk(v, source) for k, v in node.items()}
    return node


def load_yaml_config(
    path: str | Path,
    allowed_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Load a tasktrace-capture YAML config into a dict of argparse-dest → value.

    - Expands ``${VAR}`` / ``$VAR`` in string values, raising ``KeyError``
      when a referenced env var is unset.
    - Rejects keys not in ``allowed_keys`` (defaults to ``ALLOWED_KEYS``).
    - Coerces ``agent_env`` mapping to ``["KEY=VALUE", ...]``.
    - Coerces ``concurrency`` int / list / string into the comma-separated
      string shape the CLI parser already accepts.
    """
    keys = ALLOWED_KEYS if allowed_keys is None else allowed_keys
    p = Path(path)
    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top-level YAML must be a mapping, got {type(raw).__name__}")

    raw = _walk(raw, str(p))

    unknown = sorted(set(raw) - keys)
    if unknown:
        raise ValueError(
            f"{p}: unknown config key(s): {', '.join(unknown)}.\n" f"  Allowed keys: {', '.join(sorted(keys))}"
        )

    if "allow_parallel_docker" in raw and not isinstance(raw["allow_parallel_docker"], bool):
        raise ValueError(f"{p}: 'allow_parallel_docker' must be a boolean")

    if "no_node_telemetry" in raw and not isinstance(raw["no_node_telemetry"], bool):
        raise ValueError(f"{p}: 'no_node_telemetry' must be a boolean")

    if "node_telemetry_interval" in raw:
        iv = raw["node_telemetry_interval"]
        if isinstance(iv, bool) or not isinstance(iv, (int, float)) or iv < 0.5:
            raise ValueError(f"{p}: 'node_telemetry_interval' must be a number >= 0.5")

    if "agent_env" in raw:
        env = raw["agent_env"]
        if isinstance(env, dict):
            raw["agent_env"] = [f"{k}={v}" for k, v in env.items()]
        elif isinstance(env, list):
            pass
        else:
            raise ValueError(f"{p}: 'agent_env' must be a mapping or list, " f"got {type(env).__name__}")

    if "include_tasks" in raw:
        include_tasks = raw["include_tasks"]
        if (
            not isinstance(include_tasks, list)
            or not include_tasks
            or any(not isinstance(task, str) or not task.strip() for task in include_tasks)
        ):
            raise ValueError(f"{p}: 'include_tasks' must be a nonempty list of task names")
        raw["include_tasks"] = list(dict.fromkeys(include_tasks))

    if "concurrency" in raw:
        c = raw["concurrency"]
        if isinstance(c, bool):
            raise ValueError(f"{p}: 'concurrency' must be int / list / string")
        if isinstance(c, int):
            raw["concurrency"] = str(c)
        elif isinstance(c, list):
            raw["concurrency"] = ",".join(str(x) for x in c)
        elif isinstance(c, str):
            pass
        else:
            raise ValueError(
                f"{p}: 'concurrency' must be int, list[int], or 'a,b,c' string; " f"got {type(c).__name__}"
            )

    return raw

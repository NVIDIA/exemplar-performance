# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Parsing and launcher contracts for explicit environment and workload arguments."""

from __future__ import annotations

import re
import shlex
from typing import Iterable, Mapping

LLMB_CONTAINER_ENV = 'LLMB_CONTAINER_ENV'
NEMO_ENV_OVERRIDE_VAR = 'CONFIG_OVERRIDES'
MBRIDGE_EXTRA_ARGS_VAR = 'LLMB_MBRIDGE_EXTRA_ARGS'

_ENV_KEY_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def validate_env_key(key: object, *, source: str = 'env') -> str:
    """Validate and return a shell-style environment variable name."""
    if not isinstance(key, str):
        raise ValueError(f"{source} variable name '{key}' is invalid. Use shell-style environment variable names only.")
    key = key.strip()
    if not key:
        raise ValueError(f"{source} must include a non-empty variable name.")
    if not _ENV_KEY_RE.match(key):
        raise ValueError(f"{source} variable name '{key}' is invalid. Use shell-style environment variable names only.")
    return key


def validate_shell_safe_env_value(key: str, value: str) -> None:
    """Reject env values that would be corrupted by downstream shell quoting.

    Shell-special characters get mangled when the nemo launcher wraps the job
    command in `bash -c '...'` (Megatron-Bridge setup_experiment.py forwards
    `sys.argv` wholesale into a single-quoted wrapper, so any inner single
    quotes from shlex-quoted values collide with the outer ones and split the
    command into bad positional args). Reject at the CLI/YAML boundary until
    that upstream fix lands.
    """
    if value and shlex.quote(value) != value:
        raise ValueError(
            f"Env value for '{key}' contains shell-special characters, which are not "
            f"supported. Use only characters matching [A-Za-z0-9_@%+=:,./-] until the "
            f"upstream Megatron-Bridge launcher fix lands."
        )


def parse_cli_env_args(raw_env_args: Iterable[str] | None) -> dict[str, str]:
    """Parse repeatable `--env KEY=value` CLI values into an insertion-ordered dict."""
    parsed: dict[str, str] = {}

    for raw_arg in raw_env_args or ():
        if '=' not in raw_arg:
            raise ValueError("`--env` must be in `KEY=value` form.")

        key, value = raw_arg.split('=', 1)
        key = validate_env_key(key, source='`--env`')
        if key in parsed:
            raise ValueError(f"Duplicate environment variable '{key}' was specified more than once.")
        validate_shell_safe_env_value(key, value)

        parsed[key] = value

    return parsed


def parse_cli_mbridge_args(raw_args: Iterable[str] | None) -> tuple[str, ...]:
    """Validate raw Megatron-Bridge argv tokens while preserving their order."""
    parsed: list[str] = []

    for raw_arg in raw_args or ():
        if not raw_arg:
            raise ValueError("`--mbridge-arg` cannot be empty.")
        if any(char.isspace() for char in raw_arg):
            raise ValueError(
                "`--mbridge-arg` accepts one argv token at a time and cannot contain whitespace. "
                "Repeat the option for arguments that require multiple tokens."
            )
        parsed.append(raw_arg)

    return tuple(parsed)


def build_nemo_env_override_flags(overrides: Mapping[str, str]) -> str:
    """Render explicit env overrides as repeatable `-E KEY=value` flags.

    Each `KEY=value` token is shell-quoted so values containing whitespace or
    glob characters survive word-splitting when launch scripts expand the
    resulting variable unquoted (e.g. `launcher ${CONFIG_OVERRIDES}`). Wrapping
    the expansion in double quotes downstream would defeat this.
    """
    return ' '.join(f"-E {shlex.quote(f'{key}={value}')}" for key, value in overrides.items())


def apply_sbatch_explicit_env_contract(env: dict[str, str], overrides: Mapping[str, str]) -> None:
    """Expose explicit env keys to sbatch-style launch scripts."""
    if not overrides:
        return

    existing_keys = [key for key in str(env.get(LLMB_CONTAINER_ENV, '')).split(',') if key]
    env[LLMB_CONTAINER_ENV] = ','.join(dict.fromkeys([*existing_keys, *overrides]))


def apply_nemo_explicit_env_contract(env: dict[str, str], overrides: Mapping[str, str]) -> None:
    """Expose explicit env overrides to Nemo launch scripts via repeatable `-E` flags."""
    if not overrides:
        return

    override_flags = build_nemo_env_override_flags(overrides)
    existing = str(env.get(NEMO_ENV_OVERRIDE_VAR, '')).strip()
    env[NEMO_ENV_OVERRIDE_VAR] = f"{existing} {override_flags}".strip() if existing else override_flags


def build_nemo_workload_args(args: Iterable[str]) -> str:
    """Render raw workload argv tokens for Nemo-style launch scripts."""
    return ' '.join(args)


def apply_nemo_workload_args_contract(env: dict[str, str], args: Iterable[str]) -> None:
    """Expose extra workload argv tokens to Nemo-style launch scripts via CONFIG_OVERRIDES."""
    rendered_args = build_nemo_workload_args(args)
    if not rendered_args:
        return

    existing = str(env.get(NEMO_ENV_OVERRIDE_VAR, '')).strip()
    env[NEMO_ENV_OVERRIDE_VAR] = f"{existing} {rendered_args}".strip() if existing else rendered_args


def build_mbridge_extra_args(
    args: Iterable[str],
    *,
    compatibility_args: Iterable[str] = (),
    env_overrides: Mapping[str, str] | None = None,
) -> str:
    """Render legacy args, explicit env, then raw args for MBridge launch scripts."""
    segments = (
        build_nemo_workload_args(compatibility_args),
        build_nemo_env_override_flags(env_overrides or {}),
        ' '.join(args),
    )
    return ' '.join(segment for segment in segments if segment)


def apply_mbridge_extra_args_contract(
    env: dict[str, str],
    args: Iterable[str],
    *,
    compatibility_args: Iterable[str] = (),
    env_overrides: Mapping[str, str] | None = None,
) -> None:
    """Expose complete MBridge arguments for launch scripts to append last."""
    rendered_args = build_mbridge_extra_args(
        args,
        compatibility_args=compatibility_args,
        env_overrides=env_overrides,
    )
    if not rendered_args:
        return

    existing = str(env.get(MBRIDGE_EXTRA_ARGS_VAR, '')).strip()
    env[MBRIDGE_EXTRA_ARGS_VAR] = f"{existing} {rendered_args}".strip() if existing else rendered_args

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parsing and serialization helpers for Slurm submit args."""

from __future__ import annotations

import os
from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

ADDITIONAL_SLURM_PARAMS_KEY = 'ADDITIONAL_SLURM_PARAMS'
FIRST_CLASS_SLURM_KEYS = ('nodelist', 'exclude', 'reservation', 'segment', 'nice')


class SlurmParam(BaseModel):
    """A single Slurm parameter rendered as key=value or a bare flag."""

    model_config = ConfigDict(frozen=True)

    key: str
    value: str | None = None

    def render(self) -> str:
        return self.key if self.value is None else f"{self.key}={self.value}"

    def as_sbatch_arg(self) -> str:
        return f"--{self.render()}"


class SlurmArgs(BaseModel):
    """Canonical Slurm submit arguments shared across launchers."""

    model_config = ConfigDict(frozen=True)

    named_params: dict[str, str] = {}
    passthrough_params: tuple[SlurmParam, ...] = ()

    def is_empty(self) -> bool:
        return not self.named_params and not self.passthrough_params

    def get_named_param(self, key: str) -> str | None:
        return self.named_params.get(key)

    def iter_params(self) -> Iterable[SlurmParam]:
        for key in FIRST_CLASS_SLURM_KEYS:
            if key in self.named_params:
                yield SlurmParam(key=key, value=self.named_params[key])
        yield from self.passthrough_params

    def to_additional_slurm_params(self) -> str:
        return ';'.join(param.render() for param in self.iter_params())

    def to_sbatch_args(self) -> list[str]:
        return [param.as_sbatch_arg() for param in self.iter_params()]


def build_cli_slurm_args(
    *,
    nodelist: str | None = None,
    exclude: str | None = None,
    reservation: str | None = None,
    segment: int | None = None,
    nice: int | None = None,
    slurm_args: Iterable[str] | None = None,
) -> SlurmArgs | None:
    """Build canonical Slurm args from first-class CLI flags."""
    named_params: dict[str, str] = {}
    if nodelist is not None:
        named_params['nodelist'] = nodelist
    if exclude is not None:
        named_params['exclude'] = exclude
    if reservation is not None:
        named_params['reservation'] = reservation
    if segment is not None:
        named_params['segment'] = str(segment)
    if nice is not None:
        named_params['nice'] = str(nice)

    seen_keys = set(named_params)
    passthrough_params: list[SlurmParam] = []

    for raw_arg in slurm_args or ():
        raw_arg = raw_arg.strip()
        if not raw_arg:
            raise ValueError("`--slurm-arg` cannot be empty.")
        if raw_arg.startswith('--'):
            raise ValueError(
                "`--slurm-arg` values should not include a leading '--'. "
                "Use `constraint=gpu` or `exclusive` instead."
            )

        if '=' in raw_arg:
            key, value = raw_arg.split('=', 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise ValueError("`--slurm-arg` assignments must be in `key=value` form with non-empty key and value.")
            param = SlurmParam(key=key, value=value)
        else:
            key = raw_arg
            param = SlurmParam(key=key, value=None)

        if key in FIRST_CLASS_SLURM_KEYS:
            raise ValueError(f"`{key}` has a dedicated flag. " f"Use `--{key}` instead of `--slurm-arg {raw_arg}`.")

        if key in seen_keys:
            raise ValueError(f"Duplicate Slurm parameter '{key}' was specified more than once.")

        seen_keys.add(key)
        passthrough_params.append(param)

    if not named_params and not passthrough_params:
        return None

    return SlurmArgs(named_params=named_params, passthrough_params=tuple(passthrough_params))


def validate_no_additional_slurm_params_conflict(
    *,
    cli_args: SlurmArgs | None,
    cluster_environment: Mapping[str, object] | None = None,
    workload_environment: Mapping[str, object] | None = None,
    task_environment: Mapping[str, object] | None = None,
) -> None:
    """Ensure first-class CLI Slurm args do not mix with direct env-var injection."""
    if cli_args is None or cli_args.is_empty():
        return

    sources: list[str] = []
    if _has_non_empty_value(os.environ, ADDITIONAL_SLURM_PARAMS_KEY):
        sources.append('process environment')
    if _has_non_empty_value(cluster_environment, ADDITIONAL_SLURM_PARAMS_KEY):
        sources.append('cluster config environment')
    if _has_non_empty_value(workload_environment, ADDITIONAL_SLURM_PARAMS_KEY):
        sources.append('workload config environment')
    if _has_non_empty_value(task_environment, ADDITIONAL_SLURM_PARAMS_KEY):
        sources.append('task environment overrides')

    if sources:
        source_list = ', '.join(sources)
        raise ValueError(
            f"Cannot combine first-class Slurm CLI flags with {ADDITIONAL_SLURM_PARAMS_KEY} from {source_list}."
        )


def _has_non_empty_value(env: Mapping[str, object] | None, key: str) -> bool:
    if not env:
        return False
    value = env.get(key)
    if value is None:
        return False
    return str(value).strip() != ''

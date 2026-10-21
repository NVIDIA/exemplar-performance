# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict schemas for content-bound offline-replay runs.

Versions 1 and 2 describe one shared CPU pool.  Version 3 describes one ordered
sequence of concurrently submitted waves separated by completion barriers.  A
compact round-robin authoring form expands to the same explicit task-to-CPU
launch plan used by the executor.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..adapters.atif_profile import CONTROLLER_RUNTIME_MODES
from ..adapters.atif_replay import PACING_MODES
from ..measurement.container import (
    CONTAINER_RUNTIMES,
    PROFILE_SETS,
    ProfileError,
    parse_cpuset_cpu_ids,
)
from ..measurement.network_policy import NetworkPolicy, NetworkPolicyError

SCHEMA_VERSION_V1 = "tasktrace-run.v1"
SCHEMA_VERSION_V2 = "tasktrace-run.v2"
SCHEMA_VERSION_V3 = "tasktrace-run.v3"
# Backward-compatible name for callers that imported the original v1 constant.
SCHEMA_VERSION = SCHEMA_VERSION_V1

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROUND_ROBIN_SINGLE_CPU = "round-robin-single-cpu"


class RunSpecError(ValueError):
    """A run document is malformed or cannot be content-bound."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class RunTaskSpec:
    task_id: str
    atif: Path
    source_sha256: str
    image: str
    expected_image_id: str | None
    action_timeout_sec: float | None


@dataclass(frozen=True)
class RunSpec:
    schema_version: str
    manifest_path: Path
    manifest_sha256: str
    output: Path
    cpus: str
    cpu_ids: tuple[int, ...]
    repetitions: int
    tasks: tuple[RunTaskSpec, ...]
    concurrency: int = 1


MetadataScalar = str | int | float | bool


@dataclass(frozen=True)
class RunDefaultsSpec:
    pacing: str
    network_policy: str
    allowed_hosts: tuple[str, ...]
    profile_sets: tuple[str, ...]
    controller_runtime: str
    controller_cpus: str
    controller_cpu_ids: tuple[int, ...]
    action_timeout_sec: float
    worker_cpus: str | None = None
    worker_cpu_ids: tuple[int, ...] = ()
    placement: str | None = None
    workdir: str | None = None
    runtime: str = "docker"


@dataclass(frozen=True)
class RunLaunchSpec:
    task_id: str
    cpus: str
    cpu_ids: tuple[int, ...]


@dataclass(frozen=True)
class RunWaveSpec:
    wave_id: str
    scored: bool
    metadata: tuple[tuple[str, MetadataScalar], ...]
    launches: tuple[RunLaunchSpec, ...]


@dataclass(frozen=True)
class OrderedRunSpec:
    schema_version: str
    manifest_path: Path
    manifest_sha256: str
    output: Path
    input_root: Path
    defaults: RunDefaultsSpec
    tasks: tuple[RunTaskSpec, ...]
    waves: tuple[RunWaveSpec, ...]


def _object(
    value: Any,
    *,
    field: str,
    required: frozenset[str] = frozenset(),
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunSpecError(f"{field} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise RunSpecError(f"{field} keys must be strings")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise RunSpecError(f"{field} is missing: {', '.join(missing)}")
    if unknown:
        raise RunSpecError(f"{field} has unknown fields: {', '.join(unknown)}")
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RunSpecError(f"{field} must be a non-empty trimmed string")
    return value


def _identifier(value: Any, *, field: str) -> str:
    identifier = _text(value, field=field)
    if _SAFE_ID.fullmatch(identifier) is None or identifier in {".", ".."}:
        raise RunSpecError(f"{field} must be a safe 1-128 character identifier")
    return identifier


def _path(value: Any, *, field: str, base: Path) -> Path:
    path = Path(_text(value, field=field))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _file_sha256(path: Path, *, field: str) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RunSpecError(f"unable to read {field} {path}: {exc}") from exc


def _canonical_cpuset(
    value: Any,
    *,
    field: str = "cpus",
) -> tuple[str, tuple[int, ...]]:
    raw = _text(value, field=field)
    try:
        cpu_ids = parse_cpuset_cpu_ids(raw)
    except ProfileError as exc:
        raise RunSpecError(f"invalid {field}: {exc}") from exc
    ranges: list[tuple[int, int]] = []
    start = previous = cpu_ids[0]
    for cpu_id in cpu_ids[1:]:
        if cpu_id == previous + 1:
            previous = cpu_id
            continue
        ranges.append((start, previous))
        start = previous = cpu_id
    ranges.append((start, previous))
    canonical = ",".join(str(start) if start == end else f"{start}-{end}" for start, end in ranges)
    return canonical, cpu_ids


def _positive_timeout(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunSpecError(f"{field} must be a number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise RunSpecError(f"{field} must be a finite positive number")
    return timeout


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RunSpecError(f"{field} must be a positive integer")
    return value


def _load_tasks(value: Any, *, base: Path, image_base: Path | None = None) -> tuple[RunTaskSpec, ...]:
    if not isinstance(value, list) or not value:
        raise RunSpecError("tasks must be a non-empty list")
    tasks: list[RunTaskSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        field = f"tasks[{index}]"
        item = _object(
            raw,
            field=field,
            required=frozenset({"id", "atif", "image"}),
            optional=frozenset(
                {
                    "expected_source_sha256",
                    "expected_image_id",
                    "action_timeout_sec",
                }
            ),
        )
        task_id = _identifier(item["id"], field=f"{field}.id")
        if task_id in seen:
            raise RunSpecError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        atif = _path(item["atif"], field=f"{field}.atif", base=base)
        if not atif.is_file():
            raise RunSpecError(f"{field}.atif is not a file: {atif}")
        source_sha256 = _file_sha256(atif, field=f"{field}.atif")
        expected_source = item.get("expected_source_sha256")
        if expected_source is not None:
            expected_source = _text(expected_source, field=f"{field}.expected_source_sha256").lower()
            if _SHA256.fullmatch(expected_source) is None:
                raise RunSpecError(f"{field}.expected_source_sha256 must be a SHA-256 hex digest")
            if expected_source != source_sha256:
                raise RunSpecError(
                    f"{field}.atif SHA-256 mismatch: expected {expected_source}, " f"got {source_sha256}"
                )
        expected_image_id = item.get("expected_image_id")
        if expected_image_id is not None:
            expected_image_id = _text(expected_image_id, field=f"{field}.expected_image_id")
        image = _text(item["image"], field=f"{field}.image")
        if image_base is not None:
            image = str(_path(image, field=f"{field}.image", base=image_base))
        raw_timeout = item.get("action_timeout_sec")
        tasks.append(
            RunTaskSpec(
                task_id=task_id,
                atif=atif,
                source_sha256=source_sha256,
                image=image,
                expected_image_id=expected_image_id,
                action_timeout_sec=(
                    None if raw_timeout is None else _positive_timeout(raw_timeout, field=f"{field}.action_timeout_sec")
                ),
            )
        )
    return tuple(tasks)


def _load_defaults(value: Any) -> RunDefaultsSpec:
    field = "defaults"
    item = _object(
        value,
        field=field,
        required=frozenset(
            {
                "pacing",
                "network_policy",
                "profile_sets",
                "controller_runtime",
                "controller_cpus",
                "action_timeout_sec",
            }
        ),
        optional=frozenset(
            {
                "allowed_hosts",
                "runtime",
                "worker_cpus",
                "placement",
                "workdir",
            }
        ),
    )

    pacing = _text(item["pacing"], field=f"{field}.pacing")
    if pacing not in PACING_MODES:
        raise RunSpecError(f"{field}.pacing must be one of: {', '.join(PACING_MODES)}")

    network_mode = _text(item["network_policy"], field=f"{field}.network_policy")
    raw_allowed_hosts = item.get("allowed_hosts", [])
    if not isinstance(raw_allowed_hosts, list):
        raise RunSpecError(f"{field}.allowed_hosts must be a list")
    allowed_hosts = tuple(
        _text(host, field=f"{field}.allowed_hosts[{index}]") for index, host in enumerate(raw_allowed_hosts)
    )
    try:
        network_policy = NetworkPolicy(
            mode=network_mode,
            allowed_hosts=allowed_hosts,
        )
    except NetworkPolicyError as exc:
        raise RunSpecError(f"invalid {field}.network_policy: {exc}") from exc

    runtime = _text(item.get("runtime", "docker"), field=f"{field}.runtime")
    if runtime not in CONTAINER_RUNTIMES:
        raise RunSpecError(f"{field}.runtime must be one of: " + ", ".join(CONTAINER_RUNTIMES))
    if runtime == "enroot" and network_policy.mode.value not in {
        "public",
        "no-network",
    }:
        raise RunSpecError(f"{field}.network_policy must be public or no-network for enroot runtime")

    raw_profile_sets = item["profile_sets"]
    if not isinstance(raw_profile_sets, list):
        raise RunSpecError(f"{field}.profile_sets must be a list")
    if runtime == "docker" and not raw_profile_sets:
        raise RunSpecError(f"{field}.profile_sets must be a non-empty list")
    profile_sets = tuple(
        _text(profile_set, field=f"{field}.profile_sets[{index}]") for index, profile_set in enumerate(raw_profile_sets)
    )
    if len(set(profile_sets)) != len(profile_sets):
        raise RunSpecError(f"{field}.profile_sets must not contain duplicates")
    unknown_profile_sets = sorted(set(profile_sets) - set(PROFILE_SETS))
    if unknown_profile_sets:
        raise RunSpecError(f"{field}.profile_sets contains unsupported values: " + ", ".join(unknown_profile_sets))
    if runtime == "enroot" and profile_sets:
        raise RunSpecError(f"{field}.profile_sets must be empty for enroot runtime")
    if runtime == "docker" and "cpu-average" not in profile_sets:
        raise RunSpecError(f"{field}.profile_sets must include cpu-average")

    controller_runtime = _text(item["controller_runtime"], field=f"{field}.controller_runtime")
    if controller_runtime not in CONTROLLER_RUNTIME_MODES:
        raise RunSpecError(f"{field}.controller_runtime must be one of: " + ", ".join(CONTROLLER_RUNTIME_MODES))
    controller_cpus, controller_cpu_ids = _canonical_cpuset(item["controller_cpus"], field=f"{field}.controller_cpus")
    has_worker_cpus = "worker_cpus" in item
    has_placement = "placement" in item
    if has_worker_cpus != has_placement:
        raise RunSpecError(f"{field}.worker_cpus and {field}.placement must be specified together")
    worker_cpus: str | None = None
    worker_cpu_ids: tuple[int, ...] = ()
    placement: str | None = None
    if has_worker_cpus:
        worker_cpus, worker_cpu_ids = _canonical_cpuset(item["worker_cpus"], field=f"{field}.worker_cpus")
        placement = _text(item["placement"], field=f"{field}.placement")
        if placement != _ROUND_ROBIN_SINGLE_CPU:
            raise RunSpecError(f"{field}.placement must be {_ROUND_ROBIN_SINGLE_CPU!r}")
        overlap = sorted(set(controller_cpu_ids).intersection(worker_cpu_ids))
        if overlap:
            raise RunSpecError(
                f"{field}.worker_cpus overlaps {field}.controller_cpus: " + ",".join(str(cpu_id) for cpu_id in overlap)
            )
    action_timeout_sec = _positive_timeout(item["action_timeout_sec"], field=f"{field}.action_timeout_sec")
    workdir: str | None = None
    if "workdir" in item:
        workdir = _text(item["workdir"], field=f"{field}.workdir")
        if not workdir.startswith("/"):
            raise RunSpecError(f"{field}.workdir must be an absolute in-container path")

    return RunDefaultsSpec(
        pacing=pacing,
        network_policy=network_policy.mode.value,
        allowed_hosts=network_policy.allowed_hosts,
        runtime=runtime,
        profile_sets=profile_sets,
        controller_runtime=controller_runtime,
        controller_cpus=controller_cpus,
        controller_cpu_ids=controller_cpu_ids,
        action_timeout_sec=action_timeout_sec,
        worker_cpus=worker_cpus,
        worker_cpu_ids=worker_cpu_ids,
        placement=placement,
        workdir=workdir,
    )


def _metadata_scalar(value: Any, *, field: str) -> MetadataScalar:
    if isinstance(value, str):
        return _text(value, field=field)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise RunSpecError(f"{field} must be a string, boolean, integer, or finite number")


def _load_waves(
    value: Any,
    *,
    tasks: tuple[RunTaskSpec, ...],
    defaults: RunDefaultsSpec,
) -> tuple[RunWaveSpec, ...]:
    """Load explicit waves or deterministically expand compact placement.

    Compact allocation ``N`` selects the first N worker IDs and maps task
    ``i`` to CPU ``i % N``.  ``repeat`` is retained only as wave metadata and
    never creates implicit launches.
    """
    if not isinstance(value, list) or not value:
        raise RunSpecError("waves must be a non-empty list")
    task_ids = {task.task_id for task in tasks}
    controller_cpu_ids = set(defaults.controller_cpu_ids)
    compact = defaults.placement is not None
    seen_wave_ids: set[str] = set()
    referenced_task_ids: set[str] = set()
    waves: list[RunWaveSpec] = []
    for wave_index, raw_wave in enumerate(value):
        field = f"waves[{wave_index}]"
        if isinstance(raw_wave, dict):
            raw_keys = set(raw_wave)
            if compact and raw_keys.intersection({"metadata", "launches"}):
                raise RunSpecError(
                    "compact defaults cannot be combined with explicit waves; "
                    "all waves must use id/allocation/repeat form"
                )
            if not compact and raw_keys.intersection({"allocation", "repeat"}):
                raise RunSpecError(
                    "compact waves require defaults.worker_cpus and " "defaults.placement; all waves must use one form"
                )
        if compact:
            item = _object(
                raw_wave,
                field=field,
                required=frozenset({"id", "allocation", "repeat"}),
                optional=frozenset({"scored"}),
            )
        else:
            item = _object(
                raw_wave,
                field=field,
                required=frozenset({"id", "scored", "metadata", "launches"}),
            )
        wave_id = _identifier(item["id"], field=f"{field}.id")
        if wave_id in seen_wave_ids:
            raise RunSpecError(f"duplicate wave id: {wave_id}")
        seen_wave_ids.add(wave_id)

        scored = item.get("scored", True)
        if not isinstance(scored, bool):
            raise RunSpecError(f"{field}.scored must be a boolean")

        if compact:
            allocation = _positive_integer(item["allocation"], field=f"{field}.allocation")
            repeat = _positive_integer(item["repeat"], field=f"{field}.repeat")
            if allocation > len(defaults.worker_cpu_ids):
                raise RunSpecError(
                    f"{field}.allocation ({allocation}) exceeds the number of "
                    f"defaults.worker_cpus ({len(defaults.worker_cpu_ids)})"
                )
            launches = tuple(
                RunLaunchSpec(
                    task_id=task.task_id,
                    cpus=str(defaults.worker_cpu_ids[index % allocation]),
                    cpu_ids=(defaults.worker_cpu_ids[index % allocation],),
                )
                for index, task in enumerate(tasks)
            )
            referenced_task_ids.update(task.task_id for task in tasks)
            waves.append(
                RunWaveSpec(
                    wave_id=wave_id,
                    scored=scored,
                    metadata=(("allocation", allocation), ("repeat", repeat)),
                    launches=launches,
                )
            )
            continue

        raw_metadata = item["metadata"]
        if not isinstance(raw_metadata, dict):
            raise RunSpecError(f"{field}.metadata must be an object")
        if not all(isinstance(key, str) for key in raw_metadata):
            raise RunSpecError(f"{field}.metadata keys must be strings")
        metadata: list[tuple[str, MetadataScalar]] = []
        for key, raw_value in raw_metadata.items():
            metadata_key = _identifier(key, field=f"{field}.metadata key")
            metadata.append(
                (
                    metadata_key,
                    _metadata_scalar(
                        raw_value,
                        field=f"{field}.metadata.{metadata_key}",
                    ),
                )
            )

        raw_launches = item["launches"]
        if not isinstance(raw_launches, list) or not raw_launches:
            raise RunSpecError(f"{field}.launches must be a non-empty list")
        launches: list[RunLaunchSpec] = []
        wave_task_ids: set[str] = set()
        for launch_index, raw_launch in enumerate(raw_launches):
            launch_field = f"{field}.launches[{launch_index}]"
            launch_item = _object(
                raw_launch,
                field=launch_field,
                required=frozenset({"task", "cpus"}),
            )
            task_id = _identifier(launch_item["task"], field=f"{launch_field}.task")
            if task_id not in task_ids:
                raise RunSpecError(f"{launch_field}.task references unknown task: {task_id}")
            if task_id in wave_task_ids:
                raise RunSpecError(f"{field} launches task more than once: {task_id}")
            wave_task_ids.add(task_id)
            referenced_task_ids.add(task_id)
            cpus, cpu_ids = _canonical_cpuset(launch_item["cpus"], field=f"{launch_field}.cpus")
            overlap = sorted(controller_cpu_ids.intersection(cpu_ids))
            if overlap:
                raise RunSpecError(
                    f"{launch_field}.cpus overlaps defaults.controller_cpus: "
                    + ",".join(str(cpu_id) for cpu_id in overlap)
                )
            launches.append(RunLaunchSpec(task_id=task_id, cpus=cpus, cpu_ids=cpu_ids))
        waves.append(
            RunWaveSpec(
                wave_id=wave_id,
                scored=scored,
                metadata=tuple(metadata),
                launches=tuple(launches),
            )
        )

    unused_task_ids = sorted(task_ids - referenced_task_ids)
    if unused_task_ids:
        raise RunSpecError("task catalog contains unreferenced tasks: " + ", ".join(unused_task_ids))
    return tuple(waves)


def _load_ordered_run_spec(
    *,
    payload: Any,
    manifest_path: Path,
    raw: bytes,
) -> OrderedRunSpec:
    root = _object(
        payload,
        field="run",
        required=frozenset({"schema_version", "output", "defaults", "tasks", "waves"}),
        optional=frozenset({"input_root"}),
    )
    schema_version = _text(root["schema_version"], field="schema_version")
    if schema_version != SCHEMA_VERSION_V3:
        raise RunSpecError(f"unsupported run schema: {schema_version!r}")
    base = manifest_path.parent
    input_root = _path(root["input_root"], field="input_root", base=base) if "input_root" in root else base
    defaults = _load_defaults(root["defaults"])
    tasks = _load_tasks(
        root["tasks"],
        base=input_root,
        image_base=base if defaults.runtime == "enroot" else None,
    )
    waves = _load_waves(root["waves"], tasks=tasks, defaults=defaults)
    return OrderedRunSpec(
        schema_version=schema_version,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        output=_path(root["output"], field="output", base=base),
        input_root=input_root,
        defaults=defaults,
        tasks=tasks,
        waves=waves,
    )


def load_run_spec(manifest_path: Path) -> RunSpec | OrderedRunSpec:
    """Load, normalize, and content-bind one supported run YAML file."""
    manifest_path = manifest_path.resolve()
    try:
        raw = manifest_path.read_bytes()
        payload = yaml.load(raw, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise RunSpecError(f"unable to read run file {manifest_path}: {exc}") from exc
    declared_version = payload.get("schema_version") if isinstance(payload, dict) else None
    if declared_version == SCHEMA_VERSION_V3:
        return _load_ordered_run_spec(
            payload=payload,
            manifest_path=manifest_path,
            raw=raw,
        )
    required = frozenset({"schema_version", "output", "cpus", "tasks"})
    if declared_version == SCHEMA_VERSION_V2:
        required |= frozenset({"concurrency"})
    root = _object(
        payload,
        field="run",
        required=required,
        optional=frozenset({"repetitions"}),
    )
    schema_version = _text(root["schema_version"], field="schema_version")
    if schema_version not in {SCHEMA_VERSION, SCHEMA_VERSION_V2}:
        raise RunSpecError(f"unsupported run schema: {schema_version!r}")
    concurrency = 1 if schema_version == SCHEMA_VERSION else _positive_integer(root["concurrency"], field="concurrency")
    repetitions = _positive_integer(root.get("repetitions", 1), field="repetitions")
    cpus, cpu_ids = _canonical_cpuset(root["cpus"])
    base = manifest_path.parent
    tasks = _load_tasks(root["tasks"], base=base)
    attempt_count = len(tasks) * repetitions
    if concurrency > attempt_count:
        raise RunSpecError("concurrency cannot exceed the declared task-attempt count " f"({attempt_count})")
    return RunSpec(
        schema_version=schema_version,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        output=_path(root["output"], field="output", base=base),
        cpus=cpus,
        cpu_ids=cpu_ids,
        concurrency=concurrency,
        repetitions=repetitions,
        tasks=tasks,
    )

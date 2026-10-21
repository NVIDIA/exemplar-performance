# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable single-execution coordination for replay profilers.

This module is intentionally independent of the current Docker runner.  A
runtime session owns one gated sandbox execution; registered profilers observe
that session through a common lifecycle and emit the typed contracts from
``measurement.evidence``.  The existing runner can migrate to these protocols
without putting profiler-specific branches back into its execution loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .evidence import (
    CANONICAL_CLOCK_NAME,
    CanonicalTimeline,
    ClockMapping,
    ClockReference,
    EvidenceBundle,
    EvidenceRecord,
    EvidenceStream,
    LifecycleMarker,
    MeasuredScope,
    ProfilerManifest,
    RawArtifact,
)

EXECUTION_PLAN_SCHEMA_VERSION = "tasktrace-execution-plan.v1"
PROFILE_REQUEST_SCHEMA_VERSION = "tasktrace-profile-request.v1"
COORDINATION_RESULT_SCHEMA_VERSION = "tasktrace-profile-coordination-result.v1"
CANONICAL_CLOCK_ID = "host-monotonic"

DEFAULT_PROFILE_SETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "cpu-average": ("cgroup-cpu",),
        "cpu-peak": ("cgroup-cpu", "scheduler-concurrency"),
        "memory-detail": ("cgroup-memory",),
        "cpu-compare": (
            "cgroup-cpu",
            "scheduler-concurrency",
            "cgroup-memory",
        ),
        "software-deep": ("process-exec",),
        "hardware-deep": ("hardware-perf",),
    }
)

_SECRET_KEY = re.compile(
    r"(?:^|[_-])(secret|token|password|passwd|api[_-]?key|credential|authorization)(?:$|[_-])",
    re.IGNORECASE,
)


class CoordinatorValidationError(ValueError):
    """Raised before launch when a plan or profiler selection is unsafe."""


class ProfileRunError(RuntimeError):
    """A run failed closed after cleanup; partial evidence remains inspectable."""

    def __init__(self, message: str, *, result: CoordinationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class ProfileCancellationError(RuntimeError):
    """A caller requested cooperative cancellation of one active profile."""


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoordinatorValidationError(f"{name} must be a non-empty string")
    return value


def _freeze_json(value: Any, name: str) -> Any:
    """Validate a JSON value and recursively remove mutable containers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CoordinatorValidationError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        if not all(isinstance(key, str) for key in value):
            raise CoordinatorValidationError(f"{name} contains a non-string key")
        for key in sorted(value):
            frozen[key] = _freeze_json(value[key], f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{name}[{index}]") for index, item in enumerate(value))
    raise CoordinatorValidationError(f"{name} must contain only JSON values")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_digest(name: str, value: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"tasktrace-execution-plan-environment-v1\0")
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def _secret_safe_projection(value: Any, *, key: str | None = None) -> Any:
    """Redact secret-looking configuration fields while retaining identity."""
    if key is not None and _SECRET_KEY.search(key):
        encoded = _canonical_json_bytes(value)
        return {
            "redacted": True,
            "value_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {item_key: _secret_safe_projection(item, key=item_key) for item_key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_secret_safe_projection(item) for item in value]
    return value


@dataclass(frozen=True)
class MountBehavior:
    """Execution-relevant mount behavior without requiring a raw host path."""

    target: str
    kind: str = "bind"
    read_only: bool = False
    source_identity: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = PurePosixPath(_require_text(self.target, "mount.target"))
        if not target.is_absolute() or ".." in target.parts:
            raise CoordinatorValidationError("mount.target must be an absolute safe path")
        _require_text(self.kind, "mount.kind")
        if not isinstance(self.read_only, bool):
            raise CoordinatorValidationError("mount.read_only must be boolean")
        object.__setattr__(
            self,
            "source_identity",
            _freeze_json(self.source_identity, "mount.source_identity"),
        )
        object.__setattr__(self, "options", _freeze_json(self.options, "mount.options"))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kind": self.kind,
            "read_only": self.read_only,
            "source_identity": _json_value(self.source_identity),
            "options": _json_value(self.options),
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable adapter output consumed once by a runtime session."""

    workload_identity: Mapping[str, Any]
    source_identity: Mapping[str, Any]
    command: tuple[str, ...]
    cwd: str
    mounts: tuple[MountBehavior, ...]
    image_reference: str
    image_identity: Mapping[str, Any]
    runtime: str
    network: str | None
    expected_outcome: Mapping[str, Any]
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EXECUTION_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_PLAN_SCHEMA_VERSION:
            raise CoordinatorValidationError(f"unsupported execution plan schema: {self.schema_version!r}")
        workload = _freeze_json(self.workload_identity, "plan.workload_identity")
        source = _freeze_json(self.source_identity, "plan.source_identity")
        image = _freeze_json(self.image_identity, "plan.image_identity")
        outcome = _freeze_json(self.expected_outcome, "plan.expected_outcome")
        if not workload or not source or not image or not outcome:
            raise CoordinatorValidationError(
                "plan workload, source, image, and expected outcome identities must not be empty"
            )
        object.__setattr__(self, "workload_identity", workload)
        object.__setattr__(self, "source_identity", source)
        object.__setattr__(self, "image_identity", image)
        object.__setattr__(self, "expected_outcome", outcome)
        command = tuple(_require_text(item, "plan.command") for item in self.command)
        if not command:
            raise CoordinatorValidationError("plan.command must not be empty")
        object.__setattr__(self, "command", command)
        cwd = PurePosixPath(_require_text(self.cwd, "plan.cwd"))
        if not cwd.is_absolute() or ".." in cwd.parts:
            raise CoordinatorValidationError("plan.cwd must be an absolute safe path")
        _require_text(self.image_reference, "plan.image_reference")
        _require_text(self.runtime, "plan.runtime")
        if self.network is not None:
            _require_text(self.network, "plan.network")

        mounts = tuple(
            sorted(
                self.mounts,
                key=lambda mount: (
                    mount.target,
                    mount.kind,
                    _canonical_json_bytes(mount.to_manifest()),
                ),
            )
        )
        targets = [mount.target for mount in mounts]
        if len(targets) != len(set(targets)):
            raise CoordinatorValidationError("plan.mounts must have unique targets")
        object.__setattr__(self, "mounts", mounts)

        environment: dict[str, str] = {}
        for name in sorted(self.environment):
            _require_text(name, "plan.environment name")
            value = self.environment[name]
            if not isinstance(value, str):
                raise CoordinatorValidationError("plan.environment values must be strings")
            environment[name] = value
        object.__setattr__(self, "environment", MappingProxyType(environment))
        object.__setattr__(self, "extensions", _freeze_json(self.extensions, "plan.extensions"))

    def to_manifest(self) -> dict[str, Any]:
        """Return archival provenance with no verbatim environment values."""
        return {
            "schema_version": self.schema_version,
            "workload_identity": _json_value(self.workload_identity),
            "source_identity": _json_value(self.source_identity),
            "command": list(self.command),
            "cwd": self.cwd,
            "mounts": [mount.to_manifest() for mount in self.mounts],
            "image_reference": self.image_reference,
            "image_identity": _json_value(self.image_identity),
            "runtime": self.runtime,
            "network": self.network,
            "expected_outcome": _json_value(self.expected_outcome),
            "environment": [
                {
                    "name": name,
                    "value_sha256": _value_digest(name, value),
                }
                for name, value in self.environment.items()
            ],
            "extensions": _json_value(self.extensions),
        }

    @property
    def canonical_digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.to_manifest())).hexdigest()


@dataclass(frozen=True)
class ProfileRequest:
    """Profiler selection, deliberately separate from workload execution."""

    profile_sets: tuple[str, ...] = ()
    profilers: tuple[str, ...] = ()
    configuration: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    schema_version: str = PROFILE_REQUEST_SCHEMA_VERSION
    mode: str = "profile"

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_REQUEST_SCHEMA_VERSION:
            raise CoordinatorValidationError(f"unsupported profile request schema: {self.schema_version!r}")
        if self.mode not in {"profile", "baseline"}:
            raise CoordinatorValidationError("profile request mode must be 'profile' or 'baseline'")
        profile_sets = tuple(sorted({_require_text(item, "request.profile_sets") for item in self.profile_sets}))
        profilers = tuple(sorted({_require_text(item, "request.profilers") for item in self.profilers}))
        if self.mode == "profile" and not profile_sets and not profilers:
            raise CoordinatorValidationError("profile request must select a profile set or direct profiler")
        if self.mode == "baseline" and (profile_sets or profilers or self.configuration):
            raise CoordinatorValidationError("baseline profile request cannot select or configure profilers")
        object.__setattr__(self, "profile_sets", profile_sets)
        object.__setattr__(self, "profilers", profilers)
        configuration: dict[str, Mapping[str, Any]] = {}
        for profiler_id in sorted(self.configuration):
            _require_text(profiler_id, "request.configuration profiler ID")
            value = self.configuration[profiler_id]
            if not isinstance(value, Mapping):
                raise CoordinatorValidationError("request.configuration values must be objects")
            configuration[profiler_id] = _freeze_json(value, f"request.configuration.{profiler_id}")
        object.__setattr__(self, "configuration", MappingProxyType(configuration))

    def configuration_for(self, profiler_id: str) -> Mapping[str, Any]:
        return self.configuration.get(profiler_id, MappingProxyType({}))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "profile_sets": list(self.profile_sets),
            "profilers": list(self.profilers),
            "configuration": _secret_safe_projection(self.configuration),
        }


@dataclass(frozen=True)
class ProfilerDescriptor:
    profiler_id: str
    version: str
    requirements: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    intrusive: bool = False
    intrusive_reason: str | None = None
    invalidates_metrics: tuple[str, ...] = ()
    order: int = 1000
    measurement_domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.profiler_id, "profiler.profiler_id")
        _require_text(self.version, "profiler.version")
        requirements = tuple(sorted({_require_text(item, "profiler.requirements") for item in self.requirements}))
        conflicts = tuple(sorted({_require_text(item, "profiler.conflicts") for item in self.conflicts}))
        measurement_domains = tuple(
            sorted({_require_text(item, "profiler.measurement_domains") for item in self.measurement_domains})
        )
        invalidates = tuple(
            sorted({_require_text(item, "profiler.invalidates_metrics") for item in self.invalidates_metrics})
        )
        if self.profiler_id in conflicts:
            raise CoordinatorValidationError("profiler cannot conflict with itself")
        if not isinstance(self.intrusive, bool):
            raise CoordinatorValidationError("profiler.intrusive must be boolean")
        if self.intrusive and self.intrusive_reason is None:
            raise CoordinatorValidationError("intrusive profiler must declare an intrusive_reason")
        if not self.intrusive and (self.intrusive_reason is not None or invalidates):
            raise CoordinatorValidationError("non-intrusive profiler cannot declare intrusive effects")
        if self.intrusive_reason is not None:
            _require_text(self.intrusive_reason, "profiler.intrusive_reason")
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise CoordinatorValidationError("profiler.order must be an integer")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "measurement_domains", measurement_domains)
        object.__setattr__(self, "invalidates_metrics", invalidates)


class Lifecycle:
    """One coordinator-owned marker sequence shared by every profiler."""

    def __init__(self, clock_ns: Callable[[], int]) -> None:
        self._clock_ns = clock_ns
        self._markers: list[LifecycleMarker] = []

    @property
    def markers(self) -> tuple[LifecycleMarker, ...]:
        return tuple(self._markers)

    def get(self, name: str) -> LifecycleMarker | None:
        return next((marker for marker in self._markers if marker.name == name), None)

    def _mark(self, name: str) -> LifecycleMarker:
        if self.get(name) is not None:
            raise CoordinatorValidationError(f"duplicate lifecycle marker: {name}")
        timestamp_ns = self._clock_ns()
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool) or timestamp_ns < 0:
            raise CoordinatorValidationError("monotonic clock returned an invalid timestamp")
        if self._markers and timestamp_ns < self._markers[-1].canonical_timestamp_ns:
            raise CoordinatorValidationError("monotonic clock moved backwards")
        marker = LifecycleMarker(name=name, canonical_timestamp_ns=timestamp_ns)
        self._markers.append(marker)
        return marker


@dataclass(frozen=True)
class ProfilerContext:
    run_id: str
    plan: ExecutionPlan
    plan_digest: str
    request: ProfileRequest
    scope: MeasuredScope
    lifecycle: Lifecycle
    canonical_clock_id: str = CANONICAL_CLOCK_ID


@dataclass(frozen=True)
class ProfilerReport:
    """Profiler data before the coordinator creates its evidence manifest."""

    status: str
    streams: tuple[EvidenceStream, ...] = ()
    artifacts: tuple[RawArtifact, ...] = ()
    clocks: tuple[ClockReference, ...] = ()
    clock_mappings: tuple[ClockMapping, ...] = ()
    reasons: tuple[str, ...] = ()
    raw_report: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"valid", "invalid", "missing", "dropped", "unalignable"}:
            raise CoordinatorValidationError(f"unsupported profiler status: {self.status!r}")
        reasons = tuple(sorted({_require_text(reason, "profiler report reason") for reason in self.reasons}))
        if self.status == "valid" and reasons:
            raise CoordinatorValidationError("valid profiler report cannot have reasons")
        if self.status != "valid" and not reasons:
            raise CoordinatorValidationError("non-valid profiler report requires a reason")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "streams", tuple(self.streams))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "clocks", tuple(self.clocks))
        object.__setattr__(self, "clock_mappings", tuple(self.clock_mappings))
        object.__setattr__(self, "raw_report", _freeze_json(self.raw_report, "profiler.raw_report"))
        object.__setattr__(self, "provenance", _freeze_json(self.provenance, "profiler.provenance"))
        object.__setattr__(self, "extensions", _freeze_json(self.extensions, "profiler.extensions"))


@runtime_checkable
class Profiler(Protocol):
    def attach(self, context: ProfilerContext) -> None: ...

    def start(self, context: ProfilerContext, marker: LifecycleMarker) -> None: ...

    def poll(self, context: ProfilerContext) -> None: ...

    def command_end(self, context: ProfilerContext, marker: LifecycleMarker) -> None: ...

    def stop(self, context: ProfilerContext, marker: LifecycleMarker) -> None: ...

    def finalize(self, context: ProfilerContext) -> ProfilerReport: ...

    def close(self) -> None: ...


@runtime_checkable
class RuntimeSession(Protocol):
    """One gated sandbox instance. ``launch`` may be called exactly once."""

    @property
    def capabilities(self) -> frozenset[str]: ...

    def launch(
        self,
        plan: ExecutionPlan,
        *,
        run_id: str,
        marker: LifecycleMarker,
    ) -> MeasuredScope: ...

    def release_sandbox(self, marker: LifecycleMarker) -> None: ...

    def wait_command_ready(self) -> None: ...

    def release_command(self, marker: LifecycleMarker) -> None: ...

    def poll_command(self, timeout_sec: float) -> bool: ...

    def begin_finish(self, marker: LifecycleMarker) -> None: ...

    def poll_finish(self, timeout_sec: float) -> bool: ...

    def finish(self, marker: LifecycleMarker) -> Mapping[str, Any]: ...

    def abort(self, reason: str) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


ProfilerFactory = Callable[[Mapping[str, Any]], Profiler]


@dataclass(frozen=True)
class ResolvedProfiler:
    descriptor: ProfilerDescriptor
    factory: ProfilerFactory
    configuration: Mapping[str, Any]


MEASUREMENT_PERTURBATION_SCHEMA_VERSION = "measurement-perturbation.v1"


def _measurement_perturbations(
    target: ProfilerDescriptor,
    resolved: Sequence[ResolvedProfiler],
) -> tuple[dict[str, Any], ...]:
    """Return selected intrusive collectors that perturb ``target`` domains.

    The coordinator, rather than an individual profiler, owns this join.  That
    makes a newly registered intrusive profiler affect every matching built-in
    or third-party measurement without profile-set-specific branches.
    """

    target_domains = set(target.measurement_domains)
    if not target_domains:
        return ()
    perturbations: list[dict[str, Any]] = []
    for source in resolved:
        descriptor = source.descriptor
        if not descriptor.intrusive:
            continue
        affected = sorted(target_domains & set(descriptor.invalidates_metrics))
        if not affected:
            continue
        perturbations.append(
            {
                "schema_version": MEASUREMENT_PERTURBATION_SCHEMA_VERSION,
                "code": "intrusive-profiler-invalidated-measurement-domain",
                "invalidating_profiler_id": descriptor.profiler_id,
                "invalidating_profiler_version": descriptor.version,
                "measurement_domains": affected,
                "intrusive_reason": descriptor.intrusive_reason,
            }
        )
    return tuple(perturbations)


def _perturbation_reason(perturbation: Mapping[str, Any]) -> str:
    domains = ", ".join(str(item) for item in perturbation["measurement_domains"])
    return (
        f"measurement domain {domains} was perturbed by intrusive profiler "
        f"{perturbation['invalidating_profiler_id']}: "
        f"{perturbation['intrusive_reason']}"
    )


def _invalidate_record_for_perturbation(
    record: EvidenceRecord,
    *,
    measurement_domains: tuple[str, ...],
    perturbations: tuple[dict[str, Any], ...],
    reasons: tuple[str, ...],
) -> EvidenceRecord:
    # An already unalignable/dropped record must retain that more specific
    # state: converting it to ``invalid`` could require fabricating canonical
    # timing or a value.  No record that was usable as valid evidence remains
    # valid, and every record still carries the machine-readable invalidation.
    status = "invalid" if record.status in {"valid", "invalid"} else record.status
    attributes = _json_value(record.attributes)
    if isinstance(attributes.get("natural_peak_trustworthy"), bool):
        attributes["natural_peak_trustworthy"] = False
    return replace(
        record,
        status=status,
        reasons=tuple(sorted(set(record.reasons) | set(reasons))),
        attributes={
            **attributes,
            "measurement_domains": list(measurement_domains),
            "measurement_perturbations": list(perturbations),
        },
    )


def _apply_measurement_contract(
    report: ProfilerReport,
    *,
    descriptor: ProfilerDescriptor,
    perturbations: tuple[dict[str, Any], ...],
) -> ProfilerReport:
    """Annotate profiler output and fail matching domain evidence closed."""

    domains = descriptor.measurement_domains
    reasons = tuple(_perturbation_reason(item) for item in perturbations)
    streams: list[EvidenceStream] = []
    for stream in report.streams:
        records = stream.records
        status = stream.status
        stream_reasons = stream.reasons
        if perturbations:
            records = tuple(
                _invalidate_record_for_perturbation(
                    record,
                    measurement_domains=domains,
                    perturbations=perturbations,
                    reasons=reasons,
                )
                for record in stream.records
            )
            # Missing, dropped, and unalignable streams remain explicitly in
            # those unusable states; valid/invalid streams become invalid.
            status = "invalid" if stream.status in {"valid", "invalid"} else stream.status
            stream_reasons = tuple(sorted(set(stream.reasons) | set(reasons)))
        streams.append(
            replace(
                stream,
                status=status,
                reasons=stream_reasons,
                records=records,
                extensions={
                    **_json_value(stream.extensions),
                    "measurement_domains": list(domains),
                    "measurement_perturbations": list(perturbations),
                },
            )
        )

    raw_report = _json_value(report.raw_report)
    if not isinstance(raw_report, dict):
        raw_report = {"value": raw_report}
    raw_report["measurement_domain_contract"] = {
        "measurement_domains": list(domains),
        "status": "invalid" if perturbations else "unperturbed",
        "perturbations": list(perturbations),
    }
    if perturbations and isinstance(raw_report.get("natural_peak_trustworthy"), bool):
        # The exact observed peak remains retained, but an intrusive profiler
        # can change the workload's charged memory. Keep that distinction in
        # both the typed streams and the raw memory report.
        raw_report["natural_peak_trustworthy"] = False
        prior = raw_report.get("natural_peak_invalid_reasons")
        if isinstance(prior, list):
            raw_report["natural_peak_invalid_reasons"] = list(dict.fromkeys([*(str(item) for item in prior), *reasons]))

    return ProfilerReport(
        status="invalid" if perturbations else report.status,
        streams=tuple(streams),
        artifacts=report.artifacts,
        clocks=report.clocks,
        clock_mappings=report.clock_mappings,
        reasons=(tuple(sorted(set(report.reasons) | set(reasons))) if perturbations else report.reasons),
        raw_report=raw_report,
        provenance=report.provenance,
        extensions={
            **_json_value(report.extensions),
            "measurement_domains": list(domains),
            "measurement_perturbations": list(perturbations),
        },
    )


class ProfilerRegistry:
    """Profiler implementations and named sets, resolved deterministically."""

    def __init__(
        self,
        *,
        profile_sets: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        selected_sets = DEFAULT_PROFILE_SETS if profile_sets is None else profile_sets
        self._profile_sets: dict[str, tuple[str, ...]] = {}
        self._profilers: dict[str, tuple[ProfilerDescriptor, ProfilerFactory]] = {}
        for name, profiler_ids in selected_sets.items():
            self.register_profile_set(name, profiler_ids)

    def register_profile_set(self, name: str, profiler_ids: Sequence[str]) -> None:
        name = _require_text(name, "profile set name")
        if name in self._profile_sets:
            raise CoordinatorValidationError(f"duplicate profile set: {name}")
        normalized = tuple(sorted({_require_text(item, "profile set profiler ID") for item in profiler_ids}))
        if not normalized:
            raise CoordinatorValidationError("profile set must contain a profiler")
        self._profile_sets[name] = normalized

    def register(self, descriptor: ProfilerDescriptor, factory: ProfilerFactory) -> None:
        if descriptor.profiler_id in self._profilers:
            raise CoordinatorValidationError(f"duplicate profiler registration: {descriptor.profiler_id}")
        if not callable(factory):
            raise CoordinatorValidationError("profiler factory must be callable")
        self._profilers[descriptor.profiler_id] = (descriptor, factory)

    def resolve(self, request: ProfileRequest) -> tuple[ResolvedProfiler, ...]:
        selected = set(request.profilers)
        unknown_sets = sorted(set(request.profile_sets) - set(self._profile_sets))
        if unknown_sets:
            raise CoordinatorValidationError("unknown profile sets: " + ", ".join(unknown_sets))
        for name in request.profile_sets:
            selected.update(self._profile_sets[name])
        unknown = sorted(selected - set(self._profilers))
        if unknown:
            raise CoordinatorValidationError("selected profilers are not registered: " + ", ".join(unknown))
        unselected_config = sorted(set(request.configuration) - selected)
        if unselected_config:
            raise CoordinatorValidationError(
                "configuration supplied for unselected profilers: " + ", ".join(unselected_config)
            )
        for profiler_id in sorted(selected):
            descriptor = self._profilers[profiler_id][0]
            conflicts = sorted(set(descriptor.conflicts) & selected)
            if conflicts:
                raise CoordinatorValidationError(f"profiler {profiler_id!r} conflicts with: " + ", ".join(conflicts))
        resolved = [
            ResolvedProfiler(
                descriptor=self._profilers[profiler_id][0],
                factory=self._profilers[profiler_id][1],
                configuration=request.configuration_for(profiler_id),
            )
            for profiler_id in selected
        ]
        return tuple(
            sorted(
                resolved,
                key=lambda item: (item.descriptor.order, item.descriptor.profiler_id),
            )
        )


@dataclass(frozen=True)
class CoordinationFailure:
    phase: str
    message: str
    profiler_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"phase": self.phase, "message": self.message}
        if self.profiler_id is not None:
            result["profiler_id"] = self.profiler_id
        return result


@dataclass(frozen=True)
class CoordinationResult:
    run_id: str
    execution_plan_digest: str
    status: str
    resolved_profiler_ids: tuple[str, ...]
    intrusive_profiler_ids: tuple[str, ...]
    scope: MeasuredScope
    timeline: CanonicalTimeline
    manifests: tuple[ProfilerManifest, ...]
    raw_reports: Mapping[str, Any]
    runtime_report: Mapping[str, Any]
    failures: tuple[CoordinationFailure, ...]
    evidence_bundle: EvidenceBundle
    schema_version: str = COORDINATION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"valid", "invalid"}:
            raise CoordinatorValidationError("coordination result status is invalid")
        object.__setattr__(self, "raw_reports", _freeze_json(self.raw_reports, "result.raw_reports"))
        object.__setattr__(
            self,
            "runtime_report",
            _freeze_json(self.runtime_report, "result.runtime_report"),
        )


class ProfileCoordinator:
    """Run registered profilers around one gated runtime session.

    ``cancel_requested``, when supplied, is a cooperative boolean callback
    sampled at lifecycle gates and command/finish polls.  Cancellation follows
    the ordinary fail-closed abort, profiler-finalize, and session-close path.
    """

    def __init__(
        self,
        *,
        registry: ProfilerRegistry,
        session_factory: Callable[[], RuntimeSession],
        clock_ns: Callable[[], int] = time.monotonic_ns,
        run_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
        cancel_requested: Callable[[], bool] | None = None,
        poll_interval_sec: float = 0.025,
        clock_resolution_ns: int = 1,
    ) -> None:
        if poll_interval_sec <= 0:
            raise CoordinatorValidationError("poll interval must be positive")
        if clock_resolution_ns <= 0:
            raise CoordinatorValidationError("clock resolution must be positive")
        self.registry = registry
        self.session_factory = session_factory
        self.clock_ns = clock_ns
        self.run_id_factory = run_id_factory
        self.cancel_requested = cancel_requested
        self.poll_interval_sec = poll_interval_sec
        self.clock_resolution_ns = clock_resolution_ns

    def _raise_if_cancel_requested(self) -> None:
        if self.cancel_requested is None:
            return
        try:
            requested = self.cancel_requested()
        except Exception as exc:
            raise ProfileCancellationError(f"profile cancellation callback failed: {exc}") from exc
        if not isinstance(requested, bool):
            raise ProfileCancellationError("profile cancellation callback must return a boolean")
        if requested:
            raise ProfileCancellationError("profile cancellation requested")

    def run(self, plan: ExecutionPlan, request: ProfileRequest) -> CoordinationResult:
        resolved = self.registry.resolve(request)
        run_id = self.run_id_factory()
        try:
            uuid.UUID(run_id)
        except (ValueError, AttributeError) as exc:
            raise CoordinatorValidationError("run ID factory must return a UUID") from exc
        lifecycle = Lifecycle(self.clock_ns)
        session = self.session_factory()
        missing_requirements = {
            item.descriptor.profiler_id: sorted(set(item.descriptor.requirements) - set(session.capabilities))
            for item in resolved
            if set(item.descriptor.requirements) - set(session.capabilities)
        }
        if missing_requirements:
            try:
                session.close()
            except Exception:
                pass
            rendered = "; ".join(
                f"{profiler_id}: {', '.join(requirements)}"
                for profiler_id, requirements in sorted(missing_requirements.items())
            )
            raise CoordinatorValidationError("runtime session lacks profiler requirements: " + rendered)

        instances: list[tuple[ResolvedProfiler, Profiler]] = []
        try:
            for item in resolved:
                profiler = item.factory(item.configuration)
                if not isinstance(profiler, Profiler):
                    raise CoordinatorValidationError(
                        f"factory for {item.descriptor.profiler_id!r} did not return " "a Profiler implementation"
                    )
                instances.append((item, profiler))
        except Exception as exc:
            for _item, profiler in reversed(instances):
                try:
                    profiler.close()
                except Exception:
                    pass
            try:
                session.close()
            except Exception:
                pass
            raise CoordinatorValidationError(f"unable to construct selected profiler: {exc}") from exc

        failures: list[CoordinationFailure] = []
        profiler_failures: dict[str, list[str]] = {item.descriptor.profiler_id: [] for item, _profiler in instances}
        reports: dict[str, ProfilerReport] = {}
        attached: list[str] = []
        started: list[str] = []
        stopped: set[str] = set()
        stop_attempted: set[str] = set()
        command_end_attempted: set[str] = set()
        scope: MeasuredScope | None = None
        context: ProfilerContext | None = None
        runtime_report: Mapping[str, Any] = {}
        session_launched = False
        session_finished = False
        command_released = False
        execution_aborted = False
        verification_requested = bool(plan.extensions.get("verification_command_sha256"))
        active_phase = "runtime"
        active_profiler_id: str | None = None

        def check_cancellation(phase: str) -> None:
            nonlocal active_phase, active_profiler_id
            active_phase = phase
            active_profiler_id = None
            self._raise_if_cancel_requested()

        def fail(phase: str, exc: Exception, profiler_id: str | None = None) -> None:
            message = f"{type(exc).__name__}: {exc}"
            failures.append(CoordinationFailure(phase, message, profiler_id))
            if profiler_id is not None:
                profiler_failures[profiler_id].append(f"{phase} failed: {message}")

        try:
            check_cancellation("cancel_before_launch")
            active_phase = "lifecycle"
            run_start = lifecycle._mark("run_start")
            active_phase = "runtime_launch"
            scope = session.launch(plan, run_id=run_id, marker=run_start)
            session_launched = True
            if not isinstance(scope, MeasuredScope):
                raise CoordinatorValidationError("runtime session launch must return a MeasuredScope")
            check_cancellation("cancel_after_launch")
            active_phase = "lifecycle"
            lifecycle._mark("sandbox_ready")
            context = ProfilerContext(
                run_id=run_id,
                plan=plan,
                plan_digest=plan.canonical_digest,
                request=request,
                scope=scope,
                lifecycle=lifecycle,
            )
            for item, profiler in instances:
                active_phase = "attach"
                active_profiler_id = item.descriptor.profiler_id
                profiler.attach(context)
                attached.append(item.descriptor.profiler_id)
            active_profiler_id = None
            active_phase = "lifecycle"
            sandbox_start = lifecycle._mark("sandbox_start")
            active_phase = "runtime_release_sandbox"
            session.release_sandbox(sandbox_start)
            active_phase = "runtime_wait_command_ready"
            session.wait_command_ready()
            check_cancellation("cancel_before_command")
            active_phase = "lifecycle"
            lifecycle._mark("command_ready")
            measured_start = lifecycle._mark("measured_work_start")
            for item, profiler in instances:
                active_phase = "start"
                active_profiler_id = item.descriptor.profiler_id
                profiler.start(context, measured_start)
                started.append(item.descriptor.profiler_id)
            active_profiler_id = None
            active_phase = "runtime_release_command"
            session.release_command(measured_start)
            command_released = True
            while True:
                check_cancellation("cancel_during_command")
                active_phase = "runtime_poll"
                complete = session.poll_command(self.poll_interval_sec)
                for item, profiler in instances:
                    active_phase = "poll"
                    active_profiler_id = item.descriptor.profiler_id
                    profiler.poll(context)
                active_profiler_id = None
                check_cancellation("cancel_during_command")
                if complete:
                    break
            active_phase = "lifecycle"
            command_end = lifecycle._mark("command_end")
            for item, profiler in instances:
                active_phase = "command_end"
                active_profiler_id = item.descriptor.profiler_id
                command_end_attempted.add(item.descriptor.profiler_id)
                profiler.command_end(context, command_end)
            active_profiler_id = None
            active_phase = "runtime_begin_finish"
            session.begin_finish(command_end)
            while True:
                check_cancellation("cancel_during_finish")
                active_phase = "runtime_finish_poll"
                complete = session.poll_finish(self.poll_interval_sec)
                for item, profiler in instances:
                    active_phase = "poll_finish"
                    active_profiler_id = item.descriptor.profiler_id
                    profiler.poll(context)
                active_profiler_id = None
                check_cancellation("cancel_during_finish")
                if complete:
                    break
            active_phase = "lifecycle"
            measured_end = lifecycle._mark("measured_work_end")
            for item, profiler in reversed(instances):
                active_phase = "stop"
                active_profiler_id = item.descriptor.profiler_id
                stop_attempted.add(item.descriptor.profiler_id)
                profiler.stop(context, measured_end)
                stopped.add(item.descriptor.profiler_id)
            active_profiler_id = None
            finish_marker = measured_end
            if verification_requested:
                active_phase = "lifecycle"
                finish_marker = lifecycle._mark("verification_start")
            check_cancellation("cancel_before_runtime_finish")
            active_phase = "runtime_finish"
            runtime_report = session.finish(finish_marker)
            session_finished = True
            if verification_requested:
                verification = runtime_report.get("verification")
                if not isinstance(verification, Mapping) or verification.get("status") not in {"passed", "failed"}:
                    raise CoordinatorValidationError("runtime did not return a complete workload verification result")
                active_phase = "lifecycle"
                lifecycle._mark("verification_end")
            active_phase = "lifecycle"
            lifecycle._mark("run_end")
        except Exception as exc:
            fail(active_phase, exc, active_profiler_id)
            execution_aborted = True

            # Keep observers active through runtime abort so descendant cleanup
            # remains inside the controlled measurement. Only a command that
            # was actually released receives a command_end notification.
            if session_launched and not session_finished:
                try:
                    runtime_report = session.abort(str(exc))
                except Exception as abort_exc:
                    fail("runtime_abort", abort_exc)

            if context is not None and command_released:
                command_end = lifecycle.get("command_end")
                if command_end is None:
                    try:
                        command_end = lifecycle._mark("command_end")
                    except Exception as marker_exc:
                        fail("lifecycle_cleanup", marker_exc)
                if command_end is not None:
                    for item, profiler in instances:
                        profiler_id = item.descriptor.profiler_id
                        if profiler_id not in started or profiler_id in command_end_attempted:
                            continue
                        try:
                            command_end_attempted.add(profiler_id)
                            profiler.command_end(context, command_end)
                        except Exception as command_end_exc:
                            fail("command_end_cleanup", command_end_exc, profiler_id)

            if context is not None and lifecycle.get("measured_work_start") is not None:
                measured_end = lifecycle.get("measured_work_end")
                if measured_end is None:
                    try:
                        measured_end = lifecycle._mark("measured_work_end")
                    except Exception as marker_exc:
                        fail("lifecycle_cleanup", marker_exc)
                if measured_end is not None:
                    for item, profiler in reversed(instances):
                        profiler_id = item.descriptor.profiler_id
                        if profiler_id not in started or profiler_id in stopped or profiler_id in stop_attempted:
                            continue
                        try:
                            stop_attempted.add(profiler_id)
                            profiler.stop(context, measured_end)
                            stopped.add(profiler_id)
                        except Exception as stop_exc:
                            fail("stop_cleanup", stop_exc, profiler_id)
            if lifecycle.get("run_start") is not None and lifecycle.get("run_end") is None:
                try:
                    lifecycle._mark("run_end")
                except Exception as marker_exc:
                    fail("lifecycle_cleanup", marker_exc)

        if scope is None or context is None:
            for _item, profiler in reversed(instances):
                try:
                    profiler.close()
                except Exception:
                    pass
            try:
                session.close()
            except Exception:
                pass
            raise ProfileRunError(failures[0].message if failures else "runtime session failed before scope creation")

        if execution_aborted:
            primary = failures[0].message if failures else "coordinated execution aborted"
            for item, _profiler in instances:
                profiler_failures[item.descriptor.profiler_id].append(
                    f"run aborted before complete evidence: {primary}"
                )

        for item, profiler in instances:
            profiler_id = item.descriptor.profiler_id
            if profiler_id not in attached:
                continue
            try:
                report = profiler.finalize(context)
                if not isinstance(report, ProfilerReport):
                    raise CoordinatorValidationError("profiler.finalize must return ProfilerReport")
                for stream in report.streams:
                    if stream.profiler_id != profiler_id:
                        raise CoordinatorValidationError("profiler emitted a stream under another profiler ID")
                reports[profiler_id] = report
            except Exception as exc:
                fail("finalize", exc, profiler_id)

        for item, profiler in reversed(instances):
            profiler_id = item.descriptor.profiler_id
            try:
                profiler.close()
            except Exception as exc:
                fail("close", exc, profiler_id)
        try:
            session.close()
        except Exception as exc:
            fail("runtime_close", exc)

        result = self._build_result(
            run_id=run_id,
            plan=plan,
            request=request,
            resolved=resolved,
            scope=scope,
            lifecycle=lifecycle,
            reports=reports,
            profiler_failures=profiler_failures,
            runtime_report=runtime_report,
            failures=failures,
        )
        if failures:
            raise ProfileRunError(
                "profile run failed closed: "
                + "; ".join(f"{failure.phase}: {failure.message}" for failure in failures),
                result=result,
            )
        return result

    def _build_result(
        self,
        *,
        run_id: str,
        plan: ExecutionPlan,
        request: ProfileRequest,
        resolved: tuple[ResolvedProfiler, ...],
        scope: MeasuredScope,
        lifecycle: Lifecycle,
        reports: Mapping[str, ProfilerReport],
        profiler_failures: Mapping[str, list[str]],
        runtime_report: Mapping[str, Any],
        failures: list[CoordinationFailure],
    ) -> CoordinationResult:
        run_start = lifecycle.get("run_start")
        run_end = lifecycle.get("run_end")
        assert run_start is not None and run_end is not None
        if run_end.canonical_timestamp_ns <= run_start.canonical_timestamp_ns:
            raise ProfileRunError("coordinator run window did not have positive duration")
        origin = lifecycle.get("measured_work_start") or run_start
        timeline = CanonicalTimeline(
            clock_id=CANONICAL_CLOCK_ID,
            origin_ns=origin.canonical_timestamp_ns,
            start_ns=run_start.canonical_timestamp_ns,
            end_ns=run_end.canonical_timestamp_ns,
            lifecycle_markers=lifecycle.markers,
        )

        manifests: list[ProfilerManifest] = []
        streams: list[EvidenceStream] = []
        artifacts: dict[str, RawArtifact] = {}
        clocks: list[ClockReference] = [
            ClockReference(
                clock_id=CANONICAL_CLOCK_ID,
                clock_name=CANONICAL_CLOCK_NAME,
                resolution_ns=self.clock_resolution_ns,
            )
        ]
        mappings: list[ClockMapping] = []
        raw_reports: dict[str, Any] = {}
        for item in resolved:
            descriptor = item.descriptor
            profiler_id = descriptor.profiler_id
            report = reports.get(profiler_id)
            perturbations = _measurement_perturbations(descriptor, resolved)
            perturbation_reasons = tuple(_perturbation_reason(value) for value in perturbations)
            if report is not None:
                report = _apply_measurement_contract(
                    report,
                    descriptor=descriptor,
                    perturbations=perturbations,
                )
            errors = list(dict.fromkeys(profiler_failures.get(profiler_id, [])))
            if report is None:
                status = "invalid"
                reasons = tuple(
                    dict.fromkeys(
                        [
                            *(errors or ["profiler produced no final report"]),
                            *perturbation_reasons,
                        ]
                    )
                )
                report_streams: tuple[EvidenceStream, ...] = ()
            elif errors:
                status = "invalid"
                reasons = tuple(dict.fromkeys([*errors, *perturbation_reasons]))
                # A hook/cleanup failure makes otherwise valid typed evidence
                # incomplete. Retain the raw report, but do not publish its
                # records as capacity-valid streams.
                report_streams = ()
                raw_reports[profiler_id] = _json_value(report.raw_report)
            else:
                status = report.status
                reasons = report.reasons
                report_streams = report.streams
                streams.extend(report_streams)
                for artifact in report.artifacts:
                    existing = artifacts.get(artifact.artifact_id)
                    if existing is None:
                        artifacts[artifact.artifact_id] = artifact
                    elif existing != artifact:
                        raise CoordinatorValidationError(
                            "profilers declared conflicting raw artifacts with ID " f"{artifact.artifact_id!r}"
                        )
                clocks.extend(report.clocks)
                mappings.extend(report.clock_mappings)
                raw_reports[profiler_id] = _json_value(report.raw_report)
            manifests.append(
                ProfilerManifest(
                    profiler_id=profiler_id,
                    profiler_version=descriptor.version,
                    status=status,
                    stream_ids=tuple(stream.stream_id for stream in report_streams),
                    reasons=reasons,
                    intrusive=descriptor.intrusive,
                    provenance={
                        **(_json_value(report.provenance) if report is not None else {}),
                        "requirements": list(descriptor.requirements),
                        "conflicts": list(descriptor.conflicts),
                        "intrusive_reason": descriptor.intrusive_reason,
                    },
                    configuration=_secret_safe_projection(item.configuration),
                    extensions={
                        **(_json_value(report.extensions) if report is not None else {}),
                        "measurement_domains": list(descriptor.measurement_domains),
                        "measurement_perturbations": list(perturbations),
                        # Retain the original public key while making its
                        # domain semantics explicit for new consumers.
                        "invalidates_measurement_domains": list(descriptor.invalidates_metrics),
                        "invalidates_metrics": list(descriptor.invalidates_metrics),
                    },
                )
            )

        verification = runtime_report.get("verification")
        verification_failed = (
            isinstance(verification, Mapping)
            and verification.get("requested") is True
            and verification.get("status") != "passed"
        )
        run_invalid = bool(failures) or verification_failed
        evidence_bundle = EvidenceBundle(
            run_id=run_id,
            scope=scope,
            timeline=timeline,
            clocks=tuple(clocks),
            clock_mappings=tuple(mappings),
            profilers=tuple(manifests),
            streams=tuple(streams),
            artifacts=tuple(artifacts.values()),
            provenance={
                "execution_plan_sha256": plan.canonical_digest,
                "execution_plan": plan.to_manifest(),
                "profile_request": request.to_manifest(),
            },
            extensions={
                "coordination_status": "invalid" if failures else "valid",
                "coordination_failures": [failure.to_dict() for failure in failures],
                "raw_profiler_reports": raw_reports,
                "runtime_report": _json_value(runtime_report),
                "run_status": "invalid" if run_invalid else "valid",
                "workload_verification": _json_value(verification),
            },
        )
        intrusive_ids = tuple(item.descriptor.profiler_id for item in resolved if item.descriptor.intrusive)
        return CoordinationResult(
            run_id=run_id,
            execution_plan_digest=plan.canonical_digest,
            status="invalid" if run_invalid else "valid",
            resolved_profiler_ids=tuple(item.descriptor.profiler_id for item in resolved),
            intrusive_profiler_ids=intrusive_ids,
            scope=scope,
            timeline=timeline,
            manifests=tuple(manifests),
            raw_reports=raw_reports,
            runtime_report=_json_value(runtime_report),
            failures=tuple(failures),
            evidence_bundle=evidence_bundle,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest
from offline_replayer.measurement.coordinator import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    CoordinatorValidationError,
    ExecutionPlan,
    MountBehavior,
    ProfileCoordinator,
    ProfilerContext,
    ProfilerDescriptor,
    ProfileRequest,
    ProfilerRegistry,
    ProfilerReport,
    ProfileRunError,
)
from offline_replayer.measurement.evidence import (
    RECORD_SHAPES,
    EvidenceRecord,
    EvidenceStream,
    LifecycleMarker,
    MeasuredScope,
    read_bundle,
    write_bundle,
)
from offline_replayer.reporting.unified_view import (
    SHAPE_REGISTRY,
    assemble_view_model,
    write_unified_view_from_bundle,
)

RUN_ID = "12345678-1234-5678-9234-567812345678"


class FakeClock:
    def __init__(self, *, start_ns: int = 1_000_000_000, step_ns: int = 100) -> None:
        self.next_ns = start_ns
        self.step_ns = step_ns
        self.calls = 0

    def __call__(self) -> int:
        value = self.next_ns
        self.next_ns += self.step_ns
        self.calls += 1
        return value


class FakeSession:
    def __init__(
        self,
        *,
        capabilities: frozenset[str] = frozenset({"cgroup-v2", "perf"}),
        poll_results: list[bool] | None = None,
        finish_poll_results: list[bool] | None = None,
        activity_log: list[str] | None = None,
        active_profiler_ids: set[str] | None = None,
        verification: Mapping[str, Any] | None = None,
    ) -> None:
        self._capabilities = capabilities
        self.poll_results = list(poll_results or [False, True])
        self.finish_poll_results = list(finish_poll_results or [False, True])
        self.activity_log = activity_log if activity_log is not None else []
        self.active_profiler_ids = active_profiler_ids if active_profiler_ids is not None else set()
        self.drain_active_snapshots: list[frozenset[str]] = []
        self.abort_active_snapshot: frozenset[str] | None = None
        self.events: list[tuple[str, Any]] = []
        self.verification = verification
        self.launch_count = 0
        self.abort_count = 0
        self.close_count = 0
        self.scope = MeasuredScope(
            scope_id="docker-cgroup:fixture",
            kind="disposable-container-cgroup-subtree",
            identity={"container_id": "fixture", "cgroup_inode": 42},
        )

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    def launch(
        self,
        plan: ExecutionPlan,
        *,
        run_id: str,
        marker: LifecycleMarker,
    ) -> MeasuredScope:
        self.launch_count += 1
        self.events.append(("launch", run_id, marker, plan.canonical_digest))
        return self.scope

    def release_sandbox(self, marker: LifecycleMarker) -> None:
        self.events.append(("release_sandbox", marker))

    def wait_command_ready(self) -> None:
        self.events.append(("wait_command_ready",))

    def release_command(self, marker: LifecycleMarker) -> None:
        self.events.append(("release_command", marker))

    def poll_command(self, timeout_sec: float) -> bool:
        self.events.append(("poll_command", timeout_sec))
        self.activity_log.append("session.poll_command")
        return self.poll_results.pop(0)

    def begin_finish(self, marker: LifecycleMarker) -> None:
        self.events.append(("begin_finish", marker))
        self.activity_log.append("session.begin_finish")

    def poll_finish(self, timeout_sec: float) -> bool:
        self.events.append(("poll_finish", timeout_sec))
        self.activity_log.append("session.poll_finish")
        self.drain_active_snapshots.append(frozenset(self.active_profiler_ids))
        return self.finish_poll_results.pop(0)

    def finish(self, marker: LifecycleMarker) -> Mapping[str, Any]:
        self.events.append(("finish", marker))
        self.activity_log.append("session.finish")
        report: dict[str, Any] = {"exit_code": 0, "outcome_valid": True}
        if self.verification is not None:
            report["verification"] = dict(self.verification)
        return report

    def abort(self, reason: str) -> Mapping[str, Any]:
        self.abort_count += 1
        self.events.append(("abort", reason))
        self.activity_log.append("session.abort")
        self.abort_active_snapshot = frozenset(self.active_profiler_ids)
        return {"aborted": True, "reason": reason}

    def close(self) -> None:
        self.close_count += 1
        self.events.append(("close",))


class FakeProfiler:
    def __init__(
        self,
        profiler_id: str,
        *,
        global_events: list[tuple[str, str]] | None = None,
        fail_hook: str | None = None,
        configuration: Mapping[str, Any] | None = None,
        activity_log: list[str] | None = None,
        active_profiler_ids: set[str] | None = None,
    ) -> None:
        self.profiler_id = profiler_id
        self.global_events = global_events if global_events is not None else []
        self.fail_hook = fail_hook
        self.configuration = configuration or {}
        self.activity_log = activity_log if activity_log is not None else []
        self.active_profiler_ids = active_profiler_ids if active_profiler_ids is not None else set()
        self.hooks: list[str] = []
        self.contexts: list[ProfilerContext] = []
        self.start_marker: LifecycleMarker | None = None
        self.command_end_marker: LifecycleMarker | None = None
        self.stop_marker: LifecycleMarker | None = None

    def _hook(self, name: str, context: ProfilerContext | None = None) -> None:
        self.hooks.append(name)
        self.global_events.append((name, self.profiler_id))
        self.activity_log.append(f"{self.profiler_id}.{name}")
        if context is not None:
            self.contexts.append(context)
        if self.fail_hook == name:
            raise RuntimeError(f"{self.profiler_id} failed during {name}")

    def attach(self, context: ProfilerContext) -> None:
        self._hook("attach", context)

    def start(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        self.start_marker = marker
        self._hook("start", context)
        self.active_profiler_ids.add(self.profiler_id)

    def poll(self, context: ProfilerContext) -> None:
        self._hook("poll", context)

    def command_end(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        self.command_end_marker = marker
        self._hook("command_end", context)

    def stop(self, context: ProfilerContext, marker: LifecycleMarker) -> None:
        self.stop_marker = marker
        self._hook("stop", context)
        self.active_profiler_ids.discard(self.profiler_id)

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        self._hook("finalize", context)
        return ProfilerReport(
            status="valid",
            raw_report={
                "profiler_id": self.profiler_id,
                "configuration": dict(self.configuration),
                "run_id": context.run_id,
                "scope_id": context.scope.scope_id,
                "markers": [marker.name for marker in context.lifecycle.markers],
            },
        )

    def close(self) -> None:
        self._hook("close")


class SixShapeThirdPartyProfiler(FakeProfiler):
    """Test-only plugin proving the public typed-stream extension contract."""

    def __init__(self) -> None:
        super().__init__("third-party-six-shape")

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        self._hook("finalize", context)
        measured_start = context.lifecycle.get("measured_work_start")
        measured_end = context.lifecycle.get("measured_work_end")
        assert measured_start is not None and measured_end is not None
        start_ns = measured_start.canonical_timestamp_ns
        end_ns = measured_end.canonical_timestamp_ns
        assert end_ns > start_ns

        def record(
            shape: str,
            value: object,
        ) -> EvidenceRecord:
            fields: dict[str, object] = {
                "record_id": f"third-party:{shape}",
                "run_id": context.run_id,
                "scope_id": context.scope.scope_id,
                "profiler_id": self.profiler_id,
                "stream_id": f"third-party-{shape}",
                "shape": shape,
                "status": "valid",
                "value": value,
            }
            if shape in {"point_event", "sampled_gauge", "cumulative_counter"}:
                fields.update(
                    source_clock_id=context.canonical_clock_id,
                    native_timestamp_ns=start_ns,
                    canonical_timestamp_ns=start_ns,
                    alignment_error_ns=0,
                )
            elif shape in {"interval", "window_aggregate"}:
                fields.update(
                    source_clock_id=context.canonical_clock_id,
                    native_start_ns=start_ns,
                    native_end_ns=end_ns,
                    canonical_start_ns=start_ns,
                    canonical_end_ns=end_ns,
                    alignment_error_ns=0,
                )
            return EvidenceRecord(**fields)  # type: ignore[arg-type]

        values: dict[str, object] = {
            "scalar": 7,
            "point_event": {"event": "third-party-event"},
            "interval": {"activity": "third-party-interval"},
            "sampled_gauge": 11.5,
            "cumulative_counter": 19,
            "window_aggregate": {"count": 23},
        }
        streams = tuple(
            EvidenceStream(
                stream_id=f"third-party-{shape}",
                profiler_id=self.profiler_id,
                metric=f"third_party.{shape}",
                shape=shape,
                unit="1",
                status="valid",
                records=(record(shape, values[shape]),),
                display={
                    "group": "Third-party fixture",
                    "label": shape.replace("_", " "),
                },
            )
            for shape in sorted(RECORD_SHAPES)
        )
        return ProfilerReport(
            status="valid",
            streams=streams,
            raw_report={
                "profiler_id": self.profiler_id,
                "shape_count": len(streams),
            },
            provenance={"implementation": "test-only-registered-profiler"},
        )


class DomainScalarProfiler(FakeProfiler):
    """Small third-party profiler used to exercise domain invalidation."""

    def finalize(self, context: ProfilerContext) -> ProfilerReport:
        self._hook("finalize", context)
        stream_id = f"{self.profiler_id}-scalar"
        record = EvidenceRecord(
            record_id=f"{stream_id}:value",
            run_id=context.run_id,
            scope_id=context.scope.scope_id,
            profiler_id=self.profiler_id,
            stream_id=stream_id,
            shape="scalar",
            status="valid",
            value=1,
        )
        return ProfilerReport(
            status="valid",
            streams=(
                EvidenceStream(
                    stream_id=stream_id,
                    profiler_id=self.profiler_id,
                    metric=f"fixture.{self.profiler_id}",
                    shape="scalar",
                    unit="1",
                    status="valid",
                    records=(record,),
                ),
            ),
            raw_report={"profiler_id": self.profiler_id},
        )


def _plan(*, environment: Mapping[str, str] | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        workload_identity={"task": "fixture", "variant": "serial-parallel"},
        source_identity={"kind": "ATIF", "sha256": "a" * 64},
        command=("/bin/sh", "-c", "./fixture-workload"),
        cwd="/workspace",
        mounts=(
            MountBehavior(
                target="/workspace",
                read_only=False,
                source_identity={"tree_sha256": "b" * 64},
            ),
        ),
        image_reference="fixture:tag",
        image_identity={"id": "sha256:" + "c" * 64},
        runtime="docker",
        network="none",
        expected_outcome={"exit_code": 0, "source_outcome_mismatches": 0},
        environment=environment or {},
    )


def _coordinator(
    registry: ProfilerRegistry,
    session: FakeSession,
    clock: FakeClock,
    *,
    cancel_requested=None,
) -> ProfileCoordinator:
    return ProfileCoordinator(
        registry=registry,
        session_factory=lambda: session,
        clock_ns=clock,
        run_id_factory=lambda: RUN_ID,
        cancel_requested=cancel_requested,
        poll_interval_sec=0.01,
    )


def test_coordinator_launches_once_and_broadcasts_exact_shared_lifecycle() -> None:
    global_events: list[tuple[str, str]] = []
    activity_log: list[str] = []
    active_profiler_ids: set[str] = set()
    early = FakeProfiler(
        "early",
        global_events=global_events,
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    late = FakeProfiler(
        "late",
        global_events=global_events,
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    registry = ProfilerRegistry(profile_sets={})
    registry.register(
        ProfilerDescriptor("late", "1", requirements=("cgroup-v2",), order=20),
        lambda _configuration: late,
    )
    registry.register(
        ProfilerDescriptor("early", "1", requirements=("cgroup-v2",), order=10),
        lambda _configuration: early,
    )
    session = FakeSession(
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    clock = FakeClock()

    result = _coordinator(registry, session, clock).run(_plan(), ProfileRequest(profilers=("late", "early")))

    assert session.launch_count == 1
    assert result.status == "valid"
    assert result.resolved_profiler_ids == ("early", "late")
    assert result.scope is session.scope
    assert result.run_id == RUN_ID
    assert clock.calls == 8
    assert [marker.name for marker in result.timeline.lifecycle_markers] == [
        "run_start",
        "sandbox_ready",
        "sandbox_start",
        "command_ready",
        "measured_work_start",
        "command_end",
        "measured_work_end",
        "run_end",
    ]
    assert early.start_marker is late.start_marker
    assert early.command_end_marker is not None
    assert early.stop_marker is not None
    assert early.command_end_marker is late.command_end_marker
    assert early.stop_marker is late.stop_marker
    assert early.command_end_marker is not early.stop_marker
    assert early.command_end_marker.canonical_timestamp_ns < early.stop_marker.canonical_timestamp_ns
    assert {id(context) for context in early.contexts + late.contexts} == {id(early.contexts[0])}
    assert early.contexts[0] is late.contexts[0]
    assert early.contexts[0].scope is session.scope
    assert list(global_events) == [
        ("attach", "early"),
        ("attach", "late"),
        ("start", "early"),
        ("start", "late"),
        ("poll", "early"),
        ("poll", "late"),
        ("poll", "early"),
        ("poll", "late"),
        ("command_end", "early"),
        ("command_end", "late"),
        ("poll", "early"),
        ("poll", "late"),
        ("poll", "early"),
        ("poll", "late"),
        ("stop", "late"),
        ("stop", "early"),
        ("finalize", "early"),
        ("finalize", "late"),
        ("close", "late"),
        ("close", "early"),
    ]
    assert session.drain_active_snapshots == [
        frozenset({"early", "late"}),
        frozenset({"early", "late"}),
    ]
    drain_start = activity_log.index("session.begin_finish")
    first_stop = activity_log.index("late.stop")
    assert activity_log[drain_start:first_stop] == [
        "session.begin_finish",
        "session.poll_finish",
        "early.poll",
        "late.poll",
        "session.poll_finish",
        "early.poll",
        "late.poll",
    ]
    assert first_stop < activity_log.index("session.finish")
    begin_finish = next(event for event in session.events if event[0] == "begin_finish")
    finish = next(event for event in session.events if event[0] == "finish")
    assert begin_finish[1] is early.command_end_marker
    assert finish[1] is early.stop_marker
    assert result.runtime_report == {"exit_code": 0, "outcome_valid": True}
    assert [manifest.profiler_id for manifest in result.manifests] == [
        "early",
        "late",
    ]
    assert result.evidence_bundle.run_id == RUN_ID


@pytest.mark.parametrize(
    ("verification_status", "expected_run_status"),
    (("passed", "valid"), ("failed", "invalid")),
)
def test_verification_runs_after_profiler_stop_and_invalidates_failures(
    verification_status: str,
    expected_run_status: str,
) -> None:
    activity_log: list[str] = []
    profiler = FakeProfiler("observer", activity_log=activity_log)
    registry = ProfilerRegistry(profile_sets={})
    registry.register(ProfilerDescriptor("observer", "1"), lambda _configuration: profiler)
    session = FakeSession(
        activity_log=activity_log,
        verification={
            "requested": True,
            "status": verification_status,
            "valid": verification_status == "passed",
            "exit_code": 0 if verification_status == "passed" else 7,
            "reasons": ([] if verification_status == "passed" else ["verification command exited with status 7"]),
        },
    )
    plan = replace(_plan(), extensions={"verification_command_sha256": "a" * 64})

    result = _coordinator(registry, session, FakeClock()).run(plan, ProfileRequest(profilers=("observer",)))

    assert result.status == expected_run_status
    assert result.evidence_bundle.extensions["run_status"] == expected_run_status
    assert [marker.name for marker in result.timeline.lifecycle_markers][-4:] == [
        "measured_work_end",
        "verification_start",
        "verification_end",
        "run_end",
    ]
    finish_marker = next(event[1] for event in session.events if event[0] == "finish")
    assert finish_marker.name == "verification_start"
    assert activity_log.index("observer.stop") < activity_log.index("session.finish")


def test_profile_sets_resolve_deterministically_and_conflicts_fail_prelaunch() -> None:
    registry = ProfilerRegistry()
    descriptors = (
        ProfilerDescriptor("cgroup-memory", "1", order=30),
        ProfilerDescriptor("cgroup-cpu", "1", order=10),
        ProfilerDescriptor("scheduler-concurrency", "1", order=20),
        ProfilerDescriptor("process-exec", "1", order=40),
        ProfilerDescriptor("hardware-perf", "1", order=50),
    )
    for descriptor in reversed(descriptors):
        registry.register(
            descriptor,
            lambda _configuration, profiler_id=descriptor.profiler_id: FakeProfiler(profiler_id),
        )

    resolved = registry.resolve(
        ProfileRequest(
            profile_sets=("software-deep", "cpu-compare"),
            profilers=("hardware-perf", "cgroup-cpu"),
            configuration={"cgroup-memory": {"interval_sec": 0.1}},
        )
    )

    assert [item.descriptor.profiler_id for item in resolved] == [
        "cgroup-cpu",
        "scheduler-concurrency",
        "cgroup-memory",
        "process-exec",
        "hardware-perf",
    ]
    assert resolved[2].configuration == {"interval_sec": 0.1}

    conflict_registry = ProfilerRegistry(profile_sets={"both": ("left", "right")})
    conflict_registry.register(
        ProfilerDescriptor("left", "1", conflicts=("right",)),
        lambda _configuration: FakeProfiler("left"),
    )
    conflict_registry.register(
        ProfilerDescriptor("right", "1"),
        lambda _configuration: FakeProfiler("right"),
    )
    with pytest.raises(CoordinatorValidationError, match="conflicts with.*right"):
        conflict_registry.resolve(ProfileRequest(profile_sets=("both",)))


def test_explicit_baseline_uses_full_lifecycle_with_zero_profilers() -> None:
    registry = ProfilerRegistry(profile_sets={})
    session = FakeSession()

    result = _coordinator(registry, session, FakeClock()).run(_plan(), ProfileRequest(mode="baseline"))

    assert session.launch_count == 1
    assert result.status == "valid"
    assert result.resolved_profiler_ids == ()
    assert result.manifests == ()
    assert result.evidence_bundle.profilers == ()
    assert result.evidence_bundle.streams == ()
    assert result.evidence_bundle.provenance["profile_request"] == {
        "schema_version": "tasktrace-profile-request.v1",
        "mode": "baseline",
        "profile_sets": [],
        "profilers": [],
        "configuration": {},
    }
    assert [marker.name for marker in result.timeline.lifecycle_markers] == [
        "run_start",
        "sandbox_ready",
        "sandbox_start",
        "command_ready",
        "measured_work_start",
        "command_end",
        "measured_work_end",
        "run_end",
    ]

    with pytest.raises(CoordinatorValidationError, match="must select"):
        ProfileRequest()
    with pytest.raises(CoordinatorValidationError, match="cannot select"):
        ProfileRequest(mode="baseline", profilers=("unexpected",))


def test_failing_profiler_aborts_once_and_closes_every_resource() -> None:
    global_events: list[tuple[str, str]] = []
    activity_log: list[str] = []
    active_profiler_ids: set[str] = set()
    good = FakeProfiler(
        "good",
        global_events=global_events,
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    failing = FakeProfiler(
        "failing",
        global_events=global_events,
        fail_hook="poll",
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    registry = ProfilerRegistry(profile_sets={})
    registry.register(ProfilerDescriptor("good", "1", order=10), lambda _configuration: good)
    registry.register(
        ProfilerDescriptor("failing", "1", order=20),
        lambda _configuration: failing,
    )
    session = FakeSession(
        poll_results=[False],
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    clock = FakeClock()

    with pytest.raises(ProfileRunError) as captured:
        _coordinator(registry, session, clock).run(_plan(), ProfileRequest(profilers=("failing", "good")))

    result = captured.value.result
    assert result is not None
    assert result.status == "invalid"
    assert session.launch_count == 1
    assert session.abort_count == 1
    assert session.close_count == 1
    assert session.abort_active_snapshot == frozenset({"good", "failing"})
    assert activity_log.index("session.abort") < activity_log.index("good.command_end")
    assert activity_log.index("session.abort") < activity_log.index("failing.stop")
    assert good.command_end_marker is failing.command_end_marker
    assert good.stop_marker is failing.stop_marker
    assert "stop" in good.hooks and "stop" in failing.hooks
    assert good.hooks[-1] == "close" and failing.hooks[-1] == "close"
    assert result.failures[0].phase == "poll"
    assert result.failures[0].profiler_id == "failing"
    assert all(manifest.status == "invalid" for manifest in result.manifests)
    assert result.raw_reports["good"]["run_id"] == RUN_ID
    assert result.raw_reports["failing"]["run_id"] == RUN_ID
    assert [marker.name for marker in result.timeline.lifecycle_markers][-3:] == [
        "command_end",
        "measured_work_end",
        "run_end",
    ]


def test_cooperative_cancellation_aborts_active_runtime_and_closes_resources() -> None:
    activity_log: list[str] = []
    active_profiler_ids: set[str] = set()
    profiler = FakeProfiler(
        "observer",
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    registry = ProfilerRegistry(profile_sets={})
    registry.register(ProfilerDescriptor("observer", "1"), lambda _configuration: profiler)
    session = FakeSession(
        poll_results=[False, True],
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )

    def cancel_requested() -> bool:
        return any(event[0] == "poll_command" for event in session.events)

    with pytest.raises(ProfileRunError) as captured:
        _coordinator(
            registry,
            session,
            FakeClock(),
            cancel_requested=cancel_requested,
        ).run(_plan(), ProfileRequest(profilers=("observer",)))

    result = captured.value.result
    assert result is not None
    assert result.status == "invalid"
    assert result.failures[0].phase == "cancel_during_command"
    assert "ProfileCancellationError: profile cancellation requested" in (result.failures[0].message)
    assert session.abort_count == 1
    assert session.close_count == 1
    assert session.abort_active_snapshot == frozenset({"observer"})
    assert sum(event[0] == "poll_command" for event in session.events) == 1
    assert "command_end" in profiler.hooks
    assert "stop" in profiler.hooks
    assert profiler.hooks[-1] == "close"


def test_abort_before_command_release_notifies_only_started_profilers() -> None:
    activity_log: list[str] = []
    active_profiler_ids: set[str] = set()
    good = FakeProfiler(
        "good",
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    failing = FakeProfiler(
        "failing",
        fail_hook="start",
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )
    registry = ProfilerRegistry(profile_sets={})
    registry.register(ProfilerDescriptor("good", "1", order=10), lambda _configuration: good)
    registry.register(
        ProfilerDescriptor("failing", "1", order=20),
        lambda _configuration: failing,
    )
    session = FakeSession(
        activity_log=activity_log,
        active_profiler_ids=active_profiler_ids,
    )

    with pytest.raises(ProfileRunError) as captured:
        _coordinator(registry, session, FakeClock()).run(_plan(), ProfileRequest(profilers=("failing", "good")))

    result = captured.value.result
    assert result is not None
    assert result.failures[0].phase == "start"
    assert result.failures[0].profiler_id == "failing"
    assert session.abort_active_snapshot == frozenset({"good"})
    assert "command_end" not in good.hooks
    assert "command_end" not in failing.hooks
    assert "stop" in good.hooks
    assert "stop" not in failing.hooks
    assert activity_log.index("session.abort") < activity_log.index("good.stop")
    assert [marker.name for marker in result.timeline.lifecycle_markers] == [
        "run_start",
        "sandbox_ready",
        "sandbox_start",
        "command_ready",
        "measured_work_start",
        "measured_work_end",
        "run_end",
    ]


def test_intrusive_effects_are_structured_in_result_and_manifest() -> None:
    profiler = FakeProfiler("process-exec")
    registry = ProfilerRegistry(profile_sets={})
    registry.register(
        ProfilerDescriptor(
            "process-exec",
            "1",
            intrusive=True,
            intrusive_reason="ptrace changes process execution cost",
            invalidates_metrics=("capacity.cpu", "capacity.memory"),
        ),
        lambda _configuration: profiler,
    )
    result = _coordinator(registry, FakeSession(), FakeClock()).run(
        _plan(), ProfileRequest(profilers=("process-exec",))
    )

    manifest = result.manifests[0]
    assert result.intrusive_profiler_ids == ("process-exec",)
    assert manifest.intrusive is True
    assert manifest.provenance["intrusive_reason"] == ("ptrace changes process execution cost")
    assert manifest.extensions["invalidates_metrics"] == [
        "capacity.cpu",
        "capacity.memory",
    ]
    assert manifest.extensions["invalidates_measurement_domains"] == [
        "capacity.cpu",
        "capacity.memory",
    ]
    assert result.evidence_bundle.profilers[0] == manifest


def test_third_party_intrusive_profiler_invalidates_every_matching_domain() -> None:
    registry = ProfilerRegistry(profile_sets={})
    descriptors = (
        ProfilerDescriptor(
            "third-party-cpu",
            "1",
            measurement_domains=("cpu.capacity",),
        ),
        ProfilerDescriptor(
            "third-party-memory",
            "1",
            measurement_domains=("memory.capacity",),
        ),
        ProfilerDescriptor(
            "third-party-process",
            "1",
            measurement_domains=("software.process",),
        ),
        ProfilerDescriptor(
            "third-party-intruder",
            "1",
            measurement_domains=("software.process",),
            intrusive=True,
            intrusive_reason="fixture instrumentation changes capacity demand",
            invalidates_metrics=("cpu.capacity", "memory.capacity"),
        ),
    )
    for descriptor in descriptors:
        registry.register(
            descriptor,
            lambda _configuration, profiler_id=descriptor.profiler_id: (DomainScalarProfiler(profiler_id)),
        )

    result = _coordinator(registry, FakeSession(), FakeClock()).run(
        _plan(),
        ProfileRequest(profilers=tuple(item.profiler_id for item in descriptors)),
    )

    manifests = {item.profiler_id: item for item in result.manifests}
    streams = {item.profiler_id: item for item in result.evidence_bundle.streams}
    for profiler_id, domain in (
        ("third-party-cpu", "cpu.capacity"),
        ("third-party-memory", "memory.capacity"),
    ):
        manifest = manifests[profiler_id]
        stream = streams[profiler_id]
        assert manifest.status == "invalid"
        assert stream.status == "invalid"
        assert stream.records[0].status == "invalid"
        perturbation = manifest.extensions["measurement_perturbations"][0]
        assert perturbation["code"] == ("intrusive-profiler-invalidated-measurement-domain")
        assert perturbation["invalidating_profiler_id"] == "third-party-intruder"
        assert perturbation["measurement_domains"] == [domain]
        assert stream.extensions["measurement_perturbations"] == [perturbation]
        assert stream.records[0].attributes["measurement_perturbations"] == [perturbation]

    # The invalidation is a domain join, not a blanket invalidation of every
    # profiler selected in the same run.
    for profiler_id in ("third-party-process", "third-party-intruder"):
        assert manifests[profiler_id].status == "valid"
        assert streams[profiler_id].status == "valid"
        assert streams[profiler_id].records[0].status == "valid"
        assert manifests[profiler_id].extensions["measurement_perturbations"] == []


def test_cpu_domain_without_intrusive_profiler_remains_valid() -> None:
    registry = ProfilerRegistry(profile_sets={})
    registry.register(
        ProfilerDescriptor(
            "third-party-cpu",
            "1",
            measurement_domains=("cpu.capacity",),
        ),
        lambda _configuration: DomainScalarProfiler("third-party-cpu"),
    )

    result = _coordinator(registry, FakeSession(), FakeClock()).run(
        _plan(), ProfileRequest(profilers=("third-party-cpu",))
    )

    assert result.manifests[0].status == "valid"
    assert result.manifests[0].extensions["measurement_domains"] == ["cpu.capacity"]
    assert result.manifests[0].extensions["measurement_perturbations"] == []
    assert result.evidence_bundle.streams[0].status == "valid"
    assert result.evidence_bundle.streams[0].records[0].status == "valid"


def test_execution_plan_digest_is_deterministic_immutable_and_secret_safe() -> None:
    first = _plan(environment={"PUBLIC_MODE": "test", "MODEL_API_TOKEN": "super-secret"})
    second = _plan(environment={"MODEL_API_TOKEN": "super-secret", "PUBLIC_MODE": "test"})
    changed = replace(
        first,
        environment={"PUBLIC_MODE": "test", "MODEL_API_TOKEN": "other-secret"},
    )

    manifest_json = json.dumps(first.to_manifest(), sort_keys=True)
    assert first.schema_version == EXECUTION_PLAN_SCHEMA_VERSION
    assert first.canonical_digest == second.canonical_digest
    assert first.canonical_digest != changed.canonical_digest
    assert "super-secret" not in manifest_json
    assert "super-secret" not in repr(first)
    assert "MODEL_API_TOKEN" in manifest_json
    assert len(first.to_manifest()["environment"][0]["value_sha256"]) == 64
    with pytest.raises(TypeError):
        first.environment["MODEL_API_TOKEN"] = "leak"  # type: ignore[index]
    with pytest.raises(TypeError):
        first.source_identity["sha256"] = "d" * 64  # type: ignore[index]


def test_new_profiler_needs_only_registration_not_coordinator_changes() -> None:
    configurations: list[Mapping[str, Any]] = []
    profiler = FakeProfiler("third-party")
    registry = ProfilerRegistry(profile_sets={})

    def factory(configuration: Mapping[str, Any]) -> FakeProfiler:
        configurations.append(configuration)
        return profiler

    registry.register(
        ProfilerDescriptor("third-party", "7.3", requirements=("cgroup-v2",)),
        factory,
    )
    result = _coordinator(registry, FakeSession(), FakeClock()).run(
        _plan(),
        ProfileRequest(
            profilers=("third-party",),
            configuration={"third-party": {"sample_every": 3}},
        ),
    )

    assert configurations == [{"sample_every": 3}]
    assert result.resolved_profiler_ids == ("third-party",)
    assert result.manifests[0].profiler_id == "third-party"
    assert result.manifests[0].profiler_version == "7.3"
    assert result.raw_reports["third-party"]["profiler_id"] == "third-party"


def test_registered_third_party_six_shape_profiler_reaches_generic_renderer(
    tmp_path: Path,
) -> None:
    registry = ProfilerRegistry(profile_sets={})
    registry.register(
        ProfilerDescriptor(
            "third-party-six-shape",
            "1.0",
            requirements=("cgroup-v2",),
        ),
        lambda _configuration: SixShapeThirdPartyProfiler(),
    )
    session = FakeSession()

    result = _coordinator(registry, session, FakeClock()).run(
        _plan(),
        ProfileRequest(profilers=("third-party-six-shape",)),
    )

    assert session.launch_count == 1
    assert result.resolved_profiler_ids == ("third-party-six-shape",)
    assert result.manifests[0].status == "valid"
    assert result.raw_reports["third-party-six-shape"]["shape_count"] == 6
    assert {stream.shape for stream in result.evidence_bundle.streams} == set(RECORD_SHAPES)
    assert all(
        record.run_id == result.run_id and record.scope_id == result.scope.scope_id
        for stream in result.evidence_bundle.streams
        for record in stream.records
    )

    bundle_path = tmp_path / "evidence_bundle.json"
    write_bundle(result.evidence_bundle, bundle_path)
    retained_bundle = read_bundle(bundle_path)
    assert retained_bundle.to_dict() == result.evidence_bundle.to_dict()

    model = assemble_view_model(retained_bundle)
    rendered_streams = model["summaries"] + model["tracks"]
    assert {stream["shape"] for stream in rendered_streams} == set(SHAPE_REGISTRY)
    assert {stream["profiler_id"] for stream in rendered_streams} == {"third-party-six-shape"}
    assert {stream["render_kind"] for stream in rendered_streams} == {
        definition["render_kind"] for definition in SHAPE_REGISTRY.values()
    }
    rendered = write_unified_view_from_bundle(bundle_path, tmp_path / "rendered")
    for name in ("html", "svg", "animated_svg"):
        output = Path(rendered[name]).read_text()
        assert "third-party-six-shape" in output
        for shape in RECORD_SHAPES:
            assert f"third-party-{shape}" in output


def test_missing_runtime_requirement_closes_without_launching() -> None:
    registry = ProfilerRegistry(profile_sets={})
    registry.register(
        ProfilerDescriptor("needs-pmu", "1", requirements=("pmu",)),
        lambda _configuration: FakeProfiler("needs-pmu"),
    )
    session = FakeSession(capabilities=frozenset({"cgroup-v2"}))

    with pytest.raises(CoordinatorValidationError, match="needs-pmu: pmu"):
        _coordinator(registry, session, FakeClock()).run(_plan(), ProfileRequest(profilers=("needs-pmu",)))

    assert session.launch_count == 0
    assert session.close_count == 1

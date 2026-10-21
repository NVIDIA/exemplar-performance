# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.measurement.cgroup_resources import (
    AncestorCpuObserver,
    CgroupResourceSampler,
    CgroupResourceSamplingError,
    parse_cpu_max,
    parse_flat_keyed,
    parse_pressure,
)


class FakeClock:
    def __init__(self) -> None:
        self.now_ns = 1_000_000_000

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += round(seconds * 1_000_000_000)


def _write(path: Path, name: str, value: str | int) -> None:
    (path / name).write_text(f"{value}\n")


def _make_cgroup(path: Path) -> None:
    _write(path, "memory.current", 100)
    _write(path, "memory.peak", 120)
    _write(path, "memory.swap.current", 0)
    _write(path, "memory.swap.peak", 0)
    _write(path, "memory.high", "max")
    _write(path, "memory.max", "max")
    _write(path, "memory.swap.high", "max")
    _write(path, "memory.swap.max", "max")
    _write(path, "memory.stat", "anon 60\nfile 30\nkernel 10\nfuture_key 7")
    _write(path, "cpu.max", "max 100000")
    _write(
        path,
        "cpu.stat",
        "usage_usec 1000\nuser_usec 800\nsystem_usec 200\n" "nr_periods 10\nnr_throttled 0\nthrottled_usec 0",
    )
    _write(
        path,
        "memory.events",
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0",
    )
    _write(
        path,
        "memory.events.local",
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0",
    )
    _write(
        path,
        "memory.pressure",
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=10\n" "full avg10=0.00 avg60=0.00 avg300=0.00 total=2",
    )


def test_parse_flat_keyed_preserves_unknown_keys() -> None:
    assert parse_flat_keyed("anon 10\nfuture 20\n", source="memory.stat") == {
        "anon": 10,
        "future": 20,
    }


def test_parse_flat_keyed_rejects_malformed_or_negative_values() -> None:
    with pytest.raises(CgroupResourceSamplingError, match="invalid memory.stat row"):
        parse_flat_keyed("anon 1 extra\n", source="memory.stat")
    with pytest.raises(CgroupResourceSamplingError, match="negative"):
        parse_flat_keyed("anon -1\n", source="memory.stat")


def test_parse_pressure_preserves_numeric_fields() -> None:
    parsed = parse_pressure("some avg10=1.25 total=42\n")

    assert parsed == {"some": {"avg10": 1.25, "total": 42}}


def test_parse_cpu_max_preserves_quota_period_and_state() -> None:
    assert parse_cpu_max("max 100000\n") == (None, 100000, "unlimited")
    assert parse_cpu_max("200000 100000\n") == (200000, 100000, "finite")


@pytest.mark.parametrize(
    "value",
    ["max", "max nope", "0 100000", "100000 0", "-1 100000", "1 -2"],
)
def test_parse_cpu_max_rejects_malformed_or_nonpositive_values(value: str) -> None:
    with pytest.raises(CgroupResourceSamplingError):
        parse_cpu_max(value)


def test_sampler_records_memory_and_aligned_interval_cpu(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=tmp_path,
        interval_sec=0.1,
        clock_ns=clock,
    )

    sampler.start()
    clock.advance(0.05)
    sampler.mark("command_start")
    _write(tmp_path, "memory.current", 500)
    _write(tmp_path, "memory.peak", 500)
    _write(tmp_path, "memory.swap.current", 0)
    _write(tmp_path, "memory.stat", "anon 450\nfile 30\nkernel 20")
    _write(tmp_path, "cpu.stat", "usage_usec 101000\nuser_usec 90800\nsystem_usec 10200")
    clock.advance(0.1)
    assert sampler.poll() is True
    clock.advance(0.1)
    sampler.mark("command_end")
    _write(tmp_path, "memory.current", 200)
    _write(tmp_path, "cpu.stat", "usage_usec 151000\nuser_usec 120800\nsystem_usec 30200")
    clock.advance(0.05)
    report = sampler.finish()

    assert report["instrumentation_valid"] is True
    assert report["peak_valid"] is True
    assert report["natural_peak_trustworthy"] is True
    assert report["observation_boundary"] == "controlled-finish-gate"
    assert report["observation_boundary_complete"] is True
    assert report["full_cgroup_lifetime_observed"] is False
    assert report["lifetime"]["exact_through_controlled_finish_gate"] is True
    assert report["lifetime"]["exact_for_fresh_cgroup_lifetime"] is False
    assert report["lifetime"]["peak_bytes"] == 500
    assert report["observed_limit_margin"] == {
        "limit_scope": "leaf-cgroup-memory.max-at-observer-start",
        "configured_limit_state": "unlimited",
        "configured_limit_bytes": None,
        "observed_peak_bytes": 500,
        "remaining_to_limit_bytes": None,
        "peak_fraction_of_limit": None,
        "arithmetic_valid": False,
        "meaning": report["observed_limit_margin"]["meaning"],
    }
    assert report["observation_window"]["sampled_peak_bytes"] == 500
    assert report["command_window"]["sampled_peak_bytes"] == 500
    assert report["command_window"]["sampled_peak_swap_bytes"] == 0
    assert report["command_window"]["memory_stat_at_sampled_peak"]["anon"] == 450
    assert report["command_window"]["sample_count"] >= 2
    assert report["command_window"]["sampled_cpu_usage_sec"] == 0.1
    assert report["command_window"]["sampled_cpu_average_logical_cores"] == 0.5
    assert report["observation_window"]["sampled_cpu_usage_sec"] == 0.15
    assert [marker["name"] for marker in report["markers"]] == [
        "sandbox_observation_start",
        "command_start",
        "command_end",
        "sandbox_observation_end",
    ]

    rows = sampler.timeseries_payload()["samples"]
    assert rows[0]["cpu_average_logical_cores"] is None
    assert any(
        isinstance(row["cpu_average_logical_cores"], float) and row["cpu_average_logical_cores"] > 0 for row in rows[1:]
    )
    assert all(rows[index]["elapsed_sec"] <= rows[index + 1]["elapsed_sec"] for index in range(len(rows) - 1))

    raw_sources = sampler.raw_memory_source_snapshot_payload()
    assert raw_sources["schema_version"] == ("container-cgroup-memory-source-snapshots.v1")
    assert raw_sources["scope"]["leaf_path"] == "."
    assert report["raw_source_snapshot_read_count"] == len(raw_sources["reads"])


def test_sampler_reports_factual_finite_leaf_limit_margin(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    _write(tmp_path, "memory.max", 1000)
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=tmp_path,
        interval_sec=0.1,
        clock_ns=clock,
    )

    sampler.start()
    _write(tmp_path, "memory.current", 500)
    _write(tmp_path, "memory.peak", 500)
    clock.advance(0.1)
    sampler.poll()
    report = sampler.finish()

    assert report["observed_limit_margin"] == {
        "limit_scope": "leaf-cgroup-memory.max-at-observer-start",
        "configured_limit_state": "finite",
        "configured_limit_bytes": 1000,
        "observed_peak_bytes": 500,
        "remaining_to_limit_bytes": 500,
        "peak_fraction_of_limit": 0.5,
        "arithmetic_valid": True,
        "meaning": report["observed_limit_margin"]["meaning"],
    }
    assert report["natural_peak_trustworthy"] is True
    raw_sources = sampler.raw_memory_source_snapshot_payload()
    assert {row["phase"] for row in raw_sources["reads"]} == {
        "observer_start",
        "controlled_finish_gate",
    }
    finish_peak_reads = [
        row
        for row in raw_sources["reads"]
        if row["phase"] == "controlled_finish_gate"
        and row["source_scope"] == "leaf"
        and row["filename"] == "memory.peak"
    ]
    assert [row["raw_text"] for row in finish_peak_reads] == ["500\n"]
    for filename in (
        "memory.high",
        "memory.max",
        "memory.swap.high",
        "memory.swap.max",
        "memory.events",
        "memory.events.local",
        "memory.pressure",
        "memory.stat",
    ):
        assert any(row["filename"] == filename and row["available"] is True for row in raw_sources["reads"])


def test_sampler_reports_missed_deadlines_and_terminal_sample(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=tmp_path,
        interval_sec=0.1,
        clock_ns=clock,
    )
    sampler.start()

    clock.advance(0.35)
    assert sampler.poll() is True
    clock.advance(0.01)
    report = sampler.finish()

    assert sampler.missed_sample_deadlines == 2
    assert sampler.timeseries_payload()["samples"][-1]["elapsed_sec"] == 0.36
    assert report["timeline_valid_for_shape"] is False
    assert "exceeded twice" in report["timeline_invalid_reasons"][0]


def test_sampler_retains_peak_and_marks_natural_demand_untrustworthy_after_oom(
    tmp_path: Path,
) -> None:
    _make_cgroup(tmp_path)
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=tmp_path, clock_ns=clock)
    sampler.start()
    _write(
        tmp_path,
        "memory.events",
        "low 0\nhigh 0\nmax 2\noom 1\noom_kill 1\noom_group_kill 0",
    )
    clock.advance(0.1)

    report = sampler.finish()

    assert report["censoring"] == "oom"
    assert report["events_delta"]["oom_kill"] == 1
    assert report["instrumentation_valid"] is True
    assert report["lifetime"]["peak_bytes"] == 120
    assert report["peak_valid"] is True
    assert report["natural_peak_trustworthy"] is False


def test_sampler_requires_explicit_policy_when_swap_was_used(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=tmp_path, clock_ns=clock)
    sampler.start()
    _write(tmp_path, "memory.swap.current", 4096)
    _write(tmp_path, "memory.swap.peak", 4096)
    clock.advance(0.1)

    report = sampler.finish()

    assert report["censoring"] == "swap_activity"
    assert report["natural_peak_trustworthy"] is False


def test_sampler_treats_optional_files_as_unavailable_not_zero(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    (tmp_path / "memory.peak").unlink()
    (tmp_path / "memory.swap.current").unlink()
    (tmp_path / "memory.swap.peak").unlink()
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=tmp_path, clock_ns=clock)
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()

    assert report["lifetime"]["peak_bytes"] is None
    assert report["swap"]["lifetime_peak_bytes"] is None
    assert report["observation_window"]["sampled_peak_swap_bytes"] is None
    assert report["instrumentation_valid"] is True
    assert report["peak_valid"] is False
    assert report["natural_peak_trustworthy"] is False
    assert "memory.peak is unavailable" in report["natural_peak_invalid_reasons"]


def test_sampler_requires_memory_events_for_trustworthy_natural_peak(
    tmp_path: Path,
) -> None:
    _make_cgroup(tmp_path)
    (tmp_path / "memory.events").unlink()
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=tmp_path, clock_ns=clock)
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()

    assert report["capabilities"]["memory_events_at_observer_start"] is False
    assert report["instrumentation_valid"] is False
    assert report["natural_peak_trustworthy"] is False
    assert any("memory.events" in reason for reason in report["instrumentation_invalid_reasons"])


def test_sampler_requires_oom_group_kill_event_capability(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    _write(tmp_path, "memory.events", "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0")
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=tmp_path, clock_ns=clock)
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()

    assert report["instrumentation_valid"] is False
    assert report["natural_peak_trustworthy"] is False
    assert any("memory.events" in reason for reason in report["instrumentation_invalid_reasons"])


def test_sampler_distinguishes_unlimited_and_unavailable_swap_limits(
    tmp_path: Path,
) -> None:
    _make_cgroup(tmp_path)
    (tmp_path / "memory.swap.max").unlink()
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=tmp_path, clock_ns=clock)
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()

    assert report["limits"]["memory_max_state"] == "unlimited"
    assert report["limits"]["swap_max_state"] == "unavailable"
    assert report["capabilities"]["memory_swap_limit"] is False
    assert report["natural_peak_trustworthy"] is False
    assert any("swap activity" in reason for reason in report["natural_peak_invalid_reasons"])


def test_sampler_detects_backwards_event_counter(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    _write(tmp_path, "memory.events", "low 2\nhigh 0\nmax 0\noom 0\noom_kill 0")
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=tmp_path, clock_ns=clock)
    sampler.start()
    _write(tmp_path, "memory.events", "low 1\nhigh 0\nmax 0\noom 0\noom_kill 0")
    clock.advance(0.1)

    report = sampler.finish()

    assert report["instrumentation_valid"] is False
    assert any("moved backwards" in reason for reason in report["instrumentation_invalid_reasons"])


def test_sampler_writes_stable_timeseries_artifact(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    _make_cgroup(cgroup)
    clock = FakeClock()
    sampler = CgroupResourceSampler(cgroup_path=cgroup, clock_ns=clock)
    sampler.start()
    clock.advance(0.1)
    sampler.finish()
    output = tmp_path / "samples.json"

    sampler.write_timeseries(output)

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "container-cgroup-resource-timeseries.v1"
    assert payload["sample_interval_sec"] == 0.1
    assert len(payload["samples"]) == 2

    raw_output = tmp_path / "memory-sources.json"
    sampler.write_raw_memory_source_snapshots(raw_output)
    raw_payload = json.loads(raw_output.read_text())
    assert raw_payload == sampler.raw_memory_source_snapshot_payload()
    assert raw_payload["reads"][-1]["read_index"] == len(raw_payload["reads"]) - 1


def test_sampler_rejects_invalid_state_transitions(tmp_path: Path) -> None:
    _make_cgroup(tmp_path)
    sampler = CgroupResourceSampler(cgroup_path=tmp_path)

    with pytest.raises(CgroupResourceSamplingError, match="not active"):
        sampler.poll()
    sampler.start()
    with pytest.raises(CgroupResourceSamplingError, match="already started"):
        sampler.start()
    sampler.finish()
    with pytest.raises(CgroupResourceSamplingError, match="not active"):
        sampler.poll()


def test_sampler_rejects_nonpositive_interval(tmp_path: Path) -> None:
    with pytest.raises(CgroupResourceSamplingError, match="positive"):
        CgroupResourceSampler(cgroup_path=tmp_path, interval_sec=0)


def _memory_events(**updates: int) -> str:
    values = {
        "low": 0,
        "high": 0,
        "max": 0,
        "oom": 0,
        "oom_kill": 0,
        "oom_group_kill": 0,
    }
    values.update(updates)
    return "\n".join(f"{key} {value}" for key, value in values.items())


def _cpu_stat(**updates: int) -> str:
    values = {
        "usage_usec": 1000,
        "user_usec": 800,
        "system_usec": 200,
        "nr_periods": 10,
        "nr_throttled": 0,
        "throttled_usec": 0,
    }
    values.update(updates)
    return "\n".join(f"{key} {value}" for key, value in values.items())


def _cgroup_hierarchy(tmp_path: Path, *parts: str) -> tuple[Path, list[Path]]:
    root = tmp_path / "cgroup-root"
    paths = [root]
    for part in parts:
        paths.append(paths[-1] / part)
    for path in paths:
        path.mkdir()
        _make_cgroup(path)
    return root, paths


def test_ancestor_cpu_observer_is_reusable_without_memory_sampling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cgroup-root"
    parent = root / "job"
    leaf = parent / "container.scope"
    leaf.mkdir(parents=True)
    for path in (root, parent):
        _write(path, "cpu.max", "max 100000")
        _write(path, "cpu.stat", _cpu_stat())
    clock = FakeClock()
    observer = AncestorCpuObserver(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )

    observer.start()
    _write(parent, "cpu.stat", _cpu_stat(usage_usec=2000, nr_periods=11))
    clock.advance(0.1)
    report = observer.finish()

    assert report["valid_for_cpu_performance"] is True
    assert [item["path"] for item in report["ancestors"]] == ["job", "."]
    assert report["ancestors"][0]["cpu_stat_delta"]["usage_usec"] == 1000
    raw = observer.raw_source_snapshot_payload()
    assert raw["schema_version"] == ("container-cgroup-ancestor-cpu-source-snapshots.v1")
    assert len(raw["reads"]) == 8
    output = tmp_path / "ancestor-cpu-sources.json"
    observer.write_raw_source_snapshots(output)
    assert json.loads(output.read_text()) == raw


def test_ancestor_cpu_observer_rejects_incomplete_or_repeated_gates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cgroup-root"
    parent = root / "job"
    leaf = parent / "container.scope"
    leaf.mkdir(parents=True)
    for path in (root, parent):
        _write(path, "cpu.max", "max 100000")
        _write(path, "cpu.stat", _cpu_stat())
    observer = AncestorCpuObserver(cgroup_path=leaf, cgroup_root=root)

    with pytest.raises(CgroupResourceSamplingError, match="not active"):
        observer.finish()
    with pytest.raises(CgroupResourceSamplingError, match="incomplete"):
        observer.raw_source_snapshot_payload()
    observer.start()
    with pytest.raises(CgroupResourceSamplingError, match="already started"):
        observer.start()
    observer.finish()
    with pytest.raises(CgroupResourceSamplingError, match="not active"):
        observer.finish()


def test_ancestor_cpu_captures_every_parent_and_raw_sources(tmp_path: Path) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    _write(
        parent,
        "cpu.stat",
        _cpu_stat(
            usage_usec=5000,
            user_usec=4000,
            system_usec=1000,
            nr_periods=11,
        ),
    )
    clock.advance(0.1)

    report = sampler.finish()

    ancestor_cpu = report["ancestor_cpu"]
    assert ancestor_cpu["evidence_complete"] is True
    assert ancestor_cpu["limits_stable"] is True
    assert ancestor_cpu["finite_limits_present"] is False
    assert ancestor_cpu["throttling_observed"] is False
    assert ancestor_cpu["valid_for_cpu_performance"] is True
    assert [row["path"] for row in ancestor_cpu["ancestors"]] == ["parent", "."]
    parent_evidence = ancestor_cpu["ancestors"][0]
    assert parent_evidence["start"]["cpu_max_state"] == "unlimited"
    assert parent_evidence["start"]["cpu_max_period_usec"] == 100000
    assert parent_evidence["cpu_stat_delta"]["usage_usec"] == 4000
    assert parent_evidence["cpu_stat_delta"]["nr_periods"] == 1
    assert parent_evidence["throttle_deltas"] == {}
    assert report["capabilities"]["ancestor_cpu_evidence_complete"] is True

    raw = sampler.raw_memory_source_snapshot_payload()
    parent_cpu_reads = [
        row
        for row in raw["ancestor_cpu"]["reads"]
        if row["source_scope"] == "ancestor:parent" and row["filename"] in {"cpu.max", "cpu.stat"}
    ]
    assert raw["controllers"] == ["cpu", "memory"]
    assert [(row["phase"], row["filename"]) for row in parent_cpu_reads] == [
        ("observer_start", "cpu.max"),
        ("observer_start", "cpu.stat"),
        ("controlled_finish_gate", "cpu.max"),
        ("controlled_finish_gate", "cpu.stat"),
    ]
    assert parent_cpu_reads[0]["raw_text"] == "max 100000\n"
    assert "usage_usec 5000" in parent_cpu_reads[-1]["raw_text"]


def test_sampler_can_delegate_ancestor_cpu_to_mandatory_cpu_profiler(
    tmp_path: Path,
) -> None:
    root, (_root, _parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
        observe_ancestor_cpu=False,
    )

    sampler.start()
    clock.advance(0.1)
    report = sampler.finish()
    raw = sampler.raw_memory_source_snapshot_payload()

    assert report["ancestor_cpu_observer"] == ("disabled-owned-by-mandatory-cgroup-cpu-profiler")
    assert "ancestor_cpu" not in report
    assert "ancestor_cpu_evidence_complete" not in report["capabilities"]
    assert raw["controllers"] == ["memory"]
    assert "ancestor_cpu" not in raw


def test_stable_finite_ancestor_cpu_limit_is_retained_but_not_unconstrained(
    tmp_path: Path,
) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    _write(parent, "cpu.max", "200000 100000")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()

    ancestor_cpu = report["ancestor_cpu"]
    parent_evidence = ancestor_cpu["ancestors"][0]
    assert parent_evidence["start"]["cpu_max_state"] == "finite"
    assert parent_evidence["start"]["cpu_max_quota_usec"] == 200000
    assert parent_evidence["start"]["cpu_max_period_usec"] == 100000
    assert parent_evidence["cpu_max_changed"] is False
    assert parent_evidence["evidence_valid"] is True
    assert parent_evidence["valid_for_cpu_performance"] is False
    assert ancestor_cpu["evidence_complete"] is True
    assert ancestor_cpu["finite_limits_present"] is True
    assert ancestor_cpu["valid_for_cpu_performance"] is False
    assert any("finite cpu.max" in reason for reason in ancestor_cpu["invalid_reasons"])
    # CPU policy evidence does not redefine natural-memory-demand trust.
    assert report["natural_peak_trustworthy"] is True


@pytest.mark.parametrize("changed_limit", ["200000 100000", "max 200000"])
def test_ancestor_cpu_limit_change_fails_closed(tmp_path: Path, changed_limit: str) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    _write(parent, "cpu.max", changed_limit)
    clock.advance(0.1)

    ancestor_cpu = sampler.finish()["ancestor_cpu"]

    assert ancestor_cpu["limits_stable"] is False
    assert ancestor_cpu["evidence_complete"] is False
    assert ancestor_cpu["valid_for_cpu_performance"] is False
    assert ancestor_cpu["ancestors"][0]["cpu_max_changed"] is True
    assert any("cpu.max changed" in reason for reason in ancestor_cpu["invalid_reasons"])


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"nr_periods": 11, "nr_throttled": 1}, {"nr_throttled": 1}),
        (
            {"nr_periods": 11, "throttled_usec": 250},
            {"throttled_usec": 250},
        ),
        (
            {"nr_periods": 11, "nr_throttled": 1, "throttled_usec": 250},
            {"nr_throttled": 1, "throttled_usec": 250},
        ),
    ],
)
def test_positive_ancestor_cpu_throttling_fails_closed(
    tmp_path: Path,
    updates: dict[str, int],
    expected: dict[str, int],
) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    _write(parent, "cpu.stat", _cpu_stat(usage_usec=2000, **updates))
    clock.advance(0.1)

    ancestor_cpu = sampler.finish()["ancestor_cpu"]

    assert ancestor_cpu["evidence_complete"] is True
    assert ancestor_cpu["throttling_observed"] is True
    assert ancestor_cpu["valid_for_cpu_performance"] is False
    assert ancestor_cpu["ancestors"][0]["throttle_deltas"] == expected
    assert any("CPU throttling" in reason for reason in ancestor_cpu["invalid_reasons"])


def test_ancestor_cpu_counter_regression_fails_closed(tmp_path: Path) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    _write(parent, "cpu.stat", _cpu_stat(usage_usec=999))
    clock.advance(0.1)

    ancestor_cpu = sampler.finish()["ancestor_cpu"]

    assert ancestor_cpu["counter_regression_observed"] is True
    assert ancestor_cpu["evidence_complete"] is False
    assert ancestor_cpu["valid_for_cpu_performance"] is False
    assert "usage_usec" not in ancestor_cpu["ancestors"][0]["cpu_stat_delta"]
    assert any("moved backwards" in reason for reason in ancestor_cpu["invalid_reasons"])


@pytest.mark.parametrize("source", ["cpu.max", "cpu.stat"])
def test_missing_or_malformed_ancestor_cpu_source_fails_closed(tmp_path: Path, source: str) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    if source == "cpu.max":
        _write(parent, source, "not-a-limit")
    else:
        _write(parent, source, "usage_usec not-an-integer")
    clock.advance(0.1)

    report = sampler.finish()
    ancestor_cpu = report["ancestor_cpu"]

    assert ancestor_cpu["evidence_complete"] is False
    assert ancestor_cpu["valid_for_cpu_performance"] is False
    assert report["natural_peak_trustworthy"] is True
    assert any(source in reason for reason in ancestor_cpu["invalid_reasons"])


@pytest.mark.parametrize("source", ["cpu.max", "cpu.stat"])
def test_unavailable_ancestor_cpu_source_is_retained_and_fails_closed(tmp_path: Path, source: str) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    (parent / source).unlink()
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()
    ancestor_cpu = report["ancestor_cpu"]

    assert ancestor_cpu["evidence_complete"] is False
    assert ancestor_cpu["valid_for_cpu_performance"] is False
    missing_reads = [
        row
        for row in sampler.raw_memory_source_snapshot_payload()["ancestor_cpu"]["reads"]
        if row["source_scope"] == "ancestor:parent" and row["filename"] == source
    ]
    assert [row["available"] for row in missing_reads] == [False, False]
    assert all(row["raw_text"] is None for row in missing_reads)


def test_parent_memory_high_event_invalidates_clean_leaf(tmp_path: Path) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    _write(parent, "memory.events.local", _memory_events(high=1))
    clock.advance(0.1)

    report = sampler.finish()

    ancestor = report["ancestor_memory"]
    assert report["events_delta"]["high"] == 0
    assert report["instrumentation_valid"] is True
    assert report["natural_peak_trustworthy"] is False
    assert report["lifetime"]["peak_bytes"] == 120
    assert ancestor["constraint_events_observed"] is True
    assert ancestor["natural_demand_untrustworthy"] is True
    assert ancestor["ancestors"][0]["path"] == "parent"
    assert ancestor["ancestors"][0]["constraint_event_deltas"] == {"high": 1}
    assert any("attribution is ambiguous" in reason for reason in ancestor["invalid_reasons"])


@pytest.mark.parametrize(
    ("events", "expected_keys"),
    [
        ({"max": 2}, {"max"}),
        ({"oom": 1, "oom_kill": 1}, {"oom", "oom_kill"}),
    ],
)
def test_parent_max_or_oom_event_invalidates_natural_demand(
    tmp_path: Path,
    events: dict[str, int],
    expected_keys: set[str],
) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    _write(parent, "memory.events.local", _memory_events(**events))
    clock.advance(0.1)

    report = sampler.finish()

    observed = report["ancestor_memory"]["ancestors"][0]["constraint_event_deltas"]
    assert set(observed) == expected_keys
    assert report["instrumentation_valid"] is True
    assert report["natural_peak_trustworthy"] is False
    assert report["lifetime"]["peak_bytes"] == 120


def test_finite_unused_parent_limits_are_recorded_without_invalidating(
    tmp_path: Path,
) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    _write(parent, "memory.high", 512)
    _write(parent, "memory.max", 1024)
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()

    ancestor = report["ancestor_memory"]
    parent_evidence = ancestor["ancestors"][0]
    assert parent_evidence["start"]["memory_high_bytes"] == 512
    assert parent_evidence["start"]["memory_max_bytes"] == 1024
    assert parent_evidence["finite_limits_present"] is True
    assert parent_evidence["constraint_event_deltas"] == {}
    assert parent_evidence["positive_pressure_total_usec"] == {}
    assert ancestor["finite_limits_present"] is True
    assert ancestor["natural_demand_trustworthy"] is True
    assert report["natural_peak_trustworthy"] is True


def test_positive_parent_psi_is_attribution_ambiguous_and_invalidates(
    tmp_path: Path,
) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    _write(
        parent,
        "memory.pressure",
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=25\n" "full avg10=0.00 avg60=0.00 avg300=0.00 total=2",
    )
    clock.advance(0.1)

    report = sampler.finish()

    ancestor = report["ancestor_memory"]
    assert ancestor["constraint_events_observed"] is False
    assert ancestor["pressure_observed"] is True
    assert ancestor["attribution_ambiguous"] is True
    assert ancestor["ancestors"][0]["positive_pressure_total_usec"] == {"some": 15}
    assert report["instrumentation_valid"] is True
    assert report["natural_peak_trustworthy"] is False
    assert report["lifetime"]["peak_bytes"] == 120
    assert any("sibling cgroups" in reason for reason in report["natural_peak_invalid_reasons"])


@pytest.mark.parametrize("malformed", [True, False])
def test_malformed_or_backwards_ancestor_counters_fail_closed(
    tmp_path: Path,
    malformed: bool,
) -> None:
    root, (_root, parent, leaf) = _cgroup_hierarchy(tmp_path, "parent", "leaf")
    if not malformed:
        _write(parent, "memory.events.local", _memory_events(high=2))
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    if malformed:
        _write(parent, "memory.events.local", "high not-an-integer")
    else:
        _write(parent, "memory.events.local", _memory_events(high=1))
    clock.advance(0.1)

    report = sampler.finish()

    ancestor = report["ancestor_memory"]
    assert ancestor["evidence_complete"] is False
    assert ancestor["natural_demand_untrustworthy"] is True
    assert report["instrumentation_valid"] is False
    assert report["natural_peak_trustworthy"] is False
    expected = "invalid" if malformed else "moved backwards"
    assert any(expected in reason for reason in report["instrumentation_invalid_reasons"])


def test_ancestor_chain_is_ordered_from_leaf_parent_through_root(
    tmp_path: Path,
) -> None:
    root, (_root, _team, _job, leaf) = _cgroup_hierarchy(tmp_path, "team", "job", "leaf")
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )
    sampler.start()
    clock.advance(0.1)

    report = sampler.finish()

    ancestor = report["ancestor_memory"]
    assert ancestor["leaf_path"] == "team/job/leaf"
    assert ancestor["chain_order"] == "leaf-parent-to-root"
    assert [item["path"] for item in ancestor["ancestors"]] == [
        "team/job",
        "team",
        ".",
    ]
    assert ancestor["natural_demand_trustworthy"] is True


def test_unconstrainable_cgroup_v2_root_without_memory_files_is_not_an_error(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cgroup-root"
    parent = root / "system.slice"
    leaf = parent / "container.scope"
    leaf.mkdir(parents=True)
    _make_cgroup(parent)
    _make_cgroup(leaf)
    _write(root, "cpu.stat", _cpu_stat())
    clock = FakeClock()
    sampler = CgroupResourceSampler(
        cgroup_path=leaf,
        cgroup_root=root,
        clock_ns=clock,
    )

    sampler.start()
    clock.advance(0.1)
    report = sampler.finish()

    ancestor = report["ancestor_memory"]
    assert ancestor["cgroup_root_memory_controller_files_available"] is False
    assert [item["path"] for item in ancestor["ancestors"]] == ["system.slice"]
    assert ancestor["evidence_complete"] is True
    assert ancestor["natural_demand_trustworthy"] is True
    assert report["natural_peak_trustworthy"] is True
    ancestor_cpu = report["ancestor_cpu"]
    assert [item["path"] for item in ancestor_cpu["ancestors"]] == [
        "system.slice",
        ".",
    ]
    assert ancestor_cpu["ancestors"][-1]["start"]["cpu_max_state"] == ("not-applicable-root")
    assert ancestor_cpu["evidence_complete"] is True
    assert ancestor_cpu["valid_for_cpu_performance"] is True


def test_samples_expose_serial_acquisition_bounds_and_uncertainty(
    tmp_path: Path,
) -> None:
    _make_cgroup(tmp_path)
    timestamps = iter((1_000_000_000, 1_000_025_000))
    sampler = CgroupResourceSampler(
        cgroup_path=tmp_path,
        clock_ns=lambda: next(timestamps),
    )

    assert sampler.start() == 1_000_000_000
    payload = sampler.timeseries_payload()
    row = payload["samples"][0]

    assert row["timestamp_ns"] == 1_000_000_000
    assert row["elapsed_sec"] == 0.0
    assert row["acquisition_start_ns"] == row["timestamp_ns"]
    assert row["acquisition_end_ns"] == 1_000_025_000
    assert row["acquisition_duration_sec"] == 0.000025
    assert row["timestamp_alignment_uncertainty_sec"] == 0.000025
    assert payload["timestamp_semantics"] == "acquisition-start"
    assert payload["max_sample_acquisition_duration_sec"] == 0.000025

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from unittest.mock import patch

from offline_replayer.timebase import (
    boottime_clock_reference,
    bounded_realtime_monotonic_sample,
    monotonic_clock_reference,
    monotonic_mapping_from_references,
)


def test_bounded_realtime_monotonic_sample_retains_uncertainty() -> None:
    realtime_before_ns = time.time_ns()
    monotonic_before_ns = time.monotonic_ns()
    sample = bounded_realtime_monotonic_sample()
    monotonic_after_ns = time.monotonic_ns()
    realtime_after_ns = time.time_ns()

    assert monotonic_before_ns <= sample["monotonic_ns"] <= monotonic_after_ns
    assert realtime_before_ns <= sample["realtime_ns"] <= realtime_after_ns
    assert 0 <= sample["realtime_uncertainty_ns"] <= (realtime_after_ns - realtime_before_ns)


def test_bounded_realtime_monotonic_sample_retains_backward_clock_step() -> None:
    with (
        patch("offline_replayer.timebase.time.time_ns", side_effect=[16, 11]),
        patch("offline_replayer.timebase.time.monotonic_ns", return_value=99),
    ):
        sample = bounded_realtime_monotonic_sample()

    assert sample == {
        "monotonic_ns": 99,
        "realtime_ns": 13,
        "realtime_uncertainty_ns": -5,
    }


def test_bounded_realtime_monotonic_sample_midpoint_handles_u64_limits() -> None:
    with (
        patch(
            "offline_replayer.timebase.time.time_ns",
            side_effect=[(1 << 64) - 1, 0],
        ),
        patch("offline_replayer.timebase.time.monotonic_ns", return_value=99),
    ):
        sample = bounded_realtime_monotonic_sample()

    assert sample["realtime_ns"] == ((1 << 64) - 1) // 2
    assert sample["realtime_uncertainty_ns"] == -((1 << 64) - 1)


def test_boottime_clock_reference_retains_host_join_evidence() -> None:
    reference = boottime_clock_reference()

    assert reference["clock_id"] == "CLOCK_BOOTTIME"
    assert reference["clock_name"] == "CLOCK_BOOTTIME"
    assert reference["timestamp_unit"] == "nanoseconds"
    assert reference["resolution_ns"] > 0
    assert reference["storage_precision_ns"] == 1
    assert reference["accuracy_claim"] is None


def test_monotonic_clock_reference_includes_join_identity_without_accuracy_claim() -> None:
    reference = monotonic_clock_reference()

    assert reference["clock_id"] == "CLOCK_MONOTONIC"
    assert reference["timestamp_unit"] == "nanoseconds"
    assert reference["storage_precision_ns"] == 1
    assert reference["accuracy_claim"] is None
    offset = reference["time_namespace_monotonic_offset_ns"]
    assert offset is None or isinstance(offset, int)
    if reference["time_namespace"] is not None:
        assert reference["time_namespace"].startswith("time:[")
    if reference["boot_id"] is not None:
        assert len(reference["boot_id"]) == 36


def test_monotonic_mapping_uses_time_namespace_offsets_on_same_boot() -> None:
    mapping = monotonic_mapping_from_references(
        {
            "boot_id": "same-boot",
            "time_namespace": "time:[2]",
            "time_namespace_monotonic_offset_ns": 2_000_000_003,
        },
        {
            "boot_id": "same-boot",
            "time_namespace": "time:[1]",
            "time_namespace_monotonic_offset_ns": -4,
        },
    )

    assert mapping["status"] == "valid"
    assert mapping["method"] == "linux-time-namespace-offset"
    assert mapping["offset_ns"] == -2_000_000_007
    assert mapping["max_error_ns"] == 0


def test_monotonic_mapping_accepts_distinct_namespaces_with_equal_zero_offsets() -> None:
    mapping = monotonic_mapping_from_references(
        {
            "boot_id": "same-boot",
            "time_namespace": "time:[container]",
            "time_namespace_monotonic_offset_ns": 0,
        },
        {
            "boot_id": "same-boot",
            "time_namespace": "time:[host]",
            "time_namespace_monotonic_offset_ns": 0,
        },
    )

    assert mapping["status"] == "valid"
    assert mapping["method"] == "linux-time-namespace-offset"
    assert mapping["offset_ns"] == 0
    assert mapping["provenance"]["source_time_namespace"] == "time:[container]"
    assert mapping["provenance"]["canonical_time_namespace"] == "time:[host]"


def test_monotonic_mapping_fails_closed_across_boots() -> None:
    mapping = monotonic_mapping_from_references(
        {
            "boot_id": "source-boot",
            "time_namespace": "time:[1]",
            "time_namespace_monotonic_offset_ns": 0,
        },
        {
            "boot_id": "canonical-boot",
            "time_namespace": "time:[1]",
            "time_namespace_monotonic_offset_ns": 0,
        },
    )

    assert mapping["status"] == "unalignable"
    assert "offset_ns" not in mapping

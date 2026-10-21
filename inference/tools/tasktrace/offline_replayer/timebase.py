# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clock identity helpers shared by replay and profiling evidence."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any


def bounded_realtime_monotonic_sample() -> dict[str, int]:
    """Sample both clocks with an explicit CLOCK_REALTIME uncertainty bound.

    The monotonic read is bracketed by two realtime reads.  The realtime
    midpoint can therefore relate nearby realtime evidence to a replay action
    without pretending the clock reads were simultaneous. The full bracket
    width is retained as a conservative error bound.
    """
    realtime_before_ns = time.time_ns()
    monotonic_ns = time.monotonic_ns()
    realtime_after_ns = time.time_ns()
    return {
        "monotonic_ns": monotonic_ns,
        "realtime_ns": realtime_before_ns + (realtime_after_ns - realtime_before_ns) // 2,
        "realtime_uncertainty_ns": realtime_after_ns - realtime_before_ns,
    }


def _time_namespace_offsets(path: Path = Path("/proc/self/timens_offsets")) -> dict[str, int] | None:
    """Return Linux time-namespace offsets as signed nanoseconds.

    The kernel defines these offsets relative to the initial time namespace.
    They therefore provide a stronger join key than namespace inode equality:
    two namespaces with the same boot ID can still have an exact affine
    monotonic-clock mapping when their offsets are known.
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    offsets: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 3 or fields[0] not in {"monotonic", "boottime"}:
            return None
        try:
            seconds = int(fields[1])
            nanoseconds = int(fields[2])
        except ValueError:
            return None
        if not 0 <= nanoseconds < 1_000_000_000:
            return None
        offsets[fields[0]] = seconds * 1_000_000_000 + nanoseconds
    return offsets if "monotonic" in offsets else None


def monotonic_clock_reference() -> dict[str, Any]:
    """Describe the process monotonic clock without claiming its accuracy.

    A clock name alone is insufficient for joining timestamps from different
    processes. Linux time-namespace identity and kernel boot identity let the
    host coordinator prove an identity mapping, or leave the stream visibly
    unalignable when they differ.
    """
    try:
        time_namespace = os.readlink("/proc/self/ns/time")
    except OSError:
        time_namespace = None
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = None
    offsets = _time_namespace_offsets()
    return {
        "clock_id": "CLOCK_MONOTONIC",
        "timestamp_unit": "nanoseconds",
        "time_namespace": time_namespace,
        "time_namespace_monotonic_offset_ns": (offsets.get("monotonic") if offsets is not None else None),
        "boot_id": boot_id,
        "storage_precision_ns": 1,
        "accuracy_claim": None,
    }


def boottime_clock_reference() -> dict[str, Any]:
    """Describe the Linux boot-time clock used by Enroot lifecycle evidence.

    ``CLOCK_BOOTTIME`` and ``/proc/uptime`` share the same suspend-aware clock
    domain.  Retaining the boot ID and time-namespace offset lets consumers
    prove whether host lifecycle timestamps and in-sandbox replay timestamps
    can be compared rather than assuming that matching clock names are enough.
    """
    clock_id = getattr(time, "CLOCK_BOOTTIME", None)
    if clock_id is None:
        raise RuntimeError("CLOCK_BOOTTIME is unavailable on this platform")
    try:
        resolution_sec = time.clock_getres(clock_id)
    except (OSError, ValueError) as exc:
        raise RuntimeError("CLOCK_BOOTTIME resolution is unavailable") from exc
    try:
        time_namespace = os.readlink("/proc/self/ns/time")
    except OSError:
        time_namespace = None
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = None
    offsets = _time_namespace_offsets()
    return {
        "clock_id": "CLOCK_BOOTTIME",
        "clock_name": "CLOCK_BOOTTIME",
        "timestamp_unit": "nanoseconds",
        "time_namespace": time_namespace,
        "time_namespace_boottime_offset_ns": (offsets.get("boottime") if offsets is not None else None),
        "boot_id": boot_id,
        "resolution_ns": max(1, math.ceil(resolution_sec * 1_000_000_000)),
        "storage_precision_ns": 1,
        "accuracy_claim": None,
    }


def monotonic_mapping_from_references(source: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    """Build an exact Linux monotonic mapping or fail visibly.

    If both references are from the same kernel boot, Linux time-namespace
    offsets define ``canonical = source + canonical_offset - source_offset``.
    No timing sample or wall-clock assumption is involved.  Older references
    that identify the same time namespace are also safe identity mappings.
    """
    source_boot = source.get("boot_id")
    canonical_boot = canonical.get("boot_id")
    source_offset = source.get("time_namespace_monotonic_offset_ns")
    canonical_offset = canonical.get("time_namespace_monotonic_offset_ns")
    common = {
        "source_clock_id": "source-monotonic",
        "canonical_clock_id": "host-monotonic",
    }
    if (
        isinstance(source_boot, str)
        and source_boot
        and source_boot == canonical_boot
        and isinstance(source_offset, int)
        and not isinstance(source_offset, bool)
        and isinstance(canonical_offset, int)
        and not isinstance(canonical_offset, bool)
    ):
        return {
            **common,
            "status": "valid",
            "method": "linux-time-namespace-offset",
            "scale_numerator": 1,
            "scale_denominator": 1,
            "offset_ns": canonical_offset - source_offset,
            "max_error_ns": 0,
            "reasons": [],
            "provenance": {
                "boot_id": source_boot,
                "source_time_namespace": source.get("time_namespace"),
                "canonical_time_namespace": canonical.get("time_namespace"),
                "source_monotonic_offset_ns": source_offset,
                "canonical_monotonic_offset_ns": canonical_offset,
            },
        }
    if (
        isinstance(source_boot, str)
        and source_boot
        and source_boot == canonical_boot
        and isinstance(source.get("time_namespace"), str)
        and source.get("time_namespace") == canonical.get("time_namespace")
    ):
        return {
            **common,
            "status": "valid",
            "method": "linux-time-namespace-identity",
            "scale_numerator": 1,
            "scale_denominator": 1,
            "offset_ns": 0,
            "max_error_ns": 0,
            "reasons": [],
            "provenance": {
                "boot_id": source_boot,
                "time_namespace": source.get("time_namespace"),
            },
        }
    return {
        **common,
        "status": "unalignable",
        "method": "linux-monotonic-reference-insufficient",
        "reasons": [
            "source and canonical monotonic clocks lack a proven common boot " "and time-namespace offset mapping"
        ],
        "provenance": {
            "source_boot_id": source_boot,
            "canonical_boot_id": canonical_boot,
            "source_time_namespace": source.get("time_namespace"),
            "canonical_time_namespace": canonical.get("time_namespace"),
        },
    }

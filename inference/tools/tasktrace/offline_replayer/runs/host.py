# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, explicit host identity for cross-machine CPU comparisons."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Sequence

from ..measurement.container import ProfileError, parse_cpuset_cpu_ids

HOST_SCHEMA_VERSION = "tasktrace-run-host.v1"
SYS_CPU_ROOT = Path("/sys/devices/system/cpu")
CPUINFO_PATH = Path("/proc/cpuinfo")


class HostIdentityError(RuntimeError):
    """Required CPU identity or topology could not be established."""


def _cpuinfo_fields(path: Path) -> dict[str, str]:
    try:
        first_block = path.read_text().split("\n\n", 1)[0]
    except OSError as exc:
        raise HostIdentityError(f"unable to read CPU identity from {path}: {exc}") from exc
    fields: dict[str, str] = {}
    for line in first_block.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().lower()] = value.strip()
    return fields


def _integer(path: Path, *, field: str) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError) as exc:
        raise HostIdentityError(f"unable to read {field} from {path}: {exc}") from exc


def _siblings(path: Path) -> tuple[int, ...]:
    try:
        raw = path.read_text().strip()
        return parse_cpuset_cpu_ids(raw)
    except (OSError, ProfileError) as exc:
        raise HostIdentityError(f"unable to read thread siblings from {path}: {exc}") from exc


def _numa_node(cpu_dir: Path) -> int | None:
    nodes = sorted(
        int(path.name.removeprefix("node"))
        for path in cpu_dir.glob("node[0-9]*")
        if path.name.removeprefix("node").isdigit()
    )
    if len(nodes) > 1:
        raise HostIdentityError(f"CPU has multiple NUMA-node links: {cpu_dir}")
    return nodes[0] if nodes else None


def collect_host_cpu_identity(
    cpu_ids: Sequence[int],
    *,
    sys_cpu_root: Path = SYS_CPU_ROOT,
    cpuinfo_path: Path = CPUINFO_PATH,
) -> dict[str, Any]:
    """Record CPU model and the exact topology of the selected logical CPUs."""
    fields = _cpuinfo_fields(cpuinfo_path)
    model = fields.get("model name") or fields.get("hardware") or fields.get("processor") or platform.processor()
    vendor = fields.get("vendor_id") or fields.get("cpu implementer")
    if not model:
        raise HostIdentityError("host CPU model is unavailable")

    selected = []
    for cpu_id in cpu_ids:
        cpu_dir = sys_cpu_root / f"cpu{cpu_id}"
        topology = cpu_dir / "topology"
        siblings = _siblings(topology / "thread_siblings_list")
        if cpu_id not in siblings:
            raise HostIdentityError(f"CPU {cpu_id} is absent from its thread-sibling topology")
        selected.append(
            {
                "logical_cpu": cpu_id,
                "package_id": _integer(topology / "physical_package_id", field="physical package ID"),
                "core_id": _integer(topology / "core_id", field="physical core ID"),
                "thread_siblings": list(siblings),
                "numa_node": _numa_node(cpu_dir),
            }
        )

    return {
        "schema_version": HOST_SCHEMA_VERSION,
        "machine": platform.machine(),
        "kernel_release": platform.release(),
        "cpu_vendor": vendor,
        "cpu_model": model,
        "microcode": fields.get("microcode"),
        "host_logical_cpu_count": os.cpu_count(),
        "selected_cpus": selected,
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from offline_replayer.runs import host
from offline_replayer.runs.host import HostIdentityError, collect_host_cpu_identity


def _cpu(root: Path, cpu_id: int, *, package: int, core: int, siblings: str) -> None:
    topology = root / f"cpu{cpu_id}" / "topology"
    topology.mkdir(parents=True)
    (topology / "physical_package_id").write_text(f"{package}\n")
    (topology / "core_id").write_text(f"{core}\n")
    (topology / "thread_siblings_list").write_text(f"{siblings}\n")


def test_collects_selected_cpu_model_and_smt_topology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor : 0\n" "vendor_id : GenuineTest\n" "model name : Test CPU 9000\n" "microcode : 0x123\n\n"
    )
    sys_cpu = tmp_path / "sys-cpu"
    _cpu(sys_cpu, 20, package=0, core=10, siblings="20,92")
    _cpu(sys_cpu, 21, package=0, core=11, siblings="21,93")
    (sys_cpu / "cpu20/node0").mkdir()
    (sys_cpu / "cpu21/node0").mkdir()
    monkeypatch.setattr(host.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host.platform, "release", lambda: "test-kernel")
    monkeypatch.setattr(host.os, "cpu_count", lambda: 192)

    result = collect_host_cpu_identity((20, 21), sys_cpu_root=sys_cpu, cpuinfo_path=cpuinfo)

    assert result == {
        "schema_version": "tasktrace-run-host.v1",
        "machine": "x86_64",
        "kernel_release": "test-kernel",
        "cpu_vendor": "GenuineTest",
        "cpu_model": "Test CPU 9000",
        "microcode": "0x123",
        "host_logical_cpu_count": 192,
        "selected_cpus": [
            {
                "logical_cpu": 20,
                "package_id": 0,
                "core_id": 10,
                "thread_siblings": [20, 92],
                "numa_node": 0,
            },
            {
                "logical_cpu": 21,
                "package_id": 0,
                "core_id": 11,
                "thread_siblings": [21, 93],
                "numa_node": 0,
            },
        ],
    }


def test_rejects_missing_selected_cpu_topology(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nmodel name : Test CPU\n\n")

    with pytest.raises(HostIdentityError, match="thread siblings"):
        collect_host_cpu_identity((20,), sys_cpu_root=tmp_path / "missing", cpuinfo_path=cpuinfo)

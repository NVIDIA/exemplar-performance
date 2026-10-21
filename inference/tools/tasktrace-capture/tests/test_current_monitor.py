# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process telemetry for --runtime current: the trial's work is this process's descendants, not processes in a
foreign mount namespace (there are none — everything shares the container). Also the CLI's concurrency rule."""

import os
import subprocess
import sys
import time

import pytest
from tasktrace_capture import monitor
from tasktrace_capture.cli import current_runtime_concurrency_error

pytestmark = pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs procfs")


@pytest.fixture
def workload_child():
    """A python child (not skipped as idle) that lives for the test; killed afterwards."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(0.2)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()


def test_descendants_are_the_workload_in_current_mode(workload_child):
    pids = {pid for pid, _ in monitor._iter_descendant_pids(skip_idle=True)}
    assert workload_child.pid in pids
    assert os.getpid() not in pids  # the coordinator itself is not the workload
    total, cats = monitor.count_enroot_procs("current")
    assert total >= 1 and cats.get("python", 0) >= 1


def test_current_container_counts_only_while_the_trial_runs(workload_child):
    assert monitor.get_n_containers("current") == 1
    workload_child.kill()
    workload_child.wait()
    assert monitor.get_n_containers("current") == 0


def test_skip_idle_hides_sleep_children():
    proc = subprocess.Popen(["sleep", "30"])
    time.sleep(0.2)
    try:
        assert proc.pid not in {pid for pid, _ in monitor._iter_descendant_pids(skip_idle=True)}
        assert proc.pid in {pid for pid, _ in monitor._iter_descendant_pids(skip_idle=False)}
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.parametrize(
    "runtime, levels, ok",
    [
        ("current", [1], True),
        ("current", None, True),
        ("current", [2], False),
        ("current", [1, 4], False),
        ("enroot", [1, 4, 16], True),
        ("docker", [8], True),
    ],
)
def test_current_runtime_concurrency_rule_checks_every_final_level(runtime, levels, ok):
    error = current_runtime_concurrency_error(runtime, levels)
    assert (error is None) is ok
    if not ok:
        assert "one at a time" in error and "--concurrency 1" in error

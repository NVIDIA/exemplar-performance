# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from offline_replayer.adapters.trace_profile import (
    TaskTraceTraceProfileConfig,
    build_container_profile_config,
    run_tasktrace_trace_profile,
)
from offline_replayer.measurement import pipeline
from offline_replayer.measurement.network_policy import NetworkMode, NetworkPolicy


def _trace(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "record_type": "trace",
                "schema_version": "tasktrace-cpu-trace.v1",
                "task_name": "coreutils-static",
                "source_trial": "coreutils-static__trial",
                "environment": {"action_timeout_sec": 30},
            }
        )
        + "\n"
        + json.dumps(
            {
                "record_type": "action",
                "action_index": 1,
                "command": "true",
                "started_at_epoch": 1.0,
                "finished_at_epoch": 1.1,
                "wall_time_sec": 0.1,
                "returncode": 0,
            }
        )
        + "\n"
    )
    return path


def test_build_container_profile_wraps_paced_trace_replay(tmp_path: Path) -> None:
    config = TaskTraceTraceProfileConfig(
        trace=_trace(tmp_path / "tasktrace_shell_commands.jsonl"),
        image="compilebench-coreutils:local",
        output_dir=tmp_path / "output",
        cpu_limit=2.0,
        cpuset_cpus=None,
        allocated_cpus=2.0,
        pacing="recorded",
        timeout_sec=None,
        prepare_command="apt-get install -y python3",
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/workdir",
        network_policy=NetworkPolicy(
            NetworkMode.ALLOWLIST,
            ("pypi.org", "files.pythonhosted.org"),
        ),
        network_policy_proxy_image="python:policy-proxy",
        network_policy_external_network="egress-test",
        coordinated_baseline=True,
    )

    profile, trace = build_container_profile_config(config)

    assert trace.header["task_name"] == "coreutils-static"
    assert profile.cpu_limit == 2.0
    assert profile.allocated_cpus == 2.0
    assert profile.coordinated_baseline is True
    assert profile.network_policy == NetworkPolicy(
        NetworkMode.ALLOWLIST,
        ("pypi.org", "files.pythonhosted.org"),
    )
    assert profile.network_policy_proxy_image == "python:policy-proxy"
    assert profile.network_policy_external_network == "egress-test"
    assert "--pacing recorded" in profile.command
    assert "--cwd /workdir" in profile.command
    assert any("dst=/tasktrace-input,readonly" in mount for mount in profile.mounts)
    assert any("dst=/tasktrace" in mount for mount in profile.mounts)
    assert profile.workload_identity["task_name"] == "coreutils-static"
    assert profile.source_identity["format"] == "trace"
    assert profile.source_identity["sha256"] == hashlib.sha256(config.trace.read_bytes()).hexdigest()


def test_build_container_profile_passes_scheduler_and_proc_options(tmp_path: Path) -> None:
    config = TaskTraceTraceProfileConfig(
        trace=_trace(tmp_path / "tasktrace_shell_commands.jsonl"),
        image="compilebench-coreutils:local",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-3",
        allocated_cpus=None,
        pacing="recorded",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/workdir",
        scheduler_concurrency=True,
        scheduler_backend="perf",
        scheduler_perf_binary="/usr/bin/perf",
        scheduler_perf_mmap_pages=2048,
        proc_stat_comparison=True,
        proc_stat_interval_sec=0.25,
        proc_stat_baseline_sec=1.0,
        memory_profile=True,
        memory_sample_interval_sec=0.05,
        memory_limit_bytes=536_870_912,
        memory_swap_total_limit_bytes=536_870_912,
        profile_sets=("cpu-compare",),
    )

    profile, _trace_result = build_container_profile_config(config)

    assert profile.scheduler_backend == "perf"
    assert profile.scheduler_perf_binary == "/usr/bin/perf"
    assert profile.scheduler_perf_mmap_pages == 2048
    assert profile.proc_stat_comparison is True
    assert profile.proc_stat_interval_sec == 0.25
    assert profile.proc_stat_baseline_sec == 1.0
    assert profile.memory_profile is True
    assert profile.memory_sample_interval_sec == 0.05
    assert profile.memory_limit_bytes == 536_870_912
    assert profile.memory_swap_total_limit_bytes == 536_870_912
    assert profile.profile_sets == ("cpu-compare",)


def test_build_container_profile_passes_enroot_runtime(tmp_path: Path) -> None:
    config = TaskTraceTraceProfileConfig(
        trace=_trace(tmp_path / "tasktrace_shell_commands.jsonl"),
        image="/images/cio.sqsh",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-7",
        allocated_cpus=None,
        pacing="recorded",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/workdir",
        runtime="enroot",
        enroot="/usr/bin/enroot",
    )

    profile, _trace_result = build_container_profile_config(config)

    assert profile.runtime == "enroot"
    assert profile.enroot == "/usr/bin/enroot"
    assert profile.image == "/images/cio.sqsh"


def test_run_profile_preserves_tasktrace_pipeline_report_contract(tmp_path: Path, monkeypatch) -> None:
    config = TaskTraceTraceProfileConfig(
        trace=_trace(tmp_path / "tasktrace_shell_commands.jsonl"),
        image="compilebench-coreutils:local",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-3",
        allocated_cpus=None,
        pacing="recorded",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/workdir",
    )

    def fake_run(_profile):
        replay = config.output_dir / "replay" / "replay_report.json"
        replay.write_text(json.dumps({"summary": {"source_outcome_mismatch_count": 0}}))
        return {
            "image_identity": {"id": "sha256:image"},
            "cpu": {"average_logical_cores_used": 1.0},
            "scheduler_concurrency": None,
            "proc_stat_comparison": None,
            "strace": {"mode": "none"},
            "warning": None,
            "status": "valid",
            "profile_provenance": {"mode": "baseline", "profile_sets": []},
        }

    monkeypatch.setattr(pipeline, "run_profile", fake_run)

    report = run_tasktrace_trace_profile(config)

    assert report == {
        "schema_version": "container-trace-profile.v1",
        "task_name": "coreutils-static",
        "source_trial": "coreutils-static__trial",
        "source_reward": None,
        "source_trace": str(config.trace.resolve()),
        "source_trace_sha256": hashlib.sha256(config.trace.read_bytes()).hexdigest(),
        "source_trace_schema_version": "tasktrace-cpu-trace.v1",
        "source_limitations": [],
        "image": "compilebench-coreutils:local",
        "image_identity": {"id": "sha256:image"},
        "pacing": "recorded",
        "replay_report": str((config.output_dir / "replay" / "replay_report.json").resolve()),
        "container_profile_report": str((config.output_dir / "container_profile" / "profile_report.json").resolve()),
        "replay_summary": {"source_outcome_mismatch_count": 0},
        "container_cpu": {"average_logical_cores_used": 1.0},
        "scheduler_concurrency": None,
        "proc_stat_comparison": None,
        "strace": {"mode": "none"},
        "warning": None,
        "status": "valid",
        "profile_provenance": {"mode": "baseline", "profile_sets": []},
    }
    assert json.loads((config.output_dir / "pipeline_report.json").read_text()) == report

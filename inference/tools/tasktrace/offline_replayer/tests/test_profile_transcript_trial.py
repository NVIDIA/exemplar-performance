# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from offline_replayer.adapters.transcript_profile import (
    TranscriptTrialProfileConfig,
    _path_tree_sha256,
    build_container_profile_config,
    run_transcript_trial_profile,
    stage_pinchbench_initial_workspace,
)
from offline_replayer.adapters.transcript_replay import ReplayError
from offline_replayer.measurement import pipeline
from offline_replayer.measurement.network_policy import NetworkMode, NetworkPolicy


def test_workspace_provenance_hash_changes_with_content_and_symlink(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "input.txt"
    source.write_text("first")
    (workspace / "alias").symlink_to("input.txt")

    before = _path_tree_sha256(workspace)
    source.write_text("second")

    assert _path_tree_sha256(workspace) != before


def _config(tmp_path: Path, transcript: Path, **overrides: object) -> TranscriptTrialProfileConfig:
    values: dict[str, object] = {
        "transcript": transcript,
        "task_id": "task",
        "trial_name": "task.1-of-3.default",
        "image": "pinchbench:local",
        "output_dir": tmp_path / "result",
        "cpuset_cpus": "0-3",
        "allocated_cpus": None,
        "source_workspace": Path("/tmp/pinchbench/run/agent_workspace"),
        "initial_workspace": None,
        "pinchbench_skill_dir": None,
        "timeout_sec": 180.0,
        "prepare_command": "apt-get install -y strace",
        "strace_mode": "execve",
        "string_limit": 4096,
        "shell": "/bin/bash",
        "workdir": "/workspace",
    }
    values.update(overrides)
    return TranscriptTrialProfileConfig(**values)  # type: ignore[arg-type]


def _write_trace(path: Path, answer: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "task_id": "task",
                "trial_name": "task.1-of-3.default",
                "answer": answer,
            }
        )
        + "\n"
    )
    return path


def test_build_container_profile_wraps_single_trial_replay(tmp_path: Path) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: exec]\n{"command": "true"}')

    profile, action_count = build_container_profile_config(
        _config(
            tmp_path,
            transcript,
            network_policy=NetworkPolicy(NetworkMode.NO_NETWORK),
            network_policy_proxy_image="python:policy-proxy",
            network_policy_external_network="egress-test",
        )
    )

    assert action_count == 1
    assert profile.image == "pinchbench:local"
    assert profile.cpuset_cpus == "0-3"
    assert profile.network_policy == NetworkPolicy(NetworkMode.NO_NETWORK)
    assert profile.network_policy_proxy_image == "python:policy-proxy"
    assert profile.network_policy_external_network == "egress-test"
    assert "--trial-name task.1-of-3.default" in profile.command
    assert "--source-workspace /tmp/pinchbench/run/agent_workspace" in profile.command
    assert any("dst=/tasktrace-source,readonly" in mount for mount in profile.mounts)
    assert any("dst=/tasktrace-input,readonly" in mount for mount in profile.mounts)
    assert any("dst=/tasktrace" in mount for mount in profile.mounts)
    assert profile.workload_identity == {
        "task_id": "task",
        "trial_name": "task.1-of-3.default",
    }
    assert profile.source_identity["format"] == "transcript-jsonl"
    assert profile.source_identity["sha256"] == hashlib.sha256(transcript.read_bytes()).hexdigest()


def test_build_container_profile_rejects_unsupported_trial_by_default(tmp_path: Path) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: browser]\n{"url": "https://example.com"}')

    with pytest.raises(ReplayError, match="unsupported tools"):
        build_container_profile_config(_config(tmp_path, transcript))


def test_build_container_profile_can_mount_initial_workspace(tmp_path: Path) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: read]\n{"path": "input.txt"}')
    initial = tmp_path / "initial"
    initial.mkdir()

    profile, _ = build_container_profile_config(_config(tmp_path, transcript, initial_workspace=initial))

    assert "--initial-workspace /tasktrace-initial-workspace" in profile.command
    assert any("dst=/tasktrace-initial-workspace,readonly" in mount for mount in profile.mounts)


def test_build_container_profile_can_enable_scheduler_concurrency(tmp_path: Path) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: exec]\n{"command": "true"}')

    profile, _ = build_container_profile_config(_config(tmp_path, transcript, scheduler_concurrency=True))

    assert profile.scheduler_concurrency is True


def test_build_container_profile_forwards_coordinated_baseline(tmp_path: Path) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: exec]\n{"command": "true"}')

    profile, _ = build_container_profile_config(_config(tmp_path, transcript, coordinated_baseline=True))

    assert profile.coordinated_baseline is True
    assert profile.profile_sets == ()


def test_build_container_profile_forwards_memory_options(tmp_path: Path) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: exec]\n{"command": "true"}')

    profile, _ = build_container_profile_config(
        _config(
            tmp_path,
            transcript,
            memory_profile=True,
            memory_sample_interval_sec=0.05,
            memory_limit_bytes=536_870_912,
            memory_swap_total_limit_bytes=536_870_912,
            profile_sets=("cpu-compare",),
        )
    )

    assert profile.memory_profile is True
    assert profile.memory_sample_interval_sec == 0.05
    assert profile.memory_limit_bytes == 536_870_912
    assert profile.memory_swap_total_limit_bytes == 536_870_912
    assert profile.profile_sets == ("cpu-compare",)


def test_build_container_profile_passes_perf_scheduler_backend(tmp_path: Path) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: exec]\n{"command": "true"}')

    profile, _ = build_container_profile_config(
        _config(
            tmp_path,
            transcript,
            scheduler_concurrency=True,
            scheduler_backend="perf",
            scheduler_perf_binary="/usr/bin/perf",
            scheduler_perf_mmap_pages=2048,
            proc_stat_comparison=True,
            proc_stat_interval_sec=0.25,
            proc_stat_baseline_sec=2.0,
        )
    )

    assert profile.scheduler_backend == "perf"
    assert profile.scheduler_perf_binary == "/usr/bin/perf"
    assert profile.scheduler_perf_mmap_pages == 2048
    assert profile.proc_stat_comparison is True
    assert profile.proc_stat_interval_sec == 0.25
    assert profile.proc_stat_baseline_sec == 2.0


def test_stage_pinchbench_initial_workspace_restores_source_and_inline_files(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "skill"
    (skill_dir / "tasks").mkdir(parents=True)
    (skill_dir / "assets" / "logs").mkdir(parents=True)
    (skill_dir / "assets" / "logs" / "input.log").write_text("entry\n")
    (skill_dir / "tasks" / "task_fixture.md").write_text(
        "---\n"
        "id: task_fixture\n"
        "workspace_files:\n"
        "  - source: logs/input.log\n"
        "    dest: input.log\n"
        "  - path: config/settings.json\n"
        "    content: '{}'\n"
        "---\n"
        "## Prompt\n"
    )

    task_file, restored = stage_pinchbench_initial_workspace(skill_dir, "category__task_fixture", tmp_path / "seed")

    assert task_file == skill_dir / "tasks" / "task_fixture.md"
    assert restored == 2
    assert (tmp_path / "seed" / "input.log").read_text() == "entry\n"
    assert (tmp_path / "seed" / "config" / "settings.json").read_text() == "{}"


def test_run_profile_preserves_transcript_pipeline_report_contract(tmp_path: Path, monkeypatch) -> None:
    transcript = _write_trace(tmp_path / "trace.jsonl", '[Tool Call: exec]\n{"command": "true"}')
    config = _config(tmp_path, transcript, prepare_command=None, strace_mode="none")

    def fake_run(_profile):
        replay = config.output_dir / "replay" / "output" / "replay_report.json"
        replay.parent.mkdir(parents=True, exist_ok=True)
        replay.write_text(json.dumps({"summary": {"actions_replayed": 1}}))
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

    report = run_transcript_trial_profile(config)

    assert report == {
        "schema_version": "container-transcript-profile.v1",
        "task_id": "task",
        "trial_name": "task.1-of-3.default",
        "transcript": str(config.transcript.resolve()),
        "transcript_sha256": hashlib.sha256(config.transcript.read_bytes()).hexdigest(),
        "initial_workspace": None,
        "initial_workspace_sha256": None,
        "source_workspace": str(config.source_workspace.resolve()),
        "source_workspace_sha256": None,
        "source_workspace_provenance_state": "unavailable",
        "image": "pinchbench:local",
        "image_identity": {"id": "sha256:image"},
        "actions_validated": 1,
        "task_definition": None,
        "fixtures_restored": 0,
        "workspace": str((config.output_dir / "replay" / "workspace").resolve()),
        "replay_report": str((config.output_dir / "replay" / "output" / "replay_report.json").resolve()),
        "container_profile_report": str((config.output_dir / "container_profile" / "profile_report.json").resolve()),
        "replay_summary": {"actions_replayed": 1},
        "container_cpu": {"average_logical_cores_used": 1.0},
        "scheduler_concurrency": None,
        "proc_stat_comparison": None,
        "strace": {"mode": "none"},
        "warning": None,
        "status": "valid",
        "profile_provenance": {"mode": "baseline", "profile_sets": []},
    }
    assert json.loads((config.output_dir / "pipeline_report.json").read_text()) == report

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-Docker proof for coordinator-backed unprofiled baselines.

Run explicitly with the same immutable fixture image used by the M1 suite::

    TASKTRACE_M1_DOCKER_TEST_IMAGE=<immutable-image-id> \
    TASKTRACE_M1_CPUSET=16-19 \
      uv run pytest -q offline_replayer/tests/test_coordinated_baseline_docker.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from offline_replayer.measurement.container import ProfileConfig
from offline_replayer.measurement.evidence import read_bundle, validate_raw_artifacts
from offline_replayer.measurement.pipeline import run_profiled_replay

IMAGE = os.environ.get("TASKTRACE_M1_DOCKER_TEST_IMAGE")
CPUSET = os.environ.get("TASKTRACE_M1_CPUSET")
SOURCE_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not IMAGE or not CPUSET,
    reason="set TASKTRACE_M1_DOCKER_TEST_IMAGE and TASKTRACE_M1_CPUSET for real baseline E2E",
)


def _command() -> str:
    return (
        "PYTHONPATH=/tasktrace-source python3 -m "
        "offline_replayer.examples.m1_labeled_phase "
        "--phase-log /tasktrace-profile-output/phases.jsonl "
        "--sentinel /tasktrace-profile-output/executed.once "
        "--serial-cpu-sec 0.10 --memory-mib 8 --memory-hold-sec 0.10 "
        "--parallel-workers 2 --parallel-cpu-sec 0.10 "
        "--teardown-hold-sec 0.08 "
        "--replay-report /tasktrace/replay_report.json"
    )


def _run(
    output: Path,
    *,
    coordinated_baseline: bool,
    profile_sets: tuple[str, ...] = (),
) -> dict[str, Any]:
    fixture_source = SOURCE_ROOT / "offline_replayer/examples/m1_labeled_phase.py"
    profile = ProfileConfig(
        command=_command(),
        prepare_command=None,
        output_dir=output / "container_profile",
        image=IMAGE,
        container=None,
        cpuset_cpus=CPUSET,
        cpu_limit=None,
        allocated_cpus=None,
        workdir="/tmp",
        user=None,
        env=(),
        mounts=(
            f"type=bind,src={SOURCE_ROOT},dst=/tasktrace-source,readonly",
            f"type=bind,src={output / 'replay'},dst=/tasktrace",
        ),
        # Both baseline and cpu-compare own this effective setting and must
        # override the legacy direct-profiler default.
        strace_mode="execve",
        string_limit=4096,
        shell="/bin/bash",
        network="none",
        terminate_descendants_at_finish=True,
        profile_sets=profile_sets,
        coordinated_baseline=coordinated_baseline,
        workload_identity={"fixture": "coordinated-baseline-comparison"},
        source_identity={
            "format": "m1-labeled-fixture",
            "sha256": hashlib.sha256(fixture_source.read_bytes()).hexdigest(),
        },
        expected_outcome={"kind": "fixture-replay", "action_count": 1},
    )
    return run_profiled_replay(
        profile=profile,
        output_dir=output,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={
            "schema_version": "container-coordinated-baseline-e2e.v1",
            "task_name": "coordinated-baseline-comparison",
            "phase_log_path": str((output / "container_profile" / "phases.jsonl").resolve()),
        },
    )


def test_real_baseline_and_profile_share_plan_but_not_profiler_request(
    tmp_path: Path,
) -> None:
    baseline_output = tmp_path / "baseline"
    profile_output = tmp_path / "profile"

    baseline = _run(baseline_output, coordinated_baseline=True)
    profiled = _run(
        profile_output,
        coordinated_baseline=False,
        profile_sets=("cpu-compare",),
    )

    assert baseline["profile_provenance"] == {
        "mode": "baseline",
        "profile_sets": [],
    }
    assert profiled["profile_provenance"] == {
        "mode": "profile",
        "profile_sets": ["cpu-compare"],
    }
    assert baseline["execution_plan_digest"] == profiled["execution_plan_digest"]
    assert baseline["execution_plan"] == profiled["execution_plan"]
    assert baseline["replay_summary"]["actions_executed"] == 1
    assert baseline["replay_summary"]["actions_timed_out"] == 0
    assert baseline["replay_summary"]["source_outcome_mismatch_count"] == 0
    assert baseline["container_cpu"]["wall_sec"] > 0
    assert baseline["container_cpu"]["usage_sec"] > 0
    assert baseline["container_cpu"]["valid_for_cpu_performance"] is True
    assert baseline["container_cpu"]["leaf_cpu"]["valid_for_cpu_performance"] is True
    assert baseline["container_cpu"]["ancestor_cpu"]["valid_for_cpu_performance"] is True
    assert baseline["strace"]["mode"] == "none"

    direct_attachment = baseline["coordinator_evidence_bundle"]
    assert direct_attachment["mode"] == "baseline"
    assert direct_attachment["profile_sets"] == []
    assert direct_attachment["resolved_profiler_ids"] == ["cgroup-cpu"]
    assert [item["profiler_id"] for item in direct_attachment["profiler_manifests"]] == ["cgroup-cpu"]
    direct = read_bundle(Path(direct_attachment["path"]))
    profiled_direct = read_bundle(Path(profiled["coordinator_evidence_bundle"]["path"]))
    assert [item.profiler_id for item in direct.profilers] == ["cgroup-cpu"]
    assert direct.profilers[0].status == "valid"
    assert direct.streams
    assert {item.artifact_id for item in direct.artifacts} >= {"cgroup-cpu-source-snapshots"}
    assert direct.provenance["profile_request"]["mode"] == "baseline"
    assert direct.provenance["profile_request"]["profile_sets"] == []
    assert direct.provenance["execution_plan"] == baseline["execution_plan"]
    assert profiled_direct.provenance["execution_plan"] == baseline["execution_plan"]
    assert profiled_direct.provenance["profile_request"]["mode"] == "profile"
    assert profiled_direct.provenance["profile_request"]["profile_sets"] == ["cpu-compare"]
    assert direct.provenance["profile_request"] != profiled_direct.provenance["profile_request"]

    final_bundle_path = baseline_output / "evidence_bundle.json"
    final_bundle = read_bundle(final_bundle_path)
    validate_raw_artifacts(final_bundle, baseline_output)
    assert baseline["evidence_bundle"]["authoritative"] is True
    assert baseline["evidence_bundle"]["path"] == str(final_bundle_path.resolve())
    assert final_bundle.provenance["profile_request"]["mode"] == "baseline"
    assert final_bundle.provenance["pipeline_augmentation"]["workload_executions_added"] == 0
    assert {item.profiler_id for item in final_bundle.profilers} == {
        "cgroup-cpu",
        "replay-actions",
        "labeled-phases",
    }

    assert (baseline_output / "container_profile" / "executed.once").read_text() == ("executed-once\n")
    assert (profile_output / "container_profile" / "executed.once").read_text() == ("executed-once\n")
    # The inner coordinator retains its bundle but does not render. The
    # pipeline renders the augmented bundle exactly once.
    assert not (baseline_output / "container_profile" / "unified_profile_artifacts.json").exists()
    rendered = json.loads((baseline_output / "unified_profile_artifacts.json").read_text())
    assert rendered["workload_executions"] == 1
    assert rendered["generation_method"] == ("verified-direct-coordinator-bundle-augmentation")
    for key in ("html", "svg", "animated_svg"):
        assert Path(rendered[key]).is_file()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from offline_replayer import cli
from offline_replayer.measurement.process_bpf import (
    EXPECTED_SCHEMA_VERSION,
    ProcessBpfEvidenceError,
    build_process_inventory_report,
    build_process_repeatability_report,
    write_process_inventory_report,
)

ROOT_PREFIX = "/tmp/tasktrace-python -m offline_replayer.adapters.atif_replay "


def _pipeline(path: Path, **summary_overrides: object) -> Path:
    summary = {
        "actions_timed_out": 0,
        "source_outcome_mismatch_count": 0,
        "actions_skipped_after_timeout": 0,
    }
    summary.update(summary_overrides)
    path.write_text(
        json.dumps(
            {
                "status": "valid",
                "profiling_valid": True,
                "network": "none",
                "source_trajectory_sha256": "a" * 64,
                "replay_summary": summary,
                "execution_plan": {"extensions": {"cpuset_cpus": "2-3"}},
                "image_identity": {
                    "id": "sha256:image",
                    "repo_digests": ["example/task@sha256:digest"],
                },
            }
        )
    )
    return path


def _trace(path: Path, rows: list[tuple[object, ...]] | None = None) -> Path:
    rows = rows or [
        # Host-global noise is outside the rooted replay. Even malformed
        # unrelated argv evidence must not poison the scoped result.
        ("100.000001", "noise", "1", "90", "2", "0", ""),
        ("100.000002", "python", "10", "100", "2", "0", ROOT_PREFIX + "/input"),
        ("100.000003", "python", "100", "101", "2", "", "[fork]"),
        ("100.000004", "sh", "100", "101", "3", "-2", "/missing/tool arg"),
        ("100.000005", "echo", "100", "101", "3", "0", "/bin/echo hello"),
        ("100.000006", "echo", "100", "101", "3", "", "[exit_group]"),
        ("100.000007", "python", "10", "100", "2", "", "[exit_group]"),
    ]
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["EPOCH", "COMM", "PPID", "PID", "CPU", "RET", "ARGS"])
        writer.writerows(rows)
    return path


def _stderr(path: Path, text: str = "AGENTSCRY_LOST_EVENTS=0\n") -> Path:
    path.write_text(text)
    return path


def _build(
    tmp_path: Path,
    *,
    expected: dict[str, object] | None = None,
    collector_text: str = "# reviewed collector\n",
    expected_indent: int | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    collector = tmp_path / "collector.py"
    collector.write_text(collector_text)
    expected_path = None
    if expected is not None:
        expected_path = tmp_path / "expected.json"
        expected_path.write_text(json.dumps(expected, sort_keys=True, indent=expected_indent) + "\n")
    return build_process_inventory_report(
        _trace(tmp_path / "events.csv"),
        _stderr(tmp_path / "tracer.stderr"),
        _pipeline(tmp_path / "pipeline.json"),
        root_argv_prefix=ROOT_PREFIX,
        expected_manifest_path=expected_path,
        collector_path=collector,
    )


def _expectation(report: dict[str, object], *, task_count: int = 2) -> dict[str, object]:
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "source_trajectory_sha256": "a" * 64,
        "image_id": "sha256:image",
        "image_repo_digest": "example/task@sha256:digest",
        "cpuset_cpus": "2-3",
        "root_argv_prefix": ROOT_PREFIX,
        "task_count": task_count,
        "successful_exec_count": 2,
        "failed_exec_attempt_count": 1,
        "canonical_tree_sha256": report["canonical_tree_sha256"],
        "binary_inventory": report["binary_inventory"],
    }


def test_builds_ancestry_scoped_inventory_and_ignores_host_noise(tmp_path: Path) -> None:
    report = _build(tmp_path)

    assert report["valid"] is True
    assert report["loss"]["perf_buffer_lost_event_count"] == 0
    assert report["scope"]["root_task_id"] == 100
    assert report["summary"] == {
        "task_count": 2,
        "successful_exec_count": 2,
        "failed_exec_attempt_count": 1,
        "distinct_binary_count": 2,
        "exit_group_observed_task_count": 2,
        "task_count_without_exit_group": 0,
    }
    assert report["binary_inventory"] == [
        {"binary": "/bin/echo", "exec_count": 1},
        {"binary": "/tmp/tasktrace-python", "exec_count": 1},
    ]
    assert "/bin/noise" not in json.dumps(report["canonical_tree"])
    assert report["tasks"][1]["parent_task_id"] == 100
    assert report["tasks"][1]["failed_exec_attempts"][0]["retval"] == -2
    assert len(report["canonical_tree_sha256"]) == 64


def test_exact_expectation_validates_inventory_and_tree(tmp_path: Path) -> None:
    first = _build(tmp_path)
    expected = _expectation(first)

    report = _build(tmp_path, expected=expected)

    assert report["valid"] is True
    assert report["expected_validation"]["valid"] is True
    assert report["expected_validation"]["reasons"] == []
    manifest = report["expected_validation"]["manifest"]
    assert manifest == report["artifacts"]["expected_manifest"]
    assert manifest["path"].endswith("expected.json")
    assert len(manifest["sha256"]) == 64
    assert len(report["artifacts"]["collector"]["sha256"]) == 64


def test_expectation_mismatch_fails_closed_in_report(tmp_path: Path) -> None:
    first = _build(tmp_path)
    expected = _expectation(first, task_count=999)

    report = _build(tmp_path, expected=expected)

    assert report["valid"] is False
    assert report["expected_validation"]["valid"] is False
    assert "expected task_count=999" in report["invalid_reasons"][0]


@pytest.mark.parametrize(
    "diagnostics",
    ["", "AGENTSCRY_LOST_EVENTS=2\n", "AGENTSCRY_LOST_EVENTS=0\nAGENTSCRY_LOST_EVENTS=0\n"],
)
def test_missing_nonzero_or_duplicate_loss_marker_is_rejected(tmp_path: Path, diagnostics: str) -> None:
    with pytest.raises(ProcessBpfEvidenceError, match="lost-event|dropped"):
        build_process_inventory_report(
            _trace(tmp_path / "events.csv"),
            _stderr(tmp_path / "tracer.stderr", diagnostics),
            _pipeline(tmp_path / "pipeline.json"),
            root_argv_prefix=ROOT_PREFIX,
        )


def test_duplicate_fork_event_is_rejected(tmp_path: Path) -> None:
    rows = [
        ("100.000001", "python", "10", "100", "2", "0", ROOT_PREFIX + "/input"),
        ("100.000002", "python", "100", "101", "2", "", "[fork]"),
        ("100.000003", "python", "100", "101", "2", "", "[fork]"),
    ]
    with pytest.raises(ProcessBpfEvidenceError, match="duplicate fork"):
        build_process_inventory_report(
            _trace(tmp_path / "events.csv", rows),
            _stderr(tmp_path / "tracer.stderr"),
            _pipeline(tmp_path / "pipeline.json"),
            root_argv_prefix=ROOT_PREFIX,
        )


def test_unlinked_descendant_event_is_rejected(tmp_path: Path) -> None:
    rows = [
        ("100.000001", "python", "10", "100", "2", "0", ROOT_PREFIX + "/input"),
        ("100.000002", "echo", "100", "101", "2", "0", "/bin/echo missing-fork"),
    ]
    with pytest.raises(ProcessBpfEvidenceError, match="lack captured fork edges"):
        build_process_inventory_report(
            _trace(tmp_path / "events.csv", rows),
            _stderr(tmp_path / "tracer.stderr"),
            _pipeline(tmp_path / "pipeline.json"),
            root_argv_prefix=ROOT_PREFIX,
        )


def test_replay_fidelity_or_out_of_cpuset_event_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProcessBpfEvidenceError, match="fidelity"):
        build_process_inventory_report(
            _trace(tmp_path / "events.csv"),
            _stderr(tmp_path / "tracer.stderr"),
            _pipeline(tmp_path / "pipeline.json", source_outcome_mismatch_count=1),
            root_argv_prefix=ROOT_PREFIX,
        )

    rows = [
        ("100.000001", "python", "10", "100", "4", "0", ROOT_PREFIX + "/input"),
    ]
    with pytest.raises(ProcessBpfEvidenceError, match="outside the execution plan cpuset"):
        build_process_inventory_report(
            _trace(tmp_path / "outside.csv", rows),
            _stderr(tmp_path / "outside.stderr"),
            _pipeline(tmp_path / "outside-pipeline.json"),
            root_argv_prefix=ROOT_PREFIX,
        )


def test_repeatability_compares_pid_independent_tree_and_inventory(tmp_path: Path) -> None:
    seed = _build(tmp_path / "seed")
    expected = _expectation(seed)
    first = _build(tmp_path / "first", expected=expected)
    second = _build(tmp_path / "second", expected=expected)
    # Dynamic task IDs and callback timestamps are retained in each proof but
    # intentionally do not define repeatability.
    second["scope"]["root_task_id"] = 9000
    second["tasks"][0]["task_id"] = 9000
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_process_inventory_report(first, first_path)
    write_process_inventory_report(second, second_path)

    report = build_process_repeatability_report([first_path, second_path])

    assert report["valid"] is True
    assert all(report["comparisons"].values())
    assert report["shared_summary"] == first["summary"]
    assert report["shared_canonical_tree_sha256"] == first["canonical_tree_sha256"]
    assert report["shared_collector"]["sha256"] == first["artifacts"]["collector"]["sha256"]
    assert report["shared_expected_manifest"]["sha256"] == first["expected_validation"]["manifest"]["sha256"]


def test_repeatability_rejects_collector_or_manifest_identity_drift(tmp_path: Path) -> None:
    seed = _build(tmp_path / "seed")
    expected = _expectation(seed)
    first = _build(tmp_path / "first", expected=expected)
    changed_collector = _build(
        tmp_path / "collector-drift",
        expected=expected,
        collector_text="# different collector\n",
    )
    changed_manifest_bytes = _build(
        tmp_path / "manifest-drift",
        expected=expected,
        expected_indent=2,
    )
    paths = []
    for name, proof in (
        ("first", first),
        ("collector", changed_collector),
        ("manifest", changed_manifest_bytes),
    ):
        path = tmp_path / f"{name}.json"
        write_process_inventory_report(proof, path)
        paths.append(path)

    collector_report = build_process_repeatability_report(paths[:2])
    manifest_report = build_process_repeatability_report([paths[0], paths[2]])

    assert collector_report["valid"] is False
    assert collector_report["comparisons"]["collector_sha256_equal"] is False
    assert manifest_report["valid"] is False
    assert manifest_report["comparisons"]["expected_manifest_sha256_equal"] is False


def test_public_report_commands_generate_content_bound_proofs(tmp_path: Path, capsys) -> None:
    seed = _build(tmp_path / "seed")
    expected = _expectation(seed)
    run = tmp_path / "run"
    run.mkdir()
    raw = _trace(run / "events.csv")
    stderr = _stderr(run / "tracer.stderr")
    pipeline = _pipeline(run / "pipeline.json")
    expected_path = run / "expected.json"
    expected_path.write_text(json.dumps(expected, sort_keys=True) + "\n")
    collector = run / "collector.py"
    collector.write_text("# reviewed collector\n")
    proof = run / "proof.json"

    assert (
        cli.main(
            [
                "report",
                "process-inventory-bpf",
                str(raw),
                str(stderr),
                str(pipeline),
                str(expected_path),
                str(collector),
                "--output",
                str(proof),
            ]
        )
        == 0
    )
    assert "Process inventory valid: true" in capsys.readouterr().out
    value = json.loads(proof.read_text())
    assert value["expected_validation"]["manifest"] == value["artifacts"]["expected_manifest"]
    assert value["artifacts"]["collector"]["path"] == str(collector.resolve())

    second = run / "proof-copy.json"
    second.write_bytes(proof.read_bytes())
    repeated = run / "repeatability.json"
    assert (
        cli.main(
            [
                "report",
                "process-inventory-bpf-repeatability",
                str(proof),
                str(second),
                "--output",
                str(repeated),
            ]
        )
        == 0
    )
    assert "Process repeatability valid: true" in capsys.readouterr().out
    comparison = json.loads(repeated.read_text())
    assert comparison["comparisons"]["collector_sha256_equal"] is True
    assert comparison["comparisons"]["expected_manifest_sha256_equal"] is True

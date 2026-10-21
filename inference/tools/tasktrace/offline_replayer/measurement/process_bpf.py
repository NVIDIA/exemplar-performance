# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed analysis for an external BCC process-event capture.

This module does not launch a privileged tracer.  It turns a reviewed
AgentScry-style CSV and the pipeline report from the *same* replay into a
scoped process-tree/binary-inventory proof.  The raw capture is host-wide (or
CPU-filtered); attribution is rooted at one uniquely identified replay driver
and follows only captured ``sched_process_fork`` edges.

The supported capture is intentionally narrower than the in-container strace
report.  It proves successful/failed ``execve`` attempts and task ancestry
without using ptrace, so an in-workload debugger can still work.  The current
external collector does not prove cwd, argument boundaries, ``execveat``, or
signal-terminated task exits.  Those limitations remain explicit in the
report and must be closed before this becomes the default ``software-deep``
backend.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

PROCESS_BPF_SCHEMA_VERSION = "tasktrace-bpf-process-inventory.v1"
EXPECTED_SCHEMA_VERSION = "tasktrace-representative-process-expectation.v1"
REPEATABILITY_SCHEMA_VERSION = "tasktrace-representative-process-repeatability.v1"
_REQUIRED_COLUMNS = frozenset({"EPOCH", "COMM", "PPID", "PID", "CPU", "RET", "ARGS"})
_LOSS_MARKER_RE = re.compile(r"^AGENTSCRY_LOST_EVENTS=(?P<count>[0-9]+)$", re.MULTILINE)


class ProcessBpfEvidenceError(RuntimeError):
    """Raised when process evidence cannot support an exact inventory claim."""


@dataclass(frozen=True)
class _Event:
    index: int
    timestamp_ns: int
    comm: str
    ppid: int
    pid: int
    cpu: int
    retval: int | None
    args: str

    @property
    def kind(self) -> str:
        if self.args == "[fork]":
            return "fork"
        if self.args == "[exit_group]":
            return "exit_group"
        return "exec"

    @property
    def binary(self) -> str | None:
        if self.kind != "exec" or not self.args:
            return None
        return self.args.split(" ", 1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessBpfEvidenceError(f"unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ProcessBpfEvidenceError(f"{label} must contain a JSON object")
    return value


def _decimal_epoch_ns(value: str, *, row_number: int) -> int:
    try:
        nanoseconds = Decimal(value) * Decimal(1_000_000_000)
    except InvalidOperation as exc:
        raise ProcessBpfEvidenceError(f"row {row_number}: EPOCH is not a decimal timestamp") from exc
    integral = nanoseconds.to_integral_value()
    if integral != nanoseconds:
        raise ProcessBpfEvidenceError(f"row {row_number}: EPOCH has sub-nanosecond precision")
    return int(integral)


def _integer(value: str, *, field: str, row_number: int, optional: bool = False) -> int | None:
    if optional and value == "":
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise ProcessBpfEvidenceError(f"row {row_number}: {field} is not an integer") from exc
    if result < 0 and field in {"PID", "PPID", "CPU"}:
        raise ProcessBpfEvidenceError(f"row {row_number}: {field} must be non-negative")
    return result


def _read_events(path: Path) -> list[_Event]:
    try:
        source = path.open(newline="")
    except OSError as exc:
        raise ProcessBpfEvidenceError(f"unable to read process trace: {path}") from exc
    with source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ProcessBpfEvidenceError("process trace is missing columns: " + ", ".join(missing))
        events: list[_Event] = []
        for index, row in enumerate(reader, start=2):
            if None in row:
                raise ProcessBpfEvidenceError(f"row {index}: malformed extra CSV fields")
            # The source capture is host-wide before ancestry filtering. A
            # malformed unrelated row must not invalidate a well-scoped
            # replay, but the same condition inside the rooted subtree fails
            # closed below.
            args = row.get("ARGS", "")
            pid = _integer(row.get("PID", ""), field="PID", row_number=index)
            ppid = _integer(row.get("PPID", ""), field="PPID", row_number=index)
            cpu = _integer(row.get("CPU", ""), field="CPU", row_number=index)
            retval = _integer(row.get("RET", ""), field="RET", row_number=index, optional=True)
            assert pid is not None and ppid is not None and cpu is not None
            events.append(
                _Event(
                    index=index,
                    timestamp_ns=_decimal_epoch_ns(row.get("EPOCH", ""), row_number=index),
                    comm=row.get("COMM", ""),
                    ppid=ppid,
                    pid=pid,
                    cpu=cpu,
                    retval=retval,
                    args=args,
                )
            )
    if not events:
        raise ProcessBpfEvidenceError("process trace has no events")
    return events


def _cpuset(value: str) -> frozenset[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ProcessBpfEvidenceError(f"invalid empty cpuset component in {value!r}")
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            try:
                first, last = int(first_text), int(last_text)
            except ValueError as exc:
                raise ProcessBpfEvidenceError(f"invalid cpuset {value!r}") from exc
            if first < 0 or last < first:
                raise ProcessBpfEvidenceError(f"invalid cpuset {value!r}")
            result.update(range(first, last + 1))
        else:
            try:
                cpu = int(item)
            except ValueError as exc:
                raise ProcessBpfEvidenceError(f"invalid cpuset {value!r}") from exc
            if cpu < 0:
                raise ProcessBpfEvidenceError(f"invalid cpuset {value!r}")
            result.add(cpu)
    return frozenset(result)


def _loss_count(stderr_path: Path) -> int:
    try:
        text = stderr_path.read_text()
    except OSError as exc:
        raise ProcessBpfEvidenceError(f"unable to read tracer diagnostics: {stderr_path}") from exc
    matches = list(_LOSS_MARKER_RE.finditer(text))
    if len(matches) != 1:
        raise ProcessBpfEvidenceError("tracer diagnostics must contain exactly one terminal lost-event marker")
    count = int(matches[0].group("count"))
    if count != 0 or "perf-buffer loss:" in text:
        raise ProcessBpfEvidenceError(f"process trace dropped {count} event(s)")
    return count


def _pipeline_invariants(pipeline: Mapping[str, Any]) -> tuple[str, frozenset[int]]:
    summary = pipeline.get("replay_summary")
    if not isinstance(summary, Mapping):
        raise ProcessBpfEvidenceError("pipeline report has no replay_summary object")
    checks = {
        "actions_timed_out": summary.get("actions_timed_out"),
        "source_outcome_mismatch_count": summary.get("source_outcome_mismatch_count"),
        "actions_skipped_after_timeout": summary.get("actions_skipped_after_timeout"),
    }
    nonzero = {name: value for name, value in checks.items() if value != 0}
    if nonzero:
        raise ProcessBpfEvidenceError(f"replay fidelity gate failed: {nonzero}")
    if pipeline.get("status") != "valid" or pipeline.get("profiling_valid") is not True:
        raise ProcessBpfEvidenceError("pipeline report is not valid")
    network_policy = pipeline.get("network_policy")
    effective_network_mode = (
        network_policy.get("effective_mode")
        if isinstance(network_policy, Mapping)
        else ("no-network" if pipeline.get("network") == "none" else "public")
    )
    if effective_network_mode != "no-network":
        raise ProcessBpfEvidenceError("representative replay must use effective no-network policy")
    plan = pipeline.get("execution_plan")
    extensions = plan.get("extensions") if isinstance(plan, Mapping) else None
    cpuset_text = extensions.get("cpuset_cpus") if isinstance(extensions, Mapping) else None
    if not isinstance(cpuset_text, str):
        raise ProcessBpfEvidenceError("execution plan has no fixed cpuset")
    source_sha = pipeline.get("source_trajectory_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ProcessBpfEvidenceError("pipeline report has no source trajectory digest")
    return source_sha, _cpuset(cpuset_text)


def _canonical_tree(
    root_pid: int,
    *,
    children: Mapping[int, set[int]],
    successful_execs: Mapping[int, list[_Event]],
) -> dict[str, Any]:
    def visit(pid: int, ancestors: frozenset[int]) -> dict[str, Any]:
        if pid in ancestors:
            raise ProcessBpfEvidenceError("captured fork graph contains a cycle")
        child_nodes = [visit(child, ancestors | {pid}) for child in sorted(children.get(pid, set()))]
        child_nodes.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {
            "exec_chain": [event.binary for event in successful_execs.get(pid, [])],
            "children": child_nodes,
        }

    return visit(root_pid, frozenset())


def _expected_inventory(value: Any) -> dict[str, int] | None:
    if not isinstance(value, list):
        return None
    result: dict[str, int] = {}
    for item in value:
        if not isinstance(item, Mapping):
            return None
        binary, count = item.get("binary"), item.get("exec_count")
        if not isinstance(binary, str) or not isinstance(count, int) or count < 1:
            return None
        if binary in result:
            return None
        result[binary] = count
    return result


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _bound_artifact_error(value: Any, *, label: str) -> str | None:
    if not isinstance(value, Mapping):
        return f"{label} identity is missing"
    path_text, expected_sha, expected_size = (
        value.get("path"),
        value.get("sha256"),
        value.get("size_bytes"),
    )
    if (
        not isinstance(path_text, str)
        or not isinstance(expected_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        return f"{label} identity is malformed"
    path = Path(path_text)
    if not path.is_file():
        return f"{label} artifact is missing: {path}"
    if path.stat().st_size != expected_size or _sha256(path) != expected_sha:
        return f"{label} artifact no longer matches its bound content identity"
    return None


def _validate_expected(report: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if expected.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        reasons.append("expected manifest schema_version is unsupported")
    comparisons = {
        "source_trajectory_sha256": report["source"]["trajectory_sha256"],
        "image_id": report["source"]["image_id"],
        "image_repo_digest": report["source"]["image_repo_digest"],
        "cpuset_cpus": report["scope"]["cpuset_cpus"],
        "root_argv_prefix": report["scope"]["root_argv_prefix"],
        "task_count": report["summary"]["task_count"],
        "successful_exec_count": report["summary"]["successful_exec_count"],
        "failed_exec_attempt_count": report["summary"]["failed_exec_attempt_count"],
        "canonical_tree_sha256": report["canonical_tree_sha256"],
    }
    for field, actual in comparisons.items():
        if expected.get(field) != actual:
            reasons.append(f"expected {field}={expected.get(field)!r}, observed {actual!r}")
    inventory = _expected_inventory(expected.get("binary_inventory"))
    if inventory is None:
        reasons.append("expected binary_inventory is malformed")
    else:
        actual = {item["binary"]: item["exec_count"] for item in report["binary_inventory"]}
        if inventory != actual:
            reasons.append("binary inventory differs from the pinned expectation")
    return reasons


def build_process_inventory_report(
    raw_csv_path: Path,
    tracer_stderr_path: Path,
    pipeline_report_path: Path,
    *,
    root_argv_prefix: str,
    expected_manifest_path: Path | None = None,
    collector_path: Path | None = None,
) -> dict[str, Any]:
    """Build one exact, ancestry-scoped binary inventory proof."""
    if not root_argv_prefix.startswith("/"):
        raise ProcessBpfEvidenceError("root_argv_prefix must begin with an absolute path")
    pipeline = _read_object(pipeline_report_path, label="pipeline report")
    source_sha, allocated_cpus = _pipeline_invariants(pipeline)
    loss_count = _loss_count(tracer_stderr_path)
    events = _read_events(raw_csv_path)
    observed_cpus = frozenset(event.cpu for event in events)
    if not observed_cpus <= allocated_cpus:
        raise ProcessBpfEvidenceError(
            "tracer emitted events outside the execution plan cpuset: " f"{sorted(observed_cpus - allocated_cpus)}"
        )

    roots = [
        event
        for event in events
        if event.kind == "exec" and event.retval == 0 and event.args.startswith(root_argv_prefix)
    ]
    if len(roots) != 1:
        raise ProcessBpfEvidenceError(
            f"expected one successful root exec matching {root_argv_prefix!r}; " f"observed {len(roots)}"
        )
    root_pid = roots[0].pid

    fork_by_child: dict[int, _Event] = {}
    children: dict[int, set[int]] = defaultdict(set)
    for event in events:
        if event.kind != "fork":
            continue
        if event.pid in fork_by_child:
            raise ProcessBpfEvidenceError(f"task ID {event.pid} has duplicate fork events; PID reuse is unsupported")
        fork_by_child[event.pid] = event
        children[event.ppid].add(event.pid)

    reachable = {root_pid}
    pending = [root_pid]
    while pending:
        child_ids = children.get(pending.pop(), set()) - reachable
        reachable.update(child_ids)
        pending.extend(child_ids)
    unlinked = sorted(
        {
            event.pid
            for event in events
            if event.ppid in reachable
            and event.pid not in reachable
            and event.pid != event.ppid
            and event.kind != "fork"
        }
    )
    if unlinked:
        raise ProcessBpfEvidenceError(
            "descendant events lack captured fork edges: " + ", ".join(str(pid) for pid in unlinked)
        )

    scoped_events = [event for event in events if event.pid in reachable]
    empty_scoped_args = [event.index for event in scoped_events if not event.args]
    if empty_scoped_args:
        raise ProcessBpfEvidenceError(
            "scoped process events have empty ARGS at CSV rows: " + ", ".join(str(row) for row in empty_scoped_args)
        )
    successful: dict[int, list[_Event]] = defaultdict(list)
    failed: dict[int, list[_Event]] = defaultdict(list)
    exit_groups: dict[int, list[_Event]] = defaultdict(list)
    for event in scoped_events:
        if event.kind == "exec":
            (successful if event.retval == 0 else failed)[event.pid].append(event)
        elif event.kind == "exit_group":
            exit_groups[event.pid].append(event)
    for values in successful.values():
        values.sort(key=lambda event: event.index)
    for values in failed.values():
        values.sort(key=lambda event: event.index)

    inventory = Counter(event.binary for values in successful.values() for event in values)
    if None in inventory:
        raise ProcessBpfEvidenceError("a successful exec has no parseable binary path")
    tree = _canonical_tree(root_pid, children=children, successful_execs=successful)
    tree_json = json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()

    image = pipeline.get("image_identity")
    if not isinstance(image, Mapping):
        raise ProcessBpfEvidenceError("pipeline report has no image identity")
    repo_digests = image.get("repo_digests")
    if not isinstance(repo_digests, list) or len(repo_digests) != 1:
        raise ProcessBpfEvidenceError("representative image must have one pinned repo digest")
    process_rows = []
    for pid in sorted(reachable):
        creation = fork_by_child.get(pid)
        process_rows.append(
            {
                "task_id": pid,
                "parent_task_id": creation.ppid if creation is not None else None,
                "creation_observed": creation is not None,
                "successful_execs": [
                    {
                        "binary": event.binary,
                        "flat_args": event.args,
                        "callback_epoch_ns": event.timestamp_ns,
                        "cpu": event.cpu,
                    }
                    for event in successful.get(pid, [])
                ],
                "failed_exec_attempts": [
                    {
                        "binary": event.binary,
                        "flat_args": event.args,
                        "retval": event.retval,
                        "callback_epoch_ns": event.timestamp_ns,
                        "cpu": event.cpu,
                    }
                    for event in failed.get(pid, [])
                ],
                "exit_group_observed": bool(exit_groups.get(pid)),
            }
        )

    report: dict[str, Any] = {
        "schema_version": PROCESS_BPF_SCHEMA_VERSION,
        "method": "external-bcc-execve-sched_process_fork-exit_group",
        "valid": True,
        "invalid_reasons": [],
        "source": {
            "trajectory_sha256": source_sha,
            "image_id": image.get("id"),
            "image_repo_digest": repo_digests[0],
            "network": pipeline.get("network"),
            "network_policy": pipeline.get("network_policy"),
            "pipeline_status": pipeline.get("status"),
        },
        "scope": {
            "kind": "captured-fork-descendants-of-unique-replay-root",
            "root_task_id": root_pid,
            "root_argv_prefix": root_argv_prefix,
            "cpuset_cpus": pipeline["execution_plan"]["extensions"]["cpuset_cpus"],
            "observed_cpu_ids": sorted(observed_cpus),
        },
        "loss": {
            "perf_buffer_lost_event_count": loss_count,
            "terminal_marker_observed": True,
        },
        "summary": {
            "task_count": len(reachable),
            "successful_exec_count": sum(len(values) for values in successful.values()),
            "failed_exec_attempt_count": sum(len(values) for values in failed.values()),
            "distinct_binary_count": len(inventory),
            "exit_group_observed_task_count": len(exit_groups),
            "task_count_without_exit_group": len(reachable - set(exit_groups)),
        },
        "binary_inventory": [{"binary": binary, "exec_count": count} for binary, count in sorted(inventory.items())],
        "canonical_tree": tree,
        "canonical_tree_sha256": hashlib.sha256(tree_json).hexdigest(),
        "tasks": process_rows,
        "limitations": {
            "event_timestamp": (
                "EPOCH is userspace callback time, not kernel event time; it is not "
                "eligible for canonical timeline alignment"
            ),
            "argv": (
                "ARGS is flat and limited by the external collector; argument boundaries "
                "and arbitrary long values are not proven"
            ),
            "cwd": "not collected",
            "execveat": "not collected by this external collector",
            "exit": (
                "exit_group is observed, but signal-terminated tasks need "
                "sched_process_exit for complete lifecycle evidence"
            ),
            "claim": "exact successful execve binary inventory and captured task ancestry only",
        },
        "artifacts": {
            "raw_csv": _artifact(raw_csv_path),
            "tracer_stderr": _artifact(tracer_stderr_path),
            "pipeline_report": _artifact(pipeline_report_path),
            **({"collector": _artifact(collector_path)} if collector_path is not None else {}),
        },
    }
    if expected_manifest_path is not None:
        expected = _read_object(expected_manifest_path, label="expected manifest")
        reasons = _validate_expected(report, expected)
        report["expected_validation"] = {
            "valid": not reasons,
            "reasons": reasons,
            "manifest": _artifact(expected_manifest_path),
        }
        report["artifacts"]["expected_manifest"] = _artifact(expected_manifest_path)
        if reasons:
            report["valid"] = False
            report["invalid_reasons"] = reasons
    return report


def build_process_repeatability_report(proof_paths: list[Path]) -> dict[str, Any]:
    """Compare independently acquired process proofs without PID/timestamp equality."""
    if len(proof_paths) < 2:
        raise ProcessBpfEvidenceError("repeatability requires at least two process proofs")
    proofs = [_read_object(path, label="process proof") for path in proof_paths]
    reasons: list[str] = []
    for index, proof in enumerate(proofs, start=1):
        if proof.get("schema_version") != PROCESS_BPF_SCHEMA_VERSION:
            reasons.append(f"run {index} has an unsupported process proof schema")
        if proof.get("valid") is not True:
            reasons.append(f"run {index} process proof is invalid")
        expected = proof.get("expected_validation")
        expected_manifest = expected.get("manifest") if isinstance(expected, Mapping) else None
        if (
            not isinstance(expected, Mapping)
            or expected.get("valid") is not True
            or not isinstance(expected_manifest, Mapping)
            or not isinstance(expected_manifest.get("sha256"), str)
            or not isinstance(expected_manifest.get("path"), str)
        ):
            reasons.append(f"run {index} did not pass its pinned expectation")
        else:
            error = _bound_artifact_error(expected_manifest, label=f"run {index} expected manifest")
            if error is not None:
                reasons.append(error)
        artifacts = proof.get("artifacts")
        collector = artifacts.get("collector") if isinstance(artifacts, Mapping) else None
        if (
            not isinstance(collector, Mapping)
            or not isinstance(collector.get("sha256"), str)
            or not isinstance(collector.get("path"), str)
        ):
            reasons.append(f"run {index} has no content-identified collector")
        else:
            error = _bound_artifact_error(collector, label=f"run {index} collector")
            if error is not None:
                reasons.append(error)
        duplicated_expected = artifacts.get("expected_manifest") if isinstance(artifacts, Mapping) else None
        if duplicated_expected != expected_manifest:
            reasons.append(f"run {index} expected-manifest identities are internally inconsistent")
        loss = proof.get("loss")
        if (
            not isinstance(loss, Mapping)
            or loss.get("perf_buffer_lost_event_count") != 0
            or loss.get("terminal_marker_observed") is not True
        ):
            reasons.append(f"run {index} has incomplete or lossy acquisition")

    reference = proofs[0]
    comparisons = {
        "source_identity_equal": all(proof.get("source") == reference.get("source") for proof in proofs[1:]),
        "scope_configuration_equal": all(
            {key: _nested(proof, "scope", key) for key in ("kind", "root_argv_prefix", "cpuset_cpus")}
            == {key: _nested(reference, "scope", key) for key in ("kind", "root_argv_prefix", "cpuset_cpus")}
            for proof in proofs[1:]
        ),
        "summary_equal": all(proof.get("summary") == reference.get("summary") for proof in proofs[1:]),
        "binary_inventory_equal": all(
            proof.get("binary_inventory") == reference.get("binary_inventory") for proof in proofs[1:]
        ),
        "canonical_tree_equal": all(
            proof.get("canonical_tree_sha256") == reference.get("canonical_tree_sha256") for proof in proofs[1:]
        ),
        "collector_sha256_equal": all(
            _nested(proof, "artifacts", "collector", "sha256") == _nested(reference, "artifacts", "collector", "sha256")
            for proof in proofs[1:]
        ),
        "expected_manifest_sha256_equal": all(
            _nested(proof, "expected_validation", "manifest", "sha256")
            == _nested(reference, "expected_validation", "manifest", "sha256")
            for proof in proofs[1:]
        ),
    }
    reasons.extend(f"repeatability comparison failed: {name}" for name, value in comparisons.items() if not value)
    return {
        "schema_version": REPEATABILITY_SCHEMA_VERSION,
        "valid": not reasons,
        "reasons": reasons,
        "run_count": len(proofs),
        "comparisons": comparisons,
        "shared_summary": reference.get("summary") if not reasons else None,
        "shared_binary_inventory": (
            reference.get("binary_inventory") if comparisons["binary_inventory_equal"] else None
        ),
        "shared_canonical_tree_sha256": (
            reference.get("canonical_tree_sha256") if comparisons["canonical_tree_equal"] else None
        ),
        "shared_collector": (
            _nested(reference, "artifacts", "collector") if comparisons["collector_sha256_equal"] else None
        ),
        "shared_expected_manifest": (
            _nested(reference, "expected_validation", "manifest")
            if comparisons["expected_manifest_sha256_equal"]
            else None
        ),
        "proof_artifacts": [_artifact(path) for path in proof_paths],
    }


def write_process_inventory_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

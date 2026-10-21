# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finding a WorkloadRun's pods: the jobs nvcre generates for it, and only those."""

import json
import subprocess
from unittest.mock import patch

from llmb_run.job_history import _nvcre_pull_all_logs, nvcre_rank0_pod
from llmb_run.nvcre_cluster import internal_job_names

WR = "pretrain-qwen3-gpus64-de4856"
OTHER_WR = "pretrain-nemotron-gpus8-aaaaaa"


class _Cluster:
    """The objects nvcre creates for a WorkloadRun, with the shapes seen on a real cluster:
    the WorkloadRun status names no job; nvcre `Job`s carry the WorkloadRun name in a label;
    pods carry the JobSet name `<job>-workload`. Another submission shares the namespace."""

    def __init__(self, workloadrun=None):
        # (job name, group, workflow label, creation order)
        self.jobs = [
            ("pretrain-qwen3-9b707", "group-0", WR, 1),
            ("pretrain-qwen3-dc767", "group-1-overflow", WR, 3),
            ("pretrain-nemotron-6efba", "group-0", OTHER_WR, 2),
        ]
        self.workloadrun = workloadrun or {"status": {"conditions": [], "workflowRef": {"name": WR}}, "metadata": {}}
        self.calls = []

    def pods(self, job):
        return [f"{job}-workload-node-0-{i}-xyz" for i in range(2)]

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        ok = lambda out: subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")  # noqa: E731
        if argv[:3] == ["kubectl", "get", "workloadrun"]:
            return ok(json.dumps(self.workloadrun))
        if argv[:3] == ["kubectl", "get", "jobs.nvcre.nvidia.com"]:
            selector = argv[argv.index("-l") + 1]
            assert selector.startswith("nvcre.nvidia.com/workflow=")
            wanted = selector.split("=", 1)[1]
            rows = sorted((j for j in self.jobs if j[2] == wanted), key=lambda j: j[3])
            return ok("".join(f"{name}\t{group}\n" for name, group, _, _ in rows))
        if argv[:3] == ["kubectl", "get", "pods"]:
            label = argv[argv.index("-l") + 1]
            jobset = label.split("jobset-name=", 1)[1].split(",")[0]
            job = jobset.removesuffix("-workload")
            names = self.pods(job)
            if "job-completion-index=0" in label:
                return ok(names[0] if any(job == j[0] for j in self.jobs) else "")
            return ok("".join(f"{n}\n" for n in names) if any(job == j[0] for j in self.jobs) else "")
        if argv[:2] == ["kubectl", "logs"]:
            return ok(f"log of {argv[2]}")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected")


def test_jobs_are_found_through_the_workflow_label_primary_group_first():
    cluster = _Cluster()

    with patch("subprocess.run", cluster):
        assert internal_job_names("ns", WR) == ["pretrain-qwen3-9b707", "pretrain-qwen3-dc767"]
        assert internal_job_names("ns", OTHER_WR) == ["pretrain-nemotron-6efba"]  # never the other submission's jobs


def test_primary_group_comes_first_even_if_an_overflow_group_was_created_earlier():
    cluster = _Cluster()
    cluster.jobs = [
        ("overflow-job", "group-1-overflow", WR, 1),
        ("primary-job", "group-0", WR, 2),
    ]

    with patch("subprocess.run", cluster):
        assert internal_job_names("ns", WR) == ["primary-job", "overflow-job"]


def test_a_name_exposed_by_the_workloadrun_is_used_directly():
    cluster = _Cluster({"status": {"jobName": "named-by-status"}, "metadata": {}})

    with patch("subprocess.run", cluster):
        assert internal_job_names("ns", WR) == ["named-by-status"]

    assert not any(c[:3] == ["kubectl", "get", "jobs.nvcre.nvidia.com"] for c in cluster.calls)


def test_nothing_is_guessed_when_the_workloadrun_has_no_jobs():
    cluster = _Cluster()

    with patch("subprocess.run", cluster):
        assert internal_job_names("ns", "pretrain-unknown-000000") == []


def test_kubectl_failures_and_timeouts_resolve_to_nothing():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("kubectl", 30)):
        assert internal_job_names("ns", WR) == []
    with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="denied")):
        assert internal_job_names("ns", WR) == []


def test_rank0_pod_is_resolved_from_the_primary_group():
    cluster = _Cluster()

    with patch("subprocess.run", cluster):
        assert nvcre_rank0_pod("ns", WR) == "pretrain-qwen3-9b707-workload-node-0-0-xyz"


def test_all_logs_cover_every_group_of_the_workloadrun_and_nothing_else(tmp_path):
    cluster = _Cluster()

    with patch("subprocess.run", cluster):
        _nvcre_pull_all_logs("ns", WR, str(tmp_path))

    pulled = sorted(p.name for p in (tmp_path / "pod_logs").glob("*.log"))
    assert pulled == [
        "pretrain-qwen3-9b707-workload-node-0-0-xyz.log",
        "pretrain-qwen3-9b707-workload-node-0-1-xyz.log",
        "pretrain-qwen3-dc767-workload-node-0-0-xyz.log",
        "pretrain-qwen3-dc767-workload-node-0-1-xyz.log",
    ]  # both groups of this WorkloadRun; none of the nemotron job's pods
    logs_calls = [c for c in cluster.calls if c[:2] == ["kubectl", "logs"]]
    assert logs_calls and not any("--tail" in c for c in logs_calls)  # full logs, never truncated


def test_a_slow_pod_does_not_abort_the_log_pull(tmp_path):
    """One pod's log fetch timing out must only cost that pod's file, not the pull."""
    cluster = _Cluster()
    slow_pod = "pretrain-qwen3-9b707-workload-node-0-1-xyz"

    def fake_run(argv, **kwargs):
        if argv[:2] == ["kubectl", "logs"] and argv[2] == slow_pod:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return cluster(argv, **kwargs)

    with patch("subprocess.run", fake_run):
        _nvcre_pull_all_logs("ns", WR, str(tmp_path))

    assert (tmp_path / "pod_logs" / "pretrain-qwen3-9b707-workload-node-0-0-xyz.log").exists()
    assert not (tmp_path / "pod_logs" / f"{slow_pod}.log").exists()
    assert (tmp_path / "pod_logs.tar.gz").exists()  # the pull still completes for the other pods

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A submission must be bound to its own WorkloadRun, not the newest entry in the shared registry."""

import json

import pytest

from llmb_run.job_history import get_job, record_job_submission
from llmb_run.nvcre_registry import workloadrun_name_for
from llmb_run.slurm_utils import SlurmJob
from llmb_run.tasks import WorkloadTask


@pytest.fixture
def nemorun_home(tmp_path):
    """Two submissions share one registry; the one for task_b is registered last."""
    home = tmp_path / "nemorun"
    task_a = home / "experiments" / "title" / "exp1" / "task_a"
    task_b = home / "experiments" / "title" / "exp2" / "task_b"
    task_a.mkdir(parents=True)
    task_b.mkdir(parents=True)
    registry = {
        "exp1___task_a___wr-a": {"workloadrun_name": "wr-a", "job_dir": str(task_a)},
        "exp2___task_b___wr-b": {"workloadrun_name": "wr-b", "job_dir": str(task_b)},
    }
    (home / ".nvcre_jobs.json").write_text(json.dumps(registry))
    return home, task_a, task_b


def test_each_submission_resolves_to_its_own_workloadrun(nemorun_home):
    _, task_a, task_b = nemorun_home

    assert workloadrun_name_for(str(task_a)) == "wr-a"  # not the newest entry, wr-b
    assert workloadrun_name_for(str(task_b)) == "wr-b"


def test_a_directory_inside_the_job_dir_resolves_to_that_job(nemorun_home):
    _, task_a, _ = nemorun_home
    inner = task_a / "nvcre" / "stage"
    inner.mkdir(parents=True)

    assert workloadrun_name_for(str(inner)) == "wr-a"


def test_unknown_directories_are_never_guessed_from_other_entries(nemorun_home, tmp_path):
    home, _, _ = nemorun_home
    stranger = home / "experiments" / "title" / "exp3" / "task_c"
    stranger.mkdir(parents=True)

    assert workloadrun_name_for(str(stranger)) is None
    assert workloadrun_name_for(None) is None
    assert workloadrun_name_for(str(tmp_path / "no" / "registry" / "here" / "at" / "all")) is None


def test_job_history_binds_the_row_to_the_submitted_workloadrun(nemorun_home, cluster_config_factory, tmp_path):
    _, task_a, _ = nemorun_home
    config = cluster_config_factory(
        {
            "launcher": {"llmb_install": str(tmp_path / "llmb")},
            "execution_target": "kubernetes",
            "kubernetes": {"namespace": "ns"},
        }
    )
    workloads = {"w": {"metadata": {"run": {"launcher_type": "megatron_bridge"}}, "workload": "w"}}

    record_job_submission(
        config,
        WorkloadTask("w", "7b", "bf16", 8),
        SlurmJob(job_id=77, job_workdir=str(task_a), llmb_config_path=None),
        workloads,
    )

    assert get_job(config, 77)["workloadrun_name"] == "wr-a"


@pytest.mark.parametrize("payload", ["[]", '"text"', "42", "null"])
def test_a_registry_that_is_not_a_json_object_is_treated_as_empty(nemorun_home, payload):
    home, task_a, _ = nemorun_home
    (home / ".nvcre_jobs.json").write_text(payload)

    assert workloadrun_name_for(str(task_a)) is None


def test_malformed_entries_are_skipped_and_valid_ones_still_found(nemorun_home):
    home, task_a, _ = nemorun_home
    registry = json.loads((home / ".nvcre_jobs.json").read_text())
    registry["broken-1"] = "not an object"
    registry["broken-2"] = ["a", "list"]
    registry["broken-3"] = None
    (home / ".nvcre_jobs.json").write_text(json.dumps(registry))

    assert workloadrun_name_for(str(task_a)) == "wr-a"


def test_an_unreadable_registry_file_is_treated_as_empty(nemorun_home):
    home, task_a, _ = nemorun_home
    (home / ".nvcre_jobs.json").write_text("{not json")

    assert workloadrun_name_for(str(task_a)) is None

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmb_run.slurm_args import (
    ADDITIONAL_SLURM_PARAMS_KEY,
    build_cli_slurm_args,
    validate_no_additional_slurm_params_conflict,
)


def test_build_cli_slurm_args_serializes_first_class_and_passthrough():
    slurm_args = build_cli_slurm_args(
        nodelist="node001,node002",
        reservation="res_a",
        segment=8,
        nice=100,
        slurm_args=["exclusive", "constraint=gpu"],
    )

    assert slurm_args is not None
    assert set(slurm_args.to_additional_slurm_params().split(";")) == {
        "nodelist=node001,node002",
        "reservation=res_a",
        "segment=8",
        "nice=100",
        "exclusive",
        "constraint=gpu",
    }
    assert len(slurm_args.to_additional_slurm_params().split(";")) == 6
    assert set(slurm_args.to_sbatch_args()) == {
        "--nodelist=node001,node002",
        "--reservation=res_a",
        "--segment=8",
        "--nice=100",
        "--exclusive",
        "--constraint=gpu",
    }
    assert len(slurm_args.to_sbatch_args()) == 6


def test_build_cli_slurm_args_rejects_raw_token():
    with pytest.raises(ValueError, match="should not include a leading '--'"):
        build_cli_slurm_args(slurm_args=["--constraint=gpu"])


def test_build_cli_slurm_args_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="`reservation` has a dedicated flag"):
        build_cli_slurm_args(reservation="res_a", slurm_args=["reservation=res_b"])


@pytest.mark.parametrize("key", ["nodelist", "exclude", "reservation", "segment", "nice"])
def test_build_cli_slurm_args_rejects_first_class_passthrough_keys(key):
    with pytest.raises(ValueError, match=f"`{key}` has a dedicated flag"):
        build_cli_slurm_args(slurm_args=[f"{key}=value"])


def test_validate_no_additional_slurm_params_conflict_checks_all_sources(monkeypatch):
    slurm_args = build_cli_slurm_args(nodelist="node001")
    monkeypatch.setenv(ADDITIONAL_SLURM_PARAMS_KEY, "reservation=res_a")

    with pytest.raises(
        ValueError,
        match="process environment, cluster config environment, workload config environment, task environment overrides",
    ):
        validate_no_additional_slurm_params_conflict(
            cli_args=slurm_args,
            cluster_environment={ADDITIONAL_SLURM_PARAMS_KEY: "reservation=res_b"},
            workload_environment={ADDITIONAL_SLURM_PARAMS_KEY: "reservation=res_c"},
            task_environment={ADDITIONAL_SLURM_PARAMS_KEY: "reservation=res_d"},
        )

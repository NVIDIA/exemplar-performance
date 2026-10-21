# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from llmb_install.core.dependency import _resolve_dependencies
from llmb_install.downloads.huggingface import build_hf_download_plan
from llmb_install.downloads.image import get_required_images

REPO_ROOT = Path(__file__).resolve().parents[4]
DSV4_ROOT = REPO_ROOT / "inference/slurm_recipes/llm-serving/dsv4/disagg"
SRT_RELEASE = "c180328b98c3793ca84a1e24a030f90545eb7d5d"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DATASET_REVISION = "d01c4538f8b74f6e45635da1c06cb013902f3a6d"


@pytest.mark.parametrize("gpu_type", ["gb200", "gb300"])
def test_dsv4f_install_plan_is_pinned_and_llmb_owned(gpu_type: str):
    workload_dir = DSV4_ROOT / gpu_type / "sglang_dynamo"
    workload_key = f"inference_dsv4f_sglang_{gpu_type}"
    metadata = yaml.safe_load((workload_dir / "metadata.yaml").read_text())

    workloads = {workload_key: metadata}
    assert get_required_images(workloads, [workload_key]) == {
        "nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.5.1": "dynamo-sglang-1.5.1.sqsh",
        "nginx@sha256:846993cfd1ec2f814d7f3cfdc8df7aa67ecfe6ab233fd990c82d34eea47beb8e": "nginx-1.27.4.sqsh",
        "nvcr.io#nvidia/ai-dynamo/aiperf:0.12.0": "aiperf-0.12.0.sqsh",
    }

    plan = build_hf_download_plan(workloads, [workload_key], gpu_type=gpu_type)
    assert {(spec.repo_type, spec.repo_id): (spec.revision, spec.target, spec.name) for spec in plan.repos} == {
        ("model", "deepseek-ai/DeepSeek-V4-Flash-0731"): (MODEL_REVISION, "workload", "deepseek-v4-flash-0731"),
        ("dataset", "nv-camilom/agentic_coding"): (DATASET_REVISION, "workload", "dataset"),
    }

    dependencies = _resolve_dependencies(metadata)
    assert dependencies["git"]["srtctl"]["commit"] == SRT_RELEASE
    assert dependencies["pip"][0]["editable"] is True
    assert [task["name"] for task in metadata["setup"]["tasks"]] == [
        "setup_srtslurm",
        "prepare_dataset_recipe",
    ]
    assert all(task["job_type"] == "local" for task in metadata["setup"]["tasks"])
    assert set(metadata["run"]["srt_slurm"]) == {"recipe", "recipe_env", "selector"}


@pytest.mark.parametrize("gpu_type", ["gb200", "gb300"])
def test_dsv4f_recipe_uses_installed_aliases(gpu_type: str):
    workload_dir = DSV4_ROOT / gpu_type / "sglang_dynamo"
    metadata = yaml.safe_load((workload_dir / "metadata.yaml").read_text())
    recipe_name = metadata["run"]["srt_slurm"]["recipe"]
    recipe = yaml.safe_load((workload_dir / recipe_name).read_text())
    base = recipe["base"]

    assert base["model"]["path"] == "deepseek-v4-flash-0731"
    assert base["model"]["container"] == "dynamo-sglang-1.5.1"
    assert base["frontend"]["nginx_container"] == "nginx-1.27.4"
    assert base["benchmark"]["container_image"] == "aiperf-0.12.0"
    assert base["identity"]["model"] == {
        "repo": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "revision": MODEL_REVISION,
    }
    assert base["resources"]["gpu_type"] == gpu_type
    assert (
        base["resources"]["gpus_per_node"] * (base["resources"]["prefill_nodes"] + base["resources"]["decode_nodes"])
        == 8
    )
    assert base["container_mounts"] == {"$LLMB_WORKLOAD/dataset": "/datasets"}


@pytest.mark.parametrize("gpu_type", ["gb200", "gb300"])
def test_dsv4f_dataset_is_unpacked_without_staging_recipe(tmp_path: Path, gpu_type: str):
    workload_dir = DSV4_ROOT / gpu_type / "sglang_dynamo"
    installed_workload = tmp_path / "workload"
    dataset_root = installed_workload / "dataset"
    dataset_root.mkdir(parents=True)
    with zipfile.ZipFile(dataset_root / "agentic_coding.zip", "w") as archive:
        archive.writestr("agentic_coding/example.json", "{}")

    script = DSV4_ROOT.parent / "prepare_dataset_recipe.py"
    subprocess.run(
        [sys.executable, str(script)],
        cwd=workload_dir,
        env={**os.environ, "LLMB_WORKLOAD": str(installed_workload)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert not (installed_workload / "recipes").exists()
    assert (dataset_root / "agentic_coding" / "example.json").is_file()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from llmb_install.core.dependency import _resolve_dependencies
from llmb_install.downloads.huggingface import build_hf_download_plan
from llmb_install.downloads.image import get_required_images

REPO_ROOT = Path(__file__).resolve().parents[4]
SRT_RELEASE = "c180328b98c3793ca84a1e24a030f90545eb7d5d"

CASES = (
    {
        "key": "inference_glm5",
        "path": "inference/slurm_recipes/llm-serving/glm5/disagg/gb300/trtllm_dynamo",
        "gpu": "gb300",
        "repo": "nvidia/GLM-5-NVFP4",
        "revision": "dc54ff55a7e9e71b85db953d8bc22eca894b44c6",
        "model_alias": "glm5-nvfp4",
        "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.3",
        "image_alias": "trtllm-runtime-1.1.0-dev.3.sqsh",
        "container_alias": "trtllm-runtime-1.1.0-dev.3",
        "recipe_count": 6,
        "scales": [32, 36, 40],
    },
    {
        "key": "inference_kimi2.6",
        "path": "inference/slurm_recipes/llm-serving/kimi2.6/disagg/gb200/trtllm_dynamo",
        "gpu": "gb200",
        "repo": "nvidia/Kimi-K2.6-NVFP4",
        "revision": "2fd3a800dedd098b8327eb49e93ebc75f85da19f",
        "model_alias": "kimi26-nvfp4",
        "image": "nvcr.io#nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.2",
        "image_alias": "trtllm-runtime-1.1.0-dev.2.sqsh",
        "container_alias": "trtllm-runtime-1.1.0-dev.2",
        "recipe_count": 5,
        "scales": [24, 28, 40, 48],
    },
)


@pytest.mark.parametrize("case", CASES, ids=("glm5", "kimi26"))
def test_trtllm_install_plan_is_pinned_and_llmb_owned(case):
    workload_dir = REPO_ROOT / case["path"]
    metadata = yaml.safe_load((workload_dir / "metadata.yaml").read_text())
    workloads = {case["key"]: metadata}

    assert get_required_images(workloads, [case["key"]]) == {case["image"]: case["image_alias"]}
    plan = build_hf_download_plan(workloads, [case["key"]], gpu_type=case["gpu"])
    assert len(plan.repos) == 1
    assert plan.repos[0].repo_id == case["repo"]
    assert plan.repos[0].revision == case["revision"]
    assert plan.repos[0].target == "workload"
    assert plan.repos[0].name == case["model_alias"]

    dependencies = _resolve_dependencies(metadata)
    assert dependencies["git"]["srtctl"]["commit"] == SRT_RELEASE
    assert dependencies["pip"][0]["editable"] is True
    assert all(task["job_type"] == "local" for task in metadata["setup"]["tasks"])

    model_config = metadata["run"]["gpu_configs"][case["gpu"]]["model_configs"][0]
    assert model_config["scales"] == case["scales"]
    assert model_config["exact_scales"] is True


@pytest.mark.parametrize("case", CASES, ids=("glm5", "kimi26"))
def test_trtllm_recipes_use_llmb_aliases_and_pinned_identity(case):
    workload_dir = REPO_ROOT / case["path"]
    recipes = sorted((workload_dir / "recipes").rglob("*.yaml"))
    assert len(recipes) == case["recipe_count"]

    for recipe in recipes:
        data = yaml.safe_load(recipe.read_text())
        assert "base" not in data
        assert data["model"]["path"] == case["model_alias"]
        assert data["model"]["container"] == case["container_alias"]
        assert data["identity"]["model"] == {"repo": case["repo"], "revision": case["revision"]}
        assert data["frontend"]["enable_multiple_frontends"] is False


@pytest.mark.parametrize(
    ("recipe", "expected_name", "expected_concurrencies"),
    (
        (
            "recipes/gb200_nvfp4/ISL8K_OSL1K/high_pareto_conc_128_256_512_1024.yaml",
            "kimi_k26_nvfp4_trtllm_dynamo_isl8k_osl1k_high_pareto_128x256x512x1024",
            "128x256x512x1024",
        ),
        (
            "recipes/gb200_nvfp4/ISL32K_OSL4K/high_pareto_conc_128_256_512.yaml",
            "kimi_k26_nvfp4_trtllm_dynamo_isl32k_osl4k_high_pareto_128x256x512",
            "128x256x512",
        ),
    ),
)
def test_kimi_recipe_names_match_concurrency_sweeps(recipe, expected_name, expected_concurrencies):
    workload_dir = REPO_ROOT / "inference/slurm_recipes/llm-serving/kimi2.6/disagg/gb200/trtllm_dynamo"
    data = yaml.safe_load((workload_dir / recipe).read_text())

    assert data["name"] == expected_name
    assert data["benchmark"]["concurrencies"] == expected_concurrencies

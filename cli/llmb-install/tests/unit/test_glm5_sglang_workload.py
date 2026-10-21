# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from llmb_install.core.dependency import _resolve_dependencies
from llmb_install.downloads.huggingface import build_hf_download_plan
from llmb_install.downloads.image import get_required_images

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKLOAD_DIR = REPO_ROOT / "inference/slurm_recipes/llm-serving/glm5/disagg/gb300/sglang_dynamo"
SRT_RELEASE = "c180328b98c3793ca84a1e24a030f90545eb7d5d"
MODEL_REVISION = "4f96cc5eec29dcee5d6ded54f7ffe889438f9516"


def _metadata() -> dict:
    return yaml.safe_load((WORKLOAD_DIR / "metadata.yaml").read_text())


def test_glm5_sglang_install_plan_is_pinned_and_llmb_owned():
    metadata = _metadata()
    workloads = {"inference_glm5_sglang_dynamo": metadata}

    images = get_required_images(workloads, ["inference_glm5_sglang_dynamo"])
    assert images == {
        "nvcr.io#nvidia/ai-dynamo/sglang-runtime:1.3.0": "sglang-runtime-1.3.0.sqsh",
        "nginx@sha256:846993cfd1ec2f814d7f3cfdc8df7aa67ecfe6ab233fd990c82d34eea47beb8e": "nginx-1.27.4.sqsh",
    }

    plan = build_hf_download_plan(workloads, ["inference_glm5_sglang_dynamo"], gpu_type="gb300")
    assert len(plan.repos) == 1
    assert plan.repos[0].repo_id == "zai-org/GLM-5-FP8"
    assert plan.repos[0].revision == MODEL_REVISION
    assert plan.repos[0].target == "workload"
    assert plan.repos[0].name == "glm5-fp8"

    dependencies = _resolve_dependencies(metadata)
    assert dependencies["git"]["srtctl"]["commit"] == SRT_RELEASE
    assert dependencies["pip"][0]["editable"] is True

    setup_task = metadata["setup"]["tasks"][0]
    assert setup_task["name"] == "setup_srtslurm"
    assert setup_task["job_type"] == "local"
    assert not setup_task.get("requires_gpus", False)

    model_config = metadata["run"]["gpu_configs"]["gb300"]["model_configs"][0]
    assert model_config["scales"] == [40, 72]
    assert model_config["exact_scales"] is True


def test_glm5_sglang_recipes_are_schema_v2_and_use_llmb_aliases():
    recipes = sorted((WORKLOAD_DIR / "recipes/gb300").rglob("*.yaml"))
    assert len(recipes) == 14

    for recipe in recipes:
        data = yaml.safe_load(recipe.read_text())
        assert data["override_validated"] == {}
        assert data["base"]["model"]["path"] == "glm5-fp8"
        assert data["base"]["model"]["container"] == "sglang-runtime-1.3.0"
        assert data["base"]["frontend"]["nginx_container"] == "nginx-1.27.4"
        assert data["base"]["identity"]["model"]["revision"] == MODEL_REVISION

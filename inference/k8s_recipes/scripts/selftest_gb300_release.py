#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the GLM-5 GB300 AWS execution contract."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes/llm-perf/Glm5/Gb300_k8s"
STACK = ROOT / "serving/sglang-disagg-gb300"
EXPECTED_REVISION = "4f96cc5eec29dcee5d6ded54f7ffe889438f9516"
EXPECTED_HOST_PATH = f"/mnt/k8s-disks/0/llmb-model-cache/glm5-fp8/{EXPECTED_REVISION}"
EXPECTED_AIPERF = "git+https://github.com/ai-dynamo/aiperf.git@38687855e98044fcf12ee48c6794128f10b6780b"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


capabilities = load_module("gb300_capability_registry", ROOT / "scripts/capability_registry.py")

preflight = load_module("gb300_preflight", ROOT / "scripts/preflight.py")

EXPECTED_SYNTHETIC = {
    (1024, 1024): {"isl": 922, "isl_stddev": 59, "osl": 922, "osl_stddev": 59},
    (8192, 1024): {"isl": 7373, "isl_stddev": 473, "osl": 922, "osl_stddev": 59},
    (16384, 512): {"isl": 14746, "isl_stddev": 946, "osl": 461, "osl_stddev": 30},
}

recipes = sorted(RECIPES.rglob("recipe.yaml"))
assert len(recipes) == 21, f"expected 21 GB300 recipes, found {len(recipes)}"
for recipe_path in recipes:
    recipe = yaml.safe_load(recipe_path.read_text())
    envelope = recipe["envelope"]
    serving = recipe["serving"]
    staging = serving["model_cache_staging"]
    bench = recipe["bench"]
    synthetic = bench["synthetic"]
    concurrency = bench["sweep_concurrency"]

    assert serving["stack"] == "sglang-disagg-gb300"
    assert serving["model_revision"] == EXPECTED_REVISION
    assert staging == {"mode": "node-local-versioned-hostpath", "host_path": EXPECTED_HOST_PATH}
    assert bench["aiperf_ref"] == EXPECTED_AIPERF
    assert bench["aiperf_unsafe_override"] is False
    assert envelope["distribution"] == "synthetic" and envelope["mode"] == "synthetic"
    assert "dataset" not in bench and len(concurrency) == 1
    nominal = (synthetic["nominal_isl"], synthetic["nominal_osl"])
    assert nominal in EXPECTED_SYNTHETIC
    for key, value in EXPECTED_SYNTHETIC[nominal].items():
        assert synthetic[key] == value, f"wrong {key} for {recipe_path}"
    assert synthetic["request_count_multiplier"] == 10
    assert synthetic["warmup_request_multiplier"] == 2
    assert synthetic["num_entries"] == concurrency[0] * 10
    assert synthetic["seed"] == 0 and synthetic["range_ratio"] == 0.8
    assert synthetic["length_distribution"] == "bounded_uniform"
    assert capabilities._req_nvlink_imex(recipe), f"IMEX not required for {recipe_path}"
    imex_gap = next(
        gap
        for gap in capabilities.evaluate(recipe, {"nvlink_imex": {"crd_present": False, "channel_provisioned": False}})
        if gap.id == "nvlink-imex"
    )
    assert imex_gap.level == "FAIL" and imex_gap.action is None

    placements = preflight.disagg_placements(serving["disagg"])
    expected_pods = sum(
        int(role.get("replicas", 1) or 1) * int(role.get("nodes_per_worker", 1) or 1)
        for role in (serving["disagg"]["prefill"], serving["disagg"]["decode"])
    )
    expected_gpus = sum(
        int(role.get("replicas", 1) or 1) * int(role["tp"])
        for role in (serving["disagg"]["prefill"], serving["disagg"]["decode"])
    )
    assert len(placements) == expected_pods, f"wrong pod count for {recipe_path}: {len(placements)}"
    assert sum(gpus for _, gpus in placements) == expected_gpus, f"wrong GPU total for {recipe_path}"
    assert (expected_pods, expected_gpus) in {(10, 40), (18, 72)}

    rendered = recipe_path.parent / "rendered"
    assert {path.name for path in rendered.iterdir()} == {
        "bench-job.yaml",
        "frontend.yaml",
        "infra.yaml",
        "workers.yaml",
    }
    workers = (rendered / "workers.yaml").read_text()
    frontend = (rendered / "frontend.yaml").read_text()
    for text in (workers, frontend):
        assert EXPECTED_HOST_PATH in text
        assert "verify-node-model-cache" in text
        assert "NODE_CACHE_VERIFIED" in text
    assert "${MODEL_CACHE_PVC}" not in frontend and "claimName:" not in frontend
    bench_rendered = (rendered / "bench-job.yaml").read_text()
    assert f"app={recipe_path.parent.name}-decode" in bench_rendered
    assert "Waiting for $SERVER_URL/health" in bench_rendered
    assert "Listing models at $SERVER_URL/v1/models" in bench_rendered
    assert "smoke_test_model" in bench_rendered
    assert EXPECTED_AIPERF in bench_rendered and "--unsafe-override" not in bench_rendered
    assert "Synthetic load (mode: synthetic)" in bench_rendered
    assert "generate_bounded_uniform_trace()" in bench_rendered
    assert "np.random.RandomState(seed)" in bench_rendered
    assert "tokenizer.apply_chat_template(" in bench_rendered
    assert '--custom-dataset-type single_turn' in bench_rendered
    assert '--input-file "$BOUNDED_TRACE"' in bench_rendered
    assert '--random-seed "$SYN_SEED"' in bench_rendered
    assert '--sequence-distribution' not in bench_rendered
    assert '--isl "$SYN_ISL"' not in bench_rendered
    assert "DATASET_PATH=" not in bench_rendered
    assert "name: NVLINK_MULTICAST_IMEX" in bench_rendered
    assert '"fabric_mnnvl_mode"' in bench_rendered

mnnvl_env = [
    {"name": "MC_FORCE_MNNVL", "value": "1"},
    {"name": "NCCL_MNNVL_ENABLE", "value": "1"},
]
gpu_deployment = {
    "kind": "Deployment",
    "spec": {
        "template": {
            "spec": {
                "containers": [
                    {
                        "name": "sglang",
                        "env": mnnvl_env,
                        "resources": {"requests": {"nvidia.com/gpu": "4"}},
                    }
                ]
            }
        }
    },
}
claimed, claim_count = capabilities.inject_imex_claim(gpu_deployment)
assert claim_count == 1
pod_spec = claimed["spec"]["template"]["spec"]
assert pod_spec["resourceClaims"][0]["resourceClaimTemplateName"] == "llmb-imex-channel"
assert pod_spec["containers"][0]["resources"]["claims"][0]["name"] == "imex-channel"

disabled, disabled_count = capabilities.strip_forced_flashinfer(
    {
        "kind": "Deployment",
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": "sglang",
                            "env": mnnvl_env,
                            "resources": {"requests": {"nvidia.com/gpu": "4"}},
                        }
                    ]
                }
            }
        },
    },
    "absent",
)
assert disabled_count == 2
values = {item["name"]: item["value"] for item in disabled["spec"]["template"]["spec"]["containers"][0]["env"]}
assert values["MC_FORCE_MNNVL"] == "0" and values["NCCL_MNNVL_ENABLE"] == "0"

node_cache = (STACK / "node-cache/glm5-fp8-4f96cc5e.yaml").read_text()
for expected in (
    "kind: DaemonSet",
    "kubernetes.io/arch: arm64",
    EXPECTED_HOST_PATH,
    "cp -a /fsx-hf-cache/. /node-hf-cache/",
    "test \"$shards\" -eq 142",
    ".llmb-cache-ready",
    "readinessProbe:",
):
    assert expected in node_cache, f"node-cache contract missing {expected}"

profile = (ROOT / "cluster-profiles/glm5-gb300.env.example").read_text()
for variable in (
    "KUBE_CONTEXT=",
    "GPU_PRODUCT=",
    "MODEL_CACHE_PVC=",
    "MODEL_CACHE_SUBPATH=",
    'NVLINK_MULTICAST_IMEX="provisioned"',
    "IMEX_CLAIM_TEMPLATE=",
    "RDMA_UCX_NET_DEVICES=",
    'RDMA_UCX_TLS="rc,rc_x,cuda_copy,cuda_ipc"',
    'RDMA_UCX_IB_ADDR_TYPE="eth"',
    'RDMA_UCX_IB_GPU_DIRECT_RDMA="no"',
    'RDMA_UCX_MAX_RNDV_RAILS="8"',
):
    assert variable in profile, f"GB300 profile missing {variable}"

nemotron_profile = (ROOT / "cluster-profiles/nemotron-ultra-gb300.env.example").read_text()
for variable in (
    "KUBE_CONTEXT=",
    "GPU_PRODUCT=",
    "MODEL_CACHE_PVC=",
    "RDMA_UCX_NET_DEVICES=",
    'RDMA_UCX_TLS="rc,rc_x,cuda_copy"',
    'RDMA_UCX_IB_ADDR_TYPE="eth"',
    'RDMA_UCX_IB_GPU_DIRECT_RDMA="try"',
    'RDMA_UCX_MAX_RNDV_RAILS="1"',
):
    assert variable in nemotron_profile, f"Nemotron GB300 profile missing {variable}"

merge_input = "kind: ConfigMap\nmetadata:\n  name: gb300-deepep-config\n"
merge_env = {**os.environ, "RUN_OWNER_NAME": "gb300-runowner-test", "RUN_OWNER_UID": "uid-test"}
merged = subprocess.run(
    [sys.executable, str(ROOT / "scripts/merge_run_owner.py")],
    input=merge_input,
    text=True,
    capture_output=True,
    env=merge_env,
    check=True,
).stdout
assert "ownerReferences" in merged and "uid: uid-test" in merged

print("GB300 release contract: 21 recipes, AWS NVMe, IMEX, topology, synthetic load, and cleanup OK")

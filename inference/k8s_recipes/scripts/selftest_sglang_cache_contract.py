#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa

"""Regression lock for NVBug 6586100 and CPERF-4254.

The fresh-install producer and SGLang consumers must agree on the Hugging Face
secret key, completion sentinel, and exact snapshot path. Synthetic benchmark
pods must not mount the serving weights PVC. Disaggregated frontends stage only
revision-pinned metadata into pod-local storage. Trace benchmark pods retain
the cache mount because their dataset lives there.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import model_cache  # noqa: E402

fails: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        fails.append(name)


install = (ROOT / "scripts/install.py").read_text()
download = (ROOT / "serving/download/templates/model-download.yaml.j2").read_text()
agg = (ROOT / "serving/sglang-agg/templates/server.yaml.j2").read_text()
disagg = (ROOT / "serving/sglang-disagg/templates/workers.yaml.j2").read_text()
disagg_frontend = (ROOT / "serving/sglang-disagg/templates/frontend.yaml.j2").read_text()
bench = (ROOT / "serving/aiperf/templates/bench-job.yaml.j2").read_text()

print("\n-- fresh-install secret contract --")
check("install creates the HF secret key 'token'", 'f"--from-literal=token={value}"' in install)
check("download consumes ${HF_SECRET}/token", "name: ${HF_SECRET}" in download and "key: token" in download)
for stack, text in (("sglang-agg", agg), ("sglang-disagg", disagg)):
    check(f"{stack} consumes ${{HF_SECRET}}/token", "name: ${HF_SECRET}" in text and "key: token" in text)
    check(f"{stack} has no hard-coded hf-token Secret name", "name: hf-token" not in text)

print("\n-- modern Hugging Face cache layout --")
for stack, text in (("sglang-agg", agg), ("sglang-disagg", disagg)):
    check(f"{stack} verifies the modern revision sentinel", ".llmb_download_done/{{ serving.model_revision }}" in text)
    check(
        f"{stack} launches from the pinned HF snapshot",
        "${MODEL_CACHE_SUBPATH}/hub/" in text and "model_repo_cache_dir" in text and "/snapshots/" in text,
    )
    check(f"{stack} no longer requires the legacy marker", ".k8s-download-complete" not in text)

render_cells = [
    ROOT / "recipes/llm-perf/Glm5/B200_k8s/Agg/16k_512/glm5-fp8-b200-sglang15-agg-c4-256",
    next(iter(sorted((ROOT / "recipes/llm-perf/Glm5/B200_k8s/Disagg/sglang_dynamo").glob("*/*")))),
]
for cell in render_cells:
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text())
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([str(ROOT / "scripts/render.sh"), str(cell), "--to", out], check=True, stdout=subprocess.DEVNULL)
        rendered = "\n".join(p.read_text() for p in Path(out).glob("*.yaml"))
    serving = recipe["serving"]
    expected_snapshot = "/model-store/" + model_cache.snapshot_dir(
        "${MODEL_CACHE_SUBPATH}", serving["model_repo"], serving["model_revision"]
    )
    expected_sentinel = "/model-store/" + model_cache.sentinel_path("${MODEL_CACHE_SUBPATH}", serving["model_revision"])
    check(f"{cell.name}: rendered snapshot equals canonical helper", expected_snapshot in rendered, expected_snapshot)
    check(f"{cell.name}: rendered sentinel equals canonical helper", expected_sentinel in rendered, expected_sentinel)

print("\n-- cache attachment follows workload data flow --")
check("disagg frontend has no model-cache PVC claim", "${MODEL_CACHE_PVC}" not in disagg_frontend)
check(
    "disagg frontend stages pinned HF metadata into pod-local storage",
    "stage-model-metadata" in disagg_frontend
    and "snapshot_download" in disagg_frontend
    and "serving.model_revision" in disagg_frontend
    and "name: model-metadata" in disagg_frontend
    and "emptyDir:" in disagg_frontend,
)
check(
    "disagg frontend metadata download excludes model weights",
    all(pattern in disagg_frontend for pattern in ('"*.safetensors"', '"*.bin"', '"*.pt"', '"*.pth"', '"*.gguf"')),
)
check(
    "disagg frontend registers the explicit served model from its pinned local metadata",
    "--model-name {{ serving.served_model" in disagg_frontend
    and "--model-path {{ model_store }}/${MODEL_CACHE_SUBPATH}/hub/{{ model_repo_cache_dir }}/snapshots/"
    in disagg_frontend,
)
check(
    "disagg frontend consumes the profile HF secret without hard-coding it",
    "name: ${HF_SECRET}" in disagg_frontend
    and "key: token" in disagg_frontend
    and "name: hf-token" not in disagg_frontend,
)
check("disagg workers retain the model-cache mount", "mountPath: /model-store" in disagg)
check("disagg workers retain the model-cache PVC claim", "claimName: ${MODEL_CACHE_PVC}" in disagg)
agg_cells = sorted((ROOT / "recipes/llm-perf/Glm5/B200_k8s/Agg").glob("*/*/recipe.yaml"))
check("three GLM aggregate cells discovered", len(agg_cells) == 3, str(len(agg_cells)))
for recipe_path in agg_cells:
    recipe = yaml.safe_load(recipe_path.read_text())
    check(
        f"{recipe_path.parent.name}: no longer forces same-node colocation",
        "colocate_with_server" not in recipe.get("bench", {}),
    )


def render_bench(cell: Path) -> str:
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([str(ROOT / "scripts/render.sh"), str(cell), "--to", out], check=True, stdout=subprocess.DEVNULL)
        return (Path(out) / "bench-job.yaml").read_text()


aiperf_cells = []
for recipe_path in sorted((ROOT / "recipes").glob("**/recipe.yaml")):
    recipe = yaml.safe_load(recipe_path.read_text())
    if recipe.get("envelope", {}).get("launcher") == "aiperf":
        aiperf_cells.append((recipe_path.parent, recipe["envelope"].get("mode")))
synthetic_cells = [cell for cell, mode in aiperf_cells if mode == "synthetic"]
trace_cells = [cell for cell, mode in aiperf_cells if mode != "synthetic"]
check("synthetic AIPerf cells discovered", bool(synthetic_cells), str(len(synthetic_cells)))
check("trace AIPerf cells discovered", bool(trace_cells), str(len(trace_cells)))

for cell in synthetic_cells:
    rendered_bench = render_bench(cell)
    check(f"{cell.name}: synthetic bench has no model-cache mount", "mountPath: /model-cache" not in rendered_bench)
    check(
        f"{cell.name}: synthetic bench has no model-cache PVC claim",
        "name: model-cache" not in rendered_bench and "MODEL_CACHE_MOUNT" not in rendered_bench,
    )
    check(f"{cell.name}: synthetic bench still mounts artifacts", "mountPath: /artifacts" in rendered_bench)

for cell in trace_cells:
    rendered_bench = render_bench(cell)
    check(
        f"{cell.name}: trace bench retains model-cache dataset mount",
        "mountPath: /model-cache" in rendered_bench and "name: MODEL_CACHE_MOUNT" in rendered_bench,
    )
    check(
        f"{cell.name}: trace bench retains the model-cache PVC claim",
        "name: model-cache" in rendered_bench and "claimName:" in rendered_bench,
    )
    check(
        f"{cell.name}: trace recipe render remains byte-identical",
        rendered_bench == (cell / "rendered/bench-job.yaml").read_text(),
    )

if fails:
    raise SystemExit(f"\n{len(fails)} failure(s): " + ", ".join(fails))
print("\nselftest_sglang_cache_contract: PASS")

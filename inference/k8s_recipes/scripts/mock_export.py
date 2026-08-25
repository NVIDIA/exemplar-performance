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

"""mock_export.py — generate fully-populated SAMPLE exports (docs/export-sample/<lane>.csv) via the REAL
export_dataset pipeline, so the DB-ingestion shape can be inspected before live smoke data lands.

Builds synthetic on-pipeline cells in a temp tree — one per lane the export must cover —
(llm-perf pareto · llm-perf max-concurrency-sla), each with realistic
record.json + runs/index.jsonl + curated/rungs.csv, then runs export_dataset.build over that tree. The point
is to exercise EVERY column, including the ones empty in the live export today (agent goodput/cost/median,
benchmark_valid, a real SLA crossing). Deterministic — all values hardcoded, no cluster, no I/O beyond the
temp tree + the committed sample."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_dataset as ed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# a realistic GB300 cluster fingerprint — what a run-time capture would write to run_meta.cluster_info.
CLUSTER_INFO_GB300 = {
    "provider": "aws",
    "region": "us-east-2",
    "cluster_id": "example-cluster",
    "environment": "non-prod",
    "gpu_product": "NVIDIA-GB300",
    "gpus_per_node": 4,
    "instance_type": "p6e-gb300.48xlarge",
    "k8s_version": "1.29",
    "node_count": 1,
}


def _write_cell(
    root: Path,
    rel: str,
    record: dict,
    rung_header: list,
    rung_rows: list,
    run: dict,
    cluster_info: dict,
):
    cell = root / "recipes" / rel
    curated = cell / "runs" / run["run_id"] / "curated"
    curated.mkdir(parents=True, exist_ok=True)
    (cell / "record.json").write_text(json.dumps(record, indent=2))
    # Minimal recipe.yaml so results_dir.py (used by export_dataset.build for the per-run path) can read the
    # envelope. Only scenario/goal/name/distribution/mode are needed; the cell's REAL metrics all come from
    # record.json + rungs.csv, so this stub never affects the emitted rows.
    ident = record.get("identity") or {}
    _goal = ident.get("goal")
    sv = (record.get("config") or {}).get("serving") or {}
    ln = (record.get("provenance") or {}).get("launcher") or {}
    # Top-level `serving:` block (sibling of envelope, exactly as a real recipe.yaml) so export_dataset's
    # _recipe_serving surfaces served_model_id/model_repo/model_revision/max_model_len for the sample.
    serving = {
        "stack": "vllm-agg",
        "model_repo": ln.get("model_repo"),
        "model_revision": "183968f87ae4cedce3039313cac1fd43d112c578",
        "served_model": ident.get("model"),
        "tp": sv.get("tp"),
        "max_model_len": 262144,
        "gpu_mem_util": 0.90,
        "extra_args": sv.get("extra_args"),
    }
    env = {
        "envelope": {
            "scenario": ident.get("scenario"),
            "goal": _goal,
            "name": Path(rel).name,
            "distribution": ident.get("distribution"),
            "mode": ident.get("mode"),
        },
        "serving": serving,
    }
    (cell / "recipe.yaml").write_text(json.dumps(env))  # JSON is valid YAML — no yaml dep needed here
    # A committed rendered/server.yaml with the RESOLVED exec command, so launch_command populates in the
    # sample (mirrors serving/vllm-agg/templates/server.yaml.j2 after Jinja bake).
    rendered = cell / "rendered"
    rendered.mkdir(parents=True, exist_ok=True)
    xargs = "".join(f"                {a} \\\n" for a in (sv.get("extra_args") or []))
    (rendered / "server.yaml").write_text(
        '          command: ["/bin/bash", "-lc"]\n'
        "          args:\n"
        "            - |\n"
        "              set -eu\n"
        f"              SNAPSHOT=$(python3 -c \"...snapshot_download(repo_id='{serving['model_repo']}')...\")\n"
        "              exec python3 -m vllm.entrypoints.openai.api_server \\\n"
        '                --model "$SNAPSHOT" \\\n'
        f"                --served-model-name {serving['served_model']} \\\n"
        f"                --tensor-parallel-size {serving['tp']} \\\n"
        f"                --max-model-len {serving['max_model_len']} \\\n"
        f"                --gpu-memory-utilization {serving['gpu_mem_util']} \\\n"
        "                --host 0.0.0.0 --port 8000 \\\n"
        f"{xargs}"
        "                --trust-remote-code\n"
    )
    (cell / "runs" / "index.jsonl").write_text(json.dumps(run) + "\n")
    gen = "aiperf"
    rm = {
        "run_id": run["run_id"],
        "cluster": run.get("cluster"),
        "cluster_info": cluster_info,
        f"{gen}_scenario": "sweep",
        f"{gen}_extra_inputs": "--warmup-requests 1",
    }
    rm["launcher_argv"] = [
        {
            "concurrency": c,
            "command": f"aiperf profile --url 'http://server:8000' --model 'nemotron-ultra-3' --streaming "
            f"--input-file '/model-cache/datasets/{record['identity']['distribution']}/dataset.jsonl' "
            f"--concurrency {c} --unsafe-override",
        }
        for c in (64, 512)
    ]
    (curated / "run_meta.json").write_text(json.dumps(rm, indent=2))
    lines = [",".join(rung_header)] + [",".join("" if c is None else str(c) for c in r) for r in rung_rows]
    (curated / "rungs.csv").write_text("\n".join(lines) + "\n")


def _base(
    scenario,
    goal,
    metric,
    unit,
    value,
    gpu,
    n,
    tp,
    mode,
    dist,
    extra_result=None,
    valid=None,
):
    rec = {
        "identity": {
            "scenario": scenario,
            "goal": goal,
            "model": "nemotron-ultra-3",
            "gpu_type": gpu,
            "gpu_count": n,
            "arch": "arm64",
            "engine": "vllm",
            "serving_mode": "aggregated",
            "distribution": dist,
            "mode": mode,
        },
        "fingerprint": {
            "benchmark_id": f"MOCK-bid-{scenario}-{goal or 'default'}",
            "recipe_hash": f"MOCK-rhash-{scenario}-{goal or 'default'}",
            "image_digest": None,
            "experiment_id": f"MOCK-exp-{gpu}",
            "git_ref": "mock/sample",
        },
        "provenance": {
            "run_id": "",
            "cluster": "",
            "image_digest": "sha256:MOCKimagedigest0000",
            "wall_seconds": 1800,
            "gpu_hours": round(n * 0.5, 2),
            "launcher": {
                "name": "aiperf",
                "ref": "git+https://github.com/ai-dynamo/aiperf.git@fef78a9",
                "model_repo": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
                "tokenizer_id": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
            },
        },
        "config": {
            "serving": {
                "engine": "vllm",
                "serving_mode": "aggregated",
                "tp": tp,
                "dp": None,
                "max_num_seqs": 512,
                "extra_args": [
                    "--kv-cache-dtype fp8",
                    "--enable-expert-parallel",
                    "--max-num-seqs 512",
                ],
            },
            "bench": {
                "sweep_mode": "fixed",
                "sweep_concurrency": [64, 512],
                "dataset": {"subpath": f"datasets/{dist}/dataset.jsonl"},
            },
        },
        "result": {
            "metric": metric,
            "unit": unit,
            "value": value,
            "reference": None,
            "tolerance_pct": 5,
        },
    }
    if extra_result:
        rec["result"].update(extra_result)
    if valid is not None:
        rec["benchmark_valid"] = valid
    return rec


def build_mock(root: Path):
    # ── lane 1: llm-perf pareto (archived, real kv) ──────────────────────────────────────────────────────
    rec = _base(
        "llm-perf",
        "pareto",
        "pareto_geomean",
        "geomean(tps/gpu·tps/user)",
        57.6,
        "GB300",
        4,
        4,
        "mooncake-trace",
        "example-long-context-trace",
        extra_result={
            "per_point": [
                {"concurrency": 64, "g": 75.1},
                {"concurrency": 512, "g": 44.2},
            ],
            "axes": ["tps_per_gpu", "tps_per_user"],
        },
    )
    hdr = [
        "concurrency",
        "ttft_p50_ms",
        "tpot_p50_ms",
        "tps_per_gpu",
        "tps_per_user",
        "error_rate_pct",
        "kv_cache_usage_perc",
    ]
    _write_cell(
        root,
        "llm-perf/256k/mock-gb300-pareto",
        rec,
        hdr,
        [
            [64, 667.58, 36.38, 205.26, 27.49, None, 7.70],
            [512, 1479.27, 145.79, 284.58, 6.86, 0.029, 23.31],
        ],
        {
            "run_id": "mockpareto1",
            "date": "2026-07-21",
            "cluster": "example-gb300-cluster",
            "data_provenance": "archived",
            "wall_seconds": 1800,
            "gpu_count": 4,
            "git_commit": "mock0001",
            "curated": "runs/mockpareto1/curated",
        },
        CLUSTER_INFO_GB300,
    )

    # ── lane 2: llm-perf max-concurrency-sla (archived, real interpolated crossing) ──────────────────────
    rec = _base(
        "llm-perf",
        "max-concurrency-sla",
        "max_concurrency_at_sla",
        "concurrency",
        216,
        "GB300",
        4,
        4,
        "mooncake-trace",
        "example-long-context-trace",
        extra_result={
            "sla": {"ttft_ms": 10000, "tpot_ms": 100, "stop_stat": "p50"},
            "crossing": {
                "conc": 216,
                "bracket": [128, 256],
                "ratio": 2.0,
                "binding": "TPOT",
                "status": "ok",
            },
            "value_source": "interpolated_sla_crossing",
        },
    )
    _write_cell(
        root,
        "llm-perf/256k/mock-gb300-sla",
        rec,
        hdr,
        [
            [128, 940.76, 69.89, 245.33, 14.31, None, 41.2],
            [256, 1322.50, 113.54, 275.22, 8.81, None, 63.8],
        ],
        {
            "run_id": "mocksla1",
            "date": "2026-07-21",
            "cluster": "example-gb300-cluster",
            "data_provenance": "archived",
            "wall_seconds": 1800,
            "gpu_count": 4,
            "git_commit": "mock0002",
            "curated": "runs/mocksla1/curated",
        },
        CLUSTER_INFO_GB300,
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mock_export_"))
    # DETERMINISM: run_by resolves from the environment (stored config → git email → $USER), so pin a fixed
    # sample identity via a throwaway user-config dir. Without this the committed sample would carry whoever
    # regenerated it. Also exercises the runner_identity STORAGE path end-to-end (save → export reads it).
    cfg = tmp / "userconfig"
    os.environ["LLMB_CONFIG_DIR"] = str(cfg)
    import runner_identity as _ri  # noqa: E402

    _ri.save_runner("benchmarker@example.com")
    try:
        build_mock(tmp)
        ed.build(tmp)  # writes per-run runs/<run_id>/llmb_inference_export.csv
        sample_dir = ROOT / "docs" / "export-sample"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True)
        # each mock cell has exactly ONE run → copy its runs/<id>/llmb_inference_export.csv, named by the cell's lane, so
        # the doc shows one fully-populated per-run table per (scenario, goal) — the exact publish-unit shape.
        cells = ed.on_pipeline_cells(tmp)
        groups = ed.group_map(cells)
        written = []
        for cell in cells:
            rows = ed._cell_rows(cell, groups)
            if not rows:
                continue
            # build() writes each run's llmb_inference_export.csv into the nested results/<scenario>/<goal>/<name>/<run_id>/
            # tree (results_dir.py), NOT under the cell's runs/ — read it from there.
            lk = ed.lane_key(*ed._lane_of(rows))
            run_id = rows[0]["run_id"]
            src = tmp / ed._rd.results_dir(cell, run_id) / ed.EXPORT_FILENAME
            if src.is_file():
                shutil.copyfile(src, sample_dir / f"{lk}.csv")
                written.append(f"{lk}.csv")
        print(
            f"[mock_export] wrote {len(written)} per-lane sample(s) → docs/export-sample/: "
            f"{', '.join(sorted(written))}"
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

# DeepSeek-V4-Flash inference recipes — disaggregated (SGLang Dynamo)

LLMB-native inference benchmarks for DSV4F (mxfp4) on GB200 and GB300. Uses
disaggregated serving with CPU KV-cache offload on the prefill worker.

## At a glance

| Field                   | Value                                                          |
| ----------------------- | -------------------------------------------------------------- |
| **Workload keys**       | `inference_dsv4f_sglang_gb200`, `inference_dsv4f_sglang_gb300` |
| **Model**               | `deepseek-ai/DeepSeek-V4-Flash-0731` (304B)                    |
| **Serving container**   | `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1`                |
| **Benchmark container** | `nvcr.io/nvidia/ai-dynamo/aiperf:0.12.0`                       |
| **SKUs**                | GB200, GB300 (separate workloads)                              |
| **Backend**             | SGLang Dynamo, disaggregated serving                           |
| **Override env var**    | `RUN_CONF_DSV4F_RECIPE`                                        |

See the [Slurm serving README](../README.md) for shared installation,
submission, and result conventions.

## Prepare environment

Use the [main README's installer](../../../../README.md#quick-start-guide) and
select the workload matching your cluster's GPU type. Installation downloads
the pinned model and the gated
[`nv-camilom/agentic_coding` dataset](https://huggingface.co/datasets/nv-camilom/agentic_coding).
Provide an `HF_TOKEN` with dataset access during installation. LLMB unpacks the
dataset into `$LLMB_WORKLOAD/dataset/agentic_coding` and mounts it in the
benchmark container at `/datasets/agentic_coding`.

`LLMB_INSTALL` is the installation root; `LLMB_WORKLOAD` is
`$LLMB_INSTALL/workloads/<workload-key>`. Results are written under
`$LLMB_WORKLOAD/experiments/`.

## Run benchmark

From the installation directory, submit the workload for your GPU type:

```bash
cd "$LLMB_INSTALL"
llmb-run submit -w inference_dsv4f_sglang_gb200 -s 304b --dtype mxfp4 --scale 8
llmb-run submit -w inference_dsv4f_sglang_gb300 -s 304b --dtype mxfp4 --scale 8
```

Submit only the command matching the installed workload.

## Default recipe variants

| GPU   | Recipe path relative to the workload's `sglang_dynamo/` directory | Scale |
| ----- | ----------------------------------------------------------------- | ----: |
| GB200 | `recipes/dsv4f-sgl-disagg-gb200-1p1d-dep4-tp4-mtp-hicache.yaml`   |     8 |
| GB300 | `recipes/dsv4f-sgl-disagg-gb300-1p1d-dep4-tp4-mtp-hicache.yaml`   |     8 |

Each recipe allocates two nodes with four GPUs each: one prefill worker and
one decode worker. Dynamo routes requests using KV-cache events, Mooncake
transfers KV data between workers, and EAGLE provides speculative decoding.
Prefill uses HiCache with CPU offload. On both GPUs the prefill `mem-fraction-static` setting
is tweaked to reduce available GPU KV cache space and force more KV cache offload activity in order to observe its performance.

AIPerf uses the dataset to create many concurrent multi-turn, high-reuse agentic sessions with modeled tool delays. The GB300 recipe sweeps through concurrency 128, 256, and 512, running a fixed-duration benchmark at each point. The GB200 recipe does likewise with concurrency 128 and 256 (concurrency 512 will be enabled on a later release).

## Switching recipe variants

Set `RUN_CONF_DSV4F_RECIPE` to a path relative to the installed workload's
`sglang_dynamo/` directory. For example, to select the GB300 recipe explicitly:

```bash
export RUN_CONF_DSV4F_RECIPE="recipes/dsv4f-sgl-disagg-gb300-1p1d-dep4-tp4-mtp-hicache.yaml"
llmb-run submit -w inference_dsv4f_sglang_gb300 -s 304b --dtype mxfp4 --scale 8
```

The shipped workloads each have one validated recipe. Keep `--scale` equal to
the selected recipe's physical GPU count. To persist an override across
submissions, set it in `cluster_config.yaml`'s `environment:` block.

## Where artifacts and logs land

- **Model weights:** `$LLMB_WORKLOAD/deepseek-v4-flash-0731/`
- **Dataset:** `$LLMB_WORKLOAD/dataset/agentic_coding/`
- **Serving image:** `$LLMB_INSTALL/images/dynamo-sglang-1.5.1.sqsh`
- **Recipes:** `$LLMB_INSTALL/llmb_repo/inference/slurm_recipes/llm-serving/dsv4/disagg/<gpu>/sglang_dynamo/recipes/`
- **Benchmark artifacts:** `$LLMB_WORKLOAD/experiments/<configuration>/<timestamp>/<jobid>/logs/`
- **Benchmark numbers:** `<jobdir>/logs/benchmark-rollup.csv`

## Monitoring after submit

```bash
squeue --me
sacct -X -j <benchmark-jobid> --format=JobID,JobName,State,Elapsed,NNodes -P
llmb-run jobs
```

## Reading benchmark results

The
per-concurrency AIPerf exports are in the benchmark artifacts location under `logs/aiperf_<concurrency>/`.
The benchmark combines those exports into
`logs/benchmark-rollup.csv`, with one row per concurrency.

Key output columns
include `Total Token Throughput` and `Output Token Throughput per User`, in tokens per second.

## Troubleshooting

Run `llmb-run jobs` to refresh job state and `llmb-run jobs log <job-id>` to
read the orchestration log. For serving or benchmark failures, inspect the
experiment's `logs/` directory. If installation cannot access the dataset,
check that `HF_TOKEN` has access to `nv-camilom/agentic_coding`.

Very occasional errors due to generated requests exceeding model context limits may occur,
however the benchmark will check the reported request error rate and fail if it is excessive,
pointing to misconfiguration or server failure during the test.

# Kimi-K2.6 inference recipe — disaggregated (TRT-LLM Dynamo)

LLMB-native recipe for the Kimi-K2.6 NVFP4 inference benchmark on gb200, in
disaggregated-serving topology.

## At a glance

| Field                | Value                                                      |
| -------------------- | ---------------------------------------------------------- |
| **Workload key**     | `inference_kimi2.6`                                        |
| **Model**            | `nvidia/Kimi-K2.6-NVFP4` (1T total, 32B active — MoE)      |
| **Container**        | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.2` |
| **SKU**              | gb200 only                                                 |
| **Backend**          | TRT-LLM Dynamo, disaggregated serving                      |
| **Override env var** | `RUN_CONF_KIMI26_RECIPE`                                   |

See the [umbrella README](../../../../README.md) for the shared installation,
submission, results, and conventions reference.

## Prepare environment

Use the **installer** referenced in the [main README](../../../../../../../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/inference_kimi2.6`.
- Benchmark outputs and logs are collected under `${LLMB_WORKLOAD}/experiments/`.

# Run benchmark

## Using llmb-run (Recommended)

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Submit the benchmark
llmb-run submit -w inference_kimi2.6 -s 1t --dtype nvfp4 --scale 24
```

## Default recipe variant

```
recipes/gb200_nvfp4/ISL8K_OSL1K/low_latency_conc_1_4_16_32_64.yaml
```

Topology (6 nodes total on gb200, 4 GPUs/node = 24 GPUs):

- **Prefill**: 1 worker
- **Decode**: 5 workers
- **ISL/OSL**: 8k / 1k
- **Concurrency sweep**: 1, 4, 16, 32, 64 (low-latency end of the curve)
- **Observed wall time**: ~30–45 minutes

## All recipe variants

Five LLMB-owned variants are available under `recipes/` in LLMB's installed repository copy:

| Relative path                                                            | ISL/OSL  | Concurrency         | Allocated GPUs / `--scale` |
| ------------------------------------------------------------------------ | -------- | ------------------- | -------------------------- |
| `gb200_nvfp4/ISL8K_OSL1K/low_latency_conc_1_4_16_32_64.yaml`             | 8k / 1k  | 1, 4, 16, 32, 64    | 24                         |
| `gb200_nvfp4/ISL8K_OSL1K/high_pareto_conc_128_256_512_1024.yaml`         | 8k / 1k  | 128, 256, 512, 1024 | 28                         |
| `gb200_nvfp4/ISL32K_OSL4K/low_latency_conc_1_4_16_32_64.yaml`            | 32k / 4k | 1, 4, 16, 32, 64    | 24                         |
| `gb200_nvfp4/ISL32K_OSL4K/high_pareto_conc_128_256_512.yaml`             | 32k / 4k | 128, 256, 512       | 40                         |
| `gb200_nvfp4/ISL32K_OSL4K/high_pareto_conc_1024_ctx4dep8_gen1dep16.yaml` | 32k / 4k | 1024                | 48                         |

Browse what's available on your install:

```bash
find "$LLMB_INSTALL/llmb_repo/inference/slurm_recipes/llm-serving/kimi2.6/disagg/gb200/trtllm_dynamo/recipes" \
  -type f -name '*.yaml' -print
```

## Switching recipe variants

Override the default via env var (path relative to `trtllm_dynamo/`):

```bash
export RUN_CONF_KIMI26_RECIPE="recipes/gb200_nvfp4/ISL32K_OSL4K/high_pareto_conc_1024_ctx4dep8_gen1dep16.yaml"
llmb-run submit -w inference_kimi2.6 -s 1t --dtype nvfp4 --scale 48
```

The ISL32K_OSL4K long-context variants take substantially longer to run than
the ISL8K_OSL1K ones. If a run ends
in `TIMEOUT`, extend `default_time_limit` in
`$LLMB_WORKLOAD/srt-slurm/srtslurm.yaml` and resubmit.

To make the override sticky across submits, add it to `cluster_config.yaml`'s
`environment:` block at install time (or post-install by editing the file).

## Where artifacts and logs land

- **Model weights**: `$LLMB_WORKLOAD/kimi26-nvfp4/`
- **Container `.sqsh`**: `$LLMB_INSTALL/images/trtllm-runtime-1.1.0-dev.2.sqsh`
- **Recipes**: `$LLMB_INSTALL/llmb_repo/inference/slurm_recipes/llm-serving/kimi2.6/disagg/gb200/trtllm_dynamo/recipes/`
- **Benchmark artifacts**: `$LLMB_WORKLOAD/experiments/<configuration>/<timestamp>/<jobid>/`
- **Benchmark numbers**: `<jobdir>/logs/benchmark.out`

`$LLMB_WORKLOAD` = `$LLMB_INSTALL/workloads/inference_kimi2.6`

## Monitoring after submit

```bash
squeue --me
sacct -X -j <benchmark-jobid> --format=JobID,JobName,State,Elapsed,NNodes -P
llmb-run jobs
```

## Reading benchmark results

After a successful run, `benchmark.out` lands at:

```text
$LLMB_WORKLOAD/experiments/<configuration>/<timestamp>/<jobid>/logs/benchmark.out
```

`$LLMB_WORKLOAD` = `$LLMB_INSTALL/workloads/inference_kimi2.6`.

The file contains **two blocks per concurrency** in the sweep — a **warmup**
block (smaller `Successful requests`, ignore it) followed by the actual
**measured** block (larger `Successful requests`). The default recipe
(`low_latency_conc_1_4_16_32_64.yaml`) sweeps concurrency 1, 4, 16, 32, 64 —
so expect 10 blocks total (5 warmup + 5 measured). The warmup uses
`2 × concurrency` prompts, and the measured run uses `10 × concurrency`
prompts. For the default sweep, the warmup/measured prompt counts are 2/10,
8/40, 32/160, 64/320, and 128/640, respectively.

A redacted measured-block example:

```text
Maximum request concurrency: ****
============ Serving Benchmark Result ============
Successful requests:                     ****
Benchmark duration (s):                  ****
Total input tokens:                      ****
Total generated tokens:                  ****
Request throughput (req/s):              ****
Output token throughput (tok/s):         ****        ← key metric
Peak output token throughput (tok/s):    ****
Total Token throughput (tok/s):          ****        ← key metric
---------------Time to First Token----------------
Mean TTFT (ms):                          ****
Median TTFT (ms):                        ****
P99 TTFT (ms):                           ****
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          ****
Median TPOT (ms):                        ****        ← key metric
P99 TPOT (ms):                           ****
==================================================
```

The three key metrics:

| Metric                              | Meaning                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| **Output token throughput (tok/s)** | Output tokens generated per second, aggregated across all concurrent requests |
| **Total Token throughput (tok/s)**  | (Input + output) tokens per second, aggregated across all concurrent requests |
| **Median TPOT (ms)**                | Median time per output token, excluding the first                             |

## Troubleshooting

Start with `llmb-run jobs` to refresh the recorded state and
`llmb-run jobs log <job-id>` to read the orchestration log. For backend or
benchmark failures, inspect the files under the experiment's `logs/` directory.
If a run reaches the configured Slurm time limit, extend `default_time_limit`
in `$LLMB_WORKLOAD/srt-slurm/srtslurm.yaml` and resubmit.

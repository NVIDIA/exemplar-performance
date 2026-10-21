# GLM5 inference recipe — disaggregated (TRT-LLM Dynamo)

LLMB-native recipe for the GLM-5 NVFP4 inference benchmark on gb300, in
disaggregated-serving topology.

## At a glance

| Field                | Value                                                      |
| -------------------- | ---------------------------------------------------------- |
| **Workload key**     | `inference_glm5`                                           |
| **Model**            | `nvidia/GLM-5-NVFP4` (744B total, 40B active — MoE)        |
| **Container**        | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.3` |
| **SKU**              | gb300 only                                                 |
| **Backend**          | TRT-LLM Dynamo, disaggregated serving                      |
| **Override env var** | `RUN_CONF_GLM5_RECIPE`                                     |

See the [umbrella README](../../../../README.md) for the shared installation,
submission, results, and conventions reference.

## Prepare environment

Use the **installer** referenced in the [main README](../../../../../../../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/inference_glm5`.
- Benchmark outputs and logs are collected under `${LLMB_WORKLOAD}/experiments/`.

# Run benchmark

## Using llmb-run (Recommended)

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Submit the benchmark
llmb-run submit -w inference_glm5 -s 744b --dtype nvfp4 --scale 32
```

## Default recipe variant

```
recipes/8k_1k/ctx4dep2_gen3tep8_batch64_eplb0_mtp0_conc_64_128_256.yaml
```

Topology (8 nodes total on gb300, 4 GPUs/node = 32 GPUs):

- **Prefill**: 4 workers × TP2/EP2 = 8 GPUs across 2 nodes
- **Decode**: 3 workers × TP8/EP8 = 24 GPUs across 6 nodes
- **ISL/OSL**: 8k / 1k
- **Concurrency sweep**: 64, 128, 256
- **Observed wall time**: ~50 minutes

## All recipe variants

Six LLMB-owned variants are available under `recipes/` in LLMB's installed repository copy:

| Relative path                                                      | ISL/OSL  | Concurrency  | Allocated GPUs / `--scale` |
| ------------------------------------------------------------------ | -------- | ------------ | -------------------------- |
| `8k_1k/ctx1dep2_gen4tep8_batch1_eplb0_mtp0_conc_1_4.yaml`          | 8k / 1k  | 1, 4         | 36                         |
| `8k_1k/ctx4dep2_gen3tep8_batch64_eplb0_mtp0_conc_64_128_256.yaml`  | 8k / 1k  | 64, 128, 256 | 32                         |
| `8k_1k/ctx12dep2_gen1dep16_batch64_eplb0_mtp0_conc_512_1024.yaml`  | 8k / 1k  | 512, 1024    | 40                         |
| `32k_4k/ctx1dep2_gen4tep8_batch1_eplb0_mtp0_conc_1_4.yaml`         | 32k / 4k | 1, 4         | 36                         |
| `32k_4k/ctx4dep2_gen3tep8_batch64_eplb0_mtp0_conc_64_128_256.yaml` | 32k / 4k | 64, 128, 256 | 32                         |
| `32k_4k/ctx12dep2_gen1dep16_batch64_eplb0_mtp0_conc_512_1024.yaml` | 32k / 4k | 512, 1024    | 40                         |

Browse what's available on your install:

```bash
find "$LLMB_INSTALL/llmb_repo/inference/slurm_recipes/llm-serving/glm5/disagg/gb300/trtllm_dynamo/recipes" \
  -type f -name '*.yaml' -print
```

## Switching recipe variants

Override the default via env var (path relative to `trtllm_dynamo/`):

```bash
export RUN_CONF_GLM5_RECIPE="recipes/8k_1k/ctx12dep2_gen1dep16_batch64_eplb0_mtp0_conc_512_1024.yaml"
llmb-run submit -w inference_glm5 -s 744b --dtype nvfp4 --scale 40
```

The 32k_4k long-context variants take substantially longer to run than the
8k_1k ones. If a run ends in
`TIMEOUT`, extend `default_time_limit` in
`$LLMB_WORKLOAD/srt-slurm/srtslurm.yaml` and resubmit.

To make the override sticky across submits, add it to `cluster_config.yaml`'s
`environment:` block at install time (or post-install by editing the file).

## Where artifacts and logs land

- **Model weights**: `$LLMB_WORKLOAD/glm5-nvfp4/`
- **Container `.sqsh`**: `$LLMB_INSTALL/images/trtllm-runtime-1.1.0-dev.3.sqsh`
- **Recipes**: `$LLMB_INSTALL/llmb_repo/inference/slurm_recipes/llm-serving/glm5/disagg/gb300/trtllm_dynamo/recipes/`
- **Benchmark artifacts**: `$LLMB_WORKLOAD/experiments/<configuration>/<timestamp>/<jobid>/`
- **Benchmark numbers**: `<jobdir>/logs/benchmark.out`

`$LLMB_WORKLOAD` = `$LLMB_INSTALL/workloads/inference_glm5`

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

`$LLMB_WORKLOAD` = `$LLMB_INSTALL/workloads/inference_glm5`.

The file contains **two blocks per concurrency** in the sweep — a **warmup**
block (smaller `Successful requests`, ignore it) followed by the actual
**measured** block (larger `Successful requests`). The default recipe sweeps
concurrency 64, 128, 256 — so expect 6 blocks total (3 warmup + 3 measured).
The warmup uses `2 × concurrency` prompts, and the measured run uses
`16 × concurrency` prompts. For the default sweep, that means 128/1024,
256/2048, and 512/4096 warmup/measured prompts, respectively.

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

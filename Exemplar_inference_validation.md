# Exemplar Inference Performance Validation Guide

Do you want to evaluate whether your infrastructure is configured to deliver expected performance for agentic workloads?

This guide helps NVIDIA Cloud Partners run standardized agentic inference workloads in their environments, compare the results with NVIDIA baselines, and identify tuning opportunities.

> **This is a self-check, not a certification.** Following this guide does not grant certified status, and there is no pass/fail threshold. It is a *separate* program from the pre-training [Exemplar Cloud certification](Exemplar_training_validation.md) (which does have a >95% baseline bar). Here, you run the workloads, then reach out to your NVIDIA account team for baseline results to compare. We’re happy to help tune as well.

You can run on **Slurm** or **Kubernetes**. Coverage will expand to additional GPU platforms and workloads over time.

## Start with system checks and microbenchmarks

**Run these checks first, before committing GPU capacity to the inference recipes.** Check the CPU/NUMA configuration, GPU health and visibility, and GPU/network communication on the nodes you plan to use.

| Component                   | Recipe                                                                           | What it measures                                                                                                                              |
| --------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Host, GPU and network setup | [System information checks](microbenchmarks/system_info/README.md)               | CPU/NUMA configuration, GPU health and visibility, NVLink/fabric and network readiness; Slurm container and scheduler checks where applicable |
| CPU overhead                | [Kernel-launch latency and tokenization](microbenchmarks/cpu_overhead/README.md) | CPU costs affecting inference request processing                                                                                              |
| GPU communication           | [NCCL](microbenchmarks/nccl/README.md)                                           | Collective communication bandwidth and latency across GPUs and nodes                                                                          |
| System configuration report | [LLMB collector (optional)](microbenchmarks/llmb_collector/README.md)            | Collects CPU, GPU, memory, storage, software, network and environment information into `llmb-collector.json`                                  |

Slurm automation is available for all four recipes. For Kubernetes, run the [universal host checks](microbenchmarks/system_info/README.md#universal-host-checks) manually on the compute hosts. No Kubernetes support is provided yet for LLMB collector, CPU overhead or NCCL.

## Inference recipe coverage

### Where each recipe runs

Each ✓ links to the recipe for that chip and scheduler. A blank cell means no recipe is included for that combination yet.

| Recipe                                     | GB300 Slurm                                                                                   | GB300 Kubernetes                                                                                  | GB200 Slurm                                                                           | B200 Kubernetes                                                                                  |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| GLM-5 FP8                                  | [✓](inference/slurm_recipes/llm-serving/glm5/disagg/gb300/sglang_dynamo/README.md)            | [✓](inference/k8s_recipes/recipes/llm-perf/Glm5/Gb300_k8s/README.md)                              |                                                                                       | [✓](inference/k8s_recipes/recipes/llm-perf/Glm5/B200_k8s/)                                       |
| GLM-5 NVFP4                                | [✓](inference/slurm_recipes/llm-serving/glm5/disagg/gb300/trtllm_dynamo/README.md)            |                                                                                                   |                                                                                       |                                                                                                  |
| Kimi-K2.6 NVFP4                            |                                                                                               |                                                                                                   | [✓](inference/slurm_recipes/llm-serving/kimi2.6/disagg/gb200/trtllm_dynamo/README.md) |                                                                                                  |
| DeepSeek-V4-Flash MXFP4                    | [✓](inference/slurm_recipes/llm-serving/dsv4/README.md)                                       |                                                                                                   | [✓](inference/slurm_recipes/llm-serving/dsv4/README.md)                               |                                                                                                  |
| Qwen3-32B BF16                             |                                                                                               |                                                                                                   | [✓](inference/slurm_recipes/long-context/qwen3/agg/sglang/README.md)                  |                                                                                                  |
| Nemotron 3 Ultra NVFP4                     | [✓](inference/slurm_recipes/llm-serving/nemotron3_ultra/disagg/gb300/sglang_dynamo/README.md) | [✓](inference/k8s_recipes/recipes/llm-perf/nemotron_ultra_disagg/Gb300_k8s/Disagg/sglang_dynamo/) |                                                                                       | [✓](inference/k8s_recipes/recipes/llm-perf/nemotron_ultra_disagg/B200_k8s/Disagg/sglang_dynamo/) |
| TaskTrace CPU-only                         | [✓](inference/slurm_recipes/agentic/README.md)                                                | [✓](inference/k8s_recipes/recipes/agentic/)                                                       | [✓](inference/slurm_recipes/agentic/README.md)                                        |                                                                                                  |
| TaskTrace CPU+GPU (Nemotron 3 Ultra NVFP4) | [✓](inference/slurm_recipes/agentic/README.md)                                                | [✓](inference/k8s_recipes/recipes/agentic/)                                                       | [✓](inference/slurm_recipes/agentic/README.md)                                        |                                                                                                  |

### What each recipe tests

| Recipe                                     | Deployment                                                                                | Test scenarios                                                                                                                                                                       |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GLM-5 FP8                                  | disaggregated; SGLang + Dynamo. B200 Kubernetes also includes aggregated SGLang           | Synthetic 1k/1k, 8k/1k and 16k/512 ISL/OSL profiles with low-latency and high-throughput variants, swept across concurrency up to 5,700                                              |
| GLM-5 NVFP4                                | disaggregated; TRT-LLM + Dynamo                                                           | Synthetic 8k/1k and 32k/4k ISL/OSL profiles with low-latency, mid-range and high-throughput variants, swept across concurrency up to 1,024                                           |
| Kimi-K2.6 NVFP4                            | disaggregated; TRT-LLM + Dynamo                                                           | Synthetic 8k/1k and 32k/4k ISL/OSL profiles with low-latency and high-throughput variants, swept across concurrency up to 1,024                                                      |
| DeepSeek-V4-Flash MXFP4                    | disaggregated; SGLang + Dynamo                                                            | Multi-turn agentic-coding session traces with CPU KV-cache offload, swept across concurrency 128, 256 and 512                                                                        |
| Qwen3-32B BF16                             | aggregated; SGLang                                                                        | Multi-turn agentic-coding session traces with CPU KV-cache offload, swept across concurrency 4 to 64                                                                                 |
| Nemotron 3 Ultra NVFP4                     | disaggregated; SGLang + Dynamo                                                            | Synthetic 10k/16k, 50k/2k and 8k/64k ISL/OSL profiles across deployment sizes, swept across concurrency up to 32 on GB300 and up to 256 on B200 Kubernetes, depending on the profile |
| TaskTrace CPU-only                         | no local model; mini-swe-agent + TaskTrace on Grace CPUs                                  | SWE-bench task: replay recorded tool calls, or capture a live run against an external model endpoint and then replay it                                                              |
| TaskTrace CPU+GPU (Nemotron 3 Ultra NVFP4) | aggregated; vLLM serving Nemotron 3 Ultra NVFP4; mini-swe-agent + TaskTrace on Grace CPUs | SWE-bench task: replay recorded tool and LLM calls, or capture a live run against the local model and then replay it                                                                 |

## Stages of inference validation

### Kickoff with NVIDIA

Contact your NVIDIA account team to agree on the recipes, configurations and baseline results relevant to your platform. You can also run the available recipes independently for performance checks and tuning.

### Prepare a benchmark cluster

1. For Slurm, follow the [shared prerequisites](README.md#prerequisites) and [installation guide](cli/llmb-install/README.md). For Kubernetes, follow the [cluster setup guide](inference/k8s_recipes/README.md#getting-started).
2. Identify the candidate compute nodes and run the applicable [system checks and microbenchmarks](#start-with-system-checks-and-microbenchmarks). Review system-check warnings, compare CPU-overhead and NCCL results with the agreed baselines, and fix any configuration issues before running inference recipes.
3. Select the [inference recipes for your chip and scheduler](#where-each-recipe-runs) and follow each recipe's prerequisites.
4. Retain the same repository version for the readiness checks and inference validation batch. Record that revision and the selected configurations with your results.

### Run inference recipes

After reviewing the readiness results, run the selected configurations. Follow each recipe's instructions: Slurm uses `llmb-run submit`; Kubernetes uses `llmb-k8s install` and `llmb-k8s run`. Run one configuration at a time and retain its outputs with the readiness-check results.

### Verify and share results

Send the selected recipe configurations, repository revision, benchmark outputs and relevant logs to your NVIDIA account team for comparison with expected baseline performance. Follow each recipe's output instructions: GLM-5, Kimi, DeepSeek-V4-Flash and Nemotron 3 Ultra Slurm serving results are under managed `experiments/` directories; Qwen long-context outputs remain under `srt-slurm/outputs/`. TaskTrace writes `summary.json` in the experiment directory, and Kubernetes results are fetched by `llmb-k8s run`, or by `llmb-k8s collect <run-id>` for detached runs.

Use the [LLMB archive command](cli/llmb-run/README.md#archive-command) to package your Slurm results. The archive doesn't include Qwen long-context outputs or Kubernetes results, so send those separately. Share profiling files if requested.

### Optimize with NVIDIA

Work with NVIDIA performance experts to investigate tuning opportunities and rerun the affected recipes using the same version and configurations for comparison.

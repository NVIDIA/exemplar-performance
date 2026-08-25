# Agentic Inference Long-Context Benchmark

Benchmarks the LLM side of an **agentic coding workload** as a performance test.
Handling many concurrent sessions with long context, many turns, and tool delays
requires KV-cache management and offload to maintain efficiency.

- **Model**: Qwen3-32B (bf16) configured for 128k context
- **Backend**: sglang aggregated config with hierarchical cache (hicache) enabled
  for offloading to CPU
- **Driver**: srt-slurm
- **Benchmarking tool**: [AIPerf](https://github.com/ai-dynamo/aiperf) agentic
  coding test scenario. This scenario runs a fixed-time test,
  while maintaining a fixed number of concurrent multi-turn agentic sessions,
  creating requests based on Wekka-style session traces with "tool call" delays.
- **Dataset**: custom dataset built from a collection of agentic-coding benchmark
  execution trajectories.
- **SKU**: gb200

## Install

Use the installer referenced in the [main README](../../../../../../README.md).

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/inference_qwen3_long`.
- Benchmark outputs and logs are stored under `${LLMB_WORKLOAD}/srt-slurm/outputs/<job-id>/` (see "Where outputs land" below).

## Run

```bash
cd $LLMB_INSTALL
llmb-run submit -w inference_qwen3_long --scale 1
```

The script runs a sweep over session concurrencies (4,8,16,24,32,48,64), as separate slurm jobs,
one node each. The jobs currently take about 1 hour to run.

Currently AIPerf is pulled from github at run time, so the compute nodes need
network access.

### Choosing the concurrency sweep

Override the default list before submitting:

```bash
export RUN_CONF_AGENTIC_SESSION_CONCURRENCIES="32"          # single point
export RUN_CONF_AGENTIC_SESSION_CONCURRENCIES="4 8 16 32"   # custom sweep
llmb-run submit -w inference_qwen3_long --scale 1
```

## Where outputs land

Each concurrency's benchmark writes under
`$LLMB_WORKLOAD/srt-slurm/outputs/<benchmark-jobid>/`:

```text
outputs/
└── <jobid>/
    ├── config.yaml                  # rendered recipe for this run
    └── logs/
        └── aiperf/
            └── aggregate/profile_export_aiperf_aggregate.csv
```

## Viewing results

Once the sweep finishes, render the summary table (concurrency, p50 request
latency, effective total throughput) with the bundled helper:

```bash
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/inference_qwen3_long
./report.sh
```

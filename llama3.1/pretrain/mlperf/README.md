# Overview

This recipe contains information and scripts to produce performance results for the Llama 3.1 8B MLPerf Training workload. It uses the Megatron-Bridge Llama 3.1 8B performance recipe and supports benchmark runs on GB300 and GB200 systems.

**Supported model size:**

- `8b` — Llama 3.1 8B

The supported configurations are fixed MLPerf-oriented presets.

## GB300

| Precision | GPUs | Variant | SeqLen | Layers |  TP |  PP |  CP |  DP | VP  | MBS | GBS |  GA |
| :-------- | ---: | :-----: | -----: | -----: | --: | --: | --: | --: | :-- | --: | --: | --: |
| NVFP4     |    8 |   v1    |   8192 |     32 |   1 |   1 |   1 |   8 | N/A |   2 |  16 |   1 |
| NVFP4     |   72 |   v2    |   8192 |     32 |   1 |   1 |   1 |  72 | N/A |   1 |  72 |   1 |

## GB200

| Precision | GPUs | Variant | SeqLen | Layers |  TP |  PP |  CP |  DP | VP  | MBS | GBS |  GA |
| :-------- | ---: | :-----: | -----: | -----: | --: | --: | --: | --: | :-- | --: | --: | --: |
| NVFP4     |    8 |   v1    |   8192 |     32 |   1 |   1 |   1 |   8 | N/A |   2 |  16 |   1 |
| NVFP4     |   72 |   v2    |   8192 |     32 |   1 |   1 |   1 |  72 | N/A |   1 |  72 |   1 |

# Performance Measurement and Analysis

Performance is reported as:

- `s/iter` — wall-clock seconds per training step
- `TFLOPS/GPU` — sustained FLOPS achieved per GPU

Each benchmark runs 50 steps by default. Iterations 35–44 are averaged to skip warmup effects such as input prefetch, activation allocation, and JIT compilation.

## Viewing results with `llmb-run jobs`

Each `llmb-run jobs` command refreshes Slurm state and parses the training log for jobs that have finished. Run the commands from `$LLMB_INSTALL`:

```bash
# List submitted jobs and parsed metrics
llmb-run jobs

# Show full details for one job
llmb-run jobs show <job_id>

# Open the training log
llmb-run jobs log <job_id>
```

Blank `s/iter` or `TFLOPS/GPU` fields mean that the job has not finished or the log does not contain enough completed iterations. See the [llmb-run README](../../../cli/llmb-run/README.md#jobs-command) for the full command reference.

## Derived metrics

To convert step time into tokens per second:

```text
throughput (tokens/sec) = sequence length * global batch size / s/iter
```

To estimate time-to-train for a target token budget:

```text
time to train (days) = total tokens / throughput (tokens/sec) / 86400
```

To compute model FLOPs utilization (MFU):

```text
MFU = TFLOPS/GPU / peak GPU FLOPS
```

For peak theoretical throughput values, see [Peak Theoretical Throughput](../../../README.md#peak-theoretical-throughput) in the main README.

# Prerequisites

A HuggingFace account is required. Create a [HuggingFace access token](https://huggingface.co/settings/tokens) and add it to your environment:

```bash
export HF_TOKEN=<your_token>
```

Python 3.12.x or conda is also required.

## Request Access

Request Llama 3.1 access through [Meta's website](https://www.llama.com/llama-downloads/) and then request access to [Llama 3.1 8B on HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B). Approval is not automatic and may take a day or more.

## Slurm

The commands in this document use Slurm. The following settings vary by cluster; consult your system administrator for the correct values:

- `SBATCH_PARTITION` or `-p` — partition or queue
- `SBATCH_ACCOUNT` or `-A` — Slurm accounting account
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<count>` — GPUs per node when the cluster uses GRES

Errors such as `GPUs not found` or `Cannot submit to this partition without GPU resources` usually mean that a GPU resource setting is required.

## Prepare Environment

Use the installer described in the [main README](../../../README.md) to prepare the recipe environment.

The recipe uses these directories and variables:

- `LLMB_INSTALL` — top-level directory for images, repositories, virtual environments, workloads, and results
- `LLMB_WORKLOAD` — `${LLMB_INSTALL}/workloads/pretrain_llama31_mlperf`
- `${LLMB_WORKLOAD}/experiments` — benchmark logs, profiles, and checkpoints

# Prepare Dataset

No separate dataset preparation step is required by this benchmark recipe.

# Run Training

The launcher runs 50 training steps by default and writes logs and results under `${LLMB_WORKLOAD}/experiments`.

## Using `llmb-run` (Recommended)

Run benchmarks from the installation directory:

```bash
cd "$LLMB_INSTALL"

# NVFP4 on 8 GPUs
llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 8

# NVFP4 on 72 GPUs
llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 72
```

The supported precision and scale combinations are defined in `metadata.yaml`. FP8 uses the Transformer Engine current-scaling recipe internally.

### Additional Slurm Parameters

Use the built-in `llmb-run submit` Slurm flags:

```bash
# Use a reservation
llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 8 \
  --reservation my_reservation

# Select nodes
llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 72 \
  --nodelist node001,node002

# Exclude nodes
llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 72 \
  --exclude node003,node004
```

See the [llmb-run documentation](../../../cli/llmb-run/README.md) for more options.

## Direct Method

You can also invoke the launch script directly:

```bash
cd "$LLMB_INSTALL/llmb_repo/llama3.1/pretrain/mlperf"
JOB_TOTAL_GPUS=<count> GPU_TYPE=<gpu_type> DTYPE=<precision> ./launch.sh
```

Activate the environment created by the installer before running the script.

### Required Environment Variables

- `JOB_TOTAL_GPUS` — one of `8` or `72`
- `GPU_TYPE` — `gb300` or `gb200`
- `SBATCH_ACCOUNT` — Slurm account
- `SBATCH_PARTITION` — Slurm partition

### Optional Environment Variables

- `DTYPE` — precision; defaults to `nvfp4`
  - Use `nvfp4` for 8- and 72-GPU runs
- `MAX_STEPS` — training steps; defaults to `50`
- `TIME_LIMIT` — Slurm time limit; defaults to `00:30:00`
- `ADDITIONAL_SLURM_PARAMS` — additional Slurm settings passed to the launcher
- `RUN_CONF_MOUNTS` — extra comma-separated container mounts
- `ENABLE_VBOOST` — enable GPU vBoost; defaults to `false`

# Output Locations

Benchmark results are saved under `$LLMB_WORKLOAD/experiments`:

```text
experiments/
└── <experiment_name>/
    └── <experiment_name>_<timestamp>/
        └── <experiment_name>/
            ├── log-<experiment_name>.out
            ├── sbatch_<experiment_name>.out
            ├── nsys_profile/
            │   └── *.nsys-rep
            └── torch_profile/
                └── rank-*.json.gz
```

The main `log-*.out` file contains step timing and performance metrics parsed by `llmb-run jobs`.

# Profiling

Nsight Systems and PyTorch profiling are supported. The two profiling modes are mutually exclusive.

## Nsight Systems

Use `-p` with `llmb-run`:

```bash
llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 72 -p
```

For direct launches, set `ENABLE_PROFILE=true`. By default, steps 45–50 are profiled on all ranks.

Optional profiling settings:

- `PROFILE_START_STEP` — first profiled step; defaults to `45`
- `PROFILE_STOP_STEP` — final profiled step; defaults to `50`
- `ENABLE_GPU_METRICS` — collect GPU metrics; defaults to `false`

```bash
ENABLE_GPU_METRICS=true \
  llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 72 -p
```

Open the generated `.nsys-rep` files with the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started). Multi-GPU runs generate one or more reports per node or rank; see the [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).

## PyTorch Profiler

Set `ENABLE_PYTORCH_PROFILE=true`:

```bash
ENABLE_PYTORCH_PROFILE=true \
  llmb-run submit -w pretrain_llama3.1_mlperf -s 8b --dtype nvfp4 --scale 8
```

Trace files are written to `torch_profile/rank-N.json.gz`. See the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for viewing instructions.

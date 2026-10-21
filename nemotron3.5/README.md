# Overview

This recipe contains information and scripts to produce performance results for the Nemotron 3.5 Lightning **30b** (30B-A3B) pre-training workload. The model uses the Nemotron 3 Nano hybrid Mamba-Transformer mixture-of-experts architecture (52 layers) with 2 additional multi-token prediction (MTP) layers. The scripts help perform environment setup and launch benchmark jobs. Configurations use weak scaling methodology (global batch size scales proportionally with GPU count).

FP8 runs use the MXFP8 recipe. H100 supports BF16 only.

## GB300

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8/BF16  |  8   |  8192  |   52   |  1  |  1  |  1  |  8  |  8  | N/A |  4  | 512  | 16  |
| FP8/BF16  |  16  |  8192  |   52   |  1  |  1  |  1  |  8  | 16  | N/A |  4  | 1024 | 16  |
| FP8/BF16  |  32  |  8192  |   52   |  1  |  1  |  1  |  8  | 32  | N/A |  4  | 2048 | 16  |
| FP8/BF16  |  64  |  8192  |   52   |  1  |  1  |  1  |  8  | 64  | N/A |  4  | 4096 | 16  |

## GB200

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8/BF16  |  8   |  8192  |   52   |  1  |  1  |  1  |  8  |  8  | N/A |  2  | 512  | 32  |
| FP8/BF16  |  16  |  8192  |   52   |  1  |  1  |  1  |  8  | 16  | N/A |  2  | 1024 | 32  |
| FP8/BF16  |  32  |  8192  |   52   |  1  |  1  |  1  |  8  | 32  | N/A |  2  | 2048 | 32  |
| FP8/BF16  |  64  |  8192  |   52   |  1  |  1  |  1  |  8  | 64  | N/A |  2  | 4096 | 32  |

## B300

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8/BF16  |  8   |  8192  |   52   |  1  |  1  |  1  |  8  |  8  | N/A |  4  | 512  | 16  |
| FP8/BF16  |  16  |  8192  |   52   |  1  |  1  |  1  |  8  | 16  | N/A |  4  | 1024 | 16  |
| FP8/BF16  |  32  |  8192  |   52   |  1  |  1  |  1  |  8  | 32  | N/A |  4  | 2048 | 16  |
| FP8/BF16  |  64  |  8192  |   52   |  1  |  1  |  1  |  8  | 64  | N/A |  4  | 4096 | 16  |

## B200

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8/BF16  |  8   |  8192  |   52   |  1  |  1  |  1  |  8  |  8  | N/A |  2  | 512  | 32  |
| FP8/BF16  |  16  |  8192  |   52   |  1  |  1  |  1  |  8  | 16  | N/A |  2  | 1024 | 32  |
| FP8/BF16  |  32  |  8192  |   52   |  1  |  1  |  1  |  8  | 32  | N/A |  2  | 2048 | 32  |
| FP8/BF16  |  64  |  8192  |   52   |  1  |  1  |  1  |  8  | 64  | N/A |  2  | 4096 | 32  |

## H100

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| BF16      |  16  |  8192  |   52   |  1  |  1  |  1  |  8  | 16  | N/A |  1  | 512  | 32  |
| BF16      |  32  |  8192  |   52   |  1  |  1  |  1  |  8  | 32  | N/A |  1  | 1024 | 32  |
| BF16      |  64  |  8192  |   52   |  1  |  1  |  1  |  8  | 64  | N/A |  1  | 2048 | 32  |

# Performance Measurement and Analysis

Performance is reported as:

- `s/iter` — wall-clock seconds per training step
- `TFLOPS/GPU` — sustained FLOPS achieved per GPU

Each benchmark runs 50 steps; iterations 35–44 are averaged to skip warmup (input prefetch, activation allocation, JIT compilation).

## Viewing results with `llmb-run jobs`

Each `llmb-run jobs` command refreshes Slurm state and parses the training log for any job that has finished (succeeded, failed, or cancelled) — there is no background updater. Run from `$LLMB_INSTALL`:

```bash
# List all jobs you've submitted, with parsed metrics
llmb-run jobs

# Full details for one job (Job ID comes from the listing above)
llmb-run jobs show <job_id>

# Open the training log; --follow tails it, --dir prints the experiment directory
llmb-run jobs log <job_id>
```

Example `llmb-run jobs` output (illustrative values):

```text
  Workload              DType  Scale   Job ID  Profile  Submit Time       Slurm Status  Elapsed   s/iter  TFLOPS/GPU
  pretrain_example_8b   bf16     128  1234567  No       2026-04-17 13:42  COMPLETED     00:12:34    4.21     1234.56
  pretrain_example_70b  fp8      256  1234589  No       2026-04-17 14:05  RUNNING       00:03:11
```

Blank `s/iter` or `TFLOPS/GPU` means the job has not finished yet, or the log did not contain enough completed iterations. See the [llmb-run README](../cli/llmb-run/README.md#jobs-command) for the full command reference.

## Derived metrics

To convert step time into tokens per second:

```text
(throughput in tokens/sec) = (sequence length) * (global batch size) / (s/iter)
```

To estimate time-to-train for a target token budget:

```text
(time to train in days) = (total tokens) / (throughput in tokens/sec) / 86400
```

To compute model FLOPs utilization (MFU):

```text
MFU = TFLOPS/GPU / (peak GPU FLOPS)
```

For peak theoretical throughput values used in MFU calculations, see the [Peak Theoretical Throughput](../README.md#peak-theoretical-throughput) section in the main README.

# Prerequisites

Requires Python 3.12.x, or conda.

## Request Access

No special access required to run this benchmark.

## Slurm

This document references several Slurm commands and parameters. A brief summary is included below. Note that these are guidelines and might not apply to all environments. Consult your system administrator for the parameters specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Used for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES, this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_nemotron3.5`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

# Prepare Dataset

Since Nemotron 3.5 Lightning training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps by default (`MAX_STEPS`, overridable) and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder ([see Output Locations](#output-locations) for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Example: Nemotron 3.5 Lightning, BF16, 8 GPUs
llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 8

# Example: Nemotron 3.5 Lightning, FP8 MX, 64 GPUs
llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype fp8 --scale 64
```

### Additional SLURM Parameters

Use a SLURM reservation:

```bash
ADDITIONAL_SLURM_PARAMS="reservation=my_reservation" llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 16
```

Run on specific nodes:

```bash
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002" llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 16
```

Exclude specific nodes:

```bash
ADDITIONAL_SLURM_PARAMS="exclude=node003,node004" llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 16
```

Combine multiple parameters (semicolon-separated):

```bash
ADDITIONAL_SLURM_PARAMS="nodelist=node001,node002;reservation=my_reservation;exclusive" llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 16
```

For more details on llmb-run usage, see the [llmb-run documentation](../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/nemotron3.5/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [FP8_RECIPE=<recipe>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use. Supported scales: 8, 16, 32, 64 (16, 32, 64 on H100).
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `b300` - NVIDIA B300 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs
- `SBATCH_ACCOUNT` and `SBATCH_PARTITION`: Slurm account and partition ([see Slurm](#slurm)).

**Optional:**

- `DTYPE`: Precision format (default: `bf16`). Supported: `bf16`, `fp8` (`bf16` only on H100).

- `FP8_RECIPE`: FP8 recipe when `DTYPE=fp8` (default: `mx`). Only `mx` is supported.

- `TIME_LIMIT`: Slurm job time limit (default: `00:30:00`).

- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated

  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`

### Example Commands

Train Nemotron 3.5 Lightning with BF16 precision on 8 GB300 GPUs:

```shell
JOB_TOTAL_GPUS=8 GPU_TYPE=gb300 DTYPE=bf16 ./launch.sh
```

Train Nemotron 3.5 Lightning with FP8 MX precision on 16 B200 GPUs:

```shell
JOB_TOTAL_GPUS=16 GPU_TYPE=b200 DTYPE=fp8 ./launch.sh
```

Train Nemotron 3.5 Lightning with BF16 precision on 16 H100 GPUs:

```shell
JOB_TOTAL_GPUS=16 GPU_TYPE=h100 DTYPE=bf16 ./launch.sh
```

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```text
experiments/
├── <experiment_name>/
│   └── <experiment_name>_<timestamp>/
│       ├── <experiment_name>/
│       │   ├── log-<experiment_name>.out      # Main training log with performance data
│       │   ├── sbatch_<experiment_name>.out   # Batch script output
│       │   └── nsys_profile/                  # Profiling output (when enabled)
│       │       └── *.nsys-rep files
│       └── [batch scripts and other files]
```

The `<experiment_name>` follows the pattern: `pretrain_nemotron_3_5_lightning_<dtype>_gpus<scale>_<gpu_type>`, where `<dtype>` is `bf16` or `fp8_mx`.

**Key files:**

- `log-<experiment_name>.out` - Contains training step timing and performance metrics parsed by `llmb-run jobs`
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling

Profiling is supported with Nsight Systems or PyTorch Profiler.

## Run Nsight Profiling

To enable profiling with Nsight Systems, use the `-p` flag with `llmb-run` or set `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

**Note:** Nsight profiling of FP8 MX runs on GB300 and GB200 is not currently functional, so the launch script exits with an error for that combination. BF16 runs can be profiled on all supported GPUs, and FP8 MX runs can be profiled on B300 and B200.

To view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information, visit [Nsight Systems](https://docs.nvidia.com/nsight-systems/).

### Profiling job details:

- **MPI Ranks:** all ranks
- **Job Steps:** 45-50
- **Output Location:** Profiling output saved alongside training results ([see Output Locations](#output-locations))
- **Filename format:** `profile_<pid>_<SLURM_JOB_ID>_node<SLURM_NODEID>_rank<SLURM_PROCID>.nsys-rep`

**Example command:**

```shell
llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 8 -p
```

### Customizing profiling behavior:

- Specify job steps to profile:
  - `PROFILE_START_STEP`: start profiling on this job step.
    - Default: 45
  - `PROFILE_STOP_STEP`: stop profiling on this job step.
    - Default: 50
- Specify ranks to profile:
  - `PROFILE_RANKS`: comma-separated list of ranks to profile.
    - Default: all ranks
- Enable GPU metrics collection:
  - `ENABLE_GPU_METRICS`: Enable GPU metrics collection during Nsight profiling (default: false)
  * When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  * Provides additional GPU utilization, memory usage, and compute efficiency data
  * May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**

```shell
ENABLE_GPU_METRICS=true llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 8 -p
```

### Viewing results

To view the profile traces (\*.nsys-rep files) interactively:

- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system, for example `/home/nsight-traces/`.
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from that folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run.

Because most benchmarking jobs run on multiple GPUs, each run generates multiple .nsys-rep files. The [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) can help automate the analysis and get to results more quickly using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

## PyTorch Profiling

PyTorch Profiling is intended for rare, advanced debugging scenarios such as NCCL correlation analysis. To enable it, set `ENABLE_PYTORCH_PROFILE=true` when submitting your job.

> **Note:** This option is mutually exclusive with Nsight profiling (`ENABLE_PROFILE`). Both cannot be enabled at the same time.

**Example command:**

```shell
ENABLE_PYTORCH_PROFILE=true llmb-run submit -w pretrain_nemotron3.5 -s 30b --dtype bf16 --scale 8
```

Trace files are saved to `torch_profile/rank-N.json.gz` in the job output directory, where `N` is the rank number. For details on the PyTorch Profiler and how to view resulting traces, see the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

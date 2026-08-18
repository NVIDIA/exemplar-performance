# Overview

This recipe contains information and scripts to produce performance results for the DeepSeek V3 pre-training workload. The scripts help perform environment setup and launch benchmark jobs. Configurations use weak scaling methodology (global batch size scales proportionally with GPU count).

## GB300

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| NVFP4     | 128  |  4096  |   61   |  1  |  4  |  1  | 32  | 32  |  4  |  2  | 2048 | 32  |
| FP8       | 128  |  4096  |   61   |  1  |  4  |  1  | 32  | 32  |  4  |  1  | 2048 | 64  |
| BF16      | 128  |  4096  |   61   |  1  |  4  |  1  | 32  | 32  |  4  |  1  | 2048 | 64  |
| NVFP4     | 256  |  4096  |   61   |  1  |  2  |  1  | 32  | 128 |  8  |  2  | 4096 | 16  |
| FP8       | 256  |  4096  |   61   |  1  |  2  |  1  | 32  | 128 |  8  |  1  | 4096 | 32  |
| BF16      | 256  |  4096  |   61   |  1  |  4  |  1  | 64  | 64  |  4  |  1  | 4096 | 64  |
| NVFP4     | 512  |  4096  |   61   |  1  |  2  |  1  | 32  | 256 |  8  |  2  | 8192 | 16  |
| FP8       | 512  |  4096  |   61   |  1  |  2  |  1  | 32  | 256 |  8  |  1  | 8192 | 32  |
| BF16      | 512  |  4096  |   61   |  1  |  4  |  1  | 64  | 128 |  4  |  1  | 8192 | 64  |

## GB200

| Precision      | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| -------------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8            | 128  |  4096  |   61   |  2  |  2  |  1  | 32  | 32  |  8  |  1  | 2048 | 64  |
| NVFP4/FP8/BF16 | 256  |  4096  |   61   |  1  |  4  |  1  | 64  | 64  |  4  |  1  | 4096 | 64  |
| NVFP4/FP8/BF16 | 512  |  4096  |   61   |  1  |  4  |  1  | 64  | 128 |  4  |  1  | 8192 | 64  |

## B300

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| NVFP4     | 128  |  4096  |   61   |  1  |  8  |  1  |  8  | 16  |  2  |  2  | 2048 | 64  |
| FP8/BF16  | 128  |  4096  |   61   |  1  |  8  |  1  |  8  | 16  |  2  |  1  | 2048 | 128 |
| NVFP4     | 256  |  4096  |   61   |  1  |  8  |  1  |  8  | 32  |  2  |  2  | 4096 | 64  |
| FP8/BF16  | 256  |  4096  |   61   |  1  |  8  |  1  |  8  | 32  |  2  |  1  | 4096 | 128 |
| NVFP4     | 512  |  4096  |   61   |  1  |  8  |  1  |  8  | 64  |  2  |  2  | 8192 | 64  |
| FP8/BF16  | 512  |  4096  |   61   |  1  |  8  |  1  |  8  | 64  |  2  |  1  | 8192 | 128 |

## B200

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8       | 128  |  4096  |   61   |  1  |  8  |  1  |  8  | 16  |  2  |  1  | 2048 | 128 |
| FP8       | 256  |  4096  |   61   |  1  |  8  |  1  |  8  | 32  |  2  |  1  | 4096 | 128 |
| BF16      | 256  |  4096  |   61   |  1  |  8  |  1  |  8  | 32  |     |  1  | 4096 | 128 |
| FP8       | 512  |  4096  |   61   |  1  |  8  |  1  |  8  | 64  |  2  |  1  | 8192 | 128 |
| BF16      | 512  |  4096  |   61   |  1  |  8  |  1  |  8  | 64  |     |  1  | 8192 | 128 |

## H100

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | EP  | DP  | VP  | MBS |  GBS  | GA  |
| --------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :---: | :-: |
| FP8       | 1024 |  4096  |   61   |  2  |  8  |  1  | 64  | 64  |  2  |  1  | 16384 | 256 |
| BF16      | 1024 |  4096  |   61   |  2  |  8  |  1  | 64  | 64  |  4  |  1  | 16384 | 256 |

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

Blank `s/iter` or `TFLOPS/GPU` means the job has not finished yet, or the log did not contain enough completed iterations. See the [llmb-run README](../../../cli/llmb-run/README.md#jobs-command) for the full command reference.

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

For peak theoretical throughput values used in MFU calculations, see the [Peak Theoretical Throughput](../../../README.md#peak-theoretical-throughput) section in the main README.

# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via `export HF_TOKEN=<your token>`.

Requires Python 3.12.x, or conda.

## Request Access

No special access is required to run this benchmark.

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../../../README.md) (see [installer documentation](../../../cli/llmb-install/README.md) for details) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_deepseek-v3`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

# Prepare Dataset

Since DeepSeek V3 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256

# Example with different scale, and precision
llmb-run submit -w pretrain_deepseek-v3 --dtype fp8 --scale 512

```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags instead of `ADDITIONAL_SLURM_PARAMS`.

Use a Slurm reservation:

```bash
llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256 --reservation my_reservation
```

Run on specific nodes:

```bash
llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256 --nodelist node001,node002
```

Exclude specific nodes:

```bash
llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256 --exclude node003,node004
```

Combine multiple parameters:

```bash
llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256 --nodelist node001,node002 --reservation my_reservation --slurm-arg exclusive
```

For more details on `llmb-run` usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/deepseek_v3/pretrain/megatron_bridge/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `b300` - NVIDIA B300 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**

- `DTYPE`: Precision format (default: `bf16`)
  - `bf16` - BFloat16 precision
  - `fp8` - FP8 precision
- `MODEL_SIZE`: Model variant (fixed: `671b`)
  - `671b` - 671 billion parameter model (only supported size)
- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated
  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`

### Example Commands

Train DeepSeek V3 with BF16 precision on 256 GB200 GPUs:

```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 ./launch.sh
```

Train on 1024 H100 GPUs:

```shell
JOB_TOTAL_GPUS=1024 GPU_TYPE=h100 ./launch.sh
```

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```
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

The `<experiment_name>` typically follows the pattern: `pretrain_deepseek_v3_671b_<dtype>_<scale>_<config>`

**Key files:**

- `log-<experiment_name>.out` - Contains training step timing and performance metrics parsed by `llmb-run jobs`
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling

Profiling is supported with Nsight Systems or PyTorch Profiler.

## Run Nsight Profiling

To enable profiling with Nsight Systems, use the `-p` flag with `llmb-run` or set `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Profiling job details:

- **MPI Ranks:** all
- **Job Steps:** 45-50
- **Output Location:** Profiling output saved alongside training results (see Output Locations)
- **Filename format:** `profile_${SLURM_JOB_ID}_nodeId_rankId.nsys-rep`

**Example command:**

```shell
llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256 -p
```

### Customizing profiling behavior:

- Specify job steps to profile:
  - `PROFILE_START_STEP`: start profiling on this job step.
  * Default: 45
  - `PROFILE_STOP_STEP`: stop profiling on this job step.
  * Default: 50
- Enable GPU metrics collection:
  - `ENABLE_GPU_METRICS`: Enable GPU metrics collection during Nsight profiling (default: false)
  * When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  * Provides additional GPU utilization, memory usage, and compute efficiency data
  * May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**

```shell
ENABLE_GPU_METRICS=true llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256 -p
```

### Viewing results

In order to view the profile traces (\*.nsys-rep files) interactively:

- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the model or the job run.

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

## PyTorch Profiling

PyTorch Profiling is intended for rare, advanced debugging scenarios such as NCCL correlation analysis. To enable it, set `ENABLE_PYTORCH_PROFILE=true` when submitting your job.

> **Note:** This option is mutually exclusive with Nsight profiling (`ENABLE_PROFILE`). Both cannot be enabled at the same time.

**Example command:**

```shell
ENABLE_PYTORCH_PROFILE=true llmb-run submit -w pretrain_deepseek-v3 --dtype bf16 --scale 256
```

Trace files are written to `torch_profile/rank-N.json.gz` on all supported GPU types, where `N` is the rank number. For details on the PyTorch Profiler and how to view resulting traces, see the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

# Proxy Configurations

Two proxy configurations are available for performance benchmarking and debugging.

## 1. Reduced GPU-Count Proxy

Runs a reduced version of the full model on fewer GPUs while maintaining high model TFLOPS. These configurations also support testing NVL72 platforms for GB300/GB200 GPUs.

Supported configurations:

- **GB300:** 4, 8, 16, 32, 64, and 72×N GPUs
- **GB200:** 4, 8, 16, 32, 64, and 72×N GPUs
- **B300:** 8, 16, 32, and 64 GPUs
- **B200:** 8, 16, 32, and 64 GPUs

**Example:**

```bash
llmb-run submit -w pretrain_deepseek-v3 --dtype fp8 --scale 4 --proxy
```

## 2. CPU Performance Debug Proxy

Designed to isolate CPU performance bottlenecks by allowing full CG (CUDA Graphs) to be enabled or disabled. These configurations are CPU-limited when full CG is disabled, making CPU-related performance issues easier to analyze.

**Supported configurations:**

- **GB300/GB200:** 4 and 64 GPUs
- **B300/B200:** 8 GPUs

### Run with Full CG enabled (GB300 single node)

```bash
CPU_PERF_PROXY=true llmb-run submit -w pretrain_deepseek-v3 --dtype fp8 --scale 4 --proxy
```

### Run with Full CG disabled (GB300 single node)

```bash
CPU_PERF_PROXY=true DISABLE_CG=true llmb-run submit -w pretrain_deepseek-v3 --dtype fp8 --scale 4 --proxy
```

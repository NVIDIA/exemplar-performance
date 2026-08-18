# Overview

This recipe contains information and scripts to produce performance results for the GPT-OSS 20B MLPerf Training workload. The scripts help perform environment setup and launch benchmark jobs.

**Supported Model Size:**

- `20b` — GPT-OSS 20B (20 billion parameters, Mixture-of-Experts)

Configurations use weak scaling methodology (global batch size scales proportionally with GPU count). Parallelism and batch sizes come from Megatron-Bridge `configs/gpt_oss/gpt_oss_workload_base_configs.py`.

## GB300

| Precision | GPUs | SeqLen | Layers |  TP |  PP |  CP |  EP |  DP |  VP | MBS | GBS |  GA |
| :-------- | :--: | :----: | :----: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| NVFP4     |  8   |  8192  |   24   |   1 |   1 |   1 |   2 |   4 | N/A |   3 |  24 |   2 |
| FP8 (MX)  |  8   |  8192  |   24   |   1 |   1 |   1 |   1 |   8 | N/A |   3 |  24 |   1 |
| NVFP4     |  72  |  8192  |   24   |   1 |   1 |   2 |   4 |   9 | N/A |   1 |  36 |   4 |
| FP8 (MX)  |  72  |  8192  |   24   |   1 |   1 |   2 |   4 |   9 | N/A |   1 |  36 |   4 |
| FP8 (MX)  | 512  |  8192  |   24   |   2 |   1 |   4 |   8 |   8 | N/A |   1 |  64 |   8 |

## GB200

| Precision | GPUs | SeqLen | Layers |  TP |  PP |  CP |  EP |  DP |  VP | MBS | GBS |  GA |
| :-------- | :--: | :----: | :----: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| NVFP4     |  8   |  8192  |   24   |   1 |   1 |   1 |   2 |   4 | N/A |   2 |  16 |   2 |
| FP8 (MX)  |  8   |  8192  |   24   |   1 |   1 |   1 |   1 |   8 | N/A |   2 |  16 |   1 |
| NVFP4     |  72  |  8192  |   24   |   1 |   1 |   2 |   4 |   9 | N/A |   1 |  36 |   4 |
| FP8 (MX)  |  72  |  8192  |   24   |   1 |   1 |   2 |   4 |   9 | N/A |   1 |  36 |   4 |
| FP8 (MX)  | 512  |  8192  |   24   |   2 |   1 |   4 |   8 |   8 | N/A |   1 |  64 |   8 |

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

No special access is required to run this benchmark. The installer downloads the [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) model config from HuggingFace.

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
- Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../../../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_gpt_oss_mlperf`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

The model is trained from scratch; no checkpoint is loaded.

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run (NVFP4, 8 GPUs)
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype nvfp4 --scale 8

# Run a benchmark with llmb-run (FP8, 72 GPUs)
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 72

# Run a benchmark with llmb-run (FP8, 512 GPUs)
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 512
```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags instead of `ADDITIONAL_SLURM_PARAMS`.

Use a Slurm reservation:

```bash
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype nvfp4 --scale 8 --reservation my_reservation
```

Run on specific nodes:

```bash
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 72 --nodelist node001,node002
```

Exclude specific nodes:

```bash
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 72 --exclude node003,node004
```

Combine multiple parameters:

```bash
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 512 --nodelist node001,node002 --reservation my_reservation --slurm-arg exclusive
```

For more details on `llmb-run` usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

**Important**:

- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/gpt-oss/pretrain/mlperf/`

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs

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

The `<experiment_name>` typically follows the pattern: `pretrain_gpt_oss_mlperf_20b_<dtype>_<scale>_<config>`

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
llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 72 -p
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
ENABLE_GPU_METRICS=true llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 72 -p
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
ENABLE_PYTORCH_PROFILE=true llmb-run submit -w pretrain_gpt_oss_mlperf_20b --dtype fp8 --scale 72
```

Trace files are saved to `torch_profile/rank-N.json.gz` in the job output directory, where `N` is the rank number. For details on the PyTorch Profiler and how to view resulting traces, see the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

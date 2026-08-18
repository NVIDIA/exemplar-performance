# Overview

This recipe contains information and scripts to produce performance results for the Llama 2 70B MLPerf Training LoRA finetuning workload. The scripts help perform environment setup and launch benchmark jobs.

**Supported Model Size:**

- `70b` — Llama 2 70B (80 layers, LoRA finetuning)

Configurations use weak scaling methodology (global batch size scales proportionally with GPU count). Parallelism and batch sizes come from Megatron-Bridge `configs/llama/llama2_workload_base_configs.py`.

## GB300

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :-------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FP8       |  4   |  8192  |   80   |  1  |  1  |  1  |  4  | N/A |  1  |  8  |  2  |
| FP8       |  8   |  8192  |   80   |  1  |  1  |  1  |  8  | N/A |  1  |  8  |  1  |
| FP8       |  72  |  8192  |   80   |  1  |  1  |  8  |  9  | N/A |  1  |  9  |  1  |
| FP8       | 512  |  8192  |   80   |  1  |  1  |  8  | 64  | N/A |  1  | 64  |  1  |

## GB200

| Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :-------- | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FP8       |  4   |  8192  |   80   |  1  |  1  |  1  |  4  | N/A |  1  |  8  |  2  |
| FP8       |  8   |  8192  |   80   |  1  |  1  |  1  |  8  | N/A |  1  |  8  |  1  |
| FP8       |  72  |  8192  |   80   |  1  |  1  |  8  |  9  | N/A |  1  |  9  |  1  |
| FP8       | 512  |  8192  |   80   |  1  |  1  |  8  | 64  | N/A |  1  | 64  |  1  |

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
  llama2-mlperf         fp8        8  1234567  No       2026-04-17 13:42  COMPLETED     00:12:34    4.21     1234.56
  llama2-mlperf         nvfp4     72  1234589  No       2026-04-17 14:05  RUNNING       00:03:11
```

Blank `s/iter` or `TFLOPS/GPU` means the job has not finished yet, or the log did not contain enough completed iterations. See the [llmb-run README](../../cli/llmb-run/README.md#jobs-command) for the full command reference.

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

For peak theoretical throughput values used in MFU calculations, see the [Peak Theoretical Throughput](../../README.md#peak-theoretical-throughput) section in the main README.

# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via `export HF_TOKEN=<your token>`.

Requires Python 3.12.x, or conda.

## Request Access

Access to Llama 2 70B must be requested through [meta-llama/Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf). The approval process is not automatic and could take a day or more.

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/finetune_llama2_mlperf`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Prepare Checkpoint

The recommended method to prepare the model checkpoint is by using the **installer** referenced in the [main README](../../README.md).

When the installer runs the setup for this recipe, it prepares the Llama 2 70B finetuning checkpoint required by the MLPerf workload. Please note the following:

- This step may run interactively as part of the installer and can take some time to complete.
- It requires GPU resources. Ensure your SLURM partition has available GPUs.
- To verify that the checkpoint has been downloaded successfully, refer to the [Storage Requirements and Verification](#storage-requirements-and-verification) section.

## Prepare Dataset

This workload uses MLPerf mock training data by default (`MLPERF_DATA=mock`). No manual dataset preparation is required for standard benchmark runs.

# Run Finetuning

Once the environment has been prepared, it is time to train a model. The finetuning runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run (FP8, 8 GPUs)
llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 8

# Run a benchmark with llmb-run (NVFP4, 72 GPUs on GB300)
llmb-run submit -w finetune_llama2_mlperf --dtype nvfp4 --scale 72
```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags instead of `ADDITIONAL_SLURM_PARAMS`.

Use a Slurm reservation:

```bash
llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 8 --reservation my_reservation
```

Run on specific nodes:

```bash
llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 72 --nodelist node001,node002
```

Exclude specific nodes:

```bash
llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 72 --exclude node003,node004
```

Combine multiple parameters:

```bash
llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 512 --nodelist node001,node002 --reservation my_reservation --slurm-arg exclusive
```

For more details on `llmb-run` usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run finetuning directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the finetuning commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_INSTALL/llmb_repo/llama2/finetune`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use. Supported values: `4`, `8`, `72`, `512`
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs

**Optional:**

- `DTYPE`: Precision format (default: `fp8`)
  - `fp8` - FP8 precision
  - `nvfp4` - NVFP4 precision
- `MLPERF_DATA`: Dataset mode (default: `mock`)
- `MAX_STEPS`: Number of training steps (default: `50`)
- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated
  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`
- Parallelism and batch overrides: `TP`, `PP`, `CP`, `VP`, `EP`, `ET`, `MBS`, `GBS`

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```text
experiments/
├── <experiment_name>/
│   └── <experiment_name>_<timestamp>/
│       ├── <experiment_name>/
│       │   ├── log-<experiment_name>.out      # Main finetuning log with performance data
│       │   ├── sbatch_<experiment_name>.out   # Batch script output
│       │   └── nsys_profile/                  # Profiling output (when enabled)
│       │       └── *.nsys-rep files
│       └── [batch scripts and other files]
```

The `<experiment_name>` typically follows the pattern: `lora_llama2_70b_<dtype>_<scale>_<config>`

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
- **Output Location:** Profiling output saved alongside finetuning results (see Output Locations)
- **Filename format:** `profile_${SLURM_JOB_ID}_nodeId_rankId.nsys-rep`

**Example command:**

```shell
llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 8 -p
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
ENABLE_GPU_METRICS=true llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 72 -p
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
ENABLE_PYTORCH_PROFILE=true llmb-run submit -w finetune_llama2_mlperf --dtype fp8 --scale 8
```

For details on the PyTorch Profiler and how to view resulting traces, see the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

# Storage Requirements and Verification

**Note:** The Llama 2 70B checkpoint requires significant storage space. Ensure your file system has sufficient space (at least 1TB recommended) to accommodate the checkpoint and experiment outputs.

To verify that the checkpoint has been correctly prepared, check for the following directory structure within `${LLMB_WORKLOAD}/checkpoint/` after the installer's setup tasks have completed:

```text
checkpoint/
├── hub/               # HuggingFace hub cache (config, tokenizer, and base weights)
└── llama2_70b/        # Megatron-Bridge checkpoint converted from meta-llama/Llama-2-70b-hf
```

Since this workload uses MLPerf mock training data by default (`MLPERF_DATA=mock`), no dataset files are downloaded or cached — mock batches are generated synthetically at runtime, so there is no `datasets/` folder under `checkpoint/`.

The training configuration is passed the converted checkpoint directly via `--pretrained_checkpoint $LLMB_WORKLOAD/checkpoint/llama2_70b`, loaded for LoRA finetuning (`--task lora`).

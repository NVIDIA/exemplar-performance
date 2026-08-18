# Overview

This recipe contains information and scripts to produce performance results for the Llama3.1 8B, 70B, and 405B training workloads. The scripts help perform environment setup and launch benchmark jobs. Configurations use weak scaling methodology (global batch size scales proportionally with GPU count).

**Note:** pretrain_llama3.1 is the correct workload name for all sizes. For 8B and 70B, this recipe intentionally reuses the existing Megatron-Bridge llama3 configs, so setup output may reference Meta-Llama-3-\* and experiment/log names may start with pretrain_llama3. This is expected.

## GB300

### FP8

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       | 256-512 |   FP8    |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6  | 192 | False |
| 70b        | 64-512  |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  2  | GPUs\*4  |  2  | False |
| 8b         |  8-128  |   FP8    |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  4  | GPUs\*16 |  4  | False |

### NVFP4

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       | 256-512 |  NVFP4   |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6  | 192 | False |
| 70b        | 64-512  |  NVFP4   |  8192  |   80   | False |  1  |  4  |  1  |  1  |  1  | GPUs/4  |  5  |  1  | GPUs\*4  | 16  | False |
| 8b         |  8-128  |  NVFP4   |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  4  | GPUs\*16 |  4  | False |

## GB200

### FP8

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       | 256-512 |   FP8    |  8192  |  126   | False |  4  | 16  |  1  |  1  |  1  | GPUs/64 |  4  |  1  | GPUs\*6  | 384 | False |
| 70b        | 64-512  |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  2  | GPUs\*4  |  2  | False |
| 8b         |  8-128  |   FP8    |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  2  | GPUs\*16 |  8  | False |

### NVFP4

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       | 256-512 |  NVFP4   |  8192  |  126   | False |  4  | 16  |  1  |  1  |  1  | GPUs/64 |  8  |  1  | GPUs\*6  | 384 | False |
| 8b         |  8-128  |  NVFP4   |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  4  | GPUs\*16 |  4  | False |

## B300

### FP8

| Model Size |    GPUs     | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :---------: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       |   256-512   |   FP8    |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6  | 192 | False |
| 70b        |   **64**    |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  1  | GPUs\*4  |  4  | False |
| 70b        | **128-512** |   FP8    |  8192  |   80   | False |  1  |  4  |  1  |  1  |  1  | GPUs/4  |  5  |  1  | GPUs\*4  | 16  | False |
| 8b         |    8-128    |   FP8    |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  4  | GPUs\*16 |  4  | False |

### NVFP4

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       | 256-512 |  NVFP4   |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6  | 192 | False |
| 70b        | 64-512  |  NVFP4   |  8192  |   80   | False |  1  |  4  |  1  |  1  |  1  | GPUs/4  |  5  |  1  | GPUs\*4  | 16  | False |
| 8b         |  8-128  |  NVFP4   |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  4  | GPUs\*16 |  4  | False |

## B200

### FP8

| Model Size |     GPUs     | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :----------: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       |   256-1024   |   FP8    |  8192  |  126   | False |  4  |  8  |  2  |  1  |  1  | GPUs/64 |  8  |  1  | GPUs\*6  | 384 | False |
| 70b        |    **64**    |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  1  | GPUs\*4  |  4  | False |
| 70b        | **128-1024** |   FP8    |  8192  |   80   | False |  2  |  4  |  1  |  1  |  1  | GPUs/8  |  5  |  1  | GPUs\*4  | 32  | False |
| 8b         |    8-128     |   FP8    |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  2  | GPUs\*16 |  8  | False |

### NVFP4

| Model Size |   GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :------: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :------: | :-: | :---: |
| 405b       | 256-1024 |  NVFP4   |  8192  |  126   | False |  4  | 16  |  1  |  1  |  1  | GPUs/64 |  8  |  1  | GPUs\*6  | 384 | False |
| 70b        | 64-1024  |  NVFP4   |  8192  |   80   | False |  2  |  4  |  1  |  1  |  1  | GPUs/8  |  5  |  1  | GPUs\*4  | 32  | False |
| 8b         |  8-128   |  NVFP4   |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  4  | GPUs\*16 |  4  | False |

## H100

**Note:** For H100, only FP8 CS (current scaling) is supported; FP8 MX is not allowed for this GPU type.

### BF16

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |    DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :------: | :-: | :-: | :------: | :-: | :---: |
| 405b       |  1024   |   BF16   |  8192  |  126   | False |  8  |  8  |  2  |  1  |  1  | GPUs/128 |  8  |  1  |   1536   | 192 | False |
| 70b        | 64-1024 |   BF16   |  8192  |   80   | False |  4  |  4  |  2  |  1  |  1  | GPUs/32  |  5  |  1  | GPUs\*4  | 128 | False |
| 8b         |  8-128  |   BF16   |  8192  |   32   | False |  1  |  1  |  2  |  1  |  1  |  GPUs/2  | NA  |  1  | GPUs\*16 | 32  | False |

### FP8

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |    DP    | VP  | MBS |   GBS    | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :------: | :-: | :-: | :------: | :-: | :---: |
| 405b       |  1024   |   FP8    |  8192  |  126   | False |  8  |  8  |  2  |  1  |  1  | GPUs/128 |  8  |  1  |   1536   | 192 | False |
| 70b        | 64-1024 |   FP8    |  8192  |   80   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32  |  5  |  2  | GPUs\*4  |  4  | False |
| 8b         |  8-128  |   FP8    |  8192  |   32   | False |  1  |  1  |  1  |  1  |  1  |   GPUs   | NA  |  1  | GPUs\*16 | 16  | False |

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

Access requirements depend on the model size. The 405B configuration requires Llama 3.1 access, which must be requested through [Meta's website](https://www.llama.com/llama-downloads/) and then on the [HuggingFace Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B) page.

The 8B and 70B configurations intentionally reuse Megatron-Bridge `llama3` configs, so they require Llama 3 family access instead. Request access through [Meta's website](https://www.llama.com/llama-downloads/) and then request access to either [HuggingFace Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) or [HuggingFace Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B). Either approval grants access to the Llama 3 family. The approval process is not automatic and could take a day or more.

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
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_llama3.1`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

# Prepare Dataset

Since Llama3.1 training only uses synthetic datasets, this step is omitted.

# Run Training

Once the environment has been prepared, it is time to train a model. The training runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see [Output Locations](#output-locations) for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128

#Example with Llama3.1 70B
llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 64

#Example with Llama3.1 8B at a higher scale
llmb-run submit -w pretrain_llama3.1 -s 8b --dtype fp8 --scale 16
```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags instead of `ADDITIONAL_SLURM_PARAMS`.

Use a Slurm reservation:

```bash
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 --reservation my_reservation
```

Run on specific nodes:

```bash
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 --nodelist node001,node002
```

Exclude specific nodes:

```bash
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 --exclude node003,node004
```

Combine multiple parameters:

```bash
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 --nodelist node001,node002 --reservation my_reservation --slurm-arg exclusive
```

For more details on `llmb-run` usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/llama3.1/pretrain/megatron_bridge/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use (e.g., 128, 256, 512)
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `b300` - NVIDIA B300 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**

- `DTYPE`: Precision to run (default: `fp8`)
  - Supported values depend on `GPU_TYPE` and `MODEL_SIZE`. See the tables at the top of this README (and `metadata.yaml`) for supported combinations.
  - Common values: `fp8`, `bf16`, `nvfp4`
- `MODEL_SIZE`: Model variant (default: `405b`)
  - `405b` - 405 billion parameter model
  - `70b` - 70 billion parameter model
  - `8b` - 8 billion parameter model
- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated
  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`

### Example Commands

Train Llama3.1 405B with FP8 precision on 128 GB200 GPUs:

```shell
JOB_TOTAL_GPUS=128 GPU_TYPE=gb200 ./launch.sh
```

Train with FP8 precision on 256 GB200 GPUs:

```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 ./launch.sh
```

Train with FP8 precision on 1024 H100 GPUs:

```shell
JOB_TOTAL_GPUS=1024 GPU_TYPE=h100 ./launch.sh
```

Train with FP8 precision on 8 H100 GPUs with Llama3.1 8B:

```shell
MODEL_SIZE=8b JOB_TOTAL_GPUS=8 GPU_TYPE=h100 ./launch.sh
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

The `<experiment_name>` typically follows these patterns:

- For Llama3 8B/70B: `pretrain_llama3_<model_size>_<dtype>_<config>`
- For Llama3.1 405B: `pretrain_llama3.1_405b_<dtype>_<config>`

**Key files:**

- `log-<experiment_name>.out` - Contains training step timing and performance metrics parsed by `llmb-run jobs`
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling

Profiling is supported with Nsight Systems or PyTorch Profiler.

## Run Nsight Profiling

To enable profiling with Nsight Systems set variable `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

In order to view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information visit: [Nsight Systems](https://docs.nvidia.com/nsight-systems/)

### Default Profiling Settings:

- **MPI Ranks:** all ranks
- **Job Steps:** 45-50
- **Output Location:** Profiling output saved alongside training results (see Output Locations)
- **Filename format:** `profile_${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURM_LOCALID}.nsys-rep`

**Example command:**

```shell
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 -p
```

### Customizing profiling behavior:

- Specify job steps to profile:
  - `RUN_CONF_PROFILE_START_STEP`: start profiling on this job step.
    Default: 45
  - `RUN_CONF_PROFILE_STOP_STEP`: stop profiling on this job step.
    Default: 50
- Enable GPU metrics collection:
  - `ENABLE_GPU_METRICS`: Enable GPU metrics collection during Nsight profiling (default: false)
  * When set to `true` along with `ENABLE_PROFILE=true`, captures detailed GPU performance metrics
  * Provides additional GPU utilization, memory usage, and compute efficiency data
  * May require additional system configuration for GPU device metrics to work properly

**Example command with GPU metrics:**

```shell
ENABLE_GPU_METRICS=true llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 -p
```

### Viewing results

In order to view the profile traces (\*.nsys-rep files) interactively:

- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system. E.g., /home/nsight-traces/
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from /home/nsight-systems folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run.

Since most of the benchmarking jobs run on multiple GPUs, there will be multiple .nsys-rep files generated for each run. [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) will be very helpful to automate the analysis and get to results quicker by using Nsight recipes.

**See** these [tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) to get a quick start if you are new to Nsight profiling.

## PyTorch Profiling

PyTorch Profiling is intended for rare, advanced debugging scenarios such as NCCL correlation analysis. To enable it, set `ENABLE_PYTORCH_PROFILE=true` when submitting your job.

> **Note:** This option is mutually exclusive with Nsight profiling (`ENABLE_PROFILE`). Both cannot be enabled at the same time.

**Example command:**

```shell
ENABLE_PYTORCH_PROFILE=true llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128
```

Trace files are saved to `torch_profile/rank-N.json.gz` in the job output directory, where `N` is the rank number. For details on the PyTorch Profiler and how to view resulting traces, see the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

# Run With Checkpoints

Checkpoint save and load can be enabled for this workload in order to measure the impact of storage on checkpointing operations. The additional collected metrics are: time to save a checkpoint and time to load a checkpoint.

Checkpointing is enabled for 8B model size only. 70B and 405B are currently disabled for checkpointing operations due to NCCL errors.

## Save Checkpoint

Save checkpoint feature works for llama3.1 8b FP8 and NVFP4. Make sure your file system has sufficient disk space to accommodate checkpoint sizes below:

| Model | Checkpoint Size | Minimum Tested Scale GB300 | Minimum Tested Scale GB200 | Minimum Tested Scale B200 | Minimum Tested Scale H100 |
| :---: | :-------------: | :------------------------: | :------------------------: | :-----------------------: | :-----------------------: |
|  8b   |     ~105 GB     |             8              |             8              |             8             |             8             |

### How to enable

To save the checkpoints after pretraining llama3.1 model for `max_steps`, you need to set environment variable `ENABLE_CHECKPOINT=true`. At the end of the pretraining the checkpoints will be saved in the `${LLMB_WORKLOAD}/experiments` folder. There is an option to specify the folder where you want to save the checkpoints. This can be enabled by setting environment variable `CHECKPOINT_DIR=/path/to/checkpoints`.

```shell
experiment_name = pretrain_llama3_${MODEL_SIZE}_${DTYPE}_gpus${JOB_TOTAL_GPUS}_tp${tp}_pp${pp}_cp${cp}_vp${vp}_ep${ep}_mbs${mbs}_gbs${gbs}
timestamp = date '+%s'
Example directory where checkpoints are saved is ${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/checkpoints/
```

Command to run llama3.1 with checkpoint save enabled

```shell
ENABLE_CHECKPOINT=true llmb-run submit -w pretrain_llama3.1 -s <size> --dtype <precision> --scale <number>
```

### How to validate

- Check `${LLMB_WORKLOAD}/experiments/$experiment_name/${experiment_name}_${timestamp}/$experiment_name/checkpoints/iter_0000050` folder that it contains \*.distcp files
- Check job output log-\*.out file (see Training section for reference) for entries like
  ```shell
    successfully saved checkpoint from iteration      50 to /nemo_run/checkpoints [ t 1/1, p 1/1 ] (min, max) time across ranks (ms): save-checkpoint ................................: (24895.07, 24895.13)
  ```

## Load Checkpoint

Load checkpoint feature works successfully at the following scales:

| Model | Checkpoint Size | Minimum Tested Scale GB300 | Minimum Tested Scale GB200 | Minimum Tested Scale B200 | Minimum Tested Scale H100 |
| :---: | :-------------: | :------------------------: | :------------------------: | :-----------------------: | :-----------------------: |
|  8b   |     ~105 GB     |             8              |             8              |             8             |             8             |

**Note**:

- Running load checkpointing feature at other scales may run into CUDA OOM errors.

### How to enable

To resume training from saved checkpoints, you need to set `LOAD_CHECKPOINT_PATH=<path_to_checkpoint_directory>` environment variable. Make sure the checkpoint files are under the `${LLMB_WORKLOAD}/experiments` directory and `LOAD_CHECKPOINT_PATH` variable is set to: `iter_0000050` directory containing distributed checkpoint files with extension `*.distcp`.

E.g., if the checkpoint was saved under `experiments/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128_1766132151/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/checkpoints/iter_0000050/*` then set the environment variable to a directory one level higher:

`LOAD_CHECKPOINT_PATH=${LLMB_WORKLOAD}/experiments/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128_1766132151/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/checkpoints/iter_0000050`

The scripts will restore configuration from the checkpoint and resume training process. Training will run for 1 step after checkpoint has been loaded.

```shell
LOAD_CHECKPOINT_PATH=<your_path_to_checkpoint_directory> llmb-run submit -w pretrain_llama3.1 -s <size> --dtype <precision> --scale <number>
```

### How to validate

To validate that checkpoint was loaded successfully look for the entry like below in the main job log-\*.out file and make sure there is only 1 iteration of training (see Training section for reference):

```shell
checkpoint:
...
...
...
  load:.../.../experiments/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128_1766132151/pretrain_llama3_8b_bf16_gpus8_tp1_pp1_cp1_vpNone_ep1_mbs4_gbs128/checkpoints/iter_0000050
...
...
...
[2025-12-19 10:49:15] iteration        1/       1 | consumed samples:          128 | elapsed time per iteration (ms): 20289.1 | learning rate: 3.000000E-05 | global batch size:   128 | lm loss: 1.258695E+01 | loss scale: 1.0 | grad norm: 10.275 | number of skipped iterations:   0 | number of nan iterations:   0 |Number of parameters in transformer layers in billions:  6.98

Number of parameters in embedding layers in billions: 1.05
Total number of parameters in billions: 8.03
Number of parameters in most loaded shard in billions: 8.0305
Theoretical memory footprints: weight and optimizer=57438.81 MB
[Rank 0] (after 1 iterations) memory (GB) | mem-allocated-gigabytes: 51.072 | mem-active-gigabytes: 51.072 | mem-inactive-gigabytes: 10.971 | mem-reserved-gigabytes: 216.89 | mem-max-allocated-gigabytes: 194.41 | mem-max-active-gigabytes: 195.46 | mem-max-inactive-gigabytes: 14.369 | mem-max-reserved-gigabytes: 216.89 | mem-alloc-retires: 0 | mem-allocated-count: 571
Deleting CUDA graphs
[after training is done] datetime: 2025-12-19 10:49:16
```

<!-- NCCL trace support removed. Documentation section deleted intentionally. -->

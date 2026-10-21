# Overview

This recipe contains information and scripts to produce performance results for the Llama3.1 70B and 405B training workloads. The scripts help perform environment setup and launch benchmark jobs. Configurations use weak scaling methodology (global batch size scales proportionally with GPU count).

**Note:** pretrain_llama3.1 is the correct workload name for all sizes. For 70B, this recipe intentionally reuses the existing Megatron-Bridge llama3 config, so setup output may reference Meta-Llama-3-\* and experiment/log names may start with pretrain_llama3. This is expected.

## GB300

### FP8

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       | 256-512 |   FP8    |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6 | 192 | False |
| 70b        | 64-512  |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  2  | GPUs\*4 |  2  | False |

### NVFP4

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       | 256-512 |  NVFP4   |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6 | 192 | False |
| 70b        | 64-512  |  NVFP4   |  8192  |   80   | False |  1  |  4  |  1  |  1  |  1  | GPUs/4  |  5  |  1  | GPUs\*4 | 16  | False |

## GB200

### FP8

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       | 256-512 |   FP8    |  8192  |  126   | False |  4  | 16  |  1  |  1  |  1  | GPUs/64 |  4  |  1  | GPUs\*6 | 384 | False |
| 70b        | 64-512  |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  2  | GPUs\*4 |  2  | False |

### NVFP4

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       | 256-512 |  NVFP4   |  8192  |  126   | False |  4  | 16  |  1  |  1  |  1  | GPUs/64 |  8  |  1  | GPUs\*6 | 384 | False |
| 70b        | 64-512  |  NVFP4   |  8192  |   80   | False |  2  |  4  |  1  |  1  |  1  | GPUs/8  |  5  |  1  | GPUs\*4 | 32  | True  |

## B300

### FP8

| Model Size |    GPUs     | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :---------: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       |   256-512   |   FP8    |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6 | 192 | False |
| 70b        |   **64**    |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  1  | GPUs\*4 |  4  | False |
| 70b        | **128-512** |   FP8    |  8192  |   80   | False |  1  |  4  |  1  |  1  |  1  | GPUs/4  |  5  |  1  | GPUs\*4 | 16  | False |

### NVFP4

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       | 256-512 |  NVFP4   |  8192  |  126   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32 |  4  |  1  | GPUs\*6 | 192 | False |
| 70b        | 64-512  |  NVFP4   |  8192  |   80   | False |  1  |  4  |  1  |  1  |  1  | GPUs/4  |  5  |  1  | GPUs\*4 | 16  | False |

## B200

### FP8

| Model Size |    GPUs     | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :---------: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       |   256-512   |   FP8    |  8192  |  126   | False |  4  | 16  |  1  |  1  |  1  | GPUs/64 |  4  |  1  | GPUs\*6 | 384 | False |
| 70b        |   **64**    |   FP8    |  8192  |   80   | True  |  1  |  1  |  1  |  1  |  1  |  GPUs   | NA  |  1  | GPUs\*4 |  4  | False |
| 70b        | **128-512** |   FP8    |  8192  |   80   | False |  2  |  4  |  1  |  1  |  1  | GPUs/8  |  5  |  1  | GPUs\*4 | 32  | False |

### NVFP4

| Model Size |  GPUs   | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |   DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :-----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :-----: | :-: | :-: | :-----: | :-: | :---: |
| 405b       | 256-512 |  NVFP4   |  8192  |  126   | False |  4  | 16  |  1  |  1  |  1  | GPUs/64 |  8  |  1  | GPUs\*6 | 384 | False |
| 70b        | 64-512  |  NVFP4   |  8192  |   80   | False |  2  |  4  |  1  |  1  |  1  | GPUs/8  |  5  |  1  | GPUs\*4 | 32  | False |

## H100

**Note:** For H100, only FP8 CS (current scaling) is supported; FP8 MX is not allowed for this GPU type.

### BF16

| Model Size |  GPUs  | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |    DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :------: | :-: | :-: | :-----: | :-: | :---: |
| 405b       |  1024  |   BF16   |  8192  |  126   | False |  8  |  8  |  2  |  1  |  1  | GPUs/128 |  8  |  1  |  1536   | 192 | False |
| 70b        | 64-512 |   BF16   |  8192  |   80   | False |  4  |  4  |  2  |  1  |  1  | GPUs/32  |  5  |  1  | GPUs\*4 | 128 | False |

### FP8

| Model Size |  GPUs  | Datatype | SeqLen | Layers | FSDP  | TP  | PP  | CP  | EP  | ETP |    DP    | VP  | MBS |   GBS   | GA  |  CG   |
| ---------- | :----: | :------: | :----: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :------: | :-: | :-: | :-----: | :-: | :---: |
| 405b       |  1024  |   FP8    |  8192  |  126   | False |  8  |  8  |  2  |  1  |  1  | GPUs/128 |  8  |  1  |  1536   | 192 | False |
| 70b        | 64-512 |   FP8    |  8192  |   80   | False |  4  |  8  |  1  |  1  |  1  | GPUs/32  |  5  |  1  | GPUs\*4 | 128 | False |

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
  pretrain_example_70b  bf16     128  1234567  No       2026-04-17 13:42  COMPLETED     00:12:34    4.21     1234.56
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

The 70B configuration intentionally reuses the Megatron-Bridge `llama3` config, so it requires Llama 3 family access instead. Request access through [Meta's website](https://www.llama.com/llama-downloads/) and then on the [HuggingFace Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) page. The approval process is not automatic and could take a day or more.

## Slurm

This document references several Slurm commands and parameters. A brief summary is included below. Note that these are guidelines and might not apply to all environments. Consult your system administrator for the parameters specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Used for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES, this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../../../README.md) to prepare the recipe environment:

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_llama3.1`.
- Results and logs are stored under subfolders of `LLMB_WORKLOAD` (see below).

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

Train Llama3.1 70B with FP8 precision on 64 H100 GPUs:

```shell
MODEL_SIZE=70b JOB_TOTAL_GPUS=64 GPU_TYPE=h100 ./launch.sh
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
│       │   ├── checkpoints/                   # Saved checkpoints (when enabled)
│       │   └── nsys_profile/                  # Profiling output (when enabled)
│       │       └── *.nsys-rep files
│       └── [batch scripts and other files]
```

The `<experiment_name>` typically follows these patterns:

- For Llama3 70B: `pretrain_llama3_70b_<dtype>_<config>`
- For Llama3.1 405B: `pretrain_llama3.1_405b_<dtype>_<config>`

**Key files:**

- `log-<experiment_name>.out` - Contains training step timing and performance metrics parsed by `llmb-run jobs`
- `checkpoints/` - Contains saved checkpoints when `ENABLE_CHECKPOINT=true` and `CHECKPOINT_DIR` is not set (see [Run With Checkpoints](#run-with-checkpoints))
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling

Profiling is supported with Nsight Systems or PyTorch Profiler.

## Run Nsight Profiling

To enable profiling with Nsight Systems, set `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

To view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information, visit [Nsight Systems](https://docs.nvidia.com/nsight-systems/).

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

To view the profile traces (\*.nsys-rep files) interactively:

- Install the latest [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your preferred system
- Copy the generated .nsys-rep files to a folder on your preferred system, for example `/home/nsight-traces/`.
- Open Nsight Systems client, then click "File | Open" and select one or more .nsys-rep files from that folder. For more details, see [Reading Your Report in GUI guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#opening-an-existing-report).
- Once loaded you can analyze the workload behavior to learn about any performance bottlenecks associated with the job run.

Because most benchmarking jobs run on multiple GPUs, each run generates multiple .nsys-rep files. The [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis) can help automate the analysis and get to results more quickly using Nsight recipes.

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

Checkpoint save and load are supported for 70B and 405B on GB300, GB200, B300 and B200, at all scales and precisions, except 70B FP8 on GB200, which `launch.sh` restricts to 128 GPUs and up because the `fsdp_dtensor` save runs out of host memory at 64. They are not supported on H100.

## Save Checkpoint

Make sure your file system has enough space for the checkpoint sizes below. The GPU columns list the supported scales:

| Model | Precision | Checkpoint Format | Checkpoint Size |  GB300   |           GB200            |   B300   |   B200   |
| :---: | :-------: | :---------------: | :-------------: | :------: | :------------------------: | :------: | :------: |
|  70b  |    FP8    |  `fsdp_dtensor`   |     ~789 GB     |  64-512  | 128-512 (64 not supported) |  64-512  |  64-512  |
|  70b  |   NVFP4   |   `torch_dist`    |     ~920 GB     |  64-512  |           64-512           |  64-512  |  64-512  |
| 405b  |    FP8    |   `torch_dist`    |     ~5.2 TB     | 256, 512 |          256, 512          | 256, 512 | 256, 512 |
| 405b  |   NVFP4   |   `torch_dist`    |     ~5.2 TB     | 256, 512 |          256, 512          | 256, 512 | 256, 512 |

**Note:** On B300 and B200, 70B FP8 at 128 GPUs and up uses the MXFP8 recipe (`launch.sh` sets `FP8_RECIPE=mx`) and the `torch_dist` format, so its checkpoint format and size may differ from the `fsdp_dtensor` row above.

Checkpointing is functional on GB300, GB200, B300 and B200.

### How to enable

Set `ENABLE_CHECKPOINT=true` to save a checkpoint at the end of training (step 50). To save more often, set `CHECKPOINT_INTERVAL` to the number of steps between saves. Checkpoints are saved in the `checkpoints/` folder of the job's experiment directory (see [Output Locations](#output-locations)). To save them somewhere else, set `CHECKPOINT_DIR` to an absolute path; the folder is created and mounted into the container for you.

```shell
ENABLE_CHECKPOINT=true llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 128
```

### How to validate

Check that the `checkpoints/` folder contains an `iter_0000050` folder, and that the job's `log-<experiment_name>.out` file has a line like:

```shell
successfully saved checkpoint from iteration      50 to /nemo_run/checkpoints
```

## Load Checkpoint

### How to enable

Set `LOAD_CHECKPOINT_PATH` to the absolute path of the save run's `checkpoints/` folder, the one that contains `iter_0000050` and `latest_checkpointed_iteration.txt`. `launch.sh` mounts that folder into the container. Use the same model size, precision and scale as the save run.

```shell
LOAD_CHECKPOINT_PATH=/path/to/checkpoints llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 128
```

The job loads the model weights and then trains for one step. It does not restore the optimizer, learning-rate scheduler or RNG state, and the step count starts again from 0.

### How to validate

Check the job's `log-<experiment_name>.out` file for a `successfully loaded checkpoint from <LOAD_CHECKPOINT_PATH>` line, followed by a single training step (`iteration 1/1`). That step's `lm loss` should be close to the save run's loss at step 50, not the much higher loss of a fresh run's first step.

## Checkpoint Format

Recipes that use Megatron-FSDP can only save and load checkpoints in the `fsdp_dtensor` format, so `launch.sh` selects it for them: 70B FP8 on GB300 and GB200, and 70B FP8 at 64 GPUs on B300 and B200. All other configurations, including every 405B configuration, use the default `torch_dist` format. To override this choice, set `CKPT_FORMAT`.

Load runs and `fsdp_dtensor` runs pass `checkpoint.*` settings that the launcher doesn't recognize, so it prints `Ignoring unrecognized arguments: checkpoint...`. This is expected; the settings still reach training.

<!-- NCCL trace support removed. Documentation section deleted intentionally. -->

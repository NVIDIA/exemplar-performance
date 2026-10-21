# Overview

This recipe contains information and scripts to produce performance results for the LLAMA3 70B finetuning (LoRa) workload. The scripts help perform environment setup and launch benchmark jobs. It supports both BF16 and FP8 precisions.

## GB300

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     |   BF16    |  8   |  4096  |   80   |  1  |  1  |  1  |  8  | N/A |  1  | 32  |  4  |
|    70b     |    FP8    |  8   |  4096  |   80   |  1  |  2  |  1  |  4  | 20  |  1  | 32  |  8  |
|    70b     |   BF16    |  16  |  4096  |   80   |  1  |  1  |  1  | 16  | N/A |  1  | 64  |  4  |
|    70b     |    FP8    |  16  |  4096  |   80   |  1  |  2  |  1  |  8  | 20  |  1  | 64  |  8  |

## GB200

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     |   BF16    |  8   |  2048  |   80   |  1  |  2  |  1  |  4  | 20  |  1  | 64  | 16  |
|    70b     |    FP8    |  8   |  4096  |   80   |  1  |  2  |  1  |  4  | 20  |  1  | 32  |  8  |
|    70b     |   BF16    |  16  |  2048  |   80   |  1  |  2  |  1  |  8  | 20  |  1  | 128 | 16  |
|    70b     |    FP8    |  16  |  4096  |   80   |  1  |  2  |  1  |  8  | 20  |  1  | 64  |  8  |

## B300

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     |    FP8    |  8   |  4096  |   80   |  1  |  2  |  1  |  4  | N/A |  1  | 32  |  8  |
|    70b     |   BF16    |  8   |  4096  |   80   |  1  |  1  |  1  |  8  | N/A |  1  | 32  |  4  |
|    70b     |    FP8    |  16  |  4096  |   80   |  1  |  2  |  1  |  8  | N/A |  1  | 64  |  8  |
|    70b     |   BF16    |  16  |  4096  |   80   |  1  |  1  |  1  | 16  | N/A |  1  | 64  |  4  |

## B200

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     |   BF16    |  8   |  4096  |   80   |  1  |  2  |  1  |  4  | N/A |  1  | 32  |  8  |
|    70b     |    FP8    |  8   |  4096  |   80   |  1  |  2  |  1  |  4  | 20  |  1  | 32  |  8  |
|    70b     |   BF16    |  16  |  4096  |   80   |  1  |  2  |  1  |  8  | N/A |  1  | 64  |  8  |
|    70b     |    FP8    |  16  |  4096  |   80   |  1  |  2  |  1  |  8  | 20  |  1  | 64  |  8  |

## H100

| Model Size | Precision | GPUs | SeqLen | Layers | TP  | PP  | CP  | DP  | VP  | MBS | GBS | GA  |
| :--------: | :-------: | :--: | :----: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|    70b     |   BF16    |  8   |  4096  |   80   |  1  |  4  |  1  |  2  | 20  |  1  | 32  | 16  |
|    70b     |    FP8    |  8   |  4096  |   80   |  2  |  4  |  1  |  1  | 20  |  1  | 32  | 32  |
|    70b     |   BF16    |  16  |  4096  |   80   |  1  |  4  |  1  |  4  | 20  |  1  | 64  | 16  |
|    70b     |    FP8    |  16  |  4096  |   80   |  2  |  4  |  1  |  2  | 20  |  1  | 64  | 32  |

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

Access to Llama 3 70B must be requested through the [HuggingFace Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct). The approval process is not automatic and could take a day or more.

## Slurm

This document references several Slurm commands and parameters. A brief summary is included below. Note that these are guidelines and might not apply to all environments. Consult your system administrator for the parameters specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Used for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES, this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Prepare environment

Use the **installer** referenced in the [main README](../../README.md) to prepare the recipe environment:

Note, that a new directory layout and key variables are now used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/finetune_llama3`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Prepare Checkpoint

The recommended method to prepare the model checkpoint is by using the **installer** referenced in the [main README](../../README.md).

When the installer runs the setup for this recipe, it will run an interactive `srun` job to import the Llama-3 70B Finetuning checkpoint. Please note the following:

- This step runs interactively as part of the installer and may take some time to complete.
- It requires GPU resources. Ensure your SLURM partition has available GPUs.
- To verify that the checkpoint has been downloaded successfully, refer to the [Storage Requirements and Verification](#storage-requirements-and-verification) section.

If you encounter issues with the automated download via the installer, you can use the manual instructions provided in the "[Manual Checkpoint Download (Troubleshooting Only)](#manual-checkpoint-download-troubleshooting-only)" section at the end of this document. This method is intended for troubleshooting purposes only.

## Prepare Dataset

The LLAMA3 70B LoRa finetuning uses the SQUAD dataset. The dataset is generated automatically on the first run via `setup_experiment.py` and will be placed under `$LLMB_WORKLOAD/checkpoint_and_dataset/datasets`. No manual dataset preparation is required.

# Run Finetuning

Once the environment has been prepared, it is time to train a model. The finetuning runs for the first 50 steps and then stops. Log files and results are stored under the `${LLMB_WORKLOAD}/experiments/` folder (see Output Locations for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w finetune_llama3 --dtype fp8 --scale 8

# Example with BF16 precision
llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8
```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags instead of `ADDITIONAL_SLURM_PARAMS`.

Use a Slurm reservation:

```bash
llmb-run submit -w finetune_llama3 --dtype fp8 --scale 8 --reservation my_reservation
```

Run on specific nodes:

```bash
llmb-run submit -w finetune_llama3 --dtype fp8 --scale 8 --nodelist node001,node002
```

Exclude specific nodes:

```bash
llmb-run submit -w finetune_llama3 --dtype fp8 --scale 8 --exclude node003,node004
```

Combine multiple parameters:

```bash
llmb-run submit -w finetune_llama3 --dtype fp8 --scale 8 --nodelist node001,node002 --reservation my_reservation --slurm-arg exclusive
```

For more details on `llmb-run` usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run finetuning directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the finetuning commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the recipe directory: `cd $LLMB_INSTALL/llmb_repo/llama3/finetune`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [MODEL_SIZE=<size>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use (e.g., 8)
- `GPU_TYPE`: Type of GPU hardware
  - `gb300` - NVIDIA GB300 GPUs
  - `gb200` - NVIDIA GB200 GPUs
  - `b300` - NVIDIA B300 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `h100` - NVIDIA H100 GPUs

**Optional:**

- `DTYPE`: Precision format (default: `fp8`)
  - `fp8` - FP8 precision
  - `bf16` - BFloat16 precision
- `MODEL_SIZE`: Model variant (fixed: `70b`)
  - `70b` - 70 billion parameter model (only supported size)
- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated
  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`

### Example Commands

Train LLAMA3 70B LoRa with BF16 precision on 8 GB300 GPUs:

```shell
JOB_TOTAL_GPUS=8 DTYPE=bf16 GPU_TYPE=gb300 ./launch.sh
```

Train with FP8 precision on 8 GB300 GPUs:

```shell
JOB_TOTAL_GPUS=8 DTYPE=fp8 GPU_TYPE=gb300 ./launch.sh
```

Train on 8 B300 GPUs with FP8 precision:

```shell
JOB_TOTAL_GPUS=8 DTYPE=fp8 GPU_TYPE=b300 ./launch.sh
```

Train on 8 B200 GPUs with FP8 precision:

```shell
JOB_TOTAL_GPUS=8 DTYPE=fp8 GPU_TYPE=b200 ./launch.sh
```

Train on 8 H100 GPUs with FP8 precision:

```shell
JOB_TOTAL_GPUS=8 DTYPE=fp8 GPU_TYPE=h100 ./launch.sh
```

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

The `<experiment_name>` typically follows the pattern: `lora_llama3_70b_<dtype>_<scale>_<config>`

**Key files:**

- `log-<experiment_name>.out` - Contains training step timing and performance metrics parsed by `llmb-run jobs`
- `nsys_profile/` - Contains profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`

# Profiling

Profiling is supported with Nsight Systems or PyTorch Profiler.

## Run Nsight Profiling

To enable profiling with Nsight Systems, use the `-p` flag with `llmb-run` or set `ENABLE_PROFILE=true` when submitting your job. The job will run for a total of 50 steps where steps 45-50 will be profiled.

To view the resulting profiles, ensure you have the latest version of Nsight Systems installed. For more information, visit [Nsight Systems](https://docs.nvidia.com/nsight-systems/).

### Profiling job details:

- **MPI Ranks:** all (0 through `JOB_TOTAL_GPUS`-1; e.g. 0-7 at scale 8, 0-15 at scale 16)
- **Job Steps:** 45-50
- **Output Location:** Profiling output saved alongside finetuning results (see Output Locations)
- **Filename format:** `profile_${SLURM_JOB_ID}_nodeId_rankId.nsys-rep`

**Example command:**

```shell
llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8 -p
```

### Customizing profiling behavior:

- Specify job steps to profile:
  - `PROFILE_START_STEP`: start profiling on this job step.
  * Default: 45
  - `PROFILE_STOP_STEP`: stop profiling on this job step.
  * Default: 50

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
ENABLE_PYTORCH_PROFILE=true llmb-run submit -w finetune_llama3 --dtype bf16 --scale 8
```

For details on the PyTorch Profiler and how to view resulting traces, see the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

<!-- NCCL trace support removed. Documentation section deleted intentionally. -->

# Storage Requirements and Verification

**Note:** The LLAMA3 70B LoRa checkpoint is approximately **747GB** in size, which requires significant storage space. Ensure your file system has sufficient space (at least 1TB recommended) to accommodate the checkpoint, dataset, and temporary files.

To verify that the checkpoint has been correctly downloaded and is available, check for the following directory structure within `${LLMB_WORKLOAD}/checkpoint_and_dataset/` after the installer's setup tasks have completed:

```text
checkpoint_and_dataset/
├── datasets/          # Dataset files (populated on first run)
├── hub/              # HuggingFace hub cache
├── llama3_70b/        # Model checkpoints (~747GB)
└── xet/
```

The `datasets/` folder will be populated with the SQUAD dataset automatically on the first run via `setup_experiment.py`. After the first run, the folder structure will look like:

```text
datasets/
├── packed/
├── squad/
│   └── plain_text/
│       └── 0.0.0/
│           └── 7b6d24c440a36b6815f21b70d25016731768db1f/
│               ├── cache-aedd28d6a9c0b57f.arrow
│               ├── cache-ee746ac879d304bb.arrow
│               ├── dataset_info.json
│               ├── squad-train.arrow
│               └── squad-validation.arrow
├── test.jsonl
├── test.jsonl.idx.info
├── test.jsonl.idx.npy
├── training.jsonl
├── training.jsonl.idx.info
├── training.jsonl.idx.npy
├── validation.jsonl
├── validation.jsonl.idx.info
└── validation.jsonl.idx.npy
```

Within the `llama3_70b/` folder, you should be able to find the downloaded checkpoint with the following structure:

```text
llama3_70b/
├── context/
└── weights/
    ├── __0_0.distcp    # 66GB
    ├── __0_1.distcp    # 66GB
    ├── common.pt
    └── metadata.json
```

# Manual Checkpoint Download (Troubleshooting Only)

These instructions are provided for troubleshooting purposes, in case the automated checkpoint import via the installer encounters issues.

## Activate Virtual Environment

Before running the `download_ckpt_dataset.sh` script, you must activate the virtual environment associated with the `finetune_llama3` workload. This environment contains the necessary dependencies for Megatron-Bridge to run the download process.

To find the correct virtual environment path, consult the `cluster_config.yaml` file located in your `${LLMB_INSTALL}` directory. Look under the `workloads.config.finetune_llama3.venv_path` key.

Once you have the `venv_path`, activate it using one of the following commands, depending on whether it's a `venv` or `conda` environment:

- **For `venv` environments:**
  ```bash
  source <venv_path>/bin/activate
  ```
- **For `conda` environments:**
  ```bash
  conda activate <venv_path>
  ```

## Important Environment Variables

The `download_ckpt_dataset.sh` script relies on several environment variables that are typically set by the installer. When running this script manually, you must ensure these variables are set in your environment:

- `HF_TOKEN`: Your HuggingFace access token. (e.g., `export HF_TOKEN=<your token>`)
- `GPU_TYPE`: The type of GPU hardware you are using (`gb300`, `gb200`, `b300`, `b200` or `h100`). (e.g., `export GPU_TYPE=h100`)
- `LLMB_INSTALL`: The top-level installation directory for all benchmarking artifacts. (e.g., `export LLMB_INSTALL=/path/to/llm_benchmarking_install`)
- `LLMB_WORKLOAD`: The workload-specific directory. This is usually derived from `LLMB_INSTALL`. (e.g., `export LLMB_WORKLOAD=${LLMB_INSTALL}/workloads/finetune_llama3`)
- `SBATCH_ACCOUNT`: Your Slurm account. (e.g., `export SBATCH_ACCOUNT=your_slurm_account`)
- `SBATCH_PARTITION`: The Slurm partition (queue) to use. (e.g., `export SBATCH_PARTITION=your_partition`)
- `SBATCH_GPUS_PER_NODE`: If your Slurm partition requires GRES set to at least one. (e.g., `export SBATCH_GPUS_PER_NODE=1`)
- `TIME_LIMIT`: The default is 55 minutes. Increase this value if you experience timeouts, particularly if your network speed is slower. Use the `HH:MM:SS` format (e.g., `export TIME_LIMIT=01:30:00` for 1 hour and 30 minutes).

## Download Commands

Once these environment variables are set, you can use the following command to manually download the checkpoint:

```shell
./download_ckpt_dataset.sh
```

**Note:** This command may take some time to complete. Refer to the [Storage Requirements and Verification](#storage-requirements-and-verification) section to confirm successful download.

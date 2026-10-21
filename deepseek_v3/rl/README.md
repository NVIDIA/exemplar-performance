# Overview

This recipe contains information and scripts to produce performance results for the DeepSeek-V3 reinforcement learning (RL) workload using the [NeMo-RL](https://github.com/NVIDIA/NeMo-RL) framework. The scripts help perform environment setup and launch benchmark jobs using the GRPO (Group Relative Policy Optimization) algorithm and [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

## GB200

| Algorithm | On/Off Policy | T-Max Seq | #-GPUs | G-GBS | T-GBS | Generation TP/PP | Training TP/CP/EP/PP/VPP |
| :-------- | :------------ | :-------- | :----: | :---: | :---: | :--------------: | :----------------------: |
| GRPO      | 1-step Off    | 1,536     |  256   |  512  |  512  |       16/1       |      1/1/16/8/None       |

# Performance Measurement and Analysis

Performance is reported as:

- `s/iter` — wall-clock seconds per GRPO training step
- `Tokens/s/GPU` — throughput per GPU

Each benchmark runs for 10 steps; performance numbers are averaged over iterations 3–7.

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
  Workload           DType  Scale   Job ID  Profile  Submit Time       Slurm Status  Elapsed   s/iter  TFLOPS/GPU  Tokens/s/GPU
  rl_example_70b     bf16     256  1234567  No       2026-04-17 13:42  COMPLETED     00:27:49   50.01                    25.00
  rl_example_70b     bf16     256  1234589  No       2026-04-17 14:05  RUNNING       00:03:11
```

Blank `s/iter` or `Tokens/s/GPU` means the job has not finished yet, or the log did not contain enough completed iterations. See the [llmb-run README](../../cli/llmb-run/README.md#jobs-command) for the full command reference.

# Prerequisites

- A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via `export HF_TOKEN=<your token>`.

- A DeepSeek-V3 BF16 checkpoint is required. It is prepared as part of the normal installer flow (see [Prepare Model Assets](#prepare-model-assets)).

- Requires Python 3.12.x, or conda.

## Request Access

No special HuggingFace model gate is required beyond basic authentication.

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

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, checkpoints, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/rl_deepseek-v3`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see [Output Locations](#output-locations)).

# Prepare Model Assets

DeepSeek-V3 RL training requires the model weights from [`deepseek-ai/DeepSeek-V3`](https://huggingface.co/deepseek-ai/DeepSeek-V3) converted from FP8 to BF16. The installer downloads the FP8 HuggingFace snapshot and runs the conversion setup task as part of the normal workload installation.

The conversion runs as a Slurm setup job. Ensure it has completed successfully before launching the benchmark.

Prepared assets are stored under the workload directory:

- `$LLMB_WORKLOAD/DeepSeek-V3-FP8`
- `$LLMB_WORKLOAD/DeepSeek-V3-BF16`

# Run RL Training

Once the environment has been prepared, run the benchmark using the launch script. The training runs for `MAX_STEPS` steps (default: 10) and then stops. Three Slurm jobs are submitted in a dependency chain. Log files and results are stored under `${LLMB_WORKLOAD}/experiments/` (see [Output Locations](#output-locations)).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w rl_deepseek-v3 -s 671b --dtype bf16 --scale 256
```

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important:**

- Ensure your virtual environment is activated before running the commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/deepseek_v3/rl/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use
- `GPU_TYPE`: Type of GPU hardware
  - `gb200` - NVIDIA GB200 GPUs

**Optional:**

- `MAX_STEPS`: Number of GRPO training steps (default: `10`)
- `ASYNC_MODE`: Enable async RL training mode (default: `true`)
- `WANDB_API_KEY`: If set, enables Weights & Biases logging

### Example Commands

Train DeepSeek-V3 on 256 GB200 GPUs:

```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 ./launch.sh
```

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```text
experiments/
└── <job_name>/
    └── <timestamp>/
        |── ${SLURM_PROCESS_ID}-logs # Ray driver logs
        ├── logs/                    # TensorBoard event files
        ├── run.log                  # Full training stdout/stderr
        ├── metrics.json             # Extracted training metrics
```

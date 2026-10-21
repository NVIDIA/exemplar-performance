# Overview

This recipe contains information and scripts to produce performance results for the Nemotron-3 Super reinforcement learning (RL) post-training using the [NeMo-RL](https://github.com/NVIDIA/NeMo-RL) framework. The scripts help perform environment setup and launch benchmark jobs using the GRPO (Group Relative Policy Optimization) algorithm against the NeMo Gym `single_step_tool_use_with_argument_comparison` environment (stage `stage2.1-swe1`, a single-turn tool-calling "pivot" stage scored by argument comparison, distinct from the heavier multi-turn `stage2_swe2` stage), training on the [Nemotron-RL-Super-Training-Blends SWE1](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends#swe-1) dataset.

## GB200 / B200

| Algorithm | Stage         | T-Max Seq | #-GPUs | T-GPUs | G-GPUs | G-GBS | T-GBS | Generation TP/PP/EP | Training TP/CP/EP/PP |
| :-------- | :------------ | :-------- | :----: | :----: | :----: | :---: | :---: | :-----------------: | :------------------: |
| GRPO      | stage2.1-swe1 | 131,072   |   72   |   64   |   8    |  16   |  16   |        8/1/1        |       8/8/8/1        |

Generation runs disaggregated from training (`policy.generation.colocated.enabled: false`), so the generation GPUs (`G-GPUs`) are separate from the training GPUs (`T-GPUs`); `#-GPUs` is the total (`T-GPUs` + `G-GPUs`).

# Performance Measurement and Analysis

## `parse_results.py`

```bash
cd $LLMB_INSTALL/llmb_repo/nemotron3/rl/
python parse_results.py --input_dir <path to experiment results, e.g. $LLMB_WORKLOAD/experiments>
```

For every experiment found under `--input_dir` (any directory containing a `ray-driver.log`), it:

- Parses the resolved training config (the `Final config:` block logged by NeMo-RL) and per-step training results.
- Dumps `results_<job_id>.json` (`{"config": ..., "step_metrics": {...}}`)
- Prints a `job_id | num_iter | min | max | mean | p50 | p60 | p75 | p80 | p90 | p99` table per timing field, across all discovered jobs, and dumps the same stats to `timing_stats_summary.json` and `timing_stats_tables.xlsx` in `--input_dir`.
- With `--plot_step_metrics` (default: on), also saves a per-job `timing_vs_iterations_<job_id>.png` and, once across all jobs, `total_step_time_ecdf.png` in `--input_dir`.

**Options:**

- `--input_dir` (required): directory to search for `ray-driver.log` files.
- `--plot_step_metrics` / `--no-plot_step_metrics`: enable/disable timing metrics per step output (default: enabled).

# Prerequisites

- A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). Add the generated token to your environment via `export HF_TOKEN=<your token>`.

- The `NVIDIA-Nemotron-3-Super-120B-A12B-BF16` checkpoint and `Nemotron-RL-Super-Training-Blends` dataset are required. Both are prepared as part of the normal installer flow (see [Prepare Model Assets](#prepare-model-assets)).

- Requires Bash 4.2 or greater.

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
- `LLMB_WORKLOAD`: Workload-specific directory, `${LLMB_INSTALL}/workloads/rl_nemotron_3`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see [Output Locations](#output-locations)).

# Prepare Model Assets

Nemotron-3 Super RL training requires the [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) checkpoint and the [`nvidia/Nemotron-RL-Super-Training-Blends`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Super-Training-Blends) dataset. Both are downloaded by the installer as part of the normal workload installation.

The dataset requires a setup step (`dataset_preparation.sh`, run as a Slurm job) that fills dataset placeholders and splits each blend into `train-split.jsonl` / `val-split.jsonl`. Ensure it has completed successfully before launching the benchmark.

Prepared assets are stored under the workload directory:

- `$LLMB_WORKLOAD/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
- `$LLMB_WORKLOAD/Nemotron-RL-Super-Training-Blends`

# Run RL Training

Once the environment has been prepared, run the benchmark using the launch script. Log files, code snapshots, and results are stored under `${LLMB_WORKLOAD}/experiments/` (see [Output Locations](#output-locations)).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run
llmb-run submit -w rl_nemotron_3 -s 120b --dtype bf16 --scale 72
```

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important:**

- Ensure your virtual environment is activated before running the commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/nemotron3/rl/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use
- `GPU_TYPE`: Type of GPU hardware
  - `gb200` - NVIDIA GB200 GPUs
  - `b200` - NVIDIA B200 GPUs

**Optional:**

- `WANDB_API_KEY`: If set, enables Weights & Biases logging

### Example Commands

Train Nemotron 3 Super on 72 B200 GPUs:

```shell
JOB_TOTAL_GPUS=72 GPU_TYPE=b200 ./launch.sh
```

# Output Locations

All benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```text
experiments/
├── total_step_time_ecdf.png                       # parse_results.py output (across all jobs)
├── timing_stats_summary.json                      # parse_results.py output (across all jobs)
├── timing_stats_tables.xlsx                       # parse_results.py output (across all jobs)
└── <job_name>/
    └── <timestamp>/
        ├── llmb-config_<job_id>.yaml
        ├── code_snapshots/
        │   └── stage2.1-swe1/
        │       ├── run.log                        # Full training stdout/stderr
        │       └── <job_id>-logs/
        │           └── ray-driver.log             # Ray driver log (config + per-step results)
        ├── logs/
        │   └── stage2.1-swe1/
        │       └── exp_001/                       # TensorBoard/wandb logs, train_data_step*.jsonl
        ├── results_<job_id>.json                  # parse_results.py output (config + step_metrics)
        └── timing_vs_iterations_<job_id>.png      # parse_results.py output (per job)
```

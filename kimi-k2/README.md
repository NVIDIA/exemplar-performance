# Overview

This recipe contains information and scripts to produce performance results for the Kimi-K2 (1T parameter MoE) pre-training workloads. The scripts help perform environment setup and launch benchmark jobs. Configurations use weak scaling methodology (global batch size scales proportionally with GPU count).

The tables below list the GPU counts in `metadata.yaml` for this recipe (256 and 512), with parallelism and batch sizes from Megatron-Bridge `configs/kimi/kimi_workload_base_configs.py` (recipe `kimi_k2`).

## GB300

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: |
| FP8       | 256  |  4096  |  1  |  4  |  1  | 64  | 64  |  4  |  1  | 4096 |
| FP8       | 512  |  4096  |  1  |  4  |  1  | 64  | 128 |  4  |  1  | 8192 |

## GB200

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: |
| FP8       | 256  |  4096  |  1  |  4  |  1  | 64  | 64  |  4  |  1  | 2048 |
| FP8       | 512  |  4096  |  1  |  4  |  1  | 64  | 128 |  4  |  1  | 4096 |

## B300

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: |
| FP8       | 256  |  4096  |  1  | 16  |  1  | 16  | 16  | N/A |  2  | 4096 |
| FP8       | 512  |  4096  |  1  | 16  |  1  | 16  | 32  | N/A |  2  | 8192 |

## B200

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: |
| FP8       | 256  |  4096  |  1  | 16  |  1  | 16  | 16  | N/A |  1  | 4096 |
| FP8       | 512  |  4096  |  1  | 16  |  1  | 16  | 32  | N/A |  1  | 8192 |

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

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. These are a guide and might not apply to all environments. Consult your system administrator for parameters specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` – Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` – Slurm account for accounting.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` – Set to all GPUs per node if your cluster uses GRES.

These can be set via environment variables or the corresponding `sbatch` flags.

## Prepare environment

Use the **installer** referenced in the [main README](../README.md) to prepare the recipe environment.

Directory layout and key variables:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc.).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_kimi-k2`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see [Output Locations](#output-locations)).

# Prepare Dataset

Kimi-K2 training in this recipe uses synthetic data; no dataset preparation is required.

# Run Training

After the environment is prepared, run training. The run executes for the first 50 steps and then stops. Logs and results are written under `${LLMB_WORKLOAD}/experiments/` (see [Output Locations](#output-locations)).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Example: Kimi-K2 1T, FP8 MX, 256 GB300 GPUs (cluster `gpu_type` in llmb-run config must match, e.g. gb300)
llmb-run submit -w pretrain_kimi-k2 -s 1t --dtype fp8 --scale 256

# 512 GPUs (same pattern; set `gpu_type` in your llmb-run cluster config to match the cluster)
llmb-run submit -w pretrain_kimi-k2 -s 1t --dtype fp8 --scale 512
```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags instead of `ADDITIONAL_SLURM_PARAMS`.

Use a Slurm reservation:

```bash
llmb-run submit -w pretrain_kimi-k2 -s 1t --dtype fp8 --scale 256 --reservation my_reservation
```

Run on specific nodes:

```bash
llmb-run submit -w pretrain_kimi-k2 -s 1t --dtype fp8 --scale 256 --nodelist node001,node002
```

Exclude specific nodes:

```bash
llmb-run submit -w pretrain_kimi-k2 -s 1t --dtype fp8 --scale 256 --exclude node003,node004
```

Combine multiple parameters:

```bash
llmb-run submit -w pretrain_kimi-k2 -s 1t --dtype fp8 --scale 256 --nodelist node001,node002 --reservation my_reservation --slurm-arg exclusive
```

For more details on `llmb-run` usage, see the [llmb-run documentation](../cli/llmb-run/README.md).

## Direct Method

Alternatively, you can run training directly using the launch script. This method provides more control over individual parameters and environment variables.

**Important**:

- Ensure your virtual environment is activated before running the training commands below. If you used the installer with conda, run `conda activate $LLMB_INSTALL/venvs/<env_name>`. If you used the installer with python venv, run `source $LLMB_INSTALL/venvs/<env_name>/bin/activate`.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/kimi-k2/`

### Command Template

```shell
JOB_TOTAL_GPUS=<number> GPU_TYPE=<type> [DTYPE=<precision>] [ADDITIONAL_SLURM_PARAMS=<params>] ./launch.sh
```

### Environment Variables

**Required:**

- `JOB_TOTAL_GPUS`: Number of GPUs to use.
- `GPU_TYPE`: `gb300`, `gb200`, `b300`, or `b200` (listed in `metadata.yaml`). `launch.sh` rejects any other value. GB300, GB200, and B300 run on NeMo 26.06.01; B200 runs on NeMo 26.04.01.

**Optional:**

- `DTYPE`: Precision format (fixed: `fp8`).

- `FP8_RECIPE`: FP8 recipe (fixed: `mx`; default in `launch.sh`).

- `ADDITIONAL_SLURM_PARAMS`: Extra `sbatch` flags (e.g. `--nodelist`, `--reservation`), semicolon-separated

  - Example: `"nodelist=node001,node002;reservation=my_reservation;exclusive"`

**Note:** This workload only supports FP8 precision and the **MX** FP8 recipe (`fp8_mx` / `FP8_RECIPE=mx`).

### Example Commands

Kimi-K2 1T, FP8 MX, 256 GB300 GPUs:

```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=gb300 DTYPE=fp8 ./launch.sh
```

Kimi-K2 1T, FP8 MX, 512 GB300 GPUs:

```shell
JOB_TOTAL_GPUS=512 GPU_TYPE=gb300 DTYPE=fp8 ./launch.sh
```

Kimi-K2 1T, FP8 MX, 256 GB200 GPUs:

```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=gb200 DTYPE=fp8 ./launch.sh
```

Kimi-K2 1T, FP8 MX, 512 GB200 GPUs:

```shell
JOB_TOTAL_GPUS=512 GPU_TYPE=gb200 DTYPE=fp8 ./launch.sh
```

Kimi-K2 1T, FP8 MX, 256 B300 GPUs:

```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=b300 DTYPE=fp8 ./launch.sh
```

Kimi-K2 1T, FP8 MX, 512 B300 GPUs:

```shell
JOB_TOTAL_GPUS=512 GPU_TYPE=b300 DTYPE=fp8 ./launch.sh
```

Kimi-K2 1T, FP8 MX, 256 B200 GPUs:

```shell
JOB_TOTAL_GPUS=256 GPU_TYPE=b200 DTYPE=fp8 ./launch.sh
```

Kimi-K2 1T, FP8 MX, 512 B200 GPUs:

```shell
JOB_TOTAL_GPUS=512 GPU_TYPE=b200 DTYPE=fp8 ./launch.sh
```

# Output Locations

Benchmark results are saved under `$LLMB_WORKLOAD/experiments/` with the following structure:

```text
experiments/
├── <experiment_name>/
│   └── <experiment_name>_<timestamp>/
│       ├── <experiment_name>/
│       │   ├── log-<experiment_name>.out      # Main training log (used for timing analysis)
│       │   ├── sbatch_<experiment_name>.out   # Batch script output
│       │   └── nsys_profile/                   # Profiling output (when enabled)
│       │       └── *.nsys-rep
│       └── [batch scripts and other files]
```

The `<experiment_name>` typically follows the pattern: `pretrain_kimi-k2_1t_<dtype>_<scale>_<config>`.

**Key files:**

- `log-<experiment_name>.out` – Training step timing and metrics parsed by `llmb-run jobs`.
- `nsys_profile/` – Profiling traces when using the `-p` flag with `llmb-run` or when `ENABLE_PROFILE=true`.

# Run Nsight Profiling

To enable Nsight Systems profiling, set `ENABLE_PROFILE=true` or use the `-p` flag when submitting your job. The job runs 50 steps; steps 45–50 are profiled.

Install the latest [Nsight Systems](https://docs.nvidia.com/nsight-systems/) to view the resulting reports.

## Profiling job details

- **MPI Ranks:** all ranks
- **Steps profiled:** 45–50 (configurable)
- **Output:** under the same experiment directory as training results
- **Filename pattern:** `profile_{SLURM_JOBID}_{SLURM_NODEID}_{SLURM_PROCID}.nsys-rep`

**Example:**

```shell
ENABLE_PROFILE=true JOB_TOTAL_GPUS=256 GPU_TYPE=gb300 ./launch.sh
```

```shell
llmb-run submit -w pretrain_kimi-k2 -s 1t --dtype fp8 --scale 256 -p
```

## Optional profiling options

- `PROFILE_START_STEP`: first step to profile (default 45).
- `PROFILE_STOP_STEP`: last step to profile (default 50).
- `ENABLE_GPU_METRICS`: set to `true` to collect GPU metrics during Nsight profiling (default false).

**Example with GPU metrics:**

```shell
ENABLE_PROFILE=true ENABLE_GPU_METRICS=true JOB_TOTAL_GPUS=256 GPU_TYPE=gb300 ./launch.sh
```

## Viewing results

- Install the [Nsight Systems client](https://developer.nvidia.com/nsight-systems/get-started) on your machine.
- Copy the generated `.nsys-rep` files (e.g. to `/home/nsight-traces/`).
- In Nsight Systems: File → Open and select one or more `.nsys-rep` files.
- For multi-GPU runs, see the [Multi-Report Analysis Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#multi-report-analysis).

See the [Nsight Systems tutorials](https://developer.nvidia.com/nsight-systems/get-started#tutorials) for a quick start.

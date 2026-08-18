# Overview

This recipe contains information and scripts to produce performance results for the DeepSeek-V3 pre-training workload using the **TorchTitan** framework. The scripts help perform environment setup and launch benchmark jobs.

TorchTitan is a PyTorch-native platform for large-scale generative AI training.

## Supported GPU Configurations

This recipe supports **GB200**, **B200**, and **H100** GPUs. The tables below summarize benchmark configurations associated with this recipe; the current launch-script defaults and overridable environment variables are documented in [Run Training](#run-training).

BF16 is supported on all GPUs; FP8 is supported on B200 and GB200 only.

## GB200

| Size | Precision | GPUs | SeqLen | Steps | DP  | TP  | EP  | PP  | MBS | GBS  | GA  | Dataset |
| ---- | :-------: | :--: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :--: | :-: | :-----: |
| 671B |   BF16    | 256  |  4096  |  60   | 256 |  1  | 64  |  1  |  8  | 2048 |  1  |   C4    |
| 671B |    FP8    | 256  |  4096  |  60   | 256 |  1  | 64  |  1  |  8  | 2048 |  1  |   C4    |

## B200

| Size | Precision | GPUs | SeqLen | Steps | DP  | TP  | EP  | PP  | MBS | GBS  | GA  | Dataset |
| ---- | :-------: | :--: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :--: | :-: | :-----: |
| 671B |   BF16    | 256  |  4096  |  60   | 256 |  1  | 64  |  1  |  8  | 2048 |  1  |   C4    |
| 671B |    FP8    | 256  |  4096  |  60   | 256 |  1  | 64  |  1  |  8  | 2048 |  1  |   C4    |

## H100

| Size | Precision | GPUs | SeqLen | Steps | DP  | TP  | EP  | PP  | MBS | GBS  | GA  | Dataset |
| ---- | :-------: | :--: | :----: | :---: | :-: | :-: | :-: | :-: | :-: | :--: | :-: | :-----: |
| 671B |   BF16    | 512  |  4096  |  60   | 64  |  1  | 64  |  8  | 16  | 1024 |  1  |   C4    |

# Prerequisites

## Hugging Face Account

A Hugging Face account and access token are required to download the tokenizer and dataset reliably. Although the assets are public, unauthenticated downloads are heavily rate-limited. You will need to:

1. [Create a Hugging Face access token](https://huggingface.co/settings/tokens)
2. Add the generated token to your environment:

```bash
export HF_TOKEN=<your token>
```

## Python Requirements

Use the installer-managed Python environment for installation, setup tasks, and `llmb-run`.

## Request Access

No special model access is required to run this benchmark. The DeepSeek-V3.1-Base tokenizer is publicly available on Hugging Face.

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
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/pretrain_deepseek-v3-torchtitan`.
- `TORCHTITAN_HOME`: TorchTitan installation directory, e.g. `${LLMB_WORKLOAD}/torchtitan`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see [Output Locations](#output-locations) below).

## Installation Steps

The installer will automatically:

1. Pull and convert the PyTorch container image (`nvidia/pytorch:25.12-py3`)
2. Clone the TorchTitan repository (`https://github.com/elfiegg/torchtitan.git` at commit `b5351f166b2751684aeb1d2cb15ceb619d383ae0`)
3. Install TorchTitan into the container (`install_torchtitan_to_container.sh`)
4. Download the DeepSeek-V3.1-Base tokenizer from Hugging Face (`download_hf_assets.sh`)
5. Download the C4 dataset from Hugging Face using the installer-managed dataset download feature

**Note**: The tokenizer download is performed as a setup task, while the dataset is defined in the `downloads.huggingface.repos` section of `metadata.yaml`.

# Prepare Dataset

The C4 dataset is automatically downloaded by `llmb-install`. The installer fetches the English subset of the C4 dataset from Hugging Face and stores it in `$LLMB_INSTALL/datasets/c4`.

# Run Training

Once the environment has been prepared, it is time to train the model. The launch script runs for 60 steps by default (configurable with `TRAINING_STEPS`). Log files and results are stored under `${LLMB_WORKLOAD}/experiments/` in per-job folders (see [Output Locations](#output-locations) for details).

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically, creates managed experiment directories, and should be the default path for most users.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run DeepSeek-V3 671B BF16 (scale = number of GPUs)
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 512

# Run FP8 on supported GPU types
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype fp8 --scale 256
```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags. This workload uses the `configured_sbatch` launcher, so these flags are applied to the outer `sbatch` submission.

Use a Slurm reservation:

```bash
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256 --reservation my_reservation
```

Run on specific nodes:

```bash
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256 --nodelist node001,node002
```

Exclude specific nodes:

```bash
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256 --exclude node003,node004
```

Combine multiple parameters:

```bash
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256 --nodelist node001,node002 --reservation my_reservation --slurm-arg exclusive
```

For more details on `llmb-run` usage, see the [llmb-run documentation](../../../cli/llmb-run/README.md).

Most environment-variable overrides documented below can also be passed through `llmb-run` by prefixing them on the submit command. For example:

```bash
EXPERT_PARALLEL_DEGREE=16 llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256
TRAINING_STEPS=5000 LOCAL_BATCH_SIZE=8 llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256
```

## Direct Method

**Important**:

- Use the direct `sbatch launch.sh` path below only when you specifically want to bypass `llmb-run`, for example when you need lower-level control or when `llmb-run` does not expose something you need.
- Run the launch script from the installed recipe directory: `cd $LLMB_INSTALL/llmb_repo/deepseek_v3/pretrain/torchtitan/`
- No activated Python environment is required just to submit `launch.sh`; the workload executes inside the configured container.

### Environment variables

**Required:**

- `GPU_TYPE`: Type of GPU hardware

  - `h100` - NVIDIA H100 GPUs
  - `b200` - NVIDIA B200 GPUs
  - `gb200` - NVIDIA GB200 GPUs

- `JOB_TOTAL_GPUS`: Total number of GPUs to use for training

- `LLMB_INSTALL`: Path to the installation directory for all workloads

**Optional:**

- `GPUS_PER_NODE`: Number of GPUs per node (default: 8 for H100/B200, 4 for GB200)
- `DTYPE`: Training precision (default: `bf16`; `fp8` is also supported on B200 and GB200)
- `DATA_PARALLEL_SHARD_DEGREE`: Data parallel sharding degree (default: 64 for H100, `-1` for B200/GB200)
- `EXPERT_PARALLEL_DEGREE`: Expert parallel degree for MoE (default: 64)
- `PIPELINE_PARALLEL_DEGREE`: Pipeline parallel degree (default: 8 for H100, 1 for B200/GB200)
- `PIPELINE_PARALLEL_SCHEDULE`: Pipeline schedule (default: `1F1B` for H100, `Interleaved1F1B` for B200/GB200)
- `ACTIVATION_CHECKPOINT_MODE`: Activation checkpoint mode (default: `selective` for H100, `full` for B200/GB200)
- `DATASET_PATH`: Path to the dataset (default: `$LLMB_INSTALL/datasets/c4`)
- `SEQ_LEN`: Sequence length (default: 4096)
- `TRAINING_STEPS`: Number of training steps (default: 60)
- `LOCAL_BATCH_SIZE`: Local batch size per GPU (default: 16 for H100, 8 for B200/GB200)
- `LOG_RANK`: Rank to log from (default: 448 for H100, 224 for B200/GB200)
- `EP_COMM_BACKEND`: Expert parallel communication backend (default: `deepep` for H100/B200, `hybridep` for GB200)
- `RUN_CONF_IMAGE`: Override container image path
- `RUN_CONF_MOUNTS`: Additional container mounts

To control the batch allocation in the direct method, pass flags directly to `sbatch` or use the corresponding `SBATCH_*` environment variables.

## Running the Launch Script

### Command Template

```bash
GPU_TYPE=<type> JOB_TOTAL_GPUS=<number> sbatch -N <nodes> launch.sh
```

For example, to use a reservation with the direct method:

```bash
GPU_TYPE=<type> JOB_TOTAL_GPUS=<number> sbatch -N <nodes> --reservation=my_reservation launch.sh
```

### Example Commands

Train on H100 GPUs (minimum configuration):

```bash
GPU_TYPE=h100 JOB_TOTAL_GPUS=512 sbatch -N 64 launch.sh
```

Train on B200 GPUs:

```bash
GPU_TYPE=b200 JOB_TOTAL_GPUS=256 sbatch -N 32 launch.sh
```

Train on B200 GPUs with FP8:

```bash
GPU_TYPE=b200 DTYPE=fp8 JOB_TOTAL_GPUS=256 sbatch -N 32 launch.sh
```

Train on GB200 GPUs:

```bash
GPU_TYPE=gb200 JOB_TOTAL_GPUS=256 sbatch -N 64 launch.sh
```

Train on GB200 GPUs with FP8:

```bash
GPU_TYPE=gb200 DTYPE=fp8 JOB_TOTAL_GPUS=256 sbatch -N 64 launch.sh
```

Train with custom training steps:

```bash
GPU_TYPE=h100 JOB_TOTAL_GPUS=512 TRAINING_STEPS=5000 sbatch -N 64 launch.sh
```

Train with custom parallelism settings:

```bash
GPU_TYPE=h100 JOB_TOTAL_GPUS=512 \
  DATA_PARALLEL_SHARD_DEGREE=32 \
  EXPERT_PARALLEL_DEGREE=16 \
  PIPELINE_PARALLEL_DEGREE=4 \
  sbatch -N 64 launch.sh
```

## Configuration Files

The training uses a TOML configuration file located at:

```
$LLMB_INSTALL/llmb_repo/deepseek_v3/pretrain/torchtitan/deepseek_v3_671b.toml
```

This file contains:

- Model architecture specifications (DeepSeek-V3 671B)
- Optimizer settings (AdamW with lr=2.2e-4)
- Learning rate scheduler configuration (warmup_steps=100, decay_ratio=0.8, cosine decay)
- Activation checkpointing settings (full mode enabled)
- Compilation options (model and loss compilation enabled)
- Float8 quantization options (disabled by default)
- Profiling settings (disabled by default)
- Metrics logging settings (log_freq=10)

Environment variables set when invoking `launch.sh` override parts of the TOML configuration through explicit TorchTitan CLI arguments.

# Output Locations

All job outputs are organized under `$LLMB_WORKLOAD/experiments/`, but the final run-directory name depends on how the job is launched.

When launched with `llmb-run` (`configured_sbatch`), the launcher pre-creates a timestamped experiment directory:

```text
$LLMB_WORKLOAD/experiments/<workload>_<size>_<dtype>_gpus<number>/
└── <unix_timestamp>/
    ├── llmb-config_<SLURM_JOB_ID>.yaml       # Job configuration (created by llmb-run)
    ├── slurm-<SLURM_JOB_ID>.out              # Main Slurm job output
    ├── log-torchtitan_*.out                  # Training stdout (per-rank logs)
    └── outputs/                              # Training outputs and dumps
        ├── profile_trace/                    # Profiling traces (if enabled)
        └── memory_snapshot/                  # Memory snapshots (if enabled)
```

When launched directly with `sbatch launch.sh`, the script creates a job-based directory instead:

```text
$LLMB_WORKLOAD/experiments/<workload>_<size>_<dtype>_gpus<number>/
└── job_<SLURM_JOB_ID>/
    ├── log-torchtitan_*.out                  # Training logs
    └── outputs/                              # Training outputs and dumps
```

**Note:** The `<unix_timestamp>` subdirectory is specific to `llmb-run` managed launches.

**Example:** For a 671B BF16 model run on 512 GPUs, outputs are stored in:

```
$LLMB_WORKLOAD/experiments/pretrain_deepseek-v3-torchtitan_671b_bf16_gpus512/1769818909/
```

where `1769818909` is the Unix timestamp of the job launch time.

**Key files:**

- `llmb-config_*.yaml` - Job configuration including model, scale, and cluster info
- `slurm-*.out` - Slurm job outputs (main job, parsing, uploader)
- `log-torchtitan_*.out` - Training step timing and performance metrics

Additional outputs (if enabled in the TOML config):

- `outputs/profile_trace/` - Profiling traces (if enabled)
- `outputs/memory_snapshot/` - Memory snapshots (if enabled)
- `outputs/tb/` - TensorBoard logs (if enabled)
- `outputs/checkpoint/` - Model checkpoints (if enabled)

# Performance Measurement and Analysis

Performance for DeepSeek-V3 training is measured by TFLOPS per GPU (see `tflops:` entry in the log), which indicates computational throughput efficiency. Additionally, tokens per second (`tps:` entry in the log) is captured. Both metrics are logged for every training step in the main training log file.

## Analyzing Training Performance

To extract performance metrics from the training logs:

```bash
# Navigate to the experiments directory and find your job folder
cd $LLMB_WORKLOAD/experiments
ls -lt  # List experiment configurations

# Navigate to a specific experiment configuration (e.g., 671B BF16 on 512 GPUs)
cd pretrain_deepseek-v3-torchtitan_671b_bf16_gpus512/

# List runs (timestamped if launched with llmb-run, job_<id> if launched directly)
ls -lt

# Navigate to a specific run directory
cd <run_dir>/

# View the training log
tail -f log-torchtitan_*.out

# Extract timing information (after warmup)
grep "step:" log-torchtitan_*.out | tail -20
```

Look for log entries containing:

- `step:` - Training step number
- `loss:` - Training loss value
- `tps:` - Tokens per second
- `tflops:` - TFLOPS per GPU
- `mfu:` - Model FLOPs Utilization

### Extracting TPS (tokens/sec)

The training log includes per-step tokens/sec in lines like:

```
tps: 299
```

To print the most recent TPS values:

```bash
# From within a specific run directory
grep -h "tps:" log-torchtitan_*.out | tail -20
```

## Calculating Throughput

To calculate throughput in tokens per second:

```
throughput (tokens/sec) = (sequence length) × (global batch size) / (training step time in seconds)
```

## Model FLOPs Utilization (MFU)

Model FLOPs Utilization indicates how efficiently the model is using the available compute:

```
MFU = (achieved TFLOPS per GPU) / (peak theoretical TFLOPS)
```

**Peak theoretical throughput across GPUs and Data Types (in TFLOPS)**

For peak theoretical throughput values used in MFU calculations, see the [Peak Theoretical Throughput](../../../README.md#peak-theoretical-throughput) section in the main README.

# Troubleshooting

## Common Issues

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce `LOCAL_BATCH_SIZE`
2. Increase parallelism degrees (especially pipeline parallel)
3. On H100, increase activation checkpointing, for example `ACTIVATION_CHECKPOINT_MODE=full`

### NCCL Timeout

If you see NCCL timeout errors:

1. Increase `[comm] init_timeout_seconds` in the TOML config (default: 1200 seconds)
2. Check network connectivity between nodes
3. Verify Slurm allocation includes all requested GPUs

### Container Mount Issues

If the container cannot access files:

1. Verify `LLMB_INSTALL` and `LLMB_WORKLOAD` paths are accessible
2. Add additional mounts via `RUN_CONF_MOUNTS` if needed
3. Check file permissions

### GPU Type Not Supported

The error message `TorchTitan recipe supports only h100, b200, and gb200 GPU types; got ...` means:

- You're trying to use a GPU type not supported by this recipe
- Currently supported: h100, b200, gb200

# Advanced Configuration

## Custom Dataset

To use a different dataset:

1. Place your dataset in `$LLMB_INSTALL/datasets/<dataset_name>`
2. Set `DATASET_PATH=$LLMB_INSTALL/datasets/<dataset_name>` when launching
3. Update the TOML config if needed to specify the dataset format

## Profiling

There are two ways to enable PyTorch/TorchTitan profiling:

### Option 1: Using the launch flag or environment variable

Set `ENABLE_PROFILE=true` when launching (or use the `-p` flag). The launch script will pass the TorchTitan override `--profiling.enable_profiling` and write traces to the `outputs/` directory within your run folder.
With the launch-script flag path, traces are written under `outputs/torchtitan_<model>_<gpus>gpus_<jobid>_profile_trace/`.

Example:

```bash
# Using the -p flag
llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256 -p

# Or using the environment variable
ENABLE_PROFILE=true llmb-run submit -w pretrain_deepseek-v3-torchtitan --dtype bf16 --scale 256
```

To view the generated traces, inspect the `outputs/` directory within your run folder.

### Option 2: Modifying the TOML configuration file

Edit the TOML configuration file (`deepseek_v3_671b.toml`, see [Configuration Files](#configuration-files)) and set:

```toml
[profiling]
enable_profiling = true
save_traces_folder = "profile_trace"  # customize as needed
profile_freq = 10
```

With this method, traces will be saved to the `outputs/<save_traces_folder>/` directory within your run folder.

### Memory Snapshots

A memory snapshot is a point-in-time capture of the GPU allocator state saved to disk, useful for debugging memory issues such as OOM errors, fragmentation, and per-rank memory leaks. Unlike a profiler trace (which records op timing), a memory snapshot records what was allocated and held at a given step.

To enable, set in the TOML configuration file:

```toml
[profiling]
enable_memory_snapshot = true
save_memory_snapshot_folder = "memory_snapshot"  # customize as needed
```

When enabled, memory snapshots are written to `outputs/<save_memory_snapshot_folder>/iteration_<step>/rank<rank>_memory_snapshot.pickle` every `profile_freq` steps (and also on OOM).

## TensorBoard and Weights & Biases

To enable TensorBoard logging, add the following to the TOML configuration file (`deepseek_v3_671b.toml`, see [Configuration Files](#configuration-files)):

```toml
[metrics]
enable_tensorboard = true
save_tb_folder = "tb"
```

To view the generated logs, inspect the `outputs/tb/` directory within your run folder.

To enable Weights & Biases logging:

```toml
[metrics]
enable_wandb = true
```

> **Note:** For W&B, ensure you have authenticated via `wandb login` or set the `WANDB_API_KEY` environment variable.

# Additional Resources

- [TorchTitan GitHub Repository](https://github.com/pytorch/torchtitan)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan/tree/main/docs)
- [DeepSeek-V3 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Base)
- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)

# Known Limitations

1. **Dataset**: Only C4 dataset is configured by default. Custom datasets require manual configuration.
2. **Checkpointing**: Model checkpointing is disabled by default for benchmarking purposes.
3. **Monitoring**: TensorBoard and Weights & Biases integrations are disabled by default.
4. **Flex Attention**: Disabled in the current configuration (uses standard causal attention).

For production training runs, you may want to enable checkpointing and monitoring in the TOML configuration file.

# DeepSeek V4 Flash Pre-training

This recipe runs DeepSeek V4 Flash pre-training performance benchmarks with Megatron-Bridge. It uses synthetic data and weak scaling: Megatron-Bridge increases the global batch size proportionally with the GPU count.

## Supported configurations

- GPUs: GB300, GB200, B300, and B200
- Scale: 128, 256, or 512 GPUs
- Precision: FP8 with the MX recipe
- Sequence length: 4096
- Model size: `284b`

The 128-GPU performance recipe is the weak-scaling baseline. Its parallelism remains fixed at larger scales while data parallelism and global batch size increase.

### GB300

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8       | 128  |  4096  |  1  |  1  |  1  | 32  | 128 | N/A |  2  | 2048 |  8  |
| FP8       | 256  |  4096  |  1  |  1  |  1  | 32  | 256 | N/A |  2  | 4096 |  8  |
| FP8       | 512  |  4096  |  1  |  1  |  1  | 32  | 512 | N/A |  2  | 8192 |  8  |

### GB200

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8       | 128  |  4096  |  1  |  1  |  1  | 32  | 128 | N/A |  1  | 2048 | 16  |
| FP8       | 256  |  4096  |  1  |  1  |  1  | 32  | 256 | N/A |  1  | 4096 | 16  |
| FP8       | 512  |  4096  |  1  |  1  |  1  | 32  | 512 | N/A |  1  | 8192 | 16  |

### B300

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8       | 128  |  4096  |  1  |  4  |  1  |  8  | 32  | N/A |  2  | 2048 | 32  |
| FP8       | 256  |  4096  |  1  |  4  |  1  |  8  | 64  | N/A |  2  | 4096 | 32  |
| FP8       | 512  |  4096  |  1  |  4  |  1  |  8  | 128 | N/A |  2  | 8192 | 32  |

### B200

| Precision | GPUs | SeqLen | TP  | PP  | CP  | EP  | DP  | VP  | MBS | GBS  | GA  |
| --------- | :--: | :----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--: | :-: |
| FP8       | 128  |  4096  |  1  |  4  |  1  |  8  | 32  | N/A |  1  | 2048 | 64  |
| FP8       | 256  |  4096  |  1  |  4  |  1  |  8  | 64  | N/A |  1  | 4096 | 64  |
| FP8       | 512  |  4096  |  1  |  4  |  1  |  8  | 128 | N/A |  1  | 8192 | 64  |

## Prerequisites

- A Slurm cluster with GB300/GB200 nodes (four GPUs per node) or B300/B200 nodes (eight GPUs per node)
- Python 3.12 or conda
- The environment prepared with the installer described in the [main README](../../../README.md)

No dataset preparation is required because the benchmark uses synthetic data. The installer downloads the model configuration from Hugging Face.

## Run with `llmb-run`

From `$LLMB_INSTALL`, submit a benchmark using the `284b` model size:

```bash
# 128 GB300 GPUs
llmb-run submit -w pretrain_deepseek-v4 -s 284b --dtype fp8 --scale 128

# 256 GPUs
llmb-run submit -w pretrain_deepseek-v4 -s 284b --dtype fp8 --scale 256

# 512 GPUs
llmb-run submit -w pretrain_deepseek-v4 -s 284b --dtype fp8 --scale 512
```

Set `gpu_type` in the selected `llmb-run` cluster configuration to `gb300`, `gb200`, `b300`, or `b200`. See the [`llmb-run` documentation](../../../cli/llmb-run/README.md) for cluster configuration and submission options.

## Run directly

Activate the environment created by the installer, then run the launch script from the installed recipe directory:

```bash
cd "$LLMB_INSTALL/llmb_repo/deepseek_v4/pretrain/flash"

JOB_TOTAL_GPUS=128 \
GPU_TYPE=gb300 \
SBATCH_ACCOUNT=<account> \
SBATCH_PARTITION=<partition> \
./launch.sh
```

Required environment variables:

- `LLMB_INSTALL`: top-level LLMB installation directory
- `JOB_TOTAL_GPUS`: `128`, `256`, or `512`
- `GPU_TYPE`: `gb300`, `gb200`, `b300`, or `b200`
- `SBATCH_ACCOUNT`: Slurm account (Slurm only)
- `SBATCH_PARTITION`: Slurm partition (Slurm only)

On Kubernetes (`CLUSTER_TYPE=kubernetes`, set by `llmb-run` when the cluster config has `execution_target: kubernetes`) the
Slurm variables are not used. The pinned Megatron-LM checkout cannot be bind-mounted into the pod, so it is copied to the
workdir PVC and put first on `PYTHONPATH`; `kubernetes.workdir_pvc` must therefore be set in `cluster_config.yaml`.

Optional environment variables:

- `DTYPE`: fixed to `fp8` (default: `fp8`)
- `FP8_RECIPE`: fixed to `mx` (default: `mx`)
- `MAX_STEPS`: number of training steps (default: `50`)
- `TIME_LIMIT`: Slurm time limit (default: `01:10:00`)
- `ADDITIONAL_SLURM_PARAMS`: extra parameters passed to the Slurm launcher
- `LLMB_MBRIDGE_EXTRA_ARGS`: additional Megatron-Bridge setup arguments

## Results

Use `llmb-run jobs` to inspect submitted jobs and parsed performance metrics:

```bash
llmb-run jobs
llmb-run jobs show <job_id>
llmb-run jobs log <job_id>
```

Results and logs are stored under `$LLMB_WORKLOAD/experiments/`. Performance is reported as wall-clock seconds per iteration and sustained TFLOPS per GPU.

## Nsight Systems profiling

Enable profiling with `-p` when using `llmb-run`:

```bash
llmb-run submit -w pretrain_deepseek-v4 -s 284b --dtype fp8 --scale 128 -p
```

For direct launches, set `ENABLE_PROFILE=true`:

```bash
ENABLE_PROFILE=true \
JOB_TOTAL_GPUS=128 \
GPU_TYPE=gb300 \
SBATCH_ACCOUNT=<account> \
SBATCH_PARTITION=<partition> \
./launch.sh
```

By default, steps 45 through 50 are profiled on every rank. Customize profiling with:

- `PROFILE_START_STEP` (default: `45`)
- `PROFILE_STOP_STEP` (default: `50`)
- `PROFILE_RANKS` (default: all ranks)
- `ENABLE_GPU_METRICS=true` to collect GPU metrics

Profiling reports are written alongside the experiment output.

## PyTorch profiling

PyTorch Profiling is intended for rare, advanced debugging scenarios such as NCCL correlation analysis. To enable it, set `ENABLE_PYTORCH_PROFILE=true` when submitting your job.

> **Note:** This option is mutually exclusive with Nsight profiling (`ENABLE_PROFILE`). Both cannot be enabled at the same time.

**Example command:**

```bash
ENABLE_PYTORCH_PROFILE=true llmb-run submit -w pretrain_deepseek-v4 -s 284b --dtype fp8 --scale 128
```

Trace files are saved to `torch_profile/rank-N.json.gz` in the job output directory, where `N` is the rank number. For details on the PyTorch Profiler and how to view resulting traces, see the [PyTorch Profiler documentation](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

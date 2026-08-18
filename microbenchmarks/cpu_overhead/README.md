# Overview

This recipe provides instructions for running microbenchmarks that measure CPU overhead on single node.

We consider two CPU overhead scenarios:

- **Pytorch kernel launch latency**: In this test, we measure how fast the CPU can push tiny kernels into the GPU. The higher the average execution time per benchmark, the worse the CPU overhead.
- **Tokenization throughput**: This test measures CPU-only performance through the tokenization step used in inference and RL. The higher the time taken to generate N tokens, the worse the CPU overhead.

The script uses TRT-LLM release containers to benchmark tokenization throughput for the [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b) model on GB300/GB200/B300/B200/H100 platforms.

# Prerequisites

A HuggingFace account is required and you will need to [create a HuggingFace access token](https://huggingface.co/settings/tokens). You will need this token during the LLMB Installation when preparing your environment.

# Prepare Environment

The recommended way to prepare your environment is to use the **installer** referenced in the [main README](../../README.md):

The following directory layout and key variables are used in the recipe:

- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts (images, datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: Workload-specific directory, e.g. `${LLMB_INSTALL}/workloads/microbenchmark_cpu_overhead`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

## Slurm

We reference a number of Slurm commands and parameters in this document. A brief summary is included below. It's important to note these are a guide and might not be applicable to all environments. Please consult with your system administrator for the parameters that are specific to your system.

**Common parameters:**

- `SBATCH_PARTITION` or `-p` - Partition (or queue) to use.
- `SBATCH_ACCOUNT` or `-A` - Slurm account to associate with your job, different from your user. Meant for accounting purposes.
- `SBATCH_GPUS_PER_NODE` or `--gres=gpu:<num gpus>` - If your cluster is configured with GRES this should be set to all GPUs in a node. Ignore if not configured.
  - Encountering errors such as 'GPUs not found' or 'Cannot submit to this partition without GPU resources' means this setting is required.

These parameters can be set either by exporting the environment variable or using the corresponding `sbatch` flag.

## Using llmb-run (Recommended)

The easiest way to run benchmarks is using the llmb-run launcher tool. This method handles configuration automatically and provides a streamlined interface.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark with llmb-run per use case (** Recommended **)

# Run both CPU overhead tests
llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1

# Run the pytorch kernel launch test
USE_CASES="kernel_launch" llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1

# Run the tokenization throughput test
USE_CASES="tokenization" llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1
```

### Additional SLURM Parameters

For `llmb-run submit`, use the built-in Slurm flags.

Use a Slurm reservation:

```bash
llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1 --reservation my_reservation
```

Run on a specific node:

```bash
llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1 --nodelist node001
```

Exclude specific nodes:

```bash
llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1 --exclude node002,node003
```

Combine multiple parameters:

```bash
llmb-run submit -w microbenchmark_cpu_overhead --dtype mxfp4 --scale 1 --nodelist node001 --reservation my_reservation
```

For more details on llmb-run usage, see the [llmb-run documentation](../../cli/llmb-run/README.md).

### Results/Log files

Each submission gets its own timestamped experiment directory, managed by the
`configured_sbatch` launcher:

```text
$LLMB_INSTALL/workloads/microbenchmark_cpu_overhead/experiments/
└── microbenchmark_cpu_overhead_120b_mxfp4_gpus<scale>/
    └── <timestamp>/
        ├── llmb-config_<job_id>.yaml
        ├── slurm-<job_id>.out
        ├── <use_case>_overhead_<node>_<job_id>.err
        ├── <use_case>_overhead_<node>_<job_id>.out
        ├── gemm_benchmark_results.csv  # kernel_launch only
        └── dataset_1000_1000_<num_prompts>.txt  # tokenization only
```

`llmb-run submit` reports the exact experiment directory as `workdir`. Logs and
other generated artifacts from different submissions are therefore not mixed.

The `*.out` file provides key performance metrics:

```text
Kernel launch test:

Benchmarking GEMM size: 4x4 ...
Average execution time for size:4 is 11.299694776535034 us
Benchmarking GEMM size: 8x8 ...
Average execution time for size:8 is 12.973302841186523 us
Benchmarking GEMM size: 16x16 ...
Average execution time for size:16 is 12.918899536132812 us
Benchmarking GEMM size: 32x32 ...
Average execution time for size:32 is 17.679811716079712 us
Benchmarking GEMM size: 64x64 ...
Average execution time for size:64 is 17.74930453300476 us
Benchmarking GEMM size: 128x128 ...
Average execution time for size:128 is 12.853708982467651 us
Benchmarking GEMM size: 256x256 ...
Average execution time for size:256 is 12.944073677062988 us
Benchmarking GEMM size: 512x512 ...
Average execution time for size:512 is 17.642942667007446 us
Results saved to gemm_benchmark_results.csv

Tokenization test:

Tokenization time: 39s
```

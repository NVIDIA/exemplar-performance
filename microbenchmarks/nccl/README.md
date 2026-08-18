# Overview

This recipe provides a method and script to produce performance results for the NCCL benchmarks. The SLURM sbatch script launches a collection of benchmarking jobs to evaluate an Infiniband network.

## System Requirements

- SLURM workload manager
- NVIDIA GPUs: GB300, GB200, B300, B200, or H100
- InfiniBand (IB) networking with Quantum2 switches and CX7 NICs
- CUDA toolkit

# Performance Measurement and Analysis

Performance for NCCL is measured by bandwidth and latency. **Open the `.txt` files** directly in the latest `LOG_*` directory to view your results. Each `.txt` file contains the complete benchmark output for one NCCL test (sizes, algorithms, bandwidth, latency).

**Note:** The `LOG_*` directory also contains folders with similar names — ignore these. They contain debug logs and per-rank output that aren't needed for performance analysis.

Which files to open:

```text
# AllReduce results (open this .txt file)
$LLMB_WORKLOAD/experiments/microbenchmark_nccl/
  microbenchmark_nccl_<variant>_<user>_<SLURM_JOBID>/
    LOG_<DATE>-<JOBID>_sweep_N<NUM_NODES>/
      1_1_LOG_all_reduce_env1_args1_job<JOBID>_iter1.txt

# AllGather results (open this .txt file)
$LLMB_WORKLOAD/experiments/microbenchmark_nccl/
  microbenchmark_nccl_<variant>_<user>_<SLURM_JOBID>/
    LOG_<DATE>-<JOBID>_sweep_N<NUM_NODES>/
      1_2_LOG_all_gather_env1_args1_job<JOBID>_iter1.txt
```

<details>
<summary>View sample output (AllReduce excerpt)</summary>

```text
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0 side_work: 0 work_sms: 0 direction: H2D
#
# Using devices
#  Rank  0 Group  0 Pid 1006102 on pool0-01806 device  0 [0000:19:00] NVIDIA H100 80GB HBM3
#  Rank  1 Group  0 Pid 1005853 on pool0-01806 device  1 [0000:2d:00] NVIDIA H100 80GB HBM3
#  Rank  2 Group  0 Pid 1006013 on pool0-01806 device  2 [0000:3f:00] NVIDIA H100 80GB HBM3
#  Rank  3 Group  0 Pid 1006017 on pool0-01806 device  3 [0000:66:00] NVIDIA H100 80GB HBM3
#  Rank  4 Group  0 Pid 1005902 on pool0-01806 device  4 [0000:9b:00] NVIDIA H100 80GB HBM3
#  Rank  5 Group  0 Pid 1006015 on pool0-01806 device  5 [0000:ae:00] NVIDIA H100 80GB HBM3
#  Rank  6 Group  0 Pid 1005825 on pool0-01806 device  6 [0000:bf:00] NVIDIA H100 80GB HBM3
#  Rank  7 Group  0 Pid 1005906 on pool0-01806 device  7 [0000:e4:00] NVIDIA H100 80GB HBM3
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    13.67    0.00    0.00      0    13.03    0.00    0.00      0
          16             4     float     sum      -1    13.00    0.00    0.00      0    13.07    0.00    0.00      0
          32             8     float     sum      -1    68.14    0.00    0.00      0    13.52    0.00    0.00      0
          64            16     float     sum      -1    14.92    0.00    0.01      0    14.72    0.00    0.01      0
         128            32     float     sum      -1    15.87    0.01    0.01      0    15.79    0.01    0.01      0
         256            64     float     sum      -1    16.10    0.02    0.03      0    15.77    0.02    0.03      0
         512           128     float     sum      -1    16.70    0.03    0.05      0    16.01    0.03    0.06      0
        1024           256     float     sum      -1    16.94    0.06    0.11      0    16.17    0.06    0.11      0
        2048           512     float     sum      -1    17.14    0.12    0.21      0    16.43    0.12    0.22      0
        4096          1024     float     sum      -1    23.59    0.17    0.30      0    16.46    0.25    0.44      0
        8192          2048     float     sum      -1    17.40    0.47    0.82      0    16.83    0.49    0.85      0
       16384          4096     float     sum      -1    17.93    0.91    1.60      0    17.24    0.95    1.66      0
       32768          8192     float     sum      -1    19.33    1.70    2.97      0    18.68    1.75    3.07      0
       65536         16384     float     sum      -1    19.54    3.35    5.87      0    18.85    3.48    6.08      0
      131072         32768     float     sum      -1    19.89    6.59   11.53      0    19.19    6.83   11.95      0
      262144         65536     float     sum      -1    20.43   12.83   22.45      0    19.50   13.44   23.52      0
      524288        131072     float     sum      -1    20.66   25.38   44.41      0    19.70   26.61   46.57      0
     1048576        262144     float     sum      -1    27.37   38.31   67.04      0    26.93   38.94   68.14      0
     2097152        524288     float     sum      -1    42.21   49.68   86.94      0    42.09   49.83   87.20      0
     4194304       1048576     float     sum      -1    54.19   77.39  135.44      0    53.82   77.93  136.37      0
     8388608       2097152     float     sum      -1    83.04  101.01  176.77      0    82.03  102.26  178.96      0
    16777216       4194304     float     sum      -1    123.8  135.55  237.21      0    123.0  136.35  238.61      0
    33554432       8388608     float     sum      -1    199.3  168.39  294.69      0    198.9  168.73  295.27      0
    67108864      16777216     float     sum      -1    324.7  206.67  361.67      0    324.7  206.67  361.68      0
   134217728      33554432     float     sum      -1    586.4  228.88  400.53      0    586.0  229.05  400.84      0
   268435456      67108864     float     sum      -1   1112.5  241.29  422.25      0   1113.4  241.09  421.91      0
   536870912     134217728     float     sum      -1   2151.0  249.59  436.79      0   2149.3  249.79  437.13      0
  1073741824     268435456     float     sum      -1   4029.4  266.48  466.33      0   4016.4  267.34  467.85      0
  2147483648     536870912     float     sum      -1   7939.1  270.50  473.37      0   7938.0  270.53  473.43      0
  4294967296    1073741824     float     sum      -1    15749  272.71  477.24      0    15754  272.63  477.10      0
  8589934592    2147483648     float     sum      -1    31307  274.38  480.16      0    31328  274.19  479.84      0
 17179869184    4294967296     float     sum      -1    62648  274.23  479.90      0    62636  274.28  479.99      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 159.151
#
```

</details>

Expected results folder structure:

```text
$LLMB_WORKLOAD/experiments/
└── microbenchmark_nccl/
    └── microbenchmark_nccl_<variant>_<user>_<SLURM_JOBID>/
        └── LOG_<DATE>-<JOBID>_sweep_N<NUM_NODES>/
            ├── 1_1_LOG_all_reduce_env1_args1_job<JOBID>_iter1.txt        ← Open these .txt files
            ├── 1_2_LOG_all_gather_env1_args1_job<JOBID>_iter1.txt        ← 
            ├── 1_3_LOG_reduce_scatter_env1_args1_job<JOBID>_iter1.txt    ← 
            ├── 1_4_LOG_alltoall_env1_args1_job<JOBID>_iter1.txt          ← 
            ├── 1_5_LOG_alltoall_env2_args1_job<JOBID>_iter1.txt          ← 
            ├── 1_6_LOG_sendrecv_env1_args1_job<JOBID>_iter1.txt          ← 
            ├── 1_7_LOG_sendrecv_env2_args1_job<JOBID>_iter1.txt          ← 
            ├── 1_1_LOG_all_reduce_env1_args1_job<JOBID>/                 (ignore - debug logs)
            ├── 1_1_LOG_all_reduce_env1_args1_job<JOBID>_iter1/           (ignore - per-rank logs)
            ├── *.yml files                                                (ignore - metadata)
            └── ... (similar folders/files for other tests)
```

# Prerequisites

Run `install.sh` first so required tools (including `uv`) are available before NCCL setup generates `launch.sh`.

## Set environment variable (optional)

Only needed if not using `llmb-run`, or if you want to run `llmb-run` from outside `$LLMB_INSTALL`.

```shell
# Set your installation directory
export LLMB_INSTALL=<path to your installation directory> (e.g. /lustre/llmb/)
```

# Run NCCL Benchmarks

## Quick Start (llmb-run)

`llmb-run` uses your SLURM settings from install to submit jobs.

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Run a benchmark (scale = total GPUs)
llmb-run submit -w microbenchmark_nccl --scale <NUM_GPUS>

# Examples
llmb-run submit -w microbenchmark_nccl --scale 4
llmb-run submit -w microbenchmark_nccl --scale 16
```

# Nemotron 3 Ultra NVFP4 SGLang Dynamo on GB300

This workload keeps recipes and provisioned artifacts in LLMB and uses a pinned
SRT-Slurm release only for multi-node orchestration.

## Installed layout

```text
$LLMB_INSTALL/
├── images/
│   └── sglang-runtime-1.4.0-inkling-dev.1.sqsh
└── workloads/inference_nemotron3_ultra_sglang_dynamo/
    ├── nemotron3-ultra-nvfp4/
    ├── srt-slurm/
    │   └── srtslurm.yaml
    └── experiments/
```

The installer downloads `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` at
revision `183968f87ae4cedce3039313cac1fd43d112c578`, imports
`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0-inkling-dev.1`, and checks out
SRT-Slurm release `v1.0.38` (commit `c180328b98c3793ca84a1e24a030f90545eb7d5d`).

## Submit

The default is the 10K/16K, 1-prefill/1-decode recipe (2 nodes, 8 GB300 GPUs):

```bash
cd "$LLMB_INSTALL"
./llmb-run submit \
  -w inference_nemotron3_ultra_sglang_dynamo \
  -s 550b \
  --dtype nvfp4 \
  --scale 8
```

To select another recipe, set its path relative to the workload directory in
LLMB's installed repository copy and use its GPU count as `--scale`:

```bash
export RUN_CONF_NEMOTRON3_ULTRA_SGLANG_RECIPE=recipes/gb300/8k_64k/nemotron3-ultra-gb300-nvfp4-8k64k-1p3d.yaml
```

Use `./llmb-run jobs` and `./llmb-run jobs log <job-id>` to monitor the
orchestration job. Benchmark results remain in SRT-Slurm's native
`logs/benchmark.out` and `logs/benchmark-rollup.csv` artifacts.

## Recipe coverage

Each recipe sweeps concurrency 1, 4, 8, 16, and 32. Workers are TP4/EP4, one
GB300 node per worker.

| ISL/OSL | Recipe under `recipes/gb300/`                          | Prefill × Decode | `--scale` |
| ------- | ------------------------------------------------------ | ---------------- | --------- |
| 50K/2K  | `50k_2k/nemotron3-ultra-gb300-nvfp4-50k2k-1p1d.yaml`   | 1 × 1            | 8         |
| 50K/2K  | `50k_2k/nemotron3-ultra-gb300-nvfp4-50k2k-8p2d.yaml`   | 8 × 2            | 40        |
| 50K/2K  | `50k_2k/nemotron3-ultra-gb300-nvfp4-50k2k-10p2d.yaml`  | 10 × 2           | 48        |
| 10K/16K | `10k_16k/nemotron3-ultra-gb300-nvfp4-10k16k-1p1d.yaml` | 1 × 1            | 8         |
| 10K/16K | `10k_16k/nemotron3-ultra-gb300-nvfp4-10k16k-1p3d.yaml` | 1 × 3            | 16        |
| 10K/16K | `10k_16k/nemotron3-ultra-gb300-nvfp4-10k16k-4p4d.yaml` | 4 × 4            | 32        |
| 8K/64K  | `8k_64k/nemotron3-ultra-gb300-nvfp4-8k64k-1p1d.yaml`   | 1 × 1            | 8         |
| 8K/64K  | `8k_64k/nemotron3-ultra-gb300-nvfp4-8k64k-1p3d.yaml`   | 1 × 3            | 16        |
| 8K/64K  | `8k_64k/nemotron3-ultra-gb300-nvfp4-8k64k-4p4d.yaml`   | 4 × 4            | 32        |

All recipe files use SRT schema v2 and the `override_validated` selector.

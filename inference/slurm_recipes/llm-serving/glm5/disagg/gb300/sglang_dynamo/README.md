# GLM-5 FP8 SGLang Dynamo on GB300

This workload keeps validated recipes and provisioned artifacts in LLMB and
uses a pinned SRT-Slurm release only for single- and multi-node orchestration.

## Installed layout

```text
$LLMB_INSTALL/
├── images/
│   ├── nginx-1.27.4.sqsh
│   └── sglang-runtime-1.3.0.sqsh
└── workloads/inference_glm5_sglang_dynamo/
    ├── glm5-fp8/
    ├── srt-slurm/
    │   └── srtslurm.yaml
    └── experiments/
```

The installer downloads `zai-org/GLM-5-FP8` at revision
`4f96cc5eec29dcee5d6ded54f7ffe889438f9516`, imports
`nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.3.0`, and checks out SRT-Slurm release
`v1.0.38` (commit `c180328b98c3793ca84a1e24a030f90545eb7d5d`).
It also imports the pinned NGINX frontend image during installation so benchmark
startup does not depend on Docker Hub access from the allocated GPU nodes.

## Submit

The default is the validated 8K/1K, concurrency-150, 1-prefill/9-decode
configuration (10 nodes, 40 GB300 GPUs):

```bash
cd "$LLMB_INSTALL"
./llmb-run submit \
  -w inference_glm5_sglang_dynamo \
  -s 744b \
  --dtype fp8 \
  --scale 40
```

`llmb-run` invokes `srtctl apply` directly on the login node and records the
actual SRT orchestration job ID. It does not submit a one-GPU dispatcher job.
Use `--scale 40` for a 10-node recipe and `--scale 72` for an 18-node recipe;
LLMB verifies that the selected YAML requests the same physical GPU count.

To select another LLMB-owned recipe, set its path relative to the workload
directory in LLMB's installed repository copy:

```bash
export RUN_CONF_GLM5_SGLANG_RECIPE=recipes/gb300/16k_512/lowlatency/glm5-gb300-fp8-stp-lowlatency-1p9d-bs15-conc150.yaml
```

Use `./llmb-run jobs` and `./llmb-run jobs log <job-id>` to monitor the real
orchestration job. Benchmark results remain in SRT-Slurm's native
`logs/benchmark.out` and `logs/benchmark-rollup.csv` artifacts.

## Recipe coverage

Fourteen LLMB-owned variants are available under `recipes/` in LLMB's installed
repository copy:

| Relative path                                                                             | ISL/OSL   | Concurrency       | Allocated GPUs / `--scale` |
| ----------------------------------------------------------------------------------------- | --------- | ----------------- | -------------------------- |
| `gb300/1k_1k/lowlatency/glm5-gb300-fp8-stp-lowlatency-1p17d-bs32-conc512x256x128x64.yaml` | 1K / 1K   | 512, 256, 128, 64 | 72                         |
| `gb300/1k_1k/hightpt/glm5-gb300-fp8-stp-hightpt-4p14d-bs5600-conc5700.yaml`               | 1K / 1K   | 5,700             | 72                         |
| `gb300/8k_1k/lowlatency/glm5-gb300-fp8-stp-lowlatency-1p9d-bs15-conc150.yaml`             | 8K / 1K   | 150               | 40                         |
| `gb300/8k_1k/lowlatency/glm5-gb300-fp8-stp-lowlatency-1p17d-bs8-conc128x64x32.yaml`       | 8K / 1K   | 128, 64, 32       | 72                         |
| `gb300/8k_1k/hightpt/glm5-gb300-fp8-stp-hightpt-8p10d-bs880-conc900.yaml`                 | 8K / 1K   | 900               | 72                         |
| `gb300/8k_1k/hightpt/glm5-gb300-fp8-stp-hightpt-10p8d-bs1280-conc1300.yaml`               | 8K / 1K   | 1,300             | 72                         |
| `gb300/8k_1k/hightpt/glm5-gb300-fp8-stp-hightpt-12p6d-bs1680-conc1700.yaml`               | 8K / 1K   | 1,700             | 72                         |
| `gb300/8k_1k/hightpt/glm5-gb300-fp8-stp-hightpt-14p4d-bs2800-conc2800.yaml`               | 8K / 1K   | 2,800             | 72                         |
| `gb300/16k_512/lowlatency/glm5-gb300-fp8-stp-lowlatency-1p9d-bs15-conc150.yaml`           | 16K / 512 | 150               | 40                         |
| `gb300/16k_512/lowlatency/glm5-gb300-fp8-stp-lowlatency-1p17d-bs8-conc128x64x32.yaml`     | 16K / 512 | 128, 64, 32       | 72                         |
| `gb300/16k_512/hightpt/glm5-gb300-fp8-stp-hightpt-8p10d-bs880-conc900.yaml`               | 16K / 512 | 900               | 72                         |
| `gb300/16k_512/hightpt/glm5-gb300-fp8-stp-hightpt-10p8d-bs1280-conc1300.yaml`             | 16K / 512 | 1,300             | 72                         |
| `gb300/16k_512/hightpt/glm5-gb300-fp8-stp-hightpt-12p6d-bs1680-conc1700.yaml`             | 16K / 512 | 1,700             | 72                         |
| `gb300/16k_512/hightpt/glm5-gb300-fp8-stp-hightpt-14p4d-bs2800-conc2800.yaml`             | 16K / 512 | 2,800             | 72                         |

All recipe files use SRT-Slurm's `base`/override form, and the launcher selects
the `override_validated` variant.

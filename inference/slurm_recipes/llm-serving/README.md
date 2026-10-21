# LLMB Slurm inference recipes

These workloads keep the reproducibility contract in LLMB and use a pinned
SRT-Slurm release only for single- or multi-node orchestration.

| Workload                                  | GPU   | Serving engine      | LLMB scales       |
| ----------------------------------------- | ----- | ------------------- | ----------------- |
| `inference_glm5_sglang_dynamo`            | GB300 | SGLang Dynamo 1.3.0 | 40, 72            |
| `inference_glm5`                          | GB300 | TRT-LLM Dynamo      | 32, 36, 40        |
| `inference_kimi2.6`                       | GB200 | TRT-LLM Dynamo      | 24, 28, 40, 48    |
| `inference_dsv4f_sglang_gb200`            | GB200 | SGLang Dynamo 1.5.1 | 8                 |
| `inference_dsv4f_sglang_gb300`            | GB300 | SGLang Dynamo 1.5.1 | 8                 |
| `inference_nemotron3_ultra_sglang_dynamo` | GB300 | SGLang Dynamo 1.4.0 | 8, 16, 32, 40, 48 |

## Ownership and installation

LLMB owns and installs:

- pinned model revisions under `$LLMB_INSTALL/workloads/<workload>/`;
- pinned serving images under `$LLMB_INSTALL/images/`;
- validated recipe YAMLs in LLMB's installed repository copy;
- workload metadata and native SRT-Slurm benchmark artifacts.

Each workload pins SRT-Slurm commit `c180328b98c3793ca84a1e24a030f90545eb7d5d`
(the commit released as `v1.0.38`). The checkout is installed at
`$LLMB_INSTALL/workloads/<workload>/srt-slurm`, and `llmb-run` calls
`srtctl apply -f <LLMB-owned-recipe>` directly. There is no intermediate
one-GPU dispatcher job.

Use the standard LLMB installer and select the workload matching the cluster
GPU type. The installation method may use a Slurm CPU/data-mover partition for
model and image provisioning when that partition has network access and
Enroot. The benchmark itself always requests the physical GPU count encoded in
the selected recipe.

## Submit

```bash
cd "$LLMB_INSTALL"

./llmb-run submit \
  -w inference_glm5_sglang_dynamo -s 744b --dtype fp8 --scale 40

./llmb-run submit \
  -w inference_glm5 -s 744b --dtype nvfp4 --scale 32

./llmb-run submit \
  -w inference_kimi2.6 -s 1t --dtype nvfp4 --scale 24

./llmb-run submit \
  -w inference_dsv4f_sglang_gb200 -s 304b --dtype mxfp4 --scale 8

./llmb-run submit \
  -w inference_dsv4f_sglang_gb300 -s 304b --dtype mxfp4 --scale 8

./llmb-run submit \
  -w inference_nemotron3_ultra_sglang_dynamo -s 550b --dtype nvfp4 --scale 8
```

To select another recipe, export the workload's recipe variable before
submitting and use the matching physical GPU scale:

| Workload                 | Recipe variable                          |
| ------------------------ | ---------------------------------------- |
| GLM-5 SGLang             | `RUN_CONF_GLM5_SGLANG_RECIPE`            |
| GLM-5 TRT-LLM            | `RUN_CONF_GLM5_RECIPE`                   |
| Kimi-K2.6 TRT-LLM        | `RUN_CONF_KIMI26_RECIPE`                 |
| DeepSeek-V4-Flash SGLang | `RUN_CONF_DSV4F_RECIPE`                  |
| Nemotron 3 Ultra SGLang  | `RUN_CONF_NEMOTRON3_ULTRA_SGLANG_RECIPE` |

The launcher verifies that `--scale` equals the GPU count in the selected
recipe and rejects mismatches before submission.

## Results

SRT-Slurm writes runtime artifacts into the LLMB experiment directory:

```text
$LLMB_INSTALL/workloads/<workload>/experiments/
└── <configuration>/<timestamp>/<jobid>/
    ├── llmb-config_<jobid>.yaml
    ├── config_validated.yaml
    ├── <jobid>.json
    ├── logs/
    │   ├── benchmark.out
    │   ├── benchmark-rollup.csv
    │   └── ... serving and orchestration logs
```

Use `./llmb-run jobs` to refresh job state and `./llmb-run jobs log <job-id>`
to read the orchestration log. Native benchmark results remain under
`<experiment>/<jobid>/logs/`, including `benchmark.out` and
`benchmark-rollup.csv`. Exporting results to a shared inference database is
outside this launcher's scope.

## Scope and validation

GLM-5 SGLang, GLM-5 TRT-LLM, and Nemotron 3 Ultra SGLang recipes target GB300.
Kimi-K2.6 targets GB200; its schema, pins, and launcher integration can be tested
on another cluster, but an end-to-end performance run must use GB200 hardware.
The DSV4 tests target GB200 and GB300. See each workload README for its recipe
matrix and topology details.

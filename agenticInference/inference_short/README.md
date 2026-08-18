# Inference Recipes (Short Context)

LLMB inference recipes for GLM-5 and Kimi-K2.6 on NVIDIA Grace systems
(gb200, gb300). Driven through the standard install flow — `llmb-install` to
provision, `llmb-run submit` to launch a benchmark.

## Available recipes

| Workload key        | GPU SKU | Model                        | Topology              | Path                            |
| ------------------- | ------- | ---------------------------- | --------------------- | ------------------------------- |
| `inference_glm5`    | gb300   | GLM-5 NVFP4 (744B / 40B-A)   | TRT-LLM Dynamo disagg | `glm5/disagg/trtllm_dynamo/`    |
| `inference_kimi2.6` | gb200   | Kimi-K2.6 NVFP4 (1T / 32B-A) | TRT-LLM Dynamo disagg | `kimi2.6/disagg/trtllm_dynamo/` |

Both recipes are validated end-to-end on lyris (gb200/gb300 Grace cluster);
Kimi-K2.6 is additionally validated on polyphe (gb200). See the per-recipe
`README.md` inside each folder for recipe-specific details.

## Prerequisites

1. **SLURM cluster access** with one of the supported Grace SKUs (gb200, gb300).

2. **HuggingFace token** with read access to the model repositories:

   - `nvidia/GLM-5-NVFP4` (for `inference_glm5`)
   - `nvidia/Kimi-K2.6-NVFP4` (for `inference_kimi2.6`)

3. **`uv`** — the installer auto-installs it via `install.sh` if missing.

4. **`tmux` or `screen` strongly recommended** for installs. The model fetch step
   takes ~35–40 min and is driven by a foreground polling loop. If your SSH
   session drops, the poll loop dies and the installer marks the workload as
   FAILED (even though the SLURM job underneath keeps running). Running inside
   tmux/screen lets you detach safely and reconnect later.

   ```bash
   tmux new -s llmb-install
   # re-export your env vars inside the new tmux shell (they don't carry over):
   export LLMB_REPO=/path/to/exemplar-performance
   export LLMB_INSTALL=/path/to/your/llmbinstall
   export HF_TOKEN=hf_xxx
   # Ctrl-b d to detach safely; reconnect with: tmux attach -t llmb-install
   ```

   If your shell did die and the install marked FAILED: just re-run
   `llmb-install`. Setup tasks are idempotent — it'll skip what's already done
   and resume from where it left off.

## Prepare environment

Use the **installer** referenced in the [main README](../../README.md) to prepare
the recipe environment.

The following directory layout and key variables are used by the recipes:

- `LLMB_INSTALL`: top-level directory for all benchmarking artifacts (images,
  datasets, venvs, workloads, etc).
- `LLMB_WORKLOAD`: workload-specific directory, e.g.
  `${LLMB_INSTALL}/workloads/inference_glm5`.
- Benchmark outputs and logs are stored under
  `${LLMB_WORKLOAD}/srt-slurm/outputs/<job-id>/` (see
  [Where outputs land](#where-outputs-land)).

When the installer prompts for SLURM account, partition, GPU type, etc., use
values appropriate for your cluster (find them with `sinfo` and
`sacctmgr show assoc where user=$USER`). Pick GPU type `GB300` for
`inference_glm5` and `GB200` for `inference_kimi2.6`.

The model + container fetch step takes 30–40 min; running the installer inside
`tmux` (per the Prerequisites note above) lets the wait survive an SSH drop.

# Run benchmark

## Using llmb-run (Recommended)

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Submit (choose the recipe matching the SKU you installed)
llmb-run submit -w inference_glm5 --scale 1
llmb-run submit -w inference_kimi2.6 --scale 1
```

For more details on `llmb-run` usage, see the
[llmb-run documentation](../../cli/llmb-run/README.md).

## Where outputs land

After a successful `llmb-run submit`, your benchmark outputs sit under
`$LLMB_INSTALL/workloads/inference_<model>/srt-slurm/outputs/<benchmark-jobid>/`.

Concrete example, with `LLMB_INSTALL=/lustre/.../llmbinstall` and the GLM5
recipe:

```text
/lustre/.../llmbinstall/
└── workloads/
    └── inference_glm5/
        └── srt-slurm/
            └── outputs/
                └── 1921888/                         # benchmark SLURM job id
                    ├── config.yaml                  # recipe used for this run
                    └── logs/
                        ├── sweep_1921888.log        # orchestrator log (worker startup, health checks)
                        └── benchmark.out            # sa-bench numbers
```

The same layout under `inference_kimi2.6/` for the Kimi recipe. The
`$LLMB_WORKLOAD` env var, when set, equals
`$LLMB_INSTALL/workloads/inference_<model>` for the workload you submitted.

`sacct -j <jobid> --format=JobID,State,Elapsed,NodeList,Reason -P` works for any
SLURM job produced (dispatcher or benchmark).

## Reading benchmark results

After a successful run, `benchmark.out` lands at:

```text
$LLMB_WORKLOAD/srt-slurm/outputs/<benchmark-jobid>/logs/benchmark.out
```

The file contains **multiple result blocks** — one per concurrency in the
sweep. For each concurrency, two blocks appear back-to-back:

1. A **warmup** block (smaller `Successful requests`) used to prime the KV
   cache and JIT — **ignore this block**.
2. The actual **measured** block (larger `Successful requests`) — **this is
   the one to read metrics from**.

Warmup size varies by recipe (e.g. ~2 prompts for the Kimi low-latency sweep,
~128 prompts for the GLM-5 default). See each recipe's `README.md` for a real
sample block and the exact warmup/measured prompt counts.

### Key metrics

| Metric                              | Meaning                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------- |
| **Output token throughput (tok/s)** | Output tokens generated per second, aggregated across all concurrent requests |
| **Total Token throughput (tok/s)**  | (Input + output) tokens per second, aggregated across all concurrent requests |
| **Median TPOT (ms)**                | Median time per output token, excluding the first                             |

## Selecting a recipe variant

Each recipe ships with a sensible default. To pick a different variant, export
the recipe-specific env var **before** submitting:

```bash
export RUN_CONF_GLM5_RECIPE="32k_4k/ctx1dep2_gen4tep8_batch1_eplb0_mtp0_conc_1_4.yaml"
llmb-run submit -w inference_glm5 --scale 1
```

For persistent override across all submits, put it in `cluster_config.yaml`'s
`environment:` block — `llmb-run` propagates `RUN_CONF_*` env vars to every job.

| Recipe   | Env var                  | Default value                                                     |
| -------- | ------------------------ | ----------------------------------------------------------------- |
| GLM5     | `RUN_CONF_GLM5_RECIPE`   | `8k_1k/ctx4dep2_gen3tep8_batch64_eplb0_mtp0_conc_64_128_256.yaml` |
| Kimi 2.6 | `RUN_CONF_KIMI26_RECIPE` | `gb200_nvfp4/ISL8K_OSL1K/low_latency_conc_1_4_16_32_64.yaml`      |

The full variant tables (with concurrency sweeps and ISL/OSL pairs) live in each
recipe's own `README.md` under `glm5/disagg/trtllm_dynamo/` and
`kimi2.6/disagg/trtllm_dynamo/`.

## Troubleshooting

### `inference_glm5` / `inference_kimi2.6` doesn't appear in the `llmb-install` workload menu

Symptom: the workload doesn't appear in the workload-selection menu when running `llmb-install`.

Cause: the cluster's `gpu_type` (in `cluster_config.yaml`) doesn't match the recipe's declared SKU. `inference_glm5` is gb300 only; `inference_kimi2.6` is gb200 only.

Fix: use separate `$LLMB_INSTALL` directories for separate SKUs and pick the matching `gpu_type` during install.

### Install marks the workload FAILED but the underlying job is still running

Symptom: `llmb-install` reports the workload as FAILED, but `squeue --me` shows the install SLURM job still running.

Cause: your SSH session dropped and the foreground poll loop died with it. The install job continues running in the background, decoupled from the local installer.

Fix: wait for the SLURM job to finish, or `scancel` it, before re-running `llmb-install`. Task 1 (`install_srtslurm`) records the active job ID in `$LLMB_WORKLOAD/srt-slurm/.llmb-install-job-id` and refuses to rebuild `.venv` while that job is still active — this prevents a retry from pulling the rug out from under the still-running install. Once the job has terminated, the next `llmb-install` clears the marker and proceeds. For long installs, run inside `tmux` from the start.

### `llmb-run submit` errors with "workload not found" or similar

Symptom: `llmb-run submit -w <workload> --scale 1` errors with "workload not found" or a similar lookup failure.

Cause: the LLMB bootstrap venv is not active in the current shell.

Fix: activate it before submitting:

```bash
source $(dirname $LLMB_INSTALL)/llmb_venv/bin/activate
```

Or use the `llmb-run` symlink created at `$LLMB_INSTALL/llmb-run`.

### Benchmark submission fails at sbatch with "Invalid generic resource (gres) specification"

Symptom: the dispatcher's `srtctl apply` errors out at sbatch with `Invalid generic resource (gres) specification` (or a similar SBATCH-directive rejection).

Cause: your cluster doesn't accept one of the default SBATCH directives this recipe emits (`--gpus-per-node`, `--segment`, `--exclusive`).

Fix: edit `$LLMB_WORKLOAD/srt-slurm/srtslurm.yaml` and flip the offending flag back to `false`. See [Internals: SBATCH directive compatibility](#sbatch-directive-compatibility) for the full flag table and which one to disable for each rejection.

### Benchmark log shows "Exec format error"

Symptom: the benchmark log shows `Exec format error` when workers try to start.

Cause: you're on a heterogeneous cluster (x86_64 login + aarch64 compute) and a Python venv was built on the wrong architecture — the compute node tries to execute x86_64 binaries from a venv that should have been aarch64-built.

Fix: see [Internals: heterogeneous architecture handling](#heterogeneous-architecture-handling) for the diagnostic command to confirm the arch mismatch and the rebuild steps.

### Benchmark hangs during init — prefill never registers with the frontend

Symptom: the sweep log shows `Model is not ready, waiting for 1 prefills and 0 decodes` repeating every 60 seconds for the full job wall time (~1 hour), then `Model did not get healthy in 3600 seconds`. Decode workers register fine; the prefill worker's log stops after `KVCacheManager` init and never reaches `Using UCX kv-cache transceiver`.

Cause: SLURM allocated the 6 nodes (1 prefill + 5 decode) across **multiple NVL72 / NVLink switch domains**. The prefill's cross-node UCX/NIXL setup hangs trying to negotiate transfers over IB instead of NVLink. The pattern is easy to spot in `squeue`: failing runs have non-consecutive node IDs (e.g. `0039,0040,0045,0047,0048`); successful runs have consecutive IDs (e.g. `0150-0155`).

Fix: enable the `--segment` SBATCH directive in `srtslurm.yaml` so SLURM is forced to allocate within a single NVL72:

```yaml
use_segment_sbatch_directive: true
```

Edit at `$LLMB_WORKLOAD/srt-slurm/srtslurm.yaml` (no re-install needed — the next `llmb-run submit` picks it up and the benchmark sbatch emits `#SBATCH --segment=<total_nodes>`).

Requires cluster support for `--segment` (GB200 NVL72 clusters like ptyche support it). If your cluster rejects it with `sbatch: error: Invalid --segment specification`, set the flag back to `false` and contact your cluster admin about NVLink-aware allocation. See [Internals: SBATCH directive compatibility](#sbatch-directive-compatibility) for the full flag table.

______________________________________________________________________

# Internals

The sections below describe how the recipes work under the hood. **End users do
not need to read these** — they're aimed at recipe authors and at advanced
troubleshooting.

## Heterogeneous architecture handling

Both recipes are validated on heterogeneous clusters where the login node is
x86_64 and the compute nodes are aarch64 (Grace). The install path builds two
Python virtual environments:

- A **login-side** venv (x86_64) created by the LLMB installer, used only to
  *issue* commands from the login host.
- A **compute-side** venv (aarch64) built via `srun` on a Grace compute node so
  its binaries match the architecture of the nodes the actual install + benchmark
  jobs run on.

A single venv won't work for both sides — binaries built on x86_64 fail with
"Exec format error" when activated on aarch64. The dual-venv split solves this
and is the single most important implementation detail.

If you ever see "Exec format error" in a benchmark log, verify the compute-side
venv is aarch64:

```bash
file $LLMB_WORKLOAD/srt-slurm/.venv/bin/python3
# expect: "ELF 64-bit LSB executable, ARM aarch64, ..."
```

If it's x86_64, delete it and re-run `llmb-install` — the install task will
rebuild it via `srun` on a Grace compute node.

## Dispatcher pattern (what `llmb-run submit` actually does)

`llmb-run submit` sbatches a small **dispatcher** job (1 node, ~30 seconds).
The dispatcher activates the compute-side venv, switches into the cloned
srt-slurm directory, and runs `srtctl apply -f <recipe>.yaml`. `srtctl apply`
itself sbatches the actual multi-node benchmark and returns. The dispatcher
exits — fire-and-forget.

This produces two SLURM jobs per submit:

```text
JOBID    NAME      ST  NODES  comment
<short>  inferenc  CG   1     dispatcher (~30s)
<long>   <recipe>  R   N      real benchmark (multi-node, multi-minute)
```

Verify with `sacct` after both finish:

```bash
sacct -X -j <DISPATCHER_JOB>,<BENCHMARK_JOB> \
    --format=JobID,JobName,State,Elapsed,NNodes,NodeList -P
```

Expected: both `COMPLETED`, dispatcher elapsed < 1 min, benchmark `NNodes`
matches the recipe topology.

## SBATCH directive compatibility

The install task writes three SBATCH compatibility flags into
`$LLMB_WORKLOAD/srt-slurm/srtslurm.yaml`:

```yaml
use_gpus_per_node_directive: false
use_segment_sbatch_directive: false
use_exclusive_sbatch_directive: true
```

These defaults are tuned for typical NVIDIA Grace clusters. They control which
`#SBATCH` directives the benchmark sbatch script emits. If your cluster doesn't
accept them, the submission fails with `Invalid generic resource (gres) specification` or similar. Edit the file post-install:

| Flag                             | Maps to                     | Default | Flip if …                               |
| -------------------------------- | --------------------------- | ------- | --------------------------------------- |
| `use_gpus_per_node_directive`    | `#SBATCH --gpus-per-node=N` | `false` | your cluster supports `--gpus-per-node` |
| `use_segment_sbatch_directive`   | `#SBATCH --segment=...`     | `false` | your cluster uses `--segment`           |
| `use_exclusive_sbatch_directive` | `#SBATCH --exclusive`       | `true`  | your cluster doesn't allow exclusive    |

Discover what your cluster supports:

```bash
sbatch --help | grep -E "gpus-per-node|segment|exclusive"
```

After editing, the next `llmb-run submit` picks up the change — no re-install.

## Architecture support

The recipes here are validated for **Grace SKUs only (gb200, gb300)**. Two
reasons:

1. The install task hardcodes `ARCH=aarch64` and errors on non-Grace SKUs.
2. Each `metadata.yaml` declares only its target Grace SKU under `gpu_configs`,
   so `llmb-run` filters the recipe out on non-Grace clusters.

The dual-venv pattern itself is architecture-agnostic — if upstream srt-slurm
ships h100 / b-series recipes in the future, supporting them here is a small
delta (extend the ARCH case statement + add the SKU to `gpu_configs`).

## Known design notes

- **Dispatcher overhead**: every `llmb-run submit` allocates one GPU node for
  ~26 sec to run the dispatcher (`launch.sh` → `srtctl apply`). This is the
  cost of using `launcher_type: configured_sbatch`. Eliminating it requires a
  new launcher type in `llmb-run` — tracked separately, not in scope here.

- **`$LLMB_IMAGE_FOLDER`** is intentionally NOT used by these recipes. That
  variable is reserved for static, install-agnostic shared images and is not
  appropriate for install-specific artifacts like the `.sqsh` produced during
  `install_model`. The `.sqsh` stays where srtctl writes it under
  `$LLMB_WORKLOAD/srt-slurm/install/containers/`.

## Adding a new recipe

1. Create `agenticInference/inference_short/<model>/<topology>/<framework>/`
   mirroring an existing recipe (e.g.
   `agenticInference/inference_short/<model>/disagg/trtllm_dynamo/`).
2. In `metadata.yaml`:
   - `general.workload: <model>`, `general.workload_type: inference`
   - `gpu_configs.<sku>` matching your target SKU
   - `model_size` = total parameter count (e.g. `'744b'`); for MoE follow the
     existing convention (deepseek_v3 uses `'671b'` = total params)
   - `dtypes: ['<precision>']`
3. In `install_model.sh`: swap `INSTALL_NAME`, `HF_REPO_ID`, `MODEL_ALIAS`,
   `CONTAINER_IMAGE`.
4. In `launch.sh`: update `WORKLOAD_KEY`, `RECIPE_REL` default, and the
   `RUN_CONF_<NAME>_RECIPE` env var name.
5. Update the recipe table at the top of this README.
6. Static-validate before pushing:
   ```bash
   for f in agenticInference/inference_short/<model>/.../*.sh; do shellcheck "$f" && bash -n "$f"; done
   yamale -s .gitlab/ci/metadata_schema.yaml agenticInference/inference_short/<model>/.../metadata.yaml
   ```

## Conventions

- **Workload key** = `${workload_type}_${workload}` (constructed by the
  framework). For these recipes: `inference_glm5`, `inference_kimi2.6`.
- **`workload_type: inference`** is auto-exempted from `llmb-run`'s
  post-processing pipeline (`UNSUPPORTED_PP_WORKLOAD_TYPES` in
  `cli/llmb-run/src/llmb_run/internal/post_processing_pipeline.py`). These
  recipes do not ship a results parser; the benchmark writes its own outputs.
- **`scales: [1]`** → one dispatcher allocation; the real sweep matrix is owned by
  the srt-slurm recipe YAML.
- **Outputs** live under `srt-slurm/outputs/<jobid>/`, not in LLMB's experiments tree
  (which only holds the dispatcher's slurm-\*.out).
- **srt-slurm branch**: `llmb/inf-beta`, fetched fresh on every install resume.

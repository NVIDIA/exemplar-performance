# Inference Recipes (Long Context)

LLMB inference recipes for long-context inference on NVIDIA Grace systems
(verified on gb200). Driven through the standard install flow — `llmb-install` to
provision, `llmb-run submit` to launch a benchmark.

## Available recipes

Currently one recipe, future updates will include more.

| Workload key           | GPU SKU | Model            | Topology   | Path                |
| ---------------------- | ------- | ---------------- | ---------- | ------------------- |
| `inference_qwen3_long` | gb200   | QWEN3-32B (BF16) | SGLang agg | `qwen3/agg/sglang/` |

Current recipe is validated on polyphe (gb200). See the per-recipe
`README.md` inside the folder for recipe-specific details.

## Prerequisites

1. **SLURM cluster access** with one of the supported Grace SKUs (gb200).

2. **HuggingFace token** with the following access:

   - read access to the model repository `Qwen/Qwen3-32B`
   - read access to the dataset repository `nv-camilom/agentic_coding`

3. **`uv`** — the installer auto-installs it via `install.sh` if missing.

4. **`tmux` or `screen` strongly recommended** for installs. The model fetch step
   can take a long time and is driven by a foreground polling loop. If your SSH
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

## Install and Run

Follow the directions in the [individual recipe README](qwen3/agg/sglang/README.md)

## Troubleshooting

### `inference_qwen3_long` doesn't appear in the `llmb-install` workload menu

The recipe is filtered out because the cluster's `gpu_type` (in
`cluster_config.yaml`) doesn't match the recipe's declared SKU. the current `qwen3_long` recipe is gb200 only. You need separate `$LLMB_INSTALL` dirs for separate SKUs.

### Install marks the workload FAILED but the underlying job is still running

Your SSH session died, the foreground poll loop went with it. The install job
may still be running in the background — check `squeue --me`. If it is,
**wait for it to finish or `scancel` it before re-running `llmb-install`**.
Task 1 (`install_srtslurm`) records the active job ID in
`$LLMB_WORKLOAD/srt-slurm/.llmb-install-job-id` and will refuse to rebuild
`.venv` while that job is still active — this prevents a retry from pulling
the rug out from under the still-running install. Once the job has
terminated, the next `llmb-install` clears the marker and proceeds. For long
installs, run inside `tmux` from the start.

### `llmb-run submit` errors with "workload not found" or similar

Activate the LLMB bootstrap venv first:

```bash
source $(dirname $LLMB_INSTALL)/llmb_venv/bin/activate
```

Or use the `llmb-run` symlink created at `$LLMB_INSTALL/llmb-run`.

### Benchmark submission fails at sbatch with "Invalid generic resource (gres) specification"

Your cluster doesn't accept the default SBATCH directives this recipe emits.
See [Internals: SBATCH directive compatibility](#sbatch-directive-compatibility)
for which flags to flip.

### Benchmark log shows "Exec format error"

You're on a heterogeneous cluster (x86_64 login + aarch64 compute) and a venv
got built on the wrong side. See
[Internals: heterogeneous architecture handling](#heterogeneous-architecture-handling)
for the diagnostic + fix.

______________________________________________________________________

# Internals

The sections below describe how the recipes work under the hood. **End users do
not need to read these** — they're aimed at recipe authors and at advanced
troubleshooting.

## Heterogeneous architecture handling

The recipe is validated on heterogeneous clusters where the login node is
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
srt-slurm directory, and runs `srtctl apply -f <recipe>.yaml` once per
targeted `SESSION_CONCURRENCY`. `srtctl apply`
itself sbatches the actual benchmark and returns. The dispatcher
exits — fire-and-forget.

This produces multiple SLURM jobs per submit:

```text
JOBID    NAME      ST  NODES  comment
<short>  inferenc  CG   1     dispatcher (~30s)
<long>   <recipe>  R   1      real benchmark (multiple)
...
```

Verify with `sacct` after all finish:

```bash
sacct -X -j <DISPATCHER_JOB>,<BENCHMARK_JOB> \
    --format=JobID,JobName,State,Elapsed,NNodes,NodeList -P
```

Expected: all `COMPLETED`, dispatcher elapsed < 1 min, all others about 1hr

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

The recipe here is validated for **Grace SKUs only (gb200, gb300)**. Two
reasons:

1. The install task hardcodes `ARCH=aarch64` and errors on non-Grace SKUs.
2. Each `metadata.yaml` declares only its target Grace SKU under `gpu_configs`,
   so `llmb-run` filters the recipe out on non-Grace clusters.

The dual-venv pattern itself is architecture-agnostic — if upstream srt-slurm
ships h100 / b-series recipes in the future, supporting them here is a small
delta (extend the ARCH case statement + add the SKU to `gpu_configs`).

## Known design notes

- **Dispatcher overhead**: every `llmb-run submit` allocates one GPU node
  briefly to run the dispatcher (`launch.sh` → `srtctl apply`). This is the
  cost of using `launcher_type: configured_sbatch`. Eliminating it requires a
  new launcher type in `llmb-run` — tracked separately, not in scope here.

- **`$LLMB_IMAGE_FOLDER`** is intentionally NOT used by this recipe. That
  variable is reserved for static, install-agnostic shared images and is not
  appropriate for install-specific artifacts like the `.sqsh` produced during
  `install_model`. The `.sqsh` stays where srtctl writes it under
  `$LLMB_WORKLOAD/srt-slurm/install/containers/`.

## Adding a new recipe

1. Create `agenticInference/inference_long/<model>/<topology>/<framework>/`
   mirroring an existing recipe (e.g.
   `agenticInference/inference_long/<model>/disagg/trtllm_dynamo/`).

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
   for f in agenticInference/inference_long/<model>/.../*.sh; do shellcheck "$f" && bash -n "$f"; done
   yamale -s .gitlab/ci/metadata_schema.yaml agenticInference/inference_long/<model>/.../metadata.yaml
   ```

## Conventions

- **Workload key** = `${workload_type}_${workload}` (constructed by the
  framework). For this recipe: `inference_qwen3_long`
- **`workload_type: inference`** is auto-exempted from `llmb-run`'s
  post-processing pipeline (`UNSUPPORTED_PP_WORKLOAD_TYPES` in
  `cli/llmb-run/src/llmb_run/internal/post_processing_pipeline.py`). The
  recipe ships with a simple `report.sh` to parse the logs and extract the key
  performance metrics.
- **`scales: [1]`** → one dispatcher allocation; the real sweep matrix is done
  inside the dispatcher.
- **Outputs** live under `srt-slurm/outputs/<jobid>/`, not in LLMB's experiments tree
  (which only holds the dispatcher's slurm-\*.out).
- **srt-slurm branch**: `llmb/inf-beta`, fetched fresh on every install resume.

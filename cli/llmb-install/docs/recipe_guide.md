# Recipe Development Guide

This guide explains how to create and configure workload recipes using the `metadata.yaml` file. It covers all available configuration options and patterns for defining workloads.

## Overview

Each workload recipe requires a `metadata.yaml` file that defines:

- **General Information**: Workload identification and framework
- **Container Images**: Runtime environment containers
- **Repositories**: Git repositories for dependencies
- **Downloads**: Offline assets (tokenizers, models, datasets)
- **Setup**: Optional virtual environment, dependency installation, and setup tasks
- **Tools**: Workload-specific tool versions (e.g., nsys)
- **Run Configuration**: GPU configs, model sizes, and test scales

## Metadata Structure

A complete `metadata.yaml` follows this structure:

```yaml
general:
  # Workload identification
  
container:
  # Container images
  
repositories:  # Optional
  # Git repositories

downloads:  # Optional
  # Offline assets (tokenizers, models, datasets)
  
tools:  # Optional
  # Tool versions
  
setup:  # Optional
  # Dependencies and setup tasks, if needed
  
run:
  # Launch configuration and GPU configs
```

## General Section (Required)

Identifies the workload at a high level:

```yaml
general:
  workload: qwen3                  # workload model name
  workload_type: pretrain          # Type of workload
  framework: megatron_bridge       # Framework used
  model: qwen3                     # Optional: Override model name in llmb-config
```

### Fields

- **`workload`** (string, required): Name of the workload, must match the directory name
- **`workload_type`** (enum, required): One of:
  - `pretrain` - Pre-training workloads
  - `inference` - Inference workloads
  - `finetune` - Fine-tuning workloads
  - `microbenchmark` - Microbenchmark workloads
- **`framework`** (string, required): Framework name (e.g., `nemo2`, `maxtext`, `megatron`)
- **`model`** (string, optional): Model name to use in `llmb-config_jobid.yaml` for `model_info.model_name`. If not specified, defaults to the `workload` value. Useful when multiple workload directories share the same base model (e.g., `llama3.1` and `llama3.3` both use `model: llama3`)

**Note**: Version information is managed centrally in `release.yaml` at the repository root and does not need to be specified in individual recipe metadata files.

## Container Section (Required)

Defines the OCI container images that provide the runtime environment.

### Simple Format (Same Container for All GPUs)

```yaml
container:
  images: 
    - 'nvcr.io#nvidia/nemo:25.07.01'
```

### Multiple Images

```yaml
container:
  images:
    - 'nvcr.io#nvidia/nemo:25.07.01'
    - 'nvcr.io#nvidia/pytorch:24.12-py3'
```

### Custom Image Names

Override the automatically generated filename:

```yaml
container:
  images:
    - url: 'nvcr.io#nvidia/nemo:25.07.01'
      name: 'my-custom-name.sqsh'
```

### GPU-Conditional Images

Use different containers for different GPU types:

```yaml
container:
  images:
    by_gpu:
      h100: 'nvcr.io#nvidia/nemo:25.01'
      gb200: 'nvcr.io#nvidia/nemo:25.05'
      default: 'nvcr.io#nvidia/nemo:25.07.01'  # Fallback for other GPUs
```

**Note**: Image URLs use `#` instead of `/` between registry and image path.

## Repositories Section (Optional)

Defines Git repositories to clone during setup. These can be used as dependencies or referenced in the setup.

### Simple Format

```yaml
repositories:
  nemo:
    url: "https://github.com/NVIDIA/NeMo.git"
    commit: "763ffa8b00a2fca9f7a204e14111ed190de7d947"  # Full 40-char SHA
  megatron_core:
    url: "https://github.com/NVIDIA/Megatron-LM.git"
    commit: "ac198fc0d60a8c748597e01ca4c6887d3a7bcf3d"
```

### GPU-Conditional Repositories

```yaml
repositories:
  by_gpu:
    h100:
      nemo:
        url: "https://github.com/NVIDIA/NeMo.git"
        commit: "abc123..."
    gb200:
      nemo:
        url: "https://github.com/NVIDIA/NeMo.git"
        commit: "def456..."
    default:
      nemo:
        url: "https://github.com/NVIDIA/NeMo.git"
        commit: "789abc..."
```

**Important**: Commit must be the full 40-character SHA hash, not a short hash or tag.

## Downloads Section (Optional)

Specifies offline assets to download during installation. This section is used to ensure models and tokenizers are available in air-gapped or offline environments.

### HuggingFace Downloads

The `huggingface` key supports two kinds of downloads, named for where the files land:

- **`cache`** entries populate the shared HuggingFace cache at `$LLMB_INSTALL/.cache/huggingface` for offline `AutoTokenizer` / `AutoConfig` loading.
- **`repos`** entries download HuggingFace repository contents into an LLMB-managed workload or shared dataset directory.

```yaml
downloads:
  huggingface:
    cache:
      - repo_id: Qwen/Qwen3-30B-A3B
        assets: [tokenizer, config]   # Optional: defaults to both if omitted
    repos:
      - repo_id: nvidia/DeepSeek-R1-0528-FP4
        repo_type: model              # Required: 'model' or 'dataset'
        name: DeepSeek-R1-FP4         # Optional: defaults to repo basename
        when:
          gpu: [gb200, b200]
```

For recipes that only need cache assets, the bare list shorthand remains supported and is equivalent to a `cache` list:

```yaml
downloads:
  huggingface:
    - repo_id: Qwen/Qwen3-30B-A3B
      assets: [tokenizer, config]
```

Repo entries are downloaded to:

- `target: workload` -> `$LLMB_INSTALL/workloads/<workload_key>/<name>`
- `target: shared` -> `$LLMB_INSTALL/datasets/<name>`

#### Cache entry fields

- **`repo_id`** (string, required): The HuggingFace repository ID. Cache entries are always `model` repos.
- **`revision`** (string or null, optional): Branch, tag, or commit to download. Defaults to the HuggingFace default revision.
- **`assets`** (list of enums, optional): Which assets to download. Allowed values: `tokenizer`, `config`. Defaults to **both**.

#### Repo entry fields

- **`repo_id`** (string, required): The HuggingFace repository ID.
- **`repo_type`** (enum, required): HuggingFace repository type. Allowed values: `model`, `dataset`. Note that on the HuggingFace Hub, models and datasets with the same name are different repositories - a dataset entry with `repo_type: model` will fail at download time.
- **`revision`** (string or null, optional): Branch, tag, or commit to download. Defaults to the HuggingFace default revision.
- **`target`** (enum, optional): Destination root. `workload` (default) places the download under the workload directory; `shared` places it under `$LLMB_INSTALL/datasets`, shared and de-duplicated across workloads.
- **`name`** (string, optional): Destination directory name. Defaults to the HuggingFace repo basename.
- **`include`** (list of strings, optional): HuggingFace `allow_patterns` passed to `snapshot_download`.
- **`exclude`** (list of strings, optional): HuggingFace `ignore_patterns` passed to `snapshot_download`.
- **`when.gpu`** (list of GPU types, optional): Restricts the entry to matching install-time GPU types.

#### Behavior and Rules

- **Cache entries avoid weights**: cache downloads do **NOT** download model weights (SafeTensors/Pickle). They only download metadata, tokenizers, and configuration files.
- **Repo entries download repo contents**: including weights unless filtered. Use `include` / `exclude` when only part of a repository is needed.
- **Download vs. Verify**: Cache downloads run first, then a separate verification step checks that required assets load offline (`local_files_only=True`). Repo downloads are verified by a clean HuggingFace download exit.
- **Caching**: HuggingFace always uses the installer-owned cache at `$LLMB_INSTALL/.cache/huggingface`, with `HF_HOME` and `HF_HUB_CACHE` set during downloads.
- **Destination safety**: Repo destinations are constrained to LLMB-managed workload and shared dataset directories. Two different sources cannot resolve to the same destination; identical sources targeting the same shared destination are downloaded once.
- **Setup remains for transforms**: Conversions, prompt dataset generation, checkpoint transforms, repository patching, and containerized setup should remain in `setup.tasks`.
- **Migration scope**: Existing recipes with download scripts should be moved to `repos` entries in focused follow-up changes.

### Legacy: hf_tokenizers

The `hf_tokenizers` key is supported for backward compatibility but is restricted to tokenizers only. It does **not** download model configurations.

```yaml
downloads:
  hf_tokenizers:
    - 'meta-llama/Meta-Llama-3-70B'
```

> [!important]
> **Exclusivity Rule**: You cannot use both `hf_tokenizers` and `huggingface` within the same `metadata.yaml` file. Mixing them will result in a validation error.

### Migration Guidance

Existing recipes using `hf_tokenizers` should eventually migrate to the `huggingface` structure. Note that `hf_tokenizers` only downloads the tokenizer, while the new `huggingface` key defaults to both tokenizer and config.

**Legacy (Tokenizer only):**

```yaml
downloads:
  hf_tokenizers:
    - 'Qwen/Qwen3-30B-A3B'
```

**Migrated (Tokenizer only):**

```yaml
downloads:
  huggingface:
    - repo_id: Qwen/Qwen3-30B-A3B
      assets: [tokenizer]
```

### Examples

#### 1. Default (Tokenizer + Config)

Omit the `assets` field to download both.

```yaml
downloads:
  huggingface:
    - repo_id: Qwen/Qwen3-30B-A3B
```

#### 2. Tokenizer-only syntax

Current recipes usually need both tokenizer and config assets. This example uses a current repository ID only to show the `assets: [tokenizer]` syntax for a recipe that intentionally needs tokenizer files only.

```yaml
downloads:
  huggingface:
    - repo_id: Qwen/Qwen3-30B-A3B
      assets: [tokenizer]
```

#### 3. Config-only (Rare)

```yaml
downloads:
  huggingface:
    - repo_id: meta-llama/Llama-3.1-405B
      assets: [config]
```

#### 4. Repo download into a workload directory (GPU-conditional)

```yaml
downloads:
  huggingface:
    repos:
      - repo_id: deepseek-ai/DeepSeek-R1
        repo_type: model
        name: DeepSeek-R1-FP8
        when:
          gpu: [h100]
      - repo_id: nvidia/DeepSeek-R1-0528-FP4
        repo_type: model
        name: DeepSeek-R1-FP4
        when:
          gpu: [gb200, b200]
```

#### 5. Dataset repo into the shared datasets directory

```yaml
downloads:
  huggingface:
    repos:
      - repo_id: my-org/prompt-dataset
        repo_type: dataset
        revision: main
        target: shared
        include: ["*.jsonl", "*.txt"]
        exclude: ["scratch/*"]
```

#### 6. Cache assets plus a dataset repo

```yaml
downloads:
  huggingface:
    cache:
      - repo_id: my-org/model
        assets: [config]
    repos:
      - repo_id: my-org/prompt-dataset
        repo_type: dataset
        target: shared
        name: prompt-dataset
```

**Note**: Accessing private or gated models requires the `HF_TOKEN` environment variable to be set during the installation phase.

## Tools Section (Optional)

Specifies workload-specific tool versions (currently supports `nsys` for profiling).

**Only use this section when you need a specific tool version.** If your container's tools work fine, omit this section.

### Simple Format (All GPUs Same Version)

```yaml
tools:
  nsys: "2025.5.1.121-3638078"
```

### GPU-Conditional Tools

Use different versions for different GPU types:

```yaml
tools:
  nsys:
    by_gpu:
      h100: "2025.1.1.118-3638078"
      gb200: "2025.5.1.121-3638078"
      default: "2025.4.1.172-3634357"  # Optional: fallback version
```

### Partial GPU Coverage

Only specify versions for GPUs that need custom tools (others use container version):

```yaml
tools:
  nsys:
    by_gpu:
      h100: "2025.1.1.118-3638078"
      gb200: "2025.5.1.121-3638078"
      # b200 and other GPUs will use container nsys (no download)
```

**Resolution Logic**:

1. If GPU explicitly listed in `by_gpu` → use that version
2. Else if `default` key exists → use default version
3. Else → use container version (no download)

For more details, see [tools.md](tools.md).

## Setup Section (Optional)

Defines virtual environment creation, dependencies, and setup tasks. Omit this section for image-only recipes that only need container downloads and run metadata.

### Basic Setup with Dependencies

```yaml
setup:
  venv_req: true  # Create a Python virtual environment
  dependencies:
    pip:
      - package: nemo
        repo_key: nemo
        install_target: '.[nlp]'
      - 'scipy<1.13.0'
      - 'bitsandbytes==0.46.0'
      - package: megatron-core
        repo_key: megatron_core
```

### Dependencies Reference

#### Pip Dependencies

Simple string format (PyPI package):

```yaml
dependencies:
  pip:
    - 'numpy==1.24.0'
    - 'torch>=2.0'
```

Repository-based package:

```yaml
dependencies:
  pip:
    - package: nemo           # Package name
      repo_key: nemo          # References key in repositories section
      install_target: '.[nlp]'  # Optional: extras or specific target
      editable: true          # Optional: install in editable mode (-e)
```

#### Git Dependencies

```yaml
dependencies:
  git:
    my_package:
      repo_key: my_repo       # References key in repositories section
      install_method:
        type: clone           # 'clone' or 'script'
        path: 'subdir'        # Optional: subdirectory within repo
```

### Setup Tasks

Run custom commands during setup:

```yaml
setup:
  venv_req: true
  tasks:
    - name: "Download dataset"
      cmd: "python download_data.py --output $DATASET_DIR"
      job_type: local        # 'local', 'nemo2', 'srun', or 'sbatch'
      requires_gpus: false   # Optional: whether task needs GPUs
      env:                   # Optional: environment variables
        DATASET_DIR: "/data"
```

**Task Types**:

- `local`: Run on current node
- `nemo2`: Run with nemo2 launcher
- `srun`: Run via SLURM srun
- `sbatch`: Submit as SLURM batch job

Setup tasks can be used with or without `dependencies`. If `venv_req: true` is set without dependencies, the installer creates an empty workload-specific virtual environment before running tasks. If `venv_req` is omitted or false, tasks run without a virtual environment.

## Run Section (Required)

Defines how the workload is launched and what configurations to test.

### Basic Structure

```yaml
run:
  launcher_type: 'nemo'      # Launcher type
  launch_script: 'launch.sh' # Launch script path
  gpu_configs:               # Per-GPU configurations
    h100:
      model_configs:
        - model_size: '405b'
          dtypes: ['fp8', 'bf16']
          scales: [512, 1024, 2048]
```

### Launcher Types

- **`nemo`**: NeMo launcher (nemo2 workloads)
- **`megatron_bridge`**: Megatron bridge launcher
- **`configured_sbatch`**: SLURM sbatch submission with llmb-run-managed experiment directories
- **`sbatch`**: Direct SLURM sbatch submission

### `configured_sbatch` Requirements

Use `configured_sbatch` when a workload needs a custom Slurm batch script and should use `llmb-run` managed experiment directories.

For this launcher, `llmb-run` submits the workload's `launch_script` with `sbatch`, creates the experiment directory under `$LLMB_INSTALL/workloads/<workload_key>/experiments/`, and exports that path as `LLMB_EXPERIMENT_DIR`.

Recipe requirements:

- Write run artifacts such as logs, generated configs, metrics, and checkpoints under `LLMB_EXPERIMENT_DIR`.
- Use `LLMB_INSTALL` to find installed workloads, checkpoints, datasets, and shared tools.
- Use task variables such as `MODEL_SIZE`, `DTYPE`, `JOB_TOTAL_GPUS`, and `GPU_TYPE` for workload selection. Standard Slurm job variables are also available because the script runs as the `sbatch` payload.

`llmb-run` writes `llmb-config_<JOBID>.yaml` into `LLMB_EXPERIMENT_DIR` and uses the same directory for job logs, archive collection, and supported post-processing.

### Launch Script Env Contract

Launch scripts should treat values passed through `llmb-run submit --env KEY=value` or a YAML task spec `env:` block as explicit container-launch overrides.

- `llmb-run` validates `--env` keys as bash-style environment variable names and exports the corresponding `KEY=value` pairs into the job environment. YAML `env:` entries from `-f` task files receive the same treatment.
- For `sbatch` and `configured_sbatch` launchers, `llmb-run` also exports `LLMB_CONTAINER_ENV=KEY1,KEY2,...`.
  Launch scripts that invoke `srun` should pass this through to Pyxis `--container-env`, and may append additional keys if needed.
- For `nemo` launchers, `llmb-run` appends repeatable `-E KEY=value` flags into `CONFIG_OVERRIDES`.
  Launch scripts should preserve that variable and may append additional override flags to it if needed.
- For `megatron_bridge` launchers, keep recipe-generated arguments in `CONFIG_OVERRIDES` and initialize it to an empty string.
  Append `${LLMB_MBRIDGE_EXTRA_ARGS:-}` as the final expansion in the `setup_experiment.py` command so explicit environment and workload arguments take precedence.

This contract covers explicit `--env` values and YAML `env:` blocks. Environment variables from cluster config or workload config continue to flow through the normal job environment unless the launch script chooses to add them to its container override mechanism.

### GPU Configs

Define test configurations for each GPU type:

```yaml
gpu_configs:
  h100:
    model_configs:
      - model_size: '30b'
        dtypes: ['bf16']
        scales: [16, 32, 64]
  b200:
    model_configs:
      - model_size: '30b'
        dtypes: ['bf16']
        scales: [8, 16, 32, 64]
```

**Supported GPU Types**: `h100`, `b200`, `gb200`, `gb300`

### Model Configs

Each model config specifies:

#### Simple Format

```yaml
model_configs:
  - model_size: '405b'
    dtypes: ['fp8', 'nvfp4']
    scales: [256, 512]
    exact_scales: false  # Optional: allow power-of-2 extension
```

#### Per-Dtype Scales

Define different scales for different dtypes:

```yaml
model_configs:
  - model_size: '405b'
    dtypes:
      fp8: [128, 256, 512]      # Short form
      bf16:                     # Long form with exact_scales
        scales: [256, 512]
        exact_scales: true
```

**Fields**:

- **`model_size`** (string, required): Model size identifier (e.g., `'7b'`, `'405b'`)
- **`dtypes`** (required): Precision types to test. Can be:
  - Single dtype: `'fp8'`
  - List: `['fp8', 'bf16']`
  - Mapping: `fp8: [128, 256]` or `fp8: {scales: [128, 256], exact_scales: true}`
- **`scales`** (list, optional): GPU counts to test (legacy, used when dtypes is not a mapping)
- **`exact_scales`** (bool, optional): If `false` (default), scales are extended to max with power-of-2 values
- **`proxy_scales`** (list, optional): Reduced GPU counts for quick validation/debug runs. Always treated as exact scales (no power-of-2 extension). Used with `llmb-run submit --proxy`

**Supported dtypes**: `fp8`, `bf16`, `nvfp4`

#### Proxy Scales

Proxy scales enable quick validation runs on reduced GPU counts for debug workflows. These runs use altered configurations and cannot be compared to production results.

**Simple format with proxy scales:**

```yaml
model_configs:
  - model_size: '7b'
    dtypes: ['fp8', 'bf16']
    scales: [8, 16, 32, 64, 128]      # Production scales
    proxy_scales: [4, 8]               # Debug/validation scales
```

**Per-dtype proxy scales:**

```yaml
model_configs:
  - model_size: '405b'
    dtypes:
      fp8:
        scales: [512, 1024, 2048]
        proxy_scales: [128, 256]       # Reduced scales for fp8
      bf16:
        scales: [256, 512, 1024]
        proxy_scales: [64, 128]        # Reduced scales for bf16
```

**Usage:**

```bash
# Use proxy scales for quick validation
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128 --proxy

# Auto-discovery only includes workloads with proxy_scales defined
llmb-run submit --max-scale 256 --proxy
```

**Important notes:**

- Proxy scales are always treated as exact (no automatic power-of-2 extension)
- Proxy runs use altered configurations optimized for smaller scale
- Results from proxy runs cannot be used for performance validation or extrapolation
- Designed for debugging, development, and configuration testing only

## GPU-Conditional Configuration Pattern

Many sections support GPU-specific overrides using the `by_gpu` pattern.

### General Pattern

```yaml
section_name:
  by_gpu:
    h100: <value_for_h100>
    b200: <value_for_b200>
    gb200: <value_for_gb200>
    gb300: <value_for_gb300>
    default: <fallback_value>  # Optional
```

### Resolution Logic

1. Check if GPU type explicitly listed → use that value
2. Else if `default` key exists → use default value
3. Else → use top-level value or system default

### Sections Supporting by_gpu

- **`container.images`**: Different containers per GPU
- **`repositories`**: Different repository versions per GPU
- **`tools`**: Different tool versions per GPU

## Complete Example

Here's a complete `metadata.yaml` example:

```yaml
general:
  workload: qwen3
  workload_type: pretrain
  framework: megatron_bridge

container:
  images:
    - 'nvcr.io#nvidia/nemo:26.04.00'

repositories:
  megatron_bridge:
    url: "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
    commit: "f4d10a3746d1220f2aef57d54d49303b9150d901"
  nemo_run:
    url: "https://github.com/NVIDIA-NeMo/Run.git"
    commit: "64b91e0187b93475ea0d54028317e349ced7ac1b"

downloads:
  huggingface:
    - repo_id: 'Qwen/Qwen3-30B-A3B'
    - repo_id: 'Qwen/Qwen3-235B-A22B'

setup:
  venv_req: true
  dependencies:
    git:
      megatron_bridge:
        repo_key: megatron_bridge
        install_method:
          type: clone
    pip:
      - package: nemo_run
        repo_key: nemo_run

run:
  launcher_type: 'megatron_bridge'
  launch_script: 'launch.sh'
  gpu_configs:
    gb300:
      model_configs:
        - model_size: '235b'
          dtypes: ['bf16']
          scales: [256, 512]
        - model_size: '30b'
          dtypes: ['bf16']
          scales: [8, 16, 32, 64]
    gb200:
      model_configs:
        - model_size: '235b'
          dtypes: ['bf16']
          scales: [256, 512]
        - model_size: '30b'
          dtypes: ['bf16']
          scales: [8, 16, 32, 64]
    b200:
      model_configs:
        - model_size: '235b'
          dtypes: ['bf16']
          scales: [256, 512]
        - model_size: '30b'
          dtypes: ['bf16']
          scales: [8, 16, 32, 64]
    h100:
      model_configs:
        - model_size: '235b'
          dtypes: ['bf16']
          scales: [256, 512]
        - model_size: '30b'
          dtypes: ['bf16']
          scales: [16, 32, 64]
```

## Validation

Validate your metadata file:

```bash
python -m yamale -s .gitlab/ci/metadata_schema.yaml <workload>/metadata.yaml
```

The schema validates:

- Required vs optional fields
- Field types (string, int, bool, list, etc.)
- Enum values (GPU types, dtypes, launcher types)
- Format requirements (commit SHA length, version patterns)

## Best Practices

### 1. Use GPU-Conditional Config Sparingly

Only use `by_gpu` when configurations truly differ by GPU type. Simple deployments should use the same config across GPUs.

### 2. Pin Dependencies Explicitly

```yaml
# Good
dependencies:
  pip:
    - 'scipy==1.12.0'
    - 'numpy>=1.24,<2.0'

# Avoid
dependencies:
  pip:
    - 'scipy'  # No version = unpredictable behavior
```

### 3. Use Full Commit Hashes

Always use full 40-character SHA hashes for repository commits:

```yaml
repositories:
  nemo:
    url: "https://github.com/NVIDIA/NeMo.git"
    commit: "763ffa8b00a2fca9f7a204e14111ed190de7d947"  # Good
    # commit: "763ffa8"  # BAD: short hash will fail validation
```

### 4. Document Scale Choices

Include comments explaining why certain scales are chosen:

```yaml
scales: [128, 256, 512, 1024]  # Tested scales for memory-optimal configs
exact_scales: true  # Don't extend - these are the only supported scales
```

### 5. Test After Schema Changes

Always validate and test install after modifying metadata:

```bash
# Validate schema
python -m yamale -s .gitlab/ci/metadata_schema.yaml workload/metadata.yaml

# Test installation
llmb-install express /tmp/test-install --workloads your_workload
```

## Common Patterns

### Pattern: Multi-Model Workload

```yaml
run:
  gpu_configs:
    h100:
      model_configs:
        - model_size: '7b'
          dtypes: ['fp8', 'bf16']
          scales: [8, 16, 32]
        - model_size: '70b'
          dtypes: ['fp8', 'bf16']
          scales: [64, 128, 256]
        - model_size: '405b'
          dtypes: ['fp8']
          scales: [512, 1024, 2048]
```

### Pattern: Inference Workload with Setup Task

```yaml
setup:
  venv_req: true
  tasks:
    - name: "Download model weights"
      cmd: "python download_weights.py --model $MODEL_NAME"
      job_type: local
      requires_gpus: false
      env:
        MODEL_NAME: "llama-3.1-405b"
        HF_TOKEN: "$HF_TOKEN"  # References environment variable
  dependencies:
    pip:
      - 'transformers>=4.35'
      - 'accelerate>=0.24'
```

### Pattern: GPU-Specific Container and Tools

```yaml
container:
  images:
    by_gpu:
      h100: ['nvcr.io#nvidia/nemo:25.01']
      gb200: ['nvcr.io#nvidia/nemo:25.05-gb']
      default: ['nvcr.io#nvidia/nemo:25.07.01']

tools:
  nsys:
    by_gpu:
      gb200: "2025.6.0.125-3638078"  # GB200 needs newer nsys
      # Other GPUs use container nsys
```

## Troubleshooting

### Invalid Schema Errors

**Error**: `workload_type: 'training' is not valid under any of the given enum values`

**Solution**: Use valid enum values. Check the schema for allowed values:

- workload_type: `pretrain`, `inference`, `finetune`
- GPU types: `h100`, `b200`, `b300`, `gb200`, `gb300`
- dtypes: `fp8`, `bf16`, `nvfp4`

### Repository Commit Issues

**Error**: `commit: '763ffa8' is not valid - must be 40 characters`

**Solution**: Use full commit hash:

```bash
# Get full hash
git rev-parse HEAD
# Or from GitHub: click commit, copy full SHA from URL or UI
```

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'megatron'`

**Solution**: Ensure package is in dependencies and repo_key references correct repository:

```yaml
repositories:
  megatron_core:
    url: "https://github.com/NVIDIA/Megatron-LM.git"
    commit: "..."

setup:
  dependencies:
    pip:
      - package: megatron-core
        repo_key: megatron_core  # Must match repository key
```

## Additional Resources

- **[Tools Configuration Guide](tools.md)**: Detailed tool version configuration
- **[Main README](../README.md)**: Installation and usage guide
- **[Headless Installation](headless-installation.md)**: Automated deployment guide

## Schema Reference

The complete schema is defined in `.gitlab/ci/metadata_schema.yaml`. Key enums and types:

### Enums

- **GPU Types**: `h100`, `b200`, `gb200`, `gb300`, `default` (for by_gpu only)
- **Workload Types**: `pretrain`, `inference`, `finetune`, `microbenchmark`, `rl`
- **Dtypes**: `fp8`, `bf16`, `nvfp4`, `mxfp4`
- **Launcher Types**: `nemo`, `megatron_bridge`, `configured_sbatch`, `sbatch`
- **Job Types**: `local`, `nemo2`, `srun`, `sbatch`

### Format Patterns

- **commit**: Full 40-character SHA hash
- **image URLs**: Use `#` instead of `/` (e.g., `nvcr.io#nvidia/nemo:25.07.01`)

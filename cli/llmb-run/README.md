# llmb-run

## Overview

A lightweight tool for automating submission of single jobs and batches of workloads.

## Quick Start

### Installation

The recommended way to install llmb-run is using the automated installer script:

```bash
# Run the installer script
$LLMB_REPO/install.sh
```

This script will:

1. Install required dependencies
2. Install llmb-run as a Python package
3. Launch the interactive installer to:
   - Configure your SLURM cluster settings
   - Select GPU type (h100, gb200, etc.)
   - Choose and install workloads
   - Create your `cluster_config.yaml`

### First Steps

After installation completes, you must change to your installation directory before using llmb-run:

```bash
# Change to your installation directory
cd $LLMB_INSTALL

# Verify installation and list available workloads
llmb-run list

# Run your first job (example)
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256

# Check submitted jobs and results
llmb-run jobs
```

**Note**: llmb-run requires access to `cluster_config.yaml` which is located in your installation directory. Always run llmb-run commands from this directory.

### Alternative Installation Methods

If you need to install llmb-run without the automated installer, see [Alternative Installation Methods](#alternative-installation-methods) below.

## Configuration

The `cluster_config.yaml` file contains several main sections:

### top-level cluster fields

High-level cluster settings:

- `schema_version`: Use `2` for the current flattened schema
- `llmb_repo`: Path to the LLM benchmarking collection repository
- `llmb_install`: Base installation directory for workloads and data
- `gpu_type`: GPU type for your cluster (`h100`, `gb200`, etc.)
- `cluster_name` (optional): Used in generated job metadata

### install

Install metadata:

- `install.node_architecture`: Node architecture used for profiling/tool mount resolution (`x86_64` or `aarch64`)

Legacy `launcher`-based configs are still supported for backward compatibility.

### environment

Environment variables that will be appended to every job:

- `HF_TOKEN`: Hugging Face token (required for some models)
- Common settings include `RUN_CONF_*` settings

### slurm

Slurm-specific configuration:

- Common keys under `slurm` apply to both GPU and CPU targets (e.g. `account`, `qos`)
- `slurm.gpu`: Required target-specific Slurm settings for benchmark jobs
- `slurm.cpu`: Optional target-specific Slurm settings for CPU-style jobs (post-processing, etc.). If omitted, falls back to `slurm.gpu`
- For GPU requests, `gres` is supported as a compatibility alias and maps to `SBATCH_GPUS_PER_NODE` (same behavior as legacy `gpu_gres`).
- Use `gres: null` in `slurm.cpu` to explicitly unset inherited GPU request settings.
- Keys may be provided as:
  - bare names (e.g. `partition`, `mem_per_gpu`) -> converted to `SBATCH_*`
  - explicit `SBATCH_*` names
  - explicit `SLURM_*` names
- Legacy fields (`gpu_partition`, `gpu_gres`, `cpu_partition`, `cpu_gres`) are still supported for compatibility, but cannot be mixed with `slurm.gpu`/`slurm.cpu`

### workloads

Workload configuration:

- `installed`: List of workloads installed on this cluster
- `config`: Workload-specific configuration (typically managed by installer)

**Note**: The script validates workloads against the `installed` list and GPU type compatibility. Only workloads that support your cluster's GPU type and are in the installed list will be available.

## Commands

llmb-run's primary interface is the `submit` command, which handles all job submission modes. Use `list` to discover available workloads and `jobs` to inspect submitted jobs.

### CLI Structure (Global vs Command Options)

`llmb-run` has **global options** that must appear **before** the command name, and **command options** that appear after the command name.

```bash
llmb-run [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]
```

Tip: use `llmb-run -h` to see global options, and `llmb-run <command> -h` (e.g. `llmb-run submit -h`) to see command-specific options.

**Global options (apply to all commands):**

- `-v, --verbose`: Enable verbose output including debug information.

**Examples:**

```bash
# Correct: global option BEFORE the command
llmb-run -v submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256

# Incorrect: global option AFTER the command (this will not work)
llmb-run submit -v -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256
```

### Submit Command

The `submit` command is a unified interface for all job submissions. It supports these common workflows:

#### Choose a Submit Workflow

Pick the workflow that matches how you want to run:

- **Single explicit target**: You provide `--workload`, `--model-size`, `--dtype`, and `--scale`.
  - Pattern: `llmb-run submit -w <workload> -s <model-size> --dtype <dtype> --scale <scale>`
- **Target-list selection**: You provide comma-separated `--workload` targets and omit `--model-size`.
  - Pattern: `llmb-run submit -w <workload>_<model-size>,<workload> --dtype <dtype> --scale <scale>`
- **File-based (batch; special cases)**: You provide an input file and llmb-run submits the jobs listed in it.
  - Pattern: `llmb-run submit -f <file_path>`
- **Auto-discovery (submit all / many)**: You provide discovery constraints and llmb-run generates jobs from installed workload metadata.
  - Pattern: `llmb-run submit --max-scale <num_gpus>`
  - Example: `llmb-run submit --max-scale 512` (submits eligible installed workloads up to 512 GPUs; see the section below for additional limiting flags)

#### 1. Single Explicit Target

Submit one workload/model-size target with explicit parameters.

```bash
llmb-run submit -w <workload> -s <model-size> --dtype <dtype> --scale <scale>
```

**Required Flags:**

- `-w, --workload`: Name of the workload (e.g., `pretrain_llama3.1`).
- `-s, --model-size`: Model size (e.g., `405b`, `70b`).
- `--dtype`: Data type (e.g., `fp8`, `bf16`).
- `--scale`: Number of GPUs. Accepts a single value or a comma-separated list.

Use this form when you are running one workload/model-size target. It keeps workload and model size separate, which is usually easiest to read for a first run.

**Examples:**

```bash
# Run a single configuration
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256

# Run multiple scales for the same workload
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 128,256,512

# Run with proxy configuration (altered config for debug workflows)
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 16 --proxy
```

#### 2. Target-List Selection

Use a comma-separated `-w` list when you want to apply the same dtype and scale choices across multiple workload targets. Use `--scale` for exact scales, or `--min-scale` / `--max-scale` for scale discovery. In this form, `-s` is not used because it is a single global model-size flag.

Each `-w` entry can be either:

- `<workload>_<model-size>` to select one model size for a multi-model-size workload.
- `<workload>` to select all installed model sizes for that workload. For workloads with only one model size, the suffix is optional.

Normal dtype and scale validation still applies, so llmb-run only generates supported combinations for the selected targets.

**Examples:**

```bash
# Run one Llama 3.1 model size plus the single-model-size Nemotron-H workload
llmb-run submit -w pretrain_llama3.1_70b,pretrain_nemotron-h --dtype fp8 --scale 128

# Run all installed Llama 3.1 model sizes plus Nemotron-H
llmb-run submit -w pretrain_llama3.1,pretrain_nemotron-h --dtype fp8 --min-scale

# Run target-list selection with scale discovery up to 512 GPUs
llmb-run submit -w pretrain_llama3.1_70b,pretrain_nemotron-h --dtype fp8 --max-scale 512
```

#### 3. File-Based Submission (Batch)

Submit multiple jobs defined in a file.

```bash
llmb-run submit -f <file_path>
```

**Supported Formats:**

- **Simple (.txt)**: For basic configurations.
- **Advanced (.yaml)**: For complex configurations with overrides and environment variables.

See [Bulk_Examples.md](Bulk_Examples.md) for detailed file format specifications and examples.

**Example:**

```bash
llmb-run submit -f my_experiment.yaml
```

#### 4. Auto-Discovery (Submit All)

Automatically discover and submit jobs for installed workloads based on metadata.

```bash
llmb-run submit --max-scale <num_gpus>
```

**Flags:**

- `--max-scale`: Run all workloads up to this scale.
- `--min-scale`: Run only the minimum supported scale for each workload.
- `--exact-scales`: Only use scales explicitly listed in workload metadata. Use this with `--max-scale` when you want all officially listed scales up to a limit, without adding larger power-of-2 scales beyond a workload's metadata.
- `-w, --workload`: Limit discovery to specific base workloads or workload-size targets (comma-separated).
- `--scale`: specific scales to run (comma-separated).
- `--proxy`: In auto-discovery, only workloads with `proxy_scales` defined are included.

By default, `--max-scale` may extend a workload beyond its largest metadata-listed scale by adding power-of-2 scales up to the requested maximum. `--exact-scales` disables that expansion. For example, if a Llama 3.1 8B target lists scales up to 128 GPUs in metadata, `--max-scale 512 --exact-scales` will not add 256- or 512-GPU jobs for that target.

**Examples:**

```bash
# Run all installed workloads up to 512 GPUs
llmb-run submit --max-scale 512

# Run all installed workloads up to 512 GPUs, but only at metadata-listed scales
llmb-run submit --max-scale 512 --exact-scales

# Run specific scales for all workloads
llmb-run submit --scale 128,256
```

#### Submit Options (All Submit Modes)

These flags apply to all `llmb-run submit` modes (single explicit target, target-list, file-based, and auto-discovery):

- `-r, --repeats <N>`: Repeat each job N times (default: 1).
- `-p, --profile`: Enable profiling for all submitted jobs.
- `--dump-env`: For Megatron-Bridge workloads, write a rank-0 environment snapshot to a separate job log file, similar to running `env` at job start. Common secret-like keys are redacted. This is ignored for other workloads.
- `--proxy`: Use proxy scales.
- `--dry-run`: Print the jobs that would be submitted without running them.
- `--force`: Bypass dtype/scale validation for one explicit task. Use only when you intentionally need to run a configuration outside workload metadata.

#### Megatron-Bridge Options

- `--mbridge-arg=ARG`: Pass one raw argument to a Megatron-Bridge workload. Use MBridge's `--flag=value` form or a Hydra override, and repeat the option to pass additional arguments. Compatible launch scripts append them after recipe overrides, so explicit user arguments take precedence.

```bash
llmb-run submit -w pretrain_llama3.1 -s 70b --dtype fp8 --scale 128 \
  --mbridge-arg=--micro_batch_size=2 \
  --mbridge-arg=model.hidden_size=4096
```

Arguments must be non-empty and cannot contain whitespace. No other validation is performed, and compatible launch scripts may interpret shell metacharacters. Quote the entire option in your invoking shell when it contains shell syntax, for example `--mbridge-arg='model.layers=[1,2,3]'`; downstream launch scripts may still apply their own shell processing.

Arguments are ignored with a warning for non-Megatron-Bridge workloads. This is a raw debugging interface, and secrets should be passed with `--env` instead.

#### Slurm Options

These flags control Slurm submission parameters and apply to all `llmb-run submit` modes:

- `--nice <INT>`: Lower the job priority via Slurm nice.
- `--nodelist <LIST>`: Restrict the job to a specific node list.
- `--exclude <LIST>`: Exclude specific nodes from the job.
- `--reservation <NAME>`: Submit the job under a Slurm reservation.
- `--segment <INT>`: Set the Slurm segment size for the job.
- `--env <KEY=VALUE>`: Repeatable environment override for the submitted job. Use this when you need a variable treated as an explicit launcher/container override.
- `--slurm-arg <KEY=VALUE>`: Pass an arbitrary Slurm parameter. Repeatable. Accepts `key=value` pairs or bare flags (e.g., `exclusive`). Do **not** include a leading `--`.

**Examples:**

```bash
# Lower job priority
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256 --nice 100

# Pin to specific nodes
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256 --nodelist node[001-032]

# Combine multiple Slurm options
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256 \
  --reservation my-reservation --exclude node099

# Pass explicit job environment overrides
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256 \
  --env NCCL_DEBUG=INFO --env OTHER_VAR=test

# Pass arbitrary Slurm flags
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256 \
  --slurm-arg constraint=gpu --slurm-arg exclusive
```

**Notes:**

- Parameters that have a dedicated flag (`nodelist`, `exclude`, `reservation`, `segment`, `nice`) must use that flag and cannot be passed via `--slurm-arg`.
- Slurm CLI flags **cannot** be combined with the `ADDITIONAL_SLURM_PARAMS` environment variable (set in the process environment, cluster config, workload config, or task overrides). If both are present, the command will fail with an error.

### List Command

The list command helps you discover available workloads and their configurations.

#### Basic Usage

```bash
llmb-run list
```

#### Options

- `-w, --workload <name>`: Show detailed information for a specific workload

#### Examples

1. List all installed workloads:

```bash
llmb-run list
```

2. Show details for a specific workload:

```bash
llmb-run list -w pretrain_llama3.1
```

### Jobs Command

The jobs command shows the local job history for the current `$LLMB_INSTALL`. History is stored in `$LLMB_INSTALL/.llmb/jobs.sqlite3`.

#### Basic Usage

```bash
llmb-run jobs
llmb-run jobs list
```

Both commands refresh non-terminal Slurm jobs before printing the table.

The jobs table shows workload, dtype, scale, job ID, Slurm status, elapsed time, and available performance results (`s/iter` and `TFLOPS/GPU`). Result columns are populated from supported NeMo 2 and Megatron-Bridge workload logs after a job reaches a terminal Slurm state. A failed or cancelled job can still show results if the log contains enough data.

#### Other Commands

- `llmb-run jobs show <job_id>`: Show details for one job, including its log directory and any parsed results.
- `llmb-run jobs log <job_id>`: Show the active log for one job.
- `llmb-run jobs log <job_id> --follow`: Follow the active log.
- `llmb-run jobs log <job_id> --path`: Print the active log file path.
- `llmb-run jobs log <job_id> --dir`: Print the job log directory.
- `llmb-run jobs log <job_id> --list`: List matching retry log files for the job.
- `llmb-run jobs refresh <job_id> [job_id ...]`: Re-check specific jobs with Slurm and update any results shown in the jobs table.
- `llmb-run jobs rebuild`: Rebuild history by scanning `$LLMB_INSTALL/workloads/**/llmb-config_*.yaml`.

Use `llmb-run jobs rebuild` once if you want to populate history from jobs that were submitted before the local history database existed. New jobs submitted with `llmb-run submit` are recorded automatically.

`llmb-run jobs log` supports NeMo/Megatron-Bridge workload logs and managed `configured_sbatch` experiment directories. Legacy `sbatch` jobs are tracked when submitted by `llmb-run`, but log paths cannot be resolved reliably and are skipped by `jobs rebuild`.

A `PURGED` status means `sacct` no longer has a record for the job (typically due to cluster accounting retention). It does not mean the job failed — it means llmb-run has lost track of it.

### Exemplar Command (Cloud Certification)

The exemplar command runs the cloud certification workload suite.

#### Basic Usage

```bash
llmb-run exemplar
```

#### Options

- `--dry-run`: Preview all jobs without submitting
- `-r, --repeats INTEGER`: Number of times to run each job (must be >= 1). If omitted, uses `exemplar.yaml` `config.repeats` with an llmb-run fallback of 1.
- Profiling: Controlled by exemplar suite config (no CLI profiling flag).

#### Behavior

- Runs all eligible `pretrain` workloads at **scale 512**.
- Eligibility:
  - Workload type is `pretrain`.
  - The workload supports the cluster's GPU type (from `cluster_config.yaml`).
  - The per‑dtype configuration explicitly lists `scale: 512` (implicit ranges are not used).
  - The workload is listed under `workloads.installed` in `cluster_config.yaml`.
- Enforces strict validation (install gating): if any workload that meets eligibility is not installed, the command fails.
- Repeat count and profiling are controlled by `exemplar.yaml` `config.repeats` and `config.profile`; when those keys are omitted, llmb-run falls back to 1 repeat with profiling disabled. When profiling is enabled, the last repeat is profiled and earlier repeats are non-profiled. You can override repeat count for debugging via `-r/--repeats`.

#### Troubleshooting Missing Workloads

If `llmb-run exemplar` fails due to missing workloads, do **not** use `llmb-install express`.
Instead, verify your installed workloads and add missing ones:

```bash
cd $LLMB_INSTALL
llmb-install
# Select the missing workloads from the menu
```

### Archive Command

Package all experiment results into a single compressed file for submission to NVIDIA or offline analysis.

#### Basic Usage

```bash
llmb-run archive
```

This creates a `.tar.zst` archive at `$LLMB_INSTALL/llmb-archive-<timestamp>.tar.zst` containing experiment logs and configuration files from all workloads.

#### Options

- `--output <path>`: Write the archive to a custom path instead of the default location.

#### Example

```bash
# Archive to the default location
llmb-run archive

# Archive to a specific path
llmb-run archive --output /shared/results/my-cluster-results.tar.zst
```

#### What's Included

The archive collects experiment data from `$LLMB_INSTALL/workloads/*/experiments/`, including logs and `llmb-config_*.yaml` metadata files. Profiling data, including Nsight reports and PyTorch profiler traces, is excluded to keep the archive compact — profiles are typically only needed for debugging and can be shared separately if requested.

### Job Configuration Files

When you launch a job using `llmb-run`, a `llmb-config_<JOBID>.yaml` file is automatically created in the experiment's folder. This file contains comprehensive information about the job configuration and can be useful for:

- **Job tracking**: Keep a record of all job parameters and settings
- **Reproducibility**: Recreate the exact same job configuration later
- **Debugging**: Understand what parameters were used for a specific run
- **Analysis**: Extract job metadata for performance analysis

### Config File Location

- **Nemo2 launcher**: The config file is created in the experiment's working directory (returned by the launcher)
- **Configured sbatch launcher**: The config file is created in the managed experiment directory under `$LLMB_INSTALL/workloads/<workload>/experiments/`
- **Legacy sbatch launcher**: The config file is created in the current working directory

For managed `configured_sbatch` workloads, `llmb-run` passes the experiment directory to the job as `LLMB_EXPERIMENT_DIR`. `llmb-run jobs log` and `llmb-run archive` use that directory for logs and artifacts.

### Config File Structure

The `llmb-config_<JOBID>.yaml` file contains the following sections:

```yaml
job_info:
  job_id: "3530909"                    # SLURM job ID
  user: "jdoe"                         # User who launched the job (GITLAB_USER_LOGIN/GITHUB_ACTOR/SUDO_USER/OS user...)
  launch_time: "2025-01-15T10:30:45"  # ISO timestamp of job launch

workload_info:
  framework: "nemo2"                   # Framework used (nemo2, maxtext, etc.)
  gsw_version: "25.07"                 # GSW version
  fw_version: "25.04.00"               # Framework version from container image
  workload_type: "pretrain"            # Type of workload (pretrain, finetune, etc.)
  synthetic_dataset: true              # Whether synthetic dataset is used

model_info:
  model_name: "llama3.1"               # Model name
  model_size: "405b"                   # Model size
  dtype: "fp8"                         # Data type (fp8, bf16)
  scale: 256                           # Number of GPUs
  gpu_type: "h100"                     # GPU type

cluster_info:
  cluster_name: "cluster1"             # Cluster name
  gpus_per_node: "8"                   # GPUs per node configuration
  llmb_install: "/path/to/install"     # LLMB installation path
  llmb_repo: "/path/to/repo"           # Repository path
  slurm_account: "account_name"        # SLURM account
  slurm_gpu_partition: "partition"     # SLURM partition

container_info:
  images:                              # Container images used
    - "nvcr.io#nvidia/nemo:25.11.01"

job_config:
  profile_enabled: true                # Whether profiling was enabled
  proxy: false                         # Whether this run used proxy configuration
  env_overrides:                       # Environment variable overrides
    DEBUG: "true"
  model_overrides:                     # Model parameter overrides (passed as env vars to recipes)
    tp: 8
```

See [example_llmb_config.yaml](example_llmb_config.yaml) for a complete example.

## Troubleshooting

### Common Issues and Solutions

1. **Invalid Workload/Model Size**

   ```
   ERROR: Invalid Workload / Model Size: workload_name_model_size
   ```

   - Ensure the workload and model size combination exists and is compatible with your GPU type
   - Use `llmb-run list` to see available workloads
   - Use `llmb-run list -w <workload_name>` for detailed workload information

2. **Workload Not Installed**

   ```
   ERROR: Workload 'workload_name' is not installed on this cluster.
   ```

   - Check your `cluster_config.yaml` file's `workloads.installed` list
   - Ensure the workload is properly installed and listed

3. **GPU Type Not Supported**

   ```
   ERROR: GPU type 'h100' not supported for workload 'workload_name'.
   ```

   - Check if the workload supports your cluster's GPU type
   - Use `llmb-run list -w <workload_name>` to see supported GPU types

4. **Proxy Configuration Usage**

   - Proxy configurations are altered configs that run on fewer GPUs
   - Designed for debug workflows and advanced analysis only
   - Results from proxy runs cannot be compared to production configurations
   - Do not use proxy configs for performance validation or extrapolation

5. **Missing Configuration**

   ```
   FileNotFoundError: cluster_config.yaml not found
   ```

   - Solution: Create a `cluster_config.yaml` file in your working directory
   - See the Configuration section for the required format

6. **Job Submission Fails**

   - Check your Slurm account and partition settings in `cluster_config.yaml`
   - If your system does not support GRES, make sure `SBATCH_GPUS_PER_NODE` is not in your environment section
   - Re-run with verbose output to see detailed error messages, e.g. `llmb-run -v submit ...`

## Alternative Installation Methods

These methods require additional setup and are recommended only for advanced users:

### Option 1: Install using uv (Recommended for Manual Install)

`uv` is a fast Python package manager that can install tools in isolated environments.

```bash
# Install from the project directory (assuming $LLMB_REPO is your repository root)
uv tool install $LLMB_REPO/cli/llmb-run

# Or from git
# uv tool install git+https://github.com/NVIDIA/exemplar-performance#subdirectory=cli/llmb-run
```

### Option 2: Install as a Package (pip)

```bash
# Install from the project directory
cd llmb-run
pip install .

# Note: You must:
# 1. Create cluster_config.yaml manually (see Configuration section)
# 2. Always run llmb-run from the directory containing cluster_config.yaml
```

### Option 3: Direct Execution

```bash
# Make the script executable
chmod +x llmb-run

# Run directly (must be in directory with cluster_config.yaml)
./llmb-run submit --help
```

### Option 4: Python Module

```bash
# Run as a Python module (must be in directory with cluster_config.yaml)
llmb-run submit --help
```

**Note**: These alternative methods require you to:

1. Create your own `cluster_config.yaml`
2. Install workloads manually
3. Set up any required virtual environments
4. Download container images
5. Always run llmb-run from the directory containing cluster_config.yaml

For most users, we recommend using the automated installer script described in Quick Start.

## Environment Variables Reference

The following environment variables are recognized to control behavior:

| Variable       | Purpose                                | Input                            |
| -------------- | -------------------------------------- | -------------------------------- |
| `LLMB_SKIP_PP` | Disable post-processing job submission | `1`, `true`, or `yes` to disable |

## Development

This project uses `uv` for dependency management and `tox` for multi-environment testing.

### Environment Setup

1. **Install uv**: [Follow official instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. **Sync environment**: Creates a virtualenv and installs dependencies from `uv.lock`.
   ```bash
   uv sync
   ```

### Managing Dependencies

- **Add a dependency**: `uv add <package>`
- **Add a dev dependency**: `uv add --dev <package>`
- **Update lockfile**: Run this after modifying `pyproject.toml` (including version bumps) or dependencies.
  ```bash
  uv lock
  ```

### Running Tests

- **Quick (Current Python)**:
  ```bash
  uv run pytest
  ```
- **Full Matrix (Multiple Python versions)**:
  ```bash
  # Requires tox and tox-uv
  uv tool install tox --with tox-uv
  tox
  ```

## Exit Codes

`llmb-run` uses the following exit codes for automation support:

- **0**: Success. The operation completed successfully (e.g., jobs submitted, list displayed).
- **1**: Validation Error. Invalid arguments, configuration errors (missing `cluster_config.yaml`), or validation failures. These are issues that typically require user intervention to fix.
- **2**: System Error. Unexpected failures during job submission, SLURM environment issues, or other infrastructure-related failures beyond immediate user control.

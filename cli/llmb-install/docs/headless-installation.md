# Headless Installation Guide

> **For End Users**: This guide shows how to perform automated installations for CI/CD and deployment scenarios.

The installer supports automated installations for deployment scenarios and reproducible setups.

## Installation Modes Overview

| Mode             | Automation Level | Use Case                                |
| ---------------- | ---------------- | --------------------------------------- |
| **Play Mode**    | Fully headless   | CI/CD pipelines, batch deployments      |
| **Express Mode** | Minimal prompts  | Quick repeat installations, development |
| **Interactive**  | Full prompts     | First-time setup, exploration           |

**Play Mode** is truly headless - no prompts, all settings from config file.\
**Express Mode** reuses saved settings but may prompt for missing values (install path, workload selection).

## Prerequisites

Automated modes require the installer tools installed first.

**Recommended**: Install into a dedicated virtual environment to avoid version conflicts:

```bash
# From the repository root

# Create and activate a virtual environment
python -m venv llmb-venv
source llmb-venv/bin/activate

# Install the tools with uv (recommended)
uv pip install cli/llmb-run cli/llmb-install

# Or with pip (requires Python 3.12)
pip install cli/llmb-run cli/llmb-install
```

**Note**: Using `pip` requires Python 3.12 exactly. Using `uv` works with Python 3.10-3.13 and is recommended.

For system requirements (SLURM, enroot, storage, etc.), see the [main README Prerequisites](../README.md#prerequisites).

## Play Mode (Fully Headless)

Play Mode provides complete automation by loading all settings from a configuration file:

```bash
llmb-install --play my_config.yaml
```

This performs a fully headless installation:

- No user prompts or interaction required
- All settings loaded from the configuration file
- Suitable for CI/CD pipelines and automated deployments
- Complete reproducibility across environments

### Creating Configuration Files

#### Method 1: Record Mode (Recommended)

Capture configuration interactively without installing:

```bash
llmb-install --record my_config.yaml
# Edit if needed, then:
llmb-install --play my_config.yaml
```

#### Method 2: From Existing Installation

Use `cluster_config.yaml` from a previous installation as a template:

```bash
cp $LLMB_INSTALL/cluster_config.yaml my_config.yaml
# Edit for your deployment, then:
llmb-install --play my_config.yaml
```

### Configuration File Format

Complete configuration file structure:

```yaml
venv_type: uv                      # 'uv' is recommended; 'venv'/'conda' remain for compatibility
install_path: /lustre/user/llmb    # Installation directory
slurm:
  account: myaccount
  gpu_partition: gpu
  cpu_partition: cpu
  gpu_partition_gres: 8            # GPUs per node in gpu_partition
  cpu_partition_gres: null
gpu_type: h100                     # 'vr', 'gb300', 'gb200', 'b300', 'b200', 'h100'
node_architecture: x86_64
install_method: slurm              # 'local' or 'slurm'
selected_workloads:
  - pretrain_deepseek-v3
  - pretrain_llama3.1
environment_vars:                  # Optional; exported for the whole install (as if set in your shell) and saved to cluster_config.yaml for llmb-run
  HF_TOKEN: hf_xxxxx
```

**After installation**: The installer creates `cluster_config.yaml` in `$LLMB_INSTALL/` which `llmb-run` uses for job submission.

### Important: Fresh Installations Only

Play and express modes **only work with new, empty directories**. They cannot add workloads to existing installations.

If you try to use these modes where workloads are already installed, you'll get an error:

```
Error: Existing LLMB installation found at <path>
Please choose a different installation path, or remove the existing installation.
```

**Why?** These modes are designed for reproducible, from-scratch deployments (CI/CD, multi-site rollouts).

**To add workloads to an existing installation:**

```bash
cd $LLMB_INSTALL
llmb-install  # Interactive mode recognizes existing workloads and lets you add more
```

See [Installing Additional Workloads](../README.md#installing-additional-workloads) for details.

## Express Mode (Minimal Prompts)

Express Mode reuses saved system settings from previous installations. **Not fully headless** - may prompt for install path or workload selection.

### Usage

```bash
# All options specified (no prompts)
llmb-install express /work/llmb --workloads all

# List available workloads
llmb-install express --list-workloads

# Specific workloads (comma-separated, no spaces)
llmb-install express /work/llmb --workloads pretrain_deepseek-v3,pretrain_llama3.1

# Exemplar Cloud (all pretrain workloads)
llmb-install express /work/llmb --exemplar
```

### Requirements

- Previous successful installation (creates `~/.config/llmb/system_config.yaml`)
- Reuses: SLURM settings, GPU type, environment type
- Still prompts for: install path and workloads (if not specified)

**Note**: Like play mode, express only works with new, empty directories. To add workloads to an existing installation, use interactive mode (see [Installing Additional Workloads](../README.md#installing-additional-workloads)).

### Workload Selection Options

Express mode supports three workload selection methods:

- **`--workloads all`**: Install all available workloads for your GPU type
- **`--workloads <list>`**: Install specific workloads (comma-separated, e.g., `pretrain_deepseek-v3,pretrain_llama3.1`)
- **`--exemplar`**: Install only 'pretrain' reference workloads (Exemplar Cloud suite)

### Typical Workflow

1. **First installation** (interactive): `llmb-install`
2. **Subsequent installations** (express):
   - All workloads: `llmb-install express /new/path --workloads all`
   - Exemplar Cloud only: `llmb-install express /new/path --exemplar`
   - Specific workloads: `llmb-install express /new/path --workloads pretrain_deepseek-v3,pretrain_llama3.1`

## Complete Examples

### Play Mode Workflow

```bash
# 1. Create config (interactive, no installation)
llmb-install --record prod_config.yaml

# 2. Edit config if needed
# vim prod_config.yaml

# 3. Deploy headlessly (can run in CI/CD)
llmb-install --play prod_config.yaml

# 4. Subsequent deployments
llmb-install --play prod_config.yaml  # Repeatable, identical
```

### Express Mode Workflow

```bash
# First: Interactive installation creates system config
llmb-install  # Creates ~/.config/llmb/system_config.yaml

# Subsequent: Quick repeat installations
llmb-install express /work/project1 --workloads all
llmb-install express /work/project2 --workloads pretrain_deepseek-v3,pretrain_llama3.1
llmb-install express /work/exemplar --exemplar  # Pretrain workloads only
llmb-install express /work/test --workloads test_recipe
```

### Combined Flags

```bash
# Verbose, shared images, dev mode
llmb-install -v -i /shared/containers -d express /work/dev --workloads pretrain_deepseek-v3

# Express with Exemplar Cloud
llmb-install -v -i /shared/containers express /work/llmb --exemplar

# Play mode with verbose output
llmb-install -v --play config.yaml
```

## Use Cases

| Use Case               | Recommended Mode | Why                                                    |
| ---------------------- | ---------------- | ------------------------------------------------------ |
| CI/CD pipelines        | Play Mode        | Fully automated, repeatable, version-controlled config |
| Production deployments | Play Mode        | No user interaction, audit trail via config file       |
| Multi-site deployments | Play Mode        | Same config across clusters (adjust SLURM settings)    |
| Development iterations | Express Mode     | Quick repeat installs, reuse system settings           |
| Testing new workloads  | Express Mode     | Fast workflow for recipe development                   |

## Validation and Error Handling

The installer validates configuration files before installation:

**Validates**:

- Required fields present
- Valid workload names
- Compatible GPU types
- Correct YAML syntax
- SLURM settings (account, partition accessibility)

**On error**:

- Clear error messages with field details
- Non-zero exit codes for automation
- No partial installations

## Security Considerations

**Sensitive Data**: Configuration files may contain tokens (HF_TOKEN, API keys)

- Store securely with appropriate permissions (`chmod 600`)
- Avoid committing to version control (use templating/secrets management)
- Consider environment variable substitution for secrets

**Portability**: Configuration files are cluster-specific (SLURM accounts/partitions)

- Cannot directly share configs across different clusters
- Template configs and customize SLURM settings per cluster
- Non-SLURM settings (workloads, gpu_type) are portable

## Additional Resources

- **[Main README](../README.md)**: Installation guide and prerequisites
- **[Recipe Development Guide](recipe_guide.md)**: Creating workload recipes
- **[Tools Configuration](tools.md)**: Workload-specific tool versions

# Exemplar Performance Recipes Installer

The installer is an interactive tool that simplifies the setup and deployment of Exemplar Performance recipes. It automatically discovers available workloads, configures your environment, downloads required container images, and prepares workloads for execution.

## Quick Start

### Installation

The recommended way to install llmb-install is using the automated installer script:

```bash
# Run the installer script
$LLMB_REPO/install.sh
```

This script will setup the environment and install the necessary tools.

### Manual Installation

If you prefer to install manually, you can use one of the following methods.

#### Option 1: Install using uv (Recommended)

`uv` is a fast Python package manager that can install tools in isolated environments.

```bash
# Install from the project directory (assuming $LLMB_REPO is your repository root)
uv tool install $LLMB_REPO/cli/llmb-install
```

#### Option 2: Install as a Package (pip)

It is recommended to run the installer in a virtual environment with Python 3.12.x.
The top-level installer can run from an existing uv, venv, or conda environment,
but newly-created recipe environments use `uv`.

```bash
# Install installer dependencies
cd cli/llmb-install
python3 -m pip install .

# Run the installer (simple mode)
llmb-install

# Or express mode (requires previous successful install)
llmb-install express /path/to/install --workloads all
```

The installer will guide you through an interactive setup process covering:

- Installation location selection
- SLURM cluster configuration
- Node architecture (x86_64/aarch64)
- Recipe environment setup with uv
- Installation method (local/SLURM)
- Workload selection

### Installing Additional Workloads

You can add more workloads to an existing installation at any time. The installer detects the existing installation and allows you to add new workloads incrementally.

**To add workloads**, navigate to your installation directory and run the installer:

```bash
cd $LLMB_INSTALL  # e.g., cd /work/llmb
llmb-install
```

The installer will:

1. Detect your existing installation automatically
2. Prompt you to select additional workloads to install
3. Skip already-installed dependencies (images, tools, venvs)
4. Update the configuration file to include all workloads (both old and new)

**Example workflow**:

```bash
# Initial installation
llmb-install  # Install one workload

# Later, add more workloads
cd /work/llmb  # Navigate to your LLMB_INSTALL directory
llmb-install   # Select additional workloads when prompted.
```

**Note**: You only select the new workloads you want to add. The installer automatically preserves existing workloads and their configurations.

### Express Mode

After your first successful installation, the installer saves system configuration to enable faster repeat installations:

```bash
# Express mode with all options specified
llmb-install express /work/llmb --workloads all

# Express mode with specific workloads
llmb-install express /work/llmb --workloads pretrain_nemotron-h,pretrain_llama3.1

# Express mode with prompts for missing values
llmb-install express
```

Express mode uses saved system configuration (SLURM settings, GPU type, image folder, etc.) and only prompts for:

- Installation path (if not provided)
- Workload selection (if not specified)

**Requirements**: Express mode requires a previous successful installation to save system config in `~/.config/llmb/system_config.yaml`.

## Prerequisites

### System Requirements

- **Python**: 3.12+ for the top-level installer environment. `uv` is required for recipe environments and is installed automatically by `install.sh` if missing.
- **SLURM**: 22.x or newer with job scheduler access
- **Enroot**: For container image management
- **Network Access**: Required for downloading container images
- **Disk Space**: Substantial space required (see [Storage Requirements](#storage-requirements))

### Environment Setup

Fresh recipe environments are created with `uv`. The bootstrap script still
detects an already-active uv, venv, or conda environment for the top-level
`llmb-install` and `llmb-run` tools, but the installer no longer prompts for a
recipe environment manager.

- `uv` is installed automatically by `install.sh` if missing
- Existing conda/venv recipe configs are still supported for resume, incremental,
  and headless compatibility

### Python Dependencies

The installer dependencies are defined in `pyproject.toml`:

```bash
cd cli/llmb-install
pip install .
```

This installs:

- PyYAML>=6.0
- questionary>=1.10.0
- rich>=13.0 (for enhanced UI mode)
- prompt_toolkit\<3.0.52

## Storage Requirements

The installer downloads and stores significant amounts of data:

| Component            | Size Range    | Notes                 |
| -------------------- | ------------- | --------------------- |
| Container Images     | 5-60 GB each  | Architecture-specific |
| Virtual Environments | 1-10 GB each  | Per workload          |
| Workload Datasets    | 200 GB - 1 TB | Model-dependent       |

**Recommendation**: Install on high-performance shared storage (Lustre, GPFS) with sufficient space and fast I/O.

## Directory Structure

After installation, the following structure is created:

```text
$LLMB_INSTALL/
├── .cache/          # Download caches (pip, uv, tools installers)
├── bin/             # Per-architecture uv/uvx binaries for compute-node setup
├── images/          # Container images (.sqsh files)
├── datasets/        # Dataset files
├── tools/           # Workload-specific tools (nsys, etc.)
├── venvs/           # Virtual environments
├── llmb_repo/       # Copy of repository (unless in dev mode)
└── workloads/       # Installed workloads
    └── workload_name/
        ├── setup files
        └── experiments/  # Results and logs
```

## Configuration Options

### Installation Location

- Must have sufficient disk space (hundreds of GB to TB)
- Should be on shared storage accessible to all compute nodes
- Requires write permissions

### SLURM Configuration

The installer automatically detects and validates:

- **Account**: Your SLURM accounts (via `sacctmgr`)
- **Partitions**: Available partitions (via `sinfo`)
- **GPU Resources**: GPU counts per partition (via GRES)

### Node Architecture

- **x86_64**: Standard Intel/AMD processors
- **aarch64**: ARM-based systems (Grace Blackwell, etc.)

**Important**: Choosing the wrong architecture will cause "Exec format error" when running containers.

The installer also creates `$LLMB_INSTALL/bin/<arch>/uv` and `uvx` so setup scripts can use a uv binary that matches the node they run on. The host architecture is always populated; if the selected compute-node architecture differs from the installer host, the compute architecture is downloaded from uv's GNU Linux release tarballs. Set `LLMB_UV_VERSION` before installation to choose a specific uv version; otherwise the installer snapshots the active `uv --version`.

Compute-side scripts that need uv should resolve it at runtime:

```bash
export LLMB_BIN="${LLMB_BIN:-$LLMB_INSTALL/bin/$(uname -m)}"
"$LLMB_BIN/uv" run ...
```

### Installation Method

Which method to use to download the large container images and datasets. Workload specific setup and venv installation will always be on the current node.

- **Local**: Downloads run on current machine (requires enroot access)
- **SLURM**: Downloads submitted as jobs (recommended for clusters)
  - **Note:** Currently this is sequential srun jobs.
  - **Important:** SLURM installation method is not available when running the installer within a SLURM job

**SLURM Job Detection**: The installer automatically detects if it's running within a SLURM job (via `SLURM_JOB_ID` environment variable). When detected:

- SLURM installation method is disabled (cannot submit jobs from within a job)
- Automatically defaults to local installation method if enroot is available
- Exits with error if enroot is not available

## Common Issues and Solutions

### Running Installer Within SLURM Job

**Issue**: Installer fails when run within a SLURM job without enroot

```
Error: Cannot proceed with installation.
You are running within a SLURM job, but enroot is not available on this system.
```

**Solution**:

- **Option 1**: Run the installer from a login node (outside SLURM job)
- **Option 2**: Ensure enroot is available on compute nodes and use local installation method

### Installation Process is Slow/Resource Intensive

**Issue**: The installer seems very slow or stalled.

**Explanation**: The installation process, especially downloading large container images and installing all necessary pip packages, can be resource-intensive. Login nodes are often shared and with limited resources per user, which can lead to slow performance.

**Solution**:

- **Option 1**: Try running the installer again, perhaps during off-peak hours.
- **Option 2**: Obtain an interactive shell on a dedicated CPU node and run the installer there. This offloads the resource usage from the login node.
- **Option 3**: On hosts with ample memory, speed up HuggingFace downloads by raising `LLMB_HF_MAX_WORKERS` / `HF_XET_FIXED_DOWNLOAD_CONCURRENCY` (see [HuggingFace Downloads Killed / Out of Memory](#huggingface-downloads-killed--out-of-memory)).

### Package Installation Resource Limit Errors

**Issue**: Package installation fails because the host or job has tight resource limits. For example, a low open-file limit may produce `Too many open files` or `OSError: [Errno 24]`.

**Solution**: Reduce uv's package-install concurrency and re-run the installer:

```bash
UV_CONCURRENT_DOWNLOADS=4 \
UV_CONCURRENT_INSTALLS=4 \
UV_CONCURRENT_BUILDS=1 \
llmb-install
```

If the error continues, set all three values to `1`. For open-file errors, check the current shell limits with `ulimit -Sn` and `ulimit -Hn`. For job, cgroup, or system limits that you cannot change, contact your system administrator.

As a last resort, set `LLMB_USE_PIP_FALLBACK=1` to use standard pip. Standard pip is typically significantly slower for LLMB installations, so prefer limiting uv concurrency first.

### Enroot Not Available for Local Installation

**Issue**: Installer automatically selects SLURM method when enroot is missing

```
Note: enroot is not available on this system.
Local installation requires enroot for container image downloading.
Automatically selecting SLURM-based installation.
```

**Solution**:

- **Option 1**: Install enroot on the current system to enable local installation
- **Option 2**: Continue with SLURM installation method (recommended for clusters)
- **Option 3**: Manually download container images using enroot on a different system

### Python Version Compatibility

**Issue**: uv is not available

```text
Error: uv is required to create recipe environments.
```

**Solution**:

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Re-run `./install.sh` so it can install or pin the supported uv version.

### Cache Directory Warnings

**Issue**: Cache directories under `/home` may cause space issues

```
WARNING: PIP_CACHE_DIR is under /home: /home/user/.cache/pip
WARNING: UV_CACHE_DIR is under /home: /home/user/.cache/uv
```

**Solution**: The installer automatically configures cache directories under `$LLMB_INSTALL/.cache/` to avoid space issues. If you see these warnings, the installer has already handled the configuration for you.

### SLURM Account/Partition Issues

**Issue**: Account or partition not recognized
**Solution**:

- Verify with `squeue -u $USER` or `sacctmgr show associations user=$USER`
- Contact system administrator for correct account/partition names

### Network Access Issues

**Issue**: Container downloads fail
**Solution**:

- Ensure login nodes have internet access and enroot OR
- Use SLURM installation method to run downloads on nodes with access

### Insufficient Disk Space

**Issue**: Downloads fail due to space constraints
**Solution**:

- Choose installation location with adequate space
- Clean up unnecessary files in target directory
- Consider using storage with higher quotas

### HuggingFace Downloads Killed / Out of Memory

**Issue**: Model or dataset downloads abort partway — either a Rust `memory allocation of N bytes failed` error, or the process is silently killed and drops back to the shell. This happens on hosts with a tight per-job or per-user memory limit (`ulimit -v`, or a cgroup `memory.max` such as a systemd login-slice), which `ulimit` alone may not reveal.

**Explanation**: HuggingFace's Xet downloader scales memory with the number of concurrent download streams. To stay under tight caps, the installer pins conservative defaults: `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=1` (per-file streams) and 4 concurrent file downloads (~2.5 GB peak). Peak memory roughly scales with `files × per-file streams`.

**Solution**: The defaults work out of the box. On memory-rich hosts you can raise them for faster downloads (export before running the installer):

| Variable                            | Default | Effect                                                     |
| ----------------------------------- | ------- | ---------------------------------------------------------- |
| `LLMB_HF_MAX_WORKERS`               | `4`     | Number of files downloaded concurrently.                   |
| `HF_XET_FIXED_DOWNLOAD_CONCURRENCY` | `1`     | Per-file Xet download streams (read directly by `hf-xet`). |

Requires `hf-xet >= 1.3.0` (installed automatically). If a host's memory limit is below ~2.5 GB, the download may still be killed — run the installer on a node with more memory.

## Command Line Reference

The installer provides both interactive and command-line options:

```bash
# Interactive modes
llmb-install                    # Simple UI (default)

# Express mode (non-interactive with saved config)
llmb-install express /path/to/install --workloads all
llmb-install express --install-path /path --workloads workload1,workload2

# List available workloads
llmb-install express --list-workloads

# Automated/headless mode
llmb-install --play config.yaml

# Help and version
llmb-install --help
llmb-install express --help
```

## Command-Line Flags Reference

### Global Flags (All Modes)

| Flag                      | Purpose                                     | Example                              |
| ------------------------- | ------------------------------------------- | ------------------------------------ |
| `-v, --verbose`           | Enable debug logging                        | `llmb-install -v`                    |
| `-i, --image-folder PATH` | Share container images across installations | `llmb-install -i /shared/containers` |
| `-d, --dev-mode`          | Use original repo (for recipe development)  | `llmb-install -d`                    |
| `--record FILE`           | Save configuration without installing       | `llmb-install --record config.yaml`  |
| `--play FILE`             | Automated installation from config          | `llmb-install --play config.yaml`    |

**Note on image folder**:

- **Purpose**: Highly recommended for multi-user or multi-installation setups. Container images are 5-60 GB each and read-only, so sharing saves significant space with no downsides.
- **Requirement**: You need write access to the image folder.
- **Persistence**: The image folder path is saved to `~/.config/llmb/system_config.yaml` after successful installation and automatically reused in future installs.
- **Override**: Use `-i` flag to override the saved location for a specific installation, or for first time installs.

### Express Mode Flags

| Flag                        | Purpose                                                     | Example                                                                  |
| --------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| `install_path` (positional) | Installation directory                                      | `llmb-install express /work/llmb`                                        |
| `-w, --workloads`           | Workloads to install                                        | `--workloads all` or `--workloads pretrain_nemotron-h,pretrain_llama3.1` |
| `--exemplar`                | Install all 'pretrain' reference workloads (Exemplar Cloud) | `llmb-install express --exemplar`                                        |
| `--list-workloads`          | Show available workloads and exit                           | `llmb-install express --list-workloads`                                  |

**Note on flag order**: `-v/--verbose`, `-i/--image-folder`, and `-d/--dev-mode` are **global flags** and must be provided **before** the `express` subcommand (e.g. `llmb-install -v express ...`).

**Combined example**:

```bash
llmb-install -v -i /shared/containers express /work/llmb --workloads all
```

## Advanced Usage

### Development Mode

When developing or testing recipes, use `--dev-mode` to work directly with the original repository:

```bash
llmb-install -d express /work/llmb --workloads test_workload
# OR for interactive mode with streamlined selection:
llmb-install -d
```

Dev mode features:

- **Direct repo access**: Uses the original repo location (no copy to `LLMB_INSTALL/llmb_repo`), allowing git operations and version control
- **Streamlined workflow**: In interactive mode, skips the "Exemplar Cloud vs Custom" prompt and goes straight to workload selection
- **No resume support**: Resume functionality disabled in dev mode

Not recommended for production installations.

### Resuming Failed Installations

If an installation fails, simply run the installer again. It will detect the interrupted installation and offer to resume with remaining workloads.

Resume state expires after 7 days. Not available in headless/play mode or dev mode.

To disable: `export LLMB_DISABLE_RESUME=1`

### Debugging Failed Installations

**Check**:

1. Installer error messages
2. Container images: `$LLMB_INSTALL/images/`
3. Virtual environments: `$LLMB_INSTALL/venvs/`
4. System config: `~/.config/llmb/system_config.yaml`

**Manual container import** (if automatic download fails):

```bash
enroot import -o $LLMB_INSTALL/images/nvidian+nemo+25.02.01.sqsh \
    docker://nvcr.io/nvidian/nemo:25.02.01
```

Use `-a <arch>` flag if node architecture differs.

## Validation

After installation, verify setup:

1. **Check directory structure**:

   ```bash
   ls -la $LLMB_INSTALL/
   # Should show: images/ datasets/ venvs/ workloads/
   ```

2. **Verify container images**:

   ```bash
   ls -la $LLMB_INSTALL/images/*.sqsh
   # Should show downloaded container files
   ```

3. **Test virtual environments**:

   ```bash
   source $LLMB_INSTALL/venvs/<workload>_venv/bin/activate
   python --version  # Should show a 3.12.x version
   ```

## Environment Variables Reference

The installer recognizes the following environment variables to control behavior:

| Variable                      | Purpose                                                   | Input                            |
| ----------------------------- | --------------------------------------------------------- | -------------------------------- |
| `LLMB_DISABLE_RESUME`         | Disable resume functionality                              | `1`, `true`, or `yes` to disable |
| `LLMB_DISABLE_GIT_CACHE`      | Disable git repository caching                            | `1`, `true`, or `yes` to disable |
| `LLMB_USE_PIP_FALLBACK`       | Use standard pip instead of uv pip in uv environments     | `1`, `true`, or `yes` to enable  |
| `LLMB_DISABLE_MANAGED_PYTHON` | Disable enforced managed python usage for UV environments | `1`, `true`, or `yes` to disable |

**Resume Control**: Set `LLMB_DISABLE_RESUME=1` to prevent automatic resume detection and always start fresh installations.

**Git Caching**: Set `LLMB_DISABLE_GIT_CACHE=1` to skip local git cache and clone repositories directly from remote sources.

**Pip Fallback**: As a last resort, set `LLMB_USE_PIP_FALLBACK=1` to use standard pip instead of uv pip when using uv environments. Standard pip is typically significantly slower for LLMB installations; for resource-limit errors, reduce uv concurrency as described in [Package Installation Resource Limit Errors](#package-installation-resource-limit-errors) instead.

**Managed Python**: By default, UV environments use managed python versions for consistency. Set `LLMB_DISABLE_MANAGED_PYTHON=1` to use system python instead if available.

## Development

This project uses `uv` for dependency management and `tox` for multi-environment testing.

### Environment Setup

1. **Install uv**: [Follow official instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. **Sync environment**: Creates a virtualenv and installs runtime plus development dependencies from `uv.lock`.
   ```bash
   uv sync --extra dev
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
  uv run --extra dev pytest
  ```
- **Full Matrix (Multiple Python versions)**:
  ```bash
  # Requires tox and tox-uv
  uv tool install tox --with tox-uv
  tox
  ```

## Documentation

### End-User Guides

- **[Headless Installation](docs/headless-installation.md)**: Automated deployments and CI/CD integration

### Recipe Developer Documentation

- **[Recipe Development Guide](docs/recipe_guide.md)**: Complete guide to creating workload recipes and metadata.yaml
- **[Tools Configuration](docs/tools.md)**: Configuring workload-specific tools with GPU-conditional versions

## Support

For installation issues:

1. Check this README and [main FAQ](../../README.md#faq)
2. Verify system prerequisites are met
3. Contact LLMBenchmarks@nvidia.com with:
   - Installer output/error messages
   - System configuration details
   - SLURM cluster information

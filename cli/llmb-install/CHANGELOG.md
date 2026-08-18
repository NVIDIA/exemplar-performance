# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [PEP 440](https://www.python.org/dev/peps/pep-0440/) versioning with semantic versioning semantics:
**MAJOR.MINOR.PATCH** for feature parity with [SemVer](https://semver.org/).

## [1.13.3] - 2026-07-09

### Changed

- Modernized Python dependencies.

## [1.13.2] - 2026-07-06

### Fixed

- A failed resume, edit-resume, or completion action no longer reports success to the calling `install.sh`. These paths previously caught the error and returned success; they now propagate it so the installer exits with a nonzero status.

## [1.13.1] - 2026-06-22

### Fixed

- HuggingFace (Xet) downloads getting OOM-killed on hosts with tight per-job/per-user memory limits (`ulimit -v` or cgroup `memory.max`). The installer now bounds peak download memory by pinning `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=1` and limiting concurrent file downloads to 4; both are overridable (`HF_XET_FIXED_DOWNLOAD_CONCURRENCY`, `LLMB_HF_MAX_WORKERS`) on memory-rich hosts that want faster downloads.

## [1.13.0] - 2026-06-10

### Changed

- Replaced the unreleased `downloads.huggingface[].snapshot` schema with an explicit split: `downloads.huggingface.cache` for shared-cache assets and `downloads.huggingface.repos` for repo-contents downloads into LLMB-managed directories. The released bare-list form of `downloads.huggingface` remains supported as shorthand for `cache` entries. Repo entries require `repo_type`, use `target: workload|shared`, and support entry-level `when.gpu` filtering.

## [1.12.0] - 2026-06-09

### Added

- Auto-detect GPU node CPU architecture from SLURM node metadata when available, falling back to the existing manual architecture prompt when detection is unavailable, mixed, or unsupported.

### Fixed

- Setup task subprocesses now reliably inherit `PIP_CACHE_DIR` and `UV_CACHE_DIR` on every install path (incremental, headless, and edit-resume previously missed
  them), so `uv`/`pip` no longer fall back to caching under the user's home directory.
- Pressing Ctrl+C at the install method prompt now exits the installer instead of continuing to the next prompt.
- Setup tasks now run from the installed copy of the repo (`$LLMB_INSTALL/llmb_repo/<workload>`) instead of the original git clone.
- `sbatch` setup tasks now receive an explicit `--output` flag directing logs to `$LLMB_INSTALL/workloads/<workload>/`, overriding the SLURM default of writing to the submission directory.
- `sbatch` setup tasks no longer propagate the login-node virtual environment to compute nodes when the login and compute CPU architectures differ.

## [1.11.0] - 2026-06-08

### Added

- Support HuggingFace `downloads.huggingface[].snapshot` entries to download model or dataset repositories into LLMB-managed workload or dataset directories.

## [1.10.0] - 2026-06-03

### Added

- Install per-architecture `uv`/`uvx` binaries under `$LLMB_INSTALL/bin/<arch>` for compute-node setup on mixed-architecture clusters.

## [1.9.2] - 2026-06-02

### Added

- Boolean field `submodules` in `repositories`; when set to `true`, runs `git submodule update --init --recursive` after cloning.

## [1.9.1] - 2026-05-04

### Fixed

- Accept trillion-parameter (`t`) model-size suffixes when selecting Exemplar Cloud workloads from `exemplar.yaml`.

## [1.9.0] - 2026-04-24

### Changed

- Fresh interactive and express installs now create recipe environments with `uv` without prompting for an environment type; existing conda/venv configs remain supported for resume, incremental, and headless compatibility.

### Removed

- Removed unsupported legacy `setup_script` installation support; recipes should use `setup.tasks`.

### Fixed

- Workloads without Python dependencies no longer run through the legacy scripted-install path or emit deprecation warnings.

## [1.8.7] - 2026-04-01

### Changed

- Migrated config models (`InstallConfig`, `SlurmConfig`, `SystemConfig`) from dataclasses to pydantic v2; field types are now enforced at construction (e.g. `Literal` for `venv_type`, `gpu_type` validated against `SUPPORTED_GPU_TYPES`).

## [1.8.6] - 2026-03-31

### Fixed

- Resume from failed install now preserves venv paths for previously completed workloads in the state file; repeated failures no longer produce a `cluster_config.yaml` with missing venv entries.
- venv creation errors now include full stderr output alongside the failed command; previously only the exit code was shown.
- uv venv creation now removes existing directory before creating instead of relying on `uv venv --clear`, which can fail in edge-cases.
- GRES auto-detection now uses sinfo node counts (`%D`) to pick the GPU count backed by the most nodes; previously heterogeneous partitions could randomly select the minority value.

## [1.8.5] - 2026-03-24

### Fixed

- `InstallConfig` now raises on null `environment_vars` values and coerces remaining values to strings; prevents YAML nulls from leaking into subprocess env dicts.
- Top-level error handler now includes exception type and hints at `--verbose` for full traceback.

## [1.8.4] - 2026-03-23

### Fixed

- Fix v1→v2 cluster config migration dropping GRES values due to wrong field names (`gpu_gres`/`cpu_gres` → `gpu_partition_gres`/`cpu_partition_gres`).

## [1.8.3] - 2026-02-27

### Changed

- Conda environment option now shows `(deprecated)` label in the environment selection prompt.

### Fixed

- User-initiated exits now consistently return code `130` (prompt cancel/exit choices), so `install.sh` no longer shows a misleading `SUCCESS` summary on cancellation.
- Express mode now exits with code `1` when pointed at an existing installation path.

## [1.8.2] - 2026-02-27

### Fixed

- Improved wrapped installer UX with structured post-install summary handoff and clearer end-of-run messaging.
- Treat Ctrl-C as exit code 130 so wrapper scripts can distinguish cancellation and avoid printing redundant summaries.

## [1.8.1] - 2026-02-25

### Fixed

- Corrected SLURM GRES detection for comma-separated partition lists and fail fast on incompatible partition GRES combinations.

## [1.8.0] - 2026-02-18

### Changed

- `cluster_config.yaml` is now written in schema v2, requires llmb-run >= 1.10.0

## [1.7.0] - 2026-02-09

### Changed

- Clone-only git dependencies (`install_method.type: clone`) excluded from venv sharing hash (these dependencies are outside venv); workloads differing only in clone-only commits now share a venv
- Git cloning is now per-workload, allowing different clone-only commits within a shared venv group

### Fixed

- Write partial `cluster_config.yaml` when some workloads succeed but others fail
- Resume completion now includes all workloads (prior + current) in final config instead of only remaining

## [1.6.3] - 2026-02-05

### Fixed

- Headless mode now shows "workload not supported for gpu_type" instead of generic "not found" error
- Record mode (`--record`) no longer copies the repository or creates cache directories

## [1.6.2] - 2026-02-02

### Fixed

- Constrained transformers dependency to >4.57.3,\<5.0.0 (backported to 25.12 maintenance branch as 1.6.0.post1)

## [1.6.1] - 2026-01-30

### Added

- Comma-separated partition list support for SLURM configuration (e.g., `gpu1,gpu2`)

### Changed

- Improved incremental install resume prompt wording
- SLURM partition/account validation now strictly enforces valid values (fail-fast)

### Fixed

- Installation path prompt now requires absolute paths and rejects empty/confusing inputs

## [1.6.0] - 2026-01-16

### Added

- HuggingFace downloads v2 via `downloads.huggingface` (tokenizer + `config.json`, supports remote-code tokenizers).

### Changed

- Enforce `transformers>4.57.3`.

## [1.5.2] - 2026-01-20

### Fixed

- Headless playfiles now use top-level `slurm` and `environment_vars` (and fail fast on deprecated keys).

## [1.5.1] - 2026-01-09

### Added

- B300 support
- GPU ordering in GPU selection prompt

## [1.5.0] - 2025-12-24

### Changed

- Exemplar Cloud selection now uses `exemplar.yaml` per GPU type instead of selecting all pretrain workloads

## [1.4.2] - 2025-12-18

### Fixed

- Missing `urllib.error` import for download error handling in tools module
- Improved validation error messages for null/invalid `downloads` metadata

## [1.4.1] - 2025-12-17

### Fixed

- HuggingFace tokenizer downloads now uses hf login, instead of relying on token passed to individual functions.

## [1.4.0] - 2025-12-15

### Added

- HuggingFace tokenizer download support via `hf_tokenizers` in metadata.yaml

### Changed

- Refactored download code into unified `downloads/` module

## [1.3.13] - 2025-12-10

### Added

- "Exemplar Cloud" option to workload selection, which automatically selects all standard pre-training reference workloads.
- Added `--exemplar` flag to `llmb-install express` for automated installation of Exemplar Cloud workloads.

## [1.3.12] - 2025-12-10

### Added

- `setup` configuration in `metadata.yaml` now supports `by_gpu` conditional logic.

## [1.3.11] - 2025-12-05

### Changed

- Migrated dependency management to uv and added lockfiles.

## [1.3.10] - 2025-12-03

### Changed

- Enforce managed python version for UV environments by default
- Added `LLMB_DISABLE_MANAGED_PYTHON` env var to disable managed python enforcement

## [1.3.9] - 2025-12-03

### Fixed

- Minor fixes for import paths, documentation, and archive extraction robustness.

## [1.3.8] - 2025-11-07

### Added

- Copy `release.yaml` to `$LLMB_INSTALL/llmb_repo` if it exists.

## [1.3.7] - 2025-10-31

### Added

- `LLMB_USE_PIP_FALLBACK` environment variable to use standard pip instead of uv pip in uv environments

## [1.3.6] - 2025-10-29

### Fixed

- Git cache now uses `--mirror` instead of `--bare` to fetch all refs on updates
- Added commit validation after cache/target fetch operations with clear error messages

## [1.3.5] - 2025-10-29

### Fixed

- UV environment detection for conda-based virtual environments by setting `CONDA_DEFAULT_ENV`

## [1.3.4] - 2025-10-21

### Changed

- `GPU_TYPE` no longer saved to `cluster_config.yaml` environment section (still passed to setup scripts)

### Deprecated

- Legacy `setup_script` functionality (both scripted workloads and post-install scripts). Users should migrate to `tasks` feature in metadata.yaml

### Fixed

- Post-install scripts now receive `LLMB_WORKLOAD` environment variable

## [1.3.3] - 2025-10-21

### Fixed

- Repository path corruption when editing configuration in interactive mode

## [1.3.2] - 2025-10-17

### Added

- `cuda_cupti_lib` tool support for downloading CUDA CUPTI libraries as nsys profiling workaround

## [1.3.1] - 2025-10-13

### Added

- Git LFS installation verification and automatic configuration

### Fixed

- Cache clone configuration when Git LFS is required

## [1.3.0] - 2025-10-13

### Added

- **Tools Download**: Support for downloading and installing workload-specific / GPU-conditional Nsight Systems versions
  - Tools specified in `metadata.yaml` with simple or `by_gpu` conditional formats
  - Installer caching in `$LLMB_INSTALL/.cache/tools/` for bandwidth efficiency
  - Robust download/install logic with proper cleanup on failures

## [1.2.0] - 2025-10-08

### Added

- **Incremental Install**: Add workloads to existing installations without reinstalling everything
  - Run `llmb-install` from installation directory or provide existing path when prompted
  - Automatically reuses virtual environments when dependencies match
  - Inherits existing configuration (GPU type, SLURM settings, environment variables)

### Changed

- `cluster_config.yaml` includes new `install` section with installation metadata
- Resume workflow now preserves original workloads during incremental install failures

## [1.1.3] - 2025-10-03

### Fixed

- Correctly set repo root when opting for a fresh install during a resume operation.

## [1.1.2] - 2025-10-02

### Changed

- Updated venv naming for non-legacy workloads to 'venv\_{shortsha}'

## [1.1.1] - 2025-10-02

### Fixed

- SLURM partition validation now disallows blank GPU and CPU partitions in all modes
- Install paths now converted to absolute paths using pathlib in headless and express modes

### Changed

- `llmb-run` symlink now points to venv binary instead of repository script

## [1.1.0] - 2025-09-30

### Added

- **Resume Failed Installations**: Automatically save progress and resume from interruptions
- **Development Mode** (`--dev-mode`): Skip repository copying for recipe development
- **Repository Isolation**: Copy repository to installation directory for standalone operation

### Changed

- Installation state for incomplete installs tracked in `~/.config/llmb/install_state.yaml`
- Repository copied to `$LLMB_INSTALL/llmb_repo` (unless dev mode enabled)

## [1.0.4] - 2025-09-24

### Fixed

- Use the '--clear' flag when creating uv venvs. Brings in line with other methods and fixes a failure case.
- Handle an edge case for git workload clones with missing remotes.

## [1.0.3] - 2025-09-24

### Added

- GB300 support

## [1.0.2] - 2025-09-23

### Added

- Caching support for git clones, with local clones from cache. This transfers large pack files and avoids many small fs ops.

### Fixed

- Git repo validation bug, we now check if an existing repo is valid before doing any git operations.

## [1.0.1] - 2025-09-18

### Changed

- Only compile bytecode for the last package when using uv, as it compiles the full venv.

## [1.0.0] - 2025-09-17

Refactor of code base into modular design.

### Added

- **Express Mode**: `llmb-install express` for faster repeat installations using saved settings
- **Configuration Persistence**: System settings automatically saved for future use

### Changed

- **Command Interface**: Replaced `./installer.py` with `llmb-install` command
- **Configuration Management**: XDG-compliant config location (`~/.config/llmb/`)

## [0.11.2] - 2025-08-28

### Changed

- Moved installer to 'cli/llmb-install'
- Updated repo root detection logic

## [0.11.1] - 2025-08-27

### Fixed

- Downgraded 'prompt_toolkit\<3.0.52' to work around questionary incompatibility

## [0.11.0] - 2025-08-26

### Added

- UV environment support with automatic preference order: uv → venv → conda
- Unified cache management for both PIP and UV cache directories

### Changed

- Cache directories automatically configured under `$LLMB_INSTALL/.cache/`, if not set in user environment.
- venv creation now ensures it's not using a virtual environment. Usually system Python.

### Fixed

- System Python detection bug when running from virtual environments

## [0.10.2] - 2025-08-18

### Fixed

- Force conda auto_activate to false. Handles an edge case on systems where login and compute nodes have different architectures.

## [0.10.1] - 2025-08-14

### Fixed

- Force conda environment usage when installer is run from within a conda environment to prevent venv dependency issues on parent environment.

## [0.10.0] - 2025-08-11

### Added

- GPU-specific overrides in metadata:
  - `container.images` supports `by_gpu` with `h100`, `b200`, `gb200`, and `default`.
  - `repositories` supports top-level `by_gpu` to select per-GPU repo commits.
- New installer resolver `resolve_gpu_overrides` to materialize GPU-specific images/repos before install.
- Schema extended to allow future `by_model` conditionals for images and repositories (not yet wired in installer).

### Changed

- Venv grouping now hashes a canonical dependencies signature that ignores non-functional differences (e.g., script vs pip-git acquisition, wrappers), so identical effective commits share a venv.
- Editable pip dependencies force an isolated venv per recipe to avoid cross-recipe interference.

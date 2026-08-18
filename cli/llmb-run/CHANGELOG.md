# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.16.2] - 2026-08-11

### Added

- Set `PROXY_WORKLOAD=true` in the environment when launching proxy workloads via `Nemo2Launcher`.

## [1.16.1] - 2026-07-23

### Fixed

- Exclude generated CPU tokenization benchmark datasets from `llmb-run archive`.

## [1.16.0] - 2026-07-14

### Added

- Add repeatable `llmb-run submit --mbridge-arg` passthrough for Megatron-Bridge workload arguments.
- Add `user` field to `job_info` in `llmb-config_<JOBID>.yaml`, resolved from `GITLAB_USER_LOGIN`, `GITHUB_ACTOR`, `SUDO_USER`, or the OS user.

## [1.15.2] - 2026-07-10

### Changed

- Update Python dependencies.

## [1.15.1] - 2026-07-10

### Fixed

- Time bound and batch `sacct` status queries to prevent `llmb-run jobs` timeouts.

## [1.15.0] - 2026-06-11

### Added

- NeMo-RL (GRPO) performance metrics in job history: `jobs list` and `jobs show` report `s/iter` and `Tokens/s/GPU` for RL workloads, parsed from the experiment's `metrics.json`.

### Fixed

- Remove WiT submission restrictions now that it's significantly faster.

## [1.14.6] - 2026-06-06

### Added

- Nsight Systems mount mapping for NeMo `26.04.01` container images.

## [1.14.5] - 2026-06-04

### Changed

- `llmb-run archive` default output filenames now include the recipe release version from `release.yaml`.

### Fixed

- Reject blank `llmb_repo` values during configuration loading so commands fail fast with a configuration error.

## [1.14.4] - 2026-05-15

### Changed

- `llmb-run exemplar` now falls back to one non-profiled repeat when `config.repeats` and `config.profile` are omitted.

## [1.14.3] - 2026-05-13

### Added

- `llmb-run --version` now also prints the recipe version from the configured install's `release.yaml` when available.

## [1.14.2] - 2026-05-12

### Fixed

- Exclude `torch_profile/` PyTorch profiler output directories from `llmb-run archive`.

## [1.14.1] - 2026-05-08

### Fixed

- Require supported pretraining job logs to reach their reported final iteration before showing parsed performance metrics.

## [1.14.0] - 2026-05-05

### Added

- `llmb-run jobs`: local SQLite-backed job history with submission recording, Slurm status refresh, performance results for supported workload logs, detail view, launcher-aware log access, and rebuild from existing non-legacy `llmb-config_*.yaml` files.

## [1.13.1] - 2026-05-04

### Fixed

- Accept trillion-parameter (`t`) model-size suffixes in workload parsing, bulk YAML headers, and Exemplar ordering.

## [1.13.0] - 2026-04-21

### Added

- `llmb-run submit --dump-env` for Megatron-Bridge workloads, capturing a redacted rank-0 environment snapshot.

### Removed

- Removed deprecated `llmb-run single`, `llmb-run bulk`, and `llmb-run submit-all` commands. Use `llmb-run submit` for explicit, file-based, and discovery submissions.

## [1.12.1] - 2026-04-21

### Fixed

- Archive now excludes `nsys_profile/` directories to avoid packaging large profiling artifacts.

## [1.12.0] - 2026-04-21

### Added

- `llmb-run submit`: repeatable `--env KEY=value` flag for explicit job environment overrides. YAML task-spec `env:` blocks now receive the same treatment.

### Fixed

- Archive now excludes `*.pt.trace.json` files produced by newer PyTorch profiling output.
- Archive now excludes experiment-level `checkpoints/` directories, preventing Megatron-Bridge `*.distcp` checkpoint shards from bloating archives.

## [1.11.0] - 2026-04-09

### Added

- `llmb-run submit`: first-class Slurm submission flags `--nodelist`, `--exclude`, `--reservation`, `--segment`, `--nice`, and repeatable `--slurm-arg`.

## [1.10.9] - 2026-04-06

### Added

- `llmb-run submit`: `--force` flag to bypass dtype/scale validation for one explicit task.

### Fixed

- `llmb-run submit`: reject conflicting `-w workload_size -s different_size` with a clear error instead of a cryptic "workload not found" message. Matching suffixes (e.g., `-w pretrain_foo_7b -s 7b`) are now normalized automatically.

## [1.10.8] - 2026-03-31

### Added

- Nsys and CUPTI mount paths for nemo containers `25.11.01`, `26.02.00`, and `26.02.01`.

## [1.10.7] - 2026-03-11

### Changed

- Archive now excludes `*_trace.json` files (PyTorch profiler traces, e.g. torchtitan workloads).

## [1.10.6] - 2026-03-06

### Fixed

- Set str default for json normalization in exp_id calculation.
- Downgrade "No container images resolved" log from warning to debug.
- Archive now includes top-level `llmb-config_*.yaml` and `slurm-*.out` files for `microbenchmark_nccl` workload.

## [1.10.5] - 2026-03-05

### Fixed

- Exclude `microbenchmark` workload type from post-processing pipeline submission.
- All launchers now use `run.launch_script` from workload metadata instead of hardcoding `launch.sh`.

## [1.10.4] - 2026-02-25

### Changed

- `llmb-run exemplar` now treats `config.profile: true` as one profiled run per workload config, placed on the last repeat (e.g., default `repeats: 3` yields 2 normal + 1 profiled).

## [1.10.3] - 2026-02-23

### Fixed

- `ConfiguredSbatchLauncher` now also passes `--segment=<value>` as an sbatch CLI flag for backward compatibility with older Slurm versions where `SBATCH_SEGMENT_SIZE` did not exist.

## [1.10.2] - 2026-02-23

### Added

- `ConfiguredSbatchLauncher` now sets `SBATCH_SEGMENT_SIZE` automatically for `gb200`/`gb300` jobs, using the greatest divisor of the node count that fits within the 18-node NVLink domain. Overridable via OS env or `env_overrides`.

## [1.10.1] - 2026-02-19

### Added

- Background freshness check for the `gsw-common` container image: once per day, queries the registry and prompts the user to download a newer version via `srun` if one is available.

## [1.10.0] - 2026-02-17

### Changed

- Reworked Slurm configuration handling to support shared defaults plus explicit `slurm.gpu` and `slurm.cpu` target blocks, with normalized SBATCH/SLURM key mapping and legacy schema compatibility checks.
- Refactored cluster config loading to use typed top-level cluster fields while preserving backward compatibility for existing configs.

## [1.9.1] - 2026-02-13

### Added

- `--version` flag: print installed version and exit without loading cluster config.

### Changed

- `model_overrides` (YAML `params:`) are now passed as uppercased env vars (e.g. `TP`, `CP`) instead of the deprecated `OPTIMIZATION_CODE`/`OPTIMIZATION_NAME` mechanism.

### Fixed

- Workload inspector jobs now correctly apply task-level `env_overrides` (e.g. `PROFILE_START_STEP`, `PROFILE_STOP_STEP`) instead of duplicating config-level environment variables.

## [1.9.0] - 2026-02-09

### Added

- `llmb-run archive`: package workload experiment logs (`workloads/*/experiments/`) into a single `.tar.zst` archive.

## [1.8.1] - 2026-02-05

### Changed

- Update ordering of post processing pipeline jobs.

## [1.8.0] - 2026-02-04

### Added

- `--proxy` flag and `proxy_scales` metadata field for debug and advanced workflow configurations.

## [1.7.0] - 2026-01-28

### Added

- `configured_sbatch` launcher type: creates experiment directories before job submission, enabling llmb-config.yaml and post-processing for sbatch workloads.
- B300 support

## [1.6.0] - 2026-01-20

### Added

- `llmb-run submit --exact-scales` flag: limit job generation to scales explicitly listed in workload metadata (no power-of-2 scale expansion beyond metadata).

## [1.5.5] - 2026-01-15

### Fixed

- Allow `inference` and `microbenchmark` workloads to be run when explicitly selected via `llmb-run submit -w <workload>`, while keeping them excluded from discovery/bulk modes.

## [1.5.4] - 2026-01-07

### Fixed

- Hardened post processing job submission edge cases.

## [1.5.3] - 2026-01-05

### Fixed

- Improved YAML header parsing with better validation and error messages

## [1.5.2] - 2025-12-31

### Fixed

- Zero-jobs error now shows available configurations instead of generic message

## [1.5.1] - 2025-12-24

### Fixed

- Deprecated `single` command: AttributeError when accessing extra_slurm_params

## [1.5.0] - 2025-12-23

### Added

- `exemplar` command for cloud certification

## [1.4.0] - 2025-12-22

### Added

- `llmb-run submit`: unified job submission command (single, discovery, and file-based).

### Deprecated

- `llmb-run single`, `llmb-run submit-all`, `llmb-run bulk` (use `llmb-run submit`).

### Changed

- **[BREAKING INTERNAL]** Migrated CLI framework from argparse to Typer (intended to be user-facing compatible).
- Reduced default verbosity of `llmb-run submit` output (use `-v` for discovery/details).

## [1.3.10] - 2025-12-05

### Changed

- Migrated dependency management to uv and added lockfiles.

## [1.3.9] - 2025-11-19

### Changed

- Refactored post-processing steps.

## [1.3.8] - 2025-11-12

### Added

- Added `--nice` flag to `llmb-run submit-all` to lower job priority using Slurm's `--nice` feature.

## [1.3.7] - 2025-11-10

### Added

- `llmb-run submit-all` now submits jobs sorted by scale in descending order (largest jobs first).

## [1.3.6] - 2025-11-07

### Added

- `LLMB_SKIP_PP` environment variable to disable post-processing.

## [1.3.5] - 2025-11-05

### Changed

- Use `release.yaml` for version information, fall back to `metadata.yaml` if it does not exist.

## [1.3.4] - 2025-10-24

### Added

- job_config.strong_scaling to llmb-config.yaml

## [1.3.3] - 2025-10-22

### Fixed

- SbatchLauncher places slurm.out files in $LLMB_WORKLOAD instead of cwd

## [1.3.2] - 2025-10-17

### Added

- CUPTI library mount support for profiling workarounds via `cuda_cupti_lib` tool

## [1.3.1] - 2025-10-15

### Added

- 'slurm_cpu_partition' to llmb-config.yaml.

### Fixed

- llmb-config.yaml generation now handles `by_gpu` container image structure
- Improved error handling with targeted blocks for easier debugging

## [1.3.0] - 2025-10-14

### Added

- Custom Nsys version container mounts on per workload basis.

## [1.2.7] - 2025-10-02

### Fixed

- Only generate experiment_ids for nemo2 workloads.

## [1.2.6] - 2025-10-02

### Added

- Support for `megatron_bridge` launcher type (uses Nemo2Launcher)

## [1.2.5] - 2025-09-24

### Added

- GB300 support

## [1.2.4] - 2025-09-19

### Added

- `--scales` flag for submit-all command to specify exact scales to run

## [1.2.3] - 2025-09-03

### Added

- `experiment_id` field in llmb-config.yaml files based on normalized config hash and framework version

## [1.2.2] - 2025-08-30

### Added

- A check for 'uv' venv_types and properly sets VIRTUAL_ENV parameter.

## [1.2.1] - 2025-08-22

### Added

- Set the default value of `GPU_METRICS_NODES` to 0 for GPU metrics collection

## [1.2.0] - 2025-08-14

### Added

- `llmb-run submit-all`
  - `--min-scale` flag, flexible scale requirements (either `--max-scale` or `--min-scale` required)
  - `--workloads` filters on comma separated list of workload name or workloadname_size (e.g., `pretrain_nemotron4_340b` or `pretrain_nemotron4`)
  - `--dtype` filters on a comma separated list of dtypes.
- `llmb-run single`: `--force` flag to skip validation

## [1.1.0] - 2025-08-13

### Added

- Per-dtype scale configuration in recipe metadata (mapping form under `dtypes`), with optional per-dtype `exact_scales`.
- Shared normalization across the toolchain to interpret legacy and new forms consistently.

### Changed

- `llmb-run list` now shows detailed per-dtype scales by default (no `-v` required).
- `submit-all` and validation respect per-dtype scales and exactness; legacy forms remain supported.

## [1.0.1] - 2025-08-07

### Changed

- Updated Nemo2 work dir path provided to WiT module, per api change.

## [1.0.0] - 2025-08-04

### Added

- New `submit-all` command to automatically generate and submit jobs for all installed pretrain/finetune workloads:
  - Supports dynamic scale generation up to a specified `--max-scale` (GPUs).
  - Allows setting `--repeats` for multiple runs of each configuration.
  - Includes `--profile` flag to enable profiling for all generated jobs.
  - Integrates with `--dryrun` for previewing jobs.
  - Filters workloads by type (pretrain/finetune only) and cluster GPU compatibility.

## [0.10.0] - 2025-07-24

### Added

- Dynamic timelimit for workload inspector jobs based on scale:
  - Short (≤128 GPUs): 1 hour
  - Medium (129-511 GPUs): 4 hours
  - Long (≥512 GPUs): 8 hours
  - Special case: DeepSeek V3 pretrain ≥256 GPUs uses 8 hours
- Blocks medium/long WiT jobs on GPU partitions, to avoid wasting resources.
  - Heuristic: gpu_part == cpu_part, cpu_part requires GRES, or cpu_part == backfill

### Changed

- Workload inspector now uses calculated timelimit instead of default 1 hour

## [0.9.1] - Previous Release

- Previous functionality (details can be added when reviewing historical changes)

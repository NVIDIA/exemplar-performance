<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# llmb-collector

Scrape cluster and hardware info with machine-friendly report output (JSON, YAML, or formatted text).

## Install

```bash
uv sync
```

Then run with `uv run llmb-collector` or activate the venv and use the script:

```bash
source .venv/bin/activate
llmb-collector --help
```

Install optional deeper secret detection for environment sanitization:

```bash
uv sync --extra secrets
```

Subcommands:

- `llmb-collector collect ...`
- `llmb-collector list`
- `llmb-collector list host|network|container`
- `llmb-collector validate-config`

## Usage

**Host + network YAML to screen:**

```bash
llmb-collector collect --host --network --format yaml
```

**List domains and sub-collectors:**

```bash
llmb-collector list
```

**List only network domains:**

```bash
llmb-collector list network
```

**Host only (all host commands):**

```bash
llmb-collector collect --host --env --output ./results/
```

**Only CPU and GPU host domains:**

```bash
llmb-collector collect --host-collect cpu,gpu --output report.json
```

**Only network domain (separate top-level `network` group):**

```bash
llmb-collector collect --network-collect interfaces --format yaml
```

**Only InfiniBand/RDMA network commands:**

```bash
llmb-collector collect --network-collect infiniband --format yaml
```

**All host except GPU:**

```bash
llmb-collector collect --host --host-exclude gpu --output ./results/
```

**Formatted text report to file:**

```bash
llmb-collector collect --host --format text --output report.txt
```

**YAML to file (format inferred from extension):**

```bash
llmb-collector collect --host --output report.yaml
```

**Directory output (\_cloudperf/ layout for analyze compatibility):**

```bash
llmb-collector collect --host --output ./results/
```

**Container info (run inside workload container):**

```bash
llmb-collector collect --container-collect gpu --output -
```

**Validate command config:**

```bash
llmb-collector validate-config
```

## Options

- `list` — Print available domains and sub-collectors, then exit.
- `validate-config` — Report exact duplicate and near-duplicate command definitions for human review.
- `collect --host` / `collect --network` / `collect --container` / `collect --env` — Enable full host, network, container, and env collection via flags (default: all false unless overridden by env vars).
- `--host-collect` — Comma-separated host commands or domains (e.g. `cpu`, `gpu`, `memory`, `storage`, `os`, `software`) to collect as a subset. Mutually exclusive with `--host`.
- `--host-exclude` — Comma-separated host commands or domains to skip when using full host collection.
- `--network-collect` — Comma-separated network commands or domains (e.g. `interfaces`, `infiniband`) to collect as a subset. Mutually exclusive with `--network`.
- `--network-exclude` — Comma-separated network commands or domains to skip when using full network collection.
- `--container-collect` — Comma-separated container commands or domains (`gpu`, `libraries`, `runtime`) to collect as a subset. Mutually exclusive with `--container`.
- `--container-exclude` — Comma-separated container commands or domains to skip when using full container collection.
- `--output PATH` — Write to file (e.g. `report.json`), directory (e.g. `./results/`), or `-` for stdout. Omit to print to screen.
- `--format json|yaml|text` — Output format (default: json).
- `--compact` / `--no-compact` — JSON indentation (YAML always uses block style).
- `--env-pattern REGEX` — Regex for env vars to include.

Environment variables `LLMB_COLLECT_HOST`, `LLMB_COLLECT_NETWORK`, `LLMB_COLLECT_CONTAINER`, `LLMB_COLLECT_ENV`, `LLMB_HOST_COLLECT`, `LLMB_HOST_EXCLUDE`, `LLMB_NETWORK_COLLECT`, `LLMB_NETWORK_EXCLUDE`, `LLMB_CONTAINER_COLLECT`, `LLMB_CONTAINER_EXCLUDE` override the corresponding flags.

If all collection toggles are false, `llmb-collector` prints a grouped help message with available commands per subject and exits without collecting data.

## Command Config

Command argv definitions live in YAML instead of Python code. Config files are organized by realm and domain:

```text
configs/commands/
  system/
    cpu.yaml
    gpu.yaml
    memory.yaml
    os.yaml
    software.yaml
    storage.yaml
  network/
    interfaces.yaml
    infiniband.yaml
  container/
    gpu.yaml
    libraries.yaml
    runtime.yaml
```

A realm is the top-level collection area: `system`, `network`, or `container`. A domain is a file inside a realm, such as `cpu`, `interfaces`, or `infiniband`.

Each domain file defines globally unique command IDs:

```yaml
commands:
  lscpu:
    description: CPU topology and capabilities.
    argv:
      - lscpu
      - --json
```

Profiles and CLI selectors use command IDs or domain names, not full argv. For example, `--host-collect cpu,lscpu` is resolved through the registry.

Config resolution order:

1. `LLMB_COMMANDS_DIR`, if set. This directory must contain `system/`, `network/`, and `container/`.
2. Repo checkout config at `configs/commands/`.
3. Config roots discovered through Python entry points in the `llmb` group.
4. Packaged defaults at `llmb_collector/commands_config/`.

The installed wheel includes the packaged defaults. The repo-level `configs/commands/` tree is intended for local editing and review.

## Built-In Commands

System domains:

- `cpu`: `lscpu`
- `gpu`: `nvidia_smi`
- `memory`: `free`
- `storage`: `lsblk`
- `os`: `hostnamectl`, `lsb_release`
- `software`: `enroot_version`, `sinfo_version`

Network domains:

- `interfaces`: `ip`, `ethtool`
- `infiniband`: `ibstat`, `ibstat_list`, `ibv_devices`, `ibv_devinfo`, `ibv_devinfo_verbose`, `rdma_link`, `rdma_dev`, `rdma_statistic`

Container domains:

- `gpu`: `container_nvidia_smi`
- `libraries`: `container_dpkg_libnccl2`
- `runtime`: `container_python3_version`

## Config Validation

Run:

```bash
llmb-collector validate-config
```

The validator reports:

- Duplicate command IDs as load errors.
- Exact duplicate argv definitions as findings.
- Similar argv definitions as findings, including reordered flags, supersets, and high-overlap argument lists.

Findings include both command IDs, source files, and argv values. Near-duplicate findings do not block normal collection by default; they are intended for review and CI checks.

## Environment Collection

Enable environment collection with:

```bash
llmb-collector collect --env
```

The default env regex includes workload, GPU/ML, and common CI/pipeline variables, including prefixes such as `SLURM_`, `CLOUDPERF_`, `CUDA_`, `TORCH_`, `NCCL_`, `PYTHON`, `CI`, `CI_`, `GITHUB_`, `GITLAB_`, `BUILDKITE_`, `JENKINS_`, `AZURE_`, `TEAMCITY_`, `BITBUCKET_`, `DRONE_`, `CODEBUILD_`, `GITEA_`, `HARNESS_`, `VERCEL_`, and `NETLIFY_`.

Override the inclusion regex:

```bash
llmb-collector collect --env --env-pattern '^LLMB_'
```

Environment values are sanitized by default. Variables with secret-like names are emitted as `<redacted>`, for example names containing `TOKEN`, `SECRET`, `PASSWORD`, `KEY`, `CREDENTIAL`, `AUTH`, `COOKIE`, `SESSION`, `PRIVATE`, `CERT`, or `SIGNATURE`.

If installed with the `secrets` extra, `detect-secrets` is also used opportunistically to scan env values in memory:

```bash
uv sync --extra secrets
```

If `detect-secrets` is not installed, sanitization falls back to the built-in name-based redaction. Programmatic callers can customize this with `CollectConfig.env_redact_pattern` or disable it with `CollectConfig.redact_env=False`.

## Output behavior

- Disabled top-level collectors are omitted from output (no `system: null`, etc.).
- `exception` fields are omitted when no exception occurred.
- YAML output is block style and multiline command output is emitted as block scalars.
- JSON-looking `stdout` values (for example `lscpu --json`) are parsed and emitted as structured YAML.

When writing to a directory, results are emitted under `_cloudperf/`:

```bash
llmb-collector collect --host --output ./results/
```

creates:

```text
./results/_cloudperf/
```

Each command gets a subdirectory containing `cmdline`, `returncode`, `stdout`, `stderr`, and `exception`.

## Python Integration

The package exposes a runnable provider through the `llmb` Python entry point group:

```toml
[project.entry-points.llmb]
collector = "llmb_collector.provider:get_provider"
commands_config = "llmb_collector.command_loader:get_packaged_commands_config"
```

Another app can discover and run collection:

```python
from importlib.metadata import entry_points

provider = entry_points(group="llmb")["collector"].load()()
data = provider.collect(host=True, network=False, container=False, env=False)
```

The provider object provides:

- `commands_config()` — Return the bundled command config root.
- `command_registry()` — Load this package's command registry.
- `collect(config=None, **overrides)` — Run collection and return a Python dictionary.

The `commands_config` entry point is a lower-level config-only hook for tools that want to inspect or load the packaged YAML catalog without running collection.

This `llmb` group is the `llmb-collector`-aware integration path — callers know the shape of the provider class. For cross-tool consumers that want to stay decoupled from `llmb-collector` entirely, see the [Capability registry](#capability-registry) below.

## Capability registry

For consumers that want to stay decoupled from `llmb-collector` entirely, the same features are advertised as named capabilities under the neutral `llmb_capabilities` entry-point group (shared with `llmb-auth` and any other producer). The group name, the `Capability` shape, and each capability's `invoke` signature are the only contract; neither side imports the other, and any producer can register additional providers for the same capability names.

`Capability` and the `system.collect` / `system.commands-config` name constants are defined by the [`llmb-capabilities`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities) contract package, giving every producer and consumer one shared definition of the entry-point group name, the `Capability` shape, and each capability's supported `version` ceiling. `llmb_collector.capabilities` re-exports `Capability`, so embedders don't need a direct `llmb-capabilities` import just to type-hint against it.

`llmb-collector` ships two capabilities:

| `name`                   | `version` | `invoke(...)` returns | Purpose                                                                                                                                                                                                                                                                                                                               |
| ------------------------ | --------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `system.collect`         | 1         | `dict`                | `invoke(ns, *, prefix=None, config=None)` — a JSON-serializable collection dict. With `config=None` all collectors (host, network, container, env) run with packaged defaults and env redaction on; pass an explicit `llmb_collector.collect.CollectConfig` to narrow scope or tweak knobs. Runs commands in-process (no temp files). |
| `system.commands-config` | 1         | `Traversable`         | `invoke(ns, *, prefix=None)` — the packaged `commands_config` tree as an `importlib.resources` `Traversable`. Consumers that only need the on-disk path can read `metadata["commands_config_path"]` without invoking.                                                                                                                 |

Each `Capability` carries `name`, `version`, `invoke`, an optional `add_arguments(parser, prefix=None)` that contributes the flags the capability needs, `metadata` (always including `{"provider": "llmb-collector"}`; `system.commands-config` also exposes `commands_config_path` so consumers can introspect the packaged catalog without invoking), and `args_model` (`None` for both, since they're Namespace-first rather than model-first — see `llmb-capabilities`' README for the model-first alternative used by other producers). Both capabilities currently set `add_arguments=None` because they take no CLI flags. A consumer discovers, then invokes:

```python
import argparse
from importlib.metadata import entry_points

WANT = "system.collect"

caps = [c for ep in entry_points(group="llmb_capabilities") for c in ep.load() if c.name == WANT]
if not caps:
    raise RuntimeError(f"No provider found for capability: {WANT}")

parser = argparse.ArgumentParser(prog="my-tool")
for cap in caps:
    if cap.add_arguments is not None:
        cap.add_arguments(parser, prefix=cap.metadata.get("provider"))
args = parser.parse_args()

cap = caps[0]
data = cap.invoke(args, prefix=cap.metadata.get("provider"))
```

When several plugins advertise the same capability `name`, disambiguate on `metadata["provider"]` (it also makes a natural per-provider flag prefix, avoiding collisions).

`CAPABILITIES` and `Capability` are re-exported at the package top level for in-process callers that don't need entry-point discovery:

```python
from llmb_collector import CAPABILITIES, Capability
```

## Development

Run tests:

```bash
uv run pytest
```

Build the package:

```bash
uv build
```

## Releasing

Version bumps, `CHANGELOG.md`, git tags, and GitLab Release notes are handled by [python-semantic-release](https://python-semantic-release.readthedocs.io/) on merge to `main`. See [docs/releasing.md](docs/releasing.md) for the commit-prefix rules, manual CI jobs (`Release`, `Publish to Artifactory`), and troubleshooting.

## License

This project is licensed under the Apache License, Version 2.0.
See `LICENSE` for the full license text.

# Security Policy: llmb-collector

## Reporting a Vulnerability

Do not open a public issue, merge request, or discussion for a suspected
security vulnerability.

Report vulnerabilities through one of these private channels:

- [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
  (preferred)
- [psirt@nvidia.com](mailto:psirt@nvidia.com), optionally encrypted with the
  [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key)
- GitLab private vulnerability reporting through this repository's
  **Security** tab

Include the affected `llmb-collector` version or commit, vulnerability type,
reproduction steps, proof of concept if available, affected command or
configuration, and whether the issue affects externally distributed images.

NVIDIA PSIRT will acknowledge the report, assess its severity, coordinate a
fix, and publish security guidance when appropriate.

## Security Architecture and Context

`llmb-collector` is a Python CLI and library that executes predefined system
commands and collects host, GPU, network, container, and environment metadata.
It emits JSON, YAML, text, or a `_cloudperf/` directory for downstream
benchmark analysis.

The package has no network listener. Its primary security responsibilities are
executing only intended commands, preventing secret leakage during environment
collection, and controlling the contents and destination of generated reports.

**Repository Exposure Classification:** Internal (high confidence).
Basis: source is hosted on NVIDIA's internal GitLab and packages are published
to internal Artifactory.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: `llmb-collector` is included in container images distributed outside
NVIDIA. It can inspect sensitive host and workload metadata in those
environments.

### Components and trust boundaries

- `src/llmb_collector/cli.py` parses collection and output options.
- `src/llmb_collector/collect.py` executes subprocesses, collects environment
  values, redacts selected data, and writes reports.
- `src/llmb_collector/command_loader.py` loads command definitions from YAML,
  `LLMB_COMMANDS_DIR`, and Python entry points.
- `src/llmb_collector/capabilities.py` exposes `system.collect` and
  `system.commands-config`.
- `configs/commands/` and packaged `commands_config/` define executable argv.

The key trust boundaries are operator-controlled configuration, installed
entry-point providers, subprocess execution with the current OS identity,
process environment inspection, and caller-selected output paths.

### Supported versions

Security fixes target the latest tagged release from `main` and externally
distributed images rebuilt with that release. Upgrade older images and package
versions unless an advisory states otherwise.

### Scope

In scope:

- The `llmb_collector` package and CLI
- Shipped command YAML
- Environment collection and redaction
- Report serialization and output handling
- Capability and entry-point integration implemented here

Out of scope:

- Vulnerabilities in the underlying OS utilities
- Host, container-runtime, and scheduler isolation
- Downstream upload or storage systems

## Threat Model

### Arbitrary command execution through custom configuration

`LLMB_COMMANDS_DIR` can replace the command catalog with caller-controlled
YAML. The collector's subprocess runner executes the configured argv with the
collector's OS privileges. A malicious configuration can therefore execute
arbitrary programs.

Treat command configuration as executable code, protect its source and
permissions, and do not expose `LLMB_COMMANDS_DIR` to untrusted users.
Argument-list execution and timeouts reduce accidental risk but do not make
untrusted commands safe.

### Secret leakage through environment collection

Environment collection relies on name-based redaction and optional
`detect-secrets` value scanning. Callers can disable redaction or broaden the
selection pattern, and non-standard secret names can evade heuristics.

Keep redaction enabled, install the optional scanner in externally distributed
images when appropriate, minimize the collected environment, and review
reports before export.

### Infrastructure metadata disclosure

Commands such as `nvidia-smi`, `hostnamectl`, networking tools, and RDMA tools
can reveal detailed host identity and topology. Output written to shared paths
or exported from an external image may disclose sensitive infrastructure.

Collect only required realms, use private output locations, and apply
downstream access controls and retention policies.

### Shell-mediated packaged commands

Some shipped configuration invokes `bash -c`, including network-interface
collection. Although the shipped string is static, shell wrappers increase
the impact of future configuration mistakes.

Prefer direct argv commands for new collectors and review all shell-bearing
configuration as executable code.

### Unconfined filesystem writes

Caller-selected output paths are not sandboxed. A malicious or mistaken path,
including a symlinked destination, may overwrite files accessible to the
collector process.

Run with least privilege and restrict output to an approved directory owned
by the invoking identity.

### Malicious entry-point command providers

Installed Python packages can register command-config entry points. A
compromised provider can introduce executable command definitions.

Pin and verify dependencies in external images and avoid runtime package
installation. Entry-point discovery is not a trust mechanism.

### Incomplete secret detection

Short values, URLs, and values stored under innocuous environment names may
not be recognized by optional value scanning.

Redaction is defense in depth, not a guarantee. Do not place secrets in the
collector environment unless required, and prevent raw reports from becoming
public artifacts.

## Critical Security Assumptions

- Packaged command definitions and any configured `LLMB_COMMANDS_DIR` are
  controlled by trusted maintainers.
- Embedding callers do not disable redaction or broaden collection without an
  explicit data-handling review.
- The host OS and container runtime enforce process, capability, and
  filesystem isolation.
- Output destinations are authorized and protected from unintended readers.
- The deployment environment provides authentication and authorization; the
  collector itself has none.
- Collected command output is trusted only as data and is not evaluated by
  downstream consumers.
- External image builds pin dependencies, verify provenance, scan images, and
  rebuild promptly for security fixes.

## Dependency Security

Runtime dependencies are declared in `pyproject.toml`. PyYAML input is loaded
with `yaml.safe_load`; `llmb-capabilities` is obtained from NVIDIA
Artifactory. `detect-secrets` is optional defense in depth. External image
builds should pin exact versions and verify image and package provenance.

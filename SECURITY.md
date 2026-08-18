# Security Policy: Exemplar Performance Recipes

## Reporting a Vulnerability

If you discover a potential security vulnerability in Exemplar Performance Recipes, please **do not open a public issue, pull
request, or discussion**.

Report the vulnerability privately through one of these channels:

- **Web (preferred):**
  [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/)
- **Email:** [psirt@nvidia.com](mailto:psirt@nvidia.com)
  - For encrypted communication, use the
    [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key).
- **Repository private reporting:** Use the repository's **Security** tab and
  select **Report a vulnerability**, when private vulnerability reporting is
  enabled on the applicable GitHub or GitLab repository.

Please include:

- The affected project version, release, branch, or commit
- The affected recipe, CLI component, Helm chart, or script
- The vulnerability type
- Reproduction instructions
- Proof-of-concept code, if available
- The expected and observed behavior
- The potential impact, including required cluster or account privileges
- Relevant deployment details, with credentials and other sensitive data removed

Detailed reports help NVIDIA evaluate and address issues faster.

NVIDIA's Product Security Incident Response Team (PSIRT) will acknowledge the
report, validate the issue and its severity, coordinate remediation and testing,
and publish a security bulletin when appropriate under NVIDIA's coordinated
vulnerability disclosure process.

## Security Architecture & Context

The Exemplar Performance Recipes provide containerized recipes,
Python command-line tools, Helm resources, and supporting scripts for installing,
launching, and evaluating distributed AI workloads on GPU clusters.

The principal executable components include:

- `llmb-install`, which reads workload metadata, downloads model assets and
  tools, clones dependency repositories, creates Python environments, and runs
  workload setup procedures.
- `llmb-run`, which reads cluster and task configuration, constructs workload
  environments, submits Slurm jobs, records experiment metadata, parses logs,
  and packages experiment results.
- Workload `launch.sh` scripts and metadata files, which define the commands,
  containers, dependencies, mounts, and resource requirements used for each
  benchmark.
- The inference Helm chart under
  `alternative_recipes/helm-charts/inference/llm-benchmarks`, which can deploy
  model servers and benchmark clients on Kubernetes.
- Diagnostic utilities such as the system-information recipe, which collect
  host, accelerator, scheduler, container-runtime, and network configuration.

This software operates primarily as a **CLI, deployment-automation toolkit, and
collection of executable workload recipes**. Its main security responsibility is
to preserve the integrity of workload execution and prevent unintended
disclosure of cluster configuration, access credentials, model assets, and
benchmark results.

**Repository Exposure Classification:** Public.

Basis: development occurs in an internal repository, but release snapshots,
including this document and most project source, are published to a public
GitHub repository. This document is written for the public release audience.

**Service Exposure Classification:** External / Regulated (high confidence).

Basis: the CLIs, workload recipes, documentation, and Helm resources are
distributed externally and are intended to run on production-capable GPU
clusters, interact with authenticated artifact registries, and support benchmark
result submission and review.

The primary trust boundaries are:

1. **Operator input to launcher:** CLI arguments, YAML task files,
   `cluster_config.yaml`, environment overrides, and workload metadata flow into
   `llmb-run` and then into Slurm or Kubernetes workloads.
2. **Repository and upstream sources to installer:** Workload metadata directs
   `llmb-install` to Git repositories, Python packages, containers, model
   registries, and downloadable tools. Retrieved code can later execute on the
   installation host or compute nodes.
3. **Host to workload container:** Recipes mount selected host or shared-storage
   paths and execute containers with GPU, network, and scheduler access.
4. **Workload to result artifacts:** Logs, generated configuration, diagnostics,
   and parsed performance data are written under the installation tree and may
   be included in archives for external review.
5. **Development to public release:** Reviewed snapshots of approved project
   content move from the development environment into public distribution.

The project does not implement an independent user authentication service.
Identity, authorization, network isolation, and workload privileges are supplied
by the host operating system, Slurm, Kubernetes, container runtime, and artifact
registries.

### Threat Model

1. **Compromised dependency or workload supply chain:** Workload metadata consumed
   by `cli/llmb-install/src/llmb_install/core/dependency.py` selects Git
   repositories, revisions, installation scripts, and Python packages.
   `downloads/huggingface.py` downloads Python modules from metadata-selected
   Hugging Face repositories and may execute repository-supplied code while
   verifying tokenizer and configuration assets. Its tokenizer finalization path
   also retries Nemotron repositories with remote code enabled when ordinary
   loading fails. `downloads/tools.py` and `downloads/compute_uv.py` likewise
   install downloaded tools.
   A compromised upstream source, malicious metadata change, or inadequately
   reviewed revision could therefore execute code on an installation host or
   compute node. Metadata changes are code-reviewed, and dependencies are pinned
   to specific commits or versioned artifacts where upstream projects permit it.

2. **Credential disclosure through configuration and generated artifacts:**
   Install-time Hugging Face and registry credentials are normally passed through
   process environment variables and are not intended to become result artifacts.
   Task environment overrides are recorded in generated `llmb-config_*.yaml`
   files, however, and the alternative inference Helm chart accepts a token as a
   regular value. Credentials placed in recordable overrides, general
   configuration, plaintext Helm values, or logs can therefore appear in rendered
   manifests, archives, or shared result bundles.

3. **Disclosure through benchmark and diagnostic artifacts:** `llmb-run archive`
   packages generated configuration and experiment logs for submission or offline
   analysis. The `microbenchmarks/system_info/launch.sh` recipe intentionally
   records detailed, user-readable host, accelerator, Slurm, network, Enroot, and
   `/etc/enroot/environ.d` configuration needed to diagnose performance problems.
   These artifacts may reveal cluster-specific configuration or other sensitive
   operational details if shared without review.

4. **Exposure of mounted data to workload code:** Workload launch scripts and the
   alternative inference chart support operator-selected host paths, shared result
   directories, and container mounts so recipes can run across varied cluster
   environments. These mounts do not grant permissions beyond those of the
   submitting user or pod, but downloaded workload or container code can access
   any data deliberately made visible to it. An overly broad or incorrect mount
   can therefore expose or modify more user-accessible data than intended.

### Critical Security Assumptions

- Workload metadata, launch scripts, dependency revisions, container references,
  and public-release changes are reviewed and accepted only from trusted
  contributors.
- Operators running `llmb-install`, `llmb-run`, or the Helm chart are authorized
  to submit workloads and consume the requested GPU, storage, registry, and
  network resources.
- Each installation and its recipes are controlled by its operator. `llmb-run`
  executes workloads with the submitting user's existing Slurm and filesystem
  permissions and does not provide privilege separation between that operator
  and their installed workload code.
- Slurm, Kubernetes RBAC, the container runtime, the host operating system, and
  shared-storage permissions enforce isolation between users and workloads.
- The alternative inference Helm chart is used for short-lived benchmarking in
  a trusted cluster environment. Cluster administrators provide TLS,
  authentication, ingress restrictions, and network policy before making its
  `ClusterIP` model endpoint reachable outside that boundary.
- Registry, Git, package, and model sources provide authentic artifacts, and
  operators review pinned revisions and image versions before using them in
  sensitive environments.
- Credentials are supplied through protected secret-management mechanisms,
  scoped to the minimum required access, and are not placed in task environment
  overrides, general configuration fields, or files that will be archived.
- Diagnostic logs, `llmb-config_*.yaml`, profiles, and result archives are
  treated as potentially sensitive and reviewed before being submitted or
  shared.
- `/etc/enroot/environ.d` contains user-readable, non-secret cluster tuning and
  runtime configuration; administrators do not place credentials in that
  directory.

## Deployment and Operational Guidance

- Use dedicated, least-privilege tokens for Hugging Face, NGC, Git, and result
  submission.
- Prefer Kubernetes Secrets or an equivalent secret provider over plaintext Helm
  values.
- Restrict write access to installed workload directories, dependency caches,
  cluster configuration, and release configuration.
- Review repository revisions, container tags, downloaded tools, and workload
  metadata before installation.
- Limit inference services to the required namespace and add authentication,
  TLS, and network policy before increasing their exposure.
- Inspect diagnostic output and result archives for credentials, internal
  infrastructure details, proprietary configuration, or model data before
  sharing them.
- Treat system-information output as cluster configuration. Its detailed
  hardware, scheduler, network, and Enroot data is intentionally collected for
  performance diagnosis.
- Mount only the host and shared-storage paths required by a workload, especially
  when running newly introduced containers or upstream workload code.
- Run workloads under non-administrative scheduler and Kubernetes identities
  whenever possible.

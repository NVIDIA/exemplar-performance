# Security Policy: llmb-capabilities

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

Include the affected `llmb-capabilities` version, affected component such as
`discover_providers()` or `invoke_sync()`, vulnerability type, reproduction
steps, proof of concept if available, provider packages involved, and impact
to downstream tools or externally distributed images.

NVIDIA PSIRT will acknowledge the report, assess its severity, coordinate a
fix, and publish security guidance when appropriate.

## Security Architecture and Context

`llmb-capabilities` is a standard-library-only Python contract library for
capability discovery and invocation across LLMB packages. It defines
capability names, schema-version ceilings, protocols, discovery helpers,
argument resolution, sync/async invocation bridges, and provider conformance
checks.

The package does not acquire credentials, collect system data, or upload
artifacts itself. Those operations are implemented by providers such as
`llmb-auth`, `llmb-collector`, and `llmb-uploader`.

**Repository Exposure Classification:** Internal (high confidence).
Basis: source is hosted on NVIDIA's internal GitLab and releases are published
to internal Artifactory.

**Service Exposure Classification:** External / Regulated (high confidence).
Basis: the library is a runtime dependency of `llmb-collector`, which is
included in container images distributed outside NVIDIA. Discovery and
invocation defects therefore have transitive external impact.

### Components and trust boundaries

- `src/llmb_capabilities/_capability.py` defines the core `Capability`,
  `CapabilityLike`, and model-validation contracts.
- `src/llmb_capabilities/_constants.py` defines names and supported versions.
- `src/llmb_capabilities/_discovery.py` enumerates and loads installed Python
  entry points.
- `src/llmb_capabilities/_invoke.py` delegates calls to provider code.
- `src/llmb_capabilities/_resolve_args.py` maps CLI namespaces into
  provider-defined validation models.
- `src/llmb_capabilities/conformance.py` validates provider advertisements,
  primarily during provider testing.
- `_bearer_token.py`, `_oidc_id_token.py`, and `_tls_ssl_context.py` define
  the authentication and TLS capability protocols.
- `_system_collect.py`, `_system_commands_config.py`, `_storage_upload.py`,
  and `_post_processor_process.py` define collection, upload, and processing
  capability protocols.

Any installed package can register an entry point. Consumers are responsible
for provider selection, allow or deny policy, version checks, and deciding
when invocation is safe.

### Supported versions

The supported release line is `1.x`. Consumers should pin
`llmb-capabilities>=1,<2` or a narrower compatible range and upgrade to the
latest security release.

### Scope

In scope:

- Capability contracts and constants
- Entry-point discovery and loading
- Argument resolution and invocation helpers
- Provider conformance checks
- This package's release and publication pipeline

Out of scope:

- Authentication and TLS behavior implemented by `llmb-auth`
- Collection behavior implemented by `llmb-collector`
- Upload and post-processing implemented by `llmb-uploader`
- External image hardening beyond this package's transitive impact

## Threat Model

### Malicious provider entry-point injection

An attacker-controlled package can register an `llmb_capabilities` entry point.
`discover_providers()` loads entry points without signature verification or a
publisher allowlist, executing provider import code with process privileges.

Install only approved packages, pin image dependencies, verify provenance, and
apply explicit provider allowlists before invocation.

### Arbitrary behavior through invocation delegation

`invoke_sync()` and `invoke_async()` call provider-controlled `invoke`
implementations. A malicious provider can inspect caller arguments, return
forged tokens or TLS contexts, exfiltrate telemetry, or perform unauthorized
uploads.

Invocation is a trust boundary, not a sandbox. Consumers must select trusted
providers and run with least OS and network privilege.

### Missing provider or version policy

Discovery returns advertisements but does not automatically enforce consumer
preference or `SUPPORTED` version ceilings. A consumer may accidentally bind
to an unexpected or incompatible provider.

Consumers must enforce capability names, version ceilings, and provider
metadata before invocation. Providers should run conformance checks in CI.

### False assurance from runtime protocols

Runtime-checkable Python protocols verify attribute presence, not full method
signatures or semantic validity. A malformed provider may pass `isinstance`
checks while lacking a usable argument model.

Use `assert_capability_conforms()` for untrusted advertisements and explicitly
validate model-first capability requirements.

### Permissive provider-defined argument validation

`resolve_capability_args()` forwards namespace data to a provider's
`args_model.model_validate()`. Validation quality depends on the provider;
permissive models can accept unintended or attacker-controlled fields.

Use explicit namespace prefixes and strict provider models that reject unknown
fields.

### Resource exhaustion in invoke bridges

The sync/async bridges do not impose timeouts. A faulty provider can block a
thread, hang a process, or exhaust resources.

Consumers should apply application-level timeouts and external image
deployments should enforce CPU, memory, and process limits.

### Supply-chain compromise

Compromise of CI credentials, Artifactory, package resolution, or the external
image build could substitute this contract or a provider package and affect
all downstream consumers.

Restrict publication credentials, pin versions, verify package hashes, and
sign and scan externally distributed images.

## Critical Security Assumptions

- Only approved packages are installed and allowed to register capability
  entry points.
- Consumers implement provider allowlists, version checks, and capability
  selection policy.
- Provider packages correctly secure tokens, TLS, telemetry, and uploads.
- Provider-defined argument models strictly validate caller input.
- Provider CI runs conformance checks; runtime discovery alone is not treated
  as validation.
- Provider calls complete in reasonable time or consumers enforce timeouts.
- External image owners pin versions, restrict runtime installation, verify
  provenance, and monitor security updates.

## Dependency Security

This package intentionally has no runtime dependencies outside the Python
standard library. Security of the overall capability ecosystem depends on the
installed provider and consumer packages. Coordinate version pinning and
security review across `llmb-auth`, `llmb-collector`, and `llmb-uploader`,
especially in externally distributed images.

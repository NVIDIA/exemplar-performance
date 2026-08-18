# CHANGELOG

<!-- version list -->

## v1.2.0 (2026-07-18)

### Features

- **bearer_token**: Add device-code to BearerTokenMethod
  ([`39c7365`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/39c7365c5b493b2ae696a31fc353e4224db286fa))

## v1.1.0 (2026-07-17)

### Features

- **discovery**: Add installed provider discovery
  ([`ad0a186`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/ad0a18666c33a67b0cd70384d7a15278c05ddada))

## v1.0.0 (2026-07-17)

### Bug Fixes

- **bearer_token**: Replace device-code with auth-code
  ([`de54b11`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/de54b11457f7f7983406a4c8cfdde2f953acd9a7))

### Breaking Changes

- **bearer_token**: BearerTokenMethod now accepts auth-code instead of device-code.

## v0.2.1 (2026-07-14)

### Bug Fixes

- Use publish-url (not publish_url) for the nv uv index
  ([`e55654a`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/e55654a6334b645b1e2cb00d7711d3ea13e805ae))

### Continuous Integration

- Auto-run Publish to Artifactory once Release cuts a new tag
  ([`4299a0f`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/4299a0f6257daa1569cee00b36360d299fbe8db1))

- Publish via uv's named index instead of ad-hoc publish/check URLs
  ([`7175458`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/7175458ce224a888bf401ae8c7ff7b1139fd35a8))

### Testing

- **conformance**: Cover model-first narrow-protocol-mismatch cases
  ([`6ca80db`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/6ca80dbccd4baa6a5f92fbd3dc8e4f5a3bd407fa))

## v0.2.0 (2026-07-14)

### Bug Fixes

- Address MR review comments on resolve_capability_args, invoke, and conformance
  ([`38399cd`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/38399cd27b215c4996e3592d06e3641e15ff87ed))

### Chores

- **cursor**: Sync jira, commit/mr, rebase, and remaining-items skills
  ([`f9ff057`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/f9ff057ff5f1adb0e28aea815b89473c1724d481))

- **skills**: Fix grammar and markdownlint nits in cursor skills
  ([`ea94e09`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/ea94e095528303c7f32dbfcd33ee69ef5f0c680c))

### Features

- Add storage.upload and post-processor.process capabilities
  ([`e51f3a1`](https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities/-/commit/e51f3a1c59e43ee2d6671970ea9fc5a650d6343f))

## v0.1.0 (2026-07-08)

- Initial Release

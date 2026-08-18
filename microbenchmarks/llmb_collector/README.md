# llmb-collector microbenchmark

This proof-of-concept recipe runs `llmb-collector` directly on one allocated
compute node. It complements, rather than replaces, the diagnostic checks in
`microbenchmarks/system_info`.

## Run

Install LLMB with the compute-node architecture selected, then run:

```bash
cd "$LLMB_INSTALL"
llmb-run submit -w microbenchmark_llmb_collector --scale 1
```

The report is written to `llmb-collector.json` in the experiment directory
created by `llmb-run`.

## Source snapshots

`source_snapshot/` contains unmodified snapshots of separately maintained
repositories:

- llmb_collector
- llmb_capabilities

They are included so an LLMB checkout contains the versions tested for that
LLMB release, including patch releases, without requiring separately published
external package artifacts. Do not develop these packages in this directory; replace each snapshot from an
exact upstream tag instead.

The snapshots intentionally preserve the complete small upstream trees,
including tests, command configuration, changelogs, locks when present, and
licenses. Nested VCS and editor metadata are not part of the snapshots.

The launcher passes both local source trees to `uvx`, so the collector does not
resolve `llmb-capabilities` from NVIDIA's internal Python index. PyYAML and build
requirements still resolve normally and are cached by uv.

## Runtime environment

The launcher uses the architecture-specific executable installed by
`llmb-install`:

```text
$LLMB_INSTALL/bin/$(uname -m)/uvx
```

`uvx` puts ephemeral tool environments and downloaded dependencies in the uv
cache. The launcher defaults `UV_CACHE_DIR` to `$LLMB_INSTALL/.cache/uv` and
`UV_PYTHON_INSTALL_DIR` to a subdirectory there so neither package data nor a
uv-managed Python installation is written under the user's home directory. A
future `llmb-run` change can centralize these defaults for every workload.

# Recipe schemas

Every `recipe.yaml` is **declarative source-of-truth** that an agent edits and a generator
renders into flat, committed manifests (the "rendered manifests" pattern). These schemas are
what CI validates the `recipe.yaml` against.

## The layering

```
envelope.yaml                     # shared, thin (~a dozen fields) -> drives the MATRIX + CI routing
  └─ scenario-<scenario>.yaml     # per-scenario, $refs the envelope, adds serving + scenario block
```

- **Envelope** = scenario-agnostic metadata carried by *every* recipe (name, model, gpu_type, arch,
  engine, serving_mode, **scenario**, **distribution**, launcher, status, results_link, provenance).
  Deliberately short. It is the query/catalog surface, not the config.
- **Scenario schema** = the full `recipe.yaml` shape for one `scenario`. It embeds the envelope via
  `$ref` and adds the `serving` block plus the scenario-specific block.

`scenario` is the **benchmark class** — it is both the top-level `recipes/<scenario>/` fork *and* the
schema/analysis selector. Each scenario gets its own schema and its own analysis. The envelope is the
only shared contract.

## Selecting a schema (what CI does)

```
scenario = recipe.yaml .envelope.scenario
validate recipe.yaml against scenario-${scenario}.yaml   # which itself $refs envelope.yaml
```

| scenario   | launcher | scenario schema          | block    | measures                          |
| ---------- | -------- | ------------------------ | -------- | --------------------------------- |
| `llm-perf` | aiperf   | `scenario-llm-perf.yaml` | `bench:` | TPS/GPU, TPS/User, TTFT, TPOT, KV |

## Path convention

```
recipes/<scenario>/<distribution>/<model>-<hardware>-<engine>-<serving_mode>[-<variant>]/
recipes/llm-perf/256k/nemotron-ultra-3-b200-vllm-agg/
```

The top-level `<scenario>` fork mirrors the two-families design; `<distribution>` is the
workload/dataset; the leaf carries model + hardware + serving setup.

## What the schemas encode

- **Envelope is shared, scenarios are typed** — same ~dozen fields everywhere; everything else is
  per-scenario.
- **Serving stack is shared substrate** — both scenarios reference the same `serving.stack`
  (e.g. `vllm-agg`); only the launcher/scenario layer differs.
- **Cell-leaf** — 1 recipe = 1 envelope = 1 MATRIX row = 1 results set; the concurrency/trace sweep
  is the recipe's internal `rungs`. offload on/off and 256k/1m are *separate cells*.
- **Hardware is a dimension, not an overlay** — `gpu_type`/`arch` are query fields; GPU/model-tuned
  engine flags are baked into the recipe (`serving.extra_args`), not a shared `hardware/` tree.

## CI checks (per recipe)

1. Validate `recipe.yaml` against `scenario-${scenario}.yaml`.
2. Re-render templates and assert the committed `rendered/` matches (drift check).
3. Lint `rendered/*.yaml` (kubeconform).
4. Roll the envelope into the top-level MATRIX.

Bump `schema_version` (`k8s.<scenario>.v<N>`) on any breaking change to a scenario schema.

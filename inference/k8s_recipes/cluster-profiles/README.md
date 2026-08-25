# `cluster-profiles/` — per-cluster configuration

This directory is the **one place** a recipe operator describes their cluster to the recipe. Everything else in the recipe is cluster-agnostic.

## What's tracked, what isn't

Only `*.example` files are tracked. Your real `*.env` and `*.values.yaml` are gitignored on purpose — they carry namespace identifiers, owner names, and other site-specific values that don't belong in version control.

```
cluster-profiles/
├── README.md                         ← this file
├── _template.env.example             ← generic starter (Stage 0 / envsubst)
├── _template.values.yaml.example     ← generic starter (Stage 1 / Helm)
├── cluster1.env.example              ← preset for an example cluster
├── cluster1.values.yaml.example
├── cluster2.env.example              ← preset for another example cluster
├── cluster2.values.yaml.example
└── <your-cluster>.env  (or .values.yaml)   ← yours, gitignored
```

## Onboarding a new cluster — zero-touch (recommended)

The four main commands scaffold everything; you never hand-copy a template or hand-run `kubectl`:

```sh
# wizard: detect, confirm, WRITE cluster-profiles/my-cluster.env (no manual cp)
scripts/llmb-k8s init --cluster my-cluster

# ensure namespace + pull/HF secrets (idempotent), weights, stage, preflight
scripts/llmb-k8s install --cluster my-cluster

# preflight, deploy, benchmark, fetch, teardown
scripts/llmb-k8s run <cell> my-cluster
```

- **`init`** writes a complete, run-ready profile for you — no `cp _template.env.example …` step. It also
  captures the cluster's SSO/Teleport login as `CONNECT_CMD`, so any later auth✗ prints the exact login line.
- **`install`** creates the namespace and the image-pull + HF secrets **idempotently** (vanilla `kubectl`,
  safe to re-run). Secret values come from `HF_TOKEN` / `NGC_API_KEY` in your env, the profile, or
  `~/.config/llmb/secrets` — never committed. If a value is missing it names the exact key to set.

## Onboarding by hand (fallback)

If you'd rather write the profile yourself:

1. **Copy the generic template:**

   ```sh
   cp _template.env.example         my-cluster.env
   cp _template.values.yaml.example my-cluster.values.yaml
   ```

   (You only need the one matching the frontend you'll use, but copying both is fine.)

2. **Fill in the placeholders.** The fields most operators need to change are:

   | Field                                                       | What to put                                                                                                           |
   | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
   | `NAMESPACE` / `cluster.namespace`                           | Your pre-existing namespace                                                                                           |
   | `OWNER` / `owner`                                           | Your username (label only; informational)                                                                             |
   | `MODEL_CACHE_PVC` / `cluster.modelCachePvc`                 | Name of an RWX PVC your cluster admin has bound in your namespace                                                     |
   | `ARTIFACTS_STORAGE_CLASS` / `cluster.artifactsStorageClass` | An RWO StorageClass with `reclaimPolicy: Delete` (e.g. `ebs` on AWS EKS, `standard-rwo` on GKE, `managed-csi` on AKS) |
   | `GPU_COUNT` / `cluster.gpuCount`                            | GPUs to request for the SGLang server (default 4)                                                                     |
   | `USE_DYNAMO_TOLERATIONS` / `cluster.useDynamoTolerations`   | `false` on most clusters; `true` only on Dynamo dev clusters                                                          |

3. **Validate the profile, then pre-flight a cell:**

   ```sh
   scripts/llmb-k8s profile validate --cluster my-cluster
   scripts/llmb-k8s preflight <cell> my-cluster
   ```

   `profile validate` checks the profile vars are present/reachable; `preflight` does the live cluster check
   and tells you exactly what's missing (the namespace, the RWX PVC, or the `hf-token-secret`).

4. **Once pre-flight is green, run the benchmark:**

   ```sh
   scripts/llmb-k8s run <cell> my-cluster
   ```

## Why split files per cluster?

Because the same physical workload runs on very different clusters with different GPU types, taints, and storage drivers. The recipe's intrinsics (model, sweep, hicache flags) stay in `recipe.env` / `chart/values.yaml`; the per-cluster knobs stay here. Two operators can run the same recipe with completely different cluster profiles and produce comparable results.

## Adding a preset for a cluster everyone uses

If you've onboarded a cluster that others on your team will reuse, commit a `.example` for it (real values are still gitignored). Use one of the existing `*.env.example` files as a reference for how to document the cluster's quirks inline.

# `scripts/` — orchestration

Operator-facing entry points for the **declarative recipe path** (`recipes/**/recipe.yaml` →
`rendered/` → cluster). The recipe encodes all benchmark truth; the cluster profile supplies the
cluster-specific variables. Scripts only stitch them together.

Convention: every script takes `<cell-dir> <cluster-profile>` as its first two positional args and
reads everything else from `cluster-profiles/<profile>.env`. No global env vars to set, no `_lib.sh`
to source.

______________________________________________________________________

## The two entry points

### `llmb-k8s <verb> <cell-dir> <cluster-profile> [extra-args]`

Unified front-door. Dispatches each verb to the right tool for the cell's `envelope.scenario`.

Accepts **named flags** too — `llmb-k8s <verb> --recipe <cell> --cluster <name> …` — fully
backward-compatible with the positional form. The **profile is resolved first**
(`profile_resolver.py`): a missing profile fails fast with the list of existing profiles + how to create one,
and a profile that pins `KUBE_CONTEXT` is reachability-checked before any cluster work.

```
llmb-k8s preflight  <cell> <profile>               # dryrun.sh
llmb-k8s stage      <cell> <profile>               # dataset or traces, by scenario
llmb-k8s deploy     <cell> <profile>               # deploy.sh
llmb-k8s run        <cell> <profile> [run-id]      # run.sh (full lifecycle)
llmb-k8s analyze    <cell> <profile> <results-dir> # aggregate + exemplar_check, by scenario
llmb-k8s publish    <cell> <profile> <results-dir> # publish.py
llmb-k8s reclaim    <profile> [--apply]            # reclaim.py
llmb-k8s profile validate --cluster <name>         # profile_resolver.py — completeness + reachability
llmb-k8s status     <cell> <profile>               # observe.py — render/publish/last-run + live server/job
llmb-k8s jobs       <profile> [--recipe <cell>] [--all]   # observe.py — list Jobs (state · run-id · created)
llmb-k8s logs       <cell> <profile> [run-id]      # observe.py — stream the newest (or named) Job
```

The last three are **read-only** (no cluster mutation, no compat gate) — they answer "what's running / is
it hung / tail it / has the published record drifted?" without hand-built `kubectl` selectors.

Adding a new scenario = one entry in `REGISTRY` inside `llmb-k8s`. This is the stable contract
the Phase 3 main-llmb integration targets.

### `run.sh <cell-dir> <cluster-profile> [run-id] [flags]`

Full lifecycle orchestrator. Drives the complete sequence idempotently:

```
1. preflight   dryrun.sh — catch misconfig before touching the cluster
2. namespace   verify exists; --managed-ns creates it
3. stage       stage-dataset.sh
4. server      deploy.sh kubectl apply (idempotent)
5. wait-ready  kubectl rollout status (15 min timeout)
6. benchmark   sweep.sh
7. fetch       pull artifacts off PVC into results/<run-id>/
8. teardown    kubectl scale --replicas=0 (GPU slot freed; Deployment preserved for instant resume)
```

Flags:

| Flag                 | Effect                                                                                                                                                                                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--managed-ns`       | Create the namespace if it does not exist.                                                                                                                                                      |
| `--skip-stage`       | Skip staging (assume dataset/traces already on PVC)                                                                                                                                             |
| `--skip-server`      | Skip server deploy (reuse running instance)                                                                                                                                                     |
| `--smoke-only`       | Run concurrency=1 only, then stop                                                                                                                                                               |
| `--teardown`         | Scale server to 0 after sweep; `--managed-ns --teardown` also deletes namespace                                                                                                                 |
| `--no-fetch`         | Skip artifact extraction                                                                                                                                                                        |
| `--idle-guard[=MIN]` | Background hang watchdog for the benchmark phase: if the server generates no new tokens for `MIN` min (default 30), cancel the run so it stops holding the GPU node. See `idle_guard.sh` below. |
| `--rungs "N ..."`    | Override concurrency list for sweep.sh                                                                                                                                                          |

Prints the exact `publish.py` command + elapsed wall time at completion.

> **Scenario coverage:** `run.sh` drives the llm-perf scenario via `sweep.sh`.

______________________________________________________________________

## Composable primitives

Use these directly when you want finer control than `run.sh` (e.g. keep a server up across
multiple sweeps, or re-run only the bench step).

| Script                                                                                       | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dryrun.sh <cell> <profile>`                                                                 | Validate manifests + resolve all `${VARS}` without touching the cluster. Prints a summary table of resolved values, runs kubeconform if available, and reports variable resolution — genuinely-unresolved **cluster** vars fail the run, while the bench runner's own in-pod placeholders (`${STOP_STAT}`, `${KARCH}`, …) are recognised and **preserved**, not flagged. Exit 0 = clean. (`--classify` is a hidden, profile-free test sub-mode of that logic; see `selftest.py`.)                                                                                                                                                                                                                                                                                                        |
| `deploy.sh <cell> <profile> [--render-only] [--env-set K=V] [--env-unset K]`                 | Apply the server manifests (`rendered/*.yaml` except bench-job.yaml) via `kubectl apply`. Idempotent. `--env-set`/`--env-unset` (also on `run.sh`) are the **recipe-builder VARIANT path**: patch the serving containers' env at apply time so iterating on a recipe (strip a flag, try a value, run an A/B control) keeps the run-owner GC, fleet labels, artifacts PVC, preflight and wait-ready — instead of a hand-rolled `kubectl apply` that loses all of them.                                                                                                                                                                                                                                                                                                                    |
| `merge_env_override.py`                                                                      | The apply-stream stage behind `--env-set`/`--env-unset`. **Hash-neutral** (never touches `rendered/*.yaml`, so `recipe_hash` does not move) and therefore **loudly marked**: every patched object + pod template gets `llmb.nvidia.com/variant=true`, a `variant-id` label, and an annotation with the exact overrides; `run.sh` writes `results/<run-id>/_variant.json` + `run_meta.overrides`, and `publish.py` **refuses** such a run-dir (no `--force`). Fail-closed: a malformed spec aborts before anything is applied. No override → byte-identical passthrough.                                                                                                                                                                                                                  |
| `sweep.sh <cell> <profile> [run-id] [--rungs "N ..."]`                                       | Create the artifacts PVC (idempotent), collision-guard against concurrent sweeps, apply the bench Job, and follow logs. Each call gets a unique `RUN_ID`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `lane.py <cell> <stage\|bench>`                                                              | Resolves a cell's scenario to the stage/bench scripts that drive it — the shared implementation shared by `run.sh` and `llmb-k8s`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `stage-dataset.sh <cell> <profile>`                                                          | Copy the canonical trace to the cluster's model-cache PVC and verify its sha256. Idempotent: noop if already present with the right hash.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `fetch_results.sh [--with-inputs] [--partial] <run-id>`                                      | Pull bench artifacts off the RWO artifacts PVC via a transient mounter pod (tar streaming). Excludes the ~2 GiB `inputs.json` by default. `--partial` = best-effort recovery: grab whatever exists, never exit non-zero (used by run.sh's failure trap). Requires `CLUSTER` env var or `_lib.sh` env.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `recovery.py <cell> <profile> [--exit N] [--reason R] [--run-id ID] [--rungs-done/-all …]`   | Turn a failed/interrupted run into an actionable next step: likely cause + `--partial` fetch + the exact `llmb-k8s run … --skip-server` resume command. Pure; called by `run.sh`'s EXIT trap.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `reclaim.py <profile> [--apply]`                                                             | Audit the namespace for stale GPU-holding resources (CrashLoopBackOff pods, completed/failed Jobs). Dry-run by default; `--apply` acts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `idle_guard.sh <cell> <profile> [run-id] [--kill-hung] [--idle-timeout MIN] [--keep-server]` | Watchdog that frees the GPU node when a run stops making progress. **Auto-teardown:** scales the `<name>-server` Deployment to 0 when the run's Job reaches a terminal state (weights stay on the PVC → instant `deploy.sh` resume). **Hang-kill** (`--kill-hung`): while the Job is active, polls the vLLM server `/metrics`; if generation tokens flatline for `--idle-timeout` min (default 30), deletes the Job so it stops squatting the node. `--keep-server` reports/kills but leaves teardown to the caller (how `run.sh --idle-guard` uses it). `--sum-tokens` reads a `/metrics` blob on stdin and prints the summed generation-token counter (the flatline signal; unit-tested in `selftest.py`). Only ever scales that server to 0 or deletes that Job — safe to background. |
| `render.sh <cell> [--to <dir>]`                                                              | Render `recipe.yaml` + serving templates → `rendered/` via Jinja2. Run after any recipe change before `deploy.sh`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `observe.py status\|jobs\|logs …`                                                            | Read-only status/jobs/logs backing the `llmb-k8s` verbs of the same name. Composes `profile_resolver` + `provenance --impact` + `lane` + a kubectl layer pinned to the profile's `KUBE_CONTEXT`/`NAMESPACE`. Pure helpers (`last_run`/`format_jobs`/`pick_job`/`status_rows`) are unit-tested; only the kubectl parts need a cluster.                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `profile_resolver.py resolve\|list\|context\|compat …`                                       | Resolve a `--cluster` to a valid/reachable profile + the recipe-cluster target-compat gate. Backs `llmb-k8s`'s profile resolution.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

### Bench collision guard (`sweep.sh`)

Before launching a sweep, `sweep.sh` checks for any active bench Job carrying
`llmb.nvidia.com/cell=<name>` in the namespace. If one exists, it blocks with the active job name
and a `kubectl logs -f` command. Two concurrent aiperf loads against the same server corrupt both
results (shared KV cache, split throughput, inflated latency). The guard is self-cleaning — no lock
objects — and clears automatically when the Job reaches terminal state.

______________________________________________________________________

## Analysis + publishing

| Script                                                                               | What it does                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analysis/llm-perf/aggregate_metrics.py <run-dir> [--out csv]`                       | Join per-step AIPerf JSON/JSONL + `/metrics` snapshots + `gpu_stats.csv` + `run_meta.json` → `metrics_summary.csv`. One row per concurrency.                                                                                                                           |
| `analysis/llm-perf/exemplar_check.py <cell> <csv> [--json] [--set] [--actual-gpu G]` | Check a run against the cell's committed reference bar. `--set` writes the computed value as the new bar. `--actual-gpu` enforces same-GPU cross-cluster correctness.                                                                                                  |
| `analysis/llm-perf/sla_compare_dashboard.py`                                         | Interactive SLA scatter + ISL/OSL distribution panels. Overlays N runs. Requires `plotly≥5.20`.                                                                                                                                                                        |
| `publish.py <cell> <run-dir> [--set-baseline] [--dry-run]`                           | End-to-end publish: aggregate → metric → RESULTS.md data block → status bump → catalog rebuild. Dispatches by scenario.                                                                                                                                                |
| `provenance.py <cell> --stamp [--run run_meta.json]`                                 | Emit the `## Provenance` block for RESULTS.md (recipe_hash + image + dataset + wall_h + gpu_h).                                                                                                                                                                        |
| `provenance.py --check [root]`                                                       | CI gate: every published cell must have a RESULTS.md with the current recipe_hash.                                                                                                                                                                                     |
| `provenance.py [cell] --impact` / `--impact [root]`                                  | **Advisory** (never fails): after editing a recipe, does its published record still match? Prints `MATCH`/`DRIFT`/`UNPUBLISHED` (single cell) or lists drifted cells + remediation (`--all`). The soft counterpart to `--check`; `run.sh` surfaces DRIFT at preflight. |

______________________________________________________________________

## CI / repo health

| Script                       | What it does                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `selftest.py`                | Full CI gate: runs all checks below + provenance + publish --dry-run. All must pass before merging.                                   |
| `validate.py`                | Schema-validate every `recipe.yaml` against `schema/envelope.yaml` + the scenario schema.                                             |
| `check_invariants.py`        | Cross-cutting invariant checks: placeholder images, max-num-seqs coverage, adaptive sweep config.                                     |
| `lint_manifests.py`          | Parse every `rendered/*.yaml` (neutralising `${VARS}`) and verify basic k8s structure.                                                |
| `build_catalog.py [--check]` | Regenerate `catalog.json` + inject the README matrix. `--check` is the CI drift guard (read-only).                                    |
| `recipe_hash.py <cell>`      | Print the full-recipe fingerprint (recipe semantics + rendered manifests). Deterministic; moves iff the benchmark definition changes. |
| `contract_check.py`          | Verify that cross-file contracts hold (e.g. schema fields referenced in templates exist).                                             |

______________________________________________________________________

## New cell scaffolding

```bash
scripts/new-cell.sh <scenario> <distribution> <name>
# e.g.
scripts/new-cell.sh llm-perf 256k my-model-b200-vllm-agg
```

Creates the `recipes/<scenario>/<distribution>/<name>/` skeleton with a stub `recipe.yaml`,
empty `RESULTS.md`, and `.gitkeep` for `rendered/`. Use an existing recipe as the starting point.

______________________________________________________________________

## Idempotency

Every script that touches the cluster is safe to re-run:

- `deploy.sh` / `run.sh --skip-server`: `kubectl apply` is a no-op if manifests are unchanged.
- `stage-dataset.sh`: sha256 gate skips if already present.
- `sweep.sh`: each call generates a unique `RUN_ID`; Jobs never collide.
- Artifacts PVC creation (inside `sweep.sh`): `kubectl apply` of a PVC spec is idempotent.
- `run.sh --managed-ns`: namespace creation is guarded with `kubectl get namespace` first.

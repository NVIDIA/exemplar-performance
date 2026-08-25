# `serving/dynamo-disagg/` — disaggregated Dynamo backend (EXPERIMENTAL)

> **EXPERIMENTAL.** This backend is applied directly with `kubectl`; point the benchmark at its frontend Service.

## What it is

A **`DynamoGraphDeployment`** ([`dgd-disagg-1m.yaml`](dgd-disagg-1m.yaml)) with:

- **frontend / router** (Dynamo) — the OpenAI-compatible entrypoint,
- **2 prefill workers** (TP=8, one 8-GPU B200 node each),
- **1 decode worker** (TP=8).

Prefill and decode run in **separate pools**. Cross-worker **prefill→decode KV transfer** uses vLLM's **NixlConnector**
(`kv_role: kv_both`), a pure peer-to-peer KV transport.

**Note on "KVBM".** In Dynamo, the literal *KVBM* is a separate **tiered GPU→CPU→disk offload**
connector (the DynamoConnector). That is **intentionally NOT used here** — this backend does pure
cross-worker KV transfer with no CPU/disk offload. NixlConnector matches the design intent (improve prefill parallelism, not KV capacity) and is the supported path.

The worker configuration uses engine arguments supported by `dynamo.vllm`. Serving-only arguments
(`--tool-call-parser`, `--reasoning-parser`, `--enable-auto-tool-choice`,
`--enable-prompt-tokens-details`) are intentionally omitted — `dynamo.vllm` workers accept only vLLM
*engine* args and reject those; they do not affect token throughput.

## Prerequisites

- The **Dynamo operator** installed in the cluster (it reconciles `DynamoGraphDeployment`).
- A **healthy `grove-operator`** (Dynamo depends on it to materialize the multi-worker graph). A
  grove-operator outage blocks the deploy — that is a platform issue, not a recipe defect.
- On DRA nodes, device-plugin GPU scheduling needs **`schedulerName: default-scheduler`** on the
  worker pods (set via `DISAGG_SCHEDULER_NAME`; see Portability below).
- 3 GPU nodes total (2 prefill + 1 decode), each 8× B200.

## Portability

These manifests are **not** cluster-specific literals — cluster identity is parameterized with
`${VAR}` placeholders (the same ones the shared harness uses) so you render them through `envsubst`
before applying. A **whitelisted** `envsubst` is required so the placeholders resolve **without**
clobbering the shell variables (`$SNAPSHOT_DIR`, `$(python3 ...)`) inside the worker `args`.

Values that come from `recipe.env` + your cluster profile:
`NAMESPACE`, `OWNER`, `IMAGE_PULL_SECRET`, `MODEL_CACHE_PVC`, `MODEL_CACHE_SUBPATH`, `SGLANG_IMAGE`.

Disagg-only knobs you must export (they are not in the aggregated profile):

| Var                                               | Meaning                                                                                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `DISAGG_SCHEDULER_NAME`                           | `default-scheduler` unless you run a gang scheduler that refuses device-plugin GPUs on DRA nodes — see `TROUBLESHOOTING.md`.               |
| `DISAGG_PREFILL_NODE_1` / `DISAGG_PREFILL_NODE_2` | Hostnames of available 8-GPU nodes for the prefill worker(s). `serving.yaml` uses only `_1` (1P1D); `dgd-disagg-1m.yaml` uses both (2P1D). |
| `DISAGG_DECODE_NODE`                              | Hostname of a available 8-GPU node for the decode worker.                                                                                  |

Pick free nodes with the DRA-aware method in `TROUBLESHOOTING.md`
(the device-plugin allocatable count hides DRA claims). The **tolerations block is left literal** on
purpose — it is a per-cluster taint union; trim or extend it for your GPU pool. If your registry is
public, delete the `imagePullSecrets` lines instead of setting `IMAGE_PULL_SECRET`.

## Apply

```sh
cd serving/dynamo-disagg

# Load recipe + profile config, then export the disagg-only node/scheduler knobs.
set -a; . ../../recipe.env; . ../../cluster-profiles/$CLUSTER.env; set +a
export DISAGG_SCHEDULER_NAME=default-scheduler
export DISAGG_PREFILL_NODE_1=<free-8gpu-node-a>
# dgd 2P1D only
export DISAGG_PREFILL_NODE_2=<free-8gpu-node-b>
export DISAGG_DECODE_NODE=<free-8gpu-node-c>

WL='$NAMESPACE $OWNER $IMAGE_PULL_SECRET $MODEL_CACHE_PVC $MODEL_CACHE_SUBPATH $SGLANG_IMAGE $DISAGG_SCHEDULER_NAME $DISAGG_PREFILL_NODE_1 $DISAGG_PREFILL_NODE_2 $DISAGG_DECODE_NODE'

# Operator path (needs the Dynamo operator + a healthy grove-operator):
envsubst "$WL" < dgd-disagg-1m.yaml | kubectl apply -f -
kubectl get dynamographdeployment,pods -l nvidia.com/dynamo-graph-deployment-name=llmb-disagg-1m

# ...OR the operator-independent path (bypasses grove; brings up etcd+NATS itself):
# envsubst "$WL" < runtime.yaml | kubectl apply -f -
# envsubst "$WL" < serving.yaml | kubectl apply -f -
```

## Benchmark it

Point the **shared bench** at the Dynamo frontend Service. The
frontend is exposed as:

```
llmb-disagg-1m-frontend:8000
```

Set the server endpoint the bench targets to that Service (the OpenAI-compatible `/v1` API) and run the configured AIPerf workload.

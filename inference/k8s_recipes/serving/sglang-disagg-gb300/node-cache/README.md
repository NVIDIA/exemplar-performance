# GB300 versioned node cache

These DaemonSets prepare one pinned model revision on every GB300 node before serving. They separate
the slow FSx-to-NVMe copy from benchmark runs and write the ready marker only after the exact revision
resolves offline and every checkpoint shard verifies.

| DaemonSet manifest                   | Model / revision                                               | Shards | Used by                                                                     |
| ------------------------------------ | -------------------------------------------------------------- | ------ | --------------------------------------------------------------------------- |
| `glm5-fp8-4f96cc5e.yaml`             | `zai-org/GLM-5-FP8` @ `4f96cc5e…`                              | 142    | `recipes/llm-perf/Glm5/Gb300_k8s/…/glm5-gb300-*`                            |
| `nemotron-ultra-nvfp4-183968f8.yaml` | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` @ `183968f8…` | 113    | `recipes/llm-perf/nemotron_ultra_disagg/Gb300_k8s/…/nemotron-ultra-gb300-*` |

First install one GB300 cell for the model so the pinned snapshot exists on the shared model-cache PVC.
Then render the profile variables and apply the matching DaemonSet. GLM-5 example:

```bash
cp cluster-profiles/glm5-gb300.env.example cluster-profiles/glm5-gb300.env
# Edit glm5-gb300.env for the target namespace, secrets, storage, and fabric.
scripts/llmb-k8s install glm5-gb300 --recipes glm5-gb300-dyn13-8k1k-ll-c150-1p9d-bs15

set -a
source cluster-profiles/glm5-gb300.env
set +a
export MODEL_CACHE_PVC="${MODEL_CACHE_PVC_GLM5_FP8:-$MODEL_CACHE_PVC}"
envsubst '${NAMESPACE} ${SCHEDULER_NAME} ${IMAGE_PULL_SECRET} ${GPU_PRODUCT} ${MODEL_CACHE_SUBPATH} ${MODEL_CACHE_PVC}' \
  < serving/sglang-disagg-gb300/node-cache/glm5-fp8-4f96cc5e.yaml \
  | kubectl --context "$KUBE_CONTEXT" apply -f -
kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" \
  rollout status daemonset/glm5-fp8-4f96cc5e-node-cache --timeout=8h
```

Nemotron Ultra (same steps, different cell and manifest):

```bash
cp cluster-profiles/nemotron-ultra-gb300.env.example cluster-profiles/nemotron-ultra-gb300.env
# Edit nemotron-ultra-gb300.env for the target namespace, secrets, and storage.
scripts/llmb-k8s install nemotron-ultra-gb300 --recipes nemotron-ultra-gb300-sglang-dynamo14-10k16k-1p1d

set -a
source cluster-profiles/nemotron-ultra-gb300.env
set +a
export MODEL_CACHE_PVC="${MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4:-$MODEL_CACHE_PVC}"
envsubst '${NAMESPACE} ${SCHEDULER_NAME} ${IMAGE_PULL_SECRET} ${GPU_PRODUCT} ${MODEL_CACHE_SUBPATH} ${MODEL_CACHE_PVC}' \
  < serving/sglang-disagg-gb300/node-cache/nemotron-ultra-nvfp4-183968f8.yaml \
  | kubectl --context "$KUBE_CONTEXT" apply -f -
kubectl --context "$KUBE_CONTEXT" -n "$NAMESPACE" \
  rollout status daemonset/nemotron-ultra-nvfp4-183968f8-node-cache --timeout=8h
```

The DaemonSet copies from the same claim `install` downloaded into: the per-model
`MODEL_CACHE_PVC_<MODEL>` profile key when set (for example `MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4`),
otherwise `MODEL_CACHE_PVC` — the `export` line above applies that rule before `envsubst`.

Serving recipes use the same revision-specific host path read-only and fail closed when the marker,
pinned snapshot, or shard count does not match. Each recipe sets `serving.model_cache_staging.expected_shards`
(defaults to 142 when unset). A new model revision must use a new DaemonSet name and host path; never
overwrite an existing revision in place.

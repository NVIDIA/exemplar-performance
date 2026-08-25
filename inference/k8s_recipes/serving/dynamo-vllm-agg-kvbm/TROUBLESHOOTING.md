# Troubleshooting — Dynamo + vLLM aggregated KVBM

## Worker exits with a KVBM connector import error

- **Symptom:** `dynamo.vllm` reports that `kvbm.vllm_integration.connector` or `DynamoConnector` cannot be loaded.
- **Root cause:** the selected runtime image does not contain the Dynamo KVBM/vLLM integration matching the recipe.
- **Fix:** use the recipe's digest-pinned Dynamo vLLM runtime. Do not substitute a generic vLLM image.

## Pod starts but the worker cannot discover KVBM peers

- **Symptom:** repeated etcd registration/discovery errors against `127.0.0.1:2379`.
- **Root cause:** the etcd sidecar is unhealthy, or discovery environment variables were overridden.
- **Fix:** inspect the `etcd` container logs and health endpoint; retain `DYN_DISCOVERY_BACKEND=etcd` and
  `ETCD_ENDPOINTS=127.0.0.1:2379`.

## CPU cache churns or allocation fails

- **Symptom:** high KVBM eviction/churn, pinned-memory allocation failures, or the Pod is OOM-killed.
- **Root cause:** the CPU tier is smaller than the effective GPU KV tier, or the Pod memory limit cannot
  accommodate the pinned tier plus runtime overhead.
- **Fix:** keep `DYN_KVBM_CPU_CACHE_GB` at least as large as the GPU KV capacity, and raise the Pod memory
  request/limit with it. This POC deliberately uses low `gpu_mem_util` with a 20 GB CPU tier on B200.

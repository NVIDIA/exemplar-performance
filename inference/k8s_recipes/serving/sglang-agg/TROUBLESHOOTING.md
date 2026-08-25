# SGLang aggregated troubleshooting

## Server never becomes ready

- Symptom: the server pod remains in its startup probe or `/v1/models` is empty.
- Check: inspect the `verify-model` init container first, then the `sglang` container logs.
- Likely causes: missing pinned model snapshot, image/architecture mismatch, or insufficient GPU memory for the configured context and CUDA graph ceiling.

### Fresh install reports a complete cache but the server cannot find the model

- Symptom: `install` completes, but `verify-model` reports a missing completion sentinel or pinned `config.json`.
- Cause: the server and installer disagree about the Hugging Face cache layout, completion sentinel, or Secret key.
- Expected contract: `install` and the download Job create `${MODEL_CACHE_SUBPATH}/.llmb_download_done/<revision>` and store the snapshot under `${MODEL_CACHE_SUBPATH}/hub/models--<org>--<repo>/snapshots/<revision>`. The Hugging Face Secret uses key `token`.
- Fix: start from a fresh profile/install and use the rendered pinned snapshot path. Do not create `.k8s-download-complete`, flatten the hub cache, or rewrite the Secret to an `HF_TOKEN` key.

## Bench pod is stuck in `ContainerCreating` with `Multi-Attach`

- Symptom: the aggregate server is Ready, but the AIPerf pod cannot attach the model-cache PVC because the server already mounts that ReadWriteOnce claim on another node.
- Cause: an old synthetic AIPerf manifest unnecessarily mounted the serving weights PVC. Synthetic jobs generate their inputs and call the server over HTTP; they do not read model weights.
- Fix: re-render with the current shared AIPerf template. Synthetic jobs omit the model-cache volume, while trace jobs retain it because their dataset may live there.
- For a trace job whose dataset cache is RWO, set `bench.colocate_with_server: true` or use an RWX dataset cache.

## Requests fail only under load

- Symptom: AIPerf reports HTTP failures while the SGLang process remains alive.
- Check: compare server restarts, endpoint membership, and the AIPerf error summary.
- Fix: keep readiness/liveness probes disabled for a single-replica benchmark cell if event-loop starvation causes Kubernetes to remove or restart an otherwise healthy endpoint.

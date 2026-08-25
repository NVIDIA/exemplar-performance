# SGLang disaggregated troubleshooting

## Fresh install reports a complete cache but workers cannot find the model

- Symptom: `install` completes, but `verify-model` reports a missing completion sentinel or pinned `config.json`, or workers fail before registration.
- Cause: the worker template and installer disagree about the Hugging Face cache layout, completion sentinel, or Secret key.
- Expected contract: `install` and the download Job create `${MODEL_CACHE_SUBPATH}/.llmb_download_done/<revision>` and store the snapshot under `${MODEL_CACHE_SUBPATH}/hub/models--<org>--<repo>/snapshots/<revision>`. The Hugging Face Secret uses the profile-selected `${HF_SECRET}` name and key `token`.
- Fix: start from a fresh profile/install and use the rendered pinned snapshot path. Do not create `.k8s-download-complete`, flatten the hub cache, hard-code `hf-token`, or rewrite the Secret to an `HF_TOKEN` key.

## Workers load but do not register

- Check the worker logs after model verification, then verify the cluster-profile fabric values and NIXL/UCX initialization. Cache verification success proves only the pinned model layout, not transport readiness.

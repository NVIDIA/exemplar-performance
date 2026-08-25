# Troubleshooting — `serving/dynamo-disagg/` (disaggregated Dynamo + vLLM, hybrid-Mamba)

Operational notes for **NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4** disaggregated
(prefill/decode, NixlConnector cross-node KV transfer) on B200 / vLLM 0.20.1 / ai-dynamo.
Format: **Symptom → Root cause → Fix**. Add new findings here as you hit them — this file is
stored knowledge for the next human or agent (see the per-recipe TROUBLESHOOTING pattern in
`../../REFACTOR-EXECUTION-PLAN.md`).

> **The one that will cost you a day if you don't know it:** §1 (`expandable_segments`).

______________________________________________________________________

## 1. Garbage decode output (gibberish tokens), HTTP 200 — the silent footgun

- **Symptom:** requests succeed, KV transfers appear to happen, but decode emits gibberish
  (mixed scripts / random symbols, e.g. `"The capital of France is"` → `" , 100000000"`).
- **Root cause:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is **incompatible with
  NixlConnector**. PyTorch's CUDA VMM allocator remaps KV-cache virtual addresses to different
  physical pages, **invalidating the IB memory regions NIXL registers** → every cross-worker KV
  transfer reads corrupted data. `expandable_segments` is a *common, recommended* vLLM perf flag,
  which is why it's easy to inherit — and older images accept it **silently** (→ garbage). Newer
  images (≥ `vllm-runtime:1.3.0-rc.2`) raise a hard `ValidationError` that names this exactly.
- **Fix:** **do not set `expandable_segments` with NixlConnector** (removed from `serving.yaml` +
  `dgd-disagg-1m.yaml`). If you truly need it for fragmentation/OOM, enable the cumem allocator
  instead (`enable_cumem_allocator`), which routes KV allocations through a pool where
  expandable_segments is disabled.

## 2. `REMOTE_DISCONNECT` during cross-node KV transfer

- **Symptom:** decode/prefill logs `REMOTE_DISCONNECT`; cross-node requests fail.
- **Root cause:** NixlConnector uses **UCX directly** — `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`
  do **not** steer it. Without the right UCX transport + device selection, cross-node RDMA can't
  establish.
- **Fix:** `UCX_TLS=rc,rc_x,dc,dc_x,cuda_copy`, `UCX_NET_DEVICES=all`, `UCX_RNDV_SCHEME=get_zcopy`,
  generous `UCX_RC_TIMEOUT`; request `rdma/ib` on **both** prefill and decode; add `IPC_LOCK` +
  `SYS_RESOURCE` capabilities and `ulimit -l unlimited`.

## 3. Garbage output or engine-init error from block size

- **Symptom:** garbage output, or engine init fails with a KV-cache-spec/`unify_hybrid_kv_cache_specs`
  error.
- **Root cause:** default `block_size=16`; **hybrid Mamba disagg requires `--block-size 64`**, and it
  must **match** between prefill and decode (and the frontend KV router, `--kv-cache-block-size 64`).
- **Fix:** `--block-size 64` on both workers; keep `--no-disable-hybrid-kv-cache-manager` (hybrid KV
  manager must stay ON).

## 4. Decode `CrashLoopBackOff`: "Free memory on device < desired GPU memory utilization"

- **Symptom:** decode crashes ~90 s in with
  `ValueError: Free memory on device cuda:N (106/178 GiB) ... less than ... (0.85, 151 GiB)`.
- **Root cause:** on **DRA nodes the device-plugin `nvidia.com/gpu` allocatable lies** — it reports
  8 GPUs "free" while real GPU *memory* is occupied by a DRA/other workload. `--gpu-memory-utilization 0.85` then can't reserve.
- **Fix:** schedule onto a **truly-free** node (check real free memory per GPU, not the device-plugin
  count), or lower `--gpu-memory-utilization` to fit. Keep prefill/decode util **matched**.

## 5. Intermittent empty decode output

- **Symptom:** requests complete, but some decode responses are empty.
- **Action:** confirm that prefill and decode use compatible runtime versions and KV-transfer settings. If the
  issue persists, collect worker logs and escalate it to the runtime owner before publishing results.

## 6. Pod `1/1 Ready` but `/v1/*` returns 404 / empty (premature readiness)

- **Symptom:** pod shows `1/1 Ready` but completions 404 or return empty immediately after.
- **Root cause:** pod readiness can precede model registration. The endpoint is available only after the
  frontend logs `added model ... Completions is ready`.
- **Fix:** wait for that frontend log line before smoke-testing; **don't trust pod `Ready` alone.**

## 7. `DynamoGraphDeployment` applied but nothing happens

- **Symptom:** `kubectl apply` of the DGD succeeds, no prefill/decode pods appear.
- **Root cause:** a DGD is inert without a **Dynamo operator + healthy grove-operator** reconciling
  your namespace (installs are frequently **per-namespace** on shared clusters).
- **Fix:** verify an operator watches your namespace (`dynamo-system` cluster install, or one in your
  ns) and grove is healthy; **or** use the operator-independent path (`serving.yaml` + in-namespace
  `dynamo-platform-etcd` / `-nats`) — the portable, zero-cluster-install variant.

______________________________________________________________________

### Bring-up checklist

1. Operator OR operator-independent etcd/NATS present (§7).
2. Both workers: `--block-size 64`, `--no-disable-hybrid-kv-cache-manager`, **no** `expandable_segments`
   (§1, §3), UCX + `rdma/ib` (§2), matched `--gpu-memory-utilization` sized to real free memory (§4).
3. Wait for the frontend `Completions is ready` log (§6), then smoke **several sequential** prompts
   (not one — §5 only shows up on repeats).
4. If output is intermittently empty → you're at §5 (known issue); escalate image + clean node.

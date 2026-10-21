# GLM-5 GB300 K8s recipes

Dynamo 1.3.0 + SGLang 0.5.14 disaggregated recipes ported from the optimized GB300 Slurm set. All cells use the pinned GLM-5 FP8 revision, AIPerf `/v1/completions`, `ignore_eos:true`, a bounded 0.8–1.0 synthetic length distribution with seed 0, 2× concurrency excluded warmup, and 10× concurrency measured requests.

These GB300 recipes require MNNVL. The cluster profile must declare `NVLINK_MULTICAST_IMEX=provisioned` and provide the ComputeDomain claim template before deployment. LLMB then attaches an IMEX claim to every GPU worker, while disaggregated KV transfer uses NIXL over the RoCE rails declared by the cluster profile. The recipes retain `MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`, and `NCCL_CUMEM_ENABLE=1`; preflight fails before GPU resources are reserved when the required IMEX capability is unavailable. The apply-time IMEX stripping path remains a defensive safeguard for manually bypassed preflight, not a supported benchmark fallback.

The Slurm ladder points are represented as independent K8s cells with matching concurrency and prefill/decode topology. The synthetic token-length parameters match the established GLM-5 B200 K8s recipes for the 1K/1K, 8K/1K, and 16K/512 workload shapes.

| Use case | Goal | Cell | GPUs | Concurrency | P/D nodes |
|---|---|---|---:|---:|---:|
| 1k_1k | hightpt | `glm5-gb300-dyn13-1k1k-ht-c5700-4p14d` | 72 | 5700 | 4P/14D |
| 1k_1k | lowlatency | `glm5-gb300-dyn13-1k1k-ll-c64-1p17d-bs32` | 72 | 64 | 1P/17D |
| 1k_1k | lowlatency | `glm5-gb300-dyn13-1k1k-ll-c128-1p17d-bs32` | 72 | 128 | 1P/17D |
| 1k_1k | lowlatency | `glm5-gb300-dyn13-1k1k-ll-c256-1p17d-bs32` | 72 | 256 | 1P/17D |
| 1k_1k | lowlatency | `glm5-gb300-dyn13-1k1k-ll-c512-1p17d-bs32` | 72 | 512 | 1P/17D |
| 8k_1k | hightpt | `glm5-gb300-dyn13-8k1k-ht-c900-8p10d` | 72 | 900 | 8P/10D |
| 8k_1k | hightpt | `glm5-gb300-dyn13-8k1k-ht-c1300-10p8d` | 72 | 1300 | 10P/8D |
| 8k_1k | hightpt | `glm5-gb300-dyn13-8k1k-ht-c1700-12p6d` | 72 | 1700 | 12P/6D |
| 8k_1k | hightpt | `glm5-gb300-dyn13-8k1k-ht-c2800-14p4d` | 72 | 2800 | 14P/4D |
| 8k_1k | lowlatency | `glm5-gb300-dyn13-8k1k-ll-c32-1p17d-bs8` | 72 | 32 | 1P/17D |
| 8k_1k | lowlatency | `glm5-gb300-dyn13-8k1k-ll-c64-1p17d-bs8` | 72 | 64 | 1P/17D |
| 8k_1k | lowlatency | `glm5-gb300-dyn13-8k1k-ll-c128-1p17d-bs8` | 72 | 128 | 1P/17D |
| 8k_1k | lowlatency | `glm5-gb300-dyn13-8k1k-ll-c150-1p9d-bs15` | 40 | 150 | 1P/9D |
| 16K_512 | hightpt | `glm5-gb300-dyn13-16k512-ht-c900-8p10d` | 72 | 900 | 8P/10D |
| 16K_512 | hightpt | `glm5-gb300-dyn13-16k512-ht-c1300-10p8d` | 72 | 1300 | 10P/8D |
| 16K_512 | hightpt | `glm5-gb300-dyn13-16k512-ht-c1700-12p6d` | 72 | 1700 | 12P/6D |
| 16K_512 | hightpt | `glm5-gb300-dyn13-16k512-ht-c2800-14p4d` | 72 | 2800 | 14P/4D |
| 16K_512 | lowlatency | `glm5-gb300-dyn13-16k512-ll-c32-1p17d-bs8` | 72 | 32 | 1P/17D |
| 16K_512 | lowlatency | `glm5-gb300-dyn13-16k512-ll-c64-1p17d-bs8` | 72 | 64 | 1P/17D |
| 16K_512 | lowlatency | `glm5-gb300-dyn13-16k512-ll-c128-1p17d-bs8` | 72 | 128 | 1P/17D |
| 16K_512 | lowlatency | `glm5-gb300-dyn13-16k512-ll-c150-1p9d-bs15` | 40 | 150 | 1P/9D |

Before serving, prepare and verify the versioned node-local model cache as documented in [`serving/sglang-disagg-gb300/node-cache/README.md`](../../../../serving/sglang-disagg-gb300/node-cache/README.md). Workers mount that revision-specific cache read-only and fail closed if the ready marker, model revision, or 142-shard checkpoint verification does not match.

## AWS GB300 RoCE profile

Start from the GLM-5 profile example rather than reusing a profile generated for another GPU platform:

```bash
cp cluster-profiles/glm5-gb300.env.example cluster-profiles/glm5-gb300.env
```

Fill in the identity, secrets, storage, IMEX, and fabric placeholders. In particular, replace
`RDMA_UCX_NET_DEVICES` with the comma-separated `rocep*` devices present on every target GPU node. The validated
GLM-5 settings for AWS GB300 RoCE are:

```bash
RDMA_UCX_TLS="rc,rc_x,cuda_copy,cuda_ipc"
RDMA_UCX_IB_ADDR_TYPE="eth"
RDMA_UCX_IB_GPU_DIRECT_RDMA="no"
RDMA_UCX_MAX_RNDV_RAILS="8"
```

This recipe is validated with `RDMA_UCX_IB_GPU_DIRECT_RDMA="no"`. Other values, including `try`, are not supported
for this configuration and may prevent prefill-to-decode KV transfer. Validate the completed profile before
installing or running a recipe:

```bash
scripts/llmb-k8s profile validate --cluster glm5-gb300
```

## Run a recipe

Run these commands from `inference/k8s_recipes/`:

```bash
# Install the model dependencies for a cell.
scripts/llmb-k8s install glm5-gb300 --recipes glm5-gb300-dyn13-8k1k-ll-c150-1p9d-bs15

# Populate and verify the versioned NVMe cache on every GB300 node.
# Follow serving/sglang-disagg-gb300/node-cache/README.md and wait for the DaemonSet rollout.

# Run the complete lifecycle and fetch its artifacts.
scripts/llmb-k8s run recipes/llm-perf/Glm5/Gb300_k8s/Disagg/sglang_dynamo/8k_1k/lowlatency/glm5-gb300-dyn13-8k1k-ll-c150-1p9d-bs15 glm5-gb300 --teardown --fetch
```

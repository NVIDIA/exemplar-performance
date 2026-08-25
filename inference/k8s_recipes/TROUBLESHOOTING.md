<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: MIT -->

<!--
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
-->

# Troubleshooting & portability gotchas

Operational guidance for running these recipes on a new cluster. The checks below cover common issues
hit while porting the benchmark across clusters with different namespaces, storage classes, GPU
scheduling models and pull secrets. **Benchmark *findings* live in each cell's `RESULTS.md`; this
file is about getting a run to *happen* at all.**

Everything cluster-specific is a knob in your **cluster profile**
(`cluster-profiles/<cluster>.env` / `.values.yaml`) — see
[`cluster-profiles/README.md`](cluster-profiles/README.md). If you find yourself editing a chart
template, manifest or script to make a run work, stop: it almost certainly belongs in the profile.

______________________________________________________________________

## GPU scheduling: DRA vs device-plugin, `schedulerName`, and finding available nodes

**The gotcha.** Clusters expose GPUs to pods in one of two ways:

- **Device-plugin** — pods request `resources.limits.nvidia.com/gpu: 8`; the NVIDIA device plugin
  advertises a per-node allocatable count.
- **DRA (Dynamic Resource Allocation)** — pods reference `resourceclaims`; GPUs are handed out by a
  DRA driver, not the device plugin.

On a cluster running **both**, the device-plugin `allocatable.nvidia.com/gpu` count **lies**: a node
can report "8 allocatable device-plugin GPUs" while those same physical GPUs are already consumed by
DRA `resourceclaims`. A pod that requests 8 `nvidia.com/gpu` will then either fail to schedule or,
worse, land next to a DRA workload and OOM/collide on the GPUs.

**Finding an available 8-GPU node.** Don't trust `kubectl get nodes -o ...allocatable` alone. Cross-
check against *both* schedulers' bookkeeping:

```sh
# Device-plugin view: allocatable GPUs per node.
kubectl get nodes -o custom-columns=\
'NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'

# Device-plugin GPUs already REQUESTED by running pods on a node:
kubectl get pods -A --field-selector "spec.nodeName=<node>,status.phase=Running" \
  -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}{"\t"}{range .spec.containers[*]}{.resources.requests.nvidia\.com/gpu}{" "}{end}{"\n"}{end}'

# DRA view: resourceclaims already bound on the cluster (GPUs the device-plugin
# count does NOT know are gone). Empty output => no DRA claims to worry about.
kubectl get resourceclaims -A 2>/dev/null
kubectl describe node <node> | grep -A3 -iE 'Allocated resources|nvidia.com/gpu'
```

A node is genuinely free for this recipe only when `allocatable − (sum of running device-plugin GPU requests) ≥ GPU_COUNT` **and** it carries no unrelated DRA claim on those GPUs. `llmb-k8s preflight`
(via `preflight.py`) does this automatically and **is DRA-aware** (it counts DRA-claimed GPUs and warns),
but it can over/under-count — spot-check the DRA half by hand.

**Choosing `schedulerName`.** This is profile-driven; default to your cluster's default scheduler.

- If your cluster runs a gang/queue scheduler (e.g. `kai-scheduler`), be aware some of them **refuse
  to place device-plugin `nvidia.com/gpu` pods onto DRA-enabled nodes**, which can hide otherwise-free
  nodes. In that situation, setting `schedulerName: default-scheduler` on the GPU pods lets the
  device-plugin path schedule onto DRA nodes normally (this is exactly what the experimental
  `serving/dynamo-disagg/` manifests do).
- On a vanilla EKS/GKE/AKS cluster with no custom scheduler, leave `schedulerName` unset (the empty
  default) — the built-in `default-scheduler` is correct.

**Tolerations / nodeSelectors are per-cluster.** GPU pools are frequently tainted
(`dedicated=user-workload`, `kubernetes.io/arch=arm64` on GB200, `nvidia.com/gpu=present`,
autopilot `*-reserved`). The recipe emits a preset toleration union only when
`USE_DYNAMO_TOLERATIONS=true` (a no-op on clusters whose nodes don't carry those taints), plus
optional profile-driven extra tolerations for reserved pools. Vanilla clusters set
`USE_DYNAMO_TOLERATIONS=false`. See `cluster-profiles/_template.env.example`.

______________________________________________________________________

## Connection errors under load (EPERM / ConnectionRefused / connection reset)

**Symptom.** As concurrency climbs, AIPerf starts logging `EPERM`, `Connection refused`, or
`Connection reset by peer` for a growing fraction of requests — even though the server is healthy and
serving other requests fine.

**Root cause.** Naive one-connection-per-request churn against a single **ClusterIP Service VIP**
hammers the node's NAT/conntrack table (the Service VIP is DNAT'd per connection). Under a
high-concurrency sweep the conntrack table fills or ephemeral-port/`SO_REUSE` limits are hit, and new
connections are refused.

**Fixes (in order of preference):**

1. **Use `--connection-reuse-strategy sticky-user-sessions`** (the recipe default, wired via
   `CONNECTION_REUSE_STRATEGY` in `recipe.env` → the bench Job). Each simulated user/session keeps one
   long-lived connection instead of reconnecting per request. This reduces connection churn under load. Set `CONNECTION_REUSE_STRATEGY=""` to disable if you need
   the legacy behaviour.
2. **Address the server by Pod/headless instead of the ClusterIP VIP.** Pointing the bench at the
   server Pod IP (or a headless Service, `clusterIP: None`) skips the per-connection DNAT entirely, so
   conntrack pressure drops. Useful when a single very heavy client saturates one node's NAT table.
3. **Tune node conntrack** if you control the nodes: raise
   `net.netfilter.nf_conntrack_max` and `net.nf_conntrack_max`, and shorten
   `nf_conntrack_tcp_timeout_time_wait`. This is a node/kubelet-level change, not something the recipe
   can set from a pod.

______________________________________________________________________

## Request timeout

`REQUEST_TIMEOUT_SECONDS` controls AIPerf's per-request timeout. Set it high enough for the configured context lengths and expected load; requests that exceed it are reported as client timeouts.

______________________________________________________________________

## GPU telemetry `gpu_stats.csv` is empty (header only)

**Symptom.** A run's `gpu_stats.csv` is 114 bytes — just the header row — so there is no per-GPU
power/util/mem/clocks time-series and you cannot compute perf/watt.

**Why.** The CPU-only bench Pod cannot read GPU counters directly. It collects telemetry by running `nvidia-smi` in the server Pod. If the server Pod cannot be resolved or the Kubernetes command fails, the metrics endpoint may still be available while `gpu_stats.csv` contains only its header.

**Fix (already in the recipe).**

- `resolve_server_pod` now falls back to the **Service Endpoints** (`targetRef.name`) when the label
  selector matches nothing — guaranteed consistent with where `/metrics` curl traffic lands.
- The in-cluster `kube()` wrapper **brackets bare IPv6 service IPs** and falls back to plain in-cluster
  auto-config, so kubectl works on dual-stack clusters.
- The `nvidia-smi` exec no longer hard-codes `-c sglang`; it tries `SERVER_CONTAINER`, then the Pod's
  default container.
- A one-shot **`telemetry_preflight`** runs before the sweep and writes `gpu_stats.status` (and logs to
  the Job) — `gpu_telemetry=ok … gpus=N` or `gpu_telemetry=EMPTY …` with the likely cause — so an empty
  capture is caught immediately, not after the run.

**Knobs** (`recipe.env` / chart `bench.gpuTelemetry`):

- `GPU_TELEMETRY` = `auto` (default) | `exec` | `dcgm` | `off`.
- `SERVER_CONTAINER` = server container name to exec into (default `sglang`).
- `DCGM_EXPORTER_URL` = your cluster's DCGM-exporter Prometheus endpoint. Set this on clusters that **forbid exec**
  into application containers; the scrape maps `DCGM_FI_DEV_POWER_USAGE` / `_GPU_UTIL` / `_FB_USED` /
  `_SM_CLOCK` onto the same `gpu_stats.csv` schema (timestamp + memory-total left empty). `auto` prefers
  DCGM when this URL is set, else falls back to exec.

**Computing perf/watt.** Once `gpu_stats.csv` has rows, mean total power for a rung is the per-tick sum
of `power_draw_w` across all GPUs, averaged over ticks; **perf/watt = decode `output_token_throughput` ÷
mean total power (W)** (tokens/s per watt). Since the bench is ~99% prefill (see `ANALYSIS.md`), also
report throughput-per-watt against total tokens if you want a prefill-inclusive efficiency number.

______________________________________________________________________

## Slow first server start / model load

**Symptom.** The server Deployment sits `NotReady` for 15–20 minutes on first start; probes look like
they're failing.

**Why.** This is expected for a 550B NVFP4 model. First start = a ~10 GB image pull (if not cached on
the node) + ~60 GB of weights loaded across 8 GPUs + KV/hicache host-RAM allocation. The chart sizes
the probes and `progressDeadlineSeconds` for this (readiness `failureThreshold` allows ~31 min;
liveness holds off ~30 min). **Do not shorten these** unless your node loads faster — an aggressive
liveness probe will kill the pod mid-load and loop forever.

- Speed up repeat starts by pre-pulling the image onto the GPU node and keeping the model in the
  shared cache.
- The disaggregated backend loads the model **per worker** (2 prefill + 1 decode = 3 loads), so budget
  ~15–20 min × workers before it's Ready.

**Artifacts PVC must tolerate a long-running writer.** The bench Job writes to the RWO artifacts PVC
for the whole multi-hour sweep. Use a storage class with `reclaimPolicy=Delete` (so per-run PVs
auto-clean) — check this by hand (`kubectl get storageclass -o custom-columns=NAME:.metadata.name,RECLAIM:.reclaimPolicy`);
preflight does **not** verify reclaimPolicy. The **shared model+dataset cache** must be
**ReadWriteMany** (server + bench mount it read-only, download Jobs read-write): pick your cloud's RWX
filesystem (FSx/EFS on AWS, Filestore on GCP, Azure Files on AKS, Ceph/NFS on-prem).

______________________________________________________________________

## Experimental: the disaggregated Dynamo backend (`serving/dynamo-disagg/`)

**This backend is experimental and not driven by the shared `scripts/` harness** — you render it and
apply it with `kubectl` directly, then point the bench at its frontend Service. Two paths:

- **Operator path** (`dgd-disagg-1m.yaml`, a `DynamoGraphDeployment`) requires the **Dynamo operator**
  and a **healthy `grove-operator`** in the cluster. If grove is `CrashLoopBackOff` (e.g. a CRD
  shortName conflict) the graph never materializes — that's a platform issue, not a recipe defect.
- **Operator-independent path** (`runtime.yaml` + `serving.yaml`) brings up etcd + NATS + the
  frontend/prefill/decode Deployments as plain namespaced resources, bypassing grove entirely. Use
  this when the operator/grove stack is unavailable.

Both need **3 free 8-GPU nodes** (2 prefill + 1 decode) and are subject to the same DRA/scheduler
gotchas above — the manifests set `schedulerName: default-scheduler` for exactly that reason. See
[`serving/dynamo-disagg/README.md`](serving/dynamo-disagg/README.md) for the required per-cluster
edits (namespace, pull secret, node hostnames, cache PVC).

______________________________________________________________________

## "My run was halted" — reading the hang short-circuit

The governor checks both token progress and active requests. It does not stop a run merely because token output pauses between sweep stages, during warmup, or while artifacts are written.

**The one signal that halts a run is a conjunction:**

> no output-token progress **AND** requests still outstanding at the server — *both* sustained.

In a healthy gap between rungs nothing is in flight, so the second half is false and nothing accumulates.
If either half is *unmeasurable* (a `/metrics` blip, an engine that exports no in-flight gauge, a run
submitted before this existed) the answer is UNKNOWN, and UNKNOWN never halts anything.

### See it without reading code

```
scripts/llmb-k8s status <run-id> --cluster <profile>
```

```
  status.json state=running phase=generating reason= progress=198084 ...
  work        inflight=15 queued=0 tokens_last_advanced=2026-…T04:11:02Z
              work_outstanding_since=2026-…T04:11:02Z metrics=ok
  governor    action=halt reason=stalled
              detail=no output tokens for 3641s while 15 request(s) stayed outstanding for 3641s
                     (queued=0, tokens=198084, metrics=ok); over halt threshold 3600s
                     (=2x STALL_THRESHOLD 1800s)   (AUTHORITATIVE halt reason)
```

| line                     | what to read                                                                                                                                                   |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `work inflight=`         | requests the server is holding. `unknown` = unmeasurable → the halt is **disarmed**                                                                            |
| `work queued=`           | requests behind them (recorded for context; does not gate the halt)                                                                                            |
| `tokens_last_advanced`   | when the **monotonic high-water** token counter last moved. A counter reset (frontend/worker restart) can never move it backwards, so a restart is not a stall |
| `work_outstanding_since` | start of the *continuous* window during which work has been outstanding. Any idle poll **or any unreadable poll** resets it                                    |
| `metrics=`               | `ok` / `metrics-unreachable` / `metrics-no-token-counter` — UNKNOWN is stated, never silently read as zero                                                     |
| `governor action=`       | `warn` = early warning, **nothing was deleted**; `halt` = the run was deleted, for the reason printed                                                          |

### Timeline

| elapsed with **both** halves true   | what happens                                                     |
| ----------------------------------- | ---------------------------------------------------------------- |
| `STALL_THRESHOLD` (default 30m)     | `governor.json` `action=warn`, logged. **Nothing is deleted.**   |
| `2 x STALL_THRESHOLD` (default 60m) | the bench Job is deleted, `action=halt`, with the evidence above |

Tune with `STALL_THRESHOLD` / `STALL_HALT_MULT` on the governor CronJob.

### It did *not* halt and you think it should have

The governor logs its reasoning either way — a no-halt is never silent:

```
kubectl -n <ns> logs job/llmb-governor-<id>
no-halt run=<id>: token progress stale 4200s BUT no sustained outstanding work
                  (inflight=0 work_age=5s queued=0 metrics=ok) -- staleness alone is not a stall
```

`inflight=0` here means the server genuinely had nothing in flight — a rung boundary or finalisation,
not a hang. `inflight=-1` means `/metrics` could not be read, so there was no evidence to act on.

One more line worth grepping for:

```
UNSUPERVISED run=<id>: status.json is not valid JSON -> no stall/timeout supervision this pass
```

That run is receiving **no** stall or timeout supervision. Every field the wrapper writes is JSON-escaped,
so this should be impossible; if you see it, capture `/control/<run-id>/status.json` before anything
overwrites it.

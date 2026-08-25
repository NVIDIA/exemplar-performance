#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa

"""probe_fabric.py <cluster-profile> [--write] — discover live fabric facts from GPU nodes.

Queries GPU nodes in the cluster (using KUBE_CONTEXT + GPU_PRODUCT from the profile) and
discovers fabric configuration that belongs in the cluster profile, not the recipe:

  RDMA_UCX_NET_DEVICES     — UCX device list built from rdma/* extended resources in node capacity
  RDMA_UCX_MAX_RNDV_RAILS  — number of IB rails (count of rdma/* devices per node)
  RDMA_UCX_IB_ADDR_TYPE    — empty for native InfiniBand, "eth" for RoCE
                             (GID-table exec probe; falls back to label heuristic)
  RDMA_NODE_SELECTOR       — nodeSelector label(s) that identify RDMA-capable nodes
  RDMA_RESOURCE            — extended resource name the RDMA device plugin exposes
  RDMA_FABRIC_PROBED       — sentinel written on every --write so run.sh doesn't re-probe

Without --write (default / dry run): prints discovered values to stdout only.
With --write: patches the cluster profile file in-place by appending a

  # --- fabric (auto-discovered by probe-fabric) ---

block that overrides whatever RDMA_* vars were previously in the profile.

After writing, re-render any disagg cells that had ucx.net_devices baked in the recipe so the
template emits ${RDMA_UCX_NET_DEVICES} from the profile instead:

  scripts/render.sh <cell-dir>

Exit 0 on success, 1 on discovery failure, 2 on usage error.
"""

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_POD_SEQ = itertools.count(1)  # unique suffix so a clique sweep never reuses a pod name

ROOT = Path(__file__).resolve().parent.parent
_KUBE_CONTEXT = ""

# GPU-scheduled NVLink-P2P probe (nvidia-smi image; the busybox sys-pod has no nvidia-smi). Overridable
# from the profile via P2P_PROBE_IMAGE / P2P_PROBE_GPUS; the digest is the arm64 vllm-runtime the recipes
# already pull, so it is warm on the GPU nodes.
P2P_PROBE_IMAGE = (
    "nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:" "9fefbd10b63c52030326e22ed0eb9d87082fee53709fe327b5b2773e267792bc"
)
P2P_PROBE_GPUS = 2  # 2 GPUs is the minimum to observe an off-diagonal P2P pair
_P2P_NS_FAMILY = {
    "NS",
    "CNS",
    "GNS",
    "TNS",
}  # topo -p2p rw "not supported" tokens (P2P disabled)


def _kubectl(args, timeout=30):
    ctx = ["--context", _KUBE_CONTEXT] if _KUBE_CONTEXT else []
    try:
        p = subprocess.run(
            ["kubectl", *ctx, "--request-timeout=25s", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def parse_env(path):
    out = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'"):
            q = v[0]
            j = v.find(q, 1)
            v = v[1:j] if j != -1 else v[1:]
        else:
            v = re.split(r"\s+#", v, 1)[0].strip()
        out[k.strip()] = v
    return out


_RDMA_LABEL_RE = re.compile(r"[A-Za-z0-9][\w./-]*(?:rdma|infiniband)[\w./-]*$", re.I)
_ROCE_LABEL_RE = re.compile(r"roce", re.I)


def fetch_gpu_nodes(gpu_product):
    """Return (nodes_list, error_str). error_str is None on success."""
    rc, out, err = _kubectl(["get", "nodes", "-l", f"nvidia.com/gpu.product={gpu_product}", "-o", "json"])
    if rc != 0 or not out:
        return None, err.strip() or "kubectl returned no output"
    return json.loads(out).get("items", []), None


def discover_rdma_devices(nodes):
    """Read rdma/* keys from node.status.capacity.

    Returns:
      per_node      {node_name: [device_names]} — only nodes that have rdma/* resources
      resource_key  the rdma/* key used by the device plugin (e.g. 'rdma/ib' or 'rdma/mlx5_0')
    """
    per_node = {}
    resource_key = None
    for node in nodes:
        name = node["metadata"]["name"]
        capacity = (node.get("status") or {}).get("capacity") or {}
        devs = []
        for key in sorted(capacity):
            if not key.startswith("rdma/"):
                continue
            devs.append(key[len("rdma/") :])
            if resource_key is None or key == "rdma/ib":
                resource_key = key
        if devs:
            per_node[name] = devs
    return per_node, resource_key


def discover_rdma_labels(nodes):
    """Label key=value pairs present on ALL gpu nodes whose key matches rdma/infiniband."""
    n = len(nodes)
    counts = {}
    for node in nodes:
        labels = (node.get("metadata") or {}).get("labels") or {}
        for k, v in labels.items():
            if _RDMA_LABEL_RE.search(k) and v.strip().lower() in ("true", "1", "yes"):
                pair = f"{k}={v}"
                counts[pair] = counts.get(pair, 0) + 1
    return sorted(kv for kv, c in counts.items() if c == n)


def detect_roce_labels(nodes):
    """Heuristic fallback: any label KEY containing 'roce' on any GPU node → RoCE fabric."""
    return any(_ROCE_LABEL_RE.search(k) for node in nodes for k in ((node.get("metadata") or {}).get("labels") or {}))


def build_ucx_net_devices(per_node):
    """Most-common device list across nodes → 'mlx5_0:1,mlx5_1:1,...' UCX device string."""
    if not per_node:
        return None
    from collections import Counter

    counts = Counter(tuple(devs) for devs in per_node.values())
    devs = counts.most_common(1)[0][0]
    return ",".join(f"{d}:1" for d in devs)


def _spawn_sys_pod(ns, node_name):
    """Spawn a busybox pod on node_name with /sys mounted read-only.

    Returns the pod name on success, or None if the pod failed to become ready.
    The caller is responsible for deleting the pod in a finally block.
    """
    pod = f"llmb-fabric-probe-{os.getpid()}"
    overrides = json.dumps(
        {
            "spec": {
                "restartPolicy": "Never",
                "nodeName": node_name,
                "tolerations": [{"operator": "Exists"}],
                "volumes": [{"name": "sys", "hostPath": {"path": "/sys", "type": "Directory"}}],
                "containers": [
                    {
                        "name": "p",
                        "image": "busybox:1.36",
                        "command": ["sleep", "120"],
                        "volumeMounts": [{"name": "sys", "mountPath": "/sys", "readOnly": True}],
                    }
                ],
            }
        }
    )
    _kubectl(
        [
            "-n",
            ns,
            "run",
            pod,
            "--image=busybox:1.36",
            "--restart=Never",
            f"--overrides={overrides}",
        ],
        timeout=30,
    )
    rc, _, _ = _kubectl(
        ["-n", ns, "wait", "pod", pod, "--for=condition=ready", "--timeout=90s"],
        timeout=100,
    )
    return pod if rc == 0 else None


def probe_sys_ib(ns, node_name, dev_name=None):
    """Spawn a /sys probe pod on node_name and discover IB device names + fabric type.

    When dev_name is None, lists all devices under /sys/class/infiniband/ first.
    Returns a dict with keys "devices" (sorted list) and "fabric" ("ib"|"roce"|None).

    Uses link_layer (not gid_attrs/types) for fabric detection — link_layer is
    unambiguous ("InfiniBand" or "Ethernet"), whereas gid_attrs/types contains
    "IB/RoCE v1" on native-IB ConnectX adapters, causing false RoCE detection.
    """
    pod = _spawn_sys_pod(ns, node_name)
    if pod is None:
        return {"devices": [], "fabric": None}
    try:
        # Enumerate all IB devices on the node (works even without device plugin)
        rc, out, _ = _kubectl(
            [
                "-n",
                ns,
                "exec",
                pod,
                "--",
                "sh",
                "-c",
                "ls /sys/class/infiniband/ 2>/dev/null || echo NONE",
            ],
            timeout=15,
        )
        if rc != 0 or out.strip() == "NONE" or not out.strip():
            return {"devices": [], "fabric": None}
        devices = sorted(out.strip().splitlines())

        # Probe link_layer for the first device (or the one requested)
        probe_dev = dev_name or (devices[0] if devices else None)
        fabric = None
        if probe_dev and probe_dev in devices:
            rc, ll, _ = _kubectl(
                [
                    "-n",
                    ns,
                    "exec",
                    pod,
                    "--",
                    "sh",
                    "-c",
                    f"cat /sys/class/infiniband/{probe_dev}/ports/*/link_layer" f" 2>/dev/null || echo NONE",
                ],
                timeout=15,
            )
            if rc == 0 and ll.strip() != "NONE":
                lower = ll.lower()
                if "ethernet" in lower:
                    fabric = "roce"
                elif "infiniband" in lower:
                    fabric = "ib"

        return {"devices": devices, "fabric": fabric}
    finally:
        _kubectl(
            ["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--wait=false"],
            timeout=20,
        )


# ── NVLink P2P fabric-health probe ───────────────────────────────────────────
# Convert `nvidia-smi` topology and fabric-health output into a profile fact.
def parse_p2p_matrix(text):
    """PURE. `nvidia-smi topo -p2p rw` prints a GPU×GPU matrix: X=self, OK=P2P works,
    NS/CNS/GNS/TNS=not supported, U=unknown. Return 'healthy' (OK off-diagonal), 'disabled' (any NS-family
    off-diagonal — the fabric-wide-off signature), or 'unknown' (nothing parseable)."""
    if not text:
        return "unknown"
    ok = ns = 0
    for line in text.splitlines():
        toks = line.split()
        if not toks or not re.match(r"^GPU\d+$", toks[0]):
            continue  # data rows start with a GPUn label; header/legend rows don't
        for cell in toks[1:]:
            c = cell.upper()
            if c == "OK":
                ok += 1
            elif c in _P2P_NS_FAMILY:
                ns += 1
            # X (self) and U (unknown) are ignored
    if ns > 0:
        return "disabled"
    if ok > 0:
        return "healthy"
    return "unknown"


def parse_fabric_health(text):
    """PURE. Parse the per-GPU `Fabric` block of `nvidia-smi -q` →
    {state, summary, route_unhealthy, route_recovery}. Healthy pools read Summary=Healthy /
    Route Unhealthy=False; the smoking guns are Summary=Unhealthy and Route Unhealthy=True (a cluster-admin
    fabric-manager route-recovery problem). `route_recovery` (`Route Recovery in progress`) + `state`
    identify a TRANSIENT recovery window that classify_p2p must NOT cache as a hard `disabled` (Minor B).
    """
    out = {
        "state": None,
        "summary": None,
        "route_unhealthy": None,
        "route_recovery": None,
    }
    if not text:
        return out
    for line in text.splitlines():
        s = line.strip()
        m = re.match(
            r"^State\s*:\s*(Completed|In Progress|Not Started|Failed|Standby|Skipped)\b",
            s,
        )
        if m and out["state"] is None:
            out["state"] = m.group(1)
        m = re.match(r"^Summary\s*:\s*(Healthy|Unhealthy)\b", s)
        if m and out["summary"] is None:
            out["summary"] = m.group(1)
        m = re.match(r"^Route Recovery in progress\s*:\s*(\S+)", s)
        if m and out["route_recovery"] is None:
            out["route_recovery"] = m.group(1).strip().lower() == "true"
        m = re.match(r"^Route Unhealthy\s*:\s*(\S+)", s)
        if m and out["route_unhealthy"] is None:
            out["route_unhealthy"] = m.group(1).strip().lower() == "true"
    return out


def classify_p2p(matrix_state, health):
    """PURE. Fuse the topo matrix verdict + Fabric Health block into healthy|disabled|unknown.

    Minor B — require CORROBORATION before declaring `disabled`, so a transient fabric-manager
    route-recovery window (State: In Progress / `Route Recovery in progress: True`, momentarily Unhealthy)
    with an OK topo matrix is NOT cached as a hard `disabled`:
      • the P2P topo matrix showing NS off-diagonal is the REAL, unambiguous signal → `disabled` outright.
      • a bad Fabric Health block (Summary=Unhealthy / Route Unhealthy) → `disabled` ONLY when NOT mid
        route-recovery; a recovering window with a non-NS matrix is ambiguous → `unknown` (not disabled).
    """
    health = health or {}
    summary = health.get("summary")
    route_unhealthy = health.get("route_unhealthy")
    recovering = (health.get("state") == "In Progress") or (health.get("route_recovery") is True)
    health_bad = summary == "Unhealthy" or route_unhealthy is True
    if matrix_state == "disabled":
        return "disabled"  # NS off-diagonal — the genuine fabric-off signal, even mid-recovery
    if health_bad:
        # corroboration gate: trust the unhealthy block only when the fabric is not actively recovering.
        return "unknown" if recovering else "disabled"
    if matrix_state == "healthy":
        return "healthy"
    if summary == "Healthy" or route_unhealthy is False:
        return "healthy"  # matrix un-parseable but the fabric block affirms health
    return "unknown"


def _pod_not_ready_reason(ns, pod):
    """Best-effort ONE-LINE reason a probe pod never became Ready — turns an opaque `unknown` into a
    diagnostic. A single `kubectl get pod -o json`; never raises. Surfaces the scheduling/admission/pull
    signals that actually explain a stuck probe:
      • Pending + PodScheduled=False → `Unschedulable: Insufficient nvidia.com/gpu` (pool saturated)
      • Failed + UnexpectedAdmissionError → kubelet rejected an over-committed nodeName pin (devices
        unavailable) — the exact bug this probe used to hit
      • waiting=ImagePullBackOff / ErrImagePull → the vllm image couldn't be pulled (bad pull secret).
    """
    rc, out, _ = _kubectl(["-n", ns, "get", "pod", pod, "-o", "json"], timeout=20)
    if rc != 0 or not out:
        return "pod not found (never created?)"
    try:
        st = json.loads(out).get("status") or {}
    except Exception:
        return "pod status unparseable"
    bits = []
    if st.get("phase"):
        bits.append(f"phase={st['phase']}")
    if st.get("reason"):
        bits.append(f"reason={st['reason']}")
    if st.get("message"):
        bits.append(st["message"].strip().splitlines()[0][:200])
    for c in st.get("conditions") or []:  # Pending → PodScheduled=False carries Unschedulable
        if c.get("type") == "PodScheduled" and c.get("status") != "True":
            if c.get("reason"):
                bits.append(f"PodScheduled={c['reason']}")
            if c.get("message"):
                bits.append(c["message"].strip().splitlines()[0][:200])
    for cs in st.get("containerStatuses") or []:  # ImagePullBackOff / CreateContainerError / ...
        w = (cs.get("state") or {}).get("waiting") or {}
        if w.get("reason"):
            bits.append(f"waiting={w['reason']}")
        if w.get("message"):
            bits.append(w["message"].strip().splitlines()[0][:200])
    # dedupe while preserving order
    return "; ".join(dict.fromkeys(bits)) or "not ready (no reason reported)"


def _spawn_gpu_pod(
    ns,
    gpu_product,
    image,
    pull_secret=None,
    gpu_count=P2P_PROBE_GPUS,
    node_name=None,
    clique=None,
    ready_timeout=300,
):
    """Spawn a GPU-scheduled pod (requesting `gpu_count` GPUs) with an nvidia-smi-bearing image.

    DEFAULT = SCHEDULER PLACEMENT: constrain with nodeSelector {nvidia.com/gpu.product: gpu_product}
    (+ nvidia.com/gpu.clique when `clique` is given — see below) + the GPU resource request +
    tolerations, and let the k8s scheduler pick ANY node that has `gpu_count` FREE GPUs. A single pod's
    GPUs are co-located on one node, so P2P is still validly testable.

    Scheduler placement is the default so resource availability is respected.
    Use `node_name` only as a diagnostic override.

    `clique` pins the pod to one NVLink fabric domain (nvidia.com/gpu.clique). NVLink-P2P/Route health is
    a PER-CLIQUE property on GB200/GB300 NVL domains — a fault can sit in one clique while another is
    healthy — so the probe sweeps each clique separately (see probe_nvlink_p2p).

    Returns (pod_name, reason): reason is None once Ready; on failure the pod_name is still returned (so
    the caller can delete it) and reason is a one-line diagnostic from `_pod_not_ready_reason`.
    """
    # Unique per spawn: a clique sweep deletes with --wait=false, so a same-named pod could still be
    # Terminating when the next clique's pod is created → AlreadyExists. The seq suffix avoids that.
    pod = f"llmb-p2p-probe-{os.getpid()}-{next(_POD_SEQ)}"
    spec = {
        "restartPolicy": "Never",
        "tolerations": [{"operator": "Exists"}],
        "containers": [
            {
                "name": "p",
                "image": image,
                "command": ["sleep", "300"],
                "resources": {"limits": {"nvidia.com/gpu": str(gpu_count)}},
            }
        ],
    }
    if node_name:  # explicit override: force one node (scheduler-bypassing)
        spec["nodeName"] = node_name
    else:
        selector = {}
        if gpu_product:  # default: let the scheduler find a free-GPU node
            selector["nvidia.com/gpu.product"] = gpu_product
        if clique:  # constrain to one NVLink fabric domain
            selector["nvidia.com/gpu.clique"] = clique
        if selector:
            spec["nodeSelector"] = selector
    if pull_secret:
        spec["imagePullSecrets"] = [{"name": pull_secret}]
    _kubectl(
        [
            "-n",
            ns,
            "run",
            pod,
            "--image",
            image,
            "--restart=Never",
            f"--overrides={json.dumps({'spec': spec})}",
        ],
        timeout=30,
    )
    # Generous wait: the ~10GB vllm image can be cold-pulled on a fresh node (~180-300s).
    rc, _, _ = _kubectl(
        [
            "-n",
            ns,
            "wait",
            "pod",
            pod,
            "--for=condition=ready",
            f"--timeout={ready_timeout}s",
        ],
        timeout=ready_timeout + 20,
    )
    if rc == 0:
        return pod, None
    return pod, _pod_not_ready_reason(ns, pod)


def _read_pod_p2p(ns, pod):
    """Exec the two nvidia-smi reads inside a Ready probe pod and classify. PURE-of-cluster once the pod
    exists. Returns {state, matrix_state, summary, route_unhealthy[, error]}."""
    unknown = {
        "state": "unknown",
        "matrix_state": "unknown",
        "summary": None,
        "route_unhealthy": None,
    }
    try:
        rc1, topo, _ = _kubectl(
            ["-n", ns, "exec", pod, "--", "nvidia-smi", "topo", "-p2p", "rw"],
            timeout=40,
        )
        rc2, q, _ = _kubectl(["-n", ns, "exec", pod, "--", "nvidia-smi", "-q"], timeout=40)
        matrix_state = parse_p2p_matrix(topo) if rc1 == 0 else "unknown"
        health = parse_fabric_health(q) if rc2 == 0 else {}
        return {
            "state": classify_p2p(matrix_state, health),
            "matrix_state": matrix_state,
            "summary": health.get("summary"),
            "route_unhealthy": health.get("route_unhealthy"),
        }
    except Exception as e:
        return dict(unknown, error=str(e))


def _used_gpus_by_node():
    """Best-effort {node_name: gpus_requested} across all live pods. Empty on any error (e.g. no
    cluster-wide `get pods` RBAC) — the caller then treats free-GPU counts as unknown and simply probes
    every clique, safe-degrading rather than skipping a fabric domain."""
    used = {}
    rc, out, _ = _kubectl(["get", "pods", "-A", "-o", "json"], timeout=60)
    if rc != 0 or not out:
        return used
    try:
        items = json.loads(out).get("items", [])
    except Exception:
        return used
    for p in items:
        if (p.get("status") or {}).get("phase") in ("Succeeded", "Failed"):
            continue
        node = (p.get("spec") or {}).get("nodeName")
        if not node:
            continue
        tot = 0
        for c in (p.get("spec") or {}).get("containers", []):
            g = ((c.get("resources") or {}).get("limits") or {}).get("nvidia.com/gpu")
            if g is not None:
                try:
                    tot += int(g)
                except (TypeError, ValueError):
                    pass
        if tot:
            used[node] = used.get(node, 0) + tot
    return used


def cliques_with_capacity(gpu_product, gpu_count):
    """Group GPU_PRODUCT nodes by NVLink clique (nvidia.com/gpu.clique label) and split by whether the
    clique has a node with ≥ gpu_count FREE GPUs (schedulable) or not (saturated). A node with no clique
    label lands under the sentinel key None (single-domain clusters / no NVL fabric labels).

    Returns (probeable, saturated, error):
      probeable  list of clique labels (or None) with schedulable capacity
      saturated  list of (clique, max_free) that cannot host the probe right now
      error      set only when the node list itself couldn't be fetched (→ caller returns unknown).
    """
    nodes, err = fetch_gpu_nodes(gpu_product)
    if nodes is None:
        return None, None, f"cannot query nodes: {err}"
    if not nodes:
        return None, None, f"no nodes labelled nvidia.com/gpu.product={gpu_product}"
    used = _used_gpus_by_node()
    max_free = {}
    for n in nodes:
        name = n["metadata"]["name"]
        clq = ((n["metadata"].get("labels") or {}).get("nvidia.com/gpu.clique")) or None
        alloc = int(((n.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0) or 0)
        # When usage is unknown (used empty), free==alloc → we optimistically probe the clique.
        free = alloc - used.get(name, 0)
        if clq not in max_free or free > max_free[clq]:
            max_free[clq] = free
    probeable = [c for c, f in max_free.items() if f >= gpu_count]
    saturated = [(c, f) for c, f in max_free.items() if f < gpu_count]
    return probeable, saturated, None


def aggregate_clique_results(results):
    """PURE. Fuse per-clique probe dicts into one NVLINK_P2P verdict. Any clique `disabled` → the whole
    pool is `disabled` (a run could land on that fabric domain); else any `healthy` → `healthy` (with a
    note listing cliques we could not verify); else `unknown`. Always carries `per_clique` for diagnosis.
    """
    unknown = {
        "state": "unknown",
        "matrix_state": "unknown",
        "summary": None,
        "route_unhealthy": None,
    }
    disabled = [r for r in results if r.get("state") == "disabled"]
    if disabled:
        out = dict(disabled[0], per_clique=results)
        out["error"] = "P2P-disabled clique(s): " + ", ".join(str(d.get("clique")) for d in disabled)
        return out
    healthy = [r for r in results if r.get("state") == "healthy"]
    if healthy:
        out = dict(healthy[0], per_clique=results)
        unverified = [r for r in results if r.get("state") == "unknown"]
        if unverified:
            out["error"] = "healthy on probed clique(s); UNVERIFIED: " + "; ".join(
                f"{r.get('clique')}: {r.get('error')}" for r in unverified
            )
        return out
    out = dict(unknown, per_clique=results)
    out["error"] = "; ".join(f"{r.get('clique')}: {r.get('error')}" for r in results) or "no clique probeable"
    return out


def probe_nvlink_p2p(
    ns,
    gpu_product,
    image=P2P_PROBE_IMAGE,
    pull_secret=None,
    gpu_count=P2P_PROBE_GPUS,
    node_name=None,
    ready_timeout=300,
):
    """OPTIONAL / non-fatal. Detect live NVLink-P2P fabric health via a scheduler-placed GPU probe pod.
    ANY error (no free-GPU node, no nvidia-smi, exec failure) → state 'unknown' (safe-degrade) with a
    DIAGNOSTIC `error` explaining WHY (Unschedulable / admission / ImagePull / saturated clique), never
    a crash. Returns {state, matrix_state, summary, route_unhealthy[, error, per_clique]}.

    P2P health is evaluated per reachable NVLink clique. A clique without enough free GPUs
    remains `unknown` rather than being reported healthy."""
    unknown = {
        "state": "unknown",
        "matrix_state": "unknown",
        "summary": None,
        "route_unhealthy": None,
    }

    # Diagnostic override: force a single node, no clique sweep.
    if node_name:
        pod, reason = _spawn_gpu_pod(
            ns,
            gpu_product,
            image,
            pull_secret,
            gpu_count,
            node_name=node_name,
            ready_timeout=ready_timeout,
        )
        if reason is not None:
            _kubectl(
                ["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--wait=false"],
                timeout=20,
            )
            return dict(unknown, error=f"probe pod not ready — {reason}")
        try:
            return _read_pod_p2p(ns, pod)
        finally:
            _kubectl(
                ["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--wait=false"],
                timeout=20,
            )

    probeable, saturated, err = cliques_with_capacity(gpu_product, gpu_count)
    if err:
        return dict(unknown, error=err)

    results = []
    for clq in probeable:
        pod, reason = _spawn_gpu_pod(
            ns,
            gpu_product,
            image,
            pull_secret,
            gpu_count,
            clique=clq,
            ready_timeout=ready_timeout,
        )
        if reason is not None:
            _kubectl(
                ["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--wait=false"],
                timeout=20,
            )
            results.append(dict(unknown, clique=clq, error=f"probe pod not ready — {reason}"))
            continue
        try:
            r = _read_pod_p2p(ns, pod)
        finally:
            _kubectl(
                ["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--wait=false"],
                timeout=20,
            )
        r["clique"] = clq
        results.append(r)
        if r.get("state") == "disabled":
            break  # a disabled clique is decisive — stop sweeping
    for clq, free in saturated:
        results.append(
            dict(
                unknown,
                clique=clq,
                error=f"clique saturated (max free {free} < {gpu_count} GPU) — cannot verify",
            )
        )
    if not results:
        return dict(unknown, error="no GPU cliques found to probe")
    return aggregate_clique_results(results)


def patch_profile(path, facts):
    """Append auto-discovered block to the profile, replacing any previous probe-fabric block."""
    text = Path(path).read_text()
    # \n? handles a block written at byte-0 with no preceding newline
    text = re.sub(
        r"\n?# --- fabric \(auto-discovered by probe-fabric\) ---.*",
        "",
        text,
        flags=re.DOTALL,
    )
    text = text.rstrip("\n") + "\n"
    block = ["\n# --- fabric (auto-discovered by probe-fabric) ---"]
    for k, v in facts.items():
        block.append(f'{k}="{v}"')
    Path(path).write_text(text + "\n".join(block) + "\n")


def main():
    argv = sys.argv[1:]
    write = "--write" in argv
    pos = [a for a in argv if not a.startswith("--")]
    if len(pos) != 1:
        sys.exit(__doc__)

    profile_name = pos[0]
    prof_path = ROOT / "cluster-profiles" / f"{profile_name}.env"
    if not prof_path.exists():
        sys.exit(f"probe-fabric: no profile at {prof_path}")

    prof = parse_env(prof_path)
    global _KUBE_CONTEXT
    _KUBE_CONTEXT = (prof.get("KUBE_CONTEXT") or "").strip()
    gpu_product = (prof.get("GPU_PRODUCT") or "").strip()
    ns = (prof.get("NAMESPACE") or "").strip()
    if not gpu_product:
        sys.exit(f"probe-fabric: GPU_PRODUCT not set in {profile_name}.env")

    ctx_label = _KUBE_CONTEXT or "ambient"
    print(f"probe-fabric: {profile_name}  (GPU_PRODUCT={gpu_product}, context={ctx_label})")

    nodes, err = fetch_gpu_nodes(gpu_product)
    if nodes is None:
        sys.exit(f"  ❌  cannot query nodes: {err}")
    if not nodes:
        sys.exit(f"  ❌  no nodes labelled nvidia.com/gpu.product={gpu_product}")
    print(f"  {len(nodes)} GPU node(s) found")
    print()

    per_node, resource_key = discover_rdma_devices(nodes)
    rdma_labels = discover_rdma_labels(nodes)
    label_roce = detect_roce_labels(nodes)
    facts = {}

    # --- RDMA devices: device-plugin path (rdma/* node capacity) ---
    # When the RDMA device plugin is running it exposes rdma/mlx5_N: 1000 in
    # node capacity — gives us device names without needing a pod exec.
    if per_node:
        for node_name, devs in sorted(per_node.items()):
            print(f"  rdma devices  {node_name}: {devs}")
        ucx_str = build_ucx_net_devices(per_node)
        if ucx_str:
            facts["RDMA_UCX_NET_DEVICES"] = ucx_str
        if resource_key:
            facts["RDMA_RESOURCE"] = resource_key
        from collections import Counter as _Counter

        _layout_counts = _Counter(tuple(devs) for devs in per_node.values())
        most_common_devs = _layout_counts.most_common(1)[0][0]
        rail_count = len(most_common_devs)
        facts["RDMA_UCX_MAX_RNDV_RAILS"] = str(rail_count)
        print(f"  rail count    {rail_count} device(s) → UCX_MAX_RNDV_RAILS={rail_count}")
    else:
        print("  ⚠️  no rdma/* extended resources found on GPU nodes")
        print("     (RDMA device plugin not running — falling back to /sys probe)")

    # --- RDMA node selector labels ---
    if rdma_labels:
        print(f"  rdma labels   all nodes: {rdma_labels}")
        facts["RDMA_NODE_SELECTOR"] = ",".join(rdma_labels)
    else:
        print("  ⚠️  no rdma/infiniband labels found on GPU nodes (RDMA_NODE_SELECTOR unchanged)")

    # Enumerate InfiniBand devices and fabric type from sysfs.
    # `link_layer` distinguishes native InfiniBand from Ethernet/RoCE unambiguously.
    sys_result = None
    probe_node = sorted(per_node)[0] if per_node else (nodes[0]["metadata"]["name"] if (rdma_labels and ns) else None)
    if probe_node and ns:
        known_dev = per_node.get(probe_node, [None])[0]
        print(f"  sys probe     {probe_node}...", end=" ", flush=True)
        sys_result = probe_sys_ib(ns, probe_node, dev_name=known_dev)
        if sys_result["devices"] or sys_result["fabric"]:
            print(f"devices={sys_result['devices']}, fabric={sys_result['fabric'] or 'unknown'}")
        else:
            print("failed (pod error or no IB devices on node)")
    elif rdma_labels and not ns:
        print("  sys probe     skipped (NAMESPACE not in profile — run after namespace selection)")

    # Fill UCX_NET_DEVICES from /sys enumeration when device plugin wasn't available
    if not per_node and sys_result and sys_result["devices"]:
        devs = sys_result["devices"]
        ucx_str = ",".join(f"{d}:1" for d in devs)
        facts["RDMA_UCX_NET_DEVICES"] = ucx_str
        facts["RDMA_UCX_MAX_RNDV_RAILS"] = str(len(devs))
        print(f"  sys devices   {devs} → UCX_NET_DEVICES={ucx_str}")
        print(f"  rail count    {len(devs)} device(s) → UCX_MAX_RNDV_RAILS={len(devs)}")

    # Determine fabric type: /sys probe beats label heuristic
    sys_fabric = sys_result["fabric"] if sys_result else None
    is_roce = (sys_fabric == "roce") if sys_fabric else label_roce
    fabric_label = "RoCE/Ethernet" if is_roce else "native InfiniBand"
    src = "/sys link_layer" if sys_fabric else "label heuristic"
    print(f"  fabric type   {fabric_label}  ({src})")
    # UCX requires a non-empty address type: use `auto` for native InfiniBand and `eth` for RoCE.
    facts["RDMA_UCX_IB_ADDR_TYPE"] = "eth" if is_roce else "auto"

    # Optionally probe NVLink P2P health using a scheduler-placed GPU pod.
    # Probe failures remain UNKNOWN and do not block profile creation.
    have_gpu_node = bool(probe_node or (nodes[0]["metadata"]["name"] if nodes else None))
    if have_gpu_node and ns:
        p2p_image = (prof.get("P2P_PROBE_IMAGE") or "").strip() or P2P_PROBE_IMAGE
        pull_secret = (prof.get("IMAGE_PULL_SECRET") or "").strip() or None
        try:
            gpu_count = int((prof.get("P2P_PROBE_GPUS") or "").strip() or P2P_PROBE_GPUS)
        except ValueError:
            gpu_count = P2P_PROBE_GPUS
        print(
            f"  p2p probe     {gpu_product} ({gpu_count} GPU, scheduler-placed)...",
            end=" ",
            flush=True,
        )
        p2p = probe_nvlink_p2p(
            ns,
            gpu_product,
            image=p2p_image,
            pull_secret=pull_secret,
            gpu_count=gpu_count,
        )
        state = p2p.get("state", "unknown")
        extra = ""
        if p2p.get("summary") or p2p.get("route_unhealthy") is not None:
            extra = f" (Health Summary={p2p.get('summary')}, Route Unhealthy={p2p.get('route_unhealthy')})"
        elif state == "unknown" and p2p.get("error"):
            extra = f" ({p2p['error']})"  # surface WHY it couldn't probe, not an opaque unknown
        print(f"NVLINK_P2P={state}{extra}")
        facts["NVLINK_P2P"] = state
        if p2p.get("route_unhealthy") is not None:
            facts["NVLINK_P2P_ROUTE_UNHEALTHY"] = "true" if p2p.get("route_unhealthy") else "false"
        if state == "disabled":
            print(
                "     ⚠️  NVLink P2P is DISABLED fabric-wide — TP>1 throughput/goodput results would be "
                "INVALID until a cluster-admin restores the fabric (nv-fabricmanager / Route Unhealthy)."
            )
    elif not ns:
        print("  p2p probe     skipped (NAMESPACE not in profile — run after namespace selection)")

    # --- sentinel: mark that probing ran so run.sh Phase 1.5 doesn't re-fire every run ---
    if write:
        facts.setdefault("RDMA_UCX_NET_DEVICES", "all")
        facts["RDMA_FABRIC_PROBED"] = "true"

    # --- result ---
    if not facts:
        print("\nnothing to write — no fabric facts discovered")
        return 0

    print()
    print("discovered:")
    for k, v in facts.items():
        print(f'  {k}="{v}"')

    if write:
        patch_profile(prof_path, facts)
        print(f"\nwritten to cluster-profiles/{profile_name}.env")
        print("re-render any disagg cells to activate: scripts/render.sh <cell-dir>")
    else:
        print(f"\ndry run — re-run with --write to patch cluster-profiles/{profile_name}.env")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

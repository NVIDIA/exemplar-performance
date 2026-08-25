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

"""No-GPU cluster readiness checks used by profile initialization and validation.

The battery verifies API reachability, GPU labels, registry credentials, artifact storage, service-account
identity, and a small PVC streaming round trip. Probe resources use unique names and are always cleaned up.
`--fast` skips live probes when only profile parsing and reachability are needed.

Pure classifiers accept captured inputs; live probes use injected Kubernetes and HTTP runners. Confirmed
configuration failures block validation, while facts that cannot be observed safely degrade to warnings.
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

SCRIPTS = Path(__file__).resolve().parent

# Check levels. FAIL is the only one that makes `profile validate` exit non-zero (a real, actionable gap).
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

_ICON = {PASS: "✅", WARN: "🟡", FAIL: "❌", SKIP: "·"}


@dataclass
class Check:
    id: str
    level: str
    message: str
    fix: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# PURE classifiers (unit-tested with canned inputs; no cluster, no network)
# ─────────────────────────────────────────────────────────────────────────────


def classify_nodes(node_items: list, gpu_product: str, arch: str) -> Check:
    """PURE. Given the `.items` of `kubectl get nodes -o json` (already parsed), does the cluster expose the
    profile's GPU_PRODUCT / ARCH? Informational — a zero count WARNs (an autoscaled-to-zero pool is legitimate;
    the per-run preflight is the hard gate), a present-but-wrong-arch pairing is the real config smell.
    """
    matched = [
        n
        for n in node_items
        if (n.get("metadata", {}).get("labels", {}) or {}).get("nvidia.com/gpu.product") == gpu_product
    ]
    if not gpu_product:
        return Check(
            "gpu-nodes",
            WARN,
            "GPU_PRODUCT unset — cannot verify the cluster exposes the target GPU",
        )
    if not matched:
        any_gpu = sorted(
            {
                (n.get("metadata", {}).get("labels", {}) or {}).get("nvidia.com/gpu.product")
                for n in node_items
                if (n.get("metadata", {}).get("labels", {}) or {}).get("nvidia.com/gpu.product")
            }
        )
        seen = f" (products present: {', '.join(any_gpu)})" if any_gpu else " (no GPU-labelled nodes found)"
        return Check(
            "gpu-nodes",
            WARN,
            f"no nodes currently labelled nvidia.com/gpu.product={gpu_product}{seen} — "
            "autoscaled-to-zero? a run will pend until a node appears",
            fix=f"confirm GPU_PRODUCT matches the cluster (kubectl get nodes -L nvidia.com/gpu.product)",
        )
    node_archs = sorted(
        {
            (n.get("metadata", {}).get("labels", {}) or {}).get("kubernetes.io/arch")
            for n in matched
            if (n.get("metadata", {}).get("labels", {}) or {}).get("kubernetes.io/arch")
        }
    )
    if arch and node_archs and arch not in node_archs:
        return Check(
            "gpu-nodes",
            FAIL,
            f"{len(matched)} × {gpu_product} node(s), but their arch {node_archs} does not include the "
            f"profile ARCH={arch} — a digest-pinned image would hit `exec format error`",
            fix=f"set ARCH to one of {node_archs} in the profile, or use a matching cluster",
        )
    return Check(
        "gpu-nodes",
        PASS,
        f"{len(matched)} × {gpu_product} node(s) present" + (f" (arch {', '.join(node_archs)})" if node_archs else ""),
    )


def classify_reachability(rc: int, err: str) -> Check:
    """PURE. Turn a `kubectl cluster-info` outcome into a reachability verdict.

    This is the ONE probe that is a hard FAIL rather than a safe-degrade WARN, and deliberately so: an
    unreachable cluster / invalid context means NOTHING downstream could be proven, so run-readiness is
    UNKNOWN — not OK. Distinct from an individual degraded probe (RBAC-forbidden `create token`, an
    autoscaled-to-zero pool): those WARN because the cluster answered and only that one fact is unverifiable.
    A typo'd KUBE_CONTEXT must NOT safe-degrade every probe to WARN and still read RUN-READY (the silent
    success the wizard exists to kill)."""
    if rc == 0:
        return Check("reachability", PASS, "cluster reachable (cluster-info OK)")
    low = (err or "").lower()
    if "you must be logged in" in low or "401" in low or "unauthorized" in low:
        fix = "auth expired — refresh your proxy/Teleport login (e.g. `tsh kube login <ctx>`), then re-run"
    elif "x509" in low or "certificate" in low:
        fix = "certificate error — check the cluster CA / KUBE_CONTEXT (kubectl config get-contexts)"
    elif any(
        s in low
        for s in (
            "connection refused",
            "no route",
            "timed out",
            "timeout",
            "dial tcp",
            "i/o timeout",
        )
    ):
        fix = "cannot reach the API server — check VPN/network, then re-run"
    elif any(
        s in low
        for s in (
            "no context exists",
            "context was not found",
            "does not exist",
            "no configuration",
            'context "',
        )
    ):
        fix = "KUBE_CONTEXT names a context that does not exist — kubectl config get-contexts"
    else:
        fix = "verify KUBE_CONTEXT names a valid, reachable context (kubectl config get-contexts)"
    tail = f" ({err.strip()[:100]})" if (err or "").strip() else ""
    return Check(
        "reachability",
        FAIL,
        "cluster UNREACHABLE / context invalid — nothing could be proven; run-readiness is " "UNKNOWN, not OK" + tail,
        fix=fix,
    )


def probe_reachability(krun: Callable) -> Check:
    """IMPURE over `krun` (context already pinned by default_krun). `kubectl cluster-info` — the cheap,
    definitive 'can we even talk to this cluster' gate. Never raises."""
    try:
        rc, _, err = krun(["cluster-info"], timeout=15)
    except Exception as e:  # noqa: BLE001
        return classify_reachability(1, str(e))
    return classify_reachability(rc, err)


def classify_staging(result: dict) -> Check:
    """PURE. Turn the artifacts staging round-trip probe's outcome dict into a verdict. The round-trip is the
    check exercises an RWO PVC on ARTIFACTS_STORAGE_CLASS and streams a byte into it with `kubectl cp`.

    result keys: bound(bool), pod_ready(bool), cp_ok(bool), readback_ok(bool),
                 provisioning_failed(bool), sc(str), reason(str)."""
    sc = result.get("sc") or "<ARTIFACTS_STORAGE_CLASS>"
    if result.get("skipped"):
        return Check(
            "staging-roundtrip",
            SKIP,
            result.get("reason") or "staging round-trip skipped",
        )
    if result.get("provisioning_failed"):
        return Check(
            "staging-roundtrip",
            FAIL,
            f"artifacts StorageClass '{sc}' FAILED to provision a PVC ({result.get('reason','')}) — "
            "every run's artifacts PVC will pend indefinitely",
            fix=f"fix ARTIFACTS_STORAGE_CLASS in the profile (kubectl get storageclass), or pre-create "
            "the artifacts PVC",
        )
    if not result.get("bound"):
        return Check(
            "staging-roundtrip",
            WARN,
            f"artifacts PVC on '{sc}' did not bind within the probe window (no provisioning error seen)"
            f"{(' — ' + result['reason']) if result.get('reason') else ''} — slow provisioner or "
            "capacity; unverified, not blocking",
            fix="re-run validate when the cluster is quieter, or watch: kubectl get pvc -w",
        )
    if not result.get("pod_ready"):
        return Check(
            "staging-roundtrip",
            WARN,
            f"artifacts PVC on '{sc}' bound, but the probe pod did not become ready"
            f"{(' — ' + result['reason']) if result.get('reason') else ''} (e.g. busybox image "
            "unpullable) — mount+cp path UNVERIFIED, not blocking",
        )
    if not result.get("cp_ok"):
        return Check(
            "staging-roundtrip",
            FAIL,
            f"`kubectl cp` into the mounted artifacts PVC FAILED after retries "
            f"({result.get('reason','stream error')}) — this is the Teleport/API stream stall that "
            "kills staging mid-run; a real dataset/task-source copy will die the same way",
            fix="check the API proxy and storage stability for this context; stage-*.sh now retries, but a "
            "persistent stall needs the proxy fixed",
        )
    if not result.get("readback_ok"):
        return Check(
            "staging-roundtrip",
            FAIL,
            f"copied a file into the artifacts PVC on '{sc}' but read back different/absent bytes — the "
            "mount is not durable/consistent",
            fix="verify the CSI driver for this StorageClass mounts read-write correctly",
        )
    return Check(
        "staging-roundtrip",
        PASS,
        f"artifacts round-trip OK on '{sc}': PVC bound + mounted + `kubectl cp` streamed + read back",
    )


def classify_pvc_bind(result: dict) -> Check:
    """PURE. Verdict for the control-plane RWX PVC bind probe (the disconnect-resilient `submit` path needs a
    ReadWriteMany control PVC — an RWO class silently re-creates the Multi-Attach wall). Bind-only + safe-
    degrade: a class that can't provision RWX → FAIL; a slow/unverifiable bind → WARN.
    """
    sc = result.get("sc") or "<CONTROL_STORAGE_CLASS>"
    if result.get("skipped"):
        return Check(
            "control-rwx",
            SKIP,
            result.get("reason") or "control-plane RWX check skipped",
        )
    if result.get("provisioning_failed"):
        return Check(
            "control-rwx",
            FAIL,
            f"control StorageClass '{sc}' FAILED to provision a ReadWriteMany PVC "
            f"({result.get('reason','')}) — detached `submit` (status/logs concurrency) needs RWX",
            fix=f"set CONTROL_STORAGE_CLASS to a real RWX class (EFS/Filestore/FSx-Lustre/NFS): "
            "kubectl get storageclass",
        )
    if not result.get("bound"):
        return Check(
            "control-rwx",
            WARN,
            f"control RWX PVC on '{sc}' did not bind within the probe window — unverified, not blocking "
            "(only affects the detached `submit` lane)",
        )
    return Check(
        "control-rwx",
        PASS,
        f"control-plane RWX PVC binds on '{sc}' (detached `submit` supported)",
    )


# pull-secret auth states
_AUTH_OK = "ok"
_AUTH_FORBIDDEN = "forbidden"
_AUTH_UNKNOWN = "unknown"


def classify_pull_secret(
    secret: str,
    exists: bool,
    parseable: bool,
    hosts: list,
    auth: dict,
    image_results: Optional[dict] = None,
) -> Check:
    """PURE. Verdict for the pull-secret probe.

    - `image_results` (when a --recipe was passed): the full per-image kubelet pull-auth reproduction from
      capability_registry.image_pull_access_verdict semantics — a 403 on any pinned repo → FAIL (a real
      per-image org access gap).
    - else `auth` = {host: state∈{ok,forbidden,unknown}}: a registry `/v2/` auth PING per credentialed host —
      a 401/403 on the CRED itself → FAIL (expired/wrong key); an authenticating cred → PASS. Registry
      unreachable / unreadable secret → WARN (safe-degrade, never a false block)."""
    s = secret or "<IMAGE_PULL_SECRET>"
    if not exists:
        return Check(
            "pull-secret",
            WARN,
            f"pull secret '{s}' not readable in the namespace (absent or RBAC-forbidden) — pull auth "
            "unverified (preflight also checks presence)",
            fix=f"kubectl -n <ns> get secret {s}   # create/mirror a working docker-registry secret",
        )
    if not parseable:
        return Check(
            "pull-secret",
            WARN,
            f"pull secret '{s}' is not a parseable .dockerconfigjson — pull auth unverified",
        )
    # Recipe-scoped per-image org check (reuses image-pull-access): FORBIDDEN is the crash-killer.
    if image_results is not None:
        forbidden = [ref for ref, r in (image_results or {}).items() if (r or {}).get("status") == "forbidden"]
        if forbidden:
            short = forbidden[0].split("@")[0].split("/")[-1]
            more = f" (+{len(forbidden)-1} more)" if len(forbidden) > 1 else ""
            return Check(
                "pull-secret",
                FAIL,
                f"pull FORBIDDEN — secret '{s}' authenticates but its credential cannot pull the private "
                f"repo for {short}{more} (403). The kubelet gets the same 403 → ImagePullBackOff holding "
                "GPUs. Per-cluster credential gap, not a recipe bug.",
                fix=f"grant the image's NGC org to the credential in '{s}' (mirror a cluster that pulls it)",
            )
        if image_results and all((r or {}).get("status") == "pass" for r in image_results.values()):
            return Check(
                "pull-secret",
                PASS,
                f"pull secret '{s}' can authorize all {len(image_results)} pinned recipe image(s)",
            )
        return Check(
            "pull-secret",
            WARN,
            f"pull secret '{s}' — some pinned images could not be verified (registry unreachable); "
            "safe-degrade, watch for ImagePullBackOff",
        )
    # Cluster-level auth ping (no recipe): the credential must at least AUTHENTICATE to its registry.
    if not hosts:
        return Check(
            "pull-secret",
            WARN,
            f"pull secret '{s}' carries no registry credential — a private image will fail to pull",
            fix=f"add the registry credential (e.g. nvcr.io) to secret '{s}'",
        )
    bad = [h for h in hosts if auth.get(h) == _AUTH_FORBIDDEN]
    if bad:
        return Check(
            "pull-secret",
            FAIL,
            f"pull secret '{s}' has an INVALID credential for {', '.join(bad)} (registry refused the "
            "token exchange — expired/rotated key?) — every pull from that registry will 401",
            fix=f"refresh the credential for {', '.join(bad)} in secret '{s}'",
        )
    okd = [h for h in hosts if auth.get(h) == _AUTH_OK]
    if okd:
        unk = [h for h in hosts if auth.get(h) == _AUTH_UNKNOWN]
        tail = f" ({len(unk)} unverified: {', '.join(unk)})" if unk else ""
        return Check(
            "pull-secret",
            PASS,
            f"pull secret '{s}' authenticates to {', '.join(okd)}{tail}"
            " — pass --recipe <cell> to also verify per-image org access",
        )
    return Check(
        "pull-secret",
        WARN,
        f"pull secret '{s}' — could not reach the registry to verify the credential (safe-degrade)",
    )


def verdict(checks: list) -> tuple[bool, str]:
    """PURE. Reduce the battery to (run_ready, one-line summary). run_ready iff NO FAIL — WARN/UNKNOWN/SKIP
    are advisory and never flip the verdict (safe-degrade). Mirrors preflight's rc semantics.
    """
    n = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for c in checks:
        n[c.level] = n.get(c.level, 0) + 1
    ok = n[FAIL] == 0
    summary = f"{n[PASS]} passed · {n[WARN]} warned · {n[FAIL]} failed" + (f" · {n[SKIP]} skipped" if n[SKIP] else "")
    return ok, summary


def format_battery(cluster: str, checks: list) -> str:
    """PURE. Render the battery as the one-verdict panel `profile validate` prints."""
    ok, summary = verdict(checks)
    out = [f"── cluster readiness: {cluster} " + "─" * max(4, 48 - len(cluster))]
    for c in checks:
        out.append(f"  {_ICON.get(c.level, '?')} {c.message}")
        if c.fix and c.level == FAIL:
            out.append(f"       → fix: {c.fix}")
    head = "cluster run-ready ✓" if ok else "NOT run-ready — resolve the ❌ item(s) above"
    out.append(f"  → {head}   ({summary})")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Throwaway-resource cleanup tracker (belt AND suspenders: try/finally + atexit + SIGTERM)
# ─────────────────────────────────────────────────────────────────────────────
_LIVE_RESOURCES: list[tuple] = []  # (krun, ns, kind, name)
_CLEAN_INSTALLED = False


def _track(krun: Callable, ns: str, kind: str, name: str) -> None:
    _LIVE_RESOURCES.append((krun, ns, kind, name))
    global _CLEAN_INSTALLED
    if not _CLEAN_INSTALLED:
        atexit.register(_cleanup_all)
        for _sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(_sig, _on_signal)
            except Exception:
                pass
        _CLEAN_INSTALLED = True


def _untrack(krun: Callable, ns: str, kind: str, name: str) -> None:
    try:
        _LIVE_RESOURCES.remove((krun, ns, kind, name))
    except ValueError:
        pass


def _delete(krun: Callable, ns: str, kind: str, name: str) -> None:
    try:
        krun(
            ["-n", ns, "delete", kind, name, "--ignore-not-found", "--wait=false"],
            timeout=20,
        )
    except Exception:
        pass


def _cleanup_all() -> None:
    for krun, ns, kind, name in list(_LIVE_RESOURCES):
        _delete(krun, ns, kind, name)
        _untrack(krun, ns, kind, name)


def _on_signal(signum, _frame):
    _cleanup_all()
    # Re-raise the default disposition so the process still exits.
    raise SystemExit(128 + signum)


# ─────────────────────────────────────────────────────────────────────────────
# IMPURE probes (injected krun / http). Each returns a Check; never raises.
# ─────────────────────────────────────────────────────────────────────────────


def probe_nodes(krun: Callable, gpu_product: str, arch: str) -> Check:
    rc, out, _ = krun(["get", "nodes", "-o", "json"], timeout=30)
    if rc != 0 or not out:
        return Check(
            "gpu-nodes",
            WARN,
            "could not list nodes (RBAC/unreachable) — GPU availability unverified",
        )
    try:
        items = json.loads(out).get("items", [])
    except Exception:
        return Check("gpu-nodes", WARN, "could not parse node list — GPU availability unverified")
    return classify_nodes(items, gpu_product, arch)


_READY_IMAGE = "busybox:1.36"


def _pod_overrides(pvc: str, pull_secret: str, mode: str) -> str:
    spec: dict = {
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 0,
        "tolerations": [{"operator": "Exists"}],
        "volumes": [{"name": "v", "persistentVolumeClaim": {"claimName": pvc}}],
        "containers": [
            {
                "name": "p",
                "image": _READY_IMAGE,
                "command": ["sleep", "300"],
                "volumeMounts": [{"name": "v", "mountPath": mode}],
            }
        ],
    }
    if pull_secret:
        spec["imagePullSecrets"] = [{"name": pull_secret}]
    return json.dumps({"spec": spec})


def _pvc_manifest(name: str, ns: str, sc: str, access_mode: str, size: str = "1Gi") -> str:
    doc = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
            "namespace": ns,
            "labels": {"app.kubernetes.io/managed-by": "llmb-readiness"},
        },
        "spec": {
            "accessModes": [access_mode],
            "resources": {"requests": {"storage": size}},
        },
    }
    if sc:
        doc["spec"]["storageClassName"] = sc
    return json.dumps(doc)


# Provisioners that CANNOT dynamically provision a block/file PVC — an object/bucket CSI (S3, blob, GCS).
# A readycheck (or a real artifacts/model-cache) PVC on one of these NEVER binds: the external-provisioner
# emits "Waiting for a volume to be created ... 's3.csi.aws.com'" and the PVC sits Pending FOREVER (the QA
# hang). We fail FAST on the provisioner rather than burning the whole bind timeout on a guaranteed hang.
_OBJECT_STORE_PROVISIONERS = ("s3.csi", "blob.csi", "gcs.csi", "objectstorage", "cosi.")


def sc_provisioner(krun: Callable, sc: str) -> str:
    """Return a StorageClass's provisioner (lowercased), or '' when unset/unknown/unreadable (safe-degrade →
    the caller proceeds to the live bind probe rather than a false fail)."""
    if not sc:
        return ""
    rc, out, _ = krun(["get", "storageclass", sc, "-o", "jsonpath={.provisioner}"], timeout=15)
    return (out or "").strip().lower() if rc == 0 else ""


def classify_unprovisionable_sc(check_id: str, sc: str, provisioner: str) -> Optional[Check]:
    """PURE. Fail-FAST guard: if `provisioner` is an OBJECT-store CSI (cannot back a block/file PVC), return a
    FAIL Check naming the class + provisioner + the fix — so the readycheck errors out IMMEDIATELY instead of
    hanging on a Pending PVC until the bind timeout. None → the class may provision; proceed to the live
    bind probe. Unit-testable with a mocked provisioner string (no cluster)."""
    prov = (provisioner or "").lower()
    if prov and any(o in prov for o in _OBJECT_STORE_PROVISIONERS):
        return Check(
            check_id,
            FAIL,
            f"StorageClass '{sc}' uses object-store provisioner '{provisioner}', which CANNOT "
            "dynamically provision a block/file PVC — a PVC on it sits Pending forever (never binds)",
            fix="pick a block/file StorageClass (e.g. ebs / fsx-lustre) in the profile; "
            "kubectl get storageclass -o custom-columns=NAME:.metadata.name,PROV:.provisioner",
        )
    return None


def _pvc_provisioning_failed(krun: Callable, ns: str, pvc: str) -> tuple[bool, str]:
    """Read the PVC's events for a definitive ProvisioningFailed (vs still-provisioning). Returns (failed, msg)."""
    rc, out, _ = krun(
        [
            "-n",
            ns,
            "get",
            "events",
            "--field-selector",
            f"involvedObject.name={pvc},involvedObject.kind=PersistentVolumeClaim",
            "-o",
            "json",
        ],
        timeout=20,
    )
    if rc != 0 or not out:
        return False, ""
    try:
        for e in json.loads(out).get("items", []):
            # Terminal provisioner failures the CSI/external-provisioner emits on a bad StorageClass — distinct
            # from the benign, retrying "Provisioning"/"WaitForFirstConsumer" (still-in-progress → not a gap).
            if e.get("reason") == "ProvisioningFailed":
                return True, (e.get("message") or "")[:160]
    except Exception:
        pass
    return False, ""


def _pvc_phase(krun: Callable, ns: str, pvc: str) -> str:
    rc, out, _ = krun(["-n", ns, "get", "pvc", pvc, "-o", "jsonpath={.status.phase}"], timeout=15)
    return (out or "").strip() if rc == 0 else ""


def _bind_after_pod(krun: Callable, ns: str, pvc: str, pod: str, ready_timeout_s: int) -> dict:
    """Pod-TRIGGERED bind — the WaitForFirstConsumer-safe path. Many default StorageClasses (EBS, GCE-PD,
    Some filesystem-backed classes use volumeBindingMode=WaitForFirstConsumer: the PVC stays Pending until a consuming pod is
    scheduled, so a "wait for Bound THEN create the pod" ordering deadlocks. We create the pod first and wait
    on IT — its readiness implies the PVC bound + mounted. On a not-ready pod we then read the PVC phase +
    events to classify (ProvisioningFailed → FAIL; still-Pending → WARN; Bound-but-pod-unready → WARN).
    Returns {bound, pod_ready, provisioning_failed, reason}."""
    rc_w, _, err_w = krun(
        [
            "-n",
            ns,
            "wait",
            "pod",
            pod,
            "--for=condition=ready",
            f"--timeout={ready_timeout_s}s",
        ],
        timeout=ready_timeout_s + 20,
    )
    if rc_w == 0:
        return {
            "bound": True,
            "pod_ready": True,
            "provisioning_failed": False,
            "reason": "",
        }
    # Pod not ready — distinguish a storage gap from a pod/image gap.
    failed, msg = _pvc_provisioning_failed(krun, ns, pvc)
    if failed:
        return {
            "bound": False,
            "pod_ready": False,
            "provisioning_failed": True,
            "reason": msg,
        }
    phase = _pvc_phase(krun, ns, pvc)
    if phase == "Bound":
        return {
            "bound": True,
            "pod_ready": False,
            "provisioning_failed": False,
            "reason": (err_w or "").strip()[:120] or "pod not ready (bound OK)",
        }
    return {
        "bound": False,
        "pod_ready": False,
        "provisioning_failed": False,
        "reason": f"PVC {phase or 'Pending'} at timeout",
    }


def probe_staging_roundtrip(
    krun: Callable,
    ns: str,
    sc: str,
    pull_secret: str,
    access_mode: str = "ReadWriteOnce",
    *,
    suffix: Optional[str] = None,
    ready_timeout_s: int = 150,
) -> Check:
    """IMPURE. The artifacts staging round-trip: create a throwaway PVC on `sc` AND a busybox pod mounting it,
    wait on the POD (WaitForFirstConsumer-safe — see _bind_after_pod), then `kubectl cp` a tiny file in
    (retried — the Teleport stall path), read it back, and tear both down. Every resource is tracked for
    atexit/signal cleanup. Any error → a WARN/FAIL Check, never an exception."""
    if not sc:
        return classify_staging(
            {
                "skipped": True,
                "reason": "ARTIFACTS_STORAGE_CLASS unset — skipping round-trip",
            }
        )
    if not ns:
        return classify_staging({"skipped": True, "reason": "NAMESPACE unset — skipping round-trip"})
    # Fail FAST on an object-store class (s3.csi.aws.com …) — creating a throwaway PVC on it would just hang
    # Pending until the bind timeout. Look up the provisioner up front and error out with guidance instead.
    ff = classify_unprovisionable_sc("staging-roundtrip", sc, sc_provisioner(krun, sc))
    if ff is not None:
        return ff
    sfx = suffix or uuid.uuid4().hex[:6]
    pvc = f"llmb-readycheck-{sfx}"
    pod = f"llmb-readycheck-{sfx}"
    res: dict = {
        "sc": sc,
        "bound": False,
        "pod_ready": False,
        "cp_ok": False,
        "readback_ok": False,
        "provisioning_failed": False,
        "reason": "",
    }
    tmp = None
    try:
        # Create PVC + pod TOGETHER (delete-before-create is a no-op for a uuid name, but idempotent on retry).
        krun(
            ["-n", ns, "delete", "pvc", pvc, "--ignore-not-found", "--wait=false"],
            timeout=15,
        )
        _track(krun, ns, "pvc", pvc)
        rc, _, err = krun(
            ["-n", ns, "apply", "-f", "-"],
            timeout=25,
            stdin=_pvc_manifest(pvc, ns, sc, access_mode),
        )
        if rc != 0:
            res["reason"] = (err or "").strip()[:160] or "PVC apply failed"
            return classify_staging(res)
        _track(krun, ns, "pod", pod)
        rc, _, err = krun(
            [
                "-n",
                ns,
                "run",
                pod,
                "--image",
                _READY_IMAGE,
                "--restart=Never",
                f"--overrides={_pod_overrides(pvc, pull_secret, '/rw')}",
            ],
            timeout=30,
        )
        if rc != 0:
            res["reason"] = (err or "").strip()[:160] or "pod create failed"
            return classify_staging(res)
        # Wait on the POD — its readiness triggers WaitForFirstConsumer bind + implies mount succeeded.
        b = _bind_after_pod(krun, ns, pvc, pod, ready_timeout_s)
        res.update(b)
        if not (res["bound"] and res["pod_ready"]):
            return classify_staging(res)
        # kubectl cp a tiny file in — the exact streaming path that stalled. Retry (Teleport-aware).
        token = f"llmb-readycheck-{sfx}-{int(time.time())}"
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        tmp.write(token)
        tmp.flush()
        tmp.close()
        res["cp_ok"] = _cp_with_retry(krun, ns, tmp.name, f"{pod}:/rw/probe.txt")
        if not res["cp_ok"]:
            res["reason"] = "kubectl cp failed after retries"
            return classify_staging(res)
        rc_e, out_e, _ = krun(["-n", ns, "exec", pod, "--", "cat", "/rw/probe.txt"], timeout=30)
        res["readback_ok"] = rc_e == 0 and (out_e or "").strip() == token
        return classify_staging(res)
    finally:
        _delete(krun, ns, "pod", pod)
        _untrack(krun, ns, "pod", pod)
        _delete(krun, ns, "pvc", pvc)
        _untrack(krun, ns, "pvc", pvc)
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


def probe_control_rwx(
    krun: Callable,
    ns: str,
    sc: str,
    pull_secret: str = "",
    *,
    suffix: Optional[str] = None,
    ready_timeout_s: int = 120,
) -> Check:
    """IMPURE. Verify the control-plane RWX class actually provisions + mounts ReadWriteMany. Pod-triggered
    (WaitForFirstConsumer-safe): create a throwaway RWX PVC + a busybox pod mounting it and wait on the pod.
    A class that can't provision RWX at all FAILs; a slow/unverifiable bind → WARN. (A full 2-pod cross-node
    concurrency test belongs in a heavier init self-test; single-mount proves the class supports RWX.)
    """
    if not sc:
        return classify_pvc_bind(
            {
                "skipped": True,
                "reason": "CONTROL_STORAGE_CLASS unset — detached `submit` " "lane not configured (skipping RWX check)",
            }
        )
    if not ns:
        return classify_pvc_bind({"skipped": True, "reason": "NAMESPACE unset — skipping control RWX check"})
    # Fail FAST on an object-store class — an RWX PVC on an s3.csi provisioner never binds (hangs Pending).
    ff = classify_unprovisionable_sc("control-rwx", sc, sc_provisioner(krun, sc))
    if ff is not None:
        return ff
    sfx = suffix or uuid.uuid4().hex[:6]
    pvc = f"llmb-readycheck-rwx-{sfx}"
    pod = f"llmb-readycheck-rwx-{sfx}"
    res: dict = {"sc": sc, "bound": False, "provisioning_failed": False, "reason": ""}
    try:
        krun(
            ["-n", ns, "delete", "pvc", pvc, "--ignore-not-found", "--wait=false"],
            timeout=15,
        )
        _track(krun, ns, "pvc", pvc)
        rc, _, err = krun(
            ["-n", ns, "apply", "-f", "-"],
            timeout=25,
            stdin=_pvc_manifest(pvc, ns, sc, "ReadWriteMany"),
        )
        if rc != 0:
            res["reason"] = (err or "").strip()[:160] or "RWX PVC apply failed"
            return classify_pvc_bind(res)
        _track(krun, ns, "pod", pod)
        rc, _, err = krun(
            [
                "-n",
                ns,
                "run",
                pod,
                "--image",
                _READY_IMAGE,
                "--restart=Never",
                f"--overrides={_pod_overrides(pvc, pull_secret, '/rwx')}",
            ],
            timeout=30,
        )
        if rc != 0:
            res["reason"] = (err or "").strip()[:160] or "pod create failed"
            return classify_pvc_bind(res)
        b = _bind_after_pod(krun, ns, pvc, pod, ready_timeout_s)
        # bound (pod mounted the RWX PVC) is the RWX signal; pod_ready without bind is impossible here.
        res["bound"], res["provisioning_failed"], res["reason"] = (
            b["bound"],
            b["provisioning_failed"],
            b["reason"],
        )
        return classify_pvc_bind(res)
    finally:
        _delete(krun, ns, "pod", pod)
        _untrack(krun, ns, "pod", pod)
        _delete(krun, ns, "pvc", pvc)
        _untrack(krun, ns, "pvc", pvc)


def _cp_with_retry(krun: Callable, ns: str, src: str, dest: str, tries: int = 3) -> bool:
    """kubectl cp with linear backoff — the python mirror of _stream_retry.sh (streaming ops stall on a
    Teleport/API proxy hiccup). Returns True on the first success."""
    for t in range(1, tries + 1):
        rc, _, _ = krun(["-n", ns, "cp", src, dest], timeout=60)
        if rc == 0:
            return True
        if t < tries:
            time.sleep(t * 3)
    return False


# ── pull-secret registry-auth ping (no recipe) ─────────────────────────────────


def _read_pull_secret(krun: Callable, ns: str, secret: str):
    """Return (exists, dockerconfig_bytes|None). Unreadable/absent → (False, None)."""
    if not (secret and ns):
        return False, None
    rc, out, _ = krun(
        [
            "-n",
            ns,
            "get",
            "secret",
            secret,
            "-o",
            "jsonpath={.data.\\.dockerconfigjson}",
        ],
        timeout=20,
    )
    if rc != 0:
        return False, None
    if not (out or "").strip():
        return True, None
    try:
        return True, base64.b64decode(out.strip())
    except Exception:
        return True, b""


def probe_registry_auth(host: str, cred: Optional[tuple], http: Callable) -> str:
    """IMPURE over `http`. Does the secret's credential AUTHENTICATE to `host`? Standard registry-v2 ping:
    GET https://host/v2/ → 401 + Bearer challenge → token exchange with the cred. 200 token → ok;
    401/403 on the token → forbidden (bad/expired cred); anything else / unreachable → unknown (safe-degrade).
    """
    import capability_registry as _cap

    try:
        status, hdrs, _ = http("GET", f"https://{host}/v2/", {})
    except Exception:
        return _AUTH_UNKNOWN
    if status == 200:
        return _AUTH_OK  # open registry (no auth needed)
    if status not in (401, 403):
        return _AUTH_UNKNOWN
    challenge = _cap._parse_www_authenticate(
        (hdrs or {}).get("WWW-Authenticate") or (hdrs or {}).get("Www-Authenticate") or ""
    )
    if not challenge:
        return _AUTH_FORBIDDEN if (status == 403 and cred) else _AUTH_UNKNOWN
    # A catalog scope proves the credential is accepted without needing a specific repo.
    challenge = {**challenge, "scope": challenge.get("scope") or "registry:catalog:*"}
    token, terr = _cap._fetch_pull_token(challenge, cred, "", http)
    if token is not None:
        return _AUTH_OK
    if terr in (401, 403):
        return _AUTH_FORBIDDEN
    return _AUTH_UNKNOWN


def probe_pull_secret(krun: Callable, http: Callable, ns: str, secret: str, recipe: Optional[dict] = None) -> Check:
    """IMPURE. Verify the namespace's IMAGE_PULL_SECRET. With `recipe` in hand, reproduce the full per-image
    kubelet pull-auth (reuses capability_registry.probe_image_pull_access — the real org-access gap). Without a
    recipe, ping each credentialed registry's `/v2/` to prove the credential authenticates. Never raises.
    """
    import capability_registry as _cap

    exists, dockercfg = _read_pull_secret(krun, ns, secret)
    auths = _cap.parse_dockerconfigjson(dockercfg) if dockercfg else None
    parseable = auths is not None
    hosts = sorted(auths.keys()) if auths else []
    if recipe is not None:
        images = _cap.pinned_images(recipe)
        image_results = None
        if exists and parseable and images:
            try:
                image_results = _cap.probe_image_pull_access(images, dockercfg, _cap._default_pull_http)
            except Exception:
                image_results = None
        return classify_pull_secret(secret, exists, parseable, hosts, {}, image_results=image_results)
    auth = {}
    if exists and parseable:
        for h in hosts:
            auth[h] = probe_registry_auth(h, auths.get(h), http)
    return classify_pull_secret(secret, exists, parseable, hosts, auth)


# ─────────────────────────────────────────────────────────────────────────────
# The battery
# ─────────────────────────────────────────────────────────────────────────────


def probe_var_reconcile(prof: dict, recipe_cell: Optional[str]) -> Check:
    """OFFLINE (no cluster). The ${VAR}-reconciliation linchpin surfaced as a readiness Check. With a chosen
    cell, reconcile THAT cell's referenced ${VAR}s; without one (plain `profile validate`/`init`), reconcile
    broadly across every committed cell target-compatible with this cluster. A referenced profile var left
    EMPTY (the NO_INTERNET_DNS_IP case) → FAIL. Never raises → safe-degrade WARN."""
    try:
        sys.path.insert(0, str(SCRIPTS))
        import manifest_vars as _mv

        if recipe_cell:
            gaps = _mv.reconcile(Path(recipe_cell), prof)
            scope = Path(recipe_cell).name
        else:
            import profile_resolver as _pr

            cells = sorted({p.parent for p in (SCRIPTS.parent / "recipes").glob("**/recipe.yaml")})

            def _compat(cell):
                try:
                    import yaml

                    env = (yaml.safe_load((cell / "recipe.yaml").read_text()) or {}).get("envelope") or {}
                    return not _pr.check_target_compat(env, prof)
                except Exception:
                    return True  # can't judge compat → include (conservative)

            gaps = _mv.reconcile_broad(prof, cells, compatible=_compat)
            scope = "all target-compatible cells"
    except Exception as e:
        return Check(
            "var-reconcile",
            WARN,
            f"${{VAR}} reconciliation skipped ({e}) — completeness check still applies",
        )
    fails = [g for g in gaps if g.level == "FAIL"]
    if fails:
        g = fails[0]
        more = f" (+{len(fails) - 1} more)" if len(fails) > 1 else ""
        return Check(
            "var-reconcile",
            FAIL,
            f"manifest ${{VAR}} left EMPTY in the profile: {g.var}{more} — resolves to '' at runtime "
            f"and the workload fails (checked {scope})",
            fix=g.fix,
        )
    warns = [g for g in gaps if g.level == "WARN"]
    if warns:
        return Check(
            "var-reconcile",
            WARN,
            f"{len(warns)} manifest ${{VAR}}(s) not documented as profile/runtime vars "
            f"({', '.join(sorted(g.var for g in warns))}) — verify the profile template ({scope})",
        )
    return Check(
        "var-reconcile",
        PASS,
        f"every manifest ${{VAR}} resolves to a non-empty profile/runtime value ({scope})",
    )


def run_battery(
    prof: dict,
    *,
    recipe: Optional[dict] = None,
    recipe_cell: Optional[str] = None,
    fast: bool = False,
    krun: Optional[Callable] = None,
    http: Optional[Callable] = None,
) -> list:
    """Run the readiness battery for a resolved profile dict. Returns a list of Check.

    Variable reconciliation is an offline check and still runs with ``--fast``. Registry authentication,
    staging, issuer, and node probes require a live cluster and are skipped in fast mode. ``krun`` and ``http``
    may be injected by tests. Checks run from offline validation to inexpensive reads and then pod probes.
    """
    var_check = probe_var_reconcile(prof, recipe_cell)
    if fast:
        return [
            var_check,
            Check(
                "fast",
                SKIP,
                "--fast: LIVE readiness probes skipped (offline ${VAR}-reconcile still ran)",
            ),
        ]
    krun = krun or default_krun(prof)
    # Reachability precheck BEFORE the live battery. An unreachable/invalid context is a hard FAIL (nothing
    # can be proven) — short-circuit so we return that one actionable FAIL instead of a wall of safe-degrade
    # WARNs that would still read RUN-READY. (The offline ${VAR}-reconcile above still stands.)
    reach = probe_reachability(krun)
    if reach.level == FAIL:
        return [var_check, reach]
    import capability_registry as _cap  # noqa: F401  (probe_* import it lazily; ensure importable early)

    http = http or _cap._default_pull_http
    ns = (prof.get("NAMESPACE") or "").strip()
    gpu_product = (prof.get("GPU_PRODUCT") or "").strip()
    arch = (prof.get("ARCH") or "").strip()
    secret = (prof.get("IMAGE_PULL_SECRET") or "").strip()
    artifacts_sc = (prof.get("ARTIFACTS_STORAGE_CLASS") or "").strip()
    control_sc = (prof.get("CONTROL_STORAGE_CLASS") or "").strip()
    access_mode = (prof.get("ARTIFACTS_ACCESS_MODE") or "ReadWriteOnce").strip()
    checks = [
        var_check,
        reach,
        probe_nodes(krun, gpu_product, arch),
        probe_pull_secret(krun, http, ns, secret, recipe=recipe),
        probe_staging_roundtrip(krun, ns, artifacts_sc, secret, access_mode),
        probe_control_rwx(krun, ns, control_sc, secret),
    ]
    return checks


def default_krun(prof: dict) -> Callable:
    """Build the LIVE kubectl runner pinned to the profile's KUBE_CONTEXT. Supports an optional `stdin=` for
    `kubectl apply -f -`. Signature: krun(args, timeout=30, stdin=None) -> (rc, stdout, stderr).
    """
    ctx = (prof.get("KUBE_CONTEXT") or "").strip()

    def _run(args, timeout: int = 30, stdin: Optional[str] = None):
        ctx_args = ["--context", ctx] if ctx else []
        argv = ["kubectl", *ctx_args, "--request-timeout=25s", *args]
        try:
            p = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin if stdin is not None else None,
            )
            return p.returncode, p.stdout, p.stderr
        except Exception as e:
            return 1, "", str(e)

    return _run

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

"""profile_init.py <cluster> [--force]

Profile init: create or re-initialize a cluster profile.

Steps for a NEW profile:
  1. List kubectl contexts → user selects → connectivity check
  2. Namespace selection (Case A: pick existing / Case B: create new)
  3. Auto-discover GPU nodes, PVCs, secrets, storage classes in selected namespace
  4. Interactive prompts with discovered defaults for all critical profile vars
  5. Write cluster-profiles/<cluster>.env (KUBE_CONTEXT, NAMESPACE, ARCH, GPU_PRODUCT, …)
  6. Show summary table

If the profile already exists: show current state, offer to [v]alidate / [e]dit / [r]e-initialize.
--force skips the "already exists" prompt and goes straight to re-initialization.

Usage:
  scripts/profile_init.py <cluster>
  scripts/profile_init.py --cluster <cluster>
  scripts/profile_init.py <cluster> --force

Reuses: profile_resolver.list_profiles(), profile_env_path(), _read_env()
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import profile_resolver as _pr

PROFILES_DIR = _pr.PROFILES_DIR


# ---------------------------------------------------------------------------
# Injectable kubectl runner (pure for testability)
# ---------------------------------------------------------------------------


def default_krun(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run kubectl and return (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(
            ["kubectl", "--request-timeout=25s", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def list_contexts(krun=default_krun) -> list[dict]:
    """Return list of {name, current} from kubectl config get-contexts."""
    rc, out, _ = krun(["config", "get-contexts", "--no-headers", "-o", "name"])
    if rc != 0 or not out.strip():
        return []
    # Parse: get current context separately
    rc2, cur, _ = krun(["config", "current-context"])
    current = cur.strip() if rc2 == 0 else ""
    contexts = []
    for line in out.strip().splitlines():
        name = line.strip()
        if name:
            contexts.append({"name": name, "current": name == current})
    return contexts


def check_reachability(context: str, krun=default_krun) -> tuple[bool, str]:
    """Check if a context is reachable. Returns (ok, detail_message)."""
    rc, out, err = krun(["--context", context, "cluster-info"], timeout=15)
    if rc == 0:
        # Extract node count / GPU info if available
        return True, ""
    err_lower = (err or "").lower()
    if "you must be logged in" in err_lower or "401" in err_lower:
        return False, "Teleport session may have expired — refresh your proxy login"
    if "x509" in err_lower:
        return False, "Certificate error — check cluster certificate"
    if "connection refused" in err_lower or "no route" in err_lower:
        return False, "Connection refused — check VPN / network"
    return False, err.strip()[:120] if err.strip() else "unreachable"


def probe_gpu_nodes(context: str, krun=default_krun) -> list[dict]:
    """Discover GPU nodes. Returns list of {name, gpu_product, arch, gpus, cpu_alloc, mem_alloc}.

    `cpu_alloc` / `mem_alloc` are the RAW k8s quantity strings straight off `.status.allocatable`
    ("139580m" or "192"; "948007936Ki") — deliberately unparsed here so this stays a thin JSON reader and
    the normalization lives in pure, offline-testable helpers (wizard_init._cpu_to_millicores / _mem_to_gib).
    """
    rc, out, _ = krun(["--context", context, "get", "nodes", "-o", "json"])
    if rc != 0 or not out:
        return []
    try:
        items = json.loads(out).get("items", [])
    except Exception:
        return []
    nodes = []
    for node in items:
        labels = node.get("metadata", {}).get("labels", {})
        gpu_product = labels.get("nvidia.com/gpu.product", "")
        arch = labels.get("kubernetes.io/arch", "")
        if gpu_product:
            # `gpus` = the node's ALLOCATABLE GPU count. Needed by whole-node cells (GPU_PER_NODE): a
            # benchmark that must own an entire node has to request exactly this many.
            alloc = node.get("status", {}).get("allocatable", {}) or {}
            try:
                gpus = int(alloc.get("nvidia.com/gpu", 0) or 0)
            except (TypeError, ValueError):
                gpus = 0
            # SAME single pass as the GPU count — cpu/memory allocatable ride along so CPU_PER_NODE /
            # MEM_PER_NODE / WHOLE_NODE_* cost no extra kubectl call.
            nodes.append(
                {
                    "name": node["metadata"]["name"],
                    "gpu_product": gpu_product,
                    "arch": arch,
                    "gpus": gpus,
                    "cpu_alloc": str(alloc.get("cpu", "") or ""),
                    "mem_alloc": str(alloc.get("memory", "") or ""),
                }
            )
    return nodes


# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------


def list_namespaces(context: str, krun=default_krun) -> list[str]:
    """Return accessible namespace names (RBAC may limit results)."""
    rc, out, _ = krun(
        [
            "--context",
            context,
            "get",
            "namespaces",
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name",
        ]
    )
    if rc != 0 or not out.strip():
        return []
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def namespace_has_gpu_workloads(context: str, ns: str, krun=default_krun) -> bool:
    """Return True if the namespace has any pods with GPU requests."""
    rc, out, _ = krun(["--context", context, "-n", ns, "get", "pods", "-o", "json"])
    if rc != 0 or not out:
        return False
    try:
        items = json.loads(out).get("items", [])
    except Exception:
        return False
    for pod in items:
        for c in pod.get("spec", {}).get("containers", []):
            if int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0) or 0) > 0:
                return True
    return False


def create_namespace(context: str, ns: str, krun=default_krun) -> bool:
    """Create a namespace. Returns True on success."""
    rc, _, err = krun(["--context", context, "create", "namespace", ns])
    if rc != 0:
        print(f"    ✗ kubectl error: {err.strip()[:120]}")
        return False
    return True


# Namespaces that belong to cluster INFRASTRUCTURE, never to a benchmark run. `kubectl get namespaces`
# returns them alphabetically, so the first entry on a real cluster is almost always one of these
# (`argocd`, `cert-manager`, …). Defaulting to it meant pressing Enter deployed benchmark workloads into
# the cluster's system namespace, so infrastructure namespaces are never offered as the default.
_SYSTEM_NS_EXACT = {
    "argocd",
    "cert-manager",
    "default",
    "kube-node-lease",
    "kube-public",
    "kube-system",
    "cilium-secrets",
    "monitoring",
    "gpu-operator",
    "nvidia-gpu-operator",
    "local-path-storage",
}
_SYSTEM_NS_PREFIX = (
    "kube-",
    "dgxc-",
    "gpu-operator",
    "nvidia-",
    "openshift-",
    "cattle-",
    "calico-",
    "tigera-",
    "istio-",
    "conformance-",
)


def is_system_namespace(ns: str) -> bool:
    """PURE — True if `ns` is cluster infrastructure a benchmark must never be deployed into."""
    n = (ns or "").strip().lower()
    return n in _SYSTEM_NS_EXACT or n.startswith(_SYSTEM_NS_PREFIX)


def suggested_namespace(owner: str = "", suffix: str = "llmb-k8s") -> str:
    """PURE — the RECOMMENDED new-namespace name: `<simplified-username>-llmb-k8s`.

    Simplification makes any login a valid RFC-1123 label: lowercase, non-alphanumerics collapsed to a
    single '-', edges trimmed (so `first.last@example.com` → `first-last-llmb-k8s`). Falls back to a bare
    `llmb-k8s` when no owner is known. Truncated to the 63-char limit on a '-' boundary.
    """
    raw = (owner or "").strip().lower()
    raw = raw.split("@", 1)[0]  # an email login → just the local part
    cleaned = []
    for ch in raw:
        if ch.isalnum():
            cleaned.append(ch)
        elif cleaned and cleaned[-1] != "-":
            cleaned.append("-")
    base = "".join(cleaned).strip("-")
    name = f"{base}-{suffix}" if base else suffix
    return name[:63].strip("-")


def rank_namespaces(namespaces: list[str], owner: str = "", gpu_namespaces=()) -> list[str]:
    """PURE — best-first ordering for the namespace default. Preference:
         1. the caller's OWN namespaces (name contains `owner`) — what a returning user wants
         2. namespaces already running GPU workloads — i.e. where benchmarking happens here
         3. anything else that isn't cluster infrastructure
         4. system namespaces LAST (they must never be the Enter-key default)
    Stable within each tier, so the listed order is otherwise preserved."""
    own = (owner or "").strip().lower()
    gpu = set(gpu_namespaces or ())

    def tier(ns: str) -> int:
        if is_system_namespace(ns):
            return 3
        if own and own in ns.lower():
            return 0
        if ns in gpu:
            return 1
        return 2

    return sorted(namespaces, key=lambda ns: (tier(ns), namespaces.index(ns)))


def select_namespace(
    context: str,
    krun=default_krun,
    owner: str = "",
) -> str | None:
    """Interactive namespace selection.

    Returns the selected/created namespace name, or None if aborted.
    """
    print("\n── Namespace ──────────────────────────────────────────────────────")
    namespaces = list_namespaces(context, krun)

    if namespaces:
        # ONE screen, and the recommended action IS the Enter-key default: CREATE your own namespace.
        # Picking an existing one is offered alongside for people who know they want it. (Previously the
        # default was "[1]" = the first namespace alphabetically, which on a real cluster is `argocd` —
        # pressing Enter silently wrote NAMESPACE=argocd and would have deployed benchmarks into the
        # cluster's GitOps namespace.)
        suggested = suggested_namespace(owner)
        gpu_ns = {ns for ns in namespaces if namespace_has_gpu_workloads(context, ns, krun)}
        namespaces = rank_namespaces(namespaces, owner=owner, gpu_namespaces=gpu_ns)

        print("  Benchmarks run in their own namespace — the default creates one for you.")
        print()
        print(f"    [Enter]  Create a new namespace:  {suggested}   ← recommended")
        print("             (or type any name to create it instead)")
        print()
        print("    …or select an existing namespace to work in:")
        for i, ns in enumerate(namespaces, 1):
            marks = []
            if owner and owner.strip().lower() in ns.lower():
                marks.append("yours")
            if ns in gpu_ns:
                marks.append("has GPU workloads")
            if is_system_namespace(ns):
                marks.append("system — not for benchmarks")
            suffix = f"  ({' · '.join(marks)})" if marks else ""
            print(f"       {i}  {ns}{suffix}")
        print()

        while True:
            raw = input(f"? Enter = create '{suggested}'  ·  number = use existing  ·  or type a name: ").strip()
            if raw == "":
                if create_namespace(context, suggested, krun):
                    print(f"  ✓ Created namespace: {suggested}")
                    return suggested
                # Already exists (or RBAC said no) — if it's simply there, adopt it.
                if suggested in namespaces:
                    print(f"  ✓ Using existing namespace: {suggested}")
                    return suggested
                print("    Could not create it — pick an existing namespace by number, or type another name.")
                continue
            try:
                idx = int(raw)
            except ValueError:
                if raw in namespaces:
                    print(f"  ✓ Using namespace: {raw}")
                    return raw
                break  # a typed NEW name → Case B creates it (with validation)
            if 1 <= idx <= len(namespaces):
                chosen = namespaces[idx - 1]
                if is_system_namespace(chosen):
                    print(f"    ⚠ '{chosen}' is a cluster/system namespace — benchmarks should not run there.")
                    if input("      Use it anyway? [y/N]: ").strip().lower() not in (
                        "y",
                        "yes",
                    ):
                        continue
                print(f"  ✓ Using namespace: {chosen}")
                return chosen
            print(f"    Enter 1–{len(namespaces)}, a namespace name, or just press Enter.")

    # Case B: no namespaces visible or user chose to create.
    if not namespaces:
        print("  No accessible namespaces found (RBAC may be limiting visibility).")
    print()

    while True:
        ns_name = input("? Namespace name: ").strip()
        if not ns_name:
            print("    Namespace name cannot be empty.")
            continue
        # Validate k8s name: a namespace is a DNS-1123 label — ≤63 chars,
        # lowercase alphanumeric + hyphens, no leading/trailing hyphen (audit #9).
        import re

        if len(ns_name) > 63:
            print(
                f"    Namespace name too long ({len(ns_name)} chars). "
                "Kubernetes namespace names must be 63 characters or fewer."
            )
            continue
        if not re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", ns_name):
            print(
                "    Invalid namespace name. Use lowercase letters, numbers, and "
                "hyphens; must start and end with a letter or number."
            )
            continue
        break

    ans = input(f"  Create namespace {ns_name}? [Y/n]: ").strip().lower()
    if ans in ("", "y", "yes"):
        ok = create_namespace(context, ns_name, krun)
        if ok:
            print(f"  ✓ Created namespace: {ns_name}")
            return ns_name
        return None
    return None


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_pvcs(context: str, ns: str, krun=default_krun) -> list[str]:
    """Return PVC names in the namespace."""
    rc, out, _ = krun(
        [
            "--context",
            context,
            "-n",
            ns,
            "get",
            "pvc",
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name",
        ]
    )
    if rc != 0 or not out.strip():
        return []
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def discover_secrets(context: str, ns: str, krun=default_krun) -> list[str]:
    """Return non-default secret names in the namespace."""
    rc, out, _ = krun(
        [
            "--context",
            context,
            "-n",
            ns,
            "get",
            "secrets",
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name,TYPE:.type",
        ]
    )
    if rc != 0 or not out.strip():
        return []
    names = []
    for ln in out.strip().splitlines():
        parts = ln.split()
        if not parts:
            continue
        name = parts[0]
        # Skip service account tokens and default secrets.
        if name.startswith("default-token") or name.startswith("sh.helm"):
            continue
        names.append(name)
    return names


def discover_storage_classes(context: str, krun=default_krun) -> list[str]:
    """Return storage class names (cluster-scoped)."""
    rc, out, _ = krun(
        [
            "--context",
            context,
            "get",
            "storageclasses",
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name",
        ]
    )
    if rc != 0 or not out.strip():
        return []
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def discover_storage_classes_detailed(context: str, krun=default_krun) -> list[dict]:
    """Return storage classes WITH provisioner + is-default flag (cluster-scoped).

    Each entry: {"name", "provisioner", "is_default"}. Provisioner + the
    `storageclass.kubernetes.io/is-default-class` annotation are what let the wizard pick a SANE default
    class — an object-store CSI (e.g. s3.csi.aws.com) must never be defaulted as a block/file PVC backend,
    and that can only be known from the provisioner, not the name. Empty list on any failure (the caller
    safe-degrades). Uses `-o json` (annotations don't survive custom-columns cleanly).
    """
    rc, out, _ = krun(["--context", context, "get", "storageclasses", "-o", "json"])
    if rc != 0 or not (out or "").strip():
        return []
    try:
        items = json.loads(out).get("items", [])
    except Exception:
        return []
    result: list[dict] = []
    for sc in items:
        meta = sc.get("metadata", {}) or {}
        name = meta.get("name", "")
        if not name:
            continue
        ann = meta.get("annotations", {}) or {}
        is_default = (
            ann.get("storageclass.kubernetes.io/is-default-class") == "true"
            or ann.get("storageclass.beta.kubernetes.io/is-default-class") == "true"
        )
        result.append(
            {
                "name": name,
                "provisioner": sc.get("provisioner", "") or "",
                "is_default": bool(is_default),
            }
        )
    return result


def discover_schedulers(context: str, krun=default_krun) -> list[str]:
    """Return non-default scheduler names found as Deployments in kube-system."""
    rc, out, _ = krun(
        [
            "--context",
            context,
            "-n",
            "kube-system",
            "get",
            "deployments",
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name",
        ]
    )
    schedulers = ["default-scheduler"]
    if rc == 0 and out:
        for ln in out.strip().splitlines():
            name = ln.strip()
            if any(s in name.lower() for s in ("volcano", "yunikorn", "scheduler")):
                if name not in schedulers:
                    schedulers.append(name)
    return schedulers


def check_rbac_permissions(context: str, ns: str, krun=default_krun) -> dict[str, bool]:
    """Check whether the current user can perform key actions in the namespace."""
    checks = {
        "create_pods": ["auth", "can-i", "create", "pods", "-n", ns],
        "create_secrets": ["auth", "can-i", "create", "secrets", "-n", ns],
        "create_jobs": ["auth", "can-i", "create", "jobs", "-n", ns],
        "list_pvcs": ["auth", "can-i", "list", "persistentvolumeclaims", "-n", ns],
    }
    result: dict[str, bool] = {}
    for perm, args in checks.items():
        rc, out, _ = krun(["--context", context, *args])
        result[perm] = rc == 0 and out.strip().lower() == "yes"
    return result


# ---------------------------------------------------------------------------
# Profile write helpers (pure — testable)
# ---------------------------------------------------------------------------


def format_profile(
    cluster: str,
    context: str,
    namespace: str,
    owner: str,
    gpu_product: str,
    arch: str,
    pull_secret: str,
    hf_secret: str,
    model_cache_pvc: str,
    artifacts_sc: str,
    scheduler: str,
    bench_node_selector: str = "",
    control_sc: str = "",
    connect_cmd: str = "",
    cache_rwx_class: str = "",
    cache_rwo_class: str = "",
) -> str:
    """Format a cluster profile .env file.  Pure function for easy testing.

    `control_sc` is the RWX (ReadWriteMany) storage class for the detached `submit` control PVC. It is
    written whenever the wizard's NEW RWX detector proposed a class the operator confirmed; left absent
    (the documented `_template.env.example` placeholder governs) when undetermined, so an unconfirmed
    guess never lands as a bare value.

    ``cache_rwx_class`` and ``cache_rwo_class`` select the storage classes used for per-recipe model caches.
    ``model_cache_pvc`` remains as a compatibility override for existing profiles.
    """
    lines = [
        f"# Cluster profile: {cluster}",
        f"# Generated by profile_init.py  (llmb-k8s profile init --cluster {cluster})",
        f"# Edit and re-validate: llmb-k8s profile validate --cluster {cluster}",
        "",
        "# ----- Identity -------------------------------------------------------",
        f'CLUSTER="{cluster}"',
        f'NAMESPACE="{namespace}"',
        f'OWNER="{owner}"',
        f'KUBE_CONTEXT="{context}"',
        "# CONNECT_CMD (optional): the EXACT command to (re)connect to this cluster — echoed verbatim by",
        "# `llmb-k8s fleet` and on any auth✗/unreachable failure. Empty → tooling derives `tsh kube login`.",
        f'CONNECT_CMD="{connect_cmd}"',
        "",
        "# ----- GPU scheduling -------------------------------------------------",
        f'GPU_PRODUCT="{gpu_product}"',
        f'ARCH="{arch}"',
        f'SCHEDULER_NAME="{scheduler}"',
        "",
        "# ----- Image registry -------------------------------------------------",
        f'IMAGE_PULL_SECRET="{pull_secret}"',
        "",
        "# ----- Model / dataset cache — YOU NAME THE CLAIM ---------------------",
        "# MODEL_CACHE_PVC is REQUIRED: every rendered manifest mounts ${MODEL_CACHE_PVC}, and install",
        "# downloads into the claim resolved from THIS file, so the weights and the mount agree by",
        "# construction. Empty is NOT 'per-recipe mode' -- it is unconfigured, and install refuses before",
        "# downloading. Don't know the name? `llmb-k8s install <cluster> --adopt-cache` fills it in.",
        "# Per-model override, when one model's weights live elsewhere (slug = envelope.model upper-cased,",
        "# non-alphanumerics -> '_'):   MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4=\"nemotron-ultra-nvfp4-cache\"",
        f'MODEL_CACHE_RWX_CLASS="{cache_rwx_class}"',  # ReadWriteMany class (FSx/EFS/Filestore) — disagg/multi-worker
        f'MODEL_CACHE_RWO_CLASS="{cache_rwo_class}"',  # ReadWriteOnce class (EBS/PD) — single-node (e.g. KVBM)
        f'MODEL_CACHE_PVC="{model_cache_pvc}"',  # REQUIRED — the claim every model uses unless overridden
        "# Per-model override, when one model's weights live on a different claim (key = envelope.model,",
        "# upper-cased, non-alphanumerics -> '_'). `llmb-k8s install <cluster> --adopt-cache` writes these.",
        '#   MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4="nemotron-ultra-nvfp4-cache"',
        "# Where cache-MOUNTING pods may run. An NFS-backed RWX class may mount on only a subset of nodes",
        "# (seen live: 2 of 11; the rest fail `mount.nfs: rpc.statd is not running`). Empty = any node.",
        'MODEL_CACHE_NODE_SELECTOR=""',  # e.g. 'nvidia.com/gpu.present: "true"'
        'MODEL_CACHE_SUBPATH="."',
        "",
        "# ----- Secrets --------------------------------------------------------",
        f'HF_SECRET="{hf_secret}"',
        "",
        "# ----- Bench pod placement -------------------------------------------",
        f'BENCH_NODE_SELECTOR="{bench_node_selector}"',
        'BENCH_CPU_REQUEST="16"',
        "",
        "# ----- Artifacts (per-run bench output) ------------------------------",
        f'ARTIFACTS_STORAGE_CLASS="{artifacts_sc}"',
        'ARTIFACTS_SIZE="20Gi"',
    ]
    if control_sc:
        # RWX control-plane class for the detached `submit` lane. Only emitted when the wizard's RWX
        # detector proposed a class the operator confirmed — an unconfirmed/undetermined class is left
        # to the template placeholder rather than written as a bare (unproven) value.
        lines += [
            "",
            "# ----- Control plane (disconnect-resilient `submit`; ReadWriteMany) ---",
            f'CONTROL_STORAGE_CLASS="{control_sc}"',
            'CONTROL_SIZE="5Gi"',
        ]
    return "\n".join(lines) + "\n"


def write_profile(path: Path, content: str) -> None:
    """Write a profile file, creating parent directories as needed.

    ATOMIC + never-world-readable (the wizard contract §5/M4). The profile holds sensitive references
    (HF token secret name, NGC key path). The old `write_text()` THEN `chmod(0o600)` left a window
    where the file existed at the process umask (world-readable by default), and a crash between the
    two calls left a torn/truncated profile. We instead write to a temp file in the same directory,
    chmod 0600 on the temp (before it holds the final name), then atomically `replace()` into place —
    so the profile is never world-readable and never observed half-written. Mirrors the atomic
    tmp+chmod+replace pattern in cli/llmb-install/src/llmb_install/config/system.py.

    The temp file is UNIQUE per writer (tempfile.mkstemp → O_WRONLY|O_CREAT|O_EXCL, 0600) rather than a fixed
    `.{name}.tmp`, so concurrent writers can never collide on the temp path and tear each other's write, and
    there is no world-readable window even for the temp file itself.
    """
    import stat as _stat
    import tempfile as _tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = _tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmpname)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.chmod(tmp, _stat.S_IRUSR | _stat.S_IWUSR)  # 0600
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def maybe_write_profile(path: Path, content: str, dry_run: bool) -> bool:
    """Write the profile unless dry_run is set.  Returns True iff a file was written.

    In dry-run mode nothing is written to disk (audit #10); the caller is
    responsible for showing the rendered content instead.
    """
    if dry_run:
        return False
    write_profile(path, content)
    return True


def _prompt(prompt_text: str, default: str = "") -> str:
    """Prompt with an optional default, return the entered value (or default on Enter)."""
    default_str = f" [{default}]" if default else ""
    raw = input(f"? {prompt_text}{default_str}: ").strip()
    return raw if raw else default


def _prompt_from_list(prompt_text: str, choices: list[str], default: str = "") -> str:
    """Prompt with a discovered list shown inline.

    When a discovered set exists, an entry that is neither empty (→ default), a numeric selection (1-based),
    nor an exact member of the set is REJECTED with a warning and re-prompted — so a typo can't silently land
    as a bogus profile value. An operator who genuinely needs a value outside the discovered set can force it
    by prefixing `!` (e.g. `!my-custom-sc`), which is echoed in the warning. Free-form (no choices) is
    unchanged."""
    choices = [c for c in (choices or []) if c]
    if not choices:
        default_str = f" [{default}]" if default else ""
        raw = input(f"? {prompt_text}{default_str}: ").strip()
        return raw if raw else default
    default_str = f" [{default}]" if default else ""
    numbered = "  ".join(f"[{i}] {c}" for i, c in enumerate(choices, 1))
    while True:
        raw = input(f"? {prompt_text}  (choose: {numbered}){default_str}: ").strip()
        if not raw:
            return default
        if raw.startswith("!") and raw[1:].strip():
            return raw[1:].strip()  # explicit force of a value outside the discovered set
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            print(f"    Enter 1–{len(choices)}, a listed value, or !<custom>.")
            continue
        if raw in choices:
            return raw
        print(
            f"    ⚠ '{raw}' is not in the discovered set ({', '.join(choices)}). "
            f"Enter a number 1–{len(choices)}, one of those names, or !{raw} to force it."
        )


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------


def print_summary(
    cluster: str,
    context: str,
    namespace: str,
    owner: str,
    gpu_product: str,
    pull_secret: str,
    hf_secret: str,
    model_cache_pvc: str,
    artifacts_sc: str,
    profile_path: Path,
    cache_rwx_class: str = "",
    cache_rwo_class: str = "",
) -> None:
    print("\n── Profile written ────────────────────────────────────────────────")
    print(f"  {profile_path}")
    print()
    print(f"  Context      {context}")
    print(f"  Namespace    {namespace}")
    print(f"  Owner        {owner}")
    print(f"  GPU          {gpu_product or '(not set)'}")
    print(f"  Pull secret  {pull_secret or '(none)'}")
    print(f"  HF secret    {hf_secret or '(none)'}")
    if cache_rwx_class or cache_rwo_class or not model_cache_pvc:
        # Per-recipe cache: init records the storage-class candidates; install provisions per recipe.
        print(f"  Cache RWX SC {cache_rwx_class or '(cluster default)'}   (shared FS; disagg/multi-worker)")
        print(f"  Cache RWO SC {cache_rwo_class or '(cluster default)'}   (block; single-node, e.g. KVBM)")
    else:
        print(f"  Model PVC    {model_cache_pvc}   (back-compat single cache)")
    print(f"  Artifacts SC {artifacts_sc or '(none)'}")
    print()
    print(f"  To change anything:")
    print(f"    $EDITOR {profile_path}")
    print(f"    llmb-k8s profile validate --cluster {cluster}")


# ---------------------------------------------------------------------------
# Profile already-exists handler
# ---------------------------------------------------------------------------


def handle_existing_profile(cluster: str, profile_path: Path, force: bool) -> str:
    """If profile exists, show options and return 'validate'|'edit'|'reinit'|'abort'."""
    if force:
        print(f"  --force: re-initializing profile {cluster}")
        return "reinit"

    print(f"\n── Profile: {cluster} already exists ──────────────────────────────")
    print(f"  {profile_path}")
    print()
    print("  [v] Validate current profile")
    print("  [e] Edit in $EDITOR")
    print("  [r] Re-initialize (start fresh)")
    print("  [q] Quit")
    print()

    while True:
        raw = input("? Choose [v]: ").strip().lower()
        if raw in ("", "v"):
            return "validate"
        if raw == "e":
            return "edit"
        if raw == "r":
            ans = input("  Re-initialize will overwrite the profile. Continue? [y/N]: ").strip().lower()
            return "reinit" if ans in ("y", "yes") else "abort"
        if raw == "q":
            return "abort"
        print("    Enter v, e, r, or q.")


# ---------------------------------------------------------------------------
# Main init flow
# ---------------------------------------------------------------------------


def run_init(
    cluster: str,
    force: bool,
    dry_run: bool = False,
    krun=default_krun,
) -> int:
    profile_path = _pr.profile_env_path(cluster, PROFILES_DIR)

    # Handle existing profile.
    #   --dry-run: never mutate and never prompt — preview a re-init, leave the
    #              existing profile untouched (audit #10).
    if profile_path.exists() and dry_run:
        print(f"  --dry-run: previewing re-initialization of {cluster} " "(existing profile left untouched).")
    elif profile_path.exists():
        action = handle_existing_profile(cluster, profile_path, force)
        if action == "validate":
            print(f"  Running: llmb-k8s profile validate --cluster {cluster}")
            rc, out, _ = krun(["config", "current-context"])
            if rc != 0:
                print("  ✗ kubectl not reachable.")
                return 1
            r = _pr.resolve(cluster, profiles_dir=PROFILES_DIR)
            print(r.message)
            return 0 if r.ok else 1
        if action == "edit":
            editor = os.environ.get("EDITOR", "vi")
            os.execlp(editor, editor, str(profile_path))
            return 0  # unreachable — exec replaces process
        if action == "abort":
            print("  Aborted.")
            return 0
        # action == "reinit": fall through to init flow.

    # ── Step 1: Context selection ────────────────────────────────────────
    print("\n── Cluster context ────────────────────────────────────────────────")
    contexts = list_contexts(krun)
    if not contexts:
        print("  No kubectl contexts found.")
        print("  Set up a kubeconfig first: kubectl config get-contexts")
        return 1

    print("  Available contexts:")
    current_idx = 1
    for i, ctx in enumerate(contexts, 1):
        marker = "  (current)" if ctx["current"] else ""
        print(f"    {i}  {ctx['name']}{marker}")
        if ctx["current"]:
            current_idx = i
    print()

    while True:
        raw = input(f"? Select context for {cluster} [{current_idx}]: ").strip()
        if raw == "":
            raw = str(current_idx)
        try:
            idx = int(raw)
            if 1 <= idx <= len(contexts):
                chosen_ctx = contexts[idx - 1]["name"]
                break
            print(f"    Enter 1–{len(contexts)}.")
        except ValueError:
            # Allow typing context name directly.
            matches = [c["name"] for c in contexts if c["name"] == raw]
            if matches:
                chosen_ctx = matches[0]
                break
            print(f"    Enter a number (1–{len(contexts)}) or a context name.")

    print(f"\n  Connecting to {chosen_ctx}...", end=" ", flush=True)
    ok, detail = check_reachability(chosen_ctx, krun)
    if ok:
        gpu_nodes = probe_gpu_nodes(chosen_ctx, krun)
        if gpu_nodes:
            products = list({n["gpu_product"] for n in gpu_nodes})
            print(f"✓  ({len(gpu_nodes)} × {', '.join(products)})")
        else:
            print("✓  (no GPU nodes found)")
    else:
        print(f"✗")
        print(f"  Cannot reach {chosen_ctx}: {detail}")
        ans = input("  Continue anyway? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            return 1
        gpu_nodes = []

    # ── Step 2: Namespace selection ──────────────────────────────────────
    namespace = select_namespace(chosen_ctx, krun, owner=os.environ.get("USER", ""))
    if namespace is None:
        print("  Aborted.")
        return 1

    # ── Step 3: Discovery ────────────────────────────────────────────────
    print("\n── Discovering cluster resources ──────────────────────────────────")
    pvcs = discover_pvcs(chosen_ctx, namespace, krun)
    secrets = discover_secrets(chosen_ctx, namespace, krun)
    storage_classes = discover_storage_classes(chosen_ctx, krun)
    schedulers = discover_schedulers(chosen_ctx, krun)

    print(f"  PVCs in {namespace}:          {pvcs or '(none found)'}")
    print(f"  Secrets in {namespace}:       {secrets or '(none found)'}")
    print(f"  Storage classes:              {storage_classes or '(none found)'}")
    print(f"  Schedulers (kube-system):     {schedulers}")

    # RBAC check
    rbac = check_rbac_permissions(chosen_ctx, namespace, krun)
    rbac_warn = [k for k, v in rbac.items() if not v]
    if rbac_warn:
        print(f"\n  ⚠  Limited RBAC permissions detected: {rbac_warn}")
        print("  Some operations (secret creation, job submission) may fail.")
        print("  Raise with your cluster admin if you need these capabilities.")

    # ── Step 4: Interactive prompts ──────────────────────────────────────
    print("\n── Profile setup ──────────────────────────────────────────────────")

    # Owner
    default_owner = os.environ.get("USER", os.environ.get("USERNAME", ""))
    owner = _prompt("Owner", default_owner)

    # GPU product
    if gpu_nodes:
        gpu_products = list(dict.fromkeys(n["gpu_product"] for n in gpu_nodes))
        archs = list(dict.fromkeys(n["arch"] for n in gpu_nodes))
        default_gpu = gpu_products[0] if gpu_products else ""
        default_arch = archs[0] if archs else ""
    else:
        default_gpu = ""
        default_arch = ""

    gpu_product = _prompt("GPU product", default_gpu)
    arch = _prompt("CPU arch (amd64/arm64)", default_arch)

    # Image pull secret
    # Try to find likely NGC secrets.
    ngc_candidates = [s for s in secrets if any(k in s.lower() for k in ("ngc", "nvcr", "registry", "pull"))]
    default_pull = ngc_candidates[0] if ngc_candidates else (secrets[0] if secrets else "")
    pull_secret = _prompt_from_list("Image pull secret", ngc_candidates or secrets, default_pull)

    # HF token secret
    hf_candidates = [s for s in secrets if any(k in s.lower() for k in ("hf", "huggingface", "hugging-face", "token"))]
    default_hf = hf_candidates[0] if hf_candidates else "hf-token"
    hf_secret = _prompt_from_list("HF token secret", hf_candidates or secrets, default_hf)

    # Install provisions one model-cache PVC per recipe. The profile supplies candidate RWX and RWO
    # storage classes for those claims.
    rwx_candidates = [
        s
        for s in storage_classes
        if any(k in s.lower() for k in ("fsx", "efs", "file", "filestore", "rwx", "nfs", "lustre", "weka"))
    ]
    default_rwx = rwx_candidates[0] if rwx_candidates else (storage_classes[0] if storage_classes else "")
    cache_rwx_class = _prompt_from_list(
        "Model-cache RWX class (shared FS; disagg/multi-worker recipes)",
        rwx_candidates or storage_classes,
        default_rwx,
    )

    # RWO block class — reused for BOTH the RWO model caches (single-node recipes) and the artifacts PVC.
    rwo_candidates = [
        s
        for s in storage_classes
        if any(k in s.lower() for k in ("ebs", "gp", "ssd", "rwo", "managed", "standard", "block"))
    ]
    default_sc = rwo_candidates[0] if rwo_candidates else (storage_classes[0] if storage_classes else "")
    cache_rwo_class = _prompt_from_list(
        "Model-cache RWO class (block volume; single-node recipes, e.g. KVBM)",
        rwo_candidates or storage_classes,
        default_sc,
    )
    # Init no longer names a single model-cache PVC; leave the back-compat override empty (per-recipe path).
    model_cache_pvc = ""

    # Artifacts storage class (RWO; per-run bench output) — defaults to the same RWO class.
    artifacts_sc = _prompt_from_list(
        "Artifacts storage class (RWO)",
        rwo_candidates or storage_classes,
        cache_rwo_class or default_sc,
    )

    # Scheduler
    default_sched = schedulers[0] if schedulers else "default-scheduler"
    if len(schedulers) > 1:
        scheduler = _prompt_from_list("Scheduler name", schedulers, default_sched)
    else:
        scheduler = default_sched

    # ── Step 5: Write profile ────────────────────────────────────────────
    content = format_profile(
        cluster=cluster,
        context=chosen_ctx,
        namespace=namespace,
        owner=owner,
        gpu_product=gpu_product,
        arch=arch,
        pull_secret=pull_secret,
        hf_secret=hf_secret,
        model_cache_pvc=model_cache_pvc,
        artifacts_sc=artifacts_sc,
        scheduler=scheduler,
        cache_rwx_class=cache_rwx_class,
        cache_rwo_class=cache_rwo_class,
    )
    wrote = maybe_write_profile(profile_path, content, dry_run)

    if not wrote:
        # Dry-run: show exactly what would have been written, but touch nothing.
        print("\n── Dry run — profile NOT written ──────────────────────────────────")
        print(f"  Would write (mode 0600): {profile_path}")
        print()
        print(textwrap.indent(content, "  "))
        print("  (dry-run) nothing was written. Re-run without --dry-run to apply.")
        return 0

    # ── Step 6: Fabric auto-discovery ────────────────────────────────────
    # Probe the live cluster for RDMA/IB device names and node selector labels, then
    # append them to the profile. Disagg recipes rely on RDMA_UCX_NET_DEVICES being
    # set correctly — without this step the user would have to run probe-fabric manually.
    # Non-fatal: a cluster without RDMA HW (or no device plugin) just leaves "all".
    print("\n── Auto-discovering RDMA/IB fabric ────────────────────────────────")
    probe_script = ROOT / "scripts" / "probe_fabric.py"
    try:
        rc_probe = subprocess.run(
            [sys.executable, str(probe_script), cluster, "--write"],
            cwd=ROOT,
        ).returncode
        if rc_probe != 0:
            print("  ⚠  fabric probe failed (non-fatal) — run manually after fixing the issue:")
            print(f"     llmb-k8s profile probe-fabric --cluster {cluster} --write")
    except Exception as e:
        print(f"  ⚠  fabric probe skipped ({e}) — run manually:")
        print(f"     llmb-k8s profile probe-fabric --cluster {cluster} --write")

    # ── Step 6.5: Cluster-capability auto-discovery (Tier 1) ─────────────
    # Record reusable cluster capabilities in the profile.
    # Layer 1): the no-internet DNS + real kube-API endpoint IPs (the no-internet lane needs these), and
    # whether the ComputeDomain/IMEX machinery exists. These are "what the cluster CAN do"; recipe-scoped
    # guards (preflight) later intersect THIS recipe's needs with them. Non-fatal: an unresolved probe just
    # leaves its flag unwritten → the recipe-scoped guard safe-degrades. Auto-writing the two IPs here is
    # still Tier-1 "detect + set the flag", not Tier-2 provisioning.
    print("\n── Auto-discovering cluster capabilities ──────────────────────────")
    try:
        import capability_registry as _cap

        _krun_ctx = lambda a, timeout=30: default_krun(["--context", chosen_ctx, *a], timeout)  # noqa: E731
        _facts = _cap.gather_facts({"GPU_PRODUCT": gpu_product}, _krun_ctx)
        _flags = _cap.facts_to_profile_flags(_facts)
        if _flags:
            _txt = profile_path.read_text()
            profile_path.write_text(_cap.patch_profile_text(_txt, _flags))
            for _k, _v in _flags.items():
                print(f"  ✅ {_k}={_v}")
        else:
            print("  · no cluster-capability facts resolved (probes returned nothing) — safe to proceed;")
            print("    recipe-scoped guards will re-probe at preflight/validate")
    except Exception as e:
        print(f"  ⚠  capability probe skipped (non-fatal): {e}")

    # ── Step 6.7: Cluster readiness self-test ────────────────────────────
    # Complete + reachable ≠ run-ready. Run the SAME no-GPU battery `profile validate` runs (pull-secret auth,
    # artifacts StorageClass staging round-trip, control RWX, OIDC issuer, GPU-node availability) so onboarding
    # surfaces a cluster-config gap NOW, not on a multi-hour GPU run. Non-fatal: informational at init time —
    # a FAIL here is echoed but doesn't abort the write (the operator can fix the profile and re-validate).
    print("\n── Cluster readiness self-test ────────────────────────────────────")
    try:
        import cluster_readiness as _cr

        prof = _pr._read_env(profile_path)
        checks = _cr.run_battery(prof)
        print(_cr.format_battery(cluster, checks))
        ok_ready, _ = _cr.verdict(checks)
        if not ok_ready:
            print(f"  (fix the ❌ item(s) above, then re-check: llmb-k8s profile validate --cluster {cluster})")
    except Exception as _e:
        print(f"  ⚠  readiness self-test skipped (non-fatal): {_e}")
        print(f"     run it later: llmb-k8s profile validate --cluster {cluster}")

    # ── Step 7: Summary ──────────────────────────────────────────────────
    print_summary(
        cluster=cluster,
        context=chosen_ctx,
        namespace=namespace,
        owner=owner,
        gpu_product=gpu_product,
        pull_secret=pull_secret,
        hf_secret=hf_secret,
        model_cache_pvc=model_cache_pvc,
        artifacts_sc=artifacts_sc,
        profile_path=profile_path,
        cache_rwx_class=cache_rwx_class,
        cache_rwo_class=cache_rwo_class,
    )
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="profile_init.py — create or re-initialize a cluster profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Overwrite semantics:
              If a profile already exists, re-initializing overwrites it.
              Without --force you get an interactive menu ([v]alidate/[e]dit/
              [r]e-initialize/[q]uit) and the 'r' path asks for confirmation
              before overwriting.  --force is the explicit "yes, overwrite"
              escape hatch: it skips both the menu and the confirmation prompt.
              --dry-run never writes and never prompts — it prints the profile
              that WOULD be written and leaves any existing profile untouched.
              (--dry-run wins over --force if both are given.)

            Examples:
              scripts/profile_init.py example-gpu-cluster
              scripts/profile_init.py --cluster example-gpu-cluster
              scripts/profile_init.py example-gpu-cluster --force      # overwrite, no prompt
              scripts/profile_init.py example-gpu-cluster --dry-run    # preview, write nothing
        """),
    )
    parser.add_argument(
        "cluster",
        nargs="?",
        help="Cluster profile name (e.g. example-gpu-cluster). Positional or --cluster.",
    )
    parser.add_argument(
        "--cluster",
        dest="cluster_flag",
        metavar="CLUSTER",
        help="Named flag alias for the cluster argument.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing profile with no confirmation prompt "
        "(explicit opt-in; skips the 'already exists' menu).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the profile that would be written and exit without "
        "writing anything (leaves any existing profile untouched).",
    )

    args = parser.parse_args(argv)

    cluster = args.cluster_flag or args.cluster
    if not cluster:
        parser.error("cluster name required (positional or --cluster)")

    return run_init(cluster, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

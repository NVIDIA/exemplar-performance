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

"""Recipe-scoped cluster capability detection and safeguards.

`required_capabilities(recipe) ∩ missing_on(cluster)` determines which gaps matter for a run. Cluster facts
are gathered once, while recipe requirements are evaluated independently. Pure evaluators are shared by
preflight and rendering; live discovery receives an injectable Kubernetes runner.

The registry can also provision an NVLink IMEX ComputeDomain or remove a forced FlashInfer setting when the
required channel is unavailable. Apply-time patchers keep committed rendered manifests and recipe hashes
cluster-portable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# ── capability states (what a probe reports about the cluster) ─────────────────
AVAILABLE = "available"  # the cluster satisfies the capability right now
PROVISIONED = "provisioned"  # satisfied by an llmb-managed claim (nvlink-imex Tier-2 outcome)
ABSENT = "absent"  # the cluster cannot satisfy it (safe-degrade path)
UNKNOWN = "unknown"  # could not determine (auth lapse / probe error) → treat as safe-degrade
# FlashInfer remains forced only when the IMEX claim is provisioned and wired into the pod.
SATISFIED_STATES = (PROVISIONED,)

# ── the forced-attention-backend env that engages the NVLink allreduce_rms fusion ──
FLASHINFER_ENV_NAME = "VLLM_ATTENTION_BACKEND"
FLASHINFER_ENV_VALUE = "FLASHINFER"

# ── nvlink-imex Tier-2 provision identifiers ────────────────────────────────────
# ComputeDomain identifiers used to create one shared, idempotent channel per namespace.
IMEX_API_VERSION = "resource.nvidia.com/v1beta1"
IMEX_CD_KIND = "ComputeDomain"
IMEX_CD_NAME = "llmb-imex"  # the one shared ComputeDomain object per namespace
IMEX_CHANNEL_RCT = "llmb-imex-channel"  # ResourceClaimTemplate the CD controller materializes
IMEX_POD_CLAIM = "imex-channel"  # pod-local claim name that references the RCT
IMEX_CD_CRD = "computedomains.resource.nvidia.com"


# ── result types ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Gap:
    """The outcome of evaluating ONE required capability against the cluster facts.

    level: "PASS" (satisfied), "WARN" (missing but recoverable/degradable), "FAIL" (missing, no safe
           default — the run must not proceed as-is).
    action: machine-readable remedy the consumer applies — one of:
           None | "strip-forced-env" | "auto-write-profile" | "install-revision" | "rebuild-image".
    """

    id: str
    level: str
    state: str
    message: str
    action: Optional[str] = None
    fix: Optional[str] = None


@dataclass(frozen=True)
class Capability:
    """One gotcha, uniform contract. `requires` is THE SPINE — false ⇒ the entry is a silent no-op."""

    id: str
    title: str
    tier: int  # 1 = detect+guard (this slice) | 2 = auto-provision (later)
    ownership: str  # "self-serve" | "needs-cluster-admin" | "mixed"
    flags: tuple[str, ...]  # profile var(s) it reads/writes
    requires: Callable[[dict], bool]  # recipe -> bool  (recipe-scoping spine)
    probe: Callable[[dict], str]  # facts  -> one of the states above (PURE over facts)
    gap: Callable[[dict, dict, str], Gap]  # (recipe, facts, state) -> Gap  (PURE)
    provision: Optional[str] = None  # Tier-2 seam: user-facing verb that GRANTS it, or None
    provision_fn: Optional[Callable] = None  # Tier-2 impl: (recipe, facts, namespace) -> ProvisionPlan


# ── recipe-field accessors (pure; the ONLY place recipe shape is read) ─────────
def _envelope(recipe: dict) -> dict:
    return recipe.get("envelope") or {}


def _serving(recipe: dict) -> dict:
    return recipe.get("serving") or {}


def _gpu_type(recipe: dict) -> str:
    return str(_envelope(recipe).get("gpu_type", "")).strip().upper()


def _is_gb_class(recipe: dict) -> bool:
    """GB-class (Grace-Blackwell, NVLink-multicast fabric): GB200/GB300/…  Not B200/H100/etc."""
    return _gpu_type(recipe).startswith("GB")


def _env_forces_flashinfer(env: list) -> bool:
    """PURE. True iff an env list contains VLLM_ATTENTION_BACKEND=FLASHINFER (case-insensitive value)."""
    for e in env or []:
        if str(e.get("name")) == FLASHINFER_ENV_NAME and str(e.get("value")).upper() == FLASHINFER_ENV_VALUE:
            return True
    return False


def forces_flashinfer(recipe: dict) -> bool:
    """Does serving.env FORCE the FlashInfer attention backend (which engages the NVLink allreduce fusion)?"""
    return _env_forces_flashinfer(_serving(recipe).get("env"))


# ── entry 1: nvlink-imex (flagship) ────────────────────────────────────────────
def _req_nvlink_imex(recipe: dict) -> bool:
    """Required iff the recipe would ENGAGE the FlashInfer NVLink all-reduce fusion:
    serving.env forces VLLM_ATTENTION_BACKEND=FLASHINFER  AND  serving.tp > 1  AND  GB-class GPU.
    tp=1, unforced, or non-GB never engages the multicast fusion → the entry is a no-op (silent).
    """
    tp = int(_serving(recipe).get("tp", 0) or 0)
    return forces_flashinfer(recipe) and tp > 1 and _is_gb_class(recipe)


def _probe_nvlink_imex(facts: dict) -> str:
    """PURE. facts["nvlink_imex"] = {crd_present: bool, channel_provisioned: bool}.
    provisioned (llmb claim wired) > available (machinery present, unclaimed) > absent (no CD/IMEX).
    """
    f = facts.get("nvlink_imex")
    if not isinstance(f, dict):
        return UNKNOWN
    if f.get("channel_provisioned"):
        return PROVISIONED
    if f.get("crd_present"):
        return AVAILABLE  # machinery exists; Tier-1 still renders as absent (safe) until claimed
    return ABSENT


def _gap_nvlink_imex(recipe: dict, facts: dict, state: str) -> Gap:
    if state == PROVISIONED:
        return Gap(
            "nvlink-imex",
            "PASS",
            state,
            "NVLINK_MULTICAST_IMEX=provisioned — ComputeDomain claim wired; FlashInfer " "allreduce_rms fusion ACTIVE",
        )
    # available|absent|unknown → Tier-1 GUARD: strip the forced backend so cuMulticastCreate can't crash.
    recover = (
        (
            "this cluster CAN provision it — recover the fusion (Tier 2): "
            "llmb-k8s profile provision-imex --cluster <c>"
        )
        if state == AVAILABLE
        else ("this cluster has no ComputeDomain/IMEX channel provisioning")
    )
    return Gap(
        "nvlink-imex",
        "WARN",
        state,
        f"NVLINK_MULTICAST_IMEX={state} — {recover}. The forced VLLM_ATTENTION_BACKEND=FLASHINFER will be "
        "STRIPPED at APPLY time (deploy.sh → merge_imex_strip.py) so the server cannot CrashLoop on "
        "cuMulticastCreate code=800; the committed rendered/ bytes are unchanged (recipe_hash-neutral) and "
        "vLLM auto-disables only the allreduce_rms fusion (throughput only, zero quality impact).",
        action="strip-forced-env",
        fix=("llmb-k8s profile provision-imex --cluster <c>" if state == AVAILABLE else None),
    )


def relax_serving_env(serving: dict, imex_state: str) -> tuple[list, list[str]]:
    """PURE strip decision. Given serving.env (or a single container's env) and the resolved IMEX state,
    return (kept_env, stripped_names): drop every fusion-forcing env when IMEX is NOT provisioned so the
    server cannot crash. When PROVISIONED the env passes through untouched (the channel claim is wired, so
    cuMulticastCreate succeeds and the fusion runs).

    The decision is implemented here and reused by ``strip_forced_flashinfer`` and the apply-time patcher.
    Committed rendered manifests remain unchanged. Without IMEX, the forced backend is removed; when IMEX is
    provisioned, the environment is left unchanged.
    """
    env = list(serving.get("env") or [])
    if imex_state in SATISFIED_STATES:
        return env, []
    kept, stripped = [], []
    for e in env:
        if str(e.get("name")) == FLASHINFER_ENV_NAME and str(e.get("value")).upper() == FLASHINFER_ENV_VALUE:
            stripped.append(str(e.get("name")))
        else:
            kept.append(e)
    return kept, stripped


def strip_forced_flashinfer(doc: dict, imex_state: str) -> tuple[dict, int]:
    """Remove a forced FlashInfer backend when IMEX is unavailable.

    The function updates serving Deployment containers in place and returns the document plus the number of
    removed environment entries. A zero count means IMEX is available or no applicable setting was present.
    Deployment applies this transformation without changing committed rendered manifests.
    """
    if imex_state in SATISFIED_STATES:
        return doc, 0
    if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
        return doc, 0
    try:
        pod_spec = doc["spec"]["template"]["spec"]
    except (KeyError, TypeError):
        return doc, 0
    if not isinstance(pod_spec, dict):
        return doc, 0
    n = 0
    for c in pod_spec.get("containers") or []:
        if not _container_requests_gpu(c):
            continue
        kept, stripped = relax_serving_env({"env": c.get("env")}, imex_state)
        if stripped:
            c["env"] = kept
            n += len(stripped)
    return doc, n


# ── entry 1 (Tier 2): nvlink-imex auto-provision — RECOVER the fusion ───────────
# The flagship Tier-2 path. Where the cluster can self-serve a ComputeDomain (GB300 today), we PROVISION
# the IMEX channel and keep FlashInfer ON so `allreduce_rms` NVLink multicast fusion runs — instead of the
# Tier-1 degrade that strips FLASHINFER and eats the throughput hit. All builders here are PURE (no cluster
# calls): the impure edges are `scripts/provision_imex.py` (kubectl apply of the ComputeDomain) and
# `scripts/merge_imex_claim.py` (apply-time injection of the pod claim). recipe_hash never moves because the
# claim is injected in the deploy.sh pipeline, NOT baked into committed rendered/ manifests (design §4.6).
def computedomain_manifest(namespace: str, name: str = IMEX_CD_NAME, channel_rct: str = IMEX_CHANNEL_RCT) -> dict:
    """PURE. The (A) ComputeDomain object (design §4.2), matching the live GB300 v1beta1 CRD schema.
    numNodes:0 → self-gating; allocationMode Single → one IMEX channel shared across the pod's TP GPUs
    (one node, one IMEX domain) — exactly what cuMulticastCreate(FABRIC) needs. The CD controller then
    materializes a ResourceClaimTemplate named `channel_rct` targeting DeviceClass
    compute-domain-default-channel.nvidia.com, which the serving pod claims."""
    return {
        "apiVersion": IMEX_API_VERSION,
        "kind": IMEX_CD_KIND,
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"},
        },
        "spec": {
            "numNodes": 0,
            "channel": {
                "allocationMode": "Single",
                "resourceClaimTemplate": {"name": channel_rct},
            },
        },
    }


def _container_requests_gpu(container: dict) -> bool:
    """PURE. True iff the container requests nvidia.com/gpu (so it's a fusion-init container to wire)."""
    res = container.get("resources") or {}
    for scope in ("requests", "limits"):
        if str((res.get(scope) or {}).get("nvidia.com/gpu", "")).strip():
            return True
    return False


def _container_needs_imex(container: dict) -> bool:
    """PURE. True iff the container ENGAGES the NVLink fusion — it requests a GPU AND forces FLASHINFER.
    Recipe-scoping mirrors provision_imex._req_nvlink_imex at the container level: a tp=1 / non-fusion /
    non-forcing GPU container sharing a provisioned profile does NOT need (and must not get) the claim.
    """
    return _container_requests_gpu(container) and _env_forces_flashinfer(container.get("env"))


def inject_imex_claim(
    doc: dict, pod_claim: str = IMEX_POD_CLAIM, channel_rct: str = IMEX_CHANNEL_RCT
) -> tuple[dict, int]:
    """PURE, in-place. Wire the (B) DRA claim (design §4.2) into a serving Deployment `doc`:
      pod.spec.resourceClaims  += {name: pod_claim, resourceClaimTemplateName: channel_rct}
      each fusion-engaging container.resources.claims += {name: pod_claim}
    Injection is gated on the container actually FORCING FLASHINFER (not merely requesting a GPU) so a
    non-fusion / tp=1 / non-FLASHINFER cell sharing a provisioned profile never gets a claim it doesn't
    need — recipe-scoped, not profile-scoped (mirrors provision_imex._req_nvlink_imex).
    Idempotent (never double-adds). Returns (doc, n_containers_wired); n=0 ⇒ no fusion container, untouched.
    This is what merge_imex_claim.py applies to every manifest in the deploy.sh apply pipeline.
    """
    if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
        return doc, 0
    try:
        pod_spec = doc["spec"]["template"]["spec"]
    except (KeyError, TypeError):
        return doc, 0
    if not isinstance(pod_spec, dict):
        return doc, 0
    gpu_containers = [c for c in (pod_spec.get("containers") or []) if _container_needs_imex(c)]
    if not gpu_containers:
        return (
            doc,
            0,
        )  # no fusion container ⇒ nothing to wire (e.g. non-FLASHINFER cell)
    claims = pod_spec.setdefault("resourceClaims", [])
    if not any(isinstance(c, dict) and c.get("name") == pod_claim for c in claims):
        claims.append({"name": pod_claim, "resourceClaimTemplateName": channel_rct})
    n = 0
    for c in gpu_containers:
        cclaims = c.setdefault("resources", {}).setdefault("claims", [])
        if not any(isinstance(x, dict) and x.get("name") == pod_claim for x in cclaims):
            cclaims.append({"name": pod_claim})
        n += 1
    return doc, n


@dataclass(frozen=True)
class ProvisionPlan:
    """The per-cluster Tier decision for a provisionable capability. PURE output of `plan_*`.

    tier: 2 (provision — recover the capability) | 1 (degrade — Tier-1 safe fallback) | 0 (no-op).
    action: "provision" | "already-provisioned" | "degrade" | "noop".
    keep_flashinfer: whether the fusion-forcing env stays ON (True only when the channel is/was provisioned).
    manifests: ComputeDomain dict(s) to kubectl apply (empty unless action == "provision").
    profile_flags: profile vars the verb writes on success.
    message: ONE plain-English operator line (the tradeoff / what happened / how to recover).
    """

    tier: int
    action: str
    keep_flashinfer: bool
    manifests: tuple
    profile_flags: dict
    message: str


def plan_nvlink_imex(recipe: dict, facts: dict, namespace: str) -> ProvisionPlan:
    """PURE Tier-2 decision for the flagship. The verb (provision_imex.py) executes the returned plan.

    Recipe-scoping is the spine: a recipe that doesn't ENGAGE the fusion provisions NOTHING (noop). Then:
      • already provisioned      → keep FLASHINFER, nothing to apply.
      • CRD present + self-serve  → PROVISION: apply the ComputeDomain, wire the pod claim, keep FLASHINFER
                                    (RECOVER the throughput win — the purpose of Tier 2).
      • CRD or RBAC absent        → DEGRADE to Tier 1: strip FLASHINFER (correct-but-slower), with ONE line
                                    explaining the throughput tradeoff and exactly how to earn the capability.
    Never crash-loops; never silently drops the fusion without saying so."""
    if not _req_nvlink_imex(recipe):
        return ProvisionPlan(
            0,
            "noop",
            True,
            (),
            {},
            "nvlink-imex not required by this recipe (no forced FLASHINFER, or tp<=1, or non-GB) — "
            "nothing to provision.",
        )
    ns = str(namespace or "").strip()
    state = _probe_nvlink_imex(facts)
    if state == PROVISIONED:
        return ProvisionPlan(
            2,
            "already-provisioned",
            True,
            (),
            {
                "NVLINK_MULTICAST_IMEX": PROVISIONED,
                "IMEX_CLAIM_TEMPLATE": IMEX_CHANNEL_RCT,
            },
            f"NVLink IMEX already provisioned (ComputeDomain '{IMEX_CD_NAME}' in {ns or '<ns>'}): the "
            "FlashInfer allreduce_rms fusion stays ACTIVE — nothing to do.",
        )
    f = facts.get("nvlink_imex") or {}
    if f.get("crd_present") and f.get("rbac_can_create") and ns:
        return ProvisionPlan(
            2,
            "provision",
            True,
            (computedomain_manifest(ns),),
            {
                "NVLINK_MULTICAST_IMEX": PROVISIONED,
                "IMEX_CLAIM_TEMPLATE": IMEX_CHANNEL_RCT,
            },
            f"NVLink IMEX is self-serve here — provisioning ComputeDomain '{IMEX_CD_NAME}' in {ns} and "
            "wiring the serving pod's channel claim. VLLM_ATTENTION_BACKEND=FLASHINFER stays ON and the "
            "allreduce_rms NVLink multicast fusion is RECOVERED (throughput win, no quality change).",
        )
    # Degrade safely when the CRD or required RBAC is unavailable.
    if f.get("crd_present"):
        why = "this namespace cannot create ComputeDomains (no self-serve RBAC)"
        how = (
            "ask a cluster-admin to grant `create` on computedomains.resource.nvidia.com in "
            f"namespace {ns or '<ns>'}"
        )
    else:
        why = "this cluster has no ComputeDomain/IMEX machinery (DRA driver / ComputeDomain operator absent)"
        how = "ask a cluster-admin to install the NVIDIA DRA driver + ComputeDomain operator"
    return ProvisionPlan(
        1,
        "degrade",
        False,
        (),
        {"NVLINK_MULTICAST_IMEX": AVAILABLE if f.get("crd_present") else ABSENT},
        f"NVLink IMEX cannot be auto-provisioned — {why}. Falling back to Tier-1: the forced "
        "VLLM_ATTENTION_BACKEND=FLASHINFER is STRIPPED so the server runs correct-but-slower (the "
        "allreduce_rms NVLink fusion stays OFF — throughput only, zero quality impact, never a CrashLoop on "
        f"cuMulticastCreate code=800). To recover the throughput: {how}, then re-run "
        "`llmb-k8s profile provision-imex --cluster <c>`.",
    )


def provision_plan(
    recipe: dict,
    facts: dict,
    namespace: str,
    cap_id: str = "nvlink-imex",
    registry=None,
) -> ProvisionPlan:
    """Dispatch to the registry entry's Tier-2 `provision_fn`. Pure. Returns a noop plan if the entry has
    no provisioner (Tier-1-only capability)."""
    reg = registry if registry is not None else REGISTRY
    for e in reg:
        if e.id == cap_id and e.provision_fn is not None:
            return e.provision_fn(recipe, facts, namespace)
    return ProvisionPlan(0, "noop", True, (), {}, f"{cap_id}: no Tier-2 provisioner registered.")


# ── entry 2: no-internet-ips ────────────────────────────────────────────────────
def _req_no_internet_ips(recipe: dict) -> bool:
    """Required iff the recipe declares deploy.no_internet (or legacy live.no_internet)."""
    deploy = recipe.get("deploy") or recipe.get("live") or {}
    return bool(deploy.get("no_internet", False))


def _probe_no_internet_ips(facts: dict) -> str:
    """PURE. facts["no_internet"] = {dns_ip, kube_api_ip}. available iff BOTH resolved (non-empty)."""
    f = facts.get("no_internet")
    if not isinstance(f, dict):
        return UNKNOWN
    if str(f.get("dns_ip") or "").strip() and str(f.get("kube_api_ip") or "").strip():
        return AVAILABLE
    return ABSENT


def _gap_no_internet_ips(recipe: dict, facts: dict, state: str) -> Gap:
    if state == AVAILABLE:
        f = facts["no_internet"]
        return Gap(
            "no-internet-ips",
            "PASS",
            state,
            f"NO_INTERNET_DNS_IP={f['dns_ip']} / NO_INTERNET_KUBE_API_IP={f['kube_api_ip']} "
            "resolved — the no-internet NetworkPolicy allowlist will let task pods reach kube-dns "
            "and the real kube-API endpoint",
        )
    return Gap(
        "no-internet-ips",
        "FAIL",
        state,
        "NO_INTERNET_DNS_IP / NO_INTERNET_KUBE_API_IP unresolved — a no-internet session "
        "blocks the .1 kube-API Service ClusterIP, so DNS + the real kube-API endpoint IP must be pinned "
        "in the NetworkPolicy allowlist or the task pods hang at readiness.",
        action="auto-write-profile",
        fix="llmb-k8s profile init/validate re-probes and auto-writes both IPs; "
        "or set NO_INTERNET_DNS_IP + NO_INTERNET_KUBE_API_IP in the profile",
    )


# ── entry 3: model-revision-cached ──────────────────────────────────────────────
def _req_model_revision(recipe: dict) -> bool:
    """Required iff the recipe PINS a serving.model_revision (an exact commit). Unpinned → nothing to check."""
    return bool(str(_serving(recipe).get("model_revision", "")).strip())


def _cached_revisions(facts: dict, repo: str) -> Optional[list[str]]:
    """The cached snapshot shas for `repo`, or None if the cache wasn't probed. Accepts either a flat
    {revisions: [...]} or a per-repo {revisions: {repo: [...]}} facts shape."""
    f = facts.get("model_cache")
    if not isinstance(f, dict):
        return None
    revs = f.get("revisions")
    if isinstance(revs, dict):
        return list(revs.get(repo, []))
    if isinstance(revs, list):
        return list(revs)
    return None


def _probe_model_revision(facts: dict) -> str:
    """PURE, but repo/revision-specific ⇒ decided in the gap (needs the recipe). Report machinery state:
    available when a cache probe returned a revision set, absent/unknown otherwise."""
    f = facts.get("model_cache")
    if not isinstance(f, dict) or f.get("revisions") is None:
        return UNKNOWN
    return AVAILABLE


def _gap_model_revision(recipe: dict, facts: dict, state: str) -> Gap:
    serving = _serving(recipe)
    rev = str(serving.get("model_revision", "")).strip()
    repo = str(serving.get("model_repo", "")).strip()
    cached = _cached_revisions(facts, repo)
    if cached is None:
        return Gap(
            "model-revision-cached",
            "WARN",
            UNKNOWN,
            f"could not probe the model cache for revision {rev[:12]} — the server downloads (slow) "
            "or, with HF_HUB_OFFLINE, hard-fails with LocalEntryNotFoundError. Pre-stage the weights.",
            action=None,
            fix=f"llmb-k8s install --cluster <c>   # stage {repo}@{rev[:12]}",
        )
    if rev in cached:
        return Gap(
            "model-revision-cached",
            "PASS",
            AVAILABLE,
            f"model weights: snapshot {rev[:12]} present in the cluster cache",
        )
    have = ", ".join(s[:12] for s in cached) or "none"
    return Gap(
        "model-revision-cached",
        "WARN",
        ABSENT,
        f"revision {rev[:12]} for {repo} is NOT cached (cached: {have}) — the server would hit "
        "LocalEntryNotFoundError under HF_HUB_OFFLINE. Pin one of the cached revisions or install this one.",
        action="install-revision",
        fix=f"llmb-k8s install --cluster <c>   # download {rev[:12]}   (or pin a cached sha: {have})",
    )


# ── entry 4: image-arch ─────────────────────────────────────────────────────────
def pinned_images(recipe: dict) -> list[str]:
    """Every OCI image ref the recipe pins (envelope.provenance + any serving image fields). Pure.

    Covers the envelope serving image (provenance.image_ref/image_digest). Each is a pinned digest that
    can silently ImagePullBackOff / `exec format error` if its manifest lacks the target node arch
    (the image-arch gate keys on exactly this set)."""
    imgs: list[str] = []
    prov = _envelope(recipe).get("provenance") or {}
    for k in ("image_ref", "image_digest"):
        v = str(prov.get(k, "")).strip()
        if v:
            imgs.append(v)
    lv = str(_serving(recipe).get("launcher_image", "")).strip()
    if lv:
        imgs.append(lv)
    return sorted(set(imgs))


def _req_image_arch(recipe: dict) -> bool:
    """Required for ANY recipe that declares an arch — a single-platform image on the wrong node fails at
    run, not schedule, so every recipe checks its OWN pinned images cover its target arch.
    """
    return bool(str(_envelope(recipe).get("arch", "")).strip())


def _probe_image_arch(facts: dict) -> str:
    """PURE. facts["images"] = {image_ref: [platform_arch,...]} and facts["node_archs"] = [arch,...].
    Coverage is decided per-recipe in the gap; here report whether we have anything to check against.
    """
    if facts.get("images") is None and facts.get("node_archs") is None:
        return UNKNOWN
    return AVAILABLE


def _gap_image_arch(recipe: dict, facts: dict, state: str) -> Gap:
    want = str(_envelope(recipe).get("arch", "")).strip()
    node_archs = [a for a in (facts.get("node_archs") or []) if a]
    images = facts.get("images")
    # 1) node arch must be able to run the recipe's target (cheap, always available in preflight).
    if node_archs and want and want not in node_archs:
        return Gap(
            "image-arch",
            "FAIL",
            ABSENT,
            f"arch mismatch — recipe arch={want} but the cluster nodes are {sorted(set(node_archs))}; "
            "the pinned single-platform image cannot run here (fails at run, not schedule).",
            action="rebuild-image",
            fix="deploy on a matching-arch cluster: llmb-k8s profile list",
        )
    # 2) each pinned image's manifest-list must cover the target arch (the registry's real addition).
    if isinstance(images, dict) and want:
        if images:
            uncovered = [ref for ref, plats in images.items() if want not in (plats or [])]
            if uncovered:
                shown = ", ".join(r.split("@")[0].split("/")[-1] for r in uncovered)
                return Gap(
                    "image-arch",
                    "FAIL",
                    ABSENT,
                    f"pinned image(s) not a manifest list covering {want}: {shown} — the image will "
                    "fail to pull/run on this node arch.",
                    action="rebuild-image",
                    fix="rebuild the image as a multi-arch manifest list covering "
                    f"{want}, or pin an image that already does",
                )
        elif pinned_images(recipe):
            # Minor A: images is an EMPTY dict — the recipe pins image(s) but EVERY one failed registry
            # inspection (all omitted, safe-degrade). Node-arch matched, but manifest coverage was NOT
            # verified, so don't imply it was: degrade this half to UNKNOWN (WARN), never a silent PASS.
            return Gap(
                "image-arch",
                "WARN",
                UNKNOWN,
                f"node arch matches arch={want}, but could NOT inspect any pinned image's manifest "
                "to verify it is a manifest list covering that arch (registry unreachable / no "
                "inspector) — manifest coverage UNVERIFIED, proceeding safe-degrade.",
            )
    if state == UNKNOWN:
        return Gap(
            "image-arch",
            "WARN",
            UNKNOWN,
            f"could not verify the pinned image(s) are a manifest list covering arch={want}",
        )
    return Gap(
        "image-arch",
        "PASS",
        AVAILABLE,
        f"pinned image(s) cover the recipe arch={want} and match the cluster node arch",
    )


# ── entry 4 (fill the gap): FEED facts["images"] with each pinned image's manifest arch list ─────────
# `_gap_image_arch` already flags a pinned digest whose registry manifest does NOT cover the target node
# arch (the silent ImagePullBackOff / `exec format error`), but gather_facts never populated
# facts["images"], so that half of the gate was DEAD. Fill it: inspect each pinned image's manifest and
# record its platform/arch list. Best-effort + safe-degrade — an image we cannot inspect (registry
# unreachable / no tool) is OMITTED (never flagged), so a probe failure downgrades to UNKNOWN, never a
# false FAIL. This FEEDS the existing entry (preferred over a new one).
_INDEX_MEDIA_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)


def arches_from_raw_manifest(raw) -> Optional[list[str]]:
    """PURE. Parse a raw registry manifest/index (JSON str or dict) → sorted arch list when it is a
    MULTI-ARCH index (a top-level `manifests` array), skipping attestation/`unknown` entries. Returns None
    for a single-image manifest (its arch is not carried in the manifest — the caller resolves it via a
    second non-raw inspect) or unparseable input. Mirrors verify_multiarch_manifest.platforms_of_raw but
    returns bare arch tokens (arm64/amd64) to compare against envelope.arch."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    manifests = raw.get("manifests")
    if not isinstance(manifests, list):
        return None  # single-image manifest (config+layers) — arch not here
    arches: list[str] = []
    for m in manifests:
        if not isinstance(m, dict):
            continue
        plat = m.get("platform") or {}
        arch = str(plat.get("architecture", "")).strip()
        os_ = str(plat.get("os", "")).strip()
        ann = m.get("annotations") or {}
        if not arch or arch == "unknown" or os_ == "unknown":
            continue  # buildkit provenance/attestation rows carry arch "unknown"
        if ann.get("vnd.docker.reference.type") == "attestation-manifest":
            continue
        arches.append(arch)
    return sorted(set(arches))


def _gather_image_archs(recipe: dict, profile: dict, inspect: Callable) -> dict:
    """{image_ref: [arch,...]} for each pinned image, via the injected `inspect(ref) -> Optional[list]`.
    Images inspect can't resolve (None / error) are OMITTED — safe-degrade, never a false FAIL. Pure given
    a pure `inspect` (the selftests inject a canned inspect; the live default shells out to skopeo/docker).
    """
    out: dict = {}
    for ref in pinned_images(recipe):
        try:
            arches = inspect(ref)
        except Exception:
            arches = None
        if arches:
            out[ref] = sorted(set(str(a).strip() for a in arches if str(a).strip()))
    return out


def _registry_authfile(profile: dict, krun: Callable) -> Optional[str]:
    """Best-effort docker authfile materialized from the profile's IMAGE_PULL_SECRET (read via kubectl),
    so skopeo can inspect a PRIVATE registry manifest. Returns a temp path (caller unlinks) or None
    (→ unauthenticated inspect; nvcr Tier-2 reads work unauthenticated). Any failure → None (safe-degrade).
    """
    import base64
    import os as _os
    import tempfile

    ns = str(profile.get("NAMESPACE") or "").strip()
    sec = str(profile.get("IMAGE_PULL_SECRET") or "").strip()
    if not (ns and sec):
        return None
    rc, out, _ = krun(["-n", ns, "get", "secret", sec, "-o", "jsonpath={.data.\\.dockerconfigjson}"])
    if rc != 0 or not (out or "").strip():
        return None
    try:
        cfg = base64.b64decode(out.strip())
    except Exception:
        return None
    fd, path = tempfile.mkstemp(prefix="llmb-authfile-", suffix=".json")
    try:
        with _os.fdopen(fd, "wb") as f:
            f.write(cfg)
    except Exception:
        try:
            _os.unlink(path)
        except OSError:
            pass
        return None
    return path


def _run_inspect(cmd: list, timeout: int = 25) -> tuple[int, str, str]:
    """IMPURE. Run a registry-inspect subprocess; ANY error (missing tool / timeout / network) → rc=1."""
    import subprocess

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def _default_inspect_image(ref: str, profile: dict, krun: Callable) -> Optional[list[str]]:
    """IMPURE live inspect of one pinned image's registry manifest → arch list, or None (unreachable / no
    tool → safe-degrade). skopeo `inspect --raw` primary (else docker buildx imagetools); a single-image
    manifest resolves its concrete arch via a second non-raw skopeo inspect (.Architecture).
    """
    import os as _os
    import shutil

    authfile = None
    try:
        authfile = _registry_authfile(profile, krun)
        af = ["--authfile", authfile] if authfile else []
        raw = None
        if shutil.which("skopeo"):
            rc, out, _ = _run_inspect(["skopeo", "inspect", "--raw", *af, f"docker://{ref}"])
            if rc == 0 and (out or "").strip():
                raw = out
        if raw is None and shutil.which("docker"):
            rc, out, _ = _run_inspect(["docker", "buildx", "imagetools", "inspect", ref, "--raw"])
            if rc == 0 and (out or "").strip():
                raw = out
        if raw is None:
            return None
        arches = arches_from_raw_manifest(raw)
        if arches is not None:
            return arches  # multi-arch index → its platform arch list
        if shutil.which("skopeo"):  # single-image manifest → resolve its one concrete arch
            rc, out, _ = _run_inspect(["skopeo", "inspect", *af, f"docker://{ref}"])
            if rc == 0 and (out or "").strip():
                try:
                    a = str(json.loads(out).get("Architecture", "")).strip()
                    return [a] if a else None
                except Exception:
                    return None
        return None
    finally:
        if authfile:
            try:
                _os.unlink(authfile)
            except OSError:
                pass


# ── entry 5: nvlink-p2p ──────────────────────────────────────────────────────
# Detect whether multi-GPU collectives have a healthy NVLink P2P fabric.
# Remediation belongs to the cluster administrator; recipes cannot repair the fabric.
def _p2p_tp(recipe: dict) -> int:
    """PURE. Max tensor-parallel width the recipe engages across aggregated + disagg roles. TP>1 is what
    puts a P2P all-reduce/all-gather on the NVLink fabric; TP=1 never touches inter-GPU P2P.
    """
    s = _serving(recipe)
    tps = [int(s.get("tp", 0) or 0)]
    disagg = s.get("disagg") or {}
    for role in ("prefill", "decode"):
        blk = disagg.get(role) or {}
        tps.append(int(blk.get("tp", 0) or 0))
    return max(tps) if tps else 0


def _req_nvlink_p2p(recipe: dict) -> bool:
    """Required iff the recipe ENGAGES multi-GPU P2P collectives — serving.tp > 1 (or a disagg role tp>1).
    A TP=1 recipe never crosses the NVLink P2P fabric, so the entry is a silent no-op (respects the spine:
    never flag a capability a recipe doesn't need). Unlike nvlink-imex this is NOT GB-class-gated — a
    disabled P2P fabric degrades TP>1 on B200/GB200/GB300 alike."""
    return _p2p_tp(recipe) > 1


def _probe_nvlink_p2p(facts: dict) -> str:
    """PURE. facts["nvlink_p2p"] = {state: healthy|disabled|unknown, ...} (from the GPU-scheduled
    nvidia-smi probe, recorded in the profile). healthy→AVAILABLE, disabled→ABSENT, else UNKNOWN
    (safe-degrade: an un-probed / un-parseable fabric never blocks a run)."""
    f = facts.get("nvlink_p2p")
    if not isinstance(f, dict):
        return UNKNOWN
    st = str(f.get("state", "")).strip().lower()
    if st == "healthy":
        return AVAILABLE
    if st == "disabled":
        return ABSENT
    return UNKNOWN


def _p2p_result_is_perf_invariant(recipe: dict) -> bool:
    """PURE. Is THIS run's headline metric independent of throughput? Currently always False —
    every llm-perf scenario reports a throughput-derived number that a disabled fabric would INVALIDATE.
    """
    return False


def _gap_nvlink_p2p(recipe: dict, facts: dict, state: str) -> Gap:
    tp = _p2p_tp(recipe)
    scenario = str(_envelope(recipe).get("scenario", "")).strip() or "this"
    if state == AVAILABLE:
        return Gap(
            "nvlink-p2p",
            "PASS",
            state,
            "NVLINK_P2P=healthy — `nvidia-smi topo -p2p rw` shows OK off-diagonal; the "
            f"TP={tp} NVLink P2P collectives will run at full fabric bandwidth",
        )
    if state == UNKNOWN:
        return Gap(
            "nvlink-p2p",
            "WARN",
            state,
            "NVLINK_P2P=unknown — could not probe NVLink P2P health (no GPU probe recorded / probe "
            "pod didn't schedule / no nvidia-smi). Proceeding safe-degrade; if throughput looks "
            "degraded or vLLM EngineCore crashes, check `nvidia-smi topo -p2p rw` for all-NS "
            "(fabric P2P disabled) — a cluster-admin fabric-manager fix, not a recipe issue.",
        )
    # state == ABSENT → P2P disabled fabric-wide. Severity is SCENARIO-AWARE (the purpose):
    detail = ""
    f = facts.get("nvlink_p2p") or {}
    if f.get("route_unhealthy") is True:
        detail = " (Fabric Health `Route Unhealthy: True`)"
    owner = (
        f"This is a CLUSTER FABRIC fault — a cluster-admin must restore NVLink P2P via "
        f"nv-fabricmanager / IMEX route recovery{detail}; it is NOT a recipe problem."
    )
    return Gap(
        "nvlink-p2p",
        "FAIL",
        state,
        f"NVLINK_P2P=disabled — NVLink P2P is OFF fabric-wide, so a TP={tp} collective is serialized: this "
        f"{scenario} throughput/goodput result would be INVALID (degraded perf; vLLM EngineCore can also "
        "crash without an appropriate runtime fallback). Do not launch a throughput run on a "
        f"cluster whose P2P fabric is unavailable. {owner}",
        action=None,
        fix="cluster-admin: restore NVLink P2P (nv-fabricmanager / IMEX route recovery) until "
        "`nvidia-smi topo -p2p rw` shows OK off-diagonal and Fabric Health Summary=Healthy; "
        "then re-run.",
    )


# ── entry 6: model-cache-integrity ──────────────────────────────────────────
# Verify the exact pinned Hugging Face snapshot path and all resolved shards.
# The PVC probe is performed by preflight; these helpers remain pure.
_MODEL_CACHE_REQUIRED_FILES = ("config.json",)


def hf_repo_dir(repo: str) -> str:
    """PURE. huggingface_hub on-disk cache dir for a repo: 'org/name' -> 'models--org--name'."""
    return "models--" + str(repo).strip().strip("/").replace("/", "--")


def hf_repo_cache_dir(subpath: str, repo: str) -> str:
    """PURE. The repo-level cache dir (holds snapshots/ + refs/): <subpath>/hub/models--<org>--<name>.
    `subpath` of '' or '.' means the cache root is the mount root."""
    base = str(subpath or "").strip().strip("/")
    base = "" if base in ("", ".") else base + "/"
    return f"{base}hub/{hf_repo_dir(repo)}"


def server_snapshot_dir(subpath: str, repo: str, revision: str) -> str:
    """Return the exact Hugging Face snapshot directory resolved under the mounted cache."""
    return f"{hf_repo_cache_dir(subpath, repo)}/snapshots/{str(revision).strip()}"


def model_cache_verdict(probe: dict) -> str:
    """PURE. Turn a cache-path probe into AVAILABLE / ABSENT / UNKNOWN for the preflight gate.

    THE PREDICATE ITSELF NOW LIVES IN model_cache.cache_completeness — one function, so install ("should I
    download this?"), preflight ("may this run start?") and fleet ("what is on this claim?") cannot hold
    three different opinions about the same bytes. They did: fleet rendered a claim as
    "✓ downloaded · verified by PVC stamp" (it was reading the PVC label) at the same moment install called
    the same claim not-installed (it was reading a sentinel that did not exist).

    This function only MAPS that shared verdict onto preflight's tri-state, and the mapping is deliberately
    conservative: PRESENT_UNVERIFIED -> ABSENT. For a RUN that is right — an unproven cache crash-loops the
    server after it has taken the GPUs, so preflight must block. install treats the same state differently
    (it can verify-and-stamp, or ask) because it has a cheap remedy available and no GPUs at stake.
    """
    import model_cache as _mc

    state, _why = _mc.cache_completeness(probe)
    if state == _mc.STATE_UNKNOWN:
        return UNKNOWN
    return AVAILABLE if state == _mc.STATE_COMPLETE else ABSENT


def _req_model_cache_integrity(recipe: dict) -> bool:
    """Required for any recipe that serves a CACHED model at a pinned snapshot — it pins BOTH
    serving.model_repo and serving.model_revision (essentially all serving recipes). Without a pinned
    revision there is no exact snapshot path to verify (model-revision-cached / check_invariants warn to
    pin it), so the entry is a silent no-op."""
    s = _serving(recipe)
    return bool(str(s.get("model_repo", "")).strip()) and bool(str(s.get("model_revision", "")).strip())


def _probe_model_cache_integrity(facts: dict) -> str:
    """PURE over facts['model_cache_integrity'] (populated by preflight's live read-only PVC-mount probe)."""
    return model_cache_verdict(facts.get("model_cache_integrity"))


def _gap_model_cache_integrity(recipe: dict, facts: dict, state: str) -> Gap:
    s = _serving(recipe)
    repo = str(s.get("model_repo", "")).strip()
    rev = str(s.get("model_revision", "")).strip()
    probe = facts.get("model_cache_integrity") or {}
    path = probe.get("server_path") or (
        server_snapshot_dir("<subpath>", repo, rev) if repo else "<subpath>/hub/models--…/snapshots/" + rev[:12]
    )
    if state == AVAILABLE:
        _idx = " + index.json" if probe.get("index_json") else ""
        return Gap(
            "model-cache-integrity",
            "PASS",
            state,
            f"model cache: server snapshot COMPLETE — {repo}@{rev[:12]} has config.json{_idx} "
            f"+ {int(probe.get('shard_count', 0) or 0)} resolved shard(s) at {path}",
        )
    if state == UNKNOWN:
        return Gap(
            "model-cache-integrity",
            "WARN",
            state,
            f"could not verify the server's snapshot path for {repo}@{rev[:12]} "
            f"({probe.get('probe_error') or 'cache probe did not run'}) — proceeding safe-degrade; "
            "the server downloads (slow) or, under HF_HUB_OFFLINE, hard-fails with "
            "LocalEntryNotFoundError. Pre-stage the weights to be sure.",
            action=None,
            fix=f"llmb-k8s install --cluster <c>   # stage {repo}@{rev[:12]} to MODEL_CACHE_SUBPATH",
        )
    # ABSENT → the snapshot is missing or incomplete at the EXACT load path → the server WILL crash-loop.
    missing: list[str] = []
    if probe.get("exists"):
        if not probe.get("config_json"):
            missing.append("config.json")
        # NOTE: a missing model.safetensors.index.json is NOT reported here — single-file models legitimately
        # have none, and a genuinely-incomplete sharded download surfaces via the resolved-shard checks below.
        req = int(probe.get("required_shards", 0) or 0)
        ms = int(probe.get("missing_shards", 0) or 0)
        if req > 0 and ms > 0:  # Gap 1: name the specific index-referenced shards that dangle
            missing.append(f"{ms} of {req} index-referenced *.safetensors shard(s)")
        elif int(probe.get("shard_count", 0) or 0) <= 0:
            missing.append("*.safetensors shard(s)")
        # Gap 3: refs/main is orthogonal to the pinned-snapshot load path — surface it as a NOTE, never a
        # blocker (a GC'd refs/main does not stop the server loading snapshots/<model_revision>).
        note = ""
        if not probe.get("refs_consistent", True):
            note = " [note: refs/main names an absent snapshot — informational only, not the blocker here]"
        why = f"snapshot dir EXISTS but is INCOMPLETE (missing: {', '.join(missing) or 'unknown'})" + note
    else:
        why = "snapshot dir is ABSENT at the server's resolution path"
    return Gap(
        "model-cache-integrity",
        "FAIL",
        state,
        f"model cache CORRUPT/INCOMPLETE for {repo}@{rev[:12]} — {why} at {path}. The vLLM server resolves "
        "this exact path; a ref or snapshot elsewhere in the cache does not satisfy the loader.",
        action="install-revision",
        fix=f"re-stage the revision to the cache subpath: llmb-k8s install --cluster <c>   # {repo}@{rev[:12]}",
    )


# ── entry 7: image-pull-access ───────────────────────────────────────────────
# Verify that the configured pull secret can access every pinned recipe image.
# Network or parsing failures remain UNKNOWN; confirmed authorization failures block launch.
_PULL_PASS = "pass"
_PULL_FORBIDDEN = "forbidden"
_PULL_UNKNOWN = "unknown"
# Accept both OCI + docker manifest/index media types so a HEAD resolves whatever the registry serves.
_MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json,"
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)


def _normalize_registry_host(host: str) -> str:
    """PURE. A dockerconfig key or image-ref host → a bare registry host. Strips scheme/path and folds the
    Docker Hub aliases to the canonical 'docker.io' so a cred keyed on index.docker.io matches a docker.io
    image (and vice-versa)."""
    h = re.sub(r"^https?://", "", str(host or "").strip()).split("/")[0]
    if h in ("index.docker.io", "registry-1.docker.io", "docker.io"):
        return "docker.io"
    return h


def parse_dockerconfigjson(raw) -> Optional[dict]:
    """PURE. Parse a k8s `.dockerconfigjson` (raw JSON str/bytes/dict) → {registry_host: (username, password)}.
    Handles BOTH the explicit username/password shape and the base64 `auth` (user:pass) shape. Returns None
    when the payload is unparseable / has no `auths` map → the gate safe-degrades to UNKNOWN (never a false
    FAIL on a secret we couldn't read)."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode()
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, dict):
        return None
    auths = raw.get("auths")
    if not isinstance(auths, dict):
        return None
    out: dict = {}
    for host, entry in auths.items():
        if not isinstance(entry, dict):
            continue
        user = entry.get("username")
        pw = entry.get("password")
        if (user is None or pw is None) and entry.get("auth"):
            try:
                import base64 as _b64

                dec = _b64.b64decode(str(entry["auth"])).decode()
            except Exception:
                dec = ""
            if ":" in dec:
                user, pw = dec.split(":", 1)
        if user is not None and pw is not None:
            out[_normalize_registry_host(host)] = (str(user), str(pw))
    return out


def registry_cred_for(auths: Optional[dict], registry: str) -> Optional[tuple]:
    """PURE. The (username, password) the secret carries for `registry`, or None (→ unauthenticated probe:
    a public repo still HEADs 200; a private one 401/403s, which IS the gap we want to catch).
    """
    if not isinstance(auths, dict):
        return None
    return auths.get(_normalize_registry_host(registry))


def split_image_ref(ref) -> Optional[tuple[str, str, str]]:
    """PURE. 'nvcr.io/nvidian/my-image@sha256:ab…' → ('nvcr.io', 'nvidian/my-image', 'sha256:ab…').
    Handles digest (@sha256:…) and tag (:v1) refs; a host is the first path segment iff it looks like one
    (has a '.'/':' or is localhost), else the ref is a Docker Hub short name (→ docker.io, library/ padded).
    None for empty input."""
    ref = str(ref or "").strip()
    if not ref:
        return None
    name, reference = ref, ""
    if "@" in name:
        name, reference = name.rsplit("@", 1)
    parts = name.split("/")
    if len(parts) > 1 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        registry, repo = parts[0], "/".join(parts[1:])
    else:
        registry = "docker.io"
        repo = name if "/" in name else "library/" + name
    if not reference:
        seg = repo.split("/")
        if ":" in seg[-1]:  # a trailing :tag on the last repo segment
            seg[-1], reference = seg[-1].rsplit(":", 1)
            repo = "/".join(seg)
    if not reference:
        reference = "latest"
    return registry, repo, reference


def _parse_www_authenticate(header: str) -> dict:
    """PURE. Parse a `WWW-Authenticate: Bearer realm="…",service="…",scope="…"` challenge → {k: v}. Empty
    dict for a non-Bearer / absent header (→ we can't fetch a token → UNKNOWN, safe-degrade).
    """
    h = str(header or "").strip()
    if not h.lower().startswith("bearer"):
        return {}
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r'(\w+)="([^"]*)"', h[len("bearer") :])}


def _basic_auth_header(cred: Optional[tuple]) -> dict:
    if not cred:
        return {}
    import base64 as _b64

    token = _b64.b64encode(f"{cred[0]}:{cred[1]}".encode()).decode()
    return {"Authorization": "Basic " + token}


def _fetch_pull_token(challenge: dict, cred: Optional[tuple], repo: str, http: Callable):
    """IMPURE over `http`. Standard registry-v2 token exchange: GET the challenge realm with the basic-auth
    cred and the pull scope → (token, None). Returns (None, 401|403) when the token endpoint REFUSES the cred
    (the cred lacks the org — a definitive gap) and (None, 'error') on any other/network failure (→ UNKNOWN).
    """
    from urllib.parse import urlencode

    realm = challenge.get("realm")
    if not realm:
        return None, "error"
    params = []
    if challenge.get("service"):
        params.append(("service", challenge["service"]))
    params.append(("scope", challenge.get("scope") or f"repository:{repo}:pull"))
    url = realm + ("&" if "?" in realm else "?") + urlencode(params)
    try:
        status, _hdrs, body = http("GET", url, _basic_auth_header(cred))
    except Exception:
        return None, "error"
    if status in (401, 403):
        return None, status
    if status != 200:
        return None, "error"
    try:
        data = json.loads(body or "{}")
    except Exception:
        return None, "error"
    tok = data.get("token") or data.get("access_token")
    return (tok, None) if tok else (None, "error")


def _probe_one_image(ref: str, auths: Optional[dict], http: Callable) -> dict:
    """IMPURE over `http`. Reproduce the kubelet's pull AUTHORIZATION for one image: HEAD the manifest; on a
    401 challenge fetch a pull token with the secret's cred and retry. Returns a per-image result dict with
    a status ∈ {pass, forbidden, unknown} plus registry/repo for the message. Never raises (→ unknown).
    """
    parsed = split_image_ref(ref)
    if not parsed:
        return {"status": _PULL_UNKNOWN, "reason": "unparseable image ref"}
    registry, repo, reference = parsed
    base = {"registry": registry, "repo": repo}
    cred = registry_cred_for(auths, registry)
    manifest_url = f"https://{registry}/v2/{repo}/manifests/{reference}"
    accept = {"Accept": _MANIFEST_ACCEPT}
    try:
        status, hdrs, _ = http("HEAD", manifest_url, accept)
    except Exception as e:
        return {**base, "status": _PULL_UNKNOWN, "reason": f"registry unreachable: {e}"}
    if status == 200:
        return {
            **base,
            "status": _PULL_PASS,
        }  # public (or already-authorized) → clean PASS
    if status not in (401, 403):
        return {
            **base,
            "status": _PULL_UNKNOWN,
            "reason": f"unexpected manifest status {status}",
        }
    # 401/403 → a token is required. Parse the challenge and exchange the cred for a bearer.
    hdrs = hdrs or {}
    challenge = _parse_www_authenticate(hdrs.get("WWW-Authenticate") or hdrs.get("Www-Authenticate") or "")
    if not challenge:
        # A 403 with no auth challenge: the registry refused outright. With a cred in hand this is a real
        # authorization gap (FORBIDDEN); with none we can't tell how to authenticate → UNKNOWN.
        if status == 403 and cred:
            return {
                **base,
                "status": _PULL_FORBIDDEN,
                "reason": "403 with no auth challenge",
            }
        return {
            **base,
            "status": _PULL_UNKNOWN,
            "reason": f"{status} with no parseable auth challenge",
        }
    token, terr = _fetch_pull_token(challenge, cred, repo, http)
    if token is None:
        if terr in (401, 403):  # token endpoint REFUSED the cred → definitive gap
            return {
                **base,
                "status": _PULL_FORBIDDEN,
                "reason": f"token endpoint {terr} for the cred",
            }
        return {
            **base,
            "status": _PULL_UNKNOWN,
            "reason": "could not obtain a pull token",
        }
    try:
        status2, _h2, _ = http("HEAD", manifest_url, {**accept, "Authorization": f"Bearer {token}"})
    except Exception as e:
        return {**base, "status": _PULL_UNKNOWN, "reason": f"registry unreachable: {e}"}
    if status2 == 200:
        return {**base, "status": _PULL_PASS}
    if status2 in (401, 403):  # token granted but WITHOUT this repo's scope → gap
        return {
            **base,
            "status": _PULL_FORBIDDEN,
            "reason": f"manifest {status2} with bearer token",
        }
    return {
        **base,
        "status": _PULL_UNKNOWN,
        "reason": f"manifest status {status2} with bearer token",
    }


def probe_image_pull_access(images: list[str], dockerconfigjson, http: Callable) -> dict:
    """IMPURE over the injected `http(method, url, headers) -> (status, headers, body)`. For each pinned
    image, reproduce the kubelet's pull authorization using the secret's dockerconfig cred. Returns
    {image_ref: {status, registry, repo, reason?}}. Pure given a pure `http` (the selftest injects a canned
    one; preflight injects `_default_pull_http`). An unparseable dockerconfig → auths=None → every private
    image without a public fallback safe-degrades per-image (401 with no cred → UNKNOWN, not a false FAIL).
    """
    auths = parse_dockerconfigjson(dockerconfigjson) if dockerconfigjson is not None else None
    return {ref: _probe_one_image(ref, auths, http) for ref in images}


def _default_pull_http(method: str, url: str, headers: Optional[dict] = None, timeout: int = 15):
    """IMPURE default HTTP for the LIVE preflight probe. Returns (status, headers, body). An HTTPError
    (401/403/…) is a real STATUS (returned, not raised); a URLError / timeout PROPAGATES so the caller
    degrades that image to UNKNOWN. Read-only: HEAD/GET against the registry auth + manifest endpoints.
    """
    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = "" if method == "HEAD" else r.read().decode(errors="replace")
            return getattr(r, "status", 200), dict(r.headers or {}), body
    except urllib.error.HTTPError as e:
        body = "" if method == "HEAD" else (e.read().decode(errors="replace") if e.fp else "")
        return e.code, dict(e.headers or {}), body


def image_pull_access_verdict(results: Optional[dict]) -> str:
    """PURE. Aggregate the per-image probe results → a capability state. Any FORBIDDEN → ABSENT (block: the
    kubelet will 403 → ImagePullBackOff). All PASS → AVAILABLE. Otherwise (some UNKNOWN, or nothing probed)
    → UNKNOWN (safe-degrade). None (probe never ran) → UNKNOWN."""
    if not isinstance(results, dict):
        return UNKNOWN
    if any((r or {}).get("status") == _PULL_FORBIDDEN for r in results.values()):
        return ABSENT
    if results and all((r or {}).get("status") == _PULL_PASS for r in results.values()):
        return AVAILABLE
    return UNKNOWN


def _req_image_pull_access(recipe: dict) -> bool:
    """Required for ANY recipe that pins one or more OCI images (server digest, launcher image) —
    every one is a digest-pinned nvcr.io ref whose private org the cluster's pull secret may or may not
    be able to pull. A recipe pinning no images has nothing to authorize."""
    return bool(pinned_images(recipe))


def _probe_image_pull_access(facts: dict) -> str:
    """PURE over facts['image_pull_access'] = {'secret':.., 'results': {ref:{status,..}}} (populated by
    preflight's live registry-auth probe). Delegates to image_pull_access_verdict; results=None → UNKNOWN.
    """
    f = facts.get("image_pull_access")
    if not isinstance(f, dict):
        return UNKNOWN
    return image_pull_access_verdict(f.get("results"))


def _gap_image_pull_access(recipe: dict, facts: dict, state: str) -> Gap:
    f = facts.get("image_pull_access") if isinstance(facts.get("image_pull_access"), dict) else {}
    secret = str(f.get("secret") or "").strip() or "<IMAGE_PULL_SECRET>"
    results = f.get("results")
    if state == ABSENT:
        # At least one image is 403-forbidden for this secret's credential — the crash-killer.
        bad = [(ref, r) for ref, r in (results or {}).items() if (r or {}).get("status") == _PULL_FORBIDDEN]
        ref, r = bad[0]
        repo = (r or {}).get("repo") or ref
        registry = (r or {}).get("registry") or ref.split("/")[0]
        short = ref.split("@")[0].split("/")[-1] or ref
        more = f" (+{len(bad) - 1} more)" if len(bad) > 1 else ""
        return Gap(
            "image-pull-access",
            "FAIL",
            state,
            f"IMAGE PULL FORBIDDEN — the pull secret '{secret}' authenticates to {registry} but its "
            f"credential CANNOT pull the private repo {repo} (403 on {short}{more}). The kubelet gets the "
            "same 403 → the pod sticks in ImagePullBackOff holding its GPUs for the whole pull timeout. "
            "This is a per-cluster CREDENTIAL gap, NOT a recipe bug (a different, accessible org pulls fine).",
            action="grant-registry-access",
            fix=f"the NGC credential in '{secret}' lacks pull access to {repo} — grant that org to the "
            f"credential (e.g. mirror the pull secret from a cluster that CAN pull it): "
            f"kubectl -n <ns> get secret {secret} -o yaml   # diff against a working cluster",
        )
    if state == UNKNOWN:
        detail = ""
        if isinstance(results, dict) and results:
            un = [
                ref.split("@")[0].split("/")[-1] for ref, r in results.items() if (r or {}).get("status") != _PULL_PASS
            ]
            if un:
                detail = f" (unverified: {', '.join(un)})"
        return Gap(
            "image-pull-access",
            "WARN",
            state,
            f"could not verify the pull secret '{secret}' can pull every pinned image{detail} — "
            "registry unreachable / secret unreadable / no auth challenge. Proceeding safe-degrade; "
            "if a pod sticks in ImagePullBackOff, check that the secret's NGC credential has the "
            "image's org.",
            action=None,
            fix=None,
        )
    n = len(results or {})
    return Gap(
        "image-pull-access",
        "PASS",
        state,
        f"pull access: secret '{secret}' can authorize all {n} pinned image(s) "
        "(registry token + manifest HEAD returned 200)",
    )


# ── the registry ────────────────────────────────────────────────────────────────
REGISTRY: tuple[Capability, ...] = (
    Capability(
        id="nvlink-imex",
        title="NVLink multicast (IMEX)",
        tier=2,
        ownership="self-serve",
        flags=("NVLINK_MULTICAST_IMEX",),
        requires=_req_nvlink_imex,
        probe=_probe_nvlink_imex,
        gap=_gap_nvlink_imex,
        provision="llmb-k8s profile provision-imex --cluster <c>",  # Tier-2 verb (impl below)
        provision_fn=plan_nvlink_imex,  # Tier-2 decision: provision vs degrade
    ),
    Capability(
        id="no-internet-ips",
        title="no-internet DNS / kube-API IPs",
        tier=1,
        ownership="self-serve",
        flags=("NO_INTERNET_DNS_IP", "NO_INTERNET_KUBE_API_IP"),
        requires=_req_no_internet_ips,
        probe=_probe_no_internet_ips,
        gap=_gap_no_internet_ips,
    ),
    Capability(
        id="model-revision-cached",
        title="model revision cached",
        tier=1,
        ownership="self-serve",
        flags=("MODEL_CACHE_REVISIONS",),
        requires=_req_model_revision,
        probe=_probe_model_revision,
        gap=_gap_model_revision,
        provision="llmb-k8s install --cluster <c>",  # Tier-2 seam (auto-install the rev)
    ),
    Capability(
        id="image-arch",
        title="image arch / manifest list",
        tier=1,
        ownership="mixed",
        flags=("ARCH",),
        requires=_req_image_arch,
        probe=_probe_image_arch,
        gap=_gap_image_arch,
    ),
    Capability(
        id="nvlink-p2p",
        title="NVLink P2P fabric health",
        tier=1,
        ownership="needs-cluster-admin",
        flags=("NVLINK_P2P",),
        requires=_req_nvlink_p2p,
        probe=_probe_nvlink_p2p,
        gap=_gap_nvlink_p2p,
        # DETECT-ONLY: a dead NVLink fabric is a cluster-admin / nv-fabricmanager fix, not provisionable
        # by the harness (unlike nvlink-imex). No provision_fn.
    ),
    Capability(
        id="model-cache-integrity",
        title="model cache snapshot integrity",
        tier=1,
        ownership="self-serve",
        flags=("MODEL_CACHE_SUBPATH", "MODEL_CACHE_PVC"),
        requires=_req_model_cache_integrity,
        probe=_probe_model_cache_integrity,
        gap=_gap_model_cache_integrity,
        provision="llmb-k8s install --cluster <c>",  # re-stage/complete the exact snapshot at the subpath
    ),
    Capability(
        id="image-pull-access",
        title="image pull-secret can pull the recipe's images",
        tier=1,
        ownership="mixed",
        flags=("IMAGE_PULL_SECRET",),
        requires=_req_image_pull_access,
        probe=_probe_image_pull_access,
        gap=_gap_image_pull_access,
        # DETECT-ONLY here: the fix (grant the org / mirror a working cluster's cred) is an NGC/cluster-admin
        # action, so no Tier-2 provision_fn — but the FAIL blocks the launch pre-GPU.
    ),
)


# ── the runner (recipe-scoping happens HERE) ────────────────────────────────────
def required_capabilities(recipe: dict, registry=REGISTRY) -> list[Capability]:
    """The entries THIS recipe needs — the left side of `required ∩ missing`. Pure."""
    return [e for e in registry if e.requires(recipe)]


def evaluate(recipe: dict, facts: dict, registry=REGISTRY) -> list[Gap]:
    """Recipe-scoped evaluation. For each entry the recipe REQUIRES, probe the facts and return its Gap.
    Entries the recipe doesn't require are silent no-ops (never probed, never flagged). Pure over
    (recipe, facts)."""
    out: list[Gap] = []
    for e in required_capabilities(recipe, registry):
        state = e.probe(facts)
        out.append(e.gap(recipe, facts, state))
    return out


def gaps_only(recipe: dict, facts: dict, registry=REGISTRY) -> list[Gap]:
    """Just the unsatisfied intersection: entries that are required AND not PASS. Pure."""
    return [g for g in evaluate(recipe, facts, registry) if g.level != "PASS"]


# ── impure edge: gather live cluster facts (Layer 1) ────────────────────────────
# Injected a `krun(args)->(rc, stdout, stderr)` callable so it is testable with canned kubectl JSON.
def gather_facts(
    profile: dict,
    krun: Callable,
    recipe: Optional[dict] = None,
    want_repo: str = "",
    inspect_image: Optional[Callable] = None,
) -> dict:
    """Best-effort, non-fatal live probe of the recipe-agnostic cluster facts (§2.1 Layer 1). Any probe
    that errors leaves its fact UNKNOWN → the entries safe-degrade. Never mutates the cluster.

    Returns a facts dict consumable by every entry's PURE `probe`. `recipe` is optional and only used to
    scope which (potentially costly) probes to run — passing None gathers all cheap cluster facts.
    `inspect_image(ref)->Optional[list[str]]` is injected so the (image-arch) registry-inspect probe is
    testable with a canned inspector; the live default shells out to skopeo/docker (safe-degrade → None).
    """
    facts: dict = {}
    facts["nvlink_imex"] = _gather_nvlink_imex(krun, profile)
    facts["nvlink_p2p"] = _gather_nvlink_p2p(profile)
    facts["no_internet"] = _gather_no_internet_ips(krun)
    facts["node_archs"] = _gather_node_archs(profile, krun)
    # image-arch (entry 4): FEED facts["images"] = {ref: [arch,...]} from each pinned image's registry
    # manifest, so _gap_image_arch can flag a digest that lacks the target node arch. Recipe-scoped +
    # safe-degrade: only when a recipe is in hand, and an un-inspectable image is omitted (never a false FAIL).
    if recipe is not None:
        inspect = inspect_image or (lambda ref: _default_inspect_image(ref, profile, krun))
        facts["images"] = _gather_image_archs(recipe, profile, inspect)
    # model-cache-integrity (entry 6) mounts a PVC — heavier + recipe-specific, so preflight runs it and
    # injects facts["model_cache_integrity"]; MODEL_CACHE_REVISIONS may be supplied by the profile.
    return facts


def _gather_nvlink_imex(krun: Callable, profile: Optional[dict] = None) -> dict:
    """CRD presence + self-serve RBAC + whether the llmb ComputeDomain is already wired. Read-only; don't
    hard-code the served version (design Risk: DRA version skew) — presence of the CRD name is the fact.

    Facts:
      crd_present         — the ComputeDomain CRD exists (the IMEX-channel provisioner is installed).
      rbac_can_create     — THIS namespace can `create computedomains` (self-serve; drives Tier-2 vs degrade).
      channel_provisioned — the shared `llmb-imex` ComputeDomain already exists in the ns (Tier-2 done), OR
                            the profile already recorded NVLINK_MULTICAST_IMEX=provisioned. So a re-probe
                            after provision-imex keeps FLASHINFER ON instead of re-stripping it.
    """
    profile = profile or {}
    ns = str(profile.get("NAMESPACE") or "").strip()
    rc, out, _ = krun(["get", "crd", IMEX_CD_CRD, "-o", "name"])
    crd_present = rc == 0 and "computedomain" in (out or "").lower()
    rbac_can_create = False
    if crd_present:
        # RBAC must be evaluated against the TARGET namespace: a per-namespace RoleBinding grants `create`
        # only there, so probing the caller's default ns would misjudge it. Mirror the CD-exists check below.
        ns_flag = ["-n", ns] if ns else []
        rc2, out2, _ = krun(["auth", "can-i", "create", IMEX_CD_CRD, *ns_flag])
        rbac_can_create = rc2 == 0 and "yes" in (out2 or "").lower()
    channel_provisioned = str(profile.get("NVLINK_MULTICAST_IMEX") or "").strip() == PROVISIONED
    if crd_present and ns and not channel_provisioned:
        rc3, out3, _ = krun(["-n", ns, "get", IMEX_CD_KIND.lower(), IMEX_CD_NAME, "-o", "name"])
        if rc3 == 0 and IMEX_CD_NAME in (out3 or ""):
            channel_provisioned = True
    return {
        "crd_present": crd_present,
        "rbac_can_create": rbac_can_create,
        "channel_provisioned": channel_provisioned,
    }


def _gather_nvlink_p2p(profile: Optional[dict] = None) -> dict:
    """Read the NVLink P2P fabric-health fact from the cluster profile. The heavy GPU-scheduled
    nvidia-smi probe lives in probe_fabric.py (run at `profile validate`), which records
    NVLINK_P2P=healthy|disabled|unknown (+ NVLINK_P2P_ROUTE_UNHEALTHY) into the profile — the same
    discipline as the RDMA_* facts. Here we cheaply read it back; a profile without the fact → unknown
    (safe-degrade: the gate never blocks a run it couldn't measure)."""
    profile = profile or {}
    st = str(profile.get("NVLINK_P2P") or "").strip().lower()
    if st not in ("healthy", "disabled", "unknown"):
        st = "unknown"
    return {
        "state": st,
        "route_unhealthy": str(profile.get("NVLINK_P2P_ROUTE_UNHEALTHY") or "").strip().lower() == "true",
    }


def _gather_no_internet_ips(krun: Callable) -> dict:
    """kube-dns ClusterIP + the REAL kubernetes endpoint IP (NOT the blocked .1 Service ClusterIP)."""
    rc1, dns, _ = krun(
        [
            "-n",
            "kube-system",
            "get",
            "svc",
            "kube-dns",
            "-o",
            "jsonpath={.spec.clusterIP}",
        ]
    )
    rc2, api, _ = krun(
        [
            "get",
            "endpoints",
            "kubernetes",
            "-o",
            "jsonpath={.subsets[0].addresses[0].ip}",
        ]
    )
    return {
        "dns_ip": dns.strip() if rc1 == 0 else "",
        "kube_api_ip": api.strip() if rc2 == 0 else "",
    }


def _gather_node_archs(profile: dict, krun: Callable) -> list[str]:
    """Distinct kubernetes.io/arch across the profile's GPU nodes (cheap; reused by image-arch)."""
    gpu_product = (profile.get("GPU_PRODUCT") or "").strip()
    sel = ["-l", f"nvidia.com/gpu.product={gpu_product}"] if gpu_product else []
    rc, out, _ = krun(
        [
            "get",
            "nodes",
            *sel,
            "-o",
            "jsonpath={.items[*].metadata.labels.kubernetes\\.io/arch}",
        ]
    )
    if rc != 0 or not out:
        return []
    return sorted({a for a in out.split() if a})


# ── profile write (Layer-1 auto-set of cluster facts) ───────────────────────────
_BLOCK_HEADER = "# --- capabilities (auto-discovered by probe-capabilities) ---"


def facts_to_profile_flags(facts: dict) -> dict[str, str]:
    """The subset of Layer-1 facts that are AUTO-SET into the profile (Tier-1 'detect + set the flag').
    Only writes a flag when the fact resolved — never clobbers a good value with an empty probe.
    """
    flags: dict[str, str] = {}
    ni = facts.get("no_internet") or {}
    if str(ni.get("dns_ip") or "").strip():
        flags["NO_INTERNET_DNS_IP"] = ni["dns_ip"].strip()
    if str(ni.get("kube_api_ip") or "").strip():
        flags["NO_INTERNET_KUBE_API_IP"] = ni["kube_api_ip"].strip()
    imex = facts.get("nvlink_imex") or {}
    if imex.get("channel_provisioned"):
        flags["NVLINK_MULTICAST_IMEX"] = PROVISIONED
    elif imex.get("crd_present"):
        flags["NVLINK_MULTICAST_IMEX"] = AVAILABLE
    elif "crd_present" in imex:
        flags["NVLINK_MULTICAST_IMEX"] = ABSENT
    return flags


def render_profile_block(flags: dict[str, str]) -> str:
    """The idempotent capabilities block appended to a cluster profile (mirrors probe_fabric.patch_profile)."""
    lines = [_BLOCK_HEADER]
    for k, v in flags.items():
        lines.append(f'{k}="{v}"')
    return "\n".join(lines)


def patch_profile_text(text: str, flags: dict[str, str]) -> str:
    """PURE. Replace any prior capabilities block in `text` with a fresh one (idempotent re-write)."""
    text = re.sub(r"\n?" + re.escape(_BLOCK_HEADER) + r".*", "", text, flags=re.DOTALL)
    text = text.rstrip("\n") + "\n"
    if not flags:
        return text
    return text + "\n" + render_profile_block(flags) + "\n"


# ── structural self-check (every entry declares the full contract) ──────────────
def registry_is_wellformed(registry=REGISTRY) -> list[str]:
    """Return a list of contract violations (empty = well-formed). A half-defined entry fails CI."""
    problems: list[str] = []
    seen = set()
    for e in registry:
        if e.id in seen:
            problems.append(f"duplicate id: {e.id}")
        seen.add(e.id)
        for fn in ("requires", "probe", "gap"):
            if not callable(getattr(e, fn, None)):
                problems.append(f"{e.id}: {fn} is not callable")
        if e.tier not in (1, 2):
            problems.append(f"{e.id}: tier must be 1 or 2")
        if not e.flags:
            problems.append(f"{e.id}: declares no flags")
        if e.ownership not in ("self-serve", "needs-cluster-admin", "mixed"):
            problems.append(f"{e.id}: bad ownership '{e.ownership}'")
    return problems


if __name__ == "__main__":
    import sys

    probs = registry_is_wellformed()
    if probs:
        print("capability registry MALFORMED:")
        for p in probs:
            print("  -", p)
        sys.exit(1)
    print(f"capability registry OK — {len(REGISTRY)} entries: {', '.join(e.id for e in REGISTRY)}")

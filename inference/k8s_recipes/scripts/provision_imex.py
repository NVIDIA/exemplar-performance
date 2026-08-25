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

"""provision_imex.py <cluster-profile> [--recipe <cell>] [--dry-run] [--deprovision]

Tier-2 provision path for the flagship `nvlink-imex` capability.

Where a cluster can self-serve a ComputeDomain (GB300 today: RBAC confirmed), this RECOVERS the FlashInfer
`allreduce_rms` NVLink multicast fusion instead of the Tier-1 degrade (which strips forced FLASHINFER and
eats a throughput hit). It:

  1. Probes the cluster: ComputeDomain CRD present? can THIS namespace `create computedomains`? already wired?
  2. Decides the tier (capability_registry.plan_nvlink_imex):
       • self-serve RBAC present → PROVISION: `kubectl apply` the shared `llmb-imex` ComputeDomain (v1beta1),
         then write NVLINK_MULTICAST_IMEX=provisioned + IMEX_CLAIM_TEMPLATE=llmb-imex-channel to the profile.
         The per-pod channel claim is injected at APPLY time by scripts/merge_imex_claim.py (deploy.sh
         pipeline), so no committed rendered/ manifest — and no recipe_hash — moves.
       • RBAC/CRD absent (e.g. GB200 today) → DEGRADE: print ONE plain-English line explaining the throughput
         tradeoff + exactly how to earn the capability. Nothing is applied; FLASHINFER stays stripped (Tier 1).

Recipe-scoping: with --recipe <cell>, provisions ONLY if that recipe engages the fusion (forced FLASHINFER,
tp>1, GB-class); a recipe that doesn't force FLASHINFER provisions nothing. Without --recipe, the operator is
explicitly enabling the cluster capability (the shared ComputeDomain is one-per-namespace, recipe-agnostic).

--dry-run prints the plan + the ComputeDomain manifest and applies nothing. --deprovision deletes the CD and
clears the profile flags (design §4.7; the CD is shared, so deprovision is an explicit opt-in).

Exit 0 on success or graceful degrade; 1 on a hard apply failure; 2 on usage error.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capability_registry as cap  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("provision_imex: requires pyyaml (pip install pyyaml)")

_KUBE_CONTEXT = ""


def _kubectl(args, timeout=30, stdin=None):
    ctx = ["--context", _KUBE_CONTEXT] if _KUBE_CONTEXT else []
    try:
        p = subprocess.run(
            ["kubectl", *ctx, "--request-timeout=25s", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            input=stdin,
        )
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def _krun(args, timeout=30):
    """krun signature the registry expects: (args) -> (rc, stdout, stderr)."""
    return _kubectl(args, timeout=timeout)


def parse_env(path):
    out = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'"):
            v = v[1:-1] if v[-1:] == v[:1] else v[1:]
        out[k] = v
    return out


def _synthetic_fusion_recipe(gpu_product: str) -> dict:
    """A recipe-agnostic stand-in for the operator's explicit `provision-imex` (no --recipe): represents
    'I want the GB-class NVLink fusion capability'. gpu_type='GB' → GB-class; forced FLASHINFER + tp>1.
    """
    return {
        "envelope": {"gpu_type": ("GB" if not gpu_product else gpu_product.upper().replace("NVIDIA-", ""))},
        "serving": {
            "tp": 2,
            "env": [{"name": "VLLM_ATTENTION_BACKEND", "value": "FLASHINFER"}],
        },
    }


_IMEX_HEADER = "# --- nvlink-imex (auto-provisioned by provision-imex) ---"


def _patch_imex_block(text: str, flags: dict) -> str:
    """Idempotent profile block for the IMEX provision flags (mirrors probe_fabric.patch_profile)."""
    text = re.sub(r"\n?" + re.escape(_IMEX_HEADER) + r".*", "", text, flags=re.DOTALL)
    text = text.rstrip("\n") + "\n"
    if not flags:
        return text
    lines = ["\n" + _IMEX_HEADER] + [f'{k}="{v}"' for k, v in flags.items()]
    return text + "\n".join(lines) + "\n"


def main() -> int:
    argv = sys.argv[1:]
    dry_run = "--dry-run" in argv
    deprovision = "--deprovision" in argv
    recipe_cell = None
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] == "--recipe" and i + 1 < len(argv):
            recipe_cell = argv[i + 1]
            i += 2
            continue
        if not argv[i].startswith("--"):
            pos.append(argv[i])
        i += 1
    if len(pos) != 1:
        sys.exit(__doc__)

    profile_name = pos[0]
    prof_path = ROOT / "cluster-profiles" / f"{profile_name}.env"
    if not prof_path.exists():
        sys.exit(f"provision-imex: no profile at {prof_path}")
    prof = parse_env(prof_path)

    global _KUBE_CONTEXT
    _KUBE_CONTEXT = (prof.get("KUBE_CONTEXT") or "").strip()
    ns = (prof.get("NAMESPACE") or "").strip()
    gpu_product = (prof.get("GPU_PRODUCT") or "").strip()
    if not ns:
        sys.exit(f"provision-imex: NAMESPACE not set in {profile_name}.env")
    ctx_label = _KUBE_CONTEXT or "ambient"
    print(f"provision-imex: {profile_name}  (namespace={ns}, context={ctx_label})")

    # ── deprovision: delete the shared ComputeDomain + clear the flags ──────────
    if deprovision:
        if dry_run:
            print(f"  [dry-run] would: kubectl -n {ns} delete computedomain {cap.IMEX_CD_NAME}")
            return 0
        rc, out, err = _kubectl(
            [
                "-n",
                ns,
                "delete",
                cap.IMEX_CD_KIND.lower(),
                cap.IMEX_CD_NAME,
                "--ignore-not-found",
            ]
        )
        print(f"  {out.strip() or 'ComputeDomain deleted (or absent)'}")
        prof_path.write_text(_patch_imex_block(prof_path.read_text(), {}))
        print(f"  cleared IMEX flags from {profile_name}.env")
        return 0 if rc == 0 else 1

    # ── probe live facts (recipe-agnostic Layer 1) ──────────────────────────────
    facts = cap.gather_facts(
        {
            "GPU_PRODUCT": gpu_product,
            "NAMESPACE": ns,
            "NVLINK_MULTICAST_IMEX": prof.get("NVLINK_MULTICAST_IMEX", ""),
        },
        _krun,
    )
    f = facts.get("nvlink_imex") or {}
    print(
        f"  probe: crd_present={f.get('crd_present')}  rbac_can_create={f.get('rbac_can_create')}  "
        f"channel_provisioned={f.get('channel_provisioned')}"
    )

    # ── recipe (scoping) ────────────────────────────────────────────────────────
    if recipe_cell:
        rc_path = Path(recipe_cell)
        if rc_path.is_dir():
            rc_path = rc_path / "recipe.yaml"
        if not rc_path.exists():
            sys.exit(f"provision-imex: no recipe.yaml at {recipe_cell}")
        recipe = yaml.safe_load(rc_path.read_text()) or {}
    else:
        recipe = _synthetic_fusion_recipe(gpu_product)

    plan = cap.plan_nvlink_imex(recipe, facts, ns)
    print(f"\n  plan: tier={plan.tier} action={plan.action} keep_flashinfer={plan.keep_flashinfer}")
    print(f"  → {plan.message}\n")

    if plan.action == "noop":
        return 0

    if plan.action in ("degrade",):
        # Graceful tier-down: nothing applied, FLASHINFER stays stripped (Tier 1). Record the observed state.
        if not dry_run and plan.profile_flags:
            prof_path.write_text(_patch_imex_block(prof_path.read_text(), plan.profile_flags))
            print(
                f"  recorded NVLINK_MULTICAST_IMEX={plan.profile_flags.get('NVLINK_MULTICAST_IMEX')} "
                f"in {profile_name}.env (Tier-1 degrade — throughput only, never a crash)"
            )
        return 0

    # action ∈ {provision, already-provisioned}
    manifests = list(plan.manifests)
    if manifests:
        rendered = yaml.dump_all(manifests, default_flow_style=False, sort_keys=False)
        print("  ComputeDomain manifest:")
        print("    " + rendered.replace("\n", "\n    ").rstrip())
        if dry_run:
            print(f"\n  [dry-run] would: kubectl -n {ns} apply -f - (the manifest above)")
            print(f"  [dry-run] would write profile flags: {plan.profile_flags}")
            return 0
        rc, out, err = _kubectl(["-n", ns, "apply", "-f", "-"], stdin=rendered)
        if rc != 0:
            print(f"  ❌ apply failed: {err.strip() or out.strip()}", file=sys.stderr)
            return 1
        print(f"  ✅ applied: {out.strip()}")
    elif dry_run:
        print("  [dry-run] already provisioned — nothing to apply")
        return 0

    # persist the profile flags so deploy.sh/merge_imex_claim.py wires the claim + preflight keeps FLASHINFER.
    prof_path.write_text(_patch_imex_block(prof_path.read_text(), plan.profile_flags))
    print(f"  ✅ wrote {profile_name}.env: {', '.join(f'{k}={v}' for k, v in plan.profile_flags.items())}")
    print("  next: re-deploy the serving cell — merge_imex_claim.py injects the channel claim at apply time,")
    print("        FlashInfer allreduce_rms fusion runs (VLLM_ATTENTION_BACKEND=FLASHINFER kept ON).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

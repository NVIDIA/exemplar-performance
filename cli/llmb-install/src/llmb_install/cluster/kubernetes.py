# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes preflight utilities for the 'kubernetes' install method.

Best-effort checks that surface configuration problems early (before workloads
are registered) without hard-failing the install: a cluster may be reachable
only from a different host, or the CRD may be installed later.
"""

import base64
import json
import os
import shutil
import subprocess
from typing import List, Optional

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)

# The CRD that llmb-run's WorkloadRun launcher applies against.
WORKLOADRUN_CRD = "workloadruns.nvcre.nvidia.com"

_KUBECTL_TIMEOUT_SECONDS = 15


def resolve_kubeconfig(kubeconfig: Optional[str] = None) -> Optional[str]:
    """Resolve a kubeconfig path from an explicit value or standard locations.

    Precedence: explicit arg -> $KUBECONFIG (first existing entry) -> ~/.kube/config.
    Returns the resolved path, or None if none exists.
    """
    if kubeconfig:
        expanded = os.path.expanduser(kubeconfig)
        return expanded if os.path.exists(expanded) else None

    env_value = os.environ.get('KUBECONFIG')
    if env_value:
        for candidate in env_value.split(os.pathsep):
            expanded = os.path.expanduser(candidate.strip())
            if expanded and os.path.exists(expanded):
                return expanded

    default = os.path.expanduser('~/.kube/config')
    return default if os.path.exists(default) else None


def _kubectl(args: List[str], kubeconfig: Optional[str]) -> subprocess.CompletedProcess:
    cmd = ["kubectl"]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]
    cmd += args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=_KUBECTL_TIMEOUT_SECONDS)


def check_kubernetes_prerequisites(kubeconfig: Optional[str] = None) -> bool:
    """Warn about missing Kubernetes prerequisites for a WorkloadRun install.

    Checks, in order: kubectl on PATH, a resolvable kubeconfig, cluster reachable,
    and the WorkloadRun CRD present. Never raises — prints actionable warnings and
    returns True only when every check passes.
    """
    print("\nChecking Kubernetes prerequisites.")
    print("--------------------------------")

    if shutil.which("kubectl") is None:
        print("  ⚠️  'kubectl' not found on PATH. Install it before running workloads with `llmb-run`.")
        return False
    print("  ✓ kubectl found")

    resolved = resolve_kubeconfig(kubeconfig)
    if not resolved:
        target = kubeconfig or "$KUBECONFIG / ~/.kube/config"
        print(f"  ⚠️  No kubeconfig found ({target}). Configure cluster access before running workloads.")
        return False
    print(f"  ✓ kubeconfig: {resolved}")

    try:
        crd = _kubectl(["get", "crd", WORKLOADRUN_CRD], resolved)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  ⚠️  Could not query the cluster ({e}). Skipping CRD check.")
        return False

    if crd.returncode != 0:
        stderr = (crd.stderr or "").strip()
        if "not found" in stderr.lower() or "NotFound" in stderr:
            print(f"  ⚠️  CRD '{WORKLOADRUN_CRD}' is not installed on the cluster.")
            print("      Install the nvcre operator/CRD before submitting WorkloadRuns.")
        else:
            print(f"  ⚠️  Cluster not reachable or access denied: {stderr or 'unknown error'}")
        return False

    print(f"  ✓ CRD '{WORKLOADRUN_CRD}' present")
    return True


def _apply_secret(manifest: dict, kubeconfig: Optional[str]) -> subprocess.CompletedProcess:
    """Apply a Secret manifest through ``kubectl apply -f -``.

    The manifest travels over stdin, never in argv, where any local user could read the
    secret from the process list. ``apply`` also replaces an already-existing secret.
    """
    apply_cmd = ["kubectl", "apply", "-f", "-"]
    if kubeconfig:
        apply_cmd += ["--kubeconfig", kubeconfig]
    return subprocess.run(
        apply_cmd,
        input=json.dumps(manifest),
        capture_output=True,
        text=True,
        timeout=_KUBECTL_TIMEOUT_SECONDS,
    )


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def create_image_pull_secrets(
    secret_names: List[str],
    ngc_api_key: str,
    namespace: str,
    kubeconfig: Optional[str] = None,
) -> None:
    """Create NGC docker-registry pull secrets in the given namespace.

    Each secret is a ``kubernetes.io/dockerconfigjson`` secret for ``nvcr.io`` using
    ``$oauthtoken`` as the username and the NGC API key as the password. Existing secrets
    with the same name are replaced (``kubectl apply``). Every secret is attempted; a
    RuntimeError then names the ones that could not be created.
    """
    print("\nCreating image pull secrets.")
    print("----------------------------")

    if not ngc_api_key:
        print("  ⚠️  No NGC API key provided; skipping secret creation.")
        return

    username = "$oauthtoken"
    docker_config = {
        "auths": {
            "nvcr.io": {
                "username": username,
                "password": ngc_api_key,
                "auth": _b64(f"{username}:{ngc_api_key}"),
            }
        }
    }

    failed: List[str] = []
    for name in secret_names:
        manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name, "namespace": namespace},
            "type": "kubernetes.io/dockerconfigjson",
            "data": {".dockerconfigjson": _b64(json.dumps(docker_config))},
        }
        try:
            result = _apply_secret(manifest, kubeconfig)
            if result.returncode == 0:
                print(f"  ✓ Secret '{name}' created/updated in namespace '{namespace}'")
            else:
                print(f"  ⚠️  Failed to apply secret '{name}': {result.stderr.strip()}")
                failed.append(name)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"  ⚠️  Error creating secret '{name}': {e}")
            failed.append(name)

    if failed:
        # Without the secret every later nvcr.io pull fails, so do not let the install report success.
        raise RuntimeError(f"Failed to create image pull secret(s): {', '.join(failed)}")


def create_hf_token_secret(
    secret_name: str,
    hf_token: str,
    namespace: str,
    secret_key: str = "hf-token",
    kubeconfig: Optional[str] = None,
) -> bool:
    """Create a generic Kubernetes Secret holding the HuggingFace token.

    Uses ``kubectl apply`` so an existing secret is updated in place. The default key,
    ``hf-token``, is the one the inference Helm charts read. Returns True on success.
    """
    print("\nCreating HuggingFace token secret.")
    print("-----------------------------------")

    manifest = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "Opaque",
        "data": {secret_key: _b64(hf_token)},
    }
    try:
        result = _apply_secret(manifest, kubeconfig)
        if result.returncode == 0:
            print(f"  ✓ HF token secret '{secret_name}' created/updated in namespace '{namespace}'")
            return True
        print(f"  ⚠️  Failed to apply HF token secret '{secret_name}': {result.stderr.strip()}")
        return False
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  ⚠️  Error creating HF token secret '{secret_name}': {e}")
        return False

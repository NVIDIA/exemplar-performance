# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive prompt for the 'kubernetes' install method settings.

Collects the fields written to cluster_config.yaml's ``kubernetes`` block and
consumed by llmb-run's WorkloadRun launcher. Mirrors ``prompt_slurm_info`` and
returns ``{'kubernetes': {...}}`` (or None if the user cancels).
"""

import re
from typing import Any, Dict, Optional

from llmb_install.ui.interface import UIInterface

# RFC-1123 label: lowercase alphanumerics and '-', must start/end alphanumeric.
_NAMESPACE_RE = re.compile(r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$')
_PVC_NAME_RE = re.compile(r'^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$')
_PVC_NAME_MAX_LEN = 253


def _validate_namespace(value: str) -> Any:
    value = (value or '').strip()
    if not value:
        return "Namespace must not be blank."
    if not _NAMESPACE_RE.match(value):
        return "Namespace must be a valid RFC-1123 label (lowercase alphanumerics and '-')."
    return True


def _validate_optional_pvc_name(value: str) -> Any:
    value = (value or '').strip()
    if not value:
        return True
    if len(value) > _PVC_NAME_MAX_LEN or not _PVC_NAME_RE.match(value):
        return "PVC name must be a valid RFC-1123 subdomain (lowercase alphanumerics, '-' and '.')."
    return True


def prompt_kubernetes_info(
    ui: UIInterface,
    defaults: Optional[Dict[str, Any]] = None,
    express_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """Prompt for Kubernetes settings for a WorkloadRun install.

    Args:
        ui: UI interface for user interaction.
        defaults: Optional prior values ({'namespace', 'kubeconfig', 'image_pull_secrets', 'workdir_pvc'}).
        express_mode: Reserved for parity with other prompts (unused here).

    Returns:
        {'kubernetes': {'namespace', 'kubeconfig'?, 'image_pull_secrets'?, 'workdir_pvc'?}} or None if cancelled.
    """
    defaults = defaults or {}
    ui.print_section("Kubernetes Configuration")
    ui.log("Workloads will run as nvcre WorkloadRun custom resources.")
    ui.log("Provide the cluster settings used when applying them:")

    namespace = ui.prompt_text(
        "Namespace to submit WorkloadRuns into:",
        default=str(defaults.get('namespace') or 'default'),
        validate=_validate_namespace,
    )
    if namespace is None:
        return None
    namespace = namespace.strip()

    kubeconfig = ui.prompt_path(
        "Path to kubeconfig (leave blank to use $KUBECONFIG or ~/.kube/config):",
        default=str(defaults.get('kubeconfig') or ''),
    )
    if kubeconfig is None:
        return None
    kubeconfig = kubeconfig.strip()

    secrets_raw = ui.prompt_text(
        "Image pull secret name(s), comma-separated (leave blank if none):",
        default=",".join(defaults.get('image_pull_secrets') or []),
    )
    if secrets_raw is None:
        return None
    image_pull_secrets = [s.strip() for s in secrets_raw.split(',') if s.strip()]

    ngc_api_key = None
    if image_pull_secrets:
        ui.log(
            "llmb-install can create these secrets in the namespace using your NGC API key.\n"
            "The key is used only to create the Kubernetes secret and is not stored."
        )
        ngc_api_key = ui.prompt_text(
            "NGC API key (leave blank to skip secret creation):",
            default=defaults.get('ngc_api_key') or '',
            password=True,
        )
        if ngc_api_key is None:
            ngc_api_key = ''
        ngc_api_key = ngc_api_key.strip()

    workdir_pvc = ui.prompt_text(
        "PVC name for syncing job code, configs and profiling output (leave blank if none):",
        default=str(defaults.get('workdir_pvc') or ''),
        validate=_validate_optional_pvc_name,
    )
    if workdir_pvc is None:
        return None
    workdir_pvc = workdir_pvc.strip()

    run_nvcrectl_init = ui.prompt_confirm(
        "Run 'nvcrectl setup init' to install/upgrade the nvcre operator? "
        "(say No if it is already installed on the cluster)",
        default=False,
    )
    if run_nvcrectl_init is None:
        return None

    nvcrectl_chart_version = ''
    if run_nvcrectl_init:
        nvcrectl_chart_version = ui.prompt_text(
            "nvcrectl chart version to install (leave blank to let nvcrectl auto-detect):",
            default=str(defaults.get('nvcrectl_chart_version') or ''),
        )
        if nvcrectl_chart_version is None:
            return None
        nvcrectl_chart_version = nvcrectl_chart_version.strip()

    kubernetes: Dict[str, Any] = {'namespace': namespace, 'gang_scheduler_name': 'kai-scheduler'}
    if kubeconfig:
        kubernetes['kubeconfig'] = kubeconfig
    if image_pull_secrets:
        kubernetes['image_pull_secrets'] = image_pull_secrets
    if workdir_pvc:
        kubernetes['workdir_pvc'] = workdir_pvc
    if ngc_api_key:
        kubernetes['ngc_api_key'] = ngc_api_key
    if run_nvcrectl_init:
        kubernetes['run_nvcrectl_init'] = True
    if nvcrectl_chart_version:
        kubernetes['nvcrectl_chart_version'] = nvcrectl_chart_version

    return {'kubernetes': kubernetes}

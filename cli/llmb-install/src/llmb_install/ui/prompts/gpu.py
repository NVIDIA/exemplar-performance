# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""GPU configuration prompts for LLMB installer."""

from typing import Any, Dict, Optional

from llmb_install.cluster import gpu, slurm
from llmb_install.cluster.nvlink import describe_failure, detect_nvl_topology, uses_block_topology
from llmb_install.constants import NVL_DOMAINS_BY_GPU
from llmb_install.ui.interface import UIInterface


def prompt_gpu_type(
    ui: UIInterface, workloads: Dict[str, Dict[str, Any]], default: Optional[str] = None, express_mode: bool = False
) -> str:
    """Prompt the user to select GPU type based on available workloads.

    Args:
        ui: UIInterface implementation for user interaction
        workloads: Dictionary of workload metadata
        default: Default GPU type from system config (if available)
        express_mode: Whether this is being called from express mode (shows default messages)

    Returns:
        str: The selected GPU type (h100, gb200, b200)
    """
    ui.print_section("GPU Type Selection")
    ui.log("Please select the GPU type for your cluster.")
    ui.log("This will determine which workloads are available for installation.")

    # Get available GPU choices using business logic
    try:
        choices = gpu.get_available_gpu_choices(workloads)
    except SystemExit as e:
        ui.log(f"Failed to get GPU choices: {e}")
        raise

    # Use provided default if valid, otherwise use first choice
    choice_values = [choice['value'] for choice in choices]
    if default and default in choice_values:
        selected_default = default
        if express_mode:
            ui.log(f"Using saved default: {default}")
    else:
        selected_default = choices[0]['value'] if choices else None

    gpu_type = ui.prompt_select("Select GPU type:", choices=choices, default=selected_default)

    if gpu_type is None:
        ui.log("\nInstallation cancelled.")
        raise SystemExit(1)

    ui.log(f"Selected GPU type: {gpu_type}")
    return gpu_type


def prompt_node_architecture(
    ui: UIInterface,
    gpu_type: str,
    default: Optional[str] = None,
    express_mode: bool = False,
    gpu_partition: Optional[str] = None,
) -> str:
    """Prompt the user for the node CPU architecture, with auto-selection based on GPU type.

    Args:
        ui: UIInterface implementation for user interaction
        gpu_type: The selected GPU type (h100, gb200, b200)
        default: Default architecture from system config (if available)
        express_mode: Whether this is being called from express mode (shows default messages)
        gpu_partition: SLURM GPU partition to sample for architecture detection

    Returns:
        str: The selected node CPU architecture ('x86_64' or 'aarch64')
    """
    ui.print_section("GPU Node - CPU Architecture")

    # Check if architecture should be auto-selected
    if gpu.should_auto_select_architecture(gpu_type):
        arch = gpu.get_default_architecture(gpu_type)
        arch_name = "ARM-based" if arch == 'aarch64' else "x86-based"
        ui.log(f"{gpu_type.upper()} systems are {arch_name} ({arch}). Auto-selecting {arch} architecture.")
        return arch

    if gpu_partition:
        detection = slurm.detect_partition_architecture(gpu_partition)
        detected_arch = detection.get("architecture")
        if detected_arch:
            ui.log(f"✓ Auto-detected GPU node architecture: {detected_arch}")
            return detected_arch

        reason = detection.get("reason")
        architectures = detection.get("architectures", {})
        if reason == "mixed" and architectures:
            values = sorted(set(architectures.values()))
            ui.log(f"SLURM reported mixed node architectures: {', '.join(values)}.")
        elif reason == "unsupported" and architectures:
            values = sorted(set(architectures.values()))
            ui.log(f"SLURM reported unsupported node architecture: {', '.join(values)}.")
        elif reason in {"no_nodes", "no_arch"}:
            ui.log("Could not auto-detect node architecture from SLURM.")
        elif reason == "error":
            ui.log("Could not auto-detect node architecture from SLURM.")

    ui.log("Please select the CPU architecture of your GPU nodes to ensure correct container image downloads.")
    ui.log("Choosing the wrong architecture (e.g., aarch64 for an x86_64 system) will result in 'Exec format error'.")

    # TODO: we might want to figure out how to make this dynamic in the future.
    if gpu_type == 'h100' or gpu_type == 'b200':
        ui.log(f"\n{gpu_type.upper()} systems are typically x86_64 based.")

    choices = gpu.get_architecture_choices(gpu_type)
    default_arch = gpu.get_default_architecture(gpu_type)

    # Use provided default if valid, otherwise use GPU-based default
    choice_values = [choice['value'] for choice in choices]
    if default and default in choice_values:
        selected_default = default
        if express_mode:
            ui.log(f"Using saved default: {default}")
    else:
        selected_default = default_arch

    architecture = ui.prompt_select(
        "Select the CPU architecture of your GPU nodes:", choices=choices, default=selected_default
    )

    if architecture is None:
        ui.log("\nInstallation cancelled.")
        raise SystemExit(1)

    ui.log(f"Selected node architecture: {architecture}")
    return architecture


def prompt_nvl_domain(
    ui: UIInterface,
    gpu_type: str,
    default: Optional[int] = None,
    query_slurm: bool = True,
) -> tuple[int, Optional[list[int]]]:
    """Select or detect the NVLink domain size for a GPU cluster.

    Args:
        query_slurm: Detect the domain from the Slurm block topology. Pass False for
            Kubernetes installs, where there is no Slurm to query.
    """
    if not uses_block_topology(gpu_type):
        return NVL_DOMAINS_BY_GPU[gpu_type][0], None

    if not query_slurm:
        domain = min(NVL_DOMAINS_BY_GPU[gpu_type])
        ui.print_section("NVLink Domain")
        ui.log(f"Kubernetes install: no Slurm topology to query; using NVL{domain}.")
        return domain, None

    ui.print_section("NVLink Domain")
    detection = detect_nvl_topology(gpu_type)
    block_sizes = detection.block_sizes
    detected = detection.nvl_domain
    domains = NVL_DOMAINS_BY_GPU[gpu_type]
    if detection.reason == 'detected' and detected in domains:
        ui.log(f"✓ Auto-detected NVLink domain: NVL{detected}")
        return detected, block_sizes

    ui.log(describe_failure(detection))
    if detection.reason == 'detected':
        choices = [
            {'name': f'NVL{detected} (detected)', 'value': detected},
            *({'name': f'NVL{domain}', 'value': domain} for domain in domains),
        ]
        choice_values = [choice['value'] for choice in choices]
        selected_default = default if default in choice_values else detected
        choice = ui.prompt_select(
            "Select the NVLink domain size of your GPU nodes:", choices=choices, default=selected_default
        )
        if choice is None:
            ui.log("\nInstallation cancelled.")
            raise SystemExit(1)
        return choice, detection.block_sizes

    expected = min(domains)
    ui.log(f"Only a single-rack NVL{expected} cluster performs correctly without Slurm block topology.")
    if ui.prompt_confirm(f"Continue as NVL{expected}?", default=False):
        return expected, None
    else:
        ui.log("\nInstallation cancelled.")
        raise SystemExit(1)

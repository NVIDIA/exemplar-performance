# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""SLURM configuration prompts for LLMB installer."""

from typing import Dict, Optional

from llmb_install.cluster import slurm
from llmb_install.ui.interface import UIInterface

# Validation error prefix used by cluster.slurm validation functions
INVALID_PREFIX = "invalid:"


def _prompt_gpu_gres(ui: UIInterface, partition_name: str) -> Optional[int]:
    """Interactively ask the user for GPUs per node for partition_name."""
    while True:
        resp = ui.prompt_text(f"Enter GPUs per node for partition '{partition_name}' (leave blank if not applicable):")

        if resp is None:
            ui.log("\nInstallation cancelled.")
            raise SystemExit(1)

        resp = resp.strip()
        if resp == "":
            return None
        if resp.isdigit():
            val = int(resp)
            if 1 <= val <= 8:
                return val
            ui.log("Please enter an integer between 1 and 8.")
        else:
            ui.log("Please enter a valid integer.")


def _print_gres_detection_result(ui: UIInterface, partition: str, is_gpu_partition: bool, partition_info: dict):
    """Print user-friendly GRES detection results."""
    label = "GPU partition" if is_gpu_partition else "CPU partition"
    gpu_count = partition_info["gpu_count"]

    if gpu_count is not None:
        if partition_info["is_heterogeneous"]:
            ui.log(f"⚠ Partition '{partition}' has nodes with different GPU counts:")
            for gc, nc in partition_info["nodes_by_gpu_count"].items():
                ui.log(f"  {gc} GPUs — {nc} node{'s' if nc != 1 else ''}")
            ui.log(f"  Using {gpu_count} GPUs (majority).")
        ui.log(f"✓ Auto-detected {label} GRES: {gpu_count} GPUs per node")
    else:
        if not partition_info["has_gpu_lines"]:
            ui.log(f"• No GPU GRES detected for partition '{partition}'")
        else:
            ui.log(f"• Could not auto-detect GPU GRES for partition '{partition}'")
            if partition_info["unparseable_lines"]:
                ui.log(f"  Failed to parse: {', '.join(partition_info['unparseable_lines'])}")


def prompt_slurm_info(
    ui: UIInterface, defaults: Optional[Dict[str, str]] = None, express_mode: bool = False
) -> Optional[dict]:
    """Prompt the user for SLURM information using UI abstraction.

    Args:
        ui: UIInterface implementation for user interaction
        defaults: Default SLURM values from system config (if available)
        express_mode: Whether this is being called from express mode (shows default messages)

    Returns:
        dict: Dictionary containing SLURM configuration information
    """
    ui.print_section("SLURM Configuration")
    ui.log("Please provide the following SLURM information for your cluster:")
    ui.log("Note: This information will also be used for the launcher.")

    # Get accounts using business logic
    accounts = slurm.get_user_accounts()

    if accounts:
        # If there are too many accounts, limit the display to avoid overwhelming the user
        if len(accounts) > 10:
            display_accounts = accounts[:10]
            ui.log(f"Your accounts (showing 10 of {len(accounts)}): {', '.join(display_accounts)}")
            ui.log("Type '?' to see all your accounts.")
        else:
            ui.log(f"Your accounts: {', '.join(accounts)}")
    else:
        ui.log("Could not automatically detect your SLURM accounts. Please enter your account name manually.")

    def validate_account(account_input):
        if not account_input:
            return True

        validation_result = slurm.validate_account(account_input, accounts)

        if validation_result == 'valid':
            return True
        elif validation_result == 'show_all':
            # Show all accounts when user types '?'
            ui.log("\nAll available accounts:")
            # Display accounts in a more readable format, 5 per line
            for i in range(0, len(accounts), 5):
                ui.log("  " + ", ".join(accounts[i : i + 5]))
            ui.log("")
            return "Type your account name"
        elif validation_result.startswith(INVALID_PREFIX):
            return validation_result[len(INVALID_PREFIX) :]
        else:
            return validation_result

    # Use provided default if available and valid, otherwise use single account default
    if defaults and 'account' in defaults and defaults['account'] in accounts:
        default_account = defaults['account']
        if express_mode:
            ui.log(f"Using saved default account: {default_account}")
    else:
        default_account = accounts[0] if len(accounts) == 1 else ""

    account = ui.prompt_text("SLURM account:", default=default_account, validate=validate_account)

    if account is None:
        return None

    # Get available partitions using business logic
    partitions, default_partition = slurm.get_available_partitions()
    ui.log("")
    if partitions:
        # If there are too many partitions, limit the display
        if len(partitions) > 10:
            display_partitions = partitions[:10]
            ui.log(f"Available partitions (showing 10 of {len(partitions)}): {', '.join(display_partitions)}")
            ui.log("Type '?' to see all available partitions.")
        else:
            ui.log(f"Available partitions: {', '.join(partitions)}")

        if default_partition:
            ui.log(f"Default partition: {default_partition}")
    else:
        ui.log("Could not retrieve partition information. You'll need to enter it manually.")

    def validate_partition(partition_input):
        if not partition_input:
            return "Partition cannot be blank. Please enter a partition name."

        validation_result = slurm.validate_partition(partition_input, partitions)

        if validation_result == 'valid':
            return True
        elif validation_result == 'show_all':
            # Show all partitions when user types '?'
            ui.log("\nAll available partitions:")
            # Display partitions in a more readable format, 5 per line
            for i in range(0, len(partitions), 5):
                ui.log("  " + ", ".join(partitions[i : i + 5]))
            ui.log("")
            return "Type your partition name"
        elif validation_result.startswith(INVALID_PREFIX):
            return validation_result[len(INVALID_PREFIX) :]
        else:
            return validation_result

    def is_valid_partition_default(partition_default: str) -> bool:
        """Check whether a saved partition default is valid for current partitions."""
        return bool(partition_default) and slurm.validate_partition(partition_default, partitions) == 'valid'

    # Use provided default if available and valid, otherwise use detected default or single partition
    if defaults and is_valid_partition_default(defaults.get('gpu_partition', '')):
        gpu_default = ",".join(slurm.normalize_partition_list(defaults['gpu_partition']))
        if express_mode:
            ui.log(f"Using saved default GPU partition: {gpu_default}")
    elif default_partition:
        gpu_default = default_partition
    elif len(partitions) == 1:
        # If only one partition is found, use it as the default
        gpu_default = partitions[0]
    else:
        gpu_default = ""

    cpu_default = slurm.get_default_cpu_partition(partitions)
    if defaults and is_valid_partition_default(defaults.get('cpu_partition', '')):
        final_cpu_default = ",".join(slurm.normalize_partition_list(defaults['cpu_partition']))
        if express_mode:
            ui.log(f"Using saved default CPU partition: {final_cpu_default}")
    elif cpu_default:
        final_cpu_default = cpu_default
    elif len(partitions) == 1:
        # If only one partition is found, use it as the default
        final_cpu_default = partitions[0]
    else:
        final_cpu_default = ""

    ui.log("This is the partition where you want to run the benchmarks.")
    ui.log("You can specify multiple partitions as a comma-separated list (e.g., 'gpu1,gpu2').")

    while True:
        gpu_partition = ui.prompt_text("SLURM GPU partition:", default=gpu_default, validate=validate_partition)
        if gpu_partition is None:
            return None

        # Normalize comma-separated list (remove spaces between partitions for SLURM)
        gpu_partition = ",".join(slurm.normalize_partition_list(gpu_partition))

        ui.log("")
        ui.log(
            "This is the partition for tasks that do not require GPUs. If you only have GPU partitions, use a low priority queue."
        )

        cpu_partition = ui.prompt_text("SLURM CPU partition:", default=final_cpu_default, validate=validate_partition)
        if cpu_partition is None:
            return None

        # Normalize comma-separated list (remove spaces between partitions for SLURM)
        cpu_partition = ",".join(slurm.normalize_partition_list(cpu_partition))

        ui.log("")
        try:
            gres_info = slurm.detect_partition_gres([gpu_partition, cpu_partition])
            break
        except ValueError as e:
            ui.log(f"Partition GRES conflict detected: {e}")
            ui.log("Please enter partition lists with compatible GPU GRES settings.")
            gpu_default = gpu_partition
            final_cpu_default = cpu_partition
            ui.log("")
            continue
        except RuntimeError:
            # sinfo failed but system has SLURM - fall back to manual prompting
            ui.log("Falling back to manual GRES entry.")
            gres_info = {
                gpu_partition: {
                    "gpu_count": None,
                    "has_gpu_lines": True,
                    "unparseable_lines": [],
                    "is_heterogeneous": False,
                },
                cpu_partition: {
                    "gpu_count": None,
                    "has_gpu_lines": True,
                    "unparseable_lines": [],
                    "is_heterogeneous": False,
                },
            }
            break

    gpu_partition_info = gres_info[gpu_partition]
    cpu_partition_info = gres_info[cpu_partition]

    gpu_partition_gres_value = gpu_partition_info["gpu_count"]
    cpu_partition_gres_value = cpu_partition_info["gpu_count"]

    # Ensure consistency when partitions are identical
    if gpu_partition == cpu_partition:
        cpu_partition_gres_value = gpu_partition_gres_value
        cpu_partition_info = gpu_partition_info

    # Display detection results
    _print_gres_detection_result(ui, gpu_partition, True, gpu_partition_info)
    if cpu_partition != gpu_partition:
        _print_gres_detection_result(ui, cpu_partition, False, cpu_partition_info)

    # --------------------  Manual fallback prompting  -----------------------
    if gpu_partition_gres_value is None and gpu_partition_info["has_gpu_lines"]:
        gpu_partition_gres_value = _prompt_gpu_gres(ui, gpu_partition)

    if cpu_partition != gpu_partition and cpu_partition_gres_value is None and cpu_partition_info["has_gpu_lines"]:
        cpu_partition_gres_value = _prompt_gpu_gres(ui, cpu_partition)

    # Maintain consistency for identical partitions
    if gpu_partition == cpu_partition:
        cpu_partition_gres_value = gpu_partition_gres_value

    slurm_config = {
        "slurm": {
            "account": account,
            "gpu_partition": gpu_partition,
            "cpu_partition": cpu_partition,
            "gpu_partition_gres": gpu_partition_gres_value,
            "cpu_partition_gres": cpu_partition_gres_value,
        }
    }

    return slurm_config

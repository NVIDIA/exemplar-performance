# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Environment configuration prompts for LLMB installer."""

import os
from typing import Dict, Optional

from llmb_install.environment.detector import is_uv_installed
from llmb_install.ui.interface import UIInterface


def prompt_environment_type(ui: UIInterface, default: Optional[str] = None) -> str:
    """Resolve the recipe environment type for fresh interactive installs.

    uv is the only supported choice for newly-created recipe environments. The
    installer still supports existing conda/venv configs elsewhere for resume,
    incremental, and headless compatibility.

    Args:
        ui: UI interface for user interaction
        default: Deprecated saved environment type from system config, ignored for fresh installs

    Returns:
        str: Selected environment type ('uv')
    """
    ui.log("Environment Configuration")
    ui.log("------------------------")

    if not is_uv_installed():
        ui.log("Error: uv is required to create recipe environments.")
        ui.log("Please install uv and rerun the installer.")
        raise SystemExit(1)

    if default and default != "uv":
        ui.log(f"Ignoring saved environment type '{default}'; new recipe environments use uv.")
    else:
        ui.log("Using uv for recipe environments.")

    return "uv"


def prompt_environment_variables(
    ui: UIInterface, defaults: Optional[Dict[str, str]] = None, express_mode: bool = False
) -> Optional[Dict[str, str]]:
    """Prompt the user for environment variables.

    Currently prompts for HF_TOKEN, but designed to be easily expandable
    for additional environment variables in the future.

    Args:
        ui: UI interface for user interaction
        defaults: Default environment variables from system config (if available)
        express_mode: Whether this is being called from express mode

    Returns:
        Dict[str, str]: Dictionary containing the environment variables, or None if cancelled
    """
    # In express mode, if we have HF_TOKEN in defaults, use it without prompting
    if express_mode and defaults and 'HF_TOKEN' in defaults and defaults['HF_TOKEN']:
        ui.log("Using saved HF_TOKEN from system configuration.")
        return {'HF_TOKEN': defaults['HF_TOKEN']}

    ui.log("\nEnvironment Variables")
    ui.log("--------------------")
    ui.log("")

    env_vars = {}

    # HF_TOKEN validation function - now optional
    def validate_hf_token(token: str) -> bool | str:
        if not token:
            return True  # Allow empty token
        if not token.startswith('hf_'):
            return "HF_TOKEN must start with 'hf_' if provided."
        return True

    # Determine default HF_TOKEN (prefer saved config over environment)
    default_hf_token = ""
    source_message = ""

    if defaults and 'HF_TOKEN' in defaults and defaults['HF_TOKEN']:
        default_hf_token = defaults['HF_TOKEN']
        source_message = f"Found saved HF_TOKEN: {default_hf_token[:10]}..."
    else:
        env_hf_token = os.environ.get('HF_TOKEN', '')
        if env_hf_token:
            default_hf_token = env_hf_token
            source_message = f"Found existing HF_TOKEN in environment: {env_hf_token[:10]}..."

    if source_message:
        # Token found - provide brief context
        ui.log("=" * 70)
        ui.log("Hugging Face Token (HF_TOKEN)")
        ui.log("=" * 70)
        ui.log("")
        ui.log("Most workloads require a Hugging Face token to download models and datasets.")
        ui.log("")
        ui.log(source_message)
        ui.log("Press Enter to use this token, or enter a new one below.")
        ui.log("")
    else:
        # No existing token - make it clear this is important
        ui.log("=" * 70)
        ui.log("Hugging Face Token (HF_TOKEN)")
        ui.log("=" * 70)
        ui.log("")
        ui.log("Most workloads require a Hugging Face token to download models and datasets.")
        ui.log("")
        ui.log("ACTION REQUIRED:")
        ui.log("  1. Get your token: https://huggingface.co/settings/tokens")
        ui.log("  2. Enter it below")
        ui.log("")
        ui.log("Skip only if you know your workloads don't require Hugging Face access.", level='warning')
        ui.log("")

    hf_token = ui.prompt_text(
        f"Enter HF_TOKEN{' (press Enter to use existing)' if default_hf_token else ''}:",
        default=default_hf_token,
        validate=validate_hf_token,
    )

    if hf_token is None:  # User cancelled
        return None

    if hf_token:
        env_vars['HF_TOKEN'] = hf_token
        ui.log("✓ HF_TOKEN configured.", level='success')
    elif default_hf_token:
        env_vars['HF_TOKEN'] = default_hf_token
        ui.log("✓ Using existing HF_TOKEN.", level='success')
    else:
        ui.log("⚠ No HF_TOKEN provided - workloads requiring Hugging Face will fail.", level='warning')

    return env_vars

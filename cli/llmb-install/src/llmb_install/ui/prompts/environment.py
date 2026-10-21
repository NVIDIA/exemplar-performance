# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Environment configuration prompts for LLMB installer."""

import os
from typing import Dict, List, Optional

from llmb_install.config.models import EnvironmentPrompt
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


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "********"
    return f"{value[:10]}..."


def _existing_value(env_vars: Dict[str, str], name: str) -> str:
    """Return a normalized saved value, falling back to the process environment."""
    saved = (env_vars.get(name) or "").strip()
    return saved or os.environ.get(name, "").strip()


def _prompt_variable(ui: UIInterface, prompt: EnvironmentPrompt, default: str = "") -> Optional[str]:
    default = default.strip()
    status = "required" if prompt.required else "optional"
    ui.log(f"\n{prompt.name} ({status})")
    if prompt.description:
        ui.log(prompt.description)

    if default:
        ui.log(f"Found existing {prompt.name}: {_mask(default)}")

    def _validator(v: str) -> bool | str:
        if not v.strip() and default:
            return True
        if prompt.required and not v.strip():
            needed = ", ".join(prompt.workloads) or "selected workloads"
            return f"{prompt.name} is required by {needed}."
        return True

    existing_hint = ""
    if default:
        action = "press Enter to use existing"
        if not prompt.required:
            action += ", or type 'clear' to remove"
        existing_hint = f" ({action})"

    val = ui.prompt_text(
        f"Enter {prompt.name}{existing_hint}:",
        default="",
        validate=_validator,
    )
    if val is None:
        return None
    val = val.strip()
    if not val and default:
        val = default
    elif not prompt.required and val.lower() in ("none", "clear", "unset"):
        val = ""

    if not val and not prompt.required:
        ui.log(f"⚠ No {prompt.name} provided - workloads requiring it will fail.\n", level="warning")
    return val


def prompt_environment_variables(
    ui: UIInterface,
    defaults: Optional[Dict[str, str]] = None,
    express_mode: bool = False,
    prompts: Optional[List[EnvironmentPrompt]] = None,
) -> Optional[Dict[str, str]]:
    """Prompt the user for environment variables.

    Prompts for the supplied environment-variable declarations. Saved values from
    system config are preserved.

    Args:
        ui: UI interface for user interaction
        defaults: Default environment variables from system config (if available)
        express_mode: Whether this is being called from express mode
        prompts: EnvironmentPrompt declarations from the selected workloads

    Returns:
        Dict[str, str]: Dictionary containing the environment variables, or None if cancelled
    """
    env_vars = dict(defaults or {})
    prompts = prompts or []

    # In express mode: prompt only if a required variable has no saved value
    if express_mode:
        for prompt in prompts:
            if value := _existing_value(env_vars, prompt.name):
                env_vars[prompt.name] = value
        missing = [p for p in prompts if p.required and not (env_vars.get(p.name) or "").strip()]
        if not missing:
            return env_vars
        for prompt in missing:
            val = _prompt_variable(ui, prompt, default="")
            if val is None:
                return None
            env_vars[prompt.name] = val
        return env_vars

    ui.log("\nEnvironment Variables")
    ui.log("--------------------")
    ui.log("")

    for prompt in prompts:
        default_val = _existing_value(env_vars, prompt.name)
        val = _prompt_variable(ui, prompt, default=default_val)
        if val is None:
            return None
        if val:
            env_vars[prompt.name] = val
        elif not prompt.required:
            env_vars.pop(prompt.name, None)
            os.environ.pop(prompt.name, None)

    return env_vars

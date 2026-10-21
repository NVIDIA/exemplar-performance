# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Configuration file loading and saving for headless installation."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from llmb_install.config.models import PlayfileConfig, migrate_legacy_install_method


def save_installation_config(config_file: str, config_data: Dict[str, Any]) -> None:
    """Save installation configuration to a YAML file.

    Args:
        config_file: Path to the configuration file to save
        config_data: Dictionary containing all installation configuration
    """
    try:
        # Ensure the destination directory exists
        resolved_path = Path(config_file).resolve()
        os.makedirs(resolved_path.parent, exist_ok=True)

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        os.chmod(config_file, 0o600)
        print(f"✓ Configuration saved to: {config_file}")
    except Exception as e:
        print(f"Error saving configuration to {config_file}: {e}")
        raise SystemExit(1) from e


def load_installation_config(config_file: str) -> Dict[str, Any]:
    """Load installation configuration from a YAML file.

    Args:
        config_file: Path to the configuration file to load

    Returns:
        Dict containing all installation configuration

    Raises:
        SystemExit: If the configuration file cannot be loaded or is invalid
    """
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a dictionary")

        config_data = migrate_legacy_install_method(config_data)

        # Validate against playfile schema (structure, types, and playfile-specific rules)
        PlayfileConfig.model_validate(config_data)

        print(f"✓ Configuration loaded from: {config_file}")
        return config_data

    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        raise SystemExit(1) from None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file {config_file}: {e}")
        raise SystemExit(1) from e
    except ValidationError as e:
        print(f"Error: Invalid configuration in {config_file}:")
        for err in e.errors():
            loc = " -> ".join(str(part) for part in err["loc"])
            print(f"  - {loc}: {err['msg']}")
        raise SystemExit(1) from e
    except ValueError as e:
        print(f"Error: Invalid configuration in {config_file}: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        print(f"Error loading configuration from {config_file}: {e}")
        raise SystemExit(1) from e

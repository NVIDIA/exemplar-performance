# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point provider for apps that discover LLMB collectors."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any


@dataclass(frozen=True)
class LlmbCollectorProvider:
    """Runnable collector provider exposed through the ``llmb`` entry point group."""

    name: str = "llmb-collector"

    def commands_config(self) -> Traversable:
        """Return the bundled command config root."""
        return resources.files("llmb_collector").joinpath("commands_config")

    def command_registry(self):
        """Load the command registry for this provider's bundled config."""
        from llmb_collector.command_loader import load_registry_from_directory

        return load_registry_from_directory(self.commands_config())

    def collect(self, config: Any = None, **overrides) -> dict:
        """Run collection using a CollectConfig or keyword overrides."""
        from llmb_collector.collect import CollectConfig, collect

        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        if config is None:
            config = CollectConfig(**overrides)
        return collect(config)


def get_provider() -> LlmbCollectorProvider:
    """Entry point target returning this package's runnable LLMB collector provider."""
    return LlmbCollectorProvider()

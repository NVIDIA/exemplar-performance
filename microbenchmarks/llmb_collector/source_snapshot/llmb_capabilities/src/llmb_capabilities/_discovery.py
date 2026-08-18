# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry-point discovery for installed capability providers."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from importlib.metadata import entry_points
from typing import Any

from llmb_capabilities._constants import CAPABILITY_GROUP

_logger = logging.getLogger(__name__)


def _load_capabilities(entry_point: Any) -> tuple[Any, ...] | None:
    """Load one provider's iterable advertisement, or skip malformed input."""
    try:
        capabilities = entry_point.load()
    except Exception:
        _logger.warning("Skipping capability provider %r: failed to load", entry_point.name, exc_info=True)
        return None
    if not isinstance(capabilities, Iterable) or isinstance(capabilities, (str, bytes)):
        _logger.warning("Skipping capability provider %r: malformed advertisement payload", entry_point.name)
        return None
    try:
        return tuple(capabilities)
    except Exception:
        _logger.warning(
            "Skipping capability provider %r: failed to iterate advertisement",
            entry_point.name,
            exc_info=True,
        )
        return None


def _advertisement_identity(capability: Any, entry_point_name: str) -> tuple[str, str] | None:
    """Return a valid ``(provider, capability name)`` advertisement identity."""
    name = getattr(capability, "name", None)
    if not isinstance(name, str):
        return None
    metadata = getattr(capability, "metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        return None
    provider = metadata.get("provider", entry_point_name)
    if not isinstance(provider, str):
        return None
    return provider, name


def discover_providers() -> dict[str, dict[str, Any]]:
    """Return every loadable provider and its advertised capabilities.

    Results are grouped as ``{provider: {capability_name: capability}}``.
    Discovery deliberately does not apply consumer-specific capability names,
    version ceilings, or deny-lists; consumers layer those policies on the
    complete installed registry.
    """
    providers: dict[str, dict[str, Any]] = {}
    try:
        provider_entry_points = entry_points(group=CAPABILITY_GROUP)
    except Exception:  # pragma: no cover - importlib.metadata edge cases
        _logger.debug("Failed to enumerate %s entry points", CAPABILITY_GROUP, exc_info=True)
        return providers

    for entry_point in provider_entry_points:
        capabilities = _load_capabilities(entry_point)
        if capabilities is None:
            continue
        for capability in capabilities:
            identity = _advertisement_identity(capability, entry_point.name)
            if identity is None:
                continue
            provider, name = identity
            providers.setdefault(provider, {})[name] = capability
    return providers


__all__ = ["discover_providers"]

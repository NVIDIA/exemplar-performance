# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow protocol for the ``system.commands-config`` capability."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from importlib.resources.abc import Traversable
from typing import Protocol, runtime_checkable


@runtime_checkable
class SystemCommandsConfigCapability(Protocol):
    """A provider of the packaged ``commands_config`` tree.

    Today's reference producer (:mod:`llmb_collector`) advertises this
    so consumers can locate the YAML / JSON files that drive
    :data:`SYSTEM_COLLECT` without importing the collector. The return
    type is :class:`importlib.resources.abc.Traversable` so it works
    portably for directory installs, zipapps, and zip-archived wheels.

    Consumers that just want a string path can read
    ``capability.metadata["commands_config_path"]`` instead of
    invoking — but that string is only a usable filesystem path for
    directory-based installs. Use :meth:`invoke` and the
    :class:`Traversable` interface (``open()``, ``iterdir()``,
    ``read_text()``) when portability matters.
    """

    name: str
    version: int
    metadata: Mapping[str, str]

    def invoke(
        self,
        namespace: argparse.Namespace,
        /,
        *,
        prefix: str | None = None,
    ) -> Traversable:
        """Return the packaged commands-config tree as a :class:`Traversable`.

        :param namespace: an :class:`argparse.Namespace`. Reserved for
            future ``add_arguments`` support; today's producer ignores
            it.
        :param prefix: optional flag-prefix when the producer is
            embedded under a namespaced argparse section.
        :returns: a :class:`importlib.resources.abc.Traversable` that
            consumers can navigate with the standard resources API.
        """
        ...


__all__ = ["SystemCommandsConfigCapability"]

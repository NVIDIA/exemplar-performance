# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow protocol for the ``system.collect`` capability."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SystemCollectCapability(Protocol):
    """A provider of host/network/container/env system-info collection.

    Returns a JSON-serializable :class:`dict`. Today's reference
    producer (:mod:`llmb_collector`) writes that dict to
    ``_cloudperf/system_info.json`` by convention; consumers
    (:mod:`llmb_uploader`) may embed it in an upload archive at the
    same relative path.
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
        config: Any | None = None,
    ) -> dict[str, Any]:
        """Run a collection and return the JSON-serializable result.

        :param namespace: an :class:`argparse.Namespace`. Reserved for
            future ``add_arguments`` support; today's producer ignores
            it.
        :param prefix: optional flag-prefix when the producer is
            embedded under a namespaced argparse section.
        :param config: a producer-defined config object narrowing the
            collection scope, or ``None`` to use the producer's "all
            realms on" defaults. The shape is genuinely producer-
            specific (today's reference producer needs 13+ fields, not
            just realm flags), so the contract intentionally types
            this as :class:`typing.Any` rather than declaring a
            structural :class:`Protocol` that consumers couldn't
            satisfy from scratch. Consumers that want to drive scope
            should import the producer's concrete config class
            (e.g. ``llmb_collector.collect.CollectConfig``).
        :returns: a JSON-serializable dict.
        """
        ...


__all__ = ["SystemCollectCapability"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow protocol for the ``tls.ssl-context`` capability."""

from __future__ import annotations

import argparse
import ssl
from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class TlsSslContextCapability(Protocol):
    """A provider of :class:`ssl.SSLContext` instances for HTTPS calls.

    Producers (e.g. :mod:`llmb_auth`) baked their organization's
    internal CA chain into the returned context. Consumers receive a
    context they can hand directly to ``httpx.Client(verify=...)`` or
    similar, no further setup required. The user's ``--ca-bundle`` (or
    ``$LLMB_CA_BUNDLE``) is read off ``namespace`` and layered
    additively on top by the producer.
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
    ) -> ssl.SSLContext:
        """Return an :class:`ssl.SSLContext` ready for HTTPS use.

        :param namespace: an :class:`argparse.Namespace` carrying flags
            the producer contributed via ``add_arguments`` — most
            notably ``--ca-bundle`` for additive trust.
        :param prefix: optional flag-prefix when the producer was
            embedded under a namespaced argparse section. ``None`` =
            canonical default prefix.
        :returns: a context configured with the producer's baked CAs
            and any additive trust the user supplied.
        """
        ...


__all__ = ["TlsSslContextCapability"]

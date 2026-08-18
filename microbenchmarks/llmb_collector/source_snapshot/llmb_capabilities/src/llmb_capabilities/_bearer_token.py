# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow protocol for the ``oauth.bearer-token`` capability."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

#: Token-acquisition strategies accepted by
#: :meth:`BearerTokenCapability.invoke`. Today's reference producer
#: (:mod:`llmb_auth`) accepts ``"ssa"`` (service account, default for
#: CI), ``"auth-code"`` (interactive browser login), and
#: ``"device-code"`` (interactive off-device login for headless hosts).
#: Adding a new method is a non-breaking change of *this* contract
#: package only when it's also additive in every consumer's switch —
#: bump the capability ``version`` if a consumer would mis-route an
#: unknown method.
BearerTokenMethod = Literal["ssa", "auth-code", "device-code"]


@runtime_checkable
class BearerTokenCapability(Protocol):
    """A provider of bearer tokens for ``Authorization: Bearer <token>``.

    The :meth:`invoke` signature mirrors today's reference producer in
    :mod:`llmb_auth.capabilities` so the contract is the lowest common
    denominator that already works in the field. Consumers should pin
    by ``capability.name == "oauth.bearer-token"`` and
    ``capability.version <= SUPPORTED["oauth.bearer-token"]`` before
    calling :meth:`invoke`.
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
        method: BearerTokenMethod = "ssa",
        force: bool = False,
    ) -> str:
        """Return an access token string.

        :param namespace: an :class:`argparse.Namespace` carrying the
            user's auth-related flags. Producers that contribute
            ``add_arguments`` write into the same namespace; the
            producer reads back via ``namespace.<flag>`` honoring
            ``prefix``.
        :param prefix: optional flag-prefix used when the consumer
            embedded the producer's argparse contribution under a
            namespaced section. ``None`` means the canonical default
            prefix.
        :param method: ``"ssa"`` (service account), ``"auth-code"``
            (interactive browser login), or ``"device-code"`` (interactive
            off-device login for headless/SSH hosts). Producers should
            raise :class:`ValueError` for any other value rather than
            silently falling back.
        :param force: when ``True``, bypass any caching and fetch a
            fresh token. Honored on a best-effort basis by producers
            that maintain their own cache.
        :returns: the access token as a UTF-8 string.
        """
        ...


__all__ = ["BearerTokenCapability", "BearerTokenMethod"]

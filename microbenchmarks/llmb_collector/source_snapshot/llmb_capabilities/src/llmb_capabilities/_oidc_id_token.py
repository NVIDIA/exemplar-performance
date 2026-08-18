# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow protocol for the ``oidc.id-token`` capability."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class OidcIdTokenCapability(Protocol):
    """A provider of OIDC id_tokens (signed JWTs).

    Distinct from :class:`llmb_capabilities.BearerTokenCapability`: an
    OAuth2 access_token is opaque to the client and validated by the
    resource server via introspection or scope, whereas an OIDC id_token
    is a signed JWT the resource server decodes and audience-checks.
    Same ``Authorization: Bearer <string>`` HTTP envelope, different
    validation model — hence a separate capability rather than a token-
    type kwarg on ``oauth.bearer-token``.

    Unlike ``oauth.bearer-token``, there is no ``method=`` kwarg: an
    id_token is intrinsic to the OIDC layer, so a provider that has no
    OIDC-issuing session (e.g. SSA-only / ``client_credentials``) simply
    does not advertise this capability at all. Consumers pick the
    provider by ``metadata["provider"]`` when several are present.

    Consumers should pin by ``capability.name == "oidc.id-token"`` and
    ``capability.version <= SUPPORTED["oidc.id-token"]`` before calling
    :meth:`invoke`.
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
        force: bool = False,
    ) -> str:
        """Return an OIDC id_token as a signed-JWT string.

        :param namespace: an :class:`argparse.Namespace` carrying the
            user's auth-related flags. Producers that contribute
            ``add_arguments`` write into the same namespace; the
            producer reads back via ``namespace.<flag>`` honoring
            ``prefix``.
        :param prefix: optional flag-prefix used when the consumer
            embedded the producer's argparse contribution under a
            namespaced section. ``None`` means the canonical default
            prefix.
        :param force: when ``True``, bypass any caching and fetch a
            fresh id_token. Honored on a best-effort basis by producers
            that maintain their own cache.
        :returns: the id_token as a three-segment JWT string (header,
            payload, signature separated by ``.``).
        """
        ...


__all__ = ["OidcIdTokenCapability"]

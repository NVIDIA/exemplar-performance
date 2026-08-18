# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for capability registry identifiers.

The entry-point group name and the four currently-known capability names
were previously duplicated across :mod:`llmb_uploader.capability_discovery`,
:mod:`llmb_auth.capabilities`, and :mod:`llmb_collector.capabilities`. They
live here so a typo or rename is a Python-level breaking change at import
time rather than a silent "no provider found" at runtime.

The :data:`SUPPORTED` table records the highest schema version each
capability has been **defined** at within this contract package. It is
the ceiling consumers compare against to decide whether to consume a
capability advertised at a given version. Bump a value here in lockstep
with adding the corresponding ``XxxCapabilityV2`` :class:`Protocol`
sibling — never as a standalone change.
"""

from __future__ import annotations

from typing import Final

#: Entry-point group every provider advertises into and every consumer
#: enumerates. The string is part of the public contract — do not change
#: it without a major bump and a coordinated migration of all providers
#: and consumers.
CAPABILITY_GROUP: Final[str] = "llmb_capabilities"

#: Bearer token suitable for ``Authorization: Bearer <token>`` against
#: NVIDIA-internal services. Today advertised by :mod:`llmb_auth`.
BEARER_TOKEN: Final[str] = "oauth.bearer-token"

#: OIDC id_token (signed JWT) suitable for services that decode/validate
#: a Starfleet-OIDC bearer rather than an opaque OAuth2 access_token.
#: Advertised only by providers with an OIDC-issuing session (auth-code
#: today); SSA/``client_credentials``-only providers do not advertise this.
#: Today advertised by :mod:`llmb_auth`.
OIDC_ID_TOKEN: Final[str] = "oidc.id-token"

#: :class:`ssl.SSLContext` trusting NVIDIA-internal CAs, with optional
#: additive trust layered on top from the consumer's ``--ca-bundle``.
#: Today advertised by :mod:`llmb_auth`.
TLS_SSL_CONTEXT: Final[str] = "tls.ssl-context"

#: Host / network / container / env collection returned as a
#: JSON-serializable :class:`dict`. Today advertised by
#: :mod:`llmb_collector`.
SYSTEM_COLLECT: Final[str] = "system.collect"

#: The packaged ``commands_config`` tree used by :data:`SYSTEM_COLLECT`,
#: returned as an :class:`importlib.resources.abc.Traversable`. Today
#: advertised by :mod:`llmb_collector`.
SYSTEM_COMMANDS_CONFIG: Final[str] = "system.commands-config"

#: Upload of a local results archive to a storage backend (S3, Swift,
#: Google Drive, ...). The first **model-first** capability: ``invoke``
#: takes an already-built :attr:`Capability.args_model` instance rather
#: than an :class:`argparse.Namespace`. Today advertised by
#: :mod:`llmb_uploader`.
STORAGE_UPLOAD: Final[str] = "storage.upload"

#: Post-processing of a prior upload submission (e.g. triggering
#: downstream publication). Also model-first, for the same reason as
#: :data:`STORAGE_UPLOAD`. Today advertised by :mod:`llmb_uploader`.
POST_PROCESSOR_PROCESS: Final[str] = "post-processor.process"

#: Per-capability schema-version ceiling defined by this package. A
#: provider that advertises ``capability.version > SUPPORTED[name]`` is
#: announcing a future shape this contract version doesn't describe yet
#: and a consumer using this contract version cannot consume it safely.
#:
#: Bump a value here only when adding a new ``XxxCapabilityVN``
#: :class:`Protocol` alongside the existing one (additive evolution).
#: Never edit a value as a standalone change, and never remove a key
#: without a major version bump of this package.
SUPPORTED: Final[dict[str, int]] = {
    BEARER_TOKEN: 1,
    OIDC_ID_TOKEN: 1,
    TLS_SSL_CONTEXT: 1,
    SYSTEM_COLLECT: 1,
    SYSTEM_COMMANDS_CONFIG: 1,
    STORAGE_UPLOAD: 1,
    POST_PROCESSOR_PROCESS: 1,
}

__all__ = [
    "BEARER_TOKEN",
    "CAPABILITY_GROUP",
    "OIDC_ID_TOKEN",
    "POST_PROCESSOR_PROCESS",
    "STORAGE_UPLOAD",
    "SUPPORTED",
    "SYSTEM_COLLECT",
    "SYSTEM_COMMANDS_CONFIG",
    "TLS_SSL_CONTEXT",
]

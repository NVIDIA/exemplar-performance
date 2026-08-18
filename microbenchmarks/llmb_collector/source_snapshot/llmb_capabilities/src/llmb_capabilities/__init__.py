# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-package contract for the LLMB capability registry.

See the package-level README for the architecture overview, the
versioning policy, and end-to-end producer / consumer examples. The
public surface re-exported here is the *only* surface providers and
consumers should import from; underscore-prefixed modules are
implementation detail and may be reorganized between minor versions.

The conformance helper (:func:`llmb_capabilities.conformance.assert_capability_conforms`)
is exposed via its own submodule rather than this top-level namespace
so producers can keep their conformance test imports tight and
unambiguous.
"""

from __future__ import annotations

from llmb_capabilities._bearer_token import BearerTokenCapability, BearerTokenMethod
from llmb_capabilities._capability import Capability, CapabilityLike, ModelValidatable
from llmb_capabilities._constants import (
    BEARER_TOKEN,
    CAPABILITY_GROUP,
    OIDC_ID_TOKEN,
    POST_PROCESSOR_PROCESS,
    STORAGE_UPLOAD,
    SUPPORTED,
    SYSTEM_COLLECT,
    SYSTEM_COMMANDS_CONFIG,
    TLS_SSL_CONTEXT,
)
from llmb_capabilities._discovery import discover_providers
from llmb_capabilities._invoke import invoke_async, invoke_sync
from llmb_capabilities._oidc_id_token import OidcIdTokenCapability
from llmb_capabilities._post_processor_process import PostProcessorProcessCapability
from llmb_capabilities._resolve_args import resolve_capability_args
from llmb_capabilities._storage_upload import StorageUploadCapability
from llmb_capabilities._system_collect import SystemCollectCapability
from llmb_capabilities._system_commands_config import SystemCommandsConfigCapability
from llmb_capabilities._tls_ssl_context import TlsSslContextCapability

__all__ = [
    "BEARER_TOKEN",
    "BearerTokenCapability",
    "BearerTokenMethod",
    "CAPABILITY_GROUP",
    "Capability",
    "CapabilityLike",
    "ModelValidatable",
    "OIDC_ID_TOKEN",
    "OidcIdTokenCapability",
    "POST_PROCESSOR_PROCESS",
    "PostProcessorProcessCapability",
    "STORAGE_UPLOAD",
    "StorageUploadCapability",
    "SUPPORTED",
    "SYSTEM_COLLECT",
    "SYSTEM_COMMANDS_CONFIG",
    "SystemCollectCapability",
    "SystemCommandsConfigCapability",
    "TLS_SSL_CONTEXT",
    "TlsSslContextCapability",
    "discover_providers",
    "invoke_async",
    "invoke_sync",
    "resolve_capability_args",
]

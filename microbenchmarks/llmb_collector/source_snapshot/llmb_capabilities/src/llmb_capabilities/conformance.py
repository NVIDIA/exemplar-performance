# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-side conformance helper for the LLMB capability contract.

Each provider drops a single test of the form::

    from llmb_capabilities.conformance import assert_capability_conforms
    from llmb_auth.capabilities import CAPABILITIES

    def test_capabilities_conform():
        for cap in CAPABILITIES:
            assert_capability_conforms(cap)

…and the contract is enforced against the producer's ``CAPABILITIES``
tuple in their own CI. A signature drift (e.g. removing a documented
kwarg from ``invoke``) lights up *before* the provider is published, so
consumers never see the broken release.

This module deliberately raises plain :class:`AssertionError` rather
than a custom exception so the helper composes naturally with
:mod:`pytest` without forcing it as a dependency. Producers that want
``pytest.fail``-style messages can wrap the call.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Final

from llmb_capabilities._bearer_token import BearerTokenCapability
from llmb_capabilities._capability import CapabilityLike
from llmb_capabilities._constants import (
    BEARER_TOKEN,
    OIDC_ID_TOKEN,
    POST_PROCESSOR_PROCESS,
    STORAGE_UPLOAD,
    SUPPORTED,
    SYSTEM_COLLECT,
    SYSTEM_COMMANDS_CONFIG,
    TLS_SSL_CONTEXT,
)
from llmb_capabilities._oidc_id_token import OidcIdTokenCapability
from llmb_capabilities._post_processor_process import PostProcessorProcessCapability
from llmb_capabilities._storage_upload import StorageUploadCapability
from llmb_capabilities._system_collect import SystemCollectCapability
from llmb_capabilities._system_commands_config import SystemCommandsConfigCapability
from llmb_capabilities._tls_ssl_context import TlsSslContextCapability

#: Map a capability ``name`` to its narrow :class:`typing.Protocol`. New
#: capabilities should register their protocol here so the conformance
#: check covers them automatically. A name absent from this table only
#: gets the generic :class:`CapabilityLike` shape check, which is the
#: minimum bar but skips signature-level verification.
_NARROW_PROTOCOLS: Final[dict[str, type]] = {
    BEARER_TOKEN: BearerTokenCapability,
    OIDC_ID_TOKEN: OidcIdTokenCapability,
    TLS_SSL_CONTEXT: TlsSslContextCapability,
    SYSTEM_COLLECT: SystemCollectCapability,
    SYSTEM_COMMANDS_CONFIG: SystemCommandsConfigCapability,
    STORAGE_UPLOAD: StorageUploadCapability,
    POST_PROCESSOR_PROCESS: PostProcessorProcessCapability,
}

#: Model-first capability names: :attr:`Capability.args_model` must be
#: set (not ``None``) for these, since their narrow protocol's
#: ``invoke`` takes a pre-built model instance rather than an
#: :class:`argparse.Namespace`.
_REQUIRES_ARGS_MODEL: Final[frozenset[str]] = frozenset({STORAGE_UPLOAD, POST_PROCESSOR_PROCESS})


def assert_capability_conforms(capability: Any) -> None:
    """Raise :class:`AssertionError` if ``capability`` violates the contract.

    The checks are deliberately ordered most-general to most-specific
    so a clear failure message points at the first divergence:

    1. Generic shape (``CapabilityLike``) — name / version / metadata /
       args_model / invoke attributes exist.
    2. Name is one this contract version knows.
    3. Schema version does not exceed the contract's ceiling.
    4. ``metadata["provider"]`` is set (consumers route by it).
    5. ``args_model`` is set for model-first capabilities (e.g.
       :data:`STORAGE_UPLOAD`).
    6. The narrow per-capability :class:`typing.Protocol` accepts the
       object (when one is registered for the name).
    7. ``invoke`` accepts at least one positional parameter — every
       capability's documented signature starts with the namespace
       positional, so a callable with no params can't possibly satisfy
       any narrow protocol.
    8. ``invoke`` accepts every keyword parameter declared by the
       narrow protocol (or absorbs them via ``**kwargs``). The
       :func:`typing.runtime_checkable` :class:`typing.Protocol`
       :func:`isinstance` check only verifies attribute presence, not
       signature shape, so this final step catches the case where a
       producer's ``invoke`` silently dropped a documented kwarg.

    :raises AssertionError: with a message identifying the first
        violation. Never raises any other exception type.

    Each step below is its own small ``_check_*`` / ``_require_*``
    function using ``if ...: raise AssertionError(...)`` rather than a
    bare ``assert`` statement, so this check can't be silently disabled
    by running the interpreter with ``-O`` / ``PYTHONOPTIMIZE`` -
    unlike a test's own ``assert``, this function is shipped as library
    code that may run outside pytest.
    """
    _check_capability_like(capability)

    name = capability.name
    _check_known_name(name)
    _check_version_ceiling(name, capability.version)
    _check_provider_metadata(name, capability.metadata)

    if name in _REQUIRES_ARGS_MODEL:
        _check_args_model_present(name, capability.args_model)

    narrow = _NARROW_PROTOCOLS.get(name)
    if narrow is not None:
        _check_narrow_protocol(name, capability, narrow)

    signature = _require_introspectable_invoke(name, capability.invoke)
    _check_invoke_has_positional(name, signature)

    if narrow is not None:
        _check_invoke_accepts_documented_kwargs(name, narrow, signature)


def _check_capability_like(capability: Any) -> None:
    if not isinstance(capability, CapabilityLike):
        raise AssertionError(
            f"{capability!r} is missing required Capability attributes (name, version, metadata, args_model, invoke)"
        )


def _check_known_name(name: str) -> None:
    if name not in SUPPORTED:
        raise AssertionError(f"{name!r} is not a known capability name; expected one of {sorted(SUPPORTED)}")


def _check_version_ceiling(name: str, version: Any) -> None:
    # CapabilityLike's isinstance check only verifies `version` is present,
    # not that it's an int - a str/float slipping through here would
    # otherwise raise a confusing TypeError from the `>` comparison below.
    if not isinstance(version, int):
        raise AssertionError(f"{name!r}.version must be an int, got {version!r} ({type(version).__name__})")
    ceiling = SUPPORTED[name]
    if version > ceiling:
        raise AssertionError(
            f"{name!r} advertised version {version} exceeds contract ceiling {ceiling}; bump llmb-capabilities first"
        )


def _check_provider_metadata(name: str, metadata: Any) -> None:
    # Same rationale as _check_version_ceiling: CapabilityLike only checks
    # presence, so a non-Mapping (e.g. None) would otherwise raise
    # TypeError from the `in` check below instead of AssertionError.
    if not isinstance(metadata, Mapping):
        raise AssertionError(f"{name!r}.metadata must be a Mapping, got {metadata!r} ({type(metadata).__name__})")
    if "provider" not in metadata:
        raise AssertionError(
            f"{name!r} metadata is missing required 'provider' key; consumers route preference by metadata['provider']"
        )


def _check_args_model_present(name: str, args_model: Any) -> None:
    if args_model is None:
        raise AssertionError(f"{name!r} is model-first and must set args_model to the model class its invoke expects")
    if not callable(getattr(args_model, "model_validate", None)):
        raise AssertionError(
            f"{name!r}.args_model ({args_model!r}) must expose a callable 'model_validate' classmethod "
            "(see ModelValidatable) for resolve_capability_args to build an instance from it"
        )


def _check_narrow_protocol(name: str, capability: Any, narrow: type) -> None:
    if not isinstance(capability, narrow):
        raise AssertionError(
            f"{name!r} does not satisfy {narrow.__name__}; "
            "check that invoke / add_arguments / metadata are exposed correctly"
        )


def _require_introspectable_invoke(name: str, invoke: Any) -> inspect.Signature:
    try:
        return inspect.signature(invoke)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name!r}.invoke is not introspectable: {exc!s}") from exc


def _check_invoke_has_positional(name: str, signature: inspect.Signature) -> None:
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    )
    has_positional = any(p.kind in positional_kinds for p in signature.parameters.values())
    if not has_positional:
        raise AssertionError(
            f"{name!r}.invoke must accept at least one positional parameter "
            f"(argparse.Namespace or args_model instance), but its signature is {signature!s}"
        )


def _check_invoke_accepts_documented_kwargs(name: str, narrow: type, signature: inspect.Signature) -> None:
    accepts_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_var_keyword:
        return

    keyword_kinds = (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    accepted = {pname for pname, p in signature.parameters.items() if p.kind in keyword_kinds}
    missing = _protocol_invoke_kwargs(narrow) - accepted
    if missing:
        raise AssertionError(
            f"{name!r}.invoke is missing keyword parameter(s) "
            f"{sorted(missing)} required by {narrow.__name__}; actual signature is {signature!s}"
        )


def _protocol_invoke_kwargs(protocol: type) -> set[str]:
    """Return the set of keyword parameter names ``protocol.invoke`` declares.

    Only parameters that can be passed by keyword (``KEYWORD_ONLY`` and
    ``POSITIONAL_OR_KEYWORD``) are returned. ``self`` is excluded. The
    result drives the kwarg-compatibility check in
    :func:`assert_capability_conforms`: every name returned here must
    either appear by name in the capability's ``invoke`` signature or
    be absorbed by a ``**kwargs``.
    """
    try:
        sig = inspect.signature(protocol.invoke)
    except (TypeError, ValueError):
        return set()
    return {
        pname
        for pname, p in sig.parameters.items()
        if pname != "self"
        and p.kind
        in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    }


__all__ = ["assert_capability_conforms"]

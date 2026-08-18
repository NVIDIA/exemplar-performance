# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-capability narrow :class:`typing.Protocol` classes.

These focus on the runtime ``isinstance`` filter behavior. Signature-
level enforcement (kwargs, return type) is checked at static-typing
time and by :func:`llmb_capabilities.conformance.assert_capability_conforms`.
"""

from __future__ import annotations

import argparse
import ssl
from collections.abc import Mapping
from importlib.resources.abc import Traversable
from typing import get_args

from llmb_capabilities import (
    BEARER_TOKEN,
    OIDC_ID_TOKEN,
    POST_PROCESSOR_PROCESS,
    STORAGE_UPLOAD,
    SYSTEM_COLLECT,
    SYSTEM_COMMANDS_CONFIG,
    TLS_SSL_CONTEXT,
    BearerTokenCapability,
    BearerTokenMethod,
    Capability,
    OidcIdTokenCapability,
    PostProcessorProcessCapability,
    StorageUploadCapability,
    SystemCollectCapability,
    SystemCommandsConfigCapability,
    TlsSslContextCapability,
)


class _FakeCliArgs:
    @classmethod
    def model_validate(cls, obj, /):
        return cls()


def _bearer_invoke(
    namespace: argparse.Namespace,
    /,
    *,
    prefix: str | None = None,
    method: str = "ssa",
    force: bool = False,
) -> str:
    return "fake-token"


def _oidc_invoke(
    namespace: argparse.Namespace,
    /,
    *,
    prefix: str | None = None,
    force: bool = False,
) -> str:
    return "fake.id.token"


def _tls_invoke(
    namespace: argparse.Namespace,
    /,
    *,
    prefix: str | None = None,
) -> ssl.SSLContext:
    return ssl.create_default_context()


def _collect_invoke(
    namespace: argparse.Namespace,
    /,
    *,
    prefix: str | None = None,
    config: object = None,
) -> dict:
    return {}


def _commands_config_invoke(
    namespace: argparse.Namespace,
    /,
    *,
    prefix: str | None = None,
) -> Traversable:
    raise NotImplementedError  # not exercised at runtime in this test


def _storage_upload_invoke(cli_args: _FakeCliArgs, /) -> dict:
    return {}


def _post_processor_process_invoke(cli_args: _FakeCliArgs, /) -> dict:
    return {}


class TestBearerTokenProtocol:
    def test_method_names_match_supported_auth_flows(self):
        assert get_args(BearerTokenMethod) == ("ssa", "auth-code", "device-code")

    def test_dataclass_satisfies(self):
        cap = Capability(
            name=BEARER_TOKEN,
            version=1,
            invoke=_bearer_invoke,
            metadata={"provider": "test"},
        )
        assert isinstance(cap, BearerTokenCapability)

    def test_missing_metadata_fails(self):
        class Bad:
            name = BEARER_TOKEN
            version = 1

            def invoke(self, *args: object, **kwargs: object) -> str:
                return "x"

        # Protocol requires metadata at the attribute level.
        assert not isinstance(Bad(), BearerTokenCapability)


class TestOidcIdTokenProtocol:
    def test_dataclass_satisfies(self):
        cap = Capability(
            name=OIDC_ID_TOKEN,
            version=1,
            invoke=_oidc_invoke,
            metadata={"provider": "test"},
        )
        assert isinstance(cap, OidcIdTokenCapability)

    def test_missing_metadata_fails(self):
        # Protocol requires metadata at the attribute level, same as
        # BearerToken. A stub without it must fail the runtime check.
        class Bad:
            name = OIDC_ID_TOKEN
            version = 1

            def invoke(self, *args: object, **kwargs: object) -> str:
                return "x"

        assert not isinstance(Bad(), OidcIdTokenCapability)


class TestTlsSslContextProtocol:
    def test_dataclass_satisfies(self):
        cap = Capability(
            name=TLS_SSL_CONTEXT,
            version=1,
            invoke=_tls_invoke,
            metadata={"provider": "test"},
        )
        assert isinstance(cap, TlsSslContextCapability)


class TestSystemCollectProtocol:
    def test_dataclass_satisfies(self):
        cap = Capability(
            name=SYSTEM_COLLECT,
            version=1,
            invoke=_collect_invoke,
            metadata={"provider": "test"},
        )
        assert isinstance(cap, SystemCollectCapability)


class TestSystemCommandsConfigProtocol:
    def test_dataclass_satisfies(self):
        cap = Capability(
            name=SYSTEM_COMMANDS_CONFIG,
            version=1,
            invoke=_commands_config_invoke,
            metadata={"provider": "test"},
        )
        assert isinstance(cap, SystemCommandsConfigCapability)


class TestStorageUploadProtocol:
    def test_dataclass_satisfies(self):
        cap = Capability(
            name=STORAGE_UPLOAD,
            version=1,
            invoke=_storage_upload_invoke,
            metadata={"provider": "test"},
            args_model=_FakeCliArgs,
        )
        assert isinstance(cap, StorageUploadCapability)

    def test_satisfies_even_with_args_model_none(self):
        # isinstance only checks attribute presence, not value - a None
        # args_model still structurally satisfies the Protocol.
        # assert_capability_conforms (not isinstance) is what actually
        # enforces args_model being set for a model-first capability.
        cap = Capability(
            name=STORAGE_UPLOAD,
            version=1,
            invoke=_storage_upload_invoke,
            metadata={"provider": "test"},
        )
        assert isinstance(cap, StorageUploadCapability)


class TestPostProcessorProcessProtocol:
    def test_dataclass_satisfies(self):
        cap = Capability(
            name=POST_PROCESSOR_PROCESS,
            version=1,
            invoke=_post_processor_process_invoke,
            metadata={"provider": "test"},
            args_model=_FakeCliArgs,
        )
        assert isinstance(cap, PostProcessorProcessCapability)


class TestStructuralStrangers:
    """Unrelated objects should fail the narrow checks too."""

    def test_plain_object(self):
        # Sanity: a bare object satisfies neither shape.
        obj = object()
        assert not isinstance(obj, BearerTokenCapability)
        assert not isinstance(obj, OidcIdTokenCapability)
        assert not isinstance(obj, TlsSslContextCapability)
        assert not isinstance(obj, SystemCollectCapability)
        assert not isinstance(obj, SystemCommandsConfigCapability)

    def test_metadata_typed_via_mapping(self):
        # Ensure a custom Mapping[str, str] is also accepted.
        from types import MappingProxyType

        meta: Mapping[str, str] = MappingProxyType({"provider": "test"})
        cap = Capability(name=BEARER_TOKEN, version=1, invoke=_bearer_invoke, metadata=meta)
        assert isinstance(cap, BearerTokenCapability)

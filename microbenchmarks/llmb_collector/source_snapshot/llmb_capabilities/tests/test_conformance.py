# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`llmb_capabilities.conformance.assert_capability_conforms`.

Each invalid-shape case is exercised with a minimal stub so the
helper's diagnostic messages stay specific. The valid case mirrors what
the reference producers (:mod:`llmb_auth`, :mod:`llmb_collector`) build
today.
"""

from __future__ import annotations

import argparse
import ssl
from collections.abc import Mapping
from importlib.resources.abc import Traversable

import pytest

from llmb_capabilities import (
    BEARER_TOKEN,
    OIDC_ID_TOKEN,
    POST_PROCESSOR_PROCESS,
    STORAGE_UPLOAD,
    SYSTEM_COLLECT,
    SYSTEM_COMMANDS_CONFIG,
    TLS_SSL_CONTEXT,
    Capability,
)
from llmb_capabilities.conformance import assert_capability_conforms


class _FakeCliArgs:
    @classmethod
    def model_validate(cls, obj, /):
        return cls()


def _storage_upload_invoke(cli_args: _FakeCliArgs, /) -> dict:
    return {}


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
    raise NotImplementedError


def _bearer_capability(**overrides: object) -> Capability:
    base: dict[str, object] = {
        "name": BEARER_TOKEN,
        "version": 1,
        "invoke": _bearer_invoke,
        "metadata": {"provider": "test"},
    }
    base.update(overrides)
    return Capability(**base)  # type: ignore[arg-type]


def _storage_upload_capability(**overrides: object) -> Capability:
    base: dict[str, object] = {
        "name": STORAGE_UPLOAD,
        "version": 1,
        "invoke": _storage_upload_invoke,
        "metadata": {"provider": "test"},
        "args_model": _FakeCliArgs,
    }
    base.update(overrides)
    return Capability(**base)  # type: ignore[arg-type]


class TestValidCases:
    def test_bearer_token_conforms(self):
        assert_capability_conforms(_bearer_capability())

    def test_oidc_id_token_conforms(self):
        cap = Capability(
            name=OIDC_ID_TOKEN,
            version=1,
            invoke=_oidc_invoke,
            metadata={"provider": "test"},
        )
        assert_capability_conforms(cap)

    def test_tls_ssl_context_conforms(self):
        cap = Capability(
            name=TLS_SSL_CONTEXT,
            version=1,
            invoke=_tls_invoke,
            metadata={"provider": "test"},
        )
        assert_capability_conforms(cap)

    def test_system_collect_conforms(self):
        cap = Capability(
            name=SYSTEM_COLLECT,
            version=1,
            invoke=_collect_invoke,
            metadata={"provider": "test"},
        )
        assert_capability_conforms(cap)

    def test_system_commands_config_conforms(self):
        cap = Capability(
            name=SYSTEM_COMMANDS_CONFIG,
            version=1,
            invoke=_commands_config_invoke,
            metadata={"provider": "test"},
        )
        assert_capability_conforms(cap)

    def test_extra_metadata_keys_allowed(self):
        cap = _bearer_capability(metadata={"provider": "test", "extra": "value"})
        assert_capability_conforms(cap)

    def test_storage_upload_conforms(self):
        cap = Capability(
            name=STORAGE_UPLOAD,
            version=1,
            invoke=_storage_upload_invoke,
            metadata={"provider": "test"},
            args_model=_FakeCliArgs,
        )
        assert_capability_conforms(cap)

    def test_post_processor_process_conforms(self):
        cap = Capability(
            name=POST_PROCESSOR_PROCESS,
            version=1,
            invoke=_storage_upload_invoke,
            metadata={"provider": "test"},
            args_model=_FakeCliArgs,
        )
        assert_capability_conforms(cap)


class TestModelFirstRequiresArgsModel:
    def test_storage_upload_without_args_model_fails(self):
        cap = Capability(
            name=STORAGE_UPLOAD,
            version=1,
            invoke=_storage_upload_invoke,
            metadata={"provider": "test"},
        )
        with pytest.raises(AssertionError, match="model-first and must set args_model"):
            assert_capability_conforms(cap)

    def test_post_processor_process_without_args_model_fails(self):
        cap = Capability(
            name=POST_PROCESSOR_PROCESS,
            version=1,
            invoke=_storage_upload_invoke,
            metadata={"provider": "test"},
        )
        with pytest.raises(AssertionError, match="model-first and must set args_model"):
            assert_capability_conforms(cap)

    def test_namespace_first_capability_unaffected(self):
        # Bearer-token is Namespace-first; args_model staying None must
        # not be flagged.
        assert_capability_conforms(_bearer_capability())

    def test_args_model_without_model_validate_fails(self):
        # A non-None args_model that doesn't actually expose
        # model_validate would otherwise pass this check and blow up
        # later with a raw AttributeError inside resolve_capability_args.
        class NotAModel:
            pass

        cap = Capability(
            name=STORAGE_UPLOAD,
            version=1,
            invoke=_storage_upload_invoke,
            metadata={"provider": "test"},
            args_model=NotAModel,
        )
        with pytest.raises(AssertionError, match="must expose a callable 'model_validate'"):
            assert_capability_conforms(cap)


class TestInvalidShape:
    def test_object_without_required_attrs(self):
        with pytest.raises(AssertionError, match="missing required Capability attributes"):
            assert_capability_conforms(object())

    def test_dict_is_rejected(self):
        # A bare dict has no `.name` / `.version` attributes.
        with pytest.raises(AssertionError, match="missing required Capability attributes"):
            assert_capability_conforms({"name": BEARER_TOKEN, "version": 1})


class TestUnknownName:
    def test_rejects_name_not_in_supported(self):
        cap = _bearer_capability(name="unknown.capability")
        with pytest.raises(AssertionError, match="not a known capability name"):
            assert_capability_conforms(cap)


class TestVersionCeiling:
    def test_rejects_version_above_ceiling(self):
        cap = _bearer_capability(version=99)
        with pytest.raises(AssertionError, match="exceeds contract ceiling"):
            assert_capability_conforms(cap)

    def test_accepts_lower_version(self):
        # If the ceiling were ever bumped, an older v1 capability is
        # still considered conforming. With ceiling currently == 1 this
        # is the same as the boundary case but documents the intent.
        cap = _bearer_capability(version=1)
        assert_capability_conforms(cap)

    def test_rejects_non_int_version(self):
        # CapabilityLike's isinstance check only verifies `version` is
        # present, not its type; a str would otherwise reach the `>`
        # comparison and raise a bare TypeError instead of AssertionError.
        cap = _bearer_capability(version="1")
        with pytest.raises(AssertionError, match="version must be an int"):
            assert_capability_conforms(cap)


class TestProviderKey:
    def test_rejects_missing_provider(self):
        cap = _bearer_capability(metadata={})
        with pytest.raises(AssertionError, match="missing required 'provider' key"):
            assert_capability_conforms(cap)

    def test_rejects_provider_under_wrong_key(self):
        cap = _bearer_capability(metadata={"vendor": "test"})
        with pytest.raises(AssertionError, match="missing required 'provider' key"):
            assert_capability_conforms(cap)

    def test_rejects_non_mapping_metadata(self):
        # Same rationale as test_rejects_non_int_version: a None/non-Mapping
        # metadata would otherwise raise a bare TypeError from the `in` check.
        cap = _bearer_capability(metadata=None)
        with pytest.raises(AssertionError, match="metadata must be a Mapping"):
            assert_capability_conforms(cap)


class TestNarrowProtocolMismatch:
    def test_rejects_when_invoke_lacks_positional(self):
        # `invoke` with no positional violates the narrow protocol's
        # documented "(namespace, /, ...)" signature; the helper's
        # final positional-count check catches it.
        def bad_invoke(*, prefix: str | None = None) -> str:
            return ""

        # Build via __setattr__ shenanigans: the dataclass would happily
        # accept any callable, so this exercises the helper's own
        # introspection.
        cap = _bearer_capability(invoke=bad_invoke)
        # Wording covers both Namespace-first and model-first capabilities
        # (the positional isn't always an argparse.Namespace).
        with pytest.raises(AssertionError, match="argparse.Namespace or args_model instance"):
            assert_capability_conforms(cap)

    def test_rejects_when_invoke_drops_documented_kwarg(self):
        # Runtime-checkable Protocol's `isinstance` only verifies
        # attribute presence, not the `invoke` signature. The helper
        # must also catch a producer that silently drops a documented
        # kwarg (here `force`, declared on BearerTokenCapability.invoke
        # but missing from the producer's signature).
        def missing_kwarg_invoke(
            namespace: argparse.Namespace,
            /,
            *,
            prefix: str | None = None,
            method: str = "ssa",
        ) -> str:
            return "fake-token"

        cap = _bearer_capability(invoke=missing_kwarg_invoke)
        with pytest.raises(AssertionError, match=r"missing keyword parameter.*'force'"):
            assert_capability_conforms(cap)

    def test_accepts_invoke_with_var_keyword(self):
        # A producer that absorbs all kwargs via **kwargs is treated as
        # accepting every documented kwarg. This keeps the door open
        # for thin wrappers that forward to a real implementation.
        def var_keyword_invoke(
            namespace: argparse.Namespace,
            /,
            **kwargs: object,
        ) -> str:
            return "fake-token"

        cap = _bearer_capability(invoke=var_keyword_invoke)
        assert_capability_conforms(cap)

    def test_rejects_model_first_invoke_lacking_positional(self):
        # Same check as test_rejects_when_invoke_lacks_positional, but for
        # a model-first capability (STORAGE_UPLOAD) rather than a
        # Namespace-first one: the missing positional here would have
        # been an args_model instance, not an argparse.Namespace, but
        # the helper's final positional-count check is shared code and
        # must catch both shapes identically.
        def bad_invoke() -> dict:
            return {}

        cap = _storage_upload_capability(invoke=bad_invoke)
        with pytest.raises(AssertionError, match="argparse.Namespace or args_model instance"):
            assert_capability_conforms(cap)

    def test_rejects_post_processor_process_invoke_lacking_positional(self):
        # Same as test_rejects_model_first_invoke_lacking_positional, for
        # the other model-first capability - both share args_model=_FakeCliArgs
        # and PostProcessorProcessCapability's identical "(cli_args, /)" shape.
        # This confirms POST_PROCESSOR_PROCESS is registered and the shared
        # positional-count guard runs for it too; it does not, on its own,
        # prove the two capabilities are registered against distinct
        # protocols (a zero-positional bad_invoke fails this check the same
        # way regardless of which protocol backs either name).
        def bad_invoke() -> dict:
            return {}

        cap = _storage_upload_capability(name=POST_PROCESSOR_PROCESS, invoke=bad_invoke)
        with pytest.raises(AssertionError, match="argparse.Namespace or args_model instance"):
            assert_capability_conforms(cap)


class TestNonIntrospectableInvoke:
    def test_handles_non_introspectable_invoke(self):
        # A non-callable, non-introspectable `invoke` value (here a
        # bare `object()`) must be surfaced as a clean AssertionError
        # from `inspect.signature`'s `TypeError`, not propagated as a
        # bare TypeError. This keeps producer test failures uniformly
        # AssertionError so pytest reports them as test failures
        # rather than errors.
        class Stub:
            name = BEARER_TOKEN
            version = 1
            metadata: Mapping[str, str] = {"provider": "test"}
            invoke = staticmethod(object())  # type: ignore[assignment]

        with pytest.raises(AssertionError):
            assert_capability_conforms(Stub())

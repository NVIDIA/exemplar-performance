# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the canonical :class:`Capability` dataclass and structural Protocol."""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Mapping

import pytest

from llmb_capabilities import Capability, CapabilityLike, ModelValidatable


def _noop_invoke(namespace: argparse.Namespace, /, **kwargs: object) -> None:
    return None


class TestCapabilityDataclass:
    def test_minimal_construction(self):
        cap = Capability(name="x", version=1, invoke=_noop_invoke)
        assert cap.name == "x"
        assert cap.version == 1
        assert cap.invoke is _noop_invoke
        assert cap.add_arguments is None

    def test_default_metadata_is_empty(self):
        cap = Capability(name="x", version=1, invoke=_noop_invoke)
        assert dict(cap.metadata) == {}

    def test_default_metadata_is_immutable(self):
        # The default factory returns a MappingProxyType so two
        # capabilities can't accidentally share a mutable default.
        cap = Capability(name="x", version=1, invoke=_noop_invoke)
        with pytest.raises(TypeError):
            cap.metadata["provider"] = "foo"  # type: ignore[index]

    def test_metadata_round_trip(self):
        meta = {"provider": "test", "extra": "value"}
        cap = Capability(name="x", version=1, invoke=_noop_invoke, metadata=meta)
        assert cap.metadata["provider"] == "test"
        assert cap.metadata["extra"] == "value"

    def test_is_frozen(self):
        cap = Capability(name="x", version=1, invoke=_noop_invoke)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cap.name = "y"  # type: ignore[misc]

    def test_args_model_defaults_to_none(self):
        # None means Namespace-first: unaffected by resolve_capability_args.
        cap = Capability(name="x", version=1, invoke=_noop_invoke)
        assert cap.args_model is None

    def test_args_model_round_trip(self):
        class FakeModel:
            @classmethod
            def model_validate(cls, obj, /):
                return cls()

        cap = Capability(name="x", version=1, invoke=_noop_invoke, args_model=FakeModel)
        assert cap.args_model is FakeModel
        assert isinstance(cap.args_model, type)
        assert issubclass(cap.args_model, ModelValidatable)


class TestCapabilityLikeProtocol:
    def test_dataclass_satisfies_protocol(self):
        cap = Capability(name="x", version=1, invoke=_noop_invoke)
        assert isinstance(cap, CapabilityLike)

    def test_custom_class_satisfies_protocol(self):
        class CustomCapability:
            name = "y"
            version = 2
            metadata: Mapping[str, str] = {"provider": "custom"}
            args_model = None

            def invoke(self, *args: object, **kwargs: object) -> str:
                return "ok"

        assert isinstance(CustomCapability(), CapabilityLike)

    def test_missing_invoke_fails(self):
        class NoInvoke:
            name = "y"
            version = 2
            metadata: Mapping[str, str] = {}

        assert not isinstance(NoInvoke(), CapabilityLike)

    def test_missing_metadata_fails(self):
        class NoMetadata:
            name = "y"
            version = 2

            def invoke(self, *args: object, **kwargs: object) -> None:
                return None

        assert not isinstance(NoMetadata(), CapabilityLike)

    def test_plain_object_fails(self):
        assert not isinstance(object(), CapabilityLike)

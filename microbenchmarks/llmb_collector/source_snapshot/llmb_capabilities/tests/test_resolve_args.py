# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`llmb_capabilities.resolve_capability_args`."""

from __future__ import annotations

import argparse

from llmb_capabilities import STORAGE_UPLOAD, Capability, resolve_capability_args


class _FakeCliArgs:
    def __init__(self, **fields: object) -> None:
        self.fields = fields

    @classmethod
    def model_validate(cls, obj: dict, /) -> _FakeCliArgs:
        return cls(**obj)


def _invoke(cli_args: _FakeCliArgs, /) -> None:
    return None


def _capability(**overrides: object) -> Capability:
    base: dict[str, object] = {
        "name": STORAGE_UPLOAD,
        "version": 1,
        "invoke": _invoke,
        "metadata": {"provider": "test"},
        "args_model": _FakeCliArgs,
    }
    base.update(overrides)
    return Capability(**base)  # type: ignore[arg-type]


class TestNamespaceFirstCapability:
    def test_returns_none_when_args_model_is_none(self):
        cap = _capability(args_model=None)
        namespace = argparse.Namespace(local_path="results")

        assert resolve_capability_args(cap, namespace) is None


class TestModelFirstCapability:
    def test_builds_model_from_namespace(self):
        cap = _capability()
        namespace = argparse.Namespace(local_path="results", s3_bucket="my-bucket")

        cli_args = resolve_capability_args(cap, namespace)

        assert isinstance(cli_args, _FakeCliArgs)
        assert cli_args.fields == {"local_path": "results", "s3_bucket": "my-bucket"}

    def test_strips_prefix_from_every_key(self):
        cap = _capability()
        namespace = argparse.Namespace(upload_local_path="results", upload_s3_bucket="my-bucket")

        cli_args = resolve_capability_args(cap, namespace, prefix="upload")

        assert cli_args.fields == {"local_path": "results", "s3_bucket": "my-bucket"}

    def test_unprefixed_keys_are_dropped_when_prefix_given(self):
        # Only keys that actually start with f"{prefix}_" are kept (and
        # stripped); everything else is dropped rather than forwarded,
        # so an args_model with extra="forbid" doesn't spuriously fail
        # on unrelated fields from a shared Namespace.
        cap = _capability()
        namespace = argparse.Namespace(upload_local_path="results", unrelated_field="x")

        cli_args = resolve_capability_args(cap, namespace, prefix="upload")

        assert cli_args.fields == {"local_path": "results"}

    def test_calls_model_validate_exactly_once(self):
        calls: list[dict] = []

        class TrackingModel(_FakeCliArgs):
            @classmethod
            def model_validate(cls, obj, /):
                calls.append(obj)
                return super().model_validate(obj)

        cap = _capability(args_model=TrackingModel)
        namespace = argparse.Namespace(local_path="results")

        resolve_capability_args(cap, namespace)

        assert len(calls) == 1

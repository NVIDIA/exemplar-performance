# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow protocol for the ``storage.upload`` capability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from llmb_capabilities._capability import ModelValidatable


@runtime_checkable
class StorageUploadCapability(Protocol):
    """A provider that uploads a local results archive to a storage backend.

    This is a **model-first** capability, unlike
    :class:`BearerTokenCapability` / :class:`SystemCollectCapability`:
    :meth:`invoke` takes an already-built :attr:`args_model` instance
    directly rather than an :class:`argparse.Namespace`. A storage
    upload's CLI surface (destination selection, credentials,
    compression, metadata, ...) is large and structured enough that
    re-deriving the model inside every producer's ``invoke`` would
    duplicate the boundary logic
    :func:`llmb_capabilities.resolve_capability_args` centralizes
    instead. Callers build the model exactly once and pass the same
    instance to :meth:`invoke`.
    """

    name: str
    version: int
    metadata: Mapping[str, str]
    args_model: type[ModelValidatable]

    def invoke(self, cli_args: ModelValidatable, /) -> Any:
        """Perform the upload described by ``cli_args``.

        :param cli_args: an instance of :attr:`args_model`, built once
            by the caller (e.g. via
            :func:`llmb_capabilities.resolve_capability_args`) and
            passed unchanged - producers must not re-derive it from a
            namespace internally.
        :returns: a producer-defined result. May be a coroutine;
            callers should use :func:`llmb_capabilities.invoke_sync` /
            :func:`llmb_capabilities.invoke_async` rather than calling
            ``invoke`` directly unless they already know which shape
            the producer implements.
        """
        ...


__all__ = ["StorageUploadCapability"]

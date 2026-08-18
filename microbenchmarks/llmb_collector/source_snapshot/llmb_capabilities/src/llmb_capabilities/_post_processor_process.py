# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow protocol for the ``post-processor.process`` capability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from llmb_capabilities._capability import ModelValidatable


@runtime_checkable
class PostProcessorProcessCapability(Protocol):
    """A provider that triggers post-processing on a prior upload submission.

    Model-first, for the same reason as :class:`StorageUploadCapability`:
    :meth:`invoke` takes an already-built :attr:`args_model` instance
    directly rather than an :class:`argparse.Namespace`. Distinct from
    :data:`STORAGE_UPLOAD` because post-processing is triggered after an
    upload's presigned submission completes and may be invoked on its
    own (e.g. re-triggering processing for an existing submission)
    without repeating the upload itself.
    """

    name: str
    version: int
    metadata: Mapping[str, str]
    args_model: type[ModelValidatable]

    def invoke(self, cli_args: ModelValidatable, /) -> Any:
        """Trigger post-processing described by ``cli_args``.

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


__all__ = ["PostProcessorProcessCapability"]

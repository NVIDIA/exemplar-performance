# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic ``argparse.Namespace`` -> :attr:`Capability.args_model` translation.

Model-first capabilities (:data:`llmb_capabilities.STORAGE_UPLOAD`,
:data:`llmb_capabilities.POST_PROCESSOR_PROCESS`) take an already-built
model instance in ``invoke``, rather than deriving one from a
:class:`argparse.Namespace` internally the way Namespace-first
capabilities do. This module is the one place that translation happens,
so a generic caller (e.g. a future capabilities-based dispatcher) never
needs to import a producer's concrete model class.
"""

from __future__ import annotations

import argparse
from typing import Any

from llmb_capabilities._capability import CapabilityLike


def resolve_capability_args(
    capability: CapabilityLike,
    namespace: argparse.Namespace,
    *,
    prefix: str | None = None,
) -> Any | None:
    """Build ``capability.args_model`` from ``namespace``, or ``None``.

    Returns ``None`` when ``capability.args_model`` is ``None`` -
    Namespace-first capabilities are unaffected; call ``invoke`` with
    ``namespace`` directly as documented on their narrow ``Protocol``.

    Otherwise, takes ``vars(namespace)``, and when ``prefix`` is given
    keeps *only* the keys that start with ``f"{prefix}_"`` (stripping
    that prefix off each), mirroring how ``prefix`` already works for
    :class:`BearerTokenCapability` / :class:`SystemCollectCapability`.
    A shared ``Namespace`` typically carries fields from unrelated
    argument groups; forwarding those unprefixed keys to
    ``model_validate`` would spuriously fail any ``args_model`` with
    ``extra="forbid"``, so they are dropped rather than passed through.
    Calls ``capability.args_model.model_validate(...)`` exactly once.

    :param capability: any object satisfying :class:`CapabilityLike`.
    :param namespace: the parsed CLI arguments.
    :param prefix: optional flag-prefix used when the capability's
        arguments were embedded under a namespaced argparse section.
        ``None`` means the namespace keys already match the model's
        field names.
    :returns: an instance of ``capability.args_model``, or ``None``.
    """
    args_model = capability.args_model
    if args_model is None:
        return None

    data = vars(namespace)
    if prefix:
        strip = f"{prefix}_"
        data = {key.removeprefix(strip): value for key, value in data.items() if key.startswith(strip)}
    return args_model.model_validate(data)


__all__ = ["resolve_capability_args"]

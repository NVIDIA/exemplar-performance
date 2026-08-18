# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The canonical :class:`Capability` shape advertised via the entry-point group.

Today both :mod:`llmb_auth.capabilities` and
:mod:`llmb_collector.capabilities` define this dataclass themselves. Both
copies are structurally identical; this module is intended to be the
single home so a future shape change is one edit, not two.

Two layers are exposed deliberately:

* :class:`Capability` — the recommended construction class. Frozen,
  hashable-by-identity, validated by ``dataclasses``. Producers build
  instances of this and put them in their ``CAPABILITIES`` tuple.

* :class:`CapabilityLike` — a structural :class:`typing.Protocol` for
  consumers. Lets a caller treat any object exposing the right
  attributes as a capability without forcing inheritance from this
  package. ``@runtime_checkable`` so consumers can use
  ``isinstance(obj, CapabilityLike)`` as a defensive filter on
  third-party providers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


def _empty_metadata() -> Mapping[str, str]:
    """Default factory returning an immutable empty mapping.

    A factory is required because mutable defaults (``{}``) are
    forbidden in :func:`dataclasses.dataclass` fields. Using a
    :class:`types.MappingProxyType` here, rather than a fresh
    :class:`dict`, makes accidental mutation of the default an
    immediate :class:`TypeError` instead of a silent shared-state bug.
    """
    return MappingProxyType({})


@runtime_checkable
class ModelValidatable(Protocol):
    """Structural shape of a pydantic-like model class.

    This package is stdlib-only (see ``pyproject.toml``), so
    :attr:`Capability.args_model` can't be typed as ``type[BaseModel]``
    without taking a runtime dependency on pydantic. Any pydantic
    ``BaseModel`` subclass already satisfies this shape via its
    ``model_validate`` classmethod, so producers pass their concrete
    model class straight through with no adapter needed.
    """

    @classmethod
    def model_validate(cls, obj: Any, /) -> Any: ...


@dataclass(frozen=True, eq=False)
class Capability:
    """One named, versioned capability a producer advertises.

    The ``invoke`` and ``add_arguments`` callable signatures are fixed
    *per* :attr:`name` (see the per-capability :class:`typing.Protocol`
    classes in this package), not by this dataclass. The dataclass only
    constrains the advertisement envelope: name, schema version,
    callables, and metadata.

    :attr:`metadata` is treated as immutable by consumers. Producers
    that want to expose values that should not be computed at import
    time (e.g. paths inside a zipapp) should pass a
    :class:`types.MappingProxyType` over a custom :class:`Mapping`
    rather than a plain :class:`dict`.

    ``eq=False`` is deliberate: a ``Capability`` is a unique
    advertisement object, not a value type. Two instances built from
    structurally identical inputs are distinct advertisements that may
    have been produced by different providers, registered at different
    times, or wrap different ``invoke`` closures — so equality falls
    back to object identity and hashing is identity-based (inherited
    from :class:`object`). Consumers that want to dedupe should key on
    ``(cap.name, cap.metadata["provider"])`` explicitly rather than
    relying on ``==``.
    """

    name: str
    version: int
    invoke: Callable[..., Any]
    add_arguments: Callable[..., Any] | None = None
    metadata: Mapping[str, str] = field(default_factory=_empty_metadata)
    #: Optional model class ``invoke`` expects a pre-built instance of.
    #: ``None`` (the default) means this is a Namespace-first capability
    #: (``invoke`` takes an :class:`argparse.Namespace` directly, as
    #: :class:`BearerTokenCapability` / :class:`SystemCollectCapability` do).
    #: Set this on model-first capabilities (e.g. :data:`STORAGE_UPLOAD`)
    #: so :func:`llmb_capabilities.resolve_capability_args` can build the
    #: model from a namespace without the caller importing the producer's
    #: concrete model class.
    args_model: type[ModelValidatable] | None = None


@runtime_checkable
class CapabilityLike(Protocol):
    """Structural shape any consumer can rely on.

    Defined separately from :class:`Capability` so a third-party
    producer that doesn't want to depend on this package's dataclass
    (e.g. for vendoring reasons) can still satisfy the contract by
    exposing the same attributes.

    Note: ``@runtime_checkable`` :class:`typing.Protocol` checks
    *attribute presence*, not callable signatures. To verify the
    ``invoke`` shape per capability name, use the narrow protocols in
    :mod:`llmb_capabilities` or :func:`llmb_capabilities.conformance.assert_capability_conforms`.
    """

    name: str
    version: int
    metadata: Mapping[str, str]
    args_model: type[ModelValidatable] | None

    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = ["Capability", "CapabilityLike", "ModelValidatable"]

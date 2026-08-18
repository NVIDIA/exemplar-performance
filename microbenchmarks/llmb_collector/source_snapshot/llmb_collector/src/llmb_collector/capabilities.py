# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic capability registry advertised via the ``llmb_capabilities`` entry-point group.

A consumer (uploader, downloader, debug tool, ...) discovers capabilities with
``importlib.metadata.entry_points(group="llmb_capabilities")``, filters by
``Capability.name``, then ``invoke``s it. Neither side imports the other; the
contract is the group name, the :class:`Capability` shape, and each
capability's documented ``invoke`` signature. See the README "Capability
registry" section.

``Capability`` and the two capability names below come from the
`llmb-capabilities <https://gitlab-master.nvidia.com/unified-cloud-benchmarking/sanitation-tools/llmb-capabilities>`_
contract package rather than being defined/duplicated locally, so every
producer and consumer shares one definition of the entry-point group name,
the ``Capability`` shape, and each capability's supported ``version`` ceiling.
"""

from __future__ import annotations

import argparse
import types
from collections.abc import Callable, Iterator, Mapping
from importlib.resources.abc import Traversable
from typing import Final

from llmb_capabilities import SYSTEM_COLLECT as _SYSTEM_COLLECT_NAME
from llmb_capabilities import SYSTEM_COMMANDS_CONFIG as _SYSTEM_COMMANDS_CONFIG_NAME
from llmb_capabilities import Capability

from llmb_collector.collect import CollectConfig, collect
from llmb_collector.command_loader import get_packaged_commands_config

_PROVIDER: Final[str] = "llmb-collector"


class _LazyImmutableMapping(Mapping[str, str]):
    """Read-only ``str`` → ``str`` :class:`Mapping` with optional lazily-computed keys.

    Used so :class:`Capability` metadata can advertise values that should not
    be resolved at import time. The motivating example is the
    ``commands_config_path`` key on :data:`SYSTEM_COMMANDS_CONFIG`: ``str()`` on
    an :class:`importlib.resources.abc.Traversable` is fine for directory-based
    installs but yields a virtual path inside the archive for zipapp /
    zip-archived-wheel deployments, which is not a usable filesystem path.
    Deferring the call lets consumers that never read the key skip the cost
    entirely, and keeps the import side-effect-free.

    Lazy values are memoized on first access. The class deliberately omits
    ``__setitem__`` / ``__delitem__`` so callers cannot mutate the visible
    mapping at runtime.
    """

    __slots__ = ("_eager", "_lazy", "_cache")

    def __init__(
        self,
        eager: Mapping[str, str],
        lazy: Mapping[str, Callable[[], str]] | None = None,
    ) -> None:
        self._eager: Mapping[str, str] = types.MappingProxyType(dict(eager))
        self._lazy: Mapping[str, Callable[[], str]] = types.MappingProxyType(dict(lazy or {}))
        self._cache: dict[str, str] = {}

    def __getitem__(self, key: str) -> str:
        if key in self._eager:
            return self._eager[key]
        if key in self._lazy:
            cached = self._cache.get(key)
            if cached is None:
                cached = self._lazy[key]()
                self._cache[key] = cached
            return cached
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from self._eager
        yield from self._lazy

    def __len__(self) -> int:
        return len(self._eager) + len(self._lazy)


def _invoke_system_collect(
    namespace: argparse.Namespace,  # noqa: ARG001 - signature uniform with other capabilities
    *,
    prefix: str | None = None,  # noqa: ARG001 - reserved for future add_arguments support
    config: CollectConfig | None = None,
) -> dict:
    """Run a collection and return the JSON-serializable dict.

    When ``config`` is ``None``, all four realms (host, network, container,
    env) are enabled. Note this **overrides** :class:`CollectConfig`'s own
    built-in default for ``container``, which is ``False``: invoking with
    ``config=None`` is *not* equivalent to invoking with
    ``config=CollectConfig()`` — the former opts every realm in, the latter
    opts out of container. Pass an explicit :class:`CollectConfig` to narrow
    the scope or tweak knobs (e.g. ``redact_env=False``).
    """
    if config is None:
        # ``CollectConfig`` already defaults host / network / env to True; only
        # ``container`` needs to be flipped to honour the capability-level
        # "all on" promise. Writing only the divergent field keeps the
        # override explicit if the class's defaults change in the future.
        config = CollectConfig(container=True)
    return collect(config)


def _invoke_system_commands_config(
    namespace: argparse.Namespace,  # noqa: ARG001 - signature uniform with other capabilities
    *,
    prefix: str | None = None,  # noqa: ARG001 - reserved for future add_arguments support
) -> Traversable:
    """Return the packaged commands_config tree as a :class:`Traversable`.

    Consumers that just want the on-disk path can read
    ``metadata["commands_config_path"]`` without invoking. That string is only
    a usable filesystem path for directory-based installs — for zipapp or
    zip-archived wheel deployments it points inside the archive and must not
    be passed to ``open()`` / ``Path(...).is_dir()`` directly. Use the
    :class:`Traversable` returned here when portability matters.
    """
    return get_packaged_commands_config()


SYSTEM_COLLECT: Final[Capability] = Capability(
    name=_SYSTEM_COLLECT_NAME,
    version=1,
    invoke=_invoke_system_collect,
    add_arguments=None,
    metadata=types.MappingProxyType({"provider": _PROVIDER}),
)

SYSTEM_COMMANDS_CONFIG: Final[Capability] = Capability(
    name=_SYSTEM_COMMANDS_CONFIG_NAME,
    version=1,
    invoke=_invoke_system_commands_config,
    add_arguments=None,
    metadata=_LazyImmutableMapping(
        eager={"provider": _PROVIDER},
        lazy={"commands_config_path": lambda: str(get_packaged_commands_config())},
    ),
)

#: Advertised as ``llmb_capabilities:default`` in :file:`pyproject.toml`.
CAPABILITIES: Final[tuple[Capability, ...]] = (SYSTEM_COLLECT, SYSTEM_COMMANDS_CONFIG)


__all__ = [
    "CAPABILITIES",
    "SYSTEM_COLLECT",
    "SYSTEM_COMMANDS_CONFIG",
    "Capability",
]

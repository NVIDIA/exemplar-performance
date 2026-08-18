# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for capability-provider entry-point discovery."""

from __future__ import annotations

from dataclasses import dataclass, field

import llmb_capabilities._discovery as discovery_module
from llmb_capabilities import discover_providers
from llmb_capabilities._constants import CAPABILITY_GROUP


@dataclass
class _Capability:
    name: object
    provider: object = "test-provider"
    metadata: object = field(init=False)

    def __post_init__(self):
        self.metadata = {"provider": self.provider}


class _EntryPoint:
    def __init__(self, name, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


def _patch_entry_points(monkeypatch, entry_points):
    monkeypatch.setattr(
        discovery_module,
        "entry_points",
        lambda group: entry_points if group == CAPABILITY_GROUP else [],
    )


def test_groups_all_capabilities_by_metadata_provider(monkeypatch):
    bearer = _Capability("oauth.bearer-token", "llmb-auth")
    storage = _Capability("storage.upload", "llmb-uploader")
    _patch_entry_points(
        monkeypatch,
        [
            _EntryPoint("auth", lambda: (bearer,)),
            _EntryPoint("uploader", lambda: (storage,)),
        ],
    )

    assert discover_providers() == {
        "llmb-auth": {"oauth.bearer-token": bearer},
        "llmb-uploader": {"storage.upload": storage},
    }


def test_uses_entry_point_name_when_provider_metadata_is_absent(monkeypatch):
    capability = _Capability("system.collect")
    capability.metadata = {}
    _patch_entry_points(monkeypatch, [_EntryPoint("collector", lambda: (capability,))])

    assert discover_providers() == {"collector": {"system.collect": capability}}


def test_skips_broken_entry_points_and_malformed_advertisements(monkeypatch):
    def fail():
        raise RuntimeError("broken provider")

    malformed_name = _Capability(None)
    malformed_provider = _Capability("oauth.bearer-token", provider=None)
    _patch_entry_points(
        monkeypatch,
        [
            _EntryPoint("broken", fail),
            _EntryPoint("malformed", lambda: (malformed_name, malformed_provider)),
        ],
    )

    assert discover_providers() == {}


def test_skips_non_iterable_payload_and_non_mapping_metadata(monkeypatch):
    def fail_iteration():
        yield _Capability("partial")
        raise RuntimeError("broken advertisement iterator")

    non_iterable_payload = _Capability("oauth.bearer-token")
    non_mapping_metadata = _Capability("oidc.id-token")
    non_mapping_metadata.metadata = []
    valid = _Capability("system.collect", provider="llmb-collector")
    _patch_entry_points(
        monkeypatch,
        [
            _EntryPoint("non-iterable", lambda: non_iterable_payload),
            _EntryPoint("broken-iterator", fail_iteration),
            _EntryPoint("bad-metadata", lambda: (non_mapping_metadata,)),
            _EntryPoint("valid", lambda: (valid,)),
        ],
    )

    assert discover_providers() == {"llmb-collector": {"system.collect": valid}}

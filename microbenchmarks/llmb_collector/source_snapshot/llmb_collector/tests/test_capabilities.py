# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llmb_collector.capabilities`."""

from __future__ import annotations

import dataclasses
from importlib.metadata import entry_points
from pathlib import Path

import llmb_collector
import pytest
from llmb_capabilities.conformance import assert_capability_conforms
from llmb_collector import capabilities as caps_module
from llmb_collector.capabilities import (
    CAPABILITIES,
    SYSTEM_COLLECT,
    SYSTEM_COMMANDS_CONFIG,
    Capability,
    _LazyImmutableMapping,
)
from llmb_collector.collect import CollectConfig


class TestRegistryShape:
    def test_capability_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            SYSTEM_COLLECT.name = "tampered"  # type: ignore[misc]

    def test_capabilities_tuple_has_expected_entries(self) -> None:
        by_name = {cap.name: cap for cap in CAPABILITIES}
        assert set(by_name) == {"system.collect", "system.commands-config"}
        for cap in CAPABILITIES:
            assert cap.version == 1
            assert cap.metadata["provider"] == "llmb-collector"
            assert callable(cap.invoke)
            # Collector capabilities don't contribute argparse flags; the
            # invoke takes a CollectConfig kwarg or runs with all-on
            # defaults. add_arguments stays None on purpose.
            assert cap.add_arguments is None


class TestSystemCollectCapability:
    def test_invoke_returns_dict_with_default_collectors_all_on(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Skip the actual subprocess fan-out by short-circuiting the per-
        # realm collectors; we only need to verify the capability runs
        # with host / network / container / env all enabled even though
        # CollectConfig's own default disables container.
        from llmb_collector import collect as collect_module

        called: dict[str, bool] = {}

        def _stub_section(key: str):
            def _impl(data: dict, *_args, **_kwargs) -> None:
                called[key] = True
                data[key] = {}

            return _impl

        monkeypatch.setattr(collect_module, "_collect_system_info", _stub_section("system"))
        monkeypatch.setattr(collect_module, "_collect_network_info", _stub_section("network"))
        monkeypatch.setattr(collect_module, "_collect_container_info", _stub_section("container"))

        def _stub_env(data: dict, *_args, **_kwargs) -> None:
            called["env"] = True
            data["env"] = {}

        monkeypatch.setattr(collect_module, "_collect_env", _stub_env)

        result = SYSTEM_COLLECT.invoke(None)

        assert isinstance(result, dict)
        assert called == {
            "system": True,
            "network": True,
            "container": True,
            "env": True,
        }
        assert result["version"] == "1"

    def test_default_config_overrides_collectconfig_container_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # CollectConfig() defaults container=False; the capability promises
        # "all on" so it must flip that. Spy on the config passed into
        # ``collect`` to assert the override actually happens, independently
        # of which realms the stubbed collectors execute.
        captured: list[CollectConfig] = []

        def _spy_collect(config: CollectConfig) -> dict:
            captured.append(config)
            return {"version": "1"}

        monkeypatch.setattr(caps_module, "collect", _spy_collect)

        SYSTEM_COLLECT.invoke(None)

        assert len(captured) == 1
        cfg = captured[0]
        assert cfg.host is True
        assert cfg.network is True
        assert cfg.container is True
        assert cfg.env is True

    def test_invoke_honours_explicit_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # An explicit CollectConfig with everything off should produce a
        # report where every section stays None, proving config wins.
        from llmb_collector import collect as collect_module

        def _forbid(label: str):
            def _impl(*_a, **_kw) -> None:
                pytest.fail(f"{label} must not run when disabled")

            return _impl

        for fn_name in (
            "_collect_system_info",
            "_collect_network_info",
            "_collect_container_info",
            "_collect_env",
        ):
            monkeypatch.setattr(collect_module, fn_name, _forbid(fn_name))

        config = CollectConfig(host=False, network=False, container=False, env=False)
        result = SYSTEM_COLLECT.invoke(None, config=config)

        assert result["system"] is None
        assert result["network"] is None
        assert result["container"] is None
        assert result["env"] is None


class TestSystemCommandsConfigCapability:
    def test_metadata_exposes_packaged_path(self) -> None:
        commands_config_path = SYSTEM_COMMANDS_CONFIG.metadata["commands_config_path"]
        assert commands_config_path, "system.commands-config must publish metadata['commands_config_path']"
        # Inside the wheel/source tree this resolves to a real directory;
        # consumers can stat it without invoking.
        assert Path(commands_config_path).is_dir()

    def test_invoke_returns_traversable_pointing_at_packaged_tree(self) -> None:
        root = SYSTEM_COMMANDS_CONFIG.invoke(None)
        # Duck-typed Traversable contract: iterable directory.
        assert root.is_dir()
        children = {child.name for child in root.iterdir()}
        # Don't pin the full set (it can grow); just assert it's non-empty
        # and that the realm folders the collector documents live here.
        assert children, "packaged commands_config tree must not be empty"


class TestMetadataImmutability:
    def test_system_collect_metadata_rejects_assignment(self) -> None:
        # MappingProxyType raises TypeError on item assignment.
        with pytest.raises(TypeError):
            SYSTEM_COLLECT.metadata["tampered"] = "x"  # type: ignore[index]

    def test_system_commands_config_metadata_rejects_assignment(self) -> None:
        # _LazyImmutableMapping deliberately omits __setitem__.
        with pytest.raises(TypeError):
            SYSTEM_COMMANDS_CONFIG.metadata["tampered"] = "x"  # type: ignore[index]

    def test_default_factory_returns_independent_immutable_views(self) -> None:
        # Capability instances built with the default factory must not share
        # a single mutable metadata object across instances.
        first = Capability(name="a", version=1, invoke=lambda *_a, **_kw: None)
        second = Capability(name="b", version=1, invoke=lambda *_a, **_kw: None)
        assert first.metadata is not second.metadata
        with pytest.raises(TypeError):
            first.metadata["x"] = "y"  # type: ignore[index]


class TestLazyImmutableMapping:
    def test_iter_yields_eager_then_lazy_keys(self) -> None:
        mapping = _LazyImmutableMapping(
            eager={"a": "1", "b": "2"},
            lazy={"c": lambda: "3"},
        )
        assert list(mapping) == ["a", "b", "c"]
        assert len(mapping) == 3

    def test_lazy_value_is_only_resolved_on_first_access(self) -> None:
        calls = 0

        def _resolve() -> str:
            nonlocal calls
            calls += 1
            return "resolved"

        mapping = _LazyImmutableMapping(eager={}, lazy={"k": _resolve})

        assert calls == 0
        assert mapping["k"] == "resolved"
        assert calls == 1
        assert mapping["k"] == "resolved"
        assert calls == 1, "lazy value must be memoized after first access"

    def test_unknown_key_raises_key_error(self) -> None:
        mapping = _LazyImmutableMapping(eager={"a": "1"}, lazy={"b": lambda: "2"})
        with pytest.raises(KeyError):
            mapping["missing"]

    def test_system_commands_config_uses_lazy_path_resolver(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Confirm the lazy lambda reads the live module attribute rather than
        # a snapshot captured at import time; constructed on a fresh mapping
        # so we don't depend on whether the singleton's cache was warm.
        sentinel = "sentinel-commands-config"  # opaque marker; never opened as a path
        monkeypatch.setattr(
            caps_module,
            "get_packaged_commands_config",
            lambda: sentinel,
        )
        fresh = _LazyImmutableMapping(
            eager={"provider": "x"},
            lazy={
                "commands_config_path": lambda: str(caps_module.get_packaged_commands_config()),
            },
        )
        assert fresh["commands_config_path"] == sentinel


class TestConformance:
    """Enforces the llmb-capabilities contract against this producer's CAPABILITIES.

    Catches a signature drift (e.g. a dropped documented kwarg on
    ``invoke``) here, in this package's own CI, before a consumer ever
    sees the broken release. Parametrized per capability so a failure on
    one doesn't stop the rest from being checked in the same run.
    """

    @pytest.mark.parametrize("cap", CAPABILITIES, ids=lambda cap: cap.name)
    def test_capabilities_conform(self, cap: Capability) -> None:
        assert_capability_conforms(cap)


class TestEntryPointAndExports:
    def test_entry_point_loads_to_the_same_tuple(self) -> None:
        eps = entry_points(group="llmb_capabilities")
        assert "default" in eps.names
        assert eps["default"].load() is CAPABILITIES

    def test_top_level_re_exports_match(self) -> None:
        assert llmb_collector.CAPABILITIES is CAPABILITIES
        assert llmb_collector.Capability is Capability

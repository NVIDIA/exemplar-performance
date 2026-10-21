#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPU-specific metadata overrides and venv grouping behavior."""

from llmb_install.cluster.gpu import _resolve_tools_for_gpu, resolve_gpu_overrides
from llmb_install.core.dependency import group_workloads_by_dependencies
from llmb_install.downloads.tools import get_required_tools


def test_resolve_by_gpu_images_and_repos():
    workloads = {
        "wl": {
            "container": {
                "images": {
                    "by_gpu": {
                        "default": ["nvcr.io#nvidia/nemo:25.07.01"],
                        "h100": ["nvcr.io#nvidia/nemo:25.06.00"],
                    }
                }
            },
            "repositories": {
                "by_gpu": {
                    "default": {
                        "nemo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        }
                    },
                    "h100": {
                        "nemo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        }
                    },
                }
            },
        }
    }

    specialized = resolve_gpu_overrides(workloads, "h100")
    images = specialized["wl"]["container"]["images"]
    assert images == ["nvcr.io#nvidia/nemo:25.06.00"]
    assert specialized["wl"]["repositories"]["nemo"]["commit"] == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_grouping_ignores_wrappers_when_resolved():
    base_repos_flat = {
        "nemo": {
            "url": "https://github.com/NVIDIA/NeMo.git",
            "commit": "cccccccccccccccccccccccccccccccccccccccc",
        }
    }
    base_deps = {"pip": [{"package": "nemo", "repo_key": "nemo"}]}

    workloads = {
        "a": {
            "repositories": {"by_gpu": {"default": base_repos_flat, "h100": base_repos_flat}},
            "setup": {"dependencies": base_deps},
        },
        "b": {
            "repositories": base_repos_flat,
            "setup": {"dependencies": base_deps},
        },
    }

    specialized = resolve_gpu_overrides(workloads, "h100")
    groups = group_workloads_by_dependencies(specialized, ["a", "b"])

    non_scripted_groups = {k: v for k, v in groups.items() if k is not None}
    assert len(non_scripted_groups) == 1
    only_group = next(iter(non_scripted_groups.values()))
    assert set(only_group) == {"a", "b"}


def test_resolve_tools_simple_string():
    """Test simple string format for tools."""
    tools = {'nsys': '2025.5.1.120'}
    result = _resolve_tools_for_gpu(tools, 'h100')
    assert result == {'nsys': '2025.5.1.120'}


def test_resolve_tools_by_gpu_with_default():
    """Test GPU-conditional format with default fallback."""
    tools = {'nsys': {'by_gpu': {'h100': '2025.1.1', 'gb300': '2025.6.0', 'default': '2025.5.1.120'}}}

    assert _resolve_tools_for_gpu(tools, 'h100') == {'nsys': '2025.1.1'}
    assert _resolve_tools_for_gpu(tools, 'gb300') == {'nsys': '2025.6.0'}
    assert _resolve_tools_for_gpu(tools, 'b200') == {'nsys': '2025.5.1.120'}
    assert _resolve_tools_for_gpu(tools, 'gb200') == {'nsys': '2025.5.1.120'}


def test_resolve_tools_by_gpu_no_default():
    """Test GPU-conditional format without default (some GPUs use container)."""
    tools = {
        'nsys': {
            'by_gpu': {
                'h100': '2025.1.1',
                'gb300': '2025.6.0',
            }
        }
    }

    assert _resolve_tools_for_gpu(tools, 'h100') == {'nsys': '2025.1.1'}
    assert _resolve_tools_for_gpu(tools, 'gb300') == {'nsys': '2025.6.0'}
    assert _resolve_tools_for_gpu(tools, 'b200') == {}
    assert _resolve_tools_for_gpu(tools, 'gb200') == {}


def test_resolve_tools_empty():
    """Test empty tools section returns empty dict."""
    assert _resolve_tools_for_gpu({}, 'h100') == {}
    assert _resolve_tools_for_gpu(None, 'h100') == {}


def test_get_required_tools_deduplication():
    """Test that identical tool versions are deduplicated across workloads."""
    workloads = {
        'wl1': {'tools': {'nsys': '2025.1.1'}},
        'wl2': {'tools': {'nsys': '2025.1.1'}},
        'wl3': {'tools': {'nsys': '2025.1.1'}},
    }

    result = get_required_tools(workloads, ['wl1', 'wl2', 'wl3'])
    assert len(result) == 1
    assert ('nsys', '2025.1.1') in result


def test_get_required_tools_multiple_versions():
    """Test that different tool versions are collected separately."""
    workloads = {
        'wl1': {'tools': {'nsys': '2025.1.1'}},
        'wl2': {'tools': {'nsys': '2025.6.0'}},
        'wl3': {'tools': {'nsys': '2025.5.1.120'}},
    }

    result = get_required_tools(workloads, ['wl1', 'wl2', 'wl3'])
    assert len(result) == 3
    assert ('nsys', '2025.1.1') in result
    assert ('nsys', '2025.6.0') in result
    assert ('nsys', '2025.5.1.120') in result


def test_get_required_tools_with_container_versions():
    """Test that workloads without tools (using container) are skipped."""
    workloads = {
        'wl1': {'tools': {'nsys': '2025.1.1'}},
        'wl2': {'tools': {}},
        'wl3': {},
    }

    result = get_required_tools(workloads, ['wl1', 'wl2', 'wl3'])
    assert len(result) == 1
    assert ('nsys', '2025.1.1') in result


def test_resolve_gpu_overrides_includes_tools():
    """Test that resolve_gpu_overrides properly resolves tools section."""
    workloads = {
        "wl": {
            "container": {"images": ["nvcr.io#nvidia/nemo:25.07.01"]},
            "tools": {"nsys": {"by_gpu": {"h100": "2025.1.1", "default": "2025.5.1.120"}}},
        }
    }

    specialized_h100 = resolve_gpu_overrides(workloads, "h100")
    assert specialized_h100["wl"]["tools"] == {'nsys': '2025.1.1'}

    specialized_b200 = resolve_gpu_overrides(workloads, "b200")
    assert specialized_b200["wl"]["tools"] == {'nsys': '2025.5.1.120'}


def test_resolve_gpu_overrides_tools_no_default():
    """Test that resolve_gpu_overrides returns empty for unspecified GPUs without default."""
    workloads = {
        "wl": {
            "container": {"images": ["nvcr.io#nvidia/nemo:25.07.01"]},
            "tools": {
                "nsys": {
                    "by_gpu": {
                        "h100": "2025.1.1",
                    }
                }
            },
        }
    }

    specialized_h100 = resolve_gpu_overrides(workloads, "h100")
    assert specialized_h100["wl"]["tools"] == {'nsys': '2025.1.1'}

    specialized_b200 = resolve_gpu_overrides(workloads, "b200")
    assert specialized_b200["wl"]["tools"] == {}


def test_resolve_setup_simple():
    """Test standard setup section is preserved."""
    setup_config = {"venv_req": True, "dependencies": {"pip": ["package-a"]}}
    workloads = {"wl": {"setup": setup_config}}

    resolved = resolve_gpu_overrides(workloads, "h100")
    assert resolved["wl"]["setup"] == setup_config


def test_resolve_setup_by_gpu_specific():
    """Test setup resolution selects specific GPU config."""
    setup_h100 = {"venv_req": True, "dependencies": {"pip": ["package-h100"]}}
    setup_default = {"venv_req": False, "dependencies": {"pip": ["package-default"]}}

    workloads = {"wl": {"setup": {"by_gpu": {"h100": setup_h100, "default": setup_default}}}}

    resolved = resolve_gpu_overrides(workloads, "h100")
    assert resolved["wl"]["setup"] == setup_h100


def test_resolve_setup_by_gpu_default():
    """Test setup resolution falls back to default."""
    setup_h100 = {"dependencies": {"pip": ["package-h100"]}}
    setup_default = {"dependencies": {"pip": ["package-default"]}}

    workloads = {"wl": {"setup": {"by_gpu": {"h100": setup_h100, "default": setup_default}}}}

    resolved = resolve_gpu_overrides(workloads, "gb200")
    assert resolved["wl"]["setup"] == setup_default


def test_resolve_setup_by_gpu_no_match_raises():
    """Test setup resolution raises ValueError when no match found."""
    import pytest

    workloads = {"wl": {"setup": {"by_gpu": {"h100": {"dependencies": {}}}}}}

    with pytest.raises(ValueError, match="No setup defined for GPU 'gb200'"):
        resolve_gpu_overrides(workloads, "gb200")

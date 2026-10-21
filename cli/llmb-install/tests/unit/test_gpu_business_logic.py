# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU business logic functions."""

import pytest

from llmb_install.cluster.gpu import (
    _gpu_sort_key,
    filter_workloads_by_gpu_type,
    get_architecture_choices,
    get_available_gpu_choices,
    get_default_architecture,
    get_supported_gpu_types,
    resolve_gpu_overrides,
    should_auto_select_architecture,
)


class TestGetSupportedGpuTypes:
    """Test get_supported_gpu_types function."""

    def test_extract_gpu_types_from_workloads(self):
        """Test extracting GPU types from workload metadata."""
        workloads = {
            'workload1': {'run': {'gpu_configs': {'h100': {'nodes': 1}, 'b200': {'nodes': 2}}}},
            'workload2': {'run': {'gpu_configs': {'gb200': {'nodes': 4}, 'h100': {'nodes': 1}}}},
        }

        gpu_types = get_supported_gpu_types(workloads)
        assert gpu_types == {'h100', 'b200', 'gb200'}

    def test_empty_workloads(self):
        """Test with empty workloads."""
        gpu_types = get_supported_gpu_types({})
        assert gpu_types == set()

    def test_workloads_without_gpu_configs(self):
        """Test workloads without GPU configurations."""
        workloads = {'workload1': {'run': {}}, 'workload2': {'general': {'name': 'test'}}}

        gpu_types = get_supported_gpu_types(workloads)
        assert gpu_types == set()


class TestFilterWorkloadsByGpuType:
    """Test filter_workloads_by_gpu_type function."""

    def test_filter_by_gpu_type(self):
        """Test filtering workloads by GPU type."""
        workloads = {
            'workload1': {'run': {'gpu_configs': {'h100': {'nodes': 1}, 'b200': {'nodes': 2}}}},
            'workload2': {'run': {'gpu_configs': {'gb200': {'nodes': 4}}}},
            'workload3': {'run': {'gpu_configs': {'h100': {'nodes': 2}}}},
        }

        filtered = filter_workloads_by_gpu_type(workloads, 'h100')
        assert set(filtered.keys()) == {'workload1', 'workload3'}

        filtered = filter_workloads_by_gpu_type(workloads, 'gb200')
        assert set(filtered.keys()) == {'workload2'}

    def test_no_matching_gpu_type(self):
        """Test filtering with no matching GPU type."""
        workloads = {'workload1': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}}}

        filtered = filter_workloads_by_gpu_type(workloads, 'unknown_gpu')
        assert filtered == {}


class TestGpuSortKey:
    """Test _gpu_sort_key function."""

    def test_sort_key_matches_display_order(self):
        """Sorting all supported types must reproduce GPU_DISPLAY_ORDER exactly."""
        from llmb_install.constants import GPU_DISPLAY_ORDER, SUPPORTED_GPU_TYPES

        assert set(GPU_DISPLAY_ORDER) == SUPPORTED_GPU_TYPES
        shuffled = list(reversed(GPU_DISPLAY_ORDER))
        assert sorted(shuffled, key=_gpu_sort_key) == GPU_DISPLAY_ORDER

    def test_sort_key_produces_consistent_ordering(self):
        """Test that sort key produces a consistent total ordering."""
        gpu_types = ['h100', 'b200', 'gb300', 'b300', 'gb200', 'vr']
        sorted_once = sorted(gpu_types, key=_gpu_sort_key)
        sorted_twice = sorted(sorted_once, key=_gpu_sort_key)

        # Sorting should be idempotent
        assert sorted_once == sorted_twice

    def test_sort_key_case_insensitive(self):
        """Test that sort key is case insensitive."""
        assert _gpu_sort_key('H100') == _gpu_sort_key('h100')
        assert _gpu_sort_key('GB200') == _gpu_sort_key('gb200')
        assert _gpu_sort_key('VR') == _gpu_sort_key('vr')

    def test_sort_key_unknown_type_raises_error(self):
        """Test that unknown GPU type raises ValueError."""
        with pytest.raises(ValueError, match="not found in GPU_DISPLAY_ORDER"):
            _gpu_sort_key('x1000')

    def test_sort_key_invalid_format_raises_error(self):
        """Test that unrecognized GPU types raise ValueError."""
        with pytest.raises(ValueError, match="not found in GPU_DISPLAY_ORDER"):
            _gpu_sort_key('invalid-gpu')


class TestGetAvailableGpuChoices:
    """Test get_available_gpu_choices function."""

    def test_get_choices_from_workloads(self):
        """Test getting GPU choices from workloads."""
        workloads = {
            'workload1': {
                'run': {
                    'gpu_configs': {
                        'h100': {'nodes': 1},
                        'b200': {'nodes': 2},
                        'unknown_gpu': {'nodes': 1},  # This should be filtered out
                    }
                }
            }
        }

        choices = get_available_gpu_choices(workloads)

        # Should only include known GPU types
        expected_values = {'h100', 'b200'}
        actual_values = {choice['value'] for choice in choices}
        assert actual_values == expected_values

        # Check format
        for choice in choices:
            assert 'name' in choice
            assert 'value' in choice
            assert choice['name'] == choice['value'].upper()

    def test_choices_sorted_correctly(self):
        """Test that GPU choices are sorted by GPU_DISPLAY_ORDER."""
        from llmb_install.constants import GPU_DISPLAY_ORDER

        workloads = {
            'workload1': {
                'run': {
                    'gpu_configs': {
                        'h100': {'nodes': 1},
                        'b200': {'nodes': 2},
                        'gb200': {'nodes': 3},
                        'b300': {'nodes': 4},
                        'gb300': {'nodes': 5},
                    }
                }
            }
        }

        choices = get_available_gpu_choices(workloads)
        values = [choice['value'] for choice in choices]

        assert values == [g for g in GPU_DISPLAY_ORDER if g in set(values)]

    def test_no_supported_gpu_types(self):
        """Test when no supported GPU types are found."""
        workloads = {'workload1': {'run': {'gpu_configs': {'unknown_gpu': {'nodes': 1}}}}}

        with pytest.raises(ValueError, match="No supported GPU types found"):
            get_available_gpu_choices(workloads)


class TestArchitectureFunctions:
    """Test architecture-related functions."""

    def test_get_architecture_choices(self):
        """Test getting architecture choices."""
        choices = get_architecture_choices('h100')

        expected = [{'name': 'x86_64', 'value': 'x86_64'}, {'name': 'aarch64', 'value': 'aarch64'}]
        assert choices == expected

    def test_should_auto_select_architecture(self):
        """Test auto-selection logic for architecture."""
        assert should_auto_select_architecture('gb200') is True
        assert should_auto_select_architecture('gb300') is True
        assert should_auto_select_architecture('h100') is False
        assert should_auto_select_architecture('b200') is False
        assert should_auto_select_architecture('unknown') is False

    def test_get_default_architecture(self):
        """Test getting default architecture by GPU type."""
        assert get_default_architecture('h100') == 'x86_64'
        assert get_default_architecture('b200') == 'x86_64'
        assert get_default_architecture('gb200') == 'aarch64'
        assert get_default_architecture('unknown_gpu') == 'x86_64'  # Default fallback


class TestResolveGpuOverrides:
    """Test resolve_gpu_overrides function."""

    def test_resolve_simple_images(self):
        """Test resolving simple image configurations."""
        workloads = {'workload1': {'container': {'images': ['image1.tar', 'image2.tar']}}}

        resolved = resolve_gpu_overrides(workloads, 'h100')

        assert resolved['workload1']['container']['images'] == ['image1.tar', 'image2.tar']

    def test_resolve_gpu_specific_images(self):
        """Test resolving GPU-specific image configurations."""
        workloads = {
            'workload1': {
                'container': {
                    'images': {
                        'by_gpu': {
                            'h100': ['h100_image.tar'],
                            'gb200': ['gb200_image.tar'],
                            'default': ['default_image.tar'],
                        }
                    }
                }
            }
        }

        resolved_h100 = resolve_gpu_overrides(workloads, 'h100')
        assert resolved_h100['workload1']['container']['images'] == ['h100_image.tar']

        resolved_unknown = resolve_gpu_overrides(workloads, 'unknown_gpu')
        assert resolved_unknown['workload1']['container']['images'] == ['default_image.tar']

    def test_resolve_repositories(self):
        """Test resolving repository configurations."""
        workloads = {
            'workload1': {
                'repositories': {'repo1': {'url': 'https://github.com/example/repo1.git', 'commit': 'abc123'}}
            }
        }

        resolved = resolve_gpu_overrides(workloads, 'h100')

        expected_repo = {'url': 'https://github.com/example/repo1.git', 'commit': 'abc123', 'submodules': False}
        assert resolved['workload1']['repositories']['repo1'] == expected_repo

    def test_deep_copy_behavior(self):
        """Test that original workloads are not modified."""
        original_workloads = {'workload1': {'container': {'images': ['image1.tar']}}}

        resolved = resolve_gpu_overrides(original_workloads, 'h100')

        # Modify resolved version
        resolved['workload1']['container']['images'].append('new_image.tar')

        # Original should be unchanged
        assert original_workloads['workload1']['container']['images'] == ['image1.tar']


class TestPrepareWorkloadsErrorMessages:
    """Test _prepare_workloads error messages."""

    def test_workload_exists_but_unsupported_gpu_type(self, tmp_path, capsys):
        """Test error message when workload exists but doesn't support the gpu_type.

        The main case: workload exists in repo, some workloads DO support
        the GPU type, but the selected workload doesn't.
        """
        from llmb_install.core.installer import Installer

        # Create a mock installer with workloads
        installer = Installer()
        installer.workloads = {
            'workload_h100_only': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}},
            'workload_gb200_only': {'run': {'gpu_configs': {'gb200': {'nodes': 2}}}},
        }

        # Try to select h100_only workload for gb200 GPU
        # gb200_only workload exists and supports gb200, so filtered_workloads is not empty
        # But h100_only doesn't support gb200
        with pytest.raises(SystemExit) as exc_info:
            installer._prepare_workloads('gb200', ['workload_h100_only'])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "do not support gpu_type 'gb200'" in captured.out
        assert "workload_h100_only" in captured.out
        assert "supported gpu_types: h100" in captured.out

    def test_workload_truly_not_found(self, tmp_path, capsys):
        """Test error message when workload truly doesn't exist."""
        from llmb_install.core.installer import Installer

        installer = Installer()
        installer.workloads = {
            'workload_h100_only': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}},
        }

        # Try to prepare a workload that doesn't exist
        with pytest.raises(SystemExit) as exc_info:
            installer._prepare_workloads('h100', ['nonexistent_workload'])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "workloads were not found" in captured.out
        assert "nonexistent_workload" in captured.out

    def test_mixed_unsupported_and_not_found(self, tmp_path, capsys):
        """Test error message with both unsupported and not-found workloads."""
        from llmb_install.core.installer import Installer

        installer = Installer()
        installer.workloads = {
            'workload_h100_only': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}},
            'workload_b200_gb200': {'run': {'gpu_configs': {'b200': {'nodes': 2}, 'gb200': {'nodes': 4}}}},
        }

        # Try to prepare workloads: one exists but unsupported, one doesn't exist
        with pytest.raises(SystemExit) as exc_info:
            installer._prepare_workloads('gb200', ['workload_h100_only', 'nonexistent'])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()

        # Should show unsupported error
        assert "do not support gpu_type 'gb200'" in captured.out
        assert "workload_h100_only" in captured.out
        assert "supported gpu_types: h100" in captured.out

        # Should also show not-found error
        assert "workloads were not found" in captured.out
        assert "nonexistent" in captured.out

    def test_multiple_unsupported_workloads_show_their_gpus(self, tmp_path, capsys):
        """Test that multiple unsupported workloads each show their supported GPUs."""
        from llmb_install.core.installer import Installer

        installer = Installer()
        installer.workloads = {
            'workload_h100_only': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}},
            'workload_b200_only': {'run': {'gpu_configs': {'b200': {'nodes': 2}}}},
            'workload_h100_b200': {'run': {'gpu_configs': {'h100': {'nodes': 1}, 'b200': {'nodes': 2}}}},
            'workload_gb200_only': {'run': {'gpu_configs': {'gb200': {'nodes': 4}}}},
        }

        # Try to select workloads that don't support gb200 (but gb200_only does, so filtered_workloads not empty)
        with pytest.raises(SystemExit) as exc_info:
            installer._prepare_workloads('gb200', ['workload_h100_only', 'workload_b200_only', 'workload_h100_b200'])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()

        # Should show all unsupported workloads with their supported GPUs
        assert "do not support gpu_type 'gb200'" in captured.out
        assert "workload_h100_only" in captured.out
        assert "workload_b200_only" in captured.out
        assert "workload_h100_b200" in captured.out

        # Check that each workload shows its supported GPUs
        assert "workload_h100_only (supported gpu_types: h100)" in captured.out
        assert "workload_b200_only (supported gpu_types: b200)" in captured.out
        # workload_h100_b200 could be "b200, h100" or "h100, b200" depending on sorted() output
        assert "workload_h100_b200 (supported gpu_types:" in captured.out

    def test_early_exit_with_selected_workloads(self, tmp_path, capsys):
        """Test early exit error when NO workloads support GPU but user selected specific ones.

        This tests the early exit improvement: when filtered_workloads is empty but
        user selected specific workloads, show what those workloads support.
        """
        from llmb_install.core.installer import Installer

        installer = Installer()
        installer.workloads = {
            'workload_h100_only': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}},
            'workload_b200_only': {'run': {'gpu_configs': {'b200': {'nodes': 2}}}},
        }

        # Try to select workloads for gb200, but NO workloads in repo support gb200
        # This hits the early exit (filtered_workloads is empty)
        with pytest.raises(SystemExit) as exc_info:
            installer._prepare_workloads('gb200', ['workload_h100_only', 'workload_b200_only'])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()

        # Should show improved error with supported GPU types
        assert "do not support gpu_type 'gb200'" in captured.out
        assert "workload_h100_only" in captured.out
        assert "supported gpu_types: h100" in captured.out
        assert "workload_b200_only" in captured.out
        assert "supported gpu_types: b200" in captured.out

    def test_early_exit_with_invalid_gpu_type(self, tmp_path, capsys):
        """Test early exit with invalid/unknown GPU type shows generic error.

        When NO workloads support the GPU type and user didn't select specific workloads,
        or selected workloads don't exist, show generic error.
        """
        from llmb_install.core.installer import Installer

        installer = Installer()
        installer.workloads = {
            'workload_h100_only': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}},
        }

        # Try with invalid GPU type and empty selection (or selecting non-existent workloads)
        with pytest.raises(SystemExit) as exc_info:
            installer._prepare_workloads('xyz123', ['nonexistent_workload'])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()

        # Should show generic error since workload doesn't exist in self.workloads
        assert "No workloads found that support xyz123 GPU type" in captured.out

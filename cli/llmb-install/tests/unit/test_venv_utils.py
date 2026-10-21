# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for venv utility functions."""

from pathlib import Path

from llmb_install.environment.venv_utils import (
    build_venv_hash_mapping,
    extract_venv_hash,
    extract_venv_type_from_config,
    should_reuse_venv,
)


class TestExtractVenvHash:
    """Tests for extracting dependency hash from venv paths."""

    def test_valid_venv_path(self):
        """Should extract hash from valid venv path."""
        result = extract_venv_hash('/install/venvs/venv_abc123def456')
        assert result == 'abc123def456'

    def test_venv_path_with_subdirs(self):
        """Should extract hash even with subdirectories in path."""
        result = extract_venv_hash('/install/venvs/venv_abc123def456/lib/python3.10')
        assert result == 'abc123def456'

    def test_no_hash_pattern(self):
        """Should return None when no hash pattern found."""
        result = extract_venv_hash('/install/venvs/myenv')
        assert result is None

    def test_wrong_hash_length(self):
        """Should return None for hashes with wrong length."""
        result = extract_venv_hash('/install/venvs/venv_abc123')  # Only 6 chars
        assert result is None

    def test_non_hex_characters(self):
        """Should return None for invalid hex characters."""
        result = extract_venv_hash('/install/venvs/venv_ghijklmnopqr')
        assert result is None

    def test_uppercase_hex_not_matched(self):
        """Should not match uppercase hex (pattern expects lowercase)."""
        result = extract_venv_hash('/install/venvs/venv_ABC123DEF456')
        assert result is None


class TestBuildVenvHashMapping:
    """Tests for building hash->venv mapping."""

    def test_single_venv(self):
        """Should build mapping with single venv."""
        workload_venvs = {'wl1': '/venvs/venv_abc123def456'}
        result = build_venv_hash_mapping(workload_venvs)

        assert len(result) == 1
        assert result['abc123def456'] == '/venvs/venv_abc123def456'

    def test_multiple_venvs(self):
        """Should build mapping with multiple venvs."""
        workload_venvs = {
            'wl1': '/venvs/venv_abc123def456',
            'wl2': '/venvs/venv_123abc456def',
            'wl3': '/venvs/venv_fedcba987654',
        }
        result = build_venv_hash_mapping(workload_venvs)

        assert len(result) == 3
        assert result['abc123def456'] == '/venvs/venv_abc123def456'
        assert result['123abc456def'] == '/venvs/venv_123abc456def'
        assert result['fedcba987654'] == '/venvs/venv_fedcba987654'

    def test_skip_invalid_paths(self):
        """Should skip venv paths without valid hash patterns."""
        workload_venvs = {
            'wl1': '/venvs/venv_abc123def456',
            'wl2': '/venvs/legacy_env',  # No hash
            'wl3': '/venvs/venv_123abc456def',
        }
        result = build_venv_hash_mapping(workload_venvs)

        assert len(result) == 2
        assert 'abc123def456' in result
        assert '123abc456def' in result

    def test_empty_input(self):
        """Should handle empty input."""
        result = build_venv_hash_mapping({})
        assert result == {}

    def test_duplicate_hashes_last_wins(self):
        """When multiple workloads share venv, last one wins in mapping."""
        workload_venvs = {
            'wl1': '/venvs/venv_abc123def456',
            'wl2': '/venvs/venv_abc123def456',  # Same venv
        }
        result = build_venv_hash_mapping(workload_venvs)

        assert len(result) == 1
        assert result['abc123def456'] == '/venvs/venv_abc123def456'


class TestShouldReuseVenv:
    """Tests for venv reuse decision logic."""

    def test_exact_match_without_validation(self):
        """Should find match when hash exists (no path validation)."""
        mapping = {'abc123def456': '/venvs/venv_abc123def456'}
        result = should_reuse_venv('abc123def456789', mapping, validate_exists=False)

        assert result == '/venvs/venv_abc123def456'

    def test_no_match(self):
        """Should return None when hash not in mapping."""
        mapping = {'abc123def456': '/venvs/venv_abc123def456'}
        result = should_reuse_venv('fedcba987654321', mapping, validate_exists=False)

        assert result is None

    def test_hash_truncation(self):
        """Should truncate long hashes to 12 chars for comparison."""
        mapping = {'abc123def456': '/venvs/venv_abc123def456'}
        # Full hash is longer, but first 12 chars match
        result = should_reuse_venv('abc123def456000000000000', mapping, validate_exists=False)

        assert result == '/venvs/venv_abc123def456'

    def test_existing_path_validation(self, tmpdir):
        """Should validate path exists when requested."""
        venv_path = Path(tmpdir) / 'venv_abc123def456'
        venv_path.mkdir()

        mapping = {'abc123def456': str(venv_path)}
        result = should_reuse_venv('abc123def456', mapping, validate_exists=True)

        assert result == str(venv_path)

    def test_nonexistent_path_validation(self):
        """Should return None when path doesn't exist and validation enabled."""
        mapping = {'abc123def456': '/nonexistent/venv_abc123def456'}
        result = should_reuse_venv('abc123def456', mapping, validate_exists=True)

        assert result is None

    def test_empty_mapping(self):
        """Should return None for empty mapping."""
        result = should_reuse_venv('abc123def456', {}, validate_exists=False)
        assert result is None


class TestExtractVenvTypeFromConfig:
    """Tests for extracting venv_type with backward compatibility."""

    def test_extract_from_install_section(self):
        """Should extract from install section (new format)."""
        config = {
            'install': {'venv_type': 'uv'},
            'workloads': {'config': {}},
        }
        result = extract_venv_type_from_config(config)
        assert result == 'uv'

    def test_fallback_to_workload_config(self):
        """Should fall back to workload config (old format)."""
        config = {
            'workloads': {
                'config': {
                    'wl1': {'venv_type': 'conda'},
                    'wl2': {'venv_type': 'conda'},
                }
            }
        }
        result = extract_venv_type_from_config(config)
        assert result == 'conda'

    def test_install_section_takes_precedence(self):
        """Install section should take precedence over workload config."""
        config = {
            'install': {'venv_type': 'uv'},
            'workloads': {
                'config': {
                    'wl1': {'venv_type': 'conda'},  # Old format
                }
            },
        }
        result = extract_venv_type_from_config(config)
        assert result == 'uv'  # From install section

    def test_missing_everywhere(self):
        """Should return None when venv_type missing everywhere."""
        config = {
            'workloads': {
                'config': {
                    'wl1': {'venv_path': '/venv'},  # No venv_type
                }
            }
        }
        result = extract_venv_type_from_config(config)
        assert result is None

    def test_empty_config(self):
        """Should handle empty/missing sections gracefully."""
        config = {}
        result = extract_venv_type_from_config(config)
        assert result is None

    def test_skip_workloads_without_venv_type(self):
        """Should skip workloads without venv_type field."""
        config = {
            'workloads': {
                'config': {
                    'wl1': {'venv_path': '/venv1'},  # No venv_type
                    'wl2': {'venv_type': 'venv', 'venv_path': '/venv2'},
                }
            }
        }
        result = extract_venv_type_from_config(config)
        assert result == 'venv'  # Extracts from wl2

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool mounting logic (nsys and cupti workarounds)."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from llmb_run.nsys_mount_handler import (
    _find_cupti_library,
    _get_container_cupti_path,
    _resolve_tool_version,
)

# Test-specific lookup table (isolated from production data)
TEST_CUPTI_PATHS = {
    ('test/container:1.0.0', 'x86_64'): '/path/to/x64/libcupti.so.13.0',
    ('test/container:1.0.0', 'aarch64'): '/path/to/sbsa/libcupti-sbsa.so.13.0',
}


class TestResolveToolVersion:
    """Test tool version resolution logic (shared by nsys and cupti)."""

    def test_simple_version_string(self):
        """Simple string format returns the string directly."""
        result = _resolve_tool_version("1.0.0", "h100", "test_tool")
        assert result == "1.0.0"

    def test_by_gpu_exact_match(self):
        """GPU-conditional format with exact GPU match."""
        config = {"by_gpu": {"h100": "1.0.0", "b200": "2.0.0"}}
        result = _resolve_tool_version(config, "h100", "test_tool")
        assert result == "1.0.0"

    def test_by_gpu_default_fallback(self):
        """GPU-conditional format falls back to default."""
        config = {"by_gpu": {"h100": "1.0.0", "default": "3.0.0"}}
        result = _resolve_tool_version(config, "gb300", "test_tool")
        assert result == "3.0.0"

    def test_by_gpu_no_match_no_default(self):
        """GPU not in list and no default returns None."""
        config = {"by_gpu": {"h100": "1.0.0", "b200": "2.0.0"}}
        result = _resolve_tool_version(config, "gb300", "test_tool")
        assert result is None

    def test_invalid_format(self):
        """Unknown format returns None."""
        result = _resolve_tool_version(["invalid"], "h100", "test_tool")
        assert result is None


class TestFindCuptiLibrary:
    """Test CUPTI library file detection logic."""

    def test_directory_not_exists(self):
        """Returns None when directory doesn't exist."""
        result = _find_cupti_library("/nonexistent/path")
        assert result is None

    def test_finds_versioned_library(self):
        """Finds versioned .so file and ignores symlinks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_dir = Path(tmpdir) / "lib"
            lib_dir.mkdir()

            # Create versioned library file
            versioned_lib = lib_dir / "libcupti.so.2025.3.1"
            versioned_lib.touch()

            # Create symlink (should be ignored)
            symlink = lib_dir / "libcupti.so"
            symlink.symlink_to(versioned_lib)

            result = _find_cupti_library(str(lib_dir))
            assert result == str(versioned_lib)

    def test_no_versioned_library(self):
        """Returns None when no versioned library exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_dir = Path(tmpdir) / "lib"
            lib_dir.mkdir()

            # Create only symlink (no real versioned file)
            symlink = lib_dir / "libcupti.so"
            symlink.touch()

            result = _find_cupti_library(str(lib_dir))
            assert result is None

    def test_multiple_versioned_files(self):
        """Returns first match when multiple versioned files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_dir = Path(tmpdir) / "lib"
            lib_dir.mkdir()

            # Create multiple versioned files
            lib1 = lib_dir / "libcupti.so.2025.3.1"
            lib2 = lib_dir / "libcupti_static.a"
            lib1.touch()
            lib2.touch()

            result = _find_cupti_library(str(lib_dir))
            # Should find the .so.* file (not the .a file)
            assert result == str(lib1)


class TestGetContainerCuptiPath:
    """Test container CUPTI path lookup logic."""

    @patch('llmb_run.nsys_mount_handler.CONTAINER_CUPTI_PATHS', TEST_CUPTI_PATHS)
    def test_direct_lookup_x86_64(self):
        """Direct lookup for known container and architecture."""
        result = _get_container_cupti_path("test/container:1.0.0", "x86_64")
        assert result == "/path/to/x64/libcupti.so.13.0"

    @patch('llmb_run.nsys_mount_handler.CONTAINER_CUPTI_PATHS', TEST_CUPTI_PATHS)
    def test_direct_lookup_aarch64(self):
        """Direct lookup for known container on ARM architecture."""
        result = _get_container_cupti_path("test/container:1.0.0", "aarch64")
        assert result == "/path/to/sbsa/libcupti-sbsa.so.13.0"

    @patch('llmb_run.nsys_mount_handler.CONTAINER_CUPTI_PATHS', TEST_CUPTI_PATHS)
    def test_partial_match(self):
        """Partial match handles image variants."""
        # Should match even with extra suffix
        result = _get_container_cupti_path("test/container:1.0.0-custom", "x86_64")
        assert result == "/path/to/x64/libcupti.so.13.0"

    @patch('llmb_run.nsys_mount_handler.CONTAINER_CUPTI_PATHS', TEST_CUPTI_PATHS)
    def test_no_match(self):
        """Returns None for unknown container."""
        result = _get_container_cupti_path("unknown/image:1.0.0", "x86_64")
        assert result is None

    @patch('llmb_run.nsys_mount_handler.CONTAINER_CUPTI_PATHS', TEST_CUPTI_PATHS)
    def test_wrong_architecture(self):
        """Returns None when architecture doesn't match."""
        # Known container but with wrong architecture
        result = _get_container_cupti_path("test/container:1.0.0", "unknown_arch")
        assert result is None

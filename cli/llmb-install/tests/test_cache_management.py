#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cache management functionality in the installer."""

import os
import unittest
from unittest.mock import patch

from llmb_install.environment.cache import setup_cache_directories


class TestCacheManagement(unittest.TestCase):
    """Test cache management functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock print to avoid output during testing
        self.print_patcher = patch('builtins.print')
        self.mock_print = self.print_patcher.start()

        # Mock os.makedirs to avoid actual directory creation
        self.makedirs_patcher = patch('llmb_install.environment.cache.os.makedirs')
        self.mock_makedirs = self.makedirs_patcher.start()

        # Save original environment variables
        self.original_env = {}
        for var in ['LLMB_INSTALL', 'PIP_CACHE_DIR', 'UV_CACHE_DIR']:
            if var in os.environ:
                self.original_env[var] = os.environ[var]

    def tearDown(self):
        """Clean up patches."""
        self.print_patcher.stop()
        self.makedirs_patcher.stop()

        # Restore original environment
        for var in ['LLMB_INSTALL', 'PIP_CACHE_DIR', 'UV_CACHE_DIR']:
            if var in self.original_env:
                os.environ[var] = self.original_env[var]
            elif var in os.environ:
                del os.environ[var]

    def test_cache_directory_defaults_with_uv(self):
        """Test default cache directory setup when using uv."""
        with patch.dict(os.environ, {}, clear=True):
            setup_cache_directories('/test/llmb', 'uv')

            # Should set both cache directories (check inside context)
            self.assertEqual(os.environ['PIP_CACHE_DIR'], '/test/llmb/.cache/pip')
            self.assertEqual(os.environ['UV_CACHE_DIR'], '/test/llmb/.cache/uv')

        # Should create both directories
        self.mock_makedirs.assert_any_call('/test/llmb/.cache/pip', exist_ok=True)
        self.mock_makedirs.assert_any_call('/test/llmb/.cache/uv', exist_ok=True)
        self.assertEqual(self.mock_makedirs.call_count, 2)

    def test_cache_directory_defaults_with_venv(self):
        """Test default cache directory setup when using venv (no UV cache)."""
        with patch.dict(os.environ, {}, clear=True):
            setup_cache_directories('/test/llmb', 'venv')

            # Should only set pip cache directory (check inside context)
            self.assertEqual(os.environ['PIP_CACHE_DIR'], '/test/llmb/.cache/pip')
            self.assertNotIn('UV_CACHE_DIR', os.environ)

        # Should only create pip directory
        self.mock_makedirs.assert_called_once_with('/test/llmb/.cache/pip', exist_ok=True)

    @patch('llmb_install.environment.cache.is_uv_installed')
    def test_cache_directory_warnings_home_paths(self, mock_is_uv_installed):
        """Test warnings for cache directories under home directory."""
        # Ensure uv is considered installed so UV_CACHE_DIR logic is executed
        mock_is_uv_installed.return_value = True

        # Use paths that are actually under the current home directory
        home_dir = os.path.expanduser('~')
        pip_cache_path = os.path.join(home_dir, '.cache', 'pip')
        uv_cache_path = os.path.join(home_dir, '.cache', 'uv')

        with patch.dict(os.environ, {'PIP_CACHE_DIR': pip_cache_path, 'UV_CACHE_DIR': uv_cache_path}, clear=True):
            setup_cache_directories('/test/llmb', 'uv')

            # Should not change existing values (check inside context)
            self.assertEqual(os.environ['PIP_CACHE_DIR'], pip_cache_path)
            self.assertEqual(os.environ['UV_CACHE_DIR'], uv_cache_path)

        # Should warn about home paths since we're using actual home directory paths
        warning_calls = [
            call for call in self.mock_print.call_args_list if 'WARNING' in str(call) and 'home directory' in str(call)
        ]
        # Should get 2 warnings - one for PIP_CACHE_DIR and one for UV_CACHE_DIR
        self.assertEqual(len(warning_calls), 2)

    def test_cache_directories_already_set_non_home(self):
        """Test behavior when cache directories are already set to non-home paths."""
        with patch.dict(
            os.environ, {'PIP_CACHE_DIR': '/scratch/pip_cache', 'UV_CACHE_DIR': '/scratch/uv_cache'}, clear=True
        ):
            setup_cache_directories('/test/llmb', 'uv')

            # Should not change existing values (check inside context)
            self.assertEqual(os.environ['PIP_CACHE_DIR'], '/scratch/pip_cache')
            self.assertEqual(os.environ['UV_CACHE_DIR'], '/scratch/uv_cache')

        # Should not create new directories
        self.mock_makedirs.assert_not_called()

        # Should not print warnings about home paths
        warning_calls = [
            call for call in self.mock_print.call_args_list if 'WARNING' in str(call) and '/home' in str(call)
        ]
        self.assertEqual(len(warning_calls), 0)

    def test_partial_cache_configuration(self):
        """Test behavior when only one cache variable is set."""
        with patch.dict(os.environ, {'PIP_CACHE_DIR': '/scratch/pip_cache'}, clear=True):
            # Remove UV_CACHE_DIR if it exists
            os.environ.pop('UV_CACHE_DIR', None)

            setup_cache_directories('/test/llmb', 'uv')

            # PIP_CACHE_DIR should remain unchanged (check inside context)
            self.assertEqual(os.environ['PIP_CACHE_DIR'], '/scratch/pip_cache')

            # UV_CACHE_DIR should be set to default
            self.assertEqual(os.environ['UV_CACHE_DIR'], '/test/llmb/.cache/uv')

        # Should only create UV cache directory
        self.mock_makedirs.assert_called_once_with('/test/llmb/.cache/uv', exist_ok=True)

    def test_cache_directories_with_relative_install_path(self):
        """Test cache directory setup with relative install path."""
        with patch.dict(os.environ, {}, clear=True):
            setup_cache_directories('relative/path', 'uv')

            # Should work with relative paths (check inside context)
            self.assertEqual(os.environ['PIP_CACHE_DIR'], 'relative/path/.cache/pip')
            self.assertEqual(os.environ['UV_CACHE_DIR'], 'relative/path/.cache/uv')


if __name__ == '__main__':
    unittest.main()

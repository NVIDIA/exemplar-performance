#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for UV integration functionality in the llmb_install.environment.detector."""

import os
import unittest
from unittest.mock import MagicMock, patch

from llmb_install.environment.cache import setup_cache_directories
from llmb_install.environment.detector import (
    get_system_python_path,
    get_system_python_version,
)
from llmb_install.environment.venv_manager import create_virtual_environment, setup_uv_python_preference


class TestUvIntegration(unittest.TestCase):
    """Test UV integration functionality."""

    def setUp(self):
        """Set up common test fixtures."""
        # Mock print to avoid output during testing
        self.print_patcher = patch('builtins.print')
        self.mock_print = self.print_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.print_patcher.stop()

    @patch('llmb_install.environment.detector.subprocess.run')
    @patch('llmb_install.environment.detector.get_clean_environment_for_subprocess')
    def test_get_system_python_version_success(self, mock_get_clean_env, mock_subprocess):
        """Test successful system Python version detection."""
        mock_get_clean_env.return_value = {'PATH': '/usr/bin'}
        mock_result = MagicMock()
        mock_result.stdout = 'Python 3.12.0'
        mock_subprocess.return_value = mock_result

        version = get_system_python_version()

        self.assertEqual(version, (3, 12, 0))
        mock_subprocess.assert_called_once_with(
            ['python3', '--version'], env={'PATH': '/usr/bin'}, capture_output=True, text=True, check=True
        )

    @patch('llmb_install.environment.detector.subprocess.run')
    @patch('llmb_install.environment.detector.get_clean_environment_for_subprocess')
    def test_get_system_python_version_fallback(self, mock_get_clean_env, mock_subprocess):
        """Test None return when system Python detection fails."""
        mock_get_clean_env.return_value = {'PATH': '/usr/bin'}
        from subprocess import CalledProcessError

        mock_subprocess.side_effect = CalledProcessError(1, 'python3')

        version = get_system_python_version()

        self.assertIsNone(version)

    @patch('llmb_install.environment.detector.shutil.which')
    @patch('llmb_install.environment.detector.get_clean_environment_for_subprocess')
    def test_get_system_python_path_success(self, mock_get_clean_env, mock_which):
        """Test successful system Python path detection."""
        mock_get_clean_env.return_value = {'PATH': '/usr/bin'}
        mock_which.return_value = '/usr/bin/python3'

        path = get_system_python_path()

        self.assertEqual(path, '/usr/bin/python3')
        mock_which.assert_called_once_with('python3', path='/usr/bin')

    @patch('llmb_install.environment.detector.shutil.which')
    @patch('llmb_install.environment.detector.get_clean_environment_for_subprocess')
    def test_get_system_python_path_fallback(self, mock_get_clean_env, mock_which):
        """Test fallback to current executable when detection fails."""
        mock_get_clean_env.return_value = {'PATH': '/usr/bin'}
        mock_which.return_value = None  # python3 not found in PATH

        with patch('llmb_install.environment.detector.sys.executable', '/current/python'):
            path = get_system_python_path()

        self.assertEqual(path, '/current/python')

    @patch('llmb_install.environment.detector.subprocess.run')
    @patch('llmb_install.environment.venv_manager.get_venv_environment')
    def test_uv_environment_creation(self, mock_get_venv_env, mock_subprocess):
        """Test UV virtual environment creation process."""
        mock_get_venv_env.return_value = {'PATH': '/test/venv/bin:/usr/bin'}

        create_virtual_environment('/test/venv', 'uv')

        # Should call uv venv and then uv pip install pip
        self.assertEqual(mock_subprocess.call_count, 2)

        # First call: create venv
        first_call = mock_subprocess.call_args_list[0]
        self.assertEqual(first_call[0][0], ['uv', 'venv', '--clear', '--python', '3.12', '/test/venv'])

        # Second call: install pip
        second_call = mock_subprocess.call_args_list[1]
        self.assertEqual(second_call[0][0], ['uv', 'pip', 'install', 'pip'])
        self.assertEqual(second_call[1]['env'], {'PATH': '/test/venv/bin:/usr/bin'})

    def test_uv_python_preference_defaults_to_managed(self):
        """Managed python is the default when no uv preference variable is set."""
        with patch.dict(os.environ, {}, clear=True):
            setup_uv_python_preference()
            self.assertEqual(os.environ.get('UV_MANAGED_PYTHON'), '1')

    def test_uv_python_preference_respects_user_choice(self):
        """Any uv preference variable set by the user suppresses the managed default."""
        for var in ('UV_NO_MANAGED_PYTHON', 'UV_PYTHON_PREFERENCE'):
            with self.subTest(var=var), patch.dict(os.environ, {var: 'x'}, clear=True):
                setup_uv_python_preference()
                self.assertNotIn('UV_MANAGED_PYTHON', os.environ)

    def test_uv_python_preference_conflict_is_an_error(self):
        """Multiple uv preference variables are rejected."""
        with patch.dict(
            os.environ,
            {'UV_MANAGED_PYTHON': '1', 'UV_NO_MANAGED_PYTHON': '1'},
            clear=True,
        ):
            with self.assertRaises(ValueError) as cm:
                setup_uv_python_preference()

        self.assertIn('UV_MANAGED_PYTHON', str(cm.exception))
        self.assertIn('UV_NO_MANAGED_PYTHON', str(cm.exception))

    def test_uv_python_preference_warns_only_when_not_managed(self):
        """The 'not enforced' warning is emitted for system preferences only."""
        cases = {
            'UV_MANAGED_PYTHON=true': ({'UV_MANAGED_PYTHON': 'true'}, False),
            'only-managed': ({'UV_PYTHON_PREFERENCE': 'only-managed'}, False),
            'system': ({'UV_PYTHON_PREFERENCE': 'system'}, True),
            'UV_NO_MANAGED_PYTHON': ({'UV_NO_MANAGED_PYTHON': '1'}, True),
        }
        for name, (env, warns) in cases.items():
            with self.subTest(name), patch.dict(os.environ, env, clear=True):
                self.mock_print.reset_mock()
                setup_uv_python_preference()
                self.assertEqual('not enforced' in str(self.mock_print.call_args_list), warns)

    def test_uv_python_preference_deprecated_llmb_flag(self):
        """LLMB_DISABLE_MANAGED_PYTHON maps to UV_NO_MANAGED_PYTHON with a warning."""
        env = {'LLMB_DISABLE_MANAGED_PYTHON': 'true', 'UV_MANAGED_PYTHON': '1', 'UV_PYTHON_PREFERENCE': 'only-managed'}
        with patch.dict(os.environ, env, clear=True):
            setup_uv_python_preference()
            self.assertEqual(os.environ.get('UV_NO_MANAGED_PYTHON'), '1')
            self.assertNotIn('UV_MANAGED_PYTHON', os.environ)
            self.assertNotIn('UV_PYTHON_PREFERENCE', os.environ)
            self.assertIn('deprecated', str(self.mock_print.call_args_list))

    @patch('llmb_install.environment.detector.subprocess.run')
    @patch('llmb_install.environment.venv_manager.get_clean_environment_for_subprocess')
    def test_venv_creation_uses_system_python(self, mock_get_clean_env, mock_subprocess):
        """Test that venv creation uses system Python."""
        mock_get_clean_env.return_value = {'PATH': '/usr/bin'}

        create_virtual_environment('/test/venv', 'venv')

        mock_subprocess.assert_called_once_with(
            ['python3', '-m', 'venv', '--clear', '/test/venv'],
            env={'PATH': '/usr/bin'},
            check=True,
            capture_output=True,
            text=True,
        )

    def test_cache_setup_for_uv(self):
        """Test cache setup when uv is selected."""
        with (
            patch('llmb_install.environment.cache.os.makedirs') as mock_makedirs,
            patch('llmb_install.environment.cache.os.environ', {}) as mock_env,
        ):

            setup_cache_directories('/test/install', 'uv')

            # Should set both PIP and UV cache directories
            self.assertEqual(mock_env['PIP_CACHE_DIR'], '/test/install/.cache/pip')
            self.assertEqual(mock_env['UV_CACHE_DIR'], '/test/install/.cache/uv')

            # Should create both directories
            mock_makedirs.assert_any_call('/test/install/.cache/pip', exist_ok=True)
            mock_makedirs.assert_any_call('/test/install/.cache/uv', exist_ok=True)

    def test_cache_setup_for_venv(self):
        """Test cache setup when venv is selected (no UV cache)."""
        with (
            patch('llmb_install.environment.cache.os.makedirs') as mock_makedirs,
            patch('llmb_install.environment.cache.os.environ', {}) as mock_env,
        ):

            setup_cache_directories('/test/install', 'venv')

            # Should only set PIP cache directory
            self.assertEqual(mock_env['PIP_CACHE_DIR'], '/test/install/.cache/pip')
            self.assertNotIn('UV_CACHE_DIR', mock_env)

            # Should only create pip directory
            mock_makedirs.assert_called_once_with('/test/install/.cache/pip', exist_ok=True)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for virtual environment cleanup functionality in the installer."""

import os
import subprocess
import unittest
from unittest.mock import patch

from llmb_install.downloads.image import fetch_container_images
from llmb_install.environment.detector import detect_virtual_environment
from llmb_install.environment.venv_manager import get_clean_environment_for_subprocess


class TestVirtualEnvironmentDetection(unittest.TestCase):
    """Test virtual environment detection functionality."""

    def test_detect_conda_environment(self):
        """Test detection of conda environment."""
        with patch.dict(os.environ, {'CONDA_PREFIX': '/home/user/miniconda3/envs/test'}):
            self.assertTrue(detect_virtual_environment())

        with patch.dict(os.environ, {'CONDA_DEFAULT_ENV': 'test_env'}):
            self.assertTrue(detect_virtual_environment())

    def test_detect_venv_environment(self):
        """Test detection of venv environment."""
        with patch.dict(os.environ, {'VIRTUAL_ENV': '/home/user/venvs/test_venv'}):
            self.assertTrue(detect_virtual_environment())

    def test_detect_no_environment(self):
        """Test detection when not in a virtual environment."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(detect_virtual_environment())


class TestEnvironmentCleaning(unittest.TestCase):
    """Test environment cleaning functionality."""

    def setUp(self):
        """Set up test environment."""
        self.base_path = '/usr/local/bin:/usr/bin:/bin'
        self.conda_path = '/home/user/miniconda3/envs/test/bin'
        self.venv_path = '/home/user/venvs/test_venv/bin'

    def test_clean_conda_environment(self):
        """Test cleaning of conda environment variables."""
        conda_env = {
            'PATH': f"{self.conda_path}:{self.base_path}",
            'CONDA_PREFIX': '/home/user/miniconda3/envs/test',
            'CONDA_DEFAULT_ENV': 'test',
            'CONDA_PROMPT_MODIFIER': '(test)',
            'CONDA_SHLVL': '1',
            'CONDA_PYTHON_EXE': '/home/user/miniconda3/bin/python',
            'CONDA_EXE': '/home/user/miniconda3/bin/conda',
            'PYTHONPATH': '/some/python/path',
            'HOME': '/home/user',
        }

        with patch.dict(os.environ, conda_env):
            clean_env = get_clean_environment_for_subprocess()

            # Check that conda variables are removed
            self.assertNotIn('CONDA_PREFIX', clean_env)
            self.assertNotIn('CONDA_DEFAULT_ENV', clean_env)
            self.assertNotIn('CONDA_PROMPT_MODIFIER', clean_env)
            self.assertNotIn('CONDA_SHLVL', clean_env)
            self.assertNotIn('CONDA_PYTHON_EXE', clean_env)
            self.assertNotIn('CONDA_EXE', clean_env)

            # Check that Python variables are removed
            self.assertNotIn('PYTHONPATH', clean_env)

            # Check that conda bin path is removed from PATH
            self.assertNotIn(self.conda_path, clean_env['PATH'])
            self.assertIn('/usr/local/bin', clean_env['PATH'])

            # Check that other variables are preserved
            self.assertEqual(clean_env['HOME'], '/home/user')

    def test_clean_venv_environment(self):
        """Test cleaning of venv environment variables."""
        venv_env = {
            'PATH': f"{self.venv_path}:{self.base_path}",
            'VIRTUAL_ENV': '/home/user/venvs/test_venv',
            'VIRTUAL_ENV_PROMPT': '(test_venv)',
            'PYTHONHOME': '/usr',
            'HOME': '/home/user',
        }

        with patch.dict(os.environ, venv_env):
            clean_env = get_clean_environment_for_subprocess()

            # Check that venv variables are removed
            self.assertNotIn('VIRTUAL_ENV', clean_env)
            self.assertNotIn('VIRTUAL_ENV_PROMPT', clean_env)

            # Check that Python variables are removed
            self.assertNotIn('PYTHONHOME', clean_env)

            # Check that venv path is removed from PATH
            self.assertNotIn(self.venv_path, clean_env['PATH'])
            self.assertIn('/usr/local/bin', clean_env['PATH'])

            # Check that other variables are preserved
            self.assertEqual(clean_env['HOME'], '/home/user')

    def test_clean_environment_preserves_system_paths(self):
        """Test that system paths are preserved during cleaning."""
        system_paths = ['/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin', '/opt/slurm/bin']

        conda_env = {
            'PATH': f"/home/user/miniconda3/envs/test/bin:{':'.join(system_paths)}",
            'CONDA_PREFIX': '/home/user/miniconda3/envs/test',
        }

        with patch.dict(os.environ, conda_env):
            clean_env = get_clean_environment_for_subprocess()

            # Check that all system paths are preserved
            for path in system_paths:
                self.assertIn(path, clean_env['PATH'])

    def test_clean_environment_conda_bin_only(self):
        """Test that only the specific conda env bin directory is removed."""
        # This tests the new behavior where we only remove CONDA_PREFIX/bin
        conda_env = {
            'PATH': (
                "/home/user/miniconda3/envs/test/bin:"
                "/home/user/miniconda3/bin:"  # Base conda installation
                "/usr/local/bin:"
                "/usr/bin"
            ),
            'CONDA_PREFIX': '/home/user/miniconda3/envs/test',
        }

        with patch.dict(os.environ, conda_env):
            clean_env = get_clean_environment_for_subprocess()

            # Check that only the active env bin is removed
            self.assertNotIn('/home/user/miniconda3/envs/test/bin', clean_env['PATH'])
            # But the base conda bin should still be there (user might need it)
            self.assertIn('/home/user/miniconda3/bin', clean_env['PATH'])
            # System paths preserved
            self.assertIn('/usr/local/bin', clean_env['PATH'])
            self.assertIn('/usr/bin', clean_env['PATH'])

    def test_conda_auto_activate_forced_false(self):
        """Test that CONDA_AUTO_ACTIVATE is forced to false for conda environments."""
        conda_env = {
            'PATH': f"{self.conda_path}:{self.base_path}",
            'CONDA_PREFIX': '/home/user/miniconda3/envs/test',
            'CONDA_DEFAULT_ENV': 'test',
            'CONDA_AUTO_ACTIVATE': 'true',  # Set to true initially
            'HOME': '/home/user',
        }

        with patch.dict(os.environ, conda_env):
            clean_env = get_clean_environment_for_subprocess()

            # Check that CONDA_AUTO_ACTIVATE is forced to false
            self.assertEqual(clean_env['CONDA_AUTO_ACTIVATE'], 'false')

        # Test case where CONDA_AUTO_ACTIVATE is not originally set
        conda_env_no_auto_activate = {
            'PATH': f"{self.conda_path}:{self.base_path}",
            'CONDA_PREFIX': '/home/user/miniconda3/envs/test',
            'CONDA_DEFAULT_ENV': 'test',
            'HOME': '/home/user',
        }

        with patch.dict(os.environ, conda_env_no_auto_activate):
            clean_env = get_clean_environment_for_subprocess()

            # Check that CONDA_AUTO_ACTIVATE is set to false even when not originally present
            self.assertEqual(clean_env['CONDA_AUTO_ACTIVATE'], 'false')


class TestFetchContainerImagesIntegration(unittest.TestCase):
    """Test integration of environment cleaning in fetch_container_images."""

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_fetch_with_conda_environment_slurm(self, mock_makedirs, mock_exists, mock_run):
        """Test fetching images via SLURM from conda environment."""
        mock_exists.return_value = False

        images = {'test/image:latest': 'test_image.sqsh'}
        slurm_info = {
            'slurm': {'account': 'test_account', 'cpu_partition': 'test_partition', 'cpu_partition_gres': None}
        }

        with patch.dict(os.environ, {'CONDA_PREFIX': '/home/user/miniconda3/envs/test'}):
            with patch('llmb_install.downloads.image.detect_virtual_environment', return_value='conda'):
                with patch('llmb_install.downloads.image.get_clean_environment_for_subprocess') as mock_clean_env:
                    mock_clean_env.return_value = {'PATH': '/usr/bin:/bin'}

                    fetch_container_images(images, '/test/install', 'x86_64', 'slurm', slurm_info)

                    # Check that subprocess.run was called with clean environment
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args

                    # Check that the command is simple srun + enroot (no bash -c wrapper)
                    cmd = call_args[0][0]
                    self.assertEqual(cmd[0], 'srun')
                    self.assertIn('enroot', cmd)
                    self.assertNotIn('bash', cmd)
                    self.assertNotIn('-c', cmd)

                    # Check that clean environment was used
                    self.assertEqual(call_args[1]['env'], {'PATH': '/usr/bin:/bin'})

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_fetch_with_venv_environment_local(self, mock_makedirs, mock_exists, mock_run):
        """Test fetching images locally from venv environment."""
        mock_exists.return_value = False

        images = {'test/image:latest': 'test_image.sqsh'}
        slurm_info = {'slurm': {}}

        with patch.dict(os.environ, {'VIRTUAL_ENV': '/home/user/venvs/test'}):
            with patch('llmb_install.downloads.image.detect_virtual_environment', return_value='venv'):
                fetch_container_images(images, '/test/install', 'x86_64', 'local', slurm_info)

                # Check that subprocess.run was called without environment modification for local
                mock_run.assert_called_once()
                call_args = mock_run.call_args

                # For local execution, env should be None
                self.assertIsNone(call_args[1]['env'])


class TestErrorHandling(unittest.TestCase):
    """Test error handling and user feedback."""

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('builtins.print')
    def test_error_message_with_venv(self, mock_print, mock_makedirs, mock_exists, mock_run):
        """Test that helpful error message is shown when running from venv."""
        mock_exists.return_value = False
        mock_run.side_effect = subprocess.CalledProcessError(1, 'cmd')

        images = {'test/image:latest': 'test_image.sqsh'}
        slurm_info = {'slurm': {'account': 'test_account', 'cpu_partition': 'test_partition'}}

        with patch.dict(os.environ, {'VIRTUAL_ENV': '/home/user/venvs/test'}):
            with patch('llmb_install.downloads.image.detect_virtual_environment', return_value='venv'):
                with self.assertRaises(SystemExit):
                    fetch_container_images(images, '/test/install', 'x86_64', 'slurm', slurm_info)

                # Check that the architecture mismatch warning was printed
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                self.assertTrue(
                    any("Architecture mismatch" in str(call) for call in print_calls),
                    "Expected architecture mismatch warning not found",
                )


if __name__ == '__main__':
    unittest.main()

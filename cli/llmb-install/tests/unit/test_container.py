# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for container modules."""

import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from llmb_install.downloads.image import (
    _generate_image_filename,
    fetch_container_images,
    get_required_images,
    is_enroot_installed,
)


class TestEnroot(unittest.TestCase):
    """Test Enroot utilities."""

    @patch('shutil.which')
    def test_is_enroot_installed_true(self, mock_which):
        """Test enroot detection when installed."""
        mock_which.return_value = '/usr/bin/enroot'
        self.assertTrue(is_enroot_installed())
        mock_which.assert_called_once_with('enroot')

    @patch('shutil.which')
    def test_is_enroot_installed_false(self, mock_which):
        """Test enroot detection when not installed."""
        mock_which.return_value = None
        self.assertFalse(is_enroot_installed())
        mock_which.assert_called_once_with('enroot')


class TestImageManagement(unittest.TestCase):
    """Test container image management functions."""

    def test_generate_image_filename_regular_tag(self):
        """Test filename generation for regular image tags."""
        # Test basic image with tag
        image_url = "nvcr.io/nvidia/nemo:25.02.01"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "nvidia+nemo+25.02.01.sqsh")

        # Test with different registry
        image_url = "docker.io/library/ubuntu:20.04"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "library+ubuntu+20.04.sqsh")

    def test_generate_image_filename_sha_digest(self):
        """Test filename generation for SHA digest."""
        image_url = "nvcr.io/nvidia/nemo:sha256:abc123def456789012345678901234567890123456789012345678901234567890"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "nvidia+nemo+sha256-abc123def456.sqsh")

    def test_generate_image_filename_latest(self):
        """Test filename generation for images without tag."""
        image_url = "nvcr.io/nvidia/nemo"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "nvidia+nemo+latest.sqsh")

    def test_generate_image_filename_with_hash_separator(self):
        """Test filename generation with # separator in URL."""
        image_url = "nvcr.io#nvidia/nemo:25.02.01"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "nvidia+nemo+25.02.01.sqsh")

    def test_generate_image_filename_fallback(self):
        """Test filename generation fallback for malformed URLs."""
        # Test with malformed URL that will trigger fallback
        image_url = "invalid_url_format"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "invalid_url_format.sqsh")

        # Test with URL that gets parsed normally (not fallback)
        image_url = "registry.io/org:name/image:tag"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "org:name+image+tag.sqsh")

        # Test actual fallback with truly malformed URL
        image_url = "malformed::url::with:::colons"
        filename = _generate_image_filename(image_url)
        self.assertEqual(filename, "malformed__url__with___colons.sqsh")

    def test_get_required_images_string_format(self):
        """Test getting required images with string format."""
        workloads = {'workload1': {'container': {'images': 'nvcr.io/nvidia/pytorch:24.01-py3'}}}
        selected_keys = ['workload1']

        images = get_required_images(workloads, selected_keys)

        expected_filename = "nvidia+pytorch+24.01-py3.sqsh"
        self.assertEqual(len(images), 1)
        self.assertIn('nvcr.io/nvidia/pytorch:24.01-py3', images)
        self.assertEqual(images['nvcr.io/nvidia/pytorch:24.01-py3'], expected_filename)

    def test_get_required_images_list_format(self):
        """Test getting required images with list format."""
        workloads = {
            'workload1': {'container': {'images': ['nvcr.io/nvidia/pytorch:24.01-py3', 'nvcr.io/nvidia/nemo:25.02.01']}}
        }
        selected_keys = ['workload1']

        images = get_required_images(workloads, selected_keys)

        self.assertEqual(len(images), 2)
        self.assertIn('nvcr.io/nvidia/pytorch:24.01-py3', images)
        self.assertIn('nvcr.io/nvidia/nemo:25.02.01', images)

    def test_get_required_images_dict_format(self):
        """Test getting required images with dict format."""
        workloads = {
            'workload1': {
                'container': {
                    'images': [
                        {'url': 'nvcr.io/nvidia/pytorch:24.01-py3', 'name': 'custom_pytorch'},
                        {
                            'url': 'nvcr.io/nvidia/nemo:25.02.01'
                            # No name - should use generated filename
                        },
                    ]
                }
            }
        }
        selected_keys = ['workload1']

        images = get_required_images(workloads, selected_keys)

        self.assertEqual(len(images), 2)
        self.assertEqual(images['nvcr.io/nvidia/pytorch:24.01-py3'], 'custom_pytorch.sqsh')
        self.assertIn('nvcr.io/nvidia/nemo:25.02.01', images)

    def test_get_required_images_deduplication(self):
        """Test that duplicate images are deduplicated by filename."""
        workloads = {
            'workload1': {'container': {'images': ['nvcr.io/nvidia/pytorch:24.01-py3']}},
            'workload2': {
                'container': {'images': ['nvcr.io#nvidia#pytorch:24.01-py3']}  # Different URL format, same image
            },
        }
        selected_keys = ['workload1', 'workload2']

        images = get_required_images(workloads, selected_keys)

        # Should have only one image since they have the same generated filename
        self.assertEqual(len(images), 1)

    def test_get_required_images_no_container_config(self):
        """Test getting required images when workload has no container config."""
        workloads = {
            'workload1': {
                'setup': {'dependencies': {'pip': ['torch']}}
                # No container section
            }
        }
        selected_keys = ['workload1']

        images = get_required_images(workloads, selected_keys)

        self.assertEqual(len(images), 0)


class TestImageFetching(unittest.TestCase):
    """Test container image fetching functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        import shutil

        self.addCleanup(lambda: shutil.rmtree(self.temp_dir))

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('llmb_install.environment.detector.detect_virtual_environment')
    def test_fetch_container_images_local(self, mock_detect_venv, mock_makedirs, mock_exists, mock_run):
        """Test fetching container images locally."""
        mock_detect_venv.return_value = False
        mock_exists.return_value = False
        mock_run.return_value = Mock(returncode=0)

        images = {'nvcr.io/nvidia/pytorch:24.01-py3': 'nvidia+pytorch+24.01-py3.sqsh'}
        slurm_info = {'slurm': {'account': 'test', 'cpu_partition': 'cpu'}}

        with patch('builtins.print'):
            fetch_container_images(images, self.temp_dir, 'x86_64', 'local', slurm_info)

        # Verify enroot import was called
        enroot_calls = [call for call in mock_run.call_args_list if 'enroot' in str(call)]
        self.assertEqual(len(enroot_calls), 1)

        call_args = enroot_calls[0][0][0]  # Get the command arguments
        self.assertIn('enroot', call_args)
        self.assertIn('import', call_args)
        self.assertIn('-a', call_args)
        self.assertIn('x86_64', call_args)

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('llmb_install.environment.detector.detect_virtual_environment')
    def test_fetch_container_images_slurm(self, mock_detect_venv, mock_makedirs, mock_exists, mock_run):
        """Test fetching container images via SLURM."""
        mock_detect_venv.return_value = False
        mock_exists.return_value = False
        mock_run.return_value = Mock(returncode=0)

        images = {'nvcr.io/nvidia/pytorch:24.01-py3': 'nvidia+pytorch+24.01-py3.sqsh'}
        slurm_info = {'slurm': {'account': 'test_account', 'cpu_partition': 'cpu_partition', 'cpu_partition_gres': '1'}}

        with patch('builtins.print'):
            fetch_container_images(images, self.temp_dir, 'x86_64', 'slurm', slurm_info)

        # Verify srun + enroot command was called
        srun_calls = [call for call in mock_run.call_args_list if 'srun' in str(call)]
        self.assertEqual(len(srun_calls), 1)

        call_args = srun_calls[0][0][0]  # Get the command arguments
        self.assertIn('srun', call_args)
        self.assertIn('--account=test_account', call_args)
        self.assertIn('--partition=cpu_partition', call_args)
        self.assertIn('--gpus-per-node=1', call_args)
        self.assertIn('enroot', call_args)

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_fetch_container_images_skip_existing(self, mock_makedirs, mock_exists, mock_run):
        """Test that existing images are skipped."""
        mock_exists.return_value = True  # Image already exists

        images = {'nvcr.io/nvidia/pytorch:24.01-py3': 'nvidia+pytorch+24.01-py3.sqsh'}
        slurm_info = {'slurm': {'account': 'test', 'cpu_partition': 'cpu'}}

        with patch('builtins.print') as mock_print:
            fetch_container_images(images, self.temp_dir, 'x86_64', 'local', slurm_info)

        # Should not call subprocess.run for downloading
        mock_run.assert_not_called()

        # Should print skip message
        skip_calls = [call for call in mock_print.call_args_list if 'Skipping' in str(call)]
        self.assertTrue(len(skip_calls) > 0)

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('os.symlink')
    @patch('os.path.abspath')
    @patch('os.path.lexists')
    def test_fetch_container_images_with_shared_dir(
        self, mock_lexists, mock_abspath, mock_symlink, mock_makedirs, mock_exists, mock_run
    ):
        """Test fetching container images with shared directory."""
        mock_exists.return_value = False
        mock_lexists.return_value = False
        mock_abspath.return_value = '/shared/images/test.sqsh'
        mock_run.return_value = Mock(returncode=0)

        images = {'nvcr.io/nvidia/pytorch:24.01-py3': 'test.sqsh'}
        slurm_info = {'slurm': {'account': 'test', 'cpu_partition': 'cpu'}}
        shared_dir = '/shared/images'

        with patch('builtins.print'):
            fetch_container_images(images, self.temp_dir, 'x86_64', 'local', slurm_info, shared_images_dir=shared_dir)

        # Verify symlink was created
        mock_symlink.assert_called_once()
        symlink_call = mock_symlink.call_args[0]
        self.assertEqual(symlink_call[0], '/shared/images/test.sqsh')

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_fetch_container_images_failure(self, mock_makedirs, mock_exists, mock_run):
        """Test handling of image fetch failures."""
        mock_exists.return_value = False
        mock_run.side_effect = subprocess.CalledProcessError(1, 'enroot')

        images = {'nvcr.io/nvidia/pytorch:24.01-py3': 'nvidia+pytorch+24.01-py3.sqsh'}
        slurm_info = {'slurm': {'account': 'test', 'cpu_partition': 'cpu'}}

        with self.assertRaises(SystemExit):
            with patch('builtins.print'):
                fetch_container_images(images, self.temp_dir, 'x86_64', 'local', slurm_info)


if __name__ == '__main__':
    unittest.main()

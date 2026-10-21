# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.workload module."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmb_install.core.workload import (
    build_workload_dict,
    filter_tools_from_workload_list,
    find_metadata_files,
    get_setup_tasks,
    parse_metadata_file,
    run_setup_tasks,
)


class TestWorkloadMetadata(unittest.TestCase):
    """Test workload metadata functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        import shutil

        self.addCleanup(lambda: shutil.rmtree(self.temp_dir))

    def test_find_metadata_files(self):
        """Test finding metadata.yaml files while excluding deprecated directories."""
        # Create test directory structure
        (Path(self.temp_dir) / "workload1").mkdir()
        (Path(self.temp_dir) / "workload1" / "metadata.yaml").touch()

        (Path(self.temp_dir) / "deprecated" / "workload2").mkdir(parents=True)
        (Path(self.temp_dir) / "deprecated" / "workload2" / "metadata.yaml").touch()

        (Path(self.temp_dir) / "workload3" / "deprecated").mkdir(parents=True)
        (Path(self.temp_dir) / "workload3" / "deprecated" / "metadata.yaml").touch()

        (Path(self.temp_dir) / "workload4").mkdir()
        (Path(self.temp_dir) / "workload4" / "metadata.yaml").touch()

        # Test find_metadata_files
        metadata_files = find_metadata_files(self.temp_dir)

        # Should find 2 files (workload1 and workload4), not deprecated ones
        self.assertEqual(len(metadata_files), 2)
        found_parents = {f.parent.name for f in metadata_files}
        self.assertEqual(found_parents, {"workload1", "workload4"})

    def test_parse_metadata_file(self):
        """Test parsing a metadata.yaml file."""
        metadata_content = """
general:
  workload_type: inference
  workload: llama
container:
  images:
    - "nvcr.io/nvidia/pytorch:24.01-py3"
setup:
  dependencies:
    pip:
      - torch
"""
        metadata_file = Path(self.temp_dir) / "metadata.yaml"
        metadata_file.write_text(metadata_content)

        result = parse_metadata_file(metadata_file)

        self.assertEqual(result['general']['workload_type'], 'inference')
        self.assertEqual(result['general']['workload'], 'llama')
        self.assertIn('images', result['container'])
        self.assertIn('dependencies', result['setup'])

    def test_build_workload_dict(self):
        """Test building workload dictionary from metadata files."""
        # Create test workload directories and metadata
        workload1_dir = Path(self.temp_dir) / "workload1"
        workload1_dir.mkdir()

        workload1_metadata = workload1_dir / "metadata.yaml"
        workload1_metadata.write_text("""
general:
  workload_type: inference
  workload: llama
container:
  images: []
setup:
  dependencies: {}
""")

        workload2_dir = Path(self.temp_dir) / "workload2"
        workload2_dir.mkdir()

        workload2_metadata = workload2_dir / "metadata.yaml"
        workload2_metadata.write_text("""
general:
  workload_type: tools
  workload: nvtop
""")

        # Build workload dict
        workload_dict = build_workload_dict(self.temp_dir)

        # Verify results
        self.assertEqual(len(workload_dict), 2)
        self.assertIn('inference_llama', workload_dict)
        self.assertIn('tools_nvtop', workload_dict)

        llama_workload = workload_dict['inference_llama']
        self.assertEqual(llama_workload['general']['workload_type'], 'inference')
        self.assertEqual(llama_workload['general']['workload'], 'llama')
        self.assertEqual(llama_workload['path'], str(workload1_dir))

    def test_build_workload_dict_with_invalid_metadata(self):
        """Test build_workload_dict handles invalid metadata gracefully."""
        # Create invalid metadata file
        invalid_dir = Path(self.temp_dir) / "invalid"
        invalid_dir.mkdir()
        invalid_metadata = invalid_dir / "metadata.yaml"
        invalid_metadata.write_text("invalid: yaml: content: [")

        # Create valid metadata file
        valid_dir = Path(self.temp_dir) / "valid"
        valid_dir.mkdir()
        valid_metadata = valid_dir / "metadata.yaml"
        valid_metadata.write_text("""
general:
  workload_type: inference
  workload: test
""")

        with patch('builtins.print') as mock_print:
            workload_dict = build_workload_dict(self.temp_dir)

        # Should have only the valid workload
        self.assertEqual(len(workload_dict), 1)
        self.assertIn('inference_test', workload_dict)

        # Should have printed error for invalid file
        mock_print.assert_called()
        error_call = [call for call in mock_print.call_args_list if 'Error processing' in str(call)]
        self.assertTrue(len(error_call) > 0)

    def test_filter_tools_from_workload_list(self):
        """Test filtering tools from workload list."""
        workloads = {
            'inference_llama': {'general': {'workload_type': 'inference', 'workload': 'llama'}},
            'tools_nvtop': {'general': {'workload_type': 'tools', 'workload': 'nvtop'}},
            'tools_htop': {'general': {'workload_type': 'tools', 'workload': 'htop'}},
            'finetune_gpt': {'general': {'workload_type': 'finetune', 'workload': 'gpt'}},
        }

        tools = filter_tools_from_workload_list(workloads)

        self.assertEqual(len(tools), 2)
        self.assertIn('tools_nvtop', tools)
        self.assertIn('tools_htop', tools)
        self.assertNotIn('inference_llama', tools)
        self.assertNotIn('finetune_gpt', tools)

    def test_get_setup_tasks(self):
        """Test getting setup tasks from workload data."""
        # Test with explicit tasks
        workload_with_tasks = {
            'setup': {
                'tasks': [
                    {'name': 'install_deps', 'cmd': 'pip install torch'},
                    {'name': 'download_model', 'cmd': 'wget model.tar.gz'},
                ]
            }
        }

        tasks = get_setup_tasks(workload_with_tasks)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]['name'], 'install_deps')
        self.assertEqual(tasks[1]['name'], 'download_model')

        # Test with no tasks
        workload_no_tasks = {'setup': {}}
        tasks = get_setup_tasks(workload_no_tasks)
        self.assertEqual(len(tasks), 0)

        # Test with no setup section
        workload_no_setup = {}
        tasks = get_setup_tasks(workload_no_setup)
        self.assertEqual(len(tasks), 0)


class TestRunSetupTasksWithoutSlurm(unittest.TestCase):
    """A Kubernetes install has no Slurm configuration to draw on.

    Slurm-backed setup tasks (sbatch/srun/nemo2) must be skipped with a warning
    rather than crashing on the missing ``slurm_info``; local tasks still run.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        import shutil

        self.addCleanup(lambda: shutil.rmtree(self.temp_dir))

    def test_slurm_backed_tasks_are_skipped_and_local_tasks_run(self):
        workload = {
            'path': self.temp_dir,
            'setup': {
                'tasks': [
                    {'name': 'srun_task', 'cmd': './needs_slurm.sh', 'job_type': 'srun'},
                    {'name': 'local_task', 'cmd': 'echo hi', 'job_type': 'local'},
                ]
            },
        }

        executed = []
        with patch('llmb_install.core.workload.subprocess.run', side_effect=lambda cmd, **kwargs: executed.append(cmd)):
            run_setup_tasks(
                'w1',
                workload,
                venv_path=None,
                venv_type='uv',
                install_path=self.temp_dir,
                slurm_info={},
                gpu_type='gb300',
                node_architecture='aarch64',
            )

        self.assertEqual(executed, [['echo', 'hi']])


if __name__ == '__main__':
    unittest.main()

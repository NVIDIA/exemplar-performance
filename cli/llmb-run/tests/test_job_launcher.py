# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from llmb_run.config_manager import ClusterConfig
from llmb_run.constants import GPU_TYPE_TO_NUM_GPUS
from llmb_run.env_args import (
    LLMB_CONTAINER_ENV,
    MBRIDGE_EXTRA_ARGS_VAR,
    NEMO_ENV_OVERRIDE_VAR,
    apply_mbridge_extra_args_contract,
    apply_sbatch_explicit_env_contract,
    build_mbridge_extra_args,
    build_nemo_env_override_flags,
    build_nemo_workload_args,
)
from llmb_run.job_launcher import (
    ConfiguredSbatchLauncher,
    Nemo2Launcher,
    SbatchLauncher,
    SrtSlurmLauncher,
    _compute_segment_size,
    create_experiment_directory,
    create_launcher,
    filter_nemo2_status_instructions,
    get_venv_environment,
    parse_nemo2_output,
    parse_srt_slurm_json_output,
    should_enable_vboost,
)
from llmb_run.run_config import create_llmb_config
from llmb_run.slurm_args import ADDITIONAL_SLURM_PARAMS_KEY, build_cli_slurm_args
from llmb_run.tasks import WorkloadTask


class _FakeClock:
    """Stands in for job_launcher's `time` module: sleep() advances time() instead of waiting."""

    def __init__(self):
        self.now = 1_700_000_000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr('llmb_run.job_launcher.time', clock)
    return clock


def _cluster_config(
    *,
    llmb_install: str = "/tmp/llmb_install",
    llmb_repo: str = "/tmp/llmb_repo",
    gpu_type: str = "h100",
    cluster_name: str | None = None,
    node_architecture: str | None = None,
    slurm: dict | None = None,
    workloads_config: dict | None = None,
    environment: dict | None = None,
    cwd: str | None = None,
    nvl_domain: int | None = None,
    nvl_block_sizes: list[int] | None = None,
) -> ClusterConfig:
    launcher = {
        "llmb_install": llmb_install,
        "llmb_repo": llmb_repo,
        "gpu_type": gpu_type,
    }
    if cluster_name is not None:
        launcher["cluster_name"] = cluster_name
    raw = {
        "launcher": launcher,
        "slurm": slurm if slurm is not None else {"gpu_partition": "batch"},
    }
    if node_architecture is not None:
        raw["install"] = {"node_architecture": node_architecture}
    if workloads_config is not None:
        raw["workloads"] = {"config": workloads_config}
    if environment is not None:
        raw["environment"] = environment
    if cwd is not None:
        raw["cwd"] = cwd
    if nvl_domain is not None:
        raw["nvl_domain"] = nvl_domain
    if nvl_block_sizes is not None:
        raw["nvl_block_sizes"] = nvl_block_sizes

    return ClusterConfig.from_dict(raw)


def test_parse_nemo2_output_success():
    """Test successful parsing of Nemo2 launcher output."""

    successful_output = """[NeMo W 2025-06-12 12:57:25 nemo_logging:405] The package `megatron.core` was not imported in this environment which is needed for SSMs.
[NeMo I 2025-06-12 12:57:26 nemo_logging:393] Using /mnt/llmb/workloads/pretraining_nemotron/NeMo/scripts/performance/recommended_model_configs/model_configs_h100.csv for loading default recommended model configs
───────────────────────────────────────────────────────────────────────────────────── Entering Experiment pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64 with id: pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64_1749758246 ─────────────────────────────────────────────────────────────────────────────────────
[12:57:26] Launching job pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64 for experiment pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64                                                                                                                                                           experiment.py:763
[12:57:26] INFO     Launched app: slurm_tunnel://nemo_run/3530909                                                                                                                                                                                                                                                                     launcher.py:111
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── Detaching from Experiment pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64_1749758246. ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Task specific cleanup won't be run.                                                                                                                                                                                                                                                                                     experiment.py:1129
           Ephemeral logs and artifacts may be lost.

Experiment Status for pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64_1749758246

Task 0: pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64
- Status: SUBMITTED
- Executor: SlurmExecutor on @localhost
- Job id: 3530909
- Local Directory: /mnt/llmb/workloads/pretraining_nemotron/experiments/pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64/pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64_1749758246/pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64


# The experiment was run with the following tasks: ['pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64']
# You can inspect and reconstruct this experiment at a later point in time using:
experiment = run.Experiment.from_id("pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64_1749758246")
experiment.status() # Gets the overall status
experiment.logs("pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64") # Gets the log for the provided task
experiment.cancel("pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64") # Cancels the provided task if still running"""

    job_id, local_dir, error_msg = parse_nemo2_output(successful_output)

    # Assertions for successful parsing
    assert job_id == "3530909"
    assert (
        local_dir
        == "/mnt/llmb/workloads/pretraining_nemotron/experiments/pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64/pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64_1749758246/pretrain_nemotron4_340b_bf16_gpus256_tp8_pp8_cp1_vp12_mbs1_gbs64"
    )
    assert error_msg is None


def test_parse_nemo2_output_failure_no_task_headers():
    """Test failure case when no Task headers are found in output."""

    failed_output = """[NeMo I 2025-06-12 15:27:28 nemo_logging:393] Using /mnt/llmb/workloads/pretraining_nemotron/NeMo/scripts/performance/recommended_model_configs/model_configs_h100.csv for loading default recommended model configs
────────────────────────────────────────────────────────────────────────────────────── Entering Experiment pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64 with id: pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1749767248 ──────────────────────────────────────────────────────────────────────────────────────
[15:27:28] Launching job pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64 for experiment pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64                                                                                                                                                             experiment.py:763
           Error running job pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64: Encountered a bad command exit code!                                                                                                                                                                                                  experiment.py:782

           Command: 'sbatch --requeue --parsable
           /mnt/llmb/workloads/pretraining_nemotron/experiments/pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64/pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1749767248/pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_sbatch.sh
           '

           Exit code: 1

           Stdout:



           Stderr:

           sbatch: error: Cannot find GPU specification, you may not submit a job not requesting GPUs in a non-CPU partition, partition: batch
           sbatch: error: Batch job submission failed: Unspecified error


─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── Detaching from Experiment pretrain_nemotron4_15b_fp8_gpus16_tp2_pp1_cp1_vpNone_mbs2_gbs64_1749767248. ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
           Task specific cleanup won't be run.                                                                                                                                                                                                                                                                                     experiment.py:1129
           Ephemeral logs and artifacts may be lost."""

    job_id, local_dir, error_msg = parse_nemo2_output(failed_output)

    # Assertions for failed parsing
    assert job_id is None
    assert local_dir is None
    assert error_msg == "No Task headers found in output"


def test_parse_nemo2_output_multiple_tasks():
    """Test failure case when multiple tasks are found."""

    multiple_tasks_output = """[NeMo I 2025-06-12 12:57:26 nemo_logging:393] Using model configs
───────────────────────────────────────────────────────────────────────────────────── Entering Experiment test_experiment ─────────────────────────────────────────────────────────────────────────────────────
[12:57:26] Launching job test_job for experiment test_experiment                                                                                                                                                           experiment.py:763
[12:57:26] INFO     Launched app: slurm_tunnel://nemo_run/1234567                                                                                                                                                                     launcher.py:111

Experiment Status for test_experiment

Task 0: test_task_0
- Status: SUBMITTED
- Executor: SlurmExecutor on @localhost
- Job id: 1234567
- Local Directory: /path/to/task0

Task 1: test_task_1  
- Status: SUBMITTED
- Executor: SlurmExecutor on @localhost
- Job id: 1234568
- Local Directory: /path/to/task1"""

    job_id, local_dir, error_msg = parse_nemo2_output(multiple_tasks_output)

    # Assertions for multiple tasks error
    assert job_id is None
    assert local_dir is None
    assert error_msg == "Multiple tasks found (2 tasks: ['0', '1']). Expected only Task 0."


def test_parse_nemo2_output_job_id_without_directory():
    """Test failure case when job ID is found but no local directory."""

    incomplete_output = """[NeMo I 2025-06-12 12:57:26 nemo_logging:393] Using model configs

Experiment Status for test_experiment

Task 0: test_task_0
- Status: SUBMITTED
- Executor: SlurmExecutor on @localhost
- Job id: 1234567"""

    job_id, local_dir, error_msg = parse_nemo2_output(incomplete_output)

    # Assertions for incomplete task info
    assert job_id is None
    assert local_dir is None
    assert error_msg == "Found job ID 1234567 but no local directory - this should not happen"


def test_create_llmb_config_success():
    """Test successful creation of llmb-config.yaml file."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock task
        task = WorkloadTask(
            workload_key="pretraining_nemotron",
            model_size="340b",
            dtype="fp8",
            scale=256,
            profile=True,
            env_overrides={"DEBUG": "true"},
            model_overrides={"tp": 8},
        )

        # Create mock config
        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_nemotron": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {"CUSTOM_VAR": "value"},
                }
            },
            cwd=temp_dir,
        )

        # Create mock workloads
        workloads = {
            'pretraining_nemotron': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                },
                'workload': 'nemotron4',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
            }
        }

        # Test with workdir (Nemo2 launcher case)
        job_id = "12345"
        workdir = temp_dir

        config_file_path = create_llmb_config(task, job_id, workdir, config, workloads)

        # Verify file was created
        assert config_file_path is not None
        assert os.path.exists(config_file_path)
        assert config_file_path == os.path.join(temp_dir, 'llmb-config_12345.yaml')

        # Read and verify the content
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Verify job_info section
        assert config_data['job_info']['job_id'] == "12345"
        assert 'launch_time' in config_data['job_info']

        # Verify workload_info section
        assert config_data['workload_info']['framework'] == "nemo2"
        assert config_data['workload_info']['gsw_version'] == "25.07"
        assert config_data['workload_info']['fw_version'] == "25.04.01"
        assert config_data['workload_info']['workload_type'] == "pretrain"
        assert config_data['workload_info']['synthetic_dataset'] is True

        # Verify model_info section
        assert config_data['model_info']['model_name'] == "nemotron4"
        assert config_data['model_info']['model_size'] == "340b"
        assert config_data['model_info']['dtype'] == "fp8"
        assert config_data['model_info']['scale'] == 256
        assert config_data['model_info']['gpu_type'] == "h100"

        # Verify cluster_info section
        assert config_data['cluster_info']['cluster_name'] == "test_cluster"
        assert config_data['cluster_info']['gpus_per_node'] == "8"
        assert config_data['cluster_info']['llmb_install'] == "/path/to/llmb_install"
        assert config_data['cluster_info']['llmb_repo'] == "/path/to/llmb_repo"
        assert config_data['cluster_info']['slurm_account'] == "test_account"
        assert config_data['cluster_info']['slurm_gpu_partition'] == "test_partition"

        # Verify container_info section
        assert config_data['container_info']['images'] == ['nvcr.io#nvidia/nemo:25.04.01']

        # Verify job_config section
        assert config_data['job_config']['profile_enabled'] is True
        assert config_data['job_config']['env_overrides'] == {"DEBUG": "true"}
        assert config_data['job_config']['model_overrides'] == {"tp": 8}


def test_create_llmb_config_no_workdir():
    """Test creation of llmb-config.yaml when workdir is None (sbatch launcher case)."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock task
        task = WorkloadTask(
            workload_key="pretraining_llama3.1", model_size="405b", dtype="bf16", scale=512, profile=False
        )

        # Create mock config
        config = _cluster_config(
            llmb_install=f"{temp_dir}/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="gb200",
            cluster_name="test_cluster",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            cwd=temp_dir,
        )

        # Create mock workloads
        workloads = {
            'pretraining_llama3.1': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                },
                'workload': 'llama3.1',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
            }
        }

        # Test with no workdir (sbatch launcher case)
        job_id = "67890"
        workdir = None

        config_file_path = create_llmb_config(task, job_id, workdir, config, workloads)

        # Verify file was created in current working directory
        assert config_file_path is not None
        assert os.path.exists(config_file_path)

        assert config_file_path == os.path.join(
            temp_dir, 'llmb_install/workloads/pretraining_llama3.1/llmb-config_67890.yaml'
        )

        # Read and verify the content
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Verify key fields
        assert config_data['job_info']['job_id'] == "67890"
        assert 'launch_time' in config_data['job_info']
        assert config_data['workload_info']['framework'] == "nemo2"
        assert config_data['workload_info']['gsw_version'] == "25.07"
        assert config_data['workload_info']['fw_version'] == "25.04.01"
        assert config_data['workload_info']['workload_type'] == "pretrain"
        assert config_data['workload_info']['synthetic_dataset'] is True
        assert config_data['model_info']['model_name'] == "llama3.1"
        assert config_data['model_info']['model_size'] == "405b"
        assert config_data['model_info']['dtype'] == "bf16"
        assert config_data['model_info']['scale'] == 512
        assert config_data['model_info']['gpu_type'] == "gb200"
        assert config_data['cluster_info']['cluster_name'] == "test_cluster"
        assert config_data['cluster_info']['gpus_per_node'] == "8"
        assert config_data['container_info']['images'] == ['nvcr.io#nvidia/nemo:25.04.01']
        assert config_data['job_config']['profile_enabled'] is False


def test_create_llmb_config_with_model_attribute():
    """Test that general.model attribute overrides workload for model_name."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock task
        task = WorkloadTask(
            workload_key="pretraining_llama3.3", model_size="70b", dtype="fp8", scale=128, profile=False
        )

        # Create mock config
        config = _cluster_config(
            llmb_install=temp_dir,
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            cwd=temp_dir,
        )

        # Create mock workloads with model attribute set
        workloads = {
            'pretraining_llama3.3': {
                'metadata': {
                    'general': {
                        'framework': 'nemo2',
                        'gsw_version': '25.07',
                        'workload': 'llama3.3',
                        'model': 'llama3',  # Different from workload
                    },
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                },
                'workload': 'llama3.3',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
            }
        }

        # Test with workdir
        job_id = "99999"
        workdir = temp_dir

        config_file_path = create_llmb_config(task, job_id, workdir, config, workloads)

        # Verify file was created
        assert config_file_path is not None
        assert os.path.exists(config_file_path)

        # Read and verify the content
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Verify that model_name uses the 'model' attribute instead of 'workload'
        assert config_data['model_info']['model_name'] == "llama3"
        assert config_data['model_info']['model_size'] == "70b"
        assert config_data['model_info']['dtype'] == "fp8"
        assert config_data['model_info']['scale'] == 128
        assert config_data['model_info']['gpu_type'] == "h100"


def test_create_llmb_config_failure():
    """Test failure case when config file cannot be created."""

    # Create mock task
    task = WorkloadTask(workload_key="pretraining_nemotron", model_size="340b", dtype="fp8", scale=256)

    # Create mock config with invalid workdir
    config = _cluster_config(
        llmb_install="/path/to/llmb_install",
        llmb_repo="/path/to/llmb_repo",
        gpu_type="h100",
        cwd="/nonexistent/directory",
    )

    # Create mock workloads
    workloads = {
        'pretraining_nemotron': {
            'metadata': {
                'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
            },
            'workload': 'nemotron4',
            'workload_type': 'pretrain',
            'synthetic_dataset': True,
        }
    }

    # Test with invalid workdir
    job_id = "12345"
    workdir = "/nonexistent/directory"

    with patch('os.makedirs') as mock_makedirs, patch('builtins.open', new_callable=Mock) as mock_open:
        mock_makedirs.side_effect = OSError("Permission denied or invalid path")
        config_file_path = create_llmb_config(task, job_id, workdir, config, workloads)

        # Verify function returns None on failure
        assert config_file_path is None
        mock_makedirs.assert_called_once_with(workdir, exist_ok=True)
        mock_open.assert_not_called()


def test_sbatch_launcher_create_llmb_config():
    """Test that SbatchLauncher correctly calls create_llmb_config with workdir=None."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock task
        task = WorkloadTask(
            workload_key="pretraining_llama3.1",
            model_size="405b",
            dtype="bf16",
            scale=512,
            profile=False,
            env_overrides={"NCCL_DEBUG": "INFO"},
            model_overrides={},
        )

        # Create mock config
        config = _cluster_config(
            llmb_install=f"{temp_dir}/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="gb200",
            cluster_name="test_cluster",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            cwd=temp_dir,
        )

        # Create mock workloads
        workloads = {
            'pretraining_llama3.1': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                },
                'workload': 'llama3.1',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        # Create SbatchLauncher instance
        from llmb_run.job_launcher import SbatchLauncher

        launcher = SbatchLauncher(config, workloads)

        # Mock the subprocess.run to return a successful job submission
        from unittest.mock import Mock, patch

        mock_result = Mock()
        mock_result.stdout = "12345\n"  # Simulate job ID returned by sbatch
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result):
            # Call the launch method
            slurm_job = launcher.launch(task)

        # Verify the job was "submitted" successfully
        assert slurm_job.job_id == 12345
        assert slurm_job.job_workdir is None  # SbatchLauncher always returns None for workdir

        # Verify that llmb-config.yaml was created in the current working directory (cwd)
        config_file_path = os.path.join(temp_dir, 'llmb_install/workloads/pretraining_llama3.1/llmb-config_12345.yaml')
        assert os.path.exists(config_file_path)

        # Read and verify the content
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Verify key fields match the SbatchLauncher scenario
        assert config_data['job_info']['job_id'] == "12345"
        assert 'launch_time' in config_data['job_info']
        assert config_data['workload_info']['framework'] == "nemo2"
        assert config_data['workload_info']['gsw_version'] == "25.07"
        assert config_data['workload_info']['fw_version'] == "25.04.01"
        assert config_data['workload_info']['workload_type'] == "pretrain"
        assert config_data['workload_info']['synthetic_dataset'] is True
        assert config_data['model_info']['model_name'] == "llama3.1"
        assert config_data['model_info']['model_size'] == "405b"
        assert config_data['model_info']['dtype'] == "bf16"
        assert config_data['model_info']['scale'] == 512
        assert config_data['model_info']['gpu_type'] == "gb200"
        assert config_data['cluster_info']['cluster_name'] == "test_cluster"
        assert config_data['cluster_info']['gpus_per_node'] == "8"
        assert config_data['cluster_info']['llmb_install'] == f"{temp_dir}/llmb_install"
        assert config_data['cluster_info']['llmb_repo'] == "/path/to/llmb_repo"
        assert config_data['cluster_info']['slurm_account'] == "test_account"
        assert config_data['cluster_info']['slurm_gpu_partition'] == "test_partition"
        assert config_data['container_info']['images'] == ['nvcr.io#nvidia/nemo:25.04.01']
        assert config_data['job_config']['profile_enabled'] is False
        assert config_data['job_config']['env_overrides'] == {"NCCL_DEBUG": "INFO"}


def test_nemo2_launcher_create_llmb_config():
    """Test that Nemo2Launcher correctly calls create_llmb_config with workdir from launcher output."""

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock task
        task = WorkloadTask(
            workload_key="pretraining_nemotron",
            model_size="340b",
            dtype="fp8",
            scale=256,
            profile=True,
            env_overrides={"DEBUG": "true"},
            model_overrides={"tp": 8},
        )

        # Create mock config
        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_nemotron": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {"CUSTOM_VAR": "value"},
                }
            },
            cwd=temp_dir,
        )

        # Create mock workloads
        workloads = {
            'pretraining_nemotron': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                },
                'workload': 'nemotron4',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        from unittest.mock import Mock, patch

        from llmb_run.job_launcher import Nemo2Launcher

        experiment_dir = "/path/to/experiment/directory"
        config_file_path = os.path.join(experiment_dir, 'llmb-config_67890.yaml')

        # Patch os.path.exists and os.makedirs
        with (
            patch('os.path.exists') as mock_exists,
            patch('os.makedirs') as mock_makedirs,
            patch('builtins.open', create=True) as mock_open,
        ):
            # Only return True for the python3 path and experiment directory
            def exists_side_effect(path):
                if path == '/path/to/venv/bin/python3':
                    return True
                if path == experiment_dir:
                    return True
                if path == config_file_path:
                    return True
                return False

            mock_exists.side_effect = exists_side_effect
            mock_makedirs.return_value = None  # Do nothing

            launcher = Nemo2Launcher(config, workloads)

            # Mock the subprocess.run to return a successful job submission
            mock_result = Mock()
            mock_result.stdout = f"""[NeMo I 2025-06-12 12:57:26 nemo_logging:393] Using model configs\n───────────────────────────────────────────────────────────────────────────────────── Entering Experiment test_experiment ─────────────────────────────────────────────────────────────────────────────────────\n[12:57:26] Launching job test_job for experiment test_experiment                                                                                                                                                           experiment.py:763\n[12:57:26] INFO     Launched app: slurm_tunnel://nemo_run/67890                                                                                                                                                                     launcher.py:111\n\nExperiment Status for test_experiment\n\nTask 0: test_task_0\n- Status: SUBMITTED\n- Executor: SlurmExecutor on @localhost\n- Job id: 67890\n- Local Directory: {experiment_dir}"""
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result):
                # Call the launch method
                slurm_job = launcher.launch(task)

        # Verify the job was "submitted" successfully
        assert slurm_job.job_id == 67890
        assert slurm_job.job_workdir == experiment_dir  # Nemo2Launcher returns the workdir
        # We can't check file content since open is mocked, but we can check the call
        mock_open.assert_any_call(config_file_path, 'w')


def test_model_overrides_set_as_env_vars():
    """Test that model_overrides are passed as uppercased env vars to the subprocess."""

    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_llama3.1",
            model_size="405b",
            dtype="bf16",
            scale=512,
            profile=False,
            env_overrides={},
            model_overrides={"tp": 8, "cp": 2},
        )

        config = _cluster_config(
            llmb_install=f"{temp_dir}/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="gb200",
            cluster_name="test_cluster",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_llama3.1': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                },
                'workload': 'llama3.1',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        launcher = SbatchLauncher(config, workloads)

        mock_result = Mock()
        mock_result.stdout = "12345\n"
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            launcher.launch(task)

        # Get the env dict that was passed to subprocess.run
        call_env = mock_run.call_args.kwargs['env']

        # model_overrides should be uppercased env vars
        assert call_env['TP'] == '8'
        assert call_env['CP'] == '2'

        # Old OPTIMIZATION_CODE/NAME should NOT be present
        assert 'OPTIMIZATION_CODE' not in call_env
        assert 'OPTIMIZATION_NAME' not in call_env


def test_sbatch_launcher_exports_container_env_contract():
    """SbatchLauncher should export explicit container-env keys alongside env overrides."""
    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_llama3.1",
            model_size="405b",
            dtype="bf16",
            scale=512,
            profile=False,
            env_overrides={"HF_TOKEN": "abc123", "NCCL_DEBUG": "INFO"},
            explicit_env_overrides={"HF_TOKEN": "abc123", "NCCL_DEBUG": "INFO"},
        )

        config = _cluster_config(
            llmb_install=f"{temp_dir}/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="gb200",
            cluster_name="test_cluster",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_llama3.1': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                },
                'workload': 'llama3.1',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        launcher = SbatchLauncher(config, workloads)

        mock_result = Mock()
        mock_result.stdout = "12345\n"
        mock_result.returncode = 0

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            launcher.launch(task)

        call_env = mock_run.call_args.kwargs['env']
        assert call_env['HF_TOKEN'] == 'abc123'
        assert call_env['NCCL_DEBUG'] == 'INFO'
        assert call_env[LLMB_CONTAINER_ENV] == 'HF_TOKEN,NCCL_DEBUG'
        assert call_env['LLMB_REPO'] == config.llmb_repo
        assert call_env['LLMB_WORKLOAD'] == f"{config.llmb_install}/workloads/{task.workload_key}"
        assert call_env['LLMB_RECIPE_ROOT'] == workloads[task.workload_key]['dir']


def test_sbatch_explicit_env_contract_merges_existing_container_env():
    """Explicit env keys should append to, not replace, a configured container env list."""
    env = {LLMB_CONTAINER_ENV: 'HF_HOME,HF_TOKEN'}

    apply_sbatch_explicit_env_contract(env, {'HF_TOKEN': 'abc123', 'NCCL_DEBUG': 'INFO'})

    assert env[LLMB_CONTAINER_ENV] == 'HF_HOME,HF_TOKEN,NCCL_DEBUG'


def test_create_launcher_sbatch():
    """Test that create_launcher returns SbatchLauncher for sbatch launcher type."""
    config = _cluster_config(llmb_install="/test/install")
    workloads = {}

    launcher = create_launcher('sbatch', config, workloads)

    assert isinstance(launcher, SbatchLauncher)


def test_create_launcher_nemo():
    """Test that create_launcher returns Nemo2Launcher for nemo launcher type."""
    config = _cluster_config(llmb_install="/test/install")
    workloads = {}

    launcher = create_launcher('nemo', config, workloads)

    assert isinstance(launcher, Nemo2Launcher)


def test_create_launcher_megatron_bridge():
    """Test that create_launcher returns Nemo2Launcher for megatron_bridge launcher type."""
    config = _cluster_config(llmb_install="/test/install")
    workloads = {}

    launcher = create_launcher('megatron_bridge', config, workloads)

    assert isinstance(launcher, Nemo2Launcher)


def test_create_launcher_srt_slurm():
    launcher = create_launcher("srt_slurm", _cluster_config(llmb_install="/test/install"), {})
    assert isinstance(launcher, SrtSlurmLauncher)


def test_parse_srt_slurm_json_output():
    record = parse_srt_slurm_json_output(
        '{"status":"submitted","slurm_job_id":"42042","output_dir":"/outputs/42042"}\n'
    )
    assert record["slurm_job_id"] == "42042"


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("not-json\n", "Invalid JSON"),
        ('{"status":"error","error":"sbatch failed"}\n', "sbatch failed"),
        (
            '{"status":"submitted","slurm_job_id":"1","output_dir":"/o/1"}\n'
            '{"status":"submitted","slurm_job_id":"2","output_dir":"/o/2"}\n',
            "exactly one",
        ),
        ('{"status":"submitted","slurm_job_id":"1"}\n', "output_dir"),
    ],
)
def test_parse_srt_slurm_json_output_rejects_invalid_contract(output, message):
    with pytest.raises(ValueError, match=message):
        parse_srt_slurm_json_output(output)


@pytest.mark.parametrize("llmb_config_path", ["/tmp/llmb-config.yaml", None])
def test_srt_slurm_launcher_submits_real_orchestration_job_without_sbatch(
    tmp_path, monkeypatch, caplog, llmb_config_path
):
    llmb_install = tmp_path / "install"
    workload_dir = llmb_install / "workloads" / "inference_test"
    srt_root = workload_dir / "srt-slurm"
    workload_recipe_dir = tmp_path / "repo" / "recipe"
    recipe_dir = workload_recipe_dir / "recipes"
    venv_path = llmb_install / "venvs" / "inference_test_venv"
    srtctl = venv_path / "bin" / "srtctl"
    image_path = llmb_install / "images" / "test-runtime.sqsh"
    model_path = workload_dir / "test-model"
    srt_root.mkdir(parents=True)
    recipe_dir.mkdir(parents=True)
    srtctl.parent.mkdir(parents=True)
    image_path.parent.mkdir(parents=True)
    image_path.touch()
    model_path.mkdir()
    (srt_root / "srtslurm.yaml").write_text(
        "default_account: stale-account\n"
        "default_partition: stale-partition\n"
        "gpus_per_node: 8\n"
        "default_time_limit: 08:00:00\n"
        "network_interface: ibp1s0\n"
        "use_gpus_per_node_directive: false\n"
        "model_paths:\n"
        "  operator-model: /models/operator\n"
        "containers:\n"
        "  operator-runtime: /images/operator.sqsh\n"
    )
    (recipe_dir / "one-point.yaml").write_text(
        "base:\n"
        "  name: one-point\n"
        "  resources:\n"
        "    gpus_per_node: 4\n"
        "    prefill_nodes: 1\n"
        "    decode_nodes: 9\n"
        "override_validated: {}\n"
    )
    srtctl.write_text("#!/bin/sh\n")
    srtctl.chmod(0o755)

    config = _cluster_config(
        llmb_install=str(llmb_install),
        llmb_repo=str(tmp_path / "repo"),
        gpu_type="gb300",
        slurm={"account": "account", "gpu_partition": "batch", "gpu_gres": 4},
        workloads_config={"inference_test": {"venv_path": str(venv_path), "venv_type": "uv", "environment": {}}},
    )
    workloads = {
        "inference_test": {
            "dir": str(workload_recipe_dir),
            "metadata": {
                "container": {"images": [{"url": "registry/test-runtime:1", "name": "test-runtime"}]},
                "downloads": {
                    "huggingface": {
                        "repos": [
                            {
                                "repo_id": "org/test-model",
                                "repo_type": "model",
                                "revision": "0123456789abcdef",
                                "target": "workload",
                                "name": "test-model",
                            }
                        ]
                    }
                },
                "run": {
                    "launcher_type": "srt_slurm",
                    "srt_slurm": {
                        "recipe": "recipes/one-point.yaml",
                        "selector": "override_validated",
                    },
                },
            },
        }
    }
    task = WorkloadTask("inference_test", "744b", "fp8", 40, profile=False)
    output_root = tmp_path / "outputs"
    response = Mock(
        returncode=0,
        stdout='{"status":"submitted","slurm_job_id":"650470","output_dir":"' + str(output_root / "650470") + '"}\n',
        stderr="submitted",
    )
    login_venv = tmp_path / "x86-login-venv"
    monkeypatch.setenv("VIRTUAL_ENV", str(login_venv))
    monkeypatch.setenv("PYTHONPATH", str(login_venv / "lib"))
    monkeypatch.setenv("PATH", f"{login_venv / 'bin'}:/usr/bin:/bin")

    with (
        patch("llmb_run.job_launcher.create_experiment_directory", return_value=str(output_root)) as mock_experiment,
        patch("subprocess.run", return_value=response) as mock_run,
        patch("llmb_run.job_launcher.create_llmb_config", return_value=llmb_config_path) as mock_config,
    ):
        job = SrtSlurmLauncher(config, workloads).launch(task)

    assert job.job_id == 650470
    assert job.job_workdir == str((output_root / "650470").resolve())
    assert job.llmb_config_path == llmb_config_path
    command = mock_run.call_args.args[0]
    assert command[0] == str(srtctl)
    assert command[1:3] == ["apply", "-f"]
    assert command[3] == str(recipe_dir / "one-point.yaml") + ":override_validated"
    assert "sbatch" not in command
    assert command[-2:] == ["-y", "--json"]
    assert mock_run.call_args.kwargs["cwd"] == srt_root
    submission_env = mock_run.call_args.kwargs["env"]
    assert submission_env["SRTSLURM_CONFIG"] == str(srt_root / "srtslurm.yaml")
    assert submission_env["LLMB_RECIPE_ROOT"] == str(workload_recipe_dir)
    assert "VIRTUAL_ENV" not in submission_env
    assert "PYTHONPATH" not in submission_env
    assert submission_env["PATH"] == "/usr/bin:/bin"
    assert mock_experiment.call_args.kwargs["description"] == "inference_test_744b_fp8_recipe_one-point"
    generated_config = yaml.safe_load((srt_root / "srtslurm.yaml").read_text())
    assert generated_config["default_account"] == "account"
    assert generated_config["default_partition"] == "batch"
    assert generated_config["gpus_per_node"] == 4
    assert generated_config["srtctl_root"] == str(srt_root)
    assert generated_config["model_paths"] == {
        "operator-model": "/models/operator",
        "test-model": str(model_path),
    }
    assert generated_config["containers"] == {
        "operator-runtime": "/images/operator.sqsh",
        "test-runtime": str(image_path),
    }
    assert generated_config["default_time_limit"] == "08:00:00"
    assert generated_config["network_interface"] == "ibp1s0"
    assert generated_config["use_gpus_per_node_directive"] is False
    mock_config.assert_called_once()
    if llmb_config_path is None:
        assert "job 650470 was submitted, but LLMB config generation failed" in caplog.text


def test_srt_slurm_launcher_reports_process_failure_before_parsing_json(tmp_path, caplog):
    llmb_install = tmp_path / "install"
    workload_dir = llmb_install / "workloads" / "inference_test"
    srt_root = workload_dir / "srt-slurm"
    recipe_dir = tmp_path / "repo" / "recipe"
    recipe = recipe_dir / "recipes" / "one-point.yaml"
    venv_path = llmb_install / "venvs" / "inference_test_venv"
    srtctl = venv_path / "bin" / "srtctl"
    srt_root.mkdir(parents=True)
    recipe.parent.mkdir(parents=True)
    srtctl.parent.mkdir(parents=True)
    recipe.write_text("resources:\n  gpus_per_node: 4\n  nodes: 1\n")
    srtctl.write_text("#!/bin/sh\n")
    srtctl.chmod(0o755)

    config = _cluster_config(
        llmb_install=str(llmb_install),
        llmb_repo=str(tmp_path / "repo"),
        gpu_type="gb300",
        slurm={"account": "account", "gpu_partition": "batch"},
        workloads_config={"inference_test": {"venv_path": str(venv_path), "venv_type": "uv"}},
    )
    workloads = {
        "inference_test": {
            "dir": str(recipe_dir),
            "metadata": {
                "run": {
                    "launcher_type": "srt_slurm",
                    "srt_slurm": {"recipe": "recipes/one-point.yaml"},
                }
            },
        }
    }
    response = Mock(returncode=17, stdout="not-json", stderr="scheduler rejected request")

    with (
        patch.object(SrtSlurmLauncher, "_write_srtslurm_config", return_value=srt_root / "srtslurm.yaml"),
        patch("llmb_run.job_launcher.create_experiment_directory", return_value=str(tmp_path / "outputs")),
        patch("subprocess.run", return_value=response),
        caplog.at_level("ERROR"),
    ):
        job = SrtSlurmLauncher(config, workloads).launch(
            WorkloadTask("inference_test", "744b", "fp8", 4, profile=False)
        )

    assert job.job_id is None
    assert "srtctl apply exited with status 17: scheduler rejected request" in caplog.text
    assert "Invalid JSON" not in caplog.text


def test_srt_slurm_config_uses_gpu_type_fallback_without_gres(tmp_path):
    llmb_install = tmp_path / "install"
    workload_dir = llmb_install / "workloads" / "inference_test"
    srt_root = workload_dir / "srt-slurm"
    model_path = workload_dir / "test-model"
    image_path = llmb_install / "images" / "test-runtime.sqsh"
    srt_root.mkdir(parents=True)
    model_path.mkdir()
    image_path.parent.mkdir(parents=True)
    image_path.touch()

    config = _cluster_config(
        llmb_install=str(llmb_install),
        gpu_type="gb300",
        slurm={"account": "account", "gpu_partition": "batch"},
    )
    metadata = {
        "container": {"images": [{"url": "registry/test-runtime:1", "name": "test-runtime"}]},
        "downloads": {
            "huggingface": {
                "repos": [{"target": "workload", "name": "test-model"}],
            }
        },
    }

    config_path = SrtSlurmLauncher(config, {})._write_srtslurm_config(srt_root, metadata, workload_dir)
    generated_config = yaml.safe_load(config_path.read_text())

    assert generated_config["gpus_per_node"] == GPU_TYPE_TO_NUM_GPUS["gb300"]
    assert generated_config["use_gpus_per_node_directive"] is False


def test_srt_slurm_launcher_rejects_scale_that_does_not_match_recipe(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "base:\n  resources:\n    gpus_per_node: 4\n    prefill_nodes: 1\n    decode_nodes: 9\noverride_validated: {}\n"
    )

    assert SrtSlurmLauncher._recipe_gpu_count(recipe, "override_validated") == 40
    with pytest.raises(
        ValueError,
        match=r"recipe requests 40 GPUs, but LLMB --scale is 1.*--scale 40",
    ):
        SrtSlurmLauncher._validate_recipe_scale(recipe, "override_validated", 1)


def test_srt_slurm_launcher_counts_top_level_trtllm_recipe(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("resources:\n  gpus_per_node: 4\n  prefill_nodes: 6\n  decode_nodes: 4\n")

    assert SrtSlurmLauncher._recipe_gpu_count(recipe) == 40
    SrtSlurmLauncher._validate_recipe_scale(recipe, None, 40)


def test_srt_slurm_launcher_rejects_selector_for_top_level_recipe(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("resources:\n  gpus_per_node: 4\n  nodes: 1\n")

    with pytest.raises(ValueError, match="requires a recipe with a 'base' mapping"):
        SrtSlurmLauncher._recipe_gpu_count(recipe, "override")


def test_create_launcher_unknown_type():
    """Test that create_launcher raises ValueError for unknown launcher type."""
    config = _cluster_config(llmb_install="/test/install")
    workloads = {}

    with pytest.raises(ValueError, match="Unknown launcher type: invalid_type"):
        create_launcher('invalid_type', config, workloads)


def test_create_experiment_directory_success():
    """Test successful experiment directory creation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workload_dir = os.path.join(temp_dir, "workloads", "test_workload")
        os.makedirs(workload_dir)

        result = create_experiment_directory(
            workload_dir=workload_dir, workload_key="test_workload", model_size="8b", dtype="bf16", scale=8
        )

        assert os.path.exists(result)
        assert "experiments/test_workload_8b_bf16_gpus8" in result
        # Verify timestamp directory name is numeric
        timestamp_dir = os.path.basename(result)
        assert timestamp_dir.isdigit()


def test_create_experiment_directory_supports_launcher_specific_description(tmp_path):
    workload_dir = tmp_path / "workloads" / "inference_test"
    workload_dir.mkdir(parents=True)

    result = create_experiment_directory(
        workload_dir=str(workload_dir),
        workload_key="inference_test",
        model_size="744b",
        dtype="fp8",
        scale=1,
        description="inference_test_744b_fp8_recipe_one-point",
    )

    assert "experiments/inference_test_744b_fp8_recipe_one-point" in result
    assert "gpus1" not in result


def test_create_experiment_directory_collision_retry(fake_clock):
    """Test directory creation retries on collision."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workload_dir = os.path.join(temp_dir, "workloads", "test_workload")
        os.makedirs(workload_dir)

        first_dir = create_experiment_directory(
            workload_dir=workload_dir, workload_key="test_workload", model_size="8b", dtype="bf16", scale=8
        )

        # Immediately create second - tests retry logic
        second_dir = create_experiment_directory(
            workload_dir=workload_dir, workload_key="test_workload", model_size="8b", dtype="bf16", scale=8
        )

        assert first_dir != second_dir
        assert os.path.isdir(first_dir)
        assert os.path.isdir(second_dir)
        assert int(os.path.basename(second_dir)) == int(os.path.basename(first_dir)) + 1


def test_configured_sbatch_launcher_creates_directory():
    """Test ConfiguredSbatchLauncher creates experiment directory under llmb_install."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Script dir (where launch.sh lives) - can be separate from llmb_install
        script_dir = os.path.join(temp_dir, "llmb_repo", "test_workload")
        os.makedirs(script_dir)
        # Experiment output dir base (under $LLMB_INSTALL/workloads/)
        llmb_install = os.path.join(temp_dir, "llmb_install")
        llmb_workload = os.path.join(llmb_install, "workloads", "test_workload")
        os.makedirs(llmb_workload)

        config = _cluster_config(
            llmb_install=llmb_install,
            gpu_type="h100",
            slurm={"account": "test_account", "gpu_partition": "test_partition"},
        )
        workloads = {'test_workload': {'dir': script_dir}}
        task = WorkloadTask(
            workload_key='test_workload',
            model_size='8b',
            dtype='bf16',
            scale=8,
            profile=False,
        )

        launcher = ConfiguredSbatchLauncher(config, workloads)

        mock_result = Mock()
        mock_result.stdout = "12345\n"

        with (
            patch('subprocess.run', return_value=mock_result) as mock_run,
            patch('llmb_run.job_launcher.create_llmb_config') as mock_config,
        ):
            slurm_job = launcher.launch(task)

        assert slurm_job.job_id == 12345
        assert slurm_job.job_workdir is not None
        assert os.path.isdir(slurm_job.job_workdir)
        # Experiment dir is under llmb_install, not script_dir
        assert slurm_job.job_workdir.startswith(llmb_install)
        assert "experiments/test_workload_8b_bf16_gpus8" in slurm_job.job_workdir
        # sbatch runs from script_dir (where launch.sh is)
        assert mock_run.call_args.kwargs['cwd'] == script_dir
        mock_config.assert_called_once()


def test_configured_sbatch_launcher_exports_container_env_contract():
    """ConfiguredSbatchLauncher should export explicit container-env keys."""
    with tempfile.TemporaryDirectory() as temp_dir:
        script_dir = os.path.join(temp_dir, "llmb_repo", "test_workload")
        os.makedirs(script_dir)
        llmb_install = os.path.join(temp_dir, "llmb_install")
        llmb_workload = os.path.join(llmb_install, "workloads", "test_workload")
        os.makedirs(llmb_workload)

        config = _cluster_config(
            llmb_install=llmb_install,
            gpu_type="h100",
            slurm={"account": "test_account", "gpu_partition": "test_partition"},
        )
        workloads = {'test_workload': {'dir': script_dir}}
        task = WorkloadTask(
            workload_key='test_workload',
            model_size='8b',
            dtype='bf16',
            scale=8,
            profile=False,
            env_overrides={"HF_TOKEN": "abc123"},
            explicit_env_overrides={"HF_TOKEN": "abc123"},
        )

        launcher = ConfiguredSbatchLauncher(config, workloads)

        mock_result = Mock()
        mock_result.stdout = "12345\n"

        with (
            patch('subprocess.run', return_value=mock_result) as mock_run,
            patch('llmb_run.job_launcher.create_llmb_config'),
        ):
            launcher.launch(task)

        call_env = mock_run.call_args.kwargs['env']
        assert call_env['HF_TOKEN'] == 'abc123'
        assert call_env[LLMB_CONTAINER_ENV] == 'HF_TOKEN'


def _launch_and_capture_env(config, workloads, task):
    launcher = ConfiguredSbatchLauncher(config, workloads)
    mock_result = Mock()
    mock_result.stdout = "12345\n"
    with (
        patch('subprocess.run', return_value=mock_result) as mock_run,
        patch('llmb_run.job_launcher.create_llmb_config'),
    ):
        launcher.launch(task)
    return mock_run.call_args.kwargs['env']


@pytest.mark.usefixtures('fake_clock')
def test_configured_sbatch_launcher_routes_gpu_less_workloads_to_the_cpu_target():
    """run.requires_gpus: false submits to slurm.cpu; without slurm.cpu that target is a copy of slurm.gpu."""
    with tempfile.TemporaryDirectory() as temp_dir:
        script_dir = os.path.join(temp_dir, "llmb_repo", "test_workload")
        os.makedirs(script_dir)
        llmb_install = os.path.join(temp_dir, "llmb_install")
        os.makedirs(os.path.join(llmb_install, "workloads", "test_workload"))
        task = WorkloadTask(workload_key='test_workload', model_size='8b', dtype='bf16', scale=1, profile=False)
        no_gpus = {'test_workload': {'dir': script_dir, 'metadata': {'run': {'requires_gpus': False}}}}
        default = {'test_workload': {'dir': script_dir, 'metadata': {'run': {}}}}

        both = _cluster_config(
            llmb_install=llmb_install,
            gpu_type="gb300",
            slurm={"account": "acct", "gpu_partition": "gpu_part", "gpu_gres": 4, "cpu_partition": "cpu_part"},
        )
        env = _launch_and_capture_env(both, no_gpus, task)
        assert env['SBATCH_PARTITION'] == 'cpu_part'
        assert 'SBATCH_GPUS_PER_NODE' not in env  # the CPU target carries no GPU request
        assert env['GPU_TYPE'] == 'gb300'  # the launch script still learns the SKU

        # no CPU partition configured -> the GPU partition, with its GPU request
        gpu_only = _cluster_config(
            llmb_install=llmb_install,
            gpu_type="gb300",
            slurm={"account": "acct", "gpu_partition": "gpu_part", "gpu_gres": 4},
        )
        env = _launch_and_capture_env(gpu_only, no_gpus, task)
        assert env['SBATCH_PARTITION'] == 'gpu_part' and env['SBATCH_GPUS_PER_NODE'] == '4'

        # workloads that do not declare requires_gpus need GPUs
        env = _launch_and_capture_env(both, default, task)
        assert env['SBATCH_PARTITION'] == 'gpu_part' and env['SBATCH_GPUS_PER_NODE'] == '4'

        # an explicit task override still wins
        forced = WorkloadTask(
            workload_key='test_workload',
            model_size='8b',
            dtype='bf16',
            scale=1,
            profile=False,
            env_overrides={"SBATCH_PARTITION": "gpu_part"},
            explicit_env_overrides={"SBATCH_PARTITION": "gpu_part"},
        )
        env = _launch_and_capture_env(both, no_gpus, forced)
        assert env['SBATCH_PARTITION'] == 'gpu_part'


def test_create_launcher_configured_sbatch():
    """Test factory function creates ConfiguredSbatchLauncher."""
    launcher = create_launcher('configured_sbatch', _cluster_config(), {})
    assert isinstance(launcher, ConfiguredSbatchLauncher)


class TestFilterNemo2StatusInstructions:
    """Tests for filter_nemo2_status_instructions pure function."""

    def test_removes_python_api_section(self):
        output = (
            "Some output\n"
            "# The experiment was run with the following tasks: ['task1']\n"
            "experiment.status()\n"
            "experiment.logs('task1')\n"
            "experiment.cancel('task1')\n"
            "More output"
        )
        result = filter_nemo2_status_instructions(output)
        assert "experiment.status()" not in result
        assert "experiment.cancel" not in result
        assert "Some output" in result
        assert "More output" in result

    def test_removes_cli_section(self):
        output = (
            "Launch output\n"
            "# You can inspect this experiment at a later point in time using the CLI as well:\n"
            "nemo experiment status test_exp\n"
            "nemo experiment logs test_exp task_0\n"
            "nemo experiment cancel test_exp\n"
            "Final output"
        )
        result = filter_nemo2_status_instructions(output)
        assert "nemo experiment status" not in result
        assert "nemo experiment cancel" not in result
        assert "Launch output" in result
        assert "Final output" in result

    def test_cleans_up_extra_newlines(self):
        output = "Line1\n\n\n\n\nLine2"
        result = filter_nemo2_status_instructions(output)
        assert "\n\n\n" not in result

    def test_empty_string(self):
        assert filter_nemo2_status_instructions("") == ""

    def test_no_status_sections(self):
        output = "Normal output\nWith multiple lines\nNo status instructions"
        result = filter_nemo2_status_instructions(output)
        assert result == output.strip()


class TestShouldEnableVboost:
    """Tests for should_enable_vboost pure function."""

    def test_eos_cluster_from_config(self):
        config = _cluster_config(cluster_name="eos")
        assert should_enable_vboost(config) is True

    def test_eos_cluster_case_insensitive(self):
        config = _cluster_config(cluster_name="EOS")
        assert should_enable_vboost(config) is True

    def test_non_eos_cluster(self):
        config = _cluster_config(cluster_name="cluster-a")
        assert should_enable_vboost(config) is False

    def test_no_cluster_name_no_slurm(self):
        """Without cluster_name and without SLURM, returns False."""
        config = _cluster_config(cluster_name=None)
        with patch('llmb_run.job_launcher.get_cluster_name', return_value=None):
            assert should_enable_vboost(config) is False

    def test_falls_back_to_slurm_detection(self):
        """When config has no cluster_name, falls back to get_cluster_name()."""
        config = _cluster_config(cluster_name=None)
        with patch('llmb_run.job_launcher.get_cluster_name', return_value="eos"):
            assert should_enable_vboost(config) is True


class TestGetVenvEnvironment:
    """Tests for get_venv_environment pure function."""

    @pytest.fixture
    def fake_venv(self, tmp_path):
        """Create a minimal fake venv directory with python3 binary."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").touch()
        return tmp_path

    def test_venv_type_sets_virtual_env(self, fake_venv):
        """venv type sets VIRTUAL_ENV env var."""
        env = get_venv_environment(str(fake_venv), env_type="venv")
        assert env["VIRTUAL_ENV"] == str(fake_venv)
        assert str(fake_venv / "bin") in env["PATH"]
        assert "PYTHONHOME" not in env

    def test_conda_type_sets_conda_prefix(self, fake_venv, monkeypatch):
        """conda type sets CONDA_PREFIX env var, not VIRTUAL_ENV."""
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        env = get_venv_environment(str(fake_venv), env_type="conda")
        assert env["CONDA_PREFIX"] == str(fake_venv)
        assert "VIRTUAL_ENV" not in env

    def test_uv_type_sets_virtual_env(self, fake_venv):
        """uv type sets VIRTUAL_ENV like venv."""
        env = get_venv_environment(str(fake_venv), env_type="uv")
        assert env["VIRTUAL_ENV"] == str(fake_venv)

    def test_missing_python3_raises(self, tmp_path):
        """Missing python3 in bin dir raises ValueError."""
        (tmp_path / "bin").mkdir()
        with pytest.raises(ValueError, match="python3 executable not found"):
            get_venv_environment(str(tmp_path))

    def test_removes_pythonhome(self, fake_venv, monkeypatch):
        """PYTHONHOME is removed from the environment."""
        monkeypatch.setenv("PYTHONHOME", "/some/path")
        env = get_venv_environment(str(fake_venv))
        assert "PYTHONHOME" not in env


class TestComputeSegmentSize:
    @pytest.mark.parametrize(
        "num_nodes,domain_nodes,expected",
        [
            (1, 18, 1),
            (2, 18, 2),
            (4, 18, 4),
            (8, 18, 8),
            (16, 18, 16),  # largest power-of-2 <= 18
            (17, 18, 17),  # prime <= 18, valid job size
            (18, 18, 18),  # exact domain size
            (32, 18, 16),  # greatest divisor <= 18
            (64, 18, 16),
            (128, 18, 16),
            (36, 18, 18),  # multiples of 18
            (36, 144, 36),  # larger domain allows larger segment
        ],
    )
    def test_segment_size(self, num_nodes, domain_nodes, expected):
        assert _compute_segment_size(num_nodes, domain_nodes) == expected


class TestConfiguredSbatchLauncherSegmentSize:
    """Integration tests for SBATCH_SEGMENT_SIZE in ConfiguredSbatchLauncher."""

    def _run_launch(
        self, gpu_type, scale=16, env_overrides=None, environment=None, nvl_domain=None, nvl_block_sizes=None
    ):
        """Launch a task and return the env dict passed to subprocess.run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            llmb_install = os.path.join(temp_dir, "llmb_install")
            script_dir = os.path.join(temp_dir, "workload")
            llmb_workload = os.path.join(llmb_install, "workloads", "test_workload")
            os.makedirs(script_dir)
            os.makedirs(llmb_workload)

            config = _cluster_config(
                llmb_install=llmb_install,
                gpu_type=gpu_type,
                slurm={"gpu_partition": "batch"},
                environment=environment,
                nvl_domain=nvl_domain,
                nvl_block_sizes=nvl_block_sizes,
            )
            workloads = {'test_workload': {'dir': script_dir}}
            task = WorkloadTask(
                workload_key='test_workload',
                model_size='8b',
                dtype='bf16',
                scale=scale,
                profile=False,
                env_overrides=env_overrides or {},
            )

            launcher = ConfiguredSbatchLauncher(config, workloads)

            mock_result = Mock()
            mock_result.stdout = "99999\n"

            with (
                patch('subprocess.run', return_value=mock_result) as mock_run,
                patch('llmb_run.job_launcher.create_llmb_config'),
            ):
                launcher.launch(task)

            return mock_run.call_args.args[0], mock_run.call_args.kwargs['env']

    @pytest.mark.parametrize("gpu_type", ["gb200", "gb300", "vr"])
    def test_nvl_multi_node_auto_detects_segment(self, gpu_type):
        """gb200/gb300/vr auto-detect segment and emit --segment flag."""
        scale = 16
        cmd, env = self._run_launch(gpu_type=gpu_type, scale=scale)
        gpus_per_node = GPU_TYPE_TO_NUM_GPUS[gpu_type]
        num_nodes = (scale + gpus_per_node - 1) // gpus_per_node
        domain_nodes = 72 // gpus_per_node
        expected_segment = str(_compute_segment_size(num_nodes, domain_nodes))
        assert f"--segment={expected_segment}" in cmd
        assert 'SBATCH_SEGMENT_SIZE' not in env

    def test_h100_does_not_set_segment(self):
        """h100 should NOT emit --segment."""
        cmd, env = self._run_launch(gpu_type='h100', scale=8)
        assert 'SBATCH_SEGMENT_SIZE' not in env
        assert not any(arg.startswith('--segment=') for arg in cmd)

    def test_config_nvl_domain_drives_segment(self):
        """config.nvl_domain=576 on gb200 gives a 144-node domain, not 18."""
        # 36 nodes: under the old 18-node domain the answer would be 18,
        # but a 144-node domain allows segment=36.
        cmd, env = self._run_launch(gpu_type='gb200', scale=36 * 4, nvl_domain=576)
        assert '--segment=36' in cmd
        assert env['NVL_DOMAIN'] == '576'

    def test_nvl_block_sizes_beat_domain_for_segment(self):
        """Recorded Slurm block sizes win over nvl_domain for --segment.

        nvl_domain=576 gives a 144-node domain (segment would be 36), but the
        recorded base block of 18 nodes clamps the segment to 18.
        """
        cmd, env = self._run_launch(gpu_type='gb200', scale=36 * 4, nvl_domain=576, nvl_block_sizes=[18, 72, 144])
        assert '--segment=18' in cmd
        assert env['NVL_DOMAIN'] == '576'

    def test_nvl_block_sizes_without_nvl_domain(self):
        """Block sizes alone still drive --segment when nvl_domain is absent.

        Without block sizes the domain would fall back to the gb200 default
        (72 GPUs = 18 nodes), so 36 nodes would still give 18; use a case
        where min(block_sizes) < default domain: [9] -> segment=9 vs 18.
        """
        cmd, env = self._run_launch(gpu_type='gb200', scale=36 * 4, nvl_block_sizes=[9, 144])
        assert '--segment=9' in cmd
        assert env['NVL_DOMAIN'] == '72'

    @pytest.mark.parametrize(
        "gpu_type,expected_nvl,expected_mbridge",
        [
            ('vr', '72', 'vr200'),
            ('h100', '8', 'h100'),
        ],
    )
    def test_gpu_env_vars(self, gpu_type, expected_nvl, expected_mbridge):
        """Launch env exports NVL_DOMAIN and MBRIDGE_GPU_TYPE per GPU type."""
        _, env = self._run_launch(gpu_type=gpu_type, scale=8)
        assert env['NVL_DOMAIN'] == expected_nvl
        assert env['MBRIDGE_GPU_TYPE'] == expected_mbridge

    def test_llmb_path_env(self):
        """ConfiguredSbatchLauncher exports the repository, recipe, and workload paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            llmb_install = os.path.join(temp_dir, 'llmb_install')
            script_dir = os.path.join(temp_dir, 'workload')
            os.makedirs(script_dir)
            os.makedirs(os.path.join(llmb_install, 'workloads', 'test_workload'))

            config = _cluster_config(llmb_install=llmb_install, gpu_type='h100', slurm={'gpu_partition': 'batch'})
            workloads = {'test_workload': {'dir': script_dir}}
            task = WorkloadTask(
                workload_key='test_workload',
                model_size='8b',
                dtype='bf16',
                scale=8,
                profile=False,
            )
            launcher = ConfiguredSbatchLauncher(config, workloads)
            mock_result = Mock()
            mock_result.stdout = '99999\n'

            with (
                patch('subprocess.run', return_value=mock_result) as mock_run,
                patch('llmb_run.job_launcher.create_llmb_config'),
            ):
                launcher.launch(task)

        env = mock_run.call_args.kwargs['env']
        assert env['LLMB_REPO'] == config.llmb_repo
        assert env['LLMB_WORKLOAD'] == f"{config.llmb_install}/workloads/{task.workload_key}"
        assert env['LLMB_RECIPE_ROOT'] == workloads[task.workload_key]['dir']

    def test_task_llmb_repo_override_wins(self):
        """Task environment overrides take precedence over the launcher repository path."""
        _, env = self._run_launch(gpu_type='h100', scale=8, env_overrides={'LLMB_REPO': '/custom'})
        assert env['LLMB_REPO'] == '/custom'

    def test_config_nvl_domain_overrides_default_env(self):
        """config.nvl_domain wins over the per-GPU default for NVL_DOMAIN."""
        _, env = self._run_launch(gpu_type='gb200', scale=16, nvl_domain=576)
        assert env['NVL_DOMAIN'] == '576'

    def test_os_env_takes_precedence(self, monkeypatch):
        """OS env SBATCH_SEGMENT_SIZE is read and emitted as --segment flag."""
        monkeypatch.setenv('SBATCH_SEGMENT_SIZE', '9')
        cmd, env = self._run_launch(gpu_type='gb200', scale=16)
        assert '--segment=9' in cmd
        assert 'SBATCH_SEGMENT_SIZE' not in env

    def test_task_env_overrides_takes_precedence(self):
        """task.env_overrides SBATCH_SEGMENT_SIZE is emitted as --segment flag."""
        cmd, env = self._run_launch(
            gpu_type='gb200',
            scale=16,
            env_overrides={'SBATCH_SEGMENT_SIZE': '2'},
        )
        assert '--segment=2' in cmd
        assert 'SBATCH_SEGMENT_SIZE' not in env

    def test_cli_segment_override_wins_over_existing_env_sources(self, monkeypatch):
        """Explicit CLI segment overrides should beat env-based segment settings."""
        monkeypatch.setenv('SBATCH_SEGMENT_SIZE', '9')
        task = WorkloadTask(
            workload_key='test_workload',
            model_size='8b',
            dtype='bf16',
            scale=16,
            profile=False,
            env_overrides={'SBATCH_SEGMENT_SIZE': '2'},
            slurm_args=build_cli_slurm_args(segment=5),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            llmb_install = os.path.join(temp_dir, "llmb_install")
            script_dir = os.path.join(temp_dir, "workload")
            llmb_workload = os.path.join(llmb_install, "workloads", "test_workload")
            os.makedirs(script_dir)
            os.makedirs(llmb_workload)

            config = _cluster_config(
                llmb_install=llmb_install,
                gpu_type='gb200',
                slurm={"gpu_partition": "batch"},
            )
            workloads = {'test_workload': {'dir': script_dir}}
            launcher = ConfiguredSbatchLauncher(config, workloads)

            mock_result = Mock()
            mock_result.stdout = "99999\n"

            with (
                patch('subprocess.run', return_value=mock_result) as mock_run,
                patch('llmb_run.job_launcher.create_llmb_config'),
            ):
                launcher.launch(task)

            cmd = mock_run.call_args.args[0]
            env = mock_run.call_args.kwargs['env']

        assert '--segment=5' in cmd
        assert 'SBATCH_SEGMENT_SIZE' not in env

    def test_segment_flag_before_launch_script(self):
        """--segment flag must appear before launch.sh in the sbatch command."""
        cmd, _env = self._run_launch(gpu_type='gb200', scale=16)
        segment_flags = [arg for arg in cmd if arg.startswith('--segment=')]
        assert segment_flags, "Expected --segment flag in cmd"
        segment_idx = cmd.index(segment_flags[0])
        launch_idx = next(i for i, arg in enumerate(cmd) if 'launch.sh' in arg)
        assert segment_idx < launch_idx


class TestLaunchScriptFromMetadata:
    """Launchers must use run.launch_script from metadata instead of hardcoding launch.sh."""

    def test_sbatch_launcher_uses_metadata_launch_script(self):
        """SbatchLauncher should use the launch_script from workload metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _cluster_config(
                llmb_install=os.path.join(temp_dir, "llmb_install"),
                gpu_type="h100",
                slurm={"gpu_partition": "batch"},
            )
            script_dir = os.path.join(temp_dir, "workload")
            os.makedirs(script_dir)
            os.makedirs(os.path.join(temp_dir, "llmb_install", "workloads", "test_wl"))

            workloads = {
                'test_wl': {
                    'dir': script_dir,
                    'metadata': {
                        'run': {'launcher_type': 'sbatch', 'launch_script': 'custom_launch.sh'},
                    },
                }
            }
            task = WorkloadTask(
                workload_key='test_wl',
                model_size='8b',
                dtype='bf16',
                scale=8,
                profile=False,
            )
            launcher = SbatchLauncher(config, workloads)
            mock_result = Mock()
            mock_result.stdout = "111\n"

            with (
                patch('subprocess.run', return_value=mock_result) as mock_run,
                patch('llmb_run.job_launcher.create_llmb_config'),
            ):
                launcher.launch(task)

            cmd = mock_run.call_args.args[0]
            assert 'custom_launch.sh' in cmd, f"Expected custom_launch.sh in cmd, got {cmd}"

    def test_configured_sbatch_launcher_uses_metadata_launch_script(self):
        """ConfiguredSbatchLauncher should use the launch_script from workload metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            llmb_install = os.path.join(temp_dir, "llmb_install")
            script_dir = os.path.join(temp_dir, "workload")
            os.makedirs(script_dir)
            os.makedirs(os.path.join(llmb_install, "workloads", "test_wl"))

            config = _cluster_config(
                llmb_install=llmb_install,
                gpu_type="h100",
                slurm={"gpu_partition": "batch"},
            )
            workloads = {
                'test_wl': {
                    'dir': script_dir,
                    'metadata': {
                        'run': {'launcher_type': 'configured_sbatch', 'launch_script': 'run_job.sh'},
                    },
                }
            }
            task = WorkloadTask(
                workload_key='test_wl',
                model_size='8b',
                dtype='bf16',
                scale=8,
                profile=False,
            )
            launcher = ConfiguredSbatchLauncher(config, workloads)
            mock_result = Mock()
            mock_result.stdout = "222\n"

            with (
                patch('subprocess.run', return_value=mock_result) as mock_run,
                patch('llmb_run.job_launcher.create_llmb_config'),
            ):
                launcher.launch(task)

            cmd = mock_run.call_args.args[0]
            assert 'run_job.sh' in cmd, f"Expected run_job.sh in cmd, got {cmd}"

    def test_nemo2_launcher_uses_metadata_launch_script(self):
        """Nemo2Launcher should use the launch_script from workload metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _cluster_config(
                llmb_install="/path/to/llmb_install",
                gpu_type="h100",
                node_architecture="x86_64",
                slurm={"gpu_partition": "batch"},
                workloads_config={
                    "test_wl": {
                        "venv_path": "/path/to/venv",
                        "venv_type": "venv",
                        "environment": {},
                    }
                },
                cwd=temp_dir,
            )
            workloads = {
                'test_wl': {
                    'dir': '/path/to/workload',
                    'metadata': {
                        'general': {'framework': 'nemo2'},
                        'run': {'launcher_type': 'nemo', 'launch_script': 'nemo_launch.sh'},
                    },
                }
            }
            task = WorkloadTask(
                workload_key='test_wl',
                model_size='8b',
                dtype='bf16',
                scale=8,
                profile=False,
            )

            with patch('os.path.exists') as mock_exists:
                mock_exists.side_effect = lambda p: p == '/path/to/venv/bin/python3'
                launcher = Nemo2Launcher(config, workloads)

                experiment_dir = "/path/to/experiment"
                mock_result = Mock()
                mock_result.stdout = (
                    "Task 0: test\n" "- Status: SUBMITTED\n" "- Job id: 333\n" f"- Local Directory: {experiment_dir}\n"
                )
                mock_result.returncode = 0

                with patch('subprocess.run', return_value=mock_result) as mock_run:
                    launcher.launch(task)

                # Find the launch script call.
                launch_calls = [c for c in mock_run.call_args_list if 'nemo_launch.sh' in str(c.args[0])]
                assert launch_calls, (
                    f"Expected a subprocess call with nemo_launch.sh, got: "
                    f"{[c.args[0] for c in mock_run.call_args_list]}"
                )

    def test_nemo2_launcher_job_id_parse_error_is_not_venv_error(self, caplog):
        """Invalid Nemo launcher job IDs should not be logged as venv failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _cluster_config(
                llmb_install="/path/to/llmb_install",
                gpu_type="h100",
                node_architecture="x86_64",
                slurm={"gpu_partition": "batch"},
                workloads_config={
                    "test_wl": {
                        "venv_path": "/path/to/venv",
                        "venv_type": "venv",
                        "environment": {},
                    }
                },
                cwd=temp_dir,
            )
            workloads = {
                'test_wl': {
                    'dir': '/path/to/workload',
                    'metadata': {'general': {'framework': 'nemo2'}, 'run': {'launcher_type': 'nemo'}},
                }
            }
            task = WorkloadTask('test_wl', '8b', 'bf16', 8, profile=False)
            mock_result = Mock()
            mock_result.stdout = (
                "Task 0: test\n"
                "- Status: SUBMITTED\n"
                "- Job id: not-a-job\n"
                "- Local Directory: /path/to/experiment\n"
            )
            mock_result.returncode = 0

            with (
                patch('os.path.exists', side_effect=lambda p: p == '/path/to/venv/bin/python3'),
                patch('subprocess.run', return_value=mock_result),
            ):
                slurm_job = Nemo2Launcher(config, workloads).launch(task)

        assert slurm_job.job_id is None
        assert "Unable to parse Slurm job id" in caplog.text
        assert "venv error:" not in caplog.text


def test_nemo2_launcher_marshals_cli_slurm_args_into_additional_params():
    """CLI Slurm args should be serialized back into ADDITIONAL_SLURM_PARAMS for Nemo launchers."""
    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_deepseek-v3",
            model_size="671b",
            dtype="fp8",
            scale=1024,
            profile=False,
            slurm_args=build_cli_slurm_args(
                nodelist="node001,node002",
                reservation="res_a",
                segment=8,
                nice=100,
                slurm_args=["exclusive"],
            ),
        )

        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            node_architecture="x86_64",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_deepseek-v3": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {},
                }
            },
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_deepseek-v3': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                    'run': {'launcher_type': 'nemo'},
                },
                'workload': 'deepseek-v3',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        experiment_dir = "/path/to/experiment/directory"

        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == '/path/to/venv/bin/python3'
            launcher = Nemo2Launcher(config, workloads)

            mock_result = Mock()
            mock_result.stdout = (
                "Task 0: test\n" "- Status: SUBMITTED\n" "- Job id: 67890\n" f"- Local Directory: {experiment_dir}\n"
            )
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                launcher.launch(task)

        launch_call = next(call for call in mock_run.call_args_list if './launch.sh' in str(call.args[0]))
        env = launch_call.kwargs['env']
        assert env[ADDITIONAL_SLURM_PARAMS_KEY] == (
            "nodelist=node001,node002;reservation=res_a;segment=8;nice=100;exclusive"
        )


def test_nemo2_launcher_exports_container_env_contract():
    """Nemo2Launcher should export explicit env overrides for downstream -E handling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_deepseek-v3",
            model_size="671b",
            dtype="fp8",
            scale=1024,
            profile=False,
            env_overrides={"HF_TOKEN": "abc123", "NCCL_DEBUG": "INFO"},
            explicit_env_overrides={"HF_TOKEN": "abc123", "NCCL_DEBUG": "INFO"},
        )

        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            node_architecture="x86_64",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_deepseek-v3": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {},
                }
            },
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_deepseek-v3': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                    'run': {'launcher_type': 'nemo'},
                },
                'workload': 'deepseek-v3',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        experiment_dir = "/path/to/experiment/directory"

        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == '/path/to/venv/bin/python3'
            launcher = Nemo2Launcher(config, workloads)

            mock_result = Mock()
            mock_result.stdout = (
                "Task 0: test\n" "- Status: SUBMITTED\n" "- Job id: 67890\n" f"- Local Directory: {experiment_dir}\n"
            )
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                launcher.launch(task)

        launch_call = next(call for call in mock_run.call_args_list if './launch.sh' in str(call.args[0]))
        env = launch_call.kwargs['env']
        assert env['HF_TOKEN'] == 'abc123'
        assert env['NCCL_DEBUG'] == 'INFO'
        assert env[NEMO_ENV_OVERRIDE_VAR] == '-E HF_TOKEN=abc123 -E NCCL_DEBUG=INFO'
        assert env['LLMB_REPO'] == config.llmb_repo
        assert env['LLMB_WORKLOAD'] == f"{config.llmb_install}/workloads/{task.workload_key}"
        assert env['LLMB_RECIPE_ROOT'] == workloads[task.workload_key]['dir']


def test_build_nemo_env_override_flags_quotes_values_with_whitespace():
    """Values with whitespace or glob chars must be shell-quoted so unquoted expansion doesn't word-split them."""
    rendered = build_nemo_env_override_flags(
        {"HF_TOKEN": "abc123", "MSG": "hello world", "NCCL_IB_HCAS": "mlx0,mlx2,mlx10"}
    )
    assert rendered == "-E HF_TOKEN=abc123 -E 'MSG=hello world' -E NCCL_IB_HCAS=mlx0,mlx2,mlx10"


def test_build_nemo_workload_args_preserves_raw_tokens():
    """Extra workload args are passed as raw argv tokens through CONFIG_OVERRIDES."""
    assert build_nemo_workload_args(("--dump_env", "--some_flag=value")) == "--dump_env --some_flag=value"


def test_build_mbridge_extra_args_orders_compatibility_env_and_raw_args():
    """Raw MBridge passthrough arguments should follow mirrored compatibility inputs."""
    assert (
        build_mbridge_extra_args(
            ("--micro_batch_size=2", "model.hidden_size=4096"),
            compatibility_args=("--dump_env",),
            env_overrides={"NCCL_DEBUG": "INFO"},
        )
        == "--dump_env -E NCCL_DEBUG=INFO --micro_batch_size=2 model.hidden_size=4096"
    )


def test_mbridge_extra_args_append_after_existing_contract_value():
    """CLI arguments should follow any preconfigured MBridge arguments."""
    env = {MBRIDGE_EXTRA_ARGS_VAR: "model.hidden_size=2048"}

    apply_mbridge_extra_args_contract(env, ("model.hidden_size=4096",))

    assert env[MBRIDGE_EXTRA_ARGS_VAR] == "model.hidden_size=2048 model.hidden_size=4096"


def test_megatron_bridge_launcher_dual_writes_legacy_and_new_args():
    """MBridge tasks should export compatibility inputs through both contracts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_deepseek-v3",
            model_size="671b",
            dtype="fp8",
            scale=1024,
            profile=False,
            explicit_env_overrides={"NCCL_DEBUG": "INFO"},
            extra_workload_args=("--dump_env",),
            mbridge_args=("--micro_batch_size=2", "model.hidden_size=4096"),
        )

        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            node_architecture="x86_64",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_deepseek-v3": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {},
                }
            },
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_deepseek-v3': {
                'metadata': {
                    'general': {'framework': 'megatron_bridge', 'gsw_version': '26.02'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:26.02.01']},
                    'run': {'launcher_type': 'megatron_bridge'},
                },
                'workload': 'deepseek-v3',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        experiment_dir = "/path/to/experiment/directory"

        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == '/path/to/venv/bin/python3'
            launcher = Nemo2Launcher(config, workloads)

            mock_result = Mock()
            mock_result.stdout = (
                "Task 0: test\n" "- Status: SUBMITTED\n" "- Job id: 67890\n" f"- Local Directory: {experiment_dir}\n"
            )
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                launcher.launch(task)

        launch_call = next(call for call in mock_run.call_args_list if './launch.sh' in str(call.args[0]))
        env = launch_call.kwargs['env']
        assert env[NEMO_ENV_OVERRIDE_VAR] == '--dump_env -E NCCL_DEBUG=INFO'
        assert (
            env[MBRIDGE_EXTRA_ARGS_VAR] == '--dump_env -E NCCL_DEBUG=INFO --micro_batch_size=2 model.hidden_size=4096'
        )


def test_nemo_launcher_ignores_extra_workload_args():
    """Non-Megatron-Bridge Nemo-style tasks should silently ignore extra workload args."""
    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_deepseek-v3",
            model_size="671b",
            dtype="fp8",
            scale=1024,
            profile=False,
            extra_workload_args=("--dump_env",),
            mbridge_args=("--micro_batch_size=2",),
        )

        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            node_architecture="x86_64",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_deepseek-v3": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {},
                }
            },
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_deepseek-v3': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                    'run': {'launcher_type': 'nemo'},
                },
                'workload': 'deepseek-v3',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        experiment_dir = "/path/to/experiment/directory"

        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == '/path/to/venv/bin/python3'
            launcher = Nemo2Launcher(config, workloads)

            mock_result = Mock()
            mock_result.stdout = (
                "Task 0: test\n" "- Status: SUBMITTED\n" "- Job id: 67890\n" f"- Local Directory: {experiment_dir}\n"
            )
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                launcher.launch(task)

        launch_call = next(call for call in mock_run.call_args_list if './launch.sh' in str(call.args[0]))
        env = launch_call.kwargs['env']
        assert NEMO_ENV_OVERRIDE_VAR not in env
        assert MBRIDGE_EXTRA_ARGS_VAR not in env


def test_nemo2_launcher_appends_container_env_contract_to_existing_overrides():
    """Nemo2Launcher should append explicit -E flags to existing CONFIG_OVERRIDES."""
    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_deepseek-v3",
            model_size="671b",
            dtype="fp8",
            scale=1024,
            profile=False,
            env_overrides={"HF_TOKEN": "abc123"},
            explicit_env_overrides={"HF_TOKEN": "abc123"},
        )

        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            node_architecture="x86_64",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_deepseek-v3": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {NEMO_ENV_OVERRIDE_VAR: "--enable_nsys"},
                }
            },
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_deepseek-v3': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                    'run': {'launcher_type': 'nemo'},
                },
                'workload': 'deepseek-v3',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        experiment_dir = "/path/to/experiment/directory"

        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == '/path/to/venv/bin/python3'
            launcher = Nemo2Launcher(config, workloads)

            mock_result = Mock()
            mock_result.stdout = (
                "Task 0: test\n" "- Status: SUBMITTED\n" "- Job id: 67890\n" f"- Local Directory: {experiment_dir}\n"
            )
            mock_result.returncode = 0

            with patch('subprocess.run', return_value=mock_result) as mock_run:
                launcher.launch(task)

        launch_call = next(call for call in mock_run.call_args_list if './launch.sh' in str(call.args[0]))
        env = launch_call.kwargs['env']
        assert env[NEMO_ENV_OVERRIDE_VAR] == '--enable_nsys -E HF_TOKEN=abc123'


def test_nemo2_launcher_rejects_cli_slurm_mix_with_direct_env(caplog):
    """CLI Slurm args should fail fast if direct env-var injection is also configured."""
    with tempfile.TemporaryDirectory() as temp_dir:
        task = WorkloadTask(
            workload_key="pretraining_deepseek-v3",
            model_size="671b",
            dtype="fp8",
            scale=1024,
            profile=False,
            slurm_args=build_cli_slurm_args(nodelist="node001"),
        )

        config = _cluster_config(
            llmb_install="/path/to/llmb_install",
            llmb_repo="/path/to/llmb_repo",
            gpu_type="h100",
            cluster_name="test_cluster",
            node_architecture="x86_64",
            slurm={"account": "test_account", "gpu_partition": "test_partition", "gpu_gres": "8"},
            workloads_config={
                "pretraining_deepseek-v3": {
                    "venv_path": "/path/to/venv",
                    "venv_type": "venv",
                    "environment": {ADDITIONAL_SLURM_PARAMS_KEY: "reservation=res_a"},
                }
            },
            cwd=temp_dir,
        )

        workloads = {
            'pretraining_deepseek-v3': {
                'metadata': {
                    'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                    'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
                    'run': {'launcher_type': 'nemo'},
                },
                'workload': 'deepseek-v3',
                'workload_type': 'pretrain',
                'synthetic_dataset': True,
                'dir': '/path/to/workload',
            }
        }

        with patch('os.path.exists') as mock_exists:
            mock_exists.side_effect = lambda p: p == '/path/to/venv/bin/python3'
            launcher = Nemo2Launcher(config, workloads)
            slurm_job = launcher.launch(task)

        assert slurm_job.job_id is None
        assert "Cannot combine first-class Slurm CLI flags with ADDITIONAL_SLURM_PARAMS" in caplog.text
        assert "venv error:" not in caplog.text

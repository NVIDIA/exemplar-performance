import pathlib

import pytest

from llmb_run.config_manager import (
    ClusterConfig,
    InstallConfig,
    SlurmConfig,
    SlurmTargetConfig,
    WorkloadsConfig,
)


@pytest.fixture
def make_cluster_config(tmp_path):
    """Build a ClusterConfig without touching cluster_config.yaml parsing."""

    def _make(gpu_type='gb200', installed=(), llmb_install=None, llmb_repo=None):
        return ClusterConfig(
            schema_version=1,
            gpu_type=gpu_type,
            llmb_install=str(llmb_install or tmp_path / 'install'),
            llmb_repo=str(llmb_repo or tmp_path / 'repo'),
            cluster_name='test',
            install=InstallConfig(),
            slurm=SlurmConfig(gpu=SlurmTargetConfig(env_vars={}), cpu=SlurmTargetConfig(env_vars={})),
            workloads=WorkloadsConfig(installed=list(installed)),
            environment={},
            cwd=pathlib.Path(tmp_path),
        )

    return _make

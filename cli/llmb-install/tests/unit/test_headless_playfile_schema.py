# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for headless playfile schema compatibility."""

import pytest
import yaml
from pydantic import ValidationError

from llmb_install.cluster.nvlink import NvlDetection
from llmb_install.config.headless import load_installation_config
from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.core.installer import Installer


def test_playfile_schema_populates_slurm_and_env_vars(tmp_path):
    """Ensure playfiles use top-level slurm/environment_vars."""
    config_path = tmp_path / "playfile.yaml"
    playfile_data = {
        "install_path": "/tmp/llmb-install",
        "venv_type": "uv",
        "gpu_type": "h100",
        "node_architecture": "x86_64",
        "image_download_method": "slurm",
        "selected_workloads": ["nemotron"],
        "slurm": {
            "account": "test-account",
            "gpu_partition": "gpu-partition",
            "cpu_partition": "cpu-partition",
            "gpu_partition_gres": 8,
            "cpu_partition_gres": None,
        },
        "environment_vars": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    }

    config_path.write_text(yaml.safe_dump(playfile_data))

    loaded_data = load_installation_config(str(config_path))
    config = InstallConfig.model_validate(loaded_data)

    assert loaded_data["slurm"]["account"] == "test-account"
    assert loaded_data["environment_vars"] == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert config.slurm is not None
    assert config.slurm.account == "test-account"
    assert config.environment_vars == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert config.slurm.model_dump()["gpu_partition"] == "gpu-partition"


def test_playfile_accepts_legacy_install_method_key(tmp_path):
    """Playfiles written before the rename still say install_method."""
    playfile_data = {
        "install_path": "/tmp/llmb-install",
        "venv_type": "uv",
        "gpu_type": "h100",
        "node_architecture": "x86_64",
        "install_method": "slurm",
        "selected_workloads": ["nemotron"],
        "slurm": {
            "account": "a",
            "gpu_partition": "g",
            "cpu_partition": "c",
            "gpu_partition_gres": 8,
        },
    }
    config_path = tmp_path / "playfile.yaml"
    config_path.write_text(yaml.safe_dump(playfile_data))

    config = load_installation_config(str(config_path))

    assert config["image_download_method"] == "slurm"
    assert "install_method" not in config


def test_load_installation_config_rejects_deprecated_keys(tmp_path, capsys):
    """Ensure deprecated slurm_info/env_vars keys raise explicit error."""
    config_path = tmp_path / "playfile.yaml"
    playfile_data = {
        "install_path": "/tmp/llmb-install",
        "venv_type": "uv",
        "gpu_type": "h100",
        "node_architecture": "x86_64",
        "image_download_method": "slurm",
        "selected_workloads": ["nemotron"],
        "slurm_info": {
            "account": "test-account",
            "gpu_partition": "gpu-partition",
            "cpu_partition": "cpu-partition",
        },
        "env_vars": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    }

    config_path.write_text(yaml.safe_dump(playfile_data))

    with pytest.raises(SystemExit):
        load_installation_config(str(config_path))

    captured = capsys.readouterr()
    assert "slurm_info/env_vars are no longer supported" in captured.out


def test_to_play_dict_roundtrip_drops_denylisted_fields():
    """Ensure to_play_dict output is schema-compatible and pruned."""
    config = InstallConfig(
        install_path="/tmp/llmb-install",
        venv_type="uv",
        gpu_type="h100",
        node_architecture="x86_64",
        image_download_method="slurm",
        selected_workloads=["nemotron"],
        slurm=SlurmConfig(
            account="test-account",
            gpu_partition="gpu-partition",
            cpu_partition="cpu-partition",
        ),
        environment_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        ui_mode="rich",
        cache_dirs_configured=True,
        dev_mode=True,
        llmb_repo="/tmp/llmb",
        is_incremental_install=True,
    )

    playfile_data = config.to_play_dict()
    restored = InstallConfig.model_validate(playfile_data)

    assert restored.install_path == config.install_path
    assert restored.slurm is not None
    assert restored.slurm.account == "test-account"
    assert restored.environment_vars == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert "llmb_repo" not in playfile_data
    assert "dev_mode" not in playfile_data
    assert "ui_mode" not in playfile_data
    assert "cache_dirs_configured" not in playfile_data
    assert "is_incremental_install" not in playfile_data


def test_nvlink_fields_roundtrip_and_none_values_are_omitted():
    config = InstallConfig(
        install_path="/tmp/llmb-install",
        venv_type="uv",
        gpu_type="gb200",
        node_architecture="aarch64",
        image_download_method="slurm",
        selected_workloads=["nemotron"],
        nvl_domain=72,
        nvl_block_sizes=[18],
        slurm=SlurmConfig(account="acct", gpu_partition="gpu", cpu_partition="cpu"),
    )
    playfile_data = config.to_play_dict()
    assert playfile_data["nvl_domain"] == 72
    assert playfile_data["nvl_block_sizes"] == [18]

    config.nvl_domain = None
    config.nvl_block_sizes = None
    playfile_data = config.to_play_dict()
    assert "nvl_domain" not in playfile_data
    assert "nvl_block_sizes" not in playfile_data


def test_headless_fixed_gpu_overrides_playfile_nvlink_values(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "playfile.yaml"
    playfile_data = {
        "install_path": str(tmp_path / "install"),
        "venv_type": "uv",
        "gpu_type": "h100",
        "node_architecture": "x86_64",
        "image_download_method": "local",
        "selected_workloads": ["nemotron"],
        "nvl_domain": 72,
        "nvl_block_sizes": [18],
        "slurm": {
            "account": "test-account",
            "gpu_partition": "gpu-partition",
            "cpu_partition": "cpu-partition",
            "gpu_partition_gres": 8,
        },
    }
    config_path.write_text(yaml.safe_dump(playfile_data))
    captured = {}

    monkeypatch.setattr(Installer, "_handle_repository_setup", lambda self, path, dev_mode: "/tmp/llmb")
    monkeypatch.setattr(Installer, "_prepare_workloads", lambda self, gpu_type, workloads, **_: workloads)
    monkeypatch.setattr(
        Installer,
        "_perform_installation",
        lambda self, config, workloads, is_resume: captured.update(config=config),
    )

    Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    assert captured["config"].nvl_domain == 8
    assert captured["config"].nvl_block_sizes is None
    output = capsys.readouterr().out
    assert "Warning: ignoring playfile NVLink settings; H100 uses NVL8." in output
    assert "block sizes" not in output


def _headless_nvlink_playfile(tmp_path, nvl_domain, nvl_block_sizes=None):
    playfile_data = {
        "install_path": str(tmp_path / "install"),
        "venv_type": "uv",
        "gpu_type": "gb200",
        "node_architecture": "aarch64",
        "image_download_method": "local",
        "selected_workloads": ["nemotron"],
        "nvl_domain": nvl_domain,
        "slurm": {
            "account": "test-account",
            "gpu_partition": "gpu-partition",
            "cpu_partition": "cpu-partition",
        },
    }
    if nvl_block_sizes is not None:
        playfile_data["nvl_block_sizes"] = nvl_block_sizes
    config_path = tmp_path / "playfile.yaml"
    config_path.write_text(yaml.safe_dump(playfile_data))
    return config_path


def _stub_headless_install(monkeypatch, captured):
    monkeypatch.setattr(Installer, "_handle_repository_setup", lambda self, path, dev_mode: "/tmp/llmb")
    monkeypatch.setattr(Installer, "_prepare_workloads", lambda self, gpu_type, workloads, **_: workloads)
    monkeypatch.setattr(
        Installer,
        "_perform_installation",
        lambda self, config, workloads, is_resume: captured.update(config=config),
    )


def test_headless_playfile_domain_fills_missing_block_sizes(tmp_path, monkeypatch, capsys):
    config_path = _headless_nvlink_playfile(tmp_path, nvl_domain=576)
    captured = {}
    detection = NvlDetection(block_sizes=[9], nvl_domain=36, reason="detected", gpu_type="gb200")
    monkeypatch.setattr("llmb_install.core.installer.detect_nvl_topology", lambda gpu_type: detection)
    _stub_headless_install(monkeypatch, captured)

    Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    assert captured["config"].nvl_block_sizes == [9]
    assert "Warning: playfile declares NVL576 but the Slurm topology looks like NVL36." in capsys.readouterr().out


def test_headless_playfile_matching_domain_has_no_warning(tmp_path, monkeypatch, capsys):
    config_path = _headless_nvlink_playfile(tmp_path, nvl_domain=72)
    captured = {}
    detection = NvlDetection(block_sizes=[18], nvl_domain=72, reason="detected", gpu_type="gb200")
    monkeypatch.setattr("llmb_install.core.installer.detect_nvl_topology", lambda gpu_type: detection)
    _stub_headless_install(monkeypatch, captured)

    Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    assert captured["config"].nvl_block_sizes == [18]
    assert "Warning: playfile declares NVL" not in capsys.readouterr().out


@pytest.mark.parametrize("reason,error", [("no_blocks", None), ("error", "missing")])
def test_headless_playfile_domain_ignores_missing_topology(tmp_path, monkeypatch, capsys, reason, error):
    config_path = _headless_nvlink_playfile(tmp_path, nvl_domain=72)
    captured = {}
    detection = NvlDetection(block_sizes=None, nvl_domain=None, reason=reason, gpu_type="gb200", error=error)
    monkeypatch.setattr("llmb_install.core.installer.detect_nvl_topology", lambda gpu_type: detection)
    _stub_headless_install(monkeypatch, captured)

    Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    assert captured["config"].nvl_block_sizes is None
    output = capsys.readouterr().out
    assert "Warning: playfile declares NVL" not in output
    assert "Could not detect the NVLink domain" not in output


def test_headless_playfile_block_sizes_skip_detection(tmp_path, monkeypatch):
    config_path = _headless_nvlink_playfile(tmp_path, nvl_domain=72, nvl_block_sizes=[18])
    captured = {}

    def fail_detection(gpu_type):
        raise AssertionError("NVLink detection should not run when block sizes are provided")

    monkeypatch.setattr("llmb_install.core.installer.detect_nvl_topology", fail_detection)
    _stub_headless_install(monkeypatch, captured)

    Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    assert captured["config"].nvl_block_sizes == [18]


def test_headless_rejects_unsupported_detected_nvlink_domain(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "playfile.yaml"
    playfile_data = {
        "install_path": str(tmp_path / "install"),
        "venv_type": "uv",
        "gpu_type": "gb200",
        "node_architecture": "aarch64",
        "image_download_method": "local",
        "selected_workloads": ["nemotron"],
        "slurm": {
            "account": "test-account",
            "gpu_partition": "gpu-partition",
            "cpu_partition": "cpu-partition",
        },
    }
    config_path.write_text(yaml.safe_dump(playfile_data))
    monkeypatch.setattr(
        "llmb_install.core.installer.detect_nvl_topology",
        lambda gpu_type: NvlDetection(block_sizes=[9, 144], nvl_domain=576, reason="detected", gpu_type="gb200"),
    )

    with pytest.raises(SystemExit):
        Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    output = capsys.readouterr().out
    assert "Error: Detected NVL576 from the Slurm topology; GB200 systems are expected to be NVL72." in output
    assert "Set 'nvl_domain' in the playfile." in output


def test_headless_rejects_missing_nvlink_topology(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "playfile.yaml"
    playfile_data = {
        "install_path": str(tmp_path / "install"),
        "venv_type": "uv",
        "gpu_type": "gb200",
        "node_architecture": "aarch64",
        "image_download_method": "local",
        "selected_workloads": ["nemotron"],
        "slurm": {
            "account": "test-account",
            "gpu_partition": "gpu-partition",
            "cpu_partition": "cpu-partition",
        },
    }
    config_path.write_text(yaml.safe_dump(playfile_data))
    monkeypatch.setattr(
        "llmb_install.core.installer.detect_nvl_topology",
        lambda gpu_type: NvlDetection(block_sizes=None, nvl_domain=None, reason="no_blocks", gpu_type="gb200"),
    )

    with pytest.raises(SystemExit):
        Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    output = capsys.readouterr().out
    assert "Error: Could not detect the NVLink domain from the Slurm topology." in output
    assert "Set 'nvl_domain' in the playfile." in output


def test_load_installation_config_rejects_empty_selected_workloads(tmp_path, capsys):
    """Ensure empty selected_workloads fails with clear message."""
    config_path = tmp_path / "playfile.yaml"
    playfile_data = {
        "install_path": "/tmp/llmb-install",
        "venv_type": "uv",
        "gpu_type": "h100",
        "node_architecture": "x86_64",
        "image_download_method": "local",
        "selected_workloads": [],
        "slurm": {
            "account": "test-account",
            "gpu_partition": "gpu-partition",
            "cpu_partition": "cpu-partition",
        },
    }

    config_path.write_text(yaml.safe_dump(playfile_data))

    with pytest.raises(SystemExit):
        load_installation_config(str(config_path))

    captured = capsys.readouterr()
    assert "selected_workloads cannot be empty" in captured.out


def test_load_installation_config_rejects_missing_slurm(tmp_path, capsys):
    """Ensure missing slurm configuration fails validation."""
    config_path = tmp_path / "playfile.yaml"
    playfile_data = {
        "install_path": "/tmp/llmb-install",
        "venv_type": "uv",
        "gpu_type": "h100",
        "node_architecture": "x86_64",
        "image_download_method": "local",
        "selected_workloads": ["nemotron"],
    }

    config_path.write_text(yaml.safe_dump(playfile_data))

    with pytest.raises(SystemExit):
        load_installation_config(str(config_path))

    captured = capsys.readouterr()
    assert "slurm" in captured.out


class TestNonBlankStringValidation:
    """Ensure whitespace-only strings are rejected where they would cause downstream failures."""

    def test_blank_install_path_rejected(self):
        with pytest.raises(ValidationError, match="install_path"):
            InstallConfig(
                install_path="  ",
                venv_type="uv",
                gpu_type="h100",
                node_architecture="x86_64",
            )

    def test_blank_slurm_account_rejected(self):
        with pytest.raises(ValidationError, match="account"):
            SlurmConfig(account="  ", gpu_partition="gpu", cpu_partition="cpu")

    def test_blank_workload_name_rejected(self):
        with pytest.raises(ValidationError, match="selected_workloads"):
            InstallConfig(
                install_path="/tmp/test",
                venv_type="uv",
                gpu_type="h100",
                node_architecture="x86_64",
                selected_workloads=["valid", "  "],
            )


def test_headless_kubernetes_playfile_does_not_query_slurm_for_nvl_domain(tmp_path, monkeypatch):
    """A Kubernetes install has no scontrol; the NVLink domain must not need it."""
    config_path = tmp_path / "playfile.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "install_path": str(tmp_path / "install"),
                "venv_type": "uv",
                "gpu_type": "gb300",
                "node_architecture": "aarch64",
                "selected_workloads": ["nemotron"],
                "kubernetes": {"namespace": "ns"},
            }
        )
    )
    captured = {}

    def no_scontrol(gpu_type):
        raise AssertionError("Slurm topology must not be queried for a Kubernetes install")

    monkeypatch.setattr("llmb_install.core.installer.detect_nvl_topology", no_scontrol)
    monkeypatch.setattr(Installer, "_handle_repository_setup", lambda self, path, dev_mode: "/tmp/llmb")
    monkeypatch.setattr(Installer, "_prepare_workloads", lambda self, gpu_type, workloads, **_: workloads)
    monkeypatch.setattr(
        Installer,
        "_perform_installation",
        lambda self, config, workloads, is_resume: captured.update(config=config),
    )

    Installer()._run_headless_mode(type("Args", (), {"play": str(config_path), "dev_mode": False})())

    assert captured["config"].nvl_domain == 72
    assert captured["config"].nvl_block_sizes is None

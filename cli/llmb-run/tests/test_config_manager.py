# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import pathlib

import pytest

from llmb_run.config_manager import (
    ClusterConfig,
    ExecutionTarget,
    _normalize_slurm_key,
    get_cluster_config,
    resolve_slurm_targets,
)


def _cluster_config(slurm: dict) -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "launcher": {
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
            },
            "slurm": slurm,
        }
    )


def test_legacy_slurm_schema_supported_for_gpu_and_cpu_targets():
    config = _cluster_config(
        {
            "account": "test_account",
            "gpu_partition": "gpu_queue",
            "gpu_gres": 8,
            "cpu_partition": "cpu_queue",
            "cpu_gres": None,
        }
    )

    gpu_env = config.slurm.env(target="gpu")
    cpu_env = config.slurm.env(target="cpu")

    assert gpu_env["SBATCH_ACCOUNT"] == "test_account"
    assert gpu_env["SBATCH_PARTITION"] == "gpu_queue"
    assert gpu_env["SBATCH_GPUS_PER_NODE"] == "8"

    assert cpu_env["SBATCH_ACCOUNT"] == "test_account"
    assert cpu_env["SBATCH_PARTITION"] == "cpu_queue"
    assert "SBATCH_GPUS_PER_NODE" not in cpu_env


def test_cluster_config_parses_nvlink_fields():
    config = ClusterConfig.from_dict(
        {
            "schema_version": 2,
            "gpu_type": "gb200",
            "llmb_install": "/install",
            "llmb_repo": "/repo",
            "cluster_name": None,
            "nvl_domain": 72,
            "nvl_block_sizes": [18],
            "slurm": {"gpu": {"partition": "gpu"}},
        }
    )
    assert config.nvl_domain == 72
    assert config.nvl_block_sizes == [18]


def test_cluster_config_legacy_nvlink_fields_are_none():
    config = _cluster_config({"gpu_partition": "gpu"})
    assert config.nvl_domain is None
    assert config.nvl_block_sizes is None


def test_cluster_config_rejects_invalid_nvlink_domain():
    with pytest.raises(ValueError, match="'nvl_domain' must be a positive integer"):
        ClusterConfig.from_dict(
            {
                "schema_version": 2,
                "gpu_type": "gb200",
                "llmb_install": "/install",
                "llmb_repo": "/repo",
                "nvl_domain": "x",
                "slurm": {"gpu": {"partition": "gpu"}},
            }
        )


def test_new_slurm_schema_common_and_target_overrides_with_null_unset():
    config = _cluster_config(
        {
            "account": "acct",
            "qos": "free",
            "gpu": {
                "partition": "gpuq",
                "qos": "priority",
                "mem_per_gpu": "32G",
            },
            "cpu": {
                "partition": "cpuq",
                "qos": None,
            },
        }
    )

    gpu_env = config.slurm.env(target="gpu")
    cpu_env = config.slurm.env(target="cpu")

    assert gpu_env["SBATCH_ACCOUNT"] == "acct"
    assert gpu_env["SBATCH_PARTITION"] == "gpuq"
    assert gpu_env["SBATCH_QOS"] == "priority"
    assert gpu_env["SBATCH_MEM_PER_GPU"] == "32G"

    assert cpu_env["SBATCH_ACCOUNT"] == "acct"
    assert cpu_env["SBATCH_PARTITION"] == "cpuq"
    assert "SBATCH_QOS" not in cpu_env


def test_cpu_target_falls_back_to_gpu_when_cpu_block_missing():
    config = _cluster_config(
        {
            "account": "acct",
            "gpu": {
                "partition": "batch",
                "gpus_per_node": 8,
            },
        }
    )

    gpu_env = config.slurm.env(target="gpu")
    cpu_env = config.slurm.env(target="cpu")

    assert cpu_env == gpu_env


def test_reject_mixed_legacy_and_new_slurm_schema():
    config = {
        "slurm": {
            "gpu_partition": "gpuq",
            "gpu": {"partition": "gpuq"},
        }
    }

    with pytest.raises(ValueError, match="cannot be used with the new slurm.gpu/slurm.cpu schema"):
        resolve_slurm_targets(config)


def test_reject_new_schema_without_gpu_block():
    config = {
        "slurm": {
            "account": "acct",
            "partition": "gpuq",
        }
    }

    with pytest.raises(ValueError, match="slurm.gpu"):
        resolve_slurm_targets(config)


def test_reject_unknown_nested_slurm_blocks():
    config = {
        "slurm": {
            "gpu": {"partition": "gpuq"},
            "post_processing": {"partition": "cpuq"},
        }
    }

    with pytest.raises(ValueError, match="Only 'slurm.gpu' and 'slurm.cpu' are supported"):
        resolve_slurm_targets(config)


def test_reject_non_dict_slurm_section():
    config = {"slurm": "gpuq"}

    with pytest.raises(ValueError, match="'slurm' section must be a dictionary"):
        resolve_slurm_targets(config)


def test_reject_non_dict_gpu_block():
    config = {
        "slurm": {
            "gpu": "gpuq",
        }
    }

    with pytest.raises(ValueError, match="'slurm.gpu' must be a dictionary"):
        resolve_slurm_targets(config)


def test_reject_non_dict_cpu_block():
    config = {
        "slurm": {
            "gpu": {"partition": "gpuq"},
            "cpu": "cpuq",
        }
    }

    with pytest.raises(ValueError, match="'slurm.cpu' must be a dictionary"):
        resolve_slurm_targets(config)


def test_reject_when_resolved_gpu_target_missing_partition():
    config = {
        "slurm": {
            "gpu": {
                "qos": "free",
            },
        }
    }

    with pytest.raises(ValueError, match="resolved GPU target is missing partition"):
        resolve_slurm_targets(config)


def test_key_normalization_accepts_prefixed_and_bare_forms():
    config = _cluster_config(
        {
            "gpu": {
                "partition": "gpuq",
                "sbatch_qos": "free",
                "slurm_hint": "nomultithread",
                "foo_bar": "1",
                "gres": 8,
            }
        }
    )

    gpu_env = config.slurm.env(target="gpu")
    assert gpu_env["SBATCH_PARTITION"] == "gpuq"
    assert gpu_env["SBATCH_QOS"] == "free"
    assert gpu_env["SLURM_HINT"] == "nomultithread"
    assert gpu_env["SBATCH_FOO_BAR"] == "1"
    assert gpu_env["SBATCH_GPUS_PER_NODE"] == "8"


def test_target_selector_cpu_works():
    config = _cluster_config(
        {
            "account": "acct",
            "gpu_partition": "gpuq",
            "cpu_partition": "cpuq",
        }
    )

    cpu_env = config.slurm.env(target="cpu")
    assert cpu_env["SBATCH_PARTITION"] == "cpuq"


def test_missing_slurm_section_is_rejected():
    config = {
        "launcher": {
            "llmb_repo": "/tmp",
            "llmb_install": "/tmp",
            "gpu_type": "h100",
        }
    }

    with pytest.raises(ValueError, match="Missing required 'slurm' section"):
        resolve_slurm_targets(config)


def test_kubernetes_cluster_needs_no_slurm_section():
    config = {
        "launcher": {
            "llmb_repo": "/tmp",
            "llmb_install": "/tmp",
            "gpu_type": "h100",
        },
        "execution_target": "kubernetes",
    }

    assert resolve_slurm_targets(config) == {"gpu": {}, "cpu": {}}


@pytest.mark.parametrize(
    "execution_target_field",
    [{}, {"execution_target": None}],
    ids=["missing", "null"],
)
def test_legacy_slurm_config_defaults_execution_target(execution_target_field):
    raw_config = {
        "launcher": {"llmb_repo": "/repo", "llmb_install": "/install", "gpu_type": "h100"},
        "slurm": {"account": "acct", "gpu_partition": "gpuq", "cpu_partition": "cpuq"},
        **execution_target_field,
    }

    config = ClusterConfig.from_dict(raw_config)

    assert config.execution_target is ExecutionTarget.SLURM


def test_explicit_slurm_target_is_not_overridden_by_a_stray_kubernetes_block():
    config = {
        "launcher": {"llmb_repo": "/tmp", "llmb_install": "/tmp", "gpu_type": "h100"},
        "execution_target": "slurm",
        "kubernetes": {"namespace": "ns"},
    }

    with pytest.raises(ValueError, match="Missing required 'slurm' section"):
        resolve_slurm_targets(config)


def test_stray_kubernetes_block_with_non_kubernetes_target_is_rejected():
    raw_config = {
        "schema_version": 2,
        "llmb_repo": "/repo",
        "llmb_install": "/install",
        "gpu_type": "h100",
        "execution_target": "slurm",
        "slurm": {"account": "acct", "gpu": {"partition": "gpuq"}, "cpu": {"partition": "cpuq"}},
        "kubernetes": {"namespace": "ns"},
    }

    with pytest.raises(ValueError, match="'kubernetes' block is present"):
        ClusterConfig.from_dict(raw_config)


def test_unknown_execution_target_value_is_rejected():
    raw_config = {
        "schema_version": 2,
        "llmb_repo": "/repo",
        "llmb_install": "/install",
        "gpu_type": "h100",
        "execution_target": "k8s",
        "slurm": {"account": "acct", "gpu": {"partition": "gpuq"}, "cpu": {"partition": "cpuq"}},
    }

    with pytest.raises(ValueError, match="unknown execution_target 'k8s'"):
        ClusterConfig.from_dict(raw_config)


def test_cluster_config_object_supports_typed_access():
    raw_config = {
        "launcher": {
            "llmb_repo": "/repo",
            "llmb_install": "/install",
            "gpu_type": "h100",
        },
        "slurm": {
            "account": "acct",
            "gpu": {
                "partition": "gpuq",
                "gpus_per_node": 8,
            },
            "cpu": {
                "partition": "cpuq",
            },
        },
    }
    config = ClusterConfig.from_dict(raw_config)

    gpu_env = config.slurm.env(target="gpu")
    cpu_env = config.slurm.env(target="cpu")

    assert gpu_env["SBATCH_PARTITION"] == "gpuq"
    assert gpu_env["SBATCH_GPUS_PER_NODE"] == "8"
    assert cpu_env["SBATCH_PARTITION"] == "cpuq"

    assert config.slurm.partition(target="gpu") == "gpuq"
    assert config.slurm.partition(target="cpu") == "cpuq"
    assert config.slurm.gpus_per_node(target="gpu") == "8"
    assert config.slurm.gpus_per_node(target="cpu") == ""

    assert config.gpu_type == "h100"


def test_get_cluster_config_returns_cluster_config_object(tmp_path, monkeypatch):
    config_file = tmp_path / "cluster_config.yaml"
    config_file.write_text("""
launcher:
  llmb_repo: /repo
  llmb_install: /install
  gpu_type: h100
slurm:
  account: acct
  gpu:
    partition: gpuq
    gpus_per_node: 8
""".strip())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMB_INSTALL", raising=False)

    config = get_cluster_config()

    assert isinstance(config, ClusterConfig)
    assert config.cwd == tmp_path
    assert config.slurm.partition("gpu") == "gpuq"
    assert config.slurm.partition("cpu") == "gpuq"


def _write_kubernetes_cluster_config(tmp_path, kubeconfig):
    kubeconfig_line = f"  kubeconfig: {kubeconfig}\n" if kubeconfig else ""
    (tmp_path / "cluster_config.yaml").write_text(
        "launcher:\n  llmb_repo: /repo\n  llmb_install: /install\n  gpu_type: gb300\n"
        "execution_target: kubernetes\n"
        f"kubernetes:\n  namespace: ns\n{kubeconfig_line}"
    )


def test_configured_kubeconfig_is_exported_for_runtime_kubectl_and_nvcrectl(tmp_path, monkeypatch):
    _write_kubernetes_cluster_config(tmp_path, "~/clusters/prod.kubeconfig")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMB_INSTALL", raising=False)
    monkeypatch.setenv("KUBECONFIG", "/ambient/default")  # restored by monkeypatch afterwards

    get_cluster_config()

    assert os.environ["KUBECONFIG"] == os.path.expanduser("~/clusters/prod.kubeconfig")


def test_ambient_kubeconfig_is_left_alone_when_none_is_configured(tmp_path, monkeypatch):
    _write_kubernetes_cluster_config(tmp_path, None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMB_INSTALL", raising=False)
    monkeypatch.setenv("KUBECONFIG", "/ambient/default")

    get_cluster_config()

    assert os.environ["KUBECONFIG"] == "/ambient/default"


def test_invalid_target_is_rejected_for_cluster_config():
    config = ClusterConfig.from_dict(
        {
            "launcher": {
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
            },
            "slurm": {
                "gpu": {"partition": "gpuq"},
            },
        }
    )

    with pytest.raises(ValueError, match="Invalid Slurm target"):
        config.slurm.env(target="invalid")
    with pytest.raises(ValueError, match="Invalid Slurm target"):
        config.slurm.partition(target="invalid")
    with pytest.raises(ValueError, match="Invalid Slurm target"):
        config.slurm.gpus_per_node(target="invalid")


def test_slurm_env_results_are_copied_not_shared():
    config = ClusterConfig.from_dict(
        {
            "launcher": {
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
            },
            "slurm": {
                "account": "acct",
                "gpu": {"partition": "gpuq"},
            },
        }
    )

    first = config.slurm.env(target="gpu")
    first["SBATCH_ACCOUNT"] = "mutated"
    second = config.slurm.env(target="gpu")

    assert second["SBATCH_ACCOUNT"] == "acct"


def test_get_cluster_config_with_legacy_yaml(tmp_path, monkeypatch):
    """End-to-end test: loading a legacy-format cluster_config.yaml through get_cluster_config()."""
    config_file = tmp_path / "cluster_config.yaml"
    config_file.write_text("""
launcher:
  llmb_repo: /repo
  llmb_install: /install
  gpu_type: h100
slurm:
  account: legacy_acct
  gpu_partition: gpu_queue
  gpu_gres: 8
  cpu_partition: cpu_queue
  cpu_gres: null
workloads:
  installed:
    - pretrain_foo
""".strip())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMB_INSTALL", raising=False)

    config = get_cluster_config()

    assert isinstance(config, ClusterConfig)
    assert config.gpu_type == "h100"
    assert config.slurm.partition("gpu") == "gpu_queue"
    assert config.slurm.partition("cpu") == "cpu_queue"
    assert config.slurm.gpus_per_node("gpu") == "8"
    assert config.slurm.gpus_per_node("cpu") == ""
    assert config.workloads.installed == ["pretrain_foo"]

    gpu_env = config.slurm.env(target="gpu")
    assert gpu_env["SBATCH_ACCOUNT"] == "legacy_acct"


def test_normalize_slurm_key_with_hyphens():
    """Keys with hyphens should normalize to underscores in SBATCH_ prefix."""
    assert _normalize_slurm_key("mem-per-gpu") == "SBATCH_MEM_PER_GPU"
    assert _normalize_slurm_key("gpus-per-node") == "SBATCH_GPUS_PER_NODE"
    assert _normalize_slurm_key("sbatch_mem-per-gpu") == "SBATCH_MEM_PER_GPU"
    assert _normalize_slurm_key("slurm_mem-per-cpu") == "SLURM_MEM_PER_CPU"
    assert _normalize_slurm_key("gres") == "SBATCH_GPUS_PER_NODE"


def test_normalize_slurm_key_with_hyphens_in_config():
    """End-to-end: hyphenated keys in a slurm block should produce correct env vars."""
    config = _cluster_config(
        {
            "gpu": {
                "partition": "gpuq",
                "mem-per-gpu": "32G",
            },
        }
    )

    gpu_env = config.slurm.env(target="gpu")
    assert gpu_env["SBATCH_MEM_PER_GPU"] == "32G"


def test_gres_alias_supports_cpu_unset_override():
    config = _cluster_config(
        {
            "gpu": {
                "partition": "gpuq",
                "gres": 8,
            },
            "cpu": {
                "partition": "cpuq",
                "gres": None,
            },
        }
    )

    gpu_env = config.slurm.env(target="gpu")
    cpu_env = config.slurm.env(target="cpu")
    assert gpu_env["SBATCH_GPUS_PER_NODE"] == "8"
    assert "SBATCH_GPUS_PER_NODE" not in cpu_env


def test_cluster_config_from_dict_with_string_cwd():
    """Passing cwd as a string should be converted to pathlib.Path."""
    config = ClusterConfig.from_dict(
        {
            "launcher": {
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
            },
            "slurm": {
                "gpu": {"partition": "gpuq"},
            },
            "cwd": "/some/path",
        }
    )

    assert isinstance(config.cwd, pathlib.Path)
    assert config.cwd == pathlib.Path("/some/path")


def test_frozen_dataclass_rejects_mutation():
    """Frozen dataclasses should raise FrozenInstanceError on attribute assignment."""
    config = ClusterConfig.from_dict(
        {
            "launcher": {
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
            },
            "slurm": {
                "gpu": {"partition": "gpuq"},
            },
        }
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.cwd = pathlib.Path("/other")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.gpu_type = "a100"

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.slurm.gpu = None


def test_schema_v2_flattened_config_supported():
    config = ClusterConfig.from_dict(
        {
            "schema_version": 2,
            "llmb_repo": "/repo",
            "llmb_install": "/install",
            "gpu_type": "h100",
            "cluster_name": "test-cluster",
            "install": {"node_architecture": "x86_64"},
            "slurm": {
                "gpu": {"partition": "gpuq"},
            },
        }
    )

    assert config.schema_version == 2
    assert config.llmb_repo == "/repo"
    assert config.llmb_install == "/install"
    assert config.gpu_type == "h100"
    assert config.cluster_name == "test-cluster"
    assert config.install.node_architecture == "x86_64"
    assert config.slurm.partition("gpu") == "gpuq"
    assert config.slurm.partition("cpu") == "gpuq"


def test_install_node_architecture_key_supported():
    config = ClusterConfig.from_dict(
        {
            "schema_version": 2,
            "llmb_repo": "/repo",
            "llmb_install": "/install",
            "gpu_type": "h100",
            "install": {"node_architecture": "aarch64"},
            "slurm": {
                "gpu": {"partition": "gpuq"},
            },
        }
    )

    assert config.install.node_architecture == "aarch64"


def test_install_node_arch_alias_is_not_used():
    config = ClusterConfig.from_dict(
        {
            "schema_version": 2,
            "llmb_repo": "/repo",
            "llmb_install": "/install",
            "gpu_type": "h100",
            "install": {"node_arch": "aarch64"},
            "slurm": {
                "gpu": {"partition": "gpuq"},
            },
        }
    )

    # node_arch is not a supported key; keep default and preserve unknown key as extra.
    assert config.install.node_architecture == "x86_64"
    assert config.install.extras.get("node_arch") == "aarch64"


def test_schema_defaults_to_legacy_when_omitted():
    config = ClusterConfig.from_dict(
        {
            "launcher": {
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
            },
            "slurm": {
                "gpu_partition": "gpuq",
            },
        }
    )
    assert config.schema_version == 1


def test_schema_v1_rejects_empty_llmb_repo():
    with pytest.raises(ValueError, match="launcher\\.llmb_repo"):
        ClusterConfig.from_dict(
            {
                "launcher": {
                    "llmb_repo": "",
                    "llmb_install": "/install",
                    "gpu_type": "h100",
                },
                "slurm": {
                    "gpu_partition": "gpuq",
                },
            }
        )


def test_schema_v1_rejects_flattened_without_launcher():
    with pytest.raises(ValueError, match="flattened top-level keys require 'schema_version: 2'"):
        ClusterConfig.from_dict(
            {
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
                "slurm": {
                    "gpu": {"partition": "gpuq"},
                },
            }
        )


def test_schema_v1_rejects_mixed_launcher_and_flattened_keys():
    with pytest.raises(ValueError, match="legacy 'launcher' section cannot be used with flattened top-level keys"):
        ClusterConfig.from_dict(
            {
                "launcher": {
                    "llmb_repo": "/repo",
                    "llmb_install": "/install",
                    "gpu_type": "h100",
                },
                "llmb_repo": "/repo",
                "slurm": {
                    "gpu_partition": "gpuq",
                },
            }
        )


def test_schema_v2_rejects_launcher_section():
    with pytest.raises(ValueError, match="'launcher' section is not supported for schema_version: 2"):
        ClusterConfig.from_dict(
            {
                "schema_version": 2,
                "launcher": {
                    "llmb_repo": "/repo",
                    "llmb_install": "/install",
                    "gpu_type": "h100",
                },
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
                "slurm": {
                    "gpu": {"partition": "gpuq"},
                },
            }
        )


def test_schema_v2_requires_required_flattened_keys():
    with pytest.raises(ValueError, match="Missing required 'llmb_repo' field"):
        ClusterConfig.from_dict(
            {
                "schema_version": 2,
                "llmb_install": "/install",
                "gpu_type": "h100",
                "slurm": {
                    "gpu": {"partition": "gpuq"},
                },
            }
        )


def test_schema_v2_rejects_empty_llmb_repo():
    with pytest.raises(ValueError, match="Missing required 'llmb_repo' field"):
        ClusterConfig.from_dict(
            {
                "schema_version": 2,
                "llmb_repo": "  ",
                "llmb_install": "/install",
                "gpu_type": "h100",
                "slurm": {
                    "gpu": {"partition": "gpuq"},
                },
            }
        )


def test_schema_rejects_unknown_version():
    with pytest.raises(ValueError, match="Unsupported cluster configuration schema_version '3'"):
        ClusterConfig.from_dict(
            {
                "schema_version": 3,
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
                "slurm": {
                    "gpu": {"partition": "gpuq"},
                },
            }
        )


def test_schema_rejects_non_integer_version():
    with pytest.raises(ValueError, match="'schema_version' must be an integer"):
        ClusterConfig.from_dict(
            {
                "schema_version": "2",
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
                "slurm": {
                    "gpu": {"partition": "gpuq"},
                },
            }
        )


def test_schema_rejects_boolean_version():
    with pytest.raises(ValueError, match="'schema_version' must be an integer"):
        ClusterConfig.from_dict(
            {
                "schema_version": True,
                "llmb_repo": "/repo",
                "llmb_install": "/install",
                "gpu_type": "h100",
                "slurm": {
                    "gpu": {"partition": "gpuq"},
                },
            }
        )

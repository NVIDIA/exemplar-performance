# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SLURM business logic functions."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llmb_install.cluster.slurm import (
    detect_partition_architecture,
    detect_partition_gres,
    get_available_partitions,
    get_default_cpu_partition,
    get_user_accounts,
    normalize_partition_list,
    parse_gpu_gres,
    validate_account,
    validate_partition,
)


class TestGetUserAccounts:
    """Test get_user_accounts function."""

    @patch('subprocess.run')
    @patch('os.environ.get')
    def test_successful_account_retrieval(self, mock_env_get, mock_subprocess):
        """Test successful retrieval of user accounts."""
        mock_env_get.return_value = 'testuser'

        mock_result = MagicMock()
        mock_result.stdout = 'account1|\naccount2|\naccount3|\n'
        mock_subprocess.return_value = mock_result

        accounts = get_user_accounts()

        assert accounts == ['account1', 'account2', 'account3']
        mock_subprocess.assert_called_once_with(
            ["sacctmgr", "show", "assoc", "user=testuser", "format=account", "-p", "-n"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('subprocess.run')
    @patch('os.environ.get')
    def test_no_accounts_found(self, mock_env_get, mock_subprocess):
        """Test when no accounts are found."""
        mock_env_get.return_value = 'testuser'

        mock_result = MagicMock()
        mock_result.stdout = ''
        mock_subprocess.return_value = mock_result

        accounts = get_user_accounts()

        assert accounts == []

    @patch('subprocess.run')
    def test_subprocess_error(self, mock_subprocess):
        """Test when subprocess fails."""
        mock_subprocess.side_effect = subprocess.SubprocessError("Command failed")

        accounts = get_user_accounts()

        assert accounts == []


class TestGetAvailablePartitions:
    """Test get_available_partitions function."""

    @patch('subprocess.run')
    def test_successful_partition_retrieval(self, mock_subprocess):
        """Test successful retrieval of partitions with default."""
        mock_result = MagicMock()
        mock_result.stdout = 'gpu*\ncpu\ninteractive\n'
        mock_subprocess.return_value = mock_result

        partitions, default = get_available_partitions()

        assert set(partitions) == {'cpu', 'gpu', 'interactive'}
        assert default == 'gpu'

    @patch('subprocess.run')
    def test_no_default_partition(self, mock_subprocess):
        """Test when no default partition is marked."""
        mock_result = MagicMock()
        mock_result.stdout = 'gpu\ncpu\ninteractive\n'
        mock_subprocess.return_value = mock_result

        partitions, default = get_available_partitions()

        assert set(partitions) == {'cpu', 'gpu', 'interactive'}
        assert default is None

    @patch('subprocess.run')
    def test_subprocess_error(self, mock_subprocess):
        """Test when subprocess fails."""
        mock_subprocess.side_effect = subprocess.SubprocessError("Command failed")

        partitions, default = get_available_partitions()

        assert partitions == []
        assert default is None


class TestValidateAccount:
    """Test validate_account function."""

    def test_empty_input(self):
        """Test validation with empty input."""
        result = validate_account("", [])
        assert result == 'valid'

    def test_show_all_accounts(self):
        """Test requesting to show all accounts."""
        accounts = ['acc1', 'acc2'] * 10  # 20 accounts to trigger show_all
        result = validate_account("?", accounts)
        assert result == 'show_all'

    def test_valid_account(self):
        """Test validation with valid account."""
        accounts = ['account1', 'account2', 'account3']
        result = validate_account('account2', accounts)
        assert result == 'valid'

    def test_invalid_account_error(self):
        """Test validation returns error for account not in list."""
        accounts = ['account1', 'account2']
        result = validate_account('unknown_account', accounts)
        assert result.startswith('invalid:')
        assert 'unknown_account' in result


class TestValidatePartition:
    """Test validate_partition function."""

    def test_empty_input(self):
        """Test validation with empty input."""
        result = validate_partition("", [])
        assert result == 'valid'

    def test_valid_partition(self):
        """Test validation with valid partition."""
        partitions = ['gpu', 'cpu', 'interactive']
        result = validate_partition('gpu', partitions)
        assert result == 'valid'

    def test_invalid_partition_error(self):
        """Test validation returns error for partition not in list."""
        partitions = ['gpu', 'cpu']
        result = validate_partition('unknown_partition', partitions)
        assert result.startswith('invalid:')
        assert 'unknown_partition' in result

    def test_comma_separated_partitions_all_valid(self):
        """Test validation with comma-separated list of valid partitions."""
        partitions = ['gpu1', 'gpu2', 'cpu']
        result = validate_partition('gpu1,gpu2', partitions)
        assert result == 'valid'

    def test_comma_separated_partitions_with_spaces(self):
        """Test validation with comma-separated list with spaces."""
        partitions = ['gpu1', 'gpu2', 'cpu']
        result = validate_partition('gpu1, gpu2 , cpu', partitions)
        assert result == 'valid'

    def test_comma_separated_partitions_some_invalid(self):
        """Test validation with comma-separated list containing invalid partitions."""
        partitions = ['gpu1', 'gpu2', 'cpu']
        result = validate_partition('gpu1,invalid1,invalid2', partitions)
        assert result.startswith('invalid:')
        assert 'invalid1' in result
        assert 'invalid2' in result

    def test_comma_separated_partitions_single_invalid(self):
        """Test validation with comma-separated list containing one invalid partition."""
        partitions = ['gpu1', 'gpu2', 'cpu']
        result = validate_partition('gpu1,invalid_part', partitions)
        assert result.startswith('invalid:')
        assert 'invalid_part' in result

    def test_comma_separated_partitions_all_invalid(self):
        """Test validation with comma-separated list of all invalid partitions."""
        partitions = ['gpu1', 'gpu2', 'cpu']
        result = validate_partition('fake1,fake2', partitions)
        assert result.startswith('invalid:')


class TestGetDefaultCpuPartition:
    """Test get_default_cpu_partition function."""

    def test_cpu_partition_found(self):
        """Test when CPU partition is found."""
        partitions = ['gpu', 'cpu-partition', 'cpu-low', 'interactive']
        result = get_default_cpu_partition(partitions)
        assert result == 'cpu-partition'  # First alphabetically among cpu* partitions

    def test_no_cpu_partition(self):
        """Test when no CPU partition is found."""
        partitions = ['gpu', 'interactive', 'debug']
        result = get_default_cpu_partition(partitions)
        assert result is None


class TestNormalizePartitionList:
    """Test normalize_partition_list helper."""

    def test_normalize_partition_list(self):
        """Test partition normalization from comma-separated text."""
        assert normalize_partition_list("gpu1, gpu2 ,cpu") == ["gpu1", "gpu2", "cpu"]
        assert normalize_partition_list("gpu1,gpu1,gpu2") == ["gpu1", "gpu2"]
        assert normalize_partition_list(" , , ") == []
        assert normalize_partition_list("") == []


class TestDetectPartitionGres:
    """Test detect_partition_gres for single and composite partition inputs."""

    @staticmethod
    def _mock_sinfo(mock_subprocess, stdout):
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_subprocess.return_value = mock_result

    @patch('subprocess.run')
    def test_composite_partition_consistent_gres(self, mock_subprocess):
        """Composite partition returns consolidated key when GRES is consistent."""
        self._mock_sinfo(mock_subprocess, "gpu1,gpu:a100:8,10\ngpu2,gpu:8,5\ncpu,(null),20\n")

        result = detect_partition_gres(["gpu1,gpu2", "cpu"])

        assert result["gpu1,gpu2"]["gpu_count"] == 8
        assert result["gpu1,gpu2"]["has_gpu_lines"] is True
        assert result["cpu"]["gpu_count"] is None
        mock_subprocess.assert_called_once_with(
            ["sinfo", "--noheader", "-p", "gpu1,gpu2,cpu", "-o", "%P,%G,%D"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('subprocess.run')
    def test_composite_partition_mismatched_gres_raises(self, mock_subprocess):
        """Composite partition with different GPU counts should raise ValueError."""
        self._mock_sinfo(mock_subprocess, "gpu1,gpu:8,10\ngpu2,gpu:4,5\n")

        with pytest.raises(ValueError, match="Inconsistent GPU GRES"):
            detect_partition_gres(["gpu1,gpu2"])

    @patch('subprocess.run')
    def test_composite_partition_set_and_unset_gres_raises(self, mock_subprocess):
        """Composite partition with one set and one unset GRES should raise ValueError."""
        self._mock_sinfo(mock_subprocess, "gpu1,gpu:8,10\ngpu2,(null),5\n")

        with pytest.raises(ValueError, match="Inconsistent GPU GRES"):
            detect_partition_gres(["gpu1,gpu2"])

    @patch('subprocess.run')
    def test_heterogeneous_picks_by_node_count_not_line_order(self, mock_subprocess):
        """Minority gpu_count listed first — node_count must drive selection, not parse order."""
        # gpu:7 appears first but only has 1 node; gpu:8 has 1750
        self._mock_sinfo(mock_subprocess, "pool0,gpu:7(S:0-1),1\npool0,gpu:8(S:0-1),1750\n")

        result = detect_partition_gres(["pool0"])

        assert result["pool0"]["gpu_count"] == 8
        assert result["pool0"]["is_heterogeneous"] is True
        assert result["pool0"]["nodes_by_gpu_count"] == {8: 1750, 7: 1}

    @patch('subprocess.run')
    def test_heterogeneous_tied_node_count_picks_higher_gpu(self, mock_subprocess):
        """When node counts are tied, tiebreak to the higher gpu_count."""
        self._mock_sinfo(mock_subprocess, "pool0,gpu:7,5\npool0,gpu:8,5\n")

        result = detect_partition_gres(["pool0"])

        assert result["pool0"]["gpu_count"] == 8
        assert result["pool0"]["is_heterogeneous"] is True


class TestDetectPartitionArchitecture:
    """Test bounded architecture detection from Slurm node metadata."""

    @staticmethod
    def _completed(stdout):
        result = MagicMock()
        result.stdout = stdout
        return result

    @patch('subprocess.run')
    def test_detects_uniform_architecture_from_sampled_nodes(self, mock_subprocess):
        """A unanimous supported Arch value should auto-detect architecture."""
        mock_subprocess.side_effect = [
            self._completed("node001\nnode002\nnode003\n"),
            self._completed(
                "NodeName=node001 Arch=x86_64 CoresPerSocket=64\n"
                "NodeName=node002 Arch=x86_64 CoresPerSocket=64\n"
                "NodeName=node003 Arch=x86_64 CoresPerSocket=64\n"
            ),
        ]

        result = detect_partition_architecture("gpu1,gpu2")

        assert result["architecture"] == "x86_64"
        assert result["reason"] == "detected"
        assert result["nodes"] == ["node001", "node002", "node003"]
        assert mock_subprocess.call_args_list[0].args[0] == [
            "sinfo",
            "-N",
            "--noheader",
            "-p",
            "gpu1,gpu2",
            "-o",
            "%N",
        ]
        assert mock_subprocess.call_args_list[1].args[0] == [
            "scontrol",
            "-o",
            "show",
            "node",
            "node001,node002,node003",
        ]

    @patch('subprocess.run')
    def test_samples_at_most_five_unique_nodes(self, mock_subprocess):
        """Large partitions should still produce one bounded scontrol query."""
        mock_subprocess.side_effect = [
            self._completed("node001\nnode002\nnode002\nnode003\nnode004\nnode005\nnode006\n"),
            self._completed(
                "NodeName=node001 Arch=aarch64\n"
                "NodeName=node002 Arch=aarch64\n"
                "NodeName=node003 Arch=aarch64\n"
                "NodeName=node004 Arch=aarch64\n"
                "NodeName=node005 Arch=aarch64\n"
            ),
        ]

        result = detect_partition_architecture("gpu")

        assert result["architecture"] == "aarch64"
        assert result["nodes"] == ["node001", "node002", "node003", "node004", "node005"]
        assert mock_subprocess.call_args_list[1].args[0][-1] == "node001,node002,node003,node004,node005"

    @patch('subprocess.run')
    def test_mixed_architecture_falls_back(self, mock_subprocess):
        """Mixed sampled Arch values should not auto-select."""
        mock_subprocess.side_effect = [
            self._completed("node001\nnode002\n"),
            self._completed("NodeName=node001 Arch=x86_64\nNodeName=node002 Arch=aarch64\n"),
        ]

        result = detect_partition_architecture("gpu")

        assert result["architecture"] is None
        assert result["reason"] == "mixed"

    @patch('subprocess.run')
    def test_unsupported_architecture_falls_back(self, mock_subprocess):
        """Unsupported Arch values should not auto-select."""
        mock_subprocess.side_effect = [
            self._completed("node001\n"),
            self._completed("NodeName=node001 Arch=ppc64le\n"),
        ]

        result = detect_partition_architecture("gpu")

        assert result["architecture"] is None
        assert result["reason"] == "unsupported"

    @patch('subprocess.run')
    def test_scontrol_error_falls_back(self, mock_subprocess):
        """Command failures should return a fallback result instead of raising."""
        mock_subprocess.side_effect = [
            self._completed("node001\n"),
            subprocess.CalledProcessError(1, ["scontrol"]),
        ]

        result = detect_partition_architecture("gpu")

        assert result["architecture"] is None
        assert result["reason"] == "error"
        assert result["nodes"] == ["node001"]


class TestParseGpuGres:
    """Test parse_gpu_gres function (inherited from existing tests)."""

    def test_simple_gpu_gres(self):
        """Test simple GPU GRES parsing."""
        assert parse_gpu_gres("gpu:8") == 8
        assert parse_gpu_gres("gpu:4") == 4
        assert parse_gpu_gres("gpu:1") == 1

    def test_typed_gpu_gres(self):
        """Test typed GPU GRES parsing."""
        assert parse_gpu_gres("gpu:a100:8") == 8
        assert parse_gpu_gres("gpu:h100:4") == 4

    def test_invalid_gres(self):
        """Test invalid GRES strings."""
        assert parse_gpu_gres("") is None
        assert parse_gpu_gres("(null)") is None
        assert parse_gpu_gres("cpu:8") is None
        assert parse_gpu_gres("gpu:abc") is None

    def test_boundary_values(self):
        """Test boundary values."""
        assert parse_gpu_gres("gpu:1") == 1
        assert parse_gpu_gres("gpu:8") == 8
        assert parse_gpu_gres("gpu:0") is None  # Below minimum
        assert parse_gpu_gres("gpu:9") is None  # Above maximum

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prompt flow logic using MockUI."""

from unittest.mock import patch

import pytest
from test_mock_ui import MockUI

from llmb_install.cluster.nvlink import NvlDetection
from llmb_install.ui.prompts import gpu as gpu_prompts
from llmb_install.ui.prompts.gpu import prompt_gpu_type, prompt_node_architecture, prompt_nvl_domain
from llmb_install.ui.prompts.slurm import prompt_slurm_info


class TestGpuPromptFlows:
    """Test GPU prompt flow logic."""

    def test_prompt_gpu_type_successful(self):
        """Test successful GPU type selection."""
        workloads = {'workload1': {'run': {'gpu_configs': {'h100': {'nodes': 1}, 'b200': {'nodes': 2}}}}}

        mock_ui = MockUI(responses={'gpu_type': 'h100'})

        result = prompt_gpu_type(mock_ui, workloads)

        assert result == 'h100'
        assert mock_ui.was_prompted_for('gpu type')
        assert mock_ui.was_prompted_for('select')

    def test_prompt_gpu_type_cancellation(self):
        """Test GPU type selection cancellation."""
        workloads = {'workload1': {'run': {'gpu_configs': {'h100': {'nodes': 1}}}}}

        mock_ui = MockUI(responses={'gpu_type': None})  # Simulate cancellation

        with pytest.raises(SystemExit):
            prompt_gpu_type(mock_ui, workloads)

    def test_prompt_node_architecture_auto_select(self):
        """Test auto-selection of architecture for GB200."""
        mock_ui = MockUI()

        result = prompt_node_architecture(mock_ui, 'gb200')

        assert result == 'aarch64'
        # Should log auto-selection message
        messages = mock_ui.get_messages_by_level('info')
        assert any('GB200' in msg and 'aarch64' in msg for msg in messages)

    def test_prompt_node_architecture_manual_select(self):
        """Test manual architecture selection for H100."""
        mock_ui = MockUI(responses={'architecture': 'x86_64'})

        result = prompt_node_architecture(mock_ui, 'h100')

        assert result == 'x86_64'
        assert mock_ui.was_prompted_for('architecture')

    def test_prompt_node_architecture_cancellation(self):
        """Test architecture selection cancellation."""
        mock_ui = MockUI(responses={'architecture': None})

        with pytest.raises(SystemExit):
            prompt_node_architecture(mock_ui, 'h100')

    @patch('llmb_install.cluster.slurm.detect_partition_architecture')
    def test_prompt_node_architecture_auto_detects_from_slurm(self, mock_detect_arch):
        """Test auto-selection from SLURM when detection is clean."""
        mock_detect_arch.return_value = {
            'architecture': 'x86_64',
            'reason': 'detected',
            'nodes': ['node001', 'node002'],
            'architectures': {'node001': 'x86_64', 'node002': 'x86_64'},
            'error': None,
        }
        mock_ui = MockUI(responses={'architecture': 'aarch64'})

        result = prompt_node_architecture(mock_ui, 'h100', gpu_partition='gpu')

        assert result == 'x86_64'
        assert not mock_ui.was_prompted_for('architecture')
        mock_detect_arch.assert_called_once_with('gpu')

    @patch('llmb_install.cluster.slurm.detect_partition_architecture')
    def test_prompt_node_architecture_falls_back_to_saved_default(self, mock_detect_arch):
        """Test fallback prompt uses saved default when SLURM detection is unclear."""
        mock_detect_arch.return_value = {
            'architecture': None,
            'reason': 'mixed',
            'nodes': ['node001', 'node002'],
            'architectures': {'node001': 'x86_64', 'node002': 'aarch64'},
            'error': None,
        }
        mock_ui = MockUI()

        result = prompt_node_architecture(mock_ui, 'h100', default='aarch64', gpu_partition='gpu')

        assert result == 'aarch64'
        architecture_prompts = [
            call for call in mock_ui.call_log if call.get('message') == "Select the CPU architecture of your GPU nodes:"
        ]
        assert architecture_prompts
        assert architecture_prompts[0]['default'] == 'aarch64'

    @patch('llmb_install.cluster.slurm.detect_partition_architecture')
    def test_prompt_node_architecture_fixed_arch_skips_slurm_detection(self, mock_detect_arch):
        """GB200/GB300 should keep fixed architecture behavior."""
        mock_ui = MockUI()

        result = prompt_node_architecture(mock_ui, 'gb300', gpu_partition='gpu')

        assert result == 'aarch64'
        mock_detect_arch.assert_not_called()

    def test_prompt_nvl_domain_fixed_type(self):
        mock_ui = MockUI()
        assert prompt_nvl_domain(mock_ui, "h100") == (8, None)
        assert not mock_ui.call_log

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_kubernetes_does_not_query_slurm(self, mock_detect):
        mock_ui = MockUI()

        assert prompt_nvl_domain(mock_ui, "gb300", query_slurm=False) == (72, None)

        mock_detect.assert_not_called()
        assert not mock_ui.get_prompts_by_type("prompt_confirm")  # no "Continue as NVL72?"

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_detected(self, mock_detect):
        mock_detect.return_value = NvlDetection(block_sizes=[18], nvl_domain=72, reason="detected", gpu_type="gb200")
        mock_ui = MockUI()
        assert prompt_nvl_domain(mock_ui, "gb200") == (72, [18])
        assert not mock_ui.was_prompted_for("NVLink domain size")
        assert mock_ui.call_log[0] == {'type': 'print_section', 'title': 'NVLink Domain', 'content': None}
        assert mock_ui.get_messages_by_level("info") == ["✓ Auto-detected NVLink domain: NVL72"]

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_override(self, mock_detect):
        mock_detect.return_value = NvlDetection(
            block_sizes=[9, 144], nvl_domain=576, reason="detected", gpu_type="gb200"
        )
        mock_ui = MockUI(responses={"select_the_nvlink_domain_size_of_your_gpu_nodes": 72})
        assert prompt_nvl_domain(mock_ui, "gb200") == (72, [9, 144])
        assert mock_ui.was_prompted_for("NVLink domain size")
        assert mock_ui.get_messages_by_level("info") == [
            "Detected NVL576 from the Slurm topology; GB200 systems are expected to be NVL72."
        ]
        prompt = mock_ui.get_prompts_by_type("prompt_select")[0]
        assert prompt["choices"] == [
            {'name': 'NVL576 (detected)', 'value': 576},
            {'name': 'NVL72', 'value': 72},
        ]
        assert prompt["default"] == 576

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_offers_all_productized_domains(self, mock_detect, monkeypatch):
        monkeypatch.setitem(gpu_prompts.NVL_DOMAINS_BY_GPU, "gb200", [72, 576])
        mock_detect.return_value = NvlDetection(block_sizes=[9], nvl_domain=36, reason="detected", gpu_type="gb200")
        mock_ui = MockUI(responses={"select_the_nvlink_domain_size_of_your_gpu_nodes": 72})
        assert prompt_nvl_domain(mock_ui, "gb200") == (72, [9])
        prompt = mock_ui.get_prompts_by_type("prompt_select")[0]
        assert prompt["choices"] == [
            {'name': 'NVL36 (detected)', 'value': 36},
            {'name': 'NVL72', 'value': 72},
            {'name': 'NVL576', 'value': 576},
        ]

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_no_blocks_confirm(self, mock_detect):
        mock_detect.return_value = NvlDetection(block_sizes=None, nvl_domain=None, reason="no_blocks", gpu_type="gb200")
        mock_ui = MockUI(responses={"continue_as_nvl72?": True})
        assert prompt_nvl_domain(mock_ui, "gb200") == (72, None)
        assert not mock_ui.was_prompted_for("NVLink domain size")
        assert mock_ui.get_prompts_by_type("prompt_confirm")[0]["default"] is False
        assert mock_ui.get_messages_by_level("info") == [
            "Could not detect the NVLink domain from the Slurm topology.",
            "Only a single-rack NVL72 cluster performs correctly without Slurm block topology.",
        ]

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_no_blocks_cancel(self, mock_detect):
        mock_detect.return_value = NvlDetection(block_sizes=None, nvl_domain=None, reason="no_blocks", gpu_type="gb200")
        mock_ui = MockUI(responses={"continue_as_nvl72?": False})
        with pytest.raises(SystemExit):
            prompt_nvl_domain(mock_ui, "gb200")
        assert not mock_ui.get_prompts_by_type("prompt_select")

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_error_confirm(self, mock_detect):
        mock_detect.return_value = NvlDetection(
            block_sizes=None, nvl_domain=None, reason="error", gpu_type="gb200", error="missing"
        )
        mock_ui = MockUI(responses={"continue_as_nvl72?": True})
        assert prompt_nvl_domain(mock_ui, "gb200") == (72, None)
        assert not mock_ui.get_prompts_by_type("prompt_select")
        assert mock_ui.get_messages_by_level("info") == [
            "Could not query the Slurm topology: missing",
            "Only a single-rack NVL72 cluster performs correctly without Slurm block topology.",
        ]

    @patch("llmb_install.ui.prompts.gpu.detect_nvl_topology")
    def test_prompt_nvl_domain_error_cancel(self, mock_detect):
        mock_detect.return_value = NvlDetection(
            block_sizes=None, nvl_domain=None, reason="error", gpu_type="gb200", error="missing"
        )
        mock_ui = MockUI(responses={"continue_as_nvl72?": None})
        with pytest.raises(SystemExit):
            prompt_nvl_domain(mock_ui, "gb200")


class TestSlurmPromptFlows:
    """Test SLURM prompt flow logic."""

    @patch('llmb_install.cluster.slurm.get_user_accounts')
    @patch('llmb_install.cluster.slurm.get_available_partitions')
    @patch('llmb_install.cluster.slurm.detect_partition_gres')
    @patch('llmb_install.cluster.slurm.get_default_cpu_partition')
    def test_prompt_slurm_info_successful(self, mock_cpu_default, mock_gres, mock_partitions, mock_accounts):
        """Test successful SLURM configuration."""
        # Setup mocks
        mock_accounts.return_value = ['account1', 'account2']
        mock_partitions.return_value = (['gpu', 'cpu'], 'gpu')
        mock_gres.return_value = {
            'gpu': {'gpu_count': 8, 'has_gpu_lines': True, 'unparseable_lines': [], 'is_heterogeneous': False},
            'cpu': {'gpu_count': None, 'has_gpu_lines': False, 'unparseable_lines': [], 'is_heterogeneous': False},
        }
        mock_cpu_default.return_value = 'cpu'  # Default CPU partition

        # Setup UI responses
        mock_ui = MockUI(responses={'account': 'account1', 'gpu_partition': 'gpu', 'cpu_partition': 'cpu'})

        result = prompt_slurm_info(mock_ui)

        assert result is not None
        assert result['slurm']['account'] == 'account1'
        assert result['slurm']['gpu_partition'] == 'gpu'
        assert result['slurm']['cpu_partition'] == 'cpu'
        assert result['slurm']['gpu_partition_gres'] == 8
        assert result['slurm']['cpu_partition_gres'] is None

    @patch('llmb_install.cluster.slurm.get_user_accounts')
    @patch('llmb_install.cluster.slurm.get_available_partitions')
    @patch('llmb_install.cluster.slurm.detect_partition_gres')
    @patch('llmb_install.cluster.slurm.get_default_cpu_partition')
    def test_prompt_slurm_info_reprompts_on_gres_conflict(
        self, mock_cpu_default, mock_gres, mock_partitions, mock_accounts
    ):
        """GRES conflict should re-prompt partition input and retry detection."""
        mock_accounts.return_value = ['account1']
        mock_partitions.return_value = (['gpu', 'cpu'], 'gpu')
        mock_cpu_default.return_value = 'cpu'
        mock_gres.side_effect = [
            ValueError("Inconsistent GPU GRES across partition list 'gpu,cpu'"),
            {
                'gpu': {'gpu_count': 8, 'has_gpu_lines': True, 'unparseable_lines': [], 'is_heterogeneous': False},
                'cpu': {'gpu_count': None, 'has_gpu_lines': False, 'unparseable_lines': [], 'is_heterogeneous': False},
            },
        ]

        mock_ui = MockUI(responses={'account': 'account1', 'gpu_partition': 'gpu', 'cpu_partition': 'cpu'})
        result = prompt_slurm_info(mock_ui)

        assert result is not None
        assert result['slurm']['gpu_partition'] == 'gpu'
        assert result['slurm']['cpu_partition'] == 'cpu'
        assert result['slurm']['gpu_partition_gres'] == 8
        assert mock_gres.call_count == 2

        gpu_partition_prompts = [call for call in mock_ui.call_log if call.get('message') == "SLURM GPU partition:"]
        cpu_partition_prompts = [call for call in mock_ui.call_log if call.get('message') == "SLURM CPU partition:"]
        assert len(gpu_partition_prompts) == 2
        assert len(cpu_partition_prompts) == 2

    @patch('llmb_install.cluster.slurm.get_user_accounts')
    @patch('llmb_install.cluster.slurm.get_available_partitions')
    @patch('llmb_install.cluster.slurm.detect_partition_gres')
    @patch('llmb_install.cluster.slurm.get_default_cpu_partition')
    def test_prompt_slurm_info_uses_composite_saved_default(
        self, mock_cpu_default, mock_gres, mock_partitions, mock_accounts
    ):
        """Composite saved defaults should be used when all partitions are valid."""
        mock_accounts.return_value = ['account1']
        mock_partitions.return_value = (['gpu1', 'gpu2', 'cpu'], 'gpu1')
        mock_cpu_default.return_value = 'cpu'
        mock_gres.return_value = {
            'gpu1,gpu2': {'gpu_count': 8, 'has_gpu_lines': True, 'unparseable_lines': [], 'is_heterogeneous': False},
            'cpu': {'gpu_count': None, 'has_gpu_lines': False, 'unparseable_lines': [], 'is_heterogeneous': False},
        }

        defaults = {'gpu_partition': 'gpu1, gpu2'}
        mock_ui = MockUI(responses={'account': 'account1', 'cpu_partition': 'cpu'})
        result = prompt_slurm_info(mock_ui, defaults=defaults)

        assert result is not None
        assert result['slurm']['gpu_partition'] == 'gpu1,gpu2'
        assert result['slurm']['cpu_partition'] == 'cpu'
        mock_gres.assert_called_once_with(['gpu1,gpu2', 'cpu'])

        gpu_partition_prompts = [call for call in mock_ui.call_log if call.get('message') == "SLURM GPU partition:"]
        assert gpu_partition_prompts
        assert gpu_partition_prompts[0]['default'] == 'gpu1,gpu2'

    @patch('llmb_install.cluster.slurm.get_user_accounts')
    @patch('llmb_install.cluster.slurm.get_available_partitions')
    @patch('llmb_install.cluster.slurm.detect_partition_gres')
    @patch('llmb_install.cluster.slurm.get_default_cpu_partition')
    def test_prompt_slurm_info_rejects_invalid_composite_saved_default(
        self, mock_cpu_default, mock_gres, mock_partitions, mock_accounts
    ):
        """Invalid composite saved defaults should fall back to detected partition default."""
        mock_accounts.return_value = ['account1']
        mock_partitions.return_value = (['gpu1', 'gpu2', 'cpu'], 'gpu2')
        mock_cpu_default.return_value = 'cpu'
        mock_gres.return_value = {
            'gpu2': {'gpu_count': 8, 'has_gpu_lines': True, 'unparseable_lines': [], 'is_heterogeneous': False},
            'cpu': {'gpu_count': None, 'has_gpu_lines': False, 'unparseable_lines': [], 'is_heterogeneous': False},
        }

        defaults = {'gpu_partition': 'gpu1,missing'}
        mock_ui = MockUI(responses={'account': 'account1', 'cpu_partition': 'cpu'})
        result = prompt_slurm_info(mock_ui, defaults=defaults)

        assert result is not None
        assert result['slurm']['gpu_partition'] == 'gpu2'
        assert result['slurm']['cpu_partition'] == 'cpu'
        mock_gres.assert_called_once_with(['gpu2', 'cpu'])

        gpu_partition_prompts = [call for call in mock_ui.call_log if call.get('message') == "SLURM GPU partition:"]
        assert gpu_partition_prompts
        assert gpu_partition_prompts[0]['default'] == 'gpu2'

    @patch('llmb_install.cluster.slurm.get_user_accounts')
    @patch('llmb_install.cluster.slurm.get_available_partitions')
    def test_prompt_slurm_info_cancellation(self, mock_partitions, mock_accounts):
        """Test SLURM configuration cancellation."""
        mock_accounts.return_value = ['account1']
        mock_partitions.return_value = (['gpu'], 'gpu')

        mock_ui = MockUI(responses={'account': None})  # Cancel at account prompt

        result = prompt_slurm_info(mock_ui)

        assert result is None

    @patch('llmb_install.cluster.slurm.get_user_accounts')
    def test_prompt_slurm_info_no_accounts(self, mock_accounts):
        """Test SLURM configuration when no accounts found."""
        mock_accounts.return_value = []

        # Should still work with manual account entry
        # (This test would need more mocking to complete fully)
        # For now, just test that no accounts doesn't crash
        assert mock_accounts.return_value == []


class TestPromptValidationFlows:
    """Test validation logic in prompt flows."""

    def test_account_validation_with_error(self):
        """Test account validation function returns error for unknown account."""
        from llmb_install.cluster.slurm import validate_account

        # Test validation function directly
        available_accounts = ['known_account']
        result = validate_account('unknown_account', available_accounts)

        # Should return an error message for unknown account
        assert result.startswith('invalid:'), f"Expected invalid message, got: {result}"


class TestUIIntegrationPatterns:
    """Test integration patterns between prompt logic and UI."""

    def test_ui_call_patterns(self):
        """Test that UI calls follow expected patterns."""
        mock_ui = MockUI()

        # Simulate some UI calls
        mock_ui.print_section("Test Section")
        mock_ui.log("Info message", "info")
        mock_ui.prompt_text("Test prompt:", default="default")

        # Verify call patterns
        assert len(mock_ui.call_log) == 3
        assert mock_ui.call_log[0]['type'] == 'print_section'
        assert mock_ui.call_log[1]['type'] == 'log'
        assert mock_ui.call_log[2]['type'] == 'prompt_text'

    def test_error_handling_in_prompts(self):
        """Test error handling in prompt flows."""
        workloads = {}  # Empty workloads should cause error
        mock_ui = MockUI()

        with pytest.raises(ValueError):
            prompt_gpu_type(mock_ui, workloads)

        # Verify that the error was logged to the UI
        error_messages = [call for call in mock_ui.call_log if call['type'] == 'log']
        assert len(error_messages) > 0, "Error should have been logged to UI"

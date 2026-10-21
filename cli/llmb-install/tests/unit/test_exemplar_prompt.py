# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Exemplar Cloud prompt flow logic using MockUI."""

from test_mock_ui import MockUI

from llmb_install.ui.prompts.workload import prompt_workload_selection


class TestExemplarPromptFlows:
    """Test Exemplar Cloud prompt flow logic."""

    def test_exemplar_option_prompt(self, tmp_path):
        """Test that prompt appears when show_exemplar_option is True."""
        workloads = {
            'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}},
            'finetune_llama': {'general': {'workload_type': 'finetune'}},
        }

        # Create exemplar.yaml with the pretrain workload
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_gpt3_70b
""")

        # User selects 'exemplar'
        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar'})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should return only pretrain workloads from exemplar.yaml
        assert result == ['pretrain_gpt3']
        assert mode == 'exemplar'
        assert mock_ui.was_prompted_for('workload selection mode')

    def test_exemplar_filtering(self, tmp_path):
        """Test that exemplar option filters correctly."""
        workloads = {
            'pretrain_model_a': {'general': {'workload_type': 'pretrain'}},
            'pretrain_model_b': {'general': {'workload_type': 'pretrain'}},
            'finetune_model_c': {'general': {'workload_type': 'finetune'}},
            'inference_model_d': {'general': {'workload_type': 'inference'}},
        }

        # Create exemplar.yaml with selected workloads
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_model_a_70b
    - pretrain_model_b_8b
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar'})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        assert sorted(result) == ['pretrain_model_a', 'pretrain_model_b']
        assert mode == 'exemplar'

    def test_custom_option_flow(self):
        """Test that selecting Custom flows to manual selection."""
        workloads = {
            'workload1': {'general': {'workload_type': 'custom'}},
            'workload2': {'general': {'workload_type': 'custom'}},
        }

        # First prompt: custom, Second prompt (checkbox): select workload1
        mock_ui = MockUI(responses={'workload_selection_mode': 'custom', 'select_workloads': ['workload1']})

        result, mode = prompt_workload_selection(mock_ui, workloads, show_exemplar_option=True)

        assert result == ['workload1']
        assert mode == 'custom'
        assert mock_ui.was_prompted_for('workload selection mode')
        assert mock_ui.was_prompted_for('select workloads')

    def test_default_behavior_no_exemplar(self):
        """Test that without show_exemplar_option, it goes straight to manual selection."""
        workloads = {'workload1': {'general': {'workload_type': 'custom'}}}

        mock_ui = MockUI(responses={'select_workloads': ['workload1']})

        result, mode = prompt_workload_selection(mock_ui, workloads, show_exemplar_option=False)

        assert result == ['workload1']
        assert mode == 'custom'
        # Should NOT have prompted for mode selection
        assert not mock_ui.was_prompted_for('workload selection mode')
        assert mock_ui.was_prompted_for('select workloads')

    def test_exemplar_no_pretrain_found(self, tmp_path):
        """Test behavior when exemplar selected but exemplar.yaml is missing or invalid."""
        workloads = {'finetune_only': {'general': {'workload_type': 'finetune'}}}

        # No exemplar.yaml created, so it will error and fall back
        # Select exemplar, then fallback to manual selection of 'finetune_only'
        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar', 'select_workloads': ['finetune_only']})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should log error about missing exemplar.yaml and fall back to manual selection
        assert result == ['finetune_only']
        assert mode == 'custom'  # Fell back to custom
        messages = mock_ui.get_messages_by_level('error')
        assert any('exemplar.yaml not found' in msg for msg in messages)

    def test_exemplar_dedup_size_variants(self, tmp_path):
        """Test that multiple size variants of same workload are deduplicated to one base key."""
        workloads = {
            'pretrain_llama3.1': {'general': {'workload_type': 'pretrain'}},
            'finetune_mistral': {'general': {'workload_type': 'finetune'}},
        }

        # Create exemplar.yaml with multiple size variants
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_llama3.1_70b
    - pretrain_llama3.1_405b
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar'})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should return only one base key for both size variants
        assert result == ['pretrain_llama3.1']
        assert mode == 'exemplar'

    def test_exemplar_duplicate_entries_tolerated(self, tmp_path):
        """Test that duplicate workload entries in YAML still yield one base key."""
        workloads = {
            'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}},
        }

        # Create exemplar.yaml with duplicate entries
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_gpt3_70b
    - pretrain_gpt3_70b
    - pretrain_gpt3_70b
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar'})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should return only one base key despite duplicates
        assert result == ['pretrain_gpt3']
        assert mode == 'exemplar'

    def test_exemplar_values_ignored(self, tmp_path):
        """Test that YAML values (dtypes, etc.) are ignored by installer - only keys matter."""
        workloads = {
            'pretrain_model_a': {'general': {'workload_type': 'pretrain'}},
            'pretrain_model_b': {'general': {'workload_type': 'pretrain'}},
            'pretrain_model_c': {'general': {'workload_type': 'pretrain'}},
        }

        # Create exemplar.yaml with various value formats (empty dict, missing dtypes, garbage)
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_model_a_70b: {}
    - pretrain_model_b_8b:
        garbage: "ignored"
        more_garbage: 123
    - pretrain_model_c_15b:
        dtypes: []
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar'})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should select all workloads by keys, ignoring all value content
        assert sorted(result) == ['pretrain_model_a', 'pretrain_model_b', 'pretrain_model_c']
        assert mode == 'exemplar'

    def test_exemplar_missing_yaml_file(self, tmp_path):
        """Test error when exemplar.yaml doesn't exist."""
        workloads = {'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}}}

        # Do NOT create exemplar.yaml
        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar', 'select_workloads': ['pretrain_gpt3']})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should fall back to custom mode
        assert mode == 'custom'
        messages = mock_ui.get_messages_by_level('error')
        assert any('exemplar.yaml not found' in msg for msg in messages)
        assert any(str(tmp_path) in msg for msg in messages)  # Should include the path

    def test_exemplar_gpu_type_missing(self, tmp_path):
        """Test error when gpu_type section is missing from exemplar.yaml."""
        workloads = {'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}}}

        # Create exemplar.yaml without the requested gpu_type
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  gb200:
    - pretrain_gpt3_70b
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar', 'select_workloads': ['pretrain_gpt3']})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should fall back to custom mode
        assert mode == 'custom'
        messages = mock_ui.get_messages_by_level('error')
        assert any('h100' in msg.lower() and 'not found' in msg for msg in messages)

    def test_exemplar_gpu_type_empty(self, tmp_path):
        """Test error when gpu_type section is empty in exemplar.yaml."""
        workloads = {'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}}}

        # Create exemplar.yaml with empty gpu_type list
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100: []
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar', 'select_workloads': ['pretrain_gpt3']})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should fall back to custom mode
        assert mode == 'custom'
        messages = mock_ui.get_messages_by_level('error')
        assert any('empty' in msg.lower() for msg in messages)

    def test_exemplar_size_suffix_case_insensitive(self, tmp_path):
        """Test that size suffix validation is case-insensitive (70B is accepted as 70b)."""
        workloads = {'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}}}

        # Create exemplar.yaml with uppercase suffix (should be accepted)
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_gpt3_70B
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar'})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should accept uppercase B (case-insensitive validation)
        assert result == ['pretrain_gpt3']
        assert mode == 'exemplar'

    def test_exemplar_trillion_size_suffix(self, tmp_path):
        """Test that trillion-size suffixes are accepted."""
        workloads = {'pretrain_kimi-k2': {'general': {'workload_type': 'pretrain'}}}

        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_kimi-k2_1t
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar'})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        assert result == ['pretrain_kimi-k2']
        assert mode == 'exemplar'

    def test_exemplar_invalid_size_suffix_missing_underscore(self, tmp_path):
        """Test error when workload name doesn't have underscore separator."""
        workloads = {'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}}}

        # Create exemplar.yaml with missing underscore
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_gpt370b
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar', 'select_workloads': ['pretrain_gpt3']})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should fall back to custom mode with error
        assert mode == 'custom'
        messages = mock_ui.get_messages_by_level('error')
        # The error should be about invalid suffix format (gpt370b doesn't match \d+(\.\d+)?[bt])
        assert any('pretrain_gpt370b' in msg for msg in messages)

    def test_exemplar_invalid_size_suffix_no_size(self, tmp_path):
        """Test error when workload name doesn't end with size suffix."""
        workloads = {'pretrain_gpt3': {'general': {'workload_type': 'pretrain'}}}

        # Create exemplar.yaml with no size suffix
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_gpt3_invalid
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar', 'select_workloads': ['pretrain_gpt3']})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should fall back to custom mode with error
        assert mode == 'custom'
        messages = mock_ui.get_messages_by_level('error')
        assert any('invalid' in msg and 'does not match' in msg for msg in messages)

    def test_exemplar_workload_not_in_metadata(self, tmp_path):
        """Test error when workload from YAML is not in available workloads for gpu_type."""
        workloads = {
            'pretrain_llama3.1': {'general': {'workload_type': 'pretrain'}},
        }

        # Create exemplar.yaml with workload not in the metadata
        exemplar_yaml = tmp_path / "exemplar.yaml"
        exemplar_yaml.write_text("""
workloads:
  h100:
    - pretrain_nonexistent_70b
""")

        mock_ui = MockUI(responses={'workload_selection_mode': 'exemplar', 'select_workloads': ['pretrain_llama3.1']})

        result, mode = prompt_workload_selection(
            mock_ui, workloads, show_exemplar_option=True, llmb_repo=tmp_path, gpu_type='h100'
        )

        # Should fall back to custom mode with error
        assert mode == 'custom'
        messages = mock_ui.get_messages_by_level('error')
        assert any('pretrain_nonexistent' in msg and 'not available' in msg for msg in messages)

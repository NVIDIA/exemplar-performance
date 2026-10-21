# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for installation path validation."""

import os
import tempfile
from unittest.mock import MagicMock, patch


class TestInstallPathValidation:
    """Test installation path validation logic."""

    def test_empty_path_rejected(self):
        """Test that empty string is rejected with helpful message."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        # Simulate user entering empty string then cancelling
        mock_ui.prompt_path.return_value = None

        result = prompt_install_location(mock_ui)

        assert result is None
        # Check that validator was called
        mock_ui.prompt_path.assert_called_once()
        validate_func = mock_ui.prompt_path.call_args[1]['validate']

        # Test the validation function directly
        validation_result = validate_func("")
        assert validation_result.startswith("invalid:")
        assert "Path required" in validation_result
        assert "/scratch/username/llmb" in validation_result

    def test_yes_no_confusion_rejected(self):
        """Test that 'yes', 'y', etc. are rejected as invalid paths."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)
        validate_func = mock_ui.prompt_path.call_args[1]['validate']

        # Test various confusing inputs
        for invalid_input in ["yes", "y", "Y", "YES", "ok", "Ok"]:
            validation_result = validate_func(invalid_input)
            assert validation_result.startswith("invalid:")
            assert "must be absolute" in validation_result.lower()

    def test_relative_path_rejected(self):
        """Test that relative paths are rejected."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)
        validate_func = mock_ui.prompt_path.call_args[1]['validate']

        # Test relative paths
        for relative_path in ["llmb", "./llmb", "../llmb", "foo/bar"]:
            validation_result = validate_func(relative_path)
            assert validation_result.startswith("invalid:")
            assert "must be absolute" in validation_result.lower()

    def test_root_directory_rejected(self):
        """Test that root directory / is explicitly rejected."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)
        validate_func = mock_ui.prompt_path.call_args[1]['validate']

        validation_result = validate_func("/")
        assert validation_result.startswith("invalid:")
        assert "Cannot install to root directory" in validation_result

    def test_tilde_expansion_accepted(self):
        """Test that ~/llmb is accepted and expanded."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = "~/llmb-test"

        with patch('os.access', return_value=True):
            result = prompt_install_location(mock_ui)

        # Should be expanded to absolute path
        assert result is not None
        assert result.startswith("/")
        assert "~" not in result

    def test_absolute_path_accepted(self):
        """Test that absolute paths are accepted."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()

        # Use a temp directory to ensure parent exists
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "llmb-test")
            mock_ui.prompt_path.return_value = test_path

            result = prompt_install_location(mock_ui)

            assert result is not None
            assert os.path.isabs(result)

    def test_incomplete_path_during_typing(self):
        """Test that incomplete paths ending with / are allowed during typing."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)
        validate_func = mock_ui.prompt_path.call_args[1]['validate']

        # Test incomplete paths (but not root). Mock write access so this
        # assertion is stable across environments (e.g., macOS sandboxed /tmp).
        with patch('os.access', return_value=True):
            assert validate_func("/tmp/") == "valid"
            assert validate_func("/foo/bar/") == "valid"
        # Root should still be rejected
        assert validate_func("/").startswith("invalid:")

    def test_permission_check_on_existing_parent(self):
        """Test that write permission is checked on existing parent directory."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)
        validate_func = mock_ui.prompt_path.call_args[1]['validate']

        # Test with existing parent directory that lacks write permission
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "llmb-test")
            with patch('os.path.exists', return_value=True):
                with patch('os.access', return_value=False):
                    validation_result = validate_func(test_path)
                    assert validation_result.startswith("invalid:")
                    assert "No write permission" in validation_result

    def test_nonexistent_parent_allowed(self):
        """Test that paths with non-existent parents are allowed."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)
        validate_func = mock_ui.prompt_path.call_args[1]['validate']

        # Non-existent parent should be allowed (can be created later)
        with patch('os.path.exists', return_value=False):
            validation_result = validate_func("/nonexistent/parent/llmb")
            assert validation_result == "valid"

    def test_current_directory_shown(self):
        """Test that current directory is shown in the UI log."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)

        # Check that current directory was logged
        log_calls = mock_ui.log.call_args_list
        current_dir = os.getcwd()
        assert any(current_dir in str(call) for call in log_calls)

    def test_prompt_message_is_clear(self):
        """Test that the prompt message is action-oriented."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()
        mock_ui.prompt_path.return_value = None

        prompt_install_location(mock_ui)

        # Check the prompt message
        prompt_call = mock_ui.prompt_path.call_args
        prompt_message = prompt_call[0][0]
        assert "Enter installation directory path:" in prompt_message
        # Should not be a yes/no question
        assert "would you like" not in prompt_message.lower()

    def test_quotes_stripped_from_path(self):
        """Test that quotes are stripped from the path."""
        from llmb_install.ui.prompts.installation import prompt_install_location

        mock_ui = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = f'"{tmpdir}/llmb-test"'
            mock_ui.prompt_path.return_value = test_path

            result = prompt_install_location(mock_ui)

            # Quotes should be stripped
            assert '"' not in result
            assert "'" not in result

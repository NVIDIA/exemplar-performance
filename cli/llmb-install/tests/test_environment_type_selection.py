#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for recipe environment type selection in the installer."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from llmb_install.ui.prompts.environment import prompt_environment_type

sys.path.insert(0, str(Path(__file__).parent / "unit"))
from test_mock_ui import MockUI


class TestEnvironmentTypeSelection(unittest.TestCase):
    """Test uv-only environment type resolution for fresh installs."""

    def setUp(self):
        """Set up common test fixtures."""
        self.print_patcher = patch('builtins.print')
        self.mock_print = self.print_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.print_patcher.stop()

    @patch('llmb_install.ui.prompts.environment.is_uv_installed')
    def test_uv_selected_without_prompt(self, mock_is_uv_installed):
        """Fresh installs use uv without asking users to choose an environment manager."""
        mock_is_uv_installed.return_value = True

        mock_ui = MockUI(responses={"environment": "conda"})
        result = prompt_environment_type(mock_ui)

        self.assertEqual(result, 'uv')
        self.assertFalse(mock_ui.was_prompted_for("environment"))
        self.assertTrue(any("Using uv for recipe environments." in msg['message'] for msg in mock_ui.messages_logged))

    @patch('llmb_install.ui.prompts.environment.is_uv_installed')
    def test_saved_conda_default_is_ignored(self, mock_is_uv_installed):
        """Stale saved conda defaults should not affect fresh recipe env creation."""
        mock_is_uv_installed.return_value = True

        mock_ui = MockUI()
        result = prompt_environment_type(mock_ui, default='conda')

        self.assertEqual(result, 'uv')
        self.assertFalse(mock_ui.was_prompted_for("environment"))
        self.assertTrue(
            any("Ignoring saved environment type 'conda'" in msg['message'] for msg in mock_ui.messages_logged)
        )

    @patch('llmb_install.ui.prompts.environment.is_uv_installed')
    def test_missing_uv_exits(self, mock_is_uv_installed):
        """Fresh installs fail when uv is unavailable."""
        mock_is_uv_installed.return_value = False

        mock_ui = MockUI()
        with self.assertRaises(SystemExit) as exc_info:
            prompt_environment_type(mock_ui, default='conda')

        self.assertEqual(exc_info.exception.code, 1)
        self.assertFalse(mock_ui.was_prompted_for("environment"))
        self.assertTrue(any("uv is required" in msg['message'] for msg in mock_ui.messages_logged))


if __name__ == '__main__':
    unittest.main()

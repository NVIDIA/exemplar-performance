# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MockUI implementation for testing prompt flows."""

from typing import Any, Dict, List, Optional, Union

from llmb_install.ui.interface import UIInterface


class MockUI(UIInterface):
    """Mock UI implementation for testing prompt flow logic."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None):
        """Initialize MockUI with predefined responses.

        Args:
            responses: Dictionary mapping prompt types to responses
        """
        self.responses = responses or {}
        self.call_log = []  # Track all UI calls for verification
        self.messages_logged = []  # Track all log messages

    def prompt_select(
        self, message: str, choices: List[Union[Dict[str, Any], str]], default: Optional[str] = None
    ) -> Optional[str]:
        """Mock select prompt."""
        self.call_log.append({'type': 'prompt_select', 'message': message, 'choices': choices, 'default': default})

        # Return predefined response or default
        key = self._extract_key_from_message(message)
        if key in self.responses:
            return self.responses[key]
        elif default:
            return default
        elif choices:
            # Handle both string choices and dict choices
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                return first_choice.get('value', first_choice.get('name', ''))
            else:
                # Simple string choice
                return str(first_choice)
        return None

    def prompt_text(self, message: str, default: str = "", validate: Optional[callable] = None) -> Optional[str]:
        """Mock text prompt."""
        self.call_log.append(
            {'type': 'prompt_text', 'message': message, 'default': default, 'validate': validate is not None}
        )

        # Return predefined response or default
        key = self._extract_key_from_message(message)
        if key in self.responses:
            response = self.responses[key]
            # Test validation if provided
            if validate and response:
                validation_result = validate(response)
                if validation_result is not True:
                    # For testing, we'll return the response anyway
                    # but log the validation failure
                    self.call_log[-1]['validation_failed'] = validation_result
            return response
        return default

    def prompt_confirm(self, message: str, default: bool = True) -> Optional[bool]:
        """Mock confirmation prompt."""
        self.call_log.append({'type': 'prompt_confirm', 'message': message, 'default': default})

        key = self._extract_key_from_message(message)
        if key in self.responses:
            return self.responses[key]
        return default

    def prompt_checkbox(
        self,
        message: str,
        choices: List[Union[Dict[str, Any], str]],
        validate: Optional[callable] = None,
        default: Optional[List[str]] = None,
    ) -> List[str]:
        """Mock checkbox prompt."""
        self.call_log.append(
            {
                'type': 'prompt_checkbox',
                'message': message,
                'choices': choices,
                'default': default,
                'validate': validate is not None,
            }
        )

        # Return predefined response or default (select all)
        key = self._extract_key_from_message(message)
        if key in self.responses:
            return self.responses[key]

        # If choices are dicts, return values
        if choices and isinstance(choices[0], dict):
            return [c.get('value', c.get('name', '')) for c in choices]

        return choices  # Default: select all choices

    def prompt_path(self, message: str, default: str = "", validate: Optional[callable] = None) -> Optional[str]:
        """Mock path prompt."""
        self.call_log.append(
            {'type': 'prompt_path', 'message': message, 'default': default, 'validate': validate is not None}
        )

        # Return predefined response or default
        key = self._extract_key_from_message(message)
        if key in self.responses:
            response = self.responses[key]
            # Test validation if provided
            if validate and response:
                validation_result = validate(response)
                if validation_result is not True:
                    # For testing, we'll return the response anyway
                    # but log the validation failure
                    self.call_log[-1]['validation_failed'] = validation_result
            return response
        return default

    def display_progress(self, message: str, current: int, total: int) -> None:
        """Mock progress display."""
        self.call_log.append({'type': 'display_progress', 'message': message, 'current': current, 'total': total})

    def log(self, message: str, level: str = 'info') -> None:
        """Mock logging."""
        self.messages_logged.append({'message': message, 'level': level})
        self.call_log.append({'type': 'log', 'message': message, 'level': level})

    def print_section(self, title: str, content: Optional[str] = None) -> None:
        """Mock section printing."""
        self.call_log.append({'type': 'print_section', 'title': title, 'content': content})

    def print_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Mock table printing."""
        self.call_log.append({'type': 'print_table', 'headers': headers, 'rows': rows})

    def _extract_key_from_message(self, message: str) -> str:
        """Extract a key from the message for response lookup."""
        # Simple key extraction - look for common keywords
        message_lower = message.lower()
        if 'account' in message_lower:
            return 'account'
        elif 'partition' in message_lower:
            if 'gpu' in message_lower:
                return 'gpu_partition'
            elif 'cpu' in message_lower:
                return 'cpu_partition'
            return 'partition'
        elif 'gpu' in message_lower and 'type' in message_lower:
            return 'gpu_type'
        elif 'architecture' in message_lower:
            return 'architecture'
        elif 'gres' in message_lower or 'gpus per node' in message_lower:
            return 'gpu_gres'
        elif 'environment' in message_lower and 'type' in message_lower:
            return 'environment'
        elif 'workload selection mode' in message_lower:
            return 'workload_selection_mode'
        elif 'select workloads to install' in message_lower:
            return 'select_workloads'
        return message.lower().replace(' ', '_').replace(':', '')

    # Helper methods for test verification
    def get_prompts_by_type(self, prompt_type: str) -> List[Dict[str, Any]]:
        """Get all prompts of a specific type."""
        return [call for call in self.call_log if call['type'] == prompt_type]

    def get_messages_by_level(self, level: str) -> List[str]:
        """Get all log messages of a specific level."""
        return [msg['message'] for msg in self.messages_logged if msg['level'] == level]

    def was_prompted_for(self, keyword: str) -> bool:
        """Check if user was prompted for something containing keyword."""
        for call in self.call_log:
            if call['type'] in ['prompt_text', 'prompt_select', 'prompt_confirm', 'prompt_checkbox']:
                if keyword.lower() in call['message'].lower():
                    return True
        return False

    def clear_log(self):
        """Clear call log and messages for fresh test."""
        self.call_log.clear()
        self.messages_logged.clear()


class TestMockUI:
    """Test the MockUI implementation itself."""

    def test_prompt_text_with_response(self):
        """Test text prompting with predefined response."""
        mock_ui = MockUI(responses={'account': 'test_account'})

        result = mock_ui.prompt_text("SLURM account:", default="")

        assert result == 'test_account'
        assert len(mock_ui.call_log) == 1
        assert mock_ui.call_log[0]['type'] == 'prompt_text'

    def test_prompt_select_with_choices(self):
        """Test select prompting with choices."""
        choices = [{'name': 'H100', 'value': 'h100'}, {'name': 'GB200', 'value': 'gb200'}]
        mock_ui = MockUI(responses={'gpu_type': 'gb200'})

        result = mock_ui.prompt_select("Select GPU type:", choices)

        assert result == 'gb200'

    def test_log_tracking(self):
        """Test log message tracking."""
        mock_ui = MockUI()

        mock_ui.log("Test message", "info")
        mock_ui.log("Warning message", "warning")

        info_messages = mock_ui.get_messages_by_level('info')
        warning_messages = mock_ui.get_messages_by_level('warning')

        assert info_messages == ['Test message']
        assert warning_messages == ['Warning message']

    def test_was_prompted_for(self):
        """Test helper method for checking prompts."""
        mock_ui = MockUI()

        mock_ui.prompt_text("SLURM account:", default="")
        mock_ui.prompt_select("Select GPU type:", [])

        assert mock_ui.was_prompted_for('account') is True
        assert mock_ui.was_prompted_for('gpu') is True
        assert mock_ui.was_prompted_for('unknown') is False

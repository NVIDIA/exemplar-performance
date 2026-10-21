# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UI abstraction layer for LLMB Install."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


class UIInterface(ABC):
    """Base class for UI implementations."""

    @abstractmethod
    def prompt_select(
        self, message: str, choices: List[Dict[str, Any]], default: Optional[str] = None
    ) -> Optional[str]:
        """Prompt user to select from choices.

        Args:
            message: Prompt message to display.
            choices: List of choice dictionaries with 'name' and 'value' keys.
            default: Default value to select.

        Returns:
            Selected value or None if cancelled.
        """
        pass

    @abstractmethod
    def prompt_text(
        self, message: str, default: str = "", validate: Optional[Callable] = None, password: bool = False
    ) -> Optional[str]:
        """Prompt user for text input.

        Args:
            message: Prompt message to display.
            default: Default value.
            validate: Optional validation function.
            password: If True, mask input (for secrets/API keys).

        Returns:
            User input or None if cancelled.
        """
        pass

    @abstractmethod
    def prompt_confirm(self, message: str, default: bool = True) -> Optional[bool]:
        """Prompt user for yes/no confirmation.

        Args:
            message: Confirmation message.
            default: Default choice.

        Returns:
            User choice or None if cancelled.
        """
        pass

    @abstractmethod
    def display_progress(self, message: str, current: int, total: int) -> None:
        """Display progress information.

        Args:
            message: Progress message.
            current: Current step number.
            total: Total number of steps.
        """
        pass

    @abstractmethod
    def log(self, message: str, level: str = 'info') -> None:
        """Log a message.

        Args:
            message: Message to log.
            level: Log level ('info', 'warning', 'error', 'success').
        """
        pass

    @abstractmethod
    def print_section(self, title: str, content: Optional[str] = None) -> None:
        """Print a section header with optional content.

        Args:
            title: Section title.
            content: Optional content to display under the title.
        """
        pass

    @abstractmethod
    def print_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Print a table.

        Args:
            headers: Table column headers.
            rows: Table rows data.
        """
        pass

    @abstractmethod
    def prompt_path(self, message: str, default: str = "", validate: Optional[Callable] = None) -> Optional[str]:
        """Prompt the user for a file/directory path.

        Args:
            message: The prompt message
            default: Default path value
            validate: Optional validation function that returns 'valid', 'invalid:message', etc.

        Returns:
            The selected path or None if cancelled
        """
        pass

    @abstractmethod
    def prompt_checkbox(
        self,
        message: str,
        choices: List[Dict[str, Any]],
        validate: Optional[Callable] = None,
        default: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Prompt the user to select multiple items from a list.

        Args:
            message: The prompt message
            choices: List of choice dictionaries with 'name' and 'value' keys
            validate: Optional validation function that returns 'valid', 'invalid:message', etc.
            default: List of values to pre-select

        Returns:
            List of selected values or None if cancelled
        """
        pass


class HeadlessUI(UIInterface):
    """Non-interactive UI implementation for headless mode."""

    def prompt_select(
        self, message: str, choices: List[Dict[str, Any]], default: Optional[str] = None
    ) -> Optional[str]:
        """Return default value without prompting."""
        if default:
            return default
        if choices:
            # Handle both string choices and dict choices
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                return first_choice.get('value', first_choice.get('name', ''))
            else:
                # Simple string choice
                return str(first_choice)
        return None

    def prompt_text(
        self, message: str, default: str = "", validate: Optional[Callable] = None, password: bool = False
    ) -> Optional[str]:
        """Return default value without prompting."""
        return default

    def prompt_confirm(self, message: str, default: bool = True) -> Optional[bool]:
        """Return default value without prompting."""
        return default

    def display_progress(self, message: str, current: int, total: int) -> None:
        """Log progress without visual indicators."""
        self.log(f"{message} ({current}/{total})")

    def log(self, message: str, level: str = 'info') -> None:
        """Print message without formatting."""
        prefix = {'info': 'INFO', 'warning': 'WARNING', 'error': 'ERROR', 'success': 'SUCCESS'}.get(level, 'INFO')
        print(f"[{prefix}] {message}")

    def print_section(self, title: str, content: Optional[str] = None) -> None:
        """Print section without formatting."""
        print(f"\n{title}")
        print("-" * len(title))
        if content:
            print(content)

    def print_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Print simple table format."""
        if headers:
            print(" | ".join(headers))
            print("-" * sum(len(h) + 3 for h in headers))
        for row in rows:
            print(" | ".join(str(cell) for cell in row))

    def prompt_path(self, message: str, default: str = "", validate: Optional[Callable] = None) -> Optional[str]:
        """Return default path without prompting."""
        return default

    def prompt_checkbox(
        self,
        message: str,
        choices: List[Dict[str, Any]],
        validate: Optional[Callable] = None,
        default: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Return default choices if provided, otherwise return first choice without prompting."""
        if default:
            return default
        if choices:
            # Handle both string choices and dict choices
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                return [first_choice.get('value', first_choice.get('name', ''))]
            else:
                # Simple string choice
                return [str(first_choice)]
        return []

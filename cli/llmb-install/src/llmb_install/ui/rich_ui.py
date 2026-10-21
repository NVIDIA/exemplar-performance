# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rich-based enhanced UI implementation for LLMB Install."""

from typing import Any, Dict, List, Optional

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from llmb_install.ui.interface import UIInterface


class RichUI(UIInterface):
    """Rich-based enhanced UI implementation with styling, progress bars, and better UX."""

    def __init__(self):
        """Initialize Rich UI with console and progress tracking."""
        self.console = Console()
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None

    def prompt_select(
        self, message: str, choices: List[Dict[str, Any]], default: Optional[str] = None
    ) -> Optional[str]:
        """Prompt user to select from choices using Rich-styled questionary."""
        if not choices:
            return None

        # Convert to questionary format if needed
        formatted_choices = []
        default_choice = None

        for choice in choices:
            if isinstance(choice, dict):
                formatted_choices.append(choice)
                if default and choice.get('value') == default:
                    default_choice = choice
            else:
                # Simple string choice
                formatted_choices.append({'name': str(choice), 'value': str(choice)})
                if default and str(choice) == default:
                    default_choice = formatted_choices[-1]

        # Use questionary with Rich styling via questionary's style parameter
        style = questionary.Style(
            [
                ('selected', 'bold fg:#00aa00'),  # Green for selected
                ('pointer', 'bold fg:#673ab7'),  # Purple pointer
                ('highlighted', 'bold fg:#00aa00'),
                ('answer', 'bold fg:#00aa00'),
                ('question', 'bold fg:#5f87d7'),  # Blue for questions
            ]
        )

        return questionary.select(message, choices=formatted_choices, default=default_choice, style=style).ask()

    def prompt_text(
        self, message: str, default: str = "", validate: Optional[callable] = None, password: bool = False
    ) -> Optional[str]:
        """Prompt user for text input using Rich Prompt with enhanced styling."""
        if password:
            # questionary does not render Rich markup, so pass the plain message; keep the validator.
            return questionary.password(message, validate=validate).ask()

        # For simple text input without validation, use Rich Prompt for better styling
        if not validate:
            return Prompt.ask(
                f"[bold cyan]{message}[/bold cyan]", default=default if default else None, console=self.console
            )

        # For complex validation, use questionary with Rich styling
        style = questionary.Style(
            [
                ('answer', 'bold fg:#00aa00'),
                ('question', 'bold fg:#5f87d7'),
            ]
        )

        return questionary.text(message, default=default, validate=validate, style=style).ask()

    def prompt_confirm(self, message: str, default: bool = True) -> Optional[bool]:
        """Prompt user for yes/no confirmation using Rich Confirm."""
        return Confirm.ask(f"[bold yellow]{message}[/bold yellow]", default=default, console=self.console)

    def display_progress(self, message: str, current: int, total: int) -> None:
        """Display Rich progress bar with enhanced styling."""
        if not self._progress:
            # Create progress bar if it doesn't exist
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                console=self.console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(message, total=total)

        # Update existing progress
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=current, description=message)

        # Stop progress if we're done
        if current >= total and self._progress:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def log(self, message: str, level: str = 'info') -> None:
        """Log message with Rich styling and appropriate colors."""
        style_map = {'info': 'blue', 'warning': 'yellow bold', 'error': 'red bold', 'success': 'green bold'}

        # Only use icons for important messages, not normal info logs
        icon_map = {'warning': '⚠️ ', 'error': '❌ ', 'success': '✅ '}

        style = style_map.get(level, 'blue')
        icon = icon_map.get(level, '')  # No icon for info messages

        self.console.print(f"{icon}{message}", style=style)

    def print_section(self, title: str, content: Optional[str] = None) -> None:
        """Print section with Rich styling."""
        # Print section header with simple styling
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
        self.console.print(f"[blue]{'-' * len(title)}[/blue]")

        # If there's content, display it in a subtle panel
        if content:
            panel = Panel(content, border_style="dim blue", padding=(0, 1), title=None)
            self.console.print(panel)
        else:
            # Just add a blank line for spacing
            self.console.print()

    def print_table(self, headers: List[str], rows: List[List[str]]) -> None:
        """Print table using Rich Table with enhanced formatting."""
        if not headers and not rows:
            return

        table = Table(
            title=None,
            show_header=bool(headers),
            header_style="bold cyan",
            border_style="blue",
            row_styles=["", "dim"],  # Alternate row styling
        )

        # Add columns
        if headers:
            for header in headers:
                table.add_column(header, style="white")
        else:
            # Determine number of columns from first row
            if rows:
                for i in range(len(rows[0])):
                    table.add_column(f"Column {i+1}", style="white")

        # Add rows
        for row in rows:
            # Convert all cells to strings and pad if necessary
            str_row = [str(cell) for cell in row]
            # Ensure row has same number of columns as headers
            while len(str_row) < len(headers) if headers else 0:
                str_row.append("")
            table.add_row(*str_row)

        self.console.print(table)

    def prompt_path(self, message: str, default: str = "", validate: Optional[callable] = None) -> Optional[str]:
        """Prompt user for a file/directory path with Rich styling and autocomplete."""
        # Rich doesn't have built-in path prompting, so we'll use questionary with styling
        style = questionary.Style(
            [
                ('answer', 'bold fg:#00aa00'),
                ('question', 'bold fg:#5f87d7'),
            ]
        )

        def questionary_validate(path: str) -> bool | str:
            if validate:
                result = validate(path)
                if result == "valid":
                    return True
                elif result.startswith("invalid:"):
                    return result.split(":", 1)[1]
                else:
                    return True
            return True

        return questionary.path(message, default=default, validate=questionary_validate, style=style).ask()

    def prompt_checkbox(
        self,
        message: str,
        choices: List[Dict[str, str]],
        validate: Optional[callable] = None,
        default: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Prompt user to select multiple items with Rich styling."""
        style = questionary.Style(
            [
                ('selected', 'bold fg:#00aa00'),  # Green for selected
                ('pointer', 'bold fg:#673ab7'),  # Purple pointer
                ('highlighted', 'bold fg:#00aa00'),
                ('answer', 'bold fg:#00aa00'),
                ('question', 'bold fg:#5f87d7'),  # Blue for questions
            ]
        )

        def questionary_validate(selected: List[str]) -> bool | str:
            if validate:
                result = validate(selected)
                if result == "valid":
                    return True
                elif result.startswith("invalid:"):
                    return result.split(":", 1)[1]
                else:
                    return True
            return True

        # Convert choices to questionary Choice objects with pre-selection
        questionary_choices = []
        for choice in choices:
            checked = default and choice['value'] in default
            questionary_choices.append(questionary.Choice(choice['name'], value=choice['value'], checked=checked))

        return questionary.checkbox(
            message, choices=questionary_choices, validate=questionary_validate, style=style
        ).ask()

    def show_spinner(self, message: str):
        """Context manager for showing a spinner during long operations."""
        return self.console.status(f"[bold green]{message}...[/bold green]")

    def create_progress_bar(self, description: str, total: int) -> Progress:
        """Create a new progress bar for long operations."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=self.console,
        )
        progress.start()
        return progress

    def print_success_banner(self, message: str) -> None:
        """Print a success banner with celebration styling."""
        success_panel = Panel(
            f"[bold green]{message}[/bold green]",
            title="[bold green]🎉 Success! 🎉[/bold green]",
            title_align="center",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(success_panel)

    def print_error_banner(self, message: str) -> None:
        """Print an error banner with warning styling."""
        error_panel = Panel(
            f"[bold red]{message}[/bold red]",
            title="[bold red]❌ Error ❌[/bold red]",
            title_align="center",
            border_style="red",
            padding=(1, 2),
        )
        self.console.print(error_panel)

    def print_warning_banner(self, message: str) -> None:
        """Print a warning banner with caution styling."""
        warning_panel = Panel(
            f"[bold yellow]{message}[/bold yellow]",
            title="[bold yellow]⚠️  Warning ⚠️[/bold yellow]",
            title_align="center",
            border_style="yellow",
            padding=(1, 2),
        )
        self.console.print(warning_panel)

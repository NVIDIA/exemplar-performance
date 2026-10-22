# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Centralized logging utilities for LLMB Install."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, console: bool = True, format_string: Optional[str] = None
) -> logging.Logger:
    """Set up centralized logging for LLMB Install.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to log file.
        console: Whether to log to console.
        format_string: Custom format string for log messages.

    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get the root logger for LLMB
    logger = logging.getLogger("llmb_install")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers only if propagation is disabled
    # This prevents interfering with parent loggers when propagation is enabled
    if not logger.propagate:
        logger.handlers.clear()

    formatter = logging.Formatter(format_string)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"llmb_install.{name}")


class LogCapture:
    """Context manager to capture log messages for testing."""

    def __init__(self, logger_name: str = "llmb_install", level: str = "DEBUG"):
        self.logger_name = logger_name
        self.level = level
        self.messages = []
        self.handler = None
        self.original_level = None

    def __enter__(self):
        self.logger = logging.getLogger(self.logger_name)
        self.original_level = self.logger.level

        # Create a custom handler that captures messages
        self.handler = logging.Handler()
        self.handler.emit = self._capture_record

        self.logger.addHandler(self.handler)
        self.logger.setLevel(getattr(logging, self.level.upper()))

        return self

    def _capture_record(self, record):
        """Capture a log record to avoid circular reference in lambda."""
        self.messages.append(record)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.original_level is not None:
            self.logger.setLevel(self.original_level)

    def get_messages(self, level: Optional[str] = None):
        """Get captured messages, optionally filtered by level."""
        if level:
            level_num = getattr(logging, level.upper())
            return [msg for msg in self.messages if msg.levelno >= level_num]
        return self.messages

    def has_message(self, text: str, level: Optional[str] = None) -> bool:
        """Check if any captured message contains the given text."""
        messages = self.get_messages(level)
        return any(text in str(msg.getMessage()) for msg in messages)

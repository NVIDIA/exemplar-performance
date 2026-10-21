# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-trial workspace and container-home preparation."""

from __future__ import annotations

import logging
import os
import shutil

from tasktrace_capture.apptainer.config import RuntimeLayout
from tasktrace_capture.runtimes._shared import (
    DEFAULT_CLAUDE_CODE_VERSION,
    ensure_claude_code_cached_on_host,
)


def prepare_workspace(layout: RuntimeLayout, logger: logging.Logger) -> None:
    layout.prepare()

    (layout.home / ".local" / "bin").mkdir(parents=True, exist_ok=True, mode=0o700)
    (layout.home / ".local" / "share").mkdir(parents=True, exist_ok=True)
    (layout.home / ".claude").mkdir(parents=True, exist_ok=True)
    (layout.home / ".bashrc").write_text('# container bashrc\nexport PATH="$HOME/.local/bin:$PATH"\n')
    (layout.home / ".local" / "bin" / "env").write_text('export PATH="$HOME/.local/bin:$PATH"\n')
    profile = '[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"\n'
    (layout.home / ".bash_profile").write_text(profile)
    (layout.home / ".profile").write_text(profile)

    if os.environ.get("_TASKTRACE_CAPTURE_AGENT") != "claude-code":
        return
    version = os.environ.get(
        "_TASKTRACE_CAPTURE_CLAUDE_CODE_VERSION",
        DEFAULT_CLAUDE_CODE_VERSION,
    )
    cached = ensure_claude_code_cached_on_host(version=version, logger=logger)
    if cached and cached.is_file():
        destination = layout.home / ".local" / "bin" / "claude"
        shutil.copy2(cached, destination)
        destination.chmod(0o755)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for llmb_install.cli."""

import argparse
import subprocess

import pytest

from llmb_install.cli import main
from llmb_install.constants import EXIT_ERROR


def test_main_keyboard_interrupt_exits_130(monkeypatch, capsys):
    """Ctrl-C should surface as exit code 130 for wrapper scripts."""

    class _InterruptingInstaller:
        def run(self, _args):
            raise KeyboardInterrupt

    monkeypatch.setattr("llmb_install.cli.parse_arguments", lambda: argparse.Namespace())
    # Patch the module attribute so the lazy ``from ... import Installer``
    # inside main() picks up the fake class regardless of import style.
    monkeypatch.setattr("llmb_install.core.installer.Installer", _InterruptingInstaller)
    # Also patch the cli module namespace in case the import is ever hoisted
    # to module level (``from llmb_install.core.installer import Installer``).
    monkeypatch.setattr("llmb_install.cli.Installer", _InterruptingInstaller, raising=False)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 130
    captured = capsys.readouterr()
    assert "Installation cancelled by user." in captured.out


def test_main_resume_failure_exits_error(monkeypatch, capsys):
    """Resume failures that escape Installer.run must return a nonzero status."""

    class _ResumeFailingInstaller:
        def run(self, _args):
            raise RuntimeError('git fetch failed while resuming')

    monkeypatch.setattr("llmb_install.cli.parse_arguments", lambda: argparse.Namespace(verbose=False))
    monkeypatch.setattr("llmb_install.core.installer.Installer", _ResumeFailingInstaller)
    monkeypatch.setattr("llmb_install.cli.Installer", _ResumeFailingInstaller, raising=False)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == EXIT_ERROR
    captured = capsys.readouterr()
    assert "Error: RuntimeError: git fetch failed while resuming" in captured.out


@pytest.mark.parametrize("verbose", [False, True])
def test_main_subprocess_failure_prints_captured_output(monkeypatch, capsys, verbose):
    """A failed subprocess's notes and captured output are each shown once with the final error."""

    class _SubprocessFailingInstaller:
        def run(self, _args):
            error = subprocess.CalledProcessError(
                2,
                ['uv', 'venv'],
                output='Using CPython search path\n',
                stderr='error: No interpreter found for Python 3.12\n',
            )
            error.add_note("Failed to create venv")
            raise error

    monkeypatch.setattr("llmb_install.cli.parse_arguments", lambda: argparse.Namespace(verbose=verbose))
    monkeypatch.setattr("llmb_install.core.installer.Installer", _SubprocessFailingInstaller)
    monkeypatch.setattr("llmb_install.cli.Installer", _SubprocessFailingInstaller, raising=False)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == EXIT_ERROR
    captured = capsys.readouterr()
    assert "Error: CalledProcessError: Command '['uv', 'venv']' returned non-zero exit status 2." in captured.out
    assert "error: No interpreter found for Python 3.12" in captured.out
    assert captured.out.count("No interpreter found") == 1
    assert captured.out.count("Using CPython search path") == 1
    assert (captured.out + captured.err).count("Failed to create venv") == 1

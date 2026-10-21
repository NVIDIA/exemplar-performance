# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for installer summary-file writing helpers."""

from llmb_install.utils.install_summary import write_install_summary_file


def _read_summary_lines(path):
    return path.read_text(encoding='utf-8').splitlines()


def test_write_install_summary_uses_env_path(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.txt"
    monkeypatch.setenv("LLMB_INSTALL_SUMMARY_FILE", str(summary_path))

    write_install_summary_file(
        status='success',
        install_path='/tmp/install-path',
        async_jobs_submitted=True,
    )

    lines = _read_summary_lines(summary_path)
    assert "version=1" in lines
    assert "status=success" in lines
    assert "install_path=/tmp/install-path" in lines
    assert "async_jobs_submitted=true" in lines
    assert not (tmp_path / "summary.txt.tmp").exists()


def test_write_install_summary_sanitizes_failed_workloads(tmp_path):
    summary_path = tmp_path / "summary.txt"

    write_install_summary_file(
        status='failed',
        summary_path=str(summary_path),
        install_path='/tmp/install\npath',
        failed_workloads=['workload-a', 'workload-b\nbad'],
    )

    lines = _read_summary_lines(summary_path)
    assert "status=failed" in lines
    assert "install_path=/tmp/install path" in lines
    assert "failed_workloads=workload-a,workload-b bad" in lines
    assert not (tmp_path / "summary.txt.tmp").exists()


def test_write_install_summary_skips_invalid_status(tmp_path):
    summary_path = tmp_path / "summary.txt"

    write_install_summary_file(
        status='unknown',
        summary_path=str(summary_path),
    )

    assert not summary_path.exists()


def test_write_install_summary_skips_when_parent_not_writable(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.txt"
    monkeypatch.setattr("llmb_install.utils.install_summary.os.access", lambda *_: False)

    write_install_summary_file(
        status='success',
        summary_path=str(summary_path),
    )

    assert not summary_path.exists()

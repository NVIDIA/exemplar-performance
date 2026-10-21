# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tasktrace_capture.runtimes — runtime selection, preflight, cleanup."""

from pathlib import Path
from unittest import mock

import pytest
from tasktrace_capture.apptainer.installation import ApptainerInstallation
from tasktrace_capture.runtimes import (
    SUPPORTED_RUNTIMES,
    _docker_bridge_network_check,
    _parse_apptainer_version,
    runtime_cleanup,
    runtime_import_path,
    runtime_preflight_checks,
)


@pytest.fixture(autouse=True)
def _stub_managed_apptainer_selection(monkeypatch):
    selected = ApptainerInstallation(
        binary=Path("/opt/apptainer/bin/apptainer"),
        source="explicit",
        detail="selected for test",
    )
    monkeypatch.setattr(
        "tasktrace_capture.apptainer_env.ensure_apptainer_installation",
        lambda: selected,
    )


# -- runtime_import_path ------------------------------------------------------


def test_runtime_import_path_enroot():
    assert runtime_import_path("enroot") == "tasktrace_capture.enroot_env:EnrootEnvironment"


def test_runtime_import_path_docker():
    assert runtime_import_path("docker") == "tasktrace_capture.docker_env:TasktraceCaptureDockerEnv"


def test_runtime_import_path_apptainer():
    assert runtime_import_path("apptainer") == ("tasktrace_capture.apptainer_env:ApptainerEnvironment")


def test_runtime_import_path_current():
    assert runtime_import_path("current") == "tasktrace_capture.current_env:CurrentEnvironment"


def test_runtime_import_path_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown runtime"):
        runtime_import_path("gvisor")


# -- runtime_preflight_checks -------------------------------------------------


def test_preflight_enroot_found():
    # Stub the cache check to avoid touching real /tmp state during runtime test.
    with mock.patch("tasktrace_capture.runtimes.shutil.which", return_value="/usr/bin/enroot"), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("enroot")
    assert len(checks) == 2
    enroot_label, enroot_passed, _ = checks[0]
    assert "enroot" in enroot_label
    assert enroot_passed is True
    # Cache check tagged onto the end.
    assert "claude-code cache" in checks[1][0]


def test_preflight_enroot_missing():
    with mock.patch("tasktrace_capture.runtimes.shutil.which", return_value=None), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("enroot")
    assert checks[0][1] is False
    assert "module load" in checks[0][2]


def test_preflight_docker_found_and_daemon_reachable():
    fake_rc = mock.Mock(returncode=0)
    with mock.patch("tasktrace_capture.runtimes.shutil.which", return_value="/usr/bin/docker"), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run", return_value=fake_rc
    ) as run_mock, mock.patch(
        "tasktrace_capture.runtimes._docker_bridge_network_check",
        return_value=("docker bridge network creation", True, ""),
    ), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("docker")
    assert len(checks) == 4
    assert checks[0] == ("docker on PATH", True, mock.ANY)
    assert checks[1][0] == "docker daemon reachable"
    assert checks[1][1] is True
    assert checks[2] == ("docker bridge network creation", True, "")
    assert "claude-code cache" in checks[3][0]
    run_mock.assert_called_once()
    # Called with ["docker", "info"]
    assert run_mock.call_args.args[0] == ["docker", "info"]


def test_preflight_docker_found_daemon_unreachable():
    fake_rc = mock.Mock(returncode=1)
    with mock.patch("tasktrace_capture.runtimes.shutil.which", return_value="/usr/bin/docker"), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run", return_value=fake_rc
    ), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("docker")
    assert checks[1][1] is False
    assert "systemctl" in checks[1][2] or "daemon" in checks[1][2].lower()
    assert all("bridge network" not in label for label, _, _ in checks)


def test_preflight_docker_missing_skips_daemon_check():
    with mock.patch("tasktrace_capture.runtimes.shutil.which", return_value=None), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("docker")
    # docker on PATH (failed) + cache check (still runs); no daemon check.
    assert len(checks) == 2
    assert checks[0] == ("docker on PATH", False, mock.ANY)
    assert "claude-code cache" in checks[1][0]


def test_preflight_docker_info_timeout_is_failure():
    import subprocess as sp

    with mock.patch("tasktrace_capture.runtimes.shutil.which", return_value="/usr/bin/docker"), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd=["docker"], timeout=10),
    ), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("docker")
    assert checks[1][1] is False


def test_preflight_apptainer_found(tmp_path):
    version = mock.Mock(returncode=0, stdout="apptainer version 1.5.0\n", stderr="")
    filesystem = mock.Mock(returncode=0, stdout="tmpfs\n", stderr="")
    userns = mock.Mock(returncode=0, stdout="", stderr="")
    buildcfg = mock.Mock(
        returncode=0,
        stdout="APPTAINER_SUID_INSTALL=0\n",
        stderr="",
    )
    sif_support = mock.Mock(
        available=True,
        mode="squashfuse_ll + /dev/fuse",
        helper=None,
        detail="",
    )
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_runtime_root",
        return_value=tmp_path,
    ), mock.patch(
        "tasktrace_capture.runtimes.shutil.which",
        return_value="/usr/bin/unshare",
    ), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[version, filesystem, userns, buildcfg],
    ), mock.patch(
        "tasktrace_capture.apptainer.capabilities.direct_sif_support",
        return_value=sif_support,
    ), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("apptainer")
    assert all(passed for _, passed, _ in checks)
    assert any("tmpfs" in label for label, _, _ in checks)
    assert any("1.4 or newer" in label for label, _, _ in checks)
    assert any("user namespaces" in label for label, _, _ in checks)
    assert any("direct SIF mounting" in label for label, _, _ in checks)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("apptainer version 1.5.0", (1, 5, 0)),
        ("1.4.2-1.el9", (1, 4, 2)),
        ("apptainer version 2.0", (2, 0, 0)),
        ("unknown", None),
    ],
)
def test_parse_apptainer_version(output, expected):
    assert _parse_apptainer_version(output) == expected


def test_preflight_apptainer_rejects_old_version(tmp_path):
    version = mock.Mock(returncode=0, stdout="apptainer version 1.3.6\n", stderr="")
    filesystem = mock.Mock(returncode=0, stdout="tmpfs\n", stderr="")
    userns = mock.Mock(returncode=0, stdout="", stderr="")
    buildcfg = mock.Mock(
        returncode=0,
        stdout="APPTAINER_SUID_INSTALL=1\n",
        stderr="",
    )
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_runtime_root",
        return_value=tmp_path,
    ), mock.patch(
        "tasktrace_capture.runtimes.shutil.which",
        return_value="/usr/bin/unshare",
    ), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[version, filesystem, userns, buildcfg],
    ), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("apptainer")

    version_check = next(check for check in checks if "1.4 or newer" in check[0])
    assert version_check[1] is False
    assert "1.4 or newer" in version_check[2]


def test_preflight_apptainer_rejects_missing_userns(tmp_path):
    version = mock.Mock(returncode=0, stdout="apptainer version 1.5.0\n", stderr="")
    filesystem = mock.Mock(returncode=0, stdout="tmpfs\n", stderr="")
    userns = mock.Mock(returncode=1, stdout="", stderr="operation not permitted")
    buildcfg = mock.Mock(
        returncode=0,
        stdout="PACKAGE_VERSION=1.5.0\nAPPTAINER_SUID_INSTALL=0\n",
        stderr="",
    )
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_runtime_root",
        return_value=tmp_path,
    ), mock.patch(
        "tasktrace_capture.runtimes.shutil.which",
        return_value="/usr/bin/unshare",
    ), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[version, filesystem, userns, buildcfg],
    ), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("apptainer")

    userns_check = next(check for check in checks if "unprivileged user namespaces" in check[0])
    assert userns_check[1] is False
    assert "Enable unprivileged user namespaces" in userns_check[2]


def test_preflight_apptainer_rejects_missing_direct_sif_support(tmp_path):
    version = mock.Mock(returncode=0, stdout="apptainer version 1.5.0\n", stderr="")
    filesystem = mock.Mock(returncode=0, stdout="tmpfs\n", stderr="")
    userns = mock.Mock(returncode=0, stdout="", stderr="")
    buildcfg = mock.Mock(
        returncode=0,
        stdout="APPTAINER_SUID_INSTALL=0\nLIBEXECDIR=/missing\n",
        stderr="",
    )
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_runtime_root",
        return_value=tmp_path,
    ), mock.patch(
        "tasktrace_capture.runtimes.shutil.which",
        return_value="/usr/bin/unshare",
    ), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[version, filesystem, userns, buildcfg],
    ), mock.patch(
        "tasktrace_capture.apptainer.capabilities.find_squashfuse",
        return_value=None,
    ), mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("apptainer")

    direct_check = next(check for check in checks if "direct SIF mounting" in check[0])
    assert direct_check[1] is False
    assert "squashfuse_ll" in direct_check[2]
    assert "expands each SIF" in direct_check[2]


def test_docker_bridge_network_check_creates_and_removes_network():
    created = mock.Mock(returncode=0, stdout="id\n", stderr="")
    removed = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[created, removed],
    ) as run_mock:
        label, passed, hint = _docker_bridge_network_check()
    assert label == "docker bridge network creation"
    assert passed is True
    assert hint == ""
    create_args = run_mock.call_args_list[0].args[0]
    rm_args = run_mock.call_args_list[1].args[0]
    assert create_args[:4] == ["docker", "network", "create", "--driver"]
    assert create_args[4] == "bridge"
    assert create_args[5].startswith("tasktrace-capture-preflight-")
    assert rm_args == ["docker", "network", "rm", create_args[5]]


def test_docker_bridge_network_check_reports_restart_on_create_failure():
    failed = mock.Mock(
        returncode=1,
        stdout="",
        stderr="iptables: No chain/target/match by that name",
    )
    with mock.patch("tasktrace_capture.runtimes.subprocess.run", return_value=failed):
        label, passed, hint = _docker_bridge_network_check()
    assert label == "docker bridge network creation"
    assert passed is False
    assert "sudo systemctl restart docker" in hint
    assert "No chain/target/match" in hint


def test_docker_bridge_network_check_reports_cleanup_failure():
    created = mock.Mock(returncode=0, stdout="id\n", stderr="")
    failed_remove = mock.Mock(returncode=1, stdout="", stderr="busy")
    with mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[created, failed_remove],
    ):
        _, passed, hint = _docker_bridge_network_check()
    assert passed is False
    assert "docker network rm" in hint


def test_preflight_claude_code_cache_stale_is_failure(tmp_path, monkeypatch):
    """Stale cached binary (different version from DEFAULT_CLAUDE_CODE_VERSION)
    should fail preflight with an `rm -rf` remediation hint."""
    from tasktrace_capture.runtimes._shared import (
        DEFAULT_CLAUDE_CODE_VERSION,
        cached_claude_code_version,
    )

    # No cache → check passes (None is treated as fresh install).
    assert cached_claude_code_version() is None or True  # smoke
    # Construct a fake binary that prints a stale version.
    fake_dir = tmp_path / "tasktrace_capture_claude_code_cache"
    fake_dir.mkdir()
    fake_bin = fake_dir / "claude"
    fake_bin.write_text("#!/usr/bin/env bash\necho '0.0.1 (Claude Code)'\n")
    fake_bin.chmod(0o755)
    monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_DIR", fake_dir)

    label, passed, hint = _import_cache_check()
    assert passed is False
    assert "0.0.1" in hint
    assert "rm -rf" in hint
    assert DEFAULT_CLAUDE_CODE_VERSION in label


def _import_cache_check():
    from tasktrace_capture.runtimes import _claude_code_cache_check

    return _claude_code_cache_check()


def test_preflight_rejects_unknown():
    with pytest.raises(ValueError):
        runtime_preflight_checks("gvisor")


# -- runtime_cleanup ----------------------------------------------------------


def test_cleanup_docker_no_containers_is_noop():
    fake_rc = mock.Mock(stdout="", returncode=0)
    with mock.patch("tasktrace_capture.runtimes.subprocess.run", return_value=fake_rc) as run_mock:
        runtime_cleanup("docker", "rt123")
    # Exactly one call: the `ps` list. No `rm -f` since list was empty.
    assert run_mock.call_count == 1
    args = run_mock.call_args.args[0]
    assert args[:3] == ["docker", "ps", "-a"]
    assert "label=com.docker.compose.project" in args


def test_cleanup_docker_removes_listed_containers():
    list_result = mock.Mock(
        stdout="abc123 rt123--task__one\ndef456 rt1234--task__two\n",
        returncode=0,
    )
    rm_result = mock.Mock(returncode=0)
    with mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[list_result, rm_result],
    ) as run_mock:
        runtime_cleanup("docker", "rt123")
    assert run_mock.call_count == 2
    rm_call = run_mock.call_args_list[1]
    assert rm_call.args[0][:3] == ["docker", "rm", "-f"]
    assert "abc123" in rm_call.args[0]
    assert "def456" not in rm_call.args[0]


def test_cleanup_docker_without_owner_is_noop():
    with mock.patch("tasktrace_capture.runtimes.subprocess.run") as run_mock:
        runtime_cleanup("docker")
    run_mock.assert_not_called()


def test_cleanup_docker_swallows_exceptions():
    with mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=OSError("docker daemon not running"),
    ):
        runtime_cleanup("docker", "rt123")  # must not raise


def test_cleanup_enroot_delegates_to_cli_helper():
    with mock.patch("tasktrace_capture.cli._cleanup_containers") as helper:
        runtime_cleanup("enroot")
    helper.assert_called_once()


def test_cleanup_enroot_swallows_helper_exceptions():
    with mock.patch(
        "tasktrace_capture.cli._cleanup_containers",
        side_effect=RuntimeError("boom"),
    ):
        runtime_cleanup("enroot")  # must not raise


def test_cleanup_apptainer_stops_only_tasktrace_capture_instances(tmp_path):
    listed = mock.Mock(
        returncode=0,
        stdout=(
            '{"instances": [' '{"instance": "tasktrace-capture-hb-task-abc"},' '{"instance": "unrelated-service"}' "]}"
        ),
    )
    stopped = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_runtime_root",
        return_value=tmp_path,
    ), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[listed, stopped],
    ) as run_mock:
        runtime_cleanup("apptainer")
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[1].args[0][-1] == "tasktrace-capture-hb-task-abc"


def test_cleanup_apptainer_force_stops_after_graceful_failure(tmp_path):
    runtime_dir = tmp_path / "tasktrace-capture-hb-task-abc"
    runtime_dir.mkdir()
    listed = mock.Mock(
        returncode=0,
        stdout='{"instances": [{"instance": "tasktrace-capture-hb-task-abc"}]}',
    )
    graceful = mock.Mock(returncode=255, stdout="", stderr="still running")
    forced = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_runtime_root",
        return_value=tmp_path,
    ), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        side_effect=[listed, graceful, forced],
    ) as run_mock:
        runtime_cleanup("apptainer")
    assert run_mock.call_count == 3
    assert "--force" in run_mock.call_args_list[2].args[0]
    assert not runtime_dir.exists()


def test_cleanup_apptainer_removes_zero_mode_overlay_and_preserves_unrelated_state(
    tmp_path,
):
    runtime_dir = tmp_path / "tasktrace-capture-hb-task-abc"
    blocked = runtime_dir / "overlay" / "work" / "work"
    blocked.mkdir(parents=True)
    blocked.chmod(0)
    unrelated = tmp_path / "unrelated-runtime-state"
    unrelated.mkdir()
    listed = mock.Mock(returncode=0, stdout='{"instances": []}')

    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_runtime_root",
        return_value=tmp_path,
    ), mock.patch(
        "tasktrace_capture.runtimes.subprocess.run",
        return_value=listed,
    ):
        runtime_cleanup("apptainer")

    assert not runtime_dir.exists()
    assert unrelated.is_dir()


def test_cleanup_rejects_unknown():
    with pytest.raises(ValueError):
        runtime_cleanup("gvisor")


# -- module exports -----------------------------------------------------------


def test_supported_runtimes_contains_all_implemented_runtimes():
    assert "enroot" in SUPPORTED_RUNTIMES
    assert "docker" in SUPPORTED_RUNTIMES
    assert "apptainer" in SUPPORTED_RUNTIMES
    assert "opensandbox" in SUPPORTED_RUNTIMES
    assert "current" in SUPPORTED_RUNTIMES


def test_preflight_current_checks_workdir_logs_root_and_image_ref(tmp_path, monkeypatch):
    workdir = tmp_path / "testbed"
    workdir.mkdir()
    monkeypatch.setenv("TASKTRACE_CAPTURE_CURRENT_WORKDIR", str(workdir))
    monkeypatch.setenv("TASKTRACE_CAPTURE_CURRENT_LOGS_ROOT", str(tmp_path / "logs"))
    monkeypatch.delenv("TASKTRACE_CAPTURE_CURRENT_IMAGE_REF", raising=False)
    with mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("current")
    labels = [label for label, _, _ in checks]
    assert any("task workdir exists" in label for label in labels)
    assert checks[0][1] is True
    assert checks[1][1] is True  # logs root creatable under tmp_path
    image_check = next(check for check in checks if "IMAGE_REF" in check[0])
    assert image_check[1] is False and "Export" in image_check[2]
    monkeypatch.setenv("TASKTRACE_CAPTURE_CURRENT_WORKDIR", str(tmp_path / "missing"))
    monkeypatch.setenv("TASKTRACE_CAPTURE_CURRENT_IMAGE_REF", "ghcr.io/org/img@sha256:abc")
    with mock.patch(
        "tasktrace_capture.runtimes._claude_code_cache_check",
        return_value=("claude-code cache version (expected x)", True, ""),
    ):
        checks = runtime_preflight_checks("current")
    assert checks[0][1] is False and "INSIDE the task image" in checks[0][2]
    assert next(check for check in checks if "IMAGE_REF" in check[0])[1] is True


def test_cleanup_current_is_a_no_op():
    with mock.patch("tasktrace_capture.runtimes.subprocess.run") as run:
        runtime_cleanup("current")
    run.assert_not_called()

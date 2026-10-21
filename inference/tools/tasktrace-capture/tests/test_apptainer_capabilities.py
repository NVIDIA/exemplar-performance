# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest import mock

from tasktrace_capture.apptainer.capabilities import (
    buildcfg_value,
    direct_sif_support,
    find_squashfuse,
    fuse_device_access,
    probe_squashfuse,
    subordinate_id_mapping_available,
)


def test_buildcfg_value_reads_exact_key():
    output = "PREFIX=/opt/apptainer\nLIBEXECDIR=/opt/apptainer/libexec\n"

    assert buildcfg_value(output, "LIBEXECDIR") == "/opt/apptainer/libexec"
    assert buildcfg_value(output, "MISSING") is None


def test_subordinate_mapping_requires_both_files_and_helpers(
    tmp_path: Path,
):
    subuid = tmp_path / "subuid"
    subgid = tmp_path / "subgid"
    subuid.write_text("test-user:100000:65536\n")
    subgid.write_text("test-user:100000:65536\n")

    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.shutil.which",
        side_effect=lambda name, path=None: f"/usr/bin/{name}",
    ):
        assert subordinate_id_mapping_available(
            subuid_path=subuid,
            subgid_path=subgid,
            uid=1234,
            username="test-user",
        )

    subgid.write_text("")
    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.shutil.which",
        side_effect=lambda name, path=None: f"/usr/bin/{name}",
    ):
        assert not subordinate_id_mapping_available(
            subuid_path=subuid,
            subgid_path=subgid,
            uid=1234,
            username="test-user",
        )


def test_find_squashfuse_prefers_apptainer_libexec(tmp_path: Path):
    helper = tmp_path / "libexec" / "apptainer" / "bin" / "squashfuse_ll"
    helper.parent.mkdir(parents=True)
    helper.write_text("#!/bin/sh\n")
    helper.chmod(0o755)

    found = find_squashfuse(
        f"LIBEXECDIR={tmp_path / 'libexec'}\n",
        environment={"PATH": ""},
    )

    assert found == helper


def test_direct_sif_support_does_not_assume_setuid_kernel_mounting():
    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.find_squashfuse",
        return_value=None,
    ):
        support = direct_sif_support("APPTAINER_SUID_INSTALL=1\n")

    assert support.available is False
    assert "neither squashfuse_ll nor squashfuse" in support.detail


def test_direct_sif_support_rejects_rootless_install_without_helper():
    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.find_squashfuse",
        return_value=None,
    ):
        support = direct_sif_support("APPTAINER_SUID_INSTALL=0\n")

    assert support.available is False
    assert "neither squashfuse_ll nor squashfuse" in support.detail


def test_direct_sif_support_requires_accessible_fuse_device():
    helper = Path("/opt/apptainer/libexec/apptainer/bin/squashfuse_ll")
    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.find_squashfuse",
        return_value=helper,
    ), mock.patch(
        "tasktrace_capture.apptainer.capabilities.fuse_device_access",
        return_value=(False, "/dev/fuse: Permission denied"),
    ):
        support = direct_sif_support("APPTAINER_SUID_INSTALL=0\n")

    assert support.available is False
    assert "Permission denied" in support.detail
    assert support.helper == helper


def test_direct_sif_support_accepts_helper_and_fuse():
    helper = Path("/opt/apptainer/libexec/apptainer/bin/squashfuse_ll")
    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.find_squashfuse",
        return_value=helper,
    ), mock.patch(
        "tasktrace_capture.apptainer.capabilities.fuse_device_access",
        return_value=(True, ""),
    ):
        support = direct_sif_support("APPTAINER_SUID_INSTALL=0\n")

    assert support.available is True
    assert support.helper == helper
    assert "squashfuse_ll" in support.mode


def test_fuse_device_access_reports_open_failure():
    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.os.open",
        side_effect=PermissionError(13, "Permission denied"),
    ):
        available, detail = fuse_device_access()

    assert available is False
    assert "Permission denied" in detail


def test_probe_squashfuse_accepts_libfuse_help_exit_status():
    result = mock.Mock(
        returncode=2,
        stdout="squashfuse 0.6.1\nUsage: squashfuse_ll",
        stderr="",
    )
    with mock.patch(
        "tasktrace_capture.apptainer.capabilities.subprocess.run",
        return_value=result,
    ):
        available, detail = probe_squashfuse(Path("/opt/squashfuse_ll"))

    assert available is True
    assert detail == ""

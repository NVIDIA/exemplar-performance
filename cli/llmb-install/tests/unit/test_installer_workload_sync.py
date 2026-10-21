# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: after _handle_repository_setup, workload paths track llmb_repo not the git clone.

Bug: self.workloads was built from the git clone in run() and never rebuilt after
_handle_repository_setup copied the repo to $LLMB_INSTALL/llmb_repo. Setup tasks
then ran from the git clone's workload dirs instead of the installed copy.
"""

from unittest.mock import patch

from llmb_install.core.installer import Installer
from llmb_install.core.workload import build_workload_dict

_METADATA = """\
general:
  workload: test_workload
  workload_type: pretrain
  framework: nemo
"""


def test_workload_paths_track_llmb_repo_after_repository_setup(tmp_path):
    """After _handle_repository_setup, every workload path must be under llmb_repo, not the git clone."""
    git_clone = tmp_path / "git_clone"
    (git_clone / "test_workload").mkdir(parents=True)
    (git_clone / "test_workload" / "metadata.yaml").write_text(_METADATA)

    install_path = tmp_path / "install"
    llmb_repo = install_path / "llmb_repo"
    (llmb_repo / "test_workload").mkdir(parents=True)
    (llmb_repo / "test_workload" / "metadata.yaml").write_text(_METADATA)

    installer = Installer()
    installer.root_dir = str(git_clone)
    installer.workloads = build_workload_dict(str(git_clone))
    assert len(installer.workloads) == 1

    # Confirm initial state: paths point at the git clone
    for workload_data in installer.workloads.values():
        assert workload_data["path"].startswith(str(git_clone))

    with (
        patch("llmb_install.core.installer.check_repository_state", return_value="empty"),
        patch("llmb_install.core.installer.copy_repository_working_files"),
    ):
        returned_path = installer._handle_repository_setup(str(install_path), dev_mode=False)

    assert returned_path == str(llmb_repo)
    for workload_data in installer.workloads.values():
        assert workload_data["path"].startswith(
            str(llmb_repo)
        ), f"workload path {workload_data['path']!r} should be under llmb_repo, not the git clone"

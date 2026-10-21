#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dependency resolution functionality in the installer."""

import subprocess
from unittest.mock import patch

import pytest

from llmb_install.core.dependency import (
    _resolve_dependencies,
    clone_git_repos,
    group_workloads_by_dependencies,
    install_dependencies,
)


class TestDependencyResolution:
    """Test dependency resolution functionality."""

    def test_resolve_dependencies_no_dependencies(self):
        """Test resolving workload with no dependencies."""
        workload_data = {"setup": {}}
        result = _resolve_dependencies(workload_data)
        assert result is None

    def test_resolve_dependencies_empty_dependencies(self):
        """Test resolving workload with empty dependencies section."""
        workload_data = {"setup": {"dependencies": {}}}
        result = _resolve_dependencies(workload_data)
        assert result is None

    def test_resolve_dependencies_empty_pip_section(self):
        """Test resolving workload with empty pip section (should return the dict)."""
        workload_data = {"setup": {"dependencies": {"pip": []}}}
        result = _resolve_dependencies(workload_data)
        assert result == {"pip": []}

    def test_resolve_dependencies_simple_pip(self):
        """Test resolving simple pip dependencies."""
        workload_data = {"setup": {"dependencies": {"pip": ["numpy==1.24.0", "torch>=2.0"]}}}
        result = _resolve_dependencies(workload_data)
        expected = {"pip": ["numpy==1.24.0", "torch>=2.0"]}
        assert result == expected

    def test_resolve_dependencies_git_with_repo_key(self):
        """Test resolving git dependencies with repo_key references."""
        workload_data = {
            "setup": {
                "dependencies": {
                    "git": {
                        "nemo": {"repo_key": "nemo_repo", "install_method": {"type": "script", "path": "install.sh"}}
                    }
                }
            },
            "repositories": {"nemo_repo": {"url": "https://github.com/NVIDIA/NeMo.git", "commit": "abc123"}},
        }
        result = _resolve_dependencies(workload_data)
        expected = {
            "git": {
                "nemo": {
                    "url": "https://github.com/NVIDIA/NeMo.git",
                    "commit": "abc123",
                    "install_method": {"type": "script", "path": "install.sh"},
                }
            }
        }
        assert result == expected

    def test_resolve_dependencies_pip_with_repo_key(self):
        """Test resolving pip dependencies with repo_key references."""
        workload_data = {
            "setup": {"dependencies": {"pip": ["numpy==1.24.0", {"package": "custom_lib", "repo_key": "custom_repo"}]}},
            "repositories": {"custom_repo": {"url": "https://github.com/example/custom_lib.git", "commit": "def456"}},
        }
        result = _resolve_dependencies(workload_data)
        expected = {
            "pip": [
                "numpy==1.24.0",
                {"package": "custom_lib", "url": "https://github.com/example/custom_lib.git", "commit": "def456"},
            ]
        }
        assert result == expected

    def test_resolve_dependencies_editable_pip_adds_to_git(self):
        """Test that editable pip dependencies are added to git section."""
        workload_data = {
            "setup": {
                "dependencies": {
                    "pip": [
                        "numpy==1.24.0",
                        {
                            "package": "my_package",
                            "url": "https://github.com/example/my_package.git",
                            "commit": "abc123",
                            "editable": True,
                        },
                    ]
                }
            }
        }
        result = _resolve_dependencies(workload_data)
        expected = {
            "pip": [
                "numpy==1.24.0",
                {
                    "package": "my_package",
                    "url": "https://github.com/example/my_package.git",
                    "commit": "abc123",
                    "editable": True,
                },
            ],
            "git": {
                "my_package": {
                    "url": "https://github.com/example/my_package.git",
                    "commit": "abc123",
                    "install_method": {"type": "pip"},
                }
            },
        }
        assert result == expected

    def test_resolve_dependencies_install_target_adds_to_git(self):
        """Test that pip dependencies with install_target are added to git section."""
        workload_data = {
            "setup": {
                "dependencies": {
                    "pip": [
                        {
                            "package": "transformer_engine",
                            "url": "https://github.com/NVIDIA/TransformerEngine.git",
                            "commit": "v1.0",
                            "install_target": ".[pytorch]",
                        }
                    ]
                }
            }
        }
        result = _resolve_dependencies(workload_data)
        expected = {
            "pip": [
                {
                    "package": "transformer_engine",
                    "url": "https://github.com/NVIDIA/TransformerEngine.git",
                    "commit": "v1.0",
                    "install_target": ".[pytorch]",
                }
            ],
            "git": {
                "transformer_engine": {
                    "url": "https://github.com/NVIDIA/TransformerEngine.git",
                    "commit": "v1.0",
                    "install_method": {"type": "pip"},
                }
            },
        }
        assert result == expected

    def test_resolve_dependencies_combined_editable_and_install_target(self):
        """Test workload with both editable and install_target pip dependencies."""
        workload_data = {
            "setup": {
                "dependencies": {
                    "git": {
                        "existing_repo": {
                            "url": "https://github.com/example/existing.git",
                            "commit": "existing123",
                            "install_method": {"type": "script", "path": "setup.sh"},
                        }
                    },
                    "pip": [
                        "numpy==1.24.0",
                        {
                            "package": "editable_pkg",
                            "url": "https://github.com/example/editable.git",
                            "commit": "edit123",
                            "editable": True,
                        },
                        {
                            "package": "target_pkg",
                            "url": "https://github.com/example/target.git",
                            "commit": "target456",
                            "install_target": ".[all]",
                        },
                    ],
                }
            }
        }
        result = _resolve_dependencies(workload_data)

        # Check that both editable and install_target packages were added to git
        assert "git" in result
        assert "editable_pkg" in result["git"]
        assert "target_pkg" in result["git"]
        assert "existing_repo" in result["git"]

        # Verify install_method is set correctly
        assert result["git"]["editable_pkg"]["install_method"] == {"type": "pip"}
        assert result["git"]["target_pkg"]["install_method"] == {"type": "pip"}
        assert result["git"]["existing_repo"]["install_method"] == {"type": "script", "path": "setup.sh"}


class TestWorkloadGrouping:
    """Test workload grouping by dependencies."""

    def test_group_workloads_no_dependencies(self):
        """Test grouping workloads with no dependencies."""
        workloads = {"workload1": {"setup": {}}, "workload2": {"setup": {}}}
        result = group_workloads_by_dependencies(workloads, ["workload1", "workload2"])
        assert None in result
        assert set(result[None]) == {"workload1", "workload2"}

    def test_group_workloads_identical_dependencies(self):
        """Test grouping workloads with identical dependencies."""
        dependencies = {"pip": ["numpy==1.24.0", "torch>=2.0"]}
        workloads = {
            "workload1": {"setup": {"dependencies": dependencies}},
            "workload2": {"setup": {"dependencies": dependencies}},
        }
        result = group_workloads_by_dependencies(workloads, ["workload1", "workload2"])

        # Should have one group with both workloads
        assert len(result) == 1
        group_key = list(result.keys())[0]
        assert group_key is not None
        assert set(result[group_key]) == {"workload1", "workload2"}

    def test_group_workloads_different_dependencies(self):
        """Test grouping workloads with different dependencies."""
        workloads = {
            "workload1": {"setup": {"dependencies": {"pip": ["numpy==1.24.0"]}}},
            "workload2": {"setup": {"dependencies": {"pip": ["torch>=2.0"]}}},
        }
        result = group_workloads_by_dependencies(workloads, ["workload1", "workload2"])

        # Should have two separate groups
        assert len(result) == 2
        all_workloads = []
        for group_workloads in result.values():
            all_workloads.extend(group_workloads)
        assert set(all_workloads) == {"workload1", "workload2"}


class TestCloneGitRepos:
    """Test git repository cloning functionality."""

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.run')
    def test_clone_git_repos_success(self, mock_run, mock_uses_lfs):
        """Test successful git repository cloning."""
        mock_run.return_value = subprocess.CompletedProcess([], 0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        git_deps = {"nemo": {"url": "https://github.com/NVIDIA/NeMo.git", "commit": "v1.0"}}

        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                clone_git_repos(git_deps, "/tmp/test", "/tmp/install")

        # Should call: bare clone, clone from cache, remote setup, checkout (caching enabled)
        assert mock_run.call_count == 4

    @patch('subprocess.run')
    def test_clone_git_repos_already_exists(self, mock_run):
        """Test git repository cloning when directory already exists."""
        # Set up mock responses for: git rev-parse (cache), cat-file (commit),
        # git rev-parse (target), git remote (target), fetch cache, checkout
        mock_run.side_effect = [
            subprocess.CompletedProcess([], 0),  # git rev-parse for cache validation
            subprocess.CompletedProcess([], 0),  # git cat-file for commit existence
            subprocess.CompletedProcess([], 0),  # git rev-parse for target validation
            subprocess.CompletedProcess([], 0, stdout="origin\ncache\n"),  # git remote for target
            subprocess.CompletedProcess([], 0),  # git fetch cache
            subprocess.CompletedProcess([], 0),  # git checkout
        ]

        git_deps = {"nemo": {"url": "https://github.com/NVIDIA/NeMo.git", "commit": "v1.0"}}

        with patch('os.path.exists', return_value=True):
            clone_git_repos(git_deps, "/tmp/test", "/tmp/install")

        # Should call: validate cache, commit check, validate target (inc. remotes), fetch cache, checkout
        assert mock_run.call_count == 6

    def test_clone_git_repos_empty_deps(self):
        """Test git repository cloning with empty dependencies."""
        # Should not raise any exceptions
        clone_git_repos({}, "/tmp/test", "/tmp/install")


class TestInstallDependencies:
    """Test shared dependency installation functionality."""

    @patch('subprocess.run')
    @patch('os.path.join')
    def test_install_dependencies_pip_with_install_target(self, mock_join, mock_run):
        """Test installing pip dependencies with install_target."""
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        mock_join.side_effect = lambda *args: "/".join(args)

        dependencies = {
            "pip": [
                {
                    "package": "transformer_engine",
                    "url": "https://github.com/NVIDIA/TransformerEngine.git",
                    "commit": "v1.0",
                    "install_target": ".[pytorch]",
                }
            ]
        }

        with patch('os.path.exists', return_value=True):
            install_dependencies("/tmp/venv", "venv", dependencies, "/tmp/workload", {})

        # Verify pip install was called with install_target
        pip_calls = [call for call in mock_run.call_args_list if 'pip' in str(call)]
        assert len(pip_calls) == 1
        pip_call = pip_calls[0]
        assert '.[pytorch]' in pip_call[0][0]  # Should contain the install_target

    @patch('subprocess.run')
    @patch('os.path.join')
    def test_install_dependencies_editable_pip(self, mock_join, mock_run):
        """Test installing editable pip dependencies."""
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        mock_join.side_effect = lambda *args: "/".join(args)

        dependencies = {
            "pip": [
                {
                    "package": "my_package",
                    "url": "https://github.com/example/my_package.git",
                    "commit": "abc123",
                    "editable": True,
                }
            ]
        }

        with patch('os.path.exists', return_value=True):
            install_dependencies("/tmp/venv", "venv", dependencies, "/tmp/workload", {})

        # Verify pip install was called with -e flag
        pip_calls = [call for call in mock_run.call_args_list if 'pip' in str(call)]
        assert len(pip_calls) == 1
        pip_call = pip_calls[0]
        assert '-e' in pip_call[0][0]

    @patch('subprocess.run')
    @patch('os.path.join')
    def test_install_dependencies_git_pip_type(self, mock_join, mock_run):
        """Test handling git dependencies marked for pip installation."""
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        mock_join.side_effect = lambda *args: "/".join(args)

        dependencies = {
            "git": {
                "my_package": {
                    "url": "https://github.com/example/my_package.git",
                    "commit": "abc123",
                    "install_method": {"type": "pip"},
                }
            }
        }

        with patch('builtins.print') as mock_print:
            install_dependencies("/tmp/venv", "venv", dependencies, "/tmp/workload", {})

        # Verify correct message was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        pip_message_found = any("will be installed via pip" in call for call in print_calls)
        assert pip_message_found

    @patch('subprocess.run')
    @patch('os.path.join')
    def test_install_dependencies_mixed_types(self, mock_join, mock_run):
        """Test installing mixed dependency types (git script, git pip, regular pip)."""
        mock_run.return_value = subprocess.CompletedProcess([], 0)
        mock_join.side_effect = lambda *args: "/".join(args)

        dependencies = {
            "git": {
                "script_repo": {
                    "url": "https://github.com/example/script_repo.git",
                    "commit": "script123",
                    "install_method": {"type": "script", "path": "install.sh"},
                },
                "pip_repo": {
                    "url": "https://github.com/example/pip_repo.git",
                    "commit": "pip123",
                    "install_method": {"type": "pip"},
                },
            },
            "pip": [
                "numpy==1.24.0",
                {
                    "package": "pip_repo",
                    "url": "https://github.com/example/pip_repo.git",
                    "commit": "pip123",
                    "editable": True,
                },
            ],
        }

        with patch('os.path.exists', return_value=True), patch('os.chmod'), patch('builtins.print'):
            install_dependencies("/tmp/venv", "venv", dependencies, "/tmp/workload", {})

        # Verify script execution was attempted
        script_calls = [
            call for call in mock_run.call_args_list if len(call[0]) > 0 and 'install.sh' in str(call[0][0])
        ]
        assert len(script_calls) == 1

        # Verify pip installs were called
        pip_calls = [call for call in mock_run.call_args_list if 'pip' in str(call)]
        assert len(pip_calls) == 2  # numpy and pip_repo


# Fixtures for reusable test data
@pytest.fixture
def sample_workload_with_install_target():
    """Sample workload data with install_target pip dependency."""
    return {
        "setup": {
            "dependencies": {
                "pip": [
                    {
                        "package": "transformer_engine",
                        "url": "https://github.com/NVIDIA/TransformerEngine.git",
                        "commit": "v1.0",
                        "install_target": ".[pytorch]",
                    }
                ]
            }
        }
    }


@pytest.fixture
def sample_workload_with_editable():
    """Sample workload data with editable pip dependency."""
    return {
        "setup": {
            "dependencies": {
                "pip": [
                    {
                        "package": "my_package",
                        "url": "https://github.com/example/my_package.git",
                        "commit": "abc123",
                        "editable": True,
                    }
                ]
            }
        }
    }

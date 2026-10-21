#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for git caching functionality using bare repositories."""

import os
import subprocess
from unittest.mock import Mock, patch

import pytest

from llmb_install.core.dependency import (
    _cleanup_invalid_git_repo,
    _clone_or_get_from_cache,
    _commit_exists_in_repo,
    _derive_cache_name_from_url,
    _derive_repo_name_from_url,
    _get_default_branch,
    _get_git_cache_dir,
    _has_required_remotes,
    _is_valid_git_repo,
    _repo_uses_lfs,
    clone_git_repos,
)


class TestGitCaching:
    """Test git caching functionality with bare repositories."""

    def test_derive_repo_name_from_url(self):
        """Test deriving repository names from various URL formats."""
        test_cases = [
            ("https://github.com/NVIDIA/NeMo.git", "NeMo"),
            ("https://github.com/user/repo", "repo"),
            ("git@github.com:user/project.git", "project"),
            ("https://gitlab.com/group/subgroup/name.git", "name"),
        ]

        for url, expected in test_cases:
            result = _derive_repo_name_from_url(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_derive_repo_name_from_url_error_handling(self):
        """Test error handling for malformed URLs in repository name derivation."""
        error_cases = [
            ("", "Invalid repository URL"),  # Empty string
            (None, "Invalid repository URL"),  # None value
            ("https://github.com/user/.git", "Cannot derive repository name"),  # Empty repo name after removing .git
            (123, "Invalid repository URL"),  # Non-string type
        ]

        for url, expected_error in error_cases:
            with pytest.raises(ValueError, match=expected_error):
                _derive_repo_name_from_url(url)

    def test_derive_cache_name_from_url(self):
        """Test deriving unique cache names from URLs."""
        test_cases = [
            ("https://github.com/NVIDIA/NeMo.git", "github.com_NVIDIA_NeMo"),
            ("git@gitlab.com:user/project.git", "gitlab.com_user_project"),
            ("https://bitbucket.org/team/repo", "bitbucket.org_team_repo"),
        ]

        for url, expected in test_cases:
            result = _derive_cache_name_from_url(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_derive_cache_name_collision_avoidance(self):
        """Test that different URLs with same repo name get different cache names."""
        url1 = "https://github.com/user1/repo.git"
        url2 = "https://github.com/user2/repo.git"

        cache_name1 = _derive_cache_name_from_url(url1)
        cache_name2 = _derive_cache_name_from_url(url2)

        # Should have different cache names but same repo name
        assert cache_name1 != cache_name2
        assert _derive_repo_name_from_url(url1) == _derive_repo_name_from_url(url2)

    def test_get_git_cache_dir(self):
        """Test cache directory path generation."""
        install_path = "/path/to/install"
        expected = os.path.join(install_path, ".cache", "git_repos")
        result = _get_git_cache_dir(install_path)
        assert result == expected

    @patch('subprocess.run')
    def test_commit_exists_in_repo_success(self, mock_subprocess):
        """Test commit existence check when commit exists."""
        mock_subprocess.return_value = Mock(returncode=0)

        result = _commit_exists_in_repo("/path/to/repo", "abc123")

        mock_subprocess.assert_called_once_with(
            ['git', 'cat-file', '-e', 'abc123'], cwd="/path/to/repo", check=True, capture_output=True, text=True
        )
        assert result is True

    @patch('subprocess.run')
    def test_commit_exists_in_repo_failure(self, mock_subprocess):
        """Test commit existence check when commit doesn't exist."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ['git', 'cat-file'])

        result = _commit_exists_in_repo("/path/to/repo", "nonexistent")

        assert result is False

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_clone_or_get_from_cache_new_repo(self, mock_makedirs, mock_exists, mock_subprocess, mock_uses_lfs):
        """Test caching behavior when repository doesn't exist in cache."""
        # Setup mocks
        mock_exists.side_effect = lambda path: False  # Neither cache nor target exist
        mock_subprocess.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        result = _clone_or_get_from_cache(
            name="test_repo",
            repo_url="https://github.com/test/repo.git",
            install_path="/install",
            target_dir="/target",
            commit="abc123",
        )

        # Verify git operations: mirror clone, clone from cache, remote setup
        expected_calls = [
            (
                [
                    'git',
                    'clone',
                    '--mirror',
                    'https://github.com/test/repo.git',
                    '/install/.cache/git_repos/github.com_test_repo.git',
                ],
                {'check': True, 'capture_output': True, 'text': True},
            ),
            (
                [
                    'git',
                    'clone',
                    '--no-checkout',
                    '-o',
                    'cache',
                    '/install/.cache/git_repos/github.com_test_repo.git',
                    '/target/repo',
                ],
                {'check': True, 'capture_output': True, 'text': True},
            ),
            (
                ['git', 'remote', 'add', 'origin', 'https://github.com/test/repo.git'],
                {'cwd': '/target/repo', 'check': True, 'capture_output': True, 'text': True},
            ),
        ]

        assert mock_subprocess.call_count == len(expected_calls)
        for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
            actual_call = mock_subprocess.call_args_list[i]
            assert actual_call.args[0] == expected_args
            for key, value in expected_kwargs.items():
                assert actual_call.kwargs[key] == value

        assert result == "/target/repo"

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_clone_or_get_from_cache_existing_cache(self, mock_makedirs, mock_exists, mock_subprocess, mock_uses_lfs):
        """Test behavior when cache exists but target doesn't."""

        # Setup mocks - cache exists, target doesn't
        def mock_exists_side_effect(path):
            return '/install/.cache/git_repos/github.com_test_repo.git' in path

        mock_exists.side_effect = mock_exists_side_effect
        mock_subprocess.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        result = _clone_or_get_from_cache(
            name="test_repo",
            repo_url="https://github.com/test/repo.git",
            install_path="/install",
            target_dir="/target",
            commit="abc123",
        )

        # Should validate cache, check commit exists, then clone from cache to target
        expected_calls = [
            (
                ['git', '--git-dir', '/install/.cache/git_repos/github.com_test_repo.git', 'rev-parse', '--git-dir'],
                {
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'cat-file', '-e', 'abc123'],
                {
                    'cwd': '/install/.cache/git_repos/github.com_test_repo.git',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                [
                    'git',
                    'clone',
                    '--no-checkout',
                    '-o',
                    'cache',
                    '/install/.cache/git_repos/github.com_test_repo.git',
                    '/target/repo',
                ],
                {'check': True, 'capture_output': True, 'text': True},
            ),
            (
                ['git', 'remote', 'add', 'origin', 'https://github.com/test/repo.git'],
                {'cwd': '/target/repo', 'check': True, 'capture_output': True, 'text': True},
            ),
        ]

        assert mock_subprocess.call_count == len(expected_calls)
        for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
            actual_call = mock_subprocess.call_args_list[i]
            assert actual_call.args[0] == expected_args
            for key, value in expected_kwargs.items():
                assert actual_call.kwargs[key] == value

        assert result == "/target/repo"

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_clone_or_get_from_cache_existing_target(self, mock_makedirs, mock_exists, mock_subprocess):
        """Test behavior when target repo already exists and commit exists."""
        # Setup mocks - both cache and target exist
        mock_exists.return_value = True

        # Mock git commands in order: git rev-parse (cache), cat-file (commit check in cache),
        # git rev-parse (target), git remote (remote check), cat-file (commit check in target)
        # Note: With our optimization, we only fetch if commit is missing
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git rev-parse for cache validation
            Mock(returncode=0),  # git cat-file for commit existence in cache
            Mock(returncode=0),  # git rev-parse for target validation
            Mock(stdout="origin\ncache\n", returncode=0),  # git remote for target
            Mock(returncode=0),  # git cat-file for commit existence in target (exists, so no fetch needed)
        ]

        result = _clone_or_get_from_cache(
            name="test_repo",
            repo_url="https://github.com/test/repo.git",
            install_path="/install",
            target_dir="/target",
            commit="abc123",
        )

        # Should validate cache, check commit exists in cache, validate target (including remotes),
        # check commit exists in target (it does, so no fetch)
        expected_calls = [
            (
                ['git', '--git-dir', '/install/.cache/git_repos/github.com_test_repo.git', 'rev-parse', '--git-dir'],
                {
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'cat-file', '-e', 'abc123'],
                {
                    'cwd': '/install/.cache/git_repos/github.com_test_repo.git',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', '--git-dir', '/target/repo/.git', 'rev-parse', '--git-dir'],
                {
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'remote'],
                {
                    'cwd': '/target/repo',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
            (
                ['git', 'cat-file', '-e', 'abc123'],
                {
                    'cwd': '/target/repo',
                    'check': True,
                    'capture_output': True,
                    'text': True,
                },
            ),
        ]

        assert mock_subprocess.call_count == len(expected_calls)
        for i, (expected_args, expected_kwargs) in enumerate(expected_calls):
            actual_call = mock_subprocess.call_args_list[i]
            assert actual_call.args[0] == expected_args
            for key, value in expected_kwargs.items():
                assert actual_call.kwargs[key] == value

        assert result == "/target/repo"


class TestCloneGitReposIntegration:
    """Integration tests for clone_git_repos function."""

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('llmb_install.core.dependency._commit_exists_in_repo')
    def test_clone_git_repos_with_caching(
        self, mock_commit_exists, mock_makedirs, mock_exists, mock_subprocess, mock_uses_lfs
    ):
        """Test clone_git_repos with caching enabled."""
        git_deps = {'nemo': {'url': 'https://github.com/NVIDIA/NeMo.git', 'commit': 'abc123'}}

        # Setup mocks
        mock_exists.side_effect = lambda path: False  # Nothing exists initially
        mock_commit_exists.return_value = True  # Commit exists, no need to fetch
        mock_subprocess.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        with patch('builtins.print'):  # Suppress output
            clone_git_repos(git_deps, "/target", "/install")

        # Should have called: bare clone, clone from cache, remote setup, checkout
        assert mock_subprocess.call_count == 4

        # Check that checkout was called with correct commit
        checkout_calls = [call for call in mock_subprocess.call_args_list if call.args[0][0:2] == ['git', 'checkout']]
        assert len(checkout_calls) == 1
        assert checkout_calls[0].args[0] == ['git', 'checkout', '-f', 'abc123']

    @patch('os.environ.get')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('llmb_install.core.dependency._clone_or_get_from_cache')
    def test_clone_git_repos_cache_disabled(self, mock_cache_clone, mock_exists, mock_subprocess, mock_env_get):
        """Test clone_git_repos with caching disabled by environment variable."""
        git_deps = {'nemo': {'url': 'https://github.com/NVIDIA/NeMo.git', 'commit': 'abc123'}}

        # Setup mocks
        mock_env_get.return_value = 'true'  # LLMB_DISABLE_GIT_CACHE=true
        mock_exists.return_value = False
        mock_subprocess.return_value = Mock(returncode=0)

        with patch('builtins.print'):  # Suppress output
            clone_git_repos(git_deps, "/target", "/install")

        # Should NOT call cache function when disabled
        mock_cache_clone.assert_not_called()

        # Should directly clone and checkout
        clone_calls = [
            call
            for call in mock_subprocess.call_args_list
            if call.args[0][0:2] == ['git', 'clone'] and '--bare' not in call.args[0]
        ]
        assert len(clone_calls) >= 1

    def test_clone_git_repos_empty_deps(self):
        """Test clone_git_repos with empty dependencies."""
        # Should return immediately without error
        result = clone_git_repos({}, "/target", "/install")
        assert result is None

    def test_clone_git_repos_none_deps(self):
        """Test clone_git_repos with None dependencies."""
        # Should return immediately without error
        result = clone_git_repos(None, "/target", "/install")
        assert result is None

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('llmb_install.core.dependency._commit_exists_in_repo')
    def test_git_error_handling(self, mock_commit_exists, mock_makedirs, mock_exists, mock_subprocess, mock_uses_lfs):
        """Test error handling when git commands fail."""
        git_deps = {'nemo': {'url': 'https://github.com/NVIDIA/NeMo.git', 'commit': 'abc123'}}

        # Setup mocks
        mock_exists.side_effect = lambda path: False
        mock_commit_exists.return_value = True
        mock_uses_lfs.return_value = False  # Non-LFS repo

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[0:2] == ['git', 'checkout']:
                raise subprocess.CalledProcessError(1, cmd, stderr="Commit not found")
            return Mock(returncode=0, stdout='')

        mock_subprocess.side_effect = subprocess_side_effect

        with pytest.raises(subprocess.CalledProcessError):
            with patch('builtins.print'):  # Suppress output
                clone_git_repos(git_deps, "/target", "/install")


class TestGitRemoteValidation:
    """Test git remote validation functionality."""

    @patch('subprocess.run')
    def test_has_required_remotes_success(self, mock_subprocess):
        """Test checking required remotes when all exist."""
        mock_subprocess.return_value = Mock(stdout="origin\ncache\nupstream\n", returncode=0)

        result = _has_required_remotes("/path/to/repo", ["origin", "cache"])

        mock_subprocess.assert_called_once_with(
            ['git', 'remote'], cwd="/path/to/repo", check=True, capture_output=True, text=True
        )
        assert result is True

    @patch('subprocess.run')
    def test_has_required_remotes_missing(self, mock_subprocess):
        """Test checking required remotes when some are missing."""
        mock_subprocess.return_value = Mock(stdout="origin\n", returncode=0)

        result = _has_required_remotes("/path/to/repo", ["origin", "cache"])

        mock_subprocess.assert_called_once_with(
            ['git', 'remote'], cwd="/path/to/repo", check=True, capture_output=True, text=True
        )
        assert result is False

    @patch('subprocess.run')
    def test_has_required_remotes_empty_output(self, mock_subprocess):
        """Test checking required remotes when no remotes exist."""
        mock_subprocess.return_value = Mock(stdout="", returncode=0)

        result = _has_required_remotes("/path/to/repo", ["origin"])

        assert result is False

    @patch('subprocess.run')
    def test_has_required_remotes_git_error(self, mock_subprocess):
        """Test checking required remotes when git command fails."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ['git', 'remote'])

        result = _has_required_remotes("/path/to/repo", ["origin"])

        assert result is False

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_is_valid_git_repo_with_required_remotes(self, mock_exists, mock_subprocess):
        """Test git repo validation with required remotes check."""
        mock_exists.return_value = True
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git rev-parse call
            Mock(stdout="origin\ncache\n", returncode=0),  # git remote call
        ]

        result = _is_valid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        assert result is True
        # Should call both git rev-parse and git remote
        assert mock_subprocess.call_count == 2

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_is_valid_git_repo_missing_required_remotes(self, mock_exists, mock_subprocess):
        """Test git repo validation fails when required remotes are missing."""
        mock_exists.return_value = True
        mock_subprocess.side_effect = [
            Mock(returncode=0),  # git rev-parse call
            Mock(stdout="origin\n", returncode=0),  # git remote call (missing 'cache')
        ]

        result = _is_valid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        assert result is False

    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_is_valid_git_repo_no_remote_check_when_none_required(self, mock_exists, mock_subprocess):
        """Test git repo validation without remote check when no remotes required."""
        mock_exists.return_value = True
        mock_subprocess.return_value = Mock(returncode=0)  # git rev-parse call

        result = _is_valid_git_repo("/path/to/repo", is_bare=False, required_remotes=None)

        assert result is True
        # Should only call git rev-parse, not git remote
        assert mock_subprocess.call_count == 1
        assert mock_subprocess.call_args[0][0] == ['git', '--git-dir', '/path/to/repo/.git', 'rev-parse', '--git-dir']

    @patch('llmb_install.core.dependency._is_valid_git_repo')
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_cleanup_invalid_git_repo_with_required_remotes(self, mock_rmtree, mock_exists, mock_is_valid):
        """Test cleanup when repo exists but is invalid due to missing remotes."""
        mock_exists.return_value = True
        mock_is_valid.return_value = False

        _cleanup_invalid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        mock_is_valid.assert_called_once_with("/path/to/repo", is_bare=False, required_remotes=["cache"])
        mock_rmtree.assert_called_once_with("/path/to/repo")

    @patch('llmb_install.core.dependency._is_valid_git_repo')
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_cleanup_valid_git_repo_not_removed(self, mock_rmtree, mock_exists, mock_is_valid):
        """Test cleanup when repo is valid - should not be removed."""
        mock_exists.return_value = True
        mock_is_valid.return_value = True

        _cleanup_invalid_git_repo("/path/to/repo", is_bare=False, required_remotes=["cache"])

        mock_is_valid.assert_called_once_with("/path/to/repo", is_bare=False, required_remotes=["cache"])
        mock_rmtree.assert_not_called()


class TestGitLfsDetection:
    """Test git-lfs detection functionality."""

    @patch('subprocess.run')
    def test_repo_uses_lfs_with_lfs_files(self, mock_subprocess):
        """Test detection of repositories using git-lfs."""
        # Mock .gitattributes content with LFS filters
        gitattributes_content = """*.psd filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
"""
        mock_subprocess.return_value = Mock(returncode=0, stdout=gitattributes_content)

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is True
        mock_subprocess.assert_called_once_with(
            ['git', 'show', 'abc123:.gitattributes'],
            cwd="/path/to/repo",
            capture_output=True,
            text=True,
            check=False,
        )

    @patch('subprocess.run')
    def test_repo_uses_lfs_without_lfs_files(self, mock_subprocess):
        """Test detection of repositories not using git-lfs."""
        # Mock .gitattributes without LFS filters
        gitattributes_content = """* text=auto
*.py text
*.md text
"""
        mock_subprocess.return_value = Mock(returncode=0, stdout=gitattributes_content)

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is False

    @patch('subprocess.run')
    def test_repo_uses_lfs_no_gitattributes(self, mock_subprocess):
        """Test detection when .gitattributes doesn't exist."""
        # Git returns non-zero when file doesn't exist
        mock_subprocess.return_value = Mock(returncode=128, stdout='', stderr='fatal: path not in commit')

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is False

    @patch('subprocess.run')
    def test_repo_uses_lfs_handles_exceptions(self, mock_subprocess):
        """Test that exceptions are handled gracefully."""
        mock_subprocess.side_effect = OSError("Command not found")

        result = _repo_uses_lfs("/path/to/repo", "abc123")

        assert result is False


class TestDefaultBranchDetection:
    """Test default branch detection functionality."""

    @patch('subprocess.run')
    def test_get_default_branch_main(self, mock_subprocess):
        """Test getting default branch when it's 'main'."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="refs/remotes/cache/main\n")

        result = _get_default_branch("/path/to/repo", "cache")

        assert result == "main"
        mock_subprocess.assert_called_once_with(
            ['git', 'symbolic-ref', 'refs/remotes/cache/HEAD'],
            cwd="/path/to/repo",
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('subprocess.run')
    def test_get_default_branch_master(self, mock_subprocess):
        """Test getting default branch when it's 'master'."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="refs/remotes/origin/master\n")

        result = _get_default_branch("/path/to/repo", "origin")

        assert result == "master"

    @patch('subprocess.run')
    def test_get_default_branch_custom(self, mock_subprocess):
        """Test getting custom default branch names."""
        test_cases = [
            ("develop", "refs/remotes/cache/develop"),
            ("trunk", "refs/remotes/cache/trunk"),
            ("release", "refs/remotes/cache/release"),
        ]

        for expected_branch, symbolic_ref_output in test_cases:
            mock_subprocess.return_value = Mock(returncode=0, stdout=f"{symbolic_ref_output}\n")
            result = _get_default_branch("/path/to/repo", "cache")
            assert result == expected_branch, f"Failed for branch: {expected_branch}"

    @patch('subprocess.run')
    def test_get_default_branch_not_found(self, mock_subprocess):
        """Test when symbolic-ref command fails."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ['git', 'symbolic-ref'])

        result = _get_default_branch("/path/to/repo", "cache")

        assert result is None

    @patch('subprocess.run')
    def test_get_default_branch_malformed_output(self, mock_subprocess):
        """Test handling of malformed symbolic-ref output."""
        mock_subprocess.return_value = Mock(returncode=0, stdout="malformed-output")

        result = _get_default_branch("/path/to/repo", "cache")

        assert result is None

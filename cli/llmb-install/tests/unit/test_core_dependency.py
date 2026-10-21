# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.dependency module."""

import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from llmb_install.core.dependency import (
    _canonicalize_dependencies_for_grouping,
    _has_editable_pip_dep,
    _resolve_dependencies,
    clone_git_repos,
    group_workloads_by_dependencies,
    install_dependencies,
    print_dependency_group_summary,
)


class TestDependencyResolution(unittest.TestCase):
    """Test dependency resolution functions."""

    def test_has_editable_pip_dep(self):
        """Test detection of editable pip dependencies."""
        # Test with editable dependency
        deps_with_editable = {
            'pip': [
                {'package': 'torch', 'editable': True, 'url': 'https://github.com/pytorch/pytorch', 'commit': 'abc123'},
                'numpy',
            ]
        }
        self.assertTrue(_has_editable_pip_dep(deps_with_editable))

        # Test without editable dependency
        deps_without_editable = {'pip': ['torch', 'numpy']}
        self.assertFalse(_has_editable_pip_dep(deps_without_editable))

        # Test with no dependencies
        self.assertFalse(_has_editable_pip_dep(None))
        self.assertFalse(_has_editable_pip_dep({}))

        # Test with empty pip list
        deps_empty_pip = {'pip': []}
        self.assertFalse(_has_editable_pip_dep(deps_empty_pip))

    def test_resolve_dependencies_basic(self):
        """Test basic dependency resolution without repo_key references."""
        workload_data = {
            'setup': {
                'dependencies': {
                    'pip': ['torch', 'numpy'],
                    'git': {'some_repo': {'url': 'https://github.com/example/repo', 'commit': 'abc123'}},
                }
            }
        }

        resolved = _resolve_dependencies(workload_data)

        self.assertEqual(resolved['pip'], ['torch', 'numpy'])
        self.assertIn('some_repo', resolved['git'])
        self.assertEqual(resolved['git']['some_repo']['url'], 'https://github.com/example/repo')

    def test_resolve_dependencies_with_repo_key(self):
        """Test dependency resolution with repo_key references."""
        workload_data = {
            'setup': {
                'dependencies': {
                    'pip': [{'package': 'torch', 'repo_key': 'pytorch_repo', 'editable': True}],
                    'git': {'some_repo': {'repo_key': 'test_repo'}},
                }
            },
            'repositories': {
                'pytorch_repo': {'url': 'https://github.com/pytorch/pytorch', 'commit': 'def456'},
                'test_repo': {'url': 'https://github.com/test/repo', 'commit': 'ghi789'},
            },
        }

        resolved = _resolve_dependencies(workload_data)

        # Check pip dependency resolution
        pip_dep = resolved['pip'][0]
        self.assertEqual(pip_dep['package'], 'torch')
        self.assertEqual(pip_dep['url'], 'https://github.com/pytorch/pytorch')
        self.assertEqual(pip_dep['commit'], 'def456')
        self.assertTrue(pip_dep['editable'])
        self.assertNotIn('repo_key', pip_dep)

        # Check git dependency resolution
        self.assertEqual(resolved['git']['some_repo']['url'], 'https://github.com/test/repo')
        self.assertEqual(resolved['git']['some_repo']['commit'], 'ghi789')
        self.assertNotIn('repo_key', resolved['git']['some_repo'])

        # Check that editable pip dep was added to git deps
        self.assertIn('torch', resolved['git'])
        self.assertEqual(resolved['git']['torch']['url'], 'https://github.com/pytorch/pytorch')

    def test_resolve_dependencies_empty(self):
        """Test dependency resolution with no dependencies."""
        workload_data = {}
        self.assertIsNone(_resolve_dependencies(workload_data))

        workload_data = {'setup': {}}
        self.assertIsNone(_resolve_dependencies(workload_data))

    def test_canonicalize_dependencies_for_grouping(self):
        """Test canonicalization of dependencies for grouping."""
        dependencies = {
            'pip': [
                'torch',
                {'package': 'numpy', 'url': 'https://github.com/numpy/numpy', 'commit': 'abc123'},
                {'package': 'scipy'},
            ],
            'git': {
                'repo1': {'url': 'https://github.com/test/repo1', 'commit': 'def456'},
                'repo2': {'url': 'https://github.com/test/repo2', 'commit': 'ghi789'},
            },
            'sources': ['source1', 'source2'],
        }

        canonical = _canonicalize_dependencies_for_grouping(dependencies)

        self.assertIn('pip-spec|torch', canonical['pip'])
        self.assertIn('pip-dict|{"package": "scipy"}', canonical['pip'])
        self.assertIn('source1', canonical['sources'])
        self.assertIn('source2', canonical['sources'])
        self.assertIn(('https://github.com/numpy/numpy', 'abc123'), canonical['git_repos'])
        self.assertIn(('https://github.com/test/repo1', 'def456'), canonical['git_repos'])

        # Test with None
        self.assertIsNone(_canonicalize_dependencies_for_grouping(None))

    def test_group_workloads_by_dependencies(self):
        """Test grouping workloads by their dependencies."""
        workloads = {
            'workload1': {'setup': {'dependencies': {'pip': ['torch', 'numpy']}}},
            'workload2': {'setup': {'dependencies': {'pip': ['torch', 'numpy']}}},
            'workload3': {'setup': {'dependencies': {'pip': ['tensorflow']}}},
            'workload4': {
                # No dependencies - no venv required
                'setup': {}
            },
        }

        selected_keys = ['workload1', 'workload2', 'workload3', 'workload4']
        groups = group_workloads_by_dependencies(workloads, selected_keys)

        # workload1 and workload2 should be in the same group
        # workload3 should be in its own group
        # workload4 should be in the None group (no dependencies)

        self.assertIn(None, groups)
        self.assertEqual(groups[None], ['workload4'])

        # Find groups with multiple workloads
        multi_workload_groups = [group for group in groups.values() if len(group) > 1]
        self.assertEqual(len(multi_workload_groups), 1)
        self.assertIn('workload1', multi_workload_groups[0])
        self.assertIn('workload2', multi_workload_groups[0])


class TestDependencyInstallation(unittest.TestCase):
    """Test dependency installation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        import shutil

        self.addCleanup(lambda: shutil.rmtree(self.temp_dir))

    @patch('llmb_install.core.dependency._repo_uses_lfs')
    @patch('subprocess.run')
    def test_clone_git_repos(self, mock_run, mock_uses_lfs):
        """Test cloning git repositories."""
        git_deps = {
            'repo1': {'url': 'https://github.com/test/repo1.git', 'commit': 'abc123'},
            'repo2': {'url': 'https://github.com/test/repo2', 'commit': 'def456'},
        }

        mock_run.return_value = Mock(returncode=0, stdout='')
        mock_uses_lfs.return_value = False  # Non-LFS repo

        with patch('os.path.exists', return_value=False):
            with patch('os.makedirs'):
                with patch('builtins.print'):
                    clone_git_repos(git_deps, self.temp_dir, self.temp_dir)

        # Verify git operations were called - with caching we expect:
        # - bare clone for each repo (2)
        # - regular clone from cache for each repo (2)
        # - remote add for each repo (2)
        # - checkout for each repo (2)
        clone_calls = [call for call in mock_run.call_args_list if 'clone' in str(call)]
        checkout_calls = [call for call in mock_run.call_args_list if call.args[0][0:2] == ['git', 'checkout']]
        remote_calls = [call for call in mock_run.call_args_list if call.args[0][0:2] == ['git', 'remote']]

        # Verify we got the expected operations with exact counts
        self.assertEqual(
            len(clone_calls), 4, "Should have bare clone + regular clone for each repo (2 repos × 2 clones each)"
        )
        self.assertEqual(len(checkout_calls), 2, "Should have one checkout per repo")
        self.assertEqual(len(remote_calls), 2, "Should have one remote add per repo")

    @patch('subprocess.run')
    def test_clone_git_repos_with_existing_repo(self, mock_run):
        """Test cloning git repositories when repo already exists but commit is missing."""
        git_deps = {'existing_repo': {'url': 'https://github.com/test/existing.git', 'commit': 'xyz789'}}

        # Set up mock responses for the sequence of git commands
        # Simulate commit missing in target to trigger fetch
        def side_effect_fn(cmd, **kwargs):
            # First cat-file check (in cache) - exists
            if cmd == ['git', 'cat-file', '-e', 'xyz789'] and 'cache' in kwargs.get('cwd', ''):
                return Mock(returncode=0)
            # Second cat-file check (in target before fetch) - missing, will trigger fetch
            elif cmd == ['git', 'cat-file', '-e', 'xyz789'] and 'cache' not in kwargs.get('cwd', ''):
                if not hasattr(side_effect_fn, 'target_check_count'):
                    side_effect_fn.target_check_count = 0
                side_effect_fn.target_check_count += 1
                # First check: missing (raises error), second check after fetch: exists
                if side_effect_fn.target_check_count == 1:
                    raise subprocess.CalledProcessError(1, cmd)
                return Mock(returncode=0)
            elif cmd[0:2] == ['git', 'fetch']:
                return Mock(returncode=0)
            elif cmd[0:2] == ['git', 'checkout']:
                return Mock(returncode=0)
            elif cmd[0:2] == ['git', '--git-dir']:
                return Mock(returncode=0)
            elif cmd == ['git', 'remote']:
                return Mock(returncode=0, stdout="origin\ncache\n")
            return Mock(returncode=0)

        mock_run.side_effect = side_effect_fn

        with patch('os.path.exists', return_value=True):
            with patch('builtins.print'):
                clone_git_repos(git_deps, self.temp_dir, self.temp_dir)

        # When repo exists, should check commit and fetch from cache, but no clone calls
        clone_calls = [call for call in mock_run.call_args_list if 'clone' in str(call)]
        self.assertEqual(len(clone_calls), 0)

        # Should have commit check and fetch from cache
        cat_file_calls = [call for call in mock_run.call_args_list if 'cat-file' in str(call)]
        fetch_calls = [call for call in mock_run.call_args_list if 'fetch' in str(call)]
        self.assertGreaterEqual(len(cat_file_calls), 1, "Should have cat-file calls for commit checking")
        self.assertGreaterEqual(len(fetch_calls), 1, "Should have fetch calls for cache updates")

    def test_clone_git_repos_empty(self):
        """Test cloning with empty git dependencies."""
        with patch('builtins.print') as mock_print:
            clone_git_repos({}, self.temp_dir, self.temp_dir)

        # Should not print anything for empty deps
        mock_print.assert_not_called()

    @patch('subprocess.run')
    @patch('llmb_install.core.dependency._build_pip_install_command')
    def test_install_dependencies(self, mock_build_pip_cmd, mock_run):
        """Test installing shared dependencies."""
        dependencies = {
            'pip': ['torch', {'package': 'numpy', 'url': 'https://github.com/numpy/numpy', 'commit': 'abc123'}],
            'git': {
                'test_repo': {
                    'url': 'https://github.com/test/repo',
                    'commit': 'def456',
                    'install_method': {'type': 'script', 'path': 'install.sh'},
                }
            },
        }

        mock_build_pip_cmd.return_value = ['pip', 'install', 'torch']
        mock_run.return_value = Mock(returncode=0)

        venv_path = '/test/venv'
        workload_clone_path = '/test/clones'
        env = {'PATH': '/usr/bin'}

        with patch('os.path.join', side_effect=lambda *args: '/'.join(args)):
            with patch('os.chmod'):
                with patch('builtins.print'):
                    install_dependencies(venv_path, 'venv', dependencies, workload_clone_path, env)

        # Verify pip commands were built and executed
        self.assertTrue(mock_build_pip_cmd.called)

        # Verify git script was executed
        script_calls = [call for call in mock_run.call_args_list if 'install.sh' in str(call)]
        self.assertTrue(len(script_calls) > 0)


class TestDependencyDisplay(unittest.TestCase):
    """Test dependency display functions."""

    def test_print_dependency_group_summary(self):
        """Test printing dependency group summary."""
        dep_groups = {
            None: ['image_only_workload'],
            'hash1': ['workload1', 'workload2'],
            'hash2': ['workload3'],
            'editable-isolated::workload4': ['workload4'],
        }

        with patch('builtins.print') as mock_print:
            print_dependency_group_summary(dep_groups)

        # Verify various sections were printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        print_output = '\n'.join(print_calls)

        self.assertIn('Workload Installation Plan', print_output)
        self.assertIn('No dependency setup', print_output)
        self.assertIn('Individual installations', print_output)
        self.assertIn('Shared virtual environment groups', print_output)
        self.assertIn('image_only_workload', print_output)
        self.assertIn('workload1', print_output)


if __name__ == '__main__':
    unittest.main()

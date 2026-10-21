# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resume and fresh install bug fix."""

import tempfile
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.core.installer import Installer


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        install_path = temp_path / "install"
        original_repo = temp_path / "original_repo"

        install_path.mkdir(parents=True, exist_ok=True)
        original_repo.mkdir(parents=True, exist_ok=True)

        # Create a minimal workload structure in original repo
        workload_dir = original_repo / "test_workload"
        workload_dir.mkdir()
        (workload_dir / "metadata.yaml").write_text("name: test_workload\n")

        yield {
            'install_path': str(install_path),
            'original_repo': str(original_repo),
            'copied_repo': str(install_path / "llmb_repo"),
        }


@pytest.fixture
def sample_install_config(temp_dirs):
    """Create a sample InstallConfig for testing."""
    slurm_config = SlurmConfig(
        account="test_account",
        gpu_partition="gpu_partition",
        cpu_partition="cpu_partition",
        gpu_partition_gres=8,
        cpu_partition_gres=None,
    )

    return InstallConfig(
        install_path=temp_dirs['install_path'],
        venv_type="venv",
        gpu_type="h100",
        node_architecture="x86_64",
        slurm=slurm_config,
        selected_workloads=["test_workload"],
        install_method="local",
        environment_vars={},
        cache_dirs_configured=False,
        llmb_repo=temp_dirs['copied_repo'],  # This simulates resume state pointing to copied repo
    )


class TestResumeFreshInstallBugFix:
    """Test that the fresh install option during resume correctly resets repo location."""

    @patch('llmb_install.core.installer.find_llmb_repo_root')
    @patch('llmb_install.core.installer.build_workload_dict')
    @patch('llmb_install.core.installer.load_install_state')
    @patch('llmb_install.core.installer.clear_install_state')
    def test_fresh_install_resets_root_dir(
        self,
        mock_clear_state,
        mock_load_state,
        mock_build_workload_dict,
        mock_find_repo_root,
        temp_dirs,
        sample_install_config,
    ):
        """Test that opting for fresh install resets root_dir to original repo location."""

        # Setup: Simulate resume state where root_dir is set to copied repo
        installer = Installer()
        installer.root_dir = sample_install_config.llmb_repo  # Set to copied repo location

        # Mock find_llmb_repo_root to return the original repo location
        mock_find_repo_root.return_value = temp_dirs['original_repo']

        # Mock build_workload_dict to return a simple workload dict
        mock_build_workload_dict.return_value = {'test_workload': {'metadata': {'name': 'test_workload'}}}

        # Mock load_install_state to return resume state (4-tuple with None for existing_cluster_config)
        completed_workloads = []
        workload_venvs = {}
        existing_cluster_config = None
        mock_load_state.return_value = (
            sample_install_config,
            completed_workloads,
            workload_venvs,
            existing_cluster_config,
        )

        # Mock the show_configuration_summary to return 'no' (fresh install)
        with patch.object(installer, '_show_configuration_summary', return_value='no'):
            # Create a mock args object
            mock_args = MagicMock()
            mock_args.dev_mode = False
            mock_args.play = None

            # Call _check_and_handle_resume
            result = installer._check_and_handle_resume(
                mock_args,
                state_result=(sample_install_config, completed_workloads, workload_venvs, existing_cluster_config),
            )

            # Verify the result
            assert result is False, "Fresh install should return False to continue normal flow"

            # Verify that clear_install_state was called
            mock_clear_state.assert_called_once()

            # CRITICAL: Verify that root_dir was reset to original repo location
            assert (
                installer.root_dir == temp_dirs['original_repo']
            ), f"root_dir should be reset to original repo, got {installer.root_dir}"

            # Verify that force_fresh_repo_copy flag was set
            assert installer.force_fresh_repo_copy is True, "force_fresh_repo_copy should be set to True"

            # Verify that workloads were reloaded from original repo
            mock_build_workload_dict.assert_called_with(temp_dirs['original_repo'])
            assert installer.workloads == {'test_workload': {'metadata': {'name': 'test_workload'}}}

    @patch('llmb_install.core.installer.find_llmb_repo_root')
    @patch('llmb_install.core.installer.build_workload_dict')
    @patch('llmb_install.core.installer.load_install_state')
    def test_resume_keeps_copied_repo_location(
        self, mock_load_state, mock_build_workload_dict, mock_find_repo_root, sample_install_config
    ):
        """Test that opting to resume keeps the copied repo location."""

        # Setup: Simulate resume state where root_dir is set to copied repo
        installer = Installer()
        installer.root_dir = sample_install_config.llmb_repo  # Set to copied repo location

        # Mock build_workload_dict to return a simple workload dict
        mock_build_workload_dict.return_value = {'test_workload': {'metadata': {'name': 'test_workload'}}}

        # Mock load_install_state to return resume state (4-tuple with None for existing_cluster_config)
        completed_workloads = []
        workload_venvs = {}
        existing_cluster_config = None
        mock_load_state.return_value = (
            sample_install_config,
            completed_workloads,
            workload_venvs,
            existing_cluster_config,
        )

        # Mock the show_configuration_summary to return 'yes' (resume)
        with patch.object(installer, '_show_configuration_summary', return_value='yes'):
            with patch.object(installer, '_resume_installation', return_value=True):
                # Create a mock args object
                mock_args = MagicMock()
                mock_args.dev_mode = False
                mock_args.play = None

                # Call _check_and_handle_resume
                result = installer._check_and_handle_resume(
                    mock_args,
                    state_result=(sample_install_config, completed_workloads, workload_venvs, existing_cluster_config),
                )

                # Verify the result
                assert result is True, "Resume should return True"

                # Verify that root_dir was NOT changed (still points to copied repo)
                assert (
                    installer.root_dir == sample_install_config.llmb_repo
                ), "root_dir should remain at copied repo location for resume"

                # Verify that find_llmb_repo_root was NOT called
                mock_find_repo_root.assert_not_called()


class TestResumeFailurePropagation:
    """Resume action errors must reach the central CLI error handler."""

    @staticmethod
    def _resume_args():
        args = MagicMock()
        args.dev_mode = False
        args.play = None
        return args

    def test_resume_installation_failure_escapes_resume_handling(self, sample_install_config, capsys):
        installer = Installer()

        with patch.object(installer, '_show_configuration_summary', return_value='yes'):
            with patch.object(installer, '_perform_installation', side_effect=RuntimeError('git fetch failed')):
                with pytest.raises(RuntimeError, match='git fetch failed'):
                    installer._check_and_handle_resume(
                        self._resume_args(),
                        state_result=(sample_install_config, [], {}, None),
                    )

        # Reporting the error is left to the CLI handler.
        out = capsys.readouterr().out
        assert 'git fetch failed' not in out
        assert 'fresh installation' not in out

    def test_edit_resume_failure_escapes_resume_handling(self, sample_install_config, capsys):
        installer = Installer()

        with patch.object(installer, '_show_configuration_summary', return_value='edit'):
            with patch.object(installer, '_collect_limited_configuration', side_effect=RuntimeError('prompt failed')):
                with pytest.raises(RuntimeError, match='prompt failed'):
                    installer._check_and_handle_resume(
                        self._resume_args(),
                        state_result=(sample_install_config, [], {}, None),
                    )

        out = capsys.readouterr().out
        assert 'prompt failed' not in out
        assert 'fresh installation' not in out

    def test_resume_completion_failure_escapes_resume_handling(self, sample_install_config, capsys):
        installer = Installer()
        sample_install_config.selected_workloads.append('another_workload')
        completed_workloads = ['test_workload']
        state = (sample_install_config, completed_workloads, {}, {'workloads': {}})

        # Patch at the definition module: the 'clear' path re-imports
        # load_install_state locally via `from llmb_install.config.system import ...`.
        with patch.object(installer, '_show_configuration_summary', return_value='clear'):
            with patch('llmb_install.config.system.load_install_state', return_value=state):
                with patch(
                    'llmb_install.core.installer.create_cluster_config', side_effect=RuntimeError('write failed')
                ):
                    with pytest.raises(RuntimeError, match='write failed'):
                        installer._check_and_handle_resume(self._resume_args(), state_result=state)

        out = capsys.readouterr().out
        assert 'write failed' not in out
        assert 'Resume state preserved for retry.' in out

    def test_edit_completion_failure_preserves_original_exception_type(self, sample_install_config):
        """A failure in the nested edit -> _complete_installation path must surface the
        original exception (not a doubly-wrapped _ResumeExecutionError)."""
        installer = Installer()
        sample_install_config.selected_workloads.append('another_workload')
        completed_workloads = ['test_workload']
        state = (sample_install_config, completed_workloads, {}, {'workloads': {}})

        # Editing with no newly-selected workloads routes into _complete_installation,
        # which itself wraps its failure — exercising the nested-wrapping guard.
        new_config = MagicMock()
        new_config.selected_workloads = []

        # Patch at the definition module: the edit path re-imports
        # load_install_state locally via `from llmb_install.config.system import ...`.
        with patch.object(installer, '_show_configuration_summary', return_value='edit'):
            with patch.object(installer, '_collect_limited_configuration', return_value=new_config):
                with patch('llmb_install.config.system.load_install_state', return_value=state):
                    with patch(
                        'llmb_install.core.installer.create_cluster_config',
                        side_effect=RuntimeError('write failed'),
                    ):
                        with pytest.raises(RuntimeError, match='write failed') as exc_info:
                            installer._check_and_handle_resume(self._resume_args(), state_result=state)

        # Must be the original RuntimeError, not a leaked internal wrapper type.
        assert type(exc_info.value) is RuntimeError


# Patches needed to bypass pre-loop setup in _perform_installation
_INSTALLER_MODULE = 'llmb_install.core.installer'
_SETUP_PATCHES = [
    f'{_INSTALLER_MODULE}.{name}'
    for name in [
        'install_state_exists',
        'clear_install_state',
        'save_install_state',
        'create_llmb_run_symlink',
        'setup_compute_uv_binaries',
        'get_required_images',
        'fetch_container_images',
        'download_huggingface_files_for_workloads',
        'get_required_tools',
        'fetch_and_install_tools',
        'print_dependency_group_summary',
        'create_cluster_config',
        'save_system_config',
        'group_workloads_by_dependencies',
    ]
]

# Patches for functions called inside the installation loop body
_LOOP_BODY_PATCHES = [
    f'{_INSTALLER_MODULE}.{name}'
    for name in [
        'run_setup_tasks',
        'create_virtual_environment',
        'install_dependencies',
        '_resolve_dependencies',
        'get_venv_environment',
        'clone_git_repos',
    ]
]

_SHARED_SLURM_CONFIG = SlurmConfig(
    account="test_account",
    gpu_partition="gpu_partition",
    cpu_partition="cpu_partition",
    gpu_partition_gres=8,
    cpu_partition_gres=None,
)


def _make_install_config(temp_dirs, selected_workloads):
    """Create an InstallConfig with the given workloads."""
    return InstallConfig(
        install_path=temp_dirs['install_path'],
        venv_type="venv",
        gpu_type="h100",
        node_architecture="x86_64",
        slurm=_SHARED_SLURM_CONFIG,
        selected_workloads=selected_workloads,
        install_method="local",
        environment_vars={},
        cache_dirs_configured=False,
        llmb_repo=temp_dirs['copied_repo'],
    )


@pytest.fixture
def install_env(temp_dirs):
    """Set up all mocks for _perform_installation and yield (mocks, installer).

    Handles patch lifecycle via ExitStack — no manual try/finally needed.
    """
    with ExitStack() as stack:
        mocks = {}
        for target in _SETUP_PATCHES + _LOOP_BODY_PATCHES:
            name = target.rsplit('.', 1)[-1]
            mocks[name] = stack.enter_context(patch(target))

        # Common defaults
        mocks['get_required_images'].return_value = {}
        mocks['get_required_tools'].return_value = []
        mocks['install_state_exists'].return_value = False
        mocks['_resolve_dependencies'].return_value = {'pip': []}
        mocks['get_venv_environment'].return_value = {}

        installer = Installer()
        installer.root_dir = temp_dirs['original_repo']

        yield mocks, installer


class TestPartialInstallConfigWrite:
    """Test that partial cluster_config.yaml is written when some workloads fail."""

    def test_partial_config_written_on_failure(self, temp_dirs, install_env):
        """Verify create_cluster_config is called with completed_workloads when an exception occurs."""
        mocks, installer = install_env
        install_config = _make_install_config(temp_dirs, ["workload_a", "workload_b"])

        mocks['group_workloads_by_dependencies'].return_value = {
            'hash_a': ['workload_a'],
            'hash_b': ['workload_b'],
        }

        call_count = [0]

        def fake_save(ic, wk_keys, completed, venvs, ecc):
            call_count[0] += 1
            if call_count[0] == 1:
                completed.extend(wk_keys)
            else:
                raise RuntimeError("Installation failed for workload_b")

        filtered_workloads = {
            'workload_a': {'path': '/tmp/a', 'setup': {}},
            'workload_b': {'path': '/tmp/b', 'setup': {}},
        }

        with patch.object(installer, '_save_installation_progress', side_effect=fake_save):
            with pytest.raises(RuntimeError, match="Installation failed for workload_b"):
                installer._perform_installation(install_config, filtered_workloads)

        mocks['create_cluster_config'].assert_called_once()
        assert mocks['create_cluster_config'].call_args[0][2] == ['workload_a']

    def test_no_partial_config_when_no_completions(self, temp_dirs, install_env):
        """Verify create_cluster_config is NOT called when no workloads completed."""
        mocks, installer = install_env
        install_config = _make_install_config(temp_dirs, ["workload_a"])

        mocks['group_workloads_by_dependencies'].return_value = {
            'hash_a': ['workload_a'],
        }

        def fake_save(ic, wk_keys, completed, venvs, ecc):
            raise RuntimeError("Installation failed")

        filtered_workloads = {
            'workload_a': {'path': '/tmp/a', 'setup': {}},
        }

        with patch.object(installer, '_save_installation_progress', side_effect=fake_save):
            with pytest.raises(RuntimeError, match="Installation failed"):
                installer._perform_installation(install_config, filtered_workloads)

        mocks['create_cluster_config'].assert_not_called()

    def test_partial_config_includes_prior_completed_on_resume(self, temp_dirs, install_env):
        """Resume with prior completed workloads; verify partial config includes all completed."""
        mocks, installer = install_env
        install_config = _make_install_config(temp_dirs, ["workload_c", "workload_d"])

        mocks['group_workloads_by_dependencies'].return_value = {
            'hash_c': ['workload_c'],
            'hash_d': ['workload_d'],
        }

        call_count = [0]

        def fake_save(ic, wk_keys, completed, venvs, ecc):
            call_count[0] += 1
            if call_count[0] == 1:
                completed.extend(wk_keys)
                venvs['workload_c'] = '/path/to/venv_c'
            else:
                raise RuntimeError("Installation failed for workload_d")

        existing_venvs = {
            'workload_a': '/path/to/venv_a',
            'workload_b': '/path/to/venv_b',
        }

        filtered_workloads = {
            'workload_c': {'path': '/tmp/c', 'setup': {}},
            'workload_d': {'path': '/tmp/d', 'setup': {}},
        }

        with patch.object(installer, '_save_installation_progress', side_effect=fake_save):
            with pytest.raises(RuntimeError, match="Installation failed for workload_d"):
                installer._perform_installation(
                    install_config,
                    filtered_workloads,
                    is_resume=True,
                    existing_completed_workloads=['workload_a', 'workload_b'],
                    existing_workload_venvs=existing_venvs,
                )

        mocks['create_cluster_config'].assert_called_once()
        call_args = mocks['create_cluster_config'].call_args
        assert sorted(call_args[0][2]) == ['workload_a', 'workload_b', 'workload_c']

        venvs_arg = call_args[0][7]
        assert venvs_arg['workload_a'] == '/path/to/venv_a'
        assert venvs_arg['workload_b'] == '/path/to/venv_b'
        assert venvs_arg['workload_c'] == '/path/to/venv_c'

    def test_resume_success_includes_all_workloads(self, temp_dirs, install_env):
        """Resume completes all remaining workloads; verify final config has ALL workloads."""
        mocks, installer = install_env
        install_config = _make_install_config(temp_dirs, ["workload_c", "workload_d"])

        mocks['group_workloads_by_dependencies'].return_value = {
            'hash_c': ['workload_c'],
            'hash_d': ['workload_d'],
        }

        def fake_save(ic, wk_keys, completed, venvs, ecc):
            completed.extend(wk_keys)
            for wk in wk_keys:
                venvs[wk] = f'/path/to/venv_{wk}'

        existing_venvs = {
            'workload_a': '/path/to/venv_a',
            'workload_b': '/path/to/venv_b',
        }

        filtered_workloads = {
            'workload_c': {'path': '/tmp/c', 'setup': {}},
            'workload_d': {'path': '/tmp/d', 'setup': {}},
        }

        with patch.object(installer, '_save_installation_progress', side_effect=fake_save):
            installer._perform_installation(
                install_config,
                filtered_workloads,
                is_resume=True,
                existing_completed_workloads=['workload_a', 'workload_b'],
                existing_workload_venvs=existing_venvs,
            )

        mocks['create_cluster_config'].assert_called_once()
        call_args = mocks['create_cluster_config'].call_args
        assert sorted(call_args[0][2]) == ['workload_a', 'workload_b', 'workload_c', 'workload_d']

        venvs_arg = call_args[0][7]
        assert venvs_arg['workload_a'] == '/path/to/venv_a'
        assert venvs_arg['workload_b'] == '/path/to/venv_b'
        assert venvs_arg['workload_c'] == '/path/to/venv_workload_c'
        assert venvs_arg['workload_d'] == '/path/to/venv_workload_d'


_PRIOR_COMPLETED = ['workload_a', 'workload_b']
_PRIOR_VENVS = {
    'workload_a': '/path/to/venv_a',
    'workload_b': '/path/to/venv_b',
}
_REMAINING_WORKLOADS = {
    'workload_c': {'path': '/tmp/c', 'setup': {}},
}


def _get_saved_venvs(mock_save_install_state, call_index):
    """Extract the workload_venvs dict from a save_install_state call."""
    return mock_save_install_state.call_args_list[call_index][0][2]


class TestResumeStatePersistsExistingVenvs:
    """Test that resumed installs preserve existing workload venvs in the state file.

    This covers the bug where _perform_installation would overwrite the state file
    with empty venvs at startup, causing previously completed workloads to lose
    their venv paths if the resumed install was interrupted again.
    """

    def _run_resumed_install(self, install_env, temp_dirs, progress_side_effect=None):
        """Run _perform_installation in resume mode with prior completed workloads.

        Args:
            progress_side_effect: Optional side_effect for _save_installation_progress.
                If None, uses the real implementation.
        """
        mocks, installer = install_env
        install_config = _make_install_config(temp_dirs, list(_REMAINING_WORKLOADS))

        mocks['group_workloads_by_dependencies'].return_value = {
            'hash_c': ['workload_c'],
        }

        ctx = (
            patch.object(installer, '_save_installation_progress', side_effect=progress_side_effect)
            if progress_side_effect
            else nullcontext()
        )
        with ctx:
            installer._perform_installation(
                install_config,
                _REMAINING_WORKLOADS,
                is_resume=True,
                existing_completed_workloads=list(_PRIOR_COMPLETED),
                existing_workload_venvs=dict(_PRIOR_VENVS),
            )

        return mocks

    def test_initial_state_save_includes_existing_venvs(self, temp_dirs, install_env):
        """The very first state save on resume must carry forward prior venvs."""

        def fake_progress(ic, wk_keys, completed, venvs, ecc):
            completed.extend(wk_keys)
            for wk in wk_keys:
                venvs[wk] = f'/path/to/venv_{wk}'

        mocks = self._run_resumed_install(install_env, temp_dirs, progress_side_effect=fake_progress)

        saved_venvs = _get_saved_venvs(mocks['save_install_state'], 0)
        assert saved_venvs == _PRIOR_VENVS

    def test_progress_save_includes_existing_and_new_venvs(self, temp_dirs, install_env):
        """After a dep group completes, state must contain both prior and new venvs."""
        mocks = self._run_resumed_install(install_env, temp_dirs)

        saved_venvs = _get_saved_venvs(mocks['save_install_state'], -1)
        for name, path in _PRIOR_VENVS.items():
            assert saved_venvs[name] == path, f"Prior venv for {name} missing from progress save"
        assert 'workload_c' in saved_venvs, "New workload_c venv missing from progress save"

    def test_failed_resume_preserves_existing_venvs_in_state(self, temp_dirs, install_env):
        """If a resumed install fails again, the state file must still have prior venvs."""
        mocks, installer = install_env

        def fail_on_progress(ic, wk_keys, completed, venvs, ecc):
            raise RuntimeError("workload_c failed again")

        with pytest.raises(RuntimeError, match="workload_c failed again"):
            self._run_resumed_install(install_env, temp_dirs, progress_side_effect=fail_on_progress)

        # Only the initial state save ran; it must have the prior venvs
        saved_venvs = _get_saved_venvs(mocks['save_install_state'], 0)
        assert saved_venvs == _PRIOR_VENVS

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Main LLMB installer orchestrator.

This module provides the main Installer class that coordinates all aspects of the
LLMB workload installation process, from configuration gathering to final installation.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llmb_install.cluster.gpu import filter_workloads_by_gpu_type, resolve_gpu_overrides
from llmb_install.config.cluster import create_cluster_config
from llmb_install.config.headless import (
    load_installation_config,
    save_installation_config,
)
from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.config.system import (
    clear_install_state,
    install_state_exists,
    load_install_state,
    load_system_config,
    save_install_state,
    save_system_config,
)
from llmb_install.constants import EXIT_CANCELLED, EXIT_ERROR
from llmb_install.core.dependency import (
    _resolve_dependencies,
    clone_git_repos,
    group_workloads_by_dependencies,
    install_dependencies,
    print_dependency_group_summary,
)
from llmb_install.core.exemplar import get_exemplar_workloads, validate_exemplar_workloads
from llmb_install.core.workload import (
    build_workload_dict,
    filter_tools_from_workload_list,
    run_setup_tasks,
)
from llmb_install.downloads.compute_uv import setup_compute_uv_binaries
from llmb_install.downloads.huggingface import download_huggingface_files_for_workloads
from llmb_install.downloads.image import fetch_container_images, get_required_images
from llmb_install.downloads.tools import fetch_and_install_tools, get_required_tools
from llmb_install.environment.cache import setup_cache_directories
from llmb_install.environment.venv_manager import (
    create_virtual_environment,
    get_venv_environment,
)
from llmb_install.ui.interface import UIInterface

# Import prompt functions from our new modular UI system
from llmb_install.ui.prompts.environment import (
    prompt_environment_type,
    prompt_environment_variables,
)
from llmb_install.ui.prompts.installation import (
    prompt_install_location,
    prompt_install_method,
)
from llmb_install.ui.prompts.workload import prompt_workload_selection
from llmb_install.ui.rich_ui import RichUI
from llmb_install.ui.simple import SimpleUI
from llmb_install.utils.filesystem import (
    check_repository_state,
    clean_repository_directory,
    copy_repository_working_files,
    create_llmb_run_symlink,
    find_llmb_repo_root,
)

# Import utility functions from our new modules
from llmb_install.utils.git import ensure_git_lfs_configured, is_git_lfs_installed
from llmb_install.utils.install_summary import write_install_summary_file
from llmb_install.utils.logging import get_logger, setup_logging


class _ResumeExecutionError(BaseException):
    """Carry a resume action error past resume-detection error handling.

    Subclasses BaseException (not Exception) on purpose: the resume action
    methods each catch ``except Exception`` to report failures, and some call
    one another (e.g. edit -> _complete_installation). Deriving from
    BaseException lets an already-wrapped error pass straight through those
    broad handlers instead of being caught and re-wrapped. Only the explicit
    ``except _ResumeExecutionError`` handler in _check_and_handle_resume
    catches it, where it is unwrapped to the original error and re-raised.
    Do not change this to Exception.
    """

    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error = error


class Installer:
    """Main installer class that orchestrates the LLMB installation process."""

    def __init__(self):
        """Initialize the installer."""
        self.root_dir: Optional[str] = None
        self.workloads: Dict[str, Dict[str, Any]] = {}
        self.force_fresh_repo_copy: bool = False  # Flag to force repository recopy
        self.logger = get_logger(__name__)  # Add logger for cleaner messaging

    def create_ui(self, ui_mode: str) -> UIInterface:
        """Create appropriate UI implementation based on the mode.

        Args:
            ui_mode: UI mode ('simple' or 'rich')

        Returns:
            UIInterface: Appropriate UI implementation

        Raises:
            ValueError: If ui_mode is not recognized
        """
        if ui_mode == 'simple':
            return SimpleUI()
        elif ui_mode == 'rich':
            return RichUI()
        else:
            raise ValueError(f"Unknown UI mode: {ui_mode}. Must be 'simple' or 'rich'.")

    def _check_if_running_from_install_dir(self) -> Optional[str]:
        """Check if running from an existing LLMB_INSTALL directory.

        Returns:
            Install path if running from LLMB_INSTALL, None otherwise
        """
        cwd = os.getcwd()
        cluster_config_path = Path(cwd) / "cluster_config.yaml"

        if cluster_config_path.exists():
            return cwd
        return None

    def _detect_incremental_install(self, install_path: str) -> Optional[Dict[str, Any]]:
        """Check if install_path is a valid existing installation.

        Args:
            install_path: Path to check for existing installation

        Returns:
            Cluster config dict if valid for incremental install, None otherwise
        """
        from llmb_install.config.cluster import load_cluster_config

        cluster_config = load_cluster_config(install_path)
        if not cluster_config:
            return None

        # Validate required fields
        required = ['gpu_type', 'workloads', 'slurm', 'install']
        if not all(k in cluster_config for k in required):
            return None

        # Check that workloads.installed is a list
        installed = cluster_config.get('workloads', {}).get('installed', [])
        if not isinstance(installed, list):
            return None

        return cluster_config

    def _handle_repository_setup(self, install_path: str, dev_mode: bool) -> str:
        """Handle repository copying and setup logic.

        Args:
            install_path: Installation directory path
            dev_mode: Whether in development mode

        Returns:
            str: Path to the repository to use (either original or copied)

        Side effects:
            Updates self.workloads to reflect the returned path (non-dev mode only).
        """
        if dev_mode:
            print(f"Development mode: Using repository at {self.root_dir}")
            return self.root_dir

        repo_copy_path = os.path.join(install_path, "llmb_repo")

        # Check repository state to determine what to do
        repo_state = check_repository_state(install_path)
        self.logger.debug(f"Repository state for {install_path}: {repo_state}")

        if repo_state == 'existing_install':
            print(f"\nError: Existing LLMB installation found at {install_path}")
            print("This directory contains a completed installation.")
            print("Please choose a different installation path, or remove the existing installation.")
            print("Future versions will support updating existing installations.")
            raise SystemExit(EXIT_ERROR)
        elif repo_state == 'orphaned':
            self.logger.debug(f"Removing orphaned directory: {repo_copy_path}")
            clean_repository_directory(repo_copy_path)
            repo_state = 'empty'  # Now it's empty

        # Determine if we need to copy
        need_copy = repo_state == 'empty' or self.force_fresh_repo_copy

        if need_copy:
            # If forcing fresh copy, remove existing directory first
            if self.force_fresh_repo_copy and repo_state != 'empty':
                print("Fresh installation: Removing existing repository copy...")
                clean_repository_directory(repo_copy_path)

            self.force_fresh_repo_copy = False  # Reset flag after use

            print("Copying repository to installation directory...")

            try:
                copy_repository_working_files(self.root_dir, repo_copy_path)
                self.logger.debug(f"Repository copied from {self.root_dir} to {repo_copy_path}")
            except Exception as e:
                print(f"Failed to copy repository: {e}")
                raise SystemExit(EXIT_ERROR) from e
        else:
            print(f"Using existing repository copy at {repo_copy_path}")

        self.workloads = build_workload_dict(repo_copy_path)
        return repo_copy_path

    def _complete_installation(
        self,
        config: InstallConfig,
        completed_workloads: List[str],
        workload_venvs: Dict[str, str],
        existing_cluster_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Complete installation by writing cluster config and clearing state.

        Args:
            config: Installation configuration
            completed_workloads: List of completed workload names
            workload_venvs: Mapping of workload names to their venv paths
            existing_cluster_config: For incremental installs, the original cluster config to merge with

        Returns:
            True to indicate installation completed successfully
        """
        try:
            # Create final config with only completed workloads
            final_config = InstallConfig(
                install_path=config.install_path,
                gpu_type=config.gpu_type,
                node_architecture=config.node_architecture,
                venv_type=config.venv_type,
                slurm=config.slurm,
                environment_vars=config.environment_vars,
                selected_workloads=completed_workloads,  # Only the completed ones
                workload_selection_mode=config.workload_selection_mode,
                install_method=config.install_method,
                ui_mode=config.ui_mode,
                dev_mode=config.dev_mode,
                llmb_repo=config.llmb_repo,
                image_folder=config.image_folder,
                cache_dirs_configured=config.cache_dirs_configured,
            )

            # Write cluster config

            # Create slurm_info dict in the expected format
            slurm_info = {}
            if final_config.slurm:
                slurm_info = {
                    'slurm': {
                        'account': final_config.slurm.account,
                        'gpu_partition': final_config.slurm.gpu_partition,
                        'cpu_partition': final_config.slurm.cpu_partition,
                        'gpu_partition_gres': final_config.slurm.gpu_partition_gres,
                        'cpu_partition_gres': final_config.slurm.cpu_partition_gres,
                    }
                }

            create_cluster_config(
                final_config.install_path,
                final_config.llmb_repo,
                completed_workloads,
                slurm_info,
                final_config.environment_vars,
                final_config.gpu_type,
                final_config.venv_type,
                workload_venvs,
                node_architecture=final_config.node_architecture,
                install_method=final_config.install_method,
                image_folder=getattr(final_config, 'image_folder', None),
                existing_cluster_config=existing_cluster_config,  # For incremental installs
            )
            print(f"✓ Configuration saved to {final_config.install_path}/cluster_config.yaml")

            # Clear install state
            clear_install_state()

            # Save system config for future use
            save_system_config(final_config)

            print("✓ Installation completed successfully!")
            return True

        except Exception as e:
            print(f"Error completing installation: {e}")
            print("Resume state preserved for retry.")
            raise _ResumeExecutionError(e) from e

    def _check_and_handle_resume(self, args: argparse.Namespace, state_result=None) -> bool:
        """Check for resumable installation state and handle resume if requested.

        Args:
            args: Parsed command line arguments
            state_result: Optional pre-loaded state from load_install_state() to avoid redundant loading

        Returns:
            True if resume handled everything, False to continue normal flow
        """
        # Wrap entire resume logic in comprehensive error handling to prevent hangs
        try:
            # Skip resume entirely if disabled via environment variable
            if os.environ.get('LLMB_DISABLE_RESUME', '').lower() in ('1', 'true', 'yes'):
                return False

            # Skip resume for headless mode (advanced users can handle failures manually)
            if hasattr(args, 'play') and args.play:
                return False

            # Skip resume for dev mode (always start fresh)
            dev_mode = getattr(args, 'dev_mode', False)
            if dev_mode:
                # Clear any existing resume state and explain why (only if state exists)
                if state_result:
                    try:
                        clear_install_state()
                        # Reset root_dir to current repository (was set to resume state's llmb_repo)
                        self.root_dir = find_llmb_repo_root()
                        # Reload workloads from the current repository
                        self.workloads = build_workload_dict(self.root_dir)
                        print("Development mode: Using current repository")
                        print("Found resumable state - clearing (dev mode always starts fresh)")
                        print()
                    except Exception as e:
                        print(f"Warning: Could not clear resume state in dev mode: {e}")
                        raise e
                return False

            # Use provided state or return early if no state
            if not state_result:
                return False

            # Extract state components
            try:
                config, completed_workloads, workload_venvs, existing_cluster_config = state_result
            except Exception as e:
                print(f"Warning: Could not parse resume state: {e}")
                print("Continuing with fresh installation...")
                return False

            # Calculate remaining workloads
            remaining_workloads = [w for w in config.selected_workloads if w not in completed_workloads]

            if not remaining_workloads:
                print("All workloads from previous installation are already completed!")
                clear_install_state()
                return True

            # Get timestamp from state (avoiding redundant file I/O)
            time_ago = "unknown time"
            try:
                # Get timestamp from already loaded state data (more efficient)
                from llmb_install.config.system import _install_state_manager

                state_path = _install_state_manager.get_path()
                if state_path.exists():
                    import yaml

                    with open(state_path, 'r') as f:
                        data = yaml.safe_load(f)
                        timestamp_str = data.get('timestamp', '')
                        if timestamp_str:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            # Calculate time difference for human-readable display
                            time_diff = datetime.now() - timestamp
                            if time_diff.days > 0:
                                time_ago = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
                            elif time_diff.seconds > 3600:
                                hours = time_diff.seconds // 3600
                                time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
                            elif time_diff.seconds > 60:
                                minutes = time_diff.seconds // 60
                                time_ago = f"{minutes} min ago"
                            else:
                                time_ago = "just now"
            except Exception as e:
                print(f"Warning: Could not parse timestamp: {e}")
                time_ago = "unknown time"

            # Display resume information using consolidated summary method
            # Detect if this is an incremental install resume
            is_incremental = config.is_incremental_install or (existing_cluster_config is not None)

            # Extract originally installed workloads for incremental installs
            original_installed_workloads = []
            if is_incremental and existing_cluster_config:
                original_installed_workloads = existing_cluster_config.get('workloads', {}).get('installed', [])

            if is_incremental:
                print(f"Found interrupted incremental installation ({time_ago})")
                if original_installed_workloads:
                    print(f"  Original installation: {', '.join(original_installed_workloads)}")
                print(f"  Attempted to add: {', '.join(config.selected_workloads)}")
            else:
                print(f"Found interrupted installation ({time_ago})")

            # Use consolidated summary display with resume-specific options
            # Disable "start fresh" for incremental installs (it breaks repository state)
            choice = self._show_configuration_summary(
                config,
                is_resume=True,
                completed_workloads=completed_workloads,
                remaining_workloads=remaining_workloads,
                allow_fresh_start=not is_incremental,  # No fresh start for incremental installs
                is_incremental_resume=is_incremental,  # New flag for incremental-specific messaging
                original_installed_workloads=original_installed_workloads,  # Show original workloads
            )

            if choice == 'yes':
                print("Resuming installation with remaining workloads...")
                print()
                return self._resume_installation(
                    config, remaining_workloads, completed_workloads, workload_venvs, existing_cluster_config
                )
            elif choice == 'no':
                print("Starting fresh installation...")
                clear_install_state()
                print()
                self.force_fresh_repo_copy = True  # Force repository recopy for fresh install
                self.root_dir = find_llmb_repo_root()  # Reset to original repo location
                self.workloads = build_workload_dict(self.root_dir)  # Reload workloads from original repo
                return False
            elif choice == 'clear':
                # Clear resume state and return to incremental install flow
                print("\nClearing resume state...")

                # If this is an incremental install and some workloads completed,
                # we need to update cluster_config before clearing state
                if existing_cluster_config and completed_workloads:
                    print(f"Preserving {len(completed_workloads)} completed workload(s)...")

                    # Reload install_state to get the most up-to-date workload_venvs
                    # (in case new workloads completed and venvs were updated)
                    from llmb_install.config.system import load_install_state

                    state_result = load_install_state()
                    if state_result:
                        _, latest_completed, latest_venvs, _ = state_result
                        # Use the latest data
                        completed_workloads = latest_completed
                        workload_venvs = latest_venvs

                    # Complete installation to save the completed workloads
                    self._complete_installation(config, completed_workloads, workload_venvs, existing_cluster_config)
                else:
                    # No completed workloads, just clear the state
                    clear_install_state()
                    print("Resume state cleared.")

                print()
                # For incremental installs, let the user start fresh with incremental install
                # This returns False to continue with normal flow, which will detect incremental install
                return False
            elif choice == 'edit':
                print("Switching to interactive mode with saved configuration as defaults...")
                print()
                return self._edit_resume_installation(
                    config, completed_workloads, workload_venvs, existing_cluster_config
                )

        except _ResumeExecutionError as e:
            # Installation actions must reach the central CLI error handler;
            # only failures while detecting or presenting resume state should
            # fall back to the normal installation flow below.
            raise e.error from None
        except Exception as e:
            print(f"Error in resume detection: {e}")
            print("Disabling resume functionality for this session...")
            print("(Resume state preserved - you can try again later)")
            print("(Set LLMB_DISABLE_RESUME=1 to permanently disable)")
            return False

    def _resume_installation(
        self,
        config: InstallConfig,
        remaining_workloads: List[str],
        completed_workloads: List[str],
        workload_venvs: Dict[str, str],
        existing_cluster_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resume installation with remaining workloads.

        Args:
            config: Original install configuration
            remaining_workloads: List of workloads still to be installed
            completed_workloads: List of workloads already completed successfully
            workload_venvs: Mapping of workload names to their venv paths
            existing_cluster_config: For incremental installs, the original cluster config

        Returns:
            True indicating resume handled everything
        """
        try:
            # Update config with remaining workloads
            config.selected_workloads = remaining_workloads

            # Note: self.root_dir already set from resume state in run() method
            # No need to set it again here

            # Re-run dependency grouping on remaining workloads for optimal installation plan
            filtered_workloads = filter_workloads_by_gpu_type(self.workloads, config.gpu_type)
            from llmb_install.core.workload import filter_tools_from_workload_list

            tools = filter_tools_from_workload_list(self.workloads)
            filtered_workloads.update(tools)
            filtered_workloads = resolve_gpu_overrides(filtered_workloads, config.gpu_type)

            # Build existing_workload_venvs for venv reuse
            # For incremental installs, combine original venvs with those from partial install
            combined_venvs = {}
            if existing_cluster_config:
                # Extract venvs from original installation
                original_venvs = existing_cluster_config.get('workloads', {}).get('config', {})
                for wl_name, wl_config in original_venvs.items():
                    if 'venv_path' in wl_config:
                        combined_venvs[wl_name] = wl_config['venv_path']
            # Add venvs from the partial incremental install (may overlap, workload_venvs takes precedence)
            combined_venvs.update(workload_venvs)

            # Perform installation with remaining workloads
            self._perform_installation(
                config,
                filtered_workloads,
                is_resume=True,
                existing_completed_workloads=completed_workloads,
                existing_workload_venvs=combined_venvs,  # Enable venv reuse
                existing_cluster_config=existing_cluster_config,  # Pass through for incremental installs
            )

            # Clear install state on successful completion
            clear_install_state()
            print("\n✓ Installation resumed and completed successfully!")
            return True

        except Exception as e:
            print(f"\nError during resume: {e}")
            print("You may need to start a fresh installation.")
            raise _ResumeExecutionError(e) from e

    def _edit_resume_installation(
        self,
        config: InstallConfig,
        completed_workloads: List[str],
        workload_venvs: Dict[str, str],
        existing_cluster_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Edit resume installation by dropping into interactive mode with defaults from resume state.

        Args:
            config: Original install configuration to use as defaults
            completed_workloads: List of workloads already completed successfully
            workload_venvs: Mapping of workload names to their venv paths
            existing_cluster_config: For incremental installs, the original cluster config

        Returns:
            True indicating edit handled everything
        """
        try:
            # Note: self.root_dir already set from resume state in run() method
            # Note: Don't clear install state here - only clear it when installation actually starts
            # This preserves resume state if user cancels out of editing

            # Create args object with the necessary attributes from the original config
            class EditArgs:
                def __init__(self, config):
                    self.dev_mode = config.dev_mode
                    self.image_folder = config.image_folder
                    self.ui_mode = config.ui_mode or 'simple'  # Default to simple if not set

            edit_args = EditArgs(config)

            # Collect limited configuration for resume edit mode
            # This only allows editing specific fields while locking others
            new_config = self._collect_limited_configuration(
                ui_mode=edit_args.ui_mode,
                resume_config=config,
                completed_workloads=completed_workloads,
                args=edit_args,
                existing_cluster_config=existing_cluster_config,
            )

            if not new_config.selected_workloads:
                if completed_workloads:
                    print("\nNo additional workloads selected.")
                    print(
                        f"Installation complete with {len(completed_workloads)} successful workload(s): {', '.join(completed_workloads)}"
                    )

                    # Reload install_state to get the most up-to-date workload_venvs
                    # This ensures we have venvs for all completed workloads
                    from llmb_install.config.system import load_install_state

                    state_result = load_install_state()
                    if state_result:
                        _, latest_completed, latest_venvs, _ = state_result
                        # Use the latest data to ensure all completed workloads have their venvs
                        completed_workloads = latest_completed
                        workload_venvs = latest_venvs

                    # Complete the installation
                    return self._complete_installation(
                        config, completed_workloads, workload_venvs, existing_cluster_config
                    )
                elif existing_cluster_config:
                    # Incremental install with no new workloads selected and none completed
                    # Just clear the install state - nothing to update
                    from llmb_install.config.system import clear_install_state

                    print("\nNo new workloads selected.")
                    print("No changes made to existing installation.")
                    clear_install_state()
                    return True
                else:
                    print("\nNo workloads selected. Exiting.")
                    raise SystemExit(EXIT_CANCELLED)

            # Calculate which workloads still need to be installed (preserve completed progress)
            workloads_to_install = [w for w in new_config.selected_workloads if w not in completed_workloads]

            if not workloads_to_install:
                print(f"\nAll selected workloads are already completed: {', '.join(new_config.selected_workloads)}")

                # Reload install_state to get the most up-to-date workload_venvs
                from llmb_install.config.system import load_install_state

                state_result = load_install_state()
                if state_result:
                    _, latest_completed, latest_venvs, _ = state_result
                    completed_workloads = latest_completed
                    workload_venvs = latest_venvs

                return self._complete_installation(
                    new_config, completed_workloads, workload_venvs, existing_cluster_config
                )
            elif len(workloads_to_install) < len(new_config.selected_workloads):
                already_completed = [w for w in new_config.selected_workloads if w in completed_workloads]
                print(f"\nSkipping already completed workloads: {', '.join(already_completed)}")
                print(f"Will install remaining workloads: {', '.join(workloads_to_install)}")

            # Update config to only install remaining workloads
            new_config.selected_workloads = workloads_to_install

            # Show final configuration summary and get confirmation (with locked field indicators)
            if not self._confirm_configuration(new_config, is_resume=True):
                print("\nReturning to edit mode...")
                # Recursively call edit mode again if user wants to make more changes
                return self._edit_resume_installation(
                    config, completed_workloads, workload_venvs, existing_cluster_config
                )

            # Prepare workloads and perform installation with new config
            filtered_workloads = self._prepare_workloads(new_config.gpu_type, workloads_to_install)

            # Build existing_workload_venvs for venv reuse
            # For incremental installs, combine original venvs with those from partial install
            combined_venvs = {}
            if existing_cluster_config:
                # Extract venvs from original installation
                original_venvs = existing_cluster_config.get('workloads', {}).get('config', {})
                for wl_name, wl_config in original_venvs.items():
                    if 'venv_path' in wl_config:
                        combined_venvs[wl_name] = wl_config['venv_path']
            # Add venvs from the partial incremental install (may overlap, workload_venvs takes precedence)
            combined_venvs.update(workload_venvs)

            # Perform installation with the newly edited configuration
            self._perform_installation(
                new_config,
                filtered_workloads,
                is_resume=True,  # This is a resume with edits - preserve state tracking
                existing_completed_workloads=completed_workloads,
                existing_workload_venvs=combined_venvs,  # Enable venv reuse
                existing_cluster_config=existing_cluster_config,  # Pass through for incremental installs
            )

            print("\n✓ Installation completed successfully!")
            return True

        except Exception as e:
            print(f"\nError during edit: {e}")
            print("You may need to start a fresh installation.")
            raise _ResumeExecutionError(e) from e

    def run(self, args: argparse.Namespace) -> None:
        """Run the installer with the given arguments.

        Args:
            args: Parsed command line arguments
        """
        # Set up logging based on verbose flag
        log_level = "DEBUG" if args.verbose else "INFO"
        setup_logging(level=log_level)

        # Check for Git LFS availability
        if not is_git_lfs_installed():
            print("\nError: Git LFS is not installed.")
            print("Some recipes require Git LFS for downloading large files.")
            print("\nPlease install Git LFS:")
            print("  - Ubuntu/Debian: sudo apt-get install git-lfs")
            print("  - RHEL/CentOS: sudo yum install git-lfs")
            print("\nFor more information, visit: https://git-lfs.com/")
            raise SystemExit(EXIT_ERROR) from None

        # Ensure Git LFS is configured
        try:
            ensure_git_lfs_configured()
        except Exception as e:
            print(f"\nError: Failed to configure Git LFS: {e}")
            print("Please run 'git lfs install' manually and try again.")
            raise SystemExit(EXIT_ERROR) from e

        # Check for resume state BEFORE loading workloads
        # This ensures we load from the correct repository location (llmb_repo copy vs original)
        state_result = load_install_state()
        resume_config = None  # Track resume config for error handling

        if state_result:
            config, _, _, _ = state_result  # Unpack with new existing_cluster_config field
            resume_config = config  # Store for later error handling
            # Verify install path exists (state validation)
            if os.path.exists(config.install_path):
                # Set root_dir from resume state BEFORE loading workloads
                if config.llmb_repo:
                    self.root_dir = config.llmb_repo
                    self.logger.debug(f"Resume detected: Loading workloads from {self.root_dir}")
                else:
                    # Fallback to original repo if llmb_repo not in state (shouldn't happen)
                    self.root_dir = find_llmb_repo_root()
            else:
                # Install path gone, clear stale state and use original repo
                clear_install_state()
                self.root_dir = find_llmb_repo_root()
                resume_config = None  # State cleared, no longer resuming
        else:
            # No resume state - check if running from existing installation directory
            install_path_from_cwd = self._check_if_running_from_install_dir()
            if install_path_from_cwd:
                cluster_config = self._detect_incremental_install(install_path_from_cwd)
                if cluster_config:
                    # Running from LLMB_INSTALL - validate llmb_repo from cluster config
                    llmb_repo = cluster_config.get('llmb_repo')
                    if not llmb_repo or not os.path.exists(llmb_repo):
                        print(f"\nError: Existing installation at {install_path_from_cwd} cannot be used.")
                        if llmb_repo:
                            print(f"The llmb_repo path does not exist: {llmb_repo}")
                        else:
                            print("The cluster_config.yaml is missing the llmb_repo path.")
                        print("\nThis installation cannot be extended with additional workloads.")
                        print("Please start a fresh installation from a repository checkout.")
                        raise SystemExit(EXIT_ERROR)

                    self.root_dir = llmb_repo
                    self.logger.debug(f"Incremental install detected: Loading workloads from {self.root_dir}")
                else:
                    # cluster_config.yaml exists but is invalid
                    self.root_dir = find_llmb_repo_root()
            else:
                # Not in installation directory, use original repo
                self.root_dir = find_llmb_repo_root()

        # Load workloads from the correct repository location
        self.workloads = build_workload_dict(self.root_dir)

        if not self.workloads:
            # Check if this is due to corrupted resume state
            if resume_config and resume_config.llmb_repo and self.root_dir == resume_config.llmb_repo:
                # Resume state exists but llmb_repo is corrupted/missing workloads
                print("\nError: Resume state detected, but no workloads found in:")
                print(f"  {self.root_dir}")
                print("\nThis may indicate:")
                print("  • The llmb_repo directory is corrupted or incomplete")
                print("  • Files were deleted from the installation directory")
                print("  • The installation directory is on unstable storage")
                print()

                # Offer recovery options using simple UI (we don't have config yet)
                ui = self.create_ui('simple')

                choice = ui.prompt_select(
                    "How would you like to proceed?",
                    choices=[
                        {'name': 'Clear resume state and start fresh installation', 'value': 'clear'},
                        {'name': 'Exit (investigate the issue)', 'value': 'exit'},
                    ],
                    default='clear',
                )

                if choice == 'clear':
                    print("\nClearing stale resume state...")
                    clear_install_state()
                    print("Resume state cleared. Please run llmb-install again to start fresh.")
                    raise SystemExit(EXIT_CANCELLED)
                else:
                    print("\nExiting. Resume state preserved at:")
                    print("  ~/.config/llmb/install_state.yaml")
                    print("\nTo manually clear state, run:")
                    print("  rm ~/.config/llmb/install_state.yaml")
                    raise SystemExit(EXIT_CANCELLED)
            else:
                # Fresh install but no workloads found in original repo
                print(f"\nError: No workloads found in {self.root_dir}")
                print("This repository may be missing workload metadata files.")
                raise SystemExit(EXIT_ERROR)

        # Check for resume before mode selection (single integration point)
        # Pass the already-loaded state to avoid redundant loading
        if self._check_and_handle_resume(args, state_result):
            return  # Resume handled everything

        # Handle different execution modes
        if args.play:
            self._run_headless_mode(args)
        elif args.record:
            self._run_record_mode(args)
        elif args.command == 'express':
            self._run_express_mode(args)
        else:
            self._run_interactive_mode(args)

    def _run_headless_mode(self, args: argparse.Namespace) -> None:
        """Run installer in headless mode using a configuration file.

        Args:
            args: Parsed command line arguments containing the config file path
        """
        print("=== HEADLESS INSTALLATION MODE ===")
        config_data = load_installation_config(args.play)

        # Create InstallConfig from the loaded data
        # model_validate handles all necessary conversions and defaults
        config = InstallConfig.model_validate(config_data)

        # Convert install_path to absolute path using pathlib
        # This is typically done in _collect_configuration but needed here for headless
        config.install_path = str(Path(config.install_path).expanduser().resolve())

        # Update config with dev_mode from args if present (InstallConfig default is False)
        config.dev_mode = getattr(args, 'dev_mode', False)

        print(f"Environment type: {config.venv_type}")
        print(f"Install path: {config.install_path}")
        print(f"GPU type: {config.gpu_type}")
        print(f"Node architecture: {config.node_architecture}")
        print(f"Install method: {config.install_method}")
        print(f"Selected workloads: {', '.join(config.selected_workloads)}")
        print()

        # Handle repository copying here (before workload preparation)
        self.root_dir = self._handle_repository_setup(config.install_path, config.dev_mode)
        config.llmb_repo = self.root_dir  # Ensure config has the actual repo path used

        # Prepare workloads and perform installation
        filtered_workloads = self._prepare_workloads(config.gpu_type, config.selected_workloads)

        # Pass config to _perform_installation
        self._perform_installation(
            config,
            filtered_workloads,
            is_resume=False,  # Fresh headless installation
        )

    def _run_record_mode(self, args: argparse.Namespace) -> None:
        """Run installer in record mode to save configuration without installing.

        Args:
            args: Parsed command line arguments containing the config file path
        """
        print("=== RECORD MODE ===")
        print("Collecting user inputs for configuration recording...")
        print()

        # Collect all configuration interactively
        config = self._collect_configuration(ui_mode=args.ui_mode, record_mode=True, args=args)

        # Save configuration and exit
        config_data = config.to_play_dict()

        save_installation_config(args.record, config_data)
        print("\nConfiguration recorded successfully!")
        print(f"To perform headless installation, run: llmb-install --play {args.record}")

    def _run_express_mode(self, args: argparse.Namespace) -> None:
        """Run installer in express mode using saved system configuration.

        Args:
            args: Parsed command line arguments for express command
        """
        # Handle --list-workloads first
        if getattr(args, 'list_workloads', False):
            self._list_available_workloads()
            return

        # Validate that system config exists
        from llmb_install.config.system import (
            get_system_config_path,
            system_config_exists,
        )

        if not system_config_exists():
            print("Error: Express mode requires a saved system configuration.")
            print(f"Expected config file: {get_system_config_path()}")
            print("\nTo create a system config, complete a successful interactive installation first:")
            print("  llmb-install")
            print("\nThis will save your system settings (GPU type, SLURM config, etc.) for future express installs.")
            print("\nAlternatively, use '--play <config.yaml>' for full headless installation.")
            raise SystemExit(EXIT_ERROR)

        # Collect configuration using express mode with confirmation loop
        while True:
            config = self._collect_express_configuration(args)

            if not config.selected_workloads:
                print("\nNo workloads selected. Exiting.")
                raise SystemExit(EXIT_CANCELLED)

            # Show configuration summary and get confirmation
            if self._confirm_configuration(config):
                break  # User confirmed, proceed with installation
            else:
                print("\nReturning to configuration...")
                # Express mode doesn't support re-configuration, so this shouldn't happen
                print("Configuration changes not supported in express mode.")
                raise SystemExit(EXIT_CANCELLED)

        # Update config.image_folder from args if provided, overriding system config
        if getattr(args, 'image_folder', None) is not None:
            config.image_folder = args.image_folder

        # Prepare workloads and perform installation
        filtered_workloads = self._prepare_workloads(config.gpu_type, config.selected_workloads)

        self._perform_installation(
            config,
            filtered_workloads,
            is_resume=False,  # Fresh express installation
        )

    def _run_interactive_mode(self, args: argparse.Namespace) -> None:
        """Run installer in interactive mode with user prompts.

        Args:
            args: Parsed command line arguments
        """
        print()  # header spacing

        # NEW: Check if running from existing LLMB_INSTALL (entrance point #1)
        install_path_from_cwd = self._check_if_running_from_install_dir()
        if install_path_from_cwd:
            cluster_config = self._detect_incremental_install(install_path_from_cwd)
            if cluster_config:
                # Note: llmb_repo validation already happened in run() method,
                # but we validate again here for consistency and defensive programming
                llmb_repo = cluster_config.get('llmb_repo')
                if not llmb_repo or not os.path.exists(llmb_repo):
                    # This should have been caught in run(), but handle it anyway
                    print(f"\nError: Installation at {install_path_from_cwd} cannot be used.")
                    print("Cannot proceed with incremental install.")
                    raise SystemExit(EXIT_ERROR)

                # Offer incremental install
                ui = self.create_ui(args.ui_mode)
                installed_workloads = cluster_config.get('workloads', {}).get('installed', [])
                gpu_type = cluster_config.get('gpu_type', 'unknown')

                print(f"Found existing installation at {install_path_from_cwd} ({gpu_type.upper()})")
                print(f"  Currently installed: {', '.join(installed_workloads)}")
                print()

                choice = ui.prompt_select(
                    "What would you like to do?",
                    choices=[
                        {'name': 'Add workloads to this installation', 'value': 'incremental'},
                        {'name': 'Exit', 'value': 'exit'},
                    ],
                    default='incremental',
                )

                # Handle cancellation (Ctrl-C returns None)
                if choice is None:
                    print("\n\nInstallation cancelled by user.")
                    raise SystemExit(EXIT_CANCELLED)

                if choice == 'exit':
                    print("\nTo start a new installation, run llmb-install from a repository checkout.")
                    raise SystemExit(EXIT_CANCELLED)

                # Proceed with incremental install
                config, existing_venvs = self._collect_incremental_configuration(
                    args.ui_mode, install_path_from_cwd, cluster_config, args
                )

                if not config:
                    # No workloads available or user cancelled
                    print("\nNo workloads selected or none available. Exiting.")
                    return

                # Prepare workloads for installation
                filtered_workloads = self._prepare_workloads(config.gpu_type, config.selected_workloads)

                # Perform incremental installation
                self._perform_installation(
                    config,
                    filtered_workloads,
                    is_resume=False,
                    existing_workload_venvs=existing_venvs,
                    existing_cluster_config=cluster_config,
                )
                return

        # Collect configuration interactively with confirmation loop
        override_defaults = None  # Start with no overrides
        while True:
            config = self._collect_configuration(
                ui_mode=args.ui_mode,
                record_mode=False,
                args=args,
                override_defaults=override_defaults,  # Use previous config as defaults if editing
            )

            if not config.selected_workloads:
                print("\nNo workloads selected. Exiting.")
                raise SystemExit(EXIT_CANCELLED)

            # Check if this is an incremental install (from entrance point #2)
            is_incremental = getattr(config, '_is_incremental', False)
            if is_incremental:
                # Incremental install - skip confirmation, proceed directly
                existing_venvs = getattr(config, '_existing_venvs', {})
                existing_cluster_config = getattr(config, '_existing_cluster_config', None)

                filtered_workloads = self._prepare_workloads(config.gpu_type, config.selected_workloads)

                # Perform incremental installation
                self._perform_installation(
                    config,
                    filtered_workloads,
                    is_resume=False,
                    existing_workload_venvs=existing_venvs,
                    existing_cluster_config=existing_cluster_config,
                )
                return

            # Show configuration summary and get confirmation
            if self._confirm_configuration(config):
                break  # User confirmed, proceed with installation
            else:
                print("\nReturning to configuration...")
                # Use current config as defaults for next iteration
                override_defaults = config

        # Prepare workloads and perform installation
        filtered_workloads = self._prepare_workloads(config.gpu_type, config.selected_workloads)

        self._perform_installation(
            config,
            filtered_workloads,
            is_resume=False,  # Fresh interactive installation
        )

    def _collect_configuration(
        self,
        ui_mode: str = 'simple',
        record_mode: bool = False,
        args: Optional[argparse.Namespace] = None,
        override_defaults: Optional[InstallConfig] = None,
    ) -> InstallConfig:
        """Collect installation configuration from user prompts.

        Args:
            ui_mode: UI mode to use ('simple', 'rich', 'express')
            record_mode: Whether running in record mode
            args: Command line arguments
            override_defaults: Optional config to use as defaults instead of system config

        Returns:
            InstallConfig: Complete installation configuration
        """
        # Create UI instance for all prompts
        ui = self.create_ui(ui_mode)

        # Load actual system config first for stable defaults
        system_config = load_system_config()
        if system_config:
            ui.log("Found saved system configuration - using as defaults where applicable")

        # Create merged defaults dict - system config first, then resume overrides
        defaults = {}
        if system_config:
            defaults.update(system_config.model_dump())
        if override_defaults:
            defaults.update(override_defaults.model_dump())
            ui.log("Using saved installation configuration as additional defaults")  # Not wild about this language.

        # Collect basic configuration using merged defaults. Fresh installs always
        # create recipe environments with uv; legacy venv_type values are kept
        # only for resume, incremental, and headless compatibility paths.
        venv_type = prompt_environment_type(ui, defaults.get('venv_type'))
        print()

        # Loop until we get a valid install path (not an existing installation, or user accepts incremental)
        while True:
            install_path = prompt_install_location(ui, default=defaults.get('install_path'))
            if not install_path:
                print("\nInstallation cancelled.")
                raise SystemExit(EXIT_CANCELLED)

            # NEW: Check for existing installation (entrance point #2)
            cluster_config = self._detect_incremental_install(install_path)
            if cluster_config:
                # Validate llmb_repo path before offering incremental install
                llmb_repo = cluster_config.get('llmb_repo')
                if not llmb_repo or not os.path.exists(llmb_repo):
                    print(f"\nError: Existing installation at {install_path} cannot be used.")
                    if llmb_repo:
                        print(f"The llmb_repo path does not exist: {llmb_repo}")
                    else:
                        print("The cluster_config.yaml is missing the llmb_repo path.")
                    print("\nThis installation cannot be extended with additional workloads.")
                    print("Please choose a different installation path.")
                    continue  # Re-prompt for path

                installed_workloads = cluster_config.get('workloads', {}).get('installed', [])
                gpu_type = cluster_config.get('gpu_type', 'unknown')

                print(f"\nExisting installation detected at {install_path} ({gpu_type.upper()})")
                print(f"  Currently installed: {', '.join(installed_workloads)}")
                print()

                choice = ui.prompt_select(
                    "What would you like to do?",
                    choices=[
                        {'name': 'Add workloads to existing installation', 'value': 'incremental'},
                        {'name': 'Choose different installation path', 'value': 'different'},
                    ],
                    default='incremental',
                )

                # Handle cancellation (Ctrl-C returns None)
                if choice is None:
                    print("\n\nInstallation cancelled by user.")
                    raise SystemExit(EXIT_CANCELLED)

                if choice == 'incremental':
                    # Reload workloads from the correct llmb_repo for this installation
                    # (Important for entrance point #2 - interactive path prompt)
                    llmb_repo = cluster_config.get('llmb_repo')
                    if llmb_repo and llmb_repo != self.root_dir:
                        self.logger.debug(f"Reloading workloads from installation's llmb_repo: {llmb_repo}")
                        self.root_dir = llmb_repo
                        self.workloads = build_workload_dict(self.root_dir)
                        if not self.workloads:
                            print(f"\nError: Could not load workloads from {self.root_dir}")
                            print("The installation's repository may be corrupted.")
                            continue  # Re-prompt for path

                    # Return a special marker that indicates incremental install
                    # The caller (_run_interactive_mode) will handle the incremental flow
                    # For now, we need to refactor this slightly differently
                    # Since _collect_configuration is called from a loop in _run_interactive_mode,
                    # we'll handle incremental install directly here and raise a custom exception
                    # to break out of the config loop

                    config, existing_venvs = self._collect_incremental_configuration(
                        ui_mode, install_path, cluster_config, args
                    )

                    if not config:
                        # No workloads available or user cancelled - reprompt for path
                        continue

                    # Store incremental install info in the config for the caller to use
                    config._is_incremental = True
                    config._existing_venvs = existing_venvs
                    config._existing_cluster_config = cluster_config
                    return config
                else:
                    # Loop back to prompt for new path
                    continue
            else:
                # No existing installation, proceed with normal flow
                break

        # Handle repository copying here (before other prompts)
        dev_mode = getattr(args, 'dev_mode', False) if args else False

        # Handle image folder - CLI arg takes precedence, fallback to saved config
        image_folder_from_args = getattr(args, 'image_folder', None) if args else None
        image_folder = image_folder_from_args or defaults.get('image_folder')
        if image_folder and not image_folder_from_args:
            ui.log(f"Using saved image folder: {image_folder}")

        # Skip repository setup if in record mode (no filesystem modifications needed)
        if record_mode:
            self.logger.debug("Record mode: Skipping repository setup and cache configuration")
        elif override_defaults and override_defaults.llmb_repo and override_defaults.install_path == install_path:
            # Reuse existing repository location from previous configuration
            self.root_dir = override_defaults.llmb_repo
            self.logger.debug(f"Edit mode: Reusing existing repository at {self.root_dir}")
        else:
            # First time configuration or install path changed - perform repository setup
            self.root_dir = self._handle_repository_setup(install_path, dev_mode)

        # Import prompt functions for SLURM and GPU configuration
        from llmb_install.ui.prompts.gpu import prompt_gpu_type as new_prompt_gpu_type
        from llmb_install.ui.prompts.gpu import (
            prompt_node_architecture as new_prompt_node_architecture,
        )
        from llmb_install.ui.prompts.slurm import (
            prompt_slurm_info as new_prompt_slurm_info,
        )

        # Prepare SLURM defaults
        slurm_defaults = None
        slurm_config_data = defaults.get('slurm')
        if slurm_config_data:
            slurm_defaults = {
                'account': slurm_config_data['account'],
                'gpu_partition': slurm_config_data['gpu_partition'],
                'cpu_partition': slurm_config_data['cpu_partition'],
            }

        slurm_config = new_prompt_slurm_info(ui, slurm_defaults, express_mode=False)
        if not slurm_config:
            print("\nInstallation cancelled.")
            raise SystemExit(EXIT_CANCELLED)

        gpu_type = new_prompt_gpu_type(ui, self.workloads, defaults.get('gpu_type'), express_mode=False)

        node_architecture = new_prompt_node_architecture(
            ui,
            gpu_type,
            defaults.get('node_architecture'),
            express_mode=False,
            gpu_partition=slurm_config['slurm']['gpu_partition'],
        )
        print()

        # Filter workloads by GPU type
        filtered_workloads = filter_workloads_by_gpu_type(self.workloads, gpu_type)
        if not filtered_workloads:
            print(f"No workloads found that support {gpu_type} GPU type.")
            raise SystemExit(EXIT_ERROR)

        # Add tools to workload list
        tools = filter_tools_from_workload_list(self.workloads)
        filtered_workloads.update(tools)

        # Resolve GPU-specific images/repos
        filtered_workloads = resolve_gpu_overrides(filtered_workloads, gpu_type)

        install_method = prompt_install_method(ui, defaults.get('install_method'), express_mode=False)
        if install_method is None:
            print("\nInstallation cancelled.")
            raise SystemExit(EXIT_CANCELLED)
        print()

        # In dev mode, skip exemplar/custom question and go straight to workload selection
        selected, selection_mode = prompt_workload_selection(
            ui,
            filtered_workloads,
            show_exemplar_option=not dev_mode,
            default_mode=defaults.get('workload_selection_mode', 'custom'),
            llmb_repo=self.root_dir,
            gpu_type=gpu_type,
        )
        if not selected:
            print("\nInstallation cancelled.")
            raise SystemExit(EXIT_CANCELLED)

        env_vars = prompt_environment_variables(ui, defaults.get('environment_vars'), express_mode=False)
        if env_vars is None:
            print("\nInstallation cancelled.")
            raise SystemExit(EXIT_CANCELLED)
        print()

        # Create configuration object
        slurm_obj = SlurmConfig(
            account=slurm_config['slurm']['account'],
            gpu_partition=slurm_config['slurm']['gpu_partition'],
            cpu_partition=slurm_config['slurm']['cpu_partition'],
            gpu_partition_gres=slurm_config['slurm'].get('gpu_partition_gres'),
            cpu_partition_gres=slurm_config['slurm'].get('cpu_partition_gres'),
        )

        config = InstallConfig(
            install_path=install_path,
            gpu_type=gpu_type,
            node_architecture=node_architecture,
            venv_type=venv_type,
            slurm=slurm_obj,
            environment_vars=env_vars,
            selected_workloads=selected,
            workload_selection_mode=selection_mode,
            install_method=install_method,
            ui_mode=ui_mode,
            dev_mode=dev_mode,
            llmb_repo=self.root_dir,
            image_folder=image_folder,
        )

        return config

    def _collect_limited_configuration(
        self,
        ui_mode: str,
        resume_config: InstallConfig,
        completed_workloads: List[str],
        args: Optional[argparse.Namespace] = None,
        existing_cluster_config: Optional[Dict[str, Any]] = None,
    ) -> InstallConfig:
        """Collect limited configuration for resume edit mode.

        Only allows editing specific fields while locking others to prevent edge cases.
        This is a restricted version of _collect_configuration for resume scenarios.

        Args:
            ui_mode: UI mode to use ('simple', 'rich')
            resume_config: Original install configuration (locked fields come from here)
            completed_workloads: List of workloads already completed successfully
            args: Command line arguments
            existing_cluster_config: For incremental installs, the original cluster config

        Returns:
            InstallConfig: Updated configuration with only allowed changes
        """
        # Import SlurmConfig for early return case
        from llmb_install.config.models import SlurmConfig

        # Create UI instance for all prompts
        ui = self.create_ui(ui_mode)

        # 1. Virtual Environment Type - LOCKED, use from resume config
        venv_type = resume_config.venv_type
        print()

        # 2. SLURM configuration - allow changes
        from llmb_install.ui.prompts.slurm import (
            prompt_slurm_info as new_prompt_slurm_info,
        )

        # Prepare SLURM defaults from resume config
        slurm_defaults = None
        if resume_config.slurm:
            slurm_defaults = {
                'account': resume_config.slurm.account,
                'gpu_partition': resume_config.slurm.gpu_partition,
                'cpu_partition': resume_config.slurm.cpu_partition,
            }

        slurm_config = new_prompt_slurm_info(ui, slurm_defaults, express_mode=False)
        if not slurm_config:
            print("\nConfiguration cancelled.")
            raise SystemExit(EXIT_CANCELLED)

        # 3. Install method - allow changes if system supports both
        install_method = prompt_install_method(ui, resume_config.install_method, express_mode=False)
        if install_method is None:
            print("\nInstallation cancelled.")
            raise SystemExit(EXIT_CANCELLED)
        print()

        # 4. Workload selection - restricted to original selection minus completed
        # Calculate remaining workloads that can be selected
        remaining_workloads = [w for w in resume_config.selected_workloads if w not in completed_workloads]

        if not remaining_workloads:
            ui.log("All originally selected workloads have been completed!")
            selected = []
        else:
            # Prepare workloads dict for selection (filtering by original GPU type)
            filtered_workloads = filter_workloads_by_gpu_type(self.workloads, resume_config.gpu_type)
            tools = filter_tools_from_workload_list(self.workloads)
            filtered_workloads.update(tools)
            filtered_workloads = resolve_gpu_overrides(filtered_workloads, resume_config.gpu_type)

            # Filter to only show workloads from original selection that aren't completed
            available_workloads = {k: v for k, v in filtered_workloads.items() if k in remaining_workloads}

            print("Workload Status:")
            print(f"  Original selection: {', '.join(resume_config.selected_workloads)}")
            if completed_workloads:
                # Show completed as a nice list
                print(f"  Completed: {', '.join(completed_workloads)}")
            print()
            print("Remaining workloads (pre-selected for installation):")
            print("Uncheck any workloads you want to skip:")

            # Prompt for workload selection from remaining workloads
            # Remove "Install All" option and pre-select remaining workloads
            # Allow empty selection if some workloads already completed OR if this is an incremental install
            # (incremental installs have original workloads, so it's okay to select no new ones)
            is_incremental = existing_cluster_config is not None
            selected, _ = prompt_workload_selection(
                ui,
                available_workloads,
                default_selected=remaining_workloads,  # Pre-select all remaining
                show_install_all=False,  # Remove "Install All" option for resume mode
                allow_empty=bool(completed_workloads) or is_incremental,  # Allow empty if completed or incremental
            )

            if selected is None:  # User cancelled
                print("\nConfiguration cancelled.")
                raise SystemExit(EXIT_CANCELLED)

            # Check if user selected no workloads for incremental install (no new workloads to add)
            # In this case, skip remaining prompts and return early with empty selection
            if is_incremental and not selected and not completed_workloads:
                # Return early with empty workload selection - caller will handle cleanup
                return InstallConfig(
                    install_path=resume_config.install_path,
                    gpu_type=resume_config.gpu_type,
                    node_architecture=resume_config.node_architecture,
                    llmb_repo=resume_config.llmb_repo,
                    dev_mode=resume_config.dev_mode,
                    image_folder=resume_config.image_folder,
                    venv_type=venv_type,
                    slurm=SlurmConfig(
                        account=slurm_config['slurm']['account'],
                        gpu_partition=slurm_config['slurm']['gpu_partition'],
                        cpu_partition=slurm_config['slurm']['cpu_partition'],
                        gpu_partition_gres=slurm_config['slurm'].get('gpu_partition_gres'),
                        cpu_partition_gres=slurm_config['slurm'].get('cpu_partition_gres'),
                    ),
                    install_method=install_method,
                    selected_workloads=[],
                    environment_vars=resume_config.environment_vars,  # Keep existing
                    ui_mode=ui_mode,
                    cache_dirs_configured=resume_config.cache_dirs_configured,
                    is_incremental_install=resume_config.is_incremental_install,
                )

        # 5. Environment variables - allow updating
        from llmb_install.ui.prompts.environment import prompt_environment_variables

        # Re-prompt for environment variables, passing existing as defaults
        # This allows users to update HF_TOKEN or other variables
        existing_env_vars = resume_config.environment_vars.copy()
        new_env_vars = prompt_environment_variables(ui, defaults=existing_env_vars, express_mode=False)

        if new_env_vars is None:  # User cancelled
            print("\nConfiguration cancelled.")
            raise SystemExit(EXIT_CANCELLED)

        # Merge new env vars
        env_vars = new_env_vars

        # Create SLURM config object
        slurm_obj = SlurmConfig(
            account=slurm_config['slurm']['account'],
            gpu_partition=slurm_config['slurm']['gpu_partition'],
            cpu_partition=slurm_config['slurm']['cpu_partition'],
            gpu_partition_gres=slurm_config['slurm'].get('gpu_partition_gres'),
            cpu_partition_gres=slurm_config['slurm'].get('cpu_partition_gres'),
        )

        # Create updated configuration with locked and modified fields
        config = InstallConfig(
            # LOCKED FIELDS - use original values
            install_path=resume_config.install_path,
            gpu_type=resume_config.gpu_type,
            node_architecture=resume_config.node_architecture,
            llmb_repo=resume_config.llmb_repo,
            dev_mode=resume_config.dev_mode,
            image_folder=resume_config.image_folder,
            # EDITABLE FIELDS - use new values
            venv_type=venv_type,
            slurm=slurm_obj,
            install_method=install_method,
            selected_workloads=selected,
            # PRESERVED FIELDS - keep original
            environment_vars=env_vars,
            ui_mode=ui_mode,
            cache_dirs_configured=resume_config.cache_dirs_configured,
        )

        return config

    def _show_configuration_summary(
        self,
        config: InstallConfig,
        is_resume: bool = False,
        completed_workloads: Optional[List[str]] = None,
        remaining_workloads: Optional[List[str]] = None,
        allow_fresh_start: bool = False,
        is_incremental_resume: bool = False,
        original_installed_workloads: Optional[List[str]] = None,
    ) -> str:
        """Display installation configuration summary.

        Args:
            config: Installation configuration to summarize
            is_resume: Whether this is a resume scenario (shows locked fields)
            completed_workloads: List of completed workloads (for resume flow)
            remaining_workloads: List of remaining workloads (for resume flow)
            allow_fresh_start: Whether to show "start fresh" option (for resume flow)
            is_incremental_resume: Whether this is resuming an incremental install
            original_installed_workloads: Original workloads before incremental install (if incremental)

        Returns:
            User choice: 'yes' to continue, 'edit' to modify configuration, 'no' for fresh start, 'clear' to clear resume state
        """
        print("\nInstallation Summary")
        print("=" * 50)

        # Show locked fields with indicator if in resume mode (🔒 at start for visibility)
        locked_prefix = "🔒 " if is_resume else ""
        print(f"{locked_prefix}Install Path: {config.install_path}")
        print(f"{locked_prefix}GPU Type: {config.gpu_type}")
        print(f"{locked_prefix}Node Architecture: {config.node_architecture}")
        print(f"{locked_prefix}Virtual Environment: {config.venv_type}")
        print(f"Install Mode: {config.install_method}")

        if config.slurm:
            print(f"SLURM Account: {config.slurm.account}")
            print(f"GPU Partition: {config.slurm.gpu_partition}")
            print(f"CPU Partition: {config.slurm.cpu_partition}")

        if config.llmb_repo and is_resume:
            print(f"🔒 Repository: {config.llmb_repo}")

        # Show workload information (differs between resume, incremental resume, and normal flow)
        if is_incremental_resume and original_installed_workloads:
            # Incremental resume: show original, attempted to add, completed, and remaining
            print(f"🔒 Original Installation: {', '.join(original_installed_workloads)}")
            print(f"Attempted to Add: {', '.join(config.selected_workloads)}")
            if completed_workloads:
                print(f"✓ Successfully Added: {', '.join(completed_workloads)}")
            if remaining_workloads:
                print(f"Remaining to Install: {', '.join(remaining_workloads)}")
        elif completed_workloads is not None or remaining_workloads is not None:
            # Regular resume flow: show original, completed, and remaining
            print(f"Selected Workloads: {', '.join(config.selected_workloads)}")
            if completed_workloads:
                print(f"Completed Workloads: {', '.join(completed_workloads)}")
            else:
                print("Completed Workloads: None")
            if remaining_workloads:
                print(f"Remaining Workloads: {', '.join(remaining_workloads)}")
        else:
            # Normal flow: just show selected
            if config.selected_workloads:
                print(f"Selected Workloads: {', '.join(config.selected_workloads)}")
            else:
                print("Selected Workloads: None")

        if config.environment_vars:
            print(f"Environment Variables: {len(config.environment_vars)} configured")

        if config.image_folder:
            print(f"Container Image Folder: {config.image_folder}")

        if config.dev_mode:
            print(f"Development Mode: Enabled (using {config.llmb_repo})")

        print()

        # Add note about locked fields if in resume mode
        if is_resume:
            if is_incremental_resume:
                print("Note: This is resuming an incremental installation.")
                print("      Original workloads will be preserved upon completion.")
            print("      Fields marked with 🔒 cannot be changed during resume.")
            print()

        # Create UI instance for prompting
        ui = self.create_ui(config.ui_mode)

        # Build prompt choices based on context
        if is_incremental_resume:
            # Special options for incremental resume
            prompt_message = "Continue adding workloads to existing installation?"
            choices = [
                {'name': 'Yes, resume incremental installation', 'value': 'yes'},
                {'name': 'Discard pending workloads and start over', 'value': 'clear'},
                {'name': 'Edit configuration', 'value': 'edit'},
            ]
        elif allow_fresh_start:
            prompt_message = "Continue with remaining workloads?"
            choices = [
                {'name': 'Yes, resume installation', 'value': 'yes'},
                {'name': 'No, start fresh installation', 'value': 'no'},
                {'name': 'Edit configuration', 'value': 'edit'},
            ]
        else:
            prompt_message = "Continue with installation?"
            choices = [
                {'name': 'Yes, start installation', 'value': 'yes'},
                {'name': 'Edit configuration', 'value': 'edit'},
            ]

        while True:
            choice = ui.prompt_select(prompt_message, choices=choices, default='yes')

            if choice in ['yes', 'edit', 'no', 'clear']:
                return choice
            elif choice is None:
                # User cancelled (Ctrl+C), exit installer
                print("\nInstallation cancelled by user.")
                raise SystemExit(EXIT_CANCELLED)
            else:
                print("Please select a valid option.")

    def _confirm_configuration(self, config: InstallConfig, is_resume: bool = False) -> bool:
        """Show configuration summary and get user confirmation.

        Args:
            config: Installation configuration to confirm
            is_resume: Whether this is a resume scenario (shows locked fields)

        Returns:
            True if user wants to continue, False if they want to edit
        """
        choice = self._show_configuration_summary(config, is_resume=is_resume)
        return choice == 'yes'

    def _collect_incremental_configuration(
        self,
        ui_mode: str,
        install_path: str,
        cluster_config: Dict[str, Any],
        args: Optional[argparse.Namespace] = None,
    ) -> tuple[Optional[InstallConfig], Dict[str, str]]:
        """Collect configuration for incremental install - workload selection only.

        All settings are locked from the existing cluster_config. Only prompts for
        workload selection, filtering out already installed workloads.

        Args:
            ui_mode: UI mode to use ('simple', 'rich')
            install_path: Path to the existing installation
            cluster_config: Loaded cluster configuration dict
            args: Optional command line arguments (for image_folder override)

        Returns:
            Tuple of (InstallConfig, existing_workload_venvs) or (None, {}) if cancelled
        """
        # Extract existing configuration (ALL LOCKED)
        installed_workloads = cluster_config['workloads']['installed']
        existing_venvs = cluster_config['workloads'].get('config', {})

        # Build workload_venvs map
        existing_workload_venvs = {}
        for wl_name, wl_config in existing_venvs.items():
            if 'venv_path' in wl_config:
                existing_workload_venvs[wl_name] = wl_config['venv_path']

        # Extract locked configuration from cluster_config
        gpu_type = cluster_config['gpu_type']
        llmb_repo = cluster_config.get('llmb_repo')

        # Extract install-specific metadata (new in Phase 1)
        install_metadata = cluster_config.get('install', {})
        install_method = install_metadata.get('method', 'local')

        # Extract venv_type with backward compatibility fallback
        from llmb_install.environment.venv_utils import extract_venv_type_from_config

        venv_type = extract_venv_type_from_config(cluster_config)
        if venv_type:
            self.logger.debug(f"Extracted venv_type '{venv_type}' from cluster config")

        # If not found, error out - we need to know the venv type
        if not venv_type:
            print(f"\nError: Cannot determine virtual environment type for installation at {install_path}")
            print("The cluster_config.yaml is missing venv_type information.")
            print("\nThis installation appears to be corrupted or from an incompatible version.")
            print("Please start a fresh installation from a repository checkout.")
            raise SystemExit(EXIT_ERROR)

        # Get node_architecture from install section (canonical location)
        node_architecture = install_metadata.get('node_architecture')
        if not node_architecture:
            print(f"\nError: Cannot determine node architecture for installation at {install_path}")
            print("The cluster_config.yaml is missing node_architecture information.")
            raise SystemExit(EXIT_ERROR)

        # Get image_folder - from install section or -i flag override
        image_folder_from_config = install_metadata.get('image_folder')
        image_folder_from_flag = getattr(args, 'image_folder', None) if args else None
        image_folder = image_folder_from_flag or image_folder_from_config  # Flag takes precedence

        # Extract SLURM, env_vars, etc. - ALL LOCKED, just reuse
        slurm_config_dict = cluster_config.get('slurm', {})
        env_vars = cluster_config.get('environment', {})

        ui = self.create_ui(ui_mode)

        # Display configuration summary (all fields locked except workloads)
        print("\n" + "=" * 70)
        print("INCREMENTAL INSTALL - Configuration Summary")
        print("=" * 70)
        print(f"🔒 Install Path: {install_path}")
        print(f"🔒 GPU Type: {gpu_type}")
        print(f"🔒 Node Architecture: {node_architecture}")
        print(f"🔒 Virtual Environment: {venv_type}")
        if slurm_config_dict:
            print(f"🔒 SLURM Account: {slurm_config_dict.get('account', 'N/A')}")
            print(f"🔒 GPU Partition: {slurm_config_dict.get('gpu', {}).get('partition', 'N/A')}")
            print(f"🔒 CPU Partition: {slurm_config_dict.get('cpu', {}).get('partition', 'N/A')}")
        if image_folder:
            flag_indicator = " (from -i flag)" if image_folder_from_flag else ""
            print(f"🔒 Image Folder: {image_folder}{flag_indicator}")
        print(f"🔒 Already Installed: {', '.join(installed_workloads)}")
        print()
        print("Note: All settings locked. Only workload selection available.")
        print("=" * 70)
        print()

        # Filter workloads to exclude already installed
        filtered_workloads = filter_workloads_by_gpu_type(self.workloads, gpu_type)
        tools = filter_tools_from_workload_list(self.workloads)
        filtered_workloads.update(tools)
        filtered_workloads = resolve_gpu_overrides(filtered_workloads, gpu_type)

        # Remove already installed workloads
        available_workloads = {k: v for k, v in filtered_workloads.items() if k not in installed_workloads}

        if not available_workloads:
            print("No additional workloads available for this GPU type.")
            print("All supported workloads are already installed.")
            raise SystemExit(EXIT_CANCELLED)

        print(f"Available workloads to add ({len(available_workloads)}):")

        try:
            selected, _ = prompt_workload_selection(
                ui,
                available_workloads,
                show_install_all=True,  # Allow "Install All Available"
                allow_empty=False,
            )
        except KeyboardInterrupt:
            print("\n\nInstallation cancelled by user.")
            raise SystemExit(EXIT_CANCELLED) from None

        if not selected:
            print("\nNo workloads selected. Exiting.")
            raise SystemExit(EXIT_CANCELLED)

        # Build InstallConfig using existing settings
        gpu_slurm = slurm_config_dict.get('gpu', {})
        cpu_slurm = slurm_config_dict.get('cpu', {})
        slurm_obj = SlurmConfig(
            account=slurm_config_dict.get('account', ''),
            gpu_partition=gpu_slurm.get('partition', ''),
            cpu_partition=cpu_slurm.get('partition', ''),
            gpu_partition_gres=gpu_slurm.get('gres'),
            cpu_partition_gres=cpu_slurm.get('gres'),
        )

        config = InstallConfig(
            install_path=install_path,
            gpu_type=gpu_type,
            node_architecture=node_architecture,
            venv_type=venv_type,
            slurm=slurm_obj,
            environment_vars=env_vars,
            selected_workloads=selected,
            install_method=install_method,
            ui_mode=ui_mode,
            dev_mode=False,  # Incremental never uses dev mode
            llmb_repo=llmb_repo,
        )

        # Add image_folder if present
        if image_folder:
            config.image_folder = image_folder

        return (config, existing_workload_venvs)

    def _collect_express_configuration(self, args: argparse.Namespace) -> InstallConfig:
        """Collect configuration for express mode using saved system config.

        Express mode only prompts for install_path and workloads. Most values are
        taken from the saved system configuration, but fresh installs always use
        uv for newly-created recipe environments.

        Args:
            args: Parsed command line arguments

        Returns:
            InstallConfig: Complete installation configuration
        """
        # Load system config (guaranteed to exist due to create_ui check)
        system_config = load_system_config()
        if not system_config:
            print("Error: System config disappeared between checks")
            raise SystemExit(EXIT_ERROR)

        ui = SimpleUI()  # Use simple UI for minimal prompts

        # Display summary of saved configuration first
        ui.print_section("Express Mode Configuration Summary")
        ui.log(f"GPU Type: {system_config.gpu_type.upper()}")
        ui.log(f"Architecture: {system_config.node_architecture}")
        ui.log("Environment: uv")
        ui.log(f"Install Method: {system_config.install_method}")
        if system_config.slurm:
            ui.log(f"SLURM Account: {system_config.slurm.account}")
            ui.log(f"GPU Partition: {system_config.slurm.gpu_partition}")
            ui.log(f"CPU Partition: {system_config.slurm.cpu_partition}")
            if system_config.slurm.gpu_partition_gres:
                ui.log(f"GPUs per Node: {system_config.slurm.gpu_partition_gres}")
        if system_config.image_folder:
            ui.log(f"Image Folder: {system_config.image_folder}")
        if system_config.environment_vars:
            ui.log(f"Environment Variables: {len(system_config.environment_vars)} saved")
        print()

        # Get install path from arguments or prompt
        # Install path can come from positional argument or --install-path flag
        install_path_pos = getattr(args, 'install_path_pos', None)
        install_path_flag = getattr(args, 'install_path_flag', None)

        # Warn if both are provided to avoid confusion
        if install_path_pos and install_path_flag:
            ui.log(
                f"Warning: Both positional path '{install_path_pos}' and --install-path '{install_path_flag}' provided. Using positional argument."
            )

        install_path = install_path_pos or install_path_flag
        if install_path:
            # Convert to absolute path using pathlib
            install_path = str(Path(install_path).expanduser().resolve())
            ui.log(f"Using install path: {install_path}")
        else:
            install_path = prompt_install_location(ui)
            if not install_path:
                print("\nInstallation cancelled.")
                raise SystemExit(EXIT_CANCELLED)

        # NEW: Check for existing installation
        cluster_config = self._detect_incremental_install(install_path)
        if cluster_config:
            print("\nExisting installation detected at this path.")
            print("Express mode does not support incremental installation.")
            print("\nTo add workloads to an existing installation:")
            print("  1. cd to the installation directory")
            print("  2. Run: llmb-install")
            print("  3. Select workloads to add")
            raise SystemExit(EXIT_ERROR)

        # Handle repository copying here (before other prompts)
        dev_mode = getattr(args, 'dev_mode', False)
        self.root_dir = self._handle_repository_setup(install_path, dev_mode)

        # Fresh express installs always use uv for recipe environments,
        # regardless of stale saved venv_type defaults.
        venv_type = prompt_environment_type(ui, system_config.venv_type)

        # Get workloads from CLI or prompt and determine selection mode
        workload_selection_mode = 'custom'  # Default mode

        if args.workloads:
            if args.workloads.lower() == 'all':
                # Filter available workloads by saved GPU type
                filtered_workloads = filter_workloads_by_gpu_type(self.workloads, system_config.gpu_type)
                if not filtered_workloads:
                    print(f"Error: No workloads available for GPU type: {system_config.gpu_type}")
                    raise SystemExit(EXIT_ERROR)

                # Add tools
                tools = filter_tools_from_workload_list(self.workloads)
                filtered_workloads.update(tools)

                # Resolve GPU overrides
                filtered_workloads = resolve_gpu_overrides(filtered_workloads, system_config.gpu_type)

                selected = list(filtered_workloads.keys())
                ui.log(f"Installing all workloads for {system_config.gpu_type}: {', '.join(selected)}")
            else:
                # Parse comma-separated workload list
                selected = [w.strip() for w in args.workloads.split(',') if w.strip()]
                ui.log(f"Installing specified workloads: {', '.join(selected)}")
        elif getattr(args, 'exemplar', False):
            # Exemplar mode: Use exemplar.yaml to select workloads
            workload_selection_mode = 'exemplar'
            filtered_workloads = filter_workloads_by_gpu_type(self.workloads, system_config.gpu_type)
            if not filtered_workloads:
                print(f"Error: No workloads available for GPU type: {system_config.gpu_type}")
                raise SystemExit(EXIT_ERROR)

            # Add tools and resolve GPU overrides
            tools = filter_tools_from_workload_list(self.workloads)
            filtered_workloads.update(tools)
            filtered_workloads = resolve_gpu_overrides(filtered_workloads, system_config.gpu_type)

            # Get and validate workloads from exemplar.yaml
            try:
                base_keys = get_exemplar_workloads(Path(self.root_dir), system_config.gpu_type)
                selected = validate_exemplar_workloads(base_keys, filtered_workloads, system_config.gpu_type)
                ui.log(f"Installing Exemplar Cloud workloads from exemplar.yaml: {', '.join(selected)}")
            except ValueError as e:
                print(f"Error: {e}")
                raise SystemExit(EXIT_ERROR) from e
        else:
            # Prompt for workload selection
            filtered_workloads = filter_workloads_by_gpu_type(self.workloads, system_config.gpu_type)
            if not filtered_workloads:
                print(f"Error: No workloads available for GPU type: {system_config.gpu_type}")
                raise SystemExit(EXIT_ERROR)

            # Add tools
            tools = filter_tools_from_workload_list(self.workloads)
            filtered_workloads.update(tools)

            # Resolve GPU overrides
            filtered_workloads = resolve_gpu_overrides(filtered_workloads, system_config.gpu_type)

            selected, workload_selection_mode = prompt_workload_selection(ui, filtered_workloads)
            if not selected:
                print("\nInstallation cancelled.")
                raise SystemExit(EXIT_CANCELLED)

        # Use saved environment variables (express mode doesn't prompt for these)
        env_vars = system_config.environment_vars.copy() if system_config.environment_vars else {}

        # Handle image folder - use saved config if not provided via CLI
        image_folder = getattr(args, 'image_folder', None)
        if not image_folder and system_config.image_folder:
            image_folder = system_config.image_folder
            ui.log(f"Using saved image folder: {image_folder}")

        # Create SLURM config object
        slurm_obj = None
        if system_config.slurm:
            slurm_obj = SlurmConfig(
                account=system_config.slurm.account,
                gpu_partition=system_config.slurm.gpu_partition,
                cpu_partition=system_config.slurm.cpu_partition,
                gpu_partition_gres=system_config.slurm.gpu_partition_gres,
                cpu_partition_gres=system_config.slurm.cpu_partition_gres,
            )

        # Use saved values from system config, except for recipe environment type
        # which is forced to uv for fresh installs.
        config = InstallConfig(
            install_path=install_path,
            gpu_type=system_config.gpu_type,
            node_architecture=system_config.node_architecture,
            venv_type=venv_type,
            slurm=slurm_obj,
            environment_vars=env_vars,
            selected_workloads=selected,
            install_method=system_config.install_method,
            ui_mode='simple',  # Express mode uses simple UI
            image_folder=image_folder,
            dev_mode=dev_mode,
            llmb_repo=self.root_dir,
            workload_selection_mode=workload_selection_mode,
        )

        print("Starting installation...")
        return config

    def _list_available_workloads(self) -> None:
        """List all available workloads, optionally filtered by saved GPU type."""
        from llmb_install.config.system import load_system_config, system_config_exists

        # Use already loaded workloads (avoid duplicate repo root messages)
        if not hasattr(self, 'workloads') or not self.workloads:
            print("Error: No workloads found in repository")
            raise SystemExit(EXIT_ERROR)

        # Try to load system config for GPU filtering
        system_config = None
        if system_config_exists():
            system_config = load_system_config()

        if system_config:
            gpu_type_display = system_config.gpu_type.upper()
            print(f"Available workloads for {gpu_type_display}:")

            # Filter workloads by saved GPU type
            filtered_workloads = filter_workloads_by_gpu_type(self.workloads, system_config.gpu_type)
            if not filtered_workloads:
                print(f"No workloads available for {gpu_type_display}")
                return

            # Add tools
            tools = filter_tools_from_workload_list(self.workloads)
            filtered_workloads.update(tools)

            # Resolve GPU overrides
            filtered_workloads = resolve_gpu_overrides(filtered_workloads, system_config.gpu_type)

            # List all workloads
            print()
            for name in sorted(filtered_workloads.keys()):
                print(f"  {name}")

            print("\nUsage:")
            print("  llmb-install express -w all")
            print("  llmb-install express -w workload1,workload2")
        else:
            print("Available workloads by GPU type:")
            print("\nRun an installation first to save your GPU type, then use:")
            print("  llmb-install express --list-workloads")

            # Group by GPU type
            gpu_workloads = {}
            for name, workload in self.workloads.items():
                if 'run' in workload and 'gpu_configs' in workload['run']:
                    gpu_types = list(workload['run']['gpu_configs'].keys())
                    for gpu_type in gpu_types:
                        if gpu_type not in gpu_workloads:
                            gpu_workloads[gpu_type] = []
                        gpu_workloads[gpu_type].append(name)
                else:
                    # Tools or workloads without GPU configs
                    if 'Tools' not in gpu_workloads:
                        gpu_workloads['Tools'] = []
                    gpu_workloads['Tools'].append(name)

            for gpu_type in sorted(gpu_workloads.keys()):
                if gpu_type == 'Tools':
                    print(f"\n{gpu_type}:")
                else:
                    print(f"\n{gpu_type.upper()}:")
                for name in sorted(gpu_workloads[gpu_type]):
                    print(f"  {name}")

    def _categorize_missing_workloads(self, workload_names: List[str]) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
        """Categorize workloads as unsupported (exist but wrong GPU) or not found (don't exist).

        Args:
            workload_names: List of workload names to categorize

        Returns:
            Tuple of (unsupported_workloads, not_found_workloads)
            - unsupported_workloads: List of (name, supported_gpu_types)
            - not_found_workloads: List of workload names
        """
        unsupported = []
        not_found = []

        for w in workload_names:
            if w in self.workloads:
                supported_gpus = sorted(self.workloads[w].get('run', {}).get('gpu_configs', {}).keys())
                unsupported.append((w, supported_gpus))
            else:
                not_found.append(w)

        return unsupported, not_found

    def _prepare_workloads(self, gpu_type: str, selected: List[str]) -> Dict[str, Dict[str, Any]]:
        """Prepare workloads for installation by filtering and resolving GPU overrides.

        Args:
            gpu_type: Selected GPU type
            selected: List of selected workload keys

        Returns:
            Dictionary of prepared workloads
        """
        # Filter workloads by GPU type
        filtered_workloads = filter_workloads_by_gpu_type(self.workloads, gpu_type)
        if not filtered_workloads:
            # Early exit: No workloads in the repository support this GPU type
            unsupported, _ = self._categorize_missing_workloads(selected)

            if unsupported:
                # User selected specific workloads - show what they support
                print(f"\nError: The following workloads do not support gpu_type '{gpu_type}':")
                for name, gpus in unsupported:
                    print(f"  - {name} (supported gpu_types: {', '.join(gpus)})")
            else:
                # Generic error (e.g., empty selected list or invalid GPU type)
                print(f"No workloads found that support {gpu_type} GPU type.")

            raise SystemExit(EXIT_ERROR)

        # Add tools to workload list
        tools = filter_tools_from_workload_list(self.workloads)
        filtered_workloads.update(tools)

        # Resolve GPU-specific images/repos
        filtered_workloads = resolve_gpu_overrides(filtered_workloads, gpu_type)

        # Validate that all selected workloads exist
        missing_workloads = [w for w in selected if w not in filtered_workloads]
        if missing_workloads:
            unsupported, not_found = self._categorize_missing_workloads(missing_workloads)

            if unsupported:
                print(f"\nError: The following workloads do not support gpu_type '{gpu_type}':")
                for name, gpus in unsupported:
                    print(f"  - {name} (supported gpu_types: {', '.join(gpus)})")

            if not_found:
                print(f"\nError: The following workloads were not found: {not_found}")

            raise SystemExit(EXIT_ERROR)

        return filtered_workloads

    def _save_installation_progress(
        self,
        install_config: InstallConfig,
        workload_keys: List[str],
        completed_workloads: List[str],
        workload_venvs: Dict[str, str],
        existing_cluster_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save installation progress after completing a group of workloads.

        Updates the completed workloads list and persists state to disk for resume capability.
        Skips saving in dev mode to avoid state file creation during development.

        Args:
            install_config: Installation configuration
            workload_keys: List of workload names just completed
            completed_workloads: Running list of all completed workloads (mutated in-place)
            workload_venvs: Mapping of workload names to their venv paths
            existing_cluster_config: Original cluster config for incremental installs
        """
        try:
            for workload_key in workload_keys:
                if workload_key not in completed_workloads:
                    completed_workloads.append(workload_key)

            if install_config.dev_mode:
                return

            save_install_state(install_config, completed_workloads, workload_venvs, existing_cluster_config)
        except Exception as e:
            print(f"Warning: Could not update installation state: {e}")

    def _perform_installation(
        self,
        install_config: InstallConfig,
        filtered_workloads: Dict[str, Dict[str, Any]],
        is_resume: bool = False,
        existing_completed_workloads: Optional[List[str]] = None,
        existing_workload_venvs: Optional[Dict[str, str]] = None,
        existing_cluster_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Perform the actual installation process.

        This is the core installation worker function that orchestrates all installation modes
        (interactive, headless, express) and handles resume/incremental scenarios.

        Args:
            install_config: Complete installation configuration including:
                - install_path: Target directory for installation
                - gpu_type, node_architecture: Hardware configuration
                - venv_type: Python environment type (uv/venv/conda)
                - slurm: SLURM cluster configuration (if install_method='slurm')
                - selected_workloads: List of workload names to install
                - environment_vars: Environment variables to set during installation
                - dev_mode: If True, skip state saving and use repo in-place
                - image_folder: Optional shared container image directory

            filtered_workloads: Workload metadata dictionary filtered by GPU type.
                Format: {workload_name: {metadata, config, dependencies, ...}}

            is_resume: If True, preserves existing install state (don't clear).
                Used when resuming a previously interrupted installation.

            existing_completed_workloads: List of already-installed workload names.
                Required for resume/incremental installs to avoid reinstalling.

            existing_workload_venvs: Dictionary mapping workload names to their venv paths.
                Enables venv reuse in resume/incremental scenarios for efficiency.

            existing_cluster_config: Original cluster_config.yaml contents (incremental installs only).
                Preserved and merged with new workloads when adding to an existing installation.

        Usage Scenarios:
            Fresh install (interactive/headless/express):
                _perform_installation(install_config, filtered_workloads, is_resume=False)

            Resume interrupted install:
                _perform_installation(install_config, filtered_workloads, is_resume=True,
                                    existing_completed_workloads=completed,
                                    existing_workload_venvs=venvs)

            Incremental install (add workloads to existing):
                _perform_installation(install_config, filtered_workloads, is_resume=False,
                                    existing_workload_venvs=original_venvs,
                                    existing_cluster_config=original_config)

        Note:
            The name 'install_config' distinguishes this from 'existing_cluster_config'
            and other config types used throughout the installer.
        """
        # Configure uv/pip cache directories before any venv/dependency work so that
        # all install subprocesses inherit them via os.environ. Single source of truth
        # for every install mode (interactive/express/headless/resume/incremental).
        setup_cache_directories(install_config.install_path, install_config.venv_type)

        # Clear any existing install state only if this is NOT a resume operation
        if not is_resume:
            try:
                if install_state_exists():
                    clear_install_state()
                    print("Cleared previous installation state - starting fresh installation")
            except Exception as e:
                print(f"Warning: Could not clear previous installation state: {e}")

        # Initialize state tracking for resume functionality
        completed_workloads = existing_completed_workloads[:] if existing_completed_workloads else []

        # Save initial state (if not dev mode)
        if not install_config.dev_mode:
            try:
                # Save initial state BEFORE directory creation so if makedirs fails,
                # we still have state for resume.
                # Include existing workload venvs so that resumed installs preserve
                # venv paths for previously completed workloads in the state file.
                initial_venvs = dict(existing_workload_venvs) if existing_workload_venvs else {}
                save_install_state(install_config, completed_workloads, initial_venvs, existing_cluster_config)

            except Exception as e:
                print(f"Warning: Could not initialize state tracking: {e}")
                # Continue with installation even if state tracking fails
        # Setup install directory structure
        os.makedirs(install_config.install_path, exist_ok=True)
        os.makedirs(os.path.join(install_config.install_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(install_config.install_path, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(install_config.install_path, "workloads"), exist_ok=True)
        os.makedirs(os.path.join(install_config.install_path, "venvs"), exist_ok=True)
        os.makedirs(os.path.join(install_config.install_path, "tools"), exist_ok=True)
        os.makedirs(os.path.join(install_config.install_path, "bin"), exist_ok=True)

        create_llmb_run_symlink(install_config.install_path)
        setup_compute_uv_binaries(install_config.install_path, install_config.node_architecture)

        print("\nDownloading required container images.")
        print("--------------------------------")
        required_images = get_required_images(filtered_workloads, install_config.selected_workloads)
        print("\nRequired container images:")
        for image, filename in sorted(required_images.items()):
            print(f"  - {image} -> {filename}")
        print("\n")

        # Use image_folder from config
        effective_image_folder = install_config.image_folder

        slurm_info = {'slurm': install_config.slurm.model_dump()} if install_config.slurm else {}

        fetch_container_images(
            required_images,
            install_config.install_path,
            install_config.node_architecture,
            install_config.install_method,
            slurm_info,
            effective_image_folder,
        )

        # Download HuggingFace assets
        hf_token = install_config.environment_vars.get('HF_TOKEN')
        download_huggingface_files_for_workloads(
            filtered_workloads,
            install_config.selected_workloads,
            install_config.install_path,
            hf_token,
            gpu_type=install_config.gpu_type,
        )

        # Download and install required tools
        required_tools = get_required_tools(filtered_workloads, install_config.selected_workloads)
        if required_tools:
            print("\nDownloading required tools.")
            print("--------------------------------")
            print("\nRequired tools:")
            for tool_name, version in sorted(required_tools):
                print(f"  - {tool_name}: {version}")
            print("\n")
            fetch_and_install_tools(required_tools, install_config.install_path, install_config.node_architecture)

        # Initialize with existing venvs so that state saves during resume always
        # include venv paths for previously completed workloads. Without this,
        # _save_installation_progress would persist only current-run venvs to the
        # state file, causing previously completed workloads to lose their venv
        # paths if the resumed install is interrupted again.
        workload_venvs = dict(existing_workload_venvs) if existing_workload_venvs else {}
        dep_groups = group_workloads_by_dependencies(filtered_workloads, install_config.selected_workloads)

        # Build reverse mapping: dep_hash -> venv_path for incremental installs
        # This allows us to reuse existing venvs when dependencies match
        from llmb_install.environment.venv_utils import build_venv_hash_mapping

        venv_to_hash = {}
        if existing_workload_venvs:
            venv_to_hash = build_venv_hash_mapping(existing_workload_venvs)
            for dep_hash, venv_path in venv_to_hash.items():
                self.logger.debug(f"Mapped existing venv hash {dep_hash} -> {venv_path}")

        # Show the user how workloads will be grouped
        print_dependency_group_summary(dep_groups)

        print("Installing Workloads")
        print("===================")

        try:
            for dep_hash, workload_keys in dep_groups.items():
                if dep_hash is None:
                    print("\n[No Dependency Setup]")
                    print("-" * 60)
                    for workload_key in workload_keys:
                        workload_dir = os.path.join(install_config.install_path, "workloads", workload_key)
                        os.makedirs(workload_dir, exist_ok=True)
                        print(f"Installing: {workload_key}")

                        venv_path = None
                        setup_config = filtered_workloads[workload_key].get('setup', {}) or {}
                        if setup_config.get('venv_req', False):
                            venvs_dir = os.path.join(install_config.install_path, "venvs")
                            os.makedirs(venvs_dir, exist_ok=True)
                            venv_path = os.path.join(venvs_dir, f"{workload_key}_venv")
                            create_virtual_environment(venv_path, install_config.venv_type)
                            workload_venvs[workload_key] = venv_path
                        else:
                            print(f"No virtual environment required for {workload_key}")

                        run_setup_tasks(
                            workload_key,
                            filtered_workloads[workload_key],
                            venv_path,
                            install_config.venv_type,
                            install_config.install_path,
                            slurm_info,
                            install_config.environment_vars,
                            install_config.gpu_type,
                            install_config.node_architecture,
                        )

                    self._save_installation_progress(
                        install_config, workload_keys, completed_workloads, workload_venvs, existing_cluster_config
                    )

                else:  # New dependency management
                    venvs_dir = os.path.join(install_config.install_path, "venvs")

                    # Use unified naming scheme for all dependency-based workloads
                    venv_name = f"venv_{dep_hash[:12]}"

                    # Check for existing venv with matching hash (incremental install)
                    from llmb_install.environment.venv_utils import should_reuse_venv

                    existing_venv_path = should_reuse_venv(dep_hash, venv_to_hash, validate_exists=True)

                    if existing_venv_path:
                        # REUSE EXISTING VENV
                        if len(workload_keys) == 1:
                            workload_key = workload_keys[0]
                            print("\n[Reusing Existing Virtual Environment]")
                            print(f"Installing: {workload_key}")
                            print(f"Matching venv: {existing_venv_path}")
                            print("-" * 70)
                        else:
                            print("\n[Reusing Existing Shared Virtual Environment]")
                            print(f"Installing workloads: {', '.join(sorted(workload_keys))}")
                            print(f"Matching venv: {existing_venv_path}")
                            print("-" * 70)

                        # Validate venv actually exists before reusing
                        if not os.path.exists(existing_venv_path):
                            print(f"\nError: Venv path in cluster config does not exist: {existing_venv_path}")
                            print("Installation appears incomplete or corrupted.")
                            print("Please verify the installation directory or reinstall.")
                            raise SystemExit(EXIT_ERROR)

                        venv_path = existing_venv_path

                        # Clone repos and set up workload directories (dependencies already installed)
                        # Each workload resolves its own deps for cloning, since clone-only
                        # deps are excluded from the grouping hash and may differ per workload.
                        for workload_key in workload_keys:
                            workload_dir = os.path.join(install_config.install_path, "workloads", workload_key)
                            os.makedirs(workload_dir, exist_ok=True)

                            workload_deps = _resolve_dependencies(filtered_workloads[workload_key])
                            git_deps_to_clone = workload_deps.get('git', {}) if workload_deps else {}

                            if git_deps_to_clone:
                                clone_git_repos(git_deps_to_clone, workload_dir, install_config.install_path)

                            workload_venvs[workload_key] = venv_path

                        for workload_key in workload_keys:
                            workload_data = filtered_workloads[workload_key]
                            run_setup_tasks(
                                workload_key,
                                workload_data,
                                venv_path,
                                install_config.venv_type,
                                install_config.install_path,
                                slurm_info,
                                install_config.environment_vars,
                                install_config.gpu_type,
                                install_config.node_architecture,
                            )

                        print("✓ Workloads installed successfully (reused venv)")

                    else:
                        # CREATE NEW VENV (normal flow)
                        if len(workload_keys) == 1:
                            # If group has only one workload, it's still an individual installation
                            workload_key = workload_keys[0]
                            print("\n[Individual Installation - Unique Dependencies]")
                            print(f"Installing: {workload_key}")
                            print("-" * 60)
                        else:
                            # Otherwise, use a shared name for the group
                            print("\n[Shared Virtual Environment Group]")
                            print(f"Installing workloads: {', '.join(sorted(workload_keys))}")
                            print("-" * 70)

                        # 1. Create one venv for this group
                        venv_path = os.path.join(venvs_dir, venv_name)
                        create_virtual_environment(venv_path, install_config.venv_type)

                        # Get resolved dependencies from the first workload for venv installation
                        first_workload_key = workload_keys[0]
                        dependencies = _resolve_dependencies(filtered_workloads[first_workload_key])

                        if dependencies is None:
                            print(f"Warning: No dependencies found for workload group containing {first_workload_key}")
                            continue

                        # 2. For each workload, create its folder and clone any repos that need to be local.
                        # Each workload resolves its own deps for cloning, since clone-only
                        # deps are excluded from the grouping hash and may differ per workload.
                        for workload_key in workload_keys:
                            workload_dir = os.path.join(install_config.install_path, "workloads", workload_key)
                            os.makedirs(workload_dir, exist_ok=True)

                            workload_deps = _resolve_dependencies(filtered_workloads[workload_key])
                            git_deps_to_clone = workload_deps.get('git', {}) if workload_deps else {}

                            if git_deps_to_clone:
                                clone_git_repos(git_deps_to_clone, workload_dir, install_config.install_path)
                            workload_venvs[workload_key] = venv_path

                        # 3. Install dependencies into the shared venv using the first workload's clones
                        env = get_venv_environment(venv_path, install_config.venv_type)
                        env['LLMB_INSTALL'] = install_config.install_path
                        env['MANUAL_INSTALL'] = 'false'
                        env['GPU_TYPE'] = install_config.gpu_type
                        if install_config.environment_vars:
                            env.update(install_config.environment_vars)

                        first_workload_dir = os.path.join(install_config.install_path, "workloads", first_workload_key)
                        install_dependencies(venv_path, install_config.venv_type, dependencies, first_workload_dir, env)

                        # 4. For each workload, run setup tasks (if any)
                        for workload_key in workload_keys:
                            workload_data = filtered_workloads[workload_key]
                            run_setup_tasks(
                                workload_key,
                                workload_data,
                                venv_path,
                                install_config.venv_type,
                                install_config.install_path,
                                slurm_info,
                                install_config.environment_vars,
                                install_config.gpu_type,
                                install_config.node_architecture,
                            )

                    # Track completion for dependency-based workloads (after all workloads in this group complete)
                    self._save_installation_progress(
                        install_config, workload_keys, completed_workloads, workload_venvs, existing_cluster_config
                    )
        except Exception:
            # Save config for successfully completed workloads
            if completed_workloads:
                try:
                    all_venvs = dict(existing_workload_venvs) if existing_workload_venvs else {}
                    all_venvs.update(workload_venvs)
                    create_cluster_config(
                        install_config.install_path,
                        self.root_dir,
                        completed_workloads,
                        slurm_info,
                        install_config.environment_vars,
                        install_config.gpu_type,
                        install_config.venv_type,
                        all_venvs,
                        node_architecture=install_config.node_architecture,
                        install_method=install_config.install_method,
                        image_folder=install_config.image_folder,
                        existing_cluster_config=existing_cluster_config,
                    )
                except Exception as config_err:
                    self.logger.debug(f"Could not save partial cluster config: {config_err}")

            failed_workloads = [w for w in install_config.selected_workloads if w not in completed_workloads]
            write_install_summary_file(
                status='failed',
                install_path=install_config.install_path,
                failed_workloads=failed_workloads,
                logger=self.logger,
            )
            print("\n" + "=" * 60)
            print("INSTALLATION INCOMPLETE")
            print("=" * 60)
            if completed_workloads:
                print(f"  Succeeded: {', '.join(completed_workloads)}")
            print(f"  Failed:    {', '.join(failed_workloads)}")
            print("")
            print("Re-run the installer to resume. You can retry or skip")
            print("the failed workloads.")
            print("=" * 60)
            raise

        # Merge venvs from prior completed workloads with current run
        all_venvs = dict(existing_workload_venvs) if existing_workload_venvs else {}
        all_venvs.update(workload_venvs)

        create_cluster_config(
            install_config.install_path,
            self.root_dir,
            completed_workloads,  # Includes all workloads (prior + current)
            slurm_info,
            install_config.environment_vars,
            install_config.gpu_type,
            install_config.venv_type,
            all_venvs,  # Includes venvs for all completed workloads
            node_architecture=install_config.node_architecture,
            install_method=install_config.install_method,
            image_folder=install_config.image_folder,
            existing_cluster_config=existing_cluster_config,  # For incremental installs
        )

        print(f"\nInstallation complete! Workloads have been installed to: {install_config.install_path}")

        # Save system config for future defaults (excluding per-install variables)
        try:
            save_system_config(install_config)
        except Exception as e:
            print(f"Warning: Failed to save system config for future defaults: {e}")

        # Flag to track if any async jobs were submitted
        async_jobs_submitted = False

        # Pre-check for async tasks across all selected workloads (sbatch, nemo2)
        for workload_key in install_config.selected_workloads:
            workload_data = filtered_workloads[workload_key]
            for task in workload_data.get('setup', {}).get('tasks', []):
                job_type = task.get('job_type', 'local').lower()
                if job_type in ('sbatch', 'nemo2'):
                    async_jobs_submitted = True
                    break  # No need to check other tasks for this workload
            if async_jobs_submitted:  # If found in this workload, no need to check other workloads
                break

        # Add the notice here if async jobs were submitted
        if async_jobs_submitted:
            message_lines = [
                "IMPORTANT: Some installation tasks were submitted as SLURM sbatch jobs.",
                "These jobs may still be queued or running in the background.",
                "To verify their status, please run: 'squeue -u $USER'",
            ]
            max_len = max(len(line) for line in message_lines)
            border = "=" * (max_len + 4)  # +4 for padding

            print(f"\n{border}")
            for line in message_lines:
                print(f"  {line.ljust(max_len)}  ")
            print(f"{border}\n")

        print("To run benchmarks now:")
        print(f"  cd {install_config.install_path}")
        print("  llmb-run")
        print("")
        print("Optional: set LLMB_INSTALL if you want to run llmb-run from any directory:")
        print(f"  export LLMB_INSTALL={install_config.install_path}")
        print("Add this to your shell profile (e.g., ~/.bashrc or ~/.bash_aliases) if useful.")

        # Clear install state on successful completion
        if not install_config.dev_mode:
            try:
                clear_install_state()
            except Exception as e:
                print(f"Warning: Could not clear installation state: {e}")

        write_install_summary_file(
            status='success',
            install_path=install_config.install_path,
            async_jobs_submitted=async_jobs_submitted,
            logger=self.logger,
        )

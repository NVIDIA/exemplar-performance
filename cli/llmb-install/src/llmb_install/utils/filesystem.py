# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""File system utilities for LLMB installer."""

import os
import shutil
import stat
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def find_llmb_repo_root() -> str:
    """Find the LLMB repository root directory.

    Flexible detection to support both legacy and new layouts. The root is
    identified by a combination of markers rather than a single rigid path.

    Returns:
        str: Path to the LLMB repository root directory

    Raises:
        SystemExit: If no valid LLMB repository root can be found
    """

    def is_llmb_repo_root(path: Path) -> bool:
        """Check if a directory appears to be the LLMB repository root.

        Heuristics (any sufficiently strong combination is accepted):
        - install.sh at root (preferred)
        - README.md at root (preferred)
        - New layout markers under cli/: cli/llmb-install/installer.py or cli/llmb-run/pyproject.toml
        - Legacy markers: installer/installer.py or llmb-run/pyproject.toml
        - VCS markers: .jj/ or .git/ (weak indicators)
        """
        # Strong markers
        has_install_sh = (path / 'install.sh').exists()
        has_root_readme = (path / 'README.md').exists()

        # New layout
        has_new_installer = (path / 'cli' / 'llmb-install' / 'installer.py').exists()
        has_new_run = (path / 'cli' / 'llmb-run' / 'pyproject.toml').exists() or (
            path / 'cli' / 'llmb-run' / 'README.md'
        ).exists()

        # Legacy layout (pre-reorg)
        has_legacy_installer = (path / 'installer' / 'installer.py').exists()
        has_legacy_run = (path / 'llmb-run' / 'pyproject.toml').exists() or (path / 'llmb-run' / 'README.md').exists()

        # VCS markers (weak)
        has_vcs = (path / '.jj').exists() or (path / '.git').exists()

        # Acceptance rules:
        # - If install.sh + (any installer or run marker) → accept
        if has_install_sh and (has_new_installer or has_legacy_installer or has_new_run or has_legacy_run):
            return True
        # - If README + (new installer or legacy installer) → accept
        if has_root_readme and (has_new_installer or has_legacy_installer):
            return True
        # - If VCS marker + (new installer or legacy installer) → accept
        if has_vcs and (has_new_installer or has_legacy_installer):
            return True
        # - Fallback: README + (any run marker) → accept
        if has_root_readme and (has_new_run or has_legacy_run):
            return True

        return False

    def find_repo_root_upward(start_path: Path) -> Optional[Path]:
        """Search upward from start_path to find the repository root."""
        current = start_path.resolve()

        # Limit search
        max_levels = 5
        for _ in range(max_levels):
            if is_llmb_repo_root(current):
                return current
            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent
        return None

    repo_root = find_repo_root_upward(Path.cwd())
    if repo_root:
        print(f"Found LLMB repository root: {repo_root}")
        return str(repo_root)

    print("\nError: Could not find LLMB repository.")
    print("Please run this command from within an LLMB repository checkout.")
    raise SystemExit(1)


def create_llmb_run_symlink(install_path: str) -> None:
    """Create a symbolic link to the llmb-run binary in the install_path.

    This creates a symlink to the llmb-run binary in the virtual environment
    where llmb-install is currently running from, ensuring consistency between
    the installer and runner environments.

    Args:
        install_path: Installation directory where symlink will be created
    """
    print("\nCreating llmb-run symbolic link")
    print("----------------------------")

    # Find the bin directory of the current Python environment
    python_executable = Path(sys.executable)
    bin_dir = python_executable.parent

    # Look for llmb-run in the same bin directory as the current Python
    llmb_run_source = bin_dir / "llmb-run"
    llmb_run_symlink_target = Path(install_path) / "llmb-run"

    try:
        if llmb_run_source.exists():
            # Remove existing symlink if it exists to avoid FileExistsError
            if llmb_run_symlink_target.exists() or llmb_run_symlink_target.is_symlink():
                llmb_run_symlink_target.unlink()
            llmb_run_symlink_target.symlink_to(llmb_run_source)
            print(f"✓ Created symbolic link to {llmb_run_source} at {llmb_run_symlink_target}")
        else:
            print(f"✗ Warning: llmb-run binary not found at {llmb_run_source}. " "Skipping symlink creation.")
            print("  Make sure llmb-run is installed in the same environment as llmb-install.")
    except Exception as e:
        print(f"✗ Error creating symbolic link: {e}")


def _find_recipe_directories(src_path: Path) -> set:
    """Find top-level directories that contain recipe metadata.yaml files.

    Args:
        src_path: Source repository path

    Returns:
        Set of top-level directory names that contain recipes
    """
    recipe_dirs = set()

    # Find all metadata.yaml files
    for metadata_file in src_path.rglob('metadata.yaml'):
        try:
            # Get relative path from repo root
            rel_path = os.path.relpath(str(metadata_file), str(src_path))

            # TODO: deprecated/ was removed from the repo in 26.10. Drop this skip once all
            # releases that still ship it fall outside the support window (~6 months).
            if rel_path.startswith('deprecated/'):
                continue

            # Get the top-level directory
            top_level_dir = rel_path.split(os.sep)[0]
            recipe_dirs.add(top_level_dir)

        except (ValueError, OSError):
            continue

    return recipe_dirs


def _get_whitelisted_directories() -> set:
    """Get directories that should always be copied.

    Returns:
        Set of directory names to always include
    """
    # Common directories needed for recipes to function that don't have metadata.yaml files
    return {
        'common',
        'llmb-tools',
    }


def copy_repository_working_files(src_dir: str, dest_dir: str) -> None:
    """Copy only recipe directories and whitelisted folders from repository.

    This function copies:
    - Directories containing metadata.yaml files (recipes)
    - Whitelisted directories (common, scripts, etc.)
    - Skips deprecated/ directory

    Args:
        src_dir: Source repository directory
        dest_dir: Destination directory for copy

    Raises:
        Exception: If copy fails for any reason
    """
    try:
        src_path = Path(src_dir).resolve()
        dest_path = Path(dest_dir).resolve()

        print(f"Copying recipe directories from {src_path} to {dest_path}")

        # Check if source exists
        if not src_path.exists():
            raise Exception(f"Source directory does not exist: {src_path}")

        # Create destination directory
        dest_path.mkdir(parents=True, exist_ok=True)

        # Find directories to copy
        recipe_dirs = _find_recipe_directories(src_path)
        whitelisted_dirs = _get_whitelisted_directories()

        # Combine recipe directories and whitelisted directories
        dirs_to_copy = recipe_dirs.union(whitelisted_dirs)

        # Filter out directories that don't actually exist
        existing_dirs_to_copy = set()
        for dir_name in dirs_to_copy:
            src_dir_path = src_path / dir_name
            if src_dir_path.exists():
                existing_dirs_to_copy.add(dir_name)

        if not existing_dirs_to_copy:
            raise Exception("No recipe directories or whitelisted directories found to copy")

        print(f"Copying {len(existing_dirs_to_copy)} directories: {', '.join(sorted(existing_dirs_to_copy))}")

        copied_files = 0

        # Copy each directory
        for dir_name in sorted(existing_dirs_to_copy):
            src_dir_path = src_path / dir_name
            dest_dir_path = dest_path / dir_name

            print(f"  Copying {dir_name}/...")

            # Copy the entire directory tree
            try:
                shutil.copytree(src_dir_path, dest_dir_path, dirs_exist_ok=True)

                # Count files for progress indication
                for item in dest_dir_path.rglob('*'):
                    if item.is_file():
                        copied_files += 1

            except Exception as e:
                raise Exception(f"Failed to copy directory {dir_name}: {e}") from e

        # Copy release.yaml if it exists
        release_yaml_path = src_path / "release.yaml"
        if release_yaml_path.exists():
            print("  Copying release.yaml...")
            try:
                shutil.copy(release_yaml_path, dest_path)
                copied_files += 1
            except Exception as e:
                raise Exception(f"Failed to copy release.yaml: {e}") from e

        # Copy exemplar.yaml if it exists
        exemplar_yaml_path = src_path / "exemplar.yaml"
        if exemplar_yaml_path.exists():
            print("  Copying exemplar.yaml...")
            try:
                shutil.copy(exemplar_yaml_path, dest_path)
                copied_files += 1
            except Exception as e:
                raise Exception(f"Failed to copy exemplar.yaml: {e}") from e

        # Create sentinel file to mark successful completion - this is critical for validation
        sentinel_file = dest_path / '.llmb_repo_copy_complete'
        with open(sentinel_file, 'w') as f:
            f.write("Repository copy completed successfully\n")
            f.write(f"Source: {src_path}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Files copied: {copied_files}\n")
            f.write(f"Directories copied: {', '.join(sorted(existing_dirs_to_copy))}\n")

        # Make sentinel file read-only
        os.chmod(sentinel_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        print(f"✓ Repository copy completed: {copied_files} files in {len(existing_dirs_to_copy)} directories")

    except Exception as e:
        # Clean up partial copy on failure
        try:
            if dest_path.exists():
                shutil.rmtree(dest_path)
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up partial copy: {cleanup_error}")
        raise Exception(f"Repository copy failed: {e}") from e


def is_repository_copy_complete(dest_dir: str, install_root: str) -> bool:
    """Check if repository copy is complete and valid.

    Args:
        dest_dir: Destination directory to check
        install_root: Installation root directory

    Returns:
        True if copy is complete or if in dev mode scenario
    """
    try:
        dest_path = Path(dest_dir).resolve()
        install_path = Path(install_root).resolve()

        # Check if install_root is not contained in dest_dir path (dev mode scenario)
        # Use os.path.relpath which is more robust across Python versions
        try:
            os.path.relpath(str(dest_path), str(install_path))
            # Check if dest_path starts with install_path (normal mode)
            if str(dest_path).startswith(str(install_path)):
                # dest_dir is under install_root (normal mode) - check for sentinel file
                sentinel_file = dest_path / '.llmb_repo_copy_complete'
                return sentinel_file.exists()
            else:
                # dest_dir is not under install_root (dev mode scenario)
                return True
        except (ValueError, OSError):
            # If we can't determine the relationship, assume dev mode
            return True

    except Exception as e:
        print(f"Warning: Could not check repository copy status: {e}")
        return False


def check_repository_state(install_path: str) -> str:
    """Check the state of the LLMB_INSTALL directory and llmb_repo subdirectory.

    Args:
        install_path: Path to the installation directory

    Returns:
        str: One of 'empty', 'resume', 'existing_install', 'orphaned'
    """
    install_dir = Path(install_path)
    llmb_repo_dir = install_dir / "llmb_repo"
    cluster_config_file = install_dir / "cluster_config.yaml"
    resume_state_file = Path.home() / ".config" / "llmb" / "install_state.yaml"

    # Check if llmb_repo directory exists and is non-empty
    if not llmb_repo_dir.exists():
        return 'empty'

    # Check if llmb_repo directory is empty
    try:
        if not any(llmb_repo_dir.iterdir()):
            return 'empty'
    except (OSError, PermissionError):
        # If we can't read the directory, treat as empty
        return 'empty'

    # llmb_repo exists and is non-empty - check for other indicators
    has_cluster_config = cluster_config_file.exists()
    has_resume_state = False

    # Check if resume state exists and points to this install path
    if resume_state_file.exists():
        try:
            import yaml

            with open(resume_state_file, 'r') as f:
                state_data = yaml.safe_load(f)
            if state_data and state_data.get('install_config', {}).get('install_path') == install_path:
                has_resume_state = True
        except Exception:
            # If we can't read resume state, assume it doesn't exist
            pass

    if has_resume_state:
        return 'resume'
    elif has_cluster_config:
        return 'existing_install'
    else:
        return 'orphaned'


def clean_repository_directory(repo_path: str) -> None:
    """Clean (remove) a repository directory completely.

    Args:
        repo_path: Path to the repository directory to remove
    """
    repo_dir = Path(repo_path)
    if repo_dir.exists():
        try:
            shutil.rmtree(repo_dir)
        except (OSError, PermissionError) as e:
            raise Exception(f"Failed to remove repository directory {repo_path}: {e}") from e

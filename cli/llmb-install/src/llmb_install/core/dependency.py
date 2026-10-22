# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Dependency resolution and installation for LLMB workloads.

This module provides functions for resolving workload dependencies, grouping workloads
by their dependencies, and installing shared dependencies into virtual environments.
"""

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)


def _canonicalize_dependencies_for_grouping(dependencies: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Create a canonical representation of dependencies for grouping.

    Goal: group workloads that effectively install the same code even if the
    acquisition method differs (e.g., script vs pip) or wrappers were present.

    Strategy:
      - For pip entries:
          • Strings are kept as canonical strings (prefixed for type).
          • Dicts with url+commit collapse to canonical triplets, ignoring
            editable/install_target/repo_key differences.
      - For git entries:
          • Clone-only deps (install_method.type == 'clone') are excluded
            since they don't modify the venv.
          • Remaining deps collapse to canonical url+commit pairs.
      - Return an object with sorted, deduplicated lists for stable hashing.
    """
    if not dependencies:
        return None

    canonical_pip: set[str] = set()
    canonical_sources: set[str] = set()
    pip_git_pairs: set[tuple[str, str]] = set()
    git_pairs: set[tuple[str, str]] = set()

    # Pip normalization
    for item in dependencies.get('pip', []) or []:
        if isinstance(item, str):
            canonical_pip.add(f"pip-spec|{item}")
        elif isinstance(item, dict):
            url = item.get('url')
            commit = item.get('commit')
            if url and commit:
                pip_git_pairs.add((url, commit))
            else:
                # For pip dicts without url/commit, use full dict representation
                # This ensures different dicts don't get incorrectly grouped together
                canonical_pip.add(f"pip-dict|{json.dumps(item, sort_keys=True)}")

    # Git normalization — only include deps that modify the venv
    for _name, git_info in dependencies.get('git', {}).items():
        install_method = git_info.get('install_method', {})
        if install_method.get('type') == 'clone':
            continue
        url = git_info.get('url')
        commit = git_info.get('commit')
        if url and commit:
            git_pairs.add((url, commit))

    # Sources normalization
    sources = dependencies.get('sources', []) or []
    for source in sources:
        if isinstance(source, str):
            canonical_sources.add(source)

    # Merge pip-git and git-git into unified set
    all_git_pairs = pip_git_pairs | git_pairs

    # Create sorted list for deterministic output
    return {'pip': sorted(canonical_pip), 'sources': sorted(canonical_sources), 'git_repos': sorted(all_git_pairs)}


def _has_editable_pip_dep(dependencies: Optional[Dict[str, Any]]) -> bool:
    """Return True if any pip dependency requests an editable install.

    Editable installs are isolated to their own venv to avoid cross-recipe
    interference from mutable source checkouts.
    """
    if not dependencies:
        return False
    for item in dependencies.get('pip', []) or []:
        if isinstance(item, dict) and item.get('editable'):
            return True
    return False


def _resolve_dependencies(workload_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolves repo_key references in a dependency spec."""
    dependencies = copy.deepcopy(workload_data.get('setup', {}).get('dependencies'))
    if not dependencies:
        return None

    if not isinstance(dependencies, dict):
        raise TypeError(
            f"Invalid 'dependencies' format in metadata.yaml. Expected a dictionary with 'git' or 'pip' keys, but got {type(dependencies).__name__}. "
            "Please check your yaml structure (e.g. ensure items are under 'pip:' key, not a direct list)."
        )

    repositories = workload_data.get('repositories', {})

    # Resolve git dependencies
    for _name, git_dep in dependencies.get('git', {}).items():
        repo_key = git_dep.get('repo_key')
        if repo_key and repo_key in repositories:
            git_dep.update(repositories[repo_key])
            del git_dep['repo_key']  # remove the key after resolving

    # Resolve pip dependencies
    for pip_dep in dependencies.get('pip', []):
        if isinstance(pip_dep, dict):
            repo_key = pip_dep.get('repo_key')
            if repo_key and repo_key in repositories:
                pip_dep.update(repositories[repo_key])
                del pip_dep['repo_key']

    # Add pip deps that require cloning
    for pip_dep in dependencies.get('pip', []):
        if isinstance(pip_dep, dict):
            if pip_dep.get('editable') or pip_dep.get('install_target'):
                git_deps = dependencies.setdefault('git', {})

                try:
                    git_deps[pip_dep['package']] = {
                        'url': pip_dep['url'],
                        'commit': pip_dep['commit'],
                        'install_method': {'type': 'pip'},
                    }
                except KeyError as e:
                    missing_field = e.args[0] if e.args else "unknown"
                    raise KeyError(
                        f"Error in pip dependency '{pip_dep.get('package', '<unknown>')}': "
                        f"required field '{missing_field}' is missing. "
                        f"Each pip dependency that uses 'editable' or 'install_target' must specify both 'url' and 'commit' or use 'repo_key'."
                    ) from e

    # Remove empty git section if no repos were added
    if 'git' in dependencies and not dependencies['git']:
        del dependencies['git']

    return dependencies


def group_workloads_by_dependencies(
    workloads: Dict[str, Dict[str, Any]], selected_keys: List[str]
) -> Dict[Optional[str], List[str]]:
    """Group workloads by their fully resolved dependency specification."""
    dep_groups: Dict[Optional[str], List[str]] = {}

    for key in selected_keys:
        workload_data = workloads[key]
        resolved_deps = _resolve_dependencies(workload_data)

        if not resolved_deps:
            # Workloads without Python dependencies do not need dependency
            # installation. They may still need image downloads or setup tasks.
            if None not in dep_groups:
                dep_groups[None] = []
            dep_groups[None].append(key)
            continue

        # Editable installs get isolated venvs regardless of otherwise matching deps.
        if _has_editable_pip_dep(resolved_deps):
            dep_hash = f"editable-isolated::{key}"
            # print(f"[venv-grouping] {key}: forcing individual venv due to editable pip dependency")
        else:
            # Create a canonical representation that unifies equivalent sources
            # across acquisition methods (git vs pip-git) and ignores non-functional fields.
            canonical = _canonicalize_dependencies_for_grouping(resolved_deps)
            dep_string = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
            dep_hash = hashlib.sha256(dep_string.encode('utf-8')).hexdigest()
            # print(f"[venv-grouping] {key}: hash={dep_hash}")
            # print(f"[venv-grouping] {key} canonical={dep_string}")

        if dep_hash not in dep_groups:
            dep_groups[dep_hash] = []
        dep_groups[dep_hash].append(key)

    return dep_groups


def print_dependency_group_summary(dep_groups: Dict[Optional[str], List[str]]) -> None:
    """Print a user-friendly summary of how workloads are grouped by dependencies."""
    print("\nWorkload Installation Plan")
    print("=========================")

    no_dependency_workloads = dep_groups.get(None, [])
    dependency_groups = {k: v for k, v in dep_groups.items() if k is not None}

    unique_dependency_workloads = [
        group_workloads[0] for group_workloads in dependency_groups.values() if len(group_workloads) == 1
    ]

    # Count shared virtual environment groups
    shared_count = sum(len(workloads) for workloads in dependency_groups.values() if len(workloads) > 1)
    shared_groups_count = len([g for g in dependency_groups.values() if len(g) > 1])

    if no_dependency_workloads:
        print(f"\nNo dependency setup ({len(no_dependency_workloads)} workloads):")
        print("These workloads do not declare Python dependencies:")
        for workload in sorted(no_dependency_workloads):
            print(f"  • {workload}")

    if unique_dependency_workloads:
        print(f"\nIndividual installations ({len(unique_dependency_workloads)} workloads):")
        print("Each workload will have its own virtual environment:")
        for workload in sorted(unique_dependency_workloads):
            print(f"  • {workload}")

    if shared_count > 0:
        print(f"\nShared virtual environment groups ({shared_count} workloads in {shared_groups_count} groups):")
        print("These workloads share the same dependencies and will use a common virtual environment:")

        for group_workloads in dependency_groups.values():
            if len(group_workloads) > 1:
                print(f"  • {', '.join(sorted(group_workloads))}")

    print()


def _get_git_cache_dir(install_path: str) -> str:
    """Get the git cache directory path within the install directory.

    Args:
        install_path: Base installation directory

    Returns:
        str: Path to git cache directory
    """
    return os.path.join(install_path, ".cache", "git_repos")


def _derive_repo_name_from_url(repo_url: str) -> str:
    """Derive repository name from git URL, preserving original repo name.

    Args:
        repo_url: Git repository URL

    Returns:
        str: Repository name suitable for directory naming

    Raises:
        ValueError: If the URL is malformed and cannot derive a repository name
    """
    if not repo_url or not isinstance(repo_url, str):
        raise ValueError(f"Invalid repository URL: {repo_url}")

    # Remove trailing slashes and split by '/'
    url_parts = repo_url.rstrip('/').split('/')

    if not url_parts or not url_parts[-1]:
        raise ValueError(f"Cannot derive repository name from malformed URL: {repo_url}")

    repo_name = url_parts[-1].replace('.git', '')

    if not repo_name:
        raise ValueError(f"Cannot derive repository name from URL: {repo_url}")

    return repo_name


def _derive_cache_name_from_url(repo_url: str) -> str:
    """Derive a unique cache directory name from git URL to avoid collisions.

    Creates a filesystem-safe name that includes URL information to ensure
    different repositories with the same name don't collide in cache.

    Args:
        repo_url: Git repository URL

    Returns:
        str: Unique cache directory name

    Examples:
        https://github.com/NVIDIA/NeMo.git -> github.com_NVIDIA_NeMo
        git@gitlab.com:user/project.git -> gitlab.com_user_project
    """
    # Remove protocol and normalize
    url = repo_url.replace('git@', '').replace('https://', '').replace('http://', '')
    # Remove .git suffix and replace path separators with underscores
    url = url.replace('.git', '').replace(':', '_').replace('/', '_')
    # Replace any other non-filesystem-safe characters with underscores
    return re.sub(r'[^\w\-_.]', '_', url)


def _commit_exists_in_repo(repo_path: str, commit: str) -> bool:
    """Check if a specific commit exists in a git repository.

    Args:
        repo_path: Path to git repository (can be bare)
        commit: Git commit hash or reference

    Returns:
        bool: True if commit exists, False otherwise
    """
    try:
        subprocess.run(['git', 'cat-file', '-e', commit], cwd=repo_path, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _repo_uses_lfs(repo_path: str, commit: str) -> bool:
    """Check if a repository uses git-lfs by examining .gitattributes at a specific commit.

    Args:
        repo_path: Path to git repository (can be bare or non-bare)
        commit: Git commit hash to check

    Returns:
        bool: True if repository uses git-lfs, False otherwise
    """
    try:
        # Check if .gitattributes exists and contains "filter=lfs"
        result = subprocess.run(
            ['git', 'show', f'{commit}:.gitattributes'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        # If .gitattributes exists and contains LFS filter configuration, repo uses LFS
        return result.returncode == 0 and 'filter=lfs' in result.stdout
    except (OSError, FileNotFoundError, PermissionError):
        # Handle cases where git command fails to execute or repo_path is inaccessible
        # Return False to allow installation to continue (will fail later if git is actually broken)
        return False


def _get_default_branch(repo_path: str, remote_name: str = 'cache') -> Optional[str]:
    """Get the default branch name for a remote.

    Args:
        repo_path: Path to git repository
        remote_name: Name of the remote (default: 'cache')

    Returns:
        Optional[str]: Default branch name (e.g., 'main', 'master'), or None if not found
    """
    try:
        result = subprocess.run(
            ['git', 'symbolic-ref', f'refs/remotes/{remote_name}/HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        # Output is like "refs/remotes/cache/main", extract the branch name
        full_ref = result.stdout.strip()
        prefix = f'refs/remotes/{remote_name}/'
        if full_ref.startswith(prefix):
            return full_ref[len(prefix) :]
        return None
    except subprocess.CalledProcessError:
        # Remote HEAD not set or remote doesn't exist
        return None


def _has_required_remotes(repo_path: str, required_remotes: List[str]) -> bool:
    """Check if a git repository has all required remotes configured.

    Args:
        repo_path: Path to git repository
        required_remotes: List of remote names that must exist

    Returns:
        bool: True if all required remotes exist, False otherwise
    """
    try:
        result = subprocess.run(['git', 'remote'], cwd=repo_path, check=True, capture_output=True, text=True)
        existing_remotes = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
        return all(remote in existing_remotes for remote in required_remotes)
    except subprocess.CalledProcessError:
        return False


def _is_valid_git_repo(repo_path: str, is_bare: bool = False, required_remotes: Optional[List[str]] = None) -> bool:
    """Check if a directory contains a valid git repository.

    Uses git --git-dir to check repository validity without traversing up the tree.
    Optionally validates that required remotes are configured.

    Args:
        repo_path: Path to check for git repository
        is_bare: True if checking a bare repository, False for normal repository
        required_remotes: List of remote names that must exist (optional)

    Returns:
        bool: True if path contains a valid git repository (and optionally with required remotes), False otherwise
    """
    if not os.path.exists(repo_path):
        return False

    try:
        # For bare repos, the repo_path itself is the .git directory
        # For normal repos, look for .git subdirectory
        git_dir = repo_path if is_bare else os.path.join(repo_path, '.git')

        if not os.path.exists(git_dir):
            return False

        # Use git --git-dir to validate the repository
        subprocess.run(
            ['git', '--git-dir', git_dir, 'rev-parse', '--git-dir'], check=True, capture_output=True, text=True
        )

        # Check required remotes if specified
        if required_remotes:
            if not _has_required_remotes(repo_path, required_remotes):
                logger.debug(f"Repository at {repo_path} missing required remotes: {required_remotes}")
                return False

        return True
    except subprocess.CalledProcessError:
        return False


def _cleanup_invalid_git_repo(
    repo_path: str, is_bare: bool = False, description: str = "repository", required_remotes: Optional[List[str]] = None
) -> None:
    """Clean up invalid git repository if it exists.

    Validates git repository and removes it if invalid, with proper error handling.
    Optionally validates that required remotes are configured.

    Args:
        repo_path: Path to check and potentially clean up
        is_bare: True if checking a bare repository, False for normal repository
        description: Description for logging (e.g., "cache", "target repository")
        required_remotes: List of remote names that must exist (optional)
    """
    if os.path.exists(repo_path) and not _is_valid_git_repo(
        repo_path, is_bare=is_bare, required_remotes=required_remotes
    ):
        logger.warning(f"Invalid git {description} found at {repo_path}, cleaning up")
        try:
            shutil.rmtree(repo_path)
        except OSError as e:
            logger.warning(f"Failed to remove invalid {description} directory {repo_path}: {e}")


def _clone_or_get_from_cache(name: str, repo_url: str, install_path: str, target_dir: str, commit: str) -> str:
    """Clone repository to cache if needed, then clone from cache to target directory.

    Uses mirror repository caching for efficient git operations. Creates a mirror in cache,
    then clones from cache to target with proper remote setup. Only fetches cache if the
    required commit is missing.

    Args:
        name: Repository name for display purposes
        repo_url: Git repository URL
        install_path: Base installation path (for cache location)
        target_dir: Final destination directory
        commit: Git commit hash or reference needed

    Returns:
        str: Path to the cloned repository in target directory
    """
    cache_dir = _get_git_cache_dir(install_path)
    cache_name = _derive_cache_name_from_url(repo_url)
    repo_name = _derive_repo_name_from_url(repo_url)
    cache_path = os.path.join(cache_dir, cache_name + '.git')  # Mirror repos use .git suffix
    target_path = os.path.join(target_dir, repo_name)

    # Validate and clean up cache if it exists but is invalid
    _cleanup_invalid_git_repo(cache_path, is_bare=True, description="cache repository")

    # Ensure repo is in cache
    if not os.path.exists(cache_path):
        logger.debug(f"Creating mirror of {repo_url} in cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            subprocess.run(
                ['git', 'clone', '--mirror', repo_url, cache_path], check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cache {repo_url}: {e.stderr}")
            raise
    else:
        logger.debug(f"Using cached repository for {name}")

        # Check if required commit exists in cache, fetch if not
        if not _commit_exists_in_repo(cache_path, commit):
            logger.debug(f"Commit {commit} not in cache, fetching updates for {name}")
            try:
                subprocess.run(['git', 'fetch', 'origin'], cwd=cache_path, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to update cache for {name} for required commit {commit}: {e.stderr}")
                raise

            # Verify commit exists after fetch
            if not _commit_exists_in_repo(cache_path, commit):
                logger.error(
                    f"Commit {commit} not found in cache for {name} even after fetching from origin. "
                    f"Verify the commit exists in the repository at {repo_url}"
                )
                raise ValueError(f"Commit {commit} does not exist in repository {repo_url}")

    # Validate and clean up target if it exists but is invalid
    # For workload repos, require 'cache' remote to ensure proper cache-based workflow
    _cleanup_invalid_git_repo(target_path, is_bare=False, description="target repository", required_remotes=['cache'])

    # Clone from cache to target
    if not os.path.exists(target_path):
        logger.debug(f"Cloning {name} from cache to target")
        try:
            subprocess.run(
                ['git', 'clone', '--no-checkout', '-o', 'cache', cache_path, target_path],
                check=True,
                capture_output=True,
                text=True,
            )

            # Add origin back for dev workflows.
            subprocess.run(
                ['git', 'remote', 'add', 'origin', repo_url],
                cwd=target_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # For git-lfs repos, we need to set up proper remote tracking
            # Check if this repo uses LFS by examining .gitattributes
            uses_lfs = _repo_uses_lfs(cache_path, commit)
            if uses_lfs:
                logger.debug(f"Repository {name} uses git-lfs, configuring remote tracking")

                # Fetch from origin to set up remote-tracking branches and establish credentials
                # This is needed for git-lfs to work correctly with authentication
                subprocess.run(
                    ['git', 'fetch', 'origin'],
                    cwd=target_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )

                # Update default branch tracking to use origin instead of cache
                # Critical: branch.*.remote determines which remote git-lfs uses for authentication
                default_branch = _get_default_branch(target_path, 'cache')
                if default_branch:
                    try:
                        subprocess.run(
                            ['git', 'config', f'branch.{default_branch}.remote', 'origin'],
                            cwd=target_path,
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError:
                        logger.debug(f"Could not update branch.{default_branch}.remote for {name}")
                        pass
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {name} from cache: {e.stderr}")
            raise
    else:
        logger.debug(f"Repository {name} already exists, fetching from cache")

        # Check if commit already exists in target
        if not _commit_exists_in_repo(target_path, commit):
            # Commit missing, need to fetch from cache
            try:
                subprocess.run(['git', 'fetch', 'cache'], cwd=target_path, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to fetch cache updates for {name} for required commit {commit}: {e.stderr}")
                raise

            # Verify commit exists after fetch
            if not _commit_exists_in_repo(target_path, commit):
                logger.error(
                    f"Commit {commit} not found in target repository {target_path} even after fetching from cache. "
                    f"This may indicate cache corruption or an invalid commit hash."
                )
                raise ValueError(f"Commit {commit} does not exist in target repository for {name}")

    return target_path


def clone_git_repos(git_deps: Dict[str, Any], target_dir: str, install_path: str):
    """Clone git repositories into the target directory.

    Uses mirror repository caching to avoid repeated downloads when LLMB_DISABLE_GIT_CACHE is not set.
    Repositories are cached as mirror clones in {install_path}/.cache/git_repos/ and then cloned
    from cache to target_dir for efficient git operations.

    Args:
        git_deps: Dictionary of git dependencies with repo info
        target_dir: Target directory for cloned repositories
        install_path: Base installation path for cache location
    """
    if not git_deps:
        return

    # Check if caching is disabled
    cache_disabled = os.environ.get('LLMB_DISABLE_GIT_CACHE', '').lower() in ('1', 'true', 'yes')

    if cache_disabled:
        logger.info("Git caching disabled via LLMB_DISABLE_GIT_CACHE")
        use_cache = False
    else:
        use_cache = True

    print(f"Preparing git repositories for {os.path.basename(target_dir)}...")

    for name, repo_info in git_deps.items():
        repo_url = repo_info['url']
        commit = repo_info['commit']

        if use_cache:
            clone_path = _clone_or_get_from_cache(name, repo_url, install_path, target_dir, commit)
        else:
            # Direct cloning without cache
            repo_name = _derive_repo_name_from_url(repo_url)
            clone_path = os.path.join(target_dir, repo_name)
            print(f"  Cloning {name}...")

            # Validate and clean up existing repository if it's invalid
            _cleanup_invalid_git_repo(clone_path, is_bare=False, description="repository")

            if not os.path.exists(clone_path):
                try:
                    subprocess.run(
                        ['git', 'clone', '--no-checkout', repo_url, clone_path],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to clone {repo_url}: {e.stderr}")
                    raise
            else:
                try:
                    subprocess.run(
                        ['git', 'fetch', 'origin'], cwd=clone_path, check=True, capture_output=True, text=True
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to fetch {repo_url}: {e.stderr}")
                    raise

        # Checkout the specific commit
        logger.debug(f"Checking out commit {commit} for {name}")
        try:
            subprocess.run(
                ['git', 'checkout', '-f', commit], cwd=clone_path, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout commit {commit} in {clone_path}: {e.stderr}")
            raise

        # Update submodules if specified
        if repo_info.get('submodules', False):
            logger.debug(f"Checking out submodules for {name}")
            try:
                subprocess.run(
                    ['git', 'submodule', 'update', '--init', '--recursive'],
                    cwd=clone_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to checkout submodules for {name}: {e.stderr}")
                raise

        print(f"  ✓ {name}")


def _build_pip_install_command(
    venv_type: str, venv_path: str, editable: bool, package_str: str, compile_bytecode: bool = False
) -> List[str]:
    """Build the appropriate pip install command based on environment type.

    Args:
        venv_type: The virtual environment type ('uv', 'venv', or 'conda')
        venv_path: Path to the virtual environment
        editable: Whether this is an editable install
        package_str: The package specification to install
        compile_bytecode: Whether to compile bytecode for the package, only used for uv

    Returns:
        List of command arguments for pip install
    """
    # Check for pip fallback workaround
    use_pip_fallback = os.environ.get('LLMB_USE_PIP_FALLBACK', '').lower() in ('1', 'true', 'yes')

    if venv_type == 'uv' and not use_pip_fallback:
        # uv does not compile bytecode by default, so we enable it for better performance
        command = ['uv', 'pip', 'install']
        if compile_bytecode:
            command.append('--compile-bytecode')
    else:
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        command = [pip_path, 'install', '--no-cache-dir']

    if editable:
        command.append('-e')
    command.append(package_str)

    return command


def install_dependencies(
    venv_path: str, venv_type: str, dependencies: Dict[str, Any], workload_clone_path: str, env: Dict[str, str]
):
    """Install dependencies into a virtual environment.

    This function handles two types of dependencies from the workload's metadata:
    1. Git repositories: For repos that need to be installed via a script, this
       function runs the specified script from the locally cloned repository.
    2. Pip packages: Installs all packages listed under the 'pip' key. This
       can include standard packages from PyPI and direct-from-git installs.

    This function works for both individual workloads and groups of workloads
    that share the same dependencies.

    Args:
        venv_path: The path to the virtual environment.
        venv_type: The type of virtual environment ('venv', 'conda', or 'uv').
        dependencies: The fully resolved dependency dictionary.
        workload_clone_path: Path to the directory where git repos are cloned.
        env: The environment dictionary for running subprocesses.
    """
    print(f"Installing dependencies into venv: {venv_path}")

    # Install git repositories that require a script first
    git_deps = dependencies.get('git', {})
    if git_deps:
        print("  Installing git repositories via script...")
        for name, repo_info in git_deps.items():
            install_method = repo_info.get('install_method', {})
            install_type = install_method.get('type')

            # Get original repo name from URL
            repo_url = repo_info['url']
            repo_name_from_url = repo_url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join(workload_clone_path, repo_name_from_url)

            if install_type == 'script':
                print(f"    - Installing {name} using script '{install_method['path']}'...")
                try:
                    script_path = os.path.join(repo_path, install_method['path'])
                    os.chmod(script_path, 0o755)
                    subprocess.run([script_path], cwd=repo_path, check=True, env=env)
                except subprocess.CalledProcessError:
                    print(f"Error installing git repository {name} via script.")
                    raise
            elif install_type == 'clone':
                print(f"    - Git repo '{name}' is set to 'clone' only, skipping install step.")
            elif install_type == 'pip':
                print(f"    - Git repo '{name}' will be installed via pip.")
            else:
                # We can log if a repo is listed but has no valid install method.
                print(f"    - Skipping git repo '{name}': no valid install method found.")

    # Install pip packages
    pip_deps = dependencies.get('pip', [])
    if pip_deps:
        print("  Installing pip packages...")

        num_pip_deps = len(pip_deps)
        for i, item in enumerate(pip_deps, start=1):
            editable = False
            cwd = None  # CWD for subprocess, default to None
            package_name = None

            if isinstance(item, str):
                # Compatibility for simple string package definitions ie 'scipy<1.13.0'
                package_str = item
            elif isinstance(item, dict):
                package_name = item['package']
                is_git_repo = 'url' in item and 'commit' in item
                editable = item.get('editable', False)
                install_target = item.get('install_target')

                if is_git_repo:
                    repo_name_from_url = item['url'].split('/')[-1].replace('.git', '')
                    repo_path = os.path.join(workload_clone_path, repo_name_from_url)

                    if install_target:
                        # Install from a local source path (e.g., '.[all]')
                        package_str = install_target
                        cwd = repo_path
                    elif editable:
                        # Editable install from a local path
                        package_str = repo_path
                    else:
                        # Install from git+https URL
                        package_str = f"git+{item['url']}@{item['commit']}#egg={package_name}"
                else:
                    # Standard package from PyPI
                    package_str = package_name
            else:
                continue

            print(f"    - {package_name if package_name else package_str} (Editable: {editable})")

            # uv specific: uv does not compile bytecode by default. It also recompiles all packages each time.
            # We only want to compile bytecode for the last package to avoid recompiling all packages each time.
            compile_bytecode = False
            if venv_type == 'uv' and i == num_pip_deps:
                compile_bytecode = True

            pip_command = _build_pip_install_command(venv_type, venv_path, editable, package_str, compile_bytecode)

            try:
                subprocess.run(pip_command, check=True, env=env, cwd=cwd)
            except subprocess.CalledProcessError:
                print(f"Error installing pip package {package_str}.")
                raise

    print("✓ Dependency installation complete.")

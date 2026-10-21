# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Tool management for LLMB installation.

This module provides functions for collecting, downloading, and installing
workload-specific tools like nsys with GPU-conditional versioning.
"""

import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Set, Tuple

from llmb_install.constants import (
    CUDA_CUPTI_BASE_URL,
    CUDA_CUPTI_FILENAME_PATTERNS,
    NSYS_SOURCES,
    SUPPORTED_TOOLS,
)
from llmb_install.utils.download import download_file
from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)


def get_required_tools(workloads: Dict[str, Dict[str, Any]], selected_keys: List[str]) -> Set[Tuple[str, str]]:
    """Collect unique (tool_name, version) tuples from selected workloads.

    Args:
        workloads: Dictionary of all available workloads (GPU-resolved)
        selected_keys: List of selected workload keys

    Returns:
        Set of (tool_name, version) tuples
        Empty set if no tools required
    """
    required_tools: Set[Tuple[str, str]] = set()

    for key in selected_keys:
        workload = workloads[key]
        tools_config = workload.get('tools', {})

        if not tools_config:
            continue

        # tools_config is already GPU-resolved to {tool_name: version} format
        for tool_name, version in tools_config.items():
            # Validate tool is supported
            if tool_name not in SUPPORTED_TOOLS:
                print(f"Warning: Unsupported tool '{tool_name}' in workload '{key}' - skipping")
                continue

            # Log every workload's tool requirement for debugging
            logger.debug(f"Workload '{key}' requires {tool_name} {version}")

            # Add to required tools (set deduplicates automatically)
            required_tools.add((tool_name, version))

    if required_tools:
        logger.debug(f"Collected {len(required_tools)} unique tool(s) across {len(selected_keys)} workload(s)")

    return required_tools


def fetch_and_install_tools(
    tools: Set[Tuple[str, str]],
    install_path: str,
    node_architecture: str,
) -> None:
    """Download and install tools into $LLMB_INSTALL/tools/.

    Args:
        tools: Set of (tool_name, version) tuples to install
        install_path: Base installation directory
        node_architecture: Selected node architecture ('x86_64' or 'aarch64')
    """
    if not tools:
        return

    tools_base_dir = os.path.join(install_path, "tools")
    os.makedirs(tools_base_dir, exist_ok=True)

    for tool_name, version in sorted(tools):
        tool_install_dir = os.path.join(tools_base_dir, tool_name, version)

        # Check if already installed by verifying the binary exists
        # (Don't just check directory - it could exist from failed install)
        if tool_name == 'nsys':
            verify_path = os.path.join(tool_install_dir, 'bin', 'nsys')
        elif tool_name == 'cuda_cupti_lib':
            verify_path = os.path.join(tool_install_dir, 'lib', 'libcupti.so')
        else:
            verify_path = None

        if verify_path and os.path.exists(verify_path):
            print(f"Skipping {tool_name} {version} -- already installed at {tool_install_dir}")
            continue

        print(f"\nInstalling {tool_name} {version}...")
        logger.debug(f"Installing {tool_name} {version} to {tool_install_dir}")

        try:
            # Dispatch to tool-specific installer
            _install_tool(tool_name, version, tool_install_dir, install_path, node_architecture)

            print(f"✓ Successfully installed {tool_name} {version}")

        except Exception as e:
            # Clean up directory on failure so we don't skip on resume
            try:
                if os.path.exists(tool_install_dir):
                    shutil.rmtree(tool_install_dir)
                    logger.debug(f"Cleaned up failed installation directory: {tool_install_dir}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup directory {tool_install_dir}: {cleanup_err}")

            print(f"Error installing {tool_name} {version}: {str(e)}")
            raise


def _install_tool(
    tool_name: str,
    version: str,
    install_dir: str,
    install_path: str,
    arch: str,
) -> None:
    """Dispatcher for tool-specific installers.

    Args:
        tool_name: Name of the tool to install
        version: Version string
        install_dir: Directory to install the tool into (e.g., $LLMB_INSTALL/tools/nsys/version)
        install_path: Base installation directory ($LLMB_INSTALL)
        arch: Architecture ('x86_64' or 'aarch64')
    """
    if tool_name == 'nsys':
        _install_nsys(version, install_dir, install_path, arch)
    elif tool_name == 'cuda_cupti_lib':
        _install_cuda_cupti_lib(version, install_dir, install_path, arch)
    else:
        raise ValueError(f"Unsupported tool: {tool_name}")


class NsysDownloadSpec(NamedTuple):
    """Resolved download location and TLS policy for an NSys version."""

    url: str
    cache_name: str
    verify_tls: bool


def _nsys_download_spec(version: str, arch: str) -> NsysDownloadSpec:
    """Resolve the download source, URL and cache name for an NSys version.

    Selects the first source in NSYS_SOURCES whose 'version_pattern' matches the
    pinned version, and formats its URL/cache templates for `arch`.

    Args:
        version: NSys version string with build (e.g., '2025.5.1.121-3638078')
        arch: Architecture ('x86_64' or 'aarch64')

    Returns:
        NsysDownloadSpec with the download URL, cache file name and whether to
        verify TLS for the server.

    Raises:
        ValueError: If no source matches the version, or the selected source
            has no filename pattern for `arch`.
    """
    for source in NSYS_SOURCES:
        if re.search(source.version_pattern, version):
            if arch not in source.filenames:
                raise ValueError(
                    f"Unsupported architecture for nsys {version} from source with pattern"
                    f" '{source.version_pattern}': {arch}"
                )
            filename = source.filenames[arch].format(version=version)
            return NsysDownloadSpec(
                url=source.url_template.format(base_url=source.base_url, version=version, filename=filename),
                cache_name=source.cache_name_template.format(version=version, filename=filename),
                verify_tls=source.verify_tls,
            )
    raise ValueError(f"No NSys download source matches version '{version}'. Check NSYS_SOURCES in constants.py.")


def _install_nsys(
    version: str,
    install_dir: str,
    install_path: str,
    arch: str,
) -> None:
    """Download and install nsys using the .run installer.

    Args:
        version: NSys version string with build (e.g., '2025.5.1.121-3638078')
        install_dir: Directory to install nsys into (e.g., $LLMB_INSTALL/tools/nsys/version)
        install_path: Base installation directory ($LLMB_INSTALL)
        arch: Architecture ('x86_64' or 'aarch64')
    """

    # Get cache directory for downloaded installers
    cache_dir = Path(install_path) / '.cache' / 'tools' / 'nsys'
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using cache directory: {cache_dir}")

    # Construct download URL and cached file path
    spec = _nsys_download_spec(version, arch)
    download_url = spec.url
    cached_installer = str(cache_dir / spec.cache_name)

    logger.debug(f"nsys installer URL: {download_url}")

    # Download installer to cache if not already present
    # We download to a temp file and rename on success, ensuring cache only has complete files
    if os.path.exists(cached_installer):
        logger.debug(f"Using cached installer: {cached_installer}")
        print("  Using cached installer")
    else:
        print(f"  Downloading from: {download_url}")
        if not spec.verify_tls:
            # Console logging runs at INFO by default, so logger.warning() reaches the user;
            # do not also print it or it appears twice.
            logger.warning(
                f"Downloading NSys {version} from {download_url} with TLS certificate verification disabled."
            )
        logger.debug(f"Downloading to cache: {cached_installer}")
        # Download to temp file first, then move to final location on success
        temp_download = cached_installer + '.tmp'
        try:
            download_file(download_url, temp_download, verify=spec.verify_tls)
            os.rename(temp_download, cached_installer)
            os.chmod(cached_installer, 0o755)
            logger.debug("Download complete")
        finally:
            # Clean up temp file if it still exists (happens on failure/interruption)
            # If download succeeded, file was already renamed so this is a no-op
            if os.path.exists(temp_download):
                os.remove(temp_download)
                logger.debug(f"Cleaned up incomplete download: {temp_download}")

    # Create temporary directory for installer intermediate files
    temp_dir = tempfile.mkdtemp(prefix='nsys_install_')
    logger.debug(f"Created temp directory for installer: {temp_dir}")

    try:
        # Run installer with proper flags
        # --target: where installer extracts temporary files
        # -- -noprompt: no user interaction
        # -- -targetpath: final installation location
        install_cmd = [
            'bash',
            cached_installer,
            '--target',
            temp_dir,
            '--',
            '-noprompt',
            '-targetpath',
            install_dir,
        ]

        logger.debug(f"Running installer with targetpath: {install_dir}")
        print("  Running installer...")
        subprocess.run(install_cmd, check=True, capture_output=True, text=True)

        # Verify installation
        nsys_binary = os.path.join(install_dir, 'bin', 'nsys')
        if not os.path.exists(nsys_binary):
            raise RuntimeError(f"Installation succeeded but nsys binary not found at {nsys_binary}")

        logger.debug(f"Installation verified: {nsys_binary}")

    except subprocess.CalledProcessError as e:
        e.add_note("nsys installer failed.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during nsys installation: {e}")
        raise
    finally:
        # Clean up temporary directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")


def _install_cuda_cupti_lib(
    version: str,
    install_dir: str,
    install_path: str,
    arch: str,
) -> None:
    """Download and extract CUDA CUPTI library archive.

    Args:
        version: CUDA CUPTI version string (e.g., '13.0.85')
        install_dir: Directory to install into (e.g., $LLMB_INSTALL/tools/cuda_cupti_lib/version)
        install_path: Base installation directory ($LLMB_INSTALL)
        arch: Architecture ('x86_64' or 'aarch64')
    """
    if arch not in CUDA_CUPTI_FILENAME_PATTERNS:
        raise ValueError(f"Unsupported architecture for cuda_cupti_lib: {arch}")

    # Get cache directory for downloaded archives
    cache_dir = Path(install_path) / '.cache' / 'tools' / 'cuda_cupti_lib'
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using cache directory: {cache_dir}")

    # Construct download URL and cached file path
    filename = CUDA_CUPTI_FILENAME_PATTERNS[arch].format(version=version)
    arch_path = 'linux-x86_64' if arch == 'x86_64' else 'linux-sbsa'
    download_url = f"{CUDA_CUPTI_BASE_URL}/{arch_path}/{filename}"
    cached_archive = str(cache_dir / filename)

    logger.debug(f"cuda_cupti_lib archive URL: {download_url}")

    # Download archive to cache if not already present
    if os.path.exists(cached_archive):
        logger.debug(f"Using cached archive: {cached_archive}")
        print("  Using cached archive")
    else:
        print(f"  Downloading from: {download_url}")
        logger.debug(f"Downloading to cache: {cached_archive}")
        temp_download = cached_archive + '.tmp'
        try:
            download_file(download_url, temp_download)
            os.rename(temp_download, cached_archive)
            logger.debug("Download complete")
        finally:
            if os.path.exists(temp_download):
                os.remove(temp_download)
                logger.debug(f"Cleaned up incomplete download: {temp_download}")

    # Extract archive to install directory, stripping top-level directory
    try:
        os.makedirs(install_dir, exist_ok=True)
        print("  Extracting archive...")
        logger.debug(f"Extracting {cached_archive} to {install_dir}")

        with tarfile.open(cached_archive, 'r:xz') as tar:
            # Strip the top-level directory from all members
            for member in tar.getmembers():
                # Remove first path component (e.g., 'cuda_cupti-linux-x86_64-13.0.85-archive/')
                parts = member.name.split('/', 1)
                if len(parts) > 1 and parts[1]:
                    member.name = parts[1]
                    tar.extract(member, path=install_dir)

        # Verify installation
        lib_path = os.path.join(install_dir, 'lib', 'libcupti.so')
        if not os.path.exists(lib_path):
            raise RuntimeError(f"Extraction succeeded but libcupti.so not found at {lib_path}")

        logger.debug(f"Installation verified: {lib_path}")

    except Exception as e:
        logger.error(f"Failed to extract cuda_cupti_lib archive: {e}")
        raise

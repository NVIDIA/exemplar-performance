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

"""NSys mount handler for llmb-run.

This module provides functionality to mount custom nsys versions into containers,
working around nsys version bugs in container images. This is a temporary workaround
for the current release.

The module:
- Reads nsys version requirements from workload metadata
- Constructs mount strings for enroot containers
- Validates nsys binary existence before mounting
- Handles GPU-conditional tool versioning
"""

import glob
import logging
import os
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger('llmb_run.nsys_mount_handler')

# Container image to internal nsys installation directory lookup table
# This maps container image URLs to their internal nsys installation directories
# We mount the entire installation directory, not just the binary
# Add new entries as needed for different container versions
CONTAINER_NSYS_INSTALL_DIRS: Dict[str, str] = {
    'nvcr.io#nvidia/nemo:25.07.01': '/usr/local/cuda-12.9/NsightSystems-cli-2025.1.1',
    'nvcr.io#nvidia/nemo:25.09.00': '/usr/local/cuda-12.9/NsightSystems-cli-2025.4.1',
    'nvcr.io#nvidia/nemo:25.11.01': '/usr/local/cuda-13.0/NsightSystems-cli-2025.5.1',
    'nvcr.io#nvidia/nemo:26.02.00': '/usr/local/cuda-13.0/NsightSystems-cli-2025.5.1',
    'nvcr.io#nvidia/nemo:26.02.01': '/usr/local/cuda-13.0/NsightSystems-cli-2026.1.0',
    'nvcr.io#nvidia/nemo:26.04.00': '/usr/local/cuda-13.1/NsightSystems-cli-2026.1.1/',
    'nvcr.io#nvidia/nemo:26.04.01': '/usr/local/cuda-13.1/NsightSystems-cli-2026.1.1/',
    'nvcr.io#nvidia/nemo:26.06.01': '/usr/local/cuda-13.2/NsightSystems-cli-2026.2.1',
}

# Container image to CUPTI library path lookup table
# Maps (container_image, architecture) to container CUPTI library path
# Note: Both path and filename differ by architecture
CONTAINER_CUPTI_PATHS: Dict[Tuple[str, str], str] = {
    (
        'nvcr.io#nvidia/nemo:25.09.00',
        'x86_64',
    ): '/usr/local/cuda-12.9/NsightSystems-cli-2025.4.1/target-linux-x64/libcupti.so.13.0',
    (
        'nvcr.io#nvidia/nemo:25.09.00',
        'aarch64',
    ): '/usr/local/cuda-12.9/NsightSystems-cli-2025.4.1/target-linux-sbsa-armv8/libcupti-sbsa.so.13.0',
    (
        'nvcr.io#nvidia/nemo:25.11.01',
        'x86_64',
    ): '/usr/local/cuda-13.0/NsightSystems-cli-2025.5.1/target-linux-x64/libcupti.so.13.0',
    (
        'nvcr.io#nvidia/nemo:25.11.01',
        'aarch64',
    ): '/usr/local/cuda-13.0/NsightSystems-cli-2025.5.1/target-linux-sbsa-armv8/libcupti-sbsa.so.13.0',
    (
        'nvcr.io#nvidia/nemo:26.02.00',
        'x86_64',
    ): '/usr/local/cuda-13.0/NsightSystems-cli-2025.5.1/target-linux-x64/libcupti.so.13.0',
    (
        'nvcr.io#nvidia/nemo:26.02.00',
        'aarch64',
    ): '/usr/local/cuda-13.0/NsightSystems-cli-2025.5.1/target-linux-sbsa-armv8/libcupti-sbsa.so.13.0',
    (
        'nvcr.io#nvidia/nemo:26.02.01',
        'x86_64',
    ): '/usr/local/cuda-13.0/NsightSystems-cli-2026.1.0/target-linux-x64/libcupti.so.13.0',
    (
        'nvcr.io#nvidia/nemo:26.02.01',
        'aarch64',
    ): '/usr/local/cuda-13.0/NsightSystems-cli-2026.1.0/target-linux-sbsa-armv8/libcupti-sbsa.so.13.0',
    # Add entries for other containers as needed
}


def get_tool_mounts(
    llmb_install: str,
    workload_metadata: dict,
    gpu_type: str,
    arch: str,
    profiling_enabled: bool,
) -> Optional[str]:
    """Get all tool mount strings for container (nsys and/or cupti).

    Returns comma-separated mount strings for RUN_CONF_MOUNTS.
    Only returns mounts when profiling is enabled (workaround for nsys profiling).

    Args:
        llmb_install: Path to LLMB installation directory
        workload_metadata: Workload metadata from metadata.yaml
        gpu_type: GPU type (e.g., 'h100', 'gb200')
        arch: Node architecture ('x86_64' or 'aarch64')
        profiling_enabled: Whether profiling is enabled

    Returns:
        Comma-separated mount strings or None if no mounts needed
    """
    if not profiling_enabled:
        # Tool mounts are only for profiling workarounds
        return None

    mounts = []

    # Try nsys mount (directory mount for full nsys installation)
    nsys_mount = get_nsys_mount(llmb_install, workload_metadata, gpu_type, profiling_enabled)
    if nsys_mount:
        mounts.append(nsys_mount)

    # Try cupti mount (single library file mount)
    # Note: nsys and cupti mounts are mutually exclusive in practice, but we handle both
    cupti_mount = get_cupti_mount(llmb_install, workload_metadata, gpu_type, arch, profiling_enabled)
    if cupti_mount:
        mounts.append(cupti_mount)

    if not mounts:
        return None

    # Return comma-separated mount strings
    mount_string = ','.join(mounts)
    logger.debug(f"Tool mounts: {mount_string}")
    return mount_string


def get_nsys_mount(
    llmb_install: str,
    workload_metadata: dict,
    gpu_type: str,
    profiling_enabled: bool,
) -> Optional[str]:
    """Get nsys mount string for mounting custom nsys version into container.

    This function determines if a workload requires a custom nsys version for the
    given GPU type, validates the installation exists, and returns the appropriate
    mount string.

    The resolution logic matches llmb-install behavior:
    1. Check if GPU type explicitly listed in tools.nsys.by_gpu → use that version
    2. Check if 'default' exists in tools.nsys.by_gpu → use default version
    3. Neither exists → return None (use container version, no mount)

    Args:
        llmb_install: Path to LLMB installation directory (e.g., $LLMB_INSTALL)
        workload_metadata: Workload metadata dictionary from metadata.yaml
        gpu_type: GPU type for this workload (e.g., 'h100', 'gb200')
        profiling_enabled: Whether profiling is enabled for this run

    Returns:
        Mount string in enroot format (host:container) or None if no mount needed

    Raises:
        FileNotFoundError: If profiling is enabled and required nsys binary is missing
    """
    # Check if workload specifies custom nsys version
    tools_config = workload_metadata.get('tools', {})
    if not tools_config or 'nsys' not in tools_config:
        # No tools section or no nsys specified - use container version
        logger.debug("No tools.nsys specified in metadata, using container nsys")
        return None

    # Resolve nsys version based on configuration format and GPU type
    # This follows the same priority as llmb-install: explicit GPU > default > None
    nsys_version = _resolve_tool_version(tools_config.get('nsys'), gpu_type, 'nsys')
    if not nsys_version:
        # This GPU type is not configured for custom nsys - use container version
        logger.debug(f"No nsys version resolved for GPU type '{gpu_type}', using container nsys")
        return None

    logger.debug(f"Resolved nsys version '{nsys_version}' for GPU type '{gpu_type}'")

    # Construct path to host nsys installation directory
    host_nsys_install_dir = os.path.join(llmb_install, 'tools', 'nsys', nsys_version)

    # Verify installation exists by checking for the binary
    host_nsys_binary = os.path.join(host_nsys_install_dir, 'bin', 'nsys')
    if not os.path.exists(host_nsys_binary):
        if profiling_enabled:
            raise FileNotFoundError(
                f"Required nsys version '{nsys_version}' not found at {host_nsys_binary}. "
                f"This version is required for profiling with this workload. "
                f"Please run llmb-install to install the required nsys version."
            )
        else:
            # Profiling not enabled, silently skip mount
            logger.debug(
                f"nsys binary not found at {host_nsys_binary}, but profiling is disabled. " f"Using container nsys."
            )
            return None

    # Resolve container image (may be GPU-conditional)
    container_image = _resolve_container_image(workload_metadata, gpu_type)
    if not container_image:
        logger.warning("Could not resolve container image from metadata, cannot mount nsys")
        return None

    # Lookup container nsys installation directory
    container_nsys_install_dir = _get_container_nsys_install_dir(container_image)
    if not container_nsys_install_dir:
        # Container version not in lookup table, cannot mount
        logger.warning(
            f"Container image '{container_image}' not found in nsys install directory lookup table. "
            f"Cannot mount custom nsys. Using container nsys."
        )
        return None

    # Construct mount string in enroot format: host_install_dir:container_install_dir
    mount_string = f"{host_nsys_install_dir}:{container_nsys_install_dir}"
    logger.info(f"Mounting custom nsys installation: {mount_string}")

    return mount_string


def get_cupti_mount(
    llmb_install: str,
    workload_metadata: dict,
    gpu_type: str,
    arch: str,
    profiling_enabled: bool,
) -> Optional[str]:
    """Get CUPTI library mount string for mounting custom library into container.

    Resolution logic:
    1. Check if GPU type explicitly listed in tools.cuda_cupti_lib.by_gpu → use that version
    2. Check if 'default' exists → use default version
    3. Neither exists → return None (use container version)

    Args:
        llmb_install: Path to LLMB installation directory
        workload_metadata: Workload metadata from metadata.yaml
        gpu_type: GPU type (e.g., 'h100', 'gb200')
        arch: Node architecture ('x86_64' or 'aarch64')
        profiling_enabled: Whether profiling is enabled

    Returns:
        Mount string in format "host_file:container_file" or None

    Raises:
        FileNotFoundError: If profiling enabled and required library missing
    """
    # Check if workload specifies custom cuda_cupti_lib version
    tools_config = workload_metadata.get('tools', {})
    if not tools_config or 'cuda_cupti_lib' not in tools_config:
        logger.debug("No tools.cuda_cupti_lib specified, using container version")
        return None

    # Resolve version using same logic as nsys
    cupti_version = _resolve_tool_version(tools_config.get('cuda_cupti_lib'), gpu_type, 'cuda_cupti_lib')
    if not cupti_version:
        logger.debug(f"No cuda_cupti_lib version resolved for GPU '{gpu_type}', using container version")
        return None

    logger.debug(f"Resolved cuda_cupti_lib version '{cupti_version}' for GPU '{gpu_type}'")

    # Find the actual versioned library file in host installation
    host_cupti_lib_dir = os.path.join(llmb_install, 'tools', 'cuda_cupti_lib', cupti_version, 'lib')
    host_cupti_lib = _find_cupti_library(host_cupti_lib_dir)

    if not host_cupti_lib:
        if profiling_enabled:
            raise FileNotFoundError(
                f"Required cuda_cupti_lib version '{cupti_version}' not found in {host_cupti_lib_dir}. "
                f"Please run llmb-install to install the required version."
            )
        else:
            logger.debug(
                f"cuda_cupti_lib not found in {host_cupti_lib_dir}, but profiling disabled. Using container version."
            )
            return None

    # Resolve container image
    container_image = _resolve_container_image(workload_metadata, gpu_type)
    if not container_image:
        logger.warning("Could not resolve container image, cannot mount cuda_cupti_lib")
        return None

    # Lookup container CUPTI path
    container_cupti_path = _get_container_cupti_path(container_image, arch)
    if not container_cupti_path:
        logger.warning(
            f"Container '{container_image}' with arch '{arch}' not in CUPTI path lookup table. "
            f"Cannot mount custom cuda_cupti_lib. Using container version."
        )
        return None

    # Construct mount string: host_file:container_file
    mount_string = f"{host_cupti_lib}:{container_cupti_path}"
    logger.info(f"Mounting custom CUPTI library: {mount_string}")

    return mount_string


def _resolve_tool_version(tool_config: Union[str, dict], gpu_type: str, tool_name: str) -> Optional[str]:
    """Resolve tool version from metadata configuration.

    Handles both simple string format and GPU-conditional format.

    Args:
        tool_config: Tool configuration value from metadata
        gpu_type: GPU type to resolve version for
        tool_name: Name of tool (for logging)

    Returns:
        Resolved version string or None
    """
    if isinstance(tool_config, str):
        return tool_config

    if isinstance(tool_config, dict) and 'by_gpu' in tool_config:
        by_gpu = tool_config['by_gpu']

        # Try exact GPU match
        if gpu_type in by_gpu:
            return by_gpu[gpu_type]

        # Try default fallback
        if 'default' in by_gpu:
            logger.debug(f"Using default {tool_name} version for GPU '{gpu_type}'")
            return by_gpu['default']

        return None

    logger.warning(f"Unknown {tool_name} configuration format: {type(tool_config)}")
    return None


def _resolve_container_image(workload_metadata: dict, gpu_type: str) -> Optional[str]:
    """Resolve container image from metadata configuration.

    Handles both simple list format and GPU-conditional format:
    - Simple: container.images: ['image_url']
    - Conditional: container.images.by_gpu: {h100: ['image'], gb300: ['image']}

    Args:
        workload_metadata: Workload metadata dictionary from metadata.yaml
        gpu_type: GPU type to resolve image for

    Returns:
        Resolved container image URL or None if not found
    """
    container_config = workload_metadata.get('container', {})
    if not container_config:
        return None

    images_config = container_config.get('images')
    if not images_config:
        return None

    # Check if it's a GPU-conditional format
    if isinstance(images_config, dict) and 'by_gpu' in images_config:
        by_gpu = images_config['by_gpu']

        # First try exact GPU type match
        if gpu_type in by_gpu:
            gpu_images = by_gpu[gpu_type]
            return _extract_first_image(gpu_images)

        # Then try 'default' fallback
        if 'default' in by_gpu:
            default_images = by_gpu['default']
            return _extract_first_image(default_images)

        # No match found
        return None

    # Simple list format
    if isinstance(images_config, list):
        return _extract_first_image(images_config)

    # Unknown format
    logger.warning(f"Unknown container images configuration format: {type(images_config)}")
    return None


def _extract_first_image(images: list) -> Optional[str]:
    """Extract the first image URL from a list of images.

    Images can be either strings or dicts with 'url' key.

    Args:
        images: List of image specifications

    Returns:
        First image URL or None if list is empty
    """
    if not images:
        return None

    first_image = images[0]

    # Handle string format
    if isinstance(first_image, str):
        return first_image

    # Handle dict format with 'url' key
    if isinstance(first_image, dict) and 'url' in first_image:
        return first_image['url']

    logger.warning(f"Unknown image format: {type(first_image)}")
    return None


def _get_container_nsys_install_dir(container_image: str) -> Optional[str]:
    """Get the internal nsys installation directory for a container image.

    Args:
        container_image: Container image URL/path

    Returns:
        Path to nsys installation directory inside the container, or None if container not in lookup table
    """
    # Direct lookup
    if container_image in CONTAINER_NSYS_INSTALL_DIRS:
        return CONTAINER_NSYS_INSTALL_DIRS[container_image]

    # Try to extract just the image part without potential tags/variants
    # Handle cases like 'nvcr.io#nvidia/nemo:25.07.01-something'
    for known_image, install_dir in CONTAINER_NSYS_INSTALL_DIRS.items():
        if known_image in container_image:
            logger.debug(f"Partial match: '{container_image}' matches '{known_image}'")
            return install_dir

    # No match found
    return None


def _find_cupti_library(lib_dir: str) -> Optional[str]:
    """Find the versioned CUPTI library file in a directory.

    Looks for files matching libcupti*.so.* pattern (not symlinks).

    Args:
        lib_dir: Directory to search for library

    Returns:
        Full path to versioned library file, or None if not found
    """
    if not os.path.exists(lib_dir):
        logger.debug(f"Library directory does not exist: {lib_dir}")
        return None

    # Find versioned library files (e.g., libcupti.so.2025.3.1)
    pattern = os.path.join(lib_dir, 'libcupti*.so.*')
    candidates = glob.glob(pattern)

    # Filter out symlinks, keep only regular files
    real_files = [f for f in candidates if os.path.isfile(f) and not os.path.islink(f)]

    if not real_files:
        logger.debug(f"No versioned CUPTI library found in {lib_dir}")
        return None

    # Return first match (should only be one versioned file)
    cupti_lib = real_files[0]
    logger.debug(f"Found CUPTI library: {cupti_lib}")
    return cupti_lib


def _get_container_cupti_path(container_image: str, arch: str) -> Optional[str]:
    """Get the container CUPTI library path for a container image and architecture.

    Args:
        container_image: Container image URL
        arch: Architecture ('x86_64' or 'aarch64')

    Returns:
        Path to CUPTI library in container, or None if not in lookup table
    """
    # Direct lookup
    key = (container_image, arch)
    if key in CONTAINER_CUPTI_PATHS:
        return CONTAINER_CUPTI_PATHS[key]

    # Try partial match (handles image variants)
    for (known_image, known_arch), cupti_path in CONTAINER_CUPTI_PATHS.items():
        if known_arch == arch and known_image in container_image:
            logger.debug(f"Partial match: '{container_image}' matches '{known_image}' for arch '{arch}'")
            return cupti_path

    # No match found
    return None

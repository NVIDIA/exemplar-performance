# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Container image management for LLMB installation.

This module provides functions for managing container images including image collection,
filename generation, and image fetching using enroot.
"""

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from llmb_install.environment.detector import (
    detect_virtual_environment,
    get_clean_environment_for_subprocess,
    has_active_conda_environment,
)


def is_enroot_installed() -> bool:
    """Check if Enroot is installed and available in the system PATH.

    Returns:
        bool: True if enroot command is available, False otherwise
    """
    return shutil.which('enroot') is not None


def get_required_images(workloads: Dict[str, Dict[str, Any]], selected_keys: List[str]) -> Dict[str, str]:
    """Get the dictionary of container images required by the selected workloads.

    Args:
        workloads: Dictionary of all available workloads
        selected_keys: List of selected workload keys

    Returns:
        Dict[str, str]: Dictionary mapping image URLs to their desired filenames
    """
    # Use filename as key to deduplicate images that have different URL formats
    # but represent the same image (e.g., nvcr.io/nvidia/nemo vs nvcr.io#nvidia/nemo)
    filename_to_url = {}

    for key in selected_keys:
        workload = workloads[key]
        container_config = workload.get('container', {})

        # Handle image field which can be a string, list, or list of dicts
        images = container_config.get('images', [])

        if isinstance(images, str):
            # Simple string format - generate filename from URL
            filename = _generate_image_filename(images)
            filename_to_url[filename] = images
        elif isinstance(images, list):
            for image in images:
                if isinstance(image, str):
                    # Simple string format - generate filename from URL
                    filename = _generate_image_filename(image)
                    filename_to_url[filename] = image
                elif isinstance(image, dict):
                    url = image.get('url')
                    name = image.get('name')
                    if url:
                        if name:
                            filename = f"{name}.sqsh"
                            filename_to_url[filename] = url
                        else:
                            filename = _generate_image_filename(url)
                            filename_to_url[filename] = url

    # Convert back to URL -> filename mapping for compatibility with existing code
    required_images = {url: filename for filename, url in filename_to_url.items()}
    return required_images


def _generate_image_filename(image_url: str) -> str:
    """Generate a filename for a container image URL.

    Args:
        image_url: Container image URL (e.g., 'nvcr.io/nvidia/nemo:25.02.01' or 'nvcr.io#nvidia/nemo:sha256:abc123...')

    Returns:
        str: Generated filename ending with .sqsh
    """
    try:
        # Normalize URL separators - replace # with / for consistent parsing
        normalized_url = image_url.replace('#', '/')

        # Extract parent name, image name and version/digest from image URL
        parts = normalized_url.split('/')
        parent = parts[-2]  # Get 'nvidia' from URL
        name_and_tag = parts[-1]  # Get 'nemo:25.02.01' or 'nemo:sha256:abc123...'

        if ':' in name_and_tag:
            name, tag_or_digest = name_and_tag.split(':', 1)

            # Check if this is a SHA digest
            if tag_or_digest.startswith('sha256:'):
                # For SHA digests, use first 12 characters of the hash
                hash_part = tag_or_digest.split(':', 1)[1][:12]
                version = f"sha256-{hash_part}"
            else:
                # Regular tag
                version = tag_or_digest
        else:
            # No tag specified, use 'latest'
            name = name_and_tag
            version = "latest"

        return f"{parent}+{name}+{version}.sqsh"

    except (IndexError, ValueError):
        # Fallback: use a sanitized version of the full URL
        sanitized = image_url.replace('/', '_').replace(':', '_').replace('#', '_')
        return f"{sanitized}.sqsh"


def fetch_container_images(
    images: Dict[str, str],
    install_path: str,
    node_architecture: str,
    install_method: str,
    slurm_info: Dict[str, Any],
    shared_images_dir: Optional[str] = None,
) -> None:
    """Fetch container images using enroot and save them as sqsh files.

    Args:
        images: Dictionary mapping image URLs to desired filenames
        install_path: Directory where the sqsh files should be saved
        node_architecture: The selected node architecture ('x86_64' or 'aarch64')
        install_method: The selected installation method ('local' or 'slurm')
        slurm_info: Dictionary containing SLURM configuration information
        shared_images_dir: Optional path to a shared directory for images.

    The function will:
    1. Use provided filenames or generate them from image URLs
    2. Create sqsh files using enroot import
    3. Skip if sqsh file already exists
    4. If a shared_images_dir is provided, store images there and symlink to install_path/images.
    """
    local_images_dir = os.path.join(install_path, "images")
    os.makedirs(local_images_dir, exist_ok=True)

    if shared_images_dir:
        source_images_dir = shared_images_dir
        print(f"\nUsing shared image location: {source_images_dir}")
        os.makedirs(source_images_dir, exist_ok=True)
    else:
        source_images_dir = local_images_dir

    # Detect virtual environment and prepare clean environment for SLURM
    if detect_virtual_environment() and install_method == 'slurm':
        print("Detected virtual environment. Preparing clean environment for SLURM execution...")
        clean_env = get_clean_environment_for_subprocess()
    else:
        clean_env = None

    for image_url, filename in sorted(images.items()):
        try:
            source_sqsh_file = os.path.join(source_images_dir, filename)

            if os.path.exists(source_sqsh_file):
                print(f"Skipping {image_url} -- {source_sqsh_file} - file already exists.")
            else:
                cmd_args = ["enroot", "import", "-o", source_sqsh_file]
                if node_architecture:
                    cmd_args.extend(["-a", node_architecture])
                cmd_args.append(f"docker://{image_url}")

                if install_method == 'slurm':
                    # Validate required SLURM configuration fields
                    try:
                        account = slurm_info['slurm']['account']
                        cpu_partition = slurm_info['slurm']['cpu_partition']
                    except KeyError as e:
                        print(f"Error: Missing required SLURM configuration: {e}")
                        print("Please ensure your SLURM configuration includes account and cpu_partition.")
                        raise SystemExit(1) from e

                    srun_args = [
                        "srun",
                        f"--account={account}",
                        f"--partition={cpu_partition}",
                        "-t",
                        "35",
                        "--exclusive",
                        "--pty",
                    ]
                    if slurm_info['slurm'].get('cpu_partition_gres'):
                        srun_args.append(f"--gpus-per-node={slurm_info['slurm']['cpu_partition_gres']}")

                    full_cmd_args = srun_args + cmd_args
                    print(f"Submitting download via SLURM: {' '.join(full_cmd_args)}")
                else:
                    full_cmd_args = cmd_args
                    print(f"Downloading {image_url} locally: {' '.join(full_cmd_args)}")
                    clean_env = None  # Don't use clean env for local execution

                try:
                    subprocess.run(full_cmd_args, check=True, text=True, env=clean_env)
                    print(f"Successfully downloaded {source_sqsh_file}\n")
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to download {source_sqsh_file}: {str(e)}\n")
                    if install_method == 'slurm':
                        print(
                            "Please ensure the compute nodes you selected in the 'CPU_PARTITION' option are configured correctly, or try a local installation (if available)."
                        )
                        # Check if running from a virtual environment
                        if detect_virtual_environment():
                            if has_active_conda_environment():
                                venv_type = 'conda'
                            else:
                                venv_type = 'venv'
                            print(
                                f"Note: Running from a {venv_type} environment. Architecture mismatch between login and compute nodes may cause 'Exec format error'."
                            )
                            print("If using conda, please try venv instead with system python.")
                    else:
                        print("Ensure your system has appropriate resources or try a SLURM-based installation.")
                    raise SystemExit(1) from e

            # If using a shared directory, create/update symlink in the local install
            if shared_images_dir:
                symlink_path = os.path.join(local_images_dir, filename)
                abs_source_sqsh_file = os.path.abspath(source_sqsh_file)

                # Handle existing file/symlink at target
                if os.path.lexists(symlink_path):
                    if os.path.islink(symlink_path) and os.readlink(symlink_path) == abs_source_sqsh_file:
                        continue  # Symlink exists and is correct
                    os.remove(symlink_path)

                os.symlink(abs_source_sqsh_file, symlink_path)
                print(f"✓ Linked {filename} to shared image directory.")

        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            raise SystemExit(1) from e

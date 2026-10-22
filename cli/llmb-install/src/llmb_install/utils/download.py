# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Shared download helpers for LLMB installer."""

import urllib.error
import urllib.request
from pathlib import Path


def download_file(url: str, dest_path: str | Path) -> None:
    """Download a file from URL to a destination path with progress indication.

    Args:
        url: URL to download from
        dest_path: Local path to save file to

    Raises:
        RuntimeError: If the download fails for any reason.
    """
    try:

        def _progress_hook(block_num, block_size, total_size):
            """Simple progress indicator."""
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 50 == 0:  # Update every 50 blocks to avoid spam
                    print(f"  Progress: {percent}%", end='\r', flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
        print("  Progress: 100%")  # Final progress update

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download from {url}: HTTP {e.code} - {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download from {url}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url}: {str(e)}") from e

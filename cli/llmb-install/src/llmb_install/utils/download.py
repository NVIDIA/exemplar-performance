# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Shared download helpers for LLMB installer."""

import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

# Per socket operation (connect / each read), not a cap on total download time.
_DOWNLOAD_TIMEOUT_SECONDS = 60
# Large enough that per-read overhead (Python calls, progress updates) is negligible.
_CHUNK_SIZE = 1024 * 1024


def download_file(url: str, dest_path: str | Path, *, verify: bool = True) -> None:
    """Download a file from URL to a destination path with a progress bar.

    Progress is rendered with rich: a live bar on a terminal, and only the final
    state when stdout is not a TTY (CI, log files), so logs are not spammed.

    Args:
        url: URL to download from
        dest_path: Local path to save file to
        verify: Whether to verify the server's TLS certificate. Pass False only
            for servers whose CA/chain is not in the default trust store (see
            NSYS_SOURCES in constants.py); callers should log a warning when
            disabling verification.

    Raises:
        RuntimeError: If the download fails, or is cut short of its
            Content-Length (a truncated download would otherwise be cached
            as complete by callers).
    """
    context = ssl.create_default_context()
    if not verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    received = 0
    try:
        with (
            urllib.request.urlopen(url, context=context, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,
            open(dest_path, 'wb') as out_file,
            Progress(
                TextColumn("  {task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            ) as progress,
        ):
            total_size = int(response.headers.get('Content-Length') or 0)
            # total=None (no Content-Length) renders a pulsing bar instead of a percentage.
            task = progress.add_task(Path(urllib.parse.urlparse(url).path).name, total=total_size or None)
            while chunk := response.read(_CHUNK_SIZE):
                out_file.write(chunk)
                received += len(chunk)
                progress.advance(task, len(chunk))

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Failed to download from {url}: HTTP {e.code} - {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download from {url}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url}: {str(e)}") from e

    # read() returns b'' at EOF rather than raising, so a cut-off download would look
    # complete; refuse short downloads. Responses without a Content-Length (e.g.
    # chunked) cannot be checked.
    if total_size > 0 and received < total_size:
        raise RuntimeError(
            f"Failed to download from {url}: incomplete download, received {received} of {total_size} bytes"
        )

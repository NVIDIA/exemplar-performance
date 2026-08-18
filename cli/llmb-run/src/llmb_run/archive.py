# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Archive helpers for llmb-run."""

from __future__ import annotations

import datetime
import fnmatch
import os
import pathlib
import tarfile
from dataclasses import dataclass

import zstandard

from llmb_run.config_manager import ClusterConfig
from llmb_run.run_config import load_llmb_version

_ARCHIVE_PREFIX = "llmb-archive"

# TODO(temporary): microbenchmark_nccl uses the old sbatch launcher which places
# llmb-config and slurm output at the workload root instead of inside experiments/.
# Remove this constant and its usage in build_archive_file_list once the recipe
# aligns with the standard layout.
_NCCL_WORKLOAD_NAME = "microbenchmark_nccl"
_NCCL_TOP_LEVEL_PATTERNS = ("llmb-config_*.yaml", "slurm-*.out")
_ARCHIVE_EXCLUDED_PATTERNS = (
    "*.nsys-rep",
    "*_trace.json",
    "*.pt.trace.json",
    "*.tar.*",
    "dataset_1000_1000_*.txt",
)
_ARCHIVE_EXCLUDED_DIRS = {"code", "checkpoints", "nsys_profile", "torch_profile", "pytorch_profile"}


@dataclass(frozen=True)
class ArchiveStats:
    """Result metadata for an archive run."""

    output_path: pathlib.Path
    experiment_count: int
    timestamp: str


def utc_timestamp(now_utc: datetime.datetime | None = None) -> str:
    """Return compact UTC timestamp for filenames and archive roots."""
    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
    return now_utc.strftime("%Y%m%dT%H%M%SZ")


def default_archive_output(llmb_install: pathlib.Path, timestamp: str, release_version: str) -> pathlib.Path:
    """Return default output path under LLMB_INSTALL."""
    return llmb_install / f"{_ARCHIVE_PREFIX}_{release_version}_{timestamp}.tar.zst"


def build_archive_file_list(llmb_install: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    """Build list of archive entries as (source_path, workload-relative arcname)."""
    workloads_root = llmb_install / "workloads"
    if not workloads_root.exists():
        return []

    file_entries: list[tuple[pathlib.Path, str]] = []
    for workload_dir in sorted(workloads_root.iterdir()):
        if not workload_dir.is_dir():
            continue

        # TODO(temporary): remove with _NCCL_WORKLOAD_NAME / _NCCL_TOP_LEVEL_PATTERNS.
        if workload_dir.name == _NCCL_WORKLOAD_NAME:
            for f in sorted(workload_dir.iterdir()):
                if f.is_file() and any(fnmatch.fnmatch(f.name, p) for p in _NCCL_TOP_LEVEL_PATTERNS):
                    file_entries.append((f, f"{workload_dir.name}/{f.name}"))

        experiments_dir = workload_dir / "experiments"
        if not experiments_dir.is_dir():
            continue

        for root, dirs, files in os.walk(experiments_dir, followlinks=False):
            root_path = pathlib.Path(root)

            dirs[:] = [d for d in dirs if d not in _ARCHIVE_EXCLUDED_DIRS]

            symlink_dirs = [d for d in dirs if (root_path / d).is_symlink()]
            files_and_links = files + symlink_dirs

            for entry_name in files_and_links:
                source_path = root_path / entry_name
                name = source_path.name
                if any(fnmatch.fnmatch(name, pattern) for pattern in _ARCHIVE_EXCLUDED_PATTERNS):
                    continue

                relative_to_experiments = source_path.relative_to(experiments_dir)
                arcname = f"{workload_dir.name}/experiments/{relative_to_experiments.as_posix()}"
                file_entries.append((source_path, arcname))

    file_entries.sort(key=lambda item: item[1])
    return file_entries


def create_tar_zst(
    output_path: pathlib.Path, file_entries: list[tuple[pathlib.Path, str]], timestamp: str
) -> ArchiveStats:
    """Write a .tar.zst archive and return summary stats."""
    root_dir_name = f"{_ARCHIVE_PREFIX}-{timestamp}"
    cctx = zstandard.ZstdCompressor(level=5, threads=-1)
    experiment_count = sum(
        1 for _, arcname in file_entries if fnmatch.fnmatch(arcname.rsplit("/", 1)[-1], "llmb-config_*.yaml")
    )

    with output_path.open("wb") as raw_file:
        with cctx.stream_writer(raw_file) as compressed_stream:
            with tarfile.open(fileobj=compressed_stream, mode="w|", dereference=False) as tar:
                # Ensure archive always has one top-level folder even when no files match.
                root_info = tarfile.TarInfo(name=f"{root_dir_name}/")
                root_info.type = tarfile.DIRTYPE
                root_info.mode = 0o755
                root_info.mtime = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
                tar.addfile(root_info)

                for source_path, arcname in file_entries:
                    tar.add(source_path, arcname=f"{root_dir_name}/{arcname}", recursive=False)

    return ArchiveStats(
        output_path=output_path,
        experiment_count=experiment_count,
        timestamp=timestamp,
    )


def run_archive(cluster_config: ClusterConfig, output: str | None) -> ArchiveStats:
    """Create archive from installed workload experiment logs."""
    llmb_install_raw = cluster_config.llmb_install
    if not llmb_install_raw:
        raise ValueError("Missing required 'llmb_install' field in cluster configuration.")

    llmb_install = pathlib.Path(llmb_install_raw)
    timestamp = utc_timestamp()
    if output:
        output_path = pathlib.Path(output).expanduser()
    else:
        release_version = load_llmb_version(cluster_config.llmb_repo)
        if not release_version:
            raise ValueError(
                f"Unable to determine llmb_version from release.yaml under {cluster_config.llmb_repo}. "
                "Ensure release.yaml exists and contains llmb_version."
            )
        output_path = default_archive_output(llmb_install, timestamp, release_version)

    if output_path.exists():
        raise ValueError(f"Archive output already exists: {output_path}")

    if not output_path.parent.exists():
        raise ValueError(f"Output directory does not exist: {output_path.parent}")

    file_entries = build_archive_file_list(llmb_install)
    try:
        return create_tar_zst(output_path, file_entries, timestamp)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise

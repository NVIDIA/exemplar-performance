#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unpack and validate the AgentX dataset for the DSV4 workloads."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path


def main() -> None:
    workload_value = os.environ.get("LLMB_WORKLOAD", "").strip()
    if not workload_value:
        raise SystemExit("[prepare_dataset_recipe] ERROR: LLMB_WORKLOAD is not set")

    workload = Path(workload_value).resolve()
    dataset_root = workload / "dataset"
    archive_path = dataset_root / "agentic_coding.zip"
    dataset_dir = dataset_root / "agentic_coding"

    if archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                destination = (dataset_root / member.filename).resolve()
                if not destination.is_relative_to(dataset_root):
                    raise SystemExit(f"[prepare_dataset_recipe] ERROR: unsafe archive member: {member.filename}")
            archive.extractall(dataset_root)

    if not dataset_dir.is_dir() or not any(dataset_dir.iterdir()):
        raise SystemExit(f"[prepare_dataset_recipe] ERROR: dataset not found: {dataset_dir}")

    print(f"[prepare_dataset_recipe] ready: {dataset_dir}")


if __name__ == "__main__":
    main()

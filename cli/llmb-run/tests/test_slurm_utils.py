# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from unittest.mock import patch

from llmb_run.slurm_utils import get_slurm_job_statuses


def test_get_slurm_job_statuses_passes_start_time_to_sacct():
    with patch(
        "llmb_run.slurm_utils.subprocess.run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
    ) as mock_run:
        assert get_slurm_job_statuses([3, 1, 3], start_time="2026-04-16") == {}

    cmd = mock_run.call_args.args[0]
    assert "--jobs=1,3" in cmd
    assert "--starttime=2026-04-16" in cmd

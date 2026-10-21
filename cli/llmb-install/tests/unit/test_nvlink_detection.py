# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NVLink topology parsing and detection."""

from types import SimpleNamespace
from unittest.mock import patch

from llmb_install.cluster.nvlink import (
    NvlDetection,
    detect_nvl_topology,
    parse_topology_block_sizes,
    uses_block_topology,
)
from llmb_install.constants import GPU_TYPE_TO_NUM_GPUS, NVL_DOMAINS_BY_GPU, SUPPORTED_GPU_TYPES

NVL72_TOPOLOGY = """\
BlockName=block161 BlockIndex=0 Nodes=nvl72d004-T[01-18] BlockSize=18
BlockName=block129 BlockIndex=7 Nodes=nvl72d210-T[01-09,11-18],fake00001 BlockSize=18
BlockName=block058 BlockIndex=13 Nodes=nvl72d211-T[01-06,08-18],fake00002 BlockSize=18
"""

NVL576_TOPOLOGY = (
    """\
BlockName=block001 BlockIndex=0 Nodes=nvl576d002-R10-T[01-09] BlockSize=9
BlockName=block048 BlockIndex=47 Nodes=nvl576d004-R16-T[01-09] BlockSize=9
"""
    "AggregatedBlock=block[001-016] BlockIndex=48 "
    "Nodes=nvl576d002-R01-T[01-09],nvl576d002-R10-T[01-09],"
    "nvl576d002-R16-T[01-09] BlockSize=144\n"
    """\
AggregatedBlock=block[017-032] BlockIndex=49 Nodes=nvl576d003-R01-T[01-09] BlockSize=144
AggregatedBlock=block[033-048] BlockIndex=50 Nodes=nvl576d004-R01-T[01-09] BlockSize=144
"""
)

H100_TOPOLOGY = """\
SwitchName=C1 Level=2 LinkSpeed=1 Nodes=cpu1-[01-02] Switches=S1,S2
SwitchName=S1 Level=1 LinkSpeed=1 Nodes=pool0-[01729-01760] Switches=L207-DH4
SwitchName=L207-DH4 Level=0 LinkSpeed=1 Nodes=pool0-[01729-01760]
"""


def test_parse_topology_block_sizes():
    assert parse_topology_block_sizes(NVL72_TOPOLOGY) == [18]
    assert parse_topology_block_sizes(NVL576_TOPOLOGY) == [9, 144]
    assert parse_topology_block_sizes(H100_TOPOLOGY) == []


def test_gpu_type_mappings_cover_supported_types():
    assert set(GPU_TYPE_TO_NUM_GPUS) == SUPPORTED_GPU_TYPES
    assert set(NVL_DOMAINS_BY_GPU) == SUPPORTED_GPU_TYPES


def test_uses_block_topology():
    assert uses_block_topology("gb200")
    assert uses_block_topology("gb300")
    assert not uses_block_topology("h100")
    assert not uses_block_topology("b200")
    assert not uses_block_topology("b300")


@patch("llmb_install.cluster.nvlink.subprocess.run")
def test_detect_nvl72(mock_run):
    mock_run.return_value = SimpleNamespace(stdout=NVL72_TOPOLOGY)
    assert detect_nvl_topology("gb200") == NvlDetection(
        block_sizes=[18], nvl_domain=72, reason="detected", gpu_type="gb200"
    )


@patch("llmb_install.cluster.nvlink.subprocess.run")
def test_detect_nvl576(mock_run):
    mock_run.return_value = SimpleNamespace(stdout=NVL576_TOPOLOGY)
    assert detect_nvl_topology("gb300") == NvlDetection(
        block_sizes=[9, 144], nvl_domain=576, reason="detected", gpu_type="gb300"
    )


@patch("llmb_install.cluster.nvlink.subprocess.run")
def test_detect_no_blocks(mock_run):
    mock_run.return_value = SimpleNamespace(stdout=H100_TOPOLOGY)
    assert detect_nvl_topology("h100").reason == "no_blocks"


@patch("llmb_install.cluster.nvlink.subprocess.run", side_effect=FileNotFoundError("scontrol"))
def test_detect_error(_mock_run):
    assert detect_nvl_topology("gb200") == NvlDetection(
        block_sizes=None, nvl_domain=None, reason="error", gpu_type="gb200", error="scontrol"
    )

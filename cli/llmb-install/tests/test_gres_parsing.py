#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for GRES parsing functionality in the LLMB installer.
"""

from typing import Optional

import pytest

from llmb_install.cluster.slurm import parse_gpu_gres


def _deprecated_parse_gpu_gres(gres_output: str) -> Optional[int]:
    """Parse SLURM GRES output to extract GPU count.

    This is a copy of the function from installer.py for testing purposes.

    Handles common formats like:
    - gpu:8
    - gpu:a100:8
    - gpu:8(S:0-1)
    - gpu:8,mib:100
    """
    if gres_output == '(null)' or 'gpu:' not in gres_output:
        return None

    try:
        # Extract everything after 'gpu:'
        gpu_part = gres_output.split('gpu:', 1)[1]

        # Remove comma-separated additional resources (e.g., ',mib:100')
        gpu_part = gpu_part.split(',')[0]

        # Remove parenthetical information (e.g., '(S:0-1)')
        gpu_part = gpu_part.split('(')[0]

        # Handle typed GPUs (e.g., 'a100:8' -> '8')
        if ':' in gpu_part:
            gpu_part = gpu_part.split(':')[-1]

        gpu_count = int(gpu_part.strip())

        # Sanity check for reasonable GPU counts
        if gpu_count < 1 or gpu_count > 8:
            raise ValueError(f"Unusual GPU count: {gpu_count}")

        return gpu_count

    except (ValueError, IndexError):
        return None


class TestGresParsingBasic:
    """Test basic GRES parsing functionality."""

    def test_null_gres(self):
        """Test parsing of (null) GRES entries."""
        assert parse_gpu_gres("(null)") is None

    def test_no_gpu_gres(self):
        """Test parsing of non-GPU GRES entries."""
        assert parse_gpu_gres("mib:100") is None
        assert parse_gpu_gres("memory:1000") is None
        assert parse_gpu_gres("cpu:24") is None

    def test_simple_gpu_gres(self):
        """Test parsing of simple GPU GRES entries."""
        assert parse_gpu_gres("gpu:1") == 1
        assert parse_gpu_gres("gpu:2") == 2
        assert parse_gpu_gres("gpu:4") == 4
        assert parse_gpu_gres("gpu:8") == 8

    def test_typed_gpu_gres(self):
        """Test parsing of typed GPU GRES entries."""
        assert parse_gpu_gres("gpu:a100:1") == 1
        assert parse_gpu_gres("gpu:a100:4") == 4
        assert parse_gpu_gres("gpu:a100:8") == 8
        assert parse_gpu_gres("gpu:h100:8") == 8
        assert parse_gpu_gres("gpu:gb200:4") == 4
        assert parse_gpu_gres("gpu:tesla:2") == 2

    def test_gpu_with_socket_info(self):
        """Test parsing of GPU GRES with socket information."""
        assert parse_gpu_gres("gpu:8(S:0-1)") == 8
        assert parse_gpu_gres("gpu:4(S:0)") == 4
        assert parse_gpu_gres("gpu:a100:8(S:0-1)") == 8
        assert parse_gpu_gres("gpu:h100:4(S:0-3)") == 4

    def test_gpu_with_additional_resources(self):
        """Test parsing of GPU GRES with additional resources like MIG."""
        assert parse_gpu_gres("gpu:4,mib:100") == 4
        assert parse_gpu_gres("gpu:8,mib:200") == 8
        assert parse_gpu_gres("gpu:a100:4,mib:100") == 4
        assert parse_gpu_gres("gpu:8,memory:1000") == 8

    def test_complex_gpu_gres(self):
        """Test parsing of complex GPU GRES entries."""
        assert parse_gpu_gres("gpu:a100:8(S:0-1),mib:100") == 8
        assert parse_gpu_gres("gpu:h100:4(S:0-3),memory:2000") == 4
        assert parse_gpu_gres("gpu:gb200:4(S:0-1),mib:200,memory:1000") == 4


class TestGresParsingEdgeCases:
    """Test edge cases and error conditions for GRES parsing."""

    def test_invalid_gpu_counts(self):
        """Test parsing of invalid GPU counts."""
        # GPU count too low
        assert parse_gpu_gres("gpu:0") is None

        # GPU count too high (above our limit of 8)
        assert parse_gpu_gres("gpu:16") is None
        assert parse_gpu_gres("gpu:32") is None
        assert parse_gpu_gres("gpu:64") is None

    def test_malformed_gres(self):
        """Test parsing of malformed GRES entries."""
        assert parse_gpu_gres("gpu:") is None
        assert parse_gpu_gres("gpu:abc") is None
        assert parse_gpu_gres("gpu:weird:format:here") is None
        assert parse_gpu_gres("gpu:a100:") is None
        assert parse_gpu_gres("gpu:a100:abc") is None

    def test_whitespace_handling(self):
        """Test parsing with various whitespace scenarios."""
        assert parse_gpu_gres("gpu: 4") == 4
        assert parse_gpu_gres("gpu:4 ") == 4
        assert parse_gpu_gres(" gpu:4") == 4  # Leading space is actually handled correctly
        assert parse_gpu_gres("gpu: 8 ") == 8

    def test_empty_and_invalid_input(self):
        """Test parsing of empty and invalid inputs."""
        assert parse_gpu_gres("") is None
        assert parse_gpu_gres("   ") is None
        assert parse_gpu_gres("not_gres_at_all") is None
        assert parse_gpu_gres("gpu") is None


class TestGresParsingRealWorldScenarios:
    """Test real-world SLURM GRES scenarios."""

    def test_h100_scenarios(self):
        """Test H100 GPU scenarios."""
        assert parse_gpu_gres("gpu:h100:8") == 8
        assert parse_gpu_gres("gpu:h100:8(S:0-1)") == 8
        assert parse_gpu_gres("gpu:h100:4,mib:100") == 4

    def test_gb200_scenarios(self):
        """Test GB200 GPU scenarios."""
        assert parse_gpu_gres("gpu:gb200:4") == 4
        assert parse_gpu_gres("gpu:gb200:4(S:0-1)") == 4
        assert parse_gpu_gres("gpu:gb200:2,memory:2000") == 2

    def test_b200_scenarios(self):
        """Test B200 GPU scenarios."""
        assert parse_gpu_gres("gpu:b200:8") == 8
        assert parse_gpu_gres("gpu:b200:4(S:0-1)") == 4

    def test_mig_scenarios(self):
        """Test MIG (Multi-Instance GPU) scenarios."""
        assert parse_gpu_gres("gpu:a100_3g.20gb:2") == 2
        assert parse_gpu_gres("gpu:a100_7g.40gb:1") == 1
        assert parse_gpu_gres("gpu:h100_3g.26gb:4") == 4

    def test_heterogeneous_cluster_scenarios(self):
        """Test scenarios from heterogeneous clusters."""
        # Different GPU types but should parse the count correctly
        assert parse_gpu_gres("gpu:v100:4") == 4
        assert parse_gpu_gres("gpu:a40:2") == 2
        assert parse_gpu_gres("gpu:rtx3090:8") == 8


class TestSinfoOutputParsing:
    """Test parsing of complete sinfo output lines."""

    def test_sinfo_line_parsing(self):
        """Test parsing of complete sinfo output lines (partition,gres format)."""
        test_cases = [
            ("batch*,gpu:4,mib:100", "batch", "gpu:4,mib:100", 4),
            ("compute,gpu:8(S:0-1)", "compute", "gpu:8(S:0-1)", 8),
            ("gpu-nodes,gpu:a100:8", "gpu-nodes", "gpu:a100:8", 8),
            ("gpu-gb200,gpu:gb200:4", "gpu-gb200", "gpu:gb200:4", 4),
            ("cpu-partition,(null)", "cpu-partition", "(null)", None),
            ("no-gpu,mib:100", "no-gpu", "mib:100", None),
        ]

        for line, expected_partition, expected_gres, expected_count in test_cases:
            parts = line.split(',', 1)
            assert len(parts) == 2

            partition = parts[0].strip().rstrip('*')
            gres_raw = parts[1].strip()

            assert partition == expected_partition
            assert gres_raw == expected_gres
            assert parse_gpu_gres(gres_raw) == expected_count


class TestGresParsingIntegration:
    """Integration tests for GRES parsing within the full SLURM info flow."""

    def test_boundary_values(self):
        """Test boundary values for GPU counts."""
        # Valid boundary values
        assert parse_gpu_gres("gpu:1") == 1  # Minimum
        assert parse_gpu_gres("gpu:8") == 8  # Maximum

        # Invalid boundary values
        assert parse_gpu_gres("gpu:0") is None  # Below minimum
        assert parse_gpu_gres("gpu:9") is None  # Above maximum

    def test_comprehensive_scenarios(self):
        """Test comprehensive real-world scenarios."""
        # Scenarios that should work
        working_scenarios = [
            "gpu:1",
            "gpu:2",
            "gpu:4",
            "gpu:8",  # Basic counts
            "gpu:a100:4",
            "gpu:h100:8",
            "gpu:gb200:4",  # Typed GPUs
            "gpu:8(S:0-1)",
            "gpu:4(S:0)",  # Socket info
            "gpu:4,mib:100",
            "gpu:8,memory:2000",  # Additional resources
            "gpu:a100:8(S:0-1),mib:100",  # Complex combinations
        ]

        for scenario in working_scenarios:
            result = parse_gpu_gres(scenario)
            assert result is not None, f"Failed to parse valid scenario: {scenario}"
            assert 1 <= result <= 8, f"Invalid GPU count {result} for scenario: {scenario}"

        # Scenarios that should fail
        failing_scenarios = [
            "(null)",
            "mib:100",
            "memory:1000",  # No GPU
            "gpu:0",
            "gpu:16",
            "gpu:32",  # Invalid counts
            "gpu:",
            "gpu:abc",
            "gpu:weird:format:here",  # Malformed
            "",
            "   ",
            "not_gres_at_all",  # Invalid input
        ]

        for scenario in failing_scenarios:
            result = parse_gpu_gres(scenario)
            assert result is None, f"Should have failed to parse invalid scenario: {scenario}"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for tasktrace-capture data structures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ─── Time split (parse_agent_time_split return) ─────────────────────────────


class TrialSplit(BaseModel):
    """Single trial timing breakdown."""

    status: str  # "full", "partial", or "invalid"
    wall_time_sec: float | None = None
    llm_time_sec: float | None = None
    tool_time_sec: float | None = None
    n_llm_calls: int | None = None
    pct_cpu: float | None = None
    pct_llm: float | None = None
    reward: float | None = None
    n_episodes: int | None = None
    verifier_error: str | None = None  # exception_type from result.json if verifier failed
    # Phase breakdown (from harbor trial timestamps)
    env_setup_sec: float | None = None
    agent_setup_sec: float | None = None
    agent_exec_sec: float | None = None
    verifier_sec: float | None = None


class TimeSplit(BaseModel):
    """Aggregated time split across trials for one concurrency level.

    When all trials are invalid, timing fields (pct_cpu, pct_llm, etc.)
    will be None. Consumers must check before using them.
    """

    pct_cpu: float | None = None
    pct_llm: float | None = None
    tool_time_sec: float | None = None
    wall_time_sec: float | None = None
    llm_time_sec: float | None = None
    n_llm_calls: float | None = None
    avg_reward: float | None = None
    n_full: int = 0
    n_partial: int = 0
    n_invalid: int = 0
    n_verifier_errors: int = 0
    verifier_errors: list[str] = []  # unique exception_type values
    # Trials where the verifier returned a reward (others got 0.0 in avg)
    n_verified: int = 0
    survival_rate: str = "0/0"
    trials: list[TrialSplit] = []
    # Average phase breakdown across valid trials
    avg_env_setup_sec: float | None = None
    avg_agent_setup_sec: float | None = None
    avg_agent_exec_sec: float | None = None
    avg_verifier_sec: float | None = None
    # Per-call latency distribution (from session JSONL timestamps)
    llm_latency_median: float | None = None
    llm_latency_p95: float | None = None
    llm_latency_p99: float | None = None
    llm_latency_max: float | None = None
    tool_latency_median: float | None = None
    tool_latency_p95: float | None = None
    tool_latency_p99: float | None = None
    tool_latency_max: float | None = None
    n_llm_calls_tracked: int | None = None
    n_tool_calls_tracked: int | None = None


# ─── Sweep data (the big dict passed around cli.py) ─────────────────────────


class CpuInfo(BaseModel):
    model: str = "unknown"
    cores_logical: int = 0


class BaselineCpu(BaseModel):
    duration_sec: int = 5
    avg_cpu_pct: float = 0.0
    avg_cores_active: float = 0.0
    peak_cores_active: int = 0
    n_cores: int = 0


class SweepData(BaseModel):
    """Top-level experiment data written to scaling_results.json."""

    cpu_info: CpuInfo = CpuInfo()
    dataset: str = ""
    model: str = ""
    api_base: str = ""
    timestamp: str = ""
    hostname: str = ""
    runtime: str = "enroot"  # "enroot" | "docker" — container backend used
    enroot_network_policy: str = "strict"
    agent_timeout: int = 3600
    n_repeats: int = 1
    time_splits: dict[str, Any] = {}
    suite_mode: bool = False
    suite_tasks: list[str] | None = None
    suite_dataset: str | None = None
    results: list = []
    baseline_cpu: BaselineCpu = BaselineCpu()
    system_info: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}

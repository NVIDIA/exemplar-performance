# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tasktrace_capture.runner module.

Covers: _compute_trials, _parse_iso, parse_agent_time_split, and real job data parsing.

Run:  pytest tests/test_runner.py -v
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from tasktrace_capture.runner import (
    _compute_trials,
    _parse_iso,
    get_found_suite_tasks,
    parse_agent_time_split,
)

requires_enroot = pytest.mark.skipif(shutil.which("enroot") is None, reason="enroot not available")
requires_real_data = pytest.mark.skipif(not os.environ.get("BENCH_TEST_DATA_DIR"), reason="BENCH_TEST_DATA_DIR not set")


# ===========================================================================
#  Helpers -- build fake Harbor job directories
# ===========================================================================


def _make_result_json(
    tmp_path,
    trial_name,
    *,
    started_at,
    finished_at,
    api_times_msec=None,
    omit_agent_execution=False,
    omit_api_times=False,
    reward=None,
    n_episodes=None,
    exception_info=None,
):
    """Write a minimal result.json under tmp_path/<trial_name>/result.json."""
    trial_dir = tmp_path / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    agent_exec = {} if omit_agent_execution else {"started_at": started_at, "finished_at": finished_at}

    meta = {}
    if api_times_msec is not None and not omit_api_times:
        meta["api_request_times_msec"] = api_times_msec
    if n_episodes is not None:
        meta["n_episodes"] = n_episodes

    result = {
        "agent_execution": agent_exec,
        "agent_result": {"metadata": meta},
    }
    if reward is not None:
        result["verifier_result"] = {"rewards": {"reward": reward}}
    if exception_info is not None:
        result["exception_info"] = exception_info

    with open(trial_dir / "result.json", "w") as f:
        json.dump(result, f)
    return trial_dir


# ===========================================================================
#  _compute_trials
# ===========================================================================


class TestComputeTrials:
    def test_none_defaults_to_concurrent(self):
        assert _compute_trials(None, 4) == 4

    def test_trials_equal_concurrent(self):
        assert _compute_trials(4, 4) == 4

    def test_trials_greater_than_concurrent(self):
        assert _compute_trials(8, 4) == 8

    def test_trials_less_than_concurrent_bumped(self):
        assert _compute_trials(2, 4) == 4

    def test_zero_trials_defaults(self):
        assert _compute_trials(0, 3) == 3

    def test_single_concurrent(self):
        assert _compute_trials(None, 1) == 1
        assert _compute_trials(1, 1) == 1

    def test_multiple_tasks_preserve_default_attempt_count(self):
        assert _compute_trials(None, 4, fill_slots=False) == 1

    def test_multiple_tasks_preserve_explicit_attempt_count(self):
        assert _compute_trials(2, 4, fill_slots=False) == 2


# ===========================================================================
#  _parse_iso
# ===========================================================================


class TestParseIso:
    def test_trailing_z(self):
        dt = _parse_iso("2026-04-02T17:27:56.900288Z")
        assert dt.tzinfo is not None
        assert dt.year == 2026

    def test_with_offset(self):
        dt = _parse_iso("2026-04-02T17:27:56.900288+00:00")
        assert dt.tzinfo is not None

    def test_naive(self):
        dt = _parse_iso("2026-04-02T17:27:56.900288")
        assert dt.year == 2026

    def test_roundtrip_precision(self):
        ts = "2026-04-02T05:22:23.866413Z"
        dt = _parse_iso(ts)
        assert dt.microsecond == 866413


# ===========================================================================
#  parse_agent_time_split
# ===========================================================================


class TestParseAgentTimeSplit:

    def test_nonexistent_dir(self):
        assert parse_agent_time_split(Path("/does/not/exist")) is None

    def test_none_dir(self):
        assert parse_agent_time_split(None) is None

    def test_empty_dir(self, tmp_path):
        assert parse_agent_time_split(tmp_path) is None

    def test_single_full_trial(self, tmp_path):
        """900s wall, 400s LLM -> 500s tool.  No agent_timeout -> always 'full'."""
        _make_result_json(
            tmp_path,
            "trial_a",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",  # 900s
            api_times_msec=[200_000, 200_000],  # 400s total LLM
        )
        r = parse_agent_time_split(tmp_path)
        assert r is not None
        assert r.n_full == 1
        assert r.n_partial == 0
        assert r.n_invalid == 0
        assert abs(r.tool_time_sec - 500.0) < 1
        assert abs(r.llm_time_sec - 400.0) < 1
        assert r.n_llm_calls == 2
        assert abs(r.pct_cpu - 55.6) < 0.5
        assert r.survival_rate == "1/1"

    def test_full_vs_partial_classification(self, tmp_path):
        """With agent_timeout=900, a 300s trial is 'full' (completed), a 900s trial is 'partial' (timed out)."""
        _make_result_json(
            tmp_path,
            "completed_trial",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:05:00Z",  # 300s  (<810 = 90% of 900)
            api_times_msec=[50_000],
        )
        _make_result_json(
            tmp_path,
            "timeout_trial",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",  # 900s (>= 810 = 90% of 900)
            api_times_msec=[100_000],
        )
        r = parse_agent_time_split(tmp_path, agent_timeout=900)
        assert r.n_full == 1
        assert r.n_partial == 1
        # Averages use all non-invalid trials (full + partial)
        assert abs(r.tool_time_sec - 525.0) < 1  # (250+800)/2
        assert abs(r.llm_time_sec - 75.0) < 1  # (50+100)/2

    def test_only_partial_used_when_no_full(self, tmp_path):
        """When all trials timed out (partial), we fall back to them."""
        _make_result_json(
            tmp_path,
            "p1",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",  # 900s (>= 810)
            api_times_msec=[100_000],
        )
        _make_result_json(
            tmp_path,
            "p2",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:14:00Z",  # 840s (>= 810)
            api_times_msec=[80_000],
        )
        r = parse_agent_time_split(tmp_path, agent_timeout=900)
        assert r.n_full == 0
        assert r.n_partial == 2
        assert r.tool_time_sec > 0

    def test_invalid_missing_timestamps(self, tmp_path):
        _make_result_json(
            tmp_path,
            "bad",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:00:00Z",
            api_times_msec=[100],
            omit_agent_execution=True,
        )
        r = parse_agent_time_split(tmp_path)
        assert r is not None
        assert r.n_invalid == 1
        assert r.survival_rate == "0/1"

    def test_no_api_times_still_has_wall_time(self, tmp_path):
        """No api_times but valid timestamps -> full with 100% tool time."""
        _make_result_json(
            tmp_path,
            "no_api",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            omit_api_times=True,
        )
        r = parse_agent_time_split(tmp_path)
        assert r.n_full == 1
        assert r.pct_cpu == 100.0
        assert r.pct_llm == 0.0
        assert abs(r.wall_time_sec - 900.0) < 1

    def test_invalid_zero_wall(self, tmp_path):
        _make_result_json(
            tmp_path,
            "zero",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:00:00Z",
            api_times_msec=[100],
        )
        r = parse_agent_time_split(tmp_path)
        assert r.n_invalid == 1

    def test_mixed_full_partial_invalid(self, tmp_path):
        _make_result_json(
            tmp_path,
            "completed",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:03:00Z",  # 180s — completed before timeout
            api_times_msec=[50_000],
        )
        _make_result_json(
            tmp_path,
            "timeout",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",  # 900s — hit timeout
            api_times_msec=[200_000, 100_000],
        )
        _make_result_json(
            tmp_path,
            "invalid",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            omit_agent_execution=True,
        )
        r = parse_agent_time_split(tmp_path, agent_timeout=900)
        assert r.n_full == 1
        assert r.n_partial == 1
        assert r.n_invalid == 1
        assert r.survival_rate == "1/3"

    def test_all_invalid_returns_zero_survival(self, tmp_path):
        _make_result_json(
            tmp_path,
            "bad1",
            started_at="x",
            finished_at="y",
            omit_agent_execution=True,
        )
        _make_result_json(
            tmp_path,
            "bad2",
            started_at="x",
            finished_at="y",
            api_times_msec=[100],
            omit_agent_execution=True,
        )
        r = parse_agent_time_split(tmp_path)
        assert r.n_invalid == 2
        assert r.survival_rate == "0/2"
        assert r.tool_time_sec is None

    def test_per_trial_details_in_output(self, tmp_path):
        _make_result_json(
            tmp_path,
            "t1",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            api_times_msec=[100_000],
        )
        r = parse_agent_time_split(tmp_path)
        assert len(r.trials) == 1
        assert r.trials[0].status == "full"
        assert r.trials[0].wall_time_sec is not None

    def test_no_agent_timeout_means_all_full(self, tmp_path):
        """Without agent_timeout, even short trials are classified 'full'."""
        _make_result_json(
            tmp_path,
            "short",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:00:10Z",  # 10s
            api_times_msec=[5000],
        )
        r = parse_agent_time_split(tmp_path)
        assert r.n_full == 1
        assert r.n_partial == 0

    def test_threshold_exactly_90pct(self, tmp_path):
        """A trial at exactly 90% of timeout is 'partial' (likely timed out)."""
        _make_result_json(
            tmp_path,
            "edge",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:13:30Z",  # 810s = 90% of 900
            api_times_msec=[100_000],
        )
        r = parse_agent_time_split(tmp_path, agent_timeout=900)
        assert r.n_partial == 1
        assert r.n_full == 0

    def test_nested_result_json(self, tmp_path):
        """result.json files can be nested several levels deep."""
        deep = tmp_path / "task__abc" / "agent"
        deep.mkdir(parents=True)
        result = {
            "agent_execution": {
                "started_at": "2026-04-02T00:00:00Z",
                "finished_at": "2026-04-02T00:15:00Z",
            },
            "agent_result": {"metadata": {"api_request_times_msec": [300_000, 200_000]}},
        }
        with open(deep / "result.json", "w") as f:
            json.dump(result, f)
        r = parse_agent_time_split(tmp_path)
        assert r.n_full == 1
        assert abs(r.tool_time_sec - 400.0) < 1

    def test_corrupt_json_is_invalid(self, tmp_path):
        trial_dir = tmp_path / "corrupt"
        trial_dir.mkdir()
        (trial_dir / "result.json").write_text("NOT JSON {{{")
        r = parse_agent_time_split(tmp_path)
        assert r.n_invalid == 1

    def test_reward_parsed_from_verifier_result(self, tmp_path):
        """Reward comes from verifier_result.rewards.reward, not top-level."""
        _make_result_json(
            tmp_path,
            "pass_trial",
            started_at="2026-04-03T07:00:00Z",
            finished_at="2026-04-03T07:14:18Z",
            api_times_msec=[3000] * 121,
            reward=1.0,
            n_episodes=121,
        )
        r = parse_agent_time_split(tmp_path)
        assert r is not None
        assert r.avg_reward == 1.0
        assert r.trials[0].reward == 1.0
        assert r.trials[0].n_episodes == 121

    def test_reward_none_when_missing(self, tmp_path):
        """Trials without verifier_result parse; reward is None.

        When EVERY valid trial has reward=None, ``avg_reward`` is None so the
        report can render ``n/a`` instead of a misleading ``0.000``. When at
        least one trial DID produce a reward, errored trials count as 0.0 in
        the average — that path is covered by the next test.
        """
        _make_result_json(
            tmp_path,
            "no_reward",
            started_at="2026-04-03T07:00:00Z",
            finished_at="2026-04-03T07:15:00Z",
            api_times_msec=[100_000],
        )
        r = parse_agent_time_split(tmp_path)
        # All trials had reward=None → avg_reward is None (renders as n/a)
        assert r.avg_reward is None
        assert r.trials[0].reward is None

    def test_avg_reward_across_trials(self, tmp_path):
        """avg_reward is the mean of per-trial rewards."""
        _make_result_json(
            tmp_path,
            "t1",
            started_at="2026-04-03T07:00:00Z",
            finished_at="2026-04-03T07:15:00Z",
            api_times_msec=[100_000],
            reward=1.0,
        )
        _make_result_json(
            tmp_path,
            "t2",
            started_at="2026-04-03T07:00:00Z",
            finished_at="2026-04-03T07:15:00Z",
            api_times_msec=[100_000],
            reward=0.0,
        )
        r = parse_agent_time_split(tmp_path)
        assert r.avg_reward == 0.5

    def test_percentages_consistent_with_absolutes(self, tmp_path):
        """Percentages must match tool_time/wall_time, not avg of per-trial %."""
        # Trial A: short wall, high LLM fraction
        _make_result_json(
            tmp_path,
            "short",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:00:10Z",  # 10s wall
            api_times_msec=[8000],  # 8s LLM, 2s tool
        )
        # Trial B: long wall, low LLM fraction
        _make_result_json(
            tmp_path,
            "long",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:01:30Z",  # 90s wall
            api_times_msec=[10_000],  # 10s LLM, 80s tool
        )
        r = parse_agent_time_split(tmp_path)
        # avg wall = 50s, avg llm = 9s, avg tool = 41s
        assert abs(r.wall_time_sec - 50.0) < 0.1
        assert abs(r.llm_time_sec - 9.0) < 0.1
        assert abs(r.tool_time_sec - 41.0) < 0.1
        # Percentages derived from averaged absolutes, not avg of per-trial %
        assert abs(r.pct_llm - 100 * 9.0 / 50.0) < 0.5  # 18%
        assert abs(r.pct_cpu - 100 * 41.0 / 50.0) < 0.5  # 82%

    def test_verifier_error_extracted(self, tmp_path):
        """Trial-level exception_type is captured in trial and aggregated.

        The field name is historical (``verifier_error``) but we now capture
        ANY top-level ``exception_info`` from result.json — including agent-
        side crashes that also happened to have a verifier_result. See B10
        in QA notes.
        """
        _make_result_json(
            tmp_path,
            "verifier_fail",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            api_times_msec=[100_000],
            exception_info={
                "exception_type": "AddTestsDirError",
                "exception_message": "Failed to add tests directory.",
            },
            # no reward= kwarg -> verifier_result absent
        )
        r = parse_agent_time_split(tmp_path)
        assert r.n_verifier_errors == 1
        assert r.verifier_errors == ["AddTestsDirError"]
        assert r.trials[0].verifier_error == "AddTestsDirError"
        assert r.trials[0].reward is None
        # All-erred → avg_reward is None (renders as n/a)
        assert r.avg_reward is None
        assert r.n_verified == 0

    def test_verifier_error_set_even_when_reward_present(self, tmp_path):
        """When verifier_result exists AND exception_info exists, capture the
        exception. Old behaviour ignored exception_info if a verifier_result
        was present, which silently swallowed agent-level crashes (codex
        NonZeroAgentExitCodeError, terminus-2 BadRequestError on cloud
        backends) where the verifier still happened to produce a reward.
        Renamed test from ``..._not_set_...`` because the semantics flipped
        with the B10 fix."""
        _make_result_json(
            tmp_path,
            "ok_with_exc",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            api_times_msec=[100_000],
            reward=1.0,
            exception_info={
                "exception_type": "NonZeroAgentExitCodeError",
                "exception_message": "Agent crashed mid-trial.",
            },
        )
        r = parse_agent_time_split(tmp_path)
        # Exception is now captured even though verifier produced a reward.
        assert r.n_verifier_errors == 1
        assert r.verifier_errors == ["NonZeroAgentExitCodeError"]
        assert r.trials[0].verifier_error == "NonZeroAgentExitCodeError"
        # The reward IS still captured — n_verified counts it.
        assert r.trials[0].reward == 1.0
        assert r.avg_reward == 1.0
        assert r.n_verified == 1

    def test_task_timeout_classified_as_partial(self, tmp_path):
        """Trial killed by Harbor's task-level AgentTimeoutError is 'partial',
        even when wall time is below tasktrace-capture's --agent-timeout threshold."""
        _make_result_json(
            tmp_path,
            "task_timeout",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:40:00Z",  # 2400s — hit task timeout
            api_times_msec=[100_000],
            exception_info={
                "exception_type": "AgentTimeoutError",
                "exception_message": "Agent execution timed out after 2400.0 seconds",
            },
        )
        r = parse_agent_time_split(tmp_path, agent_timeout=3600)
        assert r.n_partial == 1
        assert r.n_full == 0
        assert r.trials[0].status == "partial"
        assert r.survival_rate == "0/1"

    def test_task_timeout_with_reward_still_partial(self, tmp_path):
        """AgentTimeoutError + reward=1.0 → still 'partial'. Survival != correctness."""
        _make_result_json(
            tmp_path,
            "timeout_but_passed",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:40:00Z",  # 2400s
            api_times_msec=[100_000],
            reward=1.0,
            exception_info={
                "exception_type": "AgentTimeoutError",
                "exception_message": "Agent execution timed out after 2400.0 seconds",
            },
        )
        _make_result_json(
            tmp_path,
            "fast_success",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:05:00Z",  # 300s
            api_times_msec=[50_000],
            reward=1.0,
        )
        r = parse_agent_time_split(tmp_path, agent_timeout=3600)
        assert r.n_full == 1
        assert r.n_partial == 1
        assert r.survival_rate == "1/2"
        assert r.avg_reward == 1.0  # both trials passed, reward is independent

    def test_mixed_verifier_errors_and_rewards(self, tmp_path):
        """Mix of verifier failures and successes aggregates correctly."""
        _make_result_json(
            tmp_path,
            "fail1",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            api_times_msec=[100_000],
            exception_info={"exception_type": "AddTestsDirError", "exception_message": "fail"},
        )
        _make_result_json(
            tmp_path,
            "fail2",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            api_times_msec=[100_000],
            exception_info={"exception_type": "AddTestsDirError", "exception_message": "fail"},
        )
        _make_result_json(
            tmp_path,
            "success",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:15:00Z",
            api_times_msec=[100_000],
            reward=1.0,
        )
        r = parse_agent_time_split(tmp_path)
        assert r.n_verifier_errors == 2
        assert r.verifier_errors == ["AddTestsDirError"]
        # 2 errored trials count as 0.0, 1 success = 1.0 → avg = 1/3
        assert r.avg_reward == pytest.approx(0.333, abs=0.001)
        assert r.n_verified == 1

    def test_no_fallback_to_top_level_timestamps(self, tmp_path):
        """Trials without agent_execution timestamps are invalid, not inflated."""
        # Trial with agent_execution timestamps -> valid
        _make_result_json(
            tmp_path,
            "good",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:00:10Z",
            api_times_msec=[5000],
        )
        # Trial with only top-level timestamps (simulated by writing raw JSON)
        bad_dir = tmp_path / "bad_toplevel"
        bad_dir.mkdir()
        import json as _json

        with open(bad_dir / "result.json", "w") as f:
            _json.dump(
                {
                    "started_at": "2026-04-02T00:00:00Z",
                    "finished_at": "2026-04-02T00:10:00Z",
                    "agent_execution": {},
                    "agent_result": {"metadata": {"api_request_times_msec": [5000]}},
                },
                f,
            )
        r = parse_agent_time_split(tmp_path)
        # The bad trial should be invalid, not use the 600s top-level wall
        assert r.n_invalid == 1
        assert r.n_full == 1
        assert abs(r.wall_time_sec - 10.0) < 0.1


# ===========================================================================
#  Subprocess timeout derived from agent_timeout
# ===========================================================================


class TestSubprocessTimeout:
    """Verify run_experiment passes agent_timeout + 600 to Popen.communicate."""

    def _make_mock_popen(self, captured_timeout, captured_kwargs):
        """Create a mock Popen class that captures timeout and kwargs."""

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.cmd = cmd
                self.pid = os.getpid()  # use our own PID so os.getpgid works
                self.returncode = 0
                if isinstance(cmd, list) and cmd and cmd[0] == "harbor":
                    captured_kwargs.append(kwargs)

            def communicate(self, timeout=None):
                if self.cmd and self.cmd[0] == "harbor":
                    captured_timeout.append(timeout)
                return "", ""

        return MockPopen

    def test_default_timeout_is_agent_timeout_plus_600(self, tmp_path):
        """Default agent_timeout=3600 -> communicate timeout=4200."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        captured_timeout = []
        captured_kwargs = []
        mock_popen = self._make_mock_popen(captured_timeout, captured_kwargs)

        with patch.object(real_subprocess, "Popen", mock_popen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=3600,
                jobs_dir=tmp_path,
            )
        assert captured_timeout[0] == 4200  # 3600 + 600

    def test_custom_timeout_900(self, tmp_path):
        """agent_timeout=900 -> communicate timeout=1500."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        captured_timeout = []
        captured_kwargs = []
        mock_popen = self._make_mock_popen(captured_timeout, captured_kwargs)

        with patch.object(real_subprocess, "Popen", mock_popen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=900,
                jobs_dir=tmp_path,
            )
        assert captured_timeout[0] == 1500  # 900 + 600

    def test_popen_uses_start_new_session(self, tmp_path):
        """Harbor subprocess must use start_new_session=True for process group cleanup."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        captured_timeout = []
        captured_kwargs = []
        mock_popen = self._make_mock_popen(captured_timeout, captured_kwargs)

        with patch.object(real_subprocess, "Popen", mock_popen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=3600,
                jobs_dir=tmp_path,
            )
        assert captured_kwargs[0].get("start_new_session") is True


# ===========================================================================
#  HARBOR_ENROOT_CACHE default
# ===========================================================================


class TestHarborEnrootCache:
    """Verify run_experiment sets HARBOR_ENROOT_CACHE to .enroot/cache/ by default."""

    def test_harbor_enroot_cache_set_by_default(self, tmp_path, monkeypatch):
        """When HARBOR_ENROOT_CACHE is not in env, runner sets it to <cwd>/.enroot/cache/."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        monkeypatch.delenv("HARBOR_ENROOT_CACHE", raising=False)
        captured_env = []

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = os.getpid()
                self.returncode = 0
                if isinstance(cmd, list) and cmd and cmd[0] == "harbor":
                    captured_env.append(kwargs.get("env", {}))

            def communicate(self, timeout=None):
                return "", ""

        with patch.object(real_subprocess, "Popen", MockPopen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=3600,
                jobs_dir=tmp_path,
            )
        assert len(captured_env) == 1
        cache_path = captured_env[0].get("HARBOR_ENROOT_CACHE", "")
        assert cache_path.endswith(".enroot/cache")

    def test_harbor_enroot_cache_not_overridden(self, tmp_path, monkeypatch):
        """When user sets HARBOR_ENROOT_CACHE, runner does not override it."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        monkeypatch.setenv("HARBOR_ENROOT_CACHE", "/custom/path")
        captured_env = []

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = os.getpid()
                self.returncode = 0
                if isinstance(cmd, list) and cmd and cmd[0] == "harbor":
                    captured_env.append(kwargs.get("env", {}))

            def communicate(self, timeout=None):
                return "", ""

        with patch.object(real_subprocess, "Popen", MockPopen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=3600,
                jobs_dir=tmp_path,
            )
        assert captured_env[0].get("HARBOR_ENROOT_CACHE") == "/custom/path"


# ===========================================================================
#  Harbor --agent-timeout flag
# ===========================================================================


class TestHarborAgentTimeout:
    """Verify agent_timeout is enforced via subprocess timeout, not harbor CLI flag.

    Harbor does not have --agent-timeout (only --agent-timeout-multiplier which
    is a relative multiplier, not an absolute value). The enforcement mechanism
    is subprocess.run(timeout=agent_timeout+600).
    """

    def test_agent_timeout_not_in_harbor_cmd(self, tmp_path):
        """--agent-timeout must NOT appear in the harbor command (harbor doesn't support it)."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        captured_cmd = []

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = os.getpid()
                self.returncode = 0
                if isinstance(cmd, list) and cmd and cmd[0] == "harbor":
                    captured_cmd.extend(cmd)

            def communicate(self, timeout=None):
                return "", ""

        with patch.object(real_subprocess, "Popen", MockPopen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=900,
                jobs_dir=tmp_path,
            )
        assert "--agent-timeout" not in captured_cmd

    def test_subprocess_timeout_derived_from_agent_timeout(self, tmp_path):
        """Popen.communicate timeout = agent_timeout + 600 (enforcement mechanism)."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        captured_timeout = []

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = os.getpid()
                self.returncode = 0

            def communicate(self, timeout=None):
                captured_timeout.append(timeout)
                return "", ""

        with patch.object(real_subprocess, "Popen", MockPopen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=300,
                jobs_dir=tmp_path,
            )
        assert captured_timeout[0] == 900  # 300 + 600


# ===========================================================================
#  Post-harbor container cleanup and abandoned trial detection
# ===========================================================================


class TestCleanupHarborContainers:
    """Verify _cleanup_harbor_containers removes hb__* containers."""

    def test_cleanup_calls_enroot_remove(self):
        from unittest.mock import patch

        from tasktrace_capture.runner import _cleanup_harbor_containers

        mock_calls = []

        def mock_run(cmd, **kwargs):
            mock_calls.append(cmd)
            if cmd == ["enroot", "list"]:
                return type(
                    "R",
                    (),
                    {
                        "returncode": 0,
                        "stdout": "hb__task__abc\nhb__task__def\nother_container\n",
                    },
                )()
            return type("R", (), {"returncode": 0, "stdout": ""})()

        with patch("tasktrace_capture.runner.subprocess.run", mock_run), patch(
            "tasktrace_capture.monitor.kill_orphaned_container_procs", return_value=[]
        ):
            _cleanup_harbor_containers()

        # Should remove hb__ containers (not other_container)
        assert ["enroot", "remove", "-f", "hb__task__abc"] in mock_calls
        assert ["enroot", "remove", "-f", "hb__task__def"] in mock_calls
        assert ["enroot", "remove", "-f", "other_container"] not in mock_calls

    def test_cleanup_handles_empty_list(self):
        from unittest.mock import patch

        from tasktrace_capture.runner import _cleanup_harbor_containers

        def mock_run(cmd, **kwargs):
            return type("R", (), {"returncode": 0, "stdout": "\n"})()

        with patch("tasktrace_capture.runner.subprocess.run", mock_run), patch(
            "tasktrace_capture.monitor.kill_orphaned_container_procs", return_value=[]
        ):
            _cleanup_harbor_containers()  # should not raise

    def test_cleanup_handles_enroot_failure(self):
        from unittest.mock import patch

        from tasktrace_capture.runner import _cleanup_harbor_containers

        def mock_run(cmd, **kwargs):
            return type("R", (), {"returncode": 1, "stdout": ""})()

        with patch("tasktrace_capture.runner.subprocess.run", mock_run), patch(
            "tasktrace_capture.monitor.kill_orphaned_container_procs", return_value=[]
        ):
            _cleanup_harbor_containers()  # should not raise


class TestKillOrphanedContainerProcs:
    """Verify kill_orphaned_container_procs() kills orphans but not live containers.

    Two independent safety checks must BOTH pass to kill:
    1. Namespace NOT in protected set (from enroot list)
    2. PPID=1 (orphaned — reparented to init when container was destroyed)

    This protects vLLM even if its container disappears from enroot list,
    because vLLM processes have a real parent chain (PPID != 1).
    """

    def _make_mock_proc(self, pid_ns_map, host_ns="mnt:[1111]", my_uid=1000, extra_ns=None):
        """Create mock /proc helpers for testing.

        pid_ns_map: {pid_str: {"ns": str, "uid": int, "cmdline": str, "ppid": int}}
        extra_ns: {pid_str: ns_str} — additional PID→namespace mappings for
            enroot list --fancy PIDs that aren't real processes in pid_ns_map.
        """
        import os as _os

        extra_ns = extra_ns or {}

        orig_listdir = _os.listdir
        orig_stat = _os.stat
        orig_readlink = _os.readlink

        def mock_listdir(path):
            if path == "/proc":
                return list(pid_ns_map.keys()) + ["self", "meminfo"]
            return orig_listdir(path)

        def mock_stat(path):
            for pid_str, info in pid_ns_map.items():
                if path == f"/proc/{pid_str}":
                    return type("S", (), {"st_uid": info.get("uid", my_uid)})()
            return orig_stat(path)

        def mock_readlink(path):
            if path == f"/proc/{_os.getpid()}/ns/mnt":
                return host_ns
            for pid_str, info in pid_ns_map.items():
                if path == f"/proc/{pid_str}/ns/mnt":
                    return info["ns"]
            for pid_str, ns in extra_ns.items():
                if path == f"/proc/{pid_str}/ns/mnt":
                    return ns
            if path.startswith("/proc/") and path.endswith("/ns/mnt"):
                raise FileNotFoundError(path)
            return orig_readlink(path)

        def mock_read_ppid(pid):
            pid_str = str(pid)
            if pid_str in pid_ns_map:
                return pid_ns_map[pid_str].get("ppid", 1)
            return 0

        return mock_listdir, mock_stat, mock_readlink, mock_read_ppid

    def _patches(self, mock_listdir, mock_stat, mock_readlink, mock_read_ppid, mock_subprocess_run, mock_kill=None):
        """Return a combined context manager for all the /proc + enroot mocks."""
        from unittest.mock import mock_open, patch

        killed = []
        if mock_kill is None:

            def mock_kill(pid, sig):
                killed.append(pid)

        class Ctx:
            def __init__(self):
                self.killed = killed
                self._patches = [
                    patch("tasktrace_capture.monitor.os.listdir", mock_listdir),
                    patch("tasktrace_capture.monitor.os.stat", mock_stat),
                    patch("tasktrace_capture.monitor.os.readlink", mock_readlink),
                    patch("tasktrace_capture.monitor._read_ppid", mock_read_ppid),
                    patch("tasktrace_capture.monitor.os.getuid", return_value=1000),
                    patch("tasktrace_capture.monitor.os.kill", mock_kill),
                    patch("tasktrace_capture.monitor.subprocess.run", mock_subprocess_run),
                    patch("builtins.open", mock_open(read_data=b"process")),
                ]

            def __enter__(self):
                for p in self._patches:
                    p.start()
                return self

            def __exit__(self, *args):
                for p in self._patches:
                    p.stop()

        return Ctx()

    def _enroot_mock(self, containers=None, ns_map=None, extra_readlink=None):
        """Mock enroot list + list --fancy for building the protected namespace set.

        containers: {name: namespace_id} — the namespace ID should match what
        the mock /proc readlink returns for processes in that container.

        Returns (mock_run, fancy_pid_ns_map) where fancy_pid_ns_map maps
        the fake PIDs from fancy output to their namespace IDs, so mock_readlink
        can resolve them.
        """
        containers = containers or {}  # {name: namespace_id}
        # Map of fancy-output PIDs to namespace IDs
        fancy_pids = {}
        pid_counter = 99900
        for _name, ns_id in containers.items():
            fancy_pids[str(pid_counter)] = ns_id
            pid_counter += 1

        def mock_run(cmd, **kwargs):
            if cmd == ["enroot", "list"]:
                stdout = "\n".join(containers.keys()) + "\n"
                return type("R", (), {"returncode": 0, "stdout": stdout})()
            if cmd == ["enroot", "list", "--fancy"]:
                lines = ["NAME       PID      COMM   STATE  STARTED  TIME   MNTNS       USERNS      COMMAND"]
                pid_c = 99900
                for name, ns_id in containers.items():
                    ns_num = ns_id.replace("mnt:[", "").replace("]", "")
                    lines.append(f"{name}  {pid_c}  proc  S  00:00  00:00  {ns_num}  0  cmd")
                    pid_c += 1
                stdout = "\n".join(lines) + "\n"
                return type("R", (), {"returncode": 0, "stdout": stdout})()
                raise RuntimeError("container broken")
            return type("R", (), {"returncode": 0, "stdout": ""})()

        return mock_run, fancy_pids

    def test_kills_orphan_not_in_protected_set(self):
        """Orphan (PPID=1, unprotected ns) should be killed."""
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"100": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "attack_mitm", "ppid": 1}}
        enroot, fancy_pids = self._enroot_mock({"vllm": "mnt:[2222]"})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            result = kill_orphaned_container_procs()
        assert 100 in ctx.killed
        assert len(result) == 1

    def test_preserves_process_in_protected_namespace(self):
        """vLLM in a live container's namespace should NOT be killed."""
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"200": {"ns": "mnt:[2222]", "uid": 1000, "cmdline": "vllm serve", "ppid": 1}}
        enroot, fancy_pids = self._enroot_mock({"vllm": "mnt:[2222]"})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            kill_orphaned_container_procs()
        assert len(ctx.killed) == 0

    def test_preserves_non_orphan_even_if_unprotected(self):
        """vLLM NOT in enroot list but with PPID!=1 should NOT be killed.

        This is the critical safety valve — even if enroot list misses vLLM,
        the PPID check protects it because live container processes have a
        real parent chain (slurmstepd → bash → vllm), not PPID=1.
        """
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"200": {"ns": "mnt:[2222]", "uid": 1000, "cmdline": "vllm serve", "ppid": 42127}}
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            result = kill_orphaned_container_procs()
        assert len(ctx.killed) == 0
        assert result == []

    def test_host_namespace_always_protected(self):
        """Processes in host namespace never killed regardless of PPID."""
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"300": {"ns": "mnt:[1111]", "uid": 1000, "cmdline": "tasktrace-capture", "ppid": 1}}
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            kill_orphaned_container_procs()
        assert len(ctx.killed) == 0

    def test_fails_safe_when_enroot_list_fails(self):
        """If enroot list fails, kill nothing."""
        from unittest.mock import patch

        from tasktrace_capture.monitor import kill_orphaned_container_procs

        def mock_run(cmd, **kwargs):
            if cmd == ["enroot", "list"]:
                return type("R", (), {"returncode": 1, "stdout": ""})()
            return type("R", (), {"returncode": 0, "stdout": ""})()

        killed = []
        with patch("tasktrace_capture.monitor.subprocess.run", mock_run), patch(
            "tasktrace_capture.monitor.os.kill", lambda p, s: killed.append(p)
        ):
            result = kill_orphaned_container_procs()
        assert len(killed) == 0
        assert result == []

    def test_fails_safe_when_enroot_list_times_out(self):
        """If enroot list times out, kill nothing."""
        from unittest.mock import patch

        from tasktrace_capture.monitor import kill_orphaned_container_procs

        def mock_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 5)

        killed = []
        with patch("tasktrace_capture.monitor.subprocess.run", mock_run), patch(
            "tasktrace_capture.monitor.os.kill", lambda p, s: killed.append(p)
        ):
            kill_orphaned_container_procs()
        assert len(killed) == 0

    def test_enroot_fancy_failure_still_works(self):
        """If enroot list --fancy fails, rely on PPID check as safety net."""
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"400": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "attack", "ppid": 1}}
        mocks = self._make_mock_proc(pids, "mnt:[1111]")

        def mock_run(cmd, **kwargs):
            if cmd == ["enroot", "list"]:
                return type("R", (), {"returncode": 0, "stdout": "broken\n"})()
            if cmd == ["enroot", "list", "--fancy"]:
                raise RuntimeError("fancy not supported")
            return type("R", (), {"returncode": 0, "stdout": ""})()

        with self._patches(*mocks, mock_run) as ctx:
            kill_orphaned_container_procs()
        assert 400 in ctx.killed

    def test_mixed_orphans_and_protected(self):
        """Only orphans killed when both orphans and protected exist."""
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {
            "500": {"ns": "mnt:[2222]", "uid": 1000, "cmdline": "vllm", "ppid": 42127},
            "501": {"ns": "mnt:[2222]", "uid": 1000, "cmdline": "vllm worker", "ppid": 500},
            "600": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "attack_mitm", "ppid": 1},
            "700": {"ns": "mnt:[4444]", "uid": 1000, "cmdline": "make -j128", "ppid": 1},
        }
        enroot, fancy_pids = self._enroot_mock({"vllm": "mnt:[2222]"})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            kill_orphaned_container_procs()
        assert 600 in ctx.killed
        assert 700 in ctx.killed
        assert 500 not in ctx.killed
        assert 501 not in ctx.killed

    def test_no_nonhost_processes_returns_empty(self):
        """If all processes are in host namespace, return empty."""
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"800": {"ns": "mnt:[1111]", "uid": 1000, "cmdline": "bash", "ppid": 1}}
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            result = kill_orphaned_container_procs()
        assert len(ctx.killed) == 0
        assert result == []

    def test_process_vanishes_between_scan_and_kill(self):
        """If process dies between scan and kill, skip gracefully."""
        from unittest.mock import mock_open, patch

        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"900": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "shortlived", "ppid": 1}}
        enroot, fancy_pids = self._enroot_mock({})
        ml, ms, mr, mp = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        def mock_kill(pid, sig):
            raise ProcessLookupError("No such process")

        with patch("tasktrace_capture.monitor.os.listdir", ml), patch("tasktrace_capture.monitor.os.stat", ms), patch(
            "tasktrace_capture.monitor.os.readlink", mr
        ), patch("tasktrace_capture.monitor._read_ppid", mp), patch(
            "tasktrace_capture.monitor.os.getuid", return_value=1000
        ), patch(
            "tasktrace_capture.monitor.os.kill", mock_kill
        ), patch(
            "tasktrace_capture.monitor.subprocess.run", enroot
        ), patch(
            "builtins.open", mock_open(read_data=b"shortlived")
        ):
            result = kill_orphaned_container_procs()
        assert result == []

    def test_skips_other_users_processes(self):
        """Processes owned by other users never touched."""
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"1000": {"ns": "mnt:[3333]", "uid": 9999, "cmdline": "other", "ppid": 1}}
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            kill_orphaned_container_procs()
        assert len(ctx.killed) == 0

    def test_child_of_orphan_also_killed(self):
        """attack_mitm (PPID=bash) where bash has PPID=1 should be killed.

        The agent spawns: bash (PPID=1) → attack_mitm (PPID=bash).
        The PPID check walks one level up: if parent's PPID=1, child is orphan too.
        """
        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {
            "550": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "bash -c attack", "ppid": 1},
            "551": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "./attack_mitm", "ppid": 550},
        }
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx:
            kill_orphaned_container_procs()
        assert 550 in ctx.killed
        assert 551 in ctx.killed

    def test_preserves_process_in_another_slurm_step(self):
        """/usr/sbin/sybil runs as us in step_extern under slurmstepd (PPID=1 parent).

        It passes the namespace and PPID checks, so only the cgroup check keeps it alive.
        """
        from unittest.mock import patch

        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {
            "560": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "/usr/sbin/sybil", "ppid": 1},
            "561": {"ns": "mnt:[4444]", "uid": 1000, "cmdline": "python runtests.py", "ppid": 1},
        }
        cgroups = {
            "self": b"0::/slurmstepd.scope/job_1/step_0/user/task_0\n",
            "560": b"0::/slurmstepd.scope/job_1/step_extern/slurm\n",
            "561": b"0::/slurmstepd.scope/job_1/step_0/user/task_0\n",
        }
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx, patch(
            "tasktrace_capture.monitor._read_cgroup", lambda pid: cgroups.get(str(pid))
        ):
            kill_orphaned_container_procs()
        assert 560 not in ctx.killed
        assert 561 in ctx.killed

    def test_fails_safe_when_own_cgroup_unreadable(self):
        """If our own cgroup can't be read, kill nothing."""
        from unittest.mock import patch

        from tasktrace_capture.monitor import kill_orphaned_container_procs

        pids = {"570": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "attack", "ppid": 1}}
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot) as ctx, patch("tasktrace_capture.monitor._read_cgroup", lambda pid: None):
            result = kill_orphaned_container_procs()
        assert ctx.killed == []
        assert result == []

    def test_container_listing_skips_other_slurm_steps(self):
        """Preflight and the process count leave out sybil in step_extern."""
        from unittest.mock import patch

        from tasktrace_capture.monitor import _iter_container_pids

        pids = {
            "580": {"ns": "mnt:[3333]", "uid": 1000, "cmdline": "/usr/sbin/sybil", "ppid": 1},
            "581": {"ns": "mnt:[4444]", "uid": 1000, "cmdline": "python runtests.py", "ppid": 1},
        }
        cgroups = {
            "self": b"0::/slurmstepd.scope/job_1/step_batch/user/task_0\n",
            "580": b"0::/slurmstepd.scope/job_1/step_extern/slurm\n",
            "581": b"0::/slurmstepd.scope/job_1/step_batch/user/task_0\n",
        }
        enroot, fancy_pids = self._enroot_mock({})
        mocks = self._make_mock_proc(pids, "mnt:[1111]", extra_ns=fancy_pids)

        with self._patches(*mocks, enroot), patch(
            "tasktrace_capture.monitor._read_cgroup", lambda pid: cgroups.get(str(pid))
        ):
            assert [pid for pid, _ in _iter_container_pids()] == [581]
        with self._patches(*mocks, enroot), patch("tasktrace_capture.monitor._read_cgroup", lambda pid: None):
            assert sorted(pid for pid, _ in _iter_container_pids()) == [580, 581]


class TestAbandonedTrialDetection:
    """Verify that abandoned trials (dir exists, no result.json) are detected."""

    def test_detects_abandoned_trials(self, tmp_path, capsys):
        """Trials with dirs but no result.json should trigger a warning."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        # Create a fake harbor output: 3 trial dirs, only 2 with result.json
        job_dir = tmp_path / "2026-01-01__00-00-00"
        job_dir.mkdir()
        for name in ["task__aaa", "task__bbb", "task__ccc"]:
            trial = job_dir / name
            trial.mkdir()
        # Only 2 of 3 get result.json
        (job_dir / "task__aaa" / "result.json").write_text('{"verifier_result":{"rewards":{"reward":1.0}}}')
        (job_dir / "task__bbb" / "result.json").write_text('{"verifier_result":{"rewards":{"reward":1.0}}}')
        # task__ccc is abandoned — no result.json
        # Also write the top-level result.json harbor creates
        (job_dir / "result.json").write_text('{"n_total_trials":3}')

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = os.getpid()
                self.returncode = 0

            def communicate(self, timeout=None):
                return "", ""

        with patch.object(real_subprocess, "Popen", MockPopen):
            run_experiment(
                dataset="hello-world",
                model="test",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=3600,
                jobs_dir=tmp_path,
            )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "abandoned" in captured.out
        assert "task__ccc" in captured.out


# ===========================================================================
#  End-to-end: parse_agent_time_split on real job data (if available)
# ===========================================================================

REAL_JOB_DIR_ENV = os.environ.get("BENCH_TEST_DATA_DIR")
REAL_JOB_DIR = Path(REAL_JOB_DIR_ENV) if REAL_JOB_DIR_ENV else None


@pytest.mark.skipif(
    REAL_JOB_DIR is None or not REAL_JOB_DIR.exists(), reason="BENCH_TEST_DATA_DIR not set or does not exist"
)
class TestRealJobData:
    def test_parses_real_job(self):
        r = parse_agent_time_split(REAL_JOB_DIR)
        assert r is not None
        assert r["n_full"] + r["n_partial"] + r["n_invalid"] >= 1

    def test_real_job_has_timing_fields(self):
        r = parse_agent_time_split(REAL_JOB_DIR)
        if "tool_time_sec" in r:
            assert r["tool_time_sec"] > 0
            assert r["wall_time_sec"] > 0
            assert r["llm_time_sec"] > 0
            assert r["n_llm_calls"] > 0

    def test_real_job_with_timeout(self):
        r = parse_agent_time_split(REAL_JOB_DIR, agent_timeout=120)
        assert r is not None
        assert "trials" in r


# ===========================================================================
#  get_found_suite_tasks
# ===========================================================================


def _make_suite_result_json(job_dir, task_name, trial_idx=0):
    """Write a minimal result.json with task_name for suite-mode tests."""
    trial_dir = job_dir / f"{task_name}__trial{trial_idx}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task_name": task_name,
        "agent_execution": {
            "started_at": "2026-04-02T00:00:00Z",
            "finished_at": "2026-04-02T00:15:00Z",
        },
        "agent_result": {"metadata": {}},
    }
    with open(trial_dir / "result.json", "w") as f:
        json.dump(result, f)
    return trial_dir


class TestGetFoundSuiteTasks:

    def test_none_dir(self):
        assert get_found_suite_tasks(None) == set()

    def test_nonexistent_dir(self):
        assert get_found_suite_tasks(Path("/does/not/exist")) == set()

    def test_empty_dir(self, tmp_path):
        assert get_found_suite_tasks(tmp_path) == set()

    def test_all_tasks_present(self, tmp_path):
        """All requested tasks are found in results."""
        _make_suite_result_json(tmp_path, "password-recovery")
        _make_suite_result_json(tmp_path, "compile-compcert")
        _make_suite_result_json(tmp_path, "qemu-startup")
        found = get_found_suite_tasks(tmp_path)
        assert found == {"password-recovery", "compile-compcert", "qemu-startup"}

    def test_missing_tasks_detected(self, tmp_path):
        """Only some requested tasks have results -- missing ones detectable."""
        _make_suite_result_json(tmp_path, "password-recovery")
        # "hello-world" was requested but is NOT in results
        found = get_found_suite_tasks(tmp_path)
        assert "password-recovery" in found
        assert "hello-world" not in found
        # Caller can compute: requested - found = {"hello-world"}
        missing = {"password-recovery", "hello-world"} - found
        assert missing == {"hello-world"}

    def test_multiple_trials_same_task(self, tmp_path):
        """Multiple trials of the same task still produce one entry."""
        _make_suite_result_json(tmp_path, "password-recovery", trial_idx=0)
        _make_suite_result_json(tmp_path, "password-recovery", trial_idx=1)
        found = get_found_suite_tasks(tmp_path)
        assert found == {"password-recovery"}

    def test_fallback_to_directory_name(self, tmp_path):
        """When task_name is absent from result.json, falls back to dir name parsing."""
        trial_dir = tmp_path / "compile-compcert__trial0"
        trial_dir.mkdir()
        # Write result.json WITHOUT task_name
        result = {
            "agent_execution": {
                "started_at": "2026-04-02T00:00:00Z",
                "finished_at": "2026-04-02T00:15:00Z",
            },
            "agent_result": {"metadata": {}},
        }
        with open(trial_dir / "result.json", "w") as f:
            json.dump(result, f)
        found = get_found_suite_tasks(tmp_path)
        assert "compile-compcert" in found

    def test_skips_top_level_result_json(self, tmp_path):
        """Harbor's top-level result.json (job summary) is skipped."""
        # Write a top-level result.json (not a trial)
        with open(tmp_path / "result.json", "w") as f:
            json.dump({"task_name": "should-be-ignored"}, f)
        # Write a real trial
        _make_suite_result_json(tmp_path, "real-task")
        found = get_found_suite_tasks(tmp_path)
        assert "should-be-ignored" not in found
        assert "real-task" in found

    def test_corrupt_json_skipped(self, tmp_path):
        """Corrupt result.json files are skipped without crashing."""
        trial_dir = tmp_path / "bad-task__trial0"
        trial_dir.mkdir()
        (trial_dir / "result.json").write_text("NOT VALID JSON {{{")
        _make_suite_result_json(tmp_path, "good-task")
        found = get_found_suite_tasks(tmp_path)
        assert "good-task" in found
        assert len(found) == 1  # corrupt entry not added


# ===========================================================================
#  --api-key round-robin env var plumbing
# ===========================================================================


class TestApiKeyEnvPlumbing:
    """Verify that run_experiment sets the right env vars for API key rotation."""

    def test_multiple_keys_sets_tasktrace_capture_api_keys(self, tmp_path, monkeypatch):
        """Multiple api_keys should set _TASKTRACE_CAPTURE_API_KEYS in subprocess env."""
        import subprocess

        captured_env = {}

        class FakeProc:
            stdout = iter([])
            stderr = iter([])
            returncode = 1  # force early exit
            pid = 12345

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                pass

            def terminate(self):
                pass

        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("tasktrace_capture.runner.get_active_harbor_pgid", lambda: None)

        from tasktrace_capture.runner import run_experiment

        try:
            run_experiment(
                dataset="hello-world",
                model="m",
                api_base="http://x:8000/v1",
                model_info={},
                n_concurrent=2,
                agent_timeout=60,
                api_keys=["sk-key-A", "sk-key-B"],
                jobs_dir=tmp_path,
                agent="terminus-2",
            )
        except Exception:
            pass  # will fail due to fake proc, that's fine

        assert captured_env.get("_TASKTRACE_CAPTURE_API_KEYS") == "sk-key-A,sk-key-B"

    def test_single_key_sets_anthropic_api_key(self, tmp_path, monkeypatch):
        """Single api_key should set ANTHROPIC_API_KEY directly."""
        import subprocess

        captured_env = {}

        class FakeProc:
            stdout = iter([])
            stderr = iter([])
            returncode = 1
            pid = 12345

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                pass

            def terminate(self):
                pass

        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("tasktrace_capture.runner.get_active_harbor_pgid", lambda: None)

        from tasktrace_capture.runner import run_experiment

        try:
            run_experiment(
                dataset="hello-world",
                model="m",
                api_base="http://x:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=60,
                api_keys=["sk-solo-key"],
                jobs_dir=tmp_path,
                agent="terminus-2",
            )
        except Exception:
            pass

        assert captured_env.get("ANTHROPIC_API_KEY") == "sk-solo-key"
        assert "_TASKTRACE_CAPTURE_API_KEYS" not in captured_env
        # Should also be in forward list
        fwd = captured_env.get("_TASKTRACE_CAPTURE_FORWARD_ENVS", "")
        assert "ANTHROPIC_API_KEY" in fwd

    def test_resolved_api_base_is_available_to_runtime_policy(self, tmp_path, monkeypatch):
        """The runtime must see the endpoint selected by --api-base."""
        import subprocess

        captured_env = {}

        class FakeProc:
            stdout = iter([])
            stderr = iter([])
            returncode = 1
            pid = 12345

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                pass

            def terminate(self):
                pass

        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("tasktrace_capture.runner.get_active_harbor_pgid", lambda: None)

        from tasktrace_capture.runner import run_experiment

        try:
            run_experiment(
                dataset="hello-world",
                model="m",
                api_base="https://terminus.example.test/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=60,
                jobs_dir=tmp_path,
                agent="terminus-2",
            )
        except Exception:
            pass

        assert captured_env["AGENT_API_BASE"] == ("https://terminus.example.test/v1")

    def test_no_keys_no_env_vars(self, tmp_path, monkeypatch):
        """No api_keys should not set any key-related env vars."""
        import subprocess

        captured_env = {}

        class FakeProc:
            stdout = iter([])
            stderr = iter([])
            returncode = 1
            pid = 12345

            def wait(self, timeout=None):
                return self.returncode

            def poll(self):
                return self.returncode

            def kill(self):
                pass

            def terminate(self):
                pass

        def fake_popen(cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return FakeProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr("tasktrace_capture.runner.get_active_harbor_pgid", lambda: None)

        from tasktrace_capture.runner import run_experiment

        try:
            run_experiment(
                dataset="hello-world",
                model="m",
                api_base="http://x:8000/v1",
                model_info={},
                n_concurrent=1,
                agent_timeout=60,
                api_keys=None,
                jobs_dir=tmp_path,
                agent="terminus-2",
            )
        except Exception:
            pass

        assert "_TASKTRACE_CAPTURE_API_KEYS" not in captured_env

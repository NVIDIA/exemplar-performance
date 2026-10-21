# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CLI-related functionality: parse_task_arg, _aggregate_splits,
CLI dry-run integration tests, and SSRF validation.

Extracted from test_bench.py.
"""

import json
import shutil
import subprocess

import pytest
from tasktrace_capture.cli import (
    _aggregate_splits,
    _is_private_url,
    _materialize_adhoc_task,
    parse_task_arg,
)

# ===========================================================================
#  parse_task_arg
# ===========================================================================


class TestParseTaskArg:

    def test_single_task(self):
        mode, val = parse_task_arg("terminal-bench/compile-compcert")
        assert mode == "single"
        assert val == "terminal-bench/compile-compcert"

    def test_suite_comma_separated(self):
        mode, val = parse_task_arg("terminal-bench/qemu-startup,sqlite-with-gcov,headless-terminal")
        assert mode == "suite"
        assert val["dataset"] == "terminal-bench"
        assert val["tasks"] == ["qemu-startup", "sqlite-with-gcov", "headless-terminal"]

    def test_suite_with_full_prefixes(self):
        mode, val = parse_task_arg("terminal-bench/task-a,terminal-bench/task-b")
        assert mode == "suite"
        assert val["tasks"] == ["task-a", "task-b"]

    def test_suite_mixed_prefix_no_prefix(self):
        mode, val = parse_task_arg("terminal-bench/first,second,third")
        assert mode == "suite"
        assert val["tasks"] == ["first", "second", "third"]

    def test_single_no_slash(self):
        mode, val = parse_task_arg("hello-world")
        assert mode == "single"
        assert val == "hello-world"

    def test_suite_mismatched_dataset_exits(self):
        with pytest.raises(SystemExit):
            parse_task_arg("terminal-bench/a,other-bench/b")


# ===========================================================================
#  _aggregate_splits
# ===========================================================================


class TestAggregateSplits:

    @staticmethod
    def _make_split(tool=100, wall=500, llm=400, calls=10, full=4, partial=0, invalid=0):
        total = full + partial + invalid
        return {
            "pct_cpu": round(100 * tool / wall, 1),
            "pct_llm": round(100 * llm / wall, 1),
            "tool_time_sec": tool,
            "wall_time_sec": wall,
            "llm_time_sec": llm,
            "n_llm_calls": calls,
            "n_full": full,
            "n_partial": partial,
            "n_invalid": invalid,
            "survival_rate": f"{full}/{total}",
        }

    def test_single_repeat(self):
        s = self._make_split(tool=160, wall=900, llm=740, calls=24)
        agg = _aggregate_splits([s])
        assert agg["n_repeats"] == 1
        assert agg["tool_time_sec"]["median"] == 160
        assert agg["tool_time_sec"]["min"] == 160
        assert agg["tool_time_sec"]["max"] == 160

    def test_three_repeats_median(self):
        splits = [
            self._make_split(tool=100, wall=900, llm=800, calls=20),
            self._make_split(tool=150, wall=900, llm=750, calls=22),
            self._make_split(tool=200, wall=900, llm=700, calls=25),
        ]
        agg = _aggregate_splits(splits)
        assert agg["n_repeats"] == 3
        assert agg["tool_time_sec"]["median"] == 150
        assert agg["tool_time_sec"]["min"] == 100
        assert agg["tool_time_sec"]["max"] == 200
        assert agg["n_llm_calls"]["median"] == 22

    def test_survival_rate_aggregated(self):
        splits = [
            self._make_split(full=3, partial=1, invalid=0),
            self._make_split(full=4, partial=0, invalid=0),
            self._make_split(full=2, partial=0, invalid=2),
        ]
        agg = _aggregate_splits(splits)
        assert agg["n_full"] == 9
        assert agg["n_partial"] == 1
        assert agg["n_invalid"] == 2
        assert agg["survival_rate"] == "9/12"

    def test_per_repeat_preserved(self):
        splits = [self._make_split(), self._make_split()]
        agg = _aggregate_splits(splits)
        assert len(agg["per_repeat"]) == 2

    def test_values_list_present(self):
        splits = [
            self._make_split(tool=100),
            self._make_split(tool=200),
        ]
        agg = _aggregate_splits(splits)
        assert agg["tool_time_sec"]["values"] == [100, 200]

    def test_phases_and_reward_aggregated(self):
        """Regression: --repeat mode must aggregate avg_reward and phase durations.

        Previously _aggregate_splits only aggregated pct_cpu/pct_llm/tool/wall/llm/calls.
        Phase breakdown and reward were dropped, showing as n/a in repeat-mode reports.
        """

        def _make_split_with_phases(reward, env, ags, exe, ver):
            s = self._make_split()
            s["avg_reward"] = reward
            s["avg_env_setup_sec"] = env
            s["avg_agent_setup_sec"] = ags
            s["avg_agent_exec_sec"] = exe
            s["avg_verifier_sec"] = ver
            return s

        splits = [
            _make_split_with_phases(1.0, 10.0, 3.0, 400.0, 5.0),
            _make_split_with_phases(0.5, 12.0, 4.0, 420.0, 6.0),
            _make_split_with_phases(0.75, 8.0, 2.5, 380.0, 4.5),
        ]
        agg = _aggregate_splits(splits)

        # avg_reward must be aggregated
        assert agg["avg_reward"] is not None
        assert agg["avg_reward"]["median"] == 0.75
        assert agg["avg_reward"]["min"] == 0.5
        assert agg["avg_reward"]["max"] == 1.0

        # Phase durations must be aggregated
        assert agg["avg_env_setup_sec"] is not None
        assert agg["avg_env_setup_sec"]["median"] == 10.0
        assert agg["avg_agent_setup_sec"] is not None
        assert agg["avg_agent_setup_sec"]["median"] == 3.0
        assert agg["avg_agent_exec_sec"] is not None
        assert agg["avg_agent_exec_sec"]["median"] == 400.0
        assert agg["avg_verifier_sec"] is not None
        assert agg["avg_verifier_sec"]["median"] == 5.0

    def test_none_reward_filtered_in_aggregation(self):
        """_aggregate_splits handles None avg_reward values without crashing."""
        splits = [
            {**self._make_split(), "avg_reward": None},
            {**self._make_split(), "avg_reward": 1.0},
        ]
        agg = _aggregate_splits(splits)
        # Only the non-None value should be aggregated
        assert agg["avg_reward"] is not None
        assert agg["avg_reward"]["median"] == 1.0
        assert len(agg["avg_reward"]["values"]) == 1


# ===========================================================================
#  CLI dry-run integration tests
# ===========================================================================


class TestCLIDryRun:
    """Test the tasktrace-capture CLI --dry-run with various flags."""

    def _run(self, *extra_args, expect_rc=0):
        import shutil as _shutil

        cli = _shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [
            cli,
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
            "--dry-run",
            *extra_args,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    def test_default_hello_world(self):
        r = self._run()
        assert "hello-world" in r.stdout.lower() or "Dry run" in r.stdout

    def test_single_task(self):
        r = self._run("--task", "terminal-bench/qemu-startup")
        assert "Dry run" in r.stdout

    def test_concurrency_flag(self):
        r = self._run("--task", "terminal-bench/qemu-startup", "--concurrency", "1,2,4")
        out = r.stdout
        assert "1" in out
        assert "2" in out
        assert "4" in out
        assert "Dry run" in out

    def test_sweep_flag(self):
        r = self._run("--task", "terminal-bench/qemu-startup", "--sweep", "1", "2", "8")
        assert "Dry run" in r.stdout

    def test_repeat_flag_shows_in_config(self):
        r = self._run("--task", "terminal-bench/qemu-startup", "--concurrency", "1,2", "--repeat", "3")
        assert "3" in r.stdout
        assert "Dry run" in r.stdout

    def test_agent_timeout_shows_in_config(self):
        r = self._run("--agent-timeout", "900")
        assert "900" in r.stdout

    def test_agent_kwarg_shows_in_config(self):
        r = self._run("--agent-kwarg", "temperature=0.0")
        assert "temperature=0.0" in r.stdout

    def test_suite_mode(self):
        r = self._run("--task", "terminal-bench/qemu-startup,sqlite-with-gcov", "--concurrency", "1,2")
        assert "Suite" in r.stdout or "suite" in r.stdout
        assert "Dry run" in r.stdout

    def test_package_dataset_explicit_suite(self):
        r = self._run(
            "--dataset",
            "datacurve/deep-swe",
            "--include-task",
            "datacurve/task-a",
            "--include-task",
            "datacurve/task-b",
            "--concurrency",
            "1",
        )
        assert "Suite" in r.stdout or "suite" in r.stdout
        assert "datacurve/deep-swe" in r.stdout
        assert "datacurve/task-a" in r.stdout
        assert "datacurve/task-b" in r.stdout

    def test_legacy_registry_task_displays_canonical_single_selection(self):
        r = self._run(
            "--task",
            "terminal-bench/filter-js-from-html",
            "--concurrency",
            "1",
        )
        assert "Single task" in r.stdout
        assert "terminal-bench" in r.stdout
        assert "filter-js-from-html" in r.stdout

    def test_include_task_requires_dataset(self):
        r = self._run("--include-task", "datacurve/task-a", expect_rc=2)
        assert "requires --dataset" in r.stderr

    def test_include_task_rejects_n_tasks(self):
        r = self._run(
            "--dataset",
            "datacurve/deep-swe",
            "--include-task",
            "datacurve/task-a",
            "--n-tasks",
            "1",
            expect_rc=2,
        )
        assert "cannot be combined" in r.stderr

    def test_dataset_and_task_conflict(self):
        r = self._run("-d", "hello-world", "--task", "terminal-bench/x", expect_rc=2)
        assert r.returncode == 2

    def test_trials_flag(self):
        r = self._run("--task", "terminal-bench/qemu-startup", "--concurrency", "4", "-k", "2")
        assert "Dry run" in r.stdout

    def test_repeat_defaults_to_1(self):
        r = self._run("--task", "terminal-bench/qemu-startup", "--concurrency", "1")
        # "Repeat" row should NOT appear when repeat=1
        assert "Repeat" not in r.stdout or "1x" not in r.stdout

    # ── --concurrency validation ──

    def test_concurrency_zero_rejected(self):
        r = self._run("--concurrency", "0", expect_rc=2)
        assert r.returncode == 2

    def test_concurrency_negative_rejected(self):
        r = self._run("--concurrency", "-1", expect_rc=2)
        assert r.returncode == 2

    def test_concurrency_non_numeric_rejected(self):
        r = self._run("--concurrency", "abc", expect_rc=2)
        assert r.returncode == 2

    def test_concurrency_mixed_invalid_rejected(self):
        r = self._run("--concurrency", "1,0,4", expect_rc=2)
        assert r.returncode == 2

    def test_concurrency_duplicates_rejected(self):
        r = self._run("--concurrency", "1,1", expect_rc=2)
        assert r.returncode == 2
        assert "duplicate" in r.stderr.lower()

    def test_repeat_zero_rejected(self):
        r = self._run("--repeat", "0", expect_rc=2)
        assert r.returncode == 2

    def test_repeat_negative_rejected(self):
        r = self._run("--repeat", "-1", expect_rc=2)
        assert r.returncode == 2

    def test_agent_timeout_zero_rejected(self):
        r = self._run("--agent-timeout", "0", expect_rc=2)
        assert r.returncode == 2

    def test_output_dir_with_spaces_rejected(self):
        r = self._run("--output-dir", "/tmp/test path with spaces", expect_rc=2)
        assert r.returncode == 2
        assert "spaces" in r.stderr.lower() or "space" in r.stderr.lower()


# ===========================================================================
#  --api-key flag: round-robin API key rotation
# ===========================================================================


class TestApiKeyFlag:
    """Test --api-key CLI flag for API key rotation."""

    def _run(self, *extra_args, expect_rc=0):
        import shutil as _shutil

        cli = _shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [
            cli,
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
            "--dry-run",
            *extra_args,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    def test_single_api_key_accepted(self):
        r = self._run("--api-key", "sk-test-key-1")
        assert r.returncode == 0

    def test_multiple_api_keys_accepted(self):
        r = self._run("--api-key", "sk-key-1", "--api-key", "sk-key-2")
        assert r.returncode == 0

    def test_api_key_with_concurrency(self):
        r = self._run("--api-key", "sk-key-1", "--api-key", "sk-key-2", "--concurrency", "1,4")
        assert r.returncode == 0


# ===========================================================================
#  --check without --api-base
# ===========================================================================


class TestCheckWithoutApiBase:
    """--check should run preflight checks without requiring --api-base."""

    def test_check_without_api_base_does_not_error_about_api_base(self):
        """tasktrace-capture --check should not fail with 'provide --api-base' error.

        The exit code may be non-zero if enroot/harbor are not installed,
        but the error must NOT be an argparse error about --api-base (rc=2).
        """
        cli = shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [cli, "--check", "--quiet"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        # argparse errors produce rc=2 -- this must NOT happen
        assert r.returncode != 2, f"--check should not produce argparse error.\n" f"STDERR: {r.stderr}"


# ===========================================================================
#  SSRF validation (_is_private_url)
# ===========================================================================


class TestIsPrivateUrl:
    """Validate that _is_private_url detects private/loopback addresses."""

    def test_localhost_is_private(self):
        assert _is_private_url("http://localhost:8000/v1") is True

    def test_127_0_0_1_is_private(self):
        assert _is_private_url("http://127.0.0.1:8000/v1") is True

    def test_127_x_is_private(self):
        assert _is_private_url("http://127.0.0.42:8000/v1") is True

    def test_10_x_is_private(self):
        assert _is_private_url("http://10.0.0.5:8000/v1") is True

    def test_172_16_x_is_private(self):
        assert _is_private_url("http://172.16.0.1:8000/v1") is True

    def test_172_31_x_is_private(self):
        assert _is_private_url("http://172.31.255.255:8000/v1") is True

    def test_192_168_x_is_private(self):
        assert _is_private_url("http://192.168.1.100:8000/v1") is True

    def test_169_254_link_local_is_private(self):
        assert _is_private_url("http://169.254.169.254/latest/meta-data") is True

    def test_0_0_0_0_is_private(self):
        # 0.0.0.0 is unspecified and should be treated as private
        assert _is_private_url("http://0.0.0.0:8000/v1") is True

    def test_public_ip_is_not_private(self):
        assert _is_private_url("http://8.8.8.8:8000/v1") is False

    def test_unresolvable_domain_is_not_private(self):
        # Unresolvable hostnames should not be flagged as private
        assert _is_private_url("https://this-domain-does-not-exist-xyz.example.com") is False

    def test_empty_url_is_not_private(self):
        assert _is_private_url("") is False

    def test_no_hostname_is_not_private(self):
        assert _is_private_url("not-a-url") is False

    def test_unresolvable_hostname_is_not_private(self):
        # A hostname that DNS cannot resolve should not be flagged as private
        assert _is_private_url("http://this-hostname-does-not-exist-zzz.example:8000/v1") is False

    def test_https_localhost_is_private(self):
        assert _is_private_url("https://localhost:443/v1") is True

    def test_172_32_is_not_private(self):
        """172.32.x.x is outside the RFC-1918 172.16-31 range."""
        assert _is_private_url("http://172.32.0.1:8000/v1") is False


class TestAllowPrivateApiFlag:
    """--allow-private-api flag should suppress SSRF warning in dry-run."""

    def _run(self, *extra_args, expect_rc=0):
        cli = shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [
            cli,
            "--api-base",
            "http://127.0.0.1:8000/v1",
            "--skip-preflight",
            "--dry-run",
            *extra_args,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    def test_allow_private_api_accepted(self):
        """--allow-private-api flag should be accepted without error."""
        r = self._run("--allow-private-api")
        # Should run successfully (dry-run exits 0)
        assert r.returncode == 0


# ===========================================================================
#  Input validation gaps
# ===========================================================================


class TestIssue21ValidationGaps:
    """Tests for input validation fixes."""

    def _run(self, *extra_args, expect_rc=0):
        import shutil as _shutil

        cli = _shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [
            cli,
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
            "--dry-run",
            *extra_args,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    # ── tab in --output-dir ──

    def test_output_dir_with_tab_rejected(self):
        r = self._run("--output-dir", "/tmp/test\tpath", expect_rc=2)
        assert r.returncode == 2
        assert "whitespace" in r.stderr.lower()

    def test_output_dir_with_newline_rejected(self):
        r = self._run("--output-dir", "/tmp/test\npath", expect_rc=2)
        assert r.returncode == 2

    # ── duplicate tasks in suite mode ──

    def test_suite_deduplicates_tasks(self):
        """Duplicate tasks are silently deduplicated."""
        mode, val = parse_task_arg("terminal-bench/hello-world,hello-world")
        assert mode == "suite"
        assert val["tasks"] == ["hello-world"], f"Expected deduplicated tasks, got: {val['tasks']}"

    def test_suite_deduplicates_with_prefix(self):
        """Duplicates with explicit prefix are also caught."""
        mode, val = parse_task_arg("terminal-bench/hello-world,terminal-bench/hello-world")
        assert mode == "suite"
        assert val["tasks"] == ["hello-world"]

    def test_suite_preserves_order_after_dedup(self):
        """First occurrence wins, order is preserved."""
        mode, val = parse_task_arg("terminal-bench/a,b,a,c,b")
        assert mode == "suite"
        assert val["tasks"] == ["a", "b", "c"]

    # ── --concurrency + --sweep conflict ──

    def test_concurrency_and_sweep_conflict(self):
        r = self._run(
            "--concurrency",
            "1,2",
            "--sweep",
            "4",
            "8",
            expect_rc=2,
        )
        assert r.returncode == 2
        assert "alias" in r.stderr.lower() or "both" in r.stderr.lower()


# ===========================================================================
#  Misleading Calls:1 when LLM timing data missing
# ===========================================================================


class TestIssue25MisleadingCalls:
    """When n_llm_calls and llm_time_sec are both zero/missing, Calls and
    Avg lat should show — instead of fake 1 / 0.00s."""

    def test_report_json_no_fake_calls_when_llm_data_missing(self, tmp_path):
        """report.json avg_llm_calls is None when no LLM timing data."""
        from tasktrace_capture.cli import _write_report_json

        data = {
            "dataset": "test",
            "model": "m",
            "api_base": "http://x",
            "timestamp": "2026-01-01",
            "hostname": "h",
            "cpu_info": {"model": "x", "cores_logical": 128},
            "time_splits": {
                "1": {
                    "pct_cpu": 80.0,
                    "pct_llm": 20.0,
                    "tool_time_sec": 40.0,
                    "wall_time_sec": 50.0,
                    "llm_time_sec": 0,
                    "n_llm_calls": 0,
                    "avg_reward": None,
                    "n_full": 1,
                    "n_partial": 0,
                    "n_invalid": 0,
                    "survival_rate": "1/1",
                },
            },
            "results": [{"n_concurrent": 1, "n_cores": 128}],
            "baseline_cpu": {},
            "system_info": {},
        }
        _write_report_json(data, tmp_path)
        import json

        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        timing = p["concurrency"]["1"]["timing"]
        assert timing["avg_llm_calls"] is None, f"Expected None for avg_llm_calls, got {timing['avg_llm_calls']}"
        assert timing["avg_llm_latency_sec"] is None
        assert timing["avg_llm_sec"] is None

    def test_report_json_real_calls_preserved(self, tmp_path):
        """report.json still shows real call data when available."""
        from tasktrace_capture.cli import _write_report_json

        data = {
            "dataset": "test",
            "model": "m",
            "api_base": "http://x",
            "timestamp": "2026-01-01",
            "hostname": "h",
            "cpu_info": {"model": "x", "cores_logical": 128},
            "time_splits": {
                "1": {
                    "pct_cpu": 60.0,
                    "pct_llm": 40.0,
                    "tool_time_sec": 30.0,
                    "wall_time_sec": 50.0,
                    "llm_time_sec": 20.0,
                    "n_llm_calls": 10,
                    "avg_reward": 1.0,
                    "n_full": 1,
                    "n_partial": 0,
                    "n_invalid": 0,
                    "survival_rate": "1/1",
                },
            },
            "results": [{"n_concurrent": 1, "n_cores": 128}],
            "baseline_cpu": {},
            "system_info": {},
        }
        _write_report_json(data, tmp_path)
        import json

        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        timing = p["concurrency"]["1"]["timing"]
        assert timing["avg_llm_calls"] == 10
        assert timing["avg_llm_latency_sec"] == 2.0  # 20/10

    def test_report_json_preserves_canonical_task_selection(self, tmp_path):
        from tasktrace_capture.cli import _write_report_json

        data = {
            "dataset": "filter-js-from-html",
            "suite_dataset": "terminal-bench@2.0",
            "suite_tasks": ["filter-js-from-html"],
            "model": "m",
            "api_base": "http://x",
            "timestamp": "2026-01-01",
            "hostname": "h",
            "cpu_info": {"model": "x", "cores_logical": 128},
            "time_splits": {},
            "results": [],
            "baseline_cpu": {},
            "system_info": {},
        }
        _write_report_json(data, tmp_path)
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["dataset"] == "terminal-bench@2.0"
        assert report["include_tasks"] == ["filter-js-from-html"]

    def test_report_txt_dash_when_llm_data_missing(self, tmp_path):
        """report.txt shows — for Calls/Avg lat when LLM data missing."""
        from tasktrace_capture.cli import _write_report

        data = {
            "dataset": "test",
            "model": "m",
            "api_base": "http://x",
            "timestamp": "2026-01-01",
            "hostname": "h",
            "cpu_info": {"model": "x", "cores_logical": 128},
            "time_splits": {
                "1": {
                    "pct_cpu": 80.0,
                    "pct_llm": 20.0,
                    "tool_time_sec": 40.0,
                    "wall_time_sec": 50.0,
                    "llm_time_sec": 0,
                    "n_llm_calls": 0,
                    "avg_reward": None,
                    "n_full": 1,
                    "n_partial": 0,
                    "n_invalid": 0,
                    "survival_rate": "1/1",
                },
            },
            "results": [{"n_concurrent": 1, "n_cores": 128}],
            "baseline_cpu": {},
            "system_info": {},
        }
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        # Must contain the dash character, not "1" in the Calls column
        assert "\u2014" in text, "Expected em dash (—) for missing LLM data"

    def test_summary_table_dash_when_no_llm_data(self, capsys):
        """_print_summary_table shows — for LLM columns when data missing."""
        from tasktrace_capture.cli import _print_summary_table

        data = {
            "dataset": "test",
            "model": "test-model",
            "time_splits": {
                "1": {
                    "pct_cpu": 80.0,
                    "pct_llm": 20.0,
                    "n_llm_calls": 0,
                    "llm_time_sec": 0,
                    "tool_time_sec": 40.0,
                    "wall_time_sec": 50.0,
                    "avg_reward": None,
                    "n_full": 1,
                    "n_partial": 0,
                    "n_invalid": 0,
                    "survival_rate": "1/1",
                    "trials": [
                        {
                            "status": "full",
                            "wall_time_sec": 50.0,
                            "llm_time_sec": 0,
                            "tool_time_sec": 40.0,
                            "n_llm_calls": 0,
                            "reward": 0.0,
                        }
                    ],
                }
            },
        }
        # Should not crash and should not show "1" for calls
        _print_summary_table(data)


# ===========================================================================
#  claude-code preflight validation
# ===========================================================================


class TestClaudeCodeValidation:
    """--agent claude-code should require ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY."""

    def _run(self, *extra_args, expect_rc=2):
        cli = shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [cli, "--agent", "claude-code", "--skip-preflight", "--dry-run", *extra_args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    def test_missing_base_url_errors(self):
        """claude-code without ANTHROPIC_BASE_URL should fail with parser.error."""
        r = self._run(
            "--agent-env",
            "ANTHROPIC_API_KEY=test-key",
        )
        assert "ANTHROPIC_BASE_URL" in r.stderr

    def test_missing_api_key_or_oauth_errors(self):
        """claude-code without API key or OAuth token should fail."""
        r = self._run(
            "--agent-env",
            "ANTHROPIC_BASE_URL=https://api.example.com",
        )
        assert "ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN" in r.stderr

    def test_oauth_token_does_not_require_api_key(self):
        """Claude subscription OAuth should pass without an API key."""
        r = self._run(
            "--agent-env",
            "ANTHROPIC_BASE_URL=https://api.anthropic.com",
            "--agent-env",
            "CLAUDE_CODE_OAUTH_TOKEN=oauth-token",
            expect_rc=None,
        )
        assert "ANTHROPIC_API_KEY" not in r.stderr

    def test_both_present_passes_validation(self):
        """claude-code with both env vars should pass argument validation."""
        r = self._run(
            "--agent-env",
            "ANTHROPIC_BASE_URL=https://api.example.com",
            "--agent-env",
            "ANTHROPIC_API_KEY=test-key",
            "-m",
            "azure/anthropic/claude-sonnet-4-5",
            expect_rc=None,  # don't check rc, just ensure it gets past validation
        )
        # Should NOT contain the ANTHROPIC_BASE_URL error
        assert "ANTHROPIC_BASE_URL" not in r.stderr or "requires" not in r.stderr

    def test_wrong_model_prefix_no_crash(self):
        """Model without provider prefix should not crash."""
        r = self._run(
            "--agent-env",
            "ANTHROPIC_BASE_URL=https://api.example.com",
            "--agent-env",
            "ANTHROPIC_API_KEY=test-key",
            "-m",
            "claude-sonnet-4-5",
            expect_rc=0,
        )
        assert "Dry run" in r.stdout


# ===========================================================================
#  --prompt / --prompt-file ad-hoc task materialization
# ===========================================================================


class TestPromptFlag:
    """Validate --prompt builds a Harbor task dir and rejects bad combinations."""

    def _run(self, *extra_args, expect_rc=0):
        cli = shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [
            cli,
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
            "--dry-run",
            *extra_args,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    def test_materialize_writes_task_files(self, tmp_path):
        """--prompt produces a task dir whose instruction.md matches and
        whose task.toml parses."""
        prompt = "Create hello.txt containing Hello, world!"
        task_dir = _materialize_adhoc_task(
            prompt_body=prompt,
            dest_dir=tmp_path / "adhoc_task",
            image="ubuntu:24.04",
            agent_timeout_sec=900,
        )
        assert (task_dir / "instruction.md").read_text().strip() == prompt
        assert (task_dir / "environment" / "Dockerfile").read_text() == "FROM ubuntu:24.04\n"
        assert (task_dir / "solution" / "solve.sh").stat().st_mode & 0o111
        test_sh = (task_dir / "tests" / "test.sh").read_text()
        assert "/logs/verifier/reward.txt" in test_sh

        import tomllib

        with open(task_dir / "task.toml", "rb") as f:
            parsed = tomllib.load(f)
        assert parsed["version"] == "1.0"
        assert parsed["agent"]["timeout_sec"] == 900
        assert parsed["environment"]["docker_image"] == "ubuntu:24.04"

    def test_prompt_with_task_rejected(self):
        r = self._run(
            "--prompt",
            "hi",
            "--task",
            "terminal-bench/qemu-startup",
            expect_rc=2,
        )
        assert "prompt" in r.stderr.lower()

    def test_prompt_with_dataset_rejected(self):
        r = self._run(
            "--prompt",
            "hi",
            "-d",
            "hello-world",
            expect_rc=2,
        )
        assert "prompt" in r.stderr.lower()

    def test_prompt_and_prompt_file_rejected(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("body")
        r = self._run(
            "--prompt",
            "hi",
            "--prompt-file",
            str(f),
            expect_rc=2,
        )
        assert "prompt" in r.stderr.lower()

    def test_empty_prompt_rejected(self):
        r = self._run("--prompt", "   ", expect_rc=2)
        assert "empty" in r.stderr.lower()

    def test_prompt_dry_run_shows_adhoc_mode(self):
        r = self._run("--prompt", "Create hello.txt with Hello world")
        assert "Ad-hoc" in r.stdout or "adhoc" in r.stdout.lower()


# ===========================================================================
#  3-tier agent preflight (supported / unsupported / unknown)
# ===========================================================================


class TestAgentPreflightTiers:

    def _run(self, agent, *extra_args, expect_rc=None):
        cli = shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [cli, "--agent", agent, "--skip-preflight", "--dry-run", *extra_args]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    def test_supported_agent_proceeds_silently(self):
        """mini-swe-agent (validated tier) should not print the unknown warning."""
        r = self._run("mini-swe-agent")
        assert "is not validated" not in r.stdout

    def test_unsupported_agent_hard_errors_with_no_bypass(self):
        """openhands should fail with parser.error and no bypass flag offered."""
        r = self._run("openhands", expect_rc=2)
        assert "unsupported" in r.stderr.lower()
        assert "--allow-unsupported-agent" not in r.stderr

    def test_unknown_agent_warns_and_proceeds(self):
        """codex (not in either set) should print the warning but not error."""
        r = self._run("codex")
        assert r.returncode == 0, r.stderr
        assert "is not validated" in r.stdout


# ===========================================================================
#  --config strict mode: yaml is exclusive; any other CLI flag is an error
# ===========================================================================


class TestConfigStrictMode:
    """`-c file.yaml` alongside any other CLI flag must hard-error."""

    def _run(self, tmp_path, yaml_body, *cli_extra, expect_rc=2):
        import shutil as _shutil

        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml_body)
        cli = _shutil.which("tasktrace-capture") or "tasktrace-capture"
        cmd = [cli, "-c", str(yaml_path), *cli_extra]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if expect_rc is not None:
            assert r.returncode == expect_rc, (
                f"Expected rc={expect_rc}, got {r.returncode}\n" f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
            )
        return r

    def test_yaml_only_no_extra_flags_reaches_dry_run(self, tmp_path):
        """Yaml-only invocation must not trip the strict-mode check."""
        body = (
            "dataset: hello-world\n"
            "api_base: http://placeholder:8000/v1\n"  # default agent terminus-2 requires this
            "skip_preflight: true\n"
            "dry_run: true\n"
        )
        r = self._run(tmp_path, body, expect_rc=0)
        assert "Dry run" in r.stdout

    def test_single_extra_flag_errors(self, tmp_path):
        body = "dataset: hello-world\nskip_preflight: true\n"
        r = self._run(tmp_path, body, "--concurrency", "1")
        assert "cannot mix yaml with CLI flags" in r.stderr
        assert "--concurrency 1" in r.stderr
        assert "Edit the yaml or drop --config" in r.stderr

    def test_multi_extra_flags_listed(self, tmp_path):
        body = "dataset: hello-world\nskip_preflight: true\n"
        r = self._run(
            tmp_path,
            body,
            "--concurrency",
            "4",
            "--runtime",
            "docker",
            "--agent-env",
            "K=v",
        )
        assert "--concurrency 4" in r.stderr
        assert "--runtime docker" in r.stderr
        assert "--agent-env K=v" in r.stderr

    def test_store_true_flag_listed_without_value(self, tmp_path):
        body = "dataset: hello-world\n"
        r = self._run(tmp_path, body, "--skip-preflight")
        assert "--skip-preflight" in r.stderr
        # Make sure we don't print "--skip-preflight True" or similar
        assert "--skip-preflight True" not in r.stderr
        assert "--skip-preflight False" not in r.stderr

    def test_unknown_yaml_key_errors_cleanly(self, tmp_path):
        body = "dataset: hello-world\nbogus_key: 42\n"
        r = self._run(tmp_path, body)
        assert "unknown config key" in r.stderr
        assert "bogus_key" in r.stderr

    def test_parallel_docker_yaml_reaches_dry_run(self, tmp_path):
        body = (
            "dataset: hello-world\nagent: codex\nruntime: docker\n"
            "allow_parallel_docker: true\nskip_preflight: true\ndry_run: true\n"
        )
        assert "Dry run" in self._run(tmp_path, body, expect_rc=0).stdout

    def test_parallel_docker_yaml_rejects_other_runtime(self, tmp_path):
        body = "runtime: enroot\nallow_parallel_docker: true\n"
        r = self._run(tmp_path, body)
        assert "--allow-parallel-docker requires --runtime docker" in r.stderr

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from unittest.mock import patch

import pytest

from llmb_run.job_history import record_job_submission
from llmb_run.slurm_utils import SlurmJob
from llmb_run.tasks import WorkloadTask
from llmb_run.workload_validator import ValidationErrorType


def test_version_flag(runner, tmp_path, monkeypatch):
    """Without a configured install, --version prints only the launcher line."""
    from llmb_run.main import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMB_INSTALL", raising=False)
    with patch(
        "llmb_run.main.get_cluster_config", side_effect=AssertionError("get_cluster_config should not be called")
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert re.fullmatch(r"llmb-run\s+\S+\n", result.stdout)


def test_version_flag_with_recipe(runner, tmp_path, monkeypatch):
    """When cluster_config.yaml + release.yaml resolve, --version prints both lines."""
    from llmb_run.main import app

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "release.yaml").write_text("llmb_version: '26.06.00-dev'\n")
    (tmp_path / "cluster_config.yaml").write_text(
        f"schema_version: 2\ngpu_type: h100\nllmb_install: {tmp_path}\nllmb_repo: {repo}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMB_INSTALL", raising=False)

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert re.fullmatch(
        rf"llmb-run\s+\S+\nRecipe Version 26\.06\.00-dev \({re.escape(str(repo.resolve()))}\)\n",
        result.stdout,
    )


def test_version_flag_with_recipe_from_llmb_install(runner, tmp_path, monkeypatch):
    """When CWD lacks config, --version discovers recipe version via $LLMB_INSTALL."""
    from llmb_run.main import app

    cwd = tmp_path / "cwd"
    install = tmp_path / "install"
    repo = tmp_path / "repo"
    cwd.mkdir()
    install.mkdir()
    repo.mkdir()
    (repo / "release.yaml").write_text("llmb_version: '26.07.00-dev'\n")
    (install / "cluster_config.yaml").write_text(
        f"schema_version: 2\ngpu_type: h100\nllmb_install: {install}\nllmb_repo: {repo}\n"
    )
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("LLMB_INSTALL", str(install))

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert re.fullmatch(
        rf"llmb-run\s+\S+\nRecipe Version 26\.07\.00-dev \({re.escape(str(repo.resolve()))}\)\n",
        result.stdout,
    )


def test_version_flag_silent_on_broken_release(runner, tmp_path, monkeypatch):
    """A malformed release.yaml must not error; --version falls back to one line."""
    from llmb_run.main import app

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "release.yaml").write_text("::: not valid yaml: [\n")
    (tmp_path / "cluster_config.yaml").write_text(
        f"schema_version: 2\ngpu_type: h100\nllmb_install: {tmp_path}\nllmb_repo: {repo}\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLMB_INSTALL", raising=False)

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert re.fullmatch(r"llmb-run\s+\S+\n", result.stdout)


@pytest.mark.usefixtures("mock_dependencies")
class TestCLI:
    """CLI-level tests that exercise argument parsing and mode dispatching."""

    @pytest.mark.parametrize("command", ["single", "bulk", "submit-all"])
    def test_deprecated_commands_are_removed(self, command):
        """Deprecated commands should no longer be registered."""
        from llmb_run.main import app

        registered_commands = {
            command_info.name or command_info.callback.__name__.replace('_', '-')
            for command_info in app.registered_commands
        }

        assert command not in registered_commands

    def test_list_mode_basic(self, runner):
        """`list` mode should delegate to `print_avail_workloads`."""
        from llmb_run.main import app

        # Capture that our stub gets invoked
        def _fake_print_avail(workloads, *_args, **_kwargs):
            print("AVAILABLE_WORKLOADS")
            # Basic sanity – ensure patched workloads propagate
            assert "training_foo" in workloads

        with patch('llmb_run.main.print_avail_workloads', _fake_print_avail):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "AVAILABLE_WORKLOADS" in result.stdout

    def test_submit_help_groups_slurm_flags(self, runner):
        """Submit help should expose the dedicated Slurm help panel."""
        from llmb_run.main import app

        result = runner.invoke(app, ["submit", "--help"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "Slurm" in result.stdout
        assert "--nodelist" in result.stdout
        assert "--slurm-arg" in result.stdout

    def test_jobs_default_lists_history(self, runner, tmp_path, cluster_config_factory):
        """`jobs` without a subcommand should list known job history."""
        from llmb_run.main import app

        llmb_install = tmp_path / "llmb"
        cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
        log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
        log_dir.mkdir(parents=True)
        task = WorkloadTask("training_foo", "7b", "fp16", 8)
        workloads = {
            "training_foo": {
                "metadata": {"general": {"model": "foo"}, "run": {"launcher_type": "nemo"}},
                "workload": "foo",
            }
        }
        record_job_submission(
            cluster_config,
            task,
            SlurmJob(job_id=12345, job_workdir=str(log_dir), llmb_config_path=None),
            workloads,
        )

        with (
            patch('llmb_run.main.get_cluster_config', return_value=cluster_config),
            patch('llmb_run.main.refresh_non_terminal_jobs', return_value=(0, None)),
        ):
            result = runner.invoke(app, ["jobs"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "foo_7b" in result.stdout
        assert "12345" in result.stdout

    def test_jobs_list_warns_on_sacct_failure(self, runner, tmp_path, cluster_config_factory, caplog):
        """`jobs` should still render the table and surface a warning when sacct fails."""
        from llmb_run.main import app

        llmb_install = tmp_path / "llmb"
        cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
        log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
        log_dir.mkdir(parents=True)
        task = WorkloadTask("training_foo", "7b", "fp16", 8)
        workloads = {
            "training_foo": {
                "metadata": {"general": {"model": "foo"}, "run": {"launcher_type": "nemo"}},
                "workload": "foo",
            }
        }
        record_job_submission(
            cluster_config,
            task,
            SlurmJob(job_id=12345, job_workdir=str(log_dir), llmb_config_path=None),
            workloads,
        )

        with (
            patch('llmb_run.main.get_cluster_config', return_value=cluster_config),
            patch('llmb_run.main.refresh_non_terminal_jobs', return_value=(0, "sacct query failed")),
        ):
            result = runner.invoke(app, ["jobs"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "12345" in result.stdout
        assert "sacct unavailable" in caplog.text

    def test_jobs_refresh_requires_job_id(self, runner):
        """`jobs refresh` is an explicit force-refresh command and requires IDs."""
        from llmb_run.main import app

        result = runner.invoke(app, ["jobs", "refresh"])

        assert result.exit_code != 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"

    def test_jobs_refresh_forces_requested_jobs(self, runner, tmp_path, cluster_config_factory):
        """`jobs refresh <id...>` should force-refresh the requested job records."""
        from llmb_run.main import app

        llmb_install = tmp_path / "llmb"
        cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

        with (
            patch('llmb_run.main.get_cluster_config', return_value=cluster_config),
            patch('llmb_run.main.get_job', return_value=object()),
            patch('llmb_run.main.refresh_requested_jobs', return_value=(2, None)) as mock_refresh,
        ):
            result = runner.invoke(app, ["jobs", "refresh", "123", "456"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        mock_refresh.assert_called_once_with(cluster_config, [123, 456])

    def test_jobs_refresh_errors_for_missing_job_id(self, runner, tmp_path, cluster_config_factory, caplog):
        """`jobs refresh <id>` should only operate on jobs already in history."""
        from llmb_run.main import app

        llmb_install = tmp_path / "llmb"
        cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

        with (
            patch('llmb_run.main.get_cluster_config', return_value=cluster_config),
            patch('llmb_run.main.get_job', return_value=None),
            patch('llmb_run.main.refresh_requested_jobs') as mock_refresh,
        ):
            result = runner.invoke(app, ["jobs", "refresh", "123"])

        assert result.exit_code != 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "Job ID(s) not found in llmb-run history: 123" in caplog.text
        mock_refresh.assert_not_called()

    def test_jobs_refresh_exits_nonzero_on_sacct_failure(self, runner, tmp_path, cluster_config_factory, caplog):
        """`jobs refresh` should fail loudly when sacct can't be queried."""
        from llmb_run.main import app

        llmb_install = tmp_path / "llmb"
        cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

        with (
            patch('llmb_run.main.get_cluster_config', return_value=cluster_config),
            patch('llmb_run.main.get_job', return_value=object()),
            patch('llmb_run.main.refresh_requested_jobs', return_value=(0, "sacct query failed")),
        ):
            result = runner.invoke(app, ["jobs", "refresh", "123"])

        assert result.exit_code != 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "sacct unavailable" in caplog.text

    def test_submit_file_mode_simple_file_dryrun(self, runner, tmp_path):
        """File-based submit with a simple-format spec should honor --dry-run."""
        from llmb_run.main import app

        run_tests_called = False

        def _fake_run_tests(*_args, **_kwargs):
            nonlocal run_tests_called
            run_tests_called = True

        # Create a simple-format task specification
        spec_path = tmp_path / "tasks.txt"
        spec_path.write_text("""training_foo_7b:
('fp16', [8, 16], 1)

        """)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["submit", "-f", str(spec_path), "--dry-run"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert run_tests_called is False

    def test_submit_file_mode_simple_file_launch(self, runner, tmp_path):
        """File-based submit should parse tasks and forward them to run_tests."""
        from llmb_run.main import app

        recorded_tasks = {}

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks["count"] = len(tasks)

        # Multi-line spec with various permutations
        spec_content = """training_foo_7b:
('fp16', [8, 16], 1)
(['bf16', 'fp16'], [32], 2, True)
"""
        spec_path = tmp_path / "tasks.txt"
        spec_path.write_text(spec_content)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["submit", "-f", str(spec_path)])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        # Expected task expansion: 2 (first line) + 4 (second line) = 6
        assert recorded_tasks.get("count") == 6

    def test_submit_force_rejects_multi_value_scale(self, runner, caplog):
        """`submit --force` should stay limited to one exact task."""
        from llmb_run.main import app

        result = runner.invoke(
            app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "8,16", "--force"]
        )

        assert result.exit_code != 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "--force only supports a single explicit task" in caplog.text


def test_bulk_validation_rejects_invalid_tasks(runner, tmp_path, cluster_config_factory):
    """Test that bulk validation properly rejects invalid task configurations."""
    from llmb_run.main import app

    workloads = {
        "training_strict": {
            "metadata": {
                "run": {
                    "launcher_type": "slurm",
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "7b",
                                    "dtypes": ["fp8"],  # Only fp8 supported
                                    "scales": [8, 16],
                                }
                            ]
                        }
                    },
                }
            },
            "dir": "/tmp",
            "workload": "strict",
            "workload_type": "training",
        }
    }

    cluster_config = cluster_config_factory(
        {
            "workloads": {"installed": ["training_strict"]},
        }
    )

    # Create task spec with invalid dtype (bf16 not supported)
    spec_content = """training_strict_7b:
(['fp8', 'bf16'], [8], 1)
"""
    spec_path = tmp_path / "tasks.txt"
    spec_path.write_text(spec_content)

    submitted_tasks = []

    def mock_run_tests(config, task_list, workloads, **kwargs):
        submitted_tasks.extend(task_list)

    # This test does NOT use mock_dependencies, so real validation runs.
    # Only fp8 tasks should pass validation; bf16 tasks should be rejected.
    with (
        patch('llmb_run.main.get_cluster_config', return_value=cluster_config),
        patch('llmb_run.main.get_workloads', return_value=workloads),
        patch('llmb_run.main.run_tests', mock_run_tests),
    ):
        result = runner.invoke(app, ["submit", "-f", str(spec_path)])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"

    # Check that only valid tasks were submitted (only fp8 task)
    # If the real validation logic runs, it should filter out bf16 if configured correctly.
    # The test sets up `training_strict` which only has `fp8`.
    assert len(submitted_tasks) == 1
    assert submitted_tasks[0].dtype == "fp8"


def test_bulk_validation_aborts_all_invalid_tasks(runner, tmp_path, cluster_config_factory):
    """Test that bulk validation aborts when all tasks are invalid."""
    from llmb_run.main import app

    workloads = {
        "training_strict": {
            "metadata": {
                "run": {
                    "launcher_type": "slurm",
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "7b",
                                    "dtypes": ["fp8"],  # Only fp8 supported
                                    "scales": [8, 16],
                                }
                            ]
                        }
                    },
                }
            },
            "dir": "/tmp",
            "workload": "strict",
            "workload_type": "training",
        }
    }

    cluster_config = cluster_config_factory(
        {
            "workloads": {"installed": ["training_strict"]},
        }
    )

    # Create task spec with only invalid dtype
    spec_content = """training_strict_7b:
('bf16', [8], 1)
"""
    spec_path = tmp_path / "tasks.txt"
    spec_path.write_text(spec_content)

    with (
        patch('llmb_run.main.get_cluster_config', return_value=cluster_config),
        patch('llmb_run.main.get_workloads', return_value=workloads),
    ):
        result = runner.invoke(app, ["submit", "-f", str(spec_path)])

    assert result.exit_code == 1, f"Exit code: {result.exit_code}, Output: {result.stdout}"


@pytest.mark.usefixtures("mock_dependencies")
def test_submit_discovery_mode_dryrun(runner):
    """Test discovery-mode submit with --dry-run flag."""
    from llmb_run.main import app

    # Mock the discovery task generation
    def _fake_generate_submit_all_tasks(workloads, cluster_config, max_scale, repeats, profile, **kwargs):
        return [
            WorkloadTask("training_foo", "7b", "fp8", 32, False),
            WorkloadTask("training_foo", "7b", "bf16", 32, False),
        ]

    # Mock validation to return all tasks as valid
    def _fake_validate_bulk_tasks(task_list, workloads, cluster_config):
        return task_list, {}

    run_tests_called = False

    def _fake_run_tests(*_args, **_kwargs):
        nonlocal run_tests_called
        run_tests_called = True

    with (
        patch('llmb_run.task_generation.generate_submit_all_tasks', _fake_generate_submit_all_tasks),
        patch('llmb_run.main.validate_bulk_tasks', _fake_validate_bulk_tasks),
        patch('llmb_run.main.run_tests', _fake_run_tests),
    ):
        result = runner.invoke(app, ["submit", "--max-scale", "256", "--repeats", "2", "--dry-run"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert run_tests_called is False


@pytest.mark.usefixtures("mock_dependencies")
def test_submit_discovery_mode_validation_integration(runner):
    """Test discovery-mode submit integrates with validation properly."""
    from llmb_run.main import app

    # Mock task generation to return tasks with validation issues
    def _fake_generate_submit_all_tasks(workloads, cluster_config, max_scale, repeats, profile, **kwargs):
        return [
            WorkloadTask("training_foo", "7b", "fp8", 32, False),  # Valid
            WorkloadTask("invalid_workload", "7b", "fp8", 32, False),  # Invalid
        ]

    def _fake_validate_bulk_tasks(task_list, workloads, cluster_config):
        # Return one valid task and one error
        valid_tasks = [task_list[0]]  # First task is valid
        error_summary = {
            ('error_key',): {
                'count': 1,
                'error_type': ValidationErrorType.WORKLOAD_NOT_FOUND,
                'error_msg': 'Test error message',
                'suggestions': ['training_foo'],
                'example_task': task_list[1],
            }
        }
        return valid_tasks, error_summary

    def _fake_run_tests(*_args, **_kwargs):
        pass  # Don't actually run tests

    with (
        patch('llmb_run.task_generation.generate_submit_all_tasks', _fake_generate_submit_all_tasks),
        patch('llmb_run.main.validate_bulk_tasks', _fake_validate_bulk_tasks),
        patch('llmb_run.main.run_tests', _fake_run_tests),
    ):
        result = runner.invoke(app, ["submit", "--max-scale", "128"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"


@pytest.mark.usefixtures("mock_dependencies")
def test_submit_discovery_filtering_flags(runner):
    """Test discovery-mode submit with filtering flags."""
    from llmb_run.main import app

    # Track the call to generate_submit_all_tasks to verify parameters
    generate_args = {}

    def _track_generate_submit_all_tasks(*args, **kwargs):
        generate_args.update(kwargs)
        return [WorkloadTask("training_foo", "7b", "fp8", 32, False)]

    with (
        patch('llmb_run.task_generation.generate_submit_all_tasks', _track_generate_submit_all_tasks),
        patch('llmb_run.main.validate_bulk_tasks', lambda *args: (args[0], {})),
        patch('llmb_run.main.run_tests', lambda *args, **kwargs: None),
    ):
        result = runner.invoke(
            app,
            [
                "submit",
                "--max-scale",
                "128",
                "--min-scale",
                "--dtype",
                "fp8,bf16",
                "--workload",
                "training_foo,training_bar",
            ],
        )

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    # Verify that the new parameters were passed correctly
    assert generate_args["min_scale"] is True
    assert generate_args["dtype_filter"] == ["fp8", "bf16"]
    assert generate_args["workload_filter"] == ["training_foo", "training_bar"]


@pytest.mark.usefixtures("mock_dependencies")
def test_submit_discovery_filtering_flags_with_spaces(runner):
    """Test discovery-mode submit with filtering flags containing spaces."""
    from llmb_run.main import app

    # Track the call to generate_submit_all_tasks to verify parameters
    generate_args = {}

    def _track_generate_submit_all_tasks(*args, **kwargs):
        generate_args.update(kwargs)
        return [WorkloadTask("training_foo", "7b", "fp8", 32, False)]

    with (
        patch('llmb_run.task_generation.generate_submit_all_tasks', _track_generate_submit_all_tasks),
        patch('llmb_run.main.validate_bulk_tasks', lambda *args: (args[0], {})),
        patch('llmb_run.main.run_tests', lambda *args, **kwargs: None),
    ):
        result = runner.invoke(
            app,
            [
                "submit",
                "--max-scale",
                "128",
                "--dtype",
                " fp8 , bf16 ",
                "--workload",
                " training_foo , training_bar ",
            ],
        )

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    # Verify that spaces are stripped correctly
    assert generate_args["dtype_filter"] == ["fp8", "bf16"]
    assert generate_args["workload_filter"] == ["training_foo", "training_bar"]


@pytest.mark.usefixtures("mock_dependencies")
def test_submit_missing_scale_args(runner):
    """Test that submit requires scale selection outside file mode."""
    from llmb_run.main import app

    result = runner.invoke(app, ["submit"])
    assert result.exit_code == 1, f"Exit code: {result.exit_code}, Output: {result.stdout}"


@pytest.mark.usefixtures("mock_dependencies")
def test_submit_discovery_min_scale_only(runner):
    """Test discovery-mode submit with only --min-scale flag."""
    from llmb_run.main import app

    # Track the call to generate_submit_all_tasks to verify parameters
    generate_args = {}

    def _track_generate_submit_all_tasks(*args, **kwargs):
        generate_args.update(kwargs)
        generate_args['max_scale_arg'] = args[2]  # max_scale is the 3rd positional argument
        return [WorkloadTask("training_foo", "7b", "fp8", 32, False)]

    with (
        patch('llmb_run.task_generation.generate_submit_all_tasks', _track_generate_submit_all_tasks),
        patch('llmb_run.main.validate_bulk_tasks', lambda *args: (args[0], {})),
        patch('llmb_run.main.run_tests', lambda *args, **kwargs: None),
    ):
        result = runner.invoke(app, ["submit", "--min-scale"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    # Verify that max_scale is None and min_scale is True
    assert generate_args["min_scale"] is True
    assert generate_args["max_scale_arg"] is None


@pytest.mark.usefixtures("mock_dependencies")
def test_submit_discovery_max_scale_only(runner):
    """Test discovery-mode submit with only --max-scale flag."""
    from llmb_run.main import app

    # Track the call to generate_submit_all_tasks to verify parameters
    generate_args = {}

    def _track_generate_submit_all_tasks(*args, **kwargs):
        generate_args.update(kwargs)
        generate_args['max_scale_arg'] = args[2]  # max_scale is the 3rd positional argument
        return [WorkloadTask("training_foo", "7b", "fp8", 32, False)]

    with (
        patch('llmb_run.task_generation.generate_submit_all_tasks', _track_generate_submit_all_tasks),
        patch('llmb_run.main.validate_bulk_tasks', lambda *args: (args[0], {})),
        patch('llmb_run.main.run_tests', lambda *args, **kwargs: None),
    ):
        result = runner.invoke(app, ["submit", "--max-scale", "128"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    # Verify that max_scale is set and min_scale is False
    assert generate_args["min_scale"] is False
    assert generate_args["max_scale_arg"] == 128

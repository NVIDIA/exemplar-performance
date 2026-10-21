# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from llmb_run.config_manager import ClusterConfig
from llmb_run.tasks import WorkloadTask


@pytest.mark.usefixtures("mock_dependencies")
class TestSubmitCommand:
    """Tests for the unified 'submit' command."""

    def test_submit_dryrun(self, runner, caplog):
        """Test basic dryrun submission."""
        from llmb_run.main import app

        run_tests_called = False

        def _fake_run_tests(*_args, **_kwargs):
            nonlocal run_tests_called
            run_tests_called = True

        # Using explicit workload mode: -w ... -s ... --scale ...
        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "8", "--dry-run"]
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert run_tests_called is False
        assert "Dry run enabled" in caplog.text

    def test_submit_file_mode(self, runner, tmp_path):
        """Test submission from file."""
        from llmb_run.main import app

        recorded_tasks = {}

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks["count"] = len(tasks)

        spec_path = tmp_path / "tasks.txt"
        spec_path.write_text("training_foo_7b:\n('fp16', [8], 1)\n")

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["submit", "-f", str(spec_path)])

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert recorded_tasks.get("count") == 1

    def test_submit_discovery_mode(self, runner):
        """Test discovery mode (max-scale)."""
        from llmb_run.main import app

        # Mock the task generation to return something
        def _fake_gen_tasks(*args, **kwargs):
            return [WorkloadTask("training_foo", "7b", "fp16", 8)]

        with (
            patch('llmb_run.task_generation.generate_submit_all_tasks', _fake_gen_tasks),
            patch('llmb_run.main.run_tests') as mock_run,
        ):
            result = runner.invoke(app, ["submit", "--max-scale", "16"])

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert mock_run.called

    def test_submit_validation_error_mutual_exclusivity(self, runner, caplog):
        """Test validation error when mixing incompatible flags."""
        from llmb_run.main import app

        # Mixing --file and --workload
        result = runner.invoke(app, ["submit", "-f", "foo.txt", "-w", "bar"])
        assert result.exit_code != 0
        assert "Cannot mix --file with workload specifications" in caplog.text

        # Mixing --scale and --max-scale
        caplog.clear()
        result = runner.invoke(app, ["submit", "--scale", "8", "--max-scale", "16"])
        assert result.exit_code != 0
        assert "Cannot use --scale with --max-scale" in caplog.text

    def test_submit_explicit_mode_requires_scale(self, runner, caplog):
        """Test that explicit mode (-w and -s) requires scale specification."""
        from llmb_run.main import app

        # Using -w and -s without any scale parameter should error
        result = runner.invoke(app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16"])
        assert result.exit_code != 0
        assert "Must specify scale parameter" in caplog.text
        assert "--scale" in caplog.text or "--max-scale" in caplog.text

    def test_submit_discovery_mode_requires_scale(self, runner, caplog):
        """Test that discovery mode also requires scale specification."""
        from llmb_run.main import app

        # Using -w without model size but also without scale should error
        result = runner.invoke(app, ["submit", "-w", "training_foo_7b", "--dtype", "fp16"])
        assert result.exit_code != 0
        assert "Must specify scale parameter" in caplog.text

    def test_submit_file_mode_with_nice_txt(self, runner, tmp_path):
        """Test that --nice parameter is propagated to tasks from .txt file."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        spec_path = tmp_path / "tasks.txt"
        spec_path.write_text("training_foo_7b:\n('fp16', [8, 16], 1)\n")

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["submit", "-f", str(spec_path), "--nice", "100"])

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 2  # Two tasks from two scales
        # Verify all tasks carry canonical Slurm args with nice=100
        for task in recorded_tasks:
            assert task.slurm_args is not None
            assert task.slurm_args.get_named_param('nice') == '100'

    def test_submit_file_mode_with_nice_yaml(self, runner, tmp_path):
        """Test that --nice parameter is propagated to tasks from .yaml file."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        spec_path = tmp_path / "tasks.yaml"
        spec_content = """training_foo_7b:
  dtypes: [fp16]
  scales: [8, 16]
  repeats: 1
  tasks:
    - dtypes: [fp16]
      scales: [8, 16]
"""
        spec_path.write_text(spec_content)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["submit", "-f", str(spec_path), "--nice", "100"])

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 2  # Two tasks from two scales
        # Verify all tasks carry canonical Slurm args with nice=100
        for task in recorded_tasks:
            assert task.slurm_args is not None
            assert task.slurm_args.get_named_param('nice') == '100'

    def test_submit_file_mode_with_repeatable_slurm_args(self, runner, tmp_path):
        """Repeatable --slurm-arg values should be propagated to generated tasks."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        spec_path = tmp_path / "tasks.txt"
        spec_path.write_text("training_foo_7b:\n('fp16', [8], 1)\n")

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-f",
                    str(spec_path),
                    "--slurm-arg",
                    "exclusive",
                    "--slurm-arg",
                    "constraint=gpu",
                ],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].slurm_args is not None
        assert recorded_tasks[0].slurm_args.to_additional_slurm_params() == "exclusive;constraint=gpu"

    def test_submit_explicit_mode_with_repeatable_env(self, runner):
        """Repeatable --env values should propagate to task env and explicit-env intent."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-w",
                    "training_foo",
                    "-s",
                    "7b",
                    "--dtype",
                    "fp16",
                    "--scale",
                    "8",
                    "--env",
                    "HF_TOKEN=abc123",
                    "--env",
                    "NCCL_DEBUG=INFO",
                ],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].env_overrides == {"HF_TOKEN": "abc123", "NCCL_DEBUG": "INFO"}
        assert recorded_tasks[0].explicit_env_overrides == {"HF_TOKEN": "abc123", "NCCL_DEBUG": "INFO"}
        assert recorded_tasks[0].extra_workload_args == ()

    def test_submit_explicit_mode_with_dump_env(self, runner):
        """--dump-env should propagate as an extra workload arg."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-w",
                    "training_foo",
                    "-s",
                    "7b",
                    "--dtype",
                    "fp16",
                    "--scale",
                    "8",
                    "--dump-env",
                ],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].extra_workload_args == ("--dump_env",)

    def test_submit_explicit_mode_with_mbridge_args(self, runner, mock_workloads):
        """Repeatable --mbridge-arg values should preserve exact token order."""
        from llmb_run.main import app

        mock_workloads["training_foo"]["metadata"]["run"]["launcher_type"] = "megatron_bridge"
        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-w",
                    "training_foo",
                    "-s",
                    "7b",
                    "--dtype",
                    "fp16",
                    "--scale",
                    "8",
                    "--mbridge-arg=--micro_batch_size=2",
                    "--mbridge-arg=model.hidden_size=4096",
                ],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].mbridge_args == ("--micro_batch_size=2", "model.hidden_size=4096")
        assert recorded_tasks[0].extra_workload_args == ()

    def test_submit_warns_and_ignores_mbridge_args_for_other_launchers(self, runner, caplog):
        """MBridge arguments should not reach incompatible workload tasks."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-w",
                    "training_foo",
                    "-s",
                    "7b",
                    "--dtype",
                    "fp16",
                    "--scale",
                    "8",
                    "--mbridge-arg=--micro_batch_size=2",
                ],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert recorded_tasks[0].mbridge_args == ()
        assert "Ignoring --mbridge-arg for non-Megatron-Bridge workloads: training_foo" in caplog.text

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "--dataset_paths /data/one",  # whitespace: word-splits into two argv tokens
        ],
    )
    def test_submit_rejects_invalid_mbridge_arg_tokens(self, runner, caplog, value):
        """Each --mbridge-arg occurrence must contain one non-empty token without whitespace."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                f"--mbridge-arg={value}",
            ],
        )

        assert result.exit_code != 0
        assert "`--mbridge-arg`" in caplog.text

    def test_submit_accepts_raw_mbridge_args(self, runner, mock_workloads):
        """The CLI should preserve shell syntax without attempting to validate it."""
        from llmb_run.main import app

        mock_workloads["training_foo"]["metadata"]["run"]["launcher_type"] = "megatron_bridge"
        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-w",
                    "training_foo",
                    "-s",
                    "7b",
                    "--dtype",
                    "fp16",
                    "--scale",
                    "8",
                    "--mbridge-arg=model.layers=[1,2,3]",
                    "--mbridge-arg=model.cfg={a:1,b:2}",
                    "--mbridge-arg=paths=*.bin",
                    "--mbridge-arg=model.foo='x'",
                ],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert recorded_tasks[0].mbridge_args == (
            "model.layers=[1,2,3]",
            "model.cfg={a:1,b:2}",
            "paths=*.bin",
            "model.foo='x'",
        )

    def test_submit_file_mode_with_env_merges_and_cli_wins(self, runner, tmp_path):
        """CLI --env should merge into file task envs and take precedence on collisions."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        spec_path = tmp_path / "tasks.yaml"
        spec_path.write_text("""training_foo_7b:
  defaults:
    env:
      HF_TOKEN: from-file
      WANDB_MODE: offline
  tasks:
    - dtypes: [fp16]
      scales: [8]
      overrides:
        env:
          NCCL_DEBUG: WARN
""")

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-f",
                    str(spec_path),
                    "--env",
                    "HF_TOKEN=from-cli",
                    "--env",
                    "CUDA_DEVICE_MAX_CONNECTIONS=1",
                ],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].env_overrides == {
            "HF_TOKEN": "from-cli",
            "WANDB_MODE": "offline",
            "NCCL_DEBUG": "WARN",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
        assert recorded_tasks[0].explicit_env_overrides == {
            "HF_TOKEN": "from-cli",
            "WANDB_MODE": "offline",
            "NCCL_DEBUG": "WARN",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }

    def test_submit_file_mode_with_dump_env(self, runner, tmp_path):
        """--dump-env should apply to all file-mode tasks as a CLI-only modifier."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        spec_path = tmp_path / "tasks.txt"
        spec_path.write_text("training_foo_7b:\n('fp16', [8], 1)\n")

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["submit", "-f", str(spec_path), "--dump-env"])

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].extra_workload_args == ("--dump_env",)

    def test_submit_file_mode_yaml_env_populates_explicit_overrides(self, runner, tmp_path):
        """YAML `env:` blocks alone (no CLI --env) should flow into explicit_env_overrides."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        spec_path = tmp_path / "tasks.yaml"
        spec_path.write_text("""training_foo_7b:
  defaults:
    env:
      WANDB_MODE: offline
  tasks:
    - dtypes: [fp16]
      scales: [8]
      overrides:
        env:
          NCCL_DEBUG: WARN
""")

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["submit", "-f", str(spec_path)])

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].env_overrides == {"WANDB_MODE": "offline", "NCCL_DEBUG": "WARN"}
        assert recorded_tasks[0].explicit_env_overrides == {"WANDB_MODE": "offline", "NCCL_DEBUG": "WARN"}

    def test_submit_rejects_duplicate_cli_env_key(self, runner, caplog):
        """Repeatable --env values should reject duplicate keys."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                "--env",
                "HF_TOKEN=one",
                "--env",
                "HF_TOKEN=two",
            ],
        )

        assert result.exit_code != 0
        assert "Duplicate environment variable 'HF_TOKEN'" in caplog.text

    def test_submit_rejects_invalid_cli_env_assignment(self, runner, caplog):
        """--env should require KEY=value assignments with a valid env-var name."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                "--env",
                "HF-TOKEN",
            ],
        )

        assert result.exit_code != 0
        assert "must be in `KEY=value` form" in caplog.text

    def test_submit_rejects_invalid_cli_env_name(self, runner, caplog):
        """--env should reject names that are not valid bash-style env vars."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                "--env",
                "HF-TOKEN=value",
            ],
        )

        assert result.exit_code != 0
        assert "variable name 'HF-TOKEN' is invalid" in caplog.text

    def test_submit_rejects_shell_unsafe_cli_env_value(self, runner, caplog):
        """--env values with shell-special characters should be rejected until upstream fix lands."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                "--env",
                "NCCL_DEBUG=Value with spaces!",
            ],
        )

        assert result.exit_code != 0
        assert "shell-special characters" in caplog.text
        assert "NCCL_DEBUG" in caplog.text

    def test_submit_rejects_shell_unsafe_yaml_env_value(self, runner, caplog, tmp_path):
        """YAML `env:` values with shell-special characters should be rejected at parse time."""
        from llmb_run.main import app

        spec_path = tmp_path / "tasks.yaml"
        spec_path.write_text("""training_foo_7b:
  defaults:
    env:
      NCCL_DEBUG: Value with spaces!
  tasks:
    - dtypes: [fp16]
      scales: [8]
""")

        result = runner.invoke(app, ["submit", "-f", str(spec_path)])

        assert result.exit_code != 0
        assert "shell-special characters" in caplog.text
        assert "NCCL_DEBUG" in caplog.text

    def test_submit_rejects_invalid_yaml_env_name(self, runner, caplog, tmp_path):
        """YAML `env:` keys should use the same shell-style variable validation as --env."""
        from llmb_run.main import app

        spec_path = tmp_path / "tasks.yaml"
        spec_path.write_text("""training_foo_7b:
  defaults:
    env:
      HF-TOKEN: abc123
  tasks:
    - dtypes: [fp16]
      scales: [8]
""")

        result = runner.invoke(app, ["submit", "-f", str(spec_path)])

        assert result.exit_code != 0
        assert "YAML env variable name 'HF-TOKEN' is invalid" in caplog.text

    def test_submit_rejects_duplicate_cli_slurm_key(self, runner, caplog):
        """First-class Slurm flags should conflict with duplicate --slurm-arg keys."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                "--reservation",
                "res_a",
                "--slurm-arg",
                "reservation=res_b",
            ],
        )

        assert result.exit_code != 0
        assert "`reservation` has a dedicated flag" in caplog.text

    def test_submit_rejects_raw_slurm_arg_token(self, runner, caplog):
        """--slurm-arg values should not include the raw leading '--' token."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                "--slurm-arg",
                "--constraint=gpu",
            ],
        )

        assert result.exit_code != 0
        assert "should not include a leading '--'" in caplog.text

    def test_submit_rejects_first_class_slurm_arg_passthrough(self, runner, caplog):
        """First-class Slurm options should direct users to the dedicated flags."""
        from llmb_run.main import app

        result = runner.invoke(
            app,
            [
                "submit",
                "-w",
                "training_foo",
                "-s",
                "7b",
                "--dtype",
                "fp16",
                "--scale",
                "8",
                "--slurm-arg",
                "segment=5",
            ],
        )

        assert result.exit_code != 0
        assert "`segment` has a dedicated flag" in caplog.text

    def test_submit_dry_run_rejects_workload_additional_slurm_params_conflict(self, runner, caplog):
        """Dry-run should catch workload-level ADDITIONAL_SLURM_PARAMS conflicts."""
        from llmb_run.main import app

        test_workloads = {
            "training_foo": {
                "metadata": {
                    "run": {
                        "launcher_type": "sbatch",
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp16"],
                                        "scales": [8],
                                    }
                                ]
                            }
                        },
                    }
                },
                "dir": "/tmp",
                "workload": "foo",
                "workload_type": "pretrain",
            }
        }

        cluster_config = ClusterConfig.from_dict(
            {
                "launcher": {
                    "llmb_install": "/tmp/llmb_install",
                    "llmb_repo": "/tmp/llmb_repo",
                    "gpu_type": "h100",
                },
                "slurm": {"gpu_partition": "batch"},
                "workloads": {
                    "installed": ["training_foo"],
                    "config": {
                        "training_foo": {
                            "environment": {"ADDITIONAL_SLURM_PARAMS": "reservation=res_a"},
                        }
                    },
                },
            }
        )

        with (
            patch("llmb_run.main.get_cluster_config", return_value=cluster_config),
            patch("llmb_run.main.get_workloads", return_value=test_workloads),
        ):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-w",
                    "training_foo",
                    "-s",
                    "7b",
                    "--dtype",
                    "fp16",
                    "--scale",
                    "8",
                    "--dry-run",
                    "--nodelist",
                    "node001",
                ],
            )

        assert result.exit_code != 0
        assert "Cannot combine first-class Slurm CLI flags with ADDITIONAL_SLURM_PARAMS" in caplog.text

    def test_submit_zero_jobs_shows_helpful_context(self, runner, caplog):
        """Test that zero-jobs error provides helpful context about available configurations.

        This test validates the user experience when no jobs match their criteria,
        checking for helpful output without being brittle to exact formatting changes.
        """
        from llmb_run.main import app

        # Create minimal workloads fixture with restricted scales
        test_workloads = {
            "training_foo": {
                "metadata": {
                    "run": {
                        "launcher_type": "sbatch",
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp16", "bf16"],
                                        "scales": [8, 16, 32],  # 999999 not in this list
                                    }
                                ]
                            }
                        },
                    }
                },
                "dir": "/tmp",
                "workload": "foo",
                "workload_type": "pretrain",
            },
        }

        test_cluster_config = {
            "launcher": {
                "llmb_repo": "/tmp",
                "llmb_install": "/tmp",
                "gpu_type": "h100",
            },
            "slurm": {"gpu_partition": "batch"},
            "workloads": {
                "installed": ["training_foo"],
            },
        }

        # Test 1: CLI mode with invalid scale should trigger zero-jobs path and show helpful context
        with (
            patch('llmb_run.main.get_cluster_config', return_value=ClusterConfig.from_dict(test_cluster_config)),
            patch('llmb_run.main.get_workloads', return_value=test_workloads),
        ):
            result = runner.invoke(
                app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "999999"]
            )

        assert result.exit_code != 0, "Should exit with error when no jobs match criteria"

        # Check for core error concept (flexible to wording changes)
        output_lower = caplog.text.lower()
        assert any(
            phrase in output_lower for phrase in ["no matching", "no tasks", "not found", "error"]
        ), f"Should indicate error finding configurations. Output: {caplog.text}"

        # Check for helpful context - should show workload info or "available" configurations
        # This is flexible: we're not checking exact format, just that helpful info is present
        has_context = any(
            concept in caplog.text
            for concept in [
                "training_foo",  # Workload name
                "scales",  # Mention of scales
                "dtypes",  # Mention of dtypes
                "available",  # Shows what's available
                "[8, 16, 32]",  # Actual scale values
            ]
        )
        assert has_context, f"Should show helpful context about available configurations. Output: {caplog.text}"

    def test_error_handles_workload_modelsize_suffix(self, runner, caplog):
        """Error messages should strip model size suffix when showing available configs."""
        from llmb_run.main import app

        # Setup workload with restricted scales
        test_workloads = {
            "pretrain_foo": {
                "metadata": {
                    "run": {
                        "launcher_type": "sbatch",
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp16"],
                                        "scales": [8, 16],
                                    }
                                ]
                            }
                        },
                    }
                },
                "dir": "/tmp",
                "workload": "foo",
                "workload_type": "pretrain",
            }
        }

        test_cluster_config = {
            "launcher": {
                "llmb_repo": "/tmp",
                "llmb_install": "/tmp",
                "gpu_type": "h100",
            },
            "slurm": {"gpu_partition": "batch"},
            "workloads": {
                "installed": ["pretrain_foo"],
            },
        }

        # Try to submit with invalid scale using workload_modelsize pattern
        with (
            patch('llmb_run.main.get_cluster_config', return_value=ClusterConfig.from_dict(test_cluster_config)),
            patch('llmb_run.main.get_workloads', return_value=test_workloads),
        ):
            result = runner.invoke(app, ["submit", "-w", "pretrain_foo_7b", "--dtype", "fp16", "--scale", "999999"])

        assert result.exit_code != 0, "Should exit with error when no jobs match criteria"

        # Should show info for base workload 'pretrain_foo' (not 'pretrain_foo_7b')
        assert "pretrain_foo" in caplog.text
        # Should show available scales or configuration info
        has_scale_info = any(indicator in caplog.text.lower() for indicator in ["scales", "8", "16", "available"])
        assert has_scale_info, f"Should show scale information. Output: {caplog.text}"

    def test_error_strips_suffix_for_nonexistent_workload(self, runner, caplog):
        """Error should reference base workload name even when it doesn't exist."""
        from llmb_run.main import app

        test_workloads = {
            "pretrain_other": {
                "metadata": {
                    "run": {
                        "launcher_type": "sbatch",
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp8"],
                                        "scales": [128],
                                    }
                                ]
                            }
                        },
                    }
                },
                "dir": "/tmp",
                "workload": "other",
                "workload_type": "pretrain",
            }
        }

        test_cluster_config = {
            "launcher": {
                "llmb_repo": "/tmp",
                "llmb_install": "/tmp",
                "gpu_type": "h100",
            },
            "slurm": {"gpu_partition": "batch"},
            "workloads": {
                "installed": ["pretrain_other"],
            },
        }

        with (
            patch('llmb_run.main.get_cluster_config', return_value=ClusterConfig.from_dict(test_cluster_config)),
            patch('llmb_run.main.get_workloads', return_value=test_workloads),
        ):
            result = runner.invoke(
                app, ["submit", "-w", "pretrain_nonexistent_340b", "--dtype", "fp8", "--scale", "128"]
            )

        assert result.exit_code != 0, "Should exit with error for nonexistent workload"

        # Error message should reference stripped name or show it wasn't found
        # The key is it shouldn't show "pretrain_nonexistent_340b not found"
        # but rather work with the base "pretrain_nonexistent"
        output_lower = caplog.text.lower()
        assert "pretrain_nonexistent" in output_lower or "not found" in output_lower

    def test_error_handles_multiple_workloads_with_mixed_suffixes(self, runner, caplog):
        """Multiple workloads with mix of suffixes should all be parsed correctly."""
        from llmb_run.main import app

        test_workloads = {
            "pretrain_foo": {
                "metadata": {
                    "run": {
                        "launcher_type": "sbatch",
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp16"],
                                        "scales": [8, 16],
                                    }
                                ]
                            }
                        },
                    }
                },
                "dir": "/tmp",
                "workload": "foo",
                "workload_type": "pretrain",
            },
            "pretrain_bar": {
                "metadata": {
                    "run": {
                        "launcher_type": "sbatch",
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "13b",
                                        "dtypes": ["bf16"],
                                        "scales": [32, 64],
                                    }
                                ]
                            }
                        },
                    }
                },
                "dir": "/tmp",
                "workload": "bar",
                "workload_type": "pretrain",
            },
            "pretrain_baz": {
                "metadata": {
                    "run": {
                        "launcher_type": "sbatch",
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "340b",
                                        "dtypes": ["fp8"],
                                        "scales": [128, 256],
                                    }
                                ]
                            }
                        },
                    }
                },
                "dir": "/tmp",
                "workload": "baz",
                "workload_type": "pretrain",
            },
        }

        test_cluster_config = {
            "launcher": {
                "llmb_repo": "/tmp",
                "llmb_install": "/tmp",
                "gpu_type": "h100",
            },
            "slurm": {"gpu_partition": "batch"},
            "workloads": {
                "installed": ["pretrain_foo", "pretrain_bar", "pretrain_baz"],
            },
        }

        # Test: pretrain_foo_7b,pretrain_bar,pretrain_baz_340b (mixed with/without suffixes)
        with (
            patch('llmb_run.main.get_cluster_config', return_value=ClusterConfig.from_dict(test_cluster_config)),
            patch('llmb_run.main.get_workloads', return_value=test_workloads),
        ):
            result = runner.invoke(
                app,
                [
                    "submit",
                    "-w",
                    "pretrain_foo_7b,pretrain_bar,pretrain_baz_340b",
                    "--dtype",
                    "fp16",
                    "--scale",
                    "999999",
                ],
            )

        assert result.exit_code != 0, "Should exit with error when no jobs match criteria"

        # All three base workload names should appear in error context
        output_lower = caplog.text.lower()
        assert "pretrain_foo" in output_lower, "Should show pretrain_foo"
        assert "pretrain_bar" in output_lower, "Should show pretrain_bar"
        assert "pretrain_baz" in output_lower, "Should show pretrain_baz"

    def test_submit_with_proxy_flag(self, runner, caplog):
        """Test submit with --proxy flag sets task.proxy to True."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "8", "--proxy"]
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].proxy is True

    def test_submit_proxy_dryrun(self, runner, caplog):
        """Test submit with --proxy and --dry-run flags."""
        from llmb_run.main import app

        run_tests_called = False

        def _fake_run_tests(*_args, **_kwargs):
            nonlocal run_tests_called
            run_tests_called = True

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app,
                ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "8", "--proxy", "--dry-run"],
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert run_tests_called is False
        assert "Dry run enabled" in caplog.text

    def test_submit_force_bypasses_validation(self, runner, caplog, mock_cluster_config, mock_workloads):
        """--force should submit tasks that would otherwise be rejected by validation."""
        from llmb_run.main import app

        def _always_invalid(*a, **k):
            return (False, "invalid_scale", "Scale not supported", [])

        submit_args = ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "8"]

        # Without --force: validation rejects all tasks, exit code != 0
        with (
            patch('llmb_run.main.get_cluster_config', return_value=mock_cluster_config),
            patch('llmb_run.main.get_workloads', return_value=mock_workloads),
            patch('llmb_run.main.validate_workload_with_details', side_effect=_always_invalid),
            patch('llmb_run.main.run_tests'),
        ):
            result = runner.invoke(app, submit_args)
        assert result.exit_code != 0

        # With --force: same tasks go through
        recorded_tasks = []

        with (
            patch('llmb_run.main.get_cluster_config', return_value=mock_cluster_config),
            patch('llmb_run.main.get_workloads', return_value=mock_workloads),
            patch('llmb_run.main.validate_workload_with_details', side_effect=_always_invalid),
            patch('llmb_run.main.run_tests', lambda _cfg, tasks, _wl, **_kwargs: recorded_tasks.extend(tasks)),
        ):
            result = runner.invoke(app, submit_args + ["--force"])

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1

    def test_submit_force_keeps_requested_unsupported_scale(self, runner):
        """--force should preserve an explicitly requested scale through task generation."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "24", "--force"]
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].scale == 24
        assert recorded_tasks[0].dtype == "fp16"

    def test_submit_force_keeps_requested_unsupported_dtype(self, runner):
        """--force should preserve an explicitly requested dtype through task generation."""
        from llmb_run.main import app

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(
                app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp32", "--scale", "8", "--force"]
            )

        assert result.exit_code == 0, f"Output: {result.stdout}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].scale == 8
        assert recorded_tasks[0].dtype == "fp32"

    def test_submit_force_rejects_discovery_mode(self, runner, caplog):
        """--force should be rejected outside single explicit-task mode."""
        from llmb_run.main import app

        result = runner.invoke(app, ["submit", "-w", "training_foo_7b", "--max-scale", "16", "--force"])

        assert result.exit_code != 0
        assert "--force is only supported for a single explicit task" in caplog.text

    def test_submit_force_rejects_multi_value_scale(self, runner, caplog):
        """--force should reject fan-out via comma-separated scales."""
        from llmb_run.main import app

        result = runner.invoke(
            app, ["submit", "-w", "training_foo", "-s", "7b", "--dtype", "fp16", "--scale", "8,16", "--force"]
        )

        assert result.exit_code != 0
        assert "--force only supports a single explicit task" in caplog.text

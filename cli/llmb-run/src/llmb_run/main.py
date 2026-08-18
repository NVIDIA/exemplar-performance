# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import os
import pathlib
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Annotated, Any, Mapping, Optional

import typer
import yaml

from llmb_run.archive import run_archive
from llmb_run.config_manager import ClusterConfig, get_cluster_config
from llmb_run.env_args import parse_cli_env_args, parse_cli_mbridge_args
from llmb_run.exemplar import generate_exemplar_tasks
from llmb_run.job_history import (
    format_job_details,
    format_jobs_table,
    get_job,
    list_jobs,
    rebuild_history,
    refresh_non_terminal_jobs,
    refresh_requested_jobs,
)
from llmb_run.job_launcher import run_tests
from llmb_run.job_logs import active_job_log, find_configured_sbatch_logs, find_job_logs, follow_tail, read_tail
from llmb_run.metadata_utils import parse_workload_name
from llmb_run.slurm_args import build_cli_slurm_args, validate_no_additional_slurm_params_conflict
from llmb_run.task_generation import TaskGenerationRequest, ValidationError, generate_tasks
from llmb_run.tasks import (
    WorkloadTask,
    format_task_output,
)
from llmb_run.workload_validator import (
    format_validation_error,
    get_workloads,
    print_avail_workloads,
    validate_workload_with_details,
)


class LevelFormatter(logging.Formatter):
    """Custom formatter that changes format based on log level."""

    def __init__(self, fmt_dict):
        super().__init__()
        self.fmt_dict = fmt_dict

    def format(self, record):
        # Select the format based on the log level
        fmt = self.fmt_dict.get(record.levelno, self.fmt_dict[logging.INFO])
        formatter = logging.Formatter(fmt)
        return formatter.format(record)


# Define log formats for different levels.
formatters = {
    logging.DEBUG: "DEBUG: %(message)s",
    logging.INFO: "%(message)s",
    logging.ERROR: "ERROR: %(message)s",
    logging.CRITICAL: "CRITICAL: %(message)s",
}

logger = logging.getLogger('llmb_run')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(LevelFormatter(formatters))
logger.addHandler(console_handler)


# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 1
EXIT_SYSTEM_ERROR = 2


# Create the Typer app
app = typer.Typer(
    help='llmb-run: Tool for launching multiple or single LLM benchmarking workloads.',
    no_args_is_help=True,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
jobs_app = typer.Typer(
    help='View llmb-run job history and logs.',
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(jobs_app, name="jobs")


class AppContext:
    """Context object to share configuration across commands."""

    def __init__(self):
        self.cluster_config: ClusterConfig | None = None
        self.workloads = None
        self.verbose = False


def _get_recipe_version() -> Optional[tuple[str, str]]:
    """Return (recipe_version, abs_repo_path) if discoverable, else None.

    Lightweight: never raises, never logs. Uses the same cluster_config.yaml
    resolution as the launcher (CWD takes precedence over $LLMB_INSTALL).
    """
    try:
        candidates = [pathlib.Path.cwd() / 'cluster_config.yaml']
        if install := os.environ.get('LLMB_INSTALL'):
            candidates.append(pathlib.Path(install) / 'cluster_config.yaml')
        cfg_path = next((p for p in candidates if p.exists()), None)
        if cfg_path is None:
            return None
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        repo = cfg.get('llmb_repo') or (cfg.get('launcher') or {}).get('llmb_repo')
        if not repo:
            return None
        release_data = yaml.safe_load((pathlib.Path(repo) / 'release.yaml').read_text()) or {}
        if version := release_data.get('llmb_version'):
            return str(version), str(pathlib.Path(repo).resolve())
        return None
    except Exception:
        return None


def version_callback(value: bool):
    if value:
        try:
            version = package_version("llmb-run")
        except PackageNotFoundError:
            version = "unknown"
        typer.echo(f"llmb-run {version}")
        recipe = _get_recipe_version()
        if recipe is not None:
            recipe_version, repo_path = recipe
            typer.echo(f"Recipe Version {recipe_version} ({repo_path})")
        raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: Annotated[
        bool, typer.Option('-v', '--verbose', help='Enable verbose output including debug information.')
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option('--version', help='Show version and exit.', callback=version_callback, is_eager=True),
    ] = None,
):
    """
    Main callback that runs before any command.
    Loads cluster configuration and workloads once, and configures logging.
    """
    # Check for SLURM environment
    if 'SLURM_JOB_ID' in os.environ:
        logger.error(
            "🚫: `llmb-run` does not currently support running within a SLURM allocation. Please run this script directly from a login node outside of a SLURM job."
        )
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    # Set up logging
    if verbose:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

    # Create context object
    app_ctx = AppContext()
    app_ctx.verbose = verbose
    ctx.obj = app_ctx

    # Load configuration
    try:
        app_ctx.cluster_config = get_cluster_config()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e

    # Archive and most jobs history commands only require cluster config.
    # `jobs rebuild` loads workload metadata in its command handler.
    if ctx.invoked_subcommand in {'archive', 'jobs'}:
        return

    # Best-effort gsw-common image freshness check
    try:
        from llmb_run.internal.image_updater import check_gsw_common_update

        check_gsw_common_update(app_ctx.cluster_config)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"gsw-common update check skipped: {e}")

    # Load workloads
    try:
        app_ctx.workloads = get_workloads(app_ctx.cluster_config)
    except Exception as e:
        logger.error(f"Failed to load workloads: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e


def validate_bulk_tasks(task_list, workloads, cluster_config):
    """Validate all tasks in a bulk job and return validated tasks with error summary.

    Returns:
        tuple: (validated_tasks, validation_summary)
            where validation_summary is a dict with error counts and unique error types
    """
    cluster_gpu_type = cluster_config.gpu_type
    validated_tasks = []
    error_summary = {}

    for task in task_list:
        is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
            workloads,
            task.workload_key,
            task.model_size,
            dtype=task.dtype,
            scale=task.scale,
            cluster_gpu_type=cluster_gpu_type,
            cluster_config=cluster_config,
            proxy=task.proxy,
        )
        if is_valid:
            validated_tasks.append(task)
        else:
            # Group errors by type and message for cleaner reporting
            error_key = (error_type, error_msg, tuple(str(s) for s in suggestions))
            if error_key not in error_summary:
                error_summary[error_key] = {
                    'count': 0,
                    'error_type': error_type,
                    'error_msg': error_msg,
                    'suggestions': suggestions,
                    'example_task': task,
                }
            error_summary[error_key]['count'] += 1

    return validated_tasks, error_summary


def _handle_no_tasks_error(app_ctx: AppContext, request: TaskGenerationRequest):
    """Provide helpful error when no tasks are generated."""
    logger.error("No matching job configurations found.")

    # YAML file case: specific hint, skip list output
    if request.file_path and request.file_path.endswith(('.yaml', '.yml')):
        logger.error("For YAML files, ensure you have at least one entry under 'tasks:'")
        logger.error("  Example: tasks:")
        logger.error("             - dtypes: fp8")
        logger.error("               scales: [128, 256]")
    else:
        # Show what IS available - print_avail_workloads() handles the header
        logger.info("")  # Empty line for spacing

        # Build workload filter: either specific workloads or None for all
        workload_filter = None
        if request.workload:
            # Parse comma-separated workload list and strip model size suffixes.
            # Users can specify workloads as "pretrain_foo_7b" in discovery mode,
            # but print_avail_workloads expects base workload keys like "pretrain_foo".
            parsed = [parse_workload_name(w.strip())[0] for w in request.workload.split(',') if w.strip()]
            workload_filter = parsed if parsed else None

        # Call once with the filter (single or multiple workloads, or None for all)
        print_avail_workloads(
            app_ctx.workloads,
            app_ctx.cluster_config,
            cluster_gpu_type=app_ctx.cluster_config.gpu_type,
            verbose=True,
            workload_filter=workload_filter,
        )


def report_validation_results(validated_tasks, error_summary, task_list, mode_name="job"):
    """Report validation results in a consistent format across different modes.

    Args:
        validated_tasks: List of valid tasks
        error_summary: Dictionary of validation errors
        task_list: Original list of all tasks
        mode_name: Name of the mode for error messages (e.g., "submit", "exemplar")
    """
    if error_summary:
        total_errors = sum(err['count'] for err in error_summary.values())
        logger.error(f"Validation failed for {total_errors} out of {len(task_list)} tasks:")

        for error_info in error_summary.values():
            count = error_info['count']
            example_task = error_info['example_task']
            error_type = error_info['error_type']
            error_msg = error_info['error_msg']
            suggestions = error_info['suggestions']

            # Use existing format_validation_error for consistent formatting
            formatted_error = format_validation_error(error_type, error_msg, suggestions)

            # Add count prefix with example
            prefix = f"  ❌ {count}x {example_task.workload_key}_{example_task.model_size} (dtype={example_task.dtype})"

            # Split the formatted error and add prefix to first line, indent others
            error_lines = formatted_error.split('\n')
            logger.error(f"{prefix}: {error_lines[0]}")
            for line in error_lines[1:]:
                logger.error(f"     {line}")

        if not validated_tasks:
            logger.error(f"❌ No valid tasks found. Aborting {mode_name} submission.")
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)
        else:
            logger.warning(f"⚠️  Proceeding with {len(validated_tasks)} valid tasks out of {len(task_list)} total.")
    else:
        logger.debug(f"✅ All {len(task_list)} tasks validated successfully.")


def _ctx_app_context(ctx: typer.Context) -> AppContext:
    current = ctx
    while current is not None:
        if isinstance(current.obj, AppContext):
            return current.obj
        current = current.parent
    raise RuntimeError("Missing llmb-run application context.")


def _display_workloads(app_ctx: AppContext) -> dict:
    """Workload metadata for display labels, loaded once per command.

    Empty on failure so `jobs list`/`jobs show` keep working without a
    readable llmb_repo (labels degrade to the pretrain columns).
    """
    try:
        return get_workloads(app_ctx.cluster_config)
    except Exception as e:
        logger.debug(f"Workload metadata unavailable for display: {e}")
        return {}


def _jobs_list_impl(ctx: typer.Context) -> None:
    app_ctx = _ctx_app_context(ctx)
    _, refresh_error = refresh_non_terminal_jobs(app_ctx.cluster_config)
    rows = list_jobs(app_ctx.cluster_config)
    typer.echo(format_jobs_table(rows, _display_workloads(app_ctx)))
    if refresh_error:
        # Print after the table so users notice it even when the table scrolls.
        logger.warning(f"sacct unavailable; status may be stale ({refresh_error}).")


def _get_job_or_exit(app_ctx: AppContext, job_id: int):
    row = get_job(app_ctx.cluster_config, job_id)
    if row is None:
        logger.error(f"Job {job_id} was not found in llmb-run history.")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)
    return row


@jobs_app.callback()
def jobs_callback(ctx: typer.Context):
    """View llmb-run job history."""
    if ctx.invoked_subcommand is None:
        _jobs_list_impl(ctx)


@jobs_app.command(name="list")
def jobs_list(ctx: typer.Context):
    """List known jobs and refresh non-terminal Slurm states."""
    _jobs_list_impl(ctx)


@jobs_app.command(name="show")
def jobs_show(ctx: typer.Context, job_id: Annotated[int, typer.Argument(help='Slurm job ID to show.')]):
    """Show details for a single job, including its log directory."""
    app_ctx = _ctx_app_context(ctx)
    _, refresh_error = refresh_non_terminal_jobs(app_ctx.cluster_config)
    row = _get_job_or_exit(app_ctx, job_id)
    typer.echo(format_job_details(row, _display_workloads(app_ctx)))
    if refresh_error:
        logger.warning(f"sacct unavailable; status may be stale ({refresh_error}).")


@jobs_app.command(name="log")
def jobs_log(
    ctx: typer.Context,
    job_id: Annotated[int, typer.Argument(help='Slurm job ID to inspect.')],
    tail_lines: Annotated[int, typer.Option('--tail', min=1, help='Number of lines to show.')] = 200,
    follow: Annotated[
        bool, typer.Option('-f', '--follow', help='Follow the active log file after printing the initial tail.')
    ] = False,
    print_path: Annotated[bool, typer.Option('--path', help='Print the active log file path only.')] = False,
    print_dir: Annotated[bool, typer.Option('--dir', help='Print the job log directory only.')] = False,
    list_files: Annotated[bool, typer.Option('--list', help='List all matching retry log files for the job.')] = False,
):
    """Show or follow the active log for a single job."""
    app_ctx = _ctx_app_context(ctx)
    row = _get_job_or_exit(app_ctx, job_id)
    launcher_type = row["launcher_type"]
    if launcher_type == 'sbatch':
        logger.error(f"Job {job_id} uses legacy sbatch logging, which llmb-run cannot resolve reliably.")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    log_dir = row["log_dir"]
    if not log_dir:
        logger.error(f"Job {job_id} does not have a log directory recorded. Run `llmb-run jobs rebuild` to rescan.")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    selected_modes = sum(bool(value) for value in (print_path, print_dir, list_files))
    if selected_modes > 1:
        logger.error("Use only one of --path, --dir, or --list.")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)
    if follow and selected_modes:
        logger.error("--follow can only be used when printing log contents.")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    if print_dir:
        typer.echo(log_dir)
        return

    try:
        if launcher_type in {'nemo', 'megatron_bridge'}:
            logs = find_job_logs(log_dir, job_id)
        elif launcher_type == 'configured_sbatch':
            logs = find_configured_sbatch_logs(log_dir, job_id)
        else:
            logger.error(f"Job {job_id} has unsupported launcher type '{launcher_type}'.")
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e

    active_log = active_job_log(logs)
    if list_files:
        if not logs:
            logger.info(f"No log files found for job {job_id} in {log_dir}.")
            return
        for log_file in logs:
            suffix = " (active)" if log_file == active_log else ""
            label = str(log_file.retry) if log_file.retry is not None else "slurm"
            typer.echo(f"{label}: {log_file.path}{suffix}")
        return

    if active_log is None:
        logger.error(f"No log file found for job {job_id} in {log_dir}.")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    if print_path:
        typer.echo(active_log.path)
        return

    try:
        if follow:
            raise typer.Exit(code=follow_tail(active_log.path, tail_lines))
        tail_output = read_tail(active_log.path, tail_lines)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e
    except OSError as e:
        logger.error(f"Unable to read log file {active_log.path}: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e

    if tail_output:
        typer.echo(tail_output)


@jobs_app.command(name="refresh")
def jobs_refresh(
    ctx: typer.Context,
    job_ids: Annotated[list[int], typer.Argument(help='Slurm job ID(s) to force refresh.')],
):
    """Force-refresh one or more job records."""
    app_ctx = _ctx_app_context(ctx)
    requested_job_ids = sorted({int(job_id) for job_id in job_ids})
    missing_job_ids = [job_id for job_id in requested_job_ids if get_job(app_ctx.cluster_config, job_id) is None]
    if missing_job_ids:
        ids = ", ".join(str(job_id) for job_id in missing_job_ids)
        logger.error(f"Job ID(s) not found in llmb-run history: {ids}.")
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)

    refreshed, refresh_error = refresh_requested_jobs(app_ctx.cluster_config, requested_job_ids)
    if refresh_error:
        logger.error(f"sacct unavailable: {refresh_error}.")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR)
    logger.info(f"Refreshed {refreshed} job status records.")


@jobs_app.command(name="rebuild")
def jobs_rebuild(ctx: typer.Context):
    """Rebuild job history by scanning llmb-config files under $LLMB_INSTALL."""
    app_ctx = _ctx_app_context(ctx)
    try:
        workloads = get_workloads(app_ctx.cluster_config)
    except Exception as e:
        logger.error(f"Failed to load workloads: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e

    stats = rebuild_history(app_ctx.cluster_config, workloads)
    logger.info(f"Job history database: {stats.db_path}")
    logger.info(f"Scanned {stats.scanned} llmb-config files; imported {stats.imported}, skipped {stats.skipped}.")
    if stats.refresh_error:
        logger.warning(f"sacct unavailable; refreshed statuses may be stale ({stats.refresh_error}).")


def _warn_and_clear_unsupported_mbridge_args(tasks: list[WorkloadTask], workloads: Mapping[str, Any]) -> None:
    """Warn once and clear MBridge-only arguments from incompatible tasks."""
    ignored_workloads: set[str] = set()

    for task in tasks:
        workload = workloads.get(task.workload_key, {})
        launcher_type = workload.get('metadata', {}).get('run', {}).get('launcher_type')
        if launcher_type != 'megatron_bridge' and task.mbridge_args:
            ignored_workloads.add(task.workload_key)
            task.mbridge_args = ()

    if ignored_workloads:
        names = ', '.join(sorted(ignored_workloads))
        logger.warning(f"Ignoring --mbridge-arg for non-Megatron-Bridge workloads: {names}")


def _submit_impl(ctx: typer.Context, request: TaskGenerationRequest, dryrun: bool, mode_name: str = "submit"):
    """Shared implementation for all submission commands."""
    app_ctx: AppContext = ctx.obj

    try:
        # Early check: fail fast if CLI slurm flags conflict with env/cluster-level
        # ADDITIONAL_SLURM_PARAMS. Per-workload/task checks remain in the launcher.
        if request.slurm_args:
            validate_no_additional_slurm_params_conflict(
                cli_args=request.slurm_args,
                cluster_environment=app_ctx.cluster_config.environment,
            )

        # Generate tasks
        task_list = generate_tasks(request)

        # Sort tasks by scale in descending order (largest first)
        task_list.sort(key=lambda task: task.scale, reverse=True)

        if not task_list:
            _handle_no_tasks_error(app_ctx, request)
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)

        if request.force:
            # _generate_forced_explicit_task already skipped dtype/scale filtering
            # during generation; we must also skip validate_bulk_tasks here because
            # it would re-reject the same dtype/scale combinations.
            logger.warning("⚠️  --force: bypassing dtype/scale validation.")
            validated_tasks = task_list
        else:
            # Validate tasks
            validated_tasks, error_summary = validate_bulk_tasks(task_list, app_ctx.workloads, app_ctx.cluster_config)

            # Report results
            report_validation_results(validated_tasks, error_summary, task_list, mode_name)

        _warn_and_clear_unsupported_mbridge_args(validated_tasks, app_ctx.workloads)

        if request.slurm_args:
            for task in validated_tasks:
                workload_environment = app_ctx.cluster_config.workload_config(task.workload_key).get("environment", {})
                validate_no_additional_slurm_params_conflict(
                    cli_args=request.slurm_args,
                    cluster_environment=app_ctx.cluster_config.environment,
                    workload_environment=workload_environment,
                    task_environment=task.env_overrides,
                )

        # Print the concrete jobs we’re about to submit (kept concise; launcher output follows).
        logger.info(f"Jobs ({len(validated_tasks)}):")
        for task in validated_tasks:
            logger.info(format_task_output(task, prefix="  - "))

        if dryrun:
            logger.info("Dry run enabled. Jobs will not be submitted.")
        else:
            run_tests(app_ctx.cluster_config, validated_tasks, app_ctx.workloads)

    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Submission error: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e


@app.command()
def submit(
    ctx: typer.Context,
    workload: Annotated[
        Optional[str],
        typer.Option('-w', '--workload', help='Workload name (single or comma-separated for discovery).'),
    ] = None,
    model_size: Annotated[
        Optional[str],
        typer.Option(
            '-s',
            '--model-size',
            help='Size of the model (e.g., 7b, 13b, 1t). Requires explicit single workload via -w.',
        ),
    ] = None,
    dtype: Annotated[
        Optional[str],
        typer.Option('-d', '--dtype', help='Data type (e.g., fp16, bf16). Comma-separated list allowed.'),
    ] = None,
    scale: Annotated[
        Optional[str],
        typer.Option(
            '--scale',
            help='Scale parameter (number of GPUs). Comma-separated list allowed (e.g., "8,16"). Mutually exclusive with --max-scale.',
        ),
    ] = None,
    max_scale: Annotated[
        Optional[int], typer.Option('--max-scale', help='Maximum scale to test up to (discovery/mixed mode).')
    ] = None,
    min_scale: Annotated[
        bool,
        typer.Option('--min-scale', help='Only run the minimum supported scale (discovery/mixed mode).'),
    ] = False,
    exact_scales: Annotated[
        bool,
        typer.Option(
            '--exact-scales', help='Only use scales from metadata (no power-of-2 expansion beyond metadata max).'
        ),
    ] = False,
    file_path: Annotated[
        Optional[str], typer.Option('-f', '--file', help='Path to workload specification file (.txt or .yaml).')
    ] = None,
    repeats: Annotated[int, typer.Option('-r', '--repeats', help='Number of repeats for each test configuration.')] = 1,
    profile: Annotated[bool, typer.Option('-p', '--profile', help='Enable Profiling for jobs.')] = False,
    dump_env: Annotated[
        bool,
        typer.Option(
            '--dump-env',
            help='Write a redacted rank-0 environment snapshot for Megatron-Bridge workloads. Ignored for other workloads.',
        ),
    ] = False,
    mbridge_arg_values: Annotated[
        Optional[list[str]],
        typer.Option(
            '--mbridge-arg',
            help='Repeatable raw Megatron-Bridge argument (`--flag=value` or Hydra override). Applied after recipe overrides.',
            rich_help_panel='Megatron-Bridge',
        ),
    ] = None,
    proxy: Annotated[bool, typer.Option('--proxy', help='Use proxy scales.')] = False,
    dryrun: Annotated[
        bool,
        typer.Option('--dry-run', help='List jobs without submitting them.'),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            '--force',
            help='Bypass dtype/scale validation for one explicit task. Use with caution.',
        ),
    ] = False,
    nice: Annotated[
        Optional[int],
        typer.Option('--nice', help='Lower the job priority via Slurm nice.', rich_help_panel='Slurm'),
    ] = None,
    nodelist: Annotated[
        Optional[str],
        typer.Option('--nodelist', help='Restrict the job to a specific node list.', rich_help_panel='Slurm'),
    ] = None,
    exclude: Annotated[
        Optional[str], typer.Option('--exclude', help='Exclude specific nodes from the job.', rich_help_panel='Slurm')
    ] = None,
    reservation: Annotated[
        Optional[str],
        typer.Option('--reservation', help='Submit the job under a Slurm reservation.', rich_help_panel='Slurm'),
    ] = None,
    segment: Annotated[
        Optional[int],
        typer.Option('--segment', help='Set the Slurm segment size for the job.', rich_help_panel='Slurm'),
    ] = None,
    slurm_arg_values: Annotated[
        Optional[list[str]],
        typer.Option(
            '--slurm-arg',
            help='Repeatable raw Slurm parameter in `key=value` or bare-flag form.',
            rich_help_panel='Slurm',
        ),
    ] = None,
    env_values: Annotated[
        Optional[list[str]],
        typer.Option(
            '--env',
            help='Repeatable environment variable override in `KEY=value` form.',
            rich_help_panel='Slurm',
        ),
    ] = None,
):
    """
    Submit jobs using a unified interface. Supports explicit, discovery, and file-based modes.
    """
    app_ctx: AppContext = ctx.obj

    try:
        explicit_env_overrides = parse_cli_env_args(env_values)
        mbridge_args = parse_cli_mbridge_args(mbridge_arg_values)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e

    try:
        slurm_args = build_cli_slurm_args(
            nodelist=nodelist,
            exclude=exclude,
            reservation=reservation,
            segment=segment,
            nice=nice,
            slurm_args=slurm_arg_values,
        )
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e

    request = TaskGenerationRequest(
        workloads=app_ctx.workloads,
        cluster_config=app_ctx.cluster_config,
        workload=workload,
        model_size=model_size,
        dtype=dtype,
        scale=scale,
        max_scale=max_scale,
        min_scale=min_scale,
        exact_scales=exact_scales,
        file_path=file_path,
        repeats=repeats,
        profile=profile,
        proxy=proxy,
        force=force,
        slurm_args=slurm_args,
        explicit_env_overrides=explicit_env_overrides,
        extra_workload_args=("--dump_env",) if dump_env else (),
        mbridge_args=mbridge_args,
    )

    _submit_impl(ctx, request, dryrun, mode_name="submit")


@app.command(name="list")
def list_workloads(
    ctx: typer.Context,
    workload: Annotated[
        Optional[str], typer.Option('-w', '--workload', help='Show detailed information for a specific workload.')
    ] = None,
):
    """
    List available workloads and their configurations.
    """
    app_ctx: AppContext = ctx.obj
    cluster_gpu_type = app_ctx.cluster_config.gpu_type

    # Always use print_avail_workloads, with workload_filter if specified
    result = print_avail_workloads(
        app_ctx.workloads,
        app_ctx.cluster_config,
        cluster_gpu_type=cluster_gpu_type,
        verbose=True,
        workload_filter=workload,
    )

    if workload and not result:
        raise typer.Exit(code=EXIT_VALIDATION_ERROR)


@app.command()
def exemplar(
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option('--dry-run', help='Print jobs without submitting them.'),
    ] = False,
    repeats: Annotated[
        Optional[int],
        typer.Option(
            '-r',
            '--repeats',
            min=1,
            help='Number of repeats for each test configuration. If not provided, uses exemplar.yaml config.repeats (fallback: 1).',
        ),
    ] = None,
):
    """
    Submit exemplar certification jobs from $LLMB_INSTALL/llmb_repo/exemplar.yaml.

    Runs workloads listed in exemplar.yaml for your cluster's GPU type.
    All workloads must be installed.

    Fallbacks when omitted from exemplar.yaml: scale=512, profile=false, repeats=1 (override repeats with -r).
    If profile=true, the last repeat is profiled and earlier repeats are non-profiled.
    """
    app_ctx: AppContext = ctx.obj

    try:
        # Generate exemplar tasks (includes preflight checks and strict install gating)
        # Pass CLI repeats (None if not provided); generate_exemplar_tasks will use YAML value if None
        task_list = generate_exemplar_tasks(app_ctx.workloads, app_ctx.cluster_config, repeats=repeats)

        if not task_list:
            logger.error("No exemplar tasks generated. Check your configuration and installed workloads.")
            raise typer.Exit(code=EXIT_VALIDATION_ERROR)

        # Validate tasks
        validated_tasks, error_summary = validate_bulk_tasks(task_list, app_ctx.workloads, app_ctx.cluster_config)

        # Report results
        report_validation_results(validated_tasks, error_summary, task_list, mode_name="exemplar")

        # Print the concrete jobs we're about to submit
        logger.info(f"Exemplar Certification Jobs ({len(validated_tasks)}):")
        profiled_count = sum(1 for task in validated_tasks if task.profile)
        normal_count = len(validated_tasks) - profiled_count
        logger.info(f"Profiling plan: {normal_count} standard jobs, {profiled_count} profiling-enabled jobs.")
        for task in validated_tasks:
            logger.info(format_task_output(task, prefix="  - "))

        if dry_run:
            logger.info("Dry run enabled. Jobs will not be submitted.")
        else:
            run_tests(app_ctx.cluster_config, validated_tasks, app_ctx.workloads)

    except ValidationError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Exemplar submission error: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e


@app.command()
def archive(
    ctx: typer.Context,
    output: Annotated[
        Optional[str],
        typer.Option(
            '--output',
            help='Path to output .tar.zst file. Default filename under $LLMB_INSTALL: llmb-archive_<release_version>_<timestamp>.tar.zst.',
        ),
    ] = None,
):
    """Archive workload experiment logs under $LLMB_INSTALL/workloads/*/experiments into a single .tar.zst file."""
    app_ctx: AppContext = ctx.obj

    try:
        stats = run_archive(app_ctx.cluster_config, output)
        logger.info(f"Archive created: {stats.output_path}")
        logger.info(f"Archived {stats.experiment_count} experiments.")
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=EXIT_VALIDATION_ERROR) from e
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Archive failed: {e}")
        raise typer.Exit(code=EXIT_SYSTEM_ERROR) from e


def cli():
    """Main entry point for the llmb-run CLI."""
    app()


if __name__ == '__main__':
    cli()

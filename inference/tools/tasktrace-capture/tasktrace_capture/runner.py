# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Harbor experiment runner.

Launches Harbor benchmarks via subprocess, monitors CPU in a background
thread, and aggregates the results.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .inference_metrics import (
    compute_inference_stats,
    metrics_url_from_api_base,
    sample_inference,
    snapshot_histograms,
)
from .live_cpu_profile import (
    LIVE_CPU_COLLECTOR_IMAGE_ENV,
    LIVE_CPU_CPUSET_ENV,
    LIVE_CPU_ENABLE_ENV,
    LIVE_CPU_SCHEDULER_ENV,
    LIVE_CPU_WINDOWS_ENV,
    PROFILE_FILENAME,
    SUPPORTED_WINDOWS,
    WINDOW_AGENT_EXECUTION,
    LiveCpuProfileError,
    validate_profile_cpuset_cpus,
)
from .models import TimeSplit, TrialSplit
from .monitor import sample_cpu
from .shell_capture import (
    IMAGE_BASH_ENV_ENV,
    IMAGE_BASHRC,
    TRACE_ENABLE_ENV,
    finalize_trajectories,
    supports_shell_capture_agent,
)


class _PgidHolder:
    """Thread-safe holder for the active harbor process group ID."""

    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def __setattr__(self, name, val):
        if name == "value":
            if hasattr(self, "_lock"):
                with self._lock:
                    super().__setattr__(name, val)
            else:
                super().__setattr__(name, val)
        else:
            super().__setattr__(name, val)


# Shared state: the PGID of the active harbor subprocess.
# Used by the CLI signal handler to kill the entire process tree.
_active_harbor_pgid = _PgidHolder()


def _kill_harbor_group(proc):
    """Kill the harbor process group (harbor + all children including enroot agents)."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        # Give processes 5s to exit gracefully, then force-kill
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.killpg(pgid, 0)  # check if still alive
            except ProcessLookupError:
                return  # all dead
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # already dead


def get_active_harbor_pgid():
    """Return the PGID of the running harbor subprocess, or 0 if none."""
    return _active_harbor_pgid.value


def _cleanup_harbor_containers():
    """Kill processes inside hb__* containers, remove them, then sweep orphans.

    Two phases:
    1. Kill inside + remove: enroot exec enters each hb__* container's namespace
       and kill -9 -1 hits only processes in that namespace.  Then remove.
    2. Orphan sweep: after removal, any processes that escaped (e.g. forked
       between kill and remove, or survived in D-state) are in a non-protected
       namespace.  kill_orphaned_container_procs() kills them by comparing
       against the protected set of live container namespaces.

    Phase 2 MUST run after phase 1's remove — before removal, the hb__*
    container's namespace would be in the protected set and orphans would
    be missed.
    """
    # Phase 1: remove hb__* containers.  Don't try to kill inside them first —
    # enroot has no PID namespaces, so "kill -9 -1" inside enroot exec would
    # kill ALL our processes, not just the container's.
    try:
        r = subprocess.run(["enroot", "list"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return
        for name in r.stdout.strip().split("\n"):
            name = name.strip()
            if name and name.startswith("hb__"):
                subprocess.run(["enroot", "remove", "-f", name], capture_output=True, timeout=10)
    except Exception:
        pass

    # Phase 2: kill orphaned processes from removed containers using the
    # protected namespace approach.  Now that hb__* containers are removed,
    # their processes' namespaces are no longer protected.
    #
    # Run twice with a delay: the first sweep catches most orphans, but
    # processes spawned in the final seconds before timeout may still be
    # reparenting (PPID changing from dead wrapper to 1).  The second
    # sweep catches stragglers.  Validated: single sweep misses ~70% of
    # attack_parallel.py workers; double sweep catches 100%.
    from .monitor import kill_orphaned_container_procs

    for _ in range(2):
        try:
            kill_orphaned_container_procs()
        except Exception:
            pass
        time.sleep(2)


def _compute_trials(trials, n_concurrent, *, fill_slots=True):
    """Compute Harbor ``-k`` attempts per selected task.

    A single selected task must be repeated to fill concurrent slots. For a
    dataset or suite, Harbor fills slots from the selected task set, so
    inflating ``-k`` would multiply every task unexpectedly.
    """
    if not fill_slots:
        return trials or 1
    return max(trials or n_concurrent, n_concurrent)


def normalize_registry_task_selection(task: str | None):
    """Return the canonical dataset/task selection for a legacy registry task.

    Existing ``--task DATASET/TASK`` inputs remain supported, but registry
    execution uses the same dataset plus inclusion-list representation as
    explicit ``--include-task`` inputs. Existing filesystem paths stay on the
    local-task path.
    """
    if not task or "," in task or os.path.exists(task) or "/" not in task:
        return None
    dataset, task_name = task.split("/", 1)
    if not dataset or not task_name:
        return None
    return {"dataset": dataset, "tasks": [task_name]}


def _agent_execution_window(jobs_dir):
    """(earliest agent start, latest agent finish) from the latest Harbor job's trial result.json files,
    or None when Harbor recorded no agent start/finish times."""
    if not jobs_dir or not jobs_dir.exists():
        return None

    # Find the harbor job dir (most recent subdir of jobs_dir)
    job_dirs = sorted(
        (d for d in jobs_dir.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    if not job_dirs:
        return None

    earliest_start = None
    latest_finish = None
    for result_file in job_dirs[0].rglob("result.json"):
        try:
            with open(result_file) as f:
                r = json.load(f)
            agent_exec = r.get("agent_execution") or {}
            sa = agent_exec.get("started_at")
            fa = agent_exec.get("finished_at")
            if not sa or not fa:
                continue
            start = _parse_iso(sa)
            finish = _parse_iso(fa)
            if earliest_start is None or start < earliest_start:
                earliest_start = start
            if latest_finish is None or finish > latest_finish:
                latest_finish = finish
        except Exception:
            continue

    if earliest_start is None or latest_finish is None:
        return None
    return earliest_start, latest_finish


def _filter_by_trial_times(samples, jobs_dir):
    """Filter samples to only those during actual agent execution.

    Reads started_at/finished_at from trial result.json files to determine
    the time window when agents were running. Returns samples whose timestamp
    falls within [earliest_start, latest_finish] relative to sample t=0.
    """
    window = _agent_execution_window(jobs_dir)
    if window is None:
        return []
    earliest_start, latest_finish = window

    # Convert absolute timestamps to relative offsets matching sample "t" values.
    # Samples use t=0 as the start of monitoring (before harbor even launches).
    # We need to find what sample t values correspond to the agent execution window.
    # The first sample's wall time = monitoring start. Agent timestamps are absolute.
    # We can't directly map — but we know the total monitoring duration and the
    # agent execution happened somewhere in that window.
    #
    # Approach: agent wall time is (latest_finish - earliest_start).
    # Find the contiguous window of samples that best matches this duration
    # by looking for the period with highest CPU activity.
    agent_duration = (latest_finish - earliest_start).total_seconds()
    if agent_duration <= 0 or not samples:
        return []

    total_duration = samples[-1]["t"]
    if agent_duration >= total_duration * 0.9:
        return []  # agent ran nearly the whole time, no useful filtering

    # Use a sliding window approach: find the window of agent_duration seconds
    # with the highest total CPU usage — that's when the agent was active.
    best_start_idx = 0
    best_cpu_sum = 0
    current_cpu_sum = 0
    start_idx = 0

    for end_idx, s in enumerate(samples):
        current_cpu_sum += s["cpu_total"]
        while samples[end_idx]["t"] - samples[start_idx]["t"] > agent_duration * 1.1:
            current_cpu_sum -= samples[start_idx]["cpu_total"]
            start_idx += 1
        if current_cpu_sum > best_cpu_sum:
            best_cpu_sum = current_cpu_sum
            best_start_idx = start_idx

    # Collect samples in the best window
    window_start_t = samples[best_start_idx]["t"]
    window_end_t = window_start_t + agent_duration * 1.1
    active = [s for s in samples if window_start_t <= s["t"] <= window_end_t]

    if len(active) < 3:
        return []  # too few samples, not useful

    return active


def run_experiment(
    *,
    dataset=None,
    task=None,
    include_tasks=None,
    n_tasks=None,
    model,
    api_base,
    model_info,
    n_concurrent,
    trials=None,
    agent_timeout=3600,
    extra_agent_kwargs=None,
    extra_agent_env=None,
    api_keys=None,
    swebench_image_prefix: str | None = None,
    swebench_sif_template: str | None = None,
    swebench_images=None,
    jobs_dir: Path,
    agent="terminus-2",
    runtime: str = "enroot",
    enroot_network_policy: str = "strict",
    instrument_shell_commands: bool = False,
    instrument_shell_perfetto_depth: str = "1",
    source_image_bashrc: bool = False,
    trajectory_timings: str = "keep",
    profile_live_cpu: bool = False,
    profile_windows: list[str] | None = None,
    profile_scheduler: str = "off",
    profile_scheduler_collector_image: str | None = None,
    profile_cpuset_cpus: str | None = None,
    override_cpus: int | None = None,
    sync_start: bool = True,
):
    """Run Harbor with n_concurrent agents and measure host CPU.

    Args:
        dataset: Registry dataset name (e.g. "hello-world")
        task: Single task -- local path or registry ref ("terminal-bench/compile-compcert")
        include_tasks: Canonical registry subset as a dict with "dataset" and
            "tasks" (one or more task names).
                       Generates: -d dataset -i task1 -i task2 ...
        n_tasks: Dataset mode -- maximum selected tasks (Harbor ``-l`` flag).
        n_concurrent: Total number of concurrent agents
        trials: Trials per task (-k flag)
        jobs_dir: Path to the jobs output directory
        sync_start: Synchronize the first agent command across concurrent trials.

    Returns (result_dict, harbor_stdout).
    """
    if instrument_shell_commands and not supports_shell_capture_agent(agent):
        raise ValueError(
            "--instrument-shell-commands currently supports terminus, " "claude-code, and mini-swe-agent agents"
        )
    if instrument_shell_perfetto_depth not in {"1", "max"}:
        raise ValueError("--instrument-shell-perfetto-depth must be one of: 1, max")
    if trajectory_timings not in {"keep", "omit"}:
        raise ValueError("--trajectory-timings must be one of: keep, omit")
    selected_profile_windows = profile_windows or [WINDOW_AGENT_EXECUTION]
    invalid_profile_windows = set(selected_profile_windows) - SUPPORTED_WINDOWS
    if invalid_profile_windows:
        raise ValueError("unsupported --profile-window value(s): " + ", ".join(sorted(invalid_profile_windows)))
    if profile_scheduler not in {"off", "required"}:
        raise ValueError("--profile-scheduler must be one of: off, required")
    if (profile_windows or profile_scheduler != "off" or profile_cpuset_cpus) and not profile_live_cpu:
        raise ValueError(
            "--profile-window, --profile-scheduler, and --profile-cpuset-cpus " "require --profile-live-cpu"
        )
    if profile_live_cpu and runtime != "docker":
        raise ValueError("--profile-live-cpu currently supports --runtime docker only")
    if runtime == "current" and n_concurrent != 1:
        raise ValueError("--runtime current runs trials in this container one at a time; use --n-concurrent 1")
    if enroot_network_policy not in {"strict", "best-effort"}:
        raise ValueError("--enroot-network-policy must be one of: strict, best-effort")
    if enroot_network_policy != "strict" and runtime != "enroot":
        raise ValueError("--enroot-network-policy best-effort requires --runtime enroot")
    if override_cpus is not None and override_cpus <= 0:
        raise ValueError("--override-cpus must be positive")
    if profile_cpuset_cpus is not None and not isinstance(profile_cpuset_cpus, str):
        raise ValueError("--profile-cpuset-cpus must be a Docker cpuset string")
    normalized_single_task = False
    if include_tasks is None:
        normalized = normalize_registry_task_selection(task)
        if normalized is not None:
            include_tasks = normalized
            task = None
            normalized_single_task = True
    try:
        profile_cpuset_cpus = validate_profile_cpuset_cpus(
            profile_scheduler=profile_scheduler,
            requested_cpuset=profile_cpuset_cpus,
            override_cpus=override_cpus,
        )
    except LiveCpuProfileError as exc:
        raise ValueError(str(exc)) from exc
    if profile_scheduler == "required":
        # Exact peak collection is a controlled experiment: one measured
        # cgroup occupies one explicitly selected CPU allocation at a time.
        if n_concurrent != 1:
            raise ValueError("--profile-scheduler required runs one live trial at a time; " "set concurrency to 1")
    k = _compute_trials(
        trials,
        n_concurrent,
        fill_slots=(task is not None or normalized_single_task),
    )

    if include_tasks:
        n_tasks = len(include_tasks["tasks"])
        bench_name = f"{include_tasks['dataset']} ({n_tasks} tasks)"
    else:
        bench_name = task or dataset

    print(f"\n{'=' * 60}")
    print(f"  Running '{bench_name}': {n_concurrent} concurrent, {k} attempt(s)/task")
    print(f"{'=' * 60}")

    stop = threading.Event()
    sampler_result = []
    inference_result = []

    def sampler():
        sampler_result.extend(sample_cpu(stop, runtime=runtime))

    vllm_url = metrics_url_from_api_base(api_base)
    hist_before = snapshot_histograms(vllm_url)

    def inference_sampler():
        inference_result.extend(sample_inference(stop, vllm_url))

    t = threading.Thread(target=sampler, daemon=True)
    t_inf = threading.Thread(target=inference_sampler, daemon=True)
    t.start()
    t_inf.start()

    cmd = ["harbor", "run", "-o", str(jobs_dir)]
    if include_tasks:
        cmd.extend(["-d", include_tasks["dataset"]])
        for task_name in include_tasks["tasks"]:
            cmd.extend(["-i", task_name])
    elif task:
        if "/" in task and not os.path.exists(task):
            parts = task.split("/", 1)
            cmd.extend(["-d", parts[0], "-i", parts[1]])
        else:
            cmd.extend(["-p", task])
    else:
        cmd.extend(["-d", dataset])
        if n_tasks is not None:
            cmd.extend(["-l", str(n_tasks)])
    from tasktrace_capture.runtimes import runtime_import_path

    cmd.extend(
        [
            "-m",
            model,
            "-a",
            agent,
            "--environment-import-path",
            runtime_import_path(runtime),
            "-k",
            str(k),
            "--n-concurrent",
            str(n_concurrent),
        ]
    )
    if override_cpus is not None:
        cmd.extend(["--override-cpus", str(override_cpus)])
    if agent.startswith("terminus"):
        cmd.extend(
            [
                "--agent-kwarg",
                f"api_base={api_base}",
                "--agent-kwarg",
                f"model_info={json.dumps(model_info)}",
                "--agent-kwarg",
                "temperature=1.0",
                "--agent-kwarg",
                "top_p=1.0",
                "--agent-kwarg",
                "max_tokens=32000",
            ]
        )

    for kwarg in extra_agent_kwargs or []:
        cmd.extend(["--agent-kwarg", kwarg])

    # TaskTrace Capture's --agent-env mechanism: write vars into Harbor subprocess env,
    # listed in _TASKTRACE_CAPTURE_FORWARD_ENVS so the environment plugin forwards them
    # into the container. Applies to all supported runtimes.
    run_env = os.environ.copy()
    if api_base:
        # OpenSandbox uses this canonical value to admit the resolved model
        # endpoint through a restricted task's egress policy. It remains a
        # coordinator variable unless the user explicitly forwards it.
        run_env["AGENT_API_BASE"] = api_base
    python_bin_dir = Path(sys.executable).parent
    if (python_bin_dir / "harbor").is_file():
        current_path = run_env.get("PATH", "")
        path_parts = current_path.split(os.pathsep) if current_path else []
        if not path_parts or path_parts[0] != str(python_bin_dir):
            run_env["PATH"] = os.pathsep.join([str(python_bin_dir), *path_parts])

    # Default harbor's sqsh cache to working directory instead of ~/.cache/
    # which is often a tiny NFS /home on HPC clusters.
    # For ENROOT_CACHE_PATH and ENROOT_DATA_PATH, users should export them
    # manually if /home is too small (see README).
    # Docker doesn't use a sqsh cache, so only set this for enroot.
    if runtime == "enroot" and "HARBOR_ENROOT_CACHE" not in run_env:
        _cache = os.path.join(os.getcwd(), ".enroot", "cache")
        os.makedirs(_cache, exist_ok=True)
        run_env["HARBOR_ENROOT_CACHE"] = _cache
    if runtime == "enroot":
        run_env["_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY"] = enroot_network_policy

    forward_keys = []
    for env_var in extra_agent_env or []:
        key, _, value = env_var.partition("=")
        if key:
            run_env[key] = value
            forward_keys.append(key)
    if forward_keys:
        run_env["_TASKTRACE_CAPTURE_FORWARD_ENVS"] = ",".join(forward_keys)

    # Round-robin API keys across containers when multiple keys are provided.
    if api_keys and len(api_keys) > 1:
        run_env["_TASKTRACE_CAPTURE_API_KEYS"] = ",".join(api_keys)
    elif api_keys and len(api_keys) == 1:
        # Single --api-key: just set ANTHROPIC_API_KEY directly
        run_env["ANTHROPIC_API_KEY"] = api_keys[0]
        if "ANTHROPIC_API_KEY" not in forward_keys:
            forward_keys.append("ANTHROPIC_API_KEY")
            run_env["_TASKTRACE_CAPTURE_FORWARD_ENVS"] = ",".join(forward_keys)

    # Tell the environment plugin which agent is being used so it can apply
    # agent-specific optimizations (e.g. pre-installing claude-code binary).
    run_env["_TASKTRACE_CAPTURE_AGENT"] = agent
    if swebench_image_prefix:
        run_env["_TASKTRACE_CAPTURE_SWEBENCH_IMAGE_PREFIX"] = swebench_image_prefix
    if swebench_sif_template:
        run_env["_TASKTRACE_CAPTURE_SWEBENCH_SIF_TEMPLATE"] = swebench_sif_template
    if sync_start:
        run_env["_TASKTRACE_CAPTURE_BARRIER_COUNT"] = str(n_concurrent)
    if instrument_shell_commands:
        run_env[TRACE_ENABLE_ENV] = "1"
    if source_image_bashrc:
        run_env[IMAGE_BASH_ENV_ENV] = IMAGE_BASHRC
    if profile_live_cpu:
        run_env[LIVE_CPU_ENABLE_ENV] = "1"
        run_env[LIVE_CPU_WINDOWS_ENV] = ",".join(selected_profile_windows)
        run_env[LIVE_CPU_SCHEDULER_ENV] = profile_scheduler
        if profile_scheduler_collector_image:
            run_env[LIVE_CPU_COLLECTOR_IMAGE_ENV] = profile_scheduler_collector_image
        if profile_cpuset_cpus:
            run_env[LIVE_CPU_CPUSET_ENV] = profile_cpuset_cpus

    # Forward the pinned claude-code version to the selected plugin's cache.
    # Empty/unset means use the shared default.
    for kw in extra_agent_kwargs or []:
        if kw.startswith("version="):
            run_env["_TASKTRACE_CAPTURE_CLAUDE_CODE_VERSION"] = kw.split("=", 1)[1]
            break
    # SWE-bench image remapping: use epoch-research GHCR images for arm64 support.
    if swebench_images:
        run_env["_TASKTRACE_CAPTURE_SWEBENCH_IMAGES"] = swebench_images

    print(f"  CMD: {' '.join(cmd)}")
    if extra_agent_env:
        print(f"  ENV: {', '.join(k.partition('=')[0] for k in extra_agent_env)}")
    harbor_timeout = agent_timeout + 600  # agent budget + 10 min for container setup/teardown/verifier
    harbor_stdout = ""
    harbor_stderr = ""
    harbor_rc = None
    pre_job = find_latest_harbor_job(jobs_dir)

    # Launch harbor in its own process group so we can kill the entire tree
    # (harbor + enroot containers + agent processes) on timeout or signal.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
        start_new_session=True,
    )
    # Store the process group ID so the CLI signal handler can clean up.
    _active_harbor_pgid.value = os.getpgid(proc.pid)
    try:
        harbor_stdout, harbor_stderr = proc.communicate(timeout=harbor_timeout)
        harbor_rc = proc.returncode
        if harbor_rc != 0:
            print(f"  WARNING: harbor exited with code {harbor_rc}")
            if harbor_stderr:
                print(f"  STDERR (last 500 chars): ...{harbor_stderr[-500:]}")
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Timed out after {harbor_timeout}s — killing process group")
        _kill_harbor_group(proc)
        harbor_stdout, harbor_stderr = proc.communicate(timeout=10)
        harbor_rc = proc.returncode
    finally:
        _active_harbor_pgid.value = 0

    time.sleep(3)
    hist_after = snapshot_histograms(vllm_url)
    stop.set()
    t.join(timeout=5)
    t_inf.join(timeout=5)
    sys.stderr.write("\n")

    # Kill leftover harbor containers — harbor can exit while agents are still
    # running if they exceed the task's internal timeout.  Without this, zombie
    # containers burn CPU and contaminate the next concurrency level's metrics.
    # Docker: Harbor's Trial.run() finally-block handles clean exit via
    # `docker compose down`; SIGKILL-survival cleanup is handled by
    # runtime_cleanup("docker") from the cli-level SIGINT handler.
    if runtime == "enroot":
        _cleanup_harbor_containers()
    elif runtime == "apptainer":
        from tasktrace_capture.runtimes import runtime_cleanup

        runtime_cleanup(runtime)

    # Detect abandoned trials (dir exists but no result.json).
    post_job = find_latest_harbor_job(jobs_dir)
    if instrument_shell_commands and post_job and post_job != pre_job:
        trace_outputs = finalize_trajectories(
            post_job,
            perfetto_depth=instrument_shell_perfetto_depth,
            bash_env=IMAGE_BASHRC if source_image_bashrc else None,
            trajectory_timings=trajectory_timings,
        )
        if trace_outputs and trajectory_timings == "omit":
            print(f"  Shell command timing: wrote {len(trace_outputs)} JSONL trajectory(ies) without timings")
        elif trace_outputs:
            print(f"  Shell command timing: wrote {len(trace_outputs)} " "JSONL + Perfetto artifact set(s)")
    if profile_live_cpu and post_job and post_job != pre_job:
        profile_outputs = sorted(post_job.rglob(f"agent/{PROFILE_FILENAME}"))
        if profile_outputs:
            print(f"  Live CPU profile: wrote {len(profile_outputs)} artifact(s)")
    if post_job:
        trial_dirs = [d for d in post_job.iterdir() if d.is_dir() and not d.name.startswith(".")]
        n_with_result = sum(1 for d in trial_dirs if (d / "result.json").exists())
        n_abandoned = len(trial_dirs) - n_with_result
        if n_abandoned > 0:
            abandoned = [d.name for d in trial_dirs if not (d / "result.json").exists()]
            print(f"  WARNING: {n_abandoned} trial(s) abandoned by harbor (agent ran but no result): {abandoned}")

    # If harbor failed and produced no trial results, return an error immediately.
    if harbor_rc is not None and harbor_rc != 0:
        has_results = any(jobs_dir.rglob("result.json")) if jobs_dir.exists() else False
        if not has_results:
            return {
                "n_concurrent": n_concurrent,
                "error": f"harbor failed (rc={harbor_rc})",
                "harbor_stderr": harbor_stderr[-500:] if harbor_stderr else "",
            }, harbor_stdout

    samples = sampler_result
    if not samples:
        return {"n_concurrent": n_concurrent, "error": "no samples"}, harbor_stdout

    # Filter to samples during actual agent execution using trial timestamps.
    # This excludes container setup/teardown and verifier runs.
    active_samples = _filter_by_trial_times(samples, jobs_dir)
    if not active_samples and runtime == "current" and _agent_execution_window(jobs_dir) is None:
        # Setup, trial and teardown share one process tree here, so the fallbacks below cannot isolate the trial.
        return {
            "n_concurrent": n_concurrent,
            "error": "no CPU samples inside the trial window (Harbor recorded no agent start/finish times)",
        }, harbor_stdout
    if not active_samples:
        active_samples = [s for s in samples if s["n_procs"] > 0]
    if not active_samples:
        active_samples = [s for s in samples if s["n_containers"] > 0]
    if not active_samples:
        active_samples = samples

    avg_cpu = sum(s["cpu_total"] for s in active_samples) / len(active_samples)
    peak_cpu = max(s["cpu_total"] for s in active_samples)
    avg_cores = sum(s["cores_active"] for s in active_samples) / len(active_samples)
    peak_cores = max(s["cores_active"] for s in active_samples)
    sorted_cores = sorted(s["cores_active"] for s in active_samples)
    p95_idx = min(int(0.95 * len(sorted_cores)), len(sorted_cores) - 1)
    p95_cores = sorted_cores[p95_idx]
    avg_procs = sum(s["n_procs"] for s in active_samples) / len(active_samples)
    peak_procs = max(s["n_procs"] for s in active_samples)
    n_cores = active_samples[0]["n_cores"]

    all_cats = defaultdict(int)
    for s in active_samples:
        for cat, v in s.get("proc_cats", {}).items():
            all_cats[cat] += v
    avg_cats = {cat: round(v / len(active_samples), 1) for cat, v in all_cats.items()}

    avg_read = sum(s.get("disk_read_mb_s", 0) for s in active_samples) / len(active_samples)
    avg_write = sum(s.get("disk_write_mb_s", 0) for s in active_samples) / len(active_samples)
    peak_read = max((s.get("disk_read_mb_s", 0) for s in active_samples), default=0)
    peak_write = max((s.get("disk_write_mb_s", 0) for s in active_samples), default=0)

    result = {
        "n_concurrent": n_concurrent,
        "n_cores": n_cores,
        "duration_sec": round(samples[-1]["t"], 1),
        "n_samples": len(samples),
        "n_active_samples": len(active_samples),
        "avg_cpu": round(avg_cpu, 1),
        "peak_cpu": round(peak_cpu, 1),
        "avg_cores_active": round(avg_cores, 1),
        "peak_cores_active": peak_cores,
        "p95_cores_active": p95_cores,
        "avg_procs": round(avg_procs, 1),
        "peak_procs": peak_procs,
        "pct_cpu_active": round(avg_cpu, 1),
        "pct_waiting_llm": round(100 - avg_cpu, 1),
        "avg_proc_categories": avg_cats,
        "avg_disk_read_mb_s": round(avg_read, 1),
        "avg_disk_write_mb_s": round(avg_write, 1),
        "peak_disk_read_mb_s": round(peak_read, 1),
        "peak_disk_write_mb_s": round(peak_write, 1),
        "samples": samples,
    }

    # Inference metrics (optional — absent if vLLM endpoint unreachable)
    if inference_result:
        inf_stats = compute_inference_stats(inference_result, hist_before, hist_after)
        if inf_stats:
            result["inference"] = inf_stats
        result["inference_samples"] = inference_result

    print(
        f"\n  Result: avg_cpu={avg_cpu:.1f}% "
        f"peak_cores={peak_cores}/{n_cores} "
        f"avg_procs={avg_procs:.1f} "
        f"disk:R{avg_read:.1f}/W{avg_write:.1f}MB/s"
    )
    return result, harbor_stdout


def find_latest_harbor_job(jobs_dir):
    """Return the most recently created harbor jobs/<timestamp> directory."""
    if not jobs_dir.exists():
        return None
    dirs = sorted(jobs_dir.iterdir(), key=lambda d: d.name, reverse=True)
    for d in dirs:
        if d.is_dir():
            return d
    return None


def _parse_iso(ts):
    """Parse an ISO timestamp, handling trailing 'Z'."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _parse_claude_code_timing(result_file: Path) -> tuple[float, float, int] | None:
    """Parse LLM timing from Claude Code's stream log (claude-code.txt).

    Returns (llm_sec, wall_sec, n_turns) or None if not found.
    Claude Code emits a 'result' event with duration_api_ms (LLM time)
    and duration_ms (total wall time).  When running against a non-Anthropic
    backend (e.g. vLLM), duration_api_ms is absent — fall back to the
    session JSONL timestamps.
    """
    log_file = result_file.parent / "agent" / "claude-code.txt"
    if not log_file.exists():
        return None
    try:
        with open(log_file) as f:
            for line in f:
                d = json.loads(line)
                if d.get("type") == "result":
                    api_ms = d.get("duration_api_ms")
                    total_ms = d.get("duration_ms")
                    n_turns = d.get("num_turns", 1)
                    if api_ms is not None and total_ms is not None:
                        if api_ms <= total_ms:
                            return api_ms / 1000, total_ms / 1000, n_turns
                        # duration_api_ms > duration_ms means concurrent API
                        # calls (e.g. parallel web searches) inflated the sum.
                        # Fall through to JSONL session timing which measures
                        # real wall-clock gaps and can't over-count.
    except Exception:
        pass
    return _parse_claude_code_session_timing(result_file)


def _parse_claude_code_session_timing(result_file: Path) -> tuple[float, float, int] | None:
    """Parse LLM vs tool timing from Claude Code's session JSONL.

    Claude Code session JSONLs emit ONE EVENT PER CONTENT BLOCK of a streamed
    response — separate ``assistant`` events for ``thinking``, ``text``, and
    ``tool_use`` blocks within the same response (all sharing one
    ``message.id``). So a single model turn that streams for 30 seconds
    produces several consecutive ``assistant`` events spread across that
    interval.

    We bucket each pair of consecutive events by the gap between them:

      prev → next                   what's happening              bucket
      --------------------          --------------------          ------
      user/attachment → assistant   model generating reply        LLM
      assistant → assistant         model still streaming reply   LLM
      assistant → user/attachment   tool executing in container   tool
      anything else (queue-op)      framework setup, ignore       —

    Equivalently: a gap is LLM iff the NEXT event is ``assistant``. Tool
    time is then ``wall − llm``.

    An earlier version walked ``user → first_assistant_after`` and stopped
    there, capturing only the first content block (typically the
    ``thinking`` block) and silently re-bucketing the rest of the streamed
    response as "tool". On reasoning-heavy tasks that caused 30-75
    percentage-point overcounts of tool%, with median
    ``parsed_llm / claude-code.txt:duration_api_ms`` of 0.452 across the
    SWE-bench Pro / Terminal-Bench 2.0 / CompileBench datasets. After this
    fix the same ratio is 1.002.

    Returns (llm_sec, wall_sec, n_llm_calls) or None.
    """
    sessions_dir = result_file.parent / "agent" / "sessions" / "projects" / "-app"
    if not sessions_dir.exists():
        return None
    jsonl_files = list(sessions_dir.glob("*.jsonl"))
    if not jsonl_files:
        return None
    try:
        with open(jsonl_files[0]) as f:
            events = [json.loads(line) for line in f if line.strip()]
        # Some session JSONLs end with a `last-prompt` event that has no
        # timestamp; drop any timestamp-less event before computing.
        events = [e for e in events if "timestamp" in e]
        if len(events) < 2:
            return None

        llm_total = 0.0
        n_llm_calls = 0
        for k in range(len(events) - 1):
            if events[k + 1].get("type") != "assistant":
                continue
            try:
                gap = (_parse_iso(events[k + 1]["timestamp"]) - _parse_iso(events[k]["timestamp"])).total_seconds()
            except (KeyError, ValueError):
                continue
            if gap > 0.5:  # skip near-instant transitions (streaming overhead)
                llm_total += gap
                n_llm_calls += 1

        t_start = _parse_iso(events[0]["timestamp"])
        t_end = _parse_iso(events[-1]["timestamp"])
        wall_sec = (t_end - t_start).total_seconds()
        if wall_sec <= 0 or n_llm_calls == 0:
            return None
        return llm_total, wall_sec, n_llm_calls
    except Exception:
        return None


def get_found_suite_tasks(job_dir) -> set[str]:
    """Extract the set of task names found in a harbor job directory.

    Reads top-level ``task_name`` from each trial ``result.json``.
    Falls back to parsing the trial directory name (``<task-name>__*``)
    when the JSON field is missing.

    Returns a (possibly empty) set of task name strings.
    """
    if not job_dir or not job_dir.exists():
        return set()

    found: set[str] = set()
    for result_file in job_dir.rglob("result.json"):
        # Skip Harbor's top-level job summary
        if result_file.parent == job_dir:
            continue
        try:
            with open(result_file) as f:
                r = json.load(f)
            task_name = r.get("task_name")
            if task_name:
                found.add(task_name)
            else:
                # Fallback: parse from directory name  (e.g. "password-recovery__trial0")
                dir_name = result_file.parent.name
                if "__" in dir_name:
                    found.add(dir_name.rsplit("__", 1)[0])
        except Exception:
            continue
    return found


def _parse_mini_swe_agent_timing(result_file: Path) -> tuple[float, float, int] | None:
    """Parse LLM timing from mini-swe-agent's Harbor trajectory.json.

    Harbor's trajectory.json has timestamped steps with source=agent (LLM
    response) and source=environment (tool result).  LLM time = gap between
    tool result and next agent response.  Tool time = gap between agent
    response and next tool result.

    Returns (llm_sec, wall_sec, n_turns) or None if not parseable.
    """
    # mini-swe-agent.trajectory.json has timestamps in extra.timestamp
    # on both assistant and tool messages.  Search trial subdirs for it.
    traj_file = result_file.parent / "agent" / "mini-swe-agent.trajectory.json"
    if not traj_file.exists():
        candidates = list(result_file.parent.rglob("agent/mini-swe-agent.trajectory.json"))
        if candidates:
            traj_file = candidates[0]
        else:
            return None
    try:
        with open(traj_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    messages = data.get("messages", [])
    if len(messages) < 2:
        return None

    # Extract timestamped events from messages[].extra.timestamp
    events = []
    for m in messages:
        role = m.get("role", "")
        ts = (m.get("extra") or {}).get("timestamp")
        if ts and role in ("assistant", "tool"):
            events.append((ts, role))

    if len(events) < 2:
        return None

    llm_sec = 0.0
    n_turns = 0
    for i in range(1, len(events)):
        prev_ts, prev_src = events[i - 1]
        curr_ts, curr_src = events[i]
        try:
            # Timestamps may be ISO strings or Unix epoch floats
            if isinstance(curr_ts, (int, float)) and isinstance(prev_ts, (int, float)):
                gap = curr_ts - prev_ts
            else:
                gap = (_parse_iso(str(curr_ts)) - _parse_iso(str(prev_ts))).total_seconds()
        except Exception:
            continue
        if gap < 0.05:
            continue  # framework overhead
        # tool → assistant = LLM thinking time
        if curr_src == "assistant" and prev_src == "tool":
            llm_sec += gap
            n_turns += 1

    if n_turns == 0:
        return None

    first_ts = events[0][0]
    last_ts = events[-1][0]
    try:
        if isinstance(first_ts, (int, float)) and isinstance(last_ts, (int, float)):
            wall_sec = last_ts - first_ts
        else:
            wall_sec = (_parse_iso(str(last_ts)) - _parse_iso(str(first_ts))).total_seconds()
    except Exception:
        return None

    return (round(llm_sec, 2), round(wall_sec, 2), n_turns)


def _collect_per_call_latencies(job_dir):
    """Parse session JSONL files to get per-call LLM and tool latencies.

    Walks all trial directories under *job_dir* and reads Claude Code
    session JSONL files.  For each consecutive event pair:
      - assistant → user/tool_result gap  = tool execution latency
      - user/tool_result → assistant gap  = LLM inference latency (initial
                                            response to a tool result)
      - assistant → assistant gap         = LLM inference latency (model
                                            still streaming the same reply,
                                            see note below)

    The third case exists because Claude Code emits one event per content
    block of a streamed response (``thinking`` / ``text`` / ``tool_use``
    are separate events, all sharing one ``message.id``). An earlier
    version dropped these gaps entirely — they were neither bucketed as
    LLM nor tool — which silently undercounted LLM p95/p99 latency in the
    ``latency.llm.*`` section of report.json for vLLM Claude Code sweeps.
    Matches the methodology fix in ``_parse_claude_code_session_timing``.

    Returns (llm_latencies, tool_latencies) as lists of floats (seconds).
    """
    llm_lats: list[float] = []
    tool_lats: list[float] = []
    if not job_dir or not job_dir.exists():
        return llm_lats, tool_lats

    for sessions_dir in job_dir.rglob("sessions/projects/-app"):
        for jsonl_file in sessions_dir.glob("*.jsonl"):
            try:
                events = []
                with open(jsonl_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        d = json.loads(line)
                        ts = d.get("timestamp")
                        etype = d.get("type")
                        if ts and etype in ("user", "tool_result", "assistant"):
                            events.append((etype, _parse_iso(ts)))

                for i in range(len(events) - 1):
                    gap = (events[i + 1][1] - events[i][1]).total_seconds()
                    if gap < 0.05:
                        continue
                    cur, nxt = events[i][0], events[i + 1][0]
                    if cur == "assistant" and nxt in ("user", "tool_result"):
                        tool_lats.append(gap)
                    elif cur in ("user", "tool_result") and nxt == "assistant":
                        llm_lats.append(gap)
                    elif cur == "assistant" and nxt == "assistant":
                        # Continued streaming of the same model response
                        # (next content block of the same turn). This is
                        # still GPU-side inference time, so it belongs in
                        # the LLM bucket, not in tool.
                        llm_lats.append(gap)
            except Exception:
                continue

    return llm_lats, tool_lats


def parse_agent_time_split(job_dir, agent_timeout=None, agent="terminus-2") -> TimeSplit | None:
    """Extract LLM vs CPU time split from Harbor trial result.json files.

    Returns a TimeSplit with per-trial classification and survival rate.
    When all trials are invalid, timing fields will be None.
    """
    if not job_dir or not job_dir.exists():
        return None

    trials: list[TrialSplit] = []
    for result_file in job_dir.rglob("result.json"):
        # Skip Harbor's top-level job summary (not a trial)
        if result_file.parent == job_dir:
            continue
        try:
            with open(result_file) as f:
                r = json.load(f)
            meta = (r.get("agent_result") or {}).get("metadata") or {}
            api_times = meta.get("api_request_times_msec", [])
            agent_exec = r.get("agent_execution") or {}

            # Parse phase durations
            def _phase_dur(key, r=r):
                phase = r.get(key) or {}
                ps, pf = phase.get("started_at"), phase.get("finished_at")
                if ps and pf:
                    return round((_parse_iso(pf) - _parse_iso(ps)).total_seconds(), 2)
                return None

            env_setup_sec = _phase_dur("environment_setup")
            agent_setup_sec = _phase_dur("agent_setup")
            agent_exec_sec = _phase_dur("agent_execution")
            verifier_sec = _phase_dur("verifier")

            # Extract any trial-level exception. Field is named "verifier_error"
            # for historical reasons, but we capture the exception type whether
            # the verifier ran or not — agent-level failures (e.g. codex
            # NonZeroAgentExitCodeError, terminus-2 BadRequestError on cloud
            # backends) need the same `(N err)` annotation in the report so
            # users don't read `0.000 / 1/1 / 1/1` as a normal correctness
            # failure when the agent actually crashed.
            exc_info = r.get("exception_info") or {}
            verifier_error = exc_info.get("exception_type")
            is_timeout = verifier_error == "AgentTimeoutError"

            # Use agent_execution timestamps only — top-level spans the
            # full trial (env setup + agent setup + execution + verifier)
            # and would inflate wall time for time-split calculations.
            sa = agent_exec.get("started_at")
            fa = agent_exec.get("finished_at")

            if not sa or not fa:
                # For mini-swe-agent, agent_execution timestamps may be
                # missing.  Fall back to trajectory.json timestamps for
                # wall time and time split.
                if agent == "mini-swe-agent":
                    mswea_timing = _parse_mini_swe_agent_timing(result_file)
                    if mswea_timing:
                        llm_sec, traj_wall, n_turns = mswea_timing
                        tool_sec = traj_wall - llm_sec
                        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
                        trials.append(
                            TrialSplit(
                                status="full",
                                wall_time_sec=round(traj_wall, 2),
                                llm_time_sec=round(llm_sec, 2),
                                tool_time_sec=round(tool_sec, 2),
                                n_llm_calls=n_turns,
                                pct_cpu=round(100 * tool_sec / traj_wall, 1) if traj_wall > 0 else 0,
                                pct_llm=round(100 * llm_sec / traj_wall, 1) if traj_wall > 0 else 0,
                                reward=reward,
                                verifier_error=verifier_error,
                                env_setup_sec=env_setup_sec,
                                agent_setup_sec=agent_setup_sec,
                                agent_exec_sec=agent_exec_sec,
                                verifier_sec=verifier_sec,
                            )
                        )
                        continue
                trials.append(TrialSplit(status="invalid"))
                continue

            start = _parse_iso(sa)
            end = _parse_iso(fa)
            wall_sec = (end - start).total_seconds()

            if wall_sec <= 0:
                trials.append(TrialSplit(status="invalid"))
                continue

            # If no api_times, use agent-specific timing parser
            if not api_times:
                reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
                if is_timeout:
                    no_api_status = "partial"
                elif agent_timeout and wall_sec >= agent_timeout * 0.9:
                    no_api_status = "partial"
                else:
                    no_api_status = "full"

                # Agent-specific timing parsers (TODO: refactor into per-agent modules, issue #54)
                if agent == "claude-code":
                    agent_timing = _parse_claude_code_timing(result_file)
                elif agent == "mini-swe-agent":
                    agent_timing = _parse_mini_swe_agent_timing(result_file)
                else:
                    agent_timing = None
                if agent_timing:
                    llm_sec, _, n_turns = agent_timing
                    tool_sec = wall_sec - llm_sec
                    trials.append(
                        TrialSplit(
                            status=no_api_status,
                            wall_time_sec=round(wall_sec, 2),
                            llm_time_sec=round(llm_sec, 2),
                            tool_time_sec=round(tool_sec, 2),
                            n_llm_calls=n_turns,
                            pct_cpu=round(100 * tool_sec / wall_sec, 1),
                            pct_llm=round(100 * llm_sec / wall_sec, 1),
                            reward=reward,
                            verifier_error=verifier_error,
                            env_setup_sec=env_setup_sec,
                            agent_setup_sec=agent_setup_sec,
                            agent_exec_sec=agent_exec_sec,
                            verifier_sec=verifier_sec,
                        )
                    )
                else:
                    trials.append(
                        TrialSplit(
                            status=no_api_status,
                            wall_time_sec=round(wall_sec, 2),
                            llm_time_sec=0,
                            tool_time_sec=round(wall_sec, 2),
                            n_llm_calls=0,
                            pct_cpu=100.0,
                            pct_llm=0.0,
                            reward=reward,
                            verifier_error=verifier_error,
                            env_setup_sec=env_setup_sec,
                            agent_setup_sec=agent_setup_sec,
                            agent_exec_sec=agent_exec_sec,
                            verifier_sec=verifier_sec,
                        )
                    )
                continue

            llm_sec = min(sum(api_times) / 1000, wall_sec)  # clamp for overlap
            tool_sec = wall_sec - llm_sec
            reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            n_episodes = meta.get("n_episodes", len(api_times))

            if is_timeout:
                status = "partial"
            elif agent_timeout and wall_sec >= agent_timeout * 0.9:
                status = "partial"
            else:
                status = "full"

            trials.append(
                TrialSplit(
                    status=status,
                    wall_time_sec=round(wall_sec, 2),
                    llm_time_sec=round(llm_sec, 2),
                    tool_time_sec=round(tool_sec, 2),
                    n_llm_calls=len(api_times),
                    pct_cpu=round(100 * tool_sec / wall_sec, 1),
                    pct_llm=round(100 * llm_sec / wall_sec, 1),
                    reward=reward,
                    verifier_error=verifier_error,
                    n_episodes=n_episodes,
                    env_setup_sec=env_setup_sec,
                    agent_setup_sec=agent_setup_sec,
                    agent_exec_sec=agent_exec_sec,
                    verifier_sec=verifier_sec,
                )
            )
        except Exception:
            trials.append(TrialSplit(status="invalid"))
            continue

    if not trials:
        return None

    full = [t for t in trials if t.status == "full"]
    partial = [t for t in trials if t.status == "partial"]
    invalid = [t for t in trials if t.status == "invalid"]
    valid = full + partial

    if not valid:
        return TimeSplit(
            n_invalid=len(invalid),
            survival_rate=f"0/{len(trials)}",
            trials=trials,
        )

    avg_tool = sum(t.tool_time_sec for t in valid) / len(valid)
    avg_wall = sum(t.wall_time_sec for t in valid) / len(valid)
    avg_llm = sum(t.llm_time_sec for t in valid) / len(valid)
    # How many trials the verifier actually scored (others errored out).
    n_verified = sum(1 for t in valid if t.reward is not None)
    # Verifier-errored trials have reward=None — count them as 0.0 so the
    # average reflects actual success rate, not just the passing subset.
    # BUT: if EVERY trial errored, propagate None so the report can render
    # "n/a" instead of a misleading "0.000" (the average of nothing isn't
    # zero — it's undefined). The renderer tags these with `(N err)` either
    # way; the value distinction is between "0 of N succeeded" vs "no trial
    # produced a real result."
    has_real_reward = any(t.reward is not None for t in valid)
    rewards = [t.reward if t.reward is not None else 0.0 for t in valid]
    verifier_errs = [t.verifier_error for t in valid if t.verifier_error]

    def _avg_phase(attr):
        vals = [getattr(t, attr) for t in valid if getattr(t, attr) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    # Per-call latency distribution from session JSONL timestamps.
    llm_lats, tool_lats = _collect_per_call_latencies(job_dir)

    def _pctl(data, p):
        if not data:
            return None
        s = sorted(data)
        return round(s[min(int(p * len(s)), len(s) - 1)], 2)

    return TimeSplit(
        pct_cpu=round(100 * avg_tool / avg_wall, 1) if avg_wall > 0 else None,
        pct_llm=round(100 * avg_llm / avg_wall, 1) if avg_wall > 0 else None,
        tool_time_sec=round(avg_tool, 2),
        wall_time_sec=round(avg_wall, 2),
        llm_time_sec=round(avg_llm, 2),
        n_llm_calls=round(sum(t.n_llm_calls for t in valid) / len(valid), 1),
        avg_reward=(round(sum(rewards) / len(rewards), 3) if rewards and has_real_reward else None),
        n_full=len(full),
        n_partial=len(partial),
        n_invalid=len(invalid),
        n_verifier_errors=len(verifier_errs),
        verifier_errors=sorted(set(verifier_errs)),
        n_verified=n_verified,
        survival_rate=f"{len(full)}/{len(trials)}",
        trials=trials,
        avg_env_setup_sec=_avg_phase("env_setup_sec"),
        avg_agent_setup_sec=_avg_phase("agent_setup_sec"),
        avg_agent_exec_sec=_avg_phase("agent_exec_sec"),
        avg_verifier_sec=_avg_phase("verifier_sec"),
        llm_latency_median=_pctl(llm_lats, 0.50),
        llm_latency_p95=_pctl(llm_lats, 0.95),
        llm_latency_p99=_pctl(llm_lats, 0.99),
        llm_latency_max=round(max(llm_lats), 2) if llm_lats else None,
        tool_latency_median=_pctl(tool_lats, 0.50),
        tool_latency_p95=_pctl(tool_lats, 0.95),
        tool_latency_p99=_pctl(tool_lats, 0.99),
        tool_latency_max=round(max(tool_lats), 2) if tool_lats else None,
        n_llm_calls_tracked=len(llm_lats) if llm_lats else None,
        n_tool_calls_tracked=len(tool_lats) if tool_lats else None,
    )

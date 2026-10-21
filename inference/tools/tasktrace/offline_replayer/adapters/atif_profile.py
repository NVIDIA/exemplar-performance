#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile supported ATIF trajectory actions in a measured task environment.

Docker and Enroot modes create a task environment from an image. Current mode
uses an already-provisioned, exclusive container such as a Kubernetes task
sandbox and measures its existing cgroup directly.

The profiler keeps controller work inside the measured container so cgroup and
scheduler measurements cover the actual replay workload.  It also provides
controller-placement modes needed to preserve Terminal-Bench task semantics:
some task images already contain Python, while others expect `python3` to be
absent or install it during their own recorded actions.

The supported input family is Codex-style ATIF plus the provenance-gated
``mini-swe-agent-1.1`` ATIF wrapper normalized by ``atif_replay``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..measurement.container import (
    DEFAULT_NETWORK_POLICY_PROXY_IMAGE,
    PROFILE_RUNTIMES,
    PROFILE_SETS,
    SCHEDULER_BACKENDS,
    STRACE_MODES,
    ProfileConfig,
    ProfileError,
    cpu_summary_lines,
    memory_oom_kill_observed,
    preflight_profile,
)
from ..measurement.current import validate_current_profile_config
from ..measurement.network_policy import (
    NetworkMode,
    NetworkPolicy,
    NetworkPolicyError,
    resolve_cli_network_policy,
)
from ..measurement.pipeline import ContainerMeasurementOptions, run_profiled_replay
from .atif_contract import (
    CONTROLLER_ENV_VARS_MARKER,
    workload_environment_presence_variable,
    workload_environment_variable,
)
from .atif_replay import (
    PACING_MODES,
    AtifTrajectory,
    read_trajectory,
    validate_supported_actions,
)
from .transcript_replay import ReplayError

TOOL_SOURCE_MOUNT = "/tasktrace-source"
TRAJECTORY_MOUNT = "/tasktrace-input"
REPLAY_MOUNT = "/tasktrace"
REPLAY_PYTHON = "/tmp/tasktrace-python"
HOST_RUNTIME_MOUNT = "/tasktrace-controller-runtime"
# The validated 8-core and 16-core accepted report sets used this mounted
# Python path.  Do not silently rename it in result reproduction: hiding the
# name changes the observable process environment for trajectories.
HOST_RUNTIME_EXECUTABLE = f"{HOST_RUNTIME_MOUNT}/bin/python3"
HOST_RUNTIME_LOADER = f"{HOST_RUNTIME_MOUNT}/lib/ld-linux-x86-64.so.2"
HOST_RUNTIME_SOURCE_PYTHON = Path("/usr/bin/python3")
HOST_RUNTIME_LIBRARY_ROOT = Path("/usr/lib")
HOST_RUNTIME_LOADER_NAMES = {
    "aarch64-linux-gnu": "ld-linux-aarch64.so.1",
    "x86_64-linux-gnu": "ld-linux-x86-64.so.2",
}
RUST_CONTROLLER_ENV = "TASKTRACE_RUST_REPLAY_EXECUTABLE"
RUN_CANCEL_FILE_ENV = "TASKTRACE_RUN_CANCEL_FILE"
RUST_CONTROLLER_DIGEST_ENV = "TASKTRACE_RUST_CONTROLLER_SHA256"
RUST_CONTROLLER_MOUNT = "/tasktrace-controller"
RUST_CONTROLLER_EXECUTABLE = f"{RUST_CONTROLLER_MOUNT}/tasktrace"
CONTROLLER_RUNTIME_MODES = ("image", "host-mount", "rust-host-mount")
ATIF_PROFILE_PREFLIGHT_SCHEMA_VERSION = "tasktrace-atif-profile-preflight.v1"
STAGE_REPLAY_PYTHON_COMMAND = f'cp -L "$(command -v python3)" {REPLAY_PYTHON} && chmod 755 {REPLAY_PYTHON}'
HIDE_PREPARED_PYTHON3_COMMAND = "rm -f /usr/bin/python3 /usr/local/bin/python3"


@dataclass(frozen=True)
class AtifTrajectoryProfileConfig:
    trajectory: Path
    image: str | None
    output_dir: Path
    cpu_limit: float | None
    cpuset_cpus: str | None
    allocated_cpus: float | None
    pacing: str
    timeout_sec: float | None
    prepare_command: str | None
    strace_mode: str
    string_limit: int
    shell: str
    workdir: str
    task_name: str | None = None
    capture_action_outputs: bool = False
    hide_prepared_python3: bool = False
    controller_runtime: str = "image"
    patch_roots: tuple[str, ...] = ("/tmp",)
    scheduler_concurrency: bool = False
    scheduler_attach_timeout_sec: float = 20.0
    scheduler_cpu_tolerance_fraction: float = 0.02
    scheduler_cpu_tolerance_sec: float = 0.005
    scheduler_backend: str = "bcc"
    scheduler_bcc_buffer_pages: int | None = None
    scheduler_perf_binary: str = "perf"
    scheduler_perf_mmap_pages: int | None = None
    proc_stat_comparison: bool = False
    proc_stat_interval_sec: float = 0.5
    proc_stat_baseline_sec: float = 5.0
    memory_profile: bool = False
    memory_sample_interval_sec: float = 0.1
    memory_limit_bytes: int | None = None
    memory_swap_total_limit_bytes: int | None = None
    network: str | None = None
    network_policy: NetworkPolicy | None = None
    network_policy_proxy_image: str = DEFAULT_NETWORK_POLICY_PROXY_IMAGE
    network_policy_external_network: str = "bridge"
    runtime: str = "docker"
    docker: str = "docker"
    enroot: str = "enroot"
    current_container_exclusive: bool = False
    profile_sets: tuple[str, ...] = ()
    coordinated_baseline: bool = False
    verification_command: str | None = None


@dataclass(frozen=True)
class HostControllerRuntime:
    """Paths for one self-consistent host Python controller runtime."""

    python: Path
    stdlib: Path
    multiarch: str
    library_dir: Path
    loader: Path


@dataclass(frozen=True)
class RustControllerRuntime:
    """Content identity for the delegated native replay executable."""

    executable: Path
    sha256: str


@dataclass(frozen=True)
class CurrentTaskEnvironment:
    """Environment changes that remove controller-only source and virtualenv state."""

    assignments: tuple[str, ...]
    unset_names: tuple[str, ...]


def _shell_join(arguments: list[str]) -> str:
    return " ".join(shlex.quote(argument) for argument in arguments)


def _replay_command(
    config: AtifTrajectoryProfileConfig,
    host_runtime: HostControllerRuntime | None = None,
    rust_runtime: RustControllerRuntime | None = None,
) -> str:
    """Build the in-container controller command without polluting task commands.

    Controller-only environment values are named in
    ``TASKTRACE_CONTROLLER_ENV_VARS`` so replayed shell commands can strip them
    before executing the recorded workload.
    """
    launcher = [REPLAY_PYTHON, "-m", "offline_replayer.adapters.atif_replay"]
    controller_env_names = ["PYTHONPATH"]
    environment = [
        f'{workload_environment_variable("PYTHONPATH")}="${{PYTHONPATH-}}"',
        f'{workload_environment_presence_variable("PYTHONPATH")}="${{PYTHONPATH+1}}"',
        f"PYTHONPATH={TOOL_SOURCE_MOUNT}",
    ]
    if config.controller_runtime == "host-mount":
        host_runtime = host_runtime or _host_controller_runtime()
        launcher = [
            HOST_RUNTIME_LOADER,
            "--library-path",
            f"{HOST_RUNTIME_MOUNT}/lib/{host_runtime.multiarch}",
            HOST_RUNTIME_EXECUTABLE,
            "-m",
            "offline_replayer.adapters.atif_replay",
        ]
        environment.extend(
            [
                f'{workload_environment_variable("PYTHONHOME")}="${{PYTHONHOME-}}"',
                f'{workload_environment_presence_variable("PYTHONHOME")}="${{PYTHONHOME+1}}"',
                f"PYTHONHOME={HOST_RUNTIME_MOUNT}",
            ]
        )
        controller_env_names.append("PYTHONHOME")
    elif config.controller_runtime == "rust-host-mount":
        rust_runtime = rust_runtime or _rust_controller_runtime()
        launcher = [RUST_CONTROLLER_EXECUTABLE, "replay", "atif"]
        controller_env_names = [RUST_CONTROLLER_DIGEST_ENV]
        environment = [
            f'{workload_environment_variable(RUST_CONTROLLER_DIGEST_ENV)}="${{{RUST_CONTROLLER_DIGEST_ENV}-}}"',
            f'{workload_environment_presence_variable(RUST_CONTROLLER_DIGEST_ENV)}="${{{RUST_CONTROLLER_DIGEST_ENV}+1}}"',
            f"{RUST_CONTROLLER_DIGEST_ENV}={rust_runtime.sha256}",
        ]
    environment.append(f"{CONTROLLER_ENV_VARS_MARKER}={':'.join(controller_env_names)}")
    command = launcher + [
        f"{TRAJECTORY_MOUNT}/{config.trajectory.name}",
        "--output",
        f"{REPLAY_MOUNT}/replay_report.json",
        "--cwd",
        config.workdir,
        "--pacing",
        config.pacing,
    ]
    if config.capture_action_outputs:
        command.extend(["--action-output-dir", f"{REPLAY_MOUNT}/action_outputs"])
    if config.timeout_sec is not None:
        command.extend(["--timeout-sec", str(config.timeout_sec)])
    for patch_root in config.patch_roots:
        command.extend(["--patch-root", patch_root])
    return f"{' '.join(environment)} {_shell_join(command)}"


def _current_replay_command(config: AtifTrajectoryProfileConfig) -> str:
    tool_source = Path(__file__).resolve().parents[2]
    runner = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(tool_source)!r});"
        "runpy.run_module("
        "'offline_replayer.adapters.atif_replay',"
        "run_name='__main__',alter_sys=True)"
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        runner,
        str(config.trajectory.resolve()),
        "--output",
        str((config.output_dir / "replay" / "replay_report.json").resolve()),
        "--cwd",
        config.workdir,
        "--pacing",
        config.pacing,
    ]
    if config.capture_action_outputs:
        command.extend(
            [
                "--action-output-dir",
                str((config.output_dir / "replay" / "action_outputs").resolve()),
            ]
        )
    if config.timeout_sec is not None:
        command.extend(["--timeout-sec", str(config.timeout_sec)])
    for patch_root in config.patch_roots:
        command.extend(["--patch-root", patch_root])
    return _shell_join(command)


def _normalized_environment_path(value: str, *, cwd: Path) -> Path:
    path = Path(value or ".").expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _current_task_environment(
    environment: Mapping[str, str] | None = None,
    *,
    controller_source: Path | None = None,
    controller_executable: Path | None = None,
    controller_prefix: Path | None = None,
    controller_base_prefix: Path | None = None,
    invocation_cwd: Path | None = None,
) -> CurrentTaskEnvironment:
    """Remove only environment state introduced by the replay controller.

    The current-container backend shares one filesystem and process namespace
    with the task. The controller therefore runs by absolute interpreter path,
    while task setup and recorded commands receive PATH/PYTHONPATH values with
    the controller virtualenv and source checkout removed.
    """
    source = (controller_source or Path(__file__).resolve().parents[2]).resolve()
    executable = Path(controller_executable or sys.executable).resolve()
    prefix = Path(controller_prefix or sys.prefix).resolve()
    base_prefix = Path(controller_base_prefix or sys.base_prefix).resolve()
    cwd = (invocation_cwd or Path.cwd()).resolve()
    values = dict(os.environ if environment is None else environment)

    controller_virtualenv = prefix if prefix != base_prefix else None
    virtualenv_variable_is_controller = False
    virtualenv_raw = values.get("VIRTUAL_ENV")
    if virtualenv_raw:
        candidate = _normalized_environment_path(virtualenv_raw, cwd=cwd)
        executable_inside = executable == candidate or candidate in executable.parents
        if prefix == candidate or executable_inside:
            controller_virtualenv = candidate
            virtualenv_variable_is_controller = True

    assignments: list[str] = []
    unset_names: set[str] = set()
    path_changed = False
    pythonpath_changed = False

    path_raw = values.get("PATH")
    if path_raw is not None and controller_virtualenv is not None:
        kept_path: list[str] = []
        for entry in path_raw.split(os.pathsep):
            resolved = _normalized_environment_path(entry, cwd=cwd)
            if resolved == controller_virtualenv or controller_virtualenv in resolved.parents:
                path_changed = True
            else:
                kept_path.append(entry)
        if not kept_path:
            raise ProfileError(
                "current-container profiling cannot derive a task PATH after " "removing the controller virtualenv"
            )
        if path_changed:
            assignments.append(f"PATH={os.pathsep.join(kept_path)}")

    pythonpath_raw = values.get("PYTHONPATH")
    if pythonpath_raw is not None:
        kept_pythonpath: list[str] = []
        for entry in pythonpath_raw.split(os.pathsep):
            resolved = _normalized_environment_path(entry, cwd=cwd)
            from_controller_source = resolved == source or source in resolved.parents
            from_controller_virtualenv = controller_virtualenv is not None and (
                resolved == controller_virtualenv or controller_virtualenv in resolved.parents
            )
            if from_controller_source or from_controller_virtualenv:
                pythonpath_changed = True
            else:
                kept_pythonpath.append(entry)
        if pythonpath_changed:
            unset_names.add("PYTHONPATH")
            if kept_pythonpath:
                assignments.append(f"PYTHONPATH={os.pathsep.join(kept_pythonpath)}")

    if "UV_RUN_RECURSION_DEPTH" in values:
        unset_names.add("UV_RUN_RECURSION_DEPTH")
    if controller_virtualenv is not None:
        if virtualenv_variable_is_controller:
            unset_names.add("VIRTUAL_ENV")
            if "VIRTUAL_ENV_PROMPT" in values:
                unset_names.add("VIRTUAL_ENV_PROMPT")

    return CurrentTaskEnvironment(
        assignments=tuple(assignments),
        unset_names=tuple(sorted(unset_names)),
    )


def _prepare_command(config: AtifTrajectoryProfileConfig) -> str | None:
    """Prepare only the replay driver required by the selected fidelity mode.

    Host-mounted controllers run off PATH and need no in-image staging; either
    can still hide the image's own python3 when the source observed it absent.
    """
    commands = []
    if config.prepare_command is not None:
        commands.append(f"({config.prepare_command})")
    if config.controller_runtime == "image":
        commands.append(STAGE_REPLAY_PYTHON_COMMAND)
    if config.hide_prepared_python3:
        commands.append(HIDE_PREPARED_PYTHON3_COMMAND)
    return " && ".join(commands) if commands else None


def _host_controller_runtime() -> HostControllerRuntime:
    """Resolve the system Python runtime used inside measured task images.

    The profiler may itself run under a ``uv`` or virtualenv Python version.
    Reading ``sysconfig`` from that interpreter while mounting
    ``/usr/bin/python3`` would mix runtime versions and prevent the controller
    from starting in the container.
    """
    python = HOST_RUNTIME_SOURCE_PYTHON.resolve()
    if not python.exists():
        raise ProfileError(f"host controller runtime path is missing: {python}")
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import json, sysconfig; "
                    "print(json.dumps({'stdlib': sysconfig.get_paths()['stdlib'], "
                    "'multiarch': sysconfig.get_config_var('MULTIARCH')}))"
                ),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        metadata = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as exc:
        raise ProfileError(f"could not inspect host controller runtime: {python}") from exc
    stdlib = Path(str(metadata["stdlib"])).resolve()
    multiarch = metadata.get("multiarch")
    if not isinstance(multiarch, str) or not multiarch:
        raise ProfileError("host-mounted controller runtime requires Python MULTIARCH")
    loader_name = HOST_RUNTIME_LOADER_NAMES.get(multiarch)
    if loader_name is None:
        raise ProfileError("host-mounted controller runtime does not support Python MULTIARCH: " f"{multiarch}")
    library_dir = HOST_RUNTIME_LIBRARY_ROOT / multiarch
    loader = library_dir / loader_name
    for path in (stdlib, library_dir, loader):
        if not path.exists():
            raise ProfileError(f"host controller runtime path is missing: {path}")
    return HostControllerRuntime(
        python=python,
        stdlib=stdlib,
        multiarch=multiarch,
        library_dir=library_dir,
        loader=loader,
    )


def _host_controller_runtime_mounts(runtime: HostControllerRuntime) -> tuple[str, ...]:
    """Mount a read-only host Python runtime without installing into the image.

    The mounted interpreter remains named ``python3`` because that is the
    runtime configuration used for the accepted Terminal-Bench report sets.
    """
    return (
        f"type=bind,src={runtime.python},dst={HOST_RUNTIME_EXECUTABLE},readonly",
        f"type=bind,src={runtime.stdlib},dst={HOST_RUNTIME_MOUNT}/lib/{runtime.stdlib.name},readonly",
        f"type=bind,src={runtime.library_dir},dst={HOST_RUNTIME_MOUNT}/lib/{runtime.multiarch},readonly",
        f"type=bind,src={runtime.loader},dst={HOST_RUNTIME_LOADER},readonly",
    )


def _rust_controller_runtime(
    environment: Mapping[str, str] | None = None,
) -> RustControllerRuntime:
    """Resolve and content-bind the native executable supplied by the dispatcher."""
    values = os.environ if environment is None else environment
    raw = values.get(RUST_CONTROLLER_ENV)
    if not raw:
        raise ProfileError("rust-host-mount requires invocation through the native " "tasktrace dispatcher")
    executable = Path(raw).expanduser().resolve()
    if not executable.is_file():
        raise ProfileError(f"Rust replay controller is not a file: {executable}")
    if not os.access(executable, os.X_OK):
        raise ProfileError(f"Rust replay controller is not executable: {executable}")
    return RustControllerRuntime(
        executable=executable,
        sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
    )


def _rust_controller_runtime_mounts(
    runtime: RustControllerRuntime,
) -> tuple[str, ...]:
    return (f"type=bind,src={runtime.executable},dst={RUST_CONTROLLER_EXECUTABLE},readonly",)


def _validate_current_adapter_extensions(config: AtifTrajectoryProfileConfig) -> None:
    """Reject adapter features the read-only current runtime cannot execute.

    The current backend owns its CPU-only cgroup boundary.  Docker-oriented
    profiling, verification, and network controls must fail closed instead of
    disappearing when the adapter builds the smaller current-runtime request.
    """
    scheduler_options_requested = (
        config.scheduler_attach_timeout_sec != 20.0
        or config.scheduler_cpu_tolerance_fraction != 0.02
        or config.scheduler_cpu_tolerance_sec != 0.005
        or config.scheduler_backend != "bcc"
        or config.scheduler_perf_binary != "perf"
        or config.scheduler_perf_mmap_pages is not None
    )
    if scheduler_options_requested:
        raise ProfileError("scheduler profiling options are not supported by current-container runtime")
    if config.proc_stat_interval_sec != 0.5 or config.proc_stat_baseline_sec != 5.0:
        raise ProfileError("proc-stat sampling options are not supported by current-container runtime")
    if config.docker != "docker" or config.enroot != "enroot":
        raise ProfileError("runtime executable overrides are not supported by current-container runtime")
    memory_options_requested = (
        config.memory_profile
        or config.memory_sample_interval_sec != 0.1
        or config.memory_limit_bytes is not None
        or config.memory_swap_total_limit_bytes is not None
    )
    if memory_options_requested:
        raise ProfileError("memory profiling options are not supported by " "current-container runtime")
    if config.profile_sets:
        raise ProfileError("--profile-set is not supported by current-container runtime")
    if config.coordinated_baseline:
        raise ProfileError("--coordinated-baseline is not supported by current-container runtime")
    if config.verification_command is not None:
        raise ProfileError("--verification-command is not supported by current-container runtime")
    if (
        config.network_policy is not None
        or config.network_policy_proxy_image != DEFAULT_NETWORK_POLICY_PROXY_IMAGE
        or config.network_policy_external_network != "bridge"
    ):
        raise ProfileError("network policy controls are not supported by current-container runtime")


def build_container_profile_config(
    config: AtifTrajectoryProfileConfig,
) -> tuple[ProfileConfig, AtifTrajectory]:
    """Translate trajectory replay requirements into a measured runtime request."""
    trajectory = read_trajectory(config.trajectory)
    validate_supported_actions(trajectory)
    if config.runtime == "current":
        _validate_current_adapter_extensions(config)
        if config.controller_runtime != "image":
            raise ProfileError("current-container runtime does not support a separate controller runtime")
        if config.hide_prepared_python3:
            raise ProfileError("current-container runtime cannot hide the task environment's python3")
        task_environment = _current_task_environment()
        profile = ContainerMeasurementOptions(
            image=config.image,
            cpu_limit=config.cpu_limit,
            cpuset_cpus=config.cpuset_cpus,
            allocated_cpus=config.allocated_cpus,
            strace_mode=config.strace_mode,
            string_limit=config.string_limit,
            shell=config.shell,
            workdir=config.workdir,
            scheduler_concurrency=config.scheduler_concurrency,
            scheduler_attach_timeout_sec=config.scheduler_attach_timeout_sec,
            scheduler_cpu_tolerance_fraction=config.scheduler_cpu_tolerance_fraction,
            scheduler_cpu_tolerance_sec=config.scheduler_cpu_tolerance_sec,
            scheduler_backend=config.scheduler_backend,
            scheduler_bcc_buffer_pages=config.scheduler_bcc_buffer_pages,
            scheduler_perf_binary=config.scheduler_perf_binary,
            scheduler_perf_mmap_pages=config.scheduler_perf_mmap_pages,
            proc_stat_comparison=config.proc_stat_comparison,
            proc_stat_interval_sec=config.proc_stat_interval_sec,
            proc_stat_baseline_sec=config.proc_stat_baseline_sec,
            network=config.network,
            runtime="current",
            current_container_exclusive=config.current_container_exclusive,
            env=task_environment.assignments,
            unset_env=task_environment.unset_names,
        ).build_profile(
            command=_current_replay_command(config),
            prepare_command=config.prepare_command,
            output_dir=config.output_dir / "container_profile",
            mounts=(),
        )
        validate_current_profile_config(profile)
        return profile, trajectory
    # The replay module is mounted from the repository root so its package
    # import path remains identical inside the measured container.
    host_runtime = _host_controller_runtime() if config.controller_runtime == "host-mount" else None
    rust_runtime = _rust_controller_runtime() if config.controller_runtime == "rust-host-mount" else None
    tool_source = Path(__file__).resolve().parents[2]
    replay_dir = config.output_dir / "replay"
    mounts = (
        f"type=bind,src={config.trajectory.resolve().parent},dst={TRAJECTORY_MOUNT},readonly",
        f"type=bind,src={replay_dir.resolve()},dst={REPLAY_MOUNT}",
    )
    if rust_runtime is not None:
        mounts += _rust_controller_runtime_mounts(rust_runtime)
    else:
        mounts = (
            f"type=bind,src={tool_source},dst={TOOL_SOURCE_MOUNT},readonly",
            *mounts,
        )
    if host_runtime is not None:
        mounts += _host_controller_runtime_mounts(host_runtime)
    source_identity = {
        "format": "atif",
        "schema_version": trajectory.schema_version,
        "sha256": hashlib.sha256(config.trajectory.read_bytes()).hexdigest(),
        "action_count": len(trajectory.actions),
    }
    if rust_runtime is not None:
        source_identity["rust_controller_binary_sha256"] = rust_runtime.sha256
    profile = ContainerMeasurementOptions(
        image=config.image,
        cpu_limit=config.cpu_limit,
        cpuset_cpus=config.cpuset_cpus,
        allocated_cpus=config.allocated_cpus,
        strace_mode=config.strace_mode,
        string_limit=config.string_limit,
        shell=config.shell,
        workdir=config.workdir,
        scheduler_concurrency=config.scheduler_concurrency,
        scheduler_attach_timeout_sec=config.scheduler_attach_timeout_sec,
        scheduler_cpu_tolerance_fraction=config.scheduler_cpu_tolerance_fraction,
        scheduler_cpu_tolerance_sec=config.scheduler_cpu_tolerance_sec,
        scheduler_backend=config.scheduler_backend,
        scheduler_bcc_buffer_pages=config.scheduler_bcc_buffer_pages,
        scheduler_perf_binary=config.scheduler_perf_binary,
        scheduler_perf_mmap_pages=config.scheduler_perf_mmap_pages,
        proc_stat_comparison=config.proc_stat_comparison,
        proc_stat_interval_sec=config.proc_stat_interval_sec,
        proc_stat_baseline_sec=config.proc_stat_baseline_sec,
        memory_profile=config.memory_profile,
        memory_sample_interval_sec=config.memory_sample_interval_sec,
        memory_limit_bytes=config.memory_limit_bytes,
        memory_swap_total_limit_bytes=config.memory_swap_total_limit_bytes,
        terminate_descendants_at_finish=True,
        network=config.network,
        network_policy=config.network_policy,
        network_policy_proxy_image=config.network_policy_proxy_image,
        network_policy_external_network=config.network_policy_external_network,
        runtime=config.runtime,
        docker=config.docker,
        enroot=config.enroot,
        profile_sets=config.profile_sets,
        coordinated_baseline=config.coordinated_baseline,
        verification_command=config.verification_command,
    ).build_profile(
        command=_replay_command(config, host_runtime, rust_runtime),
        prepare_command=_prepare_command(config),
        output_dir=config.output_dir / "container_profile",
        mounts=mounts,
        workload_identity={
            "task_name": config.task_name or trajectory.task_name,
            "trial_name": trajectory.trial_name,
        },
        source_identity=source_identity,
        expected_outcome={
            "kind": "source-recorded-action-outcomes",
            "action_count": len(trajectory.actions),
        },
    )
    return profile, trajectory


def run_atif_trajectory_profile(
    config: AtifTrajectoryProfileConfig,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute one replay and join fidelity checks with CPU measurements.

    Cooperative cancellation is forwarded to the coordinated Docker profile;
    unsupported runtime/profile combinations fail before execution begins.
    """
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ProfileError(f"pipeline output directory must be new or empty: {config.output_dir}")
    profile, trajectory = build_container_profile_config(config)
    controller_identity = (
        {
            "kind": "rust-host-mounted-binary",
            "sha256": profile.source_identity["rust_controller_binary_sha256"],
            "container_path": RUST_CONTROLLER_EXECUTABLE,
        }
        if config.controller_runtime == "rust-host-mount"
        else None
    )
    return run_profiled_replay(
        profile=profile,
        output_dir=config.output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={
            "schema_version": "container-atif-trajectory-profile.v1",
            "task_name": config.task_name or trajectory.task_name,
            "source_trial": trajectory.trial_name,
            "source_trajectory": str(config.trajectory.resolve()),
            "source_trajectory_sha256": hashlib.sha256(config.trajectory.read_bytes()).hexdigest(),
            "source_trajectory_schema_version": trajectory.schema_version,
            "source_agent_original_format": trajectory.source_agent_original_format,
            "agent": trajectory.agent,
            "image": config.image,
            "pacing": config.pacing,
            "controller_runtime": config.controller_runtime,
            **({"controller_identity": controller_identity} if controller_identity is not None else {}),
            "hide_prepared_python3": config.hide_prepared_python3,
            "patch_roots": list(config.patch_roots),
            "network": config.network,
            "action_output_capture": {
                "enabled": config.capture_action_outputs,
                "directory": (
                    str((config.output_dir / "replay" / "action_outputs").resolve())
                    if config.capture_action_outputs
                    else None
                ),
            },
        },
        cancel_requested=cancel_requested,
    )


def preflight_atif_trajectory_profile(
    config: AtifTrajectoryProfileConfig,
) -> dict[str, Any]:
    """Validate one ATIF profile without creating output or running its actions."""
    if config.output_dir.exists():
        if not config.output_dir.is_dir():
            raise ProfileError(f"pipeline output path is not a directory: {config.output_dir}")
        if any(config.output_dir.iterdir()):
            raise ProfileError(f"pipeline output directory must be new or empty: {config.output_dir}")
    profile, trajectory = build_container_profile_config(config)
    profile_preflight = preflight_profile(profile)
    source_sha256 = hashlib.sha256(config.trajectory.read_bytes()).hexdigest()
    checks = [
        {
            "name": "atif_source",
            "status": "passed",
            "schema_version": trajectory.schema_version,
            "supported_action_count": len(trajectory.actions),
        },
        *profile_preflight["checks"],
    ]
    return {
        **profile_preflight,
        "schema_version": ATIF_PROFILE_PREFLIGHT_SCHEMA_VERSION,
        "ready": True,
        "task_name": config.task_name or trajectory.task_name,
        "source_trial": trajectory.trial_name,
        "source_sha256": source_sha256,
        "source_schema_version": trajectory.schema_version,
        "source_action_count": len(trajectory.actions),
        "runtime": profile.runtime,
        "image_identity": profile_preflight["image_identity"],
        "profile_sets": list(profile.profile_sets),
        "cpuset_cpus": profile.cpuset_cpus,
        "allocated_cpus": profile.allocated_cpus,
        "checks": checks,
        "limitations": list(profile_preflight["limitations"]),
        **(
            {
                "controller_identity": {
                    "kind": "rust-host-mounted-binary",
                    "sha256": profile.source_identity["rust_controller_binary_sha256"],
                    "container_path": RUST_CONTROLLER_EXECUTABLE,
                }
            }
            if config.controller_runtime == "rust-host-mount"
            else {}
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile supported actions from an ATIF trajectory in a measured task environment."
    )
    parser.add_argument("trajectory", type=Path, help="ATIF trajectory.json path.")
    parser.add_argument(
        "--image",
        help="Prepared task image. Omit with --runtime current inside an existing sandbox.",
    )
    parser.add_argument(
        "--runtime",
        choices=PROFILE_RUNTIMES,
        default="docker",
        help=(
            "Execution runtime. Use current inside an exclusive, already-provisioned "
            "container such as a Kubernetes task sandbox."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    parser.add_argument("--task-name", help=argparse.SUPPRESS)
    parser.add_argument("--cpu-limit", type=float, help="Docker CPU quota matching the task declaration.")
    parser.add_argument("--cpuset-cpus", help="Optional logical CPU affinity for the replay.")
    parser.add_argument("--allocated-cpus", type=float, help="CPU denominator without --cpuset-cpus.")
    parser.add_argument("--pacing", choices=PACING_MODES, default="recorded")
    parser.add_argument("--timeout-sec", type=float, help="Per-command/session completion timeout.")
    parser.add_argument(
        "--capture-action-outputs",
        action="store_true",
        help=(
            "Retain complete foreground action stdout/stderr under the replay "
            "output directory. Interactive sessions already retain full logs."
        ),
    )
    parser.add_argument(
        "--prepare-command",
        help="Replay-environment setup performed before measured replay.",
    )
    parser.add_argument(
        "--verification-command",
        help=(
            "Command run in the replay sandbox after measurement; a nonzero "
            "exit invalidates the profile without changing measured counters."
        ),
    )
    parser.add_argument(
        "--controller-runtime",
        choices=CONTROLLER_RUNTIME_MODES,
        default="image",
        help=(
            "For Docker/Enroot, run the controller from the task image (default) "
            "or select an explicit hidden read-only Python/Rust host-mounted runtime."
        ),
    )
    parser.add_argument(
        "--hide-prepared-python3",
        action="store_true",
        help=(
            "Remove python3 from normal command lookup after staging the private "
            "controller. Use only when setup added Python and source actions "
            "observed it as absent."
        ),
    )
    parser.add_argument("--strace-mode", choices=STRACE_MODES, default="none")
    profiler_mode = parser.add_mutually_exclusive_group()
    profiler_mode.add_argument(
        "--coordinated-baseline",
        action="store_true",
        help=(
            "Run an unprofiled replay through the same coordinator, sandbox "
            "gates, lifecycle, and execution plan used by --profile-set."
        ),
    )
    profiler_mode.add_argument(
        "--profile-set",
        action="append",
        choices=PROFILE_SETS,
        default=[],
        help="Coordinated profiler preset; repeat to combine independent profilers.",
    )
    parser.add_argument("--strace-string-limit", type=int, default=4096)
    parser.add_argument("--scheduler-concurrency", action="store_true")
    parser.add_argument("--scheduler-backend", choices=SCHEDULER_BACKENDS, default="bcc")
    parser.add_argument("--scheduler-bcc-buffer-pages", type=int)
    parser.add_argument("--scheduler-perf-binary", default="perf")
    parser.add_argument("--scheduler-perf-mmap-pages", type=int)
    parser.add_argument("--proc-stat-comparison", action="store_true")
    parser.add_argument("--proc-stat-interval-sec", type=float, default=0.5)
    parser.add_argument("--proc-stat-baseline-sec", type=float, default=5.0)
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Collect controlled-lifecycle cgroup memory and aligned sampled CPU over time.",
    )
    parser.add_argument("--memory-sample-interval-sec", type=float, default=0.1)
    parser.add_argument("--memory-limit-bytes", type=int)
    parser.add_argument("--memory-swap-total-limit-bytes", type=int)
    parser.add_argument("--scheduler-attach-timeout-sec", type=float, default=20.0)
    parser.add_argument("--scheduler-cpu-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--scheduler-cpu-tolerance-sec", type=float, default=0.005)
    parser.add_argument("--shell", default="/bin/bash")
    parser.add_argument("--workdir", default="/app")
    parser.add_argument(
        "--network",
        help=(
            "Deprecated raw Docker network mode. Use --network-policy; "
            "--network none remains a compatible alias for no-network."
        ),
    )
    parser.add_argument(
        "--network-policy",
        choices=tuple(mode.value for mode in NetworkMode),
        help="Portable replay policy: public, no-network, or allowlist.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Exact DNS hostname permitted by allowlist policy; repeatable.",
    )
    parser.add_argument(
        "--network-policy-proxy-image",
        default=DEFAULT_NETWORK_POLICY_PROXY_IMAGE,
        help="Pre-pulled Python image used by the TaskTrace allowlist gateway.",
    )
    parser.add_argument(
        "--network-policy-external-network",
        default="bridge",
        help="Docker network used only by the allowlist proxy for outbound access.",
    )
    parser.add_argument(
        "--patch-root",
        action="append",
        help="Additional patchable path root in the replay environment (default: /tmp).",
    )
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--enroot", default="enroot")
    parser.add_argument(
        "--current-container-exclusive",
        action="store_true",
        help=(
            "Assert that the current container is a fresh single-task sandbox. "
            "Required for current-runtime CPU data to be marked reportable."
        ),
    )
    args = parser.parse_args(argv)
    try:
        network_policy = resolve_cli_network_policy(
            mode=args.network_policy,
            allowed_hosts=args.allowed_host,
            legacy_network=args.network,
        )
    except NetworkPolicyError as exc:
        parser.error(str(exc))
    if args.runtime != "current" and args.image is None:
        parser.error("--image is required for Docker and Enroot runtimes")
    if args.runtime == "current" and args.image is not None:
        parser.error("--runtime current does not accept --image")
    if args.runtime != "current" and args.cpuset_cpus is None and args.allocated_cpus is None:
        parser.error("provide --cpuset-cpus or --allocated-cpus for utilization reporting")
    if args.runtime == "current" and args.cpuset_cpus is not None:
        parser.error("--runtime current cannot enforce --cpuset-cpus; configure the sandbox allocation")
    if args.runtime == "current" and args.cpu_limit is not None:
        parser.error("--runtime current cannot enforce --cpu-limit; configure the sandbox allocation")
    if args.runtime == "current" and args.controller_runtime != "image":
        parser.error(f"--controller-runtime {args.controller_runtime} is not applicable " "to --runtime current")
    if args.runtime == "current" and args.hide_prepared_python3:
        parser.error("--hide-prepared-python3 is unsafe with --runtime current")
    if args.runtime == "current" and args.scheduler_concurrency:
        parser.error("--scheduler-concurrency is not supported by --runtime current")
    if args.runtime == "current" and args.scheduler_bcc_buffer_pages is not None:
        parser.error("--scheduler-bcc-buffer-pages is not supported by --runtime current")
    if args.runtime == "current" and args.proc_stat_comparison:
        parser.error("--proc-stat-comparison is not supported by --runtime current")
    if args.runtime == "current" and (
        args.network is not None
        or args.network_policy is not None
        or args.allowed_host
        or args.network_policy_proxy_image != DEFAULT_NETWORK_POLICY_PROXY_IMAGE
        or args.network_policy_external_network != "bridge"
    ):
        parser.error("network policy controls are not supported by --runtime current")
    if args.runtime == "current":
        # Current-container execution inherits the scheduler-created sandbox's
        # networking. Do not attach the default semantic public policy because
        # this runtime cannot establish or verify that policy itself.
        network_policy = None
    if args.runtime != "current" and args.current_container_exclusive:
        parser.error("--current-container-exclusive requires --runtime current")
    if args.cpu_limit is not None and args.cpu_limit <= 0:
        parser.error("--cpu-limit must be positive")
    if args.allocated_cpus is not None and args.allocated_cpus <= 0:
        parser.error("--allocated-cpus must be positive")
    if args.runtime == "enroot" and args.cpu_limit is not None:
        parser.error("--cpu-limit is Docker-only; use Slurm/taskset CPU allocation for enroot")
    if args.runtime == "enroot" and (
        args.network is not None
        or (network_policy is not None and network_policy.effective_mode is not NetworkMode.PUBLIC)
    ):
        parser.error("non-public network policy is currently Docker-only")
    if args.runtime == "enroot" and args.scheduler_concurrency:
        parser.error("--scheduler-concurrency is currently Docker-only")
    if args.runtime == "enroot" and args.proc_stat_comparison:
        parser.error("--proc-stat-comparison is currently Docker-only")
    if args.runtime == "enroot" and args.memory_profile:
        parser.error("--memory-profile is currently Docker-only")
    if args.runtime == "enroot" and args.memory_limit_bytes is not None:
        parser.error("--memory-limit-bytes is currently Docker-only")
    if args.runtime == "enroot" and args.memory_swap_total_limit_bytes is not None:
        parser.error("--memory-swap-total-limit-bytes is currently Docker-only")
    if args.runtime == "enroot" and args.strace_mode != "none":
        parser.error("strace tracing requires Docker ptrace capabilities; use --strace-mode none")
    if args.runtime == "enroot" and args.profile_set:
        parser.error("--profile-set currently requires the Docker runtime")
    if args.runtime == "enroot" and args.coordinated_baseline:
        parser.error("--coordinated-baseline currently requires the Docker runtime")
    if args.timeout_sec is not None and args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    # --hide-prepared-python3 works in every image-managed controller mode:
    # 'image' stages a private Python first, while both host-mounted runtimes
    # execute off-PATH and can hide the image's python3 directly.
    if args.strace_string_limit <= 0:
        parser.error("--strace-string-limit must be positive")
    if args.scheduler_attach_timeout_sec <= 0:
        parser.error("--scheduler-attach-timeout-sec must be positive")
    if args.scheduler_cpu_tolerance_fraction < 0 or args.scheduler_cpu_tolerance_sec < 0:
        parser.error("scheduler CPU tolerances must not be negative")
    if (
        args.scheduler_backend != "bcc"
        and not args.scheduler_concurrency
        and not {"cpu-compare", "cpu-peak"} & set(args.profile_set)
    ):
        parser.error("--scheduler-backend requires --scheduler-concurrency")
    if args.proc_stat_comparison and not args.cpuset_cpus:
        parser.error("--proc-stat-comparison requires --cpuset-cpus")
    if args.scheduler_perf_mmap_pages is not None and args.scheduler_perf_mmap_pages <= 0:
        parser.error("--scheduler-perf-mmap-pages must be positive")
    if args.scheduler_bcc_buffer_pages is not None and (
        args.scheduler_bcc_buffer_pages <= 0 or args.scheduler_bcc_buffer_pages & (args.scheduler_bcc_buffer_pages - 1)
    ):
        parser.error("--scheduler-bcc-buffer-pages must be a positive power of two")
    if args.proc_stat_interval_sec <= 0 or args.proc_stat_baseline_sec <= 0:
        parser.error("proc-stat interval and baseline durations must be positive")
    if args.memory_sample_interval_sec <= 0:
        parser.error("--memory-sample-interval-sec must be positive")
    if args.memory_limit_bytes is not None and args.memory_limit_bytes <= 0:
        parser.error("--memory-limit-bytes must be positive")
    if args.memory_swap_total_limit_bytes is not None and args.memory_swap_total_limit_bytes <= 0:
        parser.error("--memory-swap-total-limit-bytes must be positive")
    if args.memory_swap_total_limit_bytes is not None and args.memory_limit_bytes is None:
        parser.error("--memory-swap-total-limit-bytes requires --memory-limit-bytes")
    if (
        args.memory_swap_total_limit_bytes is not None
        and args.memory_limit_bytes is not None
        and args.memory_swap_total_limit_bytes < args.memory_limit_bytes
    ):
        parser.error("--memory-swap-total-limit-bytes must be at least --memory-limit-bytes")
    config = AtifTrajectoryProfileConfig(
        trajectory=args.trajectory,
        image=args.image,
        output_dir=args.output_dir,
        cpu_limit=args.cpu_limit,
        cpuset_cpus=args.cpuset_cpus,
        allocated_cpus=args.allocated_cpus,
        pacing=args.pacing,
        timeout_sec=args.timeout_sec,
        prepare_command=args.prepare_command,
        verification_command=args.verification_command,
        strace_mode=args.strace_mode,
        string_limit=args.strace_string_limit,
        shell=args.shell,
        workdir=args.workdir,
        capture_action_outputs=args.capture_action_outputs,
        hide_prepared_python3=args.hide_prepared_python3,
        controller_runtime=args.controller_runtime,
        patch_roots=tuple(args.patch_root or ["/tmp"]),
        scheduler_concurrency=args.scheduler_concurrency,
        scheduler_attach_timeout_sec=args.scheduler_attach_timeout_sec,
        scheduler_cpu_tolerance_fraction=args.scheduler_cpu_tolerance_fraction,
        scheduler_cpu_tolerance_sec=args.scheduler_cpu_tolerance_sec,
        scheduler_backend=args.scheduler_backend,
        scheduler_bcc_buffer_pages=args.scheduler_bcc_buffer_pages,
        scheduler_perf_binary=args.scheduler_perf_binary,
        scheduler_perf_mmap_pages=args.scheduler_perf_mmap_pages,
        proc_stat_comparison=args.proc_stat_comparison,
        proc_stat_interval_sec=args.proc_stat_interval_sec,
        proc_stat_baseline_sec=args.proc_stat_baseline_sec,
        memory_profile=args.memory_profile,
        memory_sample_interval_sec=args.memory_sample_interval_sec,
        memory_limit_bytes=args.memory_limit_bytes,
        memory_swap_total_limit_bytes=args.memory_swap_total_limit_bytes,
        network=args.network,
        network_policy=network_policy,
        network_policy_proxy_image=args.network_policy_proxy_image,
        network_policy_external_network=args.network_policy_external_network,
        runtime=args.runtime,
        docker=args.docker,
        enroot=args.enroot,
        current_container_exclusive=args.current_container_exclusive,
        profile_sets=tuple(args.profile_set),
        coordinated_baseline=args.coordinated_baseline,
        task_name=args.task_name,
    )
    try:
        cancellation_path_text = os.environ.get(RUN_CANCEL_FILE_ENV)
        cancellation_path = Path(cancellation_path_text).resolve() if cancellation_path_text else None
        report = run_atif_trajectory_profile(
            config,
            **({"cancel_requested": cancellation_path.exists} if cancellation_path is not None else {}),
        )
    except (OSError, ProfileError, ReplayError) as exc:
        parser.error(str(exc))
    cpu = report["container_cpu"]
    replay = report["replay_summary"]
    print(f"Task: {report['task_name']}")
    print(f"Source trial: {report['source_trial']}")
    print(
        "Actions: "
        f"{replay['actions_executed']}/{replay['actions_recorded']} executed, "
        f"{replay['actions_timed_out']} timed out, "
        f"{replay['source_outcome_mismatch_count']} source-outcome mismatches"
    )
    for line in cpu_summary_lines(cpu):
        print(line)
    measurement_validity = report.get("measurement_validity")
    if measurement_validity is not None:
        print(
            "Current-container CPU validity: "
            + ("valid" if measurement_validity.get("valid_for_average_cpu_reporting") else "invalid")
        )
    if report.get("container_memory") is not None:
        memory = report["container_memory"]
        peak = memory["lifetime"]["peak_bytes"]
        print(
            "Sandbox controlled-work memory peak: "
            + (f"{peak / (1024 * 1024):.3f} MiB" if isinstance(peak, int) else "n/a")
        )
        print(f"CPU/memory timeline: {memory['samples_path']}")
    if report.get("scheduler_concurrency") is not None:
        scheduler = report["scheduler_concurrency"]
        validation = scheduler["cgroup_validation"]
        print(f"Peak simultaneous logical cores: {scheduler['peak_simultaneous_logical_cores']}")
        print(
            "Scheduler/cgroup CPU validation: "
            f"{'valid' if validation['valid_for_concurrency_reporting'] else 'invalid'}"
        )
    if report.get("warning"):
        print(f"Warning: {report['warning']}")
    print(f"Report: {config.output_dir / 'pipeline_report.json'}")
    invalid_scheduler = (
        report.get("scheduler_concurrency") is not None
        and not report["scheduler_concurrency"]["cgroup_validation"]["valid_for_concurrency_reporting"]
    )
    invalid_measurement = measurement_validity is not None and not measurement_validity.get(
        "valid_for_average_cpu_reporting", False
    )
    invalid_memory = report.get("container_memory") is not None and (
        not report["container_memory"].get("instrumentation_valid", False)
        or not report["container_memory"].get("peak_valid", False)
        or not report["container_memory"].get("timeline_valid_for_shape", True)
        or memory_oom_kill_observed(report["container_memory"])
    )
    invalid_coordinated_profiling = report.get("profiling_valid") is False
    return int(
        report.get("status") != "valid"
        or replay["source_outcome_mismatch_count"] > 0
        or replay.get("source_outcome_comparison_unavailable_count", 0) > 0
        or invalid_scheduler
        or invalid_measurement
        or invalid_memory
        or invalid_coordinated_profiling
    )


if __name__ == "__main__":
    sys.exit(main())

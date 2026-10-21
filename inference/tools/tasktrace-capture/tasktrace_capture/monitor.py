# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CPU, disk, and process monitoring via /proc.

Samples system metrics at regular intervals and detects processes running
inside enroot containers by comparing mount namespaces.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict

CLK_TCK = os.sysconf("SC_CLK_TCK")

# Whole-disk device-name prefixes for /proc/diskstats readers. Shared with
# tasktrace_capture.node_telemetry (which extends it with md/dm- for the raw stream).
DISK_DEVICE_PREFIXES = ("sd", "nvme", "vd", "xvd")


# ─── System info collection ─────────────────────────────────────────────────


def _run(cmd, **kwargs):
    """Run a command, return stdout or None on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, **kwargs)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _collect_cpu():
    """CPU details from lscpu."""
    out = _run(["lscpu"])
    if not out:
        return None
    info = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if key == "Model name":
            info["model"] = val
        elif key == "Socket(s)":
            info["sockets"] = int(val)
        elif key == "Core(s) per socket":
            info["cores_per_socket"] = int(val)
        elif key == "Thread(s) per core":
            info["threads_per_core"] = int(val)
        elif key == "CPU max MHz":
            info["max_mhz"] = float(val)
        elif key == "CPU min MHz":
            info["min_mhz"] = float(val)
        elif key.startswith("NUMA node") and key.endswith("CPU(s)"):
            info.setdefault("numa_nodes", []).append({"node": key, "cpus": val})
    # Current frequency from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            freqs = re.findall(r"cpu MHz\s*:\s*([\d.]+)", f.read())
        if freqs:
            vals = [float(x) for x in freqs]
            info["current_mhz_avg"] = round(sum(vals) / len(vals), 1)
    except Exception:
        pass
    # Cache sizes from lscpu
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if key.startswith("L1d cache"):
            info["l1d_cache"] = val
        elif key.startswith("L1i cache"):
            info["l1i_cache"] = val
        elif key.startswith("L2 cache"):
            info["l2_cache"] = val
        elif key.startswith("L3 cache"):
            info["l3_cache"] = val
    return info


def _collect_gpu():
    """GPU info from nvidia-smi.

    Stores the full ``nvidia-smi -q`` output as a raw string so every
    detail (PCIe gen/width, ECC, clocks, power limits, serial, NVLink,
    temperatures, etc.) is captured without field-by-field parsing.
    A small structured summary is extracted alongside for quick access.
    """
    # Full dump — captures everything
    raw = _run(["nvidia-smi", "-q"])

    # Structured summary for quick access in report.txt header
    csv_out = _run(
        ["nvidia-smi", "--query-gpu=name,vbios_version,driver_version,memory.total", "--format=csv,noheader,nounits"]
    )
    summary = []
    if csv_out:
        for line in csv_out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                summary.append(
                    {
                        "name": parts[0],
                        "vbios": parts[1],
                        "driver": parts[2],
                        "memory_mb": int(float(parts[3])),
                    }
                )

    if not raw and not summary:
        return None
    result = {}
    if summary:
        result["summary"] = summary
    if raw:
        result["nvidia_smi_q"] = raw
    return result


def _collect_memory():
    """RAM from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            text = f.read()
        info = {}
        for key, field in [("MemTotal", "total_gb"), ("MemAvailable", "available_gb")]:
            m = re.search(rf"{key}:\s+(\d+)\s+kB", text)
            if m:
                info[field] = round(int(m.group(1)) / (1024 * 1024), 1)
        return info
    except Exception:
        return None


def _collect_os():
    """OS distro and kernel."""
    info = {}
    try:
        info["kernel"] = os.uname().release
    except Exception:
        pass
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    info["distro"] = line.split("=", 1)[1].strip().strip('"')
                    break
    except Exception:
        pass
    return info or None


def _collect_slurm():
    """SLURM cluster/job info from environment."""
    info = {}
    for env_key, field in [
        ("SLURM_CLUSTER_NAME", "cluster"),
        ("SLURM_JOB_ID", "job_id"),
        ("SLURM_JOB_NODELIST", "nodelist"),
        ("SLURM_JOB_PARTITION", "partition"),
        ("SLURM_JOB_NUM_NODES", "num_nodes"),
        ("SLURM_GPUS_ON_NODE", "gpus_on_node"),
    ]:
        val = os.environ.get(env_key)
        if val:
            info[field] = val
    # Bright Cluster Manager fallbacks (SLURM_CLUSTER_NAME often unset)
    if "cluster" not in info:
        for fallback in ("CLUSTER_NAME", "CMD_WLM_CLUSTER_NAME"):
            val = os.environ.get(fallback)
            if val:
                info["cluster"] = val
                break
    return info or None


def _collect_infiniband():
    """InfiniBand device info."""
    out = _run(["ibv_devinfo"])
    if not out:
        return None
    devices = []
    current = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("hca_id:"):
            if current:
                devices.append(current)
            current = {"hca_id": line.split(":", 1)[1].strip()}
        elif ":" in line and current:
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if key in ("fw_ver", "board_id", "transport", "node_guid"):
                current[key] = val
    if current:
        devices.append(current)
    return devices or None


def _collect_virtualization():
    """Detect virtualization type."""
    out = _run(["systemd-detect-virt"])
    if out is None:
        return None
    return out if out != "none" else "bare-metal"


def _collect_versions(runtime=None):
    """Selected container runtime and Harbor versions."""
    import importlib.metadata

    info = {}
    apptainer = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_BIN")
    if runtime == "apptainer" and not apptainer:
        from tasktrace_capture.apptainer_env import apptainer_binary

        apptainer = apptainer_binary()
    runtime_commands: dict[str, list[str] | None] = {
        "enroot": ["enroot", "version"],
        "docker": ["docker", "--version"],
        "apptainer": [apptainer, "version"] if apptainer else None,
    }
    commands = []
    if runtime == "opensandbox":
        try:
            info["opensandbox"] = importlib.metadata.version("opensandbox")
        except importlib.metadata.PackageNotFoundError:
            pass
    elif runtime == "current":
        # No runtime binary: record the container's own image identity.
        import platform

        from tasktrace_capture.current_env import IMAGE_REF_ENV, current_workdir

        info["current_image_ref"] = os.environ.get(IMAGE_REF_ENV) or "unknown"
        info["current_container_arch"] = platform.machine()
        info["current_workdir"] = current_workdir()
    elif runtime in runtime_commands:
        command = runtime_commands[runtime]
        if command is not None:
            commands.append((runtime, command))
    else:
        commands.extend((name, command) for name, command in runtime_commands.items() if command is not None)
    commands.append(("harbor", ["harbor", "version"]))
    for name, cmd in commands:
        kwargs = {}
        if name == "apptainer":
            from tasktrace_capture.apptainer_env import clean_apptainer_environment

            kwargs["env"] = clean_apptainer_environment()
        out = _run(cmd, **kwargs)
        if out:
            info[name] = out.splitlines()[0]
    return info or None


def _collect_kernel_cmdline():
    """Kernel boot parameters (frequency clamping, IOMMU, etc.)."""
    try:
        with open("/proc/cmdline") as f:
            return f.read().strip()
    except Exception:
        return None


def collect_system_info(runtime=None):
    """Collect comprehensive system metadata for cross-cluster comparison.

    Runs once at startup (~2-3s). Every call is try/except wrapped —
    missing commands or permissions just produce null fields.
    """
    collectors = {
        "cpu": _collect_cpu,
        "gpu": _collect_gpu,
        "memory": _collect_memory,
        "os": _collect_os,
        "slurm": _collect_slurm,
        "infiniband": _collect_infiniband,
        "virtualization": _collect_virtualization,
        "versions": lambda: _collect_versions(runtime),
        "kernel_cmdline": _collect_kernel_cmdline,
    }
    info = {}
    for key, fn in collectors.items():
        try:
            info[key] = fn()
        except Exception:
            info[key] = None
    return info


# ─── /proc readers ───────────────────────────────────────────────────────────


def read_proc_stat():
    """Parse /proc/stat for per-CPU tick counts."""
    result = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            vals = [int(x) for x in parts[1:9]]
            result[name] = {
                "user": vals[0] + vals[1],
                "system": vals[2] + vals[5] + vals[6],
                "idle": vals[3],
                "iowait": vals[4],
            }
    return result


def cpu_pct(prev, curr):
    """Compute per-CPU usage percentages between two snapshots."""
    result = {}
    for name in curr:
        if name not in prev:
            continue
        p, c = prev[name], curr[name]
        d_user = c["user"] - p["user"]
        d_sys = c["system"] - p["system"]
        d_idle = c["idle"] - p["idle"]
        d_iowait = c["iowait"] - p["iowait"]
        total = d_user + d_sys + d_idle + d_iowait
        if total == 0:
            result[name] = 0
        else:
            result[name] = round(100 * (d_user + d_sys) / total, 1)
    return result


def get_cpu_info():
    """Read CPU model and core count from /proc/cpuinfo."""
    info = {"model": "unknown", "cores_logical": 0}
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
        models = re.findall(r"model name\s*:\s*(.+)", text)
        if models:
            info["model"] = models[0].strip()
        info["cores_logical"] = len(models)
    except Exception:
        pass
    return info


def read_diskstats():
    """Read /proc/diskstats, return total sectors read/written.

    Only whole-disk rows are summed: a partition's IO is already counted in
    its parent device's row, so including nvme0n1p* alongside nvme0n1 would
    double every total (verified live: the nvme0n1 row equals the sum of its
    p1/p2/p3 rows).
    """
    total_read = 0
    total_write = 0
    try:
        whole_disks = set(os.listdir("/sys/block"))
    except OSError:
        whole_disks = None
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 14:
                    continue
                name = parts[2]
                if not any(name.startswith(p) for p in DISK_DEVICE_PREFIXES):
                    continue
                if whole_disks is not None and name not in whole_disks:
                    continue
                total_read += int(parts[5])
                total_write += int(parts[9])
    except Exception:
        pass
    return total_read, total_write


# ─── Container detection ─────────────────────────────────────────────────────


def get_n_containers(runtime="enroot"):
    """Count TaskTrace Capture/Harbor containers for the selected runtime."""
    try:
        # OpenSandbox workloads execute remotely. Querying the controller on
        # every 500 ms monitoring sample would be both expensive and misleading.
        if runtime == "opensandbox":
            return 0
        if runtime == "current":
            # Count this container only while the trial's processes run, as enroot counts its containers.
            return 1 if any(True for _ in _iter_descendant_pids(skip_idle=True)) else 0
        if runtime == "apptainer":
            from tasktrace_capture.apptainer_env import (
                APPTAINER_INSTANCE_PREFIX,
                apptainer_binary,
                clean_apptainer_environment,
            )

            binary = apptainer_binary()
            if binary is None:
                return 0
            result = subprocess.run(
                [binary, "instance", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=3,
                cwd="/",
                env=clean_apptainer_environment(),
            )
            if result.returncode != 0:
                return 0
            payload = json.loads(result.stdout or "{}")
            return sum(
                1
                for instance in payload.get("instances", [])
                if str(instance.get("instance") or instance.get("name") or "").startswith(APPTAINER_INSTANCE_PREFIX)
            )
        if runtime == "docker":
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    "label=com.docker.compose.service=main",
                    "--filter",
                    "name=__",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode != 0:
                return 0
            return len([line for line in result.stdout.strip().splitlines() if line.strip()])

        result = subprocess.run(["enroot", "list"], capture_output=True, text=True, timeout=3)
        lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
        return len(lines)
    except Exception:
        return 0


def _get_init_mnt_ns():
    """Get the host mount namespace ID.

    Try PID 1 first; fall back to our own PID (tasktrace-capture always runs on
    the host, so our namespace IS the host namespace).  On HPC clusters
    /proc/1/ns/mnt is often unreadable (root-owned PID 1).
    """
    for target in ("/proc/1/ns/mnt", f"/proc/{os.getpid()}/ns/mnt"):
        try:
            return os.readlink(target)
        except (FileNotFoundError, PermissionError):
            continue
    return None


def _iter_container_pids(skip_idle=False):
    """Yield (pid, cmdline) for every process in a non-host mount namespace
    owned by the current user.

    Enroot containers use separate mount namespaces (but not PID namespaces),
    so mount namespace != host namespace is the ground truth for "container
    process".  Unlike Docker, enroot has no PID namespace or cgroup to auto-kill
    processes when a container is removed — orphans survive with PPID=1.

    Only processes in our own cgroup count, which leaves out Slurm services in
    other steps of the same job (e.g. /usr/sbin/sybil in step_extern).
    """
    init_ns = _get_init_mnt_ns()
    if init_ns is None:
        return

    my_uid = os.getuid()
    my_cgroup = _read_cgroup("self")
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid_str}").st_uid != my_uid:
                continue
            if my_cgroup is not None and _read_cgroup(pid_str) != my_cgroup:
                continue
            if os.readlink(f"/proc/{pid_str}/ns/mnt") == init_ns:
                continue
            with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
            if not cmdline or len(cmdline) < 3:
                continue
            if skip_idle and any(cmdline.startswith(x) for x in ["sleep", "/pause", "sh -c sleep"]):
                continue
            yield int(pid_str), cmdline
        except (FileNotFoundError, PermissionError, ValueError):
            continue


def _iter_descendant_pids(root_pid=None, skip_idle=False):
    """Yield (pid, cmdline) for every descendant of root_pid (default: this process): the current runtime's
    workload, which enroot's mount-namespace check cannot see."""
    root_pid = os.getpid() if root_pid is None else root_pid
    children = defaultdict(list)
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        try:
            with open(f"/proc/{pid_str}/stat", "rb") as f:
                stat = f.read().decode("utf-8", errors="replace")
            # comm may contain spaces/parens: the ppid is the 2nd field after the closing paren
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
            children[ppid].append(int(pid_str))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    queue = list(children.get(root_pid, []))
    seen = set()
    while queue:
        pid = queue.pop()
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(children.get(pid, []))
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
        except (FileNotFoundError, PermissionError):
            continue
        if not cmdline or len(cmdline) < 3:
            continue
        if skip_idle and any(cmdline.startswith(x) for x in ["sleep", "/pause", "sh -c sleep"]):
            continue
        yield pid, cmdline


def _iter_workload_pids(runtime="enroot", skip_idle=False):
    """The processes doing the trial's work for this runtime: container processes (enroot's mount-namespace
    ground truth) or, in place, the coordinator's descendants."""
    if runtime == "current":
        return _iter_descendant_pids(skip_idle=skip_idle)
    return _iter_container_pids(skip_idle=skip_idle)


def count_enroot_procs(runtime="enroot"):
    """Count and categorize the trial's processes (inside enroot containers, or — current runtime — the
    coordinator's descendants in this container)."""
    counts = defaultdict(int)
    total = 0
    try:
        for _pid, cmdline in _iter_workload_pids(runtime, skip_idle=True):
            total += 1
            cmd = cmdline.lower()
            if "python" in cmd:
                counts["python"] += 1
            elif "pip" in cmd or "uv " in cmd:
                counts["pip/install"] += 1
            elif "grep" in cmd or "rg " in cmd:
                counts["grep"] += 1
            elif "git " in cmd:
                counts["git"] += 1
            elif "apt" in cmd or "dpkg" in cmd:
                counts["apt/dpkg"] += 1
            elif "bash" in cmd or "/bin/sh" in cmd:
                counts["bash/sh"] += 1
            elif "node" in cmd or "npm" in cmd:
                counts["node/npm"] += 1
            elif "curl" in cmd or "wget" in cmd:
                counts["curl/wget"] += 1
            elif "make" in cmd or "gcc" in cmd:
                counts["compile"] += 1
            elif "sed " in cmd or "awk " in cmd:
                counts["sed/awk"] += 1
            elif "echo" in cmd or "tee" in cmd:
                counts["echo/tee"] += 1
            else:
                counts["other"] += 1
    except Exception:
        pass
    return total, dict(counts)


def _get_protected_namespaces():
    """Build the set of mount namespace IDs that must NOT be killed.

    Protected = host namespace + every live enroot container's namespace.

    Why: enroot doesn't use PID namespaces (unlike Docker). When a container
    is removed, orphaned processes keep running in the host PID space with
    their original mount namespace.  We kill orphans by namespace, but must
    NOT kill processes in live containers (e.g. a vLLM inference server).

    A live container's namespace is obtained by exec'ing into it and reading
    /proc/self/ns/mnt.  If enroot list fails, returns None — the caller
    must NOT kill anything (fail-safe: killing vLLM is worse than missing
    an orphan).
    """
    protected = set()

    # Host namespace is always protected.
    host_ns = _get_init_mnt_ns()
    if host_ns:
        protected.add(host_ns)

    # Get all live container names.  If enroot is broken or unavailable,
    # we cannot build the protected set — return None to signal "don't kill".
    try:
        r = subprocess.run(
            ["enroot", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
    except Exception:
        return None

    # Get the actual mount namespace IDs of running container processes.
    #
    # IMPORTANT: cannot use "enroot start --rw <name> readlink ..." because
    # enroot start creates a NEW namespace each time — it would return a
    # different namespace ID than what the container's processes actually use.
    #
    # Instead, use "enroot list --fancy" which shows each container's PID
    # and MNTNS.  Then read /proc/<pid>/ns/mnt for the real namespace ID.
    try:
        r_fancy = subprocess.run(
            ["enroot", "list", "--fancy"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r_fancy.returncode == 0:
            for line in r_fancy.stdout.strip().split("\n")[1:]:  # skip header
                fields = line.split()
                if len(fields) >= 7:
                    pid_str = fields[1]
                    try:
                        ns = os.readlink(f"/proc/{pid_str}/ns/mnt")
                        protected.add(ns)
                    except (FileNotFoundError, PermissionError):
                        pass
    except Exception:
        pass  # --fancy unavailable or failed — rely on PPID check as safety net

    return protected


def _read_ppid(pid):
    """Read PPID from /proc/<pid>/status. Returns 0 on failure."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def _read_cgroup(pid):
    """Read /proc/<pid>/cgroup. Returns None on failure."""
    try:
        with open(f"/proc/{pid}/cgroup", "rb") as f:
            return f.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def kill_orphaned_container_procs():
    """Kill processes from removed containers, preserving live ones.

    After enroot remove -f, orphaned processes keep running because enroot
    doesn't use PID namespaces (unlike Docker, where killing container PID 1
    auto-kills everything via the kernel).

    Three independent safety checks — ALL must pass to kill:

    0. Cgroup check: process is in our own cgroup.  Container processes
       inherit it from us and keep it after reparenting; Slurm services in
       another step of the same job (e.g. /usr/sbin/sybil in step_extern)
       look like orphans to checks 1 and 2 but are in a different cgroup.

    1. Namespace check: process is NOT in a protected namespace (built from
       all live enroot containers via enroot list + enroot exec readlink).
       This protects vLLM and other live containers.

    2. PPID check: process (or its parent) has PPID=1, meaning it was
       reparented to init when its container was destroyed.  Live container
       processes have a real parent chain (e.g. slurmstepd → bash → vllm).
       This is a second safety valve — even if the namespace check fails
       (e.g. container removed from enroot list between check and kill),
       vLLM processes are still protected by their non-orphan PPID.

    If enroot list fails or our cgroup is unreadable, kills nothing
    (fail-safe: killing vLLM is catastrophically worse than missing an orphan).

    Must be called AFTER enroot remove — before removal, the harbor
    container's namespace would still be in the protected set.

    Returns list of (pid, cmdline) that were killed.
    """
    protected = _get_protected_namespaces()
    if protected is None:
        # Cannot determine which namespaces are safe — kill nothing.
        return []
    my_cgroup = _read_cgroup("self")
    if my_cgroup is None:
        return []

    my_uid = os.getuid()
    killed = []

    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            # Only kill our own processes.
            if os.stat(f"/proc/{pid}").st_uid != my_uid:
                continue
            # Safety check 0: must be in our cgroup (our Slurm step).
            if _read_cgroup(pid) != my_cgroup:
                continue
            # Safety check 1: namespace must NOT be protected.
            ns = os.readlink(f"/proc/{pid}/ns/mnt")
            if ns in protected:
                continue
            # Safety check 2: must be an orphan (PPID=1 or parent's PPID=1).
            # Orphaned container processes get reparented to init (PID 1) when
            # their container is destroyed.  Live container processes (vLLM)
            # have a real parent chain (slurmstepd → bash → vllm).
            ppid = _read_ppid(pid)
            is_orphan = (ppid == 1) or (ppid > 1 and _read_ppid(ppid) == 1)
            if not is_orphan:
                continue
            # Both checks passed — this is an orphan from a dead container.
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", errors="replace").replace("\x00", " ").strip()
            if not cmdline:
                continue
            os.kill(pid, signal.SIGKILL)
            killed.append((pid, cmdline))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            # Process vanished between scan and kill, or unreadable — skip.
            continue

    if killed:
        for pid, cmd in killed:
            print(f"  Killed orphan PID {pid}: {cmd[:100]}", file=sys.stderr)

    return killed


# ─── Sampling loop ───────────────────────────────────────────────────────────


def sample_baseline(duration_sec=5):
    """Sample idle CPU for duration_sec seconds; returns a summary dict.

    Call this before starting any harbor runs to establish a baseline.
    Runs silently (no stderr output).
    """
    import threading

    stop = threading.Event()
    result = []

    def _run():
        result.extend(sample_cpu(stop, quiet=True))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(duration_sec)
    stop.set()
    t.join(timeout=5)

    if not result:
        return {}
    return {
        "duration_sec": duration_sec,
        "avg_cpu_pct": round(sum(s["cpu_total"] for s in result) / len(result), 1),
        "avg_cores_active": round(sum(s["cores_active"] for s in result) / len(result), 1),
        "peak_cores_active": max(s["cores_active"] for s in result),
        "n_cores": result[0]["n_cores"],
    }


def sample_cpu(stop_event, interval=0.5, quiet=False, runtime="enroot"):
    """Sample CPU/disk/process metrics until stop_event is set.

    Returns a list of sample dicts with timestamps, CPU %, core counts,
    container counts, process counts + categories, and disk I/O rates.
    """
    samples = []
    prev_stat = read_proc_stat()
    prev_disk_read, prev_disk_write = read_diskstats()
    start = time.time()
    prev_time = start

    while not stop_event.is_set():
        time.sleep(interval)
        now = time.time()
        dt = now - prev_time
        prev_time = now

        curr_stat = read_proc_stat()
        pcts = cpu_pct(prev_stat, curr_stat)
        prev_stat = curr_stat

        curr_read, curr_write = read_diskstats()
        read_mb_s = round((curr_read - prev_disk_read) * 512 / (1024 * 1024 * dt), 2) if dt > 0 else 0
        write_mb_s = round((curr_write - prev_disk_write) * 512 / (1024 * 1024 * dt), 2) if dt > 0 else 0
        prev_disk_read, prev_disk_write = curr_read, curr_write

        n_cores_active = sum(1 for k, v in pcts.items() if k != "cpu" and v > 3)
        n_cores = sum(1 for k in pcts if k != "cpu")
        total_cpu = pcts.get("cpu", 0)
        n_containers = get_n_containers(runtime)
        n_procs, proc_cats = count_enroot_procs(runtime)

        samples.append(
            {
                "t": round(now - start, 2),
                "cpu_total": total_cpu,
                "cores_active": n_cores_active,
                "n_cores": n_cores,
                "n_containers": n_containers,
                "n_procs": n_procs,
                "proc_cats": proc_cats,
                "disk_read_mb_s": read_mb_s,
                "disk_write_mb_s": write_mb_s,
            }
        )

        if not quiet:
            sys.stderr.write(
                f"\r  [{samples[-1]['t']:5.1f}s] "
                f"CPU:{total_cpu:5.1f}% "
                f"cores:{n_cores_active}/{n_cores} "
                f"{runtime}:{n_containers} "
                f"procs:{n_procs} "
                f"disk:R{read_mb_s:.0f}/W{write_mb_s:.0f}MB/s  "
            )
            sys.stderr.flush()

    return samples

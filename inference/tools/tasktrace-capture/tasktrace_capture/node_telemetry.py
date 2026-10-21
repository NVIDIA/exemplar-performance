# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw node telemetry recorder (memory / storage / IO / pressure).

Complements :mod:`tasktrace_capture.monitor` — that module samples *aggregate* CPU at
0.5s for the live status line and scaling metrics; this one appends *raw
counters* to ``telemetry.raw`` at a slow cadence (default 5s) for post-hoc
analysis across an entire sweep:

- per-CPU jiffies from ``/proc/stat`` (per-vCPU utilization, e.g. verifying
  one-sandbox-per-thread pinning under oversubscription),
- PSI (``/proc/pressure/{cpu,io,memory}``) — kernel stall accounting,
- per-device ``/proc/diskstats`` rows,
- **storage occupancy** via statvfs for the mounts tasktrace-capture writes to
  (output dir, /tmp, enroot data/cache incl. their XDG defaults, docker
  data root, apptainer runtime root), re-resolved every sample so paths
  created mid-sweep are picked up — this is how peak-SSD usage is measured:
  container rootfs extraction is the disk ceiling at high concurrency
  (352 concurrent SWE-bench sandboxes need ~1.8TB),
- physical NICs' ``/proc/net/dev`` counters and ``MemAvailable``.

Format is line-oriented plain text; the STORAGE-INVENTORY header keys are kept
byte-identical to the original sampler-v2 format so exact-string parsers keep
working. Failures never
propagate: whole samples are dropped and counted (``dropped_samples``),
never torn.
"""

import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from .monitor import DISK_DEVICE_PREFIXES, _run

DEFAULT_INTERVAL_SEC = 5.0
# Below this the sampler's own /proc reads and writes start contaminating
# the measurements tasktrace-capture exists to take (interval 0 measured at ~3000
# samples/s). Enforced here so the CLI, YAML config, and direct construction
# all share one floor.
MIN_INTERVAL_SEC = 0.5

# monitor.py's whole-run totals track plain disks; the raw stream also wants
# RAID/device-mapper rows so md/dm hosts can attribute IO post-hoc.
_TELEMETRY_DISK_PREFIXES = DISK_DEVICE_PREFIXES + ("md", "dm-")

_DISKSTATS_HEADER = (
    "major minor name reads rd_merged rd_sectors rd_ms wr wr_merged "
    "wr_sectors wr_ms io_inflight io_ms io_weighted_ms discards dsc_merged "
    "dsc_sectors dsc_ms flush flush_ms"
)


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _physical_nics():
    """Interfaces backed by real hardware (a /sys device link), not down.

    Virtual interfaces (docker0, br-*, veth*) have no ``device`` link, so
    they never shadow the uplink. IPoIB and some bonds report operstate
    "unknown" while carrying traffic, so only "down" is excluded.
    """
    nics = []
    try:
        for name in sorted(os.listdir("/sys/class/net")):
            if name == "lo":
                continue
            if not os.path.exists(f"/sys/class/net/{name}/device"):
                continue
            if _read(f"/sys/class/net/{name}/operstate").strip() == "down":
                continue
            nics.append(name)
    except OSError:
        pass
    return nics


def _docker_data_root():
    """Docker's data root from the daemon (rootless/daemon.json aware);
    resolved once, cached, falls back to the stock path."""
    out = _run(["docker", "info", "-f", "{{.DockerRootDir}}"])
    root = (out or "").strip()
    return root or "/var/lib/docker"


def _apptainer_runtime_root():
    """Apptainer's per-trial writable storage root, from the owning module
    (expanduser + tempdir fallback semantics live there, not here)."""
    try:
        from tasktrace_capture.apptainer.config import apptainer_runtime_root

        return str(apptainer_runtime_root())
    except Exception:
        return os.environ.get("TASKTRACE_CAPTURE_APPTAINER_TMPDIR")


def _occ_candidates(output_dir, runtime=None, _docker_root_cache=None):
    """(label, path) candidates for occupancy gauges — everywhere tasktrace-capture
    writes big things. Paths may not exist yet (enroot creates its dirs on
    first import), so callers re-resolve every sample. Never raises."""
    if _docker_root_cache is None:
        _docker_root_cache = []
    try:
        home = Path.home()
    except Exception:
        home = Path("/")
    xdg_data = os.environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    xdg_cache = os.environ.get("XDG_CACHE_HOME") or str(home / ".cache")
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = None
    candidates = [
        ("output", str(output_dir)),
        ("tmp", "/tmp"),
        # enroot's defaults when the env vars are unset (enroot.conf(1)):
        # rootfs extraction is the biggest disk consumer of an enroot sweep.
        ("enroot_data", os.environ.get("ENROOT_DATA_PATH") or f"{xdg_data}/enroot"),
        ("enroot_cache", os.environ.get("ENROOT_CACHE_PATH") or f"{xdg_cache}/enroot"),
        # runner.py exports this default into harbor's env; mirror it here
        # because the tasktrace-capture process env usually doesn't carry it.
        ("harbor_cache", os.environ.get("HARBOR_ENROOT_CACHE") or (str(cwd / ".enroot" / "cache") if cwd else None)),
    ]
    if runtime == "docker":
        if not _docker_root_cache:
            _docker_root_cache.append(_docker_data_root())
        candidates.append(("docker_root", _docker_root_cache[0]))
    if runtime == "apptainer":
        candidates.append(("apptainer_root", _apptainer_runtime_root()))
    return [(label, path) for label, path in candidates if path]


class NodeTelemetryRecorder:
    """Background thread appending raw node telemetry to ``telemetry.raw``.

    Overhead is a handful of /proc reads and one unbuffered write per
    interval (~8 KB/sample on a 192-thread node → ~1.7 KB/s at the default
    cadence). A recorder that cannot open its file prints one warning and
    exits its thread. Not restartable: create a new instance after ``stop()``.
    """

    def __init__(self, output_dir, interval_sec=DEFAULT_INTERVAL_SEC, runtime=None):
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "telemetry.raw"
        try:
            interval_sec = float(interval_sec)
        except (TypeError, ValueError):
            interval_sec = DEFAULT_INTERVAL_SEC
        if not math.isfinite(interval_sec) or interval_sec <= 0:
            interval_sec = DEFAULT_INTERVAL_SEC
        self.interval_sec = max(interval_sec, MIN_INTERVAL_SEC)
        self.runtime = runtime
        self._stop = threading.Event()
        self._thread = None
        self._dropped = 0
        self._dev_labels = {}  # st_dev -> pinned label (stable series)
        self._join_failed = False

    # -- occupancy resolution (label pinned per device for the run) --------

    def _occ_mounts(self):
        picked = []
        for label, path in _occ_candidates(self.output_dir, self.runtime):
            try:
                dev = os.stat(path).st_dev
            except OSError:
                continue
            pinned = self._dev_labels.get(dev)
            if pinned is None:
                self._dev_labels[dev] = label
                pinned = label
            if pinned == label:
                picked.append((label, path))
        return picked

    # -- record builders (return text; writing happens in one call) --------

    def _stamp(self, kind):
        return f"=== {kind} {time.time():.6f} mono={time.monotonic():.3f}\n"

    def _build_inventory(self):
        # Header keys below ("-- sector_bytes:", "-- mounts(...):", ...) are
        # kept byte-identical to sampler-v2; exact-string parsers depend on
        # them. New information goes on separate "-- note:" / occ_candidate
        # lines, never into renamed keys.
        out = [self._stamp("STORAGE-INVENTORY")]
        out.append(f"-- root_workdir: {self.output_dir}\n")
        df = _run(["df", "-P", str(self.output_dir)]) or ""
        lines = df.strip().splitlines()
        if len(lines) >= 2:
            out.append(f"-- backing_mount: {lines[-1]}\n")
        out.append("-- lsblk:\n")
        out.append(_run(["lsblk", "-b", "-o", "NAME,TYPE,SIZE,MOUNTPOINT,ROTA,MODEL"]) or "")
        out.append("\n-- mounts(nvme|md|dm|tmp|local):\n")
        for line in _read("/proc/mounts").splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            dev, mountpoint = fields[0], fields[1]
            devname = dev.rsplit("/", 1)[-1]
            if (
                (dev.startswith("/dev/") and (devname.startswith(_TELEMETRY_DISK_PREFIXES) or "/mapper/" in dev))
                or mountpoint == "/tmp"
                or "/localhome" in mountpoint
            ):
                out.append(line + "\n")
        out.append(f"-- diskstats_header: {_DISKSTATS_HEADER}\n")
        out.append(
            "-- note: diskstats sector counters are always 512-byte units; "
            "sector_bytes below is each device's FORMAT size, informational only\n"
        )
        out.append("-- sector_bytes:\n")
        try:
            for dev in sorted(os.listdir("/sys/block")):
                size = _read(f"/sys/block/{dev}/queue/hw_sector_size").strip()
                if size:
                    out.append(f"{dev} {size}\n")
        except OSError:
            pass
        out.append("-- occ_candidates (label -> path; resolved per sample, " "gauges appear once paths exist):\n")
        for label, path in _occ_candidates(self.output_dir, self.runtime):
            out.append(f"occ_candidate {label} {path}\n")
        out.append(f"-- nics: {' '.join(_physical_nics()) or '(none)'}\n")
        out.append(f"-- interval_sec: {self.interval_sec:g}\n")
        return "".join(out)

    def _build_sample(self, carried_drops):
        out = [self._stamp("SAMPLE")]
        if carried_drops:
            out.append(f"dropped_samples {carried_drops}\n")
        # per-CPU jiffies (skip the aggregate "cpu " line)
        for line in _read("/proc/stat").splitlines():
            if line.startswith("cpu") and not line.startswith("cpu "):
                out.append(line + "\n")
        for kind in ("cpu", "io", "memory"):
            content = _read(f"/proc/pressure/{kind}")
            if content:
                out.append(f"psi_{kind} {content.replace(chr(10), '|')}\n")
        for line in _read("/proc/diskstats").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2].startswith(_TELEMETRY_DISK_PREFIXES):
                out.append(line + "\n")
        for label, path in self._occ_mounts():
            try:
                usage = shutil.disk_usage(path)
                out.append(f"occ_{label} {usage.total} {usage.used} {usage.free}\n")
            except OSError:
                continue
        nics = set(_physical_nics())
        if nics:
            for line in _read("/proc/net/dev").splitlines():
                name = line.split(":", 1)[0].strip()
                if ":" in line and name in nics:
                    out.append(f"net_{name} {line.strip()}\n")
        for line in _read("/proc/meminfo").splitlines():
            if line.startswith("MemAvailable"):
                out.append(f"memavail {line}\n")
                break
        return "".join(out)

    # -- lifecycle ----------------------------------------------------------

    def _emit_sample(self, f):
        """Build + single unbuffered write of one sample. The drop counter is
        cleared only AFTER a successful write, so consecutive failures
        accumulate and are reported in the next sample that lands."""
        carried = self._dropped
        try:
            f.write(self._build_sample(carried).encode())
            self._dropped -= carried
            return True
        except Exception:
            self._dropped += 1
            return False

    def _loop(self):
        try:
            # binary + unbuffered: each record is one write(2); a failed
            # record is dropped whole, never retained in a userspace buffer
            # to be delivered late after recovery.
            f = open(self.path, "ab", buffering=0)
        except OSError as exc:
            print(f"  node telemetry disabled: cannot open {self.path}: {exc}", file=sys.stderr)
            return
        with f:
            try:
                f.write(self._build_inventory().encode())
            except Exception:
                self._dropped += 1
            self._emit_sample(f)  # t=0 sample
            while not self._stop.wait(self.interval_sec):
                self._emit_sample(f)
            self._emit_sample(f)  # final sample at stop

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name="node-telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=30):
        """Signal the thread and wait for the final sample. Returns True if
        the thread exited; False if it is still stuck (e.g. statvfs/df hung
        on a dead network mount) — it is a daemon thread, so a stuck sampler
        never blocks process exit, at the cost of the trailing record.
        Safe to call repeatedly (atexit + explicit): a repeat call after a
        failed join does not wait again."""
        self._stop.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=0 if self._join_failed else timeout)
        if self._thread.is_alive():
            if not self._join_failed:
                print(
                    "  node telemetry: sampler did not exit within "
                    f"{timeout}s (hung mount?); final sample may be missing",
                    file=sys.stderr,
                )
            self._join_failed = True
            return False
        return True

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the raw node telemetry recorder."""

import time

from tasktrace_capture.node_telemetry import (
    DEFAULT_INTERVAL_SEC,
    MIN_INTERVAL_SEC,
    NodeTelemetryRecorder,
    _occ_candidates,
)


def _wait_for(predicate, timeout=15.0, poll=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


class TestNodeTelemetryRecorder:
    def test_records_inventory_and_samples(self, tmp_path):
        rec = NodeTelemetryRecorder(tmp_path, interval_sec=MIN_INTERVAL_SEC).start()
        try:
            assert _wait_for(
                lambda: rec.path.exists() and rec.path.read_text().count("=== SAMPLE") >= 2
            ), "expected at least 2 samples"
        finally:
            rec.stop()
        text = rec.path.read_text()

        # one inventory header, at the top
        assert text.startswith("=== STORAGE-INVENTORY ")
        assert text.count("=== STORAGE-INVENTORY") == 1
        assert "-- diskstats_header:" in text
        # sampler-v2 byte-compatible header keys
        assert "-- sector_bytes:" in text
        assert "-- mounts(nvme|md|dm|tmp|local):" in text
        # candidate map uses its own prefix so it can never collide with
        # occ_* gauge lines (the doc's awk one-liner depends on this)
        assert "occ_candidate output " in text

        # sample content: per-CPU jiffies but never the aggregate line
        sample = text.split("=== SAMPLE", 1)[1]
        assert any(line.startswith("cpu0 ") for line in sample.splitlines())
        assert not any(line.startswith("cpu ") for line in sample.splitlines())

        # occupancy gauges are parseable: label + 3 integer byte counts
        occ_lines = [line for line in sample.splitlines() if line.startswith("occ_")]
        assert occ_lines, "expected occ_* lines in samples"
        for line in occ_lines:
            parts = line.split()
            assert len(parts) == 4
            total, used, avail = (int(x) for x in parts[1:])
            assert total > 0 and 0 <= used <= total and 0 <= avail <= total

        # memavail present and parseable
        mem = [line for line in sample.splitlines() if line.startswith("memavail")]
        assert mem and "MemAvailable" in mem[0]

    def test_stop_is_idempotent_and_start_once(self, tmp_path):
        rec = NodeTelemetryRecorder(tmp_path, interval_sec=MIN_INTERVAL_SEC)
        assert rec.start() is rec
        first_thread = rec._thread
        assert rec.start() is rec  # second start is a no-op
        assert rec._thread is first_thread
        rec.stop()
        rec.stop()  # double-stop must not raise
        assert not rec._thread.is_alive()

    def test_occ_mounts_dedupes_by_device(self, tmp_path, monkeypatch):
        # output dir and an env-pointed dir on the same filesystem must
        # collapse to one tracked mount (first label wins)
        monkeypatch.setenv("ENROOT_DATA_PATH", str(tmp_path))
        monkeypatch.delenv("ENROOT_CACHE_PATH", raising=False)
        monkeypatch.delenv("HARBOR_ENROOT_CACHE", raising=False)
        rec = NodeTelemetryRecorder(tmp_path)
        mounts = rec._occ_mounts()
        labels = [label for label, _ in mounts]
        assert labels[0] == "output"
        assert "enroot_data" not in labels
        import os

        devs = [os.stat(path).st_dev for _, path in mounts]
        assert len(devs) == len(set(devs))

    def test_enroot_defaults_tracked_when_env_unset(self, tmp_path, monkeypatch):
        # enroot's rootfs extraction default (~/.local/share/enroot) is the
        # biggest disk consumer of a default run: it must be a candidate even
        # with no ENROOT_* env set
        monkeypatch.delenv("ENROOT_DATA_PATH", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        labels = dict(_occ_candidates(tmp_path))
        assert labels["enroot_data"].endswith("/.local/share/enroot")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        labels = dict(_occ_candidates(tmp_path))
        assert labels["enroot_data"] == str(tmp_path / "xdg" / "enroot")

    def test_device_label_pinned_for_run(self, tmp_path, monkeypatch):
        # once a device is being reported under one label, a higher-priority
        # candidate appearing later on the same device must NOT steal it —
        # per-label time series stay contiguous
        late = tmp_path / "late-enroot-data"
        monkeypatch.setenv("ENROOT_DATA_PATH", str(late))
        monkeypatch.delenv("ENROOT_CACHE_PATH", raising=False)
        monkeypatch.delenv("HARBOR_ENROOT_CACHE", raising=False)
        rec = NodeTelemetryRecorder(tmp_path)
        first = [label for label, _ in rec._occ_mounts()]
        assert "enroot_data" not in first  # path doesn't exist yet
        late.mkdir()  # same device as output
        second = [label for label, _ in rec._occ_mounts()]
        assert second == first  # pinned: no label switch

    def test_occ_candidates_survive_deleted_cwd(self, tmp_path, monkeypatch):
        import tasktrace_capture.node_telemetry as nt

        def boom():
            raise FileNotFoundError("cwd gone")

        monkeypatch.setattr(nt.Path, "cwd", staticmethod(boom))
        monkeypatch.delenv("HARBOR_ENROOT_CACHE", raising=False)
        cands = _occ_candidates(tmp_path)  # must not raise
        assert ("output", str(tmp_path)) in cands
        assert all(label != "harbor_cache" for label, _ in cands)

    def test_first_and_final_samples(self, tmp_path):
        # a run much shorter than the interval still yields >= 2 samples:
        # one immediately at start, one at stop
        rec = NodeTelemetryRecorder(tmp_path, interval_sec=30).start()
        try:
            assert _wait_for(
                lambda: rec.path.exists() and rec.path.read_text().count("=== SAMPLE") >= 1
            ), "expected the t=0 sample well before the 30s interval"
        finally:
            rec.stop()
        assert rec.path.read_text().count("=== SAMPLE") >= 2

    def test_invalid_interval_clamped_to_default(self, tmp_path):
        for bad in (0, -3, float("nan"), float("inf"), "x", None):
            rec = NodeTelemetryRecorder(tmp_path, interval_sec=bad)
            assert rec.interval_sec == DEFAULT_INTERVAL_SEC, bad
        # numeric strings coerce; sub-floor values clamp to the shared floor
        assert NodeTelemetryRecorder(tmp_path, interval_sec="2").interval_sec == 2.0
        assert NodeTelemetryRecorder(tmp_path, interval_sec=0.01).interval_sec == MIN_INTERVAL_SEC

    def test_dropped_counter_accumulates_until_success(self, tmp_path):
        # consecutive write failures must accumulate, and only a SUCCESSFUL
        # write clears the counter (regression: reset-before-write collapsed
        # any number of losses to 'dropped_samples 1')
        rec = NodeTelemetryRecorder(tmp_path)

        class FailingFile:
            def write(self, data):
                raise OSError(28, "No space left on device")

        for _ in range(5):
            assert rec._emit_sample(FailingFile()) is False
        assert rec._dropped == 5

        captured = []

        class OkFile:
            def write(self, data):
                captured.append(data)

        assert rec._emit_sample(OkFile()) is True
        assert rec._dropped == 0
        assert b"dropped_samples 5\n" in captured[0]

    def test_sample_lines_carry_monotonic(self, tmp_path):
        rec = NodeTelemetryRecorder(tmp_path, interval_sec=MIN_INTERVAL_SEC).start()
        try:
            assert _wait_for(lambda: rec.path.exists() and "=== SAMPLE" in rec.path.read_text())
        finally:
            rec.stop()
        header = [line for line in rec.path.read_text().splitlines() if line.startswith("=== SAMPLE")][0]
        parts = header.split()
        float(parts[2])  # epoch parses
        assert parts[3].startswith("mono=")
        float(parts[3][5:])

    def test_dropped_samples_are_counted(self, tmp_path):
        rec = NodeTelemetryRecorder(tmp_path, interval_sec=MIN_INTERVAL_SEC)
        rec._dropped = 3  # simulate prior write failures
        rec.start()
        try:
            assert _wait_for(lambda: rec.path.exists() and "dropped_samples 3" in rec.path.read_text())
        finally:
            rec.stop()

    def test_unopenable_path_disables_quietly(self, tmp_path):
        import os

        import pytest

        if os.geteuid() == 0:
            pytest.skip("mode 0o555 does not block root (CAP_DAC_OVERRIDE)")
        blocked = tmp_path / "blocked"
        blocked.mkdir(mode=0o555)
        rec = NodeTelemetryRecorder(blocked).start()
        try:
            assert _wait_for(lambda: not rec._thread.is_alive())
            assert not (blocked / "telemetry.raw").exists()
        finally:
            rec.stop()

    def test_recorder_survives_unreadable_mount(self, tmp_path, monkeypatch):
        from tasktrace_capture import node_telemetry

        monkeypatch.setattr(
            node_telemetry,
            "_occ_candidates",
            lambda out, runtime=None: [
                ("output", str(tmp_path)),
                ("gone", str(tmp_path / "does-not-exist")),
            ],
        )
        rec = NodeTelemetryRecorder(tmp_path, interval_sec=MIN_INTERVAL_SEC)
        rec.start()
        try:
            assert _wait_for(lambda: rec.path.exists() and rec.path.read_text().count("=== SAMPLE") >= 1)
        finally:
            rec.stop()
        # the bad mount appears in the inventory map but is skipped in
        # samples (disk_usage raises OSError -> line omitted)
        text = rec.path.read_text()
        samples = text.split("=== SAMPLE", 1)[1]
        assert "occ_gone" not in samples
        assert "occ_output" in samples

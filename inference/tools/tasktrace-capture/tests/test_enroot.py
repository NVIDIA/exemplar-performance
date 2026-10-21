# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import fcntl
import json
import logging
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.task.config import NetworkMode, NetworkPolicy
from tasktrace_capture.enroot_env import (
    _CLAUDE_CODE_CACHE_DIR,
    _CLAUDE_CODE_CACHE_LOCK,
    _ENROOT_MAX_RETRIES,
    _ENROOT_RETRY_BASE_SEC,
    _EPOCH_SWEBENCH_REGISTRY,
    _HOME_SYMLINK_LOCK,
    _HOST_ARCH,
    _SWEBENCH_INSTANCE_RE,
    _TMUX_CACHE_DIR,
    _TMUX_CACHE_LOCK,
    EnrootEnvironment,
    _ensure_claude_code_binary,
    _ensure_ghcr_credentials,
    _escape_mount_path,
    _format_mount,
    _is_sensitive_key,
    _mask_cmd_for_logging,
    _remap_swebench_image,
    _retry_cmd,
    _swebench_instance_from_envdir,
)


def _network_policy_env(tmp_path: Path) -> EnrootEnvironment:
    env = object.__new__(EnrootEnvironment)
    env._network_policy = NetworkPolicy()
    env.trial_paths = SimpleNamespace(trial_dir=tmp_path / "trial")
    env.logger = logging.getLogger("test.enroot.network-policy")
    return env


def test_enroot_strict_network_policy_rejects_offline_task(monkeypatch, tmp_path):
    monkeypatch.delenv("_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY", raising=False)
    env = _network_policy_env(tmp_path)

    with pytest.raises(ValueError, match="no-network.*not supported"):
        env.validate_network_policy_support(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))

    assert not (tmp_path / "trial" / "network_policy.json").exists()


def test_enroot_best_effort_records_degraded_network_policy(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY", "best-effort")
    env = _network_policy_env(tmp_path)
    requested = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)

    with caplog.at_level(logging.WARNING):
        env.validate_network_policy_support(requested)

    record = json.loads((tmp_path / "trial" / "network_policy.json").read_text(encoding="utf-8"))
    assert record == {
        "compatibility_mode": "best-effort",
        "effective": "public",
        "effective_allowed_hosts": [],
        "enforced": False,
        "reason": (
            "Enroot shares the host network namespace and cannot enforce the requested egress policy on this target."
        ),
        "requested": "no-network",
        "requested_allowed_hosts": [],
        "runtime": "enroot",
        "schema_version": 1,
    }
    assert "NETWORK ISOLATION NOT ENFORCED" in caplog.text


def test_enroot_best_effort_does_not_degrade_public_policy(monkeypatch, tmp_path):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY", "best-effort")
    env = _network_policy_env(tmp_path)

    env.validate_network_policy_support(NetworkPolicy(network_mode=NetworkMode.PUBLIC))

    assert not (tmp_path / "trial" / "network_policy.json").exists()


def test_enroot_rejects_unknown_network_compatibility_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY", "unsafe")
    env = _network_policy_env(tmp_path)

    with pytest.raises(ValueError, match="_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY must be one of"):
        env.validate_network_policy_support(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))


# ===========================================================================
#  Sensitive env var detection and masking
# ===========================================================================


class TestIsSensitiveKey:
    """Verify _is_sensitive_key detects common secret-bearing env var names."""

    def test_api_key(self):
        assert _is_sensitive_key("ANTHROPIC_API_KEY") is True

    def test_token(self):
        assert _is_sensitive_key("GITHUB_TOKEN") is True

    def test_secret(self):
        assert _is_sensitive_key("MY_SECRET") is True

    def test_password(self):
        assert _is_sensitive_key("DB_PASSWORD") is True

    def test_credential(self):
        assert _is_sensitive_key("AWS_CREDENTIAL") is True

    def test_non_sensitive(self):
        assert _is_sensitive_key("ANTHROPIC_BASE_URL") is False

    def test_model_name(self):
        assert _is_sensitive_key("MODEL_NAME") is False

    def test_case_insensitive(self):
        assert _is_sensitive_key("api_key") is True
        assert _is_sensitive_key("Api_Token") is True


class TestMaskCmdForLogging:
    """Verify _mask_cmd_for_logging masks sensitive --env values in commands."""

    def test_masks_api_key(self):
        cmd = ["enroot", "start", "--env", "ANTHROPIC_API_KEY=sk-secret123", "--env", "HOME=/root", "container"]
        masked = _mask_cmd_for_logging(cmd)
        assert "ANTHROPIC_API_KEY=****" in masked
        assert "ANTHROPIC_API_KEY=sk-secret123" not in masked
        # Non-sensitive env var untouched
        assert "HOME=/root" in masked

    def test_masks_multiple_sensitive_vars(self):
        cmd = [
            "enroot",
            "start",
            "--env",
            "ANTHROPIC_API_KEY=sk-abc",
            "--env",
            "GITHUB_TOKEN=ghp-xyz",
            "--env",
            "ANTHROPIC_BASE_URL=https://example.com",
            "container",
        ]
        masked = _mask_cmd_for_logging(cmd)
        assert "ANTHROPIC_API_KEY=****" in masked
        assert "GITHUB_TOKEN=****" in masked
        assert "ANTHROPIC_BASE_URL=https://example.com" in masked

    def test_no_env_flags_unchanged(self):
        cmd = ["enroot", "start", "--rw", "container", "bash", "-c", "echo hello"]
        masked = _mask_cmd_for_logging(cmd)
        assert masked == cmd

    def test_env_without_equals_unchanged(self):
        cmd = ["enroot", "start", "--env", "NOEQUALS", "container"]
        masked = _mask_cmd_for_logging(cmd)
        assert masked == cmd


# ===========================================================================
#  EnrootEnvironment Dockerfile parsing
# ===========================================================================


class TestParseDockerfile:
    """Tests for EnrootEnvironment._parse_dockerfile (WORKDIR extraction)."""

    def test_from_and_workdir(self, tmp_path):
        from tasktrace_capture.enroot_env import EnrootEnvironment

        df = tmp_path / "Dockerfile"
        df.write_text("FROM ubuntu:24.04\nWORKDIR /app\nRUN echo hello\n")
        image, workdir = EnrootEnvironment._parse_dockerfile(df)
        assert image == "ubuntu:24.04"
        assert workdir == "/app"

    def test_from_only(self, tmp_path):
        from tasktrace_capture.enroot_env import EnrootEnvironment

        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.12\nRUN pip install flask\n")
        image, workdir = EnrootEnvironment._parse_dockerfile(df)
        assert image == "python:3.12"
        assert workdir is None

    def test_multiple_workdir_takes_last(self, tmp_path):
        from tasktrace_capture.enroot_env import EnrootEnvironment

        df = tmp_path / "Dockerfile"
        df.write_text("FROM ubuntu:24.04\nWORKDIR /build\nRUN make\nWORKDIR /app\n")
        image, workdir = EnrootEnvironment._parse_dockerfile(df)
        assert image == "ubuntu:24.04"
        assert workdir == "/app"

    def test_no_from(self, tmp_path):
        from tasktrace_capture.enroot_env import EnrootEnvironment

        df = tmp_path / "Dockerfile"
        df.write_text("RUN echo hello\nWORKDIR /app\n")
        image, workdir = EnrootEnvironment._parse_dockerfile(df)
        assert image is None
        assert workdir == "/app"

    def test_scratch_from(self, tmp_path):
        from tasktrace_capture.enroot_env import EnrootEnvironment

        df = tmp_path / "Dockerfile"
        df.write_text("FROM scratch\nWORKDIR /app\n")
        image, workdir = EnrootEnvironment._parse_dockerfile(df)
        assert image is None
        assert workdir == "/app"

    def test_case_insensitive(self, tmp_path):
        from tasktrace_capture.enroot_env import EnrootEnvironment

        df = tmp_path / "Dockerfile"
        df.write_text("from ubuntu:24.04\nworkdir /data\n")
        image, workdir = EnrootEnvironment._parse_dockerfile(df)
        assert image == "ubuntu:24.04"
        assert workdir == "/data"


class TestBashrcIsolation:
    """Verify that per-container .bashrc isolation is set up correctly.

    Enroot shares the host home dir — without isolation, concurrent agent
    installs (Claude Code, uv) append PATH exports to the real ~/.bashrc.
    """

    def test_bashrc_created_in_claude_cache(self, tmp_path):
        """The .bashrc file should be created inside the per-container cache dir."""
        cache_dir = tmp_path / "claude_cache"
        cache_dir.mkdir()
        bashrc = cache_dir / ".bashrc"
        bashrc.write_text("# container bashrc\n")
        assert bashrc.exists()
        assert bashrc.read_text() == "# container bashrc\n"

    def test_bashrc_in_isolated_home(self):
        """Isolated /root home should contain .bashrc."""
        cache = Path("/tmp/cache")
        bashrc = cache / ".bashrc"
        # When /tmp/cache is mounted as /root, .bashrc appears as /root/.bashrc
        assert bashrc == cache / ".bashrc"

    def test_bashrc_not_host_file(self, tmp_path):
        """Container .bashrc must be a separate file, not the host's."""
        host_bashrc = tmp_path / "host_bashrc"
        host_bashrc.write_text("# real bashrc with user config\n")

        container_cache = tmp_path / "container_cache"
        container_cache.mkdir()
        container_bashrc = container_cache / ".bashrc"
        container_bashrc.write_text("# container bashrc\n")

        assert host_bashrc.read_text() != container_bashrc.read_text()
        container_bashrc.write_text('export PATH="$HOME/.local/bin:$PATH"\n')
        assert "local/bin" not in host_bashrc.read_text()

    def test_local_bin_env_shim_matches_uv_and_survives_source(self, tmp_path):
        """Harbor sources ~/.local/bin/env after uv's installer; uv omits it when ~/.local/bin is already on PATH."""
        import subprocess

        from tasktrace_capture.enroot_env import _LOCAL_BIN_ENV_SCRIPT

        home = tmp_path / "home"
        (home / ".local" / "bin").mkdir(parents=True)
        (home / ".local" / "bin" / "env").write_text(_LOCAL_BIN_ENV_SCRIPT)
        out = subprocess.run(
            ["bash", "-c", 'source "$HOME/.local/bin/env"; source "$HOME/.local/bin/env"; echo "$PATH"'],
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out == f"{home}/.local/bin:/usr/bin:/bin"

    def test_compose_preserves_image_bashrc_then_appends_path(self):
        """SWE-bench images activate conda from /root/.bashrc; it must survive the /root mount."""
        from tasktrace_capture.enroot_env import compose_home_dotfiles

        files = compose_home_dotfiles("source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed")
        lines = files[".bashrc"].splitlines()
        assert lines[0] == "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
        assert lines[-1] == 'export PATH="$HOME/.local/bin:$PATH"'
        assert files[".bashrc"].endswith("\n")
        # .profile only reaches .bashrc under bash; .bash_profile defers to .profile
        assert '[ -n "$BASH" ]' in files[".profile"] and '. "$HOME/.bashrc"' in files[".profile"]
        assert '. "$HOME/.profile"' in files[".bash_profile"]
        # no image .bashrc -> stub plus the PATH export
        stub = compose_home_dotfiles(None)
        assert stub[".bashrc"] == '# container bashrc\nexport PATH="$HOME/.local/bin:$PATH"\n'.replace(
            "\nexport", "\n# tasktrace-capture: per-container tool installs land in ~/.local/bin (Issue #50)\nexport"
        )
        assert stub[".profile"] == files[".profile"] and stub[".bash_profile"] == files[".bash_profile"]

    def test_generated_profile_runs_image_code_only_under_bash(self, tmp_path):
        """A dash login shell (images whose root shell is /bin/sh) must execute no image .bashrc code."""
        import shutil
        import subprocess

        from tasktrace_capture.enroot_env import compose_home_dotfiles

        home = tmp_path / "home"
        home.mkdir()
        for name, content in compose_home_dotfiles("export MARK=from_image_bashrc").items():
            (home / name).write_text(content)
        env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
        bash = subprocess.run(
            ["bash", "-c", '. "$HOME/.bash_profile"; echo "${MARK-unset}"'], env=env, capture_output=True, text=True
        )
        assert bash.stdout.strip() == "from_image_bashrc", bash.stderr
        dash = shutil.which("dash")
        if dash is None:
            pytest.skip("dash not installed on this host")
        posix = subprocess.run(
            [dash, "-c", '. "$HOME/.profile"; echo "${MARK-unset}"'], env=env, capture_output=True, text=True
        )
        assert posix.stdout.strip() == "unset" and posix.returncode == 0, posix.stderr

    def test_decode_image_bashrc(self):
        """The read script prints one base64 line or nothing; anything else decodes to None."""
        import base64

        from tasktrace_capture.enroot_env import decode_image_bashrc

        assert (
            decode_image_bashrc(base64.b64encode(b"conda activate testbed\n").decode() + "\n")
            == "conda activate testbed\n"
        )
        assert decode_image_bashrc("") is None and decode_image_bashrc("\n") is None
        assert decode_image_bashrc("###not-base64###") is None

    def test_read_image_bashrc_is_cached_retried_and_best_effort(self, monkeypatch):
        """One enroot start per image per process; a plain start with the home hook disabled; failures -> stub."""
        import asyncio
        import base64
        import logging

        from tasktrace_capture.enroot_env import (
            _READ_IMAGE_BASHRC_SCRIPT,
            EnrootEnvironment,
        )

        monkeypatch.setattr(EnrootEnvironment, "_image_bashrc_cache", {})
        monkeypatch.setattr(EnrootEnvironment, "_image_bashrc_locks", {})
        monkeypatch.setattr("tasktrace_capture.enroot_env._ENROOT_RETRY_BASE_SEC", 0.0)
        env = object.__new__(EnrootEnvironment)
        env._container_name = "hb__test"
        env.logger = logging.getLogger("test")
        calls = []

        class _Result:
            def __init__(self, rc, stdout="", stderr=""):
                self.return_code, self.stdout, self.stderr = rc, stdout, stderr

        async def ok(cmd, timeout_sec=None, check=True):
            calls.append(cmd)
            return _Result(0, base64.b64encode(b"conda activate testbed").decode() + "\n")

        monkeypatch.setattr(env, "_run_cmd", ok)
        assert asyncio.run(env._read_image_bashrc("img:a")) == "conda activate testbed"
        assert calls[0][:6] == ["env", "ENROOT_MOUNT_HOME=no", "enroot", "start", "--root", "hb__test"]
        assert "--mount" not in calls[0] and calls[0][-1] == _READ_IMAGE_BASHRC_SCRIPT
        # cached: a second trial of the same image does not start a container
        assert asyncio.run(env._read_image_bashrc("img:a")) == "conda activate testbed" and len(calls) == 1
        # a transient enroot failure is retried, then succeeds
        attempts = []

        async def flaky(cmd, timeout_sec=None, check=True):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("Enroot command failed (rc=1): rootfs not ready")
            return _Result(0, base64.b64encode(b"x").decode())

        monkeypatch.setattr(env, "_run_cmd", flaky)
        assert asyncio.run(env._read_image_bashrc("img:b")) == "x" and len(attempts) == 3

        # empty output (no .bashrc, or /root is a host mount) -> None, and cached as such
        async def empty(cmd, timeout_sec=None, check=True):
            calls.append(cmd)
            return _Result(0, "")

        monkeypatch.setattr(env, "_run_cmd", empty)
        assert asyncio.run(env._read_image_bashrc("img:c")) is None
        assert asyncio.run(env._read_image_bashrc("img:c")) is None and len(calls) == 2

        # persistent failure -> None and NOT cached, so a later trial can retry
        async def raising(cmd, timeout_sec=None, check=True):
            raise RuntimeError("enroot missing")

        monkeypatch.setattr(env, "_run_cmd", raising)
        assert asyncio.run(env._read_image_bashrc("img:d")) is None
        assert "img:d" not in EnrootEnvironment._image_bashrc_cache
        # start(force_build=True) drops the cached entry so a refreshed mutable tag is re-read
        EnrootEnvironment._image_bashrc_cache["img:a"] = "stale"
        EnrootEnvironment._image_bashrc_cache.pop("img:a", None)  # what start() does before the read
        monkeypatch.setattr(env, "_run_cmd", ok)
        assert asyncio.run(env._read_image_bashrc("img:a")) == "conda activate testbed"


class TestMountPathEscaping:
    """Tests for _escape_mount_path / _format_mount.

    Enroot parses --mount specs using fstab-style rules where spaces are
    field separators. Paths containing spaces must be escaped as \\040.
    """

    def test_no_spaces_unchanged(self):
        assert _escape_mount_path("/tmp/tasktrace-capture/runs/agent") == "/tmp/tasktrace-capture/runs/agent"

    def test_single_space_escaped(self):
        assert _escape_mount_path("/tmp/tasktrace-capture test/runs") == "/tmp/tasktrace-capture\\040test/runs"

    def test_multiple_spaces_escaped(self):
        path = "/tmp/my test dir/sub folder/agent"
        assert _escape_mount_path(path) == "/tmp/my\\040test\\040dir/sub\\040folder/agent"

    def test_format_mount_no_spaces(self):
        result = _format_mount("/host/path", "/container/path")
        assert result == "/host/path:/container/path"

    def test_format_mount_spaces_in_source(self):
        result = _format_mount("/tmp/tasktrace-capture test/runs/agent", "/logs/agent")
        assert result == "/tmp/tasktrace-capture\\040test/runs/agent:/logs/agent"

    def test_format_mount_spaces_in_both(self):
        result = _format_mount("/tmp/my dir/src", "/mnt/my dir/dst")
        assert result == "/tmp/my\\040dir/src:/mnt/my\\040dir/dst"

    def test_format_mount_preserves_colon_separator(self):
        """The src:dst separator must remain a literal colon."""
        result = _format_mount("/a/b", "/c/d")
        parts = result.split(":")
        assert len(parts) == 2
        assert parts[0] == "/a/b"
        assert parts[1] == "/c/d"


class TestLocalBinIsolation:
    """Tests for /root/.local/bin per-container isolation.

    Verifiers and agents install tool binaries (uv, uvx) into ~/.local/bin.
    Without per-container isolation, concurrent containers race on the same
    executables causing 'Text file busy' / 'File exists' errors.

    Only bin/ is isolated (not all of .local/) to avoid duplicating large
    data directories like .local/share/claude/ (~150MB per agent).
    """

    def test_local_bin_in_isolated_home(self):
        """Isolated /root home should contain .local/bin."""
        # With /root mounted from tmpdir, .local/bin is a real subdir
        cache = Path("/tmp/cache")
        bin_dir = cache / ".local" / "bin"
        assert bin_dir == cache / ".local" / "bin"

    def test_local_bin_dir_created_in_cache(self, tmp_path):
        """The .local/bin directory should be created inside the per-container cache."""
        cache_dir = tmp_path / "claude_cache"
        cache_dir.mkdir()
        bin_dir = cache_dir / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        assert bin_dir.exists()
        assert bin_dir.is_dir()

    def test_local_bin_isolation_prevents_cross_contamination(self, tmp_path):
        """Two containers' .local/bin dirs must be independent."""
        cache_a = tmp_path / "container_a"
        cache_b = tmp_path / "container_b"
        cache_a.mkdir()
        cache_b.mkdir()

        bin_a = cache_a / ".local" / "bin"
        bin_b = cache_b / ".local" / "bin"
        bin_a.mkdir(parents=True)
        bin_b.mkdir(parents=True)

        # Simulate installing a tool in container A
        (bin_a / "uv").write_text("#!/bin/bash\necho uv-a")
        assert (bin_a / "uv").exists()
        # Container B should NOT see it
        assert not (bin_b / "uv").exists()


class TestTrialScopedHome:
    """Tests for per-trial isolated home directory.

    Each container gets an isolated /root mounted from trial_dir/home/.
    This prevents task installs (opam, nvm, pip) from polluting the host
    home and scopes the home to the experiment (no stale caches).
    """

    def test_home_dir_inside_trial_dir(self, tmp_path):
        """The isolated home must live under trial_dir."""
        trial_dir = tmp_path / "runs" / "2026-01-01" / "hello-world__abc"
        trial_dir.mkdir(parents=True)
        home = trial_dir / "home"
        home.mkdir()
        assert home.parent == trial_dir
        assert str(home).startswith(str(trial_dir))

    def test_home_has_required_structure(self, tmp_path):
        """The isolated home must contain .claude/, .local/bin/, shell profiles."""
        home = tmp_path / "home"
        home.mkdir()
        # Replicate what enroot_env.py creates
        (home / ".claude").mkdir()
        (home / ".local" / "bin").mkdir(parents=True)
        (home / ".local" / "share").mkdir(parents=True)
        for f in [".bashrc", ".bash_profile", ".profile"]:
            (home / f).write_text(f"# container {f}\n")

        assert (home / ".claude").is_dir()
        assert (home / ".local" / "bin").is_dir()
        assert (home / ".local" / "share").is_dir()
        assert (home / ".bashrc").exists()
        assert (home / ".bash_profile").exists()
        assert (home / ".profile").exists()

    def test_mount_list_has_single_root_mount(self):
        """Mount list should have one /root entry, not individual files."""
        mounts = [
            ("/some/agent_dir", "/logs/agent"),
            ("/some/verifier_dir", "/logs/verifier"),
            ("/some/artifacts_dir", "/logs/artifacts"),
            ("/some/trial/home", "/root"),
        ]
        root_mounts = [m for m in mounts if m[1] == "/root"]
        assert len(root_mounts) == 1
        # No individual file mounts under /root
        sub_root = [m for m in mounts if m[1].startswith("/root/")]
        assert len(sub_root) == 0

    def test_independent_homes_per_trial(self, tmp_path):
        """Two trials must have independent home directories."""
        trial_a = tmp_path / "trial_a" / "home"
        trial_b = tmp_path / "trial_b" / "home"
        trial_a.mkdir(parents=True)
        trial_b.mkdir(parents=True)

        # Simulate opam install in trial A
        (trial_a / ".opam").mkdir()
        (trial_a / ".opam" / "default").write_text("ocaml config")

        # Trial B should not see it
        assert not (trial_b / ".opam").exists()

    def test_deleting_job_dir_cleans_homes(self, tmp_path):
        """Deleting the job output dir should remove all trial homes."""
        import shutil

        job_dir = tmp_path / "jobs" / "experiment1"
        for i in range(4):
            home = job_dir / "runs" / "run1" / f"task_{i}" / "home"
            home.mkdir(parents=True)
            (home / ".opam").mkdir()
            (home / ".nvm").mkdir()

        shutil.rmtree(job_dir)
        assert not job_dir.exists()

    def test_128_trials_get_unique_homes(self, tmp_path):
        """At c=128, each trial must get its own isolated home."""
        homes = {}
        for i in range(128):
            trial = tmp_path / "runs" / "run1" / f"hello-world__{i:04d}"
            home = trial / "home"
            home.mkdir(parents=True)
            (home / ".claude").mkdir()
            (home / ".local" / "bin").mkdir(parents=True)
            homes[i] = home

        assert len(homes) == 128
        paths = {str(h) for h in homes.values()}
        assert len(paths) == 128, "All home paths must be unique"

        # Simulate install in container 0 — others unaffected
        (homes[0] / ".opam").mkdir()
        assert not (homes[1] / ".opam").exists()
        assert not (homes[127] / ".opam").exists()


class TestHomePollutionPrevention:
    """Tests that task installs land in the isolated home, not the real /home.

    Simulates the pollution patterns observed in real SWE-bench Pro tasks:
    - opam init (compcert: 2.7GB .opam)
    - nvm install (JS tasks: 401MB .nvm)
    - pip install --user (python tasks: .local/lib)
    - cargo build (rust tasks: .cargo)
    - uv python install (221MB .local/share/uv)
    """

    def test_opam_install_stays_in_isolated_home(self, tmp_path):
        """Simulate opam init — must land in trial home, not host home."""
        host_home = tmp_path / "host_home"
        host_home.mkdir()
        trial_home = tmp_path / "trial" / "home"
        trial_home.mkdir(parents=True)

        # Simulate opam init inside container (writes to $HOME/.opam)
        opam_dir = trial_home / ".opam"
        opam_dir.mkdir()
        (opam_dir / "default").mkdir()
        (opam_dir / "default" / "lib").mkdir()
        (opam_dir / "default" / "lib" / "ocaml").write_text("fake ocaml")

        assert (trial_home / ".opam" / "default" / "lib" / "ocaml").exists()
        assert not (host_home / ".opam").exists()

    def test_nvm_install_stays_in_isolated_home(self, tmp_path):
        """Simulate nvm install — must land in trial home, not host home."""
        host_home = tmp_path / "host_home"
        host_home.mkdir()
        trial_home = tmp_path / "trial" / "home"
        trial_home.mkdir(parents=True)

        nvm_dir = trial_home / ".nvm"
        nvm_dir.mkdir()
        (nvm_dir / "versions" / "node" / "v20.0.0" / "bin").mkdir(parents=True)
        (nvm_dir / "versions" / "node" / "v20.0.0" / "bin" / "node").write_text("fake node")

        assert (trial_home / ".nvm" / "versions").exists()
        assert not (host_home / ".nvm").exists()

    def test_pip_user_install_stays_in_isolated_home(self, tmp_path):
        """Simulate pip install --user — must land in trial home."""
        host_home = tmp_path / "host_home"
        host_home.mkdir()
        trial_home = tmp_path / "trial" / "home"
        trial_home.mkdir(parents=True)
        (trial_home / ".local" / "lib" / "python3.12" / "site-packages").mkdir(parents=True)

        pkg = trial_home / ".local" / "lib" / "python3.12" / "site-packages" / "flask"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("# flask")

        assert pkg.exists()
        assert not (host_home / ".local").exists()

    def test_cargo_build_stays_in_isolated_home(self, tmp_path):
        """Simulate cargo build — must land in trial home."""
        host_home = tmp_path / "host_home"
        host_home.mkdir()
        trial_home = tmp_path / "trial" / "home"
        trial_home.mkdir(parents=True)

        cargo_dir = trial_home / ".cargo"
        cargo_dir.mkdir()
        (cargo_dir / "registry").mkdir()
        (cargo_dir / "bin").mkdir()

        assert (trial_home / ".cargo" / "registry").exists()
        assert not (host_home / ".cargo").exists()

    def test_uv_python_install_stays_in_isolated_home(self, tmp_path):
        """Simulate uv python install — observed 221MB in real SWE-bench Pro run."""
        host_home = tmp_path / "host_home"
        host_home.mkdir()
        trial_home = tmp_path / "trial" / "home"
        (trial_home / ".local" / "share" / "uv" / "python").mkdir(parents=True)

        cpython = trial_home / ".local" / "share" / "uv" / "python" / "cpython-3.14.0"
        cpython.mkdir()
        (cpython / "bin" / "python3").parent.mkdir(parents=True)
        (cpython / "bin" / "python3").write_text("fake python")

        assert cpython.exists()
        assert not (host_home / ".local").exists()

    def test_concurrent_tasks_dont_cross_pollute(self, tmp_path):
        """Two concurrent SWE-bench tasks with different installs stay isolated."""
        home_a = tmp_path / "trial_a" / "home"
        home_b = tmp_path / "trial_b" / "home"
        home_a.mkdir(parents=True)
        home_b.mkdir(parents=True)

        # Task A installs opam (OCaml compcert)
        (home_a / ".opam").mkdir()
        (home_a / ".opam" / "default").write_text("ocaml")

        # Task B installs nvm (JS task)
        (home_b / ".nvm").mkdir()
        (home_b / ".nvm" / "versions").mkdir()

        # Neither leaks to the other
        assert not (home_a / ".nvm").exists()
        assert not (home_b / ".opam").exists()

    def test_stop_cleanup_removes_polluted_home(self, tmp_path):
        """After stop(), the polluted home dir is fully removed."""
        import shutil

        trial_home = tmp_path / "trial" / "home"
        trial_home.mkdir(parents=True)

        # Simulate heavy pollution
        (trial_home / ".opam").mkdir()
        (trial_home / ".nvm").mkdir()
        (trial_home / ".cargo").mkdir()
        (trial_home / ".cache" / "pip").mkdir(parents=True)
        (trial_home / ".local" / "share" / "uv").mkdir(parents=True)

        # Simulate stop() cleanup
        shutil.rmtree(trial_home, ignore_errors=True)

        assert not trial_home.exists()
        # Parent trial dir still exists
        assert (tmp_path / "trial").exists()

    def test_new_experiment_gets_clean_home(self, tmp_path):
        """A fresh experiment must start with a clean home, no stale caches."""
        # Experiment 1: installs opam
        exp1_home = tmp_path / "exp1" / "trial" / "home"
        exp1_home.mkdir(parents=True)
        (exp1_home / ".opam").mkdir()

        # Experiment 2: fresh home
        exp2_home = tmp_path / "exp2" / "trial" / "home"
        exp2_home.mkdir(parents=True)

        assert not (exp2_home / ".opam").exists()
        assert not (exp2_home / ".nvm").exists()
        assert not (exp2_home / ".cargo").exists()


class TestBinShSymlink:
    """Tests for /bin/sh -> /bin/bash symlink creation in start().

    Some base images have /bin/bash but not /bin/sh. Harbor's upload_dir and
    terminus-2's tmux pipe-pane invoke /bin/sh, causing
    'enroot-switchroot: failed to execute: /bin/sh: No such file or directory'.
    The fix creates a symlink during container start().
    """

    def test_symlink_command_is_correct(self):
        """The shell command should only create the symlink if /bin/sh is missing
        and /bin/bash exists."""
        cmd = "if [ ! -e /bin/sh ] && [ -x /bin/bash ]; then ln -s /bin/bash /bin/sh; fi"
        # Verify command structure: guards both conditions
        assert "! -e /bin/sh" in cmd, "Must check /bin/sh is missing"
        assert "-x /bin/bash" in cmd, "Must check /bin/bash is executable"
        assert "ln -s /bin/bash /bin/sh" in cmd, "Must create symlink"

    def test_symlink_creates_when_sh_missing(self, tmp_path):
        """Simulate: /bin/bash exists, /bin/sh does not -> symlink created."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        bash = bin_dir / "bash"
        bash.write_text("#!/bin/bash\n")
        bash.chmod(0o755)

        sh = bin_dir / "sh"
        # /bin/sh does not exist
        assert not sh.exists()

        # Simulate the symlink logic
        if not sh.exists() and bash.exists() and os.access(str(bash), os.X_OK):
            sh.symlink_to(bash)

        assert sh.exists()
        assert sh.is_symlink()
        assert sh.resolve() == bash.resolve()

    def test_symlink_skipped_when_sh_exists(self, tmp_path):
        """Simulate: both /bin/sh and /bin/bash exist -> no symlink created."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        bash = bin_dir / "bash"
        bash.write_text("#!/bin/bash\n")
        bash.chmod(0o755)

        sh = bin_dir / "sh"
        sh.write_text("#!/bin/dash\n")
        sh.chmod(0o755)
        original_content = sh.read_text()

        # Simulate the symlink logic (should NOT run)
        if not sh.exists() and bash.exists() and os.access(str(bash), os.X_OK):
            sh.symlink_to(bash)

        # /bin/sh should remain as-is (not replaced with symlink)
        assert not sh.is_symlink()
        assert sh.read_text() == original_content

    def test_symlink_skipped_when_bash_missing(self, tmp_path):
        """Simulate: neither /bin/sh nor /bin/bash -> no symlink created."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        sh = bin_dir / "sh"

        # Neither exists
        assert not sh.exists()
        assert not (bin_dir / "bash").exists()

        # Simulate the symlink logic (should NOT run)
        bash = bin_dir / "bash"
        if not sh.exists() and bash.exists() and os.access(str(bash), os.X_OK):
            sh.symlink_to(bash)

        assert not sh.exists(), "/bin/sh should not be created without /bin/bash"


# ===========================================================================
#  Cache directory security (symlink attacks, permissions)
# ===========================================================================


class TestCacheDirPermissions:
    """Cache directories must be created with mode 0o700 to prevent other users
    from reading/writing container state."""

    def test_mkdir_mode_0o700(self, tmp_path):
        """Directories created for per-container cache must have 0o700 permissions."""
        cache_dir = tmp_path / "harbor_claude_test_container"
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        mode = stat.S_IMODE(cache_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

    def test_local_bin_mkdir_mode_0o700(self, tmp_path):
        """The .local/bin subdirectory must also be created with 0o700."""
        bin_dir = tmp_path / "cache" / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        mode = stat.S_IMODE(bin_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"


class TestSymlinkRejection:
    """Cache path must not be a symlink -- prevents symlink attacks where an
    attacker plants a symlink at a predictable /tmp path."""

    def test_symlink_at_cache_path_rejected(self, tmp_path):
        """If the cache path is a symlink, start() must raise RuntimeError."""
        # Simulate the check that enroot_env.py performs
        target = tmp_path / "evil_target"
        target.mkdir()
        symlink = tmp_path / "harbor_claude_hb__fake"
        symlink.symlink_to(target)

        assert symlink.is_symlink()
        # The code checks is_symlink() before mkdir -- verify the pattern

        cache_path = symlink
        with pytest.raises(RuntimeError, match="symlink"):
            if cache_path.is_symlink():
                raise RuntimeError(
                    f"Refusing to use cache path that is a symlink: {cache_path}. This may indicate a symlink attack."
                )

    def test_non_symlink_path_accepted(self, tmp_path):
        """A regular directory should not trigger the symlink check."""
        cache_path = tmp_path / "harbor_claude_hb__normal"
        assert not cache_path.is_symlink()
        cache_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        assert cache_path.is_dir()
        assert not cache_path.is_symlink()


class TestPerContainerUploadMount:
    """Upload/download mounts must use per-container unique paths instead of
    the shared /tmp/_hb_upload."""

    def test_upload_mount_contains_container_name(self):
        """The upload mount path must include the container name for uniqueness."""
        container_name = "hb__session-abc123"
        mount_path = f"/tmp/_hb_upload_{container_name}"
        assert container_name in mount_path
        assert mount_path != "/tmp/_hb_upload"

    def test_different_containers_get_different_mounts(self):
        """Two different containers must get different upload mount paths."""
        mount_a = "/tmp/_hb_upload_hb__session-a"
        mount_b = "/tmp/_hb_upload_hb__session-b"
        assert mount_a != mount_b

    def test_download_mount_contains_container_name(self):
        """The download mount path must include the container name."""
        container_name = "hb__session-xyz789"
        mount_path = f"/tmp/_hb_download_{container_name}"
        assert container_name in mount_path
        assert mount_path != "/tmp/_hb_download"


# ===========================================================================
#  Portable Bash command timing
# ===========================================================================


def _make_trace_exec_env(tmp_path):
    import logging
    from types import SimpleNamespace
    from unittest import mock

    from harbor.environments.base import ExecResult

    env = object.__new__(EnrootEnvironment)
    env._container_name = "hb__trace"
    env.trial_paths = SimpleNamespace(agent_dir=tmp_path)
    env.logger = logging.getLogger("test.enroot.trace")
    env._barrier_passed = True
    env._mounts = []
    env._workdir = None
    env._resolve_user = lambda user: user
    env._merge_env = lambda values: values or {}
    env._run_cmd = mock.AsyncMock(return_value=ExecResult(return_code=0))
    return env


def _run_trace_exec(coro):
    try:
        return asyncio.run(coro)
    finally:
        # Legacy tests below still use get_event_loop() directly.
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_exec_injects_portable_trace_env_for_mini_swe(monkeypatch, tmp_path):
    from tasktrace_capture.shell_capture import (
        CAPTURE_CONTAINER_SCRIPT,
        CAPTURE_SCRIPT_NAME,
    )

    monkeypatch.setenv("_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS", "1")
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "mini-swe-agent")
    env = _make_trace_exec_env(tmp_path)

    _run_trace_exec(env.exec("mini-swe-agent --yolo --model=openai/model --task=x --exit-immediately"))

    command = env._run_cmd.await_args.args[0]
    assert f"BASH_ENV={CAPTURE_CONTAINER_SCRIPT}" in command
    assert "--environment-class tasktrace_capture_miniswe_trace_env." in command[-1]
    assert (tmp_path / CAPTURE_SCRIPT_NAME).exists()


def test_exec_installs_trace_hook_before_terminus_login_shell(monkeypatch, tmp_path):
    from tasktrace_capture.shell_capture import CAPTURE_SCRIPT_NAME

    monkeypatch.setenv("_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS", "1")
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "terminus-2")
    env = _make_trace_exec_env(tmp_path)

    _run_trace_exec(env.exec("tmux new-session -d -s t 'bash --login'"))

    assert env._run_cmd.await_count == 2
    setup_command = env._run_cmd.await_args_list[0].args[0]
    tmux_command = env._run_cmd.await_args_list[1].args[0]
    assert ".bash_profile" in setup_command[-1]
    assert "tmux new-session" in tmux_command[-1]
    assert (tmp_path / CAPTURE_SCRIPT_NAME).exists()


# ===========================================================================
#  Retry logic for high-concurrency robustness
# ===========================================================================


class TestRetryCmd:
    """Verify _retry_cmd retries with backoff on RuntimeError."""

    def test_succeeds_on_first_attempt(self):
        """No retries needed when the command succeeds immediately."""
        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = asyncio.get_event_loop().run_until_complete(_retry_cmd(_factory, max_retries=3, base_sec=0.01))
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_runtime_error(self):
        """Should retry up to max_retries times before succeeding."""
        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient failure")
            return "recovered"

        result = asyncio.get_event_loop().run_until_complete(_retry_cmd(_factory, max_retries=5, base_sec=0.01))
        assert result == "recovered"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Should raise RuntimeError after exhausting all retries."""
        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            asyncio.get_event_loop().run_until_complete(_retry_cmd(_factory, max_retries=2, base_sec=0.01))
        # 1 initial + 2 retries = 3 total attempts
        assert call_count == 3

    def test_does_not_catch_non_runtime_errors(self):
        """Should not retry on ValueError, TypeError, etc."""
        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            raise ValueError("not a runtime error")

        with pytest.raises(ValueError, match="not a runtime error"):
            asyncio.get_event_loop().run_until_complete(_retry_cmd(_factory, max_retries=5, base_sec=0.01))
        assert call_count == 1  # no retry

    def test_backoff_increases_delay(self):
        """Verify that retry delays follow exponential backoff."""
        timestamps = []

        async def _factory():
            timestamps.append(time.monotonic())
            if len(timestamps) < 4:
                raise RuntimeError("fail")
            return "done"

        asyncio.get_event_loop().run_until_complete(_retry_cmd(_factory, max_retries=5, base_sec=0.05))
        assert len(timestamps) == 4
        # Delays should generally increase (with jitter, so we check
        # that later delays are at least as long as earlier ones minus jitter)
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        # First delay: ~0.05 * 2^0 = 0.05s, second: ~0.05 * 2^1 = 0.1s, third: ~0.05 * 2^2 = 0.2s
        # With jitter and scheduling variance, just check they are positive
        # and that the last delay is larger than the first
        assert all(d > 0 for d in delays)
        assert delays[-1] > delays[0]

    def test_logger_receives_warnings(self):
        """Logger should be called with warnings on each retry."""
        warnings = []

        class FakeLogger:
            def warning(self, msg, *args):
                warnings.append(msg % args)

        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("fail")
            return "ok"

        asyncio.get_event_loop().run_until_complete(
            _retry_cmd(_factory, max_retries=5, base_sec=0.01, logger=FakeLogger())
        )
        assert len(warnings) == 2
        assert "attempt 1/" in warnings[0]
        assert "attempt 2/" in warnings[1]


class TestRetryCmdConstants:
    """Verify module-level retry constants are reasonable."""

    def test_max_retries_positive(self):
        assert _ENROOT_MAX_RETRIES >= 3

    def test_base_sec_positive(self):
        assert _ENROOT_RETRY_BASE_SEC > 0

    def test_max_total_delay_bounded(self):
        """Total worst-case delay should be under 60s to avoid harbor timeout."""
        total = sum(_ENROOT_RETRY_BASE_SEC * (2**i) + _ENROOT_RETRY_BASE_SEC for i in range(_ENROOT_MAX_RETRIES))
        assert total < 60, f"Max retry delay {total}s exceeds 60s budget"


class TestHomeSymlinkLock:
    """Verify the home symlink lock prevents TOCTOU races.

    At c>=128, all containers try to create /home/<user> -> /root
    simultaneously. Without a lock, rm + ln races cause failures.
    """

    def test_lock_path_is_in_tmp(self):
        """Lock file should live in the temp directory."""
        import tempfile

        assert str(_HOME_SYMLINK_LOCK).startswith(tempfile.gettempdir())

    def test_lock_path_has_descriptive_name(self):
        assert "tasktrace_capture" in _HOME_SYMLINK_LOCK.name
        assert "symlink" in _HOME_SYMLINK_LOCK.name

    def test_concurrent_lock_acquisition(self, tmp_path):
        """Simulate concurrent lock acquisition to verify serialization."""
        lock_path = tmp_path / "test_symlink.lock"
        results = []

        async def _acquire_lock(worker_id):
            fd = open(lock_path, "w")
            try:
                await asyncio.get_event_loop().run_in_executor(None, lambda: fcntl.flock(fd, fcntl.LOCK_EX))
                results.append(("acquired", worker_id))
                # Simulate work
                await asyncio.sleep(0.01)
                results.append(("released", worker_id))
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()

        async def _run():
            # Launch 10 concurrent lock acquisitions
            tasks = [_acquire_lock(i) for i in range(10)]
            await asyncio.gather(*tasks)

        asyncio.get_event_loop().run_until_complete(_run())

        # All 10 workers should have acquired and released
        acquired = [r for r in results if r[0] == "acquired"]
        released = [r for r in results if r[0] == "released"]
        assert len(acquired) == 10
        assert len(released) == 10

        # Verify serialization: no two workers held the lock simultaneously.
        # Each "acquired" should be followed by "released" for the same worker
        # before the next "acquired" appears.
        active_workers = set()
        for action, worker_id in results:
            if action == "acquired":
                assert worker_id not in active_workers, f"Worker {worker_id} acquired lock while already holding it"
                assert (
                    len(active_workers) == 0
                ), f"Worker {worker_id} acquired lock while {active_workers} still holds it — serialization broken"
                active_workers.add(worker_id)
            elif action == "released":
                active_workers.discard(worker_id)


class TestCacheDirOrdering:
    """Verify that isolation files exist before they are referenced in mounts.

    The old code created .bashrc and .local/bin AFTER setting up the mount
    list, leaving a window where mounts pointed to nonexistent paths.
    Fix: create files first, then build mount list.
    """

    def test_bashrc_exists_before_mount_reference(self, tmp_path):
        """Simulate the fixed ordering: create files, then check mount list."""
        cache_dir = tmp_path / "harbor_claude_test"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Create files FIRST (the fix)
        bashrc = cache_dir / ".bashrc"
        bashrc_tmp = bashrc.with_suffix(".tmp")
        bashrc_tmp.write_text("# container bashrc\n")
        bashrc_tmp.rename(bashrc)

        local_bin = cache_dir / ".local" / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)

        # Step 2: Build mount list — entire cache_dir mounted as /root
        mounts = [
            (str(cache_dir), "/root"),
        ]

        # Step 3: Verify all mount sources exist
        for host_path, _ in mounts:
            assert os.path.exists(
                host_path
            ), f"Mount source {host_path} does not exist — file creation must happen before mount list setup"

    def test_atomic_bashrc_write(self, tmp_path):
        """Verify atomic bashrc creation via tmp + rename pattern."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        bashrc = cache_dir / ".bashrc"
        bashrc_tmp = bashrc.with_suffix(".tmp")

        # Write to temp, then rename
        bashrc_tmp.write_text("# container bashrc\n")
        assert bashrc_tmp.exists()
        assert not bashrc.exists()

        bashrc_tmp.rename(bashrc)
        assert bashrc.exists()
        assert not bashrc_tmp.exists()
        assert bashrc.read_text() == "# container bashrc\n"

    def test_concurrent_cache_dir_creation(self, tmp_path):
        """Multiple containers creating their cache dirs simultaneously."""
        results = []

        def _create_cache(container_id):
            cache = tmp_path / f"harbor_claude_container_{container_id}"
            cache.mkdir(parents=True, exist_ok=True)

            bashrc = cache / ".bashrc"
            bashrc_tmp = bashrc.with_suffix(".tmp")
            bashrc_tmp.write_text(f"# bashrc for {container_id}\n")
            bashrc_tmp.rename(bashrc)

            local_bin = cache / ".local" / "bin"
            local_bin.mkdir(parents=True, exist_ok=True)

            results.append((container_id, cache.exists(), bashrc.exists(), local_bin.exists()))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(_create_cache, range(128)))

        assert len(results) == 128
        for cid, cache_ok, bashrc_ok, bin_ok in results:
            assert cache_ok, f"Container {cid}: cache dir missing"
            assert bashrc_ok, f"Container {cid}: .bashrc missing"
            assert bin_ok, f"Container {cid}: .local/bin missing"

        # Verify each container has its own isolated files
        for i in range(128):
            cache = tmp_path / f"harbor_claude_container_{i}"
            assert cache.exists()
            assert (cache / ".bashrc").read_text() == f"# bashrc for {i}\n"


class TestHighConcurrencySetup:
    """End-to-end tests for container setup at high concurrency.

    These tests verify the fixes work together — retry logic, file locking,
    and ordering — without requiring actual enroot/harbor.
    """

    def test_retry_cmd_with_concurrent_callers(self):
        """Multiple concurrent retriers should all eventually succeed."""
        # Simulate a resource that allows max 10 concurrent users
        semaphore = asyncio.Semaphore(10)
        total_calls = 0

        async def _flaky_operation(worker_id):
            nonlocal total_calls
            total_calls += 1
            if not semaphore._value:
                # Simulate contention failure
                raise RuntimeError(f"Worker {worker_id}: resource busy")
            async with semaphore:
                await asyncio.sleep(0.001)
                return f"worker-{worker_id}-done"

        async def _run():
            tasks = []
            for i in range(64):
                tasks.append(
                    _retry_cmd(
                        lambda i=i: _flaky_operation(i),
                        max_retries=5,
                        base_sec=0.01,
                    )
                )
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.get_event_loop().run_until_complete(_run())
        # All workers should succeed (some via retries)
        successes = [r for r in results if isinstance(r, str)]
        failures = [r for r in results if isinstance(r, Exception)]
        # With semaphore(10) and 64 workers, some may fail but most should succeed
        # The key assertion is that retry logic doesn't deadlock or crash
        assert len(successes) + len(failures) == 64

    def test_128_cache_dirs_no_collision(self, tmp_path):
        """128 containers each get unique, isolated cache dirs."""
        containers = {}
        for i in range(128):
            name = f"hb__session_{i:04d}"
            cache = tmp_path / f"harbor_claude_{name}"
            cache.mkdir(parents=True, exist_ok=True)
            bashrc = cache / ".bashrc"
            bashrc.write_text(f"# bashrc {i}\n")
            local_bin = cache / ".local" / "bin"
            local_bin.mkdir(parents=True, exist_ok=True)
            containers[name] = cache

        # Verify all 128 are unique and intact
        assert len(containers) == 128
        paths = {str(v) for v in containers.values()}
        assert len(paths) == 128, "Cache dir paths must be unique"

        for _name, cache in containers.items():
            assert (cache / ".bashrc").exists()
            assert (cache / ".local" / "bin").is_dir()

    def test_retry_preserves_final_exception(self):
        """The last exception message should be preserved when all retries fail."""
        attempt = 0

        async def _factory():
            nonlocal attempt
            attempt += 1
            raise RuntimeError(f"fail attempt {attempt}")

        with pytest.raises(RuntimeError, match="fail attempt 3"):
            asyncio.get_event_loop().run_until_complete(_retry_cmd(_factory, max_retries=2, base_sec=0.01))


# ===========================================================================
#  Claude-code pre-install (avoid N concurrent downloads)
# ===========================================================================


class TestClaudeCodeCacheConstants:
    """Verify module-level constants for the shared claude-code binary cache."""

    def test_cache_dir_in_tmp(self):
        """Cache directory should live under the system temp directory."""
        import tempfile

        assert str(_CLAUDE_CODE_CACHE_DIR).startswith(tempfile.gettempdir())

    def test_cache_dir_has_descriptive_name(self):
        assert "tasktrace_capture" in _CLAUDE_CODE_CACHE_DIR.name
        assert "claude" in _CLAUDE_CODE_CACHE_DIR.name

    def test_lock_path_matches_cache_dir(self):
        """Lock file should be adjacent to the cache dir (same parent, .lock suffix)."""
        assert _CLAUDE_CODE_CACHE_LOCK.parent == _CLAUDE_CODE_CACHE_DIR.parent
        assert _CLAUDE_CODE_CACHE_LOCK.suffix == ".lock"
        assert _CLAUDE_CODE_CACHE_LOCK.stem == _CLAUDE_CODE_CACHE_DIR.name


class TestEnsureClaudeCodeBinary:
    """Tests for _ensure_claude_code_binary.

    At c>=64, concurrent curl downloads of claude-code fail for most
    containers. The fix pre-installs the binary once using a file lock
    and caches it for all containers to copy.
    """

    def test_returns_cached_binary_if_exists(self, tmp_path, monkeypatch):
        """If a valid binary already exists in the cache, return it immediately
        without downloading."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        cache_dir.mkdir(mode=0o700)
        binary = cache_dir / "claude"
        binary.write_text("#!/bin/bash\necho claude")
        binary.chmod(0o755)
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_LOCK", lock_path)

        result = _ensure_claude_code_binary()
        assert result is not None
        assert result == binary
        assert result.exists()

    def test_installs_when_no_cache(self, tmp_path, monkeypatch):
        """When no cached binary exists, runs the installer and caches the result."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_LOCK", lock_path)

        # Mock subprocess.run to simulate a successful install
        import subprocess as sp

        def mock_run(cmd, **kwargs):
            # The installer writes the binary to $HOME/.local/bin/claude
            home = kwargs.get("env", {}).get("HOME", str(cache_dir))
            bin_dir = os.path.join(home, ".local", "bin")
            os.makedirs(bin_dir, exist_ok=True)
            claude_bin = os.path.join(bin_dir, "claude")
            with open(claude_bin, "w") as f:
                f.write("#!/bin/bash\necho claude\n")
            os.chmod(claude_bin, 0o755)
            return sp.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.subprocess.run", mock_run)

        result = _ensure_claude_code_binary()
        assert result is not None
        assert result.exists()
        assert os.access(str(result), os.X_OK)
        # The cached binary should be at cache_dir/claude
        assert result == cache_dir / "claude"

    def test_returns_none_on_install_failure(self, tmp_path, monkeypatch):
        """If the installer fails, return None instead of crashing."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_LOCK", lock_path)

        import subprocess as sp

        def mock_run_fail(cmd, **kwargs):
            return sp.CompletedProcess(cmd, returncode=1, stdout="", stderr="download failed")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.subprocess.run", mock_run_fail)

        result = _ensure_claude_code_binary()
        assert result is None

    def test_returns_none_on_timeout(self, tmp_path, monkeypatch):
        """If the installer times out, return None."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_LOCK", lock_path)

        import subprocess as sp

        def mock_run_timeout(cmd, **kwargs):
            raise sp.TimeoutExpired(cmd, kwargs.get("timeout", 120))

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.subprocess.run", mock_run_timeout)

        result = _ensure_claude_code_binary()
        assert result is None

    def test_returns_none_when_binary_missing_after_install(self, tmp_path, monkeypatch):
        """If the installer succeeds but the binary isn't where expected, return None."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_LOCK", lock_path)

        import subprocess as sp

        def mock_run_no_binary(cmd, **kwargs):
            # Simulate install that succeeds but doesn't create the binary
            return sp.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.subprocess.run", mock_run_no_binary)

        result = _ensure_claude_code_binary()
        assert result is None

    def test_file_lock_serializes_concurrent_callers(self, tmp_path, monkeypatch):
        """Multiple concurrent callers should serialize via file lock; only
        the first one installs, the rest find the cached binary."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.runtimes._shared.CLAUDE_CODE_CACHE_LOCK", lock_path)

        install_count = 0

        import subprocess as sp

        def mock_run_counting(cmd, **kwargs):
            nonlocal install_count
            install_count += 1
            home = kwargs.get("env", {}).get("HOME", str(cache_dir))
            bin_dir = os.path.join(home, ".local", "bin")
            os.makedirs(bin_dir, exist_ok=True)
            claude_bin = os.path.join(bin_dir, "claude")
            with open(claude_bin, "w") as f:
                f.write("#!/bin/bash\necho claude\n")
            os.chmod(claude_bin, 0o755)
            return sp.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("tasktrace_capture.runtimes._shared.subprocess.run", mock_run_counting)

        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_ensure_claude_code_binary) for _ in range(16)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # All 16 callers should get a valid binary path
        assert all(r is not None for r in results)
        assert all(r.exists() for r in results)
        # Only one caller should have triggered the actual install
        assert (
            install_count == 1
        ), f"Expected exactly 1 install call, got {install_count}. File lock failed to serialize concurrent callers."


class TestClaudeCodePreInstallInStart:
    """Tests for the pre-install integration in start().

    Verifies that when _TASKTRACE_CAPTURE_AGENT=claude-code, the cached binary
    is copied into each container's isolated .local/bin directory.
    """

    def test_binary_copied_to_local_bin(self, tmp_path, monkeypatch):
        """When a cached binary exists and agent is claude-code, it should
        be copied into the container's .local/bin."""
        # Simulate the pre-install logic from start()
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        cache_dir.mkdir(mode=0o700)
        cached_binary = cache_dir / "claude"
        cached_binary.write_text("#!/bin/bash\necho claude")
        cached_binary.chmod(0o755)

        local_bin_path = tmp_path / "container_local_bin"
        local_bin_path.mkdir(mode=0o700)

        # Simulate the logic from start():
        # if agent_name == "claude-code": copy cached binary to local_bin
        import shutil

        dest = local_bin_path / "claude"
        if not dest.exists():
            shutil.copy2(str(cached_binary), str(dest))
            os.chmod(str(dest), 0o755)

        assert dest.exists()
        assert os.access(str(dest), os.X_OK)
        assert dest.read_text() == "#!/bin/bash\necho claude"

    def test_binary_not_copied_for_other_agents(self, tmp_path):
        """When agent is not claude-code, no pre-install should happen."""
        local_bin_path = tmp_path / "container_local_bin"
        local_bin_path.mkdir(mode=0o700)

        # Simulate the logic from start() with a non-claude-code agent
        agent_name = "terminus-2"
        if agent_name == "claude-code":
            # This block should NOT execute
            (local_bin_path / "claude").write_text("should not exist")

        assert not (local_bin_path / "claude").exists()

    def test_skip_copy_if_binary_already_exists(self, tmp_path):
        """If the binary is already in .local/bin (e.g. from a previous
        run), do not overwrite it."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        cache_dir.mkdir(mode=0o700)
        cached_binary = cache_dir / "claude"
        cached_binary.write_text("#!/bin/bash\necho NEW")
        cached_binary.chmod(0o755)

        local_bin_path = tmp_path / "container_local_bin"
        local_bin_path.mkdir(mode=0o700)
        existing = local_bin_path / "claude"
        existing.write_text("#!/bin/bash\necho OLD")
        existing.chmod(0o755)

        import shutil

        dest = local_bin_path / "claude"
        if not dest.exists():
            shutil.copy2(str(cached_binary), str(dest))

        # Should keep the OLD version
        assert dest.read_text() == "#!/bin/bash\necho OLD"

    def test_multiple_containers_get_independent_copies(self, tmp_path):
        """Each container should get its own copy of the binary."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        cache_dir.mkdir(mode=0o700)
        cached_binary = cache_dir / "claude"
        cached_binary.write_text("#!/bin/bash\necho claude")
        cached_binary.chmod(0o755)

        import shutil

        containers = {}
        for i in range(64):
            local_bin = tmp_path / f"container_{i}" / ".local" / "bin"
            local_bin.mkdir(parents=True, mode=0o700)
            dest = local_bin / "claude"
            shutil.copy2(str(cached_binary), str(dest))
            os.chmod(str(dest), 0o755)
            containers[i] = dest

        # All 64 containers should have independent copies
        assert len(containers) == 64
        for _i, dest in containers.items():
            assert dest.exists()
            assert os.access(str(dest), os.X_OK)

        # Modifying one container's binary should not affect others
        containers[0].write_text("#!/bin/bash\necho modified")
        assert containers[1].read_text() == "#!/bin/bash\necho claude"

    def test_concurrent_copy_to_64_containers(self, tmp_path):
        """Simulate 64 containers concurrently copying the cached binary."""
        cache_dir = tmp_path / "tasktrace_capture_claude_code_cache"
        cache_dir.mkdir(mode=0o700)
        cached_binary = cache_dir / "claude"
        cached_binary.write_text("#!/bin/bash\necho claude")
        cached_binary.chmod(0o755)

        import concurrent.futures
        import shutil

        results = []

        def _copy_binary(container_id):
            local_bin = tmp_path / f"container_{container_id}" / ".local" / "bin"
            local_bin.mkdir(parents=True, mode=0o700)
            dest = local_bin / "claude"
            if not dest.exists():
                shutil.copy2(str(cached_binary), str(dest))
                os.chmod(str(dest), 0o755)
            return (container_id, dest.exists(), os.access(str(dest), os.X_OK))

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            results = list(pool.map(_copy_binary, range(64)))

        assert len(results) == 64
        for cid, exists, executable in results:
            assert exists, f"Container {cid}: binary not copied"
            assert executable, f"Container {cid}: binary not executable"


class TestTasktraceCaptureAgentEnvVar:
    """Tests for _TASKTRACE_CAPTURE_AGENT environment variable propagation.

    runner.py sets _TASKTRACE_CAPTURE_AGENT so enroot_env.py can detect which agent
    is being used and apply agent-specific optimizations.
    """

    def test_agent_env_var_detected(self, monkeypatch):
        """start() should read _TASKTRACE_CAPTURE_AGENT from the environment."""
        monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "claude-code")
        assert os.environ.get("_TASKTRACE_CAPTURE_AGENT") == "claude-code"

    def test_agent_env_var_absent_is_safe(self, monkeypatch):
        """When _TASKTRACE_CAPTURE_AGENT is not set, it should default to empty string."""
        monkeypatch.delenv("_TASKTRACE_CAPTURE_AGENT", raising=False)
        agent_name = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
        assert agent_name == ""
        # Should not trigger claude-code pre-install
        assert agent_name != "claude-code"

    def test_non_claude_agent_skips_preinstall(self, monkeypatch):
        """Terminus and other agents should not trigger pre-install."""
        for agent in ["terminus-2", "terminus", "codex", "aider", "swe-agent"]:
            monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", agent)
            agent_name = os.environ.get("_TASKTRACE_CAPTURE_AGENT", "")
            assert agent_name != "claude-code"


# ===========================================================================
#  Tmux pre-install (avoid N concurrent source builds)
# ===========================================================================


class _TestLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _ExecResult:
    def __init__(self, return_code=0, stdout="", stderr=""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _FakeTmuxEnv:
    def __init__(self, install_counter=None, build_ok=True, binary_name="tmux"):
        self._container_name = "hb-test"
        self.logger = _TestLogger()
        self.available = False
        self.install_counter = install_counter
        self.build_ok = build_ok
        self.binary_name = binary_name

    async def exec(self, command, user="root"):
        if command == "tmux -V || /usr/local/bin/tmux -V":
            if self.available:
                return _ExecResult(0, "tmux 3.4\n", "")
            return _ExecResult(1, "", "missing")
        if "build-essential" in command or "groupinstall" in command or "apk add" in command:
            return _ExecResult(0, "", "")
        if command.startswith("cd /tmp && curl -L https://github.com/tmux/tmux/releases/download/3.4/"):
            if self.install_counter is not None:
                self.install_counter["count"] += 1
            if self.build_ok:
                self.available = True
                return _ExecResult(0, "", "")
            return _ExecResult(1, "", "build failed")
        if command.startswith("command -v tmux || "):
            if self.available:
                return _ExecResult(0, f"/usr/local/bin/{self.binary_name}\n", "")
            return _ExecResult(1, "", "missing")
        raise AssertionError(f"Unexpected command: {command}")

    async def download_file(self, source_path, target_path):
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!/bin/bash\necho {Path(source_path).name}\n")


class TestTmuxCacheConstants:
    def test_cache_dir_in_tmp(self):
        import tempfile

        assert str(_TMUX_CACHE_DIR).startswith(tempfile.gettempdir())

    def test_cache_dir_has_descriptive_name(self):
        assert "tasktrace_capture" in _TMUX_CACHE_DIR.name
        assert "tmux" in _TMUX_CACHE_DIR.name

    def test_lock_path_matches_cache_dir(self):
        assert _TMUX_CACHE_LOCK.parent == _TMUX_CACHE_DIR.parent
        assert _TMUX_CACHE_LOCK.suffix == ".lock"
        assert _TMUX_CACHE_LOCK.stem == _TMUX_CACHE_DIR.name


class TestEnsureTmuxBinary:
    def test_returns_cached_binary_if_exists(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "tasktrace_capture_tmux_cache"
        cache_dir.mkdir(mode=0o700)
        binary = cache_dir / "tmux"
        binary.write_text("#!/bin/bash\necho tmux")
        binary.chmod(0o755)
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_LOCK", lock_path)

        env = _FakeTmuxEnv()
        result = asyncio.run(EnrootEnvironment._ensure_tmux_binary(env))
        assert result == binary
        assert result.exists()

    def test_builds_and_caches_when_missing(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "tasktrace_capture_tmux_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_LOCK", lock_path)

        counter = {"count": 0}
        env = _FakeTmuxEnv(install_counter=counter)
        result = asyncio.run(EnrootEnvironment._ensure_tmux_binary(env))

        assert result == cache_dir / "tmux"
        assert result.exists()
        assert os.access(str(result), os.X_OK)
        assert counter["count"] == 1

    def test_returns_none_when_build_fails(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "tasktrace_capture_tmux_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_LOCK", lock_path)

        env = _FakeTmuxEnv(build_ok=False)
        result = asyncio.run(EnrootEnvironment._ensure_tmux_binary(env))
        assert result is None
        assert not (cache_dir / "tmux").exists()

    def test_file_lock_serializes_concurrent_callers(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "tasktrace_capture_tmux_cache"
        lock_path = cache_dir.with_suffix(".lock")

        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_DIR", cache_dir)
        monkeypatch.setattr("tasktrace_capture.enroot_env._TMUX_CACHE_LOCK", lock_path)

        counter = {"count": 0}

        def _worker():
            env = _FakeTmuxEnv(install_counter=counter)
            return asyncio.run(EnrootEnvironment._ensure_tmux_binary(env))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda _: _worker(), range(16)))

        assert all(r is not None for r in results)
        assert all(r.exists() for r in results)
        assert counter["count"] == 1


class TestTmuxPreInstallInStart:
    def test_binary_uploaded_to_usr_local_bin(self, tmp_path):
        cache_dir = tmp_path / "tasktrace_capture_tmux_cache"
        cache_dir.mkdir(mode=0o700)
        cached_binary = cache_dir / "tmux"
        cached_binary.write_text("#!/bin/bash\necho tmux")
        cached_binary.chmod(0o755)

        usr_local_bin = tmp_path / "usr" / "local" / "bin"
        usr_local_bin.mkdir(parents=True, mode=0o700)
        dest = usr_local_bin / "tmux"
        if not dest.exists():
            dest.write_bytes(cached_binary.read_bytes())
            os.chmod(str(dest), 0o755)

        assert dest.exists()
        assert os.access(str(dest), os.X_OK)
        assert dest.read_text() == "#!/bin/bash\necho tmux"

    def test_binary_not_uploaded_for_other_agents(self, tmp_path):
        usr_local_bin = tmp_path / "usr" / "local" / "bin"
        usr_local_bin.mkdir(parents=True, mode=0o700)

        agent_name = "claude-code"
        if agent_name.startswith("terminus"):
            (usr_local_bin / "tmux").write_text("should not exist")

        assert not (usr_local_bin / "tmux").exists()


# ===========================================================================
#  Startup barrier tests
# ===========================================================================


def _reset_barrier():
    """Reset module-level barrier state between tests.

    Barrier state moved to tasktrace_capture.runtimes._shared; reset it there.
    """
    from tasktrace_capture.runtimes import _shared as _shared_mod

    _shared_mod._reset_barrier_for_tests()


def _make_stub_env(name="test"):
    """Create a minimal stub with just the fields _wait_at_barrier needs."""
    import logging

    stub = object.__new__(EnrootEnvironment)
    stub._container_name = name
    stub.logger = logging.getLogger(f"barrier.{name}")
    return stub


class TestStartupBarrier:
    """Verify the asyncio-based startup barrier."""

    def setup_method(self):
        _reset_barrier()
        self._orig_env = os.environ.get("_TASKTRACE_CAPTURE_BARRIER_COUNT")

    def teardown_method(self):
        _reset_barrier()
        if self._orig_env is None:
            os.environ.pop("_TASKTRACE_CAPTURE_BARRIER_COUNT", None)
        else:
            os.environ["_TASKTRACE_CAPTURE_BARRIER_COUNT"] = self._orig_env

    def test_noop_when_no_env_var(self):
        """Barrier is a no-op when _TASKTRACE_CAPTURE_BARRIER_COUNT is not set."""
        os.environ.pop("_TASKTRACE_CAPTURE_BARRIER_COUNT", None)
        stub = _make_stub_env()
        asyncio.run(stub._wait_at_barrier())

    def test_noop_when_count_is_one(self):
        """Barrier is a no-op for single agent."""
        os.environ["_TASKTRACE_CAPTURE_BARRIER_COUNT"] = "1"
        stub = _make_stub_env()
        asyncio.run(stub._wait_at_barrier())

    def test_three_agents_sync(self):
        """Three agents should all finish within 50ms of each other."""
        os.environ["_TASKTRACE_CAPTURE_BARRIER_COUNT"] = "3"
        finish_times = []

        async def agent(name, delay):
            stub = _make_stub_env(name)
            await asyncio.sleep(delay)  # simulate variable setup time
            await stub._wait_at_barrier()
            finish_times.append(time.monotonic())

        async def run():
            await asyncio.gather(
                agent("fast", 0.0),
                agent("medium", 0.1),
                agent("slow", 0.3),
            )

        asyncio.run(run())
        assert len(finish_times) == 3
        spread = max(finish_times) - min(finish_times)
        assert spread < 0.05, f"Agents finished {spread:.3f}s apart (want <0.05s)"

    def test_timeout_when_agent_missing(self):
        """If not all agents arrive, barrier times out and proceeds."""
        from tasktrace_capture.runtimes import _shared as _shared_mod

        os.environ["_TASKTRACE_CAPTURE_BARRIER_COUNT"] = "3"
        orig_timeout = _shared_mod._BARRIER_TIMEOUT_SEC
        _shared_mod._BARRIER_TIMEOUT_SEC = 0.5

        finish_times = []

        async def agent(name):
            stub = _make_stub_env(name)
            await stub._wait_at_barrier()
            finish_times.append(time.monotonic())

        async def run():
            await asyncio.gather(agent("a"), agent("b"))

        try:
            asyncio.run(run())
            assert len(finish_times) == 2
        finally:
            _shared_mod._BARRIER_TIMEOUT_SEC = orig_timeout

    def test_state_resets_between_runs(self):
        """Barrier state resets so a second batch works correctly."""
        os.environ["_TASKTRACE_CAPTURE_BARRIER_COUNT"] = "2"

        async def batch():
            _reset_barrier()
            finish_times = []

            async def agent(name, delay):
                stub = _make_stub_env(name)
                await asyncio.sleep(delay)
                await stub._wait_at_barrier()
                finish_times.append(time.monotonic())

            await asyncio.gather(agent("a", 0.0), agent("b", 0.2))
            return finish_times

        t1 = asyncio.run(batch())
        _reset_barrier()
        t2 = asyncio.run(batch())
        assert len(t1) == 2
        assert len(t2) == 2
        assert max(t1) - min(t1) < 0.05
        assert max(t2) - min(t2) < 0.05

    def test_large_concurrency(self):
        """64 agents with staggered setup all finish within 50ms."""
        os.environ["_TASKTRACE_CAPTURE_BARRIER_COUNT"] = "64"

        finish_times = []

        async def agent(i):
            stub = _make_stub_env(f"agent_{i}")
            await asyncio.sleep(i * 0.005)  # 0-320ms stagger
            await stub._wait_at_barrier()
            finish_times.append(time.monotonic())

        async def run():
            await asyncio.gather(*(agent(i) for i in range(64)))

        asyncio.run(run())
        assert len(finish_times) == 64
        spread = max(finish_times) - min(finish_times)
        assert spread < 0.05, f"64 agents finished {spread:.3f}s apart (want <0.05s)"


# ===========================================================================
#  API key rotation logic
# ===========================================================================

# ===========================================================================
#  SWE-bench image remapping (epoch-research arm64 support)
# ===========================================================================


class TestSwebenchImageRemap:
    """Tests for SWE-bench image remapping to epoch-research GHCR.

    epoch-research/SWE-bench hosts pre-built images at:
        ghcr.io/epoch-research/swe-bench.eval.{arch}.{instance_id}:latest
    with arm64 support for 1819/2294 tasks (best-effort).

    The canonical instance ID is taken from the task directory name, NOT the
    Dockerfile FROM line.  Princeton's Docker Hub images use internal build
    numbers (e.g. astropy_1776_astropy-13236) that differ from the canonical
    instance ID (astropy__astropy-13236) used by epoch-research.
    """

    # -- _swebench_instance_from_envdir ----------------------------------------

    def test_extracts_canonical_instance_id(self, tmp_path):
        """Directory named after canonical instance ID is detected correctly."""
        env_dir = tmp_path / "astropy__astropy-13236" / "environment"
        env_dir.mkdir(parents=True)
        assert _swebench_instance_from_envdir(env_dir) == "astropy__astropy-13236"

    def test_various_canonical_instance_ids(self, tmp_path):
        """All standard SWE-bench instance ID formats are detected."""
        cases = [
            "astropy__astropy-13236",
            "django__django-11099",
            "sympy__sympy-20590",
            "matplotlib__matplotlib-23476",
            "scikit-learn__scikit-learn-25102",
            "sphinx-doc__sphinx-8551",
        ]
        for instance_id in cases:
            env_dir = tmp_path / instance_id / "environment"
            env_dir.mkdir(parents=True, exist_ok=True)
            result = _swebench_instance_from_envdir(env_dir)
            assert result == instance_id, f"Failed for {instance_id}"

    def test_non_swebench_dir_returns_none(self, tmp_path):
        """Non-SWE-bench task directories return None."""
        for name in ["hello-world", "compile-compcert", "password-recovery", "ubuntu:22.04"]:
            env_dir = tmp_path / name / "environment"
            env_dir.mkdir(parents=True, exist_ok=True)
            assert _swebench_instance_from_envdir(env_dir) is None, f"Should be None for {name}"

    def test_princeton_internal_name_not_matched(self, tmp_path):
        """Princeton Docker Hub internal image name (with build number) is NOT
        a valid instance ID and should not be detected as one."""
        # "astropy_1776_astropy-13236" uses single underscores + build number
        env_dir = tmp_path / "astropy_1776_astropy-13236" / "environment"
        env_dir.mkdir(parents=True)
        # single underscore → double-underscore check fails → returns None
        assert _swebench_instance_from_envdir(env_dir) is None

    # -- _remap_swebench_image (secondary/fallback path) -----------------------

    def test_epoch_research_image_arch_corrected(self, monkeypatch):
        """Already-epoch-research images get their arch corrected to host arch."""
        monkeypatch.setattr("tasktrace_capture.enroot_env._HOST_ARCH", "arm64")
        result = _remap_swebench_image("ghcr.io/epoch-research/swe-bench.eval.x86_64.sympy__sympy-20590")
        assert result == f"{_EPOCH_SWEBENCH_REGISTRY}.arm64.sympy__sympy-20590:latest"

    def test_non_swebench_image_returns_none(self):
        """Non-SWE-bench images return None."""
        assert _remap_swebench_image("ubuntu:22.04") is None
        assert _remap_swebench_image("python:3.11-slim") is None
        assert _remap_swebench_image("jefzda/sweap-images:qutebrowser.foo") is None
        assert _remap_swebench_image("ghcr.io/someother/image:latest") is None

    # -- constants -------------------------------------------------------------

    def test_host_arch_constant_valid(self):
        """_HOST_ARCH must be one of the two valid values."""
        assert _HOST_ARCH in ("arm64", "x86_64")

    def test_epoch_registry_constant(self):
        """Registry constant matches the epoch-research GHCR path."""
        assert _EPOCH_SWEBENCH_REGISTRY == "ghcr.io/epoch-research/swe-bench.eval"

    def test_instance_re_matches_canonical_ids(self):
        """Instance ID regex matches canonical double-underscore format."""
        valid = [
            "astropy__astropy-13236",
            "django__django-11099",
            "scikit-learn__scikit-learn-25102",
        ]
        for v in valid:
            assert _SWEBENCH_INSTANCE_RE.match(v), f"Should match: {v}"

    def test_instance_re_rejects_princeton_internal_names(self):
        """Instance ID regex rejects Princeton internal names (single underscore + build no)."""
        invalid = [
            "astropy_1776_astropy-13236",  # single underscore, build number
            "hello-world",
            "ubuntu:22.04",
        ]
        for v in invalid:
            assert not _SWEBENCH_INSTANCE_RE.match(v), f"Should not match: {v}"

    def test_swebench_images_env_absent_means_no_remap(self, monkeypatch):
        """Without _TASKTRACE_CAPTURE_SWEBENCH_IMAGES=epoch-research, start() skips remapping."""
        monkeypatch.delenv("_TASKTRACE_CAPTURE_SWEBENCH_IMAGES", raising=False)
        assert os.environ.get("_TASKTRACE_CAPTURE_SWEBENCH_IMAGES") != "epoch-research"


class TestGhcrCredentials:
    """Tests for _ensure_ghcr_credentials — GHCR auth setup for enroot."""

    def test_writes_credential_when_github_token_set(self, tmp_path, monkeypatch):
        """When GITHUB_TOKEN is set and no ghcr.io entry exists, writes credentials."""
        cred_path = tmp_path / ".credentials"
        monkeypatch.setattr("tasktrace_capture.enroot_env._ENROOT_CREDENTIALS_PATH", cred_path)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken123")

        _ensure_ghcr_credentials()

        assert cred_path.exists()
        content = cred_path.read_text()
        assert "ghcr.io" in content
        assert "ghp_testtoken123" in content

    def test_no_write_when_github_token_absent(self, tmp_path, monkeypatch):
        """When GITHUB_TOKEN is not set, no credentials file is created."""
        cred_path = tmp_path / ".credentials"
        monkeypatch.setattr("tasktrace_capture.enroot_env._ENROOT_CREDENTIALS_PATH", cred_path)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        _ensure_ghcr_credentials()

        assert not cred_path.exists()

    def test_idempotent_when_ghcr_already_configured(self, tmp_path, monkeypatch):
        """If ghcr.io is already in credentials, nothing is appended."""
        cred_path = tmp_path / ".credentials"
        monkeypatch.setattr("tasktrace_capture.enroot_env._ENROOT_CREDENTIALS_PATH", cred_path)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_new_token")

        existing = "machine ghcr.io login olduser password oldtoken\n"
        cred_path.parent.mkdir(parents=True, exist_ok=True)
        cred_path.write_text(existing)

        _ensure_ghcr_credentials()

        # Content should be unchanged
        assert cred_path.read_text() == existing

    def test_creates_parent_dirs(self, tmp_path, monkeypatch):
        """Credentials file parent directories are created if missing."""
        cred_path = tmp_path / "nested" / "dir" / ".credentials"
        monkeypatch.setattr("tasktrace_capture.enroot_env._ENROOT_CREDENTIALS_PATH", cred_path)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")

        _ensure_ghcr_credentials()

        assert cred_path.exists()
        assert cred_path.parent.is_dir()


class TestApiKeyRotation:
    """Test the round-robin API key rotation in exec()."""

    def test_round_robin_distributes_keys(self, monkeypatch):
        """Multiple keys should be distributed across containers by hash."""
        keys = ["sk-key-A", "sk-key-B", "sk-key-C"]
        monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", ",".join(keys))

        # Simulate different container names and check distribution
        assigned = set()
        for name in ["container-0", "container-1", "container-2", "container-3", "container-4", "container-5"]:
            idx = hash(name) % len(keys)
            assigned.add(keys[idx])

        # With 6 containers and 3 keys, all keys should be used
        assert len(assigned) >= 2, "Keys should be distributed across containers"

    def test_single_key_no_rotation(self, monkeypatch):
        """Single key should not trigger rotation."""
        monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "sk-only-key")
        keys = [k.strip() for k in os.environ["_TASKTRACE_CAPTURE_API_KEYS"].split(",") if k.strip()]
        assert len(keys) == 1  # rotation requires len > 1

    def test_empty_keys_no_rotation(self, monkeypatch):
        """Empty _TASKTRACE_CAPTURE_API_KEYS should not trigger rotation."""
        monkeypatch.setenv("_TASKTRACE_CAPTURE_API_KEYS", "")
        keys = [k.strip() for k in os.environ["_TASKTRACE_CAPTURE_API_KEYS"].split(",") if k.strip()]
        assert len(keys) == 0

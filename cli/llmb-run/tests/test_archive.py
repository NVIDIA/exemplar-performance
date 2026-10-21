# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import tarfile
from unittest.mock import patch

import pytest
import zstandard


def _read_tar_zst(archive_path: pathlib.Path) -> list[tarfile.TarInfo]:
    """Decompress and return all tar members from a .tar.zst file."""
    dctx = zstandard.ZstdDecompressor()
    with archive_path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                return list(tar)


class TestArchiveModule:
    def test_build_archive_file_list_filters_and_sorts(self, tmp_path):
        from llmb_run.archive import build_archive_file_list

        llmb_install = tmp_path / "install"
        experiments_a = llmb_install / "workloads" / "alpha" / "experiments"
        experiments_b = llmb_install / "workloads" / "beta" / "experiments"
        non_exp = llmb_install / "workloads" / "alpha" / "scratch"

        (experiments_a / "run2").mkdir(parents=True)
        (experiments_a / "run1").mkdir(parents=True)
        experiments_b.mkdir(parents=True)
        non_exp.mkdir(parents=True)

        (experiments_a / "run2" / "b.log").write_text("b")
        (experiments_a / "run1" / "a.log").write_text("a")
        (experiments_b / "x.out").write_text("x")
        (experiments_b / "train_data_step75.jsonl").write_text("train rollout")
        (experiments_b / "results.jsonl").write_text("results")
        (experiments_b / "skip.nsys-rep").write_text("profile")
        (experiments_b / "rank109_trace.json").write_text("pytorch profile")
        (experiments_b / "rank109.pt.trace.json").write_text("new pytorch profile")
        (experiments_b / "metrics.json.gz").write_text("keep compressed json")
        (experiments_b / "data.tar.gz").write_text("tarball")
        (experiments_a / "run1" / "dataset_1000_1000_100000.txt").write_text("generated dataset")
        nsys_profile_dir = experiments_a / "run1" / "nsys_profile"
        nsys_profile_dir.mkdir()
        (nsys_profile_dir / "rank0.sqlite").write_text("sqlite")
        (nsys_profile_dir / "large.csv").write_text("csv")
        (nsys_profile_dir / "summary.html").write_text("html")
        torch_profile_dir = experiments_a / "run2" / "torch_profile"
        torch_profile_dir.mkdir()
        (torch_profile_dir / "rank-0.json.gz").write_text("pytorch profile")
        pytorch_profile_dir = experiments_a / "run2" / "pytorch_profile"
        pytorch_profile_dir.mkdir()
        (pytorch_profile_dir / "rank-0.json.gz").write_text("pytorch profile")
        (experiments_a / "run1" / "code").mkdir()
        (experiments_a / "run1" / "code" / "train.py").write_text("code")
        checkpoint_dir = experiments_a / "run2" / "checkpoints" / "iter_0000050"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "__0_0.distcp").write_text("checkpoint shard")
        (non_exp / "ignore.log").write_text("no")

        entries = build_archive_file_list(llmb_install)
        arcpaths = [arc for _, arc in entries]

        assert arcpaths == sorted(arcpaths)
        assert "alpha/experiments/run1/a.log" in arcpaths
        assert "alpha/experiments/run2/b.log" in arcpaths
        assert "beta/experiments/x.out" in arcpaths
        assert "beta/experiments/metrics.json.gz" in arcpaths
        assert "beta/experiments/results.jsonl" not in arcpaths
        assert "beta/experiments/train_data_step75.jsonl" not in arcpaths
        assert "beta/experiments/skip.nsys-rep" not in arcpaths
        assert "beta/experiments/rank109_trace.json" not in arcpaths
        assert "beta/experiments/rank109.pt.trace.json" not in arcpaths
        assert "beta/experiments/data.tar.gz" not in arcpaths
        assert "alpha/experiments/run1/dataset_1000_1000_100000.txt" not in arcpaths
        assert "alpha/experiments/run2/checkpoints/iter_0000050/__0_0.distcp" not in arcpaths
        assert all("code/" not in path for path in arcpaths)
        assert all("checkpoints/" not in path for path in arcpaths)
        assert all("nsys_profile/" not in path for path in arcpaths)
        assert all("torch_profile/" not in path for path in arcpaths)
        assert all("pytorch_profile/" not in path for path in arcpaths)
        assert all("scratch" not in path for path in arcpaths)

    def test_build_archive_file_list_microbenchmark_nccl(self, tmp_path):
        """TODO(temporary): remove when microbenchmark_nccl aligns with standard layout."""
        from llmb_run.archive import build_archive_file_list

        llmb_install = tmp_path / "install"
        workload_dir = llmb_install / "workloads" / "microbenchmark_nccl"
        experiments_dir = workload_dir / "experiments"
        exp_name = "microbenchmark_nccl_nccl_fp8_container-nccl_user_123"
        log_dir = experiments_dir / exp_name / "LOG_run"
        log_dir.mkdir(parents=True)

        # Top-level workload files (non-standard location for this recipe)
        (workload_dir / "llmb-config_123.yaml").write_text("config")
        (workload_dir / "slurm-123.out").write_text("slurm output")
        (workload_dir / "Nvidia-Comms-Perf-Suite").mkdir()  # should be ignored

        # Experiment log files
        (log_dir / "result.txt").write_text("data")
        (log_dir / "result.yml").write_text("data")

        # Tar files in experiments (should be excluded)
        (experiments_dir / f"{exp_name}.tar.gz").write_text("tar")

        entries = build_archive_file_list(llmb_install)
        arcpaths = [arc for _, arc in entries]

        assert "microbenchmark_nccl/llmb-config_123.yaml" in arcpaths
        assert "microbenchmark_nccl/slurm-123.out" in arcpaths
        assert f"microbenchmark_nccl/experiments/{exp_name}/LOG_run/result.txt" in arcpaths
        assert f"microbenchmark_nccl/experiments/{exp_name}/LOG_run/result.yml" in arcpaths
        assert not any("tar.gz" in p for p in arcpaths)
        assert not any("Nvidia-Comms-Perf-Suite" in p for p in arcpaths)

    def test_default_archive_output(self, tmp_path):
        from llmb_run.archive import default_archive_output

        out = default_archive_output(tmp_path, "20260209T010203Z", "26.06.00-dev")
        assert out == tmp_path / "llmb-archive_26.06.00-dev_20260209T010203Z.tar.zst"

    def test_create_tar_zst_structure_and_exclusions(self, tmp_path):
        from llmb_run.archive import create_tar_zst

        source = tmp_path / "src"
        (source / "alpha" / "experiments" / "run").mkdir(parents=True)
        keep_file = source / "alpha" / "experiments" / "run" / "keep.log"
        keep_file.write_text("hello")
        config_file = source / "alpha" / "experiments" / "run" / "llmb-config_12345.yaml"
        config_file.write_text("config")

        out = tmp_path / "archive.tar.zst"
        stats = create_tar_zst(
            out,
            [
                (keep_file, "alpha/experiments/run/keep.log"),
                (config_file, "alpha/experiments/run/llmb-config_12345.yaml"),
            ],
            "20260209T111111Z",
        )

        assert stats.experiment_count == 1
        assert stats.timestamp == "20260209T111111Z"

        names = [m.name for m in _read_tar_zst(out)]

        assert "llmb-archive-20260209T111111Z" in names
        assert "llmb-archive-20260209T111111Z/alpha/experiments/run/keep.log" in names

    def test_create_tar_zst_empty_file_list(self, tmp_path):
        from llmb_run.archive import create_tar_zst

        out = tmp_path / "empty.tar.zst"
        stats = create_tar_zst(out, [], "20260209T000000Z")

        assert stats.experiment_count == 0
        assert stats.experiment_count == 0

        members = _read_tar_zst(out)

        assert len(members) == 1
        assert members[0].name == "llmb-archive-20260209T000000Z"
        assert members[0].isdir()

    def test_create_tar_zst_preserves_symlinks(self, tmp_path):
        from llmb_run.archive import create_tar_zst

        source = tmp_path / "src"
        (source / "experiments").mkdir(parents=True)
        target = source / "experiments" / "real.log"
        target.write_text("data")
        link = source / "experiments" / "link.log"
        link.symlink_to("real.log")

        out = tmp_path / "symlink.tar.zst"
        stats = create_tar_zst(
            out,
            [(link, "alpha/experiments/link.log")],
            "20260209T222222Z",
        )

        assert stats.experiment_count == 0

        members = {m.name: m for m in _read_tar_zst(out)}

        link_member = members["llmb-archive-20260209T222222Z/alpha/experiments/link.log"]
        assert link_member.issym()
        assert link_member.linkname == "real.log"

    def test_run_archive_existing_output_errors(self, tmp_path, cluster_config_factory):
        from llmb_run.archive import run_archive

        llmb_install = tmp_path / "install"
        llmb_install.mkdir(parents=True)
        existing = tmp_path / "already.tar.zst"
        existing.write_text("x")

        cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

        with pytest.raises(ValueError, match="already exists"):
            run_archive(cluster_config, str(existing))

    def test_run_archive_default_output_includes_release_version(self, tmp_path, cluster_config_factory):
        from llmb_run.archive import run_archive

        llmb_install = tmp_path / "install"
        repo = tmp_path / "repo"
        llmb_install.mkdir(parents=True)
        repo.mkdir()
        (repo / "release.yaml").write_text("llmb_version: '26.06.00-dev'\n")

        cluster_config = cluster_config_factory(
            {
                "launcher": {
                    "llmb_install": str(llmb_install),
                    "llmb_repo": str(repo),
                }
            }
        )

        with patch("llmb_run.archive.utc_timestamp", return_value="20260209T010203Z"):
            stats = run_archive(cluster_config, None)

        assert stats.output_path == llmb_install / "llmb-archive_26.06.00-dev_20260209T010203Z.tar.zst"
        assert stats.output_path.exists()

    def test_run_archive_default_output_requires_release_version(self, tmp_path, cluster_config_factory):
        from llmb_run.archive import run_archive

        llmb_install = tmp_path / "install"
        repo = tmp_path / "repo"
        llmb_install.mkdir(parents=True)
        repo.mkdir()
        cluster_config = cluster_config_factory(
            {"launcher": {"llmb_install": str(llmb_install), "llmb_repo": str(repo)}}
        )

        with pytest.raises(ValueError, match=f"Unable to determine llmb_version.*{repo}"):
            run_archive(cluster_config, None)


class TestArchiveCli:
    @staticmethod
    def _cluster_config(llmb_install: pathlib.Path):
        from llmb_run.config_manager import ClusterConfig

        return ClusterConfig.from_dict(
            {
                "launcher": {
                    "llmb_repo": "/tmp",
                    "llmb_install": str(llmb_install),
                    "gpu_type": "h100",
                },
                "slurm": {
                    "gpu_partition": "batch",
                    "gpu_gres": "8",
                },
                "workloads": {"installed": ["alpha"]},
            }
        )

    @pytest.mark.usefixtures("mock_dependencies")
    def test_archive_command_help_shows_versioned_default(self, runner):
        from llmb_run.main import app

        result = runner.invoke(app, ["archive", "--help"], terminal_width=140)

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert "llmb-archive_<release_version>_<timestamp>.tar.zst" in result.stdout

    @pytest.mark.usefixtures("mock_dependencies")
    def test_archive_command_with_output(self, runner, tmp_path):
        from llmb_run.main import app

        llmb_install = tmp_path / "install"
        exp_dir = llmb_install / "workloads" / "alpha" / "experiments" / "runA"
        exp_dir.mkdir(parents=True)
        (exp_dir / "result.log").write_text("ok")
        (exp_dir / "profile.nsys-rep").write_text("drop")

        output_path = tmp_path / "custom.tar.zst"

        with patch("llmb_run.main.get_cluster_config", return_value=self._cluster_config(llmb_install)):
            result = runner.invoke(app, ["archive", "--output", str(output_path)])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert output_path.exists()

        names = [m.name for m in _read_tar_zst(output_path)]

        matched = [name for name in names if name.endswith("/alpha/experiments/runA/result.log")]
        assert matched, f"Expected archived result file, got names: {names}"
        assert all(not name.endswith(".nsys-rep") for name in names)

    @pytest.mark.usefixtures("mock_dependencies")
    def test_archive_command_errors_when_output_exists(self, runner, tmp_path):
        from llmb_run.main import app

        llmb_install = tmp_path / "install"
        (llmb_install / "workloads" / "alpha" / "experiments").mkdir(parents=True)
        output_path = tmp_path / "exists.tar.zst"
        output_path.write_text("existing")

        with patch("llmb_run.main.get_cluster_config", return_value=self._cluster_config(llmb_install)):
            result = runner.invoke(app, ["archive", "--output", str(output_path)])

        assert result.exit_code == 1, f"Exit code: {result.exit_code}, Output: {result.stdout}"

    def test_archive_does_not_load_workloads(self, runner, tmp_path):
        from llmb_run.main import app

        llmb_install = tmp_path / "install"
        (llmb_install / "workloads" / "alpha" / "experiments").mkdir(parents=True)
        output_path = tmp_path / "ok.tar.zst"

        with (
            patch("llmb_run.main.get_cluster_config", return_value=self._cluster_config(llmb_install)),
            patch("llmb_run.main.get_workloads", side_effect=RuntimeError("should not load workloads")),
        ):
            result = runner.invoke(app, ["archive", "--output", str(output_path)])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
        assert output_path.exists()

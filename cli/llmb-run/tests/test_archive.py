import datetime
import os
import tarfile

import pytest
import zstandard

from llmb_run.archive import (
    build_archive_file_list,
    create_tar_zst,
    default_archive_output,
    utc_timestamp,
)


def touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def arcnames(install):
    return [arc for _, arc in build_archive_file_list(install)]


def test_utc_timestamp_format_for_given_time():
    assert utc_timestamp(datetime.datetime(2026, 3, 4, 5, 6, 7, tzinfo=datetime.timezone.utc)) == "20260304T050607Z"


def test_default_archive_output_path(tmp_path):
    out = default_archive_output(tmp_path, "20260101T000000Z", "26.04")
    assert out == tmp_path / "llmb-archive_26.04_20260101T000000Z.tar.zst"


def test_missing_workloads_dir_gives_empty_list(tmp_path):
    assert build_archive_file_list(tmp_path) == []


def test_arcname_layout_and_sorting(tmp_path):
    touch(tmp_path / "workloads/wl_b/experiments/exp1/llmb-config_2.yaml")
    touch(tmp_path / "workloads/wl_a/experiments/exp1/sub/out.log")
    touch(tmp_path / "workloads/wl_a/experiments/exp0/slurm-1.out")
    assert arcnames(tmp_path) == [
        "wl_a/experiments/exp0/slurm-1.out",
        "wl_a/experiments/exp1/sub/out.log",
        "wl_b/experiments/exp1/llmb-config_2.yaml",
    ]


def test_entries_carry_absolute_source_paths(tmp_path):
    src = touch(tmp_path / "workloads/wl/experiments/e/log.txt")
    assert build_archive_file_list(tmp_path) == [(src, "wl/experiments/e/log.txt")]


def test_workload_without_experiments_dir_and_stray_files_are_ignored(tmp_path):
    touch(tmp_path / "workloads/wl/venv/lib.py")
    touch(tmp_path / "workloads/stray.txt")
    assert build_archive_file_list(tmp_path) == []


@pytest.mark.parametrize("dirname", ["code", "checkpoints", "nsys_profile", "torch_profile", "pytorch_profile"])
def test_excluded_dirs_are_pruned_at_any_depth(tmp_path, dirname):
    exp = tmp_path / "workloads/wl/experiments"
    touch(exp / "e1" / dirname / "big.bin")
    touch(exp / "e1/nested" / dirname / "deep/big.bin")
    touch(exp / "e1/keep.txt")
    assert arcnames(tmp_path) == ["wl/experiments/e1/keep.txt"]


def test_excluded_dir_name_match_is_exact(tmp_path):
    touch(tmp_path / "workloads/wl/experiments/e/code_review/a.txt")
    touch(tmp_path / "workloads/wl/experiments/e/my_checkpoints/b.txt")
    assert len(arcnames(tmp_path)) == 2


@pytest.mark.parametrize(
    "name",
    [
        "run.nsys-rep",
        "rank0_trace.json",
        "x.pt.trace.json",
        "logs.tar.gz",
        "logs.tar.zst",
        "dataset_1000_1000_8.txt",
    ],
)
def test_excluded_file_patterns(tmp_path, name):
    exp = tmp_path / "workloads/wl/experiments/e"
    touch(exp / name)
    touch(exp / "keep.json")
    assert arcnames(tmp_path) == ["wl/experiments/e/keep.json"]


@pytest.mark.parametrize("name", ["trace.json", "dataset_512_512_1.txt", "archive.tar", "nsys-rep"])
def test_near_miss_names_are_kept(tmp_path, name):
    touch(tmp_path / "workloads/wl/experiments/e" / name)
    assert arcnames(tmp_path) == [f"wl/experiments/e/{name}"]


def test_symlinked_dir_is_archived_as_link_not_followed(tmp_path):
    exp = tmp_path / "workloads/wl/experiments/e"
    touch(tmp_path / "elsewhere/secret.txt")
    exp.mkdir(parents=True)
    os.symlink(tmp_path / "elsewhere", exp / "linkdir")
    entries = build_archive_file_list(tmp_path)
    assert [arc for _, arc in entries] == ["wl/experiments/e/linkdir"]
    assert entries[0][0].is_symlink()


def test_symlink_named_like_excluded_dir_is_dropped(tmp_path):
    exp = tmp_path / "workloads/wl/experiments/e"
    (tmp_path / "target").mkdir()
    exp.mkdir(parents=True)
    os.symlink(tmp_path / "target", exp / "checkpoints")
    assert build_archive_file_list(tmp_path) == []


def test_nccl_top_level_patterns_included(tmp_path):
    wl = tmp_path / "workloads/microbenchmark_nccl"
    touch(wl / "llmb-config_55.yaml")
    touch(wl / "slurm-55.out")
    touch(wl / "other.txt")
    touch(wl / "experiments/e/log.txt")
    assert arcnames(tmp_path) == [
        "microbenchmark_nccl/experiments/e/log.txt",
        "microbenchmark_nccl/llmb-config_55.yaml",
        "microbenchmark_nccl/slurm-55.out",
    ]


def test_nccl_special_case_does_not_apply_to_other_workloads(tmp_path):
    touch(tmp_path / "workloads/pretrain_x/llmb-config_55.yaml")
    touch(tmp_path / "workloads/pretrain_x/slurm-55.out")
    assert build_archive_file_list(tmp_path) == []


def test_create_tar_zst_roundtrip(tmp_path):
    src_a = touch(tmp_path / "a.txt", "alpha")
    src_b = touch(tmp_path / "b.yaml", "beta")
    out = tmp_path / "out.tar.zst"
    entries = [
        (src_a, "wl/experiments/e/llmb-config_1.yaml"),
        (src_b, "wl/experiments/e/llmb-config_2.yaml"),
        (tmp_path / "a.txt", "wl/experiments/e/log.txt"),
    ]
    stats = create_tar_zst(out, entries, "20260101T000000Z")

    assert stats.output_path == out
    assert stats.timestamp == "20260101T000000Z"
    assert stats.experiment_count == 2
    with out.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            names = [m.name for m in tar]
    assert names[0] == "llmb-archive-20260101T000000Z"
    assert names[1:] == [f"llmb-archive-20260101T000000Z/{arc}" for _, arc in entries]


def test_create_tar_zst_empty_still_has_root_folder(tmp_path):
    out = tmp_path / "out.tar.zst"
    stats = create_tar_zst(out, [], "T")
    assert stats.experiment_count == 0
    with out.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            assert [m.name for m in tar] == ["llmb-archive-T"]

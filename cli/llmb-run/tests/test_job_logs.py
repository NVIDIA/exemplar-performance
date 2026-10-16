import pytest

from llmb_run.job_logs import (
    JobLogFile,
    active_job_log,
    find_configured_sbatch_logs,
    find_job_logs,
    read_tail,
)


def test_find_job_logs_ignores_non_matching_and_directories(tmp_path):
    (tmp_path / "log-a_10_0.out").write_text("x")
    (tmp_path / "log-a_10_0.err").write_text("x")
    (tmp_path / "slurm-10.out").write_text("x")
    (tmp_path / "log-a_10_x.out").write_text("x")
    (tmp_path / "log-dir_10_1.out").mkdir()
    assert [f.path.name for f in find_job_logs(tmp_path, 10)] == ["log-a_10_0.out"]


def test_find_job_logs_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_job_logs(tmp_path / "nope", 1)


def test_configured_sbatch_logs_prefers_workload_logs(tmp_path):
    (tmp_path / "log-a_4_0.out").write_text("x")
    (tmp_path / "slurm-4.out").write_text("x")
    assert [f.path.name for f in find_configured_sbatch_logs(tmp_path, 4)] == ["log-a_4_0.out"]


def test_configured_sbatch_logs_falls_back_to_slurm_stdout(tmp_path):
    (tmp_path / "slurm-4.out").write_text("x")
    logs = find_configured_sbatch_logs(tmp_path, 4)
    assert logs == [JobLogFile(path=tmp_path / "slurm-4.out")]
    assert logs[0].retry is None


def test_configured_sbatch_logs_empty_when_nothing(tmp_path):
    assert find_configured_sbatch_logs(tmp_path, 4) == []


def test_active_job_log():
    assert active_job_log([]) is None
    logs = [JobLogFile(path="a", retry=0), JobLogFile(path="b", retry=1)]
    assert active_job_log(logs).path == "b"


def test_read_tail_returns_last_lines(tmp_path):
    p = tmp_path / "f.log"
    p.write_text("1\n2\n3\n4\n")
    assert read_tail(p, 2) == "3\n4"
    assert read_tail(p, 100) == "1\n2\n3\n4"


def test_read_tail_rejects_non_positive_count(tmp_path):
    p = tmp_path / "f.log"
    p.write_text("1\n")
    with pytest.raises(ValueError, match="at least 1"):
        read_tail(p, 0)

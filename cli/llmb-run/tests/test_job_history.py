import sqlite3
import sys

import pytest

from llmb_run import job_history
from llmb_run.job_history import (
    DB_SCHEMA_VERSION,
    JobRecord,
    _initialize_schema,
    base_slurm_state,
    format_job_details,
    format_jobs_table,
    get_job,
    is_terminal_state,
    job_record_from_config,
    list_jobs,
    upsert_static_job,
)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def columns(conn):
    return {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}


def stored_version(conn):
    return conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]


def test_fresh_schema_has_all_columns_and_current_version(conn):
    _initialize_schema(conn)
    assert {"tokens_per_sec_per_gpu", "perf_parse_status", "train_step_time_seconds", "tflops_per_gpu"} <= columns(conn)
    assert stored_version(conn) == str(DB_SCHEMA_VERSION)


def test_initialize_schema_is_idempotent(conn):
    _initialize_schema(conn)
    conn.execute("INSERT INTO jobs (job_id, launcher_type) VALUES (1, 'nemo')")
    _initialize_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_v1_to_v2_migration_adds_column_and_keeps_rows(conn):
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
    conn.execute("""
        CREATE TABLE jobs (
            job_id INTEGER PRIMARY KEY, launcher_type TEXT NOT NULL, workload_key TEXT,
            profile_enabled INTEGER NOT NULL DEFAULT 0, proxy INTEGER NOT NULL DEFAULT 0,
            train_step_time_seconds REAL, tflops_per_gpu REAL, perf_parse_status TEXT
        )
        """)
    conn.execute("INSERT INTO jobs (job_id, launcher_type, workload_key, tflops_per_gpu) VALUES (7, 'nemo', 'w', 12.5)")
    assert "tokens_per_sec_per_gpu" not in columns(conn)

    _initialize_schema(conn)

    assert "tokens_per_sec_per_gpu" in columns(conn)
    assert stored_version(conn) == str(DB_SCHEMA_VERSION)
    row = conn.execute("SELECT * FROM jobs WHERE job_id = 7").fetchone()
    assert row["tflops_per_gpu"] == 12.5
    assert row["tokens_per_sec_per_gpu"] is None


def test_newer_schema_version_is_refused_and_left_untouched(conn):
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO metadata VALUES ('schema_version', ?)", (str(DB_SCHEMA_VERSION + 1),))
    with pytest.raises(RuntimeError, match="newer than this llmb-run"):
        _initialize_schema(conn)
    assert stored_version(conn) == str(DB_SCHEMA_VERSION + 1)
    assert not conn.execute("SELECT name FROM sqlite_master WHERE name='jobs'").fetchall()


def test_unparseable_schema_version_is_refused(conn):
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO metadata VALUES ('schema_version', 'abc')")
    with pytest.raises(RuntimeError, match="unparseable"):
        _initialize_schema(conn)


@pytest.mark.parametrize(
    "state, base, terminal",
    [
        (None, "", False),
        ("RUNNING", "RUNNING", False),
        ("completed", "COMPLETED", True),
        ("CANCELLED by 1234", "CANCELLED", True),
        ("  TIMEOUT ", "TIMEOUT", True),
        ("PURGED", "PURGED", True),
        ("REQUEUED", "REQUEUED", False),
    ],
)
def test_state_helpers(state, base, terminal):
    assert base_slurm_state(state) == base
    assert is_terminal_state(state) is terminal


def record(job_id, **kw):
    defaults = {
        "launcher_type": "nemo",
        "workload_key": "pretrain_a",
        "model_name": "llama",
        "model_size": "70b",
        "dtype": "fp8",
        "scale": 128,
        "submit_time": "2026-01-02T03:04:05+00:00",
    }
    defaults.update(kw)
    return JobRecord(job_id=job_id, **defaults)


def test_upsert_and_get_job_round_trip(make_cluster_config):
    cfg = make_cluster_config()
    upsert_static_job(cfg, record(11, profile_enabled=True))
    row = get_job(cfg, 11)
    assert (row["workload_key"], row["dtype"], row["scale"], row["profile_enabled"]) == ("pretrain_a", "fp8", 128, 1)
    assert get_job(cfg, 12) is None


def test_upsert_keeps_original_submit_time(make_cluster_config):
    cfg = make_cluster_config()
    upsert_static_job(cfg, record(11))
    upsert_static_job(cfg, record(11, submit_time="2030-01-01T00:00:00+00:00", scale=256))
    row = get_job(cfg, 11)
    assert row["scale"] == 256
    assert row["submit_time"] == "2026-01-02T03:04:05+00:00"


def seeded_rows(make_cluster_config):
    cfg = make_cluster_config()
    upsert_static_job(cfg, record(101, profile_enabled=True))
    upsert_static_job(cfg, record(102, workload_key="pretrain_b", model_size="7b", dtype="bf16", scale=8))
    with job_history._open_history_db(cfg) as conn:
        conn.execute(
            "UPDATE jobs SET slurm_state='COMPLETED', elapsed='00:10:00', train_step_time_seconds=1.234, "
            "tflops_per_gpu=456.789, perf_parse_status='success' WHERE job_id=101"
        )
        conn.execute(
            "UPDATE jobs SET slurm_state='CANCELLED by 5', perf_parse_status='invalid_grad_norm', "
            "train_step_time_seconds=9.0, tflops_per_gpu=9.0 WHERE job_id=102"
        )
        conn.commit()
    return list_jobs(cfg)


def test_format_jobs_table_empty_hint():
    assert "llmb-run jobs rebuild" in format_jobs_table([], {})


def test_format_jobs_table_plain_text_content(make_cluster_config, monkeypatch):
    # Colour is keyed off the real stdout; force the piped case so `pytest -s` agrees.
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    out = format_jobs_table(seeded_rows(make_cluster_config), {})
    assert "\x1b[" not in out
    header = out.splitlines()[0]
    for col in ("Workload", "DType", "Scale", "Job ID", "s/iter", "TFLOPS/GPU"):
        assert col in header
    assert "Tokens/s/GPU" not in header
    line_101 = next(line for line in out.splitlines() if "101" in line)
    assert "pretrain_a_70b" in line_101 and "fp8" in line_101 and "Yes" in line_101
    assert "2026-01-02 03:04" in line_101
    assert "1.23" in line_101 and "456.79" in line_101 and "COMPLETED" in line_101


def test_format_jobs_table_masks_invalid_grad_norm_metrics(make_cluster_config):
    out = format_jobs_table(seeded_rows(make_cluster_config), {})
    line_102 = next(line for line in out.splitlines() if "102" in line)
    assert line_102.count("Invalid") == 2
    assert "9.00" not in line_102
    assert "CANCELLED by" not in line_102


def test_format_jobs_table_shows_tokens_column_only_for_rl_workloads(make_cluster_config):
    rows = seeded_rows(make_cluster_config)
    rl_workloads = {"pretrain_b": {"metadata": {"general": {"framework": "nemo-rl"}}}}
    assert "Tokens/s/GPU" in format_jobs_table(rows, rl_workloads)
    assert "Tokens/s/GPU" not in format_jobs_table(rows, {})


def test_format_job_details_layout(make_cluster_config):
    rows = seeded_rows(make_cluster_config)
    text = format_job_details(next(r for r in rows if r["job_id"] == 101), {})
    lines = text.splitlines()
    assert any(line.startswith("Job ID") and line.endswith(": 101") for line in lines)
    assert any(line.startswith("Profile") and line.endswith(": Yes") for line in lines)
    assert "TFLOPS/GPU" in text and "456.79" in text
    assert "Tokens/s/GPU" not in text
    assert "s/iter" in text and "1.23" in text


def test_format_job_details_invalid_grad_norm_label(make_cluster_config):
    rows = seeded_rows(make_cluster_config)
    text = format_job_details(next(r for r in rows if r["job_id"] == 102), {})
    assert "invalid: grad_norm=nan" in text


def write_llmb_config(install, workload, job_id, body):
    path = install / "workloads" / workload / "experiments" / "e" / f"llmb-config_{job_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


WL = {
    "pretrain_a": {"metadata": {"run": {"launcher_type": "nemo"}}},
    "old": {"metadata": {"run": {"launcher_type": "sbatch"}}},
}


def test_job_record_from_config_reads_fields_and_workload_from_path(tmp_path):
    path = write_llmb_config(
        tmp_path,
        "pretrain_a",
        "555",
        "job_info: {job_id: 555, launch_time: '2026-01-01T00:00:00'}\n"
        "model_info: {model_name: llama, model_size: 70b, dtype: fp8, scale: '64'}\n"
        "job_config: {profile_enabled: true, env_overrides: {A: '1'}}\n",
    )
    rec = job_record_from_config(tmp_path, path, WL)
    assert (rec.job_id, rec.workload_key, rec.launcher_type, rec.scale) == (555, "pretrain_a", "nemo", 64)
    assert rec.submit_time == "2026-01-01T00:00:00"  # legacy launch_time alias
    assert rec.profile_enabled is True and rec.proxy is False
    assert rec.env_overrides_json == '{"A": "1"}'


def test_job_record_from_config_falls_back_to_job_id_in_filename(tmp_path):
    path = write_llmb_config(tmp_path, "pretrain_a", "777", "model_info: {dtype: fp8}\n")
    assert job_record_from_config(tmp_path, path, WL).job_id == 777


@pytest.mark.parametrize(
    "workload, body",
    [
        ("pretrain_a", ""),
        ("pretrain_a", "a: [unclosed\n"),
        ("unknown_wl", "job_info: {job_id: 1}\n"),
        ("old", "job_info: {job_id: 1}\n"),
    ],
)
def test_job_record_from_config_skips_unusable_configs(tmp_path, workload, body):
    path = write_llmb_config(tmp_path, workload, "1", body)
    assert job_record_from_config(tmp_path, path, WL) is None


def test_job_record_from_config_skips_unparseable_job_id(tmp_path):
    path = write_llmb_config(tmp_path, "pretrain_a", "notanumber", "job_info: {}\n")
    assert job_record_from_config(tmp_path, path, WL) is None

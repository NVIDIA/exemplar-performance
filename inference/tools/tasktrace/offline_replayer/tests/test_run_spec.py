# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from offline_replayer.runs.spec import (
    OrderedRunSpec,
    RunSpec,
    RunSpecError,
    load_run_spec,
)


def _payload(tmp_path: Path) -> dict:
    (tmp_path / "one.json").write_text('{"task": 1}\n')
    (tmp_path / "two.json").write_text('{"task": 2}\n')
    return {
        "schema_version": "tasktrace-run.v1",
        "output": "results",
        "cpus": "23,20-22,22",
        "tasks": [
            {"id": "one", "atif": "one.json", "image": "image-one"},
            {"id": "two", "atif": "two.json", "image": "image-two"},
        ],
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _load(tmp_path: Path, mutate=None):
    payload = _payload(tmp_path)
    if mutate is not None:
        mutate(payload)
    return load_run_spec(_write(tmp_path, payload))


def _v3_payload(tmp_path: Path) -> dict:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    one = inputs / "one.json"
    two = inputs / "two.json"
    one.write_text('{"task": 1}\n')
    two.write_text('{"task": 2}\n')
    return {
        "schema_version": "tasktrace-run.v3",
        "output": "results/v3",
        "defaults": {
            "pacing": "llm-gaps",
            "network_policy": "public",
            "profile_sets": ["cpu-average"],
            "controller_runtime": "host-mount",
            "controller_cpus": "3,0-2",
            "action_timeout_sec": 900,
        },
        "tasks": [
            {
                "id": "one",
                "atif": "inputs/one.json",
                "expected_source_sha256": hashlib.sha256(one.read_bytes()).hexdigest(),
                "image": "image-one@sha256:one",
                "expected_image_id": "sha256:one",
            },
            {
                "id": "two",
                "atif": "inputs/two.json",
                "expected_source_sha256": hashlib.sha256(two.read_bytes()).hexdigest(),
                "image": "image-two@sha256:two",
                "expected_image_id": "sha256:two",
                "action_timeout_sec": 1200,
            },
        ],
        "waves": [
            {
                "id": "r1-n1",
                "scored": True,
                "metadata": {"repeat": 1, "allocation": 1},
                "launches": [
                    {"task": "one", "cpus": "4"},
                    {"task": "two", "cpus": "4"},
                ],
            },
            {
                "id": "r2-n2",
                "scored": False,
                "metadata": {"repeat": 2, "allocation": 2},
                "launches": [
                    {"task": "two", "cpus": "6"},
                    {"task": "one", "cpus": "5"},
                ],
            },
        ],
    }


def _load_v3(tmp_path: Path, mutate=None):
    payload = _v3_payload(tmp_path)
    if mutate is not None:
        mutate(payload)
    return load_run_spec(_write(tmp_path, payload))


def _compact_v3_payload(tmp_path: Path) -> dict:
    payload = _v3_payload(tmp_path)
    payload["defaults"].update(
        {
            "worker_cpus": "6,4-5",
            "placement": "round-robin-single-cpu",
        }
    )
    payload["waves"] = [
        {"id": "r1-n1", "allocation": 1, "repeat": 1},
        {
            "id": "r2-n2",
            "allocation": 2,
            "repeat": 2,
            "scored": False,
        },
    ]
    return payload


def _load_compact_v3(tmp_path: Path, mutate=None):
    payload = _compact_v3_payload(tmp_path)
    if mutate is not None:
        mutate(payload)
    return load_run_spec(_write(tmp_path, payload))


def test_loads_minimal_run_with_defaults_and_preserves_task_order(tmp_path: Path) -> None:
    spec = _load(tmp_path)

    assert spec.schema_version == "tasktrace-run.v1"
    assert spec.manifest_path == (tmp_path / "run.yaml").resolve()
    assert spec.manifest_sha256 == hashlib.sha256(spec.manifest_path.read_bytes()).hexdigest()
    assert spec.output == (tmp_path / "results").resolve()
    assert spec.cpus == "20-23"
    assert spec.cpu_ids == (20, 21, 22, 23)
    assert spec.concurrency == 1
    assert spec.repetitions == 1
    assert [task.task_id for task in spec.tasks] == ["one", "two"]
    assert spec.tasks[0].atif == (tmp_path / "one.json").resolve()
    assert spec.tasks[0].source_sha256 == hashlib.sha256((tmp_path / "one.json").read_bytes()).hexdigest()
    assert spec.tasks[0].expected_image_id is None
    assert spec.tasks[0].action_timeout_sec is None

    positional_v1 = RunSpec(
        spec.schema_version,
        spec.manifest_path,
        spec.manifest_sha256,
        spec.output,
        spec.cpus,
        spec.cpu_ids,
        spec.repetitions,
        spec.tasks,
    )
    assert positional_v1.concurrency == 1


def test_loads_optional_repetitions_and_task_assertions(tmp_path: Path) -> None:
    source = tmp_path / "one.json"
    payload = _payload(tmp_path)
    payload["repetitions"] = 3
    payload["tasks"] = [
        {
            "id": "task.one",
            "atif": "one.json",
            "image": "image@sha256:abc",
            "expected_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper(),
            "expected_image_id": "sha256:resolved",
            "action_timeout_sec": 12.5,
        }
    ]

    spec = load_run_spec(_write(tmp_path, payload))

    assert spec.repetitions == 3
    assert spec.tasks[0].expected_image_id == "sha256:resolved"
    assert spec.tasks[0].action_timeout_sec == 12.5


def test_loads_v2_with_required_concurrency(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["schema_version"] = "tasktrace-run.v2"
    payload["concurrency"] = 2

    spec = load_run_spec(_write(tmp_path, payload))

    assert spec.schema_version == "tasktrace-run.v2"
    assert spec.concurrency == 2
    assert spec.repetitions == 1


def test_v2_requires_concurrency(tmp_path: Path) -> None:
    with pytest.raises(RunSpecError, match="run is missing: concurrency"):
        _load(
            tmp_path,
            lambda payload: payload.__setitem__("schema_version", "tasktrace-run.v2"),
        )


@pytest.mark.parametrize("concurrency", [True, 0, -1, 1.5, "3"])
def test_v2_rejects_invalid_concurrency(tmp_path: Path, concurrency: object) -> None:
    payload = _payload(tmp_path)
    payload["schema_version"] = "tasktrace-run.v2"
    payload["concurrency"] = concurrency

    with pytest.raises(RunSpecError, match="concurrency must be a positive integer"):
        load_run_spec(_write(tmp_path, payload))


def test_v2_rejects_concurrency_above_declared_attempt_count(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["schema_version"] = "tasktrace-run.v2"
    payload["concurrency"] = 3

    with pytest.raises(RunSpecError, match="cannot exceed.*task-attempt count"):
        load_run_spec(_write(tmp_path, payload))


def test_loads_v3_ordered_waves_with_explicit_launches(tmp_path: Path) -> None:
    spec = _load_v3(tmp_path)

    assert isinstance(spec, OrderedRunSpec)
    assert spec.schema_version == "tasktrace-run.v3"
    assert spec.manifest_path == (tmp_path / "run.yaml").resolve()
    assert spec.manifest_sha256 == hashlib.sha256(spec.manifest_path.read_bytes()).hexdigest()
    assert spec.output == (tmp_path / "results/v3").resolve()
    assert spec.input_root == tmp_path.resolve()
    assert spec.defaults.runtime == "docker"
    assert spec.defaults.pacing == "llm-gaps"
    assert spec.defaults.network_policy == "public"
    assert spec.defaults.allowed_hosts == ()
    assert spec.defaults.profile_sets == ("cpu-average",)
    assert spec.defaults.controller_runtime == "host-mount"
    assert spec.defaults.controller_cpus == "0-3"
    assert spec.defaults.controller_cpu_ids == (0, 1, 2, 3)
    assert spec.defaults.action_timeout_sec == 900.0
    assert [task.task_id for task in spec.tasks] == ["one", "two"]
    assert spec.tasks[0].atif == (tmp_path / "inputs/one.json").resolve()
    assert spec.tasks[0].action_timeout_sec is None
    assert spec.tasks[1].action_timeout_sec == 1200.0
    assert [wave.wave_id for wave in spec.waves] == ["r1-n1", "r2-n2"]
    assert spec.waves[0].scored is True
    assert spec.waves[1].scored is False
    assert spec.waves[0].metadata == (("repeat", 1), ("allocation", 1))
    assert [launch.task_id for launch in spec.waves[0].launches] == ["one", "two"]
    assert [launch.cpus for launch in spec.waves[0].launches] == ["4", "4"]
    assert [launch.cpu_ids for launch in spec.waves[1].launches] == [(6,), (5,)]


def test_v3_compact_waves_expand_to_explicit_round_robin_launches(
    tmp_path: Path,
) -> None:
    spec = _load_compact_v3(tmp_path)

    assert isinstance(spec, OrderedRunSpec)
    assert spec.defaults.worker_cpus == "4-6"
    assert spec.defaults.worker_cpu_ids == (4, 5, 6)
    assert spec.defaults.placement == "round-robin-single-cpu"
    assert [wave.wave_id for wave in spec.waves] == ["r1-n1", "r2-n2"]
    assert spec.waves[0].scored is True
    assert spec.waves[1].scored is False
    assert spec.waves[0].metadata == (("allocation", 1), ("repeat", 1))
    assert spec.waves[1].metadata == (("allocation", 2), ("repeat", 2))
    assert [launch.task_id for launch in spec.waves[0].launches] == [
        "one",
        "two",
    ]
    assert [launch.cpus for launch in spec.waves[0].launches] == ["4", "4"]
    assert [launch.cpus for launch in spec.waves[1].launches] == ["4", "5"]
    assert [launch.cpu_ids for launch in spec.waves[1].launches] == [(4,), (5,)]
    assert all(len(launch.cpu_ids) == 1 for wave in spec.waves for launch in wave.launches)


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda payload: payload["defaults"].pop("placement"),
            "worker_cpus and defaults.placement must be specified together",
        ),
        (
            lambda payload: payload["defaults"].pop("worker_cpus"),
            "worker_cpus and defaults.placement must be specified together",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("placement", "packed"),
            "defaults.placement must be 'round-robin-single-cpu'",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("worker_cpus", "2-4"),
            "defaults.worker_cpus overlaps defaults.controller_cpus: 2,3",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("worker_cpus", "worker-four"),
            "invalid defaults.worker_cpus",
        ),
    ],
)
def test_v3_rejects_invalid_compact_defaults(tmp_path: Path, mutate, expected: str) -> None:
    with pytest.raises(RunSpecError, match=expected):
        _load_compact_v3(tmp_path, mutate)


@pytest.mark.parametrize("allocation", [True, 0, -1, 1.5, "2"])
def test_v3_rejects_invalid_compact_allocation(tmp_path: Path, allocation: object) -> None:
    with pytest.raises(RunSpecError, match="allocation must be a positive integer"):
        _load_compact_v3(
            tmp_path,
            lambda payload: payload["waves"][0].__setitem__("allocation", allocation),
        )


@pytest.mark.parametrize("repeat", [True, 0, -1, 1.5, "2"])
def test_v3_rejects_invalid_compact_repeat(tmp_path: Path, repeat: object) -> None:
    with pytest.raises(RunSpecError, match="repeat must be a positive integer"):
        _load_compact_v3(
            tmp_path,
            lambda payload: payload["waves"][0].__setitem__("repeat", repeat),
        )


def test_v3_rejects_compact_allocation_above_worker_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RunSpecError,
        match=r"allocation \(4\) exceeds.*worker_cpus \(3\)",
    ):
        _load_compact_v3(
            tmp_path,
            lambda payload: payload["waves"][0].__setitem__("allocation", 4),
        )


def test_v3_rejects_mixed_compact_and_explicit_wave_forms(tmp_path: Path) -> None:
    explicit_wave = _v3_payload(tmp_path)["waves"][0]

    with pytest.raises(RunSpecError, match="cannot be combined with explicit waves"):
        _load_compact_v3(
            tmp_path,
            lambda payload: payload["waves"].append(explicit_wave),
        )

    with pytest.raises(RunSpecError, match="compact waves require defaults.worker_cpus"):
        _load_v3(
            tmp_path,
            lambda payload: payload["waves"].append({"id": "compact", "allocation": 1, "repeat": 1}),
        )


def test_v3_compact_wave_rejects_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(RunSpecError, match="waves\\[0\\] has unknown fields: note"):
        _load_compact_v3(
            tmp_path,
            lambda payload: payload["waves"][0].__setitem__("note", "nope"),
        )


def test_v3_input_root_centralizes_relative_atif_paths(tmp_path: Path) -> None:
    payload = _v3_payload(tmp_path)
    source_root = tmp_path / "portable-inputs"
    source_root.mkdir()
    for task in payload["tasks"]:
        source = tmp_path / task["atif"]
        target = source_root / Path(task["atif"]).name
        target.write_bytes(source.read_bytes())
        task["atif"] = target.name
    payload["input_root"] = "portable-inputs"

    spec = load_run_spec(_write(tmp_path, payload))

    assert isinstance(spec, OrderedRunSpec)
    assert spec.input_root == source_root.resolve()
    assert [task.atif.parent for task in spec.tasks] == [
        source_root.resolve(),
        source_root.resolve(),
    ]


def test_v3_enroot_accepts_no_network_and_anchors_local_images(
    tmp_path: Path,
) -> None:
    payload = _v3_payload(tmp_path)
    payload["defaults"].update({"runtime": "enroot", "network_policy": "no-network", "profile_sets": []})
    for task in payload["tasks"]:
        task["image"] = f"images/{task['id']}.sqsh"

    spec = load_run_spec(_write(tmp_path, payload))

    assert spec.defaults.runtime == "enroot"
    for removed_field in (
        "enroot_cpu_enforcement",
        "prepare_timeout_sec",
        "replay_timeout_sec",
        "cache_policy",
    ):
        assert not hasattr(spec.defaults, removed_field)
    assert spec.defaults.network_policy == "no-network"
    assert spec.defaults.profile_sets == ()
    assert [task.image for task in spec.tasks] == [
        str((tmp_path / "images/one.sqsh").resolve()),
        str((tmp_path / "images/two.sqsh").resolve()),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enroot_cpu_enforcement", "slurm-step-cpuset"),
        ("prepare_timeout_sec", 300),
        ("replay_timeout_sec", 1800),
        ("cache_policy", "warm-host-caches-fresh-sandbox"),
    ],
)
def test_v3_rejects_removed_enroot_experiment_defaults(tmp_path: Path, field: str, value: object) -> None:
    def mutate(payload: dict) -> None:
        payload["defaults"].update(
            {
                "runtime": "enroot",
                "network_policy": "no-network",
                "profile_sets": [],
                field: value,
            }
        )

    with pytest.raises(
        RunSpecError,
        match=f"defaults has unknown fields: {field}",
    ):
        _load_v3(tmp_path, mutate)


def test_v3_normalizes_allowlist_hosts_and_preserves_profile_order(
    tmp_path: Path,
) -> None:
    def mutate(payload: dict) -> None:
        payload["defaults"].update(
            {
                "network_policy": "allowlist",
                "allowed_hosts": ["B.EXAMPLE.COM", "a.example.com", "b.example.com"],
                "profile_sets": ["cpu-average", "hardware-deep"],
            }
        )

    spec = _load_v3(tmp_path, mutate)

    assert isinstance(spec, OrderedRunSpec)
    assert spec.defaults.network_policy == "allowlist"
    assert spec.defaults.allowed_hosts == ("a.example.com", "b.example.com")
    assert spec.defaults.profile_sets == ("cpu-average", "hardware-deep")


@pytest.mark.parametrize("field", ["cpus", "concurrency", "repetitions", "runtime"])
def test_v3_rejects_legacy_or_unknown_root_fields(tmp_path: Path, field: str) -> None:
    with pytest.raises(RunSpecError, match="run has unknown fields"):
        _load_v3(tmp_path, lambda payload: payload.__setitem__(field, 1))


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda payload: payload["defaults"].__setitem__("pacing", "fast-ish"),
            "defaults.pacing must be one of",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("runtime", "podman"),
            "defaults.runtime must be one of",
        ),
        (
            lambda payload: payload["defaults"].update({"runtime": "enroot", "network_policy": "allowlist"}),
            "network_policy must be public or no-network",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("runtime", "enroot"),
            "profile_sets must be empty for enroot runtime",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("network_policy", "private"),
            "invalid defaults.network_policy",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("allowed_hosts", "example.com"),
            "defaults.allowed_hosts must be a list",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("profile_sets", []),
            "defaults.profile_sets must be a non-empty list",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("profile_sets", ["cpu-average", "cpu-average"]),
            "must not contain duplicates",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("profile_sets", ["hardware-deep"]),
            "must include cpu-average",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("controller_runtime", "local"),
            "defaults.controller_runtime must be one of",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("controller_cpus", "three"),
            "invalid defaults.controller_cpus",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("action_timeout_sec", 0),
            "defaults.action_timeout_sec",
        ),
        (
            lambda payload: payload["defaults"].__setitem__("extra", True),
            "defaults has unknown fields",
        ),
    ],
)
def test_v3_rejects_invalid_defaults(tmp_path: Path, mutate, expected: str) -> None:
    with pytest.raises(RunSpecError, match=expected):
        _load_v3(tmp_path, mutate)


def test_v3_rejects_public_policy_with_allowed_hosts(tmp_path: Path) -> None:
    with pytest.raises(RunSpecError, match="allowed hosts require mode='allowlist'"):
        _load_v3(
            tmp_path,
            lambda payload: payload["defaults"].__setitem__("allowed_hosts", ["example.com"]),
        )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda payload: payload["waves"][1].__setitem__("id", "r1-n1"),
            "duplicate wave id",
        ),
        (
            lambda payload: payload["waves"][0].__setitem__("scored", 1),
            "scored must be a boolean",
        ),
        (
            lambda payload: payload["waves"][0].__setitem__("metadata", {"dimensions": {"repeat": 1}}),
            "metadata.dimensions must be",
        ),
        (
            lambda payload: payload["waves"][0].__setitem__("metadata", {"bad key": 1}),
            "metadata key must be a safe",
        ),
        (
            lambda payload: payload["waves"][0].__setitem__("extra", True),
            "waves\\[0\\] has unknown fields",
        ),
        (
            lambda payload: payload["waves"][0]["launches"][0].__setitem__("extra", True),
            "launches\\[0\\] has unknown fields",
        ),
        (
            lambda payload: payload["waves"][0]["launches"][0].__setitem__("task", "missing"),
            "references unknown task",
        ),
        (
            lambda payload: payload["waves"][0]["launches"][1].__setitem__("task", "one"),
            "launches task more than once",
        ),
        (
            lambda payload: payload["waves"][0]["launches"][0].__setitem__("cpus", "2"),
            "overlaps defaults.controller_cpus",
        ),
        (
            lambda payload: payload["waves"][0]["launches"][0].__setitem__("cpus", "cpu-four"),
            "invalid waves\\[0\\].launches\\[0\\].cpus",
        ),
    ],
)
def test_v3_rejects_invalid_waves(tmp_path: Path, mutate, expected: str) -> None:
    with pytest.raises(RunSpecError, match=expected):
        _load_v3(tmp_path, mutate)


def test_v3_rejects_empty_waves_launches_and_unreferenced_catalog_tasks(
    tmp_path: Path,
) -> None:
    with pytest.raises(RunSpecError, match="waves must be a non-empty list"):
        _load_v3(tmp_path, lambda payload: payload.__setitem__("waves", []))
    with pytest.raises(RunSpecError, match="launches must be a non-empty list"):
        _load_v3(
            tmp_path,
            lambda payload: payload["waves"][0].__setitem__("launches", []),
        )

    def leave_one_unreferenced(payload: dict) -> None:
        for wave in payload["waves"]:
            wave["launches"] = [launch for launch in wave["launches"] if launch["task"] != "two"]

    with pytest.raises(RunSpecError, match="unreferenced tasks: two"):
        _load_v3(tmp_path, leave_one_unreferenced)


@pytest.mark.parametrize("field", ["runtime", "concurrency", "strategy", "name", "measure"])
def test_rejects_unknown_root_fields(tmp_path: Path, field: str) -> None:
    with pytest.raises(RunSpecError, match="unknown fields"):
        _load(tmp_path, lambda payload: payload.__setitem__(field, "nope"))


def test_rejects_unknown_task_fields(tmp_path: Path) -> None:
    with pytest.raises(RunSpecError, match="unknown fields"):
        _load(
            tmp_path,
            lambda payload: payload["tasks"][0].__setitem__("format", "atif"),
        )


@pytest.mark.parametrize("field", ["schema_version", "output", "cpus", "tasks"])
def test_rejects_missing_required_root_fields(tmp_path: Path, field: str) -> None:
    with pytest.raises(RunSpecError, match="is missing"):
        _load(tmp_path, lambda payload: payload.pop(field))


def test_rejects_wrong_schema(tmp_path: Path) -> None:
    with pytest.raises(RunSpecError, match="unsupported run schema"):
        _load(tmp_path, lambda payload: payload.__setitem__("schema_version", "other"))


@pytest.mark.parametrize("schema_version", [[], {}])
def test_rejects_non_text_schema_without_type_error(tmp_path: Path, schema_version: object) -> None:
    with pytest.raises(RunSpecError, match="schema_version must be a non-empty trimmed string"):
        _load(
            tmp_path,
            lambda payload: payload.__setitem__("schema_version", schema_version),
        )


@pytest.mark.parametrize(
    ("duplicate", "expected"),
    [
        (
            """\
schema_version: tasktrace-run.v1
output: results
cpus: 20-23
cpus: 24-27
tasks: []
""",
            "duplicate key 'cpus'",
        ),
        (
            """\
schema_version: tasktrace-run.v1
output: results
cpus: 20-23
tasks:
  - id: one
    atif: one.json
    image: first
    image: second
""",
            "duplicate key 'image'",
        ),
    ],
)
def test_rejects_duplicate_yaml_keys(tmp_path: Path, duplicate: str, expected: str) -> None:
    (tmp_path / "one.json").write_text("{}\n")
    manifest = tmp_path / "run.yaml"
    manifest.write_text(duplicate)

    with pytest.raises(RunSpecError, match=expected):
        load_run_spec(manifest)


@pytest.mark.parametrize("cpus", ["", "4-2", "1,,2", "one"])
def test_rejects_invalid_cpuset(tmp_path: Path, cpus: str) -> None:
    with pytest.raises(RunSpecError, match="cpus"):
        _load(tmp_path, lambda payload: payload.__setitem__("cpus", cpus))


@pytest.mark.parametrize("repetitions", [True, 0, -1, 1.5, "3"])
def test_rejects_invalid_repetitions(tmp_path: Path, repetitions: object) -> None:
    with pytest.raises(RunSpecError, match="positive integer"):
        _load(
            tmp_path,
            lambda payload: payload.__setitem__("repetitions", repetitions),
        )


def test_rejects_duplicate_or_unsafe_task_ids(tmp_path: Path) -> None:
    with pytest.raises(RunSpecError, match="duplicate task id"):
        _load(tmp_path, lambda payload: payload["tasks"][1].__setitem__("id", "one"))
    with pytest.raises(RunSpecError, match="safe"):
        _load(tmp_path, lambda payload: payload["tasks"][0].__setitem__("id", "../one"))


def test_rejects_missing_atif_and_source_digest_mismatch(tmp_path: Path) -> None:
    with pytest.raises(RunSpecError, match="is not a file"):
        _load(tmp_path, lambda payload: payload["tasks"][0].__setitem__("atif", "missing"))
    with pytest.raises(RunSpecError, match="SHA-256 mismatch"):
        _load(
            tmp_path,
            lambda payload: payload["tasks"][0].__setitem__("expected_source_sha256", "0" * 64),
        )


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf"), "10"])
def test_rejects_invalid_task_timeout(tmp_path: Path, timeout: object) -> None:
    with pytest.raises(RunSpecError, match="action_timeout_sec"):
        _load(
            tmp_path,
            lambda payload: payload["tasks"][0].__setitem__("action_timeout_sec", timeout),
        )

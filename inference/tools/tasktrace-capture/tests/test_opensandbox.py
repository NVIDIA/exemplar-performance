# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from unittest import mock

import pytest
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.config import ResourceMode
from harbor.models.trial.paths import TrialPaths
from opensandbox.models import NetworkPolicy as OpenSandboxNetworkPolicy
from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage
from tasktrace_capture.opensandbox_env import (
    OpenSandboxEnvironment,
    _add_allowed_hosts,
    _allowed_targets_digest,
    _connection_config,
    _inference_endpoint_host,
    _network_policy,
    _setup_allowed_hosts,
)
from tasktrace_capture.runtimes import (
    runtime_cleanup,
    runtime_import_path,
    runtime_preflight_checks,
)


@pytest.fixture(autouse=True)
def opensandbox_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENSANDBOX_URL", "http://sandbox.example.test/")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")
    monkeypatch.setenv("TASKTRACE_CAPTURE_OPENSANDBOX_USER", "test-user")
    for name in (
        "INFERENCE_BASE_URL",
        "AGENT_API_BASE",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_BASE",
    ):
        monkeypatch.delenv(name, raising=False)


def _make_environment(
    tmp_path: Path,
    *,
    docker_image: str | None = "ubuntu:22.04",
    network_policy: NetworkPolicy | None = None,
    workdir: str | None = None,
    cpus: int | None = None,
    memory_mb: int | None = None,
    **environment_kwargs,
) -> OpenSandboxEnvironment:
    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True)
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    return OpenSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="test-task",
        session_id="test-task__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image=docker_image,
            build_timeout_sec=180,
            workdir=workdir,
            cpus=cpus,
            memory_mb=memory_mb,
        ),
        logger=logging.getLogger("test.opensandbox"),
        network_policy=network_policy,
        **environment_kwargs,
    )


class FakeCommands:
    def __init__(self):
        self.calls = []
        self.result = Execution(
            exit_code=0,
            logs=ExecutionLogs(
                stdout=[OutputMessage(text="ok\n", timestamp=1)],
                stderr=[],
            ),
        )

    async def run(self, command, *, opts=None, handlers=None):
        self.calls.append((command, opts, handlers))
        return self.result


class FakeFiles:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.write_calls = []

    async def write_file(self, path, data, **kwargs):
        self.write_calls.append((path, data, kwargs))
        self.files[path] = data if isinstance(data, bytes) else str(data).encode()

    async def read_bytes(self, path, **kwargs):
        return self.files[path]


class FakeSandbox:
    def __init__(self):
        self.id = "sandbox-123"
        self.commands = FakeCommands()
        self.files = FakeFiles()
        self.renewed = []
        self.destroyed = False

    async def renew(self, timeout):
        self.renewed.append(timeout)

    async def destroy(self):
        self.destroyed = True


def test_runtime_import_path_resolves():
    assert runtime_import_path("opensandbox") == ("tasktrace_capture.opensandbox_env:OpenSandboxEnvironment")


def test_connection_config_uses_cell_proxy():
    config = _connection_config()
    assert config.domain == "sandbox.example.test"
    assert config.protocol == "http"
    assert config.use_server_proxy is True
    assert config.request_timeout == timedelta(seconds=180)
    assert config.headers["OPEN-SANDBOX-API-KEY"] == "test-key"


def test_connection_config_request_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TASKTRACE_CAPTURE_OPENSANDBOX_REQUEST_TIMEOUT_SECONDS", "240")

    config = _connection_config()

    assert config.request_timeout == timedelta(seconds=240)


def test_runtime_preflight_requires_controller_credentials(monkeypatch):
    checks = runtime_preflight_checks("opensandbox")
    assert [ok for _, ok, _ in checks] == [True, True]

    monkeypatch.delenv("OPENSANDBOX_API_KEY")
    checks = runtime_preflight_checks("opensandbox")
    assert [ok for _, ok, _ in checks] == [True, False]


def test_runtime_cleanup_is_noop():
    runtime_cleanup("opensandbox")


@pytest.mark.parametrize("missing", ["OPENSANDBOX_URL", "OPENSANDBOX_API_KEY"])
def test_connection_config_requires_credentials(monkeypatch, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match=missing):
        _connection_config()


def test_network_policy_translation():
    public = _network_policy(NetworkPolicy(network_mode=NetworkMode.PUBLIC))
    assert public.default_action == "allow"
    assert public.egress == []

    denied = _network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    assert denied.default_action == "deny"
    assert denied.egress == []

    allowlist = _network_policy(
        NetworkPolicy(
            network_mode=NetworkMode.ALLOWLIST,
            allowed_hosts=["api.example.com"],
        )
    )
    assert allowlist.default_action == "deny"
    assert [(rule.action, rule.target) for rule in allowlist.egress] == [("allow", "api.example.com")]


def test_no_network_allows_only_configured_nvidia_inference(monkeypatch):
    monkeypatch.setenv("INFERENCE_BASE_URL", "https://api.example.com/v1")

    policy = _network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))

    assert policy.default_action == "deny"
    assert [(rule.action, rule.target) for rule in policy.egress] == [("allow", "api.example.com")]


def test_no_network_supports_self_hosted_inference_endpoint(monkeypatch):
    monkeypatch.setenv("INFERENCE_BASE_URL", "http://10.42.18.7:8000/v1")

    policy = _network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))

    assert [(rule.action, rule.target) for rule in policy.egress] == [("allow", "10.42.18.7")]


def test_no_network_supports_resolved_agent_api_base(monkeypatch):
    monkeypatch.setenv("AGENT_API_BASE", "https://terminus.example.test/v1")

    policy = _network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))

    assert [(rule.action, rule.target) for rule in policy.egress] == [("allow", "terminus.example.test")]


def test_task_allowlist_is_combined_with_inference_endpoint(monkeypatch):
    monkeypatch.setenv("INFERENCE_BASE_URL", "https://inference.example.test/v1")

    policy = _network_policy(
        NetworkPolicy(
            network_mode=NetworkMode.ALLOWLIST,
            allowed_hosts=["artifacts.example.test"],
        )
    )

    assert [rule.target for rule in policy.egress] == [
        "artifacts.example.test",
        "inference.example.test",
    ]


def test_public_network_is_not_narrowed_by_inference_endpoint(monkeypatch):
    monkeypatch.setenv("INFERENCE_BASE_URL", "https://inference.example.test/v1")

    policy = _network_policy(NetworkPolicy(network_mode=NetworkMode.PUBLIC))

    assert policy.default_action == "allow"
    assert policy.egress == []


@pytest.mark.parametrize(
    "value, message",
    [
        ("inference.example.test/v1", "absolute http"),
        ("https://user:secret@inference.example.test/v1", "credentials"),
        ("https://inference.example.test:bad/v1", "invalid port"),
        ("https://inference.example.test/v1#secret", "fragment"),
    ],
)
def test_inference_endpoint_rejects_unsafe_or_ambiguous_urls(monkeypatch, value, message):
    monkeypatch.setenv("INFERENCE_BASE_URL", value)

    with pytest.raises(ValueError, match=message):
        _inference_endpoint_host()


def test_canonical_inference_endpoint_wins_over_provider_alias(monkeypatch):
    monkeypatch.setenv("INFERENCE_BASE_URL", "https://selected.example.test/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example.test/v1")

    assert _inference_endpoint_host() == "selected.example.test"


def test_resolved_agent_api_base_wins_over_service_default(monkeypatch):
    monkeypatch.setenv("AGENT_API_BASE", "https://selected.example.test/v1")
    monkeypatch.setenv("INFERENCE_BASE_URL", "https://default.example.test/v1")

    assert _inference_endpoint_host() == "selected.example.test"


def test_setup_allowlist_is_limited_to_supported_agents(monkeypatch):
    assert "astral.sh" in _setup_allowed_hosts("mini-swe-agent")
    assert "releases.astral.sh" in _setup_allowed_hosts("mini-swe-agent")
    assert "downloads.claude.ai" in _setup_allowed_hosts("claude-code")
    assert "deb.nodesource.com" in _setup_allowed_hosts("codex")
    assert "nodejs.org" in _setup_allowed_hosts("codex")
    assert "registry.npmjs.org" in _setup_allowed_hosts("codex")
    assert _setup_allowed_hosts("nop") == []

    monkeypatch.setenv(
        "TASKTRACE_CAPTURE_OPENSANDBOX_SETUP_ALLOWED_HOSTS",
        "packages.example.test, packages.example.test, releases.example.test",
    )
    assert _setup_allowed_hosts("mini-swe-agent") == [
        "packages.example.test",
        "releases.example.test",
    ]


def test_setup_hosts_are_added_only_to_restricted_policy():
    denied = OpenSandboxNetworkPolicy(defaultAction="deny", egress=[])
    setup = _add_allowed_hosts(denied, ["astral.sh", "pypi.org"])
    assert [rule.target for rule in setup.egress] == ["astral.sh", "pypi.org"]

    public = OpenSandboxNetworkPolicy(defaultAction="allow", egress=[])
    assert _add_allowed_hosts(public, ["astral.sh"]) is public


def test_allowed_target_digest_is_order_independent():
    assert _allowed_targets_digest(["b.example", "a.example", "b.example"]) == (
        _allowed_targets_digest(["a.example", "b.example"])
    )
    assert len(_allowed_targets_digest(["a.example"])) == 32


def test_environment_capabilities_are_explicit(tmp_path):
    environment = _make_environment(tmp_path)
    assert environment.capabilities.disable_internet is True
    assert environment.capabilities.network_allowlist is True
    assert environment.capabilities.mounted is False
    assert environment.capabilities.gpus is False


def test_dockerfile_only_accepts_from_and_workdir(tmp_path):
    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:22.04\nWORKDIR /app\n", encoding="utf-8")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = OpenSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="test-task",
        session_id="test-task__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(build_timeout_sec=180),
        logger=logging.getLogger("test.opensandbox"),
    )
    assert environment._image_from_dockerfile == "ubuntu:22.04"
    assert environment._workdir_from_dockerfile == "/app"


def test_dockerfile_build_instructions_fail_closed(tmp_path):
    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True)
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:22.04\nRUN apt-get update\n", encoding="utf-8")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    with pytest.raises(ValueError, match="cannot build Dockerfiles"):
        OpenSandboxEnvironment(
            environment_dir=environment_dir,
            environment_name="test-task",
            session_id="test-task__trial",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(build_timeout_sec=180),
            logger=logging.getLogger("test.opensandbox"),
        )


def test_start_maps_image_resources_metadata_and_policy(tmp_path):
    environment = _make_environment(
        tmp_path,
        network_policy=NetworkPolicy(
            network_mode=NetworkMode.NO_NETWORK,
        ),
    )
    sandbox = FakeSandbox()

    async def exercise():
        with (
            mock.patch(
                "tasktrace_capture.opensandbox_env.Sandbox.create",
                new=mock.AsyncMock(return_value=sandbox),
            ) as create,
            mock.patch.object(
                environment,
                "_upload_environment_dir_after_start",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(
                environment,
                "_renew_until_stopped",
                new=mock.AsyncMock(),
            ),
        ):
            await environment.start(force_build=False)
            kwargs = create.await_args.kwargs
            assert create.await_args.args == ("ubuntu:22.04",)
            assert kwargs["connection_config"].use_server_proxy is True
            assert kwargs["resource"] == {"cpu": "4", "memory": "4Gi"}
            assert kwargs["resource_requests"] == {
                "cpu": "0.25",
                "memory": "512Mi",
            }
            assert kwargs["metadata"]["user"] == "test-user"
            assert kwargs["metadata"]["task"] == "test-task"
            assert kwargs["metadata"]["network_policy_requested"] == "no-network"
            assert kwargs["metadata"]["network_policy_effective"] == "no-network"
            assert kwargs["metadata"]["network_policy_enforced"] == "true"
            assert kwargs["metadata"]["network_policy_allowed_target_count"] == "0"
            assert len(kwargs["metadata"]["network_policy_allowed_targets_hash"]) == 32
            assert kwargs["network_policy"].default_action == "deny"
            await environment.stop(delete=True)

    asyncio.run(exercise())
    assert sandbox.destroyed is True


def test_start_applies_task_resources_and_numeric_override(tmp_path):
    environment = _make_environment(
        tmp_path,
        cpus=2,
        memory_mb=8192,
        override_cpus=6,
    )
    sandbox = FakeSandbox()

    async def exercise():
        with (
            mock.patch(
                "tasktrace_capture.opensandbox_env.Sandbox.create",
                new=mock.AsyncMock(return_value=sandbox),
            ) as create,
            mock.patch.object(
                environment,
                "_upload_environment_dir_after_start",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(
                environment,
                "_renew_until_stopped",
                new=mock.AsyncMock(),
            ),
        ):
            await environment.start(force_build=False)
            kwargs = create.await_args.kwargs
            assert kwargs["resource"] == {"cpu": "6", "memory": "8192Mi"}
            assert kwargs["resource_requests"] == {
                "cpu": "6",
                "memory": "8192Mi",
            }
            await environment.stop(delete=True)

    asyncio.run(exercise())


def test_resource_configuration_preserves_explicit_enforcement_modes(tmp_path):
    environment = _make_environment(
        tmp_path,
        cpus=3,
        memory_mb=4096,
        cpu_enforcement_policy=ResourceMode.REQUEST,
        memory_enforcement_policy=ResourceMode.LIMIT,
    )

    limits, requests = environment._resource_configuration()

    assert limits == {"memory": "4096Mi"}
    assert requests == {"cpu": "3"}


def test_supported_agent_setup_and_inference_hosts_share_restricted_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("_TASKTRACE_CAPTURE_AGENT", "mini-swe-agent")
    monkeypatch.setenv("INFERENCE_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TASKTRACE_CAPTURE_OPENSANDBOX_SETUP_ALLOWED_HOSTS", "astral.sh,pypi.org")
    environment = _make_environment(
        tmp_path,
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
    )
    sandbox = FakeSandbox()

    async def exercise():
        with (
            mock.patch(
                "tasktrace_capture.opensandbox_env.Sandbox.create",
                new=mock.AsyncMock(return_value=sandbox),
            ) as create,
            mock.patch.object(
                environment,
                "_upload_environment_dir_after_start",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(
                environment,
                "_renew_until_stopped",
                new=mock.AsyncMock(),
            ),
        ):
            await environment.start(force_build=False)
            policy = create.await_args.kwargs["network_policy"]
            assert policy.default_action == "deny"
            assert [rule.target for rule in policy.egress] == [
                "api.example.com",
                "astral.sh",
                "pypi.org",
            ]
            metadata = create.await_args.kwargs["metadata"]
            assert metadata["network_policy_effective"] == "allowlist"
            assert metadata["network_policy_allowed_target_count"] == "3"
            assert metadata["network_policy_agent_setup_target_count"] == "2"
            assert metadata["network_policy_allowed_targets_hash"] == (
                _allowed_targets_digest(["api.example.com", "astral.sh", "pypi.org"])
            )
            await environment.stop(delete=True)

    asyncio.run(exercise())


def test_exec_maps_options_logs_and_return_code(tmp_path, monkeypatch):
    environment = _make_environment(tmp_path)
    sandbox = FakeSandbox()
    sandbox.commands.result = Execution(
        exit_code=7,
        logs=ExecutionLogs(
            stdout=[OutputMessage(text="stdout", timestamp=1)],
            stderr=[OutputMessage(text="stderr", timestamp=2, is_error=True)],
        ),
    )
    environment._sandbox = sandbox
    monkeypatch.setenv("_TASKTRACE_CAPTURE_FORWARD_ENVS", "FORWARDED")
    monkeypatch.setenv("FORWARDED", "yes")

    result = asyncio.run(
        environment.exec(
            "false",
            cwd="/app",
            env={"LOCAL": "value"},
            timeout_sec=19,
            user="root",
        )
    )

    assert result.return_code == 7
    assert result.stdout == "stdout"
    assert result.stderr == "stderr"
    _, opts, _ = sandbox.commands.calls[-1]
    assert opts.working_directory == "/app"
    assert opts.timeout.total_seconds() == 19
    assert opts.uid == 0
    assert opts.envs["LOCAL"] == "value"
    assert opts.envs["FORWARDED"] == "yes"


def test_exec_uses_task_workdir_before_dockerfile_workdir(tmp_path):
    environment = _make_environment(tmp_path, workdir="/task-workdir")
    environment._workdir_from_dockerfile = "/image-workdir"
    sandbox = FakeSandbox()
    environment._sandbox = sandbox

    asyncio.run(environment.exec("pwd", user="root"))

    _, opts, _ = sandbox.commands.calls[-1]
    assert opts.working_directory == "/task-workdir"


def test_file_round_trip_uses_binary_api(tmp_path):
    environment = _make_environment(tmp_path)
    sandbox = FakeSandbox()
    environment._sandbox = sandbox
    source = tmp_path / "source.bin"
    source.write_bytes(b"binary\\x00content")
    destination = tmp_path / "nested" / "destination.bin"

    async def exercise():
        await environment.upload_file(source, "/tmp/value.bin")
        await environment.download_file("/tmp/value.bin", destination)

    asyncio.run(exercise())
    assert destination.read_bytes() == b"binary\\x00content"
    assert sandbox.files.write_calls[-1][2]["mode"] == "644"


def test_directory_upload_uses_octal_archive_mode(tmp_path):
    environment = _make_environment(tmp_path)
    sandbox = FakeSandbox()
    environment._sandbox = sandbox
    source = tmp_path / "source-dir"
    source.mkdir()
    (source / "value.txt").write_text("value", encoding="utf-8")

    asyncio.run(environment.upload_dir(source, "/opt/input"))

    assert sandbox.files.write_calls[-1][2]["mode"] == "600"
    assert sandbox.commands.calls[-1][0].startswith("mkdir -p /opt/input")


def test_force_build_is_rejected(tmp_path):
    environment = _make_environment(tmp_path)
    with pytest.raises(ValueError, match="cannot force-build"):
        asyncio.run(environment.start(force_build=True))

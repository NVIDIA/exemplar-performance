# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secret values must reach kubectl over stdin, never in the process arguments."""

import base64
import json
import subprocess
from unittest.mock import patch

from llmb_install.cluster.kubernetes import create_hf_token_secret, create_image_pull_secrets


def _completed(returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr="boom")


def test_hf_token_secret_is_applied_from_stdin():
    with patch("llmb_install.cluster.kubernetes.subprocess.run", return_value=_completed()) as run:
        ok = create_hf_token_secret("hf-secret", "hf_SECRET", "ns", kubeconfig="/etc/kube/config")

    assert ok
    (argv,), kwargs = run.call_args
    assert argv == ["kubectl", "apply", "-f", "-", "--kubeconfig", "/etc/kube/config"]
    assert "hf_SECRET" not in " ".join(argv)
    manifest = json.loads(kwargs["input"])
    assert manifest["metadata"] == {"name": "hf-secret", "namespace": "ns"}
    assert base64.b64decode(manifest["data"]["hf-token"]).decode() == "hf_SECRET"  # the key the charts read


def test_hf_token_secret_uses_the_requested_key():
    with patch("llmb_install.cluster.kubernetes.subprocess.run", return_value=_completed()) as run:
        create_hf_token_secret("s", "tok", "ns", secret_key="custom")

    assert list(json.loads(run.call_args.kwargs["input"])["data"]) == ["custom"]


def test_failed_apply_is_reported_not_raised(capsys):
    with patch("llmb_install.cluster.kubernetes.subprocess.run", return_value=_completed(returncode=1)):
        assert create_hf_token_secret("s", "tok", "ns") is False

    assert "boom" in capsys.readouterr().out


def test_image_pull_secret_is_a_docker_config_applied_from_stdin():
    with patch("llmb_install.cluster.kubernetes.subprocess.run", return_value=_completed()) as run:
        create_image_pull_secrets(["ngc-a", "ngc-b"], "nvapi-SECRET", "ns")

    assert run.call_count == 2
    for call in run.call_args_list:
        assert "nvapi-SECRET" not in " ".join(call.args[0])
    manifest = json.loads(run.call_args_list[0].kwargs["input"])
    assert manifest["type"] == "kubernetes.io/dockerconfigjson"
    config = json.loads(base64.b64decode(manifest["data"][".dockerconfigjson"]))
    entry = config["auths"]["nvcr.io"]
    assert (entry["username"], entry["password"]) == ("$oauthtoken", "nvapi-SECRET")
    assert base64.b64decode(entry["auth"]).decode() == "$oauthtoken:nvapi-SECRET"


def test_no_ngc_key_means_no_kubectl_call():
    with patch("llmb_install.cluster.kubernetes.subprocess.run") as run:
        create_image_pull_secrets(["ngc"], "", "ns")

    run.assert_not_called()


def test_failed_image_pull_secret_fails_the_install_after_trying_every_secret():
    import pytest

    results = iter([_completed(returncode=1), _completed()])
    with patch("llmb_install.cluster.kubernetes.subprocess.run", side_effect=lambda *a, **k: next(results)) as run:
        with pytest.raises(RuntimeError, match="ngc-a") as exc_info:
            create_image_pull_secrets(["ngc-a", "ngc-b"], "nvapi-SECRET", "ns")

    assert run.call_count == 2  # the second secret was still attempted
    assert "ngc-b" not in str(exc_info.value)


def test_kubectl_errors_creating_an_image_pull_secret_also_fail_the_install():
    import pytest

    with patch("llmb_install.cluster.kubernetes.subprocess.run", side_effect=OSError("no kubectl")):
        with pytest.raises(RuntimeError, match="ngc-a"):
            create_image_pull_secrets(["ngc-a"], "nvapi-SECRET", "ns")

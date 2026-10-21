# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""prompt_kubernetes_info collects the settings written to cluster_config.yaml's kubernetes block."""

import pytest

from llmb_install.ui.prompts.kubernetes import _validate_optional_pvc_name, prompt_kubernetes_info


class ScriptedUI:
    """Answers each prompt whose message contains a key; anything else gets its default."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.prompts = []

    def _answer(self, message, default):
        self.prompts.append({"message": message, "default": default})
        for key, value in self.answers.items():
            if key in message:
                return value
        return default

    def print_section(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def prompt_text(self, message, default="", validate=None, password=False):
        return self._answer(message, default)

    def prompt_path(self, message, default="", validate=None):
        return self._answer(message, default)

    def prompt_confirm(self, message, default=True):
        return self._answer(message, default)


def test_pvc_name_is_collected_into_the_kubernetes_block():
    ui = ScriptedUI({"PVC name": "  my-pvc  "})

    kubernetes = prompt_kubernetes_info(ui)["kubernetes"]

    assert kubernetes["workdir_pvc"] == "my-pvc"


def test_blank_pvc_name_is_left_out():
    kubernetes = prompt_kubernetes_info(ScriptedUI({"PVC name": ""}))["kubernetes"]

    assert "workdir_pvc" not in kubernetes


def test_prior_pvc_name_is_offered_as_the_default():
    ui = ScriptedUI()

    kubernetes = prompt_kubernetes_info(ui, {"workdir_pvc": "old-pvc"})["kubernetes"]

    pvc_prompt = next(p for p in ui.prompts if "PVC name" in p["message"])
    assert pvc_prompt["default"] == "old-pvc"
    assert kubernetes["workdir_pvc"] == "old-pvc"  # accepting the default keeps it


def test_cancelling_the_pvc_prompt_cancels_the_install():
    assert prompt_kubernetes_info(ScriptedUI({"PVC name": None})) is None


def test_node_selector_is_not_prompted_for():
    ui = ScriptedUI()

    kubernetes = prompt_kubernetes_info(ui)["kubernetes"]

    assert not any("selector" in p["message"].lower() for p in ui.prompts)
    assert "node_selector_json" not in kubernetes


@pytest.mark.parametrize("name", ["", "   ", "my-pvc", "a", "team.shared-pvc", "pvc-0"])
def test_valid_pvc_names_pass_validation(name):
    assert _validate_optional_pvc_name(name) is True


@pytest.mark.parametrize("name", ["My-PVC", "-pvc", "pvc-", "pvc_1", "has space", ".pvc", "a" * 254])
def test_invalid_pvc_names_are_rejected(name):
    assert _validate_optional_pvc_name(name) is not True

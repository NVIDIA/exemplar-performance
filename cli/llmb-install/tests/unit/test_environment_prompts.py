# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for recipe environment prompts (environment_prompts)."""

import os

from llmb_install.config.models import EnvironmentPrompt
from llmb_install.core.workload import (
    HF_TOKEN_PROMPT,
    collect_environment_prompts,
    get_environment_prompts,
    missing_environment_prompts,
)
from llmb_install.ui.prompts.environment import prompt_environment_variables


class ScriptedUI:
    """Answers prompts from a script; re-asks while the validator rejects, like the real prompt."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.logs = []
        self.prompts = []

    def log(self, message="", level=None):
        self.logs.append(str(message))

    def prompt_text(self, message, default="", validate=None):
        self.prompts.append(message)
        while True:
            assert self.answers, f"no scripted answer left for prompt: {message}"
            answer = self.answers.pop(0)
            if answer is None:
                return None
            value = default if answer == "" else answer
            if validate is None or validate(value) is True:
                return value


def _workload(key, prompts, path="/repo/recipe"):
    return {"general": {"workload": key}, "path": path, "environment_prompts": prompts}


def test_get_environment_prompts_normalizes_and_skips_malformed(caplog):
    data = _workload(
        "w",
        [
            {"name": "PUBLIC_NVIDIA_API_KEY", "description": "key"},
            "SOME_TOKEN",
            {"name": "not a valid name"},
            {"description": "no name"},
            {"name": "λ_TOKEN"},
            "",
        ],
    )
    prompts = get_environment_prompts(data)
    assert [p.name for p in prompts] == ["PUBLIC_NVIDIA_API_KEY", "SOME_TOKEN"]
    assert prompts[0].name == "PUBLIC_NVIDIA_API_KEY"
    assert prompts[0].description == "key"
    assert prompts[1].name == "SOME_TOKEN"
    assert prompts[1].description == ""
    assert "Ignoring malformed environment_prompts entry" in caplog.text


def test_get_environment_prompts_absent_sections():
    assert get_environment_prompts({"path": "/x"}) == []
    assert get_environment_prompts({"path": "/x", "environment_prompts": None}) == []
    assert get_environment_prompts({"path": "/x", "environment_prompts": []}) == []


def test_collect_merges_by_name_in_selection_order():
    workloads = {
        "a": _workload("a", [{"name": "KEY"}, {"name": "ONLY_A", "description": "a"}]),
        "b": _workload("b", [{"name": "KEY", "description": "from b"}]),
        "c": _workload("c", []),
    }
    merged = collect_environment_prompts(workloads, ["b", "a", "missing", "c"])
    assert [m.name for m in merged] == ["HF_TOKEN", "KEY", "ONLY_A"]
    key = merged[1]
    assert key.description == "from b"
    assert key.workloads == ["b", "a"]
    assert merged[2].workloads == ["a"]


def test_missing_environment_prompts():
    prompts = [
        EnvironmentPrompt(name="REQ", required=True, workloads=["w"]),
        EnvironmentPrompt(name="OPT", required=False, workloads=["w"]),
        EnvironmentPrompt(name="SET", required=True, workloads=["w"]),
    ]
    assert [m.name for m in missing_environment_prompts(prompts, {"SET": "x"})] == ["REQ"]
    assert [m.name for m in missing_environment_prompts(prompts, None)] == ["REQ", "SET"]
    assert missing_environment_prompts(prompts, {"SET": "x", "REQ": ""})[0].name == "REQ"
    assert missing_environment_prompts(prompts, {"SET": "x", "REQ": "   "})[0].name == "REQ"


def test_collect_merges_required_flag():
    workloads = {
        "a": _workload("a", [{"name": "KEY", "required": False}]),
        "b": _workload("b", [{"name": "KEY", "required": True}]),
    }
    merged = collect_environment_prompts(workloads, ["a", "b"])
    assert merged[1].required is True

    # Order inverted: still required
    merged2 = collect_environment_prompts(workloads, ["b", "a"])
    assert merged2[1].required is True


def test_prompt_environment_variables_interactive_includes_prompts(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("PUBLIC_NVIDIA_API_KEY", raising=False)
    ui = ScriptedUI(["hf_abc", "nvapi-xyz"])
    out = prompt_environment_variables(
        ui,
        None,
        express_mode=False,
        prompts=[HF_TOKEN_PROMPT, EnvironmentPrompt(name="PUBLIC_NVIDIA_API_KEY", workloads=["w"])],
    )
    assert out == {"HF_TOKEN": "hf_abc", "PUBLIC_NVIDIA_API_KEY": "nvapi-xyz"}
    assert ui.prompts == ["Enter HF_TOKEN:", "Enter PUBLIC_NVIDIA_API_KEY:"]


def test_prompt_environment_variables_with_builtin_prompt(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ui = ScriptedUI(["hf_abc"])
    assert prompt_environment_variables(ui, None, express_mode=False, prompts=[HF_TOKEN_PROMPT]) == {
        "HF_TOKEN": "hf_abc"
    }
    assert ui.prompts == ["Enter HF_TOKEN:"]


def test_prompt_environment_variables_express_prompts_only_missing(monkeypatch):
    monkeypatch.delenv("NEW_KEY", raising=False)
    ui = ScriptedUI(["fresh-value"])
    prompts = [
        EnvironmentPrompt(name="SAVED_KEY", workloads=["w"]),
        EnvironmentPrompt(name="NEW_KEY", workloads=["w"]),
    ]
    out = prompt_environment_variables(
        ui, {"HF_TOKEN": "hf_saved", "SAVED_KEY": "kept"}, express_mode=True, prompts=prompts
    )
    assert out == {"HF_TOKEN": "hf_saved", "SAVED_KEY": "kept", "NEW_KEY": "fresh-value"}
    assert ui.prompts == ["Enter NEW_KEY:"]


def test_prompt_environment_variables_express_uses_process_environment(monkeypatch):
    monkeypatch.setenv("NEW_KEY", "  from-environment  ")
    ui = ScriptedUI([])

    out = prompt_environment_variables(
        ui,
        {"NEW_KEY": "   "},
        express_mode=True,
        prompts=[EnvironmentPrompt(name="NEW_KEY", workloads=["w"])],
    )

    assert out == {"NEW_KEY": "from-environment"}
    assert ui.prompts == []


def test_prompt_environment_variables_express_picks_up_optional_from_environment(monkeypatch):
    monkeypatch.setenv("OPT_KEY", "opt-value")
    ui = ScriptedUI([])

    out = prompt_environment_variables(
        ui,
        {},
        express_mode=True,
        prompts=[EnvironmentPrompt(name="OPT_KEY", required=False, workloads=["w"])],
    )

    assert out == {"OPT_KEY": "opt-value"}
    assert ui.prompts == []


def test_prompt_environment_variables_express_without_prompts_carries_saved_forward():
    ui = ScriptedUI([])
    assert prompt_environment_variables(ui, {"HF_TOKEN": "hf_saved", "OTHER": "x"}, express_mode=True) == {
        "HF_TOKEN": "hf_saved",
        "OTHER": "x",
    }
    assert ui.prompts == []


def test_prompt_environment_variables_cancel_returns_none():
    ui = ScriptedUI([None])
    assert (
        prompt_environment_variables(
            ui, None, express_mode=False, prompts=[EnvironmentPrompt(name="KEY", workloads=["w"])]
        )
        is None
    )


def test_prompt_environment_variables_defaults_from_saved_then_environment(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("FROM_ENV", "env-value-123")
    ui = ScriptedUI(["hf_saved", "", ""])  # keep HF_TOKEN, accept both defaults
    out = prompt_environment_variables(
        ui,
        {"SAVED": "saved-value-456"},
        express_mode=False,
        prompts=[
            HF_TOKEN_PROMPT,
            EnvironmentPrompt(name="SAVED", workloads=["w"]),
            EnvironmentPrompt(name="FROM_ENV", workloads=["w"]),
        ],
    )
    assert out == {"HF_TOKEN": "hf_saved", "SAVED": "saved-value-456", "FROM_ENV": "env-value-123"}
    joined = "\n".join(ui.logs)
    assert "Found existing SAVED: saved-valu..." in joined
    assert "Found existing FROM_ENV: env-value-..." in joined


def test_prompt_optional_input_may_be_skipped(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPT_KEY", raising=False)
    ui = ScriptedUI(["", ""])  # skip HF_TOKEN, skip OPT_KEY
    prompts = [HF_TOKEN_PROMPT, EnvironmentPrompt(name="OPT_KEY", required=False, workloads=["w"])]
    out = prompt_environment_variables(ui, None, express_mode=False, prompts=prompts)
    assert out == {}
    assert any("No OPT_KEY provided" in log for log in ui.logs)


def test_prompt_optional_can_be_cleared(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OPT_KEY", raising=False)
    ui = ScriptedUI(["hf_saved", "clear"])
    prompts = [HF_TOKEN_PROMPT, EnvironmentPrompt(name="OPT_KEY", required=False, workloads=["w"])]
    out = prompt_environment_variables(ui, {"OPT_KEY": "old_val"}, express_mode=False, prompts=prompts)
    assert out == {"HF_TOKEN": "hf_saved"}


def test_prompt_optional_from_environment_can_be_cleared(monkeypatch):
    monkeypatch.setenv("OPT_KEY", "old_val")
    ui = ScriptedUI(["clear"])

    out = prompt_environment_variables(
        ui,
        None,
        express_mode=False,
        prompts=[EnvironmentPrompt(name="OPT_KEY", required=False, workloads=["w"])],
    )

    assert out == {}
    assert "OPT_KEY" not in os.environ
    assert ui.prompts == ["Enter OPT_KEY (press Enter to use existing, or type 'clear' to remove):"]


def test_required_prompt_rejects_whitespace_saved_default(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("REQUIRED_KEY", raising=False)
    ui = ScriptedUI(["", "", "configured"])  # skip HF_TOKEN; reject blank required value, then replace it

    out = prompt_environment_variables(
        ui,
        {"REQUIRED_KEY": "   "},
        express_mode=False,
        prompts=[HF_TOKEN_PROMPT, EnvironmentPrompt(name="REQUIRED_KEY", workloads=["w"])],
    )

    assert out == {"REQUIRED_KEY": "configured"}


def test_recipe_hf_token_prompt_merges_with_builtin(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ui = ScriptedUI(["hf_value"])
    prompts = collect_environment_prompts(
        {"w": _workload("w", [EnvironmentPrompt(name="HF_TOKEN", required=True)])},
        ["w"],
    )

    out = prompt_environment_variables(
        ui,
        None,
        express_mode=False,
        prompts=prompts,
    )

    assert out == {"HF_TOKEN": "hf_value"}
    assert ui.prompts == ["Enter HF_TOKEN:"]


def test_environment_prompt_values_persist_like_hf_token(tmp_path, monkeypatch):
    from llmb_install.config.models import InstallConfig
    from llmb_install.config.system import SystemConfigManager

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("PUBLIC_NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = InstallConfig(
        install_path=str(tmp_path / "llmb"),
        gpu_type="gb300",
        node_architecture="aarch64",
        venv_type="uv",
        environment_vars={"HF_TOKEN": "hf_saved", "PUBLIC_NVIDIA_API_KEY": "nvapi-saved"},
        selected_workloads=["w"],
    )
    manager = SystemConfigManager()
    manager.save(config)
    saved = manager.load().environment_vars
    assert saved == {"HF_TOKEN": "hf_saved", "PUBLIC_NVIDIA_API_KEY": "nvapi-saved"}

    prompts = [HF_TOKEN_PROMPT, EnvironmentPrompt(name="PUBLIC_NVIDIA_API_KEY", workloads=["w"])]
    ui = ScriptedUI(["", ""])  # Enter accepts saved HF_TOKEN and saved PUBLIC_NVIDIA_API_KEY
    assert prompt_environment_variables(ui, saved, express_mode=False, prompts=prompts) == saved
    assert any("Found existing PUBLIC_NVIDIA_API_KEY" in line for line in ui.logs)


def test_saved_variables_not_prompted_this_run_are_carried_forward(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    saved = {"HF_TOKEN": "hf_saved", "PUBLIC_NVIDIA_API_KEY": "nvapi-saved"}
    ui = ScriptedUI([""])  # Enter keeps HF_TOKEN; nothing declared this run
    assert prompt_environment_variables(ui, saved, express_mode=False, prompts=[]) == saved
    assert prompt_environment_variables(ScriptedUI([]), saved, express_mode=True, prompts=[]) == saved

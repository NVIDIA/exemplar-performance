# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Secret prompts must still run the caller's validator."""

from unittest.mock import MagicMock, patch

import pytest

from llmb_install.ui.rich_ui import RichUI
from llmb_install.ui.simple import SimpleUI


@pytest.mark.parametrize("ui_class", [RichUI, SimpleUI])
def test_password_prompt_applies_the_validator(ui_class):
    def validate(value):
        return True if value.startswith("hf_") else "must start with hf_"

    with patch("questionary.password") as password:
        password.return_value = MagicMock(ask=MagicMock(return_value="hf_abc"))
        result = ui_class().prompt_text("Enter HF_TOKEN:", validate=validate, password=True)

    assert result == "hf_abc"
    assert password.call_args.kwargs["validate"] is validate


def test_rich_password_prompt_shows_the_plain_message():
    with patch("questionary.password") as password:
        password.return_value = MagicMock(ask=MagicMock(return_value="x"))
        RichUI().prompt_text("Enter HF_TOKEN:", password=True)

    assert password.call_args.args[0] == "Enter HF_TOKEN:"  # no literal Rich markup

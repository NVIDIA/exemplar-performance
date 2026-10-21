# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from offline_replayer.adapters.atif_replay import ReplayError, _codex_exec_actions


@pytest.mark.parametrize(
    "body",
    [
        "cmd='printf SOURCE > marker'; return tools.exec_command({cmd});",
        "return; tools.exec_command({cmd});",
        "return\n tools.exec_command({cmd});",
        "throw new Error('abort'); tools.exec_command({cmd});",
        "tools.exec_command({cmd}); cmd='changed';",
        "const r=await tools.exec_command({cmd}); return `${process.exit()}`;",
        "const cmd=await tools.exec_command({cmd}); return cmd;",
        "false && tools.exec_command({cmd});",
    ],
)
def test_rejects_effectful_mapped_callback(body: str) -> None:
    with pytest.raises(ReplayError, match="command batch"):
        _codex_exec_actions(
            "const cmds=['printf ORIGINAL > marker']; await Promise.all(cmds.map(async cmd=>{" + body + "}));",
            "call",
        )


@pytest.mark.parametrize(
    "body",
    [
        "return tools.exec_command({cmd});",
        "const r=await tools.exec_command({cmd}); return {output:r.output,label:'literal'};",
        "const r=await tools.exec_command({cmd}); return `${cmd}:${r.output}`;",
    ],
)
def test_preserves_literal_mapped_callback_presentation(body: str) -> None:
    actions = _codex_exec_actions(
        "const cmds=['printf ORIGINAL > marker']; await Promise.all(cmds.map(async cmd=>{" + body + "}));",
        "call",
    )
    assert actions[0][1]["cmd"] == "printf ORIGINAL > marker"

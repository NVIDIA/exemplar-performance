# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "tasktrace_capture" / "apptainer"
ALLOWED_INTERNAL_IMPORTS = {
    "__init__": {"environment"},
    "capabilities": set(),
    "command_policy": {"instance"},
    "config": {"installation"},
    "environment": {
        "command_policy",
        "config",
        "images",
        "instance",
        "mounts",
        "process",
        "transfers",
        "workspace",
    },
    "images": {"capabilities", "config", "process"},
    "instance": {"capabilities", "config", "mounts", "process"},
    "installation": set(),
    "mounts": {"config"},
    "process": set(),
    "transfers": {"config", "instance"},
    "workspace": {"config"},
}


def _internal_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    prefix = "tasktrace_capture.apptainer."
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(prefix):
                imports.add(node.module.removeprefix(prefix).split(".", 1)[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    imports.add(alias.name.removeprefix(prefix).split(".", 1)[0])
    return imports


def test_apptainer_internal_dependency_graph_is_explicit():
    modules = {path.stem for path in PACKAGE.glob("*.py")}
    assert modules == set(ALLOWED_INTERNAL_IMPORTS)

    for module, allowed in ALLOWED_INTERNAL_IMPORTS.items():
        imports = _internal_imports(PACKAGE / f"{module}.py")
        assert imports <= allowed, (
            f"{module}.py adds internal dependencies {sorted(imports - allowed)}; "
            "update the architecture deliberately instead of creating a back-edge"
        )

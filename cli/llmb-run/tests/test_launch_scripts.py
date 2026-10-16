"""Dry-run every Megatron-Bridge launch.sh against the matrix its metadata declares.

Each recipe's launch.sh turns GPU_TYPE / MODEL_SIZE / DTYPE / JOB_TOTAL_GPUS into
Megatron-Bridge launcher arguments and then calls setup_experiment.py. A stub
`python` on PATH records that call instead of submitting anything, so the flag
computation can be checked without a cluster. This catches the class of bug
that otherwise surfaces only as a wrong number after a multi-hour run: a size
the script does not handle, a dtype that maps to the wrong recipe, an image
version that drifted from metadata, an unexpanded variable in the argv.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

from llmb_run.constants import GPU_TYPE_TO_NUM_GPUS
from llmb_run.metadata_utils import normalize_model_dtype_config
from llmb_run.run_config import resolve_container_images

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ACCOUNT = "test-account"
PARTITION = "test-partition"

STUB = """#!/usr/bin/env bash
# Records one invocation: a separator line, then one argv token per line.
{
    echo "==invocation=="
    printf '%s\\n' "$@"
} >> "$LLMB_TEST_ARGV_FILE"
"""


def _find_bash() -> str | None:
    """A bash new enough for the scripts' own version guard (4.2+)."""
    candidates = [shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash"]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        out = subprocess.run(
            [candidate, "-c", 'echo "${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"'], capture_output=True, text=True
        )
        major, minor = (int(x) for x in out.stdout.strip().split("."))
        if (major, minor) >= (4, 2):
            return candidate
    return None


BASH = _find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="launch.sh requires bash >= 4.2")


def _megatron_bridge_recipes() -> list[tuple[pathlib.Path, dict]]:
    recipes = []
    for meta_path in sorted(REPO_ROOT.glob("**/metadata.yaml")):
        if "source_snapshot" in meta_path.parts:
            continue
        metadata = yaml.safe_load(meta_path.read_text())
        if (metadata.get("run") or {}).get("launcher_type") == "megatron_bridge":
            recipes.append((meta_path.parent, metadata))
    return recipes


def _cases() -> list[pytest.ParameterSet]:
    cases = []
    for recipe_dir, metadata in _megatron_bridge_recipes():
        for gpu_type, gpu_cfg in (metadata["run"].get("gpu_configs") or {}).items():
            for model_cfg in gpu_cfg.get("model_configs") or []:
                size = str(model_cfg["model_size"])
                for dtype, spec in normalize_model_dtype_config(model_cfg).items():
                    scales = sorted(set(spec["scales"]) | set(spec.get("proxy_scales") or []))
                    case_id = f"{recipe_dir.relative_to(REPO_ROOT).as_posix()}-{gpu_type}-{size}-{dtype}"
                    cases.append(pytest.param(recipe_dir, metadata, gpu_type, size, dtype, scales, id=case_id))
    return cases


CASES = _cases()


def _expected_image_name(metadata: dict, gpu_type: str) -> str:
    """Enroot squashfs name llmb-install derives from the first metadata image."""
    images = resolve_container_images((metadata.get("container") or {}).get("images"), gpu_type)
    assert images, f"metadata declares no container image for {gpu_type}"
    ref = images[0].split("#", 1)[-1]  # drop the registry
    return ref.replace("/", "+").replace(":", "+") + ".sqsh"


def _parse_argv(argv: list[str]) -> dict[str, list[str]]:
    """`--flag value` and `--flag=value` into flag -> values; bare flags map to ''."""
    flags: dict[str, list[str]] = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("-"):
            if "=" in token and token.startswith("--"):
                flag, value = token.split("=", 1)
                flags.setdefault(flag, []).append(value)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                flags.setdefault(token, []).append(argv[i + 1])
                i += 1
            else:
                flags.setdefault(token, []).append("")
        i += 1
    return flags


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory) -> pathlib.Path:
    bin_dir = tmp_path_factory.mktemp("stub-bin")
    for name in ("python", "python3"):
        stub = bin_dir / name
        stub.write_text(STUB)
        stub.chmod(0o755)
    return bin_dir


def _run_launch(
    recipe_dir: pathlib.Path,
    metadata: dict,
    stub_bin: pathlib.Path,
    tmp_path: pathlib.Path,
    env_overrides: dict[str, str],
    unset: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    general = metadata["general"]
    llmb_install = tmp_path / "install"
    (llmb_install / "workloads" / f"{general['workload_type']}_{general['workload']}" / "Megatron-Bridge").mkdir(
        parents=True
    )
    argv_file = tmp_path / "argv.txt"

    env = {
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "LLMB_INSTALL": str(llmb_install),
        "LLMB_TEST_ARGV_FILE": str(argv_file),
        "SBATCH_ACCOUNT": ACCOUNT,
        "SBATCH_PARTITION": PARTITION,
        **env_overrides,
    }
    for key in unset:
        env.pop(key, None)

    result = subprocess.run(
        [BASH, "./launch.sh"],
        cwd=recipe_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    invocations: list[list[str]] = []
    if argv_file.exists():
        for block in argv_file.read_text().split("==invocation==\n")[1:]:
            invocations.append(block.splitlines())
    return result, invocations


@pytest.mark.parametrize("recipe_dir, metadata, gpu_type, size, dtype, scales", CASES)
def test_launch_script_builds_launcher_args(recipe_dir, metadata, gpu_type, size, dtype, scales, stub_bin, tmp_path):
    for scale in scales:
        result, invocations = _run_launch(
            recipe_dir,
            metadata,
            stub_bin,
            tmp_path / str(scale),
            {"GPU_TYPE": gpu_type, "MODEL_SIZE": size, "DTYPE": dtype, "JOB_TOTAL_GPUS": str(scale)},
        )
        context = f"{recipe_dir.name} gpu={gpu_type} size={size} dtype={dtype} scale={scale}"
        assert result.returncode == 0, f"{context}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert len(invocations) == 1, f"{context}: expected one launcher call, got {len(invocations)}"

        argv = invocations[0]
        assert argv and argv[0].endswith("scripts/performance/setup_experiment.py"), context
        assert all(argv), f"{context}: empty argv token in {argv}"
        leaked = [token for token in argv if re.search(r"\$\{?[A-Za-z_]", token)]
        assert not leaked, f"{context}: unexpanded variable in {leaked}"

        flags = _parse_argv(argv[1:])
        assert flags["--gpu"] == [gpu_type], context
        assert flags["--num_gpus"] == [str(scale)], context
        assert flags["--account"] == [ACCOUNT], context
        assert flags["--partition"] == [PARTITION], context

        gpus_per_node = int(flags["--gpus_per_node"][0])
        assert gpus_per_node == GPU_TYPE_TO_NUM_GPUS[gpu_type], context
        if scale >= gpus_per_node:
            assert scale % gpus_per_node == 0, f"{context}: scale not a whole number of nodes"

        compute_dtype = flags["--compute_dtype"][0]
        assert compute_dtype.startswith(dtype), f"{context}: --compute_dtype {compute_dtype!r} does not match {dtype}"

        image = pathlib.Path(flags["--container_image"][0])
        assert image.name == _expected_image_name(metadata, gpu_type), f"{context}: image {image.name}"
        assert image.is_relative_to(tmp_path), f"{context}: image outside LLMB_INSTALL: {image}"

        assert flags["--model_recipe_name"][0], context
        assert flags["--log_dir"][0].startswith(str(tmp_path)), context
        assert flags["--max_steps"][0].isdigit(), context


RECIPES = _megatron_bridge_recipes()


@pytest.mark.parametrize("recipe_dir, metadata", RECIPES, ids=[d.relative_to(REPO_ROOT).as_posix() for d, _ in RECIPES])
def test_launch_script_requires_gpu_type(recipe_dir, metadata, stub_bin, tmp_path):
    result, invocations = _run_launch(
        recipe_dir,
        metadata,
        stub_bin,
        tmp_path,
        {"MODEL_SIZE": "8b", "DTYPE": "bf16", "JOB_TOTAL_GPUS": "8"},
        unset=("GPU_TYPE",),
    )
    assert result.returncode != 0
    assert invocations == []

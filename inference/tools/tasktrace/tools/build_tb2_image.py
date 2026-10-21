#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Terminal-Bench 2 task image into an enroot .sqsh, no Docker.

Replays a Dockerfile's RUN/COPY/WORKDIR/ENV steps inside a writable enroot
container started from the FROM base, then exports the result with
``enroot export``. This is a generic, task-agnostic tool — it builds any
TB2 task's image from its Dockerfile context directory.

Supported directives: FROM, RUN, COPY, WORKDIR, ENV, ARG. The build inherits the
base image's configured WORKDIR (probed from the created container), so task
Dockerfiles that rely on the base WORKDIR for relative paths (e.g. a relative
``git clone``) build correctly without an explicit WORKDIR line. Multi-stage
Dockerfiles (>1 FROM) are not supported — flatten to a single stage first.

Run on a node with enroot + registry/apt/pip egress::

  python build_tb2_image.py --task-dir <ctx> --output <out>.sqsh
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

BUILD_CONTEXT = "/tb2-build-context"


def _read_dockerfile_instructions(dockerfile: Path) -> list[tuple[str, str]]:
    """Return ordered (directive, argument) pairs, joining line continuations."""
    logical: list[str] = []
    buffer = ""
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        fragment = stripped
        if fragment.endswith("\\"):
            buffer += fragment[:-1].rstrip() + " "
            continue
        buffer += fragment
        logical.append(buffer)
        buffer = ""
    if buffer:
        logical.append(buffer)

    instructions: list[tuple[str, str]] = []
    for entry in logical:
        text = entry.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split(None, 1)
        instructions.append((parts[0].upper(), parts[1] if len(parts) > 1 else ""))
    return instructions


def _from_count(instructions: list[tuple[str, str]]) -> int:
    return sum(1 for d, _ in instructions if d == "FROM")


def _from_base_ref(instructions: list[tuple[str, str]]) -> str:
    """Return the FROM base image reference, stripping any --platform flag."""
    for directive, argument in instructions:
        if directive == "FROM":
            tokens = [t for t in argument.split() if not t.startswith("--platform")]
            if not tokens:
                raise ValueError("FROM has no image reference")
            if len(tokens) >= 3 and tokens[-2].upper() == "AS":
                tokens = tokens[:-2]
            return tokens[0]
    raise ValueError("Dockerfile has no FROM")


def _enroot_uri(image_ref: str) -> str:
    """Convert a docker image ref to enroot's [REGISTRY#]IMAGE[:TAG] URI."""
    head = image_ref.split("/", 1)[0]
    if "/" in image_ref and ("." in head or ":" in head):
        host, rest = image_ref.split("/", 1)
        return f"docker://{host}#{rest}"
    return f"docker://{image_ref}"


def _resolve_base_sqsh(image_ref: str, cache_dir: Path, enroot: str) -> Path:
    """Return a cached .sqsh for the base image, importing it if necessary."""
    safe = image_ref.replace("/", "_").replace(":", "_").replace("#", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    sqsh = cache_dir / f"{safe}.sqsh"
    if not sqsh.exists():
        _run([enroot, "import", "--output", str(sqsh), _enroot_uri(image_ref)])
    return sqsh


def _probe_base_workdir(container: str, enroot: str) -> str:
    """The base image's configured WORKDIR (so relative RUN/COPY inherit it).

    Probes the freshly-created container's default cwd. Falls back to '/' if the
    probe fails or returns a non-absolute path.
    """
    try:
        out = subprocess.run(
            [enroot, "start", container, "/bin/sh", "-c", "pwd"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith("/")]
        if lines:
            return lines[-1]
    except (subprocess.CalledProcessError, OSError):
        pass
    return "/"


def _copy_commands(argument: str, workdir: str) -> list[str]:
    """Translate a COPY into cp commands against the mounted build context."""
    # Drop --chown / --from flags (single-stage contexts don't use --from).
    tokens = [tok for tok in shlex.split(argument) if not tok.startswith("--")]
    if len(tokens) < 2:
        raise ValueError(f"COPY needs at least src and dst: {argument!r}")
    *sources, dest = tokens
    dest_abs = dest if dest.startswith("/") else f"{workdir.rstrip('/')}/{dest}"
    cmds: list[str] = []
    dest_is_dir = dest.endswith("/") or dest in {".", "./"} or len(sources) > 1
    if dest_is_dir:
        cmds.append(f"mkdir -p {shlex.quote(dest_abs)}")
    else:
        cmds.append(f"mkdir -p {shlex.quote(str(Path(dest_abs).parent))}")
    for src in sources:
        ctx_src = f"{BUILD_CONTEXT}/{src}"
        if src.endswith("/"):
            cmds.append(f"cp -a {shlex.quote(ctx_src + '.')} {shlex.quote(dest_abs.rstrip('/') + '/')}")
        else:
            cmds.append(f"cp -a {shlex.quote(ctx_src)} {shlex.quote(dest_abs)}")
    return cmds


def _environment_assignments(argument: str) -> list[str]:
    """Normalize Dockerfile ENV arguments into individual NAME=value items."""
    tokens = shlex.split(argument)
    if not tokens:
        raise ValueError("ENV needs a name and value")
    if "=" not in tokens[0]:
        if len(tokens) < 2:
            raise ValueError(f"ENV needs a value: {argument!r}")
        return [f"{tokens[0]}={' '.join(tokens[1:])}"]
    if any("=" not in token or token.startswith("=") for token in tokens):
        raise ValueError(f"invalid ENV assignment list: {argument!r}")
    return tokens


def build_script(instructions: list[tuple[str, str]], base_workdir: str = "/") -> str:
    """Generate the in-container build shell replaying the Dockerfile steps.

    ``base_workdir`` is the FROM image's configured WORKDIR; RUN/COPY before any
    explicit WORKDIR directive run there (so a relative `git clone` lands where
    the upstream Dockerfile expects), rather than being forced to '/'.
    """
    lines = ["set -eux", "export DEBIAN_FRONTEND=noninteractive"]
    workdir = base_workdir or "/"
    lines.append(f"mkdir -p {shlex.quote(workdir)} && cd {shlex.quote(workdir)}")
    for directive, argument in instructions:
        if directive == "FROM":
            continue
        elif directive == "WORKDIR":
            workdir = argument if argument.startswith("/") else f"{workdir.rstrip('/')}/{argument}"
            lines.append(f"mkdir -p {shlex.quote(workdir)} && cd {shlex.quote(workdir)}")
        elif directive == "RUN":
            lines.append(f"cd {shlex.quote(workdir)}")
            lines.append(argument)
        elif directive == "COPY":
            lines.extend(_copy_commands(argument, workdir))
        elif directive == "ENV":
            assignments = _environment_assignments(argument)
            if "=" in shlex.split(argument)[0]:
                # Keep Docker's modern assignment form so references to
                # existing variables expand in the build shell.
                lines.append(f"export {argument}")
            else:
                lines.append(f"export {shlex.quote(assignments[0])}")
            for assignment in assignments:
                lines.append(f"printf '%s\\n' {shlex.quote(assignment)} >> /etc/environment")
        elif directive == "ARG":
            assignment = argument
            if "=" in assignment:
                lines.append(f"export {assignment}")
                lines.append(f"printf '%s\\n' {shlex.quote(assignment)} >> /etc/environment")
        # ADD/ENTRYPOINT/CMD/USER/EXPOSE/LABEL are absent in TB2 task Dockerfiles.
    return "\n".join(lines) + "\n"


def _run(cmd: list[str]) -> None:
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _default_cache_dir() -> Path:
    """Base-image cache dir, env-overridable (no hardcoded user path)."""
    env = os.environ.get("TASKTRACE_BASE_CACHE_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("TASKTRACE_WORKLOAD_DIR", os.getcwd())) / "tb2-base"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one TB2 task image as an enroot .sqsh.")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--base-sqsh", type=Path, help="Override base .sqsh; default derives from FROM.")
    parser.add_argument(
        "--base-cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="Where to cache imported FROM base images (env: TASKTRACE_BASE_CACHE_DIR).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output task .sqsh path.")
    parser.add_argument("--enroot", default="enroot")
    parser.add_argument("--keep", action="store_true", help="Do not remove the build container.")
    args = parser.parse_args(argv)

    dockerfile = args.task_dir / "Dockerfile"
    if not dockerfile.is_file():
        parser.error(f"no Dockerfile in {args.task_dir}")
    if args.output.exists():
        print(f"output already exists, skipping: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    instructions = _read_dockerfile_instructions(dockerfile)
    if _from_count(instructions) > 1:
        print(
            f"ERROR ({args.task_dir.name}): multi-stage Dockerfile (>1 FROM) not supported; "
            "flatten to a single stage in the task context.",
            file=sys.stderr,
        )
        return 2
    try:
        if args.base_sqsh is not None:
            base_sqsh = args.base_sqsh
        else:
            base_ref = _from_base_ref(instructions)
            print(f"[{args.task_dir.name}] FROM base: {base_ref}")
            base_sqsh = _resolve_base_sqsh(base_ref, args.base_cache_dir, args.enroot)
    except ValueError as exc:
        print(f"ERROR ({args.task_dir.name}): {exc}", file=sys.stderr)
        return 2

    container = f"tb2build-{args.task_dir.name}-{uuid.uuid4().hex[:8]}"
    try:
        _run([args.enroot, "create", "--name", container, str(base_sqsh)])
        base_workdir = _probe_base_workdir(container, args.enroot)
        print(f"[{args.task_dir.name}] base WORKDIR: {base_workdir}")
        script = build_script(instructions, base_workdir=base_workdir)
        _run(
            [
                args.enroot,
                "start",
                "--rw",
                "--root",
                "--mount",
                f"{args.task_dir.resolve()}:{BUILD_CONTEXT}",
                container,
                "/bin/sh",
                "-c",
                script,
            ]
        )
        _run([args.enroot, "export", "--output", str(args.output), container])
    finally:
        if not args.keep:
            subprocess.run([args.enroot, "remove", "-f", container], check=False)
    print(f"BUILT {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

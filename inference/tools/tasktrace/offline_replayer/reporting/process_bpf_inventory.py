# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public report command for one loss-audited external BPF process capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..measurement.process_bpf import (
    EXPECTED_SCHEMA_VERSION,
    ProcessBpfEvidenceError,
    build_process_inventory_report,
    write_process_inventory_report,
)


def _root_prefix(path: Path) -> str:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessBpfEvidenceError(f"unable to read expected manifest: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ProcessBpfEvidenceError("expected manifest schema_version is unsupported")
    prefix = value.get("root_argv_prefix")
    if not isinstance(prefix, str) or not prefix.startswith("/"):
        raise ProcessBpfEvidenceError("expected manifest root_argv_prefix must begin with an absolute path")
    return prefix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed process-tree and binary-inventory proof from one " "loss-audited external BPF capture."
        )
    )
    parser.add_argument("raw_csv", type=Path, help="AgentScry-style raw process CSV.")
    parser.add_argument(
        "tracer_stderr",
        type=Path,
        help="Tracer diagnostics with one terminal AGENTSCRY_LOST_EVENTS marker.",
    )
    parser.add_argument(
        "pipeline_report",
        type=Path,
        help="Pipeline report from the same replay execution.",
    )
    parser.add_argument(
        "expected_manifest",
        type=Path,
        help="Pinned expected process inventory and canonical tree manifest.",
    )
    parser.add_argument(
        "collector",
        type=Path,
        help="Exact reviewed collector source used for this capture.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output proof JSON.")
    args = parser.parse_args(argv)
    try:
        report = build_process_inventory_report(
            args.raw_csv,
            args.tracer_stderr,
            args.pipeline_report,
            root_argv_prefix=_root_prefix(args.expected_manifest),
            expected_manifest_path=args.expected_manifest,
            collector_path=args.collector,
        )
        write_process_inventory_report(report, args.output)
    except ProcessBpfEvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(f"Process inventory valid: {str(report['valid']).lower()}")
    print(f"Captured tasks: {summary['task_count']}")
    print(f"Successful execs: {summary['successful_exec_count']}")
    print(f"Distinct binaries: {summary['distinct_binary_count']}")
    print(f"Report: {args.output}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

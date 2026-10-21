# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public report command for repeated external BPF process proofs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..measurement.process_bpf import (
    ProcessBpfEvidenceError,
    build_process_repeatability_report,
    write_process_inventory_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two or more process-inventory BPF proofs, requiring identical "
            "workload, tree, inventory, collector, and expected-manifest identities."
        )
    )
    parser.add_argument("proof", type=Path, nargs="+", help="Process proof JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output repeatability JSON.")
    args = parser.parse_args(argv)
    try:
        report = build_process_repeatability_report(args.proof)
        write_process_inventory_report(report, args.output)
    except ProcessBpfEvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Process repeatability valid: {str(report['valid']).lower()}")
    print(f"Compared runs: {report['run_count']}")
    print(f"Report: {args.output}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

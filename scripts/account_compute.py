#!/usr/bin/env python3
"""Create manual compute and derived timing/compute records from one benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.compute_accounting import (
    derive_efficiency_report,
    operation_record_from_benchmark,
    write_compute_record,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Versioned benchmark JSON record")
    parser.add_argument("--output", required=True, help="Manual compute JSON output")
    parser.add_argument("--report", help="Optional derived timing/compute JSON output")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    benchmark_path = Path(args.benchmark).expanduser().resolve()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    operation_record = operation_record_from_benchmark(benchmark)
    write_compute_record(args.output, operation_record, replace=args.replace)
    if args.report:
        report = derive_efficiency_report(benchmark, operation_record)
        write_compute_record(args.report, report, replace=args.replace)
    print(
        f"Wrote compute record {operation_record['operation_record_id']} "
        f"to {Path(args.output).expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()

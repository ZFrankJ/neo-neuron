#!/usr/bin/env python3
"""Run one backend-neutral wall-clock and memory benchmark repetition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.efficiency import benchmark_from_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=("torch", "mlx"), required=True)
    parser.add_argument("--device", required=True, help="Explicit cpu, mps, gpu, or cuda device")
    parser.add_argument(
        "--workload",
        choices=("train_step", "sequence_eval", "streaming_decode"),
        required=True,
    )
    parser.add_argument(
        "--timing-scope",
        choices=("model_only", "end_to_end_loop"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--measured-iterations", type=int, default=100)
    parser.add_argument("--repetition-id", required=True)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Allow fewer than 20 warm-ups and 100 measured iterations; "
            "record is explicitly provisional"
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing authoritative JSON record",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    record = benchmark_from_paths(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        backend_name=args.backend,
        device=args.device,
        workload=args.workload,
        timing_scope=args.timing_scope,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
        repetition_id=args.repetition_id,
        profile_label=args.profile_label,
        output_path=args.output,
        seed=args.seed,
        dry_run=args.dry_run,
        replace=args.replace,
    )
    print(f"Wrote benchmark record {record['record_id']} to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

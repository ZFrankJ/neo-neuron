#!/usr/bin/env python3
"""Summarize run metrics into a CSV."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize metrics JSON files")
    parser.add_argument("--root", default="runs", help="Root directory to scan")
    parser.add_argument("--out", default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    root = Path(args.root)
    metrics_files = sorted(root.rglob("metrics.json"))
    legacy_files = sorted(root.glob("metrics_*.json"))
    metrics_files.extend(legacy_files)
    if not metrics_files:
        print(f"No metrics files found under {root}")
        return

    rows = []
    for path in metrics_files:
        payload = path.read_text(encoding="utf-8")
        data = __import__("json").loads(payload)
        if path.name == "metrics.json":
            run_tag = path.parent.name
        else:
            run_tag = path.stem.replace("metrics_", "")
        row = {
            "run_tag": run_tag,
            "val_ppl": data.get("val_ppl"),
            "test_ppl": data.get("test_ppl"),
            "path": str(path),
        }
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_tag", "val_ppl", "test_ppl", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

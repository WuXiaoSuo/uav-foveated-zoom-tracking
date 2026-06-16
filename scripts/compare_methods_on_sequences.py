#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare methods on selected UAV123 sequences.")
    parser.add_argument("--results", required=True, help="Path to outputs/tables/main_results.csv.")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to print, first method is baseline.")
    parser.add_argument("--sequences", nargs="+", required=True, help="Sequence names or comma lists.")
    return parser.parse_args()


def _split(values: list[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _read_results(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(row.get("method", ""), row.get("sequence", "")): row for row in rows}


def _float(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return 0.0
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    args = parse_args()
    results = _read_results(Path(args.results))
    methods = _split(args.methods)
    sequences = _split(args.sequences)
    baseline_method = methods[0]

    fieldnames = [
        "sequence",
        "method",
        "success_auc",
        "precision_20",
        "mean_cle",
        "delta_auc_vs_ufz",
        "delta_dp_vs_ufz",
        "delta_cle_vs_ufz",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for sequence in sequences:
        baseline = results.get((baseline_method, sequence))
        for method in methods:
            row = results.get((method, sequence))
            writer.writerow(
                {
                    "sequence": sequence,
                    "method": method,
                    "success_auc": "" if row is None else row.get("success_auc", ""),
                    "precision_20": "" if row is None else row.get("precision_20", ""),
                    "mean_cle": "" if row is None else row.get("mean_cle", ""),
                    "delta_auc_vs_ufz": f"{_float(row, 'success_auc') - _float(baseline, 'success_auc'):.6f}",
                    "delta_dp_vs_ufz": f"{_float(row, 'precision_20') - _float(baseline, 'precision_20'):.6f}",
                    "delta_cle_vs_ufz": f"{_float(row, 'mean_cle') - _float(baseline, 'mean_cle'):.6f}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

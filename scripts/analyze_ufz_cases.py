#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare UFZ v1 and failure-aware UFZ v2 sequence cases.")
    parser.add_argument("--output-root", default="/root/autodl-tmp/UFZTrack/outputs")
    parser.add_argument("--baseline", default="ufz")
    parser.add_argument("--candidate", default="ufz_v2")
    return parser.parse_args()


def _read_main_results(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"main_results.csv does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(row["method"], row["sequence"]): row for row in rows if row.get("sequence") != "ALL"}


def _read_log_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "total_frames": 0,
            "zoom_8_frames": 0,
            "recovery_frames": 0,
            "caution_frames": 0,
            "predict_only_frames": 0,
            "low_assoc_frames": 0,
            "command_summary": "",
        }
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    command_counter = Counter(row.get("command", "") or "unknown" for row in rows)
    state_counter = Counter(row.get("state", "") for row in rows)
    assoc_counter = Counter(row.get("assoc_stage", "") for row in rows)
    return {
        "total_frames": len(rows),
        "zoom_8_frames": sum(row.get("zoom_level", "") == "8" for row in rows),
        "recovery_frames": state_counter["RECOVERY"],
        "caution_frames": state_counter["CAUTION"],
        "predict_only_frames": assoc_counter["predict_only"],
        "low_assoc_frames": assoc_counter["low"],
        "command_summary": ";".join(f"{key}:{command_counter[key]}" for key in sorted(command_counter)),
    }


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    table_path = output_root / "tables" / "main_results.csv"
    rows_by_key = _read_main_results(table_path)
    sequences = sorted(
        {
            sequence
            for method, sequence in rows_by_key
            if method in {args.baseline, args.candidate}
        }
    )
    output_rows: list[dict[str, Any]] = []
    for sequence in sequences:
        baseline = rows_by_key.get((args.baseline, sequence))
        candidate = rows_by_key.get((args.candidate, sequence))
        if baseline is None or candidate is None:
            warnings.warn(f"Missing pair for sequence {sequence}; skipping.", RuntimeWarning, stacklevel=2)
            continue
        log_summary = _read_log_summary(output_root / "logs" / args.candidate / f"{sequence}.csv")
        output_rows.append(
            {
                "sequence": sequence,
                "baseline": args.baseline,
                "candidate": args.candidate,
                "baseline_success_auc": baseline.get("success_auc", ""),
                "candidate_success_auc": candidate.get("success_auc", ""),
                "delta_success_auc": f"{_float(candidate, 'success_auc') - _float(baseline, 'success_auc'):.6f}",
                "baseline_precision_20": baseline.get("precision_20", ""),
                "candidate_precision_20": candidate.get("precision_20", ""),
                "delta_precision_20": f"{_float(candidate, 'precision_20') - _float(baseline, 'precision_20'):.6f}",
                "baseline_mean_iou": baseline.get("mean_iou", ""),
                "candidate_mean_iou": candidate.get("mean_iou", ""),
                "delta_mean_iou": f"{_float(candidate, 'mean_iou') - _float(baseline, 'mean_iou'):.6f}",
                **log_summary,
            }
        )

    output_path = output_root / "tables" / "ufz_case_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "baseline",
        "candidate",
        "baseline_success_auc",
        "candidate_success_auc",
        "delta_success_auc",
        "baseline_precision_20",
        "candidate_precision_20",
        "delta_precision_20",
        "baseline_mean_iou",
        "candidate_mean_iou",
        "delta_mean_iou",
        "total_frames",
        "zoom_8_frames",
        "recovery_frames",
        "caution_frames",
        "predict_only_frames",
        "low_assoc_frames",
        "command_summary",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

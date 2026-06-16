#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


RECOMMENDED_PAPER_CASES = [
    "bike1",
    "boat9",
    "truck1",
    "building4",
    "person13",
    "car17",
    "person1",
    "person20",
    "car5",
    "car10",
    "car9",
    "boat6",
    "boat7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare UFZ policies and prepare paper-case tables.")
    parser.add_argument("--output-root", default="/root/autodl-tmp/UFZTrack/outputs")
    parser.add_argument("--baseline", default="ufz")
    parser.add_argument("--candidate", default="ufz_v2_2")
    parser.add_argument("--selected-cases", nargs="+", default=None, help="Paper case names or comma lists.")
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
            "zoom_1": 0,
            "zoom_2": 0,
            "zoom_4": 0,
            "zoom_8": 0,
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
    zoom_counter = Counter(_parse_zoom_level(row.get("zoom_level", "")) for row in rows)
    return {
        "total_frames": len(rows),
        "zoom_1": zoom_counter[1],
        "zoom_2": zoom_counter[2],
        "zoom_4": zoom_counter[4],
        "zoom_8": zoom_counter[8],
        "zoom_8_frames": zoom_counter[8],
        "recovery_frames": state_counter["RECOVERY"],
        "caution_frames": state_counter["CAUTION"],
        "predict_only_frames": assoc_counter["predict_only"],
        "low_assoc_frames": assoc_counter["low"],
        "command_summary": ";".join(f"{key}:{command_counter[key]}" for key in sorted(command_counter)),
    }


def _parse_zoom_level(value: str) -> int:
    try:
        zoom = int(round(float(str(value).strip())))
    except ValueError:
        return 0
    return zoom if zoom in {1, 2, 4, 8} else 0


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0.0)
    except ValueError:
        return 0.0


def _split_names(values: list[str] | None) -> list[str]:
    if not values:
        return list(RECOMMENDED_PAPER_CASES)
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _case_type(delta_auc: float, delta_dp20: float, delta_mean_cle: float, summary: dict[str, Any]) -> str:
    used_tele = int(summary.get("zoom_4", 0)) + int(summary.get("zoom_8", 0)) > 0
    if delta_auc >= 0.05 or delta_dp20 >= 0.05:
        return "improvement_case"
    if summary.get("recovery_frames", 0) or summary.get("predict_only_frames", 0):
        return "failure_recovery_case"
    if delta_auc <= -0.05 or delta_dp20 <= -0.05:
        return "regression_case"
    if used_tele:
        return "active_zoom_stable_case"
    if delta_mean_cle > 100.0:
        return "unrecoverable_failure_case"
    return "neutral_case"


def _case_reason(case_type: str, delta_auc: float, delta_dp20: float, delta_mean_cle: float) -> str:
    if case_type == "improvement_case":
        return f"candidate improves AUC by {delta_auc:.4f} and DP@20 by {delta_dp20:.4f}"
    if case_type == "failure_recovery_case":
        return "log contains recovery or predict-only frames; useful for diagnosing hard-risk behavior"
    if case_type == "regression_case":
        return f"candidate drops AUC by {-delta_auc:.4f}; include as limitation case"
    if case_type == "active_zoom_stable_case":
        return "candidate keeps active telephoto zoom without large aggregate regression"
    if case_type == "unrecoverable_failure_case":
        return f"large CLE degradation of {delta_mean_cle:.2f}px; include as failure boundary"
    return "representative neutral comparison case"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
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
        "zoom_1",
        "zoom_2",
        "zoom_4",
        "zoom_8",
        "zoom_8_frames",
        "recovery_frames",
        "caution_frames",
        "predict_only_frames",
        "low_assoc_frames",
        "command_summary",
    ]
    generic_path = tables_dir / "ufz_case_analysis.csv"
    _write_csv(generic_path, fieldnames, output_rows)
    print(f"Wrote {generic_path}")

    candidate_path = tables_dir / f"{args.candidate}_case_analysis.csv"
    if candidate_path != generic_path:
        _write_csv(candidate_path, fieldnames, output_rows)
        print(f"Wrote {candidate_path}")

    selected_rows: list[dict[str, Any]] = []
    selected_sequences = _split_names(args.selected_cases)
    for sequence in selected_sequences:
        baseline = rows_by_key.get((args.baseline, sequence))
        candidate = rows_by_key.get((args.candidate, sequence))
        if baseline is None or candidate is None:
            warnings.warn(f"Missing selected paper case pair for {sequence}; skipping.", RuntimeWarning, stacklevel=2)
            continue
        summary = _read_log_summary(output_root / "logs" / args.candidate / f"{sequence}.csv")
        delta_auc = _float(candidate, "success_auc") - _float(baseline, "success_auc")
        delta_dp20 = _float(candidate, "precision_20") - _float(baseline, "precision_20")
        delta_mean_cle = _float(candidate, "mean_cle") - _float(baseline, "mean_cle")
        case_type = _case_type(delta_auc, delta_dp20, delta_mean_cle, summary)
        selected_rows.append(
            {
                "case_type": case_type,
                "sequence": sequence,
                "reason": _case_reason(case_type, delta_auc, delta_dp20, delta_mean_cle),
                "ufz_auc": baseline.get("success_auc", ""),
                "ufz_v2_2_auc": candidate.get("success_auc", ""),
                "delta_auc_vs_ufz": f"{delta_auc:.6f}",
                "ufz_dp20": baseline.get("precision_20", ""),
                "ufz_v2_2_dp20": candidate.get("precision_20", ""),
                "delta_dp20_vs_ufz": f"{delta_dp20:.6f}",
                "ufz_mean_cle": baseline.get("mean_cle", ""),
                "ufz_v2_2_mean_cle": candidate.get("mean_cle", ""),
                "delta_mean_cle_vs_ufz": f"{delta_mean_cle:.6f}",
                "zoom_1": summary.get("zoom_1", 0),
                "zoom_2": summary.get("zoom_2", 0),
                "zoom_4": summary.get("zoom_4", 0),
                "zoom_8": summary.get("zoom_8", 0),
                "command_summary": summary.get("command_summary", ""),
            }
        )

    selected_path = tables_dir / "selected_paper_cases.csv"
    selected_fieldnames = [
        "case_type",
        "sequence",
        "reason",
        "ufz_auc",
        "ufz_v2_2_auc",
        "delta_auc_vs_ufz",
        "ufz_dp20",
        "ufz_v2_2_dp20",
        "delta_dp20_vs_ufz",
        "ufz_mean_cle",
        "ufz_v2_2_mean_cle",
        "delta_mean_cle_vs_ufz",
        "zoom_1",
        "zoom_2",
        "zoom_4",
        "zoom_8",
        "command_summary",
    ]
    _write_csv(selected_path, selected_fieldnames, selected_rows)
    print(f"Wrote {selected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

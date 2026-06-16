#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UFZ-Track UAV123@10fps result txt files.")
    parser.add_argument("--config", default="configs/uav123_10fps.yaml")
    parser.add_argument("--methods", nargs="+", default=None, help="Optional method names or comma lists.")
    return parser.parse_args()


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _split(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _read_result_boxes(path: Path):
    from ufztrack.bbox import BBox

    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.replace(",", " ").split()
        if len(parts) < 4:
            warnings.warn(f"Invalid result box at {path}:{line_no}: {line!r}", RuntimeWarning, stacklevel=2)
            boxes.append(BBox(float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        try:
            boxes.append(BBox.from_xywh(float(v) for v in parts[:4]))
        except ValueError:
            warnings.warn(f"Non-numeric result box at {path}:{line_no}: {line!r}", RuntimeWarning, stacklevel=2)
            boxes.append(BBox(float("nan"), float("nan"), float("nan"), float("nan")))
    return boxes


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in {float("inf"), float("-inf")}:
        return 0.0
    return number


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_macro_rows(rows: list[dict[str, Any]], robust_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    main_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    robust_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("sequence") != "ALL":
            main_by_method[str(row["method"])].append(row)
    for row in robust_rows:
        if row.get("sequence") != "ALL":
            robust_by_method[str(row["method"])].append(row)

    output_rows: list[dict[str, Any]] = []
    for method in sorted(set(main_by_method) | set(robust_by_method)):
        main_values = main_by_method.get(method, [])
        robust_values = robust_by_method.get(method, [])
        output_rows.append(
            {
                "method": method,
                "sequence_count": len(main_values),
                "macro_mean_iou": _mean([_safe_float(row.get("mean_iou")) for row in main_values]),
                "macro_success_auc": _mean([_safe_float(row.get("success_auc")) for row in main_values]),
                "macro_precision_20": _mean([_safe_float(row.get("precision_20")) for row in main_values]),
                "macro_median_cle": _mean([_safe_float(row.get("median_cle")) for row in robust_values]),
                "macro_failure_rate_50": _mean([_safe_float(row.get("failure_rate_50")) for row in robust_values]),
                "macro_failure_rate_100": _mean([_safe_float(row.get("failure_rate_100")) for row in robust_values]),
            }
        )
    return output_rows


def _build_win_count_rows(rows: list[dict[str, Any]], robust_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    robust_by_key = {
        (str(row["method"]), str(row["sequence"])): row
        for row in robust_rows
        if row.get("sequence") != "ALL"
    }
    methods: set[str] = set()
    for row in rows:
        if row.get("sequence") == "ALL":
            continue
        by_sequence[str(row["sequence"])].append(row)
        methods.add(str(row["method"]))

    counts = {
        method: {
            "method": method,
            "auc_best_count": 0,
            "dp20_best_count": 0,
            "mean_cle_best_count": 0,
            "median_cle_best_count": 0,
            "failure_rate_50_best_count": 0,
            "total_sequences": 0,
        }
        for method in methods
    }
    eps = 1e-12
    for sequence, sequence_rows in by_sequence.items():
        for row in sequence_rows:
            counts[str(row["method"])]["total_sequences"] += 1

        best_auc = max(_safe_float(row.get("success_auc")) for row in sequence_rows)
        best_dp20 = max(_safe_float(row.get("precision_20")) for row in sequence_rows)
        best_mean_cle = min(_safe_float(row.get("mean_cle")) for row in sequence_rows)
        median_candidates = [
            _safe_float(robust_by_key[(str(row["method"]), sequence)].get("median_cle"))
            for row in sequence_rows
            if (str(row["method"]), sequence) in robust_by_key
        ]
        fail50_candidates = [
            _safe_float(robust_by_key[(str(row["method"]), sequence)].get("failure_rate_50"))
            for row in sequence_rows
            if (str(row["method"]), sequence) in robust_by_key
        ]
        best_median_cle = min(median_candidates) if median_candidates else 0.0
        best_fail50 = min(fail50_candidates) if fail50_candidates else 0.0

        for row in sequence_rows:
            method = str(row["method"])
            if abs(_safe_float(row.get("success_auc")) - best_auc) <= eps:
                counts[method]["auc_best_count"] += 1
            if abs(_safe_float(row.get("precision_20")) - best_dp20) <= eps:
                counts[method]["dp20_best_count"] += 1
            if abs(_safe_float(row.get("mean_cle")) - best_mean_cle) <= eps:
                counts[method]["mean_cle_best_count"] += 1
            robust = robust_by_key.get((method, sequence))
            if robust is None:
                continue
            if abs(_safe_float(robust.get("median_cle")) - best_median_cle) <= eps:
                counts[method]["median_cle_best_count"] += 1
            if abs(_safe_float(robust.get("failure_rate_50")) - best_fail50) <= eps:
                counts[method]["failure_rate_50_best_count"] += 1

    return [counts[method] for method in sorted(counts)]


def main() -> int:
    args = parse_args()
    _add_src_to_path()
    from ufztrack.dataset_uav123 import load_uav123_sequence
    from ufztrack.metrics import evaluate_frames, evaluate_sequence, evaluate_sequence_robust
    from ufztrack.visualization import plot_precision_curves, plot_success_curves

    cfg = _load_config(args.config)
    output_root = Path(cfg["outputs"]["root"])
    results_root = output_root / "results"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    requested_methods = _split(args.methods)
    method_dirs = []
    if requested_methods:
        method_dirs = [results_root / method for method in requested_methods]
    else:
        method_dirs = sorted([path for path in results_root.iterdir() if path.is_dir()], key=lambda p: p.name)

    rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []
    all_predictions = defaultdict(list)
    all_gt = defaultdict(list)
    errors_by_method = defaultdict(list)
    ious_by_method = defaultdict(list)

    for method_dir in method_dirs:
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for result_path in sorted(method_dir.glob("*.txt"), key=lambda p: p.name):
            sequence_name = result_path.stem
            preds = _read_result_boxes(result_path)
            try:
                sequence = load_uav123_sequence(
                    sequence_name,
                    image_root=cfg["dataset"]["image_root"],
                    bbox_anno_root=cfg["dataset"]["bbox_anno_root"],
                    frame_glob=cfg["dataset"].get("frame_glob", "*.jpg"),
                    frame_stride=cfg["dataset"].get("frame_stride", 1),
                    max_frames=cfg["dataset"].get("max_frames"),
                )
            except (FileNotFoundError, ValueError) as exc:
                warnings.warn(f"Skipping evaluation for {method}/{sequence_name}: {exc}", RuntimeWarning, stacklevel=2)
                continue
            n = min(len(preds), len(sequence.gt_boxes))
            preds = preds[:n]
            gt_boxes = sequence.gt_boxes[:n]
            metrics = evaluate_sequence(preds, gt_boxes)
            robust_metrics = evaluate_sequence_robust(preds, gt_boxes)
            rows.append({"method": method, "sequence": sequence_name, **asdict(metrics)})
            robust_rows.append({"method": method, "sequence": sequence_name, **asdict(robust_metrics)})
            all_predictions[method].extend(preds)
            all_gt[method].extend(gt_boxes)
            frame_metrics = evaluate_frames(preds, gt_boxes)
            errors_by_method[method].extend(frame.cle for frame in frame_metrics)
            ious_by_method[method].extend(frame.iou for frame in frame_metrics)

    for method in sorted(all_predictions):
        aggregate = evaluate_sequence(all_predictions[method], all_gt[method])
        robust_aggregate = evaluate_sequence_robust(all_predictions[method], all_gt[method])
        rows.append({"method": method, "sequence": "ALL", **asdict(aggregate)})
        robust_rows.append({"method": method, "sequence": "ALL", **asdict(robust_aggregate)})

    if not rows:
        raise RuntimeError(f"No result txt files found under {results_root}")

    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "main_results.csv"
    fieldnames = [
        "method",
        "sequence",
        "frames",
        "mean_iou",
        "success_auc",
        "precision_20",
        "mean_cle",
        "invalid_pred_count",
        "invalid_gt_count",
        "valid_eval_frames",
    ]
    _write_csv(table_path, fieldnames, rows)

    robust_path = tables_dir / "robust_results.csv"
    robust_fieldnames = [
        "method",
        "sequence",
        "frames",
        "median_cle",
        "cle_95",
        "failure_rate_50",
        "failure_rate_100",
        "valid_cle_frames",
    ]
    _write_csv(robust_path, robust_fieldnames, robust_rows)

    macro_path = tables_dir / "macro_results.csv"
    macro_fieldnames = [
        "method",
        "sequence_count",
        "macro_mean_iou",
        "macro_success_auc",
        "macro_precision_20",
        "macro_median_cle",
        "macro_failure_rate_50",
        "macro_failure_rate_100",
    ]
    _write_csv(macro_path, macro_fieldnames, _build_macro_rows(rows, robust_rows))

    win_path = tables_dir / "win_counts.csv"
    win_fieldnames = [
        "method",
        "auc_best_count",
        "dp20_best_count",
        "mean_cle_best_count",
        "median_cle_best_count",
        "failure_rate_50_best_count",
        "total_sequences",
    ]
    _write_csv(win_path, win_fieldnames, _build_win_count_rows(rows, robust_rows))

    plot_precision_curves(errors_by_method, figures_dir / "precision_plot.pdf")
    plot_success_curves(ious_by_method, figures_dir / "success_plot.pdf")
    print(f"Wrote {table_path}")
    print(f"Wrote {robust_path}")
    print(f"Wrote {macro_path}")
    print(f"Wrote {win_path}")
    print(f"Wrote {figures_dir / 'precision_plot.pdf'}")
    print(f"Wrote {figures_dir / 'success_plot.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

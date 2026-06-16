#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path
from typing import Any


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-ready UFZ case curves from existing outputs.")
    parser.add_argument("--output-root", default="/root/autodl-tmp/UFZTrack/outputs")
    parser.add_argument("--config", default="configs/uav123_10fps.yaml")
    parser.add_argument("--cases", nargs="+", required=True, help="Sequence names or comma lists.")
    parser.add_argument("--methods", nargs="+", default=["ufz", "ufz_v2_2"], help="Method names or comma lists.")
    return parser.parse_args()


def _split(values: list[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_result_boxes(path: Path):
    from ufztrack.bbox import BBox

    boxes = []
    if not path.is_file():
        warnings.warn(f"Missing result file: {path}", RuntimeWarning, stacklevel=2)
        return boxes
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.replace(",", " ").split()
        if len(parts) < 4:
            warnings.warn(f"Invalid result box at {path}:{line_no}: {line!r}", RuntimeWarning, stacklevel=2)
            continue
        try:
            boxes.append(BBox.from_xywh(float(value) for value in parts[:4]))
        except ValueError:
            warnings.warn(f"Non-numeric result box at {path}:{line_no}: {line!r}", RuntimeWarning, stacklevel=2)
    return boxes


def _read_log_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        warnings.warn(f"Missing log file: {path}", RuntimeWarning, stacklevel=2)
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value: str | None, default: float = math.nan) -> float:
    try:
        number = float(value or "")
    except ValueError:
        return default
    return number if math.isfinite(number) else default


def _series_from_log(rows: list[dict[str, str]], key: str) -> list[float]:
    return [_safe_float(row.get(key)) for row in rows]


def _cle_curve(predictions, gt_boxes) -> list[float]:
    from ufztrack.metrics import evaluate_frames

    n = min(len(predictions), len(gt_boxes))
    frame_metrics = evaluate_frames(predictions[:n], gt_boxes[:n])
    values = []
    for frame in frame_metrics:
        cle = frame.cle if math.isfinite(frame.cle) else math.nan
        values.append(min(cle, 500.0) if math.isfinite(cle) else math.nan)
    return values


def main() -> int:
    args = parse_args()
    _add_src_to_path()
    from ufztrack.dataset_uav123 import load_uav123_sequence

    cfg = _load_config(args.config)
    output_root = Path(args.output_root)
    sequences = _split(args.cases)
    methods = _split(args.methods)
    figures_dir = output_root / "figures" / "cases"
    figures_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sequence_name in sequences:
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
            warnings.warn(f"Skipping {sequence_name}: {exc}", RuntimeWarning, stacklevel=2)
            continue

        curves: dict[str, list[float]] = {}
        log_series: dict[str, dict[str, list[float]]] = {}
        for method in methods:
            predictions = _read_result_boxes(output_root / "results" / method / f"{sequence_name}.txt")
            if predictions:
                curves[method] = _cle_curve(predictions, sequence.gt_boxes)
            rows = _read_log_rows(output_root / "logs" / method / f"{sequence_name}.csv")
            if rows:
                log_series[method] = {
                    "zoom_level": _series_from_log(rows, "zoom_level"),
                    "area": _series_from_log(rows, "area"),
                    "uncertainty": _series_from_log(rows, "uncertainty"),
                }

        if not curves and not log_series:
            warnings.warn(f"No curves or logs available for {sequence_name}; skipping.", RuntimeWarning, stacklevel=2)
            continue

        fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
        fig.suptitle(f"{sequence_name}: UFZ case behavior", fontsize=13)

        for method, cle_values in curves.items():
            axes[0].plot(range(1, len(cle_values) + 1), cle_values, label=method, linewidth=1.2)
        axes[0].set_ylabel("CLE (px, capped 500)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="upper right")

        for method, series in log_series.items():
            zoom_values = series.get("zoom_level", [])
            if zoom_values:
                axes[1].step(range(1, len(zoom_values) + 1), zoom_values, where="post", label=method, linewidth=1.2)
        axes[1].set_ylabel("Zoom level")
        axes[1].set_yticks([1, 2, 4, 8])
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper right")

        plotted_area = False
        for method, series in log_series.items():
            area_values = series.get("area", [])
            if area_values:
                axes[2].plot(range(1, len(area_values) + 1), area_values, label=f"{method} area", linewidth=1.2)
                plotted_area = True
        if not plotted_area:
            for method, cle_values in curves.items():
                axes[2].plot(range(1, len(cle_values) + 1), cle_values, label=f"{method} CLE proxy", linewidth=1.0)
        axes[2].set_ylabel("BBox area ratio")
        axes[2].set_xlabel("Frame")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend(loc="upper right")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        filename = f"{sequence_name}_{'_vs_'.join(methods)}.png"
        output_path = figures_dir / filename
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

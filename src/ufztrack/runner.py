from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import yaml
from tqdm import tqdm

from .bbox import BBox
from .dataset import load_sequence
from .detector import YOLOv8Detector, select_measurement
from .eval import evaluate
from .kalman import BoxKalmanFilter, KalmanConfig
from .zoom import bbox_from_zoom_to_original, choose_zoom_level, make_zoom_crop


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_tracks(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_idx",
        "image_path",
        "zoom_level",
        "pred_x",
        "pred_y",
        "pred_w",
        "pred_h",
        "measurement_x",
        "measurement_y",
        "measurement_w",
        "measurement_h",
        "measurement_conf",
        "measurement_class",
        "used_measurement",
        "position_trace",
        "missed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    dataset_cfg = cfg["dataset"]
    detector_cfg = cfg["detector"]
    tracker_cfg = cfg["tracker"]
    zoom_cfg = cfg["zoom"]
    selection_cfg = cfg["measurement_selection"]

    sequence = load_sequence(
        dataset_root=dataset_cfg["root"],
        sequence=dataset_cfg.get("sequence"),
        image_globs=dataset_cfg["image_globs"],
        gt_files=dataset_cfg["gt_files"],
        frame_stride=dataset_cfg.get("frame_stride", 1),
        max_frames=dataset_cfg.get("max_frames"),
    )

    output_dir = Path(cfg["experiment"]["output_dir"]) / sequence.name
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOv8Detector(
        weights=detector_cfg["weights"],
        conf=float(detector_cfg["conf"]),
        iou=float(detector_cfg["iou"]),
        imgsz=int(detector_cfg["imgsz"]),
        classes=detector_cfg.get("classes"),
        device=detector_cfg.get("device"),
    )

    # UAV123 SOT rule: only the first-frame GT initializes the tracker.
    kf = BoxKalmanFilter(sequence.gt_boxes[0], KalmanConfig(**tracker_cfg))
    predictions: list[BBox] = []
    rows: list[dict[str, Any]] = []

    for frame_idx, image_path in enumerate(tqdm(sequence.image_paths, desc=sequence.name)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图像：{image_path}")
        height, width = image.shape[:2]

        if frame_idx == 0:
            predicted = kf.current_box().clip(width, height, tracker_cfg["min_box_size"])
            predictions.append(predicted)
            rows.append(_make_row(frame_idx, image_path, zoom_cfg["initial_level"], predicted, None, kf))
            continue

        prior = kf.predict().clip(width, height, tracker_cfg["min_box_size"])
        zoom_level = choose_zoom_level(
            kf.position_trace(),
            zoom_cfg["uncertainty_thresholds"],
            fallback=zoom_cfg["initial_level"],
        )
        zoomed, crop_info = make_zoom_crop(
            image,
            center_x=prior.cx,
            center_y=prior.cy,
            zoom_level=zoom_level,
            padding=float(zoom_cfg.get("crop_padding", 1.0)),
        )
        zoom_detections = detector.detect(zoomed)
        original_detections = [
            type(det)(bbox_from_zoom_to_original(det.bbox, crop_info).clip(width, height), det.confidence, det.class_id)
            for det in zoom_detections
        ]
        measurement = select_measurement(
            original_detections,
            prior,
            image_width=width,
            image_height=height,
            max_center_distance_ratio=float(selection_cfg["max_center_distance_ratio"]),
            score_weight=float(selection_cfg["score_weight"]),
        )

        if measurement is not None:
            predicted = kf.update(measurement.bbox).clip(width, height, tracker_cfg["min_box_size"])
        else:
            predicted = kf.current_box().clip(width, height, tracker_cfg["min_box_size"])
        predictions.append(predicted)
        rows.append(_make_row(frame_idx, image_path, zoom_level, predicted, measurement, kf))

    summary = evaluate(predictions, sequence.gt_boxes)
    _write_tracks(output_dir / "tracks.csv", rows)
    summary_payload = {
        "experiment": cfg["experiment"]["name"],
        "sequence": sequence.name,
        "config_path": str(config_path),
        "summary": asdict(summary),
        "notes": [
            "第一帧 GT 仅用于初始化 Kalman tracker。",
            "后续 GT 仅用于 evaluation，未参与 crop center、measurement selection、tracker update 或 zoom decision。",
            "tracks.csv 中所有 bbox 坐标均为 original image coordinates。",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_payload


def _make_row(frame_idx, image_path, zoom_level, predicted, measurement, kf):
    if measurement is None:
        measurement_values = {
            "measurement_x": "",
            "measurement_y": "",
            "measurement_w": "",
            "measurement_h": "",
            "measurement_conf": "",
            "measurement_class": "",
            "used_measurement": 0,
        }
    else:
        measurement_values = {
            "measurement_x": measurement.bbox.x,
            "measurement_y": measurement.bbox.y,
            "measurement_w": measurement.bbox.w,
            "measurement_h": measurement.bbox.h,
            "measurement_conf": measurement.confidence,
            "measurement_class": measurement.class_id,
            "used_measurement": 1,
        }
    return {
        "frame_idx": frame_idx,
        "image_path": str(image_path),
        "zoom_level": zoom_level,
        "pred_x": predicted.x,
        "pred_y": predicted.y,
        "pred_w": predicted.w,
        "pred_h": predicted.h,
        **measurement_values,
        "position_trace": kf.position_trace(),
        "missed": kf.missed,
    }

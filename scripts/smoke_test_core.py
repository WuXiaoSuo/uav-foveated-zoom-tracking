#!/usr/bin/env python3
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _assert_close(a: float, b: float, eps: float = 1e-6) -> None:
    if abs(a - b) > eps:
        raise AssertionError(f"{a} != {b}")


def main() -> int:
    _add_src_to_path()

    from ufztrack.bbox import BBox
    from ufztrack.detector_yolo import Detection
    from ufztrack.kalman_tracker import KalmanBoxTracker
    from ufztrack.measurement_selector import ByteTrackLikeSingleTargetSelector, MeasurementSelector
    from ufztrack.metrics import (
        bbox_iou,
        center_location_error,
        distance_precision,
        evaluate_sequence,
        evaluate_sequence_robust,
    )
    from ufztrack.uncertainty import estimate_blur_risk
    from ufztrack.zoom_policy import ZoomPolicy
    from ufztrack.zoom_simulator import ZoomSimulator, original_to_zoomed_bbox, zoomed_to_original_bbox

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    box = BBox(50.0, 40.0, 20.0, 16.0)
    simulator = ZoomSimulator(levels=[1, 2, 4, 8], crop_mode="predicted_center")
    _, crop = simulator.simulate(
        image,
        zoom_level=4,
        predicted_bbox=box,
        uncertainty=0.8,
        context_margin_factor=0.2,
        uncertainty_margin_gain=0.3,
        min_context_pixels=8,
    )
    zoomed_box = original_to_zoomed_bbox(box, crop)
    restored_box = zoomed_to_original_bbox(zoomed_box, crop)
    for value, expected in zip(restored_box.to_xywh(), box.to_xywh()):
        _assert_close(value, expected)

    tracker = KalmanBoxTracker(box)
    predicted = tracker.predict()
    updated = tracker.update(BBox(52.0, 41.0, 20.0, 16.0))
    if predicted.w <= 0 or updated.w <= 0 or tracker.lost_count != 0:
        raise AssertionError("Kalman tracker predict/update failed")

    selector = MeasurementSelector()
    detections = [
        Detection(BBox(5.0, 5.0, 10.0, 10.0), 0.95, None),
        Detection(BBox(52.0, 41.0, 20.0, 16.0), 0.80, None),
    ]
    selected = selector.select(updated, detections)
    if selected is None or selected.detection.bbox.x < 40.0:
        raise AssertionError("MeasurementSelector did not pick the reasonable detection")
    byetrack_selector = ByteTrackLikeSingleTargetSelector()
    low_selected = byetrack_selector.select(
        updated,
        [Detection(BBox(52.0, 41.0, 20.0, 16.0), 0.12, None)],
        current_zoom_level=4,
        lost_count=1,
    )
    if low_selected is None or low_selected.assoc_stage != "low":
        raise AssertionError("ByteTrack-like low-confidence association failed")

    iou_value = bbox_iou(BBox(0.0, 0.0, 10.0, 10.0), BBox(5.0, 0.0, 10.0, 10.0))
    _assert_close(iou_value, 1.0 / 3.0)
    cle = center_location_error(BBox(0.0, 0.0, 10.0, 10.0), BBox(3.0, 4.0, 10.0, 10.0))
    _assert_close(cle, 5.0)
    dp = distance_precision([3.0, 21.0, 20.0], threshold=20.0)
    _assert_close(dp, 2.0 / 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        invalid_metrics = evaluate_sequence(
            [BBox(float("nan"), 0.0, 10.0, 10.0), BBox(0.0, 0.0, -1.0, 10.0)],
            [BBox(0.0, 0.0, 10.0, 10.0), BBox(0.0, 0.0, 10.0, 10.0)],
        )
    if invalid_metrics.mean_iou != 0.0 or invalid_metrics.invalid_pred_count != 2:
        raise AssertionError("Invalid bbox metrics are not stable")
    robust_metrics = evaluate_sequence_robust(
        [BBox(0.0, 0.0, 10.0, 10.0), BBox(float("nan"), 0.0, 10.0, 10.0)],
        [BBox(3.0, 4.0, 10.0, 10.0), BBox(0.0, 0.0, 10.0, 10.0)],
    )
    if robust_metrics.valid_cle_frames != 1 or robust_metrics.failure_rate_50 != 0.0:
        raise AssertionError("Robust invalid-frame metrics are not stable")

    policy = ZoomPolicy("scale_only")
    decision = policy.decide(
        current_level=1,
        area_ratio=0.0001,
        confidence=1.0,
        uncertainty=0.0,
        edge_risk=0.0,
        lost_count=0,
        frame_idx=10,
    )
    if decision.level != 2 or decision.command != "zoom_in_to_2":
        raise AssertionError("ZoomPolicy did not enforce bounded-step zoom")
    v2_policy = ZoomPolicy("ufz_v2")
    v2_recovery = v2_policy.decide(
        current_level=8,
        area_ratio=0.02,
        confidence=0.1,
        uncertainty=0.9,
        edge_risk=0.2,
        lost_count=2,
        frame_idx=20,
        blur_risk=0.1,
        association_risk=1.0,
        assoc_stage="predict_only",
        kalman_innovation=1.0,
    )
    if v2_recovery.level != 4 or v2_recovery.state != "RECOVERY":
        raise AssertionError("UFZ v2 recovery did not zoom out one bounded step")
    v21_policy = ZoomPolicy("ufz_v2_1")
    v21_first = v21_policy.decide(
        current_level=1,
        area_ratio=0.001,
        confidence=0.8,
        uncertainty=0.2,
        edge_risk=0.1,
        lost_count=0,
        frame_idx=1,
        blur_risk=0.1,
        association_risk=0.1,
        assoc_stage="high",
        kalman_innovation=0.1,
        association_ambiguity=0.1,
    )
    v21_second = v21_policy.decide(
        current_level=1,
        area_ratio=0.001,
        confidence=0.8,
        uncertainty=0.2,
        edge_risk=0.1,
        lost_count=0,
        frame_idx=2,
        blur_risk=0.1,
        association_risk=0.1,
        assoc_stage="high",
        kalman_innovation=0.1,
        association_ambiguity=0.1,
    )
    if v21_first.command != "keep" or v21_second.command != "zoom_in_to_2":
        raise AssertionError("UFZ v2.1 stable small-target zoom-in failed")
    v21_lost = v21_policy.decide(
        current_level=8,
        area_ratio=0.02,
        confidence=0.1,
        uncertainty=0.9,
        edge_risk=0.1,
        lost_count=2,
        frame_idx=20,
        blur_risk=0.1,
        association_risk=1.0,
        assoc_stage="predict_only",
        kalman_innovation=1.0,
        association_ambiguity=1.0,
    )
    if v21_lost.level != 4 or v21_lost.state != "RECOVERY":
        raise AssertionError("UFZ v2.1 recovery did not zoom out one bounded step")
    v22_policy = ZoomPolicy("ufz_v2_2")
    v22_to_2 = v22_policy.decide(
        current_level=1,
        area_ratio=0.001,
        confidence=0.8,
        uncertainty=0.2,
        edge_risk=0.1,
        lost_count=0,
        frame_idx=1,
        blur_risk=0.1,
        association_risk=0.1,
        assoc_stage="high",
    )
    v22_to_4 = v22_policy.decide(
        current_level=2,
        area_ratio=0.001,
        confidence=0.8,
        uncertainty=0.2,
        edge_risk=0.1,
        lost_count=0,
        frame_idx=10,
        blur_risk=0.1,
        association_risk=0.1,
        assoc_stage="high",
    )
    v22_to_8 = v22_policy.decide(
        current_level=4,
        area_ratio=0.001,
        confidence=0.8,
        uncertainty=0.2,
        edge_risk=0.1,
        lost_count=0,
        frame_idx=20,
        blur_risk=0.1,
        association_risk=0.1,
        assoc_stage="high",
    )
    if [v22_to_2.command, v22_to_4.command, v22_to_8.command] != [
        "zoom_in_to_2",
        "zoom_in_to_4",
        "zoom_in_to_8",
    ]:
        raise AssertionError("UFZ v2.2 did not preserve v1-style progressive zoom-in")
    v22_veto = ZoomPolicy("ufz_v2_2").decide(
        current_level=1,
        area_ratio=0.001,
        confidence=0.8,
        uncertainty=0.2,
        edge_risk=0.1,
        lost_count=0,
        frame_idx=1,
        blur_risk=0.1,
        association_risk=0.95,
        assoc_stage="high",
    )
    if v22_veto.command != "keep" or not v22_veto.veto_applied:
        raise AssertionError("UFZ v2.2 hard association veto failed")
    v22_lost = ZoomPolicy("ufz_v2_2").decide(
        current_level=8,
        area_ratio=0.02,
        confidence=0.1,
        uncertainty=0.9,
        edge_risk=0.1,
        lost_count=2,
        frame_idx=1,
        blur_risk=0.1,
        association_risk=0.1,
        assoc_stage="predict_only",
    )
    if v22_lost.level != 4 or v22_lost.state != "RECOVERY":
        raise AssertionError("UFZ v2.2 lost recovery did not zoom out one bounded step")
    if not (0.0 <= estimate_blur_risk(image) <= 1.0):
        raise AssertionError("Blur risk is outside [0, 1]")

    print("Smoke test passed: zoom mapping, Kalman, selector, and metrics are OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

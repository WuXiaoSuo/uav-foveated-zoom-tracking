from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence
import warnings

from .bbox import BBox, bbox_is_valid, center_distance, iou


DEFAULT_INVALID_CLE_PENALTY = 1_000_000.0


@dataclass(frozen=True)
class SequenceMetrics:
    frames: int
    mean_iou: float
    success_auc: float
    precision_20: float
    mean_cle: float
    invalid_pred_count: int = 0
    invalid_gt_count: int = 0
    valid_eval_frames: int = 0


@dataclass(frozen=True)
class RobustSequenceMetrics:
    frames: int
    median_cle: float
    cle_95: float
    failure_rate_50: float
    failure_rate_100: float
    valid_cle_frames: int


@dataclass(frozen=True)
class FrameEval:
    iou: float
    cle: float
    valid_pred: bool
    valid_gt: bool


def bbox_iou(a: BBox, b: BBox) -> float:
    return _clip01(iou(a, b))


def center_location_error(a: BBox, b: BBox) -> float:
    return center_distance(a, b)


def distance_precision(errors: Iterable[float], threshold: float = 20.0) -> float:
    values = list(errors)
    if not values:
        return 0.0
    return sum(math.isfinite(value) and value <= threshold for value in values) / len(values)


def success_rate(ious: Iterable[float], threshold: float) -> float:
    values = list(ious)
    if not values:
        return 0.0
    return sum(math.isfinite(value) and _clip01(value) >= threshold for value in values) / len(values)


def evaluate_frame(prediction: BBox, gt_box: BBox, invalid_cle_penalty: float = DEFAULT_INVALID_CLE_PENALTY) -> FrameEval:
    pred_valid = bbox_is_valid(prediction)
    gt_valid = bbox_is_valid(gt_box)
    if not pred_valid or not gt_valid:
        return FrameEval(iou=0.0, cle=float(invalid_cle_penalty), valid_pred=pred_valid, valid_gt=gt_valid)
    overlap = bbox_iou(prediction, gt_box)
    error = center_location_error(prediction, gt_box)
    if not math.isfinite(error):
        error = float(invalid_cle_penalty)
    return FrameEval(
        iou=_clip01(overlap),
        cle=min(max(float(error), 0.0), float(invalid_cle_penalty)),
        valid_pred=True,
        valid_gt=True,
    )


def evaluate_frames(
    predictions: Sequence[BBox],
    gt_boxes: Sequence[BBox],
    invalid_cle_penalty: float = DEFAULT_INVALID_CLE_PENALTY,
) -> list[FrameEval]:
    n = min(len(predictions), len(gt_boxes))
    return [evaluate_frame(predictions[i], gt_boxes[i], invalid_cle_penalty) for i in range(n)]


def evaluate_sequence(predictions: Sequence[BBox], gt_boxes: Sequence[BBox]) -> SequenceMetrics:
    frames = evaluate_frames(predictions, gt_boxes)
    n = len(frames)
    if n == 0:
        return SequenceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    invalid_pred_count = sum(not frame.valid_pred for frame in frames)
    invalid_gt_count = sum(not frame.valid_gt for frame in frames)
    valid_eval_frames = sum(frame.valid_pred and frame.valid_gt for frame in frames)
    if invalid_pred_count or invalid_gt_count:
        warnings.warn(
            "Invalid bbox values found during evaluation: "
            f"invalid_pred_count={invalid_pred_count}, invalid_gt_count={invalid_gt_count}",
            RuntimeWarning,
            stacklevel=2,
        )

    ious = [_clip01(frame.iou) for frame in frames]
    errors = [frame.cle if math.isfinite(frame.cle) else DEFAULT_INVALID_CLE_PENALTY for frame in frames]
    success_thresholds = [i / 100.0 for i in range(101)]
    auc = _clip01(sum(success_rate(ious, threshold) for threshold in success_thresholds) / len(success_thresholds))
    return SequenceMetrics(
        frames=n,
        mean_iou=_clip01(sum(ious) / n),
        success_auc=auc,
        precision_20=distance_precision(errors, threshold=20.0),
        mean_cle=_stable_mean(errors),
        invalid_pred_count=invalid_pred_count,
        invalid_gt_count=invalid_gt_count,
        valid_eval_frames=valid_eval_frames,
    )


def evaluate_sequence_robust(predictions: Sequence[BBox], gt_boxes: Sequence[BBox]) -> RobustSequenceMetrics:
    frames = evaluate_frames(predictions, gt_boxes)
    valid_errors = [
        float(frame.cle)
        for frame in frames
        if frame.valid_pred and frame.valid_gt and math.isfinite(frame.cle)
    ]
    valid_cle_frames = len(valid_errors)
    return RobustSequenceMetrics(
        frames=len(frames),
        median_cle=_percentile(valid_errors, 50.0),
        cle_95=_percentile(valid_errors, 95.0),
        failure_rate_50=_failure_rate(valid_errors, 50.0),
        failure_rate_100=_failure_rate(valid_errors, 100.0),
        valid_cle_frames=valid_cle_frames,
    )


def precision_curve(errors: Sequence[float], thresholds: Sequence[float]) -> list[float]:
    return [distance_precision(errors, threshold) for threshold in thresholds]


def success_curve(ious: Sequence[float], thresholds: Sequence[float]) -> list[float]:
    return [success_rate(ious, threshold) for threshold in thresholds]


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def _stable_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    safe_values = [
        min(max(float(value), 0.0), DEFAULT_INVALID_CLE_PENALTY)
        if math.isfinite(value)
        else DEFAULT_INVALID_CLE_PENALTY
        for value in values
    ]
    return sum(safe_values) / len(safe_values)


def _percentile(values: Sequence[float], percentile: float) -> float:
    safe_values = sorted(float(value) for value in values if math.isfinite(value) and value >= 0.0)
    if not safe_values:
        return 0.0
    if len(safe_values) == 1:
        return safe_values[0]
    q = min(max(float(percentile), 0.0), 100.0) / 100.0
    position = q * (len(safe_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return safe_values[int(position)]
    weight = position - lower
    return safe_values[lower] * (1.0 - weight) + safe_values[upper] * weight


def _failure_rate(values: Sequence[float], threshold: float) -> float:
    safe_values = [float(value) for value in values if math.isfinite(value) and value >= 0.0]
    if not safe_values:
        return 0.0
    return sum(value > threshold for value in safe_values) / len(safe_values)

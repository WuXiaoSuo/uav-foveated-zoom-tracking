from __future__ import annotations

from dataclasses import dataclass

from .bbox import BBox, center_distance, iou


@dataclass(frozen=True)
class EvalSummary:
    frames: int
    mean_iou: float
    success_auc_0_1: float
    precision_20px: float
    mean_center_error: float


def evaluate(predictions: list[BBox], gt_boxes: list[BBox]) -> EvalSummary:
    n = min(len(predictions), len(gt_boxes))
    if n == 0:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0)

    ious = [iou(predictions[i], gt_boxes[i]) for i in range(n)]
    distances = [center_distance(predictions[i], gt_boxes[i]) for i in range(n)]
    thresholds = [i / 100.0 for i in range(101)]
    success_auc = sum(sum(v >= t for v in ious) / n for t in thresholds) / len(thresholds)
    precision_20 = sum(d <= 20.0 for d in distances) / n
    return EvalSummary(
        frames=n,
        mean_iou=sum(ious) / n,
        success_auc_0_1=success_auc,
        precision_20px=precision_20,
        mean_center_error=sum(distances) / n,
    )

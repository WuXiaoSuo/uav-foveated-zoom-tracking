from __future__ import annotations

from dataclasses import dataclass
import math

from .bbox import BBox, center_distance, iou
from .detector_yolo import Detection


@dataclass(frozen=True)
class MeasurementSelectorConfig:
    conf_min: float = 0.10
    iou_gate: float = 0.05
    dist_gate_factor: float = 2.0
    iou_weight: float = 0.6
    distance_weight: float = 0.3
    confidence_weight: float = 0.1


@dataclass(frozen=True)
class ByteTrackLikeSelectorConfig:
    high_conf_thresh: float = 0.35
    low_conf_thresh: float = 0.08
    iou_gate: float = 0.05
    center_dist_factor: float = 2.5
    area_ratio_min: float = 0.2
    area_ratio_max: float = 5.0
    iou_weight: float = 0.55
    distance_weight: float = 0.25
    area_weight: float = 0.10
    confidence_weight: float = 0.10
    ambiguity_gap: float = 0.20


@dataclass(frozen=True)
class SelectionResult:
    detection: Detection
    cost: float
    iou: float
    normalized_center_distance: float
    assoc_stage: str = "high"
    candidate_count: int = 0
    high_candidate_count: int = 0
    low_candidate_count: int = 0
    association_ambiguity: float = 0.0


class MeasurementSelector:
    """Select detection by geometric agreement with the predicted SOT box."""

    def __init__(self, config: MeasurementSelectorConfig | None = None) -> None:
        self.config = config or MeasurementSelectorConfig()

    def select(self, predicted_box: BBox, detections: list[Detection]) -> SelectionResult | None:
        best: SelectionResult | None = None
        pred_scale = max((predicted_box.w * predicted_box.h) ** 0.5, 1.0)
        distance_gate = float(self.config.dist_gate_factor) * pred_scale

        for detection in detections:
            if detection.confidence < self.config.conf_min:
                continue
            overlap = iou(predicted_box, detection.bbox)
            distance = center_distance(predicted_box, detection.bbox)
            if overlap < self.config.iou_gate:
                continue
            if distance > distance_gate:
                continue

            normalized_distance = distance / pred_scale
            cost = (
                self.config.iou_weight * (1.0 - overlap)
                + self.config.distance_weight * normalized_distance
                - self.config.confidence_weight * detection.confidence
            )
            result = SelectionResult(
                detection=detection,
                cost=float(cost),
                iou=float(overlap),
                normalized_center_distance=float(normalized_distance),
            )
            if best is None or result.cost < best.cost:
                best = result
        return best


class ByteTrackLikeSingleTargetSelector:
    """Two-stage association for one UAV123 target.

    This borrows ByteTrack's idea of using high-confidence detections first and
    low-confidence detections only for motion-gated recovery. It remains a
    single-target selector, not a MOT tracker.
    """

    def __init__(self, config: ByteTrackLikeSelectorConfig | None = None) -> None:
        self.config = config or ByteTrackLikeSelectorConfig()

    def select(
        self,
        predicted_box: BBox,
        detections: list[Detection],
        current_zoom_level: int = 1,
        lost_count: int = 0,
    ) -> SelectionResult | None:
        high = [det for det in detections if det.confidence >= self.config.high_conf_thresh]
        low = [
            det
            for det in detections
            if self.config.low_conf_thresh <= det.confidence < self.config.high_conf_thresh
        ]
        high_result = self._select_stage(
            predicted_box,
            high,
            stage="high",
            candidate_count=len(detections),
            high_candidate_count=len(high),
            low_candidate_count=len(low),
            strict=False,
        )
        if high_result is not None:
            return high_result

        low_result = self._select_stage(
            predicted_box,
            low,
            stage="low",
            candidate_count=len(detections),
            high_candidate_count=len(high),
            low_candidate_count=len(low),
            strict=True,
        )
        if low_result is not None:
            return low_result
        return None

    def _select_stage(
        self,
        predicted_box: BBox,
        detections: list[Detection],
        stage: str,
        candidate_count: int,
        high_candidate_count: int,
        low_candidate_count: int,
        strict: bool,
    ) -> SelectionResult | None:
        scored: list[SelectionResult] = []
        for detection in detections:
            result = self._score_detection(
                predicted_box,
                detection,
                stage=stage,
                candidate_count=candidate_count,
                high_candidate_count=high_candidate_count,
                low_candidate_count=low_candidate_count,
                strict=strict,
            )
            if result is not None:
                scored.append(result)
        if not scored:
            return None
        scored.sort(key=lambda item: item.cost)
        ambiguity = _association_ambiguity([item.cost for item in scored], self.config.ambiguity_gap)
        best = scored[0]
        return SelectionResult(
            detection=best.detection,
            cost=best.cost,
            iou=best.iou,
            normalized_center_distance=best.normalized_center_distance,
            assoc_stage=best.assoc_stage,
            candidate_count=best.candidate_count,
            high_candidate_count=best.high_candidate_count,
            low_candidate_count=best.low_candidate_count,
            association_ambiguity=ambiguity,
        )

    def _score_detection(
        self,
        predicted_box: BBox,
        detection: Detection,
        stage: str,
        candidate_count: int,
        high_candidate_count: int,
        low_candidate_count: int,
        strict: bool,
    ) -> SelectionResult | None:
        pred_area = max(predicted_box.area, 1.0)
        pred_scale = max(pred_area**0.5, 1.0)
        overlap = iou(predicted_box, detection.bbox)
        distance = center_distance(predicted_box, detection.bbox)
        if not math.isfinite(distance):
            return None
        area_ratio = max(detection.bbox.area, 1.0) / pred_area
        center_gate = self.config.center_dist_factor * pred_scale
        iou_gate = self.config.iou_gate
        area_min = self.config.area_ratio_min
        area_max = self.config.area_ratio_max
        if strict:
            center_gate *= 0.70
            iou_gate *= 1.5
            area_min = max(area_min, 0.35)
            area_max = min(area_max, 3.0)
        if overlap < iou_gate:
            return None
        if distance > center_gate:
            return None
        if area_ratio < area_min or area_ratio > area_max:
            return None

        normalized_distance = distance / pred_scale
        area_penalty = abs(math.log(max(area_ratio, 1e-9)))
        cost = (
            self.config.iou_weight * (1.0 - overlap)
            + self.config.distance_weight * normalized_distance
            + self.config.area_weight * area_penalty
            - self.config.confidence_weight * detection.confidence
        )
        return SelectionResult(
            detection=detection,
            cost=float(cost),
            iou=float(overlap),
            normalized_center_distance=float(normalized_distance),
            assoc_stage=stage,
            candidate_count=candidate_count,
            high_candidate_count=high_candidate_count,
            low_candidate_count=low_candidate_count,
            association_ambiguity=0.0,
        )


def _association_ambiguity(costs: list[float], ambiguity_gap: float) -> float:
    if len(costs) < 2:
        return 0.0
    gap = max(0.0, float(costs[1] - costs[0]))
    denom = max(float(ambiguity_gap), 1e-9)
    return min(max(1.0 - gap / denom, 0.0), 1.0)

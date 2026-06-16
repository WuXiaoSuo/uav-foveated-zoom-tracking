from __future__ import annotations

from dataclasses import dataclass
from math import exp
import math

from .bbox import BBox, center_distance


@dataclass(frozen=True)
class UncertaintyComponents:
    confidence_uncertainty: float
    innovation: float
    jitter: float
    edge_risk: float
    total: float


@dataclass(frozen=True)
class UncertaintyConfig:
    confidence_weight: float = 0.40
    innovation_weight: float = 0.25
    jitter_weight: float = 0.20
    edge_weight: float = 0.15
    edge_margin_ratio: float = 0.08
    lost_conf_decay: float = 0.85
    association_weight: float = 0.15
    blur_weight: float = 0.10
    blur_variance_low: float = 40.0
    blur_variance_high: float = 200.0


def effective_confidence(
    detection_confidence: float | None,
    lost_count: int,
    lost_conf_decay: float = 0.85,
) -> float:
    if detection_confidence is not None:
        return _clip01(detection_confidence)
    return _clip01(0.5 * (float(lost_conf_decay) ** max(0, int(lost_count))))


def bbox_jitter(previous_box: BBox | None, current_box: BBox) -> float:
    if previous_box is None:
        return 0.0
    scale = max((previous_box.w * previous_box.h) ** 0.5, 1.0)
    center_term = center_distance(previous_box, current_box) / scale
    size_term = (abs(previous_box.w - current_box.w) + abs(previous_box.h - current_box.h)) / max(
        previous_box.w + previous_box.h,
        1.0,
    )
    return _clip01(0.5 * center_term + 0.5 * size_term)


def edge_risk(box: BBox, image_width: int, image_height: int, margin_ratio: float = 0.08) -> float:
    margin = max(1.0, min(image_width, image_height) * float(margin_ratio))
    x1, y1, x2, y2 = box.to_xyxy()
    min_distance = min(x1, y1, image_width - x2, image_height - y2)
    return _clip01(1.0 - min_distance / margin)


def squash_innovation(value: float) -> float:
    return _clip01(1.0 - exp(-max(0.0, float(value))))


def compute_uncertainty(
    confidence: float,
    innovation: float,
    jitter: float,
    edge: float,
    association: float = 0.0,
    blur: float = 0.0,
    config: UncertaintyConfig | None = None,
) -> UncertaintyComponents:
    cfg = config or UncertaintyConfig()
    confidence_u = 1.0 - _clip01(confidence)
    innovation_u = squash_innovation(innovation)
    jitter_u = _clip01(jitter)
    edge_u = _clip01(edge)
    association_u = _clip01(association)
    blur_u = _clip01(blur)
    total_weight = max(
        cfg.confidence_weight
        + cfg.innovation_weight
        + cfg.jitter_weight
        + cfg.edge_weight
        + cfg.association_weight
        + cfg.blur_weight,
        1e-9,
    )
    total = (
        cfg.confidence_weight * confidence_u
        + cfg.innovation_weight * innovation_u
        + cfg.jitter_weight * jitter_u
        + cfg.edge_weight * edge_u
        + cfg.association_weight * association_u
        + cfg.blur_weight * blur_u
    ) / total_weight
    return UncertaintyComponents(
        confidence_uncertainty=confidence_u,
        innovation=innovation_u,
        jitter=jitter_u,
        edge_risk=edge_u,
        total=_clip01(total),
    )


def estimate_blur_risk(image, variance_low: float = 40.0, variance_high: float = 200.0) -> float:
    """Laplacian-variance blur risk in [0, 1], where 1 means very blurry."""
    try:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if getattr(image, "ndim", 0) == 3 else image
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0
    if not math.isfinite(variance):
        return 0.0
    low = max(float(variance_low), 1e-9)
    high = max(float(variance_high), low + 1e-9)
    return _clip01((high - variance) / (high - low))


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(max(float(value), 0.0), 1.0)

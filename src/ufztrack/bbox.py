from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


EPS = 1e-9


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bbox in original image coordinates: x, y, w, h."""

    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_xyxy(cls, xyxy: Iterable[float]) -> "BBox":
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        return cls(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))

    @classmethod
    def from_xywh(cls, xywh: Iterable[float]) -> "BBox":
        x, y, w, h = [float(v) for v in xywh]
        return cls(x, y, w, h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        if not bbox_is_valid(self):
            return 0.0
        area = self.w * self.h
        return area if math.isfinite(area) else 0.0

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def clip(self, image_width: int, image_height: int, min_size: float = 1.0) -> "BBox":
        if image_width <= 0 or image_height <= 0:
            return BBox(0.0, 0.0, max(float(min_size), 1.0), max(float(min_size), 1.0))
        if not bbox_is_finite(self):
            w = min(max(float(min_size), 1.0), float(image_width))
            h = min(max(float(min_size), 1.0), float(image_height))
            return BBox(0.0, 0.0, w, h)
        x1, y1, x2, y2 = self.to_xyxy()
        x1 = min(max(0.0, x1), float(image_width - 1))
        y1 = min(max(0.0, y1), float(image_height - 1))
        x2 = min(max(x1 + min_size, x2), float(image_width))
        y2 = min(max(y1 + min_size, y2), float(image_height))
        return BBox(x1, y1, x2 - x1, y2 - y1)


def bbox_is_finite(box: BBox) -> bool:
    return all(math.isfinite(value) for value in box.to_xywh())


def bbox_is_valid(box: BBox, eps: float = EPS) -> bool:
    if not bbox_is_finite(box):
        return False
    if box.w <= eps or box.h <= eps:
        return False
    x1, y1, x2, y2 = box.to_xyxy()
    return all(math.isfinite(value) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def default_bbox(image_width: int, image_height: int, min_size: float = 1.0) -> BBox:
    width = max(1.0, float(image_width))
    height = max(1.0, float(image_height))
    size = min(max(float(min_size), 1.0), width, height)
    return BBox((width - size) / 2.0, (height - size) / 2.0, size, size)


def sanitize_bbox(
    box: BBox,
    image_width: int,
    image_height: int,
    fallback: BBox | None = None,
    min_size: float = 1.0,
) -> tuple[BBox, bool]:
    """Return a finite, positive, clipped bbox and whether the input was valid."""
    input_valid = bbox_is_valid(box)
    candidate = box if input_valid else fallback
    if candidate is None or not bbox_is_valid(candidate):
        candidate = default_bbox(image_width, image_height, min_size=min_size)
    clipped = candidate.clip(image_width, image_height, min_size=min_size)
    if not bbox_is_valid(clipped):
        clipped = default_bbox(image_width, image_height, min_size=min_size).clip(
            image_width,
            image_height,
            min_size=min_size,
        )
    return clipped, input_valid


def iou(a: BBox, b: BBox) -> float:
    if not bbox_is_valid(a) or not bbox_is_valid(b):
        return 0.0
    ax1, ay1, ax2, ay2 = a.to_xyxy()
    bx1, by1, bx2, by2 = b.to_xyxy()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if not math.isfinite(inter) or not math.isfinite(union) or union <= EPS:
        return 0.0
    value = inter / union
    if not math.isfinite(value):
        return 0.0
    return min(max(value, 0.0), 1.0)


def center_distance(a: BBox, b: BBox) -> float:
    if not bbox_is_valid(a) or not bbox_is_valid(b):
        return math.nan
    dx = a.cx - b.cx
    dy = a.cy - b.cy
    value = math.hypot(dx, dy)
    return value if math.isfinite(value) else math.nan

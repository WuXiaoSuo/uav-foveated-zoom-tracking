from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .bbox import BBox


@dataclass(frozen=True)
class CropInfo:
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    output_w: int
    output_h: int
    zoom_level: int
    crop_mode: str

    @property
    def scale_x(self) -> float:
        return self.output_w / float(self.crop_w)

    @property
    def scale_y(self) -> float:
        return self.output_h / float(self.crop_h)


class ZoomSimulator:
    """Software zoom by crop-and-resize, with explicit bbox coordinate maps."""

    def __init__(self, levels: Sequence[int] = (1, 2, 4, 8), crop_mode: str = "predicted_center"):
        self.levels = tuple(int(v) for v in levels)
        if self.levels != (1, 2, 4, 8):
            allowed = {1, 2, 4, 8}
            if any(level not in allowed for level in self.levels):
                raise ValueError(f"Unsupported zoom levels: {self.levels}")
        if crop_mode not in {"predicted_center", "image_center"}:
            raise ValueError(f"Unsupported crop_mode: {crop_mode}")
        self.crop_mode = crop_mode

    def simulate(
        self,
        image: np.ndarray,
        zoom_level: int,
        predicted_bbox: BBox | None = None,
        uncertainty: float = 0.0,
        context_margin_factor: float = 0.0,
        uncertainty_margin_gain: float = 0.0,
        min_context_pixels: int = 0,
    ) -> tuple[np.ndarray, CropInfo]:
        height, width = image.shape[:2]
        level = int(zoom_level)
        if level not in self.levels:
            raise ValueError(f"zoom_level {level} is not in configured levels {self.levels}")

        if self.crop_mode == "predicted_center":
            if predicted_bbox is None:
                raise ValueError("predicted_bbox is required for crop_mode='predicted_center'")
            center_x, center_y = predicted_bbox.cx, predicted_bbox.cy
        else:
            center_x, center_y = width / 2.0, height / 2.0

        nominal_crop_w = max(2.0, width / float(level))
        nominal_crop_h = max(2.0, height / float(level))
        context_multiplier = 1.0 + max(0.0, float(context_margin_factor)) + max(
            0.0,
            float(uncertainty_margin_gain),
        ) * min(max(float(uncertainty), 0.0), 1.0)
        crop_w = nominal_crop_w * context_multiplier
        crop_h = nominal_crop_h * context_multiplier
        if predicted_bbox is not None:
            min_context = max(0, int(min_context_pixels))
            crop_w = max(crop_w, predicted_bbox.w + 2.0 * min_context)
            crop_h = max(crop_h, predicted_bbox.h + 2.0 * min_context)
        crop_w = min(width, max(2, int(round(crop_w))))
        crop_h = min(height, max(2, int(round(crop_h))))
        crop_x = _clamp_int(int(round(center_x - crop_w / 2.0)), 0, width - crop_w)
        crop_y = _clamp_int(int(round(center_y - crop_h / 2.0)), 0, height - crop_h)

        crop = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        zoomed = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
        return zoomed, CropInfo(
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            output_w=width,
            output_h=height,
            zoom_level=level,
            crop_mode=self.crop_mode,
        )


def original_to_zoomed_bbox(box: BBox, crop: CropInfo) -> BBox:
    x = (box.x - crop.crop_x) * crop.scale_x
    y = (box.y - crop.crop_y) * crop.scale_y
    w = box.w * crop.scale_x
    h = box.h * crop.scale_y
    return BBox(x, y, w, h)


def zoomed_to_original_bbox(box: BBox, crop: CropInfo) -> BBox:
    x = crop.crop_x + box.x / crop.scale_x
    y = crop.crop_y + box.y / crop.scale_y
    w = box.w / crop.scale_x
    h = box.h / crop.scale_y
    return BBox(x, y, w, h)


def _clamp_int(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return min(max(value, low), high)

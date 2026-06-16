from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .bbox import BBox


@dataclass(frozen=True)
class CropInfo:
    x0: int
    y0: int
    x1: int
    y1: int
    scale_x: float
    scale_y: float
    output_width: int
    output_height: int


def choose_zoom_level(position_trace: float, thresholds: Sequence[dict], fallback: int = 1) -> int:
    for item in thresholds:
        if position_trace <= float(item["max_trace"]):
            return int(item["level"])
    return int(fallback)


def make_zoom_crop(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    zoom_level: int,
    padding: float = 1.0,
) -> tuple[np.ndarray, CropInfo]:
    height, width = image.shape[:2]
    level = max(1, int(zoom_level))
    crop_w = max(2, int(round(width / level * padding)))
    crop_h = max(2, int(round(height / level * padding)))
    crop_w = min(width, crop_w)
    crop_h = min(height, crop_h)

    x0 = int(round(center_x - crop_w / 2.0))
    y0 = int(round(center_y - crop_h / 2.0))
    x0 = min(max(0, x0), width - crop_w)
    y0 = min(max(0, y0), height - crop_h)
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    crop = image[y0:y1, x0:x1]
    zoomed = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    info = CropInfo(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        scale_x=width / float(crop_w),
        scale_y=height / float(crop_h),
        output_width=width,
        output_height=height,
    )
    return zoomed, info


def bbox_from_zoom_to_original(box: BBox, crop: CropInfo) -> BBox:
    x = crop.x0 + box.x / crop.scale_x
    y = crop.y0 + box.y / crop.scale_y
    w = box.w / crop.scale_x
    h = box.h / crop.scale_y
    return BBox(x, y, w, h)

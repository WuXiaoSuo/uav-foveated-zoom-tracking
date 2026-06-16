from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bbox import BBox, center_distance


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int


class YOLOv8Detector:
    def __init__(self, weights: str, conf: float, iou: float, imgsz: int, classes=None, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("未安装 ultralytics，请先安装 requirements.txt。") from exc

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.classes = classes
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            conf = float(box.conf[0].detach().cpu().item())
            cls = int(box.cls[0].detach().cpu().item())
            detections.append(Detection(BBox.from_xyxy(xyxy), conf, cls))
        return detections


def select_measurement(
    detections: list[Detection],
    predicted_box: BBox,
    image_width: int,
    image_height: int,
    max_center_distance_ratio: float,
    score_weight: float,
) -> Detection | None:
    if not detections:
        return None

    max_dist = max_center_distance_ratio * ((image_width * image_width + image_height * image_height) ** 0.5)
    best: tuple[float, Detection] | None = None
    for detection in detections:
        dist = center_distance(detection.bbox, predicted_box)
        if dist > max_dist:
            continue
        score = dist - score_weight * detection.confidence * max_dist
        if best is None or score < best[0]:
            best = (score, detection)
    return best[1] if best else None

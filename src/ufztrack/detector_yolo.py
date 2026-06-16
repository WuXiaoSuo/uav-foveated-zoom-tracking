from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bbox import BBox


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_id: int | None = None


class YOLOv8Detector:
    """Thin wrapper around Ultralytics YOLOv8 for offline measurement generation."""

    def __init__(
        self,
        model: str,
        conf: float = 0.10,
        iou: float = 0.70,
        imgsz: int = 640,
        classes: list[int] | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLOv8 detection.") from exc

        self.model = YOLO(model)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
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
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else [None] * len(xyxy)

        detections: list[Detection] = []
        for coords, confidence, class_id in zip(xyxy, confs, class_ids):
            detections.append(
                Detection(
                    bbox=BBox.from_xyxy(coords),
                    confidence=float(confidence),
                    class_id=None if class_id is None else int(class_id),
                )
            )
        return detections

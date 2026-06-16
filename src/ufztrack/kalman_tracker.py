from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bbox import BBox


@dataclass(frozen=True)
class KalmanConfig:
    dt: float = 1.0
    process_noise: float = 25.0
    measurement_noise: float = 36.0
    initial_velocity_variance: float = 400.0
    min_box_size: float = 2.0


class KalmanBoxTracker:
    """Constant-velocity Kalman tracker in original image coordinates."""

    def __init__(self, initial_box: BBox, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        self.x = np.zeros((8, 1), dtype=float)
        self.x[:4, 0] = [initial_box.cx, initial_box.cy, initial_box.w, initial_box.h]
        self.p = np.eye(8, dtype=float)
        self.p[:4, :4] *= self.config.measurement_noise
        self.p[4:, 4:] *= self.config.initial_velocity_variance
        self.lost_count = 0
        self.last_innovation = 0.0

    def predict(self) -> BBox:
        dt = float(self.config.dt)
        f = np.eye(8, dtype=float)
        for i in range(4):
            f[i, i + 4] = dt

        q = np.eye(8, dtype=float) * float(self.config.process_noise)
        self.x = np.dot(f, self.x)
        self.p = np.dot(np.dot(f, self.p), f.T) + q
        self.x[2, 0] = max(self.config.min_box_size, self.x[2, 0])
        self.x[3, 0] = max(self.config.min_box_size, self.x[3, 0])
        self.lost_count += 1
        return self.current_bbox()

    def update(self, measurement: BBox) -> BBox:
        z = np.array([[measurement.cx], [measurement.cy], [measurement.w], [measurement.h]], dtype=float)
        h = np.zeros((4, 8), dtype=float)
        h[:4, :4] = np.eye(4, dtype=float)
        r = np.eye(4, dtype=float) * float(self.config.measurement_noise)

        predicted_z = np.dot(h, self.x)
        innovation = z - predicted_z
        s = np.dot(np.dot(h, self.p), h.T) + r
        k = np.dot(np.dot(self.p, h.T), np.linalg.inv(s))

        self.x = self.x + np.dot(k, innovation)
        self.p = np.dot(np.eye(8, dtype=float) - np.dot(k, h), self.p)
        self.x[2, 0] = max(self.config.min_box_size, self.x[2, 0])
        self.x[3, 0] = max(self.config.min_box_size, self.x[3, 0])
        self.last_innovation = _normalized_innovation(innovation, predicted_z)
        self.lost_count = 0
        return self.current_bbox()

    def current_bbox(self) -> BBox:
        cx, cy, w, h = self.x[:4, 0]
        w = max(float(self.config.min_box_size), float(w))
        h = max(float(self.config.min_box_size), float(h))
        return BBox(float(cx - w / 2.0), float(cy - h / 2.0), w, h)

    def position_trace(self) -> float:
        return float(np.trace(self.p[:2, :2]))


def _normalized_innovation(innovation: np.ndarray, predicted_z: np.ndarray) -> float:
    cx_delta, cy_delta, w_delta, h_delta = innovation[:, 0]
    _, _, pred_w, pred_h = predicted_z[:, 0]
    center_scale = max((max(pred_w, 1.0) * max(pred_h, 1.0)) ** 0.5, 1.0)
    size_scale = max(abs(pred_w) + abs(pred_h), 1.0)
    center_term = ((cx_delta / center_scale) ** 2 + (cy_delta / center_scale) ** 2) ** 0.5
    size_term = (abs(w_delta) + abs(h_delta)) / size_scale
    return float(center_term + size_term)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bbox import BBox


@dataclass
class KalmanConfig:
    dt: float = 1.0
    process_noise: float = 25.0
    measurement_noise: float = 36.0
    initial_velocity_variance: float = 400.0
    min_box_size: float = 2.0
    max_missed: int = 30


class BoxKalmanFilter:
    """Constant-velocity Kalman filter for one SOT target.

    State: [cx, cy, w, h, vx, vy, vw, vh].
    Measurement: [cx, cy, w, h].
    """

    def __init__(self, initial_box: BBox, config: KalmanConfig):
        self.config = config
        self.x = np.zeros((8, 1), dtype=float)
        self.x[:4, 0] = [initial_box.cx, initial_box.cy, initial_box.w, initial_box.h]
        self.p = np.eye(8, dtype=float)
        self.p[:4, :4] *= config.measurement_noise
        self.p[4:, 4:] *= config.initial_velocity_variance
        self.missed = 0

    def predict(self) -> BBox:
        dt = self.config.dt
        f = np.eye(8, dtype=float)
        for i in range(4):
            f[i, i + 4] = dt
        q = np.eye(8, dtype=float) * self.config.process_noise
        self.x = np.dot(f, self.x)
        self.p = np.dot(np.dot(f, self.p), f.T) + q
        self.missed += 1
        return self.current_box()

    def update(self, measurement: BBox) -> BBox:
        z = np.array([[measurement.cx], [measurement.cy], [measurement.w], [measurement.h]], dtype=float)
        h = np.zeros((4, 8), dtype=float)
        h[:4, :4] = np.eye(4, dtype=float)
        r = np.eye(4, dtype=float) * self.config.measurement_noise
        y = z - np.dot(h, self.x)
        s = np.dot(np.dot(h, self.p), h.T) + r
        k = np.dot(np.dot(self.p, h.T), np.linalg.inv(s))
        self.x = self.x + np.dot(k, y)
        self.p = np.dot(np.eye(8, dtype=float) - np.dot(k, h), self.p)
        self.x[2, 0] = max(self.config.min_box_size, self.x[2, 0])
        self.x[3, 0] = max(self.config.min_box_size, self.x[3, 0])
        self.missed = 0
        return self.current_box()

    def current_box(self) -> BBox:
        cx, cy, w, h = self.x[:4, 0]
        w = max(self.config.min_box_size, float(w))
        h = max(self.config.min_box_size, float(h))
        return BBox(float(cx - w / 2.0), float(cy - h / 2.0), w, h)

    def position_trace(self) -> float:
        return float(np.trace(self.p[:2, :2]))

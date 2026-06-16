from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import precision_curve, success_curve


def plot_precision_curves(
    errors_by_method: Mapping[str, Sequence[float]],
    output_path: str | Path,
    max_threshold: int = 50,
) -> None:
    thresholds = list(range(max_threshold + 1))
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for method, errors in sorted(errors_by_method.items()):
        ax.plot(thresholds, precision_curve(list(errors), thresholds), label=method)
    ax.set_xlabel("Center error threshold (px)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, max_threshold)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, output_path)


def plot_success_curves(
    ious_by_method: Mapping[str, Sequence[float]],
    output_path: str | Path,
) -> None:
    thresholds = [i / 100.0 for i in range(101)]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for method, ious in sorted(ious_by_method.items()):
        ax.plot(thresholds, success_curve(list(ious), thresholds), label=method)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Success rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, output_path)


def plot_temporal_curve(log_rows: Sequence[dict[str, str]], output_path: str | Path, title: str) -> None:
    frames = [int(row["frame"]) for row in log_rows]
    zoom = [float(row["zoom_level"]) for row in log_rows]
    uncertainty = [float(row["uncertainty"]) for row in log_rows]
    area = [float(row["area"]) for row in log_rows]
    confidence = [float(row["conf"]) for row in log_rows]
    lost = [float(row["lost"]) for row in log_rows]

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 7.0), sharex=True)
    fig.suptitle(title)
    axes[0].step(frames, zoom, where="post")
    axes[0].set_ylabel("Zoom")
    axes[0].set_yticks([1, 2, 4, 8])
    axes[1].plot(frames, uncertainty, label="uncertainty")
    axes[1].plot(frames, confidence, label="confidence")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="best")
    axes[2].plot(frames, area)
    axes[2].set_ylabel("Area ratio")
    axes[3].plot(frames, lost)
    axes[3].set_ylabel("Lost")
    axes[3].set_xlabel("Frame")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    _save(fig, output_path)


def _save(fig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

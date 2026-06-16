from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .bbox import BBox


@dataclass(frozen=True)
class SequenceData:
    name: str
    root: Path
    image_paths: list[Path]
    gt_boxes: list[BBox]


def _read_gt_file(path: Path) -> list[BBox]:
    boxes: list[BBox] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            raise ValueError(f"GT 格式错误：{path}:{line_no}")
        boxes.append(BBox.from_xywh(float(v) for v in parts[:4]))
    return boxes


def _find_gt_file(sequence_dir: Path, gt_files: Sequence[str]) -> Path:
    for rel in gt_files:
        path = sequence_dir / rel
        if path.is_file():
            return path
    raise FileNotFoundError(f"未找到 GT 文件：{sequence_dir}")


def _find_images(sequence_dir: Path, image_globs: Sequence[str]) -> list[Path]:
    images: list[Path] = []
    for pattern in image_globs:
        images.extend(sequence_dir.glob(pattern))
        if images:
            break
    images = sorted(set(images))
    if not images:
        raise FileNotFoundError(f"未找到图像帧：{sequence_dir}")
    return images


def load_sequence(
    dataset_root: str | Path,
    sequence: str | None,
    image_globs: Sequence[str],
    gt_files: Sequence[str],
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> SequenceData:
    root = Path(dataset_root).expanduser()
    sequence_dir = root / sequence if sequence else root
    sequence_dir = sequence_dir.resolve()

    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"序列目录不存在：{sequence_dir}")

    images = _find_images(sequence_dir, image_globs)
    gt_boxes = _read_gt_file(_find_gt_file(sequence_dir, gt_files))
    n = min(len(images), len(gt_boxes))
    if n == 0:
        raise ValueError(f"序列为空：{sequence_dir}")
    images = images[:n]
    gt_boxes = gt_boxes[:n]

    stride = max(1, int(frame_stride))
    indices = list(range(0, n, stride))
    if max_frames is not None:
        indices = indices[: int(max_frames)]

    return SequenceData(
        name=sequence_dir.name,
        root=sequence_dir,
        image_paths=[images[i] for i in indices],
        gt_boxes=[gt_boxes[i] for i in indices],
    )

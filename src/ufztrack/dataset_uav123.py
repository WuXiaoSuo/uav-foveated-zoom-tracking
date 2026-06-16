from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .bbox import BBox


@dataclass(frozen=True)
class UAV123Sequence:
    name: str
    image_paths: list[Path]
    gt_boxes: list[BBox]
    image_dir: Path
    anno_path: Path


def read_uav123_bbox_file(path: str | Path) -> list[BBox]:
    """Read UAV123 bbox annotations in x,y,w,h format."""
    anno_path = Path(path)
    boxes: list[BBox] = []
    for line_no, line in enumerate(anno_path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        parts = text.replace(",", " ").split()
        if len(parts) < 4:
            raise ValueError(f"Invalid bbox annotation at {anno_path}:{line_no}: {line!r}")
        try:
            x, y, w, h = (float(v) for v in parts[:4])
        except ValueError as exc:
            raise ValueError(f"Invalid bbox numbers at {anno_path}:{line_no}: {line!r}") from exc
        boxes.append(BBox.from_xywh((x, y, w, h)))
    if not boxes:
        raise ValueError(f"Empty bbox annotation file: {anno_path}")
    return boxes


def sorted_frame_paths(image_dir: str | Path, frame_glob: str = "*.jpg") -> list[Path]:
    """Return frame paths sorted by filename, e.g. 000001.jpg, 000002.jpg."""
    root = Path(image_dir)
    frames = sorted(root.glob(frame_glob), key=lambda p: p.name)
    if not frames:
        raise FileNotFoundError(f"No frames found in {root} with glob {frame_glob!r}")
    return frames


def list_sequences(
    image_root: str | Path,
    bbox_anno_root: str | Path,
    requested: Iterable[str] | None = None,
) -> list[str]:
    image_root = Path(image_root)
    bbox_root = Path(bbox_anno_root)

    if requested:
        return sorted({str(name) for name in requested if str(name).strip()})

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    if not bbox_root.is_dir():
        raise FileNotFoundError(f"BBox annotation root does not exist: {bbox_root}")

    names: list[str] = []
    for sequence_dir in sorted(image_root.iterdir(), key=lambda p: p.name):
        if not sequence_dir.is_dir():
            continue
        if (bbox_root / f"{sequence_dir.name}.txt").is_file():
            names.append(sequence_dir.name)
    if not names:
        raise FileNotFoundError(f"No UAV123 sequences with bbox annotations found under {image_root}")
    return names


def load_uav123_sequence(
    sequence_name: str,
    image_root: str | Path,
    bbox_anno_root: str | Path,
    frame_glob: str = "*.jpg",
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> UAV123Sequence:
    """Load one UAV123@10fps sequence without reading attribute labels as boxes."""
    image_dir = Path(image_root) / sequence_name
    anno_path = Path(bbox_anno_root) / f"{sequence_name}.txt"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Sequence image directory does not exist: {image_dir}")
    if not anno_path.is_file():
        raise FileNotFoundError(f"BBox annotation file does not exist: {anno_path}")
    if anno_path.parent.name == "att":
        raise ValueError(f"Attribute labels are not bbox annotations: {anno_path}")

    frames = sorted_frame_paths(image_dir, frame_glob=frame_glob)
    boxes = read_uav123_bbox_file(anno_path)
    n = min(len(frames), len(boxes))
    if n == 0:
        raise ValueError(f"Empty sequence: {sequence_name}")

    stride = max(1, int(frame_stride))
    indices = list(range(0, n, stride))
    if max_frames is not None:
        indices = indices[: max(0, int(max_frames))]
    if not indices:
        raise ValueError(f"No frames remain after stride/max_frames for sequence: {sequence_name}")

    return UAV123Sequence(
        name=sequence_name,
        image_paths=[frames[i] for i in indices],
        gt_boxes=[boxes[i] for i in indices],
        image_dir=image_dir,
        anno_path=anno_path,
    )

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check UAV123@10fps paths for UFZ-Track Experiment 1.")
    parser.add_argument("--config", default="configs/uav123_10fps.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only validate and print a summary.")
    parser.add_argument("--sequences", nargs="+", default=None, help="Optional sequence names or comma lists.")
    return parser.parse_args()


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _split_sequences(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def main() -> int:
    _add_src_to_path()
    from ufztrack.dataset_uav123 import list_sequences, load_uav123_sequence

    args = parse_args()
    cfg = _load_config(args.config)
    dataset_cfg = cfg["dataset"]
    image_root = Path(dataset_cfg["image_root"])
    bbox_root = Path(dataset_cfg["bbox_anno_root"])
    attribute_root = Path(dataset_cfg["attribute_root"])
    requested = _split_sequences(args.sequences) or dataset_cfg.get("sequences")

    missing = [str(path) for path in [image_root, bbox_root, attribute_root] if not path.exists()]
    if missing:
        payload = {
            "ok": False,
            "dry_run": bool(args.dry_run),
            "missing_paths": missing,
            "note": "attribute_root is checked only for existence; it is not used as bbox GT.",
        }
        print(json.dumps(payload, indent=2))
        return 0 if args.dry_run else 2

    sequence_names = list_sequences(image_root, bbox_root, requested=requested)
    summaries = []
    for name in sequence_names:
        seq = load_uav123_sequence(
            name,
            image_root=image_root,
            bbox_anno_root=bbox_root,
            frame_glob=dataset_cfg.get("frame_glob", "*.jpg"),
            frame_stride=dataset_cfg.get("frame_stride", 1),
            max_frames=dataset_cfg.get("max_frames"),
        )
        summaries.append(
            {
                "name": seq.name,
                "frames": len(seq.image_paths),
                "gt_boxes": len(seq.gt_boxes),
                "first_frame": str(seq.image_paths[0]),
                "bbox_annotation": str(seq.anno_path),
            }
        )

    payload = {
        "ok": True,
        "dry_run": bool(args.dry_run),
        "dataset_root": dataset_cfg["root"],
        "image_root": str(image_root),
        "bbox_anno_root": str(bbox_root),
        "attribute_root": str(attribute_root),
        "num_sequences": len(summaries),
        "sequences": summaries,
        "note": "BBox GT is read from anno/UAV123_10fps/{sequence}.txt; att files are not bbox annotations.",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

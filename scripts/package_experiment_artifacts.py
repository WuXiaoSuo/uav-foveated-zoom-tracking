#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import warnings
from pathlib import Path


DEFAULT_CASES = [
    "bike1",
    "boat9",
    "truck1",
    "building4",
    "person13",
    "car17",
    "person1",
    "person20",
    "car5",
    "car10",
    "car9",
    "boat6",
    "boat7",
]
DEFAULT_METHODS = ["ufz", "ufz_v2", "ufz_v2_2"]
DEFAULT_ARTIFACT_ROOT = "artifacts/experiments/exp1_uav123_10fps_yolov8n_ufz_v2_2"
DEFAULT_SUMMARY = "docs/experiments/runs/20260616_full_valid_ufz_v2_2_yolov8n_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package GitHub-safe UFZ Experiment 1 paper artifacts.")
    parser.add_argument("--output-root", default="/root/autodl-tmp/UFZTrack/outputs")
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--cases", nargs="+", default=None, help="Selected case names or comma lists.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Selected log methods or comma lists.")
    return parser.parse_args()


def _split(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        warnings.warn(f"Missing artifact source, skipping: {src}", RuntimeWarning, stacklevel=2)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _write_readme(path: Path) -> None:
    text = """# UFZ-Track Experiment 1 Artifacts

This package contains GitHub-safe paper artifacts for Experiment 1 on UAV123@10fps with YOLOv8n and offline simulated zoom.

Contents:

- `tables/`: aggregate metrics, robust metrics, macro summaries, win counts, and selected case analysis.
- `figures/cases/`: selected PNG case plots only.
- `logs_selected/`: selected per-sequence CSV logs for `ufz`, `ufz_v2`, and `ufz_v2_2`.
- `summaries/`: the experiment summary markdown.

No raw UAV123 images, videos, model weights, detector outputs, or large result directories are included.

Protocol notes:

- UAV123 ground truth is used only for frame-1 tracker initialization and offline evaluation.
- Crop centers, measurement selection, tracker updates, and zoom decisions do not use future GT.
- The main frozen method is `ufz_v2_2`: UFZ v1 zoom behavior with hard-risk veto and lost recovery.
- Zoom is software-simulated offline using original-image coordinate output boxes.

Large archives, if created manually, should be stored under `artifacts/archives/` and handled with Git LFS or external release storage.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_manifest(root: Path) -> None:
    manifest_path = root / "MANIFEST_SHA256.txt"
    rows: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    artifact_root = Path(args.artifact_root)
    cases = _split(args.cases, DEFAULT_CASES)
    methods = _split(args.methods, DEFAULT_METHODS)

    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_readme(artifact_root / "README.md")

    table_names = [
        "main_results.csv",
        "robust_results.csv",
        "macro_results.csv",
        "win_counts.csv",
        "ufz_case_analysis.csv",
        "ufz_v2_2_case_analysis.csv",
        "selected_paper_cases.csv",
    ]
    for table_name in table_names:
        _copy_if_exists(output_root / "tables" / table_name, artifact_root / "tables" / table_name)

    source_cases_dir = output_root / "figures" / "cases"
    for sequence in cases:
        for path in sorted(source_cases_dir.glob(f"{sequence}_*.png")):
            _copy_if_exists(path, artifact_root / "figures" / "cases" / path.name)

    for method in methods:
        for sequence in cases:
            _copy_if_exists(
                output_root / "logs" / method / f"{sequence}.csv",
                artifact_root / "logs_selected" / method / f"{sequence}.csv",
            )

    summary_path = Path(args.summary)
    _copy_if_exists(summary_path, artifact_root / "summaries" / summary_path.name)
    _write_manifest(artifact_root)

    print(f"Wrote artifact package: {artifact_root}")
    print(f"Wrote manifest: {artifact_root / 'MANIFEST_SHA256.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

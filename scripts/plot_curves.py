#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Plot one UAV123 temporal zoom curve.")
    parser.add_argument("--config", default="configs/uav123_10fps.yaml")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--method", required=True)
    return parser.parse_args()


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    _add_src_to_path()
    from ufztrack.visualization import plot_temporal_curve

    args = parse_args()
    cfg = _load_config(args.config)
    output_root = Path(cfg["outputs"]["root"])
    log_path = output_root / "logs" / args.method / f"{args.sequence}.csv"
    figure_path = output_root / "figures" / f"{args.sequence}_temporal_curve.pdf"
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    with log_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty log file: {log_path}")

    plot_temporal_curve(rows, figure_path, title=f"{args.method} / {args.sequence}")
    print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 UFZ-Track Experiment 1：UAV123@10fps offline simulated zoom。")
    parser.add_argument(
        "--config",
        default="configs/experiment1_uav123_10fps_yolov8n.yaml",
        help="实验配置文件路径。",
    )
    return parser.parse_args()


def main() -> int:
    _add_src_to_path()
    args = parse_args()

    from ufztrack.runner import run_experiment

    summary = run_experiment(args.config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

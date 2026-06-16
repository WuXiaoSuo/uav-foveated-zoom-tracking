#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import warnings
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize UFZ-Track zoom levels and commands from log CSV files.")
    parser.add_argument("--output-root", default="/root/autodl-tmp/UFZTrack/outputs")
    parser.add_argument("--method", required=True)
    parser.add_argument("--sequences", nargs="+", default=None, help="Sequence names or comma lists.")
    return parser.parse_args()


def _split_sequences(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    names: list[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _read_log(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _command_summary(counter: Counter[str]) -> str:
    return ";".join(f"{command}:{counter[command]}" for command in sorted(counter))


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    logs_dir = output_root / "logs" / args.method
    tables_dir = output_root / "tables"
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {logs_dir}")

    sequence_names = _split_sequences(args.sequences)
    if sequence_names is None:
        sequence_names = [path.stem for path in sorted(logs_dir.glob("*.csv"), key=lambda p: p.name)]

    rows: list[dict[str, str | int]] = []
    for sequence in sequence_names:
        log_path = logs_dir / f"{sequence}.csv"
        if not log_path.is_file():
            warnings.warn(f"Missing log file, skipping: {log_path}", RuntimeWarning, stacklevel=2)
            continue
        log_rows = _read_log(log_path)
        zoom_counter = Counter()
        command_counter = Counter()
        for row in log_rows:
            zoom = row.get("zoom_level", "").strip()
            if zoom in {"1", "2", "4", "8"}:
                zoom_counter[zoom] += 1
            else:
                warnings.warn(
                    f"{sequence}: invalid zoom_level {zoom!r} in {log_path}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            command = row.get("command", "").strip() or "unknown"
            command_counter[command] += 1

        rows.append(
            {
                "sequence": sequence,
                "total_frames": len(log_rows),
                "zoom_1": zoom_counter["1"],
                "zoom_2": zoom_counter["2"],
                "zoom_4": zoom_counter["4"],
                "zoom_8": zoom_counter["8"],
                "command_summary": _command_summary(command_counter),
            }
        )

    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "zoom_behavior_summary.csv"
    fieldnames = ["sequence", "total_frames", "zoom_1", "zoom_2", "zoom_4", "zoom_8", "command_summary"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

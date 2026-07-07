#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import yaml
except ImportError as exc:  # pragma: no cover - deployment dependency guard
    raise SystemExit("PyYAML is required. Install requirements.txt first.") from exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config_h30_deploy_v1.yaml")

DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "source": "",
        "type": "auto",
        "fps": 30.0,
        "max_frames": None,
        "init_bbox": None,
        "interactive_init": True,
    },
    "tracker": {
        "type": "auto",
        "template_min_score": 0.45,
        "template_search_scale": 2.5,
        "template_update_alpha": 0.05,
    },
    "gimbal": {
        "enabled": True,
        "command_sink": "dry_run",
        "deadband": 0.015,
        "yaw_kp": 35.0,
        "pitch_kp": -28.0,
        "max_yaw_rate_deg_s": 18.0,
        "max_pitch_rate_deg_s": 14.0,
    },
    "zoom": {
        "levels": [2, 5, 10],
        "disabled_levels": [20, 40],
        "initial_level": 2,
        "zoom_in_area_ratio": 0.012,
        "zoom_out_area_ratio": 0.100,
        "max_area_ratio": 0.160,
        "zoom_in_min_edge_margin": 0.160,
        "safe_edge_margin": 0.100,
        "danger_edge_margin": 0.050,
        "stable_frames_for_zoom_in": 12,
        "lost_to_wide_frames": 3,
        "cooldown_frames": 18,
    },
    "output": {
        "root": "outputs/deploy_v1",
        "draw_video": True,
        "video_name": "ufz_deploy_v1_vis.mp4",
        "csv_name": "ufz_deploy_v1_log.csv",
        "metadata_name": "ufz_deploy_v1_metadata.json",
    },
}

CSV_FIELDS = [
    "frame_index",
    "timestamp_s",
    "source_name",
    "tracker_valid",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "center_error_x",
    "center_error_y",
    "area_ratio",
    "edge_margin",
    "zoom_level",
    "zoom_command",
    "zoom_reason",
    "gimbal_yaw_rate_deg_s",
    "gimbal_pitch_rate_deg_s",
    "lost_count",
    "zoom_stable_count",
    "tracker_latency_ms",
    "frame_latency_ms",
]


@dataclass(frozen=True)
class FrameRecord:
    index: int
    name: str
    frame: Any
    timestamp_s: float


@dataclass(frozen=True)
class TargetFeatures:
    valid: bool
    bbox: tuple[float, float, float, float] | None
    center_error_x: float
    center_error_y: float
    area_ratio: float
    edge_margin: float


@dataclass(frozen=True)
class GimbalCommand:
    yaw_rate_deg_s: float
    pitch_rate_deg_s: float


@dataclass(frozen=True)
class ZoomDecision:
    level: float
    command: str
    reason: str
    lost_count: int
    stable_count: int
    changed: bool


class FrameSource:
    def __iter__(self) -> Iterator[FrameRecord]:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ImageDirectorySource(FrameSource):
    def __init__(self, root: Path, fps: float, max_frames: int | None) -> None:
        self.root = root
        self._fps = fps
        self.max_frames = max_frames
        self.files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        self.files.sort(key=lambda p: natural_key(p.name))
        if not self.files:
            raise ValueError(f"no image frames found in {root}")

    @property
    def fps(self) -> float:
        return self._fps

    def __iter__(self) -> Iterator[FrameRecord]:
        import cv2

        for index, path in enumerate(self.files, start=1):
            if self.max_frames is not None and index > self.max_frames:
                break
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"failed to read image: {path}")
            yield FrameRecord(index=index, name=path.name, frame=frame, timestamp_s=(index - 1) / self._fps)


class CaptureSource(FrameSource):
    def __init__(self, source: str, fallback_fps: float, max_frames: int | None) -> None:
        import cv2

        self.source = source
        self.max_frames = max_frames
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"failed to open video/RTSP source: {source}")
        cap_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = cap_fps if math.isfinite(cap_fps) and cap_fps > 1e-6 else fallback_fps

    @property
    def fps(self) -> float:
        return self._fps

    def __iter__(self) -> Iterator[FrameRecord]:
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            index += 1
            if self.max_frames is not None and index > self.max_frames:
                break
            yield FrameRecord(
                index=index,
                name=f"frame_{index:06d}",
                frame=frame,
                timestamp_s=(index - 1) / self._fps,
            )

    def close(self) -> None:
        self.cap.release()


class OpenCVTrackerWrapper:
    def __init__(self, tracker: Any) -> None:
        self.tracker = tracker

    def init(self, frame: Any, bbox: tuple[float, float, float, float]) -> None:
        roi = normalize_roi_bbox(bbox, "OpenCV tracker init bbox")
        ok = self.tracker.init(frame, roi)
        if ok is False:
            raise RuntimeError("OpenCV tracker initialization failed")

    def update(self, frame: Any) -> tuple[bool, tuple[float, float, float, float] | None]:
        ok, bbox = self.tracker.update(frame)
        if not ok:
            return False, None
        return True, tuple(float(v) for v in bbox)


class TemplateTracker:
    def __init__(self, min_score: float, search_scale: float, update_alpha: float) -> None:
        self.min_score = min_score
        self.search_scale = search_scale
        self.update_alpha = update_alpha
        self.template = None
        self.bbox: tuple[float, float, float, float] | None = None

    def init(self, frame: Any, bbox: tuple[float, float, float, float]) -> None:
        import cv2

        clipped = clip_bbox(bbox, frame.shape[1], frame.shape[0])
        if clipped is None:
            raise ValueError(f"invalid initial bbox: {bbox}")
        self.bbox = clipped
        x, y, w, h = [int(round(v)) for v in clipped]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = gray[y : y + h, x : x + w].copy()
        if self.template.size == 0:
            raise ValueError("empty template from initial bbox")

    def update(self, frame: Any) -> tuple[bool, tuple[float, float, float, float] | None]:
        import cv2

        if self.template is None or self.bbox is None:
            return False, None
        image_h, image_w = frame.shape[:2]
        x, y, w, h = self.bbox
        pad_w = max(w, self.search_scale * w)
        pad_h = max(h, self.search_scale * h)
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        sx1 = int(max(0, round(cx - 0.5 * pad_w)))
        sy1 = int(max(0, round(cy - 0.5 * pad_h)))
        sx2 = int(min(image_w, round(cx + 0.5 * pad_w)))
        sy2 = int(min(image_h, round(cy + 0.5 * pad_h)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search = gray[sy1:sy2, sx1:sx2]
        if search.shape[0] < self.template.shape[0] or search.shape[1] < self.template.shape[1]:
            return False, self.bbox
        result = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        new_bbox = (float(sx1 + max_loc[0]), float(sy1 + max_loc[1]), float(w), float(h))
        clipped = clip_bbox(new_bbox, image_w, image_h)
        if clipped is None or max_val < self.min_score:
            return False, self.bbox
        self.bbox = clipped
        tx, ty, tw, th = [int(round(v)) for v in clipped]
        new_template = gray[ty : ty + th, tx : tx + tw]
        if new_template.shape == self.template.shape and 0.0 < self.update_alpha <= 1.0:
            self.template = cv2.addWeighted(self.template, 1.0 - self.update_alpha, new_template, self.update_alpha, 0.0)
        return True, clipped


class CommandSink:
    def send_gimbal(self, command: GimbalCommand) -> None:
        raise NotImplementedError

    def send_zoom(self, zoom_level: float, command: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DryRunCommandSink(CommandSink):
    def send_gimbal(self, command: GimbalCommand) -> None:
        return None

    def send_zoom(self, zoom_level: float, command: str) -> None:
        return None


class PsdkCommandSink(CommandSink):
    """Placeholder for Manifold 3 PSDK integration.

    Fill send_gimbal and send_zoom with the DJI PSDK calls after validating
    axis signs, command units, zoom mapping, and safety interlocks on hardware.
    """

    def send_gimbal(self, command: GimbalCommand) -> None:
        return None

    def send_zoom(self, zoom_level: float, command: str) -> None:
        return None


class ZoomStateMachine:
    def __init__(self, cfg: dict[str, Any]) -> None:
        disabled = {float(v) for v in cfg.get("disabled_levels", [])}
        levels = [float(v) for v in cfg.get("levels", [2, 5, 10]) if float(v) not in disabled]
        levels = sorted(set(levels))
        if not levels:
            raise ValueError("zoom.levels is empty after removing disabled levels")
        self.levels = levels
        initial = float(cfg.get("initial_level", levels[0]))
        if initial not in levels:
            raise ValueError(f"initial zoom level {initial:g} is not in enabled levels {levels}")
        self.current_level = initial
        self.zoom_in_area_ratio = float(cfg.get("zoom_in_area_ratio", 0.012))
        self.zoom_out_area_ratio = float(cfg.get("zoom_out_area_ratio", 0.100))
        self.max_area_ratio = float(cfg.get("max_area_ratio", 0.160))
        self.zoom_in_min_edge_margin = float(cfg.get("zoom_in_min_edge_margin", 0.160))
        self.safe_edge_margin = float(cfg.get("safe_edge_margin", 0.100))
        self.danger_edge_margin = float(cfg.get("danger_edge_margin", 0.050))
        self.stable_frames_for_zoom_in = int(cfg.get("stable_frames_for_zoom_in", 12))
        self.lost_to_wide_frames = int(cfg.get("lost_to_wide_frames", 3))
        self.cooldown_frames = int(cfg.get("cooldown_frames", 18))
        self.cooldown_remaining = 0
        self.stable_count = 0
        self.lost_count = 0

    def update(self, features: TargetFeatures) -> ZoomDecision:
        if not features.valid:
            self.lost_count += 1
            self.stable_count = 0
            if self.lost_count >= self.lost_to_wide_frames and self.current_level > self.levels[0]:
                return self._step(-1, "lost_zoom_out")
            self._tick_cooldown()
            return self._keep("lost_keep")

        self.lost_count = 0
        if features.edge_margin <= self.danger_edge_margin or features.area_ratio >= self.max_area_ratio:
            self.stable_count = 0
            if self.current_level > self.levels[0]:
                return self._step(-1, "safety_zoom_out")
            self._tick_cooldown()
            return self._keep("safety_keep_wide")

        if features.edge_margin >= self.safe_edge_margin:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.cooldown_remaining > 0:
            self._tick_cooldown()
            return self._keep("cooldown_keep")

        if features.edge_margin < self.safe_edge_margin or features.area_ratio >= self.zoom_out_area_ratio:
            if self.current_level > self.levels[0]:
                return self._step(-1, "zoom_out_edge_or_large")
            return self._keep("wide_keep_edge_or_large")

        if (
            features.area_ratio <= self.zoom_in_area_ratio
            and features.edge_margin >= self.zoom_in_min_edge_margin
            and self.stable_count >= self.stable_frames_for_zoom_in
            and self.current_level < self.levels[-1]
        ):
            return self._step(1, "zoom_in_small_stable_target")

        return self._keep("track_keep")

    def _step(self, direction: int, reason: str) -> ZoomDecision:
        index = self.levels.index(self.current_level)
        next_index = max(0, min(len(self.levels) - 1, index + direction))
        target = self.levels[next_index]
        changed = target != self.current_level
        self.current_level = target
        self.cooldown_remaining = self.cooldown_frames if changed else self.cooldown_remaining
        command = "keep" if not changed else f"zoom_{'in' if direction > 0 else 'out'}_to_{format_zoom(target)}"
        return ZoomDecision(target, command, reason, self.lost_count, self.stable_count, changed)

    def _keep(self, reason: str) -> ZoomDecision:
        return ZoomDecision(self.current_level, "keep", reason, self.lost_count, self.stable_count, False)

    def _tick_cooldown(self) -> None:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UFZ-Deploy-v1 single H30 Z-stream closed-loop runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--input", default=None, help="Video file, image directory, or reserved RTSP URL")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-bbox", default=None, help="Manual bbox as x,y,w,h. If omitted, interactive ROI is used when enabled.")
    parser.add_argument("--tracker", default=None, help="auto, csrt, kcf, mil, or template")
    parser.add_argument("--command-sink", default=None, help="dry_run or psdk_placeholder")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-video", action="store_true", help="Disable visualization video writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import cv2

        cfg = load_config(Path(args.config))
        apply_cli_overrides(cfg, args)
        source_path = str(cfg["input"].get("source") or "").strip()
        if not source_path:
            raise ValueError("input.source is empty; pass --input or edit config_h30_deploy_v1.yaml")

        out_dir = build_output_dir(cfg, args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        source = open_frame_source(source_path, cfg)
        frame_iter = iter(source)
        first = next(frame_iter, None)
        if first is None:
            raise ValueError("input produced no frames")

        init_bbox = resolve_initial_bbox(first.frame, cfg)
        init_bbox = require_bbox(init_bbox, first.frame.shape[1], first.frame.shape[0], "initial bbox")
        tracker = create_tracker(cfg["tracker"])
        tracker.init(first.frame, init_bbox)
        zoom_sm = ZoomStateMachine(cfg["zoom"])
        gimbal_enabled = bool(cfg["gimbal"].get("enabled", True))
        sink = create_command_sink(str(cfg["gimbal"].get("command_sink", "dry_run")))

        csv_path = out_dir / str(cfg["output"].get("csv_name", "ufz_deploy_v1_log.csv"))
        video_writer = None
        if bool(cfg["output"].get("draw_video", True)) and not args.no_video:
            video_path = out_dir / str(cfg["output"].get("video_name", "ufz_deploy_v1_vis.mp4"))
            video_writer = create_video_writer(video_path, first.frame, source.fps)

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "ufz_deploy_v1_single_h30_z_stream",
            "source": source_path,
            "fps": source.fps,
            "init_bbox_xywh": list(init_bbox),
            "config": cfg,
            "notes": "No W/Z offline replay and no Oracle decisions. PSDK sink is a placeholder.",
        }
        write_json(out_dir / str(cfg["output"].get("metadata_name", "ufz_deploy_v1_metadata.json")), metadata)

        last_bbox = init_bbox
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            process_frame(
                writer=writer,
                record=first,
                frame=first.frame,
                valid=True,
                bbox=init_bbox,
                tracker_latency_ms=0.0,
                last_bbox=last_bbox,
                zoom_sm=zoom_sm,
                gimbal_cfg=cfg["gimbal"],
                gimbal_enabled=gimbal_enabled,
                sink=sink,
                video_writer=video_writer,
            )
            for record in frame_iter:
                frame_t0 = time.perf_counter()
                tracker_t0 = time.perf_counter()
                valid, bbox = tracker.update(record.frame)
                tracker_latency_ms = (time.perf_counter() - tracker_t0) * 1000.0
                clipped = clip_bbox(bbox, record.frame.shape[1], record.frame.shape[0]) if bbox is not None else None
                valid = bool(valid and clipped is not None)
                if valid and clipped is not None:
                    last_bbox = clipped
                process_frame(
                    writer=writer,
                    record=record,
                    frame=record.frame,
                    valid=valid,
                    bbox=clipped if valid else last_bbox,
                    tracker_latency_ms=tracker_latency_ms,
                    last_bbox=last_bbox,
                    zoom_sm=zoom_sm,
                    gimbal_cfg=cfg["gimbal"],
                    gimbal_enabled=gimbal_enabled,
                    sink=sink,
                    video_writer=video_writer,
                    frame_start_time=frame_t0,
                )
        if video_writer is not None:
            video_writer.release()
        source.close()
        sink.close()
        print(json.dumps({"output_dir": str(out_dir), "csv": str(csv_path)}, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def process_frame(
    *,
    writer: csv.DictWriter,
    record: FrameRecord,
    frame: Any,
    valid: bool,
    bbox: tuple[float, float, float, float] | None,
    tracker_latency_ms: float,
    last_bbox: tuple[float, float, float, float] | None,
    zoom_sm: ZoomStateMachine,
    gimbal_cfg: dict[str, Any],
    gimbal_enabled: bool,
    sink: CommandSink,
    video_writer: Any | None,
    frame_start_time: float | None = None,
) -> None:
    features = compute_features(valid, bbox, frame.shape[1], frame.shape[0])
    gimbal = compute_gimbal_command(features, gimbal_cfg) if gimbal_enabled else GimbalCommand(0.0, 0.0)
    zoom = zoom_sm.update(features)
    sink.send_gimbal(gimbal)
    if zoom.changed:
        sink.send_zoom(zoom.level, zoom.command)
    frame_latency_ms = (time.perf_counter() - frame_start_time) * 1000.0 if frame_start_time is not None else 0.0
    writer.writerow(build_log_row(record, features, zoom, gimbal, tracker_latency_ms, frame_latency_ms))
    if video_writer is not None:
        video_writer.write(draw_overlay(frame, features, zoom, gimbal, last_bbox))


def build_log_row(
    record: FrameRecord,
    features: TargetFeatures,
    zoom: ZoomDecision,
    gimbal: GimbalCommand,
    tracker_latency_ms: float,
    frame_latency_ms: float,
) -> dict[str, Any]:
    bbox = features.bbox
    return {
        "frame_index": record.index,
        "timestamp_s": fmt(record.timestamp_s),
        "source_name": record.name,
        "tracker_valid": "true" if features.valid else "false",
        "bbox_x": fmt(bbox[0]) if bbox else "",
        "bbox_y": fmt(bbox[1]) if bbox else "",
        "bbox_w": fmt(bbox[2]) if bbox else "",
        "bbox_h": fmt(bbox[3]) if bbox else "",
        "center_error_x": fmt(features.center_error_x),
        "center_error_y": fmt(features.center_error_y),
        "area_ratio": fmt(features.area_ratio),
        "edge_margin": fmt(features.edge_margin),
        "zoom_level": format_zoom(zoom.level),
        "zoom_command": zoom.command,
        "zoom_reason": zoom.reason,
        "gimbal_yaw_rate_deg_s": fmt(gimbal.yaw_rate_deg_s),
        "gimbal_pitch_rate_deg_s": fmt(gimbal.pitch_rate_deg_s),
        "lost_count": zoom.lost_count,
        "zoom_stable_count": zoom.stable_count,
        "tracker_latency_ms": fmt(tracker_latency_ms),
        "frame_latency_ms": fmt(frame_latency_ms),
    }


def compute_features(
    valid: bool,
    bbox: tuple[float, float, float, float] | None,
    image_w: int,
    image_h: int,
) -> TargetFeatures:
    if not valid or bbox is None:
        return TargetFeatures(False, bbox, 0.0, 0.0, 0.0, 0.0)
    x, y, w, h = bbox
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    area_ratio = max(0.0, (w * h) / float(image_w * image_h))
    edge_margin = min(x, y, image_w - (x + w), image_h - (y + h)) / float(min(image_w, image_h))
    return TargetFeatures(
        valid=True,
        bbox=bbox,
        center_error_x=(cx - 0.5 * image_w) / float(image_w),
        center_error_y=(cy - 0.5 * image_h) / float(image_h),
        area_ratio=area_ratio,
        edge_margin=edge_margin,
    )


def compute_gimbal_command(features: TargetFeatures, cfg: dict[str, Any]) -> GimbalCommand:
    if not features.valid:
        return GimbalCommand(0.0, 0.0)
    deadband = float(cfg.get("deadband", 0.015))
    err_x = 0.0 if abs(features.center_error_x) < deadband else features.center_error_x
    err_y = 0.0 if abs(features.center_error_y) < deadband else features.center_error_y
    yaw = clamp(err_x * float(cfg.get("yaw_kp", 35.0)), -float(cfg.get("max_yaw_rate_deg_s", 18.0)), float(cfg.get("max_yaw_rate_deg_s", 18.0)))
    pitch = clamp(err_y * float(cfg.get("pitch_kp", -28.0)), -float(cfg.get("max_pitch_rate_deg_s", 14.0)), float(cfg.get("max_pitch_rate_deg_s", 14.0)))
    return GimbalCommand(yaw, pitch)


def draw_overlay(frame: Any, features: TargetFeatures, zoom: ZoomDecision, gimbal: GimbalCommand, last_bbox: tuple[float, float, float, float] | None) -> Any:
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    center = (w // 2, h // 2)
    cv2.line(out, (center[0] - 24, center[1]), (center[0] + 24, center[1]), (255, 255, 255), 1)
    cv2.line(out, (center[0], center[1] - 24), (center[0], center[1] + 24), (255, 255, 255), 1)
    bbox = features.bbox or last_bbox
    if bbox is not None:
        x, y, bw, bh = [int(round(v)) for v in bbox]
        color = (0, 220, 0) if features.valid else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        if features.valid:
            target = (int(round(x + 0.5 * bw)), int(round(y + 0.5 * bh)))
            cv2.arrowedLine(out, center, target, (0, 255, 255), 2, tipLength=0.08)
    lines = [
        f"UFZ-Deploy-v1 H30 Z | valid={features.valid} | zoom={format_zoom(zoom.level)} | {zoom.command}",
        f"err=({features.center_error_x:+.3f},{features.center_error_y:+.3f}) area={features.area_ratio:.4f} edge={features.edge_margin:.3f}",
        f"gimbal yaw={gimbal.yaw_rate_deg_s:+.2f} deg/s pitch={gimbal.pitch_rate_deg_s:+.2f} deg/s reason={zoom.reason}",
    ]
    for i, text in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def create_video_writer(path: Path, first_frame: Any, fps: float) -> Any:
    import cv2

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create visualization video: {path}")
    return writer


def create_tracker(cfg: dict[str, Any]) -> Any:
    tracker_type = str(cfg.get("type", "auto")).lower()
    if tracker_type == "template":
        return TemplateTracker(
            min_score=float(cfg.get("template_min_score", 0.45)),
            search_scale=float(cfg.get("template_search_scale", 2.5)),
            update_alpha=float(cfg.get("template_update_alpha", 0.05)),
        )
    import cv2

    requested = [tracker_type] if tracker_type != "auto" else ["csrt", "kcf", "mil"]
    for name in requested:
        tracker = make_opencv_tracker(cv2, name)
        if tracker is not None:
            return OpenCVTrackerWrapper(tracker)
    if tracker_type != "auto":
        raise ValueError(f"OpenCV tracker '{tracker_type}' is unavailable; try --tracker template")
    return TemplateTracker(
        min_score=float(cfg.get("template_min_score", 0.45)),
        search_scale=float(cfg.get("template_search_scale", 2.5)),
        update_alpha=float(cfg.get("template_update_alpha", 0.05)),
    )


def make_opencv_tracker(cv2: Any, name: str) -> Any | None:
    suffix = name.upper()
    attr = f"Tracker{suffix}_create"
    if hasattr(cv2, attr):
        return getattr(cv2, attr)()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, attr):
        return getattr(cv2.legacy, attr)()
    return None


def create_command_sink(name: str) -> CommandSink:
    normalized = name.lower().strip()
    if normalized in {"dry_run", "dry-run", "none"}:
        return DryRunCommandSink()
    if normalized in {"psdk", "psdk_placeholder", "psdk-placeholder"}:
        print("WARNING: psdk_placeholder selected; commands are no-ops until DJI PSDK calls are wired.")
        return PsdkCommandSink()
    raise ValueError(f"unknown command sink: {name}")


def open_frame_source(source: str, cfg: dict[str, Any]) -> FrameSource:
    path = Path(source).expanduser()
    fps = float(cfg["input"].get("fps", 30.0))
    max_frames = cfg["input"].get("max_frames")
    max_frames = int(max_frames) if max_frames is not None else None
    source_type = str(cfg["input"].get("type", "auto")).lower()
    if source_type == "image_dir" or (source_type == "auto" and path.exists() and path.is_dir()):
        return ImageDirectorySource(path, fps=fps, max_frames=max_frames)
    if source.startswith("rtsp://"):
        print("WARNING: RTSP input is reserved for deployment smoke tests; latency and reconnect handling are not implemented yet.")
    return CaptureSource(source, fallback_fps=fps, max_frames=max_frames)


def resolve_initial_bbox(frame: Any, cfg: dict[str, Any]) -> tuple[int, int, int, int] | None:
    import cv2

    raw = cfg["input"].get("init_bbox")
    if raw is not None and raw != "":
        return parse_bbox(raw)
    if not bool(cfg["input"].get("interactive_init", True)):
        raise ValueError("input.init_bbox is empty and interactive_init is false")
    roi = cv2.selectROI("UFZ-Deploy-v1 init bbox", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("UFZ-Deploy-v1 init bbox")
    return normalize_roi_bbox(roi, "selected ROI bbox")


def parse_bbox(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    elif isinstance(value, Iterable):
        parts = list(value)
    else:
        raise ValueError(f"cannot parse bbox from {value!r}")
    return normalize_roi_bbox(parts, "bbox")


def normalize_roi_bbox(value: Iterable[Any], label: str = "bbox") -> tuple[int, int, int, int]:
    parts = list(value)
    if len(parts) != 4:
        raise ValueError(f"{label} must have four values x,y,w,h: {value!r}")
    try:
        floats = [float(v) for v in parts]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-numeric values: {value!r}") from exc
    if not all(math.isfinite(v) for v in floats):
        raise ValueError(f"{label} contains non-finite values: {value!r}")
    x, y, w, h = tuple(int(round(float(v))) for v in floats)
    if w <= 0 or h <= 0:
        raise ValueError(f"{label} must have positive width and height after ROI selection: {(x, y, w, h)!r}")
    return x, y, w, h


def require_bbox(bbox: tuple[float, float, float, float] | None, image_w: int, image_h: int, label: str) -> tuple[float, float, float, float]:
    clipped = clip_bbox(bbox, image_w, image_h) if bbox is not None else None
    if clipped is None:
        raise ValueError(f"invalid {label}: {bbox}")
    return clipped


def clip_bbox(bbox: tuple[float, float, float, float] | None, image_w: int, image_h: int) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    x, y, w, h = bbox
    if not all(math.isfinite(v) for v in bbox) or w <= 0 or h <= 0:
        return None
    x1 = clamp(x, 0.0, float(image_w - 1))
    y1 = clamp(y, 0.0, float(image_h - 1))
    x2 = clamp(x + w, x1 + 1.0, float(image_w))
    y2 = clamp(y + h, y1 + 1.0, float(image_h))
    return x1, y1, x2 - x1, y2 - y1


def build_output_dir(cfg: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    root = Path(str(cfg["output"].get("root", "outputs/deploy_v1"))).expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"ufz_deploy_v1_{stamp}"


def load_config(path: Path) -> dict[str, Any]:
    cfg = deep_copy(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        deep_update(cfg, loaded)
    return cfg


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.input is not None:
        cfg["input"]["source"] = args.input
    if args.init_bbox is not None:
        cfg["input"]["init_bbox"] = args.init_bbox
        cfg["input"]["interactive_init"] = False
    if args.tracker is not None:
        cfg["tracker"]["type"] = args.tracker
    if args.command_sink is not None:
        cfg["gimbal"]["command_sink"] = args.command_sink
    if args.max_frames is not None:
        cfg["input"]["max_frames"] = args.max_frames
    if args.no_video:
        cfg["output"]["draw_video"] = False


def deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def natural_key(value: str) -> list[int | str]:
    parts: list[int | str] = []
    token = ""
    for char in value:
        if char.isdigit():
            if token and not token[-1].isdigit():
                parts.append(token.lower())
                token = ""
            token += char
        else:
            if token and token[-1].isdigit():
                parts.append(int(token))
                token = ""
            token += char
    if token:
        parts.append(int(token) if token.isdigit() else token.lower())
    return parts


def format_zoom(value: float) -> str:
    return f"{int(value)}x" if abs(value - round(value)) < 1e-9 else f"{value:g}x"


def fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.9f}".rstrip("0").rstrip(".") or "0"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


if __name__ == "__main__":
    raise SystemExit(main())

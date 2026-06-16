#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import yaml
from tqdm import tqdm


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UFZ-Track Experiment 1 on UAV123@10fps.")
    parser.add_argument("--config", default="configs/uav123_10fps.yaml")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "fixed_wide",
            "fixed_tele",
            "scale_only",
            "confidence_only",
            "ufz",
            "ufz_v2",
            "ufz_v2_1",
            "ufz_v2_2",
            "ufz_v2_no_recovery",
            "ufz_v2_no_edge_suppression",
            "ufz_v2_no_lowconf_assoc",
        ],
    )
    parser.add_argument("--sequences", nargs="+", default=None, help="Sequence names or comma lists.")
    parser.add_argument("--model", default=None, help="YOLOv8 weights, e.g. yolov8n.pt.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device string.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug cap for frames per sequence.")
    parser.add_argument("--skip-missing", action="store_true", help="Warn and skip sequences with missing images or bbox GT.")
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


def _read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _write_bboxes(path: Path, boxes) -> None:
    from ufztrack.bbox import BBox, bbox_is_valid

    path.parent.mkdir(parents=True, exist_ok=True)
    last_valid = BBox(0.0, 0.0, 1.0, 1.0)
    with path.open("w", encoding="utf-8") as f:
        for frame_idx, box in enumerate(boxes, start=1):
            if bbox_is_valid(box):
                last_valid = box
            else:
                warnings.warn(
                    f"Invalid bbox reached writer at frame {frame_idx}; using last valid bbox.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                box = last_valid
            f.write(f"{box.x:.3f},{box.y:.3f},{box.w:.3f},{box.h:.3f}\n")


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["frame", "zoom_level", "uncertainty", "area", "conf", "lost", "latency", "command"]
    v2_fields = [
        "state",
        "zoom_in_score",
        "zoom_out_score",
        "scale_need",
        "edge_risk",
        "blur_risk",
        "association_risk",
        "lost_risk",
        "overzoom_risk",
        "assoc_stage",
        "candidate_count",
        "high_candidate_count",
        "low_candidate_count",
        "selected_cost",
        "association_ambiguity",
        "kalman_innovation",
        "cooldown_remaining",
        "stable_count",
        "unstable_count",
        "risk_level",
        "zoom_veto_reason",
        "proposed_action",
        "final_action",
        "veto_applied",
        "veto_reason",
        "decision_reason",
    ]
    present = {key for row in rows for key in row}
    fieldnames = base_fields + [field for field in v2_fields if field in present]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _policy_config(cfg: dict[str, Any]):
    from ufztrack.zoom_policy import ZoomPolicyConfig

    policy_cfg = cfg.get("policy", {})
    ufz_cfg = policy_cfg.get("ufz", {})
    return ZoomPolicyConfig(
        levels=cfg["zoom"].get("levels", [1, 2, 4, 8]),
        fixed_tele_level=policy_cfg.get("fixed_tele_level", 4),
        area_level_thresholds=policy_cfg.get("area_level_thresholds", {}),
        confidence_thresholds=policy_cfg.get("confidence_thresholds", {}),
        cooldown_frames=ufz_cfg.get("cooldown_frames", 5),
        hysteresis_ratio=ufz_cfg.get("hysteresis_ratio", 0.15),
        high_uncertainty=ufz_cfg.get("high_uncertainty", 0.65),
        medium_uncertainty=ufz_cfg.get("medium_uncertainty", 0.45),
        high_edge_risk=ufz_cfg.get("high_edge_risk", 0.70),
        lost_wide_after=ufz_cfg.get("lost_wide_after", 2),
        v2_threshold_in=policy_cfg.get("ufz_v2", {}).get("threshold_in", 0.20),
        v2_threshold_out=policy_cfg.get("ufz_v2", {}).get("threshold_out", 0.05),
        v2_area_min=policy_cfg.get("ufz_v2", {}).get("area_min", 0.010),
        v2_area_high=policy_cfg.get("ufz_v2", {}).get("area_high", 0.090),
        v2_high_uncertainty=policy_cfg.get("ufz_v2", {}).get("high_uncertainty", 0.60),
        v2_low_uncertainty=policy_cfg.get("ufz_v2", {}).get("low_uncertainty", 0.35),
        v2_high_blur_risk=policy_cfg.get("ufz_v2", {}).get("high_blur_risk", 0.70),
        v2_high_association_risk=policy_cfg.get("ufz_v2", {}).get("high_association_risk", 0.65),
        v2_lost_to_caution=cfg.get("recovery", {}).get("lost_to_caution", 1),
        v2_lost_to_recovery=cfg.get("recovery", {}).get("lost_to_recovery", 2),
        v2_force_wide_lost=cfg.get("recovery", {}).get("force_wide_lost", 4),
        v2_stable_frames_to_tracking=cfg.get("recovery", {}).get("stable_frames_to_tracking", 3),
        v2_min_stable_zoom_in_frames=policy_cfg.get("ufz_v2", {}).get("min_stable_zoom_in_frames", 2),
        v2_disable_recovery=policy_cfg.get("ufz_v2", {}).get("disable_recovery", False),
        v2_disable_edge_suppression=policy_cfg.get("ufz_v2", {}).get("disable_edge_suppression", False),
        v2_disable_lowconf_assoc=policy_cfg.get("ufz_v2", {}).get("disable_lowconf_assoc", False),
        v21_stable_frames_to_zoom_in=policy_cfg.get("ufz_v2_1", {}).get("stable_frames_to_zoom_in", 2),
        v21_stable_conf_thresh=policy_cfg.get("ufz_v2_1", {}).get("stable_conf_thresh", 0.25),
        v21_ambiguity_safe_thresh=policy_cfg.get("ufz_v2_1", {}).get("ambiguity_safe_thresh", 0.45),
        v21_innovation_safe_thresh=policy_cfg.get("ufz_v2_1", {}).get("innovation_safe_thresh", 0.50),
        v21_edge_safe_thresh=policy_cfg.get("ufz_v2_1", {}).get("edge_safe_thresh", 0.65),
        v21_blur_safe_thresh=policy_cfg.get("ufz_v2_1", {}).get("blur_safe_thresh", 0.65),
        v21_hold_min_frames_after_zoom=policy_cfg.get("ufz_v2_1", {}).get("hold_min_frames_after_zoom", 8),
        v21_cooldown_frames=policy_cfg.get("ufz_v2_1", {}).get("cooldown_frames", 4),
        v21_recovery_cooldown_frames=policy_cfg.get("ufz_v2_1", {}).get("recovery_cooldown_frames", 3),
        v21_hard_edge_thresh=policy_cfg.get("ufz_v2_1", {}).get("hard_edge_thresh", 0.85),
        v21_hard_blur_thresh=policy_cfg.get("ufz_v2_1", {}).get("hard_blur_thresh", 0.85),
        v21_hard_assoc_thresh=policy_cfg.get("ufz_v2_1", {}).get("hard_assoc_thresh", 0.80),
        v21_area_low=policy_cfg.get("ufz_v2_1", {}).get("area_low", 0.0025),
        v21_area_target=policy_cfg.get("ufz_v2_1", {}).get("area_target", 0.006),
        v21_area_high=policy_cfg.get("ufz_v2_1", {}).get("area_high", 0.035),
        v22_hard_edge_thresh=policy_cfg.get("ufz_v2_2", {}).get("hard_edge_thresh", 0.88),
        v22_hard_assoc_thresh=policy_cfg.get("ufz_v2_2", {}).get("hard_assoc_thresh", 0.90),
        v22_hard_blur_thresh=policy_cfg.get("ufz_v2_2", {}).get("hard_blur_thresh", 0.90),
        v22_hard_overzoom_area=policy_cfg.get("ufz_v2_2", {}).get("hard_overzoom_area", 0.040),
        v22_lost_recovery_thresh=policy_cfg.get("ufz_v2_2", {}).get("lost_recovery_thresh", 2),
        v22_force_wide_lost=policy_cfg.get("ufz_v2_2", {}).get("force_wide_lost", 4),
        v22_cooldown_frames=policy_cfg.get("ufz_v2_2", {}).get("cooldown_frames", 4),
        v22_hold_min_frames_after_zoom=policy_cfg.get("ufz_v2_2", {}).get("hold_min_frames_after_zoom", 4),
    )


def run_sequence(
    cfg: dict[str, Any],
    method: str,
    sequence_name: str,
    model_override: str | None = None,
    device_override: str | None = None,
    max_frames_override: int | None = None,
) -> dict[str, Any]:
    from ufztrack.dataset_uav123 import load_uav123_sequence
    from ufztrack.detector_yolo import Detection, YOLOv8Detector
    from ufztrack.bbox import bbox_is_valid, sanitize_bbox
    from ufztrack.kalman_tracker import KalmanBoxTracker, KalmanConfig
    from ufztrack.measurement_selector import (
        ByteTrackLikeSelectorConfig,
        ByteTrackLikeSingleTargetSelector,
        MeasurementSelector,
        MeasurementSelectorConfig,
    )
    from ufztrack.metrics import evaluate_sequence
    from ufztrack.uncertainty import (
        UncertaintyConfig,
        bbox_jitter,
        compute_uncertainty,
        edge_risk,
        effective_confidence,
        estimate_blur_risk,
    )
    from ufztrack.zoom_policy import ZoomPolicy
    from ufztrack.zoom_simulator import ZoomSimulator, zoomed_to_original_bbox

    dataset_cfg = cfg["dataset"]
    detector_cfg = cfg["detector"]
    tracker_cfg = cfg["tracker"]
    selection_cfg = cfg["measurement_selection"]
    uncertainty_cfg = cfg["uncertainty"]
    zoom_cfg = cfg["zoom"]
    association_cfg = cfg.get("association", {})
    is_v2 = method.startswith("ufz_v2")

    sequence = load_uav123_sequence(
        sequence_name,
        image_root=dataset_cfg["image_root"],
        bbox_anno_root=dataset_cfg["bbox_anno_root"],
        frame_glob=dataset_cfg.get("frame_glob", "*.jpg"),
        frame_stride=dataset_cfg.get("frame_stride", 1),
        max_frames=max_frames_override if max_frames_override is not None else dataset_cfg.get("max_frames"),
    )

    detector = YOLOv8Detector(
        model=model_override or detector_cfg.get("model", "yolov8n.pt"),
        conf=detector_cfg.get("conf", 0.10),
        iou=detector_cfg.get("iou", 0.70),
        imgsz=detector_cfg.get("imgsz", 640),
        classes=detector_cfg.get("classes"),
        device=device_override if device_override is not None else detector_cfg.get("device"),
    )
    selector = MeasurementSelector(MeasurementSelectorConfig(**selection_cfg))
    low_conf_thresh = association_cfg.get("low_conf_thresh", 0.08)
    high_conf_thresh = association_cfg.get("high_conf_thresh", 0.35)
    if method == "ufz_v2_no_lowconf_assoc" or cfg.get("policy", {}).get("ufz_v2", {}).get("disable_lowconf_assoc", False):
        low_conf_thresh = high_conf_thresh
    v2_selector = ByteTrackLikeSingleTargetSelector(
        ByteTrackLikeSelectorConfig(
            high_conf_thresh=high_conf_thresh,
            low_conf_thresh=low_conf_thresh,
            iou_gate=association_cfg.get("iou_gate", 0.05),
            center_dist_factor=association_cfg.get("center_dist_factor", 2.5),
            area_ratio_min=association_cfg.get("area_ratio_min", 0.2),
            area_ratio_max=association_cfg.get("area_ratio_max", 5.0),
        )
    )
    simulator = ZoomSimulator(levels=zoom_cfg.get("levels", [1, 2, 4, 8]), crop_mode=zoom_cfg.get("crop_mode", "predicted_center"))
    uncertainty_config = UncertaintyConfig(**uncertainty_cfg)
    policy = ZoomPolicy(method, _policy_config(cfg))

    first_image = _read_image(sequence.image_paths[0])
    height, width = first_image.shape[:2]
    min_box_size = float(tracker_cfg.get("min_box_size", 2.0))
    initial_box, initial_valid = sanitize_bbox(sequence.gt_boxes[0], width, height, min_size=min_box_size)
    if not initial_valid:
        warnings.warn(
            f"{sequence.name}: first-frame GT bbox is invalid; using sanitized initialization box.",
            RuntimeWarning,
            stacklevel=2,
        )
    tracker = KalmanBoxTracker(initial_box, KalmanConfig(**tracker_cfg))

    predictions = [initial_box]
    log_rows: list[dict[str, Any]] = [
        {
            "frame": 1,
            "zoom_level": int(zoom_cfg.get("initial_level", 1)),
            "uncertainty": f"{0.0:.6f}",
            "area": f"{initial_box.area / float(width * height):.8f}",
            "conf": f"{1.0:.6f}",
            "lost": 0,
            "latency": f"{0.0:.6f}",
            "command": "init",
        }
    ]
    if is_v2:
        log_rows[0].update(_v2_log_fields())

    current_zoom = policy.initial_runtime_level(int(zoom_cfg.get("initial_level", 1)))
    previous_uncertainty = 0.0
    for frame_index, image_path in enumerate(tqdm(sequence.image_paths[1:], desc=f"{method}:{sequence.name}"), start=2):
        start_time = time.perf_counter()
        image = _read_image(image_path)
        height, width = image.shape[:2]

        predicted_raw = tracker.predict()
        predicted_box, predicted_valid = sanitize_bbox(
            predicted_raw,
            width,
            height,
            fallback=predictions[-1],
            min_size=min_box_size,
        )
        recovered_from_invalid = not predicted_valid
        zoomed, crop = simulator.simulate(
            image,
            zoom_level=current_zoom,
            predicted_bbox=predicted_box,
            uncertainty=previous_uncertainty if is_v2 else 0.0,
            context_margin_factor=zoom_cfg.get("context_margin_factor", 0.0) if is_v2 else 0.0,
            uncertainty_margin_gain=zoom_cfg.get("uncertainty_margin_gain", 0.0) if is_v2 else 0.0,
            min_context_pixels=zoom_cfg.get("min_context_pixels", 0) if is_v2 else 0,
        )
        detections_zoomed = detector.detect(zoomed)
        detections_original = []
        for det in detections_zoomed:
            mapped_box = zoomed_to_original_bbox(det.bbox, crop)
            if not bbox_is_valid(mapped_box):
                warnings.warn(
                    f"{sequence.name} frame {frame_index}: skipping invalid mapped detection bbox.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            detection_box, _ = sanitize_bbox(mapped_box, width, height, min_size=min_box_size)
            detections_original.append(
                Detection(
                    bbox=detection_box,
                    confidence=det.confidence,
                    class_id=det.class_id,
                )
            )

        if is_v2:
            selected = v2_selector.select(
                predicted_box,
                detections_original,
                current_zoom_level=current_zoom,
                lost_count=int(tracker.lost_count),
            )
        else:
            selected = selector.select(predicted_box, detections_original)
        assoc_stage = selected.assoc_stage if selected is not None else "predict_only"
        candidate_count = len(detections_original)
        high_candidate_count = (
            selected.high_candidate_count
            if selected is not None
            else sum(det.confidence >= high_conf_thresh for det in detections_original)
        )
        low_candidate_count = (
            selected.low_candidate_count
            if selected is not None
            else sum(low_conf_thresh <= det.confidence < high_conf_thresh for det in detections_original)
        )
        selected_cost = selected.cost if selected is not None else -1.0
        association_ambiguity = selected.association_ambiguity if selected is not None else 1.0
        selected_conf = None
        if selected is not None:
            selected_conf = selected.detection.confidence
            output_raw = tracker.update(selected.detection.bbox)
        else:
            output_raw = tracker.current_bbox()
        output_box, output_valid = sanitize_bbox(
            output_raw,
            width,
            height,
            fallback=predictions[-1],
            min_size=min_box_size,
        )
        recovered_from_invalid = recovered_from_invalid or not output_valid
        lost_count = max(int(tracker.lost_count), 1 if recovered_from_invalid else 0)

        confidence = effective_confidence(
            selected_conf,
            lost_count,
            lost_conf_decay=uncertainty_cfg.get("lost_conf_decay", 0.85),
        )
        jitter = bbox_jitter(predictions[-1], output_box)
        edge = edge_risk(output_box, width, height, margin_ratio=uncertainty_cfg.get("edge_margin_ratio", 0.08))
        innovation = tracker.last_innovation if selected is not None else float(lost_count)
        blur = estimate_blur_risk(
            zoomed,
            variance_low=uncertainty_cfg.get("blur_variance_low", 40.0),
            variance_high=uncertainty_cfg.get("blur_variance_high", 200.0),
        ) if is_v2 else 0.0
        association_risk = _association_risk(
            assoc_stage=assoc_stage,
            confidence=confidence,
            detection_iou=selected.iou if selected is not None else 0.0,
            ambiguity=association_ambiguity,
        ) if is_v2 else 0.0
        components = compute_uncertainty(
            confidence,
            innovation,
            jitter,
            edge,
            association=association_risk,
            blur=blur,
            config=uncertainty_config,
        )
        area_ratio = output_box.area / float(width * height)

        decision = policy.decide(
            current_level=current_zoom,
            area_ratio=area_ratio,
            confidence=confidence,
            uncertainty=components.total,
            edge_risk=components.edge_risk,
            lost_count=lost_count,
            frame_idx=frame_index,
            blur_risk=blur,
            association_risk=association_risk,
            assoc_stage=assoc_stage,
            kalman_innovation=innovation,
            association_ambiguity=association_ambiguity,
        )
        latency = time.perf_counter() - start_time
        predictions.append(output_box)
        log_row = {
            "frame": frame_index,
            "zoom_level": int(current_zoom),
            "uncertainty": f"{components.total:.6f}",
            "area": f"{area_ratio:.8f}",
            "conf": f"{confidence:.6f}",
            "lost": int(lost_count),
            "latency": f"{latency:.6f}",
            "command": decision.command,
        }
        if is_v2:
            log_row.update(
                _v2_log_fields(
                    state=decision.state,
                    zoom_in_score=decision.zoom_in_score,
                    zoom_out_score=decision.zoom_out_score,
                    scale_need=decision.scale_need,
                    edge_risk=decision.edge_risk,
                    blur_risk=decision.blur_risk,
                    association_risk=decision.association_risk,
                    lost_risk=decision.lost_risk,
                    overzoom_risk=decision.overzoom_risk,
                    assoc_stage=assoc_stage,
                    candidate_count=candidate_count,
                    high_candidate_count=high_candidate_count,
                    low_candidate_count=low_candidate_count,
                    selected_cost=selected_cost,
                    association_ambiguity=association_ambiguity,
                    kalman_innovation=innovation,
                    cooldown_remaining=decision.cooldown_remaining,
                    stable_count=decision.stable_count,
                    unstable_count=decision.unstable_count,
                    risk_level=decision.risk_level,
                    zoom_veto_reason=decision.zoom_veto_reason,
                    proposed_action=decision.proposed_action,
                    final_action=decision.final_action,
                    veto_applied=decision.veto_applied,
                    veto_reason=decision.veto_reason,
                    decision_reason=decision.decision_reason,
                )
            )
        log_rows.append(log_row)
        previous_uncertainty = components.total
        current_zoom = decision.level

    output_root = Path(cfg["outputs"]["root"])
    result_path = output_root / "results" / method / f"{sequence.name}.txt"
    log_path = output_root / "logs" / method / f"{sequence.name}.csv"
    _write_bboxes(result_path, predictions)
    _write_log(log_path, log_rows)

    metrics = evaluate_sequence(predictions, sequence.gt_boxes)
    return {
        "method": method,
        "sequence": sequence.name,
        "frames": len(predictions),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "metrics": asdict(metrics),
    }


def _association_risk(
    assoc_stage: str,
    confidence: float,
    detection_iou: float,
    ambiguity: float,
) -> float:
    if assoc_stage == "predict_only":
        return 1.0
    stage_penalty = 0.20 if assoc_stage == "low" else 0.0
    value = 0.45 * ambiguity + 0.30 * (1.0 - confidence) + 0.25 * (1.0 - detection_iou) + stage_penalty
    return min(max(float(value), 0.0), 1.0)


def _v2_log_fields(
    state: str = "TRACKING",
    zoom_in_score: float = 0.0,
    zoom_out_score: float = 0.0,
    scale_need: float = 0.0,
    edge_risk: float = 0.0,
    blur_risk: float = 0.0,
    association_risk: float = 0.0,
    lost_risk: float = 0.0,
    overzoom_risk: float = 0.0,
    assoc_stage: str = "init",
    candidate_count: int = 0,
    high_candidate_count: int = 0,
    low_candidate_count: int = 0,
    selected_cost: float = -1.0,
    association_ambiguity: float = 0.0,
    kalman_innovation: float = 0.0,
    cooldown_remaining: int = 0,
    stable_count: int = 0,
    unstable_count: int = 0,
    risk_level: str = "low",
    zoom_veto_reason: str = "none",
    proposed_action: str = "",
    final_action: str = "",
    veto_applied: bool = False,
    veto_reason: str = "none",
    decision_reason: str = "init",
) -> dict[str, Any]:
    return {
        "state": state,
        "zoom_in_score": f"{zoom_in_score:.6f}",
        "zoom_out_score": f"{zoom_out_score:.6f}",
        "scale_need": f"{scale_need:.6f}",
        "edge_risk": f"{edge_risk:.6f}",
        "blur_risk": f"{blur_risk:.6f}",
        "association_risk": f"{association_risk:.6f}",
        "lost_risk": f"{lost_risk:.6f}",
        "overzoom_risk": f"{overzoom_risk:.6f}",
        "assoc_stage": assoc_stage,
        "candidate_count": int(candidate_count),
        "high_candidate_count": int(high_candidate_count),
        "low_candidate_count": int(low_candidate_count),
        "selected_cost": f"{selected_cost:.6f}",
        "association_ambiguity": f"{association_ambiguity:.6f}",
        "kalman_innovation": f"{kalman_innovation:.6f}",
        "cooldown_remaining": int(cooldown_remaining),
        "stable_count": int(stable_count),
        "unstable_count": int(unstable_count),
        "risk_level": risk_level,
        "zoom_veto_reason": zoom_veto_reason,
        "proposed_action": proposed_action,
        "final_action": final_action,
        "veto_applied": int(bool(veto_applied)),
        "veto_reason": veto_reason,
        "decision_reason": decision_reason,
    }


def main() -> int:
    _add_src_to_path()
    from ufztrack.dataset_uav123 import list_sequences

    args = parse_args()
    cfg = _load_config(args.config)
    requested = _split_sequences(args.sequences) or cfg["dataset"].get("sequences")
    sequence_names = list_sequences(cfg["dataset"]["image_root"], cfg["dataset"]["bbox_anno_root"], requested=requested)

    summaries = []
    for sequence_name in sequence_names:
        try:
            summaries.append(
                run_sequence(
                    cfg,
                    method=args.method,
                    sequence_name=sequence_name,
                    model_override=args.model,
                    device_override=args.device,
                    max_frames_override=args.max_frames,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            if not args.skip_missing:
                raise
            warnings.warn(f"Skipping sequence {sequence_name}: {exc}", RuntimeWarning, stacklevel=2)
    print(json.dumps({"runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

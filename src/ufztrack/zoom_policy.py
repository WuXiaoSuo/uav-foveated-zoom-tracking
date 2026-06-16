from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ZoomDecision:
    level: int
    command: str
    state: str = ""
    zoom_in_score: float = 0.0
    zoom_out_score: float = 0.0
    scale_need: float = 0.0
    edge_risk: float = 0.0
    blur_risk: float = 0.0
    association_risk: float = 0.0
    lost_risk: float = 0.0
    overzoom_risk: float = 0.0
    cooldown_remaining: int = 0
    stable_count: int = 0
    unstable_count: int = 0
    risk_level: str = ""
    zoom_veto_reason: str = ""
    proposed_action: str = ""
    final_action: str = ""
    veto_applied: bool = False
    veto_reason: str = ""
    decision_reason: str = ""


@dataclass
class ZoomPolicyConfig:
    levels: Sequence[int] = (1, 2, 4, 8)
    fixed_tele_level: int = 4
    area_level_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {
            "level8_max": 0.0025,
            "level4_max": 0.0100,
            "level2_max": 0.0400,
        }
    )
    confidence_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {
            "level1_max": 0.25,
            "level2_max": 0.55,
            "level4_max": 0.80,
        }
    )
    cooldown_frames: int = 5
    hysteresis_ratio: float = 0.15
    high_uncertainty: float = 0.65
    medium_uncertainty: float = 0.45
    high_edge_risk: float = 0.70
    lost_wide_after: int = 2
    v2_threshold_in: float = 0.20
    v2_threshold_out: float = 0.05
    v2_area_min: float = 0.010
    v2_area_high: float = 0.090
    v2_high_uncertainty: float = 0.60
    v2_low_uncertainty: float = 0.35
    v2_high_blur_risk: float = 0.70
    v2_high_association_risk: float = 0.65
    v2_lost_to_caution: int = 1
    v2_lost_to_recovery: int = 2
    v2_force_wide_lost: int = 4
    v2_stable_frames_to_tracking: int = 3
    v2_min_stable_zoom_in_frames: int = 2
    v2_disable_recovery: bool = False
    v2_disable_edge_suppression: bool = False
    v2_disable_lowconf_assoc: bool = False
    v21_stable_frames_to_zoom_in: int = 2
    v21_stable_conf_thresh: float = 0.25
    v21_ambiguity_safe_thresh: float = 0.45
    v21_innovation_safe_thresh: float = 0.50
    v21_edge_safe_thresh: float = 0.65
    v21_blur_safe_thresh: float = 0.65
    v21_hold_min_frames_after_zoom: int = 8
    v21_cooldown_frames: int = 4
    v21_recovery_cooldown_frames: int = 3
    v21_hard_edge_thresh: float = 0.85
    v21_hard_blur_thresh: float = 0.85
    v21_hard_assoc_thresh: float = 0.80
    v21_area_low: float = 0.0025
    v21_area_target: float = 0.006
    v21_area_high: float = 0.035
    v21_caution_unstable_frames: int = 3
    v21_predict_only_to_recovery: int = 2
    v22_hard_edge_thresh: float = 0.88
    v22_hard_assoc_thresh: float = 0.90
    v22_hard_blur_thresh: float = 0.90
    v22_hard_overzoom_area: float = 0.040
    v22_lost_recovery_thresh: int = 2
    v22_force_wide_lost: int = 4
    v22_cooldown_frames: int = 4
    v22_hold_min_frames_after_zoom: int = 4
    v22_predict_only_to_recovery: int = 2

    def __post_init__(self) -> None:
        area_defaults = {
            "level8_max": 0.0025,
            "level4_max": 0.0100,
            "level2_max": 0.0400,
        }
        confidence_defaults = {
            "level1_max": 0.25,
            "level2_max": 0.55,
            "level4_max": 0.80,
        }
        self.area_level_thresholds = {**area_defaults, **dict(self.area_level_thresholds)}
        self.confidence_thresholds = {**confidence_defaults, **dict(self.confidence_thresholds)}


class ZoomPolicy:
    def __init__(self, method: str, config: ZoomPolicyConfig | None = None) -> None:
        self.method = method
        self.config = config or ZoomPolicyConfig()
        self.levels = sorted(int(v) for v in self.config.levels)
        self._last_change_frame = -10**9
        self._v2_state = "TRACKING"
        self._stable_frames = 0
        self._unstable_frames = 0
        self._association_failure_streak = 0
        self._last_zoom_direction = "none"
        if method not in {
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
        }:
            raise ValueError(f"Unsupported method: {method}")

    def initial_runtime_level(self, configured_initial_level: int = 1) -> int:
        if self.method == "fixed_tele":
            return self._nearest_level(self.config.fixed_tele_level)
        if self.method == "fixed_wide":
            return 1
        return self._nearest_level(configured_initial_level)

    def decide(
        self,
        current_level: int,
        area_ratio: float,
        confidence: float,
        uncertainty: float,
        edge_risk: float,
        lost_count: int,
        frame_idx: int,
        blur_risk: float = 0.0,
        association_risk: float = 0.0,
        assoc_stage: str = "high",
        kalman_innovation: float = 0.0,
        association_ambiguity: float = 0.0,
    ) -> ZoomDecision:
        current_level = self._nearest_level(current_level)
        if self.method == "ufz_v2_2":
            return self._decide_v22(
                current_level=current_level,
                area_ratio=area_ratio,
                confidence=confidence,
                uncertainty=uncertainty,
                edge_risk=edge_risk,
                lost_count=lost_count,
                frame_idx=frame_idx,
                blur_risk=blur_risk,
                association_risk=association_risk,
                assoc_stage=assoc_stage,
            )
        if self.method == "ufz_v2_1":
            return self._decide_v21(
                current_level=current_level,
                area_ratio=area_ratio,
                confidence=confidence,
                uncertainty=uncertainty,
                edge_risk=edge_risk,
                lost_count=lost_count,
                frame_idx=frame_idx,
                blur_risk=blur_risk,
                association_risk=association_risk,
                assoc_stage=assoc_stage,
                kalman_innovation=kalman_innovation,
                association_ambiguity=association_ambiguity,
            )
        if self.method.startswith("ufz_v2"):
            return self._decide_v2(
                current_level=current_level,
                area_ratio=area_ratio,
                confidence=confidence,
                uncertainty=uncertainty,
                edge_risk=edge_risk,
                lost_count=lost_count,
                frame_idx=frame_idx,
                blur_risk=blur_risk,
                association_risk=association_risk,
                assoc_stage=assoc_stage,
                kalman_innovation=kalman_innovation,
            )
        if self.method == "fixed_wide":
            return self._finalize_decision(current_level, 1, frame_idx)
        if self.method == "fixed_tele":
            target = self._nearest_level(self.config.fixed_tele_level)
            return self._finalize_decision(current_level, target, frame_idx)
        if self.method == "scale_only":
            target = self._level_from_area(area_ratio)
            target = self._apply_common_guards(current_level, target, area_ratio, edge_risk, lost_count)
            return self._finalize_decision(current_level, target, frame_idx, bypass_cooldown=lost_count > 0)
        if self.method == "confidence_only":
            target = self._level_from_confidence(confidence)
            target = self._apply_common_guards(current_level, target, area_ratio, edge_risk, lost_count)
            return self._finalize_decision(current_level, target, frame_idx, bypass_cooldown=lost_count > 0)

        target = self._level_from_area(area_ratio)
        force_change = False
        if lost_count >= self.config.lost_wide_after:
            target = 1
            force_change = True
        elif edge_risk >= self.config.high_edge_risk:
            target = min(target, 2)
            force_change = target < current_level
        elif uncertainty >= self.config.high_uncertainty:
            target = min(target, 2)
            force_change = target < current_level
        elif uncertainty >= self.config.medium_uncertainty:
            target = min(target, max(current_level, 4))

        target = self._nearest_level(target)
        if not force_change:
            target = self._apply_hysteresis(current_level, target, area_ratio)
        target = self._apply_common_guards(current_level, target, area_ratio, edge_risk, lost_count)
        return self._finalize_decision(current_level, target, frame_idx, bypass_cooldown=lost_count > 0)

    def _decide_v2(
        self,
        current_level: int,
        area_ratio: float,
        confidence: float,
        uncertainty: float,
        edge_risk: float,
        lost_count: int,
        frame_idx: int,
        blur_risk: float,
        association_risk: float,
        assoc_stage: str,
        kalman_innovation: float,
    ) -> ZoomDecision:
        no_recovery = self.method == "ufz_v2_no_recovery" or self.config.v2_disable_recovery
        no_edge_suppression = (
            self.method == "ufz_v2_no_edge_suppression" or self.config.v2_disable_edge_suppression
        )
        scale_need = self._scale_need(area_ratio)
        lost_risk = self._lost_risk(lost_count)
        overzoom_risk = self._overzoom_risk(area_ratio)
        stable_tracking_bonus = self._stable_tracking_bonus(confidence, kalman_innovation, assoc_stage)
        zoom_in_score = scale_need + 0.5 * uncertainty + stable_tracking_bonus
        zoom_out_score = (
            (0.0 if no_edge_suppression else edge_risk)
            + blur_risk
            + association_risk
            + (0.0 if no_recovery else lost_risk)
            + overzoom_risk
        )

        self._update_v2_state(
            uncertainty=uncertainty,
            edge_risk=edge_risk,
            blur_risk=blur_risk,
            association_risk=association_risk,
            lost_count=lost_count,
            confidence=confidence,
            kalman_innovation=kalman_innovation,
            assoc_stage=assoc_stage,
            no_recovery=no_recovery,
            no_edge_suppression=no_edge_suppression,
        )

        target = current_level
        reason = "score_keep"
        bypass_cooldown = False
        if self._v2_state == "RECOVERY" and not no_recovery:
            target = 1 if lost_count >= self.config.v2_force_wide_lost else self._previous_level(current_level)
            reason = "recovery_zoom_out"
            bypass_cooldown = True
        elif lost_count > 0 and not no_recovery:
            target = self._previous_level(current_level)
            reason = "lost_zoom_out"
            bypass_cooldown = True
        elif not no_edge_suppression and edge_risk >= self.config.high_edge_risk:
            target = self._previous_level(current_level) if current_level > 1 else current_level
            reason = "edge_suppression"
        elif blur_risk >= self.config.v2_high_blur_risk:
            target = self._previous_level(current_level) if current_level > 1 else current_level
            reason = "blur_suppression"
        elif overzoom_risk >= 0.5:
            target = self._previous_level(current_level) if current_level > 1 else current_level
            reason = "overzoom_suppression"
        elif self._v2_state == "CAUTION":
            if zoom_out_score - zoom_in_score > self.config.v2_threshold_out:
                target = self._previous_level(current_level)
                reason = "caution_zoom_out"
            else:
                reason = "caution_keep"
        elif (
            zoom_in_score - zoom_out_score > self.config.v2_threshold_in
            and self._stable_frames >= self.config.v2_min_stable_zoom_in_frames
            and area_ratio < self.config.v2_area_high
        ):
            target = self._next_level(current_level)
            reason = "score_zoom_in"
        elif zoom_out_score - zoom_in_score > self.config.v2_threshold_out:
            target = self._previous_level(current_level)
            reason = "score_zoom_out"

        decision = self._finalize_decision(
            current_level,
            target,
            frame_idx,
            bypass_cooldown=bypass_cooldown,
            state=self._v2_state,
            zoom_in_score=zoom_in_score,
            zoom_out_score=zoom_out_score,
            scale_need=scale_need,
            edge_risk=edge_risk,
            blur_risk=blur_risk,
            association_risk=association_risk,
            lost_risk=lost_risk,
            overzoom_risk=overzoom_risk,
            decision_reason=reason,
        )
        if decision.command == "cooldown_keep":
            return ZoomDecision(
                **{
                    **decision.__dict__,
                    "decision_reason": f"cooldown_{reason}",
                }
            )
        return decision

    def _decide_v21(
        self,
        current_level: int,
        area_ratio: float,
        confidence: float,
        uncertainty: float,
        edge_risk: float,
        lost_count: int,
        frame_idx: int,
        blur_risk: float,
        association_risk: float,
        assoc_stage: str,
        kalman_innovation: float,
        association_ambiguity: float,
    ) -> ZoomDecision:
        stable_now = (
            confidence >= self.config.v21_stable_conf_thresh
            and lost_count == 0
            and assoc_stage in {"high", "low"}
            and association_ambiguity <= self.config.v21_ambiguity_safe_thresh
            and kalman_innovation <= self.config.v21_innovation_safe_thresh
            and edge_risk <= self.config.v21_edge_safe_thresh
            and blur_risk <= self.config.v21_blur_safe_thresh
        )
        if stable_now:
            self._stable_frames += 1
            self._unstable_frames = 0
            self._association_failure_streak = 0
        else:
            self._stable_frames = 0
            self._unstable_frames += 1
            if assoc_stage == "predict_only":
                self._association_failure_streak += 1
            elif assoc_stage in {"high", "low"}:
                self._association_failure_streak = 0

        hard_edge = edge_risk > self.config.v21_hard_edge_thresh
        hard_blur = blur_risk > self.config.v21_hard_blur_thresh
        hard_assoc = association_risk > self.config.v21_hard_assoc_thresh
        hard_lost = lost_count >= self.config.v2_lost_to_recovery
        severe_risk = hard_edge or hard_blur or hard_assoc or hard_lost
        if hard_lost or self._association_failure_streak >= self.config.v21_predict_only_to_recovery:
            state = "RECOVERY"
        elif severe_risk or self._unstable_frames >= self.config.v21_caution_unstable_frames:
            state = "CAUTION"
        else:
            state = "TRACKING"
        self._v2_state = state

        scale_need = self._v21_scale_need(area_ratio)
        overzoom_risk = self._v21_overzoom_risk(area_ratio)
        lost_risk = self._lost_risk(lost_count)
        uncertainty_need = min(max(float(uncertainty), 0.0), 1.0)
        zoom_in_score = scale_need + 0.35 * uncertainty_need + min(0.30, 0.10 * self._stable_frames)
        zoom_out_score = (
            (1.0 if hard_edge else 0.25 * edge_risk)
            + (1.0 if hard_blur else 0.25 * blur_risk)
            + (1.0 if hard_assoc else 0.25 * association_risk)
            + lost_risk
            + overzoom_risk
        )

        stable_tracking = self._stable_frames >= self.config.v21_stable_frames_to_zoom_in
        area_too_small = area_ratio <= self.config.v21_area_low
        want_zoom_in = stable_tracking and area_ratio < self.config.v21_area_target
        want_zoom_out = overzoom_risk >= 0.5 or lost_count >= self.config.v2_lost_to_recovery

        target = current_level
        reason = "normal_keep"
        veto = "none"
        cooldown_frames = self.config.v21_cooldown_frames
        bypass_cooldown = False

        if lost_count >= self.config.v2_force_wide_lost:
            target = self._previous_level(current_level)
            reason = "lost_recovery_zoom_out"
            veto = "lost_force_wide"
            cooldown_frames = self.config.v21_recovery_cooldown_frames
        elif state == "RECOVERY":
            target = self._previous_level(current_level)
            reason = "lost_recovery_zoom_out"
            veto = "lost_or_predict_only"
            cooldown_frames = self.config.v21_recovery_cooldown_frames
        elif hard_edge:
            target = self._previous_level(current_level)
            reason = "risk_veto_zoom_out"
            veto = "hard_edge"
            bypass_cooldown = True
        elif hard_blur:
            target = self._previous_level(current_level) if overzoom_risk >= 0.5 else current_level
            reason = "risk_veto_keep" if target == current_level else "risk_veto_zoom_out"
            veto = "hard_blur"
            bypass_cooldown = target != current_level
        elif hard_assoc:
            target = current_level
            reason = "risk_veto_keep"
            veto = "hard_association"
        elif want_zoom_in:
            target = self._next_level(current_level)
            reason = "scale_need_stable_zoom_in"
        elif want_zoom_out:
            target = self._previous_level(current_level)
            reason = "overzoom_zoom_out"

        hold_reason = ""
        if veto not in {"hard_edge", "hard_blur", "lost_force_wide", "lost_or_predict_only"}:
            hold_reason = self._v21_hold_reason(
                current_level=current_level,
                target_level=target,
                frame_idx=frame_idx,
                lost_count=lost_count,
                area_too_small=area_too_small,
                stable_tracking=stable_tracking,
            )
        if hold_reason:
            target = current_level
            reason = "hold_keep"
            veto = hold_reason
            bypass_cooldown = True

        if target == current_level:
            cooldown_frames = 0

        risk_level = "hard" if severe_risk else ("moderate" if self._unstable_frames > 0 or uncertainty >= 0.55 else "low")
        decision = self._finalize_decision(
            current_level,
            target,
            frame_idx,
            bypass_cooldown=bypass_cooldown,
            cooldown_frames=cooldown_frames,
            state=state,
            zoom_in_score=zoom_in_score,
            zoom_out_score=zoom_out_score,
            scale_need=scale_need,
            edge_risk=edge_risk,
            blur_risk=blur_risk,
            association_risk=association_risk,
            lost_risk=lost_risk,
            overzoom_risk=overzoom_risk,
            stable_count=self._stable_frames,
            unstable_count=self._unstable_frames,
            risk_level=risk_level,
            zoom_veto_reason=veto,
            decision_reason=reason,
        )
        if decision.command == "cooldown_keep":
            return ZoomDecision(
                **{
                    **decision.__dict__,
                    "decision_reason": f"cooldown_{reason}",
                }
            )
        return decision

    def _decide_v22(
        self,
        current_level: int,
        area_ratio: float,
        confidence: float,
        uncertainty: float,
        edge_risk: float,
        lost_count: int,
        frame_idx: int,
        blur_risk: float,
        association_risk: float,
        assoc_stage: str,
    ) -> ZoomDecision:
        proposed_target = self._propose_ufz_v1_target(
            current_level=current_level,
            area_ratio=area_ratio,
            uncertainty=uncertainty,
            edge_risk=edge_risk,
            lost_count=lost_count,
        )
        proposed_step = self._bounded_step(current_level, proposed_target)
        proposed_action = _command(current_level, proposed_step)

        if assoc_stage == "predict_only":
            self._association_failure_streak += 1
        else:
            self._association_failure_streak = 0

        lost_recovery = (
            lost_count >= self.config.v22_lost_recovery_thresh
            or self._association_failure_streak >= self.config.v22_predict_only_to_recovery
        )
        overzoom_risk = 1.0 if area_ratio >= self.config.v22_hard_overzoom_area else 0.0
        lost_risk = self._lost_risk(lost_count)
        hard_edge = edge_risk >= self.config.v22_hard_edge_thresh
        hard_assoc = association_risk >= self.config.v22_hard_assoc_thresh
        hard_blur = blur_risk >= self.config.v22_hard_blur_thresh
        hard_overzoom = overzoom_risk >= 1.0

        target = proposed_step
        reason = "follow_v1_policy"
        veto_reason = "none"
        veto_applied = False
        bypass_cooldown = False
        state = "TRACKING"

        if lost_recovery:
            target = self._previous_level(current_level)
            if lost_count >= self.config.v22_force_wide_lost:
                reason = "lost_force_wide_veto"
                veto_reason = "lost_force_wide"
            else:
                reason = "lost_recovery_veto"
                veto_reason = "lost_recovery"
            veto_applied = True
            bypass_cooldown = True
            state = "RECOVERY"
        elif proposed_action.startswith("zoom_in") and hard_edge:
            target = self._previous_level(current_level) if current_level > 1 else current_level
            reason = "hard_edge_veto"
            veto_reason = "hard_edge"
            veto_applied = True
        elif proposed_action.startswith("zoom_in") and hard_assoc:
            target = current_level
            reason = "hard_assoc_veto"
            veto_reason = "hard_association"
            veto_applied = True
        elif proposed_action.startswith("zoom_in") and hard_blur:
            target = current_level
            reason = "hard_blur_veto"
            veto_reason = "hard_blur"
            veto_applied = True
        elif proposed_action.startswith("zoom_in") and hard_overzoom:
            target = current_level
            reason = "overzoom_veto"
            veto_reason = "hard_overzoom"
            veto_applied = True

        if not veto_applied and target != current_level:
            frames_since_change = frame_idx - self._last_change_frame
            new_direction = "in" if target > current_level else "out"
            if (
                frames_since_change < self.config.v22_hold_min_frames_after_zoom
                and self._last_zoom_direction in {"in", "out"}
                and new_direction != self._last_zoom_direction
            ):
                target = current_level
                reason = "hold_keep"
                veto_reason = f"hold_after_zoom_{self._last_zoom_direction}"
                veto_applied = True

        risk_level = "hard" if lost_recovery or hard_edge or hard_assoc or hard_blur or hard_overzoom else "low"
        decision = self._finalize_decision(
            current_level,
            target,
            frame_idx,
            bypass_cooldown=bypass_cooldown,
            cooldown_frames=self.config.v22_cooldown_frames,
            state=state,
            zoom_in_score=1.0 if proposed_action.startswith("zoom_in") else 0.0,
            zoom_out_score=1.0 if proposed_action.startswith("zoom_out") or lost_recovery else 0.0,
            scale_need=self._scale_need(area_ratio),
            edge_risk=edge_risk,
            blur_risk=blur_risk,
            association_risk=association_risk,
            lost_risk=lost_risk,
            overzoom_risk=overzoom_risk,
            risk_level=risk_level,
            zoom_veto_reason=veto_reason,
            proposed_action=proposed_action,
            veto_applied=veto_applied,
            veto_reason=veto_reason,
            decision_reason=reason,
        )
        if decision.command == "cooldown_keep":
            return ZoomDecision(
                **{
                    **decision.__dict__,
                    "decision_reason": f"cooldown_{reason}",
                    "veto_reason": "cooldown",
                }
            )
        return decision

    def _level_from_area(self, area_ratio: float) -> int:
        ratio = max(0.0, float(area_ratio))
        thresholds = self.config.area_level_thresholds
        if ratio <= float(thresholds["level8_max"]):
            return 8
        if ratio <= float(thresholds["level4_max"]):
            return 4
        if ratio <= float(thresholds["level2_max"]):
            return 2
        return 1

    def _level_from_confidence(self, confidence: float) -> int:
        conf = max(0.0, min(float(confidence), 1.0))
        thresholds = self.config.confidence_thresholds
        if conf <= float(thresholds["level1_max"]):
            return 1
        if conf <= float(thresholds["level2_max"]):
            return 2
        if conf <= float(thresholds["level4_max"]):
            return 4
        return 8

    def _nearest_level(self, level: int) -> int:
        value = int(level)
        return min(self.levels, key=lambda candidate: abs(candidate - value))

    def _next_level(self, current_level: int) -> int:
        index = self.levels.index(self._nearest_level(current_level))
        return self.levels[min(index + 1, len(self.levels) - 1)]

    def _previous_level(self, current_level: int) -> int:
        index = self.levels.index(self._nearest_level(current_level))
        return self.levels[max(index - 1, 0)]

    def _bounded_step(self, current_level: int, target_level: int) -> int:
        current_level = self._nearest_level(current_level)
        target_level = self._nearest_level(target_level)
        current_index = self.levels.index(current_level)
        target_index = self.levels.index(target_level)
        if target_index > current_index:
            return self.levels[current_index + 1]
        if target_index < current_index:
            return self.levels[current_index - 1]
        return current_level

    def _apply_common_guards(
        self,
        current_level: int,
        target_level: int,
        area_ratio: float,
        edge_risk: float,
        lost_count: int,
    ) -> int:
        target_level = self._nearest_level(target_level)
        if lost_count > 0:
            if current_level <= 1:
                return 1
            return self.levels[self.levels.index(current_level) - 1]
        if target_level > current_level and edge_risk >= self.config.high_edge_risk:
            return current_level
        if target_level > current_level and area_ratio >= float(self.config.area_level_thresholds["level2_max"]):
            return current_level
        return target_level

    def _propose_ufz_v1_target(
        self,
        current_level: int,
        area_ratio: float,
        uncertainty: float,
        edge_risk: float,
        lost_count: int,
    ) -> int:
        target = self._level_from_area(area_ratio)
        force_change = False
        if lost_count >= self.config.lost_wide_after:
            target = 1
            force_change = True
        elif edge_risk >= self.config.high_edge_risk:
            target = min(target, 2)
            force_change = target < current_level
        elif uncertainty >= self.config.high_uncertainty:
            target = min(target, 2)
            force_change = target < current_level
        elif uncertainty >= self.config.medium_uncertainty:
            target = min(target, max(current_level, 4))

        target = self._nearest_level(target)
        if not force_change:
            target = self._apply_hysteresis(current_level, target, area_ratio)
        return self._apply_common_guards(current_level, target, area_ratio, edge_risk, lost_count)

    def _finalize_decision(
        self,
        current_level: int,
        target_level: int,
        frame_idx: int,
        bypass_cooldown: bool = False,
        cooldown_frames: int | None = None,
        state: str = "",
        zoom_in_score: float = 0.0,
        zoom_out_score: float = 0.0,
        scale_need: float = 0.0,
        edge_risk: float = 0.0,
        blur_risk: float = 0.0,
        association_risk: float = 0.0,
        lost_risk: float = 0.0,
        overzoom_risk: float = 0.0,
        stable_count: int = 0,
        unstable_count: int = 0,
        risk_level: str = "",
        zoom_veto_reason: str = "",
        proposed_action: str = "",
        veto_applied: bool = False,
        veto_reason: str = "",
        decision_reason: str = "",
    ) -> ZoomDecision:
        bounded_target = self._bounded_step(current_level, target_level)
        active_cooldown = self.config.cooldown_frames if cooldown_frames is None else int(cooldown_frames)
        in_cooldown = frame_idx - self._last_change_frame < active_cooldown
        cooldown_remaining = max(0, active_cooldown - (frame_idx - self._last_change_frame))
        if bounded_target != current_level and in_cooldown and not bypass_cooldown:
            action = "cooldown_keep"
            return ZoomDecision(
                current_level,
                action,
                state=state,
                zoom_in_score=zoom_in_score,
                zoom_out_score=zoom_out_score,
                scale_need=scale_need,
                edge_risk=edge_risk,
                blur_risk=blur_risk,
                association_risk=association_risk,
                lost_risk=lost_risk,
                overzoom_risk=overzoom_risk,
                cooldown_remaining=cooldown_remaining,
                stable_count=stable_count,
                unstable_count=unstable_count,
                risk_level=risk_level,
                zoom_veto_reason=zoom_veto_reason,
                proposed_action=proposed_action,
                final_action=action,
                veto_applied=veto_applied,
                veto_reason=veto_reason,
                decision_reason=decision_reason,
            )
        if bounded_target != current_level:
            self._last_change_frame = frame_idx
            self._last_zoom_direction = "in" if bounded_target > current_level else "out"
            cooldown_remaining = active_cooldown
        else:
            cooldown_remaining = 0
        action = _command(current_level, bounded_target)
        return ZoomDecision(
            bounded_target,
            action,
            state=state,
            zoom_in_score=zoom_in_score,
            zoom_out_score=zoom_out_score,
            scale_need=scale_need,
            edge_risk=edge_risk,
            blur_risk=blur_risk,
            association_risk=association_risk,
            lost_risk=lost_risk,
            overzoom_risk=overzoom_risk,
            cooldown_remaining=cooldown_remaining,
            stable_count=stable_count,
            unstable_count=unstable_count,
            risk_level=risk_level,
            zoom_veto_reason=zoom_veto_reason,
            proposed_action=proposed_action,
            final_action=action,
            veto_applied=veto_applied,
            veto_reason=veto_reason,
            decision_reason=decision_reason,
        )

    def _scale_need(self, area_ratio: float) -> float:
        area = max(0.0, float(area_ratio))
        return min(max((self.config.v2_area_min - area) / max(self.config.v2_area_min, 1e-9), 0.0), 1.0)

    def _overzoom_risk(self, area_ratio: float) -> float:
        area = max(0.0, float(area_ratio))
        return min(max((area - self.config.v2_area_high) / max(self.config.v2_area_high, 1e-9), 0.0), 1.0)

    def _lost_risk(self, lost_count: int) -> float:
        return min(max(float(lost_count) / max(float(self.config.v2_force_wide_lost), 1.0), 0.0), 1.0)

    def _stable_tracking_bonus(self, confidence: float, kalman_innovation: float, assoc_stage: str) -> float:
        if assoc_stage not in {"high", "low"}:
            return 0.0
        if confidence < 0.35 or kalman_innovation > 0.80:
            return 0.0
        return min(0.40, 0.10 * self._stable_frames)

    def _v21_scale_need(self, area_ratio: float) -> float:
        area = max(0.0, float(area_ratio))
        if area <= self.config.v21_area_low:
            return 1.0
        if area >= self.config.v21_area_target:
            return 0.0
        span = max(self.config.v21_area_target - self.config.v21_area_low, 1e-9)
        return min(max((self.config.v21_area_target - area) / span, 0.0), 1.0)

    def _v21_overzoom_risk(self, area_ratio: float) -> float:
        area = max(0.0, float(area_ratio))
        if area <= self.config.v21_area_high:
            return 0.0
        return min(max((area - self.config.v21_area_high) / max(self.config.v21_area_high, 1e-9), 0.0), 1.0)

    def _v21_hold_reason(
        self,
        current_level: int,
        target_level: int,
        frame_idx: int,
        lost_count: int,
        area_too_small: bool,
        stable_tracking: bool,
    ) -> str:
        if target_level == current_level:
            return ""
        frames_since_change = frame_idx - self._last_change_frame
        if frames_since_change >= self.config.v21_hold_min_frames_after_zoom:
            return ""
        if target_level < current_level and self._last_zoom_direction == "in" and lost_count < self.config.v2_lost_to_recovery:
            return "hold_after_zoom_in"
        if (
            target_level > current_level
            and self._last_zoom_direction == "out"
            and not (area_too_small and stable_tracking)
        ):
            return "hold_after_zoom_out"
        return ""

    def _update_v2_state(
        self,
        uncertainty: float,
        edge_risk: float,
        blur_risk: float,
        association_risk: float,
        lost_count: int,
        confidence: float,
        kalman_innovation: float,
        assoc_stage: str,
        no_recovery: bool,
        no_edge_suppression: bool,
    ) -> None:
        associated = assoc_stage in {"high", "low"}
        stable = associated and confidence >= 0.45 and kalman_innovation <= 0.60 and uncertainty <= self.config.v2_low_uncertainty
        if stable:
            self._stable_frames += 1
            self._association_failure_streak = 0
        else:
            self._stable_frames = 0
            if assoc_stage == "predict_only":
                self._association_failure_streak += 1

        caution_signal = (
            uncertainty >= self.config.v2_high_uncertainty
            or association_risk >= self.config.v2_high_association_risk
            or blur_risk >= self.config.v2_high_blur_risk
            or (not no_edge_suppression and edge_risk >= self.config.high_edge_risk)
        )
        recovery_signal = (
            not no_recovery
            and (
                lost_count >= self.config.v2_lost_to_recovery
                or self._association_failure_streak >= self.config.v2_lost_to_recovery
            )
        )
        if recovery_signal:
            self._v2_state = "RECOVERY"
        elif lost_count >= self.config.v2_lost_to_caution or caution_signal:
            self._v2_state = "CAUTION"
        elif self._v2_state in {"CAUTION", "RECOVERY"}:
            if self._stable_frames >= self.config.v2_stable_frames_to_tracking:
                self._v2_state = "TRACKING"
        else:
            self._v2_state = "TRACKING"

    def _apply_hysteresis(self, current_level: int, target_level: int, area_ratio: float) -> int:
        if target_level == current_level:
            return target_level
        margin = max(0.0, float(self.config.hysteresis_ratio))
        thresholds = self.config.area_level_thresholds
        up_thresholds = {
            2: float(thresholds["level2_max"]) * (1.0 - margin),
            4: float(thresholds["level4_max"]) * (1.0 - margin),
            8: float(thresholds["level8_max"]) * (1.0 - margin),
        }
        down_thresholds = {
            8: float(thresholds["level8_max"]) * (1.0 + margin),
            4: float(thresholds["level4_max"]) * (1.0 + margin),
            2: float(thresholds["level2_max"]) * (1.0 + margin),
        }
        ratio = float(area_ratio)
        if target_level > current_level:
            threshold = up_thresholds.get(target_level, 0.0)
            return target_level if ratio <= threshold else current_level
        threshold = down_thresholds.get(current_level, 0.0)
        return target_level if ratio >= threshold else current_level


def _command(current_level: int, target_level: int) -> str:
    if target_level > current_level:
        return f"zoom_in_to_{target_level}"
    if target_level < current_level:
        return f"zoom_out_to_{target_level}"
    return "keep"

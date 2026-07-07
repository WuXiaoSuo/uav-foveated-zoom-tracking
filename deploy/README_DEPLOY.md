# UFZ-Deploy-v1: Single H30 Z-Stream Closed Loop

This deploy target is intentionally narrow: one DJI H30 Z stream, one manually initialized target, one tracker, gimbal centering from bbox center error, and a conservative optical zoom state machine.

It does not run W/Z offline replay, Oracle decisions, cross-camera reacquisition, detector training, or DJI PSDK control by default.

## Files

```text
deploy/run_ufz_deploy_v1.py
deploy/config_h30_deploy_v1.yaml
deploy/README_DEPLOY.md
```

## What v1 Does

1. Reads an offline video file or image directory. RTSP URLs are accepted through OpenCV `VideoCapture` as a reserved smoke-test path, but reconnect and latency handling are not implemented yet.
2. Gets the initial target bbox from `--init-bbox x,y,w,h` or OpenCV interactive ROI selection.
3. Updates a tracker on each frame. The script tries OpenCV CSRT/KCF/MIL and falls back to a light template tracker if contrib trackers are unavailable.
4. Computes normalized bbox center error and emits yaw/pitch rate commands through a command sink.
5. Runs a conservative optical zoom state machine over 2x/5x/10x only. 20x and 40x are disabled in config.
6. Writes a CSV log and an annotated visualization video.

## Quick Offline Smoke Test

Headless/manual bbox:

```bash
python3 deploy/run_ufz_deploy_v1.py \
  --config deploy/config_h30_deploy_v1.yaml \
  --input /path/to/video.mp4 \
  --init-bbox 820,420,120,260 \
  --output-dir outputs/deploy_v1/smoke_video \
  --tracker auto
```

Interactive bbox selection:

```bash
python3 deploy/run_ufz_deploy_v1.py \
  --input /path/to/frames_dir \
  --output-dir outputs/deploy_v1/smoke_frames
```

Disable video writing for faster CSV-only checks:

```bash
python3 deploy/run_ufz_deploy_v1.py \
  --input /path/to/video.mp4 \
  --init-bbox 820,420,120,260 \
  --no-video
```

Outputs:

```text
outputs/deploy_v1/<run_id>/ufz_deploy_v1_log.csv
outputs/deploy_v1/<run_id>/ufz_deploy_v1_vis.mp4
outputs/deploy_v1/<run_id>/ufz_deploy_v1_metadata.json
```

## CSV Log Fields

The CSV records frame index, timestamp, tracker validity, bbox, normalized center error, area ratio, edge margin, zoom level/command/reason, gimbal yaw/pitch rates, lost count, zoom stability count, and tracker/frame latency.

These logs are for deployment debugging rather than benchmark evaluation.

## Gimbal Command Convention

The script computes:

```text
yaw_rate_deg_s   = center_error_x * yaw_kp
pitch_rate_deg_s = center_error_y * pitch_kp
```

`center_error_x` is positive when the target is right of image center. `center_error_y` is positive when the target is below image center. The default `pitch_kp` is negative, but yaw/pitch signs must be validated on the real H30 before enabling any real PSDK sink.

## Zoom State Machine

Enabled levels are:

```text
[2x, 5x, 10x]
```

Disabled for this phase:

```text
[20x, 40x]
```

The state machine zooms out on invalid/lost tracking, edge risk, or very large target area. It zooms in only when the target is small, away from image edges, valid, stable for several frames, and the zoom cooldown has expired.

This is deliberately conservative for the first Manifold 3 + H30 deployment window.

## PSDK Integration Hook

`PsdkCommandSink` in `run_ufz_deploy_v1.py` is a placeholder. Wire real DJI calls there after validating:

```text
1. H30 zoom level command mapping for 2x/5x/10x.
2. Gimbal yaw/pitch sign conventions.
3. Rate units and saturation limits.
4. Emergency stop / operator override.
5. RTSP latency and dropped-frame behavior.
```

Until then, keep:

```yaml
gimbal:
  command_sink: dry_run
```

## Four-Day Deployment Checklist

1. Day 1: Run offline video and image-dir replay with `--init-bbox`; verify CSV and visualization video.
2. Day 2: Tune yaw/pitch gain signs on a static target with command sink still dry-run.
3. Day 3: Wire PSDK placeholder behind an operator kill switch; test 2x/5x/10x only.
4. Day 4: Outdoor H30 smoke test with conservative zoom thresholds and CSV/video logging enabled.

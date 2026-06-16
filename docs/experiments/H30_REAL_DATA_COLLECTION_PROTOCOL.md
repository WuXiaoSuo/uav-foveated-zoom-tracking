# H30 Real Data Collection Protocol

## Goal

Collect a small, reproducible real-world dataset to validate UFZ-v2.2 beyond offline UAV123 software zoom. The experiment should compare fixed zoom baselines and failure-aware UFZ-v2.2 under the same DJI M400 + H30 + Manifold 3 setup.

## Hardware Setup

- Platform: DJI M400.
- Camera: DJI H30.
- Onboard computer: Manifold 3.
- Detector/tracker stack: same Experiment 1 code path where possible, using YOLOv8n unless a separate detector comparison is explicitly planned.
- Recording: save camera frames or video with synchronized timestamps and zoom telemetry.

## Modes

Collect each scene under these modes:

- `fixed_wide`: keep wide search zoom.
- `fixed_tele`: keep a fixed telephoto zoom level selected before flight.
- `ufz_v2_2`: bounded UFZ-v2.2 policy with hard-risk veto and lost recovery.

Do not mix modes inside a single annotated sequence unless the sequence is clearly labeled as a transition or calibration run.

## Clip Design

- Duration: 30-90 seconds per video.
- Repetitions: 3-5 videos per scene and mode when battery, safety, and site constraints allow.
- Frame rate: record the native stream rate, then document any downsampling used for evaluation.
- Resolution: record the source resolution and evaluation resolution.
- Annotation format: UAV123-style `x,y,w,h` per frame.

## Scene Types

Prioritize scenes that stress zoom control and failure recovery:

- Pedestrian or cyclist with changing apparent scale.
- Vehicle moving along a road or open area.
- Boat or water-like background if available.
- Building facade or high-edge-risk static target.
- Target passing near frame boundaries.
- Long-range small target with cluttered background.
- Temporary occlusion or low-confidence association scenario.

## Flight And Camera Procedure

1. Calibrate timestamps for camera frames, zoom commands, and command acknowledgements.
2. Verify H30 zoom telemetry reports focal length or zoom ratio.
3. Start recording before the target enters the primary evaluation interval.
4. For each sequence, keep target identity fixed and avoid intentional target switching.
5. Use bounded zoom commands only for `ufz_v2_2`.
6. Record command send time and acknowledgement time for latency analysis.
7. Stop recording only after the target is clearly gone or the planned duration is complete.

## Required Logs

For every frame or decision step, log:

- `frame_id`
- `timestamp`
- `zoom_level`
- `zoom_command`
- `reported_focal_length`
- `reported_zoom_ratio`
- `command_sent_time`
- `command_ack_time`
- `bbox_x,bbox_y,bbox_w,bbox_h`
- `det_conf`
- `uncertainty`
- `target_area`
- `edge_risk`
- `association_risk`
- `lost_count`
- `latency_ms`
- `decision_reason`

See `docs/experiments/H30_LOG_FIELDS.md` for field definitions.

## Annotation Rules

- Use UAV123 `x,y,w,h` boxes in original frame coordinates.
- Mark one target identity per sequence.
- Do not use annotations for online zoom decisions.
- Keep frames with full occlusion as invalid or absent according to the chosen annotation tool, and document that choice.
- Store annotation files separately from raw videos; do not commit raw data or videos to Git.

## File Management

- Local raw data should remain under `data/` or external storage.
- Server outputs should remain under `/root/autodl-tmp/UFZTrack/outputs`.
- Downloaded experiment outputs should go under `server_outputs/{experiment_name}/outputs/`.
- GitHub-safe summaries, templates, and small CSV/PNG artifacts may go under `docs/experiments/` and `artifacts/experiments/`.

## Safety And Privacy

- Follow local UAV flight regulations and site permissions.
- Keep safe distance from people, vehicles, buildings, and obstacles.
- Avoid collecting identifiable private information when possible.
- Blur or exclude faces, plates, and private areas before any public release.
- Stop collection if the operator loses safe visual control or the target scenario becomes unsafe.

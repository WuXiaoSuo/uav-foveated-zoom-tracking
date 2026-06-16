# H30 Log Fields

This file defines the minimum per-frame or per-decision log fields for the DJI M400 + H30 + Manifold 3 real-data experiment.

| field | description |
|---|---|
| `frame_id` | 1-based frame index in the evaluated sequence. |
| `timestamp` | Camera frame timestamp, preferably in seconds or ISO time with timezone. |
| `zoom_level` | Discrete evaluation zoom level, normally one of `1`, `2`, `4`, `8`. |
| `zoom_command` | Command emitted by the policy, such as `keep`, `zoom_in_to_2`, or `zoom_out_to_1`. |
| `reported_focal_length` | Focal length reported by the H30 telemetry, if available. |
| `reported_zoom_ratio` | Optical zoom ratio reported by the H30 telemetry, if available. |
| `command_sent_time` | Timestamp when a zoom command was sent. |
| `command_ack_time` | Timestamp when the command was acknowledged or observed in telemetry. |
| `bbox_x` | Tracker output bbox x coordinate in original image coordinates. |
| `bbox_y` | Tracker output bbox y coordinate in original image coordinates. |
| `bbox_w` | Tracker output bbox width in original image coordinates. |
| `bbox_h` | Tracker output bbox height in original image coordinates. |
| `det_conf` | Detector or association confidence used by the tracker. |
| `uncertainty` | Combined policy uncertainty signal. |
| `target_area` | Target bbox area ratio relative to image area. |
| `edge_risk` | Risk that target is too close to crop or image edge. |
| `association_risk` | Risk that the selected measurement is unreliable or missing. |
| `lost_count` | Consecutive lost or predict-only frame count. |
| `latency_ms` | End-to-end decision latency in milliseconds. |
| `decision_reason` | Human-readable reason emitted by the policy. |

Recommended CSV header:

```text
frame_id,timestamp,zoom_level,zoom_command,reported_focal_length,reported_zoom_ratio,command_sent_time,command_ack_time,bbox_x,bbox_y,bbox_w,bbox_h,det_conf,uncertainty,target_area,edge_risk,association_risk,lost_count,latency_ms,decision_reason
```

Use empty strings for telemetry fields that are unavailable, but keep the columns present so analysis scripts remain stable.

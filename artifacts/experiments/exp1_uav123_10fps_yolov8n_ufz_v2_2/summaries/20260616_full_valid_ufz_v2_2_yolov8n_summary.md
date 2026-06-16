# 20260616 Full Valid UFZ-v2.2 YOLOv8n Summary

## Objective

Freeze `ufz_v2_2` as the Experiment 1 paper candidate and prepare GitHub-safe analysis artifacts for UAV123@10fps offline simulated zoom.

## Dataset And Protocol

- Dataset: UAV123@10fps single-object tracking benchmark.
- Detector: YOLOv8n, no retraining.
- Zoom: offline software zoom with bounded levels `[1, 2, 4, 8]`.
- Coordinates: all tracker outputs are saved in original image coordinates.
- Ground truth use: frame 1 initialization and final evaluation only. GT is not used for crop center, association, tracker update, or zoom decisions.
- Output root on server: `/root/autodl-tmp/UFZTrack/outputs`.

## Evaluated Methods

- `confidence_only`
- `fixed_tele`
- `fixed_wide`
- `scale_only`
- `ufz`
- `ufz_v2`
- `ufz_v2_2`

`ufz_v2_1` is not selected as the paper candidate because earlier validation showed it became overly conservative, often staying at zoom 1 and hurting active-zoom cases such as `bike1`.

## Main Results ALL

These rows were read from `server_outputs/exp1_full_valid_ufz_v2_2_yolov8n_20260616/outputs/tables/main_results.csv`.

| method | frames | mean_iou | success_auc | precision_20 | mean_cle |
|---|---:|---:|---:|---:|---:|
| confidence_only | 19596 | 0.301952 | 0.306878 | 0.372729 | 11174.15 |
| fixed_tele | 19596 | 0.398844 | 0.402195 | 0.453664 | 11163.46 |
| fixed_wide | 19596 | 0.378811 | 0.382533 | 0.470759 | 11118.15 |
| scale_only | 19596 | 0.373975 | 0.377880 | 0.454225 | 11134.54 |
| ufz | 19596 | 0.395718 | 0.399244 | 0.481017 | 11107.52 |
| ufz_v2 | 19596 | 0.380611 | 0.384226 | 0.471882 | 11109.20 |
| ufz_v2_2 | 19596 | 0.394341 | 0.397844 | 0.481170 | 11107.97 |

Honest conclusion: UFZ-v2.2 achieves competitive aggregate AUC compared with the original UFZ policy and fixed telephoto zoom, while obtaining the best DP@20 among evaluated methods. It preserves active zoom behavior and introduces hard-risk vetoes for safer deployment. However, it does not significantly outperform fixed telephoto zoom in AUC, and several unrecoverable tracker failures remain.

## Robust Metrics

The updated evaluator now writes the following additional tables:

- `outputs/tables/robust_results.csv`
- `outputs/tables/macro_results.csv`
- `outputs/tables/win_counts.csv`

The robust table adds median CLE, CLE 95th percentile, failure rate over 50 px, failure rate over 100 px, and valid CLE frame counts. The macro table averages sequence-level values instead of pooling all frames. The win-count table counts per-sequence wins by AUC, DP@20, mean CLE, median CLE, and failure_rate_50.

Regenerate these on the server after the code update:

```bash
python scripts/eval_uav123.py --config configs/uav123_10fps.yaml
```

## Zoom Behavior

From the v2.2 full-valid zoom summary:

- `bike1`: active zoom preserved, with `zoom_4=532` frames and no direct unbounded jump.
- `boat9`: uses zoom 1/2/4 with bounded in/out commands and only a few cooldown frames.
- `building4`: uses zoom 1/2/4 with risk veto/cooldown events, useful as a limitation and recovery-analysis case.
- No evidence of hundreds of consecutive `cooldown_keep` frames in the inspected v2.2 summary.

## Case Analysis

Paper-case analysis is generated with:

```bash
python scripts/analyze_ufz_cases.py \
  --output-root /root/autodl-tmp/UFZTrack/outputs \
  --baseline ufz \
  --candidate ufz_v2_2

python scripts/plot_paper_cases.py \
  --output-root /root/autodl-tmp/UFZTrack/outputs \
  --cases bike1 boat9 truck1 building4 person13 car17 \
  --methods ufz ufz_v2_2
```

Recommended paper cases: `bike1`, `boat9`, `truck1`, `building4`, `person13`, `car17`, `person1`, `person20`, `car5`, `car10`, `car9`, `boat6`, `boat7`.

## Why UFZ-v2.2 Is The Main Variant

`ufz_v2_2` keeps the original UFZ v1 zoom-in behavior as the proposal policy, then applies only hard-risk vetoes and lost recovery. This avoids the v2.1 failure mode where stable gating and conservative recovery made the method behave like fixed wide zoom. In the full-valid summary, `ufz_v2_2` remains close to `ufz` on AUC and slightly improves DP@20.

## Why UFZ-v2.1 Is Excluded

Prior validation showed `ufz_v2_1` over-constrained zoom-in, especially on `bike1`, where it failed to enter higher telephoto levels. Its gating and stable-count requirements were too conservative for a paper mainline. It can be discussed as an ablation or failed design iteration, not as the final method.

## Limitations

- Aggregate AUC is still slightly below fixed telephoto zoom in this full-valid snapshot.
- DP@20 gains are small and should be reported without overclaiming.
- Several sequences remain dominated by detector/tracker failure and cannot be solved by zoom policy alone.
- The offline software zoom experiment does not model H30 actuator latency, focal-length reporting, or real optical blur.

## H30 Plan

Next step is a small real-data collection with DJI M400 + H30 + Manifold 3:

- Collect 30-90 second clips, 3-5 clips per scene type.
- Run `fixed_wide`, `fixed_tele`, and `ufz_v2_2`.
- Log timestamp, zoom level, zoom command, focal length or zoom ratio, bbox, uncertainty, edge/association risk, lost count, and latency.
- Annotate with UAV123 `x,y,w,h` format for offline comparison.
- Follow the dedicated H30 collection protocol and log-field documentation in `docs/experiments/`.

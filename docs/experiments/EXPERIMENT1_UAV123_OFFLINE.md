# Experiment 1: UAV123@10fps Offline Simulated Zoom

This is the minimal runnable UFZ-Track Experiment 1 protocol for UAV123@10fps.

## Dataset Layout

The config expects the verified server paths:

- Dataset root: `/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps`
- Images: `/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps/data_seq/UAV123_10fps/{sequence_name}/`
- BBox GT: `/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps/anno/UAV123_10fps/{sequence_name}.txt`
- Attributes: `/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps/anno/UAV123_10fps/att/{sequence_name}.txt`

Only the bbox GT files are parsed as `x,y,w,h`. Attribute files are never used as boxes.
Frame files are read by filename order, e.g. `000001.jpg`, `000002.jpg`.

## Tracking Protocol

Frame 1:

- Read the first GT bbox.
- Initialize the Kalman tracker in original image coordinates.
- Log initial zoom level `1`.

Frame t >= 2:

1. Predict a bbox with the Kalman tracker in original image coordinates.
2. Crop and resize the current image with `ZoomSimulator` using the current zoom level.
3. Run YOLOv8 on the zoomed image.
4. Map detections back to original image coordinates.
5. Select the measurement by predicted-box agreement only.
6. Update the Kalman tracker if a detection is selected; otherwise keep predict-only state.
7. Compute confidence, innovation, jitter, target area, edge risk, and lost count.
8. Decide the next-frame zoom level with the chosen policy.
9. Save result bbox and per-frame log.

GT is not used for crop center, measurement selection, tracker update, or zoom decision after
initialization. GT is used again only by evaluation scripts.

## Methods

- `fixed_wide`: zoom level `1`
- `fixed_tele`: zoom level `4`
- `scale_only`: area-based zoom
- `confidence_only`: confidence-based zoom
- `ufz`: area + uncertainty + edge risk + cooldown/hysteresis

## Commands

```bash
python scripts/prepare_uav123.py --config configs/uav123_10fps.yaml --dry-run
python scripts/smoke_test_core.py
python scripts/run_uav123.py --config configs/uav123_10fps.yaml --method fixed_wide --sequences person1 --model yolov8n.pt
python scripts/run_uav123.py --config configs/uav123_10fps.yaml --method ufz --sequences person1 --model yolov8n.pt
python scripts/eval_uav123.py --config configs/uav123_10fps.yaml
python scripts/plot_curves.py --config configs/uav123_10fps.yaml --sequence person1 --method ufz
```

## Outputs

- Results: `/root/autodl-tmp/UFZTrack/outputs/results/{method}/{seq}.txt`
- Logs: `/root/autodl-tmp/UFZTrack/outputs/logs/{method}/{seq}.csv`
- Table: `/root/autodl-tmp/UFZTrack/outputs/tables/main_results.csv`
- Precision plot: `/root/autodl-tmp/UFZTrack/outputs/figures/precision_plot.pdf`
- Success plot: `/root/autodl-tmp/UFZTrack/outputs/figures/success_plot.pdf`
- Temporal plot: `/root/autodl-tmp/UFZTrack/outputs/figures/{seq}_temporal_curve.pdf`

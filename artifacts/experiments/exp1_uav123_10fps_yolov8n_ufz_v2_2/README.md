# UFZ-Track Experiment 1 Artifacts

This package contains GitHub-safe paper artifacts for Experiment 1 on UAV123@10fps with YOLOv8n and offline simulated zoom.

Contents:

- `tables/`: aggregate metrics, robust metrics, macro summaries, win counts, and selected case analysis.
- `figures/cases/`: selected PNG case plots only.
- `logs_selected/`: selected per-sequence CSV logs for `ufz`, `ufz_v2`, and `ufz_v2_2`.
- `summaries/`: the experiment summary markdown.

No raw UAV123 images, videos, model weights, detector outputs, or large result directories are included.

Protocol notes:

- UAV123 ground truth is used only for frame-1 tracker initialization and offline evaluation.
- Crop centers, measurement selection, tracker updates, and zoom decisions do not use future GT.
- The main frozen method is `ufz_v2_2`: UFZ v1 zoom behavior with hard-risk veto and lost recovery.
- Zoom is software-simulated offline using original-image coordinate output boxes.

Large archives, if created manually, should be stored under `artifacts/archives/` and handled with Git LFS or external release storage.

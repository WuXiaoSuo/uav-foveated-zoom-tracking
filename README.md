# UAV Foveated Zoom Tracking

This repository contains code, experiment configuration, documentation, and paper-ready artifacts for **UFZ-Track: Uncertainty-Guided Foveated Zoom for UAV Small Object Tracking**.

The project studies a bounded active perception setting: the UAV platform is not autonomously controlled, and the tracking system improves long-range small-object observability by regulating camera optical/software zoom.

## Core Pipeline

```text
video stream
  -> lightweight detector
  -> Kalman-based tracking
  -> uncertainty / target scale estimation
  -> bounded zoom policy
  -> optical or simulated foveated zoom regulation
```

## Repository Layout

```text
uav-foveated-zoom-tracking/
├── configs/              # experiment configuration
├── scripts/              # runnable experiment and analysis scripts
├── src/ufztrack/         # Python package code
├── docs/                 # experiment docs and paper-facing notes
├── tools/                # local helper scripts
├── artifacts/            # GitHub-safe small experiment artifacts
├── data/                 # local data placeholder, ignored by Git
├── outputs/              # local/server output placeholder, ignored by Git
├── server_outputs/       # downloaded server outputs, ignored by Git
└── weights/              # model weights, ignored by Git
```

Raw datasets, model weights, videos, full outputs, and downloaded server outputs are intentionally excluded from Git.

## Experiment 1

Experiment 1 implements UAV123@10fps offline simulated zoom tracking. The main frozen policy variant is `ufz_v2_2`, which keeps the original UFZ zoom proposal behavior and applies only hard-risk vetoes plus lost recovery.

Useful commands:

```bash
python scripts/smoke_test_core.py
python scripts/eval_uav123.py --config configs/uav123_10fps.yaml
python scripts/analyze_ufz_cases.py --output-root /root/autodl-tmp/UFZTrack/outputs --baseline ufz --candidate ufz_v2_2
python scripts/package_experiment_artifacts.py --output-root /root/autodl-tmp/UFZTrack/outputs
```

## Helper Scripts

The shell helpers in `tools/` identify the project root in this order:

1. `UFZTRACK_PROJECT_DIR`
2. `git rev-parse --show-toplevel`
3. `/root/autodl-tmp/UFZTrack/code/uav-foveated-zoom-tracking`

```bash
bash tools/start_ufztrack.sh
```

Checks the current UFZ-Track environment. On a local Mac without CUDA it reports a warning instead of failing.

```bash
bash tools/save_ufztrack.sh "commit message"
```

Stages and commits project code, configuration, docs, and tools while avoiding raw data, outputs, and weights.

```bash
bash tools/backup_ufztrack.sh
```

Backs up code, configuration, scripts, docs, and tools. The default backup location is `/tmp/ufztrack_backups/`; override it with `UFZTRACK_BACKUP_DIR`.

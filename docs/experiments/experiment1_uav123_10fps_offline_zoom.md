# Experiment 1: UAV123@10fps Offline Simulated Zoom

## 目标

实现 UFZ-Track 第一阶段最小可运行实验：在 UAV123 单目标追踪序列上，以 10fps 抽帧方式模拟 foveated/software zoom，并用 YOLOv8 生成 detection measurement，用 Kalman filter 维护单目标状态。

## 关键约束

- UAV123 是 single object tracking benchmark。
- 第一帧 GT bbox 只用于初始化 Kalman tracker。
- 第二帧及之后，GT 只能用于 evaluation。
- 第二帧及之后，GT 不能用于 crop center、measurement selection、tracker update 或 zoom decision。
- YOLOv8 只作为 detection measurement generator。
- tracker 状态由 Kalman filter 维护。
- software zoom levels 为 `[1, 2, 4, 8]`。
- 输出的 prediction bbox 和 measurement bbox 均为 original image coordinates。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

编辑配置：

```yaml
dataset:
  root: data/UAV123
  sequence: null
```

`root` 可以指向单个 UAV123 序列目录；如果设置 `sequence`，则使用 `root/sequence`。

运行 debug 版 YOLOv8n：

```bash
python scripts/run_experiment1_uav123_10fps.py --config configs/experiment1_uav123_10fps_yolov8n.yaml
```

## 输出

默认输出到：

```text
outputs/experiment1_yolov8n_debug/<sequence_name>/
```

主要文件：

- `tracks.csv`：每帧预测 bbox、measurement bbox、zoom level 和 Kalman uncertainty。所有 bbox 都是 original image coordinates。
- `summary.json`：mean IoU、success AUC、20px precision、mean center error。

## 当前不实现

- 不训练检测器。
- 不实现 YOLO11。
- 不实现 DFL uncertainty。
- 不使用后续 GT 做任何 tracking 或 zoom 决策。

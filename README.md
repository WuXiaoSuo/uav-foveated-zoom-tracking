# UAV Foveated Zoom Tracking

本仓库用于管理论文 **UFZ-Track: Uncertainty-Guided Foveated Zoom for UAV Small Object Tracking** 的代码、实验配置、论文文档和实验记录。

本项目研究一种受限主动感知问题：在 UAV 本体不参与自主控制、边缘算力受限的条件下，仅通过调节云台相机的 optical/foveated zoom，提高远距离小目标追踪的可观测性和稳定性。

## 研究目标

远距离 UAV 巡检、监视和搜寻任务中，目标通常只占据很少像素，导致检测置信度低、定位误差大、追踪易漂移。相比单纯增加 detector/tracker 复杂度，本项目关注：

```text
视频流
  -> 轻量检测器
  -> Kalman-based tracking
  -> tracking uncertainty / target scale estimation
  -> safe zoom policy
  -> optical/foveated zoom regulation

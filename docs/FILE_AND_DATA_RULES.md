# UFZ-Track File and Data Management Rules

本项目是 `uav-foveated-zoom-tracking`，请严格遵守以下文件与数据保存规范。任何代码生成、实验脚本、结果输出、文档记录都必须符合本规范。

---

# 1. 项目根目录

本地项目根目录固定为：

```bash
~/Documents/Codex/2026-06-12/uav/uav-foveated-zoom-tracking
```

服务器项目根目录固定为：

```bash
/root/autodl-tmp/UFZTrack/code/uav-foveated-zoom-tracking
```

严禁在父目录生成项目文件。

不要在下面这个父目录直接创建 `configs/`、`src/`、`scripts/`、`outputs/`：

```bash
~/Documents/Codex/2026-06-12/uav
```

所有项目文件只能放在：

```bash
~/Documents/Codex/2026-06-12/uav/uav-foveated-zoom-tracking
```

---

# 2. 标准目录结构

项目根目录应保持如下结构：

```text
uav-foveated-zoom-tracking/
├── configs/                 # 实验配置文件
├── scripts/                 # 可执行实验脚本
├── src/
│   └── ufztrack/             # 核心 Python 包
├── docs/
│   ├── experiments/          # 实验说明
│   │   └── runs/             # 每次实验记录 md
│   └── paper/                # 论文草稿、图表说明、写作材料
├── tools/                    # 同步、下载、清理等辅助脚本
├── server_outputs/           # 从服务器下载的实验结果，本地归档，不提交 Git
├── outputs/                  # 本地临时输出，不提交 Git
├── data/                     # 本地数据集，不提交 Git
├── weights/                  # 权重文件，不提交 Git
├── archive/                  # 临时归档旧文件，不提交 Git
├── README.md
└── requirements.txt
```

不要新建第二个 `uav-foveated-zoom-tracking` 仓库。
不要在根目录外创建 `src/`、`scripts/`、`configs/`。

---

# 3. 代码文件保存规则

核心代码只能放在：

```text
src/ufztrack/
```

实验入口脚本只能放在：

```text
scripts/
```

配置文件只能放在：

```text
configs/
```

实验文档只能放在：

```text
docs/experiments/
```

每次实验记录放在：

```text
docs/experiments/runs/
```

论文相关材料放在：

```text
docs/paper/
```

---

# 4. UAV123 数据集路径规则

服务器上的 UAV123@10fps 数据集路径固定为：

```text
/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps
```

其中：

```text
image_root:
/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps/data_seq/UAV123_10fps

bbox annotation root:
/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps/anno/UAV123_10fps

attribute root:
/root/autodl-tmp/UFZTrack/datasets/UAV123_10fps/anno/UAV123_10fps/att
```

注意：

```text
anno/UAV123_10fps/{sequence}.txt 是 bbox 标注，格式为 x,y,w,h。
anno/UAV123_10fps/att/{sequence}.txt 是属性标签，不能当成 bbox。
```

代码必须以 bbox annotation 是否存在作为序列有效性的判断依据。
如果某个序列只有图像目录但没有 bbox annotation，则不能参与当前 tracking evaluation。

---

# 5. 服务器输出路径规则

服务器所有实验输出固定写入：

```text
/root/autodl-tmp/UFZTrack/outputs
```

内部结构固定为：

```text
/root/autodl-tmp/UFZTrack/outputs/
├── results/
│   └── {method}/{sequence}.txt
├── logs/
│   └── {method}/{sequence}.csv
├── tables/
│   └── main_results.csv
└── figures/
    ├── precision_plot.pdf
    ├── success_plot.pdf
    └── {sequence}_temporal_curve.pdf
```

不要把服务器输出写到项目代码目录内部。
不要把服务器输出写到系统盘 `/root/` 下的其他临时目录。

---

# 6. 本地实验结果归档规则

从服务器下载的实验结果统一保存到：

```text
server_outputs/
```

命名格式：

```text
server_outputs/{experiment_name}_{dataset}_{model}_{date}/
```

推荐例子：

```text
server_outputs/exp1_uav123_10fps_yolov8n_20260616/
server_outputs/exp1_uav123_10fps_yolov8s_20260618/
server_outputs/exp1_valid10_yolov8n_20260617/
```

每个归档目录内部应保持服务器输出结构：

```text
server_outputs/exp1_uav123_10fps_yolov8n_20260616/
└── outputs/
    ├── results/
    ├── logs/
    ├── tables/
    └── figures/
```

---

# 7. 实验记录 md 命名规则

每次实验必须写一份实验记录，保存到：

```text
docs/experiments/runs/
```

文件命名格式：

```text
YYYYMMDD_exp{number}_{model}_{sequence_or_group}_{note}.md
```

例子：

```text
docs/experiments/runs/20260616_exp1_yolov8n_person1_bike1.md
docs/experiments/runs/20260617_exp1_yolov8n_valid10.md
docs/experiments/runs/20260618_exp1_yolov8s_valid10.md
```

实验记录必须包含：

```text
1. 日期
2. 数据集
3. 模型
4. 方法列表
5. 序列列表
6. 运行命令
7. 输出路径
8. 主要指标
9. zoom behavior 统计
10. 失败问题
11. 下一步计划
```

---

# 8. Git 提交规则

可以提交 Git 的内容：

```text
configs/
scripts/
src/
docs/
requirements.txt
README.md
.gitignore
```

禁止提交 Git 的内容：

```text
server_outputs/
outputs/
data/
weights/
archive/
*.pt
*.onnx
*.engine
*.mp4
*.avi
*.mov
*.zip
*.tar.gz
__pycache__/
*.pyc
.DS_Store
```

如果 `.gitignore` 缺失以上规则，请补充。

---

# 9. 实验输出文件格式

## result 文件

路径：

```text
outputs/results/{method}/{sequence}.txt
```

每行格式：

```text
x,y,w,h
```

坐标必须是 original image coordinates。

## log 文件

路径：

```text
outputs/logs/{method}/{sequence}.csv
```

必须包含列：

```text
frame,zoom_level,uncertainty,area,conf,lost,latency,command
```

## table 文件

路径：

```text
outputs/tables/main_results.csv
```

必须包含列：

```text
method,sequence,frames,mean_iou,success_auc,precision_20,mean_cle
```

## figure 文件

路径：

```text
outputs/figures/
```

推荐文件名：

```text
precision_plot.pdf
success_plot.pdf
{sequence}_temporal_curve.pdf
```

---

# 10. 实验命名规范

方法名固定为：

```text
fixed_wide
fixed_tele
scale_only
confidence_only
ufz
```

不要随意改名，例如不要写成：

```text
FixedWide
fixed-wide
wide
ours
UFZ
```

序列名必须与 UAV123 annotation 文件名完全一致。

模型名建议固定为：

```text
yolov8n.pt
yolov8s.pt
```

第一阶段只使用 YOLOv8n 做 debug，YOLOv8s 做主实验。
不要默认切换到 YOLO11 或其他 detector。

---

# 11. 服务器与本地同步规则

从本地同步代码到服务器时，只同步代码、配置、文档，不同步数据、输出、权重。

推荐命令：

```bash
rsync -avP \
  -e "ssh -p <PORT>" \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude ".DS_Store" \
  --exclude ".venv/" \
  --exclude "data/" \
  --exclude "outputs/" \
  --exclude "server_outputs/" \
  --exclude "weights/" \
  --exclude "archive/" \
  ./ \
  root@connect.nmb2.seetacloud.com:/root/autodl-tmp/UFZTrack/code/uav-foveated-zoom-tracking/
```

从服务器下载实验结果到本地时，只下载 outputs：

```bash
rsync -avP \
  -e "ssh -p <PORT>" \
  root@connect.nmb2.seetacloud.com:/root/autodl-tmp/UFZTrack/outputs/ \
  ./server_outputs/{run_name}/outputs/
```

其中 `<PORT>` 必须使用 AutoDL 页面当前显示的最新 SSH 端口。

---

# 12. 不允许的行为

严禁执行以下操作：

```text
1. 在父目录创建项目代码文件。
2. 把数据集放入 Git。
3. 把 outputs、server_outputs、weights 提交 Git。
4. 删除已有实验结果。
5. 删除已有文档记录。
6. 随意重命名 method、sequence、config。
7. 使用 GT 参与 crop center、detection filtering、tracking update 或 zoom decision。
8. 把 attribute 文件当成 bbox annotation。
9. 遇到缺失 sequence 时让整个 batch 崩溃，应该支持 warning + skip。
10. 未经确认执行 git pull、git push、git rebase、git reset、git clean。
```

---

# 13. 推荐新增功能

后续请优先实现：

```text
1. run_uav123.py 增加 --skip-missing。
2. 自动生成 valid sequence list。
3. 自动保存每次 run 的 config snapshot。
4. 自动保存 command line 到 logs。
5. 自动生成 experiment summary markdown。
```

---

# 14. 最终原则

本项目的文件管理原则是：

```text
代码归代码；
数据归数据；
服务器输出归 server_outputs；
实验记录归 docs/experiments/runs；
论文材料归 docs/paper；
不要让项目根目录再变乱。
```

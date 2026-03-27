
## 任务：初始化 YOLO 钢材缺陷检测项目框架

我正在做一个简历级的工业缺陷检测项目，需要你帮我搭建项目骨架。

### 执行方式

请一次性生成所有需要创建的文件，不要分步等我确认。我会在你全部生成后统一检查、跑通、修改。这不是一个渐进式学习项目，而是"AI 生成骨架 → 我理解和调优"的工作模式。

### 项目信息

- 项目名：yolo_defect
- GitHub：https://github.com/LiuSiChengGitHub/yolo_defect
- Conda 环境：yolo_defect
- 项目根目录：D:\01_Base\CodingSpace\yolo_defect（但代码里全部用相对路径，因为我有两台电脑，绝对路径不同）

### 数据集信息（重要！请仔细阅读）

NEU-DET 数据集仅 28MB，已下载到项目内，直接提交到 GitHub，确保 clone 即可复现。

**实际目录结构：**

```
data/NEU-DET/
├── train/                        # 训练集（每类约 240 张，编号 1-240）
│   ├── annotations/              #   VOC XML 标注（扁平目录）
│   │   ├── crazing_1.xml
│   │   ├── inclusion_1.xml
│   │   ├── ...
│   │   └── scratches_240.xml
│   └── images/                   #   JPG 图片（按类名分子目录）
│       ├── crazing/
│       │   ├── crazing_1.jpg ... crazing_240.jpg
│       ├── inclusion/
│       │   ├── inclusion_1.jpg ... inclusion_240.jpg
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/                   # 验证集（每类约 60 张，编号 241-300）
    ├── annotations/              #   同样结构
    │   ├── crazing_241.xml ... crazing_300.xml
    │   └── ...
    └── images/
        ├── crazing/
        │   ├── crazing_241.jpg ... crazing_300.jpg
        └── ... (同 train 结构)
```

**关键特征：**

- 数据集已经预先划分好 train/validation（约 80:20），不需要自己做随机划分
- 图片格式：JPG，200×200 像素
- 图片按类名分子目录存放：`images/{class_name}/{class_name}_N.jpg`
- 标注是扁平目录：`annotations/{class_name}_N.xml`
- 6 个类别：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- 注意 `rolled-in_scale` 包含连字符，文件名解析需特殊处理

### 项目背景

- 数据集：NEU-DET 钢材表面缺陷（6 类，每类 300 张，共 1800 张）
- 模型：YOLOv8n
- 目标：mAP@0.5 > 0.70，最终导出 ONNX + FastAPI 服务化
- 这个项目要放到 GitHub 上给面试官看，代码质量和项目结构很重要

### 请按以下结构创建目录和文件

标注 [已存在] 的不要创建或覆盖，只创建 [需创建] 的。

```
yolo_defect/                          [已存在]
├── README.md                         [需创建] 项目说明，英文为主，结构见下方
├── requirements.txt                  [需创建] 从 environment.yml 提取 pip 依赖
├── LICENSE                           [需创建] MIT
├── environment.yml                   [已存在] 不要动
├── .gitignore                        [已存在] 不要动
├── data/
│   ├── data.yaml                     [需创建] YOLO 数据集配置（路径用相对路径）
│   └── NEU-DET/                      [已存在] 原始数据集，不要动
│       ├── train/                    #   训练集（每类约240张）
│       │   ├── annotations/          #     VOC XML 标注（扁平）
│       │   └── images/               #     JPG 图片（按类名分子目录）
│       └── validation/               #   验证集（每类约60张）
│           ├── annotations/
│           └── images/
├── scripts/
│   ├── prepare_data.py               [需创建] VOC XML → YOLO TXT 格式转换
│   ├── data_analysis.py              [需创建] 数据分布分析 + 可视化
│   ├── train.py                      [需创建] 训练入口（读取 yaml 配置）
│   ├── evaluate.py                   [需创建] 评估 + PR曲线 + 混淆矩阵
│   ├── export_onnx.py                [需创建] ONNX 导出
│   └── inference_onnx.py             [需创建] ONNX Runtime 推理（单张+批量）
├── src/
│   ├── __init__.py                   [需创建]
│   └── detector.py                   [需创建] YOLODetector 类（ONNX 推理，FastAPI 复用）
├── api/
│   └── .gitkeep                      [需创建] Week 2 填充 FastAPI 服务
├── configs/
│   └── train_config.yaml             [需创建] 超参数配置（model/epochs/imgsz/lr 等）
├── models/
│   └── .gitkeep                      [需创建] 存放导出的 ONNX 模型
├── docs/
│   ├── tasks/                        [已存在] 项目 prompt 文档，不要动
│   ├── experiment_log.md             [需创建] 实验记录模板（对比表格）
│   └── assets/
│       └── .gitkeep                  [需创建] 存放 PR 曲线、Demo GIF 等
└── runs/
    └── .gitkeep                      [需创建] YOLO 训练输出（.gitignore 已忽略）
```

### README.md 要求

用英文+中文写两版，先英文版，后面跟中文版。

这份 README 同时承担三个角色：

- **GitHub 展示页**：面试官 10 秒内知道项目做了什么
- **学习笔记**：我自己查阅时能快速回忆关键概念和踩坑经验
- **AI 上下文**：未来新的 AI 对话读完 README 就能完全理解项目状态

包含以下章节（注意：带 📝 标记的章节需要包含学习笔记性质的内容，不只是命令和表格）：

1. **项目标题 + 一句话描述 + 技术栈 badges**（Python, PyTorch, YOLOv8, ONNX）
    
2. **Demo**（占位符 `<!-- TODO: Add demo GIF -->`）
    
3. **Highlights**（4-5 个项目亮点，emoji 列表）
    
4. **Quick Start**（clone → install → 推理 3 行命令，强调数据集已包含在仓库内无需额外下载）
    
5. 📝 **Dataset**
    
    - NEU-DET 简介 + 原始来源链接 + 6 类缺陷说明（英文名 + 中文名）
    - 数据集统计：总量、每类数量、图片尺寸、格式
    - 说明数据集已预划分 train（每类~240张）/ validation（每类~60张），已包含在 `data/NEU-DET/`
    - 原始数据目录结构说明（images 按类分子目录，annotations 扁平目录）
    - 标注格式说明：VOC XML 中 `<bndbox>` 包含 xmin/ymin/xmax/ymax
6. 📝 **Data Preparation**
    
    - 运行 `prepare_data.py` 的命令
    - 解释转换做了什么：VOC XML → YOLO TXT（class_id cx cy w h，归一化坐标）
    - 说明类别映射：crazing=0, inclusion=1, patches=2, pitted_surface=3, rolled-in_scale=4, scratches=5
    - 说明输出目录结构：`data/images/{train,val}/` + `data/labels/{train,val}/` + `data/data.yaml`
    - 注意事项：`rolled-in_scale` 类名包含连字符，文件名解析需用已知类名列表匹配
7. 📝 **Training**
    
    - 训练命令（通过 yaml 配置 + 通过 CLI）
    - 超参数表格（含每个参数的作用说明）
    - 简要说明 YOLOv8 训练流程：预训练权重加载 → 数据增强(mosaic/mixup) → 多尺度训练 → 自动保存 best/last
8. **Results**（占位，后续填真实数据）
    
    - 实验对比表格（imgsz / lr / epochs / mAP@0.5 / mAP@50-95 / 训练时间 / 备注）
    - PR 曲线占位图
    - 混淆矩阵占位图
    - 各类 AP 占位表
9. 📝 **ONNX Deployment**
    
    - 导出命令 + 推理命令
    - 解释为什么用 ONNX：跨平台、框架无关、推理加速
    - 性能对比表格占位（PyTorch vs ONNX 的 mAP / FPS / 模型大小）
    - `src/detector.py` 的 YOLODetector 类说明：preprocess → predict → draw 三步流程
    - 说明这个类后续会被 FastAPI 直接复用
10. **Project Structure**
    
    - 完整目录树 + 每个文件/目录的一句话说明
    - 设计原则说明：`scripts/` 放一次性脚本，`src/` 放可复用逻辑，`configs/` 分离超参数
11. **Tech Stack**（表格：工具 + 用途 + 版本）
    
12. 📝 **Key Design Decisions**（新增章节）
    
    - 为什么选 YOLOv8n 而不是 v5 或更大的模型（边缘部署友好、速度优先）
    - 为什么数据集放在项目内（仅 28MB，clone 即复现）
    - 为什么用 yaml 管理超参而不是命令行参数（实验可追溯、对比方便）
    - 为什么 `src/detector.py` 单独封装（FastAPI 复用、关注点分离）
13. **Roadmap / TODO**
    
    - 占位列表：SAM 集成、FastAPI 服务化、Docker 容器化、坏样本分析
14. **License**
    

风格要求：

- 面试官快速浏览时看标题、badges、Demo GIF、Highlights、Quick Start 就够了
- 我自己学习查阅时看 📝 章节的详细说明
- 新的 AI 对话读完整个 README 就能接手项目继续开发

### scripts/ 文件要求

通用要求：

- argparse 命令行参数（不硬编码路径）
- docstring 说明用途
- `if __name__ == "__main__":` 入口
- 代码注释用英文
- 脚本开头用 `os.path.dirname(os.path.abspath(__file__))` 定位脚本位置，再用 `os.path.join(script_dir, '..')` 得到项目根目录

---

**prepare_data.py**

功能：将 NEU-DET 原始数据（VOC XML）转换为 YOLO 格式。

输入数据路径（默认 `data/NEU-DET`，支持 `--data-root` 覆盖）：

- 训练图片：`{data-root}/train/images/{class_name}/{class_name}_N.jpg`
- 训练标注：`{data-root}/train/annotations/{class_name}_N.xml`
- 验证图片：`{data-root}/validation/images/{class_name}/{class_name}_N.jpg`
- 验证标注：`{data-root}/validation/annotations/{class_name}_N.xml`

输出（写到项目内 `data/` 目录）：

- `data/images/train/` — 训练集图片（从原始位置复制或软链接）
- `data/images/val/` — 验证集图片
- `data/labels/train/` — 训练集 YOLO TXT 标签
- `data/labels/val/` — 验证集 YOLO TXT 标签
- `data/data.yaml` — 自动生成

核心逻辑：

- 数据集已经预划分好 train/validation，不需要随机划分，直接分别处理两个子集
- 遍历 annotations/ 目录下的 XML 文件
- 从 XML 文件名提取类别：用已知类名列表 `['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']` 做前缀匹配（注意 `rolled-in_scale` 包含连字符，不能简单按下划线分割）
- 解析 XML 中 `<bndbox>` 的 `<xmin>`, `<ymin>`, `<xmax>`, `<ymax>`
- 转换为 YOLO 格式：`class_id cx cy w h`（归一化到 0-1）
- 一张图可能有多个 bbox，每个 bbox 一行
- 类别映射：crazing=0, inclusion=1, patches=2, pitted_surface=3, rolled-in_scale=4, scratches=5
- 通过 XML 文件名找到对应的图片：`{class_name}/{xml_stem}.jpg`
- 打印统计：每类在 train/val 中各多少张
- 运行：`python scripts/prepare_data.py`

---

**data_analysis.py**

- 分析 `data/NEU-DET/` 原始数据（支持 `--data-root`）
- 遍历 train/annotations/ 和 validation/annotations/
- 统计每类样本数（train + val），画柱状图
- 统计每张图的 bbox 数量分布
- 统计 bbox 宽高分布，画散点图
- 统计图片尺寸
- 保存图到 `docs/assets/`
- 打印文字摘要

---

**train.py**

- 从 `configs/train_config.yaml` 读取超参数
- 调用 ultralytics YOLO API 训练
- 支持 `--config` 指定配置文件
- 输出到 `runs/`

---

**evaluate.py**

- 加载训练好的模型（`--weights` 参数）
- 在验证集上评估
- 生成 PR 曲线、混淆矩阵
- 保存误检案例（confidence 最低的 N 张）
- 图保存到 `docs/assets/`

---

**export_onnx.py**

- 加载 best.pt（`--weights` 参数）
- 导出 ONNX（simplify=True）
- 保存到 `models/best.onnx`
- 打印模型大小

---

**inference_onnx.py**

- 调用 `src/detector.py` 的 YOLODetector
- 支持单张推理（`--image`）和文件夹批量推理（`--image-dir`）
- 画框 + 类名 + confidence
- 保存到 `--output-dir`
- 打印 FPS

---

### configs/train_config.yaml

```yaml
# YOLOv8 Training Configuration
# Modify this file for different experiments

model: yolov8n.pt          # Model: yolov8n/s/m/l/x
data: data/data.yaml        # Dataset config
epochs: 50                  # Total training epochs
imgsz: 640                  # Input image size
batch: 16                   # Batch size (-1 for auto)
lr0: 0.01                   # Initial learning rate
optimizer: auto             # Optimizer: SGD, Adam, AdamW, auto
mosaic: 1.0                 # Mosaic augmentation (0.0 to disable)
mixup: 0.0                  # Mixup augmentation
device: 0                   # CUDA device (0 for first GPU)
workers: 8                  # Dataloader workers
project: runs/detect        # Save directory
name: train                 # Experiment name
```

---

### src/detector.py

封装 `YOLODetector` 类：

- `__init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45)` — 加载 ONNX 模型
- `preprocess(image)` — BGR→RGB, resize, normalize(0-1), HWC→CHW, add batch dim
- `predict(image)` — ONNX 推理 → NMS → 返回 detections list
- `draw(image, detections, class_names)` — 画框 + 类名 + confidence
- 后续 FastAPI 直接 `from src.detector import YOLODetector`

---

### docs/experiment_log.md

实验记录模板：

- 实验目的
- 变量说明
- 训练结果对比表格（实验名 / imgsz / lr0 / epochs / mosaic / mAP@0.5 / mAP@50-95 / 训练时间 / 备注）
- 部署性能对比表格（格式 / mAP@0.5 / FPS(CPU) / FPS(GPU) / 模型大小）
- 误检分析区域（留空）
- 结论 + 下一步

---

### 注意事项

- 不要生成任何训练数据或模型权重
- `data/NEU-DET/` 已有完整数据，不要动
- `docs/tasks/` 已有文件，不要动
- `.gitignore` 和 `environment.yml` 已存在，不要动
- 所有路径使用相对路径，绝对不要硬编码 `D:\` 开头的路径
- `.gitkeep` 用于保留空目录
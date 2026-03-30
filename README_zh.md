# 钢材表面缺陷检测 — 基于 YOLOv8

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange?logo=onnx)
![License](https://img.shields.io/badge/License-MIT-yellow)

端到端工业缺陷检测流水线：从数据准备到 ONNX 部署，基于 NEU-DET 钢材表面数据集与 YOLOv8 构建。

![推理演示](docs/assets/demo_inference_result.gif)

## 项目亮点

- **当前最佳实验结果** — 当前最佳模型 `final_train_2` 达到 **mAP@0.5 = 0.743**
- **PyTorch / ONNX 一致性抽查** — 50 张图里有 **48/50** 的检测框数量完全一致，总检测框数 **147 vs 147**
- **CPU 推理速度有原始结果** — PyTorch CPU 基准测试在 **100** 张计时图片上达到 **8.43 FPS**
- **Docker 已验证** — `python:3.9-slim` 镜像已成功跑通 `/health` 和 `/detect`
- **克隆即用** — 数据集（28MB）已包含在仓库内，无需额外下载

## 关键指标

| 指标 | 当前结果 |
|------|----------|
| 最佳模型 | `final_train_2` |
| mAP@0.5 | **0.743** |
| mAP@50-95 | **0.388** |
| PT/ONNX 检测框数一致率 | **48 / 50**（**96%**） |
| 平均检测框数差值 | **0.04** |
| PyTorch CPU 基准测试 | **8.43 FPS** / **118.66 ms** 每张 |
| 模型大小（`best.pt` / `best.onnx`） | ~6.0 MiB / ~11.8 MiB |

## 快速开始

```bash
# 克隆（数据集已包含，约 28MB）
git clone https://github.com/LiuSiChengGitHub/yolo_defect.git
cd yolo_defect

# 安装依赖
conda env create -f environment.yml
conda activate yolo_defect

# 数据准备（VOC XML -> YOLO TXT）
python scripts/prepare_data.py

# 训练
python scripts/train.py

# 从默认训练输出导出 ONNX
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt

# 用真实验证集样例做推理
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg
```

## 数据集

### NEU-DET：东北大学钢材表面缺陷数据库

**来源：** [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)

NEU-DET 数据集由东北大学宋克臣教授团队发布，包含 1,800 张热轧钢带表面灰度图像，是工业缺陷检测领域最常用的公开基准数据集之一。涵盖 6 类典型表面缺陷：

| 类别 ID | 英文名 | 中文名 | 描述 | 检测难度 |
|---------|--------|--------|------|----------|
| 0 | crazing | 龟裂 | 表面细密裂纹网络 | 高（纹理细密，与背景区分度低） |
| 1 | inclusion | 夹杂 | 钢材内嵌入的异物 | 中 |
| 2 | patches | 斑块 | 不规则变色区域 | 中 |
| 3 | pitted_surface | 麻面 | 表面分布的小凹坑 | 中 |
| 4 | rolled-in_scale | 压入氧化铁皮 | 轧制过程中压入表面的氧化皮 | 中 |
| 5 | scratches | 划痕 | 机械接触产生的线性痕迹 | 低（线性特征明显） |

### 数据统计

- **论文 / 官方描述中的总数：** 1,800（每类 300 张）
- **`data/NEU-DET/` 中实际可读取的 JPG 数：** 1,800
- **图片尺寸：** 200 x 200 像素
- **格式：** JPG（标注中标记为 depth=1 灰度，但实际可以作为 3 通道 BGR 读取）
- **`data/images/` 中已生成的 YOLO 图片划分：** 训练集 1,439 张，验证集 361 张
- **数据集已包含在 `data/NEU-DET/` 目录中**

### 原始数据目录结构

```
data/NEU-DET/
├── train/                         # 训练集（可读取图片 1,439 张）
│   ├── annotations/               # VOC XML 标注（扁平目录，所有类混在一起）
│   │   ├── crazing_1.xml          #   文件名格式：{类名}_{编号}.xml
│   │   ├── inclusion_1.xml
│   │   ├── rolled-in_scale_1.xml  #   注意：类名含连字符！
│   │   └── ...
│   └── images/                    # JPG 图片（按类名分子目录）
│       ├── crazing/               #   文件名格式：{类名}/{类名}_{编号}.jpg
│       │   ├── crazing_1.jpg
│       │   └── ...
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/                    # 验证集（XML 361 个，可读取图片 361 张），结构同 train
    ├── annotations/
    └── images/
```

> **注意设计上的不对称：** annotations 是扁平目录（所有类的 XML 混在一起），而 images 按类名分子目录。这种不对称在数据准备脚本中需要特殊处理。

### 标注格式说明

原始数据使用 **VOC XML** 格式（源自 Pascal VOC 目标检测挑战赛）。每张图对应一个 XML 文件：

```xml
<annotation>
    <size>
        <width>200</width>            <!-- 图片宽度 -->
        <height>200</height>           <!-- 图片高度 -->
        <depth>1</depth>               <!-- 通道数（灰度=1） -->
    </size>
    <object>                           <!-- 一个标注目标（可以有多个 object） -->
        <name>crazing</name>           <!-- 类别名称 -->
        <bndbox>
            <xmin>2</xmin>             <!-- 左上角 x 坐标（像素） -->
            <ymin>2</ymin>             <!-- 左上角 y 坐标（像素） -->
            <xmax>193</xmax>           <!-- 右下角 x 坐标（像素） -->
            <ymax>194</ymax>           <!-- 右下角 y 坐标（像素） -->
        </bndbox>
    </object>
    <!-- 一张图可能有多个 <object>，如 rolled-in_scale 常有 2-3 个 bbox -->
</annotation>
```

> **格式说明：** VOC 格式用绝对像素坐标的角点表示 (xmin, ymin, xmax, ymax)，而 YOLO 格式用归一化的中心点+宽高 (cx, cy, w, h)。这是从标注文件转换到训练输入时最核心的差异。

## 数据准备

### 转换说明

`prepare_data.py` 将 VOC XML 标注转换为 YOLO TXT 格式（Ultralytics YOLOv8 要求的输入格式）。

**VOC 格式**（绝对像素坐标，角点表示）：
```
xmin, ymin, xmax, ymax  →  例如: 2, 2, 193, 194
```

**YOLO 格式**（归一化中心坐标）：
```
class_id cx cy w h  →  例如: 0 0.487500 0.490000 0.955000 0.960000
```

**归一化转换公式：**
```
cx = (xmin + xmax) / 2 / image_width    # 中心点 x，归一化到 0-1
cy = (ymin + ymax) / 2 / image_height   # 中心点 y，归一化到 0-1
w  = (xmax - xmin) / image_width        # 宽度，归一化到 0-1
h  = (ymax - ymin) / image_height       # 高度，归一化到 0-1
```

> **为什么要归一化？** 归一化后坐标与图片分辨率无关。训练时 YOLO 会把 200x200 的原图 resize 到 640x640，归一化坐标会自动适配，不用手动调整标签值。

### 类别映射

| 类别名称 | 类别 ID | 说明 |
|----------|---------|------|
| crazing | 0 | 顺序固定，与 data.yaml 中的 names 对应 |
| inclusion | 1 | |
| patches | 2 | |
| pitted_surface | 3 | |
| rolled-in_scale | 4 | 注意：名字含连字符，不能用下划线分割提取类名 |
| scratches | 5 | |

### 运行

```bash
python scripts/prepare_data.py
# 或指定自定义路径：
python scripts/prepare_data.py --data-root data/NEU-DET --output-dir data
```

### 输出目录结构

```
data/
├── images/
│   ├── train/          # 扁平目录，所有训练图片（从子目录复制过来）
│   └── val/            # 扁平目录，所有验证图片
├── labels/
│   ├── train/          # YOLO TXT 标签（每张图一个 .txt，与图片同名）
│   └── val/
└── data.yaml           # YOLO 数据集配置文件
```

> **YOLO 对目录的要求：** `images/` 和 `labels/` 必须平级，且文件一一对应（`crazing_1.jpg` ↔ `crazing_1.txt`）。所以必须把按类名分的图片"拍平"到一个目录里。

### 踩坑注意事项

- 数据集**已经预划分**好训练集/验证集，不需要也不应该自己做随机划分
- `rolled-in_scale` 类名包含连字符 `-`，如果用 `filename.split('_')[0]` 提取类名会得到 `rolled-in`（错误！）。正确做法是用已知类名列表做前缀匹配，按长度从长到短排序确保最长匹配优先
- 图片必须从按类名分的子目录复制到扁平输出目录（YOLO 格式的硬性要求）
- 如果你手动更新了原始数据集，请重新执行 `prepare_data.py`，让 `data/images/` 和 `data/labels/` 与 `data/NEU-DET/` 保持一致

## 数据分析

对转换后的数据集运行 `data_analysis.py` 可得出以下结论：6 个类别整体上仍然分布均衡，无需过采样或类别加权。所有图片均为 200×200 px。每张图的 bbox 数量在 1 至 9 个之间（均值 2.33），目标密度适中。Bbox 尺寸差异极大——从 8×9 px 的细长划痕到近 199×199 px 的大面积裂纹——是一个多尺度检测的挑战性场景。YOLOv8 的 anchor-free 设计无需手动设置 anchor，天然适合处理这种宽泛的尺寸分布。分析图表已保存至 `docs/assets/`。

```bash
python scripts/data_analysis.py
```

## 训练

### 运行训练

```bash
# 方式一：通过 YAML 配置文件（推荐，实验可追溯）
python scripts/train.py --config configs/train_config.yaml

# 方式二：通过 Ultralytics CLI（快速实验）
yolo detect train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 超参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `yolov8n.pt` | 模型变体。`n`=nano（最快），`s`/`m`/`l`/`x` 依次增大 |
| `data` | `data/data.yaml` | 数据集配置文件，定义路径和类名 |
| `epochs` | 50 | 总训练轮数。太少欠拟合，太多过拟合 |
| `imgsz` | 640 | 输入图片尺寸。原图 200x200 会被 resize 到 640x640 |
| `batch` | 16 | 批大小。更大 = 梯度更稳定，但需要更多显存 |
| `lr0` | 0.01 | 初始学习率。训练过程中会按 schedule 自动衰减 |
| `optimizer` | `auto` | 优化器。auto 会根据模型自动选择 SGD 或 AdamW |
| `mosaic` | 1.0 | Mosaic 数据增强概率。4 张图拼成 1 张 |
| `mixup` | 0.0 | Mixup 数据增强概率。两张图按比例混合 |
| `device` | 0 | CUDA 设备编号。`cpu` 则用 CPU 训练 |
| `workers` | 8 | 数据加载的工作进程数 |

### 训练流程详解

1. **加载预训练权重（迁移学习）**
   - YOLOv8n 使用在 COCO 数据集（80 类、33 万张图）上预训练的权重
   - 骨干网络（Backbone）已经学会了通用的特征提取能力（边缘、纹理、形状等）
   - 我们只需要在 NEU-DET 上微调（Fine-tune），让模型学习钢材缺陷的特定特征

2. **数据增强（在线增强，不额外占磁盘）**
   - **Mosaic**：把 4 张图拼成一张，每张占一个象限。好处是一次看到更多目标，提升小目标检测
   - **Mixup**：两张图按随机比例混合叠加，起正则化效果
   - **随机翻转**：水平/垂直翻转，增加数据多样性
   - **HSV 调整**：随机调整色相、饱和度、亮度，增强对光照变化的鲁棒性
   - **尺度抖动**：随机缩放输入图片，让模型适应不同大小的目标

3. **多尺度训练**
   - 训练时随机调整输入尺寸（如 480-640-800），让模型在不同分辨率下都能检测
   - 推理时固定为设定的 imgsz

4. **自动保存检查点**
   - `best.pt`：验证集上 mAP 最高的那个 epoch 的权重（用于最终评估和部署）
   - `last.pt`：最后一个 epoch 的权重（用于断点续训）
   - 保存路径：`runs/detect/train/weights/`

## 实验结果

### 实验对比

| 实验 | 模型 | imgsz | lr0 | epochs | mAP@0.5 | mAP@50-95 | 训练时间 | 备注 |
|------|------|-------|-----|--------|---------|-----------|----------|------|
| baseline | yolov8n | 640 | 0.01 | 50 | **0.734** | 0.390 | 9.4 分钟 | 默认配置，已超过 0.70 目标 |
| exp1 | yolov8n | 512 | 0.01* | 50 | 0.733 | 0.391 | 7.2 分钟 | 更快，但 hardest classes 明显下降 |
| exp2 | yolov8n | 800 | 0.01* | 50 | 0.742 | 0.385 | 13.4 分钟 | `optimizer=auto` 家族里最好的图片尺寸结果 |
| exp3_lr01 | yolov8n (SGD) | 640 | 0.01 | 50 | 0.736 | **0.395** | 9.0 分钟 | 固定 SGD 后最好的严格指标结果 |
| exp4 | yolov8n | 800 | 0.01* | 50 | 0.741 | 0.384 | 13.6 分钟 | `mixup=0.1` 没有带来提升 |
| exp5 | yolov8n | 800 | 0.01* | 50 | 0.740 | 0.387 | 13.3 分钟 | 去样本混合增强对照组 |
| final_train | yolov8n | 800 | 0.01* | 100 | 0.729 | 0.379 | 26.1 分钟 | 单纯拉长训练并没有变更优 |
| final_train_2 | yolov8n (SGD) | 800 | 0.01 | 100 | **0.743** | 0.388 | 25.9 分钟 | 手动组合最优候选后的当前最佳 `mAP@0.5` |

\* 本次训练中 `optimizer=auto` 自动选择了 `AdamW(lr=0.001)`，所以 `lr0=0.01` 不是实际生效学习率。

### 当前模型候选

- **`final_train_2`**：如果主打 `mAP@0.5`，这是当前最适合作为部署主模型的 checkpoint
- **`exp3_lr01`**：如果想强调更严格的 `mAP@50-95` 和更干净的 lr 对照设计，它仍然很重要
- **`final_train`**：它证明了一个关键点，单纯增加 epoch 并不会自动得到更好的最终模型

### 各类 AP（当前最佳：`final_train_2`）

| 类别 | AP@0.5 | Precision | Recall |
|------|--------|-----------|--------|
| patches | 0.920 | 0.856 | 0.850 |
| inclusion | 0.827 | 0.773 | 0.742 |
| pitted_surface | 0.807 | 0.821 | 0.701 |
| scratches | 0.803 | 0.602 | 0.843 |
| rolled-in_scale | 0.553 | 0.507 | 0.462 |
| crazing | 0.550 | 0.513 | 0.543 |

### 对比结论

- `imgsz=800` 对整体 `mAP@0.5` 方向是有帮助的，但它本身并没有解决 `crazing`
- 固定 SGD 的学习率对比说明，在当前项目里 `lr0=0.01` 明显优于 `0.001`
- `mixup=0.1` 不适合这个依赖细纹理的工业缺陷任务，去掉样本混合增强更稳
- 手动组合出的 `final_train_2` 成为了当前 `mAP@0.5` 最好的模型，并把 `crazing` 提升到了 `0.550`
- 最实用的经验不是"训练更久一定更好"，而是"要先把参数组合设计对，再给更长训练预算"

### 训练曲线

![Training Results](docs/assets/results_final_train_2.png)

### PR 曲线（当前最佳）

![PR Curve](docs/assets/PR_curve_final_train_2.png)

### 混淆矩阵（当前最佳）

![Confusion Matrix](docs/assets/confusion_matrix_final_train_2.png)

### 预测样例

![Validation Predictions](docs/assets/val_pred_sample_final_train_2.jpg)

## ONNX 部署

### 为什么选择 ONNX？

ONNX（Open Neural Network Exchange）是微软和 Facebook 联合推出的开放神经网络格式：

- **跨平台** — 无需安装 PyTorch，Windows/Linux/macOS/边缘设备均可运行
- **框架无关** — 推理时不依赖训练框架，部署环境只需要轻量的 ONNX Runtime
- **性能优化** — ONNX Runtime 提供硬件加速（CUDA, TensorRT, DirectML），推理速度通常优于原生 PyTorch
- **体积更小** — 不用打包整个 PyTorch 运行时，部署镜像更小

> **部署取舍：** 不直接使用 PyTorch 进行交付，主要是因为 ONNX Runtime 体积更小、依赖更轻，也更适合跨平台部署和边缘场景。

### 导出命令

```bash
# Quick Start 默认导出路径：来自 `scripts/train.py` 的默认训练输出
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
# 输出: models/best.onnx
```

如果希望复现 README 中展示的最佳指标，请改为导出最佳实验 checkpoint：

```bash
python scripts/export_onnx.py --weights runs/detect/final_train_2/weights/best.pt --imgsz 800
```

### 推理命令

```bash
# 单张推理
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg

# 批量推理（整个目录）
python scripts/inference_onnx.py --model models/best.onnx --image-dir data/images/val --output-dir results/
```

当前导出的 ONNX 模型使用 `imgsz=800`，因此模型输入是 `[1, 3, 800, 800]`，原始输出张量是 `[1, 10, 13125]`（`4 个框参数 + 6 个类别分数`，覆盖全部候选位置）。

### 性能对比

| 检查项 | 数值 | 证据来源 |
|--------|------|----------|
| 最佳 PyTorch 验证结果 | **mAP@0.5 = 0.7433**，**mAP@50-95 = 0.3880** | `docs/experiment_log.md` |
| PyTorch CPU 基准测试 | **8.43 FPS**，**118.66 ms/张**，共 **100** 张计时图片 | `results/pytorch_benchmark_100.json` |
| PT / ONNX 检测框数一致率 | **48 / 50**（**96%**） | `results/pt_onnx_compare/compare_50_summary.json` |
| PT / ONNX 总检测框数 | **147 vs 147** | `results/pt_onnx_compare/compare_50_summary.json` |
| 平均绝对检测框数差值 | **0.04** | `results/pt_onnx_compare/compare_50_summary.json` |
| 当前本地模型大小 | `best.pt = 6,286,072 bytes`，`best.onnx = 12,336,935 bytes` | 本地模型文件 |

- 当前仓库已经提交了 PyTorch 验证结果摘要、PyTorch CPU benchmark 和 50 张 PT / ONNX 对比结果。
- GPU benchmark 和 API benchmark 的原始日志暂未随仓库提交，因此 README 里不再保留没有原始证据文件支撑的 GPU / QPS 数字。

### YOLODetector 类（`src/detector.py`）

`src/detector.py` 封装了完整的 ONNX 推理流程，三步 API 设计：

1. **`preprocess(image)`** — 图片预处理
   - BGR → RGB（OpenCV 读的是 BGR，模型期望 RGB）
   - Resize 到模型输入尺寸（本项目当前为 `800x800`）
   - 像素值归一化到 0-1（除以 255）
   - HWC → CHW（维度重排，PyTorch/ONNX 的标准）
   - 添加 batch 维度（3维→4维）

2. **`predict(image)`** — 模型推理 + 后处理
   - ONNX Runtime 前向推理
   - 解析输出张量（当前项目 `imgsz=800` 时输出形状 `[1, 10, 13125]`）
   - 置信度过滤（默认 > 0.25）
   - **NMS（非极大值抑制）**：同一目标可能被多个框检测到，NMS 只保留最优框

3. **`draw(image, detections, class_names)`** — 结果可视化
   - 画边界框 + 类名 + 置信度分数

> **NMS 核心流程：**
> 1. 按置信度从高到低排序所有检测框
> 2. 取最高分的框，与其余框逐一计算 IoU（交并比）
> 3. IoU > 阈值的框被抑制（认为检测的是同一个目标）
> 4. 重复直到处理完所有框
>
> 本项目在 `detector.py` 中手动实现了 NMS（不依赖 torchvision）。

该类的设计目的是**复用**：`scripts/inference_onnx.py` 和未来的 FastAPI 服务都直接 `from src.detector import YOLODetector`，推理逻辑只写一份。

另外，`scripts/debug_detector.py` 用于手动展开预处理与 ONNX 前向过程，并打印 5 个关键 shape，适合排查预处理问题。

### FastAPI API 使用

项目现在已经包含一个最小可用的 FastAPI 服务，入口是 `api/app.py`，目前提供两个接口：

- `GET /health`：健康检查，用来确认服务是否启动、模型是否加载成功
- `POST /detect`：上传单张图片，返回检测结果 JSON

启动服务：

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

健康检查示例：

```bash
curl http://127.0.0.1:8000/health
```

示例响应：

```json
{
  "status": "ok",
  "model": "best.onnx",
  "request_stats": {
    "total_requests": 0,
    "avg_response_time_ms": 0.0
  }
}
```

检测请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@data/images/val/crazing_241.jpg"
```

示例响应：

```json
{
  "filename": "crazing_241.jpg",
  "count": 3,
  "image_size": {
    "width": 200,
    "height": 200
  },
  "model": "best.onnx",
  "conf_thresh": 0.25,
  "iou_thresh": 0.45,
  "inference_time_ms": 20.57,
  "detections": [
    {
      "class_id": 0,
      "class_name": "crazing",
      "confidence": 0.4457,
      "bbox": [-1.34, 53.68, 176.91, 146.24]
    }
  ]
}
```

使用说明：

- 上传字段名必须是 `file`
- 返回结果是 JSON，不是画框后的图片
- `inference_time_ms` 是服务端模型推理时间，并发场景下客户端总等待时间通常会更长
- 可以用 `scripts/benchmark_api.py` 对 `POST /detect` 做简单压测，统计平均响应时间和 QPS

本地实测结果：

- `GET /health` 已返回 `200 OK`，且 `status=ok`
- `POST /detect` 对 `data/images/val/crazing_241.jpg` 返回 `count=3`
- `scripts/benchmark_api.py` 已提供本地并发压测入口，但原始 benchmark 日志暂未随仓库提交，因此这里不再写 QPS 数字

### Docker 容器化部署

项目当前已经提供最小可用的 Docker 部署方案：

- 基础镜像：`python:3.9-slim`
- 依赖文件：`requirements-api.txt`
- 复制内容：`src/`、`api/`、`models/`
- 暴露端口：`8000`

构建与启动：

```bash
docker build -t yolo-defect-api .
docker run --rm -p 8000:8000 yolo-defect-api
```

快速验证：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/detect \
  -F file=@data/images/val/crazing_241.jpg
```

当前 Docker 本地实测：

- `GET /health` 返回 `status=ok`、`model=best.onnx`
- `POST /detect` 对 `crazing_241.jpg` 返回 `count=3`
- 目前只保留接口级验证结论；Docker benchmark 原始日志暂未随仓库提交

## 项目结构

```
yolo_defect/
├── Dockerfile                    # FastAPI 部署镜像
├── README.md                     # 项目说明（英文版）
├── README_zh.md                  # 项目说明（中文版）
├── LICENSE                       # MIT 开源协议
├── requirements-api.txt          # Docker / API 运行时最小依赖
├── requirements.txt              # pip 依赖列表
├── environment.yml               # Conda 环境配置（含 PyTorch + CUDA）
├── .gitignore                    # Git 忽略规则
├── data/
│   ├── data.yaml                 # YOLO 数据集配置（prepare_data.py 自动生成）
│   └── NEU-DET/                  # 原始数据集（28MB，提交到 git）
│       ├── train/                #   训练集 (~240张/类)
│       └── validation/           #   验证集 (~60张/类)
├── scripts/                      # 一次性脚本（命令行运行）
│   ├── prepare_data.py           #   VOC XML → YOLO TXT 格式转换
│   ├── data_analysis.py          #   数据集统计与可视化
│   ├── train.py                  #   训练入口（读取 YAML 配置）
│   ├── evaluate.py               #   模型评估 + PR 曲线 + 混淆矩阵
│   ├── export_onnx.py            #   ONNX 模型导出
│   ├── debug_detector.py         #   中间值打印 / ONNX 输出观察
│   ├── compare_pt_onnx.py        #   PyTorch vs ONNX 50张近似对比
│   ├── benchmark_pytorch.py      #   PyTorch 100张 CPU FPS 测试
│   ├── benchmark_api.py          #   POST /detect 并发压测脚本
│   ├── analyze_failures.py       #   误检 / 漏检案例分析
│   ├── select_representative_examples.py # README 代表样本筛选
│   └── inference_onnx.py         #   ONNX 推理（单张 + 批量）
├── src/                          # 可复用模块
│   ├── __init__.py
│   └── detector.py               #   YOLODetector 类（ONNX 推理，FastAPI 复用）
├── api/                          # FastAPI 服务
│   └── app.py                    #   `GET /health` + `POST /detect`
├── configs/
│   ├── train_config.yaml         # baseline 训练超参数配置
│   └── exp*.yaml                 # 各组实验配置（imgsz / lr / augment / final）
├── models/
│   └── .gitkeep                  # 导出的 ONNX 模型（gitignored）
├── docs/
│   ├── experiment_log.md         # 实验记录模板
│   └── assets/                   # PR 曲线、Demo GIF、分析图表
└── runs/                         # YOLO 训练输出（gitignored）
```

### 设计原则

- **`scripts/`**：一次性脚本，用 argparse 接收参数，从命令行运行。每个脚本独立，做一件事。
- **`src/`**：可复用模块。`detector.py` 同时被推理脚本和 FastAPI 服务 import，避免代码重复。
- **`configs/`**：超参数与代码分离。调参时改配置文件，不用改代码。用 git diff 可以对比两次实验的参数差异。

## 技术栈

| 工具 | 用途 | 版本 |
|------|------|------|
| Python | 编程语言 | 3.9 |
| PyTorch | 深度学习框架 | 2.0.0 |
| Ultralytics | YOLOv8 训练和推理 | latest |
| ONNX | 开放神经网络格式 | latest |
| ONNX Runtime | 优化推理引擎 | latest (GPU) |
| OpenCV | 图像处理 | (via ultralytics) |
| Matplotlib | 可视化绘图 | (via ultralytics) |
| FastAPI | REST API 服务 | latest |
| Conda | 环境管理 | — |

## 关键设计决策

### 模型选择

- **YOLOv8 vs YOLOv5：** YOLOv8 是最新一代，架构改进包括 C2f 模块（替代 C3）、Anchor-Free 检测头（不需要预定义锚框）、解耦头（分类和回归分开处理）。同等大小下 YOLOv8 精度更高。
- **为什么 nano (n) 版本：** NEU-DET 只有 1,800 张图，数据集很小。用更大的模型（s/m/l）容易过拟合，且推理速度慢。nano 版本仅 3.2M 参数，在边缘设备上也能实时运行。
- **灵活升级：** 如果 nano 精度不够，改一行配置就能换成 s 或 m，无需改代码。

### 数据集收录

NEU-DET 数据集只有 28MB（远小于 GitHub 的 100MB 单文件限制）。放在仓库里意味着：
- `git clone` 后立刻可以跑，不需要手动下载、注册账号、解压
- 保证完全可复现——每个人用的是完全相同的数据
- 对外验证成本低——几分钟内就能验证结果

### 配置管理

- **可追溯**：每次实验的配置是一个文件，可以 git commit 保存
- **可对比**：用 `diff exp1.yaml exp2.yaml` 直接看两次实验改了什么
- **可复现**：`python train.py --config exp1.yaml` 就能精确重现实验

### 推理模块

- **关注点分离**：推理逻辑不依赖 ultralytics 或 PyTorch，只依赖 ONNX Runtime
- **代码复用**：推理脚本和 FastAPI 服务共用同一份推理代码
- **可测试性**：可以对 detector 类单独写单元测试，不用启动整个训练框架

## 路线图

- [x] 基线训练与实验记录
- [x] 超参数调优（imgsz / lr / augment 对比）
- [x] 坏样本分析（误检/漏检案例）
- [x] ONNX 导出与 CPU 推理验证
- [x] ONNX 精度对齐（PyTorch vs ONNX）
- [x] FastAPI 服务化（`POST /detect` 上传图片返回 JSON）
- [x] Docker 容器化部署
- [x] Demo GIF 推理演示
- [ ] TensorRT / C++ ONNX Runtime 优化（V2）
- [ ] CI/CD 流程与自动化测试

## 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

NEU-DET 数据集由东北大学提供，学术引用请参考：

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.

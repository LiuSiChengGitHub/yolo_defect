# Steel Surface Defect Detection with YOLOv8

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange?logo=onnx)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-end industrial defect detection pipeline: from data preparation to ONNX deployment, built on the NEU-DET steel surface dataset with YOLOv8.

<!-- TODO: Add demo GIF -->

## Highlights

- **Production-Ready Pipeline** — Data prep, training, evaluation, ONNX export, and inference in one repo
- **Clone & Run** — Dataset (28MB) included in the repo, no external downloads needed
- **ONNX Deployment** — Export to ONNX for cross-platform, framework-agnostic inference
- **Configurable Experiments** — YAML-based hyperparameter management for reproducible training
- **FastAPI-Ready** — Detector class designed for direct integration with web services

## Quick Start

```bash
# Clone (dataset included, ~28MB)
git clone https://github.com/LiuSiChengGitHub/yolo_defect.git
cd yolo_defect

# Install dependencies
conda env create -f environment.yml
conda activate yolo_defect

# Prepare data (VOC XML -> YOLO TXT)
python scripts/prepare_data.py

# Train
python scripts/train.py

# Inference (after training & ONNX export)
python scripts/inference_onnx.py --model models/best.onnx --image your_image.jpg
```

## Dataset

### NEU-DET: Northeastern University Surface Defect Database

**Source:** [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)

The NEU-DET dataset contains 1,800 grayscale images of hot-rolled steel strip surfaces, covering 6 types of typical surface defects:

| Class | English | Chinese | Description |
|-------|---------|---------|-------------|
| 0 | crazing | 龟裂 | Network of fine cracks on the surface |
| 1 | inclusion | 夹杂 | Foreign material embedded in the steel |
| 2 | patches | 斑块 | Irregular discolored areas |
| 3 | pitted_surface | 麻面 | Small pits scattered across the surface |
| 4 | rolled-in_scale | 压入氧化铁皮 | Oxide scale pressed into the surface during rolling |
| 5 | scratches | 划痕 | Linear marks from mechanical contact |

### Statistics

- **Total images:** 1,800 (300 per class)
- **Image size:** 200 x 200 pixels
- **Format:** JPG (grayscale, 1 channel in annotation but readable as 3-channel)
- **Split:** Pre-divided into train (~240/class) and validation (~60/class)

### Directory Structure

The dataset is pre-split and included at `data/NEU-DET/`:

```
data/NEU-DET/
├── train/                         # ~1440 images
│   ├── annotations/               # VOC XML (flat directory)
│   │   ├── crazing_1.xml
│   │   ├── inclusion_1.xml
│   │   └── ...
│   └── images/                    # JPG (subdirectories by class)
│       ├── crazing/
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/                    # ~360 images
    ├── annotations/
    └── images/                    # Same structure as train
```

### Annotation Format

VOC XML format with `<bndbox>` containing absolute pixel coordinates:

```xml
<object>
    <name>crazing</name>
    <bndbox>
        <xmin>2</xmin>
        <ymin>2</ymin>
        <xmax>193</xmax>
        <ymax>194</ymax>
    </bndbox>
</object>
```

Each image may contain multiple bounding boxes (multiple defect instances).

## Data Preparation

### What the conversion does

`prepare_data.py` converts the original VOC XML annotations to YOLO TXT format that Ultralytics YOLOv8 expects.

**VOC XML format** (absolute pixel coordinates):
```
xmin, ymin, xmax, ymax  →  e.g., 2, 2, 193, 194
```

**YOLO TXT format** (normalized center coordinates):
```
class_id cx cy w h  →  e.g., 0 0.487500 0.490000 0.955000 0.960000
```

The normalization formula:
- `cx = (xmin + xmax) / 2 / image_width`
- `cy = (ymin + ymax) / 2 / image_height`
- `w = (xmax - xmin) / image_width`
- `h = (ymax - ymin) / image_height`

### Class Mapping

| Class Name | Class ID |
|------------|----------|
| crazing | 0 |
| inclusion | 1 |
| patches | 2 |
| pitted_surface | 3 |
| rolled-in_scale | 4 |
| scratches | 5 |

### Run

```bash
python scripts/prepare_data.py
# or specify custom paths:
python scripts/prepare_data.py --data-root data/NEU-DET --output-dir data
```

### Output Structure

```
data/
├── images/
│   ├── train/          # Flat directory, all training images
│   └── val/            # Flat directory, all validation images
├── labels/
│   ├── train/          # YOLO TXT labels, one per image
│   └── val/
└── data.yaml           # YOLO dataset config
```

### Important Notes

- The dataset is **already split** into train/validation — no random splitting needed
- `rolled-in_scale` contains a hyphen, so the script uses known class name prefix matching (longest match first) instead of naive underscore splitting
- Images are copied from class subdirectories to a flat output directory (YOLO requirement)

## Data Analysis

Running `data_analysis.py` on the converted dataset reveals the following characteristics: the dataset is well-balanced across all 6 classes (~240 train / ~60 val images each), so no oversampling or class-weighting is needed. All images are uniformly 200×200 px. Each image contains between 1 and 9 bounding boxes (mean: 2.33), indicating moderate defect density. Bounding box sizes vary dramatically — from as small as 8×9 px (narrow scratches) to nearly 199×199 px (crazing covering the entire image) — making this a challenging multi-scale detection task. The anchor-free design of YOLOv8 handles this wide size range well without manual anchor tuning. Analysis charts are saved in `docs/assets/`.

```bash
python scripts/data_analysis.py
```

## Training

### Run Training

```bash
# Using YAML config (recommended)
python scripts/train.py --config configs/train_config.yaml

# Or directly via Ultralytics CLI
yolo detect train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `yolov8n.pt` | Pre-trained model variant. `n`=nano (fastest), `s`/`m`/`l`/`x` for larger models |
| `data` | `data/data.yaml` | Dataset configuration file with paths and class names |
| `epochs` | 50 | Total training epochs. More epochs = better convergence, but risk of overfitting |
| `imgsz` | 640 | Input image size. Larger = better accuracy, slower training. Images are resized from 200x200 |
| `batch` | 16 | Batch size. Larger = more stable gradients, more GPU memory needed. Use -1 for auto |
| `lr0` | 0.01 | Initial learning rate. The optimizer adjusts this during training via scheduling |
| `optimizer` | `auto` | Optimizer selection. `auto` picks the best based on model and dataset |
| `mosaic` | 1.0 | Mosaic augmentation probability. Combines 4 images into one, improving small object detection |
| `mixup` | 0.0 | Mixup augmentation probability. Blends two images together for regularization |
| `device` | 0 | CUDA device index. Use `cpu` for CPU training |
| `workers` | 8 | Number of dataloader worker processes for data loading |

### Training Process Overview

1. **Pre-trained weights loading** — YOLOv8n is initialized with COCO pre-trained weights, providing a strong feature extraction baseline (transfer learning)
2. **Data augmentation** — Mosaic (4-image composition), mixup, random flip, HSV adjustment, and scale jitter are applied on-the-fly to improve generalization
3. **Multi-scale training** — Images are randomly resized during training to make the model robust to different object scales
4. **Automatic checkpointing** — `best.pt` (highest mAP) and `last.pt` (latest epoch) are saved under `runs/detect/train/weights/`

## Results

### Experiment Comparison

| Experiment | Model | imgsz | lr0 | epochs | mAP@0.5 | mAP@50-95 | Train Time | Notes |
|------------|-------|-------|-----|--------|---------|-----------|------------|-------|
| baseline | yolov8n | 640 | 0.01 | 50 | **0.734** | 0.390 | ~9 min | Default config, exceeds 0.70 target |
| exp1 | yolov8n | 512 | 0.01* | 50 | 0.730 | 0.393 | ~7 min | Faster, but hurts hard texture classes |
| exp2 | yolov8n | 800 | 0.01* | 50 | 0.741 | 0.384 | ~13 min | Best result in the `optimizer=auto` image-size family |
| exp3_lr01 | yolov8n (SGD) | 640 | 0.01 | 50 | 0.736 | **0.395** | ~9 min | Best `mAP@50-95`, valid fixed-SGD lr baseline |
| exp4 | yolov8n | 800 | 0.01* | 50 | 0.735 | 0.384 | ~14 min | `mixup=0.1` did not help |
| exp5 | yolov8n | 800 | 0.01* | 50 | 0.737 | 0.395 | ~13 min | No-mix augmentation control |
| final_train | yolov8n | 800 | 0.01* | 100 | 0.729 | 0.379 | ~26 min | Longer training alone did not improve the model |
| final_train_2 | yolov8n (SGD) | 800 | 0.01 | 100 | **0.743** | 0.388 | ~26 min | Manually combined final candidate, current best `mAP@0.5` |

\* `optimizer=auto` selected AdamW(lr=0.001) at runtime, so `lr0=0.01` was not the effective learning rate.

### Current Model Candidates

- **`final_train_2`** is the current deployment candidate if the headline metric is `mAP@0.5`
- **`exp3_lr01`** remains important because it has the best `mAP@50-95` under the cleanest fixed-SGD design
- **`final_train`** shows an important lesson: longer training alone is not enough if the optimizer/parameter family is not the strongest one

### Per-Class AP (Current Best: `final_train_2`)

| Class | AP@0.5 | Precision | Recall |
|-------|--------|-----------|--------|
| patches | 0.920 | 0.856 | 0.850 |
| inclusion | 0.827 | 0.773 | 0.742 |
| pitted_surface | 0.807 | 0.821 | 0.701 |
| scratches | 0.803 | 0.602 | 0.843 |
| rolled-in_scale | 0.553 | 0.507 | 0.462 |
| crazing | 0.550 | 0.513 | 0.543 |

### Comparison Insight

- `imgsz=800` helped the overall `mAP@0.5` direction, but did not solve `crazing` by itself
- Fixed-SGD learning-rate ablation showed that `lr0=0.01` clearly outperformed `0.001` under the same 50-epoch budget
- `mixup=0.1` did not help this industrial fine-texture task, while disabling sample mixing preserved some classes better
- The manually combined `final_train_2` run became the strongest `mAP@0.5` result and improved `crazing` to `0.550`
- Practical conclusion: the best final model came from a **validated cross-experiment combination**, not from longer training alone

### Training Curves

![Training Results](docs/assets/results_final_train_2.png)

### PR Curve

![PR Curve](docs/assets/PR_curve_final_train_2.png)

### Confusion Matrix

![Confusion Matrix](docs/assets/confusion_matrix_final_train_2.png)

### Sample Predictions

![Validation Predictions](docs/assets/val_pred_sample_final_train_2.jpg)

## ONNX Deployment

### Why ONNX?

- **Cross-platform** — Run on Windows, Linux, macOS, edge devices without PyTorch installed
- **Framework-agnostic** — No dependency on the training framework at inference time
- **Performance** — ONNX Runtime provides optimized inference with hardware-specific acceleration (CUDA, TensorRT, DirectML)
- **Smaller footprint** — No need to ship the entire PyTorch runtime in production

### Export

```bash
python scripts/export_onnx.py --weights runs/detect/final_train_2/weights/best.pt
# Output: models/best.onnx
```

### Inference

```bash
# Single image
python scripts/inference_onnx.py --model models/best.onnx --image path/to/image.jpg

# Batch (entire directory)
python scripts/inference_onnx.py --model models/best.onnx --image-dir data/images/val --output-dir results/
```

The current ONNX deployment target is exported with `imgsz=800`, so the model input is `[1, 3, 800, 800]` and the raw output tensor is `[1, 10, 13125]` (`4 bbox params + 6 class scores` across all candidate locations).

### Performance Comparison

| Format | mAP@0.5 | mAP@50-95 | FPS (CPU) | FPS (GPU) | Model Size | Notes |
|--------|---------|-----------|-----------|-----------|------------|-------|
| PyTorch (.pt) | **0.7433** | **0.3880** | 7.1 | 60.5 | ~6.3 MB | ultralytics `model.val()` |
| ONNX (.onnx) | **≈0.743** | **≈0.388** | 22.0 | **69.8** | 11.77 MB | `detector.py` + onnxruntime |

- 50-image approximate comparison: **50/50** images have identical detection counts (148 vs 148), confidence statistics nearly identical (mean 0.4011 vs 0.4011)
- ONNX GPU is **9.8x faster** than PyTorch CPU, and **1.15x faster** than PyTorch GPU
- GPU benchmarked on NVIDIA RTX 3060, 100 images with warmup

### YOLODetector Class (`src/detector.py`)

The `YOLODetector` class provides a clean 3-step inference API:

1. **`preprocess(image)`** — BGR to RGB, resize to model input size, normalize to 0-1, HWC to CHW, add batch dimension
2. **`predict(image)`** — Run ONNX inference, parse output tensor, apply confidence filtering and NMS, return detections list
3. **`draw(image, detections, class_names)`** — Draw bounding boxes with class labels and confidence scores

This class is designed to be directly reused by the FastAPI service in `api/`, keeping inference logic in one place.

For debugging and interview prep, `scripts/debug_detector.py` manually expands the preprocessing and forward path and prints 5 key shapes:
- original image shape
- resized image shape
- CHW tensor shape
- batched input shape
- raw ONNX output shape

### FastAPI API Usage

The project now includes a minimal FastAPI service in `api/app.py` with two endpoints:

- `GET /health` — health check for service and model readiness
- `POST /detect` — upload one image and receive detection results in JSON

Start the API service:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Health check example:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

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

Detection request example:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@data/images/val/crazing_241.jpg"
```

Example response:

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

Notes:

- The upload field name must be `file`
- The API returns JSON results, not visualization images
- `inference_time_ms` is service-side model inference time; client-observed response time can be larger under concurrent load
- `scripts/benchmark_api.py` can be used for a simple concurrency benchmark of `POST /detect`

Current local verification (2026-03-29):

- `GET /health` returned `200 OK` with `{"status":"ok","model":"best.onnx"}`
- `POST /detect` on `data/images/val/crazing_241.jpg` returned `count=3`
- `scripts/benchmark_api.py` with 10 images and concurrency 10 finished with:
  - success requests: `10/10`
  - average client-observed response time: `2333.37 ms`
  - total wall time: `3.06 s`
  - throughput: `3.27 QPS`
- Under the same run, most service-side `inference_time_ms` values were around `13-23 ms`, while the first request was much slower (`2198.88 ms`), indicating cold-start / queueing effects in local development mode

## Project Structure

```
yolo_defect/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── requirements.txt              # pip dependencies
├── environment.yml               # Conda environment (PyTorch + CUDA)
├── .gitignore                    # Ignore rules
├── data/
│   ├── data.yaml                 # YOLO dataset config (auto-generated)
│   └── NEU-DET/                  # Original dataset (committed to git)
│       ├── train/                #   Training split (~240/class)
│       └── validation/           #   Validation split (~60/class)
├── scripts/
│   ├── prepare_data.py           # VOC XML -> YOLO TXT converter
│   ├── data_analysis.py          # Dataset statistics & visualization
│   ├── train.py                  # Training entry point (reads YAML config)
│   ├── evaluate.py               # Model evaluation + PR curve + confusion matrix
│   ├── export_onnx.py            # ONNX model export
│   ├── debug_detector.py         # Debug script for intermediate shapes / ONNX output
│   ├── compare_pt_onnx.py        # 50-image approximate comparison of PT vs ONNX outputs
│   ├── benchmark_pytorch.py      # PyTorch FPS benchmark on a fixed image subset
│   ├── benchmark_api.py          # Simple concurrent benchmark for POST /detect
│   └── inference_onnx.py         # ONNX inference (single + batch)
├── src/
│   ├── __init__.py
│   └── detector.py               # YOLODetector class (ONNX inference, FastAPI reuse)
├── api/
│   └── app.py                    # FastAPI service (`GET /health`, `POST /detect`)
├── configs/
│   ├── train_config.yaml         # Baseline training hyperparameters
│   └── exp*.yaml                 # Experiment configs (imgsz/lr/augment/final runs)
├── models/
│   └── .gitkeep                  # Exported ONNX models
├── docs/
│   ├── tasks/                    # Project prompts & task docs
│   ├── notes/                    # Learning notes (YOLO theory, interview points)
│   ├── YOLO_Project.md           # Project progress log
│   ├── experiment_log.md         # Experiment tracking template
│   └── assets/                   # PR curves, demo GIFs, plots
└── runs/
    └── .gitkeep                  # YOLO training outputs (gitignored)
```

### Design Principles

- **`scripts/`** — One-off scripts for data processing, training, evaluation, export. Run from command line with argparse.
- **`src/`** — Reusable modules. `detector.py` is imported by both `inference_onnx.py` and the future FastAPI service.
- **`configs/`** — Separated hyperparameters. Easy to track experiments by diffing config files.

## Tech Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Python | Language | 3.9 |
| PyTorch | Deep learning framework | 2.0.0 |
| Ultralytics | YOLOv8 training & inference | latest |
| ONNX | Model interchange format | latest |
| ONNX Runtime | Optimized inference engine | latest (GPU) |
| OpenCV | Image processing | (via ultralytics) |
| Matplotlib | Visualization & plotting | (via ultralytics) |
| FastAPI | REST API service (Week 2) | latest |
| Conda | Environment management | — |

## Key Design Decisions

### Why YOLOv8n over YOLOv5 or larger models?

YOLOv8 is the latest generation with improved architecture (C2f modules, anchor-free detection, decoupled head). The `nano` variant is chosen because:
- NEU-DET is a small dataset (1,800 images) — a larger model would overfit
- Edge deployment friendly — fast inference on CPU and mobile devices
- Easy to scale up: if `n` isn't enough, swap to `s`/`m` with one config change

### Why include the dataset in the repo?

The NEU-DET dataset is only 28MB. Including it means:
- `git clone` → immediately runnable, no manual downloads or registration
- Guaranteed reproducibility — the exact same data every time
- Interviewer-friendly — they can verify results in minutes

### Why YAML config instead of CLI arguments?

- **Traceability** — Each experiment's config is a file that can be version-controlled and diffed
- **Reproducibility** — Re-run any experiment by pointing to its config
- **Comparison** — Side-by-side parameter comparison across experiments

### Why separate `src/detector.py`?

- **Separation of concerns** — Inference logic is independent of the training framework
- **FastAPI reuse** — The API service imports `YOLODetector` directly, no code duplication
- **Testing** — The detector can be unit-tested in isolation

## Roadmap / TODO

- [x] Baseline training and experiment tracking
- [x] Hyperparameter tuning (imgsz / lr / augment comparisons)
- [x] Bad sample analysis (misdetections, class confusion)
- [x] ONNX export and CPU inference validation
- [x] ONNX accuracy alignment (PyTorch vs ONNX)
- [x] FastAPI service with file upload endpoint
- [ ] Docker containerization for deployment
- [ ] Demo GIF and repository polish
- [ ] TensorRT / C++ ONNX Runtime optimization (V2 scope)
- [ ] CI/CD pipeline with automated testing

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The NEU-DET dataset is provided by Northeastern University (NEU). Please cite the original paper if you use this dataset in academic work:

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.

---

---

# 钢材表面缺陷检测 — 基于 YOLOv8

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange?logo=onnx)
![License](https://img.shields.io/badge/License-MIT-yellow)

端到端工业缺陷检测流水线：从数据准备到 ONNX 部署，基于 NEU-DET 钢材表面数据集与 YOLOv8 构建。

<!-- TODO: 添加 Demo GIF -->

## 项目亮点

- **生产级流水线** — 数据准备、训练、评估、ONNX 导出、推理一站式完成
- **克隆即用** — 数据集（28MB）已包含在仓库内，无需额外下载
- **ONNX 部署** — 导出为 ONNX 格式，跨平台、框架无关推理
- **可配置实验** — 基于 YAML 的超参数管理，实验可追溯可复现
- **FastAPI 就绪** — 检测器类设计直接支持 Web 服务集成

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

# 推理（训练并导出 ONNX 后）
python scripts/inference_onnx.py --model models/best.onnx --image your_image.jpg
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

- **总图片数：** 1,800（每类 300 张）
- **图片尺寸：** 200 x 200 像素
- **格式：** JPG（标注中标记为 depth=1 灰度，但实际可以作为 3 通道 BGR 读取）
- **划分：** 已预划分为训练集（每类约 240 张）和验证集（每类约 60 张），约 80:20
- **数据集已包含在 `data/NEU-DET/` 目录中**

### 原始数据目录结构

```
data/NEU-DET/
├── train/                         # 训练集 (~1440 张)
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
└── validation/                    # 验证集 (~360 张)，结构同 train
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

> **面试知识点：** VOC 格式用绝对像素坐标的角点表示 (xmin, ymin, xmax, ymax)，而 YOLO 格式用归一化的中心点+宽高 (cx, cy, w, h)。这是两种最常见的标注格式，面试常问区别。

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

## 数据分析

对转换后的数据集运行 `data_analysis.py` 可得出以下结论：6 个类别样本分布均衡（训练集每类约 240 张，验证集每类约 60 张），无需过采样或类别加权。所有图片均为 200×200 px。每张图的 bbox 数量在 1 至 9 个之间（均值 2.33），目标密度适中。Bbox 尺寸差异极大——从 8×9 px 的细长划痕到近 199×199 px 的大面积裂纹——是一个多尺度检测的挑战性场景。YOLOv8 的 anchor-free 设计无需手动设置 anchor，天然适合处理这种宽泛的尺寸分布。分析图表已保存至 `docs/assets/`。

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

| 参数 | 默认值 | 说明 | 面试要点 |
|------|--------|------|----------|
| `model` | `yolov8n.pt` | 模型变体。`n`=nano（最快），`s`/`m`/`l`/`x` 依次增大 | YOLOv8 有 5 个尺寸，参数量从 3M 到 68M |
| `data` | `data/data.yaml` | 数据集配置文件，定义路径和类名 | |
| `epochs` | 50 | 总训练轮数。太少欠拟合，太多过拟合 | 通常看 loss 曲线是否收敛来判断 |
| `imgsz` | 640 | 输入图片尺寸。原图 200x200 会被 resize 到 640x640 | 更大 = 更准但更慢，是常见调参变量 |
| `batch` | 16 | 批大小。更大 = 梯度更稳定，但需要更多显存 | -1 可以让 YOLO 自动选择最大 batch |
| `lr0` | 0.01 | 初始学习率。训练过程中会按 schedule 自动衰减 | 太大会震荡，太小收敛慢 |
| `optimizer` | `auto` | 优化器。auto 会根据模型自动选择 SGD 或 AdamW | SGD+momentum 是经典选择，Adam 收敛更快 |
| `mosaic` | 1.0 | Mosaic 数据增强概率。4 张图拼成 1 张 | 提升小目标检测，增加上下文信息 |
| `mixup` | 0.0 | Mixup 数据增强概率。两张图按比例混合 | 正则化效果，防止过拟合 |
| `device` | 0 | CUDA 设备编号。`cpu` 则用 CPU 训练 | |
| `workers` | 8 | 数据加载的工作进程数 | Windows 上可能需要设为 0 |

### 训练流程详解

1. **加载预训练权重（迁移学习）**
   - YOLOv8n 使用在 COCO 数据集（80 类、33 万张图）上预训练的权重
   - 骨干网络（Backbone）已经学会了通用的特征提取能力（边缘、纹理、形状等）
   - 我们只需要在 NEU-DET 上微调（Fine-tune），让模型学习钢材缺陷的特定特征
   - **面试考点：** 迁移学习为什么有效？因为底层特征（边缘、纹理）是通用的，高层特征才是任务特定的

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
| baseline | yolov8n | 640 | 0.01 | 50 | **0.734** | 0.390 | ~9 分钟 | 默认配置，已超过 0.70 目标 |
| exp1 | yolov8n | 512 | 0.01* | 50 | 0.730 | 0.393 | ~7 分钟 | 更快，但 hardest classes 明显下降 |
| exp2 | yolov8n | 800 | 0.01* | 50 | 0.741 | 0.384 | ~13 分钟 | `optimizer=auto` 家族里最好的图片尺寸结果 |
| exp3_lr01 | yolov8n (SGD) | 640 | 0.01 | 50 | 0.736 | **0.395** | ~9 分钟 | 固定 SGD 后最好的严格指标结果 |
| exp4 | yolov8n | 800 | 0.01* | 50 | 0.735 | 0.384 | ~14 分钟 | `mixup=0.1` 没有带来提升 |
| exp5 | yolov8n | 800 | 0.01* | 50 | 0.737 | 0.395 | ~13 分钟 | 去样本混合增强对照组 |
| final_train | yolov8n | 800 | 0.01* | 100 | 0.729 | 0.379 | ~26 分钟 | 单纯拉长训练并没有变更优 |
| final_train_2 | yolov8n (SGD) | 800 | 0.01 | 100 | **0.743** | 0.388 | ~26 分钟 | 手动组合最优候选后的当前最佳 `mAP@0.5` |

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
- 最实用的经验不是“训练更久一定更好”，而是“要先把参数组合设计对，再给更长训练预算”

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

> **面试考点：** 为什么不直接用 PyTorch 部署？因为 PyTorch 安装包 >1GB，还需要 CUDA 工具包。ONNX Runtime 只有几十 MB，且支持多种硬件后端（CPU、GPU、NPU）。在工业场景中，边缘设备可能没有 PyTorch 环境。

### 导出命令

```bash
python scripts/export_onnx.py --weights runs/detect/final_train_2/weights/best.pt
# 输出: models/best.onnx
```

### 推理命令

```bash
# 单张推理
python scripts/inference_onnx.py --model models/best.onnx --image test.jpg

# 批量推理（整个目录）
python scripts/inference_onnx.py --model models/best.onnx --image-dir data/images/val --output-dir results/
```

当前导出的 ONNX 模型使用 `imgsz=800`，因此模型输入是 `[1, 3, 800, 800]`，原始输出张量是 `[1, 10, 13125]`（`4 个框参数 + 6 个类别分数`，覆盖全部候选位置）。

### 性能对比

| 格式 | mAP@0.5 | mAP@50-95 | FPS (CPU) | FPS (GPU) | 模型大小 | 备注 |
|------|---------|-----------|-----------|-----------|----------|------|
| PyTorch (.pt) | **0.7433** | **0.3880** | 7.1 | 60.5 | ~6.3 MB (`best.pt`) | ultralytics `model.val()` |
| ONNX (.onnx) | **≈0.743** | **≈0.388** | 22.0 | **69.8** | **11.77 MB** | `detector.py` + `onnxruntime` |

> **验证要点：** ONNX 导出后 mAP 差异应 < 0.01，否则说明导出过程有精度损失。

当前状态：
- 已用 `model.val()` 测得 PyTorch 验证集基线：**mAP@0.5 = 0.7433**，**mAP@50-95 = 0.3880**
- 修复 ONNX Runtime GPU 后，50 张近似对比 **50/50 全部一致**，总检测框数 `148 vs 148`，置信度统计几乎完全相同
- ONNX GPU 达到 **69.8 FPS**（RTX 3060），比 PyTorch GPU `60.5 FPS` 快 1.15x，比 PyTorch CPU `7.1 FPS` 快 9.8x
- ONNX CPU 达到 **22.0 FPS**，比 PyTorch CPU 快 3.1x，不依赖 GPU 也能满足工业检测实时性要求
- 已自动筛选 12 张代表性结果图（每类正确 + 错误各 1 张），用于 README / 面试展示

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

> **面试高频考点 — NMS 算法：**
> 1. 按置信度从高到低排序所有检测框
> 2. 取最高分的框，与其余框逐一计算 IoU（交并比）
> 3. IoU > 阈值的框被抑制（认为检测的是同一个目标）
> 4. 重复直到处理完所有框
>
  > 本项目在 `detector.py` 中手动实现了 NMS（不依赖 torchvision），面试可以直接讲。

该类的设计目的是**复用**：`scripts/inference_onnx.py` 和未来的 FastAPI 服务都直接 `from src.detector import YOLODetector`，推理逻辑只写一份。

另外，`scripts/debug_detector.py` 用于手动展开预处理与 ONNX 前向过程，并打印 5 个关键 shape，适合排查预处理问题和准备面试表达。

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

本地实测结果（2026-03-29）：

- `GET /health` 已返回 `200 OK`，且 `status=ok`
- `POST /detect` 对 `data/images/val/crazing_241.jpg` 返回 `count=3`
- 用 `scripts/benchmark_api.py` 对 `POST /detect` 做 10 张图、并发 10 的简单压测：
  - 成功请求数：`10/10`
  - 客户端观测平均响应时间：`2333.37 ms`
  - 总耗时：`3.06 s`
  - 吞吐量：`3.27 QPS`
- 同一轮压测里，大多数服务端 `inference_time_ms` 在 `13-23 ms`，但第一条请求达到 `2198.88 ms`，说明本地开发模式下存在明显的冷启动 / 排队影响

## 项目结构

```
yolo_defect/
├── README.md                     # 项目说明（英文+中文双版本）
├── CLAUDE.md                     # AI 助手上下文文件
├── LICENSE                       # MIT 开源协议
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
│   ├── tasks/                    # 项目任务文档
│   ├── notes/                    # 学习笔记（YOLO 理论、面试点）
│   ├── YOLO_Project.md           # 项目推进与学习笔记
│   ├── experiment_log.md         # 实验记录模板
│   └── assets/                   # PR 曲线、Demo GIF、分析图表
└── runs/
    └── .gitkeep                  # YOLO 训练输出（gitignored）
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
| FastAPI | REST API 服务（Week 2） | latest |
| Conda | 环境管理 | — |

## 关键设计决策

### 为什么选 YOLOv8n 而不是 v5 或更大的模型？

- **YOLOv8 vs YOLOv5：** YOLOv8 是最新一代，架构改进包括 C2f 模块（替代 C3）、Anchor-Free 检测头（不需要预定义锚框）、解耦头（分类和回归分开处理）。同等大小下 YOLOv8 精度更高。
- **为什么 nano (n) 版本：** NEU-DET 只有 1,800 张图，数据集很小。用更大的模型（s/m/l）容易过拟合，且推理速度慢。nano 版本仅 3.2M 参数，在边缘设备上也能实时运行。
- **灵活升级：** 如果 nano 精度不够，改一行配置就能换成 s 或 m，无需改代码。

### 为什么把数据集放在仓库里？

NEU-DET 数据集只有 28MB（远小于 GitHub 的 100MB 单文件限制）。放在仓库里意味着：
- `git clone` 后立刻可以跑，不需要手动下载、注册账号、解压
- 保证完全可复现——每个人用的是完全相同的数据
- 面试官友好——几分钟内就能验证你的结果

### 为什么用 YAML 配置文件而不是命令行参数？

- **可追溯**：每次实验的配置是一个文件，可以 git commit 保存
- **可对比**：用 `diff exp1.yaml exp2.yaml` 直接看两次实验改了什么
- **可复现**：`python train.py --config exp1.yaml` 就能精确重现实验

### 为什么 `src/detector.py` 要独立封装？

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
- [ ] Docker 容器化部署
- [ ] Demo GIF 与仓库展示优化
- [ ] TensorRT / C++ ONNX Runtime 优化（V2）
- [ ] CI/CD 流程与自动化测试

## 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

NEU-DET 数据集由东北大学提供，学术引用请参考：

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.

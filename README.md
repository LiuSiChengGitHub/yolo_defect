# Industrial Vision AI Runtime for Steel Surface Defect Detection

[中文版](README_zh.md)

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange?logo=onnx)
![OpenCV](https://img.shields.io/badge/OpenCV-C%2B%2B-green)
![CMake](https://img.shields.io/badge/CMake-planned-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

V2 positioning: this repository is being upgraded from a YOLOv8 defect-detection demo into an industrial vision AI inference runtime and C++ engineering project.

YOLOv8 and NEU-DET are the model and dataset carriers. The autumn-recruiting story is not "I trained a detector"; it is "I turned a vision model into a deployable runtime with C++ / ONNX Runtime C++ / OpenCV / CMake / GTest / benchmark evidence and deployment-optimization analysis."

Current V1 assets remain valuable: training, ONNX export, PyTorch-vs-ONNX consistency checks, Python ONNX Runtime inference, FastAPI, Docker, and benchmark scripts. V2 builds on these assets through `cpp_infer/` instead of rewriting them.

The project entry is intentionally concentrated in this README and `README_zh.md`. `AGENTS.md` only records Codex collaboration boundaries; task queue and change notes live in the README to keep the project easy to review before interviews.

![Inference Demo](docs/assets/demo_inference_result.gif)

## Project 1 Runtime Blueprint

### 1. Project Positioning and Top-Level Design

This repository is **Project 1: Industrial Vision Edge AI Runtime and C++ Engineering System**. Its core value is not to retrain a detector inside this repo, but to turn industrial defect-detection model artifacts into a runnable, testable, benchmarkable, and interview-explainable C++ runtime.

Two model sources are planned:

- **YOLOv8 + NEU-DET:** the stable P0 runtime baseline. It is used to finish the C++ deployment chain quickly because its output format is simple and the existing repo already has training, ONNX export, Python inference, FastAPI, Docker, and benchmark evidence.
- **`paper_detect` D010 / D-FINE-S + DeepPCB:** the later research-side artifact source. `paper_detect` owns training, validation, ablation, official test, result cards, and qualitative figures; this repo consumes those artifacts through a contract and, when ready, adapts them into the runtime.

The top-level design follows this rule: **training and research artifacts flow into Project 1; Project 1 owns deployment, runtime behavior, testing, benchmark evidence, and runtime event output.**

### 2. Problem Solved

Project 1 solves the gap between "a detector exists" and "a detector can be deployed and explained as engineering software":

- Convert images and model artifacts into a reproducible C++ inference path.
- Make preprocessing, inference, postprocessing, NMS, benchmark, and output writing observable as separate modules.
- Record commands, sample outputs, failures, and trade-offs so the project can be reproduced during autumn recruiting.
- Prepare `inference_event` output for Project 2, where industrial runtime events become backend incidents and Agent-assisted diagnosis input.

### 3. End-to-End Runtime Link

Planned full chain:

```text
model artifact
-> artifact contract / model card
-> RuntimeConfig
-> OpenCV image read
-> letterbox preprocess / RGB / float32 / NCHW tensor
-> ONNX Runtime C++ session
-> raw output shape check
-> postprocess / score filter / NMS / coordinate restore
-> detection JSON
-> visualization
-> benchmark report
-> optional INT8 PTQ / TensorRT attempt
-> sample inference_event for Project 2
```

Current verified chain through P1-03:

```text
cpp_infer/configs/default_config.txt
-> RuntimeConfig
-> data/images/val/crazing_241.jpg
-> OpenCV BGR image
-> letterbox preprocess
-> RGB float32 NCHW tensor
-> stable CLI summary
-> CTest smoke
```

### 4. Core Module Responsibilities

| Module | Responsibility | Current Status |
|--------|----------------|----------------|
| `ConfigLoader` | Parse runtime config such as input size, class names, thresholds, and backend. | P1-02 verified |
| `ImagePreprocessor` | Read images with OpenCV, letterbox, BGR->RGB, normalize, and produce NCHW float tensor. | P1-03 verified |
| `OnnxRunner` | Load ONNX Runtime session, inspect input/output names and shapes, build input tensor, run inference. | Placeholder for P1-04 |
| `PostProcessor` | Decode raw model outputs, filter scores, restore coordinates to the original image. | Placeholder for P1-05 |
| `NmsProcessor` | Provide testable IoU and NMS logic; this is a good code-practice candidate. | Placeholder for P1-05/P1-07 |
| `ResultWriter / Visualizer` | Write detection JSON and visualization images for demo evidence. | Placeholder |
| `BenchmarkRunner` | Measure preprocess, inference, postprocess, and end-to-end latency with warmup/repeat. | Placeholder for P1-06 |
| `ArtifactRegistry / ModelCard` | Record artifact source, model family, dataset, metrics, config, postprocess type, runtime status, and paths. | Placeholder for D010 L1 |
| `Tests` | Use CTest smoke now; add GTest groups for config, preprocess, NMS, postprocess, and artifact schema later. | CTest smoke now, GTest placeholder |

### 5. Quick Start

Current C++ runtime smoke path:

```cmd
:: Run from a Visual Studio 2026 Developer Command Prompt.
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_03
set PATH=D:\01_Base\Tools\opencv\build\x64\vc16\bin;%PATH%

cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles" -DOpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt --image data\images\val\crazing_241.jpg
ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

The older Python/YOLO quick start remains below for V1 baseline reproduction. The C++ path above is the V2 deployment entry.

### 6. Demo Input and Output

Current demo input:

```text
config: cpp_infer/configs/default_config.txt
image:  data/images/val/crazing_241.jpg
```

Current P1-03 demo output summary:

```text
P1-03 Preprocess summary
original_size: 200x200
channels: 3
input_size: 800x800
resized_size: 800x800
scale: 4.000000
padding: left=0, top=0, right=0, bottom=0
color: BGR->RGB
normalization: float32 [0, 1]
layout: NCHW
tensor_shape: 1x3x800x800
tensor_elements: 1920000
```

Future demo output placeholders:

```text
detection_json: samples/outputs/crazing_241_detections.json
visualization:   samples/outputs/crazing_241_vis.jpg
benchmark_json:  samples/outputs/benchmark_yolo_fp32.json
event_json:      samples/outputs/inference_event_sample.json
```

### 7. Test Commands

Current CTest smoke:

```cmd
ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

Expected current result:

```text
100% tests passed, 0 tests failed out of 3
```

Future GTest placeholder:

```cmd
"%BUILD_DIR%\bin\yolo_defect_cpp_tests.exe" --gtest_filter=*
```

### 8. Key Data and Artifact Results

| Item | Current Record |
|------|----------------|
| P0 dataset | NEU-DET steel surface defects, 1,800 images, 6 classes, 200x200 pixels |
| P0 model | YOLOv8n baseline and tuned variants |
| Best current YOLO result | `final_train_2`, mAP@0.5 = 0.743, mAP@50-95 = 0.388 |
| ONNX/PyTorch alignment | 50/50 images have identical detection-count matches; total detections 146 vs 146 |
| Existing ONNX benchmark | ONNX CPU 24.4 FPS, ONNX GPU 72.1 FPS on RTX 3060 |
| Current C++ runtime state | P1-03 config + OpenCV preprocess + CTest smoke verified |
| Incoming research artifact | `paper_detect` D010, D-FINE-S architecture, DeepPCB dataset |
| D010 reported result from route plan | formal validation AP50-95 = 0.847057; official test AP50-95 = 0.830385 |
| D010 relationship | D010 is the proposed artifact; D003 is the reference/ancestor; D010 improves all 6 classes over D003 in formal and official-test deltas |
| D010 integration level | L0 result card first, L1 model artifact contract second, L2 ONNX/runtime adapter only when artifact stability is confirmed |

Pending artifact paths:

```text
artifacts/paper_detect_d010/result_card.md        # placeholder
artifacts/paper_detect_d010/model_artifact.yaml   # placeholder
artifacts/paper_detect_d010/metrics_table.csv     # placeholder
artifacts/paper_detect_d010/qualitative/          # placeholder
```

### 9. Key Design Trade-Offs

- **Runtime first, training second:** this repo keeps old training assets but does not make training the V2 main story.
- **YOLO baseline before D010 adapter:** YOLO/ONNX is the quickest stable path to finish C++ preprocess, inference, postprocess, JSON, benchmark, and tests.
- **Layered D010 adoption:** D010 starts as artifact evidence; C++ D-FINE postprocess is not allowed to block the runtime baseline.
- **Simple C++ over broad framework work:** C++17, CMake, OpenCV, ONNX Runtime C++, GTest, and benchmark output are enough for the interview target.
- **Smoke tests before full unit tests:** CTest smoke keeps each small stage runnable; GTest comes when there is stable logic worth testing deeply.
- **Failure records matter:** TensorRT, INT8, or D-FINE runtime failures are acceptable if commands, errors, root causes, and fallback decisions are documented.

### 10. Task Queue

The detailed P1 queue is maintained in the Roadmap section below. At the phase level:

| Phase | Goal | Project 1 Focus |
|-------|------|-----------------|
| Phase 0: plan freeze | Align README and engineering entry | Position Project 1 as C++ edge runtime; record paper_detect D010 artifact path |
| Phase 1: C++ Runtime P0 | Make the model run in C++ | Config, preprocess, ONNX Runtime, postprocess/NMS, JSON, visualization, benchmark |
| Phase 2: deployment evaluation | Add engineering evidence | benchmark protocol, GTest, error handling, INT8 PTQ attempt |
| Phase 3: paper_detect artifact adapter | Add research-side artifact credibility | D010 result card, model_artifact contract, optional D-FINE runtime adapter |
| Phase 4: cloud-edge collaboration | Connect Project 1 to Project 2 | sample inference_event JSON |
| Phase 5: resume freeze | Stop adding big features | README, demo, tests, reports, FAQ, interview script |

### 11. Version Changes and Progress Records

Current state: P1-00 to P1-03 are complete and verified. The next implementation stage is still P1-04 ONNX Runtime session smoke unless the user explicitly asks for a documentation or artifact-contract step first.

The chronological V2 entry log is kept in the Roadmap section below and must be updated after every small stage.

### 12. Teaching Log From Project Start to Now

| Stage | What Was Done | Purpose | Implementation / Evidence | Issue and Debugging Lesson |
|-------|---------------|---------|----------------------------|----------------------------|
| P1-00 | Froze V2 positioning, protected legacy assets, created `cpp_infer/` entry. | Stop the repo from drifting between training demo and runtime project. | README/README_zh/AGENTS plus C++ workspace skeleton. | Keep README as the main story; avoid scattering tasks into many docs. |
| P1-01 | Added minimal C++17/CMake executable and CTest help smoke. | Prove the repo can build a C++ runtime target. | `yolo_defect_cpp --help` and CTest smoke. | Visual Studio multi-config builds need `ctest -C Debug`. |
| P1-02 | Added no-dependency ConfigLoader and `--config` CLI path. | Make runtime behavior config-driven before adding image/model code. | Parsed input size, class names, thresholds, backend; printed stable summary. | CLI argument errors became the first useful smoke-test failure signal. |
| P1-03 | Added OpenCV image read and YOLO-style preprocess. | Convert a real image into the model-ready tensor format. | `original_size`, `scale`, `padding`, `BGR->RGB`, `[0,1]`, `NCHW`, `1x3x800x800`; CTest 3/3 passed. | OpenCV Windows pack required `OpenCV_DIR=...\x64\vc16\lib` and `PATH=...\x64\vc16\bin`. |

## Highlights

- **Best Experimental Result** — Best checkpoint `final_train_2` reaches **mAP@0.5 = 0.743** on NEU-DET
- **PyTorch vs ONNX Consistency Check** — 50-image comparison shows **50/50** identical detection-count matches, with total detections **146 vs 146**
- **Inference Speed Benchmark** — PyTorch CPU **8.43 FPS**; PyTorch GPU (RTX 3060) **110.8 FPS**; ONNX CPU **24.4 FPS**; ONNX GPU (RTX 3060) **72.1 FPS** — all measured on 100 timed images (5 warmup)
- **Docker Verified** — `python:3.9-slim` image has been tested with `/health` and `/detect`
- **Clone & Run** — Dataset (28MB) included in the repo, no external downloads needed

## Key Metrics

| Metric | Value |
|--------|-------|
| Best model | `final_train_2` |
| mAP@0.5 | **0.743** |
| mAP@50-95 | **0.388** |
| PT/ONNX same-count ratio | **50 / 50** images (**100%**) |
| Mean abs count diff | **0.000** |
| PyTorch CPU benchmark | **8.43 FPS** / **118.66 ms** per image |
| PyTorch GPU benchmark (RTX 3060) | **110.8 FPS** / **9.0 ms** per image |
| ONNX CPU benchmark | **24.4 FPS** / **40.9 ms** per image |
| ONNX GPU benchmark (RTX 3060) | **72.1 FPS** / **13.9 ms** per image |
| Model size (`best.pt` / `best.onnx`) | ~6.0 MiB / ~11.8 MiB |

## V1 Python Baseline Quick Start

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

# Export ONNX from the default training output
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt

# Inference on a real validation image
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg
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

- **Dataset paper / official description:** 1,800 images (300 per class)
- **Files bundled in `data/NEU-DET/`:** 1,800 readable JPG images
- **Image size:** 200 x 200 pixels
- **Format:** JPG (grayscale, 1 channel in annotation but readable as 3-channel)
- **Generated YOLO copy in `data/images/`:** 1,439 train + 361 val images

### Directory Structure

The dataset is pre-split and included at `data/NEU-DET/`:

```
data/NEU-DET/
├── train/                         # 1,439 readable images
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
└── validation/                    # 361 XMLs, 361 readable images
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
- If the raw dataset is updated manually, rerun `prepare_data.py` so `data/images/` and `data/labels/` stay in sync with `data/NEU-DET/`

## Data Analysis

Running `data_analysis.py` on the converted dataset reveals the following characteristics: the dataset is effectively balanced across all 6 classes, so no oversampling or class-weighting is needed. All images are uniformly 200×200 px. Each image contains between 1 and 9 bounding boxes (mean: 2.33), indicating moderate defect density. Bounding box sizes vary dramatically — from as small as 8×9 px (narrow scratches) to nearly 199×199 px (crazing covering the entire image) — making this a challenging multi-scale detection task. The anchor-free design of YOLOv8 handles this wide size range well without manual anchor tuning. Analysis charts are saved in `docs/assets/`.

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
| baseline | yolov8n | 640 | 0.01 | 50 | **0.734** | 0.390 | 9.4 min | Default config, exceeds 0.70 target |
| exp1 | yolov8n | 512 | 0.01* | 50 | 0.733 | 0.391 | 7.2 min | Faster, but hurts hard texture classes |
| exp2 | yolov8n | 800 | 0.01* | 50 | 0.742 | 0.385 | 13.4 min | Best result in the `optimizer=auto` image-size family |
| exp3_lr01 | yolov8n (SGD) | 640 | 0.01 | 50 | 0.736 | **0.395** | 9.0 min | Best `mAP@50-95`, valid fixed-SGD lr baseline |
| exp4 | yolov8n | 800 | 0.01* | 50 | 0.741 | 0.384 | 13.6 min | `mixup=0.1` did not help |
| exp5 | yolov8n | 800 | 0.01* | 50 | 0.740 | 0.387 | 13.3 min | No-mix augmentation control |
| final_train | yolov8n | 800 | 0.01* | 100 | 0.729 | 0.379 | 26.1 min | Longer training alone did not improve the model |
| final_train_2 | yolov8n (SGD) | 800 | 0.01 | 100 | **0.743** | 0.388 | 25.9 min | Manually combined final candidate, current best `mAP@0.5` |

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
# Quick Start path: export the checkpoint produced by the default `scripts/train.py` run
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
# Output: models/best.onnx
```

If you want to reproduce the best reported metrics in this README, export the best experiment checkpoint instead:

```bash
python scripts/export_onnx.py --weights runs/detect/final_train_2/weights/best.pt --imgsz 800
```

### Inference

```bash
# Single image
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg

# Batch (entire directory)
python scripts/inference_onnx.py --model models/best.onnx --image-dir data/images/val --output-dir results/
```

The current ONNX deployment target is exported with `imgsz=800`, so the model input is `[1, 3, 800, 800]` and the raw output tensor is `[1, 10, 13125]` (`4 bbox params + 6 class scores` across all candidate locations).

### Performance Comparison

| Check | Value | Evidence |
|-------|-------|----------|
| Best PyTorch validation result | **mAP@0.5 = 0.7433**, **mAP@50-95 = 0.3880** | `docs/experiment_log.md` |
| PyTorch CPU benchmark | **8.43 FPS**, **118.66 ms/image** over **100** timed images | `results/pytorch_benchmark_100.json` |
| PyTorch GPU benchmark (RTX 3060) | **110.8 FPS**, **9.0 ms/image** over **100** timed images | `results/pytorch_benchmark_gpu.json` |
| ONNX CPU benchmark | **24.4 FPS**, **40.9 ms/image** over **100** timed images | `results/onnx_benchmark_cpu.json` |
| ONNX GPU benchmark (RTX 3060) | **72.1 FPS**, **13.9 ms/image** over **100** timed images | `results/onnx_benchmark_gpu.json` |
| PT vs ONNX detection-count match | **50 / 50** images (**100%**) | `results/pt_onnx_compare/compare_50_summary.json` |
| Total detections in PT vs ONNX check | **146 vs 146** | `results/pt_onnx_compare/compare_50_summary.json` |
| Mean absolute detection-count difference | **0.000** | `results/pt_onnx_compare/compare_50_summary.json` |
| Current local model sizes | `best.pt = 6,286,072 bytes`, `best.onnx = 12,336,935 bytes` | local artifacts |

### YOLODetector Class (`src/detector.py`)

The `YOLODetector` class provides a clean 3-step inference API:

1. **`preprocess(image)`** — BGR to RGB, letterbox resize (aspect-ratio preserving with gray padding, matching Ultralytics training preprocessing), normalize to 0-1, HWC to CHW, add batch dimension
2. **`predict(image)`** — Run ONNX inference, parse output tensor, apply confidence filtering and NMS, return detections list
3. **`draw(image, detections, class_names)`** — Draw bounding boxes with class labels and confidence scores

This class is designed to be directly reused by the FastAPI service in `api/`, keeping inference logic in one place.

For debugging, `scripts/debug_detector.py` manually expands the preprocessing and forward path and prints 5 key shapes:
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
      "bbox": [0.0, 53.68, 176.91, 146.23]
    }
  ]
}
```

Notes:

- The upload field name must be `file`
- The API returns JSON results, not visualization images
- `inference_time_ms` is service-side model inference time; client-observed response time can be larger under concurrent load
- `scripts/benchmark_api.py` can be used for a simple concurrency benchmark of `POST /detect`

Current local verification:

- `GET /health` returned `200 OK` with `{"status":"ok","model":"best.onnx"}`
- `POST /detect` on `data/images/val/crazing_241.jpg` returned `count=3`
- `scripts/benchmark_api.py` is included for local concurrency testing, but its raw benchmark log is not committed yet, so throughput numbers are omitted here

### Docker Deployment

A minimal deployment image is now provided via `Dockerfile`:

- base image: `python:3.9-slim`
- runtime deps only: `requirements-api.txt`
- copied into image: `src/`, `api/`, `models/`
- exposed port: `8000`

Build and run:

```bash
docker build -t yolo-defect-api .
docker run --rm -p 8000:8000 yolo-defect-api
```

Quick verification:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/detect \
  -F file=@data/images/val/crazing_241.jpg
```

Current Docker verification:

- `GET /health` returned `status=ok`, `model=best.onnx`
- `POST /detect` on `crazing_241.jpg` returned `count=3`
- Numeric Docker benchmark logs are not committed yet, so only endpoint-level verification is reported here

## Research Repository Coordination

This repository now works together with `paper_detect` instead of duplicating it.

| Repository | Role | Responsibilities |
|------------|------|------------------|
| `paper_detect` | Research and paper repository | Dataset splits, baselines, method changes, ablations, formal evaluation, formal ONNX export, PyTorch/ONNX consistency checks, Python ORT benchmark, paper tables and figures |
| `yolo_defect` | Portfolio and deployment engineering repository | Stable demo, ONNX/Python/C++ inference, OpenCV preprocessing, CMake, GTest, deployment benchmark, FastAPI/Docker, README and interview-facing documentation |

In short, `paper_detect` proves why the model is better; `yolo_defect` proves how the model runs reliably.

The two repositories should exchange artifacts, benchmark logs, commit/tag references, environment notes, and result tables. They should not copy entire codebases back and forth.

### Artifact Contract

An artifact is not just one model file. It is a small evidence package that explains, reproduces, and validates a model:

```text
artifacts/2026-06-20_method_v1/
├── best.pt
├── best.onnx
├── train_config.yaml
├── export_config.yaml
├── input_spec.json
├── class_map.json
├── metrics.json
├── per_class_ap.csv
├── complexity.json
├── compare_pt_onnx.json
├── latency_python_ort.json
└── result_card.md
```

Formal training, evaluation, and paper ONNX export are owned by `paper_detect`. This repository consumes the exported artifact, especially `best.onnx`, `input_spec.json`, `class_map.json`, and `result_card.md`, then runs deployment-oriented validation and benchmarks.

If this repository re-exports ONNX, that export is only for portfolio reproduction or sanity checks. It is not the source of formal paper results.

### Benchmark Feedback

Deployment logs generated here can flow back into `paper_detect` for paper tables, for example:

```text
results/cpp_benchmark/
├── 2026-06-20_method_v1_cpp_ort_cpu.json
├── 2026-06-20_method_v1_cpp_ort_gpu.json
└── 2026-06-20_method_v1_consistency.json
```

Each benchmark log should record:

- the artifact/model used;
- the `yolo_defect` commit or tag;
- the command line;
- hardware and operating system;
- ONNX Runtime version and execution provider;
- whether preprocessing and postprocessing/NMS are included;
- warmup, repeat count, mean latency, P50/P90/P99 latency, and FPS.

As long as the experiment really runs, the logs are traceable, and the paper describes the measurement scope honestly, this is a valid cross-repository experiment organization.

### Smoke Tests

Smoke tests prove that a chain runs without crashing. They are not final accuracy or performance claims.

Required smoke tests:

- train smoke test: one short training run in `paper_detect`;
- export smoke test: one temporary ONNX export in `paper_detect`;
- Python ORT smoke test: load ONNX and print output shapes;
- C++ ORT smoke test: compile, load ONNX, print input/output shapes and basic latency in `yolo_defect`.

Current repository versioning:

- `v0.1-intern0`: stable internship/portfolio baseline before C++ deployment work;
- `deploy-cpp`: branch for C++ ONNX Runtime, OpenCV, CMake, GTest, and benchmark development.

## Project Structure

```
yolo_defect/
├── Dockerfile                    # Docker image for FastAPI deployment
├── AGENTS.md                     # Codex collaboration boundaries for V2 work
├── README.md                     # This file (English)
├── README_zh.md                  # Chinese version
├── LICENSE                       # MIT License
├── requirements-api.txt          # Minimal runtime dependencies for Docker/API
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
│   ├── benchmark_pytorch.py      # PyTorch CPU/GPU FPS benchmark on a fixed image subset
│   ├── benchmark_onnx.py         # ONNX CPU/GPU FPS benchmark on a fixed image subset
│   ├── benchmark_api.py          # Simple concurrent benchmark for POST /detect
│   ├── analyze_failures.py       # Failure-case analysis for false positives/negatives
│   ├── select_representative_examples.py  # Select representative examples for README
│   └── inference_onnx.py         # ONNX inference (single + batch)
├── src/
│   ├── __init__.py
│   └── detector.py               # YOLODetector class (ONNX inference, FastAPI reuse)
├── api/
│   └── app.py                    # FastAPI service (`GET /health`, `POST /detect`)
├── cpp_infer/                    # V2 C++ runtime workspace
│   ├── README.md                 # C++ runtime scope and planned layout
│   ├── configs/                  # Future C++ runtime config files
│   ├── include/yolo_defect_cpp/  # Future public headers
│   ├── src/                      # Future C++ implementation files
│   └── tests/                    # Future GTest cases
├── configs/
│   ├── train_config.yaml         # Baseline training hyperparameters
│   └── exp*.yaml                 # Experiment configs (imgsz/lr/augment/final runs)
├── models/
│   └── .gitkeep                  # Exported ONNX models (gitignored)
├── docs/
│   ├── experiment_log.md         # Experiment tracking template
│   └── assets/                   # PR curves, demo GIFs, plots
└── runs/                         # YOLO training outputs (gitignored)
```

### Design Principles

- **`scripts/`** — One-off scripts for data processing, training, evaluation, export. Run from command line with argparse.
- **`src/`** — Reusable modules. `detector.py` is imported by both `inference_onnx.py` and the FastAPI service.
- **`cpp_infer/`** — V2 C++ deployment workspace. It will own CMake, OpenCV preprocessing, ONNX Runtime C++ inference, postprocessing, benchmark, and GTest.
- **`configs/`** — Separated hyperparameters. Easy to track experiments by diffing config files.

## Tech Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Python | Language | 3.9 |
| C++ | V2 runtime language | C++17 |
| PyTorch | Deep learning framework | 2.0.0 |
| Ultralytics | YOLOv8 training & inference | latest |
| ONNX | Model interchange format | latest |
| ONNX Runtime | Python baseline and planned C++ inference engine | latest (GPU) |
| OpenCV | Python image utilities and planned C++ preprocessing/visualization | (via ultralytics) / planned C++ |
| CMake | Planned C++ build system | planned |
| GTest | Planned C++ unit tests | planned |
| Matplotlib | Visualization & plotting | (via ultralytics) |
| FastAPI | REST API service | latest |
| Conda | Environment management | — |

## Key Design Decisions

### Model Selection

YOLOv8 is the latest generation with improved architecture (C2f modules, anchor-free detection, decoupled head). The `nano` variant is chosen because:
- NEU-DET is a small dataset (1,800 images) — a larger model would overfit
- Edge deployment friendly — fast inference on CPU and mobile devices
- Easy to scale up: if `n` isn't enough, swap to `s`/`m` with one config change

### Dataset Inclusion

The NEU-DET dataset is only 28MB. Including it means:
- `git clone` → immediately runnable, no manual downloads or registration
- Guaranteed reproducibility — the exact same data every time
- Easy to verify — anyone can reproduce the pipeline in minutes

### Config Management

- **Traceability** — Each experiment's config is a file that can be version-controlled and diffed
- **Reproducibility** — Re-run any experiment by pointing to its config
- **Comparison** — Side-by-side parameter comparison across experiments

### Detector Module

- **Separation of concerns** — Inference logic is independent of the training framework
- **FastAPI reuse** — The API service imports `YOLODetector` directly, no code duplication
- **Testing** — The detector can be unit-tested in isolation

## Roadmap

### V1 Baseline Already Done

- [x] Baseline training and experiment tracking
- [x] Hyperparameter tuning (imgsz / lr / augment comparisons)
- [x] Bad sample analysis (misdetections, class confusion)
- [x] ONNX export and CPU inference validation
- [x] ONNX accuracy alignment (PyTorch vs ONNX)
- [x] FastAPI service with file upload endpoint
- [x] Docker containerization for deployment
- [x] Demo GIF for inference walkthrough

### V2 P1 Task Queue

The V2 queue follows `docs/路线0628.md`, especially sections 5.3, 5.5, and 6.1-6.8. README is the task and change-log entry point; standalone task documents are intentionally avoided unless a future module becomes too large for README. Small-stage plans are created dynamically after each completed stage instead of being fully frozen up front.

| ID | Status | Task | Scope | Acceptance |
|----|--------|------|-------|------------|
| P1-00 | Done | README / AGENTS / C++ workspace entry | Freeze V2 positioning, Codex boundaries, task queue, and `cpp_infer/` skeleton | README/README_zh explain that YOLO/NEU-DET are carriers and C++ Runtime is the core; `AGENTS.md` protects legacy assets; `cpp_infer/` exists without full inference implementation |
| P1-01 | Verified with VS Developer Command Prompt | CMake skeleton | Add the first minimal CMake project and executable target | `cpp_infer` has a minimal C++17 CMake target, executable target, and CTest smoke test. Configure/build/run pass in the Visual Studio 2026 Developer Command Prompt; Visual Studio multi-config builds require `ctest -C Debug` |
| P1-02 | Verified with NMake CTest smoke | ConfigLoader | Load `input_width`, `input_height`, `class_names`, `score_threshold`, `nms_threshold`, and `backend` | `cpp_infer/configs/default_config.txt` is parsed into a typed `RuntimeConfig`; `yolo_defect_cpp --config ...` prints a stable config summary; CTest covers the config smoke path without OpenCV, ONNX Runtime, GTest, preprocessing, postprocessing, NMS, or benchmark wiring |
| P1-03 | Verified with OpenCV CTest smoke | OpenCV preprocess | Read an image, print shape/channels, letterbox, BGR to RGB, normalize, HWC to CHW | `--config ... --image ...` reads a real validation image and prints original shape, target input size, scale, padding, color conversion, normalization, NCHW tensor shape, and tensor element count |
| P1-04 | Pending | ONNX Runtime session smoke | Load `models/best.onnx`, create a session, print input/output names and shapes | Model loading works and failure paths explain missing model/runtime/provider issues |
| P1-05 | Pending | Postprocess / NMS | Decode model output into boxes, map coordinates back to the original image, filter and NMS | Single-image detection JSON is produced with class, confidence, and box fields |
| P1-06 | Pending | Benchmark | Measure preprocess / infer / postprocess / end-to-end latency with warmup and repeat | Benchmark output includes mean, P50, P95, FPS, repeat count, and model/image metadata |
| P1-07 | Pending | GTest | Add focused tests for config, preprocess, NMS, and postprocess | At least three meaningful C++ test groups can run from a documented command |
| P1-08 | Pending | INT8 PTQ | Try post-training quantization and compare FP32 vs INT8 | A comparison table records model size, latency/FPS, detection consistency, and any accuracy or compatibility trade-off |
| P1-09 | Pending | TensorRT attempt | Try TensorRT FP16/INT8 conversion or document the blocked path | Report records engine build command, benchmark if successful, or clear failure reason if unsupported |
| P1-10 | Placeholder | paper_detect D010 L0 result-card sync | Add lightweight D010/D003/D001 result card, metrics, per-class delta, qualitative figure, and config summary placeholders under `artifacts/paper_detect_d010/` | README explains D010 as a research-side artifact source without making this repo responsible for D-FINE training |
| P1-11 | Placeholder | model_artifact contract | Define a minimal artifact contract for YOLO baseline and paper_detect D010: source repo, branch/commit, method, dataset, metrics, preprocess, postprocess type, runtime status, and paths | `model_artifact.yaml` style schema can be explained and later consumed by C++/Python tooling |
| P1-12 | Placeholder | inference_event sample | Define sample output event for Project 2: asset/image/model artifact id, detections, runtime timings, benchmark profile, warning flags, and timestamp | Project 1 can explain how edge runtime output becomes Project 2 incident input |

### P1-01 CMake Skeleton Commands

P1-01 only establishes a C++17/CMake entry point. It intentionally does not include OpenCV, ONNX Runtime, GTest, preprocessing, postprocessing, or NMS.

```powershell
# Configure
cmake -S cpp_infer -B cpp_infer\build

# Build
cmake --build cpp_infer\build

# Run: Visual Studio multi-config generators usually place the executable here
.\cpp_infer\build\bin\Debug\yolo_defect_cpp.exe --help

# Run: single-config generators usually place the executable here
.\cpp_infer\build\bin\yolo_defect_cpp.exe --help

# Smoke test for Visual Studio multi-config generators
ctest --test-dir cpp_infer\build -C Debug --output-on-failure
```

Local verification on 2026-06-05: configure and build passed in the Visual Studio 2026 Developer Command Prompt. `ctest --test-dir cpp_infer\build --output-on-failure` failed because Visual Studio is a multi-config generator and needs a configuration name. `ctest --test-dir cpp_infer\build -C Debug --output-on-failure` passed, and `cpp_infer\build\bin\Debug\yolo_defect_cpp.exe --help` printed the P1-01 skeleton help text.

### P1-02 ConfigLoader Commands

P1-02 adds a no-third-party-dependency `key = value` config parser and a `--config` CLI path. It intentionally does not connect OpenCV, ONNX Runtime, GTest, preprocessing, postprocessing, NMS, or benchmark logic.

```cmd
:: Run from a Visual Studio 2026 Developer Command Prompt.
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_02
cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles"
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt

ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

Expected config summary fields:

- `input_width: 800`
- `input_height: 800`
- `class_count: 6`
- `class_names: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`
- `score_threshold: 0.25`
- `nms_threshold: 0.45`
- `backend: cpu`

Local verification on 2026-06-10: configure/build/run/CTest passed in a Visual Studio 2026 Developer Command Prompt with the NMake build tree under `%TEMP%`. The config smoke test first failed against the P1-01 skeleton with `Unknown argument: --config`, then passed after the ConfigLoader implementation. After P1-03, use the P1-03 configure command because the executable target now links OpenCV.

### P1-03 OpenCV Preprocess Commands

P1-03 adds OpenCV image reading and YOLO-style letterbox preprocessing. It intentionally does not connect ONNX Runtime, inference, postprocessing, NMS, benchmark, or GTest.

```cmd
:: Run from a Visual Studio 2026 Developer Command Prompt.
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_03
set PATH=D:\01_Base\Tools\opencv\build\x64\vc16\bin;%PATH%

cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles" -DOpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt --image data\images\val\crazing_241.jpg

ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

Expected preprocess summary fields:

- `original_size: 200x200`
- `channels: 3`
- `input_size: 800x800`
- `resized_size: 800x800`
- `scale: 4.000000`
- `padding: left=0, top=0, right=0, bottom=0`
- `color: BGR->RGB`
- `normalization: float32 [0, 1]`
- `layout: NCHW`
- `tensor_shape: 1x3x800x800`
- `tensor_elements: 1920000`

Local verification on 2026-06-13: the P1-03 smoke test first failed against the P1-02 CLI with `--config expects exactly one config file path.` Configure/build/run/CTest then passed after adding OpenCV and `ImagePreprocessor`. The local OpenCV Windows pack requires `OpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib`; pointing to the top-level `D:\01_Base\Tools\opencv\build` was not sufficient for this NMake build.

### V2 Entry Log

| Date | Change | Purpose |
|------|--------|---------|
| 2026-06-04 | Established P1-00 V2 entry: README positioning, Codex boundary file, and `cpp_infer/` skeleton | Make the project explainable as an industrial vision AI Runtime project before deeper C++ implementation starts |
| 2026-06-05 | Verified P1-01 CMake skeleton in the Visual Studio 2026 Developer Command Prompt | Confirmed configure/build/run/CTest smoke test; documented the `ctest -C Debug` requirement for Visual Studio multi-config builds |
| 2026-06-10 | Added P1-02 ConfigLoader and `--config` smoke path | Introduced a typed no-dependency runtime config parser and documented the build/run/CTest evidence before moving toward OpenCV preprocessing |
| 2026-06-13 | Added P1-03 OpenCV read-image and letterbox preprocess smoke path | Confirmed real-image preprocessing output, including original shape, RGB conversion, normalization, NCHW layout, scale, padding, and tensor shape before ONNX Runtime integration |
| 2026-06-29 | Aligned README with `docs/路线0628.md` Project 1 plan | Recorded top-level design, D010/paper_detect artifact path, required README sections, phase queue placeholders, and teaching log so later work stays on the C++ Runtime route |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The NEU-DET dataset is provided by Northeastern University (NEU). Please cite the original paper if you use this dataset in academic work:

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.

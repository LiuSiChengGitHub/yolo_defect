# Experiment Log

## Objective

Track and compare training configurations to find the best model for NEU-DET steel surface defect detection.
Target: mAP@0.5 > 0.70. Weights: `runs/detect/<name>/weights/best.pt`. Charts: `docs/assets/`.

---

## Training Results

| Exp | Model | imgsz | lr0 | epochs | mosaic | batch | mAP@0.5 | mAP@50-95 | Time | Weights |
|-----|-------|-------|-----|--------|--------|-------|---------|-----------|------|---------|
| **baseline** | yolov8n | 640 | 0.01 | 50 | 1.0 | 16 | **0.734** | 0.390 | 9.4 min | train2/best.pt |
| exp1 | | | | | | | | | | |
| exp2 | | | | | | | | | | |
| exp3 | | | | | | | | | | |

## Per-Class AP@0.5

| Class | baseline | exp1 | exp2 | exp3 |
|-------|----------|------|------|------|
| crazing | 0.542 | | | |
| inclusion | 0.793 | | | |
| patches | 0.928 | | | |
| pitted_surface | 0.830 | | | |
| rolled-in_scale | 0.581 | | | |
| scratches | 0.731 | | | |

## Deployment Performance

| Format | mAP@0.5 | FPS (CPU) | FPS (GPU) | Model Size |
|--------|---------|-----------|-----------|------------|
| PyTorch (.pt) | - | - | - | 6.3 MB |
| ONNX (.onnx) | - | - | - | - |

---

## Baseline Analysis (2026-03-24)

**Config:** YOLOv8n, epochs=50, imgsz=640, batch=16, lr0=0.01, RTX 3060

**Key findings:**
- mAP@0.5 = 0.734, exceeds target at baseline — no emergency need for tuning
- Loss still declining at epoch 50: room for improvement with more epochs
- Main problem: detection failure (漏检) into background, not class confusion
  - crazing: 52% missed (AP=0.542) — diffuse texture, low contrast with background
  - rolled-in_scale: 42% missed (AP=0.581)
- Best class: patches (AP=0.928) — distinct block patterns, easy to distinguish

**User analysis:** *(fill in your own interpretation here)*

---

## Experiment Plans

| Exp | Change | Hypothesis | Priority |
|-----|--------|------------|----------|
| exp1 | epochs: 50→150 | Loss not converged, more epochs should gain +3-5% | High |
| exp2 | model: yolov8n→yolov8s | More parameters → better feature extraction for crazing | High |
| exp3 | cls: 0.5→1.0 | Higher classification loss weight → less missed detections | Medium |
| exp4 | Best combination of above | - | After exp1-3 |

---

## Conclusions & Next Steps

*(To be filled after Step 3 experiments)*

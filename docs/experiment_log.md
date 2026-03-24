# Experiment Log

## Objective

Track and compare training configurations to find the best model for NEU-DET steel surface defect detection.
Target: mAP@0.5 > 0.70. Weights: `runs/detect/<name>/weights/best.pt`. Charts: `docs/assets/`.

---

## Training Results

| Exp | Model | imgsz | lr0 | epochs | mosaic | batch | mAP@0.5 | mAP@50-95 | Time | Weights |
|-----|-------|-------|-----|--------|--------|-------|---------|-----------|------|---------|
| **baseline** | yolov8n | 640 | 0.01 | 50 | 1.0 | 16 | **0.734** | 0.390 | 9.4 min | train2/best.pt |
| exp1 | yolov8n | 512 | 0.01 | 50 | 1.0 | 16 | 0.733 | **0.391** | 7.2 min | exp1/best.pt |
| exp2 | | | | | | | | | | |
| exp3 | | | | | | | | | | |

## Per-Class AP@0.5

| Class | baseline | exp1 | exp2 | exp3 |
|-------|----------|------|------|------|
| crazing | 0.542 | 0.476 | | |
| inclusion | 0.793 | 0.835 | | |
| patches | 0.928 | 0.909 | | |
| pitted_surface | 0.830 | 0.821 | | |
| rolled-in_scale | 0.581 | 0.520 | | |
| scratches | 0.731 | 0.821 | | |

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

## Exp1 Analysis (2026-03-24)

**Config:** YOLOv8n, epochs=50, imgsz=512, batch=16, lr0=0.01, mosaic=1.0, RTX 3060

**Key findings:**
- Best checkpoint reached **mAP@0.5 = 0.733** and **mAP@50-95 = 0.391**, essentially flat vs baseline on the main metric
- Training time dropped from **9.4 min → 7.2 min**, about **23% faster**
- Smaller input size hurt the two hardest classes:
  - crazing: **0.542 → 0.476**
  - rolled-in_scale: **0.581 → 0.520**
- Some easier/distinct classes improved or stayed strong:
  - inclusion: **0.793 → 0.835**
  - scratches: **0.731 → 0.821**
- Practical interpretation: `imgsz=512` keeps overall mAP close, but loses fine-grained texture detail on the hardest industrial defect classes

**Important note for future experiments:**
- Ultralytics printed `optimizer=auto`, which **overrode `lr0=0.01` and selected AdamW(lr=0.001)** automatically
- This means a future "learning-rate-only" ablation is **not valid** unless we first set a fixed optimizer instead of `auto`

**User analysis:** *(fill in your own interpretation here)*

---

## Experiment Plans

| Exp | Change | Hypothesis | Priority |
|-----|--------|------------|----------|
| exp1 | imgsz: 640→512 | Faster training, but may lose fine texture detail | Done |
| exp2 | imgsz: 640→800 | More spatial detail may help hard classes | Next |
| exp3 | fixed optimizer + lr ablation | Need disable `optimizer=auto` before testing lr fairly | Next |
| exp4 | epochs: 50→150 | Loss not converged, more epochs may still help | Next |

---

## Conclusions & Next Steps

*(To be filled after Step 3 experiments)*

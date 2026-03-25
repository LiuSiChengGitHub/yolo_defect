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
| exp2 | yolov8n | 800 | 0.01* | 50 | 1.0 | 16 | **0.742** | 0.385 | 13.4 min | exp2/best.pt |
| exp3-lr01 | yolov8n (SGD) | 640 | 0.01 | 50 | 1.0 | 16 | **0.736** | **0.395** | 9.0 min | exp3_lr01/best.pt |
| exp3-lr001 | yolov8n (SGD) | 640 | 0.001 | 50 | 1.0 | 16 | 0.711 | 0.387 | 10.1 min | exp3_lr001/best.pt |
| exp4 | yolov8n | 800 | 0.01* | 50 | 1.0 | 16 | 0.741 | 0.384 | 13.6 min | exp4/best.pt |

## Per-Class AP@0.5

| Class | baseline | exp1 | exp2 | exp3-lr01 | exp3-lr001 | exp4 |
|-------|----------|------|------|-----------|------------|------|
| crazing | 0.542 | 0.476 | 0.476 | 0.470 | 0.462 | 0.464 |
| inclusion | 0.793 | 0.835 | 0.833 | 0.819 | 0.813 | 0.812 |
| patches | 0.928 | 0.909 | 0.921 | 0.926 | 0.921 | 0.930 |
| pitted_surface | 0.830 | 0.821 | 0.807 | 0.817 | 0.802 | 0.813 |
| rolled-in_scale | 0.581 | 0.520 | 0.596 | 0.537 | 0.507 | 0.577 |
| scratches | 0.731 | 0.821 | 0.815 | 0.846 | 0.762 | 0.815 |

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

## Exp2 Analysis (2026-03-25)

**Config:** YOLOv8n, epochs=50, imgsz=800, batch=16, lr0=0.01, mosaic=1.0, RTX 3060

**Key findings:**
- Best checkpoint reached **mAP@0.5 = 0.742**, the highest among baseline/exp1/exp2 image-size experiments
- Final validation on `best.pt` gave **mAP@0.5 = 0.741** and **mAP@50-95 = 0.384**
- Training time increased from **9.4 min → 13.4 min**, about **43% slower** than baseline
- Larger input size did help some classes:
  - rolled-in_scale: **0.581 → 0.596**
  - inclusion: **0.793 → 0.833**
  - scratches: **0.731 → 0.815**
- But the hardest class `crazing` did **not** improve:
  - crazing: **0.542 → 0.476**
- Practical interpretation: `imgsz=800` improves overall mAP50 and some texture classes, but does not solve the core `crazing` failure while costing noticeably more training time

**Important note:**
- This run still used `optimizer=auto`, so Ultralytics again selected **AdamW(lr=0.001)**
- Therefore exp2 is a fair comparison against baseline/exp1 on the **imgsz** dimension, but not against fixed-SGD learning-rate experiments

**User analysis:** *(fill in your own interpretation here)*

---

## Exp3 Analysis (2026-03-25)

**Config design:** fix `optimizer=SGD`, keep `imgsz=640 / epochs=50 / batch=16 / mosaic=1.0` unchanged, compare only `lr0=0.01` vs `lr0=0.001`

**Why this design matters:**
- Previous baseline used `optimizer=auto`, so `lr0=0.01` was not a fair control
- This time `args.yaml` confirms both runs actually used `SGD`, making the learning-rate ablation valid

**Key findings:**
- `exp3-lr01` reached **mAP@0.5 = 0.736** and **mAP@50-95 = 0.395**, slightly above baseline
- `exp3-lr001` reached **mAP@0.5 = 0.711** and **mAP@50-95 = 0.387**, clearly below both `exp3-lr01` and baseline
- Lower learning rate did **not** improve the hardest classes:
  - crazing: **0.470 → 0.462**
  - rolled-in_scale: **0.537 → 0.507**
- Training also became a bit slower with smaller lr:
  - `exp3-lr01`: **9.0 min**
  - `exp3-lr001`: **10.1 min**
- Practical interpretation: under the same 50-epoch budget, `lr0=0.001` is likely too conservative for this task and leads to under-training rather than better stability

**Engineering note:**
- While starting this experiment, `train.py` hit a Windows encoding issue because YAML was opened with the default locale codec
- Fixed by reading config files with `encoding="utf-8"`, so future YAML files can safely contain Chinese comments

**User analysis:** *(fill in your own interpretation here)*

---

## Exp4 Analysis (2026-03-25)

**Config design:** use the current best full setup from `exp2` (`imgsz=800`, `optimizer=auto`) and add only `mixup=0.1` while keeping `mosaic=1.0`

**How this experiment was set up:**
- Step 1: take `exp2.yaml` as the base config because it had the best end-to-end mAP50 among the completed full runs
- Step 2: keep `imgsz=800`, `epochs=50`, `batch=16`, `mosaic=1.0`, `optimizer=auto`
- Step 3: change only `mixup: 0.0 -> 0.1`
- Step 4: after training, verify in `runs/detect/exp4/args.yaml` that `mixup: 0.1` actually took effect

**Key findings:**
- Best checkpoint reached **mAP@0.5 = 0.741** and **mAP@50-95 = 0.384**, essentially flat to slightly below `exp2`
- Final validation on `best.pt` gave **mAP@0.5 = 0.735** and **mAP@50-95 = 0.384**
- Per-class AP did not show clear gains over `exp2`:
  - crazing: **0.476 -> 0.464**
  - rolled-in_scale: **0.596 -> 0.577**
  - scratches: **0.815 -> 0.815** (roughly flat)
- Training time stayed high at about **13.6 min**, similar to `exp2`
- Practical interpretation: adding `mixup=0.1` on top of strong mosaic augmentation did not improve the project bottlenecks and may have made fine defect textures harder to preserve

**Important note:**
- This run still used `optimizer=auto`, so Ultralytics again selected **AdamW(lr=0.001)**
- Therefore exp4 is best understood as an **augmentation-only** comparison against `exp2`

**User analysis:** *(fill in your own interpretation here)*

---

## Experiment Plans

| Exp | Change | Hypothesis | Priority |
|-----|--------|------------|----------|
| exp1 | imgsz: 640→512 | Faster training, but may lose fine texture detail | Done |
| exp2 | imgsz: 640→800 | More spatial detail may help hard classes | Done |
| exp3 | fixed SGD + lr0: 0.01→0.001 | Lower lr may be steadier, but can underfit within 50 epochs | Done |
| exp4 | mixup: 0.0→0.1 on top of exp2 | Stronger mixing may improve generalization | Done |
| exp5 | epochs: 50→150 | Loss not converged, more epochs may still help | Next |

---

## Conclusions & Next Steps

*(To be filled after Step 3 experiments)*

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
| exp5 | yolov8n | 800 | 0.01* | 50 | 0.0 | 16 | 0.740 | 0.387 | 13.3 min | exp5/best.pt |
| final_train | yolov8n | 800 | 0.01* | 100 | 1.0 | 16 | 0.729 | 0.379 | 26.1 min | final_train/best.pt |
| final_train_2 | yolov8n (SGD) | 800 | 0.01 | 100 | 1.0 | 16 | **0.743** | 0.388 | 25.9 min | final_train_2/best.pt |

## Per-Class AP@0.5

| Class | baseline | exp1 | exp2 | exp3-lr01 | exp3-lr001 | exp4 | exp5 | final_train | final_train_2 |
|-------|----------|------|------|-----------|------------|------|------|-------------|---------------|
| crazing | 0.542 | 0.476 | 0.476 | 0.470 | 0.462 | 0.464 | 0.462 | 0.468 | 0.550 |
| inclusion | 0.793 | 0.835 | 0.833 | 0.819 | 0.813 | 0.812 | 0.838 | 0.813 | 0.827 |
| patches | 0.928 | 0.909 | 0.921 | 0.926 | 0.921 | 0.930 | 0.928 | 0.920 | 0.920 |
| pitted_surface | 0.830 | 0.821 | 0.807 | 0.817 | 0.802 | 0.813 | 0.800 | 0.812 | 0.807 |
| rolled-in_scale | 0.581 | 0.520 | 0.596 | 0.537 | 0.507 | 0.577 | 0.522 | 0.544 | 0.553 |
| scratches | 0.731 | 0.821 | 0.815 | 0.846 | 0.762 | 0.815 | 0.874 | 0.816 | 0.803 |

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

## Exp5 Analysis (2026-03-26)

**Config design:** use the same base as `exp2/exp4` (`imgsz=800`, `optimizer=auto`) and explicitly disable both sample-mixing augmentations: `mosaic=0.0`, `mixup=0.0`

**How this experiment was set up:**
- Step 1: keep the same base setup as `exp2/exp4`
- Step 2: set `mosaic: 0.0` and `mixup: 0.0`
- Step 3: after training, verify in `runs/detect/exp5/args.yaml` that both values actually became `0.0`

**Key findings:**
- Best checkpoint reached **mAP@0.5 = 0.740** and **mAP@50-95 = 0.387**
- Final validation on `best.pt` gave **mAP@0.5 = 0.737** and **mAP@50-95 = 0.395**
- Compared with `exp4`, turning off sample-mixing augmentation improved some difficult classes:
  - inclusion: **0.812 -> 0.838**
  - scratches: **0.815 -> 0.874**
  - rolled-in_scale stayed weaker than `exp2` but was better than `exp4`: **0.577 -> 0.522** vs `exp2=0.596`
- `crazing` remained difficult and stayed low at **0.462**
- Practical interpretation: for this industrial defect task, completely removing mosaic/mixup did not hurt overall performance and may preserve natural texture cues better than stronger mixing augmentation

**Important note:**
- This run still used `optimizer=auto`, so Ultralytics again selected **AdamW(lr=0.001)**
- Therefore exp5 should be compared mainly against `exp2` and `exp4` as an augmentation ablation family

**User analysis:** *(fill in your own interpretation here)*

---

## Final Train Analysis (2026-03-26)

**Config design:** start from the strongest full-run image-size family (`imgsz=800`), keep `optimizer=auto`, keep `mosaic=1.0`, remove `mixup` based on exp4/exp5, and extend training to **100 epochs** for a final longer-budget run

**How this final run was set up:**
- Step 1: choose the image-size direction from `exp2` because it gave the highest complete-run mAP@0.5
- Step 2: keep `mixup=0.0` because `exp4` showed no benefit from `mixup=0.1`
- Step 3: keep `mosaic=1.0` instead of fully disabling it, because `exp5` improved some classes but did not beat `exp2` overall
- Step 4: increase only `epochs: 50 -> 100` and save as `configs/final_train.yaml`
- Step 5: after training, validate `runs/detect/final_train/weights/best.pt` separately to record the final deployment-ready metrics

**Key findings:**
- Standalone validation on `best.pt` gave **mAP@0.5 = 0.729** and **mAP@50-95 = 0.379**
- During training, the highest single-row `mAP@0.5` reached **0.736** at epoch 87, but that was **not** the final `best.pt`
- This happened because Ultralytics selects `best.pt` by overall fitness centered on stricter metrics, not by `mAP@0.5` alone
- Compared with `exp2`, extending training from **50 -> 100 epochs** did **not** improve the final best model:
  - exp2 `best.pt`: **0.741 / 0.384**
  - final_train `best.pt`: **0.729 / 0.379**
- Hard classes still did not break through:
  - crazing: **0.476 -> 0.468** (vs exp2)
  - rolled-in_scale: **0.596 -> 0.544** (vs exp2)
- Total training time increased to about **26.1 min**, almost double the 50-epoch `imgsz=800` runs

**Important note:**
- This run still used `optimizer=auto`, and `args.yaml` confirms Ultralytics kept **AdamW(lr=0.001)**
- The final result is therefore best interpreted as a **longer-training verification** of the `imgsz=800` family, not as a new optimizer/lr conclusion
- Practical takeaway: under the current augmentation + optimizer family, simply training longer did not solve the core industrial defect bottlenecks

**User analysis:** *(fill in your own interpretation here)*

---

## Final Train 2 Analysis (2026-03-26)

**Config design:** manually combine the strongest candidate settings across the completed experiments: `imgsz=800`, `optimizer=SGD`, `lr0=0.01`, `mosaic=1.0`, `mixup=0.0`, `epochs=100`

**How this final run was set up:**
- Step 1: keep `imgsz=800` from the strongest image-size family
- Step 2: use fixed `SGD + lr0=0.01` from the valid lr-ablation winner
- Step 3: keep `mixup=0.0` because `mixup=0.1` did not help in exp4
- Step 4: keep `mosaic=1.0` as the main augmentation baseline and extend training to 100 epochs
- Step 5: the user manually created the config and launched training; evaluation and documentation were completed afterward

**Key findings:**
- Standalone validation on `best.pt` gave **mAP@0.5 = 0.743** and **mAP@50-95 = 0.388**
- This is now the **highest mAP@0.5** among all completed runs
- The best epoch in `results.csv` was epoch **90**, with `mAP@0.5 = 0.743` and `mAP@50-95 = 0.388`, very close to the standalone validation result
- Compared with previous top runs:
  - vs exp2 `best.pt`: **0.741 / 0.384 → 0.743 / 0.388**
  - vs exp3-lr01 `best.pt`: **0.736 / 0.395 → 0.743 / 0.388**
- The hardest class `crazing` improved the most:
  - crazing: **0.468 / 0.476 family results → 0.550**
- But some classes did not become global bests:
  - rolled-in_scale stayed below exp2 (**0.553 vs 0.596**)
  - scratches stayed below exp5 (**0.803 vs 0.874**)
- Total training time was about **25.9 min**

**Important note:**
- This run is not inherited from one earlier experiment; it is a **new manually combined final candidate**
- The result is useful exactly because it verifies that a cross-family combination can outperform the previous end-to-end winner on the headline metric
- Practical takeaway: optimizer family matters, and longer training became valuable only after switching to the stronger manual combination

**User analysis:** *(fill in your own interpretation here)*

---

## Experiment Plans

| Exp | Change | Hypothesis | Priority |
|-----|--------|------------|----------|
| exp1 | imgsz: 640→512 | Faster training, but may lose fine texture detail | Done |
| exp2 | imgsz: 640→800 | More spatial detail may help hard classes | Done |
| exp3 | fixed SGD + lr0: 0.01→0.001 | Lower lr may be steadier, but can underfit within 50 epochs | Done |
| exp4 | mixup: 0.0→0.1 on top of exp2 | Stronger mixing may improve generalization | Done |
| exp5 | mosaic: 1.0→0.0, mixup: 0.0 | No-mix augmentation control on top of exp2 family | Done |
| final_train | use current best full setup + epochs: 100 | Verify whether longer training improves the deployment candidate | Done |
| final_train_2 | manually combine imgsz=800 + SGD + lr0=0.01 + mixup=0.0 + epochs=100 | Validate the “all current best knobs together” candidate | Done |
| exp6 | epochs: 50→150 | Loss not converged, more epochs may still help | Optional |

---

## Conclusions & Next Steps

- By **mAP@0.5**, the strongest run is now **final_train_2** (`0.743` on `best.pt`)
- By **mAP@50-95**, the strongest result remains **exp3-lr01** (`0.395` on `best.pt`) under a cleaner fixed-SGD design
- The `final_train` run shows that longer training alone is not enough; the improved `final_train_2` result came from a better overall parameter combination
- Recommended deployment/reporting choice for now:
  - If headline metric is `mAP@0.5`, use **final_train_2**
  - If you want the cleanest lr-ablation methodology and strongest `mAP@50-95`, keep **exp3-lr01** as an important comparison model
-  next priority should move to **ONNX export, inference verification, and service packaging**

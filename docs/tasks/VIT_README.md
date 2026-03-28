# TransUNet Brain MRI Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![ViT](https://img.shields.io/badge/Vision_Transformer-Hybrid-green)
![Medical Imaging](https://img.shields.io/badge/Medical_Imaging-Brain_MRI-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A PyTorch implementation of **TransUNet** for brain MRI tumor segmentation, combining Vision Transformer (ViT) for global context and CNN for local detail extraction. Achieves **Dice 0.788** on the LGG binary segmentation benchmark.

<!-- Demo visualization placeholder: add sample prediction images here -->
<!-- ![Demo](outputs/prediction_01.png) -->

## Highlights

- **Hybrid ViT + CNN Architecture** — Custom TransUNet implementation from scratch, not a third-party wrapper
- **Patient-Level Data Split** — Prevents data leakage by ensuring all slices from one patient stay in the same set
- **Production-Ready Training** — Focal loss, warmup + cosine LR scheduling, AMP, checkpoint resume, artifact export
- **Comparison Baseline** — Built-in U-Net for benchmark-style experiments

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train (with real LGG data)
python src/train.py --data_dir ./data/lgg-mri-segmentation/kaggle_3m \
    --model transunet_lite --epochs 50 --batch_size 4 --lr 1e-4 \
    --loss focal_dice --warmup_epochs 5 --save_dir ./checkpoints/lgg_baseline --amp

# 3. Inference
python src/inference.py --data_dir ./data/lgg-mri-segmentation/kaggle_3m \
    --checkpoint ./checkpoints/lgg_baseline/best_model.pth \
    --num_samples 10 --output_dir ./outputs/lgg_baseline
```

No real data? Generate synthetic samples for testing:
```bash
python src/generate_synthetic_data.py --output_dir ./data/synthetic_mri --num_samples 100
```

## Architecture

```
Input MRI (3, 256, 256)
        |
   +----+----+
   |         |
   v         v
 ViT        CNN
Encoder    Encoder
   |         |
   |    (B,256,16,16)      (B,512,16,16)  +  skip connections
   |         |                               (B,64,256,256)
   +----+----+                               (B,128,128,128)
        |                                    (B,256,64,64)
     1x1 Conv Fusion (1280 -> 512)           (B,512,32,32)
        |
   Decoder (4x UpBlock with skip connections)
        |
Output Mask (1, 256, 256)
```

| Component | Details |
|---|---|
| ViT Encoder | patch_size=16, embed_dim=256, depth=6, heads=8 (Lite) |
| CNN Encoder | 4-stage, base_channels=32, MaxPool downsampling |
| Decoder | ConvTranspose2d upsampling + skip connection fusion |
| Fusion | Concatenate ViT + CNN bottleneck features, 1x1 conv reduction |

## Dataset

**LGG MRI Segmentation** (Kaggle)
- Source: [kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- 110 patients, ~3,929 MRI slices with binary tumor masks
- Format: `.tif` images with corresponding `_mask.tif` files
- Place under `./data/lgg-mri-segmentation/kaggle_3m/`

## Training

```bash
# TransUNet-Lite (recommended)
python src/train.py --data_dir ./data/lgg-mri-segmentation/kaggle_3m \
    --model transunet_lite --epochs 60 --batch_size 4 --lr 1e-4 \
    --loss focal_dice --warmup_epochs 5 --save_dir ./checkpoints/transunet --amp

# U-Net comparison baseline
python src/train.py --data_dir ./data/lgg-mri-segmentation/kaggle_3m \
    --model unet --epochs 50 --batch_size 4 --lr 1e-4 \
    --loss bce_dice --save_dir ./checkpoints/unet
```

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning Rate | 1e-4 with warmup + cosine decay |
| Loss | Dice + Focal BCE (focal_dice) |
| Image Size | 256 x 256 |
| Batch Size | 4 |
| Data Split | 70/30 patient-level |
| Augmentation | Random flip, rotation (90), contrast jitter |

## Results

| Model | Params | Val Dice | Val IoU | Best Epoch |
|---|---:|---:|---:|---:|
| **TransUNet-Lite** | 9.44M | **0.788** | — | 47/50 |
| UNet | 7.76M | — | — | — |

> U-Net comparison experiment is available but not yet run on the full LGG dataset.

Every training run automatically exports:
`best_model.pth`, `last_model.pth`, `train_config.json`, `history.csv`, `training_curves.png`, `run_summary.json`

## Project Structure

```
ViT_Seg/
├── src/
│   ├── model.py                 # TransUNet, TransUNetLite, UNet architectures
│   ├── dataset.py               # Data loading with patient-level split
│   ├── losses.py                # Dice, BCE+Dice, Focal+Dice, CE+Dice losses
│   ├── metrics.py               # Dice, IoU, accuracy, MetricTracker
│   ├── train.py                 # Training entry point with full artifact export
│   ├── inference.py             # Inference, visualization, summary export
│   ├── generate_synthetic_data.py  # Synthetic dataset generator
│   └── check_env.py            # Environment diagnostics
├── docs/
│   ├── learning_guide.md        # ViT + TransUNet learning guide (Chinese)
│   └── interview_prep_plan.md   # 7-day interview prep plan (Chinese)
├── README.md
├── CLAUDE.md                    # AI assistant execution handbook
├── requirements.txt
└── LICENSE
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **TransUNet over pure ViT** | Pure ViT loses spatial detail at patch boundaries; CNN encoder + skip connections recover fine-grained features |
| **Dice + BCE/Focal loss** | Dice handles class imbalance globally; BCE/Focal provides stable per-pixel gradients — they complement each other |
| **Patient-level split** | Slice-level random split leaks patient identity (adjacent slices look nearly identical), inflating metrics artificially |
| **TransUNet-Lite variant** | Full TransUNet (768-dim, 12 layers) needs >6GB VRAM; Lite (256-dim, 6 layers) trains on consumer GPUs with minimal accuracy loss |
| **Warmup + cosine schedule** | Prevents early gradient explosion in Transformer parameters while allowing fine convergence |

## Tech Stack

| Component | Technology |
|---|---|
| Framework | PyTorch 2.0+ |
| Model | Custom TransUNet / TransUNetLite / UNet |
| Training | AdamW, AMP, warmup + cosine LR, checkpoint resume |
| Loss Functions | DiceLoss, BCEDiceLoss, FocalDiceLoss, CEDiceLoss |
| Data | PIL + NumPy, patient-level split |
| Visualization | Matplotlib (overlay, side-by-side comparison) |
| Experiment Tracking | JSON/CSV artifact export |

## License

[MIT](LICENSE)

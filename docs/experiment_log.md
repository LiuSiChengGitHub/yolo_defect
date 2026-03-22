# Experiment Log

## Objective

Track and compare different training configurations to find the best model for NEU-DET steel surface defect detection.

## Variables

| Variable | Description | Range |
|----------|-------------|-------|
| `imgsz` | Input image size | 320, 640, 800 |
| `lr0` | Initial learning rate | 0.001 - 0.01 |
| `epochs` | Training epochs | 50, 100, 150 |
| `mosaic` | Mosaic augmentation ratio | 0.0 - 1.0 |
| `mixup` | Mixup augmentation ratio | 0.0 - 0.5 |
| `model` | Model variant | yolov8n, yolov8s |

## Training Results

| Experiment | imgsz | lr0 | epochs | mosaic | mAP@0.5 | mAP@50-95 | Train Time | Notes |
|------------|-------|-----|--------|--------|---------|-----------|------------|-------|
| baseline | 640 | 0.01 | 50 | 1.0 | - | - | - | Default config |

## Per-Class AP

| Class | AP@0.5 (Exp 1) | AP@0.5 (Exp 2) | AP@0.5 (Exp 3) |
|-------|-----------------|-----------------|-----------------|
| crazing | - | - | - |
| inclusion | - | - | - |
| patches | - | - | - |
| pitted_surface | - | - | - |
| rolled-in_scale | - | - | - |
| scratches | - | - | - |

## Deployment Performance

| Format | mAP@0.5 | FPS (CPU) | FPS (GPU) | Model Size |
|--------|---------|-----------|-----------|------------|
| PyTorch (.pt) | - | - | - | - |
| ONNX (.onnx) | - | - | - | - |

## Failure Analysis

<!-- Add misdetection/false positive analysis here -->

## Conclusions & Next Steps

<!-- Summarize findings and plan next experiments -->

"""PyTorch 推理速度测试：在固定图片子集上测量 FPS。"""

import argparse
import json
import os
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch YOLO inference speed"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    parser.add_argument(
        "--weights",
        default=os.path.join(project_root, "runs", "detect", "final_train_2", "weights", "best.pt"),
        help="Path to PyTorch weights (.pt)",
    )
    parser.add_argument(
        "--image-dir",
        default=os.path.join(project_root, "data", "images", "val"),
        help="Directory of images to benchmark",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of timed images (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup images before timing (default: 5)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Inference image size (default: 800)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device, e.g. cpu or 0 (default: cpu)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "results", "pytorch_benchmark_100.json"),
        help="Path to save benchmark summary JSON",
    )
    args = parser.parse_args()

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [
        os.path.join(args.image_dir, f)
        for f in sorted(os.listdir(args.image_dir))
        if f.lower().endswith(supported)
    ]

    total_needed = args.warmup + args.num_images
    image_paths = image_paths[:total_needed]
    if len(image_paths) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} images, but found only {len(image_paths)}"
        )

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # 先把图读进内存，避免磁盘 IO 影响 FPS 结果
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        images.append((os.path.basename(path), image))

    warmup_images = images[:args.warmup]
    timed_images = images[args.warmup:]

    print(f"Warmup: {len(warmup_images)} image(s)")
    for _, image in warmup_images:
        model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

    print(f"Timing: {len(timed_images)} image(s)")
    total_time = 0.0
    total_detections = 0

    for idx, (name, image) in enumerate(timed_images, start=1):
        start = time.perf_counter()
        results = model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        elapsed = time.perf_counter() - start
        total_time += elapsed

        boxes = results[0].boxes
        det_count = len(boxes) if boxes is not None else 0
        total_detections += det_count

        print(f"[{idx}/{len(timed_images)}] {name}: {elapsed * 1000:.1f} ms, {det_count} detection(s)")

    avg_time = total_time / len(timed_images)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0

    summary = {
        "weights": args.weights,
        "device": args.device,
        "imgsz": args.imgsz,
        "conf_thresh": args.conf,
        "iou_thresh": args.iou,
        "warmup_images": args.warmup,
        "timed_images": len(timed_images),
        "avg_time_ms": avg_time * 1000.0,
        "fps": fps,
        "avg_detections_per_image": total_detections / len(timed_images),
    }

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("PyTorch Benchmark Summary")
    print("=" * 60)
    print(f"Device:                    {args.device}")
    print(f"Timed images:              {len(timed_images)}")
    print(f"Average latency:           {summary['avg_time_ms']:.1f} ms/image")
    print(f"Average FPS:               {summary['fps']:.2f}")
    print(f"Average detections/image:  {summary['avg_detections_per_image']:.2f}")
    print(f"Saved JSON:                {output_path}")


if __name__ == "__main__":
    main()

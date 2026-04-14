"""ONNX 推理速度测试：在固定图片子集上测量端到端 FPS。"""

import argparse
import json
import os
import sys
import time

import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

from src.detector import YOLODetector  # noqa: E402


def provider_list(provider):
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX YOLO inference speed"
    )
    parser.add_argument(
        "--model",
        default=os.path.join(project_root, "models", "best.onnx"),
        help="Path to ONNX model",
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
        "--provider",
        choices=("cpu", "cuda"),
        default="cuda",
        help="ONNX Runtime provider to benchmark (default: cuda)",
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
        "--output",
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

    print(f"Loading model: {args.model}")
    detector = YOLODetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        providers=provider_list(args.provider),
    )
    session_providers = detector.session.get_providers()
    requested_provider = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    if requested_provider not in session_providers:
        raise RuntimeError(
            f"Requested {requested_provider}, but session providers are {session_providers}"
        )

    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        images.append((os.path.basename(path), image))

    warmup_images = images[:args.warmup]
    timed_images = images[args.warmup:]

    print(f"Providers: {session_providers}")
    print(f"Warmup: {len(warmup_images)} image(s)")
    for _, image in warmup_images:
        detector.predict(image)

    print(f"Timing: {len(timed_images)} image(s)")
    total_time = 0.0
    total_detections = 0

    for idx, (name, image) in enumerate(timed_images, start=1):
        start = time.perf_counter()
        detections = detector.predict(image)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        total_detections += len(detections)
        print(f"[{idx}/{len(timed_images)}] {name}: {elapsed * 1000:.1f} ms, {len(detections)} detection(s)")

    avg_time = total_time / len(timed_images)
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    output = args.output
    if output is None:
        suffix = "gpu" if args.provider == "cuda" else "cpu"
        output = os.path.join(project_root, "results", f"onnx_benchmark_{suffix}.json")

    summary = {
        "model": args.model,
        "requested_provider": requested_provider,
        "session_providers": session_providers,
        "benchmark_scope": "preprocess + session.run + postprocess/NMS, excluding image file IO and drawing",
        "conf_thresh": args.conf,
        "iou_thresh": args.iou,
        "warmup_images": args.warmup,
        "timed_images": len(timed_images),
        "avg_time_ms": avg_time * 1000.0,
        "fps": fps,
        "avg_detections_per_image": total_detections / len(timed_images),
    }

    output_path = os.path.abspath(output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("ONNX Benchmark Summary")
    print("=" * 60)
    print(f"Provider:                  {requested_provider}")
    print(f"Session providers:         {session_providers}")
    print(f"Timed images:              {len(timed_images)}")
    print(f"Average latency:           {summary['avg_time_ms']:.1f} ms/image")
    print(f"Average FPS:               {summary['fps']:.2f}")
    print(f"Average detections/image:  {summary['avg_detections_per_image']:.2f}")
    print(f"Saved JSON:                {output_path}")


if __name__ == "__main__":
    main()

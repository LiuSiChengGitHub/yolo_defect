"""PyTorch vs ONNX 精度对比：逐图对比检测结果，验证 ONNX 导出一致性。"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

from src.detector import YOLODetector


def summarize_confidences(confidences):
    """统计一组置信度分布，便于做近似精度对比。"""
    if len(confidences) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

    arr = np.array(confidences, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Approximate comparison of PyTorch vs ONNX detection outputs"
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(project_root, "runs", "detect", "final_train_2", "weights", "best.pt"),
        help="Path to PyTorch weights (.pt)",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(project_root, "models", "best.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image-dir",
        default=os.path.join(project_root, "data", "images", "val"),
        help="Directory of images to sample from",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of images to compare (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Inference image size for PyTorch (default: 800)",
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
        help="PyTorch inference device, e.g. cpu or 0 (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "results", "pt_onnx_compare"),
        help="Directory to save CSV and JSON summary",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [
        os.path.join(args.image_dir, f)
        for f in sorted(os.listdir(args.image_dir))
        if f.lower().endswith(supported)
    ][: args.num_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.image_dir}")

    print(f"Loading PyTorch model: {args.weights}")
    pt_model = YOLO(args.weights)

    print(f"Loading ONNX model: {args.model}")
    onnx_detector = YOLODetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

    rows = []
    pt_all_conf = []
    onnx_all_conf = []

    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{idx}/{len(image_paths)}] skip unreadable image: {image_path}")
            continue

        pt_results = pt_model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        pt_boxes = pt_results[0].boxes
        pt_conf = pt_boxes.conf.detach().cpu().numpy().tolist() if pt_boxes is not None else []

        onnx_detections = onnx_detector.predict(image)
        onnx_conf = [det["confidence"] for det in onnx_detections]

        pt_all_conf.extend(pt_conf)
        onnx_all_conf.extend(onnx_conf)

        pt_count = len(pt_conf)
        onnx_count = len(onnx_conf)
        row = {
            "filename": os.path.basename(image_path),
            "pt_count": pt_count,
            "onnx_count": onnx_count,
            "count_diff": onnx_count - pt_count,
            "pt_mean_conf": float(np.mean(pt_conf)) if pt_conf else None,
            "onnx_mean_conf": float(np.mean(onnx_conf)) if onnx_conf else None,
            "pt_max_conf": float(np.max(pt_conf)) if pt_conf else None,
            "onnx_max_conf": float(np.max(onnx_conf)) if onnx_conf else None,
        }
        rows.append(row)

        print(
            f"[{idx}/{len(image_paths)}] {row['filename']}: "
            f"PT={pt_count}, ONNX={onnx_count}, diff={row['count_diff']}"
        )

    same_count_images = sum(1 for row in rows if row["pt_count"] == row["onnx_count"])
    abs_diffs = [abs(row["count_diff"]) for row in rows]
    summary = {
        "num_images": len(rows),
        "weights": args.weights,
        "onnx_model": args.model,
        "imgsz": args.imgsz,
        "conf_thresh": args.conf,
        "iou_thresh": args.iou,
        "pt_device": args.device,
        "same_count_images": int(same_count_images),
        "same_count_ratio": float(same_count_images / len(rows)) if rows else 0.0,
        "mean_abs_count_diff": float(np.mean(abs_diffs)) if rows else 0.0,
        "max_abs_count_diff": int(np.max(abs_diffs)) if rows else 0,
        "pt_total_detections": int(sum(row["pt_count"] for row in rows)),
        "onnx_total_detections": int(sum(row["onnx_count"] for row in rows)),
        "pt_confidence_summary": summarize_confidences(pt_all_conf),
        "onnx_confidence_summary": summarize_confidences(onnx_all_conf),
    }

    csv_path = os.path.join(output_dir, "compare_50_images.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "pt_count",
                "onnx_count",
                "count_diff",
                "pt_mean_conf",
                "onnx_mean_conf",
                "pt_max_conf",
                "onnx_max_conf",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(output_dir, "compare_50_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Approximate PyTorch vs ONNX Comparison Summary")
    print("=" * 60)
    print(f"Images compared:           {summary['num_images']}")
    print(f"Same detection count:      {summary['same_count_images']} / {summary['num_images']} "
          f"({summary['same_count_ratio']:.2%})")
    print(f"Mean abs count diff:       {summary['mean_abs_count_diff']:.3f}")
    print(f"Max abs count diff:        {summary['max_abs_count_diff']}")
    print(f"PT total detections:       {summary['pt_total_detections']}")
    print(f"ONNX total detections:     {summary['onnx_total_detections']}")
    print(f"PT confidence mean:        {summary['pt_confidence_summary']['mean']:.4f}")
    print(f"ONNX confidence mean:      {summary['onnx_confidence_summary']['mean']:.4f}")
    print(f"PT confidence median:      {summary['pt_confidence_summary']['median']:.4f}")
    print(f"ONNX confidence median:    {summary['onnx_confidence_summary']['median']:.4f}")
    print(f"Saved CSV:                 {csv_path}")
    print(f"Saved JSON:                {json_path}")


if __name__ == "__main__":
    main()

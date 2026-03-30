"""ONNX 推理脚本：支持单张和批量推理，输出标注图和 FPS 统计。"""

import argparse
import os
import time

import cv2

# 将项目根目录加入 Python 搜索路径，这样才能 import src.detector
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

from src.detector import YOLODetector

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def process_image(detector, image_path, output_dir):
    """对单张图片进行 ONNX 推理并保存结果。

    流程：读取图片 → 调用 detector.predict() → 画框 → 保存

    Args:
        detector: YOLODetector 实例
        image_path: 输入图片路径
        output_dir: 输出目录

    Returns:
        推理耗时（秒）
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Warning: cannot read {image_path}")
        return 0

    # 计时推理过程（不含图片读取和保存，只算模型推理）
    start = time.time()
    detections = detector.predict(image)
    elapsed = time.time() - start

    # 画框 + 类名 + 置信度
    result_img = detector.draw(image, detections, CLASS_NAMES)

    # 保存结果图片
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result_img)

    n_det = len(detections)
    print(f"  {filename}: {n_det} detection(s), {elapsed * 1000:.1f} ms")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="ONNX inference with YOLODetector"
    )
    parser.add_argument(
        "--model",
        default=os.path.join(project_root, "models", "best.onnx"),
        help="Path to ONNX model (default: models/best.onnx)",
    )
    parser.add_argument(
        "--image",
        help="Path to a single image for inference",
    )
    parser.add_argument(
        "--image-dir",
        help="Path to directory of images for batch inference",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "results"),
        help="Directory to save results (default: results/)",
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
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide --image or --image-dir")

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 加载 ONNX 检测器
    # YOLODetector 封装了 ONNX Runtime 的所有逻辑：加载模型 → 预处理 → 推理 → 后处理（NMS）→ 画框
    print(f"Loading model: {args.model}")
    detector = YOLODetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    )

    # 收集所有待处理图片
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_dir:
        supported = (".jpg", ".jpeg", ".png", ".bmp")
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith(supported):
                image_paths.append(os.path.join(args.image_dir, f))

    # 逐张推理
    print(f"Processing {len(image_paths)} image(s)...")
    total_time = 0
    for path in image_paths:
        total_time += process_image(detector, path, output_dir)

    # 计算平均 FPS
    if image_paths:
        avg_time = total_time / len(image_paths)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\nAverage: {avg_time * 1000:.1f} ms/image, {fps:.1f} FPS")
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

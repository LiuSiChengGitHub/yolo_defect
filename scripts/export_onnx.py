"""
export_onnx.py - Export trained YOLOv8 model to ONNX format.

Usage:
    python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
    python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt --output models/best.onnx
"""

import argparse
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX format"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (.pt file)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "models", "best.onnx"),
        help="Output ONNX path (default: models/best.onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX model (default: True)",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 1. 加载训练好的 PyTorch 模型
    # ============================================================
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # ============================================================
    # 2. 导出为 ONNX 格式
    # ============================================================
    # ONNX (Open Neural Network Exchange) 是什么？
    # - 开放的神经网络模型格式，由微软和 Facebook 联合推出
    # - 让模型在不同框架（PyTorch、TensorFlow）之间互通
    # - 部署时不需要安装 PyTorch，只需要轻量的 ONNX Runtime
    #
    # simplify=True 做了什么？
    # - 使用 onnx-simplifier 工具简化计算图
    # - 合并冗余节点、常量折叠、消除无用操作
    # - 通常能减小模型大小并提升推理速度
    #
    # imgsz 参数的作用：
    # - 导出时固定输入尺寸，ONNX 模型的输入形状是静态的
    # - 推理时必须用相同的 imgsz 预处理图片
    print(f"Exporting to ONNX (imgsz={args.imgsz}, simplify={args.simplify})...")
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
    )

    # 如果导出路径和目标路径不同，移动到指定位置
    if export_path and os.path.abspath(export_path) != os.path.abspath(args.output):
        import shutil
        shutil.move(export_path, args.output)
        export_path = args.output

    # 打印模型大小
    if export_path and os.path.exists(export_path):
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"\nExport complete!")
        print(f"  Path: {export_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"\nExport path: {export_path}")


if __name__ == "__main__":
    main()

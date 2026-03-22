"""
evaluate.py - Evaluate trained YOLOv8 model on validation set.

Generates PR curves, confusion matrix, and saves misdetection examples.

Usage:
    python scripts/evaluate.py --weights runs/detect/train/weights/best.pt
    python scripts/evaluate.py --weights runs/detect/train/weights/best.pt --save-failures 10
"""

import argparse
import os
import shutil

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 model on NEU-DET validation set"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to model weights (.pt file)",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(project_root, "data", "data.yaml"),
        help="Path to data.yaml (default: data/data.yaml)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation (default: 640)",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(project_root, "docs", "assets"),
        help="Directory to save evaluation plots (default: docs/assets)",
    )
    parser.add_argument(
        "--save-failures",
        type=int,
        default=10,
        help="Number of lowest-confidence predictions to save (default: 10)",
    )
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # ============================================================
    # 1. 加载训练好的模型
    # ============================================================
    # best.pt 是训练过程中 mAP 最高的那个 epoch 保存的权重
    # last.pt 是最后一个 epoch 的权重
    # 通常评估用 best.pt
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # ============================================================
    # 2. 在验证集上运行评估
    # ============================================================
    # model.val() 会：
    # - 遍历验证集所有图片
    # - 计算 mAP@0.5、mAP@50-95、Precision、Recall
    # - plots=True 会自动生成 PR 曲线、混淆矩阵等图表
    #
    # 面试考点：
    # - mAP@0.5：IoU 阈值=0.5 时的平均精度，宽松评估
    # - mAP@50-95：IoU 从 0.5 到 0.95 步长 0.05 的平均，严格评估
    # - Precision：检测到的目标中有多少是正确的（误检率的反面）
    # - Recall：所有真实目标中有多少被检测到了（漏检率的反面）
    print(f"Evaluating on: {args.data}")
    metrics = model.val(
        data=os.path.abspath(args.data),
        imgsz=args.imgsz,
        plots=True,  # 自动生成 PR 曲线、混淆矩阵等
    )

    # ============================================================
    # 3. 打印评估指标
    # ============================================================
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"mAP@0.5:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")

    # 逐类别 AP（Average Precision）
    # 面试考点：哪类缺陷最难检测？为什么？
    # 通常 crazing（龟裂）最难，因为纹理细密、与背景区分度低
    print(f"\n{'Class':<20} {'AP@0.5':>8} {'AP@50-95':>10}")
    print("-" * 40)
    class_names = model.names  # 从模型中获取类名映射
    for i, (ap50, ap) in enumerate(zip(metrics.box.ap50, metrics.box.ap)):
        name = class_names.get(i, f"class_{i}")
        print(f"{name:<20} {ap50:>8.4f} {ap:>10.4f}")

    # ============================================================
    # 4. 复制评估图表到 docs/assets/
    # ============================================================
    # ultralytics 会把评估图表保存到 runs/ 目录
    # 这里复制到 docs/assets/ 方便 README 引用和 GitHub 展示
    val_dir = metrics.save_dir if hasattr(metrics, "save_dir") else None
    if val_dir and os.path.isdir(val_dir):
        for plot_name in ["PR_curve.png", "confusion_matrix.png",
                          "F1_curve.png", "P_curve.png", "R_curve.png"]:
            src = os.path.join(val_dir, plot_name)
            if os.path.exists(src):
                dst = os.path.join(save_dir, plot_name)
                shutil.copy2(src, dst)
                print(f"Saved: {dst}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

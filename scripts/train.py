"""
train.py - Train YOLOv8 model on NEU-DET dataset.

Reads hyperparameters from a YAML config file and launches training
using the Ultralytics YOLO API.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
"""

import argparse
import os

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET dataset")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    parser.add_argument(
        "--config",
        default=os.path.join(project_root, "configs", "train_config.yaml"),
        help="Path to training config YAML (default: configs/train_config.yaml)",
    )
    args = parser.parse_args()

    # ============================================================
    # 1. 加载超参数配置
    # ============================================================
    # 用 YAML 文件管理超参数而不是命令行参数，好处：
    # - 每次实验的配置可以保存为文件，方便追溯和对比
    # - 不用记住一长串命令行参数
    # - 可以用 git diff 对比两次实验的参数差异
    config_path = os.path.abspath(args.config)
    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 提取模型名称（如 'yolov8n.pt'），从 config 字典中弹出
    # 因为 YOLO() 构造函数需要模型名，而 model.train() 不接受 'model' 参数
    model_name = config.pop("model", "yolov8n.pt")

    # 将相对路径转为绝对路径，确保从任何目录运行脚本都能找到文件
    if "data" in config:
        data_path = config["data"]
        if not os.path.isabs(data_path):
            config["data"] = os.path.join(project_root, data_path)

    if "project" in config:
        project_path = config["project"]
        if not os.path.isabs(project_path):
            config["project"] = os.path.join(project_root, project_path)

    print(f"Model:  {model_name}")
    print(f"Config: {config}")
    print()

    # ============================================================
    # 2. 加载模型并开始训练
    # ============================================================
    # YOLO(model_name) 做了什么：
    # - 如果是 'yolov8n.pt'：下载 COCO 预训练权重（第一次运行时）
    # - 加载模型架构 + 预训练权重（迁移学习的基础）
    #
    # model.train(**config) 做了什么：
    # - 按 config 里的超参数启动训练
    # - 自动进行数据增强（mosaic、mixup、翻转等）
    # - 每个 epoch 结束后在验证集上评估
    # - 自动保存 best.pt（最高 mAP）和 last.pt（最新 epoch）
    # - 训练日志和权重保存到 runs/detect/train/
    model = YOLO(model_name)
    results = model.train(**config)

    print("\nTraining complete!")
    print(f"Results saved to: {config.get('project', 'runs')}/{config.get('name', 'train')}")


if __name__ == "__main__":
    main()

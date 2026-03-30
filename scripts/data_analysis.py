"""数据分析：统计类别分布、bbox 尺寸分布和每图 bbox 数量，生成可视化图表。"""

import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


def extract_class_from_filename(filename):
    """从文件名中提取类别名（最长前缀匹配）。"""
    for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
        if filename.startswith(class_name + "_"):
            return class_name
    return None


def parse_annotations(ann_dir):
    """Parse all XML annotations in a directory.

    解析某个目录下所有 XML 标注文件，统计数据集信息。

    统计内容：
    - 每个类别的图片数量
    - 每张图有多少个 bbox（反映缺陷密度）
    - 每个 bbox 的宽高（反映缺陷尺寸分布）
    - 图片尺寸（检查是否统一）

    Returns:
        class_counts: 每类图片数
        bbox_per_image: 每张图的 bbox 数量列表
        bbox_sizes: 每个 bbox 的 (宽, 高) 列表
        img_sizes: 所有出现过的图片尺寸集合
    """
    class_counts = defaultdict(int)
    bbox_per_image = []
    bbox_sizes = []
    img_sizes = set()

    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])
    for xml_file in xml_files:
        stem = os.path.splitext(xml_file)[0]
        class_name = extract_class_from_filename(stem)
        if class_name:
            class_counts[class_name] += 1

        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()

        # 提取图片尺寸
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        img_sizes.add((w, h))

        # 统计每张图的 bbox 数量
        objects = root.findall("object")
        bbox_per_image.append(len(objects))

        # 统计每个 bbox 的宽高
        for obj in objects:
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bbox_sizes.append((xmax - xmin, ymax - ymin))

    return class_counts, bbox_per_image, bbox_sizes, img_sizes


def plot_class_distribution(train_counts, val_counts, save_dir):
    """绘制训练集/验证集的类别分布柱状图。。
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35  # 柱子宽度

    train_vals = [train_counts.get(c, 0) for c in CLASS_NAMES]
    val_vals = [val_counts.get(c, 0) for c in CLASS_NAMES]

    # 并排柱状图：蓝色=训练集，橙色=验证集
    bars1 = ax.bar(x - width / 2, train_vals, width, label="Train", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, val_vals, width, label="Validation", color="#DD8452")

    ax.set_xlabel("Defect Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("NEU-DET Class Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.legend()

    # 在每个柱子上方标注数值
    for bar in bars1:
        ax.annotate(str(int(bar.get_height())),
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(str(int(bar.get_height())),
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_bbox_per_image(bbox_per_image, save_dir):
    """绘制每张图 bbox 数量的直方图。
    如果一张图有很多 bbox，训练时需要更大的 max_det 参数。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    max_count = max(bbox_per_image) if bbox_per_image else 1
    bins = range(0, max_count + 2)
    ax.hist(bbox_per_image, bins=bins, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("Number of Bounding Boxes per Image")
    ax.set_ylabel("Frequency")
    ax.set_title("Bounding Box Count Distribution")
    ax.set_xticks(range(0, max_count + 2))
    plt.tight_layout()
    path = os.path.join(save_dir, "bbox_per_image.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_bbox_sizes(bbox_sizes, save_dir):
    """绘制 bbox 宽高散点图。
    - 如果目标很小（<32px），考虑用更大的 imgsz 或小目标增强策略
    - 如果目标接近整张图大小，说明缺陷覆盖面积大（如 crazing）
    - NEU-DET 原图只有 200x200，bbox 尺寸范围差异大
    """
    if not bbox_sizes:
        return
    widths, heights = zip(*bbox_sizes)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(widths, heights, alpha=0.3, s=10, color="#4C72B0")
    ax.set_xlabel("Bbox Width (pixels)")
    ax.set_ylabel("Bbox Height (pixels)")
    ax.set_title("Bounding Box Size Distribution")
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 210)
    ax.set_aspect("equal")  # 等比例坐标轴，方便直观理解
    plt.tight_layout()
    path = os.path.join(save_dir, "bbox_sizes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NEU-DET dataset distribution and statistics"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    parser.add_argument(
        "--data-root",
        default=os.path.join(project_root, "data", "NEU-DET"),
        help="Path to NEU-DET dataset root (default: data/NEU-DET)",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(project_root, "docs", "assets"),
        help="Directory to save plots (default: docs/assets)",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Save dir:  {save_dir}")
    print()

    # 分别解析训练集和验证集的标注
    train_ann = os.path.join(data_root, "train", "annotations")
    val_ann = os.path.join(data_root, "validation", "annotations")

    print("Parsing train annotations...")
    train_counts, train_bbox_per_img, train_bbox_sizes, train_img_sizes = \
        parse_annotations(train_ann)

    print("Parsing validation annotations...")
    val_counts, val_bbox_per_img, val_bbox_sizes, val_img_sizes = \
        parse_annotations(val_ann)

    # 合并两个子集的统计数据
    all_bbox_per_img = train_bbox_per_img + val_bbox_per_img
    all_bbox_sizes = train_bbox_sizes + val_bbox_sizes
    all_img_sizes = train_img_sizes | val_img_sizes  # 集合并集

    # 生成可视化图表
    print("\nGenerating plots...")
    plot_class_distribution(train_counts, val_counts, save_dir)
    plot_bbox_per_image(all_bbox_per_img, save_dir)
    plot_bbox_sizes(all_bbox_sizes, save_dir)

    # 打印文字摘要
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"Total images: {total_train + total_val} (train: {total_train}, val: {total_val})")
    print(f"Image sizes:  {all_img_sizes}")
    print(f"Classes:      {len(CLASS_NAMES)}")
    print()

    print(f"{'Class':<20} {'Train':>6} {'Val':>6} {'Total':>6}")
    print("-" * 40)
    for cls in CLASS_NAMES:
        t = train_counts.get(cls, 0)
        v = val_counts.get(cls, 0)
        print(f"{cls:<20} {t:>6} {v:>6} {t + v:>6}")

    print(f"\nBbox per image: min={min(all_bbox_per_img)}, "
          f"max={max(all_bbox_per_img)}, "
          f"mean={np.mean(all_bbox_per_img):.2f}")

    if all_bbox_sizes:
        widths, heights = zip(*all_bbox_sizes)
        print(f"Bbox width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
        print(f"Bbox height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()

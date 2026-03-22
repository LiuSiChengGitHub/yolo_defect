"""
prepare_data.py - Convert NEU-DET VOC XML annotations to YOLO TXT format.

This script reads the pre-split NEU-DET dataset (train/validation) and converts
VOC XML annotations to YOLO format (class_id cx cy w h, normalized 0-1).

Input:  data/NEU-DET/{train,validation}/{annotations,images}/
Output: data/{images,labels}/{train,val}/ + data/data.yaml

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-root data/NEU-DET --output-dir data
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET  # Python 内置 XML 解析库
from collections import defaultdict

# ============================================================
# 类别定义
# ============================================================
# NEU-DET 数据集共 6 类钢材表面缺陷
# 类别顺序决定了 class_id，YOLO 标签文件里用数字 0-5 代表
CLASS_NAMES = [
    "crazing",         # 0 - 龟裂：表面细密裂纹网络
    "inclusion",       # 1 - 夹杂：钢材内嵌入的异物
    "patches",         # 2 - 斑块：不规则变色区域
    "pitted_surface",  # 3 - 麻面：表面分布的小凹坑
    "rolled-in_scale", # 4 - 压入氧化铁皮：轧制时压入的氧化皮（注意名字里有连字符！）
    "scratches",       # 5 - 划痕：机械接触产生的线性痕迹
]
# 建立 类名→数字ID 的映射字典，方便后续查找
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def extract_class_from_filename(filename):
    """Extract class name from annotation filename using known class list.

    从标注文件名中提取类别名称。

    为什么不能简单用下划线分割？
    因为 'rolled-in_scale_1' 中的 'rolled-in_scale' 本身就包含下划线，
    如果按下划线分割会得到错误结果。
    所以用已知类名列表做前缀匹配，而且要按长度从长到短排序，
    确保 'rolled-in_scale' 优先于 'rolled' 被匹配到（最长匹配优先原则）。

    Args:
        filename: 不含扩展名的文件名，如 'rolled-in_scale_1'

    Returns:
        匹配到的类名字符串，匹配失败返回 None
    """
    # sorted(..., key=len, reverse=True) 确保长的类名优先匹配
    for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
        if filename.startswith(class_name + "_"):
            return class_name
    return None


def parse_voc_xml(xml_path):
    """Parse a VOC XML annotation file and extract bounding boxes.

    解析 VOC XML 标注文件，提取所有目标的边界框坐标。

    VOC XML 格式说明：
    - <size> 标签包含图片的宽(width)、高(height)、通道数(depth)
    - <object> 标签包含每个标注目标，一张图可以有多个 object
    - <bndbox> 包含 xmin/ymin/xmax/ymax，是绝对像素坐标（左上角和右下角）

    Args:
        xml_path: XML 标注文件路径

    Returns:
        boxes: [(类名, xmin, ymin, xmax, ymax), ...] 所有边界框列表
        img_size: (width, height) 图片尺寸
    """
    # 解析 XML 文件为树结构
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 提取图片尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # 遍历所有 <object> 标签，提取边界框
    boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text       # 类别名称
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)        # 左上角 x
        ymin = int(bbox.find("ymin").text)        # 左上角 y
        xmax = int(bbox.find("xmax").text)        # 右下角 x
        ymax = int(bbox.find("ymax").text)        # 右下角 y
        boxes.append((class_name, xmin, ymin, xmax, ymax))

    return boxes, (width, height)


def voc_to_yolo(boxes, img_size):
    """Convert VOC bounding boxes to YOLO format (normalized center x, y, w, h).

    VOC 格式 → YOLO 格式的坐标转换。

    VOC 格式：绝对像素坐标 (xmin, ymin, xmax, ymax)
    YOLO 格式：归一化的中心坐标 (class_id, cx, cy, w, h)

    转换公式：
        cx = (xmin + xmax) / 2 / img_width    # 中心点 x（归一化到 0-1）
        cy = (ymin + ymax) / 2 / img_height   # 中心点 y（归一化到 0-1）
        w  = (xmax - xmin) / img_width         # 框宽度（归一化到 0-1）
        h  = (ymax - ymin) / img_height        # 框高度（归一化到 0-1）

    为什么要归一化？
    归一化后坐标与图片尺寸无关，模型训练时可以自由缩放图片（如 200→640），
    标签值不需要跟着变。这是 YOLO 系列的标准做法。

    Args:
        boxes: [(类名, xmin, ymin, xmax, ymax), ...]
        img_size: (width, height)

    Returns:
        YOLO 格式字符串列表：['class_id cx cy w h', ...]
    """
    img_w, img_h = img_size
    yolo_lines = []
    for class_name, xmin, ymin, xmax, ymax in boxes:
        if class_name not in CLASS_TO_ID:
            print(f"  Warning: unknown class '{class_name}', skipping")
            continue
        class_id = CLASS_TO_ID[class_name]
        # 计算归一化的中心坐标和宽高
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return yolo_lines


def process_split(data_root, split_name, output_dir, yolo_split_name):
    """Process one data split (train or validation).

    处理一个数据子集（train 或 validation）。

    流程：
    1. 遍历 annotations/ 目录下所有 XML 文件
    2. 对每个 XML：解析 → 转换坐标 → 写 YOLO TXT 标签
    3. 同时把对应的图片从 images/{class_name}/ 复制到扁平目录 images/{train|val}/

    为什么图片要复制到扁平目录？
    YOLO 要求 images/ 和 labels/ 下的文件一一对应（同名不同后缀），
    且不能有子目录层级。所以需要把按类名分子目录的图片"拍平"到一个目录。

    Args:
        data_root: NEU-DET 数据集根目录 (data/NEU-DET)
        split_name: 源数据子目录名 ('train' 或 'validation')
        output_dir: 输出根目录 (data)
        yolo_split_name: 输出子集名 ('train' 或 'val')

    Returns:
        每个类别处理的图片数量统计字典
    """
    ann_dir = os.path.join(data_root, split_name, "annotations")
    img_base_dir = os.path.join(data_root, split_name, "images")

    # 创建输出目录
    out_img_dir = os.path.join(output_dir, "images", yolo_split_name)
    out_lbl_dir = os.path.join(output_dir, "labels", yolo_split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    stats = defaultdict(int)  # 用 defaultdict 统计每类图片数，省去初始化
    xml_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])

    for xml_file in xml_files:
        stem = os.path.splitext(xml_file)[0]  # 去掉 .xml 后缀，如 'crazing_1'

        # 从文件名提取类别（用前缀匹配而非下划线分割）
        class_name = extract_class_from_filename(stem)
        if class_name is None:
            print(f"  Warning: cannot determine class for '{xml_file}', skipping")
            continue

        # 解析 XML 标注文件
        xml_path = os.path.join(ann_dir, xml_file)
        boxes, img_size = parse_voc_xml(xml_path)

        # 转换为 YOLO 格式
        yolo_lines = voc_to_yolo(boxes, img_size)
        if not yolo_lines:
            continue

        # 写 YOLO 标签文件（每个 bbox 一行）
        label_path = os.path.join(out_lbl_dir, stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # 复制图片到扁平输出目录
        # 原始路径: images/{class_name}/{stem}.jpg → 目标路径: images/{split}/{stem}.jpg
        src_img = os.path.join(img_base_dir, class_name, stem + ".jpg")
        dst_img = os.path.join(out_img_dir, stem + ".jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)  # copy2 保留文件元数据
        else:
            print(f"  Warning: image not found: {src_img}")

        stats[class_name] += 1

    return stats


def generate_data_yaml(output_dir):
    """Generate YOLO data.yaml configuration file.

    生成 YOLO 数据集配置文件 data.yaml。
    这个文件告诉 YOLO 训练器：
    - 数据集在哪里 (path)
    - 训练集和验证集的相对路径
    - 有哪些类别、类别数量

    注意 path 用相对路径 '../data'，这样无论在哪台机器上都能正确定位。

    Args:
        output_dir: 输出根目录
    """
    yaml_content = (
        "# NEU-DET Steel Surface Defect Dataset - YOLO Format\n"
        "# Auto-generated by scripts/prepare_data.py\n\n"
        "path: ../data\n"              # 数据集根目录（相对于项目根目录）
        "train: images/train\n"        # 训练集图片路径（相对于 path）
        "val: images/val\n\n"          # 验证集图片路径（相对于 path）
        "names:\n"
    )
    for idx, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {idx}: {name}\n"
    yaml_content += f"\nnc: {len(CLASS_NAMES)}\n"  # nc = number of classes

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Generated {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NEU-DET VOC XML to YOLO TXT format"
    )
    # 通过脚本自身位置定位项目根目录，避免硬编码绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    parser.add_argument(
        "--data-root",
        default=os.path.join(project_root, "data", "NEU-DET"),
        help="Path to NEU-DET dataset root (default: data/NEU-DET)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "data"),
        help="Output directory for YOLO format data (default: data)",
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Data root:  {data_root}")
    print(f"Output dir: {output_dir}")
    print()

    # 处理训练集：train/ → images/train/ + labels/train/
    print("Processing train split...")
    train_stats = process_split(data_root, "train", output_dir, "train")

    # 处理验证集：validation/ → images/val/ + labels/val/
    # 注意源目录叫 "validation"，输出目录叫 "val"（YOLO 惯例）
    print("Processing validation split...")
    val_stats = process_split(data_root, "validation", output_dir, "val")

    # 生成 data.yaml 配置文件
    generate_data_yaml(output_dir)

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"{'Class':<20} {'Train':>6} {'Val':>6} {'Total':>6}")
    print("-" * 40)
    total_train, total_val = 0, 0
    for class_name in CLASS_NAMES:
        t = train_stats.get(class_name, 0)
        v = val_stats.get(class_name, 0)
        total_train += t
        total_val += v
        print(f"{class_name:<20} {t:>6} {v:>6} {t + v:>6}")
    print("-" * 40)
    print(f"{'Total':<20} {total_train:>6} {total_val:>6} {total_train + total_val:>6}")
    print()
    print(f"Output images: {output_dir}/images/{{train,val}}/")
    print(f"Output labels: {output_dir}/labels/{{train,val}}/")
    print(f"Data config:   {output_dir}/data.yaml")
    print("\nDone!")


if __name__ == "__main__":
    main()

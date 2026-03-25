"""
analyze_failures.py - Find and save typical false-positive / false-negative cases.

在验证集上逐张运行检测，基于“同类 + IoU 阈值”的规则匹配预测框与真值框，
然后筛出失败最明显的图片，保存到 docs/assets/ 方便做误检案例分析。

Usage:
    python scripts/analyze_failures.py --weights runs/detect/exp3_lr01/weights/best.pt
"""

import argparse
import os

import cv2
import yaml
from ultralytics import YOLO


def parse_args():
    """解析命令行参数。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    parser = argparse.ArgumentParser(
        description="Analyze false positives and false negatives on validation set"
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(
            project_root, "runs", "detect", "exp3_lr01", "weights", "best.pt"
        ),
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
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for prediction (default: 0.25)",
    )
    parser.add_argument(
        "--pred-iou",
        type=float,
        default=0.45,
        help="IoU threshold used by NMS during prediction (default: 0.45)",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching prediction to ground truth (default: 0.5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of failure cases to save (default: 10)",
    )
    parser.add_argument(
        "--tag",
        default="exp3_lr01",
        help="Tag used in output folder and summary file names",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "docs", "assets"),
        help="Output directory root (default: docs/assets)",
    )
    return parser.parse_args()


def resolve_dataset_paths(data_yaml_path):
    """从 data.yaml 解析验证集图片目录、标签目录和类名。"""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
    dataset_root = os.path.abspath(os.path.join(yaml_dir, data_cfg["path"]))
    val_rel = data_cfg["val"]

    val_image_dir = os.path.abspath(os.path.join(dataset_root, val_rel))
    val_label_rel = val_rel.replace("images", "labels", 1)
    val_label_dir = os.path.abspath(os.path.join(dataset_root, val_label_rel))

    names_cfg = data_cfg["names"]
    if isinstance(names_cfg, dict):
        class_names = [names_cfg[i] for i in sorted(names_cfg)]
    else:
        class_names = list(names_cfg)

    return val_image_dir, val_label_dir, class_names


def load_yolo_labels(label_path, image_w, image_h):
    """读取 YOLO TXT 标签并转成绝对坐标框。"""
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])

            x1 = (cx - w / 2.0) * image_w
            y1 = (cy - h / 2.0) * image_h
            x2 = (cx + w / 2.0) * image_w
            y2 = (cy + h / 2.0) * image_h

            labels.append(
                {
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return labels


def compute_iou(box_a, box_b):
    """计算两个框的 IoU。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_predictions(ground_truths, predictions, match_iou):
    """按“同类 + IoU>=阈值”匹配预测框与真值框。"""
    matched_pred_indices = set()
    matched_gt_indices = set()
    matches = []

    for gt_idx, gt in enumerate(ground_truths):
        best_pred_idx = None
        best_iou = 0.0

        for pred_idx, pred in enumerate(predictions):
            if pred_idx in matched_pred_indices:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue

            iou = compute_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        if best_pred_idx is not None and best_iou >= match_iou:
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(best_pred_idx)
            matches.append((gt_idx, best_pred_idx, best_iou))

    false_negatives = [
        {"gt_index": idx, **ground_truths[idx]}
        for idx in range(len(ground_truths))
        if idx not in matched_gt_indices
    ]
    false_positives = [
        {"pred_index": idx, **predictions[idx]}
        for idx in range(len(predictions))
        if idx not in matched_pred_indices
    ]

    return matches, false_negatives, false_positives


def collect_predictions(model, image_path, imgsz, conf, pred_iou):
    """调用 Ultralytics 模型获取单张图的预测结果。"""
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        iou=pred_iou,
        verbose=False,
    )

    detections = []
    if not results:
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    xyxy_list = boxes.xyxy.cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    cls_list = boxes.cls.cpu().tolist()

    for bbox, score, class_id in zip(xyxy_list, conf_list, cls_list):
        detections.append(
            {
                "class_id": int(class_id),
                "confidence": float(score),
                "bbox": bbox,
            }
        )

    return detections


def infer_case_reasons(false_negatives, false_positives, image_w, image_h, class_names):
    """根据失败模式给出初步原因假设，帮助后续人工分析。"""
    reasons = []

    if false_negatives:
        fn_classes = [class_names[item["class_id"]] for item in false_negatives]
        if any(name in {"crazing", "scratches"} for name in fn_classes):
            reasons.append("细纹理/细长结构对比度弱，特征不够稳定")

        max_area_ratio = 0.0
        for item in false_negatives:
            x1, y1, x2, y2 = item["bbox"]
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, image_w * image_h)
            max_area_ratio = max(max_area_ratio, area_ratio)
        if max_area_ratio > 0.45:
            reasons.append("缺陷范围大且边界弥散，框的定位目标不够明确")

    if false_positives:
        reasons.append("局部背景纹理与缺陷模式相似，产生误检")

    if len(false_negatives) + len(false_positives) >= 3:
        reasons.append("同图目标较多，定位与去重更容易出错")

    if not reasons:
        reasons.append("需要人工放大查看局部纹理，再判断是定位误差还是分类偏差")

    return reasons[:3]


def build_overlay_reason_lines(reason_lines):
    """将摘要里的中文原因压缩成图片上可显示的英文短语。"""
    mapping = {
        "细纹理/细长结构对比度弱，特征不够稳定": "weak fine texture",
        "缺陷范围大且边界弥散，框的定位目标不够明确": "diffuse boundary",
        "局部背景纹理与缺陷模式相似，产生误检": "background look-alike",
        "同图目标较多，定位与去重更容易出错": "crowded objects",
        "需要人工放大查看局部纹理，再判断是定位误差还是分类偏差": "needs manual zoom-in",
    }
    return [mapping.get(line, "manual inspection") for line in reason_lines]


def draw_failure_case(
    image,
    false_negatives,
    false_positives,
    class_names,
    case_title,
    reason_lines,
):
    """在图上绘制漏检/误检信息，生成案例图。"""
    canvas = image.copy()

    # 漏检真值框：橙色
    for item in false_negatives:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        class_name = class_names[item["class_id"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            canvas,
            f"FN GT:{class_name}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )

    # 误检预测框：红色
    for item in false_positives:
        x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
        class_name = class_names[item["class_id"]]
        score = item["confidence"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"FP {class_name}:{score:.2f}",
            (x1, min(canvas.shape[0] - 8, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # 顶部信息栏
    header_h = 124
    result = cv2.copyMakeBorder(
        canvas, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )
    cv2.putText(
        result,
        case_title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        "Orange=FN ground truth, Red=FP prediction",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    overlay_lines = build_overlay_reason_lines(reason_lines)
    for idx, line in enumerate(overlay_lines):
        cv2.putText(
            result,
            f"Reason {idx + 1}: {line}",
            (10, 68 + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    return result


def write_summary_markdown(summary_path, case_infos, class_names):
    """保存案例分析摘要，方便后续手工补充结论。"""
    lines = [
        "# Failure Case Analysis",
        "",
        "以下案例由脚本按“同类 + IoU>=0.5”的规则自动筛选，优先保留漏检/误检最多的验证集图片。",
        "",
        "| Case | Image | Failure Type | FN Classes | FP Classes | Initial Hypothesis |",
        "|------|-------|--------------|------------|------------|--------------------|",
    ]

    for item in case_infos:
        fn_classes = sorted({class_names[x["class_id"]] for x in item["false_negatives"]})
        fp_classes = sorted({class_names[x["class_id"]] for x in item["false_positives"]})
        failure_type = []
        if item["false_negatives"]:
            failure_type.append(f"FN x{len(item['false_negatives'])}")
        if item["false_positives"]:
            failure_type.append(f"FP x{len(item['false_positives'])}")
        lines.append(
            f"| {item['case_name']} | {item['image_name']} | "
            f"{' + '.join(failure_type)} | "
            f"{', '.join(fn_classes) if fn_classes else '-'} | "
            f"{', '.join(fp_classes) if fp_classes else '-'} | "
            f"{'；'.join(item['reason_lines'])} |"
        )

    lines.extend(
        [
            "",
            "## 人工分析建议",
            "",
            "1. 先看橙色 FN 框：确认是完全没框出来，还是只框到局部。",
            "2. 再看红色 FP 框：判断是不是背景纹理、边缘反光、相邻缺陷导致的误报。",
            "3. 结合类别特性写原因：纹理细、边界弥散、目标密集、标签宽泛、分辨率不足。",
            "4. 最后给出改进方向：调 imgsz、延长 epochs、换模型、调 cls、做针对性增强。",
            "",
        ]
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    weights_path = os.path.abspath(args.weights)
    data_yaml_path = os.path.abspath(args.data)
    output_root = os.path.abspath(args.output_dir)
    output_dir = os.path.join(output_root, f"failure_cases_{args.tag}")
    os.makedirs(output_dir, exist_ok=True)

    val_image_dir, val_label_dir, class_names = resolve_dataset_paths(data_yaml_path)

    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    image_names = sorted(
        name for name in os.listdir(val_image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )

    case_infos = []
    for image_name in image_names:
        image_path = os.path.join(val_image_dir, image_name)
        label_path = os.path.join(
            val_label_dir, os.path.splitext(image_name)[0] + ".txt"
        )

        image = cv2.imread(image_path)
        if image is None:
            continue

        image_h, image_w = image.shape[:2]
        ground_truths = load_yolo_labels(label_path, image_w, image_h)
        predictions = collect_predictions(
            model=model,
            image_path=image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            pred_iou=args.pred_iou,
        )

        matches, false_negatives, false_positives = match_predictions(
            ground_truths=ground_truths,
            predictions=predictions,
            match_iou=args.match_iou,
        )

        if not false_negatives and not false_positives:
            continue

        failure_score = len(false_negatives) * 2 + len(false_positives)
        reason_lines = infer_case_reasons(
            false_negatives=false_negatives,
            false_positives=false_positives,
            image_w=image_w,
            image_h=image_h,
            class_names=class_names,
        )

        case_infos.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "matches": matches,
                "failure_score": failure_score,
                "reason_lines": reason_lines,
            }
        )

    # 优先保留失败更多、尤其漏检更多的图片
    case_infos.sort(
        key=lambda item: (
            item["failure_score"],
            len(item["false_negatives"]),
            len(item["false_positives"]),
        ),
        reverse=True,
    )
    selected_cases = case_infos[: args.top_k]

    print(f"Found {len(case_infos)} failure images, saving top {len(selected_cases)} cases.")
    for idx, item in enumerate(selected_cases, start=1):
        image = cv2.imread(item["image_path"])
        case_name = f"case_{idx:02d}"
        case_title = (
            f"{case_name} | {item['image_name']} | "
            f"FN={len(item['false_negatives'])} FP={len(item['false_positives'])}"
        )
        vis = draw_failure_case(
            image=image,
            false_negatives=item["false_negatives"],
            false_positives=item["false_positives"],
            class_names=class_names,
            case_title=case_title,
            reason_lines=item["reason_lines"],
        )
        save_path = os.path.join(output_dir, f"{case_name}_{item['image_name']}")
        cv2.imwrite(save_path, vis)
        item["case_name"] = case_name
        print(f"Saved: {save_path}")

    summary_path = os.path.join(output_dir, f"failure_summary_{args.tag}.md")
    write_summary_markdown(summary_path, selected_cases, class_names)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

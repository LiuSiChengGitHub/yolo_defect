“””代表性样例筛选：每类自动选取正确/错误案例各 1 张，用于 README 展示。”””

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.insert(0, project_root)

from src.detector import YOLODetector


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Select representative success/failure examples from ONNX predictions"
    )
    parser.add_argument(
        "--model",
        default=os.path.join(project_root, "models", "best.onnx"),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(project_root, "data", "data.yaml"),
        help="Path to data.yaml",
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
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for matching prediction to GT (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "docs", "assets", "representative_examples"),
        help="Directory to save selected examples",
    )
    return parser.parse_args()


def resolve_dataset_paths(data_yaml_path):
    """从 data.yaml 解析验证集图片、标签和类名。"""
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
    """读取 YOLO TXT 标签并转为绝对坐标。"""
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
            labels.append({"class_id": class_id, "bbox": [x1, y1, x2, y2]})
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
    """按“同类 + IoU>=阈值”匹配预测框与 GT。"""
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


def build_case_metrics(gt_boxes, predictions, matches, false_negatives, false_positives, target_class_id):
    """提取当前图片对某个类的成功/失败特征。"""
    gt_target = [item for item in gt_boxes if item["class_id"] == target_class_id]
    matched_target = []
    matched_ious = []
    matched_conf = []

    for gt_idx, pred_idx, iou in matches:
        if gt_boxes[gt_idx]["class_id"] == target_class_id:
            matched_target.append((gt_idx, pred_idx))
            matched_ious.append(iou)
            matched_conf.append(predictions[pred_idx]["confidence"])

    fn_target = [item for item in false_negatives if item["class_id"] == target_class_id]
    fp_target = [item for item in false_positives if item["class_id"] == target_class_id]
    fp_other = [item for item in false_positives if item["class_id"] != target_class_id]

    success = (
        len(gt_target) > 0
        and len(fn_target) == 0
        and len(fp_target) == 0
        and len(fp_other) == 0
    )
    error = len(gt_target) > 0 and (len(fn_target) > 0 or len(fp_target) > 0 or len(fp_other) > 0)

    success_score = (
        len(matched_target),
        float(np.mean(matched_ious)) if matched_ious else 0.0,
        float(np.mean(matched_conf)) if matched_conf else 0.0,
    )
    error_score = (
        len(fn_target) * 10
        + len(fp_target) * 4
        + len(fp_other) * 3,
        len(fn_target),
        len(fp_target) + len(fp_other),
    )

    reasons = []
    if len(fn_target) > 0:
        reasons.append(f"FN x{len(fn_target)}")
    if len(fp_target) > 0:
        reasons.append(f"same-class FP x{len(fp_target)}")
    if len(fp_other) > 0:
        wrong_names = sorted({item["class_id"] for item in fp_other})
        reasons.append(f"wrong-class FP x{len(fp_other)}")
        reasons.append("contains wrong labels")
    if not reasons and success:
        reasons.append("all GT matched")
        reasons.append("no obvious extra boxes")

    return {
        "success": success,
        "error": error,
        "success_score": success_score,
        "error_score": error_score,
        "fn_target": fn_target,
        "fp_target": fp_target,
        "fp_other": fp_other,
        "matched_ious": matched_ious,
        "matched_conf": matched_conf,
        "reasons": reasons,
    }


def draw_case_visualization(image, gt_boxes, predictions, matches, false_negatives, false_positives, class_names, title_lines):
    """生成统一风格的代表图。"""
    vis = image.copy()

    matched_pred_indices = {pred_idx for _, pred_idx, _ in matches}
    matched_gt_indices = {gt_idx for gt_idx, _, _ in matches}

    # 先画所有 GT：匹配成功的用绿色，未匹配的用橙色
    for gt_idx, gt in enumerate(gt_boxes):
        x1, y1, x2, y2 = [int(v) for v in gt["bbox"]]
        class_name = class_names[gt["class_id"]]
        color = (0, 180, 0) if gt_idx in matched_gt_indices else (0, 165, 255)
        prefix = "GT" if gt_idx in matched_gt_indices else "FN GT"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{prefix}:{class_name}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    # 再画预测框：匹配成功的用青色，未匹配的 FP 用红色
    for pred_idx, pred in enumerate(predictions):
        x1, y1, x2, y2 = [int(v) for v in pred["bbox"]]
        class_name = class_names[pred["class_id"]]
        conf = pred["confidence"]
        color = (255, 255, 0) if pred_idx in matched_pred_indices else (0, 0, 255)
        prefix = "Pred" if pred_idx in matched_pred_indices else "FP"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            vis,
            f"{prefix}:{class_name} {conf:.2f}",
            (x1, min(vis.shape[0] - 8, y2 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

    header_h = 86
    canvas = cv2.copyMakeBorder(
        vis, header_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
    )

    for idx, line in enumerate(title_lines):
        cv2.putText(
            canvas,
            line,
            (10, 24 + idx * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 if idx == 0 else 0.44,
            (30, 30, 30),
            1 if idx > 0 else 2,
            cv2.LINE_AA,
        )

    return canvas


def make_grid(image_paths, save_path, cell_w=420, cell_h=286):
    """把 12 张图拼成一个 3x4 总览图，便于快速确认。"""
    rows = 4
    cols = 3
    canvas = np.full((rows * cell_h, cols * cell_w, 3), 248, dtype=np.uint8)

    for idx, image_path in enumerate(image_paths[: rows * cols]):
        image = cv2.imread(image_path)
        if image is None:
            continue

        image = cv2.resize(image, (cell_w, cell_h))
        row = idx // cols
        col = idx % cols
        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w
        canvas[y1:y2, x1:x2] = image

    cv2.imwrite(save_path, canvas)


def main():
    args = parse_args()

    model_path = os.path.abspath(args.model)
    data_yaml_path = os.path.abspath(args.data)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    val_image_dir, val_label_dir, class_names = resolve_dataset_paths(data_yaml_path)
    detector = YOLODetector(model_path, conf_thresh=args.conf, iou_thresh=args.iou)

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    image_names = sorted(
        name for name in os.listdir(val_image_dir) if name.lower().endswith(supported)
    )

    best_success = {class_id: None for class_id in range(len(class_names))}
    best_error = {class_id: None for class_id in range(len(class_names))}

    for image_name in image_names:
        image_path = os.path.join(val_image_dir, image_name)
        label_path = os.path.join(val_label_dir, os.path.splitext(image_name)[0] + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            continue

        image_h, image_w = image.shape[:2]
        gt_boxes = load_yolo_labels(label_path, image_w, image_h)
        if not gt_boxes:
            continue

        predictions = detector.predict(image)
        matches, false_negatives, false_positives = match_predictions(
            gt_boxes, predictions, args.match_iou
        )

        gt_class_ids = sorted({item["class_id"] for item in gt_boxes})
        for class_id in gt_class_ids:
            metrics = build_case_metrics(
                gt_boxes, predictions, matches, false_negatives, false_positives, class_id
            )

            candidate = {
                "image_name": image_name,
                "image_path": image_path,
                "gt_boxes": gt_boxes,
                "predictions": predictions,
                "matches": matches,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "metrics": metrics,
            }

            if metrics["success"]:
                current = best_success[class_id]
                if current is None or metrics["success_score"] > current["metrics"]["success_score"]:
                    best_success[class_id] = candidate

            if metrics["error"]:
                current = best_error[class_id]
                if current is None or metrics["error_score"] > current["metrics"]["error_score"]:
                    best_error[class_id] = candidate

    saved_paths = []
    summary = {"classes": {}}

    for class_id, class_name in enumerate(class_names):
        summary["classes"][class_name] = {}

        success_case = best_success[class_id]
        if success_case is not None:
            title_lines = [
                f"{class_name} | correct | {success_case['image_name']}",
                "Green GT=matched, Cyan Pred=matched",
                f"Reason: {', '.join(success_case['metrics']['reasons'])}",
            ]
            success_img = draw_case_visualization(
                cv2.imread(success_case["image_path"]),
                success_case["gt_boxes"],
                success_case["predictions"],
                success_case["matches"],
                success_case["false_negatives"],
                success_case["false_positives"],
                class_names,
                title_lines,
            )
            success_path = os.path.join(output_dir, f"{class_name}_correct_{success_case['image_name']}")
            cv2.imwrite(success_path, success_img)
            saved_paths.append(success_path)
            summary["classes"][class_name]["correct"] = {
                "image": success_case["image_name"],
                "reason": success_case["metrics"]["reasons"],
            }

        error_case = best_error[class_id]
        if error_case is not None:
            title_lines = [
                f"{class_name} | error | {error_case['image_name']}",
                "Orange GT=FN, Red Pred=FP, Green/Cyan=matched",
                f"Reason: {', '.join(error_case['metrics']['reasons'])}",
            ]
            error_img = draw_case_visualization(
                cv2.imread(error_case["image_path"]),
                error_case["gt_boxes"],
                error_case["predictions"],
                error_case["matches"],
                error_case["false_negatives"],
                error_case["false_positives"],
                class_names,
                title_lines,
            )
            error_path = os.path.join(output_dir, f"{class_name}_error_{error_case['image_name']}")
            cv2.imwrite(error_path, error_img)
            saved_paths.append(error_path)
            summary["classes"][class_name]["error"] = {
                "image": error_case["image_name"],
                "reason": error_case["metrics"]["reasons"],
            }

    grid_path = os.path.join(output_dir, "representative_examples_grid.jpg")
    make_grid(saved_paths, grid_path)

    summary["grid"] = os.path.basename(grid_path)
    summary_path = os.path.join(output_dir, "representative_examples_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Representative examples saved to:")
    print(output_dir)
    print(f"Summary: {summary_path}")
    print(f"Grid:    {grid_path}")
    for class_name, items in summary["classes"].items():
        correct = items.get("correct", {}).get("image", "-")
        error = items.get("error", {}).get("image", "-")
        print(f"{class_name:<18} correct={correct:<20} error={error}")


if __name__ == "__main__":
    main()

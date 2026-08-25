#!/usr/bin/env python3
"""Evaluate S2-01 FP32/INT8 runtime, product, and task-quality correctness.

The command deliberately separates three questions:

* can Python ORT (and optionally the C++ CLI) load and run both artifacts;
* how much do product detections differ at score=0.25/NMS=0.45;
* how much does task quality differ on the frozen 361-image validation set.

It reuses ``compare_consistency.py`` for contract loading, preprocessing,
postprocessing, deterministic product matching, C++ JSON validation, and the
strict Python/C++ implementation-consistency tolerances.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)


TOOLS_ROOT = Path(__file__).resolve().parent
CPP_INFER_ROOT = TOOLS_ROOT.parent
REPO_ROOT = CPP_INFER_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import generate_s2_01_manifests as protocol  # noqa: E402
import s2_01_protocol as machine_protocol  # noqa: E402


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "s2_01_fp32_int8_correctness_and_quality"


class EvaluationError(RuntimeError):
    """S2-01 correctness evaluation cannot continue safely."""


def fail(object_name: str, expected: str, actual: str, action: str) -> None:
    raise EvaluationError(
        f"S2-01 validation failed: object {object_name}; expected {expected}; "
        f"actual {actual}; action: {action}"
    )


def reject_duplicate_keys(
    pairs: Iterable[Tuple[str, Any]],
) -> MutableMapping[str, Any]:
    result: MutableMapping[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(
                f"json.field[{key}]",
                "each JSON field exactly once",
                "duplicate field",
                "remove the duplicate declaration",
            )
        result[key] = value
    return result


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file, object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(
            f"json.path='{path}'",
            "one readable UTF-8 JSON object",
            str(error),
            "fix the path, encoding, or JSON syntax",
        )
    if not isinstance(value, dict):
        fail(
            f"json.path='{path}'",
            "an object at the document root",
            type(value).__name__,
            "replace the document root",
        )
    return value


def validate_exact_fields(
    value: Mapping[str, Any], expected: Iterable[str], object_name: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    if missing or unknown:
        fail(
            object_name,
            f"exact fields {sorted(expected_set)}",
            f"missing={missing}, unknown={unknown}",
            "regenerate the frozen manifest",
        )


def resolve_declared_file(
    declaration_path: Path, raw_path: Any, object_name: str
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        fail(object_name, "a non-empty relative path", repr(raw_path), "fix the manifest")
    try:
        resolved = (declaration_path.parent / raw_path).resolve(strict=True)
    except OSError as error:
        fail(object_name, "an existing file", str(error), "restore the frozen input")
    if not resolved.is_file():
        fail(object_name, "an existing regular file", str(resolved), "restore the input")
    return resolved


def validate_hash(raw_hash: Any, object_name: str) -> str:
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        fail(object_name, "64 hexadecimal characters", repr(raw_hash), "fix the manifest")
    try:
        int(raw_hash, 16)
    except ValueError:
        fail(object_name, "64 hexadecimal characters", repr(raw_hash), "fix the manifest")
    return raw_hash.upper()


def validate_classes(raw_classes: Any) -> List[Mapping[str, Any]]:
    expected = protocol.class_entries()
    if raw_classes != expected:
        fail(
            "manifest.classes",
            repr(expected),
            repr(raw_classes),
            "preserve artifact class ids and order",
        )
    return expected


def validate_protocol_fields(manifest: Mapping[str, Any], kind: str) -> None:
    common = {
        "schema_version",
        "manifest_kind",
        "manifest_id",
        "dataset",
        "preprocess",
        "classes",
        "samples",
        "integrity",
    }
    if kind == "quality":
        common |= {
            "evaluation",
            "product_matching_protocol",
            "product_matching_gates",
            "quality_gates",
        }
    validate_exact_fields(manifest, common, f"{kind}_manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        fail(
            f"{kind}_manifest.schema_version",
            str(SCHEMA_VERSION),
            repr(manifest["schema_version"]),
            "use the supported schema",
        )
    expected_kind = (
        "static_ptq_calibration" if kind == "calibration" else "detection_task_quality"
    )
    expected_id = (
        protocol.CALIBRATION_MANIFEST_ID
        if kind == "calibration"
        else protocol.QUALITY_MANIFEST_ID
    )
    if manifest["manifest_kind"] != expected_kind or manifest["manifest_id"] != expected_id:
        fail(
            f"{kind}_manifest.identity",
            repr((expected_kind, expected_id)),
            repr((manifest["manifest_kind"], manifest["manifest_id"])),
            "use the frozen S2-01 manifest",
        )
    validate_classes(manifest["classes"])


def load_frozen_manifest(path: Path, kind: str) -> Mapping[str, Any]:
    manifest_path = path.resolve(strict=True)
    manifest = load_json(manifest_path)
    validate_protocol_fields(manifest, kind)
    samples = manifest["samples"]
    expected_count = 180 if kind == "calibration" else 361
    if not isinstance(samples, list) or len(samples) != expected_count:
        fail(
            f"{kind}_manifest.samples",
            f"exactly {expected_count} entries",
            f"count={len(samples) if isinstance(samples, list) else 'not-a-list'}",
            "regenerate the manifest before quantization/evaluation",
        )

    sample_fields = {
        "sample_id",
        "source_class_id",
        "source_class_name",
        "image_path",
        "image_sha256",
    }
    if kind == "quality":
        sample_fields |= {
            "label_path",
            "label_canonical_lf_sha256",
            "ground_truth_box_count",
        }
    resolved_samples: List[Mapping[str, Any]] = []
    seen: Dict[str, set] = {
        "sample_id": set(),
        "image_path": set(),
        "image_sha256": set(),
    }
    class_counts: collections.Counter = collections.Counter()
    gt_class_counts: collections.Counter = collections.Counter()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail(
                f"{kind}_manifest.samples[{index}]",
                "an object",
                type(sample).__name__,
                "regenerate the manifest",
            )
        validate_exact_fields(sample, sample_fields, f"{kind}_manifest.samples[{index}]")
        sample_id = sample["sample_id"]
        source_class_id = sample["source_class_id"]
        source_class_name = sample["source_class_name"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(source_class_id, int)
            or not 0 <= source_class_id < len(protocol.CLASS_NAMES)
            or source_class_name != protocol.CLASS_NAMES[source_class_id]
            or not sample_id.startswith(source_class_name + "_")
        ):
            fail(
                f"{kind}_manifest.samples[{index}].identity",
                "a sample id aligned with the declared source class",
                repr((sample_id, source_class_id, source_class_name)),
                "regenerate the manifest",
            )
        image_path = resolve_declared_file(
            manifest_path, sample["image_path"], f"{kind}.samples[{index}].image_path"
        )
        expected_image_hash = validate_hash(
            sample["image_sha256"], f"{kind}.samples[{index}].image_sha256"
        )
        actual_image_hash = protocol.sha256_file(image_path)
        if actual_image_hash != expected_image_hash:
            fail(
                f"{kind}.samples[{index}].image_sha256",
                expected_image_hash,
                actual_image_hash,
                "restore the image or create a new manifest version",
            )
        current_values = {
            "sample_id": sample_id,
            "image_path": image_path,
            "image_sha256": expected_image_hash,
        }
        for field, value in current_values.items():
            if value in seen[field]:
                fail(
                    f"{kind}.samples[{index}].{field}",
                    "a unique value",
                    str(value),
                    "remove duplicate calibration/evaluation inputs",
                )
            seen[field].add(value)

        resolved: Dict[str, Any] = {
            **sample,
            "image_sha256": expected_image_hash,
            "resolved_image_path": image_path,
        }
        if kind == "quality":
            label_path = resolve_declared_file(
                manifest_path,
                sample["label_path"],
                f"quality.samples[{index}].label_path",
            )
            expected_label_hash = validate_hash(
                sample["label_canonical_lf_sha256"],
                f"quality.samples[{index}].label_canonical_lf_sha256",
            )
            actual_label_hash = protocol.canonical_lf_sha256(label_path)
            if actual_label_hash != expected_label_hash:
                fail(
                    f"quality.samples[{index}].label_canonical_lf_sha256",
                    expected_label_hash,
                    actual_label_hash,
                    "restore the label or create a new manifest version",
                )
            boxes = protocol.parse_yolo_label(label_path)
            if sample["ground_truth_box_count"] != len(boxes):
                fail(
                    f"quality.samples[{index}].ground_truth_box_count",
                    str(len(boxes)),
                    repr(sample["ground_truth_box_count"]),
                    "regenerate the manifest",
                )
            gt_class_counts.update(box["class_id"] for box in boxes)
            resolved.update(
                {
                    "label_canonical_lf_sha256": expected_label_hash,
                    "resolved_label_path": label_path,
                    "normalized_ground_truth": boxes,
                }
            )
        class_counts[source_class_id] += 1
        resolved_samples.append(resolved)

    expected_source_counts = (
        {class_id: 30 for class_id in range(6)}
        if kind == "calibration"
        else {0: 61, 1: 60, 2: 60, 3: 60, 4: 60, 5: 60}
    )
    if dict(class_counts) != expected_source_counts:
        fail(
            f"{kind}.source_class_counts",
            repr(expected_source_counts),
            repr(dict(class_counts)),
            "regenerate the frozen manifest",
        )
    if kind == "quality":
        expected_gt = {0: 165, 1: 159, 2: 193, 3: 87, 4: 132, 5: 121}
        if dict(gt_class_counts) != expected_gt:
            fail(
                "quality.ground_truth_class_counts",
                repr(expected_gt),
                repr(dict(gt_class_counts)),
                "restore the frozen validation labels",
            )
        if manifest["product_matching_protocol"] != dict(
            protocol.PRODUCT_MATCHING_PROTOCOL
        ):
            fail(
                "quality.product_matching_protocol",
                repr(dict(protocol.PRODUCT_MATCHING_PROTOCOL)),
                repr(manifest["product_matching_protocol"]),
                "do not change assignment, tie-break, or acceptance semantics after quantization",
            )
        if manifest["product_matching_gates"] != dict(protocol.PRODUCT_MATCHING_GATES):
            fail(
                "quality.product_matching_gates",
                repr(dict(protocol.PRODUCT_MATCHING_GATES)),
                repr(manifest["product_matching_gates"]),
                "do not relax gates after quantization",
            )
        if manifest["quality_gates"] != dict(protocol.QUALITY_GATES):
            fail(
                "quality.quality_gates",
                repr(dict(protocol.QUALITY_GATES)),
                repr(manifest["quality_gates"]),
                "do not relax gates after quantization",
            )
        expected_evaluation = {
            "quality_score_floor": protocol.QUALITY_SCORE_FLOOR,
            "product_score_threshold": protocol.PRODUCT_SCORE_THRESHOLD,
            "nms_threshold": protocol.NMS_THRESHOLD,
            "nms_mode": "class_agnostic",
            "iou_thresholds": list(protocol.IOU_THRESHOLDS),
            "ap_interpolation": "COCO_101_point_precision_envelope",
            "max_detections_per_image": None,
            "metric_claim": "COCO_style_101_point_without_area_ranges_or_max_dets",
        }
        for field, expected in expected_evaluation.items():
            if manifest["evaluation"].get(field) != expected:
                fail(
                    f"quality.evaluation.{field}",
                    repr(expected),
                    repr(manifest["evaluation"].get(field)),
                    "use the predeclared quality protocol",
                )

    rows = protocol.sample_rows(samples, kind)
    actual_sample_set_hash = protocol.canonical_rows_sha256(rows)
    declared_sample_set_hash = validate_hash(
        manifest["integrity"].get("sample_set_sha256"),
        f"{kind}.integrity.sample_set_sha256",
    )
    if actual_sample_set_hash != declared_sample_set_hash:
        fail(
            f"{kind}.integrity.sample_set_sha256",
            declared_sample_set_hash,
            actual_sample_set_hash,
            "regenerate the manifest",
        )
    return {
        **manifest,
        "manifest_path": manifest_path,
        "manifest_canonical_lf_sha256": protocol.canonical_lf_sha256(manifest_path),
        "resolved_samples": resolved_samples,
    }


def load_product_manifest(path: Path) -> Mapping[str, Any]:
    manifest_path = path.resolve(strict=True)
    manifest = load_json(manifest_path)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 30:
        fail(
            "product_manifest.samples",
            "the frozen 30-image consistency sample list",
            repr(len(samples) if isinstance(samples, list) else type(samples).__name__),
            "use consistency_manifest.json",
        )
    resolved_samples = []
    seen_ids = set()
    seen_paths = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail("product_manifest.sample", "an object", type(sample).__name__, "fix the manifest")
        sample_id = sample.get("sample_id")
        image_path = resolve_declared_file(
            manifest_path, sample.get("image_path"), f"product.samples[{index}].image_path"
        )
        expected_hash = validate_hash(
            sample.get("image_sha256"), f"product.samples[{index}].image_sha256"
        )
        actual_hash = protocol.sha256_file(image_path)
        if actual_hash != expected_hash:
            fail(
                f"product.samples[{index}].image_sha256",
                expected_hash,
                actual_hash,
                "restore the frozen product sample",
            )
        if not isinstance(sample_id, str) or sample_id in seen_ids or image_path in seen_paths:
            fail(
                f"product.samples[{index}]",
                "unique sample ids and image paths",
                repr((sample_id, image_path)),
                "fix the manifest",
            )
        seen_ids.add(sample_id)
        seen_paths.add(image_path)
        resolved_samples.append(
            {**sample, "image_sha256": expected_hash, "resolved_image_path": image_path}
        )
    return {
        **manifest,
        "manifest_path": manifest_path,
        "manifest_canonical_lf_sha256": protocol.canonical_lf_sha256(manifest_path),
        "resolved_samples": resolved_samples,
    }


def load_consistency_tool() -> Any:
    tool_path = TOOLS_ROOT / "compare_consistency.py"
    spec = importlib.util.spec_from_file_location("s2_compare_consistency", tool_path)
    if spec is None or spec.loader is None:
        fail(
            "compare_consistency.import",
            f"an importable module at {tool_path}",
            repr(spec),
            "restore the Stage One consistency tool",
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def continuous_iou(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    intersection_width = max(0.0, min(lhs[2], rhs[2]) - max(lhs[0], rhs[0]))
    intersection_height = max(0.0, min(lhs[3], rhs[3]) - max(lhs[1], rhs[1]))
    intersection = intersection_width * intersection_height
    lhs_area = max(0.0, lhs[2] - lhs[0]) * max(0.0, lhs[3] - lhs[1])
    rhs_area = max(0.0, rhs[2] - rhs[0]) * max(0.0, rhs[3] - rhs[1])
    union = lhs_area + rhs_area - intersection
    return 0.0 if union <= 0.0 else min(1.0, max(0.0, intersection / union))


def percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def class_histogram(detections: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts = collections.Counter(int(item["class_id"]) for item in detections)
    return {str(class_id): counts[class_id] for class_id in sorted(counts)}


def product_difference(
    fp32_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    int8_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    match_detections: Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    gates: Mapping[str, float],
) -> Mapping[str, Any]:
    if set(fp32_by_sample) != set(int8_by_sample):
        fail(
            "product.samples",
            "identical FP32/INT8 sample ids",
            repr((sorted(fp32_by_sample), sorted(int8_by_sample))),
            "run both artifacts on the frozen product manifest",
        )
    per_image = []
    accepted_matches = []
    fp32_total = 0
    int8_total = 0
    fp32_class_counts: collections.Counter = collections.Counter()
    int8_class_counts: collections.Counter = collections.Counter()
    for sample_id in fp32_by_sample:
        fp32 = list(fp32_by_sample[sample_id])
        int8 = list(int8_by_sample[sample_id])
        fp32_total += len(fp32)
        int8_total += len(int8)
        fp32_class_counts.update(int(item["class_id"]) for item in fp32)
        int8_class_counts.update(int(item["class_id"]) for item in int8)
        raw_matching = match_detections(fp32, int8)
        match_records = [
            {
                "class_id": int(item["class_id"]),
                "fp32_index": int(item["python_index"]),
                "int8_index": int(item["cpp_index"]),
                "fp32_detection": item["python_detection"],
                "int8_detection": item["cpp_detection"],
                "confidence_abs_error": float(item["confidence_abs_error"]),
                "bbox_coordinate_abs_errors": [
                    float(value) for value in item["bbox_coordinate_abs_errors"]
                ],
                "bbox_coordinate_abs_error_max": float(
                    item["bbox_coordinate_abs_error_max"]
                ),
                "matching_iou": float(item["matching_iou"]),
                "accepted": float(item["matching_iou"])
                >= gates["pair_iou_min"],
            }
            for item in raw_matching["matches"]
        ]
        matched = [
            item
            for item in raw_matching["matches"]
            if float(item["matching_iou"]) >= gates["pair_iou_min"]
        ]
        matched_fp32_indices = {int(item["python_index"]) for item in matched}
        matched_int8_indices = {int(item["cpp_index"]) for item in matched}
        accepted_matches.extend(matched)
        per_image.append(
            {
                "sample_id": sample_id,
                "fp32_detection_count": len(fp32),
                "int8_detection_count": len(int8),
                "fp32_class_counts": class_histogram(fp32),
                "int8_class_counts": class_histogram(int8),
                "matched_detection_count": len(matched),
                "unmatched_fp32_indices": sorted(set(range(len(fp32))) - matched_fp32_indices),
                "unmatched_int8_indices": sorted(set(range(len(int8))) - matched_int8_indices),
                "match_records": match_records,
                "unmatched_fp32_detections": [
                    {"index": index, "detection": fp32[index]}
                    for index in sorted(set(range(len(fp32))) - matched_fp32_indices)
                ],
                "unmatched_int8_detections": [
                    {"index": index, "detection": int8[index]}
                    for index in sorted(set(range(len(int8))) - matched_int8_indices)
                ],
                "min_matched_iou": min(
                    (float(item["matching_iou"]) for item in matched), default=None
                ),
                "max_confidence_abs_error": max(
                    (float(item["confidence_abs_error"]) for item in matched),
                    default=None,
                ),
            }
        )
    ious = [float(item["matching_iou"]) for item in accepted_matches]
    confidence_errors = [
        float(item["confidence_abs_error"]) for item in accepted_matches
    ]
    coordinate_errors = [
        float(error)
        for item in accepted_matches
        for error in item["bbox_coordinate_abs_errors"]
    ]
    match_count = len(accepted_matches)
    fp32_retention = match_count / fp32_total if fp32_total else (1.0 if not int8_total else 0.0)
    int8_precision = match_count / int8_total if int8_total else (1.0 if not fp32_total else 0.0)
    metrics = {
        "images_total": len(fp32_by_sample),
        "fp32_detections_total": fp32_total,
        "int8_detections_total": int8_total,
        "matched_detections_total": match_count,
        "fp32_retention": fp32_retention,
        "int8_agreement_precision": int8_precision,
        "matched_iou_mean": mean_or_none(ious),
        "matched_iou_p05": percentile(ious, 0.05),
        "matched_iou_min": min(ious) if ious else None,
        "confidence_abs_error_mean": mean_or_none(confidence_errors),
        "confidence_abs_error_p95": percentile(confidence_errors, 0.95),
        "confidence_abs_error_max": max(confidence_errors) if confidence_errors else None,
        "bbox_coordinate_abs_error_pixels_mean": mean_or_none(coordinate_errors),
        "bbox_coordinate_abs_error_pixels_p95": percentile(coordinate_errors, 0.95),
        "bbox_coordinate_abs_error_pixels_max": max(coordinate_errors) if coordinate_errors else None,
        "fp32_class_counts": {str(key): fp32_class_counts[key] for key in sorted(fp32_class_counts)},
        "int8_class_counts": {str(key): int8_class_counts[key] for key in sorted(int8_class_counts)},
    }
    checks = {
        "fp32_retention": fp32_retention >= gates["fp32_retention_min"],
        "int8_agreement_precision": int8_precision >= gates["int8_agreement_precision_min"],
        "matched_mean_iou": metrics["matched_iou_mean"] is not None
        and metrics["matched_iou_mean"] >= gates["matched_mean_iou_min"],
        "matched_iou_p05": metrics["matched_iou_p05"] is not None
        and metrics["matched_iou_p05"] >= gates["matched_iou_p05_min"],
        "confidence_abs_error_mean": metrics["confidence_abs_error_mean"] is not None
        and metrics["confidence_abs_error_mean"]
        <= gates["confidence_abs_error_mean_max"],
        "confidence_abs_error_p95": metrics["confidence_abs_error_p95"] is not None
        and metrics["confidence_abs_error_p95"]
        <= gates["confidence_abs_error_p95_max"],
    }
    return {
        "passed": all(checks.values()),
        "matching_protocol": dict(protocol.PRODUCT_MATCHING_PROTOCOL),
        "gates": dict(gates),
        "checks": checks,
        "metrics": metrics,
        "per_image": per_image,
    }


def ap_101_point(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    class_id: int,
    iou_threshold: float,
) -> Tuple[float, int, int]:
    gt_for_class: Dict[str, List[Mapping[str, Any]]] = {
        sample_id: [box for box in boxes if int(box["class_id"]) == class_id]
        for sample_id, boxes in ground_truth.items()
    }
    gt_total = sum(len(boxes) for boxes in gt_for_class.values())
    class_predictions = sorted(
        (item for item in predictions if int(item["class_id"]) == class_id),
        key=lambda item: (
            -float(item["confidence"]),
            str(item["sample_id"]),
            *(float(value) for value in item["bbox_xyxy"]),
        ),
    )
    if gt_total == 0:
        return 0.0, 0, len(class_predictions)
    matched: Dict[str, set] = collections.defaultdict(set)
    true_positives: List[int] = []
    false_positives: List[int] = []
    for prediction in class_predictions:
        sample_id = str(prediction["sample_id"])
        candidates = gt_for_class.get(sample_id, [])
        ranked = sorted(
            (
                (continuous_iou(prediction["bbox_xyxy"], gt["bbox_xyxy"]), index)
                for index, gt in enumerate(candidates)
                if index not in matched[sample_id]
            ),
            key=lambda value: (-value[0], value[1]),
        )
        if ranked and ranked[0][0] >= iou_threshold:
            matched[sample_id].add(ranked[0][1])
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls: List[float] = []
    precisions: List[float] = []
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / gt_total)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    sampled_precision = []
    for recall_target_index in range(101):
        recall_target = recall_target_index / 100.0
        sampled_precision.append(
            max(
                (
                    precision
                    for recall, precision in zip(recalls, precisions)
                    if recall >= recall_target
                ),
                default=0.0,
            )
        )
    return sum(sampled_precision) / 101.0, gt_total, len(class_predictions)


def fixed_operating_point_metrics(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    score_threshold: float,
    iou_threshold: float = 0.50,
) -> Mapping[str, Any]:
    filtered = [
        item for item in predictions if float(item["confidence"]) > score_threshold
    ]
    true_positive = 0
    false_positive = 0
    gt_total = sum(len(boxes) for boxes in ground_truth.values())
    matched: Dict[Tuple[str, int], set] = collections.defaultdict(set)
    for prediction in sorted(
        filtered,
        key=lambda item: (
            -float(item["confidence"]),
            str(item["sample_id"]),
            int(item["class_id"]),
            *(float(value) for value in item["bbox_xyxy"]),
        ),
    ):
        sample_id = str(prediction["sample_id"])
        class_id = int(prediction["class_id"])
        key = (sample_id, class_id)
        candidates = [
            (index, gt)
            for index, gt in enumerate(ground_truth.get(sample_id, []))
            if int(gt["class_id"]) == class_id and index not in matched[key]
        ]
        ranked = sorted(
            (
                (continuous_iou(prediction["bbox_xyxy"], gt["bbox_xyxy"]), index)
                for index, gt in candidates
            ),
            key=lambda value: (-value[0], value[1]),
        )
        if ranked and ranked[0][0] >= iou_threshold:
            matched[key].add(ranked[0][1])
            true_positive += 1
        else:
            false_positive += 1
    false_negative = gt_total - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / gt_total if gt_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "score_threshold": score_threshold,
        "matching_iou_threshold": iou_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_quality_metrics(
    predictions_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    ground_truth_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    class_names: Sequence[str],
) -> Mapping[str, Any]:
    predictions = [
        {**prediction, "sample_id": sample_id}
        for sample_id, sample_predictions in predictions_by_sample.items()
        for prediction in sample_predictions
    ]
    per_class = []
    all_ap50 = []
    all_ap50_95 = []
    for class_id, class_name in enumerate(class_names):
        aps = []
        gt_total = 0
        prediction_total = 0
        for threshold in protocol.IOU_THRESHOLDS:
            ap, current_gt_total, current_prediction_total = ap_101_point(
                predictions, ground_truth_by_sample, class_id, threshold
            )
            aps.append(ap)
            gt_total = current_gt_total
            prediction_total = current_prediction_total
        ap50 = aps[0]
        ap50_95 = sum(aps) / len(aps)
        all_ap50.append(ap50)
        all_ap50_95.append(ap50_95)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth_boxes": gt_total,
                "predictions_at_quality_floor": prediction_total,
                "ap50": ap50,
                "ap50_95": ap50_95,
                "ap_by_iou": {
                    f"{threshold:.2f}": ap
                    for threshold, ap in zip(protocol.IOU_THRESHOLDS, aps)
                },
            }
        )
    return {
        "images_total": len(ground_truth_by_sample),
        "ground_truth_boxes_total": sum(
            len(boxes) for boxes in ground_truth_by_sample.values()
        ),
        "predictions_at_quality_floor_total": len(predictions),
        "map50": sum(all_ap50) / len(all_ap50),
        "map50_95": sum(all_ap50_95) / len(all_ap50_95),
        "per_class": per_class,
        "product_operating_point": fixed_operating_point_metrics(
            predictions,
            ground_truth_by_sample,
            protocol.PRODUCT_SCORE_THRESHOLD,
        ),
    }


def quality_comparison(
    fp32: Mapping[str, Any], int8: Mapping[str, Any], gates: Mapping[str, float]
) -> Mapping[str, Any]:
    comparison_epsilon = 1.0e-12
    fp32_per_class = {item["class_id"]: item for item in fp32["per_class"]}
    int8_per_class = {item["class_id"]: item for item in int8["per_class"]}
    per_class_deltas = []
    for class_id in sorted(fp32_per_class):
        fp32_item = fp32_per_class[class_id]
        int8_item = int8_per_class[class_id]
        per_class_deltas.append(
            {
                "class_id": class_id,
                "class_name": fp32_item["class_name"],
                "ap50_delta": int8_item["ap50"] - fp32_item["ap50"],
                "ap50_95_delta": int8_item["ap50_95"] - fp32_item["ap50_95"],
            }
        )
    deltas = {
        "map50_delta": int8["map50"] - fp32["map50"],
        "map50_95_delta": int8["map50_95"] - fp32["map50_95"],
        "per_class": per_class_deltas,
    }
    checks = {
        "map50_95_drop": deltas["map50_95_delta"] + comparison_epsilon
        >= -gates["map50_95_absolute_drop_max"],
        "map50_drop": deltas["map50_delta"] + comparison_epsilon
        >= -gates["map50_absolute_drop_max"],
        "per_class_ap50_drop": all(
            item["ap50_delta"] + comparison_epsilon
            >= -gates["per_class_ap50_absolute_drop_max"]
            for item in per_class_deltas
        ),
    }
    return {
        "passed": all(checks.values()),
        "gates": dict(gates),
        "checks": checks,
        "deltas": deltas,
        "fp32": fp32,
        "int8": int8,
    }


def ground_truth_xyxy(
    normalized_boxes: Sequence[Mapping[str, Any]], width: int, height: int
) -> List[Mapping[str, Any]]:
    result = []
    for box in normalized_boxes:
        center_x = float(box["center_x"]) * width
        center_y = float(box["center_y"]) * height
        box_width = float(box["width"]) * width
        box_height = float(box["height"]) * height
        result.append(
            {
                "class_id": int(box["class_id"]),
                "bbox_xyxy": [
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                ],
            }
        )
    return result


def validate_contract_pair(
    fp32: Mapping[str, Any], int8: Mapping[str, Any]
) -> None:
    common_fields = (
        "input_name",
        "input_shape",
        "output_name",
        "output_shape",
        "class_names",
        "nms_mode",
    )
    for field in common_fields:
        if fp32[field] != int8[field]:
            fail(
                f"contract.{field}",
                repr(fp32[field]),
                repr(int8[field]),
                "align the INT8 artifact with the FP32 source contract",
            )
    for name, contract in (("fp32", fp32), ("int8", int8)):
        if not math.isclose(
            float(contract["score_threshold"]),
            protocol.PRODUCT_SCORE_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            fail(
                f"{name}.score_threshold",
                str(protocol.PRODUCT_SCORE_THRESHOLD),
                repr(contract["score_threshold"]),
                "use the frozen product operating point",
            )
        if not math.isclose(
            float(contract["nms_threshold"]),
            protocol.NMS_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            fail(
                f"{name}.nms_threshold",
                str(protocol.NMS_THRESHOLD),
                repr(contract["nms_threshold"]),
                "use the frozen NMS protocol",
            )


def assert_optional_path(
    actual: Path, declared: Optional[Path], object_name: str
) -> None:
    if declared is None:
        return
    try:
        expected = declared.resolve(strict=True)
    except OSError as error:
        fail(object_name, "an existing explicitly supplied path", str(error), "fix the CLI path")
    if actual.resolve() != expected:
        fail(
            object_name,
            str(expected),
            str(actual.resolve()),
            "pass the config/artifact/model that belong to one contract",
        )


def run_python_product(
    consistency: Any,
    contract: Mapping[str, Any],
    session: Any,
    samples: Sequence[Mapping[str, Any]],
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    results = {}
    for sample in samples:
        tensor, transform = consistency.preprocess_image(
            sample["resolved_image_path"], contract["input_shape"]
        )
        raw_outputs = session.run(
            [contract["output_name"]], {contract["input_name"]: tensor}
        )
        if len(raw_outputs) != 1:
            fail(
                f"python.product.{sample['sample_id']}.outputs",
                "one output",
                str(len(raw_outputs)),
                "request only the contract output",
            )
        results[sample["sample_id"]] = consistency.postprocess_raw_output(
            raw_outputs[0], contract, transform
        )
    return results


def run_python_quality(
    consistency: Any,
    contract: Mapping[str, Any],
    session: Any,
    samples: Sequence[Mapping[str, Any]],
) -> Tuple[
    Mapping[str, Sequence[Mapping[str, Any]]],
    Mapping[str, Sequence[Mapping[str, Any]]],
]:
    quality_contract = {
        **contract,
        "score_threshold": protocol.QUALITY_SCORE_FLOOR,
        "nms_threshold": protocol.NMS_THRESHOLD,
    }
    predictions = {}
    ground_truth = {}
    for sample in samples:
        tensor, transform = consistency.preprocess_image(
            sample["resolved_image_path"], contract["input_shape"]
        )
        raw_outputs = session.run(
            [contract["output_name"]], {contract["input_name"]: tensor}
        )
        if len(raw_outputs) != 1:
            fail(
                f"python.quality.{sample['sample_id']}.outputs",
                "one output",
                str(len(raw_outputs)),
                "request only the contract output",
            )
        predictions[sample["sample_id"]] = consistency.postprocess_raw_output(
            raw_outputs[0], quality_contract, transform
        )
        ground_truth[sample["sample_id"]] = ground_truth_xyxy(
            sample["normalized_ground_truth"],
            int(transform["original_width"]),
            int(transform["original_height"]),
        )
    return predictions, ground_truth


def strict_python_cpp_comparison(
    consistency: Any,
    python_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    cpp_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    per_image = []
    all_matches = []
    for sample_id in python_by_sample:
        python_detections = list(python_by_sample[sample_id])
        cpp_detections = list(cpp_by_sample[sample_id])
        matching = consistency.match_detections(python_detections, cpp_detections)
        failures = []
        if len(python_detections) != len(cpp_detections):
            failures.append("detection_count")
        if class_histogram(python_detections) != class_histogram(cpp_detections):
            failures.append("class_histogram")
        if matching["unmatched_python_indices"] or matching["unmatched_cpp_indices"]:
            failures.append("unmatched_detection")
        for match in matching["matches"]:
            if match["confidence_abs_error"] > consistency.FROZEN_REQUIREMENTS["confidence_abs_error_max"]:
                failures.append("confidence_abs_error")
            if match["bbox_coordinate_abs_error_max"] > consistency.FROZEN_REQUIREMENTS["bbox_coordinate_abs_error_max_pixels"]:
                failures.append("bbox_coordinate_abs_error")
            if match["matching_iou"] < consistency.FROZEN_REQUIREMENTS["matching_iou_min"]:
                failures.append("matching_iou")
        all_matches.extend(matching["matches"])
        per_image.append(
            {
                "sample_id": sample_id,
                "passed": not failures,
                "failures": sorted(set(failures)),
                "python_detection_count": len(python_detections),
                "cpp_detection_count": len(cpp_detections),
            }
        )
    return {
        "passed": all(item["passed"] for item in per_image),
        "requirements": dict(consistency.FROZEN_REQUIREMENTS),
        "images_total": len(per_image),
        "images_passed": sum(1 for item in per_image if item["passed"]),
        "matched_detections_total": len(all_matches),
        "max_confidence_abs_error": max(
            (float(item["confidence_abs_error"]) for item in all_matches),
            default=None,
        ),
        "max_bbox_coordinate_abs_error_pixels": max(
            (
                float(item["bbox_coordinate_abs_error_max"])
                for item in all_matches
            ),
            default=None,
        ),
        "min_matching_iou": min(
            (float(item["matching_iou"]) for item in all_matches), default=None
        ),
        "per_image": per_image,
    }


def run_cpp_validation(
    consistency: Any,
    cpp_cli: Path,
    contracts: Mapping[str, Mapping[str, Any]],
    python_results: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    samples: Sequence[Mapping[str, Any]],
    temporary_parent: Path,
) -> Mapping[str, Any]:
    inspections = {
        name: consistency.inspect_cpp_cli(cpp_cli, contract["config_path"])
        for name, contract in contracts.items()
    }
    cpp_results: Dict[str, Dict[str, Sequence[Mapping[str, Any]]]] = {
        name: {} for name in contracts
    }
    temporary_prefix = f".s2_01_cpp_{os.getpid()}_"
    generated_paths: List[Path] = []
    try:
        for name, contract in contracts.items():
            for sample in samples:
                output_path = temporary_parent / (
                    f"{temporary_prefix}{name}_{sample['sample_id']}.json"
                )
                if output_path.exists():
                    fail(
                        "cpp.temporary_output",
                        "an unused process-specific path",
                        str(output_path),
                        "remove the stale file after confirming no evaluator is running",
                    )
                generated_paths.append(output_path)
                raw_json = consistency.run_cpp_image(
                    cpp_cli,
                    contract,
                    sample["resolved_image_path"],
                    output_path,
                )
                _, transform = consistency.preprocess_image(
                    sample["resolved_image_path"], contract["input_shape"]
                )
                cpp_results[name][sample["sample_id"]] = consistency.validate_cpp_json(
                    raw_json, contract, sample["resolved_image_path"], transform
                )
    finally:
        for generated_path in generated_paths:
            if generated_path.is_file():
                generated_path.unlink()
    comparisons = {
        name: strict_python_cpp_comparison(
            consistency, python_results[name], cpp_results[name]
        )
        for name in contracts
    }
    return {
        "requested": True,
        "passed": all(item["passed"] for item in comparisons.values()),
        "cli_path": str(cpp_cli.resolve()),
        "inspections": inspections,
        "python_cpp_consistency": comparisons,
    }


def contract_evidence(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "config_path": str(contract["config_path"]),
        "config_canonical_lf_sha256": protocol.canonical_lf_sha256(
            contract["config_path"]
        ),
        "artifact_path": str(contract["artifact_path"]),
        "artifact_canonical_lf_sha256": protocol.canonical_lf_sha256(
            contract["artifact_path"]
        ),
        "model_path": str(contract["model_path"]),
        "model_id": contract["model_id"],
        "model_sha256": contract["model_actual_sha256"],
        "model_size_bytes": contract["model_path"].stat().st_size,
        "input_name": contract["input_name"],
        "input_shape": contract["input_shape"],
        "output_name": contract["output_name"],
        "output_shape": contract["output_shape"],
        "class_names": contract["class_names"],
        "score_threshold": contract["score_threshold"],
        "nms_threshold": contract["nms_threshold"],
        "nms_mode": contract["nms_mode"],
    }


def manifest_evidence(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "manifest_id": manifest["manifest_id"],
        "path": str(manifest["manifest_path"]),
        "canonical_lf_sha256": manifest["manifest_canonical_lf_sha256"],
        "sample_set_sha256": manifest["integrity"]["sample_set_sha256"],
        "sample_count": len(manifest["resolved_samples"]),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    serialized = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialized)
    temporary_path.replace(path)


def run_evaluation(arguments: argparse.Namespace) -> Mapping[str, Any]:
    frozen_protocol = machine_protocol.load_s2_01_protocol(arguments.protocol)
    calibration_manifest = load_frozen_manifest(
        arguments.calibration_manifest, "calibration"
    )
    quality_manifest = load_frozen_manifest(arguments.quality_manifest, "quality")
    for object_name, actual_path, expected_path in (
        (
            "calibration_manifest",
            calibration_manifest["manifest_path"],
            frozen_protocol.calibration_manifest_path,
        ),
        (
            "quality_manifest",
            quality_manifest["manifest_path"],
            frozen_protocol.quality_manifest_path,
        ),
    ):
        if actual_path != expected_path:
            fail(
                f"protocol.{object_name}",
                str(expected_path),
                str(actual_path),
                "evaluate the exact manifest validated before quantization",
            )
    calibration_hashes = {
        sample["image_sha256"] for sample in calibration_manifest["resolved_samples"]
    }
    quality_hashes = {
        sample["image_sha256"] for sample in quality_manifest["resolved_samples"]
    }
    if calibration_hashes & quality_hashes:
        fail(
            "manifest.split_leakage",
            "no calibration/quality image SHA overlap",
            repr(sorted(calibration_hashes & quality_hashes)[:3]),
            "use train-only calibration and validation-only quality data",
        )
    product_manifest = load_product_manifest(arguments.product_manifest)
    if product_manifest["manifest_path"] != frozen_protocol.consistency_manifest_path:
        fail(
            "protocol.product_manifest",
            str(frozen_protocol.consistency_manifest_path),
            str(product_manifest["manifest_path"]),
            "evaluate the exact product manifest frozen before quantization",
        )

    consistency = load_consistency_tool()
    consistency.require_dependencies()
    fp32_contract = consistency.load_contract(arguments.fp32_config)
    int8_contract = consistency.load_contract(arguments.int8_config)
    validate_contract_pair(fp32_contract, int8_contract)
    assert_optional_path(
        fp32_contract["artifact_path"], arguments.fp32_artifact, "fp32.artifact_path"
    )
    assert_optional_path(
        int8_contract["artifact_path"], arguments.int8_artifact, "int8.artifact_path"
    )
    assert_optional_path(fp32_contract["model_path"], arguments.fp32_model, "fp32.model_path")
    assert_optional_path(int8_contract["model_path"], arguments.int8_model, "int8.model_path")
    if fp32_contract["model_path"] != frozen_protocol.source_model_path:
        fail(
            "protocol.fp32_model",
            str(frozen_protocol.source_model_path),
            str(fp32_contract["model_path"]),
            "select the frozen FP32 source artifact",
        )
    if int8_contract["model_path"] != frozen_protocol.output_model_path:
        fail(
            "protocol.int8_model",
            str(frozen_protocol.output_model_path),
            str(int8_contract["model_path"]),
            "select the INT8 model derived by the frozen PTQ protocol",
        )

    fp32_session = consistency.create_python_session(fp32_contract)
    int8_session = consistency.create_python_session(int8_contract)
    product_samples = product_manifest["resolved_samples"]
    python_product = {
        "fp32": run_python_product(
            consistency, fp32_contract, fp32_session, product_samples
        ),
        "int8": run_python_product(
            consistency, int8_contract, int8_session, product_samples
        ),
    }
    product_result = product_difference(
        python_product["fp32"],
        python_product["int8"],
        consistency.match_detections,
        quality_manifest["product_matching_gates"],
    )

    fp32_quality_predictions, ground_truth = run_python_quality(
        consistency,
        fp32_contract,
        fp32_session,
        quality_manifest["resolved_samples"],
    )
    int8_quality_predictions, int8_ground_truth = run_python_quality(
        consistency,
        int8_contract,
        int8_session,
        quality_manifest["resolved_samples"],
    )
    if ground_truth != int8_ground_truth:
        fail(
            "quality.ground_truth",
            "identical ground truth for both artifacts",
            "different decoded values",
            "reuse one frozen manifest and preprocess geometry",
        )
    fp32_quality = compute_quality_metrics(
        fp32_quality_predictions, ground_truth, fp32_contract["class_names"]
    )
    int8_quality = compute_quality_metrics(
        int8_quality_predictions, ground_truth, int8_contract["class_names"]
    )
    quality_result = quality_comparison(
        fp32_quality, int8_quality, quality_manifest["quality_gates"]
    )

    if arguments.cpp_cli is None:
        cpp_validation = {
            "requested": False,
            "passed": False,
            "status": "not_requested; formal S2-01 evidence requires --cpp-cli",
        }
    else:
        cpp_cli = arguments.cpp_cli.resolve(strict=True)
        if not cpp_cli.is_file():
            fail("cpp_cli", "an existing executable file", str(cpp_cli), "build the Release CLI")
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        cpp_validation = run_cpp_validation(
            consistency,
            cpp_cli,
            {"fp32": fp32_contract, "int8": int8_contract},
            python_product,
            product_samples,
            arguments.output_json.parent.resolve(),
        )

    passed = (
        product_result["passed"]
        and quality_result["passed"]
        and bool(cpp_validation["passed"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "passed": passed,
        "runtime_legality": {
            "python_fp32_session_and_finite_outputs": True,
            "python_int8_session_and_finite_outputs": True,
            "cpp": cpp_validation,
        },
        "protocol": {
            "protocol_id": frozen_protocol.protocol_id,
            "path": str(frozen_protocol.declaration_path),
            "canonical_lf_sha256": machine_protocol.sha256_file_canonical_lf(
                frozen_protocol.declaration_path
            ),
            "quality_score_floor": protocol.QUALITY_SCORE_FLOOR,
            "product_score_threshold": protocol.PRODUCT_SCORE_THRESHOLD,
            "nms_threshold": protocol.NMS_THRESHOLD,
            "nms_mode": "class_agnostic",
            "iou_thresholds": list(protocol.IOU_THRESHOLDS),
            "ap_interpolation": "COCO_101_point_precision_envelope",
            "profiler_or_benchmark_enabled": False,
        },
        "manifests": {
            "calibration": manifest_evidence(calibration_manifest),
            "quality": manifest_evidence(quality_manifest),
            "product": {
                "manifest_id": product_manifest.get("manifest_id"),
                "path": str(product_manifest["manifest_path"]),
                "canonical_lf_sha256": product_manifest[
                    "manifest_canonical_lf_sha256"
                ],
                "sample_count": len(product_samples),
            },
            "calibration_quality_image_sha_overlap_count": 0,
        },
        "artifacts": {
            "fp32": contract_evidence(fp32_contract),
            "int8": contract_evidence(int8_contract),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "onnxruntime_version": consistency.ort.__version__,
            "opencv_version": consistency.cv2.__version__,
            "numpy_version": consistency.np.__version__,
            "platform": platform.platform(),
            "provider": consistency.CPU_PROVIDER,
            "execution_mode": "sequential",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "graph_optimization_level": "all",
        },
        "product_detection_difference": product_result,
        "task_quality": quality_result,
        "limitations": [
            "The task metric is an ONNX-only relative FP32/INT8 comparison; the matching .pt checkpoint is unavailable.",
            "The frozen set contains 361 current tracked validation images and intentionally ignores the stale data/labels/val.cache file.",
            "This evaluator uses class-agnostic NMS=0.45 to preserve the product contract, so values are not asserted to equal historical Ultralytics validation metrics.",
            "All YOLO TXT boxes are counted; VOC difficult/truncated flags were not preserved by the existing conversion.",
        ],
    }


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    fixture_root = CPP_INFER_ROOT / "tests" / "fixtures"
    parser = argparse.ArgumentParser(
        description="Compare frozen FP32/INT8 product detections and task quality"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Frozen S2-01 machine protocol used by the PTQ run",
    )
    parser.add_argument(
        "--fp32-config",
        type=Path,
        default=CPP_INFER_ROOT / "configs" / "default_config.txt",
    )
    parser.add_argument("--int8-config", type=Path, required=True)
    parser.add_argument("--fp32-artifact", type=Path)
    parser.add_argument("--int8-artifact", type=Path)
    parser.add_argument("--fp32-model", type=Path)
    parser.add_argument("--int8-model", type=Path)
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=fixture_root / "s2_01_calibration_manifest.json",
    )
    parser.add_argument(
        "--quality-manifest",
        type=Path,
        default=fixture_root / "s2_01_quality_manifest.json",
    )
    parser.add_argument(
        "--product-manifest",
        type=Path,
        default=fixture_root / "consistency_manifest.json",
    )
    parser.add_argument(
        "--cpp-cli",
        type=Path,
        required=True,
        help="Release C++ CLI; required for formal session/finite/product validation",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        evidence = run_evaluation(arguments)
    except Exception as error:
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "evidence_type": EVIDENCE_TYPE,
            "passed": False,
            "setup_error": str(error),
        }
    try:
        write_json(arguments.output_json.resolve(), evidence)
    except Exception as write_error:
        print(f"Could not write evidence: {write_error}", file=sys.stderr)
        return 1
    print(
        f"S2-01 correctness evidence: passed={evidence.get('passed')}; "
        f"output={arguments.output_json.resolve()}"
    )
    if "setup_error" in evidence:
        print(evidence["setup_error"], file=sys.stderr)
    return 0 if evidence.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

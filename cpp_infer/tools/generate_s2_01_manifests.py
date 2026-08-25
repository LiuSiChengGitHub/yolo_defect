#!/usr/bin/env python3
"""Generate and verify the frozen S2-01 calibration and quality manifests.

The generator intentionally reads image/label files directly. Ultralytics cache
files are neither read nor referenced because they are derived, path-sensitive
state rather than dataset authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
CLASS_NAMES: Tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
CALIBRATION_INDICES: Tuple[int, ...] = tuple(1 + 8 * index for index in range(30))
QUALITY_INDICES: Mapping[str, Tuple[int, ...]] = {
    "crazing": tuple(range(240, 301)),
    **{name: tuple(range(241, 301)) for name in CLASS_NAMES[1:]},
}

CALIBRATION_MANIFEST_ID = "neu_det_train_ptq_calibration_6x30_v1"
QUALITY_MANIFEST_ID = "neu_det_val_361_quality_v1"

QUALITY_SCORE_FLOOR = 0.001
PRODUCT_SCORE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
IOU_THRESHOLDS: Tuple[float, ...] = tuple(
    round(0.50 + 0.05 * index, 2) for index in range(10)
)

PRODUCT_MATCHING_GATES: Mapping[str, float] = {
    "pair_iou_min": 0.50,
    "fp32_retention_min": 0.95,
    "int8_agreement_precision_min": 0.95,
    "matched_mean_iou_min": 0.90,
    "matched_iou_p05_min": 0.75,
    "confidence_abs_error_mean_max": 0.05,
    "confidence_abs_error_p95_max": 0.10,
}
PRODUCT_MATCHING_PROTOCOL: Mapping[str, str] = {
    "scope": "per_image_then_exact_class_id",
    "assignment": "greedy_one_to_one_descending_iou",
    "box_iou": "float32_continuous_xyxy",
    "edge_tie_break": (
        "fp32_detection_key_then_int8_detection_key_then_original_indices"
    ),
    "detection_key": (
        "class_id,negative_confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2"
    ),
    "pair_acceptance": "matching_iou_greater_than_or_equal_to_pair_iou_min",
    "percentile_interpolation": (
        "linear_between_adjacent_order_statistics_at_p_times_n_minus_1"
    ),
    "unmatched_policy": "every_detection_not_in_an_accepted_pair_is_unmatched",
}
QUALITY_GATES: Mapping[str, float] = {
    "map50_95_absolute_drop_max": 0.020,
    "map50_absolute_drop_max": 0.010,
    "per_class_ap50_absolute_drop_max": 0.050,
}


class ManifestError(RuntimeError):
    """A frozen manifest cannot be generated or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_utf8_lf_bytes(raw_bytes: bytes) -> bytes:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeError as error:
        raise ManifestError(f"Could not decode UTF-8 text: {error}") from error
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def canonical_lf_bytes(path: Path) -> bytes:
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise ManifestError(
            f"Could not read UTF-8 label '{path}': {error}"
        ) from error
    try:
        return normalize_utf8_lf_bytes(raw_bytes)
    except ManifestError as error:
        raise ManifestError(f"Could not read UTF-8 label '{path}': {error}") from error


def canonical_lf_sha256(path: Path) -> str:
    return hashlib.sha256(canonical_lf_bytes(path)).hexdigest().upper()


def canonical_rows_sha256(rows: Iterable[Sequence[Any]]) -> str:
    serialized = "".join(
        "\t".join(str(value) for value in row) + "\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest().upper()


def relative_declaration_path(manifest_path: Path, target_path: Path) -> str:
    import os

    return Path(
        os.path.relpath(target_path.resolve(), manifest_path.parent.resolve())
    ).as_posix()


def class_entries() -> List[Mapping[str, Any]]:
    return [
        {"class_id": class_id, "class_name": class_name}
        for class_id, class_name in enumerate(CLASS_NAMES)
    ]


def parse_yolo_label_text(
    text: str, object_name: str = "<in-memory YOLO label>"
) -> List[Mapping[str, Any]]:
    boxes: List[Mapping[str, Any]] = []
    seen_lines = set()
    for line_number, raw_line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if line in seen_lines:
            raise ManifestError(
                f"Duplicate label line in '{object_name}' at line {line_number}: {line}"
            )
        seen_lines.add(line)
        fields = re.split(r"\s+", line)
        if len(fields) != 5:
            raise ManifestError(
                f"Label '{object_name}' line {line_number} must have five fields"
            )
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = (
                float(value) for value in fields[1:]
            )
        except ValueError as error:
            raise ManifestError(
                f"Label '{object_name}' line {line_number} has a non-numeric field"
            ) from error
        values = (center_x, center_y, width, height)
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ManifestError(
                f"Label '{object_name}' line {line_number} has class_id={class_id}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ManifestError(
                f"Label '{object_name}' line {line_number} contains NaN or Infinity"
            )
        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0
        epsilon = 1.0e-9
        if (
            width <= 0.0
            or height <= 0.0
            or x1 < -epsilon
            or y1 < -epsilon
            or x2 > 1.0 + epsilon
            or y2 > 1.0 + epsilon
        ):
            raise ManifestError(
                f"Label '{object_name}' line {line_number} has an invalid normalized box"
            )
        boxes.append(
            {
                "class_id": class_id,
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
            }
        )
    if not boxes:
        raise ManifestError(f"Label '{object_name}' contains no boxes")
    return boxes


def parse_yolo_label(path: Path) -> List[Mapping[str, Any]]:
    return parse_yolo_label_text(
        canonical_lf_bytes(path).decode("utf-8"), str(path)
    )


def require_file(path: Path, object_name: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ManifestError(f"Missing {object_name}: '{resolved}'")
    return resolved


def calibration_sample(
    repo_root: Path,
    manifest_path: Path,
    class_id: int,
    class_name: str,
    index: int,
) -> Mapping[str, Any]:
    sample_id = f"{class_name}_{index}"
    image_path = require_file(
        repo_root / "data" / "images" / "train" / f"{sample_id}.jpg",
        "calibration image",
    )
    return {
        "sample_id": sample_id,
        "source_class_id": class_id,
        "source_class_name": class_name,
        "image_path": relative_declaration_path(manifest_path, image_path),
        "image_sha256": sha256_file(image_path),
    }


def quality_sample(
    repo_root: Path,
    manifest_path: Path,
    class_id: int,
    class_name: str,
    index: int,
) -> Mapping[str, Any]:
    sample_id = f"{class_name}_{index}"
    image_path = require_file(
        repo_root / "data" / "images" / "val" / f"{sample_id}.jpg",
        "quality image",
    )
    label_path = require_file(
        repo_root / "data" / "labels" / "val" / f"{sample_id}.txt",
        "quality label",
    )
    boxes = parse_yolo_label(label_path)
    return {
        "sample_id": sample_id,
        "source_class_id": class_id,
        "source_class_name": class_name,
        "image_path": relative_declaration_path(manifest_path, image_path),
        "image_sha256": sha256_file(image_path),
        "label_path": relative_declaration_path(manifest_path, label_path),
        "label_canonical_lf_sha256": canonical_lf_sha256(label_path),
        "ground_truth_box_count": len(boxes),
    }


def sample_rows(samples: Sequence[Mapping[str, Any]], kind: str) -> List[Sequence[Any]]:
    rows: List[Sequence[Any]] = []
    for sample in samples:
        row: List[Any] = [
            sample["sample_id"],
            sample["source_class_id"],
            sample["source_class_name"],
            sample["image_path"],
            sample["image_sha256"],
        ]
        if kind == "quality":
            row.extend(
                [
                    sample["label_path"],
                    sample["label_canonical_lf_sha256"],
                    sample["ground_truth_box_count"],
                ]
            )
        rows.append(row)
    return rows


def validate_unique_samples(samples: Sequence[Mapping[str, Any]], kind: str) -> None:
    for field in ("sample_id", "image_path", "image_sha256"):
        values = [sample[field] for sample in samples]
        if len(values) != len(set(values)):
            raise ManifestError(f"{kind} samples contain duplicate {field} values")


def build_calibration_manifest(
    repo_root: Path, manifest_path: Path
) -> Mapping[str, Any]:
    samples = [
        calibration_sample(
            repo_root, manifest_path, class_id, class_name, sample_index
        )
        for class_id, class_name in enumerate(CLASS_NAMES)
        for sample_index in CALIBRATION_INDICES
    ]
    validate_unique_samples(samples, "calibration")
    source_counts = Counter(sample["source_class_id"] for sample in samples)
    if source_counts != Counter({class_id: 30 for class_id in range(6)}):
        raise ManifestError(f"Unexpected calibration class counts: {source_counts}")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "static_ptq_calibration",
        "manifest_id": CALIBRATION_MANIFEST_ID,
        "dataset": {
            "name": "NEU-DET",
            "split": "train",
            "selection_rule": "for each artifact class, select index i=1+8*k for k=0..29",
            "samples_per_source_class": 30,
            "sample_count": len(samples),
            "source_class_semantics": "filename prefix used only for balanced sampling; labels are not consumed by PTQ",
        },
        "preprocess": {
            "type": "letterbox_rgb_0_1_nchw",
            "decode": "OpenCV IMREAD_COLOR uint8 BGR",
            "input_shape": [1, 3, 800, 800],
            "resize_interpolation": "INTER_LINEAR",
            "letterbox_pad_value": 114,
            "resize_rounding": "floor(positive_value+0.5)",
            "color_conversion": "BGR_to_RGB",
            "normalization": "float32_divide_by_255",
            "layout": "NCHW",
            "tensor_contiguity": "C_contiguous",
        },
        "classes": class_entries(),
        "samples": samples,
        "integrity": {
            "sample_row_format": "sample_id<TAB>source_class_id<TAB>source_class_name<TAB>image_path<TAB>image_sha256<LF>",
            "sample_set_sha256": canonical_rows_sha256(
                sample_rows(samples, "calibration")
            ),
            "image_hash_semantics": "SHA-256 of raw image file bytes",
        },
    }


def build_quality_manifest(repo_root: Path, manifest_path: Path) -> Mapping[str, Any]:
    samples = [
        quality_sample(repo_root, manifest_path, class_id, class_name, sample_index)
        for class_id, class_name in enumerate(CLASS_NAMES)
        for sample_index in QUALITY_INDICES[class_name]
    ]
    validate_unique_samples(samples, "quality")
    source_counts = Counter(sample["source_class_id"] for sample in samples)
    expected_source_counts = Counter({0: 61, 1: 60, 2: 60, 3: 60, 4: 60, 5: 60})
    if source_counts != expected_source_counts:
        raise ManifestError(f"Unexpected quality source counts: {source_counts}")

    ground_truth_counts: Counter = Counter()
    for sample in samples:
        label_path = (manifest_path.parent / sample["label_path"]).resolve()
        ground_truth_counts.update(
            box["class_id"] for box in parse_yolo_label(label_path)
        )
    expected_ground_truth = Counter({0: 165, 1: 159, 2: 193, 3: 87, 4: 132, 5: 121})
    if ground_truth_counts != expected_ground_truth:
        raise ManifestError(
            f"Unexpected quality ground-truth counts: {ground_truth_counts}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": "detection_task_quality",
        "manifest_id": QUALITY_MANIFEST_ID,
        "dataset": {
            "name": "NEU-DET",
            "split": "validation",
            "selection_rule": "all tracked flattened validation images: crazing 240..300 and every other class 241..300",
            "sample_count": len(samples),
            "source_class_counts": {
                str(class_id): expected_source_counts[class_id]
                for class_id in range(6)
            },
            "ground_truth_box_count": sum(ground_truth_counts.values()),
            "ground_truth_class_counts": {
                str(class_id): ground_truth_counts[class_id]
                for class_id in range(6)
            },
            "source_class_semantics": "filename prefix is not ground truth; each YOLO label may contain multiple classes",
            "cache_policy": "read image and label paths from this manifest; do not read data/labels/val.cache",
        },
        "preprocess": {
            "type": "letterbox_rgb_0_1_nchw",
            "input_shape": [1, 3, 800, 800],
            "letterbox_pad_value": 114,
        },
        "evaluation": {
            "label_format": "YOLO normalized class_id center_x center_y width height",
            "box_geometry": "continuous xyxy; area=(x2-x1)*(y2-y1)",
            "quality_score_floor": QUALITY_SCORE_FLOOR,
            "product_score_threshold": PRODUCT_SCORE_THRESHOLD,
            "score_comparison": "strict_greater_than",
            "nms_threshold": NMS_THRESHOLD,
            "nms_mode": "class_agnostic",
            "iou_thresholds": list(IOU_THRESHOLDS),
            "ap_interpolation": "COCO_101_point_precision_envelope",
            "max_detections_per_image": None,
            "metric_claim": "COCO_style_101_point_without_area_ranges_or_max_dets",
            "ground_truth_policy": "count every YOLO TXT box; VOC difficult/truncated flags are not represented",
        },
        "product_matching_protocol": dict(PRODUCT_MATCHING_PROTOCOL),
        "product_matching_gates": dict(PRODUCT_MATCHING_GATES),
        "quality_gates": dict(QUALITY_GATES),
        "classes": class_entries(),
        "samples": samples,
        "integrity": {
            "sample_row_format": "sample_id<TAB>source_class_id<TAB>source_class_name<TAB>image_path<TAB>image_sha256<TAB>label_path<TAB>label_canonical_lf_sha256<TAB>ground_truth_box_count<LF>",
            "sample_set_sha256": canonical_rows_sha256(
                sample_rows(samples, "quality")
            ),
            "image_hash_semantics": "SHA-256 of raw image file bytes",
            "label_hash_semantics": "decode UTF-8, normalize CRLF and CR to LF, preserve all other text bytes, then SHA-256",
        },
    }


def serialize_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def write_json_lf(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialize_json(value))
    temporary_path.replace(path)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Could not load JSON '{path}': {error}") from error


def check_document(path: Path, expected: Mapping[str, Any]) -> None:
    actual = load_json(path)
    if actual != expected:
        raise ManifestError(
            f"Frozen manifest '{path}' differs from deterministic generator output"
        )


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    repo_root = default_repo_root()
    fixture_root = repo_root / "cpp_infer" / "tests" / "fixtures"
    parser = argparse.ArgumentParser(
        description="Generate or verify frozen S2-01 calibration/quality manifests"
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=fixture_root / "s2_01_calibration_manifest.json",
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=fixture_root / "s2_01_quality_manifest.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing manifests without writing",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    repo_root = arguments.repo_root.resolve()
    calibration_path = arguments.calibration_output.resolve()
    quality_path = arguments.quality_output.resolve()
    try:
        calibration = build_calibration_manifest(repo_root, calibration_path)
        quality = build_quality_manifest(repo_root, quality_path)
        calibration_hashes = {
            sample["image_sha256"] for sample in calibration["samples"]
        }
        quality_hashes = {sample["image_sha256"] for sample in quality["samples"]}
        overlap = sorted(calibration_hashes & quality_hashes)
        if overlap:
            raise ManifestError(
                f"Calibration and quality image hashes overlap: {overlap[:3]}"
            )
        if arguments.check:
            check_document(calibration_path, calibration)
            check_document(quality_path, quality)
            action = "verified"
        else:
            write_json_lf(calibration_path, calibration)
            write_json_lf(quality_path, quality)
            action = "generated"
        print(
            f"S2-01 manifests {action}: calibration={len(calibration['samples'])} "
            f"sha={calibration['integrity']['sample_set_sha256']}; "
            f"quality={len(quality['samples'])} "
            f"sha={quality['integrity']['sample_set_sha256']}"
        )
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

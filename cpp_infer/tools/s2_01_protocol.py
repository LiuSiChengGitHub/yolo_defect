#!/usr/bin/env python3
"""Strict, dependency-free loader for the frozen S2-01 PTQ protocol.

The module deliberately imports only the Python standard library.  This lets
the Stage-1 ``TestBase`` interpreter validate protocol, hashes, and path
semantics even though that interpreter does not provide the ``onnx`` package.

Protocol JSON schema v1
-----------------------

The root object has these exact fields::

    {
      "schema_version": 1,
      "protocol_id": "...",
      "source_model": {
        "path": "...",
        "sha256": "64 uppercase hex characters",
        "size_bytes": 123
      },
      "calibration": {
        "manifest_path": "...",
        "manifest_id": "...",
        "manifest_sha256_canonical_lf": "64 uppercase hex characters",
        "sample_count": 180,
        "expected_class_counts": {"class_name": 30},
        "preprocess": { ... the exact manifest mapping below ... }
      },
      "quantization": { ... the exact frozen mapping below ... },
      "model_contract": {
        "input_name": "images",
        "input_shape": [1, 3, 800, 800],
        "input_dtype": "float32",
        "output_name": "output0",
        "output_shape": [1, 10, 13125],
        "output_dtype": "float32"
      },
      "environment": {
        "onnx_version": "1.19.1",
        "onnxruntime_version": "1.19.2",
        "numpy_version": "2.0.2",
        "opencv_version": "4.13.0",
        "execution_provider": "CPUExecutionProvider"
      },
      "correctness": {
        "consistency_manifest": {
          "path": "...", "manifest_id": "...",
          "sha256_canonical_lf": "...", "sample_count": 30
        },
        "quality_manifest": {
          "path": "...", "manifest_id": "...",
          "sha256_canonical_lf": "...", "sample_count": 361,
          "ground_truth_box_count": 857
        },
        "gate_sources": {
          "matching_protocol": "quality_manifest.product_matching_protocol",
          "product_matching": "quality_manifest.product_matching_gates",
          "task_quality": "quality_manifest.quality_gates"
        }
      },
      "benchmark": {
        "build_type": "Release", "execution_provider": "CPUExecutionProvider",
        "execution_mode": "sequential", "intra_op_num_threads": 1,
        "inter_op_num_threads": 1, "graph_optimization_level": "all",
        "warmup": 10, "repeat": 100, "profiling_enabled": false,
        "sample": {
          "sample_id": "crazing_241", "image_path": "...",
          "image_sha256": "..."
        }
      },
      "profiling": {
        "execution_provider": "CPUExecutionProvider", "runs": 10,
        "sample_source": "benchmark.sample",
        "separate_from_formal_benchmark": true, "performance_gate": false
      },
      "output": {
        "model_path": "...",
        "report_path": "..."
      }
    }

Every relative path in the protocol is resolved from the protocol file.  The
calibration manifest has exact root fields ``schema_version``,
``manifest_kind``, ``manifest_id``, ``dataset``, ``preprocess``, ``classes``,
``samples``, and ``integrity``.  Every sample has exact fields ``sample_id``,
``source_class_id``, ``source_class_name``, ``image_path``, and
``image_sha256``.  Image paths are resolved from the manifest file.

``manifest_sha256_canonical_lf`` hashes the UTF-8 manifest after normalizing
CRLF and lone CR line endings to LF.  It does not reformat JSON or add/remove a
final newline.  Model and image hashes always cover the raw file bytes.

``protocol_id`` is also semantic: v1 quantizes all 64 Conv candidates with
MinMax and therefore requires an empty ``nodes_to_exclude`` array; v2 requires
the exact source-graph ordered 19-node ``/model.22`` detection-head exclusion
list, targets the remaining 45 Conv nodes, and keeps MinMax; v3 keeps the exact
v2 node policy but uses Entropy calibration; v4 uses MinMax, excludes the 37
source-ordered neck/head Conv nodes beginning at ``/model.12/cv1/conv/Conv``,
and targets only the 27 ``model.0..9`` backbone Conv nodes; v5 targets the 13
early ``model.0..4`` Conv nodes; v6 targets the complementary 14 late
``model.5..9`` Conv nodes; v7 targets the 7 mid ``model.5..6`` Conv nodes; v8
targets the 7 deep ``model.7..9`` Conv nodes; v9, v10, and v11 target the
source-graph prefixes ending at ``model.2``, ``model.1``, and ``model.0``
respectively (6, 2, and 1 Conv nodes).  All block-ablation protocols retain
MinMax and exclude every non-target Conv in source order.  No other parameter
difference or arbitrary selective policy is accepted under these identifiers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple


SCHEMA_VERSION = 1
CALIBRATION_MANIFEST_SCHEMA_VERSION = 1
FROZEN_CALIBRATION_SAMPLE_COUNT = 180
FROZEN_SELECTED_CONV_COUNT = 64
FROZEN_HEAD_FP32_EXCLUDED_CONV_COUNT = 19
FROZEN_HEAD_FP32_TARGET_CONV_COUNT = 45
FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_COUNT = 37
FROZEN_BACKBONE_ONLY_TARGET_CONV_COUNT = 27
FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT = 13
FROZEN_EARLY_BACKBONE_EXCLUDED_CONV_COUNT = 51
FROZEN_LATE_BACKBONE_TARGET_CONV_COUNT = 14
FROZEN_LATE_BACKBONE_EXCLUDED_CONV_COUNT = 50
FROZEN_MID_BACKBONE_TARGET_CONV_COUNT = 7
FROZEN_MID_BACKBONE_EXCLUDED_CONV_COUNT = 57
FROZEN_DEEP_BACKBONE_TARGET_CONV_COUNT = 7
FROZEN_DEEP_BACKBONE_EXCLUDED_CONV_COUNT = 57
FROZEN_PREFIX_MODEL0_2_TARGET_CONV_COUNT = 6
FROZEN_PREFIX_MODEL0_2_EXCLUDED_CONV_COUNT = 58
FROZEN_PREFIX_MODEL0_1_TARGET_CONV_COUNT = 2
FROZEN_PREFIX_MODEL0_1_EXCLUDED_CONV_COUNT = 62
FROZEN_PREFIX_MODEL0_TARGET_CONV_COUNT = 1
FROZEN_PREFIX_MODEL0_EXCLUDED_CONV_COUNT = 63
FROZEN_CONSISTENCY_SAMPLE_COUNT = 30
FROZEN_QUALITY_SAMPLE_COUNT = 361
FROZEN_QUALITY_GROUND_TRUTH_BOX_COUNT = 857
FROZEN_CONSISTENCY_MANIFEST_ID = "neu_det_val_6x5_v1"
FROZEN_QUALITY_MANIFEST_ID = "neu_det_val_361_quality_v1"

PROTOCOL_ID_V1 = "s2_01_static_ptq_qdq_s8s8_cpu_v1"
PROTOCOL_ID_V2 = "s2_01_static_ptq_qdq_s8s8_head_fp32_cpu_v2"
PROTOCOL_ID_V3 = "s2_01_static_ptq_qdq_s8s8_head_fp32_entropy_cpu_v3"
PROTOCOL_ID_V4 = "s2_01_static_ptq_qdq_s8s8_backbone_only_cpu_v4"
PROTOCOL_ID_V5 = "s2_01_static_ptq_qdq_s8s8_early_backbone_cpu_v5"
PROTOCOL_ID_V6 = "s2_01_static_ptq_qdq_s8s8_late_backbone_cpu_v6"
PROTOCOL_ID_V7 = "s2_01_static_ptq_qdq_s8s8_mid_backbone_cpu_v7"
PROTOCOL_ID_V8 = "s2_01_static_ptq_qdq_s8s8_deep_backbone_cpu_v8"
PROTOCOL_ID_V9 = "s2_01_static_ptq_qdq_s8s8_prefix_model0_2_cpu_v9"
PROTOCOL_ID_V10 = "s2_01_static_ptq_qdq_s8s8_prefix_model0_1_cpu_v10"
PROTOCOL_ID_V11 = "s2_01_static_ptq_qdq_s8s8_prefix_model0_cpu_v11"
SUPPORTED_PROTOCOL_IDS: Tuple[str, ...] = (
    PROTOCOL_ID_V1,
    PROTOCOL_ID_V2,
    PROTOCOL_ID_V3,
    PROTOCOL_ID_V4,
    PROTOCOL_ID_V5,
    PROTOCOL_ID_V6,
    PROTOCOL_ID_V7,
    PROTOCOL_ID_V8,
    PROTOCOL_ID_V9,
    PROTOCOL_ID_V10,
    PROTOCOL_ID_V11,
)

FROZEN_EARLY_BACKBONE_TARGET_CONV_NODES: Tuple[str, ...] = (
    "/model.0/conv/Conv",
    "/model.1/conv/Conv",
    "/model.2/cv1/conv/Conv",
    "/model.2/m.0/cv1/conv/Conv",
    "/model.2/m.0/cv2/conv/Conv",
    "/model.2/cv2/conv/Conv",
    "/model.3/conv/Conv",
    "/model.4/cv1/conv/Conv",
    "/model.4/m.0/cv1/conv/Conv",
    "/model.4/m.0/cv2/conv/Conv",
    "/model.4/m.1/cv1/conv/Conv",
    "/model.4/m.1/cv2/conv/Conv",
    "/model.4/cv2/conv/Conv",
)

FROZEN_LATE_BACKBONE_TARGET_CONV_NODES: Tuple[str, ...] = (
    "/model.5/conv/Conv",
    "/model.6/cv1/conv/Conv",
    "/model.6/m.0/cv1/conv/Conv",
    "/model.6/m.0/cv2/conv/Conv",
    "/model.6/m.1/cv1/conv/Conv",
    "/model.6/m.1/cv2/conv/Conv",
    "/model.6/cv2/conv/Conv",
    "/model.7/conv/Conv",
    "/model.8/cv1/conv/Conv",
    "/model.8/m.0/cv1/conv/Conv",
    "/model.8/m.0/cv2/conv/Conv",
    "/model.8/cv2/conv/Conv",
    "/model.9/cv1/conv/Conv",
    "/model.9/cv2/conv/Conv",
)

FROZEN_MID_BACKBONE_TARGET_CONV_NODES: Tuple[str, ...] = (
    FROZEN_LATE_BACKBONE_TARGET_CONV_NODES[
        :FROZEN_MID_BACKBONE_TARGET_CONV_COUNT
    ]
)

FROZEN_DEEP_BACKBONE_TARGET_CONV_NODES: Tuple[str, ...] = (
    FROZEN_LATE_BACKBONE_TARGET_CONV_NODES[
        FROZEN_MID_BACKBONE_TARGET_CONV_COUNT:
    ]
)

FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    "/model.22/cv2.0/cv2.0.0/conv/Conv",
    "/model.22/cv2.0/cv2.0.1/conv/Conv",
    "/model.22/cv2.0/cv2.0.2/Conv",
    "/model.22/cv2.1/cv2.1.0/conv/Conv",
    "/model.22/cv2.1/cv2.1.1/conv/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv",
    "/model.22/cv2.2/cv2.2.0/conv/Conv",
    "/model.22/cv2.2/cv2.2.1/conv/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv",
    "/model.22/cv3.0/cv3.0.0/conv/Conv",
    "/model.22/cv3.0/cv3.0.1/conv/Conv",
    "/model.22/cv3.0/cv3.0.2/Conv",
    "/model.22/cv3.1/cv3.1.0/conv/Conv",
    "/model.22/cv3.1/cv3.1.1/conv/Conv",
    "/model.22/cv3.1/cv3.1.2/Conv",
    "/model.22/cv3.2/cv3.2.0/conv/Conv",
    "/model.22/cv3.2/cv3.2.1/conv/Conv",
    "/model.22/cv3.2/cv3.2.2/Conv",
    "/model.22/dfl/conv/Conv",
)

FROZEN_BACKBONE_ONLY_NECK_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    "/model.12/cv1/conv/Conv",
    "/model.12/m.0/cv1/conv/Conv",
    "/model.12/m.0/cv2/conv/Conv",
    "/model.12/cv2/conv/Conv",
    "/model.15/cv1/conv/Conv",
    "/model.15/m.0/cv1/conv/Conv",
    "/model.15/m.0/cv2/conv/Conv",
    "/model.15/cv2/conv/Conv",
    "/model.16/conv/Conv",
    "/model.18/cv1/conv/Conv",
    "/model.18/m.0/cv1/conv/Conv",
    "/model.18/m.0/cv2/conv/Conv",
    "/model.18/cv2/conv/Conv",
    "/model.19/conv/Conv",
    "/model.21/cv1/conv/Conv",
    "/model.21/m.0/cv1/conv/Conv",
    "/model.21/m.0/cv2/conv/Conv",
    "/model.21/cv2/conv/Conv",
)

FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_BACKBONE_ONLY_NECK_EXCLUDED_CONV_NODES
    + FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES
)

FROZEN_EARLY_BACKBONE_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_LATE_BACKBONE_TARGET_CONV_NODES
    + FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES
)

FROZEN_LATE_BACKBONE_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_EARLY_BACKBONE_TARGET_CONV_NODES
    + FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES
)

FROZEN_MID_BACKBONE_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_EARLY_BACKBONE_TARGET_CONV_NODES
    + FROZEN_DEEP_BACKBONE_TARGET_CONV_NODES
    + FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES
)

FROZEN_DEEP_BACKBONE_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_EARLY_BACKBONE_TARGET_CONV_NODES
    + FROZEN_MID_BACKBONE_TARGET_CONV_NODES
    + FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES
)

FROZEN_SOURCE_ORDERED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_EARLY_BACKBONE_TARGET_CONV_NODES
    + FROZEN_LATE_BACKBONE_TARGET_CONV_NODES
    + FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES
)

FROZEN_PREFIX_MODEL0_2_TARGET_CONV_NODES: Tuple[str, ...] = (
    FROZEN_SOURCE_ORDERED_CONV_NODES[:FROZEN_PREFIX_MODEL0_2_TARGET_CONV_COUNT]
)
FROZEN_PREFIX_MODEL0_2_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_SOURCE_ORDERED_CONV_NODES[FROZEN_PREFIX_MODEL0_2_TARGET_CONV_COUNT:]
)

FROZEN_PREFIX_MODEL0_1_TARGET_CONV_NODES: Tuple[str, ...] = (
    FROZEN_SOURCE_ORDERED_CONV_NODES[:FROZEN_PREFIX_MODEL0_1_TARGET_CONV_COUNT]
)
FROZEN_PREFIX_MODEL0_1_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_SOURCE_ORDERED_CONV_NODES[FROZEN_PREFIX_MODEL0_1_TARGET_CONV_COUNT:]
)

FROZEN_PREFIX_MODEL0_TARGET_CONV_NODES: Tuple[str, ...] = (
    FROZEN_SOURCE_ORDERED_CONV_NODES[:FROZEN_PREFIX_MODEL0_TARGET_CONV_COUNT]
)
FROZEN_PREFIX_MODEL0_EXCLUDED_CONV_NODES: Tuple[str, ...] = (
    FROZEN_SOURCE_ORDERED_CONV_NODES[FROZEN_PREFIX_MODEL0_TARGET_CONV_COUNT:]
)

EXPECTED_CALIBRATION_PREPROCESS: Mapping[str, Any] = {
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
}

EXPECTED_CALIBRATION_DATASET: Mapping[str, Any] = {
    "name": "NEU-DET",
    "split": "train",
    "selection_rule": (
        "for each artifact class, select index i=1+8*k for k=0..29"
    ),
    "samples_per_source_class": 30,
    "sample_count": FROZEN_CALIBRATION_SAMPLE_COUNT,
    "source_class_semantics": (
        "filename prefix used only for balanced sampling; labels are not "
        "consumed by PTQ"
    ),
}

EXPECTED_CALIBRATION_CLASSES: Tuple[Mapping[str, Any], ...] = tuple(
    {"class_id": class_id, "class_name": class_name}
    for class_id, class_name in enumerate(
        (
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches",
        )
    )
)

CALIBRATION_SAMPLE_ROW_FORMAT = (
    "sample_id<TAB>source_class_id<TAB>source_class_name<TAB>image_path<TAB>"
    "image_sha256<LF>"
)
CALIBRATION_IMAGE_HASH_SEMANTICS = "SHA-256 of raw image file bytes"

EXPECTED_QUANTIZATION: Mapping[str, Any] = {
    "preprocess": {
        "skip_optimization": True,
        "skip_symbolic_shape": True,
        "skip_onnx_shape": False,
    },
    "format": "QDQ",
    "activation_type": "QInt8",
    "weight_type": "QInt8",
    "op_types_to_quantize": ["Conv"],
    "expected_selected_node_count": FROZEN_SELECTED_CONV_COUNT,
    "per_channel": True,
    "reduce_range": False,
    "calibrate_method": "MinMax",
    "nodes_to_exclude": [],
    "use_external_data_format": False,
    "extra_options": {
        "ActivationSymmetric": False,
        "WeightSymmetric": True,
        "CalibTensorRangeSymmetric": False,
        "CalibMovingAverage": False,
        "ForceQuantizeNoInputCheck": False,
        "AddQDQPairToWeight": False,
        "DedicatedQDQPair": False,
        "QDQKeepRemovableActivations": False,
    },
}

EXPECTED_ENVIRONMENT: Mapping[str, str] = {
    "onnx_version": "1.19.1",
    "onnxruntime_version": "1.19.2",
    "numpy_version": "2.0.2",
    "opencv_version": "4.13.0",
    "execution_provider": "CPUExecutionProvider",
}

EXPECTED_PRODUCT_MATCHING_GATES: Mapping[str, float] = {
    "pair_iou_min": 0.50,
    "fp32_retention_min": 0.95,
    "int8_agreement_precision_min": 0.95,
    "matched_mean_iou_min": 0.90,
    "matched_iou_p05_min": 0.75,
    "confidence_abs_error_mean_max": 0.05,
    "confidence_abs_error_p95_max": 0.10,
}

EXPECTED_PRODUCT_MATCHING_PROTOCOL: Mapping[str, str] = {
    "scope": "per_image_then_exact_class_id",
    "assignment": "greedy_one_to_one_descending_iou",
    "box_iou": "float32_continuous_xyxy",
    "edge_tie_break": (
        "fp32_detection_key_then_int8_detection_key_then_original_indices"
    ),
    "detection_key": (
        "class_id,negative_confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2"
    ),
    "pair_acceptance": (
        "matching_iou_greater_than_or_equal_to_pair_iou_min"
    ),
    "percentile_interpolation": (
        "linear_between_adjacent_order_statistics_at_p_times_n_minus_1"
    ),
    "unmatched_policy": (
        "every_detection_not_in_an_accepted_pair_is_unmatched"
    ),
}

EXPECTED_QUALITY_EVALUATION: Mapping[str, Any] = {
    "label_format": (
        "YOLO normalized class_id center_x center_y width height"
    ),
    "box_geometry": "continuous xyxy; area=(x2-x1)*(y2-y1)",
    "quality_score_floor": 0.001,
    "product_score_threshold": 0.25,
    "score_comparison": "strict_greater_than",
    "nms_threshold": 0.45,
    "nms_mode": "class_agnostic",
    "iou_thresholds": [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ],
    "ap_interpolation": "COCO_101_point_precision_envelope",
    "max_detections_per_image": None,
    "metric_claim": "COCO_style_101_point_without_area_ranges_or_max_dets",
    "ground_truth_policy": (
        "count every YOLO TXT box; VOC difficult/truncated flags are not "
        "represented"
    ),
}

EXPECTED_QUALITY_GATES: Mapping[str, float] = {
    "map50_95_absolute_drop_max": 0.020,
    "map50_absolute_drop_max": 0.010,
    "per_class_ap50_absolute_drop_max": 0.050,
}

EXPECTED_CORRECTNESS_GATE_SOURCES: Mapping[str, str] = {
    "matching_protocol": "quality_manifest.product_matching_protocol",
    "product_matching": "quality_manifest.product_matching_gates",
    "task_quality": "quality_manifest.quality_gates",
}

EXPECTED_BENCHMARK_SETTINGS: Mapping[str, Any] = {
    "build_type": "Release",
    "execution_provider": "CPUExecutionProvider",
    "execution_mode": "sequential",
    "intra_op_num_threads": 1,
    "inter_op_num_threads": 1,
    "graph_optimization_level": "all",
    "warmup": 10,
    "repeat": 100,
    "profiling_enabled": False,
}

EXPECTED_PROFILING: Mapping[str, Any] = {
    "execution_provider": "CPUExecutionProvider",
    "runs": 10,
    "sample_source": "benchmark.sample",
    "separate_from_formal_benchmark": True,
    "performance_gate": False,
}

_SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")


class S201ProtocolError(RuntimeError):
    """An actionable frozen-protocol or file-integrity failure."""


def fail(object_name: str, expected: str, actual: str, action: str) -> None:
    raise S201ProtocolError(
        "S2-01 protocol validation failed: "
        f"object={object_name}; expected={expected}; actual={actual}; "
        f"action={action}"
    )


@dataclass(frozen=True)
class CalibrationSample:
    """One hash-verified calibration image resolved from the manifest."""

    sample_id: str
    source_class_id: int
    source_class_name: str
    image_path: Path
    image_sha256: str


@dataclass(frozen=True)
class FrozenS201Protocol:
    """Validated, resolved S2-01 protocol consumed by the quantization tool."""

    declaration_path: Path
    document: Mapping[str, Any]
    protocol_id: str
    source_model_path: Path
    source_model_sha256: str
    source_model_size_bytes: int
    calibration_manifest_path: Path
    calibration_manifest_document: Mapping[str, Any]
    calibration_manifest_id: str
    calibration_manifest_sha256_canonical_lf: str
    calibration_samples: Tuple[CalibrationSample, ...]
    calibration_preprocess: Mapping[str, Any]
    quantization: Mapping[str, Any]
    model_contract: Mapping[str, Any]
    environment: Mapping[str, str]
    correctness: Mapping[str, Any]
    consistency_manifest_path: Path
    quality_manifest_path: Path
    quality_evaluation: Mapping[str, Any]
    product_matching_protocol: Mapping[str, str]
    product_matching_gates: Mapping[str, float]
    quality_gates: Mapping[str, float]
    benchmark: Mapping[str, Any]
    benchmark_sample_path: Path
    profiling: Mapping[str, Any]
    output_model_path: Path
    output_report_path: Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def canonical_rows_sha256(rows: Iterable[Sequence[Any]]) -> str:
    """Hash tab-separated rows with a required LF terminator per row."""

    serialized = "".join(
        "\t".join(str(value) for value in row) + "\n" for row in rows
    ).encode("utf-8")
    return sha256_bytes(serialized)


def sha256_file_raw(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail(
            f"file.raw_sha256[{path}]",
            "a readable regular file",
            str(error),
            "restore the declared file and its read permissions",
        )
    return digest.hexdigest().upper()


def canonical_lf_bytes(raw_bytes: bytes, object_name: str) -> bytes:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(
            object_name,
            "UTF-8 text whose line endings can be normalized to LF",
            str(error),
            "save the JSON as UTF-8 without binary or locale-specific encoding",
        )
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sha256_file_canonical_lf(path: Path) -> str:
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        fail(
            f"file.canonical_lf_sha256[{path}]",
            "a readable UTF-8 regular file",
            str(error),
            "restore the declared file and its read permissions",
        )
    return sha256_bytes(
        canonical_lf_bytes(raw_bytes, f"file.canonical_lf[{path}]")
    )


def _reject_duplicate_json_keys(
    pairs: Iterable[Tuple[str, Any]],
) -> MutableMapping[str, Any]:
    value: MutableMapping[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(
                f"json.field[{key}]",
                "each JSON field to occur exactly once",
                "duplicate field",
                "remove the duplicate declaration",
            )
        value[key] = item
    return value


def _reject_nonfinite_json_number(token: str) -> None:
    fail(
        "json.number",
        "an RFC-compliant finite JSON number",
        token,
        "replace NaN or Infinity with an explicit finite value",
    )


def load_json(path: Path, object_name: str) -> Mapping[str, Any]:
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_number,
        )
    except S201ProtocolError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(
            object_name,
            "one readable UTF-8 JSON document",
            str(error),
            "fix the path, encoding, or JSON syntax",
        )
    if not isinstance(value, dict):
        fail(
            object_name,
            "a JSON object at the document root",
            type(value).__name__,
            "replace the root value with an object",
        )
    return value


def _expect_mapping(value: Any, object_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        fail(
            object_name,
            "a JSON object",
            type(value).__name__,
            "restore the documented object structure",
        )
    return value


def _expect_exact_keys(
    value: Mapping[str, Any], required_keys: Sequence[str], object_name: str
) -> None:
    expected = set(required_keys)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        fail(
            object_name,
            f"exact fields {sorted(expected)}",
            f"missing={missing}, unknown={unknown}",
            "restore the frozen schema instead of adding implicit defaults",
        )


def _expect_string(value: Any, object_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(
            object_name,
            "a non-empty string",
            repr(value),
            "set the documented string value",
        )
    return value


def _expect_int(value: Any, object_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(
            object_name,
            f"an integer >= {minimum}",
            repr(value),
            "set the documented integer value",
        )
    return value


def _expect_sha256(value: Any, object_name: str) -> str:
    text = _expect_string(value, object_name)
    if not _SHA256_PATTERN.fullmatch(text):
        fail(
            object_name,
            "exactly 64 uppercase hexadecimal characters",
            repr(text),
            "recompute SHA-256 from the declared bytes and store it uppercase",
        )
    return text


def _expect_shape(value: Any, object_name: str) -> Sequence[int]:
    if not isinstance(value, list) or not value:
        fail(
            object_name,
            "a non-empty JSON array of positive integers",
            repr(value),
            "restore the frozen tensor shape",
        )
    for index, dimension in enumerate(value):
        if type(dimension) is not int or dimension <= 0:
            fail(
                f"{object_name}[{index}]",
                "a positive integer dimension",
                repr(dimension),
                "restore the frozen static tensor shape",
            )
    return value


def _resolve_input_path(
    raw_value: Any, base_directory: Path, object_name: str
) -> Path:
    value = _expect_string(raw_value, object_name)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        fail(
            object_name,
            "an existing regular file",
            f"{candidate}: {error}",
            "restore the frozen input or correct the declaration-relative path",
        )
    if not resolved.is_file():
        fail(
            object_name,
            "an existing regular file",
            str(resolved),
            "replace the path with a regular file",
        )
    return resolved


def _resolve_output_path(
    raw_value: Any, base_directory: Path, object_name: str
) -> Path:
    value = _expect_string(raw_value, object_name)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    try:
        return candidate.resolve(strict=False)
    except OSError as error:
        fail(
            object_name,
            "a resolvable output file path",
            f"{candidate}: {error}",
            "correct the path syntax",
        )


def _expect_frozen_mapping(
    actual: Any, expected: Mapping[str, Any], object_name: str
) -> Mapping[str, Any]:
    mapping = _expect_mapping(actual, object_name)
    if mapping != expected:
        fail(
            object_name,
            json.dumps(expected, sort_keys=True, separators=(",", ":")),
            json.dumps(mapping, sort_keys=True, separators=(",", ":")),
            "restore the predeclared S2-01 v1 protocol before quantization",
        )
    return mapping


def expected_quantization_for_protocol(
    protocol_id: str,
) -> Mapping[str, Any]:
    """Return the exact v1..v11 PTQ mapping without shared mutable lists."""

    if protocol_id == PROTOCOL_ID_V1:
        excluded_nodes: Sequence[str] = ()
    elif protocol_id in (PROTOCOL_ID_V2, PROTOCOL_ID_V3):
        excluded_nodes = FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V4:
        excluded_nodes = FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V5:
        excluded_nodes = FROZEN_EARLY_BACKBONE_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V6:
        excluded_nodes = FROZEN_LATE_BACKBONE_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V7:
        excluded_nodes = FROZEN_MID_BACKBONE_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V8:
        excluded_nodes = FROZEN_DEEP_BACKBONE_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V9:
        excluded_nodes = FROZEN_PREFIX_MODEL0_2_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V10:
        excluded_nodes = FROZEN_PREFIX_MODEL0_1_EXCLUDED_CONV_NODES
    elif protocol_id == PROTOCOL_ID_V11:
        excluded_nodes = FROZEN_PREFIX_MODEL0_EXCLUDED_CONV_NODES
    else:
        fail(
            "protocol.protocol_id",
            f"one of {list(SUPPORTED_PROTOCOL_IDS)}",
            repr(protocol_id),
            "select one of the eleven frozen S2-01 PTQ protocols",
        )
    expected = dict(EXPECTED_QUANTIZATION)
    expected["preprocess"] = dict(EXPECTED_QUANTIZATION["preprocess"])
    expected["op_types_to_quantize"] = list(
        EXPECTED_QUANTIZATION["op_types_to_quantize"]
    )
    expected["nodes_to_exclude"] = list(excluded_nodes)
    expected["extra_options"] = dict(EXPECTED_QUANTIZATION["extra_options"])
    if protocol_id == PROTOCOL_ID_V3:
        expected["calibrate_method"] = "Entropy"
    return expected


def _validate_quantization(
    value: Any, protocol_id: str
) -> Mapping[str, Any]:
    quantization = _expect_mapping(value, "protocol.quantization")
    _expect_exact_keys(
        quantization,
        tuple(EXPECTED_QUANTIZATION),
        "protocol.quantization",
    )
    excluded = quantization["nodes_to_exclude"]
    if not isinstance(excluded, list):
        fail(
            "protocol.quantization.nodes_to_exclude",
            "a JSON array",
            type(excluded).__name__,
            "declare the exact ordered exclusion list",
        )
    validated_excluded = []
    for index, raw_name in enumerate(excluded):
        validated_excluded.append(
            _expect_string(
                raw_name,
                f"protocol.quantization.nodes_to_exclude[{index}]",
            )
        )
    if len(set(validated_excluded)) != len(validated_excluded):
        duplicates = sorted(
            name
            for name in set(validated_excluded)
            if validated_excluded.count(name) > 1
        )
        fail(
            "protocol.quantization.nodes_to_exclude",
            "unique Conv node identities",
            f"duplicates={duplicates}",
            "remove repeated exclusions without changing graph order",
        )
    expected = expected_quantization_for_protocol(protocol_id)
    expected_excluded = expected["nodes_to_exclude"]
    if validated_excluded != expected_excluded:
        fail(
            "protocol.quantization.nodes_to_exclude",
            repr(expected_excluded),
            repr(validated_excluded),
            "restore the exact source-graph ordered protocol exclusion policy",
        )
    if quantization != expected:
        fail(
            "protocol.quantization",
            json.dumps(expected, sort_keys=True, separators=(",", ":")),
            json.dumps(quantization, sort_keys=True, separators=(",", ":")),
            "allow only the frozen exclusion and calibration-method differences",
        )
    return quantization


def _validate_model_contract(value: Any) -> Mapping[str, Any]:
    contract = _expect_mapping(value, "protocol.model_contract")
    _expect_exact_keys(
        contract,
        (
            "input_name",
            "input_shape",
            "input_dtype",
            "output_name",
            "output_shape",
            "output_dtype",
        ),
        "protocol.model_contract",
    )
    _expect_string(contract["input_name"], "protocol.model_contract.input_name")
    _expect_shape(contract["input_shape"], "protocol.model_contract.input_shape")
    _expect_string(
        contract["output_name"], "protocol.model_contract.output_name"
    )
    _expect_shape(
        contract["output_shape"], "protocol.model_contract.output_shape"
    )
    for key in ("input_dtype", "output_dtype"):
        if contract[key] != "float32":
            fail(
                f"protocol.model_contract.{key}",
                "float32",
                repr(contract[key]),
                "keep external model I/O float32; INT8 is an internal QDQ detail",
            )
    preprocess = EXPECTED_CALIBRATION_PREPROCESS
    if list(contract["input_shape"]) != list(preprocess["input_shape"]):
        fail(
            "protocol.model_contract.input_shape",
            "the same shape as calibration.preprocess.input_shape",
            repr(contract["input_shape"]),
            "make the calibration reader consume the declared model input",
        )
    return contract


def _validate_environment(value: Any) -> Mapping[str, str]:
    environment = _expect_mapping(value, "protocol.environment")
    _expect_exact_keys(
        environment, tuple(EXPECTED_ENVIRONMENT), "protocol.environment"
    )
    if environment != EXPECTED_ENVIRONMENT:
        fail(
            "protocol.environment",
            repr(dict(EXPECTED_ENVIRONMENT)),
            repr(environment),
            "select the pinned S2-01 interpreter and CPU provider",
        )
    return environment


def _validate_manifest_hash(
    path: Path, expected_hash: str, object_name: str
) -> None:
    actual_hash = sha256_file_canonical_lf(path)
    if actual_hash != expected_hash:
        fail(
            f"{object_name}.sha256_canonical_lf",
            expected_hash,
            actual_hash,
            "restore the frozen manifest bytes before formal PTQ",
        )


def _validate_correctness(
    value: Any, declaration_directory: Path
) -> Tuple[
    Mapping[str, Any],
    Path,
    Mapping[str, Any],
    Path,
    Mapping[str, Any],
]:
    correctness = _expect_mapping(value, "protocol.correctness")
    _expect_exact_keys(
        correctness,
        ("consistency_manifest", "quality_manifest", "gate_sources"),
        "protocol.correctness",
    )

    consistency_reference = _expect_mapping(
        correctness["consistency_manifest"],
        "protocol.correctness.consistency_manifest",
    )
    _expect_exact_keys(
        consistency_reference,
        ("path", "manifest_id", "sha256_canonical_lf", "sample_count"),
        "protocol.correctness.consistency_manifest",
    )
    consistency_path = _resolve_input_path(
        consistency_reference["path"],
        declaration_directory,
        "protocol.correctness.consistency_manifest.path",
    )
    consistency_id = _expect_string(
        consistency_reference["manifest_id"],
        "protocol.correctness.consistency_manifest.manifest_id",
    )
    if consistency_id != FROZEN_CONSISTENCY_MANIFEST_ID:
        fail(
            "protocol.correctness.consistency_manifest.manifest_id",
            FROZEN_CONSISTENCY_MANIFEST_ID,
            consistency_id,
            "restore the 6x5 product-consistency manifest reference",
        )
    consistency_hash = _expect_sha256(
        consistency_reference["sha256_canonical_lf"],
        "protocol.correctness.consistency_manifest.sha256_canonical_lf",
    )
    consistency_count = _expect_int(
        consistency_reference["sample_count"],
        "protocol.correctness.consistency_manifest.sample_count",
    )
    if consistency_count != FROZEN_CONSISTENCY_SAMPLE_COUNT:
        fail(
            "protocol.correctness.consistency_manifest.sample_count",
            str(FROZEN_CONSISTENCY_SAMPLE_COUNT),
            str(consistency_count),
            "restore all frozen product-consistency samples",
        )
    _validate_manifest_hash(
        consistency_path,
        consistency_hash,
        "protocol.correctness.consistency_manifest",
    )
    consistency_document = load_json(
        consistency_path, "correctness.consistency_manifest"
    )
    if consistency_document.get("schema_version") != SCHEMA_VERSION:
        fail(
            "correctness.consistency_manifest.schema_version",
            str(SCHEMA_VERSION),
            repr(consistency_document.get("schema_version")),
            "restore the tracked consistency manifest",
        )
    if consistency_document.get("manifest_id") != consistency_id:
        fail(
            "correctness.consistency_manifest.manifest_id",
            consistency_id,
            repr(consistency_document.get("manifest_id")),
            "point the protocol at the declared consistency manifest",
        )
    consistency_samples = consistency_document.get("samples")
    if not isinstance(consistency_samples, list) or len(
        consistency_samples
    ) != FROZEN_CONSISTENCY_SAMPLE_COUNT:
        fail(
            "correctness.consistency_manifest.samples",
            f"an array of {FROZEN_CONSISTENCY_SAMPLE_COUNT} samples",
            (
                type(consistency_samples).__name__
                if not isinstance(consistency_samples, list)
                else f"array length {len(consistency_samples)}"
            ),
            "restore the frozen product-consistency manifest",
        )

    quality_reference = _expect_mapping(
        correctness["quality_manifest"],
        "protocol.correctness.quality_manifest",
    )
    _expect_exact_keys(
        quality_reference,
        (
            "path",
            "manifest_id",
            "sha256_canonical_lf",
            "sample_count",
            "ground_truth_box_count",
        ),
        "protocol.correctness.quality_manifest",
    )
    quality_path = _resolve_input_path(
        quality_reference["path"],
        declaration_directory,
        "protocol.correctness.quality_manifest.path",
    )
    quality_id = _expect_string(
        quality_reference["manifest_id"],
        "protocol.correctness.quality_manifest.manifest_id",
    )
    if quality_id != FROZEN_QUALITY_MANIFEST_ID:
        fail(
            "protocol.correctness.quality_manifest.manifest_id",
            FROZEN_QUALITY_MANIFEST_ID,
            quality_id,
            "restore the full frozen validation-manifest reference",
        )
    quality_hash = _expect_sha256(
        quality_reference["sha256_canonical_lf"],
        "protocol.correctness.quality_manifest.sha256_canonical_lf",
    )
    quality_count = _expect_int(
        quality_reference["sample_count"],
        "protocol.correctness.quality_manifest.sample_count",
    )
    if quality_count != FROZEN_QUALITY_SAMPLE_COUNT:
        fail(
            "protocol.correctness.quality_manifest.sample_count",
            str(FROZEN_QUALITY_SAMPLE_COUNT),
            str(quality_count),
            "restore the complete frozen task-quality evaluation set",
        )
    ground_truth_count = _expect_int(
        quality_reference["ground_truth_box_count"],
        "protocol.correctness.quality_manifest.ground_truth_box_count",
    )
    if ground_truth_count != FROZEN_QUALITY_GROUND_TRUTH_BOX_COUNT:
        fail(
            "protocol.correctness.quality_manifest.ground_truth_box_count",
            str(FROZEN_QUALITY_GROUND_TRUTH_BOX_COUNT),
            str(ground_truth_count),
            "restore the full 857-box task-quality population",
        )
    _validate_manifest_hash(
        quality_path, quality_hash, "protocol.correctness.quality_manifest"
    )
    quality_document = load_json(quality_path, "correctness.quality_manifest")
    _expect_exact_keys(
        quality_document,
        (
            "schema_version",
            "manifest_kind",
            "manifest_id",
            "dataset",
            "preprocess",
            "evaluation",
            "product_matching_protocol",
            "product_matching_gates",
            "quality_gates",
            "classes",
            "samples",
            "integrity",
        ),
        "correctness.quality_manifest",
    )
    if quality_document["schema_version"] != SCHEMA_VERSION:
        fail(
            "correctness.quality_manifest.schema_version",
            str(SCHEMA_VERSION),
            repr(quality_document["schema_version"]),
            "restore the current quality-manifest schema",
        )
    if quality_document["manifest_kind"] != "detection_task_quality":
        fail(
            "correctness.quality_manifest.manifest_kind",
            "detection_task_quality",
            repr(quality_document["manifest_kind"]),
            "pass the labeled quality manifest",
        )
    if quality_document["manifest_id"] != quality_id:
        fail(
            "correctness.quality_manifest.manifest_id",
            quality_id,
            repr(quality_document["manifest_id"]),
            "point the protocol at the declared quality manifest",
        )
    quality_dataset = _expect_mapping(
        quality_document["dataset"], "correctness.quality_manifest.dataset"
    )
    if quality_dataset.get("sample_count") != quality_count:
        fail(
            "correctness.quality_manifest.dataset.sample_count",
            str(quality_count),
            repr(quality_dataset.get("sample_count")),
            "restore all frozen evaluation samples",
        )
    if quality_dataset.get("ground_truth_box_count") != ground_truth_count:
        fail(
            "correctness.quality_manifest.dataset.ground_truth_box_count",
            str(ground_truth_count),
            repr(quality_dataset.get("ground_truth_box_count")),
            "restore all frozen ground-truth boxes",
        )
    quality_samples = quality_document["samples"]
    if not isinstance(quality_samples, list) or len(quality_samples) != quality_count:
        fail(
            "correctness.quality_manifest.samples",
            f"an array of {quality_count} samples",
            (
                type(quality_samples).__name__
                if not isinstance(quality_samples, list)
                else f"array length {len(quality_samples)}"
            ),
            "restore the frozen labeled evaluation set",
        )
    quality_evaluation = _expect_frozen_mapping(
        quality_document["evaluation"],
        EXPECTED_QUALITY_EVALUATION,
        "correctness.quality_manifest.evaluation",
    )
    matching_protocol = _expect_frozen_mapping(
        quality_document["product_matching_protocol"],
        EXPECTED_PRODUCT_MATCHING_PROTOCOL,
        "correctness.quality_manifest.product_matching_protocol",
    )
    product_gates = _expect_frozen_mapping(
        quality_document["product_matching_gates"],
        EXPECTED_PRODUCT_MATCHING_GATES,
        "correctness.quality_manifest.product_matching_gates",
    )
    quality_gates = _expect_frozen_mapping(
        quality_document["quality_gates"],
        EXPECTED_QUALITY_GATES,
        "correctness.quality_manifest.quality_gates",
    )
    _expect_frozen_mapping(
        correctness["gate_sources"],
        EXPECTED_CORRECTNESS_GATE_SOURCES,
        "protocol.correctness.gate_sources",
    )
    return (
        correctness,
        consistency_path,
        consistency_document,
        quality_path,
        {
            "document": quality_document,
            "quality_evaluation": quality_evaluation,
            "product_matching_protocol": matching_protocol,
            "product_matching_gates": product_gates,
            "quality_gates": quality_gates,
        },
    )


def _validate_benchmark(
    value: Any,
    declaration_directory: Path,
    consistency_path: Path,
    consistency_document: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Path]:
    benchmark = _expect_mapping(value, "protocol.benchmark")
    _expect_exact_keys(
        benchmark,
        (*EXPECTED_BENCHMARK_SETTINGS.keys(), "sample"),
        "protocol.benchmark",
    )
    for field_name, expected_value in EXPECTED_BENCHMARK_SETTINGS.items():
        if benchmark[field_name] != expected_value:
            fail(
                f"protocol.benchmark.{field_name}",
                repr(expected_value),
                repr(benchmark[field_name]),
                "restore the same-machine unprofiled Release/CPU protocol",
            )
    sample = _expect_mapping(benchmark["sample"], "protocol.benchmark.sample")
    _expect_exact_keys(
        sample,
        ("sample_id", "image_path", "image_sha256"),
        "protocol.benchmark.sample",
    )
    sample_id = _expect_string(
        sample["sample_id"], "protocol.benchmark.sample.sample_id"
    )
    if sample_id != "crazing_241":
        fail(
            "protocol.benchmark.sample.sample_id",
            "crazing_241",
            sample_id,
            "restore the fixed single-image benchmark sample",
        )
    sample_path = _resolve_input_path(
        sample["image_path"],
        declaration_directory,
        "protocol.benchmark.sample.image_path",
    )
    sample_hash = _expect_sha256(
        sample["image_sha256"], "protocol.benchmark.sample.image_sha256"
    )
    actual_hash = sha256_file_raw(sample_path)
    if actual_hash != sample_hash:
        fail(
            "protocol.benchmark.sample.image_sha256",
            sample_hash,
            actual_hash,
            "restore the fixed raw benchmark image bytes",
        )
    consistency_matches = [
        item
        for item in consistency_document["samples"]
        if isinstance(item, dict) and item.get("sample_id") == sample_id
    ]
    if len(consistency_matches) != 1:
        fail(
            "protocol.benchmark.sample.sample_id",
            "exactly one matching consistency-manifest sample",
            f"matches={len(consistency_matches)}",
            "restore the fixed product-consistency manifest",
        )
    consistency_sample = consistency_matches[0]
    consistency_sample_hash = _expect_sha256(
        consistency_sample.get("image_sha256"),
        "correctness.consistency_manifest.sample[crazing_241].image_sha256",
    )
    consistency_sample_path = _resolve_input_path(
        consistency_sample.get("image_path"),
        consistency_path.parent,
        "correctness.consistency_manifest.sample[crazing_241].image_path",
    )
    if consistency_sample_hash != sample_hash or consistency_sample_path != sample_path:
        fail(
            "protocol.benchmark.sample",
            "the exact crazing_241 path and raw SHA from consistency_manifest",
            (
                f"path={sample_path}, sha={sample_hash}; consistency_path="
                f"{consistency_sample_path}, consistency_sha={consistency_sample_hash}"
            ),
            "remove the independent benchmark-sample drift",
        )
    return benchmark, sample_path


def _validate_profiling(value: Any) -> Mapping[str, Any]:
    return _expect_frozen_mapping(
        value, EXPECTED_PROFILING, "protocol.profiling"
    )


def _validate_expected_class_counts(value: Any) -> Mapping[str, int]:
    counts = _expect_mapping(
        value, "protocol.calibration.expected_class_counts"
    )
    if not counts:
        fail(
            "protocol.calibration.expected_class_counts",
            "at least one declared source class",
            "empty object",
            "freeze the balanced calibration population",
        )
    validated: Dict[str, int] = {}
    for raw_name, raw_count in counts.items():
        name = _expect_string(
            raw_name, "protocol.calibration.expected_class_counts.class_name"
        )
        validated[name] = _expect_int(
            raw_count,
            f"protocol.calibration.expected_class_counts[{name}]",
            minimum=1,
        )
    if sum(validated.values()) != FROZEN_CALIBRATION_SAMPLE_COUNT:
        fail(
            "protocol.calibration.expected_class_counts.total",
            str(FROZEN_CALIBRATION_SAMPLE_COUNT),
            str(sum(validated.values())),
            "restore the frozen 180-image class allocation",
        )
    frozen_counts = {
        entry["class_name"]: 30 for entry in EXPECTED_CALIBRATION_CLASSES
    }
    if validated != frozen_counts:
        fail(
            "protocol.calibration.expected_class_counts",
            repr(frozen_counts),
            repr(validated),
            "restore the six balanced NEU-DET source-class populations",
        )
    return validated


def _load_calibration_manifest(
    manifest_path: Path,
    expected_manifest_id: str,
    expected_hash: str,
    expected_class_counts: Mapping[str, int],
) -> Tuple[Mapping[str, Any], Tuple[CalibrationSample, ...]]:
    actual_manifest_hash = sha256_file_canonical_lf(manifest_path)
    if actual_manifest_hash != expected_hash:
        fail(
            "calibration.manifest_sha256_canonical_lf",
            expected_hash,
            actual_manifest_hash,
            "restore the frozen manifest bytes before the first formal PTQ run",
        )

    manifest = load_json(manifest_path, "calibration.manifest")
    _expect_exact_keys(
        manifest,
        (
            "schema_version",
            "manifest_kind",
            "manifest_id",
            "dataset",
            "preprocess",
            "classes",
            "samples",
            "integrity",
        ),
        "calibration.manifest",
    )
    if manifest["schema_version"] != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        fail(
            "calibration.manifest.schema_version",
            str(CALIBRATION_MANIFEST_SCHEMA_VERSION),
            repr(manifest["schema_version"]),
            "restore the current manifest schema",
        )
    if manifest["manifest_kind"] != "static_ptq_calibration":
        fail(
            "calibration.manifest.manifest_kind",
            "static_ptq_calibration",
            repr(manifest["manifest_kind"]),
            "pass the calibration manifest rather than the quality manifest",
        )
    manifest_id = _expect_string(
        manifest["manifest_id"], "calibration.manifest.manifest_id"
    )
    if manifest_id != expected_manifest_id:
        fail(
            "calibration.manifest.manifest_id",
            expected_manifest_id,
            manifest_id,
            "point the protocol at its frozen calibration manifest",
        )
    _expect_frozen_mapping(
        manifest["dataset"],
        EXPECTED_CALIBRATION_DATASET,
        "calibration.manifest.dataset",
    )
    _expect_frozen_mapping(
        manifest["preprocess"],
        EXPECTED_CALIBRATION_PREPROCESS,
        "calibration.manifest.preprocess",
    )

    raw_classes = manifest["classes"]
    if not isinstance(raw_classes, list):
        fail(
            "calibration.manifest.classes",
            "the frozen six-entry class array",
            type(raw_classes).__name__,
            "restore the class_id/class_name declarations",
        )
    declared_classes = []
    for index, raw_class in enumerate(raw_classes):
        object_name = f"calibration.manifest.classes[{index}]"
        class_entry = _expect_mapping(raw_class, object_name)
        _expect_exact_keys(
            class_entry, ("class_id", "class_name"), object_name
        )
        declared_classes.append(
            {
                "class_id": _expect_int(
                    class_entry["class_id"], f"{object_name}.class_id"
                ),
                "class_name": _expect_string(
                    class_entry["class_name"], f"{object_name}.class_name"
                ),
            }
        )
    if declared_classes != list(EXPECTED_CALIBRATION_CLASSES):
        fail(
            "calibration.manifest.classes",
            repr(list(EXPECTED_CALIBRATION_CLASSES)),
            repr(declared_classes),
            "restore the ordered NEU-DET class declarations",
        )

    integrity = _expect_mapping(
        manifest["integrity"], "calibration.manifest.integrity"
    )
    _expect_exact_keys(
        integrity,
        (
            "sample_row_format",
            "sample_set_sha256",
            "image_hash_semantics",
        ),
        "calibration.manifest.integrity",
    )
    if integrity["sample_row_format"] != CALIBRATION_SAMPLE_ROW_FORMAT:
        fail(
            "calibration.manifest.integrity.sample_row_format",
            CALIBRATION_SAMPLE_ROW_FORMAT,
            repr(integrity["sample_row_format"]),
            "restore the frozen row serialization contract",
        )
    if integrity["image_hash_semantics"] != CALIBRATION_IMAGE_HASH_SEMANTICS:
        fail(
            "calibration.manifest.integrity.image_hash_semantics",
            CALIBRATION_IMAGE_HASH_SEMANTICS,
            repr(integrity["image_hash_semantics"]),
            "hash each image's raw file bytes",
        )
    declared_sample_set_hash = _expect_sha256(
        integrity["sample_set_sha256"],
        "calibration.manifest.integrity.sample_set_sha256",
    )

    raw_samples = manifest["samples"]
    if not isinstance(raw_samples, list):
        fail(
            "calibration.manifest.samples",
            "a JSON array of exactly 180 samples",
            type(raw_samples).__name__,
            "restore the frozen sample list",
        )
    if len(raw_samples) != FROZEN_CALIBRATION_SAMPLE_COUNT:
        fail(
            "calibration.manifest.samples.count",
            str(FROZEN_CALIBRATION_SAMPLE_COUNT),
            str(len(raw_samples)),
            "restore all frozen calibration entries",
        )

    sample_rows = []
    for index, raw_sample in enumerate(raw_samples):
        object_name = f"calibration.manifest.samples[{index}]"
        sample = _expect_mapping(raw_sample, object_name)
        _expect_exact_keys(
            sample,
            (
                "sample_id",
                "source_class_id",
                "source_class_name",
                "image_path",
                "image_sha256",
            ),
            object_name,
        )
        sample_rows.append(
            (
                sample["sample_id"],
                sample["source_class_id"],
                sample["source_class_name"],
                sample["image_path"],
                sample["image_sha256"],
            )
        )
    actual_sample_set_hash = canonical_rows_sha256(sample_rows)
    if actual_sample_set_hash != declared_sample_set_hash:
        fail(
            "calibration.manifest.integrity.sample_set_sha256",
            declared_sample_set_hash,
            actual_sample_set_hash,
            "restore the exact ordered sample declarations",
        )

    sample_ids = set()
    image_paths = set()
    image_hashes = set()
    declared_name_to_id = {
        entry["class_name"]: entry["class_id"] for entry in declared_classes
    }
    class_counts: Counter = Counter()
    resolved_samples = []
    for index, raw_sample in enumerate(raw_samples):
        object_name = f"calibration.manifest.samples[{index}]"
        sample = _expect_mapping(raw_sample, object_name)
        _expect_exact_keys(
            sample,
            (
                "sample_id",
                "source_class_id",
                "source_class_name",
                "image_path",
                "image_sha256",
            ),
            object_name,
        )
        sample_id = _expect_string(sample["sample_id"], f"{object_name}.sample_id")
        if sample_id in sample_ids:
            fail(
                f"{object_name}.sample_id",
                "a unique sample identifier",
                sample_id,
                "remove the duplicate calibration entry",
            )
        sample_ids.add(sample_id)

        class_id = _expect_int(
            sample["source_class_id"], f"{object_name}.source_class_id"
        )
        class_name = _expect_string(
            sample["source_class_name"], f"{object_name}.source_class_name"
        )
        if class_name not in expected_class_counts:
            fail(
                f"{object_name}.source_class_name",
                f"one of {sorted(expected_class_counts)}",
                class_name,
                "restore the frozen balanced class allocation",
            )
        if declared_name_to_id[class_name] != class_id:
            fail(
                f"{object_name}.source_class_id",
                f"declared id {declared_name_to_id[class_name]} for class {class_name}",
                str(class_id),
                "restore the manifest class declaration/sample relationship",
            )
        class_counts[class_name] += 1

        image_path = _resolve_input_path(
            sample["image_path"], manifest_path.parent, f"{object_name}.image_path"
        )
        canonical_image_path = str(image_path).casefold()
        if canonical_image_path in image_paths:
            fail(
                f"{object_name}.image_path",
                "a calibration image used exactly once",
                str(image_path),
                "remove the duplicate path",
            )
        image_paths.add(canonical_image_path)
        expected_image_hash = _expect_sha256(
            sample["image_sha256"], f"{object_name}.image_sha256"
        )
        if expected_image_hash in image_hashes:
            fail(
                f"{object_name}.image_sha256",
                "a unique raw image digest",
                expected_image_hash,
                "remove the duplicate calibration image bytes",
            )
        image_hashes.add(expected_image_hash)
        actual_image_hash = sha256_file_raw(image_path)
        if actual_image_hash != expected_image_hash:
            fail(
                f"{object_name}.image_sha256",
                expected_image_hash,
                actual_image_hash,
                "restore the frozen raw image bytes; do not recalibrate on drifted data",
            )
        resolved_samples.append(
            CalibrationSample(
                sample_id=sample_id,
                source_class_id=class_id,
                source_class_name=class_name,
                image_path=image_path,
                image_sha256=expected_image_hash,
            )
        )

    if dict(class_counts) != dict(expected_class_counts):
        fail(
            "calibration.manifest.samples.class_counts",
            repr(dict(expected_class_counts)),
            repr(dict(class_counts)),
            "restore the frozen balanced 180-image population",
        )
    return manifest, tuple(resolved_samples)


def load_s2_01_protocol(path: Path) -> FrozenS201Protocol:
    """Load and fully validate protocol, manifest, source, and 180 images."""

    try:
        declaration_path = Path(path).resolve(strict=True)
    except OSError as error:
        fail(
            "protocol.path",
            "an existing regular JSON file",
            f"{path}: {error}",
            "pass --protocol with the frozen S2-01 JSON path",
        )
    if not declaration_path.is_file():
        fail(
            "protocol.path",
            "an existing regular JSON file",
            str(declaration_path),
            "pass a file rather than a directory",
        )

    document = load_json(declaration_path, "protocol")
    _expect_exact_keys(
        document,
        (
            "schema_version",
            "protocol_id",
            "source_model",
            "calibration",
            "quantization",
            "model_contract",
            "environment",
            "correctness",
            "benchmark",
            "profiling",
            "output",
        ),
        "protocol",
    )
    if document["schema_version"] != SCHEMA_VERSION:
        fail(
            "protocol.schema_version",
            str(SCHEMA_VERSION),
            repr(document["schema_version"]),
            "use the current machine protocol schema",
        )
    protocol_id = _expect_string(document["protocol_id"], "protocol.protocol_id")
    if protocol_id not in SUPPORTED_PROTOCOL_IDS:
        fail(
            "protocol.protocol_id",
            f"one of {list(SUPPORTED_PROTOCOL_IDS)}",
            protocol_id,
            "select one of the eleven frozen S2-01 PTQ protocols",
        )

    source = _expect_mapping(document["source_model"], "protocol.source_model")
    _expect_exact_keys(
        source, ("path", "sha256", "size_bytes"), "protocol.source_model"
    )
    source_model_path = _resolve_input_path(
        source["path"], declaration_path.parent, "protocol.source_model.path"
    )
    source_model_sha256 = _expect_sha256(
        source["sha256"], "protocol.source_model.sha256"
    )
    source_model_size_bytes = _expect_int(
        source["size_bytes"], "protocol.source_model.size_bytes", minimum=1
    )
    actual_source_size = source_model_path.stat().st_size
    if actual_source_size != source_model_size_bytes:
        fail(
            "source_model.size_bytes",
            str(source_model_size_bytes),
            str(actual_source_size),
            "restore the frozen FP32 source before quantization",
        )
    actual_source_hash = sha256_file_raw(source_model_path)
    if actual_source_hash != source_model_sha256:
        fail(
            "source_model.sha256",
            source_model_sha256,
            actual_source_hash,
            "restore the exact FP32 source ONNX; do not rewrite the protocol after drift",
        )

    calibration = _expect_mapping(
        document["calibration"], "protocol.calibration"
    )
    _expect_exact_keys(
        calibration,
        (
            "manifest_path",
            "manifest_id",
            "manifest_sha256_canonical_lf",
            "sample_count",
            "expected_class_counts",
            "preprocess",
        ),
        "protocol.calibration",
    )
    manifest_path = _resolve_input_path(
        calibration["manifest_path"],
        declaration_path.parent,
        "protocol.calibration.manifest_path",
    )
    manifest_id = _expect_string(
        calibration["manifest_id"], "protocol.calibration.manifest_id"
    )
    manifest_hash = _expect_sha256(
        calibration["manifest_sha256_canonical_lf"],
        "protocol.calibration.manifest_sha256_canonical_lf",
    )
    sample_count = _expect_int(
        calibration["sample_count"], "protocol.calibration.sample_count"
    )
    if sample_count != FROZEN_CALIBRATION_SAMPLE_COUNT:
        fail(
            "protocol.calibration.sample_count",
            str(FROZEN_CALIBRATION_SAMPLE_COUNT),
            str(sample_count),
            "restore the predeclared formal calibration count",
        )
    expected_class_counts = _validate_expected_class_counts(
        calibration["expected_class_counts"]
    )
    calibration_preprocess = _expect_frozen_mapping(
        calibration["preprocess"],
        EXPECTED_CALIBRATION_PREPROCESS,
        "protocol.calibration.preprocess",
    )
    manifest, samples = _load_calibration_manifest(
        manifest_path, manifest_id, manifest_hash, expected_class_counts
    )

    quantization = _validate_quantization(document["quantization"], protocol_id)
    model_contract = _validate_model_contract(document["model_contract"])
    environment = _validate_environment(document["environment"])
    (
        correctness,
        consistency_manifest_path,
        consistency_document,
        quality_manifest_path,
        quality_validation,
    ) = _validate_correctness(document["correctness"], declaration_path.parent)
    benchmark, benchmark_sample_path = _validate_benchmark(
        document["benchmark"],
        declaration_path.parent,
        consistency_manifest_path,
        consistency_document,
    )
    profiling = _validate_profiling(document["profiling"])

    output = _expect_mapping(document["output"], "protocol.output")
    _expect_exact_keys(
        output, ("model_path", "report_path"), "protocol.output"
    )
    output_model_path = _resolve_output_path(
        output["model_path"], declaration_path.parent, "protocol.output.model_path"
    )
    output_report_path = _resolve_output_path(
        output["report_path"], declaration_path.parent, "protocol.output.report_path"
    )
    if output_model_path == output_report_path:
        fail(
            "protocol.output",
            "different model and report paths",
            str(output_model_path),
            "choose one .onnx path and one .json path",
        )
    protected_paths = {
        declaration_path,
        source_model_path,
        manifest_path,
        consistency_manifest_path,
        quality_manifest_path,
        benchmark_sample_path,
        *(sample.image_path for sample in samples),
    }
    for name, output_path in (
        ("model_path", output_model_path),
        ("report_path", output_report_path),
    ):
        if output_path in protected_paths:
            fail(
                f"protocol.output.{name}",
                "a path different from every frozen input",
                str(output_path),
                "protect source, protocol, manifest, and calibration images from overwrite",
            )

    return FrozenS201Protocol(
        declaration_path=declaration_path,
        document=document,
        protocol_id=protocol_id,
        source_model_path=source_model_path,
        source_model_sha256=source_model_sha256,
        source_model_size_bytes=source_model_size_bytes,
        calibration_manifest_path=manifest_path,
        calibration_manifest_document=manifest,
        calibration_manifest_id=manifest_id,
        calibration_manifest_sha256_canonical_lf=manifest_hash,
        calibration_samples=samples,
        calibration_preprocess=calibration_preprocess,
        quantization=quantization,
        model_contract=model_contract,
        environment=environment,
        correctness=correctness,
        consistency_manifest_path=consistency_manifest_path,
        quality_manifest_path=quality_manifest_path,
        quality_evaluation=quality_validation["quality_evaluation"],
        product_matching_protocol=quality_validation[
            "product_matching_protocol"
        ],
        product_matching_gates=quality_validation["product_matching_gates"],
        quality_gates=quality_validation["quality_gates"],
        benchmark=benchmark,
        benchmark_sample_path=benchmark_sample_path,
        profiling=profiling,
        output_model_path=output_model_path,
        output_report_path=output_report_path,
    )

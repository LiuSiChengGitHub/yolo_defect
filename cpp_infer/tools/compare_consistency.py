#!/usr/bin/env python3
"""Compare the frozen YOLOv8/NEU-DET Python ORT and C++ ORT paths."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

try:
    import cv2
    import numpy as np
    import onnxruntime as ort
except ImportError as dependency_error:  # pragma: no cover - exercised by entry point
    cv2 = None
    np = None
    ort = None
    _DEPENDENCY_ERROR = dependency_error
else:
    _DEPENDENCY_ERROR = None


SCHEMA_VERSION = 1
EXPECTED_ORT_VERSION = "1.19.2"
CPU_PROVIDER = "CPUExecutionProvider"
PAD_VALUE = 114
FROZEN_REQUIREMENTS: Mapping[str, Any] = {
    "python_provider": CPU_PROVIDER,
    "detection_count": "exact",
    "class_id": "exact",
    "confidence_abs_error_max": 1.0e-4,
    "bbox_coordinate_abs_error_max_pixels": 1.0e-2,
    "matching_iou_min": 0.999,
    "matching_strategy":
        "class_id_then_greedy_max_iou_with_canonical_value_tie_break",
}

RUNTIME_FIELDS = {
    "schema_version",
    "artifact_spec_path",
    "score_threshold",
    "nms_threshold",
    "provider",
}
ARTIFACT_FIELDS = {
    "schema_version",
    "model_id",
    "model_family",
    "model_path",
    "model_sha256",
    "opset",
    "source",
    "provenance",
    "artifact_license",
    "input_name",
    "input_shape",
    "input_dtype",
    "input_layout",
    "output_name",
    "output_shape",
    "output_dtype",
    "output_layout",
    "class_names",
    "preprocess_type",
    "postprocess_type",
    "nms_mode",
}
MANIFEST_FIELDS = {
    "schema_version",
    "manifest_id",
    "selection",
    "config_path",
    "requirements",
    "classes",
    "samples",
}


class ConsistencyError(RuntimeError):
    """An actionable consistency setup, execution, or validation failure."""


def fail(object_name: str, expected: str, actual: str, action: str) -> None:
    raise ConsistencyError(
        "Consistency validation failed: object "
        f"{object_name}; expected {expected}; actual {actual}; action: {action}"
    )


def require_dependencies() -> None:
    if _DEPENDENCY_ERROR is not None:
        fail(
            "python.dependencies",
            "importable numpy, cv2, and onnxruntime 1.19.2",
            repr(_DEPENDENCY_ERROR),
            "run with the documented Python environment; do not install "
            "packages silently from this tool",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _reject_duplicate_json_keys(
    pairs: Iterable[Tuple[str, Any]],
) -> MutableMapping[str, Any]:
    value: MutableMapping[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail(
                f"json.field[{key}]",
                "each JSON field to occur once",
                "duplicate field",
                "remove the duplicate declaration",
            )
        value[key] = item
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(
                input_file, object_pairs_hook=_reject_duplicate_json_keys
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(
            f"json.path='{path}'",
            "one readable UTF-8 JSON document",
            str(error),
            "fix the file path, encoding, or JSON syntax",
        )
    if not isinstance(value, dict):
        fail(
            f"json.path='{path}'",
            "a JSON object at the document root",
            type(value).__name__,
            "replace the root value with an object",
        )
    return value


def validate_exact_fields(
    value: Mapping[str, Any], expected_fields: set, object_name: str
) -> None:
    missing = sorted(expected_fields - set(value))
    unknown = sorted(set(value) - expected_fields)
    if missing:
        fail(
            object_name,
            f"required fields {sorted(expected_fields)}",
            f"missing {missing}",
            "add every required field using the documented schema",
        )
    if unknown:
        fail(
            object_name,
            f"only fields {sorted(expected_fields)}",
            f"unknown {unknown}",
            "remove unknown fields or explicitly version the schema",
        )


def validate_frozen_requirements(requirements: Any) -> None:
    if requirements != FROZEN_REQUIREMENTS:
        fail(
            "manifest.requirements",
            repr(dict(FROZEN_REQUIREMENTS)),
            repr(requirements),
            "restore the predeclared S1-07 thresholds; do not tune after a run",
        )


def parse_key_values(path: Path, expected_fields: set) -> Mapping[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        fail(
            f"declaration.path='{path}'",
            "one readable UTF-8 key=value file",
            str(error),
            "correct the declaration path or file encoding",
        )

    values: Dict[str, str] = {}
    line_numbers: Dict[str, int] = {}
    for line_number, original_line in enumerate(lines, start=1):
        stripped = original_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            fail(
                f"{path}:{line_number}",
                "key = value",
                repr(original_line),
                "add one '=' separator",
            )
        key, raw_value = (part.strip() for part in stripped.split("=", 1))
        if not key or not raw_value:
            fail(
                f"{path}:{line_number}",
                "a non-empty key and value",
                repr(original_line),
                "fill both sides of the declaration",
            )
        if key in values:
            fail(
                f"{path}:{line_number}, field {key}",
                "one declaration",
                f"duplicate of line {line_numbers[key]}",
                "remove the duplicate field",
            )
        if key not in expected_fields:
            fail(
                f"{path}:{line_number}, field {key}",
                f"one of {sorted(expected_fields)}",
                "unknown field",
                "remove the field or explicitly extend the schema",
            )
        values[key] = raw_value
        line_numbers[key] = line_number

    missing = sorted(expected_fields - set(values))
    if missing:
        fail(
            f"declaration.path='{path}'",
            f"required fields {sorted(expected_fields)}",
            f"missing {missing}",
            "add every missing field",
        )
    return values


def parse_integer(raw_value: str, object_name: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]+", raw_value):
        fail(object_name, "an integer", repr(raw_value), "fix the number")
    return int(raw_value)


def parse_threshold(raw_value: str, object_name: str) -> float:
    try:
        value = float(raw_value)
    except ValueError:
        fail(
            object_name,
            "a finite number in [0,1]",
            repr(raw_value),
            "fix the threshold",
        )
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        fail(
            object_name,
            "a finite number in [0,1]",
            repr(raw_value),
            "fix the threshold without weakening the consistency gate",
        )
    return value


def parse_shape(raw_value: str, object_name: str) -> List[int]:
    parts = [part.strip() for part in raw_value.split(",")]
    if not parts or any(not part for part in parts):
        fail(
            object_name,
            "a comma-separated static shape",
            repr(raw_value),
            "provide every dimension",
        )
    shape = [parse_integer(part, object_name) for part in parts]
    if any(dimension <= 0 for dimension in shape):
        fail(
            object_name,
            "strictly positive static dimensions",
            repr(shape),
            "fix the artifact tensor declaration",
        )
    return shape


def load_contract(config_path: Path) -> Mapping[str, Any]:
    config_path = config_path.resolve(strict=True)
    runtime = parse_key_values(config_path, RUNTIME_FIELDS)
    if parse_integer(runtime["schema_version"], "runtime.schema_version") != 1:
        fail(
            "runtime.schema_version",
            "1",
            runtime["schema_version"],
            "use the supported RuntimeConfig v1",
        )
    artifact_path = (
        config_path.parent / runtime["artifact_spec_path"]
    ).resolve(strict=True)
    artifact = parse_key_values(artifact_path, ARTIFACT_FIELDS)
    if parse_integer(artifact["schema_version"], "artifact.schema_version") != 1:
        fail(
            "artifact.schema_version",
            "1",
            artifact["schema_version"],
            "use the supported ModelArtifactSpec v1",
        )

    model_path = (artifact_path.parent / artifact["model_path"]).resolve(strict=True)
    if not model_path.is_file():
        fail(
            "artifact.model_path",
            "an existing regular ONNX file",
            str(model_path),
            "restore the declared model artifact",
        )
    declared_model_sha = artifact["model_sha256"].upper()
    if not re.fullmatch(r"[0-9A-F]{64}", declared_model_sha):
        fail(
            "artifact.model_sha256",
            "64 hexadecimal characters",
            artifact["model_sha256"],
            "fix the declaration",
        )
    actual_model_sha = sha256_file(model_path)
    if actual_model_sha != declared_model_sha:
        fail(
            "artifact.model_sha256",
            declared_model_sha,
            actual_model_sha,
            "use the declared artifact or update provenance before comparing",
        )

    class_names = [name.strip() for name in artifact["class_names"].split(",")]
    if not class_names or any(not name for name in class_names):
        fail(
            "artifact.class_names",
            "at least one non-empty class name",
            repr(class_names),
            "fix the artifact declaration",
        )
    if len(set(class_names)) != len(class_names):
        fail(
            "artifact.class_names",
            "unique class names",
            repr(class_names),
            "remove duplicate class names",
        )

    input_shape = parse_shape(artifact["input_shape"], "artifact.input_shape")
    output_shape = parse_shape(artifact["output_shape"], "artifact.output_shape")
    baseline_values = {
        "runtime.provider": (runtime["provider"], "cpu"),
        "artifact.model_family": (artifact["model_family"], "yolov8"),
        "artifact.input_dtype": (artifact["input_dtype"], "float32"),
        "artifact.input_layout": (artifact["input_layout"], "nchw"),
        "artifact.output_dtype": (artifact["output_dtype"], "float32"),
        "artifact.output_layout": (artifact["output_layout"], "bcn"),
        "artifact.preprocess_type": (
            artifact["preprocess_type"], "letterbox_rgb_0_1_nchw"
        ),
        "artifact.postprocess_type": (
            artifact["postprocess_type"], "yolov8_raw"
        ),
        "artifact.nms_mode": (artifact["nms_mode"], "class_agnostic"),
    }
    for object_name, (actual, expected) in baseline_values.items():
        if actual != expected:
            fail(
                object_name,
                expected,
                actual,
                "use the frozen YOLOv8 CPU baseline contract",
            )
    if input_shape[:2] != [1, 3] or len(input_shape) != 4:
        fail(
            "artifact.input_shape",
            "float32 NCHW [1,3,H,W]",
            repr(input_shape),
            "fix the artifact input declaration",
        )
    if len(output_shape) != 3 or output_shape[0] != 1:
        fail(
            "artifact.output_shape",
            "BCN [1,4+C,N]",
            repr(output_shape),
            "fix the artifact output declaration",
        )
    expected_channels = 4 + len(class_names)
    if output_shape[1] != expected_channels:
        fail(
            "artifact.output_shape.channels",
            str(expected_channels),
            str(output_shape[1]),
            "align output channels with class_names",
        )

    return {
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "artifact_path": artifact_path,
        "artifact_sha256": sha256_file(artifact_path),
        "model_path": model_path,
        "model_actual_sha256": actual_model_sha,
        "model_declared_sha256": declared_model_sha,
        "model_id": artifact["model_id"],
        "opset": parse_integer(artifact["opset"], "artifact.opset"),
        "input_name": artifact["input_name"],
        "input_shape": input_shape,
        "output_name": artifact["output_name"],
        "output_shape": output_shape,
        "class_names": class_names,
        "score_threshold": parse_threshold(
            runtime["score_threshold"], "runtime.score_threshold"
        ),
        "nms_threshold": parse_threshold(
            runtime["nms_threshold"], "runtime.nms_threshold"
        ),
        "nms_mode": artifact["nms_mode"],
    }


def load_manifest(manifest_path: Path, contract: Mapping[str, Any]) -> Mapping[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = load_json(manifest_path)
    validate_exact_fields(manifest, MANIFEST_FIELDS, "consistency_manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        fail(
            "manifest.schema_version",
            str(SCHEMA_VERSION),
            repr(manifest["schema_version"]),
            "use the supported consistency manifest schema",
        )
    if not isinstance(manifest["manifest_id"], str) or not manifest["manifest_id"]:
        fail(
            "manifest.manifest_id",
            "a non-empty string",
            repr(manifest["manifest_id"]),
            "provide a stable manifest identifier",
        )
    validate_frozen_requirements(manifest["requirements"])

    declared_config = manifest["config_path"]
    if not isinstance(declared_config, str) or not declared_config:
        fail(
            "manifest.config_path",
            "a non-empty declaration-relative path",
            repr(declared_config),
            "point the manifest at the baseline RuntimeConfig",
        )
    resolved_config = (manifest_path.parent / declared_config).resolve(strict=True)
    if resolved_config != contract["config_path"]:
        fail(
            "manifest.config_path",
            str(contract["config_path"]),
            str(resolved_config),
            "use one identical RuntimeConfig for Python and C++",
        )

    classes = manifest["classes"]
    if not isinstance(classes, list):
        fail("manifest.classes", "a list", type(classes).__name__, "fix the manifest")
    expected_classes = [
        {"class_id": index, "class_name": name}
        for index, name in enumerate(contract["class_names"])
    ]
    if classes != expected_classes:
        fail(
            "manifest.classes",
            repr(expected_classes),
            repr(classes),
            "preserve artifact class order and ids",
        )
    selection = manifest["selection"]
    if not isinstance(selection, dict) or selection.get("samples_per_class") != 5:
        fail(
            "manifest.selection.samples_per_class",
            "5",
            repr(selection),
            "keep the frozen six-class, five-image sampling rule",
        )

    samples = manifest["samples"]
    if not isinstance(samples, list) or len(samples) != 30:
        fail(
            "manifest.samples",
            "exactly 30 samples",
            f"count={len(samples) if isinstance(samples, list) else 'not-a-list'}",
            "provide exactly five images for each of six classes",
        )
    required_sample_fields = {
        "sample_id",
        "source_class_id",
        "source_class_name",
        "image_path",
        "image_sha256",
    }
    resolved_samples: List[Mapping[str, Any]] = []
    seen_ids = set()
    seen_paths = set()
    class_counts: collections.Counter = collections.Counter()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail(
                f"manifest.samples[{index}]",
                "an object",
                type(sample).__name__,
                "fix the sample entry",
            )
        validate_exact_fields(
            sample, required_sample_fields, f"manifest.samples[{index}]"
        )
        sample_id = sample["sample_id"]
        class_id = sample["source_class_id"]
        class_name = sample["source_class_name"]
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            fail(
                f"manifest.samples[{index}].sample_id",
                "a unique non-empty string",
                repr(sample_id),
                "fix the sample id",
            )
        if not isinstance(class_id, int) or not 0 <= class_id < len(classes):
            fail(
                f"manifest.samples[{index}].source_class_id",
                f"an integer in [0,{len(classes) - 1}]",
                repr(class_id),
                "use the artifact class id",
            )
        if class_name != contract["class_names"][class_id]:
            fail(
                f"manifest.samples[{index}].source_class_name",
                contract["class_names"][class_id],
                repr(class_name),
                "align the source class with the artifact class id",
            )
        declared_image = sample["image_path"]
        if not isinstance(declared_image, str) or not declared_image:
            fail(
                f"manifest.samples[{index}].image_path",
                "a non-empty declaration-relative path",
                repr(declared_image),
                "provide the tracked validation image path",
            )
        image_path = (manifest_path.parent / declared_image).resolve(strict=True)
        if not image_path.is_file() or image_path in seen_paths:
            fail(
                f"manifest.samples[{index}].image_path",
                "a unique existing regular image file",
                str(image_path),
                "fix duplicate or missing sample paths",
            )
        expected_hash = sample["image_sha256"]
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9A-Fa-f]{64}", expected_hash
        ):
            fail(
                f"manifest.samples[{index}].image_sha256",
                "64 hexadecimal characters",
                repr(expected_hash),
                "record the exact image SHA-256",
            )
        actual_hash = sha256_file(image_path)
        if actual_hash != expected_hash.upper():
            fail(
                f"manifest.samples[{index}].image_sha256",
                expected_hash.upper(),
                actual_hash,
                "restore the frozen image or create a new manifest version",
            )
        seen_ids.add(sample_id)
        seen_paths.add(image_path)
        class_counts[class_id] += 1
        resolved_samples.append(
            {
                **sample,
                "image_sha256": expected_hash.upper(),
                "resolved_image_path": image_path,
            }
        )
    expected_counts = {class_id: 5 for class_id in range(len(classes))}
    if dict(class_counts) != expected_counts:
        fail(
            "manifest.samples.class_counts",
            repr(expected_counts),
            repr(dict(class_counts)),
            "provide exactly five frozen images per artifact class",
        )
    return {
        **manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "resolved_samples": resolved_samples,
        "class_counts": expected_counts,
    }


def cxx_round_positive(value: float) -> int:
    return int(math.floor(value + 0.5))


def preprocess_image(
    image_path: Path, input_shape: Sequence[int]
) -> Tuple[Any, Mapping[str, Any]]:
    require_dependencies()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        fail(
            f"python.image_path='{image_path}'",
            "an OpenCV-decodable color image",
            "decode returned empty",
            "restore the frozen manifest image",
        )
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        fail(
            f"python.image_path='{image_path}'",
            "uint8 HWC BGR with three channels",
            f"shape={image.shape}, dtype={image.dtype}",
            "use the same OpenCV IMREAD_COLOR semantics as C++",
        )
    original_height, original_width = image.shape[:2]
    input_height = int(input_shape[2])
    input_width = int(input_shape[3])
    scale = min(
        float(input_width) / float(original_width),
        float(input_height) / float(original_height),
    )
    resized_width = max(1, cxx_round_positive(original_width * scale))
    resized_height = max(1, cxx_round_positive(original_height * scale))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    pad_width = input_width - resized_width
    pad_height = input_height - resized_height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / np.float32(255.0)
    tensor = np.ascontiguousarray(
        np.transpose(rgb_float, (2, 0, 1))[np.newaxis, ...], dtype=np.float32
    )
    if list(tensor.shape) != list(input_shape) or not tensor.flags.c_contiguous:
        fail(
            "python.input_tensor",
            f"contiguous float32 shape {list(input_shape)}",
            f"shape={list(tensor.shape)}, dtype={tensor.dtype}, "
            f"contiguous={tensor.flags.c_contiguous}",
            "inspect letterbox and NCHW conversion",
        )
    if not np.isfinite(tensor).all():
        fail(
            "python.input_tensor.values",
            "all finite float32 values",
            "NaN or Infinity",
            "inspect decode and normalization",
        )
    return tensor, {
        "original_width": original_width,
        "original_height": original_height,
        "original_channels": int(image.shape[2]),
        "input_width": input_width,
        "input_height": input_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "scale": scale,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }


def create_python_session(contract: Mapping[str, Any]) -> Any:
    require_dependencies()
    if ort.__version__ != EXPECTED_ORT_VERSION:
        fail(
            "python.onnxruntime.version",
            EXPECTED_ORT_VERSION,
            ort.__version__,
            "run with the pinned Python ORT environment",
        )
    available = ort.get_available_providers()
    if CPU_PROVIDER not in available:
        fail(
            "python.onnxruntime.available_providers",
            f"a list containing {CPU_PROVIDER}",
            repr(available),
            "use an ONNX Runtime build with the CPU execution provider",
        )
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 2
    session = ort.InferenceSession(
        str(contract["model_path"]),
        sess_options=options,
        providers=[(CPU_PROVIDER, {"use_arena": "1"})],
    )
    if session.get_providers() != [CPU_PROVIDER]:
        fail(
            "python.session.providers",
            repr([CPU_PROVIDER]),
            repr(session.get_providers()),
            "remove fallback or accelerator providers from the Python session",
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        fail(
            "python.session.io_count",
            "one input and one output",
            f"inputs={len(inputs)}, outputs={len(outputs)}",
            "use the declared single-input/single-output model",
        )
    expected_metadata = [
        (
            "input[0]",
            inputs[0].name,
            inputs[0].type,
            list(inputs[0].shape),
            contract["input_name"],
            "tensor(float)",
            contract["input_shape"],
        ),
        (
            "output[0]",
            outputs[0].name,
            outputs[0].type,
            list(outputs[0].shape),
            contract["output_name"],
            "tensor(float)",
            contract["output_shape"],
        ),
    ]
    for (
        object_name,
        actual_name,
        actual_type,
        actual_shape,
        expected_name,
        expected_type,
        expected_shape,
    ) in expected_metadata:
        actual = (actual_name, actual_type, actual_shape)
        expected = (expected_name, expected_type, expected_shape)
        if actual != expected:
            fail(
                f"python.session.{object_name}",
                repr(expected),
                repr(actual),
                "verify the artifact path and tensor contract",
            )
    return session


def iou_float32(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    intersection_width = max(0.0, min(lhs[2], rhs[2]) - max(lhs[0], rhs[0]))
    intersection_height = max(0.0, min(lhs[3], rhs[3]) - max(lhs[1], rhs[1]))
    intersection = intersection_width * intersection_height
    lhs_area = max(0.0, lhs[2] - lhs[0]) * max(0.0, lhs[3] - lhs[1])
    rhs_area = max(0.0, rhs[2] - rhs[0]) * max(0.0, rhs[3] - rhs[1])
    union = lhs_area + rhs_area - intersection
    if union <= 0.0:
        return 0.0
    value = min(1.0, max(0.0, intersection / union))
    if np is None:
        return float(value)
    return float(np.float32(value))


def postprocess_raw_output(
    output: Any,
    contract: Mapping[str, Any],
    transform: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    require_dependencies()
    if not isinstance(output, np.ndarray):
        fail(
            "python.raw_output",
            "a NumPy tensor",
            type(output).__name__,
            "request the declared output from ONNX Runtime",
        )
    if output.dtype != np.float32 or list(output.shape) != contract["output_shape"]:
        fail(
            "python.raw_output",
            f"float32 shape {contract['output_shape']}",
            f"dtype={output.dtype}, shape={list(output.shape)}",
            "verify Python ORT output metadata and artifact contract",
        )
    if not np.isfinite(output).all():
        fail(
            "python.raw_output.values",
            "all finite float32 values",
            "NaN or Infinity",
            "inspect preprocess and ORT execution",
        )
    candidate_count = output.shape[2]
    score_threshold = np.float32(contract["score_threshold"])
    candidates: List[Mapping[str, Any]] = []
    for candidate_index in range(candidate_count):
        scores = output[0, 4:, candidate_index]
        class_id = int(np.argmax(scores))
        confidence = np.float32(scores[class_id])
        if not bool(confidence > score_threshold):
            continue
        center_x = np.float32(output[0, 0, candidate_index])
        center_y = np.float32(output[0, 1, candidate_index])
        half_width = np.float32(
            np.float32(output[0, 2, candidate_index]) / np.float32(2.0)
        )
        half_height = np.float32(
            np.float32(output[0, 3, candidate_index]) / np.float32(2.0)
        )
        candidates.append(
            {
                "class_id": class_id,
                "class_name": contract["class_names"][class_id],
                "confidence": float(confidence),
                "bbox_xyxy": [
                    float(np.float32(center_x - half_width)),
                    float(np.float32(center_y - half_height)),
                    float(np.float32(center_x + half_width)),
                    float(np.float32(center_y + half_height)),
                ],
                "candidate_index": candidate_index,
            }
        )

    order = sorted(
        range(len(candidates)), key=lambda index: -candidates[index]["confidence"]
    )
    suppressed = [False] * len(candidates)
    kept: List[Mapping[str, Any]] = []
    nms_threshold = np.float32(contract["nms_threshold"])
    for order_index, current in enumerate(order):
        if suppressed[current]:
            continue
        kept.append(candidates[current])
        for other in order[order_index + 1:]:
            if not suppressed[other] and bool(
                np.float32(
                    iou_float32(
                        candidates[current]["bbox_xyxy"],
                        candidates[other]["bbox_xyxy"],
                    )
                )
                > nms_threshold
            ):
                suppressed[other] = True

    restored: List[Mapping[str, Any]] = []
    for detection in kept:
        model_box = detection["bbox_xyxy"]
        x_values = [
            (float(model_box[index]) - transform["pad_left"]) / transform["scale"]
            for index in (0, 2)
        ]
        y_values = [
            (float(model_box[index]) - transform["pad_top"]) / transform["scale"]
            for index in (1, 3)
        ]
        restored.append(
            {
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "bbox_xyxy": [
                    float(np.float32(min(max(x_values[0], 0.0), transform["original_width"]))),
                    float(np.float32(min(max(y_values[0], 0.0), transform["original_height"]))),
                    float(np.float32(min(max(x_values[1], 0.0), transform["original_width"]))),
                    float(np.float32(min(max(y_values[1], 0.0), transform["original_height"]))),
                ],
            }
        )
    return restored


def detection_key(detection: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        int(detection["class_id"]),
        -float(detection["confidence"]),
        *(float(value) for value in detection["bbox_xyxy"]),
    )


def class_histogram(detections: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts = collections.Counter(int(item["class_id"]) for item in detections)
    return {str(class_id): counts[class_id] for class_id in sorted(counts)}


def match_detections(
    python_detections: Sequence[Mapping[str, Any]],
    cpp_detections: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    python_by_class: Dict[int, List[Tuple[int, Mapping[str, Any]]]] = collections.defaultdict(list)
    cpp_by_class: Dict[int, List[Tuple[int, Mapping[str, Any]]]] = collections.defaultdict(list)
    for index, detection in enumerate(python_detections):
        python_by_class[int(detection["class_id"])].append((index, detection))
    for index, detection in enumerate(cpp_detections):
        cpp_by_class[int(detection["class_id"])].append((index, detection))

    matches: List[Mapping[str, Any]] = []
    unmatched_python = set(range(len(python_detections)))
    unmatched_cpp = set(range(len(cpp_detections)))
    for class_id in sorted(set(python_by_class) | set(cpp_by_class)):
        edges = []
        for python_index, python_detection in python_by_class[class_id]:
            for cpp_index, cpp_detection in cpp_by_class[class_id]:
                iou = iou_float32(
                    python_detection["bbox_xyxy"], cpp_detection["bbox_xyxy"]
                )
                edges.append(
                    (
                        -iou,
                        detection_key(python_detection),
                        detection_key(cpp_detection),
                        python_index,
                        cpp_index,
                        iou,
                    )
                )
        edges.sort()
        used_python = set()
        used_cpp = set()
        for _, _, _, python_index, cpp_index, iou in edges:
            if python_index in used_python or cpp_index in used_cpp:
                continue
            used_python.add(python_index)
            used_cpp.add(cpp_index)
            unmatched_python.discard(python_index)
            unmatched_cpp.discard(cpp_index)
            python_detection = python_detections[python_index]
            cpp_detection = cpp_detections[cpp_index]
            bbox_errors = [
                abs(float(lhs) - float(rhs))
                for lhs, rhs in zip(
                    python_detection["bbox_xyxy"], cpp_detection["bbox_xyxy"]
                )
            ]
            matches.append(
                {
                    "class_id": class_id,
                    "python_index": python_index,
                    "cpp_index": cpp_index,
                    "python_detection": python_detection,
                    "cpp_detection": cpp_detection,
                    "confidence_abs_error": abs(
                        float(python_detection["confidence"])
                        - float(cpp_detection["confidence"])
                    ),
                    "bbox_coordinate_abs_errors": bbox_errors,
                    "bbox_coordinate_abs_error_max": max(bbox_errors),
                    "matching_iou": iou,
                }
            )
    matches.sort(
        key=lambda item: (
            item["class_id"],
            detection_key(item["python_detection"]),
            detection_key(item["cpp_detection"]),
        )
    )
    return {
        "matches": matches,
        "unmatched_python_indices": sorted(unmatched_python),
        "unmatched_cpp_indices": sorted(unmatched_cpp),
    }


def validate_cpp_json(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
    image_path: Path,
    transform: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    if value.get("schema_version") != 1:
        fail("cpp_json.schema_version", "1", repr(value.get("schema_version")), "use the S1-05 JSON schema")
    model = value.get("model", {})
    runtime = value.get("runtime", {})
    image = value.get("image", {})
    expected_contract_values = [
        ("model.model_id", model.get("model_id"), contract["model_id"]),
        (
            "model.declared_sha256",
            model.get("declared_sha256"),
            contract["model_declared_sha256"],
        ),
        ("runtime.actual_provider", runtime.get("actual_provider"), CPU_PROVIDER),
        (
            "runtime.provider_evidence",
            runtime.get("provider_evidence"),
            "explicit_cpu_ep_registration_and_session_creation",
        ),
        ("runtime.score_threshold", runtime.get("score_threshold"), contract["score_threshold"]),
        ("runtime.nms_threshold", runtime.get("nms_threshold"), contract["nms_threshold"]),
        ("runtime.nms_mode", runtime.get("nms_mode"), contract["nms_mode"]),
    ]
    for object_name, actual, expected in expected_contract_values:
        if actual != expected:
            fail(
                f"cpp_json.{object_name}",
                repr(expected),
                repr(actual),
                "ensure C++ uses the identical manifest RuntimeConfig",
            )
    cpp_image_path = image.get("path")
    if not isinstance(cpp_image_path, str) or os.path.normcase(
        os.path.realpath(cpp_image_path)
    ) != os.path.normcase(os.path.realpath(image_path)):
        fail(
            "cpp_json.image.path",
            str(image_path),
            repr(cpp_image_path),
            "compare the same frozen manifest image",
        )
    expected_original = {
        "width": transform["original_width"],
        "height": transform["original_height"],
        "channels": transform["original_channels"],
    }
    expected_input = {
        "width": transform["input_width"], "height": transform["input_height"]
    }
    if image.get("original_size") != expected_original or image.get("input_size") != expected_input:
        fail(
            "cpp_json.image.metadata",
            repr({"original_size": expected_original, "input_size": expected_input}),
            repr(image),
            "verify that both implementations decoded and resized the same image",
        )
    detections = value.get("detections")
    if not isinstance(detections, list):
        fail("cpp_json.detections", "an array", type(detections).__name__, "fix the C++ JSON output")
    validated = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            fail(f"cpp_json.detections[{index}]", "an object", type(detection).__name__, "fix the output schema")
        class_id = detection.get("class_id")
        if not isinstance(class_id, int) or not 0 <= class_id < len(contract["class_names"]):
            fail(f"cpp_json.detections[{index}].class_id", "a valid artifact class id", repr(class_id), "inspect C++ postprocess")
        if detection.get("class_name") != contract["class_names"][class_id]:
            fail(f"cpp_json.detections[{index}].class_name", contract["class_names"][class_id], repr(detection.get("class_name")), "inspect JSON serialization")
        confidence = detection.get("confidence")
        box = detection.get("bbox_xyxy")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            fail(f"cpp_json.detections[{index}].confidence", "a finite number", repr(confidence), "inspect C++ output validation")
        if not isinstance(box, list) or len(box) != 4 or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in box
        ):
            fail(f"cpp_json.detections[{index}].bbox_xyxy", "four finite coordinates", repr(box), "inspect C++ coordinate restore")
        validated.append(
            {
                "class_id": class_id,
                "class_name": detection["class_name"],
                "confidence": float(confidence),
                "bbox_xyxy": [float(coordinate) for coordinate in box],
            }
        )
    return validated


def parse_inspection_output(output: str) -> Mapping[str, str]:
    parsed: Dict[str, str] = {}
    for line in output.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            parsed[key.strip()] = value.strip()
    required = {
        "ort_version": EXPECTED_ORT_VERSION,
        "configured_provider": "cpu",
        "session_provider": CPU_PROVIDER,
        "provider_evidence":
            "explicit_cpu_ep_registration_and_session_creation",
        "execution_mode": "sequential",
        "intra_op_num_threads": "1",
        "inter_op_num_threads":
            "1 (not used by sequential execution mode)",
        "graph_optimization_level": "all",
        "metadata_contract_validation": "passed",
    }
    for key, expected in required.items():
        if parsed.get(key) != expected:
            fail(
                f"cpp_inspect.{key}",
                expected,
                repr(parsed.get(key)),
                "inspect the C++ ORT SDK, provider, and SessionOptions",
            )
    return parsed


def inspect_cpp_cli(cpp_cli: Path, config_path: Path) -> Mapping[str, str]:
    process = subprocess.run(
        [str(cpp_cli), "--config", str(config_path), "--inspect-model"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        fail(
            "cpp_cli.inspect_model",
            "exit code 0",
            f"exit={process.returncode}; stdout={process.stdout!r}; stderr={process.stderr!r}",
            "run --inspect-model manually and fix the SDK/model/provider error",
        )
    return parse_inspection_output(process.stdout)


def run_cpp_image(
    cpp_cli: Path, contract: Mapping[str, Any], image_path: Path, output_path: Path
) -> Mapping[str, Any]:
    process = subprocess.run(
        [
            str(cpp_cli),
            "--config",
            str(contract["config_path"]),
            "--image",
            str(image_path),
            "--output-json",
            str(output_path),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        fail(
            f"cpp_cli.image='{image_path}'",
            "exit code 0 and one detection JSON",
            f"exit={process.returncode}; stdout={process.stdout!r}; stderr={process.stderr!r}",
            "re-run the C++ CLI for this image and inspect contract/preprocess/ORT/postprocess",
        )
    if not output_path.is_file():
        fail(
            f"cpp_cli.output_json='{output_path}'",
            "one generated regular JSON file",
            "missing",
            "inspect the C++ output path policy and stdout",
        )
    return load_json(output_path)


def evaluate_image(
    sample: Mapping[str, Any],
    contract: Mapping[str, Any],
    session: Any,
    cpp_cli: Path,
    cpp_output_path: Path,
) -> Mapping[str, Any]:
    tensor, transform = preprocess_image(
        sample["resolved_image_path"], contract["input_shape"]
    )
    raw_outputs = session.run(
        [contract["output_name"]], {contract["input_name"]: tensor}
    )
    if len(raw_outputs) != 1:
        fail("python.session.run.outputs", "one output", str(len(raw_outputs)), "request only the declared output")
    python_detections = postprocess_raw_output(raw_outputs[0], contract, transform)
    cpp_json = run_cpp_image(
        cpp_cli,
        contract,
        sample["resolved_image_path"],
        cpp_output_path,
    )
    cpp_detections = validate_cpp_json(
        cpp_json, contract, sample["resolved_image_path"], transform
    )
    matching = match_detections(python_detections, cpp_detections)
    failures = []
    python_histogram = class_histogram(python_detections)
    cpp_histogram = class_histogram(cpp_detections)
    if len(python_detections) != len(cpp_detections):
        failures.append(
            f"detection_count expected exact; python={len(python_detections)}, cpp={len(cpp_detections)}"
        )
    if python_histogram != cpp_histogram:
        failures.append(
            f"class_id counts expected exact; python={python_histogram}, cpp={cpp_histogram}"
        )
    if matching["unmatched_python_indices"] or matching["unmatched_cpp_indices"]:
        failures.append(
            "deterministic class/IoU matching left unmatched detections: "
            f"python={matching['unmatched_python_indices']}, cpp={matching['unmatched_cpp_indices']}"
        )
    for match_index, match in enumerate(matching["matches"]):
        if match["python_detection"]["class_id"] != match["cpp_detection"]["class_id"]:
            failures.append(f"match[{match_index}] class_id differs")
        if match["confidence_abs_error"] > FROZEN_REQUIREMENTS["confidence_abs_error_max"]:
            failures.append(
                f"match[{match_index}] confidence_abs_error={match['confidence_abs_error']} exceeds "
                f"{FROZEN_REQUIREMENTS['confidence_abs_error_max']}"
            )
        if match["bbox_coordinate_abs_error_max"] > FROZEN_REQUIREMENTS["bbox_coordinate_abs_error_max_pixels"]:
            failures.append(
                f"match[{match_index}] bbox_coordinate_abs_error_max="
                f"{match['bbox_coordinate_abs_error_max']} exceeds "
                f"{FROZEN_REQUIREMENTS['bbox_coordinate_abs_error_max_pixels']}"
            )
        if match["matching_iou"] < FROZEN_REQUIREMENTS["matching_iou_min"]:
            failures.append(
                f"match[{match_index}] matching_iou={match['matching_iou']} below "
                f"{FROZEN_REQUIREMENTS['matching_iou_min']}"
            )
    confidence_errors = [item["confidence_abs_error"] for item in matching["matches"]]
    coordinate_errors = [
        error
        for item in matching["matches"]
        for error in item["bbox_coordinate_abs_errors"]
    ]
    ious = [item["matching_iou"] for item in matching["matches"]]
    return {
        "sample_id": sample["sample_id"],
        "source_class_id": sample["source_class_id"],
        "source_class_name": sample["source_class_name"],
        "image_path": sample["image_path"],
        "image_sha256": sample["image_sha256"],
        "python_detection_count": len(python_detections),
        "cpp_detection_count": len(cpp_detections),
        "python_class_id_counts": python_histogram,
        "cpp_class_id_counts": cpp_histogram,
        "python_detections": python_detections,
        "cpp_detections": cpp_detections,
        **matching,
        "metrics": {
            "max_confidence_abs_error": max(confidence_errors) if confidence_errors else None,
            "max_bbox_coordinate_abs_error_pixels": max(coordinate_errors) if coordinate_errors else None,
            "min_matching_iou": min(ious) if ious else None,
        },
        "passed": not failures,
        "failures": failures,
    }


def mean_or_none(values: Sequence[float]) -> Any:
    return sum(values) / len(values) if values else None


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(serialized)
    temporary_path.replace(path)


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_summary(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    inspection: Mapping[str, str],
    image_results: Sequence[Mapping[str, Any]],
    cpp_opencv_version: str,
) -> Mapping[str, Any]:
    matches = [match for image in image_results for match in image.get("matches", [])]
    confidence_errors = [match["confidence_abs_error"] for match in matches]
    coordinate_errors = [
        error for match in matches for error in match["bbox_coordinate_abs_errors"]
    ]
    ious = [match["matching_iou"] for match in matches]
    failed_images = [
        {"sample_id": image["sample_id"], "failures": image["failures"]}
        for image in image_results
        if not image["passed"]
    ]
    source_class_results = []
    for source_class_id, source_class_name in enumerate(contract["class_names"]):
        class_images = [
            image
            for image in image_results
            if image["source_class_id"] == source_class_id
        ]
        class_matches = [
            match for image in class_images for match in image.get("matches", [])
        ]
        class_confidence_errors = [
            match["confidence_abs_error"] for match in class_matches
        ]
        class_coordinate_errors = [
            error
            for match in class_matches
            for error in match["bbox_coordinate_abs_errors"]
        ]
        class_ious = [match["matching_iou"] for match in class_matches]
        source_class_results.append(
            {
                "source_class_id": source_class_id,
                "source_class_name": source_class_name,
                "images_total": len(class_images),
                "images_passed": sum(
                    1 for image in class_images if image["passed"]
                ),
                "python_detections_total": sum(
                    image["python_detection_count"] for image in class_images
                ),
                "cpp_detections_total": sum(
                    image["cpp_detection_count"] for image in class_images
                ),
                "matched_detections_total": len(class_matches),
                "max_confidence_abs_error": (
                    max(class_confidence_errors)
                    if class_confidence_errors
                    else None
                ),
                "max_bbox_coordinate_abs_error_pixels": (
                    max(class_coordinate_errors)
                    if class_coordinate_errors
                    else None
                ),
                "min_matching_iou": min(class_ious) if class_ious else None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "python_ort_vs_cpp_ort_detection_consistency",
        "passed": not failed_images and len(image_results) == 30,
        "manifest": {
            "manifest_id": manifest["manifest_id"],
            "path": display_path(manifest["manifest_path"]),
            "sha256": manifest["manifest_sha256"],
            "sample_count": len(manifest["resolved_samples"]),
            "class_count": len(manifest["classes"]),
            "samples_per_class": manifest["selection"]["samples_per_class"],
            "class_sample_counts": {
                str(key): value for key, value in manifest["class_counts"].items()
            },
        },
        "contract": {
            "config_path": display_path(contract["config_path"]),
            "config_sha256": contract["config_sha256"],
            "artifact_path": display_path(contract["artifact_path"]),
            "artifact_sha256": contract["artifact_sha256"],
            "model_id": contract["model_id"],
            "model_declared_sha256": contract["model_declared_sha256"],
            "model_actual_sha256": contract["model_actual_sha256"],
            "opset": contract["opset"],
            "input_name": contract["input_name"],
            "input_shape": contract["input_shape"],
            "output_name": contract["output_name"],
            "output_shape": contract["output_shape"],
            "class_names": contract["class_names"],
            "score_threshold": contract["score_threshold"],
            "nms_threshold": contract["nms_threshold"],
            "nms_mode": contract["nms_mode"],
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_onnxruntime_version": ort.__version__,
            "python_opencv_version": cv2.__version__,
            "python_numpy_version": np.__version__,
            "python_available_providers": ort.get_available_providers(),
            "python_session_providers": [CPU_PROVIDER],
            "cpp_onnxruntime_version": inspection["ort_version"],
            "cpp_opencv_version": cpp_opencv_version,
            "cpp_opencv_version_source": "CMake OpenCV_VERSION passed by the comparison command",
            "cpp_session_provider": inspection["session_provider"],
            "cpp_provider_evidence": inspection.get("provider_evidence"),
            "execution_mode": "sequential",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "graph_optimization_level": "all",
            "platform": platform.platform(),
        },
        "requirements": dict(FROZEN_REQUIREMENTS),
        "result": {
            "images_total": len(image_results),
            "images_passed": sum(1 for image in image_results if image["passed"]),
            "images_failed": len(failed_images),
            "python_detections_total": sum(image["python_detection_count"] for image in image_results),
            "cpp_detections_total": sum(image["cpp_detection_count"] for image in image_results),
            "matched_detections_total": len(matches),
            "max_confidence_abs_error": max(confidence_errors) if confidence_errors else None,
            "mean_confidence_abs_error": mean_or_none(confidence_errors),
            "max_bbox_coordinate_abs_error_pixels": max(coordinate_errors) if coordinate_errors else None,
            "mean_bbox_coordinate_abs_error_pixels": mean_or_none(coordinate_errors),
            "min_matching_iou": min(ious) if ious else None,
            "mean_matching_iou": mean_or_none(ious),
        },
        "source_class_results": source_class_results,
        "failed_images": failed_images,
        "limitations": [
            "This compares Python ONNX Runtime with C++ ONNX Runtime for the same ONNX artifact; the matching best.pt is unavailable, so this is not a new three-way PyTorch/ONNX/C++ run.",
            "Python OpenCV and C++ OpenCV versions are recorded separately; the frozen tolerances were not changed after execution.",
            "Session provider evidence is not per-node execution-placement profiling.",
            "This is implementation consistency on a fixed 30-image set, not model accuracy evaluation.",
        ],
    }


def run_comparison(
    manifest_path: Path,
    cpp_cli: Path,
    output_dir: Path,
    cpp_opencv_version: str,
) -> Mapping[str, Any]:
    require_dependencies()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", cpp_opencv_version):
        fail(
            "cpp_opencv_version",
            "the numeric CMake OpenCV_VERSION used to build the CLI",
            repr(cpp_opencv_version),
            "pass the configure-time OpenCV version without inventing a value",
        )
    if not cpp_cli.resolve().is_file():
        fail(
            "cpp_cli.path",
            "an existing yolo_defect_cpp executable",
            str(cpp_cli),
            "build the clean Release target and pass its absolute path",
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = load_json(manifest_path.resolve(strict=True))
    declared_config = manifest_json.get("config_path")
    if not isinstance(declared_config, str) or not declared_config:
        fail("manifest.config_path", "a non-empty path", repr(declared_config), "fix the manifest")
    config_path = (manifest_path.resolve().parent / declared_config).resolve(strict=True)
    contract = load_contract(config_path)
    manifest = load_manifest(manifest_path, contract)
    session = create_python_session(contract)
    inspection = inspect_cpp_cli(cpp_cli.resolve(), contract["config_path"])

    image_results: List[Mapping[str, Any]] = []
    temporary_root = output_dir / f"cpp_intermediate_{os.getpid()}"
    if temporary_root.exists():
        fail(
            "consistency.cpp_intermediate",
            "an unused process-specific path",
            str(temporary_root),
            "remove the stale directory after confirming no comparison is running",
        )
    temporary_root.mkdir()
    generated_cpp_paths: List[Path] = []
    try:
        for sample in manifest["resolved_samples"]:
            cpp_output_path = temporary_root / f"{sample['sample_id']}.json"
            generated_cpp_paths.append(cpp_output_path)
            try:
                result = evaluate_image(
                    sample,
                    contract,
                    session,
                    cpp_cli.resolve(),
                    cpp_output_path,
                )
            except Exception as error:  # preserve per-image diagnostics/evidence
                result = {
                    "sample_id": sample["sample_id"],
                    "source_class_id": sample["source_class_id"],
                    "source_class_name": sample["source_class_name"],
                    "image_path": sample["image_path"],
                    "image_sha256": sample["image_sha256"],
                    "python_detection_count": 0,
                    "cpp_detection_count": 0,
                    "python_class_id_counts": {},
                    "cpp_class_id_counts": {},
                    "python_detections": [],
                    "cpp_detections": [],
                    "matches": [],
                    "unmatched_python_indices": [],
                    "unmatched_cpp_indices": [],
                    "metrics": {
                        "max_confidence_abs_error": None,
                        "max_bbox_coordinate_abs_error_pixels": None,
                        "min_matching_iou": None,
                    },
                    "passed": False,
                    "failures": [str(error)],
                }
            image_results.append(result)
            marker = "PASS" if result["passed"] else "FAIL"
            print(
                f"[{marker}] {result['sample_id']}: "
                f"python={result['python_detection_count']}, "
                f"cpp={result['cpp_detection_count']}, "
                f"max_conf_error={result['metrics']['max_confidence_abs_error']}, "
                f"max_bbox_error={result['metrics']['max_bbox_coordinate_abs_error_pixels']}, "
                f"min_iou={result['metrics']['min_matching_iou']}"
            )
            for failure in result["failures"]:
                print(f"  diagnostic: {failure}")
    finally:
        for generated_path in generated_cpp_paths:
            if generated_path.exists() and generated_path.is_file():
                generated_path.unlink()
        try:
            temporary_root.rmdir()
        except OSError as cleanup_error:
            print(
                "Could not remove the process-specific intermediate directory "
                f"'{temporary_root}': {cleanup_error}",
                file=sys.stderr,
            )

    per_image_document = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "python_ort_vs_cpp_ort_per_image_consistency",
        "manifest_id": manifest["manifest_id"],
        "requirements": dict(FROZEN_REQUIREMENTS),
        "images": image_results,
    }
    summary = build_summary(
        manifest, contract, inspection, image_results, cpp_opencv_version
    )
    write_json(output_dir / "per_image.json", per_image_document)
    write_json(output_dir / "summary.json", summary)
    print(
        "Consistency summary: "
        f"passed={summary['passed']}, "
        f"images={summary['result']['images_passed']}/{summary['result']['images_total']}, "
        f"matched={summary['result']['matched_detections_total']}, "
        f"max_conf_error={summary['result']['max_confidence_abs_error']}, "
        f"max_bbox_error={summary['result']['max_bbox_coordinate_abs_error_pixels']}, "
        f"min_iou={summary['result']['min_matching_iou']}"
    )
    return summary


def write_setup_failure(output_dir: Path, error: Exception) -> None:
    try:
        output_dir = output_dir.resolve()
        write_json(
            output_dir / "per_image.json",
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_type": "python_ort_vs_cpp_ort_per_image_consistency",
                "requirements": dict(FROZEN_REQUIREMENTS),
                "images": [],
                "setup_error": str(error),
            },
        )
        write_json(
            output_dir / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "evidence_type": "python_ort_vs_cpp_ort_detection_consistency",
                "passed": False,
                "requirements": dict(FROZEN_REQUIREMENTS),
                "result": {
                    "images_total": 0,
                    "images_passed": 0,
                    "images_failed": 0,
                },
                "setup_error": str(error),
            },
        )
    except Exception as write_error:
        print(
            f"Could not write setup-failure evidence: {write_error}",
            file=sys.stderr,
        )


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen Python CPU ONNX Runtime detections with the "
            "C++ CPU ONNX Runtime CLI without order-dependent matching."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cpp-cli", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cpp-opencv-version", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        summary = run_comparison(
            arguments.manifest,
            arguments.cpp_cli,
            arguments.output_dir,
            arguments.cpp_opencv_version,
        )
    except Exception as error:
        write_setup_failure(arguments.output_dir, error)
        print(str(error), file=sys.stderr)
        return 1
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

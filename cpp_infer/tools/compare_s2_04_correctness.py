#!/usr/bin/env python3
"""Gate S2-04 CPU/FP16 TensorRT correctness and repeat stability.

The inputs are three directories containing the product CLI's existing
``*.detections.json`` files for the frozen 30-image manifest.  The tool checks
the frozen artifact declaration and model bytes, validates provider placement
and product semantics in every file, performs three pairwise comparisons, and
writes both summary and per-image evidence.  A correctness mismatch is still
written to both outputs and returns a non-zero process status.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "s2_04_tensorrt_fp16_correctness"
CPU_PROVIDER = "CPUExecutionProvider"
TRT_PROVIDER = "TensorrtExecutionProvider"
NATIVE_TRT_PROVIDER = "TensorRTNative"
CUDA_PROVIDER = "CUDAExecutionProvider"
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
SELECTION_INDICES = [241, 255, 270, 285, 300]
HOLDOUT_SELECTION_INDICES = [242, 256, 271, 286, 299]
NATIVE_HOLDOUT_SELECTION_INDICES_V3 = [243, 257, 272, 287, 298]
NATIVE_HOLDOUT_SELECTION_INDICES_V4 = [244, 258, 273, 288, 297]
NATIVE_ENGINE_SHA256_V3 = "CE2D25F9ABD4370A7E5325777CEB1982E42E82C7A3238C24A48E7FDD264467D2"
NATIVE_PRECISION_POLICY_V3 = "fp16_class_sigmoid_fp32_else_no_tf32"
NATIVE_ENGINE_SHA256_V4 = "E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746"
NATIVE_PRECISION_POLICY_V4 = "fp16_dfl_softmax_fp32_else_no_tf32"
FROZEN_GATE: Mapping[str, Any] = {
    "comparisons": [
        "cpu_vs_tensorrt_run_a",
        "cpu_vs_tensorrt_run_b",
        "tensorrt_run_a_vs_tensorrt_run_b",
    ],
    "detection_count": "exact",
    "class_id": "exact",
    "class_name": "exact",
    "confidence_abs_error_max": 0.005,
    "bbox_coordinate_abs_error_max_pixels": 0.5,
    "matching_iou_min": 0.995,
    "matching_strategy": "class_id_then_greedy_max_iou_with_canonical_value_tie_break",
    "same_gate_applies_to_repeated_tensorrt_runs": True,
}
HOLDOUT_GATE: Mapping[str, Any] = {
    "comparisons": [
        "cpu_vs_tensorrt_run_a",
        "cpu_vs_tensorrt_run_b",
        "tensorrt_run_a_vs_tensorrt_run_b",
    ],
    "detection_count": "exact",
    "class_id": "exact",
    "class_name": "exact",
    "confidence_abs_error_max": 0.005,
    "bbox_coordinate_abs_error_max_pixels": 1.0,
    "matching_iou_min": 0.9,
    "matching_strategy": "class_id_then_greedy_max_iou_with_canonical_value_tie_break",
    "same_gate_applies_to_repeated_tensorrt_runs": True,
}
FROZEN_ARTIFACT: Mapping[str, Any] = {
    "model_id": "yolov8n_neu_det_final_train_2",
    "model_family": "yolov8",
    "model_sha256": "7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68",
    "opset": 17,
    "input_name": "images",
    "input_shape": [1, 3, 800, 800],
    "input_dtype": "float32",
    "input_layout": "nchw",
    "output_name": "output0",
    "output_shape": [1, 10, 13125],
    "output_dtype": "float32",
    "output_layout": "bcn",
    "class_names": CLASS_NAMES,
    "preprocess_type": "letterbox_rgb_0_1_nchw",
    "postprocess_type": "yolov8_raw",
    "score_threshold": 0.25,
    "nms_threshold": 0.45,
    "nms_mode": "class_agnostic",
}
PROTOCOL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "s2_04_tensorrt_fp16_correctness_v1": {
        "evidence_type": EVIDENCE_TYPE,
        "tensorrt_provider": TRT_PROVIDER,
        "tensorrt_config_provider": "tensorrt",
        "execution_contract": {
            "cpu_precision": "fp32", "tensorrt_precision": "fp16",
            "tensorrt_run_count": 2, "same_artifact_for_all_runs": True,
            "same_preprocess_and_postprocess_for_all_runs": True,
            "provider_expectations_are_explicit_cli_arguments": True,
        },
        "provider_contract": {
            "cpu_actual_provider": CPU_PROVIDER,
            "tensorrt_actual_provider": TRT_PROVIDER,
            "cuda_fallback_provider": CUDA_PROVIDER,
            "provider_names_are_not_user_redefinable": True,
        },
        "gate": FROZEN_GATE,
        "sample_contract": {
            "manifest_id": "neu_det_val_6x5_v1",
            "sample_count": 30,
            "samples_per_class": 5,
            "selection_indices": SELECTION_INDICES,
        },
    },
    "s2_04_tensorrt_fp16_correctness_v2": {
        "evidence_type": EVIDENCE_TYPE,
        "tensorrt_provider": TRT_PROVIDER,
        "tensorrt_config_provider": "tensorrt",
        "execution_contract": {
            "cpu_precision": "fp32", "tensorrt_precision": "fp16",
            "tensorrt_run_count": 2, "same_artifact_for_all_runs": True,
            "same_preprocess_and_postprocess_for_all_runs": True,
            "provider_expectations_are_explicit_cli_arguments": True,
        },
        "provider_contract": {
            "cpu_actual_provider": CPU_PROVIDER,
            "tensorrt_actual_provider": TRT_PROVIDER,
            "cuda_fallback_provider": CUDA_PROVIDER,
            "provider_names_are_not_user_redefinable": True,
        },
        "gate": HOLDOUT_GATE,
        "sample_contract": {
            "manifest_id": "neu_det_val_s2_04_holdout_6x5_v2",
            "sample_count": 30,
            "samples_per_class": 5,
            "selection_indices": HOLDOUT_SELECTION_INDICES,
        },
    },
    "s2_04_tensorrt_native_fp16_correctness_v3": {
        "evidence_type": "s2_04_tensorrt_native_fp16_correctness",
        "tensorrt_provider": NATIVE_TRT_PROVIDER,
        "tensorrt_config_provider": "tensorrt_native",
        "gate": HOLDOUT_GATE,
        "sample_contract": {
            "manifest_id": "neu_det_val_s2_04_native_holdout_6x5_v3",
            "sample_count": 30,
            "samples_per_class": 5,
            "selection_indices": NATIVE_HOLDOUT_SELECTION_INDICES_V3,
        },
        "execution_contract": {
            "cpu_precision": "fp32",
            "tensorrt_precision": "mixed_fp16_fp32",
            "tensorrt_run_count": 2,
            "same_onnx_artifact_for_all_runs": True,
            "same_native_engine_for_both_tensorrt_runs": True,
            "same_preprocess_and_postprocess_for_all_runs": True,
            "provider_expectations_are_explicit_cli_arguments": True,
        },
        "provider_contract": {
            "cpu_actual_provider": CPU_PROVIDER,
            "tensorrt_actual_provider": NATIVE_TRT_PROVIDER,
            "fallback_provider": None,
            "provider_names_are_not_user_redefinable": True,
        },
        "engine_contract": {
            "engine_sha256": NATIVE_ENGINE_SHA256_V3,
            "tensorrt_runtime_version": "10.4.0",
            "tensorrt_header_package_version": "10.4.0.26",
            "cuda_runtime_version": "12.6",
            "compute_capability": "8.9",
            "precision_policy": NATIVE_PRECISION_POLICY_V3,
            "builder_contract": "trtexec --fp16 --noTF32 --precisionConstraints=obey --layerPrecisions=*:fp32,/model.22/Sigmoid:fp16",
            "portable_across_gpu_or_tensorrt_versions": False,
        },
    },
    "s2_04_tensorrt_native_fp16_correctness_v4": {
        "evidence_type": "s2_04_tensorrt_native_fp16_correctness",
        "tensorrt_provider": NATIVE_TRT_PROVIDER,
        "tensorrt_config_provider": "tensorrt_native",
        "gate": HOLDOUT_GATE,
        "sample_contract": {
            "manifest_id": "neu_det_val_s2_04_native_holdout_6x5_v4",
            "sample_count": 30,
            "samples_per_class": 5,
            "selection_indices": NATIVE_HOLDOUT_SELECTION_INDICES_V4,
        },
        "execution_contract": {
            "cpu_precision": "fp32",
            "tensorrt_precision": "mixed_fp16_fp32",
            "tensorrt_run_count": 2,
            "same_onnx_artifact_for_all_runs": True,
            "same_native_engine_for_both_tensorrt_runs": True,
            "same_preprocess_and_postprocess_for_all_runs": True,
            "provider_expectations_are_explicit_cli_arguments": True,
        },
        "provider_contract": {
            "cpu_actual_provider": CPU_PROVIDER,
            "tensorrt_actual_provider": NATIVE_TRT_PROVIDER,
            "fallback_provider": None,
            "provider_names_are_not_user_redefinable": True,
        },
        "engine_contract": {
            "engine_sha256": NATIVE_ENGINE_SHA256_V4,
            "tensorrt_runtime_version": "10.4.0",
            "tensorrt_header_package_version": "10.4.0.26",
            "cuda_runtime_version": "12.6",
            "compute_capability": "8.9",
            "precision_policy": NATIVE_PRECISION_POLICY_V4,
            "builder_contract": "trtexec --fp16 --noTF32 --precisionConstraints=obey --layerPrecisions=*:fp32,/model.22/dfl/Softmax:fp16",
            "portable_across_gpu_or_tensorrt_versions": False,
        },
    },
}


class CorrectnessError(RuntimeError):
    """An actionable setup, schema, or evidence-writing failure."""


def fail(object_name: str, expected: str, actual: Any, action: str) -> NoReturn:
    raise CorrectnessError(
        "S2-04 correctness failed: "
        f"object={object_name}; expected={expected}; actual={actual!r}; action={action}"
    )


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> MutableMapping[str, Any]:
    result: MutableMapping[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("json", "unique object keys", key, "remove the duplicate key")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    fail("json.number", "a finite JSON number", value, "replace NaN or Infinity")


def load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(str(path), "one readable UTF-8 JSON document", str(error), "fix the path, encoding, or JSON syntax")
    if not isinstance(value, dict):
        fail(str(path), "a JSON object root", type(value).__name__, "replace the root value")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail(str(path), "a readable regular file", str(error), "restore the declared model artifact")
    return digest.hexdigest().upper()


def resolve_source(protocol_path: Path, raw_value: Any, object_name: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value:
        fail(object_name, "a non-empty protocol-relative path", raw_value, "fix the protocol source")
    path = (protocol_path.resolve().parent / raw_value).resolve()
    if not path.is_file():
        fail(object_name, "an existing regular file", str(path), "restore the declared source")
    return path


def validate_protocol(
    protocol: Mapping[str, Any], protocol_path: Path
) -> Tuple[Mapping[str, Path], Mapping[str, Any]]:
    if protocol.get("schema_version") != SCHEMA_VERSION:
        fail("protocol.schema_version", "1", protocol.get("schema_version"), "use the supported protocol")
    protocol_id = protocol.get("protocol_id")
    specification = PROTOCOL_SPECS.get(str(protocol_id))
    if specification is None:
        fail(
            "protocol.protocol_id",
            f"one of {sorted(PROTOCOL_SPECS)}",
            protocol_id,
            "use one tracked predeclared S2-04 protocol",
        )
    if protocol.get("frozen_before_formal_run") is not True:
        fail("protocol.frozen_before_formal_run", "true", protocol.get("frozen_before_formal_run"), "do not use a post-hoc protocol")
    expected_gate = specification["gate"]
    if protocol.get("correctness_gate") != expected_gate:
        fail("protocol.correctness_gate", str(dict(expected_gate)), protocol.get("correctness_gate"), "restore the selected predeclared FP16 gate; do not tune after a run")
    if protocol.get("artifact_semantics") != FROZEN_ARTIFACT:
        fail("protocol.artifact_semantics", "the frozen current YOLO artifact semantics", protocol.get("artifact_semantics"), "restore the v1 artifact contract")
    sample_contract = protocol.get("sample_contract")
    expected_sample_contract = specification["sample_contract"]
    if sample_contract != expected_sample_contract:
        fail("protocol.sample_contract", str(expected_sample_contract), sample_contract, "restore the frozen 30-image selection")
    execution = protocol.get("execution_contract")
    expected_execution = specification["execution_contract"]
    if execution != expected_execution:
        fail("protocol.execution_contract", str(expected_execution), execution, "restore the FP32/FP16 repeated-run contract")
    expected_providers = specification["provider_contract"]
    if protocol.get("provider_contract") != expected_providers:
        fail("protocol.provider_contract", str(expected_providers), protocol.get("provider_contract"), "restore the exact frozen provider identities")
    expected_engine_contract = specification.get("engine_contract")
    if protocol.get("engine_contract") != expected_engine_contract:
        fail(
            "protocol.engine_contract",
            str(expected_engine_contract),
            protocol.get("engine_contract"),
            "restore the frozen native engine SHA/policy or remove the field for ORT protocols",
        )
    sources = protocol.get("sources")
    if not isinstance(sources, dict):
        fail("protocol.sources", "an object", type(sources).__name__, "restore protocol sources")
    required = {
        "consistency_manifest_path",
        "cpu_runtime_config_path",
        "tensorrt_runtime_config_path",
        "artifact_spec_path",
    }
    if set(sources) != required:
        fail("protocol.sources", f"exact fields {sorted(required)}", sorted(sources), "restore every frozen source")
    sources_result = {
        key: resolve_source(protocol_path, value, f"protocol.sources.{key}")
        for key, value in sources.items()
    }
    return sources_result, specification


def parse_key_value_declaration(path: Path) -> Mapping[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        fail(str(path), "readable UTF-8 key=value text", str(error), "restore the artifact declaration")
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            fail(f"{path}:{line_number}", "key = value", line, "fix the declaration")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key or not value or key in values:
            fail(f"{path}:{line_number}", "one unique non-empty key/value", line, "fix the declaration")
        values[key] = value
    return values


def validate_runtime_configs(
    sources: Mapping[str, Path], specification: Mapping[str, Any]
) -> Mapping[str, Any]:
    artifact_path = sources["artifact_spec_path"].resolve()
    cpu_path = sources["cpu_runtime_config_path"]
    trt_path = sources["tensorrt_runtime_config_path"]
    cpu = parse_key_value_declaration(cpu_path)
    trt = parse_key_value_declaration(trt_path)
    expected_cpu = {
        "schema_version": "1",
        "provider": "cpu",
        "score_threshold": "0.25",
        "nms_threshold": "0.45",
    }
    expected_trt = {
        "schema_version": "2",
        "provider": specification["tensorrt_config_provider"],
        "device_id": "0",
        "precision": "fp16",
        "score_threshold": "0.25",
        "nms_threshold": "0.45",
    }
    for label, values, expected in (
        ("cpu", cpu, expected_cpu),
        ("tensorrt", trt, expected_trt),
    ):
        for key, expected_value in expected.items():
            if values.get(key) != expected_value:
                fail(f"{label}_runtime_config.{key}", expected_value, values.get(key), "restore the frozen correctness configuration")
        raw_artifact = values.get("artifact_spec_path")
        if not raw_artifact:
            fail(f"{label}_runtime_config.artifact_spec_path", "a non-empty path", raw_artifact, "restore the shared artifact declaration")
        resolved_artifact = (sources[f"{label if label == 'cpu' else 'tensorrt'}_runtime_config_path"].resolve().parent / raw_artifact).resolve()
        if resolved_artifact != artifact_path:
            fail(f"{label}_runtime_config.artifact_spec_path", str(artifact_path), str(resolved_artifact), "use one identical artifact for CPU and TensorRT")
    result: Dict[str, Any] = {
        "cpu": {
            "path": str(cpu_path.resolve()),
            "provider": "cpu",
            "precision": "fp32",
            "artifact_spec_path": str(artifact_path),
        },
        "tensorrt": {
            "path": str(trt_path.resolve()),
            "provider": specification["tensorrt_config_provider"],
            "precision": "mixed_fp16_fp32"
            if specification["tensorrt_config_provider"] == "tensorrt_native"
            else "fp16",
            "artifact_spec_path": str(artifact_path),
        },
    }
    engine_contract = specification.get("engine_contract")
    if engine_contract is not None:
        declared_engine_sha = trt.get("tensorrt_engine_sha256", "").upper()
        if declared_engine_sha != engine_contract["engine_sha256"]:
            fail("tensorrt_runtime_config.tensorrt_engine_sha256", engine_contract["engine_sha256"], declared_engine_sha, "restore the frozen native config")
        raw_engine_path = trt.get("tensorrt_engine_path")
        if not raw_engine_path:
            fail("tensorrt_runtime_config.tensorrt_engine_path", "a non-empty path", raw_engine_path, "restore the frozen native config")
        engine_path = (trt_path.resolve().parent / raw_engine_path).resolve()
        if not engine_path.is_file():
            fail("tensorrt_runtime_config.tensorrt_engine_path", "an existing engine", str(engine_path), "build and place the frozen engine")
        actual_engine_sha = sha256_file(engine_path)
        if actual_engine_sha != engine_contract["engine_sha256"]:
            fail("native_engine.sha256", engine_contract["engine_sha256"], actual_engine_sha, "restore the exact constrained engine bytes")
        result["tensorrt"]["engine_path"] = str(engine_path)
        result["tensorrt"]["declared_and_actual_engine_sha256"] = actual_engine_sha
        result["tensorrt"]["precision_policy"] = engine_contract["precision_policy"]
    return result


def parse_shape(value: str, object_name: str) -> List[int]:
    try:
        shape = [int(part.strip()) for part in value.split(",")]
    except ValueError:
        fail(object_name, "a comma-separated integer shape", value, "fix the artifact declaration")
    if not shape or any(dimension <= 0 for dimension in shape):
        fail(object_name, "positive static dimensions", shape, "fix the artifact declaration")
    return shape


def validate_artifact_spec(path: Path) -> Mapping[str, Any]:
    raw = parse_key_value_declaration(path)
    try:
        semantics = {
            "model_id": raw["model_id"],
            "model_family": raw["model_family"],
            "model_sha256": raw["model_sha256"].upper(),
            "opset": int(raw["opset"]),
            "input_name": raw["input_name"],
            "input_shape": parse_shape(raw["input_shape"], "artifact.input_shape"),
            "input_dtype": raw["input_dtype"],
            "input_layout": raw["input_layout"],
            "output_name": raw["output_name"],
            "output_shape": parse_shape(raw["output_shape"], "artifact.output_shape"),
            "output_dtype": raw["output_dtype"],
            "output_layout": raw["output_layout"],
            "class_names": [part.strip() for part in raw["class_names"].split(",")],
            "preprocess_type": raw["preprocess_type"],
            "postprocess_type": raw["postprocess_type"],
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
            "nms_mode": raw["nms_mode"],
        }
        model_path = (path.resolve().parent / raw["model_path"]).resolve()
    except (KeyError, ValueError) as error:
        fail("artifact_spec", "all frozen artifact fields", str(error), "restore the v1 artifact declaration")
    if semantics != FROZEN_ARTIFACT:
        fail("artifact_spec.semantics", "the protocol's frozen artifact semantics", semantics, "use the current YOLO artifact without semantic drift")
    if not model_path.is_file():
        fail("artifact.model_path", "an existing regular ONNX file", str(model_path), "restore models/best.onnx")
    actual_sha = sha256_file(model_path)
    if actual_sha != semantics["model_sha256"]:
        fail("artifact.model_sha256", semantics["model_sha256"], actual_sha, "use the declared ONNX or version a new protocol")
    return {"declaration_path": path, "model_path": model_path, **semantics}


def portable_basename(value: str) -> str:
    return re.split(r"[\\/]", value)[-1]


def validate_manifest(
    document: Mapping[str, Any],
    manifest_path: Path,
    sample_contract: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    expected_manifest_id = sample_contract["manifest_id"]
    selection_indices = sample_contract["selection_indices"]
    if document.get("schema_version") != 1 or document.get("manifest_id") != expected_manifest_id:
        fail("manifest.identity", f"schema=1 and manifest_id={expected_manifest_id}", {"schema": document.get("schema_version"), "id": document.get("manifest_id")}, "use the protocol-selected frozen manifest")
    classes = document.get("classes")
    expected_classes = [
        {"class_id": index, "class_name": name}
        for index, name in enumerate(CLASS_NAMES)
    ]
    if classes != expected_classes:
        fail("manifest.classes", str(expected_classes), classes, "restore artifact class order")
    samples = document.get("samples")
    expected_sample_count = sample_contract["sample_count"]
    if not isinstance(samples, list) or len(samples) != expected_sample_count:
        fail("manifest.samples", f"exactly {expected_sample_count} samples", type(samples).__name__ if not isinstance(samples, list) else len(samples), "restore the frozen manifest")
    expected_ids = [f"{name}_{index}" for name in CLASS_NAMES for index in selection_indices]
    actual_ids: List[str] = []
    verified_samples: List[Mapping[str, Any]] = []
    for position, sample in enumerate(samples):
        if not isinstance(sample, dict):
            fail(f"manifest.samples[{position}]", "an object", type(sample).__name__, "fix the manifest")
        sample_id = sample.get("sample_id")
        image_path = sample.get("image_path")
        if not isinstance(sample_id, str) or not isinstance(image_path, str):
            fail(f"manifest.samples[{position}]", "string sample_id and image_path", sample, "fix the manifest")
        if portable_basename(image_path) != f"{sample_id}.jpg":
            fail(f"manifest.samples[{position}].image_path", f"a path ending in {sample_id}.jpg", image_path, "restore the frozen sample mapping")
        expected_class_id = position // len(selection_indices)
        if sample.get("source_class_id") != expected_class_id or sample.get("source_class_name") != CLASS_NAMES[expected_class_id]:
            fail(f"manifest.samples[{position}].source_class", f"id={expected_class_id}, name={CLASS_NAMES[expected_class_id]}", {"id": sample.get("source_class_id"), "name": sample.get("source_class_name")}, "restore the frozen class-major selection")
        declared_sha = sample.get("image_sha256")
        if not isinstance(declared_sha, str) or re.fullmatch(r"[0-9A-Fa-f]{64}", declared_sha) is None:
            fail(f"manifest.samples[{position}].image_sha256", "64 hexadecimal characters", declared_sha, "restore the frozen image identity")
        resolved_image = (manifest_path.resolve().parent / image_path).resolve()
        if not resolved_image.is_file():
            fail(f"manifest.samples[{position}].image_path", "an existing regular image", str(resolved_image), "restore the frozen validation dataset")
        actual_sha = sha256_file(resolved_image)
        if actual_sha != declared_sha.upper():
            fail(f"manifest.samples[{position}].image_sha256", declared_sha.upper(), actual_sha, "use the frozen image bytes or version a new protocol")
        actual_ids.append(sample_id)
        verified_samples.append(
            {
                **sample,
                "image_sha256": declared_sha.upper(),
                "verified_image_path": str(resolved_image),
            }
        )
    if actual_ids != expected_ids:
        fail("manifest.sample_ids", str(expected_ids), actual_ids, "restore the frozen class-major order")
    return verified_samples


def finite_number(value: Any, object_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(object_name, "a finite number", value, "fix the detection JSON")
    result = float(value)
    if not math.isfinite(result):
        fail(object_name, "a finite number", value, "fix the detection JSON")
    return result


def validate_detection_document(
    document: Mapping[str, Any],
    *,
    path: Path,
    sample_by_filename: Mapping[str, Mapping[str, Any]],
    expected_provider: str,
    expected_engine_contract: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if document.get("schema_version") != 1:
        fail(f"{path}.schema_version", "1", document.get("schema_version"), "write product detection schema v1")
    model = document.get("model")
    image = document.get("image")
    runtime = document.get("runtime")
    detections = document.get("detections")
    if not isinstance(model, dict) or not isinstance(image, dict) or not isinstance(runtime, dict) or not isinstance(detections, list):
        fail(str(path), "model/image/runtime objects and detections array", "missing or wrong types", "use an existing product detection JSON")
    if model.get("model_id") != FROZEN_ARTIFACT["model_id"] or str(model.get("declared_sha256", "")).upper() != FROZEN_ARTIFACT["model_sha256"]:
        fail(f"{path}.model", "the frozen model id and SHA-256", model, "rerun with the protocol artifact")
    image_path = image.get("path")
    if not isinstance(image_path, str) or portable_basename(image_path) not in sample_by_filename:
        fail(f"{path}.image.path", "one frozen manifest image", image_path, "run exactly the 30-image manifest")
    sample = sample_by_filename[portable_basename(image_path)]
    source_image = Path(image_path).expanduser()
    if not source_image.is_absolute():
        source_image = path.resolve().parent / source_image
    source_image = source_image.resolve()
    if not source_image.is_file():
        fail(f"{path}.image.path", "a readable source image from this formal run", str(source_image), "run the comparator on the same Linux host/filesystem as inference")
    source_image_sha = sha256_file(source_image)
    if source_image_sha != sample["image_sha256"]:
        fail(f"{path}.image.path SHA-256", sample["image_sha256"], source_image_sha, "run inference on the exact frozen manifest image bytes")
    expected_input_size = {"width": 800, "height": 800}
    if image.get("input_size") != expected_input_size:
        fail(f"{path}.image.input_size", str(expected_input_size), image.get("input_size"), "preserve artifact preprocessing dimensions")
    original_size = image.get("original_size")
    if not isinstance(original_size, dict):
        fail(f"{path}.image.original_size", "an object", original_size, "write product image metadata")
    width = finite_number(original_size.get("width"), f"{path}.image.original_size.width")
    height = finite_number(original_size.get("height"), f"{path}.image.original_size.height")
    if width <= 0.0 or height <= 0.0 or original_size.get("channels") != 3:
        fail(f"{path}.image.original_size", "positive width/height and 3 channels", original_size, "use the decoded source image metadata")
    if runtime.get("actual_provider") != expected_provider:
        fail(f"{path}.runtime.actual_provider", expected_provider, runtime.get("actual_provider"), "fix provider registration or the explicit CLI expectation")
    if not isinstance(runtime.get("provider_evidence"), str) or not runtime["provider_evidence"]:
        fail(f"{path}.runtime.provider_evidence", "a non-empty execution claim", runtime.get("provider_evidence"), "record actionable provider evidence")
    if expected_engine_contract is not None and expected_provider == NATIVE_TRT_PROVIDER:
        engine_sha = expected_engine_contract["engine_sha256"]
        expected_evidence = (
            "native_tensorrt_enqueue_v3;"
            f"precision_policy={expected_engine_contract['precision_policy']};"
            f"declared_engine_sha256={engine_sha};"
            f"actual_engine_sha256={engine_sha};"
            "tensorrt_runtime=10.4.0;compiled_headers=10.4.0.26;"
            "cuda_runtime=12.6;"
            "compute_capability=8.9;fallback=none"
        )
        if runtime["provider_evidence"] != expected_evidence:
            fail(f"{path}.runtime.provider_evidence", expected_evidence, runtime["provider_evidence"], "execute the SHA-bound native engine without fallback")
    expected_runtime = {
        "score_threshold": FROZEN_ARTIFACT["score_threshold"],
        "nms_threshold": FROZEN_ARTIFACT["nms_threshold"],
        "nms_mode": FROZEN_ARTIFACT["nms_mode"],
    }
    for key, expected in expected_runtime.items():
        actual = runtime.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12):
                fail(f"{path}.runtime.{key}", str(expected), actual, "preserve frozen postprocess semantics")
        elif actual != expected:
            fail(f"{path}.runtime.{key}", str(expected), actual, "preserve frozen postprocess semantics")
    normalized_detections: List[Mapping[str, Any]] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            fail(f"{path}.detections[{index}]", "an object", type(detection).__name__, "fix detection serialization")
        class_id = detection.get("class_id")
        class_name = detection.get("class_name")
        if isinstance(class_id, bool) or not isinstance(class_id, int) or not 0 <= class_id < len(CLASS_NAMES) or class_name != CLASS_NAMES[class_id]:
            fail(f"{path}.detections[{index}].class", "an exact artifact class id/name pair", {"id": class_id, "name": class_name}, "preserve class semantics")
        confidence = finite_number(detection.get("confidence"), f"{path}.detections[{index}].confidence")
        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            fail(f"{path}.detections[{index}].bbox_xyxy", "four coordinates", bbox, "fix detection serialization")
        coordinates = [finite_number(value, f"{path}.detections[{index}].bbox_xyxy") for value in bbox]
        if confidence < 0.0 or confidence > 1.0 or coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]:
            fail(f"{path}.detections[{index}]", "confidence in [0,1] and ordered xyxy", detection, "fix decode/postprocess output")
        tolerance = 1.0e-5
        if coordinates[0] < -tolerance or coordinates[1] < -tolerance or coordinates[2] > width + tolerance or coordinates[3] > height + tolerance:
            fail(f"{path}.detections[{index}].bbox_xyxy", f"coordinates clipped to [0,{width}]x[0,{height}]", coordinates, "preserve coordinate restoration and clipping")
        normalized_detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox_xyxy": coordinates,
            }
        )
    return {
        "sample_id": sample["sample_id"],
        "path": str(path.resolve()),
        "image": {"path": image_path, "original_size": dict(original_size), "input_size": dict(image["input_size"])},
        "runtime": {"actual_provider": runtime["actual_provider"], "provider_evidence": runtime["provider_evidence"], **expected_runtime},
        "model": {"model_id": model["model_id"], "declared_sha256": str(model["declared_sha256"]).upper()},
        "detections": normalized_detections,
    }


def collect_run(
    directory: Path,
    *,
    samples: Sequence[Mapping[str, Any]],
    expected_provider: str,
    run_name: str,
    expected_engine_contract: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Mapping[str, Any]]:
    if not directory.is_dir():
        fail(run_name, "an existing directory", str(directory), "provide the product output directory")
    sample_by_filename = {portable_basename(str(sample["image_path"])): sample for sample in samples}
    files = sorted(directory.rglob("*.detections.json"), key=lambda value: str(value).encode("utf-8"))
    if len(files) != len(samples):
        fail(run_name, f"exactly {len(samples)} *.detections.json files", len(files), "run the frozen manifest into a clean output directory")
    results: Dict[str, Mapping[str, Any]] = {}
    for path in files:
        value = validate_detection_document(
            load_json_object(path),
            path=path,
            sample_by_filename=sample_by_filename,
            expected_provider=expected_provider,
            expected_engine_contract=expected_engine_contract,
        )
        sample_id = str(value["sample_id"])
        if sample_id in results:
            fail(run_name, "one detection JSON per sample", sample_id, "remove duplicate outputs and rerun")
        results[sample_id] = value
    expected_ids = [str(sample["sample_id"]) for sample in samples]
    if set(results) != set(expected_ids):
        fail(run_name, f"sample ids {expected_ids}", sorted(results), "run exactly the frozen manifest")
    return results


def intersection_over_union(left: Sequence[float], right: Sequence[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else (1.0 if list(left) == list(right) else 0.0)


def detection_key(detection: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        detection["class_id"],
        *detection["bbox_xyxy"],
        detection["confidence"],
        detection["class_name"],
    )


def greedy_matches(
    reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    pairs: List[Tuple[float, Tuple[Any, ...], Tuple[Any, ...], int, int]] = []
    for reference_index, reference_detection in enumerate(reference):
        for candidate_index, candidate_detection in enumerate(candidate):
            if reference_detection["class_id"] == candidate_detection["class_id"]:
                iou = intersection_over_union(reference_detection["bbox_xyxy"], candidate_detection["bbox_xyxy"])
                pairs.append((-iou, detection_key(reference_detection), detection_key(candidate_detection), reference_index, candidate_index))
    pairs.sort()
    used_reference = set()
    used_candidate = set()
    matches: List[Tuple[int, int, float]] = []
    for negative_iou, _reference_key, _candidate_key, reference_index, candidate_index in pairs:
        if reference_index in used_reference or candidate_index in used_candidate:
            continue
        used_reference.add(reference_index)
        used_candidate.add(candidate_index)
        matches.append((reference_index, candidate_index, -negative_iou))
    matches.sort()
    return (
        matches,
        sorted(set(range(len(reference))) - used_reference),
        sorted(set(range(len(candidate))) - used_candidate),
    )


def compare_image(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    comparison_name: str,
    gate: Mapping[str, Any] = FROZEN_GATE,
) -> Mapping[str, Any]:
    failures: List[str] = []
    for field in ("model",):
        if reference[field] != candidate[field]:
            failures.append(f"{field} differs")
    if reference["image"]["original_size"] != candidate["image"]["original_size"] or reference["image"]["input_size"] != candidate["image"]["input_size"]:
        failures.append("image size semantics differ")
    for key in ("score_threshold", "nms_threshold", "nms_mode"):
        if reference["runtime"][key] != candidate["runtime"][key]:
            failures.append(f"runtime.{key} differs")
    reference_detections = reference["detections"]
    candidate_detections = candidate["detections"]
    if len(reference_detections) != len(candidate_detections):
        failures.append(f"detection_count expected {len(reference_detections)}, actual {len(candidate_detections)}")
    reference_classes = collections.Counter(value["class_id"] for value in reference_detections)
    candidate_classes = collections.Counter(value["class_id"] for value in candidate_detections)
    if reference_classes != candidate_classes:
        failures.append(f"class histogram expected {dict(reference_classes)}, actual {dict(candidate_classes)}")
    matches, unmatched_reference, unmatched_candidate = greedy_matches(reference_detections, candidate_detections)
    if unmatched_reference or unmatched_candidate:
        failures.append(f"unmatched reference={unmatched_reference}, candidate={unmatched_candidate}")
    match_rows: List[Mapping[str, Any]] = []
    for reference_index, candidate_index, iou in matches:
        left = reference_detections[reference_index]
        right = candidate_detections[candidate_index]
        confidence_error = abs(left["confidence"] - right["confidence"])
        coordinate_errors = [abs(a - b) for a, b in zip(left["bbox_xyxy"], right["bbox_xyxy"])]
        row_passed = (
            left["class_id"] == right["class_id"]
            and left["class_name"] == right["class_name"]
            and confidence_error <= gate["confidence_abs_error_max"]
            and max(coordinate_errors) <= gate["bbox_coordinate_abs_error_max_pixels"]
            and iou >= gate["matching_iou_min"]
        )
        if not row_passed:
            failures.append(
                f"match {reference_index}->{candidate_index} exceeds gate: confidence_abs={confidence_error:.9g}, bbox_abs_max={max(coordinate_errors):.9g}, iou={iou:.9g}"
            )
        match_rows.append(
            {
                "reference_index": reference_index,
                "candidate_index": candidate_index,
                "class_id": left["class_id"],
                "class_name": left["class_name"],
                "confidence_abs_error": confidence_error,
                "bbox_coordinate_abs_errors_pixels": coordinate_errors,
                "bbox_coordinate_abs_error_max_pixels": max(coordinate_errors),
                "iou": iou,
                "passed": row_passed,
            }
        )
    return {
        "comparison": comparison_name,
        "reference_provider": reference["runtime"]["actual_provider"],
        "candidate_provider": candidate["runtime"]["actual_provider"],
        "reference_detection_count": len(reference_detections),
        "candidate_detection_count": len(candidate_detections),
        "matches": match_rows,
        "unmatched_reference_indices": unmatched_reference,
        "unmatched_candidate_indices": unmatched_candidate,
        "failures": failures,
        "passed": not failures,
    }


def aggregate_comparison(name: str, items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    matches = [match for item in items for match in item["matches"]]
    return {
        "comparison": name,
        "sample_count": len(items),
        "failed_sample_count": sum(not item["passed"] for item in items),
        "failed_sample_ids": [item["sample_id"] for item in items if not item["passed"]],
        "matched_detection_count": len(matches),
        "confidence_abs_error_max": max((row["confidence_abs_error"] for row in matches), default=0.0),
        "bbox_coordinate_abs_error_max_pixels": max((row["bbox_coordinate_abs_error_max_pixels"] for row in matches), default=0.0),
        "matching_iou_min": min((row["iou"] for row in matches), default=None),
        "passed": all(item["passed"] for item in items),
    }


def compare_detection_runs(
    samples: Sequence[Mapping[str, Any]],
    cpu: Mapping[str, Mapping[str, Any]],
    trt_a: Mapping[str, Mapping[str, Any]],
    trt_b: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any] = FROZEN_GATE,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    specifications = [
        ("cpu_vs_tensorrt_run_a", cpu, trt_a),
        ("cpu_vs_tensorrt_run_b", cpu, trt_b),
        ("tensorrt_run_a_vs_tensorrt_run_b", trt_a, trt_b),
    ]
    comparison_items: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    aggregates: List[Mapping[str, Any]] = []
    for name, reference_run, candidate_run in specifications:
        rows: List[Mapping[str, Any]] = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            row = dict(
                compare_image(
                    reference_run[sample_id],
                    candidate_run[sample_id],
                    name,
                    gate,
                )
            )
            row["sample_id"] = sample_id
            rows.append(row)
            comparison_items.setdefault(sample_id, {})[name] = row
        aggregates.append(aggregate_comparison(name, rows))
    per_image_items = [
        {
            "sequence_index": index,
            "sample_id": sample["sample_id"],
            "source_image_path": sample["image_path"],
            "source_image_sha256": sample["image_sha256"],
            "comparisons": comparison_items[str(sample["sample_id"])],
            "passed": all(row["passed"] for row in comparison_items[str(sample["sample_id"])].values()),
        }
        for index, sample in enumerate(samples)
    ]
    passed = all(value["passed"] for value in aggregates)
    return (
        {"comparisons": aggregates, "passed": passed},
        {"items": per_image_items, "passed": passed},
    )


def validate_provider_expectations(
    cpu_provider: str, tensorrt_provider: str, expected_tensorrt_provider: str
) -> None:
    expected = {"cpu": CPU_PROVIDER, "tensorrt": expected_tensorrt_provider}
    actual = {"cpu": cpu_provider, "tensorrt": tensorrt_provider}
    if actual != expected:
        fail("provider expectations", str(expected), actual, "pass the protocol-selected frozen provider names; provider identity is not user-redefinable")


def write_json(path: Path, document: Mapping[str, Any], overwrite: bool) -> None:
    destination = path.resolve()
    if destination.exists() and not overwrite:
        fail("output", "a new path or --overwrite", str(destination), "choose a new evidence path")
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(destination)
    except OSError as error:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        fail("output", "an atomically writable JSON destination", str(error), "fix the output path or permissions")


def reject_protected_destination(
    destination: Path,
    *,
    protected_files: Sequence[Path],
    protected_directories: Sequence[Path],
) -> None:
    resolved = destination.resolve()
    exact_files = {path.resolve() for path in protected_files}
    protected_directory = None
    for directory in protected_directories:
        root = directory.resolve()
        try:
            resolved.relative_to(root)
            protected_directory = root
            break
        except ValueError:
            continue
    if resolved in exact_files or protected_directory is not None:
        fail(
            "output.protected_input",
            "an evidence path distinct from every frozen input and outside every input/cache directory",
            str(resolved),
            "choose a dedicated evidence-output directory; --overwrite never authorizes replacing frozen inputs",
        )


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--cpu-dir", required=True, type=Path)
    parser.add_argument("--tensorrt-run-a-dir", required=True, type=Path)
    parser.add_argument("--tensorrt-run-b-dir", required=True, type=Path)
    parser.add_argument("--expected-cpu-provider", required=True)
    parser.add_argument("--expected-tensorrt-provider", required=True)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--per-image-output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        if arguments.summary_output.resolve() == arguments.per_image_output.resolve():
            fail("outputs", "two distinct paths", str(arguments.summary_output), "choose separate summary and per-image files")
        for output in (arguments.summary_output, arguments.per_image_output):
            if output.exists() and not arguments.overwrite:
                fail("output", "a new path or --overwrite", str(output), "choose a new evidence path")
        protocol = load_json_object(arguments.protocol)
        sources, protocol_specification = validate_protocol(
            protocol, arguments.protocol
        )
        validate_provider_expectations(
            arguments.expected_cpu_provider,
            arguments.expected_tensorrt_provider,
            str(protocol_specification["tensorrt_provider"]),
        )
        runtime_configs = validate_runtime_configs(sources, protocol_specification)
        artifact = validate_artifact_spec(sources["artifact_spec_path"])
        samples = validate_manifest(
            load_json_object(sources["consistency_manifest_path"]),
            sources["consistency_manifest_path"],
            protocol_specification["sample_contract"],
        )
        protected_files = [
            arguments.protocol,
            sources["consistency_manifest_path"],
            sources["cpu_runtime_config_path"],
            sources["tensorrt_runtime_config_path"],
            sources["artifact_spec_path"],
            Path(artifact["model_path"]),
            *(Path(str(sample["verified_image_path"])) for sample in samples),
        ]
        protected_directories = [
            arguments.cpu_dir,
            arguments.tensorrt_run_a_dir,
            arguments.tensorrt_run_b_dir,
        ]
        engine_path_value = runtime_configs["tensorrt"].get("engine_path")
        if engine_path_value is not None:
            engine_path = Path(str(engine_path_value))
            protected_files.append(engine_path)
            protected_directories.append(engine_path.parent)
        for output in (arguments.summary_output, arguments.per_image_output):
            reject_protected_destination(
                output,
                protected_files=protected_files,
                protected_directories=protected_directories,
            )
        engine_contract = protocol_specification.get("engine_contract")
        cpu = collect_run(arguments.cpu_dir, samples=samples, expected_provider=arguments.expected_cpu_provider, run_name="cpu_run")
        trt_a = collect_run(arguments.tensorrt_run_a_dir, samples=samples, expected_provider=arguments.expected_tensorrt_provider, run_name="tensorrt_run_a", expected_engine_contract=engine_contract)
        trt_b = collect_run(arguments.tensorrt_run_b_dir, samples=samples, expected_provider=arguments.expected_tensorrt_provider, run_name="tensorrt_run_b", expected_engine_contract=engine_contract)
        final_runtime_configs = validate_runtime_configs(sources, protocol_specification)
        if final_runtime_configs != runtime_configs:
            fail("runtime_configs.after_runs", str(runtime_configs), final_runtime_configs, "discard evidence because a frozen source changed during comparison")
        gate = protocol_specification["gate"]
        result, per_image = compare_detection_runs(
            samples, cpu, trt_a, trt_b, gate
        )
        timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        source_paths = {
            "protocol": str(arguments.protocol.resolve()),
            "manifest": str(sources["consistency_manifest_path"]),
            "artifact_spec": str(sources["artifact_spec_path"]),
            "model": str(artifact["model_path"]),
            "cpu_directory": str(arguments.cpu_dir.resolve()),
            "tensorrt_run_a_directory": str(arguments.tensorrt_run_a_dir.resolve()),
            "tensorrt_run_b_directory": str(arguments.tensorrt_run_b_dir.resolve()),
        }
        summary_document = {
            "schema_version": SCHEMA_VERSION,
            "evidence_type": protocol_specification["evidence_type"],
            "protocol_id": protocol["protocol_id"],
            "timestamp_utc": timestamp,
            "passed": result["passed"],
            "sources": source_paths,
            "provider_expectations": {
                "cpu": arguments.expected_cpu_provider,
                "tensorrt": arguments.expected_tensorrt_provider,
            },
            "artifact": {
                "model_id": artifact["model_id"],
                "declared_and_actual_sha256": artifact["model_sha256"],
                "semantics_verified_against_protocol": True,
            },
            "runtime_configs": runtime_configs,
            "gate": dict(gate),
            "comparisons": result["comparisons"],
            "performance_publication_allowed": result["passed"],
        }
        per_image_document = {
            "schema_version": SCHEMA_VERSION,
            "evidence_type": str(protocol_specification["evidence_type"]) + "_per_image",
            "protocol_id": protocol["protocol_id"],
            "timestamp_utc": timestamp,
            "passed": per_image["passed"],
            "sources": source_paths,
            "gate": dict(gate),
            "items": per_image["items"],
        }
        write_json(arguments.summary_output, summary_document, arguments.overwrite)
        write_json(arguments.per_image_output, per_image_document, arguments.overwrite)
    except CorrectnessError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        "S2-04 correctness: "
        f"passed={summary_document['passed']}, samples={len(samples)}, "
        f"summary={arguments.summary_output.resolve()}, per_image={arguments.per_image_output.resolve()}"
    )
    return 0 if summary_document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

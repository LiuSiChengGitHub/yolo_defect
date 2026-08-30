"""Strictly validate an S2-03 bounded-concurrency BatchSummary JSON file.

The validator is intentionally standard-library-only so the same command can
run in Windows, Linux x86_64, and the Linux AArch64/QEMU workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, NoReturn, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "cpp_ort_multi_image_batch_summary"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VALID_STATUSES = {"succeeded", "partial_failure", "cancelled", "fatal"}
VALID_ITEM_STATUSES = {"succeeded", "failed", "cancelled"}
EXPECTED_ORDERING = {
    "directory": "recursive UTF-8 generic relative-path lexical order; supported regular files only; symlinks not followed",
    "manifest": "UTF-8 path-list declaration order",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "evidence_type",
    "timestamp_utc",
    "status",
    "cooperative_stop_requested",
    "command_arguments",
    "environment",
    "runtime",
    "model",
    "input",
    "output",
    "counts",
    "queue",
    "timing",
    "latency_ms",
    "throughput_images_per_second",
    "memory",
    "items",
    "limitations",
    "fatal_error",
}


class BatchSummaryValidationError(AssertionError):
    """Raised when a BatchSummary violates the public schema or invariants."""


def fail(message: str) -> NoReturn:
    raise BatchSummaryValidationError(message)


def reject_constant(value: str) -> NoReturn:
    fail(f"JSON: expected RFC-compliant finite number, actual {value}")


def reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON object: expected unique keys, duplicate {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicate_keys,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"{path}: expected readable strict UTF-8 JSON, actual {error}")


def expect_exact_keys(value: Any, expected: Set[str], object_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{object_name}: expected object, actual {type(value).__name__}")
    actual = set(value)
    if actual != expected:
        fail(
            f"{object_name}: expected keys {sorted(expected)}, actual {sorted(actual)}; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return value


def expect_string(value: Any, object_name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        fail(f"{object_name}: expected string, actual {value!r}")
    if nonempty and not value.strip():
        fail(f"{object_name}: expected non-empty string, actual {value!r}")
    if "\x00" in value:
        fail(f"{object_name}: expected string without NUL, actual {value!r}")
    return value


def expect_nullable_string(value: Any, object_name: str) -> Optional[str]:
    if value is None:
        return None
    return expect_string(value, object_name)


def expect_bool(value: Any, object_name: str) -> bool:
    if type(value) is not bool:
        fail(f"{object_name}: expected boolean, actual {value!r}")
    return value


def expect_int(
    value: Any,
    object_name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        fail(f"{object_name}: expected integer, actual {value!r}")
    if minimum is not None and value < minimum:
        fail(f"{object_name}: expected >= {minimum}, actual {value}")
    if maximum is not None and value > maximum:
        fail(f"{object_name}: expected <= {maximum}, actual {value}")
    return value


def expect_finite_number(
    value: Any,
    object_name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if type(value) not in (int, float):
        fail(f"{object_name}: expected number, actual {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        fail(f"{object_name}: expected finite number, actual {value!r}")
    if minimum is not None and converted < minimum:
        fail(f"{object_name}: expected >= {minimum}, actual {converted}")
    if maximum is not None and converted > maximum:
        fail(f"{object_name}: expected <= {maximum}, actual {converted}")
    return converted


def expect_string_array(
    value: Any,
    object_name: str,
    *,
    allow_empty: bool = False,
    unique: bool = False,
) -> List[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        fail(f"{object_name}: expected {'possibly empty ' if allow_empty else 'non-empty '}string array")
    result = [expect_string(item, f"{object_name}[{index}]") for index, item in enumerate(value)]
    if unique and len(result) != len(set(result)):
        fail(f"{object_name}: expected unique strings, actual duplicates")
    return result


def normalized_path(value: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(value)))


def path_from_json(value: Any, object_name: str, base: Path) -> Path:
    raw = expect_string(value, object_name)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_architecture(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    aliases = {
        "amd64": "x86_64",
        "x8664": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }
    return aliases.get(normalized, normalized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        fail(f"{path}: expected readable regular file for SHA-256, actual {error}")
    return digest.hexdigest().upper()


def discover_directory(directory: Path) -> List[Path]:
    discovered: List[Tuple[str, Path]] = []
    try:
        for current_root, directory_names, file_names in os.walk(directory, followlinks=False):
            current = Path(current_root)
            directory_names[:] = [
                name for name in directory_names if not (current / name).is_symlink()
            ]
            for name in file_names:
                candidate = current / name
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                if candidate.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                relative_key = candidate.relative_to(directory).as_posix()
                discovered.append((relative_key, candidate.resolve(strict=True)))
    except OSError as error:
        fail(f"input.source_path: directory discovery failed: {error}")
    discovered.sort(key=lambda pair: pair[0])
    return [path for _, path in discovered]


def discover_manifest(manifest: Path) -> List[Path]:
    try:
        document = manifest.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        fail(f"input.source_path: manifest read failed: {error}")
    paths: List[Path] = []
    seen: Set[str] = set()
    for line_number, raw_line in enumerate(document.split("\n"), 1):
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        # Match the C++ path-list grammar exactly: only ASCII horizontal
        # whitespace is ignored around a declaration. Unicode whitespace may
        # legally be part of a filename and must not be reinterpreted here.
        stripped = line.strip(" \t\v\f")
        if not stripped or stripped.startswith("#"):
            continue
        declared = Path(stripped)
        if declared.is_absolute():
            fail(f"manifest line {line_number}: expected relative path, actual {stripped!r}")
        candidate = (manifest.parent / declared).resolve(strict=False)
        if not candidate.is_file():
            fail(f"manifest line {line_number}: expected existing regular file, actual {candidate}")
        if candidate.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
            fail(f"manifest line {line_number}: unsupported image extension for {candidate}")
        identity = normalized_path(str(candidate))
        if identity in seen:
            fail(f"manifest line {line_number}: duplicate canonical image path {candidate}")
        seen.add(identity)
        paths.append(candidate)
    return paths


def validate_detection_document(
    document: Any,
    *,
    object_name: str,
    expected_source: Path,
    expected_model_id: str,
    expected_model_sha256: str,
    expected_provider: str,
    expected_score_threshold: float,
    expected_nms_threshold: float,
    expected_nms_mode: str,
) -> int:
    root = expect_exact_keys(document, {"schema_version", "model", "image", "runtime", "detections"}, object_name)
    if expect_int(root["schema_version"], f"{object_name}.schema_version") != 1:
        fail(f"{object_name}.schema_version: expected 1")
    model = expect_exact_keys(root["model"], {"model_id", "declared_sha256"}, f"{object_name}.model")
    if expect_string(model["model_id"], f"{object_name}.model.model_id") != expected_model_id:
        fail(f"{object_name}.model.model_id: does not match BatchSummary model")
    if expect_string(model["declared_sha256"], f"{object_name}.model.declared_sha256").upper() != expected_model_sha256.upper():
        fail(f"{object_name}.model.declared_sha256: does not match BatchSummary model")
    image = expect_exact_keys(root["image"], {"path", "original_size", "input_size"}, f"{object_name}.image")
    if normalized_path(expect_string(image["path"], f"{object_name}.image.path")) != normalized_path(str(expected_source)):
        fail(f"{object_name}.image.path: does not match item source_path")
    original_size = expect_exact_keys(image["original_size"], {"width", "height", "channels"}, f"{object_name}.image.original_size")
    for key in ("width", "height", "channels"):
        expect_int(original_size[key], f"{object_name}.image.original_size.{key}", minimum=1)
    input_size = expect_exact_keys(image["input_size"], {"width", "height"}, f"{object_name}.image.input_size")
    for key in ("width", "height"):
        expect_int(input_size[key], f"{object_name}.image.input_size.{key}", minimum=1)
    runtime = expect_exact_keys(
        root["runtime"],
        {"actual_provider", "provider_evidence", "score_threshold", "nms_threshold", "nms_mode"},
        f"{object_name}.runtime",
    )
    expected_runtime_strings = {
        "actual_provider": expected_provider,
        "nms_mode": expected_nms_mode,
    }
    for key, expected in expected_runtime_strings.items():
        actual = expect_string(runtime[key], f"{object_name}.runtime.{key}")
        if actual != expected:
            fail(f"{object_name}.runtime.{key}: expected {expected!r}, actual {actual!r}")
    expect_string(runtime["provider_evidence"], f"{object_name}.runtime.provider_evidence")
    for key, expected in (("score_threshold", expected_score_threshold), ("nms_threshold", expected_nms_threshold)):
        actual = expect_finite_number(runtime[key], f"{object_name}.runtime.{key}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12):
            fail(f"{object_name}.runtime.{key}: expected {expected}, actual {actual}")
    detections = root["detections"]
    if not isinstance(detections, list):
        fail(f"{object_name}.detections: expected array")
    previous_confidence = math.inf
    for index, detection_value in enumerate(detections):
        detection_name = f"{object_name}.detections[{index}]"
        detection = expect_exact_keys(
            detection_value,
            {"class_id", "class_name", "confidence", "bbox_xyxy"},
            detection_name,
        )
        expect_int(detection["class_id"], f"{detection_name}.class_id", minimum=0)
        expect_string(detection["class_name"], f"{detection_name}.class_name")
        confidence = expect_finite_number(detection["confidence"], f"{detection_name}.confidence", minimum=0.0, maximum=1.0)
        if confidence > previous_confidence:
            fail(f"{detection_name}.confidence: expected non-increasing order")
        previous_confidence = confidence
        bbox = detection["bbox_xyxy"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            fail(f"{detection_name}.bbox_xyxy: expected four-number array")
        coordinates = [expect_finite_number(value, f"{detection_name}.bbox_xyxy[{axis}]") for axis, value in enumerate(bbox)]
        if coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]:
            fail(f"{detection_name}.bbox_xyxy: expected x1 <= x2 and y1 <= y2")
    return len(detections)


def validate_document(
    document: Any,
    *,
    summary_path: Optional[Path] = None,
    expected_status: Optional[str] = None,
    expected_counts: Optional[Dict[str, int]] = None,
    expected_target_architecture: Optional[str] = None,
    expected_runtime_kernel_architecture: Optional[str] = None,
    expected_execution_context: Optional[str] = None,
    expected_requested_workers: Optional[int] = None,
    expected_effective_workers: Optional[int] = None,
    expected_input_kind: Optional[str] = None,
    expected_memory_publishable: Optional[bool] = None,
    check_referenced_files: bool = True,
) -> Dict[str, Any]:
    root = expect_exact_keys(document, TOP_LEVEL_KEYS, "root")
    if expect_int(root["schema_version"], "schema_version") != SCHEMA_VERSION:
        fail(f"schema_version: expected {SCHEMA_VERSION}, actual {root['schema_version']!r}")
    if expect_string(root["evidence_type"], "evidence_type") != EVIDENCE_TYPE:
        fail(f"evidence_type: expected {EVIDENCE_TYPE!r}, actual {root['evidence_type']!r}")
    timestamp = expect_string(root["timestamp_utc"], "timestamp_utc")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp) is None:
        fail(f"timestamp_utc: expected YYYY-MM-DDTHH:MM:SSZ, actual {timestamp!r}")
    status = expect_string(root["status"], "status")
    if status not in VALID_STATUSES:
        fail(f"status: expected one of {sorted(VALID_STATUSES)}, actual {status!r}")
    if expected_status is not None and status != expected_status:
        fail(f"status: expected {expected_status!r}, actual {status!r}")
    cooperative_stop_requested = expect_bool(
        root["cooperative_stop_requested"], "cooperative_stop_requested"
    )
    command = expect_string_array(root["command_arguments"], "command_arguments")
    if "--batch" not in command:
        fail("command_arguments: expected --batch")

    base = summary_path.resolve(strict=False).parent if summary_path is not None else Path.cwd()
    environment = expect_exact_keys(
        root["environment"],
        {
            "hostname", "processor", "logical_cpu_count", "os_name", "os_version",
            "target_architecture", "runtime_kernel_architecture", "execution_context",
            "compiler_id", "compiler_version", "build_type", "cxx_standard",
            "opencv_version", "onnxruntime_version",
        },
        "environment",
    )
    for key in (
        "hostname", "processor", "os_name", "os_version", "target_architecture",
        "runtime_kernel_architecture", "execution_context", "compiler_id",
        "compiler_version", "opencv_version",
    ):
        expect_string(environment[key], f"environment.{key}")
    expect_string(
        environment["onnxruntime_version"],
        "environment.onnxruntime_version",
        nonempty=status != "fatal",
    )
    expect_int(environment["logical_cpu_count"], "environment.logical_cpu_count", minimum=1)
    if expect_string(environment["build_type"], "environment.build_type") != "Release":
        fail(f"environment.build_type: expected 'Release', actual {environment['build_type']!r}")
    if expect_int(environment["cxx_standard"], "environment.cxx_standard") != 17:
        fail(f"environment.cxx_standard: expected 17, actual {environment['cxx_standard']!r}")
    if expected_target_architecture is not None:
        actual_architecture = normalize_architecture(environment["target_architecture"])
        expected_architecture = normalize_architecture(expected_target_architecture)
        if actual_architecture != expected_architecture:
            fail(
                "environment.target_architecture: expected architecture equivalent to "
                f"{expected_target_architecture!r}, actual {environment['target_architecture']!r}"
            )
    if expected_runtime_kernel_architecture is not None:
        actual_architecture = normalize_architecture(
            environment["runtime_kernel_architecture"]
        )
        expected_architecture = normalize_architecture(
            expected_runtime_kernel_architecture
        )
        if actual_architecture != expected_architecture:
            fail(
                "environment.runtime_kernel_architecture: expected architecture "
                f"equivalent to {expected_runtime_kernel_architecture!r}, actual "
                f"{environment['runtime_kernel_architecture']!r}"
            )
    if expected_execution_context is not None and environment["execution_context"] != expected_execution_context:
        fail(
            f"environment.execution_context: expected {expected_execution_context!r}, "
            f"actual {environment['execution_context']!r}"
        )

    runtime = expect_exact_keys(
        root["runtime"],
        {
            "config_path", "requested_provider", "actual_provider", "provider_evidence",
            "execution_mode", "intra_op_num_threads", "inter_op_num_threads",
            "graph_optimization_level", "score_threshold", "nms_threshold", "nms_mode",
            "requested_workers", "effective_workers", "session_count",
            "session_initialization_ms",
        },
        "runtime",
    )
    config_path = path_from_json(runtime["config_path"], "runtime.config_path", base)
    expected_runtime_strings = {"requested_provider": "cpu"}
    if status != "fatal":
        expected_runtime_strings.update(
            {
                "actual_provider": "CPUExecutionProvider",
                "execution_mode": "sequential",
                "graph_optimization_level": "all",
            }
        )
    for key, expected in expected_runtime_strings.items():
        actual = expect_string(runtime[key], f"runtime.{key}")
        if actual != expected:
            fail(f"runtime.{key}: expected {expected!r}, actual {actual!r}")
    for key in ("actual_provider", "provider_evidence", "execution_mode", "graph_optimization_level"):
        expect_string(runtime[key], f"runtime.{key}", nonempty=status != "fatal")
    score_threshold = expect_finite_number(runtime["score_threshold"], "runtime.score_threshold", minimum=0.0, maximum=1.0)
    nms_threshold = expect_finite_number(runtime["nms_threshold"], "runtime.nms_threshold", minimum=0.0, maximum=1.0)
    nms_mode = expect_string(runtime["nms_mode"], "runtime.nms_mode")
    if nms_mode not in {"class_agnostic", "class_aware"}:
        fail(f"runtime.nms_mode: expected class_agnostic or class_aware, actual {nms_mode!r}")
    intra_op_threads = expect_int(runtime["intra_op_num_threads"], "runtime.intra_op_num_threads", minimum=0)
    inter_op_threads = expect_int(runtime["inter_op_num_threads"], "runtime.inter_op_num_threads", minimum=0)
    if status != "fatal" and (intra_op_threads != 1 or inter_op_threads != 1):
        fail("runtime ORT threads: expected intra/inter-op 1/1")
    requested_workers = expect_int(runtime["requested_workers"], "runtime.requested_workers", minimum=1, maximum=64)
    effective_workers = expect_int(runtime["effective_workers"], "runtime.effective_workers", minimum=1, maximum=64)
    session_count = expect_int(runtime["session_count"], "runtime.session_count", minimum=0, maximum=64)
    if effective_workers > requested_workers or session_count > effective_workers:
        fail(
            "runtime worker/session invariant: expected session_count <= effective_workers "
            "<= requested_workers"
        )
    if status != "fatal" and session_count != effective_workers:
        fail("runtime.session_count: expected one initialized session per effective worker")
    if expected_requested_workers is not None and requested_workers != expected_requested_workers:
        fail(f"runtime.requested_workers: expected {expected_requested_workers}, actual {requested_workers}")
    if expected_effective_workers is not None and effective_workers != expected_effective_workers:
        fail(f"runtime.effective_workers: expected {expected_effective_workers}, actual {effective_workers}")
    initialization_values = runtime["session_initialization_ms"]
    if not isinstance(initialization_values, list) or len(initialization_values) != session_count:
        fail("runtime.session_initialization_ms: expected one entry per session")
    for index, value in enumerate(initialization_values):
        expect_finite_number(value, f"runtime.session_initialization_ms[{index}]", minimum=0.0)

    model = expect_exact_keys(
        root["model"],
        {
            "model_id", "model_family", "model_path", "declared_sha256", "opset",
            "input_name", "input_shape", "input_dtype", "input_layout",
        },
        "model",
    )
    model_id = expect_string(model["model_id"], "model.model_id")
    expect_string(model["model_family"], "model.model_family")
    model_path = path_from_json(model["model_path"], "model.model_path", base)
    model_sha256 = expect_string(model["declared_sha256"], "model.declared_sha256").upper()
    if re.fullmatch(r"[0-9A-F]{64}", model_sha256) is None:
        fail(f"model.declared_sha256: expected 64 hexadecimal characters, actual {model_sha256!r}")
    expect_int(model["opset"], "model.opset", minimum=1)
    expect_string(model["input_name"], "model.input_name")
    input_shape = model["input_shape"]
    if not isinstance(input_shape, list) or len(input_shape) != 4:
        fail(f"model.input_shape: expected four-integer array, actual {input_shape!r}")
    for index, value in enumerate(input_shape):
        expect_int(value, f"model.input_shape[{index}]", minimum=1)
    if expect_string(model["input_dtype"], "model.input_dtype") != "float32":
        fail("model.input_dtype: expected 'float32'")
    if expect_string(model["input_layout"], "model.input_layout") != "nchw":
        fail("model.input_layout: expected 'nchw'")

    input_value = expect_exact_keys(root["input"], {"kind", "source_path", "ordering"}, "input")
    input_kind = expect_string(input_value["kind"], "input.kind")
    if input_kind not in {"directory", "manifest"}:
        fail(f"input.kind: expected directory or manifest, actual {input_kind!r}")
    if expected_input_kind is not None and input_kind != expected_input_kind:
        fail(f"input.kind: expected {expected_input_kind!r}, actual {input_kind!r}")
    input_source_path = path_from_json(input_value["source_path"], "input.source_path", base)
    input_ordering = expect_string(input_value["ordering"], "input.ordering")
    if input_ordering != EXPECTED_ORDERING[input_kind]:
        fail(
            f"input.ordering: expected {EXPECTED_ORDERING[input_kind]!r} for "
            f"{input_kind}, actual {input_ordering!r}"
        )

    output = expect_exact_keys(
        root["output"],
        {"directory", "batch_summary_path", "item_directory", "json_outputs", "image_outputs", "overwrite_existing"},
        "output",
    )
    output_directory = path_from_json(output["directory"], "output.directory", base)
    recorded_summary_path = path_from_json(output["batch_summary_path"], "output.batch_summary_path", base)
    item_directory = path_from_json(output["item_directory"], "output.item_directory", base)
    if not is_relative_to(item_directory, output_directory):
        fail("output.item_directory: expected path inside output.directory")
    if item_directory.name != "items":
        fail(f"output.item_directory: expected final component 'items', actual {item_directory.name!r}")
    if not expect_bool(output["json_outputs"], "output.json_outputs"):
        fail("output.json_outputs: expected true")
    image_outputs = expect_bool(output["image_outputs"], "output.image_outputs")
    expect_bool(output["overwrite_existing"], "output.overwrite_existing")
    if summary_path is not None and normalized_path(str(recorded_summary_path)) != normalized_path(str(summary_path.resolve(strict=False))):
        fail(
            f"output.batch_summary_path: expected {summary_path.resolve(strict=False)}, "
            f"actual {recorded_summary_path}"
        )

    counts = expect_exact_keys(root["counts"], {"discovered", "enqueued", "started", "succeeded", "failed", "cancelled"}, "counts")
    count_values = {key: expect_int(value, f"counts.{key}", minimum=0) for key, value in counts.items()}
    if expected_counts:
        for key, expected in expected_counts.items():
            if key not in count_values:
                fail(f"expected count name: unknown {key!r}")
            if count_values[key] != expected:
                fail(f"counts.{key}: expected {expected}, actual {count_values[key]}")
    discovered = count_values["discovered"]
    if discovered < 1:
        fail("counts.discovered: expected at least one task")
    if count_values["started"] != count_values["succeeded"] + count_values["failed"]:
        fail("counts: expected started == succeeded + failed")
    if discovered != count_values["succeeded"] + count_values["failed"] + count_values["cancelled"]:
        fail("counts: expected discovered == succeeded + failed + cancelled")
    if not count_values["started"] <= count_values["enqueued"] <= discovered:
        fail("counts: expected started <= enqueued <= discovered")
    if effective_workers != min(requested_workers, discovered):
        fail("runtime.effective_workers: expected min(requested_workers, counts.discovered)")

    fatal_error = expect_nullable_string(root["fatal_error"], "fatal_error")
    expected_derived_status = "succeeded"
    if fatal_error is not None:
        expected_derived_status = "fatal"
    elif cooperative_stop_requested or count_values["cancelled"]:
        expected_derived_status = "cancelled"
    elif count_values["failed"]:
        expected_derived_status = "partial_failure"
    if status != expected_derived_status:
        fail(
            "status: expected derivation fatal_error > cooperative stop/cancelled "
            f"> failed > success, actual status={status!r}, "
            f"cooperative_stop_requested={cooperative_stop_requested!r}"
        )
    if status == "succeeded" and count_values["succeeded"] != discovered:
        fail("status succeeded: expected every task succeeded")
    if status in {"succeeded", "partial_failure"} and count_values["enqueued"] != discovered:
        fail("counts.enqueued: expected all discovered tasks enqueued for non-cancelled completion")

    queue = expect_exact_keys(root["queue"], {"capacity", "peak_depth", "producer_wait_count", "producer_wait_ms"}, "queue")
    queue_capacity = expect_int(queue["capacity"], "queue.capacity", minimum=1, maximum=4096)
    peak_depth = expect_int(queue["peak_depth"], "queue.peak_depth", minimum=0, maximum=queue_capacity)
    producer_wait_count = expect_int(queue["producer_wait_count"], "queue.producer_wait_count", minimum=0)
    producer_wait_ms = expect_finite_number(queue["producer_wait_ms"], "queue.producer_wait_ms", minimum=0.0)
    if producer_wait_count == 0 and producer_wait_ms != 0.0:
        fail("queue.producer_wait_ms: expected 0 when producer_wait_count is 0")
    if count_values["enqueued"] and peak_depth < 1:
        fail("queue.peak_depth: expected at least 1 when tasks were enqueued")

    timing = expect_exact_keys(root["timing"], {"processing_wall_ms", "includes", "excludes"}, "timing")
    processing_wall_ms = expect_finite_number(timing["processing_wall_ms"], "timing.processing_wall_ms", minimum=0.0)
    expect_string_array(timing["includes"], "timing.includes", unique=True)
    expect_string_array(timing["excludes"], "timing.excludes", unique=True)
    if count_values["started"] > 0 and processing_wall_ms <= 0.0:
        fail("timing.processing_wall_ms: expected positive value after tasks started")

    latency = expect_exact_keys(root["latency_ms"], {"sample_count", "mean_ms", "p50_ms", "p95_ms"}, "latency_ms")
    latency_sample_count = expect_int(latency["sample_count"], "latency_ms.sample_count", minimum=0)
    if latency_sample_count != count_values["succeeded"]:
        fail("latency_ms.sample_count: expected counts.succeeded")
    latency_numbers = {
        key: expect_finite_number(latency[key], f"latency_ms.{key}", minimum=0.0)
        for key in ("mean_ms", "p50_ms", "p95_ms")
    }
    if latency_numbers["p50_ms"] > latency_numbers["p95_ms"]:
        fail("latency_ms: expected p50_ms <= p95_ms")
    if latency_sample_count == 0 and any(value != 0.0 for value in latency_numbers.values()):
        fail("latency_ms: expected zero statistics when sample_count is zero")

    throughput = expect_finite_number(root["throughput_images_per_second"], "throughput_images_per_second", minimum=0.0)
    expected_throughput = 0.0 if processing_wall_ms == 0.0 else count_values["succeeded"] * 1000.0 / processing_wall_ms
    if not math.isclose(throughput, expected_throughput, rel_tol=1.0e-9, abs_tol=1.0e-9):
        fail(
            "throughput_images_per_second: expected counts.succeeded * 1000 / "
            f"timing.processing_wall_ms ({expected_throughput}), actual {throughput}"
        )

    memory = expect_exact_keys(root["memory"], {"supported", "status", "metric", "bytes", "mebibytes", "scope", "reason", "publishable"}, "memory")
    memory_supported = expect_bool(memory["supported"], "memory.supported")
    memory_status = expect_string(memory["status"], "memory.status")
    memory_metric = expect_string(memory["metric"], "memory.metric", nonempty=memory_supported)
    memory_publishable = expect_bool(memory["publishable"], "memory.publishable")
    expect_string(memory["scope"], "memory.scope")
    if expected_memory_publishable is not None and memory_publishable != expected_memory_publishable:
        fail(f"memory.publishable: expected {expected_memory_publishable}, actual {memory_publishable}")
    if memory_supported:
        if memory_status != "supported":
            fail("memory.status: expected 'supported' when memory.supported is true")
        if memory_metric not in {"peak_working_set", "peak_rss"}:
            fail(f"memory.metric: expected peak_working_set or peak_rss, actual {memory_metric!r}")
        memory_bytes = expect_int(memory["bytes"], "memory.bytes", minimum=1)
        memory_mebibytes = expect_finite_number(memory["mebibytes"], "memory.mebibytes", minimum=0.0)
        expected_mebibytes = memory_bytes / (1024.0 * 1024.0)
        if not math.isclose(memory_mebibytes, expected_mebibytes, rel_tol=1.0e-9, abs_tol=1.0e-9):
            fail(f"memory.mebibytes: expected bytes/(1024*1024)={expected_mebibytes}, actual {memory_mebibytes}")
        if memory_publishable:
            if memory["reason"] is not None:
                fail("memory.reason: expected null for supported publishable memory")
        else:
            expect_string(memory["reason"], "memory.reason")
    else:
        if memory_status not in {"unsupported", "unavailable"}:
            fail("memory.status: expected 'unsupported' or 'unavailable' when memory.supported is false")
        if memory["bytes"] is not None or memory["mebibytes"] is not None:
            fail("memory: expected null bytes/mebibytes when unsupported")
        expect_string(memory["reason"], "memory.reason")
        if memory_publishable:
            fail("memory.publishable: unsupported memory cannot be publishable")
    if normalize_architecture(environment["target_architecture"]) != normalize_architecture(environment["runtime_kernel_architecture"]) and memory_publishable:
        fail("memory.publishable: expected false when target and runtime-kernel architectures differ")

    items = root["items"]
    if not isinstance(items, list) or len(items) != discovered:
        fail(f"items: expected array of counts.discovered ({discovered}) entries")
    expected_source_paths: Optional[List[Path]] = None
    if check_referenced_files:
        if not config_path.is_file():
            fail(f"runtime.config_path: expected existing regular file, actual {config_path}")
        if not model_path.is_file():
            fail(f"model.model_path: expected existing regular file, actual {model_path}")
        if sha256_file(model_path) != model_sha256:
            fail("model.declared_sha256: does not match referenced model bytes")
        if input_kind == "directory":
            if not input_source_path.is_dir():
                fail(f"input.source_path: expected directory, actual {input_source_path}")
            expected_source_paths = discover_directory(input_source_path)
        else:
            if not input_source_path.is_file():
                fail(f"input.source_path: expected manifest file, actual {input_source_path}")
            expected_source_paths = discover_manifest(input_source_path)
        if len(expected_source_paths) != discovered:
            fail(f"input discovery: expected {discovered} tasks, rediscovered {len(expected_source_paths)}")
        if count_values["succeeded"] > 0 and (
            not output_directory.is_dir() or not item_directory.is_dir()
        ):
            fail(
                "output paths: expected existing output.directory and "
                "output.item_directory after at least one succeeded item"
            )
        if summary_path is not None and not summary_path.is_file():
            fail(f"summary path: expected existing regular file, actual {summary_path}")

    item_status_counts = {"succeeded": 0, "failed": 0, "cancelled": 0}
    seen_sources: Set[str] = set()
    seen_outputs: Set[str] = set()
    successful_latencies: List[float] = []
    for index, item_value in enumerate(items):
        item_name = f"items[{index}]"
        item = expect_exact_keys(
            item_value,
            {"sequence_index", "status", "source_path", "json_output_path", "image_output_path", "detection_count", "latency_ms", "error"},
            item_name,
        )
        if expect_int(item["sequence_index"], f"{item_name}.sequence_index") != index:
            fail(f"{item_name}.sequence_index: expected {index}, actual {item['sequence_index']!r}")
        item_status = expect_string(item["status"], f"{item_name}.status")
        if item_status not in VALID_ITEM_STATUSES:
            fail(f"{item_name}.status: expected one of {sorted(VALID_ITEM_STATUSES)}, actual {item_status!r}")
        item_status_counts[item_status] += 1
        source_path = path_from_json(item["source_path"], f"{item_name}.source_path", base)
        source_identity = normalized_path(str(source_path))
        if source_identity in seen_sources:
            fail(f"{item_name}.source_path: duplicate canonical source path {source_path}")
        seen_sources.add(source_identity)
        if expected_source_paths is not None and source_identity != normalized_path(str(expected_source_paths[index])):
            fail(f"{item_name}.source_path: does not match deterministic discovered task at index {index}")
        json_output_raw = expect_nullable_string(item["json_output_path"], f"{item_name}.json_output_path")
        image_output_raw = expect_nullable_string(item["image_output_path"], f"{item_name}.image_output_path")
        detection_count = expect_int(item["detection_count"], f"{item_name}.detection_count", minimum=0)
        item_latency = expect_finite_number(item["latency_ms"], f"{item_name}.latency_ms", minimum=0.0)
        error = expect_nullable_string(item["error"], f"{item_name}.error")
        expected_json_name = f"{index:06d}.detections.json"
        expected_image_name = f"{index:06d}.visualized.png"
        if item_status == "succeeded":
            if json_output_raw is None or error is not None:
                fail(f"{item_name}: succeeded item requires JSON output and null error")
            if item_latency <= 0.0:
                fail(f"{item_name}.latency_ms: expected positive latency for succeeded item")
            json_output_path = path_from_json(json_output_raw, f"{item_name}.json_output_path", base)
            if json_output_path.name != expected_json_name or not is_relative_to(json_output_path, item_directory):
                fail(f"{item_name}.json_output_path: expected {expected_json_name} inside item_directory")
            output_identity = normalized_path(str(json_output_path))
            if output_identity in seen_outputs:
                fail(f"{item_name}.json_output_path: duplicate output path")
            seen_outputs.add(output_identity)
            if image_outputs:
                if image_output_raw is None:
                    fail(f"{item_name}.image_output_path: expected visualization output")
                image_output_path = path_from_json(image_output_raw, f"{item_name}.image_output_path", base)
                if image_output_path.name != expected_image_name or not is_relative_to(image_output_path, item_directory):
                    fail(f"{item_name}.image_output_path: expected {expected_image_name} inside item_directory")
                image_identity = normalized_path(str(image_output_path))
                if image_identity in seen_outputs:
                    fail(f"{item_name}.image_output_path: duplicate output path")
                seen_outputs.add(image_identity)
                if check_referenced_files and not image_output_path.is_file():
                    fail(f"{item_name}.image_output_path: expected existing regular file")
            elif image_output_raw is not None:
                fail(f"{item_name}.image_output_path: expected null when image_outputs is false")
            if check_referenced_files:
                if not json_output_path.is_file():
                    fail(f"{item_name}.json_output_path: expected existing regular file")
                detection_document = load_json(json_output_path)
                actual_detection_count = validate_detection_document(
                    detection_document,
                    object_name=f"{item_name}.detection_json",
                    expected_source=source_path,
                    expected_model_id=model_id,
                    expected_model_sha256=model_sha256,
                    expected_provider=runtime["actual_provider"],
                    expected_score_threshold=score_threshold,
                    expected_nms_threshold=nms_threshold,
                    expected_nms_mode=nms_mode,
                )
                if actual_detection_count != detection_count:
                    fail(f"{item_name}.detection_count: expected {actual_detection_count} from JSON, actual {detection_count}")
            successful_latencies.append(item_latency)
        else:
            if json_output_raw is not None or image_output_raw is not None:
                fail(f"{item_name}: non-succeeded item must not claim output files")
            if detection_count != 0:
                fail(f"{item_name}.detection_count: expected 0 for non-succeeded item")
            if item_status == "failed":
                if error is None:
                    fail(f"{item_name}: failed item requires a non-null error")
            else:
                if item_latency != 0.0:
                    fail(f"{item_name}.latency_ms: expected 0 for cancelled item")

    if item_status_counts["succeeded"] != count_values["succeeded"] or item_status_counts["failed"] != count_values["failed"] or item_status_counts["cancelled"] != count_values["cancelled"]:
        fail("items statuses: expected exact agreement with counts")
    if successful_latencies:
        sorted_latencies = sorted(successful_latencies)
        nearest_rank = lambda percentile: sorted_latencies[max(0, math.ceil(percentile * len(sorted_latencies)) - 1)]
        expected_latency_values = {
            "mean_ms": sum(successful_latencies) / len(successful_latencies),
            "p50_ms": nearest_rank(0.50),
            "p95_ms": nearest_rank(0.95),
        }
        for key, expected in expected_latency_values.items():
            if not math.isclose(latency_numbers[key], expected, rel_tol=1.0e-9, abs_tol=1.0e-9):
                fail(f"latency_ms.{key}: expected aggregate {expected} from successful items, actual {latency_numbers[key]}")

    expect_string_array(root["limitations"], "limitations", unique=True)
    return root


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_path", type=Path)
    parser.add_argument("--expected-status", choices=sorted(VALID_STATUSES))
    for count_name in ("discovered", "enqueued", "started", "succeeded", "failed", "cancelled"):
        parser.add_argument(f"--expected-{count_name}", type=int)
    parser.add_argument("--expected-target-architecture")
    parser.add_argument("--expected-runtime-kernel-architecture")
    parser.add_argument("--expected-execution-context")
    parser.add_argument("--expected-requested-workers", type=int)
    parser.add_argument("--expected-effective-workers", type=int)
    parser.add_argument("--expected-input-kind", choices=("directory", "manifest"))
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument("--expect-publishable-memory", action="store_true")
    memory_group.add_argument("--expect-unpublishable-memory", action="store_true")
    parser.add_argument(
        "--no-check-referenced-files",
        action="store_true",
        help="Validate the summary contract only when source/model/item files were intentionally not retained.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    expected_counts = {
        key: value
        for key in ("discovered", "enqueued", "started", "succeeded", "failed", "cancelled")
        if (value := getattr(arguments, f"expected_{key}")) is not None
    }
    expected_memory_publishable: Optional[bool] = None
    if arguments.expect_publishable_memory:
        expected_memory_publishable = True
    elif arguments.expect_unpublishable_memory:
        expected_memory_publishable = False
    try:
        document = load_json(arguments.summary_path)
        validated = validate_document(
            document,
            summary_path=arguments.summary_path,
            expected_status=arguments.expected_status,
            expected_counts=expected_counts,
            expected_target_architecture=arguments.expected_target_architecture,
            expected_runtime_kernel_architecture=(
                arguments.expected_runtime_kernel_architecture
            ),
            expected_execution_context=arguments.expected_execution_context,
            expected_requested_workers=arguments.expected_requested_workers,
            expected_effective_workers=arguments.expected_effective_workers,
            expected_input_kind=arguments.expected_input_kind,
            expected_memory_publishable=expected_memory_publishable,
            check_referenced_files=not arguments.no_check_referenced_files,
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    counts = validated["counts"]
    print(
        "BatchSummary validation: "
        f"status={validated['status']}, discovered={counts['discovered']}, "
        f"succeeded={counts['succeeded']}, failed={counts['failed']}, "
        f"cancelled={counts['cancelled']}, path={arguments.summary_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strictly validate the cross-platform C++ Release benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, NoReturn, Optional, Set, Tuple


EXPECTED_EVIDENCE_TYPE = "cpp_ort_single_image_release_benchmark"
EXPECTED_MODEL_ID = "yolov8n_neu_det_final_train_2"
EXPECTED_MODEL_FAMILY = "yolov8"
EXPECTED_MODEL_SHA256 = (
    "7B8A37610018A6AE6CACDFC869590A95"
    "BBE31AFB7579C39BE0FFEC537196AF68"
)
EXPECTED_MODEL_SIZE_BYTES = 12_336_935
EXPECTED_INPUT_NAME = "images"
EXPECTED_INPUT_SHAPE = [1, 3, 800, 800]
EXPECTED_INPUT_DTYPE = "float32"
EXPECTED_INPUT_LAYOUT = "nchw"
EXPECTED_SCORE_THRESHOLD = 0.25
EXPECTED_NMS_THRESHOLD = 0.45
EXPECTED_NMS_MODE = "class_agnostic"
EXPECTED_DETECTION_COUNT = 3
EXPECTED_SAMPLE_SHA256 = (
    "1D65EF27EAA9BF27608D954DFE57B40E"
    "401FC1AED435884400F35E8000BBF98D"
)
EXPECTED_PROVIDER_EVIDENCE = (
    "explicit_cpu_ep_registration_and_session_creation"
)
LATENCY_SEGMENTS = (
    "image_decode",
    "preprocess",
    "session_run",
    "postprocess",
    "pipeline",
    "end_to_end",
)


def fail(message: str) -> NoReturn:
    raise AssertionError(message)


def reject_constant(value: str) -> NoReturn:
    fail(f"JSON: expected RFC-compliant finite number, actual {value}")


def reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"JSON object: expected unique keys, duplicate {key!r}")
        result[key] = value
    return result


def expect_exact_keys(value: Any, expected: Set[str], object_name: str) -> None:
    if not isinstance(value, dict):
        fail(
            f"{object_name}: expected JSON object, actual "
            f"{type(value).__name__}"
        )
    actual = set(value)
    if actual != expected:
        fail(
            f"{object_name}: expected keys {sorted(expected)}, actual "
            f"{sorted(actual)}; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def expect_string(value: Any, object_name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        fail(f"{object_name}: expected string, actual {value!r}")
    if nonempty and not value:
        fail(f"{object_name}: expected non-empty string, actual empty")
    return value


def expect_int(
    value: Any,
    object_name: str,
    *,
    minimum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        fail(f"{object_name}: expected integer, actual {value!r}")
    if minimum is not None and value < minimum:
        fail(
            f"{object_name}: expected integer >= {minimum}, actual {value}"
        )
    return value


def expect_finite_number(
    value: Any,
    object_name: str,
    *,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if type(value) not in (int, float):
        fail(f"{object_name}: expected JSON number, actual {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        fail(f"{object_name}: expected finite number, actual {value!r}")
    if strictly_positive and converted <= 0.0:
        fail(f"{object_name}: expected positive number, actual {converted}")
    if minimum is not None and converted < minimum:
        fail(
            f"{object_name}: expected number >= {minimum}, actual {converted}"
        )
    return converted


def expect_close(value: Any, expected: float, object_name: str) -> float:
    converted = expect_finite_number(value, object_name)
    if not math.isclose(converted, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        fail(f"{object_name}: expected {expected}, actual {converted}")
    return converted


def expect_string_array(value: Any, object_name: str) -> List[str]:
    if not isinstance(value, list) or not value:
        fail(f"{object_name}: expected non-empty string array, actual {value!r}")
    result: List[str] = []
    for index, item in enumerate(value):
        result.append(expect_string(item, f"{object_name}[{index}]"))
    if len(result) != len(set(result)):
        fail(f"{object_name}: expected unique disclosures, actual duplicates")
    return result


def expect_int_array(value: Any, expected: List[int], object_name: str) -> None:
    if not isinstance(value, list):
        fail(f"{object_name}: expected integer array, actual {value!r}")
    actual = [
        expect_int(item, f"{object_name}[{index}]")
        for index, item in enumerate(value)
    ]
    if actual != expected:
        fail(f"{object_name}: expected {expected}, actual {actual}")


def normalized_path(value: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(value)))


def sha256_file(path_value: str, object_name: str) -> Tuple[Path, int, str]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        if not path.is_file():
            fail(
                f"{object_name}: expected referenced regular file, actual "
                f"{str(path)!r}"
            )
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        fail(f"{object_name}: expected readable file, actual {error}")
    return path, size, digest.hexdigest().upper()


def require_disclosure(
    values: Iterable[str],
    required_terms: Iterable[str],
    object_name: str,
) -> None:
    normalized = [value.casefold() for value in values]
    terms = [term.casefold() for term in required_terms]
    if not any(all(term in value for term in terms) for value in normalized):
        fail(
            f"{object_name}: expected one entry containing {terms}, actual "
            f"{normalized}"
        )


def validate_latency_statistics(
    value: Any,
    object_name: str,
    expected_repeat: int,
) -> float:
    expect_exact_keys(
        value,
        {"sample_count", "mean", "p50", "p95"},
        object_name,
    )
    sample_count = expect_int(
        value["sample_count"], f"{object_name}.sample_count", minimum=1
    )
    if sample_count != expected_repeat:
        fail(
            f"{object_name}.sample_count: expected {expected_repeat}, "
            f"actual {sample_count}"
        )
    mean_ms = expect_finite_number(
        value["mean"], f"{object_name}.mean", minimum=0.0
    )
    p50_ms = expect_finite_number(
        value["p50"], f"{object_name}.p50", minimum=0.0
    )
    p95_ms = expect_finite_number(
        value["p95"], f"{object_name}.p95", minimum=0.0
    )
    if p50_ms > p95_ms:
        fail(
            f"{object_name}: expected p50 <= p95, actual "
            f"p50={p50_ms}, p95={p95_ms}"
        )
    return mean_ms


def validate_document(
    document: Any,
    expected_image: str,
    expected_warmup: int,
    expected_repeat: int,
    expected_model_id: str = EXPECTED_MODEL_ID,
    expected_model_sha256: str = EXPECTED_MODEL_SHA256,
    expected_model_size_bytes: int = EXPECTED_MODEL_SIZE_BYTES,
    expected_detection_count: int = EXPECTED_DETECTION_COUNT,
) -> None:
    expect_exact_keys(
        document,
        {
            "schema_version",
            "evidence_type",
            "timestamp_utc",
            "command",
            "protocol",
            "environment",
            "runtime",
            "model",
            "sample",
            "postprocess",
            "latency_ms",
            "throughput_images_per_second",
            "memory",
            "timing_exclusions",
            "limitations",
        },
        "root",
    )

    if expect_int(document["schema_version"], "schema_version") != 1:
        fail(
            f"schema_version: expected 1, actual "
            f"{document['schema_version']!r}"
        )
    if (
        expect_string(document["evidence_type"], "evidence_type")
        != EXPECTED_EVIDENCE_TYPE
    ):
        fail(
            f"evidence_type: expected {EXPECTED_EVIDENCE_TYPE!r}, actual "
            f"{document['evidence_type']!r}"
        )
    timestamp = expect_string(document["timestamp_utc"], "timestamp_utc")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", timestamp) is None:
        fail(
            "timestamp_utc: expected UTC YYYY-MM-DDTHH:MM:SSZ, actual "
            f"{timestamp!r}"
        )

    command = expect_string_array(document["command"], "command")
    for required_argument in ("--benchmark", "--warmup", "--repeat", "--benchmark-json"):
        if required_argument not in command:
            fail(
                f"command: expected argument {required_argument!r}, actual "
                f"{command!r}"
            )

    protocol = document["protocol"]
    expect_exact_keys(
        protocol,
        {
            "batch_size",
            "sample_count",
            "warmup",
            "repeat",
            "clock",
            "timing_unit",
            "percentile_method",
        },
        "protocol"
    )
    expected_protocol = {
        "batch_size": 1,
        "sample_count": 1,
        "warmup": expected_warmup,
        "repeat": expected_repeat,
    }
    for key, expected in expected_protocol.items():
        actual = expect_int(protocol[key], f"protocol.{key}", minimum=0)
        if actual != expected:
            fail(f"protocol.{key}: expected {expected}, actual {actual}")
    expected_protocol_strings = {
        "clock": "std::chrono::steady_clock",
        "timing_unit": "milliseconds",
        "percentile_method": "empirical_nearest_rank_ceiling",
    }
    for key, expected in expected_protocol_strings.items():
        actual = expect_string(protocol[key], f"protocol.{key}")
        if actual != expected:
            fail(f"protocol.{key}: expected {expected!r}, actual {actual!r}")

    environment = document["environment"]
    expect_exact_keys(
        environment,
        {
            "machine",
            "os",
            "compiler",
            "build",
            "opencv_version",
            "onnxruntime_version",
        },
        "environment",
    )
    machine = environment["machine"]
    expect_exact_keys(
        machine,
        {"hostname", "processor", "architecture", "logical_cpu_count"},
        "environment.machine",
    )
    for key in ("hostname", "processor", "architecture"):
        expect_string(machine[key], f"environment.machine.{key}")
    expect_int(
        machine["logical_cpu_count"],
        "environment.machine.logical_cpu_count",
        minimum=1,
    )
    operating_system = environment["os"]
    expect_exact_keys(operating_system, {"name", "version"}, "environment.os")
    os_name = expect_string(operating_system["name"], "environment.os.name")
    expect_string(operating_system["version"], "environment.os.version")
    compiler = environment["compiler"]
    expect_exact_keys(compiler, {"id", "version"}, "environment.compiler")
    expect_string(compiler["id"], "environment.compiler.id")
    expect_string(compiler["version"], "environment.compiler.version")
    build = environment["build"]
    expect_exact_keys(build, {"type", "cxx_standard"}, "environment.build")
    if expect_string(build["type"], "environment.build.type") != "Release":
        fail(
            f"environment.build.type: expected 'Release', actual "
            f"{build['type']!r}"
        )
    if expect_int(build["cxx_standard"], "environment.build.cxx_standard") != 17:
        fail(
            "environment.build.cxx_standard: expected 17, actual "
            f"{build['cxx_standard']!r}"
        )
    opencv_version = expect_string(
        environment["opencv_version"], "environment.opencv_version"
    )
    if re.fullmatch(r"4\..+", opencv_version) is None:
        fail(
            "environment.opencv_version: expected OpenCV 4.x, actual "
            f"{opencv_version!r}"
        )
    onnxruntime_version = expect_string(
        environment["onnxruntime_version"], "environment.onnxruntime_version"
    )
    if onnxruntime_version != "1.19.2":
        fail(
            "environment.onnxruntime_version: expected '1.19.2', actual "
            f"{onnxruntime_version!r}"
        )

    runtime = document["runtime"]
    expect_exact_keys(
        runtime,
        {
            "requested_provider",
            "actual_provider",
            "provider_evidence",
            "session",
        },
        "runtime",
    )
    if expect_string(runtime["requested_provider"], "runtime.requested_provider") != "cpu":
        fail(
            "runtime.requested_provider: expected 'cpu', actual "
            f"{runtime['requested_provider']!r}"
        )
    if expect_string(runtime["actual_provider"], "runtime.actual_provider") != "CPUExecutionProvider":
        fail(
            "runtime.actual_provider: expected 'CPUExecutionProvider', actual "
            f"{runtime['actual_provider']!r}"
        )
    if (
        expect_string(runtime["provider_evidence"], "runtime.provider_evidence")
        != EXPECTED_PROVIDER_EVIDENCE
    ):
        fail(
            f"runtime.provider_evidence: expected {EXPECTED_PROVIDER_EVIDENCE!r}, "
            f"actual {runtime['provider_evidence']!r}"
        )
    session = runtime["session"]
    expect_exact_keys(
        session,
        {
            "execution_mode",
            "intra_op_num_threads",
            "inter_op_num_threads",
            "graph_optimization_level",
            "initialization_ms",
            "profiling_enabled",
        },
        "runtime.session",
    )
    if expect_string(session["execution_mode"], "runtime.session.execution_mode") != "sequential":
        fail(
            "runtime.session.execution_mode: expected 'sequential', actual "
            f"{session['execution_mode']!r}"
        )
    for key in ("intra_op_num_threads", "inter_op_num_threads"):
        if expect_int(session[key], f"runtime.session.{key}") != 1:
            fail(f"runtime.session.{key}: expected 1, actual {session[key]!r}")
    if (
        expect_string(
            session["graph_optimization_level"],
            "runtime.session.graph_optimization_level",
        )
        != "all"
    ):
        fail(
            "runtime.session.graph_optimization_level: expected 'all', actual "
            f"{session['graph_optimization_level']!r}"
        )
    expect_finite_number(
        session["initialization_ms"],
        "runtime.session.initialization_ms",
        minimum=0.0,
    )
    if session["profiling_enabled"] is not False:
        fail(
            "runtime.session.profiling_enabled: expected false for formal "
            f"benchmark evidence, actual {session['profiling_enabled']!r}"
        )

    model = document["model"]
    expect_exact_keys(
        model,
        {
            "model_id",
            "model_family",
            "path",
            "declared_sha256",
            "file_size_bytes",
            "opset",
            "input",
        },
        "model",
    )
    expected_model_strings = {
        "model_id": expected_model_id,
        "model_family": EXPECTED_MODEL_FAMILY,
        "declared_sha256": expected_model_sha256,
    }
    for key, expected in expected_model_strings.items():
        actual = expect_string(model[key], f"model.{key}")
        if actual != expected:
            fail(f"model.{key}: expected {expected!r}, actual {actual!r}")
    model_path_value = expect_string(model["path"], "model.path")
    model_path, actual_model_size, actual_model_sha256 = sha256_file(
        model_path_value, "model.path"
    )
    if actual_model_sha256 != expected_model_sha256:
        fail(
            f"model.path SHA-256: expected {expected_model_sha256}, actual "
            f"{actual_model_sha256} for {str(model_path)!r}"
        )
    if model["declared_sha256"] != actual_model_sha256:
        fail(
            "model.declared_sha256: expected the actual model.path SHA-256 "
            f"{actual_model_sha256}, actual {model['declared_sha256']!r}"
        )
    recorded_model_size = expect_int(
        model["file_size_bytes"], "model.file_size_bytes", minimum=1
    )
    if recorded_model_size != expected_model_size_bytes:
        fail(
            f"model.file_size_bytes: expected {expected_model_size_bytes}, "
            f"actual {recorded_model_size}"
        )
    if recorded_model_size != actual_model_size:
        fail(
            f"model.file_size_bytes: expected actual file size "
            f"{actual_model_size}, actual {recorded_model_size}"
        )
    if expect_int(model["opset"], "model.opset", minimum=1) != 17:
        fail(f"model.opset: expected 17, actual {model['opset']!r}")
    model_input = model["input"]
    expect_exact_keys(
        model_input, {"name", "shape", "dtype", "layout"}, "model.input"
    )
    if expect_string(model_input["name"], "model.input.name") != EXPECTED_INPUT_NAME:
        fail(
            f"model.input.name: expected {EXPECTED_INPUT_NAME!r}, actual "
            f"{model_input['name']!r}"
        )
    expect_int_array(
        model_input["shape"], EXPECTED_INPUT_SHAPE, "model.input.shape"
    )
    if expect_string(model_input["dtype"], "model.input.dtype") != EXPECTED_INPUT_DTYPE:
        fail(
            f"model.input.dtype: expected {EXPECTED_INPUT_DTYPE!r}, actual "
            f"{model_input['dtype']!r}"
        )
    if expect_string(model_input["layout"], "model.input.layout") != EXPECTED_INPUT_LAYOUT:
        fail(
            f"model.input.layout: expected {EXPECTED_INPUT_LAYOUT!r}, actual "
            f"{model_input['layout']!r}"
        )

    sample = document["sample"]
    expect_exact_keys(
        sample,
        {
            "image_path",
            "file_size_bytes",
            "original_shape",
            "sample_count",
        },
        "sample",
    )
    image_path = expect_string(sample["image_path"], "sample.image_path")
    if normalized_path(image_path) != normalized_path(expected_image):
        fail(
            f"sample.image_path: expected {expected_image!r}, actual "
            f"{image_path!r}"
        )
    resolved_image, actual_image_size, actual_image_sha256 = sha256_file(
        image_path, "sample.image_path"
    )
    if actual_image_sha256 != EXPECTED_SAMPLE_SHA256:
        fail(
            f"sample.image_path SHA-256: expected {EXPECTED_SAMPLE_SHA256}, "
            f"actual {actual_image_sha256} for {str(resolved_image)!r}"
        )
    recorded_image_size = expect_int(
        sample["file_size_bytes"], "sample.file_size_bytes", minimum=1
    )
    if recorded_image_size != actual_image_size:
        fail(
            f"sample.file_size_bytes: expected actual file size "
            f"{actual_image_size}, actual {recorded_image_size}"
        )
    expect_int_array(
        sample["original_shape"], [200, 200, 3], "sample.original_shape"
    )
    if expect_int(sample["sample_count"], "sample.sample_count") != 1:
        fail(
            f"sample.sample_count: expected 1, actual "
            f"{sample['sample_count']!r}"
        )

    postprocess = document["postprocess"]
    expect_exact_keys(
        postprocess,
        {"score_threshold", "nms_threshold", "nms_mode", "detection_count"},
        "postprocess",
    )
    expect_close(
        postprocess["score_threshold"],
        EXPECTED_SCORE_THRESHOLD,
        "postprocess.score_threshold",
    )
    expect_close(
        postprocess["nms_threshold"],
        EXPECTED_NMS_THRESHOLD,
        "postprocess.nms_threshold",
    )
    if expect_string(postprocess["nms_mode"], "postprocess.nms_mode") != EXPECTED_NMS_MODE:
        fail(
            f"postprocess.nms_mode: expected {EXPECTED_NMS_MODE!r}, actual "
            f"{postprocess['nms_mode']!r}"
        )
    if (
        expect_int(
            postprocess["detection_count"],
            "postprocess.detection_count",
            minimum=0,
        )
        != expected_detection_count
    ):
        fail(
            f"postprocess.detection_count: expected {expected_detection_count}, "
            f"actual {postprocess['detection_count']!r}"
        )

    latency = document["latency_ms"]
    expect_exact_keys(latency, set(LATENCY_SEGMENTS), "latency_ms")
    means = {
        segment: validate_latency_statistics(
            latency[segment], f"latency_ms.{segment}", expected_repeat
        )
        for segment in LATENCY_SEGMENTS
    }
    if means["pipeline"] <= 0.0 or means["end_to_end"] <= 0.0:
        fail(
            "latency_ms: expected positive pipeline/end_to_end means for "
            f"throughput, actual pipeline={means['pipeline']}, "
            f"end_to_end={means['end_to_end']}"
        )

    throughput = document["throughput_images_per_second"]
    expect_exact_keys(throughput, {"pipeline", "end_to_end"}, "throughput")
    for key in ("pipeline", "end_to_end"):
        actual = expect_finite_number(
            throughput[key],
            f"throughput_images_per_second.{key}",
            strictly_positive=True,
        )
        expected = 1000.0 / means[key]
        if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-10):
            fail(
                f"throughput_images_per_second.{key}: expected "
                f"1000/mean_ms={expected}, actual {actual}"
            )

    memory = document["memory"]
    expect_exact_keys(
        memory,
        {"status", "metric", "bytes", "mebibytes", "scope", "reason"},
        "memory",
    )
    status = expect_string(memory["status"], "memory.status")
    metric = expect_string(memory["metric"], "memory.metric")
    scope = expect_string(memory["scope"], "memory.scope")
    expected_memory_metric = {
        "Windows": "peak_working_set",
        "Linux": "peak_rss",
    }.get(os_name)
    if expected_memory_metric is not None:
        if status != "supported" or metric != expected_memory_metric:
            fail(
                f"memory: expected supported {os_name} "
                f"{expected_memory_metric}, actual "
                f"status={status!r}, metric={metric!r}"
            )
        memory_bytes = expect_int(memory["bytes"], "memory.bytes", minimum=1)
        memory_mib = expect_finite_number(
            memory["mebibytes"], "memory.mebibytes", strictly_positive=True
        )
        if not math.isclose(
            memory_mib,
            memory_bytes / (1024.0 * 1024.0),
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        ):
            fail(
                "memory.mebibytes: expected bytes/(1024*1024), actual "
                f"bytes={memory_bytes}, mebibytes={memory_mib}"
            )
        if memory["reason"] is not None:
            fail(
                f"memory.reason: expected null on supported {os_name}, actual "
                f"{memory['reason']!r}"
            )
    else:
        if status != "unsupported":
            fail(
                "memory: expected explicit unsupported status outside "
                "Windows/Linux, "
                f"actual status={status!r}"
            )
        if memory["bytes"] is not None or memory["mebibytes"] is not None:
            fail(
                "memory: expected null bytes/mebibytes when unsupported, actual "
                f"bytes={memory['bytes']!r}, mebibytes={memory['mebibytes']!r}"
            )
        reason = expect_string(memory["reason"], "memory.reason")
        if not reason:
            fail("memory.reason: expected non-empty unsupported reason")
    require_disclosure([scope], ("process lifetime",), "memory.scope")

    exclusions = expect_string_array(
        document["timing_exclusions"], "timing_exclusions"
    )
    require_disclosure(exclusions, ("session", "initial"), "timing_exclusions")
    require_disclosure(exclusions, ("json", "write"), "timing_exclusions")
    require_disclosure(exclusions, ("visualization",), "timing_exclusions")

    limitations = expect_string_array(document["limitations"], "limitations")
    require_disclosure(limitations, ("one", "image"), "limitations")
    require_disclosure(limitations, ("file cache",), "limitations")
    require_disclosure(
        limitations, ("process-lifetime", "peak", "memory"), "limitations"
    )
    require_disclosure(limitations, ("per-node",), "limitations")
    require_disclosure(limitations, ("historical python ort",), "limitations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--expected-warmup", type=int, default=1)
    parser.add_argument("--expected-repeat", type=int, default=2)
    parser.add_argument("--expected-model-id", default=EXPECTED_MODEL_ID)
    parser.add_argument("--expected-model-sha256", default=EXPECTED_MODEL_SHA256)
    parser.add_argument(
        "--expected-model-size-bytes", type=int, default=EXPECTED_MODEL_SIZE_BYTES
    )
    parser.add_argument(
        "--expected-detection-count", type=int, default=EXPECTED_DETECTION_COUNT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_warmup < 0:
        fail(
            f"--expected-warmup: expected non-negative, actual "
            f"{args.expected_warmup}"
        )
    if args.expected_repeat <= 0:
        fail(
            f"--expected-repeat: expected positive, actual "
            f"{args.expected_repeat}"
        )
    if (
        len(args.expected_model_sha256) != 64
        or any(character not in "0123456789ABCDEF" for character in args.expected_model_sha256)
    ):
        fail(
            "--expected-model-sha256: expected 64 uppercase hexadecimal characters, "
            f"actual {args.expected_model_sha256!r}"
        )
    if args.expected_model_size_bytes <= 0:
        fail(
            "--expected-model-size-bytes: expected positive, "
            f"actual {args.expected_model_size_bytes}"
        )
    if args.expected_detection_count < 0:
        fail(
            "--expected-detection-count: expected non-negative, "
            f"actual {args.expected_detection_count}"
        )
    try:
        encoded = args.json_path.read_bytes()
    except OSError as error:
        fail(f"benchmark JSON: expected readable file, actual {error}")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        fail(f"benchmark JSON: expected UTF-8, actual decode failure: {error}")
    try:
        document = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        fail(f"benchmark JSON: expected valid JSON, actual parse failure: {error}")

    validate_document(
        document,
        args.expected_image,
        args.expected_warmup,
        args.expected_repeat,
        args.expected_model_id,
        args.expected_model_sha256,
        args.expected_model_size_bytes,
        args.expected_detection_count,
    )
    print(
        "Cross-platform benchmark JSON passed strict schema, Release/CPU protocol, "
        "six-segment finite statistics, throughput, memory, and disclosure "
        "validation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

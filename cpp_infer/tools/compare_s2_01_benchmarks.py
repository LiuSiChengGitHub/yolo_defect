#!/usr/bin/env python3
"""Compare unprofiled FP32/INT8 C++ benchmarks under the frozen S2-01 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import s2_01_protocol  # noqa: E402


SCHEMA_VERSION = 1
LATENCY_SEGMENTS = (
    "image_decode",
    "preprocess",
    "session_run",
    "postprocess",
    "pipeline",
    "end_to_end",
)


class BenchmarkComparisonError(RuntimeError):
    """Raised when two benchmark documents are not safely comparable."""


def fail(object_name: str, expected: str, actual: str, action: str) -> NoReturn:
    raise BenchmarkComparisonError(
        "S2-01 benchmark comparison failed: "
        f"object={object_name}; expected={expected}; actual={actual}; action={action}"
    )


def _reject_duplicate_keys(pairs):
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("json", "unique object keys", f"duplicate {key!r}", "regenerate the evidence")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: fail(
                    "json.number", "a finite RFC-compliant number", token, "regenerate the evidence"
                ),
            )
    except BenchmarkComparisonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail("json.path", "readable UTF-8 JSON", f"{path}: {error}", "pass a generated evidence file")


def _mapping(value: Any, object_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        fail(object_name, "a JSON object", type(value).__name__, "regenerate the benchmark")
    return value


def _string(value: Any, object_name: str) -> str:
    if not isinstance(value, str) or not value:
        fail(object_name, "a non-empty string", repr(value), "regenerate the benchmark")
    return value


def _integer(value: Any, object_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(object_name, f"an integer >= {minimum}", repr(value), "regenerate the benchmark")
    return value


def _number(value: Any, object_name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(object_name, f"a finite number >= {minimum}", repr(value), "regenerate the benchmark")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        fail(object_name, f"a finite number >= {minimum}", repr(value), "regenerate the benchmark")
    return converted


def _sha256_file(path: Path) -> Tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        fail("model.path", "a readable model file", f"{path}: {error}", "restore the selected artifact")
    return size, digest.hexdigest().upper()


def _resolve_recorded_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _validate_statistics(value: Any, object_name: str, expected_repeat: int) -> Mapping[str, float]:
    statistics = _mapping(value, object_name)
    if set(statistics) != {"sample_count", "mean", "p50", "p95"}:
        fail(
            object_name,
            "sample_count/mean/p50/p95 only",
            repr(sorted(statistics)),
            "use the C++ benchmark evidence schema",
        )
    if _integer(statistics["sample_count"], f"{object_name}.sample_count", 1) != expected_repeat:
        fail(object_name, f"{expected_repeat} samples", repr(statistics["sample_count"]), "rerun the frozen repeat count")
    result = {
        key: _number(statistics[key], f"{object_name}.{key}")
        for key in ("mean", "p50", "p95")
    }
    if result["p50"] > result["p95"]:
        fail(object_name, "p50 <= p95", repr(result), "inspect percentile calculation")
    return result


def validate_benchmark(
    document: Any,
    precision: str,
    *,
    validate_referenced_model: bool,
) -> Mapping[str, Any]:
    root = _mapping(document, precision)
    if _integer(root.get("schema_version"), f"{precision}.schema_version") != 1:
        fail(f"{precision}.schema_version", "1", repr(root.get("schema_version")), "use the current C++ writer")
    if root.get("evidence_type") != "cpp_ort_single_image_release_benchmark":
        fail(f"{precision}.evidence_type", "cpp_ort_single_image_release_benchmark", repr(root.get("evidence_type")), "do not compare historical Python timing")

    protocol = _mapping(root.get("protocol"), f"{precision}.protocol")
    repeat = _integer(protocol.get("repeat"), f"{precision}.protocol.repeat", 1)
    warmup = _integer(protocol.get("warmup"), f"{precision}.protocol.warmup")
    if warmup != 10 or repeat != 100:
        fail(
            f"{precision}.protocol.warmup_repeat",
            "warmup=10 and repeat=100",
            f"warmup={warmup}, repeat={repeat}",
            "rerun the frozen formal benchmark rather than a smoke benchmark",
        )
    if _integer(protocol.get("batch_size"), f"{precision}.protocol.batch_size", 1) != 1:
        fail(f"{precision}.protocol.batch_size", "1", repr(protocol.get("batch_size")), "use batch one")
    if _integer(protocol.get("sample_count"), f"{precision}.protocol.sample_count", 1) != 1:
        fail(f"{precision}.protocol.sample_count", "1", repr(protocol.get("sample_count")), "use the fixed image")

    environment = _mapping(root.get("environment"), f"{precision}.environment")
    build = _mapping(environment.get("build"), f"{precision}.environment.build")
    if build.get("type") != "Release" or build.get("cxx_standard") != 17:
        fail(f"{precision}.environment.build", "Release C++17", repr(build), "rerun the clean Release build")

    runtime = _mapping(root.get("runtime"), f"{precision}.runtime")
    if runtime.get("requested_provider") != "cpu" or runtime.get("actual_provider") != "CPUExecutionProvider":
        fail(f"{precision}.runtime.provider", "cpu / CPUExecutionProvider", repr(runtime), "restore the CPU protocol")
    session = _mapping(runtime.get("session"), f"{precision}.runtime.session")
    expected_session_policy = {
        "execution_mode": "sequential",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "graph_optimization_level": "all",
    }
    for field, expected in expected_session_policy.items():
        if session.get(field) != expected:
            fail(
                f"{precision}.runtime.session.{field}",
                repr(expected),
                repr(session.get(field)),
                "restore the frozen CPU session policy",
            )
    if session.get("profiling_enabled") is not False:
        fail(f"{precision}.runtime.session.profiling_enabled", "false", repr(session.get("profiling_enabled")), "run benchmark with a fresh unprofiled session")
    initialization_ms = _number(session.get("initialization_ms"), f"{precision}.runtime.session.initialization_ms")

    model = _mapping(root.get("model"), f"{precision}.model")
    declared_sha = _string(model.get("declared_sha256"), f"{precision}.model.declared_sha256").upper()
    if len(declared_sha) != 64 or any(character not in "0123456789ABCDEF" for character in declared_sha):
        fail(f"{precision}.model.declared_sha256", "64 uppercase hexadecimal characters", declared_sha, "fix the artifact contract")
    recorded_size = _integer(model.get("file_size_bytes"), f"{precision}.model.file_size_bytes", 1)
    if validate_referenced_model:
        model_path = _resolve_recorded_path(_string(model.get("path"), f"{precision}.model.path"))
        if not model_path.is_file():
            fail(f"{precision}.model.path", "an existing regular file", str(model_path), "restore or regenerate the artifact")
        actual_size, actual_sha = _sha256_file(model_path)
        if actual_size != recorded_size or actual_sha != declared_sha:
            fail(
                f"{precision}.model.integrity",
                f"size={recorded_size}, sha256={declared_sha}",
                f"size={actual_size}, sha256={actual_sha}",
                "regenerate the benchmark only after validating the artifact",
            )

    latency_root = _mapping(root.get("latency_ms"), f"{precision}.latency_ms")
    latency = {
        segment: _validate_statistics(
            latency_root.get(segment), f"{precision}.latency_ms.{segment}", repeat
        )
        for segment in LATENCY_SEGMENTS
    }
    throughput = _mapping(root.get("throughput_images_per_second"), f"{precision}.throughput")
    pipeline_throughput = _number(throughput.get("pipeline"), f"{precision}.throughput.pipeline")
    end_to_end_throughput = _number(throughput.get("end_to_end"), f"{precision}.throughput.end_to_end")
    memory = _mapping(root.get("memory"), f"{precision}.memory")
    if memory.get("status") != "supported" or memory.get("metric") != "peak_working_set":
        fail(f"{precision}.memory", "supported peak_working_set", repr(memory), "run on the Windows evidence platform")
    memory_bytes = _integer(memory.get("bytes"), f"{precision}.memory.bytes", 1)

    return {
        "document": root,
        "protocol": protocol,
        "environment": environment,
        "runtime": runtime,
        "session": session,
        "model": model,
        "sample": _mapping(root.get("sample"), f"{precision}.sample"),
        "postprocess": _mapping(root.get("postprocess"), f"{precision}.postprocess"),
        "initialization_ms": initialization_ms,
        "latency": latency,
        "pipeline_throughput": pipeline_throughput,
        "end_to_end_throughput": end_to_end_throughput,
        "memory_bytes": memory_bytes,
        "model_size_bytes": recorded_size,
        "model_sha256": declared_sha,
    }


def _without(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _require_equal(name: str, lhs: Any, rhs: Any) -> None:
    if lhs != rhs:
        fail(name, "identical FP32 and INT8 protocol values", f"fp32={lhs!r}, int8={rhs!r}", "rerun both models under one frozen command protocol")


def _metric_comparison(fp32_value: float, int8_value: float, *, lower_is_better: bool) -> Mapping[str, Any]:
    delta = int8_value - fp32_value
    ratio = None if int8_value == 0.0 else fp32_value / int8_value
    if math.isclose(fp32_value, int8_value, rel_tol=1.0e-12, abs_tol=1.0e-12):
        direction = "equal"
    elif (int8_value < fp32_value) == lower_is_better:
        direction = "int8_better"
    else:
        direction = "int8_worse"
    return {
        "fp32": fp32_value,
        "int8": int8_value,
        "int8_minus_fp32": delta,
        "fp32_div_int8": ratio,
        "int8_div_fp32": None if fp32_value == 0.0 else int8_value / fp32_value,
        "direction": direction,
    }


def compare_documents(
    fp32_document: Any,
    int8_document: Any,
    correctness_document: Any,
    protocol_binding: Mapping[str, Any],
    *,
    validate_referenced_models: bool = True,
    correctness_policy: str = "required",
) -> Mapping[str, Any]:
    correctness = _mapping(correctness_document, "correctness")
    binding = _mapping(protocol_binding, "protocol_binding")
    if correctness_policy not in ("required", "advisory"):
        fail(
            "correctness_policy",
            "'required' or 'advisory'",
            repr(correctness_policy),
            "select an explicit publication policy",
        )
    correctness_passed = correctness.get("passed") is True
    if not correctness_passed and correctness_policy == "required":
        fail("correctness.passed", "true from the same S2-01 run", repr(correctness.get("passed")), "fix correctness before publishing performance")
    if correctness.get("evidence_type") != "s2_01_fp32_int8_correctness_and_quality":
        fail(
            "correctness.evidence_type",
            "s2_01_fp32_int8_correctness_and_quality",
            repr(correctness.get("evidence_type")),
            "pass the formal three-layer S2-01 correctness result",
        )
    runtime_legality = _mapping(correctness.get("runtime_legality"), "correctness.runtime_legality")
    cpp_legality = _mapping(runtime_legality.get("cpp"), "correctness.runtime_legality.cpp")
    if (
        runtime_legality.get("python_fp32_session_and_finite_outputs") is not True
        or runtime_legality.get("python_int8_session_and_finite_outputs") is not True
        or cpp_legality.get("requested") is not True
        or cpp_legality.get("passed") is not True
    ):
        fail(
            "correctness.runtime_legality",
            "Python FP32/INT8 and requested C++ legality all passed",
            repr(runtime_legality),
            "complete all three runtime-legality gates before benchmarking",
        )

    fp32 = validate_benchmark(fp32_document, "fp32", validate_referenced_model=validate_referenced_models)
    int8 = validate_benchmark(int8_document, "int8", validate_referenced_model=validate_referenced_models)
    _require_equal("protocol", fp32["protocol"], int8["protocol"])
    _require_equal("environment", fp32["environment"], int8["environment"])
    _require_equal("runtime", _without(fp32["runtime"], "session"), _without(int8["runtime"], "session"))
    _require_equal(
        "runtime.session_policy",
        _without(fp32["session"], "initialization_ms"),
        _without(int8["session"], "initialization_ms"),
    )
    _require_equal("sample", fp32["sample"], int8["sample"])
    _require_equal(
        "postprocess",
        _without(fp32["postprocess"], "detection_count"),
        _without(int8["postprocess"], "detection_count"),
    )
    for field in ("model_family", "opset", "input"):
        _require_equal(f"model.{field}", fp32["model"].get(field), int8["model"].get(field))
    if fp32["model_sha256"] == int8["model_sha256"]:
        fail("model.sha256", "different FP32 source and INT8 derived digests", fp32["model_sha256"], "select the two distinct artifacts")
    expected_binding = {
        "source_model_sha256": fp32["model_sha256"],
        "derived_model_sha256": int8["model_sha256"],
        "warmup": 10,
        "repeat": 100,
    }
    for field, expected in expected_binding.items():
        if binding.get(field) != expected:
            fail(
                f"protocol_binding.{field}",
                repr(expected),
                repr(binding.get(field)),
                "bind both benchmarks to the validated frozen machine protocol",
            )
    correctness_protocol = _mapping(
        correctness.get("protocol"), "correctness.protocol"
    )
    for field in ("protocol_id", "canonical_lf_sha256"):
        if correctness_protocol.get(field) != binding.get(field):
            fail(
                f"correctness.protocol.{field}",
                repr(binding.get(field)),
                repr(correctness_protocol.get(field)),
                "use correctness evidence produced from the same frozen protocol",
            )
    correctness_artifacts = _mapping(
        correctness.get("artifacts"), "correctness.artifacts"
    )
    for precision, expected_sha in (
        ("fp32", fp32["model_sha256"]),
        ("int8", int8["model_sha256"]),
    ):
        artifact = _mapping(
            correctness_artifacts.get(precision),
            f"correctness.artifacts.{precision}",
        )
        if artifact.get("model_sha256") != expected_sha:
            fail(
                f"correctness.artifacts.{precision}.model_sha256",
                expected_sha,
                repr(artifact.get("model_sha256")),
                "benchmark the exact artifacts that passed correctness",
            )

    latency: MutableMapping[str, Any] = {}
    for segment in LATENCY_SEGMENTS:
        latency[segment] = {
            statistic: _metric_comparison(
                fp32["latency"][segment][statistic],
                int8["latency"][segment][statistic],
                lower_is_better=True,
            )
            for statistic in ("mean", "p50", "p95")
        }

    size_fp32 = fp32["model_size_bytes"]
    size_int8 = int8["model_size_bytes"]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "s2_01_fp32_int8_cpp_benchmark_comparison",
        "passed": True,
        "correctness_prerequisite": {
            "policy": correctness_policy,
            "passed": correctness_passed,
            "blocking": correctness_policy == "required",
            "accepted_for_comparison": correctness_passed or correctness_policy == "advisory",
            "evidence_type": correctness.get("evidence_type"),
        },
        "protocol_binding": dict(binding),
        "comparability": {
            "same_machine_environment": True,
            "same_release_build": True,
            "same_provider_and_threads": True,
            "same_sample_and_postprocess": True,
            "same_warmup_repeat": True,
            "profiling_disabled": True,
            "separate_process_requirement": "Each source JSON must be produced by a distinct CLI process so process-lifetime peak memory is not inherited.",
        },
        "models": {
            "fp32": {
                "model_id": fp32["model"].get("model_id"),
                "sha256": fp32["model_sha256"],
                "file_size_bytes": size_fp32,
            },
            "int8": {
                "model_id": int8["model"].get("model_id"),
                "sha256": int8["model_sha256"],
                "file_size_bytes": size_int8,
            },
            "size": {
                "int8_minus_fp32_bytes": size_int8 - size_fp32,
                "int8_div_fp32": size_int8 / size_fp32,
                "reduction_fraction": 1.0 - (size_int8 / size_fp32),
            },
        },
        "session_initialization_ms": _metric_comparison(
            fp32["initialization_ms"], int8["initialization_ms"], lower_is_better=True
        ),
        "latency_ms": latency,
        "throughput_images_per_second": {
            "pipeline": _metric_comparison(
                fp32["pipeline_throughput"], int8["pipeline_throughput"], lower_is_better=False
            ),
            "end_to_end": _metric_comparison(
                fp32["end_to_end_throughput"], int8["end_to_end_throughput"], lower_is_better=False
            ),
        },
        "peak_working_set_bytes": _metric_comparison(
            float(fp32["memory_bytes"]), float(int8["memory_bytes"]), lower_is_better=True
        ),
        "interpretation": {
            "speed_is_not_a_pass_condition": True,
            "pipeline_mean_outcome": latency["pipeline"]["mean"]["direction"],
            "session_run_mean_outcome": latency["session_run"]["mean"]["direction"],
        },
        "limitations": [
            "Peak Working Set is a process-lifetime high-water mark, not incremental model memory.",
            "One fixed image and warm operating-system caches do not represent every workload.",
            "Initialization has one observation per process; no P50/P95 is claimed for it.",
            "No CPU affinity, priority elevation, or idle-system lock is applied.",
            "A slower INT8 result remains valid evidence and does not fail this comparison.",
            *(
                [
                    "Correctness and quality are advisory for this exercise run; their failed gates remain visible and are not rewritten as passed."
                ]
                if correctness_policy == "advisory" and not correctness_passed
                else []
            ),
        ],
        "runtime": {"python_version": platform.python_version(), "platform": platform.platform()},
    }


def write_json(path: Path, value: Mapping[str, Any], overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        fail("output.path", "a path that does not exist", f"{path} exists", "choose a fresh path or pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", required=True, type=Path)
    parser.add_argument("--int8", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--correctness-policy",
        choices=("required", "advisory"),
        default="required",
        help="Keep the frozen correctness result blocking (default) or bind it as non-blocking exercise context.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        frozen_protocol = s2_01_protocol.load_s2_01_protocol(arguments.protocol)
        derived_size, derived_sha = _sha256_file(frozen_protocol.output_model_path)
        protocol_binding = {
            "protocol_id": frozen_protocol.protocol_id,
            "path": str(frozen_protocol.declaration_path),
            "canonical_lf_sha256": s2_01_protocol.sha256_file_canonical_lf(
                frozen_protocol.declaration_path
            ),
            "source_model_sha256": frozen_protocol.source_model_sha256,
            "derived_model_sha256": derived_sha,
            "derived_model_size_bytes": derived_size,
            "warmup": frozen_protocol.benchmark["warmup"],
            "repeat": frozen_protocol.benchmark["repeat"],
        }
        comparison = compare_documents(
            load_json(arguments.fp32),
            load_json(arguments.int8),
            load_json(arguments.correctness),
            protocol_binding,
            correctness_policy=arguments.correctness_policy,
        )
        write_json(arguments.output, comparison, arguments.overwrite)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    pipeline = comparison["latency_ms"]["pipeline"]["mean"]
    print(
        "S2-01 benchmark comparison: "
        f"passed=True, fp32={pipeline['fp32']:.6f} ms, "
        f"int8={pipeline['int8']:.6f} ms, outcome={pipeline['direction']}, "
        f"output={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

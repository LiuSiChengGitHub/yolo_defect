#!/usr/bin/env python3
"""Assemble cross-bound S2-01 acceptance or exercise-completion evidence.

This command does not run inference, profiling, or quantization.  It validates
the already-produced machine evidence, proves that every document refers to
the same frozen protocol and model lineage, verifies the raw profile traces,
and emits one compact record.  Strict correctness remains the default.  An
explicit advisory policy preserves failed product/quality gates while allowing
the user-approved PTQ/profiling exercise deliverables to close.  INT8 speed is
reported but is never used as an acceptance gate.
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
from typing import Any, Dict, Iterable, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import s2_01_protocol  # noqa: E402


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "s2_01_acceptance"
ADVISORY_EVIDENCE_TYPE = "s2_01_exercise_completion"
FROZEN_PROTOCOL_ID = "s2_01_static_ptq_qdq_s8s8_cpu_v1"
SUPPORTED_PROTOCOL_IDS = frozenset(
    {
        FROZEN_PROTOCOL_ID,
        "s2_01_static_ptq_qdq_s8s8_head_fp32_cpu_v2",
        "s2_01_static_ptq_qdq_s8s8_head_fp32_entropy_cpu_v3",
        "s2_01_static_ptq_qdq_s8s8_backbone_only_cpu_v4",
        "s2_01_static_ptq_qdq_s8s8_early_backbone_cpu_v5",
        "s2_01_static_ptq_qdq_s8s8_late_backbone_cpu_v6",
        "s2_01_static_ptq_qdq_s8s8_deep_backbone_cpu_v8",
        "s2_01_static_ptq_qdq_s8s8_mid_backbone_cpu_v7",
        "s2_01_static_ptq_qdq_s8s8_prefix_model0_2_cpu_v9",
        "s2_01_static_ptq_qdq_s8s8_prefix_model0_1_cpu_v10",
        "s2_01_static_ptq_qdq_s8s8_prefix_model0_cpu_v11",
    }
)
SUPPORTED_PROTOCOL_SHA256 = {
    FROZEN_PROTOCOL_ID: "0EC9A7B1CF5E4F246CF3AC15275EF06D7C67FB6C0CE11C5218391CFACE5B73F2",
    "s2_01_static_ptq_qdq_s8s8_head_fp32_cpu_v2": "D083F182E24290DFBA7864A2C840803DDE82AEC49EAE6405F1C63EB1B2C22068",
    "s2_01_static_ptq_qdq_s8s8_head_fp32_entropy_cpu_v3": "4CFAB466786FF02A79F9F43020B5B5C06FA2B495D4FDE3AC560A624B17BD4DF3",
    "s2_01_static_ptq_qdq_s8s8_backbone_only_cpu_v4": "EE1FE1998DD20404497E24613F449660F9EF91F3CA2DEBB5E2CCDD7801761935",
    "s2_01_static_ptq_qdq_s8s8_early_backbone_cpu_v5": "C9441D27762F46736A2D8F87A4226C2108834E6C3F7301E5A8AD9C2B61754D30",
    "s2_01_static_ptq_qdq_s8s8_late_backbone_cpu_v6": "60600A9BA8C6262C8CC5439F6051F82848EA70DE4BB53223B2EEA0C842E46615",
    "s2_01_static_ptq_qdq_s8s8_deep_backbone_cpu_v8": "B5FB4969BD356D31ED62FD6D0E61F22136860E8A82B61AB2F83964F80EC700FD",
    "s2_01_static_ptq_qdq_s8s8_mid_backbone_cpu_v7": "7CEA43AB52C030AFA47DC3C733A0F9D96055BAAD30B0296C93ECE06E9529B4DD",
    "s2_01_static_ptq_qdq_s8s8_prefix_model0_2_cpu_v9": "AB31FEEAE6E46B9D82544028AC7D596DAFB1AA125C90C34A8DC031865DD02D4B",
    "s2_01_static_ptq_qdq_s8s8_prefix_model0_1_cpu_v10": "18C66D0F163EFE6DEF58ED12CC927F5351BBDAF862795BAA0FB037295B9F082C",
    "s2_01_static_ptq_qdq_s8s8_prefix_model0_cpu_v11": "C4E9B351E291791E2A893E8044001821AE1918D1D321A256B9D03E30D5408FB2",
}
SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")
LATENCY_SEGMENTS = (
    "image_decode",
    "preprocess",
    "session_run",
    "postprocess",
    "pipeline",
    "end_to_end",
)
PATH_BINDING_KEYS = (
    "quant_source_model",
    "quant_derived_model",
    "correctness_fp32_model",
    "correctness_int8_model",
    "fp32_benchmark_model",
    "int8_benchmark_model",
    "fp32_benchmark_sample",
    "int8_benchmark_sample",
    "fp32_profile_model",
    "int8_profile_model",
    "fp32_profile_sample",
    "int8_profile_sample",
)


class EvidenceAssemblyError(RuntimeError):
    """The supplied evidence cannot prove S2-01 acceptance."""


def fail(object_name: str, expected: str, actual: str, action: str) -> NoReturn:
    raise EvidenceAssemblyError(
        "S2-01 evidence assembly failed: "
        f"object={object_name}; expected={expected}; actual={actual}; action={action}"
    )


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> MutableMapping[str, Any]:
    result: MutableMapping[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(
                "json",
                "unique object keys",
                f"duplicate key {key!r}",
                "regenerate the evidence document",
            )
        result[key] = value
    return result


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: fail(
                    "json.number",
                    "a finite RFC-compliant number",
                    token,
                    "regenerate the evidence without NaN or Infinity",
                ),
            )
    except EvidenceAssemblyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(
            f"json.path={path}",
            "one readable UTF-8 JSON object",
            str(error),
            "pass a current machine-generated evidence file",
        )
    if not isinstance(value, dict):
        fail(
            f"json.path={path}",
            "a JSON object at the root",
            type(value).__name__,
            "pass an evidence summary rather than a raw trace",
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail("file.sha256", "a readable regular file", f"{path}: {error}", "restore the evidence input")
    return digest.hexdigest().upper()


def canonical_lf_sha256(path: Path) -> str:
    try:
        raw = path.read_bytes()
        canonical = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except (OSError, UnicodeError) as error:
        fail(
            "file.canonical_lf_sha256",
            "readable UTF-8 text",
            f"{path}: {error}",
            "restore the protocol or manifest",
        )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def _mapping(value: Any, object_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        fail(object_name, "a JSON object", type(value).__name__, "regenerate the evidence")
    return value


def _sequence(value: Any, object_name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        fail(object_name, "a JSON array", type(value).__name__, "regenerate the evidence")
    return value


def _string(value: Any, object_name: str) -> str:
    if not isinstance(value, str) or not value:
        fail(object_name, "a non-empty string", repr(value), "regenerate the evidence")
    return value


def _integer(value: Any, object_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(object_name, f"an integer >= {minimum}", repr(value), "regenerate the evidence")
    return value


def _number(value: Any, object_name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(object_name, f"a finite number >= {minimum}", repr(value), "regenerate the evidence")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        fail(object_name, f"a finite number >= {minimum}", repr(value), "regenerate the evidence")
    return converted


def _sha256(value: Any, object_name: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        fail(object_name, "64 hexadecimal characters", repr(value), "regenerate the evidence")
    return value.upper()


def _require_true(value: Any, object_name: str, action: str = "rerun the failed gate") -> None:
    if value is not True:
        fail(object_name, "true", repr(value), action)


def _require_equal(actual: Any, expected: Any, object_name: str, action: str) -> None:
    if actual != expected:
        fail(object_name, repr(expected), repr(actual), action)


def _schema_and_type(document: Any, evidence_type: str, object_name: str) -> Mapping[str, Any]:
    root = _mapping(document, object_name)
    _require_equal(root.get("schema_version"), SCHEMA_VERSION, f"{object_name}.schema_version", "use the current schema")
    _require_equal(root.get("evidence_type"), evidence_type, f"{object_name}.evidence_type", "pass the requested evidence file")
    return root


def _without(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    return {key: item for key, item in value.items() if key not in keys}


def _expected_metric_comparison(
    fp32_value: float, int8_value: float, *, lower_is_better: bool
) -> Mapping[str, Any]:
    if math.isclose(fp32_value, int8_value, rel_tol=1.0e-12, abs_tol=1.0e-12):
        direction = "equal"
    elif (int8_value < fp32_value) == lower_is_better:
        direction = "int8_better"
    else:
        direction = "int8_worse"
    return {
        "fp32": fp32_value,
        "int8": int8_value,
        "int8_minus_fp32": int8_value - fp32_value,
        "fp32_div_int8": None if int8_value == 0.0 else fp32_value / int8_value,
        "int8_div_fp32": None if fp32_value == 0.0 else int8_value / fp32_value,
        "direction": direction,
    }


def _numbers_equal(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        return False
    return math.isfinite(float(actual)) and math.isclose(
        float(actual), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-12
    )


def _validate_metric_comparison(
    value: Any,
    fp32_value: float,
    int8_value: float,
    *,
    lower_is_better: bool,
    object_name: str,
) -> Mapping[str, Any]:
    actual = _mapping(value, object_name)
    expected = _expected_metric_comparison(
        fp32_value, int8_value, lower_is_better=lower_is_better
    )
    _require_equal(set(actual), set(expected), f"{object_name}.fields", "regenerate the benchmark comparison")
    for field, expected_value in expected.items():
        if field == "direction":
            _require_equal(actual[field], expected_value, f"{object_name}.{field}", "regenerate the comparison from the supplied raw benchmarks")
        elif not _numbers_equal(actual[field], expected_value):
            fail(
                f"{object_name}.{field}",
                repr(expected_value),
                repr(actual[field]),
                "regenerate the comparison from the supplied raw benchmarks",
            )
    return actual


def _validate_protocol(
    document: Any, protocol_sha256: str, integrity: Mapping[str, Any]
) -> Mapping[str, Any]:
    protocol = _mapping(document, "protocol")
    _require_equal(protocol.get("schema_version"), 1, "protocol.schema_version", "use the frozen S2-01 protocol")
    protocol_id = _string(protocol.get("protocol_id"), "protocol.protocol_id")
    if protocol_id not in SUPPORTED_PROTOCOL_IDS:
        fail(
            "protocol.protocol_id",
            repr(sorted(SUPPORTED_PROTOCOL_IDS)),
            protocol_id,
            "use an immutable reviewed S2-01 protocol version",
        )
    protocol_hash = _sha256(protocol_sha256, "protocol.canonical_lf_sha256")
    _require_equal(
        protocol_hash,
        SUPPORTED_PROTOCOL_SHA256[protocol_id],
        "protocol.canonical_lf_sha256",
        "restore the reviewed immutable protocol version",
    )

    source = _mapping(protocol.get("source_model"), "protocol.source_model")
    source_integrity = _mapping(integrity.get("source_model"), "integrity.source_model")
    source_sha = _sha256(source.get("sha256"), "protocol.source_model.sha256")
    source_size = _integer(source.get("size_bytes"), "protocol.source_model.size_bytes", 1)
    _require_equal(_sha256(source_integrity.get("sha256"), "integrity.source_model.sha256"), source_sha, "integrity.source_model.sha256", "restore the frozen FP32 source model")
    _require_equal(_integer(source_integrity.get("size_bytes"), "integrity.source_model.size_bytes", 1), source_size, "integrity.source_model.size_bytes", "restore the frozen FP32 source model")

    calibration = _mapping(protocol.get("calibration"), "protocol.calibration")
    correctness = _mapping(protocol.get("correctness"), "protocol.correctness")
    consistency = _mapping(correctness.get("consistency_manifest"), "protocol.correctness.consistency_manifest")
    quality = _mapping(correctness.get("quality_manifest"), "protocol.correctness.quality_manifest")
    manifest_specs = {
        "calibration": (
            _string(calibration.get("manifest_id"), "protocol.calibration.manifest_id"),
            _sha256(calibration.get("manifest_sha256_canonical_lf"), "protocol.calibration.manifest_sha256_canonical_lf"),
        ),
        "product": (
            _string(consistency.get("manifest_id"), "protocol.correctness.consistency_manifest.manifest_id"),
            _sha256(consistency.get("sha256_canonical_lf"), "protocol.correctness.consistency_manifest.sha256_canonical_lf"),
        ),
        "quality": (
            _string(quality.get("manifest_id"), "protocol.correctness.quality_manifest.manifest_id"),
            _sha256(quality.get("sha256_canonical_lf"), "protocol.correctness.quality_manifest.sha256_canonical_lf"),
        ),
    }
    for name, (_, expected_hash) in manifest_specs.items():
        actual = _mapping(integrity.get(f"{name}_manifest"), f"integrity.{name}_manifest")
        _require_equal(_sha256(actual.get("sha256"), f"integrity.{name}_manifest.sha256"), expected_hash, f"integrity.{name}_manifest.sha256", "restore the manifest frozen before quantization")

    benchmark = _mapping(protocol.get("benchmark"), "protocol.benchmark")
    profiling = _mapping(protocol.get("profiling"), "protocol.profiling")
    sample = _mapping(benchmark.get("sample"), "protocol.benchmark.sample")
    sample_integrity = _mapping(integrity.get("benchmark_sample"), "integrity.benchmark_sample")
    sample_sha = _sha256(sample.get("image_sha256"), "protocol.benchmark.sample.image_sha256")
    _require_equal(_sha256(sample_integrity.get("sha256"), "integrity.benchmark_sample.sha256"), sample_sha, "integrity.benchmark_sample.sha256", "restore the frozen benchmark image")
    _require_equal(benchmark.get("profiling_enabled"), False, "protocol.benchmark.profiling_enabled", "keep formal benchmark and profiling separate")
    _require_equal(profiling.get("separate_from_formal_benchmark"), True, "protocol.profiling.separate_from_formal_benchmark", "keep profiler runs separate")
    _require_equal(profiling.get("performance_gate"), False, "protocol.profiling.performance_gate", "do not make profiler timing a speed gate")

    quantization = _mapping(protocol.get("quantization"), "protocol.quantization")
    excluded_nodes = tuple(
        _string(value, f"protocol.quantization.nodes_to_exclude[{index}]")
        for index, value in enumerate(
            _sequence(
                quantization.get("nodes_to_exclude", []),
                "protocol.quantization.nodes_to_exclude",
            )
        )
    )
    if len(excluded_nodes) != len(set(excluded_nodes)):
        fail(
            "protocol.quantization.nodes_to_exclude",
            "unique node names",
            repr(excluded_nodes),
            "remove duplicate exclusions from the frozen protocol",
        )
    selected_node_count = _integer(
        quantization.get("expected_selected_node_count"),
        "protocol.quantization.expected_selected_node_count",
        1,
    )
    if len(excluded_nodes) >= selected_node_count:
        fail(
            "protocol.quantization.nodes_to_exclude",
            f"fewer than {selected_node_count} exclusions",
            str(len(excluded_nodes)),
            "retain at least one selected Conv for static PTQ",
        )
    return {
        "protocol_id": protocol_id,
        "canonical_lf_sha256": protocol_hash,
        "source_sha256": source_sha,
        "source_size_bytes": source_size,
        "calibration_manifest_id": manifest_specs["calibration"][0],
        "calibration_manifest_sha256": manifest_specs["calibration"][1],
        "product_manifest_id": manifest_specs["product"][0],
        "product_manifest_sha256": manifest_specs["product"][1],
        "quality_manifest_id": manifest_specs["quality"][0],
        "quality_manifest_sha256": manifest_specs["quality"][1],
        "calibration_sample_count": _integer(calibration.get("sample_count"), "protocol.calibration.sample_count", 1),
        "selected_node_count": selected_node_count,
        "excluded_nodes": excluded_nodes,
        "target_quantized_node_count": selected_node_count - len(excluded_nodes),
        "quantization_declaration": quantization,
        "benchmark": benchmark,
        "benchmark_sample_sha256": sample_sha,
        "profiling": profiling,
        "correctness_declaration": correctness,
    }


def _validate_quantization(
    document: Any,
    protocol_document: Mapping[str, Any],
    protocol: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = _schema_and_type(document, "s2_01_static_ptq_artifact_card", "quant_report")
    _require_true(
        root.get("passed"),
        "quant_report.passed",
        "rerun PTQ and fix the failed artifact-card gate",
    )
    binding = _mapping(root.get("protocol"), "quant_report.protocol")
    _require_equal(binding.get("protocol_id"), protocol["protocol_id"], "quant_report.protocol.protocol_id", "use the card produced by this protocol")
    _require_equal(_sha256(binding.get("canonical_lf_sha256"), "quant_report.protocol.canonical_lf_sha256"), protocol["canonical_lf_sha256"], "quant_report.protocol.canonical_lf_sha256", "rerun PTQ from the frozen protocol")
    protocol_integrity = _mapping(integrity.get("protocol"), "integrity.protocol")
    _require_equal(
        _sha256(binding.get("raw_sha256"), "quant_report.protocol.raw_sha256"),
        _sha256(protocol_integrity.get("sha256"), "integrity.protocol.sha256"),
        "quant_report.protocol.raw_sha256",
        "rerun PTQ from the exact protocol bytes",
    )

    quantization = _mapping(root.get("quantization"), "quant_report.quantization")
    _require_equal(
        _mapping(quantization.get("parameters"), "quant_report.quantization.parameters"),
        protocol["quantization_declaration"],
        "quant_report.quantization.parameters",
        "rerun PTQ with every frozen quantization parameter",
    )

    contract = _mapping(root.get("artifact_contract"), "quant_report.artifact_contract")
    source_contract = _mapping(contract.get("source_model"), "quant_report.artifact_contract.source_model")
    _require_equal(_sha256(source_contract.get("sha256"), "quant_report.artifact_contract.source_model.sha256"), protocol["source_sha256"], "quant_report.artifact_contract.source_model.sha256", "select the frozen source model")
    _require_equal(_integer(source_contract.get("size_bytes"), "quant_report.artifact_contract.source_model.size_bytes", 1), protocol["source_size_bytes"], "quant_report.artifact_contract.source_model.size_bytes", "select the frozen source model")

    artifacts = _mapping(root.get("artifacts"), "quant_report.artifacts")
    source = _mapping(artifacts.get("source"), "quant_report.artifacts.source")
    derived = _mapping(artifacts.get("derived"), "quant_report.artifacts.derived")
    derived_integrity = _mapping(integrity.get("derived_model"), "integrity.derived_model")
    derived_sha = _sha256(derived.get("sha256"), "quant_report.artifacts.derived.sha256")
    derived_size = _integer(derived.get("size_bytes"), "quant_report.artifacts.derived.size_bytes", 1)
    _require_equal(_sha256(source.get("sha256"), "quant_report.artifacts.source.sha256"), protocol["source_sha256"], "quant_report.artifacts.source.sha256", "regenerate the artifact card")
    _require_equal(_integer(source.get("size_bytes"), "quant_report.artifacts.source.size_bytes", 1), protocol["source_size_bytes"], "quant_report.artifacts.source.size_bytes", "regenerate the artifact card")
    _require_equal(_sha256(derived_integrity.get("sha256"), "integrity.derived_model.sha256"), derived_sha, "integrity.derived_model.sha256", "restore the published INT8 model")
    _require_equal(_integer(derived_integrity.get("size_bytes"), "integrity.derived_model.size_bytes", 1), derived_size, "integrity.derived_model.size_bytes", "restore the published INT8 model")
    if derived_sha == protocol["source_sha256"]:
        fail("quant_report.artifacts.derived.sha256", "a digest distinct from FP32", derived_sha, "publish the derived INT8 model")
    for name, artifact in (("source", source), ("derived", derived)):
        _require_equal(artifact.get("onnx_checker"), "passed", f"quant_report.artifacts.{name}.onnx_checker", "rerun ONNX checker and PTQ")

    calibration = _mapping(root.get("calibration"), "quant_report.calibration")
    _require_equal(calibration.get("manifest_id"), protocol["calibration_manifest_id"], "quant_report.calibration.manifest_id", "use the frozen calibration manifest")
    _require_equal(_sha256(calibration.get("manifest_sha256_canonical_lf"), "quant_report.calibration.manifest_sha256_canonical_lf"), protocol["calibration_manifest_sha256"], "quant_report.calibration.manifest_sha256_canonical_lf", "rerun PTQ from the frozen calibration manifest")
    for field in ("sample_count_expected", "sample_count_hash_verified", "sample_count_consumed"):
        _require_equal(calibration.get(field), protocol["calibration_sample_count"], f"quant_report.calibration.{field}", "consume every frozen calibration sample exactly once")

    downstream = _mapping(root.get("frozen_downstream_protocol"), "quant_report.frozen_downstream_protocol")
    downstream_correctness = _mapping(downstream.get("correctness"), "quant_report.frozen_downstream_protocol.correctness")
    _require_equal(downstream_correctness.get("declaration"), protocol_document.get("correctness"), "quant_report.frozen_downstream_protocol.correctness.declaration", "use the card generated before downstream evaluation")
    downstream_benchmark = _mapping(downstream.get("benchmark"), "quant_report.frozen_downstream_protocol.benchmark")
    _require_equal(downstream_benchmark.get("declaration"), protocol_document.get("benchmark"), "quant_report.frozen_downstream_protocol.benchmark.declaration", "use the same benchmark protocol")
    _require_equal(downstream.get("profiling"), protocol_document.get("profiling"), "quant_report.frozen_downstream_protocol.profiling", "use the same profiling protocol")

    graph = _mapping(root.get("graph_audit"), "quant_report.graph_audit")
    selection = _mapping(graph.get("selection"), "quant_report.graph_audit.selection")
    result = _mapping(graph.get("result"), "quant_report.graph_audit.result")
    expected_nodes = protocol["selected_node_count"]
    excluded_nodes = list(protocol["excluded_nodes"])
    target_quantized_nodes = protocol["target_quantized_node_count"]
    _require_equal(selection.get("selected_count"), expected_nodes, "quant_report.graph_audit.selection.selected_count", "rerun the frozen Conv-only QDQ PTQ")
    selected_nodes = [
        _string(value, f"quant_report.graph_audit.selection.selected_conv_nodes[{index}]")
        for index, value in enumerate(
            _sequence(selection.get("selected_conv_nodes"), "quant_report.graph_audit.selection.selected_conv_nodes")
        )
    ]
    _require_equal(len(selected_nodes), expected_nodes, "quant_report.graph_audit.selection.selected_conv_nodes.length", "regenerate the complete graph audit")
    _require_equal(len(set(selected_nodes)), expected_nodes, "quant_report.graph_audit.selection.selected_conv_nodes.uniqueness", "remove duplicated graph-audit identities")
    _require_equal(selection.get("excluded_conv_count"), len(excluded_nodes), "quant_report.graph_audit.selection.excluded_conv_count", "regenerate the model from the exact frozen exclusion list")
    _require_equal(
        list(_sequence(selection.get("excluded_conv_nodes"), "quant_report.graph_audit.selection.excluded_conv_nodes")),
        excluded_nodes,
        "quant_report.graph_audit.selection.excluded_conv_nodes",
        "regenerate the model from the exact frozen exclusion list",
    )
    target_nodes = [node for node in selected_nodes if node not in set(excluded_nodes)]
    _require_equal(
        list(_sequence(selection.get("target_conv_nodes"), "quant_report.graph_audit.selection.target_conv_nodes")),
        target_nodes,
        "quant_report.graph_audit.selection.target_conv_nodes",
        "record the ordered selected-minus-excluded Conv partition",
    )
    _require_equal(selection.get("target_conv_count"), target_quantized_nodes, "quant_report.graph_audit.selection.target_conv_count", "record the frozen target count")
    _require_equal(result.get("quantized_conv_count"), target_quantized_nodes, "quant_report.graph_audit.result.quantized_conv_count", "inspect failed or excluded Conv nodes")
    _require_equal(
        list(_sequence(result.get("quantized_conv_nodes"), "quant_report.graph_audit.result.quantized_conv_nodes")),
        target_nodes,
        "quant_report.graph_audit.result.quantized_conv_nodes",
        "quantize every and only frozen target Conv",
    )
    _require_equal(result.get("intentional_unquantized_conv_count"), len(excluded_nodes), "quant_report.graph_audit.result.intentional_unquantized_conv_count", "inspect the frozen selective-PTQ policy")
    _require_equal(
        list(_sequence(result.get("intentional_unquantized_conv_nodes"), "quant_report.graph_audit.result.intentional_unquantized_conv_nodes")),
        excluded_nodes,
        "quant_report.graph_audit.result.intentional_unquantized_conv_nodes",
        "regenerate the model from the exact frozen exclusion list",
    )
    _require_equal(result.get("unquantized_conv_count"), 0, "quant_report.graph_audit.result.unquantized_conv_count", "inspect target Conv nodes that did not form QDQ")
    _require_equal(
        list(_sequence(result.get("unquantized_conv_nodes"), "quant_report.graph_audit.result.unquantized_conv_nodes")),
        [],
        "quant_report.graph_audit.result.unquantized_conv_nodes",
        "inspect target Conv nodes that did not form QDQ",
    )
    _require_equal(result.get("failed_conv_count"), 0, "quant_report.graph_audit.result.failed_conv_count", "inspect QDQ structure failures")
    _require_equal(
        list(_sequence(result.get("failed_conv_nodes"), "quant_report.graph_audit.result.failed_conv_nodes")),
        [],
        "quant_report.graph_audit.result.failed_conv_nodes",
        "inspect every structural QDQ audit failure",
    )
    _require_equal(result.get("excluded_policy_violation_count"), 0, "quant_report.graph_audit.result.excluded_policy_violation_count", "keep every excluded Conv directly FP32")
    _require_equal(
        list(_sequence(result.get("excluded_policy_violations"), "quant_report.graph_audit.result.excluded_policy_violations")),
        [],
        "quant_report.graph_audit.result.excluded_policy_violations",
        "keep every excluded Conv directly FP32",
    )

    scope = _mapping(contract.get("quantized_op_scope"), "quant_report.artifact_contract.quantized_op_scope")
    expected_scope = {
        "op_types_to_quantize": ["Conv"],
        "selected_conv_count": expected_nodes,
        "excluded_conv_count": len(excluded_nodes),
        "excluded_conv_nodes": list(excluded_nodes),
        "target_conv_count": target_quantized_nodes,
        "unselected_source_nodes_remain_in_declared_precision": True,
    }
    _require_equal(scope, expected_scope, "quant_report.artifact_contract.quantized_op_scope", "regenerate the artifact contract from the frozen graph partition")

    size_comparison = _mapping(root.get("model_size_comparison"), "quant_report.model_size_comparison")
    expected_ratio = derived_size / protocol["source_size_bytes"]
    expected_size = {
        "source_fp32_size_bytes": protocol["source_size_bytes"],
        "derived_int8_size_bytes": derived_size,
        "size_delta_bytes": derived_size - protocol["source_size_bytes"],
        "int8_to_fp32_ratio": expected_ratio,
        "size_reduction_percent": (1.0 - expected_ratio) * 100.0,
    }
    _require_equal(set(size_comparison), set(expected_size), "quant_report.model_size_comparison.fields", "regenerate the size comparison")
    for field, expected_value in expected_size.items():
        actual_value = size_comparison.get(field)
        if isinstance(expected_value, float):
            if not _numbers_equal(actual_value, expected_value):
                fail(f"quant_report.model_size_comparison.{field}", repr(expected_value), repr(actual_value), "regenerate the size comparison from actual files")
        else:
            _require_equal(actual_value, expected_value, f"quant_report.model_size_comparison.{field}", "regenerate the size comparison from actual files")

    runtime = _mapping(root.get("runtime_validation"), "quant_report.runtime_validation")
    for name in ("source_python_ort", "derived_python_ort"):
        session = _mapping(runtime.get(name), f"quant_report.runtime_validation.{name}")
        _require_equal(session.get("status"), "passed", f"quant_report.runtime_validation.{name}.status", "rerun Python ORT legality")
        output = _mapping(session.get("output"), f"quant_report.runtime_validation.{name}.output")
        _require_true(output.get("all_finite"), f"quant_report.runtime_validation.{name}.output.all_finite")
    return {
        "passed": True,
        "source_sha256": protocol["source_sha256"],
        "derived_sha256": derived_sha,
        "source_size_bytes": protocol["source_size_bytes"],
        "derived_size_bytes": derived_size,
        "quantized_conv_count": target_quantized_nodes,
        "intentional_unquantized_conv_count": len(excluded_nodes),
    }


def _validate_correctness(
    document: Any,
    protocol: Mapping[str, Any],
    quant: Mapping[str, Any],
    correctness_policy: str,
) -> Mapping[str, Any]:
    root = _schema_and_type(document, "s2_01_fp32_int8_correctness_and_quality", "correctness")
    if correctness_policy not in ("required", "advisory"):
        fail(
            "correctness_policy",
            "'required' or 'advisory'",
            repr(correctness_policy),
            "select an explicit completion policy",
        )
    reported_passed = root.get("passed")
    if type(reported_passed) is not bool:
        fail(
            "correctness.passed",
            "a boolean",
            repr(reported_passed),
            "regenerate the correctness evidence",
        )
    if correctness_policy == "required":
        _require_true(
            reported_passed,
            "correctness.passed",
            "fix correctness or explicitly select advisory exercise completion",
        )
    binding = _mapping(root.get("protocol"), "correctness.protocol")
    _require_equal(binding.get("protocol_id"), protocol["protocol_id"], "correctness.protocol.protocol_id", "rerun correctness from this protocol")
    _require_equal(_sha256(binding.get("canonical_lf_sha256"), "correctness.protocol.canonical_lf_sha256"), protocol["canonical_lf_sha256"], "correctness.protocol.canonical_lf_sha256", "rerun correctness from this protocol")
    _require_equal(binding.get("profiler_or_benchmark_enabled"), False, "correctness.protocol.profiler_or_benchmark_enabled", "run correctness separately from benchmark/profile")

    legality = _mapping(root.get("runtime_legality"), "correctness.runtime_legality")
    _require_true(legality.get("python_fp32_session_and_finite_outputs"), "correctness.runtime_legality.python_fp32_session_and_finite_outputs")
    _require_true(legality.get("python_int8_session_and_finite_outputs"), "correctness.runtime_legality.python_int8_session_and_finite_outputs")
    cpp = _mapping(legality.get("cpp"), "correctness.runtime_legality.cpp")
    _require_true(cpp.get("requested"), "correctness.runtime_legality.cpp.requested")
    _require_true(cpp.get("passed"), "correctness.runtime_legality.cpp.passed")

    manifests = _mapping(root.get("manifests"), "correctness.manifests")
    expected_manifests = {
        "calibration": (protocol["calibration_manifest_id"], protocol["calibration_manifest_sha256"]),
        "product": (protocol["product_manifest_id"], protocol["product_manifest_sha256"]),
        "quality": (protocol["quality_manifest_id"], protocol["quality_manifest_sha256"]),
    }
    for name, (expected_id, expected_hash) in expected_manifests.items():
        manifest = _mapping(manifests.get(name), f"correctness.manifests.{name}")
        _require_equal(manifest.get("manifest_id"), expected_id, f"correctness.manifests.{name}.manifest_id", "rerun correctness from the frozen manifest")
        _require_equal(_sha256(manifest.get("canonical_lf_sha256"), f"correctness.manifests.{name}.canonical_lf_sha256"), expected_hash, f"correctness.manifests.{name}.canonical_lf_sha256", "rerun correctness from the frozen manifest")

    artifacts = _mapping(root.get("artifacts"), "correctness.artifacts")
    fp32 = _mapping(artifacts.get("fp32"), "correctness.artifacts.fp32")
    int8 = _mapping(artifacts.get("int8"), "correctness.artifacts.int8")
    _require_equal(_sha256(fp32.get("model_sha256"), "correctness.artifacts.fp32.model_sha256"), quant["source_sha256"], "correctness.artifacts.fp32.model_sha256", "evaluate the exact FP32 source")
    _require_equal(_sha256(int8.get("model_sha256"), "correctness.artifacts.int8.model_sha256"), quant["derived_sha256"], "correctness.artifacts.int8.model_sha256", "evaluate the exact derived INT8 model")
    _require_equal(fp32.get("model_size_bytes"), quant["source_size_bytes"], "correctness.artifacts.fp32.model_size_bytes", "evaluate the exact FP32 source")
    _require_equal(int8.get("model_size_bytes"), quant["derived_size_bytes"], "correctness.artifacts.int8.model_size_bytes", "evaluate the exact derived INT8 model")

    product = _mapping(root.get("product_detection_difference"), "correctness.product_detection_difference")
    quality = _mapping(root.get("task_quality"), "correctness.task_quality")
    product_passed = product.get("passed")
    quality_passed = quality.get("passed")
    for value, object_name in (
        (product_passed, "correctness.product_detection_difference.passed"),
        (quality_passed, "correctness.task_quality.passed"),
    ):
        if type(value) is not bool:
            fail(object_name, "a boolean", repr(value), "regenerate the correctness evidence")
    strict_acceptance_passed = bool(reported_passed and product_passed and quality_passed)
    if correctness_policy == "required":
        _require_true(product_passed, "correctness.product_detection_difference.passed")
        _require_true(quality_passed, "correctness.task_quality.passed")
    return {
        "policy": correctness_policy,
        "reported_passed": reported_passed,
        "strict_acceptance_passed": strict_acceptance_passed,
        "accepted_for_completion": strict_acceptance_passed or correctness_policy == "advisory",
        "python_runtime_legality": True,
        "cpp_runtime_legality": True,
        "product_detection_difference_passed": product_passed,
        "task_quality_passed": quality_passed,
    }


def _validate_statistics(value: Any, object_name: str, repeat: int) -> Mapping[str, float]:
    statistics = _mapping(value, object_name)
    _require_equal(statistics.get("sample_count"), repeat, f"{object_name}.sample_count", "rerun the frozen benchmark")
    result = {field: _number(statistics.get(field), f"{object_name}.{field}") for field in ("mean", "p50", "p95")}
    if result["p50"] > result["p95"]:
        fail(object_name, "p50 <= p95", repr(result), "inspect benchmark percentile calculation")
    return result


def _validate_benchmark(
    document: Any,
    precision: str,
    protocol: Mapping[str, Any],
    quant: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = _schema_and_type(document, "cpp_ort_single_image_release_benchmark", f"{precision}_benchmark")
    declaration = protocol["benchmark"]
    benchmark_protocol = _mapping(root.get("protocol"), f"{precision}_benchmark.protocol")
    repeat = _integer(benchmark_protocol.get("repeat"), f"{precision}_benchmark.protocol.repeat", 1)
    _require_equal(repeat, declaration.get("repeat"), f"{precision}_benchmark.protocol.repeat", "rerun the frozen benchmark")
    _require_equal(benchmark_protocol.get("warmup"), declaration.get("warmup"), f"{precision}_benchmark.protocol.warmup", "rerun the frozen benchmark")
    _require_equal(benchmark_protocol.get("batch_size"), 1, f"{precision}_benchmark.protocol.batch_size", "use batch one")
    _require_equal(benchmark_protocol.get("sample_count"), 1, f"{precision}_benchmark.protocol.sample_count", "use the frozen single image")

    environment = _mapping(root.get("environment"), f"{precision}_benchmark.environment")
    build = _mapping(environment.get("build"), f"{precision}_benchmark.environment.build")
    _require_equal(build.get("type"), declaration.get("build_type"), f"{precision}_benchmark.environment.build.type", "rerun a Release build")
    _require_equal(build.get("cxx_standard"), 17, f"{precision}_benchmark.environment.build.cxx_standard", "use the C++17 runtime")
    runtime = _mapping(root.get("runtime"), f"{precision}_benchmark.runtime")
    _require_equal(runtime.get("requested_provider"), "cpu", f"{precision}_benchmark.runtime.requested_provider", "use the CPU backend")
    _require_equal(runtime.get("actual_provider"), declaration.get("execution_provider"), f"{precision}_benchmark.runtime.actual_provider", "use the frozen provider")
    session = _mapping(runtime.get("session"), f"{precision}_benchmark.runtime.session")
    session_expected = {
        "execution_mode": declaration.get("execution_mode"),
        "intra_op_num_threads": declaration.get("intra_op_num_threads"),
        "inter_op_num_threads": declaration.get("inter_op_num_threads"),
        "graph_optimization_level": declaration.get("graph_optimization_level"),
        "profiling_enabled": False,
    }
    for field, expected in session_expected.items():
        _require_equal(session.get(field), expected, f"{precision}_benchmark.runtime.session.{field}", "rerun an unprofiled frozen benchmark")
    initialization_ms = _number(session.get("initialization_ms"), f"{precision}_benchmark.runtime.session.initialization_ms")

    model = _mapping(root.get("model"), f"{precision}_benchmark.model")
    expected_sha = quant["source_sha256"] if precision == "fp32" else quant["derived_sha256"]
    expected_size = quant["source_size_bytes"] if precision == "fp32" else quant["derived_size_bytes"]
    _require_equal(_sha256(model.get("declared_sha256"), f"{precision}_benchmark.model.declared_sha256"), expected_sha, f"{precision}_benchmark.model.declared_sha256", "benchmark the exact accepted artifact")
    _require_equal(model.get("file_size_bytes"), expected_size, f"{precision}_benchmark.model.file_size_bytes", "benchmark the exact accepted artifact")

    sample = _mapping(root.get("sample"), f"{precision}_benchmark.sample")
    _require_equal(sample.get("sample_count"), 1, f"{precision}_benchmark.sample.sample_count", "use the frozen single image")
    latency_root = _mapping(root.get("latency_ms"), f"{precision}_benchmark.latency_ms")
    latency = {segment: _validate_statistics(latency_root.get(segment), f"{precision}_benchmark.latency_ms.{segment}", repeat) for segment in LATENCY_SEGMENTS}
    throughput = _mapping(root.get("throughput_images_per_second"), f"{precision}_benchmark.throughput_images_per_second")
    throughput_values = {name: _number(throughput.get(name), f"{precision}_benchmark.throughput_images_per_second.{name}") for name in ("pipeline", "end_to_end")}
    memory = _mapping(root.get("memory"), f"{precision}_benchmark.memory")
    _require_equal(memory.get("status"), "supported", f"{precision}_benchmark.memory.status", "run on the Windows evidence platform")
    _require_equal(memory.get("metric"), "peak_working_set", f"{precision}_benchmark.memory.metric", "record Peak Working Set")
    memory_bytes = _integer(memory.get("bytes"), f"{precision}_benchmark.memory.bytes", 1)
    return {
        "document": root,
        "protocol": benchmark_protocol,
        "environment": environment,
        "runtime": runtime,
        "session": session,
        "model": model,
        "sample": sample,
        "postprocess": _mapping(root.get("postprocess"), f"{precision}_benchmark.postprocess"),
        "initialization_ms": initialization_ms,
        "latency": latency,
        "throughput": throughput_values,
        "memory_bytes": memory_bytes,
        "model_sha256": expected_sha,
        "model_size_bytes": expected_size,
    }


def _validate_benchmark_comparison(
    document: Any,
    protocol: Mapping[str, Any],
    correctness: Mapping[str, Any],
    fp32: Mapping[str, Any],
    int8: Mapping[str, Any],
    correctness_policy: str,
) -> Mapping[str, Any]:
    root = _schema_and_type(document, "s2_01_fp32_int8_cpp_benchmark_comparison", "benchmark_comparison")
    _require_true(root.get("passed"), "benchmark_comparison.passed")
    prerequisite = _mapping(root.get("correctness_prerequisite"), "benchmark_comparison.correctness_prerequisite")
    expected_prerequisite = {
        "policy": correctness_policy,
        "passed": correctness["reported_passed"],
        "blocking": correctness_policy == "required",
        "accepted_for_comparison": correctness["accepted_for_completion"],
        "evidence_type": "s2_01_fp32_int8_correctness_and_quality",
    }
    for field, expected in expected_prerequisite.items():
        _require_equal(
            prerequisite.get(field),
            expected,
            f"benchmark_comparison.correctness_prerequisite.{field}",
            "regenerate the comparison with the same explicit correctness policy",
        )

    binding = _mapping(root.get("protocol_binding"), "benchmark_comparison.protocol_binding")
    expected_binding = {
        "protocol_id": protocol["protocol_id"],
        "canonical_lf_sha256": protocol["canonical_lf_sha256"],
        "source_model_sha256": fp32["model_sha256"],
        "derived_model_sha256": int8["model_sha256"],
        "warmup": protocol["benchmark"].get("warmup"),
        "repeat": protocol["benchmark"].get("repeat"),
    }
    for field, expected in expected_binding.items():
        _require_equal(binding.get(field), expected, f"benchmark_comparison.protocol_binding.{field}", "regenerate comparison from the same protocol")

    comparable = _mapping(root.get("comparability"), "benchmark_comparison.comparability")
    for field in (
        "same_machine_environment",
        "same_release_build",
        "same_provider_and_threads",
        "same_sample_and_postprocess",
        "same_warmup_repeat",
        "profiling_disabled",
    ):
        _require_true(comparable.get(field), f"benchmark_comparison.comparability.{field}")
    _require_equal(fp32["protocol"], int8["protocol"], "benchmarks.protocol", "rerun both benchmarks under one protocol")
    _require_equal(fp32["environment"], int8["environment"], "benchmarks.environment", "rerun both benchmarks on the same machine/build")
    _require_equal(_without(fp32["runtime"], "session"), _without(int8["runtime"], "session"), "benchmarks.runtime", "rerun both benchmarks with one provider")
    _require_equal(_without(fp32["session"], "initialization_ms"), _without(int8["session"], "initialization_ms"), "benchmarks.runtime.session", "rerun both benchmarks with one session policy")
    _require_equal(fp32["sample"], int8["sample"], "benchmarks.sample", "rerun both benchmarks on the frozen image")
    _require_equal(_without(fp32["postprocess"], "detection_count"), _without(int8["postprocess"], "detection_count"), "benchmarks.postprocess", "rerun both benchmarks with one postprocess protocol")

    models = _mapping(root.get("models"), "benchmark_comparison.models")
    for precision, validated in (("fp32", fp32), ("int8", int8)):
        model = _mapping(models.get(precision), f"benchmark_comparison.models.{precision}")
        _require_equal(_sha256(model.get("sha256"), f"benchmark_comparison.models.{precision}.sha256"), validated["model_sha256"], f"benchmark_comparison.models.{precision}.sha256", "regenerate comparison from the supplied benchmark")
        _require_equal(model.get("file_size_bytes"), validated["model_size_bytes"], f"benchmark_comparison.models.{precision}.file_size_bytes", "regenerate comparison from the supplied benchmark")
    size = _mapping(models.get("size"), "benchmark_comparison.models.size")
    expected_size = {
        "int8_minus_fp32_bytes": int8["model_size_bytes"] - fp32["model_size_bytes"],
        "int8_div_fp32": int8["model_size_bytes"] / fp32["model_size_bytes"],
        "reduction_fraction": 1.0 - int8["model_size_bytes"] / fp32["model_size_bytes"],
    }
    for field, expected in expected_size.items():
        if not _numbers_equal(size.get(field), expected):
            fail(
                f"benchmark_comparison.models.size.{field}",
                repr(expected),
                repr(size.get(field)),
                "regenerate the comparison from the supplied benchmark",
            )

    _validate_metric_comparison(root.get("session_initialization_ms"), fp32["initialization_ms"], int8["initialization_ms"], lower_is_better=True, object_name="benchmark_comparison.session_initialization_ms")
    latency_comparison = _mapping(root.get("latency_ms"), "benchmark_comparison.latency_ms")
    for segment in LATENCY_SEGMENTS:
        segment_result = _mapping(latency_comparison.get(segment), f"benchmark_comparison.latency_ms.{segment}")
        for statistic in ("mean", "p50", "p95"):
            _validate_metric_comparison(segment_result.get(statistic), fp32["latency"][segment][statistic], int8["latency"][segment][statistic], lower_is_better=True, object_name=f"benchmark_comparison.latency_ms.{segment}.{statistic}")
    throughput = _mapping(root.get("throughput_images_per_second"), "benchmark_comparison.throughput_images_per_second")
    for name in ("pipeline", "end_to_end"):
        _validate_metric_comparison(throughput.get(name), fp32["throughput"][name], int8["throughput"][name], lower_is_better=False, object_name=f"benchmark_comparison.throughput_images_per_second.{name}")
    _validate_metric_comparison(root.get("peak_working_set_bytes"), float(fp32["memory_bytes"]), float(int8["memory_bytes"]), lower_is_better=True, object_name="benchmark_comparison.peak_working_set_bytes")
    interpretation = _mapping(root.get("interpretation"), "benchmark_comparison.interpretation")
    _require_true(interpretation.get("speed_is_not_a_pass_condition"), "benchmark_comparison.interpretation.speed_is_not_a_pass_condition", "keep speed descriptive rather than an acceptance gate")
    pipeline_outcome = _expected_metric_comparison(
        fp32["latency"]["pipeline"]["mean"],
        int8["latency"]["pipeline"]["mean"],
        lower_is_better=True,
    )["direction"]
    session_run_outcome = _expected_metric_comparison(
        fp32["latency"]["session_run"]["mean"],
        int8["latency"]["session_run"]["mean"],
        lower_is_better=True,
    )["direction"]
    _require_equal(interpretation.get("pipeline_mean_outcome"), pipeline_outcome, "benchmark_comparison.interpretation.pipeline_mean_outcome", "regenerate the comparison from the supplied raw benchmarks")
    _require_equal(interpretation.get("session_run_mean_outcome"), session_run_outcome, "benchmark_comparison.interpretation.session_run_mean_outcome", "regenerate the comparison from the supplied raw benchmarks")
    return {
        "passed": True,
        "correctness_policy": correctness_policy,
        "correctness_prerequisite_reported_passed": correctness["reported_passed"],
        "speed_is_pass_gate": False,
        "pipeline_mean_outcome": pipeline_outcome,
        "session_run_mean_outcome": session_run_outcome,
        "model_size": size,
    }


def _validate_profile(
    document: Any,
    precision: str,
    protocol: Mapping[str, Any],
    quant: Mapping[str, Any],
    trace_integrity: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = _schema_and_type(document, "onnxruntime_node_profile_summary", f"{precision}_profile_summary")
    _require_true(root.get("passed"), f"{precision}_profile_summary.passed")
    expected_sha = quant["source_sha256"] if precision == "fp32" else quant["derived_sha256"]
    expected_size = quant["source_size_bytes"] if precision == "fp32" else quant["derived_size_bytes"]
    model = _mapping(root.get("model"), f"{precision}_profile_summary.model")
    _require_equal(model.get("precision"), precision, f"{precision}_profile_summary.model.precision", "pass the matching precision summary")
    _require_equal(_sha256(model.get("declared_sha256"), f"{precision}_profile_summary.model.declared_sha256"), expected_sha, f"{precision}_profile_summary.model.declared_sha256", "profile the exact accepted artifact")
    artifact = _mapping(root.get("artifact"), f"{precision}_profile_summary.artifact")
    _require_equal(_sha256(artifact.get("model_sha256"), f"{precision}_profile_summary.artifact.model_sha256"), expected_sha, f"{precision}_profile_summary.artifact.model_sha256", "profile the exact accepted artifact")
    _require_equal(artifact.get("model_size_bytes"), expected_size, f"{precision}_profile_summary.artifact.model_size_bytes", "profile the exact accepted artifact")

    binding = _mapping(root.get("protocol_binding"), f"{precision}_profile_summary.protocol_binding")
    expected_binding = {
        "protocol_id": protocol["protocol_id"],
        "canonical_lf_sha256": protocol["canonical_lf_sha256"],
        "profile_runs": protocol["profiling"].get("runs"),
        "provider": protocol["profiling"].get("execution_provider"),
        "sample_sha256": protocol["benchmark_sample_sha256"],
        "separate_from_formal_benchmark": True,
    }
    for field, expected in expected_binding.items():
        _require_equal(binding.get(field), expected, f"{precision}_profile_summary.protocol_binding.{field}", "regenerate the profile from this protocol")
    profile_protocol = _mapping(root.get("protocol"), f"{precision}_profile_summary.protocol")
    _require_equal(profile_protocol.get("expected_profile_runs"), protocol["profiling"].get("runs"), f"{precision}_profile_summary.protocol.expected_profile_runs", "rerun the frozen number of profile calls")
    _require_equal(profile_protocol.get("expected_provider"), protocol["profiling"].get("execution_provider"), f"{precision}_profile_summary.protocol.expected_provider", "profile the frozen provider")

    trace = _mapping(root.get("trace"), f"{precision}_profile_summary.trace")
    actual_trace = _mapping(trace_integrity, f"integrity.{precision}_trace")
    trace_sha = _sha256(trace.get("sha256"), f"{precision}_profile_summary.trace.sha256")
    _require_equal(_sha256(actual_trace.get("sha256"), f"integrity.{precision}_trace.sha256"), trace_sha, f"integrity.{precision}_trace.sha256", "restore the exact raw ORT trace")
    _require_equal(_integer(actual_trace.get("size_bytes"), f"integrity.{precision}_trace.size_bytes", 1), trace.get("size_bytes"), f"integrity.{precision}_trace.size_bytes", "restore the exact raw ORT trace")
    expected_runs = protocol["profiling"].get("runs")
    _require_equal(
        trace.get("session_model_run_event_count"),
        expected_runs,
        f"{precision}_profile_summary.trace.session_model_run_event_count",
        "regenerate the trace with the frozen number of Session::Run calls",
    )

    precision_signature = _mapping(
        model.get("trace_precision_signature"),
        f"{precision}_profile_summary.model.trace_precision_signature",
    )
    _require_true(
        precision_signature.get("verified"),
        f"{precision}_profile_summary.model.trace_precision_signature.verified",
        "regenerate the summary with artifact-backed optimized-graph signature validation",
    )

    result = _mapping(root.get("result"), f"{precision}_profile_summary.result")
    kernel_total_ms = _number(result.get("kernel_event_total_ms"), f"{precision}_profile_summary.result.kernel_event_total_ms", minimum=0.0)
    if kernel_total_ms <= 0.0:
        fail(f"{precision}_profile_summary.result.kernel_event_total_ms", "a positive duration", repr(kernel_total_ms), "regenerate a trace with completed Session::Run calls")
    _integer(result.get("unique_node_count"), f"{precision}_profile_summary.result.unique_node_count", 1)
    _require_equal(result.get("node_call_count_min"), expected_runs, f"{precision}_profile_summary.result.node_call_count_min", "regenerate the complete trace")
    _require_equal(result.get("node_call_count_max"), expected_runs, f"{precision}_profile_summary.result.node_call_count_max", "regenerate the complete trace")
    if not _sequence(result.get("top_nodes"), f"{precision}_profile_summary.result.top_nodes"):
        fail(f"{precision}_profile_summary.result.top_nodes", "at least one ranked node", "empty", "regenerate the profile summary")
    if not _sequence(result.get("top_operators"), f"{precision}_profile_summary.result.top_operators"):
        fail(f"{precision}_profile_summary.result.top_operators", "at least one ranked operator", "empty", "regenerate the profile summary")
    providers = _sequence(result.get("providers"), f"{precision}_profile_summary.result.providers")
    if not providers:
        fail(f"{precision}_profile_summary.result.providers", "one provider aggregate", "empty", "regenerate the profile summary")
    expected_provider = protocol["profiling"].get("execution_provider")
    for index, provider_row in enumerate(providers):
        provider = _mapping(provider_row, f"{precision}_profile_summary.result.providers[{index}]")
        _require_equal(provider.get("provider"), expected_provider, f"{precision}_profile_summary.result.providers[{index}].provider", "profile only the frozen execution provider")
    mapping = _mapping(root.get("segmented_benchmark_mapping"), f"{precision}_profile_summary.segmented_benchmark_mapping")
    _require_equal(mapping.get("outer_metric"), "latency_ms.session_run", f"{precision}_profile_summary.segmented_benchmark_mapping.outer_metric", "preserve the benchmark/profile scope mapping")
    _require_true(mapping.get("excluded_from_formal_benchmark"), f"{precision}_profile_summary.segmented_benchmark_mapping.excluded_from_formal_benchmark")
    overhead = _mapping(root.get("profiling_overhead"), f"{precision}_profile_summary.profiling_overhead")
    _require_true(overhead.get("present"), f"{precision}_profile_summary.profiling_overhead.present")
    _require_equal(overhead.get("quantified"), False, f"{precision}_profile_summary.profiling_overhead.quantified", "do not present profiled time as benchmark latency")
    return {
        "passed": True,
        "trace": {
            "path": actual_trace.get("path"),
            "sha256": trace_sha,
            "size_bytes": trace.get("size_bytes"),
        },
        "kernel_event_total_ms": kernel_total_ms,
        "unique_node_count": result.get("unique_node_count"),
        "top_nodes": result.get("top_nodes"),
        "top_operators": result.get("top_operators"),
        "profiling_overhead_is_not_benchmark": True,
    }


def assemble_documents(
    protocol_document: Mapping[str, Any],
    protocol_sha256: str,
    quant_report: Mapping[str, Any],
    correctness_document: Mapping[str, Any],
    fp32_benchmark_document: Mapping[str, Any],
    int8_benchmark_document: Mapping[str, Any],
    benchmark_comparison_document: Mapping[str, Any],
    fp32_profile_document: Mapping[str, Any],
    int8_profile_document: Mapping[str, Any],
    *,
    integrity: Mapping[str, Any],
    trace_integrity: Mapping[str, Mapping[str, Any]],
    path_bindings: Mapping[str, Any],
    input_records: Optional[Mapping[str, Any]] = None,
    correctness_policy: str = "required",
) -> Mapping[str, Any]:
    """Validate and assemble already-loaded evidence documents.

    The helper is intentionally filesystem-free so unit tests can use only
    in-memory documents.  The CLI performs file, path, and raw-trace checks and
    passes their measured integrity records here.
    """

    protocol = _validate_protocol(protocol_document, protocol_sha256, integrity)
    for key in PATH_BINDING_KEYS:
        _require_true(path_bindings.get(key), f"path_bindings.{key}", "use evidence recorded from the protocol-resolved path")
    quant = _validate_quantization(quant_report, protocol_document, protocol, integrity)
    correctness = _validate_correctness(
        correctness_document, protocol, quant, correctness_policy
    )
    fp32_benchmark = _validate_benchmark(fp32_benchmark_document, "fp32", protocol, quant)
    int8_benchmark = _validate_benchmark(int8_benchmark_document, "int8", protocol, quant)
    benchmark = _validate_benchmark_comparison(
        benchmark_comparison_document,
        protocol,
        correctness,
        fp32_benchmark,
        int8_benchmark,
        correctness_policy,
    )
    fp32_profile = _validate_profile(
        fp32_profile_document,
        "fp32",
        protocol,
        quant,
        _mapping(trace_integrity.get("fp32"), "trace_integrity.fp32"),
    )
    int8_profile = _validate_profile(
        int8_profile_document,
        "int8",
        protocol,
        quant,
        _mapping(trace_integrity.get("int8"), "trace_integrity.int8"),
    )
    checks = {
        "protocol_and_manifest_hashes": True,
        "source_and_derived_model_lineage": True,
        "static_ptq_graph_and_python_legality": True,
        "correctness_evidence_bound_and_policy_applied": correctness[
            "accepted_for_completion"
        ],
        "cpp_runtime_legality": True,
        "same_protocol_unprofiled_benchmarks": True,
        "benchmark_comparison_bound_to_raw_inputs": True,
        "fp32_raw_trace_exists_and_hash_matches": True,
        "int8_raw_trace_exists_and_hash_matches": True,
        "profiles_bound_to_benchmark_session_run_scope": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": (
            EVIDENCE_TYPE
            if correctness_policy == "required"
            else ADVISORY_EVIDENCE_TYPE
        ),
        "passed": all(checks.values()),
        "strict_acceptance_passed": correctness["strict_acceptance_passed"],
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "canonical_lf_sha256": protocol["canonical_lf_sha256"],
        },
        "evidence_inputs": dict(input_records or {}),
        "lineage": {
            "source_fp32": {
                "sha256": quant["source_sha256"],
                "size_bytes": quant["source_size_bytes"],
            },
            "derived_int8": {
                "sha256": quant["derived_sha256"],
                "size_bytes": quant["derived_size_bytes"],
                "quantized_conv_count": quant["quantized_conv_count"],
            },
            "manifests": {
                "calibration": {
                    "manifest_id": protocol["calibration_manifest_id"],
                    "canonical_lf_sha256": protocol["calibration_manifest_sha256"],
                },
                "product": {
                    "manifest_id": protocol["product_manifest_id"],
                    "canonical_lf_sha256": protocol["product_manifest_sha256"],
                },
                "quality": {
                    "manifest_id": protocol["quality_manifest_id"],
                    "canonical_lf_sha256": protocol["quality_manifest_sha256"],
                },
            },
        },
        "correctness": correctness,
        "performance": benchmark,
        "profiling": {"fp32": fp32_profile, "int8": int8_profile},
        "acceptance_checks": checks,
        "policy": {
            "correctness_policy": correctness_policy,
            "correctness_results_rewritten": False,
            "runtime_legality_required": True,
            "formal_benchmark_profiling_enabled": False,
            "profile_trace_used_as_formal_latency": False,
            "int8_speed_is_pass_gate": False,
            "slower_int8_is_valid_evidence": True,
        },
        "limitations": [
            "INT8 latency direction is reported but is not an acceptance condition.",
            *(
                [
                    "Product-difference and task-quality gates are diagnostic, non-blocking results under the user-approved exercise policy."
                ]
                if correctness_policy == "advisory"
                else []
            ),
            "ORT profile traces include instrumentation overhead and are diagnostic only.",
            "Peak Working Set is a process-lifetime high-water mark rather than incremental model memory.",
        ],
    }


def _resolve_protocol_path(declaration_path: Path, raw_path: Any, object_name: str) -> Path:
    value = _string(raw_path, object_name)
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (declaration_path.parent / path).resolve()
    if not resolved.is_file():
        fail(object_name, "an existing regular file", str(resolved), "restore the protocol-bound file")
    return resolved


def _recorded_path_matches(raw_path: Any, expected: Path, anchors: Sequence[Path]) -> bool:
    if not isinstance(raw_path, str) or not raw_path:
        return False
    recorded = Path(raw_path)
    expected_resolved = expected.resolve()
    if recorded.is_absolute():
        return recorded.resolve() == expected_resolved
    return any((anchor / recorded).resolve() == expected_resolved for anchor in anchors)


def _resolve_trace(
    raw_path: Any,
    expected_sha256: Any,
    expected_size: Any,
    anchors: Sequence[Path],
    object_name: str,
) -> Mapping[str, Any]:
    path_text = _string(raw_path, f"{object_name}.path")
    expected_sha = _sha256(expected_sha256, f"{object_name}.sha256")
    size = _integer(expected_size, f"{object_name}.size_bytes", 1)
    raw = Path(path_text)
    candidates = [raw.resolve()] if raw.is_absolute() else [(anchor / raw).resolve() for anchor in anchors]
    matches = []
    seen = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key in seen or not candidate.is_file() or candidate.stat().st_size != size:
            continue
        seen.add(key)
        if sha256_file(candidate) == expected_sha:
            matches.append(candidate)
    if len(matches) != 1:
        fail(
            object_name,
            "one existing trace whose size and SHA match the profile summary",
            repr([str(path) for path in matches]),
            "pass the summary generated from an unambiguous raw trace path",
        )
    return {"path": str(matches[0]), "sha256": expected_sha, "size_bytes": size}


def _input_record(path: Path) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + f".tmp.{os.getpid()}")
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        temporary.replace(resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--quant-report", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--fp32-benchmark", required=True, type=Path)
    parser.add_argument("--int8-benchmark", required=True, type=Path)
    parser.add_argument("--benchmark-comparison", required=True, type=Path)
    parser.add_argument("--fp32-profile-summary", required=True, type=Path)
    parser.add_argument("--int8-profile-summary", required=True, type=Path)
    parser.add_argument(
        "--correctness-policy",
        choices=("required", "advisory"),
        default="required",
        help="Require every correctness/quality gate (default) or retain them as non-blocking exercise diagnostics.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    named_paths = {
        "protocol": arguments.protocol,
        "quant_report": arguments.quant_report,
        "correctness": arguments.correctness,
        "fp32_benchmark": arguments.fp32_benchmark,
        "int8_benchmark": arguments.int8_benchmark,
        "benchmark_comparison": arguments.benchmark_comparison,
        "fp32_profile_summary": arguments.fp32_profile_summary,
        "int8_profile_summary": arguments.int8_profile_summary,
    }
    try:
        resolved_inputs = {name: path.resolve(strict=True) for name, path in named_paths.items()}
        if arguments.output.resolve() in set(resolved_inputs.values()):
            fail("output", "a path distinct from every input", str(arguments.output.resolve()), "choose a dedicated acceptance path")
        documents = {name: load_json(path) for name, path in resolved_inputs.items()}
        frozen = s2_01_protocol.load_s2_01_protocol(resolved_inputs["protocol"])
        protocol_hash = canonical_lf_sha256(frozen.declaration_path)
        derived_path = frozen.output_model_path.resolve()
        if not derived_path.is_file():
            fail("protocol.output.model_path", "the published derived INT8 model", str(derived_path), "run PTQ before evidence assembly")

        integrity = {
            "protocol": _input_record(frozen.declaration_path),
            "source_model": _input_record(frozen.source_model_path),
            "derived_model": _input_record(derived_path),
            "calibration_manifest": {
                "path": str(frozen.calibration_manifest_path),
                "sha256": canonical_lf_sha256(frozen.calibration_manifest_path),
            },
            "product_manifest": {
                "path": str(frozen.consistency_manifest_path),
                "sha256": canonical_lf_sha256(frozen.consistency_manifest_path),
            },
            "quality_manifest": {
                "path": str(frozen.quality_manifest_path),
                "sha256": canonical_lf_sha256(frozen.quality_manifest_path),
            },
            "benchmark_sample": _input_record(frozen.benchmark_sample_path),
        }
        repo_root = frozen.declaration_path.parents[2]
        cpp_infer_root = frozen.declaration_path.parent.parent
        common_anchors = (Path.cwd(), repo_root, cpp_infer_root, frozen.declaration_path.parent)
        fp32_benchmark = documents["fp32_benchmark"]
        int8_benchmark = documents["int8_benchmark"]
        quant = documents["quant_report"]
        correctness = documents["correctness"]
        fp32_profile = documents["fp32_profile_summary"]
        int8_profile = documents["int8_profile_summary"]
        path_bindings = {
            "quant_source_model": _recorded_path_matches(_mapping(_mapping(quant.get("artifacts"), "quant_report.artifacts").get("source"), "quant_report.artifacts.source").get("path"), frozen.source_model_path, common_anchors),
            "quant_derived_model": _recorded_path_matches(_mapping(_mapping(quant.get("artifacts"), "quant_report.artifacts").get("derived"), "quant_report.artifacts.derived").get("path"), derived_path, common_anchors),
            "correctness_fp32_model": _recorded_path_matches(_mapping(_mapping(correctness.get("artifacts"), "correctness.artifacts").get("fp32"), "correctness.artifacts.fp32").get("model_path"), frozen.source_model_path, common_anchors),
            "correctness_int8_model": _recorded_path_matches(_mapping(_mapping(correctness.get("artifacts"), "correctness.artifacts").get("int8"), "correctness.artifacts.int8").get("model_path"), derived_path, common_anchors),
            "fp32_benchmark_model": _recorded_path_matches(_mapping(fp32_benchmark.get("model"), "fp32_benchmark.model").get("path"), frozen.source_model_path, common_anchors),
            "int8_benchmark_model": _recorded_path_matches(_mapping(int8_benchmark.get("model"), "int8_benchmark.model").get("path"), derived_path, common_anchors),
            "fp32_benchmark_sample": _recorded_path_matches(_mapping(fp32_benchmark.get("sample"), "fp32_benchmark.sample").get("image_path"), frozen.benchmark_sample_path, common_anchors),
            "int8_benchmark_sample": _recorded_path_matches(_mapping(int8_benchmark.get("sample"), "int8_benchmark.sample").get("image_path"), frozen.benchmark_sample_path, common_anchors),
            "fp32_profile_model": _recorded_path_matches(_mapping(fp32_profile.get("artifact"), "fp32_profile.artifact").get("model_path"), frozen.source_model_path, common_anchors),
            "int8_profile_model": _recorded_path_matches(_mapping(int8_profile.get("artifact"), "int8_profile.artifact").get("model_path"), derived_path, common_anchors),
            "fp32_profile_sample": _recorded_path_matches(_mapping(fp32_profile.get("protocol_binding"), "fp32_profile.protocol_binding").get("sample_path"), frozen.benchmark_sample_path, common_anchors),
            "int8_profile_sample": _recorded_path_matches(_mapping(int8_profile.get("protocol_binding"), "int8_profile.protocol_binding").get("sample_path"), frozen.benchmark_sample_path, common_anchors),
        }
        trace_integrity = {}
        for precision, summary_name in (("fp32", "fp32_profile_summary"), ("int8", "int8_profile_summary")):
            summary = documents[summary_name]
            trace = _mapping(summary.get("trace"), f"{precision}_profile_summary.trace")
            anchors = common_anchors + (resolved_inputs[summary_name].parent,)
            trace_integrity[precision] = _resolve_trace(
                trace.get("path"), trace.get("sha256"), trace.get("size_bytes"), anchors, f"{precision}_trace"
            )

        input_records = {name: _input_record(path) for name, path in resolved_inputs.items()}
        acceptance = assemble_documents(
            documents["protocol"],
            protocol_hash,
            quant,
            correctness,
            fp32_benchmark,
            int8_benchmark,
            documents["benchmark_comparison"],
            fp32_profile,
            int8_profile,
            integrity=integrity,
            trace_integrity=trace_integrity,
            path_bindings=path_bindings,
            input_records=input_records,
            correctness_policy=arguments.correctness_policy,
        )
        write_json(arguments.output, acceptance)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        "S2-01 completion evidence: "
        f"passed={acceptance['passed']}; "
        f"strict_acceptance_passed={acceptance['strict_acceptance_passed']}; "
        f"source_sha={acceptance['lineage']['source_fp32']['sha256']}; "
        f"derived_sha={acceptance['lineage']['derived_int8']['sha256']}; "
        f"speed_gate={acceptance['policy']['int8_speed_is_pass_gate']}; "
        f"output={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

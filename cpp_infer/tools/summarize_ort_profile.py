#!/usr/bin/env python3
"""Summarize one ONNX Runtime Chrome trace without treating it as a benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import s2_01_protocol  # noqa: E402


SCHEMA_VERSION = 1
KERNEL_SUFFIX = "_kernel_time"
SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


class ProfileSummaryError(RuntimeError):
    """Raised when a trace cannot support the frozen S2-01 evidence schema."""


def fail(object_name: str, expected: str, actual: str, action: str) -> NoReturn:
    raise ProfileSummaryError(
        "ORT profile summary failed: "
        f"object={object_name}; expected={expected}; actual={actual}; action={action}"
    )


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(
                "profile_json",
                "unique JSON object keys",
                f"duplicate key {key!r}",
                "regenerate the trace with the pinned ONNX Runtime",
            )
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: fail(
                    "profile_json.number",
                    "RFC-compliant finite JSON numbers",
                    token,
                    "regenerate the trace and do not accept NaN or Infinity",
                ),
            )
    except ProfileSummaryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(
            "profile_json.path",
            "one readable UTF-8 ONNX Runtime Chrome trace",
            f"{path}: {error}",
            "pass the exact path returned by Session::EndProfilingAllocated",
        )


def _finite_non_negative(value: Any, object_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(object_name, "a finite non-negative number", repr(value), "inspect the trace event")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        fail(object_name, "a finite non-negative number", repr(value), "inspect the trace event")
    return converted


def _nonempty_string(value: Any, object_name: str) -> str:
    if not isinstance(value, str) or not value:
        fail(object_name, "a non-empty string", repr(value), "inspect the trace event args")
    return value


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _rank_rows(
    totals: Mapping[Tuple[str, ...], Mapping[str, Any]],
    kernel_total_us: float,
    field_names: Sequence[str],
    top_n: int,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    rows: List[MutableMapping[str, Any]] = []
    for key, values in totals.items():
        total_us = float(values["total_us"])
        row: MutableMapping[str, Any] = {
            field: key[index] for index, field in enumerate(field_names)
        }
        row.update(
            {
                "calls": int(values["calls"]),
                "total_ms": total_us / 1000.0,
                "mean_ms": total_us / max(1, int(values["calls"])) / 1000.0,
                "percentage": (100.0 * total_us / kernel_total_us),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["total_ms"]),
            *(str(row[field]) for field in field_names),
        )
    )
    cumulative = 0.0
    for rank, row in enumerate(rows, start=1):
        cumulative += float(row["percentage"])
        row["rank"] = rank
        row["cumulative_percentage"] = min(100.0, cumulative)
    return rows, rows[:top_n]


def summarize_trace(
    trace_path: Path,
    *,
    model_id: str,
    declared_model_sha256: str,
    precision: str,
    expected_provider: str,
    expected_profile_runs: int,
    top_n: int,
    protocol_binding: Optional[Mapping[str, Any]] = None,
    artifact_evidence: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if top_n < 1 or top_n > 1000:
        fail("top_n", "an integer in [1,1000]", str(top_n), "choose a bounded summary size")
    if expected_profile_runs < 1:
        fail(
            "expected_profile_runs",
            "a positive integer",
            str(expected_profile_runs),
            "record the exact number of profiled Session::Run calls",
        )
    if precision not in {"fp32", "int8"}:
        fail("precision", "fp32 or int8", repr(precision), "pass the artifact precision")
    if not trace_path.resolve().is_file():
        fail(
            "profile_json.path",
            "an existing regular file",
            str(trace_path),
            "pass the exact path printed by the C++ profile command",
        )

    resolved_trace = trace_path.resolve()
    document = load_json(resolved_trace)
    return summarize_events(
        document,
        trace_path=resolved_trace,
        model_id=model_id,
        declared_model_sha256=declared_model_sha256,
        precision=precision,
        expected_provider=expected_provider,
        expected_profile_runs=expected_profile_runs,
        top_n=top_n,
        trace_size_bytes=resolved_trace.stat().st_size,
        trace_sha256=hashlib.sha256(resolved_trace.read_bytes()).hexdigest().upper(),
        protocol_binding=protocol_binding,
        artifact_evidence=artifact_evidence,
    )


def summarize_events(
    document: Any,
    *,
    trace_path: Path,
    model_id: str,
    declared_model_sha256: str,
    precision: str,
    expected_provider: str,
    expected_profile_runs: int,
    top_n: int,
    trace_size_bytes: Optional[int] = None,
    trace_sha256: Optional[str] = None,
    protocol_binding: Optional[Mapping[str, Any]] = None,
    artifact_evidence: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    if not isinstance(document, list):
        fail(
            "profile_json.root",
            "a Chrome trace JSON array",
            type(document).__name__,
            "use the raw trace returned by ONNX Runtime rather than a summary",
        )
    if not model_id:
        fail(
            "model_id",
            "a non-empty artifact identifier",
            repr(model_id),
            "pass the selected artifact model_id",
        )
    if not SHA256_PATTERN.fullmatch(declared_model_sha256):
        fail(
            "declared_model_sha256",
            "64 hexadecimal characters",
            repr(declared_model_sha256),
            "pass the hash from the validated artifact contract",
        )

    node_totals: MutableMapping[Tuple[str, str, str], MutableMapping[str, Any]] = defaultdict(
        lambda: {"calls": 0, "total_us": 0.0}
    )
    operator_totals: MutableMapping[Tuple[str, str], MutableMapping[str, Any]] = defaultdict(
        lambda: {"calls": 0, "total_us": 0.0}
    )
    provider_totals: MutableMapping[Tuple[str], MutableMapping[str, Any]] = defaultdict(
        lambda: {"calls": 0, "total_us": 0.0}
    )
    ignored_node_events = 0
    kernel_events = 0
    kernel_total_us = 0.0
    session_model_run_events = 0

    for index, event in enumerate(document):
        if not isinstance(event, dict):
            continue
        if (
            event.get("cat") == "Session"
            and event.get("ph") == "X"
            and event.get("name") == "model_run"
        ):
            session_model_run_events += 1
        if event.get("cat") != "Node":
            continue
        name = event.get("name")
        if (
            event.get("ph") != "X"
            or not isinstance(name, str)
            or not name.endswith(KERNEL_SUFFIX)
        ):
            ignored_node_events += 1
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            fail(
                f"profile_json[{index}].args",
                "an object with op_name and provider",
                repr(args),
                "regenerate the trace with ORT profiling enabled",
            )
        op_type = _nonempty_string(args.get("op_name"), f"profile_json[{index}].args.op_name")
        provider = _nonempty_string(args.get("provider"), f"profile_json[{index}].args.provider")
        duration_us = _finite_non_negative(event.get("dur"), f"profile_json[{index}].dur")
        node_name = name[: -len(KERNEL_SUFFIX)]
        if not node_name:
            fail(
                f"profile_json[{index}].name",
                "a non-empty node name before _kernel_time",
                repr(name),
                "inspect the generated trace",
            )
        kernel_events += 1
        kernel_total_us += duration_us
        for totals, key in (
            (node_totals, (node_name, op_type, provider)),
            (operator_totals, (op_type, provider)),
            (provider_totals, (provider,)),
        ):
            totals[key]["calls"] += 1
            totals[key]["total_us"] += duration_us

    if kernel_events == 0 or kernel_total_us <= 0.0:
        fail(
            "profile_json.kernel_events",
            "at least one positive-duration Node *_kernel_time event",
            f"events={kernel_events}, total_us={kernel_total_us}",
            "verify profiling was enabled and at least one Session::Run completed",
        )
    if session_model_run_events != expected_profile_runs:
        fail(
            "profile_json.session_model_run_events",
            f"exactly {expected_profile_runs} completed Session model_run events",
            str(session_model_run_events),
            "regenerate the trace with the frozen --profile-runs value",
        )
    unexpected_providers = sorted(
        provider[0] for provider in provider_totals if provider[0] != expected_provider
    )
    if unexpected_providers:
        fail(
            "profile_json.provider_placement",
            f"all node kernel events on {expected_provider}",
            repr(unexpected_providers),
            "inspect provider registration and the actual per-node placement",
        )

    all_nodes, top_nodes = _rank_rows(
        node_totals, kernel_total_us, ("node_name", "op_type", "provider"), top_n
    )
    all_operators, top_operators = _rank_rows(
        operator_totals, kernel_total_us, ("op_type", "provider"), top_n
    )
    all_providers, _ = _rank_rows(
        provider_totals, kernel_total_us, ("provider",), top_n
    )
    operator_types = {row["op_type"] for row in all_operators}
    quantization_operator_types = {
        "QLinearConv",
        "QuantizeLinear",
        "DequantizeLinear",
    }
    observed_quantization_operator_types = sorted(
        operator_types & quantization_operator_types
    )
    precision_signature_verified = False
    if artifact_evidence:
        if precision == "fp32" and observed_quantization_operator_types:
            fail(
                "profile_json.precision_signature",
                "no QLinearConv/QuantizeLinear/DequantizeLinear operators for the FP32 artifact",
                repr(observed_quantization_operator_types),
                "pass the FP32 trace returned by the matching profile process",
            )
        if precision == "int8":
            missing_quantization_operators = sorted(
                quantization_operator_types - operator_types
            )
            if missing_quantization_operators:
                fail(
                    "profile_json.precision_signature",
                    "QLinearConv, QuantizeLinear, and DequantizeLinear operators for the QDQ INT8 artifact",
                    f"missing={missing_quantization_operators!r}",
                    "pass the INT8 trace returned by the matching profile process",
                )
        precision_signature_verified = True
    node_call_counts = [int(row["calls"]) for row in all_nodes]
    mismatched_call_counts = [
        {"node_name": row["node_name"], "actual_calls": int(row["calls"])}
        for row in all_nodes
        if int(row["calls"]) != expected_profile_runs
    ]
    if mismatched_call_counts:
        fail(
            "profile_json.node_call_counts",
            f"every executed node called exactly {expected_profile_runs} times",
            repr(mismatched_call_counts[:5]),
            "regenerate the trace with the frozen --profile-runs value and one deterministic static-shape input",
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "onnxruntime_node_profile_summary",
        "passed": True,
        "trace": {
            "path": _display_path(trace_path),
            "size_bytes": trace_size_bytes,
            "sha256": trace_sha256,
            "event_count": len(document),
            "session_model_run_event_count": session_model_run_events,
            "node_kernel_event_count": kernel_events,
            "ignored_non_kernel_node_event_count": ignored_node_events,
            "duration_unit_in_trace": "microseconds",
        },
        "model": {
            "model_id": model_id,
            "declared_sha256": declared_model_sha256.upper(),
            "precision": precision,
            "trace_precision_signature": {
                "verified": precision_signature_verified,
                "method": "optimized_graph_operator_inventory",
                "observed_quantization_operator_types": observed_quantization_operator_types,
                "scope": (
                    "Prevents accidental FP32/INT8 trace swaps; the raw ORT trace does not embed the model file SHA-256."
                ),
            },
        },
        "artifact": dict(artifact_evidence) if artifact_evidence else None,
        "protocol_binding": dict(protocol_binding) if protocol_binding else None,
        "protocol": {
            "expected_profile_runs": expected_profile_runs,
            "event_filter": "cat == Node and ph == X and name endswith _kernel_time",
            "aggregation_duration_unit": "milliseconds",
            "expected_provider": expected_provider,
            "top_n": top_n,
        },
        "result": {
            "kernel_event_total_ms": kernel_total_us / 1000.0,
            "unique_node_count": len(all_nodes),
            "unique_operator_provider_count": len(all_operators),
            "node_call_count_min": min(node_call_counts),
            "node_call_count_max": max(node_call_counts),
            "providers": all_providers,
            "top_nodes": top_nodes,
            "top_operators": top_operators,
            "all_operators": all_operators,
        },
        "segmented_benchmark_mapping": {
            "outer_metric": "latency_ms.session_run",
            "relationship": (
                "Node kernel durations are diagnostic components within profiled "
                "Session::Run calls; their sum is not substituted for wall-clock benchmark latency."
            ),
            "excluded_from_formal_benchmark": True,
        },
        "profiling_overhead": {
            "present": True,
            "quantified": False,
            "policy": "diagnostic_trace_durations_are_never_used_as_formal_latency",
            "formal_latency_source": "separate C++ --benchmark process with profiling_enabled=false",
        },
        "limitations": [
            "Profiling adds instrumentation overhead, so trace durations are not formal latency evidence.",
            "ORT_ENABLE_ALL profiles the optimized execution graph; fused runtime nodes need not match the original ONNX node list.",
            "Summed node kernel durations can differ from outer Session::Run wall time because scheduling, allocation, framework work, and overlap are not identical scopes.",
            "CPUExecutionProvider placement does not by itself prove that every kernel used integer instructions; combine this trace with the QDQ graph inventory.",
            "All profiled calls in the trace contribute to call counts and aggregate duration.",
        ],
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def write_json(path: Path, document: Mapping[str, Any], overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        fail(
            "summary_json.path",
            "a path that does not exist",
            f"{path} already exists",
            "choose a fresh path or pass --overwrite explicitly",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_artifact_evidence(path: Path) -> Mapping[str, Any]:
    declaration_path = path.resolve(strict=True)
    fields: Dict[str, str] = {}
    for line_number, raw_line in enumerate(
        declaration_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            fail(
                f"artifact[{line_number}]",
                "key = value",
                repr(raw_line),
                "restore the ModelArtifactSpec",
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value or key in fields:
            fail(
                f"artifact[{line_number}]",
                "one unique non-empty key/value",
                repr(raw_line),
                "restore the ModelArtifactSpec",
            )
        fields[key] = value
    for key in ("model_id", "model_path", "model_sha256"):
        if key not in fields:
            fail(
                f"artifact.{key}",
                "a declared value",
                "missing",
                "use the validated artifact contract",
            )
    declared_sha = fields["model_sha256"]
    if not SHA256_PATTERN.fullmatch(declared_sha):
        fail(
            "artifact.model_sha256",
            "64 hexadecimal characters",
            repr(declared_sha),
            "restore the derived artifact digest",
        )
    model_path = Path(fields["model_path"])
    if model_path.is_absolute():
        fail(
            "artifact.model_path",
            "a declaration-relative portable path",
            str(model_path),
            "remove the machine-specific absolute path",
        )
    model_path = (declaration_path.parent / model_path).resolve(strict=True)
    raw_model = model_path.read_bytes()
    actual_sha = hashlib.sha256(raw_model).hexdigest().upper()
    if actual_sha != declared_sha.upper():
        fail(
            "artifact.model_sha256",
            declared_sha.upper(),
            actual_sha,
            "restore the artifact/model pair before profiling",
        )
    raw_declaration = declaration_path.read_bytes()
    canonical_declaration = raw_declaration.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return {
        "path": _display_path(declaration_path),
        "canonical_lf_sha256": hashlib.sha256(canonical_declaration).hexdigest().upper(),
        "model_id": fields["model_id"],
        "model_path": _display_path(model_path),
        "model_sha256": actual_sha,
        "model_size_bytes": len(raw_model),
    }


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--precision", required=True, choices=("fp32", "int8"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        frozen_protocol = s2_01_protocol.load_s2_01_protocol(arguments.protocol)
        artifact = load_artifact_evidence(arguments.artifact)
        expected_model_path = (
            frozen_protocol.source_model_path
            if arguments.precision == "fp32"
            else frozen_protocol.output_model_path
        )
        if Path(artifact["model_path"]).resolve() != expected_model_path:
            fail(
                "artifact.model_path",
                str(expected_model_path),
                artifact["model_path"],
                "select the artifact bound to the frozen precision",
            )
        protocol_binding = {
            "protocol_id": frozen_protocol.protocol_id,
            "path": _display_path(frozen_protocol.declaration_path),
            "canonical_lf_sha256": s2_01_protocol.sha256_file_canonical_lf(
                frozen_protocol.declaration_path
            ),
            "profile_runs": frozen_protocol.profiling["runs"],
            "provider": frozen_protocol.profiling["execution_provider"],
            "sample_path": _display_path(frozen_protocol.benchmark_sample_path),
            "sample_sha256": frozen_protocol.benchmark["sample"]["image_sha256"],
            "separate_from_formal_benchmark": True,
        }
        summary = summarize_trace(
            arguments.trace,
            model_id=artifact["model_id"],
            declared_model_sha256=artifact["model_sha256"],
            precision=arguments.precision,
            expected_provider=frozen_protocol.profiling["execution_provider"],
            expected_profile_runs=frozen_protocol.profiling["runs"],
            top_n=arguments.top_n,
            protocol_binding=protocol_binding,
            artifact_evidence=artifact,
        )
        write_json(arguments.output, summary, arguments.overwrite)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        "ORT profile summary: "
        f"precision={arguments.precision}, "
        f"kernel_ms={summary['result']['kernel_event_total_ms']:.6f}, "
        f"nodes={summary['result']['unique_node_count']}, "
        f"output={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

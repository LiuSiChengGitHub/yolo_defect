#!/usr/bin/env python3
"""Summarize an ORT TensorRT profile and require real TensorRT node execution."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "s2_04_ort_tensorrt_profile_summary"
DEFAULT_TRT_PROVIDER = "TensorrtExecutionProvider"
DEFAULT_CUDA_PROVIDER = "CUDAExecutionProvider"
DEFAULT_CPU_PROVIDER = "CPUExecutionProvider"
FROZEN_MODEL_ID = "yolov8n_neu_det_final_train_2"
FROZEN_MODEL_SHA256 = "7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68"


class ProfileSummaryError(RuntimeError):
    """An actionable malformed-trace or missing-execution failure."""


def fail(object_name: str, expected: str, actual: Any, action: str) -> NoReturn:
    raise ProfileSummaryError(
        "S2-04 ORT profile validation failed: "
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
    fail("json.number", "a finite number", value, "replace NaN or Infinity")


def load_trace(path: Path) -> Sequence[Mapping[str, Any]]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail("trace", "one readable ORT JSON trace", str(error), "fix the trace path, encoding, or JSON syntax")
    if not isinstance(value, list) or any(not isinstance(event, dict) for event in value):
        fail("trace", "an array of event objects", type(value).__name__, "pass the file returned by Ort::EndProfiling")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail("artifact.model_path", "a readable ONNX file", str(error), "restore the frozen current model")
    return digest.hexdigest().upper()


def load_frozen_artifact(path: Path) -> Mapping[str, Any]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        fail("artifact_spec", "readable UTF-8 key=value text", str(error), "pass the current artifact declaration")
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            fail(f"{path}:{line_number}", "key = value", line, "fix the artifact declaration")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key or not value or key in values:
            fail(f"{path}:{line_number}", "one unique non-empty key/value", line, "fix the artifact declaration")
        values[key] = value
    if values.get("model_id") != FROZEN_MODEL_ID or str(values.get("model_sha256", "")).upper() != FROZEN_MODEL_SHA256:
        fail("artifact.identity", f"model_id={FROZEN_MODEL_ID}, sha256={FROZEN_MODEL_SHA256}", {"model_id": values.get("model_id"), "sha256": values.get("model_sha256")}, "use the frozen current YOLO artifact")
    raw_model_path = values.get("model_path")
    if not raw_model_path:
        fail("artifact.model_path", "a non-empty path", raw_model_path, "restore the artifact declaration")
    model_path = (path.resolve().parent / raw_model_path).resolve()
    actual_sha = sha256_file(model_path)
    if actual_sha != FROZEN_MODEL_SHA256:
        fail("artifact.model_sha256", FROZEN_MODEL_SHA256, actual_sha, "use the declared current ONNX bytes")
    return {
        "artifact_spec_path": str(path.resolve()),
        "model_path": str(model_path),
        "model_id": FROZEN_MODEL_ID,
        "model_sha256": FROZEN_MODEL_SHA256,
    }


def duration_microseconds(event: Mapping[str, Any], object_name: str) -> float:
    value = event.get("dur")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(object_name, "a finite non-negative dur in microseconds", value, "use complete-duration ORT events")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        fail(object_name, "a finite non-negative dur in microseconds", value, "use an unmodified ORT trace")
    return result


def is_kernel_node_event(event: Mapping[str, Any]) -> bool:
    if event.get("cat") != "Node" or event.get("ph") != "X":
        return False
    name = event.get("name")
    arguments = event.get("args")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return False
    provider = arguments.get("provider")
    if not isinstance(provider, str) or not provider:
        return False
    return name.endswith("_kernel_time")


def provider_row(provider: str, events: Sequence[Mapping[str, Any]], total_us: float) -> Mapping[str, Any]:
    duration_us = sum(float(event["duration_us"]) for event in events)
    operator_calls: Dict[str, int] = defaultdict(int)
    operator_duration_us: Dict[str, float] = defaultdict(float)
    for event in events:
        operator = str(event["op_type"])
        operator_calls[operator] += 1
        operator_duration_us[operator] += float(event["duration_us"])
    operators = [
        {
            "op_type": operator,
            "calls": operator_calls[operator],
            "total_duration_ms": operator_duration_us[operator] / 1000.0,
        }
        for operator in sorted(
            operator_calls,
            key=lambda value: (-operator_duration_us[value], value),
        )
    ]
    return {
        "provider": provider,
        "calls": len(events),
        "unique_node_count": len({str(event["name"]) for event in events}),
        "total_duration_ms": duration_us / 1000.0,
        "mean_duration_ms": duration_us / (1000.0 * len(events)) if events else 0.0,
        "percentage_of_node_duration": 100.0 * duration_us / total_us if total_us > 0.0 else 0.0,
        "operators": operators,
    }


def summarize_events(
    events: Sequence[Mapping[str, Any]],
    *,
    trace_path: Path,
    model_id: str,
    model_sha256: str,
    expected_profile_runs: Optional[int],
    tensorrt_provider: str = DEFAULT_TRT_PROVIDER,
    cuda_provider: str = DEFAULT_CUDA_PROVIDER,
    cpu_provider: str = DEFAULT_CPU_PROVIDER,
    top_n: int = 20,
) -> Mapping[str, Any]:
    if model_id != FROZEN_MODEL_ID:
        fail("model_id", FROZEN_MODEL_ID, model_id, "derive identity from the frozen artifact declaration")
    normalized_sha = model_sha256.upper()
    if normalized_sha != FROZEN_MODEL_SHA256:
        fail("model_sha256", FROZEN_MODEL_SHA256, model_sha256, "derive identity from the frozen artifact declaration")
    if expected_profile_runs is not None and expected_profile_runs <= 0:
        fail("expected_profile_runs", "a positive integer", expected_profile_runs, "pass the profile runner's run count")
    if top_n <= 0:
        fail("top_n", "a positive integer", top_n, "choose at least one top node")
    provider_names = [tensorrt_provider, cuda_provider, cpu_provider]
    expected_provider_names = [DEFAULT_TRT_PROVIDER, DEFAULT_CUDA_PROVIDER, DEFAULT_CPU_PROVIDER]
    if provider_names != expected_provider_names:
        fail("provider_names", str(expected_provider_names), provider_names, "use exact ORT provider identities; provider meaning is not user-redefinable")

    model_run_events = [
        event
        for event in events
        if event.get("cat") == "Session"
        and event.get("ph") == "X"
        and event.get("name") == "model_run"
    ]
    if expected_profile_runs is not None and len(model_run_events) != expected_profile_runs:
        fail("session_model_run_events", str(expected_profile_runs), len(model_run_events), "profile exactly the declared number of product runs")

    normalized_events: List[Mapping[str, Any]] = []
    ignored_node_events = 0
    for index, event in enumerate(events):
        if event.get("cat") == "Node" and event.get("ph") == "X" and not is_kernel_node_event(event):
            ignored_node_events += 1
        if not is_kernel_node_event(event):
            continue
        arguments = event["args"]
        normalized_events.append(
            {
                "name": event["name"],
                "provider": arguments["provider"],
                "op_type": arguments.get("op_name", "<unknown>"),
                "duration_us": duration_microseconds(event, f"events[{index}].dur"),
            }
        )
    if not normalized_events:
        fail("node_events", "at least one provider-attributed Node complete event", 0, "enable ORT profiling around real inference runs")
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event in normalized_events:
        grouped[str(event["provider"])].append(event)
    if not grouped[tensorrt_provider]:
        fail("tensorrt_node_events", f">=1 Node event with provider={tensorrt_provider}", 0, "inspect TensorRT EP registration, partitioning, and fallback diagnostics")
    tensorrt_duration_us = sum(float(event["duration_us"]) for event in grouped[tensorrt_provider])
    if tensorrt_duration_us <= 0.0:
        fail("tensorrt_node_duration", "positive aggregate TensorRT duration", tensorrt_duration_us, "capture real TensorRT kernel execution rather than a zero-duration marker")
    total_us = sum(float(event["duration_us"]) for event in normalized_events)
    if total_us <= 0.0:
        fail("node_duration", "positive aggregate duration", total_us, "capture a non-empty real inference profile")

    ordered_providers = provider_names + sorted(set(grouped) - set(provider_names))
    provider_rows = [provider_row(provider, grouped.get(provider, []), total_us) for provider in ordered_providers]
    top_nodes = sorted(
        (
            {
                "name": event["name"],
                "op_type": event["op_type"],
                "provider": event["provider"],
                "duration_ms": float(event["duration_us"]) / 1000.0,
            }
            for event in normalized_events
        ),
        key=lambda row: (-row["duration_ms"], row["provider"], row["name"]),
    )[:top_n]
    provider_by_name = {row["provider"]: row for row in provider_rows}
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "passed": True,
        "trace": {
            "path": str(trace_path.resolve()),
            "event_count": len(events),
            "session_model_run_event_count": len(model_run_events),
            "expected_profile_runs": expected_profile_runs,
            "provider_attributed_node_event_count": len(normalized_events),
            "ignored_non_kernel_node_event_count": ignored_node_events,
            "aggregate_node_duration_ms": total_us / 1000.0,
            "duration_unit_in_source": "microseconds",
        },
        "model": {
            "model_id": model_id,
            "declared_sha256": normalized_sha,
            "precision": "fp16",
        },
        "execution_proof": {
            "tensorrt_provider": tensorrt_provider,
            "tensorrt_node_event_count": provider_by_name[tensorrt_provider]["calls"],
            "tensorrt_node_duration_ms": provider_by_name[tensorrt_provider]["total_duration_ms"],
            "real_tensorrt_node_execution_observed": True,
            "cuda_fallback_node_event_count": provider_by_name[cuda_provider]["calls"],
            "cpu_fallback_node_event_count": provider_by_name[cpu_provider]["calls"],
            "fallback_is_allowed_but_reported": True,
        },
        "providers": provider_rows,
        "top_nodes": top_nodes,
        "limitations": [
            "ORT node durations are profiling measurements and are not formal benchmark latency.",
            "Provider-attributed Node events prove ORT TensorRT EP execution; they do not imply that every graph node ran in TensorRT.",
        ],
    }


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


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-spec", required=True, type=Path)
    parser.add_argument("--expected-profile-runs", required=True, type=int)
    parser.add_argument("--tensorrt-provider", default=DEFAULT_TRT_PROVIDER)
    parser.add_argument("--cuda-provider", default=DEFAULT_CUDA_PROVIDER)
    parser.add_argument("--cpu-provider", default=DEFAULT_CPU_PROVIDER)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        artifact = load_frozen_artifact(arguments.artifact_spec)
        summary = summarize_events(
            load_trace(arguments.trace),
            trace_path=arguments.trace,
            model_id=artifact["model_id"],
            model_sha256=artifact["model_sha256"],
            expected_profile_runs=arguments.expected_profile_runs,
            tensorrt_provider=arguments.tensorrt_provider,
            cuda_provider=arguments.cuda_provider,
            cpu_provider=arguments.cpu_provider,
            top_n=arguments.top_n,
        )
        summary = dict(summary)
        summary["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        summary["artifact"] = artifact
        write_json(arguments.output, summary, arguments.overwrite)
    except ProfileSummaryError as error:
        print(str(error), file=sys.stderr)
        return 1
    proof = summary["execution_proof"]
    print(
        "S2-04 ORT profile: passed=True, "
        f"trt_calls={proof['tensorrt_node_event_count']}, "
        f"cuda_fallback_calls={proof['cuda_fallback_node_event_count']}, "
        f"cpu_fallback_calls={proof['cpu_fallback_node_event_count']}, "
        f"output={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

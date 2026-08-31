#!/usr/bin/env python3
"""Run the existing C++ benchmark and add S2-04 GPU/cache evidence.

The product benchmark command is passed verbatim after ``--``.  This wrapper
does not reinterpret its timings: it embeds the resulting benchmark JSON,
extracts the existing initialization/latency/throughput/host-memory metrics,
samples GPU memory with ``nvidia-smi``, and inventories TensorRT cache files
before and after the run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
FROZEN_MODEL_ID = "yolov8n_neu_det_final_train_2"
FROZEN_MODEL_SHA256 = "7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68"
FROZEN_IMAGE_NAME = "crazing_241.jpg"
FROZEN_IMAGE_SHA256 = "1D65EF27EAA9BF27608D954DFE57B40E401FC1AED435884400F35E8000BBF98D"
FROZEN_WARMUP = 10
FROZEN_REPEAT = 100
TRT_PROVIDER = "TensorrtExecutionProvider"
REQUESTED_TRT_PROVIDER = "tensorrt"
NATIVE_TRT_PROVIDER = "TensorRTNative"
REQUESTED_NATIVE_TRT_PROVIDER = "tensorrt_native"
NATIVE_ENGINE_SHA256 = "E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746"
NATIVE_PRECISION_POLICY = "fp16_dfl_softmax_fp32_else_no_tf32"
TARGET_GPU_NAME = "NVIDIA GeForce RTX 4060 Laptop GPU"
EXPECTED_ORT_VERSION = "1.20.1"

BACKEND_CONTRACTS: Mapping[str, Mapping[str, str]] = {
    "ort_ep": {
        "requested_provider": REQUESTED_TRT_PROVIDER,
        "actual_provider": TRT_PROVIDER,
        "source_evidence_type": "cpp_ort_single_image_release_benchmark",
        "wrapper_evidence_type": "s2_04_tensorrt_fp16_gpu_benchmark",
        "precision": "fp16",
    },
    "native": {
        "requested_provider": REQUESTED_NATIVE_TRT_PROVIDER,
        "actual_provider": NATIVE_TRT_PROVIDER,
        "source_evidence_type": "cpp_native_tensorrt_single_image_release_benchmark",
        "wrapper_evidence_type": "s2_04_tensorrt_native_fp16_gpu_benchmark",
        "precision": "mixed_fp16_fp32",
    },
}


def expected_native_provider_evidence(engine_sha256: str = NATIVE_ENGINE_SHA256) -> str:
    return (
        "native_tensorrt_enqueue_v3;"
        f"precision_policy={NATIVE_PRECISION_POLICY};"
        f"declared_engine_sha256={engine_sha256};"
        f"actual_engine_sha256={engine_sha256};"
        "tensorrt_runtime=10.4.0;compiled_headers=10.4.0.26;"
        "cuda_runtime=12.6;compute_capability=8.9;fallback=none"
    )


class GpuBenchmarkError(RuntimeError):
    """An actionable wrapper, source benchmark, cache, or GPU metric failure."""


def fail(object_name: str, expected: str, actual: Any, action: str) -> NoReturn:
    raise GpuBenchmarkError(
        "S2-04 GPU benchmark failed: "
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


def load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail("benchmark_json", "one readable UTF-8 JSON object", str(error), "inspect the product benchmark failure")
    if not isinstance(value, dict):
        fail("benchmark_json", "an object root", type(value).__name__, "use the product benchmark JSON")
    return value


def finite_number(value: Any, object_name: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(object_name, "a finite number", value, "fix the source benchmark schema")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        fail(object_name, f"a finite number >= {minimum}" if minimum is not None else "a finite number", value, "fix the source benchmark schema")
    return result


def require_mapping(value: Any, object_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        fail(object_name, "an object", type(value).__name__, "fix the source benchmark schema")
    return value


def latency_statistics(value: Any, object_name: str, expected_sample_count: int) -> Mapping[str, Any]:
    row = require_mapping(value, object_name)
    sample_count = row.get("sample_count")
    if sample_count != expected_sample_count:
        fail(f"{object_name}.sample_count", str(expected_sample_count), sample_count, "run the frozen repeat count and retain every sample")
    return {
        "sample_count": sample_count,
        "mean_ms": finite_number(row.get("mean"), f"{object_name}.mean", minimum=0.0),
        "p50_ms": finite_number(row.get("p50"), f"{object_name}.p50", minimum=0.0),
        "p95_ms": finite_number(row.get("p95"), f"{object_name}.p95", minimum=0.0),
    }


def extract_benchmark_metrics(
    document: Mapping[str, Any],
    *,
    expected_requested_provider: str,
    expected_actual_provider: str,
    expected_source_evidence_type: Optional[str] = None,
    expected_provider_evidence: Optional[str] = None,
    precision: Optional[str] = None,
) -> Mapping[str, Any]:
    native = expected_requested_provider == REQUESTED_NATIVE_TRT_PROVIDER
    if expected_source_evidence_type is None:
        expected_source_evidence_type = (
            "cpp_native_tensorrt_single_image_release_benchmark"
            if native
            else "cpp_ort_single_image_release_benchmark"
        )
    if precision is None:
        precision = "mixed_fp16_fp32" if native else "fp16"
    if document.get("schema_version") != 1 or document.get("evidence_type") != expected_source_evidence_type:
        fail("benchmark.identity", f"schema v1 {expected_source_evidence_type}", {"schema": document.get("schema_version"), "type": document.get("evidence_type")}, "run the matching product --benchmark workflow")
    runtime = require_mapping(document.get("runtime"), "benchmark.runtime")
    if runtime.get("requested_provider") != expected_requested_provider:
        fail("benchmark.runtime.requested_provider", expected_requested_provider, runtime.get("requested_provider"), "use the TensorRT runtime config or correct the explicit expectation")
    if runtime.get("actual_provider") != expected_actual_provider:
        fail("benchmark.runtime.actual_provider", expected_actual_provider, runtime.get("actual_provider"), "inspect TensorRT EP registration and fallback diagnostics")
    provider_evidence = runtime.get("provider_evidence")
    if not isinstance(provider_evidence, str) or not provider_evidence:
        fail("benchmark.runtime.provider_evidence", "a non-empty execution claim", provider_evidence, "record provider registration evidence")
    if expected_provider_evidence is not None and provider_evidence != expected_provider_evidence:
        fail("benchmark.runtime.provider_evidence", expected_provider_evidence, provider_evidence, "execute the SHA-bound native engine directly without fallback")
    session = require_mapping(runtime.get("session"), "benchmark.runtime.session")
    initialization_ms = finite_number(session.get("initialization_ms"), "benchmark.runtime.session.initialization_ms", minimum=0.0)
    if session.get("profiling_enabled") is not False:
        fail("benchmark.runtime.session.profiling_enabled", "false", session.get("profiling_enabled"), "disable profiling for the formal benchmark")
    environment = require_mapping(document.get("environment"), "benchmark.environment")
    operating_system = require_mapping(environment.get("os"), "benchmark.environment.os")
    machine = require_mapping(environment.get("machine"), "benchmark.environment.machine")
    build = require_mapping(environment.get("build"), "benchmark.environment.build")
    if operating_system.get("name") != "Linux":
        fail("benchmark.environment.os.name", "Linux", operating_system.get("name"), "run formal TensorRT evidence on Linux x86_64")
    if str(machine.get("architecture", "")).lower() not in {"x86_64", "amd64"}:
        fail("benchmark.environment.machine.architecture", "x86_64", machine.get("architecture"), "run on the declared Linux x86_64 target")
    if build.get("type") != "Release":
        fail("benchmark.environment.build.type", "Release", build.get("type"), "benchmark an optimized Release build")
    if environment.get("onnxruntime_version") != EXPECTED_ORT_VERSION:
        fail("benchmark.environment.onnxruntime_version", EXPECTED_ORT_VERSION, environment.get("onnxruntime_version"), "use the frozen ORT GPU C++ SDK")
    protocol = require_mapping(document.get("protocol"), "benchmark.protocol")
    expected_protocol = {
        "batch_size": 1,
        "sample_count": 1,
        "warmup": FROZEN_WARMUP,
        "repeat": FROZEN_REPEAT,
        "clock": "std::chrono::steady_clock",
        "timing_unit": "milliseconds",
        "percentile_method": "empirical_nearest_rank_ceiling",
    }
    if protocol != expected_protocol:
        fail("benchmark.protocol", str(expected_protocol), protocol, "use the frozen formal benchmark protocol")
    latency = require_mapping(document.get("latency_ms"), "benchmark.latency_ms")
    latency_rows = {
        name: latency_statistics(latency.get(name), f"benchmark.latency_ms.{name}", FROZEN_REPEAT)
        for name in (
            "image_decode",
            "preprocess",
            "session_run",
            "postprocess",
            "pipeline",
            "end_to_end",
        )
    }
    throughput = require_mapping(document.get("throughput_images_per_second"), "benchmark.throughput_images_per_second")
    throughput_rows = {
        name: finite_number(throughput.get(name), f"benchmark.throughput_images_per_second.{name}", minimum=1.0e-300)
        for name in ("pipeline", "end_to_end")
    }
    memory = require_mapping(document.get("memory"), "benchmark.memory")
    if memory.get("status") != "supported" or memory.get("metric") != "peak_rss":
        fail("benchmark.memory", "supported Linux peak_rss", {"status": memory.get("status"), "metric": memory.get("metric")}, "run on Linux x86_64 with process RSS support")
    memory_bytes = memory.get("bytes")
    if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes <= 0:
        fail("benchmark.memory.bytes", "a positive integer", memory_bytes, "retain the Linux process peak-RSS query")
    memory_mib = finite_number(memory.get("mebibytes"), "benchmark.memory.mebibytes", minimum=1.0e-300)
    if not math.isclose(memory_mib, memory_bytes / (1024.0 * 1024.0), rel_tol=1.0e-9, abs_tol=1.0e-9):
        fail("benchmark.memory.mebibytes", "bytes/(1024*1024)", memory_mib, "fix the source benchmark memory conversion")
    model = require_mapping(document.get("model"), "benchmark.model")
    model_sha = str(model.get("declared_sha256", "")).upper()
    if model.get("model_id") != FROZEN_MODEL_ID or model_sha != FROZEN_MODEL_SHA256:
        fail("benchmark.model", f"model_id={FROZEN_MODEL_ID}, sha256={FROZEN_MODEL_SHA256}", {"model_id": model.get("model_id"), "sha256": model_sha}, "benchmark the frozen current YOLO artifact")
    return {
        "requested_provider": runtime["requested_provider"],
        "actual_provider": runtime["actual_provider"],
        "provider_evidence": provider_evidence,
        "precision": precision,
        "environment": {
            "os_name": "Linux",
            "architecture": machine["architecture"],
            "build_type": "Release",
            "onnxruntime_version": EXPECTED_ORT_VERSION,
        },
        "model_id": model.get("model_id"),
        "model_declared_sha256": model_sha,
        "warmup": FROZEN_WARMUP,
        "repeat": FROZEN_REPEAT,
        "session_initialization_ms": initialization_ms,
        "latency_ms": latency_rows,
        "throughput_images_per_second": throughput_rows,
        "host_peak_rss": {
            "bytes": memory_bytes,
            "mebibytes": memory_mib,
            "scope": memory.get("scope"),
        },
    }


def parse_memory_mib(value: str) -> Optional[float]:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:\s*MiB)?", stripped, flags=re.IGNORECASE)
    if match is None:
        return None
    result = float(match.group(1))
    return result if math.isfinite(result) and result >= 0.0 else None


def parse_process_memory_csv(
    text: str, selected_gpu_uuid: Optional[str] = None
) -> Mapping[int, float]:
    totals: Dict[int, float] = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) == 3:
            gpu_uuid, raw_pid, raw_memory = (value.strip() for value in row)
            if selected_gpu_uuid is not None and gpu_uuid != selected_gpu_uuid:
                continue
        elif len(row) == 2 and selected_gpu_uuid is None:
            raw_pid, raw_memory = (value.strip() for value in row)
        else:
            continue
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        memory = parse_memory_mib(raw_memory)
        if pid > 0 and memory is not None:
            totals[pid] = totals.get(pid, 0.0) + memory
    return totals


def parse_device_memory_csv(text: str) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for row in csv.reader(io.StringIO(text)):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != 4:
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue
        memory = parse_memory_mib(row[3])
        if index < 0 or memory is None:
            continue
        rows.append(
            {
                "index": index,
                "name": row[1].strip(),
                "uuid": row[2].strip(),
                "memory_used_mib": memory,
            }
        )
    return rows


def inventory_cache(directory: Path) -> List[Mapping[str, Any]]:
    if not directory.exists():
        return []
    if not directory.is_dir():
        fail("cache_directory", "a directory or absent path", str(directory), "fix the TensorRT cache setting")
    rows: List[Mapping[str, Any]] = []
    try:
        candidates = sorted((path for path in directory.rglob("*") if path.is_file()), key=lambda path: path.relative_to(directory).as_posix().encode("utf-8"))
        for path in candidates:
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size_bytes": stat.st_size,
                    "modified_time_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path, "cache_inventory.file"),
                }
            )
    except OSError as error:
        fail("cache_inventory", "readable cache metadata", str(error), "fix cache permissions and retry")
    return rows


def compare_cache_inventories(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    before_by_path = {str(row["relative_path"]): row for row in before}
    after_by_path = {str(row["relative_path"]): row for row in after}
    created = sorted(set(after_by_path) - set(before_by_path))
    removed = sorted(set(before_by_path) - set(after_by_path))
    shared = sorted(set(before_by_path) & set(after_by_path))
    modified = [
        path
        for path in shared
        if before_by_path[path]["size_bytes"] != after_by_path[path]["size_bytes"]
        or before_by_path[path]["modified_time_ns"] != after_by_path[path]["modified_time_ns"]
        or before_by_path[path].get("sha256") != after_by_path[path].get("sha256")
    ]
    unchanged = [path for path in shared if path not in set(modified)]
    if created or modified:
        state = "built_or_updated"
    elif after:
        state = "reused_without_file_change"
    else:
        state = "empty"
    return {
        "state": state,
        "created": created,
        "modified": modified,
        "removed": removed,
        "unchanged": unchanged,
    }


def engine_cache_gate_passed(
    inventory: Sequence[Mapping[str, Any]],
    *,
    expected_relative_path: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> bool:
    if expected_relative_path is not None or expected_sha256 is not None:
        if expected_relative_path is None or expected_sha256 is None:
            return False
        expected_sha256 = expected_sha256.upper()
        return any(
            row.get("relative_path") == expected_relative_path
            and isinstance(row.get("size_bytes"), int)
            and not isinstance(row.get("size_bytes"), bool)
            and int(row["size_bytes"]) > 0
            and row.get("sha256") == expected_sha256
            for row in inventory
        )
    return any(
        str(row.get("relative_path", "")).lower().endswith(".engine")
        and isinstance(row.get("size_bytes"), int)
        and not isinstance(row.get("size_bytes"), bool)
        and int(row["size_bytes"]) > 0
        and isinstance(row.get("sha256"), str)
        and re.fullmatch(r"[0-9A-F]{64}", str(row["sha256"])) is not None
        for row in inventory
    )


def select_gpu_memory_evidence(
    *,
    pid: int,
    process_samples: Sequence[Mapping[str, Any]],
    device_samples: Sequence[Mapping[str, Any]],
    baseline_devices: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
    interval_ms: int,
) -> Mapping[str, Any]:
    if process_samples:
        peak = max(process_samples, key=lambda row: float(row["memory_used_mib"]))
        selected_device = next(
            (row for row in baseline_devices if row["uuid"] == peak["gpu_uuid"]),
            None,
        )
        return {
            "supported": True,
            "metric": "nvidia_smi_process_used_gpu_memory",
            "scope": f"target benchmark process pid={pid} on selected gpu_uuid={peak['gpu_uuid']}",
            "selected_device": selected_device,
            "pid_specific_metric_used": True,
            "device_wide_fallback_used": False,
            "sample_interval_ms": interval_ms,
            "sample_count": len(process_samples),
            "peak_memory_used_mib": peak["memory_used_mib"],
            "peak_sample_elapsed_ms": peak["elapsed_ms"],
            "baseline_device_memory_used_mib": None,
            "peak_minus_baseline_mib": None,
            "samples": list(process_samples),
            "sampling_errors": list(errors),
        }
    flattened = [
        {
            "elapsed_ms": sample["elapsed_ms"],
            **device,
        }
        for sample in device_samples
        for device in sample["devices"]
    ]
    if flattened:
        peak = max(flattened, key=lambda row: float(row["memory_used_mib"]))
        matching_baseline = next(
            (row for row in baseline_devices if row["index"] == peak["index"]),
            None,
        )
        baseline = float(matching_baseline["memory_used_mib"]) if matching_baseline is not None else None
        return {
            "supported": True,
            "metric": "nvidia_smi_device_memory.used",
            "scope": f"device-wide GPU index={peak['index']} uuid={peak['uuid']}; may include unrelated processes",
            "selected_device": matching_baseline,
            "pid_specific_metric_used": False,
            "device_wide_fallback_used": True,
            "fallback_reason": "nvidia-smi compute-apps did not expose a numeric memory row for the target PID",
            "sample_interval_ms": interval_ms,
            "sample_count": len(flattened),
            "peak_memory_used_mib": peak["memory_used_mib"],
            "peak_sample_elapsed_ms": peak["elapsed_ms"],
            "baseline_device_memory_used_mib": baseline,
            "peak_minus_baseline_mib": max(0.0, float(peak["memory_used_mib"]) - baseline) if baseline is not None else None,
            "samples": list(device_samples),
            "sampling_errors": list(errors),
        }
    return {
        "supported": False,
        "metric": None,
        "scope": None,
        "selected_device": baseline_devices[0] if len(baseline_devices) == 1 else None,
        "pid_specific_metric_used": False,
        "device_wide_fallback_used": True,
        "sample_interval_ms": interval_ms,
        "sample_count": 0,
        "peak_memory_used_mib": None,
        "peak_sample_elapsed_ms": None,
        "baseline_device_memory_used_mib": None,
        "peak_minus_baseline_mib": None,
        "samples": [],
        "sampling_errors": list(errors),
        "reason": "neither PID-specific nor device-wide nvidia-smi memory samples were available",
    }


def gpu_memory_gate_passed(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("supported") is not True:
        return False
    peak = evidence.get("peak_memory_used_mib")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(float(peak)) or float(peak) <= 0.0:
        return False
    if evidence.get("device_wide_fallback_used") is True:
        delta = evidence.get("peak_minus_baseline_mib")
        return (
            not isinstance(delta, bool)
            and isinstance(delta, (int, float))
            and math.isfinite(float(delta))
            and float(delta) > 0.0
        )
    return evidence.get("pid_specific_metric_used") is True


class NvidiaSmiSampler:
    def __init__(
        self,
        executable: str,
        pid: int,
        gpu_index: int,
        interval_ms: int,
        selected_gpu_uuid: Optional[str] = None,
    ) -> None:
        self.executable = executable
        self.pid = pid
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.selected_gpu_uuid = selected_gpu_uuid
        self.started = time.monotonic()
        self.process_samples: List[Mapping[str, Any]] = []
        self.device_samples: List[Mapping[str, Any]] = []
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="s2-04-nvidia-smi", daemon=True)

    def _query(self, arguments: Sequence[str], label: str) -> Optional[str]:
        try:
            result = subprocess.run(
                [self.executable, *arguments],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self.errors.append(f"{label}: {error}")
            return None
        if result.returncode != 0:
            self.errors.append(f"{label}: exit={result.returncode}, stderr={result.stderr.strip()}")
            return None
        return result.stdout

    def query_devices(self) -> List[Mapping[str, Any]]:
        output = self._query(
            [
                f"--id={self.gpu_index}",
                "--query-gpu=index,name,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            "device_memory",
        )
        return parse_device_memory_csv(output) if output is not None else []

    def _sample_once(self) -> None:
        elapsed_ms = (time.monotonic() - self.started) * 1000.0
        output = self._query(
            [
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            "process_memory",
        )
        process_values = (
            parse_process_memory_csv(output, self.selected_gpu_uuid)
            if output is not None and self.selected_gpu_uuid is not None
            else {}
        )
        if self.pid in process_values:
            self.process_samples.append(
                {
                    "elapsed_ms": elapsed_ms,
                    "pid": self.pid,
                    "gpu_uuid": self.selected_gpu_uuid,
                    "memory_used_mib": process_values[self.pid],
                }
            )
            return
        devices = self.query_devices()
        if devices:
            self.device_samples.append({"elapsed_ms": elapsed_ms, "devices": devices})

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_ms / 1000.0)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            self.errors.append("sampler thread did not stop within 10 seconds")


def command_option_value(command: Sequence[str], option: str) -> Optional[str]:
    for index, argument in enumerate(command):
        if argument == option:
            return command[index + 1] if index + 1 < len(command) else None
        prefix = option + "="
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return None


def parse_key_value_declaration(path: Path) -> Mapping[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        fail("runtime_config", "readable UTF-8 key=value text", str(error), "fix the command's --config path")
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            fail(f"{path}:{line_number}", "key = value", line, "fix the runtime declaration")
        key, value = (part.strip() for part in stripped.split("=", 1))
        if not key or not value or key in values:
            fail(f"{path}:{line_number}", "one unique non-empty key/value", line, "fix the runtime declaration")
        values[key] = value
    return values


def sha256_file(path: Path, object_name: str = "artifact.model_path") -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail(object_name, "a readable regular file", str(error), "fix the evidence artifact path or permissions")
    return digest.hexdigest().upper()


def validate_frozen_artifact(path: Path) -> Mapping[str, Any]:
    values = parse_key_value_declaration(path)
    model_id = values.get("model_id")
    declared_sha = str(values.get("model_sha256", "")).upper()
    if model_id != FROZEN_MODEL_ID or declared_sha != FROZEN_MODEL_SHA256:
        fail("artifact.identity", f"model_id={FROZEN_MODEL_ID}, sha256={FROZEN_MODEL_SHA256}", {"model_id": model_id, "sha256": declared_sha}, "use the frozen current YOLO artifact")
    raw_model_path = values.get("model_path")
    if not raw_model_path:
        fail("artifact.model_path", "a non-empty path", raw_model_path, "restore the artifact declaration")
    model_path = (path.resolve().parent / raw_model_path).resolve()
    actual_sha = sha256_file(model_path)
    if actual_sha != FROZEN_MODEL_SHA256:
        fail("artifact.model_sha256", FROZEN_MODEL_SHA256, actual_sha, "benchmark the declared current ONNX bytes")
    return {
        "artifact_spec_path": str(path.resolve()),
        "model_path": str(model_path),
        "model_id": FROZEN_MODEL_ID,
        "declared_and_actual_sha256": FROZEN_MODEL_SHA256,
    }


def validate_tensorrt_runtime_config(
    path: Path,
    *,
    cache_directory: Path,
    expected_requested_provider: str,
    expected_device_id: int,
    expected_engine_sha256: Optional[str] = None,
) -> Mapping[str, Any]:
    if not path.is_file():
        fail("runtime_config", "an existing regular file", str(path), "fix the command's --config argument")
    values = parse_key_value_declaration(path)
    expected_values = {
        "schema_version": "2",
        "provider": expected_requested_provider,
        "precision": "fp16",
        "device_id": str(expected_device_id),
        "score_threshold": "0.25",
        "nms_threshold": "0.45",
    }
    for key, expected in expected_values.items():
        if values.get(key) != expected:
            fail(f"runtime_config.{key}", expected, values.get(key), "use the frozen TensorRT FP16 runtime config")
    for key in ("artifact_spec_path", "tensorrt_engine_cache_path"):
        if not values.get(key):
            fail(f"runtime_config.{key}", "a non-empty path", values.get(key), "restore the TensorRT config")
    declared_cache = (path.resolve().parent / values["tensorrt_engine_cache_path"]).resolve()
    if declared_cache != cache_directory.resolve():
        fail("runtime_config.tensorrt_engine_cache_path", str(cache_directory.resolve()), str(declared_cache), "make --cache-dir match the product config exactly")
    result: Dict[str, Any] = {
        "path": str(path.resolve()),
        "schema_version": 2,
        "provider": values["provider"],
        "precision": values["precision"],
        "device_id": expected_device_id,
        "artifact_spec_path": str((path.resolve().parent / values["artifact_spec_path"]).resolve()),
        "tensorrt_engine_cache_path": str(declared_cache),
    }
    native = expected_requested_provider == REQUESTED_NATIVE_TRT_PROVIDER
    if native:
        if expected_engine_sha256 is None:
            fail("runtime_config.native_engine_contract", "an explicit frozen engine SHA-256", None, "use the fixed native benchmark contract")
        declared_engine_sha = str(values.get("tensorrt_engine_sha256", "")).upper()
        if declared_engine_sha != expected_engine_sha256.upper():
            fail("runtime_config.tensorrt_engine_sha256", expected_engine_sha256.upper(), declared_engine_sha, "restore the frozen native config")
        raw_engine_path = values.get("tensorrt_engine_path")
        if not raw_engine_path:
            fail("runtime_config.tensorrt_engine_path", "a non-empty direct engine path", raw_engine_path, "restore the frozen native config")
        declared_engine_path = path.resolve().parent / raw_engine_path
        if declared_engine_path.is_symlink():
            fail("runtime_config.tensorrt_engine_path", "a direct regular file, not a symlink", str(declared_engine_path), "restore the exact engine bytes inside the cache namespace")
        engine_path = declared_engine_path.resolve()
        if not engine_path.is_file() or engine_path.parent != declared_cache:
            fail("runtime_config.tensorrt_engine_path", f"a regular file directly inside {declared_cache}", str(engine_path), "place the frozen engine directly in its dedicated cache namespace")
        actual_engine_sha = sha256_file(engine_path, "runtime_config.tensorrt_engine_path")
        if actual_engine_sha != expected_engine_sha256.upper():
            fail("runtime_config.tensorrt_engine_path SHA-256", expected_engine_sha256.upper(), actual_engine_sha, "restore the exact frozen engine bytes")
        result.update(
            {
                "tensorrt_engine_path": str(engine_path),
                "tensorrt_engine_relative_path": engine_path.relative_to(declared_cache).as_posix(),
                "declared_and_actual_engine_sha256": actual_engine_sha,
                "precision_policy": NATIVE_PRECISION_POLICY,
            }
        )
    elif "tensorrt_engine_path" in values or "tensorrt_engine_sha256" in values:
        fail("runtime_config.native_engine_fields", "absent for ORT TensorRT EP mode", {key: values.get(key) for key in ("tensorrt_engine_path", "tensorrt_engine_sha256") if key in values}, "use the ORT EP config or select native mode")
    return result


def file_fingerprint(path: Path) -> Optional[Tuple[int, int]]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError as error:
        fail("benchmark_json", "readable metadata", str(error), "fix output permissions")
    return stat.st_size, stat.st_mtime_ns


def reject_protected_destination(
    destination: Path,
    *,
    object_name: str,
    protected_files: Sequence[Path],
    protected_directory: Path,
) -> None:
    resolved = destination.resolve()
    exact = {path.resolve() for path in protected_files}
    inside_protected_directory = False
    try:
        resolved.relative_to(protected_directory.resolve())
        inside_protected_directory = True
    except ValueError:
        pass
    if resolved in exact or inside_protected_directory:
        fail(
            object_name,
            "a destination outside the config, artifact, model, input image, and TensorRT cache",
            str(resolved),
            "choose a dedicated evidence-output directory; --overwrite never authorizes replacing frozen inputs",
        )


def write_json(path: Path, document: Mapping[str, Any], overwrite: bool) -> None:
    destination = path.resolve()
    if destination.exists() and not overwrite:
        fail("combined_output", "a new path or --overwrite", str(destination), "choose a new evidence path")
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
        fail("combined_output", "an atomically writable JSON destination", str(error), "fix the output path or permissions")


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-mode", choices=sorted(BACKEND_CONTRACTS), default="ort_ep", help="Select one fixed, non-relabelable TensorRT evidence contract.")
    parser.add_argument("--benchmark-json", required=True, type=Path, help="Path the wrapped CLI receives via its own --benchmark-json option.")
    parser.add_argument("--output", required=True, type=Path, help="Combined S2-04 evidence JSON.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--expected-requested-provider")
    parser.add_argument("--expected-actual-provider")
    parser.add_argument("--expected-gpu-name", default=TARGET_GPU_NAME)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Product benchmark command after --.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        if not command:
            fail("command", "the product benchmark command after --", command, "append -- <yolo_defect_cpp> --benchmark ...")
        if "--benchmark" not in command:
            fail("command", "--benchmark", command, "invoke the existing formal benchmark mode")
        raw_runtime_config = command_option_value(command, "--config")
        if raw_runtime_config is None:
            fail("command", "--config <path>", command, "pass the frozen TensorRT FP16 runtime config")
        raw_command_benchmark_json = command_option_value(command, "--benchmark-json")
        if raw_command_benchmark_json is None:
            fail("command", "--benchmark-json <path>", command, "pass the same source path to wrapper and product CLI")
        command_benchmark_json = Path(raw_command_benchmark_json).resolve()
        benchmark_json = arguments.benchmark_json.resolve()
        if command_benchmark_json != benchmark_json:
            fail("command.--benchmark-json", str(benchmark_json), str(command_benchmark_json), "make wrapper and product CLI paths identical")
        if arguments.output.resolve() == benchmark_json:
            fail("output", "a path distinct from source benchmark JSON", str(arguments.output), "preserve the product benchmark as embedded source evidence")
        if arguments.output.exists() and not arguments.overwrite:
            fail("output", "a new path or --overwrite", str(arguments.output), "choose a new combined evidence path")
        if arguments.gpu_index < 0 or arguments.sample_interval_ms < 20:
            fail("sampling", "gpu_index >= 0 and sample_interval_ms >= 20", {"gpu_index": arguments.gpu_index, "interval": arguments.sample_interval_ms}, "fix wrapper arguments")
        backend_contract = BACKEND_CONTRACTS[arguments.backend_mode]
        expected_provider_values = {
            "requested": backend_contract["requested_provider"],
            "actual": backend_contract["actual_provider"],
        }
        actual_provider_values = {
            "requested": arguments.expected_requested_provider or expected_provider_values["requested"],
            "actual": arguments.expected_actual_provider or expected_provider_values["actual"],
        }
        if actual_provider_values != expected_provider_values:
            fail("provider expectations", str(expected_provider_values), actual_provider_values, "use exact TensorRT provider identities; provider meaning is not user-redefinable")
        if arguments.expected_gpu_name != TARGET_GPU_NAME:
            fail("expected_gpu_name", TARGET_GPU_NAME, arguments.expected_gpu_name, "use the frozen RTX 4060 Laptop target identity")

        cache_dir = arguments.cache_dir.resolve()
        runtime_config = validate_tensorrt_runtime_config(
            Path(raw_runtime_config).resolve(),
            cache_directory=cache_dir,
            expected_requested_provider=expected_provider_values["requested"],
            expected_device_id=arguments.gpu_index,
            expected_engine_sha256=(
                NATIVE_ENGINE_SHA256 if arguments.backend_mode == "native" else None
            ),
        )
        artifact = validate_frozen_artifact(Path(runtime_config["artifact_spec_path"]))
        raw_image_path = command_option_value(command, "--image")
        if raw_image_path is None:
            fail("command", "--image <path>", command, "benchmark the frozen single image explicitly")
        input_image_path = Path(raw_image_path).resolve()
        if not input_image_path.is_file():
            fail("command.--image", "an existing regular image", str(input_image_path), "restore the frozen benchmark sample")
        input_image_sha256 = sha256_file(input_image_path, "command.--image")
        if (
            input_image_path.name != FROZEN_IMAGE_NAME
            or input_image_sha256 != FROZEN_IMAGE_SHA256
        ):
            fail(
                "command.--image identity",
                f"{FROZEN_IMAGE_NAME} SHA-256 {FROZEN_IMAGE_SHA256}",
                {"name": input_image_path.name, "sha256": input_image_sha256},
                "benchmark the frozen single-image sample",
            )
        protected_files = [
            Path(raw_runtime_config),
            Path(runtime_config["artifact_spec_path"]),
            Path(artifact["model_path"]),
            input_image_path,
        ]
        if "tensorrt_engine_path" in runtime_config:
            protected_files.append(Path(runtime_config["tensorrt_engine_path"]))
        reject_protected_destination(
            benchmark_json,
            object_name="benchmark_json.destination",
            protected_files=protected_files,
            protected_directory=cache_dir,
        )
        reject_protected_destination(
            arguments.output,
            object_name="output.destination",
            protected_files=protected_files,
            protected_directory=cache_dir,
        )
        cache_before = inventory_cache(cache_dir)
        source_before = file_fingerprint(benchmark_json)
        baseline_probe = NvidiaSmiSampler(arguments.nvidia_smi, -1, arguments.gpu_index, arguments.sample_interval_ms)
        baseline_devices = baseline_probe.query_devices()
        baseline_errors = list(baseline_probe.errors)
        if len(baseline_devices) != 1 or baseline_devices[0]["name"] != TARGET_GPU_NAME:
            fail("selected_gpu", f"one gpu_index={arguments.gpu_index} named {TARGET_GPU_NAME}", baseline_devices, "fix --gpu-index, nvidia-smi visibility, or the target machine")
        selected_gpu_uuid = str(baseline_devices[0]["uuid"])
        try:
            process = subprocess.Popen(command)
        except OSError as error:
            fail("command.launch", "an executable product benchmark", str(error), "fix the binary path or dynamic-library environment")
        sampler = NvidiaSmiSampler(
            arguments.nvidia_smi,
            process.pid,
            arguments.gpu_index,
            arguments.sample_interval_ms,
            selected_gpu_uuid,
        )
        sampler.start()
        try:
            return_code = process.wait()
        finally:
            sampler.stop()
        if return_code != 0:
            fail("command.exit_code", "0", return_code, "inspect the product CLI output and TensorRT diagnostics")
        source_after = file_fingerprint(benchmark_json)
        if source_after is None or source_after == source_before:
            fail("benchmark_json.freshness", "a newly created or modified source file", {"before": source_before, "after": source_after}, "ensure --benchmark-json points at this run and pass product --overwrite when needed")
        source_document = load_json_object(benchmark_json)
        metrics = extract_benchmark_metrics(
            source_document,
            expected_requested_provider=expected_provider_values["requested"],
            expected_actual_provider=expected_provider_values["actual"],
            expected_source_evidence_type=backend_contract["source_evidence_type"],
            expected_provider_evidence=(
                expected_native_provider_evidence()
                if arguments.backend_mode == "native"
                else None
            ),
            precision=backend_contract["precision"],
        )
        cache_after = inventory_cache(cache_dir)
        cache_changes = compare_cache_inventories(cache_before, cache_after)
        final_runtime_config = validate_tensorrt_runtime_config(
            Path(raw_runtime_config).resolve(),
            cache_directory=cache_dir,
            expected_requested_provider=expected_provider_values["requested"],
            expected_device_id=arguments.gpu_index,
            expected_engine_sha256=(
                NATIVE_ENGINE_SHA256 if arguments.backend_mode == "native" else None
            ),
        )
        runtime_identity_stable = final_runtime_config == runtime_config
        gpu_memory = select_gpu_memory_evidence(
            pid=process.pid,
            process_samples=sampler.process_samples,
            device_samples=sampler.device_samples,
            baseline_devices=baseline_devices,
            errors=baseline_errors + sampler.errors,
            interval_ms=arguments.sample_interval_ms,
        )
        gpu_memory_gate = gpu_memory_gate_passed(gpu_memory)
        cache_gate = engine_cache_gate_passed(
            cache_after,
            expected_relative_path=runtime_config.get("tensorrt_engine_relative_path"),
            expected_sha256=runtime_config.get("declared_and_actual_engine_sha256"),
        )
        native_cache_immutable = (
            arguments.backend_mode != "native"
            or not cache_changes["created"]
            and not cache_changes["modified"]
            and not cache_changes["removed"]
        )
        passed = (
            gpu_memory_gate
            and cache_gate
            and runtime_identity_stable
            and native_cache_immutable
        )
        timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        document = {
            "schema_version": SCHEMA_VERSION,
            "evidence_type": backend_contract["wrapper_evidence_type"],
            "backend_mode": arguments.backend_mode,
            "timestamp_utc": timestamp,
            "passed": passed,
            "command": command,
            "command_exit_code": return_code,
            "source_benchmark": {
                "path": str(benchmark_json),
                "document": source_document,
            },
            "runtime_config": runtime_config,
            "artifact": artifact,
            "benchmark_sample": {
                "path": str(input_image_path),
                "filename": FROZEN_IMAGE_NAME,
                "declared_and_actual_sha256": FROZEN_IMAGE_SHA256,
            },
            "metrics": metrics,
            "gpu_memory": gpu_memory,
            "engine_cache": {
                "directory": str(cache_dir),
                "before": {"file_count": len(cache_before), "files": cache_before},
                "after": {"file_count": len(cache_after), "files": cache_after},
                "changes": cache_changes,
                "non_empty_after_run": bool(cache_after),
                "valid_non_empty_engine_artifact_after_run": cache_gate,
                "frozen_native_engine_unchanged_during_run": native_cache_immutable,
            },
            "gates": {
                "product_benchmark_succeeded": True,
                "expected_provider_verified": True,
                "profiling_disabled": True,
                "linux_host_peak_rss_recorded": True,
                "gpu_peak_memory_recorded": gpu_memory_gate,
                "engine_cache_non_empty": cache_gate,
                "runtime_and_engine_identity_stable": runtime_identity_stable,
                "native_engine_cache_immutable": native_cache_immutable,
            },
            "limitations": (
                [
                    "session_run is the native boundary covering H2D, TensorRT enqueueV3, D2H, and stream synchronization; preprocess and postprocess remain separately measured.",
                    "The load-only native backend has no fallback provider; the exact provider_evidence and frozen engine SHA are checked before and after this run.",
                    "environment.onnxruntime_version is linked-SDK inventory for the shared product binary; native TensorRT runtime/provider_evidence is the execution authority for this mode.",
                    "Correctness publication is a separate frozen-holdout gate and is not implied by this benchmark wrapper's passed field.",
                    "Device-wide nvidia-smi fallback includes unrelated GPU consumers; its baseline and scope are reported explicitly.",
                    "This is WSL2/Linux x86_64 local GPU / edge-node evidence, not Jetson, native Linux, or native ARM64 GPU evidence.",
                ]
                if arguments.backend_mode == "native"
                else [
                    "session_run is the existing ORT Session::Run boundary; it can include TensorRT, registered fallback providers, and transfer/synchronization overhead.",
                    "Real TensorRT node execution must additionally be proved by summarize_s2_04_ort_profile.py; benchmark provider metadata alone is insufficient.",
                    "Device-wide nvidia-smi fallback includes unrelated GPU consumers; its baseline and scope are reported explicitly.",
                    "This is WSL2/Linux x86_64 local GPU / edge-node evidence, not Jetson, native Linux, or native ARM64 GPU evidence.",
                ]
            ),
        }
        write_json(arguments.output, document, arguments.overwrite)
    except GpuBenchmarkError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        "S2-04 GPU benchmark: "
        f"passed={document['passed']}, initialization_ms={metrics['session_initialization_ms']:.6f}, "
        f"session_run_p50_ms={metrics['latency_ms']['session_run']['p50_ms']:.6f}, "
        f"session_run_p95_ms={metrics['latency_ms']['session_run']['p95_ms']:.6f}, "
        f"gpu_peak_mib={gpu_memory['peak_memory_used_mib']}, output={arguments.output.resolve()}"
    )
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

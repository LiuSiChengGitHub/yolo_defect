"""Compare S2-03 workers=1 and workers=4 BatchSummary evidence.

The comparison first validates both source summaries, then requires the same
ordered tasks, environment, model, config and CPU execution policy. Every
successful per-image detection JSON must be both byte-identical and
semantically identical. Throughput and peak-memory deltas are descriptive;
speedup is deliberately not a pass condition.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, NoReturn, Optional, Sequence, Tuple

import validate_batch_summary as validator


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "s2_03_bounded_concurrency_comparison"


class BatchComparisonError(AssertionError):
    """Raised when two batch runs are not comparable or detections differ."""


def fail(message: str) -> NoReturn:
    raise BatchComparisonError(message)


def normalized_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return os.path.normcase(str(path.resolve(strict=False)))


def require_equal(object_name: str, left: Any, right: Any) -> None:
    if left != right:
        fail(f"{object_name}: expected equal values, workers=1 {left!r}, workers=4 {right!r}")


def read_bytes(path: Path, object_name: str) -> bytes:
    try:
        if not path.is_file():
            fail(f"{object_name}: expected existing regular file, actual {path}")
        return path.read_bytes()
    except OSError as error:
        fail(f"{object_name}: expected readable file, actual {error}")


def describe_byte_difference(left: bytes, right: bytes) -> str:
    shared_length = min(len(left), len(right))
    first_difference = next(
        (index for index in range(shared_length) if left[index] != right[index]),
        shared_length,
    )
    return (
        f"workers=1 bytes={len(left)}, workers=4 bytes={len(right)}, "
        f"first_difference_offset={first_difference}"
    )


def resolve_item_json(item: Mapping[str, Any], summary_path: Path, object_name: str) -> Path:
    value = item.get("json_output_path")
    if not isinstance(value, str) or not value:
        fail(f"{object_name}.json_output_path: expected successful item JSON path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = summary_path.resolve(strict=False).parent / path
    return path.resolve(strict=False)


def metric_comparison(workers_1: float, workers_4: float, *, higher_is_better: bool) -> Dict[str, Any]:
    if not math.isfinite(workers_1) or not math.isfinite(workers_4) or workers_1 <= 0.0 or workers_4 <= 0.0:
        fail(f"metric comparison: expected positive finite values, actual {workers_1}, {workers_4}")
    ratio = workers_4 / workers_1
    delta = workers_4 - workers_1
    tolerance = max(abs(workers_1), abs(workers_4), 1.0) * 1.0e-12
    if abs(delta) <= tolerance:
        direction = "equal"
    elif (delta > 0.0) == higher_is_better:
        direction = "workers_4_better"
    else:
        direction = "workers_4_worse"
    return {
        "workers_1": workers_1,
        "workers_4": workers_4,
        "workers_4_div_workers_1": ratio,
        "workers_4_minus_workers_1": delta,
        "delta_fraction_of_workers_1": delta / workers_1,
        "direction": direction,
    }


def comparable_runtime(runtime: Mapping[str, Any], base: Path) -> Dict[str, Any]:
    ignored = {"requested_workers", "effective_workers", "session_count", "session_initialization_ms"}
    comparable = {key: value for key, value in runtime.items() if key not in ignored}
    comparable["config_path"] = normalized_path(str(runtime["config_path"]), base)
    return comparable


def comparable_model(model: Mapping[str, Any], base: Path) -> Dict[str, Any]:
    comparable = dict(model)
    comparable["model_path"] = normalized_path(str(model["model_path"]), base)
    comparable["declared_sha256"] = str(model["declared_sha256"]).upper()
    return comparable


def comparable_input(input_value: Mapping[str, Any], base: Path) -> Dict[str, Any]:
    comparable = dict(input_value)
    comparable["source_path"] = normalized_path(str(input_value["source_path"]), base)
    return comparable


def compare_documents(
    workers_1: Mapping[str, Any],
    workers_4: Mapping[str, Any],
    *,
    workers_1_summary_path: Path,
    workers_4_summary_path: Path,
) -> Dict[str, Any]:
    if workers_1["status"] != "succeeded" or workers_4["status"] != "succeeded":
        fail("status: formal throughput comparison requires two fully succeeded runs")
    for label, document, requested_workers in (
        ("workers_1", workers_1, 1),
        ("workers_4", workers_4, 4),
    ):
        runtime = document["runtime"]
        if runtime["requested_workers"] != requested_workers or runtime["effective_workers"] != requested_workers:
            fail(
                f"{label}.runtime: expected requested/effective workers={requested_workers}, "
                f"actual {runtime['requested_workers']}/{runtime['effective_workers']}"
            )
        counts = document["counts"]
        if counts["succeeded"] != counts["discovered"] or counts["failed"] or counts["cancelled"]:
            fail(f"{label}.counts: expected every discovered task to succeed")
        if document["memory"]["publishable"] is not True:
            fail(f"{label}.memory.publishable: expected true for formal x86 performance comparison")
        if document["queue"]["capacity"] != 8:
            fail(
                f"{label}.queue.capacity: expected frozen formal protocol value 8, "
                f"actual {document['queue']['capacity']}"
            )

    base_1 = workers_1_summary_path.resolve(strict=False).parent
    base_4 = workers_4_summary_path.resolve(strict=False).parent
    require_equal("environment", workers_1["environment"], workers_4["environment"])
    require_equal("runtime/config", comparable_runtime(workers_1["runtime"], base_1), comparable_runtime(workers_4["runtime"], base_4))
    require_equal("model", comparable_model(workers_1["model"], base_1), comparable_model(workers_4["model"], base_4))
    require_equal("input", comparable_input(workers_1["input"], base_1), comparable_input(workers_4["input"], base_4))
    require_equal("counts.discovered", workers_1["counts"]["discovered"], workers_4["counts"]["discovered"])
    require_equal("output.image_outputs", workers_1["output"]["image_outputs"], workers_4["output"]["image_outputs"])
    if workers_1["output"]["image_outputs"]:
        fail("output.image_outputs: formal throughput comparison requires JSON-only runs")
    require_equal("memory.metric", workers_1["memory"]["metric"], workers_4["memory"]["metric"])
    if not workers_1["memory"]["supported"] or not workers_4["memory"]["supported"]:
        fail("memory.supported: formal comparison requires supported process peak-memory metrics")

    items_1 = workers_1["items"]
    items_4 = workers_4["items"]
    require_equal("items.length", len(items_1), len(items_4))
    compared_items: List[Dict[str, Any]] = []
    for index, (item_1, item_4) in enumerate(zip(items_1, items_4)):
        if item_1["sequence_index"] != index or item_4["sequence_index"] != index:
            fail(f"items[{index}].sequence_index: expected deterministic contiguous indices")
        if item_1["status"] != "succeeded" or item_4["status"] != "succeeded":
            fail(f"items[{index}].status: formal comparison requires matching successes")
        source_1 = normalized_path(str(item_1["source_path"]), base_1)
        source_4 = normalized_path(str(item_4["source_path"]), base_4)
        require_equal(f"items[{index}].source_path", source_1, source_4)
        json_path_1 = resolve_item_json(item_1, workers_1_summary_path, f"workers_1.items[{index}]")
        json_path_4 = resolve_item_json(item_4, workers_4_summary_path, f"workers_4.items[{index}]")
        bytes_1 = read_bytes(json_path_1, f"workers_1.items[{index}].json")
        bytes_4 = read_bytes(json_path_4, f"workers_4.items[{index}].json")
        if bytes_1 != bytes_4:
            fail(
                f"items[{index}].detection_json: expected byte-identical outputs, "
                f"{describe_byte_difference(bytes_1, bytes_4)}"
            )
        document_1 = validator.load_json(json_path_1)
        document_4 = validator.load_json(json_path_4)
        if document_1 != document_4:
            fail(f"items[{index}].detection_json: expected semantic equality")
        compared_items.append(
            {
                "sequence_index": index,
                "source_path": str(item_1["source_path"]),
                "workers_1_json_path": str(json_path_1),
                "workers_4_json_path": str(json_path_4),
                "byte_count": len(bytes_1),
                "byte_equal": True,
                "semantic_equal": True,
            }
        )

    throughput = metric_comparison(
        float(workers_1["throughput_images_per_second"]),
        float(workers_4["throughput_images_per_second"]),
        higher_is_better=True,
    )
    wall = metric_comparison(
        float(workers_1["timing"]["processing_wall_ms"]),
        float(workers_4["timing"]["processing_wall_ms"]),
        higher_is_better=False,
    )
    memory = metric_comparison(
        float(workers_1["memory"]["bytes"]),
        float(workers_4["memory"]["bytes"]),
        higher_is_better=False,
    )
    memory["metric"] = workers_1["memory"]["metric"]
    memory["scope"] = workers_1["memory"]["scope"]
    memory["publishable"] = True

    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "timestamp_utc": timestamp,
        "passed": True,
        "sources": {
            "workers_1_summary": str(workers_1_summary_path.resolve(strict=False)),
            "workers_4_summary": str(workers_4_summary_path.resolve(strict=False)),
        },
        "comparability": {
            "same_environment": True,
            "same_release_build": True,
            "same_cpu_provider_and_ort_thread_policy": True,
            "same_model": True,
            "same_runtime_config": True,
            "same_ordered_tasks": True,
            "same_frozen_queue_capacity": True,
            "queue_capacity": 8,
            "json_only_output_policy": True,
            "detection_json_byte_equal": True,
            "detection_json_semantic_equal": True,
            "compared_item_count": len(compared_items),
        },
        "processing_wall_ms": wall,
        "throughput_images_per_second": throughput,
        "peak_process_memory_bytes": memory,
        "items": compared_items,
        "interpretation": {
            "speedup_is_not_a_pass_condition": True,
            "throughput_outcome": throughput["direction"],
            "memory_outcome": memory["direction"],
        },
        "limitations": [
            "Throughput and peak process memory are comparable only within this recorded machine and platform.",
            "Peak process memory is a process-lifetime high-water mark, not incremental queue or model memory.",
            "A slower workers=4 result remains valid evidence and does not fail this comparison.",
            "This comparison proves deterministic concurrent batch=1 semantics, not true tensor batching.",
        ],
    }


def write_json(path: Path, document: Mapping[str, Any], overwrite: bool) -> None:
    destination = path.resolve(strict=False)
    if destination.exists() and not overwrite:
        fail(f"output: {destination} already exists; choose a new path or pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-1-summary", required=True, type=Path)
    parser.add_argument("--workers-4-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        workers_1 = validator.validate_document(
            validator.load_json(arguments.workers_1_summary),
            summary_path=arguments.workers_1_summary,
            expected_status="succeeded",
            expected_requested_workers=1,
            expected_effective_workers=1,
            expected_memory_publishable=True,
            check_referenced_files=True,
        )
        workers_4 = validator.validate_document(
            validator.load_json(arguments.workers_4_summary),
            summary_path=arguments.workers_4_summary,
            expected_status="succeeded",
            expected_requested_workers=4,
            expected_effective_workers=4,
            expected_memory_publishable=True,
            check_referenced_files=True,
        )
        comparison = compare_documents(
            workers_1,
            workers_4,
            workers_1_summary_path=arguments.workers_1_summary,
            workers_4_summary_path=arguments.workers_4_summary,
        )
        write_json(arguments.output, comparison, arguments.overwrite)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    throughput = comparison["throughput_images_per_second"]
    memory = comparison["peak_process_memory_bytes"]
    print(
        "S2-03 batch comparison: passed=True, "
        f"throughput_ratio={throughput['workers_4_div_workers_1']:.6f}, "
        f"memory_delta_bytes={memory['workers_4_minus_workers_1']:.0f}, "
        f"items={comparison['comparability']['compared_item_count']}, "
        f"output={arguments.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

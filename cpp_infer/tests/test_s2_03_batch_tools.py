"""Pure standard-library tests for the S2-03 batch evidence tools."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIRECTORY = CPP_INFER_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIRECTORY))
import compare_batch_runs as comparison  # noqa: E402
import validate_batch_summary as validator  # noqa: E402


MODEL_SHA256 = "A" * 64


@contextlib.contextmanager
def workspace_directory():
    path = CPP_INFER_ROOT / "build" / f"s2_03_python_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def detection_document(source_path: Path) -> dict:
    return {
        "schema_version": 1,
        "model": {"model_id": "model", "declared_sha256": MODEL_SHA256},
        "image": {
            "path": str(source_path),
            "original_size": {"width": 200, "height": 200, "channels": 3},
            "input_size": {"width": 800, "height": 800},
        },
        "runtime": {
            "actual_provider": "CPUExecutionProvider",
            "provider_evidence": "explicit_cpu_ep_registration_and_session_creation",
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
            "nms_mode": "class_agnostic",
        },
        "detections": [],
    }


def make_summary(base: Path, *, workers: int, wall_ms: float, memory_bytes: int) -> dict:
    run_directory = base / f"workers_{workers}"
    item_directory = run_directory / "items"
    source_paths = [base / "inputs" / f"image_{index}.jpg" for index in range(4)]
    items = []
    for index, source_path in enumerate(source_paths):
        items.append(
            {
                "sequence_index": index,
                "status": "succeeded",
                "source_path": str(source_path),
                "json_output_path": str(item_directory / f"{index:06d}.detections.json"),
                "image_output_path": None,
                "detection_count": 0,
                "latency_ms": 100.0,
                "error": None,
            }
        )
    return {
        "schema_version": 1,
        "evidence_type": "cpp_ort_multi_image_batch_summary",
        "timestamp_utc": "2026-08-30T00:00:00Z",
        "status": "succeeded",
        "cooperative_stop_requested": False,
        "command_arguments": ["yolo_defect_cpp", "--batch"],
        "environment": {
            "hostname": "host",
            "processor": "cpu",
            "logical_cpu_count": 8,
            "os_name": "Linux",
            "os_version": "test",
            "target_architecture": "x86_64",
            "runtime_kernel_architecture": "x86_64",
            "execution_context": "native_or_unknown",
            "compiler_id": "GNU",
            "compiler_version": "test",
            "build_type": "Release",
            "cxx_standard": 17,
            "opencv_version": "4.8.0",
            "onnxruntime_version": "1.19.2",
        },
        "runtime": {
            "config_path": str(base / "default_config.txt"),
            "requested_provider": "cpu",
            "actual_provider": "CPUExecutionProvider",
            "provider_evidence": "explicit_cpu_ep_registration_and_session_creation",
            "execution_mode": "sequential",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "graph_optimization_level": "all",
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
            "nms_mode": "class_agnostic",
            "requested_workers": workers,
            "effective_workers": workers,
            "session_count": workers,
            "session_initialization_ms": [10.0] * workers,
        },
        "model": {
            "model_id": "model",
            "model_family": "yolov8",
            "model_path": str(base / "model.onnx"),
            "declared_sha256": MODEL_SHA256,
            "opset": 17,
            "input_name": "images",
            "input_shape": [1, 3, 800, 800],
            "input_dtype": "float32",
            "input_layout": "nchw",
        },
        "input": {
            "kind": "manifest",
            "source_path": str(base / "manifest.txt"),
            "ordering": "UTF-8 path-list declaration order",
        },
        "output": {
            "directory": str(run_directory),
            "batch_summary_path": str(run_directory / "summary.json"),
            "item_directory": str(item_directory),
            "json_outputs": True,
            "image_outputs": False,
            "overwrite_existing": False,
        },
        "counts": {
            "discovered": 4,
            "enqueued": 4,
            "started": 4,
            "succeeded": 4,
            "failed": 0,
            "cancelled": 0,
        },
        "queue": {
            "capacity": 8,
            "peak_depth": 4,
            "producer_wait_count": 0,
            "producer_wait_ms": 0.0,
        },
        "timing": {
            "processing_wall_ms": wall_ms,
            "includes": ["queue wait and per-item inference/output"],
            "excludes": ["task discovery and session construction"],
        },
        "latency_ms": {
            "sample_count": 4,
            "mean_ms": 100.0,
            "p50_ms": 100.0,
            "p95_ms": 100.0,
        },
        "throughput_images_per_second": 4000.0 / wall_ms,
        "memory": {
            "supported": True,
            "status": "supported",
            "metric": "peak_rss",
            "bytes": memory_bytes,
            "mebibytes": memory_bytes / (1024.0 * 1024.0),
            "scope": "process lifetime",
            "reason": None,
            "publishable": True,
        },
        "items": items,
        "limitations": ["synthetic pure-Python fixture"],
        "fatal_error": None,
    }


class BatchSummaryValidatorTest(unittest.TestCase):
    def test_accepts_valid_contract_and_expected_flags(self):
        with workspace_directory() as base:
            document = make_summary(base, workers=4, wall_ms=200.0, memory_bytes=200_000_000)
            validated = validator.validate_document(
                document,
                summary_path=base / "workers_4" / "summary.json",
                expected_status="succeeded",
                expected_counts={"discovered": 4, "succeeded": 4},
                expected_target_architecture="amd64",
                expected_runtime_kernel_architecture="amd64",
                expected_requested_workers=4,
                expected_effective_workers=4,
                expected_input_kind="manifest",
                expected_memory_publishable=True,
                check_referenced_files=False,
            )
            self.assertEqual(validated["counts"]["succeeded"], 4)

    def test_fatal_before_processing_does_not_require_item_output_directories(self):
        with workspace_directory() as base:
            document = make_summary(
                base, workers=4, wall_ms=200.0, memory_bytes=200_000_000
            )
            inputs = base / "inputs"
            inputs.mkdir(parents=True)
            source_paths = [inputs / f"image_{index}.jpg" for index in range(4)]
            for source_path in source_paths:
                source_path.write_bytes(b"synthetic fixture")
            with (base / "manifest.txt").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write("\n".join(f"inputs/{path.name}" for path in source_paths) + "\n")
            (base / "default_config.txt").write_text(
                "synthetic fixture\n", encoding="utf-8"
            )
            model_bytes = b"synthetic model fixture"
            (base / "model.onnx").write_bytes(model_bytes)
            document["model"]["declared_sha256"] = hashlib.sha256(
                model_bytes
            ).hexdigest().upper()
            document["status"] = "fatal"
            document["fatal_error"] = "worker session initialization failed"
            document["runtime"]["session_count"] = 0
            document["runtime"]["session_initialization_ms"] = []
            document["counts"].update(
                {
                    "enqueued": 0,
                    "started": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "cancelled": 4,
                }
            )
            document["queue"].update(
                {"peak_depth": 0, "producer_wait_count": 0, "producer_wait_ms": 0.0}
            )
            document["timing"]["processing_wall_ms"] = 0.0
            document["latency_ms"].update(
                {"sample_count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
            )
            document["throughput_images_per_second"] = 0.0
            for item in document["items"]:
                item.update(
                    {
                        "status": "cancelled",
                        "json_output_path": None,
                        "image_output_path": None,
                        "detection_count": 0,
                        "latency_ms": 0.0,
                        "error": "cancelled before processing",
                    }
                )
            summary_path = base / "evidence" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            document["output"]["batch_summary_path"] = str(summary_path)
            with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(document, stream)
                stream.write("\n")

            validated = validator.validate_document(
                document,
                summary_path=summary_path,
                expected_status="fatal",
                expected_runtime_kernel_architecture="x86_64",
                check_referenced_files=True,
            )
            self.assertEqual(validated["counts"]["cancelled"], 4)
            self.assertFalse((base / "workers_4").exists())

    def test_accepts_stop_after_every_item_started(self):
        with workspace_directory() as base:
            document = make_summary(
                base, workers=4, wall_ms=200.0, memory_bytes=200_000_000
            )
            document["status"] = "cancelled"
            document["cooperative_stop_requested"] = True

            validated = validator.validate_document(
                document,
                expected_status="cancelled",
                check_referenced_files=False,
            )

            self.assertEqual(validated["counts"]["cancelled"], 0)
            self.assertTrue(validated["cooperative_stop_requested"])

    def test_rejects_count_and_item_invariant_break(self):
        with workspace_directory() as base:
            document = make_summary(base, workers=1, wall_ms=400.0, memory_bytes=100_000_000)
            document["counts"]["failed"] = 1
            with self.assertRaises(validator.BatchSummaryValidationError) as context:
                validator.validate_document(document, check_referenced_files=False)
            self.assertIn("started == succeeded + failed", str(context.exception))

    def test_accepts_zero_duration_failed_item_like_cpp_schema(self):
        with workspace_directory() as base:
            document = make_summary(
                base, workers=1, wall_ms=400.0, memory_bytes=100_000_000
            )
            document["status"] = "partial_failure"
            document["counts"].update({"succeeded": 3, "failed": 1})
            document["latency_ms"]["sample_count"] = 3
            document["throughput_images_per_second"] = 7.5
            document["items"][3].update(
                {
                    "status": "failed",
                    "json_output_path": None,
                    "detection_count": 0,
                    "latency_ms": 0.0,
                    "error": "decode failed before measurable work elapsed",
                }
            )

            validated = validator.validate_document(
                document,
                expected_status="partial_failure",
                check_referenced_files=False,
            )
            self.assertEqual(validated["counts"]["failed"], 1)

    def test_frozen_path_manifest_matches_consistency_manifest(self):
        fixture_directory = CPP_INFER_ROOT / "tests" / "fixtures"
        consistency = json.loads((fixture_directory / "consistency_manifest.json").read_text(encoding="utf-8"))
        expected = [sample["image_path"] for sample in consistency["samples"]]
        actual = [
            line.strip()
            for line in (fixture_directory / "s2_03_consistency_manifest.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(actual, expected)

    def test_manifest_rediscovery_preserves_unicode_filename_whitespace(self):
        with workspace_directory() as base:
            filename = "\u00a0surface.jpg"
            source = base / filename
            source.write_bytes(b"synthetic fixture")
            manifest = base / "unicode_whitespace.txt"
            with manifest.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(filename + "\n")

            self.assertEqual(validator.discover_manifest(manifest), [source.resolve()])


class BatchComparisonTest(unittest.TestCase):
    def setUp(self):
        self.context = workspace_directory()
        self.base = self.context.__enter__()
        self.workers_1 = make_summary(self.base, workers=1, wall_ms=400.0, memory_bytes=100_000_000)
        self.workers_4 = make_summary(self.base, workers=4, wall_ms=500.0, memory_bytes=180_000_000)
        for document in (self.workers_1, self.workers_4):
            for item in document["items"]:
                destination = Path(item["json_output_path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        json.dumps(
                            detection_document(Path(item["source_path"])),
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    )

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def compare(self) -> dict:
        return comparison.compare_documents(
            self.workers_1,
            self.workers_4,
            workers_1_summary_path=self.base / "workers_1" / "summary.json",
            workers_4_summary_path=self.base / "workers_4" / "summary.json",
        )

    def test_slower_concurrency_is_valid_and_deltas_are_machine_readable(self):
        result = self.compare()
        self.assertTrue(result["passed"])
        self.assertEqual(result["throughput_images_per_second"]["workers_4_div_workers_1"], 0.8)
        self.assertEqual(result["throughput_images_per_second"]["direction"], "workers_4_worse")
        self.assertEqual(result["peak_process_memory_bytes"]["workers_4_minus_workers_1"], 80_000_000.0)
        self.assertTrue(result["interpretation"]["speedup_is_not_a_pass_condition"])

    def test_rejects_semantically_equal_but_byte_different_detection_json(self):
        path = Path(self.workers_4["items"][0]["json_output_path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
        with self.assertRaises(comparison.BatchComparisonError) as context:
            self.compare()
        self.assertIn("byte-identical", str(context.exception))

    def test_rejects_config_mismatch(self):
        self.workers_4["runtime"]["config_path"] = str(self.base / "other_config.txt")
        with self.assertRaises(comparison.BatchComparisonError) as context:
            self.compare()
        self.assertIn("runtime/config", str(context.exception))

    def test_rejects_non_frozen_queue_capacity(self):
        self.workers_4["queue"]["capacity"] = 4
        with self.assertRaises(comparison.BatchComparisonError) as context:
            self.compare()
        self.assertIn("queue.capacity", str(context.exception))


if __name__ == "__main__":
    unittest.main()

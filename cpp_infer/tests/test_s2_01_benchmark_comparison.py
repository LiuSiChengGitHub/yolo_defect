"""Pure tests for S2-01 FP32/INT8 benchmark comparability and deltas."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = CPP_INFER_ROOT / "tools" / "compare_s2_01_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("compare_s2_01_benchmarks", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import benchmark tool from {TOOL_PATH}")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def statistics(mean: float) -> dict:
    return {"sample_count": 100, "mean": mean, "p50": mean * 0.9, "p95": mean * 1.2}


def benchmark_document(*, precision: str, latency_scale: float, size: int, sha: str) -> dict:
    latency = {
        "image_decode": statistics(1.0 * latency_scale),
        "preprocess": statistics(8.0 * latency_scale),
        "session_run": statistics(100.0 * latency_scale),
        "postprocess": statistics(1.0 * latency_scale),
        "pipeline": statistics(110.0 * latency_scale),
        "end_to_end": statistics(111.0 * latency_scale),
    }
    return {
        "schema_version": 1,
        "evidence_type": "cpp_ort_single_image_release_benchmark",
        "timestamp_utc": "2026-08-25T00:00:00Z",
        "command": ["yolo_defect_cpp", "--benchmark"],
        "protocol": {
            "batch_size": 1,
            "sample_count": 1,
            "warmup": 10,
            "repeat": 100,
            "clock": "std::chrono::steady_clock",
            "timing_unit": "milliseconds",
            "percentile_method": "empirical_nearest_rank_ceiling",
        },
        "environment": {
            "machine": {
                "hostname": "host",
                "processor": "cpu",
                "architecture": "x86_64",
                "logical_cpu_count": 16,
            },
            "os": {"name": "Windows", "version": "test"},
            "compiler": {"id": "MSVC", "version": "test"},
            "build": {"type": "Release", "cxx_standard": 17},
            "opencv_version": "4.8.0",
            "onnxruntime_version": "1.19.2",
        },
        "runtime": {
            "requested_provider": "cpu",
            "actual_provider": "CPUExecutionProvider",
            "provider_evidence": "explicit_cpu_ep_registration_and_session_creation",
            "session": {
                "execution_mode": "sequential",
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
                "graph_optimization_level": "all",
                "initialization_ms": 50.0 * latency_scale,
                "profiling_enabled": False,
            },
        },
        "model": {
            "model_id": f"model_{precision}",
            "model_family": "yolov8",
            "path": f"models/{precision}.onnx",
            "declared_sha256": sha,
            "file_size_bytes": size,
            "opset": 17,
            "input": {
                "name": "images",
                "shape": [1, 3, 800, 800],
                "dtype": "float32",
                "layout": "nchw",
            },
        },
        "sample": {
            "image_path": "data/images/val/crazing_241.jpg",
            "file_size_bytes": 23845,
            "original_shape": [200, 200, 3],
            "sample_count": 1,
        },
        "postprocess": {
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
            "nms_mode": "class_agnostic",
            "detection_count": 3,
        },
        "latency_ms": latency,
        "throughput_images_per_second": {
            "pipeline": 1000.0 / latency["pipeline"]["mean"],
            "end_to_end": 1000.0 / latency["end_to_end"]["mean"],
        },
        "memory": {
            "status": "supported",
            "metric": "peak_working_set",
            "bytes": int(100_000_000 * latency_scale),
            "mebibytes": 100.0,
            "scope": "process lifetime",
            "reason": None,
        },
        "timing_exclusions": ["json write"],
        "limitations": ["synthetic"],
    }


class BenchmarkComparisonTest(unittest.TestCase):
    def setUp(self):
        self.fp32 = benchmark_document(
            precision="fp32", latency_scale=1.0, size=12_000_000, sha="A" * 64
        )
        self.int8 = benchmark_document(
            precision="int8", latency_scale=0.5, size=3_000_000, sha="B" * 64
        )
        self.protocol_binding = {
            "protocol_id": "s2_01_test_v1",
            "canonical_lf_sha256": "C" * 64,
            "source_model_sha256": "A" * 64,
            "derived_model_sha256": "B" * 64,
            "warmup": 10,
            "repeat": 100,
        }
        self.correctness = {
            "passed": True,
            "evidence_type": "s2_01_fp32_int8_correctness_and_quality",
            "protocol": {
                "protocol_id": "s2_01_test_v1",
                "canonical_lf_sha256": "C" * 64,
            },
            "runtime_legality": {
                "python_fp32_session_and_finite_outputs": True,
                "python_int8_session_and_finite_outputs": True,
                "cpp": {"requested": True, "passed": True},
            },
            "artifacts": {
                "fp32": {"model_sha256": "A" * 64},
                "int8": {"model_sha256": "B" * 64},
            },
        }

    def compare(self):
        return comparison.compare_documents(
            self.fp32,
            self.int8,
            self.correctness,
            self.protocol_binding,
            validate_referenced_models=False,
        )

    def test_computes_latency_throughput_memory_and_size_deltas(self):
        result = self.compare()
        self.assertTrue(result["passed"])
        pipeline = result["latency_ms"]["pipeline"]["mean"]
        self.assertEqual(pipeline["fp32"], 110.0)
        self.assertEqual(pipeline["int8"], 55.0)
        self.assertEqual(pipeline["fp32_div_int8"], 2.0)
        self.assertEqual(pipeline["direction"], "int8_better")
        self.assertEqual(result["models"]["size"]["reduction_fraction"], 0.75)
        self.assertEqual(
            result["throughput_images_per_second"]["pipeline"]["direction"],
            "int8_better",
        )

    def test_rejects_profiled_benchmark(self):
        self.int8["runtime"]["session"]["profiling_enabled"] = True
        with self.assertRaises(comparison.BenchmarkComparisonError) as context:
            self.compare()
        self.assertIn("profiling_enabled", str(context.exception))

    def test_rejects_protocol_mismatch(self):
        self.int8["protocol"]["repeat"] = 99
        for value in self.int8["latency_ms"].values():
            value["sample_count"] = 99
        with self.assertRaises(comparison.BenchmarkComparisonError) as context:
            self.compare()
        self.assertIn("protocol", str(context.exception))

    def test_rejects_failed_correctness_prerequisite(self):
        self.correctness["passed"] = False
        with self.assertRaises(comparison.BenchmarkComparisonError) as context:
            self.compare()
        self.assertIn("correctness.passed", str(context.exception))

    def test_advisory_policy_preserves_failed_correctness_without_blocking_comparison(self):
        self.correctness["passed"] = False
        result = comparison.compare_documents(
            self.fp32,
            self.int8,
            self.correctness,
            self.protocol_binding,
            validate_referenced_models=False,
            correctness_policy="advisory",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["correctness_prerequisite"]["policy"], "advisory")
        self.assertFalse(result["correctness_prerequisite"]["passed"])
        self.assertFalse(result["correctness_prerequisite"]["blocking"])
        self.assertTrue(result["correctness_prerequisite"]["accepted_for_comparison"])

    def test_rejects_unknown_correctness_policy(self):
        with self.assertRaises(comparison.BenchmarkComparisonError) as context:
            comparison.compare_documents(
                self.fp32,
                self.int8,
                self.correctness,
                self.protocol_binding,
                validate_referenced_models=False,
                correctness_policy="ignored",
            )
        self.assertIn("correctness_policy", str(context.exception))

    def test_rejects_correctness_for_another_artifact(self):
        self.correctness["artifacts"]["int8"]["model_sha256"] = "D" * 64
        with self.assertRaises(comparison.BenchmarkComparisonError) as context:
            self.compare()
        self.assertIn("correctness.artifacts.int8", str(context.exception))

    def test_allows_detection_count_difference_but_not_threshold_change(self):
        self.int8["postprocess"]["detection_count"] = 2
        self.assertTrue(self.compare()["passed"])
        self.int8["postprocess"]["score_threshold"] = 0.3
        with self.assertRaises(comparison.BenchmarkComparisonError) as context:
            self.compare()
        self.assertIn("postprocess", str(context.exception))


if __name__ == "__main__":
    unittest.main()

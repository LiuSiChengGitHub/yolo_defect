"""Pure standard-library tests for the S2-04 evidence tools."""

from __future__ import annotations

import sys
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIRECTORY = CPP_INFER_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIRECTORY))

import compare_s2_04_correctness as correctness  # noqa: E402
import run_s2_04_gpu_benchmark as gpu_benchmark  # noqa: E402
import summarize_s2_04_ort_profile as profile  # noqa: E402


MODEL_SHA256 = "7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68"


def detection_run(provider: str, detections: list) -> dict:
    return {
        "model": {
            "model_id": "yolov8n_neu_det_final_train_2",
            "declared_sha256": MODEL_SHA256,
        },
        "image": {
            "path": "/dataset/crazing_241.jpg",
            "original_size": {"width": 200, "height": 200, "channels": 3},
            "input_size": {"width": 800, "height": 800},
        },
        "runtime": {
            "actual_provider": provider,
            "provider_evidence": "synthetic provider evidence",
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
            "nms_mode": "class_agnostic",
        },
        "detections": detections,
    }


def detection(class_id: int, confidence: float, bbox: list) -> dict:
    return {
        "class_id": class_id,
        "class_name": correctness.CLASS_NAMES[class_id],
        "confidence": confidence,
        "bbox_xyxy": bbox,
    }


class CorrectnessLogicTest(unittest.TestCase):
    def test_correctness_outputs_cannot_replace_frozen_inputs_or_run_files(self):
        root = Path(tempfile.gettempdir()) / (
            f"yolo_defect_s2_04_correctness_paths_{uuid.uuid4().hex}"
        )
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        protocol = root / "protocol.json"
        protocol.write_text("{}", encoding="utf-8")
        image = root / "source.jpg"
        image.write_bytes(b"frozen image")
        run_directory = root / "run_a"
        run_directory.mkdir()

        for destination in (protocol, image, run_directory / "result.json"):
            with self.subTest(destination=destination):
                with self.assertRaises(correctness.CorrectnessError) as context:
                    correctness.reject_protected_destination(
                        destination,
                        protected_files=[protocol, image],
                        protected_directories=[run_directory],
                    )
                self.assertIn("output.protected_input", str(context.exception))

        correctness.reject_protected_destination(
            root / "evidence" / "summary.json",
            protected_files=[protocol, image],
            protected_directories=[run_directory],
        )

    def test_v2_protocol_uses_a_disjoint_frozen_holdout(self):
        protocol_path = (
            CPP_INFER_ROOT / "protocols" / "s2_04_tensorrt_fp16_protocol_v2.json"
        )
        protocol = correctness.load_json_object(protocol_path)
        sources, specification = correctness.validate_protocol(
            protocol, protocol_path
        )
        samples = correctness.validate_manifest(
            correctness.load_json_object(sources["consistency_manifest_path"]),
            sources["consistency_manifest_path"],
            specification["sample_contract"],
        )

        self.assertEqual(len(samples), 30)
        self.assertTrue(
            set(correctness.SELECTION_INDICES).isdisjoint(
                correctness.HOLDOUT_SELECTION_INDICES
            )
        )
        self.assertEqual(specification["gate"], correctness.HOLDOUT_GATE)

    def test_v3_protocol_binds_native_engine_and_third_disjoint_holdout(self):
        protocol_path = (
            CPP_INFER_ROOT
            / "protocols"
            / "s2_04_tensorrt_native_fp16_protocol_v3.json"
        )
        protocol = correctness.load_json_object(protocol_path)
        sources, specification = correctness.validate_protocol(
            protocol, protocol_path
        )
        samples = correctness.validate_manifest(
            correctness.load_json_object(sources["consistency_manifest_path"]),
            sources["consistency_manifest_path"],
            specification["sample_contract"],
        )

        self.assertEqual(len(samples), 30)
        selections = [
            set(correctness.SELECTION_INDICES),
            set(correctness.HOLDOUT_SELECTION_INDICES),
            set(correctness.NATIVE_HOLDOUT_SELECTION_INDICES_V3),
        ]
        self.assertTrue(selections[0].isdisjoint(selections[1]))
        self.assertTrue(selections[0].isdisjoint(selections[2]))
        self.assertTrue(selections[1].isdisjoint(selections[2]))
        self.assertEqual(
            specification["engine_contract"]["engine_sha256"],
            correctness.NATIVE_ENGINE_SHA256_V3,
        )
        self.assertEqual(
            specification["engine_contract"]["tensorrt_runtime_version"],
            "10.4.0",
        )

    def test_v4_protocol_binds_new_engine_and_fourth_disjoint_holdout(self):
        protocol_path = (
            CPP_INFER_ROOT
            / "protocols"
            / "s2_04_tensorrt_native_fp16_protocol_v4.json"
        )
        protocol = correctness.load_json_object(protocol_path)
        sources, specification = correctness.validate_protocol(
            protocol, protocol_path
        )
        samples = correctness.validate_manifest(
            correctness.load_json_object(sources["consistency_manifest_path"]),
            sources["consistency_manifest_path"],
            specification["sample_contract"],
        )

        self.assertEqual(len(samples), 30)
        selections = [
            set(correctness.SELECTION_INDICES),
            set(correctness.HOLDOUT_SELECTION_INDICES),
            set(correctness.NATIVE_HOLDOUT_SELECTION_INDICES_V3),
            set(correctness.NATIVE_HOLDOUT_SELECTION_INDICES_V4),
        ]
        for left_index, left in enumerate(selections):
            for right in selections[left_index + 1 :]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(
            specification["engine_contract"]["engine_sha256"],
            correctness.NATIVE_ENGINE_SHA256_V4,
        )
        self.assertEqual(
            specification["engine_contract"]["precision_policy"],
            correctness.NATIVE_PRECISION_POLICY_V4,
        )

    def test_v2_source_pixel_gate_does_not_relabel_v1_as_passing(self):
        reference = detection_run(
            "CPUExecutionProvider",
            [
                detection(
                    1,
                    0.390283734,
                    [133.766891, 136.543976, 146.773392, 165.108795],
                )
            ],
        )
        candidate = detection_run(
            "TensorrtExecutionProvider",
            [
                detection(
                    1,
                    0.390475690,
                    [133.565399, 136.846954, 147.483185, 165.773041],
                )
            ],
        )

        strict_v1 = correctness.compare_image(
            reference, candidate, "strict_v1", correctness.FROZEN_GATE
        )
        holdout_v2 = correctness.compare_image(
            reference, candidate, "holdout_v2", correctness.HOLDOUT_GATE
        )

        self.assertFalse(strict_v1["passed"])
        self.assertTrue(holdout_v2["passed"])

    def test_matching_is_order_independent_and_accepts_predeclared_fp16_error(self):
        reference = detection_run(
            "CPUExecutionProvider",
            [
                detection(0, 0.70, [10.0, 10.0, 110.0, 110.0]),
                detection(0, 0.60, [120.0, 20.0, 180.0, 80.0]),
            ],
        )
        candidate = detection_run(
            "TensorrtExecutionProvider",
            [
                detection(0, 0.597, [120.05, 20.05, 180.05, 80.05]),
                detection(0, 0.704, [10.05, 10.05, 110.05, 110.05]),
            ],
        )

        result = correctness.compare_image(reference, candidate, "cpu_vs_trt")

        self.assertTrue(result["passed"])
        self.assertEqual(
            [(row["reference_index"], row["candidate_index"]) for row in result["matches"]],
            [(0, 1), (1, 0)],
        )
        self.assertLessEqual(
            max(row["confidence_abs_error"] for row in result["matches"]),
            0.005,
        )
        self.assertGreaterEqual(
            min(row["iou"] for row in result["matches"]),
            0.995,
        )

    def test_same_gate_rejects_repeat_confidence_drift(self):
        run_a = detection_run(
            "TensorrtExecutionProvider",
            [detection(0, 0.70, [10.0, 10.0, 110.0, 110.0])],
        )
        run_b = detection_run(
            "TensorrtExecutionProvider",
            [detection(0, 0.706, [10.0, 10.0, 110.0, 110.0])],
        )

        result = correctness.compare_image(
            run_a, run_b, "tensorrt_run_a_vs_tensorrt_run_b"
        )

        self.assertFalse(result["passed"])
        self.assertIn("exceeds gate", result["failures"][0])

    def test_exact_class_histogram_is_a_gate(self):
        reference = detection_run(
            "CPUExecutionProvider",
            [detection(0, 0.70, [10.0, 10.0, 110.0, 110.0])],
        )
        candidate = detection_run(
            "TensorrtExecutionProvider",
            [detection(1, 0.70, [10.0, 10.0, 110.0, 110.0])],
        )

        result = correctness.compare_image(reference, candidate, "cpu_vs_trt")

        self.assertFalse(result["passed"])
        self.assertTrue(any("class histogram" in value for value in result["failures"]))

    def test_provider_expectation_cannot_relabel_cpu_as_tensorrt(self):
        with self.assertRaises(correctness.CorrectnessError) as context:
            correctness.validate_provider_expectations(
                correctness.CPU_PROVIDER,
                correctness.CPU_PROVIDER,
                correctness.TRT_PROVIDER,
            )
        self.assertIn("not user-redefinable", str(context.exception))


class ProfileLogicTest(unittest.TestCase):
    def test_reports_tensorrt_and_both_fallback_providers(self):
        events = [
            {"cat": "Session", "ph": "X", "name": "model_run", "dur": 1500},
            {
                "cat": "Node",
                "ph": "X",
                "name": "trt_subgraph_kernel_time",
                "dur": 1000,
                "args": {"provider": "TensorrtExecutionProvider", "op_name": "TRTKernel"},
            },
            {
                "cat": "Node",
                "ph": "X",
                "name": "cuda_node_kernel_time",
                "dur": 250,
                "args": {"provider": "CUDAExecutionProvider", "op_name": "Resize"},
            },
            {
                "cat": "Node",
                "ph": "X",
                "name": "cpu_node_kernel_time",
                "dur": 100,
                "args": {"provider": "CPUExecutionProvider", "op_name": "Shape"},
            },
            {
                "cat": "Node",
                "ph": "X",
                "name": "trt_subgraph_fence_before",
                "dur": 9999,
                "args": {"provider": "TensorrtExecutionProvider", "op_name": "TRTKernel"},
            },
        ]

        summary = profile.summarize_events(
            events,
            trace_path=Path("synthetic_profile.json"),
            model_id=profile.FROZEN_MODEL_ID,
            model_sha256=profile.FROZEN_MODEL_SHA256,
            expected_profile_runs=1,
        )

        proof = summary["execution_proof"]
        self.assertTrue(proof["real_tensorrt_node_execution_observed"])
        self.assertEqual(proof["tensorrt_node_event_count"], 1)
        self.assertEqual(proof["cuda_fallback_node_event_count"], 1)
        self.assertEqual(proof["cpu_fallback_node_event_count"], 1)
        self.assertEqual(summary["trace"]["ignored_non_kernel_node_event_count"], 1)
        self.assertEqual(summary["trace"]["aggregate_node_duration_ms"], 1.35)

    def test_rejects_profile_without_tensorrt_node(self):
        events = [
            {"cat": "Session", "ph": "X", "name": "model_run", "dur": 10},
            {
                "cat": "Node",
                "ph": "X",
                "name": "cuda_kernel_time",
                "dur": 10,
                "args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"},
            },
        ]
        with self.assertRaises(profile.ProfileSummaryError) as context:
            profile.summarize_events(
                events,
                trace_path=Path("cuda_only.json"),
                model_id=profile.FROZEN_MODEL_ID,
                model_sha256=profile.FROZEN_MODEL_SHA256,
                expected_profile_runs=1,
            )
        self.assertIn("tensorrt_node_events", str(context.exception))

    def test_rejects_zero_duration_tensorrt_marker(self):
        events = [
            {"cat": "Session", "ph": "X", "name": "model_run", "dur": 10},
            {
                "cat": "Node",
                "ph": "X",
                "name": "trt_kernel_time",
                "dur": 0,
                "args": {"provider": "TensorrtExecutionProvider", "op_name": "TRTKernel"},
            },
            {
                "cat": "Node",
                "ph": "X",
                "name": "cpu_kernel_time",
                "dur": 10,
                "args": {"provider": "CPUExecutionProvider", "op_name": "Shape"},
            },
        ]
        with self.assertRaises(profile.ProfileSummaryError) as context:
            profile.summarize_events(
                events,
                trace_path=Path("zero_trt.json"),
                model_id=profile.FROZEN_MODEL_ID,
                model_sha256=profile.FROZEN_MODEL_SHA256,
                expected_profile_runs=1,
            )
        self.assertIn("tensorrt_node_duration", str(context.exception))

    def test_provider_arguments_cannot_relabel_cuda_as_tensorrt(self):
        with self.assertRaises(profile.ProfileSummaryError) as context:
            profile.summarize_events(
                [],
                trace_path=Path("renamed.json"),
                model_id=profile.FROZEN_MODEL_ID,
                model_sha256=profile.FROZEN_MODEL_SHA256,
                expected_profile_runs=1,
                tensorrt_provider="CUDAExecutionProvider",
                cuda_provider="TensorrtExecutionProvider",
            )
        self.assertIn("provider meaning is not user-redefinable", str(context.exception))


def benchmark_document() -> dict:
    latency_row = {"sample_count": 100, "mean": 2.0, "p50": 1.8, "p95": 2.7}
    return {
        "schema_version": 1,
        "evidence_type": "cpp_ort_single_image_release_benchmark",
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
            "machine": {"architecture": "x86_64"},
            "os": {"name": "Linux"},
            "build": {"type": "Release"},
            "onnxruntime_version": "1.20.1",
        },
        "runtime": {
            "requested_provider": "tensorrt",
            "actual_provider": "TensorrtExecutionProvider",
            "provider_evidence": "TensorRT EP registered with CUDA fallback",
            "session": {"initialization_ms": 1234.5, "profiling_enabled": False},
        },
        "model": {
            "model_id": gpu_benchmark.FROZEN_MODEL_ID,
            "declared_sha256": gpu_benchmark.FROZEN_MODEL_SHA256,
        },
        "latency_ms": {
            "image_decode": dict(latency_row),
            "preprocess": dict(latency_row),
            "session_run": dict(latency_row),
            "postprocess": dict(latency_row),
            "pipeline": dict(latency_row),
            "end_to_end": dict(latency_row),
        },
        "throughput_images_per_second": {"pipeline": 500.0, "end_to_end": 490.0},
        "memory": {
            "status": "supported",
            "metric": "peak_rss",
            "bytes": 256 * 1024 * 1024,
            "mebibytes": 256.0,
            "scope": "process lifetime",
        },
    }


class GpuBenchmarkLogicTest(unittest.TestCase):
    def test_output_destination_cannot_alias_frozen_engine_or_cache(self):
        root = Path(tempfile.gettempdir()) / (
            f"yolo_defect_s2_04_destination_{uuid.uuid4().hex}"
        )
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        cache = root / "cache"
        cache.mkdir()
        engine = cache / "frozen.engine"
        engine.write_bytes(b"engine")
        for destination in (engine, cache / "future.json"):
            with self.subTest(destination=destination):
                with self.assertRaises(gpu_benchmark.GpuBenchmarkError):
                    gpu_benchmark.reject_protected_destination(
                        destination,
                        object_name="output.destination",
                        protected_files=[engine],
                        protected_directory=cache,
                    )

    def test_process_csv_aggregates_same_pid_across_rows(self):
        parsed = gpu_benchmark.parse_process_memory_csv(
            "123, 128\n123, 64 MiB\n456, [N/A]\nmalformed\n"
        )
        self.assertEqual(parsed, {123: 192.0})

    def test_process_csv_filters_selected_gpu_uuid(self):
        parsed = gpu_benchmark.parse_process_memory_csv(
            "GPU-selected,123,128\nGPU-other,123,512\n",
            "GPU-selected",
        )
        self.assertEqual(parsed, {123: 128.0})

    def test_device_csv_and_device_wide_fallback_report_baseline_delta(self):
        devices = gpu_benchmark.parse_device_memory_csv(
            '0,"NVIDIA GeForce RTX 4060 Laptop GPU",GPU-123,512\n'
        )
        evidence = gpu_benchmark.select_gpu_memory_evidence(
            pid=42,
            process_samples=[],
            device_samples=[{"elapsed_ms": 100.0, "devices": devices}],
            baseline_devices=[{**devices[0], "memory_used_mib": 100.0}],
            errors=[],
            interval_ms=100,
        )
        self.assertTrue(evidence["supported"])
        self.assertTrue(evidence["device_wide_fallback_used"])
        self.assertEqual(evidence["peak_memory_used_mib"], 512.0)
        self.assertEqual(evidence["peak_minus_baseline_mib"], 412.0)

    def test_pid_specific_metric_takes_precedence(self):
        evidence = gpu_benchmark.select_gpu_memory_evidence(
            pid=42,
            process_samples=[
                {"elapsed_ms": 50.0, "pid": 42, "gpu_uuid": "GPU-1", "memory_used_mib": 300.0},
                {"elapsed_ms": 100.0, "pid": 42, "gpu_uuid": "GPU-1", "memory_used_mib": 350.0},
            ],
            device_samples=[
                {
                    "elapsed_ms": 75.0,
                    "devices": [
                        {"index": 0, "name": "GPU", "uuid": "GPU-1", "memory_used_mib": 900.0}
                    ],
                }
            ],
            baseline_devices=[
                {"index": 0, "name": gpu_benchmark.TARGET_GPU_NAME, "uuid": "GPU-1", "memory_used_mib": 0.0}
            ],
            errors=[],
            interval_ms=50,
        )
        self.assertTrue(evidence["pid_specific_metric_used"])
        self.assertEqual(evidence["peak_memory_used_mib"], 350.0)
        self.assertTrue(gpu_benchmark.gpu_memory_gate_passed(evidence))

    def test_device_fallback_requires_observed_growth_over_baseline(self):
        evidence = {
            "supported": True,
            "peak_memory_used_mib": 512.0,
            "pid_specific_metric_used": False,
            "device_wide_fallback_used": True,
            "peak_minus_baseline_mib": 0.0,
        }
        self.assertFalse(gpu_benchmark.gpu_memory_gate_passed(evidence))

    def test_cache_inventory_diff_distinguishes_build_from_reuse(self):
        before = [
            {"relative_path": "engine.bin", "size_bytes": 10, "modified_time_ns": 1}
        ]
        after = [
            {"relative_path": "engine.bin", "size_bytes": 11, "modified_time_ns": 2},
            {"relative_path": "context.bin", "size_bytes": 5, "modified_time_ns": 2},
        ]
        changes = gpu_benchmark.compare_cache_inventories(before, after)
        self.assertEqual(changes["state"], "built_or_updated")
        self.assertEqual(changes["created"], ["context.bin"])
        self.assertEqual(changes["modified"], ["engine.bin"])

    def test_cache_gate_requires_non_empty_hashed_engine(self):
        self.assertFalse(
            gpu_benchmark.engine_cache_gate_passed(
                [{"relative_path": "unrelated.txt", "size_bytes": 1, "sha256": "A" * 64}]
            )
        )
        self.assertTrue(
            gpu_benchmark.engine_cache_gate_passed(
                [{"relative_path": "model.engine", "size_bytes": 1024, "sha256": "A" * 64}]
            )
        )

    def test_native_cache_gate_requires_exact_engine_path_and_sha(self):
        inventory = [
            {
                "relative_path": "frozen.engine",
                "size_bytes": 1024,
                "sha256": gpu_benchmark.NATIVE_ENGINE_SHA256,
            }
        ]
        self.assertTrue(
            gpu_benchmark.engine_cache_gate_passed(
                inventory,
                expected_relative_path="frozen.engine",
                expected_sha256=gpu_benchmark.NATIVE_ENGINE_SHA256,
            )
        )
        self.assertFalse(
            gpu_benchmark.engine_cache_gate_passed(
                inventory,
                expected_relative_path="other.engine",
                expected_sha256=gpu_benchmark.NATIVE_ENGINE_SHA256,
            )
        )

    def test_extracts_existing_benchmark_metrics_without_reinterpreting_boundaries(self):
        metrics = gpu_benchmark.extract_benchmark_metrics(
            benchmark_document(),
            expected_requested_provider="tensorrt",
            expected_actual_provider="TensorrtExecutionProvider",
        )
        self.assertEqual(metrics["session_initialization_ms"], 1234.5)
        self.assertEqual(metrics["latency_ms"]["session_run"]["p50_ms"], 1.8)
        self.assertEqual(metrics["latency_ms"]["end_to_end"]["p95_ms"], 2.7)
        self.assertEqual(metrics["host_peak_rss"]["mebibytes"], 256.0)

    def test_extracts_native_metrics_only_with_exact_direct_execution_evidence(self):
        document = benchmark_document()
        document["evidence_type"] = (
            "cpp_native_tensorrt_single_image_release_benchmark"
        )
        document["runtime"]["requested_provider"] = "tensorrt_native"
        document["runtime"]["actual_provider"] = "TensorRTNative"
        document["runtime"]["provider_evidence"] = (
            gpu_benchmark.expected_native_provider_evidence()
        )
        metrics = gpu_benchmark.extract_benchmark_metrics(
            document,
            expected_requested_provider="tensorrt_native",
            expected_actual_provider="TensorRTNative",
            expected_source_evidence_type=(
                "cpp_native_tensorrt_single_image_release_benchmark"
            ),
            expected_provider_evidence=(
                gpu_benchmark.expected_native_provider_evidence()
            ),
            precision="mixed_fp16_fp32",
        )
        self.assertEqual(metrics["precision"], "mixed_fp16_fp32")
        document["runtime"]["provider_evidence"] += ";fallback=cpu"
        with self.assertRaises(gpu_benchmark.GpuBenchmarkError):
            gpu_benchmark.extract_benchmark_metrics(
                document,
                expected_requested_provider="tensorrt_native",
                expected_actual_provider="TensorRTNative",
                expected_provider_evidence=(
                    gpu_benchmark.expected_native_provider_evidence()
                ),
            )


if __name__ == "__main__":
    unittest.main()

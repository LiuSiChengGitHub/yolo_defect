#!/usr/bin/env python3
"""Pure in-memory tests for the S2-01 evidence assembler."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER_PATH = CPP_INFER_ROOT / "tools" / "assemble_s2_01_evidence.py"


def import_tool():
    spec = importlib.util.spec_from_file_location("s2_01_evidence_assembler", ASSEMBLER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {ASSEMBLER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assembler = import_tool()


SOURCE_SHA = "A" * 64
DERIVED_SHA = "B" * 64
PROTOCOL_SHA = "0EC9A7B1CF5E4F246CF3AC15275EF06D7C67FB6C0CE11C5218391CFACE5B73F2"
CALIBRATION_SHA = "D" * 64
PRODUCT_SHA = "E" * 64
QUALITY_SHA = "F" * 64
SAMPLE_SHA = "1" * 64
FP32_TRACE_SHA = "2" * 64
INT8_TRACE_SHA = "3" * 64
SOURCE_SIZE = 12_000_000
DERIVED_SIZE = 4_000_000


def metric_pair(fp32, int8, lower_is_better):
    return dict(
        assembler._expected_metric_comparison(
            float(fp32), float(int8), lower_is_better=lower_is_better
        )
    )


def protocol_document():
    correctness = {
        "consistency_manifest": {
            "path": "consistency.json",
            "manifest_id": "product_v1",
            "sha256_canonical_lf": PRODUCT_SHA,
            "sample_count": 30,
        },
        "quality_manifest": {
            "path": "quality.json",
            "manifest_id": "quality_v1",
            "sha256_canonical_lf": QUALITY_SHA,
            "sample_count": 361,
            "ground_truth_box_count": 857,
        },
        "gate_sources": {
            "matching_protocol": "quality_manifest.product_matching_protocol",
            "product_matching": "quality_manifest.product_matching_gates",
            "task_quality": "quality_manifest.quality_gates",
        },
    }
    benchmark = {
        "build_type": "Release",
        "execution_provider": "CPUExecutionProvider",
        "execution_mode": "sequential",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "graph_optimization_level": "all",
        "warmup": 10,
        "repeat": 100,
        "profiling_enabled": False,
        "sample": {
            "sample_id": "crazing_241",
            "image_path": "image.jpg",
            "image_sha256": SAMPLE_SHA,
        },
    }
    profiling = {
        "execution_provider": "CPUExecutionProvider",
        "runs": 10,
        "sample_source": "benchmark.sample",
        "separate_from_formal_benchmark": True,
        "performance_gate": False,
    }
    return {
        "schema_version": 1,
        "protocol_id": assembler.FROZEN_PROTOCOL_ID,
        "source_model": {
            "path": "best.onnx",
            "sha256": SOURCE_SHA,
            "size_bytes": SOURCE_SIZE,
        },
        "calibration": {
            "manifest_path": "calibration.json",
            "manifest_id": "calibration_v1",
            "manifest_sha256_canonical_lf": CALIBRATION_SHA,
            "sample_count": 180,
        },
        "quantization": {
            "expected_selected_node_count": 64,
            "nodes_to_exclude": [],
        },
        "correctness": correctness,
        "benchmark": benchmark,
        "profiling": profiling,
    }


def quant_report(protocol):
    selected_nodes = [f"Conv_{index}" for index in range(64)]
    return {
        "schema_version": 1,
        "evidence_type": "s2_01_static_ptq_artifact_card",
        "passed": True,
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "canonical_lf_sha256": PROTOCOL_SHA,
            "raw_sha256": PROTOCOL_SHA,
        },
        "artifact_contract": {
            "source_model": {"sha256": SOURCE_SHA, "size_bytes": SOURCE_SIZE},
            "quantized_op_scope": {
                "op_types_to_quantize": ["Conv"],
                "selected_conv_count": 64,
                "excluded_conv_count": 0,
                "excluded_conv_nodes": [],
                "target_conv_count": 64,
                "unselected_source_nodes_remain_in_declared_precision": True,
            },
        },
        "quantization": {"parameters": copy.deepcopy(protocol["quantization"])},
        "artifacts": {
            "source": {
                "path": "best.onnx",
                "sha256": SOURCE_SHA,
                "size_bytes": SOURCE_SIZE,
                "onnx_checker": "passed",
            },
            "derived": {
                "path": "best.int8.onnx",
                "sha256": DERIVED_SHA,
                "size_bytes": DERIVED_SIZE,
                "onnx_checker": "passed",
            },
        },
        "calibration": {
            "manifest_id": "calibration_v1",
            "manifest_sha256_canonical_lf": CALIBRATION_SHA,
            "sample_count_expected": 180,
            "sample_count_hash_verified": 180,
            "sample_count_consumed": 180,
        },
        "frozen_downstream_protocol": {
            "correctness": {"declaration": copy.deepcopy(protocol["correctness"])},
            "benchmark": {"declaration": copy.deepcopy(protocol["benchmark"])},
            "profiling": copy.deepcopy(protocol["profiling"]),
        },
        "graph_audit": {
            "selection": {
                "selected_count": 64,
                "selected_conv_nodes": selected_nodes,
                "excluded_conv_count": 0,
                "excluded_conv_nodes": [],
                "target_conv_count": 64,
                "target_conv_nodes": selected_nodes,
            },
            "result": {
                "quantized_conv_count": 64,
                "quantized_conv_nodes": selected_nodes,
                "intentional_unquantized_conv_count": 0,
                "intentional_unquantized_conv_nodes": [],
                "unquantized_conv_count": 0,
                "unquantized_conv_nodes": [],
                "failed_conv_count": 0,
                "failed_conv_nodes": [],
                "excluded_policy_violation_count": 0,
                "excluded_policy_violations": [],
            },
        },
        "runtime_validation": {
            "source_python_ort": {
                "status": "passed",
                "output": {"all_finite": True},
            },
            "derived_python_ort": {
                "status": "passed",
                "output": {"all_finite": True},
            },
        },
        "model_size_comparison": {
            "source_fp32_size_bytes": SOURCE_SIZE,
            "derived_int8_size_bytes": DERIVED_SIZE,
            "size_delta_bytes": DERIVED_SIZE - SOURCE_SIZE,
            "int8_to_fp32_ratio": DERIVED_SIZE / SOURCE_SIZE,
            "size_reduction_percent": (1.0 - DERIVED_SIZE / SOURCE_SIZE) * 100.0,
        },
    }


def correctness_document():
    return {
        "schema_version": 1,
        "evidence_type": "s2_01_fp32_int8_correctness_and_quality",
        "passed": True,
        "runtime_legality": {
            "python_fp32_session_and_finite_outputs": True,
            "python_int8_session_and_finite_outputs": True,
            "cpp": {"requested": True, "passed": True},
        },
        "protocol": {
            "protocol_id": assembler.FROZEN_PROTOCOL_ID,
            "canonical_lf_sha256": PROTOCOL_SHA,
            "profiler_or_benchmark_enabled": False,
        },
        "manifests": {
            "calibration": {
                "manifest_id": "calibration_v1",
                "canonical_lf_sha256": CALIBRATION_SHA,
            },
            "product": {
                "manifest_id": "product_v1",
                "canonical_lf_sha256": PRODUCT_SHA,
            },
            "quality": {
                "manifest_id": "quality_v1",
                "canonical_lf_sha256": QUALITY_SHA,
            },
        },
        "artifacts": {
            "fp32": {
                "model_path": "best.onnx",
                "model_sha256": SOURCE_SHA,
                "model_size_bytes": SOURCE_SIZE,
            },
            "int8": {
                "model_path": "best.int8.onnx",
                "model_sha256": DERIVED_SHA,
                "model_size_bytes": DERIVED_SIZE,
            },
        },
        "product_detection_difference": {"passed": True},
        "task_quality": {"passed": True},
    }


def benchmark_document(precision, value):
    model_sha = SOURCE_SHA if precision == "fp32" else DERIVED_SHA
    model_size = SOURCE_SIZE if precision == "fp32" else DERIVED_SIZE
    return {
        "schema_version": 1,
        "evidence_type": "cpp_ort_single_image_release_benchmark",
        "protocol": {
            "batch_size": 1,
            "sample_count": 1,
            "warmup": 10,
            "repeat": 100,
        },
        "environment": {
            "machine": {"hostname": "test"},
            "build": {"type": "Release", "cxx_standard": 17},
        },
        "runtime": {
            "requested_provider": "cpu",
            "actual_provider": "CPUExecutionProvider",
            "session": {
                "execution_mode": "sequential",
                "intra_op_num_threads": 1,
                "inter_op_num_threads": 1,
                "graph_optimization_level": "all",
                "profiling_enabled": False,
                "initialization_ms": float(value) / 2.0,
            },
        },
        "model": {
            "model_id": precision,
            "model_family": "yolov8",
            "path": "best.onnx" if precision == "fp32" else "best.int8.onnx",
            "declared_sha256": model_sha,
            "file_size_bytes": model_size,
            "opset": 17,
            "input": {"name": "images", "shape": [1, 3, 800, 800]},
        },
        "sample": {"image_path": "image.jpg", "sample_count": 1},
        "postprocess": {
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
            "nms_mode": "class_agnostic",
            "detection_count": 3,
        },
        "latency_ms": {
            segment: {
                "sample_count": 100,
                "mean": float(value),
                "p50": float(value),
                "p95": float(value) * 1.1,
            }
            for segment in assembler.LATENCY_SEGMENTS
        },
        "throughput_images_per_second": {
            "pipeline": 1000.0 / float(value),
            "end_to_end": 900.0 / float(value),
        },
        "memory": {
            "status": "supported",
            "metric": "peak_working_set",
            "bytes": int(value * 1_000_000),
        },
    }


def benchmark_comparison(fp32, int8, *, correctness_passed=True, policy="required"):
    latency = {}
    for segment in assembler.LATENCY_SEGMENTS:
        latency[segment] = {
            statistic: metric_pair(
                fp32["latency_ms"][segment][statistic],
                int8["latency_ms"][segment][statistic],
                True,
            )
            for statistic in ("mean", "p50", "p95")
        }
    return {
        "schema_version": 1,
        "evidence_type": "s2_01_fp32_int8_cpp_benchmark_comparison",
        "passed": True,
        "correctness_prerequisite": {
            "policy": policy,
            "passed": correctness_passed,
            "blocking": policy == "required",
            "accepted_for_comparison": correctness_passed or policy == "advisory",
            "evidence_type": "s2_01_fp32_int8_correctness_and_quality",
        },
        "protocol_binding": {
            "protocol_id": assembler.FROZEN_PROTOCOL_ID,
            "canonical_lf_sha256": PROTOCOL_SHA,
            "source_model_sha256": SOURCE_SHA,
            "derived_model_sha256": DERIVED_SHA,
            "warmup": 10,
            "repeat": 100,
        },
        "comparability": {
            "same_machine_environment": True,
            "same_release_build": True,
            "same_provider_and_threads": True,
            "same_sample_and_postprocess": True,
            "same_warmup_repeat": True,
            "profiling_disabled": True,
        },
        "models": {
            "fp32": {
                "sha256": SOURCE_SHA,
                "file_size_bytes": SOURCE_SIZE,
            },
            "int8": {
                "sha256": DERIVED_SHA,
                "file_size_bytes": DERIVED_SIZE,
            },
            "size": {
                "int8_minus_fp32_bytes": DERIVED_SIZE - SOURCE_SIZE,
                "int8_div_fp32": DERIVED_SIZE / SOURCE_SIZE,
                "reduction_fraction": 1.0 - DERIVED_SIZE / SOURCE_SIZE,
            },
        },
        "session_initialization_ms": metric_pair(
            fp32["runtime"]["session"]["initialization_ms"],
            int8["runtime"]["session"]["initialization_ms"],
            True,
        ),
        "latency_ms": latency,
        "throughput_images_per_second": {
            name: metric_pair(
                fp32["throughput_images_per_second"][name],
                int8["throughput_images_per_second"][name],
                False,
            )
            for name in ("pipeline", "end_to_end")
        },
        "peak_working_set_bytes": metric_pair(
            fp32["memory"]["bytes"], int8["memory"]["bytes"], True
        ),
        "interpretation": {
            "speed_is_not_a_pass_condition": True,
            "pipeline_mean_outcome": "int8_worse",
            "session_run_mean_outcome": "int8_worse",
        },
    }


def profile_document(precision, trace_sha):
    model_sha = SOURCE_SHA if precision == "fp32" else DERIVED_SHA
    model_size = SOURCE_SIZE if precision == "fp32" else DERIVED_SIZE
    return {
        "schema_version": 1,
        "evidence_type": "onnxruntime_node_profile_summary",
        "passed": True,
        "trace": {
            "path": f"{precision}.trace.json",
            "size_bytes": 1000,
            "sha256": trace_sha,
            "session_model_run_event_count": 10,
        },
        "model": {
            "model_id": precision,
            "declared_sha256": model_sha,
            "precision": precision,
            "trace_precision_signature": {
                "verified": True,
                "method": "optimized_graph_operator_inventory",
            },
        },
        "artifact": {
            "model_path": "best.onnx" if precision == "fp32" else "best.int8.onnx",
            "model_sha256": model_sha,
            "model_size_bytes": model_size,
        },
        "protocol_binding": {
            "protocol_id": assembler.FROZEN_PROTOCOL_ID,
            "canonical_lf_sha256": PROTOCOL_SHA,
            "profile_runs": 10,
            "provider": "CPUExecutionProvider",
            "sample_sha256": SAMPLE_SHA,
            "sample_path": "image.jpg",
            "separate_from_formal_benchmark": True,
        },
        "protocol": {
            "expected_profile_runs": 10,
            "expected_provider": "CPUExecutionProvider",
        },
        "result": {
            "kernel_event_total_ms": 12.0,
            "unique_node_count": 2,
            "node_call_count_min": 10,
            "node_call_count_max": 10,
            "providers": [{"provider": "CPUExecutionProvider", "calls": 20}],
            "top_nodes": [{"node_name": "Conv_0", "total_ms": 8.0}],
            "top_operators": [{"op_type": "Conv", "total_ms": 8.0}],
        },
        "segmented_benchmark_mapping": {
            "outer_metric": "latency_ms.session_run",
            "excluded_from_formal_benchmark": True,
        },
        "profiling_overhead": {"present": True, "quantified": False},
    }


def valid_bundle():
    protocol = protocol_document()
    fp32 = benchmark_document("fp32", 10.0)
    int8 = benchmark_document("int8", 12.0)
    return {
        "protocol_document": protocol,
        "protocol_sha256": PROTOCOL_SHA,
        "quant_report": quant_report(protocol),
        "correctness_document": correctness_document(),
        "fp32_benchmark_document": fp32,
        "int8_benchmark_document": int8,
        "benchmark_comparison_document": benchmark_comparison(fp32, int8),
        "fp32_profile_document": profile_document("fp32", FP32_TRACE_SHA),
        "int8_profile_document": profile_document("int8", INT8_TRACE_SHA),
        "integrity": {
            "protocol": {"sha256": PROTOCOL_SHA},
            "source_model": {"sha256": SOURCE_SHA, "size_bytes": SOURCE_SIZE},
            "derived_model": {"sha256": DERIVED_SHA, "size_bytes": DERIVED_SIZE},
            "calibration_manifest": {"sha256": CALIBRATION_SHA},
            "product_manifest": {"sha256": PRODUCT_SHA},
            "quality_manifest": {"sha256": QUALITY_SHA},
            "benchmark_sample": {"sha256": SAMPLE_SHA},
        },
        "trace_integrity": {
            "fp32": {
                "path": "fp32.trace.json",
                "sha256": FP32_TRACE_SHA,
                "size_bytes": 1000,
            },
            "int8": {
                "path": "int8.trace.json",
                "sha256": INT8_TRACE_SHA,
                "size_bytes": 1000,
            },
        },
        "path_bindings": {key: True for key in assembler.PATH_BINDING_KEYS},
    }


class EvidenceAssemblerTest(unittest.TestCase):
    def assemble(self, bundle):
        return assembler.assemble_documents(**bundle)

    def test_slower_int8_is_valid_evidence_and_not_a_speed_gate(self):
        result = self.assemble(valid_bundle())
        self.assertTrue(result["passed"])
        self.assertEqual("int8_worse", result["performance"]["pipeline_mean_outcome"])
        self.assertFalse(result["policy"]["int8_speed_is_pass_gate"])

    def test_rejects_failed_correctness(self):
        bundle = valid_bundle()
        bundle["correctness_document"]["passed"] = False
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "correctness.passed"):
            self.assemble(bundle)

    def test_advisory_policy_preserves_failed_quality_status_and_completes(self):
        bundle = valid_bundle()
        bundle["correctness_document"]["passed"] = False
        bundle["correctness_document"]["product_detection_difference"]["passed"] = False
        bundle["benchmark_comparison_document"] = benchmark_comparison(
            bundle["fp32_benchmark_document"],
            bundle["int8_benchmark_document"],
            correctness_passed=False,
            policy="advisory",
        )
        bundle["correctness_policy"] = "advisory"
        result = self.assemble(bundle)
        self.assertTrue(result["passed"])
        self.assertFalse(result["strict_acceptance_passed"])
        self.assertEqual(result["evidence_type"], assembler.ADVISORY_EVIDENCE_TYPE)
        self.assertFalse(result["correctness"]["reported_passed"])
        self.assertFalse(
            result["correctness"]["product_detection_difference_passed"]
        )
        self.assertTrue(result["correctness"]["task_quality_passed"])
        self.assertFalse(result["policy"]["correctness_results_rewritten"])

    def test_rejects_failed_quantization_report(self):
        bundle = valid_bundle()
        bundle["quant_report"]["passed"] = False
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "quant_report.passed"):
            self.assemble(bundle)

    def test_rejects_protocol_hash_drift(self):
        bundle = valid_bundle()
        bundle["quant_report"]["protocol"]["canonical_lf_sha256"] = "0" * 64
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "canonical_lf_sha256"):
            self.assemble(bundle)

    def test_rejects_quantization_parameter_drift(self):
        bundle = valid_bundle()
        bundle["quant_report"]["quantization"]["parameters"]["format"] = "QOperator"
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "quantization.parameters"):
            self.assemble(bundle)

    def test_rejects_quantized_node_identity_drift(self):
        bundle = valid_bundle()
        bundle["quant_report"]["graph_audit"]["result"]["quantized_conv_nodes"] = []
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "quantized_conv_nodes"):
            self.assemble(bundle)

    def test_rejects_artifact_scope_drift(self):
        bundle = valid_bundle()
        bundle["quant_report"]["artifact_contract"]["quantized_op_scope"]["target_conv_count"] = 63
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "quantized_op_scope"):
            self.assemble(bundle)

    def test_rejects_derived_model_lineage_drift(self):
        bundle = valid_bundle()
        bundle["correctness_document"]["artifacts"]["int8"]["model_sha256"] = "0" * 64
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "artifacts.int8.model_sha256"):
            self.assemble(bundle)

    def test_rejects_quality_manifest_hash_drift(self):
        bundle = valid_bundle()
        bundle["correctness_document"]["manifests"]["quality"]["canonical_lf_sha256"] = "0" * 64
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "manifests.quality"):
            self.assemble(bundle)

    def test_rejects_comparison_not_derived_from_raw_benchmarks(self):
        bundle = valid_bundle()
        bundle["benchmark_comparison_document"]["latency_ms"]["session_run"]["mean"]["int8"] = 99.0
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "latency_ms.session_run.mean.int8"):
            self.assemble(bundle)

    def test_rejects_raw_trace_hash_drift(self):
        bundle = valid_bundle()
        bundle["trace_integrity"]["int8"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "integrity.int8_trace.sha256"):
            self.assemble(bundle)

    def test_rejects_unbound_recorded_path(self):
        bundle = valid_bundle()
        bundle["path_bindings"]["fp32_benchmark_sample"] = False
        with self.assertRaisesRegex(assembler.EvidenceAssemblyError, "path_bindings.fp32_benchmark_sample"):
            self.assemble(bundle)


if __name__ == "__main__":
    unittest.main()

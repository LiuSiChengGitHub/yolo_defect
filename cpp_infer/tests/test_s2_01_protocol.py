"""Pure standard-library tests for the frozen S2-01 machine protocol."""

from __future__ import annotations

import builtins
import copy
import importlib
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_INFER_ROOT = REPO_ROOT / "cpp_infer"
TOOLS_ROOT = CPP_INFER_ROOT / "tools"
FIXTURE_ROOT = CPP_INFER_ROOT / "tests" / "fixtures"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import s2_01_protocol as protocol  # noqa: E402


ORIGINAL_LOAD_JSON = protocol.load_json
SOURCE_MODEL_PATH = REPO_ROOT / "models" / "best.onnx"
CALIBRATION_PATH = FIXTURE_ROOT / "s2_01_calibration_manifest.json"
CONSISTENCY_PATH = FIXTURE_ROOT / "consistency_manifest.json"
QUALITY_PATH = FIXTURE_ROOT / "s2_01_quality_manifest.json"
BENCHMARK_PATH = REPO_ROOT / "data" / "images" / "val" / "crazing_241.jpg"


class S201ProtocolTest(unittest.TestCase):
    """Inject the root JSON in memory and read tracked frozen inputs only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.declaration_path = Path(__file__).resolve()
        cls.calibration_document = ORIGINAL_LOAD_JSON(
            CALIBRATION_PATH, "test.calibration"
        )

    def _protocol_document(self) -> dict:
        return {
            "schema_version": 1,
            "protocol_id": protocol.PROTOCOL_ID_V1,
            "source_model": {
                "path": str(SOURCE_MODEL_PATH),
                "sha256": protocol.sha256_file_raw(SOURCE_MODEL_PATH),
                "size_bytes": SOURCE_MODEL_PATH.stat().st_size,
            },
            "calibration": {
                "manifest_path": str(CALIBRATION_PATH),
                "manifest_id": self.calibration_document["manifest_id"],
                "manifest_sha256_canonical_lf": (
                    protocol.sha256_file_canonical_lf(CALIBRATION_PATH)
                ),
                "sample_count": protocol.FROZEN_CALIBRATION_SAMPLE_COUNT,
                "expected_class_counts": {
                    entry["class_name"]: 30
                    for entry in protocol.EXPECTED_CALIBRATION_CLASSES
                },
                "preprocess": dict(protocol.EXPECTED_CALIBRATION_PREPROCESS),
            },
            "quantization": copy.deepcopy(
                protocol.expected_quantization_for_protocol(
                    protocol.PROTOCOL_ID_V1
                )
            ),
            "model_contract": {
                "input_name": "images",
                "input_shape": [1, 3, 800, 800],
                "input_dtype": "float32",
                "output_name": "output0",
                "output_shape": [1, 10, 13125],
                "output_dtype": "float32",
            },
            "environment": dict(protocol.EXPECTED_ENVIRONMENT),
            "correctness": {
                "consistency_manifest": {
                    "path": str(CONSISTENCY_PATH),
                    "manifest_id": protocol.FROZEN_CONSISTENCY_MANIFEST_ID,
                    "sha256_canonical_lf": (
                        protocol.sha256_file_canonical_lf(CONSISTENCY_PATH)
                    ),
                    "sample_count": protocol.FROZEN_CONSISTENCY_SAMPLE_COUNT,
                },
                "quality_manifest": {
                    "path": str(QUALITY_PATH),
                    "manifest_id": protocol.FROZEN_QUALITY_MANIFEST_ID,
                    "sha256_canonical_lf": (
                        protocol.sha256_file_canonical_lf(QUALITY_PATH)
                    ),
                    "sample_count": protocol.FROZEN_QUALITY_SAMPLE_COUNT,
                    "ground_truth_box_count": (
                        protocol.FROZEN_QUALITY_GROUND_TRUTH_BOX_COUNT
                    ),
                },
                "gate_sources": dict(
                    protocol.EXPECTED_CORRECTNESS_GATE_SOURCES
                ),
            },
            "benchmark": {
                **dict(protocol.EXPECTED_BENCHMARK_SETTINGS),
                "sample": {
                    "sample_id": "crazing_241",
                    "image_path": str(BENCHMARK_PATH),
                    "image_sha256": protocol.sha256_file_raw(BENCHMARK_PATH),
                },
            },
            "profiling": dict(protocol.EXPECTED_PROFILING),
            "output": {
                "model_path": str(
                    CPP_INFER_ROOT / "results" / "unit_test_int8.onnx"
                ),
                "report_path": str(
                    CPP_INFER_ROOT / "results" / "unit_test_int8.card.json"
                ),
            },
        }

    def _load(
        self,
        document: dict,
        calibration_override: dict = None,
        raw_sha_side_effect=None,
    ):
        declaration_key = str(self.declaration_path).casefold()
        calibration_key = str(CALIBRATION_PATH.resolve()).casefold()

        def in_memory_load(path, object_name):
            path_key = str(Path(path).resolve()).casefold()
            if path_key == declaration_key:
                return document
            if calibration_override is not None and path_key == calibration_key:
                return calibration_override
            return ORIGINAL_LOAD_JSON(Path(path), object_name)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    protocol, "load_json", side_effect=in_memory_load
                )
            )
            if raw_sha_side_effect is not None:
                stack.enter_context(
                    mock.patch.object(
                        protocol,
                        "sha256_file_raw",
                        side_effect=raw_sha_side_effect,
                    )
                )
            return protocol.load_s2_01_protocol(self.declaration_path)

    def test_loads_complete_frozen_protocol(self) -> None:
        loaded = self._load(self._protocol_document())

        self.assertEqual(180, len(loaded.calibration_samples))
        self.assertTrue(CONSISTENCY_PATH.samefile(loaded.consistency_manifest_path))
        self.assertTrue(QUALITY_PATH.samefile(loaded.quality_manifest_path))
        self.assertTrue(BENCHMARK_PATH.samefile(loaded.benchmark_sample_path))
        self.assertIsNone(loaded.quality_evaluation["max_detections_per_image"])
        self.assertEqual(
            "COCO_style_101_point_without_area_ranges_or_max_dets",
            loaded.quality_evaluation["metric_claim"],
        )
        self.assertEqual(
            protocol.EXPECTED_PRODUCT_MATCHING_PROTOCOL,
            loaded.product_matching_protocol,
        )
        self.assertEqual(
            protocol.EXPECTED_PRODUCT_MATCHING_GATES,
            loaded.product_matching_gates,
        )
        self.assertFalse(loaded.profiling["performance_gate"])

    def test_loads_v2_with_exact_head_fp32_exclusions(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V2
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V2)
        )

        loaded = self._load(document)

        self.assertEqual(protocol.PROTOCOL_ID_V2, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            list(protocol.FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES),
            loaded.quantization["nodes_to_exclude"],
        )
        self.assertEqual(
            protocol.FROZEN_HEAD_FP32_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT
            - len(loaded.quantization["nodes_to_exclude"]),
        )

    def test_loads_v3_with_entropy_and_v2_exclusions(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V3
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V3)
        )

        loaded = self._load(document)

        self.assertEqual(protocol.PROTOCOL_ID_V3, loaded.protocol_id)
        self.assertEqual("Entropy", loaded.quantization["calibrate_method"])
        self.assertEqual(
            list(protocol.FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES),
            loaded.quantization["nodes_to_exclude"],
        )

        quantize = importlib.import_module("quantize_s2_01")
        minmax = SimpleNamespace(name="MinMax")
        entropy = SimpleNamespace(name="Entropy")
        dependencies = SimpleNamespace(
            CalibrationMethod=SimpleNamespace(
                MinMax=minmax, Entropy=entropy
            ),
            entropy_calibrater_signature=(
                "(model_path, symmetric=False, num_bins=128, "
                "num_quantized_bins=128)"
            ),
        )
        self.assertIs(
            entropy,
            quantize._resolve_calibration_method(dependencies, "Entropy"),
        )
        evidence = quantize._calibration_method_evidence(
            loaded.quantization, dependencies
        )
        self.assertEqual("Entropy", evidence["name"])
        self.assertFalse(evidence["symmetric"])
        self.assertEqual(
            (
                "protocol.quantization.extra_options."
                "CalibTensorRangeSymmetric"
            ),
            evidence["symmetric_source"],
        )
        self.assertEqual(
            {"num_bins": 128, "num_quantized_bins": 128},
            {
                key: evidence["entropy_histogram"][key]
                for key in ("num_bins", "num_quantized_bins")
            },
        )

    def test_loads_v4_with_backbone_only_targets(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V4
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V4)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V4, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual("/model.12/cv1/conv/Conv", excluded[0])
        self.assertEqual(
            list(protocol.FROZEN_BACKBONE_ONLY_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES),
            excluded[-protocol.FROZEN_HEAD_FP32_EXCLUDED_CONV_COUNT :],
        )
        self.assertEqual(
            protocol.FROZEN_BACKBONE_ONLY_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_v5_with_early_backbone_targets(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V5
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V5)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V5, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_EARLY_BACKBONE_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual("/model.5/conv/Conv", excluded[0])
        self.assertEqual(
            list(protocol.FROZEN_EARLY_BACKBONE_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_LATE_BACKBONE_TARGET_CONV_NODES),
            excluded[: protocol.FROZEN_LATE_BACKBONE_TARGET_CONV_COUNT],
        )
        self.assertEqual(
            protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_v6_with_late_backbone_targets(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V6
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V6)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V6, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_LATE_BACKBONE_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual(
            list(protocol.FROZEN_LATE_BACKBONE_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_NODES),
            excluded[: protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT],
        )
        self.assertEqual(
            "/model.12/cv1/conv/Conv",
            excluded[protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT],
        )
        self.assertEqual(
            protocol.FROZEN_LATE_BACKBONE_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_v7_with_mid_backbone_targets(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V7
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V7)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V7, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_MID_BACKBONE_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual(
            list(protocol.FROZEN_MID_BACKBONE_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_DEEP_BACKBONE_TARGET_CONV_NODES),
            excluded[
                protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT :
                protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT
                + protocol.FROZEN_DEEP_BACKBONE_TARGET_CONV_COUNT
            ],
        )
        self.assertEqual(
            protocol.FROZEN_MID_BACKBONE_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_v8_with_deep_backbone_targets(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V8
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V8)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V8, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_DEEP_BACKBONE_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual(
            list(protocol.FROZEN_DEEP_BACKBONE_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_MID_BACKBONE_TARGET_CONV_NODES),
            excluded[
                protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT :
                protocol.FROZEN_EARLY_BACKBONE_TARGET_CONV_COUNT
                + protocol.FROZEN_MID_BACKBONE_TARGET_CONV_COUNT
            ],
        )
        self.assertEqual(
            protocol.FROZEN_DEEP_BACKBONE_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_v9_with_model0_2_prefix_targets(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V9
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V9)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V9, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_PREFIX_MODEL0_2_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual("/model.3/conv/Conv", excluded[0])
        self.assertEqual(
            list(protocol.FROZEN_PREFIX_MODEL0_2_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_PREFIX_MODEL0_2_TARGET_CONV_NODES),
            list(
                protocol.FROZEN_SOURCE_ORDERED_CONV_NODES[
                    : protocol.FROZEN_PREFIX_MODEL0_2_TARGET_CONV_COUNT
                ]
            ),
        )
        self.assertEqual(
            protocol.FROZEN_PREFIX_MODEL0_2_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_v10_with_model0_1_prefix_targets(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V10
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V10)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V10, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_PREFIX_MODEL0_1_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual("/model.2/cv1/conv/Conv", excluded[0])
        self.assertEqual(
            list(protocol.FROZEN_PREFIX_MODEL0_1_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_PREFIX_MODEL0_1_TARGET_CONV_NODES),
            list(
                protocol.FROZEN_SOURCE_ORDERED_CONV_NODES[
                    : protocol.FROZEN_PREFIX_MODEL0_1_TARGET_CONV_COUNT
                ]
            ),
        )
        self.assertEqual(
            protocol.FROZEN_PREFIX_MODEL0_1_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_v11_with_model0_prefix_target(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_V11
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(protocol.PROTOCOL_ID_V11)
        )

        loaded = self._load(document)
        excluded = loaded.quantization["nodes_to_exclude"]

        self.assertEqual(protocol.PROTOCOL_ID_V11, loaded.protocol_id)
        self.assertEqual("MinMax", loaded.quantization["calibrate_method"])
        self.assertEqual(
            protocol.FROZEN_PREFIX_MODEL0_EXCLUDED_CONV_COUNT,
            len(excluded),
        )
        self.assertEqual("/model.1/conv/Conv", excluded[0])
        self.assertEqual(
            list(protocol.FROZEN_PREFIX_MODEL0_EXCLUDED_CONV_NODES),
            excluded,
        )
        self.assertEqual(
            list(protocol.FROZEN_PREFIX_MODEL0_TARGET_CONV_NODES),
            list(
                protocol.FROZEN_SOURCE_ORDERED_CONV_NODES[
                    : protocol.FROZEN_PREFIX_MODEL0_TARGET_CONV_COUNT
                ]
            ),
        )
        self.assertEqual(
            protocol.FROZEN_PREFIX_MODEL0_TARGET_CONV_COUNT,
            protocol.FROZEN_SELECTED_CONV_COUNT - len(excluded),
        )

    def test_loads_round2_u8s8_as_single_variable_change(self) -> None:
        document = self._protocol_document()
        document["protocol_id"] = protocol.PROTOCOL_ID_R2_U8S8
        document["quantization"] = copy.deepcopy(
            protocol.expected_quantization_for_protocol(
                protocol.PROTOCOL_ID_R2_U8S8
            )
        )

        loaded = self._load(document)

        self.assertEqual(protocol.PROTOCOL_ID_R2_U8S8, loaded.protocol_id)
        self.assertEqual("QUInt8", loaded.quantization["activation_type"])
        self.assertEqual("QInt8", loaded.quantization["weight_type"])
        self.assertEqual([], loaded.quantization["nodes_to_exclude"])
        baseline = protocol.expected_quantization_for_protocol(
            protocol.PROTOCOL_ID_V1
        )
        changed_keys = {
            key
            for key in baseline
            if baseline[key] != loaded.quantization[key]
        }
        self.assertEqual({"activation_type"}, changed_keys)

    def test_rejects_cross_protocol_calibration_methods(self) -> None:
        cases = (
            (protocol.PROTOCOL_ID_V1, "Entropy"),
            (protocol.PROTOCOL_ID_V2, "Entropy"),
            (protocol.PROTOCOL_ID_V3, "MinMax"),
            (protocol.PROTOCOL_ID_V4, "Entropy"),
            (protocol.PROTOCOL_ID_V5, "Entropy"),
            (protocol.PROTOCOL_ID_V6, "Entropy"),
            (protocol.PROTOCOL_ID_V7, "Entropy"),
            (protocol.PROTOCOL_ID_V8, "Entropy"),
            (protocol.PROTOCOL_ID_V9, "Entropy"),
            (protocol.PROTOCOL_ID_V10, "Entropy"),
            (protocol.PROTOCOL_ID_V11, "Entropy"),
        )
        for protocol_id, wrong_method in cases:
            with self.subTest(protocol_id=protocol_id):
                document = self._protocol_document()
                document["protocol_id"] = protocol_id
                document["quantization"] = copy.deepcopy(
                    protocol.expected_quantization_for_protocol(protocol_id)
                )
                document["quantization"]["calibrate_method"] = wrong_method
                with self.assertRaisesRegex(
                    protocol.S201ProtocolError, "calibrate_method"
                ):
                    self._load(document)

    def test_rejects_v7_v8_exclusion_drift(self) -> None:
        def wrong_count(value):
            value["quantization"]["nodes_to_exclude"].pop()

        def wrong_identity(value):
            value["quantization"]["nodes_to_exclude"][13] = "/wrong/Conv"

        def wrong_order(value):
            nodes = value["quantization"]["nodes_to_exclude"]
            nodes[13], nodes[14] = nodes[14], nodes[13]

        def duplicate(value):
            value["quantization"]["nodes_to_exclude"][-1] = value[
                "quantization"
            ]["nodes_to_exclude"][0]

        mutations = (
            ("count", wrong_count, "nodes_to_exclude"),
            ("identity", wrong_identity, "nodes_to_exclude"),
            ("order", wrong_order, "nodes_to_exclude"),
            ("duplicate", duplicate, "duplicates"),
        )
        for protocol_id in (protocol.PROTOCOL_ID_V7, protocol.PROTOCOL_ID_V8):
            for name, mutate, message in mutations:
                with self.subTest(protocol_id=protocol_id, drift=name):
                    document = self._protocol_document()
                    document["protocol_id"] = protocol_id
                    document["quantization"] = copy.deepcopy(
                        protocol.expected_quantization_for_protocol(protocol_id)
                    )
                    mutate(document)
                    with self.assertRaisesRegex(
                        protocol.S201ProtocolError, message
                    ):
                        self._load(document)

    def test_rejects_v9_v10_v11_exclusion_drift(self) -> None:
        def wrong_count(value):
            value["quantization"]["nodes_to_exclude"].pop()

        def wrong_identity(value):
            value["quantization"]["nodes_to_exclude"][0] = "/wrong/Conv"

        def wrong_order(value):
            nodes = value["quantization"]["nodes_to_exclude"]
            nodes[0], nodes[1] = nodes[1], nodes[0]

        def duplicate(value):
            value["quantization"]["nodes_to_exclude"][-1] = value[
                "quantization"
            ]["nodes_to_exclude"][0]

        mutations = (
            ("count", wrong_count, "nodes_to_exclude"),
            ("identity", wrong_identity, "nodes_to_exclude"),
            ("order", wrong_order, "nodes_to_exclude"),
            ("duplicate", duplicate, "duplicates"),
        )
        for protocol_id in (
            protocol.PROTOCOL_ID_V9,
            protocol.PROTOCOL_ID_V10,
            protocol.PROTOCOL_ID_V11,
        ):
            for name, mutate, message in mutations:
                with self.subTest(protocol_id=protocol_id, drift=name):
                    document = self._protocol_document()
                    document["protocol_id"] = protocol_id
                    document["quantization"] = copy.deepcopy(
                        protocol.expected_quantization_for_protocol(protocol_id)
                    )
                    mutate(document)
                    with self.assertRaisesRegex(
                        protocol.S201ProtocolError, message
                    ):
                        self._load(document)

    def test_rejects_v5_v6_exclusion_drift(self) -> None:
        def wrong_count(value):
            value["quantization"]["nodes_to_exclude"].pop()

        def wrong_identity(value):
            value["quantization"]["nodes_to_exclude"][0] = "/wrong/Conv"

        def wrong_order(value):
            nodes = value["quantization"]["nodes_to_exclude"]
            nodes[0], nodes[1] = nodes[1], nodes[0]

        def duplicate(value):
            value["quantization"]["nodes_to_exclude"][-1] = value[
                "quantization"
            ]["nodes_to_exclude"][0]

        mutations = (
            ("count", wrong_count, "nodes_to_exclude"),
            ("identity", wrong_identity, "nodes_to_exclude"),
            ("order", wrong_order, "nodes_to_exclude"),
            ("duplicate", duplicate, "duplicates"),
        )
        for protocol_id in (protocol.PROTOCOL_ID_V5, protocol.PROTOCOL_ID_V6):
            for name, mutate, message in mutations:
                with self.subTest(protocol_id=protocol_id, drift=name):
                    document = self._protocol_document()
                    document["protocol_id"] = protocol_id
                    document["quantization"] = copy.deepcopy(
                        protocol.expected_quantization_for_protocol(protocol_id)
                    )
                    mutate(document)
                    with self.assertRaisesRegex(
                        protocol.S201ProtocolError, message
                    ):
                        self._load(document)

    def test_rejects_v4_exclusion_count_identity_order_and_duplicates(self) -> None:
        def wrong_count(value):
            value["quantization"]["nodes_to_exclude"].pop()

        def wrong_identity(value):
            value["quantization"]["nodes_to_exclude"][0] = "/model.11/Conv"

        def wrong_order(value):
            nodes = value["quantization"]["nodes_to_exclude"]
            nodes[0], nodes[1] = nodes[1], nodes[0]

        def duplicate(value):
            value["quantization"]["nodes_to_exclude"][-1] = value[
                "quantization"
            ]["nodes_to_exclude"][0]

        cases = (
            ("count", wrong_count, "nodes_to_exclude"),
            ("identity", wrong_identity, "nodes_to_exclude"),
            ("order", wrong_order, "nodes_to_exclude"),
            ("duplicate", duplicate, "duplicates"),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                document = self._protocol_document()
                document["protocol_id"] = protocol.PROTOCOL_ID_V4
                document["quantization"] = copy.deepcopy(
                    protocol.expected_quantization_for_protocol(
                        protocol.PROTOCOL_ID_V4
                    )
                )
                mutate(document)
                with self.assertRaisesRegex(protocol.S201ProtocolError, message):
                    self._load(document)

    def test_rejects_v2_exclusion_identity_order_and_duplicates(self) -> None:
        def empty(value):
            value["quantization"]["nodes_to_exclude"] = []

        def wrong_identity(value):
            value["quantization"]["nodes_to_exclude"][-1] = "/model.21/Conv"

        def wrong_order(value):
            nodes = value["quantization"]["nodes_to_exclude"]
            nodes[0], nodes[1] = nodes[1], nodes[0]

        def duplicate(value):
            value["quantization"]["nodes_to_exclude"][-1] = value[
                "quantization"
            ]["nodes_to_exclude"][0]

        cases = (
            ("empty", empty, "nodes_to_exclude"),
            ("identity", wrong_identity, "nodes_to_exclude"),
            ("order", wrong_order, "nodes_to_exclude"),
            ("duplicate", duplicate, "duplicates"),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name):
                document = self._protocol_document()
                document["protocol_id"] = protocol.PROTOCOL_ID_V2
                document["quantization"] = copy.deepcopy(
                    protocol.expected_quantization_for_protocol(
                        protocol.PROTOCOL_ID_V2
                    )
                )
                mutate(document)
                with self.assertRaisesRegex(protocol.S201ProtocolError, message):
                    self._load(document)

    def test_rejects_any_v1_exclusion(self) -> None:
        document = self._protocol_document()
        document["quantization"]["nodes_to_exclude"] = list(
            protocol.FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES
        )

        with self.assertRaisesRegex(protocol.S201ProtocolError, "nodes_to_exclude"):
            self._load(document)

    def test_graph_audit_separates_intentional_fp32_from_failures(self) -> None:
        quantize = importlib.import_module("quantize_s2_01")

        class DataType:
            @staticmethod
            def Name(value):
                return {1: "FLOAT", 3: "INT8"}[value]

        tensor_proto = SimpleNamespace(
            FLOAT=1, INT8=3, DataType=DataType
        )
        dependencies = SimpleNamespace(
            onnx=SimpleNamespace(TensorProto=tensor_proto)
        )
        excluded_name = protocol.FROZEN_HEAD_FP32_EXCLUDED_CONV_NODES[0]
        selected_names = ["body_conv_0", excluded_name, "body_conv_1"]
        source_nodes = [
            SimpleNamespace(
                name=name, op_type="Conv", input=[], output=[f"{index}_out"]
            )
            for index, name in enumerate(selected_names)
        ]

        def initializer(name, data_type, dims):
            return SimpleNamespace(
                name=name, data_type=data_type, dims=list(dims)
            )

        def target_qdq(name, prefix):
            nodes = [
                SimpleNamespace(
                    name=f"{prefix}_activation_q",
                    op_type="QuantizeLinear",
                    input=[f"{prefix}_raw"],
                    output=[f"{prefix}_activation_quantized"],
                ),
                SimpleNamespace(
                    name=f"{prefix}_activation_dq",
                    op_type="DequantizeLinear",
                    input=[f"{prefix}_activation_quantized"],
                    output=[f"{prefix}_activation_dequantized"],
                ),
                SimpleNamespace(
                    name=f"{prefix}_weight_dq",
                    op_type="DequantizeLinear",
                    input=[f"{prefix}_weight", f"{prefix}_weight_scale"],
                    output=[f"{prefix}_weight_dequantized"],
                ),
                SimpleNamespace(
                    name=name,
                    op_type="Conv",
                    input=[
                        f"{prefix}_activation_dequantized",
                        f"{prefix}_weight_dequantized",
                    ],
                    output=[f"{prefix}_conv_output"],
                ),
                SimpleNamespace(
                    name=f"{prefix}_output_q",
                    op_type="QuantizeLinear",
                    input=[f"{prefix}_conv_output"],
                    output=[f"{prefix}_output_quantized"],
                ),
            ]
            initializers = [
                initializer(f"{prefix}_weight", tensor_proto.INT8, [2, 1, 1, 1]),
                initializer(f"{prefix}_weight_scale", tensor_proto.FLOAT, [2]),
            ]
            return nodes, initializers

        derived_nodes = []
        derived_initializers = []
        for name, prefix in (("body_conv_0", "a"), ("body_conv_1", "b")):
            nodes, initializers = target_qdq(name, prefix)
            derived_nodes.extend(nodes)
            derived_initializers.extend(initializers)
        derived_nodes.append(
            SimpleNamespace(
                name=excluded_name,
                op_type="Conv",
                input=["head_activation", "head_fp32_weight"],
                output=["head_output"],
            )
        )
        derived_initializers.append(
            initializer("head_fp32_weight", tensor_proto.FLOAT, [2, 1, 1, 1])
        )
        source_model = SimpleNamespace(
            graph=SimpleNamespace(node=source_nodes, initializer=[])
        )
        derived_model = SimpleNamespace(
            graph=SimpleNamespace(
                node=derived_nodes, initializer=derived_initializers
            )
        )

        audit = quantize._audit_qdq_graph(
            source_model,
            derived_model,
            selected_names,
            [excluded_name],
            dependencies,
        )

        self.assertEqual(2, audit["selection"]["target_conv_count"])
        self.assertEqual(1, audit["selection"]["excluded_conv_count"])
        self.assertEqual(2, audit["result"]["quantized_conv_count"])
        self.assertEqual(
            [excluded_name],
            audit["result"]["intentional_unquantized_conv_nodes"],
        )
        self.assertEqual(0, audit["result"]["failed_conv_count"])
        self.assertEqual(
            0, audit["result"]["excluded_policy_violation_count"]
        )

        excluded_node = next(
            node for node in derived_nodes if node.name == excluded_name
        )
        excluded_node.input[1] = "a_weight_dequantized"
        violation_audit = quantize._audit_qdq_graph(
            source_model,
            derived_model,
            selected_names,
            [excluded_name],
            dependencies,
        )
        self.assertEqual(0, violation_audit["result"]["failed_conv_count"])
        self.assertEqual(
            1,
            violation_audit["result"]["excluded_policy_violation_count"],
        )

    def test_canonical_manifest_hash_normalizes_only_line_endings(self) -> None:
        lf_bytes = b'{\n  "schema_version": 1\n}\n'
        crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
        cr_bytes = lf_bytes.replace(b"\n", b"\r")

        self.assertEqual(
            lf_bytes,
            protocol.canonical_lf_bytes(crlf_bytes, "test.manifest"),
        )
        self.assertEqual(
            lf_bytes,
            protocol.canonical_lf_bytes(cr_bytes, "test.manifest"),
        )
        self.assertEqual(
            protocol.sha256_bytes(lf_bytes),
            protocol.sha256_bytes(
                protocol.canonical_lf_bytes(crlf_bytes, "test.manifest")
            ),
        )

    def test_rejects_source_raw_sha_drift(self) -> None:
        document = self._protocol_document()
        document["source_model"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(protocol.S201ProtocolError, "source_model.sha256"):
            self._load(document)

    def test_rejects_calibration_image_raw_sha_drift(self) -> None:
        document = self._protocol_document()
        original_raw_hash = protocol.sha256_file_raw

        def drift_one_image(path):
            if Path(path).name == "crazing_1.jpg":
                return "0" * 64
            return original_raw_hash(Path(path))

        with self.assertRaisesRegex(protocol.S201ProtocolError, "image_sha256"):
            self._load(document, raw_sha_side_effect=drift_one_image)

    def test_rejects_manifest_sample_set_drift(self) -> None:
        document = self._protocol_document()
        calibration = copy.deepcopy(self.calibration_document)
        calibration["integrity"]["sample_set_sha256"] = "0" * 64

        with self.assertRaisesRegex(protocol.S201ProtocolError, "sample_set_sha256"):
            self._load(document, calibration_override=calibration)

    def test_rejects_downstream_protocol_drift(self) -> None:
        def wrong_quality_count(value):
            value["correctness"]["quality_manifest"]["sample_count"] = 360

        def wrong_benchmark_warmup(value):
            value["benchmark"]["warmup"] = 9

        def wrong_profile_runs(value):
            value["profiling"]["runs"] = 9

        def wrong_quantization(value):
            value["quantization"]["per_channel"] = False

        cases = (
            ("correctness", wrong_quality_count),
            ("benchmark", wrong_benchmark_warmup),
            ("profiling", wrong_profile_runs),
            ("quantization", wrong_quantization),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                document = self._protocol_document()
                mutate(document)
                with self.assertRaises(protocol.S201ProtocolError):
                    self._load(document)

    def test_quantization_module_imports_without_ml_packages(self) -> None:
        blocked_packages = {"cv2", "numpy", "onnx", "onnxruntime"}
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked_packages:
                raise AssertionError(f"eager ML dependency import: {name}")
            return original_import(name, globals, locals, fromlist, level)

        sys.modules.pop("quantize_s2_01", None)
        with mock.patch("builtins.__import__", side_effect=guarded_import):
            imported = importlib.import_module("quantize_s2_01")
        self.assertTrue(callable(imported.run_quantization))


if __name__ == "__main__":
    unittest.main()

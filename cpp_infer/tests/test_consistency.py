#!/usr/bin/env python3
"""Pure tests for the S1-07 consistency manifest and matching semantics."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = CPP_INFER_ROOT / "tools" / "compare_consistency.py"
MANIFEST_PATH = CPP_INFER_ROOT / "tests" / "fixtures" / "consistency_manifest.json"

SPEC = importlib.util.spec_from_file_location("compare_consistency", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import consistency tool from {TOOL_PATH}")
consistency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(consistency)


def detection(class_id, confidence, box):
    return {
        "class_id": class_id,
        "class_name": f"class_{class_id}",
        "confidence": confidence,
        "bbox_xyxy": list(box),
    }


def canonical_pairs(matching):
    return sorted(
        (
            consistency.detection_key(item["python_detection"]),
            consistency.detection_key(item["cpp_detection"]),
        )
        for item in matching["matches"]
    )


class FrozenManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_manifest = consistency.load_json(MANIFEST_PATH)
        config_path = (
            MANIFEST_PATH.parent / raw_manifest["config_path"]
        ).resolve(strict=True)
        cls.contract = consistency.load_contract(config_path)
        cls.manifest = consistency.load_manifest(MANIFEST_PATH, cls.contract)

    def test_manifest_has_six_classes_and_five_hashed_images_each(self):
        self.assertEqual(len(self.manifest["classes"]), 6)
        self.assertEqual(len(self.manifest["resolved_samples"]), 30)
        self.assertEqual(self.manifest["class_counts"], {index: 5 for index in range(6)})
        self.assertEqual(
            self.manifest["requirements"], consistency.FROZEN_REQUIREMENTS
        )
        for sample in self.manifest["resolved_samples"]:
            self.assertTrue(sample["resolved_image_path"].is_file())
            self.assertEqual(
                consistency.sha256_file(sample["resolved_image_path"]),
                sample["image_sha256"],
            )

    def test_manifest_rejects_a_post_run_tolerance_relaxation(self):
        relaxed = dict(consistency.FROZEN_REQUIREMENTS)
        relaxed["confidence_abs_error_max"] = 0.1
        with self.assertRaisesRegex(
            consistency.ConsistencyError,
            "restore the predeclared S1-07 thresholds",
        ):
            consistency.validate_frozen_requirements(relaxed)


class DeterministicMatchingTest(unittest.TestCase):
    def test_class_then_max_iou_matching_ignores_array_order(self):
        python_detections = [
            detection(0, 0.90, [0.0, 0.0, 10.0, 10.0]),
            detection(0, 0.80, [30.0, 30.0, 40.0, 40.0]),
            detection(1, 0.70, [60.0, 60.0, 70.0, 70.0]),
        ]
        cpp_detections = [
            detection(1, 0.700001, [60.0001, 60.0, 70.0, 70.0]),
            detection(0, 0.800001, [30.0, 30.0001, 40.0, 40.0]),
            detection(0, 0.900001, [0.0, 0.0, 10.0001, 10.0]),
        ]

        first = consistency.match_detections(
            python_detections, cpp_detections
        )
        second = consistency.match_detections(
            list(reversed(python_detections)), list(reversed(cpp_detections))
        )

        self.assertEqual(canonical_pairs(first), canonical_pairs(second))
        self.assertEqual(first["unmatched_python_indices"], [])
        self.assertEqual(first["unmatched_cpp_indices"], [])
        self.assertTrue(
            all(item["matching_iou"] >= 0.999 for item in first["matches"])
        )

    def test_class_mismatch_is_not_hidden_by_high_iou(self):
        python_detections = [detection(0, 0.9, [0.0, 0.0, 10.0, 10.0])]
        cpp_detections = [detection(1, 0.9, [0.0, 0.0, 10.0, 10.0])]

        matching = consistency.match_detections(
            python_detections, cpp_detections
        )

        self.assertEqual(matching["matches"], [])
        self.assertEqual(matching["unmatched_python_indices"], [0])
        self.assertEqual(matching["unmatched_cpp_indices"], [0])


class PythonReferencePostprocessTest(unittest.TestCase):
    def test_strict_threshold_and_stable_class_agnostic_nms(self):
        contract = {
            "output_shape": [1, 5, 3],
            "class_names": ["defect"],
            "score_threshold": 0.25,
            "nms_threshold": 0.45,
        }
        output = np.zeros(contract["output_shape"], dtype=np.float32)
        output[0, 0, :] = np.array([10.0, 50.0, 50.0], dtype=np.float32)
        output[0, 1, :] = np.array([10.0, 50.0, 50.0], dtype=np.float32)
        output[0, 2, :] = np.array([4.0, 20.0, 20.0], dtype=np.float32)
        output[0, 3, :] = np.array([4.0, 20.0, 20.0], dtype=np.float32)
        output[0, 4, :] = np.array([0.25, 0.9, 0.8], dtype=np.float32)
        transform = {
            "original_width": 100,
            "original_height": 100,
            "input_width": 100,
            "input_height": 100,
            "scale": 1.0,
            "pad_left": 0,
            "pad_top": 0,
        }

        detections = consistency.postprocess_raw_output(
            output, contract, transform
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["confidence"], 0.9, places=6)
        self.assertEqual(detections[0]["bbox_xyxy"], [40.0, 40.0, 60.0, 60.0])


if __name__ == "__main__":
    unittest.main()

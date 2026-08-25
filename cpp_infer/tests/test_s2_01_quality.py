#!/usr/bin/env python3
"""Pure tests for the frozen S2-01 data and quality protocol."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CPP_INFER_ROOT.parent
GENERATOR_PATH = CPP_INFER_ROOT / "tools" / "generate_s2_01_manifests.py"
EVALUATOR_PATH = CPP_INFER_ROOT / "tools" / "evaluate_s2_01_correctness.py"
CALIBRATION_PATH = (
    CPP_INFER_ROOT / "tests" / "fixtures" / "s2_01_calibration_manifest.json"
)
QUALITY_PATH = CPP_INFER_ROOT / "tests" / "fixtures" / "s2_01_quality_manifest.json"


def import_tool(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = import_tool("s2_01_manifest_generator", GENERATOR_PATH)
evaluator = import_tool("s2_01_quality_evaluator", EVALUATOR_PATH)


def detection(class_id, confidence, box):
    return {
        "class_id": class_id,
        "class_name": generator.CLASS_NAMES[class_id],
        "confidence": confidence,
        "bbox_xyxy": list(box),
    }


def one_class_metric(ap50, ap50_95, class_id=0):
    return {
        "class_id": class_id,
        "class_name": generator.CLASS_NAMES[class_id],
        "ap50": ap50,
        "ap50_95": ap50_95,
    }


class FrozenManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = evaluator.load_frozen_manifest(
            CALIBRATION_PATH, "calibration"
        )
        cls.quality = evaluator.load_frozen_manifest(QUALITY_PATH, "quality")

    def test_calibration_is_exactly_six_by_thirty_strided_train_images(self):
        samples = self.calibration["resolved_samples"]
        self.assertEqual(len(samples), 180)
        self.assertEqual(
            self.calibration["integrity"]["sample_set_sha256"],
            "FDEF7FB3B64E222386387438C0B4A32A6BDECF9761E5ED5C60E9A17B7311AE5F",
        )
        expected_indices = list(generator.CALIBRATION_INDICES)
        for class_id, class_name in enumerate(generator.CLASS_NAMES):
            actual_indices = [
                int(sample["sample_id"][len(class_name) + 1 :])
                for sample in samples
                if sample["source_class_id"] == class_id
            ]
            self.assertEqual(actual_indices, expected_indices)
        self.assertEqual(len({sample["image_sha256"] for sample in samples}), 180)

    def test_quality_is_the_full_current_361_image_validation_set(self):
        samples = self.quality["resolved_samples"]
        self.assertEqual(len(samples), 361)
        self.assertEqual(
            self.quality["integrity"]["sample_set_sha256"],
            "F90692D9898C6F92D94BD4CE3B2AD4DF996A864A0AF7FC0DAFCE97B33C780E33",
        )
        self.assertEqual(
            self.quality["dataset"]["source_class_counts"],
            {"0": 61, "1": 60, "2": 60, "3": 60, "4": 60, "5": 60},
        )
        self.assertEqual(self.quality["dataset"]["ground_truth_box_count"], 857)
        self.assertEqual(
            self.quality["dataset"]["ground_truth_class_counts"],
            {"0": 165, "1": 159, "2": 193, "3": 87, "4": 132, "5": 121},
        )
        self.assertIn("crazing_240", {sample["sample_id"] for sample in samples})
        self.assertEqual(len({sample["image_sha256"] for sample in samples}), 361)

    def test_calibration_and_quality_have_no_image_hash_overlap(self):
        calibration_hashes = {
            sample["image_sha256"] for sample in self.calibration["resolved_samples"]
        }
        quality_hashes = {
            sample["image_sha256"] for sample in self.quality["resolved_samples"]
        }
        self.assertFalse(calibration_hashes & quality_hashes)

    def test_filename_source_class_is_not_used_as_ground_truth(self):
        sample = next(
            item
            for item in self.quality["resolved_samples"]
            if item["sample_id"] == "patches_245"
        )
        ground_truth_classes = {
            box["class_id"] for box in sample["normalized_ground_truth"]
        }
        self.assertEqual(sample["source_class_id"], 2)
        self.assertIn(1, ground_truth_classes)

    def test_generator_recreates_the_committed_documents_semantically(self):
        expected_calibration = generator.build_calibration_manifest(
            REPO_ROOT, CALIBRATION_PATH
        )
        expected_quality = generator.build_quality_manifest(REPO_ROOT, QUALITY_PATH)
        self.assertEqual(generator.load_json(CALIBRATION_PATH), expected_calibration)
        self.assertEqual(generator.load_json(QUALITY_PATH), expected_quality)


class HashAndLabelTest(unittest.TestCase):
    def test_label_hash_normalizes_line_endings_only(self):
        content = "0 0.5 0.5 0.2 0.2\n1 0.2 0.3 0.1 0.1\n"
        normalized = {
            generator.normalize_utf8_lf_bytes(value.encode("utf-8"))
            for value in (
                content,
                content.replace("\n", "\r\n"),
                content.replace("\n", "\r"),
            )
        }
        self.assertEqual(normalized, {content.encode("utf-8")})

    def test_quality_label_parser_rejects_duplicate_boxes(self):
        with self.assertRaises(generator.ManifestError):
            generator.parse_yolo_label_text(
                "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
                "duplicate-test",
            )

    def test_quality_label_parser_rejects_out_of_bounds_boxes(self):
        with self.assertRaises(generator.ManifestError):
            generator.parse_yolo_label_text(
                "0 0.95 0.5 0.2 0.2\n", "out-of-bounds-test"
            )


class AveragePrecisionTest(unittest.TestCase):
    def test_perfect_predictions_have_one_ap_at_every_iou(self):
        ground_truth = {
            "a": [{"class_id": 0, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}],
            "b": [{"class_id": 0, "bbox_xyxy": [5.0, 5.0, 15.0, 15.0]}],
        }
        predictions = [
            {**detection(0, 0.9, [0.0, 0.0, 10.0, 10.0]), "sample_id": "a"},
            {**detection(0, 0.8, [5.0, 5.0, 15.0, 15.0]), "sample_id": "b"},
        ]
        for threshold in generator.IOU_THRESHOLDS:
            ap, gt_count, prediction_count = evaluator.ap_101_point(
                predictions, ground_truth, 0, threshold
            )
            self.assertAlmostEqual(ap, 1.0)
            self.assertEqual(gt_count, 2)
            self.assertEqual(prediction_count, 2)

    def test_one_of_two_ground_truth_boxes_reaches_only_half_recall(self):
        ground_truth = {
            "a": [{"class_id": 0, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}],
            "b": [{"class_id": 0, "bbox_xyxy": [5.0, 5.0, 15.0, 15.0]}],
        }
        predictions = [
            {**detection(0, 0.9, [0.0, 0.0, 10.0, 10.0]), "sample_id": "a"}
        ]
        ap, _, _ = evaluator.ap_101_point(predictions, ground_truth, 0, 0.5)
        self.assertAlmostEqual(ap, 51.0 / 101.0)

    def test_product_operating_point_uses_strict_score_threshold(self):
        ground_truth = {
            "a": [{"class_id": 0, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
        }
        predictions = [
            {
                **detection(0, generator.PRODUCT_SCORE_THRESHOLD, [0, 0, 10, 10]),
                "sample_id": "a",
            }
        ]
        metrics = evaluator.fixed_operating_point_metrics(
            predictions, ground_truth, generator.PRODUCT_SCORE_THRESHOLD
        )
        self.assertEqual(metrics["true_positive"], 0)
        self.assertEqual(metrics["false_negative"], 1)


class ProductDifferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.consistency = evaluator.load_consistency_tool()

    def test_identical_reordered_detections_pass_predeclared_gates(self):
        first = detection(0, 0.9, [0.0, 0.0, 10.0, 10.0])
        second = detection(1, 0.8, [20.0, 20.0, 30.0, 30.0])
        result = evaluator.product_difference(
            {"image": [first, second]},
            {"image": [second, first]},
            self.consistency.match_detections,
            generator.PRODUCT_MATCHING_GATES,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["matched_detections_total"], 2)
        self.assertAlmostEqual(result["metrics"]["fp32_retention"], 1.0)

    def test_class_change_is_reported_as_unmatched_and_fails(self):
        fp32 = detection(0, 0.9, [0.0, 0.0, 10.0, 10.0])
        int8 = detection(1, 0.9, [0.0, 0.0, 10.0, 10.0])
        result = evaluator.product_difference(
            {"image": [fp32]},
            {"image": [int8]},
            self.consistency.match_detections,
            generator.PRODUCT_MATCHING_GATES,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["matched_detections_total"], 0)
        self.assertEqual(result["metrics"]["fp32_retention"], 0.0)


class QualityGateTest(unittest.TestCase):
    def make_metrics(self, map50, map50_95, ap50_values):
        return {
            "map50": map50,
            "map50_95": map50_95,
            "per_class": [
                one_class_metric(ap50, map50_95, class_id)
                for class_id, ap50 in enumerate(ap50_values)
            ],
        }

    def test_exact_declared_drop_boundaries_pass(self):
        fp32 = self.make_metrics(0.90, 0.70, [0.90] * 6)
        int8 = self.make_metrics(0.89, 0.68, [0.85] * 6)
        result = evaluator.quality_comparison(
            fp32, int8, generator.QUALITY_GATES
        )
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_any_per_class_ap50_drop_beyond_five_points_fails(self):
        fp32 = self.make_metrics(0.90, 0.70, [0.90] * 6)
        int8 = self.make_metrics(0.895, 0.69, [0.849, 0.90, 0.90, 0.90, 0.90, 0.90])
        result = evaluator.quality_comparison(
            fp32, int8, generator.QUALITY_GATES
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["per_class_ap50_drop"])


if __name__ == "__main__":
    unittest.main()

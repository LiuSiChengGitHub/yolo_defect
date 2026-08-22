"""Validate the frozen S1-05 single-image detection JSON contract."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, NoReturn, Optional, Set


EXPECTED_MODEL_ID = "yolov8n_neu_det_final_train_2"
EXPECTED_SHA256 = (
    "7B8A37610018A6AE6CACDFC869590A95"
    "BBE31AFB7579C39BE0FFEC537196AF68"
)
EXPECTED_PROVIDER = "CPUExecutionProvider"
EXPECTED_PROVIDER_EVIDENCE = (
    "explicit_cpu_ep_registration_and_session_creation"
)
EXPECTED_SCORE_THRESHOLD = 0.25
EXPECTED_NMS_THRESHOLD = 0.45
EXPECTED_NMS_MODE = "class_agnostic"
EXPECTED_CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


def fail(message: str) -> NoReturn:
    raise AssertionError(message)


def expect_exact_keys(value: Any, expected: Set[str], object_name: str) -> None:
    if not isinstance(value, dict):
        fail(
            f"{object_name}: expected JSON object, actual "
            f"{type(value).__name__}"
        )
    actual = set(value)
    if actual != expected:
        fail(
            f"{object_name}: expected keys {sorted(expected)}, actual "
            f"{sorted(actual)}; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def expect_int(value: Any, object_name: str) -> int:
    if type(value) is not int:
        fail(f"{object_name}: expected integer, actual {value!r}")
    return value


def expect_string(value: Any, object_name: str) -> str:
    if not isinstance(value, str):
        fail(f"{object_name}: expected string, actual {value!r}")
    return value


def expect_finite_number(value: Any, object_name: str) -> float:
    if type(value) not in (int, float):
        fail(f"{object_name}: expected JSON number, actual {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        fail(f"{object_name}: expected finite number, actual {value!r}")
    return converted


def expect_close(value: Any, expected: float, object_name: str) -> float:
    converted = expect_finite_number(value, object_name)
    if not math.isclose(converted, expected, rel_tol=0.0, abs_tol=1.0e-12):
        fail(f"{object_name}: expected {expected}, actual {converted}")
    return converted


def normalized_path(value: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(value)))


def validate_document(document: Any, expected_image: Optional[str]) -> int:
    expect_exact_keys(
        document,
        {"schema_version", "model", "image", "runtime", "detections"},
        "root",
    )

    if expect_int(document["schema_version"], "schema_version") != 1:
        fail(
            "schema_version: expected 1, actual "
            f"{document['schema_version']!r}"
        )

    model = document["model"]
    expect_exact_keys(model, {"model_id", "declared_sha256"}, "model")
    if expect_string(model["model_id"], "model.model_id") != EXPECTED_MODEL_ID:
        fail(
            f"model.model_id: expected {EXPECTED_MODEL_ID!r}, actual "
            f"{model['model_id']!r}"
        )
    declared_sha256 = expect_string(
        model["declared_sha256"], "model.declared_sha256"
    )
    if declared_sha256 != EXPECTED_SHA256:
        fail(
            f"model.declared_sha256: expected {EXPECTED_SHA256!r}, actual "
            f"{declared_sha256!r}"
        )

    image = document["image"]
    expect_exact_keys(
        image, {"path", "original_size", "input_size"}, "image"
    )
    image_path = expect_string(image["path"], "image.path")
    if not image_path:
        fail("image.path: expected non-empty path, actual empty string")
    if expected_image is not None:
        if normalized_path(image_path) != normalized_path(expected_image):
            fail(
                f"image.path: expected {expected_image!r}, actual "
                f"{image_path!r}"
            )

    original_size = image["original_size"]
    expect_exact_keys(
        original_size, {"width", "height", "channels"},
        "image.original_size"
    )
    if expect_int(original_size["width"], "image.original_size.width") != 200:
        fail(
            "image.original_size.width: expected 200, actual "
            f"{original_size['width']!r}"
        )
    if expect_int(original_size["height"], "image.original_size.height") != 200:
        fail(
            "image.original_size.height: expected 200, actual "
            f"{original_size['height']!r}"
        )
    if expect_int(
        original_size["channels"], "image.original_size.channels"
    ) != 3:
        fail(
            "image.original_size.channels: expected 3, actual "
            f"{original_size['channels']!r}"
        )

    input_size = image["input_size"]
    expect_exact_keys(input_size, {"width", "height"}, "image.input_size")
    if expect_int(input_size["width"], "image.input_size.width") != 800:
        fail(
            "image.input_size.width: expected 800, actual "
            f"{input_size['width']!r}"
        )
    if expect_int(input_size["height"], "image.input_size.height") != 800:
        fail(
            "image.input_size.height: expected 800, actual "
            f"{input_size['height']!r}"
        )

    runtime = document["runtime"]
    expect_exact_keys(
        runtime,
        {
            "actual_provider",
            "provider_evidence",
            "score_threshold",
            "nms_threshold",
            "nms_mode",
        },
        "runtime",
    )
    actual_provider = expect_string(
        runtime["actual_provider"], "runtime.actual_provider"
    )
    if actual_provider != EXPECTED_PROVIDER:
        fail(
            f"runtime.actual_provider: expected {EXPECTED_PROVIDER!r}, "
            f"actual {actual_provider!r}"
        )
    provider_evidence = expect_string(
        runtime["provider_evidence"], "runtime.provider_evidence"
    )
    if provider_evidence != EXPECTED_PROVIDER_EVIDENCE:
        fail(
            "runtime.provider_evidence: expected "
            f"{EXPECTED_PROVIDER_EVIDENCE!r}, actual {provider_evidence!r}"
        )
    score_threshold = expect_close(
        runtime["score_threshold"],
        EXPECTED_SCORE_THRESHOLD,
        "runtime.score_threshold",
    )
    expect_close(
        runtime["nms_threshold"],
        EXPECTED_NMS_THRESHOLD,
        "runtime.nms_threshold",
    )
    nms_mode = expect_string(runtime["nms_mode"], "runtime.nms_mode")
    if nms_mode != EXPECTED_NMS_MODE:
        fail(
            f"runtime.nms_mode: expected {EXPECTED_NMS_MODE!r}, actual "
            f"{nms_mode!r}"
        )

    detections = document["detections"]
    if not isinstance(detections, list):
        fail(
            "detections: expected array (including [] for no detections), "
            f"actual {type(detections).__name__}"
        )

    previous_confidence = math.inf
    for index, detection in enumerate(detections):
        object_name = f"detections[{index}]"
        expect_exact_keys(
            detection,
            {"class_id", "class_name", "confidence", "bbox_xyxy"},
            object_name,
        )
        class_id = expect_int(detection["class_id"], f"{object_name}.class_id")
        if not 0 <= class_id < len(EXPECTED_CLASSES):
            fail(
                f"{object_name}.class_id: expected [0,{len(EXPECTED_CLASSES) - 1}], "
                f"actual {class_id}"
            )
        class_name = expect_string(
            detection["class_name"], f"{object_name}.class_name"
        )
        if class_name != EXPECTED_CLASSES[class_id]:
            fail(
                f"{object_name}.class_name: expected "
                f"{EXPECTED_CLASSES[class_id]!r} for class_id {class_id}, "
                f"actual {class_name!r}"
            )

        confidence = expect_finite_number(
            detection["confidence"], f"{object_name}.confidence"
        )
        if not confidence > score_threshold:
            fail(
                f"{object_name}.confidence: expected > {score_threshold}, "
                f"actual {confidence}"
            )
        if confidence > 1.0:
            fail(
                f"{object_name}.confidence: expected <= 1, actual {confidence}"
            )
        if confidence > previous_confidence:
            fail(
                f"{object_name}.confidence: expected detections in "
                f"non-increasing confidence order; previous "
                f"{previous_confidence}, actual {confidence}"
            )
        previous_confidence = confidence

        bbox = detection["bbox_xyxy"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            fail(
                f"{object_name}.bbox_xyxy: expected four-number array, "
                f"actual {bbox!r}"
            )
        x1, y1, x2, y2 = (
            expect_finite_number(value, f"{object_name}.bbox_xyxy[{axis}]")
            for axis, value in enumerate(bbox)
        )
        if not 0.0 <= x1 <= x2 <= 200.0:
            fail(
                f"{object_name}.bbox_xyxy: expected 0 <= x1 <= x2 <= 200, "
                f"actual {bbox!r}"
            )
        if not 0.0 <= y1 <= y2 <= 200.0:
            fail(
                f"{object_name}.bbox_xyxy: expected 0 <= y1 <= y2 <= 200, "
                f"actual {bbox!r}"
            )

    return len(detections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen S1-05 detection JSON schema."
    )
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-image")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.json_path.open("r", encoding="utf-8") as source:
            document = json.load(source)
        detection_count = validate_document(document, args.expected_image)
    except (OSError, UnicodeError, json.JSONDecodeError, AssertionError) as error:
        print(f"S1-05 detection JSON validation failed: {error}")
        return 1

    print(
        "S1-05 detection JSON validation passed: "
        f"path={args.json_path}, detections={detection_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

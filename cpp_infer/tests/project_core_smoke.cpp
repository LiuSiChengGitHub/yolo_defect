#include "yolo_defect_cpp/project_core.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool nearly_equal(float lhs, float rhs) {
  return std::abs(lhs - rhs) <= 1.0e-6F;
}

int fail(const std::string& message) {
  std::cerr << "project-core portability smoke failed: " << message << "\n";
  return 1;
}

}  // namespace

int main() {
  try {
    yolo_defect_cpp::InferenceOutput output;
    output.shape = {1, 6, 2};
    output.values = {
        5.0F, 6.0F,
        5.0F, 5.0F,
        10.0F, 10.0F,
        10.0F, 10.0F,
        0.9F, 0.1F,
        0.1F, 0.8F};

    const std::vector<yolo_defect_cpp::Detection> decoded =
        yolo_defect_cpp::decode_yolov8_raw_output(
            output, {"defect_a", "defect_b"}, 0.25);
    if (decoded.size() != 2U || decoded[0].class_id != 0 ||
        decoded[1].class_id != 1) {
      return fail("decode did not keep the two cross-class candidates");
    }

    const std::vector<yolo_defect_cpp::Detection> kept =
        yolo_defect_cpp::class_agnostic_nms(decoded, 0.45);
    if (kept.size() != 1U || kept.front().class_id != 0 ||
        !nearly_equal(kept.front().confidence, 0.9F)) {
      return fail(
          "class-agnostic NMS did not suppress the overlapping other class");
    }

    yolo_defect_cpp::PreprocessResult preprocess;
    preprocess.original_width = 20;
    preprocess.original_height = 10;
    preprocess.original_channels = 3;
    preprocess.input_width = 20;
    preprocess.input_height = 20;
    preprocess.resized_width = 20;
    preprocess.resized_height = 10;
    preprocess.pad_top = 5;
    preprocess.pad_bottom = 5;
    preprocess.scale = 1.0;

    const std::vector<yolo_defect_cpp::Detection> restored =
        yolo_defect_cpp::restore_detections_to_original(kept, preprocess);
    if (restored.size() != 1U) {
      return fail("coordinate restore changed the kept detection count");
    }
    const yolo_defect_cpp::BoundingBox& box = restored.front().bbox_xyxy;
    if (!nearly_equal(box.x1, 0.0F) || !nearly_equal(box.y1, 0.0F) ||
        !nearly_equal(box.x2, 10.0F) || !nearly_equal(box.y2, 5.0F)) {
      return fail("letterbox coordinate restore or clipping changed semantics");
    }

    std::cout << "project-core portability smoke passed: "
                 "decode -> class-agnostic NMS -> coordinate restore\n";
    return 0;
  } catch (const std::exception& error) {
    return fail(std::string("unexpected exception: ") + error.what());
  }
}

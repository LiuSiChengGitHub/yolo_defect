#ifndef YOLO_DEFECT_CPP_POSTPROCESSOR_H_
#define YOLO_DEFECT_CPP_POSTPROCESSOR_H_

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/image_preprocessor.h"
#include "yolo_defect_cpp/onnx_runner.h"

#include <cstddef>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct BoundingBox {
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
};

struct Detection {
  int class_id = -1;
  std::string class_name;
  float confidence = 0.0F;
  BoundingBox bbox_xyxy;
};

void validate_yolov8_raw_output(const InferenceOutput& output,
                                std::size_t class_count);

BoundingBox xywh_to_xyxy(float center_x, float center_y,
                         float width, float height);

float intersection_over_union(const BoundingBox& lhs,
                              const BoundingBox& rhs);

std::vector<Detection> decode_yolov8_raw_output(
    const InferenceOutput& output,
    const std::vector<std::string>& class_names,
    double score_threshold);

std::vector<Detection> class_agnostic_nms(
    const std::vector<Detection>& candidates,
    double nms_threshold);

BoundingBox restore_letterbox_box(
    const BoundingBox& model_input_box,
    const PreprocessResult& preprocess);

std::vector<Detection> restore_detections_to_original(
    const std::vector<Detection>& detections,
    const PreprocessResult& preprocess);

std::vector<Detection> postprocess_yolov8_raw(
    const InferenceOutput& output,
    const RuntimeContract& contract,
    const PreprocessResult& preprocess);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_POSTPROCESSOR_H_

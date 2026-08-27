#ifndef YOLO_DEFECT_CPP_PROJECT_CORE_H_
#define YOLO_DEFECT_CPP_PROJECT_CORE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct InferenceOutput {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

struct PreprocessResult {
  int original_width = 0;
  int original_height = 0;
  int original_channels = 0;
  int input_width = 0;
  int input_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_left = 0;
  int pad_top = 0;
  int pad_right = 0;
  int pad_bottom = 0;
  double scale = 0.0;
  std::vector<float> tensor_nchw;
};

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

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_PROJECT_CORE_H_

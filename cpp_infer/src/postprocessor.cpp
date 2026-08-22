#include "yolo_defect_cpp/postprocessor.h"

#include "yolo_defect_cpp/model_metadata.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

[[noreturn]] void throw_postprocess_error(
    const std::string& object,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  std::ostringstream message;
  message << "YOLOv8 postprocess validation failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

std::string format_float(float value) {
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

void validate_threshold(double threshold, const std::string& object) {
  if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
    std::ostringstream actual;
    actual << threshold;
    throw_postprocess_error(
        object, "a finite value in [0,1]", actual.str(),
        "fix the RuntimeConfig threshold before calling postprocess");
  }
}

void validate_box(const BoundingBox& box, const std::string& object) {
  const float coordinates[] = {box.x1, box.y1, box.x2, box.y2};
  const char* names[] = {"x1", "y1", "x2", "y2"};
  for (std::size_t index = 0; index < 4; ++index) {
    if (!std::isfinite(coordinates[index])) {
      throw_postprocess_error(
          object + "." + names[index], "a finite coordinate",
          format_float(coordinates[index]),
          "inspect raw model values and box conversion before IoU/NMS");
    }
  }
}

std::size_t checked_output_element_count(
    const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    const std::int64_t dimension = shape[index];
    if (dimension < 0) {
      throw_postprocess_error(
          "output.shape[" + std::to_string(index) + "]",
          "a non-negative static dimension", std::to_string(dimension),
          "export a static YOLOv8 output or provide a resolved runtime shape");
    }
    const auto unsigned_dimension = static_cast<std::uint64_t>(dimension);
    if (unsigned_dimension >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      throw_postprocess_error(
          "output.shape[" + std::to_string(index) + "]",
          "a dimension representable by size_t", std::to_string(dimension),
          "inspect corrupt or unsupported output dimensions");
    }
    const std::size_t size_dimension =
        static_cast<std::size_t>(unsigned_dimension);
    if (size_dimension != 0 &&
        count > std::numeric_limits<std::size_t>::max() / size_dimension) {
      throw_postprocess_error(
          "output.shape", "an element count representable by size_t",
          format_shape(shape),
          "inspect corrupt dimensions before indexing the raw output");
    }
    count *= size_dimension;
  }
  return count;
}

void validate_class_names(const std::vector<std::string>& class_names) {
  if (class_names.empty()) {
    throw_postprocess_error(
        "class_names", "at least one non-empty class", "[]",
        "fix the ModelArtifactSpec class_names declaration");
  }
  for (std::size_t index = 0; index < class_names.size(); ++index) {
    if (class_names[index].empty()) {
      throw_postprocess_error(
          "class_names[" + std::to_string(index) + "]",
          "a non-empty class name", "empty",
          "fix the ModelArtifactSpec class_names declaration");
    }
  }
}

void validate_preprocess_transform(const PreprocessResult& preprocess) {
  if (preprocess.original_width <= 0 || preprocess.original_height <= 0) {
    throw_postprocess_error(
        "preprocess.original_size", "positive width and height",
        std::to_string(preprocess.original_width) + "x" +
            std::to_string(preprocess.original_height),
        "use the PreprocessResult produced for the source image");
  }
  if (preprocess.input_width <= 0 || preprocess.input_height <= 0 ||
      preprocess.resized_width <= 0 || preprocess.resized_height <= 0) {
    throw_postprocess_error(
        "preprocess.letterbox_size",
        "positive input and resized dimensions",
        "input=" + std::to_string(preprocess.input_width) + "x" +
            std::to_string(preprocess.input_height) + ", resized=" +
            std::to_string(preprocess.resized_width) + "x" +
            std::to_string(preprocess.resized_height),
        "pass the exact PreprocessResult used to create the model input");
  }
  if (!std::isfinite(preprocess.scale) || preprocess.scale <= 0.0) {
    std::ostringstream actual;
    actual << preprocess.scale;
    throw_postprocess_error(
        "preprocess.scale", "a finite value greater than zero",
        actual.str(),
        "pass the exact letterbox scale returned by preprocess");
  }
  if (preprocess.pad_left < 0 || preprocess.pad_top < 0 ||
      preprocess.pad_right < 0 || preprocess.pad_bottom < 0) {
    throw_postprocess_error(
        "preprocess.padding", "non-negative padding", "negative padding",
        "inspect letterbox padding construction");
  }
  const std::int64_t horizontal_extent =
      static_cast<std::int64_t>(preprocess.pad_left) +
      static_cast<std::int64_t>(preprocess.resized_width) +
      static_cast<std::int64_t>(preprocess.pad_right);
  if (horizontal_extent != preprocess.input_width) {
    throw_postprocess_error(
        "preprocess.horizontal_geometry",
        "pad_left + resized_width + pad_right == input_width",
        std::to_string(preprocess.pad_left) + " + " +
            std::to_string(preprocess.resized_width) + " + " +
            std::to_string(preprocess.pad_right) + " != " +
            std::to_string(preprocess.input_width),
        "use matching letterbox metadata from the same preprocess call");
  }
  const std::int64_t vertical_extent =
      static_cast<std::int64_t>(preprocess.pad_top) +
      static_cast<std::int64_t>(preprocess.resized_height) +
      static_cast<std::int64_t>(preprocess.pad_bottom);
  if (vertical_extent != preprocess.input_height) {
    throw_postprocess_error(
        "preprocess.vertical_geometry",
        "pad_top + resized_height + pad_bottom == input_height",
        std::to_string(preprocess.pad_top) + " + " +
            std::to_string(preprocess.resized_height) + " + " +
            std::to_string(preprocess.pad_bottom) + " != " +
            std::to_string(preprocess.input_height),
        "use matching letterbox metadata from the same preprocess call");
  }
}

void validate_detection(const Detection& detection, std::size_t index) {
  if (!std::isfinite(detection.confidence)) {
    throw_postprocess_error(
        "detections[" + std::to_string(index) + "].confidence",
        "a finite confidence", format_float(detection.confidence),
        "inspect decode output before NMS");
  }
  validate_box(detection.bbox_xyxy,
               "detections[" + std::to_string(index) + "].bbox_xyxy");
}

void validate_postprocess_contract(const RuntimeContract& contract) {
  if (contract.artifact.model_family != ModelFamily::kYoloV8) {
    throw_postprocess_error(
        "artifact.model_family", "yolov8",
        to_string(contract.artifact.model_family),
        "dispatch this artifact to its model-family-specific postprocessor");
  }
  if (contract.artifact.output.dtype != TensorDataType::kFloat32 ||
      contract.artifact.output.layout != TensorLayout::kBcn) {
    throw_postprocess_error(
        "artifact.output_contract", "float32 BCN",
        to_string(contract.artifact.output.dtype) + " " +
            to_string(contract.artifact.output.layout),
        "fix the artifact declaration or select a compatible postprocessor");
  }
  if (contract.artifact.postprocess_type != PostprocessType::kYoloV8Raw) {
    throw_postprocess_error(
        "artifact.postprocess_type", "yolov8_raw",
        to_string(contract.artifact.postprocess_type),
        "dispatch to the postprocessor declared by the artifact");
  }
  if (contract.artifact.nms_mode != NmsMode::kClassAgnostic) {
    throw_postprocess_error(
        "artifact.nms_mode", "class_agnostic",
        to_string(contract.artifact.nms_mode),
        "use the baseline class-agnostic NMS or update contract and tests "
        "together");
  }
  validate_class_names(contract.artifact.class_names);
}

}  // namespace

void validate_yolov8_raw_output(const InferenceOutput& output,
                                std::size_t class_count) {
  if (class_count == 0) {
    throw_postprocess_error(
        "class_count", "at least one class", "0",
        "provide the class_names from ModelArtifactSpec");
  }
  if (output.shape.size() != 3) {
    throw_postprocess_error(
        "output.rank", "3 for [1,4+C,N]",
        std::to_string(output.shape.size()),
        "inspect the selected ONNX output and BCN layout declaration");
  }
  if (output.shape[0] != 1) {
    throw_postprocess_error(
        "output.batch", "1", std::to_string(output.shape[0]),
        "use the P0 batch=1 output or add an explicit batch adapter");
  }
  const std::size_t expected_channels = 4 + class_count;
  if (output.shape[1] != static_cast<std::int64_t>(expected_channels)) {
    throw_postprocess_error(
        "output.channels", "4 + class_count = " +
            std::to_string(expected_channels),
        std::to_string(output.shape[1]),
        "verify the artifact class_names and YOLOv8 output selection");
  }
  if (output.shape[2] < 0) {
    throw_postprocess_error(
        "output.candidates", "a non-negative candidate count",
        std::to_string(output.shape[2]),
        "use the resolved static output shape from ONNX Runtime");
  }

  const std::size_t expected_elements =
      checked_output_element_count(output.shape);
  if (output.values.size() != expected_elements) {
    throw_postprocess_error(
        "output.elements", std::to_string(expected_elements),
        std::to_string(output.values.size()),
        "verify the owned InferenceOutput copy and its shape");
  }
  for (std::size_t index = 0; index < output.values.size(); ++index) {
    if (!std::isfinite(output.values[index])) {
      throw_postprocess_error(
          "output.values[" + std::to_string(index) + "]",
          "a finite float32 value", format_float(output.values[index]),
          "inspect preprocess, model execution, and raw output ownership");
    }
  }
}

BoundingBox xywh_to_xyxy(float center_x, float center_y,
                         float width, float height) {
  const BoundingBox box{
      center_x - width / 2.0F,
      center_y - height / 2.0F,
      center_x + width / 2.0F,
      center_y + height / 2.0F};
  validate_box(box, "xywh_to_xyxy.result");
  return box;
}

float intersection_over_union(const BoundingBox& lhs,
                              const BoundingBox& rhs) {
  validate_box(lhs, "iou.lhs");
  validate_box(rhs, "iou.rhs");

  const double intersection_width = std::max(
      0.0, static_cast<double>(std::min(lhs.x2, rhs.x2)) -
               static_cast<double>(std::max(lhs.x1, rhs.x1)));
  const double intersection_height = std::max(
      0.0, static_cast<double>(std::min(lhs.y2, rhs.y2)) -
               static_cast<double>(std::max(lhs.y1, rhs.y1)));
  const double intersection = intersection_width * intersection_height;
  const double lhs_area =
      std::max(0.0, static_cast<double>(lhs.x2) - lhs.x1) *
      std::max(0.0, static_cast<double>(lhs.y2) - lhs.y1);
  const double rhs_area =
      std::max(0.0, static_cast<double>(rhs.x2) - rhs.x1) *
      std::max(0.0, static_cast<double>(rhs.y2) - rhs.y1);
  const double union_area = lhs_area + rhs_area - intersection;
  if (union_area <= 0.0F) {
    return 0.0F;
  }
  const double iou = intersection / union_area;
  if (!std::isfinite(iou)) {
    throw_postprocess_error(
        "iou.result", "a finite value in [0,1]", "non-finite",
        "inspect extreme or corrupt box coordinates before NMS");
  }
  return static_cast<float>(std::clamp(iou, 0.0, 1.0));
}

std::vector<Detection> decode_yolov8_raw_output(
    const InferenceOutput& output,
    const std::vector<std::string>& class_names,
    double score_threshold) {
  validate_class_names(class_names);
  validate_threshold(score_threshold, "runtime.score_threshold");
  validate_yolov8_raw_output(output, class_names.size());
  const float effective_score_threshold =
      static_cast<float>(score_threshold);

  const std::size_t candidate_count =
      static_cast<std::size_t>(output.shape[2]);
  std::vector<Detection> detections;
  detections.reserve(candidate_count);

  const auto value_at = [&output, candidate_count](
                            std::size_t channel,
                            std::size_t candidate) -> float {
    return output.values[channel * candidate_count + candidate];
  };

  for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
    std::size_t best_class = 0;
    float best_score = value_at(4, candidate);
    for (std::size_t class_index = 1;
         class_index < class_names.size(); ++class_index) {
      const float score = value_at(4 + class_index, candidate);
      if (score > best_score) {
        best_score = score;
        best_class = class_index;
      }
    }

    if (!(best_score > effective_score_threshold)) {
      continue;
    }

    Detection detection;
    detection.class_id = static_cast<int>(best_class);
    detection.class_name = class_names[best_class];
    detection.confidence = best_score;
    detection.bbox_xyxy = xywh_to_xyxy(
        value_at(0, candidate), value_at(1, candidate),
        value_at(2, candidate), value_at(3, candidate));
    detections.push_back(std::move(detection));
  }
  return detections;
}

std::vector<Detection> class_agnostic_nms(
    const std::vector<Detection>& candidates,
    double nms_threshold) {
  validate_threshold(nms_threshold, "runtime.nms_threshold");
  const float effective_nms_threshold =
      static_cast<float>(nms_threshold);
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    validate_detection(candidates[index], index);
  }
  if (candidates.empty()) {
    return {};
  }

  std::vector<std::size_t> order(candidates.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(),
      [&candidates](std::size_t lhs, std::size_t rhs) {
        return candidates[lhs].confidence > candidates[rhs].confidence;
      });

  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<Detection> kept;
  kept.reserve(candidates.size());
  for (std::size_t order_index = 0;
       order_index < order.size(); ++order_index) {
    const std::size_t current = order[order_index];
    if (suppressed[current]) {
      continue;
    }
    kept.push_back(candidates[current]);
    for (std::size_t remaining = order_index + 1;
         remaining < order.size(); ++remaining) {
      const std::size_t other = order[remaining];
      if (!suppressed[other] &&
          intersection_over_union(candidates[current].bbox_xyxy,
                                  candidates[other].bbox_xyxy) >
              effective_nms_threshold) {
        suppressed[other] = true;
      }
    }
  }
  return kept;
}

BoundingBox restore_letterbox_box(
    const BoundingBox& model_input_box,
    const PreprocessResult& preprocess) {
  validate_box(model_input_box, "restore.model_input_box");
  validate_preprocess_transform(preprocess);

  const auto restore_x = [&preprocess](float coordinate) {
    const double restored =
        (static_cast<double>(coordinate) - preprocess.pad_left) /
        preprocess.scale;
    return static_cast<float>(std::clamp(
        restored, 0.0, static_cast<double>(preprocess.original_width)));
  };
  const auto restore_y = [&preprocess](float coordinate) {
    const double restored =
        (static_cast<double>(coordinate) - preprocess.pad_top) /
        preprocess.scale;
    return static_cast<float>(std::clamp(
        restored, 0.0, static_cast<double>(preprocess.original_height)));
  };

  return BoundingBox{
      restore_x(model_input_box.x1),
      restore_y(model_input_box.y1),
      restore_x(model_input_box.x2),
      restore_y(model_input_box.y2)};
}

std::vector<Detection> restore_detections_to_original(
    const std::vector<Detection>& detections,
    const PreprocessResult& preprocess) {
  if (detections.empty()) {
    return {};
  }
  validate_preprocess_transform(preprocess);

  std::vector<Detection> restored = detections;
  for (std::size_t index = 0; index < restored.size(); ++index) {
    validate_detection(restored[index], index);
    restored[index].bbox_xyxy =
        restore_letterbox_box(restored[index].bbox_xyxy, preprocess);
  }
  return restored;
}

std::vector<Detection> postprocess_yolov8_raw(
    const InferenceOutput& output,
    const RuntimeContract& contract,
    const PreprocessResult& preprocess) {
  validate_postprocess_contract(contract);
  if (contract.artifact.input.shape.size() != 4) {
    throw_postprocess_error(
        "artifact.input.shape", "rank 4 NCHW",
        format_shape(contract.artifact.input.shape),
        "fix the artifact input tensor declaration");
  }
  if (preprocess.input_height != contract.artifact.input.shape[2] ||
      preprocess.input_width != contract.artifact.input.shape[3]) {
    throw_postprocess_error(
        "preprocess.input_size",
        std::to_string(contract.artifact.input.shape[3]) + "x" +
            std::to_string(contract.artifact.input.shape[2]),
        std::to_string(preprocess.input_width) + "x" +
            std::to_string(preprocess.input_height),
        "use the PreprocessResult created for the same artifact input");
  }
  if (output.shape != contract.artifact.output.shape) {
    throw_postprocess_error(
        "output.shape", format_shape(contract.artifact.output.shape),
        format_shape(output.shape),
        "use the InferenceOutput returned for the declared artifact");
  }

  const std::vector<Detection> decoded = decode_yolov8_raw_output(
      output, contract.artifact.class_names,
      contract.runtime.score_threshold);
  const std::vector<Detection> kept = class_agnostic_nms(
      decoded, contract.runtime.nms_threshold);
  return restore_detections_to_original(kept, preprocess);
}

}  // namespace yolo_defect_cpp

#include "yolo_defect_cpp/postprocessor.h"

#include "yolo_defect_cpp/model_metadata.h"

#include <sstream>
#include <stdexcept>
#include <string>
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

void validate_contract_class_names(
    const std::vector<std::string>& class_names) {
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
  validate_contract_class_names(contract.artifact.class_names);
}

}  // namespace

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

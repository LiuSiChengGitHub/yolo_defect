#include "yolo_defect_cpp/detector_pipeline.h"

#include "yolo_defect_cpp/image_preprocessor.h"
#include "yolo_defect_cpp/onnx_runner.h"
#include "yolo_defect_cpp/postprocessor.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

[[noreturn]] void throw_pipeline_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Single-image pipeline validation failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

std::filesystem::path normalize_source_image_path(
    const std::filesystem::path& declared_path) {
  if (declared_path.empty()) {
    throw_pipeline_error(
        "image_path", "a non-empty path to one readable image file",
        "an empty path",
        "pass --image <path> for a single JPEG, PNG, or other OpenCV-"
        "decodable image");
  }

  std::error_code error;
  std::filesystem::path absolute_path =
      std::filesystem::absolute(declared_path, error);
  if (error) {
    throw_pipeline_error(
        "image_path", "a path resolvable from the current working directory",
        "'" + declared_path.string() + "' (" + error.message() + ")",
        "correct the image path or run the CLI from the intended working "
        "directory");
  }
  absolute_path = absolute_path.lexically_normal();

  const bool exists = std::filesystem::exists(absolute_path, error);
  if (error) {
    throw_pipeline_error(
        "image_path", "an accessible existing image file",
        "'" + absolute_path.string() + "' (" + error.message() + ")",
        "check the path and filesystem permissions");
  }
  if (!exists) {
    throw_pipeline_error(
        "image_path", "an existing image file",
        "missing path '" + absolute_path.string() + "'",
        "correct --image or restore the input file");
  }

  const bool is_regular =
      std::filesystem::is_regular_file(absolute_path, error);
  if (error) {
    throw_pipeline_error(
        "image_path", "an accessible regular image file",
        "'" + absolute_path.string() + "' (" + error.message() + ")",
        "check the path and filesystem permissions");
  }
  if (!is_regular) {
    throw_pipeline_error(
        "image_path", "a regular image file",
        "non-file path '" + absolute_path.string() + "'",
        "pass one image file rather than a directory or special file");
  }

  std::filesystem::path canonical_path =
      std::filesystem::canonical(absolute_path, error);
  if (error) {
    throw_pipeline_error(
        "image_path", "an image path that can be canonicalized",
        "'" + absolute_path.string() + "' (" + error.message() + ")",
        "check the source path, link target, and filesystem permissions");
  }
  return canonical_path;
}

SingleImageDetectionResult make_detection_result(
    const RuntimeContract& contract,
    const std::filesystem::path& source_image_path,
    const PreprocessResult& preprocess,
    const ModelMetadata& metadata,
    std::vector<Detection> detections) {
  SingleImageDetectionResult result;
  result.schema_version = 1;
  result.model_id = contract.artifact.model_id;
  result.declared_model_sha256 = contract.artifact.model_sha256;
  result.class_names = contract.artifact.class_names;
  result.image.source_path = source_image_path;
  result.image.original_width = preprocess.original_width;
  result.image.original_height = preprocess.original_height;
  result.image.original_channels = preprocess.original_channels;
  result.image.input_width = preprocess.input_width;
  result.image.input_height = preprocess.input_height;
  result.actual_provider = metadata.session_provider;
  result.provider_evidence = metadata.provider_evidence;
  result.score_threshold = contract.runtime.score_threshold;
  result.nms_threshold = contract.runtime.nms_threshold;
  result.nms_mode = contract.artifact.nms_mode;
  result.detections = std::move(detections);
  return result;
}

std::vector<std::filesystem::path> protected_pipeline_paths(
    const RuntimeContract& contract,
    const std::filesystem::path& source_image_path) {
  std::vector<std::filesystem::path> paths = {
      source_image_path,
      contract.runtime.declaration_path,
      contract.artifact.declaration_path,
      contract.artifact.model_path,
  };
  if (contract.runtime.tensorrt.has_value()) {
    const TensorRtProviderConfig& tensorrt = *contract.runtime.tensorrt;
    paths.push_back(tensorrt.engine_cache_path);
    if (tensorrt.native_engine_path.has_value()) {
      paths.push_back(*tensorrt.native_engine_path);
    }
  }
  return paths;
}

}  // namespace

class DetectorPipeline::Impl {
 public:
  explicit Impl(RuntimeContract contract)
      : contract_(std::move(contract)), runner_(contract_) {}

  SingleImagePipelineResult run(
      const std::filesystem::path& image_path,
      const DetectionOutputRequest& output_request) {
    if (!output_request.json_path.has_value() &&
        !output_request.image_path.has_value()) {
      throw_pipeline_error(
          "output_request", "at least one JSON or visualization output",
          "neither output path was requested",
          "pass --output-json <path>, --output-image <path>, or both");
    }

    const std::filesystem::path source_image_path =
        normalize_source_image_path(image_path);

    PreprocessResult preprocess = preprocess_image(
        source_image_path, contract_.artifact);
    InferenceOutput raw_output = runner_.run(
        contract_.artifact.input.shape, preprocess.tensor_nchw);
    std::vector<Detection> detections = postprocess_yolov8_raw(
        raw_output, contract_, preprocess);

    SingleImagePipelineResult result;
    result.detection_result = make_detection_result(
        contract_, source_image_path, preprocess, runner_.metadata(),
        std::move(detections));
    result.outputs = write_detection_outputs(
        result.detection_result, output_request,
        protected_pipeline_paths(contract_, source_image_path));
    return result;
  }

  const ModelMetadata& metadata() const noexcept {
    return runner_.metadata();
  }

  double session_initialization_ms() const noexcept {
    return runner_.session_initialization_ms();
  }

 private:
  RuntimeContract contract_;
  OnnxRunner runner_;
};

DetectorPipeline::DetectorPipeline(RuntimeContract contract)
    : impl_(std::make_unique<Impl>(std::move(contract))) {}

DetectorPipeline::~DetectorPipeline() = default;

DetectorPipeline::DetectorPipeline(DetectorPipeline&&) noexcept = default;

DetectorPipeline& DetectorPipeline::operator=(
    DetectorPipeline&&) noexcept = default;

SingleImagePipelineResult DetectorPipeline::run(
    const std::filesystem::path& image_path,
    const DetectionOutputRequest& output_request) {
  if (!impl_) {
    throw_pipeline_error(
        "DetectorPipeline", "a live pipeline instance",
        "a moved-from pipeline",
        "invoke run only on the pipeline that owns the Runtime session");
  }
  return impl_->run(image_path, output_request);
}

const ModelMetadata& DetectorPipeline::metadata() const {
  if (!impl_) {
    throw_pipeline_error(
        "DetectorPipeline", "a live pipeline instance",
        "a moved-from pipeline",
        "inspect metadata only on the pipeline that owns the Runtime "
        "session");
  }
  return impl_->metadata();
}

double DetectorPipeline::session_initialization_ms() const {
  if (!impl_) {
    throw_pipeline_error(
        "DetectorPipeline", "a live pipeline instance",
        "a moved-from pipeline",
        "inspect initialization timing only on the pipeline that owns the "
        "Runtime session");
  }
  return impl_->session_initialization_ms();
}

}  // namespace yolo_defect_cpp

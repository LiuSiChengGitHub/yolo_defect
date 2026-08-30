#include "yolo_defect_cpp/profile_runner.h"

#include "yolo_defect_cpp/image_preprocessor.h"
#include "yolo_defect_cpp/onnx_runner.h"
#include "yolo_defect_cpp/postprocessor.h"

#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

constexpr std::size_t kMaximumProfileRuns = 1000000;

[[noreturn]] void throw_profile_runner_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "ORT profile execution failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

std::filesystem::path prepare_profile_prefix(
    const std::filesystem::path& declared_prefix) {
  if (declared_prefix.empty()) {
    throw_profile_runner_error(
        "profile_file_prefix", "a non-empty file prefix", "empty",
        "pass --profile-prefix <path>");
  }

  std::error_code error;
  std::filesystem::path prefix =
      std::filesystem::absolute(declared_prefix, error);
  if (error) {
    throw_profile_runner_error(
        "profile_file_prefix",
        "a path resolvable from the current working directory",
        declared_prefix.string() + " (" + error.message() + ")",
        "correct the profile prefix path");
  }
  prefix = prefix.lexically_normal();
  if (prefix.filename().empty()) {
    throw_profile_runner_error(
        "profile_file_prefix", "a file prefix, not a directory",
        prefix.string(), "append a filename prefix to the output directory");
  }

  const std::filesystem::path parent = prefix.parent_path();
  const bool parent_exists = std::filesystem::exists(parent, error);
  if (error) {
    throw_profile_runner_error(
        "profile_file_prefix.parent", "an inspectable directory",
        parent.string() + " (" + error.message() + ")",
        "check parent-directory permissions");
  }
  if (parent_exists) {
    const bool is_directory =
        std::filesystem::is_directory(parent, error);
    if (error || !is_directory) {
      throw_profile_runner_error(
          "profile_file_prefix.parent", "a directory",
          error ? parent.string() + " (" + error.message() + ")"
                : "non-directory path '" + parent.string() + "'",
          "choose a writable output directory");
    }
  } else if (!std::filesystem::create_directories(parent, error) && error) {
    throw_profile_runner_error(
        "profile_file_prefix.parent", "a creatable directory",
        parent.string() + " (" + error.message() + ")",
        "choose a writable output location");
  }
  return prefix;
}

}  // namespace

class ProfileRunner::Impl {
 public:
  explicit Impl(RuntimeContract contract) : contract_(std::move(contract)) {}

  ProfileResult run(const ProfileRequest& request) {
    if (request.run_count == 0 ||
        request.run_count > kMaximumProfileRuns) {
      throw_profile_runner_error(
          "profile_runs", "an integer in [1,1000000]",
          std::to_string(request.run_count),
          "use the frozen --profile-runs 10 count for formal profiling");
    }
    const std::filesystem::path profile_prefix =
        prepare_profile_prefix(request.profile_file_prefix);
    PreprocessResult preprocess = preprocess_image(
        request.image_path, contract_.artifact);

    OnnxRunnerOptions runner_options;
    runner_options.profile_file_prefix = profile_prefix;
    OnnxRunner runner(contract_, std::move(runner_options));
    if (!runner.profiling_enabled()) {
      throw_profile_runner_error(
          "profiling.state", "enabled", "disabled",
          "pass the validated prefix into OnnxRunnerOptions");
    }

    InferenceOutput output;
    for (std::size_t index = 0; index < request.run_count; ++index) {
      output = runner.run(contract_.artifact.input.shape,
                          preprocess.tensor_nchw);
    }
    const std::filesystem::path trace_path = runner.end_profiling();

    std::vector<Detection> detections = postprocess_yolov8_raw(
        output, contract_, preprocess);

    ProfileResult result;
    result.trace_path = trace_path;
    result.run_count = request.run_count;
    result.model_id = contract_.artifact.model_id;
    result.declared_model_sha256 = contract_.artifact.model_sha256;
    result.actual_provider = runner.metadata().session_provider;
    result.session_initialization_ms =
        runner.session_initialization_ms();
    result.output_shape = std::move(output.shape);
    result.output_element_count = output.values.size();
    result.detection_count = detections.size();
    if (!std::isfinite(result.session_initialization_ms) ||
        result.session_initialization_ms < 0.0) {
      throw_profile_runner_error(
          "session_initialization_ms", "a finite non-negative duration",
          std::to_string(result.session_initialization_ms),
          "verify OnnxRunner steady-clock timing");
    }
    return result;
  }

 private:
  RuntimeContract contract_;
};

ProfileRunner::ProfileRunner(RuntimeContract contract)
    : impl_(std::make_unique<Impl>(std::move(contract))) {}

ProfileRunner::~ProfileRunner() = default;
ProfileRunner::ProfileRunner(ProfileRunner&&) noexcept = default;
ProfileRunner& ProfileRunner::operator=(ProfileRunner&&) noexcept = default;

ProfileResult ProfileRunner::run(const ProfileRequest& request) {
  if (!impl_) {
    throw_profile_runner_error(
        "ProfileRunner", "a live runner instance", "moved-from instance",
        "invoke run only on the object that owns the Runtime contract");
  }
  return impl_->run(request);
}

}  // namespace yolo_defect_cpp

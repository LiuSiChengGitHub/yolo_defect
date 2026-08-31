#ifndef YOLO_DEFECT_CPP_ONNX_RUNNER_H_
#define YOLO_DEFECT_CPP_ONNX_RUNNER_H_

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/model_metadata.h"
#include "yolo_defect_cpp/project_core.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace yolo_defect_cpp {

struct TimedInferenceOutput {
  InferenceOutput output;
  // Measures one backend invocation. For ORT this is Session::Run; for the
  // load-only native TensorRT path it includes H2D, enqueueV3, D2H, and stream
  // synchronization. Input/output validation remains outside the interval.
  double session_run_ms = 0.0;
};

struct OnnxRunnerOptions {
  // ORT treats this as a file prefix and chooses the final trace filename.
  // The actual filename is returned by OnnxRunner::end_profiling().
  std::optional<std::filesystem::path> profile_file_prefix;
};

class OnnxRunner {
 public:
  explicit OnnxRunner(const RuntimeContract& contract,
                      OnnxRunnerOptions options = {});
  ~OnnxRunner();

  OnnxRunner(const OnnxRunner&) = delete;
  OnnxRunner& operator=(const OnnxRunner&) = delete;
  OnnxRunner(OnnxRunner&&) noexcept;
  OnnxRunner& operator=(OnnxRunner&&) noexcept;

  const ModelMetadata& metadata() const noexcept;
  double session_initialization_ms() const noexcept;
  bool profiling_enabled() const noexcept;
  std::filesystem::path end_profiling();
  InferenceOutput run(const std::vector<std::int64_t>& input_shape,
                      std::vector<float>& input_values);
  TimedInferenceOutput run_with_session_timing(
      const std::vector<std::int64_t>& input_shape,
      std::vector<float>& input_values);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_ONNX_RUNNER_H_

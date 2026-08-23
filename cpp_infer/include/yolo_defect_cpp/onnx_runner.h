#ifndef YOLO_DEFECT_CPP_ONNX_RUNNER_H_
#define YOLO_DEFECT_CPP_ONNX_RUNNER_H_

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/model_metadata.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace yolo_defect_cpp {

struct InferenceOutput {
  std::vector<std::int64_t> shape;
  std::vector<float> values;
};

struct TimedInferenceOutput {
  InferenceOutput output;
  // Measures only Ort::Session::Run. Input validation/tensor construction and
  // output validation/copy are deliberately outside this interval.
  double session_run_ms = 0.0;
};

class OnnxRunner {
 public:
  explicit OnnxRunner(const RuntimeContract& contract);
  ~OnnxRunner();

  OnnxRunner(const OnnxRunner&) = delete;
  OnnxRunner& operator=(const OnnxRunner&) = delete;
  OnnxRunner(OnnxRunner&&) noexcept;
  OnnxRunner& operator=(OnnxRunner&&) noexcept;

  const ModelMetadata& metadata() const noexcept;
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

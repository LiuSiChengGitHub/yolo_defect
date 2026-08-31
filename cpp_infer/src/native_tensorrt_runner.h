#ifndef YOLO_DEFECT_CPP_NATIVE_TENSORRT_RUNNER_H_
#define YOLO_DEFECT_CPP_NATIVE_TENSORRT_RUNNER_H_

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/model_metadata.h"
#include "yolo_defect_cpp/project_core.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace yolo_defect_cpp {

struct NativeTimedInferenceOutput {
  InferenceOutput output;
  // Includes native H2D, enqueueV3, D2H, and stream synchronization.
  double backend_run_ms = 0.0;
};

class NativeTensorRtRunner {
 public:
  explicit NativeTensorRtRunner(const RuntimeContract& contract);
  ~NativeTensorRtRunner();

  NativeTensorRtRunner(const NativeTensorRtRunner&) = delete;
  NativeTensorRtRunner& operator=(const NativeTensorRtRunner&) = delete;

  const ModelMetadata& metadata() const noexcept;
  double initialization_ms() const noexcept;
  NativeTimedInferenceOutput run_with_timing(
      const std::vector<std::int64_t>& input_shape,
      const std::vector<float>& input_values);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_NATIVE_TENSORRT_RUNNER_H_

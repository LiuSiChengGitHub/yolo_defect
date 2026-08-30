#ifndef YOLO_DEFECT_CPP_BATCH_EXECUTOR_H_
#define YOLO_DEFECT_CPP_BATCH_EXECUTOR_H_

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/model_metadata.h"
#include "yolo_defect_cpp/result_writer.h"

#include <cstddef>
#include <filesystem>
#include <memory>

namespace yolo_defect_cpp {
namespace internal {

struct BatchExecutionResult {
  std::size_t detection_count = 0;
  WrittenDetectionOutputs outputs;
};

class BatchTaskExecutor {
 public:
  virtual ~BatchTaskExecutor() = default;

  virtual const ModelMetadata& metadata() const = 0;
  virtual double session_initialization_ms() const = 0;
  virtual BatchExecutionResult run(
      const std::filesystem::path& source_path,
      const DetectionOutputRequest& output_request) = 0;
};

class BatchExecutorFactory {
 public:
  virtual ~BatchExecutorFactory() = default;

  virtual std::unique_ptr<BatchTaskExecutor> create(
      const RuntimeContract& contract,
      std::size_t worker_index) = 0;
};

}  // namespace internal
}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BATCH_EXECUTOR_H_

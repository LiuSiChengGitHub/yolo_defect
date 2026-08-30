#ifndef YOLO_DEFECT_CPP_BATCH_RUNNER_H_
#define YOLO_DEFECT_CPP_BATCH_RUNNER_H_

#include "yolo_defect_cpp/batch_result.h"
#include "yolo_defect_cpp/config_loader.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

namespace internal {
class BatchExecutorFactory;
}  // namespace internal

struct BatchRequest {
  BatchInputKind input_kind = BatchInputKind::kDirectory;
  std::filesystem::path input_path;
  std::filesystem::path output_directory;
  std::filesystem::path summary_path;
  std::size_t requested_workers = 1;
  std::size_t queue_capacity = 2;
  bool output_images = false;
  bool overwrite_existing = false;
  std::vector<std::string> command_arguments;
};

// Performs all discovery and validation before returning any tasks.
std::vector<BatchTask> discover_batch_tasks(
    BatchInputKind input_kind,
    const std::filesystem::path& input_path);

class BatchRunner {
 public:
  explicit BatchRunner(RuntimeContract contract);
  // Test seam for lifecycle/concurrency contracts. Production callers use
  // the one-argument constructor, which always creates DetectorPipeline-
  // backed executors. The internal interface lives in src/batch_executor.h.
  BatchRunner(
      RuntimeContract contract,
      std::shared_ptr<internal::BatchExecutorFactory> executor_factory);
  ~BatchRunner();

  BatchRunner(const BatchRunner&) = delete;
  BatchRunner& operator=(const BatchRunner&) = delete;
  BatchRunner(BatchRunner&&) noexcept;
  BatchRunner& operator=(BatchRunner&&) noexcept;

  BatchSummary run(const BatchRequest& request);

  // Cooperative and thread-safe. Queued tasks are cancelled, blocked queue
  // operations are awakened, and already-running DetectorPipeline::run calls
  // are allowed to finish. run() performs the joins before it returns.
  void request_stop() noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BATCH_RUNNER_H_

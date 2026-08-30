#ifndef YOLO_DEFECT_CPP_BATCH_RESULT_H_
#define YOLO_DEFECT_CPP_BATCH_RESULT_H_

#include "yolo_defect_cpp/benchmark_result.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

enum class BatchInputKind {
  kDirectory,
  kManifest,
};

enum class BatchItemStatus {
  kSucceeded,
  kFailed,
  kCancelled,
};

enum class BatchStatus {
  kSucceeded,
  kPartialFailure,
  kCancelled,
  kFatal,
};

struct BatchTask {
  std::size_t sequence_index = 0;
  std::filesystem::path source_path;
  // Directory-relative or manifest-declared path used to make discovery
  // order inspectable. Inference always uses source_path.
  std::filesystem::path logical_path;
};

struct BatchItemResult {
  std::size_t sequence_index = 0;
  BatchItemStatus status = BatchItemStatus::kCancelled;
  std::filesystem::path source_path;
  std::optional<std::filesystem::path> json_output_path;
  std::optional<std::filesystem::path> image_output_path;
  std::size_t detection_count = 0;
  double latency_ms = 0.0;
  std::string error;
};

struct BatchEnvironment {
  std::string hostname;
  std::string processor;
  std::size_t logical_cpu_count = 0;
  std::string os_name;
  std::string os_version;
  std::string target_architecture;
  std::string runtime_kernel_architecture;
  std::string execution_context;
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  int cxx_standard = 17;
  std::string opencv_version;
  std::string onnxruntime_version;
};

struct BatchRuntimeMetadata {
  std::filesystem::path config_path;
  std::string requested_provider;
  std::string actual_provider;
  std::string provider_evidence;
  std::string execution_mode;
  int intra_op_num_threads = 0;
  int inter_op_num_threads = 0;
  std::string graph_optimization_level;
  double score_threshold = 0.0;
  double nms_threshold = 0.0;
  std::string nms_mode;
  std::size_t requested_workers = 0;
  std::size_t effective_workers = 0;
  std::size_t session_count = 0;
  std::vector<double> session_initialization_ms;
};

struct BatchModelMetadata {
  std::string model_id;
  std::string model_family;
  std::filesystem::path model_path;
  std::string declared_sha256;
  int opset = 0;
  std::string input_name;
  std::vector<std::int64_t> input_shape;
  std::string input_dtype;
  std::string input_layout;
};

struct BatchInputMetadata {
  BatchInputKind kind = BatchInputKind::kDirectory;
  std::filesystem::path source_path;
  std::string ordering;
};

struct BatchOutputMetadata {
  std::filesystem::path directory;
  std::filesystem::path batch_summary_path;
  std::filesystem::path item_directory;
  bool json_outputs = true;
  bool image_outputs = false;
  bool overwrite_existing = false;
};

struct BatchCounts {
  std::size_t discovered = 0;
  std::size_t enqueued = 0;
  std::size_t started = 0;
  std::size_t succeeded = 0;
  std::size_t failed = 0;
  std::size_t cancelled = 0;
};

struct BatchQueueStatistics {
  std::size_t capacity = 0;
  std::size_t peak_depth = 0;
  std::size_t producer_wait_count = 0;
  double producer_wait_ms = 0.0;
};

struct BatchTiming {
  double processing_wall_ms = 0.0;
  std::vector<std::string> includes;
  std::vector<std::string> excludes;
};

struct BatchMemoryEvidence {
  bool supported = false;
  std::string status;
  std::string metric;
  std::uint64_t bytes = 0;
  double mebibytes = 0.0;
  std::string scope;
  std::string reason;
  bool publishable = true;
};

struct BatchSummary {
  int schema_version = 1;
  std::string evidence_type = "cpp_ort_multi_image_batch_summary";
  std::string timestamp_utc;
  BatchStatus status = BatchStatus::kFatal;
  // True when the public cooperative-stop API was invoked. This remains
  // observable even if every task had already started and therefore no item
  // can remain in kCancelled state.
  bool cooperative_stop_requested = false;
  std::vector<std::string> command_arguments;
  BatchEnvironment environment;
  BatchRuntimeMetadata runtime;
  BatchModelMetadata model;
  BatchInputMetadata input;
  BatchOutputMetadata output;
  BatchCounts counts;
  BatchQueueStatistics queue;
  BatchTiming timing;
  // Successful-item latency only. Failed-item attempt durations remain in
  // items and cancelled items have zero latency.
  LatencyStatistics latency_ms;
  double throughput_images_per_second = 0.0;
  BatchMemoryEvidence memory;
  std::vector<BatchItemResult> items;
  std::vector<std::string> limitations;
  std::string fatal_error;
};

std::string to_string(BatchInputKind value);
std::string to_string(BatchItemStatus value);
std::string to_string(BatchStatus value);

void validate_batch_summary(const BatchSummary& summary);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BATCH_RESULT_H_

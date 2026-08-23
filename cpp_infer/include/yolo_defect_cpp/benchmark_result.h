#ifndef YOLO_DEFECT_CPP_BENCHMARK_RESULT_H_
#define YOLO_DEFECT_CPP_BENCHMARK_RESULT_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct LatencyStatistics {
  std::size_t sample_count = 0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
};

struct BenchmarkLatencyResults {
  LatencyStatistics image_decode;
  LatencyStatistics preprocess;
  LatencyStatistics session_run;
  LatencyStatistics postprocess;
  LatencyStatistics pipeline;
  LatencyStatistics end_to_end;
  double pipeline_throughput_images_per_second = 0.0;
  double end_to_end_throughput_images_per_second = 0.0;
};

struct BenchmarkMemoryEvidence {
  bool supported = false;
  std::string status;
  std::string metric;
  std::uint64_t bytes = 0;
  double mebibytes = 0.0;
  std::string scope;
  std::string reason;
};

struct BenchmarkEnvironment {
  std::string hostname;
  std::string processor;
  std::string architecture;
  std::size_t logical_cpu_count = 0;
  std::string os_name;
  std::string os_version;
  std::string compiler_id;
  std::string compiler_version;
  std::string build_type;
  int cxx_standard = 17;
  std::string opencv_version;
  std::string onnxruntime_version;
};

struct BenchmarkRuntimeMetadata {
  std::string requested_provider;
  std::string actual_provider;
  std::string provider_evidence;
  std::string execution_mode;
  int intra_op_num_threads = 0;
  int inter_op_num_threads = 0;
  std::string graph_optimization_level;
};

struct BenchmarkModelMetadata {
  std::string model_id;
  std::string model_family;
  std::string model_path;
  std::string declared_sha256;
  std::uint64_t file_size_bytes = 0;
  int opset = 0;
  std::string input_name;
  std::vector<std::int64_t> input_shape;
  std::string input_dtype;
  std::string input_layout;
};

struct BenchmarkSampleMetadata {
  std::string image_path;
  std::uint64_t file_size_bytes = 0;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::size_t sample_count = 1;
  std::size_t detection_count = 0;
};

struct BenchmarkResult {
  int schema_version = 1;
  std::string evidence_type;
  std::string timestamp_utc;
  std::vector<std::string> command_arguments;

  std::size_t batch_size = 1;
  std::size_t sample_count = 1;
  std::size_t warmup = 0;
  std::size_t repeat = 0;

  BenchmarkEnvironment environment;
  BenchmarkRuntimeMetadata runtime;
  BenchmarkModelMetadata model;
  BenchmarkSampleMetadata sample;

  double score_threshold = 0.0;
  double nms_threshold = 0.0;
  std::string nms_mode;

  BenchmarkLatencyResults latency;
  BenchmarkMemoryEvidence memory;
  std::vector<std::string> timing_exclusions;
  std::vector<std::string> limitations;
};

LatencyStatistics calculate_latency_statistics(
    const std::vector<double>& samples_ms);

double calculate_throughput_images_per_second(
    const LatencyStatistics& latency);

void validate_benchmark_result(const BenchmarkResult& result);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BENCHMARK_RESULT_H_

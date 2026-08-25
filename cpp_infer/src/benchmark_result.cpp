#include "yolo_defect_cpp/benchmark_result.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo_defect_cpp {
namespace {

[[noreturn]] void throw_benchmark_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Benchmark validation failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

void validate_non_empty(const std::string& value,
                        const std::string& object) {
  if (value.empty()) {
    throw_benchmark_error(
        object, "a non-empty evidence value", "empty",
        "populate the benchmark metadata before serialization");
  }
}

bool is_sha256(const std::string& value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isxdigit(character) != 0;
         });
}

void validate_finite_non_negative(double value,
                                  const std::string& object) {
  if (!std::isfinite(value) || value < 0.0) {
    throw_benchmark_error(
        object, "a finite non-negative number", std::to_string(value),
        "inspect the steady-clock samples and statistics calculation");
  }
}

void validate_statistics(const LatencyStatistics& statistics,
                         std::size_t expected_samples,
                         const std::string& object) {
  if (statistics.sample_count != expected_samples) {
    throw_benchmark_error(
        object + ".sample_count", std::to_string(expected_samples),
        std::to_string(statistics.sample_count),
        "retain exactly one timing value per measured repeat");
  }
  validate_finite_non_negative(statistics.mean_ms, object + ".mean_ms");
  validate_finite_non_negative(statistics.p50_ms, object + ".p50_ms");
  validate_finite_non_negative(statistics.p95_ms, object + ".p95_ms");
  if (statistics.p50_ms > statistics.p95_ms) {
    throw_benchmark_error(
        object + ".percentiles", "p50_ms <= p95_ms",
        "p50_ms=" + std::to_string(statistics.p50_ms) +
            ", p95_ms=" + std::to_string(statistics.p95_ms),
        "sort the complete sample vector before nearest-rank lookup");
  }
}

}  // namespace

LatencyStatistics calculate_latency_statistics(
    const std::vector<double>& samples_ms) {
  if (samples_ms.empty()) {
    throw_benchmark_error(
        "latency_samples", "at least one measured duration", "empty",
        "set --repeat to a positive integer and collect every iteration");
  }

  std::vector<double> sorted = samples_ms;
  long double sum = 0.0L;
  for (std::size_t index = 0; index < sorted.size(); ++index) {
    const double value = sorted[index];
    if (!std::isfinite(value) || value < 0.0) {
      throw_benchmark_error(
          "latency_samples[" + std::to_string(index) + "]",
          "a finite non-negative duration in milliseconds",
          std::to_string(value),
          "discard failed iterations instead of publishing invalid timing "
          "data");
    }
    sum += static_cast<long double>(value);
  }
  std::sort(sorted.begin(), sorted.end());

  const auto nearest_rank = [&sorted](double percentile) {
    const double rank =
        std::ceil(percentile * static_cast<double>(sorted.size()));
    const std::size_t index = static_cast<std::size_t>(
        std::max(1.0, rank) - 1.0);
    return sorted[index];
  };

  LatencyStatistics statistics;
  statistics.sample_count = sorted.size();
  statistics.mean_ms = static_cast<double>(
      sum / static_cast<long double>(sorted.size()));
  statistics.p50_ms = nearest_rank(0.50);
  statistics.p95_ms = nearest_rank(0.95);
  return statistics;
}

double calculate_throughput_images_per_second(
    const LatencyStatistics& latency) {
  if (!std::isfinite(latency.mean_ms) || latency.mean_ms <= 0.0) {
    throw_benchmark_error(
        "latency.mean_ms", "a finite positive mean latency",
        std::to_string(latency.mean_ms),
        "collect a real measured pipeline before computing throughput");
  }
  const double throughput = 1000.0 / latency.mean_ms;
  if (!std::isfinite(throughput) || throughput <= 0.0) {
    throw_benchmark_error(
        "throughput_images_per_second", "a finite positive value",
        std::to_string(throughput),
        "inspect the mean latency and floating-point environment");
  }
  return throughput;
}

void validate_benchmark_result(const BenchmarkResult& result) {
  if (result.schema_version != 1) {
    throw_benchmark_error(
        "schema_version", "1", std::to_string(result.schema_version),
        "use the supported benchmark evidence schema");
  }
  if (result.evidence_type != "cpp_ort_single_image_release_benchmark") {
    throw_benchmark_error(
        "evidence_type", "cpp_ort_single_image_release_benchmark",
        result.evidence_type,
        "do not mix historical Python and current C++ benchmark schemas");
  }
  validate_non_empty(result.timestamp_utc, "timestamp_utc");
  if (result.command_arguments.empty()) {
    throw_benchmark_error(
        "command", "at least the executable argument", "empty",
        "pass the original CLI argv into BenchmarkRequest");
  }
  for (std::size_t index = 0; index < result.command_arguments.size();
       ++index) {
    validate_non_empty(result.command_arguments[index],
                       "command[" + std::to_string(index) + "]");
  }
  if (result.batch_size != 1 || result.sample_count != 1 ||
      result.sample.sample_count != 1) {
    throw_benchmark_error(
        "protocol.batch_and_sample_count", "batch=1 and sample_count=1",
        "batch=" + std::to_string(result.batch_size) +
            ", samples=" + std::to_string(result.sample_count),
        "keep S1-08 on the fixed single-image baseline protocol");
  }
  if (result.repeat == 0) {
    throw_benchmark_error(
        "protocol.repeat", "a positive integer", "0",
        "pass --repeat 100 for the formal S1-08 evidence run");
  }

  const BenchmarkEnvironment& environment = result.environment;
  validate_non_empty(environment.hostname, "environment.machine.hostname");
  validate_non_empty(environment.processor, "environment.machine.processor");
  validate_non_empty(environment.architecture,
                     "environment.machine.architecture");
  if (environment.logical_cpu_count == 0) {
    throw_benchmark_error(
        "environment.machine.logical_cpu_count", "a positive count", "0",
        "record the host logical processor count");
  }
  validate_non_empty(environment.os_name, "environment.os.name");
  validate_non_empty(environment.os_version, "environment.os.version");
  validate_non_empty(environment.compiler_id, "environment.compiler.id");
  validate_non_empty(environment.compiler_version,
                     "environment.compiler.version");
  if (environment.compiler_id == "unknown" ||
      environment.compiler_version == "unknown") {
    throw_benchmark_error(
        "environment.compiler", "configured compiler id and version",
        environment.compiler_id + " " + environment.compiler_version,
        "inject CMake compiler metadata into the Release Runtime target");
  }
  if (environment.build_type != "Release") {
    throw_benchmark_error(
        "environment.build.type", "Release", environment.build_type,
        "perform a clean configure with -DCMAKE_BUILD_TYPE=Release");
  }
  if (environment.cxx_standard != 17) {
    throw_benchmark_error(
        "environment.build.cxx_standard", "17",
        std::to_string(environment.cxx_standard),
        "build the documented C++17 Runtime target");
  }
  validate_non_empty(environment.opencv_version,
                     "environment.opencv_version");
  validate_non_empty(environment.onnxruntime_version,
                     "environment.onnxruntime_version");
  if (environment.opencv_version != "4.8.0" ||
      environment.onnxruntime_version != "1.19.2") {
    throw_benchmark_error(
        "environment.runtime_versions", "OpenCV 4.8.0 and ORT 1.19.2",
        "OpenCV=" + environment.opencv_version +
            ", ORT=" + environment.onnxruntime_version,
        "use the pinned S1-08 OpenCV and ONNX Runtime dependencies");
  }

  const BenchmarkRuntimeMetadata& runtime = result.runtime;
  if (runtime.requested_provider != "cpu" ||
      runtime.actual_provider != "CPUExecutionProvider") {
    throw_benchmark_error(
        "runtime.provider", "requested cpu and actual CPUExecutionProvider",
        "requested=" + runtime.requested_provider +
            ", actual=" + runtime.actual_provider,
        "use the fixed CPU RuntimeConfig and verify the created ORT session");
  }
  validate_non_empty(runtime.provider_evidence,
                     "runtime.provider_evidence");
  if (runtime.execution_mode != "sequential" ||
      runtime.intra_op_num_threads != 1 ||
      runtime.inter_op_num_threads != 1 ||
      runtime.graph_optimization_level != "all") {
    throw_benchmark_error(
        "runtime.thread_policy",
        "sequential, intra_op=1, inter_op=1, graph_optimization=all",
        runtime.execution_mode + ", intra_op=" +
            std::to_string(runtime.intra_op_num_threads) +
            ", inter_op=" + std::to_string(runtime.inter_op_num_threads) +
            ", graph_optimization=" + runtime.graph_optimization_level,
        "restore the fixed OnnxRunner CPU session policy");
  }
  if (!std::isfinite(runtime.session_initialization_ms) ||
      runtime.session_initialization_ms < 0.0) {
    throw_benchmark_error(
        "runtime.session.initialization_ms",
        "one finite non-negative Ort::Session construction duration",
        std::to_string(runtime.session_initialization_ms),
        "measure around the unprofiled Ort::Session constructor");
  }
  if (runtime.profiling_enabled) {
    throw_benchmark_error(
        "runtime.session.profiling_enabled", "false", "true",
        "run ORT profiling in the separate profile workflow and recreate an "
        "unprofiled session for the formal benchmark");
  }

  const BenchmarkModelMetadata& model = result.model;
  validate_non_empty(model.model_id, "model.model_id");
  validate_non_empty(model.model_family, "model.model_family");
  validate_non_empty(model.model_path, "model.path");
  validate_non_empty(model.declared_sha256, "model.declared_sha256");
  if (!is_sha256(model.declared_sha256)) {
    throw_benchmark_error(
        "model.declared_sha256", "64 hexadecimal characters",
        model.declared_sha256,
        "copy the actual digest into the selected FP32 or INT8 artifact "
        "contract");
  }
  const std::vector<std::int64_t> expected_input_shape = {1, 3, 800, 800};
  if (model.model_family != "yolov8" || model.file_size_bytes == 0 ||
      model.opset != 17 ||
      model.input_name != "images" ||
      model.input_shape != expected_input_shape ||
      model.input_dtype != "float32" || model.input_layout != "nchw") {
    throw_benchmark_error(
        "model.contract",
        "a non-empty YOLOv8 FP32-or-INT8 artifact with opset 17 and "
        "images float32 [1,3,800,800] NCHW external I/O",
        "id=" + model.model_id +
            ", size=" + std::to_string(model.file_size_bytes) +
            ", opset=" + std::to_string(model.opset),
        "load one validated S2-01 artifact and pass its correctness gate "
        "before benchmarking");
  }
  validate_non_empty(model.input_name, "model.input.name");
  validate_non_empty(model.input_dtype, "model.input.dtype");
  validate_non_empty(model.input_layout, "model.input.layout");

  const BenchmarkSampleMetadata& sample = result.sample;
  validate_non_empty(sample.image_path, "sample.image_path");
  if (sample.file_size_bytes == 0 || sample.width <= 0 ||
      sample.height <= 0 || sample.channels != 3) {
    throw_benchmark_error(
        "sample.metadata", "one non-empty decoded 3-channel image",
        "file_size=" + std::to_string(sample.file_size_bytes) +
            ", shape=" + std::to_string(sample.width) + "x" +
            std::to_string(sample.height) + "x" +
            std::to_string(sample.channels),
        "use the fixed readable baseline image");
  }
  if (sample.file_size_bytes != 23845 || sample.width != 200 ||
      sample.height != 200) {
    throw_benchmark_error(
        "sample.baseline", "crazing_241.jpg, 23845 bytes, 200x200x3",
        "size=" + std::to_string(sample.file_size_bytes) +
            ", shape=" + std::to_string(sample.width) + "x" +
            std::to_string(sample.height) + "x" +
            std::to_string(sample.channels),
        "use the frozen baseline image and rerun S1-07 consistency");
  }
  if (!std::isfinite(result.score_threshold) ||
      result.score_threshold < 0.0 || result.score_threshold > 1.0 ||
      !std::isfinite(result.nms_threshold) || result.nms_threshold < 0.0 ||
      result.nms_threshold > 1.0) {
    throw_benchmark_error(
        "postprocess.thresholds", "finite values in [0,1]",
        "score=" + std::to_string(result.score_threshold) +
            ", nms=" + std::to_string(result.nms_threshold),
        "use the validated RuntimeConfig thresholds");
  }
  if (std::abs(result.score_threshold - 0.25) > 1.0e-12 ||
      std::abs(result.nms_threshold - 0.45) > 1.0e-12) {
    throw_benchmark_error(
        "postprocess.thresholds", "score=0.25 and nms=0.45",
        "score=" + std::to_string(result.score_threshold) +
            ", nms=" + std::to_string(result.nms_threshold),
        "restore the frozen default RuntimeConfig before benchmarking");
  }
  if (result.nms_mode != "class_agnostic") {
    throw_benchmark_error(
        "postprocess.nms_mode", "class_agnostic", result.nms_mode,
        "restore the frozen YOLOv8 artifact contract");
  }

  validate_statistics(result.latency.image_decode, result.repeat,
                      "latency_ms.image_decode");
  validate_statistics(result.latency.preprocess, result.repeat,
                      "latency_ms.preprocess");
  validate_statistics(result.latency.session_run, result.repeat,
                      "latency_ms.session_run");
  validate_statistics(result.latency.postprocess, result.repeat,
                      "latency_ms.postprocess");
  validate_statistics(result.latency.pipeline, result.repeat,
                      "latency_ms.pipeline");
  validate_statistics(result.latency.end_to_end, result.repeat,
                      "latency_ms.end_to_end");
  const double expected_pipeline_throughput =
      calculate_throughput_images_per_second(result.latency.pipeline);
  const double expected_end_to_end_throughput =
      calculate_throughput_images_per_second(result.latency.end_to_end);
  const auto validate_throughput = [](double actual, double expected,
                                      const std::string& object) {
    const double tolerance = std::max(1.0, std::abs(expected)) * 1.0e-12;
    if (!std::isfinite(actual) || actual <= 0.0 ||
        std::abs(actual - expected) > tolerance) {
      throw_benchmark_error(
          object, "1000 / mean_ms", std::to_string(actual),
          "derive batch-1 throughput from the matching mean latency");
    }
  };
  validate_throughput(result.latency.pipeline_throughput_images_per_second,
                      expected_pipeline_throughput,
                      "throughput.pipeline_images_per_second");
  validate_throughput(result.latency.end_to_end_throughput_images_per_second,
                      expected_end_to_end_throughput,
                      "throughput.end_to_end_images_per_second");

  const BenchmarkMemoryEvidence& memory = result.memory;
  validate_non_empty(memory.status, "memory.status");
  validate_non_empty(memory.metric, "memory.metric");
  validate_non_empty(memory.scope, "memory.scope");
  if (memory.supported) {
    if (memory.status != "supported" || memory.bytes == 0 ||
        !std::isfinite(memory.mebibytes) || memory.mebibytes <= 0.0) {
      throw_benchmark_error(
          "memory", "supported Peak Working Set with positive bytes/MiB",
          "status=" + memory.status +
              ", bytes=" + std::to_string(memory.bytes),
          "query the process memory API after timed iterations");
    }
    const double expected_mebibytes =
        static_cast<double>(memory.bytes) / (1024.0 * 1024.0);
    const double tolerance =
        std::max(1.0, expected_mebibytes) * 1.0e-12;
    if (std::abs(memory.mebibytes - expected_mebibytes) > tolerance) {
      throw_benchmark_error(
          "memory.mebibytes", "bytes / (1024 * 1024)",
          std::to_string(memory.mebibytes),
          "derive MiB from the recorded Peak Working Set byte count");
    }
  } else if (memory.status != "unsupported" || memory.reason.empty()) {
    throw_benchmark_error(
        "memory", "unsupported with a non-empty reason",
        "status=" + memory.status + ", reason=" + memory.reason,
        "report unsupported explicitly instead of publishing zero memory");
  }
  if (result.timing_exclusions.empty() || result.limitations.empty()) {
    throw_benchmark_error(
        "disclosures", "non-empty timing exclusions and limitations",
        "missing disclosure list",
        "state what the benchmark does not time and cannot prove");
  }
  for (std::size_t index = 0; index < result.timing_exclusions.size();
       ++index) {
    validate_non_empty(
        result.timing_exclusions[index],
        "timing_exclusions[" + std::to_string(index) + "]");
  }
  for (std::size_t index = 0; index < result.limitations.size(); ++index) {
    validate_non_empty(result.limitations[index],
                       "limitations[" + std::to_string(index) + "]");
  }
}

}  // namespace yolo_defect_cpp

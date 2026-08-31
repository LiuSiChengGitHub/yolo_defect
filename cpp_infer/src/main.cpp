#include "yolo_defect_cpp/benchmark_runner.h"
#include "yolo_defect_cpp/benchmark_writer.h"
#include "yolo_defect_cpp/batch_runner.h"
#include "yolo_defect_cpp/batch_writer.h"
#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/detector_pipeline.h"
#include "yolo_defect_cpp/image_preprocessor.h"
#include "yolo_defect_cpp/onnx_runner.h"
#include "yolo_defect_cpp/profile_runner.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct CliOptions {
  bool show_help = false;
  bool inspect_model = false;
  bool raw_output_summary = false;
  bool benchmark = false;
  bool profile = false;
  bool batch = false;
  bool output_images = false;
  bool overwrite_existing = false;
  bool warmup_provided = false;
  bool repeat_provided = false;
  bool profile_runs_provided = false;
  bool workers_provided = false;
  bool queue_capacity_provided = false;
  std::size_t warmup = 10;
  std::size_t repeat = 100;
  std::size_t profile_runs = 10;
  std::size_t workers = 1;
  std::size_t queue_capacity = 2;
  std::string config_path;
  std::string image_path;
  std::string output_json_path;
  std::string output_image_path;
  std::string benchmark_json_path;
  std::string profile_prefix_path;
  std::string input_directory_path;
  std::string manifest_path;
  std::string output_directory_path;
  std::string batch_summary_path;
};

struct NumericSummary {
  std::size_t finite_values = 0;
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
};

volatile std::sig_atomic_t g_batch_signal = 0;

void handle_batch_signal(int signal_number) {
  g_batch_signal = signal_number;
}

class ScopedBatchSignalHandlers {
 public:
  ScopedBatchSignalHandlers() {
    g_batch_signal = 0;
    previous_interrupt_ = std::signal(SIGINT, handle_batch_signal);
    if (previous_interrupt_ == SIG_ERR) {
      throw std::runtime_error(
          "Batch signal setup failed: object=SIGINT; expected=an installed "
          "cooperative handler; actual=std::signal returned SIG_ERR; "
          "action=verify the process signal environment.");
    }
#ifdef SIGTERM
    previous_terminate_ = std::signal(SIGTERM, handle_batch_signal);
    if (previous_terminate_ == SIG_ERR) {
      std::signal(SIGINT, previous_interrupt_);
      throw std::runtime_error(
          "Batch signal setup failed: object=SIGTERM; expected=an installed "
          "cooperative handler; actual=std::signal returned SIG_ERR; "
          "action=verify the process signal environment.");
    }
#endif
#ifdef SIGBREAK
    previous_break_ = std::signal(SIGBREAK, handle_batch_signal);
    if (previous_break_ == SIG_ERR) {
#ifdef SIGTERM
      std::signal(SIGTERM, previous_terminate_);
#endif
      std::signal(SIGINT, previous_interrupt_);
      throw std::runtime_error(
          "Batch signal setup failed: object=SIGBREAK; expected=an installed "
          "cooperative handler; actual=std::signal returned SIG_ERR; "
          "action=verify the Windows console signal environment.");
    }
#endif
  }

  ~ScopedBatchSignalHandlers() {
#ifdef SIGBREAK
    std::signal(SIGBREAK, previous_break_);
#endif
#ifdef SIGTERM
    std::signal(SIGTERM, previous_terminate_);
#endif
    std::signal(SIGINT, previous_interrupt_);
  }

  ScopedBatchSignalHandlers(const ScopedBatchSignalHandlers&) = delete;
  ScopedBatchSignalHandlers& operator=(
      const ScopedBatchSignalHandlers&) = delete;

 private:
  using SignalHandler = void (*)(int);
  SignalHandler previous_interrupt_ = SIG_DFL;
#ifdef SIGTERM
  SignalHandler previous_terminate_ = SIG_DFL;
#endif
#ifdef SIGBREAK
  SignalHandler previous_break_ = SIG_DFL;
#endif
};

class BatchStopMonitor {
 public:
  explicit BatchStopMonitor(yolo_defect_cpp::BatchRunner& runner)
      : runner_(runner), thread_([this]() { monitor(); }) {}

  ~BatchStopMonitor() { (void)stop(); }

  BatchStopMonitor(const BatchStopMonitor&) = delete;
  BatchStopMonitor& operator=(const BatchStopMonitor&) = delete;

  bool stop() noexcept {
    observe_pending_signal();
    done_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
    // Close the polling tail window: a signal may have set the async-safe
    // flag while the monitor was sleeping immediately before run() returned.
    observe_pending_signal();
    return signal_observed_.load(std::memory_order_acquire);
  }

 private:
  void observe_pending_signal() noexcept {
    if (g_batch_signal != 0) {
      signal_observed_.store(true, std::memory_order_release);
      runner_.request_stop();
    }
  }

  void monitor() noexcept {
    while (!done_.load(std::memory_order_acquire)) {
      if (g_batch_signal != 0) {
        signal_observed_.store(true, std::memory_order_release);
        runner_.request_stop();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  yolo_defect_cpp::BatchRunner& runner_;
  std::atomic<bool> done_{false};
  std::atomic<bool> signal_observed_{false};
  std::thread thread_;
};

std::string format_shape(const std::vector<std::int64_t>& shape) {
  std::ostringstream output;
  output << "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index > 0) {
      output << ",";
    }
    output << shape[index];
  }
  output << "]";
  return output.str();
}

std::string format_class_names(const std::vector<std::string>& class_names) {
  std::ostringstream output;
  for (std::size_t index = 0; index < class_names.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << class_names[index];
  }
  return output.str();
}

void print_help(const char* program_name) {
  std::cout
      << "yolo_defect_cpp - S2-04 accelerator-capable Runtime CLI\n"
      << "\n"
      << "Usage:\n"
      << "  " << program_name << " [--help]\n"
      << "  " << program_name << " --config <config_path>\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path>\n"
      << "  " << program_name
      << " --config <config_path> --inspect-model\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path>"
         " --raw-output-summary\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path>"
         " [--output-json <path>] [--output-image <path>] [--overwrite]\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path> --benchmark"
         " --warmup <count> --repeat <count>"
         " --benchmark-json <path> [--overwrite]\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path> --profile"
         " --profile-prefix <path> [--profile-runs <count>]\n"
      << "  " << program_name
      << " --config <config_path> --batch"
         " (--input-dir <directory> | --manifest <path-list>)"
         " --output-dir <directory> --batch-summary <path>"
         " [--workers <1..64>] [--queue-capacity <1..4096>]"
         " [--output-images] [--overwrite]\n"
      << "\n"
      << "Scope:\n"
      << "  Loads and validates RuntimeConfig + ModelArtifactSpec.\n"
      << "  Optional --image keeps the existing OpenCV preprocess smoke.\n"
      << "  --inspect-model creates the configured backend, reads model\n"
      << "  metadata, and validates it against the artifact contract.\n"
      << "  --raw-output-summary preprocesses one image, runs one synchronous\n"
      << "  raw inference, and prints only bounded tensor summaries.\n"
      << "  --output-json and --output-image run the S1-05 single-image\n"
      << "  pipeline: preprocess, configured inference, tested postprocess,\n"
      << "  and files.\n"
      << "  --benchmark runs the Release-only batch-1 provider protocol.\n"
      << "  Defaults are --warmup 10 and --repeat 100. It reports mean/P50/P95\n"
      << "  for image decode, preprocess, backend run, postprocess, pipeline,\n"
      << "  and end-to-end, then writes machine-readable --benchmark-json.\n"
      << "  Backend/model initialization, statistics, JSON writing, and drawing\n"
      << "  are outside repeated timing; drawing is not executed.\n"
      << "  --profile creates a separate profiling-enabled ORT session, runs\n"
      << "  the fixed preprocessed image (default 10 times), calls EndProfiling,\n"
      << "  and reports the actual ORT-generated JSON trace path. Profile timing\n"
      << "  contains instrumentation overhead and is not benchmark evidence.\n"
      << "  --batch discovers deterministic tasks and runs a bounded queue.\n"
      << "  Each worker owns one existing batch=1 DetectorPipeline/ORT session;\n"
      << "  JSON is written per success and --output-images also writes PNG.\n"
      << "  Directories are recursive and sorted; UTF-8 manifest paths remain\n"
      << "  in declaration order and resolve from the manifest directory.\n"
      << "  Output parents are created recursively. Existing regular files are\n"
      << "  rejected unless --overwrite is explicit; paths matching protected\n"
      << "  inputs are rejected before writing. Relative CLI image/output paths\n"
      << "  use the current working directory. No GUI, true ONNX batch, video,\n"
      << "  service, GPU stream concurrency, or lock-free queue exists.\n";
}

void print_banner() {
  std::cout
      << "yolo_defect_cpp - S2-04 accelerator-capable Runtime CLI\n"
      << "V2 Runtime: industrial vision AI deployment workspace\n"
      << "Current scope: validated single-image detection, S1-07 consistency "
         "evidence, S1-08 Release CPU benchmark evidence, and S2-01 ORT "
         "profiling, bounded multi-image CPU execution, and an opt-in Linux "
         "TensorRT path\n"
      << "Run with --help for single-image, batch, benchmark, and profile "
         "commands.\n"
      << "Batch remains concurrent batch=1 work; true batch, video, services, "
         "and GPU concurrency are out of scope.\n";
}

void print_contract_summary(
    const yolo_defect_cpp::RuntimeContract& contract) {
  const yolo_defect_cpp::RuntimeConfig& runtime = contract.runtime;
  const yolo_defect_cpp::ModelArtifactSpec& artifact = contract.artifact;

  std::cout
      << "S1-01 Runtime/artifact contract summary\n"
      << "runtime_config_path: " << runtime.declaration_path.string() << "\n"
      << "runtime_schema_version: " << runtime.schema_version << "\n"
      << "artifact_spec_path: " << runtime.artifact_spec_path.string() << "\n"
      << "artifact_schema_version: " << artifact.schema_version << "\n"
      << "model_id: " << artifact.model_id << "\n"
      << "model_family: "
      << yolo_defect_cpp::to_string(artifact.model_family) << "\n"
      << "model_path: " << artifact.model_path.string() << "\n"
      << "declared_model_sha256: " << artifact.model_sha256 << "\n"
      << "opset: " << artifact.opset << "\n"
      << "source: " << artifact.source << "\n"
      << "provenance: " << artifact.provenance << "\n"
      << "artifact_license: " << artifact.artifact_license << "\n"
      << "input_name: " << artifact.input.name << "\n"
      << "input_shape: " << format_shape(artifact.input.shape) << "\n"
      << "input_dtype: "
      << yolo_defect_cpp::to_string(artifact.input.dtype) << "\n"
      << "input_layout: "
      << yolo_defect_cpp::to_string(artifact.input.layout) << "\n"
      << "output_name: " << artifact.output.name << "\n"
      << "output_shape: " << format_shape(artifact.output.shape) << "\n"
      << "output_dtype: "
      << yolo_defect_cpp::to_string(artifact.output.dtype) << "\n"
      << "output_layout: "
      << yolo_defect_cpp::to_string(artifact.output.layout) << "\n"
      << "class_count: " << artifact.class_names.size() << "\n"
      << "class_names: " << format_class_names(artifact.class_names) << "\n"
      << std::fixed << std::setprecision(6)
      << "score_threshold: " << runtime.score_threshold << "\n"
      << "nms_threshold: " << runtime.nms_threshold << "\n"
      << "configured_provider: "
      << yolo_defect_cpp::to_string(runtime.provider) << "\n"
      << "preprocess_type: "
      << yolo_defect_cpp::to_string(artifact.preprocess_type) << "\n"
      << "postprocess_type: "
      << yolo_defect_cpp::to_string(artifact.postprocess_type) << "\n"
      << "nms_mode: " << yolo_defect_cpp::to_string(artifact.nms_mode) << "\n"
      << "scope: declaration contract only; this command did not create a "
         "backend/session or run inference.\n";
  if (runtime.tensorrt.has_value()) {
    const yolo_defect_cpp::TensorRtProviderConfig& tensorrt =
        *runtime.tensorrt;
    std::cout
        << "tensorrt_device_id: " << tensorrt.device_id << "\n"
        << "tensorrt_precision: "
        << yolo_defect_cpp::to_string(tensorrt.precision) << "\n"
        << "tensorrt_max_workspace_size_bytes: "
        << tensorrt.max_workspace_size_bytes << "\n"
        << "tensorrt_engine_cache_path: "
        << tensorrt.engine_cache_path.string() << "\n";
  }
}

void print_preprocess_summary(
    const std::string& image_path,
    const yolo_defect_cpp::RuntimeContract& contract,
    const yolo_defect_cpp::PreprocessResult& result) {
  std::cout
      << "S1-01 Preprocess summary\n"
      << "runtime_config_path: "
      << contract.runtime.declaration_path.string() << "\n"
      << "artifact_spec_path: "
      << contract.artifact.declaration_path.string() << "\n"
      << "model_id: " << contract.artifact.model_id << "\n"
      << "image_path: " << image_path << "\n"
      << "original_size: " << result.original_width << "x"
      << result.original_height << "\n"
      << "channels: " << result.original_channels << "\n"
      << "input_size: " << result.input_width << "x" << result.input_height
      << "\n"
      << "resized_size: " << result.resized_width << "x"
      << result.resized_height << "\n"
      << std::fixed << std::setprecision(6)
      << "scale: " << result.scale << "\n"
      << "padding: left=" << result.pad_left
      << ", top=" << result.pad_top
      << ", right=" << result.pad_right
      << ", bottom=" << result.pad_bottom << "\n"
      << "color: BGR->RGB\n"
      << "normalization: float32 [0, 1]\n"
      << "layout: NCHW\n"
      << "tensor_shape: 1x3x" << result.input_height << "x"
      << result.input_width << "\n"
      << "tensor_elements: " << result.tensor_nchw.size() << "\n"
      << "scope: contract + preprocess only; this command did not create a "
         "backend/session or run inference.\n";
}

void print_model_metadata_summary(
    const yolo_defect_cpp::RuntimeContract& contract,
    const yolo_defect_cpp::ModelMetadata& metadata) {
  std::cout
      << "C++ runtime model inspection\n"
      << "model_path: " << contract.artifact.model_path.string() << "\n"
      << "ort_version: " << metadata.ort_version << "\n"
      << "configured_provider: "
      << yolo_defect_cpp::to_string(contract.runtime.provider) << "\n"
      << "available_providers: "
      << yolo_defect_cpp::format_string_list(
             metadata.available_providers)
      << "\n"
      << "session_provider: " << metadata.session_provider << "\n"
      << "provider_evidence: " << metadata.provider_evidence << "\n"
      << "registered_provider_chain: "
      << yolo_defect_cpp::format_string_list(
             metadata.registered_provider_chain) << "\n"
      << "inference_precision: " << metadata.inference_precision << "\n"
      << "device_id: " << metadata.device_id << "\n"
      << "engine_cache_enabled: "
      << (metadata.engine_cache_enabled ? "true" : "false") << "\n"
      << "engine_cache_path: " << metadata.engine_cache_path << "\n"
      << "engine_cache_prefix: " << metadata.engine_cache_prefix << "\n"
      << "engine_cache_state: " << metadata.engine_cache_state << "\n"
      << "engine_cache_files_before: "
      << metadata.engine_cache_files_before << "\n"
      << "engine_cache_files_after: "
      << metadata.engine_cache_files_after << "\n"
      << "execution_mode: " << metadata.execution_mode << "\n"
      << "intra_op_num_threads: " << metadata.intra_op_num_threads << "\n"
      << "inter_op_num_threads: " << metadata.inter_op_num_threads
      << (metadata.execution_mode == "sequential"
              ? " (not used by sequential execution mode)"
              : "")
      << "\n"
      << "graph_optimization_level: "
      << metadata.graph_optimization_level << "\n"
      << "input_count: " << metadata.inputs.size() << "\n";

  for (std::size_t index = 0; index < metadata.inputs.size(); ++index) {
    const yolo_defect_cpp::TensorMetadata& input =
        metadata.inputs[index];
    std::cout
        << "input[" << index << "].name: " << input.name << "\n"
        << "input[" << index << "].value_type: "
        << yolo_defect_cpp::to_string(input.value_type) << "\n"
        << "input[" << index << "].shape: "
        << yolo_defect_cpp::format_shape(input.shape) << "\n"
        << "input[" << index << "].dtype: "
        << yolo_defect_cpp::to_string(input.dtype) << "\n";
  }

  std::cout << "output_count: " << metadata.outputs.size() << "\n";
  for (std::size_t index = 0; index < metadata.outputs.size(); ++index) {
    const yolo_defect_cpp::TensorMetadata& output =
        metadata.outputs[index];
    std::cout
        << "output[" << index << "].name: " << output.name << "\n"
        << "output[" << index << "].value_type: "
        << yolo_defect_cpp::to_string(output.value_type) << "\n"
        << "output[" << index << "].shape: "
        << yolo_defect_cpp::format_shape(output.shape) << "\n"
        << "output[" << index << "].dtype: "
        << yolo_defect_cpp::to_string(output.dtype) << "\n";
  }

  std::cout
      << "metadata_contract_validation: passed\n"
      << "scope: backend creation + metadata validation only; no input "
         "tensor, backend run, inference result, or postprocess.\n";
}

NumericSummary summarize_values(const std::vector<float>& values) {
  NumericSummary summary;
  for (float value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    ++summary.finite_values;
    summary.minimum = std::min(summary.minimum, value);
    summary.maximum = std::max(summary.maximum, value);
  }
  return summary;
}

void print_raw_output_summary(
    const std::vector<std::int64_t>& input_shape,
    const std::vector<float>& input_values,
    const yolo_defect_cpp::InferenceOutput& output) {
  const NumericSummary input_summary = summarize_values(input_values);
  const NumericSummary output_summary = summarize_values(output.values);

  std::cout
      << "S1-03 raw output summary\n"
      << "input_shape: " << format_shape(input_shape) << "\n"
      << "input_elements: " << input_values.size() << "\n"
      << "input_finite_values: " << input_summary.finite_values << "/"
      << input_values.size() << "\n"
      << std::setprecision(9)
      << "input_min: " << input_summary.minimum << "\n"
      << "input_max: " << input_summary.maximum << "\n"
      << "output_shape: "
      << yolo_defect_cpp::format_shape(output.shape) << "\n"
      << "output_elements: " << output.values.size() << "\n"
      << "output_finite_values: " << output_summary.finite_values << "/"
      << output.values.size() << "\n"
      << "output_min: " << output_summary.minimum << "\n"
      << "output_max: " << output_summary.maximum << "\n"
      << "session_run: completed\n"
      << "raw_output_ownership: copied_to_InferenceOutput\n"
      << "scope: raw inference only; no decode, NMS, JSON, visualization, "
         "or benchmark.\n";
}

void print_single_image_summary(
    const yolo_defect_cpp::SingleImagePipelineResult& result,
    bool overwrite_existing) {
  const yolo_defect_cpp::SingleImageDetectionResult& detection =
      result.detection_result;
  std::cout
      << "S1-05 single-image detection completed\n"
      << "schema_version: " << detection.schema_version << "\n"
      << "model_id: " << detection.model_id << "\n"
      << "image_path: " << detection.image.source_path.string() << "\n"
      << "original_size: " << detection.image.original_width << "x"
      << detection.image.original_height << "\n"
      << "input_size: " << detection.image.input_width << "x"
      << detection.image.input_height << "\n"
      << "actual_provider: " << detection.actual_provider << "\n"
      << std::setprecision(9)
      << "score_threshold: " << detection.score_threshold << "\n"
      << "nms_threshold: " << detection.nms_threshold << "\n"
      << "nms_mode: " << yolo_defect_cpp::to_string(detection.nms_mode)
      << "\n"
      << "detection_count: " << detection.detections.size() << "\n"
      << "output_json: "
      << (result.outputs.json_path.has_value()
              ? result.outputs.json_path->string()
              : "<not requested>")
      << "\n"
      << "output_image: "
      << (result.outputs.image_path.has_value()
              ? result.outputs.image_path->string()
              : "<not requested>")
      << "\n"
      << "overwrite_existing: "
      << (overwrite_existing ? "true" : "false") << "\n"
      << "scope: one image only; stable JSON/visualization written without "
         "GUI. No batch, concurrency, service, consistency, or benchmark.\n";
}

void print_latency_statistics(
    const std::string& name,
    const yolo_defect_cpp::LatencyStatistics& statistics) {
  std::cout
      << name << ".mean_ms: " << statistics.mean_ms << "\n"
      << name << ".p50_ms: " << statistics.p50_ms << "\n"
      << name << ".p95_ms: " << statistics.p95_ms << "\n";
}

void print_benchmark_summary(
    const yolo_defect_cpp::BenchmarkResult& result,
    const std::filesystem::path& output_path) {
  std::cout
      << "Reproducible Release benchmark completed\n"
      << "benchmark_json: " << output_path.string() << "\n"
      << "build_type: " << result.environment.build_type << "\n"
      << "requested_provider: " << result.runtime.requested_provider << "\n"
      << "actual_provider: " << result.runtime.actual_provider << "\n"
      << "execution_mode: " << result.runtime.execution_mode << "\n"
      << "intra_op_num_threads: "
      << result.runtime.intra_op_num_threads << "\n"
      << "inter_op_num_threads: "
      << result.runtime.inter_op_num_threads << "\n"
      << std::fixed << std::setprecision(6)
      << "session_initialization_ms: "
      << result.runtime.session_initialization_ms << "\n"
      << "profiling_enabled: "
      << (result.runtime.profiling_enabled ? "true" : "false") << "\n"
      << "batch_size: " << result.batch_size << "\n"
      << "sample_count: " << result.sample_count << "\n"
      << "warmup: " << result.warmup << "\n"
      << "repeat: " << result.repeat << "\n"
      << "detection_count: " << result.sample.detection_count << "\n";
  print_latency_statistics("image_decode", result.latency.image_decode);
  print_latency_statistics("preprocess", result.latency.preprocess);
  print_latency_statistics("session_run", result.latency.session_run);
  print_latency_statistics("postprocess", result.latency.postprocess);
  print_latency_statistics("pipeline", result.latency.pipeline);
  print_latency_statistics("end_to_end", result.latency.end_to_end);
  std::cout
      << "pipeline.throughput_images_per_second: "
      << result.latency.pipeline_throughput_images_per_second << "\n"
      << "end_to_end.throughput_images_per_second: "
      << result.latency.end_to_end_throughput_images_per_second << "\n"
      << "memory.status: " << result.memory.status << "\n";
  if (result.memory.supported) {
    std::cout
        << "memory.metric: " << result.memory.metric << "\n"
        << "memory.bytes: " << result.memory.bytes << "\n"
        << "memory.mebibytes: " << result.memory.mebibytes << "\n";
  } else {
    std::cout << "memory.reason: " << result.memory.reason << "\n";
  }
  std::cout
      << "timing_exclusions: Runtime setup around the separately recorded "
         "backend initialization, statistics, benchmark JSON write, and "
         "visualization\n"
      << "scope: fixed single image, batch=1, configured provider, "
         "warm-cache Release "
         "benchmark; see JSON limitations before comparison.\n";
}

void print_profile_summary(
    const yolo_defect_cpp::ProfileResult& result) {
  std::cout
      << "ORT profiling completed\n"
      << "profile_trace_path: " << result.trace_path.string() << "\n"
      << "profile_runs: " << result.run_count << "\n"
      << "model_id: " << result.model_id << "\n"
      << "declared_model_sha256: " << result.declared_model_sha256 << "\n"
      << "actual_provider: " << result.actual_provider << "\n"
      << std::fixed << std::setprecision(6)
      << "session_initialization_ms_with_profiling: "
      << result.session_initialization_ms << "\n"
      << "output_shape: " << format_shape(result.output_shape) << "\n"
      << "output_elements: " << result.output_element_count << "\n"
      << "detection_count: " << result.detection_count << "\n"
      << "profiling_overhead: enabled; trace timing is diagnostic only\n"
      << "scope: one preprocessed image and one profiling-enabled ORT session; "
         "formal benchmark evidence must come from --benchmark.\n";
}

void print_batch_summary(
    const yolo_defect_cpp::BatchSummary& summary,
    const std::filesystem::path& summary_path) {
  std::cout
      << "S2-03 bounded multi-image batch completed\n"
      << "batch_summary: " << summary_path.string() << "\n"
      << "status: " << yolo_defect_cpp::to_string(summary.status) << "\n"
      << "input_kind: " << yolo_defect_cpp::to_string(summary.input.kind)
      << "\n"
      << "input_path: " << summary.input.source_path.string() << "\n"
      << "requested_workers: " << summary.runtime.requested_workers << "\n"
      << "effective_workers: " << summary.runtime.effective_workers << "\n"
      << "queue_capacity: " << summary.queue.capacity << "\n"
      << "queue_peak_depth: " << summary.queue.peak_depth << "\n"
      << "producer_wait_count: " << summary.queue.producer_wait_count << "\n"
      << "discovered: " << summary.counts.discovered << "\n"
      << "succeeded: " << summary.counts.succeeded << "\n"
      << "failed: " << summary.counts.failed << "\n"
      << "cancelled: " << summary.counts.cancelled << "\n"
      << std::fixed << std::setprecision(6)
      << "processing_wall_ms: " << summary.timing.processing_wall_ms << "\n"
      << "throughput_images_per_second: "
      << summary.throughput_images_per_second << "\n"
      << "memory_metric: " << summary.memory.metric << "\n"
      << "memory_mebibytes: " << summary.memory.mebibytes << "\n"
      << "memory_publishable: "
      << (summary.memory.publishable ? "true" : "false") << "\n"
      << "scope: independent batch=1 DetectorPipeline workers with bounded "
         "queue; no true batch, video, service, GPU concurrency, or "
         "lock-free queue.\n";
}

int batch_exit_code(yolo_defect_cpp::BatchStatus status) {
  switch (status) {
    case yolo_defect_cpp::BatchStatus::kSucceeded:
      return 0;
    case yolo_defect_cpp::BatchStatus::kPartialFailure:
      return 2;
    case yolo_defect_cpp::BatchStatus::kCancelled:
      return 130;
    case yolo_defect_cpp::BatchStatus::kFatal:
      return 1;
  }
  return 1;
}

std::size_t parse_count_value(const std::string& value,
                              const std::string& option,
                              std::size_t minimum,
                              std::size_t maximum = 1000000) {
  std::size_t parsed = 0;
  const char* const begin = value.data();
  const char* const end = begin + value.size();
  const std::from_chars_result conversion =
      std::from_chars(begin, end, parsed, 10);
  if (value.empty() || conversion.ec != std::errc{} ||
      conversion.ptr != end || parsed < minimum ||
      parsed > maximum) {
    std::string action =
        "use --warmup 10 and --repeat 100 for the formal S1-08 baseline";
    if (option == "--profile-runs") {
      action = "use the frozen --profile-runs 10 count for formal profiling";
    } else if (option == "--workers") {
      action = "choose a bounded number of independent worker sessions";
    } else if (option == "--queue-capacity") {
      action = "choose a bounded queue capacity";
    }
    throw std::runtime_error(
        "CLI argument error: object=" + option +
        "; expected=an integer in [" + std::to_string(minimum) +
        "," + std::to_string(maximum) + "]; actual='" + value +
        "'; action=" + action + ".");
  }
  return parsed;
}

CliOptions parse_cli(int argc, char* argv[]) {
  CliOptions options;
  const auto read_path_value = [argc, argv](int& index,
                                             const std::string& option) {
    if (index + 1 >= argc) {
      throw std::runtime_error(
          "CLI argument error: object=" + option +
          "; expected=one non-empty path; actual=missing value; action="
          "provide a path immediately after " + option + ".");
    }
    const std::string value = argv[index + 1];
    if (value.empty() || value.rfind("--", 0) == 0) {
      throw std::runtime_error(
          "CLI argument error: object=" + option +
          "; expected=one non-empty path; actual='" + value +
          "'; action=provide a path immediately after " + option + ".");
    }
    ++index;
    return value;
  };

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      options.show_help = true;
      continue;
    }

    if (argument == "--config") {
      if (!options.config_path.empty()) {
        throw std::runtime_error("--config was provided more than once.");
      }
      options.config_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--image") {
      if (!options.image_path.empty()) {
        throw std::runtime_error("--image was provided more than once.");
      }
      options.image_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--output-json") {
      if (!options.output_json_path.empty()) {
        throw std::runtime_error(
            "--output-json was provided more than once.");
      }
      options.output_json_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--output-image") {
      if (!options.output_image_path.empty()) {
        throw std::runtime_error(
            "--output-image was provided more than once.");
      }
      options.output_image_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--benchmark-json") {
      if (!options.benchmark_json_path.empty()) {
        throw std::runtime_error(
            "--benchmark-json was provided more than once.");
      }
      options.benchmark_json_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--profile-prefix") {
      if (!options.profile_prefix_path.empty()) {
        throw std::runtime_error(
            "--profile-prefix was provided more than once.");
      }
      options.profile_prefix_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--input-dir") {
      if (!options.input_directory_path.empty()) {
        throw std::runtime_error("--input-dir was provided more than once.");
      }
      options.input_directory_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--manifest") {
      if (!options.manifest_path.empty()) {
        throw std::runtime_error("--manifest was provided more than once.");
      }
      options.manifest_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--output-dir") {
      if (!options.output_directory_path.empty()) {
        throw std::runtime_error("--output-dir was provided more than once.");
      }
      options.output_directory_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--batch-summary") {
      if (!options.batch_summary_path.empty()) {
        throw std::runtime_error(
            "--batch-summary was provided more than once.");
      }
      options.batch_summary_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--benchmark") {
      if (options.benchmark) {
        throw std::runtime_error("--benchmark was provided more than once.");
      }
      options.benchmark = true;
      continue;
    }

    if (argument == "--profile") {
      if (options.profile) {
        throw std::runtime_error("--profile was provided more than once.");
      }
      options.profile = true;
      continue;
    }

    if (argument == "--batch") {
      if (options.batch) {
        throw std::runtime_error("--batch was provided more than once.");
      }
      options.batch = true;
      continue;
    }

    if (argument == "--output-images") {
      if (options.output_images) {
        throw std::runtime_error(
            "--output-images was provided more than once.");
      }
      options.output_images = true;
      continue;
    }

    if (argument == "--warmup") {
      if (options.warmup_provided) {
        throw std::runtime_error("--warmup was provided more than once.");
      }
      const std::string value = read_path_value(index, argument);
      options.warmup = parse_count_value(value, argument, 0);
      options.warmup_provided = true;
      continue;
    }

    if (argument == "--repeat") {
      if (options.repeat_provided) {
        throw std::runtime_error("--repeat was provided more than once.");
      }
      const std::string value = read_path_value(index, argument);
      options.repeat = parse_count_value(value, argument, 1);
      options.repeat_provided = true;
      continue;
    }

    if (argument == "--profile-runs") {
      if (options.profile_runs_provided) {
        throw std::runtime_error(
            "--profile-runs was provided more than once.");
      }
      const std::string value = read_path_value(index, argument);
      options.profile_runs = parse_count_value(value, argument, 1);
      options.profile_runs_provided = true;
      continue;
    }

    if (argument == "--workers") {
      if (options.workers_provided) {
        throw std::runtime_error("--workers was provided more than once.");
      }
      const std::string value = read_path_value(index, argument);
      options.workers = parse_count_value(value, argument, 1, 64);
      options.workers_provided = true;
      continue;
    }

    if (argument == "--queue-capacity") {
      if (options.queue_capacity_provided) {
        throw std::runtime_error(
            "--queue-capacity was provided more than once.");
      }
      const std::string value = read_path_value(index, argument);
      options.queue_capacity = parse_count_value(value, argument, 1, 4096);
      options.queue_capacity_provided = true;
      continue;
    }

    if (argument == "--overwrite") {
      if (options.overwrite_existing) {
        throw std::runtime_error("--overwrite was provided more than once.");
      }
      options.overwrite_existing = true;
      continue;
    }

    if (argument == "--inspect-model") {
      if (options.inspect_model) {
        throw std::runtime_error(
            "--inspect-model was provided more than once.");
      }
      options.inspect_model = true;
      continue;
    }

    if (argument == "--raw-output-summary") {
      if (options.raw_output_summary) {
        throw std::runtime_error(
            "--raw-output-summary was provided more than once.");
      }
      options.raw_output_summary = true;
      continue;
    }

    throw std::runtime_error("Unknown argument: " + argument);
  }

  if (options.batch && !options.queue_capacity_provided) {
    options.queue_capacity = options.workers * 2;
  }

  if (!options.image_path.empty() && options.config_path.empty()) {
    throw std::runtime_error("--image requires --config.");
  }
  if (options.inspect_model && options.config_path.empty()) {
    throw std::runtime_error("--inspect-model requires --config.");
  }
  if (options.raw_output_summary && options.config_path.empty()) {
    throw std::runtime_error("--raw-output-summary requires --config.");
  }
  if (options.raw_output_summary && options.image_path.empty()) {
    throw std::runtime_error("--raw-output-summary requires --image.");
  }
  const bool output_requested = !options.output_json_path.empty() ||
                                !options.output_image_path.empty();
  const bool batch_option_requested =
      !options.input_directory_path.empty() || !options.manifest_path.empty() ||
      !options.output_directory_path.empty() ||
      !options.batch_summary_path.empty() || options.output_images ||
      options.workers_provided || options.queue_capacity_provided;
  if (!options.batch && batch_option_requested) {
    throw std::runtime_error(
        "--input-dir/--manifest/--output-dir/--batch-summary/--workers/"
        "--queue-capacity/--output-images require --batch.");
  }
  if (options.batch && options.config_path.empty()) {
    throw std::runtime_error("--batch requires --config.");
  }
  if (options.batch &&
      (options.input_directory_path.empty() == options.manifest_path.empty())) {
    throw std::runtime_error(
        "--batch requires exactly one of --input-dir or --manifest.");
  }
  if (options.batch && options.output_directory_path.empty()) {
    throw std::runtime_error("--batch requires --output-dir.");
  }
  if (options.batch && options.batch_summary_path.empty()) {
    throw std::runtime_error("--batch requires --batch-summary.");
  }
  if (options.batch && !options.image_path.empty()) {
    throw std::runtime_error("--batch and --image are mutually exclusive.");
  }
  if (options.batch && output_requested) {
    throw std::runtime_error(
        "--batch and --output-json/--output-image are mutually exclusive.");
  }
  if (options.batch &&
      (options.benchmark || options.profile || options.inspect_model ||
       options.raw_output_summary)) {
    throw std::runtime_error(
        "--batch is mutually exclusive with --benchmark, --profile, "
        "--inspect-model, and --raw-output-summary.");
  }
  if (!options.benchmark && !options.benchmark_json_path.empty()) {
    throw std::runtime_error("--benchmark-json requires --benchmark.");
  }
  if (!options.benchmark && options.warmup_provided) {
    throw std::runtime_error("--warmup requires --benchmark.");
  }
  if (!options.benchmark && options.repeat_provided) {
    throw std::runtime_error("--repeat requires --benchmark.");
  }
  if (!options.profile && !options.profile_prefix_path.empty()) {
    throw std::runtime_error("--profile-prefix requires --profile.");
  }
  if (!options.profile && options.profile_runs_provided) {
    throw std::runtime_error("--profile-runs requires --profile.");
  }
  if (options.benchmark && options.config_path.empty()) {
    throw std::runtime_error("--benchmark requires --config.");
  }
  if (options.benchmark && options.image_path.empty()) {
    throw std::runtime_error("--benchmark requires --image.");
  }
  if (options.benchmark && options.benchmark_json_path.empty()) {
    throw std::runtime_error("--benchmark requires --benchmark-json.");
  }
  if (options.benchmark && output_requested) {
    throw std::runtime_error(
        "--benchmark and --output-json/--output-image are mutually "
        "exclusive.");
  }
  if (options.benchmark && options.inspect_model) {
    throw std::runtime_error(
        "--benchmark and --inspect-model are mutually exclusive.");
  }
  if (options.benchmark && options.raw_output_summary) {
    throw std::runtime_error(
        "--benchmark and --raw-output-summary are mutually exclusive.");
  }
  if (options.profile && options.config_path.empty()) {
    throw std::runtime_error("--profile requires --config.");
  }
  if (options.profile && options.image_path.empty()) {
    throw std::runtime_error("--profile requires --image.");
  }
  if (options.profile && options.profile_prefix_path.empty()) {
    throw std::runtime_error("--profile requires --profile-prefix.");
  }
  if (options.profile && options.benchmark) {
    throw std::runtime_error(
        "--profile and --benchmark are mutually exclusive.");
  }
  if (options.profile && output_requested) {
    throw std::runtime_error(
        "--profile and --output-json/--output-image are mutually exclusive.");
  }
  if (options.profile && options.inspect_model) {
    throw std::runtime_error(
        "--profile and --inspect-model are mutually exclusive.");
  }
  if (options.profile && options.raw_output_summary) {
    throw std::runtime_error(
        "--profile and --raw-output-summary are mutually exclusive.");
  }
  if (output_requested && options.config_path.empty()) {
    throw std::runtime_error(
        "--output-json/--output-image require --config.");
  }
  if (output_requested && options.inspect_model) {
    throw std::runtime_error(
        "--output-json/--output-image and --inspect-model are mutually "
        "exclusive.");
  }
  if (output_requested && options.raw_output_summary) {
    throw std::runtime_error(
        "--output-json/--output-image and --raw-output-summary are mutually "
        "exclusive.");
  }
  if (output_requested && options.image_path.empty()) {
    throw std::runtime_error(
        "--output-json/--output-image require --image.");
  }
  if (options.overwrite_existing && !output_requested &&
      options.benchmark_json_path.empty() && !options.batch) {
    throw std::runtime_error(
        "--overwrite requires --output-json or --output-image, or "
        "--benchmark-json, or --batch.");
  }
  if (options.inspect_model && options.raw_output_summary) {
    throw std::runtime_error(
        "--inspect-model and --raw-output-summary are mutually exclusive.");
  }
  if (options.inspect_model && !options.image_path.empty()) {
    throw std::runtime_error(
        "--inspect-model and --image are mutually exclusive.");
  }
  return options;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const CliOptions options = parse_cli(argc, argv);
    if (options.show_help) {
      print_help(argv[0]);
      return 0;
    }

    if (!options.config_path.empty()) {
      const yolo_defect_cpp::RuntimeContract contract =
          yolo_defect_cpp::load_runtime_contract(options.config_path);
      if (options.batch) {
        if (contract.runtime.provider !=
            yolo_defect_cpp::ExecutionProvider::kCpu) {
          throw std::runtime_error(
              "--batch currently supports provider=cpu only; use the "
              "single-image or --benchmark path for the S2-04 TensorRT "
              "evidence instead of creating concurrent GPU sessions that "
              "share one engine cache; --profile is available only for the "
              "ORT TensorRT EP backend.");
        }
        yolo_defect_cpp::BatchRequest request;
        request.input_kind = options.input_directory_path.empty()
                                 ? yolo_defect_cpp::BatchInputKind::kManifest
                                 : yolo_defect_cpp::BatchInputKind::kDirectory;
        request.input_path = options.input_directory_path.empty()
                                 ? options.manifest_path
                                 : options.input_directory_path;
        request.output_directory = options.output_directory_path;
        request.summary_path = options.batch_summary_path;
        request.requested_workers = options.workers;
        request.queue_capacity = options.queue_capacity;
        request.output_images = options.output_images;
        request.overwrite_existing = options.overwrite_existing;
        request.command_arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
          request.command_arguments.emplace_back(argv[index]);
        }

        yolo_defect_cpp::BatchRunner runner(contract);
        ScopedBatchSignalHandlers signal_handlers;
        BatchStopMonitor stop_monitor(runner);
        yolo_defect_cpp::BatchSummary summary = runner.run(request);
        const bool signal_observed = stop_monitor.stop();
        if (signal_observed) {
          summary.cooperative_stop_requested = true;
          if (summary.fatal_error.empty()) {
            summary.status = yolo_defect_cpp::BatchStatus::kCancelled;
          }
          yolo_defect_cpp::validate_batch_summary(summary);
        }
        yolo_defect_cpp::write_batch_summary_json(
            summary, request.summary_path, request.overwrite_existing);
        print_batch_summary(summary, request.summary_path);
        return batch_exit_code(summary.status);
      } else if (options.benchmark) {
        yolo_defect_cpp::BenchmarkRequest request;
        request.image_path = options.image_path;
        request.warmup = options.warmup;
        request.repeat = options.repeat;
        request.command_arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
          request.command_arguments.emplace_back(argv[index]);
        }

        yolo_defect_cpp::BenchmarkRunner runner(contract);
        const yolo_defect_cpp::BenchmarkResult result = runner.run(request);
        std::vector<std::filesystem::path> protected_paths = {
            contract.runtime.declaration_path,
            contract.artifact.declaration_path,
            contract.artifact.model_path,
            std::filesystem::path(options.image_path)};
        if (contract.runtime.tensorrt.has_value()) {
          const yolo_defect_cpp::TensorRtProviderConfig& tensorrt =
              *contract.runtime.tensorrt;
          protected_paths.push_back(tensorrt.engine_cache_path);
          if (tensorrt.native_engine_path.has_value()) {
            protected_paths.push_back(*tensorrt.native_engine_path);
          }
        }
        const std::filesystem::path written_path =
            yolo_defect_cpp::write_benchmark_json(
                result, options.benchmark_json_path,
                options.overwrite_existing, protected_paths);
        print_benchmark_summary(result, written_path);
      } else if (options.profile) {
        yolo_defect_cpp::ProfileRequest request;
        request.image_path = options.image_path;
        request.profile_file_prefix = options.profile_prefix_path;
        request.run_count = options.profile_runs;

        yolo_defect_cpp::ProfileRunner runner(contract);
        const yolo_defect_cpp::ProfileResult result = runner.run(request);
        print_profile_summary(result);
      } else if (options.inspect_model) {
        const yolo_defect_cpp::OnnxRunner runner(contract);
        print_model_metadata_summary(contract, runner.metadata());
      } else if (options.image_path.empty()) {
        print_contract_summary(contract);
      } else {
        const bool output_requested =
            !options.output_json_path.empty() ||
            !options.output_image_path.empty();
        if (output_requested) {
          yolo_defect_cpp::DetectionOutputRequest request;
          if (!options.output_json_path.empty()) {
            request.json_path = options.output_json_path;
          }
          if (!options.output_image_path.empty()) {
            request.image_path = options.output_image_path;
          }
          request.overwrite_existing = options.overwrite_existing;

          yolo_defect_cpp::DetectorPipeline pipeline(contract);
          const yolo_defect_cpp::SingleImagePipelineResult pipeline_result =
              pipeline.run(options.image_path, request);
          print_single_image_summary(
              pipeline_result, options.overwrite_existing);
        } else {
          yolo_defect_cpp::PreprocessResult result =
              yolo_defect_cpp::preprocess_image(options.image_path,
                                                contract.artifact);
          if (!options.raw_output_summary) {
            print_preprocess_summary(options.image_path, contract, result);
          } else {
            const std::vector<std::int64_t> input_shape = {
                1, 3, result.input_height, result.input_width};
            yolo_defect_cpp::InferenceOutput output;
            {
              yolo_defect_cpp::OnnxRunner runner(contract);
              output = runner.run(input_shape, result.tensor_nchw);
            }
            print_raw_output_summary(
                input_shape, result.tensor_nchw, output);
          }
        }
      }
      return 0;
    }

    print_banner();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n"
              << "Run with --help to see the current CLI scope.\n";
    return 1;
  }
}

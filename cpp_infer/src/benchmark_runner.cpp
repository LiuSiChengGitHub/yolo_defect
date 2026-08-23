#include "yolo_defect_cpp/benchmark_runner.h"

#include "image_decoder.h"
#include "yolo_defect_cpp/image_preprocessor.h"
#include "yolo_defect_cpp/onnx_runner.h"
#include "yolo_defect_cpp/postprocessor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <winternl.h>
#endif

#ifndef YOLO_DEFECT_BUILD_TYPE
#define YOLO_DEFECT_BUILD_TYPE "unknown"
#endif
#ifndef YOLO_DEFECT_COMPILER_ID
#define YOLO_DEFECT_COMPILER_ID "unknown"
#endif
#ifndef YOLO_DEFECT_COMPILER_VERSION
#define YOLO_DEFECT_COMPILER_VERSION "unknown"
#endif
#ifndef YOLO_DEFECT_OPENCV_VERSION
#define YOLO_DEFECT_OPENCV_VERSION "unknown"
#endif

namespace yolo_defect_cpp {
namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr std::size_t kMaximumIterations = 1000000;
constexpr const char* kCpuProvider = "CPUExecutionProvider";
constexpr const char* kBaselineModelId =
    "yolov8n_neu_det_final_train_2";
constexpr const char* kBaselineModelSha256 =
    "7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68";
constexpr std::uint64_t kBaselineModelSizeBytes = 12336935;
constexpr const char* kBaselineImageFilename = "crazing_241.jpg";
constexpr std::uint64_t kBaselineImageSizeBytes = 23845;

[[noreturn]] void throw_runner_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Benchmark execution failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

double elapsed_ms(SteadyClock::time_point start,
                  SteadyClock::time_point end,
                  const std::string& object) {
  const double duration =
      std::chrono::duration<double, std::milli>(end - start).count();
  if (!std::isfinite(duration) || duration < 0.0) {
    throw_runner_error(
        object, "a finite non-negative steady-clock duration",
        std::to_string(duration),
        "verify the platform steady_clock before publishing benchmark data");
  }
  return duration;
}

std::uint64_t regular_file_size(const std::filesystem::path& path,
                                 const std::string& object) {
  std::error_code error;
  const bool is_regular = std::filesystem::is_regular_file(path, error);
  if (error || !is_regular) {
    throw_runner_error(
        object, "an existing regular file",
        error ? path.string() + " (" + error.message() + ")"
              : path.string() + " is not a regular file",
        "restore the fixed artifact/sample and check read permissions");
  }
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size == 0 ||
      size > static_cast<std::uintmax_t>(
                 std::numeric_limits<std::uint64_t>::max())) {
    throw_runner_error(
        object, "a non-empty regular file with a representable size",
        error ? path.string() + " (" + error.message() + ")"
              : std::to_string(size) + " bytes",
        "restore the fixed artifact/sample and check read permissions");
  }
  return static_cast<std::uint64_t>(size);
}

std::string evidence_path(const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::path absolute =
      std::filesystem::absolute(path, error).lexically_normal();
  if (error) {
    return path.generic_u8string();
  }
  const std::filesystem::path current =
      std::filesystem::current_path(error);
  if (error) {
    return absolute.generic_u8string();
  }
  const std::filesystem::path relative =
      std::filesystem::relative(absolute, current, error);
  if (!error && !relative.empty()) {
    const auto first = relative.begin();
    if (first == relative.end() || *first != "..") {
      return relative.generic_u8string();
    }
  }
  return absolute.generic_u8string();
}

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
#ifdef _WIN32
  if (gmtime_s(&utc, &now) != 0) {
    throw_runner_error(
        "timestamp_utc", "a UTC timestamp", "gmtime_s failed",
        "verify the Windows system clock");
  }
#else
  if (gmtime_r(&now, &utc) == nullptr) {
    throw_runner_error(
        "timestamp_utc", "a UTC timestamp", "gmtime_r failed",
        "verify the system clock");
  }
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

#ifdef _WIN32
std::string wide_to_utf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int required = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
      static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) {
    return "unavailable";
  }
  std::string converted(static_cast<std::size_t>(required), '\0');
  if (WideCharToMultiByte(
          CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
          static_cast<int>(value.size()), converted.data(), required,
          nullptr, nullptr) != required) {
    return "unavailable";
  }
  return converted;
}

std::string windows_hostname() {
  wchar_t buffer[MAX_COMPUTERNAME_LENGTH + 1]{};
  DWORD length = MAX_COMPUTERNAME_LENGTH + 1;
  if (!GetComputerNameW(buffer, &length)) {
    return "unavailable";
  }
  return wide_to_utf8(std::wstring(buffer, length));
}

std::string windows_processor() {
  const DWORD required =
      GetEnvironmentVariableW(L"PROCESSOR_IDENTIFIER", nullptr, 0);
  if (required == 0) {
    return "unavailable";
  }
  std::vector<wchar_t> buffer(required);
  const DWORD written = GetEnvironmentVariableW(
      L"PROCESSOR_IDENTIFIER", buffer.data(), required);
  if (written == 0 || written >= required) {
    return "unavailable";
  }
  return wide_to_utf8(std::wstring(buffer.data(), written));
}

std::string windows_architecture() {
  SYSTEM_INFO information{};
  GetNativeSystemInfo(&information);
  switch (information.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_AMD64:
      return "x86_64";
    case PROCESSOR_ARCHITECTURE_ARM64:
      return "arm64";
    case PROCESSOR_ARCHITECTURE_INTEL:
      return "x86";
    default:
      return "unknown";
  }
}

std::string windows_version() {
  using RtlGetVersionFunction = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
  const HMODULE module = GetModuleHandleW(L"ntdll.dll");
  if (module == nullptr) {
    return "unavailable";
  }
  const auto function = reinterpret_cast<RtlGetVersionFunction>(
      GetProcAddress(module, "RtlGetVersion"));
  if (function == nullptr) {
    return "unavailable";
  }
  RTL_OSVERSIONINFOW version{};
  version.dwOSVersionInfoSize = sizeof(version);
  if (function(&version) != 0) {
    return "unavailable";
  }
  return std::to_string(version.dwMajorVersion) + "." +
         std::to_string(version.dwMinorVersion) + "." +
         std::to_string(version.dwBuildNumber);
}

BenchmarkMemoryEvidence query_peak_memory() {
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters))) {
    throw_runner_error(
        "memory.peak_working_set", "a successful GetProcessMemoryInfo call",
        "Win32 error " + std::to_string(GetLastError()),
        "verify the Psapi runtime and process-query permissions");
  }
  BenchmarkMemoryEvidence evidence;
  evidence.supported = true;
  evidence.status = "supported";
  evidence.metric = "peak_working_set";
  evidence.bytes = static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
  evidence.mebibytes =
      static_cast<double>(evidence.bytes) / (1024.0 * 1024.0);
  evidence.scope =
      "process lifetime including config/session initialization, warmup, "
      "timed iterations, retained sample vectors, statistics, and benchmark "
      "harness; queried before JSON serialization/write";
  return evidence;
}
#else
BenchmarkMemoryEvidence query_peak_memory() {
  BenchmarkMemoryEvidence evidence;
  evidence.supported = false;
  evidence.status = "unsupported";
  evidence.metric = "peak_rss";
  evidence.scope = "process lifetime through timed iterations";
  evidence.reason =
      "S1-08 currently implements Peak Working Set only on Windows";
  return evidence;
}
#endif

BenchmarkEnvironment collect_environment(const ModelMetadata& metadata) {
  BenchmarkEnvironment environment;
#ifdef _WIN32
  environment.hostname = windows_hostname();
  environment.processor = windows_processor();
  environment.architecture = windows_architecture();
  environment.os_name = "Windows";
  environment.os_version = windows_version();
  const DWORD active_processors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  environment.logical_cpu_count =
      active_processors == 0
          ? static_cast<std::size_t>(std::thread::hardware_concurrency())
          : static_cast<std::size_t>(active_processors);
#else
  environment.hostname = "unavailable";
  environment.processor = "unavailable";
  environment.architecture = "unknown";
  environment.os_name = "non-Windows";
  environment.os_version = "unavailable";
  environment.logical_cpu_count =
      static_cast<std::size_t>(std::thread::hardware_concurrency());
#endif
  if (environment.logical_cpu_count == 0) {
    environment.logical_cpu_count = 1;
  }
  environment.compiler_id = YOLO_DEFECT_COMPILER_ID;
  environment.compiler_version = YOLO_DEFECT_COMPILER_VERSION;
  environment.build_type = YOLO_DEFECT_BUILD_TYPE;
  environment.cxx_standard = 17;
  environment.opencv_version = YOLO_DEFECT_OPENCV_VERSION;
  environment.onnxruntime_version = metadata.ort_version;
  return environment;
}

struct IterationResult {
  double image_decode_ms = 0.0;
  double preprocess_ms = 0.0;
  double session_run_ms = 0.0;
  double postprocess_ms = 0.0;
  double pipeline_ms = 0.0;
  double end_to_end_ms = 0.0;
  int width = 0;
  int height = 0;
  int channels = 0;
  std::size_t detection_count = 0;
  std::vector<Detection> detections;
};

struct IterationIdentity {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<Detection> detections;
};

IterationResult run_iteration(
    const std::filesystem::path& image_path,
    const RuntimeContract& contract,
    OnnxRunner& runner) {
  internal::initialize_image_decoder_logging();
  const auto end_to_end_start = SteadyClock::now();
  internal::DecodedBgrImage decoded =
      internal::decode_normalized_bgr_image(image_path);

  const auto pipeline_start = SteadyClock::now();
  const auto preprocess_start = SteadyClock::now();
  PreprocessResult preprocess =
      preprocess_image(decoded.image, contract.artifact);
  const auto preprocess_end = SteadyClock::now();

  TimedInferenceOutput inference = runner.run_with_session_timing(
      contract.artifact.input.shape, preprocess.tensor_nchw);

  const auto postprocess_start = SteadyClock::now();
  std::vector<Detection> detections = postprocess_yolov8_raw(
      inference.output, contract, preprocess);
  const auto postprocess_end = SteadyClock::now();
  const auto pipeline_end = postprocess_end;
  const auto end_to_end_end = pipeline_end;

  IterationResult result;
  result.image_decode_ms = decoded.imread_ms;
  result.preprocess_ms = elapsed_ms(
      preprocess_start, preprocess_end, "preprocess.duration");
  result.session_run_ms = inference.session_run_ms;
  result.postprocess_ms = elapsed_ms(
      postprocess_start, postprocess_end, "postprocess.duration");
  result.pipeline_ms = elapsed_ms(
      pipeline_start, pipeline_end, "pipeline.duration");
  result.end_to_end_ms = elapsed_ms(
      end_to_end_start, end_to_end_end, "end_to_end.duration");
  result.width = preprocess.original_width;
  result.height = preprocess.original_height;
  result.channels = preprocess.original_channels;
  result.detection_count = detections.size();
  result.detections = std::move(detections);
  return result;
}

void validate_iteration_identity(
    const IterationResult& iteration,
    std::optional<IterationIdentity>& expected_identity,
    std::size_t iteration_index,
    const std::string& phase) {
  if (!expected_identity.has_value()) {
    expected_identity = IterationIdentity{
        iteration.width, iteration.height, iteration.channels,
        iteration.detections};
    return;
  }
  const IterationIdentity& expected = *expected_identity;
  if (iteration.detection_count != expected.detections.size()) {
    throw_runner_error(
        phase + "[" + std::to_string(iteration_index) +
            "].detection_count",
        std::to_string(expected.detections.size()),
        std::to_string(iteration.detection_count),
        "stop the benchmark and investigate correctness/determinism before "
        "publishing performance evidence");
  }
  if (iteration.width != expected.width ||
      iteration.height != expected.height ||
      iteration.channels != expected.channels) {
    throw_runner_error(
        phase + "[" + std::to_string(iteration_index) + "].image_shape",
        std::to_string(expected.width) + "x" +
            std::to_string(expected.height) + "x" +
            std::to_string(expected.channels),
        std::to_string(iteration.width) + "x" +
            std::to_string(iteration.height) + "x" +
            std::to_string(iteration.channels),
        "keep the fixed sample unchanged during every repeat");
  }

  constexpr float kDetectionStabilityTolerance = 1.0e-6F;
  const auto validate_float = [&](float actual, float expected_value,
                                  std::size_t detection_index,
                                  const std::string& field) {
    if (!std::isfinite(actual) ||
        std::abs(actual - expected_value) > kDetectionStabilityTolerance) {
      throw_runner_error(
          phase + "[" + std::to_string(iteration_index) + "].detections[" +
              std::to_string(detection_index) + "]." + field,
          std::to_string(expected_value) + " +/- 1e-6",
          std::to_string(actual),
          "stop the benchmark and investigate inference/postprocess "
          "determinism before publishing performance evidence");
    }
  };
  for (std::size_t index = 0; index < iteration.detections.size(); ++index) {
    const Detection& actual = iteration.detections[index];
    const Detection& expected_detection = expected.detections[index];
    if (actual.class_id != expected_detection.class_id ||
        actual.class_name != expected_detection.class_name) {
      throw_runner_error(
          phase + "[" + std::to_string(iteration_index) + "].detections[" +
              std::to_string(index) + "].class",
          std::to_string(expected_detection.class_id) + ":" +
              expected_detection.class_name,
          std::to_string(actual.class_id) + ":" + actual.class_name,
          "stop the benchmark and investigate deterministic decode/NMS "
          "ordering before publishing performance evidence");
    }
    validate_float(actual.confidence, expected_detection.confidence, index,
                   "confidence");
    validate_float(actual.bbox_xyxy.x1, expected_detection.bbox_xyxy.x1,
                   index, "bbox_xyxy.x1");
    validate_float(actual.bbox_xyxy.y1, expected_detection.bbox_xyxy.y1,
                   index, "bbox_xyxy.y1");
    validate_float(actual.bbox_xyxy.x2, expected_detection.bbox_xyxy.x2,
                   index, "bbox_xyxy.x2");
    validate_float(actual.bbox_xyxy.y2, expected_detection.bbox_xyxy.y2,
                   index, "bbox_xyxy.y2");
  }
}

}  // namespace

class BenchmarkRunner::Impl {
 public:
  explicit Impl(RuntimeContract contract)
      : contract_(std::move(contract)), runner_(contract_) {
    if (std::string(YOLO_DEFECT_BUILD_TYPE) != "Release") {
      throw_runner_error(
          "build_type", "Release", YOLO_DEFECT_BUILD_TYPE,
          "use a clean out-of-tree configure with "
          "-DCMAKE_BUILD_TYPE=Release");
    }
    if (contract_.runtime.provider != ExecutionProvider::kCpu ||
        runner_.metadata().session_provider != kCpuProvider) {
      throw_runner_error(
          "provider", "requested cpu and actual CPUExecutionProvider",
          "requested=" + to_string(contract_.runtime.provider) +
              ", actual=" + runner_.metadata().session_provider,
          "restore the fixed CPU RuntimeConfig/session before benchmarking");
    }
    if (contract_.artifact.input.shape.size() != 4 ||
        contract_.artifact.input.shape[0] != 1) {
      throw_runner_error(
          "artifact.input.shape", "batch=1 static NCHW",
          format_shape(contract_.artifact.input.shape),
          "use the validated single-image baseline artifact");
    }
    const std::vector<std::int64_t> expected_input = {1, 3, 800, 800};
    if (contract_.artifact.model_id != kBaselineModelId ||
        contract_.artifact.model_sha256 != kBaselineModelSha256 ||
        contract_.artifact.opset != 17 ||
        contract_.artifact.input.name != "images" ||
        contract_.artifact.input.shape != expected_input ||
        to_string(contract_.artifact.input.dtype) != "float32" ||
        to_string(contract_.artifact.input.layout) != "nchw") {
      throw_runner_error(
          "baseline.model_contract",
          "model_id=" + std::string(kBaselineModelId) +
              ", SHA-256=" + kBaselineModelSha256 +
              ", opset=17, images float32 [1,3,800,800] NCHW",
          "model_id=" + contract_.artifact.model_id +
              ", SHA-256=" + contract_.artifact.model_sha256 +
              ", opset=" + std::to_string(contract_.artifact.opset) +
              ", input=" + format_shape(contract_.artifact.input.shape),
          "use cpp_infer/configs/default_config.txt and the frozen S1-08 "
          "YOLOv8/NEU-DET artifact");
    }
    const std::uint64_t model_size = regular_file_size(
        contract_.artifact.model_path, "baseline.model_size");
    if (model_size != kBaselineModelSizeBytes) {
      throw_runner_error(
          "baseline.model_size", std::to_string(kBaselineModelSizeBytes),
          std::to_string(model_size),
          "restore models/best.onnx and rerun the S1-07 SHA/correctness gate");
    }
    if (std::abs(contract_.runtime.score_threshold - 0.25) > 1.0e-12 ||
        std::abs(contract_.runtime.nms_threshold - 0.45) > 1.0e-12 ||
        to_string(contract_.artifact.nms_mode) != "class_agnostic") {
      throw_runner_error(
          "baseline.postprocess_contract",
          "score_threshold=0.25, nms_threshold=0.45, "
          "nms_mode=class_agnostic",
          "score_threshold=" +
              std::to_string(contract_.runtime.score_threshold) +
              ", nms_threshold=" +
              std::to_string(contract_.runtime.nms_threshold) +
              ", nms_mode=" + to_string(contract_.artifact.nms_mode),
          "restore the frozen default RuntimeConfig/artifact before "
          "benchmarking");
    }
  }

  BenchmarkResult run(const BenchmarkRequest& request) {
    if (request.warmup > kMaximumIterations) {
      throw_runner_error(
          "warmup", "an integer in [0,1000000]",
          std::to_string(request.warmup),
          "use --warmup 10 for the formal baseline");
    }
    if (request.repeat == 0 || request.repeat > kMaximumIterations) {
      throw_runner_error(
          "repeat", "an integer in [1,1000000]",
          std::to_string(request.repeat),
          "use --repeat 100 for the formal baseline");
    }
    if (request.command_arguments.empty()) {
      throw_runner_error(
          "command", "the original CLI argument vector", "empty",
          "pass argv from the thin CLI into BenchmarkRequest");
    }

    const std::filesystem::path image_path =
        internal::normalize_image_file(request.image_path);
    const std::uint64_t image_size =
        regular_file_size(image_path, "sample.file_size_bytes");
    if (image_path.filename() != kBaselineImageFilename ||
        image_size != kBaselineImageSizeBytes) {
      throw_runner_error(
          "baseline.sample",
          std::string(kBaselineImageFilename) + " with " +
              std::to_string(kBaselineImageSizeBytes) + " bytes",
          image_path.filename().string() + " with " +
              std::to_string(image_size) + " bytes",
          "use data/images/val/crazing_241.jpg and rerun the S1-07 "
          "manifest SHA/correctness gate");
    }
    const std::uint64_t model_size = regular_file_size(
        contract_.artifact.model_path, "model.file_size_bytes");
    const std::string started_at = utc_timestamp();

    std::optional<IterationIdentity> expected_identity;
    for (std::size_t index = 0; index < request.warmup; ++index) {
      const IterationResult warmup =
          run_iteration(image_path, contract_, runner_);
      validate_iteration_identity(
          warmup, expected_identity, index, "warmup");
    }

    std::vector<double> image_decode_samples;
    std::vector<double> preprocess_samples;
    std::vector<double> session_run_samples;
    std::vector<double> postprocess_samples;
    std::vector<double> pipeline_samples;
    std::vector<double> end_to_end_samples;
    image_decode_samples.reserve(request.repeat);
    preprocess_samples.reserve(request.repeat);
    session_run_samples.reserve(request.repeat);
    postprocess_samples.reserve(request.repeat);
    pipeline_samples.reserve(request.repeat);
    end_to_end_samples.reserve(request.repeat);

    for (std::size_t index = 0; index < request.repeat; ++index) {
      const IterationResult measured =
          run_iteration(image_path, contract_, runner_);
      validate_iteration_identity(
          measured, expected_identity, index, "repeat");
      image_decode_samples.push_back(measured.image_decode_ms);
      preprocess_samples.push_back(measured.preprocess_ms);
      session_run_samples.push_back(measured.session_run_ms);
      postprocess_samples.push_back(measured.postprocess_ms);
      pipeline_samples.push_back(measured.pipeline_ms);
      end_to_end_samples.push_back(measured.end_to_end_ms);
    }

    if (expected_identity->width != 200 ||
        expected_identity->height != 200 ||
        expected_identity->channels != 3 ||
        expected_identity->detections.size() != 3) {
      throw_runner_error(
          "baseline.sample_result", "decoded shape 200x200x3 and 3 detections",
          "shape=" + std::to_string(expected_identity->width) + "x" +
              std::to_string(expected_identity->height) + "x" +
              std::to_string(expected_identity->channels) +
              ", detections=" +
              std::to_string(expected_identity->detections.size()),
          "stop publication and rerun S1-07 consistency before diagnosing "
          "the model/config/sample");
    }

    BenchmarkResult result;
    result.schema_version = 1;
    result.evidence_type = "cpp_ort_single_image_release_benchmark";
    result.timestamp_utc = started_at;
    result.command_arguments = request.command_arguments;
    result.batch_size = 1;
    result.sample_count = 1;
    result.warmup = request.warmup;
    result.repeat = request.repeat;
    result.environment = collect_environment(runner_.metadata());

    const ModelMetadata& metadata = runner_.metadata();
    result.runtime.requested_provider = to_string(contract_.runtime.provider);
    result.runtime.actual_provider = metadata.session_provider;
    result.runtime.provider_evidence = metadata.provider_evidence;
    result.runtime.execution_mode = metadata.execution_mode;
    result.runtime.intra_op_num_threads = metadata.intra_op_num_threads;
    result.runtime.inter_op_num_threads = metadata.inter_op_num_threads;
    result.runtime.graph_optimization_level =
        metadata.graph_optimization_level;

    result.model.model_id = contract_.artifact.model_id;
    result.model.model_family = to_string(contract_.artifact.model_family);
    result.model.model_path = evidence_path(contract_.artifact.model_path);
    result.model.declared_sha256 = contract_.artifact.model_sha256;
    result.model.file_size_bytes = model_size;
    result.model.opset = contract_.artifact.opset;
    result.model.input_name = contract_.artifact.input.name;
    result.model.input_shape = contract_.artifact.input.shape;
    result.model.input_dtype = to_string(contract_.artifact.input.dtype);
    result.model.input_layout = to_string(contract_.artifact.input.layout);

    result.sample.image_path = evidence_path(image_path);
    result.sample.file_size_bytes = image_size;
    result.sample.width = expected_identity->width;
    result.sample.height = expected_identity->height;
    result.sample.channels = expected_identity->channels;
    result.sample.sample_count = 1;
    result.sample.detection_count = expected_identity->detections.size();

    result.score_threshold = contract_.runtime.score_threshold;
    result.nms_threshold = contract_.runtime.nms_threshold;
    result.nms_mode = to_string(contract_.artifact.nms_mode);

    result.latency.image_decode =
        calculate_latency_statistics(image_decode_samples);
    result.latency.preprocess =
        calculate_latency_statistics(preprocess_samples);
    result.latency.session_run =
        calculate_latency_statistics(session_run_samples);
    result.latency.postprocess =
        calculate_latency_statistics(postprocess_samples);
    result.latency.pipeline =
        calculate_latency_statistics(pipeline_samples);
    result.latency.end_to_end =
        calculate_latency_statistics(end_to_end_samples);
    result.latency.pipeline_throughput_images_per_second =
        calculate_throughput_images_per_second(result.latency.pipeline);
    result.latency.end_to_end_throughput_images_per_second =
        calculate_throughput_images_per_second(result.latency.end_to_end);
    result.memory = query_peak_memory();

    result.timing_exclusions = {
        "RuntimeConfig/ModelArtifactSpec loading and validation",
        "Ort::Env/Session/model initialization and metadata validation",
        "initial image path validation and file-size queries",
        "statistics calculation and Peak Working Set query",
        "benchmark JSON serialization and filesystem write",
        "visualization/GUI rendering (not executed)"};
    result.limitations = {
        "One 200x200 validation image, batch=1, one Windows CPU machine; "
        "results do not represent the full dataset or other hardware.",
        "Repeated imread uses a warmed operating-system file cache and is "
        "not cold-disk latency.",
        "No CPU affinity, elevated process priority, or idle-system lock was "
        "applied; concurrent system load can change latency.",
        "session_run measures only Ort::Session::Run; pipeline additionally "
        "includes input validation/tensor construction and output "
        "validation/copy.",
        "Peak Working Set is the process-lifetime peak including session "
        "initialization, warmup, measured iterations, retained samples, "
        "statistics, and harness state; it is not per-stage or incremental "
        "inference memory.",
        "Actual provider is session-level evidence, not per-node placement "
        "profiling.",
        "Historical Python ORT 24.4/72.1 FPS used a different protocol and "
        "must not be compared unconditionally with this C++ result.",
        "The benchmark records the declared model SHA and fixed file sizes; "
        "the required immediately preceding S1-07 gate and JSON validator "
        "recompute the actual model/sample SHA-256 values."};

    validate_benchmark_result(result);
    return result;
  }

 private:
  RuntimeContract contract_;
  OnnxRunner runner_;
};

BenchmarkRunner::BenchmarkRunner(RuntimeContract contract)
    : impl_(std::make_unique<Impl>(std::move(contract))) {}

BenchmarkRunner::~BenchmarkRunner() = default;
BenchmarkRunner::BenchmarkRunner(BenchmarkRunner&&) noexcept = default;
BenchmarkRunner& BenchmarkRunner::operator=(BenchmarkRunner&&) noexcept =
    default;

BenchmarkResult BenchmarkRunner::run(const BenchmarkRequest& request) {
  if (!impl_) {
    throw_runner_error(
        "BenchmarkRunner", "a live runner instance", "moved-from instance",
        "invoke run only on the object that owns the ORT session");
  }
  return impl_->run(request);
}

}  // namespace yolo_defect_cpp

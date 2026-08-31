#include "yolo_defect_cpp/batch_runner.h"

#include "batch_executor.h"
#include "batch_path_safety.h"
#include "bounded_queue.h"
#include "platform_info.h"
#include "yolo_defect_cpp/detector_pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
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
#ifndef YOLO_DEFECT_TARGET_ARCHITECTURE
#define YOLO_DEFECT_TARGET_ARCHITECTURE "unknown"
#endif

namespace yolo_defect_cpp {
namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr std::size_t kMaximumWorkers = 64;
constexpr std::size_t kMaximumQueueCapacity = 4096;
constexpr const char* kDirectoryOrdering =
    "recursive UTF-8 generic relative-path lexical order; supported "
    "regular files only; symlinks not followed";
constexpr const char* kManifestOrdering =
    "UTF-8 path-list declaration order";

[[noreturn]] void throw_batch_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Batch execution failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

std::string display_path(const std::filesystem::path& path) {
  try {
    return path.generic_u8string();
  } catch (const std::exception&) {
    return "<path cannot be converted to UTF-8>";
  }
}

std::filesystem::path absolute_normalized(
    const std::filesystem::path& path, const std::string& object) {
  if (path.empty()) {
    throw_batch_error(
        object, "a non-empty path", "empty",
        "provide the matching batch CLI path argument");
  }
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    throw_batch_error(
        object, "a path resolvable from the working directory",
        display_path(path) + " (" + error.message() + ")",
        "correct the path or working directory");
  }
  return absolute.lexically_normal();
}

std::filesystem::path canonical_existing(
    const std::filesystem::path& path, const std::string& object) {
  const std::filesystem::path absolute = absolute_normalized(path, object);
  std::error_code error;
  const std::filesystem::path canonical =
      std::filesystem::canonical(absolute, error);
  if (error) {
    throw_batch_error(
        object, "an existing accessible path",
        display_path(absolute) + " (" + error.message() + ")",
        "correct the path and check filesystem permissions");
  }
  return canonical;
}

std::filesystem::path weakly_canonical_output(
    const std::filesystem::path& path, const std::string& object) {
  const std::filesystem::path absolute = absolute_normalized(path, object);
  std::error_code error;
  const std::filesystem::path normalized =
      std::filesystem::weakly_canonical(absolute, error);
  if (error) {
    throw_batch_error(
        object, "an output path with inspectable existing ancestors",
        display_path(absolute) + " (" + error.message() + ")",
        "correct the output path and check ancestor permissions");
  }
  return normalized.lexically_normal();
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   if (character >= 'A' && character <= 'Z') {
                     return static_cast<char>(character - 'A' + 'a');
                   }
                   return static_cast<char>(character);
                 });
  return value;
}

std::string normalized_architecture(std::string value) {
  value = lowercase_ascii(std::move(value));
  if (value == "amd64" || value == "x64" || value == "x86-64") {
    return "x86_64";
  }
  if (value == "arm64" || value == "armv8" || value == "armv8-a") {
    return "aarch64";
  }
  return value;
}

std::filesystem::path normalized_location_path(
    const std::filesystem::path& path) {
  std::filesystem::path normalized = path.lexically_normal();
#ifdef _WIN32
  normalized.make_preferred();
#endif
  return normalized;
}

struct PathLocationLess {
  bool operator()(const std::filesystem::path& lhs,
                  const std::filesystem::path& rhs) const {
    const std::filesystem::path normalized_lhs =
        normalized_location_path(lhs);
    const std::filesystem::path normalized_rhs =
        normalized_location_path(rhs);
#ifdef _WIN32
    const int comparison = CompareStringOrdinal(
        normalized_lhs.native().c_str(), -1,
        normalized_rhs.native().c_str(), -1, TRUE);
    if (comparison == CSTR_LESS_THAN) {
      return true;
    }
    if (comparison == CSTR_EQUAL || comparison == CSTR_GREATER_THAN) {
      return false;
    }
    // CompareStringOrdinal can fail only for invalid arguments. Paths are
    // NUL-terminated std::wstring values, but retain a strict fallback so a
    // set comparator never collapses unrelated paths on an unexpected OS
    // failure.
    return normalized_lhs.native() < normalized_rhs.native();
#else
    return normalized_lhs.native() < normalized_rhs.native();
#endif
  }
};

bool path_text_equal(const std::filesystem::path& lhs,
                     const std::filesystem::path& rhs) {
  const PathLocationLess less;
  return !less(lhs, rhs) && !less(rhs, lhs);
}

bool paths_refer_to_same_location(const std::filesystem::path& lhs,
                                  const std::filesystem::path& rhs) {
  if (path_text_equal(lhs, rhs)) {
    return true;
  }
  std::error_code lhs_error;
  std::error_code rhs_error;
  const bool lhs_exists = std::filesystem::exists(lhs, lhs_error);
  const bool rhs_exists = std::filesystem::exists(rhs, rhs_error);
  if (lhs_error || rhs_error || !lhs_exists || !rhs_exists) {
    return false;
  }
  std::error_code equivalent_error;
  const bool equivalent =
      std::filesystem::equivalent(lhs, rhs, equivalent_error);
  return !equivalent_error && equivalent;
}

bool is_same_or_descendant(const std::filesystem::path& candidate,
                           const std::filesystem::path& root) {
  const std::filesystem::path normalized_candidate =
      normalized_location_path(candidate);
  const std::filesystem::path normalized_root =
      normalized_location_path(root);
  auto candidate_component = normalized_candidate.begin();
  const auto candidate_end = normalized_candidate.end();
  for (auto root_component = normalized_root.begin();
       root_component != normalized_root.end(); ++root_component) {
    if (candidate_component == candidate_end ||
        !path_text_equal(*candidate_component, *root_component)) {
      return false;
    }
    ++candidate_component;
  }
  return true;
}

bool windows_reparse_point(const std::filesystem::path& path,
                           const std::string& object) {
  std::error_code error;
  const bool reparse =
      internal::batch_path_is_reparse_point(path, error);
  if (error) {
    throw_batch_error(
        object, "queryable Windows file attributes",
        display_path(path) + " (" + error.message() + ")",
        "check path permissions and filesystem state");
  }
  return reparse;
}

void validate_directory_target_object(
    const std::filesystem::path& path, const std::string& object) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (error && status.type() != std::filesystem::file_type::not_found) {
    throw_batch_error(
        object, "an inspectable missing path or real directory",
        display_path(path) + " (" + error.message() + ")",
        "check path permissions and filesystem state");
  }
  if (!std::filesystem::exists(status)) {
    return;
  }
  if (std::filesystem::is_symlink(status) ||
      windows_reparse_point(path, object) ||
      !std::filesystem::is_directory(status)) {
    throw_batch_error(
        object, "a missing path or real directory without symlink/reparse "
                "indirection",
        display_path(path) +
            " is a symlink, reparse point, file, or special object",
        "remove the indirection/object and use a dedicated real output "
        "directory");
  }
}

void validate_summary_target(const std::filesystem::path& summary_path,
                             bool overwrite_existing) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(summary_path, error);
  if (error && status.type() != std::filesystem::file_type::not_found) {
    throw_batch_error(
        "output.batch_summary_path", "an inspectable file target",
        display_path(summary_path) + " (" + error.message() + ")",
        "check output path permissions");
  }
  if (!std::filesystem::exists(status)) {
    return;
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw_batch_error(
        "output.batch_summary_path", "a missing path or regular file",
        display_path(summary_path) + " is a directory, symlink, or special "
            "filesystem object",
        "choose a normal JSON output filename");
  }
  if (!overwrite_existing) {
    throw_batch_error(
        "output.batch_summary_path", "a path that does not exist",
        display_path(summary_path) + " already exists",
        "choose a new path or pass --overwrite explicitly");
  }
}

std::string format_sequence(std::size_t index) {
  std::ostringstream output;
  output << std::setw(6) << std::setfill('0') << index;
  return output.str();
}

std::filesystem::path planned_json_path(
    const std::filesystem::path& item_directory, std::size_t index) {
  return item_directory /
         (format_sequence(index) + ".detections.json");
}

std::filesystem::path planned_image_path(
    const std::filesystem::path& item_directory, std::size_t index) {
  return item_directory /
         (format_sequence(index) + ".visualized.png");
}

struct NormalizedRequestPaths {
  std::filesystem::path input;
  std::filesystem::path output_directory;
  std::filesystem::path item_directory;
  std::filesystem::path summary;
};

NormalizedRequestPaths validate_request_and_outputs(
    const BatchRequest& request, const RuntimeContract& contract,
    const std::vector<BatchTask>& tasks) {
  if (request.requested_workers == 0 ||
      request.requested_workers > kMaximumWorkers) {
    throw_batch_error(
        "workers", "an integer in [1,64]",
        std::to_string(request.requested_workers),
        "choose a bounded worker count such as 1 or 4");
  }
  if (request.queue_capacity == 0 ||
      request.queue_capacity > kMaximumQueueCapacity) {
    throw_batch_error(
        "queue_capacity", "an integer in [1,4096]",
        std::to_string(request.queue_capacity),
        "choose a finite queue capacity such as 2 * workers");
  }
  if (request.command_arguments.empty()) {
    throw_batch_error(
        "command_arguments", "the original non-empty argv vector", "empty",
        "copy argv into BatchRequest before invoking run");
  }
  for (std::size_t index = 0; index < request.command_arguments.size();
       ++index) {
    if (request.command_arguments[index].empty()) {
      throw_batch_error(
          "command_arguments[" + std::to_string(index) + "]",
          "a non-empty argument", "empty",
          "retain each original command argument as text");
    }
  }

  NormalizedRequestPaths paths;
  paths.input = canonical_existing(request.input_path, "input.source_path");
  const std::filesystem::path declared_output_directory =
      absolute_normalized(request.output_directory, "output.directory");
  const std::filesystem::path declared_item_directory =
      (declared_output_directory / "items").lexically_normal();
  validate_directory_target_object(declared_output_directory,
                                   "output.directory");
  validate_directory_target_object(declared_item_directory,
                                   "output.item_directory");
  paths.output_directory = weakly_canonical_output(
      declared_output_directory, "output.directory");
  paths.item_directory =
      weakly_canonical_output(declared_item_directory,
                              "output.item_directory");
  paths.summary = weakly_canonical_output(
      request.summary_path, "output.batch_summary_path");

  // Re-check declared objects after canonicalization to narrow the window in
  // which another process could replace either directory with indirection.
  validate_directory_target_object(declared_output_directory,
                                   "output.directory");
  validate_directory_target_object(declared_item_directory,
                                   "output.item_directory");
  if (!is_same_or_descendant(paths.item_directory,
                             paths.output_directory)) {
    throw_batch_error(
        "output.item_directory",
        "a canonical path contained by the canonical output directory",
        display_path(paths.item_directory) + " escapes " +
            display_path(paths.output_directory),
        "remove symlink/reparse indirection and use output/items beneath "
        "the dedicated output root");
  }
  if (request.input_kind == BatchInputKind::kDirectory &&
      is_same_or_descendant(paths.output_directory, paths.input)) {
    throw_batch_error(
        "output.directory", "a path outside the input directory tree",
        display_path(paths.output_directory) + " lies under " +
            display_path(paths.input),
        "choose a sibling output directory so recursive discovery cannot "
        "ingest generated files");
  }
  validate_summary_target(paths.summary, request.overwrite_existing);

  std::set<std::filesystem::path, PathLocationLess> protected_locations;
  for (const BatchTask& task : tasks) {
    protected_locations.insert(normalized_location_path(task.source_path));
  }
  protected_locations.insert(normalized_location_path(canonical_existing(
      contract.runtime.declaration_path, "runtime.config_path")));
  protected_locations.insert(normalized_location_path(canonical_existing(
      contract.artifact.declaration_path, "model.artifact_path")));
  protected_locations.insert(normalized_location_path(canonical_existing(
      contract.artifact.model_path, "model.model_path")));
  protected_locations.insert(normalized_location_path(paths.input));

  std::set<std::filesystem::path, PathLocationLess> planned_locations;
  const auto check_planned = [&](const std::filesystem::path& target,
                                 const std::string& object) {
    const std::filesystem::path normalized_target =
        normalized_location_path(target);
    std::error_code target_error;
    const bool target_exists =
        std::filesystem::exists(normalized_target, target_error);
    if (target_error) {
      throw_batch_error(
          object, "an output path with queryable filesystem identity",
          display_path(normalized_target) + " (" +
              target_error.message() + ")",
          "check output path permissions and filesystem state");
    }
    bool protected_overlap =
        protected_locations.find(normalized_target) !=
        protected_locations.end();
    if (!protected_overlap && target_exists) {
      for (const std::filesystem::path& protected_path :
           protected_locations) {
        if (paths_refer_to_same_location(normalized_target,
                                         protected_path)) {
          protected_overlap = true;
          break;
        }
      }
    }
    if (protected_overlap) {
      throw_batch_error(
          object, "a path distinct from every source/config/artifact/model "
                  "input",
          display_path(target) + " overlaps a protected input",
          "choose a dedicated batch output directory and summary path");
    }
    bool duplicate_output =
        planned_locations.find(normalized_target) != planned_locations.end();
    if (!duplicate_output && target_exists) {
      for (const std::filesystem::path& planned_path : planned_locations) {
        if (paths_refer_to_same_location(normalized_target, planned_path)) {
          duplicate_output = true;
          break;
        }
      }
    }
    if (duplicate_output ||
        !planned_locations.insert(normalized_target).second) {
      throw_batch_error(
          object, "a unique planned output path", display_path(target),
          "choose distinct output and summary paths");
    }
  };
  check_planned(paths.summary, "output.batch_summary_path");
  for (const BatchTask& task : tasks) {
    check_planned(planned_json_path(paths.item_directory,
                                    task.sequence_index),
                  "output.items.json_path");
    if (request.output_images) {
      check_planned(planned_image_path(paths.item_directory,
                                       task.sequence_index),
                    "output.items.image_path");
    }
  }
  return paths;
}

BatchEnvironment collect_environment() {
  BatchEnvironment environment;
  const internal::PlatformInfo platform = internal::collect_platform_info();
  environment.hostname = platform.hostname;
  environment.processor = platform.processor;
  environment.logical_cpu_count = platform.logical_cpu_count;
  environment.os_name = platform.os_name;
  environment.os_version = platform.os_version;
  environment.target_architecture = YOLO_DEFECT_TARGET_ARCHITECTURE;
  const char* declared_kernel_architecture =
      std::getenv("YOLO_DEFECT_RUNTIME_KERNEL_ARCHITECTURE");
  if (declared_kernel_architecture != nullptr &&
      declared_kernel_architecture[0] != '\0') {
    // qemu-user intentionally virtualizes uname(2) for the guest process. A
    // host-side acceptance wrapper may therefore inject the real host-kernel
    // architecture alongside an explicit emulation execution context.
    environment.runtime_kernel_architecture = declared_kernel_architecture;
  } else {
    environment.runtime_kernel_architecture = platform.architecture;
  }
  environment.compiler_id = YOLO_DEFECT_COMPILER_ID;
  environment.compiler_version = YOLO_DEFECT_COMPILER_VERSION;
  environment.build_type = YOLO_DEFECT_BUILD_TYPE;
  environment.cxx_standard = 17;
  environment.opencv_version = YOLO_DEFECT_OPENCV_VERSION;

  const char* declared_context =
      std::getenv("YOLO_DEFECT_EXECUTION_CONTEXT");
  if (declared_context != nullptr && declared_context[0] != '\0') {
    environment.execution_context = declared_context;
  } else if (normalized_architecture(environment.target_architecture) !=
                 normalized_architecture(
                     environment.runtime_kernel_architecture) &&
             normalized_architecture(environment.target_architecture) !=
                 "unknown" &&
             normalized_architecture(
                 environment.runtime_kernel_architecture) != "unknown") {
    environment.execution_context = "emulated_or_cross_arch_inferred";
  } else {
    environment.execution_context = "native_or_unknown";
  }
  return environment;
}

bool performance_is_publishable(const BatchEnvironment& environment) {
  const std::string context = lowercase_ascii(environment.execution_context);
  if (context.find("qemu") != std::string::npos ||
      context.find("emulat") != std::string::npos) {
    return false;
  }
  const std::string target =
      normalized_architecture(environment.target_architecture);
  const std::string runtime =
      normalized_architecture(environment.runtime_kernel_architecture);
  return target == "unknown" || runtime == "unknown" || target == runtime;
}

BatchMemoryEvidence collect_memory(const BatchEnvironment& environment) {
  BatchMemoryEvidence result;
  try {
    const BenchmarkMemoryEvidence evidence =
        internal::query_peak_process_memory();
    result.supported = evidence.supported;
    result.status = evidence.status;
    result.metric = evidence.metric;
    result.bytes = evidence.bytes;
    result.mebibytes = evidence.mebibytes;
    result.scope =
        "process lifetime including task discovery, worker session "
        "construction, bounded-queue execution, per-image output, retained "
        "results, joins, and the memory query; excludes BatchSummary JSON "
        "serialization/write";
    result.reason = evidence.reason;
  } catch (const std::exception& error) {
    result.supported = false;
    result.status = "unavailable";
    result.metric = "peak_process_memory";
    result.scope = "process lifetime before BatchSummary serialization";
    result.reason = error.what();
  }
  result.publishable =
      result.supported && performance_is_publishable(environment);
  if (!result.publishable) {
    if (!result.reason.empty()) {
      result.reason += "; ";
    }
    result.reason +=
        "emulated or cross-architecture execution is functional evidence, "
        "not publishable native performance evidence";
  }
  return result;
}

bool metadata_matches(const ModelMetadata& lhs,
                      const ModelMetadata& rhs) {
  return lhs.ort_version == rhs.ort_version &&
         lhs.session_provider == rhs.session_provider &&
         lhs.provider_evidence == rhs.provider_evidence &&
         lhs.intra_op_num_threads == rhs.intra_op_num_threads &&
         lhs.inter_op_num_threads == rhs.inter_op_num_threads &&
         lhs.execution_mode == rhs.execution_mode &&
         lhs.graph_optimization_level == rhs.graph_optimization_level &&
         lhs.inputs.size() == rhs.inputs.size() &&
         lhs.outputs.size() == rhs.outputs.size();
}

double elapsed_ms(SteadyClock::time_point begin,
                  SteadyClock::time_point end) {
  const double value =
      std::chrono::duration<double, std::milli>(end - begin).count();
  if (!std::isfinite(value) || value < 0.0) {
    throw_batch_error(
        "timing", "a finite non-negative steady-clock duration",
        std::to_string(value),
        "verify the platform steady_clock implementation");
  }
  return value;
}

void initialize_item_results(const std::vector<BatchTask>& tasks,
                             std::vector<BatchItemResult>& results) {
  results.resize(tasks.size());
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    results[index].sequence_index = index;
    results[index].source_path = tasks[index].source_path;
    results[index].status = BatchItemStatus::kCancelled;
    results[index].error = "cancelled before start";
  }
}

void populate_base_summary(BatchSummary& summary,
                           const BatchRequest& request,
                           const RuntimeContract& contract,
                           const NormalizedRequestPaths& paths,
                           const std::vector<BatchTask>& tasks) {
  summary.timestamp_utc = internal::utc_timestamp();
  summary.command_arguments = request.command_arguments;
  summary.environment = collect_environment();

  summary.runtime.config_path = canonical_existing(
      contract.runtime.declaration_path, "runtime.config_path");
  summary.runtime.requested_provider =
      to_string(contract.runtime.provider);
  summary.runtime.score_threshold = contract.runtime.score_threshold;
  summary.runtime.nms_threshold = contract.runtime.nms_threshold;
  summary.runtime.nms_mode = to_string(contract.artifact.nms_mode);
  summary.runtime.requested_workers = request.requested_workers;
  summary.runtime.effective_workers =
      std::min(request.requested_workers, tasks.size());

  summary.model.model_id = contract.artifact.model_id;
  summary.model.model_family = to_string(contract.artifact.model_family);
  summary.model.model_path = canonical_existing(
      contract.artifact.model_path, "model.model_path");
  summary.model.declared_sha256 = contract.artifact.model_sha256;
  summary.model.opset = contract.artifact.opset;
  summary.model.input_name = contract.artifact.input.name;
  summary.model.input_shape = contract.artifact.input.shape;
  summary.model.input_dtype = to_string(contract.artifact.input.dtype);
  summary.model.input_layout = to_string(contract.artifact.input.layout);

  summary.input.kind = request.input_kind;
  summary.input.source_path = paths.input;
  summary.input.ordering = request.input_kind == BatchInputKind::kDirectory
                               ? kDirectoryOrdering
                               : kManifestOrdering;

  summary.output.directory = paths.output_directory;
  summary.output.batch_summary_path = paths.summary;
  summary.output.item_directory = paths.item_directory;
  summary.output.json_outputs = true;
  summary.output.image_outputs = request.output_images;
  summary.output.overwrite_existing = request.overwrite_existing;

  summary.counts.discovered = tasks.size();
  summary.queue.capacity = request.queue_capacity;
  summary.timing.includes = {
      "bounded-queue producer waits and task hand-off",
      "image decode, preprocess, one batch=1 Ort::Session::Run, "
      "postprocess, and per-image output writes",
      "worker drain/cancellation and thread joins"};
  summary.timing.excludes = {
      "directory/manifest discovery and validation",
      "RuntimeConfig/ModelArtifactSpec loading",
      "worker DetectorPipeline and Ort::Session construction",
      "BatchSummary validation, serialization, and filesystem write"};
  summary.limitations = {
      "Concurrency is image-level batch=1 work; this is not true tensor "
      "batching.",
      "Each worker owns one DetectorPipeline and CPU Ort::Session with "
      "sequential execution and intra/inter-op thread counts fixed at 1/1.",
      "No video, service, GPU concurrency, or lock-free queue is included.",
      "Process peak memory is a lifetime high-water mark, not incremental "
      "per-worker or per-stage memory.",
      "QEMU user-mode execution, when declared or inferred, proves only "
      "AArch64 functional portability and is not performance evidence."};

  initialize_item_results(tasks, summary.items);
}

void populate_runtime_metadata(BatchSummary& summary,
                               const ModelMetadata& metadata) {
  summary.environment.onnxruntime_version = metadata.ort_version;
  summary.runtime.actual_provider = metadata.session_provider;
  summary.runtime.provider_evidence = metadata.provider_evidence;
  summary.runtime.execution_mode = metadata.execution_mode;
  summary.runtime.intra_op_num_threads = metadata.intra_op_num_threads;
  summary.runtime.inter_op_num_threads = metadata.inter_op_num_threads;
  summary.runtime.graph_optimization_level =
      metadata.graph_optimization_level;
}

void derive_counts_and_status(BatchSummary& summary) {
  summary.counts.succeeded = 0;
  summary.counts.failed = 0;
  summary.counts.cancelled = 0;
  std::vector<double> successful_latencies;
  for (const BatchItemResult& item : summary.items) {
    switch (item.status) {
      case BatchItemStatus::kSucceeded:
        ++summary.counts.succeeded;
        successful_latencies.push_back(item.latency_ms);
        break;
      case BatchItemStatus::kFailed:
        ++summary.counts.failed;
        break;
      case BatchItemStatus::kCancelled:
        ++summary.counts.cancelled;
        break;
    }
  }
  summary.counts.started =
      summary.counts.succeeded + summary.counts.failed;

  if (successful_latencies.empty()) {
    summary.latency_ms = {};
  } else {
    summary.latency_ms =
        calculate_latency_statistics(successful_latencies);
  }
  if (summary.counts.succeeded == 0 ||
      summary.timing.processing_wall_ms <= 0.0) {
    summary.throughput_images_per_second = 0.0;
  } else {
    summary.throughput_images_per_second =
        1000.0 * static_cast<double>(summary.counts.succeeded) /
        summary.timing.processing_wall_ms;
  }

  if (!summary.fatal_error.empty()) {
    summary.status = BatchStatus::kFatal;
  } else if (summary.cooperative_stop_requested ||
             summary.counts.cancelled != 0) {
    summary.status = BatchStatus::kCancelled;
  } else if (summary.counts.failed != 0) {
    summary.status = BatchStatus::kPartialFailure;
  } else {
    summary.status = BatchStatus::kSucceeded;
  }
}

struct WorkerStartGate {
  std::mutex mutex;
  std::condition_variable condition;
  std::size_t ready = 0;
  bool released = false;
};

class DetectorPipelineBatchExecutor final
    : public internal::BatchTaskExecutor {
 public:
  explicit DetectorPipelineBatchExecutor(RuntimeContract contract)
      : pipeline_(std::move(contract)) {}

  const ModelMetadata& metadata() const override {
    return pipeline_.metadata();
  }

  double session_initialization_ms() const override {
    return pipeline_.session_initialization_ms();
  }

  internal::BatchExecutionResult run(
      const std::filesystem::path& source_path,
      const DetectionOutputRequest& output_request) override {
    SingleImagePipelineResult result =
        pipeline_.run(source_path, output_request);
    internal::BatchExecutionResult execution;
    execution.detection_count = result.detection_result.detections.size();
    execution.outputs = std::move(result.outputs);
    return execution;
  }

 private:
  DetectorPipeline pipeline_;
};

class DetectorPipelineExecutorFactory final
    : public internal::BatchExecutorFactory {
 public:
  std::unique_ptr<internal::BatchTaskExecutor> create(
      const RuntimeContract& contract,
      std::size_t /*worker_index*/) override {
    return std::make_unique<DetectorPipelineBatchExecutor>(contract);
  }
};

}  // namespace

namespace internal {

bool BatchPathLocationLess::operator()(
    const std::filesystem::path& lhs,
    const std::filesystem::path& rhs) const {
  return PathLocationLess{}(lhs, rhs);
}

bool batch_path_text_equal(const std::filesystem::path& lhs,
                           const std::filesystem::path& rhs) {
  return path_text_equal(lhs, rhs);
}

bool batch_path_is_same_or_descendant(
    const std::filesystem::path& candidate,
    const std::filesystem::path& root) {
  return is_same_or_descendant(candidate, root);
}

bool batch_path_is_reparse_point(
    const std::filesystem::path& path,
    std::error_code& error) noexcept {
#ifdef _WIN32
  const DWORD attributes = GetFileAttributesW(path.native().c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    error = std::error_code(static_cast<int>(GetLastError()),
                            std::system_category());
    return false;
  }
  error.clear();
  return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
#else
  (void)path;
  error.clear();
  return false;
#endif
}

}  // namespace internal

class BatchRunner::Impl {
 public:
  Impl(RuntimeContract contract,
       std::shared_ptr<internal::BatchExecutorFactory> executor_factory)
      : contract_(std::move(contract)),
        executor_factory_(std::move(executor_factory)) {
    if (contract_.runtime.provider != ExecutionProvider::kCpu) {
      throw_batch_error(
          "runtime.provider", "cpu for the public BatchRunner workflow",
          to_string(contract_.runtime.provider),
          "use the single-image or benchmark workflow for GPU providers; "
          "BatchRunner deliberately avoids constructing concurrent GPU "
          "sessions that share one engine cache");
    }
    if (!executor_factory_) {
      throw_batch_error(
          "executor_factory", "a non-null executor factory", "null",
          "use the default BatchRunner constructor or inject a live fake "
          "factory in tests");
    }
  }

  BatchSummary run(const BatchRequest& request) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (run_started_) {
        throw_batch_error(
            "BatchRunner", "a new runner for each batch invocation",
            "run() was already called",
            "construct a new BatchRunner so stop/session state cannot leak "
            "between batches");
      }
      run_started_ = true;
    }

    const std::vector<BatchTask> tasks =
        discover_batch_tasks(request.input_kind, request.input_path);
    const NormalizedRequestPaths paths = validate_request_and_outputs(
        request, contract_, tasks);

    BatchSummary summary;
    populate_base_summary(summary, request, contract_, paths, tasks);

    std::vector<std::unique_ptr<internal::BatchTaskExecutor>> executors;
    executors.reserve(summary.runtime.effective_workers);
    try {
      for (std::size_t index = 0;
           index < summary.runtime.effective_workers; ++index) {
        auto executor = executor_factory_->create(contract_, index);
        if (!executor) {
          throw_batch_error(
              "worker_executors[" + std::to_string(index) + "]",
              "a non-null initialized task executor", "null",
              "return one executor instance per worker from the injected "
              "factory");
        }
        if (executors.empty()) {
          populate_runtime_metadata(summary, executor->metadata());
        } else if (!metadata_matches(executors.front()->metadata(),
                                     executor->metadata())) {
          throw_batch_error(
              "worker_sessions[" + std::to_string(index) + "]",
              "the same validated provider/thread/model metadata as worker "
              "0",
              "session metadata differs",
              "stop before processing and diagnose non-deterministic Runtime "
              "session construction");
        }
        summary.runtime.session_initialization_ms.push_back(
            executor->session_initialization_ms());
        executors.push_back(std::move(executor));
      }
      summary.runtime.session_count = executors.size();
    } catch (const std::exception& error) {
      summary.runtime.session_count =
          summary.runtime.session_initialization_ms.size();
      summary.fatal_error =
          std::string("worker session initialization failed before task ") +
          "processing: " + error.what();
      for (BatchItemResult& item : summary.items) {
        item.error =
            "cancelled because worker session initialization failed";
      }
      summary.cooperative_stop_requested =
          cooperative_stop_requested_.load(std::memory_order_acquire);
      summary.memory = collect_memory(summary.environment);
      derive_counts_and_status(summary);
      validate_batch_summary(summary);
      return summary;
    }

    auto queue = std::make_shared<internal::BoundedQueue<std::size_t>>(
        request.queue_capacity);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      active_queue_ = queue;
    }
    if (stop_requested_.load(std::memory_order_acquire)) {
      queue->request_stop();
    }

    WorkerStartGate gate;
    std::mutex fatal_mutex;
    std::atomic<bool> fatal_observed{false};
    const auto set_fatal = [&](const std::string& message) noexcept {
      try {
        if (!fatal_observed.exchange(true, std::memory_order_acq_rel)) {
          std::lock_guard<std::mutex> lock(fatal_mutex);
          summary.fatal_error = message;
        }
      } catch (...) {
        // Preserve cooperative shutdown even if recording diagnostics fails.
      }
      stop_requested_.store(true, std::memory_order_release);
      queue->request_stop();
    };

    std::vector<std::thread> workers;
    workers.reserve(summary.runtime.effective_workers);
    try {
      for (std::size_t worker_index = 0;
           worker_index < summary.runtime.effective_workers;
           ++worker_index) {
        workers.emplace_back([&, worker_index] {
          try {
            {
              std::unique_lock<std::mutex> lock(gate.mutex);
              ++gate.ready;
              gate.condition.notify_all();
              gate.condition.wait(lock, [&gate] { return gate.released; });
            }

            while (const std::optional<std::size_t> index = queue->pop()) {
              BatchItemResult& item = summary.items[*index];
              item.error.clear();
              const auto item_started = SteadyClock::now();
              try {
                DetectionOutputRequest output_request;
                output_request.json_path = planned_json_path(
                    paths.item_directory, item.sequence_index);
                if (request.output_images) {
                  output_request.image_path = planned_image_path(
                      paths.item_directory, item.sequence_index);
                }
                output_request.overwrite_existing =
                    request.overwrite_existing;

                internal::BatchExecutionResult execution_result =
                    executors[worker_index]->run(
                        item.source_path, output_request);
                item.latency_ms = elapsed_ms(item_started,
                                             SteadyClock::now());
                item.detection_count = execution_result.detection_count;
                item.json_output_path = execution_result.outputs.json_path;
                item.image_output_path = execution_result.outputs.image_path;
                item.status = BatchItemStatus::kSucceeded;
              } catch (const std::exception& error) {
                item.latency_ms = elapsed_ms(item_started,
                                             SteadyClock::now());
                item.status = BatchItemStatus::kFailed;
                item.detection_count = 0;
                item.json_output_path.reset();
                item.image_output_path.reset();
                item.error = error.what();
              } catch (...) {
                item.latency_ms = elapsed_ms(item_started,
                                             SteadyClock::now());
                item.status = BatchItemStatus::kFailed;
                item.detection_count = 0;
                item.json_output_path.reset();
                item.image_output_path.reset();
                item.error = "non-standard exception while processing image";
              }
            }
          } catch (const std::exception& error) {
            set_fatal("worker infrastructure failure: " +
                      std::string(error.what()));
          } catch (...) {
            set_fatal("worker infrastructure failure: non-standard "
                      "exception");
          }
        });
      }
    } catch (const std::exception& error) {
      set_fatal("worker thread creation failed: " +
                std::string(error.what()));
    }

    {
      std::unique_lock<std::mutex> lock(gate.mutex);
      gate.condition.wait(lock, [&] {
        return gate.ready == workers.size();
      });
      gate.released = true;
    }
    const auto processing_started = SteadyClock::now();
    gate.condition.notify_all();

    if (workers.size() == summary.runtime.effective_workers &&
        !fatal_observed.load(std::memory_order_acquire)) {
      try {
        for (const BatchTask& task : tasks) {
          if (stop_requested_.load(std::memory_order_acquire) ||
              !queue->push(task.sequence_index)) {
            break;
          }
          ++summary.counts.enqueued;
        }
        if (stop_requested_.load(std::memory_order_acquire)) {
          queue->request_stop();
        } else {
          queue->close();
        }
      } catch (const std::exception& error) {
        set_fatal("batch producer infrastructure failure: " +
                  std::string(error.what()));
      }
    }

    for (std::thread& worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    const auto processing_finished = SteadyClock::now();
    summary.timing.processing_wall_ms =
        elapsed_ms(processing_started, processing_finished);

    const internal::BoundedQueueStatistics queue_statistics =
        queue->statistics();
    summary.queue.capacity = queue_statistics.capacity;
    summary.queue.peak_depth = queue_statistics.peak_depth;
    summary.queue.producer_wait_count =
        queue_statistics.producer_wait_count;
    summary.queue.producer_wait_ms =
        std::chrono::duration<double, std::milli>(
            queue_statistics.producer_wait_duration).count();

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      active_queue_.reset();
    }

    const std::string cancellation_reason = summary.fatal_error.empty()
        ? "cancelled before start by cooperative stop request"
        : "cancelled before start by fatal infrastructure shutdown";
    for (BatchItemResult& item : summary.items) {
      if (item.status == BatchItemStatus::kCancelled) {
        item.error = cancellation_reason;
      }
    }
    summary.cooperative_stop_requested =
        cooperative_stop_requested_.load(std::memory_order_acquire);
    derive_counts_and_status(summary);
    summary.memory = collect_memory(summary.environment);
    validate_batch_summary(summary);
    return summary;
  }

  void request_stop() noexcept {
    cooperative_stop_requested_.store(true, std::memory_order_release);
    stop_requested_.store(true, std::memory_order_release);
    std::shared_ptr<internal::BoundedQueue<std::size_t>> queue;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      queue = active_queue_;
    }
    if (queue) {
      queue->request_stop();
    }
  }

 private:
  RuntimeContract contract_;
  std::shared_ptr<internal::BatchExecutorFactory> executor_factory_;
  std::atomic<bool> cooperative_stop_requested_{false};
  std::atomic<bool> stop_requested_{false};
  std::mutex state_mutex_;
  bool run_started_ = false;
  std::shared_ptr<internal::BoundedQueue<std::size_t>> active_queue_;
};

BatchRunner::BatchRunner(RuntimeContract contract)
    : BatchRunner(
          std::move(contract),
          std::make_shared<DetectorPipelineExecutorFactory>()) {}

BatchRunner::BatchRunner(
    RuntimeContract contract,
    std::shared_ptr<internal::BatchExecutorFactory> executor_factory)
    : impl_(std::make_unique<Impl>(
          std::move(contract), std::move(executor_factory))) {}

BatchRunner::~BatchRunner() = default;
BatchRunner::BatchRunner(BatchRunner&&) noexcept = default;
BatchRunner& BatchRunner::operator=(BatchRunner&&) noexcept = default;

BatchSummary BatchRunner::run(const BatchRequest& request) {
  if (!impl_) {
    throw_batch_error(
        "BatchRunner", "a live runner instance", "moved-from instance",
        "invoke run only on the object that owns batch state");
  }
  return impl_->run(request);
}

void BatchRunner::request_stop() noexcept {
  if (impl_) {
    impl_->request_stop();
  }
}

}  // namespace yolo_defect_cpp

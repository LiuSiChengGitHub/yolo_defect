#include "yolo_defect_cpp/batch_writer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo_defect_cpp {
namespace {

[[noreturn]] void throw_summary_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "BatchSummary validation failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

void validate_utf8(const std::string& value, const std::string& object) {
  const auto continuation = [](unsigned char byte) {
    return byte >= 0x80U && byte <= 0xBFU;
  };
  std::size_t index = 0;
  while (index < value.size()) {
    const unsigned char first = static_cast<unsigned char>(value[index]);
    std::size_t length = 0;
    if (first <= 0x7FU) {
      length = 1;
    } else if (first >= 0xC2U && first <= 0xDFU) {
      length = 2;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      length = 3;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      length = 4;
    } else {
      throw_summary_error(
          object, "valid UTF-8", "invalid leading byte at byte " +
              std::to_string(index),
          "populate paths and diagnostics with valid UTF-8 text");
    }
    if (index + length > value.size()) {
      throw_summary_error(
          object, "valid UTF-8", "truncated sequence at byte " +
              std::to_string(index),
          "populate paths and diagnostics with complete UTF-8 text");
    }
    for (std::size_t offset = 1; offset < length; ++offset) {
      if (!continuation(
              static_cast<unsigned char>(value[index + offset]))) {
        throw_summary_error(
            object, "valid UTF-8", "invalid continuation byte at byte " +
                std::to_string(index + offset),
            "populate paths and diagnostics with valid UTF-8 text");
      }
    }
    if (length == 3) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xE0U && second < 0xA0U) ||
          (first == 0xEDU && second > 0x9FU)) {
        throw_summary_error(
            object, "canonical UTF-8", "invalid three-byte sequence at byte " +
                std::to_string(index),
            "remove overlong or surrogate encodings");
      }
    } else if (length == 4) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xF0U && second < 0x90U) ||
          (first == 0xF4U && second > 0x8FU)) {
        throw_summary_error(
            object, "UTF-8 in the Unicode range",
            "invalid four-byte sequence at byte " +
                std::to_string(index),
            "remove overlong or out-of-range encodings");
      }
    }
    index += length;
  }
}

void validate_non_empty(const std::string& value,
                        const std::string& object) {
  if (value.empty()) {
    throw_summary_error(
        object, "non-empty UTF-8 text", "empty",
        "populate the field before serializing BatchSummary");
  }
  validate_utf8(value, object);
}

std::string path_to_utf8(const std::filesystem::path& path,
                         const std::string& object) {
  if (path.empty()) {
    throw_summary_error(
        object, "a non-empty filesystem path", "empty",
        "retain the normalized path in BatchSummary");
  }
  std::string value;
  try {
    value = path.generic_u8string();
  } catch (const std::exception& error) {
    throw_summary_error(
        object, "a path convertible to UTF-8", error.what(),
        "use a valid Unicode filesystem path");
  }
  validate_utf8(value, object);
  return value;
}

std::string escape_json(const std::string& value,
                        const std::string& object) {
  validate_utf8(value, object);
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (unsigned char byte : value) {
    switch (byte) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          escaped += "\\u00";
          escaped.push_back(kHex[(byte >> 4U) & 0x0FU]);
          escaped.push_back(kHex[byte & 0x0FU]);
        } else {
          escaped.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  escaped.push_back('"');
  return escaped;
}

std::string format_double(double value, const std::string& object) {
  if (!std::isfinite(value)) {
    throw_summary_error(
        object, "a finite JSON number", "NaN or Infinity",
        "discard invalid timing/memory data before serialization");
  }
  if (value == 0.0) {
    value = 0.0;
  }
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return output.str();
}

void validate_finite_non_negative(double value,
                                  const std::string& object) {
  if (!std::isfinite(value) || value < 0.0) {
    throw_summary_error(
        object, "a finite non-negative number", std::to_string(value),
        "derive the value from validated steady-clock/process metrics");
  }
}

bool is_sha256(const std::string& value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(value.begin(), value.end(),
                     [](unsigned char character) {
                       return (character >= '0' && character <= '9') ||
                              (character >= 'a' && character <= 'f') ||
                              (character >= 'A' && character <= 'F');
                     });
}

void validate_string_list(const std::vector<std::string>& values,
                          const std::string& object) {
  if (values.empty()) {
    throw_summary_error(
        object, "at least one disclosure", "empty",
        "state the timing scope and evidence limitations");
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    validate_non_empty(values[index],
                       object + "[" + std::to_string(index) + "]");
  }
}

void validate_latency(const BatchSummary& summary) {
  const LatencyStatistics& latency = summary.latency_ms;
  if (latency.sample_count != summary.counts.succeeded) {
    throw_summary_error(
        "latency_ms.sample_count", std::to_string(summary.counts.succeeded),
        std::to_string(latency.sample_count),
        "calculate aggregate latency from successful items only");
  }
  validate_finite_non_negative(latency.mean_ms, "latency_ms.mean_ms");
  validate_finite_non_negative(latency.p50_ms, "latency_ms.p50_ms");
  validate_finite_non_negative(latency.p95_ms, "latency_ms.p95_ms");
  validate_finite_non_negative(summary.throughput_images_per_second,
                               "throughput_images_per_second");
  if (summary.counts.succeeded == 0) {
    if (latency.mean_ms != 0.0 || latency.p50_ms != 0.0 ||
        latency.p95_ms != 0.0 ||
        summary.throughput_images_per_second != 0.0) {
      throw_summary_error(
          "latency_ms", "zero statistics and throughput with no successes",
          "non-zero aggregate", "publish per-attempt failed latency only in "
          "items");
    }
    return;
  }
  if (latency.mean_ms <= 0.0 || latency.p50_ms > latency.p95_ms ||
      summary.timing.processing_wall_ms <= 0.0) {
    throw_summary_error(
        "latency_ms", "positive valid successful latency and p50 <= p95",
        "mean=" + std::to_string(latency.mean_ms) +
            ", p50=" + std::to_string(latency.p50_ms) +
            ", p95=" + std::to_string(latency.p95_ms),
        "retain one complete positive timing per successful image");
  }
  const double expected =
      1000.0 * static_cast<double>(summary.counts.succeeded) /
      summary.timing.processing_wall_ms;
  const double tolerance = std::max(1.0, expected) * 1.0e-12;
  if (summary.throughput_images_per_second <= 0.0 ||
      std::abs(summary.throughput_images_per_second - expected) >
          tolerance) {
    throw_summary_error(
        "throughput_images_per_second",
        "1000 * succeeded / timing.processing_wall_ms",
        std::to_string(summary.throughput_images_per_second),
        "derive throughput from the full bounded-queue processing wall "
        "interval");
  }
}

void write_string_array(std::ostringstream& output,
                        const std::vector<std::string>& values,
                        const std::string& indent,
                        const std::string& object) {
  if (values.empty()) {
    output << "[]";
    return;
  }
  output << "[\n";
  for (std::size_t index = 0; index < values.size(); ++index) {
    output << indent << "  "
           << escape_json(values[index], object + "[" +
                                         std::to_string(index) + "]");
    if (index + 1 != values.size()) {
      output << ',';
    }
    output << '\n';
  }
  output << indent << ']';
}

void write_shape(std::ostringstream& output,
                 const std::vector<std::int64_t>& shape) {
  output << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << shape[index];
  }
  output << ']';
}

void ensure_parent_directory(const std::filesystem::path& output_path) {
  const std::filesystem::path parent = output_path.parent_path();
  if (parent.empty()) {
    return;
  }
  std::error_code error;
  if (std::filesystem::exists(parent, error)) {
    if (error || !std::filesystem::is_directory(parent, error) || error) {
      throw_summary_error(
          "output.parent", "an existing directory",
          path_to_utf8(parent, "output.parent") +
              (error ? " (" + error.message() + ")" : ""),
          "choose a writable summary directory");
    }
    return;
  }
  if (error ||
      (!std::filesystem::create_directories(parent, error) && error)) {
    throw_summary_error(
        "output.parent", "a creatable directory",
        path_to_utf8(parent, "output.parent") +
            (error ? " (" + error.message() + ")" : ""),
        "choose a writable summary location");
  }
}

std::filesystem::path normalized_output_path(
    const std::filesystem::path& path) {
  if (path.empty()) {
    throw_summary_error(
        "output_path", "a non-empty JSON path", "empty",
        "pass the same path recorded in summary.output.batch_summary_path");
  }
  std::error_code error;
  std::filesystem::path normalized = std::filesystem::absolute(path, error);
  if (error) {
    throw_summary_error(
        "output_path", "a path resolvable from the working directory",
        path_to_utf8(path, "output_path") + " (" + error.message() + ")",
        "correct the output path or working directory");
  }
  normalized = std::filesystem::weakly_canonical(normalized, error);
  if (error) {
    throw_summary_error(
        "output_path", "a path with inspectable existing ancestors",
        path_to_utf8(path, "output_path") + " (" + error.message() + ")",
        "correct the output path and ancestor permissions");
  }
  return normalized.lexically_normal();
}

void validate_target_state(const std::filesystem::path& path,
                           bool overwrite_existing) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (error && status.type() != std::filesystem::file_type::not_found) {
    throw_summary_error(
        "output_path", "an inspectable output target",
        path_to_utf8(path, "output_path") + " (" + error.message() + ")",
        "check path permissions");
  }
  if (!std::filesystem::exists(status)) {
    return;
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw_summary_error(
        "output_path", "a missing path or existing regular file",
        path_to_utf8(path, "output_path") +
            " is a directory, symlink, or special object",
        "choose a regular JSON file path");
  }
  if (!overwrite_existing) {
    throw_summary_error(
        "output_path", "a path that does not exist",
        path_to_utf8(path, "output_path") + " already exists",
        "choose a new file or explicitly enable overwrite_existing");
  }
}

}  // namespace

void validate_batch_summary(const BatchSummary& summary) {
  if (summary.schema_version != 1) {
    throw_summary_error(
        "schema_version", "1", std::to_string(summary.schema_version),
        "use the S2-03 BatchSummary schema");
  }
  if (summary.evidence_type != "cpp_ort_multi_image_batch_summary") {
    throw_summary_error(
        "evidence_type", "cpp_ort_multi_image_batch_summary",
        summary.evidence_type,
        "do not mix single-image benchmark and batch schemas");
  }
  validate_non_empty(summary.timestamp_utc, "timestamp_utc");
  if (summary.command_arguments.empty()) {
    throw_summary_error(
        "command_arguments", "at least the executable argument", "empty",
        "retain the original argv vector");
  }
  for (std::size_t index = 0; index < summary.command_arguments.size();
       ++index) {
    validate_non_empty(summary.command_arguments[index],
                       "command_arguments[" + std::to_string(index) + "]");
  }

  const BatchEnvironment& environment = summary.environment;
  validate_non_empty(environment.hostname, "environment.hostname");
  validate_non_empty(environment.processor, "environment.processor");
  if (environment.logical_cpu_count == 0) {
    throw_summary_error(
        "environment.logical_cpu_count", "a positive count", "0",
        "record the platform logical processor count");
  }
  validate_non_empty(environment.os_name, "environment.os_name");
  validate_non_empty(environment.os_version, "environment.os_version");
  validate_non_empty(environment.target_architecture,
                     "environment.target_architecture");
  validate_non_empty(environment.runtime_kernel_architecture,
                     "environment.runtime_kernel_architecture");
  validate_non_empty(environment.execution_context,
                     "environment.execution_context");
  validate_non_empty(environment.compiler_id, "environment.compiler_id");
  validate_non_empty(environment.compiler_version,
                     "environment.compiler_version");
  validate_non_empty(environment.build_type, "environment.build_type");
  if (environment.cxx_standard != 17) {
    throw_summary_error(
        "environment.cxx_standard", "17",
        std::to_string(environment.cxx_standard),
        "build the Runtime as C++17");
  }
  validate_non_empty(environment.opencv_version,
                     "environment.opencv_version");
  if (summary.status != BatchStatus::kFatal ||
      summary.runtime.session_count != 0) {
    validate_non_empty(environment.onnxruntime_version,
                       "environment.onnxruntime_version");
  }

  const BatchRuntimeMetadata& runtime = summary.runtime;
  path_to_utf8(runtime.config_path, "runtime.config_path");
  if (runtime.requested_provider != "cpu") {
    throw_summary_error(
        "runtime.requested_provider", "cpu", runtime.requested_provider,
        "use the fixed CPU RuntimeConfig for S2-03");
  }
  validate_finite_non_negative(runtime.score_threshold,
                               "runtime.score_threshold");
  validate_finite_non_negative(runtime.nms_threshold,
                               "runtime.nms_threshold");
  validate_non_empty(runtime.nms_mode, "runtime.nms_mode");
  if (runtime.requested_workers == 0 || runtime.requested_workers > 64 ||
      runtime.effective_workers == 0 ||
      runtime.effective_workers > runtime.requested_workers ||
      runtime.effective_workers > summary.counts.discovered) {
    throw_summary_error(
        "runtime.workers",
        "1 <= effective <= requested <= 64 and effective <= discovered",
        "requested=" + std::to_string(runtime.requested_workers) +
            ", effective=" + std::to_string(runtime.effective_workers),
        "derive effective workers as min(requested, discovered)");
  }
  if (runtime.session_count != runtime.session_initialization_ms.size() ||
      runtime.session_count > runtime.effective_workers) {
    throw_summary_error(
        "runtime.session_count",
        "the initialization sample count and at most effective_workers",
        std::to_string(runtime.session_count),
        "retain one initialization observation for each constructed worker "
        "session");
  }
  for (std::size_t index = 0;
       index < runtime.session_initialization_ms.size(); ++index) {
    validate_finite_non_negative(
        runtime.session_initialization_ms[index],
        "runtime.session_initialization_ms[" + std::to_string(index) + "]");
  }
  if (runtime.session_count != 0) {
    if (runtime.actual_provider != "CPUExecutionProvider" ||
        runtime.provider_evidence.empty() ||
        runtime.execution_mode != "sequential" ||
        runtime.intra_op_num_threads != 1 ||
        runtime.inter_op_num_threads != 1 ||
        runtime.graph_optimization_level != "all") {
      throw_summary_error(
          "runtime.session_policy",
          "CPUExecutionProvider, sequential, intra/inter-op 1/1, graph all",
          "provider=" + runtime.actual_provider +
              ", mode=" + runtime.execution_mode +
              ", threads=" +
              std::to_string(runtime.intra_op_num_threads) + "/" +
              std::to_string(runtime.inter_op_num_threads),
          "restore the fixed OnnxRunner CPU policy");
    }
    validate_non_empty(runtime.provider_evidence,
                       "runtime.provider_evidence");
  } else if (summary.status != BatchStatus::kFatal) {
    throw_summary_error(
        "runtime.session_count", "effective_workers for a non-fatal run", "0",
        "construct every worker session before releasing tasks");
  }
  if (summary.status != BatchStatus::kFatal &&
      runtime.session_count != runtime.effective_workers) {
    throw_summary_error(
        "runtime.session_count", std::to_string(runtime.effective_workers),
        std::to_string(runtime.session_count),
        "preconstruct one DetectorPipeline per effective worker");
  }

  const BatchModelMetadata& model = summary.model;
  validate_non_empty(model.model_id, "model.model_id");
  validate_non_empty(model.model_family, "model.model_family");
  path_to_utf8(model.model_path, "model.model_path");
  if (!is_sha256(model.declared_sha256)) {
    throw_summary_error(
        "model.declared_sha256", "64 hexadecimal characters",
        model.declared_sha256,
        "copy the validated artifact declaration digest");
  }
  if (model.opset <= 0 || model.input_shape.empty()) {
    throw_summary_error(
        "model.contract", "a positive opset and non-empty input shape",
        "opset=" + std::to_string(model.opset),
        "copy ModelArtifactSpec into BatchSummary");
  }
  validate_non_empty(model.input_name, "model.input_name");
  validate_non_empty(model.input_dtype, "model.input_dtype");
  validate_non_empty(model.input_layout, "model.input_layout");

  path_to_utf8(summary.input.source_path, "input.source_path");
  validate_non_empty(summary.input.ordering, "input.ordering");
  path_to_utf8(summary.output.directory, "output.directory");
  path_to_utf8(summary.output.batch_summary_path,
               "output.batch_summary_path");
  path_to_utf8(summary.output.item_directory, "output.item_directory");
  if (!summary.output.json_outputs) {
    throw_summary_error(
        "output.json_outputs", "true", "false",
        "always retain one detection JSON for each successful image");
  }

  const BatchCounts& counts = summary.counts;
  if (counts.discovered == 0 || counts.discovered != summary.items.size()) {
    throw_summary_error(
        "counts.discovered", "a positive value equal to items.size()",
        std::to_string(counts.discovered),
        "retain every discovered task in deterministic order");
  }
  if (counts.discovered !=
          counts.succeeded + counts.failed + counts.cancelled ||
      counts.started != counts.succeeded + counts.failed ||
      counts.started > counts.enqueued || counts.enqueued > counts.discovered) {
    throw_summary_error(
        "counts",
        "discovered=succeeded+failed+cancelled, "
        "started=succeeded+failed, started<=enqueued<=discovered",
        "discovered=" + std::to_string(counts.discovered) +
            ", enqueued=" + std::to_string(counts.enqueued) +
            ", started=" + std::to_string(counts.started) +
            ", succeeded=" + std::to_string(counts.succeeded) +
            ", failed=" + std::to_string(counts.failed) +
            ", cancelled=" + std::to_string(counts.cancelled),
        "derive counts from the final ordered item array");
  }

  if (summary.queue.capacity == 0 || summary.queue.capacity > 4096 ||
      summary.queue.peak_depth > summary.queue.capacity) {
    throw_summary_error(
        "queue", "capacity in [1,4096] and peak_depth <= capacity",
        "capacity=" + std::to_string(summary.queue.capacity) +
            ", peak=" + std::to_string(summary.queue.peak_depth),
        "use the mutex/condition-variable BoundedQueue statistics");
  }
  validate_finite_non_negative(summary.queue.producer_wait_ms,
                               "queue.producer_wait_ms");
  validate_finite_non_negative(summary.timing.processing_wall_ms,
                               "timing.processing_wall_ms");
  validate_string_list(summary.timing.includes, "timing.includes");
  validate_string_list(summary.timing.excludes, "timing.excludes");

  std::size_t succeeded = 0;
  std::size_t failed = 0;
  std::size_t cancelled = 0;
  for (std::size_t index = 0; index < summary.items.size(); ++index) {
    const BatchItemResult& item = summary.items[index];
    const std::string object = "items[" + std::to_string(index) + "]";
    if (item.sequence_index != index) {
      throw_summary_error(
          object + ".sequence_index", std::to_string(index),
          std::to_string(item.sequence_index),
          "retain deterministic discovery order in the summary");
    }
    path_to_utf8(item.source_path, object + ".source_path");
    validate_finite_non_negative(item.latency_ms, object + ".latency_ms");
    switch (item.status) {
      case BatchItemStatus::kSucceeded:
        ++succeeded;
        if (!item.json_output_path.has_value() ||
            item.error.size() != 0 || item.latency_ms <= 0.0 ||
            (summary.output.image_outputs &&
             !item.image_output_path.has_value()) ||
            (!summary.output.image_outputs &&
             item.image_output_path.has_value())) {
          throw_summary_error(
              object, "successful output paths, positive latency, no error",
              "inconsistent success result",
              "record only outputs confirmed by DetectorPipeline::run");
        }
        path_to_utf8(*item.json_output_path,
                     object + ".json_output_path");
        if (item.image_output_path.has_value()) {
          path_to_utf8(*item.image_output_path,
                       object + ".image_output_path");
        }
        break;
      case BatchItemStatus::kFailed:
        ++failed;
        if (item.json_output_path.has_value() ||
            item.image_output_path.has_value() || item.error.empty()) {
          throw_summary_error(
              object, "null output paths and an actionable error",
              "inconsistent failed result",
              "continue other images but retain the per-image exception");
        }
        validate_utf8(item.error, object + ".error");
        break;
      case BatchItemStatus::kCancelled:
        ++cancelled;
        if (item.json_output_path.has_value() ||
            item.image_output_path.has_value() || item.detection_count != 0 ||
            item.latency_ms != 0.0 || item.error.empty()) {
          throw_summary_error(
              object,
              "no outputs/detections/latency and a cancellation reason",
              "inconsistent cancelled result",
              "leave unstarted tasks untouched during cooperative stop");
        }
        validate_utf8(item.error, object + ".error");
        break;
    }
  }
  if (succeeded != counts.succeeded || failed != counts.failed ||
      cancelled != counts.cancelled) {
    throw_summary_error(
        "counts", "exact agreement with item statuses",
        "item status totals differ", "derive counts after all workers join");
  }
  validate_latency(summary);

  const BatchMemoryEvidence& memory = summary.memory;
  validate_non_empty(memory.status, "memory.status");
  validate_non_empty(memory.metric, "memory.metric");
  validate_non_empty(memory.scope, "memory.scope");
  validate_utf8(memory.reason, "memory.reason");
  validate_finite_non_negative(memory.mebibytes, "memory.mebibytes");
  if (memory.supported) {
    if (memory.bytes == 0 || memory.mebibytes <= 0.0) {
      throw_summary_error(
          "memory", "positive bytes and MiB when supported",
          "bytes=" + std::to_string(memory.bytes) +
              ", MiB=" + std::to_string(memory.mebibytes),
          "query the platform process high-water mark");
    }
    const double expected =
        static_cast<double>(memory.bytes) / (1024.0 * 1024.0);
    const double tolerance = std::max(1.0, expected) * 1.0e-12;
    if (std::abs(memory.mebibytes - expected) > tolerance) {
      throw_summary_error(
          "memory.mebibytes", "bytes / (1024 * 1024)",
          std::to_string(memory.mebibytes),
          "derive MiB from the recorded byte count");
    }
  } else {
    if (memory.reason.empty()) {
      throw_summary_error(
          "memory.reason", "a reason when memory is unsupported/unavailable",
          "empty", "retain the platform query failure or support boundary");
    }
    if (memory.publishable) {
      throw_summary_error(
          "memory.publishable", "false when the memory query is unsupported",
          "true", "publish a process-memory result only after a supported "
                  "platform query succeeds");
    }
  }
  validate_string_list(summary.limitations, "limitations");

  const bool has_fatal_error = !summary.fatal_error.empty();
  validate_utf8(summary.fatal_error, "fatal_error");
  BatchStatus expected_status = BatchStatus::kSucceeded;
  if (has_fatal_error) {
    expected_status = BatchStatus::kFatal;
  } else if (summary.cooperative_stop_requested || counts.cancelled != 0) {
    expected_status = BatchStatus::kCancelled;
  } else if (counts.failed != 0) {
    expected_status = BatchStatus::kPartialFailure;
  }
  if (summary.status != expected_status) {
    throw_summary_error(
        "status", to_string(expected_status), to_string(summary.status),
        "derive status from fatal_error and final counts");
  }
}

std::string serialize_batch_summary_json(const BatchSummary& summary) {
  validate_batch_summary(summary);
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output
      << "{\n"
      << "  \"schema_version\": " << summary.schema_version << ",\n"
      << "  \"evidence_type\": "
      << escape_json(summary.evidence_type, "evidence_type") << ",\n"
      << "  \"timestamp_utc\": "
      << escape_json(summary.timestamp_utc, "timestamp_utc") << ",\n"
      << "  \"status\": " << escape_json(to_string(summary.status), "status")
      << ",\n"
      << "  \"cooperative_stop_requested\": "
      << (summary.cooperative_stop_requested ? "true" : "false")
      << ",\n"
      << "  \"command_arguments\": ";
  write_string_array(output, summary.command_arguments, "  ",
                     "command_arguments");
  output << ",\n";

  const BatchEnvironment& environment = summary.environment;
  output
      << "  \"environment\": {\n"
      << "    \"hostname\": "
      << escape_json(environment.hostname, "environment.hostname") << ",\n"
      << "    \"processor\": "
      << escape_json(environment.processor, "environment.processor") << ",\n"
      << "    \"logical_cpu_count\": " << environment.logical_cpu_count
      << ",\n"
      << "    \"os_name\": "
      << escape_json(environment.os_name, "environment.os_name") << ",\n"
      << "    \"os_version\": "
      << escape_json(environment.os_version, "environment.os_version")
      << ",\n"
      << "    \"target_architecture\": "
      << escape_json(environment.target_architecture,
                     "environment.target_architecture")
      << ",\n"
      << "    \"runtime_kernel_architecture\": "
      << escape_json(environment.runtime_kernel_architecture,
                     "environment.runtime_kernel_architecture")
      << ",\n"
      << "    \"execution_context\": "
      << escape_json(environment.execution_context,
                     "environment.execution_context")
      << ",\n"
      << "    \"compiler_id\": "
      << escape_json(environment.compiler_id, "environment.compiler_id")
      << ",\n"
      << "    \"compiler_version\": "
      << escape_json(environment.compiler_version,
                     "environment.compiler_version")
      << ",\n"
      << "    \"build_type\": "
      << escape_json(environment.build_type, "environment.build_type")
      << ",\n"
      << "    \"cxx_standard\": " << environment.cxx_standard << ",\n"
      << "    \"opencv_version\": "
      << escape_json(environment.opencv_version,
                     "environment.opencv_version")
      << ",\n"
      << "    \"onnxruntime_version\": "
      << escape_json(environment.onnxruntime_version,
                     "environment.onnxruntime_version")
      << "\n"
      << "  },\n";

  const BatchRuntimeMetadata& runtime = summary.runtime;
  output
      << "  \"runtime\": {\n"
      << "    \"config_path\": "
      << escape_json(path_to_utf8(runtime.config_path, "runtime.config_path"),
                     "runtime.config_path")
      << ",\n"
      << "    \"requested_provider\": "
      << escape_json(runtime.requested_provider,
                     "runtime.requested_provider")
      << ",\n"
      << "    \"actual_provider\": "
      << escape_json(runtime.actual_provider, "runtime.actual_provider")
      << ",\n"
      << "    \"provider_evidence\": "
      << escape_json(runtime.provider_evidence,
                     "runtime.provider_evidence")
      << ",\n"
      << "    \"execution_mode\": "
      << escape_json(runtime.execution_mode, "runtime.execution_mode")
      << ",\n"
      << "    \"intra_op_num_threads\": "
      << runtime.intra_op_num_threads << ",\n"
      << "    \"inter_op_num_threads\": "
      << runtime.inter_op_num_threads << ",\n"
      << "    \"graph_optimization_level\": "
      << escape_json(runtime.graph_optimization_level,
                     "runtime.graph_optimization_level")
      << ",\n"
      << "    \"score_threshold\": "
      << format_double(runtime.score_threshold, "runtime.score_threshold")
      << ",\n"
      << "    \"nms_threshold\": "
      << format_double(runtime.nms_threshold, "runtime.nms_threshold")
      << ",\n"
      << "    \"nms_mode\": "
      << escape_json(runtime.nms_mode, "runtime.nms_mode") << ",\n"
      << "    \"requested_workers\": " << runtime.requested_workers
      << ",\n"
      << "    \"effective_workers\": " << runtime.effective_workers
      << ",\n"
      << "    \"session_count\": " << runtime.session_count << ",\n"
      << "    \"session_initialization_ms\": [";
  for (std::size_t index = 0;
       index < runtime.session_initialization_ms.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << format_double(
        runtime.session_initialization_ms[index],
        "runtime.session_initialization_ms[" + std::to_string(index) + "]");
  }
  output << "]\n  },\n";

  const BatchModelMetadata& model = summary.model;
  output
      << "  \"model\": {\n"
      << "    \"model_id\": "
      << escape_json(model.model_id, "model.model_id") << ",\n"
      << "    \"model_family\": "
      << escape_json(model.model_family, "model.model_family") << ",\n"
      << "    \"model_path\": "
      << escape_json(path_to_utf8(model.model_path, "model.model_path"),
                     "model.model_path")
      << ",\n"
      << "    \"declared_sha256\": "
      << escape_json(model.declared_sha256, "model.declared_sha256")
      << ",\n"
      << "    \"opset\": " << model.opset << ",\n"
      << "    \"input_name\": "
      << escape_json(model.input_name, "model.input_name") << ",\n"
      << "    \"input_shape\": ";
  write_shape(output, model.input_shape);
  output
      << ",\n    \"input_dtype\": "
      << escape_json(model.input_dtype, "model.input_dtype")
      << ",\n    \"input_layout\": "
      << escape_json(model.input_layout, "model.input_layout")
      << "\n  },\n";

  output
      << "  \"input\": {\n"
      << "    \"kind\": "
      << escape_json(to_string(summary.input.kind), "input.kind") << ",\n"
      << "    \"source_path\": "
      << escape_json(path_to_utf8(summary.input.source_path,
                                  "input.source_path"),
                     "input.source_path")
      << ",\n"
      << "    \"ordering\": "
      << escape_json(summary.input.ordering, "input.ordering")
      << "\n  },\n";

  output
      << "  \"output\": {\n"
      << "    \"directory\": "
      << escape_json(path_to_utf8(summary.output.directory,
                                  "output.directory"),
                     "output.directory")
      << ",\n"
      << "    \"batch_summary_path\": "
      << escape_json(path_to_utf8(summary.output.batch_summary_path,
                                  "output.batch_summary_path"),
                     "output.batch_summary_path")
      << ",\n"
      << "    \"item_directory\": "
      << escape_json(path_to_utf8(summary.output.item_directory,
                                  "output.item_directory"),
                     "output.item_directory")
      << ",\n"
      << "    \"json_outputs\": "
      << (summary.output.json_outputs ? "true" : "false") << ",\n"
      << "    \"image_outputs\": "
      << (summary.output.image_outputs ? "true" : "false") << ",\n"
      << "    \"overwrite_existing\": "
      << (summary.output.overwrite_existing ? "true" : "false")
      << "\n  },\n";

  const BatchCounts& counts = summary.counts;
  output
      << "  \"counts\": {\"discovered\": " << counts.discovered
      << ", \"enqueued\": " << counts.enqueued
      << ", \"started\": " << counts.started
      << ", \"succeeded\": " << counts.succeeded
      << ", \"failed\": " << counts.failed
      << ", \"cancelled\": " << counts.cancelled << "},\n"
      << "  \"queue\": {\"capacity\": " << summary.queue.capacity
      << ", \"peak_depth\": " << summary.queue.peak_depth
      << ", \"producer_wait_count\": "
      << summary.queue.producer_wait_count
      << ", \"producer_wait_ms\": "
      << format_double(summary.queue.producer_wait_ms,
                       "queue.producer_wait_ms")
      << "},\n"
      << "  \"timing\": {\n"
      << "    \"processing_wall_ms\": "
      << format_double(summary.timing.processing_wall_ms,
                       "timing.processing_wall_ms")
      << ",\n    \"includes\": ";
  write_string_array(output, summary.timing.includes, "    ",
                     "timing.includes");
  output << ",\n    \"excludes\": ";
  write_string_array(output, summary.timing.excludes, "    ",
                     "timing.excludes");
  output << "\n  },\n";

  output
      << "  \"latency_ms\": {\"sample_count\": "
      << summary.latency_ms.sample_count << ", \"mean_ms\": "
      << format_double(summary.latency_ms.mean_ms, "latency_ms.mean_ms")
      << ", \"p50_ms\": "
      << format_double(summary.latency_ms.p50_ms, "latency_ms.p50_ms")
      << ", \"p95_ms\": "
      << format_double(summary.latency_ms.p95_ms, "latency_ms.p95_ms")
      << "},\n"
      << "  \"throughput_images_per_second\": "
      << format_double(summary.throughput_images_per_second,
                       "throughput_images_per_second")
      << ",\n";

  const BatchMemoryEvidence& memory = summary.memory;
  output
      << "  \"memory\": {\"supported\": "
      << (memory.supported ? "true" : "false")
      << ", \"status\": " << escape_json(memory.status, "memory.status")
      << ", \"metric\": " << escape_json(memory.metric, "memory.metric")
      << ", \"bytes\": ";
  if (memory.supported) {
    output << memory.bytes;
  } else {
    output << "null";
  }
  output << ", \"mebibytes\": ";
  if (memory.supported) {
    output << format_double(memory.mebibytes, "memory.mebibytes");
  } else {
    output << "null";
  }
  output << ", \"scope\": " << escape_json(memory.scope, "memory.scope")
         << ", \"reason\": ";
  if (memory.reason.empty()) {
    output << "null";
  } else {
    output << escape_json(memory.reason, "memory.reason");
  }
  output << ", \"publishable\": "
         << (memory.publishable ? "true" : "false") << "},\n";

  output << "  \"items\": [\n";
  for (std::size_t index = 0; index < summary.items.size(); ++index) {
    const BatchItemResult& item = summary.items[index];
    const std::string object = "items[" + std::to_string(index) + "]";
    output
        << "    {\n"
        << "      \"sequence_index\": " << item.sequence_index << ",\n"
        << "      \"status\": "
        << escape_json(to_string(item.status), object + ".status") << ",\n"
        << "      \"source_path\": "
        << escape_json(path_to_utf8(item.source_path,
                                    object + ".source_path"),
                       object + ".source_path")
        << ",\n"
        << "      \"json_output_path\": ";
    if (item.json_output_path.has_value()) {
      output << escape_json(
          path_to_utf8(*item.json_output_path,
                       object + ".json_output_path"),
          object + ".json_output_path");
    } else {
      output << "null";
    }
    output << ",\n      \"image_output_path\": ";
    if (item.image_output_path.has_value()) {
      output << escape_json(
          path_to_utf8(*item.image_output_path,
                       object + ".image_output_path"),
          object + ".image_output_path");
    } else {
      output << "null";
    }
    output
        << ",\n      \"detection_count\": " << item.detection_count
        << ",\n      \"latency_ms\": "
        << format_double(item.latency_ms, object + ".latency_ms")
        << ",\n      \"error\": ";
    if (item.error.empty()) {
      output << "null";
    } else {
      output << escape_json(item.error, object + ".error");
    }
    output << "\n    }";
    if (index + 1 != summary.items.size()) {
      output << ',';
    }
    output << '\n';
  }
  output << "  ],\n  \"limitations\": ";
  write_string_array(output, summary.limitations, "  ", "limitations");
  output << ",\n  \"fatal_error\": ";
  if (summary.fatal_error.empty()) {
    output << "null";
  } else {
    output << escape_json(summary.fatal_error, "fatal_error");
  }
  output << "\n}\n";
  return output.str();
}

void write_batch_summary_json(
    const BatchSummary& summary,
    const std::filesystem::path& output_path,
    bool overwrite_existing) {
  const std::string document = serialize_batch_summary_json(summary);
  const std::filesystem::path normalized =
      normalized_output_path(output_path);
  const std::filesystem::path recorded = normalized_output_path(
      summary.output.batch_summary_path);
  if (normalized != recorded) {
    throw_summary_error(
        "output_path", "the path recorded in output.batch_summary_path",
        path_to_utf8(normalized, "output_path"),
        "pass BatchRequest.summary_path unchanged to the writer");
  }
  validate_target_state(normalized, overwrite_existing);
  ensure_parent_directory(normalized);
  validate_target_state(normalized, overwrite_existing);

  std::ofstream output(normalized, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw_summary_error(
        "output_path", "a writable JSON file",
        path_to_utf8(normalized, "output_path"),
        "check directory permissions and file locks");
  }
  output.write(document.data(), static_cast<std::streamsize>(document.size()));
  output.flush();
  if (!output) {
    throw_summary_error(
        "output_path", "the complete BatchSummary JSON written",
        "filesystem write failure at " +
            path_to_utf8(normalized, "output_path"),
        "check free space and filesystem health");
  }
  output.close();
  if (!output) {
    throw_summary_error(
        "output_path", "a successfully closed BatchSummary JSON",
        "filesystem close failure at " +
            path_to_utf8(normalized, "output_path"),
        "check free space and filesystem health");
  }
}

}  // namespace yolo_defect_cpp

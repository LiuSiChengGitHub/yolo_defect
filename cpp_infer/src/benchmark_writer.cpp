#include "yolo_defect_cpp/benchmark_writer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace yolo_defect_cpp {
namespace {

[[noreturn]] void throw_writer_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Benchmark output error: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

std::string display_path(const std::filesystem::path& path) {
  try {
    return path.u8string();
  } catch (const std::exception&) {
    return "<path cannot be converted to UTF-8>";
  }
}

void validate_utf8(const std::string& value, const std::string& object) {
  const auto is_continuation = [](unsigned char byte) {
    return byte >= 0x80U && byte <= 0xBFU;
  };

  std::size_t index = 0;
  while (index < value.size()) {
    const unsigned char first =
        static_cast<unsigned char>(value[index]);
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
      throw_writer_error(
          object, "valid UTF-8",
          "an invalid leading byte at byte " + std::to_string(index),
          "store benchmark metadata as UTF-8 before JSON output");
    }

    if (index + length > value.size()) {
      throw_writer_error(
          object, "valid UTF-8",
          "a truncated sequence at byte " + std::to_string(index),
          "store benchmark metadata as UTF-8 before JSON output");
    }
    for (std::size_t offset = 1; offset < length; ++offset) {
      if (!is_continuation(
              static_cast<unsigned char>(value[index + offset]))) {
        throw_writer_error(
            object, "valid UTF-8",
            "an invalid continuation byte at byte " +
                std::to_string(index + offset),
            "store benchmark metadata as UTF-8 before JSON output");
      }
    }

    if (length == 3) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xE0U && second < 0xA0U) ||
          (first == 0xEDU && second > 0x9FU)) {
        throw_writer_error(
            object, "valid UTF-8 without overlong or surrogate sequences",
            "an invalid three-byte sequence at byte " +
                std::to_string(index),
            "store benchmark metadata as canonical UTF-8");
      }
    } else if (length == 4) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xF0U && second < 0x90U) ||
          (first == 0xF4U && second > 0x8FU)) {
        throw_writer_error(
            object, "valid UTF-8 in the Unicode range",
            "an invalid four-byte sequence at byte " +
                std::to_string(index),
            "store benchmark metadata as canonical UTF-8");
      }
    }
    index += length;
  }
}

std::string escape_json_string(const std::string& value,
                               const std::string& object) {
  validate_utf8(value, object);
  static constexpr char kHexDigits[] = "0123456789ABCDEF";

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
          escaped.push_back(kHexDigits[(byte >> 4U) & 0x0FU]);
          escaped.push_back(kHexDigits[byte & 0x0FU]);
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
    throw_writer_error(
        object, "a finite JSON number", "NaN or Infinity",
        "validate benchmark values before serialization");
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

void append_string_array(std::ostringstream& output,
                         const std::vector<std::string>& values,
                         const std::string& object,
                         const std::string& indentation) {
  if (values.empty()) {
    output << "[]";
    return;
  }

  output << "[\n";
  for (std::size_t index = 0; index < values.size(); ++index) {
    output << indentation << "  "
           << escape_json_string(
                  values[index],
                  object + "[" + std::to_string(index) + "]");
    if (index + 1 != values.size()) {
      output << ",";
    }
    output << "\n";
  }
  output << indentation << "]";
}

void append_shape(std::ostringstream& output,
                  const std::vector<std::int64_t>& shape) {
  output << "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << shape[index];
  }
  output << "]";
}

void append_latency_statistics(std::ostringstream& output,
                               const LatencyStatistics& statistics,
                               const std::string& object,
                               const std::string& indentation) {
  output << "{\n"
         << indentation << "  \"sample_count\": "
         << statistics.sample_count << ",\n"
         << indentation << "  \"mean\": "
         << format_double(statistics.mean_ms, object + ".mean") << ",\n"
         << indentation << "  \"p50\": "
         << format_double(statistics.p50_ms, object + ".p50") << ",\n"
         << indentation << "  \"p95\": "
         << format_double(statistics.p95_ms, object + ".p95") << "\n"
         << indentation << "}";
}

std::filesystem::path normalize_cli_path(
    const std::filesystem::path& path, const std::string& object) {
  if (path.empty()) {
    throw_writer_error(
        object, "a non-empty CLI output path", "empty",
        "provide a file path after --benchmark-json");
  }

  std::error_code error;
  std::filesystem::path absolute_path =
      std::filesystem::absolute(path, error);
  if (error) {
    throw_writer_error(
        object, "a path resolvable from the current working directory",
        display_path(path) + " (" + error.message() + ")",
        "check the current working directory and output path syntax");
  }
  absolute_path = absolute_path.lexically_normal();

  // Resolve existing parent links/junctions while preserving the final path
  // component so that an existing final symlink remains detectable below.
  const std::filesystem::path parent = absolute_path.parent_path();
  if (parent.empty()) {
    return absolute_path;
  }
  const std::filesystem::path resolved_parent =
      std::filesystem::weakly_canonical(parent, error);
  if (error) {
    throw_writer_error(
        object,
        "a path whose existing parent components can be resolved",
        display_path(absolute_path) + " (" + error.message() + ")",
        "check parent-directory permissions and link/junction targets");
  }
  return resolved_parent / absolute_path.filename();
}

#ifdef _WIN32
std::wstring lowercase_windows_path(std::wstring value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](wchar_t character) {
                   return static_cast<wchar_t>(std::towlower(character));
                 });
  return value;
}
#endif

bool paths_refer_to_same_location(const std::filesystem::path& lhs,
                                  const std::filesystem::path& rhs) {
  if (lhs == rhs) {
    return true;
  }
#ifdef _WIN32
  if (lowercase_windows_path(lhs.native()) ==
      lowercase_windows_path(rhs.native())) {
    return true;
  }
#endif

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

void validate_target_state(const std::filesystem::path& path,
                           const std::string& object,
                           bool overwrite_existing) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (error && status.type() != std::filesystem::file_type::not_found) {
    throw_writer_error(
        object, "an inspectable output path",
        display_path(path) + " (" + error.message() + ")",
        "check path permissions and filesystem state");
  }
  if (!std::filesystem::exists(status)) {
    return;
  }
  if (std::filesystem::is_symlink(status)) {
    throw_writer_error(
        object, "a missing path or existing regular file",
        display_path(path) + " (symbolic link)",
        "choose a direct regular output path rather than a symlink");
  }
  if (std::filesystem::is_directory(status)) {
    throw_writer_error(
        object, "a file path, not a directory", display_path(path),
        "append a benchmark JSON filename");
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw_writer_error(
        object, "a missing path or existing regular file",
        display_path(path) + " (non-regular filesystem object)",
        "choose a regular output file and do not target devices or special "
        "filesystem objects");
  }
  if (!overwrite_existing) {
    throw_writer_error(
        object, "a path that does not exist",
        display_path(path) + " already exists",
        "choose a new path or explicitly pass --overwrite");
  }
}

void ensure_parent_directory(const std::filesystem::path& output_path,
                             const std::string& object) {
  const std::filesystem::path parent = output_path.parent_path();
  if (parent.empty()) {
    return;
  }

  std::error_code error;
  const bool parent_exists = std::filesystem::exists(parent, error);
  if (error) {
    throw_writer_error(
        object + ".parent", "an inspectable parent directory",
        display_path(parent) + " (" + error.message() + ")",
        "check parent path permissions");
  }
  if (parent_exists) {
    const bool is_directory = std::filesystem::is_directory(parent, error);
    if (error || !is_directory) {
      throw_writer_error(
          object + ".parent", "an existing directory",
          display_path(parent) +
              (error ? " (" + error.message() + ")" : ""),
          "choose a writable directory path");
    }
    return;
  }

  if (!std::filesystem::create_directories(parent, error) && error) {
    throw_writer_error(
        object + ".parent", "a creatable directory",
        display_path(parent) + " (" + error.message() + ")",
        "choose a writable output location");
  }
}

void write_text_file(const std::filesystem::path& path,
                     const std::string& content,
                     const std::string& object) {
  if (content.size() > static_cast<std::size_t>(
                           std::numeric_limits<std::streamsize>::max())) {
    throw_writer_error(
        object, "an output size representable by streamsize",
        std::to_string(content.size()) + " bytes",
        "keep benchmark evidence bounded");
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw_writer_error(
        object, "a writable output file", display_path(path),
        "check directory permissions and whether another process locks it");
  }
  output.write(content.data(),
               static_cast<std::streamsize>(content.size()));
  output.flush();
  if (!output) {
    throw_writer_error(
        object, std::to_string(content.size()) +
                    " bytes written successfully",
        "a filesystem write failure at " + display_path(path),
        "check free space, permissions, and filesystem health");
  }
  output.close();
  if (!output) {
    throw_writer_error(
        object, "a successfully closed output file",
        "a filesystem close failure at " + display_path(path),
        "check free space, permissions, and filesystem health");
  }
}

}  // namespace

std::string serialize_benchmark_json(const BenchmarkResult& result) {
  validate_benchmark_result(result);

  std::ostringstream output;
  output.imbue(std::locale::classic());
  output
      << "{\n"
      << "  \"schema_version\": " << result.schema_version << ",\n"
      << "  \"evidence_type\": "
      << escape_json_string(result.evidence_type, "evidence_type")
      << ",\n"
      << "  \"timestamp_utc\": "
      << escape_json_string(result.timestamp_utc, "timestamp_utc")
      << ",\n"
      << "  \"command\": ";
  append_string_array(output, result.command_arguments, "command", "  ");
  output
      << ",\n"
      << "  \"protocol\": {\n"
      << "    \"batch_size\": " << result.batch_size << ",\n"
      << "    \"sample_count\": " << result.sample_count << ",\n"
      << "    \"warmup\": " << result.warmup << ",\n"
      << "    \"repeat\": " << result.repeat << ",\n"
      << "    \"clock\": \"std::chrono::steady_clock\",\n"
      << "    \"timing_unit\": \"milliseconds\",\n"
      << "    \"percentile_method\": "
         "\"empirical_nearest_rank_ceiling\"\n"
      << "  },\n"
      << "  \"environment\": {\n"
      << "    \"machine\": {\n"
      << "      \"hostname\": "
      << escape_json_string(result.environment.hostname,
                            "environment.machine.hostname")
      << ",\n"
      << "      \"processor\": "
      << escape_json_string(result.environment.processor,
                            "environment.machine.processor")
      << ",\n"
      << "      \"architecture\": "
      << escape_json_string(result.environment.architecture,
                            "environment.machine.architecture")
      << ",\n"
      << "      \"logical_cpu_count\": "
      << result.environment.logical_cpu_count << "\n"
      << "    },\n"
      << "    \"os\": {\n"
      << "      \"name\": "
      << escape_json_string(result.environment.os_name,
                            "environment.os.name")
      << ",\n"
      << "      \"version\": "
      << escape_json_string(result.environment.os_version,
                            "environment.os.version")
      << "\n"
      << "    },\n"
      << "    \"compiler\": {\n"
      << "      \"id\": "
      << escape_json_string(result.environment.compiler_id,
                            "environment.compiler.id")
      << ",\n"
      << "      \"version\": "
      << escape_json_string(result.environment.compiler_version,
                            "environment.compiler.version")
      << "\n"
      << "    },\n"
      << "    \"build\": {\n"
      << "      \"type\": "
      << escape_json_string(result.environment.build_type,
                            "environment.build.type")
      << ",\n"
      << "      \"cxx_standard\": "
      << result.environment.cxx_standard << "\n"
      << "    },\n"
      << "    \"opencv_version\": "
      << escape_json_string(result.environment.opencv_version,
                            "environment.opencv_version")
      << ",\n"
      << "    \"onnxruntime_version\": "
      << escape_json_string(result.environment.onnxruntime_version,
                            "environment.onnxruntime_version")
      << "\n"
      << "  },\n"
      << "  \"runtime\": {\n"
      << "    \"requested_provider\": "
      << escape_json_string(result.runtime.requested_provider,
                            "runtime.requested_provider")
      << ",\n"
      << "    \"actual_provider\": "
      << escape_json_string(result.runtime.actual_provider,
                            "runtime.actual_provider")
      << ",\n"
      << "    \"provider_evidence\": "
      << escape_json_string(result.runtime.provider_evidence,
                            "runtime.provider_evidence")
      << ",\n"
      << "    \"session\": {\n"
      << "      \"execution_mode\": "
      << escape_json_string(result.runtime.execution_mode,
                            "runtime.session.execution_mode")
      << ",\n"
      << "      \"intra_op_num_threads\": "
      << result.runtime.intra_op_num_threads << ",\n"
      << "      \"inter_op_num_threads\": "
      << result.runtime.inter_op_num_threads << ",\n"
      << "      \"graph_optimization_level\": "
      << escape_json_string(
             result.runtime.graph_optimization_level,
             "runtime.session.graph_optimization_level")
      << ",\n"
      << "      \"initialization_ms\": "
      << format_double(result.runtime.session_initialization_ms,
                       "runtime.session.initialization_ms")
      << ",\n"
      << "      \"profiling_enabled\": "
      << (result.runtime.profiling_enabled ? "true" : "false")
      << "\n"
      << "    }\n"
      << "  },\n"
      << "  \"model\": {\n"
      << "    \"model_id\": "
      << escape_json_string(result.model.model_id, "model.model_id")
      << ",\n"
      << "    \"model_family\": "
      << escape_json_string(result.model.model_family,
                            "model.model_family")
      << ",\n"
      << "    \"path\": "
      << escape_json_string(result.model.model_path, "model.path")
      << ",\n"
      << "    \"declared_sha256\": "
      << escape_json_string(result.model.declared_sha256,
                            "model.declared_sha256")
      << ",\n"
      << "    \"file_size_bytes\": " << result.model.file_size_bytes
      << ",\n"
      << "    \"opset\": " << result.model.opset << ",\n"
      << "    \"input\": {\n"
      << "      \"name\": "
      << escape_json_string(result.model.input_name, "model.input.name")
      << ",\n"
      << "      \"shape\": ";
  append_shape(output, result.model.input_shape);
  output
      << ",\n"
      << "      \"dtype\": "
      << escape_json_string(result.model.input_dtype,
                            "model.input.dtype")
      << ",\n"
      << "      \"layout\": "
      << escape_json_string(result.model.input_layout,
                            "model.input.layout")
      << "\n"
      << "    }\n"
      << "  },\n"
      << "  \"sample\": {\n"
      << "    \"image_path\": "
      << escape_json_string(result.sample.image_path,
                            "sample.image_path")
      << ",\n"
      << "    \"file_size_bytes\": " << result.sample.file_size_bytes
      << ",\n"
      << "    \"original_shape\": [" << result.sample.height << ", "
      << result.sample.width << ", " << result.sample.channels << "],\n"
      << "    \"sample_count\": " << result.sample.sample_count << "\n"
      << "  },\n"
      << "  \"postprocess\": {\n"
      << "    \"score_threshold\": "
      << format_double(result.score_threshold,
                       "postprocess.score_threshold")
      << ",\n"
      << "    \"nms_threshold\": "
      << format_double(result.nms_threshold,
                       "postprocess.nms_threshold")
      << ",\n"
      << "    \"nms_mode\": "
      << escape_json_string(result.nms_mode, "postprocess.nms_mode")
      << ",\n"
      << "    \"detection_count\": " << result.sample.detection_count
      << "\n"
      << "  },\n"
      << "  \"latency_ms\": {\n"
      << "    \"image_decode\": ";
  append_latency_statistics(output, result.latency.image_decode,
                            "latency_ms.image_decode", "    ");
  output << ",\n    \"preprocess\": ";
  append_latency_statistics(output, result.latency.preprocess,
                            "latency_ms.preprocess", "    ");
  output << ",\n    \"session_run\": ";
  append_latency_statistics(output, result.latency.session_run,
                            "latency_ms.session_run", "    ");
  output << ",\n    \"postprocess\": ";
  append_latency_statistics(output, result.latency.postprocess,
                            "latency_ms.postprocess", "    ");
  output << ",\n    \"pipeline\": ";
  append_latency_statistics(output, result.latency.pipeline,
                            "latency_ms.pipeline", "    ");
  output << ",\n    \"end_to_end\": ";
  append_latency_statistics(output, result.latency.end_to_end,
                            "latency_ms.end_to_end", "    ");
  output
      << "\n"
      << "  },\n"
      << "  \"throughput_images_per_second\": {\n"
      << "    \"pipeline\": "
      << format_double(
             result.latency.pipeline_throughput_images_per_second,
             "throughput_images_per_second.pipeline")
      << ",\n"
      << "    \"end_to_end\": "
      << format_double(
             result.latency.end_to_end_throughput_images_per_second,
             "throughput_images_per_second.end_to_end")
      << "\n"
      << "  },\n"
      << "  \"memory\": {\n"
      << "    \"status\": "
      << escape_json_string(result.memory.status, "memory.status")
      << ",\n"
      << "    \"metric\": "
      << escape_json_string(result.memory.metric, "memory.metric")
      << ",\n"
      << "    \"bytes\": ";
  if (result.memory.supported) {
    output << result.memory.bytes;
  } else {
    output << "null";
  }
  output << ",\n    \"mebibytes\": ";
  if (result.memory.supported) {
    output << format_double(result.memory.mebibytes, "memory.mebibytes");
  } else {
    output << "null";
  }
  output
      << ",\n"
      << "    \"scope\": "
      << escape_json_string(result.memory.scope, "memory.scope")
      << ",\n"
      << "    \"reason\": ";
  if (result.memory.supported) {
    output << "null\n";
  } else {
    output << escape_json_string(result.memory.reason, "memory.reason")
           << "\n";
  }
  output
      << "  },\n"
      << "  \"timing_exclusions\": ";
  append_string_array(output, result.timing_exclusions,
                      "timing_exclusions", "  ");
  output << ",\n  \"limitations\": ";
  append_string_array(output, result.limitations, "limitations", "  ");
  output << "\n}\n";
  return output.str();
}

std::filesystem::path write_benchmark_json(
    const BenchmarkResult& result,
    const std::filesystem::path& output_path,
    bool overwrite_existing,
    const std::vector<std::filesystem::path>& protected_paths) {
  // Serialize first so invalid evidence cannot create directories or alter an
  // existing file.
  const std::string json = serialize_benchmark_json(result);
  const std::filesystem::path normalized_output =
      normalize_cli_path(output_path, "benchmark_json.path");

  for (std::size_t index = 0; index < protected_paths.size(); ++index) {
    const std::filesystem::path normalized_protected = normalize_cli_path(
        protected_paths[index],
        "protected_paths[" + std::to_string(index) + "]");
    if (paths_refer_to_same_location(normalized_output,
                                     normalized_protected)) {
      throw_writer_error(
          "benchmark_json.path", "a path different from every protected input",
          display_path(normalized_output),
          "choose an output path that cannot overwrite config, artifact, "
          "model, or source image inputs");
    }
  }

  validate_target_state(normalized_output, "benchmark_json.path",
                        overwrite_existing);
  ensure_parent_directory(normalized_output, "benchmark_json.path");
  // Recheck after parent creation to avoid silently accepting a target that
  // appeared while directory creation was in progress.
  validate_target_state(normalized_output, "benchmark_json.path",
                        overwrite_existing);
  write_text_file(normalized_output, json, "benchmark_json.path");
  return normalized_output;
}

}  // namespace yolo_defect_cpp

#include "yolo_defect_cpp/config_loader.h"

#include "key_value_parser.h"

#include <charconv>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

namespace yolo_defect_cpp {
namespace {

constexpr int kCpuSchemaVersion = 1;
constexpr int kTensorRtSchemaVersion = 2;
constexpr const char* kSchemaName = "RuntimeConfig";

const std::set<std::string>& known_fields() {
  static const std::set<std::string> fields = {
      "schema_version", "artifact_spec_path", "score_threshold",
      "nms_threshold", "provider", "device_id", "precision",
      "tensorrt_engine_cache_path",
      "tensorrt_max_workspace_size_bytes", "tensorrt_engine_path",
      "tensorrt_engine_sha256"};
  return fields;
}

void validate_schema_version(const detail::ParsedKeyValueFile& parsed,
                             int version) {
  if (version != kCpuSchemaVersion && version != kTensorRtSchemaVersion) {
    detail::throw_field_error(
        parsed, kSchemaName, "schema_version", "unsupported schema version",
        "one of [1, 2]", std::to_string(version),
        "migrate the runtime config to the supported schema");
  }
}

void validate_threshold(const detail::ParsedKeyValueFile& parsed,
                        const std::string& name,
                        double value) {
  if (value < 0.0 || value > 1.0) {
    detail::throw_field_error(
        parsed, kSchemaName, name, "threshold is outside the valid range",
        "a finite value in [0, 1]", std::to_string(value),
        "choose a threshold between 0 and 1 inclusive");
  }
}

ExecutionProvider parse_provider(
    const detail::ParsedKeyValueFile& parsed, int schema_version) {
  const std::string& value =
      detail::require_field(parsed, kSchemaName, "provider").value;
  if (value == "cpu") {
    return ExecutionProvider::kCpu;
  }
  if (value == "tensorrt" && schema_version == kTensorRtSchemaVersion) {
    return ExecutionProvider::kTensorRt;
  }
  if (value == "tensorrt_native" &&
      schema_version == kTensorRtSchemaVersion) {
    return ExecutionProvider::kTensorRtNative;
  }
  const std::string expected =
      schema_version == kCpuSchemaVersion
          ? "[cpu]"
          : "[cpu, tensorrt, tensorrt_native]";
  detail::throw_field_error(
      parsed, kSchemaName, "provider", "unsupported enum value", expected,
      value,
      schema_version == kCpuSchemaVersion
          ? "use cpu or migrate to RuntimeConfig schema_version = 2 for "
            "the Linux TensorRT path"
          : "select cpu, tensorrt, or tensorrt_native");
}

bool has_field(const detail::ParsedKeyValueFile& parsed,
               const std::string& name) {
  return parsed.fields.find(name) != parsed.fields.end();
}

InferencePrecision parse_precision(
    const detail::ParsedKeyValueFile& parsed) {
  const std::string& value =
      detail::require_field(parsed, kSchemaName, "precision").value;
  if (value == "fp16") {
    return InferencePrecision::kFloat16;
  }
  if (value == "fp32") {
    return InferencePrecision::kFloat32;
  }
  detail::throw_field_error(
      parsed, kSchemaName, "precision", "unsupported enum value",
      "[fp32, fp16]", value,
      "use fp16 for the S2-04 TensorRT acceptance path");
}

std::uint64_t parse_positive_uint64_field(
    const detail::ParsedKeyValueFile& parsed,
    const std::string& name) {
  const std::string& value =
      detail::require_field(parsed, kSchemaName, name).value;
  std::uint64_t result = 0;
  const auto parsed_result =
      std::from_chars(value.data(), value.data() + value.size(), result, 10);
  if (value.empty() || parsed_result.ec != std::errc{} ||
      parsed_result.ptr != value.data() + value.size() || result == 0) {
    detail::throw_field_error(
        parsed, kSchemaName, name, "invalid unsigned integer",
        "a positive base-10 byte count", value,
        "replace it with the TensorRT workspace limit in bytes");
  }
  return result;
}

std::string parse_sha256_field(const detail::ParsedKeyValueFile& parsed,
                               const std::string& name) {
  std::string value =
      detail::require_field(parsed, kSchemaName, name).value;
  const bool all_hex =
      std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isxdigit(character) != 0;
      });
  if (value.size() != 64 || !all_hex) {
    detail::throw_field_error(
        parsed, kSchemaName, name, "invalid SHA-256 declaration",
        "exactly 64 hexadecimal characters", value,
        "copy the SHA-256 of the frozen TensorRT engine bytes");
  }
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::toupper(character));
                 });
  return value;
}

TensorRtProviderConfig parse_tensorrt_config(
    const detail::ParsedKeyValueFile& parsed,
    ExecutionProvider provider) {
  TensorRtProviderConfig result;
  result.device_id =
      detail::parse_integer_field(parsed, kSchemaName, "device_id");
  if (result.device_id < 0) {
    detail::throw_field_error(
        parsed, kSchemaName, "device_id", "device id is negative",
        "an integer >= 0", std::to_string(result.device_id),
        "select a visible NVIDIA GPU index");
  }
  result.precision = parse_precision(parsed);
  result.max_workspace_size_bytes = parse_positive_uint64_field(
      parsed, "tensorrt_max_workspace_size_bytes");
  result.engine_cache_path = detail::resolve_declared_path(
      parsed, kSchemaName, "tensorrt_engine_cache_path");
  if (result.engine_cache_path.filename().empty()) {
    detail::throw_field_error(
        parsed, kSchemaName, "tensorrt_engine_cache_path",
        "cache path has no final directory component",
        "a non-empty dedicated cache directory", result.engine_cache_path.string(),
        "choose a cache directory bound to model/ORT/TensorRT/GPU/precision");
  }
  if (provider == ExecutionProvider::kTensorRtNative) {
    if (result.precision != InferencePrecision::kFloat16) {
      detail::throw_field_error(
          parsed, kSchemaName, "precision",
          "the frozen native TensorRT backend only supports its validated "
          "mixed FP16/FP32 engine policy",
          "fp16", to_string(result.precision),
          "use precision = fp16 with the S2-04 frozen engine or implement "
          "a separately versioned native policy");
    }
    result.native_engine_path = detail::resolve_declared_path(
        parsed, kSchemaName, "tensorrt_engine_path");
    if (result.native_engine_path->filename().empty()) {
      detail::throw_field_error(
          parsed, kSchemaName, "tensorrt_engine_path",
          "engine path has no filename", "a frozen TensorRT engine file",
          result.native_engine_path->string(),
          "point to the precision-constrained engine built by trtexec");
    }
    result.native_engine_sha256 =
        parse_sha256_field(parsed, "tensorrt_engine_sha256");
  } else {
    for (const std::string& field : {
             std::string("tensorrt_engine_path"),
             std::string("tensorrt_engine_sha256")}) {
      if (has_field(parsed, field)) {
        detail::throw_field_error(
            parsed, kSchemaName, field,
            "native TensorRT-only field is present for the ORT TensorRT EP",
            "the field to be absent when provider = tensorrt",
            detail::require_field(parsed, kSchemaName, field).value,
            "remove the native engine field or select provider = "
            "tensorrt_native");
      }
    }
  }
  return result;
}

void validate_provider_specific_fields(
    const detail::ParsedKeyValueFile& parsed,
    const RuntimeConfig& config) {
  const std::set<std::string> tensorrt_fields = {
      "device_id", "precision", "tensorrt_engine_cache_path",
      "tensorrt_max_workspace_size_bytes", "tensorrt_engine_path",
      "tensorrt_engine_sha256"};
  if (config.provider == ExecutionProvider::kTensorRt ||
      config.provider == ExecutionProvider::kTensorRtNative) {
    return;
  }
  for (const std::string& field : tensorrt_fields) {
    if (has_field(parsed, field)) {
      detail::throw_field_error(
          parsed, kSchemaName, field,
          "TensorRT-only field is present for the CPU provider",
          "the field to be absent when provider = cpu",
          detail::require_field(parsed, kSchemaName, field).value,
          "remove TensorRT fields or select a TensorRT provider in schema 2");
    }
  }
}

void validate_artifact_spec_path(
    const detail::ParsedKeyValueFile& parsed,
    const std::filesystem::path& artifact_spec_path) {
  std::error_code error;
  const bool exists = std::filesystem::exists(artifact_spec_path, error);
  if (error || !exists) {
    detail::throw_field_error(
        parsed, kSchemaName, "artifact_spec_path",
        "artifact declaration does not exist", "an existing declaration file",
        artifact_spec_path.string(),
        "correct the path relative to the runtime config file");
  }
  if (!std::filesystem::is_regular_file(artifact_spec_path, error) || error) {
    detail::throw_field_error(
        parsed, kSchemaName, "artifact_spec_path",
        "artifact declaration is not a regular file",
        "a regular artifact declaration file", artifact_spec_path.string(),
        "point artifact_spec_path to the model artifact declaration");
  }
}

}  // namespace

RuntimeConfig load_runtime_config(
    const std::filesystem::path& config_path) {
  const detail::ParsedKeyValueFile parsed =
      detail::parse_key_value_file(config_path, kSchemaName, known_fields());

  RuntimeConfig config;
  config.declaration_path = parsed.declaration_path;
  config.schema_version =
      detail::parse_integer_field(parsed, kSchemaName, "schema_version");
  validate_schema_version(parsed, config.schema_version);
  config.artifact_spec_path = detail::resolve_declared_path(
      parsed, kSchemaName, "artifact_spec_path");
  validate_artifact_spec_path(parsed, config.artifact_spec_path);
  config.score_threshold =
      detail::parse_number_field(parsed, kSchemaName, "score_threshold");
  config.nms_threshold =
      detail::parse_number_field(parsed, kSchemaName, "nms_threshold");
  validate_threshold(parsed, "score_threshold", config.score_threshold);
  validate_threshold(parsed, "nms_threshold", config.nms_threshold);
  config.provider = parse_provider(parsed, config.schema_version);
  if (config.provider == ExecutionProvider::kTensorRt ||
      config.provider == ExecutionProvider::kTensorRtNative) {
    config.tensorrt = parse_tensorrt_config(parsed, config.provider);
  }
  validate_provider_specific_fields(parsed, config);
  return config;
}

RuntimeContract load_runtime_contract(
    const std::filesystem::path& config_path) {
  RuntimeContract contract;
  contract.runtime = load_runtime_config(config_path);
  contract.artifact =
      load_model_artifact_spec(contract.runtime.artifact_spec_path);
  return contract;
}

std::string to_string(ExecutionProvider value) {
  switch (value) {
    case ExecutionProvider::kCpu:
      return "cpu";
    case ExecutionProvider::kTensorRt:
      return "tensorrt";
    case ExecutionProvider::kTensorRtNative:
      return "tensorrt_native";
  }
  throw std::logic_error("Unknown ExecutionProvider enum value.");
}

std::string to_string(InferencePrecision value) {
  switch (value) {
    case InferencePrecision::kFloat32:
      return "fp32";
    case InferencePrecision::kFloat16:
      return "fp16";
  }
  throw std::logic_error("Unknown InferencePrecision enum value.");
}

}  // namespace yolo_defect_cpp

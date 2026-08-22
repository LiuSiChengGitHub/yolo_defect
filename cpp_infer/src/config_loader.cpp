#include "yolo_defect_cpp/config_loader.h"

#include "key_value_parser.h"

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

namespace yolo_defect_cpp {
namespace {

constexpr int kSupportedSchemaVersion = 1;
constexpr const char* kSchemaName = "RuntimeConfig";

const std::set<std::string>& known_fields() {
  static const std::set<std::string> fields = {
      "schema_version", "artifact_spec_path", "score_threshold",
      "nms_threshold", "provider"};
  return fields;
}

void validate_schema_version(const detail::ParsedKeyValueFile& parsed,
                             int version) {
  if (version != kSupportedSchemaVersion) {
    detail::throw_field_error(
        parsed, kSchemaName, "schema_version", "unsupported schema version",
        std::to_string(kSupportedSchemaVersion), std::to_string(version),
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
    const detail::ParsedKeyValueFile& parsed) {
  const std::string& value =
      detail::require_field(parsed, kSchemaName, "provider").value;
  if (value != "cpu") {
    detail::throw_field_error(
        parsed, kSchemaName, "provider", "unsupported enum value", "[cpu]",
        value,
        "use cpu for the pinned Windows x64 CPU ONNX Runtime SDK");
  }
  return ExecutionProvider::kCpu;
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
  config.provider = parse_provider(parsed);
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
  }
  throw std::logic_error("Unknown ExecutionProvider enum value.");
}

}  // namespace yolo_defect_cpp

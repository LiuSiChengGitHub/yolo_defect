#ifndef YOLO_DEFECT_CPP_CONFIG_LOADER_H_
#define YOLO_DEFECT_CPP_CONFIG_LOADER_H_

#include "yolo_defect_cpp/artifact_spec.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace yolo_defect_cpp {

enum class ExecutionProvider {
  kCpu,
  kTensorRt,
  kTensorRtNative,
};

enum class InferencePrecision {
  kFloat32,
  kFloat16,
};

struct TensorRtProviderConfig {
  int device_id = 0;
  InferencePrecision precision = InferencePrecision::kFloat16;
  std::uint64_t max_workspace_size_bytes = 0;
  std::filesystem::path engine_cache_path;
  std::optional<std::filesystem::path> native_engine_path;
  std::optional<std::string> native_engine_sha256;
};

struct RuntimeConfig {
  int schema_version = 0;
  std::filesystem::path declaration_path;
  std::filesystem::path artifact_spec_path;
  double score_threshold = 0.0;
  double nms_threshold = 0.0;
  ExecutionProvider provider = ExecutionProvider::kCpu;
  std::optional<TensorRtProviderConfig> tensorrt;
};

struct RuntimeContract {
  RuntimeConfig runtime;
  ModelArtifactSpec artifact;
};

RuntimeConfig load_runtime_config(
    const std::filesystem::path& config_path);

RuntimeContract load_runtime_contract(
    const std::filesystem::path& config_path);

std::string to_string(ExecutionProvider value);
std::string to_string(InferencePrecision value);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_CONFIG_LOADER_H_

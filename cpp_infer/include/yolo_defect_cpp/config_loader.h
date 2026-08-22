#ifndef YOLO_DEFECT_CPP_CONFIG_LOADER_H_
#define YOLO_DEFECT_CPP_CONFIG_LOADER_H_

#include "yolo_defect_cpp/artifact_spec.h"

#include <filesystem>
#include <string>

namespace yolo_defect_cpp {

enum class ExecutionProvider {
  kCpu,
};

struct RuntimeConfig {
  int schema_version = 0;
  std::filesystem::path declaration_path;
  std::filesystem::path artifact_spec_path;
  double score_threshold = 0.0;
  double nms_threshold = 0.0;
  ExecutionProvider provider = ExecutionProvider::kCpu;
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

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_CONFIG_LOADER_H_

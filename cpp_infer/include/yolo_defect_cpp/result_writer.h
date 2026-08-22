#ifndef YOLO_DEFECT_CPP_RESULT_WRITER_H_
#define YOLO_DEFECT_CPP_RESULT_WRITER_H_

#include "yolo_defect_cpp/detection_result.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct DetectionOutputRequest {
  std::optional<std::filesystem::path> json_path;
  std::optional<std::filesystem::path> image_path;
  bool overwrite_existing = false;
};

struct WrittenDetectionOutputs {
  std::optional<std::filesystem::path> json_path;
  std::optional<std::filesystem::path> image_path;
};

std::string serialize_detection_json(
    const SingleImageDetectionResult& result);

WrittenDetectionOutputs write_detection_outputs(
    const SingleImageDetectionResult& result,
    const DetectionOutputRequest& request,
    const std::vector<std::filesystem::path>& protected_paths);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_RESULT_WRITER_H_

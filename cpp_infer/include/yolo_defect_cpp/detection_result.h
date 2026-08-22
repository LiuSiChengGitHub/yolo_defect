#ifndef YOLO_DEFECT_CPP_DETECTION_RESULT_H_
#define YOLO_DEFECT_CPP_DETECTION_RESULT_H_

#include "yolo_defect_cpp/artifact_spec.h"
#include "yolo_defect_cpp/postprocessor.h"

#include <filesystem>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct DetectionImageMetadata {
  std::filesystem::path source_path;
  int original_width = 0;
  int original_height = 0;
  int original_channels = 0;
  int input_width = 0;
  int input_height = 0;
};

struct SingleImageDetectionResult {
  int schema_version = 1;
  std::string model_id;
  std::string declared_model_sha256;

  // Kept out of the JSON document. The writer uses this copy to verify that
  // every Detection still agrees with the artifact contract.
  std::vector<std::string> class_names;

  DetectionImageMetadata image;
  std::string actual_provider;
  std::string provider_evidence;
  double score_threshold = 0.0;
  double nms_threshold = 0.0;
  NmsMode nms_mode = NmsMode::kClassAgnostic;
  std::vector<Detection> detections;
};

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_DETECTION_RESULT_H_

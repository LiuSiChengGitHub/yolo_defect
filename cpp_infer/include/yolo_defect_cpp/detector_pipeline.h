#ifndef YOLO_DEFECT_CPP_DETECTOR_PIPELINE_H_
#define YOLO_DEFECT_CPP_DETECTOR_PIPELINE_H_

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/detection_result.h"
#include "yolo_defect_cpp/model_metadata.h"
#include "yolo_defect_cpp/result_writer.h"

#include <filesystem>
#include <memory>

namespace yolo_defect_cpp {

struct SingleImagePipelineResult {
  SingleImageDetectionResult detection_result;
  WrittenDetectionOutputs outputs;
};

class DetectorPipeline {
 public:
  explicit DetectorPipeline(RuntimeContract contract);
  ~DetectorPipeline();

  DetectorPipeline(const DetectorPipeline&) = delete;
  DetectorPipeline& operator=(const DetectorPipeline&) = delete;
  DetectorPipeline(DetectorPipeline&&) noexcept;
  DetectorPipeline& operator=(DetectorPipeline&&) noexcept;

  SingleImagePipelineResult run(
      const std::filesystem::path& image_path,
      const DetectionOutputRequest& output_request);

  const ModelMetadata& metadata() const;
  double session_initialization_ms() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_DETECTOR_PIPELINE_H_

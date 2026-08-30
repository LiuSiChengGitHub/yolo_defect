#ifndef YOLO_DEFECT_CPP_IMAGE_PREPROCESSOR_H_
#define YOLO_DEFECT_CPP_IMAGE_PREPROCESSOR_H_

#include "yolo_defect_cpp/artifact_spec.h"
#include "yolo_defect_cpp/project_core.h"

#include <filesystem>

namespace cv {
class Mat;
}

namespace yolo_defect_cpp {

PreprocessResult preprocess_image(const std::filesystem::path& image_path,
                                   const ModelArtifactSpec& artifact);

PreprocessResult preprocess_image(const cv::Mat& bgr_image,
                                   const ModelArtifactSpec& artifact);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_IMAGE_PREPROCESSOR_H_

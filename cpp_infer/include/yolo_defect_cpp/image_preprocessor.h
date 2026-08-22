#ifndef YOLO_DEFECT_CPP_IMAGE_PREPROCESSOR_H_
#define YOLO_DEFECT_CPP_IMAGE_PREPROCESSOR_H_

#include "yolo_defect_cpp/artifact_spec.h"

#include <string>
#include <vector>

namespace cv {
class Mat;
}

namespace yolo_defect_cpp {

struct PreprocessResult {
  int original_width = 0;
  int original_height = 0;
  int original_channels = 0;
  int input_width = 0;
  int input_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  int pad_left = 0;
  int pad_top = 0;
  int pad_right = 0;
  int pad_bottom = 0;
  double scale = 0.0;
  std::vector<float> tensor_nchw;
};

PreprocessResult preprocess_image(const std::string& image_path,
                                   const ModelArtifactSpec& artifact);

PreprocessResult preprocess_image(const cv::Mat& bgr_image,
                                   const ModelArtifactSpec& artifact);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_IMAGE_PREPROCESSOR_H_

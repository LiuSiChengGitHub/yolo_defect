#ifndef YOLO_DEFECT_CPP_IMAGE_DECODER_H_
#define YOLO_DEFECT_CPP_IMAGE_DECODER_H_

#include <filesystem>

#include <opencv2/core/mat.hpp>

namespace yolo_defect_cpp {
namespace internal {

struct DecodedBgrImage {
  cv::Mat image;
  // Measures encoded-file reading plus cv::imdecode. Path normalization and
  // decoded-image validation deliberately sit outside this interval.
  double imread_ms = 0.0;
};

void initialize_image_decoder_logging();

std::filesystem::path normalize_image_file(
    const std::filesystem::path& declared_path);

DecodedBgrImage decode_normalized_bgr_image(
    const std::filesystem::path& normalized_path);

}  // namespace internal
}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_IMAGE_DECODER_H_

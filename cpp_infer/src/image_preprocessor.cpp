#include "yolo_defect_cpp/image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo_defect_cpp {
namespace {

constexpr int kLetterboxPadValue = 114;

[[noreturn]] void throw_preprocess_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Preprocess validation failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

void validate_input_spec(const ModelArtifactSpec& artifact) {
  if (artifact.input.layout != TensorLayout::kNchw ||
      artifact.input.dtype != TensorDataType::kFloat32 ||
      artifact.input.shape.size() != 4 || artifact.input.shape[0] != 1 ||
      artifact.input.shape[1] != 3) {
    throw_preprocess_error(
        "artifact.input", "float32 NCHW [1,3,height,width]",
        "layout=" + to_string(artifact.input.layout) + ", dtype=" +
            to_string(artifact.input.dtype) + ", shape rank=" +
            std::to_string(artifact.input.shape.size()),
        "fix the ModelArtifactSpec input tensor declaration");
  }
  const std::int64_t height = artifact.input.shape[2];
  const std::int64_t width = artifact.input.shape[3];
  const std::int64_t maximum_int =
      static_cast<std::int64_t>(std::numeric_limits<int>::max());
  if (height <= 0 || width <= 0 ||
      height > maximum_int || width > maximum_int) {
    throw_preprocess_error(
        "artifact.input.shape[2:4]",
        "positive height/width representable by OpenCV int",
        std::to_string(height) + "x" + std::to_string(width),
        "use valid static model input dimensions");
  }
}

std::vector<float> to_nchw_tensor(const cv::Mat& rgb_float) {
  const std::size_t height = static_cast<std::size_t>(rgb_float.rows);
  const std::size_t width = static_cast<std::size_t>(rgb_float.cols);
  const std::size_t channels =
      static_cast<std::size_t>(rgb_float.channels());
  if (channels != 3) {
    throw std::runtime_error("Preprocess expects a 3-channel RGB image.");
  }
  if (height != 0 && width >
          std::numeric_limits<std::size_t>::max() / height) {
    throw_preprocess_error(
        "tensor_nchw.plane_elements", "a size representable by size_t",
        std::to_string(height) + "x" + std::to_string(width),
        "reduce the model input dimensions");
  }
  const std::size_t channel_stride = height * width;
  if (channel_stride != 0 && channels >
          std::numeric_limits<std::size_t>::max() / channel_stride) {
    throw_preprocess_error(
        "tensor_nchw.elements", "a size representable by size_t",
        std::to_string(channels) + "x" +
            std::to_string(channel_stride),
        "reduce the model input dimensions");
  }

  std::vector<float> tensor(channels * channel_stride);
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const cv::Vec3f pixel = rgb_float.at<cv::Vec3f>(
          static_cast<int>(y), static_cast<int>(x));
      for (std::size_t c = 0; c < channels; ++c) {
        tensor[c * channel_stride + y * width + x] =
            pixel[static_cast<int>(c)];
      }
    }
  }
  return tensor;
}

}  // namespace

PreprocessResult preprocess_image(const std::string& image_path,
                                   const ModelArtifactSpec& artifact) {
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

  cv::Mat bgr_image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr_image.empty()) {
    throw std::runtime_error("Failed to read image file: " + image_path);
  }

  return preprocess_image(bgr_image, artifact);
}

PreprocessResult preprocess_image(const cv::Mat& bgr_image,
                                   const ModelArtifactSpec& artifact) {
  validate_input_spec(artifact);
  if (bgr_image.empty()) {
    throw std::runtime_error(
        "Preprocess received an empty cv::Mat; expected a non-empty "
        "3-channel BGR image. Action: verify synthetic image construction "
        "or image decoding before preprocess.");
  }
  if (bgr_image.type() != CV_8UC3) {
    throw_preprocess_error(
        "bgr_image.type", "CV_8UC3",
        "type=" + std::to_string(bgr_image.type()) + ", depth=" +
            std::to_string(bgr_image.depth()) + ", channels=" +
            std::to_string(bgr_image.channels()),
        "convert the source image to an 8-bit 3-channel BGR matrix");
  }

  PreprocessResult result;
  result.original_width = bgr_image.cols;
  result.original_height = bgr_image.rows;
  result.original_channels = bgr_image.channels();
  result.input_width = static_cast<int>(artifact.input.shape[3]);
  result.input_height = static_cast<int>(artifact.input.shape[2]);

  const double scale_x =
      static_cast<double>(result.input_width) / result.original_width;
  const double scale_y =
      static_cast<double>(result.input_height) / result.original_height;
  result.scale = std::min(scale_x, scale_y);
  result.resized_width =
      std::max(1, static_cast<int>(std::round(result.original_width * result.scale)));
  result.resized_height =
      std::max(1, static_cast<int>(std::round(result.original_height * result.scale)));

  cv::Mat resized_bgr;
  cv::resize(bgr_image, resized_bgr,
             cv::Size(result.resized_width, result.resized_height));

  const int pad_width = result.input_width - result.resized_width;
  const int pad_height = result.input_height - result.resized_height;
  result.pad_left = pad_width / 2;
  result.pad_right = pad_width - result.pad_left;
  result.pad_top = pad_height / 2;
  result.pad_bottom = pad_height - result.pad_top;

  cv::Mat letterboxed_bgr;
  cv::copyMakeBorder(resized_bgr, letterboxed_bgr,
                     result.pad_top, result.pad_bottom,
                     result.pad_left, result.pad_right,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(kLetterboxPadValue, kLetterboxPadValue,
                                kLetterboxPadValue));

  cv::Mat rgb_image;
  cv::cvtColor(letterboxed_bgr, rgb_image, cv::COLOR_BGR2RGB);

  cv::Mat rgb_float;
  rgb_image.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);
  result.tensor_nchw = to_nchw_tensor(rgb_float);

  return result;
}

}  // namespace yolo_defect_cpp

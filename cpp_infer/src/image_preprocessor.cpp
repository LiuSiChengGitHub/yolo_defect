#include "yolo_defect_cpp/image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo_defect_cpp {
namespace {

constexpr int kLetterboxPadValue = 114;

void validate_input_size(const RuntimeConfig& config) {
  if (config.input_width <= 0 || config.input_height <= 0) {
    throw std::runtime_error("Preprocess input size must be greater than 0.");
  }
}

std::vector<float> to_nchw_tensor(const cv::Mat& rgb_float) {
  const int height = rgb_float.rows;
  const int width = rgb_float.cols;
  const int channels = rgb_float.channels();
  if (channels != 3) {
    throw std::runtime_error("Preprocess expects a 3-channel RGB image.");
  }

  std::vector<float> tensor(static_cast<std::size_t>(channels) * height * width);
  const int channel_stride = height * width;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const cv::Vec3f pixel = rgb_float.at<cv::Vec3f>(y, x);
      for (int c = 0; c < channels; ++c) {
        tensor[static_cast<std::size_t>(c * channel_stride + y * width + x)] =
            pixel[c];
      }
    }
  }
  return tensor;
}

}  // namespace

PreprocessResult preprocess_image(const std::string& image_path,
                                  const RuntimeConfig& config) {
  validate_input_size(config);
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

  cv::Mat bgr_image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr_image.empty()) {
    throw std::runtime_error("Failed to read image file: " + image_path);
  }

  PreprocessResult result;
  result.original_width = bgr_image.cols;
  result.original_height = bgr_image.rows;
  result.original_channels = bgr_image.channels();
  result.input_width = config.input_width;
  result.input_height = config.input_height;

  const double scale_x =
      static_cast<double>(config.input_width) / result.original_width;
  const double scale_y =
      static_cast<double>(config.input_height) / result.original_height;
  result.scale = std::min(scale_x, scale_y);
  result.resized_width =
      std::max(1, static_cast<int>(std::round(result.original_width * result.scale)));
  result.resized_height =
      std::max(1, static_cast<int>(std::round(result.original_height * result.scale)));

  cv::Mat resized_bgr;
  cv::resize(bgr_image, resized_bgr,
             cv::Size(result.resized_width, result.resized_height));

  const int pad_width = config.input_width - result.resized_width;
  const int pad_height = config.input_height - result.resized_height;
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

#include "image_decoder.h"

#include <chrono>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>

namespace yolo_defect_cpp {
namespace internal {
namespace {

[[noreturn]] void throw_decode_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Image decode validation failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

}  // namespace

void initialize_image_decoder_logging() {
  static std::once_flag log_level_once;
  std::call_once(log_level_once, []() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  });
}

std::filesystem::path normalize_image_file(
    const std::filesystem::path& declared_path) {
  if (declared_path.empty()) {
    throw_decode_error(
        "image_path", "a non-empty path to a readable color image", "empty",
        "provide --image <path> and verify the current working directory");
  }

  std::error_code error;
  std::filesystem::path absolute_path =
      std::filesystem::absolute(declared_path, error);
  if (error) {
    throw_decode_error(
        "image_path", "a path resolvable from the current working directory",
        "'" + declared_path.string() + "' (" + error.message() + ")",
        "correct --image or use an absolute image path");
  }
  absolute_path = absolute_path.lexically_normal();

  const bool exists = std::filesystem::exists(absolute_path, error);
  if (error || !exists) {
    throw_decode_error(
        "image_path='" + absolute_path.string() + "'",
        "an existing readable regular image file",
        error ? "filesystem inspection failed: " + error.message()
              : "path does not exist",
        "correct --image or restore the fixed benchmark sample");
  }
  const bool regular = std::filesystem::is_regular_file(absolute_path, error);
  if (error || !regular) {
    throw_decode_error(
        "image_path='" + absolute_path.string() + "'", "a regular image file",
        error ? "filesystem inspection failed: " + error.message()
              : "path exists but is not a regular file",
        "point --image to a readable encoded image file");
  }

  std::filesystem::path canonical_path =
      std::filesystem::canonical(absolute_path, error);
  if (error) {
    throw_decode_error(
        "image_path='" + absolute_path.string() + "'",
        "an image path that can be canonicalized", error.message(),
        "check the source path, link target, and filesystem permissions");
  }
  return canonical_path;
}

DecodedBgrImage decode_normalized_bgr_image(
    const std::filesystem::path& normalized_path) {
  DecodedBgrImage decoded;
  try {
    const auto decode_start = std::chrono::steady_clock::now();
    decoded.image = cv::imread(normalized_path.string(), cv::IMREAD_COLOR);
    const auto decode_end = std::chrono::steady_clock::now();
    decoded.imread_ms =
        std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();
  } catch (const cv::Exception& error) {
    throw_decode_error(
        "image_path='" + normalized_path.string() + "'",
        "an OpenCV-decodable 8-bit color image",
        "OpenCV decoder exception: " + std::string(error.what()),
        "check that the file is undamaged and uses a codec available in this "
        "OpenCV build");
  }

  if (!std::isfinite(decoded.imread_ms) || decoded.imread_ms < 0.0) {
    throw_decode_error(
        "image_decode.duration", "a finite non-negative steady-clock duration",
        std::to_string(decoded.imread_ms),
        "verify the platform clock before publishing benchmark evidence");
  }
  if (decoded.image.empty()) {
    throw_decode_error(
        "image_path='" + normalized_path.string() + "'",
        "an OpenCV-decodable 8-bit color image",
        "file exists but OpenCV decoding returned an empty image",
        "check for damaged or unsupported image content, then retry with a "
        "known-good image");
  }
  if (decoded.image.type() != CV_8UC3) {
    throw_decode_error(
        "decoded_image.type", "CV_8UC3",
        "type=" + std::to_string(decoded.image.type()) + ", channels=" +
            std::to_string(decoded.image.channels()),
        "use OpenCV IMREAD_COLOR and a supported 8-bit image codec");
  }
  return decoded;
}

}  // namespace internal
}  // namespace yolo_defect_cpp

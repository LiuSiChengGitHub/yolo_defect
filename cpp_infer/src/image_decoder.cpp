#include "image_decoder.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

std::string display_path(const std::filesystem::path& path) {
  try {
    return path.generic_u8string();
  } catch (const std::exception&) {
    return "<path cannot be converted to UTF-8>";
  }
}

std::vector<unsigned char> read_encoded_image(
    const std::filesystem::path& path) {
  std::error_code error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error) {
    throw_decode_error(
        "image_path='" + display_path(path) + "'",
        "a readable non-empty encoded image file",
        "filesystem size inspection failed: " + error.message(),
        "check the image path and read permission");
  }
  if (file_size == 0 ||
      file_size > static_cast<std::uintmax_t>(
                      std::numeric_limits<std::size_t>::max()) ||
      file_size > static_cast<std::uintmax_t>(
                      std::numeric_limits<std::streamsize>::max())) {
    throw_decode_error(
        "image_path='" + display_path(path) + "'",
        "a non-empty encoded image whose size fits memory",
        std::to_string(file_size) + " bytes",
        "check whether the image is empty or unreasonably large");
  }

  // The filesystem::path overload preserves the native wide path on Windows;
  // OpenCV's narrow-string imread overload cannot reliably open such paths.
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw_decode_error(
        "image_path='" + display_path(path) + "'",
        "a readable encoded image file", "file open failed",
        "check the image path and read permission");
  }

  std::vector<unsigned char> bytes(static_cast<std::size_t>(file_size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.gcount() !=
                    static_cast<std::streamsize>(bytes.size())) {
    throw_decode_error(
        "image_path='" + display_path(path) + "'",
        std::to_string(bytes.size()) + " readable bytes",
        std::to_string(input.gcount()) + " bytes read",
        "check whether the image changed while it was being decoded");
  }
  return bytes;
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
        "'" + display_path(declared_path) + "' (" + error.message() + ")",
        "correct --image or use an absolute image path");
  }
  absolute_path = absolute_path.lexically_normal();

  const bool exists = std::filesystem::exists(absolute_path, error);
  if (error || !exists) {
    throw_decode_error(
        "image_path='" + display_path(absolute_path) + "'",
        "an existing readable regular image file",
        error ? "filesystem inspection failed: " + error.message()
              : "path does not exist",
        "correct --image or restore the fixed benchmark sample");
  }
  const bool regular = std::filesystem::is_regular_file(absolute_path, error);
  if (error || !regular) {
    throw_decode_error(
        "image_path='" + display_path(absolute_path) + "'",
        "a regular image file",
        error ? "filesystem inspection failed: " + error.message()
              : "path exists but is not a regular file",
        "point --image to a readable encoded image file");
  }

  std::filesystem::path canonical_path =
      std::filesystem::canonical(absolute_path, error);
  if (error) {
    throw_decode_error(
        "image_path='" + display_path(absolute_path) + "'",
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
    const std::vector<unsigned char> encoded =
        read_encoded_image(normalized_path);
    decoded.image = cv::imdecode(encoded, cv::IMREAD_COLOR);
    const auto decode_end = std::chrono::steady_clock::now();
    decoded.imread_ms =
        std::chrono::duration<double, std::milli>(
            decode_end - decode_start).count();
  } catch (const cv::Exception& error) {
    throw_decode_error(
        "image_path='" + display_path(normalized_path) + "'",
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
        "image_path='" + display_path(normalized_path) + "'",
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

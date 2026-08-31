#include "yolo_defect_cpp/result_writer.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iterator>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

[[noreturn]] void throw_output_error(
    const std::string& object,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  throw std::runtime_error(
      "Detection output error for '" + object + "': expected " +
      expected + "; actual " + actual + "; action: " + action + ".");
}

std::string display_path(const std::filesystem::path& path) {
  try {
    return path.u8string();
  } catch (const std::exception&) {
    return "<path cannot be converted to UTF-8>";
  }
}

void validate_utf8(const std::string& value, const std::string& object) {
  const auto continuation = [](unsigned char byte) {
    return byte >= 0x80U && byte <= 0xBFU;
  };

  std::size_t index = 0;
  while (index < value.size()) {
    const unsigned char first =
        static_cast<unsigned char>(value[index]);
    std::size_t length = 0;
    if (first <= 0x7FU) {
      length = 1;
    } else if (first >= 0xC2U && first <= 0xDFU) {
      length = 2;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      length = 3;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      length = 4;
    } else {
      throw_output_error(
          object, "valid UTF-8", "an invalid leading byte at byte " +
              std::to_string(index),
          "store contract and path strings as UTF-8 before JSON output");
    }

    if (index + length > value.size()) {
      throw_output_error(
          object, "valid UTF-8", "a truncated sequence at byte " +
              std::to_string(index),
          "store contract and path strings as UTF-8 before JSON output");
    }
    for (std::size_t offset = 1; offset < length; ++offset) {
      if (!continuation(
              static_cast<unsigned char>(value[index + offset]))) {
        throw_output_error(
            object, "valid UTF-8", "an invalid continuation byte at byte " +
                std::to_string(index + offset),
            "store contract and path strings as UTF-8 before JSON output");
      }
    }

    if (length == 3) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xE0U && second < 0xA0U) ||
          (first == 0xEDU && second > 0x9FU)) {
        throw_output_error(
            object, "valid UTF-8 without overlong or surrogate sequences",
            "an invalid three-byte sequence at byte " +
                std::to_string(index),
            "store contract and path strings as canonical UTF-8");
      }
    } else if (length == 4) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xF0U && second < 0x90U) ||
          (first == 0xF4U && second > 0x8FU)) {
        throw_output_error(
            object, "valid UTF-8 in the Unicode range",
            "an invalid four-byte sequence at byte " +
                std::to_string(index),
            "store contract and path strings as canonical UTF-8");
      }
    }
    index += length;
  }
}

std::string path_to_utf8(const std::filesystem::path& path,
                         const std::string& object) {
  std::string value;
  try {
    value = path.u8string();
  } catch (const std::exception& error) {
    throw_output_error(
        object, "a path convertible to UTF-8", error.what(),
        "use a valid Unicode filesystem path");
  }
  validate_utf8(value, object);
  return value;
}

std::string escape_json_string(const std::string& value,
                               const std::string& object) {
  validate_utf8(value, object);
  static constexpr char kHexDigits[] = "0123456789ABCDEF";

  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (unsigned char byte : value) {
    switch (byte) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          escaped += "\\u00";
          escaped.push_back(kHexDigits[(byte >> 4U) & 0x0FU]);
          escaped.push_back(kHexDigits[byte & 0x0FU]);
        } else {
          escaped.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  escaped.push_back('"');
  return escaped;
}

std::string format_double(double value, const std::string& object) {
  if (!std::isfinite(value)) {
    throw_output_error(
        object, "a finite JSON number", "NaN or Infinity",
        "validate runtime numeric values before serialization");
  }
  if (value == 0.0) {
    value = 0.0;
  }
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return output.str();
}

std::string format_float(float value, const std::string& object) {
  if (!std::isfinite(value)) {
    throw_output_error(
        object, "a finite JSON number", "NaN or Infinity",
        "validate detection numeric values before serialization");
  }
  if (value == 0.0F) {
    value = 0.0F;
  }
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(std::numeric_limits<float>::max_digits10)
         << value;
  return output.str();
}

bool is_sha256(const std::string& value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(
      value.begin(), value.end(), [](unsigned char character) {
        return std::isxdigit(character) != 0;
      });
}

void validate_result(const SingleImageDetectionResult& result) {
  if (result.schema_version != 1) {
    throw_output_error(
        "schema_version", "1", std::to_string(result.schema_version),
        "use the S1-05 single-image detection JSON schema");
  }
  if (result.model_id.empty()) {
    throw_output_error(
        "model.model_id", "a non-empty model identifier", "empty",
        "copy ModelArtifactSpec.model_id into the pipeline result");
  }
  validate_utf8(result.model_id, "model.model_id");
  if (!is_sha256(result.declared_model_sha256)) {
    throw_output_error(
        "model.declared_sha256", "64 hexadecimal characters",
        result.declared_model_sha256.empty()
            ? "empty"
            : result.declared_model_sha256,
        "copy the declared ModelArtifactSpec SHA-256 without claiming a "
        "runtime re-hash");
  }

  if (result.class_names.empty()) {
    throw_output_error(
        "class_names", "at least one non-empty class", "[]",
        "copy the validated artifact class list into the result");
  }
  std::unordered_set<std::string> unique_class_names;
  for (std::size_t index = 0; index < result.class_names.size(); ++index) {
    const std::string object =
        "class_names[" + std::to_string(index) + "]";
    if (result.class_names[index].empty()) {
      throw_output_error(
          object, "a non-empty class name", "empty",
          "use the validated ModelArtifactSpec class list");
    }
    validate_utf8(result.class_names[index], object);
    if (!unique_class_names.insert(result.class_names[index]).second) {
      throw_output_error(
          object, "a unique class name", result.class_names[index],
          "remove duplicate artifact classes before inference");
    }
  }

  if (result.image.source_path.empty()) {
    throw_output_error(
        "image.path", "a non-empty source image path", "empty",
        "copy the single-image CLI input path into the result");
  }
  path_to_utf8(result.image.source_path, "image.path");
  if (result.image.original_width <= 0 ||
      result.image.original_height <= 0 ||
      result.image.original_channels != 3) {
    throw_output_error(
        "image.original_size", "positive width/height and 3 channels",
        std::to_string(result.image.original_width) + "x" +
            std::to_string(result.image.original_height) + "x" +
            std::to_string(result.image.original_channels),
        "copy metadata from the validated CV_8UC3 preprocess result");
  }
  if (result.image.input_width <= 0 || result.image.input_height <= 0) {
    throw_output_error(
        "image.input_size", "positive width and height",
        std::to_string(result.image.input_width) + "x" +
            std::to_string(result.image.input_height),
        "copy the artifact input size from the preprocess result");
  }

  if (result.actual_provider.empty()) {
    throw_output_error(
        "runtime.actual_provider", "a non-empty session provider", "empty",
        "copy OnnxRunner metadata while the runner is alive");
  }
  validate_utf8(result.actual_provider, "runtime.actual_provider");
  if (result.provider_evidence.empty()) {
    throw_output_error(
        "runtime.provider_evidence", "non-empty provider evidence", "empty",
        "copy OnnxRunner provider evidence while the runner is alive");
  }
  validate_utf8(result.provider_evidence, "runtime.provider_evidence");

  const auto validate_threshold = [](double value,
                                     const std::string& object) {
    if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
      throw_output_error(
          object, "a finite value in [0,1]",
          std::isfinite(value) ? format_double(value, object)
                               : "NaN or Infinity",
          "copy the validated RuntimeConfig threshold");
    }
  };
  validate_threshold(result.score_threshold, "runtime.score_threshold");
  validate_threshold(result.nms_threshold, "runtime.nms_threshold");
  if (result.nms_mode != NmsMode::kClassAgnostic) {
    throw_output_error(
        "runtime.nms_mode", "class_agnostic", "unknown enum value",
        "copy the validated ModelArtifactSpec NMS mode");
  }

  const float maximum_x = static_cast<float>(result.image.original_width);
  const float maximum_y = static_cast<float>(result.image.original_height);
  const float effective_score_threshold =
      static_cast<float>(result.score_threshold);
  for (std::size_t index = 0; index < result.detections.size(); ++index) {
    const Detection& detection = result.detections[index];
    const std::string prefix =
        "detections[" + std::to_string(index) + "]";
    if (detection.class_id < 0 ||
        static_cast<std::size_t>(detection.class_id) >=
            result.class_names.size()) {
      throw_output_error(
          prefix + ".class_id",
          "an index in [0," +
              std::to_string(result.class_names.size() - 1) + "]",
          std::to_string(detection.class_id),
          "use detections decoded with the same artifact class list");
    }
    const std::string& expected_name =
        result.class_names[static_cast<std::size_t>(detection.class_id)];
    if (detection.class_name != expected_name) {
      throw_output_error(
          prefix + ".class_name", expected_name, detection.class_name,
          "keep class_id and class_name from the same decoded candidate");
    }
    validate_utf8(detection.class_name, prefix + ".class_name");
    if (!std::isfinite(detection.confidence) ||
        detection.confidence < 0.0F || detection.confidence > 1.0F) {
      throw_output_error(
          prefix + ".confidence", "a finite value in [0,1]",
          std::isfinite(detection.confidence)
              ? format_float(detection.confidence,
                             prefix + ".confidence")
              : "NaN or Infinity",
          "inspect YOLO class-score decode before output");
    }
    if (!(detection.confidence > effective_score_threshold)) {
      throw_output_error(
          prefix + ".confidence",
          "a float32 confidence strictly greater than score_threshold " +
              format_float(effective_score_threshold,
                           "runtime.score_threshold"),
          format_float(detection.confidence, prefix + ".confidence"),
          "serialize only candidates retained by strict confidence "
          "filtering");
    }
    if (index > 0 &&
        result.detections[index - 1].confidence < detection.confidence) {
      throw_output_error(
          prefix + ".confidence",
          "detections in non-increasing confidence order",
          format_float(result.detections[index - 1].confidence,
                       "detections[" + std::to_string(index - 1) +
                           "].confidence") +
              " followed by " +
              format_float(detection.confidence,
                           prefix + ".confidence"),
          "preserve the stable class-agnostic NMS output order");
    }

    const BoundingBox& box = detection.bbox_xyxy;
    const std::array<float, 4> coordinates = {
        box.x1, box.y1, box.x2, box.y2};
    for (std::size_t coordinate = 0; coordinate < coordinates.size();
         ++coordinate) {
      if (!std::isfinite(coordinates[coordinate])) {
        throw_output_error(
            prefix + ".bbox_xyxy[" + std::to_string(coordinate) + "]",
            "a finite original-image coordinate", "NaN or Infinity",
            "inspect coordinate restore and clip before output");
      }
    }
    if (box.x1 > box.x2 || box.y1 > box.y2) {
      throw_output_error(
          prefix + ".bbox_xyxy", "x1 <= x2 and y1 <= y2",
          "[" + format_float(box.x1, prefix) + "," +
              format_float(box.y1, prefix) + "," +
              format_float(box.x2, prefix) + "," +
              format_float(box.y2, prefix) + "]",
          "inspect xywh conversion and restored box ordering");
    }
    if (box.x1 < 0.0F || box.y1 < 0.0F || box.x2 > maximum_x ||
        box.y2 > maximum_y) {
      throw_output_error(
          prefix + ".bbox_xyxy",
          "coordinates clipped to [0,original_width] and "
          "[0,original_height]",
          "[" + format_float(box.x1, prefix) + "," +
              format_float(box.y1, prefix) + "," +
              format_float(box.x2, prefix) + "," +
              format_float(box.y2, prefix) + "]",
          "run letterbox restore and clip exactly once before output");
    }
  }
}

std::filesystem::path normalize_cli_path(
    const std::filesystem::path& path,
    const std::string& object) {
  if (path.empty()) {
    throw_output_error(
        object, "a non-empty CLI output path", "empty",
        "provide a file path after the output option");
  }
  std::error_code error;
  const std::filesystem::path absolute_path =
      std::filesystem::absolute(path, error);
  if (error) {
    throw_output_error(
        object, "a path resolvable from the current working directory",
        display_path(path) + " (" + error.message() + ")",
        "check the current working directory and output path syntax");
  }
  const std::filesystem::path normalized_path =
      absolute_path.lexically_normal();
  const std::filesystem::path resolved_path =
      std::filesystem::weakly_canonical(normalized_path, error);
  if (error) {
    throw_output_error(
        object,
        "a path whose existing parent components can be resolved",
        display_path(normalized_path) + " (" + error.message() + ")",
        "check parent-directory permissions and link/junction targets");
  }
  return resolved_path;
}

#ifdef _WIN32
std::wstring lowercase_windows_path(std::wstring value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](wchar_t character) {
                   return static_cast<wchar_t>(std::towlower(character));
                 });
  return value;
}
#endif

bool paths_refer_to_same_location(
    const std::filesystem::path& lhs,
    const std::filesystem::path& rhs) {
  if (lhs == rhs) {
    return true;
  }
#ifdef _WIN32
  if (lowercase_windows_path(lhs.native()) ==
      lowercase_windows_path(rhs.native())) {
    return true;
  }
#endif

  std::error_code lhs_error;
  std::error_code rhs_error;
  const bool lhs_exists = std::filesystem::exists(lhs, lhs_error);
  const bool rhs_exists = std::filesystem::exists(rhs, rhs_error);
  if (lhs_error || rhs_error || !lhs_exists || !rhs_exists) {
    return false;
  }
  std::error_code equivalent_error;
  const bool equivalent =
      std::filesystem::equivalent(lhs, rhs, equivalent_error);
  return !equivalent_error && equivalent;
}

bool path_is_inside_protected_directory(
    const std::filesystem::path& output,
    const std::filesystem::path& protected_path) {
  std::error_code error;
  if (!std::filesystem::is_directory(protected_path, error) || error) {
    return false;
  }
  const std::filesystem::path relative =
      output.lexically_relative(protected_path);
  if (relative.empty() || relative.is_absolute()) {
    return false;
  }
  for (const std::filesystem::path& component : relative) {
    if (component == "..") {
      return false;
    }
  }
  return true;
}

void validate_target_state(const std::filesystem::path& path,
                           const std::string& object,
                           bool overwrite_existing) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (error && status.type() != std::filesystem::file_type::not_found) {
    throw_output_error(
        object, "an inspectable output path",
        display_path(path) + " (" + error.message() + ")",
        "check path permissions and filesystem state");
  }
  if (!std::filesystem::exists(status)) {
    return;
  }
  if (std::filesystem::is_directory(status)) {
    throw_output_error(
        object, "a file path, not a directory", display_path(path),
        "append an output filename");
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw_output_error(
        object, "a missing path or existing regular file",
        display_path(path) + " (non-regular filesystem object)",
        "choose a regular output file and do not target symlinks or devices");
  }
  if (!overwrite_existing) {
    throw_output_error(
        object, "a path that does not exist", display_path(path) +
            " already exists",
        "choose a new path or explicitly enable overwrite_existing");
  }
}

void ensure_parent_directory(const std::filesystem::path& output_path,
                             const std::string& object) {
  const std::filesystem::path parent = output_path.parent_path();
  if (parent.empty()) {
    return;
  }
  std::error_code error;
  if (std::filesystem::exists(parent, error)) {
    if (error || !std::filesystem::is_directory(parent, error) || error) {
      throw_output_error(
          object + ".parent", "an existing directory",
          display_path(parent) +
              (error ? " (" + error.message() + ")" : ""),
          "choose a writable directory path");
    }
    return;
  }
  if (error) {
    throw_output_error(
        object + ".parent", "an inspectable parent directory",
        display_path(parent) + " (" + error.message() + ")",
        "check parent path permissions");
  }
  if (!std::filesystem::create_directories(parent, error) && error) {
    throw_output_error(
        object + ".parent", "a creatable directory",
        display_path(parent) + " (" + error.message() + ")",
        "choose a writable output location");
  }
}

std::vector<unsigned char> read_binary_file(
    const std::filesystem::path& path,
    const std::string& object) {
  std::error_code error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error) {
    throw_output_error(
        object, "a readable regular image file",
        display_path(path) + " (" + error.message() + ")",
        "check the source image path and permissions");
  }
  if (file_size == 0 ||
      file_size > static_cast<std::uintmax_t>(
                      std::numeric_limits<std::size_t>::max()) ||
      file_size > static_cast<std::uintmax_t>(
                      std::numeric_limits<std::streamsize>::max())) {
    throw_output_error(
        object, "a non-empty image whose size fits memory",
        std::to_string(file_size) + " bytes",
        "check whether the image is empty or unreasonably large");
  }

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw_output_error(
        object, "a readable image file", display_path(path),
        "check the source path and read permission");
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(file_size));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.gcount() !=
                    static_cast<std::streamsize>(bytes.size())) {
    throw_output_error(
        object, std::to_string(bytes.size()) + " readable bytes",
        std::to_string(input.gcount()) + " bytes read",
        "check whether the source image changed during output generation");
  }
  return bytes;
}

std::string make_label(const Detection& detection) {
  std::string class_name = detection.class_name;
  for (char& character : class_name) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (byte < 0x20U || byte == 0x7FU) {
      character = '?';
    }
  }
  std::ostringstream label;
  label.imbue(std::locale::classic());
  label << class_name << " " << std::fixed << std::setprecision(3)
        << detection.confidence;
  return label.str();
}

cv::Scalar detection_color(int class_id) {
  static const std::array<cv::Scalar, 8> kPalette = {
      cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255),
      cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
      cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72),
      cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61)};
  return kPalette[static_cast<std::size_t>(class_id) % kPalette.size()];
}

std::vector<unsigned char> render_visualization(
    const SingleImageDetectionResult& result,
    const std::filesystem::path& output_path) {
  const std::filesystem::path source_path = normalize_cli_path(
      result.image.source_path, "image.path");
  const std::vector<unsigned char> source_bytes =
      read_binary_file(source_path, "image.path");

  cv::Mat image;
  try {
    image = cv::imdecode(source_bytes, cv::IMREAD_COLOR);
  } catch (const cv::Exception& error) {
    throw_output_error(
        "image.path", "an OpenCV-decodable color image",
        display_path(source_path) + " (" + error.what() + ")",
        "check that the source file is a supported, undamaged image");
  }
  if (image.empty()) {
    throw_output_error(
        "image.path", "an OpenCV-decodable color image",
        display_path(source_path) + " decoded to an empty cv::Mat",
        "check that the source file is a supported, undamaged image");
  }
  if (image.cols != result.image.original_width ||
      image.rows != result.image.original_height || image.channels() != 3) {
    throw_output_error(
        "image.decoded_metadata",
        std::to_string(result.image.original_width) + "x" +
            std::to_string(result.image.original_height) + "x3",
        std::to_string(image.cols) + "x" + std::to_string(image.rows) +
            "x" + std::to_string(image.channels()),
        "use the same unchanged source image that produced the detections");
  }

  const int maximum_x = image.cols - 1;
  const int maximum_y = image.rows - 1;
  constexpr int kThickness = 2;
  constexpr int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double kFontScale = 0.5;
  for (const Detection& detection : result.detections) {
    const BoundingBox& box = detection.bbox_xyxy;
    const int x1 = std::clamp(
        static_cast<int>(std::lround(box.x1)), 0, maximum_x);
    const int y1 = std::clamp(
        static_cast<int>(std::lround(box.y1)), 0, maximum_y);
    const int x2 = std::clamp(
        static_cast<int>(std::lround(box.x2)), 0, maximum_x);
    const int y2 = std::clamp(
        static_cast<int>(std::lround(box.y2)), 0, maximum_y);
    const cv::Scalar color = detection_color(detection.class_id);
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color,
                  kThickness, cv::LINE_8);

    const std::string label = make_label(detection);
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        label, kFontFace, kFontScale, 1, &baseline);
    const int text_x = x1;
    const int text_y = std::clamp(
        std::max(text_size.height + 4, y1 - 4), 0, maximum_y);
    const int background_right =
        std::min(maximum_x, text_x + text_size.width + 4);
    const int background_top =
        std::max(0, text_y - text_size.height - 3);
    const int background_bottom =
        std::min(maximum_y, text_y + baseline + 2);
    cv::rectangle(image, cv::Point(text_x, background_top),
                  cv::Point(background_right, background_bottom), color,
                  cv::FILLED, cv::LINE_8);
    cv::putText(image, label,
                cv::Point(std::min(maximum_x, text_x + 2), text_y),
                kFontFace, kFontScale, cv::Scalar(255, 255, 255), 1,
                cv::LINE_8);
  }

  std::string extension =
      path_to_utf8(output_path.extension(), "output.image_path.extension");
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  if (extension.empty()) {
    throw_output_error(
        "output.image_path.extension",
        "an OpenCV-supported image extension such as .jpg or .png", "empty",
        "append a supported extension to --output-image");
  }

  std::vector<int> parameters;
  if (extension == ".jpg" || extension == ".jpeg") {
    parameters = {cv::IMWRITE_JPEG_QUALITY, 95};
  } else if (extension == ".png") {
    parameters = {cv::IMWRITE_PNG_COMPRESSION, 3};
  } else if (extension == ".webp") {
    parameters = {cv::IMWRITE_WEBP_QUALITY, 95};
  }

  std::vector<unsigned char> encoded;
  try {
    if (!cv::imencode(extension, image, encoded, parameters) ||
        encoded.empty()) {
      throw_output_error(
          "output.image_path", "a successful non-empty OpenCV encoding",
          display_path(output_path),
          "use an extension supported by the installed OpenCV imgcodecs");
    }
  } catch (const cv::Exception& error) {
    throw_output_error(
        "output.image_path", "an OpenCV-supported image extension",
        extension + " (" + error.what() + ")",
        "use .jpg, .png, or another codec available in this OpenCV build");
  }
  return encoded;
}

void write_binary_file(const std::filesystem::path& path,
                       const unsigned char* data,
                       std::size_t size,
                       const std::string& object) {
  if (size > static_cast<std::size_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    throw_output_error(
        object, "an output size representable by streamsize",
        std::to_string(size) + " bytes",
        "choose a normal single-image output");
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw_output_error(
        object, "a writable output file", display_path(path),
        "check directory permissions and whether another process locks it");
  }
  output.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(size));
  output.flush();
  if (!output) {
    throw_output_error(
        object, std::to_string(size) + " bytes written successfully",
        "a filesystem write failure at " + display_path(path),
        "check free space, permissions, and filesystem health");
  }
  output.close();
  if (!output) {
    throw_output_error(
        object, "a successfully closed output file",
        "a filesystem close failure at " + display_path(path),
        "check free space, permissions, and filesystem health");
  }
}

}  // namespace

std::string serialize_detection_json(
    const SingleImageDetectionResult& result) {
  validate_result(result);

  const std::string image_path =
      path_to_utf8(result.image.source_path, "image.path");
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output
      << "{\n"
      << "  \"schema_version\": " << result.schema_version << ",\n"
      << "  \"model\": {\n"
      << "    \"model_id\": "
      << escape_json_string(result.model_id, "model.model_id") << ",\n"
      << "    \"declared_sha256\": "
      << escape_json_string(result.declared_model_sha256,
                            "model.declared_sha256")
      << "\n"
      << "  },\n"
      << "  \"image\": {\n"
      << "    \"path\": " << escape_json_string(image_path, "image.path")
      << ",\n"
      << "    \"original_size\": {\"width\": "
      << result.image.original_width << ", \"height\": "
      << result.image.original_height << ", \"channels\": "
      << result.image.original_channels << "},\n"
      << "    \"input_size\": {\"width\": " << result.image.input_width
      << ", \"height\": " << result.image.input_height << "}\n"
      << "  },\n"
      << "  \"runtime\": {\n"
      << "    \"actual_provider\": "
      << escape_json_string(result.actual_provider,
                            "runtime.actual_provider")
      << ",\n"
      << "    \"provider_evidence\": "
      << escape_json_string(result.provider_evidence,
                            "runtime.provider_evidence")
      << ",\n"
      << "    \"score_threshold\": "
      << format_double(result.score_threshold, "runtime.score_threshold")
      << ",\n"
      << "    \"nms_threshold\": "
      << format_double(result.nms_threshold, "runtime.nms_threshold")
      << ",\n"
      << "    \"nms_mode\": "
      << escape_json_string(to_string(result.nms_mode), "runtime.nms_mode")
      << "\n"
      << "  },\n";

  if (result.detections.empty()) {
    output << "  \"detections\": []\n";
  } else {
    output << "  \"detections\": [\n";
    for (std::size_t index = 0; index < result.detections.size(); ++index) {
      const Detection& detection = result.detections[index];
      const std::string prefix =
          "detections[" + std::to_string(index) + "]";
      output
          << "    {\n"
          << "      \"class_id\": " << detection.class_id << ",\n"
          << "      \"class_name\": "
          << escape_json_string(detection.class_name,
                                prefix + ".class_name")
          << ",\n"
          << "      \"confidence\": "
          << format_float(detection.confidence, prefix + ".confidence")
          << ",\n"
          << "      \"bbox_xyxy\": ["
          << format_float(detection.bbox_xyxy.x1,
                          prefix + ".bbox_xyxy[0]")
          << ", "
          << format_float(detection.bbox_xyxy.y1,
                          prefix + ".bbox_xyxy[1]")
          << ", "
          << format_float(detection.bbox_xyxy.x2,
                          prefix + ".bbox_xyxy[2]")
          << ", "
          << format_float(detection.bbox_xyxy.y2,
                          prefix + ".bbox_xyxy[3]")
          << "]\n"
          << "    }";
      if (index + 1 != result.detections.size()) {
        output << ",";
      }
      output << "\n";
    }
    output << "  ]\n";
  }
  output << "}\n";
  return output.str();
}

WrittenDetectionOutputs write_detection_outputs(
    const SingleImageDetectionResult& result,
    const DetectionOutputRequest& request,
    const std::vector<std::filesystem::path>& protected_paths) {
  validate_result(result);
  if (!request.json_path.has_value() && !request.image_path.has_value()) {
    throw_output_error(
        "output.request", "at least one JSON or image output path", "none",
        "provide --output-json, --output-image, or both");
  }

  WrittenDetectionOutputs written;
  if (request.json_path.has_value()) {
    written.json_path = normalize_cli_path(*request.json_path,
                                           "output.json_path");
  }
  if (request.image_path.has_value()) {
    written.image_path = normalize_cli_path(*request.image_path,
                                            "output.image_path");
  }
  if (written.json_path.has_value() && written.image_path.has_value() &&
      paths_refer_to_same_location(*written.json_path,
                                   *written.image_path)) {
    throw_output_error(
        "output.paths", "different JSON and image output files",
        display_path(*written.json_path),
        "choose separate --output-json and --output-image paths");
  }

  std::vector<std::filesystem::path> normalized_protected_paths;
  normalized_protected_paths.reserve(protected_paths.size() + 1);
  normalized_protected_paths.push_back(normalize_cli_path(
      result.image.source_path, "protected_paths.source_image"));
  for (std::size_t index = 0; index < protected_paths.size(); ++index) {
    normalized_protected_paths.push_back(normalize_cli_path(
        protected_paths[index],
        "protected_paths[" + std::to_string(index) + "]"));
  }

  const auto validate_not_protected =
      [&normalized_protected_paths](const std::filesystem::path& output_path,
                                    const std::string& object) {
        for (const std::filesystem::path& protected_path :
             normalized_protected_paths) {
          if (paths_refer_to_same_location(output_path, protected_path) ||
              path_is_inside_protected_directory(output_path,
                                                 protected_path)) {
            throw_output_error(
                object, "a path different from every protected input",
                display_path(output_path),
                "choose an output path that cannot overwrite config, "
                "artifact, model, source image, native engine, or any "
                "TensorRT cache content");
          }
        }
      };

  if (written.json_path.has_value()) {
    validate_not_protected(*written.json_path, "output.json_path");
    validate_target_state(*written.json_path, "output.json_path",
                          request.overwrite_existing);
  }
  if (written.image_path.has_value()) {
    validate_not_protected(*written.image_path, "output.image_path");
    validate_target_state(*written.image_path, "output.image_path",
                          request.overwrite_existing);
  }

  std::optional<std::string> json_document;
  std::optional<std::vector<unsigned char>> visualization;
  if (written.json_path.has_value()) {
    json_document = serialize_detection_json(result);
  }
  if (written.image_path.has_value()) {
    visualization = render_visualization(result, *written.image_path);
  }

  if (written.json_path.has_value()) {
    ensure_parent_directory(*written.json_path, "output.json_path");
  }
  if (written.image_path.has_value()) {
    ensure_parent_directory(*written.image_path, "output.image_path");
  }

  // Re-check immediately before opening to reduce accidental overwrite risk
  // if another process created a target while inference/encoding completed.
  if (written.json_path.has_value()) {
    validate_target_state(*written.json_path, "output.json_path",
                          request.overwrite_existing);
    const auto* data = reinterpret_cast<const unsigned char*>(
        json_document->data());
    write_binary_file(*written.json_path, data, json_document->size(),
                      "output.json_path");
  }
  if (written.image_path.has_value()) {
    validate_target_state(*written.image_path, "output.image_path",
                          request.overwrite_existing);
    write_binary_file(*written.image_path, visualization->data(),
                      visualization->size(), "output.image_path");
  }
  return written;
}

}  // namespace yolo_defect_cpp

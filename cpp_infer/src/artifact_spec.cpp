#include "yolo_defect_cpp/artifact_spec.h"

#include "key_value_parser.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>

namespace yolo_defect_cpp {
namespace {

constexpr int kSupportedSchemaVersion = 1;
constexpr const char* kSchemaName = "ModelArtifactSpec";

const std::set<std::string>& known_fields() {
  static const std::set<std::string> fields = {
      "schema_version",  "model_id",       "model_family",
      "model_path",      "model_sha256",   "opset",
      "source",          "provenance",     "artifact_license",
      "input_name",      "input_shape",    "input_dtype",
      "input_layout",    "output_name",    "output_shape",
      "output_dtype",    "output_layout",  "class_names",
      "preprocess_type", "postprocess_type", "nms_mode"};
  return fields;
}

const detail::ParsedField& field(
    const detail::ParsedKeyValueFile& parsed,
    const std::string& name) {
  return detail::require_field(parsed, kSchemaName, name);
}

void validate_schema_version(const detail::ParsedKeyValueFile& parsed,
                             int version) {
  if (version != kSupportedSchemaVersion) {
    detail::throw_field_error(
        parsed, kSchemaName, "schema_version", "unsupported schema version",
        std::to_string(kSupportedSchemaVersion), std::to_string(version),
        "migrate the artifact declaration to the supported schema");
  }
}

void validate_positive_integer(const detail::ParsedKeyValueFile& parsed,
                               const std::string& name,
                               int value) {
  if (value <= 0) {
    detail::throw_field_error(parsed, kSchemaName, name,
                              "value must be positive", "an integer > 0",
                              std::to_string(value),
                              "replace it with a positive integer");
  }
}

template <typename Enum>
Enum parse_enum(const detail::ParsedKeyValueFile& parsed,
                const std::string& name,
                const std::string& supported_value,
                Enum result) {
  const std::string& value = field(parsed, name).value;
  if (value != supported_value) {
    detail::throw_field_error(
        parsed, kSchemaName, name, "unsupported enum value",
        "[" + supported_value + "]", value,
        "choose a value supported by schema v1");
  }
  return result;
}

std::string normalize_sha256(const detail::ParsedKeyValueFile& parsed) {
  std::string sha256 = field(parsed, "model_sha256").value;
  const bool all_hex =
      std::all_of(sha256.begin(), sha256.end(), [](unsigned char character) {
        return std::isxdigit(character) != 0;
      });
  if (sha256.size() != 64 || !all_hex) {
    detail::throw_field_error(
        parsed, kSchemaName, "model_sha256", "invalid SHA-256 declaration",
        "exactly 64 hexadecimal characters", sha256,
        "replace it with the artifact's complete SHA-256 digest");
  }
  std::transform(sha256.begin(), sha256.end(), sha256.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::toupper(character));
                 });
  return sha256;
}

void validate_model_path(const detail::ParsedKeyValueFile& parsed,
                         const std::filesystem::path& model_path) {
  std::error_code error;
  const bool exists = std::filesystem::exists(model_path, error);
  if (error || !exists) {
    detail::throw_field_error(
        parsed, kSchemaName, "model_path", "model artifact does not exist",
        "an existing regular ONNX file", model_path.string(),
        "correct the path relative to the artifact declaration file");
  }
  if (!std::filesystem::is_regular_file(model_path, error) || error) {
    detail::throw_field_error(
        parsed, kSchemaName, "model_path",
        "model artifact is not a regular file", "a regular ONNX file",
        model_path.string(), "point model_path to the ONNX model file");
  }
}

void validate_class_names(const detail::ParsedKeyValueFile& parsed,
                          const std::vector<std::string>& class_names) {
  std::set<std::string> unique_names;
  for (const std::string& class_name : class_names) {
    if (!unique_names.insert(class_name).second) {
      detail::throw_field_error(
          parsed, kSchemaName, "class_names", "duplicate class name",
          "unique class names in model channel order", class_name,
          "remove or rename the duplicate while preserving model class order");
    }
  }
}

void validate_input_shape(const detail::ParsedKeyValueFile& parsed,
                          const std::vector<std::int64_t>& shape) {
  if (shape.size() != 4) {
    detail::throw_field_error(
        parsed, kSchemaName, "input_shape", "invalid input rank",
        "NCHW rank 4: 1,3,height,width", field(parsed, "input_shape").value,
        "declare the complete static YOLOv8 input shape");
  }
  if (shape[0] != 1 || shape[1] != 3) {
    detail::throw_field_error(
        parsed, kSchemaName, "input_shape", "unsupported input dimensions",
        "batch=1 and channels=3 in NCHW order",
        field(parsed, "input_shape").value,
        "set the first two dimensions to 1,3");
  }
  if (shape[2] > INT_MAX || shape[3] > INT_MAX) {
    detail::throw_field_error(
        parsed, kSchemaName, "input_shape", "input size exceeds C++ image limits",
        "height and width <= INT_MAX", field(parsed, "input_shape").value,
        "use deployable image dimensions");
  }
}

void validate_output_shape(const detail::ParsedKeyValueFile& parsed,
                           const std::vector<std::int64_t>& shape,
                           std::size_t class_count) {
  if (shape.size() != 3) {
    detail::throw_field_error(
        parsed, kSchemaName, "output_shape", "invalid output rank",
        "BCN rank 3: 1,4+class_count,prediction_count",
        field(parsed, "output_shape").value,
        "declare the complete static YOLOv8 raw-output shape");
  }
  const std::int64_t expected_channels =
      4 + static_cast<std::int64_t>(class_count);
  if (shape[0] != 1 || shape[1] != expected_channels) {
    detail::throw_field_error(
        parsed, kSchemaName, "output_shape",
        "output shape is inconsistent with class_names",
        "batch=1 and channels=4+class_count=" +
            std::to_string(expected_channels),
        field(parsed, "output_shape").value,
        "fix output_shape or class_names to match the exported model");
  }
}

}  // namespace

ModelArtifactSpec load_model_artifact_spec(
    const std::filesystem::path& artifact_spec_path) {
  const detail::ParsedKeyValueFile parsed =
      detail::parse_key_value_file(artifact_spec_path, kSchemaName,
                                   known_fields());

  ModelArtifactSpec spec;
  spec.declaration_path = parsed.declaration_path;
  spec.schema_version =
      detail::parse_integer_field(parsed, kSchemaName, "schema_version");
  validate_schema_version(parsed, spec.schema_version);
  spec.model_id = field(parsed, "model_id").value;
  spec.model_family =
      parse_enum(parsed, "model_family", "yolov8", ModelFamily::kYoloV8);
  spec.model_path =
      detail::resolve_declared_path(parsed, kSchemaName, "model_path");
  validate_model_path(parsed, spec.model_path);
  spec.model_sha256 = normalize_sha256(parsed);
  spec.opset = detail::parse_integer_field(parsed, kSchemaName, "opset");
  validate_positive_integer(parsed, "opset", spec.opset);
  spec.source = field(parsed, "source").value;
  spec.provenance = field(parsed, "provenance").value;
  spec.artifact_license = field(parsed, "artifact_license").value;

  spec.input.name = field(parsed, "input_name").value;
  spec.input.shape =
      detail::parse_shape_field(parsed, kSchemaName, "input_shape");
  spec.input.dtype = parse_enum(parsed, "input_dtype", "float32",
                                TensorDataType::kFloat32);
  spec.input.layout =
      parse_enum(parsed, "input_layout", "nchw", TensorLayout::kNchw);

  spec.output.name = field(parsed, "output_name").value;
  spec.output.shape =
      detail::parse_shape_field(parsed, kSchemaName, "output_shape");
  spec.output.dtype = parse_enum(parsed, "output_dtype", "float32",
                                 TensorDataType::kFloat32);
  spec.output.layout =
      parse_enum(parsed, "output_layout", "bcn", TensorLayout::kBcn);

  spec.class_names =
      detail::parse_list_field(parsed, kSchemaName, "class_names");
  validate_class_names(parsed, spec.class_names);
  spec.preprocess_type = parse_enum(
      parsed, "preprocess_type", "letterbox_rgb_0_1_nchw",
      PreprocessType::kLetterboxRgbZeroToOneNchw);
  spec.postprocess_type = parse_enum(parsed, "postprocess_type", "yolov8_raw",
                                     PostprocessType::kYoloV8Raw);
  spec.nms_mode = parse_enum(parsed, "nms_mode", "class_agnostic",
                             NmsMode::kClassAgnostic);

  validate_input_shape(parsed, spec.input.shape);
  validate_output_shape(parsed, spec.output.shape, spec.class_names.size());
  return spec;
}

std::string to_string(ModelFamily value) {
  switch (value) {
    case ModelFamily::kYoloV8:
      return "yolov8";
  }
  throw std::logic_error("Unknown ModelFamily enum value.");
}

std::string to_string(TensorDataType value) {
  switch (value) {
    case TensorDataType::kFloat32:
      return "float32";
  }
  throw std::logic_error("Unknown TensorDataType enum value.");
}

std::string to_string(TensorLayout value) {
  switch (value) {
    case TensorLayout::kNchw:
      return "nchw";
    case TensorLayout::kBcn:
      return "bcn";
  }
  throw std::logic_error("Unknown TensorLayout enum value.");
}

std::string to_string(PreprocessType value) {
  switch (value) {
    case PreprocessType::kLetterboxRgbZeroToOneNchw:
      return "letterbox_rgb_0_1_nchw";
  }
  throw std::logic_error("Unknown PreprocessType enum value.");
}

std::string to_string(PostprocessType value) {
  switch (value) {
    case PostprocessType::kYoloV8Raw:
      return "yolov8_raw";
  }
  throw std::logic_error("Unknown PostprocessType enum value.");
}

std::string to_string(NmsMode value) {
  switch (value) {
    case NmsMode::kClassAgnostic:
      return "class_agnostic";
  }
  throw std::logic_error("Unknown NmsMode enum value.");
}

}  // namespace yolo_defect_cpp

#ifndef YOLO_DEFECT_CPP_ARTIFACT_SPEC_H_
#define YOLO_DEFECT_CPP_ARTIFACT_SPEC_H_

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

enum class ModelFamily {
  kYoloV8,
};

enum class TensorDataType {
  kFloat32,
};

enum class TensorLayout {
  kNchw,
  kBcn,
};

enum class PreprocessType {
  kLetterboxRgbZeroToOneNchw,
};

enum class PostprocessType {
  kYoloV8Raw,
};

enum class NmsMode {
  kClassAgnostic,
};

struct TensorSpec {
  std::string name;
  std::vector<std::int64_t> shape;
  TensorDataType dtype = TensorDataType::kFloat32;
  TensorLayout layout = TensorLayout::kNchw;
};

struct ModelArtifactSpec {
  int schema_version = 0;
  std::filesystem::path declaration_path;
  std::string model_id;
  ModelFamily model_family = ModelFamily::kYoloV8;
  std::filesystem::path model_path;
  std::string model_sha256;
  int opset = 0;
  std::string source;
  std::string provenance;
  std::string artifact_license;
  TensorSpec input;
  TensorSpec output;
  std::vector<std::string> class_names;
  PreprocessType preprocess_type =
      PreprocessType::kLetterboxRgbZeroToOneNchw;
  PostprocessType postprocess_type = PostprocessType::kYoloV8Raw;
  NmsMode nms_mode = NmsMode::kClassAgnostic;
};

ModelArtifactSpec load_model_artifact_spec(
    const std::filesystem::path& artifact_spec_path);

std::string to_string(ModelFamily value);
std::string to_string(TensorDataType value);
std::string to_string(TensorLayout value);
std::string to_string(PreprocessType value);
std::string to_string(PostprocessType value);
std::string to_string(NmsMode value);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_ARTIFACT_SPEC_H_

#ifndef YOLO_DEFECT_CPP_MODEL_METADATA_H_
#define YOLO_DEFECT_CPP_MODEL_METADATA_H_

#include "yolo_defect_cpp/config_loader.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

enum class ModelValueType {
  kTensor,
  kSequence,
  kMap,
  kOptional,
  kUnknown,
};

enum class ObservedTensorDataType {
  kUndefined,
  kFloat32,
  kUInt8,
  kInt8,
  kUInt16,
  kInt16,
  kInt32,
  kInt64,
  kString,
  kBool,
  kFloat16,
  kFloat64,
  kUInt32,
  kUInt64,
  kComplex64,
  kComplex128,
  kBFloat16,
  kUnsupported,
};

struct TensorMetadata {
  std::string name;
  ModelValueType value_type = ModelValueType::kUnknown;
  ObservedTensorDataType dtype = ObservedTensorDataType::kUndefined;
  std::vector<std::int64_t> shape;
};

struct ModelMetadata {
  std::string ort_version;
  std::vector<std::string> available_providers;
  std::string session_provider;
  std::string provider_evidence;
  std::vector<std::string> registered_provider_chain;
  std::string inference_precision;
  int device_id = -1;
  bool engine_cache_enabled = false;
  std::string engine_cache_path;
  std::string engine_cache_prefix;
  std::string engine_cache_state;
  std::size_t engine_cache_files_before = 0;
  std::size_t engine_cache_files_after = 0;
  int intra_op_num_threads = 0;
  int inter_op_num_threads = 0;
  std::string execution_mode;
  std::string graph_optimization_level;
  std::vector<TensorMetadata> inputs;
  std::vector<TensorMetadata> outputs;
};

void validate_model_metadata(const ModelMetadata& actual,
                             const RuntimeContract& expected);

std::string to_string(ModelValueType value);
std::string to_string(ObservedTensorDataType value);
std::string format_shape(const std::vector<std::int64_t>& shape);
std::string format_string_list(const std::vector<std::string>& values);

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_MODEL_METADATA_H_

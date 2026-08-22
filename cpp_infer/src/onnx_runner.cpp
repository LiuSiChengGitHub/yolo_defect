#include "yolo_defect_cpp/onnx_runner.h"

#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

constexpr int kIntraOpNumThreads = 1;
constexpr int kInterOpNumThreads = 1;
constexpr const char* kCpuProviderName = "CPUExecutionProvider";

ModelValueType convert_value_type(ONNXType value) {
  switch (value) {
    case ONNX_TYPE_TENSOR:
      return ModelValueType::kTensor;
    case ONNX_TYPE_SEQUENCE:
      return ModelValueType::kSequence;
    case ONNX_TYPE_MAP:
      return ModelValueType::kMap;
    case ONNX_TYPE_OPTIONAL:
      return ModelValueType::kOptional;
    case ONNX_TYPE_UNKNOWN:
    case ONNX_TYPE_OPAQUE:
    case ONNX_TYPE_SPARSETENSOR:
      return ModelValueType::kUnknown;
  }
  return ModelValueType::kUnknown;
}

ObservedTensorDataType convert_tensor_dtype(
    ONNXTensorElementDataType value) {
  switch (value) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      return ObservedTensorDataType::kUndefined;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return ObservedTensorDataType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return ObservedTensorDataType::kUInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return ObservedTensorDataType::kInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return ObservedTensorDataType::kUInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return ObservedTensorDataType::kInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return ObservedTensorDataType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return ObservedTensorDataType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return ObservedTensorDataType::kString;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return ObservedTensorDataType::kBool;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return ObservedTensorDataType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return ObservedTensorDataType::kFloat64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return ObservedTensorDataType::kUInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return ObservedTensorDataType::kUInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return ObservedTensorDataType::kComplex64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return ObservedTensorDataType::kComplex128;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return ObservedTensorDataType::kBFloat16;
    default:
      return ObservedTensorDataType::kUnsupported;
  }
}

TensorMetadata inspect_input(const Ort::Session& session,
                             std::size_t index,
                             OrtAllocator* allocator) {
  TensorMetadata metadata;
  const Ort::AllocatedStringPtr name =
      session.GetInputNameAllocated(index, allocator);
  metadata.name = name ? name.get() : "";

  const Ort::TypeInfo type_info = session.GetInputTypeInfo(index);
  metadata.value_type = convert_value_type(type_info.GetONNXType());
  if (metadata.value_type == ModelValueType::kTensor) {
    const Ort::ConstTensorTypeAndShapeInfo tensor_info =
        type_info.GetTensorTypeAndShapeInfo();
    metadata.dtype = convert_tensor_dtype(tensor_info.GetElementType());
    metadata.shape = tensor_info.GetShape();
  }
  return metadata;
}

TensorMetadata inspect_output(const Ort::Session& session,
                              std::size_t index,
                              OrtAllocator* allocator) {
  TensorMetadata metadata;
  const Ort::AllocatedStringPtr name =
      session.GetOutputNameAllocated(index, allocator);
  metadata.name = name ? name.get() : "";

  const Ort::TypeInfo type_info = session.GetOutputTypeInfo(index);
  metadata.value_type = convert_value_type(type_info.GetONNXType());
  if (metadata.value_type == ModelValueType::kTensor) {
    const Ort::ConstTensorTypeAndShapeInfo tensor_info =
        type_info.GetTensorTypeAndShapeInfo();
    metadata.dtype = convert_tensor_dtype(tensor_info.GetElementType());
    metadata.shape = tensor_info.GetShape();
  }
  return metadata;
}

std::runtime_error make_ort_error(const RuntimeContract& contract,
                                  const Ort::Exception& error) {
  return std::runtime_error(
      "ONNX session/metadata error for model '" +
      contract.artifact.model_path.string() +
      "': expected a loadable ONNX model compatible with the pinned ORT "
      "1.19.2 Windows x64 CPU SDK; actual ORT error code " +
      std::to_string(static_cast<int>(error.GetOrtErrorCode())) + ": " +
      error.what() +
      "; action: verify the model path and SHA-256, ONNX export/opset, "
      "CPU provider availability, and that the staged DLL matches "
      "ONNXRUNTIME_ROOT.");
}

[[noreturn]] void throw_inference_error(
    const std::filesystem::path& model_path,
    const std::string& object,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  throw std::runtime_error(
      "ONNX raw inference error for '" + object + "' in model '" +
      model_path.string() + "': expected " + expected + "; actual " +
      actual + "; action: " + action + ".");
}

std::string format_float_value(float value) {
  if (std::isnan(value)) {
    return "NaN";
  }
  if (std::isinf(value)) {
    return value > 0.0F ? "+Infinity" : "-Infinity";
  }
  std::ostringstream output;
  output << value;
  return output.str();
}

std::size_t checked_element_count(
    const std::vector<std::int64_t>& shape,
    const std::filesystem::path& model_path,
    const std::string& object) {
  if (shape.empty()) {
    throw_inference_error(
        model_path, object, "a non-empty static shape",
        format_shape(shape),
        "verify the tensor contract before constructing an Ort::Value");
  }

  std::size_t element_count = 1;
  for (std::size_t index = 0; index < shape.size(); ++index) {
    const std::int64_t dimension = shape[index];
    if (dimension <= 0) {
      throw_inference_error(
          model_path, object, "all dimensions to be positive",
          format_shape(shape),
          "use the static shape validated from the model artifact");
    }
    const std::uint64_t unsigned_dimension =
        static_cast<std::uint64_t>(dimension);
    if (unsigned_dimension >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()) ||
        element_count >
            std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(unsigned_dimension)) {
      throw_inference_error(
          model_path, object, "a shape whose element count fits size_t",
          format_shape(shape),
          "check for corrupt or unsupported tensor dimensions");
    }
    element_count *= static_cast<std::size_t>(unsigned_dimension);
  }
  if (element_count >
      std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw_inference_error(
        model_path, object,
        "a shape whose float32 byte count fits size_t",
        format_shape(shape),
        "check for corrupt or unsupported tensor dimensions before "
        "constructing an Ort::Value");
  }
  return element_count;
}

void validate_input_values(
    const std::vector<std::int64_t>& input_shape,
    const std::vector<float>& input_values,
    const TensorMetadata& expected_input,
    const std::filesystem::path& model_path) {
  if (input_shape != expected_input.shape) {
    throw_inference_error(
        model_path, "input.tensor_shape",
        format_shape(expected_input.shape), format_shape(input_shape),
        "derive [1,3,height,width] from the validated PreprocessResult "
        "and do not call Session::Run with a different shape");
  }

  const std::size_t expected_elements = checked_element_count(
      input_shape, model_path, "input.tensor_shape");
  if (input_values.size() != expected_elements) {
    throw_inference_error(
        model_path, "input.tensor_elements",
        std::to_string(expected_elements),
        std::to_string(input_values.size()),
        "verify PreprocessResult NCHW construction; the tensor is rejected "
        "before Ort::Value creation and Session::Run");
  }

  for (std::size_t index = 0; index < input_values.size(); ++index) {
    if (!std::isfinite(input_values[index])) {
      throw_inference_error(
          model_path,
          "input.tensor_values[" + std::to_string(index) + "]",
          "a finite float32 value", format_float_value(input_values[index]),
          "check image decoding, normalization, and NCHW conversion before "
          "Session::Run");
    }
  }
}

InferenceOutput copy_and_validate_output(
    std::vector<Ort::Value>& ort_outputs,
    const TensorMetadata& expected_output,
    const std::filesystem::path& model_path) {
  if (ort_outputs.size() != 1) {
    throw_inference_error(
        model_path, "output.count", "1",
        std::to_string(ort_outputs.size()),
        "verify the requested output names and the single-output model "
        "contract");
  }

  Ort::Value& ort_output = ort_outputs.front();
  if (!ort_output.HasValue() || !ort_output.IsTensor()) {
    throw_inference_error(
        model_path, "output[0].value_type", "a populated tensor",
        ort_output.HasValue() ? "non-tensor value" : "empty Ort::Value",
        "verify the ONNX output contract and Session::Run result");
  }

  const Ort::TensorTypeAndShapeInfo output_info =
      ort_output.GetTensorTypeAndShapeInfo();
  const ObservedTensorDataType output_dtype =
      convert_tensor_dtype(output_info.GetElementType());
  if (output_dtype != ObservedTensorDataType::kFloat32) {
    throw_inference_error(
        model_path, "output[0].dtype", "float32",
        to_string(output_dtype),
        "verify the actual model output type before consuming raw values");
  }

  const std::vector<std::int64_t> output_shape = output_info.GetShape();
  if (output_shape != expected_output.shape) {
    throw_inference_error(
        model_path, "output[0].shape",
        format_shape(expected_output.shape), format_shape(output_shape),
        "verify the model artifact and Session::Run output contract");
  }

  const std::size_t shape_elements = checked_element_count(
      output_shape, model_path, "output[0].shape");
  const std::size_t ort_elements = output_info.GetElementCount();
  if (ort_elements != shape_elements) {
    throw_inference_error(
        model_path, "output[0].elements",
        std::to_string(shape_elements), std::to_string(ort_elements),
        "verify the ORT output buffer and model shape metadata");
  }

  const float* output_data = ort_output.GetTensorData<float>();
  if (output_data == nullptr) {
    throw_inference_error(
        model_path, "output[0].data",
        "a non-null buffer for " + std::to_string(ort_elements) +
            " float32 values",
        "null",
        "inspect the ORT Session::Run result and model output allocation");
  }

  InferenceOutput owned_output;
  owned_output.shape = output_shape;
  owned_output.values.assign(output_data, output_data + ort_elements);

  for (std::size_t index = 0; index < owned_output.values.size(); ++index) {
    if (!std::isfinite(owned_output.values[index])) {
      throw_inference_error(
          model_path,
          "output[0].values[" + std::to_string(index) + "]",
          "a finite float32 value",
          format_float_value(owned_output.values[index]),
          "check preprocess values, model integrity, and ORT provider "
          "execution");
    }
  }

  return owned_output;
}

std::runtime_error make_run_ort_error(
    const std::filesystem::path& model_path,
    const Ort::Exception& error) {
  return std::runtime_error(
      "ONNX raw inference error for 'Session::Run' in model '" +
      model_path.string() +
      "': expected one successful CPU raw inference; actual ORT error code " +
      std::to_string(static_cast<int>(error.GetOrtErrorCode())) + ": " +
      error.what() +
      "; action: verify input shape/data, model integrity, provider "
      "availability, and that the staged DLL matches ONNXRUNTIME_ROOT.");
}

}  // namespace

class OnnxRunner::Impl {
 public:
  explicit Impl(const RuntimeContract& contract)
      : model_path_(contract.artifact.model_path),
        env_(ORT_LOGGING_LEVEL_WARNING, "yolo_defect_runtime"),
        session_options_(),
        session_(nullptr),
        allocator_() {
    metadata_.ort_version = Ort::GetVersionString();
    metadata_.available_providers = Ort::GetAvailableProviders();
    const bool cpu_is_available =
        std::find(metadata_.available_providers.begin(),
                  metadata_.available_providers.end(),
                  kCpuProviderName) != metadata_.available_providers.end();
    if (!cpu_is_available) {
      throw std::runtime_error(
          "ONNX provider error for model '" +
          contract.artifact.model_path.string() +
          "': expected available providers to contain CPUExecutionProvider; "
          "actual " + format_string_list(metadata_.available_providers) +
          "; action: verify that the staged onnxruntime.dll comes from the "
          "official Windows x64 CPU SDK selected by ONNXRUNTIME_ROOT.");
    }

    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.SetIntraOpNumThreads(kIntraOpNumThreads);
    session_options_.SetInterOpNumThreads(kInterOpNumThreads);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    session_options_.SetLogId("yolo_defect_runtime_session");
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CPU(session_options_, 1));

    session_ = Ort::Session(env_, contract.artifact.model_path.c_str(),
                            session_options_);

    metadata_.session_provider = kCpuProviderName;
    metadata_.provider_evidence =
        "explicit_cpu_ep_registration_and_session_creation";
    metadata_.intra_op_num_threads = kIntraOpNumThreads;
    metadata_.inter_op_num_threads = kInterOpNumThreads;
    metadata_.execution_mode = "sequential";
    metadata_.graph_optimization_level = "all";

    const std::size_t input_count = session_.GetInputCount();
    metadata_.inputs.reserve(input_count);
    for (std::size_t index = 0; index < input_count; ++index) {
      metadata_.inputs.push_back(
          inspect_input(session_, index, allocator_));
    }

    const std::size_t output_count = session_.GetOutputCount();
    metadata_.outputs.reserve(output_count);
    for (std::size_t index = 0; index < output_count; ++index) {
      metadata_.outputs.push_back(
          inspect_output(session_, index, allocator_));
    }

    validate_model_metadata(metadata_, contract);
  }

  const ModelMetadata& metadata() const noexcept { return metadata_; }

  InferenceOutput run(
      const std::vector<std::int64_t>& input_shape,
      std::vector<float>& input_values) {
    validate_input_values(input_shape, input_values,
                          metadata_.inputs.front(), model_path_);

    try {
      Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
          OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_values.data(), input_values.size(),
          input_shape.data(), input_shape.size());

      const char* input_names[] = {metadata_.inputs.front().name.c_str()};
      const char* output_names[] = {metadata_.outputs.front().name.c_str()};
      Ort::RunOptions run_options{nullptr};

      std::vector<Ort::Value> ort_outputs = session_.Run(
          run_options, input_names, &input_tensor, 1, output_names, 1);
      return copy_and_validate_output(
          ort_outputs, metadata_.outputs.front(), model_path_);
    } catch (const Ort::Exception& error) {
      throw make_run_ort_error(model_path_, error);
    }
  }

 private:
  std::filesystem::path model_path_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  ModelMetadata metadata_;
};

OnnxRunner::OnnxRunner(const RuntimeContract& contract) {
  try {
    impl_ = std::make_unique<Impl>(contract);
  } catch (const Ort::Exception& error) {
    throw make_ort_error(contract, error);
  }
}

OnnxRunner::~OnnxRunner() = default;
OnnxRunner::OnnxRunner(OnnxRunner&&) noexcept = default;
OnnxRunner& OnnxRunner::operator=(OnnxRunner&&) noexcept = default;

const ModelMetadata& OnnxRunner::metadata() const noexcept {
  return impl_->metadata();
}

InferenceOutput OnnxRunner::run(
    const std::vector<std::int64_t>& input_shape,
    std::vector<float>& input_values) {
  return impl_->run(input_shape, input_values);
}

}  // namespace yolo_defect_cpp

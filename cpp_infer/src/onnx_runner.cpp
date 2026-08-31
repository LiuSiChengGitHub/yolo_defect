#include "yolo_defect_cpp/onnx_runner.h"

#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#ifdef YOLO_DEFECT_NATIVE_TENSORRT_BUILD
#include "native_tensorrt_runner.h"
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

constexpr int kIntraOpNumThreads = 1;
constexpr int kInterOpNumThreads = 1;
constexpr const char* kCpuProviderName = "CPUExecutionProvider";
constexpr const char* kCudaProviderName = "CUDAExecutionProvider";
constexpr const char* kTensorRtProviderName = "TensorrtExecutionProvider";

std::string expected_provider_name(ExecutionProvider provider) {
  switch (provider) {
    case ExecutionProvider::kCpu:
      return kCpuProviderName;
    case ExecutionProvider::kTensorRt:
      return kTensorRtProviderName;
    case ExecutionProvider::kTensorRtNative:
      return "TensorRTNative";
  }
  throw std::logic_error("Unknown ExecutionProvider enum value.");
}

void require_available_provider(const ModelMetadata& metadata,
                                const RuntimeContract& contract,
                                const std::string& provider_name,
                                const std::string& purpose) {
  const bool is_available =
      std::find(metadata.available_providers.begin(),
                metadata.available_providers.end(), provider_name) !=
      metadata.available_providers.end();
  if (is_available) {
    return;
  }
  throw std::runtime_error(
      "ONNX provider error for model '" +
      contract.artifact.model_path.string() + "': expected " +
      provider_name + " for " + purpose + "; actual available providers " +
      format_string_list(metadata.available_providers) +
      "; action: point ONNXRUNTIME_ROOT at the matching official C++ SDK, "
      "verify its provider shared libraries with ldd, and rebuild the "
      "dedicated Linux GPU target when TensorRT is requested.");
}

std::string sanitize_cache_component(const std::string& value) {
  std::string result;
  result.reserve(value.size());
  for (const unsigned char character : value) {
    if (std::isalnum(character) != 0) {
      result.push_back(static_cast<char>(std::tolower(character)));
    } else {
      result.push_back('_');
    }
  }
  return result;
}

std::filesystem::path prepare_engine_cache_directory(
    const std::filesystem::path& declared_path,
    const std::filesystem::path& model_path) {
  if (declared_path.empty()) {
    throw std::runtime_error(
        "ONNX TensorRT provider error for model '" + model_path.string() +
        "': expected a non-empty engine cache directory; actual empty path; "
        "action: declare tensorrt_engine_cache_path in RuntimeConfig v2.");
  }
  std::error_code error;
  std::filesystem::path path =
      std::filesystem::absolute(declared_path, error);
  if (error) {
    throw std::runtime_error(
        "ONNX TensorRT provider error for model '" + model_path.string() +
        "': expected a resolvable engine cache directory; actual '" +
        declared_path.string() + "' (" + error.message() +
        "); action: correct tensorrt_engine_cache_path.");
  }
  path = path.lexically_normal();

  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (!error && std::filesystem::is_symlink(status)) {
    throw std::runtime_error(
        "ONNX TensorRT provider error for model '" + model_path.string() +
        "': expected a direct engine cache directory; actual symbolic link '" +
        path.string() +
        "'; action: use a dedicated non-symlink cache namespace.");
  }
  if (!error && std::filesystem::exists(status)) {
    if (!std::filesystem::is_directory(status)) {
      throw std::runtime_error(
          "ONNX TensorRT provider error for model '" + model_path.string() +
          "': expected an engine cache directory; actual non-directory '" +
          path.string() +
          "'; action: choose a dedicated directory path.");
    }
    return path;
  }
  if (error && status.type() != std::filesystem::file_type::not_found) {
    throw std::runtime_error(
        "ONNX TensorRT provider error for model '" + model_path.string() +
        "': expected an inspectable engine cache path; actual '" +
        path.string() + "' (" + error.message() +
        "); action: check parent permissions.");
  }
  error.clear();
  if (!std::filesystem::create_directories(path, error) && error) {
    throw std::runtime_error(
        "ONNX TensorRT provider error for model '" + model_path.string() +
        "': expected a creatable engine cache directory; actual '" +
        path.string() + "' (" + error.message() +
        "); action: select a writable cache namespace.");
  }
  return path;
}

std::size_t count_cache_files(const std::filesystem::path& directory,
                              const std::string& prefix) {
  std::error_code error;
  std::filesystem::directory_iterator iterator(directory, error);
  if (error) {
    throw std::runtime_error(
        "ONNX TensorRT cache inventory failed for '" + directory.string() +
        "': expected a readable cache directory; actual " + error.message() +
        "; action: check cache permissions before creating the session.");
  }
  std::size_t count = 0;
  for (const std::filesystem::directory_entry& entry : iterator) {
    const std::string filename = entry.path().filename().string();
    if (filename.rfind(prefix, 0) != 0) {
      continue;
    }
    const bool is_regular = entry.is_regular_file(error);
    if (error) {
      throw std::runtime_error(
          "ONNX TensorRT cache inventory failed for '" +
          entry.path().string() + "': expected an inspectable cache entry; "
          "actual " + error.message() +
          "; action: repair permissions or use a fresh cache namespace.");
    }
    if (is_regular) {
      ++count;
    }
  }
  return count;
}

struct TensorRtOptionsDeleter {
  void operator()(OrtTensorRTProviderOptionsV2* options) const noexcept {
    if (options != nullptr) {
      Ort::GetApi().ReleaseTensorRTProviderOptions(options);
    }
  }
};

struct CudaOptionsDeleter {
  void operator()(OrtCUDAProviderOptionsV2* options) const noexcept {
    if (options != nullptr) {
      Ort::GetApi().ReleaseCUDAProviderOptions(options);
    }
  }
};

using TensorRtOptionsPtr =
    std::unique_ptr<OrtTensorRTProviderOptionsV2, TensorRtOptionsDeleter>;
using CudaOptionsPtr =
    std::unique_ptr<OrtCUDAProviderOptionsV2, CudaOptionsDeleter>;

std::vector<const char*> c_string_views(
    const std::vector<std::string>& values) {
  std::vector<const char*> result;
  result.reserve(values.size());
  for (const std::string& value : values) {
    result.push_back(value.c_str());
  }
  return result;
}

[[noreturn]] void throw_profile_error(
    const std::filesystem::path& model_path,
    const std::string& object,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  throw std::runtime_error(
      "ONNX profiling error for '" + object + "' in model '" +
      model_path.string() + "': expected " + expected + "; actual " +
      actual + "; action: " + action + ".");
}

std::filesystem::path normalize_profile_prefix(
    const std::filesystem::path& declared_prefix,
    const std::filesystem::path& model_path) {
  if (declared_prefix.empty()) {
    throw_profile_error(
        model_path, "profile_file_prefix", "a non-empty file prefix",
        "empty", "pass --profile-prefix <path> with a filename prefix");
  }

  std::error_code error;
  std::filesystem::path prefix =
      std::filesystem::absolute(declared_prefix, error);
  if (error) {
    throw_profile_error(
        model_path, "profile_file_prefix",
        "a path resolvable from the current working directory",
        declared_prefix.string() + " (" + error.message() + ")",
        "correct the prefix path");
  }
  prefix = prefix.lexically_normal();
  if (prefix.filename().empty()) {
    throw_profile_error(
        model_path, "profile_file_prefix", "a file prefix, not a directory",
        prefix.string(), "append a filename prefix to the output directory");
  }

  const std::filesystem::path parent = prefix.parent_path();
  const bool parent_exists = std::filesystem::exists(parent, error);
  if (error || !parent_exists) {
    throw_profile_error(
        model_path, "profile_file_prefix.parent",
        "an existing accessible directory",
        error ? parent.string() + " (" + error.message() + ")"
              : "missing directory '" + parent.string() + "'",
        "create the profile output directory before constructing the ORT "
        "session");
  }
  const bool parent_is_directory =
      std::filesystem::is_directory(parent, error);
  if (error || !parent_is_directory) {
    throw_profile_error(
        model_path, "profile_file_prefix.parent", "a directory",
        error ? parent.string() + " (" + error.message() + ")"
              : "non-directory path '" + parent.string() + "'",
        "choose an existing directory for the trace prefix");
  }

  const std::filesystem::file_status prefix_status =
      std::filesystem::symlink_status(prefix, error);
  if (error &&
      prefix_status.type() != std::filesystem::file_type::not_found) {
    throw_profile_error(
        model_path, "profile_file_prefix", "an inspectable file prefix",
        prefix.string() + " (" + error.message() + ")",
        "check the target filesystem permissions");
  }
  if (!error && std::filesystem::is_directory(prefix_status)) {
    throw_profile_error(
        model_path, "profile_file_prefix", "a file prefix, not a directory",
        prefix.string(), "append a filename prefix to the directory");
  }
  return prefix;
}

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
  const bool tensorrt_requested =
      contract.runtime.provider == ExecutionProvider::kTensorRt ||
      contract.runtime.provider == ExecutionProvider::kTensorRtNative;
  const std::string provider_expectation =
      tensorrt_requested
          ? (contract.runtime.provider == ExecutionProvider::kTensorRtNative
                 ? "the isolated native TensorRT 10.4/CUDA 12.6 load-only "
                   "backend and its frozen sm89 engine"
                 : "the isolated ORT 1.20.1 GPU SDK with TensorRT 10.4, "
                   "CUDA 12.6, cuDNN 9.x, and the TensorRT->CUDA->CPU "
                   "provider chain")
          : "the pinned ORT 1.19.2 CPU SDK selected for this platform";
  const std::string cache_diagnostic =
      tensorrt_requested && contract.runtime.tensorrt.has_value()
          ? ", cache='" +
                contract.runtime.tensorrt->engine_cache_path.string() + "'"
          : "";
  return std::runtime_error(
      "ONNX session/metadata error for model '" +
      contract.artifact.model_path.string() +
      "': expected a loadable ONNX model compatible with " +
      provider_expectation + "; requested_provider=" +
      to_string(contract.runtime.provider) + cache_diagnostic +
      "; actual ORT error code " +
      std::to_string(static_cast<int>(error.GetOrtErrorCode())) + ": " +
      error.what() +
      "; action: verify the model path and SHA-256, ONNX export/opset, "
      "provider shared-library dependencies with ldd, LD_LIBRARY_PATH, and "
      "that the loaded shared library matches ONNXRUNTIME_ROOT.");
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
    const ModelMetadata& metadata,
    const Ort::Exception& error) {
  return std::runtime_error(
      "ONNX raw inference error for 'Session::Run' in model '" +
      model_path.string() +
      "': expected one successful " + metadata.session_provider +
      " raw inference; provider_evidence=" + metadata.provider_evidence +
      "; actual ORT error code " +
      std::to_string(static_cast<int>(error.GetOrtErrorCode())) + ": " +
      error.what() +
      "; action: verify input shape/data, model integrity, provider "
      "dependencies, engine-cache identity/permissions, and that the loaded "
      "ORT shared library matches ONNXRUNTIME_ROOT.");
}

}  // namespace

class OnnxRunner::Impl {
 public:
  explicit Impl(const RuntimeContract& contract, OnnxRunnerOptions options)
      : model_path_(contract.artifact.model_path),
        options_(std::move(options)),
        env_(ORT_LOGGING_LEVEL_WARNING, "yolo_defect_runtime"),
        session_options_(),
        session_(nullptr),
        allocator_() {
    if (contract.runtime.provider == ExecutionProvider::kTensorRtNative) {
      if (options_.profile_file_prefix.has_value()) {
        throw std::runtime_error(
            "Native TensorRT profiling error for model '" +
            contract.artifact.model_path.string() +
            "': expected ORT profiling only for provider=tensorrt; actual "
            "provider=tensorrt_native; action: use the retained ORT "
            "TensorRT EP profile for node placement or capture the native "
            "engine with trtexec/Nsight separately.");
      }
#ifndef YOLO_DEFECT_NATIVE_TENSORRT_BUILD
      throw std::runtime_error(
          "Native TensorRT provider error for model '" +
          contract.artifact.model_path.string() +
          "': expected a binary configured with "
          "YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND=ON; actual native "
          "backend not compiled; action: use the dedicated Linux x86_64 "
          "GPU build with explicit TensorRT and CUDA roots.");
#else
      native_runner_ = std::make_unique<NativeTensorRtRunner>(contract);
      metadata_ = native_runner_->metadata();
      // The binary still links the isolated ORT GPU SDK for the CPU/TRT-EP
      // paths. This field is environment inventory, not native placement
      // evidence; provider_evidence records the actual native execution.
      metadata_.ort_version = Ort::GetVersionString();
      session_initialization_ms_ = native_runner_->initialization_ms();
      return;
#endif
    }

    metadata_.ort_version = Ort::GetVersionString();
    metadata_.available_providers = Ort::GetAvailableProviders();
    require_available_provider(metadata_, contract, kCpuProviderName,
                               "the terminal fallback and CPU gate");

    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.SetIntraOpNumThreads(kIntraOpNumThreads);
    session_options_.SetInterOpNumThreads(kInterOpNumThreads);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    session_options_.SetLogId("yolo_defect_runtime_session");

    if (contract.runtime.provider == ExecutionProvider::kCpu) {
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_CPU(session_options_, 1));
      metadata_.registered_provider_chain = {kCpuProviderName};
      metadata_.inference_precision = "fp32";
    } else {
#ifndef YOLO_DEFECT_TENSORRT_EP_BUILD
      throw std::runtime_error(
          "ONNX TensorRT provider error for model '" +
          contract.artifact.model_path.string() +
          "': expected a binary configured with "
          "YOLO_DEFECT_REQUIRE_TENSORRT_EP=ON and ORT GPU 1.20.1; actual "
          "this binary is a CPU gate build; action: use the dedicated Linux "
          "x86_64 GPU build tree and keep the CPU build tree unchanged.");
#else
      if (!contract.runtime.tensorrt.has_value()) {
        throw std::runtime_error(
            "ONNX TensorRT provider error for model '" +
            contract.artifact.model_path.string() +
            "': expected parsed TensorRT RuntimeConfig v2 options; actual "
            "missing options; action: reload the contract from a valid "
            "provider=tensorrt config.");
      }
      require_available_provider(metadata_, contract, kTensorRtProviderName,
                                 "primary TensorRT execution");
      require_available_provider(metadata_, contract, kCudaProviderName,
                                 "unsupported-node CUDA fallback");

      const TensorRtProviderConfig& config = *contract.runtime.tensorrt;
      metadata_.device_id = config.device_id;
      metadata_.inference_precision = to_string(config.precision);
      metadata_.engine_cache_enabled = true;
      metadata_.engine_cache_path = prepare_engine_cache_directory(
          config.engine_cache_path, contract.artifact.model_path).string();
      metadata_.engine_cache_prefix = sanitize_cache_component(
          contract.artifact.model_id + "_" +
          contract.artifact.model_sha256 + "_ort" +
          metadata_.ort_version + "_" + metadata_.inference_precision +
          "_device" + std::to_string(config.device_id));
      metadata_.engine_cache_files_before = count_cache_files(
          metadata_.engine_cache_path, metadata_.engine_cache_prefix);

      const std::vector<std::string> tensorrt_keys = {
          "device_id", "trt_max_workspace_size", "trt_fp16_enable",
          "trt_engine_cache_enable", "trt_engine_cache_path",
          "trt_engine_cache_prefix", "trt_timing_cache_enable",
          "trt_timing_cache_path"};
      const std::vector<std::string> tensorrt_values = {
          std::to_string(config.device_id),
          std::to_string(config.max_workspace_size_bytes),
          config.precision == InferencePrecision::kFloat16 ? "1" : "0",
          "1", metadata_.engine_cache_path, metadata_.engine_cache_prefix,
          "1", metadata_.engine_cache_path};
      const std::vector<const char*> tensorrt_key_views =
          c_string_views(tensorrt_keys);
      const std::vector<const char*> tensorrt_value_views =
          c_string_views(tensorrt_values);

      OrtTensorRTProviderOptionsV2* raw_tensorrt_options = nullptr;
      Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(
          &raw_tensorrt_options));
      TensorRtOptionsPtr tensorrt_options(raw_tensorrt_options);
      Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
          tensorrt_options.get(), tensorrt_key_views.data(),
          tensorrt_value_views.data(), tensorrt_key_views.size()));
      session_options_.AppendExecutionProvider_TensorRT_V2(
          *tensorrt_options);

      const std::vector<std::string> cuda_keys = {
          "device_id", "do_copy_in_default_stream"};
      const std::vector<std::string> cuda_values = {
          std::to_string(config.device_id), "1"};
      const std::vector<const char*> cuda_key_views =
          c_string_views(cuda_keys);
      const std::vector<const char*> cuda_value_views =
          c_string_views(cuda_values);
      OrtCUDAProviderOptionsV2* raw_cuda_options = nullptr;
      Ort::ThrowOnError(
          Ort::GetApi().CreateCUDAProviderOptions(&raw_cuda_options));
      CudaOptionsPtr cuda_options(raw_cuda_options);
      Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
          cuda_options.get(), cuda_key_views.data(), cuda_value_views.data(),
          cuda_key_views.size()));
      session_options_.AppendExecutionProvider_CUDA_V2(*cuda_options);

      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_CPU(session_options_, 1));
      metadata_.registered_provider_chain = {
          kTensorRtProviderName, kCudaProviderName, kCpuProviderName};
#endif
    }

    if (options_.profile_file_prefix.has_value()) {
      profile_file_prefix_ = normalize_profile_prefix(
          *options_.profile_file_prefix, model_path_);
      session_options_.EnableProfiling(profile_file_prefix_.c_str());
      profiling_enabled_ = true;
    }

    const auto session_start = std::chrono::steady_clock::now();
    session_ = Ort::Session(env_, contract.artifact.model_path.c_str(),
                            session_options_);
    const auto session_end = std::chrono::steady_clock::now();
    session_initialization_ms_ =
        std::chrono::duration<double, std::milli>(
            session_end - session_start).count();
    if (!std::isfinite(session_initialization_ms_) ||
        session_initialization_ms_ < 0.0) {
      throw_profile_error(
          model_path_, "session_initialization_ms",
          "a finite non-negative steady-clock duration",
          std::to_string(session_initialization_ms_),
          "verify the platform steady_clock before publishing benchmark "
          "evidence");
    }

    metadata_.session_provider =
        expected_provider_name(contract.runtime.provider);
    if (contract.runtime.provider == ExecutionProvider::kCpu) {
      metadata_.provider_evidence =
          "explicit_cpu_ep_registration_and_session_creation";
    } else {
      refresh_engine_cache_evidence();
    }
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

  double session_initialization_ms() const noexcept {
    return session_initialization_ms_;
  }

  bool profiling_enabled() const noexcept { return profiling_enabled_; }

  std::filesystem::path end_profiling() {
#ifdef YOLO_DEFECT_NATIVE_TENSORRT_BUILD
    if (native_runner_) {
      throw_profile_error(
          model_path_, "profiling.backend", "an ORT execution provider",
          "TensorRTNative",
          "use provider=tensorrt for ORT JSON profiling or capture the "
          "native engine with trtexec/Nsight");
    }
#endif
    if (!profiling_enabled_) {
      throw_profile_error(
          model_path_, "profiling.state", "profiling enabled at session "
          "creation", "disabled",
          "construct OnnxRunner with OnnxRunnerOptions.profile_file_prefix");
    }
    if (profiling_ended_) {
      throw_profile_error(
          model_path_, "profiling.state", "one finalization after the last "
          "Session::Run", "EndProfiling was already called",
          "retain the path returned by the first end_profiling call");
    }

    try {
      Ort::AllocatedStringPtr allocated_path =
          session_.EndProfilingAllocated(allocator_);
      profiling_ended_ = true;
      if (!allocated_path || allocated_path.get()[0] == '\0') {
        throw_profile_error(
            model_path_, "profile_trace.path",
            "a non-empty filename returned by ORT", "empty",
            "check the profile prefix and ORT runtime diagnostics");
      }

      std::filesystem::path trace_path =
          std::filesystem::u8path(allocated_path.get());
      std::error_code error;
      trace_path = std::filesystem::absolute(trace_path, error);
      if (error) {
        throw_profile_error(
            model_path_, "profile_trace.path",
            "a returned path resolvable from the current working directory",
            std::string(allocated_path.get()) + " (" + error.message() + ")",
            "inspect the ORT profile prefix and process working directory");
      }
      trace_path = trace_path.lexically_normal();
      const bool is_regular =
          std::filesystem::is_regular_file(trace_path, error);
      if (error || !is_regular) {
        throw_profile_error(
            model_path_, "profile_trace.path",
            "an existing regular trace file",
            error ? trace_path.string() + " (" + error.message() + ")"
                  : "missing or non-file path '" + trace_path.string() + "'",
            "check output permissions and the ORT profiling diagnostic");
      }
      const std::uintmax_t trace_size =
          std::filesystem::file_size(trace_path, error);
      if (error || trace_size == 0) {
        throw_profile_error(
            model_path_, "profile_trace.size", "a non-empty JSON trace",
            error ? trace_path.string() + " (" + error.message() + ")"
                  : "0 bytes",
            "rerun profiling in a writable directory after at least one "
            "Session::Run");
      }
      return trace_path;
    } catch (const Ort::Exception& error) {
      throw_profile_error(
          model_path_, "SessionEndProfiling", "a successfully finalized ORT "
          "JSON trace",
          "ORT error code " +
              std::to_string(static_cast<int>(error.GetOrtErrorCode())) +
              ": " + error.what(),
          "check profile output permissions and the loaded ORT shared library");
    }
  }

  TimedInferenceOutput run_with_session_timing(
      const std::vector<std::int64_t>& input_shape,
      std::vector<float>& input_values) {
#ifdef YOLO_DEFECT_NATIVE_TENSORRT_BUILD
    if (native_runner_) {
      NativeTimedInferenceOutput native =
          native_runner_->run_with_timing(input_shape, input_values);
      TimedInferenceOutput result;
      result.output = std::move(native.output);
      result.session_run_ms = native.backend_run_ms;
      return result;
    }
#endif
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

      const auto run_start = std::chrono::steady_clock::now();
      std::vector<Ort::Value> ort_outputs = session_.Run(
          run_options, input_names, &input_tensor, 1, output_names, 1);
      const auto run_end = std::chrono::steady_clock::now();

      TimedInferenceOutput result;
      result.session_run_ms =
          std::chrono::duration<double, std::milli>(
              run_end - run_start).count();
      if (!std::isfinite(result.session_run_ms) ||
          result.session_run_ms < 0.0) {
        throw_inference_error(
            model_path_, "Session::Run.duration", "a finite non-negative "
            "steady-clock duration", std::to_string(result.session_run_ms),
            "verify the platform steady_clock implementation before "
            "publishing benchmark evidence");
      }
      result.output = copy_and_validate_output(
          ort_outputs, metadata_.outputs.front(), model_path_);
      if (!engine_cache_evidence_refreshed_after_first_run_) {
        refresh_engine_cache_evidence();
        engine_cache_evidence_refreshed_after_first_run_ = true;
      }
      return result;
    } catch (const Ort::Exception& error) {
      throw make_run_ort_error(model_path_, metadata_, error);
    }
  }

 private:
  void refresh_engine_cache_evidence() {
    if (!metadata_.engine_cache_enabled) {
      return;
    }
    metadata_.engine_cache_files_after = count_cache_files(
        metadata_.engine_cache_path, metadata_.engine_cache_prefix);
    if (metadata_.engine_cache_files_before > 0) {
      metadata_.engine_cache_state =
          "warm_cache_present_before_session_creation";
    } else if (metadata_.engine_cache_files_after > 0) {
      metadata_.engine_cache_state =
          "cold_cache_materialized_by_session_or_first_run";
    } else {
      metadata_.engine_cache_state = "cache_not_materialized";
    }
    metadata_.provider_evidence =
        "explicit_tensorrt_ep_then_cuda_fallback_then_cpu_registration;"
        "precision=" + metadata_.inference_precision +
        ";cache_state=" + metadata_.engine_cache_state +
        ";per_node_execution_requires_ort_profile";
  }

  std::filesystem::path model_path_;
  OnnxRunnerOptions options_;
  std::filesystem::path profile_file_prefix_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  ModelMetadata metadata_;
  double session_initialization_ms_ = 0.0;
  bool profiling_enabled_ = false;
  bool profiling_ended_ = false;
  bool engine_cache_evidence_refreshed_after_first_run_ = false;
#ifdef YOLO_DEFECT_NATIVE_TENSORRT_BUILD
  std::unique_ptr<NativeTensorRtRunner> native_runner_;
#endif
};

OnnxRunner::OnnxRunner(const RuntimeContract& contract,
                       OnnxRunnerOptions options) {
  try {
    impl_ = std::make_unique<Impl>(contract, std::move(options));
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

double OnnxRunner::session_initialization_ms() const noexcept {
  return impl_->session_initialization_ms();
}

bool OnnxRunner::profiling_enabled() const noexcept {
  return impl_->profiling_enabled();
}

std::filesystem::path OnnxRunner::end_profiling() {
  return impl_->end_profiling();
}

InferenceOutput OnnxRunner::run(
    const std::vector<std::int64_t>& input_shape,
    std::vector<float>& input_values) {
  return impl_->run_with_session_timing(input_shape, input_values).output;
}

TimedInferenceOutput OnnxRunner::run_with_session_timing(
    const std::vector<std::int64_t>& input_shape,
    std::vector<float>& input_values) {
  return impl_->run_with_session_timing(input_shape, input_values);
}

}  // namespace yolo_defect_cpp

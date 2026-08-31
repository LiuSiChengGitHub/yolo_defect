#include "native_tensorrt_runner.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

constexpr const char* kNativeProviderName = "TensorRTNative";
constexpr const char* kPrecisionPolicy =
    "fp16_dfl_softmax_fp32_else_no_tf32";
constexpr const char* kFrozenModelId = "yolov8n_neu_det_final_train_2";
constexpr const char* kFrozenModelSha256 =
    "7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68";
constexpr const char* kFrozenEngineSha256 =
    "E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746";

std::string format_tensorrt_version(std::int32_t version) {
  return std::to_string(version / 10000) + "." +
         std::to_string((version % 10000) / 100) + "." +
         std::to_string(version % 100);
}

std::string compiled_tensorrt_header_version() {
  return std::to_string(NV_TENSORRT_MAJOR) + "." +
         std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH) + "." +
         std::to_string(NV_TENSORRT_BUILD);
}

[[noreturn]] void throw_native_error(
    const std::filesystem::path& engine_path,
    const std::string& object,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  throw std::runtime_error(
      "Native TensorRT error for '" + object + "' in engine '" +
      engine_path.string() + "': expected " + expected + "; actual " +
      actual + "; action: " + action + ".");
}

void check_cuda(cudaError_t status,
                const std::filesystem::path& engine_path,
                const std::string& object,
                const std::string& action) {
  if (status == cudaSuccess) {
    return;
  }
  throw_native_error(engine_path, object, "cudaSuccess",
                     std::string(cudaGetErrorName(status)) + ": " +
                         cudaGetErrorString(status),
                     action);
}

std::size_t checked_element_count(
    const std::vector<std::int64_t>& shape,
    const std::filesystem::path& engine_path,
    const std::string& object) {
  if (shape.empty()) {
    throw_native_error(engine_path, object, "a non-empty static shape",
                       format_shape(shape),
                       "rebuild the engine with static model dimensions");
  }
  std::size_t count = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) >
            std::numeric_limits<std::size_t>::max() ||
        count > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(dimension)) {
      throw_native_error(
          engine_path, object,
          "positive static dimensions whose element count fits size_t",
          format_shape(shape),
          "rebuild the frozen engine from the declared static ONNX");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw_native_error(engine_path, object,
                       "a float32 byte count that fits size_t",
                       format_shape(shape),
                       "inspect the engine for corrupt tensor dimensions");
  }
  return count;
}

std::vector<std::int64_t> convert_dims(
    const nvinfer1::Dims& dimensions,
    const std::filesystem::path& engine_path,
    const std::string& object) {
  if (dimensions.nbDims <= 0) {
    throw_native_error(engine_path, object, "a positive tensor rank",
                       std::to_string(dimensions.nbDims),
                       "rebuild the engine with static explicit dimensions");
  }
  std::vector<std::int64_t> result;
  result.reserve(static_cast<std::size_t>(dimensions.nbDims));
  for (int index = 0; index < dimensions.nbDims; ++index) {
    result.push_back(dimensions.d[index]);
  }
  checked_element_count(result, engine_path, object);
  return result;
}

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (message == nullptr || severity > Severity::kWARNING) {
      return;
    }
    try {
      const std::lock_guard<std::mutex> lock(mutex_);
      std::cerr << "[TensorRT] "
                << (severity <= Severity::kERROR ? "error: " : "warning: ")
                << message << '\n';
    } catch (...) {
      // TensorRT requires a noexcept, thread-safe logger.
    }
  }

 private:
  std::mutex mutex_;
};

std::vector<char> read_bound_file_bytes(
    const std::filesystem::path& file_path,
    const std::filesystem::path& engine_path,
    const std::string& object,
    const std::string& action) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(file_path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    throw_native_error(
        engine_path, object + ".path", "an existing direct regular file",
        error ? error.message() : "missing, symlink, or non-regular path",
        action);
  }
  const std::uintmax_t size = std::filesystem::file_size(file_path, error);
  if (error || size == 0 ||
      size > static_cast<std::uintmax_t>(
                 std::numeric_limits<std::streamsize>::max())) {
    throw_native_error(
        engine_path, object + ".size", "a readable non-empty regular file",
        error ? error.message() : std::to_string(size) + " bytes",
        action);
  }
  std::vector<char> bytes(static_cast<std::size_t>(size));
  std::ifstream stream(file_path, std::ios::binary);
  if (!stream ||
      !stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    throw_native_error(
        engine_path, object + ".read", "all declared file bytes",
        "short or failed binary read",
        action);
  }
  return bytes;
}

std::size_t count_direct_cache_files(
    const std::filesystem::path& cache_path,
    const std::filesystem::path& engine_path) {
  std::error_code error;
  std::filesystem::directory_iterator iterator(cache_path, error);
  if (error) {
    throw_native_error(
        engine_path, "engine_cache.inventory", "a readable cache directory",
        error.message(), "fix cache permissions and restore the frozen engine");
  }
  std::size_t count = 0;
  for (const auto& entry : iterator) {
    const std::filesystem::file_status status =
        entry.symlink_status(error);
    if (error) {
      throw_native_error(
          engine_path, "engine_cache.inventory", "inspectable cache entries",
          error.message(), "fix cache permissions and retry inspection");
    }
    if (std::filesystem::is_regular_file(status)) {
      ++count;
    }
  }
  return count;
}

std::uint32_t rotate_right(std::uint32_t value, unsigned int bits) {
  return (value >> bits) | (value << (32U - bits));
}

std::string sha256_bytes(const std::vector<char>& bytes) {
  static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  std::vector<std::uint8_t> padded;
  padded.reserve(bytes.size() + 72U);
  for (const char byte : bytes) {
    padded.push_back(static_cast<std::uint8_t>(byte));
  }
  const std::uint64_t bit_length =
      static_cast<std::uint64_t>(bytes.size()) * 8U;
  padded.push_back(0x80U);
  while ((padded.size() % 64U) != 56U) {
    padded.push_back(0U);
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    padded.push_back(
        static_cast<std::uint8_t>((bit_length >> shift) & 0xffU));
  }

  std::array<std::uint32_t, 8> hash = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  for (std::size_t offset = 0; offset < padded.size(); offset += 64U) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index) {
      const std::size_t position = offset + index * 4U;
      words[index] =
          (static_cast<std::uint32_t>(padded[position]) << 24U) |
          (static_cast<std::uint32_t>(padded[position + 1U]) << 16U) |
          (static_cast<std::uint32_t>(padded[position + 2U]) << 8U) |
          static_cast<std::uint32_t>(padded[position + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index) {
      const std::uint32_t s0 =
          rotate_right(words[index - 15U], 7U) ^
          rotate_right(words[index - 15U], 18U) ^
          (words[index - 15U] >> 3U);
      const std::uint32_t s1 =
          rotate_right(words[index - 2U], 17U) ^
          rotate_right(words[index - 2U], 19U) ^
          (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    std::uint32_t a = hash[0];
    std::uint32_t b = hash[1];
    std::uint32_t c = hash[2];
    std::uint32_t d = hash[3];
    std::uint32_t e = hash[4];
    std::uint32_t f = hash[5];
    std::uint32_t g = hash[6];
    std::uint32_t h = hash[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum1 = rotate_right(e, 6U) ^
                                 rotate_right(e, 11U) ^
                                 rotate_right(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sum1 + choose + kRoundConstants[index] + words[index];
      const std::uint32_t sum0 = rotate_right(a, 2U) ^
                                 rotate_right(a, 13U) ^
                                 rotate_right(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }

  std::ostringstream output;
  output << std::uppercase << std::hex << std::setfill('0');
  for (const std::uint32_t value : hash) {
    output << std::setw(8) << value;
  }
  return output.str();
}

TensorMetadata inspect_tensor(const nvinfer1::ICudaEngine& engine,
                              const char* name,
                              const std::filesystem::path& engine_path) {
  TensorMetadata metadata;
  metadata.name = name == nullptr ? "" : name;
  metadata.value_type = ModelValueType::kTensor;
  const nvinfer1::DataType data_type =
      engine.getTensorDataType(metadata.name.c_str());
  metadata.dtype = data_type == nvinfer1::DataType::kFLOAT
                       ? ObservedTensorDataType::kFloat32
                       : ObservedTensorDataType::kUnsupported;
  metadata.shape = convert_dims(
      engine.getTensorShape(metadata.name.c_str()), engine_path,
      "tensor['" + metadata.name + "'].shape");
  if (engine.getTensorFormat(metadata.name.c_str()) !=
      nvinfer1::TensorFormat::kLINEAR) {
    throw_native_error(
        engine_path, "tensor['" + metadata.name + "'].format",
        "linear contiguous format", "a non-linear TensorRT tensor format",
        "rebuild with FP32 linear external I/O");
  }
  if (engine.getTensorLocation(metadata.name.c_str()) !=
      nvinfer1::TensorLocation::kDEVICE) {
    throw_native_error(
        engine_path, "tensor['" + metadata.name + "'].location",
        "device memory", "host memory",
        "rebuild with normal GPU input/output tensors or implement an "
        "explicit host-I/O adapter");
  }
  return metadata;
}

}  // namespace

class NativeTensorRtRunner::Impl {
 public:
  explicit Impl(const RuntimeContract& contract)
      : model_path_(contract.artifact.model_path) {
    if (contract.runtime.provider != ExecutionProvider::kTensorRtNative ||
        !contract.runtime.tensorrt.has_value()) {
      throw std::logic_error(
          "NativeTensorRtRunner requires provider=tensorrt_native with "
          "parsed RuntimeConfig v2 options.");
    }
    const TensorRtProviderConfig& config = *contract.runtime.tensorrt;
    if (!config.native_engine_path.has_value() ||
        !config.native_engine_sha256.has_value()) {
      throw std::logic_error(
          "NativeTensorRtRunner requires the frozen engine path and SHA.");
    }
    engine_path_ = *config.native_engine_path;
    declared_engine_sha256_ = *config.native_engine_sha256;
    device_id_ = config.device_id;

    if (config.precision != InferencePrecision::kFloat16 ||
        contract.artifact.model_id != kFrozenModelId ||
        contract.artifact.model_sha256 != kFrozenModelSha256 ||
        declared_engine_sha256_ != kFrozenEngineSha256) {
      throw_native_error(
          engine_path_, "frozen_native_contract",
          std::string("provider=tensorrt_native, precision=fp16, model_id=") +
              kFrozenModelId + ", model_sha256=" + kFrozenModelSha256 +
              ", engine_sha256=" + kFrozenEngineSha256,
          "precision=" + to_string(config.precision) + ", model_id=" +
              contract.artifact.model_id + ", model_sha256=" +
              contract.artifact.model_sha256 + ", engine_sha256=" +
              declared_engine_sha256_,
          "use the frozen S2-04 native config/artifact or implement and "
          "validate a separately versioned backend contract");
    }

    if (engine_path_.parent_path().lexically_normal() !=
        config.engine_cache_path.lexically_normal()) {
      throw_native_error(
          engine_path_, "engine_cache_identity",
          "tensorrt_engine_path to be directly inside " +
              config.engine_cache_path.string(),
          engine_path_.parent_path().string(),
          "use the dedicated model/TRT/CUDA/sm/precision cache namespace");
    }

    const auto initialization_start = std::chrono::steady_clock::now();
    check_cuda(cudaSetDevice(device_id_), engine_path_, "cudaSetDevice",
               "select a visible NVIDIA device and verify the WSL driver");

    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device_id_), engine_path_,
               "cudaGetDeviceProperties",
               "verify the selected CUDA device and driver/runtime stack");
    if (properties.major != 8 || properties.minor != 9) {
      throw_native_error(
          engine_path_, "device.compute_capability",
          "8.9 for this frozen engine",
          std::to_string(properties.major) + "." +
              std::to_string(properties.minor) + " (" + properties.name + ")",
          "rebuild and re-freeze the engine for the actual GPU architecture");
    }

    int cuda_runtime_version = 0;
    check_cuda(cudaRuntimeGetVersion(&cuda_runtime_version), engine_path_,
               "cudaRuntimeGetVersion",
               "load the isolated CUDA 12.6 runtime before this binary");
    if (cuda_runtime_version / 1000 != 12 ||
        (cuda_runtime_version % 1000) / 10 != 6) {
      throw_native_error(
          engine_path_, "cuda_runtime.version", "12.6",
          std::to_string(cuda_runtime_version),
          "load the CUDA 12.6 libraries used to build the frozen engine");
    }

    const std::int32_t tensorrt_runtime_version = getInferLibVersion();
    if (tensorrt_runtime_version != NV_TENSORRT_VERSION) {
      throw_native_error(
          engine_path_, "tensorrt_runtime.version",
          format_tensorrt_version(NV_TENSORRT_VERSION) +
              " matching the compile-time headers",
          format_tensorrt_version(tensorrt_runtime_version),
          "load the libnvinfer runtime from the same isolated TensorRT "
          "10.4.0.26 package as the pinned headers");
    }
    tensorrt_runtime_version_ =
        format_tensorrt_version(tensorrt_runtime_version);
    tensorrt_header_package_version_ = compiled_tensorrt_header_version();

    const std::vector<char> model_bytes = read_bound_file_bytes(
        model_path_, engine_path_, "model_artifact",
        "restore the exact frozen ONNX bytes declared by the artifact");
    const std::string actual_model_sha256 = sha256_bytes(model_bytes);
    if (actual_model_sha256 != kFrozenModelSha256) {
      throw_native_error(
          engine_path_, "model_artifact.sha256", kFrozenModelSha256,
          actual_model_sha256,
          "restore the exact frozen ONNX before claiming same-artifact "
          "native execution");
    }
    std::vector<char> engine_bytes = read_bound_file_bytes(
        engine_path_, engine_path_, "engine",
        "restore the exact frozen engine bytes at tensorrt_engine_path");
    actual_engine_sha256_ = sha256_bytes(engine_bytes);
    if (actual_engine_sha256_ != declared_engine_sha256_) {
      throw_native_error(
          engine_path_, "engine.sha256", declared_engine_sha256_,
          actual_engine_sha256_,
          "restore the exact frozen engine bytes or version a new native "
          "engine/config/protocol before running inference");
    }
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw_native_error(
          engine_path_, "createInferRuntime", "a TensorRT runtime",
          "null runtime (see preceding TensorRT diagnostics)",
          "load libnvinfer 10.4.0.26 and check its CUDA dependencies");
    }
    engine_.reset(runtime_->deserializeCudaEngine(
        engine_bytes.data(), engine_bytes.size()));
    if (!engine_) {
      throw_native_error(
          engine_path_, "deserializeCudaEngine",
          "an engine compatible with TensorRT 10.4, CUDA 12.6, and sm89",
          "null engine (see preceding TensorRT diagnostics)",
          "restore the declared engine SHA or rebuild it on this stack");
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw_native_error(
          engine_path_, "createExecutionContext", "one execution context",
          "null context (see preceding TensorRT diagnostics)",
          "check GPU memory and rebuild the compatible engine if needed");
    }

    const int io_count = engine_->getNbIOTensors();
    if (io_count != 2) {
      throw_native_error(engine_path_, "engine.io_count", "2",
                         std::to_string(io_count),
                         "rebuild from the frozen one-input/one-output ONNX");
    }
    for (int index = 0; index < io_count; ++index) {
      const char* name = engine_->getIOTensorName(index);
      if (name == nullptr || name[0] == '\0') {
        throw_native_error(engine_path_, "engine.io_name", "a non-empty name",
                           "null or empty",
                           "rebuild the engine from the declared ONNX");
      }
      const TensorMetadata tensor =
          inspect_tensor(*engine_, name, engine_path_);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        metadata_.inputs.push_back(tensor);
      } else if (engine_->getTensorIOMode(name) ==
                 nvinfer1::TensorIOMode::kOUTPUT) {
        metadata_.outputs.push_back(tensor);
      } else {
        throw_native_error(engine_path_, "tensor['" + tensor.name + "'].mode",
                           "input or output", "none",
                           "rebuild the serialized engine");
      }
    }
    if (metadata_.inputs.size() != 1 || metadata_.outputs.size() != 1) {
      throw_native_error(
          engine_path_, "engine.io_modes", "one input and one output",
          "inputs=" + std::to_string(metadata_.inputs.size()) +
              ", outputs=" + std::to_string(metadata_.outputs.size()),
          "rebuild from the frozen YOLO ONNX");
    }

    metadata_.ort_version =
        "not_applicable_native_tensorrt_" + tensorrt_runtime_version_;
    metadata_.available_providers = {kNativeProviderName};
    metadata_.session_provider = kNativeProviderName;
    metadata_.registered_provider_chain = {kNativeProviderName};
    metadata_.inference_precision = to_string(config.precision);
    metadata_.device_id = device_id_;
    metadata_.engine_cache_enabled = true;
    metadata_.engine_cache_path = config.engine_cache_path.string();
    metadata_.engine_cache_prefix = engine_path_.filename().string();
    metadata_.engine_cache_state = "frozen_native_engine_loaded";
    metadata_.engine_cache_files_before = count_direct_cache_files(
        config.engine_cache_path, engine_path_);
    metadata_.engine_cache_files_after = metadata_.engine_cache_files_before;
    metadata_.intra_op_num_threads = 0;
    metadata_.inter_op_num_threads = 0;
    metadata_.execution_mode = "synchronous_non_default_cuda_stream";
    metadata_.graph_optimization_level = "frozen_engine_build_time";
    metadata_.provider_evidence =
        std::string("native_tensorrt_enqueue_v3;precision_policy=") +
        kPrecisionPolicy + ";declared_engine_sha256=" +
        declared_engine_sha256_ + ";actual_engine_sha256=" +
        actual_engine_sha256_ +
        ";tensorrt_runtime=" + tensorrt_runtime_version_ +
        ";compiled_headers=" + tensorrt_header_package_version_ +
        ";cuda_runtime=12.6;"
        "compute_capability=8.9;fallback=none";

    validate_model_metadata(metadata_, contract);

    input_elements_ = checked_element_count(
        metadata_.inputs.front().shape, engine_path_, "input.shape");
    output_elements_ = checked_element_count(
        metadata_.outputs.front().shape, engine_path_, "output.shape");
    input_bytes_ = input_elements_ * sizeof(float);
    output_bytes_ = output_elements_ * sizeof(float);
    check_cuda(cudaMalloc(&device_input_, input_bytes_), engine_path_,
               "cudaMalloc(input)",
               "free GPU memory or reduce the model input contract");
    try {
      check_cuda(cudaMalloc(&device_output_, output_bytes_), engine_path_,
                 "cudaMalloc(output)",
                 "free GPU memory or inspect the output contract");
      check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                 engine_path_, "cudaStreamCreateWithFlags",
                 "check CUDA runtime health and available resources");
    } catch (...) {
      if (device_output_ != nullptr) {
        cudaFree(device_output_);
        device_output_ = nullptr;
      }
      cudaFree(device_input_);
      device_input_ = nullptr;
      throw;
    }

    const auto initialization_end = std::chrono::steady_clock::now();
    initialization_ms_ = std::chrono::duration<double, std::milli>(
                             initialization_end - initialization_start)
                             .count();
    if (!std::isfinite(initialization_ms_) || initialization_ms_ < 0.0) {
      throw_native_error(
          engine_path_, "initialization.duration",
          "a finite non-negative steady-clock duration",
          std::to_string(initialization_ms_),
          "verify the platform steady_clock implementation");
    }
  }

  ~Impl() {
    if (device_id_ >= 0) {
      cudaSetDevice(device_id_);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
    if (device_output_ != nullptr) {
      cudaFree(device_output_);
    }
    if (device_input_ != nullptr) {
      cudaFree(device_input_);
    }
  }

  const ModelMetadata& metadata() const noexcept { return metadata_; }
  double initialization_ms() const noexcept { return initialization_ms_; }

  NativeTimedInferenceOutput run_with_timing(
      const std::vector<std::int64_t>& input_shape,
      const std::vector<float>& input_values) {
    if (input_shape != metadata_.inputs.front().shape) {
      throw_native_error(engine_path_, "input.shape",
                         format_shape(metadata_.inputs.front().shape),
                         format_shape(input_shape),
                         "pass the validated preprocess tensor shape");
    }
    if (input_values.size() != input_elements_) {
      throw_native_error(
          engine_path_, "input.element_count",
          std::to_string(input_elements_),
          std::to_string(input_values.size()),
          "pass the complete float32 NCHW preprocess tensor");
    }
    for (std::size_t index = 0; index < input_values.size(); ++index) {
      if (!std::isfinite(input_values[index])) {
        throw_native_error(
            engine_path_, "input.values[" + std::to_string(index) + "]",
            "a finite float32 value", std::to_string(input_values[index]),
            "fix preprocess normalization before native inference");
      }
    }

    check_cuda(cudaSetDevice(device_id_), engine_path_, "cudaSetDevice(run)",
               "keep the configured GPU visible to the calling thread");
    std::vector<float> output_values(output_elements_);
    const auto run_start = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpyAsync(device_input_, input_values.data(),
                               input_bytes_, cudaMemcpyHostToDevice, stream_),
               engine_path_, "cudaMemcpyAsync(H2D)",
               "check the host input buffer and CUDA context");
    if (!context_->setInputTensorAddress(
            metadata_.inputs.front().name.c_str(), device_input_) ||
        !context_->setOutputTensorAddress(
            metadata_.outputs.front().name.c_str(), device_output_)) {
      throw_native_error(
          engine_path_, "setTensorAddress", "accepted input/output buffers",
          "false (see preceding TensorRT diagnostics)",
          "verify engine I/O names, alignment, and tensor byte counts");
    }
    if (!context_->enqueueV3(stream_)) {
      throw_native_error(
          engine_path_, "enqueueV3", "true",
          "false (see preceding TensorRT diagnostics)",
          "check engine compatibility, tensor addresses, and GPU memory");
    }
    check_cuda(cudaMemcpyAsync(output_values.data(), device_output_,
                               output_bytes_, cudaMemcpyDeviceToHost, stream_),
               engine_path_, "cudaMemcpyAsync(D2H)",
               "check the output buffer and CUDA context");
    check_cuda(cudaStreamSynchronize(stream_), engine_path_,
               "cudaStreamSynchronize",
               "inspect the preceding TensorRT/CUDA asynchronous failure");
    const auto run_end = std::chrono::steady_clock::now();

    for (std::size_t index = 0; index < output_values.size(); ++index) {
      if (!std::isfinite(output_values[index])) {
        throw_native_error(
            engine_path_, "output.values[" + std::to_string(index) + "]",
            "a finite float32 value", std::to_string(output_values[index]),
            "check input normalization and the frozen engine integrity");
      }
    }

    NativeTimedInferenceOutput result;
    result.backend_run_ms =
        std::chrono::duration<double, std::milli>(run_end - run_start).count();
    if (!std::isfinite(result.backend_run_ms) ||
        result.backend_run_ms < 0.0) {
      throw_native_error(
          engine_path_, "backend_run.duration",
          "a finite non-negative steady-clock duration",
          std::to_string(result.backend_run_ms),
          "verify the platform steady_clock implementation");
    }
    result.output.shape = metadata_.outputs.front().shape;
    result.output.values = std::move(output_values);
    return result;
  }

 private:
  std::filesystem::path model_path_;
  std::filesystem::path engine_path_;
  std::string declared_engine_sha256_;
  std::string actual_engine_sha256_;
  std::string tensorrt_runtime_version_;
  std::string tensorrt_header_package_version_;
  int device_id_ = -1;
  TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  void* device_input_ = nullptr;
  void* device_output_ = nullptr;
  std::size_t input_elements_ = 0;
  std::size_t output_elements_ = 0;
  std::size_t input_bytes_ = 0;
  std::size_t output_bytes_ = 0;
  ModelMetadata metadata_;
  double initialization_ms_ = 0.0;
};

NativeTensorRtRunner::NativeTensorRtRunner(const RuntimeContract& contract)
    : impl_(std::make_unique<Impl>(contract)) {}

NativeTensorRtRunner::~NativeTensorRtRunner() = default;

const ModelMetadata& NativeTensorRtRunner::metadata() const noexcept {
  return impl_->metadata();
}

double NativeTensorRtRunner::initialization_ms() const noexcept {
  return impl_->initialization_ms();
}

NativeTimedInferenceOutput NativeTensorRtRunner::run_with_timing(
    const std::vector<std::int64_t>& input_shape,
    const std::vector<float>& input_values) {
  return impl_->run_with_timing(input_shape, input_values);
}

}  // namespace yolo_defect_cpp

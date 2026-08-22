#include "yolo_defect_cpp/model_metadata.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace yolo_defect_cpp {
namespace {

[[noreturn]] void throw_contract_error(
    const RuntimeContract& contract,
    const std::string& object,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  throw std::runtime_error(
      "Model metadata contract error for '" + object + "' in model '" +
      contract.artifact.model_path.string() + "': expected " + expected +
      "; actual " + actual + "; action: " + action + ".");
}

void validate_provider(const ModelMetadata& actual,
                       const RuntimeContract& expected) {
  const std::string expected_provider =
      expected.runtime.provider == ExecutionProvider::kCpu
          ? "CPUExecutionProvider"
          : "<unsupported-provider>";
  const bool provider_is_available =
      std::find(actual.available_providers.begin(),
                actual.available_providers.end(),
                expected_provider) != actual.available_providers.end();
  if (!provider_is_available) {
    throw_contract_error(
        expected, "runtime.available_providers",
        "a list containing " + expected_provider,
        format_string_list(actual.available_providers),
        "verify that the loaded ONNX Runtime DLL matches the configured SDK "
        "and includes the requested execution provider");
  }
  if (actual.session_provider != expected_provider) {
    throw_contract_error(
        expected, "session.provider", expected_provider,
        actual.session_provider,
        "check provider selection in OnnxRunner and the RuntimeConfig "
        "provider value");
  }
}

void validate_tensor_common(const TensorMetadata& actual,
                            const TensorSpec& expected_tensor,
                            const RuntimeContract& expected,
                            const std::string& object) {
  if (actual.name != expected_tensor.name) {
    throw_contract_error(
        expected, object + ".name", "'" + expected_tensor.name + "'",
        "'" + actual.name + "'",
        "verify the exported ONNX tensor name and update the artifact "
        "declaration only when it describes the same model");
  }
  if (actual.value_type != ModelValueType::kTensor) {
    throw_contract_error(
        expected, object + ".value_type", "tensor",
        to_string(actual.value_type),
        "export a tensor input/output or add an explicit adapter for the "
        "observed ONNX value type");
  }
  if (actual.dtype != ObservedTensorDataType::kFloat32) {
    throw_contract_error(
        expected, object + ".dtype", "float32", to_string(actual.dtype),
        "verify the ONNX export dtype; schema v1 accepts only float32 "
        "tensors");
  }
}

void validate_input(const TensorMetadata& actual,
                    const RuntimeContract& expected) {
  validate_tensor_common(actual, expected.artifact.input, expected,
                         "input[0]");
  if (actual.shape.size() != 4) {
    throw_contract_error(
        expected, "input[0].shape",
        "static NCHW rank 4 " + format_shape(expected.artifact.input.shape),
        format_shape(actual.shape),
        "re-export the model with one static NCHW image input");
  }
  if (actual.shape[0] != 1 || actual.shape[1] != 3) {
    throw_contract_error(
        expected, "input[0].shape", "batch=1 and channels=3 in NCHW order",
        format_shape(actual.shape),
        "verify the model export and the artifact input layout declaration");
  }
  if (actual.shape != expected.artifact.input.shape) {
    throw_contract_error(
        expected, "input[0].shape",
        format_shape(expected.artifact.input.shape),
        format_shape(actual.shape),
        "verify the configured input height/width and the actual exported "
        "model; dynamic or different dimensions require a new contract");
  }
}

void validate_output(const TensorMetadata& actual,
                     const RuntimeContract& expected) {
  validate_tensor_common(actual, expected.artifact.output, expected,
                         "output[0]");
  if (actual.shape.size() != 3) {
    throw_contract_error(
        expected, "output[0].shape",
        "static BCN rank 3 " + format_shape(expected.artifact.output.shape),
        format_shape(actual.shape),
        "verify that this is the declared YOLOv8 raw-output export");
  }
  if (actual.shape[0] != 1) {
    throw_contract_error(
        expected, "output[0].batch", "1",
        std::to_string(actual.shape[0]),
        "export a batch-1 model or define and implement a different runtime "
        "contract");
  }

  const std::int64_t expected_channels =
      4 + static_cast<std::int64_t>(expected.artifact.class_names.size());
  if (actual.shape[1] != expected_channels) {
    throw_contract_error(
        expected, "output[0].class_channels",
        "4 + class_count = " + std::to_string(expected_channels),
        std::to_string(actual.shape[1]),
        "verify the class_names order/count and the model export output "
        "channels");
  }
  if (actual.shape[2] <= 0) {
    throw_contract_error(
        expected, "output[0].prediction_count", "a static value > 0",
        std::to_string(actual.shape[2]),
        "re-export with a static positive prediction dimension");
  }
  if (actual.shape != expected.artifact.output.shape) {
    throw_contract_error(
        expected, "output[0].shape",
        format_shape(expected.artifact.output.shape),
        format_shape(actual.shape),
        "verify the input size/export settings and update the artifact "
        "declaration only from actual model evidence");
  }
}

}  // namespace

void validate_model_metadata(const ModelMetadata& actual,
                             const RuntimeContract& expected) {
  validate_provider(actual, expected);
  if (actual.inputs.size() != 1) {
    throw_contract_error(
        expected, "model.input_count", "1",
        std::to_string(actual.inputs.size()),
        "export a single-input model or implement an explicit multi-input "
        "adapter");
  }
  if (actual.outputs.size() != 1) {
    throw_contract_error(
        expected, "model.output_count", "1",
        std::to_string(actual.outputs.size()),
        "export a single-output model or implement an explicit multi-output "
        "adapter");
  }
  validate_input(actual.inputs.front(), expected);
  validate_output(actual.outputs.front(), expected);
}

std::string to_string(ModelValueType value) {
  switch (value) {
    case ModelValueType::kTensor:
      return "tensor";
    case ModelValueType::kSequence:
      return "sequence";
    case ModelValueType::kMap:
      return "map";
    case ModelValueType::kOptional:
      return "optional";
    case ModelValueType::kUnknown:
      return "unknown";
  }
  throw std::logic_error("Unknown ModelValueType enum value.");
}

std::string to_string(ObservedTensorDataType value) {
  switch (value) {
    case ObservedTensorDataType::kUndefined:
      return "undefined";
    case ObservedTensorDataType::kFloat32:
      return "float32";
    case ObservedTensorDataType::kUInt8:
      return "uint8";
    case ObservedTensorDataType::kInt8:
      return "int8";
    case ObservedTensorDataType::kUInt16:
      return "uint16";
    case ObservedTensorDataType::kInt16:
      return "int16";
    case ObservedTensorDataType::kInt32:
      return "int32";
    case ObservedTensorDataType::kInt64:
      return "int64";
    case ObservedTensorDataType::kString:
      return "string";
    case ObservedTensorDataType::kBool:
      return "bool";
    case ObservedTensorDataType::kFloat16:
      return "float16";
    case ObservedTensorDataType::kFloat64:
      return "float64";
    case ObservedTensorDataType::kUInt32:
      return "uint32";
    case ObservedTensorDataType::kUInt64:
      return "uint64";
    case ObservedTensorDataType::kComplex64:
      return "complex64";
    case ObservedTensorDataType::kComplex128:
      return "complex128";
    case ObservedTensorDataType::kBFloat16:
      return "bfloat16";
    case ObservedTensorDataType::kUnsupported:
      return "unsupported";
  }
  throw std::logic_error("Unknown ObservedTensorDataType enum value.");
}

std::string format_shape(const std::vector<std::int64_t>& shape) {
  std::ostringstream output;
  output << "[";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index > 0) {
      output << ",";
    }
    output << shape[index];
  }
  output << "]";
  return output.str();
}

std::string format_string_list(const std::vector<std::string>& values) {
  std::ostringstream output;
  output << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      output << ",";
    }
    output << values[index];
  }
  output << "]";
  return output.str();
}

}  // namespace yolo_defect_cpp

#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/model_metadata.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

yolo_defect_cpp::ModelMetadata make_valid_metadata(
    const yolo_defect_cpp::RuntimeContract& contract) {
  yolo_defect_cpp::ModelMetadata metadata;
  if (contract.runtime.provider ==
      yolo_defect_cpp::ExecutionProvider::kTensorRt) {
    metadata.ort_version = "synthetic-1.20.1";
    metadata.available_providers = {
        "TensorrtExecutionProvider", "CUDAExecutionProvider",
        "CPUExecutionProvider"};
    metadata.session_provider = "TensorrtExecutionProvider";
    metadata.registered_provider_chain = {
        "TensorrtExecutionProvider", "CUDAExecutionProvider",
        "CPUExecutionProvider"};
    metadata.inference_precision =
        yolo_defect_cpp::to_string(contract.runtime.tensorrt->precision);
    metadata.device_id = contract.runtime.tensorrt->device_id;
    metadata.engine_cache_enabled = true;
    metadata.engine_cache_path =
        contract.runtime.tensorrt->engine_cache_path.string();
    metadata.engine_cache_prefix = "synthetic_contract_test";
    metadata.engine_cache_state = "synthetic";
  } else if (contract.runtime.provider ==
             yolo_defect_cpp::ExecutionProvider::kTensorRtNative) {
    metadata.ort_version = "synthetic-1.20.1-inventory-only";
    metadata.available_providers = {"TensorRTNative"};
    metadata.session_provider = "TensorRTNative";
    metadata.registered_provider_chain = {"TensorRTNative"};
    metadata.inference_precision =
        yolo_defect_cpp::to_string(contract.runtime.tensorrt->precision);
    metadata.device_id = contract.runtime.tensorrt->device_id;
    metadata.engine_cache_enabled = true;
    metadata.engine_cache_path =
        contract.runtime.tensorrt->engine_cache_path.string();
    metadata.engine_cache_prefix = contract.runtime.tensorrt
                                       ->native_engine_path->filename()
                                       .string();
    metadata.engine_cache_state = "frozen_native_engine_loaded";
  } else {
    metadata.ort_version = "synthetic-1.19.2";
    metadata.available_providers = {"CPUExecutionProvider"};
    metadata.session_provider = "CPUExecutionProvider";
    metadata.registered_provider_chain = {"CPUExecutionProvider"};
    metadata.inference_precision = "fp32";
  }
  metadata.provider_evidence = "synthetic_contract_test";
  if (contract.runtime.provider ==
      yolo_defect_cpp::ExecutionProvider::kTensorRtNative) {
    metadata.intra_op_num_threads = 0;
    metadata.inter_op_num_threads = 0;
    metadata.execution_mode = "synchronous_non_default_cuda_stream";
    metadata.graph_optimization_level = "frozen_engine_build_time";
  } else {
    metadata.intra_op_num_threads = 1;
    metadata.inter_op_num_threads = 1;
    metadata.execution_mode = "sequential";
    metadata.graph_optimization_level = "all";
  }

  yolo_defect_cpp::TensorMetadata input;
  input.name = contract.artifact.input.name;
  input.value_type = yolo_defect_cpp::ModelValueType::kTensor;
  input.dtype = yolo_defect_cpp::ObservedTensorDataType::kFloat32;
  input.shape = contract.artifact.input.shape;
  metadata.inputs.push_back(input);

  yolo_defect_cpp::TensorMetadata output;
  output.name = contract.artifact.output.name;
  output.value_type = yolo_defect_cpp::ModelValueType::kTensor;
  output.dtype = yolo_defect_cpp::ObservedTensorDataType::kFloat32;
  output.shape = contract.artifact.output.shape;
  metadata.outputs.push_back(output);
  return metadata;
}

std::string mutate_for_case(
    const std::string& case_name,
    yolo_defect_cpp::ModelMetadata& metadata) {
  if (case_name == "valid" || case_name == "native_valid") {
    return "";
  }
  if (case_name == "input_count_mismatch") {
    metadata.inputs.push_back(metadata.inputs.front());
    return "model.input_count";
  }
  if (case_name == "output_count_mismatch") {
    metadata.outputs.clear();
    return "model.output_count";
  }
  if (case_name == "input_name_mismatch") {
    metadata.inputs.front().name = "wrong_images";
    return "input[0].name";
  }
  if (case_name == "input_shape_mismatch") {
    metadata.inputs.front().shape = {1, 3, 640, 640};
    return "input[0].shape";
  }
  if (case_name == "input_dtype_mismatch") {
    metadata.inputs.front().dtype =
        yolo_defect_cpp::ObservedTensorDataType::kFloat16;
    return "input[0].dtype";
  }
  if (case_name == "output_dtype_mismatch") {
    metadata.outputs.front().dtype =
        yolo_defect_cpp::ObservedTensorDataType::kFloat16;
    return "output[0].dtype";
  }
  if (case_name == "output_name_mismatch") {
    metadata.outputs.front().name = "wrong_output";
    return "output[0].name";
  }
  if (case_name == "output_shape_mismatch") {
    --metadata.outputs.front().shape.back();
    return "output[0].shape";
  }
  if (case_name == "class_count_mismatch") {
    --metadata.outputs.front().shape[1];
    return "output[0].class_channels";
  }
  if (case_name == "provider_unavailable") {
    metadata.available_providers.clear();
    return "runtime.available_providers";
  }
  if (case_name == "session_provider_mismatch") {
    metadata.session_provider = "CUDAExecutionProvider";
    return "session.provider";
  }
  if (case_name == "tensorrt_missing_cuda") {
    metadata.available_providers = {
        "TensorrtExecutionProvider", "CPUExecutionProvider"};
    return "runtime.available_providers";
  }
  if (case_name == "tensorrt_wrong_chain") {
    metadata.registered_provider_chain = {
        "CUDAExecutionProvider", "TensorrtExecutionProvider",
        "CPUExecutionProvider"};
    return "session.registered_provider_chain";
  }
  if (case_name == "tensorrt_precision_mismatch") {
    metadata.inference_precision = "fp32";
    return "session.inference_precision";
  }
  if (case_name == "native_wrong_chain") {
    metadata.registered_provider_chain = {"TensorRTNative",
                                          "CPUExecutionProvider"};
    return "session.registered_provider_chain";
  }
  if (case_name == "native_wrong_cache") {
    metadata.engine_cache_path += "_wrong";
    return "session.native_engine";
  }
  if (case_name == "native_wrong_engine") {
    metadata.engine_cache_prefix = "wrong.engine";
    return "session.native_engine";
  }
  if (case_name == "native_precision_mismatch") {
    metadata.inference_precision = "fp32";
    return "session.inference_precision";
  }
  if (case_name == "native_execution_policy_mismatch") {
    metadata.execution_mode = "sequential";
    return "session.native_execution_policy";
  }
  throw std::runtime_error("Unknown synthetic metadata test case: " +
                           case_name);
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <runtime_config> <case_name>\n";
    return 2;
  }

  try {
    yolo_defect_cpp::RuntimeContract contract =
        yolo_defect_cpp::load_runtime_contract(argv[1]);
    const std::string case_name = argv[2];
    yolo_defect_cpp::ModelMetadata metadata =
        make_valid_metadata(contract);
    const std::string expected_error =
        mutate_for_case(case_name, metadata);

    if (case_name == "valid" || case_name == "native_valid") {
      yolo_defect_cpp::validate_model_metadata(metadata, contract);
      std::cout << "Synthetic metadata contract validation passed.\n";
      return 0;
    }

    try {
      yolo_defect_cpp::validate_model_metadata(metadata, contract);
    } catch (const std::exception& error) {
      const std::string message = error.what();
      if (message.find(expected_error) == std::string::npos ||
          message.find("expected") == std::string::npos ||
          message.find("actual") == std::string::npos ||
          message.find("action:") == std::string::npos) {
        std::cerr << "Observed the wrong validation error: " << message
                  << "\n";
        return 1;
      }
      std::cout << "Observed expected actionable error: " << message
                << "\n";
      return 0;
    }

    std::cerr << "Expected metadata validation failure for case '"
              << case_name << "', but validation passed.\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Metadata validator test setup failed: " << error.what()
              << "\n";
    return 1;
  }
}

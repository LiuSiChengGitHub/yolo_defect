#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/detector_pipeline.h"
#include "yolo_defect_cpp/image_preprocessor.h"
#include "yolo_defect_cpp/onnx_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  bool show_help = false;
  bool inspect_model = false;
  bool raw_output_summary = false;
  bool overwrite_existing = false;
  std::string config_path;
  std::string image_path;
  std::string output_json_path;
  std::string output_image_path;
};

struct NumericSummary {
  std::size_t finite_values = 0;
  float minimum = std::numeric_limits<float>::infinity();
  float maximum = -std::numeric_limits<float>::infinity();
};

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

std::string format_class_names(const std::vector<std::string>& class_names) {
  std::ostringstream output;
  for (std::size_t index = 0; index < class_names.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << class_names[index];
  }
  return output.str();
}

void print_help(const char* program_name) {
  std::cout
      << "yolo_defect_cpp - S1-05 single-image detection CLI\n"
      << "\n"
      << "Usage:\n"
      << "  " << program_name << " [--help]\n"
      << "  " << program_name << " --config <config_path>\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path>\n"
      << "  " << program_name
      << " --config <config_path> --inspect-model\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path>"
         " --raw-output-summary\n"
      << "  " << program_name
      << " --config <config_path> --image <image_path>"
         " [--output-json <path>] [--output-image <path>] [--overwrite]\n"
      << "\n"
      << "Scope:\n"
      << "  Loads and validates RuntimeConfig + ModelArtifactSpec.\n"
      << "  Optional --image keeps the existing OpenCV preprocess smoke.\n"
      << "  --inspect-model creates an ORT CPU session, reads actual model\n"
      << "  metadata, and validates it against the artifact contract.\n"
      << "  --raw-output-summary preprocesses one image, runs one synchronous\n"
      << "  raw inference, and prints only bounded tensor summaries.\n"
      << "  --output-json and --output-image run the S1-05 single-image\n"
      << "  pipeline: preprocess, ORT inference, tested postprocess, and files.\n"
      << "  Output parents are created recursively. Existing regular files are\n"
      << "  rejected unless --overwrite is explicit; paths matching protected\n"
      << "  inputs are rejected before writing. Relative CLI image/output paths\n"
      << "  use the current working\n"
      << "  directory. No GUI, batch, concurrency, service, or benchmark exists.\n";
}

void print_banner() {
  std::cout
      << "yolo_defect_cpp - S1-05 single-image detection CLI\n"
      << "V2 Runtime: industrial vision AI deployment workspace\n"
      << "Current scope: contract + preprocess + ORT + postprocess + stable "
         "single-image JSON/visualization\n"
      << "Run with --help for the reproducible single-image command.\n"
      << "Batch, concurrency, service, consistency, and benchmark are not "
         "part of S1-05.\n";
}

void print_contract_summary(
    const yolo_defect_cpp::RuntimeContract& contract) {
  const yolo_defect_cpp::RuntimeConfig& runtime = contract.runtime;
  const yolo_defect_cpp::ModelArtifactSpec& artifact = contract.artifact;

  std::cout
      << "S1-01 Runtime/artifact contract summary\n"
      << "runtime_config_path: " << runtime.declaration_path.string() << "\n"
      << "runtime_schema_version: " << runtime.schema_version << "\n"
      << "artifact_spec_path: " << runtime.artifact_spec_path.string() << "\n"
      << "artifact_schema_version: " << artifact.schema_version << "\n"
      << "model_id: " << artifact.model_id << "\n"
      << "model_family: "
      << yolo_defect_cpp::to_string(artifact.model_family) << "\n"
      << "model_path: " << artifact.model_path.string() << "\n"
      << "declared_model_sha256: " << artifact.model_sha256 << "\n"
      << "opset: " << artifact.opset << "\n"
      << "source: " << artifact.source << "\n"
      << "provenance: " << artifact.provenance << "\n"
      << "artifact_license: " << artifact.artifact_license << "\n"
      << "input_name: " << artifact.input.name << "\n"
      << "input_shape: " << format_shape(artifact.input.shape) << "\n"
      << "input_dtype: "
      << yolo_defect_cpp::to_string(artifact.input.dtype) << "\n"
      << "input_layout: "
      << yolo_defect_cpp::to_string(artifact.input.layout) << "\n"
      << "output_name: " << artifact.output.name << "\n"
      << "output_shape: " << format_shape(artifact.output.shape) << "\n"
      << "output_dtype: "
      << yolo_defect_cpp::to_string(artifact.output.dtype) << "\n"
      << "output_layout: "
      << yolo_defect_cpp::to_string(artifact.output.layout) << "\n"
      << "class_count: " << artifact.class_names.size() << "\n"
      << "class_names: " << format_class_names(artifact.class_names) << "\n"
      << std::fixed << std::setprecision(6)
      << "score_threshold: " << runtime.score_threshold << "\n"
      << "nms_threshold: " << runtime.nms_threshold << "\n"
      << "configured_provider: "
      << yolo_defect_cpp::to_string(runtime.provider) << "\n"
      << "preprocess_type: "
      << yolo_defect_cpp::to_string(artifact.preprocess_type) << "\n"
      << "postprocess_type: "
      << yolo_defect_cpp::to_string(artifact.postprocess_type) << "\n"
      << "nms_mode: " << yolo_defect_cpp::to_string(artifact.nms_mode) << "\n"
      << "scope: declaration contract only; this command did not create an "
         "ONNX Runtime session or run inference.\n";
}

void print_preprocess_summary(
    const std::string& image_path,
    const yolo_defect_cpp::RuntimeContract& contract,
    const yolo_defect_cpp::PreprocessResult& result) {
  std::cout
      << "S1-01 Preprocess summary\n"
      << "runtime_config_path: "
      << contract.runtime.declaration_path.string() << "\n"
      << "artifact_spec_path: "
      << contract.artifact.declaration_path.string() << "\n"
      << "model_id: " << contract.artifact.model_id << "\n"
      << "image_path: " << image_path << "\n"
      << "original_size: " << result.original_width << "x"
      << result.original_height << "\n"
      << "channels: " << result.original_channels << "\n"
      << "input_size: " << result.input_width << "x" << result.input_height
      << "\n"
      << "resized_size: " << result.resized_width << "x"
      << result.resized_height << "\n"
      << std::fixed << std::setprecision(6)
      << "scale: " << result.scale << "\n"
      << "padding: left=" << result.pad_left
      << ", top=" << result.pad_top
      << ", right=" << result.pad_right
      << ", bottom=" << result.pad_bottom << "\n"
      << "color: BGR->RGB\n"
      << "normalization: float32 [0, 1]\n"
      << "layout: NCHW\n"
      << "tensor_shape: 1x3x" << result.input_height << "x"
      << result.input_width << "\n"
      << "tensor_elements: " << result.tensor_nchw.size() << "\n"
      << "scope: contract + preprocess only; this command did not create an "
         "ONNX Runtime session or run inference.\n";
}

void print_model_metadata_summary(
    const yolo_defect_cpp::RuntimeContract& contract,
    const yolo_defect_cpp::ModelMetadata& metadata) {
  std::cout
      << "S1-02 ONNX model inspection\n"
      << "model_path: " << contract.artifact.model_path.string() << "\n"
      << "ort_version: " << metadata.ort_version << "\n"
      << "configured_provider: "
      << yolo_defect_cpp::to_string(contract.runtime.provider) << "\n"
      << "available_providers: "
      << yolo_defect_cpp::format_string_list(
             metadata.available_providers)
      << "\n"
      << "session_provider: " << metadata.session_provider << "\n"
      << "provider_evidence: " << metadata.provider_evidence << "\n"
      << "execution_mode: " << metadata.execution_mode << "\n"
      << "intra_op_num_threads: " << metadata.intra_op_num_threads << "\n"
      << "inter_op_num_threads: " << metadata.inter_op_num_threads
      << " (not used by sequential execution mode)\n"
      << "graph_optimization_level: "
      << metadata.graph_optimization_level << "\n"
      << "input_count: " << metadata.inputs.size() << "\n";

  for (std::size_t index = 0; index < metadata.inputs.size(); ++index) {
    const yolo_defect_cpp::TensorMetadata& input =
        metadata.inputs[index];
    std::cout
        << "input[" << index << "].name: " << input.name << "\n"
        << "input[" << index << "].value_type: "
        << yolo_defect_cpp::to_string(input.value_type) << "\n"
        << "input[" << index << "].shape: "
        << yolo_defect_cpp::format_shape(input.shape) << "\n"
        << "input[" << index << "].dtype: "
        << yolo_defect_cpp::to_string(input.dtype) << "\n";
  }

  std::cout << "output_count: " << metadata.outputs.size() << "\n";
  for (std::size_t index = 0; index < metadata.outputs.size(); ++index) {
    const yolo_defect_cpp::TensorMetadata& output =
        metadata.outputs[index];
    std::cout
        << "output[" << index << "].name: " << output.name << "\n"
        << "output[" << index << "].value_type: "
        << yolo_defect_cpp::to_string(output.value_type) << "\n"
        << "output[" << index << "].shape: "
        << yolo_defect_cpp::format_shape(output.shape) << "\n"
        << "output[" << index << "].dtype: "
        << yolo_defect_cpp::to_string(output.dtype) << "\n";
  }

  std::cout
      << "metadata_contract_validation: passed\n"
      << "scope: session creation + metadata validation only; no input "
         "tensor, Session::Run, inference, or postprocess.\n";
}

NumericSummary summarize_values(const std::vector<float>& values) {
  NumericSummary summary;
  for (float value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    ++summary.finite_values;
    summary.minimum = std::min(summary.minimum, value);
    summary.maximum = std::max(summary.maximum, value);
  }
  return summary;
}

void print_raw_output_summary(
    const std::vector<std::int64_t>& input_shape,
    const std::vector<float>& input_values,
    const yolo_defect_cpp::InferenceOutput& output) {
  const NumericSummary input_summary = summarize_values(input_values);
  const NumericSummary output_summary = summarize_values(output.values);

  std::cout
      << "S1-03 raw output summary\n"
      << "input_shape: " << format_shape(input_shape) << "\n"
      << "input_elements: " << input_values.size() << "\n"
      << "input_finite_values: " << input_summary.finite_values << "/"
      << input_values.size() << "\n"
      << std::setprecision(9)
      << "input_min: " << input_summary.minimum << "\n"
      << "input_max: " << input_summary.maximum << "\n"
      << "output_shape: "
      << yolo_defect_cpp::format_shape(output.shape) << "\n"
      << "output_elements: " << output.values.size() << "\n"
      << "output_finite_values: " << output_summary.finite_values << "/"
      << output.values.size() << "\n"
      << "output_min: " << output_summary.minimum << "\n"
      << "output_max: " << output_summary.maximum << "\n"
      << "session_run: completed\n"
      << "raw_output_ownership: copied_to_InferenceOutput\n"
      << "scope: raw inference only; no decode, NMS, JSON, visualization, "
         "or benchmark.\n";
}

void print_single_image_summary(
    const yolo_defect_cpp::SingleImagePipelineResult& result,
    bool overwrite_existing) {
  const yolo_defect_cpp::SingleImageDetectionResult& detection =
      result.detection_result;
  std::cout
      << "S1-05 single-image detection completed\n"
      << "schema_version: " << detection.schema_version << "\n"
      << "model_id: " << detection.model_id << "\n"
      << "image_path: " << detection.image.source_path.string() << "\n"
      << "original_size: " << detection.image.original_width << "x"
      << detection.image.original_height << "\n"
      << "input_size: " << detection.image.input_width << "x"
      << detection.image.input_height << "\n"
      << "actual_provider: " << detection.actual_provider << "\n"
      << std::setprecision(9)
      << "score_threshold: " << detection.score_threshold << "\n"
      << "nms_threshold: " << detection.nms_threshold << "\n"
      << "nms_mode: " << yolo_defect_cpp::to_string(detection.nms_mode)
      << "\n"
      << "detection_count: " << detection.detections.size() << "\n"
      << "output_json: "
      << (result.outputs.json_path.has_value()
              ? result.outputs.json_path->string()
              : "<not requested>")
      << "\n"
      << "output_image: "
      << (result.outputs.image_path.has_value()
              ? result.outputs.image_path->string()
              : "<not requested>")
      << "\n"
      << "overwrite_existing: "
      << (overwrite_existing ? "true" : "false") << "\n"
      << "scope: one image only; stable JSON/visualization written without "
         "GUI. No batch, concurrency, service, consistency, or benchmark.\n";
}

CliOptions parse_cli(int argc, char* argv[]) {
  CliOptions options;
  const auto read_path_value = [argc, argv](int& index,
                                             const std::string& option) {
    if (index + 1 >= argc) {
      throw std::runtime_error(
          "CLI argument error: object=" + option +
          "; expected=one non-empty path; actual=missing value; action="
          "provide a path immediately after " + option + ".");
    }
    const std::string value = argv[index + 1];
    if (value.empty() || value.rfind("--", 0) == 0) {
      throw std::runtime_error(
          "CLI argument error: object=" + option +
          "; expected=one non-empty path; actual='" + value +
          "'; action=provide a path immediately after " + option + ".");
    }
    ++index;
    return value;
  };

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      options.show_help = true;
      continue;
    }

    if (argument == "--config") {
      if (!options.config_path.empty()) {
        throw std::runtime_error("--config was provided more than once.");
      }
      options.config_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--image") {
      if (!options.image_path.empty()) {
        throw std::runtime_error("--image was provided more than once.");
      }
      options.image_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--output-json") {
      if (!options.output_json_path.empty()) {
        throw std::runtime_error(
            "--output-json was provided more than once.");
      }
      options.output_json_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--output-image") {
      if (!options.output_image_path.empty()) {
        throw std::runtime_error(
            "--output-image was provided more than once.");
      }
      options.output_image_path = read_path_value(index, argument);
      continue;
    }

    if (argument == "--overwrite") {
      if (options.overwrite_existing) {
        throw std::runtime_error("--overwrite was provided more than once.");
      }
      options.overwrite_existing = true;
      continue;
    }

    if (argument == "--inspect-model") {
      if (options.inspect_model) {
        throw std::runtime_error(
            "--inspect-model was provided more than once.");
      }
      options.inspect_model = true;
      continue;
    }

    if (argument == "--raw-output-summary") {
      if (options.raw_output_summary) {
        throw std::runtime_error(
            "--raw-output-summary was provided more than once.");
      }
      options.raw_output_summary = true;
      continue;
    }

    throw std::runtime_error("Unknown argument: " + argument);
  }

  if (!options.image_path.empty() && options.config_path.empty()) {
    throw std::runtime_error("--image requires --config.");
  }
  if (options.inspect_model && options.config_path.empty()) {
    throw std::runtime_error("--inspect-model requires --config.");
  }
  if (options.raw_output_summary && options.config_path.empty()) {
    throw std::runtime_error("--raw-output-summary requires --config.");
  }
  if (options.raw_output_summary && options.image_path.empty()) {
    throw std::runtime_error("--raw-output-summary requires --image.");
  }
  const bool output_requested = !options.output_json_path.empty() ||
                                !options.output_image_path.empty();
  if (output_requested && options.config_path.empty()) {
    throw std::runtime_error(
        "--output-json/--output-image require --config.");
  }
  if (output_requested && options.inspect_model) {
    throw std::runtime_error(
        "--output-json/--output-image and --inspect-model are mutually "
        "exclusive.");
  }
  if (output_requested && options.raw_output_summary) {
    throw std::runtime_error(
        "--output-json/--output-image and --raw-output-summary are mutually "
        "exclusive.");
  }
  if (output_requested && options.image_path.empty()) {
    throw std::runtime_error(
        "--output-json/--output-image require --image.");
  }
  if (options.overwrite_existing && !output_requested) {
    throw std::runtime_error(
        "--overwrite requires --output-json or --output-image.");
  }
  if (options.inspect_model && options.raw_output_summary) {
    throw std::runtime_error(
        "--inspect-model and --raw-output-summary are mutually exclusive.");
  }
  if (options.inspect_model && !options.image_path.empty()) {
    throw std::runtime_error(
        "--inspect-model and --image are mutually exclusive.");
  }
  return options;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const CliOptions options = parse_cli(argc, argv);
    if (options.show_help) {
      print_help(argv[0]);
      return 0;
    }

    if (!options.config_path.empty()) {
      const yolo_defect_cpp::RuntimeContract contract =
          yolo_defect_cpp::load_runtime_contract(options.config_path);
      if (options.inspect_model) {
        const yolo_defect_cpp::OnnxRunner runner(contract);
        print_model_metadata_summary(contract, runner.metadata());
      } else if (options.image_path.empty()) {
        print_contract_summary(contract);
      } else {
        const bool output_requested =
            !options.output_json_path.empty() ||
            !options.output_image_path.empty();
        if (output_requested) {
          yolo_defect_cpp::DetectionOutputRequest request;
          if (!options.output_json_path.empty()) {
            request.json_path = options.output_json_path;
          }
          if (!options.output_image_path.empty()) {
            request.image_path = options.output_image_path;
          }
          request.overwrite_existing = options.overwrite_existing;

          yolo_defect_cpp::DetectorPipeline pipeline(contract);
          const yolo_defect_cpp::SingleImagePipelineResult pipeline_result =
              pipeline.run(options.image_path, request);
          print_single_image_summary(
              pipeline_result, options.overwrite_existing);
        } else {
          yolo_defect_cpp::PreprocessResult result =
              yolo_defect_cpp::preprocess_image(options.image_path,
                                                contract.artifact);
          if (!options.raw_output_summary) {
            print_preprocess_summary(options.image_path, contract, result);
          } else {
            const std::vector<std::int64_t> input_shape = {
                1, 3, result.input_height, result.input_width};
            yolo_defect_cpp::InferenceOutput output;
            {
              yolo_defect_cpp::OnnxRunner runner(contract);
              output = runner.run(input_shape, result.tensor_nchw);
            }
            print_raw_output_summary(
                input_shape, result.tensor_nchw, output);
          }
        }
      }
      return 0;
    }

    print_banner();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n"
              << "Run with --help to see the current CLI scope.\n";
    return 1;
  }
}

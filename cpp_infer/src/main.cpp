#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/image_preprocessor.h"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct CliOptions {
  bool show_help = false;
  std::string config_path;
  std::string image_path;
};

void print_help(const char* program_name) {
  std::cout
      << "yolo_defect_cpp - P1-01 CMake skeleton\n"
      << "\n"
      << "Usage:\n"
      << "  " << program_name << " [--help]\n"
      << "  " << program_name << " --config <config_path>\n"
      << "  " << program_name << " --config <config_path> --image <image_path>\n"
      << "\n"
      << "Scope:\n"
      << "  This target keeps the P1-01 CMake skeleton, P1-02 ConfigLoader,\n"
      << "  and P1-03 OpenCV image preprocessing smoke path.\n"
      << "  It does not use ONNX Runtime, GTest, inference, postprocessing,\n"
      << "  NMS, or benchmark yet.\n";
}

void print_banner() {
  std::cout
      << "yolo_defect_cpp - P1-01 CMake skeleton\n"
      << "V2 Runtime: industrial vision AI deployment workspace\n"
      << "Current scope: C++17/CMake + ConfigLoader + OpenCV preprocess\n"
      << "Run with --config <config_path> to print a config summary.\n"
      << "Run with --config <config_path> --image <image_path> to preprocess an image.\n";
}

void print_config_summary(const std::string& config_path,
                          const yolo_defect_cpp::RuntimeConfig& config) {
  std::cout
      << "P1-02 Config summary\n"
      << "config_path: " << config_path << "\n"
      << "input_width: " << config.input_width << "\n"
      << "input_height: " << config.input_height << "\n"
      << "class_count: " << config.class_names.size() << "\n"
      << "class_names: ";

  for (std::size_t i = 0; i < config.class_names.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << config.class_names[i];
  }

  std::cout
      << "\n"
      << "score_threshold: " << config.score_threshold << "\n"
      << "nms_threshold: " << config.nms_threshold << "\n"
      << "backend: " << config.backend << "\n"
      << "scope: config summary only; pass --image to run P1-03 preprocess.\n";
}

void print_preprocess_summary(const std::string& config_path,
                              const std::string& image_path,
                              const yolo_defect_cpp::RuntimeConfig& config,
                              const yolo_defect_cpp::PreprocessResult& result) {
  std::cout
      << "P1-03 Preprocess summary\n"
      << "config_path: " << config_path << "\n"
      << "image_path: " << image_path << "\n"
      << "original_size: " << result.original_width << "x"
      << result.original_height << "\n"
      << "channels: " << result.original_channels << "\n"
      << "input_size: " << config.input_width << "x" << config.input_height << "\n"
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
      << "tensor_shape: 1x3x" << config.input_height << "x"
      << config.input_width << "\n"
      << "tensor_elements: " << result.tensor_nchw.size() << "\n"
      << "scope: preprocess only; ONNX Runtime/inference/postprocess/"
      << "NMS/benchmark are not wired yet.\n";
}

CliOptions parse_cli(int argc, char* argv[]) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      continue;
    }

    if (arg == "--config") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--config expects one config file path.");
      }
      if (!options.config_path.empty()) {
        throw std::runtime_error("--config was provided more than once.");
      }
      options.config_path = argv[++i];
      continue;
    }

    if (arg == "--image") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--image expects one image file path.");
      }
      if (!options.image_path.empty()) {
        throw std::runtime_error("--image was provided more than once.");
      }
      options.image_path = argv[++i];
      continue;
    }

    throw std::runtime_error("Unknown argument: " + arg);
  }

  if (!options.image_path.empty() && options.config_path.empty()) {
    throw std::runtime_error("--image requires --config.");
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
      const yolo_defect_cpp::RuntimeConfig config =
          yolo_defect_cpp::load_config(options.config_path);
      if (options.image_path.empty()) {
        print_config_summary(options.config_path, config);
      } else {
        const yolo_defect_cpp::PreprocessResult result =
            yolo_defect_cpp::preprocess_image(options.image_path, config);
        print_preprocess_summary(options.config_path, options.image_path, config, result);
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

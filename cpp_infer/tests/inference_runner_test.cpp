#include "yolo_defect_cpp/config_loader.h"
#include "yolo_defect_cpp/image_preprocessor.h"
#include "yolo_defect_cpp/onnx_runner.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <runtime_config> <image_path>\n";
    return 2;
  }

  try {
    const yolo_defect_cpp::RuntimeContract contract =
        yolo_defect_cpp::load_runtime_contract(argv[1]);
    yolo_defect_cpp::PreprocessResult input =
        yolo_defect_cpp::preprocess_image(argv[2], contract.artifact);
    const std::vector<std::int64_t> input_shape = {
        1, 3, input.input_height, input.input_width};

    const std::size_t expected_elements = input.tensor_nchw.size();
    input.tensor_nchw.pop_back();
    const std::size_t actual_elements = input.tensor_nchw.size();

    yolo_defect_cpp::OnnxRunner runner(contract);
    try {
      static_cast<void>(
          runner.run(input_shape, input.tensor_nchw));
    } catch (const std::exception& error) {
      const std::string message = error.what();
      const std::vector<std::string> required_texts = {
          "input.tensor_elements",
          "expected " + std::to_string(expected_elements),
          "actual " + std::to_string(actual_elements),
          "before Ort::Value creation and Session::Run",
          "action:"};
      for (const std::string& required_text : required_texts) {
        if (message.find(required_text) == std::string::npos) {
          std::cerr << "Wrong invalid-length error. Missing '"
                    << required_text << "' in:\n"
                    << message << "\n";
          return 1;
        }
      }
      std::cout
          << "Invalid input length was rejected before Session::Run: "
          << message << "\n";
      return 0;
    }

    std::cerr
        << "Expected invalid input length to fail before Session::Run.\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Inference runner test setup failed: "
              << error.what() << "\n";
    return 1;
  }
}

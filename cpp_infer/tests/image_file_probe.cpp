#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

int fail(const std::string& image_path, const std::string& expected,
         const std::string& actual, const std::string& action) {
  std::cerr << "Image probe error for object output_image '" << image_path
            << "': expected " << expected << "; actual " << actual
            << "; action: " << action << ".\n";
  return 1;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <fixed_sample_output_image>\n";
    return 2;
  }

  const std::filesystem::path image_path = argv[1];
  std::error_code path_error;
  const bool is_regular =
      std::filesystem::is_regular_file(image_path, path_error);
  if (path_error || !is_regular) {
    const std::string actual = path_error
        ? "filesystem error " + path_error.message()
        : "path is missing or is not a regular file";
    return fail(image_path.string(), "an existing regular image file",
                actual,
                "check --output-image and the output directory");
  }

  try {
    const cv::Mat image =
        cv::imread(image_path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
      return fail(image_path.string(),
                  "an OpenCV-readable encoded image",
                  "cv::imread returned an empty cv::Mat",
                  "check the encoded file contents and file extension");
    }
    if (image.cols != 200 || image.rows != 200) {
      return fail(image_path.string(), "size 200x200",
                  "size " + std::to_string(image.cols) + "x" +
                      std::to_string(image.rows),
                  "draw detections on the original image, not the 800x800 "
                  "model input");
    }
    if (image.type() != CV_8UC3) {
      return fail(image_path.string(), "OpenCV type CV_8UC3",
                  "type=" + std::to_string(image.type()) +
                      ", depth=" + std::to_string(image.depth()) +
                      ", channels=" + std::to_string(image.channels()),
                  "write a standard 8-bit three-channel BGR image");
    }

    std::cout << "S1-05 visualization OpenCV probe passed\n"
              << "image_path: " << image_path.string() << "\n"
              << "image_size: " << image.cols << "x" << image.rows << "\n"
              << "image_type: CV_8UC3\n";
    return 0;
  } catch (const cv::Exception& error) {
    return fail(image_path.string(), "successful cv::imread",
                std::string("OpenCV exception: ") + error.what(),
                "check image codec availability and output file integrity");
  } catch (const std::exception& error) {
    return fail(image_path.string(), "successful filesystem/image check",
                error.what(),
                "check the output path and generated image");
  }
}

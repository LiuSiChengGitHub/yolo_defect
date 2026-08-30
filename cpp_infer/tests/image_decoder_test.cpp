#include "image_decoder.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace yolo_defect_cpp {
namespace internal {
namespace {

class ScopedTestDirectory {
 public:
  ScopedTestDirectory() {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    path_ = std::filesystem::temp_directory_path() /
            ("yolo_defect_image_decoder_" + std::to_string(suffix));
    std::filesystem::create_directories(path_);
  }

  ~ScopedTestDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::filesystem::path unicode_image_name() {
#ifdef _WIN32
  return std::filesystem::path(L"\u4E2D\u6587\u56FE\u7247.png");
#else
  return std::filesystem::u8path(
      "\xE4\xB8\xAD\xE6\x96\x87\xE5\x9B\xBE\xE7\x89\x87.png");
#endif
}

void write_encoded_bytes(const std::filesystem::path& path,
                         const std::vector<unsigned char>& bytes) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output.is_open());
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(output.good());
}

TEST(ImageDecoderPathTest, DecodesColorImageFromUnicodeFilesystemPath) {
  ScopedTestDirectory directory;
  const std::filesystem::path image_path =
      directory.path() / unicode_image_name();

  cv::Mat source(2, 3, CV_8UC3);
  source.at<cv::Vec3b>(0, 0) = cv::Vec3b(1, 2, 3);
  source.at<cv::Vec3b>(0, 1) = cv::Vec3b(10, 20, 30);
  source.at<cv::Vec3b>(0, 2) = cv::Vec3b(40, 50, 60);
  source.at<cv::Vec3b>(1, 0) = cv::Vec3b(70, 80, 90);
  source.at<cv::Vec3b>(1, 1) = cv::Vec3b(100, 110, 120);
  source.at<cv::Vec3b>(1, 2) = cv::Vec3b(200, 210, 220);

  std::vector<unsigned char> encoded;
  ASSERT_TRUE(cv::imencode(".png", source, encoded));
  write_encoded_bytes(image_path, encoded);
  ASSERT_TRUE(std::filesystem::is_regular_file(image_path));

  const std::filesystem::path normalized = normalize_image_file(image_path);
  const DecodedBgrImage decoded = decode_normalized_bgr_image(normalized);

  ASSERT_FALSE(decoded.image.empty());
  EXPECT_EQ(decoded.image.type(), CV_8UC3);
  EXPECT_EQ(decoded.image.rows, source.rows);
  EXPECT_EQ(decoded.image.cols, source.cols);
  EXPECT_EQ(cv::norm(decoded.image, source, cv::NORM_INF), 0.0);
  EXPECT_GE(decoded.imread_ms, 0.0);
}

}  // namespace
}  // namespace internal
}  // namespace yolo_defect_cpp

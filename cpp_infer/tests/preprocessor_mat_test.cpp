#include "yolo_defect_cpp/image_preprocessor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

namespace yolo_defect_cpp {
namespace {

ModelArtifactSpec make_artifact(int input_height, int input_width) {
  ModelArtifactSpec artifact;
  artifact.input.dtype = TensorDataType::kFloat32;
  artifact.input.layout = TensorLayout::kNchw;
  artifact.input.shape = {1, 3, input_height, input_width};
  return artifact;
}

template <typename Callable>
std::string capture_runtime_error(Callable&& callable) {
  try {
    callable();
  } catch (const std::runtime_error& error) {
    return error.what();
  } catch (...) {
    ADD_FAILURE() << "Expected std::runtime_error, but another exception type "
                     "was thrown.";
    return {};
  }
  ADD_FAILURE() << "Expected std::runtime_error, but no exception was thrown.";
  return {};
}

float tensor_value(const PreprocessResult& result, int channel,
                   int y, int x) {
  const std::size_t plane =
      static_cast<std::size_t>(result.input_height) * result.input_width;
  return result.tensor_nchw[
      static_cast<std::size_t>(channel) * plane +
      static_cast<std::size_t>(y) * result.input_width + x];
}

TEST(PreprocessMatTest, ConvertsKnownBgrPixelsToRgbFloatNchw) {
  const ModelArtifactSpec artifact = make_artifact(2, 2);
  cv::Mat bgr_image(2, 2, CV_8UC3);
  bgr_image.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
  bgr_image.at<cv::Vec3b>(0, 1) = cv::Vec3b(40, 50, 60);
  bgr_image.at<cv::Vec3b>(1, 0) = cv::Vec3b(70, 80, 90);
  bgr_image.at<cv::Vec3b>(1, 1) = cv::Vec3b(100, 110, 120);

  const PreprocessResult result = preprocess_image(bgr_image, artifact);

  EXPECT_EQ(result.original_width, 2);
  EXPECT_EQ(result.original_height, 2);
  EXPECT_EQ(result.original_channels, 3);
  EXPECT_EQ(result.input_width, 2);
  EXPECT_EQ(result.input_height, 2);
  EXPECT_EQ(result.resized_width, 2);
  EXPECT_EQ(result.resized_height, 2);
  EXPECT_DOUBLE_EQ(result.scale, 1.0);
  EXPECT_EQ(result.pad_left, 0);
  EXPECT_EQ(result.pad_right, 0);
  EXPECT_EQ(result.pad_top, 0);
  EXPECT_EQ(result.pad_bottom, 0);
  ASSERT_EQ(result.tensor_nchw.size(), 12U);

  const float expected_nchw[] = {
      30.0F, 60.0F, 90.0F, 120.0F,
      20.0F, 50.0F, 80.0F, 110.0F,
      10.0F, 40.0F, 70.0F, 100.0F};
  for (std::size_t index = 0; index < result.tensor_nchw.size(); ++index) {
    EXPECT_NEAR(result.tensor_nchw[index], expected_nchw[index] / 255.0F,
                1.0e-6F)
        << "flattened NCHW index=" << index;
  }
}

TEST(PreprocessMatTest,
     LandscapeImageUsesOddBottomPaddingForNonSquareModelInput) {
  const ModelArtifactSpec artifact = make_artifact(6, 10);
  const cv::Mat bgr_image(4, 8, CV_8UC3, cv::Scalar(1, 2, 3));

  const PreprocessResult result = preprocess_image(bgr_image, artifact);

  EXPECT_EQ(result.original_width, 8);
  EXPECT_EQ(result.original_height, 4);
  EXPECT_EQ(result.input_width, 10);
  EXPECT_EQ(result.input_height, 6);
  EXPECT_DOUBLE_EQ(result.scale, 1.25);
  EXPECT_EQ(result.resized_width, 10);
  EXPECT_EQ(result.resized_height, 5);
  EXPECT_EQ(result.pad_left, 0);
  EXPECT_EQ(result.pad_right, 0);
  EXPECT_EQ(result.pad_top, 0);
  EXPECT_EQ(result.pad_bottom, 1);
  ASSERT_EQ(result.tensor_nchw.size(), 180U);

  for (int channel = 0; channel < 3; ++channel) {
    EXPECT_NEAR(tensor_value(result, channel, 5, 0), 114.0F / 255.0F,
                1.0e-6F);
  }
}

TEST(PreprocessMatTest,
     PortraitImageUsesOddRightPaddingForNonSquareModelInput) {
  const ModelArtifactSpec artifact = make_artifact(6, 10);
  const cv::Mat bgr_image(8, 4, CV_8UC3, cv::Scalar(1, 2, 3));

  const PreprocessResult result = preprocess_image(bgr_image, artifact);

  EXPECT_EQ(result.original_width, 4);
  EXPECT_EQ(result.original_height, 8);
  EXPECT_EQ(result.input_width, 10);
  EXPECT_EQ(result.input_height, 6);
  EXPECT_DOUBLE_EQ(result.scale, 0.75);
  EXPECT_EQ(result.resized_width, 3);
  EXPECT_EQ(result.resized_height, 6);
  EXPECT_EQ(result.pad_left, 3);
  EXPECT_EQ(result.pad_right, 4);
  EXPECT_EQ(result.pad_top, 0);
  EXPECT_EQ(result.pad_bottom, 0);
  ASSERT_EQ(result.tensor_nchw.size(), 180U);

  for (int channel = 0; channel < 3; ++channel) {
    EXPECT_NEAR(tensor_value(result, channel, 0, 0), 114.0F / 255.0F,
                1.0e-6F);
    EXPECT_NEAR(tensor_value(result, channel, 0, 9), 114.0F / 255.0F,
                1.0e-6F);
  }
}

TEST(PreprocessMatTest, EmptyMatrixReportsActionableError) {
  const ModelArtifactSpec artifact = make_artifact(6, 10);
  const cv::Mat empty_image;

  const std::string message = capture_runtime_error(
      [&] { preprocess_image(empty_image, artifact); });

  EXPECT_NE(message.find("empty cv::Mat"), std::string::npos) << message;
  EXPECT_NE(message.find("expected"), std::string::npos) << message;
  EXPECT_NE(message.find("Action:"), std::string::npos) << message;
}

TEST(PreprocessMatTest, NonThreeChannelMatrixReportsActionableError) {
  const ModelArtifactSpec artifact = make_artifact(6, 10);
  const cv::Mat grayscale_image(8, 4, CV_8UC1, cv::Scalar(10));

  const std::string message = capture_runtime_error(
      [&] { preprocess_image(grayscale_image, artifact); });

  EXPECT_NE(message.find("expected CV_8UC3"), std::string::npos)
      << message;
  EXPECT_NE(message.find("channels=1"), std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(PreprocessMatTest, FloatMatrixIsRejectedBeforeNormalization) {
  const ModelArtifactSpec artifact = make_artifact(6, 10);
  const cv::Mat float_image(8, 4, CV_32FC3, cv::Scalar(0.1F, 0.2F, 0.3F));

  const std::string message = capture_runtime_error(
      [&] { preprocess_image(float_image, artifact); });

  EXPECT_NE(message.find("object bgr_image.type"), std::string::npos)
      << message;
  EXPECT_NE(message.find("expected CV_8UC3"), std::string::npos)
      << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(PreprocessMatTest, InvalidModelInputDimensionsAreRejected) {
  const ModelArtifactSpec artifact = make_artifact(0, 10);
  const cv::Mat image(8, 4, CV_8UC3, cv::Scalar(1, 2, 3));

  const std::string message = capture_runtime_error(
      [&] { preprocess_image(image, artifact); });

  EXPECT_NE(message.find("object artifact.input.shape[2:4]"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("expected positive height/width"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

}  // namespace
}  // namespace yolo_defect_cpp

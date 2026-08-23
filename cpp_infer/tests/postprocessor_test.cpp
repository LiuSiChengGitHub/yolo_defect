#include "yolo_defect_cpp/postprocessor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

const std::vector<std::string> kSixClasses = {
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches"};

InferenceOutput make_output(std::size_t class_count,
                            std::size_t candidate_count) {
  InferenceOutput output;
  output.shape = {
      1,
      static_cast<std::int64_t>(4 + class_count),
      static_cast<std::int64_t>(candidate_count)};
  output.values.assign((4 + class_count) * candidate_count, 0.0F);
  return output;
}

void set_bcn_value(InferenceOutput& output, std::size_t channel,
                   std::size_t candidate, float value) {
  const std::size_t candidate_count =
      static_cast<std::size_t>(output.shape[2]);
  output.values[channel * candidate_count + candidate] = value;
}

Detection make_detection(int class_id, float confidence,
                         BoundingBox box) {
  Detection detection;
  detection.class_id = class_id;
  detection.class_name = "class_" + std::to_string(class_id);
  detection.confidence = confidence;
  detection.bbox_xyxy = box;
  return detection;
}

template <typename Callable>
std::string capture_runtime_error(Callable&& callable) {
  try {
    std::forward<Callable>(callable)();
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

void expect_actionable_error(const std::string& message,
                             const std::string& object,
                             const std::string& expected,
                             const std::string& actual) {
  EXPECT_NE(message.find(object), std::string::npos) << message;
  EXPECT_NE(message.find(expected), std::string::npos) << message;
  EXPECT_NE(message.find(actual), std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

PreprocessResult make_portrait_preprocess() {
  PreprocessResult preprocess;
  preprocess.original_width = 4;
  preprocess.original_height = 8;
  preprocess.original_channels = 3;
  preprocess.input_width = 10;
  preprocess.input_height = 6;
  preprocess.resized_width = 3;
  preprocess.resized_height = 6;
  preprocess.pad_left = 3;
  preprocess.pad_right = 4;
  preprocess.pad_top = 0;
  preprocess.pad_bottom = 0;
  preprocess.scale = 0.75;
  return preprocess;
}

RuntimeContract make_contract(std::size_t candidate_count,
                              std::vector<std::string> class_names,
                              double score_threshold = 0.25,
                              double nms_threshold = 0.45) {
  RuntimeContract contract;
  contract.runtime.score_threshold = score_threshold;
  contract.runtime.nms_threshold = nms_threshold;
  contract.artifact.model_family = ModelFamily::kYoloV8;
  contract.artifact.output.dtype = TensorDataType::kFloat32;
  contract.artifact.output.layout = TensorLayout::kBcn;
  contract.artifact.output.shape = {
      1,
      static_cast<std::int64_t>(4 + class_names.size()),
      static_cast<std::int64_t>(candidate_count)};
  contract.artifact.input.shape = {1, 3, 100, 100};
  contract.artifact.class_names = std::move(class_names);
  contract.artifact.postprocess_type = PostprocessType::kYoloV8Raw;
  contract.artifact.nms_mode = NmsMode::kClassAgnostic;
  return contract;
}

TEST(YoloDecodeTest, DecodesBcnLayoutAndClassArgmaxWithoutObjectness) {
  InferenceOutput output = make_output(kSixClasses.size(), 3);

  set_bcn_value(output, 0, 0, 20.0F);
  set_bcn_value(output, 1, 0, 30.0F);
  set_bcn_value(output, 2, 0, 10.0F);
  set_bcn_value(output, 3, 0, 8.0F);
  set_bcn_value(output, 4 + 1, 0, 0.1F);
  set_bcn_value(output, 4 + 2, 0, 0.8F);

  set_bcn_value(output, 0, 1, 60.0F);
  set_bcn_value(output, 1, 1, 50.0F);
  set_bcn_value(output, 2, 1, 20.0F);
  set_bcn_value(output, 3, 1, 10.0F);
  set_bcn_value(output, 4 + 5, 1, 0.9F);

  set_bcn_value(output, 0, 2, 100.0F);
  set_bcn_value(output, 1, 2, 100.0F);
  set_bcn_value(output, 2, 2, 4.0F);
  set_bcn_value(output, 3, 2, 4.0F);
  set_bcn_value(output, 4 + 1, 2, 0.25F);

  const std::vector<Detection> decoded =
      decode_yolov8_raw_output(output, kSixClasses, 0.25);

  ASSERT_EQ(decoded.size(), 2U);
  EXPECT_EQ(decoded[0].class_id, 2);
  EXPECT_EQ(decoded[0].class_name, "patches");
  EXPECT_FLOAT_EQ(decoded[0].confidence, 0.8F);
  EXPECT_FLOAT_EQ(decoded[0].bbox_xyxy.x1, 15.0F);
  EXPECT_FLOAT_EQ(decoded[0].bbox_xyxy.y1, 26.0F);
  EXPECT_FLOAT_EQ(decoded[0].bbox_xyxy.x2, 25.0F);
  EXPECT_FLOAT_EQ(decoded[0].bbox_xyxy.y2, 34.0F);

  EXPECT_EQ(decoded[1].class_id, 5);
  EXPECT_EQ(decoded[1].class_name, "scratches");
  EXPECT_FLOAT_EQ(decoded[1].confidence, 0.9F);
  EXPECT_FLOAT_EQ(decoded[1].bbox_xyxy.x1, 50.0F);
  EXPECT_FLOAT_EQ(decoded[1].bbox_xyxy.y1, 45.0F);
  EXPECT_FLOAT_EQ(decoded[1].bbox_xyxy.x2, 70.0F);
  EXPECT_FLOAT_EQ(decoded[1].bbox_xyxy.y2, 55.0F);
}

TEST(YoloDecodeTest, EqualClassScoresChooseTheLowerClassId) {
  InferenceOutput output = make_output(kSixClasses.size(), 1);
  set_bcn_value(output, 0, 0, 10.0F);
  set_bcn_value(output, 1, 0, 10.0F);
  set_bcn_value(output, 2, 0, 2.0F);
  set_bcn_value(output, 3, 0, 2.0F);
  set_bcn_value(output, 4 + 1, 0, 0.7F);
  set_bcn_value(output, 4 + 4, 0, 0.7F);

  const std::vector<Detection> decoded =
      decode_yolov8_raw_output(output, kSixClasses, 0.1);

  ASSERT_EQ(decoded.size(), 1U);
  EXPECT_EQ(decoded[0].class_id, 1);
  EXPECT_EQ(decoded[0].class_name, "inclusion");
  EXPECT_FLOAT_EQ(decoded[0].confidence, 0.7F);
}

TEST(YoloDecodeTest, StrictThresholdKeepsOnlyScoresGreaterThanThreshold) {
  InferenceOutput output = make_output(kSixClasses.size(), 3);
  for (std::size_t candidate = 0; candidate < 3; ++candidate) {
    set_bcn_value(output, 0, candidate, 10.0F);
    set_bcn_value(output, 1, candidate, 10.0F);
    set_bcn_value(output, 2, candidate, 2.0F);
    set_bcn_value(output, 3, candidate, 2.0F);
  }
  set_bcn_value(output, 4, 0, 0.24F);
  set_bcn_value(output, 4, 1, 0.25F);
  set_bcn_value(output, 4, 2, 0.26F);

  const std::vector<Detection> decoded =
      decode_yolov8_raw_output(output, kSixClasses, 0.25);

  ASSERT_EQ(decoded.size(), 1U);
  EXPECT_FLOAT_EQ(decoded[0].confidence, 0.26F);
}

TEST(YoloDecodeTest, ThresholdComparisonUsesFloat32BaselineDomain) {
  InferenceOutput output = make_output(kSixClasses.size(), 1);
  set_bcn_value(output, 0, 0, 10.0F);
  set_bcn_value(output, 1, 0, 10.0F);
  set_bcn_value(output, 2, 0, 2.0F);
  set_bcn_value(output, 3, 0, 2.0F);
  set_bcn_value(output, 4, 0, 0.3F);

  EXPECT_TRUE(
      decode_yolov8_raw_output(output, kSixClasses, 0.3).empty());
}

TEST(YoloDecodeTest, ReturnsEmptyWhenNoCandidatePasses) {
  InferenceOutput output = make_output(kSixClasses.size(), 2);
  set_bcn_value(output, 4, 0, 0.25F);
  set_bcn_value(output, 4, 1, 0.1F);

  EXPECT_TRUE(
      decode_yolov8_raw_output(output, kSixClasses, 0.25).empty());
}

TEST(YoloDecodeTest, ZeroCandidateTensorReturnsEmpty) {
  const InferenceOutput output = make_output(kSixClasses.size(), 0);

  EXPECT_NO_THROW(validate_yolov8_raw_output(output, kSixClasses.size()));
  EXPECT_TRUE(
      decode_yolov8_raw_output(output, kSixClasses, 0.25).empty());
}

TEST(YoloOutputValidationTest, WrongRankReportsActionableError) {
  InferenceOutput output;
  output.shape = {1, 10};
  output.values.assign(10, 0.0F);

  const std::string message = capture_runtime_error(
      [&output] { validate_yolov8_raw_output(output, 6); });

  expect_actionable_error(message, "object output.rank",
                          "expected 3 for [1,4+C,N]", "actual 2");
}

TEST(YoloOutputValidationTest, WrongChannelCountReportsActionableError) {
  InferenceOutput output;
  output.shape = {1, 9, 1};
  output.values.assign(9, 0.0F);

  const std::string message = capture_runtime_error(
      [&output] { validate_yolov8_raw_output(output, 6); });

  expect_actionable_error(message, "object output.channels",
                          "expected 4 + class_count = 10", "actual 9");
}

TEST(YoloOutputValidationTest, WrongBatchReportsActionableError) {
  InferenceOutput output;
  output.shape = {2, 10, 1};
  output.values.assign(20, 0.0F);

  const std::string message = capture_runtime_error(
      [&output] { validate_yolov8_raw_output(output, 6); });

  expect_actionable_error(message, "object output.batch", "expected 1",
                          "actual 2");
}

TEST(YoloOutputValidationTest, ElementCountMismatchReportsActionableError) {
  InferenceOutput output;
  output.shape = {1, 10, 2};
  output.values.assign(19, 0.0F);

  const std::string message = capture_runtime_error(
      [&output] { validate_yolov8_raw_output(output, 6); });

  expect_actionable_error(message, "object output.elements", "expected 20",
                          "actual 19");
}

TEST(YoloOutputValidationTest, NonFiniteValueReportsActionableError) {
  InferenceOutput output = make_output(kSixClasses.size(), 1);
  output.values[7] = std::numeric_limits<float>::quiet_NaN();

  const std::string message = capture_runtime_error(
      [&output] { validate_yolov8_raw_output(output, 6); });

  EXPECT_NE(message.find("object output.values[7]"), std::string::npos)
      << message;
  EXPECT_NE(message.find("expected a finite float32 value"),
            std::string::npos)
      << message;
  EXPECT_NE(message.find("actual"), std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(YoloOutputValidationTest, EmptyClassNamesReportActionableError) {
  const InferenceOutput output = make_output(1, 1);

  const std::string message = capture_runtime_error(
      [&output] { decode_yolov8_raw_output(output, {}, 0.25); });

  expect_actionable_error(message, "object class_names",
                          "expected at least one non-empty class",
                          "actual []");
}

TEST(BoxConversionTest, ConvertsCenterXywhToCornerXyxy) {
  const BoundingBox box = xywh_to_xyxy(20.0F, 30.0F, 10.0F, 8.0F);

  EXPECT_FLOAT_EQ(box.x1, 15.0F);
  EXPECT_FLOAT_EQ(box.y1, 26.0F);
  EXPECT_FLOAT_EQ(box.x2, 25.0F);
  EXPECT_FLOAT_EQ(box.y2, 34.0F);
}

TEST(IoUTest, ComputesIdenticalDisjointTouchingAndPartialOverlap) {
  const BoundingBox base{0.0F, 0.0F, 10.0F, 10.0F};

  EXPECT_FLOAT_EQ(intersection_over_union(base, base), 1.0F);
  EXPECT_FLOAT_EQ(
      intersection_over_union(base, {20.0F, 20.0F, 30.0F, 30.0F}),
      0.0F);
  EXPECT_FLOAT_EQ(
      intersection_over_union(base, {10.0F, 0.0F, 20.0F, 10.0F}),
      0.0F);
  EXPECT_NEAR(
      intersection_over_union(base, {5.0F, 0.0F, 15.0F, 10.0F}),
      1.0F / 3.0F, 1.0e-6F);
}

TEST(IoUTest, DegenerateBoxesReturnZeroInsteadOfNonFiniteValue) {
  const float iou = intersection_over_union(
      {1.0F, 1.0F, 1.0F, 5.0F}, {2.0F, 2.0F, 2.0F, 6.0F});

  EXPECT_FLOAT_EQ(iou, 0.0F);
  EXPECT_TRUE(std::isfinite(iou));
}

TEST(ClassAgnosticNmsTest, SuppressesHighOverlapAcrossDifferentClasses) {
  const std::vector<Detection> candidates = {
      make_detection(0, 0.9F, {0.0F, 0.0F, 10.0F, 10.0F}),
      make_detection(5, 0.8F, {1.0F, 1.0F, 11.0F, 11.0F}),
      make_detection(2, 0.7F, {20.0F, 20.0F, 30.0F, 30.0F})};

  const std::vector<Detection> kept =
      class_agnostic_nms(candidates, 0.45);

  ASSERT_EQ(kept.size(), 2U);
  EXPECT_EQ(kept[0].class_id, 0);
  EXPECT_EQ(kept[1].class_id, 2);
}

TEST(ClassAgnosticNmsTest, KeepsLowOverlapAndSortsByDescendingScore) {
  const std::vector<Detection> candidates = {
      make_detection(0, 0.6F, {0.0F, 0.0F, 10.0F, 10.0F}),
      make_detection(1, 0.9F, {20.0F, 20.0F, 30.0F, 30.0F})};

  const std::vector<Detection> kept =
      class_agnostic_nms(candidates, 0.45);

  ASSERT_EQ(kept.size(), 2U);
  EXPECT_EQ(kept[0].class_id, 1);
  EXPECT_EQ(kept[1].class_id, 0);
}

TEST(ClassAgnosticNmsTest, EqualScoresPreserveOriginalCandidateOrder) {
  const std::vector<Detection> candidates = {
      make_detection(3, 0.8F, {0.0F, 0.0F, 10.0F, 10.0F}),
      make_detection(1, 0.8F, {0.0F, 0.0F, 10.0F, 10.0F})};

  const std::vector<Detection> kept =
      class_agnostic_nms(candidates, 0.45);

  ASSERT_EQ(kept.size(), 1U);
  EXPECT_EQ(kept[0].class_id, 3);
}

TEST(ClassAgnosticNmsTest, EmptyInputReturnsEmpty) {
  EXPECT_TRUE(class_agnostic_nms({}, 0.45).empty());
}

TEST(ClassAgnosticNmsTest, IoUEqualToThresholdIsNotSuppressed) {
  const std::vector<Detection> candidates = {
      make_detection(0, 0.9F, {0.0F, 0.0F, 10.0F, 10.0F}),
      make_detection(1, 0.8F, {0.0F, 0.0F, 10.0F, 10.0F})};

  const std::vector<Detection> kept = class_agnostic_nms(candidates, 1.0);

  EXPECT_EQ(kept.size(), 2U);
}

TEST(ClassAgnosticNmsTest, ThresholdComparisonUsesFloat32BaselineDomain) {
  const std::vector<Detection> candidates = {
      make_detection(0, 0.9F, {0.0F, 0.0F, 13.0F, 10.0F}),
      make_detection(1, 0.8F, {7.0F, 0.0F, 20.0F, 10.0F})};

  ASSERT_FLOAT_EQ(
      intersection_over_union(candidates[0].bbox_xyxy,
                              candidates[1].bbox_xyxy),
      0.3F);
  EXPECT_EQ(class_agnostic_nms(candidates, 0.3).size(), 2U);
}

TEST(CoordinateRestoreTest, RestoresOddPaddingAndClipsToSourceBounds) {
  const PreprocessResult preprocess = make_portrait_preprocess();

  const BoundingBox restored = restore_letterbox_box(
      {3.75F, 0.75F, 5.25F, 5.25F}, preprocess);
  EXPECT_NEAR(restored.x1, 1.0F, 1.0e-6F);
  EXPECT_NEAR(restored.y1, 1.0F, 1.0e-6F);
  EXPECT_NEAR(restored.x2, 3.0F, 1.0e-6F);
  EXPECT_NEAR(restored.y2, 7.0F, 1.0e-6F);

  const BoundingBox clipped = restore_letterbox_box(
      {-100.0F, -100.0F, 100.0F, 100.0F}, preprocess);
  EXPECT_FLOAT_EQ(clipped.x1, 0.0F);
  EXPECT_FLOAT_EQ(clipped.y1, 0.0F);
  EXPECT_FLOAT_EQ(clipped.x2, 4.0F);
  EXPECT_FLOAT_EQ(clipped.y2, 8.0F);
}

TEST(PostprocessOrderTest, RunsNmsBeforeCoordinateRestoreAndClip) {
  const RuntimeContract contract = make_contract(2, {"defect"});
  InferenceOutput output = make_output(1, 2);

  // In model-input coordinates these boxes have IoU 0.2, so both survive
  // NMS. After removing the top padding and clipping, both become the same
  // [0,0,10,5] source-space box. This distinguishes the required ordering.
  set_bcn_value(output, 0, 0, 5.0F);
  set_bcn_value(output, 1, 0, 27.0F);
  set_bcn_value(output, 2, 0, 10.0F);
  set_bcn_value(output, 3, 0, 6.0F);
  set_bcn_value(output, 4, 0, 0.9F);

  set_bcn_value(output, 0, 1, 5.0F);
  set_bcn_value(output, 1, 1, 15.0F);
  set_bcn_value(output, 2, 1, 10.0F);
  set_bcn_value(output, 3, 1, 30.0F);
  set_bcn_value(output, 4, 1, 0.8F);

  PreprocessResult preprocess;
  preprocess.original_width = 100;
  preprocess.original_height = 50;
  preprocess.original_channels = 3;
  preprocess.input_width = 100;
  preprocess.input_height = 100;
  preprocess.resized_width = 100;
  preprocess.resized_height = 50;
  preprocess.pad_left = 0;
  preprocess.pad_right = 0;
  preprocess.pad_top = 25;
  preprocess.pad_bottom = 25;
  preprocess.scale = 1.0;

  const std::vector<Detection> detections =
      postprocess_yolov8_raw(output, contract, preprocess);

  ASSERT_EQ(detections.size(), 2U);
  for (const Detection& detection : detections) {
    EXPECT_FLOAT_EQ(detection.bbox_xyxy.x1, 0.0F);
    EXPECT_FLOAT_EQ(detection.bbox_xyxy.y1, 0.0F);
    EXPECT_FLOAT_EQ(detection.bbox_xyxy.x2, 10.0F);
    EXPECT_FLOAT_EQ(detection.bbox_xyxy.y2, 5.0F);
  }
}

TEST(PostprocessEmptyTest, ValidTensorWithNoScoreAboveThresholdIsEmpty) {
  const RuntimeContract contract = make_contract(2, {"defect"});
  InferenceOutput output = make_output(1, 2);
  set_bcn_value(output, 4, 0, 0.25F);
  set_bcn_value(output, 4, 1, 0.1F);

  PreprocessResult preprocess;
  preprocess.original_width = 100;
  preprocess.original_height = 100;
  preprocess.original_channels = 3;
  preprocess.input_width = 100;
  preprocess.input_height = 100;
  preprocess.resized_width = 100;
  preprocess.resized_height = 100;
  preprocess.scale = 1.0;

  EXPECT_TRUE(
      postprocess_yolov8_raw(output, contract, preprocess).empty());
}

TEST(PostprocessContractTest, RejectsPreprocessFromDifferentInputSize) {
  const RuntimeContract contract = make_contract(1, {"defect"});
  InferenceOutput output = make_output(1, 1);
  set_bcn_value(output, 4, 0, 0.9F);
  const PreprocessResult preprocess = make_portrait_preprocess();

  const std::string message = capture_runtime_error(
      [&] { postprocess_yolov8_raw(output, contract, preprocess); });

  expect_actionable_error(message, "object preprocess.input_size",
                          "expected 100x100", "actual 10x6");
}

}  // namespace
}  // namespace yolo_defect_cpp

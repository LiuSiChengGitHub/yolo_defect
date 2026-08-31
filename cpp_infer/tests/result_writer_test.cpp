#include "yolo_defect_cpp/result_writer.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>

namespace yolo_defect_cpp {
namespace {

SingleImageDetectionResult make_valid_result() {
  SingleImageDetectionResult result;
  result.schema_version = 1;
  result.model_id = "test_model";
  result.declared_model_sha256 = std::string(64, 'A');
  result.class_names = {"defect"};
  result.image.source_path = "image.jpg";
  result.image.original_width = 200;
  result.image.original_height = 200;
  result.image.original_channels = 3;
  result.image.input_width = 800;
  result.image.input_height = 800;
  result.actual_provider = "CPUExecutionProvider";
  result.provider_evidence =
      "explicit_cpu_ep_registration_and_session_creation";
  result.score_threshold = 0.25;
  result.nms_threshold = 0.45;
  result.nms_mode = NmsMode::kClassAgnostic;

  Detection detection;
  detection.class_id = 0;
  detection.class_name = "defect";
  detection.confidence = 0.875F;
  detection.bbox_xyxy = {1.0F, 2.0F, 30.0F, 40.0F};
  result.detections = {detection};
  return result;
}

template <typename Callable>
std::string capture_runtime_error(Callable&& callable) {
  try {
    callable();
  } catch (const std::runtime_error& error) {
    return error.what();
  } catch (...) {
    ADD_FAILURE() << "Expected std::runtime_error, but another exception "
                     "type was thrown.";
    return {};
  }
  ADD_FAILURE() << "Expected std::runtime_error, but no exception was thrown.";
  return {};
}

class CommaDecimalPoint : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

class ScopedGlobalLocale {
 public:
  explicit ScopedGlobalLocale(const std::locale& replacement)
      : original_(std::locale()) {
    std::locale::global(replacement);
  }

  ~ScopedGlobalLocale() { std::locale::global(original_); }

  ScopedGlobalLocale(const ScopedGlobalLocale&) = delete;
  ScopedGlobalLocale& operator=(const ScopedGlobalLocale&) = delete;

 private:
  std::locale original_;
};

TEST(ResultWriterJsonTest, SerializesFrozenSchemaWithStableFieldOrder) {
  const SingleImageDetectionResult result = make_valid_result();

  const std::string json = serialize_detection_json(result);

  const std::string expected =
      "{\n"
      "  \"schema_version\": 1,\n"
      "  \"model\": {\n"
      "    \"model_id\": \"test_model\",\n"
      "    \"declared_sha256\": \"" +
      std::string(64, 'A') +
      "\"\n"
      "  },\n"
      "  \"image\": {\n"
      "    \"path\": \"image.jpg\",\n"
      "    \"original_size\": {\"width\": 200, \"height\": 200, "
      "\"channels\": 3},\n"
      "    \"input_size\": {\"width\": 800, \"height\": 800}\n"
      "  },\n"
      "  \"runtime\": {\n"
      "    \"actual_provider\": \"CPUExecutionProvider\",\n"
      "    \"provider_evidence\": "
      "\"explicit_cpu_ep_registration_and_session_creation\",\n"
      "    \"score_threshold\": 0.25,\n"
      "    \"nms_threshold\": 0.45000000000000001,\n"
      "    \"nms_mode\": \"class_agnostic\"\n"
      "  },\n"
      "  \"detections\": [\n"
      "    {\n"
      "      \"class_id\": 0,\n"
      "      \"class_name\": \"defect\",\n"
      "      \"confidence\": 0.875,\n"
      "      \"bbox_xyxy\": [1, 2, 30, 40]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  EXPECT_EQ(json, expected);
}

TEST(ResultWriterJsonTest, EscapesQuotesBackslashesAndControlBytes) {
  SingleImageDetectionResult result = make_valid_result();
  std::string special = "prefix\"\\";
  special.push_back('\b');
  special.push_back('\f');
  special.push_back('\n');
  special.push_back('\r');
  special.push_back('\t');
  special.push_back('\0');
  special.push_back('\x01');
  special += "suffix";
  result.model_id = special;
  result.class_names = {special};
  result.detections[0].class_name = special;

  const std::string json = serialize_detection_json(result);

  const std::string escaped =
      R"(prefix\"\\\b\f\n\r\t\u0000\u0001suffix)";
  EXPECT_NE(json.find("\"model_id\": \"" + escaped + "\""),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("\"class_name\": \"" + escaped + "\""),
            std::string::npos)
      << json;
  EXPECT_EQ(json.find(special), std::string::npos)
      << "Raw control bytes must never appear in JSON string values.";
}

TEST(ResultWriterJsonTest, EmptyDetectionsSerializeAsAnEmptyArray) {
  SingleImageDetectionResult result = make_valid_result();
  result.detections.clear();

  const std::string json = serialize_detection_json(result);

  EXPECT_NE(json.find("  \"detections\": []\n"), std::string::npos)
      << json;
  EXPECT_EQ(json.find("\"class_id\""), std::string::npos) << json;
}

TEST(ResultWriterJsonTest, RejectsNonFiniteDetectionNumbers) {
  SingleImageDetectionResult result = make_valid_result();
  result.detections[0].confidence =
      std::numeric_limits<float>::quiet_NaN();

  const std::string message = capture_runtime_error(
      [&result] { (void)serialize_detection_json(result); });

  EXPECT_NE(message.find("detections[0].confidence"), std::string::npos)
      << message;
  EXPECT_NE(message.find("finite"), std::string::npos) << message;
  EXPECT_NE(message.find("NaN or Infinity"), std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(ResultWriterJsonTest, RejectsNonFiniteRuntimeNumbers) {
  SingleImageDetectionResult result = make_valid_result();
  result.nms_threshold = std::numeric_limits<double>::infinity();

  const std::string message = capture_runtime_error(
      [&result] { (void)serialize_detection_json(result); });

  EXPECT_NE(message.find("runtime.nms_threshold"), std::string::npos)
      << message;
  EXPECT_NE(message.find("finite value in [0,1]"), std::string::npos)
      << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(ResultWriterJsonTest, IgnoresTheProcessDecimalLocale) {
  const std::locale comma_locale(
      std::locale::classic(), new CommaDecimalPoint());
  const ScopedGlobalLocale locale_guard(comma_locale);

  const std::string json = serialize_detection_json(make_valid_result());

  EXPECT_NE(json.find("\"score_threshold\": 0.25"), std::string::npos)
      << json;
  EXPECT_NE(json.find("\"nms_threshold\": 0.45000000000000001"),
            std::string::npos)
      << json;
  EXPECT_EQ(json.find("0,25"), std::string::npos) << json;
}

TEST(ResultWriterFileTest, RejectsRegularFileUsedAsOutputParent) {
  const auto unique_suffix = std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count();
  const std::filesystem::path test_root =
      std::filesystem::temp_directory_path() /
      ("yolo_defect_s1_06_output_" + std::to_string(unique_suffix));
  const std::filesystem::path source_image = test_root / "source_image.jpg";
  const std::filesystem::path blocked_parent = test_root / "blocked_parent";
  ASSERT_TRUE(std::filesystem::create_directories(test_root));
  {
    std::ofstream source(source_image, std::ios::binary);
    ASSERT_TRUE(source.is_open());
    source << "synthetic source placeholder";
  }
  {
    std::ofstream blocker(blocked_parent, std::ios::binary);
    ASSERT_TRUE(blocker.is_open());
    blocker << "this regular file cannot be an output directory";
  }

  SingleImageDetectionResult result = make_valid_result();
  result.image.source_path = source_image;
  DetectionOutputRequest request;
  request.json_path = blocked_parent / "detections.json";

  const std::string message = capture_runtime_error([&] {
    (void)write_detection_outputs(result, request, {});
  });

  EXPECT_NE(message.find("output.json_path"), std::string::npos) << message;
  EXPECT_NE(message.find("expected"), std::string::npos) << message;
  EXPECT_NE(message.find("actual"), std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;

  std::error_code cleanup_error;
  std::filesystem::remove_all(test_root, cleanup_error);
  EXPECT_FALSE(cleanup_error) << cleanup_error.message();
}

TEST(ResultWriterFileTest, RejectsOutputInsideProtectedCacheDirectory) {
  const auto unique_suffix = std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count();
  const std::filesystem::path test_root =
      std::filesystem::temp_directory_path() /
      ("yolo_defect_protected_cache_" + std::to_string(unique_suffix));
  const std::filesystem::path source_image = test_root / "source.jpg";
  const std::filesystem::path cache = test_root / "engine_cache";
  ASSERT_TRUE(std::filesystem::create_directories(cache));
  {
    std::ofstream source(source_image, std::ios::binary);
    ASSERT_TRUE(source.is_open());
    source << "synthetic source placeholder";
  }
  SingleImageDetectionResult result = make_valid_result();
  result.image.source_path = source_image;
  DetectionOutputRequest request;
  request.json_path = cache / "frozen.engine";
  request.overwrite_existing = true;

  const std::string message = capture_runtime_error([&] {
    (void)write_detection_outputs(result, request, {cache});
  });

  EXPECT_NE(message.find("output.json_path"), std::string::npos) << message;
  EXPECT_NE(message.find("protected input"), std::string::npos) << message;
  EXPECT_FALSE(std::filesystem::exists(*request.json_path));
  std::error_code cleanup_error;
  std::filesystem::remove_all(test_root, cleanup_error);
  EXPECT_FALSE(cleanup_error) << cleanup_error.message();
}

}  // namespace
}  // namespace yolo_defect_cpp

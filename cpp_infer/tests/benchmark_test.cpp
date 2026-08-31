#include "yolo_defect_cpp/benchmark_result.h"
#include "yolo_defect_cpp/benchmark_writer.h"

#include <gtest/gtest.h>

#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo_defect_cpp {
namespace {

void expect_benchmark_failure(const std::function<void()>& operation,
                              const std::string& object_text) {
  try {
    operation();
    FAIL() << "Expected benchmark validation to reject " << object_text;
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(object_text), std::string::npos) << message;
    EXPECT_NE(message.find("expected"), std::string::npos) << message;
    EXPECT_NE(message.find("actual"), std::string::npos) << message;
    EXPECT_NE(message.find("action:"), std::string::npos) << message;
  } catch (...) {
    FAIL() << "Expected std::runtime_error for " << object_text;
  }
}

BenchmarkResult make_valid_benchmark_result(bool native) {
  BenchmarkResult result;
  result.evidence_type =
      native ? "cpp_native_tensorrt_single_image_release_benchmark"
             : "cpp_ort_single_image_release_benchmark";
  result.timestamp_utc = "2026-08-31T00:00:00Z";
  result.command_arguments = {"yolo_defect_cpp", "--benchmark"};
  result.repeat = 1;
  result.environment.hostname = "test-host";
  result.environment.processor = "test-cpu";
  result.environment.architecture = "x86_64";
  result.environment.logical_cpu_count = 1;
  result.environment.os_name = "Linux";
  result.environment.os_version = "test-linux";
  result.environment.compiler_id = "GNU";
  result.environment.compiler_version = "13.3.0";
  result.environment.build_type = "Release";
  result.environment.opencv_version = "4.6.0";
  result.environment.onnxruntime_version = "1.20.1";
  result.runtime.requested_provider =
      native ? "tensorrt_native" : "cpu";
  result.runtime.actual_provider =
      native ? "TensorRTNative" : "CPUExecutionProvider";
  result.runtime.provider_evidence = "test execution evidence";
  result.runtime.execution_mode =
      native ? "synchronous_non_default_cuda_stream" : "sequential";
  result.runtime.intra_op_num_threads = native ? 0 : 1;
  result.runtime.inter_op_num_threads = native ? 0 : 1;
  result.runtime.graph_optimization_level =
      native ? "frozen_engine_build_time" : "all";
  result.runtime.session_initialization_ms = 1.0;
  result.model.model_id = "test-model";
  result.model.model_family = "yolov8";
  result.model.model_path = "model.onnx";
  result.model.declared_sha256 = std::string(64, 'A');
  result.model.file_size_bytes = 1;
  result.model.opset = 17;
  result.model.input_name = "images";
  result.model.input_shape = {1, 3, 800, 800};
  result.model.input_dtype = "float32";
  result.model.input_layout = "nchw";
  result.sample.image_path = "crazing_241.jpg";
  result.sample.file_size_bytes = 23845;
  result.sample.width = 200;
  result.sample.height = 200;
  result.sample.channels = 3;
  result.score_threshold = 0.25;
  result.nms_threshold = 0.45;
  result.nms_mode = "class_agnostic";
  const LatencyStatistics latency{1, 2.0, 2.0, 2.0};
  result.latency.image_decode = latency;
  result.latency.preprocess = latency;
  result.latency.session_run = latency;
  result.latency.postprocess = latency;
  result.latency.pipeline = latency;
  result.latency.end_to_end = latency;
  result.latency.pipeline_throughput_images_per_second = 500.0;
  result.latency.end_to_end_throughput_images_per_second = 500.0;
  result.memory.supported = true;
  result.memory.status = "supported";
  result.memory.metric = "peak_rss";
  result.memory.bytes = 1024U * 1024U;
  result.memory.mebibytes = 1.0;
  result.memory.scope = "process lifetime";
  result.timing_exclusions = {"setup"};
  result.limitations = {"bounded test"};
  return result;
}

TEST(BenchmarkStatisticsTest, UsesKnownMeanAndNearestRankPercentiles) {
  const std::vector<double> samples_ms = {
      10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0};

  const LatencyStatistics statistics =
      calculate_latency_statistics(samples_ms);

  EXPECT_EQ(statistics.sample_count, 10U);
  EXPECT_DOUBLE_EQ(statistics.mean_ms, 5.5);
  EXPECT_DOUBLE_EQ(statistics.p50_ms, 5.0);
  EXPECT_DOUBLE_EQ(statistics.p95_ms, 10.0);
}

TEST(BenchmarkStatisticsTest, ASingleSampleIsEveryStatistic) {
  const LatencyStatistics statistics =
      calculate_latency_statistics({12.75});

  EXPECT_EQ(statistics.sample_count, 1U);
  EXPECT_DOUBLE_EQ(statistics.mean_ms, 12.75);
  EXPECT_DOUBLE_EQ(statistics.p50_ms, 12.75);
  EXPECT_DOUBLE_EQ(statistics.p95_ms, 12.75);
}

TEST(BenchmarkStatisticsTest, RejectsAnEmptySampleSet) {
  expect_benchmark_failure(
      [] { calculate_latency_statistics({}); }, "latency_samples");
}

TEST(BenchmarkStatisticsTest, RejectsANegativeDuration) {
  expect_benchmark_failure(
      [] { calculate_latency_statistics({1.0, -0.01, 2.0}); },
      "latency_samples[1]");
}

TEST(BenchmarkStatisticsTest, RejectsANanDuration) {
  expect_benchmark_failure(
      [] {
        calculate_latency_statistics(
            {1.0, std::numeric_limits<double>::quiet_NaN()});
      },
      "latency_samples[1]");
}

TEST(BenchmarkStatisticsTest, RejectsAnInfiniteDuration) {
  expect_benchmark_failure(
      [] {
        calculate_latency_statistics(
            {std::numeric_limits<double>::infinity()});
      },
      "latency_samples[0]");
}

TEST(BenchmarkThroughputTest, DerivesBatchOneThroughputFromMeanLatency) {
  LatencyStatistics latency;
  latency.sample_count = 100;
  latency.mean_ms = 4.0;
  latency.p50_ms = 3.5;
  latency.p95_ms = 6.0;

  EXPECT_DOUBLE_EQ(calculate_throughput_images_per_second(latency), 250.0);
}

TEST(BenchmarkThroughputTest, RejectsNonPositiveAndNonFiniteMeanLatency) {
  const std::vector<double> invalid_means = {
      0.0,
      -1.0,
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
  };

  for (const double invalid_mean : invalid_means) {
    SCOPED_TRACE(::testing::Message() << "mean_ms=" << invalid_mean);
    expect_benchmark_failure(
        [invalid_mean] {
          LatencyStatistics latency;
          latency.sample_count = 1;
          latency.mean_ms = invalid_mean;
          calculate_throughput_images_per_second(latency);
        },
        "latency.mean_ms");
  }
}

TEST(BenchmarkEvidenceTypeTest, AcceptsProviderCoupledOrtAndNativeTypes) {
  EXPECT_NO_THROW(validate_benchmark_result(make_valid_benchmark_result(false)));
  EXPECT_NO_THROW(validate_benchmark_result(make_valid_benchmark_result(true)));
}

TEST(BenchmarkEvidenceTypeTest, RejectsOrtTypeForNativeProvider) {
  BenchmarkResult result = make_valid_benchmark_result(true);
  result.evidence_type = "cpp_ort_single_image_release_benchmark";
  expect_benchmark_failure(
      [&result] { validate_benchmark_result(result); }, "evidence_type");
}

TEST(BenchmarkEvidenceTypeTest, RejectsJsonInsideProtectedCacheDirectory) {
  const auto unique_suffix = std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count();
  const std::filesystem::path test_root =
      std::filesystem::temp_directory_path() /
      ("yolo_defect_benchmark_cache_" + std::to_string(unique_suffix));
  const std::filesystem::path cache = test_root / "engine_cache";
  ASSERT_TRUE(std::filesystem::create_directories(cache));
  const std::filesystem::path output = cache / "frozen.engine";
  BenchmarkResult result = make_valid_benchmark_result(true);

  expect_benchmark_failure(
      [&result, &output, &cache] {
        (void)write_benchmark_json(result, output, true, {cache});
      },
      "benchmark_json.path");
  EXPECT_FALSE(std::filesystem::exists(output));
  std::error_code cleanup_error;
  std::filesystem::remove_all(test_root, cleanup_error);
  EXPECT_FALSE(cleanup_error) << cleanup_error.message();
}

}  // namespace
}  // namespace yolo_defect_cpp

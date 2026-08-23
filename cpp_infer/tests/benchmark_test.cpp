#include "yolo_defect_cpp/benchmark_result.h"

#include <gtest/gtest.h>

#include <cmath>
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

}  // namespace
}  // namespace yolo_defect_cpp

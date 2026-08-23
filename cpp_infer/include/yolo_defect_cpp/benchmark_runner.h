#ifndef YOLO_DEFECT_CPP_BENCHMARK_RUNNER_H_
#define YOLO_DEFECT_CPP_BENCHMARK_RUNNER_H_

#include "yolo_defect_cpp/benchmark_result.h"
#include "yolo_defect_cpp/config_loader.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace yolo_defect_cpp {

struct BenchmarkRequest {
  std::filesystem::path image_path;
  std::size_t warmup = 10;
  std::size_t repeat = 100;
  std::vector<std::string> command_arguments;
};

class BenchmarkRunner {
 public:
  explicit BenchmarkRunner(RuntimeContract contract);
  ~BenchmarkRunner();

  BenchmarkRunner(const BenchmarkRunner&) = delete;
  BenchmarkRunner& operator=(const BenchmarkRunner&) = delete;
  BenchmarkRunner(BenchmarkRunner&&) noexcept;
  BenchmarkRunner& operator=(BenchmarkRunner&&) noexcept;

  BenchmarkResult run(const BenchmarkRequest& request);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BENCHMARK_RUNNER_H_
